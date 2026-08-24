"""Stable organizer facade for the accepted fixed path and research modes.

The fixed organizer deliberately continues to own its checkpoint-visible
parameters directly.  Exchangeable/adaptive implementation lives behind the
``exchangeable`` attribute exactly as it did before the Stage-7 cleanup.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import UnifiedForwardConfig
from .organization.exchangeable import ExchangeableSlotOrganizer
from .organization.helpers import (
    EPS,
    _as_batched_coords,
    _assignment_purity,
    _descriptor_first_features,
    _masked_softmax,
    _mechanism_descriptors,
    _relative_delta,
    _scheduled_stabilized_assignment,
    _stabilize_all_edge_softmax_assignment,
    _weighted_coords,
    _weighted_scale,
    deterministic_slot_codes,
)
from .routing import normalize_assignment


class HypergraphOrganizerCore(nn.Module):
    """Organize module and environment tokens into ``K`` latent hyperedges."""

    def __init__(self, config: UnifiedForwardConfig):
        """Initialize HypergraphOrganizerCore and its required state."""

        super().__init__()
        self.config = config
        if config.organizer_mode == "exchangeable_slots":
            self.exchangeable = ExchangeableSlotOrganizer(config)
            return
        hidden_dim = int(config.hidden_dim)
        num_hyperedges = int(config.num_hyperedges)
        self.module_score = nn.Linear(hidden_dim, num_hyperedges)
        self.env_score = nn.Linear(hidden_dim, num_hyperedges)
        self.module_to_hyper = nn.Linear(hidden_dim, hidden_dim)
        self.env_to_hyper = nn.Linear(hidden_dim, hidden_dim)
        self.hyper_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.me_query = nn.Linear(hidden_dim, hidden_dim)
        self.me_key = nn.Linear(hidden_dim, hidden_dim)
        self.me_context_proj = nn.Linear(hidden_dim, hidden_dim)

    def set_edge_capacity(self, capacity: int) -> None:
        """Set runtime capacity for the exchangeable organizer."""

        if self.config.organizer_mode == "exchangeable_slots":
            self.exchangeable.set_edge_capacity(capacity)

    def set_training_progress(self, *, epoch: int, total_epochs: Optional[int] = None) -> None:
        """Set selection warmup progress; fixed projection ignores it."""

        if self.config.organizer_mode == "exchangeable_slots":
            self.exchangeable.set_training_progress(epoch=epoch, total_epochs=total_epochs)

    def selection_state(self) -> Dict[str, Optional[int]]:
        """Return explicit organizer selection progress for checkpoint metadata."""

        if self.config.organizer_mode == "exchangeable_slots":
            return self.exchangeable.selection_state()
        return {"epoch": None, "total_epochs": None}

    def forward(
        self,
        module_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        module_centers: torch.Tensor,
        env_coords: torch.Tensor,
        module_present: torch.Tensor,
        geometry_mode: Optional[str] = None,
        candidate_codes: Optional[torch.Tensor] = None,
        edge_capacity: Optional[int] = None,
        selection_override: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build incidences, hyperedge states, geometry, and diagnostics.

        ``module_tokens [B,M,H]`` and ``env_tokens [B,E,H]`` are assigned to
        ``K`` hyperedges. ``A_me [B,M,E]`` optionally supplies module-to-
        environment context; ``A_mh [B,M,K]`` and ``A_eh [B,E,K]`` aggregate
        both node types. The output ``hyper_state [B,K,H]`` is accompanied by
        source/region centroids ``[B,K,2]`` and mechanism descriptors. Inactive
        module rows receive zero assignment mass.
        """

        cfg = self.config
        if geometry_mode is not None and geometry_mode != cfg.geometry_mode:
            cfg = UnifiedForwardConfig.from_dict({**cfg.to_dict(), "geometry_mode": geometry_mode})
        if cfg.organizer_mode == "exchangeable_slots":
            return self.exchangeable(
                module_tokens=module_tokens,
                env_tokens=env_tokens,
                module_centers=module_centers,
                env_coords=env_coords,
                module_present=module_present,
                cfg=cfg,
                candidate_codes=candidate_codes,
                edge_capacity=edge_capacity,
                selection_override=selection_override,
            )

        batch_size, _, hidden_dim = module_tokens.shape
        env_coords_b = _as_batched_coords(env_coords.to(module_tokens.device, module_tokens.dtype), batch_size)
        module_present = module_present.to(device=module_tokens.device, dtype=module_tokens.dtype)

        if cfg.use_A_me_auxiliary:
            q = self.me_query(module_tokens)
            k = self.me_key(env_tokens)
            logits = torch.einsum("bmh,beh->bme", q, k) / math.sqrt(float(hidden_dim))
            A_me = torch.softmax(logits, dim=-1) * module_present.unsqueeze(-1)
            module_env_context = torch.einsum("bme,beh->bmh", A_me, env_tokens)
            module_tokens_for_hyper = module_tokens + 0.25 * self.me_context_proj(module_env_context)
            module_tokens_for_hyper = module_tokens_for_hyper * module_present.unsqueeze(-1)
        else:
            A_me = torch.zeros(
                module_tokens.shape[0],
                module_tokens.shape[1],
                env_tokens.shape[1],
                device=module_tokens.device,
                dtype=module_tokens.dtype,
            )
            module_env_context = torch.zeros_like(module_tokens)
            module_tokens_for_hyper = module_tokens

        module_logits = self.module_score(module_tokens_for_hyper)
        if cfg.hyper_module_assignment_mode == "uniform":
            A_mh = module_present.unsqueeze(-1).expand_as(module_logits) / float(max(module_logits.shape[-1], 1))
        else:
            module_mask = module_present.unsqueeze(-1).expand_as(module_logits)
            if cfg.module_assignment_normalizer == "softmax":
                A_mh = _masked_softmax(module_logits, module_mask, dim=-1)
            else:
                A_mh = normalize_assignment(
                    module_logits,
                    mode=cfg.module_assignment_normalizer,
                    mask=module_mask > 0,
                )
            A_mh = A_mh * module_present.unsqueeze(-1)

        module_mass_raw = A_mh.sum(dim=1)
        hyper_module_mass = module_mass_raw / module_mass_raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        source_weights = A_mh / A_mh.sum(dim=1, keepdim=True).clamp_min(EPS)
        hyper_source_coords = _weighted_coords(module_centers, source_weights, cfg)
        hyper_source_variance, hyper_source_scale = _weighted_scale(
            module_centers,
            source_weights,
            hyper_source_coords,
            cfg,
        )

        env_logits = self.env_score(env_tokens)
        delta = _relative_delta(hyper_source_coords[:, None, :, :], env_coords_b[:, :, None, :], cfg)
        dist = torch.sqrt(delta.square().sum(dim=-1) + EPS)
        scale_x, scale_y = cfg.spatial_scale()
        scale = 0.25 * math.sqrt(scale_x**2 + scale_y**2)
        geometry_bias = -dist / max(scale, EPS)
        if cfg.environment_assignment_normalizer == "softmax":
            A_eh = torch.softmax(env_logits + geometry_bias, dim=-1)
        else:
            A_eh = normalize_assignment(
                env_logits + geometry_bias,
                mode=cfg.environment_assignment_normalizer,
            )

        env_mass_raw = A_eh.sum(dim=1)
        hyper_env_mass = env_mass_raw / env_mass_raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        region_weights = A_eh / A_eh.sum(dim=1, keepdim=True).clamp_min(EPS)
        hyper_region_coords = _weighted_coords(env_coords_b, region_weights, cfg)
        hyper_region_variance, hyper_region_scale = _weighted_scale(
            env_coords_b,
            region_weights,
            hyper_region_coords,
            cfg,
        )
        hyper_strength = torch.sqrt(hyper_module_mass * hyper_env_mass + EPS)
        hyper_module_purity = _assignment_purity(A_mh)
        hyper_env_purity = _assignment_purity(A_eh)
        edge_active_mask = torch.ones_like(hyper_strength)
        mechanism_descriptor_features = _descriptor_first_features(
            hyper_source_coords,
            hyper_source_scale,
            hyper_region_coords,
            hyper_region_scale,
            hyper_module_mass,
            hyper_env_mass,
            hyper_module_purity,
            hyper_env_purity,
            edge_active_mask,
            cfg,
        )
        (
            mechanism_geometry_features,
            mechanism_mass_features,
            mechanism_raw_features,
            hyper_source_region_distance,
            hyper_source_region_downstream,
            hyper_source_region_lateral,
        ) = _mechanism_descriptors(
            hyper_source_coords,
            hyper_region_coords,
            hyper_module_mass,
            hyper_env_mass,
            hyper_strength,
            module_mass_raw,
            env_mass_raw,
            module_present,
            env_tokens.shape[1],
            cfg,
        )

        module_summary = torch.einsum("bmk,bmh->bkh", A_mh, self.module_to_hyper(module_tokens_for_hyper))
        module_summary = module_summary / module_mass_raw.unsqueeze(-1).clamp_min(EPS)
        env_summary = torch.einsum("bek,beh->bkh", A_eh, self.env_to_hyper(env_tokens))
        env_summary = env_summary / env_mass_raw.unsqueeze(-1).clamp_min(EPS)
        hyper_state = self.hyper_mix(module_summary + env_summary)

        output: Dict[str, torch.Tensor] = {
            "A_mh": A_mh,
            "A_eh": A_eh,
            "hyper_state": hyper_state,
            "hyper_source_coords": hyper_source_coords,
            "hyper_region_coords": hyper_region_coords,
            "hyper_source_variance": hyper_source_variance,
            "hyper_source_scale": hyper_source_scale,
            "hyper_region_variance": hyper_region_variance,
            "hyper_region_scale": hyper_region_scale,
            "hyper_module_mass_raw": module_mass_raw,
            "hyper_env_mass_raw": env_mass_raw,
            "hyper_module_mass": hyper_module_mass,
            "hyper_env_mass": hyper_env_mass,
            "hyper_module_purity": hyper_module_purity,
            "hyper_env_purity": hyper_env_purity,
            "candidate_module_mass_fraction": hyper_module_mass,
            "candidate_environment_mass_fraction": hyper_env_mass,
            "candidate_module_purity": hyper_module_purity,
            "candidate_environment_purity": hyper_env_purity,
            "candidate_source_coords": hyper_source_coords,
            "candidate_source_scale": hyper_source_scale,
            "candidate_region_coords": hyper_region_coords,
            "candidate_region_scale": hyper_region_scale,
            "hyper_strength": hyper_strength,
            "edge_quality": torch.sqrt(hyper_module_purity * hyper_env_purity),
            "edge_active_mask": edge_active_mask,
            "hard_selected_edge_mask": edge_active_mask,
            "edge_transition_gate": edge_active_mask,
            "edge_viable_mask": edge_active_mask,
            "effective_edge_mask": edge_active_mask,
            "candidate_edge_count": edge_active_mask.new_full(
                (batch_size,), float(edge_active_mask.shape[-1])
            ),
            "selected_edge_count": edge_active_mask.sum(dim=-1),
            "viable_selected_edge_count": edge_active_mask.sum(dim=-1),
            "hard_selected_edge_count": edge_active_mask.sum(dim=-1),
            "edge_transition_gate_sum": edge_active_mask.sum(dim=-1),
            "empty_selected_edge_count": edge_active_mask.new_zeros(edge_active_mask.shape[0]),
            "pre_fallback_zero_support_module_rows": edge_active_mask.new_zeros(edge_active_mask.shape[0]),
            "post_fallback_zero_support_module_rows": edge_active_mask.new_zeros(edge_active_mask.shape[0]),
            "pre_fallback_zero_support_environment_rows": edge_active_mask.new_zeros(edge_active_mask.shape[0]),
            "post_fallback_zero_support_environment_rows": edge_active_mask.new_zeros(edge_active_mask.shape[0]),
            "active_edge_count": edge_active_mask.sum(dim=-1),
            "mechanism_geometry_features": mechanism_geometry_features,
            "mechanism_mass_features": mechanism_mass_features,
            "mechanism_raw_features": mechanism_raw_features,
            "mechanism_descriptor_features": mechanism_descriptor_features,
            "hyper_source_region_distance": hyper_source_region_distance,
            "hyper_source_region_downstream": hyper_source_region_downstream,
            "hyper_source_region_lateral": hyper_source_region_lateral,
            "module_tokens": module_tokens,
            "module_tokens_for_hyper": module_tokens_for_hyper,
            "env_tokens": env_tokens,
            "env_coords": env_coords_b,
            "module_centers": module_centers,
            "module_present": module_present,
            "A_me": A_me,
            "module_env_context": module_env_context,
            "selection_transition_fraction": A_mh.new_zeros(()),
            "module_sparsity_fraction": A_mh.new_zeros(()),
            "environment_sparsity_fraction": A_mh.new_zeros(()),
            "query_sparsity_fraction": A_mh.new_zeros(()),
            "training_progress_epoch": A_mh.new_zeros(()),
            "routing_execution_gathered": A_mh.new_tensor(float(cfg.routing_execution == "gathered")),
            "hyper_module_assignment_uniform": A_mh.new_tensor(float(cfg.hyper_module_assignment_mode == "uniform")),
        }

        return output
