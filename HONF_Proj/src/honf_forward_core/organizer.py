"""CORE HONF hypergraph organizer.

Inputs are encoded generic module tokens, environment tokens, module centers,
environment coordinates, and a module-present mask. Outputs are A_me, A_mh,
A_eh, hyperedge states, source/region coordinates, mechanism descriptors, and
diagnostics used by the field decoder and visualizers. This module is reusable
across domains and contains no ChannelThermal-specific wall or inlet logic.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import UnifiedForwardConfig
from .routing import normalize_assignment


EPS = 1e-6


def _masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """Normalize logits along ``dim`` while assigning masked entries zero mass."""

    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    masked = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
    out = torch.softmax(masked, dim=dim) * mask
    return out / out.sum(dim=dim, keepdim=True).clamp_min(EPS)


def _as_batched_coords(coords: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast shared ``[N,2]`` coordinates to ``[B,N,2]`` when needed."""

    if coords.ndim == 2:
        return coords.unsqueeze(0).expand(batch_size, -1, -1)
    return coords


def _relative_delta(src: torch.Tensor, dst: torch.Tensor, cfg: UnifiedForwardConfig) -> torch.Tensor:
    """Compute ``dst-src``, using minimum-image offsets for periodic domains."""

    delta = dst - src
    periodic_axes = cfg.periodic_dimensions()
    if periodic_axes:
        scale_x, scale_y = cfg.spatial_scale()
        lengths = torch.tensor(
            [max(scale_x, EPS), max(scale_y, EPS)],
            device=delta.device,
            dtype=delta.dtype,
        )
        wrapped = torch.remainder(delta + 0.5 * lengths, lengths) - 0.5 * lengths
        mask = torch.tensor(
            [axis in periodic_axes for axis in range(2)],
            device=delta.device,
            dtype=torch.bool,
        )
        delta = torch.where(mask, wrapped, delta)
    return delta


def _weighted_coords(coords: torch.Tensor, weights: torch.Tensor, cfg: UnifiedForwardConfig) -> torch.Tensor:
    """Reduce node coordinates ``[B,N,2]`` into ``K`` weighted centroids."""

    denom = weights.sum(dim=1).clamp_min(EPS).unsqueeze(-1)
    periodic_axes = cfg.periodic_dimensions()
    if not periodic_axes:
        return torch.einsum("bnk,bnd->bkd", weights, coords) / denom

    scale_x, scale_y = cfg.spatial_scale()
    lengths = torch.tensor(
        [max(scale_x, EPS), max(scale_y, EPS)],
        device=coords.device,
        dtype=coords.dtype,
    )
    if periodic_axes != (0, 1):
        normalized_weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(EPS)
        outputs = []
        for axis in range(2):
            if axis in periodic_axes:
                angles = 2.0 * math.pi * coords[..., axis] / lengths[axis]
                sin_sum = torch.einsum("bnk,bn->bk", normalized_weights, torch.sin(angles))
                cos_sum = torch.einsum("bnk,bn->bk", normalized_weights, torch.cos(angles))
                mean = torch.remainder(torch.atan2(sin_sum, cos_sum) / (2.0 * math.pi) * lengths[axis], lengths[axis])
            else:
                mean = torch.einsum("bnk,bn->bk", weights, coords[..., axis]) / denom.squeeze(-1)
            outputs.append(mean)
        return torch.stack(outputs, dim=-1)
    angles = 2.0 * math.pi * coords[:, None, :, :] / lengths
    weight_t = (weights / weights.sum(dim=1, keepdim=True).clamp_min(EPS)).transpose(1, 2).unsqueeze(-1)
    sin_sum = (weight_t * torch.sin(angles)).sum(dim=2)
    cos_sum = (weight_t * torch.cos(angles)).sum(dim=2)
    mean_angle = torch.atan2(sin_sum, cos_sum)
    return torch.remainder(mean_angle / (2.0 * math.pi) * lengths, lengths)


def _weighted_scale(
    coords: torch.Tensor,
    weights: torch.Tensor,
    centroids: torch.Tensor,
    cfg: UnifiedForwardConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return diagonal variance and scale for weighted node coordinates."""

    delta = _relative_delta(centroids[:, None, :, :], coords[:, :, None, :], cfg)
    variance = torch.einsum("bnk,bnkd->bkd", weights, delta.square())
    return variance, torch.sqrt(variance.clamp_min(EPS))


def _assignment_purity(assignment: torch.Tensor) -> torch.Tensor:
    """Measure the winner-owned fraction of every assignment column."""

    winners = assignment.argmax(dim=-1)
    winner_mask = torch.nn.functional.one_hot(winners, num_classes=assignment.shape[-1]).to(assignment.dtype)
    numerator = (assignment * winner_mask).sum(dim=1)
    return numerator / assignment.sum(dim=1).clamp_min(EPS)


def _descriptor_first_features(
    hyper_source_coords: torch.Tensor,
    hyper_source_scale: torch.Tensor,
    hyper_region_coords: torch.Tensor,
    hyper_region_scale: torch.Tensor,
    hyper_module_mass: torch.Tensor,
    hyper_env_mass: torch.Tensor,
    hyper_module_purity: torch.Tensor,
    hyper_env_purity: torch.Tensor,
    edge_active_mask: torch.Tensor,
    cfg: UnifiedForwardConfig,
) -> torch.Tensor:
    """Build normalized mechanism descriptors with geometry as primary state."""

    scale_x, scale_y = cfg.spatial_scale()
    scales = hyper_source_coords.new_tensor([max(scale_x, EPS), max(scale_y, EPS)])
    displacement = _relative_delta(hyper_source_coords, hyper_region_coords, cfg)
    distance_scale = max(math.sqrt(scale_x**2 + scale_y**2), EPS)
    distance = torch.sqrt(displacement.square().sum(dim=-1, keepdim=True) + EPS) / distance_scale
    return torch.cat(
        [
            hyper_source_coords / scales,
            hyper_source_scale / scales,
            hyper_region_coords / scales,
            hyper_region_scale / scales,
            displacement / scales,
            distance,
            hyper_module_mass.unsqueeze(-1),
            hyper_env_mass.unsqueeze(-1),
            hyper_module_purity.unsqueeze(-1),
            hyper_env_purity.unsqueeze(-1),
            edge_active_mask.unsqueeze(-1),
        ],
        dim=-1,
    )


def _mechanism_descriptors(
    hyper_source_coords: torch.Tensor,
    hyper_region_coords: torch.Tensor,
    hyper_module_mass: torch.Tensor,
    hyper_env_mass: torch.Tensor,
    hyper_strength: torch.Tensor,
    module_mass_raw: torch.Tensor,
    env_mass_raw: torch.Tensor,
    module_present: torch.Tensor,
    env_count: int,
    cfg: UnifiedForwardConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generic source-region descriptors for hyperedge field mechanisms.

    These features describe where a hyperedge draws from, where its field
    context is concentrated, and how much module/environment mass it carries.
    They deliberately avoid case-specific wall, plume, or thermal rules.
    """

    scale_x, scale_y = cfg.spatial_scale()
    lx = max(scale_x, EPS)
    ly = max(scale_y, EPS)
    diag = max(math.sqrt(lx * lx + ly * ly), EPS)
    displacement = _relative_delta(hyper_source_coords, hyper_region_coords, cfg)
    dx = displacement[..., 0:1]
    dy = displacement[..., 1:2]
    if 0 in cfg.periodic_dimensions():
        downstream = torch.remainder(hyper_region_coords[..., 0:1] - hyper_source_coords[..., 0:1], lx) / lx
        upstream = torch.remainder(hyper_source_coords[..., 0:1] - hyper_region_coords[..., 0:1], lx) / lx
    else:
        downstream = torch.relu(dx) / lx
        upstream = torch.relu(-dx) / lx
    lateral = dy.abs() / ly
    distance = torch.sqrt(dx.square() + dy.square() + EPS) / diag
    mechanism_geometry_features = torch.cat(
        [
            hyper_source_coords[..., 0:1] / lx,
            hyper_source_coords[..., 1:2] / ly,
            hyper_region_coords[..., 0:1] / lx,
            hyper_region_coords[..., 1:2] / ly,
            dx / lx,
            dy / ly,
            distance,
            downstream,
            upstream,
            lateral,
        ],
        dim=-1,
    )

    module_count = module_present.sum(dim=-1, keepdim=True).clamp_min(1.0)
    env_count_t = module_present.new_tensor(float(max(env_count, 1)))
    module_raw_norm = module_mass_raw / module_count
    env_raw_norm = env_mass_raw / env_count_t
    module_raw_log = torch.log1p(module_mass_raw) / torch.log1p(module_count)
    env_raw_log = torch.log1p(env_mass_raw) / torch.log1p(env_count_t)
    mechanism_mass_features = torch.stack(
        [
            hyper_module_mass,
            hyper_env_mass,
            hyper_strength,
            module_raw_norm,
            env_raw_norm,
            module_raw_log,
            env_raw_log,
        ],
        dim=-1,
    )
    mechanism_raw_features = torch.cat([mechanism_geometry_features, mechanism_mass_features], dim=-1)
    return mechanism_geometry_features, mechanism_mass_features, mechanism_raw_features, distance, downstream, lateral


def deterministic_slot_codes(
    capacity: int,
    hidden_dim: int,
    *,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate zero-mean, non-trainable candidate codes as ``[K,H]``."""

    count = int(capacity)
    width = int(hidden_dim)
    if count <= 0 or width <= 0:
        raise ValueError("Slot-code capacity and hidden dimension must be positive.")
    positions = (torch.arange(count, device=device, dtype=dtype) + 0.5) / float(count)
    dimensions = torch.arange(width, device=device)
    if mode == "sinusoidal":
        frequencies = torch.div(dimensions, 2, rounding_mode="floor").to(dtype=dtype) + 1.0
        angles = 2.0 * math.pi * positions[:, None] * frequencies[None, :]
        codes = torch.where((dimensions % 2)[None, :] == 0, torch.sin(angles), torch.cos(angles))
    elif mode == "low_discrepancy":
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        codes = torch.frac(
            (torch.arange(count, device=device, dtype=dtype)[:, None] + 1.0)
            * (dimensions.to(dtype=dtype)[None, :] + 1.0)
            * golden
        )
        codes = 2.0 * codes - 1.0
    else:
        raise ValueError(f"Unsupported slot code mode: {mode!r}.")
    codes = codes - codes.mean(dim=0, keepdim=True)
    return codes / torch.sqrt(codes.square().mean().clamp_min(EPS))


class ExchangeableSlotOrganizer(nn.Module):
    """Discover anonymous candidate edges with shared iterative refinement."""

    def __init__(self, config: UnifiedForwardConfig):
        """Initialize capacity-independent shared organizer parameters."""

        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.module_query = nn.Linear(hidden_dim, hidden_dim)
        self.module_key = nn.Linear(hidden_dim, hidden_dim)
        self.module_value = nn.Linear(hidden_dim, hidden_dim)
        self.env_query = nn.Linear(hidden_dim, hidden_dim)
        self.env_key = nn.Linear(hidden_dim, hidden_dim)
        self.env_value = nn.Linear(hidden_dim, hidden_dim)
        self.slot_base = nn.Linear(hidden_dim, hidden_dim)
        self.slot_scale = nn.Linear(hidden_dim, hidden_dim)
        self.slot_update = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.slot_norm = nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity()
        self.me_query = nn.Linear(hidden_dim, hidden_dim)
        self.me_key = nn.Linear(hidden_dim, hidden_dim)
        self.me_context_proj = nn.Linear(hidden_dim, hidden_dim)
        self._runtime_edge_capacity = int(config.edge_capacity)
        self._selection_epoch = 0
        self._selection_total_epochs: Optional[int] = None

    def set_edge_capacity(self, capacity: int) -> None:
        """Set the runtime candidate budget without changing parameters."""

        if int(capacity) <= 0:
            raise ValueError("Runtime edge capacity must be positive.")
        if int(capacity) < int(self.config.minimum_active_edges):
            raise ValueError("Runtime edge capacity cannot be smaller than minimum_active_edges.")
        self._runtime_edge_capacity = int(capacity)

    def set_training_progress(self, *, epoch: int, total_epochs: Optional[int] = None) -> None:
        """Update detached selection warmup state once per training epoch."""

        if int(epoch) < 0:
            raise ValueError("Training epoch must be nonnegative.")
        if total_epochs is not None and int(total_epochs) <= 0:
            raise ValueError("total_epochs must be positive when provided.")
        self._selection_epoch = int(epoch)
        self._selection_total_epochs = None if total_epochs is None else int(total_epochs)

    def forward(
        self,
        module_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        module_centers: torch.Tensor,
        env_coords: torch.Tensor,
        module_present: torch.Tensor,
        *,
        cfg: UnifiedForwardConfig,
        candidate_codes: Optional[torch.Tensor] = None,
        edge_capacity: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Refine candidate slots, select active edges, and export incidences."""

        batch_size, _, hidden_dim = module_tokens.shape
        capacity = self._runtime_edge_capacity if edge_capacity is None else int(edge_capacity)
        if capacity <= 0:
            raise ValueError("Exchangeable organizer requires a positive runtime edge capacity.")
        if capacity < int(cfg.minimum_active_edges):
            raise ValueError("Runtime edge capacity cannot be smaller than minimum_active_edges.")
        module_present = module_present.to(device=module_tokens.device, dtype=module_tokens.dtype)
        env_coords_b = _as_batched_coords(env_coords.to(module_tokens.device, module_tokens.dtype), batch_size)
        module_tokens_for_hyper, A_me, module_env_context = self._module_environment_context(
            module_tokens,
            env_tokens,
            module_present,
            cfg,
        )
        slots, codes = self._initialize_slots(
            module_tokens_for_hyper,
            env_tokens,
            module_present,
            capacity,
            candidate_codes,
        )
        previous_region_coords: Optional[torch.Tensor] = None
        previous_region_scale: Optional[torch.Tensor] = None
        for _ in range(int(cfg.slot_refinement_steps)):
            candidate_A_mh, candidate_A_eh, previous_region_coords, previous_region_scale = self._candidate_assignments(
                module_tokens_for_hyper,
                env_tokens,
                slots,
                module_centers,
                env_coords_b,
                module_present,
                cfg,
                previous_region_coords=previous_region_coords,
                previous_region_scale=previous_region_scale,
            )
            module_weights = candidate_A_mh / candidate_A_mh.sum(dim=1, keepdim=True).clamp_min(EPS)
            env_weights = candidate_A_eh / candidate_A_eh.sum(dim=1, keepdim=True).clamp_min(EPS)
            module_update = torch.einsum(
                "bmk,bmh->bkh",
                module_weights,
                self.module_value(module_tokens_for_hyper),
            )
            env_update = torch.einsum("bek,beh->bkh", env_weights, self.env_value(env_tokens))
            slots = self.slot_update(
                torch.cat([module_update, env_update], dim=-1).reshape(batch_size * capacity, 2 * hidden_dim),
                slots.reshape(batch_size * capacity, hidden_dim),
            ).reshape(batch_size, capacity, hidden_dim)
            slots = self.slot_norm(slots)
        candidate_A_mh, candidate_A_eh, _, _ = self._candidate_assignments(
            module_tokens_for_hyper,
            env_tokens,
            slots,
            module_centers,
            env_coords_b,
            module_present,
            cfg,
            previous_region_coords=previous_region_coords,
            previous_region_scale=previous_region_scale,
        )
        module_purity = _assignment_purity(candidate_A_mh)
        env_purity = _assignment_purity(candidate_A_eh)
        edge_quality = torch.sqrt(module_purity * env_purity)
        edge_active_mask = self._select_active_edges(
            candidate_A_mh,
            candidate_A_eh,
            edge_quality,
            codes,
            module_present,
            cfg,
        )
        A_mh = self._mask_and_renormalize(candidate_A_mh, edge_active_mask, module_present)
        A_eh = self._mask_and_renormalize(candidate_A_eh, edge_active_mask, None)
        return self._assemble_output(
            module_tokens=module_tokens,
            module_tokens_for_hyper=module_tokens_for_hyper,
            env_tokens=env_tokens,
            module_centers=module_centers,
            env_coords=env_coords_b,
            module_present=module_present,
            A_me=A_me,
            module_env_context=module_env_context,
            candidate_A_mh=candidate_A_mh,
            candidate_A_eh=candidate_A_eh,
            A_mh=A_mh,
            A_eh=A_eh,
            candidate_state=slots,
            candidate_codes=codes,
            edge_quality=edge_quality,
            edge_active_mask=edge_active_mask,
            cfg=cfg,
        )

    def _module_environment_context(
        self,
        module_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        module_present: torch.Tensor,
        cfg: UnifiedForwardConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retain the optional module/environment coupling with shared maps."""

        if cfg.use_A_me_auxiliary:
            logits = torch.einsum(
                "bmh,beh->bme",
                self.me_query(module_tokens),
                self.me_key(env_tokens),
            ) / math.sqrt(float(module_tokens.shape[-1]))
            A_me = torch.softmax(logits, dim=-1) * module_present.unsqueeze(-1)
            module_env_context = torch.einsum("bme,beh->bmh", A_me, env_tokens)
            enriched = module_tokens + 0.25 * self.me_context_proj(module_env_context)
            return enriched * module_present.unsqueeze(-1), A_me, module_env_context
        A_me = module_tokens.new_zeros(module_tokens.shape[0], module_tokens.shape[1], env_tokens.shape[1])
        return module_tokens, A_me, torch.zeros_like(module_tokens)

    def _initialize_slots(
        self,
        module_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        module_present: torch.Tensor,
        capacity: int,
        candidate_codes: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create anonymous case-conditioned candidates from deterministic codes."""

        module_pool = (module_tokens * module_present.unsqueeze(-1)).sum(dim=1)
        module_pool = module_pool / module_present.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = module_pool + env_tokens.mean(dim=1)
        if candidate_codes is None:
            codes = deterministic_slot_codes(
                capacity,
                module_tokens.shape[-1],
                mode=self.config.slot_code_mode,
                device=module_tokens.device,
                dtype=module_tokens.dtype,
            )
        else:
            codes = candidate_codes.to(device=module_tokens.device, dtype=module_tokens.dtype)
            if codes.ndim not in {2, 3} or codes.shape[-2:] != (capacity, module_tokens.shape[-1]):
                raise ValueError("candidate_codes must have shape [K,H] or [B,K,H].")
            if codes.ndim == 3 and codes.shape[0] != module_tokens.shape[0]:
                raise ValueError("Batched candidate_codes must match the organizer batch size.")
        if codes.ndim == 2:
            codes_b = codes.unsqueeze(0).expand(module_tokens.shape[0], -1, -1)
        else:
            codes_b = codes
        base = self.slot_base(pooled).unsqueeze(1)
        scale = torch.nn.functional.softplus(self.slot_scale(pooled)).unsqueeze(1) + EPS
        return base + scale * codes_b, codes_b

    def _candidate_assignments(
        self,
        module_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        slots: torch.Tensor,
        module_centers: torch.Tensor,
        env_coords: torch.Tensor,
        module_present: torch.Tensor,
        cfg: UnifiedForwardConfig,
        *,
        previous_region_coords: Optional[torch.Tensor],
        previous_region_scale: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute competitive assignments and the current localized regions."""

        scale = math.sqrt(float(slots.shape[-1]))
        module_logits = torch.einsum(
            "bmh,bkh->bmk",
            self.module_query(module_tokens),
            self.module_key(slots),
        ) / scale
        if cfg.hyper_module_assignment_mode == "uniform":
            candidate_A_mh = module_present.unsqueeze(-1).expand_as(module_logits) / float(slots.shape[1])
        else:
            candidate_A_mh = normalize_assignment(
                module_logits,
                mode=cfg.module_assignment_normalizer,
                mask=module_present.unsqueeze(-1) > 0,
            )
        module_weights = candidate_A_mh / candidate_A_mh.sum(dim=1, keepdim=True).clamp_min(EPS)
        source_coords = _weighted_coords(module_centers, module_weights, cfg)
        env_logits = torch.einsum(
            "beh,bkh->bek",
            self.env_query(env_tokens),
            self.env_key(slots),
        ) / scale
        if cfg.environment_locality_mode == "compact_kernel":
            env_logits = env_logits + self._environment_locality_bias(
                env_coords,
                source_coords if previous_region_coords is None else previous_region_coords,
                previous_region_scale,
                cfg,
            )
        candidate_A_eh = normalize_assignment(
            env_logits,
            mode=cfg.environment_assignment_normalizer,
        )
        env_weights = candidate_A_eh / candidate_A_eh.sum(dim=1, keepdim=True).clamp_min(EPS)
        region_coords = _weighted_coords(env_coords, env_weights, cfg)
        _, region_scale = _weighted_scale(env_coords, env_weights, region_coords, cfg)
        return candidate_A_mh, candidate_A_eh, region_coords, region_scale

    @staticmethod
    def _environment_locality_bias(
        env_coords: torch.Tensor,
        region_coords: torch.Tensor,
        region_scale: Optional[torch.Tensor],
        cfg: UnifiedForwardConfig,
    ) -> torch.Tensor:
        """Return a broad-first compact-support bias for environment tokens."""

        scale_x, scale_y = cfg.spatial_scale()
        if region_scale is None:
            anisotropic_scale = env_coords.new_tensor([max(scale_x, EPS), max(scale_y, EPS)])
            anisotropic_scale = anisotropic_scale.view(1, 1, 1, 2)
        else:
            minimum = env_coords.new_tensor(
                [
                    max(scale_x * float(cfg.minimum_region_scale), EPS),
                    max(scale_y * float(cfg.minimum_region_scale), EPS),
                ]
            )
            anisotropic_scale = torch.maximum(region_scale, minimum).unsqueeze(1)
        delta = _relative_delta(region_coords[:, None, :, :], env_coords[:, :, None, :], cfg)
        radius_square = (delta / anisotropic_scale).square().sum(dim=-1)
        compactness = torch.relu(1.0 - radius_square).square()
        return float(cfg.environment_locality_strength) * torch.log(compactness + EPS)

    def _select_active_edges(
        self,
        candidate_A_mh: torch.Tensor,
        candidate_A_eh: torch.Tensor,
        edge_quality: torch.Tensor,
        candidate_codes: torch.Tensor,
        module_present: torch.Tensor,
        cfg: UnifiedForwardConfig,
    ) -> torch.Tensor:
        """Select the smallest quality/novelty set that reaches coverage."""

        if cfg.edge_selection_mode == "all":
            return torch.ones_like(edge_quality).detach()
        warmup = self.training and self._selection_epoch < int(cfg.selection_warmup_epochs)
        selected_masks = []
        tie_weights = torch.linspace(
            0.5,
            1.5,
            candidate_codes.shape[-1],
            device=candidate_codes.device,
            dtype=candidate_codes.dtype,
        )
        tie_scores = torch.einsum("bkh,h->bk", candidate_codes, tie_weights)
        with torch.no_grad():
            for batch_index in range(edge_quality.shape[0]):
                order = sorted(
                    range(edge_quality.shape[1]),
                    key=lambda index: (
                        float(edge_quality[batch_index, index]),
                        float(tie_scores[batch_index, index]),
                    ),
                    reverse=True,
                )
                if warmup:
                    count = min(int(cfg.initial_active_edges), len(order))
                    selected = order[:count]
                else:
                    selected = self._coverage_selection(
                        candidate_A_mh[batch_index],
                        candidate_A_eh[batch_index],
                        module_present[batch_index],
                        order,
                        cfg,
                    )
                mask = torch.zeros_like(edge_quality[batch_index])
                mask[selected] = 1.0
                selected_masks.append(mask)
        return torch.stack(selected_masks, dim=0).detach()

    def _coverage_selection(
        self,
        module_assignment: torch.Tensor,
        env_assignment: torch.Tensor,
        module_present: torch.Tensor,
        order: list[int],
        cfg: UnifiedForwardConfig,
    ) -> list[int]:
        """Perform one detached coverage and redundancy selection."""

        selected: list[int] = []
        deferred: list[int] = []
        for candidate in order:
            is_novel = self._is_novel(
                module_assignment,
                env_assignment,
                candidate,
                selected,
                float(cfg.selection_maximum_redundancy),
            )
            if is_novel or len(selected) < int(cfg.minimum_active_edges):
                selected.append(candidate)
            else:
                deferred.append(candidate)
            if self._coverage_reached(module_assignment, env_assignment, module_present, selected, cfg):
                return selected
        for candidate in deferred:
            selected.append(candidate)
            if self._coverage_reached(module_assignment, env_assignment, module_present, selected, cfg):
                break
        if not selected:
            selected.append(order[0])
        return selected

    @staticmethod
    def _is_novel(
        module_assignment: torch.Tensor,
        env_assignment: torch.Tensor,
        candidate: int,
        selected: list[int],
        maximum_redundancy: float,
    ) -> bool:
        """Check maximum module/environment cosine overlap with selected edges."""

        if not selected:
            return True
        module_candidate = module_assignment[:, candidate]
        env_candidate = env_assignment[:, candidate]
        for previous in selected:
            module_previous = module_assignment[:, previous]
            env_previous = env_assignment[:, previous]
            module_overlap = torch.dot(module_candidate, module_previous) / (
                module_candidate.norm() * module_previous.norm() + EPS
            )
            env_overlap = torch.dot(env_candidate, env_previous) / (
                env_candidate.norm() * env_previous.norm() + EPS
            )
            if float(torch.maximum(module_overlap, env_overlap)) > maximum_redundancy:
                return False
        return True

    @staticmethod
    def _coverage_reached(
        module_assignment: torch.Tensor,
        env_assignment: torch.Tensor,
        module_present: torch.Tensor,
        selected: list[int],
        cfg: UnifiedForwardConfig,
    ) -> bool:
        """Check active-token coverage for a candidate subset."""

        if len(selected) < int(cfg.minimum_active_edges):
            return False
        threshold = float(cfg.selection_token_threshold)
        module_mass = module_assignment[:, selected].sum(dim=-1)
        active_modules = module_present > 0
        module_coverage = (
            ((module_mass >= threshold) & active_modules).sum().float()
            / active_modules.sum().clamp_min(1).float()
        )
        env_coverage = (env_assignment[:, selected].sum(dim=-1) >= threshold).float().mean()
        target = float(cfg.selection_coverage_rate)
        return float(module_coverage) >= target and float(env_coverage) >= target

    @staticmethod
    def _mask_and_renormalize(
        assignment: torch.Tensor,
        edge_active_mask: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Zero inactive candidates and recover unit row assignment mass."""

        selected = assignment * edge_active_mask.unsqueeze(1)
        selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(EPS)
        if token_mask is not None:
            selected = selected * token_mask.unsqueeze(-1)
        return selected

    def _assemble_output(
        self,
        *,
        module_tokens: torch.Tensor,
        module_tokens_for_hyper: torch.Tensor,
        env_tokens: torch.Tensor,
        module_centers: torch.Tensor,
        env_coords: torch.Tensor,
        module_present: torch.Tensor,
        A_me: torch.Tensor,
        module_env_context: torch.Tensor,
        candidate_A_mh: torch.Tensor,
        candidate_A_eh: torch.Tensor,
        A_mh: torch.Tensor,
        A_eh: torch.Tensor,
        candidate_state: torch.Tensor,
        candidate_codes: torch.Tensor,
        edge_quality: torch.Tensor,
        edge_active_mask: torch.Tensor,
        cfg: UnifiedForwardConfig,
    ) -> Dict[str, torch.Tensor]:
        """Assemble selected geometry, descriptors, and organizer diagnostics."""

        module_mass_raw = A_mh.sum(dim=1)
        env_mass_raw = A_eh.sum(dim=1)
        hyper_module_mass = module_mass_raw / module_mass_raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        hyper_env_mass = env_mass_raw / env_mass_raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        source_weights = A_mh / module_mass_raw.unsqueeze(1).clamp_min(EPS)
        region_weights = A_eh / env_mass_raw.unsqueeze(1).clamp_min(EPS)
        hyper_source_coords = _weighted_coords(module_centers, source_weights, cfg)
        hyper_region_coords = _weighted_coords(env_coords, region_weights, cfg)
        hyper_source_variance, hyper_source_scale = _weighted_scale(
            module_centers,
            source_weights,
            hyper_source_coords,
            cfg,
        )
        hyper_region_variance, hyper_region_scale = _weighted_scale(
            env_coords,
            region_weights,
            hyper_region_coords,
            cfg,
        )
        hyper_strength = torch.sqrt(hyper_module_mass * hyper_env_mass + EPS) * edge_active_mask
        hyper_module_purity = _assignment_purity(A_mh)
        hyper_env_purity = _assignment_purity(A_eh)
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
        selected_candidate_mass_m = (candidate_A_mh * edge_active_mask.unsqueeze(1)).sum(dim=-1)
        selected_candidate_mass_e = (candidate_A_eh * edge_active_mask.unsqueeze(1)).sum(dim=-1)
        active_modules = module_present > 0
        threshold = float(cfg.selection_token_threshold)
        module_coverage = (
            ((selected_candidate_mass_m >= threshold) & active_modules).sum(dim=-1).float()
            / active_modules.sum(dim=-1).clamp_min(1).float()
        )
        env_coverage = (selected_candidate_mass_e >= threshold).float().mean(dim=-1)
        module_entry_count = active_modules.sum(dim=-1).clamp_min(1) * A_mh.shape[-1]
        candidate_module_nonzero_fraction = (
            ((candidate_A_mh > 0) & active_modules.unsqueeze(-1)).sum(dim=(1, 2)).float()
            / module_entry_count.float()
        )
        module_nonzero_fraction = (
            ((A_mh > 0) & active_modules.unsqueeze(-1)).sum(dim=(1, 2)).float()
            / module_entry_count.float()
        )
        candidate_env_nonzero_fraction = (candidate_A_eh > 0).float().mean(dim=(1, 2))
        env_nonzero_fraction = (A_eh > 0).float().mean(dim=(1, 2))
        return {
            "A_mh": A_mh,
            "A_eh": A_eh,
            "candidate_A_mh": candidate_A_mh,
            "candidate_A_eh": candidate_A_eh,
            "hyper_state": candidate_state * edge_active_mask.unsqueeze(-1),
            "candidate_hyper_state": candidate_state,
            "candidate_slot_codes": candidate_codes,
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
            "hyper_strength": hyper_strength,
            "edge_quality": edge_quality,
            "edge_active_mask": edge_active_mask,
            "candidate_edge_count": edge_quality.new_full((edge_quality.shape[0],), float(edge_quality.shape[1])),
            "active_edge_count": edge_active_mask.sum(dim=-1),
            "selection_module_coverage": module_coverage,
            "selection_environment_coverage": env_coverage,
            "candidate_module_assignment_nonzero_fraction": candidate_module_nonzero_fraction,
            "candidate_environment_assignment_nonzero_fraction": candidate_env_nonzero_fraction,
            "module_assignment_nonzero_fraction": module_nonzero_fraction,
            "environment_assignment_nonzero_fraction": env_nonzero_fraction,
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
            "env_coords": env_coords,
            "module_centers": module_centers,
            "module_present": module_present,
            "A_me": A_me,
            "module_env_context": module_env_context,
            "hyper_module_assignment_uniform": A_mh.new_tensor(
                float(cfg.hyper_module_assignment_mode == "uniform")
            ),
        }


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
            "hyper_strength": hyper_strength,
            "edge_quality": torch.sqrt(hyper_module_purity * hyper_env_purity),
            "edge_active_mask": edge_active_mask,
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
            "hyper_module_assignment_uniform": A_mh.new_tensor(float(cfg.hyper_module_assignment_mode == "uniform")),
        }

        return output
