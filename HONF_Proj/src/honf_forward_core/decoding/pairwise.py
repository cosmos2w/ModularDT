"""CORE HONF hypergraph-centric field decoder.

Inputs are query coordinates, optional query time, organizer outputs, and an
encoded global context token. Outputs include `pred_field`, hyperedge routing
diagnostics, optional c_H value context diagnostics, and pairwise-kernel
diagnostics. This module is reusable across domains; ChannelThermal-specific
environment semantics are supplied before the core is called.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import UnifiedForwardConfig
from ..nn import FourierFeatures, LazyMLP, MLP
from ..routing import locality_bias, normalize_assignment


EPS = 1e-6


def _routed_module_retention_statistics(
    retained_mass: torch.Tensor,
    routed_pair_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Summarize retained module mass over decoder-routed query-edge pairs only."""

    retained = retained_mass.detach()
    routed = routed_pair_mask.detach().to(device=retained.device, dtype=torch.bool)
    routed_values = retained.masked_select(routed)
    zero = retained.new_zeros(())
    if routed_values.numel() == 0:
        minimum = p05 = mean = zero
    else:
        minimum = routed_values.amin()
        p05 = torch.quantile(routed_values.float(), 0.05).to(dtype=retained.dtype)
        mean = routed_values.mean()
    return {
        "routed_module_retained_mass_mean": mean,
        "routed_module_retained_mass_p05": p05,
        "routed_module_retained_mass_min": minimum,
        "routed_query_edge_pair_count": routed.sum().to(dtype=retained.dtype),
    }


def _wrap_periodic_delta(
    delta: torch.Tensor,
    lengths: torch.Tensor,
    periodic_axes: tuple[int, ...],
) -> torch.Tensor:
    """Apply the minimum-image convention only along declared axes."""

    if not periodic_axes:
        return delta
    wrapped = torch.remainder(delta + 0.5 * lengths, lengths) - 0.5 * lengths
    mask = torch.tensor(
        [axis in periodic_axes for axis in range(2)],
        device=delta.device,
        dtype=torch.bool,
    )
    return torch.where(mask, wrapped, delta)


def rectangular_boundary_features(query_xy: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
    """Legacy rectangle features retained for historical checkpoint configs."""

    lx = max(float(Lx), EPS)
    ly = max(float(Ly), EPS)
    x = query_xy[..., 0:1]
    y = query_xy[..., 1:2]
    return torch.cat([x / lx, y / ly, y / ly, (ly - y) / ly, x / lx, (lx - x) / lx], dim=-1)


def sparse_topk_softmax(
    logits: torch.Tensor,
    topk: int,
    temperature: float = 1.0,
    detach_mask: bool = True,
) -> torch.Tensor:
    """Softmax over all hyperedges or query-local top-k hyperedges."""

    k = int(topk)
    temperature = max(float(temperature), EPS)
    if k <= 0 or k >= logits.shape[-1]:
        return torch.softmax(logits / temperature, dim=-1)
    _, indices = torch.topk(logits, k=k, dim=-1)
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, indices, True)
    if detach_mask:
        mask = mask.detach()
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked_logits / temperature, dim=-1)


class HyperedgeMechanismEncoder(nn.Module):
    """Enrich hyperedge state with generic source-region mechanism descriptors."""

    def __init__(self, config: UnifiedForwardConfig):
        """Initialize HyperedgeMechanismEncoder and its required state."""

        super().__init__()
        hidden_dim = int(config.hidden_dim)
        mechanism_hidden_dim = int(config.mechanism_hidden_dim or hidden_dim)
        self.net = LazyMLP(
            hidden_dim=mechanism_hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=float(config.dropout),
        )

    def forward(self, hyper_state: torch.Tensor, mechanism_features: torch.Tensor) -> torch.Tensor:
        """Refine ``hyper_state [B,K,H]`` using descriptors ``[B,K,D]``."""

        mechanism_delta = self.net(torch.cat([hyper_state, mechanism_features], dim=-1))
        return hyper_state + mechanism_delta


class DescriptorFirstMechanismEncoder(nn.Module):
    """Construct edge state primarily from explicit mechanism descriptors."""

    def __init__(self, config: UnifiedForwardConfig):
        """Initialize shared descriptor and bounded content projections."""

        super().__init__()
        hidden_dim = int(config.hidden_dim)
        mechanism_hidden_dim = int(config.mechanism_hidden_dim or hidden_dim)
        self.descriptor_encoder = LazyMLP(
            hidden_dim=mechanism_hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=float(config.dropout),
        )
        self.content_encoder = MLP(
            hidden_dim,
            mechanism_hidden_dim,
            hidden_dim,
            num_layers=2,
            dropout=float(config.dropout),
            include_zero_dropout=True,
        )
        self.content_scale = float(config.mechanism_latent_residual_scale)
        self.norm = nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity()

    def forward(self, hyper_state: torch.Tensor, mechanism_features: torch.Tensor) -> torch.Tensor:
        """Combine descriptor state with a bounded shared content residual."""

        mechanism_state = self.descriptor_encoder(mechanism_features)
        content_state = self.content_encoder(hyper_state)
        return self.norm(mechanism_state + self.content_scale * content_state)


class HypergraphGatedPairwiseKernel(nn.Module):
    """Query-module pairwise kernel routed through learned hypergraph incidences."""

    def __init__(self, config: UnifiedForwardConfig):
        """Initialize HypergraphGatedPairwiseKernel and its required state."""

        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        kernel_hidden_dim = int(config.pairwise_kernel_hidden_dim or hidden_dim)
        self.relative_fourier = FourierFeatures(None, int(config.pairwise_kernel_fourier_frequencies))
        self.pair_mlp = LazyMLP(
            hidden_dim=kernel_hidden_dim,
            out_dim=hidden_dim,
            num_layers=int(config.pairwise_kernel_num_layers),
            dropout=float(config.dropout),
        )
        gate_init = min(max(float(config.pairwise_kernel_gate_init), 1e-4), 1.0 - 1e-4)
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.pairwise_kernel_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

    def forward(
        self,
        query_xy: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
        hyper_attention: torch.Tensor,
        *,
        gathered_execution: bool = False,
        return_routing_maps: bool = False,
        reduce_pair_context: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Aggregate query-module interactions through hyperedge routing.

        Queries ``[B,Q,2]`` and modules ``[B,M,*]`` form pair embeddings
        ``[B,Q,M,H]``. ``A_mh [B,M,K]`` pools them per hyperedge and
        ``hyper_attention [B,Q,K]`` reduces them to context ``[B,Q,H]``.
        Diagnostics stay scalar unless routing maps are requested.
        """

        cfg = self.config
        module_centers = organizer_output["module_centers"]
        module_tokens = organizer_output["module_tokens"]
        module_present = organizer_output["module_present"].to(device=query_xy.device, dtype=query_xy.dtype)
        A_mh = organizer_output["A_mh"].to(device=query_xy.device, dtype=query_xy.dtype)
        if cfg.pairwise_kernel_normalize_by_edge_mass:
            edge_module_weight = A_mh / A_mh.sum(dim=1, keepdim=True).clamp_min(EPS)
        else:
            edge_module_weight = A_mh
        if gathered_execution:
            edge_pair_context, selected_modules, evaluated_pairs, retained_module_mass = self._gathered_edge_pair_context(
                query_xy,
                module_centers,
                module_tokens,
                module_present,
                edge_module_weight,
                hyper_attention,
                organizer_output.get("module_features_raw"),
            )
        else:
            edge_pair_context = self._dense_edge_pair_context(
                query_xy,
                module_centers,
                module_tokens,
                module_present,
                edge_module_weight,
                organizer_output.get("module_features_raw"),
            )
            selected_modules = query_xy.new_tensor(float(module_present.shape[1]))
            evaluated_pairs = query_xy.new_tensor(
                float(query_xy.shape[0] * query_xy.shape[1] * module_present.shape[1])
            )
            retained_module_mass = edge_module_weight.sum(dim=1)[:, None, :].expand(
                -1, query_xy.shape[1], -1
            )
        if reduce_pair_context:
            pair_context = torch.einsum("bqk,bqkh->bqh", hyper_attention, edge_pair_context)
        else:
            pair_context = edge_pair_context.new_zeros(
                edge_pair_context.shape[0],
                edge_pair_context.shape[1],
                edge_pair_context.shape[-1],
            )
        gate = torch.sigmoid(self.pairwise_kernel_logit)
        available_modules = query_xy.new_tensor(float(module_present.shape[1]))
        retained_module_mass_detached = retained_module_mass.detach()
        routed_pair_mask = hyper_attention.detach() > 0
        retention_diagnostics = _routed_module_retention_statistics(
            retained_module_mass_detached,
            routed_pair_mask,
        )
        diagnostics = {
            "pairwise_kernel_gate": gate.detach(),
            "pairwise_context_norm": pair_context.detach().norm(dim=-1).mean(),
            "pairwise_edge_context_norm": edge_pair_context.detach().norm(dim=-1).mean(),
            "pairwise_edge_usage_mean": hyper_attention.detach().mean(),
            "pairwise_active_hyperedge_count": (hyper_attention.detach() > 0).float().sum(dim=-1).mean(),
            "pairwise_uses_sparse_hyper_attention": hyper_attention.new_tensor(
                float(
                    cfg.hyper_query_attention_mode != "uniform"
                    and (int(cfg.hyper_attention_topk) > 0 or cfg.query_assignment_normalizer == "entmax15")
                )
            ),
            "pairwise_available_modules": available_modules,
            "pairwise_selected_modules": selected_modules,
            "pairwise_selection_ratio": selected_modules / available_modules.clamp_min(1.0),
            "pairwise_evaluated_pair_count": evaluated_pairs,
            "pairwise_dense_route_count": query_xy.new_tensor(
                float(query_xy.shape[0] * query_xy.shape[1] * module_present.shape[1])
            ),
            "pairwise_gathered_route_count": evaluated_pairs if gathered_execution else query_xy.new_zeros(()),
            "retained_module_incidence_mass": retained_module_mass_detached,
            "routed_query_edge_pair_mask": routed_pair_mask,
            "all_candidate_module_retained_mass_min": retained_module_mass_detached.amin(),
            "all_candidate_module_retained_mass_p05": torch.quantile(
                retained_module_mass_detached.float(), 0.05
            ).to(query_xy.dtype),
            "all_candidate_module_retained_mass_mean": retained_module_mass_detached.mean(),
            **retention_diagnostics,
        }
        if return_routing_maps:
            # CORE HONF diagnostic: this dense [B,Q,K] tensor is only materialized
            # for explicit evaluation-time routing maps, never during normal train.
            diagnostics["pairwise_edge_contribution"] = (
                gate * hyper_attention[..., None] * edge_pair_context
            ).detach().norm(dim=-1)
        return gate * pair_context, gate * edge_pair_context, diagnostics

    def _dense_edge_pair_context(
        self,
        query_xy: torch.Tensor,
        module_centers: torch.Tensor,
        module_tokens: torch.Tensor,
        module_present: torch.Tensor,
        edge_module_weight: torch.Tensor,
        raw_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate the pair MLP over every padded query-module pair."""

        rel = self._relative_features(query_xy, module_centers)
        rel_encoded = self.relative_fourier(rel) if self.config.pairwise_kernel_use_fourier else rel
        pieces = [rel_encoded, module_present[:, None, :, None].expand(-1, query_xy.shape[1], -1, -1)]
        if self.config.pairwise_kernel_include_module_token:
            pieces.append(module_tokens[:, None, :, :].expand(-1, query_xy.shape[1], -1, -1))
        if self.config.pairwise_kernel_include_module_features and torch.is_tensor(raw_features):
            pieces.append(
                raw_features[:, None, :, :]
                .to(device=query_xy.device, dtype=query_xy.dtype)
                .expand(-1, query_xy.shape[1], -1, -1)
            )
        pair_embed = self.pair_mlp(torch.cat(pieces, dim=-1)) * module_present[:, None, :, None]
        return torch.einsum("bmk,bqmh->bqkh", edge_module_weight, pair_embed)

    def _gathered_edge_pair_context(
        self,
        query_xy: torch.Tensor,
        module_centers: torch.Tensor,
        module_tokens: torch.Tensor,
        module_present: torch.Tensor,
        edge_module_weight: torch.Tensor,
        hyper_attention: torch.Tensor,
        raw_features: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather active relevant modules before evaluating the pair MLP."""

        beta = torch.einsum("bqk,bmk->bqm", hyper_attention, edge_module_weight)
        contexts = []
        retained_masses = []
        selected_total = 0
        evaluated_pairs = 0
        limit = int(self.config.query_module_limit)
        for batch_index in range(query_xy.shape[0]):
            active_indices = torch.nonzero(module_present[batch_index] > 0, as_tuple=False).squeeze(-1)
            available = int(active_indices.numel())
            if available == 0:
                contexts.append(
                    query_xy.new_zeros(query_xy.shape[1], hyper_attention.shape[-1], module_tokens.shape[-1])
                )
                retained_masses.append(query_xy.new_zeros(query_xy.shape[1], hyper_attention.shape[-1]))
                continue
            base_count = available if limit <= 0 else min(limit, available)
            importance = beta[batch_index, :, active_indices]
            order = torch.argsort(importance, dim=-1, descending=True)
            ranked_indices = active_indices[order]
            ranked_edge_weight = edge_module_weight[batch_index][ranked_indices]
            cumulative_edge_mass = ranked_edge_weight.cumsum(dim=1)
            routed_edges = hyper_attention[batch_index] > 0
            mass_floor = float(self.config.module_incidence_retained_mass_floor)
            required_by_edge = (cumulative_edge_mass < mass_floor).sum(dim=1) + 1
            required_by_edge = torch.minimum(
                required_by_edge,
                torch.full_like(required_by_edge, available),
            )
            required_by_edge = torch.where(
                routed_edges,
                required_by_edge,
                torch.zeros_like(required_by_edge),
            )
            selected_counts = torch.maximum(
                required_by_edge.amax(dim=-1),
                torch.full(
                    (query_xy.shape[1],),
                    base_count,
                    device=query_xy.device,
                    dtype=torch.long,
                ),
            ).clamp(max=available)
            maximum_count = int(selected_counts.amax().item())
            selected_total += int(selected_counts.sum().item())
            evaluated_pairs += int(selected_counts.sum().item())
            selected_indices = ranked_indices[:, :maximum_count]
            selected_presence = (
                torch.arange(maximum_count, device=query_xy.device)[None, :]
                < selected_counts[:, None]
            ).to(dtype=query_xy.dtype).unsqueeze(-1)
            selected_centers = module_centers[batch_index][selected_indices]
            rel = self._selected_relative_features(query_xy[batch_index], selected_centers)
            rel_encoded = self.relative_fourier(rel) if self.config.pairwise_kernel_use_fourier else rel
            pieces = [rel_encoded, selected_presence]
            if self.config.pairwise_kernel_include_module_token:
                pieces.append(module_tokens[batch_index][selected_indices])
            if self.config.pairwise_kernel_include_module_features and torch.is_tensor(raw_features):
                pieces.append(
                    raw_features[batch_index][selected_indices].to(device=query_xy.device, dtype=query_xy.dtype)
                )
            pair_embed = self.pair_mlp(torch.cat(pieces, dim=-1)) * selected_presence
            selected_edge_weight = edge_module_weight[batch_index][selected_indices] * selected_presence
            retained_mass = selected_edge_weight.sum(dim=1)
            selected_edge_weight = selected_edge_weight / retained_mass.unsqueeze(1).clamp_min(EPS)
            contexts.append(torch.einsum("qmk,qmh->qkh", selected_edge_weight, pair_embed))
            retained_masses.append(retained_mass)
        return (
            torch.stack(contexts, dim=0),
            query_xy.new_tensor(
                float(selected_total) / float(max(query_xy.shape[0] * query_xy.shape[1], 1))
            ),
            query_xy.new_tensor(float(evaluated_pairs)),
            torch.stack(retained_masses, dim=0),
        )

    def _selected_relative_features(
        self,
        query_xy: torch.Tensor,
        selected_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Return geometry for gathered centers shaped ``[Q,R,2]``."""

        cfg = self.config
        scale_x, scale_y = cfg.spatial_scale()
        lx = max(scale_x, EPS)
        ly = max(scale_y, EPS)
        diag = max(math.sqrt(lx * lx + ly * ly), EPS)
        delta = query_xy[:, None, :] - selected_centers
        if cfg.periodic_dimensions():
            lengths = query_xy.new_tensor([lx, ly])
            delta = _wrap_periodic_delta(delta, lengths, cfg.periodic_dimensions())
        dx = delta[..., 0:1]
        dy = delta[..., 1:2]
        distance = torch.sqrt(dx.square() + dy.square() + EPS)
        return torch.cat(
            [dx / lx, dy / ly, distance / diag, torch.relu(dx) / lx, torch.relu(-dx) / lx, dy.abs() / ly],
            dim=-1,
        )

    def _relative_features(self, query_xy: torch.Tensor, module_centers: torch.Tensor) -> torch.Tensor:
        """Return normalized query-to-module offsets and distances ``[B,Q,M,6]``."""

        cfg = self.config
        scale_x, scale_y = cfg.spatial_scale()
        lx = max(scale_x, EPS)
        ly = max(scale_y, EPS)
        diag = max(math.sqrt(lx * lx + ly * ly), EPS)
        delta = query_xy[:, :, None, :] - module_centers[:, None, :, :]
        if cfg.periodic_dimensions():
            lengths = torch.tensor([lx, ly], device=query_xy.device, dtype=query_xy.dtype)
            delta = _wrap_periodic_delta(delta, lengths, cfg.periodic_dimensions())
        dx = delta[..., 0:1]
        dy = delta[..., 1:2]
        distance = torch.sqrt(dx.square() + dy.square() + EPS)
        return torch.cat(
            [
                dx / lx,
                dy / ly,
                distance / diag,
                torch.relu(dx) / lx,
                torch.relu(-dx) / lx,
                dy.abs() / ly,
            ],
            dim=-1,
        )

