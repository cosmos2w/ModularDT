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

from .config import UnifiedForwardConfig
from .nn import FourierFeatures, LazyMLP, MLP
from .routing import locality_bias, normalize_assignment


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


class HypergraphFieldDecoder(nn.Module):
    """Decode query fields from organized hyperedge state and ablated context."""

    def __init__(self, config: UnifiedForwardConfig):
        """Initialize HypergraphFieldDecoder and its required state."""

        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        field_dim = int(config.field_dim)

        self.query_fourier = FourierFeatures(None, int(config.query_fourier_frequencies))
        self.query_encoder = LazyMLP(hidden_dim, hidden_dim, 2, float(config.dropout))
        self.query_to_hyper = nn.Linear(hidden_dim, hidden_dim)
        self.hyper_key = nn.Linear(hidden_dim, hidden_dim)
        self.hyper_value = nn.Linear(hidden_dim, hidden_dim)
        self.hyper_geometry_bias = nn.Linear(10, 1)
        if config.mechanism_state_mode == "descriptor_first":
            self.mechanism_encoder = DescriptorFirstMechanismEncoder(config)
        elif config.use_hyper_mechanism_encoder:
            self.mechanism_encoder = HyperedgeMechanismEncoder(config)
        else:
            self.mechanism_encoder = None
        self.pairwise_kernel = (
            HypergraphGatedPairwiseKernel(config) if config.decoder_uses("pairwise") else None
        )

        if config.field_assembly_mode == "context_fusion":
            self.nonhyper_query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.direct_key = nn.Linear(hidden_dim, hidden_dim)
            self.direct_value = nn.Linear(hidden_dim, hidden_dim)
            self.global_proj = nn.Linear(hidden_dim, hidden_dim)
            self.near_proj = nn.Linear(hidden_dim, hidden_dim)
            self.context_norm = nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity()
            gate_init = min(max(float(config.direct_residual_gate_init), 1e-4), 1.0 - 1e-4)
            gate_logit = math.log(gate_init / (1.0 - gate_init))
            self.direct_residual_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
            self.pred_head = MLP(
                hidden_dim,
                hidden_dim,
                field_dim,
                num_layers=2,
                dropout=float(config.dropout),
                include_zero_dropout=True,
            )
            if config.output_mean_residual_split:
                self.mean_head = MLP(
                    hidden_dim,
                    hidden_dim,
                    field_dim,
                    num_layers=2,
                    dropout=float(config.dropout),
                    include_zero_dropout=True,
                )
                self.residual_head = MLP(
                    hidden_dim,
                    hidden_dim,
                    field_dim,
                    num_layers=2,
                    dropout=float(config.dropout),
                    include_zero_dropout=True,
                )
        else:
            self.background_query = nn.Linear(hidden_dim, hidden_dim)
            self.background_env_key = nn.Linear(hidden_dim, hidden_dim)
            self.background_env_value = nn.Linear(hidden_dim, hidden_dim)
            self.background_global = nn.Linear(hidden_dim, hidden_dim)
            self.background_input_norm = nn.LayerNorm(3 * hidden_dim)
            self.edge_input_norm = nn.LayerNorm(3 * hidden_dim + 10)
            additive_gate_init = min(max(float(config.additive_edge_gate_init), 1.0e-4), 1.0 - 1.0e-4)
            additive_gate_logit = math.log(additive_gate_init / (1.0 - additive_gate_init))
            self.additive_edge_gate = nn.Parameter(torch.tensor(additive_gate_logit, dtype=torch.float32))
            self.background_head = MLP(
                3 * hidden_dim,
                hidden_dim,
                field_dim,
                num_layers=2,
                dropout=float(config.dropout),
                include_zero_dropout=True,
            )
            self.edge_head = MLP(
                3 * hidden_dim + 10,
                hidden_dim,
                field_dim,
                num_layers=2,
                dropout=float(config.dropout),
                include_zero_dropout=True,
            )
            output_std = float(config.additive_output_init_std)
            for head in (self.background_head, self.edge_head):
                final = head.net[-1]
                if not isinstance(final, nn.Linear):
                    raise RuntimeError("Additive output heads must end in a linear layer.")
                nn.init.normal_(final.weight, mean=0.0, std=output_std)
                nn.init.zeros_(final.bias)

    def forward(
        self,
        query_xy: torch.Tensor,
        query_time: Optional[torch.Tensor],
        organizer_output: Dict[str, torch.Tensor],
        global_context: Optional[torch.Tensor],
        query_features: Optional[torch.Tensor] = None,
        *,
        return_routing_maps: bool = False,
        return_edge_fields: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Decode field values at query coordinates.

        ``query_xy [B,Q,2]`` becomes ``query_state [B,Q,H]``. According to
        ``decoder_mode``, queries attend to ``hyper_state [B,K,H]`` through
        ``alpha_qk [B,Q,K]`` and may add pairwise, global, direct-memory, or
        near-module context. The prediction head returns ``pred_field
        [B,Q,F]``. Dense routing tensors are produced only on request.
        """

        cfg = self.config
        encoded_query_features = self._query_features(query_xy, query_time, query_features)
        query_state = self.query_encoder(encoded_query_features)

        context_fusion = cfg.field_assembly_mode == "context_fusion"
        uses_hyper = cfg.decoder_uses("hyper")
        uses_hyper_value = bool(cfg.use_hyper_value_context and context_fusion)
        uses_global = bool(context_fusion and self._uses_global())
        uses_direct = bool(context_fusion and self._uses_direct())
        uses_near = bool(context_fusion and self._uses_near_module())
        uses_pairwise = bool(cfg.decoder_uses("pairwise") and self.pairwise_kernel is not None)
        execution_flag = organizer_output.get("routing_execution_gathered")
        if torch.is_tensor(execution_flag):
            gathered_execution = bool(float(execution_flag.detach().reshape(-1)[0]) >= 0.5)
        else:
            gathered_execution = cfg.routing_execution == "gathered"
        hyper_context = torch.zeros_like(query_state)
        nonhyper_context = torch.zeros_like(query_state)
        edge_pair_context: Optional[torch.Tensor] = None
        hyper_state = organizer_output["hyper_state"]
        diagnostics: Dict[str, torch.Tensor | str] = {"decoder_mode": cfg.decoder_mode}
        if not context_fusion:
            diagnostics["field_assembly_mode"] = cfg.field_assembly_mode
        diagnostics["query_feature_dim"] = torch.tensor(float(encoded_query_features.shape[-1]), device=query_xy.device, dtype=query_xy.dtype)
        diagnostics["uses_query_fourier"] = torch.tensor(float(int(cfg.query_fourier_frequencies) > 0), device=query_xy.device, dtype=query_xy.dtype)
        diagnostics["uses_boundary_features"] = torch.tensor(
            float(cfg.boundary_feature_mode in {"rectangular", "channel"} or query_features is not None),
            device=query_xy.device,
            dtype=query_xy.dtype,
        )
        if uses_hyper:
            hyper_state_raw = organizer_output["hyper_state"]
            mechanism_features = self._mechanism_features(organizer_output)
            if self.mechanism_encoder is not None and torch.is_tensor(mechanism_features):
                hyper_state = self.mechanism_encoder(hyper_state_raw, mechanism_features)
                diagnostics["use_hyper_mechanism_encoder"] = torch.tensor(1.0, device=query_xy.device, dtype=query_xy.dtype)
                diagnostics["mechanism_state_norm"] = hyper_state.detach().norm(dim=-1).mean()
                diagnostics["mechanism_raw_feature_dim"] = torch.tensor(
                    float(mechanism_features.shape[-1]),
                    device=query_xy.device,
                    dtype=query_xy.dtype,
                )
            else:
                hyper_state = hyper_state_raw
                diagnostics["use_hyper_mechanism_encoder"] = torch.tensor(0.0, device=query_xy.device, dtype=query_xy.dtype)
                diagnostics["mechanism_state_norm"] = hyper_state.detach().norm(dim=-1).mean()
            geometry_features = organizer_output.get("mechanism_geometry_features")
            mass_features = organizer_output.get("mechanism_mass_features")
            if torch.is_tensor(geometry_features):
                diagnostics["mechanism_geometry_feature_mean"] = geometry_features.detach().mean()
            if torch.is_tensor(mass_features):
                diagnostics["mechanism_mass_feature_mean"] = mass_features.detach().mean()
            hyper_logits = torch.einsum(
                "bqh,bkh->bqk",
                self.query_to_hyper(query_state),
                self.hyper_key(hyper_state),
            ) / math.sqrt(float(query_state.shape[-1]))
            if cfg.use_hyper_geometry_bias:
                geometry_bias = self.hyper_geometry_bias(self._hyper_geometry_features(query_xy, organizer_output)).squeeze(-1)
                hyper_logits = hyper_logits + float(cfg.hyper_geometry_bias_scale) * geometry_bias
            else:
                geometry_bias = torch.zeros_like(hyper_logits)
            query_locality_mode = (
                cfg.environment_locality_mode
                if cfg.query_locality_mode == "inherit_environment"
                else cfg.query_locality_mode
            )
            if not context_fusion and query_locality_mode != "none":
                query_locality_bias = self._query_locality_bias(query_xy, organizer_output)
                hyper_logits = hyper_logits + query_locality_bias
            else:
                query_locality_bias = torch.zeros_like(hyper_logits)
            edge_active_mask = organizer_output.get("effective_edge_mask")
            if not torch.is_tensor(edge_active_mask):
                edge_active_mask = organizer_output.get("edge_active_mask")
            if not torch.is_tensor(edge_active_mask):
                edge_active_mask = torch.ones_like(hyper_logits[:, 0, :])
            edge_active_mask = edge_active_mask.to(device=hyper_logits.device, dtype=hyper_logits.dtype)
            uses_descriptive_query_normalizer = (
                not context_fusion or cfg.query_assignment_normalizer != "softmax"
            )
            if uses_descriptive_query_normalizer:
                hyper_logits = hyper_logits.masked_fill(
                    edge_active_mask[:, None, :] <= 0,
                    torch.finfo(hyper_logits.dtype).min,
                )
            if cfg.hyper_query_attention_mode == "uniform":
                if not uses_descriptive_query_normalizer:
                    hyper_attention = torch.full_like(hyper_logits, 1.0 / float(max(hyper_logits.shape[-1], 1)))
                else:
                    hyper_attention = edge_active_mask[:, None, :].expand_as(hyper_logits)
                    hyper_attention = hyper_attention / hyper_attention.sum(dim=-1, keepdim=True).clamp_min(EPS)
            else:
                if not uses_descriptive_query_normalizer:
                    hyper_attention = sparse_topk_softmax(
                        hyper_logits,
                        topk=int(cfg.hyper_attention_topk),
                        temperature=float(cfg.hyper_attention_temperature),
                        detach_mask=bool(cfg.sparse_hyper_attention_detach_mask),
                    )
                else:
                    hyper_attention = normalize_assignment(
                        hyper_logits / max(float(cfg.hyper_attention_temperature), EPS),
                        mode=cfg.query_assignment_normalizer,
                        mask=edge_active_mask[:, None, :] > 0,
                        entmax_blend=float(
                            organizer_output.get(
                                "query_sparsity_fraction",
                                hyper_logits.new_zeros(()),
                            )
                        ),
                    )
                    hyper_attention = self._limit_probability_routes(hyper_attention)
            if not context_fusion and gathered_execution:
                hyper_attention, retained_query_mass = self._limit_query_edge_routes(
                    hyper_attention,
                    edge_active_mask,
                )
            else:
                retained_query_mass = hyper_attention.sum(dim=-1)
            if uses_hyper_value:
                hyper_context = torch.einsum(
                    "bqk,bkh->bqh",
                    hyper_attention,
                    self.hyper_value(hyper_state),
                )
            else:
                hyper_context = torch.zeros_like(query_state)
            c_h_context = hyper_context
            diagnostics["hyper_value_context_norm"] = c_h_context.detach().norm(dim=-1).mean()
            diagnostics["hyper_attention_mean"] = hyper_attention.mean(dim=1)
            hyper_entropy = -(hyper_attention * torch.log(hyper_attention.clamp_min(EPS))).sum(dim=-1)
            diagnostics["hyper_attention_topk"] = torch.tensor(float(cfg.hyper_attention_topk), device=query_xy.device, dtype=query_xy.dtype)
            diagnostics["hyper_attention_temperature"] = torch.tensor(float(cfg.hyper_attention_temperature), device=query_xy.device, dtype=query_xy.dtype)
            diagnostics["hyper_query_attention_uniform"] = torch.tensor(
                float(cfg.hyper_query_attention_mode == "uniform"),
                device=query_xy.device,
                dtype=query_xy.dtype,
            )
            diagnostics["hyper_attention_entropy"] = hyper_entropy.detach().mean()
            diagnostics["hyper_attention_effective_edges"] = torch.exp(hyper_entropy.detach()).mean()
            diagnostics["hyper_attention_max"] = hyper_attention.detach().amax(dim=-1).mean()
            diagnostics["hyper_attention_nonzero_count"] = (hyper_attention.detach() > 0).float().sum(dim=-1).mean()
            diagnostics["effective_query_edge_count"] = (hyper_attention.detach() > 0).float().sum(dim=-1).mean()
            diagnostics["query_edge_retained_probability_mass"] = retained_query_mass.detach()
            diagnostics["query_edge_retained_probability_mass_min"] = retained_query_mass.detach().amin()
            diagnostics["query_edge_retained_probability_mass_p05"] = torch.quantile(
                retained_query_mass.detach().float(), 0.05
            ).to(query_xy.dtype)
            diagnostics["query_edge_retained_probability_mass_mean"] = retained_query_mass.detach().mean()
            diagnostics["hyper_geometry_bias_mean"] = geometry_bias.detach().mean()
            diagnostics["hyper_geometry_bias_std"] = geometry_bias.detach().std(unbiased=False)
            if not context_fusion:
                diagnostics["query_locality_bias_mean"] = query_locality_bias.detach().mean()
                diagnostics["query_locality_bias_std"] = query_locality_bias.detach().std(unbiased=False)
                diagnostics["query_assignment_nonzero_fraction"] = (hyper_attention.detach() > 0).float().mean()
                diagnostics["mean_query_nonzero_edges"] = (hyper_attention.detach() > 0).float().sum(dim=-1).mean()
                diagnostics["query_assignment_normalizer"] = cfg.query_assignment_normalizer
                diagnostics["routing_execution"] = "gathered" if gathered_execution else "dense"
            if uses_pairwise:
                pair_context, routed_edge_context, pair_diagnostics = self.pairwise_kernel(
                    query_xy,
                    organizer_output,
                    hyper_attention,
                    gathered_execution=gathered_execution,
                    return_routing_maps=return_routing_maps,
                    reduce_pair_context=context_fusion,
                )
                if not context_fusion:
                    edge_pair_context = routed_edge_context
                else:
                    del routed_edge_context
                if context_fusion:
                    hyper_context = hyper_context + pair_context
                diagnostics.update(pair_diagnostics)
                if return_routing_maps:
                    diagnostics["c_pair_norm"] = pair_context.detach().norm(dim=-1)
            elif return_routing_maps:
                diagnostics["c_pair_norm"] = torch.zeros(query_xy.shape[:2], device=query_xy.device, dtype=query_xy.dtype)
            if return_routing_maps:
                # CORE HONF diagnostic: alpha_qk is query-dependent and therefore
                # requested only for explicit routing visualization/export.
                diagnostics["query_hyper_attention"] = hyper_attention.detach()
                diagnostics["dominant_hyperedge"] = hyper_attention.detach().argmax(dim=-1)
                diagnostics["hyper_attention_entropy_map"] = hyper_entropy.detach()
                diagnostics["c_H_norm"] = c_h_context.detach().norm(dim=-1)
        else:
            nonhyper_context = self.nonhyper_query_proj(query_state)
            diagnostics["hyper_geometry_bias_mean"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)
            diagnostics["hyper_geometry_bias_std"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)
            diagnostics["use_hyper_mechanism_encoder"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)
        if not uses_pairwise:
            diagnostics["pairwise_kernel_gate"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)
            diagnostics["pairwise_context_norm"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)
            diagnostics["pairwise_edge_context_norm"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)
            diagnostics["pairwise_edge_usage_mean"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)
            diagnostics["pairwise_active_hyperedge_count"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)
            diagnostics["pairwise_uses_sparse_hyper_attention"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)
            diagnostics.setdefault("hyper_value_context_norm", torch.zeros((), device=query_xy.device, dtype=query_xy.dtype))
            if return_routing_maps:
                batch, num_query = query_xy.shape[:2]
                num_hyper = int(organizer_output.get("hyper_state", query_xy.new_zeros(batch, 0, query_state.shape[-1])).shape[1])
                diagnostics["query_hyper_attention"] = query_xy.new_zeros(batch, num_query, num_hyper)
                diagnostics["pairwise_edge_contribution"] = query_xy.new_zeros(batch, num_query, num_hyper)
                diagnostics["dominant_hyperedge"] = torch.zeros(batch, num_query, device=query_xy.device, dtype=torch.long)
                diagnostics["hyper_attention_entropy_map"] = query_xy.new_zeros(batch, num_query)
                diagnostics["c_H_norm"] = query_xy.new_zeros(batch, num_query)
                diagnostics["c_pair_norm"] = query_xy.new_zeros(batch, num_query)
        context = hyper_context + nonhyper_context

        diagnostics["uses_hyper_context"] = torch.tensor(float(uses_hyper), device=query_xy.device, dtype=query_xy.dtype)
        diagnostics["uses_hyper_value_context"] = torch.tensor(float(uses_hyper and uses_hyper_value), device=query_xy.device, dtype=query_xy.dtype)
        diagnostics["uses_global_context"] = torch.tensor(float(uses_global), device=query_xy.device, dtype=query_xy.dtype)
        diagnostics["uses_direct_context"] = torch.tensor(float(uses_direct), device=query_xy.device, dtype=query_xy.dtype)
        diagnostics["uses_near_module_context"] = torch.tensor(float(uses_near), device=query_xy.device, dtype=query_xy.dtype)
        diagnostics["pairwise_kernel_enabled"] = torch.tensor(float(uses_pairwise), device=query_xy.device, dtype=query_xy.dtype)

        if uses_global and global_context is not None:
            addition = self.global_proj(global_context).unsqueeze(1)
            context = context + addition
            nonhyper_context = nonhyper_context + addition

        if uses_direct:
            direct_context, direct_attention = self._direct_context(query_state, organizer_output)
            gate = torch.sigmoid(self.direct_residual_logit)
            context = context + gate * direct_context
            nonhyper_context = nonhyper_context + gate * direct_context
            diagnostics["direct_attention_mean"] = direct_attention.mean(dim=1)
            diagnostics["direct_residual_gate"] = gate.detach()
        else:
            diagnostics["direct_residual_gate"] = torch.zeros((), device=query_xy.device, dtype=query_xy.dtype)

        if uses_near:
            addition = self.near_proj(self._near_module_context(query_xy, organizer_output))
            context = context + addition
            nonhyper_context = nonhyper_context + addition

        diagnostics["hyper_context_norm"] = hyper_context.detach().norm(dim=-1).mean()
        diagnostics["total_hyper_context_norm"] = diagnostics["hyper_context_norm"]
        diagnostics["nonhyper_context_norm"] = nonhyper_context.detach().norm(dim=-1).mean()
        diagnostics["context_norm"] = context.detach().norm(dim=-1).mean()

        output: Dict[str, torch.Tensor | str] = dict(diagnostics)
        if not context_fusion:
            if edge_pair_context is None:
                raise RuntimeError("edge_additive field assembly requires edge-local pair context.")
            output["mechanism_state"] = hyper_state
            output.update(
                self._edge_additive_output(
                    query_xy=query_xy,
                    query_state=query_state,
                    hyper_state=hyper_state,
                    hyper_attention=hyper_attention,
                    edge_pair_context=edge_pair_context,
                    organizer_output=organizer_output,
                    global_context=global_context,
                    gathered_execution=gathered_execution,
                    return_edge_fields=bool(return_edge_fields),
                )
            )
            return output  # type: ignore[return-value]

        context = self.context_norm(context)
        if cfg.output_mean_residual_split:
            pred_mean = self.mean_head(context)
            pred_residual = self.residual_head(context)
            output["pred_mean"] = pred_mean
            output["pred_residual"] = pred_residual
            output["pred_field"] = pred_mean + pred_residual
        else:
            output["pred_field"] = self.pred_head(context)
        return output  # type: ignore[return-value]

    def _additive_background_field(
        self,
        query_state: torch.Tensor,
        env_tokens: torch.Tensor,
        global_context: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the configured key-compatible additive background path."""

        if self.config.additive_background_mode == "dense_query_attention":
            # Keep the legacy operation order literal: existing checkpoints in
            # the default mode must retain bitwise-identical arithmetic.
            env_logits = torch.einsum(
                "bqh,beh->bqe",
                self.background_query(query_state),
                self.background_env_key(env_tokens),
            ) / math.sqrt(float(query_state.shape[-1]))
            env_attention = torch.softmax(env_logits, dim=-1)
            env_context = torch.einsum(
                "bqe,beh->bqh",
                env_attention,
                self.background_env_value(env_tokens),
            )
        else:
            case_query_state = (
                torch.zeros_like(query_state[:, 0, :])
                if global_context is None
                else global_context
            )
            env_logits = torch.einsum(
                "bh,beh->be",
                self.background_query(case_query_state),
                self.background_env_key(env_tokens),
            ) / math.sqrt(float(query_state.shape[-1]))
            env_attention = torch.softmax(env_logits, dim=-1)
            pooled_env_context = torch.einsum(
                "be,beh->bh",
                env_attention,
                self.background_env_value(env_tokens),
            )
            env_context = pooled_env_context.unsqueeze(1).expand_as(query_state)
        if global_context is None:
            global_state = torch.zeros_like(query_state)
        else:
            global_state = self.background_global(global_context).unsqueeze(1).expand_as(query_state)
        background_input = self.background_input_norm(
            torch.cat([query_state, global_state, env_context], dim=-1)
        )
        return self.background_head(background_input), env_attention

    def _edge_additive_output(
        self,
        *,
        query_xy: torch.Tensor,
        query_state: torch.Tensor,
        hyper_state: torch.Tensor,
        hyper_attention: torch.Tensor,
        edge_pair_context: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
        global_context: Optional[torch.Tensor],
        gathered_execution: bool,
        return_edge_fields: bool,
    ) -> Dict[str, torch.Tensor]:
        """Assemble an exact background-plus-edge field decomposition."""

        env_tokens = organizer_output["env_tokens"]
        background, env_attention = self._additive_background_field(
            query_state,
            env_tokens,
            global_context,
        )
        additive_gate = torch.sigmoid(self.additive_edge_gate)

        geometry_features = self._hyper_geometry_features(query_xy, organizer_output)
        edge_active_mask = organizer_output.get("effective_edge_mask")
        if not torch.is_tensor(edge_active_mask):
            edge_active_mask = organizer_output.get("edge_active_mask")
        if not torch.is_tensor(edge_active_mask):
            edge_active_mask = torch.ones_like(hyper_attention[:, 0, :])
        if gathered_execution:
            edge_sum, edge_abs_mean, edge_rms, edge_energy, edge_field, selected_routes = (
                self._gathered_edge_execution(
                    query_state,
                    hyper_state,
                    geometry_features,
                    edge_pair_context,
                    hyper_attention,
                    edge_active_mask,
                    additive_gate,
                    return_edge_fields=return_edge_fields,
                )
            )
        else:
            edge_sum, edge_abs_mean, edge_rms, edge_energy, edge_field = self._dense_edge_execution(
                query_state,
                hyper_state,
                geometry_features,
                edge_pair_context,
                hyper_attention,
                edge_active_mask,
                additive_gate,
                return_edge_fields=return_edge_fields,
            )
            selected_routes = query_state.new_tensor(
                float(query_state.shape[0] * query_state.shape[1] * hyper_state.shape[1])
            )
        pred_field = background + edge_sum
        background_norm = background.detach().norm(dim=-1).mean()
        edge_norm = edge_sum.detach().norm(dim=-1).mean()
        field_norm = pred_field.detach().norm(dim=-1).mean()
        cancellation_ratio = torch.relu(
            (background_norm + edge_norm - field_norm)
            / (background_norm + edge_norm + EPS)
        )

        available_routes = query_state.new_tensor(
            float(query_state.shape[0] * query_state.shape[1] * hyper_state.shape[1])
        )
        output = {
            "pred_field": pred_field,
            "additive_background_mode": self.config.additive_background_mode,
            "background_attention_element_count": query_state.new_tensor(
                float(env_attention.numel())
            ),
            "edge_contribution_abs_mean": edge_abs_mean,
            "edge_contribution_rms": edge_rms,
            "edge_contribution_energy_fraction": (
                edge_energy / edge_energy.sum(dim=1, keepdim=True).clamp_min(EPS)
            ),
            "additive_edge_gate": additive_gate.detach(),
            "background_field_norm": background_norm,
            "summed_edge_field_norm": edge_norm,
            "edge_field_fraction": (
                edge_norm / (field_norm + EPS)
            ),
            "background_edge_cancellation_ratio": cancellation_ratio,
            "edge_head_available_routes": available_routes,
            "edge_head_selected_routes": selected_routes,
            "edge_head_selection_ratio": selected_routes / available_routes.clamp_min(1.0),
            "edge_head_evaluated_route_count": selected_routes,
            "edge_head_dense_route_count": available_routes,
            "edge_head_gathered_route_count": (
                selected_routes if gathered_execution else query_state.new_zeros(())
            ),
        }
        if return_edge_fields:
            output["pred_field_background"] = background
            if edge_field is None:
                raise RuntimeError("Requested edge fields were not materialized.")
            output["pred_field_by_edge"] = edge_field
        return output

    def _dense_edge_execution(
        self,
        query_state: torch.Tensor,
        hyper_state: torch.Tensor,
        geometry_features: torch.Tensor,
        edge_pair_context: torch.Tensor,
        hyper_attention: torch.Tensor,
        edge_active_mask: torch.Tensor,
        additive_gate: torch.Tensor,
        *,
        return_edge_fields: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate every candidate edge head as the dense reference path."""

        query_by_edge = query_state[:, :, None, :].expand(-1, -1, hyper_state.shape[1], -1)
        state_by_query = hyper_state[:, None, :, :].expand(-1, query_state.shape[1], -1, -1)
        edge_input = self.edge_input_norm(
            torch.cat([query_by_edge, state_by_query, geometry_features, edge_pair_context], dim=-1)
        )
        raw_edge_field = self.edge_head(edge_input)
        active = edge_active_mask.to(device=raw_edge_field.device, dtype=raw_edge_field.dtype)[:, None, :, None]
        edge_field = additive_gate * active * hyper_attention.unsqueeze(-1) * raw_edge_field
        edge_sum = edge_field.sum(dim=2)
        detached = edge_field.detach()
        edge_mean_square = detached.square().mean(dim=1)
        edge_rms = torch.where(edge_mean_square > 0, torch.sqrt(edge_mean_square), edge_mean_square)
        return (
            edge_sum,
            detached.abs().mean(dim=1),
            edge_rms,
            detached.square().sum(dim=1),
            edge_field if return_edge_fields else None,
        )

    def _gathered_edge_execution(
        self,
        query_state: torch.Tensor,
        hyper_state: torch.Tensor,
        geometry_features: torch.Tensor,
        edge_pair_context: torch.Tensor,
        hyper_attention: torch.Tensor,
        edge_active_mask: torch.Tensor,
        additive_gate: torch.Tensor,
        *,
        return_edge_fields: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Evaluate the shared edge head only on selected nonzero routes."""

        active = edge_active_mask.to(device=hyper_attention.device, dtype=torch.bool)[:, None, :]
        selected_mask = active & (hyper_attention > 0)
        selected_indices = torch.nonzero(selected_mask, as_tuple=False)
        batch_size, num_queries, num_edges = hyper_attention.shape
        field_dim = int(self.config.field_dim)
        edge_sum = query_state.new_zeros(batch_size, num_queries, field_dim)
        abs_sum = query_state.new_zeros(batch_size * num_edges, field_dim)
        square_sum = query_state.new_zeros(batch_size * num_edges, field_dim)
        edge_field: Optional[torch.Tensor] = None
        if return_edge_fields:
            edge_field = query_state.new_zeros(batch_size, num_queries, num_edges, field_dim)
        if selected_indices.shape[0] > 0:
            batch_index, query_index, edge_index = selected_indices.unbind(dim=1)
            edge_input = self.edge_input_norm(
                torch.cat(
                    [
                        query_state[batch_index, query_index],
                        hyper_state[batch_index, edge_index],
                        geometry_features[batch_index, query_index, edge_index],
                        edge_pair_context[batch_index, query_index, edge_index],
                    ],
                    dim=-1,
                )
            )
            raw_selected = self.edge_head(edge_input)
            selected_field = additive_gate * hyper_attention[batch_index, query_index, edge_index, None] * raw_selected
            edge_sum = edge_sum.index_put(
                (batch_index, query_index),
                selected_field,
                accumulate=True,
            )
            flat_edge_index = batch_index * num_edges + edge_index
            detached = selected_field.detach()
            abs_sum = abs_sum.index_add(0, flat_edge_index, detached.abs())
            square_sum = square_sum.index_add(0, flat_edge_index, detached.square())
            if edge_field is not None:
                edge_field = edge_field.index_put(
                    (batch_index, query_index, edge_index),
                    selected_field,
                    accumulate=False,
                )
        abs_mean = abs_sum.reshape(batch_size, num_edges, field_dim) / float(max(num_queries, 1))
        edge_energy = square_sum.reshape(batch_size, num_edges, field_dim)
        mean_square = edge_energy / float(max(num_queries, 1))
        edge_rms = torch.where(mean_square > 0, torch.sqrt(mean_square), mean_square)
        return (
            edge_sum,
            abs_mean,
            edge_rms,
            edge_energy,
            edge_field,
            query_state.new_tensor(float(selected_indices.shape[0])),
        )

    def _limit_probability_routes(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Optionally retain only the highest-probability query routes."""

        limit = int(self.config.hyper_attention_topk)
        if limit <= 0 or limit >= probabilities.shape[-1]:
            return probabilities
        indices = torch.topk(probabilities, k=limit, dim=-1).indices
        mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(-1, indices, True)
        if self.config.sparse_hyper_attention_detach_mask:
            mask = mask.detach()
        limited = probabilities * mask.to(dtype=probabilities.dtype)
        return limited / limited.sum(dim=-1, keepdim=True).clamp_min(EPS)

    def _limit_query_edge_routes(
        self,
        probabilities: torch.Tensor,
        edge_active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the gathered edge limit once and conserve retained route mass."""

        active = edge_active_mask.to(device=probabilities.device, dtype=torch.bool)[:, None, :]
        support = active & (probabilities > 0)
        selected_mask = support
        limit = int(self.config.query_edge_limit)
        if 0 < limit < probabilities.shape[-1]:
            ranked_probability, ranked_indices = torch.sort(probabilities, dim=-1, descending=True)
            cumulative_mass = ranked_probability.cumsum(dim=-1)
            mass_floor = float(self.config.query_edge_retained_mass_floor)
            required_for_mass = (cumulative_mass < mass_floor).sum(dim=-1) + 1
            support_count = support.sum(dim=-1)
            selected_count = torch.maximum(
                required_for_mass,
                torch.full_like(required_for_mass, limit),
            )
            selected_count = torch.minimum(selected_count, support_count)
            ranked_position = torch.arange(
                probabilities.shape[-1],
                device=probabilities.device,
            ).view(1, 1, -1)
            ranked_mask = ranked_position < selected_count.unsqueeze(-1)
            limit_mask = torch.zeros_like(selected_mask).scatter_(
                -1,
                ranked_indices,
                ranked_mask,
            )
            selected_mask = support & limit_mask.detach()
        retained = probabilities * selected_mask.to(dtype=probabilities.dtype)
        retained_mass = retained.sum(dim=-1)
        normalized = retained / retained_mass.unsqueeze(-1).clamp_min(EPS)
        return normalized, retained_mass

    def _query_locality_bias(
        self,
        query_xy: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the configured normalized-distance bias for query-to-edge routing."""

        region = organizer_output["hyper_region_coords"]
        region_scale = organizer_output["hyper_region_scale"]
        scale_x, scale_y = self.config.spatial_scale()
        minimum = query_xy.new_tensor(
            [
                max(scale_x * float(self.config.minimum_region_scale), EPS),
                max(scale_y * float(self.config.minimum_region_scale), EPS),
            ]
        )
        anisotropic_scale = torch.maximum(region_scale, minimum)
        delta = query_xy[:, :, None, :] - region[:, None, :, :]
        if self.config.periodic_dimensions():
            lengths = query_xy.new_tensor([max(scale_x, EPS), max(scale_y, EPS)])
            delta = _wrap_periodic_delta(delta, lengths, self.config.periodic_dimensions())
        radius_square = (delta / anisotropic_scale[:, None, :, :]).square().sum(dim=-1)
        return locality_bias(
            radius_square,
            mode=(
                self.config.environment_locality_mode
                if self.config.query_locality_mode == "inherit_environment"
                else self.config.query_locality_mode
            ),
            strength=self.config.environment_locality_strength,
            radius_cap=self.config.locality_radius_cap,
        )

    def _query_features(
        self,
        query_xy: torch.Tensor,
        query_time: Optional[torch.Tensor],
        case_query_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode normalized coordinates, time, Fourier, and boundary features."""

        scale_x, scale_y = self.config.spatial_scale()
        lx = max(scale_x, EPS)
        ly = max(scale_y, EPS)
        xy = torch.stack([query_xy[..., 0] / lx, query_xy[..., 1] / ly], dim=-1)
        if query_time is None:
            t = torch.zeros_like(query_xy[..., :1])
        else:
            t = query_time[..., :1]
        if self.config.query_time_mode == "phase":
            t_sin = torch.sin(2.0 * math.pi * t)
            t_cos = torch.cos(2.0 * math.pi * t)
        elif self.config.query_time_mode == "physical_time":
            t_sin = torch.sin(t)
            t_cos = torch.cos(t)
        else:
            t = torch.zeros_like(t)
            t_sin = torch.zeros_like(t)
            t_cos = torch.ones_like(t)
        base = torch.cat([xy, t, t_sin, t_cos], dim=-1)
        query_fourier = self.query_fourier(xy)
        pieces = [base, query_fourier[..., xy.shape[-1] :]]
        if self.config.boundary_feature_mode in {"rectangular", "channel"}:
            pieces.append(rectangular_boundary_features(query_xy, lx, ly))
        if case_query_features is not None:
            if case_query_features.shape[:2] != query_xy.shape[:2]:
                raise ValueError("query_features must align with query coordinates as [B,Q,Fq].")
            pieces.append(case_query_features.to(device=query_xy.device, dtype=query_xy.dtype))
        return torch.cat(pieces, dim=-1)

    def _mechanism_features(self, organizer_output: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Select and concatenate enabled per-hyperedge mechanism descriptors."""

        if self.config.mechanism_state_mode == "descriptor_first":
            descriptors = organizer_output.get("mechanism_descriptor_features")
            return descriptors if torch.is_tensor(descriptors) else None
        pieces: list[torch.Tensor] = []
        has_split_features = False
        geometry = organizer_output.get("mechanism_geometry_features")
        mass = organizer_output.get("mechanism_mass_features")
        has_split_features = torch.is_tensor(geometry) or torch.is_tensor(mass)
        if self.config.mechanism_include_geometry:
            if torch.is_tensor(geometry):
                pieces.append(geometry)
        if self.config.mechanism_include_masses:
            if torch.is_tensor(mass):
                pieces.append(mass)
        if pieces:
            return torch.cat(pieces, dim=-1)
        if has_split_features:
            return None
        raw = organizer_output.get("mechanism_raw_features")
        return raw if torch.is_tensor(raw) else None

    def _hyper_geometry_features(
        self,
        query_xy: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Describe queries relative to hyperedge source and region centroids."""

        source = organizer_output["hyper_source_coords"]
        region = organizer_output["hyper_region_coords"]
        source_delta, source_downstream, source_lateral = self._relative_geometry(query_xy, source)
        region_delta, region_downstream, region_lateral = self._relative_geometry(query_xy, region)
        scale_x, scale_y = self.config.spatial_scale()
        diag = math.sqrt(max(scale_x, EPS) ** 2 + max(scale_y, EPS) ** 2)
        source_dist = torch.sqrt(source_delta.square().sum(dim=-1, keepdim=True) + EPS) / max(diag, EPS)
        region_dist = torch.sqrt(region_delta.square().sum(dim=-1, keepdim=True) + EPS) / max(diag, EPS)
        lx = max(scale_x, EPS)
        ly = max(scale_y, EPS)
        return torch.cat(
            [
                source_delta[..., 0:1] / lx,
                source_delta[..., 1:2] / ly,
                region_delta[..., 0:1] / lx,
                region_delta[..., 1:2] / ly,
                source_dist,
                region_dist,
                source_downstream,
                region_downstream,
                source_lateral,
                region_lateral,
            ],
            dim=-1,
        )

    def _relative_geometry(
        self,
        query_xy: torch.Tensor,
        hyper_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return query-to-hyperedge offsets plus downstream/lateral distances."""

        scale_x, scale_y = self.config.spatial_scale()
        lx = max(scale_x, EPS)
        ly = max(scale_y, EPS)
        delta = query_xy[:, :, None, :] - hyper_coords[:, None, :, :]
        periodic_axes = self.config.periodic_dimensions()
        if periodic_axes:
            lengths = torch.tensor([lx, ly], device=query_xy.device, dtype=query_xy.dtype)
            raw_dx = delta[..., 0]
            delta = _wrap_periodic_delta(delta, lengths, periodic_axes)
            downstream = (
                torch.remainder(raw_dx, lx).unsqueeze(-1) / lx
                if 0 in periodic_axes
                else torch.relu(delta[..., 0:1]) / lx
            )
        else:
            downstream = torch.relu(delta[..., 0:1]) / lx
        lateral = delta[..., 1:2].abs() / ly
        return delta, downstream, lateral

    def _direct_context(
        self,
        query_state: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend directly from each query to module and environment tokens."""

        tokens = torch.cat([organizer_output["module_tokens"], organizer_output["env_tokens"]], dim=1)
        logits = torch.einsum("bqh,bnh->bqn", query_state, self.direct_key(tokens)) / math.sqrt(float(query_state.shape[-1]))
        module_present = organizer_output["module_present"]
        env_mask = torch.ones(
            module_present.shape[0],
            organizer_output["env_tokens"].shape[1],
            device=module_present.device,
            dtype=module_present.dtype,
        )
        mask = torch.cat([module_present, env_mask], dim=1).unsqueeze(1)
        logits = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1) * mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return torch.einsum("bqn,bnh->bqh", attention, self.direct_value(tokens)), attention

    def _near_module_context(
        self,
        query_xy: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Gaussian-pool nearby module tokens for every query point."""

        module_centers = organizer_output["module_centers"]
        module_tokens = organizer_output["module_tokens"]
        module_present = organizer_output["module_present"]
        delta = query_xy[:, :, None, :] - module_centers[:, None, :, :]
        periodic_axes = self.config.periodic_dimensions()
        if periodic_axes:
            scale_x, scale_y = self.config.spatial_scale()
            lengths = torch.tensor(
                [max(scale_x, EPS), max(scale_y, EPS)],
                device=query_xy.device,
                dtype=query_xy.dtype,
            )
            delta = _wrap_periodic_delta(delta, lengths, periodic_axes)
        dist2 = delta.square().sum(dim=-1)
        context_scale = self.config.local_context_scale
        if context_scale is None:
            context_scale = self.config.module_radius
        sigma2 = max(float(context_scale) ** 2, EPS)
        weights = torch.exp(-dist2 / (2.0 * sigma2)) * module_present[:, None, :]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return torch.einsum("bqm,bmh->bqh", weights, module_tokens)

    def _uses_global(self) -> bool:
        """Return the global-context decision owned by ``decoder_mode``."""

        return self.config.decoder_uses("global")

    def _uses_direct(self) -> bool:
        """Return the direct-memory decision owned by ``decoder_mode``."""

        return self.config.decoder_uses("direct")

    def _uses_near_module(self) -> bool:
        """Return the local-neighborhood decision owned by ``decoder_mode``."""

        return self.config.decoder_uses("near")
