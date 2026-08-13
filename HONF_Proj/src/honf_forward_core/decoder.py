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


EPS = 1e-6


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
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        rel = self._relative_features(query_xy, module_centers)
        if cfg.pairwise_kernel_use_fourier:
            rel_encoded = self.relative_fourier(rel)
        else:
            rel_encoded = rel
        pieces = [rel_encoded, module_present[:, None, :, None].expand(-1, query_xy.shape[1], -1, -1)]
        if cfg.pairwise_kernel_include_module_token:
            pieces.append(module_tokens[:, None, :, :].expand(-1, query_xy.shape[1], -1, -1))
        if cfg.pairwise_kernel_include_module_features:
            raw_features = organizer_output.get("module_features_raw")
            if torch.is_tensor(raw_features):
                pieces.append(raw_features[:, None, :, :].to(device=query_xy.device, dtype=query_xy.dtype).expand(-1, query_xy.shape[1], -1, -1))

        pair_input = torch.cat(pieces, dim=-1)
        pair_embed = self.pair_mlp(pair_input) * module_present[:, None, :, None]
        if cfg.pairwise_kernel_normalize_by_edge_mass:
            edge_module_weight = A_mh / A_mh.sum(dim=1, keepdim=True).clamp_min(EPS)
        else:
            edge_module_weight = A_mh
        edge_pair_context = torch.einsum("bmk,bqmh->bqkh", edge_module_weight, pair_embed)
        pair_context = torch.einsum("bqk,bqkh->bqh", hyper_attention, edge_pair_context)
        gate = torch.sigmoid(self.pairwise_kernel_logit)
        diagnostics = {
            "pairwise_kernel_gate": gate.detach(),
            "pairwise_context_norm": pair_context.detach().norm(dim=-1).mean(),
            "pairwise_edge_context_norm": edge_pair_context.detach().norm(dim=-1).mean(),
            "pairwise_edge_usage_mean": hyper_attention.detach().mean(),
            "pairwise_active_hyperedge_count": (hyper_attention.detach() > 0).float().sum(dim=-1).mean(),
            "pairwise_uses_sparse_hyper_attention": hyper_attention.new_tensor(
                float(int(cfg.hyper_attention_topk) > 0 and cfg.hyper_query_attention_mode != "uniform")
            ),
        }
        if return_routing_maps:
            # CORE HONF diagnostic: this dense [B,Q,K] tensor is only materialized
            # for explicit evaluation-time routing maps, never during normal train.
            diagnostics["pairwise_edge_contribution"] = (
                gate * hyper_attention[..., None] * edge_pair_context
            ).detach().norm(dim=-1)
        return gate * pair_context, diagnostics

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
        self.mechanism_encoder = HyperedgeMechanismEncoder(config) if config.use_hyper_mechanism_encoder else None
        self.nonhyper_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.direct_key = nn.Linear(hidden_dim, hidden_dim)
        self.direct_value = nn.Linear(hidden_dim, hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        self.near_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity()
        gate_init = min(max(float(config.direct_residual_gate_init), 1e-4), 1.0 - 1e-4)
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.direct_residual_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
        self.pairwise_kernel = (
            HypergraphGatedPairwiseKernel(config) if config.decoder_uses("pairwise") else None
        )

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

    def forward(
        self,
        query_xy: torch.Tensor,
        query_time: Optional[torch.Tensor],
        organizer_output: Dict[str, torch.Tensor],
        global_context: Optional[torch.Tensor],
        query_features: Optional[torch.Tensor] = None,
        *,
        return_routing_maps: bool = False,
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

        uses_hyper = cfg.decoder_uses("hyper")
        uses_hyper_value = bool(cfg.use_hyper_value_context)
        uses_global = self._uses_global()
        uses_direct = self._uses_direct()
        uses_near = self._uses_near_module()
        uses_pairwise = bool(cfg.decoder_uses("pairwise") and self.pairwise_kernel is not None)
        hyper_context = torch.zeros_like(query_state)
        nonhyper_context = torch.zeros_like(query_state)
        diagnostics: Dict[str, torch.Tensor | str] = {"decoder_mode": cfg.decoder_mode}
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
            if cfg.hyper_query_attention_mode == "uniform":
                hyper_attention = torch.full_like(hyper_logits, 1.0 / float(max(hyper_logits.shape[-1], 1)))
            else:
                hyper_attention = sparse_topk_softmax(
                    hyper_logits,
                    topk=int(cfg.hyper_attention_topk),
                    temperature=float(cfg.hyper_attention_temperature),
                    detach_mask=bool(cfg.sparse_hyper_attention_detach_mask),
                )
            hyper_value_context = torch.einsum("bqk,bkh->bqh", hyper_attention, self.hyper_value(hyper_state))
            if uses_hyper_value:
                hyper_context = hyper_value_context
            else:
                hyper_context = torch.zeros_like(hyper_value_context)
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
            diagnostics["hyper_geometry_bias_mean"] = geometry_bias.detach().mean()
            diagnostics["hyper_geometry_bias_std"] = geometry_bias.detach().std(unbiased=False)
            if uses_pairwise:
                pair_context, pair_diagnostics = self.pairwise_kernel(
                    query_xy,
                    organizer_output,
                    hyper_attention,
                    return_routing_maps=return_routing_maps,
                )
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

        context = self.context_norm(context)
        output: Dict[str, torch.Tensor | str] = dict(diagnostics)
        if cfg.output_mean_residual_split:
            pred_mean = self.mean_head(context)
            pred_residual = self.residual_head(context)
            output["pred_mean"] = pred_mean
            output["pred_residual"] = pred_residual
            output["pred_field"] = pred_mean + pred_residual
        else:
            output["pred_field"] = self.pred_head(context)
        return output  # type: ignore[return-value]

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
