"""Stable decoder facade centered on the Stage-7 context-fusion path.

Checkpoint-visible layers remain registered directly on
``HypergraphFieldDecoder``.  The inherited research mixin contains no module
state and only preserves additive/gathered compatibility execution.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import UnifiedForwardConfig
from .decoding.pairwise import (
    DescriptorFirstMechanismEncoder,
    HyperedgeMechanismEncoder,
    HypergraphGatedPairwiseKernel,
    _routed_module_retention_statistics,
    _wrap_periodic_delta,
    rectangular_boundary_features,
    sparse_topk_softmax,
)
from .decoding.research import ResearchDecoderExecutionMixin
from .nn import FourierFeatures, LazyMLP, MLP
from .routing import locality_bias, normalize_assignment


EPS = 1e-6


class HypergraphFieldDecoder(ResearchDecoderExecutionMixin, nn.Module):
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
        tensor_support = False
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
            # Phase 2 advertises a separate hard mask and soft survival tensor.
            # Keep this route distinct from the Phase-1 edge_survival_weight
            # branch so the established adaptive arithmetic remains unchanged.
            tensor_hard_support = organizer_output.get("hard_case_edge_mask")
            tensor_soft_support = organizer_output.get("edge_survival_soft")
            tensor_support = torch.is_tensor(tensor_hard_support) and torch.is_tensor(tensor_soft_support)
            base_hyper_logits = hyper_logits
            if tensor_support:
                tensor_hard_support = tensor_hard_support.to(
                    device=hyper_logits.device,
                    dtype=hyper_logits.dtype,
                )
                tensor_soft_support = tensor_soft_support.to(
                    device=hyper_logits.device,
                    dtype=hyper_logits.dtype,
                )
                if tensor_hard_support.shape != edge_active_mask.shape:
                    raise ValueError(
                        "hard_case_edge_mask must have shape [B,K] matching the runtime hyperedge axis."
                    )
                if tensor_soft_support.shape != edge_active_mask.shape:
                    raise ValueError(
                        "edge_survival_soft must have shape [B,K] matching the runtime hyperedge axis."
                    )
                edge_active_mask = tensor_hard_support
                hyper_logits = hyper_logits.masked_fill(
                    tensor_hard_support[:, None, :] <= 0,
                    torch.finfo(hyper_logits.dtype).min,
                )
            residual_support = organizer_output.get("edge_survival_weight")
            adaptive_support = torch.is_tensor(residual_support)
            if adaptive_support and not tensor_support:
                residual_support = residual_support.to(
                    device=hyper_logits.device,
                    dtype=hyper_logits.dtype,
                )
                if residual_support.shape != edge_active_mask.shape:
                    raise ValueError(
                        "edge_survival_weight must have shape [B,K] matching the runtime hyperedge axis."
                    )
                # The residual organizer exports either differentiable soft
                # support during training or a detached 0/1 hard mask during
                # evaluation.  Apply it as a log prior before normalization;
                # masking again below makes hard-inactive attention exactly
                # zero instead of merely underflowing to a tiny probability.
                residual_support_mask = residual_support > 0
                hyper_logits = hyper_logits + torch.log(residual_support.clamp_min(EPS))[:, None, :]
                hyper_logits = hyper_logits.masked_fill(
                    ~residual_support_mask[:, None, :],
                    torch.finfo(hyper_logits.dtype).min,
                )
            uses_descriptive_query_normalizer = (
                tensor_support
                or adaptive_support
                or not context_fusion
                or cfg.query_assignment_normalizer != "softmax"
            )
            if uses_descriptive_query_normalizer:
                hyper_logits = hyper_logits.masked_fill(
                    edge_active_mask[:, None, :] <= 0,
                    torch.finfo(hyper_logits.dtype).min,
                )
            if cfg.hyper_query_attention_mode == "uniform":
                if not uses_descriptive_query_normalizer:
                    hyper_attention = torch.full_like(hyper_logits, 1.0 / float(max(hyper_logits.shape[-1], 1)))
                elif adaptive_support and not tensor_support:
                    hyper_attention = residual_support[:, None, :].expand_as(hyper_logits)
                    hyper_attention = hyper_attention / hyper_attention.sum(dim=-1, keepdim=True).clamp_min(EPS)
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
            if tensor_support:
                # ``hyper_attention`` above is the hard masked route.  Build a
                # second soft route from the unmasked logits, then blend with
                # stop-gradient hard values.  The resulting forward tensor is
                # exactly hard-supported in both train and eval.
                hard_attention = hyper_attention
                soft_logits = base_hyper_logits + torch.log(
                    tensor_soft_support.clamp_min(EPS)
                )[:, None, :]
                soft_mask = tensor_soft_support[:, None, :] > 0
                if cfg.hyper_query_attention_mode == "uniform":
                    soft_attention = tensor_soft_support[:, None, :].expand_as(soft_logits)
                    soft_attention = soft_attention / soft_attention.sum(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(EPS)
                else:
                    soft_attention = normalize_assignment(
                        soft_logits / max(float(cfg.hyper_attention_temperature), EPS),
                        mode=cfg.query_assignment_normalizer,
                        mask=soft_mask,
                        entmax_blend=float(
                            organizer_output.get(
                                "query_sparsity_fraction",
                                soft_logits.new_zeros(()),
                            )
                        ),
                    )
                    soft_attention = self._limit_probability_routes(soft_attention)
                hyper_attention = soft_attention + (hard_attention - soft_attention).detach()
                diagnostics["case_adaptive_support_mean"] = tensor_soft_support.detach().mean()
                diagnostics["case_adaptive_active_edge_count"] = tensor_hard_support.detach().sum(dim=-1).mean()
                soft_count = organizer_output.get("case_adaptive_soft_edge_count")
                if torch.is_tensor(soft_count):
                    diagnostics["case_adaptive_soft_edge_count"] = soft_count.detach().to(
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    )
                else:
                    diagnostics["case_adaptive_soft_edge_count"] = tensor_soft_support.detach().sum(dim=-1)
            elif adaptive_support:
                # Only the boolean support is applied a second time; the
                # fractional training weight was already used as the log
                # routing prior above.
                hyper_attention = hyper_attention * residual_support_mask[:, None, :].to(
                    dtype=hyper_attention.dtype
                )
                hyper_attention = hyper_attention / hyper_attention.sum(dim=-1, keepdim=True).clamp_min(EPS)
                diagnostics["case_adaptive_support_mean"] = residual_support.detach().mean()
                hard_support = organizer_output.get("hard_case_edge_mask")
                if not torch.is_tensor(hard_support):
                    hard_support = residual_support.detach() > 0
                diagnostics["case_adaptive_active_edge_count"] = (
                    (hard_support.to(
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    ).sum(dim=-1).mean())
                )
                # Preserve the organizer's per-case soft count in the merged
                # model output.  During evaluation ``edge_survival_weight`` is
                # intentionally hard, so deriving this diagnostic from that
                # tensor would silently report K_hard instead of the true
                # soft-support effective count.
                soft_count = organizer_output.get("case_adaptive_soft_edge_count")
                if torch.is_tensor(soft_count):
                    diagnostics["case_adaptive_soft_edge_count"] = soft_count.detach().to(
                        device=hyper_logits.device,
                        dtype=hyper_logits.dtype,
                    )
                else:
                    diagnostics["case_adaptive_soft_edge_count"] = residual_support.detach().sum(dim=-1)
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
            if return_routing_maps and not tensor_support:
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
            strength=(
                self.config.environment_locality_strength
                if self.config.query_locality_strength is None
                else self.config.query_locality_strength
            ),
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
