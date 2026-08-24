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
        if config.pairwise_kernel_mode == "legacy_mlp":
            # Keep the accepted Run-1000/Run-1401 parameter path byte-for-byte
            # stable. The candidate kernel deliberately registers no pair_mlp.
            self.pair_mlp = LazyMLP(
                hidden_dim=kernel_hidden_dim,
                out_dim=hidden_dim,
                num_layers=int(config.pairwise_kernel_num_layers),
                dropout=float(config.dropout),
            )
        else:
            rank = kernel_hidden_dim
            layers = int(config.pairwise_kernel_num_layers)
            self.factorized_module_encoder = LazyMLP(
                hidden_dim=rank,
                out_dim=rank,
                num_layers=layers,
                dropout=float(config.dropout),
            )
            self.factorized_relative_encoder = LazyMLP(
                hidden_dim=rank,
                out_dim=rank,
                num_layers=layers,
                dropout=float(config.dropout),
            )
            self.factorized_interaction_gate = nn.Linear(rank, rank)
            self.factorized_norm = nn.LayerNorm(rank)
            self.factorized_output = nn.Linear(rank, hidden_dim)
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
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor | str]]:
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
        if cfg.pairwise_aggregation_mode == "fused_query_module":
            return self._forward_fused_query_module(
                query_xy,
                organizer_output,
                hyper_attention,
                module_centers,
                module_tokens,
                module_present,
                edge_module_weight,
                gathered_execution=gathered_execution,
                return_routing_maps=return_routing_maps,
                reduce_pair_context=reduce_pair_context,
            )
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

    def _forward_fused_query_module(
        self,
        query_xy: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
        hyper_attention: torch.Tensor,
        module_centers: torch.Tensor,
        module_tokens: torch.Tensor,
        module_present: torch.Tensor,
        edge_module_weight: torch.Tensor,
        *,
        gathered_execution: bool,
        return_routing_maps: bool,
        reduce_pair_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor | str]]:
        """Fuse query-edge and edge-module routing before pair evaluation.

        ``beta_qm = sum_k alpha_qk A_mk`` is kept unnormalized after pruning.
        Sparse execution materializes neither ``[B,Q,M,H]`` pair embeddings
        nor ``[B,Q,K,H]`` edge contexts. The dense legacy-kernel reference
        evaluates all pairs but never retains a full edge-context tensor.
        """

        batch_size, num_queries = query_xy.shape[:2]
        num_modules = module_present.shape[1]
        hidden_dim = module_tokens.shape[-1]
        beta = torch.einsum("bqk,bmk->bqm", hyper_attention, edge_module_weight)
        active_mask = module_present > 0
        beta = beta * active_mask[:, None, :].to(dtype=beta.dtype)
        if gathered_execution:
            selected_mask = self._retained_beta_mask(beta, active_mask)
        else:
            selected_mask = active_mask[:, None, :].expand(-1, num_queries, -1)
        selected_mask = selected_mask.detach()
        selected = torch.nonzero(selected_mask, as_tuple=False)
        legacy_full_support = (
            self.config.pairwise_kernel_mode == "legacy_mlp"
            and (
                not gathered_execution
                or (
                    float(self.config.query_module_retained_mass_floor) >= 1.0
                    and int(self.config.query_module_limit) <= 0
                )
            )
        )
        if legacy_full_support:
            dense_pair_embed = self._dense_pair_embeddings(
                query_xy,
                module_centers,
                module_tokens,
                module_present,
                organizer_output.get("module_features_raw"),
            )
            pair_context = self._legacy_dense_parity_context(
                hyper_attention,
                edge_module_weight,
                dense_pair_embed,
            )
            pair_embed = dense_pair_embed[selected[:, 0], selected[:, 1], selected[:, 2]]
        else:
            pair_embed = self._selected_pair_embeddings(
                query_xy,
                organizer_output,
                module_centers,
                module_tokens,
                selected,
            )
            flat_query_index = selected[:, 0] * num_queries + selected[:, 1]
            selected_beta = beta[selected[:, 0], selected[:, 1], selected[:, 2]]
            flat_context = query_xy.new_zeros(batch_size * num_queries, hidden_dim)
            if selected.numel() > 0:
                flat_context = flat_context.index_add(
                    0,
                    flat_query_index,
                    selected_beta[:, None] * pair_embed,
                )
            pair_context = flat_context.view(batch_size, num_queries, hidden_dim)
        if not reduce_pair_context:
            pair_context = torch.zeros_like(pair_context)

        selected_counts = selected_mask.sum(dim=-1)
        available_counts = active_mask.sum(dim=-1)[:, None].expand(-1, num_queries)
        retained_beta = (beta * selected_mask.to(dtype=beta.dtype)).sum(dim=-1)
        total_beta = beta.sum(dim=-1)
        retained_fraction = retained_beta / total_beta.clamp_min(EPS)
        floor = float(self.config.query_module_retained_mass_floor)
        violations = retained_beta + 1.0e-7 < min(max(floor, 0.0), 1.0)
        gate = torch.sigmoid(self.pairwise_kernel_logit)
        selected_mean = selected_counts.to(dtype=query_xy.dtype).mean()
        available_mean = available_counts.to(dtype=query_xy.dtype).mean()
        evaluated_pairs = query_xy.new_tensor(float(selected.shape[0]))
        dense_pairs = available_counts.sum().to(dtype=query_xy.dtype)
        zero = query_xy.new_zeros(())
        diagnostics: Dict[str, torch.Tensor | str] = {
            "pairwise_kernel_gate": gate.detach(),
            "pairwise_context_norm": pair_context.detach().norm(dim=-1).mean(),
            "pairwise_edge_context_norm": pair_context.detach().norm(dim=-1).mean(),
            "pairwise_edge_usage_mean": hyper_attention.detach().mean(),
            "pairwise_active_hyperedge_count": (hyper_attention.detach() > 0).float().sum(dim=-1).mean(),
            "pairwise_uses_sparse_hyper_attention": hyper_attention.new_tensor(
                float(
                    self.config.hyper_query_attention_mode != "uniform"
                    and (
                        int(self.config.hyper_attention_topk) > 0
                        or self.config.query_assignment_normalizer == "entmax15"
                    )
                )
            ),
            "pairwise_available_modules": available_mean,
            "pairwise_available_active_modules": available_mean,
            "pairwise_selected_modules": selected_mean,
            "pairwise_selection_ratio": selected_mean / available_mean.clamp_min(1.0),
            "pairwise_evaluated_pair_count": evaluated_pairs,
            "pairwise_dense_route_count": dense_pairs,
            "pairwise_gathered_route_count": evaluated_pairs if gathered_execution else zero,
            "query_module_retained_beta_mass_mean": retained_beta.detach().mean(),
            "query_module_retained_beta_mass_p05": torch.quantile(
                retained_beta.detach().float(), 0.05
            ).to(dtype=query_xy.dtype),
            "query_module_retained_beta_mass_min": retained_beta.detach().amin(),
            "query_module_retained_beta_fraction_mean": retained_fraction.detach().mean(),
            "query_module_retained_beta_floor_violation_fraction": violations.float().mean(),
            "pairwise_aggregation_mode": "fused_query_module",
            "pairwise_kernel_mode": self.config.pairwise_kernel_mode,
            "pairwise_execution_mode": "gathered" if gathered_execution else "dense",
        }

        if return_routing_maps:
            edge_pair_context = self._diagnostic_edge_pair_context(
                pair_embed,
                selected,
                edge_module_weight,
                batch_size=batch_size,
                num_queries=num_queries,
                hidden_dim=hidden_dim,
            )
            diagnostics["query_module_routing_beta"] = beta.detach()
            diagnostics["query_module_selected_mask"] = selected_mask.detach()
            diagnostics["pairwise_edge_contribution"] = (
                gate * hyper_attention[..., None] * edge_pair_context
            ).detach().norm(dim=-1)
        else:
            edge_pair_context = query_xy.new_empty(batch_size, num_queries, 0, hidden_dim)
        return gate * pair_context, gate * edge_pair_context, diagnostics

    @staticmethod
    def _legacy_dense_parity_context(
        hyper_attention: torch.Tensor,
        edge_module_weight: torch.Tensor,
        dense_pair_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve accepted dense numerics without a full edge-context tensor.

        The beta contraction is algebraically identical, but its changed
        floating-point association is amplified by accepted field heads beyond
        the 2e-6 replay gate. Hidden-width blocks retain the historical
        contraction order while bounding the transient edge-local allocation.
        """

        hidden_dim = dense_pair_embed.shape[-1]
        blocks = []
        for hidden_start in range(0, hidden_dim, 32):
            edge_block = torch.einsum(
                "bmk,bqmh->bqkh",
                edge_module_weight,
                dense_pair_embed[..., hidden_start : hidden_start + 32],
            )
            blocks.append(
                torch.einsum("bqk,bqkh->bqh", hyper_attention, edge_block)
            )
        return torch.cat(blocks, dim=-1)

    def _dense_pair_embeddings(
        self,
        query_xy: torch.Tensor,
        module_centers: torch.Tensor,
        module_tokens: torch.Tensor,
        module_present: torch.Tensor,
        raw_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate the accepted legacy pair MLP over the dense module axis."""

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
        return self.pair_mlp(torch.cat(pieces, dim=-1)) * module_present[:, None, :, None]

    def _retained_beta_mask(
        self,
        beta: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select the smallest capped module prefix meeting retained beta mass."""

        num_modules = beta.shape[-1]
        limit = int(self.config.query_module_limit)
        maximum = num_modules if limit <= 0 else min(limit, num_modules)
        if maximum == 0:
            return torch.zeros_like(beta, dtype=torch.bool)
        ranking_scores = beta.masked_fill(~active_mask[:, None, :], float("-inf"))
        ranked_values, ranked_indices = torch.topk(
            ranking_scores,
            k=maximum,
            dim=-1,
            largest=True,
            sorted=True,
        )
        ranked_valid = torch.isfinite(ranked_values)
        ranked_mass = torch.where(ranked_valid, ranked_values, torch.zeros_like(ranked_values))
        floor = float(self.config.query_module_retained_mass_floor)
        if floor <= 0.0:
            selected_count = torch.zeros_like(ranked_mass[..., 0], dtype=torch.long)
        else:
            selected_count = (ranked_mass.cumsum(dim=-1) < floor).sum(dim=-1) + 1
            selected_count = torch.minimum(selected_count, ranked_valid.sum(dim=-1))
        selected_rank = (
            torch.arange(maximum, device=beta.device)
            .view(*([1] * (beta.ndim - 1)), maximum)
            < selected_count[..., None]
        ) & ranked_valid
        mask = torch.zeros_like(beta, dtype=torch.bool)
        mask.scatter_(-1, ranked_indices, selected_rank)
        return mask

    def _selected_pair_embeddings(
        self,
        query_xy: torch.Tensor,
        organizer_output: Dict[str, torch.Tensor],
        module_centers: torch.Tensor,
        module_tokens: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate only selected active query-module pairs."""

        if selected.numel() == 0:
            return query_xy.new_empty(0, module_tokens.shape[-1])
        batch_index, query_index, module_index = selected.unbind(dim=-1)
        relative = self._flat_relative_features(
            query_xy[batch_index, query_index],
            module_centers[batch_index, module_index],
        )
        relative_encoded = (
            self.relative_fourier(relative)
            if self.config.pairwise_kernel_use_fourier
            else relative
        )
        if self.config.pairwise_kernel_mode == "legacy_mlp":
            pieces = [relative_encoded, relative.new_ones(relative.shape[0], 1)]
            if self.config.pairwise_kernel_include_module_token:
                pieces.append(module_tokens[batch_index, module_index])
            raw_features = organizer_output.get("module_features_raw")
            if self.config.pairwise_kernel_include_module_features and torch.is_tensor(raw_features):
                pieces.append(
                    raw_features[batch_index, module_index].to(
                        device=query_xy.device,
                        dtype=query_xy.dtype,
                    )
                )
            return self.pair_mlp(torch.cat(pieces, dim=-1))

        module_codes = self.prepare_module_codes(organizer_output)
        module_code = module_codes[batch_index, module_index]
        relative_code = self.factorized_relative_encoder(relative_encoded)
        interaction = module_code * relative_code
        interaction_gate = torch.sigmoid(
            self.factorized_interaction_gate(module_code + relative_code)
        )
        return self.factorized_output(
            self.factorized_norm(
                module_code + relative_code + interaction_gate * interaction
            )
        )

    def prepare_module_codes(
        self,
        organizer_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode active modules once and cache the runtime tensor in prepared state."""

        cache_key = "_runtime_pairwise_factorized_module_codes"
        cached = organizer_output.get(cache_key)
        if torch.is_tensor(cached):
            return cached
        module_tokens = organizer_output["module_tokens"]
        module_present = organizer_output["module_present"] > 0
        selected = torch.nonzero(module_present, as_tuple=False)
        pieces = [module_tokens.new_ones(selected.shape[0], 1)]
        if self.config.pairwise_kernel_include_module_token:
            pieces.append(module_tokens[selected[:, 0], selected[:, 1]])
        raw_features = organizer_output.get("module_features_raw")
        if self.config.pairwise_kernel_include_module_features and torch.is_tensor(raw_features):
            pieces.append(
                raw_features[selected[:, 0], selected[:, 1]].to(
                    device=module_tokens.device,
                    dtype=module_tokens.dtype,
                )
            )
        encoded = self.factorized_module_encoder(torch.cat(pieces, dim=-1))
        flat_index = selected[:, 0] * module_tokens.shape[1] + selected[:, 1]
        flat_codes = module_tokens.new_zeros(
            module_tokens.shape[0] * module_tokens.shape[1],
            encoded.shape[-1],
        )
        flat_codes = flat_codes.index_copy(0, flat_index, encoded)
        module_codes = flat_codes.view(module_tokens.shape[0], module_tokens.shape[1], -1)
        organizer_output[cache_key] = module_codes
        return module_codes

    def _diagnostic_edge_pair_context(
        self,
        pair_embed: torch.Tensor,
        selected: torch.Tensor,
        edge_module_weight: torch.Tensor,
        *,
        batch_size: int,
        num_queries: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        """Reconstruct edge-local maps only for explicit diagnostic output."""

        num_edges = edge_module_weight.shape[-1]
        flat = pair_embed.new_zeros(batch_size * num_queries * num_edges, hidden_dim)
        if selected.numel() > 0:
            batch_index, query_index, module_index = selected.unbind(dim=-1)
            contributions = (
                edge_module_weight[batch_index, module_index, :, None]
                * pair_embed[:, None, :]
            )
            edge_index = torch.arange(num_edges, device=selected.device)[None, :]
            flat_index = (
                (batch_index[:, None] * num_queries + query_index[:, None]) * num_edges
                + edge_index
            )
            flat = flat.index_add(
                0,
                flat_index.reshape(-1),
                contributions.reshape(-1, hidden_dim),
            )
        return flat.view(batch_size, num_queries, num_edges, hidden_dim)

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

    def _flat_relative_features(
        self,
        query_xy: torch.Tensor,
        module_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized geometry for aligned flat query-module pairs."""

        cfg = self.config
        scale_x, scale_y = cfg.spatial_scale()
        lx = max(scale_x, EPS)
        ly = max(scale_y, EPS)
        diag = max(math.sqrt(lx * lx + ly * ly), EPS)
        delta = query_xy - module_centers
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
