"""Non-registering compatibility helpers for additive and gathered execution."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch


EPS = 1e-6


class ResearchDecoderExecutionMixin:
    """Keep optional Stage-1--6 decoder execution out of the primary facade."""

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
