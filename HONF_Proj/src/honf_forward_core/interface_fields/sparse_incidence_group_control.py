"""Run-1501 phase-local sparse-incidence group-control field."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .group_control_pairwise import GroupControlPairwiseField
from .group_control_router import GroupQueryRoute
from .occupancy_group_router import direct_mask_intersection, pack_16bit_mask
from .sparse_incidence_router import (
    SparseIncidenceGroupRouter,
    SparseIncidencePreparedGroupControl,
)
from .types import EncodedInterfaceCase


class SparseIncidenceGroupControlPairwiseField(GroupControlPairwiseField):
    """Rectangular Run-1406 reader with phase-local Run-1501 routing."""

    router_class = SparseIncidenceGroupRouter
    phase_interaction_diagnostics = True
    phase_diagnostic_prefix = ("sparse_incidence_", "group_control_")
    diagnostic_executor_independent = True
    executor_policy = "rectangular_reference"
    ledger_rectangular_rows = True

    def __init__(
        self,
        hidden_dim: int,
        message_hidden_dim: int,
        num_heads: int,
        fourier_frequencies: int,
        *,
        group_count: int = 12,
        group_control_dim: int = 16,
        spatial_dim: int = 2,
        module_temperature: float = 1.0,
        environment_temperature: float = 1.0,
        query_temperature: float = 1.0,
        activation_checkpointing: bool = False,
        query_tile_size: int = 128,
        source_tile_size: int = 128,
    ) -> None:
        if int(group_count) != 12:
            raise ValueError(
                "sparse_incidence_group_control_honf requires group_count=12."
            )
        if int(group_control_dim) != 16:
            raise ValueError(
                "sparse_incidence_group_control_honf requires group_control_dim=16."
            )
        super().__init__(
            hidden_dim,
            message_hidden_dim,
            num_heads,
            fourier_frequencies,
            group_count=group_count,
            group_control_dim=group_control_dim,
            spatial_dim=spatial_dim,
            module_temperature=module_temperature,
            environment_temperature=environment_temperature,
            query_temperature=query_temperature,
            activation_checkpointing=activation_checkpointing,
            query_tile_size=query_tile_size,
            source_tile_size=source_tile_size,
        )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        """Rebuild Dense preparation and all routing state for this phase."""

        dense_encoded = replace(
            encoded,
            coordinate_scale=self._dense_coordinate_scale(
                encoded, int(module_states.shape[0])
            ),
        )
        fine = self.prepare_fine_messages(dense_encoded, module_states)
        global_env = encoded.global_token[:, None, :].expand(
            -1, fine["env_tokens"].shape[1], -1
        )
        contextual_env = fine["env_tokens"] + self.env_update(
            torch.cat(
                [fine["env_tokens"], fine["environment_messages"], global_env],
                dim=-1,
            )
        )
        if not isinstance(self.router, SparseIncidenceGroupRouter):
            raise TypeError(
                "sparse-incidence backend requires SparseIncidenceGroupRouter."
            )
        controls = self.router.prepare(
            encoded, fine["module_tokens"], contextual_env
        )
        state = self._prepare_from_fine(
            encoded,
            module_states,
            fine,
            contextual_env,
            controls,
            return_routing_maps=bool(return_routing_maps),
        )
        state.update(
            {
                "sparse_incidence_kappa": controls.kappa,
                "sparse_incidence_pi": controls.pi,
                "sparse_incidence_active_mask": controls.phase_occupied,
                "sparse_incidence_module_centres": controls.module_centres,
                "sparse_incidence_environment_centres": controls.environment_centres,
                "sparse_incidence_joint_centres": controls.joint_centres,
            }
        )
        aux = state["group_control_preparation_aux"]
        aux.update(
            {
                "sparse_incidence_registered_capacity": module_states.new_full(
                    (int(module_states.shape[0]),), float(self.group_count)
                ).detach(),
                "sparse_incidence_kappa": controls.kappa.detach(),
                "sparse_incidence_pi": controls.pi.detach(),
                "sparse_incidence_phase_occupied": controls.phase_occupied.detach(),
                "sparse_incidence_module_mass": controls.module_mass.detach(),
                "sparse_incidence_environment_mass": controls.environment_mass.detach(),
                "sparse_incidence_proposal_module_mass": controls.proposal_module_mass.detach(),
                "sparse_incidence_proposal_environment_mass": controls.proposal_environment_mass.detach(),
                "sparse_incidence_proposal_occupied": controls.proposal_occupied.detach(),
                "sparse_incidence_module_centres": controls.module_centres.detach(),
                "sparse_incidence_environment_centres": controls.environment_centres.detach(),
                "sparse_incidence_joint_centres": controls.joint_centres.detach(),
            }
        )
        return state

    def preparation_aux(
        self,
        state: dict[str, Any],
        *,
        include_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        summary = super().preparation_aux(
            state, include_diagnostics=include_diagnostics
        )
        if include_diagnostics:
            controls = state["group_control_state"]
            if not isinstance(controls, SparseIncidencePreparedGroupControl):
                raise TypeError(
                    "sparse-incidence diagnostics require phase-local controls."
                )
            summary.update(
                {
                    "sparse_incidence_module_incidence": controls.module_membership.detach(),
                    "sparse_incidence_environment_incidence": controls.environment_membership.detach(),
                    "sparse_incidence_phase_occupied": controls.phase_occupied.detach(),
                    "sparse_incidence_pi": controls.pi.detach(),
                    "sparse_incidence_kappa": controls.kappa.detach(),
                }
            )
        return summary

    def _route(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
    ) -> GroupQueryRoute:
        if not isinstance(self.router, SparseIncidenceGroupRouter):
            raise TypeError(
                "sparse-incidence backend requires SparseIncidenceGroupRouter."
            )
        return self.router.route_queries(
            encoded,
            state["group_control_state"],
            receivers,
            receiver_features,
        )

    def _read_module(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        route: GroupQueryRoute,
        *,
        include_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context, aux = super()._read_module(
            state,
            encoded,
            receivers,
            route,
            include_diagnostics=include_diagnostics,
        )
        controls = state["group_control_state"]
        ratio = controls.kappa.to(device=context.device, dtype=context.dtype) / float(
            self.group_count
        )
        return context * ratio[:, None, None], aux

    def _read_environment(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        route: GroupQueryRoute,
        *,
        include_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context, aux = super()._read_environment(
            state,
            encoded,
            receivers,
            receiver_features,
            route,
            include_diagnostics=include_diagnostics,
        )
        controls = state["group_control_state"]
        ratio = controls.kappa.to(device=context.device, dtype=context.dtype) / float(
            self.group_count
        )
        return context * ratio[:, None, None], aux

    def read(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context, aux = super().read(
            state,
            encoded,
            receivers,
            receiver_features,
            return_routing_maps=bool(return_routing_maps),
        )
        if return_routing_maps:
            route = self._route(state, encoded, receivers, receiver_features)
            controls = state["group_control_state"]
            query_mask = pack_16bit_mask(route.assignment)
            module_mask = pack_16bit_mask(controls.module_membership)
            environment_mask = pack_16bit_mask(controls.environment_membership)
            aux.update(
                {
                    "sparse_incidence_query_routing": route.assignment.detach(),
                    "sparse_incidence_query_logits": route.logits.detach(),
                    "sparse_incidence_query_keys": route.query_keys.detach(),
                    "sparse_incidence_query_mask": query_mask,
                    "sparse_incidence_module_source_mask": module_mask,
                    "sparse_incidence_environment_source_mask": environment_mask,
                    "sparse_incidence_module_mask_support": direct_mask_intersection(
                        query_mask, module_mask
                    ).detach(),
                    "sparse_incidence_environment_mask_support": direct_mask_intersection(
                        query_mask, environment_mask
                    ).detach(),
                }
            )
        return context, aux


__all__ = ["SparseIncidenceGroupControlPairwiseField"]
