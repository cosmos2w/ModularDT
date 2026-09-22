"""Run-1409 occupancy-adaptive geometric group-control reader."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .group_control_pairwise import GroupControlPairwiseField
from .group_control_router import GroupQueryRoute, PreparedGroupControl
from .occupancy_group_router import (
    OccupancyAdaptiveGroupRouter,
    direct_mask_intersection,
    pack_16bit_mask,
)
from .types import EncodedInterfaceCase


def _plan_capacity(plan: Any) -> int:
    value = getattr(
        plan,
        "registered_capacity",
        getattr(plan, "registered_kmax", getattr(plan, "kmax", None)),
    )
    if value is None:
        raise TypeError("occupancy plan must expose registered_capacity or kmax.")
    return int(value)


def _plan_active_mask(plan: Any, *, device: torch.device) -> torch.Tensor:
    direct = getattr(plan, "active_mask", None)
    if torch.is_tensor(direct):
        return direct.to(device=device, dtype=torch.bool)
    prototype_ids = getattr(plan, "prototype_ids", None)
    packed_valid = getattr(plan, "packed_valid", None)
    if not torch.is_tensor(prototype_ids) or not torch.is_tensor(packed_valid):
        raise TypeError("occupancy plan must expose active_mask or prototype_ids/packed_valid.")
    mask = torch.zeros(
        (int(prototype_ids.shape[0]), _plan_capacity(plan)),
        device=device,
        dtype=torch.bool,
    )
    ids = prototype_ids.to(device=device, dtype=torch.long)
    valid = packed_valid.to(device=device, dtype=torch.bool) & (ids >= 0)
    if bool(valid.any()):
        batch = torch.arange(int(ids.shape[0]), device=device)[:, None].expand_as(ids)
        mask[batch[valid], ids[valid]] = True
    return mask


class OccupancyAdaptiveGroupControlPairwiseField(GroupControlPairwiseField):
    """Dense Run-1406 preparation with a P0 occupancy plan and Kmax=12."""

    router_class = OccupancyAdaptiveGroupRouter
    phase_interaction_diagnostics = True
    phase_diagnostic_prefix = ("occupancy_group_", "group_control_")
    # Run 1409 keeps exact rectangular readers while the representation is
    # evaluated. Logical support is still reported when maps are requested.
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
        execution_mode: str = "full_width",
    ) -> None:
        if int(group_count) != 12:
            raise ValueError("occupancy_adaptive_group_control_honf requires group_count=12.")
        if int(group_control_dim) != 16:
            raise ValueError("occupancy_adaptive_group_control_honf requires group_control_dim=16.")
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
        self.set_execution_mode(execution_mode)

    def set_execution_mode(self, mode: str) -> None:
        """Choose registered-width or original-ID packed group columns."""

        normalized = {"compact": "packed", "full": "full_width"}.get(str(mode), str(mode))
        if normalized not in {"full_width", "packed"}:
            raise ValueError("occupancy execution_mode must be 'full_width' or 'packed'.")
        self.execution_mode = normalized

    def _prepare_occupancy_state(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        phase_shared_state: Any,
        return_routing_maps: bool,
    ) -> dict[str, Any]:
        dense_encoded = replace(
            encoded,
            coordinate_scale=self._dense_coordinate_scale(encoded, int(module_states.shape[0])),
        )
        fine = self.prepare_fine_messages(dense_encoded, module_states)
        global_env = encoded.global_token[:, None, :].expand(-1, fine["env_tokens"].shape[1], -1)
        contextual_env = fine["env_tokens"] + self.env_update(
            torch.cat([fine["env_tokens"], fine["environment_messages"], global_env], dim=-1)
        )
        if not isinstance(self.router, OccupancyAdaptiveGroupRouter):
            raise TypeError("occupancy backend requires OccupancyAdaptiveGroupRouter.")
        controls, plan, runtime = self.router.prepare_occupancy(
            encoded,
            fine["module_tokens"],
            contextual_env,
            plan=phase_shared_state,
            compact=self.execution_mode == "packed",
        )
        state = self._prepare_from_fine(
            encoded,
            module_states,
            fine,
            contextual_env,
            controls,
            return_routing_maps=bool(return_routing_maps),
        )
        # The plan is the only P0 state handed to P1/P2. Current phase
        # controls, centres, active source occupancy, and kappa are rebuilt.
        state["occupancy_group_plan"] = plan
        state["phase_shared_group_control"] = plan
        state["occupancy_group_runtime"] = runtime
        state["occupancy_group_kappa"] = runtime["kappa"]
        state["occupancy_group_active_mask"] = runtime["active_mask"]
        state["occupancy_group_module_centres"] = runtime["module_centres"]
        state["occupancy_group_environment_centres"] = runtime["environment_centres"]
        state["occupancy_group_joint_centres"] = runtime["joint_centres"]
        aux = state["group_control_preparation_aux"]
        aux.update(
            {
                "occupancy_group_registered_capacity": module_states.new_full(
                    (int(module_states.shape[0]),), float(self.group_count)
                ).detach(),
                "occupancy_group_k_plan_per_case": plan.k_plan.to(module_states.dtype).detach(),
                "occupancy_group_kappa": runtime["kappa"].detach(),
                "occupancy_group_module_mass": controls.module_mass.detach(),
                "occupancy_group_environment_mass": controls.environment_mass.detach(),
                "occupancy_group_proposal_module_mass": runtime[
                    "proposal_module_mass"
                ].detach(),
                "occupancy_group_proposal_environment_mass": runtime[
                    "proposal_environment_mass"
                ].detach(),
                "occupancy_group_proposal_occupied": runtime[
                    "proposal_occupied"
                ].detach(),
                "occupancy_group_module_centres": runtime["module_centres"].detach(),
                "occupancy_group_environment_centres": runtime["environment_centres"].detach(),
                "occupancy_group_joint_centres": runtime["joint_centres"].detach(),
            }
        )
        if return_routing_maps:
            aux.update(
                {
                    "occupancy_group_plan_active_mask": _plan_active_mask(
                        plan, device=module_states.device
                    ).detach(),
                    "occupancy_group_plan_prototype_ids": plan.prototype_ids.detach(),
                    "occupancy_group_plan_packed_valid": plan.packed_valid.detach(),
                }
            )
        return state

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        phase_shared_state: Any = None,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        if phase_shared_state is not None:
            _plan_capacity(phase_shared_state)
        return self._prepare_occupancy_state(
            encoded,
            module_states,
            phase_shared_state=phase_shared_state,
            return_routing_maps=bool(return_routing_maps),
        )

    def preparation_aux(
        self,
        state: dict[str, Any],
        *,
        include_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        summary = super().preparation_aux(state, include_diagnostics=include_diagnostics)
        if include_diagnostics:
            controls: PreparedGroupControl = state["group_control_state"]
            plan: Any = state["occupancy_group_plan"]
            summary.update(
                {
                    "occupancy_group_plan_active_mask": _plan_active_mask(
                        plan, device=state["occupancy_group_kappa"].device
                    ).detach(),
                    "occupancy_group_plan_prototype_ids": plan.prototype_ids.detach(),
                    "occupancy_group_plan_packed_valid": plan.packed_valid.detach(),
                    "occupancy_group_module_incidence": controls.module_membership.detach(),
                    "occupancy_group_environment_incidence": controls.environment_membership.detach(),
                    "occupancy_group_query_plan_capacity": state["occupancy_group_kappa"].new_full(
                        (int(state["occupancy_group_kappa"].shape[0]),), float(self.group_count)
                    ),
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
        if not isinstance(self.router, OccupancyAdaptiveGroupRouter):
            raise TypeError("occupancy backend requires OccupancyAdaptiveGroupRouter.")
        return self.router.route_queries(
            encoded,
            state["group_control_state"],
            receivers,
            receiver_features,
            plan=state["occupancy_group_plan"],
            module_centres=state["occupancy_group_module_centres"],
            environment_centres=state["occupancy_group_environment_centres"],
            active_mask=state["occupancy_group_active_mask"],
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
        ratio = state["occupancy_group_kappa"].to(
            device=context.device,
            dtype=context.dtype,
        ) / float(self.group_count)
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
        ratio = state["occupancy_group_kappa"].to(
            device=context.device,
            dtype=context.dtype,
        ) / float(self.group_count)
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
        """Run the inherited rectangular reader and expose exact bit support."""

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
                    "occupancy_group_query_mask": query_mask.detach(),
                    "occupancy_group_module_source_mask": module_mask.detach(),
                    "occupancy_group_environment_source_mask": environment_mask.detach(),
                    "occupancy_group_module_mask_support": direct_mask_intersection(
                        query_mask, module_mask
                    ).detach(),
                    "occupancy_group_environment_mask_support": direct_mask_intersection(
                        query_mask, environment_mask
                    ).detach(),
                }
            )
        return context, aux


OccupancyGroupControlPairwiseField = OccupancyAdaptiveGroupControlPairwiseField

__all__ = [
    "OccupancyAdaptiveGroupControlPairwiseField",
    "OccupancyGroupControlPairwiseField",
]
