"""Run-1500 mass-competitive group-control field.

Only the case-level router differs from the historical occupancy backend.  The
Dense MM/ME/EM preparation, Run-1406 fine QM/QE readers, three-term context,
rectangular formation executor, and direct 16-bit support accounting are
reused unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from .mass_competitive_router import (
    MassCompetitiveGroupPlan,
    MassCompetitiveGroupRouter,
    MassCompetitivePreparedGroupControl,
)
from .occupancy_group_control import OccupancyAdaptiveGroupControlPairwiseField
from .types import EncodedInterfaceCase


class MassCompetitiveGroupControlPairwiseField(
    OccupancyAdaptiveGroupControlPairwiseField
):
    """Exact rectangular Run-1406 reader with a Run-1500 P0 gamma plan."""

    router_class = MassCompetitiveGroupRouter
    phase_diagnostic_prefix = ("mass_competitive_", "group_control_")
    diagnostic_executor_independent = True
    executor_policy = "rectangular_reference"
    ledger_rectangular_rows = True

    def _prepare_occupancy_state(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        phase_shared_state: Any,
        return_routing_maps: bool,
    ) -> dict[str, Any]:
        if phase_shared_state is not None and not isinstance(
            phase_shared_state, MassCompetitiveGroupPlan
        ):
            raise TypeError("phase_shared_state must be a MassCompetitiveGroupPlan.")
        state = super()._prepare_occupancy_state(
            encoded,
            module_states,
            phase_shared_state=phase_shared_state,
            return_routing_maps=bool(return_routing_maps),
        )
        controls = state["group_control_state"]
        plan = state["occupancy_group_plan"]
        runtime = state["occupancy_group_runtime"]
        if not isinstance(controls, MassCompetitivePreparedGroupControl):
            raise TypeError("mass-competitive backend returned the wrong control state.")
        if not isinstance(plan, MassCompetitiveGroupPlan):
            raise TypeError("mass-competitive backend returned the wrong P0 plan.")
        state.update(
            {
                "mass_competitive_plan": plan,
                "mass_competitive_runtime": runtime,
                "mass_competitive_gamma": runtime["gamma"],
                "mass_competitive_pi": runtime["pi"],
                "mass_competitive_kappa": runtime["kappa"],
                "mass_competitive_active_mask": runtime["active_mask"],
                "mass_competitive_module_centres": runtime["module_centres"],
                "mass_competitive_environment_centres": runtime[
                    "environment_centres"
                ],
                "mass_competitive_joint_centres": runtime["joint_centres"],
            }
        )
        aux = state["group_control_preparation_aux"]
        aux.update(
            {
                "mass_competitive_registered_capacity": module_states.new_full(
                    (int(module_states.shape[0]),), float(self.group_count)
                ).detach(),
                "mass_competitive_k_case_per_case": plan.k_case.to(
                    module_states.dtype
                ).detach(),
                "mass_competitive_gamma": runtime["gamma"].detach(),
                "mass_competitive_pi": runtime["pi"].detach(),
                "mass_competitive_kappa": runtime["kappa"].detach(),
                "mass_competitive_module_mass": controls.module_mass.detach(),
                "mass_competitive_environment_mass": controls.environment_mass.detach(),
                "mass_competitive_precompetition_module_mass": runtime[
                    "precompetition_module_mass"
                ].detach(),
                "mass_competitive_precompetition_environment_mass": runtime[
                    "precompetition_environment_mass"
                ].detach(),
                "mass_competitive_module_centres": runtime[
                    "module_centres"
                ].detach(),
                "mass_competitive_environment_centres": runtime[
                    "environment_centres"
                ].detach(),
                "mass_competitive_joint_centres": runtime["joint_centres"].detach(),
            }
        )
        if return_routing_maps:
            aux.update(
                {
                    "mass_competitive_plan_active_mask": plan.active_mask.detach(),
                    "mass_competitive_plan_prototype_ids": plan.prototype_ids.detach(),
                    "mass_competitive_plan_packed_valid": plan.packed_valid.detach(),
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
            state,
            include_diagnostics=include_diagnostics,
        )
        if include_diagnostics:
            controls = state["group_control_state"]
            plan = state["mass_competitive_plan"]
            if not isinstance(controls, MassCompetitivePreparedGroupControl):
                raise TypeError("mass-competitive diagnostics require their control state.")
            summary.update(
                {
                    "mass_competitive_plan_active_mask": plan.active_mask.detach(),
                    "mass_competitive_plan_prototype_ids": plan.prototype_ids.detach(),
                    "mass_competitive_plan_packed_valid": plan.packed_valid.detach(),
                    "mass_competitive_gamma": controls.gamma.detach(),
                    "mass_competitive_pi": controls.pi_precompetition.detach(),
                    "mass_competitive_kappa": controls.kappa.detach(),
                    "mass_competitive_module_incidence": controls.module_membership.detach(),
                    "mass_competitive_environment_incidence": controls.environment_membership.detach(),
                }
            )
        return summary

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
            aliases = {
                "mass_competitive_query_mask": "occupancy_group_query_mask",
                "mass_competitive_module_source_mask": "occupancy_group_module_source_mask",
                "mass_competitive_environment_source_mask": "occupancy_group_environment_source_mask",
                "mass_competitive_module_mask_support": "occupancy_group_module_mask_support",
                "mass_competitive_environment_mask_support": "occupancy_group_environment_mask_support",
            }
            aux.update(
                {
                    target: aux[source]
                    for target, source in aliases.items()
                    if source in aux
                }
            )
        return context, aux


__all__ = ["MassCompetitiveGroupControlPairwiseField"]
