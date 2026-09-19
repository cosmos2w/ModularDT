"""Run-1407 phase-shared group-control interface reader.

The controller is built once from contextualized P0 source states.  Later
physical phases retain that live controller graph and refresh only the fine
source-dependent values (QM source affines and QE K/V/value rows).  The
rectangular reader inherited from Run 1406 remains the reference executor;
support selection is intentionally a separate policy seam.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .group_control_pairwise import GroupControlPairwiseField
from .group_control_router import (
    PhaseSharedGroupControl,
    PreparedGroupControl,
    PrototypeAnchoredGroupRouter,
)
from .group_control_support import SixBitSupportIndex, build_six_bit_source_support
from .types import EncodedInterfaceCase


class PhaseSharedGroupControlPairwiseField(GroupControlPairwiseField):
    """Run-1407 group-control field with one live P0 controller."""

    router_class = PrototypeAnchoredGroupRouter
    # The selected policy is independent of map/diagnostic requests. The
    # rectangular executor remains an explicitly named exact reference.
    diagnostic_executor_independent = True
    executor_policy = "hybrid_support"
    ledger_rectangular_rows = True
    # Calibrated on the formal M=12/Q=128-2048 shapes. Selected pair execution
    # becomes beneficial only after removing at least half the padded module
    # rectangle; the environment reader remains on its faster rectangular QE
    # path. The exact rectangular reference is selectable for parity/benchmark.
    module_selected_max_fraction = 0.5
    phase_diagnostic_prefix = "group_control_"

    def _build_phase_shared_state(
        self,
        state: dict[str, Any],
    ) -> PhaseSharedGroupControl:
        controls: PreparedGroupControl = state["group_control_state"]
        if not isinstance(self.router, PrototypeAnchoredGroupRouter):
            raise TypeError("phase-shared backend requires PrototypeAnchoredGroupRouter.")
        return PhaseSharedGroupControl(
            controls=controls,
            module_control_bank=state["module_control_bank"],
            environment_source_group_control=state["environment_source_group_control"],
            environment_head_control=state["environment_head_control"],
            environment_head_source_control=state["environment_head_source_control"],
            normalized_query_keys=self.router.normalized_query_key_bank(controls),
            environment_value_gain=state["environment_value_gain"],
            module_source_support=build_six_bit_source_support(
                controls.module_membership,
                source_measure=controls.module_measure,
            ),
            environment_source_support=build_six_bit_source_support(
                controls.environment_membership,
                source_measure=controls.environment_measure,
            ),
        )

    @staticmethod
    def _validate_phase_shared_state(
        phase_shared_state: PhaseSharedGroupControl,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
    ) -> None:
        if not isinstance(phase_shared_state, PhaseSharedGroupControl):
            raise TypeError("phase_shared_state must be a PhaseSharedGroupControl instance.")
        controls = phase_shared_state.controls
        batch, modules = int(module_states.shape[0]), int(module_states.shape[1])
        if tuple(controls.module_membership.shape[:2]) != (batch, modules):
            raise ValueError("phase_shared module membership does not match current module sources.")
        if int(controls.environment_membership.shape[0]) != batch:
            raise ValueError("phase_shared environment membership batch does not match the phase.")
        if int(phase_shared_state.normalized_query_keys.shape[0]) != batch:
            raise ValueError("phase_shared query keys have an invalid batch shape.")
        if int(phase_shared_state.normalized_query_keys.shape[-1]) != self_control_dim(controls):
            raise ValueError("phase_shared query key width does not match the controller.")
        if int(phase_shared_state.environment_value_gain.shape[0]) != batch:
            raise ValueError("phase_shared environment value gain batch does not match the phase.")
        if int(phase_shared_state.module_source_support.source_masks.shape[0]) != batch:
            raise ValueError("phase_shared module support batch does not match the phase.")
        if int(phase_shared_state.environment_source_support.source_masks.shape[0]) != batch:
            raise ValueError("phase_shared environment support batch does not match the phase.")
        if tuple(encoded.global_token.shape[:1]) != (batch,):
            raise ValueError("phase_shared state batch does not match encoded case.")

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        phase_shared_state: PhaseSharedGroupControl | None = None,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        """Prepare one phase while retaining the P0 controller by identity."""

        dense_encoded = replace(
            encoded,
            coordinate_scale=self._dense_coordinate_scale(
                encoded, int(module_states.shape[0])
            ),
        )
        fine = self.prepare_fine_messages(dense_encoded, module_states)
        global_env = encoded.global_token[:, None, :].expand(-1, fine["env_tokens"].shape[1], -1)
        contextual_env = fine["env_tokens"] + self.env_update(
            torch.cat([fine["env_tokens"], fine["environment_messages"], global_env], dim=-1)
        )

        if phase_shared_state is None:
            controls = self.router.prepare(encoded, fine["module_tokens"], contextual_env)
            state = self._prepare_from_fine(
                encoded,
                module_states,
                fine,
                contextual_env,
                controls,
                return_routing_maps=bool(return_routing_maps),
            )
            shared = self._build_phase_shared_state(state)
            state["phase_shared_group_control"] = shared
            state["group_control_preparation_aux"].update(
                {
                    "phase_shared_group_control_built_from_p0": torch.ones(
                        (int(module_states.shape[0]),),
                        device=module_states.device,
                        dtype=module_states.dtype,
                    ).detach(),
                    "phase_shared_group_control_controller_reused": torch.zeros(
                        (int(module_states.shape[0]),),
                        device=module_states.device,
                        dtype=module_states.dtype,
                    ).detach(),
                }
            )
            return state

        self._validate_phase_shared_state(phase_shared_state, encoded, module_states)
        state = self._prepare_from_fine(
            encoded,
            module_states,
            fine,
            contextual_env,
            phase_shared_state.controls,
            phase_shared=phase_shared_state,
            return_routing_maps=bool(return_routing_maps),
        )
        state["group_control_preparation_aux"].update(
            {
                "phase_shared_group_control_built_from_p0": torch.zeros(
                    (int(module_states.shape[0]),),
                    device=module_states.device,
                    dtype=module_states.dtype,
                ).detach(),
                "phase_shared_group_control_controller_reused": torch.ones(
                    (int(module_states.shape[0]),),
                    device=module_states.device,
                    dtype=module_states.dtype,
                ).detach(),
            }
        )
        return state

    def _module_execution_plan(
        self,
        state: dict[str, Any],
        route: Any,
        overlap: torch.Tensor,
        *,
        source_measure: torch.Tensor,
        include_diagnostics: bool,
    ) -> tuple[bool, torch.Tensor | None, SixBitSupportIndex | None]:
        """Choose the calibrated module executor independently of diagnostics."""

        del include_diagnostics, source_measure
        if self.executor_policy == "rectangular_reference":
            return True, None, None
        if self.executor_policy != "hybrid_support":
            raise ValueError(f"Unsupported phase-shared executor policy: {self.executor_policy!r}.")
        shared = state.get("phase_shared_group_control")
        if not isinstance(shared, PhaseSharedGroupControl):
            raise TypeError("phase-shared module execution requires its live P0 index.")
        support = shared.module_source_support.bind_queries(route.assignment)
        rectangular_rows = int(overlap.numel())
        support_rows = int(support.support_pair_count)
        selected = (
            rectangular_rows > 0
            and support_rows <= int(float(self.module_selected_max_fraction) * rectangular_rows)
        )
        return (not selected), (support.selected_pair_indices() if selected else None), support


def self_control_dim(controls: PreparedGroupControl) -> int:
    """Return the controller width without coupling validation to K/D literals."""

    return int(controls.group_control.shape[-1])


__all__ = ["PhaseSharedGroupControlPairwiseField"]
