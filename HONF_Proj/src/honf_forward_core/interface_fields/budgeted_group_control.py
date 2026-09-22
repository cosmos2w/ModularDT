"""Opt-in case-budgeted Run-1406 group-control reader.

The backend keeps the maintained Dense MM/ME/EM preparation and the single
fine module/environment readers from Run 1406.  It changes the controller in
three explicit ways: a case-level hard-concrete availability plan, gated
source/query entmax assignments, and the gate-reference ``kappa`` amplitude
scale.  The gate plan is shared by physical phases; source memberships,
collective controls and fine projected values are rebuilt by ``prepare`` for
each phase.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from honf_forward_core.routing import entmax15

from .case_group_budget import CaseGroupBudget, CaseGroupGate
from .group_control_pairwise import GroupControlPairwiseField
from .group_control_router import GroupQueryRoute, PreparedGroupControl
from .types import EncodedInterfaceCase

from .group_control_router import LowDimensionalGroupRouter


class BudgetedGroupRouter(LowDimensionalGroupRouter):
    """Low-dimensional Run-1406 router with case-conditioned availability."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        group_count: int = 12,
        control_dim: int = 16,
        fourier_frequencies: int = 4,
        spatial_dim: int = 2,
        module_temperature: float = 1.0,
        environment_temperature: float = 1.0,
        query_temperature: float = 1.0,
        gate_hidden_dim: int = 32,
        hard_concrete_temperature: float = 2.0 / 3.0,
        stretch_lower: float = -0.1,
        stretch_upper: float = 1.1,
        initial_optional_open_probability: float = 0.95,
        always_available_group: int = 0,
    ) -> None:
        super().__init__(
            hidden_dim,
            group_count=group_count,
            control_dim=control_dim,
            fourier_frequencies=fourier_frequencies,
            spatial_dim=spatial_dim,
            module_temperature=module_temperature,
            environment_temperature=environment_temperature,
            query_temperature=query_temperature,
        )
        self.case_gate = CaseGroupGate(
            control_dim,
            group_count,
            hidden_dim=gate_hidden_dim,
            temperature=hard_concrete_temperature,
            stretch_lower=stretch_lower,
            stretch_upper=stretch_upper,
            initial_optional_open_probability=initial_optional_open_probability,
            always_available_group=always_available_group,
        )

    def _source_controls(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Compute the same D-wide controls used by assignments and gates."""

        if module_states.ndim != 3 or environment_states.ndim != 3:
            raise ValueError("module_states and environment_states must have shape [B,S,H].")
        batch = int(module_states.shape[0])
        if int(environment_states.shape[0]) != batch:
            raise ValueError("module_states and environment_states must share their batch dimension.")
        if int(module_states.shape[-1]) != self.hidden_dim or int(environment_states.shape[-1]) != self.hidden_dim:
            raise ValueError("Source state width does not match the low-dimensional router.")
        if encoded.global_token.shape != (batch, self.hidden_dim):
            raise ValueError("encoded.global_token must have shape [B,H].")
        if tuple(encoded.module_centers.shape[:2]) != tuple(module_states.shape[:2]):
            raise ValueError("module_states and encoded.module_centers must align.")
        if tuple(encoded.env_coords.shape[:2]) != tuple(environment_states.shape[:2]):
            raise ValueError("environment_states and encoded.env_coords must align.")
        if int(encoded.module_centers.shape[-1]) != self.spatial_dim:
            raise ValueError("Module coordinate dimension does not match the router.")
        if int(encoded.env_coords.shape[-1]) != self.spatial_dim:
            raise ValueError("Environment coordinate dimension does not match the router.")
        if encoded.env_weights.ndim != 2 or tuple(encoded.env_weights.shape) != tuple(encoded.env_coords.shape[:2]):
            raise ValueError("encoded.env_weights must align with encoded.env_coords as [B,E].")
        if not bool(torch.isfinite(encoded.env_weights).all()) or bool((encoded.env_weights <= 0.0).any()):
            raise ValueError("encoded.env_weights must contain finite strictly positive masses.")

        global_control = self.global_projection(self.global_norm(encoded.global_token))
        module_control = self._source_control(
            module_states,
            encoded.module_centers,
            global_control,
            self.module_projection,
            self.module_source_norm,
            self.module_global_projection,
            self.module_position_projection,
            encoded,
        )
        environment_control = self._source_control(
            environment_states,
            encoded.env_coords,
            global_control,
            self.environment_projection,
            self.environment_source_norm,
            self.environment_global_projection,
            self.environment_position_projection,
            encoded,
        )
        active_modules = encoded.module_present > 0.5
        module_count = active_modules.sum(dim=1, keepdim=True)
        module_measure = active_modules.to(module_states.dtype) / module_count.clamp_min(1).to(module_states.dtype)
        environment_measure = encoded.env_weights / encoded.env_weights.sum(dim=1, keepdim=True)
        gate_context = torch.cat(
            [
                torch.einsum("bm,bmd->bd", module_measure, module_control),
                torch.einsum("be,bed->bd", environment_measure, environment_control),
                global_control,
                torch.log1p(module_count.to(module_states.dtype)),
            ],
            dim=-1,
        )
        return (
            global_control,
            module_control,
            environment_control,
            active_modules,
            module_measure,
            environment_measure,
        ), gate_context

    @staticmethod
    def _available_assign(
        logits: torch.Tensor,
        budget: CaseGroupBudget,
        *,
        source_mask: torch.Tensor | None = None,
        gate_prior: torch.Tensor | None = None,
        gate_support: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the gate log prior and mask before 1.5-entmax."""

        prior = (budget.log_prior() if gate_prior is None else gate_prior).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        available = (budget.support if gate_support is None else gate_support).to(
            device=logits.device,
            dtype=torch.bool,
        )
        mask = available[:, None, :]
        if source_mask is not None:
            mask = mask & source_mask.to(device=logits.device, dtype=torch.bool)
        return entmax15(logits + prior[:, None, :], dim=-1, mask=mask)

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
        *,
        budget: CaseGroupBudget | None = None,
        gate_noise: torch.Tensor | None = None,
        deterministic_gates: bool = False,
        compact: bool = False,
    ) -> tuple[PreparedGroupControl, CaseGroupBudget]:
        """Prepare fresh memberships/controls around one shared gate plan."""

        source_values, gate_context = self._source_controls(
            encoded,
            module_states,
            environment_states,
        )
        (
            global_control,
            module_control,
            environment_control,
            active_modules,
            module_measure,
            environment_measure,
        ) = source_values
        if budget is None:
            prototypes = self.group_codes.to(device=module_states.device, dtype=module_states.dtype)
            optional_logits = self.case_gate.logits(gate_context, prototypes)
            budget = self.case_gate.build_budget(
                optional_logits,
                noise=gate_noise,
                deterministic=bool(deterministic_gates),
            )
        else:
            if not isinstance(budget, CaseGroupBudget):
                raise TypeError("budget must be a CaseGroupBudget instance.")
            if budget.batch_size != int(module_states.shape[0]) or budget.capacity != self.group_count:
                raise ValueError("case budget shape does not match the router.")
            if budget.z.device != module_states.device:
                raise ValueError("case budget must remain on the source-state device.")

        full_codes = self.group_codes.to(device=module_states.device, dtype=module_states.dtype)
        if compact:
            prototype_ids = budget.packed_ids
            packed_valid = budget.packed_valid
            codes = full_codes[prototype_ids]
            z = torch.gather(budget.z, 1, prototype_ids)
            available = torch.gather(budget.support, 1, prototype_ids) & packed_valid
            safe_z = torch.where(available, z, torch.ones_like(z))
            prior = torch.log(safe_z).masked_fill(~available, 0.0)
            group_width = int(prototype_ids.shape[1])
        else:
            prototype_ids = None
            packed_valid = None
            codes = full_codes[None].expand(int(module_states.shape[0]), -1, -1)
            prior = budget.log_prior()
            available = budget.support
            group_width = self.group_count
        module_logits = torch.einsum("bmd,bkd->bmk", module_control, codes)
        module_logits = module_logits / (float(self.control_dim) ** 0.5)
        module_logits = module_logits / float(self.module_temperature)
        module_membership = self._available_assign(
            module_logits,
            budget,
            source_mask=active_modules[..., None],
            gate_prior=prior,
            gate_support=available,
        )
        module_membership = module_membership * active_modules[..., None].to(module_membership.dtype)
        environment_logits = torch.einsum("bed,bkd->bek", environment_control, codes)
        environment_logits = environment_logits / (float(self.control_dim) ** 0.5)
        environment_logits = environment_logits / float(self.environment_temperature)
        environment_membership = self._available_assign(
            environment_logits,
            budget,
            gate_prior=prior,
            gate_support=available,
        )

        module_mass = torch.einsum("bm,bmk->bk", module_measure, module_membership)
        environment_mass = torch.einsum("be,bek->bk", environment_measure, environment_membership)
        kappa = budget.kappa.to(device=module_states.device, dtype=module_states.dtype)
        module_moment = torch.einsum(
            "bm,bmk,bmd->bkd",
            module_measure,
            module_membership,
            module_control,
        ) * kappa[:, None, None]
        environment_moment = torch.einsum(
            "be,bek,bed->bkd",
            environment_measure,
            environment_membership,
            environment_control,
        ) * kappa[:, None, None]
        group_input = torch.cat(
            [
                module_moment,
                environment_moment,
                kappa[:, None, None] * module_mass[..., None],
                kappa[:, None, None] * environment_mass[..., None],
                global_control[:, None, :].expand(-1, group_width, -1),
                codes,
            ],
            dim=-1,
        )
        group_control = torch.tanh(self.group_control(group_input))
        group_control = torch.where(
            available[..., None],
            group_control,
            torch.zeros_like(group_control),
        )
        return (
            PreparedGroupControl(
                module_membership=module_membership,
                environment_membership=environment_membership,
                module_measure=module_measure,
                environment_measure=environment_measure,
                module_mass=module_mass,
                environment_mass=environment_mass,
                module_control=module_control,
                environment_control=environment_control,
                group_control=group_control,
                global_control=global_control,
            ),
            budget,
        )

    def route_queries(
        self,
        encoded: EncodedInterfaceCase,
        state: PreparedGroupControl,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
        *,
        budget: CaseGroupBudget,
        prototype_ids: torch.Tensor | None = None,
        packed_valid: torch.Tensor | None = None,
    ) -> GroupQueryRoute:
        """Route queries against the same available columns as the sources."""

        if receivers.ndim != 3 or int(receivers.shape[0]) != int(state.group_control.shape[0]):
            raise ValueError("receivers must have shape [B,Q,d] aligned with prepared controls.")
        if int(receivers.shape[-1]) != self.spatial_dim:
            raise ValueError("Receiver coordinate dimension does not match the router.")
        if receiver_features is None:
            scale = self._scale(encoded)
            query_features = self.query_fourier(receivers / scale)
        else:
            if receiver_features.ndim != 3 or tuple(receiver_features.shape[:2]) != tuple(receivers.shape[:2]):
                raise ValueError("receiver_features must align with receivers along [B,Q].")
            expected_width = int(
                (self.spatial_dim if self.query_fourier.include_input else 0)
                + 2 * self.spatial_dim * self.query_fourier.num_frequencies
            )
            if int(receiver_features.shape[-1]) == expected_width:
                query_features = receiver_features
            else:
                scale = self._scale(encoded)
                query_features = self.query_fourier(receivers / scale)
        query_input = torch.cat(
            [
                query_features,
                state.global_control[:, None, :].expand(-1, receivers.shape[1], -1),
            ],
            dim=-1,
        )
        query_control = self.query_projection(query_input)
        if prototype_ids is None:
            prior = budget.log_prior().to(device=query_control.device, dtype=query_control.dtype)
            available = budget.support.to(device=query_control.device)
        else:
            if prototype_ids.ndim != 2 or int(prototype_ids.shape[0]) != int(receivers.shape[0]):
                raise ValueError("prototype_ids must have shape [B,Kpack].")
            ids = prototype_ids.to(device=query_control.device, dtype=torch.long)
            if packed_valid is None:
                packed_valid = torch.ones_like(ids, dtype=torch.bool)
            valid = packed_valid.to(device=query_control.device, dtype=torch.bool)
            if valid.shape != ids.shape:
                raise ValueError("packed_valid must align with prototype_ids.")
            z = torch.gather(budget.z.to(device=query_control.device, dtype=query_control.dtype), 1, ids)
            available = torch.gather(budget.support.to(device=query_control.device), 1, ids) & valid
            safe_z = torch.where(available, z, torch.ones_like(z))
            prior = torch.log(safe_z).masked_fill(~available, 0.0)
        transformed_group = self.query_group_projection(state.group_control)
        logits = torch.einsum("bqd,bkd->bqk", query_control, transformed_group)
        logits = logits / (float(self.control_dim) ** 0.5)
        logits = logits / float(self.query_temperature)
        logits = logits + prior[:, None, :]
        assignment = entmax15(logits, dim=-1, mask=available[:, None, :])
        return GroupQueryRoute(
            query_control=query_control,
            assignment=assignment,
            logits=logits,
        )


class BudgetedGroupControlPairwiseField(GroupControlPairwiseField):
    """Run-1406 physical reader with case-budgeted group columns."""

    router_class = BudgetedGroupRouter
    phase_interaction_diagnostics = True
    phase_diagnostic_prefix = "case_group_budget_"
    # Executor choice is a performance seam, independent of diagnostics.
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
        gate_hidden_dim: int = 32,
        hard_concrete_temperature: float = 2.0 / 3.0,
        stretch_lower: float = -0.1,
        stretch_upper: float = 1.1,
        initial_optional_open_probability: float = 0.95,
        always_available_group: int = 0,
        execution_mode: str = "full_width",
    ) -> None:
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
        self.router = BudgetedGroupRouter(
            hidden_dim,
            group_count=group_count,
            control_dim=group_control_dim,
            fourier_frequencies=fourier_frequencies,
            spatial_dim=spatial_dim,
            module_temperature=module_temperature,
            environment_temperature=environment_temperature,
            query_temperature=query_temperature,
            gate_hidden_dim=gate_hidden_dim,
            hard_concrete_temperature=hard_concrete_temperature,
            stretch_lower=stretch_lower,
            stretch_upper=stretch_upper,
            initial_optional_open_probability=initial_optional_open_probability,
            always_available_group=always_available_group,
        )
        self.execution_mode = str(execution_mode)
        if self.execution_mode not in {"full_width", "compact"}:
            raise ValueError("execution_mode must be 'full_width' or 'compact'.")

    def set_execution_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode not in {"full_width", "compact"}:
            raise ValueError("execution_mode must be 'full_width' or 'compact'.")
        self.execution_mode = mode

    def _budget_scale(self, state: dict[str, Any], value: torch.Tensor) -> torch.Tensor:
        budget: CaseGroupBudget = state["case_group_budget"]
        return budget.kappa.to(device=value.device, dtype=value.dtype).view(-1, 1, 1) / float(self.group_count)

    def _route(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
    ) -> GroupQueryRoute:
        budget: CaseGroupBudget = state["case_group_budget"]
        if self.execution_mode == "compact":
            return self.router.route_queries(
                encoded,
                state["group_control_state"],
                receivers,
                receiver_features,
                budget=budget,
                prototype_ids=state["case_group_budget_compact_ids"],
                packed_valid=state["case_group_budget_compact_valid"],
            )
        return self.router.route_queries(
            encoded,
            state["group_control_state"],
            receivers,
            receiver_features,
            budget=budget,
        )

    def _module_execution_plan(
        self,
        state: dict[str, Any],
        route: GroupQueryRoute,
        overlap: torch.Tensor,
        *,
        source_measure: torch.Tensor,
        include_diagnostics: bool,
    ) -> tuple[bool, torch.Tensor | None, Any]:
        # The formal Run-1409 path has no measured gathered-reader crossover.
        # Keep the executor rectangular and report support rows separately in
        # the inherited ledger.  A future measured policy can override this
        # method without changing the gate plan or compact-column semantics.
        del state, route, overlap, source_measure, include_diagnostics
        if self.executor_policy != "rectangular_reference":
            raise ValueError(
                "budgeted_group_control_honf currently supports only the measured-independent "
                "'rectangular_reference' executor policy."
            )
        return True, None, None

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
        # Parent arithmetic has one historical K factor in both linear and
        # bias terms.  A single post-read scale is algebraically equivalent
        # and applies to both rectangular and selected readers without
        # duplicating the fine pair path.
        return context * self._budget_scale(state, context), aux

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
        return context * self._budget_scale(state, context), aux

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
        phase_shared_state: CaseGroupBudget | None = None,
    ) -> dict[str, Any]:
        """Prepare P0 or refresh P1/P2 around the same gate plan."""

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
        controls, budget = self.router.prepare(
            encoded,
            fine["module_tokens"],
            contextual_env,
            budget=phase_shared_state,
            deterministic_gates=(phase_shared_state is None and not self.training),
            compact=self.execution_mode == "compact",
        )
        state = self._prepare_from_fine(
            encoded,
            module_states,
            fine,
            contextual_env,
            controls,
            return_routing_maps=bool(return_routing_maps),
        )
        state["case_group_budget"] = budget
        if self.execution_mode == "compact":
            state["case_group_budget_compact_ids"] = budget.packed_ids
            state["case_group_budget_compact_valid"] = budget.packed_valid
            state["case_group_budget_compact_width"] = int(budget.packed_width)
        state["case_group_budget_gate_reused"] = torch.tensor(
            0.0 if phase_shared_state is None else 1.0,
            device=module_states.device,
            dtype=module_states.dtype,
        ).expand(int(module_states.shape[0])).detach()
        state["case_group_budget_fine_source_refresh"] = torch.ones(
            int(module_states.shape[0]),
            device=module_states.device,
            dtype=module_states.dtype,
        ).detach()
        return state

    def preparation_aux(
        self,
        state: dict[str, Any],
        *,
        include_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        summary = super().preparation_aux(state, include_diagnostics=include_diagnostics)
        budget: CaseGroupBudget = state["case_group_budget"]
        executed_width = int(state["group_control_state"].group_control.shape[1])
        live_count = budget.live_count.to(dtype=state["module_tokens"].dtype)
        summary.update(
            {
                "case_group_budget_sampled_live_count": live_count.detach(),
                "case_group_budget_kappa": budget.kappa.to(
                    dtype=state["module_tokens"].dtype
                ).detach(),
                "case_group_budget_expected_optional_count_detached": budget.expected_optional_count.detach(),
                "case_group_budget_packed_width": state["module_tokens"].new_full(
                    (budget.batch_size,), float(budget.packed_width)
                ).detach(),
                "case_group_budget_executed_group_width": state["module_tokens"].new_full(
                    (budget.batch_size,), float(executed_width)
                ).detach(),
                "case_group_budget_packed_padding_columns": state["module_tokens"].new_full(
                    (budget.batch_size,),
                    float(budget.packed_width),
                ).sub(live_count).detach(),
                "case_group_budget_executed_padding_columns": state["module_tokens"].new_full(
                    (budget.batch_size,),
                    float(executed_width),
                ).sub(live_count).detach(),
                "case_group_budget_gate_reused": state[
                    "case_group_budget_gate_reused"
                ],
                "case_group_budget_fine_source_refresh": state[
                    "case_group_budget_fine_source_refresh"
                ],
            }
        )
        if include_diagnostics:
            summary.update(
                {
                    "case_group_budget_gate_values": budget.z.detach(),
                    "case_group_budget_gate_support": budget.support.detach(),
                    "case_group_budget_packed_prototype_ids": budget.packed_ids.detach(),
                    "case_group_budget_packed_valid": budget.packed_valid.detach(),
                    "case_group_budget_gate_positive_probability": budget.positive_probability.detach(),
                    "case_group_budget_gate_reference_eta": budget.eta.detach(),
                }
            )
        return summary


__all__ = [
    "BudgetedGroupControlPairwiseField",
    "BudgetedGroupRouter",
]
