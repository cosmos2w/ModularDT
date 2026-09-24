"""Run 1503-v2 reversible query-access coalescence over Run 1502.

The case/phase preparation fuses only low-dimensional access descriptors and
packs source incidence plus source-resolved control moments. Physical QM and
QE rows remain fine grained and the inherited unique-source readers execute
each supported physical pair once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import torch

from honf_forward_core.organization.group_fusion import (
    GroupFusionPlan,
    build_group_descriptors,
    effective_query_centers,
    source_footprint_weights,
    weighted_sparsemax,
)

from .group_control_router import GroupQueryRoute
from .sparse_incidence_group_control import SparseIncidenceGroupControlPairwiseField
from .sparse_incidence_router import (
    SparseIncidenceGroupRouter,
    SparseIncidencePreparedGroupControl,
)
from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class CoalescedPreparedGroupControl(SparseIncidencePreparedGroupControl):
    """Fused incidence plus exact source-resolved controller moments."""

    fusion_plan: GroupFusionPlan
    module_source_moment: torch.Tensor
    environment_source_moment: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class CoalescedGroupQueryRoute(GroupQueryRoute):
    """The reader consumes density while diagnostics retain quotient mass."""

    query_mass: torch.Tensor
    query_density: torch.Tensor


class CoalescedSparseIncidencePairwiseField(
    SparseIncidenceGroupControlPairwiseField
):
    """Run-1502 fine reader with reversible case/phase access coalescence."""

    phase_diagnostic_prefix = ("coalescence_", "sparse_incidence_", "group_control_")
    # Rectangular execution remains the default. The inherited opt-in fused
    # policy is separately guarded and consumes the source-resolved B banks.
    executor_policy = "rectangular_reference"
    diagnostic_executor_independent = True

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
        environment_refinement_normalizer: str = "sparsemax",
        fusion_max_iterations: int = 64,
        fusion_eta_final: float = 0.5,
        fusion_ramp_epochs: int = 150,
    ) -> None:
        if int(fusion_max_iterations) != 64:
            raise ValueError("Run 1503-v2 uses exactly 64 convex-fusion iterations.")
        if float(fusion_eta_final) != 0.5:
            raise ValueError("Run 1503-v2 uses eta_final=0.5.")
        if int(fusion_ramp_epochs) != 150:
            raise ValueError("Run 1503-v2 ramps fusion through epoch 150.")
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
            environment_refinement_normalizer=environment_refinement_normalizer,
        )
        self.fusion_max_iterations = int(fusion_max_iterations)
        self.fusion_eta_final = float(fusion_eta_final)
        self.fusion_ramp_epochs = int(fusion_ramp_epochs)
        self._training_epoch: int | None = None
        self._training_total_epochs: int | None = None

    def set_training_progress(
        self, *, epoch: int, total_epochs: int | None = None
    ) -> None:
        """Set the absolute epoch that controls the prescribed fusion ramp."""

        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a nonnegative integer.")
        if total_epochs is not None and (
            isinstance(total_epochs, bool) or int(total_epochs) <= 0
        ):
            raise ValueError("total_epochs must be a positive integer when provided.")
        self._training_epoch = int(epoch)
        self._training_total_epochs = (
            None if total_epochs is None else int(total_epochs)
        )

    def selection_state(self) -> dict[str, int | None]:
        return {"epoch": self._training_epoch, "total_epochs": self._training_total_epochs}

    def _fusion_strength(self) -> float:
        if self._training_epoch is None:
            raise RuntimeError(
                "coalesced_sparse_incidence_honf requires set_training_progress() "
                "before prepare/evaluation so fusion uses the checkpoint epoch."
            )
        return self.fusion_eta_final * min(
            max(float(self._training_epoch), 0.0) / float(self.fusion_ramp_epochs),
            1.0,
        )

    @staticmethod
    def _mean_group_controls(
        controls: SparseIncidencePreparedGroupControl,
        plan: GroupFusionPlan,
    ) -> torch.Tensor:
        batch, groups, width = controls.group_control.shape
        index = plan.class_index.clamp_min(0)
        valid_input = controls.phase_occupied.to(controls.group_control.dtype)
        values = controls.group_control * valid_input[..., None]
        packed = controls.group_control.new_zeros(batch, groups, width).scatter_add(
            1,
            index[..., None].expand(batch, groups, width),
            values,
        )
        return packed / plan.multiplicity.clamp_min(1).to(packed.dtype)[..., None] * plan.valid[..., None]

    def _prepare_environment_bank_with_gain(
        self,
        contextual_env: torch.Tensor,
        controls: SparseIncidencePreparedGroupControl,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Project fine QE values under the exact source-resolved moments."""

        if not isinstance(controls, CoalescedPreparedGroupControl):
            return super()._prepare_environment_bank_with_gain(contextual_env, controls)
        key, value = self.env_attention.project_source(contextual_env)
        source_group_control = controls.environment_source_moment.sum(dim=2)
        value_gain = 1.0 + torch.tanh(
            self.environment_value_control(source_group_control)
        )
        gain = value_gain.reshape(
            contextual_env.shape[0],
            contextual_env.shape[1],
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        head_control = self.environment_score_control(controls.group_control)
        head_source_control = self.environment_score_control(
            controls.environment_source_moment
        ).permute(0, 2, 1, 3)
        return (
            key,
            value * gain,
            source_group_control,
            head_control,
            head_source_control,
            value_gain,
        )

    def _module_control_bank(self, state: dict[str, Any]) -> torch.Tensor:
        controls = state["group_control_state"]
        if isinstance(controls, CoalescedPreparedGroupControl):
            cached = state.get("module_control_bank")
            if state.get("module_control_bank_source_moment") is controls.module_source_moment:
                return cached
            batch, sources, _, width = controls.module_source_moment.shape
            return controls.module_source_moment.permute(0, 2, 1, 3).reshape(
                batch, int(controls.module_membership.shape[-1]), sources * width
            )
        return super()._module_control_bank(state)

    def _require_moment_safe_executor(self) -> None:
        if self.executor_policy not in {
            "rectangular_reference",
            "fused_unique_pairs",
        }:
            raise RuntimeError(
                "coalesced sparse-incidence supports rectangular execution or the "
                "opt-in fused unique-pair reader, which contracts explicit source moments."
            )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        """Build a reversible plan once for the current case and phase."""

        if not isinstance(self.router, SparseIncidenceGroupRouter):
            raise TypeError("coalesced backend requires SparseIncidenceGroupRouter.")
        fusion_strength = self._fusion_strength()
        if fusion_strength == 0.0:
            # This is the strict 1502 identity reference. Bypass all repacking
            # and alternate contraction orders so zero strength is bitwise the
            # parent's full preparation/read path.
            state = super().prepare(
                encoded,
                module_states,
                return_routing_maps=bool(return_routing_maps),
            )
            state["coalescence_fusion_strength"] = module_states.new_zeros(
                int(module_states.shape[0])
            )
            return state
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
        provisional = self.router.prepare(
            encoded, fine["module_tokens"], contextual_env
        )
        query_keys = self.router.normalized_query_key_bank(provisional)
        module_query_centers, environment_query_centers = effective_query_centers(
            provisional.module_centres,
            provisional.environment_centres,
            provisional.module_mass,
            provisional.environment_mass,
        )
        geometry_scale = self.router._geometry_scale(encoded)
        descriptors = build_group_descriptors(
            query_keys,
            module_query_centers,
            environment_query_centers,
            geometry_scale,
        )
        footprint_weights = source_footprint_weights(
            provisional.module_membership,
            provisional.environment_membership,
            provisional.module_measure,
            provisional.environment_measure,
            encoded.module_present > 0.5,
            descriptors,
            provisional.phase_occupied,
        )
        fusion_plan = GroupFusionPlan.build(
            descriptors,
            footprint_weights,
            provisional.phase_occupied,
            fusion_strength,
            control_dim=self.group_control_dim,
            geometry_scale=geometry_scale,
            parent_query_keys=query_keys,
            parent_module_query_centers=module_query_centers,
            parent_environment_query_centers=environment_query_centers,
        )
        module_incidence, module_moment = fusion_plan.reduce_source_banks(
            provisional.module_membership, provisional.group_control
        )
        environment_incidence, environment_moment = fusion_plan.reduce_source_banks(
            provisional.environment_membership, provisional.group_control
        )
        (
            module_mass,
            environment_mass,
            module_centers,
            environment_centers,
            joint_centers,
        ) = self.router._joint_centres(
            module_incidence,
            environment_incidence,
            provisional.module_measure,
            provisional.environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        fused_group_control = self._mean_group_controls(provisional, fusion_plan)
        controls = replace(
            provisional,
            module_membership=module_incidence,
            environment_membership=environment_incidence,
            module_mass=module_mass,
            environment_mass=environment_mass,
            group_control=fused_group_control,
            phase_occupied=fusion_plan.valid,
            module_centres=module_centers,
            environment_centres=environment_centers,
            joint_centres=joint_centers,
        )
        controls = CoalescedPreparedGroupControl(
            **{
                name: getattr(controls, name)
                for name in SparseIncidencePreparedGroupControl.__dataclass_fields__
            },
            fusion_plan=fusion_plan,
            module_source_moment=module_moment,
            environment_source_moment=environment_moment,
        )
        state = self._prepare_from_fine(
            encoded,
            module_states,
            fine,
            contextual_env,
            controls,
            return_routing_maps=bool(return_routing_maps),
        )
        module_control_bank = module_moment.permute(0, 2, 1, 3).reshape(
            int(module_states.shape[0]),
            int(module_incidence.shape[-1]),
            int(module_states.shape[1]) * self.group_control_dim,
        )
        state["module_control_bank"] = module_control_bank
        state["module_control_bank_membership"] = controls.module_membership
        state["module_control_bank_group_control"] = controls.group_control
        state["module_control_bank_source_moment"] = module_moment
        state["coalescence_plan"] = fusion_plan
        state["coalescence_strength"] = fusion_plan.fusion_strength
        state["coalescence_geometry_scale"] = geometry_scale
        state["coalescence_footprint_weights"] = footprint_weights
        if return_routing_maps:
            state["coalescence_provisional_module_incidence"] = (
                provisional.module_membership
            )
            state["coalescence_provisional_environment_incidence"] = (
                provisional.environment_membership
            )
            state["coalescence_provisional_group_control"] = provisional.group_control
        state.update(
            {
                "sparse_incidence_kappa": provisional.kappa,
                "sparse_incidence_pi": provisional.pi,
                "sparse_incidence_active_mask": provisional.phase_occupied,
                "sparse_incidence_module_centres": provisional.module_centres,
                "sparse_incidence_environment_centres": provisional.environment_centres,
                "sparse_incidence_joint_centres": provisional.joint_centres,
            }
        )
        aux = state["group_control_preparation_aux"]
        eta = fusion_plan.fusion_strength
        aux.update(
            {
                "sparse_incidence_registered_capacity": module_states.new_full(
                    (int(module_states.shape[0]),), float(self.group_count)
                ).detach(),
                "sparse_incidence_kappa": provisional.kappa.detach(),
                "sparse_incidence_pi": provisional.pi.detach(),
                "sparse_incidence_phase_occupied": provisional.phase_occupied.detach(),
                "sparse_incidence_module_mass": provisional.module_mass.detach(),
                "sparse_incidence_environment_mass": provisional.environment_mass.detach(),
                "sparse_incidence_proposal_module_mass": provisional.proposal_module_mass.detach(),
                "sparse_incidence_proposal_environment_mass": provisional.proposal_environment_mass.detach(),
                "sparse_incidence_proposal_occupied": provisional.proposal_occupied.detach(),
                "sparse_incidence_module_centres": provisional.module_centres.detach(),
                "sparse_incidence_environment_centres": provisional.environment_centres.detach(),
                "sparse_incidence_joint_centres": provisional.joint_centres.detach(),
                "coalescence_registered_capacity": module_states.new_full(
                    (int(module_states.shape[0]),), float(self.group_count)
                ).detach(),
                "coalescence_k_proposal": fusion_plan.active_proposal_count.detach(),
                "coalescence_k_case": fusion_plan.case_group_count.detach(),
                "coalescence_multiplicity": fusion_plan.multiplicity.detach(),
                "coalescence_active_proposals": fusion_plan.active.detach(),
                "coalescence_valid_packed": fusion_plan.valid.detach(),
                "coalescence_fusion_strength": eta.detach(),
                "coalescence_sigma": fusion_plan.sigma.detach(),
                "coalescence_lambda": fusion_plan.lambda_b.detach(),
                "coalescence_merge_displacement": fusion_plan.merge_displacement.detach(),
                "coalescence_projection_displacement": fusion_plan.projection_displacement.detach(),
                "coalescence_primal_residual": fusion_plan.primal_residual.detach(),
                "coalescence_dual_residual": fusion_plan.dual_residual.detach(),
            }
        )
        return state

    def preparation_aux(
        self,
        state: dict[str, Any],
        *,
        include_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        if "coalescence_plan" not in state:
            summary = super().preparation_aux(
                state, include_diagnostics=include_diagnostics
            )
            strength = state.get("coalescence_fusion_strength")
            if torch.is_tensor(strength):
                summary["coalescence_fusion_strength"] = strength.detach()
            return summary
        summary = super().preparation_aux(
            state, include_diagnostics=include_diagnostics
        )
        plan: GroupFusionPlan = state["coalescence_plan"]
        provisional_active = state["sparse_incidence_active_mask"]
        summary.update(
            {
                "coalescence_k_proposal": plan.active_proposal_count.detach(),
                "coalescence_k_case": plan.case_group_count.detach(),
                "coalescence_multiplicity": plan.multiplicity.detach(),
                "coalescence_fusion_strength": plan.fusion_strength.detach(),
                "coalescence_sigma": plan.sigma.detach(),
                "coalescence_lambda": plan.lambda_b.detach(),
                "coalescence_merge_displacement": plan.merge_displacement.detach(),
                "coalescence_projection_displacement": plan.projection_displacement.detach(),
                "coalescence_primal_residual": plan.primal_residual.detach(),
                "coalescence_dual_residual": plan.dual_residual.detach(),
                "coalescence_provisional_phase_occupied": provisional_active.detach(),
                "coalescence_valid_packed": plan.valid.detach(),
            }
        )
        # Keep the historical sparse-incidence key semantically stable across
        # maps-on and maps-off calls: it always describes the provisional
        # parent organization, while coalescence_valid_packed is the quotient.
        summary["sparse_incidence_phase_occupied"] = provisional_active.detach()
        if include_diagnostics:
            controls = state["group_control_state"]
            if not isinstance(controls, CoalescedPreparedGroupControl):
                raise TypeError("coalescence diagnostics require coalesced controls.")
            summary.update(
                {
                    "coalescence_fused_descriptors": plan.fused_descriptors.detach(),
                    "coalescence_provisional_descriptors": plan.input_descriptors.detach(),
                    "coalescence_class_index": plan.class_index.detach(),
                    "coalescence_pair_fused": plan.pair_fused.detach(),
                    "coalescence_footprint_weights": state[
                        "coalescence_footprint_weights"
                    ].detach(),
                    "coalescence_module_incidence": controls.module_membership.detach(),
                    "coalescence_environment_incidence": controls.environment_membership.detach(),
                    "coalescence_module_source_moment": controls.module_source_moment.detach(),
                    "coalescence_environment_source_moment": controls.environment_source_moment.detach(),
                    "coalescence_routing_module_centers": plan.module_query_centers.detach(),
                    "coalescence_routing_environment_centers": plan.environment_query_centers.detach(),
                    "coalescence_provisional_module_incidence": state[
                        "coalescence_provisional_module_incidence"
                    ].detach(),
                    "coalescence_provisional_environment_incidence": state[
                        "coalescence_provisional_environment_incidence"
                    ].detach(),
                    "coalescence_provisional_group_control": state[
                        "coalescence_provisional_group_control"
                    ].detach(),
                }
            )
        return summary

    def _route(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
    ) -> CoalescedGroupQueryRoute:
        if "coalescence_plan" not in state:
            parent_route = super()._route(
                state, encoded, receivers, receiver_features
            )
            return CoalescedGroupQueryRoute(
                query_control=parent_route.query_control,
                assignment=parent_route.assignment,
                logits=parent_route.logits,
                query_keys=parent_route.query_keys,
                query_mass=parent_route.assignment,
                query_density=parent_route.assignment,
            )
        if not isinstance(self.router, SparseIncidenceGroupRouter):
            raise TypeError("coalesced backend requires SparseIncidenceGroupRouter.")
        controls = state["group_control_state"]
        if not isinstance(controls, CoalescedPreparedGroupControl):
            raise TypeError("coalesced backend requires prepared fusion moments.")
        if receivers.ndim != 3 or int(receivers.shape[0]) != int(controls.group_control.shape[0]):
            raise ValueError("receivers must have shape [B,Q,d] aligned with fused controls.")
        if int(receivers.shape[-1]) != self.spatial_dim:
            raise ValueError("receiver coordinate dimension does not match the sparse-incidence router.")
        if receiver_features is None:
            query_features = self.router.query_fourier(
                receivers / self.router._scale(encoded)
            )
        else:
            if receiver_features.ndim != 3 or tuple(receiver_features.shape[:2]) != tuple(receivers.shape[:2]):
                raise ValueError("receiver_features must align with receivers along [B,Q].")
            expected_width = int(
                (self.spatial_dim if self.router.query_fourier.include_input else 0)
                + 2 * self.spatial_dim * self.router.query_fourier.num_frequencies
            )
            query_features = (
                receiver_features
                if int(receiver_features.shape[-1]) == expected_width
                else self.router.query_fourier(
                    receivers / self.router._scale(encoded)
                )
            )
        query_input = torch.cat(
            [
                query_features,
                controls.global_control[:, None, :].expand(
                    -1, receivers.shape[1], -1
                ),
            ],
            dim=-1,
        )
        query_control = self.router.query_projection(query_input)
        scaled_query = self.router._rms_scale(query_control)
        plan = controls.fusion_plan
        query_keys = plan.query_keys
        logits = torch.einsum(
            "bqd,bkd->bqk",
            scaled_query,
            plan.fused_descriptors[..., : self.group_control_dim],
        )
        module_centers = plan.module_query_centers
        environment_centers = plan.environment_query_centers
        geometry = -0.5 * (
            torch.linalg.vector_norm(
                receivers[:, :, None, :] - module_centers[:, None, :, :], dim=-1
            )
            + torch.linalg.vector_norm(
                receivers[:, :, None, :] - environment_centers[:, None, :, :],
                dim=-1,
            )
        ) / plan.geometry_scale[:, None, None]
        logits = logits + geometry
        mass, density = weighted_sparsemax(
            logits,
            plan.multiplicity,
            plan.valid[:, None, :],
        )
        return CoalescedGroupQueryRoute(
            query_control=query_control,
            assignment=density,
            logits=logits,
            query_keys=query_keys,
            query_mass=mass,
            query_density=density,
        )

    def read(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._require_moment_safe_executor()
        context, aux = super().read(
            state,
            encoded,
            receivers,
            receiver_features,
            return_routing_maps=bool(return_routing_maps),
        )
        if "coalescence_plan" not in state:
            return context, aux
        if return_routing_maps:
            route = self._route(state, encoded, receivers, receiver_features)
            plan: GroupFusionPlan = state["coalescence_plan"]
            virtual_support = (
                (route.query_density > 0.0).to(plan.multiplicity.dtype)
                * plan.multiplicity[:, None, :]
            ).sum(dim=-1)
            coalesced_kq = (route.query_mass > 0.0).sum(dim=-1)
            aux.update(
                {
                    "group_control_query_routing": route.query_mass.detach(),
                    "group_control_query_density": route.query_density.detach(),
                    "coalescence_query_mass": route.query_mass.detach(),
                    "coalescence_query_density": route.query_density.detach(),
                    "coalescence_query_logits": route.logits.detach(),
                    "coalescence_query_kq": coalesced_kq.detach(),
                    "coalescence_virtual_constituent_support": virtual_support.detach(),
                    "sparse_incidence_query_routing": route.query_mass.detach(),
                    "sparse_incidence_query_logits": route.logits.detach(),
                    "sparse_incidence_query_keys": route.query_keys.detach(),
                }
            )
        return context, aux


@dataclass(frozen=True)
class ConvergedCoalescedPreparedGroupControl(
    SparseIncidencePreparedGroupControl
):
    """V3 quotient banks, with the parent controls retained for identity rows."""

    fusion_plan: GroupFusionPlan
    module_source_moment: torch.Tensor
    environment_source_moment: torch.Tensor
    parent_controls: SparseIncidencePreparedGroupControl | None = None
    parent_identity_rows: torch.Tensor | None = None
    identity_free_quotient: bool = False


class ConvergedIdentityPreservingCoalescencePairwiseField(
    SparseIncidenceGroupControlPairwiseField
):
    """Run-1503-v3 converged quotient with an exact Run-1502 singleton path.

    This is a separate architecture from the fixed-64 Run-1503-v2 backend.
    Numerical planning is detached; physical keys, incidences, source moments,
    centers, and the inherited fine QM/QE readers remain differentiable.
    """

    phase_diagnostic_prefix = (
        "coalescence_",
        "sparse_incidence_",
        "group_control_",
    )
    executor_policy = "rectangular_reference"
    diagnostic_executor_independent = True

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
        environment_refinement_normalizer: str = "sparsemax",
        fusion_max_iterations: int = 512,
        fusion_eps_abs: float = 1.0e-9,
        fusion_eps_rel: float = 1.0e-8,
        fusion_eta_final: float = 0.5,
        fusion_ramp_epochs: int = 150,
    ) -> None:
        if int(fusion_max_iterations) != 512:
            raise ValueError(
                "converged_identity_preserving_coalescence_honf uses a "
                "512-iteration numerical ceiling."
            )
        if float(fusion_eta_final) != 0.5:
            raise ValueError("v3 coalescence uses eta_final=0.5.")
        if int(fusion_ramp_epochs) != 150:
            raise ValueError("v3 coalescence ramps fusion through epoch 150.")
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
            environment_refinement_normalizer=environment_refinement_normalizer,
        )
        self.fusion_max_iterations = int(fusion_max_iterations)
        self.fusion_eps_abs = float(fusion_eps_abs)
        self.fusion_eps_rel = float(fusion_eps_rel)
        self.fusion_eta_final = float(fusion_eta_final)
        self.fusion_ramp_epochs = int(fusion_ramp_epochs)
        self._training_epoch: int | None = None
        self._training_total_epochs: int | None = None

    def set_training_progress(
        self, *, epoch: int, total_epochs: int | None = None
    ) -> None:
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a nonnegative integer.")
        if total_epochs is not None and (
            isinstance(total_epochs, bool) or int(total_epochs) <= 0
        ):
            raise ValueError("total_epochs must be a positive integer when provided.")
        self._training_epoch = int(epoch)
        self._training_total_epochs = (
            None if total_epochs is None else int(total_epochs)
        )

    def selection_state(self) -> dict[str, int | None]:
        return {"epoch": self._training_epoch, "total_epochs": self._training_total_epochs}

    def _fusion_strength(self) -> float:
        if self._training_epoch is None:
            raise RuntimeError(
                "converged_identity_preserving_coalescence_honf requires "
                "set_training_progress() before prepare/evaluation."
            )
        return self.fusion_eta_final * min(
            max(float(self._training_epoch), 0.0) / float(self.fusion_ramp_epochs),
            1.0,
        )

    @staticmethod
    def _batch_select(
        rows: torch.Tensor, parent: torch.Tensor, compact: torch.Tensor
    ) -> torch.Tensor:
        shape = (int(rows.shape[0]),) + (1,) * (compact.ndim - 1)
        return torch.where(rows.reshape(shape), parent, compact)

    @staticmethod
    def _class_membership(
        plan: GroupFusionPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = int(plan.active.shape[1])
        membership = torch.nn.functional.one_hot(
            plan.class_index.clamp_min(0), num_classes=groups
        ).to(dtype=plan.input_descriptors.dtype)
        membership = membership * plan.active[..., None].to(membership.dtype)
        roots = membership.argmax(dim=1)
        return membership, roots

    @staticmethod
    def _gather_class_values(values: torch.Tensor, roots: torch.Tensor) -> torch.Tensor:
        width = int(values.shape[-1])
        return values.gather(1, roots[..., None].expand(-1, -1, width))

    @staticmethod
    def _direct_singleton_source_banks(
        incidence: torch.Tensor,
        controls: torch.Tensor,
        roots: torch.Tensor,
        singleton_classes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sources, groups = incidence.shape
        source_index = roots[:, None, :].expand(batch, sources, groups)
        class_incidence = incidence.gather(2, source_index)
        class_control = controls.gather(
            1, roots[..., None].expand(-1, -1, int(controls.shape[-1]))
        )
        class_moment = class_incidence[..., None] * class_control[:, None, :, :]
        class_incidence = torch.where(
            singleton_classes[:, None, :],
            class_incidence,
            torch.zeros_like(class_incidence),
        )
        class_moment = torch.where(
            singleton_classes[:, None, :, None],
            class_moment,
            torch.zeros_like(class_moment),
        )
        return class_incidence, class_moment

    @staticmethod
    def _mean_group_controls_v3(
        controls: SparseIncidencePreparedGroupControl,
        plan: GroupFusionPlan,
        membership: torch.Tensor,
        roots: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, width = controls.group_control.shape
        summed = torch.einsum(
            "bkr,bkd->brd", membership, controls.group_control
        )
        means = summed / plan.multiplicity.clamp_min(1).to(summed.dtype)[..., None]
        singleton_classes = plan.valid & (plan.multiplicity == 1)
        parent_keys = controls.group_control.gather(
            1, roots[..., None].expand(batch, roots.shape[1], width)
        )
        means = torch.where(singleton_classes[..., None], parent_keys, means)
        return means * plan.valid[..., None].to(means.dtype)

    def _prepare_environment_bank_with_gain(
        self,
        contextual_env: torch.Tensor,
        controls: SparseIncidencePreparedGroupControl,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not isinstance(controls, ConvergedCoalescedPreparedGroupControl):
            return super()._prepare_environment_bank_with_gain(
                contextual_env, controls
            )
        key, value = self.env_attention.project_source(contextual_env)
        source_group_control = controls.environment_source_moment.sum(dim=2)
        value_gain = 1.0 + torch.tanh(
            self.environment_value_control(source_group_control)
        )
        gain = value_gain.reshape(
            contextual_env.shape[0],
            contextual_env.shape[1],
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        head_control = self.environment_score_control(controls.group_control)
        head_source_control = self.environment_score_control(
            controls.environment_source_moment
        ).permute(0, 2, 1, 3)
        compact = (
            key,
            value * gain,
            source_group_control,
            head_control,
            head_source_control,
            value_gain,
        )
        if controls.identity_free_quotient:
            return compact
        rows = controls.parent_identity_rows
        if rows is None or not bool(rows.any()):
            return compact
        if controls.parent_controls is None:
            raise RuntimeError("identity rows are missing their Run-1502 controls.")
        parent = super()._prepare_environment_bank_with_gain(
            contextual_env, controls.parent_controls
        )
        return tuple(
            self._batch_select(rows, parent_value, compact_value)
            for parent_value, compact_value in zip(parent, compact, strict=True)
        )

    def _module_control_bank(self, state: dict[str, Any]) -> torch.Tensor:
        controls = state["group_control_state"]
        if isinstance(controls, ConvergedCoalescedPreparedGroupControl):
            cached = state.get("module_control_bank")
            if state.get("module_control_bank_source_moment") is controls.module_source_moment:
                return cached
            batch, sources, _, width = controls.module_source_moment.shape
            return controls.module_source_moment.permute(0, 2, 1, 3).reshape(
                batch, int(controls.module_membership.shape[-1]), sources * width
            )
        return super()._module_control_bank(state)

    def _pair_module_moment(
        self,
        state: dict[str, Any],
        alpha: torch.Tensor,
        batch_index: torch.Tensor,
        query_index: torch.Tensor,
        source_index: torch.Tensor,
        *,
        overlap: torch.Tensor,
        chunk_size: int = 131_072,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Use the literal parent contraction for singleton rows in a mix."""

        controls = state.get("group_control_state")
        if not isinstance(controls, ConvergedCoalescedPreparedGroupControl):
            yield from super()._pair_module_moment(
                state,
                alpha,
                batch_index,
                query_index,
                source_index,
                overlap=overlap,
                chunk_size=chunk_size,
            )
            return
        source_moment = state.get("module_control_bank_source_moment")
        if not torch.is_tensor(source_moment):
            raise RuntimeError("v3 coalesced QM requires its source-resolved B bank.")
        if not controls.identity_free_quotient and (
            controls.parent_controls is None or controls.parent_identity_rows is None
        ):
            raise RuntimeError("v3 coalesced QM is missing parent identity controls.")
        total = int(batch_index.shape[0])
        for start in range(0, total, int(chunk_size)):
            stop = min(start + int(chunk_size), total)
            batches = batch_index[start:stop]
            queries = query_index[start:stop]
            sources = source_index[start:stop]
            alpha_pair = alpha[batches, queries]
            moment_pair = source_moment[batches, sources]
            rho = overlap[batches, queries, sources]
            moment = torch.einsum("pk,pkd->pd", alpha_pair, moment_pair)
            if not controls.identity_free_quotient:
                identity_pair = controls.parent_identity_rows[batches]
                if bool(identity_pair.any()):
                    parent = controls.parent_controls
                    membership_pair = parent.module_membership[batches, sources]
                    group_control = parent.group_control[batches]
                    parent_moment = torch.einsum(
                        "pk,pk,pkd->pd",
                        alpha_pair,
                        membership_pair,
                        group_control,
                    )
                    moment = torch.where(identity_pair[:, None], parent_moment, moment)
            yield rho, moment

    def _pair_environment_head_moment(
        self,
        state: dict[str, Any],
        alpha: torch.Tensor,
        batch_index: torch.Tensor,
        query_index: torch.Tensor,
        source_index: torch.Tensor,
        *,
        overlap: torch.Tensor,
        chunk_size: int = 131_072,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Preserve the parent head-space operation order for identity rows."""

        controls = state.get("group_control_state")
        if not isinstance(controls, ConvergedCoalescedPreparedGroupControl):
            yield from super()._pair_environment_head_moment(
                state,
                alpha,
                batch_index,
                query_index,
                source_index,
                overlap=overlap,
                chunk_size=chunk_size,
            )
            return
        source_control = state.get("environment_head_source_control")
        if not torch.is_tensor(source_control):
            raise RuntimeError("v3 coalesced QE requires its source-resolved head bank.")
        if not controls.identity_free_quotient and (
            controls.parent_controls is None or controls.parent_identity_rows is None
        ):
            raise RuntimeError("v3 coalesced QE is missing parent identity controls.")
        groups = int(source_control.shape[1])
        group_index = torch.arange(groups, device=alpha.device)
        total = int(batch_index.shape[0])
        for start in range(0, total, int(chunk_size)):
            stop = min(start + int(chunk_size), total)
            batches = batch_index[start:stop]
            queries = query_index[start:stop]
            sources = source_index[start:stop]
            alpha_pair = alpha[batches, queries]
            head_pair = source_control[
                batches[:, None], group_index[None, :], sources[:, None], :
            ]
            rho = overlap[batches, queries, sources]
            zeta = torch.einsum("pk,pkh->ph", alpha_pair, head_pair)
            if not controls.identity_free_quotient:
                identity_pair = controls.parent_identity_rows[batches]
                if bool(identity_pair.any()):
                    parent = controls.parent_controls
                    membership_pair = parent.environment_membership[batches, sources]
                    head_control = state["environment_head_control"][batches]
                    parent_zeta = torch.einsum(
                        "pk,pk,pkh->ph",
                        alpha_pair,
                        membership_pair,
                        head_control,
                    )
                    zeta = torch.where(identity_pair[:, None], parent_zeta, zeta)
            yield rho, zeta

    def _require_moment_safe_executor(self) -> None:
        if self.executor_policy not in {
            "rectangular_reference",
            "fused_unique_pairs",
        }:
            raise RuntimeError(
                "v3 coalescence supports rectangular execution or the existing "
                "opt-in fused unique-pair reader."
            )

    def _plan_metadata(
        self, state: dict[str, Any], plan: GroupFusionPlan
    ) -> None:
        state["coalescence_plan"] = plan
        state["coalescence_fusion_strength"] = plan.fusion_strength
        state["coalescence_footprint_weights"] = state.pop(
            "_coalescence_footprint_weights"
        )
        aux = state["group_control_preparation_aux"]
        aux.update(
            {
                "coalescence_registered_capacity": state["module_tokens"].new_full(
                    (int(state["module_tokens"].shape[0]),),
                    float(self.group_count),
                ).detach(),
                "coalescence_k_proposal": plan.active_proposal_count.detach(),
                "coalescence_k_case": plan.case_group_count.detach(),
                "coalescence_multiplicity": plan.multiplicity.detach(),
                "coalescence_active_proposals": plan.active.detach(),
                "coalescence_valid_packed": plan.valid.detach(),
                "coalescence_fusion_strength": plan.fusion_strength.detach(),
                "coalescence_sigma": plan.sigma.detach(),
                "coalescence_lambda": plan.lambda_b.detach(),
                "coalescence_merge_displacement": plan.merge_displacement.detach(),
                "coalescence_projection_displacement": plan.projection_displacement.detach(),
                "coalescence_primal_residual": plan.primal_residual.detach(),
                "coalescence_dual_residual": plan.dual_residual.detach(),
                "coalescence_solver_converged": plan.solver_converged.detach(),
                "coalescence_solver_iterations": plan.solver_iterations.detach(),
                "coalescence_primal_tolerance": plan.primal_tolerance.detach(),
                "coalescence_dual_tolerance": plan.dual_tolerance.detach(),
            }
        )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(self.router, SparseIncidenceGroupRouter):
            raise TypeError("v3 coalescence requires SparseIncidenceGroupRouter.")
        fusion_strength = self._fusion_strength()
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
        provisional = self.router.prepare(
            encoded, fine["module_tokens"], contextual_env
        )
        parent_query_keys = self.router.normalized_query_key_bank(provisional)
        parent_module_centers, parent_environment_centers = effective_query_centers(
            provisional.module_centres,
            provisional.environment_centres,
            provisional.module_mass,
            provisional.environment_mass,
        )
        geometry_scale = self.router._geometry_scale(encoded)
        # Descriptors and footprints define a detached discrete plan. The
        # physical access representation below is rebuilt from live parent
        # keys and source incidence, retaining the desired gradients.
        with torch.no_grad():
            descriptors = build_group_descriptors(
                parent_query_keys.detach(),
                parent_module_centers.detach(),
                parent_environment_centers.detach(),
                geometry_scale.detach(),
            )
            footprint_weights = source_footprint_weights(
                provisional.module_membership.detach(),
                provisional.environment_membership.detach(),
                provisional.module_measure.detach(),
                provisional.environment_measure.detach(),
                (encoded.module_present > 0.5).detach(),
                descriptors,
                provisional.phase_occupied.detach(),
            )
        plan = GroupFusionPlan.build_converged(
            descriptors,
            footprint_weights,
            provisional.phase_occupied,
            fusion_strength,
            max_iterations=self.fusion_max_iterations,
            eps_abs=self.fusion_eps_abs,
            eps_rel=self.fusion_eps_rel,
            control_dim=self.group_control_dim,
            geometry_scale=geometry_scale.detach(),
            parent_query_keys=parent_query_keys.detach(),
            parent_module_query_centers=parent_module_centers.detach(),
            parent_environment_query_centers=parent_environment_centers.detach(),
        )
        # A no-merge row uses the parent's group IDs, including holes, so all
        # parent banks remain in exactly the original K columns.
        identity_rows = plan.case_group_count == plan.active_proposal_count
        original_ids = torch.arange(
            int(plan.active.shape[1]), device=plan.active.device, dtype=torch.long
        )[None, :].expand_as(plan.class_index)
        plan = replace(
            plan,
            class_index=torch.where(
                identity_rows[:, None] & plan.active,
                original_ids,
                plan.class_index,
            ),
            multiplicity=torch.where(
                identity_rows[:, None],
                plan.active.to(torch.long),
                plan.multiplicity,
            ),
            valid=torch.where(identity_rows[:, None], plan.active, plan.valid),
        )
        identity_rows = plan.case_group_count == plan.active_proposal_count

        if bool(identity_rows.all()):
            # Reuse the already-prepared parent controls and fine tensors. This
            # exactly preserves parent banks/read paths without a second pass
            # through any stochastic fine preparation.
            state = self._prepare_from_fine(
                encoded,
                module_states,
                fine,
                contextual_env,
                provisional,
                return_routing_maps=bool(return_routing_maps),
            )
            state["coalescence_all_singleton"] = True
            state["coalescence_parent_identity_rows"] = identity_rows
            state["_coalescence_footprint_weights"] = footprint_weights.detach()
            self._plan_metadata(state, plan)
            state.update(
                {
                    "sparse_incidence_kappa": provisional.kappa,
                    "sparse_incidence_pi": provisional.pi,
                    "sparse_incidence_active_mask": provisional.phase_occupied,
                    "sparse_incidence_module_centres": provisional.module_centres,
                    "sparse_incidence_environment_centres": provisional.environment_centres,
                    "sparse_incidence_joint_centres": provisional.joint_centres,
                }
            )
            return state

        membership, roots = self._class_membership(plan)
        singleton_classes = plan.valid & (plan.multiplicity == 1)
        module_incidence, module_moment = plan.reduce_source_banks(
            provisional.module_membership, provisional.group_control
        )
        environment_incidence, environment_moment = plan.reduce_source_banks(
            provisional.environment_membership, provisional.group_control
        )
        direct_module, direct_module_moment = self._direct_singleton_source_banks(
            provisional.module_membership,
            provisional.group_control,
            roots,
            singleton_classes,
        )
        direct_environment, direct_environment_moment = self._direct_singleton_source_banks(
            provisional.environment_membership,
            provisional.group_control,
            roots,
            singleton_classes,
        )
        module_incidence = torch.where(
            singleton_classes[:, None, :], direct_module, module_incidence
        )
        module_moment = torch.where(
            singleton_classes[:, None, :, None], direct_module_moment, module_moment
        )
        environment_incidence = torch.where(
            singleton_classes[:, None, :], direct_environment, environment_incidence
        )
        environment_moment = torch.where(
            singleton_classes[:, None, :, None],
            direct_environment_moment,
            environment_moment,
        )
        (
            module_mass,
            environment_mass,
            module_centers,
            environment_centers,
            joint_centers,
        ) = self.router._joint_centres(
            module_incidence,
            environment_incidence,
            provisional.module_measure,
            provisional.environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        compact_module_centers, compact_environment_centers = effective_query_centers(
            module_centers,
            environment_centers,
            module_mass,
            environment_mass,
        )
        parent_class_module_centers = self._gather_class_values(
            parent_module_centers, roots
        )
        parent_class_environment_centers = self._gather_class_values(
            parent_environment_centers, roots
        )
        compact_module_centers = torch.where(
            singleton_classes[..., None],
            parent_class_module_centers,
            compact_module_centers,
        )
        compact_environment_centers = torch.where(
            singleton_classes[..., None],
            parent_class_environment_centers,
            compact_environment_centers,
        )
        parent_module_raw = self._gather_class_values(provisional.module_centres, roots)
        parent_environment_raw = self._gather_class_values(
            provisional.environment_centres, roots
        )
        parent_joint_raw = self._gather_class_values(provisional.joint_centres, roots)
        module_mass = torch.where(
            singleton_classes,
            provisional.module_mass.gather(1, roots),
            module_mass,
        )
        environment_mass = torch.where(
            singleton_classes,
            provisional.environment_mass.gather(1, roots),
            environment_mass,
        )
        module_centers = torch.where(
            singleton_classes[..., None], parent_module_raw, module_centers
        )
        environment_centers = torch.where(
            singleton_classes[..., None], parent_environment_raw, environment_centers
        )
        joint_centers = torch.where(
            singleton_classes[..., None], parent_joint_raw, joint_centers
        )
        compact_group_control = self._mean_group_controls_v3(
            provisional, plan, membership, roots
        )

        # Entire rows with no accepted merge take the original parent tensors
        # verbatim. This also handles sparse occupancy IDs in mixed batches.
        module_incidence = self._batch_select(
            identity_rows, provisional.module_membership, module_incidence
        )
        environment_incidence = self._batch_select(
            identity_rows, provisional.environment_membership, environment_incidence
        )
        module_moment = self._batch_select(
            identity_rows,
            provisional.module_membership[..., None]
            * provisional.group_control[:, None, :, :],
            module_moment,
        )
        environment_moment = self._batch_select(
            identity_rows,
            provisional.environment_membership[..., None]
            * provisional.group_control[:, None, :, :],
            environment_moment,
        )
        module_mass = self._batch_select(identity_rows, provisional.module_mass, module_mass)
        environment_mass = self._batch_select(
            identity_rows, provisional.environment_mass, environment_mass
        )
        module_centers = self._batch_select(
            identity_rows, provisional.module_centres, module_centers
        )
        environment_centers = self._batch_select(
            identity_rows, provisional.environment_centres, environment_centers
        )
        joint_centers = self._batch_select(
            identity_rows, provisional.joint_centres, joint_centers
        )
        compact_group_control = self._batch_select(
            identity_rows, provisional.group_control, compact_group_control
        )
        compact_module_centers = self._batch_select(
            identity_rows, parent_module_centers, compact_module_centers
        )
        compact_environment_centers = self._batch_select(
            identity_rows, parent_environment_centers, compact_environment_centers
        )
        controls_base = replace(
            provisional,
            module_membership=module_incidence,
            environment_membership=environment_incidence,
            module_mass=module_mass,
            environment_mass=environment_mass,
            group_control=compact_group_control,
            phase_occupied=plan.valid,
            module_centres=module_centers,
            environment_centres=environment_centers,
            joint_centres=joint_centers,
        )
        controls_base = replace(
            controls_base,
            phase_occupied=self._batch_select(
                identity_rows, provisional.phase_occupied, plan.valid
            ),
        )
        controls = ConvergedCoalescedPreparedGroupControl(
            **{
                name: getattr(controls_base, name)
                for name in SparseIncidencePreparedGroupControl.__dataclass_fields__
            },
            fusion_plan=plan,
            module_source_moment=module_moment,
            environment_source_moment=environment_moment,
            parent_controls=provisional,
            parent_identity_rows=identity_rows,
        )
        state = self._prepare_from_fine(
            encoded,
            module_states,
            fine,
            contextual_env,
            controls,
            return_routing_maps=bool(return_routing_maps),
        )
        module_control_bank = module_moment.permute(0, 2, 1, 3).reshape(
            int(module_states.shape[0]),
            int(module_incidence.shape[-1]),
            int(module_states.shape[1]) * self.group_control_dim,
        )
        state["module_control_bank"] = module_control_bank
        state["module_control_bank_membership"] = controls.module_membership
        state["module_control_bank_group_control"] = controls.group_control
        state["module_control_bank_source_moment"] = module_moment
        state["coalescence_parent_identity_rows"] = identity_rows
        state["coalescence_all_singleton"] = False
        state["coalescence_query_keys"] = self._compact_query_keys(
            parent_query_keys, membership, roots, plan
        )
        state["coalescence_module_query_centers"] = compact_module_centers
        state["coalescence_environment_query_centers"] = compact_environment_centers
        state["_coalescence_footprint_weights"] = footprint_weights.detach()
        if return_routing_maps:
            state["coalescence_provisional_module_incidence"] = (
                provisional.module_membership
            )
            state["coalescence_provisional_environment_incidence"] = (
                provisional.environment_membership
            )
            state["coalescence_provisional_group_control"] = provisional.group_control
        self._plan_metadata(state, plan)
        state.update(
            {
                "sparse_incidence_kappa": provisional.kappa,
                "sparse_incidence_pi": provisional.pi,
                "sparse_incidence_active_mask": provisional.phase_occupied,
                "sparse_incidence_module_centres": provisional.module_centres,
                "sparse_incidence_environment_centres": provisional.environment_centres,
                "sparse_incidence_joint_centres": provisional.joint_centres,
            }
        )
        return state

    def _compact_query_keys(
        self,
        parent_query_keys: torch.Tensor,
        membership: torch.Tensor,
        roots: torch.Tensor,
        plan: GroupFusionPlan,
    ) -> torch.Tensor:
        summed = torch.einsum("bkr,bkd->brd", membership, parent_query_keys)
        mean = summed / plan.multiplicity.clamp_min(1).to(summed.dtype)[..., None]
        merged_keys = self.router._rms_scale(mean)
        singleton = plan.valid & (plan.multiplicity == 1)
        parent_keys = self._gather_class_values(parent_query_keys, roots)
        keys = torch.where(singleton[..., None], parent_keys, merged_keys)
        return keys * plan.valid[..., None].to(keys.dtype)

    def preparation_aux(
        self,
        state: dict[str, Any],
        *,
        include_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        summary = super().preparation_aux(
            state, include_diagnostics=include_diagnostics
        )
        plan = state.get("coalescence_plan")
        if not isinstance(plan, GroupFusionPlan):
            return summary
        identity_rows = state["coalescence_parent_identity_rows"]
        summary.update(
            {
                "coalescence_k_proposal": plan.active_proposal_count.detach(),
                "coalescence_k_case": plan.case_group_count.detach(),
                "coalescence_multiplicity": plan.multiplicity.detach(),
                "coalescence_fusion_strength": plan.fusion_strength.detach(),
                "coalescence_sigma": plan.sigma.detach(),
                "coalescence_lambda": plan.lambda_b.detach(),
                "coalescence_merge_displacement": plan.merge_displacement.detach(),
                "coalescence_projection_displacement": plan.projection_displacement.detach(),
                "coalescence_primal_residual": plan.primal_residual.detach(),
                "coalescence_dual_residual": plan.dual_residual.detach(),
                "coalescence_solver_converged": plan.solver_converged.detach(),
                "coalescence_solver_iterations": plan.solver_iterations.detach(),
                "coalescence_primal_tolerance": plan.primal_tolerance.detach(),
                "coalescence_dual_tolerance": plan.dual_tolerance.detach(),
                "coalescence_parent_identity_rows": identity_rows.detach(),
                "coalescence_provisional_phase_occupied": state[
                    "sparse_incidence_active_mask"
                ].detach(),
                "coalescence_valid_packed": plan.valid.detach(),
            }
        )
        summary["sparse_incidence_phase_occupied"] = state[
            "sparse_incidence_active_mask"
        ].detach()
        if include_diagnostics:
            controls = state["group_control_state"]
            if isinstance(controls, ConvergedCoalescedPreparedGroupControl):
                module_incidence = controls.module_membership
                environment_incidence = controls.environment_membership
                module_moment = controls.module_source_moment
                environment_moment = controls.environment_source_moment
            else:
                module_incidence = controls.module_membership
                environment_incidence = controls.environment_membership
                module_moment = (
                    module_incidence[..., None] * controls.group_control[:, None, :, :]
                )
                environment_moment = (
                    environment_incidence[..., None]
                    * controls.group_control[:, None, :, :]
                )
            summary.update(
                {
                    "coalescence_fused_descriptors": plan.fused_descriptors.detach(),
                    "coalescence_provisional_descriptors": plan.input_descriptors.detach(),
                    "coalescence_class_index": plan.class_index.detach(),
                    "coalescence_pair_fused": plan.pair_fused.detach(),
                    "coalescence_footprint_weights": state[
                        "coalescence_footprint_weights"
                    ].detach(),
                    "coalescence_module_incidence": module_incidence.detach(),
                    "coalescence_environment_incidence": environment_incidence.detach(),
                    "coalescence_module_source_moment": module_moment.detach(),
                    "coalescence_environment_source_moment": environment_moment.detach(),
                    "coalescence_query_keys": state.get(
                        "coalescence_query_keys", plan.parent_query_keys
                    ).detach()
                    if state.get("coalescence_query_keys", plan.parent_query_keys)
                    is not None
                    else plan.fused_descriptors.new_zeros(
                        plan.fused_descriptors.shape[0],
                        plan.fused_descriptors.shape[1],
                        plan.control_dim,
                    ),
                    "coalescence_routing_module_centers": state.get(
                        "coalescence_module_query_centers",
                        plan.parent_module_query_centers,
                    ).detach(),
                    "coalescence_routing_environment_centers": state.get(
                        "coalescence_environment_query_centers",
                        plan.parent_environment_query_centers,
                    ).detach(),
                    "coalescence_provisional_module_incidence": state.get(
                        "coalescence_provisional_module_incidence",
                        controls.module_membership,
                    ).detach(),
                    "coalescence_provisional_environment_incidence": state.get(
                        "coalescence_provisional_environment_incidence",
                        controls.environment_membership,
                    ).detach(),
                    "coalescence_provisional_group_control": state.get(
                        "coalescence_provisional_group_control", controls.group_control
                    ).detach(),
                }
            )
        return summary

    def _route(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
    ) -> CoalescedGroupQueryRoute:
        parent_route = super()._route(
            state, encoded, receivers, receiver_features
        )
        plan = state.get("coalescence_plan")
        if not isinstance(plan, GroupFusionPlan) or state.get(
            "coalescence_all_singleton", False
        ):
            return CoalescedGroupQueryRoute(
                query_control=parent_route.query_control,
                assignment=parent_route.assignment,
                logits=parent_route.logits,
                query_keys=parent_route.query_keys,
                query_mass=parent_route.assignment,
                query_density=parent_route.assignment,
            )
        scaled_query = self.router._rms_scale(parent_route.query_control)
        query_keys = state["coalescence_query_keys"]
        logits = torch.einsum("bqd,bkd->bqk", scaled_query, query_keys)
        logits = logits / (float(self.group_control_dim) ** 0.5)
        module_centers = state["coalescence_module_query_centers"]
        environment_centers = state["coalescence_environment_query_centers"]
        geometry = -0.5 * (
            torch.linalg.vector_norm(
                receivers[:, :, None, :] - module_centers[:, None, :, :], dim=-1
            )
            + torch.linalg.vector_norm(
                receivers[:, :, None, :] - environment_centers[:, None, :, :],
                dim=-1,
            )
        ) / plan.geometry_scale[:, None, None]
        logits = logits + geometry
        mass, density = weighted_sparsemax(
            logits, plan.multiplicity, plan.valid[:, None, :]
        )
        identity_rows = state["coalescence_parent_identity_rows"]
        assignment = self._batch_select(identity_rows, parent_route.assignment, density)
        logits = self._batch_select(identity_rows, parent_route.logits, logits)
        query_keys = self._batch_select(
            identity_rows, parent_route.query_keys, query_keys
        )
        mass = self._batch_select(identity_rows, parent_route.assignment, mass)
        density = self._batch_select(identity_rows, parent_route.assignment, density)
        return CoalescedGroupQueryRoute(
            query_control=parent_route.query_control,
            assignment=assignment,
            logits=logits,
            query_keys=query_keys,
            query_mass=mass,
            query_density=density,
        )

    def read(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._require_moment_safe_executor()
        read_state = state
        if state.get("coalescence_all_singleton", False):
            # Inherited QM/QE helpers use the plan marker to select source-B
            # contractions. This state is the exact parent state, so remove
            # only that dispatch marker for the parent read.
            read_state = dict(state)
            read_state.pop("coalescence_plan", None)
        context, aux = super().read(
            read_state,
            encoded,
            receivers,
            receiver_features,
            return_routing_maps=bool(return_routing_maps),
        )
        plan = state.get("coalescence_plan")
        if not isinstance(plan, GroupFusionPlan) or not return_routing_maps:
            return context, aux
        route = self._route(state, encoded, receivers, receiver_features)
        virtual_support = (
            (route.query_density > 0.0).to(plan.multiplicity.dtype)
            * plan.multiplicity[:, None, :]
        ).sum(dim=-1)
        coalesced_kq = (route.query_mass > 0.0).sum(dim=-1)
        aux.update(
            {
                "group_control_query_routing": route.query_mass.detach(),
                "group_control_query_density": route.query_density.detach(),
                "coalescence_query_mass": route.query_mass.detach(),
                "coalescence_query_density": route.query_density.detach(),
                "coalescence_query_logits": route.logits.detach(),
                "coalescence_query_kq": coalesced_kq.detach(),
                "coalescence_virtual_constituent_support": virtual_support.detach(),
                "sparse_incidence_query_routing": route.query_mass.detach(),
                "sparse_incidence_query_logits": route.logits.detach(),
                "sparse_incidence_query_keys": route.query_keys.detach(),
            }
        )
        return context, aux


__all__ = [
    "CoalescedGroupQueryRoute",
    "CoalescedPreparedGroupControl",
    "CoalescedSparseIncidencePairwiseField",
    "ConvergedCoalescedPreparedGroupControl",
    "ConvergedIdentityPreservingCoalescencePairwiseField",
]
