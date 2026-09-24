"""Continuous access-function contraction with exact source-moment packing.

Run 1503-v4 keeps the Run 1502 fine physical reader.  A fixed tree transforms
the parent's finite query logits; only exactly equal transformed functions are
packed.  The source incidence and source-control moment remain separate banks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from honf_forward_core.organization.functional_fusion_tree import (
    FunctionalProbeCatalogue,
    build_source_action_grams,
    build_tree_transform,
    combined_post_transform_discrepancy,
    contracted_logits,
    node_contraction,
    node_source_action_scores,
    pack_exact_closed_subtrees,
)
from honf_forward_core.organization.group_fusion import weighted_sparsemax

from .coalesced_sparse_incidence import (
    CoalescedGroupQueryRoute,
    ConvergedCoalescedPreparedGroupControl,
    ConvergedIdentityPreservingCoalescencePairwiseField,
)
from .core import EncodedInterfaceCase
from .sparse_incidence_group_control import SparseIncidenceGroupControlPairwiseField
from .sparse_incidence_router import (
    SparseIncidenceGroupRouter,
    SparseIncidencePreparedGroupControl,
)


@dataclass(frozen=True)
class FunctionalTreePlan:
    """Per-phase live access transform and exact integer quotient metadata."""

    parent_controls: SparseIncidencePreparedGroupControl
    transform: torch.Tensor
    membership: torch.Tensor
    multiplicity: torch.Tensor
    valid: torch.Tensor
    class_id: torch.Tensor
    node_score2: torch.Tensor
    node_gamma: torch.Tensor
    node_residual_scale: torch.Tensor
    closed_nodes: torch.Tensor
    combined_score2: torch.Tensor
    node_scores: Any
    combined_scores: Any


class ContinuousFunctionalCoalescencePairwiseField(
    ConvergedIdentityPreservingCoalescencePairwiseField
):
    """Solver-free v4 quotient over continuously contracted access functions."""

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
        functional_tree_subsets: list[list[int]] | tuple[tuple[int, ...], ...],
        read_input_close_rms: float = 0.02,
        read_input_keep_rms: float = 0.06,
    ) -> None:
        if int(group_count) != 12 or int(group_control_dim) != 16:
            raise ValueError("v4 requires K=12 and D=16.")
        # Skip the v3 ADMM constructor, retaining its source-moment methods
        # and the identical parent neural parameter initialization order.
        SparseIncidenceGroupControlPairwiseField.__init__(
            self,
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
        nodes = torch.zeros((len(functional_tree_subsets), group_count), dtype=torch.bool)
        for row, subset in enumerate(functional_tree_subsets):
            nodes[row, list(subset)] = True
        if tuple(nodes.shape) != (11, 12) or not bool(nodes[-1].all()):
            raise ValueError("v4 requires an eleven-node tree ending at all twelve leaves.")
        self.register_buffer("functional_tree_membership", nodes, persistent=True)
        self.read_input_close_rms = float(read_input_close_rms)
        self.read_input_keep_rms = float(read_input_keep_rms)
        self._training_epoch: int | None = None
        self._training_total_epochs: int | None = None

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        functional_probes: FunctionalProbeCatalogue | None = None,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(self.router, SparseIncidenceGroupRouter):
            raise TypeError("v4 requires SparseIncidenceGroupRouter.")
        if self._training_epoch is None:
            raise RuntimeError("v4 requires set_training_progress before preparation.")
        if self._training_epoch <= 50:
            # Exact parent calculation, including neural RNG and first gradients.
            return SparseIncidenceGroupControlPairwiseField.prepare(
                self, encoded, module_states, return_routing_maps=return_routing_maps
            )
        if functional_probes is None:
            raise ValueError("active v4 preparation requires phase-local functional probes.")

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
        catalogue = functional_probes.with_environment(
            encoded.env_coords, encoded.env_weights
        )
        logits_by_role = {
            role: self.router.route_queries(
                encoded, provisional, family.coordinates
            ).logits
            for role, family in catalogue.families().items()
        }
        grams = build_source_action_grams(
            provisional.module_membership,
            provisional.module_measure,
            provisional.environment_membership,
            provisional.environment_measure,
            provisional.group_control,
            self.module_control_gain,
            self.environment_score_control,
        )
        node_scores = node_source_action_scores(
            logits_by_role,
            provisional.phase_occupied,
            grams,
            catalogue,
            self.functional_tree_membership,
        )
        contraction = node_contraction(
            node_scores.score2,
            provisional.pi,
            self.functional_tree_membership,
            provisional.phase_occupied,
            self._training_epoch,
            close_rms=self.read_input_close_rms,
            keep_rms=self.read_input_keep_rms,
        )
        transform = build_tree_transform(
            contraction.residual_scale, self.functional_tree_membership
        )
        closed = contraction.gamma == 1.0
        packed = pack_exact_closed_subtrees(
            closed, self.functional_tree_membership, provisional.phase_occupied
        )
        transformed_by_role = {
            role: contracted_logits(logits, transform)
            for role, logits in logits_by_role.items()
        }
        combined = combined_post_transform_discrepancy(
            logits_by_role,
            transformed_by_role,
            provisional.phase_occupied,
            grams,
            catalogue,
        )
        membership = packed.membership.to(dtype=provisional.module_membership.dtype)
        multiplicity = packed.multiplicity
        valid = packed.valid
        module_incidence = torch.einsum(
            "bsk,bkr->bsr", provisional.module_membership, membership
        )
        environment_incidence = torch.einsum(
            "bsk,bkr->bsr", provisional.environment_membership, membership
        )
        module_moment = torch.einsum(
            "bsk,bkd,bkr->bsrd",
            provisional.module_membership,
            provisional.group_control,
            membership,
        )
        environment_moment = torch.einsum(
            "bsk,bkd,bkr->bsrd",
            provisional.environment_membership,
            provisional.group_control,
            membership,
        )
        (
            module_mass,
            environment_mass,
            module_centres,
            environment_centres,
            joint_centres,
        ) = self.router._joint_centres(
            module_incidence,
            environment_incidence,
            provisional.module_measure,
            provisional.environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        mean_control = torch.einsum(
            "bkr,bkd->brd", membership, provisional.group_control
        ) / multiplicity.clamp_min(1).to(module_states.dtype)[..., None]
        mean_control = mean_control * valid[..., None].to(mean_control.dtype)
        controls_base = replace(
            provisional,
            module_membership=module_incidence,
            environment_membership=environment_incidence,
            module_mass=module_mass,
            environment_mass=environment_mass,
            group_control=mean_control,
            phase_occupied=valid,
            module_centres=module_centres,
            environment_centres=environment_centres,
            joint_centres=joint_centres,
        )
        plan = FunctionalTreePlan(
            parent_controls=provisional,
            transform=transform,
            membership=membership,
            multiplicity=multiplicity,
            valid=valid,
            class_id=packed.class_id,
            node_score2=node_scores.score2,
            node_gamma=contraction.gamma,
            node_residual_scale=contraction.residual_scale,
            closed_nodes=closed,
            combined_score2=combined.score2,
            node_scores=node_scores,
            combined_scores=combined,
        )
        # The inherited v3 moment reader uses this type and the plan marker,
        # but no v3 solver or hard centroid/key reconstruction is invoked.
        controls = ConvergedCoalescedPreparedGroupControl(
            **{
                name: getattr(controls_base, name)
                for name in SparseIncidencePreparedGroupControl.__dataclass_fields__
            },
            fusion_plan=plan,  # type: ignore[arg-type]
            module_source_moment=module_moment,
            environment_source_moment=environment_moment,
            parent_controls=provisional,
            parent_identity_rows=None,
            identity_free_quotient=True,
        )
        state = self._prepare_from_fine(
            encoded,
            module_states,
            fine,
            contextual_env,
            controls,
            return_routing_maps=return_routing_maps,
        )
        state["module_control_bank"] = module_moment.permute(0, 2, 1, 3).reshape(
            int(module_states.shape[0]),
            int(multiplicity.shape[1]),
            int(module_states.shape[1]) * self.group_control_dim,
        )
        state["module_control_bank_membership"] = controls.module_membership
        state["module_control_bank_group_control"] = controls.group_control
        state["module_control_bank_source_moment"] = module_moment
        state["functional_tree_plan"] = plan
        state["coalescence_plan"] = plan
        state["coalescence_all_singleton"] = False
        state["coalescence_parent_identity_rows"] = torch.zeros_like(valid[:, 0])
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

    def _route(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
    ) -> CoalescedGroupQueryRoute:
        plan = state.get("functional_tree_plan")
        if not isinstance(plan, FunctionalTreePlan):
            return super()._route(state, encoded, receivers, receiver_features)
        parent = self.router.route_queries(
            encoded, plan.parent_controls, receivers, receiver_features
        )
        roots = plan.membership.argmax(dim=1)
        compact_transform = plan.transform.gather(
            1,
            roots[..., None].expand(
                -1, -1, int(plan.transform.shape[-1])
            ),
        )
        logits = torch.einsum("bqk,brk->bqr", parent.logits, compact_transform)
        mass, density = weighted_sparsemax(
            logits, plan.multiplicity, plan.valid[:, None, :]
        )
        query_keys = (
            None
            if parent.query_keys is None
            else torch.einsum("brk,bkd->brd", compact_transform, parent.query_keys)
        )
        return CoalescedGroupQueryRoute(
            query_control=parent.query_control,
            assignment=density,
            logits=logits,
            query_keys=query_keys,
            query_mass=mass,
            query_density=density,
        )

    def preparation_aux(
        self, state: dict[str, Any], *, include_diagnostics: bool = False
    ) -> dict[str, torch.Tensor]:
        summary = super().preparation_aux(
            state, include_diagnostics=include_diagnostics
        )
        plan = state.get("functional_tree_plan")
        if not isinstance(plan, FunctionalTreePlan):
            return summary
        summary.update(
            {
                "coalescence_k_proposal": plan.parent_controls.phase_occupied.sum(
                    dim=-1
                ).detach(),
                "coalescence_k_case": plan.valid.sum(dim=-1).detach(),
                "coalescence_multiplicity": plan.multiplicity.detach(),
                "coalescence_transition_nodes": (
                    (plan.node_gamma > 0.0) & (plan.node_gamma < 1.0)
                ).sum(dim=-1).detach(),
                "coalescence_closed_nodes": plan.closed_nodes.sum(dim=-1).detach(),
                "coalescence_node_score2": plan.node_score2.detach(),
                "coalescence_node_gamma": plan.node_gamma.detach(),
                "coalescence_combined_score2": plan.combined_score2.detach(),
                "coalescence_valid_packed": plan.valid.detach(),
            }
        )
        if include_diagnostics:
            summary.update(
                {
                    "coalescence_transform": plan.transform.detach(),
                    "coalescence_class_index": plan.class_id.detach(),
                    "coalescence_module_incidence": state[
                        "group_control_state"
                    ].module_membership.detach(),
                    "coalescence_environment_incidence": state[
                        "group_control_state"
                    ].environment_membership.detach(),
                    "coalescence_module_source_moment": state[
                        "module_control_bank_source_moment"
                    ].detach(),
                    "coalescence_environment_source_moment": state[
                        "group_control_state"
                    ].environment_source_moment.detach(),
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
            return_routing_maps=return_routing_maps,
        )
        plan = state.get("functional_tree_plan")
        if not isinstance(plan, FunctionalTreePlan) or not return_routing_maps:
            return context, aux
        route = self._route(state, encoded, receivers, receiver_features)
        aux.update(
            {
                "group_control_query_routing": route.query_mass.detach(),
                "group_control_query_density": route.query_density.detach(),
                "coalescence_query_mass": route.query_mass.detach(),
                "coalescence_query_density": route.query_density.detach(),
                "coalescence_query_logits": route.logits.detach(),
                "coalescence_query_kq": (route.query_mass > 0.0).sum(dim=-1).detach(),
                "coalescence_virtual_constituent_support": (
                    (route.query_density > 0.0).to(plan.multiplicity.dtype)
                    * plan.multiplicity[:, None, :]
                ).sum(dim=-1).detach(),
                "sparse_incidence_query_routing": route.query_mass.detach(),
                "sparse_incidence_query_logits": route.logits.detach(),
            }
        )
        return context, aux


__all__ = ["ContinuousFunctionalCoalescencePairwiseField", "FunctionalTreePlan"]
