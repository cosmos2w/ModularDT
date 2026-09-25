"""Run-1503-v5 task-trained functional coalescence over the Run-1502 reader."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from honf_forward_core.organization.learned_functional_detail import (
    FunctionalDetailPlan,
    LearnedFunctionalDetailController,
)
from honf_forward_core.organization.group_fusion import weighted_sparsemax

from .group_control_router import GroupQueryRoute
from .sparse_incidence_group_control import SparseIncidenceGroupControlPairwiseField
from .sparse_incidence_router import (
    SparseIncidenceGroupRouter,
    SparseIncidencePreparedGroupControl,
)
from .routing_index.sparse_projection import masked_sparsemax
from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class TaskTrainedFunctionalPreparedGroupControl(
    SparseIncidencePreparedGroupControl
):
    """Prepared controls with source-resolved incidence/control moments."""

    module_source_moment: torch.Tensor
    environment_source_moment: torch.Tensor


@dataclass(frozen=True)
class TaskTrainedFunctionalPlan:
    """Phase-local transform and exact compact quotient metadata."""

    parent_controls: SparseIncidencePreparedGroupControl
    detail: FunctionalDetailPlan
    membership: torch.Tensor
    multiplicity: torch.Tensor
    valid: torch.Tensor
    class_id: torch.Tensor
    compact: bool


@dataclass(frozen=True, kw_only=True)
class TaskTrainedFunctionalQueryRoute(GroupQueryRoute):
    """Query density consumed by the reader, with its mass retained for maps."""

    query_mass: torch.Tensor
    query_density: torch.Tensor


class TaskTrainedFunctionalCoalescencePairwiseField(
    SparseIncidenceGroupControlPairwiseField
):
    """Conditional functional detail controller with exact moment compaction.

    Training and its rectangular reference use all twelve virtual functions.
    Deterministic evaluation packs only nodes whose retention is exactly zero.
    Both paths use the Run-1502 fine physical reader and original source rows.
    """

    phase_diagnostic_prefix = ("functional_detail_", "coalescence_", "sparse_incidence_", "group_control_")
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
        functional_tree_subsets: list[list[int]] | tuple[tuple[int, ...], ...] | torch.Tensor,
        functional_detail_hidden_dim: int = 32,
        functional_detail_initial_logit: float = 1.6,
        functional_detail_inference_mode: str = "auto",
    ) -> None:
        if int(group_count) != 12 or int(group_control_dim) != 16:
            raise ValueError("task-trained functional coalescence requires K=12 and D=16.")
        if functional_detail_inference_mode not in {"auto", "compact", "virtual"}:
            raise ValueError(
                "functional_detail_inference_mode must be one of auto, compact, or virtual."
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
            environment_refinement_normalizer=environment_refinement_normalizer,
        )
        membership = self._membership_tensor(functional_tree_subsets, group_count)
        leaf_feature_dim = 2 * group_control_dim + 2 + 2 + 1 + 2 * spatial_dim + 1 + group_control_dim
        descriptor_dim = 3 * leaf_feature_dim + 4
        # Controller initialization must not consume the parent's initialization
        # RNG stream, so the Run-1502 neural tensors retain their seeded values.
        with torch.random.fork_rng(devices=[]):
            self.functional_detail_controller = LearnedFunctionalDetailController(
                membership,
                descriptor_dim,
                hidden_dim=functional_detail_hidden_dim,
                initial_logit=functional_detail_initial_logit,
            )
        self.functional_detail_inference_mode = str(functional_detail_inference_mode)
        self._training_epoch: int | None = None
        self._training_total_epochs: int | None = None

    @staticmethod
    def _membership_tensor(
        tree: list[list[int]] | tuple[tuple[int, ...], ...] | torch.Tensor,
        group_count: int,
    ) -> torch.Tensor:
        if isinstance(tree, torch.Tensor):
            if tree.ndim != 2 or tuple(tree.shape[1:]) != (int(group_count),):
                raise ValueError("functional_tree_subsets tensor must have shape [N,K].")
            return tree.to(dtype=torch.bool)
        membership = torch.zeros((len(tree), int(group_count)), dtype=torch.bool)
        for row, subset in enumerate(tree):
            if not subset:
                raise ValueError("functional tree nodes cannot be empty.")
            indices = [int(index) for index in subset]
            if any(index < 0 or index >= int(group_count) for index in indices):
                raise ValueError("functional tree leaf index is outside [0,K).")
            if len(set(indices)) != len(indices):
                raise ValueError("functional tree node contains duplicate leaf indices.")
            membership[row, indices] = True
        return membership

    @property
    def functional_tree_membership(self) -> torch.Tensor:
        return self.functional_detail_controller.node_membership

    def set_training_progress(
        self, *, epoch: int, total_epochs: int | None = None
    ) -> None:
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a nonnegative integer.")
        if total_epochs is not None and (
            isinstance(total_epochs, bool) or int(total_epochs) <= 0
        ):
            raise ValueError("total_epochs must be positive when provided.")
        self._training_epoch = int(epoch)
        self._training_total_epochs = None if total_epochs is None else int(total_epochs)

    def selection_state(self) -> dict[str, int | None]:
        return {"epoch": self._training_epoch, "total_epochs": self._training_total_epochs}

    def _ramp(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._training_epoch is None:
            raise RuntimeError(
                "task_trained_functional_coalescence_honf requires "
                "set_training_progress() before prepare/evaluation."
            )
        position = (float(self._training_epoch) - 50.0) / 100.0
        position = min(max(position, 0.0), 1.0)
        smooth = position**3 * (position * (6.0 * position - 15.0) + 10.0)
        return torch.tensor(smooth, device=device, dtype=dtype)

    def _leaf_descriptors(
        self,
        encoded: EncodedInterfaceCase,
        controls: SparseIncidencePreparedGroupControl,
    ) -> torch.Tensor:
        if not isinstance(self.router, SparseIncidenceGroupRouter):
            raise TypeError("task-trained coalescence requires SparseIncidenceGroupRouter.")
        dtype = controls.group_control.dtype
        device = controls.group_control.device
        query_keys = self.router.normalized_query_key_bank(controls)
        total_mass = (
            controls.module_mass + controls.environment_mass
        ).sum(dim=-1, keepdim=True)
        safe_total = total_mass.clamp_min(torch.finfo(dtype).eps)
        module_share = controls.module_mass / safe_total
        environment_share = controls.environment_mass / safe_total
        mass_logs = torch.log1p(
            torch.stack([controls.module_mass, controls.environment_mass], dim=-1).clamp_min(0.0)
        )
        geometry_scale = self.router._geometry_scale(encoded).to(device=device, dtype=dtype)
        safe_geometry_scale = geometry_scale.clamp_min(torch.finfo(dtype).eps)
        module_centres = controls.module_centres / safe_geometry_scale[:, None, None]
        environment_centres = controls.environment_centres / safe_geometry_scale[:, None, None]
        module_centres = torch.where(
            (controls.module_mass > 0.0)[..., None],
            module_centres,
            torch.zeros_like(module_centres),
        )
        environment_centres = torch.where(
            (controls.environment_mass > 0.0)[..., None],
            environment_centres,
            torch.zeros_like(environment_centres),
        )
        active = controls.phase_occupied.to(dtype=dtype)
        global_control = controls.global_control[:, None, :].expand(
            -1, self.group_count, -1
        )
        return torch.cat(
            [
                query_keys,
                controls.group_control,
                module_share[..., None],
                environment_share[..., None],
                mass_logs,
                controls.pi[..., None],
                module_centres,
                environment_centres,
                active[..., None],
                global_control,
            ],
            dim=-1,
        )

    def _node_descriptors(
        self,
        encoded: EncodedInterfaceCase,
        controls: SparseIncidencePreparedGroupControl,
    ) -> torch.Tensor:
        leaf = self._leaf_descriptors(encoded, controls)
        active = controls.phase_occupied.to(dtype=leaf.dtype)
        child_masks = self.functional_detail_controller.child_masks.to(
            device=leaf.device
        )
        valid_child = child_masks[None, :, :, :] & active[:, None, None, :].bool()
        counts = valid_child.sum(dim=-1).to(dtype=leaf.dtype)
        sums = torch.einsum(
            "bnck,bkf->bncf", valid_child.to(leaf.dtype), leaf
        )
        means = sums / counts.clamp_min(1.0)[..., None]
        means = torch.where(counts[..., None] > 0.0, means, torch.zeros_like(means))
        centered = leaf[:, None, None, :, :] - means[..., None, :]
        squared = centered.square() * valid_child[..., None].to(leaf.dtype)
        variances = squared.sum(dim=-2) / counts.clamp_min(1.0)[..., None]
        variances = torch.where(
            counts[..., None] > 0.0, variances, torch.zeros_like(variances)
        )
        left_mean, right_mean = means.unbind(dim=2)
        left_variance, right_variance = variances.unbind(dim=2)
        left_count, right_count = counts.unbind(dim=2)
        static_child_sizes = child_masks.sum(dim=-1).to(dtype=leaf.dtype)
        left_size, right_size = static_child_sizes.unbind(dim=1)
        scale = float(self.group_count)
        symmetric = torch.cat(
            [
                left_mean + right_mean,
                (left_mean - right_mean).square(),
                left_variance + right_variance,
                ((left_size + right_size) / scale)[None, :, None].expand(
                    leaf.shape[0], -1, -1
                ),
                ((left_size - right_size).square() / (scale * scale))[None, :, None].expand(
                    leaf.shape[0], -1, -1
                ),
                ((left_count + right_count) / scale)[..., None],
                ((left_count - right_count).square() / (scale * scale))[..., None],
            ],
            dim=-1,
        )
        return symmetric

    def _source_moments(
        self,
        controls: SparseIncidencePreparedGroupControl,
        membership: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        module_incidence = torch.einsum(
            "bsk,bkr->bsr", controls.module_membership, membership
        )
        environment_incidence = torch.einsum(
            "bsk,bkr->bsr", controls.environment_membership, membership
        )
        module_moment = torch.einsum(
            "bsk,bkd,bkr->bsrd",
            controls.module_membership,
            controls.group_control,
            membership,
        )
        environment_moment = torch.einsum(
            "bsk,bkd,bkr->bsrd",
            controls.environment_membership,
            controls.group_control,
            membership,
        )
        return module_incidence, environment_incidence, module_moment, environment_moment

    def _make_prepared_controls(
        self,
        encoded: EncodedInterfaceCase,
        provisional: SparseIncidencePreparedGroupControl,
        membership: torch.Tensor,
        multiplicity: torch.Tensor,
        valid: torch.Tensor,
        compact: bool,
    ) -> tuple[TaskTrainedFunctionalPreparedGroupControl, torch.Tensor, torch.Tensor]:
        if compact:
            (
                module_incidence,
                environment_incidence,
                module_moment,
                environment_moment,
            ) = self._source_moments(provisional, membership)
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
            ) / multiplicity.clamp_min(1).to(provisional.group_control.dtype)[..., None]
            mean_control = mean_control * valid[..., None].to(mean_control.dtype)
            pi = torch.einsum("bk,bkr->br", provisional.pi, membership)
            proposal_module_mass = torch.einsum(
                "bk,bkr->br", provisional.proposal_module_mass, membership
            )
            proposal_environment_mass = torch.einsum(
                "bk,bkr->br", provisional.proposal_environment_mass, membership
            )
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
                pi=pi,
                proposal_module_mass=proposal_module_mass,
                proposal_environment_mass=proposal_environment_mass,
                proposal_occupied=valid,
            )
        else:
            # Virtual training keeps the original K columns and the parent
            # provisional kappa.  The B banks are the unchanged A_k h_k moments.
            module_incidence = provisional.module_membership
            environment_incidence = provisional.environment_membership
            module_moment = torch.einsum(
                "bsk,bkd->bskd", module_incidence, provisional.group_control
            )
            environment_moment = torch.einsum(
                "bsk,bkd->bskd", environment_incidence, provisional.group_control
            )
            controls_base = provisional
        controls = TaskTrainedFunctionalPreparedGroupControl(
            **{
                name: getattr(controls_base, name)
                for name in SparseIncidencePreparedGroupControl.__dataclass_fields__
            },
            module_source_moment=module_moment,
            environment_source_moment=environment_moment,
        )
        return controls, module_incidence, module_moment

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
        if not isinstance(controls, TaskTrainedFunctionalPreparedGroupControl):
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
        if isinstance(controls, TaskTrainedFunctionalPreparedGroupControl):
            cached = state.get("module_control_bank")
            if state.get("module_control_bank_source_moment") is controls.module_source_moment:
                return cached
            batch, sources, _, width = controls.module_source_moment.shape
            return controls.module_source_moment.permute(0, 2, 1, 3).reshape(
                batch,
                int(controls.module_membership.shape[-1]),
                sources * width,
            )
        return super()._module_control_bank(state)

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
        functional_detail_stochastic_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if not isinstance(self.router, SparseIncidenceGroupRouter):
            raise TypeError("task-trained coalescence requires SparseIncidenceGroupRouter.")
        if self._training_epoch is None:
            raise RuntimeError(
                "task-trained coalescence requires set_training_progress() before preparation."
            )
        batch = int(module_states.shape[0])
        if self._training_epoch <= 50:
            # Do not run the new head or alter the parent operation/RNG stream
            # during the exact Run-1502 warmup interval.
            state = SparseIncidenceGroupControlPairwiseField.prepare(
                self,
                encoded,
                module_states,
                return_routing_maps=bool(return_routing_maps),
            )
            controls = state["group_control_state"]
            active_count = controls.phase_occupied.sum(dim=-1).to(module_states.dtype)
            zeros = torch.zeros_like(active_count)
            state.update(
                {
                    "functional_detail_actual_R": active_count.detach(),
                    "functional_detail_expected_R": active_count,
                    "functional_detail_expected_complexity": zeros,
                    "functional_detail_execution_width": module_states.new_full(
                        (batch,), float(self.group_count)
                    ),
                }
            )
            return state

        dense_encoded = replace(
            encoded,
            coordinate_scale=self._dense_coordinate_scale(encoded, batch),
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
        provisional = self.router.prepare(encoded, fine["module_tokens"], contextual_env)
        descriptors = self._node_descriptors(encoded, provisional)
        if functional_detail_stochastic_mask is None:
            stochastic_mask = torch.full(
                (batch,), bool(self.training), device=module_states.device, dtype=torch.bool
            )
        else:
            stochastic_mask = torch.as_tensor(
                functional_detail_stochastic_mask,
                device=module_states.device,
                dtype=torch.bool,
            )
            if tuple(stochastic_mask.shape) != (batch,):
                raise ValueError("functional_detail_stochastic_mask must have shape [B].")
        detail = self.functional_detail_controller.plan(
            descriptors,
            provisional.phase_occupied,
            provisional.pi,
            stochastic_mask=stochastic_mask,
            ramp=self._ramp(module_states.device, module_states.dtype),
        )

        if self.training:
            compact = False
        elif self.functional_detail_inference_mode == "compact":
            compact = True
        else:
            # The bounded Q8192 timing comparison favored the virtual policy
            # on both measured receiver chunks. Keep exact compact inference
            # available as an explicit option.
            compact = False
        if compact:
            packed = self.functional_detail_controller.pack(
                detail, provisional.phase_occupied
            )
            membership = packed.membership.to(
                device=module_states.device, dtype=module_states.dtype
            )
            multiplicity = packed.multiplicity
            valid = packed.valid
            class_id = packed.class_id
        else:
            eye = torch.eye(
                self.group_count, device=module_states.device, dtype=module_states.dtype
            )[None, :, :].expand(batch, -1, -1)
            membership = eye * provisional.phase_occupied[..., None].to(eye.dtype)
            multiplicity = provisional.phase_occupied.to(torch.long)
            valid = provisional.phase_occupied
            group_index = torch.arange(
                self.group_count, device=module_states.device, dtype=torch.long
            )[None, :].expand(batch, -1)
            class_id = torch.where(
                provisional.phase_occupied,
                group_index,
                torch.full_like(group_index, -1),
            )

        if compact:
            controls, module_incidence, module_moment = self._make_prepared_controls(
                encoded, provisional, membership, multiplicity, valid, compact=True
            )
        else:
            # The virtual training/reference path preserves the original
            # Run-1502 controls and reader banks.  Only query logits change.
            controls = provisional
            module_incidence = provisional.module_membership
            module_moment = None
        plan = TaskTrainedFunctionalPlan(
            parent_controls=provisional,
            detail=detail,
            membership=membership,
            multiplicity=multiplicity,
            valid=valid,
            class_id=class_id,
            compact=compact,
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
                "functional_detail_plan": plan,
                "functional_detail_actual_R": detail.actual_R,
                "functional_detail_expected_R": detail.expected_R,
                "functional_detail_expected_complexity": detail.expected_complexity,
                "functional_detail_stochastic_mask": detail.stochastic_mask,
                "functional_detail_ramp": detail.ramp,
                "functional_detail_execution_width": module_states.new_full(
                    (batch,), float(membership.shape[-1])
                ),
                "sparse_incidence_kappa": provisional.kappa,
                "sparse_incidence_pi": provisional.pi,
                "sparse_incidence_active_mask": provisional.phase_occupied,
                "sparse_incidence_module_centres": provisional.module_centres,
                "sparse_incidence_environment_centres": provisional.environment_centres,
                "sparse_incidence_joint_centres": provisional.joint_centres,
            }
        )
        if compact:
            if not isinstance(controls, TaskTrainedFunctionalPreparedGroupControl):
                raise RuntimeError("compact inference requires prepared source moments.")
            module_control_bank = controls.module_source_moment.permute(
                0, 2, 1, 3
            ).reshape(
                batch,
                int(module_incidence.shape[-1]),
                int(module_states.shape[1]) * self.group_control_dim,
            )
            state.update(
                {
                    "coalescence_plan": plan,
                    "module_control_bank": module_control_bank,
                    "module_control_bank_membership": controls.module_membership,
                    "module_control_bank_group_control": controls.group_control,
                    "module_control_bank_source_moment": controls.module_source_moment,
                }
            )
        aux = state["group_control_preparation_aux"]
        aux.update(
            {
                "coalescence_k_proposal": provisional.phase_occupied.sum(dim=-1).detach(),
                "coalescence_k_case": detail.actual_R.detach(),
                "coalescence_multiplicity": multiplicity.detach(),
                "coalescence_valid_packed": valid.detach(),
                "functional_detail_actual_R": detail.actual_R.detach(),
                "functional_detail_expected_R": detail.expected_R.detach(),
                "functional_detail_expected_complexity": detail.expected_complexity.detach(),
                "functional_detail_retention": detail.retention.detach(),
                "functional_detail_probability": detail.probability.detach(),
                "functional_detail_closed_nodes": detail.closed_nodes.detach(),
            }
        )
        return state

    def _route(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
    ) -> TaskTrainedFunctionalQueryRoute:
        plan = state.get("functional_detail_plan")
        if not isinstance(plan, TaskTrainedFunctionalPlan):
            parent = super()._route(state, encoded, receivers, receiver_features)
            return TaskTrainedFunctionalQueryRoute(
                query_control=parent.query_control,
                assignment=parent.assignment,
                logits=parent.logits,
                query_keys=parent.query_keys,
                query_mass=parent.assignment,
                query_density=parent.assignment,
            )
        parent_query_control, parent_logits, parent_query_keys = self.router.query_logits(
            encoded, plan.parent_controls, receivers, receiver_features
        )
        logits_virtual = torch.einsum(
            "bqk,bik->bqi", parent_logits, plan.detail.transform
        )
        if not plan.compact:
            assignment = masked_sparsemax(
                logits_virtual,
                plan.parent_controls.phase_occupied[:, None, :].expand_as(logits_virtual),
            )
            query_keys = torch.einsum(
                "bik,bkd->bid", plan.detail.transform, parent_query_keys
            )
            return TaskTrainedFunctionalQueryRoute(
                query_control=parent_query_control,
                assignment=assignment,
                logits=logits_virtual,
                query_keys=query_keys,
                query_mass=assignment,
                query_density=assignment,
            )

        roots = plan.membership.argmax(dim=1)
        representative_transform = plan.detail.transform.gather(
            1,
            roots[..., None].expand(
                -1, -1, int(plan.detail.transform.shape[-1])
            ),
        )
        compact_logits = torch.einsum(
            "bqk,brk->bqr", parent_logits, representative_transform
        )
        mass, density = weighted_sparsemax(
            compact_logits,
            plan.multiplicity,
            plan.valid[:, None, :],
        )
        query_keys = torch.einsum(
            "brk,bkd->brd", representative_transform, parent_query_keys
        )
        return TaskTrainedFunctionalQueryRoute(
            query_control=parent_query_control,
            assignment=density,
            logits=compact_logits,
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
        plan = state.get("functional_detail_plan")
        if not isinstance(plan, TaskTrainedFunctionalPlan):
            return summary
        detail = plan.detail
        summary.update(
            {
                "coalescence_k_proposal": plan.parent_controls.phase_occupied.sum(
                    dim=-1
                ).detach(),
                "coalescence_k_case": detail.actual_R.detach(),
                "coalescence_multiplicity": plan.multiplicity.detach(),
                "coalescence_valid_packed": plan.valid.detach(),
                "functional_detail_actual_R": detail.actual_R.detach(),
                "functional_detail_expected_R": detail.expected_R.detach(),
                "functional_detail_expected_complexity": detail.expected_complexity.detach(),
                "functional_detail_ramp": detail.ramp.detach(),
                "functional_detail_retention": detail.retention.detach(),
                "functional_detail_probability": detail.probability.detach(),
                "functional_detail_closed_nodes": detail.closed_nodes.detach(),
            }
        )
        if include_diagnostics:
            controls = state["group_control_state"]
            if isinstance(controls, TaskTrainedFunctionalPreparedGroupControl):
                module_source_moment = controls.module_source_moment
                environment_source_moment = controls.environment_source_moment
            else:
                # The virtual reference deliberately retains the Run-1502
                # reader banks in its hot path. Materialize equivalent source
                # moments only for an explicit diagnostic request.
                module_source_moment = torch.einsum(
                    "bsk,bkd->bskd",
                    controls.module_membership,
                    controls.group_control,
                )
                environment_source_moment = torch.einsum(
                    "bsk,bkd->bskd",
                    controls.environment_membership,
                    controls.group_control,
                )
            summary.update(
                {
                    "functional_detail_transform": detail.transform.detach(),
                    "functional_detail_class_index": plan.class_id.detach(),
                    "functional_detail_membership": plan.membership.detach(),
                    "functional_detail_module_incidence": controls.module_membership.detach(),
                    "functional_detail_environment_incidence": controls.environment_membership.detach(),
                    "functional_detail_module_source_moment": module_source_moment.detach(),
                    "functional_detail_environment_source_moment": environment_source_moment.detach(),
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
        if return_routing_maps and isinstance(
            state.get("functional_detail_plan"), TaskTrainedFunctionalPlan
        ):
            route = self._route(state, encoded, receivers, receiver_features)
            aux.update(
                {
                    "functional_detail_query_mass": route.query_mass.detach(),
                    "functional_detail_query_density": route.query_density.detach(),
                    "functional_detail_query_logits": route.logits.detach(),
                }
            )
        return context, aux


__all__ = [
    "TaskTrainedFunctionalCoalescencePairwiseField",
    "TaskTrainedFunctionalPlan",
    "TaskTrainedFunctionalPreparedGroupControl",
]
