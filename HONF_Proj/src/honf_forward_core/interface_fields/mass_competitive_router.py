"""Mass-competitive adaptive routing for the opt-in Run-1500 core.

The router reuses the Run-1409 source controls and one-step geometry helpers,
but replaces exact-positive occupancy with one fixed case-level competition::

    pi = 0.5 * (mu_module + mu_environment)
    gamma = sparsemax(log(pi + 1e-8))

Only positive-``gamma`` prototype IDs form the case hypergraph.  The live P0
``gamma`` tensor is reused through P1/P2 so the physical objective can train
the source-mass competition; integer IDs and support masks are detached
metadata.  No gate network, count loss, temperature schedule, top-k rule, or
``2**K`` support table is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from honf_forward_core.routing import entmax15

from .group_control_router import GroupQueryRoute, PreparedGroupControl
from .occupancy_group_router import (
    OCCUPANCY_CONTROL_DIM,
    OCCUPANCY_KMAX,
    OccupancyGroupPlan,
    OccupancyGroupRouter,
)
from .routing_index.sparse_projection import masked_sparsemax
from .types import EncodedInterfaceCase

MASS_COMPETITIVE_EPSILON = 1.0e-8


def mass_competition(
    pi: torch.Tensor,
    *,
    epsilon: float = MASS_COMPETITIVE_EPSILON,
) -> torch.Tensor:
    """Apply the fixed Run-1500 ``sparsemax(log(pi + epsilon))`` rule."""

    if not isinstance(pi, torch.Tensor) or pi.ndim != 2:
        raise ValueError("pi must have shape [B,K].")
    if not pi.is_floating_point():
        raise TypeError("pi must be floating point.")
    if not bool(torch.isfinite(pi).all()) or bool((pi < 0.0).any()):
        raise ValueError("pi must be finite and nonnegative.")
    if not torch.isfinite(torch.tensor(float(epsilon))) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    total = pi.sum(dim=-1)
    if not bool(
        torch.allclose(
            total,
            torch.ones_like(total),
            atol=2.0e-5,
            rtol=0.0,
        )
    ):
        raise ValueError("every pi row must sum to one.")
    gamma = masked_sparsemax(torch.log(pi + float(epsilon)))
    if bool((gamma.sum(dim=-1) <= 0.0).any()):
        raise RuntimeError("mass competition produced an empty case hypergraph.")
    return gamma


def _safe_log_gamma(gamma: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Take logs only after zero-``gamma`` columns have been replaced."""

    safe = torch.where(active, gamma, torch.ones_like(gamma))
    return torch.log(safe)


@dataclass(frozen=True)
class MassCompetitiveGroupPlan(OccupancyGroupPlan):
    """Immutable P0 case plan reused through P1/P2.

    Continuous tensors remain connected to the P0 graph during training.
    Prototype IDs, packed validity, and ``k_plan`` are detached metadata.
    """

    gamma_p0: torch.Tensor | None = None
    pi_p0: torch.Tensor | None = None
    precompetition_module_mass_p0: torch.Tensor | None = None
    precompetition_environment_mass_p0: torch.Tensor | None = None
    precompetition_module_centres_p0: torch.Tensor | None = None
    precompetition_environment_centres_p0: torch.Tensor | None = None
    precompetition_joint_centres_p0: torch.Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        batch = int(self.prototype_ids.shape[0])
        kmax = int(self.registered_kmax)
        for name in (
            "gamma_p0",
            "pi_p0",
            "precompetition_module_mass_p0",
            "precompetition_environment_mass_p0",
        ):
            value = getattr(self, name)
            if not torch.is_tensor(value) or tuple(value.shape) != (batch, kmax):
                raise ValueError(f"{name} must have shape [B,Kmax].")
            if value.device != self.prototype_ids.device:
                raise ValueError(f"{name} must share the plan device.")
            if not bool(torch.isfinite(value).all()) or bool((value < 0.0).any()):
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in (
            "precompetition_module_centres_p0",
            "precompetition_environment_centres_p0",
            "precompetition_joint_centres_p0",
        ):
            value = getattr(self, name)
            if not torch.is_tensor(value) or value.ndim != 3:
                raise ValueError(f"{name} must have shape [B,Kmax,d].")
            if tuple(value.shape[:2]) != (batch, kmax):
                raise ValueError(f"{name} must align with [B,Kmax].")
            if value.device != self.prototype_ids.device or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite on the plan device.")
        if not bool(torch.allclose(self.gamma_p0.sum(dim=-1), torch.ones(batch, device=self.gamma_p0.device, dtype=self.gamma_p0.dtype), atol=2.0e-5, rtol=0.0)):
            raise ValueError("gamma_p0 rows must sum to one.")
        if not bool(torch.equal(self.gamma_p0 > 0.0, self.planned_mask)):
            raise ValueError("positive gamma support must equal the planned prototype IDs.")
        expected_kappa = self.gamma_p0.square().sum(dim=-1).reciprocal()
        if not bool(torch.allclose(self.kappa_p0, expected_kappa, atol=2.0e-5, rtol=2.0e-5)):
            raise ValueError("kappa_p0 must equal 1/sum(gamma**2).")

    @property
    def gamma(self) -> torch.Tensor:
        assert self.gamma_p0 is not None
        return self.gamma_p0

    @property
    def pi(self) -> torch.Tensor:
        assert self.pi_p0 is not None
        return self.pi_p0

    @property
    def k_case(self) -> torch.Tensor:
        return self.k_plan

    @property
    def K_case(self) -> torch.Tensor:
        return self.k_plan


@dataclass(frozen=True)
class MassCompetitivePreparedGroupControl(PreparedGroupControl):
    """Live phase-local controls plus the P0 mass-competition plan."""

    plan: MassCompetitiveGroupPlan
    prototype_ids: torch.Tensor
    packed_valid: torch.Tensor
    phase_occupied: torch.Tensor
    module_centres: torch.Tensor
    environment_centres: torch.Tensor
    joint_centres: torch.Tensor
    kappa: torch.Tensor
    gamma: torch.Tensor
    pi_precompetition: torch.Tensor
    proposal_module_mass: torch.Tensor
    proposal_environment_mass: torch.Tensor
    precompetition_module_mass: torch.Tensor
    precompetition_environment_mass: torch.Tensor

    @property
    def k_plan(self) -> torch.Tensor:
        return self.plan.k_plan

    @property
    def K_plan(self) -> torch.Tensor:
        return self.plan.k_plan

    @property
    def k_case(self) -> torch.Tensor:
        return self.plan.k_plan

    @property
    def phase_group_mask(self) -> torch.Tensor:
        return self.packed_valid


class MassCompetitiveGroupRouter(OccupancyGroupRouter):
    """Run-1500 source-mass competition and geometry-aware query router."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        group_count: int = OCCUPANCY_KMAX,
        control_dim: int = OCCUPANCY_CONTROL_DIM,
        fourier_frequencies: int = 4,
        spatial_dim: int = 2,
        module_temperature: float = 1.0,
        environment_temperature: float = 1.0,
        query_temperature: float = 1.0,
        geometry_fraction: float = 0.25,
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
            geometry_fraction=geometry_fraction,
        )

    def _selected_codes(
        self,
        plan: MassCompetitiveGroupPlan | None,
        *,
        compact: bool,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_codes = self.group_codes.to(device=reference.device, dtype=reference.dtype)
        if plan is None:
            ids = torch.arange(self.group_count, device=reference.device, dtype=torch.long)[None, :].expand(reference.shape[0], -1)
            return full_codes, ids, torch.ones_like(ids, dtype=torch.bool)
        if not isinstance(plan, MassCompetitiveGroupPlan):
            raise TypeError("plan must be a MassCompetitiveGroupPlan instance.")
        if int(plan.registered_kmax) != self.group_count:
            raise ValueError("mass-competitive plan capacity does not match the router.")
        if compact:
            ids = plan.packed_prototype_ids.to(device=reference.device)
            valid = plan.packed_valid.to(device=reference.device)
            return full_codes[ids.clamp_min(0)], ids, valid
        ids = torch.arange(self.group_count, device=reference.device, dtype=torch.long)[None, :].expand(reference.shape[0], -1)
        valid = plan.planned_mask.to(device=reference.device)
        return full_codes[None, :, :].expand(reference.shape[0], -1, -1), ids, valid

    def _selected_gamma(
        self,
        plan: MassCompetitiveGroupPlan,
        prototype_ids: torch.Tensor,
        valid: torch.Tensor,
        *,
        compact: bool,
    ) -> torch.Tensor:
        gamma = plan.gamma.to(device=prototype_ids.device)
        if compact:
            gamma = gamma.gather(1, prototype_ids.clamp_min(0))
        return gamma * valid.to(dtype=gamma.dtype)

    def _build_mass_assignments(
        self,
        encoded: EncodedInterfaceCase,
        module_control: torch.Tensor,
        environment_control: torch.Tensor,
        active_modules: torch.Tensor,
        module_measure: torch.Tensor,
        environment_measure: torch.Tensor,
        codes: torch.Tensor,
        valid: torch.Tensor,
        gamma: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        module_logits = self._content_logits(module_control, codes)
        environment_logits = self._content_logits(environment_control, codes)
        module_mask = valid[:, None, :] & active_modules[..., None]
        environment_mask = valid[:, None, :]
        module_proposal = entmax15(module_logits, dim=-1, mask=module_mask)
        module_proposal = module_proposal * active_modules[..., None].to(module_proposal.dtype)
        environment_proposal = entmax15(environment_logits, dim=-1, mask=environment_mask)
        (
            proposal_module_mass,
            proposal_environment_mass,
            _proposal_module_centres,
            _proposal_environment_centres,
            proposal_joint_centres,
        ) = self._joint_centres(
            module_proposal,
            environment_proposal,
            module_measure,
            environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        proposal_group_valid = valid & (
            (proposal_module_mass + proposal_environment_mass) > 0.0
        )
        if bool((proposal_group_valid.sum(dim=-1) < 1).any()):
            raise RuntimeError("content proposal produced a case with no occupied group.")
        scale = self._geometry_scale(encoded)
        module_bias = -torch.linalg.vector_norm(
            encoded.module_centers[:, :, None, :] - proposal_joint_centres[:, None, :, :],
            dim=-1,
        ) / scale[:, None, None]
        environment_bias = -torch.linalg.vector_norm(
            encoded.env_coords[:, :, None, :] - proposal_joint_centres[:, None, :, :],
            dim=-1,
        ) / scale[:, None, None]
        refined_module = entmax15(
            module_logits + module_bias,
            dim=-1,
            mask=module_mask & proposal_group_valid[:, None, :],
        )
        refined_module = refined_module * active_modules[..., None].to(refined_module.dtype)
        refined_environment = entmax15(
            environment_logits + environment_bias,
            dim=-1,
            mask=environment_mask & proposal_group_valid[:, None, :],
        )
        (
            precompetition_module_mass,
            precompetition_environment_mass,
            precompetition_module_centres,
            precompetition_environment_centres,
            precompetition_joint_centres,
        ) = self._joint_centres(
            refined_module,
            refined_environment,
            module_measure,
            environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        pi = 0.5 * (precompetition_module_mass + precompetition_environment_mass)
        if gamma is None:
            gamma = mass_competition(pi)
        active = valid & (gamma > 0.0)
        if bool((active.sum(dim=-1) < 1).any()):
            raise RuntimeError("mass competition produced a case with no active group.")
        log_gamma = _safe_log_gamma(gamma, active)
        module_membership = entmax15(
            module_logits + module_bias + log_gamma[:, None, :],
            dim=-1,
            mask=module_mask & active[:, None, :],
        )
        module_membership = module_membership * active_modules[..., None].to(module_membership.dtype)
        environment_membership = entmax15(
            environment_logits + environment_bias + log_gamma[:, None, :],
            dim=-1,
            mask=environment_mask & active[:, None, :],
        )
        return (
            module_membership,
            environment_membership,
            module_proposal,
            environment_proposal,
            proposal_module_mass,
            proposal_environment_mass,
            precompetition_module_mass,
            precompetition_environment_mass,
            precompetition_module_centres,
            precompetition_environment_centres,
            precompetition_joint_centres,
            pi,
            gamma,
            active,
        )

    def _make_mass_plan(
        self,
        module_membership: torch.Tensor,
        environment_membership: torch.Tensor,
        module_measure: torch.Tensor,
        environment_measure: torch.Tensor,
        encoded: EncodedInterfaceCase,
        *,
        gamma: torch.Tensor,
        pi: torch.Tensor,
        proposal_module_mass: torch.Tensor,
        proposal_environment_mass: torch.Tensor,
        precompetition_module_mass: torch.Tensor,
        precompetition_environment_mass: torch.Tensor,
        precompetition_module_centres: torch.Tensor,
        precompetition_environment_centres: torch.Tensor,
        precompetition_joint_centres: torch.Tensor,
    ) -> MassCompetitiveGroupPlan:
        (
            module_mass,
            environment_mass,
            module_centres,
            environment_centres,
            joint_centres,
        ) = self._joint_centres(
            module_membership,
            environment_membership,
            module_measure,
            environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        active = gamma > 0.0
        k_case = active.sum(dim=-1).to(dtype=torch.long)
        full_ids = torch.arange(self.group_count, device=gamma.device, dtype=torch.long)[None, :].expand(gamma.shape[0], -1)
        prototype_ids = full_ids.masked_fill(~active, -1).detach()
        packed_width = int(k_case.max().item())
        order = torch.argsort((~active).to(torch.int64), dim=-1, stable=True)
        packed_ids = full_ids.gather(1, order[:, :packed_width]).detach()
        packed_valid = torch.arange(packed_width, device=gamma.device)[None, :] < k_case[:, None]
        kappa = gamma.square().sum(dim=-1).reciprocal()
        return MassCompetitiveGroupPlan(
            registered_kmax=self.group_count,
            prototype_ids=prototype_ids,
            packed_prototype_ids=packed_ids,
            packed_valid=packed_valid.detach(),
            k_plan=k_case.detach(),
            module_mass_p0=module_mass,
            environment_mass_p0=environment_mass,
            module_centres_p0=module_centres,
            environment_centres_p0=environment_centres,
            joint_centres_p0=joint_centres,
            kappa_p0=kappa,
            proposal_module_mass_p0=proposal_module_mass,
            proposal_environment_mass_p0=proposal_environment_mass,
            proposal_occupied_p0=(proposal_module_mass + proposal_environment_mass) > 0.0,
            gamma_p0=gamma,
            pi_p0=pi,
            precompetition_module_mass_p0=precompetition_module_mass,
            precompetition_environment_mass_p0=precompetition_environment_mass,
            precompetition_module_centres_p0=precompetition_module_centres,
            precompetition_environment_centres_p0=precompetition_environment_centres,
            precompetition_joint_centres_p0=precompetition_joint_centres,
        )

    def _controls_from_mass_assignments(
        self,
        global_control: torch.Tensor,
        module_control: torch.Tensor,
        environment_control: torch.Tensor,
        module_membership: torch.Tensor,
        environment_membership: torch.Tensor,
        module_measure: torch.Tensor,
        environment_measure: torch.Tensor,
        valid: torch.Tensor,
        codes: torch.Tensor,
        plan: MassCompetitiveGroupPlan,
        encoded: EncodedInterfaceCase,
        gamma: torch.Tensor,
        pi: torch.Tensor,
        proposal_module_mass: torch.Tensor,
        proposal_environment_mass: torch.Tensor,
        precompetition_module_mass: torch.Tensor,
        precompetition_environment_mass: torch.Tensor,
    ) -> MassCompetitivePreparedGroupControl:
        (
            module_mass,
            environment_mass,
            module_centres,
            environment_centres,
            joint_centres,
        ) = self._joint_centres(
            module_membership,
            environment_membership,
            module_measure,
            environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        kappa = gamma.square().sum(dim=-1).reciprocal()
        group_codes = codes if codes.ndim == 3 else codes[None].expand(module_mass.shape[0], -1, -1)
        width = int(group_codes.shape[1])
        module_moment = torch.einsum(
            "bm,bmk,bmd->bkd", module_measure, module_membership, module_control
        ) * kappa[:, None, None]
        environment_moment = torch.einsum(
            "be,bek,bed->bkd", environment_measure, environment_membership, environment_control
        ) * kappa[:, None, None]
        group_input = torch.cat(
            [
                module_moment,
                environment_moment,
                kappa[:, None, None] * module_mass[..., None],
                kappa[:, None, None] * environment_mass[..., None],
                global_control[:, None, :].expand(-1, width, -1),
                group_codes,
            ],
            dim=-1,
        )
        group_control = torch.tanh(self.group_control(group_input))
        group_control = group_control * valid[..., None].to(group_control.dtype)
        return MassCompetitivePreparedGroupControl(
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
            plan=plan,
            prototype_ids=(
                plan.prototype_ids if width == self.group_count else plan.packed_prototype_ids
            ),
            packed_valid=valid,
            phase_occupied=(module_mass + environment_mass) > 0.0,
            module_centres=module_centres,
            environment_centres=environment_centres,
            joint_centres=joint_centres,
            kappa=kappa,
            gamma=gamma,
            pi_precompetition=pi,
            proposal_module_mass=proposal_module_mass,
            proposal_environment_mass=proposal_environment_mass,
            precompetition_module_mass=precompetition_module_mass,
            precompetition_environment_mass=precompetition_environment_mass,
        )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
        *,
        plan: MassCompetitiveGroupPlan | None = None,
        compact: bool = False,
    ) -> MassCompetitivePreparedGroupControl:
        (
            global_control,
            module_control,
            environment_control,
            active_modules,
            module_measure,
            environment_measure,
        ) = self._source_controls_and_measures(encoded, module_states, environment_states)
        codes, prototype_ids, valid = self._selected_codes(
            plan, compact=bool(compact), reference=module_states
        )
        gamma = None
        if plan is not None:
            gamma = self._selected_gamma(
                plan,
                prototype_ids,
                valid,
                compact=bool(compact),
            )
        built = self._build_mass_assignments(
            encoded,
            module_control,
            environment_control,
            active_modules,
            module_measure,
            environment_measure,
            codes,
            valid,
            gamma,
        )
        (
            module_membership,
            environment_membership,
            _module_proposal,
            _environment_proposal,
            proposal_module_mass,
            proposal_environment_mass,
            precompetition_module_mass,
            precompetition_environment_mass,
            precompetition_module_centres,
            precompetition_environment_centres,
            precompetition_joint_centres,
            pi,
            gamma,
            active,
        ) = built
        if plan is None:
            plan = self._make_mass_plan(
                module_membership,
                environment_membership,
                module_measure,
                environment_measure,
                encoded,
                gamma=gamma,
                pi=pi,
                proposal_module_mass=proposal_module_mass,
                proposal_environment_mass=proposal_environment_mass,
                precompetition_module_mass=precompetition_module_mass,
                precompetition_environment_mass=precompetition_environment_mass,
                precompetition_module_centres=precompetition_module_centres,
                precompetition_environment_centres=precompetition_environment_centres,
                precompetition_joint_centres=precompetition_joint_centres,
            )
            if compact:
                prototype_ids = plan.packed_prototype_ids.to(device=module_states.device)
                packed_valid = plan.packed_valid.to(device=module_states.device)
                module_membership = self._select_columns(
                    module_membership, prototype_ids, packed_valid
                )
                environment_membership = self._select_columns(
                    environment_membership, prototype_ids, packed_valid
                )
                proposal_module_mass = self._select_group_values(
                    proposal_module_mass, prototype_ids, packed_valid
                )
                proposal_environment_mass = self._select_group_values(
                    proposal_environment_mass, prototype_ids, packed_valid
                )
                precompetition_module_mass = self._select_group_values(
                    precompetition_module_mass, prototype_ids, packed_valid
                )
                precompetition_environment_mass = self._select_group_values(
                    precompetition_environment_mass, prototype_ids, packed_valid
                )
                pi = self._select_group_values(pi, prototype_ids, packed_valid)
                gamma = self._select_group_values(gamma, prototype_ids, packed_valid)
                codes = self.group_codes.to(
                    device=module_states.device, dtype=module_states.dtype
                )[prototype_ids.clamp_min(0)]
                active = packed_valid
        return self._controls_from_mass_assignments(
            global_control,
            module_control,
            environment_control,
            module_membership,
            environment_membership,
            module_measure,
            environment_measure,
            active,
            codes,
            plan,
            encoded,
            gamma,
            pi,
            proposal_module_mass,
            proposal_environment_mass,
            precompetition_module_mass,
            precompetition_environment_mass,
        )

    def route_queries(
        self,
        encoded: EncodedInterfaceCase,
        state: MassCompetitivePreparedGroupControl,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
        *,
        plan: MassCompetitiveGroupPlan | None = None,
        module_centres: torch.Tensor | None = None,
        environment_centres: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> GroupQueryRoute:
        if not isinstance(state, MassCompetitivePreparedGroupControl):
            raise TypeError("mass-competitive router requires its prepared control state.")
        if plan is not None and plan is not state.plan:
            raise ValueError("route plan must be the P0 plan carried by the prepared state.")
        if receivers.ndim != 3 or int(receivers.shape[0]) != int(state.group_control.shape[0]):
            raise ValueError("receivers must have shape [B,Q,d] aligned with prepared controls.")
        if receiver_features is None:
            query_features = self.query_fourier(receivers / self._scale(encoded))
        else:
            expected_width = int(
                (self.spatial_dim if self.query_fourier.include_input else 0)
                + 2 * self.spatial_dim * self.query_fourier.num_frequencies
            )
            query_features = (
                receiver_features
                if int(receiver_features.shape[-1]) == expected_width
                else self.query_fourier(receivers / self._scale(encoded))
            )
        query_input = torch.cat(
            [
                query_features,
                state.global_control[:, None, :].expand(-1, receivers.shape[1], -1),
            ],
            dim=-1,
        )
        query_control = self.query_projection(query_input)
        transformed_group = self.query_group_projection(state.group_control)
        logits = torch.einsum("bqd,bkd->bqk", query_control, transformed_group)
        logits = logits / (float(self.control_dim) ** 0.5)
        module_values = state.module_centres if module_centres is None else module_centres
        environment_values = (
            state.environment_centres
            if environment_centres is None
            else environment_centres
        )
        module_has_mass = state.module_mass > 0.0
        environment_has_mass = state.environment_mass > 0.0
        module_query_centres = torch.where(
            module_has_mass[..., None], module_values, environment_values
        )
        environment_query_centres = torch.where(
            environment_has_mass[..., None], environment_values, module_values
        )
        geometry = -0.5 * (
            torch.linalg.vector_norm(
                receivers[:, :, None, :] - module_query_centres[:, None, :, :], dim=-1
            )
            + torch.linalg.vector_norm(
                receivers[:, :, None, :] - environment_query_centres[:, None, :, :], dim=-1
            )
        ) / self._geometry_scale(encoded)[:, None, None]
        valid = state.packed_valid
        if active_mask is not None:
            if tuple(active_mask.shape) != tuple(valid.shape):
                raise ValueError("active_mask must align with selected group columns.")
            valid = valid & active_mask.to(device=valid.device, dtype=torch.bool)
        log_gamma = _safe_log_gamma(state.gamma, valid)
        routed_logits = logits + geometry + log_gamma[:, None, :]
        assignment = entmax15(routed_logits, dim=-1, mask=valid[:, None, :])
        return GroupQueryRoute(
            query_control=query_control,
            assignment=assignment,
            logits=routed_logits,
        )

    def prepare_mass_competitive(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
        *,
        plan: MassCompetitiveGroupPlan | None = None,
        compact: bool = False,
    ) -> tuple[
        MassCompetitivePreparedGroupControl,
        MassCompetitiveGroupPlan,
        dict[str, torch.Tensor],
    ]:
        controls = self.prepare(
            encoded,
            module_states,
            environment_states,
            plan=plan,
            compact=bool(compact),
        )
        runtime = {
            "kappa": controls.kappa,
            "gamma": controls.gamma,
            "pi": controls.pi_precompetition,
            "active_mask": controls.packed_valid,
            "phase_occupied": controls.phase_occupied,
            "module_centres": controls.module_centres,
            "environment_centres": controls.environment_centres,
            "joint_centres": controls.joint_centres,
            "module_mass": controls.module_mass,
            "environment_mass": controls.environment_mass,
            "module_membership": controls.module_membership,
            "environment_membership": controls.environment_membership,
            "query_group_mask": controls.packed_valid,
            "proposal_module_mass": controls.proposal_module_mass,
            "proposal_environment_mass": controls.proposal_environment_mass,
            "proposal_occupied": (
                controls.proposal_module_mass + controls.proposal_environment_mass
            ) > 0.0,
            "precompetition_module_mass": controls.precompetition_module_mass,
            "precompetition_environment_mass": controls.precompetition_environment_mass,
        }
        return controls, controls.plan, runtime

    def prepare_occupancy(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
        *,
        plan: MassCompetitiveGroupPlan | None = None,
        compact: bool = False,
    ) -> tuple[
        MassCompetitivePreparedGroupControl,
        MassCompetitiveGroupPlan,
        dict[str, torch.Tensor],
    ]:
        """Compatibility hook for the reused rectangular field wrapper."""

        return self.prepare_mass_competitive(
            encoded,
            module_states,
            environment_states,
            plan=plan,
            compact=compact,
        )


__all__ = [
    "MASS_COMPETITIVE_EPSILON",
    "MassCompetitiveGroupPlan",
    "MassCompetitiveGroupRouter",
    "MassCompetitivePreparedGroupControl",
    "mass_competition",
]
