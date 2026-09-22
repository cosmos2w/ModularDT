"""Occupancy-derived geometric routing for the opt-in Run-1409 core.

The router in this module keeps the registered prototype bank fixed at
``Kmax`` columns, but derives the case plan from exact positive entmax source
occupancy.  The plan is discrete metadata (prototype IDs, validity masks and
bit masks); memberships, centres, controls and query assignments remain live
differentiable tensors.

The implementation deliberately does not build a ``2**K`` support table.
``Kmax=12`` fits in a 16-bit integer word, so support diagnostics use direct
bit intersections between source and query signatures.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from honf_forward_core.routing import entmax15

from .group_control_router import (
    GroupQueryRoute,
    LowDimensionalGroupRouter,
    PreparedGroupControl,
)
from .types import EncodedInterfaceCase

OCCUPANCY_KMAX = 12
OCCUPANCY_CONTROL_DIM = 16
OCCUPANCY_MASK_BITS = 16


def _validate_incidence(incidence: torch.Tensor, name: str) -> tuple[int, int, int]:
    if not isinstance(incidence, torch.Tensor) or incidence.ndim != 3:
        raise ValueError(f"{name} must have shape [B,S,K].")
    if not (incidence.is_floating_point() or incidence.dtype == torch.bool):
        raise TypeError(f"{name} must be floating point or boolean.")
    if incidence.is_floating_point():
        if not bool(torch.isfinite(incidence).all()):
            raise ValueError(f"{name} must contain only finite values.")
        if bool((incidence < 0).any()):
            raise ValueError(f"{name} must be nonnegative.")
    return tuple(int(value) for value in incidence.shape)


def pack_16bit_mask(incidence: torch.Tensor) -> torch.Tensor:
    """Pack positive group membership into one integer word per source/query.

    The output is detached integer metadata.  The helper accepts up to sixteen
    group columns, which covers the Run-1409 ``Kmax=12`` contract without
    creating a lookup table whose size grows exponentially with ``K``.
    """

    _, _, group_count = _validate_incidence(incidence, "incidence")
    if group_count <= 0 or group_count > OCCUPANCY_MASK_BITS:
        raise ValueError(
            f"incidence group width must be in [1,{OCCUPANCY_MASK_BITS}]."
        )
    shifts = torch.arange(group_count, device=incidence.device, dtype=torch.long)
    powers = torch.ones_like(shifts).bitwise_left_shift(shifts)
    return (incidence.gt(0).to(dtype=torch.long) * powers).sum(dim=-1).detach()


def occupancy_mask(incidence: torch.Tensor) -> torch.Tensor:
    """Descriptive alias for :func:`pack_16bit_mask`."""

    return pack_16bit_mask(incidence)


def direct_mask_intersection(
    query_masks: torch.Tensor,
    source_masks: torch.Tensor,
) -> torch.Tensor:
    """Return exact ``[B,Q,S]`` support from packed source/query masks."""

    if not isinstance(query_masks, torch.Tensor) or query_masks.ndim != 2:
        raise ValueError("query_masks must have shape [B,Q].")
    if not isinstance(source_masks, torch.Tensor) or source_masks.ndim != 2:
        raise ValueError("source_masks must have shape [B,S].")
    if query_masks.dtype != torch.long or source_masks.dtype != torch.long:
        raise TypeError("query_masks and source_masks must use torch.long metadata.")
    if query_masks.shape[0] != source_masks.shape[0]:
        raise ValueError("query_masks and source_masks must share their batch dimension.")
    if query_masks.device != source_masks.device:
        raise ValueError("query_masks and source_masks must share one device.")
    return query_masks[..., None].bitwise_and(source_masks[:, None, :]).ne(0)


def occupancy_support(
    query_assignment: torch.Tensor,
    source_membership: torch.Tensor,
) -> torch.Tensor:
    """Pack live incidence and compute exact unique source support."""

    query_masks = pack_16bit_mask(query_assignment)
    source_masks = pack_16bit_mask(source_membership)
    return direct_mask_intersection(query_masks, source_masks)


@dataclass(frozen=True)
class OccupancyGroupPlan:
    """P0 case plan carried unchanged through P1/P2.

    ``prototype_ids`` has width ``Kmax`` and stores original prototype IDs,
    with ``-1`` in inactive columns.  ``packed_prototype_ids`` stores the
    same original IDs in a compact leading width; ``packed_valid`` marks its
    per-case padding.  All plan tensors are detached integer/diagnostic
    metadata or detached P0 summaries.
    """

    registered_kmax: int
    prototype_ids: torch.Tensor
    packed_prototype_ids: torch.Tensor
    packed_valid: torch.Tensor
    k_plan: torch.Tensor
    module_mass_p0: torch.Tensor
    environment_mass_p0: torch.Tensor
    module_centres_p0: torch.Tensor
    environment_centres_p0: torch.Tensor
    joint_centres_p0: torch.Tensor
    kappa_p0: torch.Tensor
    proposal_module_mass_p0: torch.Tensor | None = None
    proposal_environment_mass_p0: torch.Tensor | None = None
    proposal_occupied_p0: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.registered_kmax, bool) or int(self.registered_kmax) <= 0:
            raise ValueError("registered_kmax must be a positive integer.")
        kmax = int(self.registered_kmax)
        if kmax > OCCUPANCY_MASK_BITS:
            raise ValueError(f"registered_kmax must be <= {OCCUPANCY_MASK_BITS}.")
        if self.prototype_ids.ndim != 2 or self.prototype_ids.dtype != torch.long:
            raise ValueError("prototype_ids must be a [B,Kmax] torch.long tensor.")
        if int(self.prototype_ids.shape[1]) != kmax:
            raise ValueError("prototype_ids width must equal registered_kmax.")
        batch = int(self.prototype_ids.shape[0])
        if self.packed_prototype_ids.ndim != 2 or self.packed_prototype_ids.dtype != torch.long:
            raise ValueError("packed_prototype_ids must be a [B,Kpacked] torch.long tensor.")
        if tuple(self.packed_valid.shape) != tuple(self.packed_prototype_ids.shape):
            raise ValueError("packed_valid must align with packed_prototype_ids.")
        if self.packed_valid.dtype != torch.bool:
            raise ValueError("packed_valid must be boolean metadata.")
        if int(self.packed_prototype_ids.shape[0]) != batch:
            raise ValueError("packed_prototype_ids must share the plan batch dimension.")
        if self.k_plan.shape != (batch,):
            raise ValueError("k_plan must have shape [B].")
        if self.k_plan.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError("k_plan must be integer metadata.")
        if bool((self.k_plan < 1).any()) or bool((self.k_plan > kmax).any()):
            raise ValueError("every plan must contain between one and Kmax occupied groups.")
        shape_values = (
            ("module_mass_p0", self.module_mass_p0),
            ("environment_mass_p0", self.environment_mass_p0),
            ("module_centres_p0", self.module_centres_p0),
            ("environment_centres_p0", self.environment_centres_p0),
            ("joint_centres_p0", self.joint_centres_p0),
            ("proposal_module_mass_p0", self.proposal_module_mass_p0),
            ("proposal_environment_mass_p0", self.proposal_environment_mass_p0),
            ("proposal_occupied_p0", self.proposal_occupied_p0),
            ("kappa_p0", self.kappa_p0),
        )
        for name, values in shape_values:
            if values is None:
                continue
            if name == "kappa_p0":
                valid_shape = values.shape == (batch,)
            else:
                valid_shape = int(values.shape[0]) == batch and int(values.shape[1]) == kmax
            if not valid_shape:
                raise ValueError(f"{name} must be batch aligned with [B,Kmax].")
        tensors = (
            self.packed_prototype_ids,
            self.packed_valid,
            self.k_plan,
            self.module_mass_p0,
            self.environment_mass_p0,
            self.module_centres_p0,
            self.environment_centres_p0,
            self.joint_centres_p0,
            self.proposal_module_mass_p0,
            self.proposal_environment_mass_p0,
            self.proposal_occupied_p0,
            self.kappa_p0,
        )
        tensors = tuple(value for value in tensors if value is not None)
        if any(value.device != self.prototype_ids.device for value in tensors):
            raise ValueError("all occupancy plan tensors must share one device.")

    @property
    def Kmax(self) -> int:
        return int(self.registered_kmax)

    @property
    def kmax(self) -> int:
        return int(self.registered_kmax)

    @property
    def registered_capacity(self) -> int:
        return int(self.registered_kmax)

    @property
    def K_plan(self) -> torch.Tensor:
        return self.k_plan

    @property
    def active_prototype_ids(self) -> torch.Tensor:
        return self.prototype_ids

    @property
    def original_prototype_ids(self) -> torch.Tensor:
        return self.prototype_ids

    @property
    def packed_ids(self) -> torch.Tensor:
        return self.packed_prototype_ids

    @property
    def planned_mask(self) -> torch.Tensor:
        return self.prototype_ids.ge(0)

    @property
    def active_mask(self) -> torch.Tensor:
        """P0 occupied columns in registered prototype-ID coordinates."""

        return self.planned_mask

    @property
    def proposal_module_mass(self) -> torch.Tensor:
        """P0 content proposal mass before the geometry refinement."""

        if self.proposal_module_mass_p0 is None:
            return self.module_mass_p0
        return self.proposal_module_mass_p0

    @property
    def proposal_environment_mass(self) -> torch.Tensor:
        if self.proposal_environment_mass_p0 is None:
            return self.environment_mass_p0
        return self.proposal_environment_mass_p0

    @property
    def proposal_occupied(self) -> torch.Tensor:
        if self.proposal_occupied_p0 is None:
            return (self.module_mass_p0 + self.environment_mass_p0) > 0.0
        return self.proposal_occupied_p0

    @property
    def module_mass(self) -> torch.Tensor:
        return self.module_mass_p0

    @property
    def environment_mass(self) -> torch.Tensor:
        return self.environment_mass_p0

    @property
    def kappa(self) -> torch.Tensor:
        return self.kappa_p0


@dataclass(frozen=True)
class OccupancyPreparedGroupControl(PreparedGroupControl):
    """Run-1409 live controls plus the immutable P0 ID plan."""

    plan: OccupancyGroupPlan
    prototype_ids: torch.Tensor
    packed_valid: torch.Tensor
    phase_occupied: torch.Tensor
    module_centres: torch.Tensor
    environment_centres: torch.Tensor
    joint_centres: torch.Tensor
    kappa: torch.Tensor
    proposal_module_mass: torch.Tensor
    proposal_environment_mass: torch.Tensor
    proposal_occupied: torch.Tensor

    @property
    def k_plan(self) -> torch.Tensor:
        return self.plan.k_plan

    @property
    def K_plan(self) -> torch.Tensor:
        return self.plan.k_plan

    @property
    def phase_group_mask(self) -> torch.Tensor:
        return self.phase_occupied


class OccupancyGroupRouter(LowDimensionalGroupRouter):
    """Run-1409 content plus one-step geometric occupancy router."""

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
        if int(group_count) != OCCUPANCY_KMAX:
            raise ValueError(
                f"occupancy_adaptive_group_control_honf requires group_count={OCCUPANCY_KMAX}."
            )
        if int(control_dim) != OCCUPANCY_CONTROL_DIM:
            raise ValueError(
                "occupancy_adaptive_group_control_honf requires "
                f"group_control_dim={OCCUPANCY_CONTROL_DIM}."
            )
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
        if int(group_count) > OCCUPANCY_MASK_BITS:
            raise ValueError(f"occupancy group_count must be <= {OCCUPANCY_MASK_BITS}.")
        if not torch.isfinite(torch.tensor(float(geometry_fraction))) or float(geometry_fraction) <= 0.0:
            raise ValueError("geometry_fraction must be finite and positive.")
        self.geometry_fraction = float(geometry_fraction)
        # Approximately orthogonal rows provide diverse initial prototypes and
        # keep epoch-one entmax from receiving identical content logits.
        with torch.no_grad():
            nn.init.orthogonal_(self.group_codes)

    @property
    def prototypes(self) -> torch.Tensor:
        """Compatibility/readability alias for the learned prototype bank."""

        return self.group_codes

    @property
    def group_prototypes(self) -> torch.Tensor:
        return self.group_codes

    def _source_controls_and_measures(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if module_states.ndim != 3 or environment_states.ndim != 3:
            raise ValueError("module_states and environment_states must have shape [B,S,H].")
        batch = int(module_states.shape[0])
        if int(environment_states.shape[0]) != batch:
            raise ValueError("module_states and environment_states must share their batch dimension.")
        if int(module_states.shape[-1]) != self.hidden_dim or int(environment_states.shape[-1]) != self.hidden_dim:
            raise ValueError("source state width does not match the occupancy router.")
        if tuple(encoded.global_token.shape) != (batch, self.hidden_dim):
            raise ValueError("encoded.global_token must have shape [B,H].")
        if tuple(encoded.module_centers.shape[:2]) != tuple(module_states.shape[:2]):
            raise ValueError("module_states and encoded.module_centers must align.")
        if tuple(encoded.env_coords.shape[:2]) != tuple(environment_states.shape[:2]):
            raise ValueError("environment_states and encoded.env_coords must align.")
        if int(encoded.module_centers.shape[-1]) != self.spatial_dim or int(encoded.env_coords.shape[-1]) != self.spatial_dim:
            raise ValueError("source coordinate dimension does not match the occupancy router.")
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
        return (
            global_control,
            module_control,
            environment_control,
            active_modules,
            module_measure,
            environment_measure,
        )

    @staticmethod
    def _content_logits(
        source_control: torch.Tensor,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        if codes.ndim == 2:
            logits = torch.einsum("bsd,kd->bsk", source_control, codes)
        elif codes.ndim == 3:
            logits = torch.einsum("bsd,bkd->bsk", source_control, codes)
        else:
            raise ValueError("codes must have shape [K,D] or [B,K,D].")
        return logits / (float(source_control.shape[-1]) ** 0.5)

    def _geometry_scale(self, encoded: EncodedInterfaceCase) -> torch.Tensor:
        scale = self._scale(encoded)
        diagonal = torch.linalg.vector_norm(scale[:, 0, :], dim=-1)
        return (float(self.geometry_fraction) * diagonal).clamp_min(torch.finfo(scale.dtype).eps)

    @staticmethod
    def _centres(
        membership: torch.Tensor,
        measure: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mass = torch.einsum("bs,bsk->bk", measure, membership)
        numerator = torch.einsum("bs,bsk,bsd->bkd", measure, membership, coordinates)
        safe_mass = mass.clamp_min(torch.finfo(membership.dtype).eps)
        centres = numerator / safe_mass[..., None]
        return mass, centres

    @classmethod
    def _joint_centres(
        cls,
        module_membership: torch.Tensor,
        environment_membership: torch.Tensor,
        module_measure: torch.Tensor,
        environment_measure: torch.Tensor,
        module_coordinates: torch.Tensor,
        environment_coordinates: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        module_mass, module_centres = cls._centres(
            module_membership, module_measure, module_coordinates
        )
        environment_mass, environment_centres = cls._centres(
            environment_membership, environment_measure, environment_coordinates
        )
        total = module_mass + environment_mass
        safe_total = total.clamp_min(torch.finfo(module_mass.dtype).eps)
        joint = (
            module_mass[..., None] * module_centres
            + environment_mass[..., None] * environment_centres
        ) / safe_total[..., None]
        only_module = (module_mass > 0.0) & (environment_mass <= 0.0)
        only_environment = (environment_mass > 0.0) & (module_mass <= 0.0)
        joint = torch.where(only_module[..., None], module_centres, joint)
        joint = torch.where(only_environment[..., None], environment_centres, joint)
        return module_mass, environment_mass, module_centres, environment_centres, joint

    @staticmethod
    def _select_columns(
        values: torch.Tensor,
        prototype_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        safe_ids = prototype_ids.clamp_min(0)
        selected = values.gather(-1, safe_ids[:, None, :].expand(*values.shape[:-1], safe_ids.shape[-1]))
        return selected * valid[:, None, :].to(dtype=selected.dtype)

    @staticmethod
    def _select_group_values(
        values: torch.Tensor,
        prototype_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        safe_ids = prototype_ids.clamp_min(0)
        if values.ndim == 2:
            selected = values.gather(1, safe_ids)
            return selected * valid.to(dtype=selected.dtype)
        selected = values.gather(1, safe_ids[..., None].expand(-1, -1, values.shape[-1]))
        return selected * valid[..., None].to(dtype=selected.dtype)

    def _make_plan(
        self,
        module_membership: torch.Tensor,
        environment_membership: torch.Tensor,
        module_measure: torch.Tensor,
        environment_measure: torch.Tensor,
        encoded: EncodedInterfaceCase,
        kappa: torch.Tensor,
        *,
        proposal_module_mass: torch.Tensor | None = None,
        proposal_environment_mass: torch.Tensor | None = None,
        proposal_occupied: torch.Tensor | None = None,
    ) -> OccupancyGroupPlan:
        module_mass, environment_mass, module_centres, environment_centres, joint_centres = self._joint_centres(
            module_membership,
            environment_membership,
            module_measure,
            environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        occupied = (module_mass + environment_mass) > 0.0
        k_plan = occupied.sum(dim=-1).to(dtype=torch.long)
        if bool((k_plan < 1).any()):
            raise RuntimeError("occupancy router produced a case with no occupied groups.")
        if proposal_module_mass is None:
            proposal_module_mass = module_mass
        if proposal_environment_mass is None:
            proposal_environment_mass = environment_mass
        if proposal_occupied is None:
            proposal_occupied = (proposal_module_mass + proposal_environment_mass) > 0.0
        full_ids = torch.arange(
            self.group_count,
            device=module_membership.device,
            dtype=torch.long,
        )[None, :].expand(module_membership.shape[0], -1)
        prototype_ids = full_ids.masked_fill(~occupied, -1).detach()
        packed_width = int(k_plan.max().item())
        order = torch.argsort((~occupied).to(torch.int64), dim=-1, stable=True)
        packed_prototype_ids = full_ids.gather(1, order[:, :packed_width]).detach()
        packed_valid = (
            torch.arange(packed_width, device=module_membership.device)[None, :]
            < k_plan[:, None]
        )
        return OccupancyGroupPlan(
            registered_kmax=self.group_count,
            prototype_ids=prototype_ids,
            packed_prototype_ids=packed_prototype_ids,
            packed_valid=packed_valid.detach(),
            k_plan=k_plan.detach(),
            module_mass_p0=module_mass.detach(),
            environment_mass_p0=environment_mass.detach(),
            module_centres_p0=module_centres.detach(),
            environment_centres_p0=environment_centres.detach(),
            joint_centres_p0=joint_centres.detach(),
            kappa_p0=kappa.detach(),
            proposal_module_mass_p0=proposal_module_mass.detach(),
            proposal_environment_mass_p0=proposal_environment_mass.detach(),
            proposal_occupied_p0=proposal_occupied.detach(),
        )

    def _selected_codes(
        self,
        plan: OccupancyGroupPlan | None,
        *,
        compact: bool,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_codes = self.group_codes.to(device=reference.device, dtype=reference.dtype)
        if plan is None:
            ids = torch.arange(self.group_count, device=reference.device, dtype=torch.long)[None, :].expand(reference.shape[0], -1)
            valid = torch.ones_like(ids, dtype=torch.bool)
            # P0 is always formed at registered width.  The caller can then
            # select its newly derived plan for a packed runtime state.
            return full_codes, ids, valid
        if int(plan.registered_kmax) != self.group_count:
            raise ValueError("occupancy plan capacity does not match the router.")
        if compact:
            ids = plan.packed_prototype_ids.to(device=reference.device)
            valid = plan.packed_valid.to(device=reference.device)
            codes = full_codes[ids.clamp_min(0)]
        else:
            ids = torch.arange(self.group_count, device=reference.device, dtype=torch.long)[None, :].expand(reference.shape[0], -1)
            valid = plan.planned_mask.to(device=reference.device)
            codes = full_codes[None, :, :].expand(reference.shape[0], -1, -1)
        return codes, ids, valid

    def _build_assignments(
        self,
        encoded: EncodedInterfaceCase,
        module_control: torch.Tensor,
        environment_control: torch.Tensor,
        active_modules: torch.Tensor,
        module_measure: torch.Tensor,
        environment_measure: torch.Tensor,
        codes: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        module_logits = self._content_logits(module_control, codes)
        environment_logits = self._content_logits(environment_control, codes)
        module_mask = valid[:, None, :] & active_modules[..., None]
        environment_mask = valid[:, None, :]
        module_proposal = entmax15(
            module_logits / float(self.module_temperature),
            dim=-1,
            mask=module_mask,
        )
        module_proposal = module_proposal * active_modules[..., None].to(module_proposal.dtype)
        environment_proposal = entmax15(
            environment_logits / float(self.environment_temperature),
            dim=-1,
            mask=environment_mask,
        )
        proposal_module_mass, proposal_environment_mass, _, _, proposal_joint_centres = self._joint_centres(
            module_proposal,
            environment_proposal,
            module_measure,
            environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        scale = self._geometry_scale(encoded)
        module_bias = -torch.linalg.vector_norm(
            encoded.module_centers[:, :, None, :] - proposal_joint_centres[:, None, :, :],
            dim=-1,
        ) / scale[:, None, None]
        environment_bias = -torch.linalg.vector_norm(
            encoded.env_coords[:, :, None, :] - proposal_joint_centres[:, None, :, :],
            dim=-1,
        ) / scale[:, None, None]
        proposal_occupied = (proposal_module_mass + proposal_environment_mass) > 0.0
        final_valid = valid & proposal_occupied
        module_membership = entmax15(
            module_logits / float(self.module_temperature) + module_bias,
            dim=-1,
            mask=module_mask & final_valid[:, None, :],
        )
        module_membership = module_membership * active_modules[..., None].to(module_membership.dtype)
        environment_membership = entmax15(
            environment_logits / float(self.environment_temperature) + environment_bias,
            dim=-1,
            mask=environment_mask & final_valid[:, None, :],
        )
        return (
            module_membership,
            environment_membership,
            proposal_module_mass,
            proposal_environment_mass,
            proposal_joint_centres,
            final_valid,
        )

    def _controls_from_assignments(
        self,
        global_control: torch.Tensor,
        module_control: torch.Tensor,
        environment_control: torch.Tensor,
        module_membership: torch.Tensor,
        environment_membership: torch.Tensor,
        module_measure: torch.Tensor,
        environment_measure: torch.Tensor,
        valid: torch.Tensor,
        phase_occupied: torch.Tensor,
        codes: torch.Tensor,
        plan: OccupancyGroupPlan,
        encoded: EncodedInterfaceCase,
        proposal_module_mass: torch.Tensor,
        proposal_environment_mass: torch.Tensor,
        proposal_occupied: torch.Tensor,
    ) -> OccupancyPreparedGroupControl:
        module_mass, environment_mass, module_centres, environment_centres, joint_centres = self._joint_centres(
            module_membership,
            environment_membership,
            module_measure,
            environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        pi = 0.5 * (module_mass + environment_mass)
        kappa = 1.0 / pi.square().sum(dim=-1).clamp_min(torch.finfo(pi.dtype).eps)
        group_codes = codes if codes.ndim == 3 else codes[None].expand(module_mass.shape[0], -1, -1)
        group_width = int(group_codes.shape[1])
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
                global_control[:, None, :].expand(-1, group_width, -1),
                group_codes,
            ],
            dim=-1,
        )
        group_control = torch.tanh(self.group_control(group_input))
        group_control = group_control * (valid & phase_occupied)[..., None].to(group_control.dtype)
        return OccupancyPreparedGroupControl(
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
            prototype_ids=plan.prototype_ids if group_width == self.group_count else plan.packed_prototype_ids,
            packed_valid=valid,
            phase_occupied=phase_occupied,
            module_centres=module_centres,
            environment_centres=environment_centres,
            joint_centres=joint_centres,
            kappa=kappa,
            proposal_module_mass=proposal_module_mass,
            proposal_environment_mass=proposal_environment_mass,
            proposal_occupied=proposal_occupied,
        )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
        *,
        plan: OccupancyGroupPlan | None = None,
        compact: bool = False,
    ) -> OccupancyPreparedGroupControl:
        """Prepare P0 or refresh a P1/P2 phase around the same ID plan."""

        (
            global_control,
            module_control,
            environment_control,
            active_modules,
            module_measure,
            environment_measure,
        ) = self._source_controls_and_measures(encoded, module_states, environment_states)
        if plan is not None and not isinstance(plan, OccupancyGroupPlan):
            raise TypeError("plan must be an OccupancyGroupPlan instance.")
        codes, _prototype_ids, valid = self._selected_codes(
            plan, compact=bool(compact), reference=module_states
        )
        (
            module_membership,
            environment_membership,
            _proposal_module_mass,
            _proposal_environment_mass,
            _proposal_joint_centres,
            _phase_valid,
        ) = self._build_assignments(
            encoded,
            module_control,
            environment_control,
            active_modules,
            module_measure,
            environment_measure,
            codes,
            valid,
        )
        module_mass, environment_mass, _, _, _ = self._joint_centres(
            module_membership,
            environment_membership,
            module_measure,
            environment_measure,
            encoded.module_centers,
            encoded.env_coords,
        )
        phase_occupied = (module_mass + environment_mass) > 0.0
        if plan is None:
            # P0 is first computed at full registered width so the plan keeps
            # exact original IDs; compact output is selected below afterward.
            plan = self._make_plan(
                module_membership,
                environment_membership,
                module_measure,
                environment_measure,
                encoded,
                1.0 / (0.5 * (module_mass + environment_mass)).square().sum(dim=-1).clamp_min(
                    torch.finfo(module_mass.dtype).eps
                ),
                proposal_module_mass=_proposal_module_mass,
                proposal_environment_mass=_proposal_environment_mass,
                proposal_occupied=(
                    _proposal_module_mass + _proposal_environment_mass
                ) > 0.0,
            )
        if compact:
            # Rebuild the current assignments over the planned packed IDs so
            # every live column has the same normalization as its full path.
            full_codes = self.group_codes.to(device=module_states.device, dtype=module_states.dtype)
            packed_ids = plan.packed_prototype_ids.to(device=module_states.device)
            packed_valid = plan.packed_valid.to(device=module_states.device)
            packed_codes = full_codes[packed_ids.clamp_min(0)]
            full_plan_valid = plan.planned_mask.to(device=module_states.device)
            full_codes_batch = full_codes[None].expand(module_states.shape[0], -1, -1)
            (
                full_module,
                full_environment,
                full_proposal_module_mass,
                full_proposal_environment_mass,
                _full_proposal_joint_centres,
                _,
            ) = self._build_assignments(
                encoded,
                module_control,
                environment_control,
                active_modules,
                module_measure,
                environment_measure,
                full_codes_batch,
                full_plan_valid,
            )
            module_membership = self._select_columns(full_module, packed_ids, packed_valid)
            environment_membership = self._select_columns(full_environment, packed_ids, packed_valid)
            codes = packed_codes
            valid = packed_valid
            module_mass, environment_mass, _, _, _ = self._joint_centres(
                module_membership,
                environment_membership,
                module_measure,
                environment_measure,
                encoded.module_centers,
                encoded.env_coords,
            )
            phase_occupied = (module_mass + environment_mass) > 0.0
            proposal_module_mass = self._select_group_values(
                full_proposal_module_mass, packed_ids, packed_valid
            )
            proposal_environment_mass = self._select_group_values(
                full_proposal_environment_mass, packed_ids, packed_valid
            )
            proposal_occupied = (
                proposal_module_mass + proposal_environment_mass
            ) > 0.0
        else:
            proposal_module_mass = _proposal_module_mass
            proposal_environment_mass = _proposal_environment_mass
            proposal_occupied = (
                proposal_module_mass + proposal_environment_mass
            ) > 0.0
        return self._controls_from_assignments(
            global_control,
            module_control,
            environment_control,
            module_membership,
            environment_membership,
            module_measure,
            environment_measure,
            valid,
            phase_occupied,
            codes,
            plan,
            encoded,
            proposal_module_mass,
            proposal_environment_mass,
            proposal_occupied,
        )

    def route_queries(
        self,
        encoded: EncodedInterfaceCase,
        state: OccupancyPreparedGroupControl,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
        *,
        plan: OccupancyGroupPlan | None = None,
        module_centres: torch.Tensor | None = None,
        environment_centres: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> GroupQueryRoute:
        """Route queries over planned and currently occupied groups."""

        if not isinstance(state, OccupancyPreparedGroupControl):
            raise TypeError("occupancy router requires OccupancyPreparedGroupControl state.")
        if plan is not None and plan is not state.plan:
            raise ValueError("route plan must be the same P0 plan carried by the prepared state.")
        if receivers.ndim != 3 or int(receivers.shape[0]) != int(state.group_control.shape[0]):
            raise ValueError("receivers must have shape [B,Q,d] aligned with prepared controls.")
        if int(receivers.shape[-1]) != self.spatial_dim:
            raise ValueError("receiver coordinate dimension does not match the occupancy router.")
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
            query_features = receiver_features if int(receiver_features.shape[-1]) == expected_width else self.query_fourier(
                receivers / self._scale(encoded)
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
        logits = logits / float(self.query_temperature)
        scale = self._geometry_scale(encoded)
        # The query rule uses both live source-type centres.  When a planned
        # group has no current mass of one type, the available type supplies
        # both terms; a phase-empty group is masked immediately afterward.
        module_centre_values = state.module_centres if module_centres is None else module_centres
        environment_centre_values = (
            state.environment_centres if environment_centres is None else environment_centres
        )
        module_has_mass = state.module_mass > 0.0
        environment_has_mass = state.environment_mass > 0.0
        module_query_centres = torch.where(
            module_has_mass[..., None],
            module_centre_values,
            environment_centre_values,
        )
        environment_query_centres = torch.where(
            environment_has_mass[..., None],
            environment_centre_values,
            module_centre_values,
        )
        geometry = -0.5 * (
            torch.linalg.vector_norm(
                receivers[:, :, None, :] - module_query_centres[:, None, :, :], dim=-1
            )
            + torch.linalg.vector_norm(
                receivers[:, :, None, :] - environment_query_centres[:, None, :, :], dim=-1
            )
        ) / scale[:, None, None]
        logits = logits + geometry
        valid = state.packed_valid & state.phase_occupied
        if active_mask is not None:
            if tuple(active_mask.shape) != tuple(valid.shape):
                raise ValueError("active_mask must align with the current selected group columns.")
            valid = valid & active_mask.to(device=valid.device, dtype=torch.bool)
        assignment = entmax15(logits, dim=-1, mask=valid[:, None, :])
        return GroupQueryRoute(
            query_control=query_control,
            assignment=assignment,
            logits=logits,
        )

    def prepare_occupancy(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
        *,
        plan: OccupancyGroupPlan | None = None,
        compact: bool = False,
    ) -> tuple[OccupancyPreparedGroupControl, OccupancyGroupPlan, dict[str, torch.Tensor]]:
        """Runtime-facing wrapper returning controls, P0 plan and diagnostics."""

        controls = self.prepare(
            encoded,
            module_states,
            environment_states,
            plan=plan,
            compact=bool(compact),
        )
        current_mask = controls.packed_valid & controls.phase_occupied
        runtime = {
            "kappa": controls.kappa,
            "active_mask": current_mask,
            "module_centres": controls.module_centres,
            "environment_centres": controls.environment_centres,
            "joint_centres": controls.joint_centres,
            "module_mass": controls.module_mass,
            "environment_mass": controls.environment_mass,
            "module_membership": controls.module_membership,
            "environment_membership": controls.environment_membership,
            "query_group_mask": current_mask,
            "proposal_module_mass": controls.proposal_module_mass,
            "proposal_environment_mass": controls.proposal_environment_mass,
            "proposal_occupied": controls.proposal_occupied,
        }
        return controls, controls.plan, runtime


__all__ = [
    "OCCUPANCY_CONTROL_DIM",
    "OCCUPANCY_KMAX",
    "OccupancyAdaptiveGroupRouter",
    "OccupancyGroupPlan",
    "OccupancyGroupRouter",
    "OccupancyPreparedGroupControl",
    "direct_mask_intersection",
    "occupancy_mask",
    "occupancy_support",
    "pack_16bit_mask",
]

# The longer name is used by the opt-in backend; retain the concise name for
# direct core tests and future callers.
OccupancyAdaptiveGroupRouter = OccupancyGroupRouter
