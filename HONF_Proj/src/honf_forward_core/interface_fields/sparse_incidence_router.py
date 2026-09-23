"""Phase-local sparse-incidence routing for the opt-in Run 1501 core.

The registered prototype bank stays at ``K=12``.  Source organization is the
Run-1409 content-entmax plus one physical geometry refinement, but no
case-level prototype plan is formed or carried between physical phases.
Queries use prototype-anchored RMS-normalized keys and masked sparsemax.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .group_control_router import GroupQueryRoute, PreparedGroupControl
from .occupancy_group_router import (
    OCCUPANCY_CONTROL_DIM,
    OCCUPANCY_KMAX,
    OccupancyGroupRouter,
)
from .routing_index.sparse_projection import masked_sparsemax
from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class SparseIncidencePreparedGroupControl(PreparedGroupControl):
    """Live source organization and query geometry for one physical phase."""

    phase_occupied: torch.Tensor
    module_centres: torch.Tensor
    environment_centres: torch.Tensor
    joint_centres: torch.Tensor
    pi: torch.Tensor
    kappa: torch.Tensor
    proposal_module_mass: torch.Tensor
    proposal_environment_mass: torch.Tensor
    proposal_occupied: torch.Tensor

    @property
    def phase_group_mask(self) -> torch.Tensor:
        return self.phase_occupied


class SparseIncidenceGroupRouter(OccupancyGroupRouter):
    """K=12 source entmax organization with query-local sparsemax routes."""

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
        if abs(float(query_temperature) - 1.0) > 1.0e-12:
            raise ValueError(
                "sparse-incidence query sparsemax has fixed unit temperature."
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
            geometry_fraction=geometry_fraction,
        )

    @staticmethod
    def _rms_scale(values: torch.Tensor) -> torch.Tensor:
        return values / torch.sqrt(
            values.square().mean(dim=-1, keepdim=True) + 1.0e-6
        )

    def query_key_bank(
        self, state: SparseIncidencePreparedGroupControl
    ) -> torch.Tensor:
        prototypes = self.group_codes.to(
            device=state.group_control.device,
            dtype=state.group_control.dtype,
        )
        return prototypes[None, :, :] + self.query_group_projection(
            state.group_control
        )

    def normalized_query_key_bank(
        self, state: SparseIncidencePreparedGroupControl
    ) -> torch.Tensor:
        return self._rms_scale(self.query_key_bank(state))

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
    ) -> SparseIncidencePreparedGroupControl:
        """Recompute the complete source organization for one phase."""

        (
            global_control,
            module_control,
            environment_control,
            active_modules,
            module_measure,
            environment_measure,
        ) = self._source_controls_and_measures(
            encoded, module_states, environment_states
        )
        batch = int(module_states.shape[0])
        codes = self.group_codes.to(
            device=module_states.device, dtype=module_states.dtype
        )
        valid = torch.ones(
            (batch, self.group_count),
            device=module_states.device,
            dtype=torch.bool,
        )
        (
            module_membership,
            environment_membership,
            proposal_module_mass,
            proposal_environment_mass,
            _proposal_joint_centres,
            proposal_occupied,
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
        phase_occupied = (module_mass + environment_mass) > 0.0
        if bool((phase_occupied.sum(dim=-1) < 1).any()):
            raise RuntimeError(
                "sparse-incidence source organization produced an empty phase."
            )
        pi = 0.5 * (module_mass + environment_mass)
        kappa = pi.square().sum(dim=-1).clamp_min(
            torch.finfo(pi.dtype).eps
        ).reciprocal()
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
                global_control[:, None, :].expand(-1, self.group_count, -1),
                codes[None, :, :].expand(batch, -1, -1),
            ],
            dim=-1,
        )
        group_control = torch.tanh(self.group_control(group_input))
        group_control = group_control * phase_occupied[..., None].to(
            group_control.dtype
        )
        return SparseIncidencePreparedGroupControl(
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
            phase_occupied=phase_occupied,
            module_centres=module_centres,
            environment_centres=environment_centres,
            joint_centres=joint_centres,
            pi=pi,
            kappa=kappa,
            proposal_module_mass=proposal_module_mass,
            proposal_environment_mass=proposal_environment_mass,
            proposal_occupied=proposal_occupied,
        )

    def route_queries(
        self,
        encoded: EncodedInterfaceCase,
        state: SparseIncidencePreparedGroupControl,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
    ) -> GroupQueryRoute:
        """Route queries using RMS keys, live centres and masked sparsemax."""

        if not isinstance(state, SparseIncidencePreparedGroupControl):
            raise TypeError(
                "sparse-incidence router requires its phase-local control state."
            )
        if receivers.ndim != 3 or int(receivers.shape[0]) != int(
            state.group_control.shape[0]
        ):
            raise ValueError(
                "receivers must have shape [B,Q,d] aligned with prepared controls."
            )
        if int(receivers.shape[-1]) != self.spatial_dim:
            raise ValueError(
                "receiver coordinate dimension does not match the sparse-incidence router."
            )
        if receiver_features is None:
            query_features = self.query_fourier(receivers / self._scale(encoded))
        else:
            if receiver_features.ndim != 3 or tuple(
                receiver_features.shape[:2]
            ) != tuple(receivers.shape[:2]):
                raise ValueError(
                    "receiver_features must align with receivers along [B,Q]."
                )
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
                state.global_control[:, None, :].expand(
                    -1, receivers.shape[1], -1
                ),
            ],
            dim=-1,
        )
        query_control = self.query_projection(query_input)
        scaled_query = self._rms_scale(query_control)
        scaled_keys = self.normalized_query_key_bank(state)
        logits = torch.einsum("bqd,bkd->bqk", scaled_query, scaled_keys)
        logits = logits / (float(self.control_dim) ** 0.5)

        module_has_mass = state.module_mass > 0.0
        environment_has_mass = state.environment_mass > 0.0
        module_query_centres = torch.where(
            module_has_mass[..., None],
            state.module_centres,
            state.environment_centres,
        )
        environment_query_centres = torch.where(
            environment_has_mass[..., None],
            state.environment_centres,
            state.module_centres,
        )
        geometry_scale = self._geometry_scale(encoded)
        geometry = -0.5 * (
            torch.linalg.vector_norm(
                receivers[:, :, None, :]
                - module_query_centres[:, None, :, :],
                dim=-1,
            )
            + torch.linalg.vector_norm(
                receivers[:, :, None, :]
                - environment_query_centres[:, None, :, :],
                dim=-1,
            )
        ) / geometry_scale[:, None, None]
        logits = logits + geometry
        valid = state.phase_occupied[:, None, :].expand_as(logits)
        assignment = masked_sparsemax(logits, valid)
        return GroupQueryRoute(
            query_control=query_control,
            assignment=assignment,
            logits=logits,
            query_keys=scaled_keys,
        )


__all__ = [
    "SparseIncidenceGroupRouter",
    "SparseIncidencePreparedGroupControl",
]
