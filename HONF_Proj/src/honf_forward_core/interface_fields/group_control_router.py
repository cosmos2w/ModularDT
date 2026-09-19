"""Low-dimensional group control for the Run-1406 interface reader.

The Run-1406 router keeps the group dimension on the control side only.  Fine
source states stay at the model width (normally 256); source assignments,
group moments, and query assignments are all computed at ``control_dim``
(16 in the formal profile).  In particular, this module does not construct
source-by-group hidden-state banks or assignment-dependent centroids.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from honf_forward_core.nn import MLP, FourierFeatures
from honf_forward_core.routing import entmax15

from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class PreparedGroupControl:
    """Prepared source memberships and bounded group controls.

    Tensors retain their autograd graph.  ``module_measure`` and
    ``environment_measure`` are normalized source measures, whereas
    ``module_mass`` and ``environment_mass`` are the corresponding weighted
    group occupancies.  ``group_control`` is bounded by the router's final
    ``tanh`` and has width ``D`` rather than the fine hidden width ``H``.
    """

    module_membership: torch.Tensor
    environment_membership: torch.Tensor
    module_measure: torch.Tensor
    environment_measure: torch.Tensor
    module_mass: torch.Tensor
    environment_mass: torch.Tensor
    module_control: torch.Tensor
    environment_control: torch.Tensor
    group_control: torch.Tensor
    global_control: torch.Tensor

    @property
    def A_m(self) -> torch.Tensor:
        return self.module_membership

    @property
    def A_e(self) -> torch.Tensor:
        return self.environment_membership

    @property
    def h(self) -> torch.Tensor:
        return self.group_control

    @property
    def omega_m(self) -> torch.Tensor:
        return self.module_measure

    @property
    def omega_e(self) -> torch.Tensor:
        return self.environment_measure


@dataclass(frozen=True)
class GroupQueryRoute:
    """One shared query-to-group route used by both fine source types."""

    query_control: torch.Tensor
    assignment: torch.Tensor
    logits: torch.Tensor

    @property
    def v_q(self) -> torch.Tensor:
        return self.query_control

    @property
    def alpha(self) -> torch.Tensor:
        return self.assignment


class LowDimensionalGroupRouter(nn.Module):
    """Fixed-K entmax routing with a small signed control state.

    ``control_dim`` is configurable for reusable tests and future profiles;
    the Run-1406 profile fixes it at 16.  All three assignment families use
    1.5-entmax at unit temperature in that profile.  Empty candidates are not
    hard-masked at query time: their source overlap is zero and therefore they
    contribute no fine response, as required by the Run-1406 closure.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        group_count: int = 6,
        control_dim: int = 16,
        fourier_frequencies: int = 4,
        spatial_dim: int = 2,
        module_temperature: float = 1.0,
        environment_temperature: float = 1.0,
        query_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive.")
        if int(group_count) <= 0:
            raise ValueError("group_count must be positive.")
        if int(control_dim) <= 0:
            raise ValueError("control_dim must be positive.")
        if int(spatial_dim) <= 0:
            raise ValueError("spatial_dim must be positive.")
        for name, value in (
            ("module_temperature", module_temperature),
            ("environment_temperature", environment_temperature),
            ("query_temperature", query_temperature),
        ):
            if not torch.isfinite(torch.tensor(float(value))) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        self.hidden_dim = int(hidden_dim)
        self.group_count = int(group_count)
        self.control_dim = int(control_dim)
        self.spatial_dim = int(spatial_dim)
        self.module_temperature = float(module_temperature)
        self.environment_temperature = float(environment_temperature)
        self.query_temperature = float(query_temperature)

        # Unit-scale independent entries are intentional.  Scores apply the
        # one displayed 1/sqrt(D) factor exactly once.
        self.group_codes = nn.Parameter(torch.randn(self.group_count, self.control_dim))

        self.global_norm = nn.LayerNorm(self.hidden_dim)
        self.module_source_norm = nn.LayerNorm(self.hidden_dim)
        self.environment_source_norm = nn.LayerNorm(self.hidden_dim)
        self.global_projection = nn.Linear(self.hidden_dim, self.control_dim)
        self.module_projection = nn.Linear(self.hidden_dim, self.control_dim)
        self.environment_projection = nn.Linear(self.hidden_dim, self.control_dim)
        self.module_global_projection = nn.Linear(self.control_dim, self.control_dim, bias=False)
        self.environment_global_projection = nn.Linear(self.control_dim, self.control_dim, bias=False)

        self.position_fourier = FourierFeatures(None, int(fourier_frequencies))
        self.module_position_projection = nn.LazyLinear(self.control_dim)
        self.environment_position_projection = nn.LazyLinear(self.control_dim)

        # Input is [b^M, b^E, K mu^M, K mu^E, g_c, c_k].  The output tanh is
        # the sole bounded-control operation; no occupancy division occurs.
        self.group_control = MLP(
            4 * self.control_dim + 2,
            self.control_dim,
            self.control_dim,
            num_layers=2,
        )

        self.query_fourier = FourierFeatures(None, int(fourier_frequencies))
        self.query_projection = nn.Sequential(
            nn.LazyLinear(self.control_dim),
            nn.GELU(),
            nn.Linear(self.control_dim, self.control_dim),
        )
        self.query_group_projection = nn.Linear(
            self.control_dim,
            self.control_dim,
            bias=False,
        )

    def _scale(self, encoded: EncodedInterfaceCase) -> torch.Tensor:
        scale = encoded.coordinate_scale
        if scale.ndim == 1:
            scale = scale.reshape(1, 1, -1)
        elif scale.ndim == 2:
            scale = scale[:, None, :]
        if scale.ndim != 3 or int(scale.shape[-1]) != self.spatial_dim:
            raise ValueError(
                "encoded.coordinate_scale must have shape [d], [B,d], or [B,1,d] "
                f"with d={self.spatial_dim}."
            )
        batch = int(encoded.global_token.shape[0])
        if int(scale.shape[0]) not in {1, batch}:
            raise ValueError("encoded.coordinate_scale batch dimension does not match the case batch.")
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
            raise ValueError("encoded.coordinate_scale must be finite and strictly positive.")
        if int(scale.shape[0]) == 1:
            scale = scale.expand(batch, -1, -1)
        return scale

    def _source_control(
        self,
        source_states: torch.Tensor,
        coordinates: torch.Tensor,
        global_control: torch.Tensor,
        source_projection: nn.Module,
        source_norm: nn.Module,
        global_projection: nn.Module,
        position_projection: nn.Module,
        encoded: EncodedInterfaceCase,
    ) -> torch.Tensor:
        scale = self._scale(encoded)
        position = position_projection(self.position_fourier(coordinates / scale))
        source = source_projection(source_norm(source_states))
        global_term = global_projection(global_control)[:, None, :]
        return source + position + global_term

    def _assign(
        self,
        source_control: torch.Tensor,
        *,
        temperature: float,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        codes = self.group_codes.to(
            device=source_control.device,
            dtype=source_control.dtype,
        )
        logits = torch.einsum("bsd,kd->bsk", source_control, codes)
        logits = logits / (float(self.control_dim) ** 0.5)
        logits = logits / float(temperature)
        return entmax15(logits, dim=-1, mask=source_mask)

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
    ) -> PreparedGroupControl:
        """Prepare low-dimensional memberships and bounded group moments."""

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
        module_membership = self._assign(
            module_control,
            temperature=self.module_temperature,
            source_mask=active_modules[..., None],
        )
        module_membership = module_membership * active_modules[..., None].to(module_membership.dtype)

        if encoded.env_weights.ndim != 2 or tuple(encoded.env_weights.shape) != tuple(encoded.env_coords.shape[:2]):
            raise ValueError("encoded.env_weights must align with encoded.env_coords as [B,E].")
        if not bool(torch.isfinite(encoded.env_weights).all()) or bool((encoded.env_weights <= 0.0).any()):
            raise ValueError("encoded.env_weights must contain finite strictly positive masses.")
        environment_membership = self._assign(
            environment_control,
            temperature=self.environment_temperature,
        )

        module_count = active_modules.sum(dim=1, keepdim=True)
        module_measure = active_modules.to(module_states.dtype) / module_count.clamp_min(1).to(module_states.dtype)
        environment_measure = encoded.env_weights / encoded.env_weights.sum(dim=1, keepdim=True)
        module_mass = torch.einsum("bm,bmk->bk", module_measure, module_membership)
        environment_mass = torch.einsum("be,bek->bk", environment_measure, environment_membership)
        module_moment = torch.einsum(
            "bm,bmk,bmd->bkd",
            module_measure,
            module_membership,
            module_control,
        ) * float(self.group_count)
        environment_moment = torch.einsum(
            "be,bek,bed->bkd",
            environment_measure,
            environment_membership,
            environment_control,
        ) * float(self.group_count)
        codes = self.group_codes.to(device=module_states.device, dtype=module_states.dtype)
        group_input = torch.cat(
            [
                module_moment,
                environment_moment,
                float(self.group_count) * module_mass[..., None],
                float(self.group_count) * environment_mass[..., None],
                global_control[:, None, :].expand(-1, self.group_count, -1),
                codes[None, :, :].expand(batch, -1, -1),
            ],
            dim=-1,
        )
        group_control = torch.tanh(self.group_control(group_input))
        return PreparedGroupControl(
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
        )

    def route_queries(
        self,
        encoded: EncodedInterfaceCase,
        state: PreparedGroupControl,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
    ) -> GroupQueryRoute:
        """Compute one D-wide query route shared by module and environment reads."""

        if receivers.ndim != 3 or int(receivers.shape[0]) != int(state.group_control.shape[0]):
            raise ValueError("receivers must have shape [B,Q,d] aligned with prepared controls.")
        if int(receivers.shape[-1]) != self.spatial_dim:
            raise ValueError("Receiver coordinate dimension does not match the router.")
        if receiver_features is None:
            scale = self._scale(encoded)
            query_features = self.query_fourier(receivers / scale)
        else:
            if receiver_features.ndim != 3:
                raise ValueError("receiver_features must have shape [B,Q,F].")
            if tuple(receiver_features.shape[:2]) != tuple(receivers.shape[:2]):
                raise ValueError("receiver_features must align with receivers along [B,Q].")
            # The core and this router share the same Fourier convention.  A
            # width check prevents unrelated decoder features from silently
            # entering the query projection; standalone callers can omit the
            # tensor and retain the internal Fourier fallback above.
            expected_width = int(
                (self.spatial_dim if self.query_fourier.include_input else 0)
                + 2 * self.spatial_dim * self.query_fourier.num_frequencies
            )
            if int(receiver_features.shape[-1]) == expected_width:
                query_features = receiver_features
            else:
                # Preserve the historical standalone API, where callers
                # sometimes pass arbitrary feature tensors intended only for
                # the environment reader.  Such tensors are not compatible
                # with this router and therefore use the coordinate fallback.
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
        transformed_group = self.query_group_projection(state.group_control)
        logits = torch.einsum("bqd,bkd->bqk", query_control, transformed_group)
        logits = logits / (float(self.control_dim) ** 0.5)
        logits = logits / float(self.query_temperature)
        assignment = entmax15(logits, dim=-1)
        return GroupQueryRoute(
            query_control=query_control,
            assignment=assignment,
            logits=logits,
        )

    def forward(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
    ) -> PreparedGroupControl:
        return self.prepare(encoded, module_states, environment_states)


__all__ = ["GroupQueryRoute", "LowDimensionalGroupRouter", "PreparedGroupControl"]
