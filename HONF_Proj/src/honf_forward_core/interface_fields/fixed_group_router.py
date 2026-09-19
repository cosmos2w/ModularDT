"""Fixed-six-group routing for the Run-1405 interface field.

The router owns the *control* side of the fixed hypergraph.  It does not
produce a field value: memberships, centres, and the group control states are
consumed by :mod:`fixed_group_pairwise` to condition fine source/query
interactions.  Keeping this boundary explicit is useful both for the
scientific ablation (remove ``h_k`` from fine kernels) and for avoiding an
accidental direct group-value bypass into the field head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from honf_forward_core.nn import MLP, FourierFeatures, LazyMLP
from honf_forward_core.routing import entmax15

from .types import EncodedInterfaceCase


@dataclass(frozen=True)
class FixedGroupState:
    """Prepared fixed-group memberships and control states.

    All tensors retain the live autograd graph.  The ``*_membership`` fields
    have shape ``[B, source, K]`` and are exact-zero outside entmax support;
    inactive modules are zero in ``module_membership`` as well.  ``valid`` is
    a physical occupancy mask, not a learned value mask, and is used only to
    suppress reads from empty groups.
    """

    module_membership: torch.Tensor
    environment_membership: torch.Tensor
    module_mass: torch.Tensor
    environment_mass: torch.Tensor
    module_centres: torch.Tensor
    environment_centres: torch.Tensor
    group_state: torch.Tensor
    valid: torch.Tensor
    module_support: torch.Tensor
    environment_support: torch.Tensor

    # The mathematical notation in the Run-1405 plan uses A_m, A_e, r_m,
    # r_e, m_m, m_e, and h.  These aliases make tests/evidence tooling able to
    # use that notation without duplicating the state or exposing new tensors.
    @property
    def A_m(self) -> torch.Tensor:
        return self.module_membership

    @property
    def A_e(self) -> torch.Tensor:
        return self.environment_membership

    @property
    def r_m(self) -> torch.Tensor:
        return self.module_centres

    @property
    def r_e(self) -> torch.Tensor:
        return self.environment_centres

    @property
    def m_m(self) -> torch.Tensor:
        return self.module_mass

    @property
    def m_e(self) -> torch.Tensor:
        return self.environment_mass

    @property
    def h(self) -> torch.Tensor:
        return self.group_state


@dataclass(frozen=True)
class FixedGroupQueryRoute:
    """Query-to-group descriptors and entmax routing weights."""

    descriptor: torch.Tensor
    assignment: torch.Tensor
    logits: torch.Tensor

    @property
    def u_q(self) -> torch.Tensor:
        return self.descriptor

    @property
    def alpha(self) -> torch.Tensor:
        return self.assignment


class FixedGroupRouter(nn.Module):
    """Differentiable fixed-``K=6`` module/environment/query router.

    ``module_states`` and ``environment_states`` are expected to be the
    contextualized fine source states produced by the Dense preparation pass.
    The router is deliberately agnostic to the adapter feature widths: the
    only runtime-width input is ``receiver_features`` in
    :meth:`route_queries`, which is handled by a ``LazyMLP``.

    The three normalizers are intentionally fixed to exact 1.5-entmax with a
    unit temperature in this first experiment.  ``h_k`` is masked to zero for
    physically empty groups and is never itself a field-value contribution.
    """

    GROUP_COUNT = 6

    def __init__(
        self,
        hidden_dim: int,
        *,
        group_count: int = GROUP_COUNT,
        group_code_dim: int = 32,
        fourier_frequencies: int = 4,
        spatial_dim: int = 2,
        module_temperature: float = 1.0,
        environment_temperature: float = 1.0,
        query_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if int(group_count) != self.GROUP_COUNT:
            raise ValueError("Run-1405 FixedGroupRouter requires exactly group_count=6.")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive.")
        if int(group_code_dim) <= 0:
            raise ValueError("group_code_dim must be positive.")
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
        self.group_count = self.GROUP_COUNT
        self.group_code_dim = int(group_code_dim)
        self.spatial_dim = int(spatial_dim)
        self.module_temperature = float(module_temperature)
        self.environment_temperature = float(environment_temperature)
        self.query_temperature = float(query_temperature)

        code_scale = float(self.group_code_dim) ** -0.5
        self.group_codes = nn.Parameter(
            torch.randn(self.group_count, self.group_code_dim) * code_scale
        )
        # These are normalized-domain anchors.  Sigmoid keeps the fallback in
        # [0, 1] while retaining a live gradient whenever a group is empty.
        self.fallback_anchor_logits = nn.Parameter(
            torch.zeros(self.group_count, self.spatial_dim)
        )

        source_input_dim = 2 * self.hidden_dim + self.group_code_dim
        self.module_score = MLP(
            source_input_dim,
            self.hidden_dim,
            1,
            num_layers=2,
        )
        self.environment_score = MLP(
            source_input_dim,
            self.hidden_dim,
            1,
            num_layers=2,
        )
        self.relative_fourier = FourierFeatures(None, int(fourier_frequencies))
        self.environment_geometry_bias = LazyMLP(
            self.hidden_dim,
            out_dim=1,
            num_layers=2,
        )
        self.group_state = LazyMLP(self.hidden_dim, num_layers=2)
        self.query_descriptor = LazyMLP(self.hidden_dim, num_layers=2)
        self.query_score = nn.Linear(self.hidden_dim, 1)
        self.query_position_fourier = FourierFeatures(None, int(fourier_frequencies))

    @property
    def normalized_fallback_anchors(self) -> torch.Tensor:
        """Return learned fallback anchors in normalized physical coordinates."""

        return torch.sigmoid(self.fallback_anchor_logits)

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
        if int(scale.shape[0]) not in {1, int(encoded.global_token.shape[0])}:
            raise ValueError("encoded.coordinate_scale batch dimension does not match the case batch.")
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
            raise ValueError("encoded.coordinate_scale must be finite and strictly positive.")
        if int(scale.shape[0]) == 1:
            scale = scale.expand(int(encoded.global_token.shape[0]), -1, -1)
        return scale

    def _fallback_centres(self, encoded: EncodedInterfaceCase) -> torch.Tensor:
        scale = self._scale(encoded)
        normalized = self.normalized_fallback_anchors.to(
            device=scale.device,
            dtype=scale.dtype,
        )
        # The interface-field coordinate convention is a zero-origin domain
        # with ``coordinate_scale`` as its physical extent.  Keep this map
        # explicit so empty-group behaviour is deterministic and testable.
        return normalized[None, :, :] * scale[:, :1, :]

    @staticmethod
    def _weighted_centres(
        memberships: torch.Tensor,
        coordinates: torch.Tensor,
        mass: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        weighted = torch.einsum("bsk,bsd->bkd", memberships, coordinates)
        safe_mass = mass.clamp_min(torch.finfo(mass.dtype).tiny)
        centres = weighted / safe_mass[..., None]
        return torch.where((mass > 0.0)[..., None], centres, fallback)

    def _source_logits(
        self,
        source_states: torch.Tensor,
        global_token: torch.Tensor,
        scorer: nn.Module,
    ) -> torch.Tensor:
        batch, source_count, _ = source_states.shape
        codes = self.group_codes.to(device=source_states.device, dtype=source_states.dtype)
        source = source_states[:, :, None, :].expand(-1, -1, self.group_count, -1)
        global_values = global_token[:, None, None, :].expand(-1, source_count, self.group_count, -1)
        code_values = codes[None, None, :, :].expand(batch, source_count, -1, -1)
        return scorer(torch.cat([source, global_values, code_values], dim=-1)).squeeze(-1)

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
    ) -> FixedGroupState:
        """Build memberships, centres, and ``h_k`` for one source state."""

        if module_states.ndim != 3 or environment_states.ndim != 3:
            raise ValueError("module_states and environment_states must have shape [B,S,H].")
        batch = int(module_states.shape[0])
        if tuple(environment_states.shape[:1]) != (batch,):
            raise ValueError("module_states and environment_states must share their batch dimension.")
        if int(module_states.shape[-1]) != self.hidden_dim or int(environment_states.shape[-1]) != self.hidden_dim:
            raise ValueError("Source state width does not match FixedGroupRouter.hidden_dim.")
        if encoded.global_token.shape != (batch, self.hidden_dim):
            raise ValueError("encoded.global_token must have shape [B, hidden_dim].")
        if encoded.module_centers.shape[:2] != module_states.shape[:2]:
            raise ValueError("module_states and encoded.module_centers must align.")
        if encoded.env_coords.shape[:2] != environment_states.shape[:2]:
            raise ValueError("environment_states and encoded.env_coords must align.")
        if int(encoded.module_centers.shape[-1]) != self.spatial_dim:
            raise ValueError("Module coordinate dimension does not match the router.")
        if int(encoded.env_coords.shape[-1]) != self.spatial_dim:
            raise ValueError("Environment coordinate dimension does not match the router.")

        active_modules = encoded.module_present > 0.5
        module_logits = self._source_logits(module_states, encoded.global_token, self.module_score)
        module_membership = entmax15(
            module_logits / self.module_temperature,
            dim=-1,
            mask=active_modules[..., None],
        )
        # The mask above already gives exact zero rows.  Keep the explicit
        # multiplication as a defensive invariant if a caller supplies a
        # non-binary presence tensor.
        module_membership = module_membership * active_modules[..., None].to(module_membership.dtype)
        module_mass = module_membership.sum(dim=1)

        fallback = self._fallback_centres(encoded)
        module_centres = self._weighted_centres(
            module_membership,
            encoded.module_centers,
            module_mass,
            fallback,
        )

        environment_logits = self._source_logits(
            environment_states,
            encoded.global_token,
            self.environment_score,
        )
        scale = self._scale(encoded)
        environment_relative = (
            encoded.env_coords[:, :, None, :] - module_centres[:, None, :, :]
        ) / scale[:, :, None, :]
        environment_bias = self.environment_geometry_bias(
            self.relative_fourier(environment_relative)
        ).squeeze(-1)
        environment_membership = entmax15(
            (environment_logits + environment_bias) / self.environment_temperature,
            dim=-1,
        )
        environment_mass = torch.einsum(
            "be,bek->bk", encoded.env_weights, environment_membership
        )
        environment_centres = self._weighted_centres(
            environment_membership * encoded.env_weights[..., None],
            encoded.env_coords,
            environment_mass,
            fallback,
        )

        module_summary = torch.einsum(
            "bmk,bmh->bkh", module_membership, module_states
        ) / module_mass.clamp_min(torch.finfo(module_mass.dtype).tiny)[..., None]
        environment_summary = torch.einsum(
            "bek,beh->bkh",
            environment_membership * encoded.env_weights[..., None],
            environment_states,
        ) / environment_mass.clamp_min(torch.finfo(environment_mass.dtype).tiny)[..., None]

        module_position = self.relative_fourier(module_centres / scale)
        environment_position = self.relative_fourier(environment_centres / scale)
        codes = self.group_codes.to(device=module_states.device, dtype=module_states.dtype)
        code_values = codes[None, :, :].expand(batch, -1, -1)
        group_input = torch.cat(
            [
                module_summary,
                environment_summary,
                encoded.global_token[:, None, :].expand(-1, self.group_count, -1),
                module_position,
                environment_position,
                torch.log1p(module_mass)[..., None],
                torch.log1p(environment_mass)[..., None],
                code_values,
            ],
            dim=-1,
        )
        group_state = self.group_state(group_input)
        valid = (module_mass > 0.0) | (environment_mass > 0.0)
        group_state = group_state * valid[..., None].to(group_state.dtype)

        return FixedGroupState(
            module_membership=module_membership,
            environment_membership=environment_membership,
            module_mass=module_mass,
            environment_mass=environment_mass,
            module_centres=module_centres,
            environment_centres=environment_centres,
            group_state=group_state,
            valid=valid,
            module_support=(module_membership > 0.0).sum(dim=1),
            environment_support=(environment_membership > 0.0).sum(dim=1),
        )

    def route_queries(
        self,
        encoded: EncodedInterfaceCase,
        state: FixedGroupState,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
    ) -> FixedGroupQueryRoute:
        """Return one shared entmax ``alpha_qk`` route for both fine terms."""

        if receivers.ndim != 3 or int(receivers.shape[0]) != int(state.group_state.shape[0]):
            raise ValueError("receivers must have shape [B,Q,d] aligned with the prepared state.")
        if int(receivers.shape[-1]) != self.spatial_dim:
            raise ValueError("Receiver coordinate dimension does not match the router.")
        scale = self._scale(encoded)
        if receiver_features is None:
            receiver_features = self.query_position_fourier(receivers / scale)
        if receiver_features.ndim != 3 or tuple(receiver_features.shape[:2]) != tuple(receivers.shape[:2]):
            raise ValueError("receiver_features must align with receivers as [B,Q,F].")
        if receiver_features.device != receivers.device:
            raise ValueError("receiver_features and receivers must share a device.")

        _, query_count, _ = receivers.shape
        group_states = state.group_state[:, None, :, :].expand(-1, query_count, -1, -1)
        global_values = encoded.global_token[:, None, None, :].expand(
            -1, query_count, self.group_count, -1
        )
        module_relative = (
            receivers[:, :, None, :] - state.module_centres[:, None, :, :]
        ) / scale[:, :, None, :]
        environment_relative = (
            receivers[:, :, None, :] - state.environment_centres[:, None, :, :]
        ) / scale[:, :, None, :]
        query_input = torch.cat(
            [
                receiver_features[:, :, None, :].expand(-1, -1, self.group_count, -1),
                group_states,
                global_values,
                self.relative_fourier(module_relative),
                self.relative_fourier(environment_relative),
            ],
            dim=-1,
        )
        descriptor = self.query_descriptor(query_input)
        logits = self.query_score(descriptor).squeeze(-1) / self.query_temperature
        assignment = entmax15(
            logits,
            dim=-1,
            mask=state.valid[:, None, :],
        )
        return FixedGroupQueryRoute(
            descriptor=descriptor,
            assignment=assignment,
            logits=logits,
        )

    def forward(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        environment_states: torch.Tensor,
    ) -> FixedGroupState:
        """Alias for :meth:`prepare` for small standalone numerical tests."""

        return self.prepare(encoded, module_states, environment_states)


__all__ = ["FixedGroupQueryRoute", "FixedGroupRouter", "FixedGroupState"]
