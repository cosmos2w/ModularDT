"""Run-1406 low-dimensional group-control pairwise reader.

The backend keeps Dense's physical MM/ME/EM preparation and replaces only the
receiver-side group ensemble.  Six assignment columns are collapsed to a
scalar overlap and a bounded D-wide control moment before the expensive fine
function is called.  Consequently a logical source/group multiplicity never
causes another H-wide module function or another environmental K/V bank.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .dense_pairwise import DensePairwiseField
from .group_control_router import (
    GroupQueryRoute,
    LowDimensionalGroupRouter,
    PreparedGroupControl,
)
from .types import EncodedInterfaceCase


class GroupControlPairwiseField(DensePairwiseField):
    """Dense preparation plus one fine evaluation per unique source pair."""

    phase_interaction_diagnostics = True
    phase_diagnostic_prefix = "group_control_"

    def __init__(
        self,
        hidden_dim: int,
        message_hidden_dim: int,
        num_heads: int,
        fourier_frequencies: int,
        *,
        group_count: int = 6,
        group_control_dim: int = 16,
        spatial_dim: int = 2,
        module_temperature: float = 1.0,
        environment_temperature: float = 1.0,
        query_temperature: float = 1.0,
        activation_checkpointing: bool = False,
        query_tile_size: int = 128,
        source_tile_size: int = 128,
    ) -> None:
        super().__init__(
            hidden_dim,
            message_hidden_dim,
            num_heads,
            fourier_frequencies,
            activation_checkpointing=activation_checkpointing,
        )
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if int(query_tile_size) <= 0 or int(source_tile_size) <= 0:
            raise ValueError("query_tile_size and source_tile_size must be positive.")
        self.hidden_dim = int(hidden_dim)
        self.message_hidden_dim = int(message_hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.group_count = int(group_count)
        self.group_control_dim = int(group_control_dim)
        self.spatial_dim = int(spatial_dim)
        self.query_tile_size = int(query_tile_size)
        self.source_tile_size = int(source_tile_size)
        self.router = LowDimensionalGroupRouter(
            hidden_dim,
            group_count=group_count,
            control_dim=group_control_dim,
            fourier_frequencies=fourier_frequencies,
            spatial_dim=spatial_dim,
            module_temperature=module_temperature,
            environment_temperature=environment_temperature,
            query_temperature=query_temperature,
        )

        # These are the only new pair-control maps.  They are bias-free so a
        # zero moment is an identity modulation, not a learned additive route.
        self.module_control_gain = nn.Linear(
            self.group_control_dim,
            self.message_hidden_dim,
            bias=False,
        )
        self.environment_value_control = nn.Linear(
            self.group_control_dim,
            self.hidden_dim,
            bias=False,
        )
        self.environment_score_control = nn.Linear(
            self.group_control_dim,
            self.num_heads,
            bias=False,
        )

    # ------------------------------------------------------------------
    # Preparation and first-affine materialization
    # ------------------------------------------------------------------
    @staticmethod
    def _dense_coordinate_scale(
        encoded: EncodedInterfaceCase,
        batch: int,
    ) -> torch.Tensor:
        """Broadcast coordinate scales for Dense's four-dimensional pairs."""

        scale = encoded.coordinate_scale
        if scale.ndim == 1:
            return scale[None, None, None, :]
        if scale.ndim == 2:
            if int(scale.shape[0]) not in {1, batch}:
                raise ValueError("coordinate_scale batch dimension does not match the case batch.")
            return scale[:, None, None, :]
        if scale.ndim == 3:
            if int(scale.shape[0]) not in {1, batch} or int(scale.shape[1]) != 1:
                raise ValueError("coordinate_scale must have shape [d], [B,d], or [B,1,d].")
            return scale[:, :, None, :]
        raise ValueError("coordinate_scale must have shape [d], [B,d], or [B,1,d].")

    def _ensure_module_first_layer(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
    ) -> nn.Linear:
        """Materialize Dense QM's LazyLinear without duplicating its weights."""

        first = self.query_module_message.net[0]
        has_uninitialized = getattr(first, "has_uninitialized_params", None)
        if callable(has_uninitialized) and bool(has_uninitialized()):
            zeros = module_states.new_zeros(1, self.spatial_dim)
            relative_width = int(self.relative_fourier(zeros).shape[-1])
            # Materialize with a real source/global row and an explicitly
            # zero relative feature, matching the split affine contract.
            dummy = torch.cat(
                [
                    module_states.reshape(-1, self.hidden_dim)[:1],
                    module_states.new_zeros(1, relative_width),
                    encoded.global_token[:1],
                ],
                dim=-1,
            )
            # This one-row call only materializes LazyLinear.  It is not a
            # receiver/source computation and is never part of the read.
            self.query_module_message(dummy)
            first = self.query_module_message.net[0]
        if not isinstance(first, nn.Linear):
            raise TypeError("Dense QM first layer did not materialize as nn.Linear.")
        return first

    def _prepare_module_affine(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
    ) -> torch.Tensor:
        """Cache ``W_z z_i + W_g g + b`` for the split Dense QM affine."""

        first = self._ensure_module_first_layer(encoded, module_states)
        weight = first.weight
        bias = first.bias
        relative_width = int(weight.shape[1]) - 2 * self.hidden_dim
        if relative_width <= 0:
            raise RuntimeError("Dense QM first layer has no relative-coordinate slice.")
        if int(weight.shape[0]) != self.message_hidden_dim:
            raise RuntimeError("Dense QM first-layer width differs from message_hidden_dim.")
        source_weight = weight[:, : self.hidden_dim]
        global_weight = weight[:, self.hidden_dim + relative_width :]
        source_affine = F.linear(module_states, source_weight, None)
        global_affine = F.linear(encoded.global_token, global_weight, None)[:, None, :]
        return source_affine + global_affine + bias[None, None, :]

    def _module_first_slices(self) -> tuple[torch.Tensor, int]:
        first = self.query_module_message.net[0]
        if not isinstance(first, nn.Linear):
            raise TypeError("Dense QM first layer is not materialized.")
        relative_width = int(first.weight.shape[1]) - 2 * self.hidden_dim
        if relative_width <= 0:
            raise RuntimeError("Dense QM first layer has an invalid relative slice.")
        return first.weight[:, self.hidden_dim : self.hidden_dim + relative_width], relative_width

    def _module_tail(self, values: torch.Tensor) -> torch.Tensor:
        """Apply Dense QM's layers after its first GELU exactly once."""

        layers = self.query_module_message.net
        if len(layers) < 5:
            raise RuntimeError("Dense QM must contain first affine, GELU, J-width affine, GELU, and output.")
        values = layers[2](values)
        values = layers[3](values)
        return layers[4](values)

    def _module_psi(
        self,
        source_affine: torch.Tensor,
        relative_features: torch.Tensor,
        control_moment: torch.Tensor,
    ) -> torch.Tensor:
        relative_weight, _ = self._module_first_slices()
        relative_affine = F.linear(relative_features, relative_weight, None)
        hidden = F.gelu(source_affine + relative_affine)
        gain = 1.0 + torch.tanh(self.module_control_gain(control_moment))
        hidden = hidden * gain
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self._module_tail, hidden, use_reentrant=False)
        return self._module_tail(hidden)

    def _prepare_environment_bank(
        self,
        contextual_env: torch.Tensor,
        controls: PreparedGroupControl,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare one inherited K/V bank and source-local value controls."""

        key, value = self.env_attention.project_source(contextual_env)
        source_group_control = torch.einsum(
            "bek,bkd->bed",
            controls.environment_membership,
            controls.group_control,
        )
        gain = 1.0 + torch.tanh(self.environment_value_control(source_group_control))
        gain = gain.reshape(
            contextual_env.shape[0],
            contextual_env.shape[1],
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        # The QE control is projected once from D to heads.  Keeping the
        # source contraction in ``[B,K,E,heads]`` form lets each receiver tile
        # contract alpha against it without ever creating ``[B,Q,E,D]``.
        head_control = self.environment_score_control(controls.group_control)
        head_source_control = (
            controls.environment_membership.transpose(1, 2).unsqueeze(-1)
            * head_control.unsqueeze(2)
        )
        return key, value * gain, source_group_control, head_control, head_source_control

    @staticmethod
    def _group_centres(
        membership: torch.Tensor,
        measure: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Return diagnostic centres, with NaN for genuinely empty groups."""

        mass = torch.einsum("bs,bsk->bk", measure, membership)
        numerator = torch.einsum("bs,bsk,bsd->bkd", measure, membership, coordinates)
        # The empty branch is selected after the division; using one as the
        # denominator there avoids introducing a tiny pseudo-mass into the
        # diagnostic centre calculation.
        safe_mass = torch.where(mass > 0.0, mass, torch.ones_like(mass))
        centres = numerator / safe_mass[..., None]
        return torch.where(
            mass[..., None] > 0.0,
            centres,
            torch.full_like(centres, float("nan")),
        )

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        """Reuse Dense MM/ME/EM preparation and cache only small controls."""

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
        controls = self.router.prepare(encoded, fine["module_tokens"], contextual_env)
        module_affine = self._prepare_module_affine(encoded, fine["module_tokens"])
        (
            environment_key,
            environment_value,
            source_group_control,
            head_control,
            head_source_control,
        ) = self._prepare_environment_bank(
            contextual_env,
            controls,
        )
        # The module control moment is a contraction over the fixed K group
        # axis.  Materialize the source/group/control product once in a
        # GEMM-friendly [B,K,M*D] bank; each receiver chunk then performs one
        # batched alpha @ bank operation instead of a generic four-index
        # einsum.  This is runtime preparation state only and introduces no
        # trainable parameters or state-dict entries.
        module_control_bank = (
            controls.module_membership.transpose(1, 2).unsqueeze(-1)
            * controls.group_control.unsqueeze(2)
        ).reshape(
            int(module_states.shape[0]),
            self.group_count,
            int(module_states.shape[1]) * self.group_control_dim,
        )
        preparation_aux = {
            "group_control_module_mass": controls.module_mass.detach(),
            "group_control_environment_mass": controls.environment_mass.detach(),
            "group_control_group_control_norm": torch.linalg.vector_norm(
                controls.group_control,
                dim=-1,
            ).detach(),
        }
        batch = int(module_states.shape[0])
        module_incidence = (
            (controls.module_membership > 0.0)
            & (controls.module_measure[..., None] > 0.0)
        )
        environment_incidence = (
            (controls.environment_membership > 0.0)
            & (controls.environment_measure[..., None] > 0.0)
        )
        active_groups = (controls.module_mass > 0.0) | (
            controls.environment_mass > 0.0
        )
        preparation_aux.update(
            {
                # These compact learned-routing summaries are the
                # group-control analogue of active-edge monitoring.  They
                # reuse memberships already required by the predictor and do
                # not enable full maps, support gathers, or work ledgers.
                "group_count_per_case": active_groups.sum(dim=-1).to(
                    module_states.dtype
                ).detach(),
                "module_group_incidence_count_per_case": module_incidence.sum(
                    dim=(1, 2)
                ).to(module_states.dtype).detach(),
                "environment_group_incidence_count_per_case": (
                    environment_incidence.sum(dim=(1, 2))
                    .to(module_states.dtype)
                    .detach()
                ),
                "group_module_degree": module_incidence.sum(dim=1).to(
                    module_states.dtype
                ).detach(),
                "group_environment_degree": environment_incidence.sum(dim=1).to(
                    module_states.dtype
                ).detach(),
            }
        )
        module_rows = module_states.new_full((batch,), float(module_states.shape[1]))
        module_valid_rows = (controls.module_measure > 0.0).sum(dim=-1).to(module_states.dtype)
        environment_rows = module_states.new_full(
            (batch,), float(contextual_env.shape[1])
        )
        preparation_aux.update(
            {
                # These are preparation-time source projections, distinct
                # from per-query fine rows reported by ``read``.  The valid /
                # padded split makes the module rectangular work explicit.
                "group_control_module_source_projection_rows": module_rows.detach(),
                "group_control_module_source_projection_valid_rows": module_valid_rows.detach(),
                "group_control_module_source_projection_padded_rows": (
                    module_rows - module_valid_rows
                ).detach(),
                "group_control_environment_source_projection_rows": environment_rows.detach(),
                "group_control_environment_source_projection_valid_rows": environment_rows.detach(),
                "group_control_environment_source_projection_padded_rows": module_states.new_zeros(
                    (batch,)
                ).detach(),
                "group_control_module_qm_first_affine_rows": module_rows.detach(),
                "group_control_module_qm_first_affine_valid_rows": module_valid_rows.detach(),
                "group_control_module_qm_first_affine_padded_rows": (
                    module_rows - module_valid_rows
                ).detach(),
                "group_control_module_qm_global_affine_rows": module_states.new_full(
                    (batch,), 1.0
                ).detach(),
                "group_control_environment_kv_projection_rows": environment_rows.detach(),
                "group_control_environment_kv_projection_valid_rows": environment_rows.detach(),
                "group_control_environment_kv_projection_padded_rows": module_states.new_zeros(
                    (batch,)
                ).detach(),
                "group_control_environment_value_control_rows": environment_rows.detach(),
                "group_control_environment_head_control_projection_rows": module_states.new_full(
                    (batch,), float(self.group_count)
                ).detach(),
            }
        )
        if return_routing_maps:
            preparation_aux.update(
                {
                    "group_control_module_centres": self._group_centres(
                        controls.module_membership,
                        controls.module_measure,
                        encoded.module_centers,
                    ).detach(),
                    "group_control_environment_centres": self._group_centres(
                        controls.environment_membership,
                        controls.environment_measure,
                        encoded.env_coords,
                    ).detach(),
                }
            )
        return {
            "module_tokens": fine["module_tokens"],
            "env_tokens": contextual_env,
            "group_control_state": controls,
            "module_first_affine": module_affine,
            "module_control_bank": module_control_bank,
            # Keep tensor identities so focused tests/evidence callers that
            # replace the immutable PreparedGroupControl can request a safe
            # fallback rebuild without affecting the normal hot path.
            "module_control_bank_membership": controls.module_membership,
            "module_control_bank_group_control": controls.group_control,
            "environment_keys": environment_key,
            "environment_values": environment_value,
            "environment_source_group_control": source_group_control,
            "environment_head_control": head_control,
            "environment_head_source_control": head_source_control,
            "group_control_preparation_aux": preparation_aux,
        }

    def preparation_aux(
        self,
        state: dict[str, Any],
        *,
        include_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return detached summaries, with full memberships only on request."""

        controls: PreparedGroupControl = state["group_control_state"]
        summary = {
            key: value.detach()
            for key, value in state.get("group_control_preparation_aux", {}).items()
        }
        if include_diagnostics:
            summary.update(
                {
                    "group_control_module_incidence": controls.module_membership.detach(),
                    "group_control_environment_incidence": controls.environment_membership.detach(),
                    "group_control_module_measure": controls.module_measure.detach(),
                    "group_control_environment_measure": controls.environment_measure.detach(),
                    "group_control_module_control": controls.module_control.detach(),
                    "group_control_environment_control": controls.environment_control.detach(),
                    "group_control_h": controls.group_control.detach(),
                    "group_control_global": controls.global_control.detach(),
                    "group_control_environment_head_control": state[
                        "environment_head_control"
                    ].detach(),
                }
            )
        return summary

    # ------------------------------------------------------------------
    # Control contractions and support accounting
    # ------------------------------------------------------------------
    def _checkpoint_active(self) -> bool:
        return bool(self.activation_checkpointing and self.training and torch.is_grad_enabled())

    @staticmethod
    def _overlap(alpha: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
        return torch.bmm(alpha, membership.transpose(1, 2))

    def _module_control_bank(self, state: dict[str, Any]) -> torch.Tensor:
        """Return the prepared ``[B,K,M*D]`` module-control bank.

        ``PreparedGroupControl`` is immutable, but a few evidence/unit-test
        paths intentionally replace it in a copied runtime state.  Rebuild in
        that exceptional case; the ordinary prepared state takes the cached
        bank with no extra contraction.
        """

        controls: PreparedGroupControl = state["group_control_state"]
        bank = state.get("module_control_bank")
        if (
            torch.is_tensor(bank)
            and state.get("module_control_bank_membership") is controls.module_membership
            and state.get("module_control_bank_group_control") is controls.group_control
        ):
            return bank
        return (
            controls.module_membership.transpose(1, 2).unsqueeze(-1)
            * controls.group_control.unsqueeze(2)
        ).reshape(
            int(controls.module_membership.shape[0]),
            self.group_count,
            int(controls.module_membership.shape[1]) * self.group_control_dim,
        )

    @staticmethod
    def _logical_paths(alpha: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
        return torch.bmm(
            (alpha > 0.0).to(dtype=alpha.dtype),
            (membership > 0.0).to(dtype=alpha.dtype).transpose(1, 2),
        )

    @staticmethod
    def _complete_support(
        overlap: torch.Tensor,
        source_measure: torch.Tensor,
        *,
        require_all_sources: bool = False,
    ) -> bool:
        valid_source = source_measure > 0.0
        if not bool(valid_source.any()):
            return False
        # The rectangular complete path is only valid when every padded slot
        # is absent.  Otherwise it would execute a fine row for a zero-mass
        # source and overstate the strict active-pair work bound.
        if require_all_sources and not bool(valid_source.all()):
            return False
        return bool(((overlap > 0.0) | ~valid_source[:, None, :]).all())

    @staticmethod
    def _pair_controls(
        alpha: torch.Tensor,
        membership: torch.Tensor,
        group_control: torch.Tensor,
        batch_index: torch.Tensor,
        query_index: torch.Tensor,
        source_index: torch.Tensor,
        *,
        overlap: torch.Tensor | None = None,
        chunk_size: int = 131_072,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(rho,n)`` chunks for unique physical pair indices."""

        total = int(batch_index.shape[0])
        for start in range(0, total, int(chunk_size)):
            stop = min(start + int(chunk_size), total)
            batches = batch_index[start:stop]
            queries = query_index[start:stop]
            sources = source_index[start:stop]
            alpha_pair = alpha[batches, queries]
            membership_pair = membership[batches, sources]
            controls = group_control[batches]
            if overlap is None:
                rho = (alpha_pair * membership_pair).sum(dim=-1)
            else:
                rho = overlap[batches, queries, sources]
            moment = torch.einsum(
                "pk,pk,pkd->pd",
                alpha_pair,
                membership_pair,
                controls,
            )
            yield rho, moment

    def _environment_control_tile(
        self,
        state: dict[str, Any],
        route: GroupQueryRoute,
        query_start: int,
        query_stop: int,
        source_start: int,
        source_stop: int,
        *,
        overlap: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scalar overlap and per-head QE control for one tile.

        The source-side ``A_e * B_score(h)`` contraction is prepared once.
        This is the Section 9.3 head-space contraction; in particular, no
        receiver/environment tensor with a trailing D control dimension is
        materialized.
        """

        controls: PreparedGroupControl = state["group_control_state"]
        alpha_tile = route.assignment[:, query_start:query_stop]
        membership_tile = controls.environment_membership[:, source_start:source_stop]
        if overlap is None:
            rho = torch.bmm(alpha_tile, membership_tile.transpose(1, 2))
        else:
            rho = overlap[:, query_start:query_stop, source_start:source_stop]
        source_control = state["environment_head_source_control"][
            :, :, source_start:source_stop, :
        ]
        batch, groups, sources, heads = source_control.shape
        zeta = torch.bmm(
            alpha_tile,
            source_control.reshape(batch, groups, sources * heads),
        ).reshape(batch, query_stop - query_start, sources, heads)
        return rho, zeta.permute(0, 3, 1, 2)

    def _environment_control_full(
        self,
        state: dict[str, Any],
        route: GroupQueryRoute,
    ) -> torch.Tensor:
        """Contract all query/source controls for one receiver chunk."""

        source_control = state["environment_head_source_control"]
        batch, groups, sources, heads = source_control.shape
        zeta = torch.bmm(
            route.assignment,
            source_control.reshape(batch, groups, sources * heads),
        ).reshape(batch, route.assignment.shape[1], sources, heads)
        return zeta.permute(0, 3, 1, 2)

    @staticmethod
    def _pair_head_controls(
        alpha: torch.Tensor,
        membership: torch.Tensor,
        head_control: torch.Tensor,
        batch_index: torch.Tensor,
        query_index: torch.Tensor,
        source_index: torch.Tensor,
        *,
        overlap: torch.Tensor | None = None,
        chunk_size: int = 131_072,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield exact ``(rho,zeta)`` chunks with zeta already in head space."""

        total = int(batch_index.shape[0])
        for start in range(0, total, int(chunk_size)):
            stop = min(start + int(chunk_size), total)
            batches = batch_index[start:stop]
            queries = query_index[start:stop]
            sources = source_index[start:stop]
            alpha_pair = alpha[batches, queries]
            membership_pair = membership[batches, sources]
            heads = head_control[batches]
            if overlap is None:
                rho = (alpha_pair * membership_pair).sum(dim=-1)
            else:
                rho = overlap[batches, queries, sources]
            zeta = torch.einsum(
                "pk,pk,pkh->ph",
                alpha_pair,
                membership_pair,
                heads,
            )
            yield rho, zeta

    def _support_indices(
        self,
        overlap: torch.Tensor,
        source_measure: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nonzero(
            (overlap > 0.0) & (source_measure[:, None, :] > 0.0),
            as_tuple=False,
        )

    # ------------------------------------------------------------------
    # Module reader
    # ------------------------------------------------------------------
    def _module_finalize(
        self,
        weighted_sum: torch.Tensor,
        overlap_mass: torch.Tensor,
        active_module_count: torch.Tensor,
    ) -> torch.Tensor:
        output = self.query_module_output
        linear = F.linear(weighted_sum, output.weight, None)
        a_m = active_module_count / (1.0 + active_module_count)
        return (
            float(self.group_count) * a_m[:, None, None] * linear
            + float(self.group_count) * overlap_mass[..., None] * output.bias
        )

    def _read_module_complete(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        route: GroupQueryRoute,
        overlap: torch.Tensor,
        *,
        source_measure: torch.Tensor,
        active_module_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_count, _ = receivers.shape
        scales = self._dense_coordinate_scale(encoded, batch)
        # The core owns the receiver memory bound.  Consume this complete
        # chunk in one rectangular pass; ``query_tile_size`` remains relevant
        # only to the opt-in gathered/evidence reference path.
        rho = overlap
        control_bank = self._module_control_bank(state)
        moment = torch.bmm(route.assignment, control_bank).reshape(
            batch,
            query_count,
            int(encoded.module_centers.shape[1]),
            self.group_control_dim,
        )
        relative = (
            receivers[:, :, None, :] - encoded.module_centers[:, None, :, :]
        ) / scales
        psi = self._module_psi(
            state["module_first_affine"][:, None, :, :],
            self.relative_fourier(relative),
            moment,
        )
        weighted = psi * source_measure[:, None, :, None] * rho[..., None]
        weighted_sum = weighted.sum(dim=2)
        overlap_mass = (source_measure[:, None, :] * rho).sum(dim=2)
        return self._module_finalize(weighted_sum, overlap_mass, active_module_count), overlap_mass

    def _read_module_partial(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        route: GroupQueryRoute,
        overlap: torch.Tensor,
        *,
        source_measure: torch.Tensor,
        active_module_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_count, _ = receivers.shape
        pair = self._support_indices(overlap, source_measure)
        weighted_sum = receivers.new_zeros(batch * query_count, self.hidden_dim)
        overlap_mass = receivers.new_zeros(batch * query_count)
        if int(pair.shape[0]) == 0:
            return self._module_finalize(
                weighted_sum.reshape(batch, query_count, self.hidden_dim),
                overlap_mass.reshape(batch, query_count),
                active_module_count,
            ), overlap_mass.reshape(batch, query_count)
        scales = encoded.coordinate_scale
        if scales.ndim == 1:
            scales = scales[None, :]
        elif scales.ndim == 3:
            scales = scales[:, 0, :]
        if int(scales.shape[0]) == 1 and batch != 1:
            scales = scales.expand(batch, -1)
        for start in range(0, int(pair.shape[0]), self.source_tile_size * self.query_tile_size):
            stop = min(start + self.source_tile_size * self.query_tile_size, int(pair.shape[0]))
            batches, queries, sources = pair[start:stop].unbind(dim=1)
            pair_rho, pair_moment = next(
                self._pair_controls(
                    route.assignment,
                    state["group_control_state"].module_membership,
                    state["group_control_state"].group_control,
                    batches,
                    queries,
                    sources,
                    overlap=overlap,
                    chunk_size=stop - start,
                )
            )
            relative = (
                receivers[batches, queries] - encoded.module_centers[batches, sources]
            ) / scales[batches]
            psi = self._module_psi(
                state["module_first_affine"][batches, sources],
                self.relative_fourier(relative),
                pair_moment,
            )
            weight = source_measure[batches, sources] * pair_rho
            query_global = batches * query_count + queries
            weighted_sum.index_add_(0, query_global, psi * weight[:, None])
            overlap_mass.index_add_(0, query_global, weight)
        overlap_mass = overlap_mass.reshape(batch, query_count)
        return self._module_finalize(
            weighted_sum.reshape(batch, query_count, self.hidden_dim),
            overlap_mass,
            active_module_count,
        ), overlap_mass

    def _read_module(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        route: GroupQueryRoute,
        *,
        include_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        controls: PreparedGroupControl = state["group_control_state"]
        overlap = self._overlap(route.assignment, controls.module_membership)
        # Normal Run-1406 execution is deliberately rectangular.  The
        # support test and gathered fallback remain available to an explicit
        # evidence/map request, where their bookkeeping is untimed.
        complete = (
            self._complete_support(
                overlap,
                controls.module_measure,
                require_all_sources=True,
            )
            if include_diagnostics
            else True
        )
        active_count = (encoded.module_present > 0.5).sum(dim=-1).to(receivers.dtype)
        if complete:
            context, overlap_mass = self._read_module_complete(
                state,
                encoded,
                receivers,
                route,
                overlap,
                source_measure=controls.module_measure,
                active_module_count=active_count,
            )
        else:
            context, overlap_mass = self._read_module_partial(
                state,
                encoded,
                receivers,
                route,
                overlap,
                source_measure=controls.module_measure,
                active_module_count=active_count,
            )
        if not include_diagnostics:
            # The context is the only prediction output; retain the overlap
            # mass because it is a lightweight maintained diagnostic, while
            # keeping logical paths, per-query work ledgers, and provenance
            # out of the timed/training path.
            return context, {
                "group_control_module_overlap_mass_per_query": overlap_mass,
            }
        support = (overlap > 0.0) & (controls.module_measure[:, None, :] > 0.0)
        logical = self._logical_paths(route.assignment, controls.module_membership)
        valid_denominator = (
            (controls.module_measure > 0.0).sum(dim=-1).to(receivers.dtype)
            * float(receivers.shape[1])
        ).sum()
        padded_denominator = receivers.new_tensor(float(overlap.numel())) - valid_denominator
        aux = {
            "group_control_module_unique_pairs_per_query": support.sum(dim=-1).to(receivers.dtype),
            "group_control_module_logical_paths_per_query": logical.sum(dim=-1),
            "group_control_module_overlap_mass_per_query": overlap_mass,
            "group_control_module_unique_pairs": support.sum().to(receivers.dtype),
            "group_control_module_logical_paths": logical.sum(),
            "group_control_module_fine_rows": support.sum().to(receivers.dtype),
            "group_control_module_fine_rows_forward": support.sum().to(receivers.dtype),
            "group_control_module_fine_rows_padded": receivers.new_zeros(()),
            "group_control_module_fine_forward_rows": support.sum().to(receivers.dtype),
            "group_control_module_padded_rows": receivers.new_zeros(()),
            "group_control_module_valid_pair_denominator": valid_denominator,
            "group_control_module_padded_pair_denominator": padded_denominator,
            "group_control_module_checkpoint_recomputations": (
                support.sum().to(receivers.dtype)
                if self._checkpoint_active()
                else receivers.new_zeros(())
            ),
            "group_control_module_fine_rows_recompute": (
                support.sum().to(receivers.dtype)
                if self._checkpoint_active()
                else receivers.new_zeros(())
            ),
            "group_control_module_scalar_control_rows": receivers.new_tensor(
                float(overlap.numel())
            ),
            "group_control_module_complete_support": receivers.new_tensor(float(complete)),
            "group_control_module_partial_support": receivers.new_tensor(float(not complete)),
        }
        return context, aux

    def read_module(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        route: GroupQueryRoute | None = None,
    ) -> torch.Tensor:
        if route is None:
            route = self._route(state, encoded, receivers, receiver_features)
        context, _ = self._read_module(state, encoded, receivers, route)
        return context

    # ------------------------------------------------------------------
    # Environment reader
    # ------------------------------------------------------------------
    def _environment_geometry_bias(
        self,
        receivers: torch.Tensor,
        source_coords: torch.Tensor,
        encoded: EncodedInterfaceCase,
    ) -> torch.Tensor:
        scale = encoded.coordinate_scale
        if scale.ndim == 1:
            scale = scale[None, None, None, :]
        elif scale.ndim == 2:
            scale = scale[:, None, None, :]
        elif scale.ndim == 3:
            scale = scale[:, :, None, :]
        relative = (receivers[:, :, None, :] - source_coords[:, None, :, :]) / scale
        return self._mlp(
            self.env_geometry_bias,
            self.relative_fourier(relative),
        )

    def _read_environment_complete(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        route: GroupQueryRoute,
        overlap: torch.Tensor,
        *,
        source_measure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_count, _ = receivers.shape
        query = self.env_attention.project_query(self.env_query(receiver_features))
        keys = state["environment_keys"]
        values = state["environment_values"]
        # ``overlap`` is computed once by ``_read_environment`` and is reused
        # for both the control contraction and the value normalization.
        rho = overlap
        score_control = self._environment_control_full(state, route)
        scores = torch.matmul(query, keys.transpose(-1, -2)) / (float(self.head_dim) ** 0.5)
        geometry = self._environment_geometry_bias(
            receivers,
            encoded.env_coords,
            encoded,
        ).permute(0, 3, 1, 2)
        scores = scores * (1.0 + torch.tanh(score_control)) + geometry
        tiny = torch.finfo(scores.dtype).tiny
        safe_measure = source_measure.clamp_min(tiny)
        safe_rho = rho.clamp_min(tiny)
        scores = scores + torch.log(safe_measure)[:, None, None, :]
        scores = scores + torch.log(safe_rho)[:, None, :, :]
        valid = (rho > 0.0) & (source_measure[:, None, :] > 0.0)
        valid_heads = valid[:, None, :, :]
        # Set unsupported entries to -inf before softmax.  A separate tensor
        # row-support guard avoids the all--inf softmax NaN while retaining an
        # exactly zero response for receivers with no environment support.
        masked_scores = torch.where(
            valid_heads,
            scores,
            torch.full_like(scores, -torch.inf),
        )
        row_has_support = valid_heads.any(dim=-1, keepdim=True)
        safe_scores = torch.where(row_has_support, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.softmax(safe_scores, dim=-1)
        weights = weights * valid_heads.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(tiny)
        response = torch.matmul(weights, values).transpose(1, 2).reshape(
            batch,
            query_count,
            self.hidden_dim,
        )
        response = self.env_attention.output(response)
        overlap_mass = (source_measure[:, None, :] * rho).sum(dim=-1)
        context = float(self.group_count) * overlap_mass[..., None] * response
        return context, overlap_mass

    def _read_environment_partial(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        route: GroupQueryRoute,
        overlap: torch.Tensor,
        *,
        source_measure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_count, _ = receivers.shape
        query = self.env_attention.project_query(self.env_query(receiver_features))
        keys = state["environment_keys"]
        values = state["environment_values"]
        controls: PreparedGroupControl = state["group_control_state"]
        context = receivers.new_zeros(batch, query_count, self.hidden_dim)
        overlap_mass = receivers.new_zeros(batch, query_count)
        for query_start in range(0, query_count, self.query_tile_size):
            query_stop = min(query_start + self.query_tile_size, query_count)
            rho, _ = self._environment_control_tile(
                state,
                route,
                query_start,
                query_stop,
                0,
                int(encoded.env_coords.shape[1]),
                overlap=overlap,
            )
            overlap_mass[:, query_start:query_stop] = (
                source_measure[:, None, :] * rho
            ).sum(dim=-1)
            pair = torch.nonzero(
                (rho > 0.0) & (source_measure[:, None, :] > 0.0),
                as_tuple=False,
            )
            if int(pair.shape[0]) == 0:
                continue
            pair_batch, pair_query, pair_source = pair.unbind(dim=1)
            pair_rho, pair_zeta = next(
                self._pair_head_controls(
                    route.assignment,
                    controls.environment_membership,
                    state["environment_head_control"],
                    pair_batch,
                    pair_query + query_start,
                    pair_source,
                    overlap=overlap,
                    chunk_size=int(pair.shape[0]),
                )
            )
            # ``pair_query`` is tile-local; query projections are prepared for
            # the complete receiver set, so restore the global row here.
            query_pair = query[
                pair_batch, :, query_start + pair_query, :
            ].permute(1, 0, 2)
            key_pair = keys[pair_batch, :, pair_source, :].permute(1, 0, 2)
            scores = (query_pair * key_pair).sum(dim=-1) / (float(self.head_dim) ** 0.5)
            score_control = pair_zeta.transpose(0, 1)
            source_coordinates = encoded.env_coords[pair_batch, pair_source]
            query_coordinates = receivers[pair_batch, query_start + pair_query]
            scale = encoded.coordinate_scale
            if scale.ndim == 1:
                scale = scale[None, :]
            elif scale.ndim == 3:
                scale = scale[:, 0, :]
            if int(scale.shape[0]) == 1 and batch != 1:
                scale = scale.expand(batch, -1)
            pair_relative = (query_coordinates - source_coordinates) / scale[pair_batch]
            geometry = self._mlp(
                self.env_geometry_bias,
                self.relative_fourier(pair_relative),
            ).transpose(0, 1)
            scores = scores * (1.0 + torch.tanh(score_control)) + geometry
            scores = scores + torch.log(
                source_measure[pair_batch, pair_source]
            )[None, :]
            scores = scores + torch.log(pair_rho)[None, :]
            local_query = pair_batch * (query_stop - query_start) + pair_query
            segment_index = local_query[None, :].expand(self.num_heads, -1)
            max_scores = torch.full(
                (self.num_heads, batch * (query_stop - query_start)),
                -torch.inf,
                device=scores.device,
                dtype=scores.dtype,
            )
            max_scores.scatter_reduce_(1, segment_index, scores, reduce="amax", include_self=True)
            unnormalized = torch.exp(scores - max_scores[:, local_query])
            normalizers = scores.new_zeros((self.num_heads, batch * (query_stop - query_start)))
            normalizers.index_add_(1, local_query, unnormalized)
            attention = unnormalized / normalizers[:, local_query]
            value_pair = values[pair_batch, :, pair_source, :].permute(1, 0, 2)
            response_flat = receivers.new_zeros(
                self.num_heads,
                batch * (query_stop - query_start),
                self.head_dim,
            )
            response_flat.index_add_(
                1,
                local_query,
                attention[:, :, None] * value_pair,
            )
            response = response_flat.permute(1, 0, 2).reshape(
                batch,
                query_stop - query_start,
                self.hidden_dim,
            )
            response = self.env_attention.output(response)
            context[:, query_start:query_stop] = (
                float(self.group_count)
                * overlap_mass[:, query_start:query_stop, None]
                * response
            )
        return context, overlap_mass

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
        controls: PreparedGroupControl = state["group_control_state"]
        overlap = self._overlap(route.assignment, controls.environment_membership)
        complete = (
            self._complete_support(overlap, controls.environment_measure)
            if include_diagnostics
            else True
        )
        if complete:
            if self._checkpoint_active():
                # The complete QE calculation materializes the score-control,
                # score, geometry, support-mask, softmax, and renormalization
                # tensors for the whole receiver/source rectangle.  Keep the
                # boundary around the complete calculation so those tensors
                # are recomputed during backward; overlap mass remains an
                # explicit output for the normal auxiliary path.
                def recompute_complete(
                    read_receivers: torch.Tensor,
                    read_features: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    return self._read_environment_complete(
                        state,
                        encoded,
                        read_receivers,
                        read_features,
                        route,
                        overlap,
                        source_measure=controls.environment_measure,
                    )

                context, overlap_mass = checkpoint(
                    recompute_complete,
                    receivers,
                    receiver_features,
                    use_reentrant=False,
                )
            else:
                context, overlap_mass = self._read_environment_complete(
                    state,
                    encoded,
                    receivers,
                    receiver_features,
                    route,
                    overlap,
                    source_measure=controls.environment_measure,
                )
        else:
            context, overlap_mass = self._read_environment_partial(
                state,
                encoded,
                receivers,
                receiver_features,
                route,
                overlap,
                source_measure=controls.environment_measure,
            )
        if not include_diagnostics:
            return context, {
                "group_control_environment_overlap_mass_per_query": overlap_mass,
            }
        support = (overlap > 0.0) & (controls.environment_measure[:, None, :] > 0.0)
        logical = self._logical_paths(route.assignment, controls.environment_membership)
        valid_denominator = (
            (controls.environment_measure > 0.0).sum(dim=-1).to(receivers.dtype)
            * float(receivers.shape[1])
        ).sum()
        padded_denominator = receivers.new_tensor(float(overlap.numel())) - valid_denominator
        aux = {
            "group_control_environment_unique_pairs_per_query": support.sum(dim=-1).to(receivers.dtype),
            "group_control_environment_logical_paths_per_query": logical.sum(dim=-1),
            "group_control_environment_overlap_mass_per_query": overlap_mass,
            "group_control_environment_unique_pairs": support.sum().to(receivers.dtype),
            "group_control_environment_logical_paths": logical.sum(),
            "group_control_environment_fine_rows": support.sum().to(receivers.dtype),
            "group_control_environment_fine_rows_forward": support.sum().to(receivers.dtype),
            "group_control_environment_fine_rows_padded": receivers.new_zeros(()),
            "group_control_environment_fine_forward_rows": support.sum().to(receivers.dtype),
            "group_control_environment_padded_rows": receivers.new_zeros(()),
            "group_control_environment_valid_pair_denominator": valid_denominator,
            "group_control_environment_padded_pair_denominator": padded_denominator,
            "group_control_environment_geometry_rows_forward": support.sum().to(receivers.dtype),
            "group_control_environment_content_dot_rows_forward": (
                support.sum().to(receivers.dtype) * float(self.num_heads)
            ),
            "group_control_environment_scalar_control_rows": receivers.new_tensor(
                float(overlap.numel())
            ),
            "group_control_environment_checkpoint_recomputations": (
                support.sum().to(receivers.dtype)
                if self._checkpoint_active()
                else receivers.new_zeros(())
            ),
            "group_control_environment_fine_rows_recompute": (
                support.sum().to(receivers.dtype)
                if self._checkpoint_active()
                else receivers.new_zeros(())
            ),
            "group_control_environment_complete_support": receivers.new_tensor(float(complete)),
            "group_control_environment_partial_support": receivers.new_tensor(float(not complete)),
        }
        return context, aux

    def read_environment(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        route: GroupQueryRoute | None = None,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read the environmental term using the fixed-group API shape.

        The low-dimensional backend exposes packed pair diagnostics through
        :meth:`read`; it does not materialize a dense attention map here.
        """

        if route is None:
            route = self._route(state, encoded, receivers, receiver_features)
        context, _ = self._read_environment(
            state,
            encoded,
            receivers,
            receiver_features,
            route,
            include_diagnostics=bool(return_routing_maps),
        )
        return context, None

    # ------------------------------------------------------------------
    # Public read and opt-in maps
    # ------------------------------------------------------------------
    def _route(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor | None = None,
    ) -> GroupQueryRoute:
        return self.router.route_queries(
            encoded,
            state["group_control_state"],
            receivers,
            receiver_features,
        )

    @staticmethod
    def _pair_map(
        overlap: torch.Tensor,
        source_measure: torch.Tensor,
        *,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        indices = torch.nonzero(
            (overlap > 0.0) & (source_measure[:, None, :] > 0.0),
            as_tuple=False,
        )
        return {
            f"{prefix}_pair_batch": indices[:, 0],
            f"{prefix}_pair_query": indices[:, 1],
            f"{prefix}_pair_source": indices[:, 2],
        }

    def read(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return exactly ``C_M + C_E`` plus small execution diagnostics."""

        debug = bool(return_routing_maps)
        route = self._route(state, encoded, receivers, receiver_features)
        module_context, module_aux = self._read_module(
            state,
            encoded,
            receivers,
            route,
            include_diagnostics=debug,
        )
        environment_context, environment_aux = self._read_environment(
            state,
            encoded,
            receivers,
            receiver_features,
            route,
            include_diagnostics=debug,
        )
        aux: dict[str, torch.Tensor] = {
            **module_aux,
            **environment_aux,
            "group_control_query_control_norm": torch.linalg.vector_norm(
                route.query_control,
                dim=-1,
            ),
            # Entmax produces exact zeros, so this is a direct learned
            # query-to-group activity count rather than a thresholded proxy.
            "group_read_degree": (route.assignment > 0.0).sum(dim=-1).to(
                receivers.dtype
            ),
        }
        if return_routing_maps:
            controls: PreparedGroupControl = state["group_control_state"]
            aux.update(
                {
                    "group_control_query_routing": route.assignment,
                    "group_control_query_logits": route.logits,
                    "group_control_module_incidence": controls.module_membership,
                    "group_control_environment_incidence": controls.environment_membership,
                    "group_control_h": controls.group_control,
                    "group_control_environment_head_control": state[
                        "environment_head_control"
                    ],
                    "group_control_module_overlap": self._overlap(
                        route.assignment,
                        controls.module_membership,
                    ),
                    "group_control_environment_overlap": self._overlap(
                        route.assignment,
                        controls.environment_membership,
                    ),
                }
            )
            aux.update(
                self._pair_map(
                    aux["group_control_module_overlap"],
                    controls.module_measure,
                    prefix="group_control_module",
                )
            )
            aux.update(
                self._pair_map(
                    aux["group_control_environment_overlap"],
                    controls.environment_measure,
                    prefix="group_control_environment",
                )
            )
        # Diagnostics are evidence-only outputs.  Detaching them keeps a
        # timed/training read from retaining a second graph; the returned
        # context remains fully differentiable.
        return module_context + environment_context, {
            key: value.detach() for key, value in aux.items()
        }


__all__ = ["GroupControlPairwiseField"]
