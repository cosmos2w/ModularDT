"""Run-1408 hypergraph-conditioned sparse environmental quadrature.

The parent Run-1407 controller, dense MM/ME/EM preparation, module reader,
three-term context, and branch amplitude are retained.  This module replaces
only QE: every receiver/group pair proposes four sites and positive
within-group masses, then current-phase projected environmental K/V and the
projected scalar/head-control maps are read by local regular-grid
interpolation at those 24 sites.

The case adapter owns the grid.  ``EncodedInterfaceCase.sampler_layout`` must
be a validated :class:`RegularGridLayout` from ``environment_sampling.py``;
this generic backend never guesses a grid from token order and never falls
back to a dense environmental reader.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .environment_sampling import RegularGridLayout, sample_regular_grid
from .group_control_router import GroupQueryRoute, PreparedGroupControl
from .phase_shared_group_control import PhaseSharedGroupControlPairwiseField


def _require_layout(encoded: Any) -> RegularGridLayout:
    layout = getattr(encoded, "sampler_layout", None)
    if not isinstance(layout, RegularGridLayout):
        raise TypeError(
            "hypergraph_quadrature_honf requires adapter-supplied "
            "environment_sampling.RegularGridLayout metadata; no guessed grid "
            "or dense QE fallback is available."
        )
    token_count = int(encoded.env_coords.shape[1])
    if int(layout.num_tokens) != token_count:
        raise ValueError("sampler_layout token count must match encoded.env_coords.")
    # Metadata is adapter-owned, but checking the coordinate/token contract at
    # the boundary prevents a stale or permuted layout from silently reading a
    # different physical grid.  The adapter mapping itself remains the source
    # of truth for token order.
    x_axis, y_axis = layout.centre_axes
    mapping = layout.token_to_grid.to(device=encoded.env_coords.device)
    expected = torch.stack(
        [
            x_axis.to(encoded.env_coords)[mapping[:, 1]],
            y_axis.to(encoded.env_coords)[mapping[:, 0]],
        ],
        dim=-1,
    )
    coordinates = encoded.env_coords[0]
    if not torch.allclose(coordinates, expected, rtol=4.0e-5, atol=4.0e-6):
        raise ValueError("sampler_layout centre axes/token_to_grid do not match encoded.env_coords.")
    return layout


def _grid_to_token_ids(layout: RegularGridLayout, grid_indices: torch.Tensor) -> torch.Tensor:
    """Map row/column lattice indices to original adapter token IDs."""

    linear = layout.linearize_indices(grid_indices)
    token_to_linear = layout.linear_token_indices.to(device=linear.device)
    inverse = torch.empty((layout.num_tokens,), device=linear.device, dtype=torch.long)
    inverse.scatter_(
        0,
        token_to_linear,
        torch.arange(layout.num_tokens, device=linear.device),
    )
    return inverse[linear]


def interpolate_regular_grid(
    source_bank: torch.Tensor,
    sample_coordinates: torch.Tensor,
    layout: RegularGridLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a token-order bank and return executed interpolation metadata.

    ``source_bank`` is ``[B,C,E]`` and output values are ``[B,C,Q,J]``.
    Corner IDs are original adapter token IDs, rather than row-major lattice
    IDs, so visualizations remain valid under a consistent token permutation.
    """

    if source_bank.ndim != 3:
        raise ValueError("source_bank must have shape [B,C,E].")
    if sample_coordinates.ndim != 4 or int(sample_coordinates.shape[-1]) != 2:
        raise ValueError("sample_coordinates must have shape [B,Q,J,2].")
    # The environment_sampling API performs the explicit token-to-grid pack
    # and uses grid_sample(align_corners=True); incidence is computed from the
    # same centre-hull convention for actual work ledgers.
    grid_values = layout.tokens_to_grid(source_bank.transpose(1, 2))
    sampled = sample_regular_grid(
        grid_values,
        sample_coordinates,
        layout,
        return_incidence=True,
    )
    values = sampled.values.permute(0, 3, 1, 2)
    incidence = sampled.incidence
    corner_token_ids = _grid_to_token_ids(layout, incidence.corner_indices)
    cell_ids = layout.linearize_indices(incidence.cell_indices)
    return values, corner_token_ids, incidence.corner_weights, cell_ids


def stable_masked_log_normalize(
    scores: torch.Tensor,
    masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize positive mass-weighted scores without ``log(0)``/NaNs."""

    if scores.ndim != 4 or masses.ndim != 3:
        raise ValueError("scores must be [B,H,Q,J] and masses must be [B,Q,J].")
    tiny = torch.finfo(scores.dtype).tiny
    valid = masses > 0.0
    log_mass = torch.where(
        valid,
        torch.log(masses.clamp_min(tiny)),
        torch.zeros_like(masses),
    )
    weighted = scores + log_mass[:, None, :, :]
    weighted = torch.where(
        valid[:, None, :, :],
        weighted,
        torch.full_like(weighted, -torch.inf),
    )
    row_has_support = valid.any(dim=-1, keepdim=True)
    safe_weighted = torch.where(
        row_has_support[:, None, :, :],
        weighted,
        torch.zeros_like(weighted),
    )
    normalizer = torch.logsumexp(safe_weighted, dim=-1, keepdim=True)
    weights = torch.exp(safe_weighted - normalizer)
    weights = weights * valid[:, None, :, :].to(dtype=weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(tiny)
    return weights, row_has_support.squeeze(-1)


def mass_weighted_response(
    scores: torch.Tensor,
    values: torch.Tensor,
    masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the mass-weighted numerator/denominator response identity."""

    if values.ndim != 5 or tuple(values.shape[:4]) != tuple(scores.shape):
        raise ValueError("values must have shape [B,H,Q,J,D] aligned with scores.")
    weights, row_valid = stable_masked_log_normalize(scores, masses)
    response = torch.einsum("bhqj,bhqjd->bhqd", weights, values)
    response = torch.where(row_valid[:, None, :, None], response, torch.zeros_like(response))
    return response, row_valid


class HypergraphQuadratureField(PhaseSharedGroupControlPairwiseField):
    """Run-1408 phase-shared group-control field with sampled QE only."""

    phase_diagnostic_prefix = "group_control_"
    samples_per_group = 4

    def __init__(self, *args: object, samples_per_group: int = 4, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if int(self.group_count) != 6:
            raise ValueError("hypergraph_quadrature_honf requires group_count=6.")
        if int(self.group_control_dim) != 16:
            raise ValueError("hypergraph_quadrature_honf requires group_control_dim=16.")
        if int(self.spatial_dim) != 2:
            raise ValueError("hypergraph_quadrature_honf currently supports spatial_dim=2 only.")
        if int(samples_per_group) != 4:
            raise ValueError("hypergraph_quadrature_honf requires samples_per_group=4.")
        self.samples_per_group = int(samples_per_group)
        control_dim = int(self.group_control_dim)
        output_width = self.samples_per_group * (self.spatial_dim + 2)
        self.sample_mlp = nn.Sequential(
            nn.Linear(2 * control_dim, 2 * control_dim),
            nn.GELU(),
            nn.Linear(2 * control_dim, output_width),
        )
        nn.init.normal_(self.sample_mlp[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.sample_mlp[-1].bias)

        # Deterministic interior 6-by-4 normalized anchor table.  The table
        # is learnable but is not a sweep or a data-derived initialization.
        positions = [
            (
                (float(group) + 0.5) / float(self.group_count),
                (float(sample) + 0.5) / float(self.samples_per_group),
            )
            for group in range(self.group_count)
            for sample in range(self.samples_per_group)
        ]
        anchors = torch.tensor(positions, dtype=torch.get_default_dtype()).reshape(
            self.group_count,
            self.samples_per_group,
            self.spatial_dim,
        )
        self.reference_anchor_logits = nn.Parameter(torch.logit(anchors))

    @property
    def anchor_logits(self) -> torch.Tensor:
        return self.reference_anchor_logits

    def _sample_program(
        self,
        route: GroupQueryRoute,
        controls: PreparedGroupControl,
        receivers: torch.Tensor,
        layout: RegularGridLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate ``xi`` and within-group ``beta`` for one receiver tile."""

        batch, query_count, _ = receivers.shape
        groups = int(self.group_count)
        sites = int(self.samples_per_group)
        query_group = torch.cat(
            [
                route.query_control[:, :, None, :].expand(-1, -1, groups, -1),
                controls.group_control[:, None, :, :].expand(-1, query_count, -1, -1),
            ],
            dim=-1,
        )
        raw = self.sample_mlp(query_group).reshape(
            batch,
            query_count,
            groups,
            sites,
            self.spatial_dim + 2,
        )
        center_hull = layout.center_hull.to(device=raw.device, dtype=raw.dtype)
        # Predicted module-port receivers may sit just outside the fluid
        # rectangle while still being valid coupling locations. Quadrature
        # sites themselves must remain in Omega, so project only the receiver
        # reference point to the adapter-owned physical bounds before the
        # convex centre-hull construction. Keep the generic layout transform
        # strict: this clamping is a reader policy, not interpolation padding.
        physical_bounds = layout.physical_bounds.to(device=receivers.device, dtype=receivers.dtype)
        receiver_reference = torch.minimum(
            torch.maximum(receivers, physical_bounds[0]),
            physical_bounds[1],
        )
        query_center = layout.physical_to_centre_points(receiver_reference).to(
            device=raw.device,
            dtype=raw.dtype,
        )
        anchor_fraction = torch.sigmoid(
            self.reference_anchor_logits.to(device=raw.device, dtype=raw.dtype)[None, None]
            + raw[..., : self.spatial_dim]
        )
        anchors = center_hull[0] + (center_hull[1] - center_hull[0]) * anchor_fraction
        mixing = torch.sigmoid(raw[..., self.spatial_dim : self.spatial_dim + 1])
        coordinates = (
            (1.0 - mixing) * query_center[:, :, None, None, :]
            + mixing * anchors
        )
        beta = torch.softmax(raw[..., self.spatial_dim + 1], dim=-1)
        return coordinates, beta

    @staticmethod
    def _unique_cell_counts(cell_ids: torch.Tensor) -> torch.Tensor:
        sorted_ids = torch.sort(cell_ids, dim=-1).values
        return torch.ones_like(sorted_ids[..., 0]) + (
            sorted_ids[..., 1:] != sorted_ids[..., :-1]
        ).sum(dim=-1)

    def _quadrature_read(
        self,
        state: dict[str, Any],
        encoded: Any,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        route: GroupQueryRoute,
        *,
        return_maps: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        controls: PreparedGroupControl = state["group_control_state"]
        layout = _require_layout(encoded)
        coordinates, beta = self._sample_program(route, controls, receivers, layout)
        batch, query_count, groups, sites, _ = coordinates.shape
        sample_count = groups * sites
        flat_coordinates = coordinates.reshape(batch, query_count, sample_count, self.spatial_dim)
        masses = (
            route.assignment[:, :, :, None]
            * controls.environment_mass[:, None, :, None]
            * beta
        ).reshape(batch, query_count, sample_count)
        mass_total = masses.sum(dim=-1)

        # Interpolate current-phase projected K/V, including the inherited
        # source-side value gain.  No pooled group value enters this path.
        key_bank = state["environment_keys"]
        value_bank = state["environment_values"]
        # K/V arrive as [B,heads,E,head_dim].  Move head_dim next to the
        # head axis before flattening so each packed channel retains the
        # correct (head, feature) source row order.
        flat_key_bank = key_bank.permute(0, 1, 3, 2).reshape(
            batch, self.hidden_dim, int(key_bank.shape[2])
        )
        flat_value_bank = value_bank.permute(0, 1, 3, 2).reshape(
            batch, self.hidden_dim, int(value_bank.shape[2])
        )
        head_source = state["environment_head_source_control"]
        head_source_bank = head_source.permute(0, 1, 3, 2).reshape(
            batch,
            self.group_count * self.num_heads,
            int(head_source.shape[2]),
        )
        # One packed bank and one grid_sample call keep the fixed geometry and
        # sampled-bank accounting honest: [B, 2*H + K*n_h, Ny, Nx].
        packed_bank = torch.cat((flat_key_bank, flat_value_bank, head_source_bank), dim=1)
        sampled_packed_flat, corner_indices, corner_weights, cell_ids = interpolate_regular_grid(
            packed_bank,
            flat_coordinates,
            layout,
        )
        sampled_key_flat = sampled_packed_flat[:, : self.hidden_dim]
        sampled_value_flat = sampled_packed_flat[:, self.hidden_dim : 2 * self.hidden_dim]
        sampled_head_flat = sampled_packed_flat[:, 2 * self.hidden_dim :]
        sampled_keys = sampled_key_flat.reshape(
            batch,
            self.num_heads,
            self.head_dim,
            query_count,
            sample_count,
        ).permute(0, 1, 3, 4, 2)
        sampled_values = sampled_value_flat.reshape(
            batch,
            self.num_heads,
            self.head_dim,
            query_count,
            sample_count,
        ).permute(0, 1, 3, 4, 2)

        # The packed scalar/head-control channels are read at the same sites,
        # then contracted alpha over groups for zeta(q,xi).
        sampled_head = sampled_head_flat.reshape(
            batch,
            self.group_count,
            self.num_heads,
            query_count,
            sample_count,
        ).permute(0, 3, 4, 1, 2)
        zeta = torch.einsum("bqk,bqjkh->bhqj", route.assignment, sampled_head)

        query = self.env_attention.project_query(self.env_query(receiver_features))
        scores = (query[:, :, :, None, :] * sampled_keys).sum(dim=-1) / (float(self.head_dim) ** 0.5)
        scale = self._dense_coordinate_scale(encoded, batch)
        relative = (receivers[:, :, None, :] - flat_coordinates) / scale
        geometry = self._mlp(self.env_geometry_bias, self.relative_fourier(relative)).permute(0, 3, 1, 2)
        scores = scores * (1.0 + torch.tanh(zeta)) + geometry
        response, row_valid = mass_weighted_response(scores, sampled_values, masses)
        response = response.transpose(1, 2).reshape(batch, query_count, self.hidden_dim)
        response = self.env_attention.output(response)
        # Retain exactly the parent's K*G_q branch amplitude and remove the
        # output bias too when G_q is exactly zero.
        context = float(self.group_count) * mass_total[..., None] * response
        context = torch.where(mass_total[..., None] > 0.0, context, torch.zeros_like(context))

        # A sampled site's executed incidence is its four corner token loads,
        # including zero-weight corners at singleton axes.  Count the union of
        # those actual token IDs per receiver; the lower-cell IDs remain a
        # separate visualization map and are not substituted for accesses.
        corner_unique = self._unique_cell_counts(corner_indices.reshape(batch, query_count, sample_count * 4))
        lower_cell_unique = self._unique_cell_counts(cell_ids)
        nonzero_mass = (masses > 0.0).sum(dim=-1).to(receivers.dtype)
        rows_per_query = receivers.new_full((batch, query_count), float(sample_count))
        corners_per_query = rows_per_query * 4.0
        packed_bank_elements = int(sampled_packed_flat.numel())
        key_value_elements = int(sampled_keys.numel() + sampled_values.numel())
        bank_bytes = float(packed_bank_elements * sampled_packed_flat.element_size())
        aux: dict[str, torch.Tensor] = {
            "group_control_environment_overlap_mass_per_query": mass_total,
            "group_control_environment_sample_slots_per_query": rows_per_query,
            "group_control_environment_nonzero_sample_mass_per_query": nonzero_mass,
            "group_control_environment_fine_rows_per_query": rows_per_query,
            "group_control_environment_content_dot_rows_per_query": rows_per_query * float(self.num_heads),
            "group_control_environment_geometry_rows_per_query": rows_per_query,
            "group_control_environment_interpolation_corner_loads_per_query": corners_per_query,
            "group_control_environment_unique_cells_touched_per_query": corner_unique.to(receivers.dtype),
            "group_control_environment_lower_cells_touched_per_query": lower_cell_unique.to(receivers.dtype),
            "group_control_environment_sample_coordinates": coordinates,
            "group_control_environment_sample_beta": beta,
            "group_control_environment_sample_lambda": masses.reshape(batch, query_count, groups, sites),
            "group_control_environment_sample_mass": masses.reshape(batch, query_count, groups, sites),
            "group_control_environment_interpolation_corner_indices": corner_indices.reshape(batch, query_count, groups, sites, 4),
            "group_control_environment_interpolation_corner_weights": corner_weights.reshape(batch, query_count, groups, sites, 4),
            "group_control_environment_interpolation_cell_indices": cell_ids.reshape(batch, query_count, groups, sites),
            "group_control_environment_sampled_bank_bytes": receivers.new_tensor(bank_bytes),
            "group_control_environment_sampled_bank_key_value_elements": receivers.new_tensor(float(key_value_elements)),
            "group_control_environment_sampled_bank_elements": receivers.new_tensor(float(packed_bank_elements)),
            "group_control_environment_sampled_row_valid": row_valid.to(receivers.dtype),
            # These counters describe executed fixed slots, including slots
            # with zero mass.  They are deliberately independent of A^E.
            "group_control_environment_sample_slots": rows_per_query.sum(),
            "group_control_environment_nonzero_sample_masses": nonzero_mass.sum(),
            "group_control_environment_fine_rows": rows_per_query.sum(),
            "group_control_environment_fine_rows_forward": rows_per_query.sum(),
            "group_control_environment_geometry_rows_forward": rows_per_query.sum(),
            "group_control_environment_content_dot_rows_forward": rows_per_query.sum() * float(self.num_heads),
            "group_control_environment_interpolation_corner_loads": corners_per_query.sum(),
            "group_control_environment_unique_cells_touched": corner_unique.sum(),
            "group_control_environment_lower_cells_touched": lower_cell_unique.sum(),
            "group_control_environment_checkpoint_recomputations": receivers.new_tensor(0.0),
            "group_control_environment_fine_rows_recompute": receivers.new_tensor(0.0),
            "group_control_environment_complete_support": receivers.new_tensor(0.0),
            "group_control_environment_partial_support": receivers.new_tensor(1.0),
        }
        if return_maps:
            return context, mass_total, aux
        compact = {
            key: value
            for key, value in aux.items()
            if key.endswith(
                (
                    "_per_query",
                    "_bytes",
                    "_elements",
                    "_rows",
                    "_masses",
                    "_slots",
                    "_loads",
                    "_cells_touched",
                    "_support",
                    "_recomputations",
                    "_recompute",
                    "_forward",
                )
            )
        }
        return context, mass_total, compact

    def _read_environment(
        self,
        state: dict[str, Any],
        encoded: Any,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        route: GroupQueryRoute,
        *,
        include_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run only the sampled QE block, with the parent checkpoint seam."""

        checkpoint_active = self._checkpoint_active()
        if checkpoint_active and not include_diagnostics:
            def recompute(
                read_receivers: torch.Tensor,
                read_features: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                context, mass, aux = self._quadrature_read(
                    state,
                    encoded,
                    read_receivers,
                    read_features,
                    route,
                    return_maps=False,
                )
                return (
                    context,
                    mass,
                    aux["group_control_environment_nonzero_sample_mass_per_query"].detach(),
                    aux["group_control_environment_unique_cells_touched_per_query"].detach(),
                )

            context, mass, nonzero_mass, unique_cells = checkpoint(
                recompute,
                receivers,
                receiver_features,
                use_reentrant=False,
            )
            batch, query_count = int(receivers.shape[0]), int(receivers.shape[1])
            sample_count = int(self.group_count * self.samples_per_group)
            rows = receivers.new_tensor(float(batch * query_count * sample_count))
            packed_channels = int(2 * self.hidden_dim + self.group_count * self.num_heads)
            packed_elements = int(batch * query_count * sample_count * packed_channels)
            key_value_elements = int(2 * batch * query_count * sample_count * self.hidden_dim)
            bank_bytes = float(packed_elements * state["environment_keys"].element_size())
            compact = {
                "group_control_environment_overlap_mass_per_query": mass,
                "group_control_environment_sample_slots_per_query": receivers.new_full((batch, query_count), float(sample_count)),
                "group_control_environment_nonzero_sample_mass_per_query": nonzero_mass,
                "group_control_environment_unique_cells_touched_per_query": unique_cells,
                "group_control_environment_sample_slots": rows,
                "group_control_environment_nonzero_sample_masses": nonzero_mass.sum(),
                "group_control_environment_fine_rows": rows,
                "group_control_environment_fine_rows_forward": rows,
                "group_control_environment_geometry_rows_forward": rows,
                "group_control_environment_content_dot_rows_forward": rows * float(self.num_heads),
                "group_control_environment_interpolation_corner_loads": rows * 4.0,
                "group_control_environment_unique_cells_touched": unique_cells.sum(),
                "group_control_environment_sampled_bank_bytes": receivers.new_tensor(bank_bytes),
                "group_control_environment_sampled_bank_key_value_elements": receivers.new_tensor(float(key_value_elements)),
                "group_control_environment_sampled_bank_elements": receivers.new_tensor(float(packed_elements)),
                "group_control_environment_checkpoint_recomputations": rows,
                "group_control_environment_fine_rows_recompute": rows,
                "group_control_environment_complete_support": receivers.new_tensor(0.0),
                "group_control_environment_partial_support": receivers.new_tensor(1.0),
            }
            return context, compact

        context, mass, aux = self._quadrature_read(
            state,
            encoded,
            receivers,
            receiver_features,
            route,
            return_maps=bool(include_diagnostics),
        )
        if not include_diagnostics:
            return context, {"group_control_environment_overlap_mass_per_query": mass}
        return context, aux

    def read_environment(
        self,
        state: dict[str, Any],
        encoded: Any,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        route: GroupQueryRoute | None = None,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

    def read(
        self,
        state: dict[str, Any],
        encoded: Any,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Use the inherited QM and only the sampled QE replacement."""

        debug = bool(return_routing_maps)
        route = self._route(state, encoded, receivers, receiver_features)
        module_context, module_aux = self._read_module(state, encoded, receivers, route, include_diagnostics=debug)
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
            "group_control_query_control_norm": torch.linalg.vector_norm(route.query_control, dim=-1),
            "group_read_degree": (route.assignment > 0.0).sum(dim=-1).to(receivers.dtype),
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
                    "group_control_environment_head_control": state["environment_head_control"],
                    "group_control_environment_overlap": self._overlap(route.assignment, controls.environment_membership),
                }
            )
            if route.query_keys is not None:
                aux["group_control_query_keys"] = route.query_keys
        return module_context + environment_context, {key: value.detach() for key, value in aux.items()}


__all__ = [
    "HypergraphQuadratureField",
    "interpolate_regular_grid",
    "mass_weighted_response",
    "stable_masked_log_normalize",
]
