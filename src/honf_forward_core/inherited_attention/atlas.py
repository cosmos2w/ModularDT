"""Typed atlas construction and continuous geometry-owned receiver covers."""

from __future__ import annotations

import torch
from torch import Tensor

from .contracts import PreparedGridTopology, ReceiverCover, RouteScoreSnapshot, TypedAssociationAtlas
from .normalization import measure_relative_normalize


def route_raw_logits(snapshot: RouteScoreSnapshot) -> Tensor:
    """Return the already constructed, live ``[B,H,E,N]`` score matrix."""
    return snapshot.raw_logits


def build_typed_atlas(
    snapshot: RouteScoreSnapshot, *, normalization: str = "softmax"
) -> TypedAssociationAtlas:
    """Normalize one last-cycle EM/EE score snapshot over its own donor type."""
    donor_valid = snapshot.donor_mask & (snapshot.donor_measure > 0)
    probabilities, normalized_measure = measure_relative_normalize(
        route_raw_logits(snapshot),
        snapshot.donor_measure,
        snapshot.donor_mask,
        snapshot.destination_mask,
        normalization=normalization,
    )
    return TypedAssociationAtlas(
        kind="module" if snapshot.route == "em" else "environment",
        raw_logits=snapshot.raw_logits,
        probabilities=probabilities,
        anchor_ids=snapshot.anchor_ids,
        donor_ids=snapshot.donor_ids,
        donor_valid=donor_valid,
        destination_valid=snapshot.destination_mask,
        normalized_measure=normalized_measure,
        normalization=normalization,
    )


def prepare_grid_topology(
    anchor_coordinates: Tensor,
    environment_ids: Tensor,
    environment_mask: Tensor,
    *,
    frame_key,
    scene_version=None,
    owner_token=None,
    normalizer_key=None,
    scene_key=None,
    legal_bounds: Tensor | None = None,
) -> PreparedGridTopology:
    """Prepare the scene-owned sorted axes and lattice-to-token mapping once."""
    if anchor_coordinates.ndim != 3 or anchor_coordinates.shape[-1] not in (2, 3):
        raise ValueError("anchor_coordinates must have shape [B,E,D] with D equal to two or three.")
    batch, batch_anchor_count, coordinate_dim = anchor_coordinates.shape
    if environment_ids.ndim != 2 or environment_ids.shape != (batch, batch_anchor_count):
        raise ValueError("environment_ids must have shape [B,E] matching anchor_coordinates.")
    if environment_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Environment IDs must use an integer tensor dtype.")
    if environment_mask.shape != environment_ids.shape or environment_mask.dtype != torch.bool:
        raise ValueError("environment_mask must be boolean [B,E].")
    if any(tensor.device != anchor_coordinates.device for tensor in (environment_ids, environment_mask)):
        raise ValueError("Grid mapping, environment IDs and coordinates must share a device.")
    if not torch.isfinite(anchor_coordinates[environment_mask]).all():
        raise ValueError("Present environment anchor coordinates must be finite.")
    if (environment_mask.sum(dim=-1) == 0).any():
        raise ValueError("Every scene must have at least one present environment anchor.")

    axis_source_indices: list[Tensor] = []
    axis_values: list[Tensor] = []
    axis_sizes: list[int] = []
    for axis_index in range(coordinate_dim):
        coordinate = anchor_coordinates[..., axis_index]
        sorted_values, sorted_source_indices = torch.sort(
            torch.where(environment_mask, coordinate, torch.full_like(coordinate, torch.inf)),
            dim=-1,
            stable=True,
        )
        sorted_valid = torch.isfinite(sorted_values)
        previous = torch.cat((torch.full_like(sorted_values[:, :1], -torch.inf), sorted_values[:, :-1]), dim=-1)
        group_start = sorted_valid & (sorted_values != previous)
        group_count = group_start.sum(dim=-1)
        if not torch.equal(group_count, group_count[:1].expand_as(group_count)):
            raise ValueError("Every batched scene must have the same regular-grid axis sizes.")
        axis_size = int(group_count[0])
        if axis_size < 1:
            raise ValueError("A receiver grid axis cannot be empty.")
        unique_positions = torch.where(
            group_start,
            torch.arange(batch_anchor_count, device=anchor_coordinates.device)[None, :].expand(batch, -1),
            torch.full_like(sorted_source_indices, batch_anchor_count),
        ).sort(dim=-1, stable=True).values[:, :axis_size]
        # Select an exact source coordinate to preserve native grid identities.
        source_indices = torch.gather(sorted_source_indices, 1, unique_positions)
        axis_source_indices.append(source_indices)
        axis_values.append(torch.gather(anchor_coordinates[..., axis_index], 1, source_indices))
        axis_sizes.append(axis_size)

    expected_lattice_size = 1
    for size in axis_sizes:
        expected_lattice_size *= size
    expected_counts = torch.full(
        (batch,), expected_lattice_size, dtype=torch.long, device=environment_mask.device
    )
    if not torch.equal(environment_mask.sum(dim=-1), expected_counts):
        raise ValueError("Present environment anchors must form a complete regular grid.")
    grid_axes = [torch.arange(size, device=environment_ids.device) for size in axis_sizes]
    grid_tuple = torch.stack(torch.meshgrid(*grid_axes, indexing="ij"), dim=-1).reshape(
        expected_lattice_size, coordinate_dim
    )
    grid_coordinates = torch.stack(
        [torch.gather(axis, 1, grid_tuple[:, dim][None, :].expand(batch, -1))
         for dim, axis in enumerate(axis_values)],
        dim=-1,
    )
    grid_matches = (grid_coordinates[:, :, None, :] == anchor_coordinates[:, None, :, :]).all(dim=-1)
    grid_matches = grid_matches & environment_mask[:, None, :]
    match_count = grid_matches.sum(dim=-1)
    if not torch.equal(match_count, torch.ones_like(match_count)):
        raise ValueError("Present environment coordinates must map one-to-one to a complete regular grid.")
    scene_positions = torch.arange(batch_anchor_count, device=environment_ids.device)[None, None, :]
    grid_anchor_indices = torch.where(grid_matches, scene_positions, batch_anchor_count).amin(dim=-1)
    grid_anchor_indices = grid_anchor_indices.reshape(batch, *axis_sizes)

    if legal_bounds is not None:
        if legal_bounds.shape != (batch, 2, coordinate_dim):
            raise ValueError("legal_bounds must have shape [B,2,D].")
        if legal_bounds.device != anchor_coordinates.device:
            raise ValueError("legal_bounds must share the environment coordinate device.")
        if not torch.isfinite(legal_bounds).all() or (legal_bounds[:, 1] < legal_bounds[:, 0]).any():
            raise ValueError("legal_bounds must be finite with ordered lower/upper bounds.")
    return PreparedGridTopology(
        axis_source_indices=tuple(axis_source_indices),
        grid_anchor_indices=grid_anchor_indices,
        anchor_coordinates=anchor_coordinates,
        environment_ids=environment_ids,
        environment_mask=environment_mask,
        legal_bounds=legal_bounds,
        frame_key=frame_key,
        scene_version=scene_version,
        owner_token=object() if owner_token is None else owner_token,
        normalizer_key=normalizer_key,
        scene_key=scene_key,
    )


def smoothstep_grid_receiver_cover(
    query_coordinates: Tensor,
    anchor_coordinates: Tensor | None = None,
    environment_ids: Tensor | None = None,
    environment_mask: Tensor | None = None,
    query_ids: Tensor | None = None,
    *,
    frame_key,
    query_mask: Tensor | None = None,
    legal_bounds: Tensor | None = None,
    topology: PreparedGridTopology | None = None,
    scene_version=None,
    owner_token=None,
    normalizer_key=None,
    scene_key=None,
) -> ReceiverCover:
    """Build a C1 cover, reusing scene-owned topology when supplied.

    ``anchor_coordinates`` is ``[B,E,D]`` in the compatibility path and is
    mapped once per call. The reusable path accepts a ``PreparedGridTopology``
    made from that scene and performs only query-side search/weight work. Axis
    identities are cached as integer offsets; each call gathers live anchor
    coordinates so the cover retains gradients to selected anchor locations.
    """
    if query_coordinates.ndim != 3:
        raise ValueError("query_coordinates must have shape [B,Q,D].")
    batch, query_count, coordinate_dim = query_coordinates.shape
    if coordinate_dim not in (2, 3):
        raise ValueError("A receiver grid must use 2-D or 3-D coordinates.")
    if query_ids is None or query_ids.shape != (batch, query_count):
        raise ValueError("query_ids must have shape [B,Q].")
    if query_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Query IDs must use an integer tensor dtype.")
    if query_mask is None:
        query_mask = torch.ones((batch, query_count), dtype=torch.bool, device=query_coordinates.device)
    elif query_mask.shape != (batch, query_count) or query_mask.dtype != torch.bool:
        raise ValueError("query_mask must be boolean [B,Q].")
    topology_was_supplied = topology is not None
    if topology is None:
        if anchor_coordinates is None or environment_ids is None or environment_mask is None:
            raise ValueError("The compatibility path requires scene coordinates, IDs and mask.")
        topology = prepare_grid_topology(
            anchor_coordinates,
            environment_ids,
            environment_mask,
            frame_key=frame_key,
            scene_version=scene_version,
            owner_token=owner_token,
            normalizer_key=normalizer_key,
            scene_key=scene_key,
            legal_bounds=legal_bounds,
        )
    else:
        for name, supplied, stored in (
            ("anchor coordinates", anchor_coordinates, topology.anchor_coordinates),
            ("environment IDs", environment_ids, topology.environment_ids),
            ("environment mask", environment_mask, topology.environment_mask),
        ):
            if supplied is not None and supplied is not stored:
                raise ValueError(f"Supplied {name} do not belong to the prepared topology.")
        if legal_bounds is not None:
            if topology.legal_bounds is None or (
                legal_bounds is not topology.legal_bounds and not torch.equal(legal_bounds, topology.legal_bounds)
            ):
                raise ValueError("Supplied legal bounds differ from the prepared topology.")
    topology.validate_binding(
        frame_key=frame_key,
        scene_version=scene_version,
        owner_token=owner_token,
        normalizer_key=normalizer_key,
        scene_key=scene_key,
        anchor_coordinates=anchor_coordinates,
        environment_ids=environment_ids,
        environment_mask=environment_mask,
    )
    anchor_coordinates = topology.anchor_coordinates
    environment_ids = topology.environment_ids
    environment_mask = topology.environment_mask
    if topology.grid_anchor_indices.shape[0] != batch or len(topology.axis_source_indices) != coordinate_dim:
        raise ValueError("Prepared topology batch and coordinate dimensions must match the query.")
    if query_coordinates.device != anchor_coordinates.device or query_ids.device != query_coordinates.device:
        raise ValueError("Receiver coordinates, IDs and prepared topology must share a device.")
    if query_mask.device != query_coordinates.device:
        raise ValueError("Receiver query mask must share the coordinate device.")
    if not torch.isfinite(query_coordinates).all():
        raise ValueError("Receiver coordinates must be finite, including padded rows.")

    axes = [
        torch.gather(anchor_coordinates[..., axis_index], 1, source_indices).to(dtype=query_coordinates.dtype)
        for axis_index, source_indices in enumerate(topology.axis_source_indices)
    ]
    legal_bounds = topology.legal_bounds
    if legal_bounds is None:
        legal_bounds = torch.stack(
            (
                torch.stack([axis[:, 0] for axis in axes], dim=-1),
                torch.stack([axis[:, -1] for axis in axes], dim=-1),
            ),
            dim=1,
        )
    outside_live = ((query_coordinates < legal_bounds[:, None, 0]) |
                    (query_coordinates > legal_bounds[:, None, 1])) & query_mask[:, :, None]
    if outside_live.any():
        raise ValueError("Receiver queries must lie within the adapter's legal domain.")
    # Padded rows carry zero mass and do not need valid physical coordinates.
    # Give them a legal dummy location so searchsorted/fraction arithmetic stays
    # finite before those rows are masked from the returned cover.
    effective_query_coordinates = torch.where(
        query_mask[:, :, None], query_coordinates, legal_bounds[:, None, 0]
    )

    lower_indices: list[Tensor] = []
    upper_indices: list[Tensor] = []
    lower_weights: list[Tensor] = []
    upper_weights: list[Tensor] = []
    for axis_index, axis in enumerate(axes):
        size = axis.shape[1]
        if size == 1:
            lower = torch.zeros((batch, query_count), dtype=torch.long, device=axis.device)
            upper = lower
            fraction = query_coordinates.new_zeros((batch, query_count))
        else:
            upper_bound = torch.searchsorted(
                axis.contiguous(),
                effective_query_coordinates[..., axis_index].contiguous(),
                right=True,
            )
            lower = (upper_bound - 1).clamp_(0, size - 2)
            upper = lower + 1
            left_value = torch.gather(axis, 1, lower)
            right_value = torch.gather(axis, 1, upper)
            fraction = ((effective_query_coordinates[..., axis_index] - left_value) / (right_value - left_value)).clamp(
                0.0, 1.0
            )
        smooth = 3.0 * fraction.square() - 2.0 * fraction.pow(3)
        lower_indices.append(lower)
        upper_indices.append(upper)
        lower_weights.append(1.0 - smooth)
        upper_weights.append(smooth)

    corner_indices: list[list[Tensor]] = []
    corner_weights: list[Tensor] = []
    for corner in range(1 << coordinate_dim):
        indices: list[Tensor] = []
        weight = torch.ones_like(effective_query_coordinates[..., 0])
        for axis_index in range(coordinate_dim):
            choose_upper = (corner >> axis_index) & 1
            if choose_upper:
                indices.append(upper_indices[axis_index])
                weight = weight * upper_weights[axis_index]
            else:
                indices.append(lower_indices[axis_index])
                weight = weight * lower_weights[axis_index]
        corner_indices.append(indices)
        corner_weights.append(weight)

    flattened_grid = topology.grid_anchor_indices.reshape(batch, -1)
    mapped_positions: list[Tensor] = []
    for indices in corner_indices:
        linear = indices[0]
        for axis_index in range(1, coordinate_dim):
            linear = linear * topology.axis_source_indices[axis_index].shape[1] + indices[axis_index]
        mapped_positions.append(torch.gather(flattened_grid, 1, linear))
    anchor_indices = torch.stack(mapped_positions, dim=-1).to(torch.long)
    anchor_ids = torch.gather(
        environment_ids[:, None, :].expand(batch, query_count, environment_ids.shape[1]),
        2,
        anchor_indices,
    )
    mapped_valid = torch.gather(
        environment_mask[:, None, :].expand(batch, query_count, environment_mask.shape[1]),
        2,
        anchor_indices,
    )
    live_invalid = query_mask[:, :, None] & ~mapped_valid
    if live_invalid.any():
        raise ValueError("Every live receiver must map to present environment anchors.")
    validity = query_mask[:, :, None].expand_as(anchor_indices)
    weights = torch.stack(corner_weights, dim=-1)
    weights = weights * validity.to(weights.dtype)
    retain_topology = topology_was_supplied or any(
        value is not None for value in (scene_version, owner_token, normalizer_key, scene_key)
    )
    cover = ReceiverCover(
        anchor_indices=anchor_indices,
        anchor_ids=anchor_ids,
        weights=weights,
        validity=validity,
        query_ids=query_ids,
        frame_key=frame_key,
        query_mask=query_mask,
        topology=topology if retain_topology else None,
    )
    cover.validate_partition()
    if retain_topology:
        # A scene-bound cover validates its full native ID/bounds relation once,
        # then records a private version-bound proof for repeated decode calls.
        validate_cover_anchor_identity(cover, environment_ids)
        cover._seal_topology_validation(
            environment_ids,
            atol=2.0e-6,
            rtol=2.0e-6,
        )
    return cover


def validate_cover_anchor_identity(cover: ReceiverCover, anchor_ids: Tensor) -> None:
    """Check cover execution offsets against the prepared native anchor IDs."""
    cover.validate_unchanged()
    if cover._has_validation_seal(anchor_ids, unchanged_checked=True):
        return
    if anchor_ids.ndim != 2 or anchor_ids.shape[0] != cover.batch_size:
        raise ValueError("Prepared anchor IDs must have shape [B,E].")
    if cover.anchor_indices.numel() and (
        (cover.anchor_indices < 0).any() or (cover.anchor_indices >= anchor_ids.shape[1]).any()
    ):
        raise ValueError("Receiver cover contains an invalid anchor offset.")
    expected_ids = torch.gather(
        anchor_ids[:, None, :].expand(cover.batch_size, cover.query_count, anchor_ids.shape[1]),
        2,
        cover.anchor_indices.to(torch.long),
    )
    if not torch.equal(expected_ids[cover.validity], cover.anchor_ids[cover.validity]):
        raise ValueError("Receiver cover anchor IDs do not match prepared environment order.")
