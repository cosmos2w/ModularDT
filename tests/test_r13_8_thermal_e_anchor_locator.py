from __future__ import annotations

from dataclasses import replace

import torch

from channelthermal.e_anchor_locator import attach_thermal_e_anchor_query
from honf_forward_core.content_hypergraph import QueryBatch, SceneBatch
from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1, slice_query_v1


def _scene(*, shuffled: bool = False) -> SceneBatch:
    nx, ny = 4, 3
    x = (torch.arange(nx, dtype=torch.float32) + 0.5)
    y = (torch.arange(ny, dtype=torch.float32) + 0.5)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    coordinates = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    ids = torch.arange(600, 600 + nx * ny)
    if shuffled:
        order = torch.tensor([8, 1, 10, 3, 6, 0, 11, 2, 9, 5, 7, 4])
        coordinates, ids = coordinates[order], ids[order]
    count = coordinates.shape[0]
    return SceneBatch(
        module_features=torch.zeros((1, 0, 1)), environment_features=torch.zeros((1, count, 1)),
        module_coordinates=torch.zeros((1, 0, 2)), environment_coordinates=coordinates[None],
        module_mask=torch.zeros((1, 0), dtype=torch.bool),
        environment_mask=torch.ones((1, count), dtype=torch.bool),
        module_measure=torch.zeros((1, 0)), environment_measure=torch.ones((1, count)),
        module_ids=torch.zeros((1, 0), dtype=torch.long), environment_ids=ids[None],
        local_radius=torch.empty((1, 0)), operating_features=torch.zeros((1, 0)),
        coordinate_origin=torch.zeros((1, 2)), coordinate_scale=torch.tensor([[4.0, 3.0]]),
        scene_key="thermal-e-locator-test", coordinate_frame_key="thermal-e-locator-frame",
    )


def _query(coordinates: torch.Tensor) -> QueryBatch:
    return QueryBatch(
        coordinates=coordinates.reshape(1, -1, 2),
        query_mask=torch.ones((1, coordinates.shape[0]), dtype=torch.bool),
        scene_key="thermal-e-locator-test", coordinate_frame_key="thermal-e-locator-frame",
    )


def _slot(scene: SceneBatch, ix: int, iy: int) -> int:
    target = torch.tensor([ix + 0.5, iy + 0.5])
    return int((scene.environment_coordinates[0] == target).all(-1).nonzero()[0])


def test_e_anchor_uses_half_open_physical_cells_and_actual_shuffled_slots() -> None:
    scene = _scene(shuffled=True)
    query = _query(torch.tensor([[0.0, 0.0], [1.0, 1.0], [4.0, 3.0]]))
    attached = attach_thermal_e_anchor_query(query, scene, "e_anchor", environment_nx=4, environment_ny=3)

    assert attached.anchor_indices.shape == (1, 3, 1)
    assert attached.anchor_indices[0, :, 0].tolist() == [
        _slot(scene, 0, 0), _slot(scene, 1, 1), _slot(scene, 3, 2),
    ]
    torch.testing.assert_close(attached.anchor_weights, torch.ones((1, 3, 1)))
    assert attached.anchor_indices[0, 2, 0] != 11  # the grid index is not assumed to be the slot


def test_local_e_cover_is_sparse_smooth_and_has_live_coordinate_derivatives() -> None:
    scene = _scene(shuffled=True)
    coordinates = torch.tensor([[1.0, 1.0], [1.5, 0.5], [0.01, 0.01]], requires_grad=True)
    attached = attach_thermal_e_anchor_query(
        _query(coordinates), scene, "local_e_cover", environment_nx=4, environment_ny=3,
    )
    active = attached.anchor_weights > 0
    assert active.sum(-1).tolist() == [[4, 1, 1]]
    torch.testing.assert_close(attached.anchor_weights[0, 0], torch.full((4,), 0.25))
    torch.testing.assert_close(attached.anchor_weights.sum(-1), torch.ones((1, 3)))
    assert attached.anchor_indices[0, 1].tolist().count(-1) == 3
    assert attached.anchor_indices[0, 2].tolist().count(-1) == 3

    live_x = attached.anchor_weights[0, 0]
    anchor_x = scene.environment_coordinates[0, attached.anchor_indices[0, 0], 0]
    centroid = (live_x * anchor_x).sum()
    grad = torch.autograd.grad(centroid, coordinates)[0]
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad[0], torch.tensor([1.5, 0.0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad[1], torch.zeros(2), atol=1e-6, rtol=1e-6)


def test_query_tiling_and_row_selection_preserve_live_anchor_carrier() -> None:
    scene = _scene()
    attached = attach_thermal_e_anchor_query(
        _query(torch.tensor([[0.75, 0.75], [2.25, 1.25], [3.99, 2.99]])),
        scene, "local_e_cover", environment_nx=4, environment_ny=3,
    )
    tile = slice_query_v1(attached, 1, 3)
    selected = select_query_rows_v1(attached, torch.tensor([2, 0]))
    torch.testing.assert_close(tile.anchor_indices, attached.anchor_indices[:, 1:3])
    torch.testing.assert_close(tile.anchor_weights, attached.anchor_weights[:, 1:3])
    torch.testing.assert_close(selected.anchor_indices, attached.anchor_indices[:, [2, 0]])
    torch.testing.assert_close(selected.anchor_weights, attached.anchor_weights[:, [2, 0]])

    moved_coordinates = tile.coordinates.clone()
    moved_coordinates[0, 0, 0] -= 0.2
    rebuilt = attach_thermal_e_anchor_query(
        replace(tile, coordinates=moved_coordinates, anchor_indices=None, anchor_weights=None),
        scene, "local_e_cover", environment_nx=4, environment_ny=3,
    )
    assert not torch.equal(rebuilt.anchor_weights[0, 0], tile.anchor_weights[0, 0])
