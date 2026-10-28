"""Generic irregular-mesh coverage through the anchored core carrier."""
from __future__ import annotations

from dataclasses import replace

import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.interaction_reused_member_routing import AnchoredMemberRoutingCore


_VERTICES = {
    "A": (0.0, 0.0),
    "B": (2.0, 0.0),
    "C": (0.0, 1.0),
    "D": (1.0, 2.0),
}
_ORIGINAL_ID = {"A": 71, "B": 13, "C": 99, "D": 29}
_NODE_FEATURE = {
    "A": (0.7, -0.2),
    "B": (-0.3, 0.9),
    "C": (0.4, 0.1),
    "D": (-0.8, -0.5),
}
_NODE_MEASURE = {"A": 0.5, "B": 1.5, "C": 2.5, "D": 4.0}
_TRIANGLES = (("A", "B", "C"), ("B", "C", "D"))


def _base_config() -> dict:
    return ContentHypergraphCoreV2(
        module_input_dim=3,
        environment_input_dim=2,
        coordinate_dim=2,
        query_input_dim=0,
        operating_input_dim=4,
        output_dim=4,
        architecture="A2-D",
        context_dim=8,
        num_heads=2,
        context_depth=2,
        local_hidden_dim=8,
        dropout=0.0,
        init_seed=913,
    ).get_config()


def _scene(order: tuple[str, ...]) -> SceneBatch:
    dtype = torch.float64
    e_coordinates = torch.tensor(
        [[_VERTICES[name] if name != "pad" else (0.0, 0.0) for name in order]], dtype=dtype,
    )
    e_features = torch.tensor(
        [[_NODE_FEATURE[name] if name != "pad" else (0.0, 0.0) for name in order]], dtype=dtype,
    )
    e_ids = torch.tensor([[_ORIGINAL_ID[name] if name != "pad" else -1 for name in order]])
    e_measure = torch.tensor([[_NODE_MEASURE[name] if name != "pad" else 0.0 for name in order]], dtype=dtype)
    e_mask = torch.tensor([[name != "pad" for name in order]], dtype=torch.bool)
    # The M axis is present as a padded carrier, but the scene has no M records.
    return SceneBatch(
        module_features=torch.zeros((1, 2, 3), dtype=dtype),
        environment_features=e_features,
        module_coordinates=torch.zeros((1, 2, 2), dtype=dtype),
        environment_coordinates=e_coordinates,
        module_mask=torch.zeros((1, 2), dtype=torch.bool),
        environment_mask=e_mask,
        module_measure=torch.zeros((1, 2), dtype=dtype),
        environment_measure=e_measure,
        module_ids=torch.full((1, 2), -1, dtype=torch.long),
        environment_ids=e_ids,
        local_radius=torch.ones((1, 2), dtype=dtype),
        operating_features=torch.tensor([[0.2, -0.1, 0.4, 0.8]], dtype=dtype),
        coordinate_origin=torch.tensor([[0.0, 0.0]], dtype=dtype),
        coordinate_scale=torch.tensor([[2.0, 2.0]], dtype=dtype),
        module_measure_reference=torch.ones(1, dtype=dtype),
        environment_measure_reference=torch.ones(1, dtype=dtype),
        scene_key="r13-8-irregular-shared-edge",
        normalizer_key="r13-8-irregular-normalizer",
        coordinate_frame_key="r13-8-irregular-frame",
        module_characteristic_lengths=torch.ones((1, 2, 2), dtype=dtype),
        environment_characteristic_lengths=torch.ones((1, len(order), 2), dtype=dtype),
    )


def _slot_by_original_id(scene: SceneBatch) -> dict[int, int]:
    return {
        int(scene.environment_ids[0, slot]): slot
        for slot in range(scene.num_environment_tokens)
        if bool(scene.environment_mask[0, slot])
    }


def _barycentric(point: torch.Tensor, triangle: tuple[str, str, str], scene: SceneBatch,
                 slots: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    vertex_slots = torch.tensor([slots[_ORIGINAL_ID[name]] for name in triangle], dtype=torch.long)
    vertices = scene.environment_coordinates[0, vertex_slots]
    matrix = torch.stack((vertices[0] - vertices[2], vertices[1] - vertices[2]), dim=-1)
    first_two = torch.linalg.solve(matrix, point - vertices[2])
    weights = torch.cat((first_two, (1.0 - first_two.sum()).reshape(1)))
    return vertex_slots, weights


def _attach_barycentric_coverage(query: QueryBatch, scene: SceneBatch, *, triangle_override: int | None = None) -> QueryBatch:
    """Test adapter: locate a containing triangle, then attach its live barycentric cover."""
    slots = _slot_by_original_id(scene)
    rows, weight_rows = [], []
    for point in query.coordinates[0]:
        candidates = []
        triangle_indices = range(len(_TRIANGLES)) if triangle_override is None else (triangle_override,)
        for triangle_index in triangle_indices:
            triangle = _TRIANGLES[triangle_index]
            vertex_slots, raw_weights = _barycentric(point, triangle, scene, slots)
            if bool((raw_weights.detach() >= -1.0e-10).all()):
                weights = raw_weights.clamp_min(0)
                candidates.append((vertex_slots, weights / weights.sum()))
        if not candidates:
            raise ValueError("Query lies outside the irregular triangular coverage.")
        vertex_slots, weights = candidates[0]
        indices = torch.where(weights > 0, vertex_slots, torch.full_like(vertex_slots, -1))
        weights = torch.where(weights > 0, weights, torch.zeros_like(weights))
        rows.append(indices)
        weight_rows.append(weights)
    return replace(
        query,
        anchor_indices=torch.stack(rows)[None],
        anchor_weights=torch.stack(weight_rows)[None],
    )


def _query(coordinates: torch.Tensor) -> QueryBatch:
    return QueryBatch(
        coordinates=coordinates,
        query_mask=torch.ones(coordinates.shape[:2], dtype=torch.bool, device=coordinates.device),
        scene_key="r13-8-irregular-shared-edge",
        coordinate_frame_key="r13-8-irregular-frame",
    )


def _dense_slot_weights(query: QueryBatch, scene: SceneBatch) -> torch.Tensor:
    result = query.anchor_weights.new_zeros((query.batch_size, query.num_queries, scene.num_environment_tokens))
    active = query.anchor_weights > 0
    b, q, k = active.nonzero(as_tuple=True)
    return result.index_put((b, q, query.anchor_indices[b, q, k]), query.anchor_weights[b, q, k])


def test_irregular_barycentric_cover_is_legal_permutation_invariant_and_c0_with_empty_m() -> None:
    # Unequal triangles share edge BC. The E tensor is intentionally shuffled,
    # IDs are unrelated to slots, and a padded E slot carries no measure.
    areas = []
    for triangle in _TRIANGLES:
        a, b, c = (torch.tensor(_VERTICES[name], dtype=torch.float64) for name in triangle)
        areas.append(float(torch.linalg.det(torch.stack((b - a, c - a), dim=-1)).abs() / 2))
    assert areas == [1.0, 1.5]
    scene = _scene(("D", "B", "pad", "A", "C"))
    permuted_scene = _scene(("A", "pad", "C", "D", "B"))
    midpoint = torch.tensor([1.0, 0.5], dtype=torch.float64)
    epsilons = (1.0e-3, 1.0e-4)
    coordinates = torch.tensor(
        [[
            [1.0 - epsilons[0], 0.5 - 0.5 * epsilons[0]],
            [1.0 - epsilons[1], 0.5 - 0.5 * epsilons[1]],
            midpoint.tolist(),
            [1.0, 0.5 + 1.5 * epsilons[1]],
            [1.0, 0.5 + 1.5 * epsilons[0]],
        ]],
        dtype=torch.float64,
    ).requires_grad_(True)
    raw_query = _query(coordinates)
    query = _attach_barycentric_coverage(raw_query, scene)
    permuted_query = _attach_barycentric_coverage(raw_query, permuted_scene)

    assert query.anchor_indices.shape == (1, 5, 3)
    torch.testing.assert_close(query.anchor_weights.sum(-1), torch.ones((1, 5), dtype=torch.float64))
    assert (query.anchor_weights >= 0).all()
    assert not (query.anchor_indices[query.anchor_weights > 0] == 2).any()  # padded slot is never selected
    assert bool(scene.environment_mask[0, 2]) is False
    assert scene.module_mask.sum().item() == 0
    assert len(set(scene.environment_measure[0][scene.environment_mask[0]].tolist())) == 4

    # The same shared-edge point has the same physical-node measure from either
    # adjacent cell, despite different triangle vertex order and E slot order.
    edge_query = _query(midpoint.reshape(1, 1, 2))
    edge_left = _attach_barycentric_coverage(edge_query, scene, triangle_override=0)
    edge_right = _attach_barycentric_coverage(edge_query, scene, triangle_override=1)
    torch.testing.assert_close(_dense_slot_weights(edge_left, scene).sum(-1), torch.ones((1, 1), dtype=torch.float64))
    edge_by_id_left = {
        int(scene.environment_ids[0, slot]): edge_left.anchor_weights[0, 0, index]
        for index, slot in enumerate(edge_left.anchor_indices[0, 0].tolist()) if slot >= 0
    }
    edge_by_id_right = {
        int(scene.environment_ids[0, slot]): edge_right.anchor_weights[0, 0, index]
        for index, slot in enumerate(edge_right.anchor_indices[0, 0].tolist()) if slot >= 0
    }
    assert edge_by_id_left.keys() == edge_by_id_right.keys() == {_ORIGINAL_ID["B"], _ORIGINAL_ID["C"]}
    for original_id in edge_by_id_left:
        torch.testing.assert_close(edge_by_id_left[original_id], edge_by_id_right[original_id])

    model = AnchoredMemberRoutingCore(_base_config(), locator_mode="local_e_cover", query_tile_size=8).double().eval()
    prepared = model.prepare(scene)
    output = model.decode(prepared, query, return_trace=True)
    permuted_output = model.decode(model.prepare(permuted_scene), permuted_query)
    torch.testing.assert_close(output.joint_fields, permuted_output.joint_fields, rtol=1e-10, atol=1e-11)
    assert output.trace["context_M"].count_nonzero().item() == 0
    assert output.trace["candidate_mask_M"].count_nonzero().item() == 0
    assert output.trace["anchor_read_strength_M"].count_nonzero().item() == 0
    assert not output.trace["anchor_read_is_module"].any()

    exact_left = model.decode(prepared, edge_left).joint_fields
    exact_right = model.decode(prepared, edge_right).joint_fields
    torch.testing.assert_close(exact_left, exact_right, rtol=1e-10, atol=1e-11)
    edge_value = output.joint_fields[:, 2]
    left_far, left_near = output.joint_fields[:, 0], output.joint_fields[:, 1]
    right_near, right_far = output.joint_fields[:, 3], output.joint_fields[:, 4]
    assert (left_near - edge_value).norm() < (left_far - edge_value).norm()
    assert (right_near - edge_value).norm() < (right_far - edge_value).norm()

    grad_weights, grad_coordinates = torch.autograd.grad(
        output.joint_fields.square().sum(), (query.anchor_weights, query.coordinates),
    )
    assert torch.isfinite(grad_weights).all() and grad_weights.abs().sum() > 0
    assert torch.isfinite(grad_coordinates).all() and grad_coordinates.abs().sum() > 0
