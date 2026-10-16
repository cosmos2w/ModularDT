from __future__ import annotations

import pytest
import torch

from honf_forward_core.inherited_attention import (
    PreparedProjectedAnchorValues,
    ReceiverCover,
    gather_cover_indexed,
    gather_projected_cover_indexed,
    prepare_grid_topology,
    smoothstep_grid_receiver_cover,
    validate_cover_anchor_identity,
)


def _scene_grid(*, requires_grad: bool = False, singleton_x: bool = False):
    x = torch.tensor([0.0] if singleton_x else [-1.0, 1.0], dtype=torch.float64)
    y = torch.tensor([0.0, 2.0, 3.0], dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    base = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    permutation = torch.tensor([2, 0, 5, 1, 4, 3] if not singleton_x else [2, 0, 1])
    coordinates = base[permutation]
    ids_by_base = torch.tensor([31, 13, 67, 29, 41, 53] if not singleton_x else [31, 13, 67])
    ids = ids_by_base[permutation]
    if not singleton_x:
        coordinates = torch.cat((coordinates, torch.tensor([[-9.0, -9.0]], dtype=coordinates.dtype)), dim=0)
        ids = torch.cat((ids, torch.tensor([997], dtype=torch.long)))
        mask = torch.tensor([True] * 6 + [False])
    else:
        mask = torch.ones(3, dtype=torch.bool)
    coordinates = coordinates.unsqueeze(0).clone().requires_grad_(requires_grad)
    return coordinates, ids.unsqueeze(0), mask.unsqueeze(0)


def _cover_from_weights(indices: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, query_mask=None):
    batch, queries, slots = indices.shape
    anchor_ids = torch.gather(
        ids[:, None, :].expand(batch, queries, ids.shape[1]), 2, indices.to(torch.long)
    )
    validity = torch.ones_like(indices, dtype=torch.bool)
    if query_mask is not None:
        validity = validity & query_mask[:, :, None]
    return ReceiverCover(
        anchor_indices=indices,
        anchor_ids=anchor_ids,
        weights=weights,
        validity=validity,
        query_ids=torch.arange(queries, device=indices.device)[None].expand(batch, -1),
        frame_key="synthetic-frame",
        query_mask=query_mask,
    )


def _legacy_smoothstep_cover_2d(query, coordinates, ids, mask, query_ids, query_mask, legal_bounds):
    """Independent small-grid oracle matching the pre-topology atlas implementation."""
    batch, query_count, _ = query.shape
    indices = torch.zeros((batch, query_count, 4), dtype=torch.long)
    anchor_ids = torch.zeros_like(indices)
    weights = query.new_zeros((batch, query_count, 4))
    validity = query_mask[:, :, None].expand(batch, query_count, 4).clone()
    for batch_index in range(batch):
        axes = [torch.unique(coordinates[batch_index, mask[batch_index], dim], sorted=True) for dim in range(2)]
        lattice = torch.empty((axes[0].numel(), axes[1].numel()), dtype=torch.long)
        for x_index, x_value in enumerate(axes[0]):
            for y_index, y_value in enumerate(axes[1]):
                matches = mask[batch_index] & (coordinates[batch_index] == torch.stack((x_value, y_value))).all(-1)
                lattice[x_index, y_index] = torch.nonzero(matches, as_tuple=False).item()
        for query_index in range(query_count):
            point = query[batch_index, query_index]
            if not query_mask[batch_index, query_index]:
                point = legal_bounds[batch_index, 0]
            lower = []
            upper = []
            low_weight = []
            high_weight = []
            for axis_index, axis in enumerate(axes):
                if axis.numel() == 1:
                    lo = hi = 0
                    fraction = point.new_tensor(0.0)
                else:
                    hi_search = int(torch.searchsorted(axis, point[axis_index], right=True))
                    lo = max(0, min(hi_search - 1, axis.numel() - 2))
                    hi = lo + 1
                    fraction = ((point[axis_index] - axis[lo]) / (axis[hi] - axis[lo])).clamp(0.0, 1.0)
                smooth = 3.0 * fraction.square() - 2.0 * fraction.pow(3)
                lower.append(lo)
                upper.append(hi)
                low_weight.append(1.0 - smooth)
                high_weight.append(smooth)
            for corner in range(4):
                x_upper, y_upper = (corner & 1) != 0, (corner & 2) != 0
                x_index = upper[0] if x_upper else lower[0]
                y_index = upper[1] if y_upper else lower[1]
                indices[batch_index, query_index, corner] = lattice[x_index, y_index]
                weights[batch_index, query_index, corner] = (
                    (high_weight[0] if x_upper else low_weight[0])
                    * (high_weight[1] if y_upper else low_weight[1])
                ) * query_mask[batch_index, query_index].to(query.dtype)
        anchor_ids[batch_index] = ids[batch_index][indices[batch_index]]
    return indices, anchor_ids, weights, validity


def test_prepared_topology_matches_compatibility_path_and_keeps_native_permutation():
    coordinates, ids, mask = _scene_grid()
    legal_bounds = torch.tensor([[[-1.0, 0.0], [1.0, 3.0]]], dtype=coordinates.dtype)
    query = torch.tensor([[[-0.25, 1.0], [1.0, 3.0], [-1.0, 0.0], [0.2, 1.0]]], dtype=coordinates.dtype)
    query_ids = torch.tensor([[9, 4, 11, 0]])
    query_mask = torch.tensor([[True, True, True, False]])
    topology = prepare_grid_topology(
        coordinates, ids, mask, frame_key="synthetic-frame", scene_version=((1, 2, 3),),
        owner_token=object(), normalizer_key="normalizer", scene_key="scene-a", legal_bounds=legal_bounds,
    )
    expected = smoothstep_grid_receiver_cover(
        query, coordinates, ids, mask, query_ids, frame_key="synthetic-frame",
        query_mask=query_mask, legal_bounds=legal_bounds,
    )
    actual = smoothstep_grid_receiver_cover(
        query, query_ids=query_ids, frame_key="synthetic-frame", query_mask=query_mask,
        topology=topology, scene_version=((1, 2, 3),), owner_token=topology.owner_token,
        normalizer_key="normalizer", scene_key="scene-a",
    )
    assert actual.topology is topology
    assert topology.axis_source_indices[0].dtype == torch.long
    assert topology.axis_source_indices[1].dtype == torch.long
    torch.testing.assert_close(actual.anchor_indices, expected.anchor_indices)
    torch.testing.assert_close(actual.anchor_ids, expected.anchor_ids)
    torch.testing.assert_close(actual.validity, expected.validity)
    torch.testing.assert_close(actual.weights, expected.weights)
    assert torch.equal(topology.grid_anchor_indices, torch.tensor([[[1, 3, 0], [5, 4, 2]]]))
    assert not actual.weights[0, 3].any()


def test_batched_topology_reuses_independent_permuted_grids_and_mixed_masks():
    axes = (
        (torch.tensor([-2.0, 2.0]), torch.tensor([0.0, 1.0, 3.0])),
        (torch.tensor([10.0, 12.0]), torch.tensor([-3.0, -1.0, 4.0])),
    )
    coordinates_by_batch = []
    ids_by_batch = []
    for batch_index, ((x, y), permutation) in enumerate(
        zip(axes, (torch.tensor([2, 0, 5, 1, 4, 3]), torch.tensor([5, 3, 1, 4, 0, 2])))
    ):
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        ids = torch.arange(101 + batch_index * 100, 107 + batch_index * 100)
        coordinates_by_batch.append(torch.cat((grid[permutation], torch.tensor([[88.0, 99.0]])), dim=0))
        ids_by_batch.append(torch.cat((ids[permutation], torch.tensor([900 + batch_index]))))
    coordinates = torch.stack(coordinates_by_batch).to(torch.float64)
    ids = torch.stack(ids_by_batch)
    mask = torch.tensor([[True] * 6 + [False], [True] * 6 + [False]])
    query = torch.tensor(
        [[[-2.2, 0.1], [100.0, 100.0], [1.2, 2.4]],
         [[100.0, 100.0], [10.7, -1.4], [12.2, 4.2]]],
        dtype=torch.float64,
    )
    query_ids = torch.tensor([[9, 4, 11], [17, 15, 13]])
    query_mask = torch.tensor([[True, False, True], [False, True, True]])
    legal_bounds = torch.tensor(
        [[[-2.5, -0.4], [2.5, 3.4]], [[9.5, -3.4], [12.5, 4.4]]],
        dtype=torch.float64,
    )
    topology = prepare_grid_topology(
        coordinates,
        ids,
        mask,
        frame_key="batched-frame",
        scene_version=("batch-a", "batch-b"),
        owner_token=object(),
        legal_bounds=legal_bounds,
    )
    expected = smoothstep_grid_receiver_cover(
        query,
        coordinates,
        ids,
        mask,
        query_ids,
        frame_key="batched-frame",
        query_mask=query_mask,
        legal_bounds=legal_bounds,
    )
    legacy = _legacy_smoothstep_cover_2d(
        query, coordinates, ids, mask, query_ids, query_mask, legal_bounds
    )
    actual = smoothstep_grid_receiver_cover(
        query,
        query_ids=query_ids,
        frame_key="batched-frame",
        query_mask=query_mask,
        topology=topology,
        scene_version=("batch-a", "batch-b"),
        owner_token=topology.owner_token,
    )
    actual.validate_partition()
    assert actual.topology is topology
    assert actual.anchor_indices.shape == (2, 3, 4)
    assert not actual.validity[~query_mask].any()
    assert not actual.weights[~query_mask].any()
    for field in ("anchor_indices", "anchor_ids", "validity", "weights"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field))
    for field, old_value in zip(("anchor_indices", "anchor_ids", "weights", "validity"), legacy):
        torch.testing.assert_close(getattr(actual, field), old_value)
    assert not torch.equal(topology.grid_anchor_indices[0], topology.grid_anchor_indices[1])
    for batch_index in range(2):
        selected_ids = ids[batch_index][actual.anchor_indices[batch_index]]
        torch.testing.assert_close(selected_ids[actual.validity[batch_index]],
                                   actual.anchor_ids[batch_index][actual.validity[batch_index]])


def test_cover_gathers_live_query_and_selected_anchor_gradients():
    coordinates, ids, mask = _scene_grid(requires_grad=True)
    topology = prepare_grid_topology(
        coordinates, ids, mask, frame_key="synthetic-frame", scene_version=("live",),
        owner_token=object(),
    )
    query = torch.tensor([[[0.15, 1.1]]], dtype=torch.float64, requires_grad=True)
    cover = smoothstep_grid_receiver_cover(
        query, query_ids=torch.tensor([[7]]), frame_key="synthetic-frame", topology=topology,
        scene_version=("live",), owner_token=topology.owner_token,
    )
    payload = torch.arange(coordinates.shape[1], dtype=torch.float64).reshape(1, -1, 1).requires_grad_()
    selected = payload[torch.arange(1)[:, None, None], cover.anchor_indices]
    objective = (selected[..., 0] * cover.weights).sum()
    query_grad, coordinate_grad = torch.autograd.grad(objective, (query, coordinates))
    assert torch.isfinite(query_grad).all() and query_grad.abs().sum() > 0
    assert torch.isfinite(coordinate_grad).all() and coordinate_grad.abs().sum() > 0
    assert not any(tensor.requires_grad for tensor in topology.axis_source_indices)


def test_live_query_and_anchor_geometry_weights_pass_finite_difference_checks():
    coordinates, ids, mask = _scene_grid()
    query = torch.tensor([[[0.17, 1.23]]], dtype=torch.float64, requires_grad=True)
    translation = torch.zeros((1, 1, 2), dtype=torch.float64, requires_grad=True)

    def weights(query_coordinates, scene_translation):
        anchor_coordinates = coordinates + scene_translation
        topology = prepare_grid_topology(
            anchor_coordinates, ids, mask, frame_key="synthetic-frame"
        )
        cover = smoothstep_grid_receiver_cover(
            query_coordinates,
            query_ids=torch.tensor([[2]]),
            frame_key="synthetic-frame",
            topology=topology,
        )
        return cover.weights

    assert torch.autograd.gradcheck(weights, (query, translation), eps=1e-6, atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(weights, (query, translation), eps=1e-6, atol=3e-5, rtol=3e-4)


def test_topology_rejects_scene_binding_and_source_mutations():
    coordinates, ids, mask = _scene_grid(requires_grad=True)
    topology = prepare_grid_topology(
        coordinates, ids, mask, frame_key="synthetic-frame", scene_version=("v1",),
        owner_token=object(), normalizer_key="norm-a", scene_key="scene-a",
    )
    query = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    cover = smoothstep_grid_receiver_cover(
        query, query_ids=torch.tensor([[1]]), frame_key="synthetic-frame", topology=topology,
        scene_version=("v1",), owner_token=topology.owner_token,
    )
    with pytest.raises(ValueError, match="scene version"):
        cover.validate_topology(scene_version=("v2",))
    with pytest.raises(ValueError, match="normalizer"):
        topology.validate_binding(normalizer_key="norm-b")
    with torch.no_grad():
        coordinates[0, 0, 0].add_(0.01)
    with pytest.raises(RuntimeError, match="topology changed"):
        cover.validate_unchanged()


def _make_sealed_cover():
    coordinates, ids, mask = _scene_grid()
    legal_bounds = torch.tensor([[[-1.0, 0.0], [1.0, 3.0]]], dtype=coordinates.dtype)
    topology = prepare_grid_topology(
        coordinates,
        ids,
        mask,
        frame_key="sealed-frame",
        scene_version=("sealed-scene",),
        owner_token=object(),
        legal_bounds=legal_bounds,
    )
    query = torch.tensor([[[0.2, 1.1], [-0.3, 2.2]]], dtype=coordinates.dtype)
    query_ids = torch.tensor([[101, 103]])
    cover = smoothstep_grid_receiver_cover(
        query,
        query_ids=query_ids,
        frame_key="sealed-frame",
        query_mask=torch.ones_like(query_ids, dtype=torch.bool),
        topology=topology,
        scene_version=("sealed-scene",),
        owner_token=topology.owner_token,
    )
    assert cover._has_validation_seal(ids)
    return coordinates, ids, mask, topology, cover


@pytest.mark.parametrize(
    "field, mutate",
    [
        ("weights", lambda cover: cover.weights[0, 0, 0].add_(0.01)),
        ("anchor_indices", lambda cover: cover.anchor_indices[0, 0, 0].add_(1)),
        ("validity", lambda cover: cover.validity[0, 0, 0].logical_not_()),
        ("query_mask", lambda cover: cover.query_mask[0, 0].logical_not_()),
        ("anchor_ids", lambda cover: cover.anchor_ids[0, 0, 0].add_(1)),
    ],
)
def test_validation_seal_rejects_cover_tensor_mutations(field, mutate):
    _coordinates, ids, _mask, _topology, cover = _make_sealed_cover()
    with torch.no_grad():
        mutate(cover)
    with pytest.raises(RuntimeError, match="Receiver-cover tensors changed"):
        validate_cover_anchor_identity(cover, ids)


@pytest.mark.parametrize("field", ["grid_anchor_indices", "legal_bounds", "environment_ids"])
def test_validation_seal_rejects_topology_tensor_mutations(field):
    _coordinates, _ids, _mask, topology, cover = _make_sealed_cover()
    tensor = getattr(topology, field)
    with torch.no_grad():
        tensor.reshape(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match="Prepared grid topology changed"):
        cover.validate_unchanged()


def test_validation_seal_is_not_inherited_by_replaced_cover_or_owner():
    from dataclasses import replace

    _coordinates, ids, _mask, topology, cover = _make_sealed_cover()
    changed_ids = cover.anchor_ids.clone()
    changed_ids[0, 0, 0] += 1
    replaced_cover = replace(cover, anchor_ids=changed_ids)
    assert not replaced_cover._has_validation_seal(ids)
    with pytest.raises(ValueError, match="do not match prepared environment order"):
        validate_cover_anchor_identity(replaced_cover, ids)

    replacement_topology = replace(topology, owner_token=object())
    owner_rebound_cover = replace(
        cover, topology=replacement_topology, owner_token=replacement_topology.owner_token
    )
    with pytest.raises(ValueError, match="different scene cache owner"):
        owner_rebound_cover.validate_topology(owner_token=topology.owner_token)


def test_validation_seal_keeps_actual_payload_anchor_bounds_checks():
    _coordinates, ids, _mask, _topology, cover = _make_sealed_cover()
    projected_values = torch.randn(1, 3, 2, dtype=cover.weights.dtype)
    with pytest.raises(ValueError, match="invalid anchor offset"):
        gather_projected_cover_indexed(projected_values, cover, anchor_ids=ids)
    with pytest.raises(ValueError, match="invalid anchor offset"):
        gather_projected_cover_indexed(projected_values, cover, validate_identity=False)

    typed_values = torch.randn(1, 2, 3, 4, dtype=cover.weights.dtype)
    with pytest.raises(ValueError, match="invalid anchor offset"):
        gather_cover_indexed(typed_values, cover, anchor_ids=ids)
    with pytest.raises(ValueError, match="invalid anchor offset"):
        gather_cover_indexed(typed_values, cover, validate_identity=False)


def test_topology_cover_semantic_checks_are_sealed_once_for_repeated_decodes(monkeypatch):
    coordinates, ids, mask = _scene_grid()
    topology = prepare_grid_topology(
        coordinates,
        ids,
        mask,
        frame_key="count-frame",
        scene_version=("count-scene",),
        owner_token=object(),
    )
    payload_values = torch.randn(1, coordinates.shape[1], 2, dtype=coordinates.dtype)
    payload = PreparedProjectedAnchorValues(
        payload_values, ids, "count-frame", ("count-scene",), topology.owner_token
    )
    original_allclose = torch.allclose
    original_equal = torch.equal
    calls = {"partition": 0, "identity": 0}

    def count_allclose(*args, **kwargs):
        calls["partition"] += 1
        return original_allclose(*args, **kwargs)

    def count_equal(*args, **kwargs):
        calls["identity"] += 1
        return original_equal(*args, **kwargs)

    monkeypatch.setattr(torch, "allclose", count_allclose)
    monkeypatch.setattr(torch, "equal", count_equal)
    cover = smoothstep_grid_receiver_cover(
        torch.tensor([[[0.2, 1.1], [-0.3, 2.2]]], dtype=coordinates.dtype),
        query_ids=torch.tensor([[5, 6]]),
        frame_key="count-frame",
        topology=topology,
        scene_version=("count-scene",),
        owner_token=topology.owner_token,
    )
    assert calls == {"partition": 1, "identity": 1}
    for _ in range(3):
        cover.validate_partition()
        validate_cover_anchor_identity(cover, ids)
        gather_projected_cover_indexed(payload, cover)
    assert calls == {"partition": 1, "identity": 1}
    # An equal replacement source has no identity-bound seal and uses the full oracle.
    validate_cover_anchor_identity(cover, ids.clone())
    assert calls == {"partition": 1, "identity": 2}


def test_singleton_axis_legal_edges_and_padded_queries_keep_partition_rules():
    coordinates, ids, mask = _scene_grid(singleton_x=True)
    topology = prepare_grid_topology(
        coordinates, ids, mask, frame_key="singleton-frame", scene_version=("s",), owner_token=object(),
    )
    query = torch.tensor([[[0.0, 0.0], [0.0, 3.0], [0.0, 2.0]]], dtype=torch.float64)
    query_mask = torch.tensor([[True, True, False]])
    cover = smoothstep_grid_receiver_cover(
        query, query_ids=torch.tensor([[4, 5, 6]]), frame_key="singleton-frame",
        query_mask=query_mask, topology=topology, scene_version=("s",), owner_token=topology.owner_token,
    )
    cover.validate_partition()
    assert cover.anchor_indices.shape == (1, 3, 4)
    assert torch.allclose(cover.weights[:, :2].sum(-1), torch.ones(1, 2, dtype=torch.float64))
    assert not cover.weights[:, 2].any()
    assert torch.equal(cover.anchor_ids[0, 0], torch.tensor([31, 31, 13, 13]))
    assert torch.equal(cover.anchor_ids[0, 1], torch.tensor([13, 13, 67, 67]))


def test_projected_indexed_gather_matches_dense_values_and_both_gradients():
    torch.manual_seed(20261016)
    values = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(
        [[[0.2, 0.3, 0.5, 0.0], [0.0, 0.4, 0.6, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    indices = torch.tensor([[[1, 1, 3, 0], [2, 0, 2, 1], [0, 1, 2, 3]]], dtype=torch.long)
    query_mask = torch.tensor([[True, True, False]])
    ids = torch.tensor([[8, 3, 13, 21]])
    cover = _cover_from_weights(indices, weights, ids, query_mask)
    payload = PreparedProjectedAnchorValues(
        values, ids, "synthetic-frame", ("scene-version",), object()
    )
    actual = gather_projected_cover_indexed(payload, cover)
    effective = weights * cover.validity.to(weights.dtype)
    dense = values.new_zeros(1, 3, 4).scatter_add(-1, indices, effective)
    expected = torch.einsum("bqe,bec->bqc", dense, values)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), (values, weights), retain_graph=True)
    expected_grads = torch.autograd.grad((expected * upstream).sum(), (values, weights))
    for left, right in zip(actual_grads, expected_grads):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    assert torch.equal(actual[0, 2], torch.zeros_like(actual[0, 2]))


def test_projected_indexed_gather_gradcheck_gradgradcheck_and_order_recovery():
    ids = torch.tensor([[12, 7, 19]])
    indices = torch.tensor([[[0, 1, 1, 2], [2, 0, 2, 1]]], dtype=torch.long)
    query_ids = torch.tensor([[9, 4]])
    validity = torch.ones_like(indices, dtype=torch.bool)
    owner = object()

    def read(values, weights):
        cover = ReceiverCover(indices, torch.gather(ids[:, None, :].expand(1, 2, 3), 2, indices),
                              weights, validity, query_ids, "synthetic-frame")
        payload = PreparedProjectedAnchorValues(values, ids, "synthetic-frame", ("v",), owner)
        return gather_projected_cover_indexed(
            payload, cover, expected_frame_key="synthetic-frame",
            expected_scene_version=("v",), expected_owner_token=owner,
        )

    values = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[[0.2, 0.3, 0.1, 0.4], [0.1, 0.4, 0.2, 0.3]]],
                           dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(read, (values, weights), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(read, (values, weights), eps=1e-6, atol=2e-5, rtol=2e-4)
    full = read(values, weights)
    reverse_cover = ReceiverCover(
        indices.flip(1),
        torch.gather(ids[:, None, :].expand(1, 2, 3), 2, indices.flip(1)),
        weights.flip(1),
        validity.flip(1),
        query_ids.flip(1),
        "synthetic-frame",
    )
    reverse = gather_projected_cover_indexed(
        PreparedProjectedAnchorValues(values, ids, "synthetic-frame", ("v",), owner),
        reverse_cover,
        expected_scene_version=("v",),
        expected_owner_token=owner,
    )
    torch.testing.assert_close(reverse, full.flip(1))
    with pytest.raises(ValueError, match="stale for the current scene version"):
        gather_projected_cover_indexed(
            PreparedProjectedAnchorValues(values, ids, "synthetic-frame", ("v",), owner),
            ReceiverCover(indices, torch.gather(ids[:, None, :].expand(1, 2, 3), 2, indices),
                          weights, validity, query_ids, "synthetic-frame"),
            expected_scene_version=("stale",), expected_owner_token=owner,
        )
