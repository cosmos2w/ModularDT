from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from honf_forward_core.content_hypergraph.contracts import QueryBatch
from honf_forward_core.content_hypergraph.staged_attention_v2 import StagedContextCycleV2
from honf_forward_core.inherited_attention import (
    InheritedQueryBatch,
    PreparedAnchorValues,
    ReceiverCover,
    RouteScoreSnapshot,
    TypedTokenValues,
    build_typed_atlas,
    gather_cover_dense_reference,
    gather_cover_indexed,
    gather_source_response_indexed,
    measure_relative_entmax15,
    measure_relative_entmax15_reference,
    measure_relative_normalize,
    measure_relative_softmax,
    normalize_donor_measure,
    prepare_anchor_values,
    prepare_source_indexed_anchor_response,
    smoothstep_grid_receiver_cover,
    with_receiver_cover,
)


def _snapshot(logits: torch.Tensor, measure: torch.Tensor, *, route: str = "em") -> RouteScoreSnapshot:
    batch, heads, anchors, donors = logits.shape
    head_width = 3
    donor_valid = measure > 0
    anchor_ids = torch.arange(100, 100 + anchors, dtype=torch.long, device=logits.device)[None].expand(batch, -1)
    donor_ids = torch.arange(10, 10 + donors, dtype=torch.long, device=logits.device)[None].expand(batch, -1)
    return RouteScoreSnapshot(
        route=route,
        query_scaled=logits.new_zeros(batch, heads, anchors, head_width),
        key_normalized=logits.new_zeros(batch, heads, donors, head_width),
        geometry_bias=logits,
        raw_logits=logits,
        donor_mask=donor_valid,
        destination_mask=torch.ones(batch, anchors, dtype=torch.bool, device=logits.device),
        donor_measure=measure,
        anchor_ids=anchor_ids,
        donor_ids=donor_ids,
        score_origin=f"last_{route}",
    )


def _cover(
    anchor_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    indices: torch.Tensor | None = None,
) -> ReceiverCover:
    batch, queries, slots = weights.shape
    anchors = anchor_ids.shape[1]
    if indices is None:
        indices = torch.arange(queries * slots, device=weights.device).reshape(1, queries, slots) % anchors
        indices = indices.expand(batch, -1, -1)
    resolved = torch.gather(
        anchor_ids[:, None, :].expand(batch, queries, anchors), 2, indices.to(torch.long)
    )
    return ReceiverCover(
        anchor_indices=indices.to(torch.long),
        anchor_ids=resolved,
        weights=weights,
        validity=torch.ones_like(weights, dtype=torch.bool),
        query_ids=torch.arange(queries, dtype=torch.long, device=weights.device)[None].expand(batch, -1),
        frame_key="fixture-frame",
    )


def test_measure_relative_softmax_matches_weighted_masked_reference_and_empty_types():
    logits = torch.tensor(
        [[[[0.1, 1.2, -0.4, 0.7], [0.2, -0.3, 0.4, 0.0]]], [[[0.4, -0.2, 0.3, 1.0], [0.8, -0.5, 0.2, -1.0]]]],
        dtype=torch.float64,
    )
    measure = torch.tensor([[0.2, 0.0, 0.5, 0.3], [0.1, 0.2, 0.4, 0.3]], dtype=torch.float64)
    donor_valid = torch.tensor([[True, True, True, True], [False, False, False, False]])
    destination_valid = torch.tensor([[True, True], [True, False]])
    actual = measure_relative_softmax(logits, measure, donor_valid, destination_valid)
    normalized = normalize_donor_measure(measure, donor_valid)
    effective_donor_valid = donor_valid & (measure > 0)
    allowed = effective_donor_valid[:, None, None, :] & destination_valid[:, None, :, None]
    safe_measure = torch.where(normalized > 0, normalized, torch.ones_like(normalized))
    ref_logits = (logits + torch.log(safe_measure)[:, None, None, :]).masked_fill(~allowed, -torch.inf)
    has_donor = allowed.any(dim=-1, keepdim=True)
    ref_logits = torch.where(has_donor, ref_logits, torch.zeros_like(ref_logits))
    expected = torch.softmax(ref_logits, dim=-1) * allowed
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))
    torch.testing.assert_close(actual[0, :, 0].sum(dim=-1), torch.ones_like(actual[0, :, 0, 0]))
    torch.testing.assert_close(actual[0, :, 1].sum(dim=-1), torch.ones_like(actual[0, :, 1, 0]))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        normalize_donor_measure(
            torch.tensor([[0.2, -0.1]], dtype=torch.float64),
            torch.tensor([[True, True]]),
        )


def test_measure_relative_entmax15_reference_shift_support_and_refinement():
    logits = torch.tensor([5.0, 0.0, -3.0, -6.0], dtype=torch.float64)
    measure = torch.tensor([0.1, 0.25, 0.4, 0.25], dtype=torch.float64)
    expected = measure_relative_entmax15_reference(logits, measure)
    batched = measure_relative_entmax15(
        logits.reshape(1, 1, 1, -1), measure.reshape(1, -1),
        torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool),
    )[0, 0, 0]
    torch.testing.assert_close(batched, expected, atol=2.0e-12, rtol=2.0e-12)
    assert torch.all(batched >= 0)
    torch.testing.assert_close(batched.sum(), torch.tensor(1.0, dtype=torch.float64), atol=2.0e-12, rtol=0)
    assert (batched == 0).any()
    shifted = measure_relative_entmax15(
        (logits + 13.0).reshape(1, 1, 1, -1), measure.reshape(1, -1),
        torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool),
    )[0, 0, 0]
    torch.testing.assert_close(batched, shifted, atol=2.0e-12, rtol=2.0e-12)

    split_logits = torch.tensor([5.0, 5.0, 0.0, -3.0, -6.0], dtype=torch.float64)
    split_measure = torch.tensor([0.04, 0.06, 0.25, 0.4, 0.25], dtype=torch.float64)
    split = measure_relative_entmax15_reference(split_logits, split_measure)
    torch.testing.assert_close(torch.stack((split[:2].sum(), *split[2:])), expected, atol=2.0e-12, rtol=2.0e-12)


def test_measure_relative_entmax15_composed_score_and_measure_gradients():
    logits = torch.tensor([[[[1.3, 0.1, -0.4, -1.2]]]], dtype=torch.float64, requires_grad=True)
    measure = torch.tensor([[0.13, 0.22, 0.41, 0.24]], dtype=torch.float64, requires_grad=True)
    donor_valid = torch.ones(1, 4, dtype=torch.bool)
    destination_valid = torch.ones(1, 1, dtype=torch.bool)

    def normalized(scores, masses):
        return measure_relative_entmax15(scores, masses, donor_valid, destination_valid)

    assert torch.autograd.gradcheck(normalized, (logits, measure), eps=1.0e-6, atol=2.0e-5, rtol=2.0e-4)
    p, normalized_measure = measure_relative_normalize(
        logits, measure, donor_valid, destination_valid, normalization="entmax15"
    )
    torch.testing.assert_close(normalized_measure.sum(dim=-1), torch.ones(1, dtype=torch.float64))
    assert torch.isfinite(p).all()


def test_entmax15_fp32_support_boundary_has_a_normalized_root_and_finite_gradients():
    # The second donor is at the support threshold. Rounded FP32 prefix
    # intervals used to leave a gap: neither of the two roots was accepted.
    first = torch.cat((torch.tensor([0.0014990998897701502]), torch.linspace(0.001, 0.999, 257)))
    measure = torch.stack((first, 1 - first), dim=-1).requires_grad_()
    scores = torch.stack((torch.zeros_like(first), -2 / first.sqrt()), dim=-1)
    scores = scores[:, None, None, :].requires_grad_()
    valid = torch.ones_like(measure, dtype=torch.bool)
    destinations = torch.ones(len(first), 1, dtype=torch.bool)
    actual = measure_relative_entmax15(scores, measure, valid, destinations)
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual[..., 0]), atol=2e-7, rtol=0)
    for index in (0, 1, 64, 128, 257):
        expected = measure_relative_entmax15_reference(scores[index, 0, 0], measure[index])
        torch.testing.assert_close(actual[index, 0, 0].double(), expected, atol=2e-7, rtol=0)
    gradients = torch.autograd.grad((actual * torch.tensor([0.3, -0.7])).sum(), (scores, measure))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_measure_relative_entmax15_wide_scores_empty_rows_and_present_measure_validation():
    scores = torch.tensor([[[[1000.0, -1000.0, -2500.0, -4000.0]]]], dtype=torch.float64, requires_grad=True)
    measure = torch.tensor([[0.01, 0.19, 0.30, 0.50]], dtype=torch.float64, requires_grad=True)
    valid = torch.ones(1, 4, dtype=torch.bool)
    destination = torch.ones(1, 1, dtype=torch.bool)
    actual = measure_relative_entmax15(scores, measure, valid, destination)
    expected = measure_relative_entmax15_reference(scores[0, 0, 0], measure[0])
    torch.testing.assert_close(actual[0, 0, 0], expected, atol=3.0e-12, rtol=3.0e-12)
    objective = (actual * torch.tensor([[[[1.0, -0.4, 0.7, 1.3]]]], dtype=torch.float64)).sum()
    score_grad, measure_grad = torch.autograd.grad(objective, (scores, measure))
    assert torch.isfinite(score_grad).all() and torch.isfinite(measure_grad).all()

    empty_scores = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    empty_measure = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
    empty_valid = torch.zeros(1, 4, dtype=torch.bool)
    empty_destination = torch.tensor([[True, False, True]])
    empty = measure_relative_entmax15(empty_scores, empty_measure, empty_valid, empty_destination)
    assert torch.equal(empty, torch.zeros_like(empty))
    empty_grads = torch.autograd.grad(empty.square().sum(), (empty_scores, empty_measure))
    assert all(torch.isfinite(gradient).all() for gradient in empty_grads)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        measure_relative_entmax15(
            torch.zeros(1, 1, 1, 2),
            torch.tensor([[0.5, float("nan")]]),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
        )


def test_soft_and_entmax_atlas_reassociation_matches_literal_dense_and_gradients():
    torch.manual_seed(19)
    logits = torch.tensor(
        [[[[1.1, 0.25, -0.35, -1.3], [0.45, 0.9, -0.2, -0.8], [-0.1, 0.55, 1.0, -0.5]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    measure = torch.tensor([[0.16, 0.28, 0.36, 0.20]], dtype=torch.float64, requires_grad=True)
    values = torch.randn(1, 1, 4, 5, dtype=torch.float64, requires_grad=True)
    cover_weights = torch.tensor([[[0.3, 0.7], [0.65, 0.35]]], dtype=torch.float64, requires_grad=True)
    snapshot = _snapshot(logits, measure)
    cover = _cover(snapshot.anchor_ids, cover_weights, indices=torch.tensor([[[0, 1], [1, 2]]]))
    for mode in ("softmax", "entmax15"):
        atlas = build_typed_atlas(snapshot, normalization=mode)
        token_values = TypedTokenValues(
            kind="module",
            values=values,
            token_ids=snapshot.donor_ids,
            valid=snapshot.donor_mask,
            measure=measure,
        )
        prepared = prepare_anchor_values(atlas, token_values)
        prepared_selected = prepare_anchor_values(atlas, token_values, executor="selected")
        expected_selected_counts = (atlas.probabilities > 0).sum(dim=-1)
        torch.testing.assert_close(prepared_selected.executed_edge_counts, expected_selected_counts)
        torch.testing.assert_close(
            prepared_selected.candidate_edge_counts,
            snapshot.donor_mask.sum(dim=-1)[:, None, None].expand_as(expected_selected_counts)
            * snapshot.destination_mask[:, None, :],
        )
        assert torch.all(
            prepared_selected.executed_edge_counts <= prepared_selected.candidate_edge_counts
        )
        dense_route_work = torch.full_like(expected_selected_counts, logits.shape[-1])
        torch.testing.assert_close(prepared.executed_edge_counts, dense_route_work)
        reassociated = gather_cover_indexed(prepared, cover)
        selected_reassociated = gather_cover_indexed(prepared_selected, cover)
        dense_cover = torch.nn.functional.one_hot(cover.anchor_indices, num_classes=3).to(torch.float64)
        dense_cover = torch.einsum("bqce,bqc->bqe", dense_cover, cover.weights)
        beta = torch.einsum("bqe,bhen->bqhn", dense_cover, atlas.probabilities)
        literal = torch.einsum("bqhn,bhnd->bqhd", beta, values)
        torch.testing.assert_close(reassociated, literal, atol=3.0e-12, rtol=3.0e-12)
        torch.testing.assert_close(selected_reassociated, literal, atol=3.0e-12, rtol=3.0e-12)
        leaves = (logits, measure, values, cover_weights)
        grad_reassociated = torch.autograd.grad(reassociated.square().sum(), leaves, retain_graph=True)
        grad_selected = torch.autograd.grad(
            selected_reassociated.square().sum(), leaves, retain_graph=True
        )
        grad_literal = torch.autograd.grad(literal.square().sum(), leaves, retain_graph=True)
        for actual, selected, expected in zip(grad_reassociated, grad_selected, grad_literal):
            torch.testing.assert_close(actual, expected, atol=5.0e-11, rtol=5.0e-11)
            torch.testing.assert_close(selected, expected, atol=5.0e-11, rtol=5.0e-11)


def test_indexed_cover_matches_dense_reference_and_preserves_native_id_mapping():
    ids = torch.tensor([[80, 20, 10, 60, 40, 30]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    anchor_coordinates = torch.tensor([[[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [2.0, 1.0]]], dtype=torch.float64)
    query = torch.tensor([[[0.45, 0.35], [1.4, 0.5]]], dtype=torch.float64, requires_grad=True)
    cover = smoothstep_grid_receiver_cover(
        query,
        anchor_coordinates,
        ids,
        mask,
        torch.tensor([[17, 18]], dtype=torch.long),
        frame_key="native-frame",
        legal_bounds=torch.tensor([[[0.0, 0.0], [2.0, 1.0]]], dtype=torch.float64),
    )
    assert cover.slots == 4
    torch.testing.assert_close(cover.weights.sum(dim=-1), torch.ones(1, 2, dtype=torch.float64))
    assert torch.equal(cover.anchor_ids[0, 0], torch.tensor([10, 40, 80, 20]))
    smooth_x = 3.0 * 0.45**2 - 2.0 * 0.45**3
    smooth_y = 3.0 * 0.35**2 - 2.0 * 0.35**3
    expected_weights = torch.tensor(
        [(1 - smooth_x) * (1 - smooth_y), smooth_x * (1 - smooth_y), (1 - smooth_x) * smooth_y, smooth_x * smooth_y],
        dtype=torch.float64,
    )
    torch.testing.assert_close(cover.weights[0, 0], expected_weights)

    anchor_values = torch.randn(1, 2, 6, 3, dtype=torch.float64, requires_grad=True)
    prepared = PreparedAnchorValues(kind="environment", values=anchor_values, anchor_ids=ids)
    indexed = gather_cover_indexed(prepared, cover)
    dense = gather_cover_dense_reference(prepared, cover)
    torch.testing.assert_close(indexed, dense, atol=2.0e-12, rtol=2.0e-12)
    grads_indexed = torch.autograd.grad(indexed.square().sum(), (query, anchor_values), retain_graph=True)
    grads_dense = torch.autograd.grad(dense.square().sum(), (query, anchor_values))
    for actual, expected in zip(grads_indexed, grads_dense):
        torch.testing.assert_close(actual, expected, atol=3.0e-11, rtol=3.0e-11)


def test_smoothstep_cover_is_c1_at_cell_boundaries_and_rejects_outside_domain():
    ids = torch.tensor([[12, 10, 15, 13, 11, 14]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    anchor_coordinates = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [2.0, 1.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]], dtype=torch.float64)
    values = torch.tensor([[[[0.2], [1.0], [3.0], [-1.0], [0.7], [2.0]]]], dtype=torch.float64)

    def read(x):
        query = torch.tensor([[[x, 0.37]]], dtype=torch.float64, requires_grad=True)
        cover = smoothstep_grid_receiver_cover(
            query, anchor_coordinates, ids, mask, torch.tensor([[99]], dtype=torch.long),
            frame_key="grid", legal_bounds=torch.tensor([[[0.0, 0.0], [2.0, 1.0]]], dtype=torch.float64),
        )
        output = gather_cover_indexed(PreparedAnchorValues("environment", values, ids), cover)
        grad = torch.autograd.grad(output.sum(), query)[0][0, 0, 0]
        return output.detach(), grad.detach()

    left, left_slope = read(1.0 - 1.0e-5)
    right, right_slope = read(1.0 + 1.0e-5)
    assert torch.max(torch.abs(left - right)) < 1.0e-8
    assert abs(float(left_slope)) < 2.0e-3
    assert abs(float(right_slope)) < 2.0e-3
    with pytest.raises(ValueError, match="legal domain"):
        smoothstep_grid_receiver_cover(
            torch.tensor([[[2.1, 0.5]]], dtype=torch.float64), anchor_coordinates, ids, mask,
            torch.tensor([[101]], dtype=torch.long), frame_key="grid",
            legal_bounds=torch.tensor([[[0.0, 0.0], [2.0, 1.0]]], dtype=torch.float64),
        )


def test_grid_axis_extraction_preserves_exact_repeated_fp32_coordinates():
    x_axis = torch.tensor([0.1, 1.0], dtype=torch.float32)
    y_axis = torch.arange(7, dtype=torch.float32)
    coordinates = torch.cartesian_prod(x_axis, y_axis)[None]
    ids = torch.arange(200, 214, dtype=torch.long)[None]
    mask = torch.ones_like(ids, dtype=torch.bool)
    query = torch.tensor([[[0.1, 3.2]]], dtype=torch.float32)
    cover = smoothstep_grid_receiver_cover(
        query,
        coordinates,
        ids,
        mask,
        torch.tensor([[71]], dtype=torch.long),
        frame_key="exact-fp32-grid",
        legal_bounds=torch.tensor([[[0.1, 0.0], [1.0, 6.0]]], dtype=torch.float32),
    )
    resolved = ids[0, cover.anchor_indices[0, 0]]
    expected = torch.tensor([203, 210, 204, 211], dtype=torch.long)
    assert torch.equal(resolved, expected)


def test_inherited_query_batch_wraps_without_changing_querybatch():
    query = QueryBatch(
        coordinates=torch.tensor([[[0.2, 0.3], [0.8, 0.4]]]),
        query_ids=torch.tensor([[5, 9]]),
        coordinate_frame_key="fixture-frame",
        receiver_key="native-receivers",
    )
    ids = torch.tensor([[10, 11]], dtype=torch.long)
    cover = _cover(ids, torch.tensor([[[0.4, 0.6], [0.7, 0.3]]]), indices=torch.tensor([[[0, 1], [1, 0]]]))
    cover = ReceiverCover(
        anchor_indices=cover.anchor_indices,
        anchor_ids=cover.anchor_ids,
        weights=cover.weights,
        validity=cover.validity,
        query_ids=query.query_ids,
        frame_key="fixture-frame",
    )
    wrapped = with_receiver_cover(query, cover)
    assert isinstance(wrapped, InheritedQueryBatch)
    assert wrapped.receiver_cover is cover
    assert wrapped.receiver_key == "native-receivers"
    assert wrapped.query_ids is query.query_ids
    assert not hasattr(query, "receiver_cover")
    with pytest.raises(ValueError, match="query IDs"):
        wrong = ReceiverCover(
            anchor_indices=cover.anchor_indices,
            anchor_ids=cover.anchor_ids,
            weights=cover.weights,
            validity=cover.validity,
            query_ids=torch.tensor([[6, 9]]),
            frame_key="fixture-frame",
        )
        with_receiver_cover(query, wrong)


def test_receiver_cover_detects_tensor_mutation_before_execution():
    ids = torch.tensor([[21, 22]], dtype=torch.long)
    cover = _cover(ids, torch.tensor([[[0.25, 0.75]]]), indices=torch.tensor([[[1, 0]]]))
    assert {name for name, _version in cover.tensor_versions} == {
        "anchor_indices", "anchor_ids", "weights", "validity", "query_ids"
    }
    cover.weights[0, 0, 0] = 0.5
    with pytest.raises(RuntimeError, match="changed after cover construction"):
        gather_cover_indexed(torch.randn(1, 1, 2, 3), cover, validate_identity=False)


def test_receiver_cover_accepts_inference_mode_geometry_tensors():
    ids = torch.tensor([[31, 32]], dtype=torch.long)
    with torch.inference_mode():
        cover = _cover(ids, torch.tensor([[[0.4, 0.6]]]), indices=torch.tensor([[[0, 1]]]))
        values = PreparedAnchorValues("environment", torch.ones(1, 1, 2, 1), ids)
        output = gather_cover_indexed(values, cover)
    torch.testing.assert_close(output, torch.ones_like(output))
    assert all(version == -1 for _name, version in cover.tensor_versions)


def test_source_indexed_affine_transport_matches_dense_formula_and_adjoint():
    torch.manual_seed(27)
    logits = torch.tensor(
        [[[[0.9, 0.25, -0.6, -1.4], [0.2, 0.8, -0.1, -0.9], [-0.4, 0.7, 0.95, -0.3]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    measure = torch.tensor([[0.12, 0.28, 0.35, 0.25]], dtype=torch.float64, requires_grad=True)
    carriers = torch.randn(1, 1, 4, 2, dtype=torch.float64, requires_grad=True)
    controls = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
    phi = torch.randn(1, 2, 1, 2, dtype=torch.float64, requires_grad=True)
    cover_weights = torch.tensor([[[0.4, 0.6], [0.7, 0.3]]], dtype=torch.float64, requires_grad=True)
    snapshot = _snapshot(logits, measure)
    atlas = build_typed_atlas(snapshot, normalization="entmax15")
    cover = _cover(snapshot.anchor_ids, cover_weights, indices=torch.tensor([[[0, 2], [1, 2]]]))

    anchor_response = prepare_source_indexed_anchor_response(atlas, carriers, controls)
    anchor_response_selected = prepare_source_indexed_anchor_response(
        atlas, carriers, controls, executor="selected"
    )
    torch.testing.assert_close(
        anchor_response_selected.executed_edge_counts,
        (atlas.probabilities > 0).sum(dim=-1),
    )
    torch.testing.assert_close(
        anchor_response_selected.candidate_edge_counts,
        snapshot.donor_mask.sum(dim=-1)[:, None, None].expand_as(atlas.probabilities[..., 0])
        * snapshot.destination_mask[:, None, :],
    )
    factorized = gather_source_response_indexed(anchor_response, cover, phi)
    factorized_selected = gather_source_response_indexed(anchor_response_selected, cover, phi)
    dense_cover = torch.nn.functional.one_hot(cover.anchor_indices, num_classes=3).to(torch.float64)
    dense_cover = torch.einsum("bqce,bqc->bqe", dense_cover, cover.weights)
    density = atlas.probabilities / atlas.normalized_measure[:, None, None, :]
    literal = torch.einsum("bqe,bhen,bhnr,bn,bqhr->bq", dense_cover, density, carriers, controls, phi)
    torch.testing.assert_close(factorized, literal, atol=5.0e-12, rtol=5.0e-12)
    torch.testing.assert_close(factorized_selected, literal, atol=5.0e-12, rtol=5.0e-12)
    leaves = (logits, measure, carriers, controls, phi, cover_weights)
    grad_factorized = torch.autograd.grad(factorized.square().sum(), leaves, retain_graph=True)
    grad_factorized_selected = torch.autograd.grad(
        factorized_selected.square().sum(), leaves, retain_graph=True
    )
    grad_literal = torch.autograd.grad(literal.square().sum(), leaves)
    for actual, selected, expected in zip(grad_factorized, grad_factorized_selected, grad_literal):
        torch.testing.assert_close(actual, expected, atol=2.0e-10, rtol=2.0e-10)
        torch.testing.assert_close(selected, expected, atol=2.0e-10, rtol=2.0e-10)


def test_staged_cycle_exports_only_opted_in_last_route_score_snapshots():
    torch.manual_seed(31)
    batch, modules, environments, width = 1, 3, 4, 8
    scene = SimpleNamespace(
        module_characteristic_lengths=torch.ones(batch, modules, 2),
        environment_characteristic_lengths=torch.ones(batch, environments, 2),
        module_measure=torch.tensor([[0.2, 0.3, 0.5]]),
        environment_measure=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        module_measure_reference=torch.ones(batch),
        environment_measure_reference=torch.ones(batch),
        module_coordinates=torch.rand(batch, modules, 2),
        environment_coordinates=torch.rand(batch, environments, 2),
        module_mask=torch.ones(batch, modules, dtype=torch.bool),
        environment_mask=torch.ones(batch, environments, dtype=torch.bool),
        module_ids=torch.tensor([[4, 2, 8]]),
        environment_ids=torch.tensor([[19, 3, 11, 7]]),
    )
    block = StagedContextCycleV2(width, heads=2, coordinate_dim=2).eval()
    module_states = torch.randn(batch, modules, width)
    environment_states = torch.randn(batch, environments, width)
    frame_scale = torch.ones(batch, 2)
    ordinary = block(module_states, environment_states, scene, frame_scale)
    exported = block(module_states, environment_states, scene, frame_scale, export_route_scores=True)
    torch.testing.assert_close(ordinary[0], exported[0])
    torch.testing.assert_close(ordinary[1], exported[1])
    assert "em_score_snapshot" not in ordinary[2]
    em = exported[2]["em_score_snapshot"]
    ee = exported[2]["ee_score_snapshot"]
    assert isinstance(em, RouteScoreSnapshot) and isinstance(ee, RouteScoreSnapshot)
    assert em.score_origin == "last_em" and ee.score_origin == "last_ee"
    assert torch.equal(em.anchor_ids, scene.environment_ids)
    assert torch.equal(em.donor_ids, scene.module_ids)
    assert torch.equal(ee.anchor_ids, scene.environment_ids)
    assert torch.equal(ee.donor_ids, scene.environment_ids)
    assert em.raw_logits.shape == (batch, 2, environments, modules)
    assert ee.raw_logits.shape == (batch, 2, environments, environments)
    atlas = build_typed_atlas(em)
    explicit = measure_relative_softmax(
        em.raw_logits, em.donor_measure, em.donor_mask, em.destination_mask
    )
    torch.testing.assert_close(atlas.probabilities, explicit)
