from __future__ import annotations

import pytest
import torch

from honf_forward_core.content_hypergraph.source_read_plans import (
    SourceReadControllerV1,
    build_degree_bucket_packet_layout,
    build_source_read_plan_v1,
    combine_source_gates,
    directional_support,
    nearest_support,
    s_attn95_support,
    s_attn99_support,
    shuffle_support,
    smooth_near_gate,
    smooth_threshold_gate,
)


def _calibrated_controller(*, width: int = 8, coordinate_dim: int = 2) -> SourceReadControllerV1:
    controller = SourceReadControllerV1(
        width=width,
        coordinate_dim=coordinate_dim,
        heads=4,
        thermal=True,
        rank=3,
        scorer_width=5,
    )
    controller.install_calibration_(
        thresholds=(0.75, 1.5),
        context_scales=(1.0, 2.0, 3.0, 4.0),
        heating_envelope=2.5,
    )
    return controller


def test_controller_scores_without_heat_and_freezes_calibration_buffers():
    torch.manual_seed(401)
    controller = _calibrated_controller()
    query = torch.randn(2, 3, 8, requires_grad=True)
    source = torch.randn(2, 5, 8, requires_grad=True)
    descriptor = torch.randn(2, 3, 5, 5)
    scores = controller(query, source, descriptor)
    assert scores.shape == (2, 3, 5, 5)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()
    scores.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert source.grad is not None and torch.isfinite(source.grad).all()

    assert controller.project_source_states(source.detach()).shape == (2, 5, 8)
    assert controller.project_query_states(query.detach()).shape == (2, 3, 8)
    assert controller.utility_scorer[0].in_features == 8 + 8 + 5
    assert controller.utility_scorer[-1].out_features == 5
    assert controller.thresholds.numel() == 2
    assert torch.equal(controller.threshold_count, torch.tensor(2))
    assert (controller.context_scales > 0).all()
    assert controller.heating_envelope.item() > 0

    controller.mark_trained_()
    controller.freeze_metadata_()
    frozen = {name: value.clone() for name, value in controller.named_buffers()}
    controller(query.detach(), source.detach(), descriptor)
    assert all(torch.equal(frozen[name], value) for name, value in controller.named_buffers())


def test_dense_teacher_targets_are_headwise_and_thermal_is_not_alpha_weighted():
    controller = _calibrated_controller()
    alpha = torch.tensor([[[[0.0, 0.5, 1.0, 0.25], [0.0, 0.0, 0.0, 0.0]]]])
    fine = torch.ones(1, 1, 2, 8)
    base = torch.full_like(fine, 0.25)
    fine_k = torch.tensor([[[2.0, -1.0]]])
    base_k = torch.tensor([[[0.5, -0.25]]])
    targets = controller.dense_utility_targets(
        alpha,
        fine,
        fine_k,
        temperature_scale=1.0,
        completion_messages=base,
        completion_k=base_k,
    )
    assert targets.shape == (1, 1, 2, 5)
    block_norm = (2.0**0.5) * 0.75
    expected_flow = torch.tensor(
        [[[[0.0, block_norm * 0.5 / 2.0, block_norm / 3.0, block_norm * 0.25 / 4.0],
           [0.0, 0.0, 0.0, 0.0]]]]
    )
    torch.testing.assert_close(targets[..., :4], expected_flow)
    # H_i*|K_fine-K_complete| is independent of alpha, including zero alpha.
    torch.testing.assert_close(targets[..., 4], torch.tensor([[[3.75, 1.875]]]))
    assert not targets.requires_grad


def test_endpoint_flat_utility_and_near_gates_keep_live_gradients():
    scores = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], requires_grad=True)
    gate = smooth_threshold_gate(scores, 1.0)
    torch.testing.assert_close(gate, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    score_grad = torch.autograd.grad(gate.sum(), scores)[0]
    assert score_grad[0] == 0 and score_grad[1] == 0
    assert score_grad[2] > 0
    assert score_grad[3] == 0 and score_grad[4] == 0

    distance = torch.tensor([0.0, 1.0, 1.5, 2.0, 2.5], requires_grad=True)
    near = smooth_near_gate(distance, 2.0)
    torch.testing.assert_close(near, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]))
    distance_grad = torch.autograd.grad(near.sum(), distance)[0]
    assert distance_grad[0] == 0 and distance_grad[1] == 0
    assert distance_grad[2] < 0
    assert distance_grad[3] == 0 and distance_grad[4] == 0

    utility = torch.tensor([0.2, 0.7], requires_grad=True)
    near_part = torch.tensor([0.3, 0.4], requires_grad=True)
    combined = combine_source_gates(utility, near_part)
    torch.testing.assert_close(combined, 1.0 - (1.0 - utility) * (1.0 - near_part))
    assert torch.autograd.grad(combined.sum(), (utility, near_part))[0].abs().sum() > 0


def test_near_gate_broadcasts_varied_per_source_radii_and_keeps_radius_gradients():
    distance = torch.tensor(
        [[[0.75, 1.5, 3.0], [0.75, 1.5, 3.0]]], requires_grad=True
    )
    radius = torch.tensor([[[1.0, 2.0, 4.0]]], requires_grad=True)
    gate = smooth_near_gate(distance, radius)
    torch.testing.assert_close(gate, torch.full_like(distance, 0.5))
    distance_grad, radius_grad = torch.autograd.grad(gate.sum(), (distance, radius))
    assert (distance_grad < 0).all()
    assert (radius_grad > 0).all()

    with pytest.raises(ValueError, match="finite and positive"):
        smooth_near_gate(distance.detach(), torch.tensor([[[1.0, 0.0, 4.0]]]))


def test_plan_gate_validation_rejects_nonfinite_or_out_of_range_cpu_values():
    alpha = torch.zeros(1, 1, 2, 4)
    present = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="gates must lie in \\[0,1\\]"):
        build_source_read_plan_v1(torch.tensor([[[0.0, 1.1]]]), alpha, present)
    with pytest.raises(ValueError, match="gates must be finite"):
        build_source_read_plan_v1(torch.tensor([[[0.0, float("nan")]]]), alpha, present)
    with pytest.raises(ValueError, match="utility_gate must be finite"):
        combine_source_gates(torch.tensor([float("nan")]), torch.tensor([0.0]))


def test_flat_pairs_and_degree_packets_preserve_support_alpha_and_protected_zero_mass():
    gates = torch.tensor(
        [
            [[0.0, 0.2, 0.0, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.4, 0.0, 0.2], [0.0, 0.6, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        ]
    )
    alpha = torch.arange(2 * 3 * 5 * 4, dtype=torch.float32).reshape(2, 3, 5, 4) / 100.0
    alpha[0, 0, 2] = 0.0
    source_present = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    query_valid = torch.tensor([[True, True, False], [True, True, True]])
    protected = torch.zeros(2, 3, 5, dtype=torch.bool)
    protected[0, 0, 2] = True  # Present, zero-attention source remains thermally addressable.
    protected[1, 2, 4] = True

    plan = build_source_read_plan_v1(
        gates,
        alpha,
        source_present,
        query_valid_mask=query_valid,
        protected_mask=protected,
    )
    assert plan.support_mask[0, 0].tolist() == [False, True, True, True, False]
    assert plan.support_mask[0, 1].sum() == 0
    assert plan.support_mask[0, 2].sum() == 0
    assert plan.support_mask[1, 2].tolist() == [False, False, False, False, True]
    assert plan.gates[0, 0, 2] == 1.0
    assert torch.equal(plan.original_alpha, alpha)

    packed = plan.packed_pairs
    expected_triplets = plan.support_mask.nonzero(as_tuple=True)
    assert torch.equal(packed.batch_index, expected_triplets[0])
    assert torch.equal(packed.query_index, expected_triplets[1])
    assert torch.equal(packed.source_index, expected_triplets[2])
    assert packed.pair_count == int(plan.support_mask.sum())
    assert torch.equal(
        packed.original_alpha,
        alpha[packed.batch_index, packed.query_index, packed.source_index],
    )

    layout = build_degree_bucket_packet_layout(plan)
    reconstructed = torch.zeros_like(plan.support_mask)
    for bucket in layout.buckets:
        assert bucket.source_indices.shape == bucket.padding_mask.shape
        bucket_queries = zip(bucket.batch_index.tolist(), bucket.query_index.tolist(), strict=True)
        for row, (batch_index, query_index) in enumerate(bucket_queries):
            valid_sources = bucket.source_indices[row, ~bucket.padding_mask[row]]
            assert (valid_sources >= 0).all()
            assert valid_sources.numel() == int(plan.support_mask[batch_index, query_index].sum())
            reconstructed[batch_index, query_index, valid_sources] = True
    assert torch.equal(reconstructed, plan.support_mask)
    assert layout.valid_pair_count == plan.pair_count
    assert layout.allocated_pair_slots == layout.valid_pair_count + layout.padding_pair_count
    assert layout.bucket_assignment_query_count == 6
    assert layout.empty_query_index.numel() == 2


def test_degree_buckets_use_general_indices_beyond_machine_word_masks_and_empty_sources():
    sources = 130
    gates = torch.zeros(1, 2, sources)
    gates[0, 0, torch.tensor([0, 64, 129])] = 1.0
    alpha = torch.zeros(1, 2, sources, 4)
    present = torch.ones(1, sources, dtype=torch.bool)
    plan = build_source_read_plan_v1(gates, alpha, present)
    layout = build_degree_bucket_packet_layout(plan)
    bucket = next(bucket for bucket in layout.buckets if bucket.width == 4 and bucket.query_count)
    assert bucket.query_index.tolist() == [0]
    assert bucket.source_indices[0].tolist() == [0, 64, 129, -1]
    assert bucket.padding_mask[0].tolist() == [False, False, False, True]
    assert layout.valid_pair_count == 3
    assert layout.padding_pair_count == 1

    empty_gates = torch.zeros(2, 3, 0)
    empty_alpha = torch.zeros(2, 3, 0, 4)
    empty_present = torch.zeros(2, 0, dtype=torch.bool)
    empty_plan = build_source_read_plan_v1(empty_gates, empty_alpha, empty_present)
    empty_layout = build_degree_bucket_packet_layout(empty_plan)
    assert empty_plan.pair_count == 0
    assert empty_layout.buckets == ()
    assert empty_layout.empty_batch_index.numel() == 6
    assert empty_layout.empty_query_index.numel() == 6


def test_s_attn_95_99_union_heads_and_keep_protected_zero_attention_sources():
    alpha = torch.tensor(
        [[[[0.90, 0.00], [0.085, 0.95], [0.015, 0.05], [0.00, 0.00]]]]
    )
    present = torch.ones(1, 4, dtype=torch.bool)
    protected = torch.tensor([[False, False, False, True]])
    support95 = s_attn95_support(alpha, present)
    support99 = s_attn99_support(alpha, present, protected_mask=protected)
    assert support95.tolist() == [[[True, True, False, False]]]
    assert support99.tolist() == [[[True, True, True, True]]]

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = s_attn95_support(alpha.index_select(2, permutation), present.index_select(1, permutation))
    assert torch.equal(permuted, support95.index_select(2, permutation))


def test_equal_count_controls_include_protected_sources_and_are_permutation_equivariant():
    query = torch.tensor([[[0.0, 0.0]]])
    source = torch.tensor([[[1.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -3.0], [0.2, 0.1]]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    target = 3
    protected = torch.tensor([[False, True, False, False, False]])
    direction = torch.tensor([1.0, 0.0])
    nearest = nearest_support(query, source, valid, target, protected_mask=protected)
    directional = directional_support(query, source, direction, valid, target, protected_mask=protected)
    scores = torch.tensor([[[0.1, 0.8, 0.2, 0.4, 0.3]]])
    shuffled = shuffle_support(valid, target, protected_mask=protected, shuffle_scores=scores)
    for support in (nearest, directional, shuffled):
        assert int(support.sum()) == target
        assert support[0, 0, 1]

    permutation = torch.tensor([4, 2, 0, 3, 1])
    source_permuted = source.index_select(1, permutation)
    valid_permuted = valid.index_select(1, permutation)
    protected_permuted = protected.index_select(1, permutation)
    nearest_permuted = nearest_support(
        query, source_permuted, valid_permuted, target, protected_mask=protected_permuted
    )
    directional_permuted = directional_support(
        query, source_permuted, direction, valid_permuted, target, protected_mask=protected_permuted
    )
    shuffled_permuted = shuffle_support(
        valid_permuted,
        target,
        protected_mask=protected_permuted,
        shuffle_scores=scores.index_select(2, permutation),
    )
    assert torch.equal(nearest_permuted, nearest.index_select(2, permutation))
    assert torch.equal(directional_permuted, directional.index_select(2, permutation))
    assert torch.equal(shuffled_permuted, shuffled.index_select(2, permutation))


def test_controller_and_rank_completion_are_source_permutation_and_query_subset_consistent():
    torch.manual_seed(402)
    controller = _calibrated_controller().eval()
    query = torch.randn(2, 4, 8)
    source = torch.randn(2, 6, 8)
    descriptor = torch.randn(2, 4, 6, 5)
    alpha = torch.rand(2, 4, 6, 4)
    scores = controller(query, source, descriptor)
    completion = controller.completion_headwise_sum(query, source, alpha)

    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    permuted_scores = controller(query, source.index_select(1, permutation), descriptor.index_select(2, permutation))
    permuted_completion = controller.completion_headwise_sum(
        query,
        source.index_select(1, permutation),
        alpha.index_select(2, permutation),
    )
    torch.testing.assert_close(permuted_scores, scores.index_select(2, permutation))
    torch.testing.assert_close(permuted_completion, completion)

    query_subset = torch.tensor([0, 2])
    subset_scores = controller(
        query.index_select(1, query_subset),
        source,
        descriptor.index_select(1, query_subset),
    )
    torch.testing.assert_close(subset_scores, scores.index_select(1, query_subset))
    subset_completion = controller.completion_headwise_sum(
        query.index_select(1, query_subset),
        source,
        alpha.index_select(1, query_subset),
    )
    torch.testing.assert_close(subset_completion, completion.index_select(1, query_subset))


def test_prepared_completion_pair_and_headwise_contractions_match_dense_formula():
    torch.manual_seed(403)
    controller = _calibrated_controller()
    query = torch.randn(2, 3, 8)
    source = torch.randn(2, 5, 8)
    alpha = torch.rand(2, 3, 5, 4)
    q16 = controller.project_completion_queries(query)
    s16 = controller.project_completion_sources(source)
    headwise = controller.completion_headwise_sum_from_projections(q16, s16, alpha)

    pair_b = torch.tensor([0, 1, 1])
    pair_q = torch.tensor([2, 0, 1])
    pair_s = torch.tensor([4, 3, 1])
    pair_hidden = controller.completion_pair_hidden_from_projections(q16, s16, pair_b, pair_q, pair_s)
    expected_pairs = controller.completion_output(
        q16[pair_b, pair_q] * s16[pair_b, pair_s]
    )
    torch.testing.assert_close(pair_hidden, expected_pairs)

    dense_features = q16[:, :, None, None, :] * s16[:, None, :, None, :]
    dense_features = dense_features.expand(2, 3, 5, 4, 3)
    dense_completion = controller.completion_output(dense_features)
    expected = torch.einsum("bqmh,bqmhd->bqhd", alpha, dense_completion)
    torch.testing.assert_close(headwise, expected, atol=2.0e-6, rtol=2.0e-6)

    weights = torch.rand(2, 3, 5)
    weighted = controller.completion_weighted_sum_from_projections(q16, s16, weights)
    expected_weighted = torch.einsum("bqm,bqmd->bqd", weights, dense_completion[..., 0, :])
    torch.testing.assert_close(weighted, expected_weighted, atol=2.0e-6, rtol=2.0e-6)


def test_utility_controller_omits_completion_parameters_and_rejects_completion_calls():
    utility = SourceReadControllerV1(
        width=8,
        coordinate_dim=2,
        heads=4,
        thermal=True,
        rank=3,
        scorer_width=5,
        completion=False,
    )
    complete = SourceReadControllerV1(
        width=8,
        coordinate_dim=2,
        heads=4,
        thermal=True,
        rank=3,
        scorer_width=5,
    )

    completion_parameter_names = (
        "completion_query_projection.weight",
        "completion_source_projection.weight",
        "completion_output.weight",
    )
    utility_names = {name for name, _ in utility.named_parameters()}
    complete_named_parameters = dict(complete.named_parameters())
    complete_names = set(complete_named_parameters)
    assert all(name not in utility_names for name in completion_parameter_names)
    assert all(name in complete_names for name in completion_parameter_names)
    assert utility.completion_query_projection is None
    assert utility.completion_source_projection is None
    assert utility.completion_output is None
    assert not any("completion_" in key for key in utility.state_dict())
    assert sum(parameter.numel() for parameter in utility.parameters()) == sum(
        parameter.numel() for parameter in complete.parameters()
    ) - sum(complete_named_parameters[name].numel() for name in completion_parameter_names)

    query = torch.randn(1, 2, 8)
    source = torch.randn(1, 3, 8)
    alpha = torch.randn(1, 2, 3, 4)
    pair_indices = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="completion is disabled"):
        utility.project_completion_queries(query)
    with pytest.raises(ValueError, match="completion is disabled"):
        utility.project_completion_sources(source)
    with pytest.raises(ValueError, match="completion is disabled"):
        utility.completion_pair_hidden_from_projections(
            torch.zeros(1, 2, 3), torch.zeros(1, 3, 3), pair_indices, pair_indices, pair_indices
        )
    with pytest.raises(ValueError, match="completion is disabled"):
        utility.completion_headwise_sum_from_projections(
            torch.zeros(1, 2, 3), torch.zeros(1, 3, 3), alpha
        )
    with pytest.raises(ValueError, match="completion is disabled"):
        utility.completion_weighted_sum_from_projections(
            torch.zeros(1, 2, 3), torch.zeros(1, 3, 3), alpha[..., 0]
        )
    with pytest.raises(ValueError, match="completion is disabled"):
        utility.completion_pair_hidden(query, source, pair_indices, pair_indices, pair_indices)
    with pytest.raises(ValueError, match="completion is disabled"):
        utility.completion_headwise_sum(query, source, alpha)
