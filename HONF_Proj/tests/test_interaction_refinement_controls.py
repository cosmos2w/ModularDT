from __future__ import annotations

import pytest
import torch

from honf_forward_core.interface_fields.interaction_refinement import (
    PreparedRefinementResponse,
    RefinedSourceResponseOperator,
    RefinementPolicy,
    SourceRefinement,
)
from honf_forward_core.interface_fields.source_response_operator import SourceResponseOperator


def _fine_and_refined():
    torch.manual_seed(701)
    fine = SourceResponseOperator(
        source_width=2,
        context_width=1,
        environment_width=3,
        mode="direct",
        spatial_dim=2,
        hidden=8,
        message=8,
        output_width=1,
        far_hidden=8,
    )
    refined = RefinedSourceResponseOperator.from_fine(
        fine,
        base_width=4,
        router_hidden=8,
    )
    return fine, refined


def _inputs(*, requires_grad=False):
    sources = torch.tensor(
        [[[0.2, 0.1], [0.4, 0.2], [0.7, 0.3], [0.1, 0.4], [0.8, 0.6]]],
        requires_grad=requires_grad,
    )
    centers = torch.tensor(
        [[[0.51, 0.50], [0.30, 0.50], [0.60, 0.50], [0.50, 0.70], [0.10, 0.20]]],
        requires_grad=requires_grad,
    )
    return {
        "sources": sources,
        "context": torch.tensor([[0.3]]),
        "centers": centers,
        "present": torch.ones(1, 5),
        "lengths": torch.ones(1, 2),
        "source_lengths": torch.full((1, 5), 0.02),
        "source_measures": torch.ones(1, 5),
        "environment_tokens": torch.empty(1, 0, 3),
        "environment_coords": torch.empty(1, 0, 2),
        "source_ids": torch.arange(5)[None],
    }


def _receivers(*, requires_grad=False):
    return torch.tensor([[[0.50, 0.50], [0.72, 0.52]]], requires_grad=requires_grad)


def test_all_fine_and_all_fine_training_replay_preserve_the_original_operator_exactly():
    fine, refined = _fine_and_refined()
    inputs = _inputs()
    receivers = _receivers()
    forcing = torch.tensor([[0.2, -0.1, 0.05, 0.13, -0.07]], dtype=torch.float64)

    fine_context = fine.prepare_context(**inputs)
    fine_response = fine.prepare_receivers(fine_context, receivers)
    fine_values = fine.apply_forcing(fine_response, forcing, accumulation_dtype=torch.float64)
    fine_kernel = fine_response.dense_kernel(accumulation_dtype=torch.float64)

    refined_context = refined.prepare_context(**inputs)
    refined.set_execution(mode="all_fine", phase="hard", training_signal=False)
    inference_response = refined.prepare_receivers(refined_context, receivers)
    inference_values = refined.apply_forcing(inference_response, forcing, accumulation_dtype=torch.float64)
    inference_kernel = inference_response.dense_kernel(accumulation_dtype=torch.float64)
    assert torch.equal(inference_values, fine_values)
    assert torch.equal(inference_kernel, fine_kernel)

    refined.set_execution(mode="all_fine", phase="hard", training_signal=True)
    training_response = refined.prepare_receivers(refined_context, receivers)
    training_values = refined.apply_forcing(training_response, forcing, accumulation_dtype=torch.float64)
    assert training_response is training_response.full_response
    assert torch.equal(training_values, fine_values)
    assert torch.equal(training_response.dense_kernel(accumulation_dtype=torch.float64), fine_kernel)
    assert training_response.refinement_aux["fine_rows"] == receivers.shape[1] * inputs["centers"].shape[1]


def test_nearest_upstream_and_shuffle_controls_have_equal_counts_and_keep_protected_local_pairs():
    _fine, refined = _fine_and_refined()
    inputs = _inputs()
    context = refined.prepare_context(**inputs)
    query = torch.tensor([[[0.50, 0.50]]])
    probability = torch.tensor([[[0.1, 0.9, 0.1, 0.1, 0.1]]])
    protected = refined.near_weight(query, context.centers, context.source_lengths, context.present) > 0
    assert protected.tolist() == [[[True, False, False, False, False]]]

    controls = {}
    for mode in ("nearest", "upstream", "shuffle"):
        policy = RefinementPolicy(mode=mode, phase="hard", threshold=0.5)
        controls[mode] = SourceRefinement.route(
            context,
            query,
            probability,
            protected,
            policy,
        )

    assert all(route.sum().item() == 2 for route in controls.values())
    assert all(route[0, 0, 0] for route in controls.values())
    assert controls["nearest"].tolist() == [[[True, False, True, False, False]]]
    assert controls["upstream"].tolist() == [[[True, True, False, False, False]]]
    assert torch.equal(
        controls["shuffle"],
        SourceRefinement.route(
            context,
            query,
            probability,
            protected,
            RefinementPolicy(mode="shuffle", phase="hard", threshold=0.5),
        ),
    )


def test_source_permutation_preserves_physical_kernel_columns_route_membership_and_output():
    _fine, refined = _fine_and_refined()
    inputs = _inputs()
    receivers = _receivers()
    refined.set_execution(mode="adaptive", phase="hard", threshold=0.95)
    original = refined.prepare_receivers(refined.prepare_context(**inputs), receivers)
    assert isinstance(original, PreparedRefinementResponse)

    permutation = torch.tensor([2, 0, 4, 1, 3])
    permuted_inputs = dict(inputs)
    for name in ("sources", "centers", "present", "source_lengths", "source_measures", "source_ids"):
        permuted_inputs[name] = inputs[name][:, permutation]
    permuted = refined.prepare_receivers(refined.prepare_context(**permuted_inputs), receivers)
    forcing = torch.tensor([[0.21, -0.08, 0.04, 0.13, -0.11]], dtype=torch.float64)

    torch.testing.assert_close(permuted.keep, original.keep[:, :, permutation])
    torch.testing.assert_close(
        permuted.dense_kernel(accumulation_dtype=torch.float64),
        original.dense_kernel(accumulation_dtype=torch.float64)[:, :, permutation],
        atol=2e-6,
        rtol=2e-6,
    )
    original_value = refined.apply_increment(
        original,
        forcing,
        accumulation_dtype=torch.float64,
    )
    permuted_value = refined.apply_increment(
        permuted,
        forcing[:, permutation],
        accumulation_dtype=torch.float64,
    )
    torch.testing.assert_close(permuted_value, original_value, atol=2e-6, rtol=2e-6)


def test_fixed_route_protects_exact_near_source_and_preserves_live_input_and_control_gradients():
    _fine, refined = _fine_and_refined()
    inputs = _inputs(requires_grad=True)
    context = refined.prepare_context(**inputs)
    receivers = torch.tensor([[[0.50, 0.50]]], requires_grad=True)
    fixed_route = torch.tensor([[[False, False, True, False, False]]])
    refined.set_execution(mode="adaptive", phase="hard", threshold=0.99)
    response = refined.prepare_receivers(context, receivers, fixed_route=fixed_route)

    assert isinstance(response, PreparedRefinementResponse)
    assert response.keep.tolist() == [[[True, False, True, False, False]]]
    full_response = refined.prepare_receivers(context, receivers)
    torch.testing.assert_close(
        response.dense_kernel(accumulation_dtype=torch.float64)[:, :, 0],
        full_response.dense_kernel(accumulation_dtype=torch.float64)[:, :, 0],
        atol=0,
        rtol=0,
    )

    forcing = torch.tensor([[0.21, -0.08, 0.04, 0.13, -0.11]], requires_grad=True)
    values = refined.apply_increment(
        response,
        forcing,
        accumulation_dtype=torch.float64,
    )
    cotangent = torch.tensor([[[0.7]]], dtype=torch.float64)
    gradient = torch.autograd.grad(
        (values * cotangent).sum(),
        (inputs["sources"], inputs["centers"], receivers, forcing),
    )
    assert all(torch.isfinite(item).all() and item.abs().sum() > 0 for item in gradient)

    expected_forcing_gradient = torch.einsum(
        "bqmo,bqo->bm",
        response.dense_kernel(accumulation_dtype=torch.float64),
        cotangent,
    ).to(forcing.dtype)
    torch.testing.assert_close(gradient[-1], expected_forcing_gradient, atol=2e-13, rtol=2e-13)

    fixed_route[0, 0, 1] = True
    with pytest.raises(ValueError, match="Prepared response context/receiver tensors changed"):
        refined.apply_increment(response, forcing, accumulation_dtype=torch.float64)


def test_training_router_surrogate_updates_scorer_without_changing_input_gradients():
    _fine, refined = _fine_and_refined()
    inputs = _inputs(requires_grad=True)
    receivers = torch.tensor([[[0.50, 0.50]]], requires_grad=True)
    forcing = torch.tensor([[0.21, -0.08, 0.04, 0.13, -0.11]])

    def gradients(training_signal):
        refined.zero_grad(set_to_none=True)
        context = refined.prepare_context(**inputs)
        refined.set_execution(
            mode="adaptive",
            phase="hard",
            threshold=0.999,
            training_signal=training_signal,
        )
        response = refined.prepare_receivers(context, receivers)
        values = refined.apply_increment(response, forcing).sum()
        watched = (
            inputs["sources"],
            inputs["centers"],
            receivers,
            *tuple(refined.refinement.router.parameters()),
        )
        result = torch.autograd.grad(values, watched, allow_unused=True)
        return values.detach(), result

    ordinary_value, ordinary_grad = gradients(False)
    training_value, training_grad = gradients(True)
    torch.testing.assert_close(training_value, ordinary_value, atol=0, rtol=0)
    for actual, expected in zip(training_grad[:3], ordinary_grad[:3], strict=True):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert all(gradient is None for gradient in ordinary_grad[3:])
    assert all(gradient is not None for gradient in training_grad[3:])
    assert sum(gradient.abs().sum() for gradient in training_grad[3:]) > 0


def test_refined_coefficients_are_heat_independent_and_precise_factor_assembly_is_explicit():
    _fine, refined = _fine_and_refined()
    inputs = _inputs()
    context = refined.prepare_context(**inputs)
    receivers = _receivers()
    refined.set_execution(mode="adaptive", phase="soft", threshold=0.85, temperature=0.7)
    response = refined.prepare_receivers(context, receivers)
    assert isinstance(response, PreparedRefinementResponse)

    base = response.base_far.to(torch.float64)
    fine = response.fine_far.to(torch.float64)
    probability = response.probability.to(torch.float64)
    protected = response.protected
    blend = torch.where(protected, torch.ones_like(probability), probability)
    far = base + blend[..., None] * (fine - base)
    near_weight = response.near_weight.to(torch.float64)
    expected_kernel = far * (1 - near_weight[..., None])
    if response.near_indices.numel():
        batch, query, source = response.near_indices
        correction = (
            near_weight[batch, query, source, None]
            * response.near_values.to(torch.float64)
        )
        expected_kernel = expected_kernel.index_put(
            (batch, query, source),
            expected_kernel[batch, query, source] + correction,
        )
    expected_kernel = expected_kernel / response.forcing_scale
    precise_kernel = response.dense_kernel(accumulation_dtype=torch.float64)
    torch.testing.assert_close(precise_kernel, expected_kernel, atol=0, rtol=0)

    first_control = torch.tensor([[0.2, -0.1, 0.05, 0.13, -0.07]], dtype=torch.float64)
    second_control = torch.tensor([[-0.3, 0.15, -0.075, -0.195, 0.105]], dtype=torch.float64)
    first = refined.apply_increment(response, first_control, accumulation_dtype=torch.float64)
    second = refined.apply_increment(response, second_control, accumulation_dtype=torch.float64)
    expected_first = torch.einsum("bqmo,bm->bqo", precise_kernel, first_control)
    expected_second = torch.einsum("bqmo,bm->bqo", precise_kernel, second_control)
    torch.testing.assert_close(first, expected_first, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(second, expected_second, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(response.dense_kernel(accumulation_dtype=torch.float64), precise_kernel)
    with pytest.raises(TypeError):
        refined.prepare_receivers(context, receivers, heat=first_control)
