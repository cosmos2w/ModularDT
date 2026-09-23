"""CPU algebra, limiting-case, and chunked-read checks for Run 1503."""

from __future__ import annotations

import torch

from honf_forward_core.config import UnifiedForwardConfig
from honf_forward_core.interface_fields import (
    AdaptiveHyperedgeOpeningPairwiseField,
    InterfaceFieldCore,
    SparseIncidenceGroupControlPairwiseField,
    adaptive_hyperedge_opening,
    combine_opened_group_responses,
    masked_mass_softmax,
    mass_weighted_environment_aggregates,
    opening_blend,
)
from honf_forward_core.interface_fields.group_control_router import GroupQueryRoute
from honf_forward_core.interface_fields.routing_index.sparse_projection import masked_sparsemax
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def test_mass_weighted_groups_are_permutation_invariant_and_partition_measure() -> None:
    states = torch.tensor([[[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]]])
    coordinates = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]])
    membership = torch.tensor([[[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]]])
    measure = torch.tensor([[0.2, 0.3, 0.5]])
    expected = mass_weighted_environment_aggregates(
        states, coordinates, membership, measure
    )
    permutation = torch.tensor([2, 0, 1])
    permuted = mass_weighted_environment_aggregates(
        states[:, permutation],
        coordinates[:, permutation],
        membership[:, permutation],
        measure[:, permutation],
    )
    torch.testing.assert_close(expected.mass, permuted.mass)
    torch.testing.assert_close(expected.centroids, permuted.centroids)
    torch.testing.assert_close(expected.radius_sq, permuted.radius_sq)
    torch.testing.assert_close(expected.states, permuted.states)
    torch.testing.assert_close(expected.source_mass.sum(dim=1), expected.mass)
    torch.testing.assert_close(expected.mass.sum(dim=-1), torch.ones(1))


def test_zero_mass_and_split_source_boundaries_are_finite_and_mass_preserving() -> None:
    states = torch.tensor([[[2.0, -3.0], [2.0, -3.0]]])
    coordinates = torch.tensor([[[0.0, 0.0], [4.0, 0.0]]])
    # The second source is split across two groups; the fourth group is empty.
    membership = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.4, 0.6, 0.0]]])
    measure = torch.tensor([[0.5, 0.5]])
    result = mass_weighted_environment_aggregates(
        states, coordinates, membership, measure
    )
    torch.testing.assert_close(result.mass, torch.tensor([[0.5, 0.2, 0.3, 0.0]]))
    torch.testing.assert_close(result.states[0, 0], states[0, 0])
    torch.testing.assert_close(result.states[0, 1], states[0, 1])
    torch.testing.assert_close(result.states[0, 3], torch.zeros(2))
    assert torch.isfinite(result.states).all()
    assert torch.isfinite(result.centroids).all()
    assert torch.isfinite(result.radius_sq).all()


def test_one_source_group_aggregate_reproduces_source_state() -> None:
    states = torch.tensor([[[1.5, -2.0, 4.0]]])
    coordinates = torch.tensor([[[3.0, 5.0]]])
    membership = torch.zeros((1, 1, 3))
    membership[..., 1] = 1.0
    measure = torch.tensor([[0.37]])
    result = mass_weighted_environment_aggregates(
        states, coordinates, membership, measure
    )
    torch.testing.assert_close(result.mass[0, 1], measure[0, 0])
    torch.testing.assert_close(result.states[0, 1], states[0, 0])
    torch.testing.assert_close(result.centroids[0, 1], coordinates[0, 0])
    torch.testing.assert_close(result.radius_sq[0, 1], torch.zeros(()))


def test_same_logits_drive_mass_softmax_and_sparse_opening_blend() -> None:
    logits = torch.tensor([[[2.0, 0.5, -1.0], [0.0, 2.0, -1.0]]])
    mass = torch.tensor([[0.25, 0.75, 0.0]])
    p = masked_mass_softmax(logits, mass)
    assert p.shape == logits.shape
    torch.testing.assert_close(p.sum(dim=-1), torch.ones((1, 2)))
    assert torch.equal(p[..., 2], torch.zeros((1, 2)))
    alpha = torch.tensor([[[0.0, 1.0, 0.0], [0.4, 0.6, 0.0]]])
    opening = opening_blend(alpha, p)
    assert torch.equal(opening[alpha == 0.0], torch.zeros_like(opening[alpha == 0.0]))
    coarse = torch.ones((1, 2, 3, 4))
    fine = 2.0 * coarse
    total, coarse_part, fine_part, returned_opening = combine_opened_group_responses(
        coarse, fine, p, alpha
    )
    torch.testing.assert_close(returned_opening, opening)
    torch.testing.assert_close(total, coarse_part + fine_part)
    assert torch.isfinite(total).all()


def test_router_membership_and_query_routes_are_normalized_from_one_logit_bank() -> None:
    encoded, module_states = _encoded_case(seed=1507)
    model = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
    )
    state = model.prepare(encoded, module_states)
    controls = state["group_control_state"]
    membership = controls.environment_membership
    assert bool((membership >= 0.0).all())
    torch.testing.assert_close(
        membership.sum(dim=-1),
        torch.ones_like(membership.sum(dim=-1)),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    receivers = torch.rand(2, 5, 2) * torch.tensor([12.0, 6.0])
    receiver_features = torch.randn(2, 5, 6)
    route = model._route(state, encoded, receivers, receiver_features)
    alpha = route.assignment
    assert bool((alpha >= 0.0).all())
    torch.testing.assert_close(alpha.sum(dim=-1), torch.ones((2, 5)))
    valid = controls.phase_occupied[:, None, :].expand_as(route.logits)
    torch.testing.assert_close(alpha, masked_sparsemax(route.logits, valid))
    p = masked_mass_softmax(route.logits, state["adaptive_environment_mass"])
    torch.testing.assert_close(p.sum(dim=-1), torch.ones((2, 5)))
    torch.testing.assert_close(
        opening_blend(alpha, p),
        torch.clamp(alpha / (p + 1.0e-8), min=0.0, max=1.0),
    )


def test_coarse_gate_reuses_route_logits_and_aggregate_value() -> None:
    encoded, module_states = _encoded_case(seed=1511)
    model = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
    )
    state = model.prepare(encoded, module_states)
    receivers = torch.rand(2, 4, 2) * torch.tensor([12.0, 6.0])
    receiver_features = torch.randn(2, 4, 6)
    controls = state["group_control_state"]
    route = model._route(state, encoded, receivers, receiver_features)
    coarse = model._coarse_group_response(
        state,
        encoded,
        receivers,
        receiver_features,
        controls,
        route.logits,
    )
    values = state["adaptive_coarse_values"].permute(0, 2, 1, 3)
    gate = 1.0 + torch.tanh(route.logits) / (float(model.head_dim) ** 0.5)
    expected = model.env_attention.output(
        (values[:, None, :, :, :] * gate[..., None, None]).reshape(
            2, 4, model.group_count, model.hidden_dim
        )
    )
    torch.testing.assert_close(coarse, expected, rtol=0.0, atol=0.0)
    shifted_receivers = receivers + torch.tensor([1.0, 0.0])
    shifted_route = model._route(
        state, encoded, shifted_receivers, receiver_features
    )
    assert not torch.allclose(route.logits, shifted_route.logits)
    shifted_coarse = model._coarse_group_response(
        state,
        encoded,
        shifted_receivers,
        receiver_features,
        controls,
        shifted_route.logits,
    )
    assert not torch.allclose(coarse, shifted_coarse)
    zero_value_state = dict(state)
    zero_value_state["adaptive_coarse_values"] = torch.zeros_like(values).permute(0, 2, 1, 3)
    zero_value_coarse = model._coarse_group_response(
        zero_value_state,
        encoded,
        receivers,
        receiver_features,
        controls,
        route.logits,
    )
    assert not torch.allclose(coarse, zero_value_coarse)


def test_one_active_group_has_finite_normalized_mass_route() -> None:
    logits = torch.zeros((1, 3, 4))
    mass = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    p = masked_mass_softmax(logits, mass)
    alpha = torch.zeros_like(p)
    alpha[..., 1] = 1.0
    opening = opening_blend(alpha, p)
    torch.testing.assert_close(p, alpha)
    torch.testing.assert_close(p.sum(dim=-1), torch.ones((1, 3)))
    assert torch.isfinite(opening).all()


def test_all_open_groups_partition_source_mass_without_duplicate_measure() -> None:
    encoded, module_states = _encoded_case(seed=1508)
    model = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
    )
    state = model.prepare(encoded, module_states)
    batch, query_count = 2, 4
    receivers = torch.rand(batch, query_count, 2) * torch.tensor([12.0, 6.0])
    receiver_features = torch.randn(batch, query_count, 6)
    controls = state["group_control_state"]
    alpha = receivers.new_full((batch, query_count, model.group_count), 1.0 / model.group_count)
    route = GroupQueryRoute(
        query_control=receivers.new_zeros((batch, query_count, model.group_control_dim)),
        assignment=alpha,
        logits=receivers.new_zeros(alpha.shape),
    )
    context, aux = model._read_environment(
        state,
        encoded,
        receivers,
        receiver_features,
        route,
        include_diagnostics=True,
    )
    assert torch.isfinite(context).all()
    torch.testing.assert_close(
        state["adaptive_environment_source_mass"].sum(dim=(1, 2)),
        controls.environment_measure.sum(dim=-1),
    )
    torch.testing.assert_close(
        aux["group_control_adaptive_query_count_per_group"],
        torch.full((batch, model.group_count), float(query_count)),
    )
    positive_groups = state["adaptive_environment_mass"] > 0.0
    assert bool((aux["group_control_adaptive_source_count_per_group"][positive_groups] > 0.0).all())


def test_no_open_group_for_query_keeps_complete_coarse_mixture() -> None:
    encoded, module_states = _encoded_case(seed=1509)
    model = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
    )
    state = model.prepare(encoded, module_states)
    batch, query_count = 2, 3
    receivers = torch.rand(batch, query_count, 2) * torch.tensor([12.0, 6.0])
    receiver_features = torch.randn(batch, query_count, 6)
    alpha = receivers.new_zeros((batch, query_count, model.group_count))
    alpha[:, 1, 0] = 1.0
    route = GroupQueryRoute(
        query_control=receivers.new_zeros((batch, query_count, model.group_control_dim)),
        assignment=alpha,
        logits=receivers.new_zeros(alpha.shape),
    )
    context, aux = model._read_environment(
        state,
        encoded,
        receivers,
        receiver_features,
        route,
        include_diagnostics=True,
    )
    torch.testing.assert_close(
        aux["group_control_adaptive_fine_contribution"][:, 0],
        torch.zeros_like(aux["group_control_adaptive_fine_contribution"][:, 0]),
    )
    torch.testing.assert_close(
        context[:, 0], aux["group_control_adaptive_coarse_contribution"][:, 0],
    )
    assert torch.isfinite(context).all()


def test_identical_within_group_states_align_coarse_and_fine_responses() -> None:
    encoded, module_states = _encoded_case(seed=1510)
    model = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
    )
    state = model.prepare(encoded, module_states)
    controls = state["group_control_state"]
    raw = state["adaptive_environment_raw_values"].clone()
    raw[:] = raw[:, :, :1, :]
    coarse = state["adaptive_coarse_values"].clone()
    gain = 1.0 + torch.tanh(model.environment_value_control(controls.group_control))
    gain = gain.reshape(2, model.group_count, model.num_heads, model.head_dim).permute(0, 2, 1, 3)
    coarse[:] = raw[:, :, :1, :] * gain
    modified = dict(state)
    modified["adaptive_environment_raw_values"] = raw
    modified["adaptive_coarse_values"] = coarse
    receivers = torch.rand(2, 4, 2) * torch.tensor([12.0, 6.0])
    receiver_features = torch.randn(2, 4, 6)
    route = model._route(modified, encoded, receivers, receiver_features)
    _, aux = model._read_environment(
        modified,
        encoded,
        receivers,
        receiver_features,
        route,
        include_diagnostics=True,
    )
    active = (
        modified["adaptive_environment_mass"][:, None, :] > 0.0
    ) & (route.assignment > 0.0)
    coarse_group = aux["group_control_adaptive_coarse_group_response"]
    fine_group = aux["group_control_adaptive_fine_group_response"]
    torch.testing.assert_close(
        coarse_group[active],
        fine_group[active],
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_alpha_support_transition_has_finite_design_and_output_gradients() -> None:
    mass = torch.ones((1, 2))
    for delta in (1.0e-4, -1.0e-4):
        design = torch.tensor(0.5 + delta, requires_grad=True)
        logits = torch.stack((0.5 + design, 0.5 - design), dim=-1).reshape(1, 1, 2)
        alpha = masked_sparsemax(logits)
        p = masked_mass_softmax(logits, mass)
        coarse = torch.ones((1, 1, 2, 3), requires_grad=True)
        fine = (2.0 * torch.ones((1, 1, 2, 3))).requires_grad_()
        total, _, _, _ = combine_opened_group_responses(coarse, fine, p, alpha)
        total.square().sum().backward()
        assert torch.isfinite(design.grad)
        assert torch.isfinite(coarse.grad).all()
        assert torch.isfinite(fine.grad).all()


def test_fine_group_checkpoint_matches_forward_and_gradient_and_is_used(monkeypatch) -> None:
    encoded, source_states = _encoded_case(seed=1512)
    model = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
        activation_checkpointing=False,
    )
    model.train()
    baseline_states = source_states.detach().clone().requires_grad_()
    baseline_state = model.prepare(encoded, baseline_states)
    receivers = torch.rand(2, 6, 2) * torch.tensor([12.0, 6.0])
    baseline_features = torch.randn(2, 6, 6, requires_grad=True)
    baseline_route = model._route(baseline_state, encoded, receivers, baseline_features)
    baseline_context, _ = model._read_environment(
        baseline_state,
        encoded,
        receivers,
        baseline_features,
        baseline_route,
        include_diagnostics=False,
    )
    baseline_context.square().mean().backward()
    baseline_state_gradient = baseline_states.grad.detach().clone()
    baseline_feature_gradient = baseline_features.grad.detach().clone()
    baseline_parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    checkpointed_states = source_states.detach().clone().requires_grad_()
    checkpointed_state = model.prepare(encoded, checkpointed_states)
    checkpointed_features = baseline_features.detach().clone().requires_grad_()
    checkpointed_route = model._route(
        checkpointed_state, encoded, receivers, checkpointed_features
    )
    model.activation_checkpointing = True
    checkpoint_calls = []
    original_checkpoint = adaptive_hyperedge_opening.checkpoint

    def record_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(True)
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(adaptive_hyperedge_opening, "checkpoint", record_checkpoint)
    checkpointed_context, _ = model._read_environment(
        checkpointed_state,
        encoded,
        receivers,
        checkpointed_features,
        checkpointed_route,
        include_diagnostics=False,
    )
    torch.testing.assert_close(
        checkpointed_context,
        baseline_context,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert checkpoint_calls
    checkpointed_context.square().mean().backward()
    torch.testing.assert_close(
        checkpointed_states.grad,
        baseline_state_gradient,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        checkpointed_features.grad,
        baseline_feature_gradient,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    for name, gradient in baseline_parameter_gradients.items():
        checkpointed_gradient = dict(model.named_parameters())[name].grad
        assert checkpointed_gradient is not None
        torch.testing.assert_close(
            checkpointed_gradient,
            gradient,
            rtol=1.0e-5,
            atol=1.0e-6,
        )


def test_batched_fine_executor_matches_scalar_reference_forward_and_gradients() -> None:
    """The K-batched path must preserve the old reader and route contract."""

    encoded, source_states = _encoded_case(seed=1513)
    model = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
        activation_checkpointing=False,
    )
    model.train()
    receivers = torch.rand(2, 7, 2) * torch.tensor([12.0, 6.0])
    receiver_features = torch.randn(2, 7, 6)

    def run_reader(reader):
        model.zero_grad(set_to_none=True)
        states = source_states.detach().clone().requires_grad_()
        features = receiver_features.detach().clone().requires_grad_()
        state = model.prepare(encoded, states)
        route = model._route(state, encoded, receivers, features)
        context, aux = reader(
            state,
            encoded,
            receivers,
            features,
            route,
            include_diagnostics=True,
        )
        context.square().mean().backward()
        parameter_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return context.detach(), aux, states.grad.detach().clone(), features.grad.detach().clone(), parameter_gradients

    scalar = run_reader(model._read_environment_scalar_reference)
    batched = run_reader(model._read_environment)
    torch.testing.assert_close(batched[0], scalar[0], rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(batched[2], scalar[2], rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(batched[3], scalar[3], rtol=2.0e-5, atol=2.0e-6)
    assert batched[4].keys() == scalar[4].keys()
    for name in scalar[4]:
        torch.testing.assert_close(batched[4][name], scalar[4][name], rtol=2.0e-5, atol=2.0e-6)

    for key in (
        "group_control_adaptive_environment_p",
        "group_control_adaptive_environment_alpha",
        "group_control_adaptive_opening_blend",
        "group_control_adaptive_query_count_per_group",
        "group_control_adaptive_source_count_per_group",
        "group_control_adaptive_fine_group_rows",
        "group_control_adaptive_fine_group_rows_logical",
        "group_control_adaptive_fine_group_rows_forward",
        "group_control_adaptive_fine_group_rows_padded",
    ):
        torch.testing.assert_close(batched[1][key], scalar[1][key], rtol=0.0, atol=0.0)
    assert batched[1]["group_control_adaptive_fine_batched_block_calls"].item() == 1.0
    assert batched[1]["group_control_adaptive_fine_scalar_block_calls"].item() > 1.0
    torch.testing.assert_close(
        batched[1]["group_control_adaptive_fine_gemm_launch_reduction"],
        batched[1]["group_control_adaptive_fine_scalar_gemm_launches"]
        - batched[1]["group_control_adaptive_fine_batched_gemm_launches"],
        rtol=0.0,
        atol=0.0,
    )


def test_batched_fine_executor_reduces_scalar_block_calls_to_one(monkeypatch) -> None:
    encoded, module_states = _encoded_case(seed=1514)
    model = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
    )
    model.eval()
    state = model.prepare(encoded, module_states)
    receivers = torch.rand(2, 8, 2) * torch.tensor([12.0, 6.0])
    receiver_features = torch.randn(2, 8, 6)
    route = model._route(state, encoded, receivers, receiver_features)

    scalar_calls: list[bool] = []
    scalar_block = model._fine_group_response_block

    def record_scalar(*args):
        scalar_calls.append(True)
        return scalar_block(*args)

    monkeypatch.setattr(model, "_fine_group_response_block", record_scalar)
    _, scalar_aux = model._read_environment_scalar_reference(
        state,
        encoded,
        receivers,
        receiver_features,
        route,
        include_diagnostics=True,
    )

    batched_calls: list[bool] = []
    batched_block = model._fine_group_response_batched_block

    def record_batched(*args):
        batched_calls.append(True)
        return batched_block(*args)

    monkeypatch.setattr(model, "_fine_group_response_batched_block", record_batched)
    _, batched_aux = model._read_environment(
        state,
        encoded,
        receivers,
        receiver_features,
        route,
        include_diagnostics=True,
    )
    assert len(scalar_calls) == int(scalar_aux["group_control_adaptive_fine_scalar_block_calls"])
    assert len(scalar_calls) > 1
    assert len(batched_calls) == 1
    assert batched_aux["group_control_adaptive_fine_batched_block_calls"].item() == 1.0
    assert batched_aux["group_control_adaptive_fine_batched_gemm_launches"].item() == 1.0
    assert batched_aux["group_control_adaptive_fine_gemm_launch_reduction"].item() == len(scalar_calls) - 1


def test_hybrid_executor_boundary_selects_batched_at_128_and_scalar_above(
    monkeypatch,
) -> None:
    encoded, module_states = _encoded_case(seed=1515)
    model = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
    )
    model.eval()
    state = model.prepare(encoded, module_states)

    batched_calls: list[bool] = []
    batched_block = model._fine_group_response_batched_block

    def record_batched(*args):
        batched_calls.append(True)
        return batched_block(*args)

    scalar_calls: list[bool] = []
    scalar_block = model._fine_group_response_block

    def record_scalar(*args):
        scalar_calls.append(True)
        return scalar_block(*args)

    monkeypatch.setattr(model, "_fine_group_response_batched_block", record_batched)
    monkeypatch.setattr(model, "_fine_group_response_block", record_scalar)

    def read(query_count: int):
        receivers = torch.rand(2, query_count, 2) * torch.tensor([12.0, 6.0])
        features = torch.randn(2, query_count, 6)
        route = model._route(state, encoded, receivers, features)
        return model._read_environment(
            state,
            encoded,
            receivers,
            features,
            route,
            include_diagnostics=True,
        )

    _, at_boundary = read(128)
    assert len(batched_calls) == 1
    assert not scalar_calls
    assert bool((at_boundary["group_control_adaptive_fine_executor_selected"] == 1.0).all())
    assert at_boundary["group_control_adaptive_fine_executor_batch_limit"].item() == 128.0
    torch.testing.assert_close(
        at_boundary["group_control_adaptive_batched_fine_rows_padded"],
        at_boundary["group_control_adaptive_batched_fine_rows"]
        - at_boundary["group_control_adaptive_fine_rows_logical"],
        rtol=0.0,
        atol=0.0,
    )

    batched_calls.clear()
    scalar_calls.clear()
    receivers_above = torch.rand(2, 129, 2) * torch.tensor([12.0, 6.0])
    features_above = torch.randn(2, 129, 6)
    route_above = model._route(state, encoded, receivers_above, features_above)
    selected, selected_aux = model._read_environment(
        state,
        encoded,
        receivers_above,
        features_above,
        route_above,
        include_diagnostics=True,
    )
    assert not batched_calls
    assert len(scalar_calls) > 0
    assert bool((selected_aux["group_control_adaptive_fine_executor_selected"] == 0.0).all())
    assert selected_aux["group_control_adaptive_fine_batched_block_calls"].item() == 0.0
    assert selected_aux["group_control_adaptive_batched_fine_rows"].item() == 0.0
    assert selected_aux["group_control_adaptive_batched_fine_rows_padded"].item() == 0.0

    reference, reference_aux = model._read_environment_scalar_reference(
        state,
        encoded,
        receivers_above,
        features_above,
        route_above,
        include_diagnostics=True,
    )
    torch.testing.assert_close(selected, reference, rtol=2.0e-5, atol=2.0e-6)
    for key in (
        "group_control_adaptive_environment_p",
        "group_control_adaptive_environment_alpha",
        "group_control_adaptive_opening_blend",
        "group_control_adaptive_fine_group_rows_forward",
    ):
        torch.testing.assert_close(selected_aux[key], reference_aux[key], rtol=0.0, atol=0.0)


def _encoded_case(seed: int = 1503) -> tuple[EncodedInterfaceCase, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    batch, modules, environments, hidden = 2, 4, 9, 16
    extent = torch.tensor([12.0, 6.0])
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden, generator=generator),
        env_tokens=torch.randn(batch, environments, hidden, generator=generator),
        global_token=torch.randn(batch, hidden, generator=generator),
        module_centers=torch.rand(batch, modules, 2, generator=generator) * extent,
        env_coords=torch.rand(batch, environments, 2, generator=generator) * extent,
        module_present=torch.tensor(
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        ),
        module_features=torch.randn(batch, modules, 3, generator=generator),
        env_features=None,
        env_weights=torch.rand(batch, environments, generator=generator) + 0.2,
        coordinate_scale=extent,
    )
    states = torch.randn(batch, modules, hidden, generator=generator, requires_grad=True)
    return encoded, states


def _small_config() -> UnifiedForwardConfig:
    return UnifiedForwardConfig.from_dict(
        {
            "forward_architecture": "adaptive_hyperedge_opening_honf",
            "field_dim": 5,
            "hidden_dim": 16,
            "domain_length_x": 12.0,
            "domain_length_y": 6.0,
            "module_radius": 0.45,
            "coordinate_scale": [12.0, 6.0],
            "interface_model": {
                "message_hidden_dim": 8,
                "attention_heads": 2,
                "relative_fourier_frequencies": 2,
                "receiver_chunk_size": 3,
                "activation_checkpointing": False,
                "group_count": 12,
                "source_normalizer": "entmax15",
                "query_normalizer": "sparsemax",
                "module_temperature": 1.0,
                "environment_temperature": 1.0,
                "query_temperature": 1.0,
                "group_control_dim": 16,
                "environment_refinement_normalizer": "sparsemax",
            },
        }
    )


def test_group_major_backend_has_finite_limiting_paths_and_chunked_aux_shapes() -> None:
    encoded, module_states = _encoded_case()
    model = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
    )
    state = model.prepare(encoded, module_states)
    receivers = torch.rand(2, 7, 2) * torch.tensor([12.0, 6.0])
    receiver_features = torch.randn(2, 7, 6)
    route = model._route(state, encoded, receivers, receiver_features)
    context, aux = model._read_environment(
        state,
        encoded,
        receivers,
        receiver_features,
        route,
        include_diagnostics=True,
    )
    assert context.shape == (2, 7, 16)
    assert torch.isfinite(context).all()
    assert aux["group_control_adaptive_fine_group_rows"].shape == (2, 12)
    assert aux["group_control_adaptive_fine_rows_per_query"].shape == (2, 7)
    assert aux["group_control_environment_unique_pairs_per_query"].shape == (2, 7)
    assert aux["group_control_adaptive_coarse_group_response"].shape == (2, 7, 12, 16)
    assert torch.isfinite(aux["group_control_adaptive_opening_blend"]).all()
    loss = context.square().mean()
    loss.backward()
    assert module_states.grad is not None and torch.isfinite(module_states.grad).all()

    config = _small_config()
    assert config.forward_architecture == "adaptive_hyperedge_opening_honf"
    assert config.to_dict()["interface_model"]["environment_refinement_normalizer"] == "sparsemax"


def test_run1502_parameter_contract_strict_loads_into_run1503_backend() -> None:
    encoded, module_states = _encoded_case(seed=1504)
    run1502 = SparseIncidenceGroupControlPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
        environment_refinement_normalizer="sparsemax",
    )
    run1503 = AdaptiveHyperedgeOpeningPairwiseField(
        hidden_dim=16,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=2,
    )
    run1502.prepare(encoded, module_states)
    run1503.prepare(encoded, module_states.detach())
    run1503.load_state_dict(run1502.state_dict(), strict=True)


def test_run1502_small_core_state_dict_strict_loads_into_run1503_core() -> None:
    encoded, module_states = _encoded_case(seed=1506)
    run1503_config = _small_config()
    run1502_payload = run1503_config.to_dict()
    run1502_payload["forward_architecture"] = "sparse_incidence_group_control_honf"
    run1502_config = UnifiedForwardConfig.from_dict(run1502_payload)
    run1502_core = InterfaceFieldCore(run1502_config)
    run1503_core = InterfaceFieldCore(run1503_config)
    run1502_prepared = run1502_core.prepare(encoded, module_states)
    run1503_prepared = run1503_core.prepare(encoded, module_states.detach())
    receivers = torch.rand(2, 5, 2) * torch.tensor([12.0, 6.0])
    run1502_core.read(run1502_prepared, receivers)
    run1503_core.read(run1503_prepared, receivers)
    run1503_core.load_state_dict(run1502_core.state_dict(), strict=True)


def test_core_maps_off_and_maps_on_chunk_merge_keep_adaptive_shapes() -> None:
    encoded, module_states = _encoded_case(seed=1505)
    core = InterfaceFieldCore(_small_config())
    prepared = core.prepare(encoded, module_states)
    receivers = torch.rand(2, 8, 2) * torch.tensor([12.0, 6.0])
    maps_off = core.read(
        prepared,
        receivers,
        receiver_chunk_size=3,
        return_routing_maps=False,
    )
    assert maps_off.context.shape == (2, 8, 16)
    assert maps_off.interaction_aux["group_control_adaptive_fine_rows_per_query"].shape == (2, 8)
    maps_on = core.read(
        prepared,
        receivers,
        receiver_chunk_size=3,
        return_routing_maps=True,
    )
    assert maps_on.interaction_aux["group_control_adaptive_environment_p"].shape == (2, 8, 12)
    assert maps_on.interaction_aux["group_control_adaptive_fine_group_rows"].shape == (2, 12)

    whole = core.read(
        prepared,
        receivers,
        receiver_chunk_size=8,
        return_routing_maps=True,
    )
    torch.testing.assert_close(maps_on.context, whole.context, rtol=1.0e-5, atol=1.0e-6)
    for key in (
        "group_control_adaptive_query_count_per_group",
        "group_control_adaptive_fine_group_rows_logical",
        "group_control_adaptive_fine_rows_per_query",
    ):
        torch.testing.assert_close(
            maps_on.interaction_aux[key], whole.interaction_aux[key], rtol=0.0, atol=0.0
        )
    torch.testing.assert_close(
        maps_on.interaction_aux["group_control_adaptive_fine_rows_logical"],
        whole.interaction_aux["group_control_adaptive_fine_rows_logical"],
        rtol=0.0,
        atol=0.0,
    )
    # Logical support is chunk-invariant; the batched executor's common
    # padding and launch counts intentionally follow the receiver tiles.
    for key in (
        "group_control_adaptive_fine_scalar_block_calls",
        "group_control_adaptive_fine_batched_block_calls",
        "group_control_adaptive_fine_scalar_gemm_launches",
        "group_control_adaptive_fine_batched_gemm_launches",
        "group_control_adaptive_fine_gemm_launch_reduction",
        "group_control_adaptive_batched_fine_rows",
        "group_control_adaptive_batched_fine_rows_padded",
    ):
        assert key in maps_on.interaction_aux
        assert key in whole.interaction_aux
    assert maps_on.interaction_aux["group_control_adaptive_fine_batched_gemm_launches"].item() == 3.0
    assert whole.interaction_aux["group_control_adaptive_fine_batched_gemm_launches"].item() == 1.0
    assert maps_on.interaction_aux["group_control_adaptive_fine_scalar_gemm_launches"].item() >= whole.interaction_aux["group_control_adaptive_fine_scalar_gemm_launches"].item()
    for aux in (maps_on.interaction_aux, whole.interaction_aux):
        torch.testing.assert_close(
            aux["group_control_adaptive_fine_group_rows_forward"]
            - aux["group_control_adaptive_fine_group_rows_logical"],
            aux["group_control_adaptive_fine_group_rows_padded"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            aux["group_control_adaptive_fine_rows"],
            aux["group_control_adaptive_fine_group_rows"].sum(),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            aux["group_control_adaptive_fine_work_ratio"],
            aux["group_control_adaptive_fine_rows"]
            / aux["group_control_adaptive_full_rectangle_rows"].clamp_min(1.0),
            rtol=0.0,
            atol=0.0,
        )


def test_core_merge_preserves_mixed_hybrid_executor_selection() -> None:
    encoded, module_states = _encoded_case(seed=1516)
    core = InterfaceFieldCore(_small_config())
    prepared = core.prepare(encoded, module_states)
    receivers = torch.rand(2, 300, 2) * torch.tensor([12.0, 6.0])
    read = core.read(
        prepared,
        receivers,
        receiver_chunk_size=256,
        return_routing_maps=True,
    )
    selected = read.interaction_aux["group_control_adaptive_fine_executor_selected"]
    assert selected.shape == (2, 300)
    assert bool((selected[:, :256] == 0.0).all())
    assert bool((selected[:, 256:] == 1.0).all())
    assert read.interaction_aux["group_control_adaptive_fine_executor_batch_limit"].item() == 128.0
    # The merge must sum actual per-tile executor counters, while retaining
    # the per-query selection vector for mixed tile sizes.
    assert read.interaction_aux["group_control_adaptive_fine_batched_block_calls"].item() == 1.0
    assert read.interaction_aux["group_control_adaptive_fine_batched_gemm_launches"].item() == 1.0
