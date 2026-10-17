from __future__ import annotations

import io

import pytest
import torch

from honf_forward_core.native_retrieval.budget_state import BudgetDualState, BudgetSchedule
from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore
from honf_forward_core.native_retrieval.budgeted_graph import (
    BudgetedGraphParameters,
    build_budgeted_native_graph,
    calibrate_membership_bias,
    dense_st_gate,
    expected_final_work,
    final_support,
)
from honf_forward_core.native_retrieval.seed_dispatch import SeedDispatchParameters, dispatch_native_seed
from tests.test_content_hypergraph_v2_core import _core, _scene


def _native_graph(*, present: torch.Tensor | None = None, dtype: torch.dtype = torch.float64):
    scores = torch.tensor(
        [[
            [[0.0, 0.0, -4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, -4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]],
        dtype=dtype,
    )
    if present is None:
        present = torch.ones((1, 3), dtype=torch.bool)
    ids = torch.tensor([[90, 10, 30]], dtype=torch.long)
    coordinates = torch.tensor([[[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]]], dtype=dtype)
    lengths = torch.ones_like(coordinates)
    parameters = BudgetedGraphParameters(2, dtype=dtype)
    graph = build_budgeted_native_graph(
        scores,
        present,
        ids,
        coordinates,
        parameters,
        module_characteristic_lengths=lengths,
        deterministic=True,
        straight_through=True,
    )
    return graph, parameters


def test_membership_is_one_head_calibrated_binary_map_with_self_presence_semantics():
    present = torch.tensor([[True, True, False]])
    graph, parameters = _native_graph(present=present)
    assert float(parameters.scale) == pytest.approx(1.0)
    assert set(dict(parameters.named_parameters())) == {"head_logits", "raw_scale", "bias"}
    assert graph.membership_hard.dtype == torch.bool
    assert torch.equal(graph.membership_hard[0, 0], torch.tensor([True, True, False]))
    assert graph.membership_probabilities[0, 0, 0] == 1.0
    assert not graph.membership_hard[0, 2].any()
    assert not graph.membership_hard[0, :, 2].any()
    assert torch.equal(graph.membership_straight_through, graph.membership_hard.to(torch.float64))
    assert torch.equal(graph.member_valid.sum(-1), graph.membership_hard.sum(-1))
    assert torch.equal(graph.member_indices[0, 0, :1], torch.tensor([0]))

    # The two heads prefer opposite donors; their shared convex mixture admits
    # neither donor at a zero logit threshold, unlike an OR of headwise supports.
    all_present, parameters = _native_graph()
    assert not all_present.membership_hard[0, 0, 1]
    assert not all_present.membership_hard[0, 0, 2]
    assert all_present.membership_probabilities[0, 0, 1].item() == pytest.approx(
        torch.sigmoid(torch.tensor(-2.0)).item()
    )
    with torch.no_grad():
        parameters.raw_scale.fill_(-100.0)
    assert float(parameters.scale) >= 0.25
    with torch.no_grad():
        parameters.raw_scale.fill_(100.0)
    assert float(parameters.scale) <= 4.0


def test_zero_information_measure_does_not_remove_a_present_physical_source_and_empty_axes_work():
    # The graph takes presence rather than positive measure as physical eligibility;
    # a zero-measure present source remains a seed/self member and can enter a signed law.
    graph, _ = _native_graph()
    assert graph.seed_valid[0, 0]
    assert graph.membership_hard[0, 0, 0]
    assert graph.membership_hard[0, 1, 1]

    empty_scores = torch.empty((1, 2, 0, 0), dtype=torch.float64)
    empty_mask = torch.empty((1, 0), dtype=torch.bool)
    empty_ids = torch.empty((1, 0), dtype=torch.long)
    empty_geometry = torch.empty((1, 0, 2), dtype=torch.float64)
    empty_graph = build_budgeted_native_graph(
        empty_scores,
        empty_mask,
        empty_ids,
        empty_geometry,
        BudgetedGraphParameters(2, dtype=torch.float64),
        deterministic=True,
    )
    assert empty_graph.membership_hard.shape == (1, 0, 0)
    assert empty_graph.member_indices.shape == (1, 0, 0)
    assert final_support(
        torch.full((1, 2), -1, dtype=torch.long),
        empty_graph,
        torch.empty((1, 2, 0), dtype=torch.bool),
    ).shape == (1, 2, 0)


def test_membership_bias_calibration_uses_only_present_nonself_scores():
    scores = torch.tensor(
        [[[[0.0, -1.0, -2.0], [-1.0, 0.0, -3.0], [-2.0, -2.0, 0.0]]]],
        dtype=torch.float64,
    )
    present = torch.tensor([[True, True, False]])
    bias = calibrate_membership_bias(scores, present, target=0.8)
    # The present nonself scores have row-relative values 0 and 0, so the calibrated
    # logistic-noise inclusion probability is the requested target in both directions.
    assert torch.sigmoid(bias).item() == pytest.approx(0.8, abs=1.0e-12)
    with pytest.raises(ValueError, match="at least one"):
        calibrate_membership_bias(scores, torch.tensor([[True, False, False]]), target=0.8)


def test_dispatch_ties_use_geometry_not_ids_and_indistinguishable_ties_are_flagged():
    parameters = SeedDispatchParameters(2, dtype=torch.float64)
    valid = torch.ones((1, 3), dtype=torch.bool)
    raw = torch.tensor([[[[2.0, 2.0, 0.0]], [[2.0, 2.0, 0.0]]]], dtype=torch.float64)
    geometry = torch.tensor([[[2.0, 0.0, 1.0], [1.0, 0.0, 1.0], [3.0, 0.0, 1.0]]], dtype=torch.float64)
    ids = torch.tensor([[90, 10, 30]])
    plan = dispatch_native_seed(raw, valid, geometry, parameters, deterministic=True, seed_ids=ids)
    assert plan.dispatch_seed.item() == 1
    assert plan.dispatch_seed_ids.item() == 10
    assert plan.tied_seed_mask[0, 0, :2].all()
    assert not plan.tie_indeterminate.any()

    permutation = torch.tensor([2, 0, 1])
    permuted = dispatch_native_seed(
        raw[..., permutation],
        valid[:, permutation],
        geometry[:, permutation],
        parameters,
        deterministic=True,
        seed_ids=ids[:, permutation],
    )
    assert permuted.dispatch_seed_ids.item() == plan.dispatch_seed_ids.item()

    duplicate_geometry = geometry.clone()
    duplicate_geometry[:, 1] = duplicate_geometry[:, 0]
    ambiguous = dispatch_native_seed(
        raw,
        valid,
        duplicate_geometry,
        parameters,
        deterministic=True,
        seed_ids=ids,
    )
    assert ambiguous.tie_indeterminate.item()
    assert ambiguous.dispatch_seed.item() == -1
    assert ambiguous.tied_seed_mask[0, 0, :2].all()
    assert ambiguous.hard.sum() == 0

    empty = dispatch_native_seed(
        torch.empty((1, 2, 4, 0), dtype=torch.float64),
        torch.empty((1, 0), dtype=torch.bool),
        torch.empty((1, 0, 3), dtype=torch.float64),
        parameters,
        deterministic=True,
    )
    assert empty.dispatch_seed.shape == (1, 4)
    assert (empty.dispatch_seed == -1).all()
    assert empty.probabilities.shape == (1, 4, 0)


def test_expected_work_matches_monte_carlo_and_protection_is_union_not_duplicate_work():
    memberships = torch.tensor(
        [[[1.0, 0.5, 0.0], [0.0, 1.0, 0.2], [0.4, 0.0, 1.0]]], dtype=torch.float64
    )
    dispatch = torch.tensor([[[0.25, 0.25, 0.5], [0.4, 0.4, 0.2]]], dtype=torch.float64)
    protection = torch.tensor([[[False, True, False], [True, False, False]]])
    query_mask = torch.ones((1, 2), dtype=torch.bool)
    present = torch.ones((1, 3), dtype=torch.bool)
    work = expected_final_work(dispatch, memberships, protection, query_mask, present)
    assert torch.equal(work.expected_support[0, 0, 1], torch.tensor(1.0, dtype=torch.float64))
    assert work.case_floor_rows.item() == 2.0
    assert work.case_removable_rows.item() == 4.0
    assert work.target_case_native_fraction(0.5).item() == pytest.approx(2.0 / 3.0)

    torch.manual_seed(28)
    draws = 20_000
    chosen_seeds = torch.stack(
        [torch.multinomial(dispatch[0, q].expand(draws, -1), 1).squeeze(-1) for q in range(2)],
        dim=1,
    )
    shared_membership_draws = torch.rand((draws, 3, 3), dtype=torch.float64) < memberships[0]
    query_index = torch.arange(3).view(1, 1, 3).expand(draws, 2, 3)
    selected = shared_membership_draws[
        torch.arange(draws).view(-1, 1, 1), chosen_seeds[:, :, None], query_index
    ]
    sample_support = selected | protection[0].unsqueeze(0)
    sampled_rows_per_query = sample_support.sum(dim=-1).to(torch.float64).mean(dim=0)
    mc_max_abs_error = (sampled_rows_per_query - work.expected_rows_per_query[0]).abs().max()
    assert mc_max_abs_error.item() < 0.025
    sample_fraction = sample_support.sum(dim=(-1, -2)).to(torch.float64).mean() / 6.0
    assert sample_fraction.item() == pytest.approx(work.case_native_fraction[0].item(), abs=0.015)

    # Query padding contributes no work and no removable denominator.
    padded = expected_final_work(
        dispatch,
        memberships,
        protection,
        torch.tensor([[True, False]]),
        present,
    )
    assert padded.case_possible_rows.item() == 3.0
    assert padded.case_removable_rows.item() == 2.0
    assert padded.expected_rows_per_query[0, 1].item() == 0.0


def test_dense_router_gate_has_binary_forward_support_and_live_surrogate_gradient():
    graph, membership_parameters = _native_graph()
    query_scores = torch.tensor([[[[1.0, 0.0, -1.0]], [[-1.0, 0.0, 1.0]]]], dtype=torch.float64)
    dispatch_parameters = SeedDispatchParameters(2, dtype=torch.float64)
    dispatch = dispatch_native_seed(
        query_scores,
        graph.seed_valid,
        graph.source_geometry,
        dispatch_parameters,
        gumbel_noise=torch.zeros((1, 1, 3), dtype=torch.float64),
        deterministic=False,
        straight_through=True,
    )
    protection = torch.tensor([[[False, True, False]]])
    gate = dense_st_gate(
        dispatch.straight_through,
        graph.membership_straight_through,
        protection,
        module_present=graph.module_present,
    )
    hard = final_support(dispatch.dispatch_seed, graph, protection).to(torch.float64)
    torch.testing.assert_close(gate, hard, rtol=0.0, atol=0.0)
    values = torch.tensor([[[0.3, -1.1, 2.0]]], dtype=torch.float64)
    gradient = torch.autograd.grad((gate * values).sum(), (dispatch_parameters.head_logits, membership_parameters.bias))
    assert gradient[0].abs().sum() > 0
    assert gradient[1].abs().sum() > 0


def test_schedule_boundaries_and_dual_state_are_serializable_and_update_every_ten_steps():
    at_501 = BudgetSchedule.state_at_epoch(501)
    at_700 = BudgetSchedule.state_at_epoch(700)
    at_701 = BudgetSchedule.state_at_epoch(701)
    at_1500 = BudgetSchedule.state_at_epoch(1500)
    at_2500 = BudgetSchedule.state_at_epoch(2500)
    assert at_501["rho"] == 0.98 and at_501["temperature"] == 1.0
    assert at_501["stochastic_expectation_exact"]
    assert at_700["rho"] == 0.85 and at_700["temperature"] == 0.5
    assert at_701["rho"] == 0.85 and not at_701["stochastic_expectation_exact"]
    assert at_701["dense_router_step"] is None
    assert BudgetSchedule.state_at_epoch(701, optimizer_macro_step=3)["dense_router_step"] is False
    assert BudgetSchedule.state_at_epoch(701, optimizer_macro_step=4)["dense_router_step"] is True
    assert at_1500["rho"] == at_2500["rho"] == 0.60
    assert at_1500["noise_scale"] == 0.0 and at_1500["deterministic"]

    dual = BudgetDualState(lambda0=0.2, rng_identity="fixed-test-key", dtype=torch.float64)
    dual.apply_schedule(at_701)
    residual = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    assert dual.penalty_residual(residual).item() == pytest.approx(0.021)
    assert torch.autograd.grad(dual.penalty_residual(residual), residual)[0].item() > 0.0
    for macro_step in range(1, 10):
        assert not dual.observe_residual(torch.tensor(0.1, dtype=torch.float64), macro_step)
    assert dual.dual_lambda.item() == pytest.approx(0.2)
    assert dual.observe_residual(torch.tensor(0.1, dtype=torch.float64), 10)
    assert dual.dual_lambda.item() == pytest.approx(0.22)

    serialized = io.BytesIO()
    torch.save(dual.state_dict(), serialized)
    serialized.seek(0)
    restored = BudgetDualState(lambda0=0.0, dtype=torch.float64)
    restored.load_state_dict(torch.load(serialized, weights_only=False))
    assert restored.rng_identity == "fixed-test-key"
    assert restored.phase == "hard_selected_with_dense_router_steps"
    assert restored.dual_lambda.item() == pytest.approx(dual.dual_lambda.item())


def test_graph_and_dispatch_traces_reject_in_place_mutation():
    graph, _ = _native_graph()
    with pytest.raises(ValueError, match="only -1"):
        final_support(
            torch.full((1, 1), -2, dtype=torch.long),
            graph,
            torch.zeros((1, 1, 3), dtype=torch.bool),
        )
    with torch.no_grad():
        graph.membership_hard[0, 0, 0] = False
    with pytest.raises(ValueError, match="changed in place"):
        graph.assert_current()

    raw = torch.tensor([[[[2.0, 1.0, 0.0]]]], dtype=torch.float64)
    params = SeedDispatchParameters(1, dtype=torch.float64)
    plan = dispatch_native_seed(
        raw,
        torch.ones((1, 3), dtype=torch.bool),
        graph.source_geometry,
        params,
        deterministic=True,
    )
    with torch.no_grad():
        plan.probabilities[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="changed in place"):
        plan.assert_current()


def test_budgeted_constructor_rejects_direct_environment_value_execution():
    parent = _core("A2-D")
    with pytest.raises(ValueError, match="separately controlled pilot family"):
        BudgetedNativeCore.from_a2d(parent, environment_value_reader=True)


def test_counter_noise_is_rng_neutral_and_permutation_invariant():
    model = BudgetedNativeCore.from_a2d(_core("A2-D"), routing_seed=73).train()
    model.budgeted_begin_update(701, 11)
    scene = _scene(1)
    query_ids = torch.tensor([[811, 407, 102]], dtype=torch.long)
    module_ids = scene.module_ids
    before = torch.random.get_rng_state()
    noise = model._uniform(query_ids, module_ids, scene, "dispatch")
    assert torch.equal(before, torch.random.get_rng_state())

    query_permutation = torch.tensor([2, 0, 1])
    module_permutation = torch.tensor([1, 2, 0])
    permuted = model._uniform(query_ids[:, query_permutation], module_ids[:, module_permutation], scene,
                              "dispatch")
    torch.testing.assert_close(permuted, noise[:, query_permutation][:, :, module_permutation],
                               rtol=0.0, atol=0.0)

    graph_noise = model._uniform(module_ids, module_ids, scene, "membership")
    graph_permuted = model._uniform(module_ids[:, module_permutation], module_ids[:, module_permutation],
                                    scene, "membership")
    torch.testing.assert_close(graph_permuted,
                               graph_noise[:, module_permutation][:, :, module_permutation],
                               rtol=0.0, atol=0.0)
