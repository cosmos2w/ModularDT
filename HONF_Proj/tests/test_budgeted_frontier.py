"""Focused contracts for budgeted cuts, physical-pair P, and inverse export."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CaseLocalReceiverTree,
    InteractionContext,
    MechanismPlan,
    ReceiverAnchorUniverse,
    compile_mechanism_execution_view,
)
from honf_forward_core.interface_fields.budgeted_frontier import (
    BudgetConditionedDirectPairScorer,
    FrontierUtilityHead,
    FrontierUtilityPrediction,
    canonical_pair_catalog,
    direct_pair_weights_at_threshold,
    enumerate_frontier_cuts,
    frontier_from_paths,
    frontier_paths,
    normalized_frontier_distortion_targets,
    normalized_frontier_work_targets,
    project_direct_pair_budget_by_fraction,
    project_unique_pair_budget,
    select_frontier_by_predictions,
    split_gates_for_frontier,
)
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.native_direct_pair import hard_value_soft_direct_forward
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _case() -> EncodedInterfaceCase:
    batch, modules, environments, hidden = 1, 3, 4, 8
    generator = torch.Generator().manual_seed(411)
    scale = torch.tensor([4.0, 3.0])
    return EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden, generator=generator),
        env_tokens=torch.randn(batch, environments, hidden, generator=generator),
        global_token=torch.randn(batch, hidden, generator=generator),
        module_centers=torch.tensor([[[0.4, 0.5], [2.8, 2.0], [0.0, 0.0]]]),
        env_coords=torch.tensor([[[0.1, 0.2], [1.2, 0.4], [2.3, 1.8], [3.7, 2.6]]]),
        module_present=torch.tensor([[1.0, 1.0, 0.0]]),
        module_features=torch.randn(batch, modules, 2, generator=generator),
        env_features=torch.randn(batch, environments, 2, generator=generator),
        env_weights=torch.tensor([[0.4, 0.8, 1.2, 0.6]]),
        coordinate_scale=scale.reshape(1, 1, 2),
    )


def _tree(case: EncodedInterfaceCase) -> CaseLocalReceiverTree:
    coordinates = torch.cat((case.env_coords[0], case.module_centers[0, :2]), dim=0)
    weights = torch.cat((case.env_weights[0], torch.ones(2)))
    roles = torch.tensor([0, 0, 0, 0, 1, 1])
    universe = ReceiverAnchorUniverse(coordinates, weights, roles, case.coordinate_scale[0, 0])
    return CaseLocalReceiverTree.build(
        universe, max_nodes=15, min_leaf_anchors=1, overlap_fraction=0.08
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_factorized_direct_scorer_matches_dense_pair_values_and_gradients(dtype: torch.dtype) -> None:
    """The faster first layer preserves the saved direct scorer's computation."""

    torch.manual_seed(718)
    scorer = BudgetConditionedDirectPairScorer(
        receiver_feature_dim=7, source_feature_dim=8, hidden_dim=12, source_chunk_size=2,
        factorized_first_layer=True,
    ).to(dtype=dtype)
    receivers = torch.randn(5, 7, dtype=dtype, requires_grad=True)
    sources = torch.randn(4, 8, dtype=dtype, requires_grad=True)
    receiver_coordinates = torch.randn(5, 3, dtype=dtype, requires_grad=True)
    source_coordinates = torch.randn(4, 3, dtype=dtype, requires_grad=True)
    parameters = tuple(scorer.scorers["QE"].parameters())

    factorized = scorer(
        receivers, sources, receiver_coordinates, source_coordinates,
        mechanism="QE", budget_fraction=0.9,
    )
    relative = receiver_coordinates[:, None, :] - source_coordinates[None, :, :]
    pair_features = torch.cat((
        receivers[:, None, :].expand(-1, sources.shape[0], -1),
        sources[None, :, :].expand(receivers.shape[0], -1, -1),
        scorer._spatial_summary(relative),
        receivers.new_full((receivers.shape[0], sources.shape[0], 1), 0.9),
    ), dim=-1)
    dense = scorer.scorers["QE"](pair_features).squeeze(-1)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(factorized, dense, rtol=tolerance, atol=tolerance)

    inputs = (receivers, sources, receiver_coordinates, source_coordinates, *parameters)
    factorized_gradients = torch.autograd.grad(factorized.square().sum(), inputs, retain_graph=True)
    dense_gradients = torch.autograd.grad(dense.square().sum(), inputs)
    for actual, expected in zip(factorized_gradients, dense_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_depth_three_frontiers_have_26_complete_cuts_and_realized_gates() -> None:
    coordinates = torch.arange(8, dtype=torch.float32)[:, None]
    universe = ReceiverAnchorUniverse(
        coordinates,
        torch.ones(8),
        torch.zeros(8, dtype=torch.long),
        torch.tensor([8.0]),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=15, min_leaf_anchors=1)
    cuts = enumerate_frontier_cuts(tree, max_depth=3)
    assert len(tree.nodes) == 15
    assert len(cuts) == 26
    for cut in cuts:
        gates = split_gates_for_frontier(tree, cut, max_depth=3)
        active = (tree.access(universe.coordinates, gates) > 0.0).any(dim=0)
        assert tuple(sorted(torch.nonzero(active).flatten().tolist())) == tuple(sorted(cut))


def test_frontier_paths_remap_complete_cut_when_current_tree_ids_shift() -> None:
    def make_tree(*, expand_leftmost_leaf: bool) -> SimpleNamespace:
        nodes: list[SimpleNamespace] = []

        def add(depth: int, path: str) -> int:
            index = len(nodes)
            node = SimpleNamespace(left=None, right=None)
            nodes.append(node)
            if depth < 3 or (expand_leftmost_leaf and path == "LLL"):
                node.left = add(depth + 1, path + "L")
                node.right = add(depth + 1, path + "R")
            return index

        add(0, "")
        return SimpleNamespace(nodes=nodes)

    source = make_tree(expand_leftmost_leaf=False)
    moved = make_tree(expand_leftmost_leaf=True)
    source_cut = enumerate_frontier_cuts(source, max_depth=3)[-1]
    paths = frontier_paths(source, source_cut, max_depth=3)
    moved_cut = frontier_from_paths(moved, paths, max_depth=3)
    assert len(paths) == len(moved_cut) == 8
    assert moved_cut != source_cut
    assert moved_cut in enumerate_frontier_cuts(moved, max_depth=3)
    assert frontier_paths(moved, moved_cut, max_depth=3) == paths
    with pytest.raises(ValueError, match="unavailable"):
        frontier_from_paths(SimpleNamespace(nodes=[SimpleNamespace(left=None, right=None)]), paths)


def test_canonical_pair_projection_handles_ties_caps_diagonal_and_empty_routes() -> None:
    case = _case()
    tree = _tree(case)
    gates = torch.zeros(len(tree.nodes))
    gates[0] = 1.0
    scores = torch.ones((len(tree.nodes), 3))
    catalog = canonical_pair_catalog(case, tree, "MM")
    assert not catalog.pair_validity[0, 0]
    assert not catalog.pair_validity[1, 1]
    assert not catalog.pair_validity[2].any()

    full = project_unique_pair_budget(
        scores,
        tree,
        gates,
        budget_fraction=1.0,
        source_validity=catalog.source_validity,
        receiver_validity=catalog.receiver_validity,
        pair_validity=catalog.pair_validity,
        receiver_weights=catalog.receiver_weights,
        receiver_coordinates=catalog.receiver_coordinates,
    )
    assert full.selected_unique_pairs == 2
    assert full.full_unique_pairs == 2
    assert full.achieved_work == full.full_access_work == 2.0
    assert not full.sparse_success

    empty_cap = project_unique_pair_budget(
        scores,
        tree,
        gates,
        budget_fraction=0.0,
        source_validity=catalog.source_validity,
        receiver_validity=catalog.receiver_validity,
        pair_validity=catalog.pair_validity,
        receiver_weights=catalog.receiver_weights,
        receiver_coordinates=catalog.receiver_coordinates,
    )
    assert empty_cap.membership.sum() == 0
    assert torch.isposinf(empty_cap.threshold)
    assert empty_cap.achieved_work == 0.0
    assert not empty_cap.sparse_success

    grouped_scores = torch.zeros_like(scores)
    active_node = int(torch.nonzero((tree.access(catalog.receiver_coordinates, gates) > 0.0).any(dim=0), as_tuple=False)[0])
    grouped_scores[active_node] = torch.tensor([0.0, 1.0, 2.0])
    nontrivial_grouped = project_unique_pair_budget(
        grouped_scores,
        tree,
        gates,
        budget_fraction=0.5,
        source_validity=catalog.source_validity,
        receiver_validity=catalog.receiver_validity,
        pair_validity=catalog.pair_validity,
        receiver_weights=catalog.receiver_weights,
        receiver_coordinates=catalog.receiver_coordinates,
    )
    assert 0 < nontrivial_grouped.selected_unique_pairs < nontrivial_grouped.full_unique_pairs
    assert nontrivial_grouped.sparse_success

    no_reached_nodes = project_unique_pair_budget(
        scores,
        tree,
        torch.zeros_like(gates),
        budget_fraction=0.5,
        source_validity=catalog.source_validity,
        receiver_validity=catalog.receiver_validity,
        pair_validity=catalog.pair_validity,
        receiver_weights=catalog.receiver_weights,
        receiver_coordinates=catalog.receiver_coordinates,
    )
    assert no_reached_nodes.full_access_work > 0.0
    assert no_reached_nodes.selected_unique_pairs == 0
    assert not no_reached_nodes.sparse_success

    no_pairs = project_unique_pair_budget(
        scores,
        tree,
        gates,
        budget_fraction=0.4,
        source_validity=catalog.source_validity,
        receiver_validity=catalog.receiver_validity,
        pair_validity=torch.zeros_like(catalog.pair_validity),
        receiver_weights=catalog.receiver_weights,
        receiver_coordinates=catalog.receiver_coordinates,
    )
    assert no_pairs.full_unique_pairs == 0
    assert no_pairs.achieved_work == 0.0
    assert not no_pairs.sparse_success

    tied = project_direct_pair_budget_by_fraction(
        torch.ones((2, 3)),
        budget_fraction=0.4,
        eligible_pairs=torch.ones((2, 3), dtype=torch.bool),
    )
    assert tied.selected_unique_pairs == 0  # equal-score support is atomic
    assert not tied.sparse_success
    weights = direct_pair_weights_at_threshold(
        torch.zeros((2, 3), requires_grad=True),
        tied,
        eligible_pairs=torch.ones((2, 3), dtype=torch.bool),
    )
    assert weights.hard_weights.sum() == 0
    assert torch.isfinite(weights.soft_threshold)
    weights.soft_weights.sum().backward()
    assert weights.soft_weights.grad_fn is not None

    direct_full = project_direct_pair_budget_by_fraction(
        torch.zeros((2, 3)), budget_fraction=1.0
    )
    assert direct_full.selected_unique_pairs == 6
    assert torch.isneginf(direct_full.threshold)
    direct_empty = project_direct_pair_budget_by_fraction(
        torch.zeros((2, 3)), budget_fraction=0.0
    )
    assert direct_empty.selected_unique_pairs == 0
    assert torch.isposinf(direct_empty.threshold)
    assert not direct_empty.sparse_success
    nontrivial_direct = project_direct_pair_budget_by_fraction(
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        budget_fraction=0.5,
        eligible_pairs=torch.ones((2, 2), dtype=torch.bool),
    )
    assert 0 < nontrivial_direct.selected_unique_pairs < nontrivial_direct.full_unique_pairs
    assert nontrivial_direct.sparse_success
    no_eligible_direct = project_direct_pair_budget_by_fraction(
        torch.zeros((2, 3)),
        budget_fraction=0.5,
        eligible_pairs=torch.zeros((2, 3), dtype=torch.bool),
    )
    assert no_eligible_direct.full_unique_pairs == 0
    assert not no_eligible_direct.sparse_success


def test_budget_and_stage_c_targets_are_dimensionless_and_fit_first_order() -> None:
    head = FrontierUtilityHead(hidden_dim=12, budget_dim=5, role_count=2)
    embeddings = torch.randn(7, 12)
    cuts = ((0,), (1, 2), (3, 4, 5, 6))
    budget = torch.tensor([0.25, 1.0, 1.0, 0.5, 1.0])
    target_distortion = torch.tensor([[0.05, 0.1], [0.6, 0.3], [0.9, 0.8]])
    prediction = head(embeddings, cuts, budget)
    work_target = normalized_frontier_work_targets(torch.tensor([8.0, 4.0, 2.0]), 8.0)
    assert torch.all((prediction.predicted_work_fraction >= 0) & (prediction.predicted_work_fraction <= 1))
    assert torch.equal(work_target, torch.tensor([1.0, 0.5, 0.25]))

    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    before = None
    after = None
    for _ in range(30):
        optimizer.zero_grad()
        current = head(embeddings, cuts, budget)
        loss = (current.role_distortion - target_distortion).square().mean() + (
            current.predicted_work_fraction - work_target
        ).square().mean()
        if before is None:
            before = float(loss.detach())
        loss.backward()
        optimizer.step()
        after = float(loss.detach())
    assert after is not None and before is not None and after < before

    raw = torch.tensor([[2.0, 5.0], [8.0, 5.5], [3.0, 9.0], [1.0, 4.0]])
    baseline = torch.tensor([2.0, 5.0])
    distortion = normalized_frontier_distortion_targets(raw, baseline)
    assert torch.equal(distortion[0], torch.zeros(2))
    assert torch.allclose(distortion[1], torch.tensor([3.0, 0.1]))
    assert torch.equal(distortion[3], torch.zeros(2))
    targets = FrontierUtilityPrediction(
        cuts=cuts,
        role_distortion=torch.tensor([[0.1, 0.1], [0.6, 0.2], [0.2, 0.2]]),
        predicted_work_fraction=torch.tensor([0.8, 0.4, 0.3]),
    )
    adequate = select_frontier_by_predictions(
        targets, role_tolerance=torch.tensor([0.25, 0.25]), packet_counts=torch.tensor([1, 2, 4])
    )
    assert adequate.selected_frontier == cuts[2]
    unsupported = select_frontier_by_predictions(
        targets, role_tolerance=torch.tensor([0.05, 0.05]), packet_counts=torch.tensor([1, 2, 4])
    )
    assert unsupported.unsupported_at_budget
    assert unsupported.selected_frontier is None
    assert unsupported.least_risk_frontier == cuts[0]


def _native_core() -> tuple[InterfaceFieldCore, BatchData]:
    config = UnifiedForwardConfig(
        forward_architecture="dense_pairwise_field",
        field_dim=2,
        spatial_dim=2,
        coordinate_scale=[4.0, 3.0],
        hidden_dim=8,
        dropout=0.0,
        geometry_mode="nonperiodic",
        boundary_feature_mode="none",
        interface_model=InterfaceFieldConfig(
            message_hidden_dim=8,
            attention_heads=2,
            coarse_latent_count=2,
            coarse_blocks=1,
            relative_fourier_frequencies=2,
            receiver_chunk_size=8,
            activation_checkpointing=False,
        ),
    )
    batch = BatchData(
        module_centers=torch.tensor([[[0.4, 0.5], [2.8, 2.0], [0.0, 0.0]]]),
        module_present=torch.tensor([[1.0, 1.0, 0.0]]),
        module_features=torch.randn(1, 3, 2, generator=torch.Generator().manual_seed(41)),
        global_context=torch.randn(1, 4, generator=torch.Generator().manual_seed(42)),
        query_xy=torch.tensor([[[0.7, 0.8], [3.0, 2.2]]]),
        query_time=None,
        target_field=torch.zeros(1, 2, 2),
        case_name="direct-pair-unit-test",
        metadata={},
        env_coords=torch.tensor([[[0.1, 0.2], [1.2, 0.4], [2.3, 1.8], [3.7, 2.6]]]),
        env_features=torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(43)),
        query_features=torch.randn(1, 2, 3, generator=torch.Generator().manual_seed(44)),
        env_weights=torch.tensor([[0.4, 0.8, 1.2, 0.6]]),
    )
    return InterfaceFieldCore(config), batch


def test_direct_pair_plan_is_exact_native_p_and_has_no_grouped_or_stale_alias() -> None:
    core, batch = _native_core()
    core.backend.set_cover_mode("external")
    encoded = core.encode_case(batch)
    tree = core.backend.build_case_trees(encoded)[0]
    base = MechanismPlan.full_access(tree, encoded.module_present[0], int(encoded.env_coords.shape[1]))
    module_valid = encoded.module_present[0] > 0.5
    env_valid = encoded.env_weights[0] > 0.0
    query = batch.query_xy[0]
    query_valid = torch.ones(query.shape[0], dtype=torch.bool)
    score_parameters = torch.nn.Parameter(torch.tensor([-0.3, 0.2, -0.1, 0.1, -0.2]))
    hard_plan = base
    soft_plan = base
    route_records = (
        ("MM", encoded.module_centers[0], module_valid[:, None] & module_valid[None, :] & ~torch.eye(3, dtype=torch.bool), module_valid),
        ("ME", encoded.module_centers[0], module_valid[:, None] & env_valid[None, :], module_valid),
        ("EM", encoded.env_coords[0], env_valid[:, None] & module_valid[None, :], env_valid),
        ("QM", query, query_valid[:, None] & module_valid[None, :], query_valid),
        ("QE", query, query_valid[:, None] & env_valid[None, :], query_valid),
    )
    for route_index, (mechanism, receiver_coords, eligible, receiver_validity) in enumerate(route_records):
        eligible_float = eligible.to(torch.float32)
        hard_support = eligible & (
            torch.arange(eligible.numel()).reshape_as(eligible).remainder(2) == 0
        )
        soft_weights = torch.sigmoid(score_parameters[route_index]) * eligible_float
        hard_plan = hard_plan.with_direct_pair_access(
            mechanism, receiver_coords, hard_support.float(), receiver_validity=receiver_validity
        )
        soft_plan = soft_plan.with_direct_pair_access(
            mechanism, receiver_coords, soft_weights, receiver_validity=receiver_validity
        )
    assert hard_plan.permission_status("QM") == "direct_pair_permission"
    assert "QM" not in hard_plan.explicit_bypass_keys
    assert not hard_plan.is_full_access(mechanisms=("QM", "QE"))
    with pytest.raises(ValueError, match="cannot be represented as node memberships"):
        hard_plan.permission_matrix("QM")
    with pytest.raises(ValueError, match="compiled node memberships"):
        compile_mechanism_execution_view(
            hard_plan,
            module_present=encoded.module_present[0],
            environment_count=int(encoded.env_coords.shape[1]),
        )

    states = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(46), requires_grad=True)
    query_coords = batch.query_xy.detach().clone().requires_grad_(True)
    result = hard_value_soft_direct_forward(
        core,
        encoded,
        states,
        (hard_plan,),
        (soft_plan,),
        query_coords,
        batch.query_features,
        receiver_chunk_size=8,
    )
    torch.testing.assert_close(result.prediction, result.hard_prediction, rtol=0.0, atol=0.0)
    result.soft_prediction.sum().backward(retain_graph=True)
    assert score_parameters.grad is not None and bool((score_parameters.grad.abs() > 0).any())
    assert states.grad is None and query_coords.grad is None
    assert all(parameter.grad is None for parameter in core.parameters())

    # The exact native read rejects a changed panel rather than silently
    # replaying direct scores at new candidate coordinates.
    prepared = core.prepare(encoded, states.detach(), fixed_cover_plans=(hard_plan,))
    with pytest.raises(ValueError, match="different receiver panel/order"):
        core.decode_queries(
            prepared,
            query_coords.detach() + 0.01,
            batch.query_features,
            receiver_chunk_size=8,
        )

    changed_centers = encoded.module_centers.clone()
    changed_centers[0, 0, 0] += 0.05
    stale_candidate = replace(encoded, module_centers=changed_centers)
    with pytest.raises(ValueError, match="stale for current receiver anchor coordinates"):
        core.backend.rebind_fixed_plans(stale_candidate, (hard_plan,))


def test_dynamic_direct_policy_scores_current_panels_across_phases() -> None:
    core, batch = _native_core()
    core.backend.set_cover_mode("external")
    encoded = core.encode_case(batch)
    tree = core.backend.build_case_trees(encoded)[0]
    env_valid = encoded.env_weights[0] > 0.0
    anchor_coordinates = tree.universe.coordinates
    scale = encoded.coordinate_scale[0, 0]
    anchor_features = core.receiver_fourier(anchor_coordinates / scale)

    class RecordingScorer(BudgetConditionedDirectPairScorer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.receiver_panels: list[torch.Tensor] = []
            self.grad_modes: list[bool] = []

        def forward(self, receiver_features, source_features, receiver_coordinates,
                    source_coordinates, *, mechanism, budget_fraction):
            self.receiver_panels.append(receiver_coordinates.detach().clone())
            self.grad_modes.append(torch.is_grad_enabled())
            return super().forward(
                receiver_features, source_features, receiver_coordinates,
                source_coordinates, mechanism=mechanism,
                budget_fraction=budget_fraction,
            )

    scorer = RecordingScorer(
        receiver_feature_dim=int(anchor_features.shape[-1]),
        source_feature_dim=int(encoded.env_tokens.shape[-1]),
        hidden_dim=12,
        source_chunk_size=2,
    )
    canonical_scores = scorer(
        anchor_features,
        encoded.env_tokens[0],
        anchor_coordinates,
        encoded.env_coords[0],
        mechanism="QE",
        budget_fraction=0.5,
    )
    canonical_eligible = (tree.universe.weights > 0.0)[:, None] & env_valid[None, :]
    projection = project_direct_pair_budget_by_fraction(
        canonical_scores,
        budget_fraction=0.5,
        eligible_pairs=canonical_eligible,
        receiver_weights=tree.universe.weights,
    )
    threshold_weights = direct_pair_weights_at_threshold(
        canonical_scores,
        projection,
        eligible_pairs=canonical_eligible,
    )

    def policy_plan(*, hard: bool) -> MechanismPlan:
        plan = MechanismPlan.full_access(
            tree, encoded.module_present[0], int(encoded.env_coords.shape[1])
        )
        for phase in ("P0", "P1"):
            plan = plan.with_direct_pair_policy(
                "QE",
                scorer,
                source_coordinates=encoded.env_coords[0],
                source_features=encoded.env_tokens[0],
                # The policy binding is Boolean: every positive native
                # quadrature measure is valid, including values below 0.5.
                source_validity=env_valid,
                hard_threshold=projection.threshold,
                soft_threshold=threshold_weights.soft_threshold,
                budget_fraction=0.5,
                hard=hard,
                phase=phase,
            )
        return plan

    hard_plan, soft_plan = policy_plan(hard=True), policy_plan(hard=False)
    scorer.grad_modes.clear()
    hard_access = hard_plan.direct_pair_policy_for("QE", phase="P0").access_for(
        "QE", anchor_coordinates, anchor_features,
    )
    assert scorer.grad_modes == [False]
    assert not hard_access.requires_grad
    scorer.grad_modes.clear()
    soft_access = soft_plan.direct_pair_policy_for("QE", phase="P0").access_for(
        "QE", anchor_coordinates, anchor_features,
    )
    assert scorer.grad_modes == [True]
    assert soft_access.requires_grad
    scorer.grad_modes.clear()
    states = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(73), requires_grad=True)
    query0 = torch.tensor([[[0.3, 0.4], [1.7, 1.1]]])
    result = hard_value_soft_direct_forward(
        core,
        encoded,
        states,
        (hard_plan,),
        (soft_plan,),
        query0,
        batch.query_features,
        receiver_chunk_size=1,
        interaction_context=InteractionContext(phase="P0"),
    )
    result.soft_prediction.sum().backward()
    assert any(parameter.grad is not None and bool((parameter.grad.abs() > 0).any()) for parameter in scorer.parameters())
    assert False in scorer.grad_modes and True in scorer.grad_modes
    assert states.grad is None
    first_phase_panels = [panel for panel in scorer.receiver_panels if panel.shape[0] == 1]
    assert any(torch.allclose(panel, query0[0, :1]) for panel in first_phase_panels)
    assert any(torch.allclose(panel, query0[0, 1:]) for panel in first_phase_panels)

    # The same frozen policy is rebound across native phases even after the
    # intermediate module state changes. New receiver rows are scored live.
    states_p1 = states.detach() + 0.25
    prepared_p1 = core.prepare(
        encoded,
        states_p1,
        fixed_cover_plans=(soft_plan,),
        interaction_context=InteractionContext(phase="P1"),
    )
    query1 = torch.tensor([[[0.9, 2.4], [3.2, 1.7], [2.1, 0.3]]])
    query_features_p1 = torch.cat(
        (batch.query_features, batch.query_features[:, :1]), dim=1
    )
    count_before = len(scorer.receiver_panels)
    core.decode_queries(
        prepared_p1,
        query1,
        query_features_p1,
        receiver_chunk_size=2,
        interaction_context=InteractionContext(phase="P1"),
    )
    current_panels = scorer.receiver_panels[count_before:]
    assert any(torch.allclose(panel, query1[0, :2]) for panel in current_panels)
    assert any(torch.allclose(panel, query1[0, 2:]) for panel in current_panels)


def test_organizer_preserves_legacy_logits_and_budget_trial_bypasses_unbudgeted_routes() -> None:
    from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer

    case = _case()
    tree = _tree(case)
    organizer = InputOnlyCoverOrganizer(
        state_dim=8,
        module_feature_dim=2,
        environment_feature_dim=2,
        hidden_dim=12,
        role_count=4,
    )
    prep = {
        "module_states": case.module_tokens,
        "environment_states": case.env_tokens,
        "global_state": case.global_token,
    }
    default = organizer.score_cases(case, prep, (tree,))
    explicit_full_budget = organizer.score_cases(case, prep, (tree,), budgets={})
    for left, right in zip(default[0].mechanism_logits.values(), explicit_full_budget[0].mechanism_logits.values(), strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    plans = organizer.plans_from_scores(
        explicit_full_budget,
        case,
        (tree,),
        hard=True,
        budget_fractions={"QE": 0.5, "MM": 0.5},
    )
    plan = plans[0]
    assert plan.permission_status("QE") == "mechanism_permission"
    for mechanism in ("ME", "EM", "QM"):
        assert plan.permission_status(mechanism) == "full_access_bypass_missing_key"
    compiled = compile_mechanism_execution_view(
        plan,
        module_present=case.module_present[0],
        environment_count=int(case.env_coords.shape[1]),
    )
    assert compiled.permission_key_missing("ME")


def test_organizer_case_scale_broadcasts_a_single_physical_extent() -> None:
    from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer

    shared = torch.tensor([[[12.0, 6.0]]])
    per_case = torch.tensor([[[12.0, 6.0]], [[8.0, 4.0]]])
    torch.testing.assert_close(InputOnlyCoverOrganizer._case_scale(shared, 1), shared[0, 0])
    torch.testing.assert_close(InputOnlyCoverOrganizer._case_scale(shared[:, 0], 1), shared[0, 0])
    torch.testing.assert_close(InputOnlyCoverOrganizer._case_scale(per_case, 1), per_case[1, 0])
