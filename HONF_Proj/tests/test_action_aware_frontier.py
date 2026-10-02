"""Action identity, risk sign, and full-fallback tests for the new selector."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from honf_forward_core.interface_fields.action_aware_frontier import (
    ActionAwareRiskHead,
    TypedActionSources,
    action_risk_loss,
    describe_frontier_action,
    describe_realized_plan,
    incumbent_role_log_limit,
    receiver_role_descriptors,
    select_action_by_risk,
    signed_log_role_risk,
)
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CaseLocalReceiverTree,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.input_cover_organizer import OrganizerScores


def _action() -> tuple[torch.Tensor, torch.Tensor, dict[str, TypedActionSources]]:
    packet_features = torch.tensor([[0.2, -0.1, 0.4, 0.8], [-0.3, 0.7, 0.1, 0.5]])
    packet_coordinates = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    sources = torch.tensor([[0.2, 0.0, 0.6, 0.1], [0.8, 0.3, 0.2, 0.4], [0.0, 0.9, 0.5, 0.2]])
    coordinates = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.4, 0.6]])
    measure = torch.tensor([0.2, 0.5, 0.3])
    score = torch.tensor([[0.3, 0.8, -0.2], [0.5, 0.1, 0.7]])
    return packet_features, packet_coordinates, {
        "MM": TypedActionSources(
            sources, coordinates, measure,
            torch.tensor([[1., 0., 1.], [0., 1., 1.]]), score,
        ),
        "QE": TypedActionSources(
            sources + 0.25, coordinates, measure,
            torch.tensor([[1., 1., 0.], [0., 1., 1.]]), score - 0.1,
        ),
    }


def test_action_summary_is_permutation_invariant_but_source_action_sensitive() -> None:
    torch.manual_seed(61)
    packet_features, packet_coordinates, actions = _action()
    original = describe_frontier_action(packet_features, packet_coordinates, actions)
    source_order = torch.tensor([2, 0, 1])
    packet_order = torch.tensor([1, 0])
    rearranged = {
        key: TypedActionSources(
            item.source_features[source_order], item.source_coordinates[source_order],
            item.source_measure[source_order], item.permission[packet_order][:, source_order],
            item.score[packet_order][:, source_order],
        ) for key, item in actions.items()
    }
    shuffled = describe_frontier_action(
        packet_features[packet_order], packet_coordinates[packet_order], rearranged,
    )
    torch.testing.assert_close(shuffled, original[packet_order])
    head = ActionAwareRiskHead(
        packet_feature_dim=original.shape[1], budget_dim=2, receiver_role_dim=3,
        hidden_dim=24,
    )
    budget = torch.tensor([0.9, 0.9])
    roles = torch.tensor([[1.0, 0.0, 0.2], [0.0, 1.0, 0.5]])
    base_risk = head(original, budget, roles)
    torch.testing.assert_close(head(shuffled, budget, roles), base_risk, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(head(original, budget, roles, nonredundant_k=2), base_risk)
    assert not torch.allclose(head(original, budget, roles, nonredundant_k=1), base_risk)
    changed = dict(actions)
    changed["QE"] = replace(
        actions["QE"], permission=torch.tensor([[0., 0., 1.], [1., 0., 0.]]),
    )
    changed_rows = describe_frontier_action(packet_features, packet_coordinates, changed)
    assert not torch.allclose(changed_rows, original)
    assert float((head(changed_rows, budget, roles) - base_risk).abs().max()) > 1e-6


def test_signed_risk_and_exact_work_selection_reject_untrained_or_unsafe_cuts() -> None:
    risks = signed_log_role_risk(
        torch.tensor([0.9, 1.2]), torch.tensor([1.0, 1.0]), torch.tensor([0.001, 0.001]),
    )
    assert risks[0] < 0 and risks[1] > 0
    predicted = torch.tensor([[0.03, 0.02], [0.01, 0.01], [0.02, 0.04], [0.0, 0.0]])
    exact_work = torch.tensor([0.7, 0.2, 0.5, 1.0])
    packet_counts = torch.tensor([2, 1, 4, 1])
    trained = torch.tensor([True, False, True, True])
    limit = torch.tensor([0.05, 0.05])
    saved_margin = torch.tensor([0.1, 0.1])
    historical_before = select_action_by_risk(
        predicted, exact_work, packet_counts, trained, limit,
        full_access_index=3, empirical_margin=saved_margin,
    )
    no_margin_diagnostic = select_action_by_risk(
        predicted, exact_work, packet_counts, trained, limit,
        full_access_index=3, empirical_margin=torch.zeros_like(saved_margin),
    )
    historical_after = select_action_by_risk(
        predicted, exact_work, packet_counts, trained, limit,
        full_access_index=3, empirical_margin=saved_margin,
    )
    assert no_margin_diagnostic.index == 2 and no_margin_diagnostic.safe_sparse_indices == (0, 2)
    assert historical_before.index == historical_after.index == 3
    assert historical_before.unsupported_at_budget and historical_after.unsupported_at_budget
    torch.testing.assert_close(saved_margin, torch.tensor([0.1, 0.1]), rtol=0, atol=0)
    fitted = torch.zeros_like(predicted, requires_grad=True)
    loss = action_risk_loss(fitted, predicted)
    loss.backward()
    assert torch.isfinite(loss) and fitted.grad is not None and bool(fitted.grad.abs().sum() > 0)


def test_incumbent_gate_keeps_numerical_floor_distinct_from_physical_allowance() -> None:
    incumbent = torch.tensor([1.0, 0.001])
    floor = torch.tensor([0.01, 0.01])
    allowance = torch.tensor([0.0, 0.0002])
    limit = incumbent_role_log_limit(incumbent, floor, allowance)
    measured = signed_log_role_risk(
        1.10 * incumbent + allowance, incumbent, floor,
    )
    torch.testing.assert_close(limit, measured)
    assert limit[1] > 0
    assert not torch.allclose(limit, torch.full_like(limit, torch.log(torch.tensor(1.10))))


def test_realized_plan_adapter_uses_actual_typed_permissions_and_role_geometry() -> None:
    coordinates = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    universe = ReceiverAnchorUniverse(
        coordinates, torch.ones(4), torch.tensor([0, 0, 1, 1]), torch.ones(2),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=7, min_leaf_anchors=1)
    node_count = len(tree.nodes)
    scores = OrganizerScores(
        split_logits=torch.zeros(node_count),
        module_logits=torch.zeros(node_count, 2),
        environment_logits=torch.zeros(node_count, 2),
        mechanism_logits={
            "MM": torch.randn(node_count, 2), "QE": torch.randn(node_count, 2),
        },
        node_embeddings=torch.randn(node_count, 4),
        module_embeddings=torch.randn(2, 4),
        environment_embeddings=torch.randn(2, 4),
        budget_vector=torch.ones(5),
    )
    encoded = SimpleNamespace(
        module_centers=coordinates[:2][None],
        module_present=torch.ones(1, 2),
        env_coords=coordinates[2:][None],
        env_weights=torch.ones(1, 2),
    )
    full = MechanismPlan.full_access(tree, torch.ones(2), 2)
    packet_rows = describe_realized_plan(scores, full, encoded, (0,))
    assert packet_rows.shape[0] == 1
    role_rows = receiver_role_descriptors(tree, 2)
    assert role_rows.shape == (2, 8)
    sparse = MechanismPlan(
        tree, full.split_gates, torch.ones(2), 2,
        {"MM": torch.tensor([[1., 0.]]).expand(node_count, -1),
         "QE": torch.tensor([[0., 1.]]).expand(node_count, -1)},
    )
    sparse_rows = describe_realized_plan(scores, sparse, encoded, (0,))
    assert not torch.allclose(packet_rows, sparse_rows)
