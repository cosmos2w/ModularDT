from __future__ import annotations

import torch

from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CaseLocalReceiverTree,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.input_cover_organizer import (
    InputOnlyCoverOrganizer,
    OrganizerScores,
    masked_binary_logit_loss,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _tree(device: torch.device, offset: float = 0.0) -> CaseLocalReceiverTree:
    universe = ReceiverAnchorUniverse(
        coordinates=torch.tensor(
            [[offset + 0.0, 0.0, 0.0], [offset + 0.0, 0.0, 1.0],
             [offset + 10.0, 0.0, 0.0], [offset + 10.0, 0.0, 1.0]],
            device=device,
        ),
        weights=torch.ones(4, device=device),
        roles=torch.tensor([0, 1, 0, 1], device=device),
        coordinate_scale=torch.tensor([10.0, 1.0, 1.0], device=device),
    )
    return CaseLocalReceiverTree.build(universe, max_nodes=3, min_leaf_anchors=1)


def _encoded(device: torch.device | None = None) -> EncodedInterfaceCase:
    device = torch.device("cpu") if device is None else device
    return EncodedInterfaceCase(
        module_tokens=torch.randn(2, 3, 8, device=device),
        env_tokens=torch.randn(2, 4, 8, device=device),
        global_token=torch.randn(2, 8, device=device),
        module_centers=torch.tensor(
            [
                [[0.0, 0.0, 0.5], [10.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.5], [12.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
            ],
            device=device,
        ),
        env_coords=torch.tensor(
            [
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 1.0], [11.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [11.0, 0.0, 1.0], [13.0, 0.0, 1.0]],
            ],
            device=device,
        ),
        module_present=torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], device=device),
        module_features=torch.ones(2, 3, 2, device=device),
        env_features=torch.randn(2, 4, 3, device=device),
        env_weights=torch.ones(2, 4, device=device),
        coordinate_scale=torch.tensor([[10.0, 1.0, 1.0], [10.0, 1.0, 1.0]], device=device),
    )


def test_input_only_organizer_returns_hard_case_plans_and_masks_padding() -> None:
    encoded = _encoded()
    trees = (_tree(torch.device("cpu")), _tree(torch.device("cpu"), offset=2.0))
    organizer = InputOnlyCoverOrganizer(
        state_dim=8,
        module_feature_dim=2,
        environment_feature_dim=3,
        hidden_dim=16,
    ).eval()
    prepared_state = {
        "module_states": encoded.module_tokens,
        "environment_states": encoded.env_tokens,
        "global_state": encoded.global_token,
        "target_field": object(),
        "case_id": "must not be inspected",
        "test_array": object(),
    }

    plans = organizer.plan_cases(encoded, prepared_state, trees)

    assert len(plans) == 2
    for plan in plans:
        assert isinstance(plan, MechanismPlan)
        assert set(plan.split_gates.unique().tolist()) <= {0.0, 1.0}
        for mechanism in ("MM", "ME", "EM", "QM", "QE"):
            assert set(plan.permission_matrix(mechanism).unique().tolist()) <= {0.0, 1.0}
    assert not bool(plans[0].module_membership[:, 2].any())
    assert not bool(plans[1].module_membership[:, 2].any())


def test_input_only_organizer_supports_two_dimensional_cases() -> None:
    device = torch.device("cpu")
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(1, 2, 4, device=device),
        env_tokens=torch.randn(1, 3, 4, device=device),
        global_token=torch.randn(1, 4, device=device),
        module_centers=torch.tensor([[[0.0, 0.5], [2.0, 0.5]]], device=device),
        env_coords=torch.tensor([[[-0.5, 0.0], [1.0, 1.0], [2.5, 0.0]]], device=device),
        module_present=torch.ones(1, 2, device=device),
        module_features=torch.randn(1, 2, 2, device=device),
        env_features=torch.randn(1, 3, 3, device=device),
        env_weights=torch.ones(1, 3, device=device),
        coordinate_scale=torch.tensor([[3.0, 1.0]], device=device),
    )
    tree = CaseLocalReceiverTree.build(
        ReceiverAnchorUniverse(
            coordinates=torch.tensor(
                [[0.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]], device=device
            ),
            weights=torch.ones(4, device=device),
            roles=torch.tensor([0, 1, 0, 1], device=device),
            coordinate_scale=torch.tensor([3.0, 1.0], device=device),
        ),
        max_nodes=3,
        min_leaf_anchors=1,
    )
    organizer = InputOnlyCoverOrganizer(
        state_dim=4, module_feature_dim=2, environment_feature_dim=3, hidden_dim=12
    ).eval()

    plans = organizer.plan_cases(encoded, {}, (tree,))

    assert len(plans) == 1 and isinstance(plans[0], MechanismPlan)
    assert plans[0].permission_matrix("QM").shape == (3, 2)
    assert plans[0].permission_matrix("QE").shape == (3, 3)


def test_organizer_soft_plans_keep_prediction_gradients_and_hard_k_is_measured_from_plan() -> None:
    encoded = _encoded()
    trees = (_tree(torch.device("cpu")), _tree(torch.device("cpu"), offset=2.0))
    organizer = InputOnlyCoverOrganizer(
        state_dim=8,
        module_feature_dim=2,
        environment_feature_dim=3,
        hidden_dim=16,
    ).train()
    scores = organizer.score_cases(encoded, {}, trees)
    soft_plans = organizer.plans_from_scores(scores, encoded, trees, hard=False)
    assert all(bool(((plan.split_gates > 0) & (plan.split_gates < 1)).any()) for plan in soft_plans)
    soft_plans[0].split_gates.sum().backward()
    assert any(parameter.grad is not None for parameter in organizer.parameters())

    hard_plans = organizer.plans_from_scores(scores, encoded, trees, hard=True)
    measured_k = tuple(
        plan.access(plan.tree.universe.coordinates).active_groups_on_anchors for plan in hard_plans
    )
    assert all(isinstance(value, int) and value >= 0 for value in measured_k)


def test_typed_quadrature_atom_split_preserves_hard_source_support() -> None:
    encoded_batch = _encoded()
    encoded = EncodedInterfaceCase(**{
        name: (
            value[:1]
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 2
            else value
        )
        for name, value in encoded_batch.__dict__.items()
    })
    encoded = EncodedInterfaceCase(
        **{
            **encoded.__dict__,
            "env_weights": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        }
    )
    split = EncodedInterfaceCase(
        **{
            **encoded.__dict__,
            "env_tokens": torch.cat((encoded.env_tokens[:, :-1], encoded.env_tokens[:, -1:], encoded.env_tokens[:, -1:]), dim=1),
            "env_coords": torch.cat((encoded.env_coords[:, :-1], encoded.env_coords[:, -1:], encoded.env_coords[:, -1:]), dim=1),
            "env_features": torch.cat((encoded.env_features[:, :-1], encoded.env_features[:, -1:], encoded.env_features[:, -1:]), dim=1),
            "env_weights": torch.tensor([[1.0, 2.0, 3.0, 2.0, 2.0]]),
        }
    )
    tree = _tree(torch.device("cpu"))
    organizer = InputOnlyCoverOrganizer(
        state_dim=8,
        module_feature_dim=2,
        environment_feature_dim=3,
        hidden_dim=16,
        quadrature_invariant_source_measure=True,
    ).eval()

    # Make ME/QE select only a source whose aggregated measure exceeds the
    # support threshold. Splitting its weight into two identical source slots
    # must duplicate that decision without changing other source decisions.
    with torch.no_grad():
        for parameter in organizer.source_encoder.parameters():
            parameter.zero_()
        source_first = organizer.source_encoder[0]
        source_first.weight[0, organizer.source_feature_dim + organizer.state_dim] = 2.0
        organizer.source_encoder[2].weight[0, 0] = 2.0
        for mechanism in ("ME", "QE"):
            scorer = organizer.pair_scorers[mechanism]
            for parameter in scorer.parameters():
                parameter.zero_()
            scorer[0].weight[0, 16] = 1.0
            scorer[2].weight[0, 0] = 1.0
            scorer[4].weight[0, 0] = 1.0
            scorer[4].bias[0] = -1.0

    full_plan = organizer.plan_cases(encoded, {}, (tree,))[0]
    split_plan = organizer.plan_cases(split, {}, (tree,))[0]
    for mechanism in ("ME", "QE"):
        full_support = full_plan.permission_matrix(mechanism)
        split_support = split_plan.permission_matrix(mechanism)
        assert bool(full_support[:, -1].all())
        assert not bool(full_support[:, :-1].any())
        torch.testing.assert_close(split_support[:, :-2], full_support[:, :-1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(split_support[:, -2], full_support[:, -1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(split_support[:, -1], full_support[:, -1], rtol=0.0, atol=0.0)


def test_masked_logit_loss_ignores_unknown_oracle_decisions() -> None:
    logits = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
    targets = torch.tensor([1.0, float("nan"), 0.0])
    observed = torch.tensor([True, False, True])
    loss = masked_binary_logit_loss(logits, targets, observed)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad[0].abs() > 0
    assert logits.grad[1] == 0
    assert logits.grad[2].abs() > 0


def test_class_balanced_masked_logit_loss_gives_both_observed_classes_equal_weight() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
    targets = torch.tensor([1.0, 1.0, 1.0, 0.0])
    observed = torch.tensor([True, True, True, True])

    ordinary = masked_binary_logit_loss(logits, targets, observed)
    balanced = masked_binary_logit_loss(logits, targets, observed, balance_classes=True)

    assert torch.allclose(balanced, torch.tensor(0.69314718), atol=1.0e-6)
    assert torch.allclose(ordinary, balanced)

    selected_targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    selected_logits = torch.tensor([0.0, 0.0, 0.0, 4.0, -4.0])
    selected_observed = torch.ones_like(selected_targets, dtype=torch.bool)
    class_balanced = masked_binary_logit_loss(
        selected_logits, selected_targets, selected_observed, balance_classes=True
    )
    class_means = torch.nn.functional.binary_cross_entropy_with_logits(
        selected_logits, selected_targets, reduction="none"
    )
    expected = 0.5 * (class_means[:3].mean() + class_means[3:].mean())
    assert torch.allclose(class_balanced, expected)


def test_hard_cover_conversion_applies_the_declared_threshold_to_logits() -> None:
    encoded = _encoded()
    trees = (_tree(torch.device("cpu")), _tree(torch.device("cpu"), offset=2.0))
    organizer = InputOnlyCoverOrganizer(
        state_dim=8,
        module_feature_dim=2,
        environment_feature_dim=3,
        hidden_dim=16,
        split_threshold=0.75,
        source_threshold=0.25,
    )
    scores = (
        OrganizerScores(
            split_logits=torch.tensor([2.0, 1.0, -1.0]),
            module_logits=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            environment_logits=torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]),
        ),
    )
    plan = organizer.plans_from_scores(scores * 2, encoded, trees, hard=True)[0]
    assert plan.split_gates.tolist() == [1.0, 0.0, 0.0]
    assert plan.module_membership[0, :2].tolist() == [1.0, 1.0]
    assert not bool(plan.module_membership[:, 2].any())
    assert plan.environment_membership[0].tolist() == [1.0, 1.0, 1.0, 1.0]
