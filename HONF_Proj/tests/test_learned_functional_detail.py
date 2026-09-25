"""Focused algebra and gradient contracts for task-trained detail retention."""

from __future__ import annotations

import torch

from honf_forward_core.organization.functional_fusion_tree import build_tree_transform
from honf_forward_core.organization.learned_functional_detail import (
    LearnedFunctionalDetailController,
    expected_class_count,
)


def _unequal_tree() -> torch.Tensor:
    # Deliberately scrambled: the largest split is 4:1, and row order must not
    # affect the canonical tree or the Haar/product equivalence.
    membership = torch.zeros((4, 5), dtype=torch.bool)
    membership[0, [0, 1, 2, 3, 4]] = True
    membership[1, [2, 3]] = True
    membership[2, [0, 1, 2, 3]] = True
    membership[3, [0, 1]] = True
    return membership


def _controller() -> LearnedFunctionalDetailController:
    return LearnedFunctionalDetailController(_unequal_tree(), 7, hidden_dim=8)


def test_haar_transform_matches_projection_product_with_nested_zero_nodes() -> None:
    controller = _controller()
    active = torch.ones((3, 5), dtype=torch.bool)
    torch.manual_seed(1503)
    retention = torch.rand(3, 4, dtype=torch.float64)
    retention[0, 3] = 0.0  # canonical root closes
    retention[1, 2] = 0.0  # the {0,1,2,3} node closes
    retention[1, 0] = 0.0  # nested {0,1} closure is hidden by its parent
    retention[2, 1] = 0.0  # {2,3} closes independently
    haar_transform = controller.transform_from_retention(retention)
    product_transform = build_tree_transform(
        retention, controller.node_membership
    )
    torch.testing.assert_close(haar_transform, product_transform, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(
        haar_transform[0], torch.full((5, 5), 0.2, dtype=torch.float64), atol=2e-14, rtol=2e-14
    )
    # The nested node's own detail is zero, while the sibling remains open.
    torch.testing.assert_close(haar_transform[1, 0], haar_transform[1, 1], atol=2e-14, rtol=2e-14)
    assert not torch.allclose(haar_transform[1, 0], haar_transform[1, 4])
    torch.testing.assert_close(haar_transform[2, 2], haar_transform[2, 3], atol=2e-14, rtol=2e-14)
    # A tiny positive retention stays mathematically open; packing never uses
    # numerical rank or a fuzzy transform-equality threshold.
    with torch.no_grad():
        controller.network[-1].weight.zero_()
        controller.network[-1].bias.fill_(-2.3)
    tiny_plan = controller.plan(
        torch.zeros(1, 4, 7),
        torch.ones(1, 5, dtype=torch.bool),
        torch.ones(1, 5),
        stochastic_mask=torch.zeros(1, dtype=torch.bool),
    )
    assert bool((tiny_plan.retention > 0.0).all())
    assert not bool(tiny_plan.closed_nodes.any())
    assert tiny_plan.actual_R.tolist() == [5]


def test_expected_r_matches_sampled_tree_partitions_for_full_and_masked_cases() -> None:
    controller = _controller()
    membership = controller.node_membership
    refs = controller.child_refs
    probability = torch.tensor(
        [[0.25, 0.45, 0.70, 0.85], [0.25, 0.45, 0.70, 0.85]],
        dtype=torch.float64,
    )
    active = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 1, 1]], dtype=torch.bool
    )
    closure_eligible = torch.tensor(
        [[1, 1, 1, 1], [0, 1, 0, 0]], dtype=torch.bool
    )
    expected = expected_class_count(
        probability, active, membership, refs, closure_eligible
    )
    generator = torch.Generator().manual_seed(15031)
    samples = 30_000
    draws = torch.rand((samples, 2, 4), generator=generator, dtype=torch.float64)
    p_eff = torch.where(closure_eligible[None, :, :], probability[None, :, :], 1.0)
    closed = (draws > p_eff) & closure_eligible[None, :, :]
    widths = controller._actual_class_count(
        closed.reshape(samples * 2, 4), active.repeat(samples, 1)
    ).to(torch.float64).reshape(samples, 2)
    torch.testing.assert_close(
        widths.mean(dim=0), expected, atol=0.025, rtol=0.0
    )
    assert expected[0] < 5.0
    # In the masked case, no node containing the inactive leaf may close.
    assert expected[1] < 4.0


def test_expected_complexity_updates_controller_without_source_feature_gradient() -> None:
    controller = _controller()
    descriptors = torch.randn(2, 4, 7, requires_grad=True)
    active = torch.ones(2, 5, dtype=torch.bool)
    proposal_mass = torch.ones(2, 5, requires_grad=True)
    plan = controller.plan(
        descriptors,
        active,
        proposal_mass,
        stochastic_mask=torch.zeros(2, dtype=torch.bool),
    )
    cost_gradients = torch.autograd.grad(
        plan.expected_complexity.sum(),
        (descriptors, proposal_mass, *controller.parameters()),
        allow_unused=True,
    )
    assert cost_gradients[0] is None
    assert cost_gradients[1] is None
    assert any(
        gradient is not None and torch.count_nonzero(gradient).item() > 0
        for gradient in cost_gradients[2:]
    )

    task_loss = (plan.transform * torch.randn_like(plan.transform)).sum()
    task_gradients = torch.autograd.grad(
        task_loss,
        (descriptors, *controller.parameters()),
        allow_unused=True,
    )
    assert task_gradients[0] is not None
    assert any(
        gradient is not None and torch.count_nonzero(gradient).item() > 0
        for gradient in task_gradients[1:]
    )


def test_partial_mass_eligibility_fades_without_exact_closure() -> None:
    controller = _controller()
    descriptors = torch.zeros(1, 4, 7)
    active = torch.ones(1, 5, dtype=torch.bool)
    mass = torch.ones(1, 5)
    mass[0, 0] = 1.5e-6
    with torch.no_grad():
        controller.network[-1].weight.zero_()
        controller.network[-1].bias.fill_(-8.0)
    plan = controller.plan(
        descriptors,
        active,
        mass,
        stochastic_mask=torch.zeros(1, dtype=torch.bool),
        ramp=1.0,
    )
    # Low mass smoothly attenuates some contrasts, yet every affected node has
    # positive residual retention and is excluded from exact compaction.
    assert not bool(plan.closed_nodes[0, 0])
    assert plan.expected_R[0] >= 1.0
    packed = controller.pack(plan, active)
    assert packed.class_id.shape == (1, 5)
    assert int(packed.multiplicity.sum()) == 5
