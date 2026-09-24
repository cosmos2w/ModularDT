"""Algebra contracts for Run 1503-v4 continuous functional coalescence."""

from __future__ import annotations

import torch

from honf_forward_core.interface_fields.routing_index.sparse_projection import (
    masked_sparsemax,
)
from honf_forward_core.organization.functional_fusion_tree import (
    FunctionalProbeCatalogue,
    FunctionalProbeFamily,
    build_source_action_grams,
    build_tree_transform,
    combined_post_transform_discrepancy,
    contracted_logits,
    deterministic_complete_linkage,
    node_contraction,
    node_source_action_scores,
    pack_exact_closed_subtrees,
)
from honf_forward_core.organization.group_fusion import weighted_sparsemax


def _catalogue(
    groups: int,
    *,
    batch: int = 1,
    probes: int = 1,
    device: torch.device | None = None,
) -> FunctionalProbeCatalogue:
    coordinates = torch.zeros((batch, probes, 2), dtype=torch.float32, device=device)
    valid = torch.ones((batch, probes), dtype=torch.bool, device=device)
    weights = torch.ones((batch, probes), dtype=torch.float32, device=device)
    invalid = torch.zeros_like(valid)
    zero_weights = torch.zeros_like(weights)
    ports = FunctionalProbeFamily(coordinates, invalid, zero_weights)
    outside = FunctionalProbeFamily(coordinates, invalid, zero_weights)
    environment = FunctionalProbeFamily(coordinates, valid, weights)
    del groups  # K is defined by the finite raw-logit bank, not the coordinates.
    return FunctionalProbeCatalogue(
        phase="P1",
        physical_ports=ports,
        outside_temperature=outside,
        environment=environment,
    )


def test_source_action_grams_are_differentiable_four_block_psd_inputs() -> None:
    torch.manual_seed(15030401)
    module_a = torch.rand((2, 4, 3), requires_grad=True)
    module_measure = torch.rand((2, 4)) + 0.2
    environment_a = torch.rand((2, 5, 3), requires_grad=True)
    environment_measure = torch.rand((2, 5)) + 0.2
    group_control = torch.rand((2, 3, 4), requires_grad=True)
    module_gain = torch.nn.Linear(4, 2, bias=False)
    environment_score = torch.nn.Linear(4, 3, bias=False)

    grams = build_source_action_grams(
        module_a,
        module_measure,
        environment_a,
        environment_measure,
        group_control,
        module_gain,
        environment_score,
    )
    assert grams.shape == (2, 4, 3, 3)
    torch.testing.assert_close(grams, grams.transpose(-1, -2))
    assert torch.linalg.eigvalsh(grams.detach().double()).amin() >= -1.0e-6
    grams.square().sum().backward()
    assert module_a.grad is not None and torch.isfinite(module_a.grad).all()
    assert environment_a.grad is not None and torch.isfinite(environment_a.grad).all()
    assert group_control.grad is not None and torch.isfinite(group_control.grad).all()
    assert module_gain.weight.grad is not None and torch.isfinite(module_gain.weight.grad).all()
    assert environment_score.weight.grad is not None and torch.isfinite(environment_score.weight.grad).all()


def test_source_action_score_keeps_one_sided_zero_energy_ratio_and_zeroes_empty_block() -> None:
    catalogue = _catalogue(2)
    proposal_valid = torch.ones((1, 2), dtype=torch.bool)
    logits = torch.tensor([[[1.0, 0.0]]])
    logits_by_role = {
        "environment": logits,
        "physical_ports": logits,
        "outside_temperature": logits,
    }
    node = torch.tensor([[True, True]])
    grams = torch.zeros((1, 4, 2, 2))
    # Baseline alpha=(1,0) has zero energy; tying the pair changes its second
    # coordinate, so the numerator must survive the 1e-8 denominator floor.
    grams[0, 0, 1, 1] = 1.0
    score = node_source_action_scores(
        logits_by_role, proposal_valid, grams, catalogue, node
    )
    assert score.numerator.shape == (1, 3, 4, 1)
    assert score.denominator.shape == score.numerator.shape
    torch.testing.assert_close(score.numerator[0, 0, 0, 0], torch.tensor(0.25))
    torch.testing.assert_close(score.denominator[0, 0, 0, 0], torch.tensor(0.0))
    torch.testing.assert_close(score.ratio[0, 0, 0, 0], torch.tensor(2.5e7))
    torch.testing.assert_close(score.score2[0, 0], torch.tensor(2.5e7))

    empty_energy = node_source_action_scores(
        logits_by_role,
        proposal_valid,
        torch.zeros_like(grams),
        catalogue,
        node,
    )
    assert torch.equal(empty_energy.score2, torch.zeros_like(empty_energy.score2))
    assert torch.equal(empty_energy.numerator, torch.zeros_like(empty_energy.numerator))
    assert torch.equal(empty_energy.denominator, torch.zeros_like(empty_energy.denominator))


def test_continuous_taper_has_exact_schedule_and_flat_score_endpoints() -> None:
    nodes = torch.tensor(
        [[True, True, False], [True, True, True], [False, False, True]]
    )
    active = torch.ones((1, 3), dtype=torch.bool)
    mass = torch.ones((1, 3))
    close2 = 0.02**2
    keep2 = 0.06**2
    score2 = torch.tensor([[close2, keep2, 0.5 * (close2 + keep2)]], requires_grad=True)
    final = node_contraction(score2, mass, nodes, active, 150)
    torch.testing.assert_close(final.taper[0, 0], torch.tensor(0.0))
    torch.testing.assert_close(final.taper[0, 1], torch.tensor(1.0))
    torch.testing.assert_close(final.taper[0, 2], torch.tensor(0.5))
    torch.testing.assert_close(final.gamma, 1.0 - final.taper)
    torch.testing.assert_close(final.residual_scale, final.taper)
    final.taper.sum().backward()
    assert score2.grad is not None
    torch.testing.assert_close(score2.grad[0, 0], torch.tensor(0.0))
    torch.testing.assert_close(score2.grad[0, 1], torch.tensor(0.0))
    assert score2.grad[0, 2] > 0.0

    scheduled_identity = node_contraction(
        torch.zeros((1, 3)),
        mass,
        nodes,
        active,
        50,
    )
    assert torch.equal(scheduled_identity.taper, torch.ones_like(scheduled_identity.taper))
    assert torch.equal(scheduled_identity.gamma, torch.zeros_like(scheduled_identity.gamma))
    assert torch.equal(
        scheduled_identity.residual_scale,
        torch.ones_like(scheduled_identity.residual_scale),
    )

    faded = node_contraction(
        torch.zeros((1, 3)),
        torch.tensor([[1.0e-6, 1.0, 1.0]]),
        nodes,
        active,
        150,
    )
    assert torch.equal(faded.eligibility[0, 0], torch.tensor(0.0))
    assert torch.equal(faded.gamma[0, 0], torch.tensor(0.0))


def test_partial_transform_changes_logits_even_when_no_class_is_closed() -> None:
    nodes = torch.tensor([[True, True, False], [True, True, True]])
    residual = torch.tensor([[0.5, 1.0]], requires_grad=True)
    transform = build_tree_transform(residual, nodes)
    logits = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]])
    transformed = contracted_logits(logits, transform)
    assert torch.allclose(transformed[..., :2], transformed[..., :2].mean(dim=-1, keepdim=True)) is False
    assert not torch.equal(transformed, logits)
    transformed.square().sum().backward()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()

    exact_residual = torch.ones((1, 2))
    identity = build_tree_transform(exact_residual, nodes)
    assert torch.equal(identity, torch.eye(3)[None, :, :])


def test_closed_subtree_packing_prefers_ancestor_and_ignores_inactive_parent() -> None:
    groups = 8
    nodes = torch.zeros((4, groups), dtype=torch.bool)
    nodes[0, [6, 7]] = True
    nodes[1, [0, 6, 7]] = True
    nodes[2, [3, 4]] = True
    nodes[3, :] = True  # contains inactive leaf 5 and must not close
    closed = torch.ones((1, 4), dtype=torch.bool)
    active = torch.ones((1, groups), dtype=torch.bool)
    active[0, 5] = False
    packed = pack_exact_closed_subtrees(closed, nodes, active)
    assert packed.membership.shape == (1, groups, 4)
    assert packed.multiplicity[0].tolist() == [3, 1, 1, 2]
    assert packed.class_id[0, 0] == packed.class_id[0, 6] == packed.class_id[0, 7]
    assert packed.root_leaf[0, 0] == packed.root_leaf[0, 6] == packed.root_leaf[0, 7] == 0
    assert packed.class_id[0, 3] == packed.class_id[0, 4]
    assert packed.root_leaf[0, 3] == packed.root_leaf[0, 4] == 3
    assert packed.class_id[0, 5] == -1
    assert packed.root_leaf[0, 5] == -1
    assert packed.valid[0].all()


def test_virtual_sparsemax_matches_weighted_quotient_and_source_moments() -> None:
    torch.manual_seed(15030402)
    batch, queries, groups, sources, width = 2, 7, 5, 4, 3
    nodes = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
    residual = torch.zeros((batch, 2))
    transform = build_tree_transform(residual, nodes)
    active = torch.tensor([[True, True, True, True, True], [True, True, True, True, False]])
    closed = torch.ones((batch, 2), dtype=torch.bool)
    packed = pack_exact_closed_subtrees(closed, nodes, active)
    logits = torch.randn((batch, queries, groups))
    virtual_logits = contracted_logits(logits, transform)
    alpha_virtual = masked_sparsemax(
        virtual_logits, active[:, None, :].expand(batch, queries, groups)
    )

    multiplicity = packed.multiplicity.to(dtype=logits.dtype)
    compact_logits = torch.einsum(
        "bkr,bqk->bqr", packed.membership, virtual_logits
    ) / multiplicity.clamp_min(1.0)[:, None, :]
    compact_mass, compact_density = weighted_sparsemax(
        compact_logits, packed.multiplicity, packed.valid[:, None, :]
    )
    del compact_mass
    alpha_packed = torch.einsum(
        "bkr,bqr->bqk", packed.membership, compact_density
    )
    torch.testing.assert_close(alpha_virtual, alpha_packed, atol=1.0e-6, rtol=1.0e-6)

    incidence = torch.rand((batch, sources, groups))
    controls = torch.rand((batch, groups, width))
    incidence_bar = torch.einsum("bsk,bkr->bsr", incidence, packed.membership)
    moment_bar = torch.einsum(
        "bsk,bkd,bkr->bsrd", incidence, controls, packed.membership
    )
    expected_moment = torch.zeros_like(moment_bar)
    for batch_index in range(batch):
        for source_index in range(sources):
            for class_index in range(packed.membership.shape[-1]):
                members = packed.membership[batch_index, :, class_index].bool()
                expected_moment[batch_index, source_index, class_index] = (
                    incidence[batch_index, source_index, members][:, None]
                    * controls[batch_index, members]
                ).sum(dim=0)
    torch.testing.assert_close(moment_bar, expected_moment)
    torch.testing.assert_close(
        incidence_bar.sum(dim=-1),
        (incidence * active[:, None, :]).sum(dim=-1),
    )


def test_combined_post_transform_discrepancy_keeps_role_block_summaries() -> None:
    catalogue = _catalogue(3)
    proposal_valid = torch.ones((1, 3), dtype=torch.bool)
    parent = torch.tensor([[[1.0, 0.0, -1.0]]])
    transform = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]])
    transformed = contracted_logits(parent, transform)
    parent_by_role = {role: parent for role in ("environment", "physical_ports", "outside_temperature")}
    transformed_by_role = {
        role: transformed for role in parent_by_role
    }
    grams = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 4, 3, 3).clone()
    result = combined_post_transform_discrepancy(
        parent_by_role,
        transformed_by_role,
        proposal_valid,
        grams,
        catalogue,
    )
    assert result.score2.shape == (1,)
    assert result.numerator.shape == (1, 3, 4)
    assert result.denominator.shape == (1, 3, 4)
    assert result.ratio.shape == (1, 3, 4)
    assert torch.isfinite(result.ratio).all()


def test_complete_linkage_is_laminar_and_lexicographic_on_exact_ties() -> None:
    dissimilarity = torch.ones((4, 4))
    dissimilarity.fill_diagonal_(0.0)
    subsets = deterministic_complete_linkage(dissimilarity)
    assert subsets == [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
    for index, left in enumerate(subsets):
        for right in subsets[index + 1 :]:
            overlap = set(left) & set(right)
            assert not overlap or set(left) <= set(right) or set(right) <= set(left)


def test_proposal_and_tree_leaf_permutation_equivariance() -> None:
    torch.manual_seed(15030403)
    batch, queries, groups = 2, 3, 5
    nodes = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, False, False, True, True],
            [True, True, True, True, True],
        ]
    )
    residual = torch.tensor([[0.15, 0.35, 0.7, 0.9], [0.8, 0.1, 0.45, 0.6]])
    active = torch.tensor([[True, True, True, True, True], [True, True, True, False, True]])
    logits = torch.randn((batch, queries, groups))
    grams = torch.randn((batch, 4, groups, groups))
    grams = grams @ grams.transpose(-1, -2)
    transform = build_tree_transform(residual, nodes)
    contracted = contracted_logits(logits, transform)

    permutation = torch.tensor([3, 0, 4, 1, 2])
    nodes_permuted = nodes[:, permutation]
    residual_permuted = residual
    active_permuted = active[:, permutation]
    logits_permuted = logits[..., permutation]
    grams_permuted = grams[:, :, permutation][:, :, :, permutation]
    transform_permuted = build_tree_transform(residual_permuted, nodes_permuted)
    contracted_permuted = contracted_logits(logits_permuted, transform_permuted)

    expected_transform = transform[:, permutation][:, :, permutation]
    expected_logits = contracted[..., permutation]
    torch.testing.assert_close(transform_permuted, expected_transform)
    torch.testing.assert_close(contracted_permuted, expected_logits)

    # The node score is invariant when leaves, masks, logits, and Gram axes
    # move together. Tree-node order remains fixed.
    catalogue = _catalogue(groups, batch=batch, probes=queries)
    by_role = {
        role: logits
        for role in ("environment", "physical_ports", "outside_temperature")
    }
    by_role_permuted = {role: value[..., permutation] for role, value in by_role.items()}
    score = node_source_action_scores(by_role, active, grams, catalogue, nodes)
    score_permuted = node_source_action_scores(
        by_role_permuted,
        active_permuted,
        grams_permuted,
        catalogue,
        nodes_permuted,
    )
    torch.testing.assert_close(score_permuted.score2, score.score2)
    torch.testing.assert_close(score_permuted.numerator, score.numerator)
    torch.testing.assert_close(score_permuted.denominator, score.denominator)



def test_mixed_packed_width_preserves_inactive_holes_and_unequal_multiplicity() -> None:
    nodes = torch.zeros((4, 5), dtype=torch.bool)
    nodes[0, [0, 1]] = True
    nodes[1, [0, 1, 2]] = True
    nodes[2, [3, 4]] = True
    nodes[3, :] = True
    closed = torch.tensor(
        [
            [True, True, False, False],
            [False, False, False, False],
        ]
    )
    active = torch.tensor(
        [
            [True, True, True, False, True],
            [True, False, True, True, True],
        ]
    )
    packed = pack_exact_closed_subtrees(closed, nodes, active)
    assert packed.membership.shape == (2, 5, 4)
    assert packed.valid.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert packed.multiplicity.tolist() == [[3, 1, 0, 0], [1, 1, 1, 1]]
    assert packed.class_id[0].tolist() == [0, 0, 0, -1, 1]
    assert packed.class_id[1].tolist() == [0, -1, 1, 2, 3]
    assert torch.equal(packed.membership[0, 3], torch.zeros(4))

    # Unequal class sizes are multiplicity weights in compact sparsemax. A
    # padded invalid compact column must carry no mass or density.
    compact_logits = torch.tensor(
        [
            [[0.2, -0.1, 0.0, 0.0]],
            [[0.3, -0.2, 0.1, -0.1]],
        ]
    )
    mass, density = weighted_sparsemax(
        compact_logits, packed.multiplicity, packed.valid[:, None, :]
    )
    assert torch.equal(mass[0, 0, 2:], torch.zeros(2))
    assert torch.equal(density[0, 0, 2:], torch.zeros(2))
    torch.testing.assert_close(mass.sum(dim=-1), torch.ones((2, 1)))


def test_missing_probe_role_errors_and_zero_support_is_finite_zero() -> None:
    groups = 3
    catalogue = _catalogue(groups)
    proposal_valid = torch.ones((1, groups), dtype=torch.bool)
    logits = torch.tensor([[[0.2, -0.1, 0.0]]])
    all_roles = {
        role: logits
        for role in ("environment", "physical_ports", "outside_temperature")
    }
    nodes = torch.tensor([[True, True, False]])
    grams = torch.eye(groups).reshape(1, 1, groups, groups).expand(1, 4, groups, groups)

    try:
        node_source_action_scores(
            {"environment": logits, "physical_ports": logits},
            proposal_valid,
            grams,
            catalogue,
            nodes,
        )
    except ValueError as error:
        assert "outside_temperature" in str(error)
    else:
        raise AssertionError("a missing source role must be rejected")

    no_support_catalogue = FunctionalProbeCatalogue(
        phase="P2",
        physical_ports=catalogue.physical_ports,
        outside_temperature=catalogue.outside_temperature,
        environment=FunctionalProbeFamily(
            coordinates=torch.zeros((1, 1, 2)),
            valid_mask=torch.zeros((1, 1), dtype=torch.bool),
            weights=torch.zeros((1, 1)),
        ),
    )
    unsupported = {role: logits for role in all_roles}
    no_support = node_source_action_scores(
        unsupported,
        proposal_valid,
        grams,
        no_support_catalogue,
        nodes,
    )
    assert torch.isfinite(no_support.score2).all()
    assert torch.equal(no_support.score2, torch.zeros_like(no_support.score2))
    assert torch.equal(no_support.numerator, torch.zeros_like(no_support.numerator))
    assert torch.equal(no_support.denominator, torch.zeros_like(no_support.denominator))


def test_raw_mixture_of_parent_logits_differs_from_renormalized_centroid_route() -> None:
    # Parent route logits include independently RMS-normalized keys and the
    # actual geometry at each leaf. The v4 tie is their arithmetic mean;
    # rebuilding from a mean key and a centroid geometry is not equivalent.
    sqrt2 = torch.sqrt(torch.tensor(2.0))
    raw_keys = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    parent_keys = raw_keys / torch.sqrt(raw_keys.square().mean(dim=-1, keepdim=True) + 1e-6)
    query = torch.tensor([1.0, 0.0])
    scaled_query = query / torch.sqrt(query.square().mean() + 1e-6)
    key_logits = (parent_keys @ scaled_query) / sqrt2
    centers = torch.tensor([-1.0, 2.0])
    parent_geometry = -centers.abs()  # both module and environment centers coincide
    parent_logits = key_logits + parent_geometry
    raw_tie = parent_logits.mean()

    mean_key = raw_keys.mean(dim=0)
    renormalized_mean_key = mean_key / torch.sqrt(mean_key.square().mean() + 1e-6)
    centroid_key_logit = (renormalized_mean_key @ scaled_query) / sqrt2
    centroid_geometry = -centers.mean().abs()
    centroid_reconstruction = centroid_key_logit + centroid_geometry
    assert not torch.isclose(raw_tie, centroid_reconstruction, atol=1e-3, rtol=1e-3)
