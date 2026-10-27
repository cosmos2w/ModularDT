from __future__ import annotations

import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from honf_forward_core.interaction_reused_member_routing.typed_organization import (
    bounded_four_cycle_perturbation,
    bounded_row_member_transfer,
    organization_conditioned_role_contribution,
)
from honf_forward_core.interaction_reused_member_routing.typed_organization_core import (
    TypedOrganizationMemberRoutingCore,
)


def _me_contribution(prior: torch.Tensor, ell: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    likelihood = torch.exp(ell) * candidate.to(ell.dtype)
    weighted = prior * likelihood
    total = weighted.sum(-1, keepdim=True)
    return torch.where(total > 0, weighted / total.clamp_min(torch.finfo(weighted.dtype).tiny),
                       torch.zeros_like(weighted))


def _transparent_loop(
    retrieval_R: torch.Tensor,
    proposal_weights: torch.Tensor,
    ell: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    """Small equation-level oracle, intentionally written as nested loops."""
    batch, queries, members = ell.shape
    outputs = []
    for b in range(batch):
        query_outputs = []
        for q in range(queries):
            selected = [i for i in range(members) if bool(candidate[b, q, i])]
            if not selected:
                query_outputs.append(torch.zeros_like(ell[b, q]))
                continue
            likelihood = {i: torch.exp(ell[b, q, i]) for i in selected}
            retained = {}
            partitions = {}
            total_role_mass = ell.new_zeros(())
            for a in range(retrieval_R.shape[1]):
                retained[a] = sum((retrieval_R[b, a, i] for i in selected), ell.new_zeros(()))
                partitions[a] = sum(
                    (retrieval_R[b, a, i] * likelihood[i] for i in selected), ell.new_zeros(()),
                )
                total_role_mass = total_role_mass + proposal_weights[b, q, a] * retained[a]
            values = []
            for i in range(members):
                if i not in selected or not bool(total_role_mass > 0):
                    values.append(ell.new_zeros(()))
                    continue
                value = ell.new_zeros(())
                for a in range(retrieval_R.shape[1]):
                    if bool(retained[a] > 0) and bool(partitions[a] > 0):
                        view_mass = proposal_weights[b, q, a] * retained[a] / total_role_mass
                        value = value + view_mass * retrieval_R[b, a, i] * likelihood[i] / partitions[a]
                values.append(value)
            result = torch.stack(values)
            result = result / result.sum().clamp_min(torch.finfo(result.dtype).tiny)
            query_outputs.append(result)
        outputs.append(torch.stack(query_outputs))
    return torch.stack(outputs)


def test_role_normalization_empty_padding_and_pure_role_reads_are_exact() -> None:
    retrieval = torch.tensor(
        [[[0.40, 0.30, 0.20, 0.10, 0.00, 0.00],
          [0.10, 0.20, 0.30, 0.10, 0.15, 0.15],
          [0.00, 0.00, 0.10, 0.30, 0.30, 0.30]]],
        dtype=torch.double,
    )
    proposal = torch.tensor([[[0.2, 0.5, 0.3], [0.3, 0.4, 0.3]]], dtype=torch.double)
    ell = torch.tensor([[[0.4, -0.2, 1.1, 0.7, -0.8, 0.3],
                         [0.1, 0.2, -0.3, 0.4, 0.5, -0.6]]], dtype=torch.double)
    # The third M slot and final E slot are padded; candidates never admit them.
    candidate_m = torch.tensor([[[True, True, False], [False, False, False]]])
    candidate_e = torch.tensor([[[True, True, False], [False, False, False]]])

    contribution_m = organization_conditioned_role_contribution(
        retrieval[:, :, :3], proposal, ell[:, :, :3], candidate_m,
    )
    contribution_e = organization_conditioned_role_contribution(
        retrieval[:, :, 3:6], proposal, ell[:, :, 3:6], candidate_e,
    )

    torch.testing.assert_close(contribution_m.sum(-1), torch.tensor([[1.0, 0.0]], dtype=torch.double))
    torch.testing.assert_close(contribution_e.sum(-1), torch.tensor([[1.0, 0.0]], dtype=torch.double))
    assert torch.equal(contribution_m[..., 2], torch.zeros_like(contribution_m[..., 2]))
    assert torch.equal(contribution_e[..., 2], torch.zeros_like(contribution_e[..., 2]))
    assert torch.equal(contribution_m[:, 1], torch.zeros_like(contribution_m[:, 1]))
    assert torch.equal(contribution_e[:, 1], torch.zeros_like(contribution_e[:, 1]))

    # An allowed one-role read has unit conditional mass; the other role can
    # remain exactly empty without a quota or a shared M/E denominator.
    only_m = organization_conditioned_role_contribution(
        retrieval[:, :, :3], proposal[:, :1], ell[:, :1, :3], candidate_m[:, :1],
    )
    no_e = organization_conditioned_role_contribution(
        retrieval[:, :, 3:6], proposal[:, :1], ell[:, :1, 3:6], torch.zeros_like(candidate_e[:, :1]),
    )
    only_e = organization_conditioned_role_contribution(
        retrieval[:, :, 3:6], proposal[:, :1], ell[:, :1, 3:6], candidate_e[:, :1],
    )
    no_m = organization_conditioned_role_contribution(
        retrieval[:, :, :3], proposal[:, :1], ell[:, :1, :3], torch.zeros_like(candidate_m[:, :1]),
    )
    torch.testing.assert_close(only_m.sum(-1), torch.ones((1, 1), dtype=torch.double))
    torch.testing.assert_close(only_e.sum(-1), torch.ones((1, 1), dtype=torch.double))
    assert torch.equal(no_e, torch.zeros_like(no_e))
    assert torch.equal(no_m, torch.zeros_like(no_m))


def test_values_and_gradients_match_transparent_equation_loop() -> None:
    torch.manual_seed(136)
    retrieval = (torch.rand(2, 4, 5, dtype=torch.double) + 0.15).requires_grad_()
    proposal = (torch.rand(2, 3, 4, dtype=torch.double) + 0.1).requires_grad_()
    ell = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    candidate = torch.tensor(
        [[[True, True, False, True, False], [True, False, True, True, True], [False, False, False, False, False]],
         [[True, True, True, False, True], [False, True, True, False, False], [True, True, False, True, True]]],
    )
    probe = torch.randn(2, 3, 5, dtype=torch.double)

    actual = organization_conditioned_role_contribution(retrieval, proposal, ell, candidate)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), (retrieval, proposal, ell))

    reference_retrieval = retrieval.detach().clone().requires_grad_()
    reference_proposal = proposal.detach().clone().requires_grad_()
    reference_ell = ell.detach().clone().requires_grad_()
    expected = _transparent_loop(reference_retrieval, reference_proposal, reference_ell, candidate)
    expected_gradients = torch.autograd.grad((expected * probe).sum(),
                                             (reference_retrieval, reference_proposal, reference_ell))

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-11)
        assert torch.isfinite(actual_gradient).all()


def test_each_role_reads_all_address_rows_including_cross_role_addresses() -> None:
    base_config = ContentHypergraphCoreV2(
        module_input_dim=9,
        environment_input_dim=6,
        coordinate_dim=2,
        query_input_dim=0,
        operating_input_dim=12,
        output_dim=4,
        architecture="A2-D",
        context_dim=16,
        num_heads=4,
        context_depth=2,
        local_hidden_dim=16,
        dropout=0.0,
        init_seed=731,
    ).get_config()
    core = TypedOrganizationMemberRoutingCore(
        base_config,
        address_mode="ME",
        input_dataset="synthetic-cross-role-addresses",
        query_tile_size=1,
    )
    # E-owned address rows (indices 2 and 3) provide distinct legal reads of
    # M members. They must compete alongside M-owned rows for the M role.
    retrieval = torch.tensor(
        [[[0.80, 0.20, 0.00, 0.00],
          [0.20, 0.80, 0.00, 0.00],
          [0.95, 0.05, 0.00, 0.00],
          [0.05, 0.95, 0.00, 0.00]]],
        dtype=torch.double,
    )
    proposal = torch.tensor([[[0.10, 0.10, 0.40, 0.40]]], dtype=torch.double)
    prior = torch.bmm(proposal, retrieval)
    ell = torch.tensor([[[0.1, 1.3, 0.0, 0.0]]], dtype=torch.double)
    candidate = torch.tensor([[[True, True, False, False]]])
    marginal = _me_contribution(prior, ell, candidate)

    gate, value, _ = core._role_read_contributions(
        marginal,
        prior,
        ell,
        candidate,
        full_candidate=candidate,
        proposal=proposal,
        retrieval_R=retrieval,
        row_mass=torch.ones((1, 4), dtype=torch.double),
        module_count=2,
    )
    expected_m = organization_conditioned_role_contribution(
        retrieval[..., :2], proposal, ell[..., :2], candidate[..., :2],
    )
    sliced_m = organization_conditioned_role_contribution(
        retrieval[:, :2, :2], proposal[..., :2], ell[..., :2], candidate[..., :2],
    )
    torch.testing.assert_close(value[..., :2], expected_m, rtol=1e-12, atol=1e-12)
    assert (expected_m - sliced_m).abs().sum() > 1e-3
    torch.testing.assert_close(gate[..., :2],
                               _me_contribution(prior[..., :2], ell[..., :2], candidate[..., :2]),
                               rtol=1e-12, atol=1e-12)
    assert torch.equal(value[..., 2:], torch.zeros_like(value[..., 2:]))


def test_fixed_marginal_four_cycle_changes_h_and_preserves_me() -> None:
    weights = torch.tensor([[[0.5, 0.5]]], dtype=torch.double)
    ell = torch.tensor([[[0.0, 1.6]]], dtype=torch.double)
    candidate = torch.ones((1, 1, 2), dtype=torch.bool)
    path_mass = torch.tensor([[0.35, 0.15], [0.15, 0.35]], dtype=torch.double)
    for requested_dose in (0.025, 0.05, 0.10):
        perturbed, dose = bounded_four_cycle_perturbation(
            path_mass, (0, 1), (0, 1), requested_dose,
        )
        assert dose == requested_dose
        torch.testing.assert_close(perturbed.sum(0), path_mass.sum(0), rtol=0, atol=1e-15)
        torch.testing.assert_close(perturbed.sum(1), path_mass.sum(1), rtol=0, atol=1e-15)
        assert (perturbed > 0).equal(path_mass > 0)
        moved_path_mass = 0.5 * (perturbed - path_mass).abs().sum()
        torch.testing.assert_close(moved_path_mass, torch.tensor(2 * dose, dtype=torch.double),
                                   rtol=0, atol=1e-15)

        retrieval = (path_mass / 0.5)[None]
        perturbed_retrieval = (perturbed / 0.5)[None]
        prior = torch.bmm(weights, retrieval)
        perturbed_prior = torch.bmm(weights, perturbed_retrieval)
        torch.testing.assert_close(prior, perturbed_prior, rtol=0, atol=1e-15)

        before = organization_conditioned_role_contribution(retrieval, weights, ell, candidate)
        after = organization_conditioned_role_contribution(perturbed_retrieval, weights, ell, candidate)
        torch.testing.assert_close(_me_contribution(prior, ell, candidate),
                                   _me_contribution(perturbed_prior, ell, candidate), rtol=0, atol=1e-15)
        assert (before - after).abs().sum() > 1e-3

    clipped, clipped_dose = bounded_four_cycle_perturbation(path_mass, (0, 1), (0, 1), 0.5)
    assert clipped_dose == 0.15
    assert (clipped >= 0).all()
    torch.testing.assert_close(clipped.sum(0), path_mass.sum(0), rtol=0, atol=1e-15)
    torch.testing.assert_close(clipped.sum(1), path_mass.sum(1), rtol=0, atol=1e-15)


def test_bounded_row_transfer_is_a_separate_p_changing_finite_control() -> None:
    retrieval = torch.tensor([[0.7, 0.3], [0.0, 1.0]], dtype=torch.double)
    proposal = torch.tensor([[0.4, 0.6]], dtype=torch.double)
    perturbed, applied = bounded_row_member_transfer(retrieval, 0, 0, 1, 0.35)
    assert applied == 0.35
    torch.testing.assert_close(perturbed.sum(-1), retrieval.sum(-1), rtol=0, atol=1e-15)
    assert torch.equal((perturbed > 0)[1], (retrieval > 0)[1])
    assert perturbed[0, 1] > retrieval[0, 1]
    assert not torch.allclose(proposal @ perturbed, proposal @ retrieval, rtol=0, atol=1e-8)

    sparse = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.double)
    expanded, clipped_dose = bounded_row_member_transfer(sparse, 0, 0, 1, 0.9)
    assert clipped_dose == 0.9
    assert expanded[0, 1] > 0
    torch.testing.assert_close(expanded.sum(-1), sparse.sum(-1), rtol=0, atol=1e-15)


def test_exact_degeneracies_match_the_original_role_marginal_law() -> None:
    weights = torch.tensor([[[0.25, 0.45, 0.30]]], dtype=torch.double)
    candidate = torch.tensor([[[True, True, True, False]]])
    score = torch.tensor([[[0.0, 1.2, -0.7, 2.0]]], dtype=torch.double)

    # Constant likelihood leaves the weighted member marginal unchanged for
    # any row composition.
    retrieval = torch.tensor(
        [[[0.60, 0.10, 0.20, 0.10], [0.05, 0.55, 0.30, 0.10], [0.15, 0.20, 0.55, 0.10]]],
        dtype=torch.double,
    )
    flat_score = torch.tensor([[[0.4, 0.4, 0.4, 10.0]]], dtype=torch.double)
    prior = torch.bmm(weights, retrieval)
    actual = organization_conditioned_role_contribution(retrieval, weights, flat_score, candidate)
    torch.testing.assert_close(actual, _me_contribution(prior, flat_score, candidate), rtol=1e-12, atol=1e-12)

    # All selected rows have the same conditional member composition, though
    # their retained row masses differ.
    same_conditional = torch.tensor(
        [[[0.20, 0.30, 0.10, 0.40], [0.60, 0.90, 0.20, 0.30],
          [0.10, 0.15, 0.80, 0.25]]],
        dtype=torch.double,
    )
    selected = candidate.clone()
    selected[..., 2] = False
    conditional_prior = torch.bmm(weights, same_conditional)
    actual = organization_conditioned_role_contribution(same_conditional, weights, score, selected)
    torch.testing.assert_close(actual, _me_contribution(conditional_prior, score, selected),
                               rtol=1e-12, atol=1e-12)

    # A single active view and a singleton candidate each collapse exactly to
    # the original marginal law.
    one_active = torch.tensor([[[0.0, 0.0, 0.0], [0.30, 0.25, 0.45], [0.0, 0.0, 0.0]]],
                              dtype=torch.double)
    one_weight = torch.tensor([[[0.2, 0.6, 0.2]]], dtype=torch.double)
    one_prior = torch.bmm(one_weight, one_active)
    actual = organization_conditioned_role_contribution(one_active, one_weight, score[..., :3],
                                                        torch.ones((1, 1, 3), dtype=torch.bool))
    torch.testing.assert_close(actual, _me_contribution(one_prior, score[..., :3],
                                                        torch.ones((1, 1, 3), dtype=torch.bool)),
                               rtol=1e-12, atol=1e-12)

    singleton = torch.tensor([[[False, True, False, False]]])
    actual = organization_conditioned_role_contribution(retrieval, weights, score, singleton)
    torch.testing.assert_close(actual, singleton.to(torch.double), rtol=0, atol=0)
