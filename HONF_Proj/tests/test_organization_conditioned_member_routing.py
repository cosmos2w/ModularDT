from __future__ import annotations

import math

import pytest
import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.interaction_reused_member_routing.consistent import (
    ConsistentJointMemberRoutingCore,
    compose_joint_member_prior,
    joint_likelihood_contribution,
    joint_role_statistics,
)
from honf_forward_core.interaction_reused_member_routing.organization_conditioned import (
    OrganizationConditionedMemberRoutingCore,
    organization_conditioned_member_contribution,
)
from honf_forward_core.interaction_reused_member_routing.relative import read_probability


def _mixture_prior(proposal: torch.Tensor, retrieval: torch.Tensor) -> torch.Tensor:
    composed = torch.bmm(proposal, retrieval)
    return composed / composed.sum(-1, keepdim=True).clamp_min(torch.finfo(composed.dtype).tiny)


def _contribute(
    proposal: torch.Tensor,
    retrieval: torch.Tensor,
    scores: torch.Tensor,
    candidate: torch.Tensor,
    *,
    row_mass: torch.Tensor | None = None,
    calibration: str = "normalized",
):
    if row_mass is None:
        row_mass = torch.ones(retrieval.shape[:2], dtype=retrieval.dtype, device=retrieval.device)
    prior = _mixture_prior(proposal, retrieval)
    return organization_conditioned_member_contribution(
        prior,
        scores,
        candidate,
        proposal=proposal,
        retrieval_R=retrieval,
        row_mass=row_mass,
        retrieval_calibration=calibration,
    )


def test_controlled_matched_marginal_directories_change_b_but_not_a() -> None:
    proposal = torch.tensor([[[0.5, 0.5]]], dtype=torch.float64)
    retrieval_1 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]], dtype=torch.float64)
    retrieval_2 = torch.tensor([[[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]]], dtype=torch.float64)
    scores = torch.tensor([[[0.0, math.log(2.0), math.log(4.0)]]], dtype=torch.float64)
    candidate = torch.ones_like(scores, dtype=torch.bool)
    prior_1 = _mixture_prior(proposal, retrieval_1)
    prior_2 = _mixture_prior(proposal, retrieval_2)
    torch.testing.assert_close(prior_1, torch.tensor([[[0.5, 0.25, 0.25]]], dtype=torch.float64))
    torch.testing.assert_close(prior_2, prior_1)

    b_1, _, _, trace_1 = organization_conditioned_member_contribution(
        prior_1, scores, candidate, proposal=proposal, retrieval_R=retrieval_1,
        row_mass=torch.ones((1, 2), dtype=torch.float64), retrieval_calibration="normalized",
    )
    b_2, _, _, _ = organization_conditioned_member_contribution(
        prior_2, scores, candidate, proposal=proposal, retrieval_R=retrieval_2,
        row_mass=torch.ones((1, 2), dtype=torch.float64), retrieval_calibration="normalized",
    )
    a_1, _, _ = joint_likelihood_contribution(prior_1, scores, candidate)
    a_2, _, _ = joint_likelihood_contribution(prior_2, scores, candidate)

    torch.testing.assert_close(a_1, torch.tensor([[[0.25, 0.25, 0.5]]], dtype=torch.float64))
    torch.testing.assert_close(a_2, a_1)
    torch.testing.assert_close(b_1, torch.tensor([[[0.5, 1.0 / 6.0, 1.0 / 3.0]]], dtype=torch.float64))
    torch.testing.assert_close(b_2, torch.tensor([[[0.25, 0.25, 0.5]]], dtype=torch.float64))
    torch.testing.assert_close(trace_1["organization_conditioned_retained_prior_mass"],
                               torch.ones((1, 1), dtype=torch.float64))
    assert trace_1["organization_conditioned_pre_normalization_error"].abs().max().item() < 1e-12


@pytest.mark.parametrize("calibration", ["normalized", "row_strength"])
def test_constant_scores_make_b_equal_a_for_each_retrieval_calibration(calibration: str) -> None:
    proposal = torch.tensor([[[0.7, 0.3]]], dtype=torch.float64)
    retrieval = torch.tensor([[[0.6, 0.3, 0.1], [0.0, 0.5, 0.5]]], dtype=torch.float64)
    row_mass = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    scores = torch.full((1, 1, 3), 3.5, dtype=torch.float64)
    candidate = torch.tensor([[[True, True, False]]])
    raw = retrieval * row_mass[..., None]
    prior, _ = compose_joint_member_prior(
        proposal, retrieval, raw, row_mass, torch.ones_like(scores, dtype=torch.bool),
        calibration=calibration,
    )
    b, _, _, _ = organization_conditioned_member_contribution(
        prior, scores, candidate, proposal=proposal, retrieval_R=retrieval,
        row_mass=row_mass, retrieval_calibration=calibration,
    )
    a, _, _ = joint_likelihood_contribution(prior, scores, candidate)
    torch.testing.assert_close(b, a, rtol=1e-12, atol=1e-12)


def test_identical_rows_reduce_b_to_a_for_nonconstant_scores() -> None:
    proposal = torch.tensor([[[0.15, 0.85]]], dtype=torch.float64)
    row = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
    retrieval = torch.stack((row, row), dim=0)[None]
    scores = torch.tensor([[[-1.0, 0.25, 1.2]]], dtype=torch.float64)
    candidate = torch.tensor([[[True, False, True]]])
    prior = _mixture_prior(proposal, retrieval)

    b, _, _, _ = organization_conditioned_member_contribution(
        prior, scores, candidate, proposal=proposal, retrieval_R=retrieval,
        row_mass=torch.ones((1, 2), dtype=torch.float64), retrieval_calibration="normalized",
    )
    a, _, _ = joint_likelihood_contribution(prior, scores, candidate)
    torch.testing.assert_close(b, a, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("score_law", ["inherited", "bounded_tanh"])
def test_extreme_raw_scores_are_transformed_before_b_uses_the_common_offset(score_law: str) -> None:
    proposal = torch.tensor([[[0.45, 0.55]]], dtype=torch.float64)
    retrieval = torch.tensor([[[0.8, 0.2, 0.0], [0.1, 0.4, 0.5]]], dtype=torch.float64)
    raw_scores = torch.tensor([[[-1000.0, 30.0, 1000.0]]], dtype=torch.float64)
    candidate = torch.ones_like(raw_scores, dtype=torch.bool)
    prior = _mixture_prior(proposal, retrieval)
    ell = ConsistentJointMemberRoutingCore._score_values(raw_scores, prior, candidate, score_law)
    assert ell.abs().max().item() <= 4.0

    actual, _, _, trace = _contribute(proposal, retrieval, ell, candidate)
    expected = _explicit_reference(
        proposal, retrieval, ell, candidate,
        torch.ones(retrieval.shape[:2], dtype=retrieval.dtype), "normalized",
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert trace["organization_conditioned_pre_normalization_error"].abs().max().item() < 1e-12


def test_empty_views_and_small_retained_mass_are_finite_and_fade_by_d() -> None:
    epsilon = 1.0e-5
    retrieval = torch.tensor(
        [[[0.5, 0.5, 0.0], [epsilon, 0.0, 1.0 - epsilon], [0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    proposal = torch.tensor([[[0.5, 0.5, 0.7]]], dtype=torch.float64)
    scores = torch.tensor([[[0.0, math.log(2.0), 4.0]]], dtype=torch.float64)
    candidate = torch.tensor([[[True, True, False]]])
    contribution, _, _, trace = _contribute(proposal, retrieval, scores, candidate)

    expected_view_mix = torch.tensor([[[1.0 / (1.0 + epsilon), epsilon / (1.0 + epsilon), 0.0]]],
                                     dtype=torch.float64)
    torch.testing.assert_close(trace["organization_conditioned_view_mixture_mass"],
                               expected_view_mix, rtol=1e-11, atol=1e-12)
    assert torch.isfinite(contribution).all()
    torch.testing.assert_close(contribution.sum(-1), torch.ones((1, 1), dtype=torch.float64))
    assert contribution[0, 0, 0] == pytest.approx(1.0 / 3.0, abs=2e-5)
    assert contribution[0, 0, 1] == pytest.approx(2.0 / 3.0, abs=2e-5)
    assert trace["organization_conditioned_retained_view_mass"][0, 0, 2].item() == 0.0
    assert trace["organization_conditioned_scaled_view_likelihood_mass"][0, 0, 2].item() == 0.0


def test_empty_candidate_set_returns_zero_contribution_and_normalizers() -> None:
    proposal = torch.tensor([[[0.4, 0.6]]], dtype=torch.float64)
    retrieval = torch.tensor([[[0.7, 0.3], [0.2, 0.8]]], dtype=torch.float64)
    scores = torch.tensor([[[4.0, -4.0]]], dtype=torch.float64)
    candidate = torch.zeros_like(scores, dtype=torch.bool)
    contribution, log_weight, common_offset, trace = _contribute(proposal, retrieval, scores, candidate)

    assert torch.equal(contribution, torch.zeros_like(contribution))
    assert torch.isneginf(log_weight).all()
    assert torch.equal(common_offset, torch.zeros_like(common_offset))
    assert torch.equal(trace["organization_conditioned_retained_prior_mass"],
                       torch.zeros_like(common_offset))
    assert torch.equal(trace["organization_conditioned_pre_normalization_error"],
                       torch.zeros_like(common_offset))
    assert torch.isfinite(contribution).all()


def test_overlap_coalesces_to_original_members_and_row_mass_splitting_is_invariant() -> None:
    proposal = torch.tensor([[[0.5, 0.5]]], dtype=torch.float64)
    retrieval = torch.tensor([[[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]], dtype=torch.float64)
    scores = torch.tensor([[[0.0, math.log(2.0), math.log(3.0)]]], dtype=torch.float64)
    candidate = torch.ones_like(scores, dtype=torch.bool)
    first, _, _, _ = _contribute(proposal, retrieval, scores, candidate)
    expected = torch.tensor([[[1.0 / 6.0, 8.0 / 15.0, 3.0 / 10.0]]], dtype=torch.float64)
    torch.testing.assert_close(first, expected, rtol=1e-12, atol=1e-12)

    split_retrieval = torch.cat((retrieval[:, :1], retrieval[:, :1], retrieval[:, 1:]), dim=1)
    split_proposal = torch.tensor([[[0.2, 0.3, 0.5]]], dtype=torch.float64)
    split, _, _, _ = _contribute(split_proposal, split_retrieval, scores, candidate)
    torch.testing.assert_close(split, first, rtol=1e-12, atol=1e-12)


def test_member_typed_permutation_equivariance() -> None:
    proposal = torch.tensor([[[0.35, 0.65]]], dtype=torch.float64)
    retrieval = torch.tensor([[[0.6, 0.0, 0.4, 0.0], [0.1, 0.5, 0.0, 0.4]]], dtype=torch.float64)
    scores = torch.tensor([[[-0.6, 0.8, 1.3, -0.2]]], dtype=torch.float64)
    candidate = torch.tensor([[[True, True, True, False]]])
    expected, _, _, _ = _contribute(proposal, retrieval, scores, candidate)

    # First two columns represent M members and the final two E members.
    for permutation in (torch.tensor([1, 0, 2, 3]), torch.tensor([0, 1, 3, 2]),
                        torch.tensor([2, 0, 3, 1])):
        inverse = torch.argsort(permutation)
        permuted, _, _, _ = _contribute(proposal, retrieval[..., permutation], scores[..., permutation],
                                        candidate[..., permutation])
        torch.testing.assert_close(permuted[..., inverse], expected, rtol=1e-12, atol=1e-12)


def _explicit_reference(
    proposal: torch.Tensor,
    retrieval: torch.Tensor,
    scores: torch.Tensor,
    candidate: torch.Tensor,
    row_mass: torch.Tensor,
    calibration: str,
) -> torch.Tensor:
    """Small loop oracle; deliberately favors clarity over contraction cost."""
    outputs = []
    for batch_index in range(scores.shape[0]):
        queries = []
        for query_index in range(scores.shape[1]):
            selected = candidate[batch_index, query_index]
            if not bool(selected.any()):
                queries.append(torch.zeros_like(scores[batch_index, query_index]))
                continue
            weights = proposal[batch_index, query_index]
            if calibration == "row_strength":
                weights = weights * row_mass[batch_index]
            weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
            offset = scores[batch_index, query_index, selected].amax()
            likelihood = torch.exp(scores[batch_index, query_index] - offset)
            retained_rows = []
            for address in range(retrieval.shape[1]):
                raw_mass = retrieval[batch_index, address, selected].sum()
                retained_rows.append(raw_mass if bool(raw_mass > 0) else torch.zeros_like(raw_mass))
            retained = torch.stack(retained_rows)
            total_retained = (weights * retained).sum()
            result = []
            for member in range(scores.shape[-1]):
                if not bool(selected[member]):
                    result.append(scores.new_zeros(()))
                    continue
                value = scores.new_zeros(())
                for address in range(retrieval.shape[1]):
                    row = retrieval[batch_index, address]
                    row_likelihood_mass = (row[selected] * likelihood[selected]).sum()
                    if bool((retained[address] > 0) & (row_likelihood_mass > 0)):
                        value = value + (
                            likelihood[member]
                            * weights[address]
                            * retained[address]
                            * row[member]
                            / (total_retained * row_likelihood_mass)
                        )
                result.append(value)
            queries.append(torch.stack(result))
        outputs.append(torch.stack(queries))
    return torch.stack(outputs)


@pytest.mark.parametrize("calibration", ["normalized", "row_strength"])
def test_b_contraction_matches_loop_reference_in_forward_and_backward(calibration: str) -> None:
    proposal = torch.tensor([[[0.25, 0.55, 0.20], [0.6, 0.1, 0.3]]], dtype=torch.float64,
                            requires_grad=True)
    retrieval = torch.tensor(
        [[[0.6, 0.4, 0.0, 0.0], [0.1, 0.2, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    scores = torch.tensor([[[0.4, -0.3, 0.8, -1.0], [-0.2, 0.6, 0.1, 0.9]]],
                          dtype=torch.float64, requires_grad=True)
    row_mass = torch.tensor([[0.3, 0.8, 0.5]], dtype=torch.float64, requires_grad=True)
    candidate = torch.tensor([[[True, True, True, False], [True, True, True, False]]])
    prior = _mixture_prior(proposal, retrieval)

    actual, _, _, trace = organization_conditioned_member_contribution(
        prior, scores, candidate, proposal=proposal, retrieval_R=retrieval,
        row_mass=row_mass, retrieval_calibration=calibration,
    )
    expected = _explicit_reference(proposal, retrieval, scores, candidate, row_mass, calibration)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)

    probe = torch.tensor([[[0.2, -0.7, 1.3, 0.4], [-0.4, 0.6, -0.8, 0.1]]], dtype=torch.float64)
    actual_grads = torch.autograd.grad((actual * probe).sum(), (proposal, retrieval, scores, row_mass),
                                       retain_graph=True, allow_unused=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), (proposal, retrieval, scores, row_mass),
                                         allow_unused=True)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        if actual_grad is None or expected_grad is None:
            assert actual_grad is expected_grad
        else:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-10)
    assert trace["organization_conditioned_pre_normalization_error"].abs().max().item() < 1e-12


def test_fp32_pre_normalization_mass_error_is_only_roundoff() -> None:
    torch.manual_seed(20261027)
    proposal = torch.softmax(torch.randn(3, 7, 5), dim=-1)
    retrieval = torch.softmax(torch.randn(3, 5, 12), dim=-1)
    scores = 4.0 * torch.tanh(torch.randn(3, 7, 12))
    candidate = torch.rand(3, 7, 12) > 0.2
    row_mass = torch.rand(3, 5)
    prior = torch.bmm(proposal, retrieval)

    for calibration in ("normalized", "row_strength"):
        _, _, _, trace = organization_conditioned_member_contribution(
            prior, scores, candidate, proposal=proposal, retrieval_R=retrieval,
            row_mass=row_mass, retrieval_calibration=calibration,
        )
        error = trace["organization_conditioned_pre_normalization_error"]
        assert torch.isfinite(error).all()
        assert error.abs().max().item() <= 3.0e-7


def _base_config() -> dict:
    return ContentHypergraphCoreV2(
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


def _scene(module_count: int = 3, environment_count: int = 5) -> SceneBatch:
    module_x = torch.linspace(0.15, 1.65, module_count)
    environment_x = torch.linspace(0.0, 1.8, environment_count)
    return SceneBatch(
        module_features=torch.randn(1, module_count, 9),
        environment_features=torch.randn(1, environment_count, 6),
        module_coordinates=torch.stack((module_x, torch.linspace(-0.1, 0.2, module_count)), -1)[None],
        environment_coordinates=torch.stack((environment_x, torch.zeros_like(environment_x)), -1)[None],
        module_mask=torch.ones((1, module_count), dtype=torch.bool),
        environment_mask=torch.ones((1, environment_count), dtype=torch.bool),
        module_measure=torch.linspace(0.0, 1.0, module_count)[None],
        environment_measure=torch.linspace(0.0, 1.0, environment_count)[None],
        module_ids=torch.arange(20, 20 + module_count)[None],
        environment_ids=torch.arange(100, 100 + environment_count)[None],
        local_radius=torch.full((1, module_count), 0.2),
        operating_features=torch.randn(1, 12),
        coordinate_origin=torch.tensor([[0.0, -0.5]]),
        coordinate_scale=torch.tensor([[2.0, 1.0]]),
        module_measure_reference=torch.ones(1),
        environment_measure_reference=torch.ones(1),
        scene_key="r13-5b-organization-conditioned-test-scene",
        normalizer_key="r13-5b-organization-conditioned-test-normalizer",
        coordinate_frame_key="r13-5b-organization-conditioned-test-frame",
        module_characteristic_lengths=torch.full((1, module_count, 2), 0.12),
        environment_characteristic_lengths=torch.full((1, environment_count, 2), 0.1),
    )


def _query(count: int = 5) -> QueryBatch:
    x = torch.linspace(0.05, 1.75, count)
    return QueryBatch(
        coordinates=torch.stack((x, torch.linspace(-0.15, 0.25, count)), -1)[None],
        query_mask=torch.ones(1, count, dtype=torch.bool),
        query_ids=torch.arange(500, 500 + count)[None],
        scene_key="r13-5b-organization-conditioned-test-scene",
        coordinate_frame_key="r13-5b-organization-conditioned-test-frame",
    )


def _make_core(core_type, *, tile_size: int = 2):
    return core_type(
        _base_config(), address_mode="ME", input_dataset="synthetic-r13-5b",
        query_tile_size=tile_size, retrieval_calibration="normalized", score_calibration="inherited",
    )


def _row_loop_b_contribution(
    member_prior: torch.Tensor,
    scores: torch.Tensor,
    candidate: torch.Tensor,
    *,
    proposal: torch.Tensor,
    retrieval_R: torch.Tensor,
    row_mass: torch.Tensor,
    retrieval_calibration: str,
):
    """Differentiable small-shape B oracle using explicit address/member loops."""
    del member_prior
    batches, queries, members = scores.shape
    outputs, log_weights, offsets = [], [], []
    tiny = torch.finfo(scores.dtype).tiny
    for batch_index in range(batches):
        batch_outputs, batch_log_weights, batch_offsets = [], [], []
        for query_index in range(queries):
            keep = candidate[batch_index, query_index]
            local_scores = scores[batch_index, query_index]
            local_retrieval = retrieval_R[batch_index]
            if not bool(keep.any()):
                batch_outputs.append(torch.zeros_like(local_scores))
                batch_log_weights.append(torch.full_like(local_scores, -torch.inf))
                batch_offsets.append(local_scores.new_zeros(()))
                continue
            address_weight = proposal[batch_index, query_index]
            if retrieval_calibration == "row_strength":
                address_weight = address_weight * row_mass[batch_index]
            address_weight = address_weight / address_weight.sum().clamp_min(tiny)
            score_offset = local_scores[keep].amax()
            likelihood = torch.exp(local_scores - score_offset) * keep.to(local_scores.dtype)
            retained = torch.stack([
                (local_retrieval[address] * keep.to(local_retrieval.dtype)).sum()
                for address in range(local_retrieval.shape[0])
            ])
            row_likelihood = torch.stack([
                (local_retrieval[address] * likelihood.to(local_retrieval.dtype)).sum()
                for address in range(local_retrieval.shape[0])
            ])
            denominator = (address_weight * retained).sum()
            member_base = []
            for member in range(members):
                row_terms = []
                for address in range(local_retrieval.shape[0]):
                    if bool((retained[address] > 0) & (row_likelihood[address] > 0)
                            & (denominator > 0)):
                        row_terms.append(
                            address_weight[address] * retained[address]
                            / (denominator * row_likelihood[address])
                            * local_retrieval[address, member]
                        )
                    else:
                        row_terms.append(scores.new_zeros(()))
                member_base.append(torch.stack(row_terms).sum() * keep[member].to(scores.dtype))
            base = torch.stack(member_base)
            valid_weight = keep & (base > 0)
            log_weight = torch.where(
                valid_weight,
                base.clamp_min(tiny).log() + local_scores - score_offset,
                torch.full_like(local_scores, -torch.inf),
            )
            offset = torch.where(valid_weight.any(), log_weight.amax(), log_weight.new_zeros(()))
            unnormalized = torch.where(
                valid_weight,
                torch.exp(log_weight - offset),
                torch.zeros_like(log_weight),
            )
            normalizer = unnormalized.sum()
            contribution = torch.where(
                normalizer > 0,
                unnormalized / normalizer.clamp_min(tiny),
                torch.zeros_like(unnormalized),
            )
            batch_outputs.append(contribution)
            batch_log_weights.append(log_weight)
            batch_offsets.append(offset)
        outputs.append(torch.stack(batch_outputs))
        log_weights.append(torch.stack(batch_log_weights))
        offsets.append(torch.stack(batch_offsets))
    return torch.stack(outputs), torch.stack(log_weights), torch.stack(offsets), {}


class _RowLoopOrganizationConditionedCore(OrganizationConditionedMemberRoutingCore):
    def _joint_member_contribution(self, member_prior, ell, candidate, **kwargs):
        return _row_loop_b_contribution(member_prior, ell, candidate, **kwargs)


def test_b_reuses_a_parameter_inventory_and_roundtrips_its_new_identity() -> None:
    torch.manual_seed(1913)
    core_a = _make_core(ConsistentJointMemberRoutingCore)
    torch.manual_seed(1913)
    core_b = _make_core(OrganizationConditionedMemberRoutingCore)

    state_a = core_a.state_dict()
    state_b = core_b.state_dict()
    assert len(state_a) == len(state_b) == 190
    assert tuple(state_a) == tuple(state_b)
    for name, value in state_a.items():
        torch.testing.assert_close(value, state_b[name], rtol=0, atol=0)
    assert core_b.architecture_name == "R13-5B-ME-OrganizationConditionedJoint95"
    assert core_b.get_config()["schema"] == "interaction_reused_organization_conditioned_member_routing_v1"
    assert OrganizationConditionedMemberRoutingCore.from_config(core_b.get_config()).get_config() == core_b.get_config()
    with pytest.raises(ValueError, match="ME"):
        OrganizationConditionedMemberRoutingCore(
            _base_config(), address_mode="M", input_dataset="synthetic-r13-5b",
        )


def test_b_actual_paired_decode_and_all_parameter_gradients_match_row_loop_reference() -> None:
    torch.manual_seed(1931)
    optimized = _make_core(OrganizationConditionedMemberRoutingCore, tile_size=2)
    reference = _make_core(_RowLoopOrganizationConditionedCore, tile_size=2)
    reference.load_state_dict(optimized.state_dict(), strict=True)
    optimized.train()
    reference.train()
    scene, query = _scene(), _query()

    optimized_prepared = optimized.prepare(scene)
    reference_prepared = reference.prepare(scene)
    actual_full, actual_restricted, _actual_shared = optimized.decode_paired(
        optimized_prepared, query,
    )
    expected_full, expected_restricted, _expected_shared = reference.decode_paired(
        reference_prepared, query,
    )
    torch.testing.assert_close(actual_full.joint_fields, expected_full.joint_fields, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        actual_restricted.joint_fields, expected_restricted.joint_fields, rtol=2e-6, atol=2e-7,
    )

    probe = torch.linspace(-0.8, 1.1, actual_full.joint_fields.numel()).reshape_as(actual_full.joint_fields)
    actual_loss = (actual_full.joint_fields * probe).sum() + 0.73 * (actual_restricted.joint_fields * probe).sum()
    expected_loss = (expected_full.joint_fields * probe).sum() + 0.73 * (
        expected_restricted.joint_fields * probe
    ).sum()
    actual_loss.backward()
    expected_loss.backward()

    expected_paths = (
        "member_router.proposal_M", "member_router.member_M", "member_router.member_E", "typed_read_gates.M",
        "typed_read_gates.E", "member_router.value_M", "member_router.value_E",
        "member_router.joint_head", "applied_incidence.routes.mm",
        "applied_incidence.routes.em", "applied_incidence.routes.ee", "applied_incidence.routes.me",
    )
    actual_parameters = dict(optimized.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    compared = [
        name for name in actual_parameters
        if any(path in name for path in expected_paths)
    ]
    assert compared
    for name in compared:
        actual_grad, expected_grad = actual_parameters[name].grad, reference_parameters[name].grad
        assert actual_grad is not None and expected_grad is not None, name
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-7, msg=name)

    expected_counts = {
        "query_B_retained_view_mass_contraction_pairs",
        "query_B_view_likelihood_mass_contraction_pairs",
        "query_B_coalesced_member_base_contraction_pairs",
    }
    assert expected_counts <= set(actual_full.work_counts)
    assert expected_counts <= set(actual_restricted.work_counts)
    expected_pairs_per_contraction = query.batch_size * query.num_queries * 8 * 8
    assert all(actual_full.work_counts[name] == expected_pairs_per_contraction for name in expected_counts)
    assert all(actual_restricted.work_counts[name] == expected_pairs_per_contraction for name in expected_counts)


def test_b_decode_trace_derives_conditional_and_gate_statistics_from_b_contribution() -> None:
    torch.manual_seed(1943)
    core = _make_core(OrganizationConditionedMemberRoutingCore, tile_size=5)
    core.eval()
    prepared = core.prepare(_scene())
    output = core.decode(
        prepared, _query(4), intervention={"read_mode": "restricted"}, return_trace=True,
    )
    trace = output.trace
    assert trace is not None
    assert trace["organization_conditioned_pre_normalization_error"].abs().max().item() < 2e-6
    torch.testing.assert_close(
        trace["member_joint_likelihood"].sum(-1), torch.ones_like(trace["member_joint_log_weight_offset"]),
        rtol=2e-6, atol=2e-6,
    )
    for role, eligible in (
        ("M", prepared.scene.module_mask),
        ("E", prepared.scene.environment_mask & (prepared.scene.environment_measure > 0)),
    ):
        contribution = trace[f"member_contribution_pre_strength_{role}"]
        scores = trace[f"member_ell_{role}"]
        candidate = trace[f"candidate_mask_{role}"]
        conditional, expectation, entropy = joint_role_statistics(
            contribution, scores, candidate, eligible.sum(-1),
        )
        torch.testing.assert_close(trace[f"member_conditional_likelihood_{role}"], conditional)
        torch.testing.assert_close(trace[f"member_conditional_expectation_{role}"], expectation)
        torch.testing.assert_close(trace[f"member_conditional_entropy_{role}"], entropy)
        expected_strength = read_probability(trace[f"member_read_contrast_{role}"]) * candidate.any(-1)
        torch.testing.assert_close(trace[f"member_read_strength_{role}"], expected_strength)
