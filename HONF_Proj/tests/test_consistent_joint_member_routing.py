from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.interaction_reused_member_routing.consistent import (
    ConsistentJointMemberRoutingCore,
    _PAIRED_SHARED_COUNT_KEYS,
    compose_joint_member_prior,
    joint_likelihood_contribution,
    joint_role_statistics,
)
from honf_forward_core.interaction_reused_member_routing.joint import (
    JointMemberRoutingCore,
    joint_member_prefix_mask,
)
from honf_forward_core.interaction_reused_member_routing.organization_conditioned import (
    OrganizationConditionedMemberRoutingCore,
)
from honf_forward_core.interaction_reused_member_routing.guided import prior_weighted_scores


def _base_config(*, output_dim: int = 4, context_depth: int = 2) -> dict:
    return ContentHypergraphCoreV2(
        module_input_dim=9,
        environment_input_dim=6,
        coordinate_dim=2,
        query_input_dim=0,
        operating_input_dim=12,
        output_dim=output_dim,
        architecture="A2-D",
        context_dim=16,
        num_heads=4,
        context_depth=context_depth,
        local_hidden_dim=16,
        dropout=0.0,
        init_seed=731,
    ).get_config()


def _scene(module_count: int = 3, environment_count: int = 6) -> SceneBatch:
    module_x = torch.linspace(0.15, 1.65, module_count)
    module_coordinates = torch.stack((module_x, torch.linspace(-0.1, 0.2, module_count)), dim=-1)
    environment_x = torch.linspace(0.0, 1.8, environment_count)
    environment_coordinates = torch.stack((environment_x, torch.zeros_like(environment_x)), dim=-1)
    return SceneBatch(
        module_features=torch.randn(1, module_count, 9),
        environment_features=torch.randn(1, environment_count, 6),
        module_coordinates=module_coordinates[None],
        environment_coordinates=environment_coordinates[None],
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
        scene_key="r13-5a-consistent-test-scene",
        normalizer_key="r13-5a-consistent-test-normalizer",
        coordinate_frame_key="r13-5a-consistent-test-frame",
        module_characteristic_lengths=torch.full((1, module_count, 2), 0.12),
        environment_characteristic_lengths=torch.full((1, environment_count, 2), 0.1),
    )


def _query(count: int = 4) -> QueryBatch:
    x = torch.linspace(0.05, 1.75, count)
    return QueryBatch(
        coordinates=torch.stack((x, torch.linspace(-0.15, 0.25, count)), dim=-1)[None],
        query_mask=torch.ones(1, count, dtype=torch.bool),
        query_ids=torch.arange(500, 500 + count)[None],
        scene_key="r13-5a-consistent-test-scene",
        coordinate_frame_key="r13-5a-consistent-test-frame",
    )


def _core(*, tile_size: int = 4, retrieval_calibration: str = "normalized",
          score_calibration: str = "inherited") -> ConsistentJointMemberRoutingCore:
    return ConsistentJointMemberRoutingCore(
        _base_config(),
        address_mode="ME",
        input_dataset="synthetic-r13-5a-consistent",
        query_tile_size=tile_size,
        retrieval_calibration=retrieval_calibration,
        score_calibration=score_calibration,
    )


class _UnfactoredScorerMixin:
    """Test oracle using the same normalized inputs and original concatenated MLP."""

    def _address_logits(self, prepared, query, qstates, address_allowed, *, normalized_query=None):
        scene = prepared.scene
        batch, queries, address_count = address_allowed.shape
        modules, environments = scene.num_modules, scene.num_environment_tokens
        if address_count != modules + environments:
            raise ValueError("Joint address eligibility does not match the padded M/E inventories.")
        if normalized_query is None:
            normalized_query = self._routing_features(qstates)
        keys_by_role = {
            "M": prepared._consistent_normalized_member_keys_M,
            "E": prepared._consistent_normalized_member_keys_E,
        }
        outputs = []
        for role, mask, count in (
            ("M", address_allowed[..., :modules], modules),
            ("E", address_allowed[..., modules:], environments),
        ):
            scores = qstates.new_zeros(batch, queries, count)
            rows = self._pair_rows(mask)
            if rows.numel():
                b, q, member = rows.unbind(-1)
                features = torch.cat((
                    normalized_query[b, q],
                    keys_by_role[role][b, member],
                    self._pair_geometry(scene, query, role, rows),
                ), -1)
                scores = scores.index_put(
                    (b, q, member), self.member_router.proposal_M(features).squeeze(-1),
                )
            outputs.append(scores)
        return torch.cat(outputs, dim=-1), outputs[0], outputs[1]

    def _member_base_logits(
        self, prepared, query, qstates, source, candidates, score_mode, *, normalized_query=None,
    ):
        if score_mode != "learned":
            return super()._member_base_logits(prepared, query, qstates, source, candidates, score_mode)
        keys = (prepared._consistent_normalized_member_keys_M if source == "M"
                else prepared._consistent_normalized_member_keys_E)
        output = qstates.new_zeros(query.batch_size, query.num_queries, keys.shape[1])
        rows = self._pair_rows(candidates)
        if not rows.numel():
            return output, 0
        if normalized_query is None:
            normalized_query = self._routing_features(qstates)
        b, q, member = rows.unbind(-1)
        features = torch.cat((
            normalized_query[b, q], keys[b, member],
            self._pair_geometry(prepared.scene, query, source, rows),
        ), -1)
        scorer = self.member_router.member_M if source == "M" else self.member_router.member_E
        output = output.index_put((b, q, member), scorer(features).squeeze(-1))
        return output, len(rows)


class _UnfactoredAffineScorerCore(_UnfactoredScorerMixin, ConsistentJointMemberRoutingCore):
    pass


class _UnfactoredAffineScorerB(_UnfactoredScorerMixin, OrganizationConditionedMemberRoutingCore):
    pass


def test_candidate_a_reuses_joint_state_and_config_identity() -> None:
    torch.manual_seed(1913)
    original = JointMemberRoutingCore(
        _base_config(), input_dataset="synthetic-r13-5a-consistent", query_tile_size=3,
    )
    torch.manual_seed(1913)
    consistent = _core(tile_size=3)

    original_state = original.state_dict()
    consistent_state = consistent.state_dict()
    assert len(original_state) == len(consistent_state) == 190
    assert tuple(original_state) == tuple(consistent_state)
    for name, value in original_state.items():
        torch.testing.assert_close(value, consistent_state[name], rtol=0, atol=0)
    assert consistent.architecture_name == "R13-5A-ME-ConsistentJoint95"
    assert ConsistentJointMemberRoutingCore.from_config(consistent.get_config()).get_config() == consistent.get_config()


def test_joint_contribution_uses_one_global_normalizer_and_stable_common_offset() -> None:
    prior = torch.tensor([[[0.5, 0.25, 0.25]]], dtype=torch.float64)
    ell = torch.tensor([[[0.0, math.log(2.0), math.log(4.0)]]], dtype=torch.float64)
    allowed = torch.ones_like(prior, dtype=torch.bool)

    contribution, log_weight, offset = joint_likelihood_contribution(prior, ell, allowed)
    torch.testing.assert_close(contribution, torch.tensor([[[0.25, 0.25, 0.5]]], dtype=torch.float64))
    torch.testing.assert_close(contribution.sum(-1), torch.ones((1, 1), dtype=torch.float64))
    torch.testing.assert_close(log_weight, torch.tensor([[[-math.log(2), -math.log(2), 0.0]]], dtype=torch.float64))
    torch.testing.assert_close(offset, torch.tensor([[0.0]], dtype=torch.float64))

    shifted, _, shifted_offset = joint_likelihood_contribution(prior, ell + 900.0, allowed)
    torch.testing.assert_close(shifted, contribution, rtol=0, atol=1e-12)
    torch.testing.assert_close(shifted_offset, offset + 900.0, rtol=0, atol=1e-12)


def test_joint_likelihood_is_typed_permutation_equivariant_and_empty_safe() -> None:
    prior = torch.tensor([[[0.2, 0.3, 0.5]]])
    ell = torch.tensor([[[0.3, -0.7, 0.1]]])
    allowed = torch.tensor([[[True, True, True]]])
    expected, _, _ = joint_likelihood_contribution(prior, ell, allowed)
    permutation = torch.tensor([2, 0, 1])
    actual, _, _ = joint_likelihood_contribution(prior[..., permutation], ell[..., permutation],
                                                 allowed[..., permutation])
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(actual[..., inverse], expected)

    empty, empty_log, empty_offset = joint_likelihood_contribution(
        prior[..., :0], ell[..., :0], allowed[..., :0],
    )
    assert empty.shape == empty_log.shape == (1, 1, 0)
    assert empty_offset.shape == (1, 1)
    assert torch.isfinite(empty_offset).all()


def test_candidate_prefix_boundary_and_ties_use_role_then_original_id() -> None:
    weights = torch.tensor([[[0.5, 0.25, 0.25]]])
    allowed = torch.ones_like(weights, dtype=torch.bool)
    module_ids = torch.tensor([[10]])
    environment_ids = torch.tensor([[300, 100]])
    selected = joint_member_prefix_mask(weights, allowed, module_ids, environment_ids, 0.75)
    assert selected.tolist() == [[[True, False, True]]]

    tied = torch.tensor([[[0.5, 0.5]]])
    tied_allowed = torch.ones_like(tied, dtype=torch.bool)
    tied_selected = joint_member_prefix_mask(
        tied, tied_allowed, torch.tensor([[900]]), torch.tensor([[2]]), 0.5,
    )
    # Typed ties are resolved by the declared M-then-E role order.
    assert tied_selected.tolist() == [[[True, False]]]


def test_relation_calibration_changes_row_strength_only_without_role_renormalizing() -> None:
    retrieval = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    row_mass = torch.tensor([[0.1, 0.9]], dtype=torch.float64)
    raw_coefficients = retrieval * row_mass[..., None]
    proposal = torch.tensor([[[0.5, 0.5]]], dtype=torch.float64)
    member_valid = torch.ones((1, 1, 2), dtype=torch.bool)

    normalized, _ = compose_joint_member_prior(
        proposal, retrieval, raw_coefficients, row_mass, member_valid, calibration="normalized",
    )
    row_strength, _ = compose_joint_member_prior(
        proposal, retrieval, raw_coefficients, row_mass, member_valid, calibration="row_strength",
    )
    torch.testing.assert_close(normalized, torch.tensor([[[0.5, 0.5]]], dtype=torch.float64))
    torch.testing.assert_close(row_strength, torch.tensor([[[0.1, 0.9]]], dtype=torch.float64))
    assert normalized.sum(-1).item() == pytest.approx(1.0)
    assert row_strength.sum(-1).item() == pytest.approx(1.0)


def test_score_calibration_diagnostic_separates_candidate_dependent_and_fixed_laws() -> None:
    scores = torch.tensor([[[0.5, 1.5, 4.0]]])
    prior = torch.tensor([[[0.6, 0.3, 0.1]]])
    full_candidates = torch.ones_like(prior, dtype=torch.bool)
    restricted_candidates = torch.tensor([[[True, True, False]]])

    inherited_full = ConsistentJointMemberRoutingCore._score_values(
        scores, prior, full_candidates, "inherited",
    )
    inherited_restricted = ConsistentJointMemberRoutingCore._score_values(
        scores, prior, restricted_candidates, "inherited",
    )
    bounded_full = ConsistentJointMemberRoutingCore._score_values(
        scores, prior, full_candidates, "bounded_tanh",
    )
    bounded_restricted = ConsistentJointMemberRoutingCore._score_values(
        scores, prior, restricted_candidates, "bounded_tanh",
    )
    assert not torch.allclose(inherited_full[..., :2], inherited_restricted[..., :2])
    torch.testing.assert_close(bounded_full[..., :2], bounded_restricted[..., :2])
    torch.testing.assert_close(bounded_full, 4.0 * torch.tanh(scores / 4.0))


def test_inherited_calibration_matches_legacy_role_reference_before_global_a_mass() -> None:
    torch.manual_seed(71)
    model = _core(tile_size=4, score_calibration="inherited")
    model.eval()
    prepared = model.prepare(_scene(3, 4))
    query = _query(2)
    fixed_m = torch.tensor([[[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]]])
    fixed_e = torch.tensor([[[1.0, 1.2, 1.4, 1.6], [1.1, 1.3, 1.5, 1.7]]])
    output = model.decode(
        prepared, query,
        intervention={"read_mode": "full", "fixed_member_logits_M": fixed_m,
                      "fixed_member_logits_E": fixed_e},
        return_trace=True,
    )
    trace = output.trace
    assert trace is not None
    ell_m = prior_weighted_scores(trace["member_logits_M"], trace["prior_P_M"], trace["candidate_mask_M"])
    ell_e = prior_weighted_scores(trace["member_logits_E"], trace["prior_P_E"], trace["candidate_mask_E"])
    expected_ell = torch.cat((ell_m, ell_e), dim=-1)
    actual_ell = torch.cat((trace["member_ell_M"], trace["member_ell_E"]), dim=-1)
    torch.testing.assert_close(actual_ell, expected_ell)
    expected_b, _, _ = joint_likelihood_contribution(
        trace["prior_P_joint"], expected_ell, trace["joint_candidate_mask"],
    )
    torch.testing.assert_close(trace["member_joint_likelihood"], expected_b)


@pytest.mark.parametrize("retrieval_calibration,weighting_code", [("normalized", 0), ("row_strength", 1)])
def test_trace_attributes_applied_route_and_self_mass_once(retrieval_calibration, weighting_code) -> None:
    torch.manual_seed(77)
    model = _core(tile_size=4, retrieval_calibration=retrieval_calibration)
    model.eval()
    prepared = model.prepare(_scene(3, 5))
    output = model.decode(prepared, _query(3), return_trace=True)
    assert output.trace is not None
    trace = output.trace
    route_keys = (
        "address_block_mass_MM", "address_block_mass_ME", "address_block_mass_EM",
        "address_block_mass_EE", "address_block_mass_self_M",
    )
    assert all(key in trace for key in route_keys)
    attributed = sum(trace[key] for key in route_keys)
    torch.testing.assert_close(attributed, torch.ones_like(attributed), atol=2e-6, rtol=2e-6)
    assert trace["address_block_mass_provenance_code"].item() == 0
    assert trace["address_block_mass_weighting_code"].item() == weighting_code
    proposal = trace["proposal_weights_joint"]
    if retrieval_calibration == "row_strength":
        effective = proposal * prepared.retrieval_row_mass[:, None, :]
        effective = effective / effective.sum(-1, keepdim=True)
    else:
        effective = proposal
    blocks = {
        "MM": (0, 3, 0, 3),
        "ME": (0, 3, 3, 8),
        "EM": (3, 8, 0, 3),
        "EE": (3, 8, 3, 8),
    }
    for route, (row_start, row_stop, col_start, col_stop) in blocks.items():
        numerator = (
            prepared.H_exchange[:, row_start:row_stop, col_start:col_stop]
            * prepared.member_measure_joint[:, None, col_start:col_stop]
        )
        normalized_rows = numerator / prepared.retrieval_row_mass[:, row_start:row_stop, None]
        expected = (
            effective[..., row_start:row_stop] * normalized_rows.sum(-1)[:, None, :]
        ).sum(-1)
        torch.testing.assert_close(trace[f"address_block_mass_{route}"], expected)


def test_vanishing_e_mass_produces_vanishing_context_and_empty_role_stats_are_zero() -> None:
    values = torch.tensor([[[3.0], [7.0]]], dtype=torch.float64)
    strengths = torch.tensor([[0.4]], dtype=torch.float64)
    contexts = []
    for e_mass in (1.0e-2, 1.0e-5):
        prior = torch.tensor([[[1.0 - e_mass, e_mass]]], dtype=torch.float64)
        contribution, _, _ = joint_likelihood_contribution(
            prior, torch.zeros_like(prior), torch.ones_like(prior, dtype=torch.bool),
        )
        contexts.append(strengths[..., None] * contribution[..., 1:] * values[:, 1:, :])
    assert contexts[1].abs().max() < contexts[0].abs().max() * 0.002

    contribution = torch.zeros((1, 2, 3), dtype=torch.float64)
    scores = torch.randn_like(contribution)
    empty_role = torch.zeros_like(contribution, dtype=torch.bool)
    conditional, expectation, entropy = joint_role_statistics(
        contribution, scores, empty_role, torch.tensor([0]),
    )
    assert torch.equal(conditional, torch.zeros_like(conditional))
    assert torch.equal(expectation, torch.zeros_like(expectation))
    assert torch.equal(entropy, torch.zeros_like(entropy))


@pytest.mark.parametrize("module_count,environment_count", [(0, 0), (0, 4), (1, 0), (3, 5)])
def test_a_full_and_restricted_reads_keep_joint_mass_and_empty_role_contexts(module_count, environment_count) -> None:
    torch.manual_seed(17 + module_count + environment_count)
    model = _core(tile_size=4)
    model.eval()
    prepared = model.prepare(_scene(module_count, environment_count))
    query = _query(3)
    full = model.decode(prepared, query, intervention={"read_mode": "full"}, return_trace=True)
    restricted = model.decode(prepared, query, intervention={"read_mode": "restricted"}, return_trace=True)

    assert torch.isfinite(full.joint_fields).all()
    assert torch.isfinite(restricted.joint_fields).all()
    for output in (full, restricted):
        trace = output.trace
        assert trace is not None
        assert trace["H"].shape[-1] == module_count + environment_count
        assert trace["retrieval_R"].shape == trace["H"].shape
        assert trace["prior_P_joint"].shape == (1, 3, module_count + environment_count)
        assert trace["joint_candidate_mask"].shape == trace["prior_P_joint"].shape
        assert torch.equal(trace["original_module_ids"], prepared.scene.module_ids)
        assert torch.equal(trace["original_environment_ids"], prepared.scene.environment_ids)
        total = trace["member_joint_likelihood"].sum(-1)
        expected_total = torch.full_like(total, 1.0) if module_count + environment_count else torch.zeros_like(total)
        torch.testing.assert_close(total, expected_total, atol=1e-6, rtol=0)
        for role in ("M", "E"):
            pre = trace[f"member_contribution_pre_strength_{role}"]
            post = trace[f"member_contribution_{role}"]
            strength = trace[f"member_read_strength_{role}"]
            torch.testing.assert_close(post.sum(-1), strength * pre.sum(-1), atol=1e-6, rtol=1e-6)
        if module_count == 0:
            assert torch.count_nonzero(trace["context_M"]) == 0
            assert torch.count_nonzero(trace["member_contribution_M"]) == 0
        if environment_count == 0:
            assert torch.count_nonzero(trace["context_E"]) == 0
            assert torch.count_nonzero(trace["member_contribution_E"]) == 0
    assert full.work_counts["query_joint_contribution_normalization_rows"] == 3 * (module_count + environment_count)
    assert restricted.work_counts["query_joint_prefix_sort_rows"] == 3 * (module_count + environment_count)


def test_a_task_loss_reaches_both_original_member_readers_joint_head_and_all_applied_blocks() -> None:
    torch.manual_seed(18)
    model = _core(tile_size=3)
    model.train()
    prepared = model.prepare(_scene(3, 6))
    output = model.decode(prepared, _query(4), intervention={"read_mode": "restricted"})
    loss = output.joint_fields.square().mean()
    loss.backward()

    required = {
        "M member score": model.member_router.member_M[-1].weight,
        "E member score": model.member_router.member_E[-1].weight,
        "M read gate": model.typed_read_gates["M"][0].weight,
        "E read gate": model.typed_read_gates["E"][0].weight,
        "M original values": model.member_router.value_M.weight,
        "E original values": model.member_router.value_E.weight,
        "joint field head": model.member_router.joint_head.weight,
    }
    required.update({
        f"applied {route.upper()} relation": model.applied_incidence.routes[route].head_logits
        for route in ("mm", "em", "ee", "me")
    })
    norms = {name: 0.0 if param.grad is None else float(param.grad.norm()) for name, param in required.items()}
    assert all(norm > 1e-10 for norm in norms.values()), norms


def test_paired_decode_matches_separate_reads_and_charges_common_plan_once() -> None:
    torch.manual_seed(1918)
    model = _core(tile_size=4)
    model.eval()
    prepared = model.prepare(_scene(3, 5))
    query = _query(4)
    full, restricted, shared = model.decode_paired(prepared, query, return_trace=True)
    full_reference = model.decode(prepared, query, intervention={"read_mode": "full"}, return_trace=True)
    restricted_reference = model.decode(
        prepared, query, intervention={"read_mode": "restricted"}, return_trace=True,
    )

    torch.testing.assert_close(full.joint_fields, full_reference.joint_fields, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(restricted.joint_fields, restricted_reference.joint_fields, rtol=1e-6, atol=1e-7)
    assert full.trace is not None and restricted.trace is not None
    assert full_reference.trace is not None and restricted_reference.trace is not None
    torch.testing.assert_close(full.trace["full_eligible_mask"], restricted.trace["full_eligible_mask"])
    torch.testing.assert_close(full.trace["full_eligible_mask"], full_reference.trace["full_eligible_mask"])
    torch.testing.assert_close(
        restricted.trace["joint_candidate_mask"], restricted_reference.trace["joint_candidate_mask"],
    )
    assert shared["query_joint_address_score_rows"] == full_reference.work_counts["query_joint_address_score_rows"]
    assert shared["query_joint_address_score_rows"] == restricted_reference.work_counts["query_joint_address_score_rows"]
    assert full.work_counts["query_joint_address_score_rows"] == 0
    assert restricted.work_counts["query_joint_address_score_rows"] == 0
    assert shared["query_receiver_rows"] == full_reference.work_counts["query_receiver_rows"]
    assert full.work_counts["query_receiver_rows"] == restricted.work_counts["query_receiver_rows"] == 0
    assert set(shared) == set(_PAIRED_SHARED_COUNT_KEYS) | {"query_routing_normalization_rows"}
    for name in _PAIRED_SHARED_COUNT_KEYS:
        assert full.work_counts.get(name, 0) == restricted.work_counts.get(name, 0) == 0
        assert shared[name] == full_reference.work_counts[name]

    paired_routing_norm = (
        full.work_counts["query_routing_normalization_rows"]
        + restricted.work_counts["query_routing_normalization_rows"]
        + shared["query_routing_normalization_rows"]
    )
    separate_routing_norm = (
        full_reference.work_counts["query_routing_normalization_rows"]
        + restricted_reference.work_counts["query_routing_normalization_rows"]
    )
    assert paired_routing_norm == (
        separate_routing_norm - shared["query_routing_normalization_rows"]
    )


def test_paired_count_inventory_preserves_derived_subclass_counters() -> None:
    model = _core()
    extra = "query_B_view_member_contraction_rows"
    count_names = model._paired_count_names((*model._additional_decode_count_keys, extra))
    full_counts = {name: 0 for name in _PAIRED_SHARED_COUNT_KEYS}
    restricted_counts = dict(full_counts)
    full_counts[extra] = 13
    restricted_counts[extra] = 5
    full_branch, restricted_branch, shared = model._split_paired_counts(
        full_counts, restricted_counts, {"routing_normalization_shared_rows": 0}, count_names,
    )
    assert full_branch[extra] == 13
    assert restricted_branch[extra] == 5
    assert extra not in shared


def test_paired_checkpoint_backward_matches_separate_full_and_restricted_calls() -> None:
    torch.manual_seed(1919)
    paired_model = _core(tile_size=2)
    separate_model = _core(tile_size=2)
    separate_model.load_state_dict(paired_model.state_dict())
    paired_model.train()
    separate_model.train()
    scene = _scene(3, 5)
    query = _query(5)

    paired_prepared = paired_model.prepare(scene)
    paired_full, paired_restricted, shared = paired_model.decode_paired(paired_prepared, query)
    paired_loss = (paired_full.joint_fields.square().mean()
                   + 0.7 * paired_restricted.joint_fields.square().mean())
    paired_loss.backward()

    separate_prepared = separate_model.prepare(scene)
    separate_full = separate_model.decode(separate_prepared, query, intervention={"read_mode": "full"})
    separate_restricted = separate_model.decode(
        separate_prepared, query, intervention={"read_mode": "restricted"},
    )
    separate_loss = (separate_full.joint_fields.square().mean()
                     + 0.7 * separate_restricted.joint_fields.square().mean())
    separate_loss.backward()

    torch.testing.assert_close(paired_full.joint_fields, separate_full.joint_fields, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        paired_restricted.joint_fields, separate_restricted.joint_fields, rtol=1e-6, atol=1e-7,
    )
    for (paired_name, paired_parameter), (separate_name, separate_parameter) in zip(
            paired_model.named_parameters(), separate_model.named_parameters()):
        assert paired_name == separate_name
        if paired_parameter.grad is None or separate_parameter.grad is None:
            assert paired_parameter.grad is None and separate_parameter.grad is None, paired_name
        else:
            torch.testing.assert_close(
                paired_parameter.grad, separate_parameter.grad, rtol=2e-5, atol=2e-7,
                msg=lambda msg: f"{paired_name}: {msg}",
            )
    assert shared["query_joint_address_score_rows"] > 0
    assert paired_full.work_counts["query_joint_prefix_sort_rows"] == 0
    assert paired_restricted.work_counts["query_joint_prefix_sort_rows"] > 0


class _ReferenceKeyNormalizationCore(ConsistentJointMemberRoutingCore):
    """Test oracle retaining the original per-query/per-key address LN path."""

    def prepare(self, scene, **options):
        return JointMemberRoutingCore.prepare(self, scene, **options)

    def _validate_prepared_state(self, prepared):
        JointMemberRoutingCore._validate_prepared_state(self, prepared)

    def _address_logits(self, prepared, query, qstates, address_allowed, *, normalized_query=None):
        return JointMemberRoutingCore._address_logits(
            self, prepared, query, qstates, address_allowed,
        )

    def _member_base_logits(self, prepared, query, qstates, source, candidates, score_mode, *, normalized_query=None):
        return JointMemberRoutingCore._member_base_logits(
            self, prepared, query, qstates, source, candidates, score_mode,
        )


def _masked_scene(module_count: int, environment_count: int, pad_modules: int, pad_environments: int):
    scene = _scene(module_count + pad_modules, environment_count + pad_environments)
    if pad_modules:
        scene.module_mask[:, module_count:] = False
    if pad_environments:
        scene.environment_mask[:, environment_count:] = False
    return scene


@pytest.mark.parametrize(
    "module_count,environment_count,pad_modules,pad_environments",
    [(1, 0, 0, 0), (3, 5, 1, 2), (0, 4, 2, 1), (4, 0, 1, 0)],
)
@pytest.mark.parametrize("read_mode", ["full", "restricted"])
def test_affine_scorers_match_unfactored_forward_and_gradients(
    module_count, environment_count, pad_modules, pad_environments, read_mode,
) -> None:
    torch.manual_seed(2129 + module_count + environment_count)
    optimized = _core(tile_size=4)
    reference = _UnfactoredAffineScorerCore(
        _base_config(), address_mode="ME", input_dataset="synthetic-r13-5a-consistent",
        query_tile_size=4,
    )
    reference.load_state_dict(optimized.state_dict(), strict=True)
    scene = _masked_scene(module_count, environment_count, pad_modules, pad_environments)
    query = replace(_query(4), query_mask=torch.tensor([[True, True, False, True]]))
    outputs = []
    parameter_gradients = []
    prepared_gradients = []

    for model in (optimized, reference):
        model.train()
        model.zero_grad(set_to_none=True)
        prepared = model.prepare(scene)
        for name in ("member_keys_M", "member_keys_E", "member_values_M", "member_values_E",
                     "H_exchange", "H_self"):
            value = getattr(prepared, name, None)
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.retain_grad()
        output = model.decode(
            prepared, query, intervention={"read_mode": read_mode}, return_trace=True,
        )
        trace = output.trace
        assert trace is not None
        loss = (
            output.joint_fields.square().mean()
            + 0.02 * trace["proposal_weights_joint"].square().mean()
            + 0.01 * trace["prior_P_joint"].square().mean()
            + 0.01 * trace["member_contribution_M"].square().mean()
            + 0.01 * trace["member_contribution_E"].square().mean()
            + 0.01 * trace["context_M"].square().mean()
            + 0.01 * trace["context_E"].square().mean()
        )
        loss.backward()
        outputs.append(output)
        parameter_gradients.append({
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        })
        prepared_gradients.append({
            name: None if getattr(prepared, name, None) is None or getattr(prepared, name).grad is None
            else getattr(prepared, name).grad.detach().clone()
            for name in ("member_keys_M", "member_keys_E", "member_values_M", "member_values_E",
                         "H_exchange", "H_self")
        })

    torch.testing.assert_close(outputs[0].joint_fields, outputs[1].joint_fields, rtol=3e-5, atol=3e-6)
    assert outputs[0].trace is not None and outputs[1].trace is not None
    for name in (
        "proposal_weights_joint", "prior_P_joint", "full_eligible_mask", "joint_candidate_mask",
        "member_contribution_pre_strength_M", "member_contribution_pre_strength_E",
        "member_contribution_M", "member_contribution_E", "context_M", "context_E",
    ):
        left, right = outputs[0].trace[name], outputs[1].trace[name]
        if left.dtype == torch.bool:
            assert torch.equal(left, right), name
        else:
            torch.testing.assert_close(left, right, rtol=4e-5, atol=4e-6, msg=name)

    assert parameter_gradients[0].keys() == parameter_gradients[1].keys()
    for name in parameter_gradients[0]:
        left, right = parameter_gradients[0][name], parameter_gradients[1][name]
        if left is None or right is None:
            assert left is right, name
        else:
            torch.testing.assert_close(left, right, rtol=5e-4, atol=5e-6, msg=name)
    for name in prepared_gradients[0]:
        left, right = prepared_gradients[0][name], prepared_gradients[1][name]
        if left is None or right is None:
            assert left is right, name
        else:
            torch.testing.assert_close(left, right, rtol=5e-4, atol=5e-6, msg=name)


def test_affine_member_score_ties_preserve_joint_prefix_role_and_id_order() -> None:
    torch.manual_seed(2141)
    optimized = _core(tile_size=4)
    reference = _UnfactoredAffineScorerCore(
        _base_config(), address_mode="ME", input_dataset="synthetic-r13-5a-consistent",
        query_tile_size=4,
    )
    reference.load_state_dict(optimized.state_dict(), strict=True)
    for model in (optimized, reference):
        with torch.no_grad():
            for scorer in (model.member_router.proposal_M, model.member_router.member_M,
                           model.member_router.member_E):
                scorer[0].weight.zero_()
                scorer[0].bias.zero_()
                scorer[2].weight.zero_()
                scorer[2].bias.zero_()

    scene = _scene(2, 2)
    scene = replace(
        scene,
        module_ids=torch.tensor([[900, 2]]),
        environment_ids=torch.tensor([[700, 3]]),
        environment_measure=torch.ones_like(scene.environment_measure),
    )
    query = _query(1)
    raw_scores = []
    for model in (optimized, reference):
        prepared = model.prepare(scene)
        qstates = model._query_states(prepared, query)
        allowed = torch.ones((1, 1, 4), dtype=torch.bool)
        address, _, _ = model._address_logits(prepared, query, qstates, allowed)
        assert torch.equal(address, torch.zeros_like(address))
        for role in ("M", "E"):
            logits, scored_rows = model._member_base_logits(
                prepared, query, qstates, role, allowed[..., :2] if role == "M" else allowed[..., 2:],
                "learned",
            )
            assert scored_rows == 2
            assert torch.equal(logits, torch.zeros_like(logits))
            raw_scores.append(logits)

    prior = torch.full((1, 1, 4), 0.25)
    tied_allowed = torch.ones_like(prior, dtype=torch.bool)
    contributions = [
        joint_likelihood_contribution(prior, torch.cat(raw_scores[i:i + 2], dim=-1), tied_allowed)[0]
        for i in (0, 2)
    ]
    for contribution in contributions:
        assert torch.equal(contribution, prior)
        selected = joint_member_prefix_mask(
            contribution, tied_allowed, scene.module_ids, scene.environment_ids, 0.25,
        )
        # All four masses tie: M precedes E, then the smaller original M ID wins.
        assert selected.tolist() == [[[False, True, False, False]]]
    assert torch.equal(contributions[0], contributions[1])


def test_affine_key_terms_remain_live_validated_and_b_state_identity_is_unchanged() -> None:
    torch.manual_seed(2153)
    a_model = _core(tile_size=2)
    b_model = OrganizationConditionedMemberRoutingCore(
        _base_config(), address_mode="ME", input_dataset="synthetic-r13-5a-consistent",
        query_tile_size=2,
    )
    assert tuple(a_model.state_dict()) == tuple(b_model.state_dict())
    assert len(a_model.state_dict()) == len(b_model.state_dict()) == 190
    b_model.load_state_dict(a_model.state_dict(), strict=True)
    scene = _scene(3, 4)
    prepared = b_model.prepare(scene)
    assert set(prepared._consistent_affine_key_terms) == {
        "proposal_M_M", "proposal_M_E", "member_M_M", "member_E_E",
    }
    for term in prepared._consistent_affine_key_terms.values():
        assert term.grad_fn is not None
    b_reference = _UnfactoredAffineScorerB(
        _base_config(), address_mode="ME", input_dataset="synthetic-r13-5a-consistent",
        query_tile_size=2,
    )
    b_reference.load_state_dict(b_model.state_dict(), strict=True)
    query = _query(2)
    b_output = b_model.decode(prepared, query, intervention={"read_mode": "restricted"}, return_trace=True)
    reference_output = b_reference.decode(
        b_reference.prepare(scene), query,
        intervention={"read_mode": "restricted"}, return_trace=True,
    )
    assert torch.isfinite(b_output.joint_fields).all()
    torch.testing.assert_close(b_output.joint_fields, reference_output.joint_fields, rtol=4e-5, atol=4e-6)
    assert b_output.trace is not None and reference_output.trace is not None
    assert torch.equal(
        b_output.trace["joint_candidate_mask"], reference_output.trace["joint_candidate_mask"],
    )

    stale = b_model.prepare(_scene(2, 3))
    with torch.no_grad():
        stale._consistent_affine_key_terms["member_M_M"].add_(1.)
    with pytest.raises(ValueError, match="affine scorer key terms changed"):
        b_model.validate_prepared_state(stale, _query(1))

    stale_weight = b_model.prepare(_scene(2, 3))
    with torch.no_grad():
        b_model.member_router.member_E[0].weight.add_(0.01)
    with pytest.raises(ValueError, match="parameters changed after scene preparation"):
        b_model.validate_prepared_state(stale_weight, _query(1))


def test_affine_factoring_preserves_paired_checkpoint_backward() -> None:
    torch.manual_seed(2161)
    optimized = _core(tile_size=2)
    reference = _UnfactoredAffineScorerCore(
        _base_config(), address_mode="ME", input_dataset="synthetic-r13-5a-consistent",
        query_tile_size=2,
    )
    reference.load_state_dict(optimized.state_dict(), strict=True)
    scene, query = _scene(5, 4), _query(5)
    outputs = []
    gradients = []
    for model in (optimized, reference):
        model.train()
        model.zero_grad(set_to_none=True)
        prepared = model.prepare(scene)
        full, restricted, _ = model.decode_paired(prepared, query)
        loss = full.joint_fields.square().mean() + 0.7 * restricted.joint_fields.square().mean()
        loss.backward()
        outputs.append((full.joint_fields.detach().clone(), restricted.joint_fields.detach().clone()))
        gradients.append({name: None if parameter.grad is None else parameter.grad.detach().clone()
                          for name, parameter in model.named_parameters()})
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=3e-5, atol=3e-6)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        left, right = gradients[0][name], gradients[1][name]
        if left is None or right is None:
            assert left is right, name
        else:
            torch.testing.assert_close(left, right, rtol=5e-4, atol=5e-6, msg=name)


@pytest.mark.parametrize(
    "module_count,environment_count,pad_modules,pad_environments",
    [(1, 0, 0, 0), (3, 5, 1, 2), (12, 7, 3, 4)],
)
@pytest.mark.parametrize("read_mode", ["full", "restricted"])
def test_scene_local_normalized_keys_preserve_full_forward_trace_and_gradients(
    module_count, environment_count, pad_modules, pad_environments, read_mode,
) -> None:
    torch.manual_seed(1927 + module_count + environment_count)
    optimized = _core(tile_size=8)
    reference = _ReferenceKeyNormalizationCore(
        _base_config(), address_mode="ME", input_dataset="synthetic-r13-5a-consistent", query_tile_size=8,
    )
    reference.load_state_dict(optimized.state_dict())
    optimized.eval()
    reference.eval()
    scene = _masked_scene(module_count, environment_count, pad_modules, pad_environments)
    query = _query(4)
    query = replace(query, query_mask=torch.tensor([[True, True, False, True]]))

    outputs = []
    gradient_sets = []
    prepared_scenes = []
    for model in (optimized, reference):
        model.zero_grad(set_to_none=True)
        prepared = model.prepare(scene)
        prepared_scenes.append(prepared)
        prepared.member_keys_M.retain_grad()
        prepared.member_keys_E.retain_grad()
        output = model.decode(prepared, query, intervention={"read_mode": read_mode}, return_trace=True)
        trace = output.trace
        assert trace is not None
        loss = (
            output.joint_fields.square().mean()
            + 0.03 * trace["proposal_weights_joint"].square().mean()
            + 0.02 * trace["prior_P_joint"].square().mean()
            + 0.01 * trace["context_M"].square().mean()
            + 0.01 * trace["context_E"].square().mean()
            + 0.001 * trace["H"].square().mean()
        )
        loss.backward()
        outputs.append(output)
        gradient_sets.append({name: None if parameter.grad is None else parameter.grad.detach().clone()
                              for name, parameter in model.named_parameters()})
        gradient_sets[-1]["__prepared_member_keys_M"] = (
            None if prepared.member_keys_M.grad is None else prepared.member_keys_M.grad.detach().clone()
        )
        gradient_sets[-1]["__prepared_member_keys_E"] = (
            None if prepared.member_keys_E.grad is None else prepared.member_keys_E.grad.detach().clone()
        )

    torch.testing.assert_close(outputs[0].joint_fields, outputs[1].joint_fields, rtol=3e-5, atol=3e-6)
    assert outputs[0].trace is not None and outputs[1].trace is not None
    for key in (
        "proposal_weights_joint", "prior_P_joint", "joint_candidate_mask", "context_M", "context_E",
        "H", "member_contribution_M", "member_contribution_E",
    ):
        left, right = outputs[0].trace[key], outputs[1].trace[key]
        if left.dtype == torch.bool or not left.is_floating_point():
            assert torch.equal(left, right), key
        else:
            torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-7, msg=key)
    assert gradient_sets[0].keys() == gradient_sets[1].keys()
    for name in gradient_sets[0]:
        left, right = gradient_sets[0][name], gradient_sets[1][name]
        if left is None or right is None:
            assert left is right, name
        else:
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-7, msg=name)
    assert prepared_scenes[0].work_counts["normalized_member_key_rows_M"] == scene.batch_size * scene.num_modules
    assert prepared_scenes[0].work_counts["normalized_member_key_rows_E"] == scene.batch_size * scene.num_environment_tokens
    assert outputs[0].work_counts["query_routing_feature_normalization_rows"] == query.batch_size * query.num_queries


def test_prepared_normalized_member_keys_are_live_and_validated() -> None:
    model = _core()
    prepared = model.prepare(_scene(2, 3))
    torch.testing.assert_close(
        prepared._consistent_normalized_member_keys_M,
        model._routing_features(prepared.member_keys_M),
        rtol=0,
        atol=0,
    )
    assert prepared._consistent_normalized_member_keys_M.grad_fn is not None
    query = _query(1)
    with torch.no_grad():
        prepared._consistent_normalized_member_keys_M.add_(1.0)
    with pytest.raises(ValueError, match="normalized member-key views changed"):
        model.validate_prepared_state(prepared, query)


def test_scene_local_normalized_keys_survive_tiled_checkpoint_backward() -> None:
    torch.manual_seed(1951)
    optimized = _core(tile_size=2)
    reference = _ReferenceKeyNormalizationCore(
        _base_config(), address_mode="ME", input_dataset="synthetic-r13-5a-consistent", query_tile_size=2,
    )
    reference.load_state_dict(optimized.state_dict())
    optimized.train()
    reference.train()
    scene = _masked_scene(5, 4, 2, 1)
    query = _query(5)
    outputs = []
    gradients = []
    for model in (optimized, reference):
        model.zero_grad(set_to_none=True)
        prepared = model.prepare(scene)
        output = model.decode(prepared, query, intervention={"read_mode": "restricted"})
        loss = output.joint_fields.square().mean()
        loss.backward()
        outputs.append(output.joint_fields.detach().clone())
        gradients.append({name: None if parameter.grad is None else parameter.grad.detach().clone()
                          for name, parameter in model.named_parameters()})
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-6, atol=2e-7)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        left, right = gradients[0][name], gradients[1][name]
        if left is None or right is None:
            assert left is right, name
        else:
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-7, msg=name)
