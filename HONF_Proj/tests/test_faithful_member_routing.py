from __future__ import annotations

import torch
import pytest

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.interaction_reused_member_routing.consistent import ConsistentJointMemberRoutingCore
from honf_forward_core.interaction_reused_member_routing.faithful import (
    FaithfulMECore,
    original_role_conditional_likelihood,
    retained_role_prior_fraction,
)
from honf_forward_core.interaction_reused_member_routing.joint import JointMemberRoutingCore


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


def _scene() -> SceneBatch:
    torch.manual_seed(7601)
    batch, modules, environments = 2, 3, 4
    module_mask = torch.tensor([[True, True, True], [True, False, False]])
    environment_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    module_coordinates = torch.tensor([
        [[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]],
        [[0.2, -0.1], [0.0, 0.0], [0.0, 0.0]],
    ])
    environment_coordinates = torch.tensor([
        [[0.0, 0.4], [0.4, 0.4], [0.8, 0.4], [1.2, 0.4]],
        [[0.1, 0.5], [0.8, 0.5], [0.0, 0.0], [0.0, 0.0]],
    ])
    return SceneBatch(
        module_features=torch.randn(batch, modules, 9),
        environment_features=torch.randn(batch, environments, 6),
        module_coordinates=module_coordinates,
        environment_coordinates=environment_coordinates,
        module_mask=module_mask,
        environment_mask=environment_mask,
        module_measure=torch.ones(batch, modules),
        environment_measure=torch.tensor([[0.4, 0.5, 0.6, 0.7], [0.3, 0.8, 0.0, 0.0]]),
        module_ids=torch.tensor([[27, 34, 58], [81, -1, -1]]),
        environment_ids=torch.tensor([[110, 114, 121, 130], [210, 211, -1, -1]]),
        local_radius=torch.full((batch, modules), 0.2),
        operating_features=torch.randn(batch, 12),
        coordinate_origin=torch.zeros(batch, 2),
        coordinate_scale=torch.ones(batch, 2),
        module_measure_reference=torch.ones(batch),
        environment_measure_reference=torch.ones(batch),
        scene_key="r13-6-faithful-test-scene",
        normalizer_key="r13-6-faithful-test-normalizer",
        coordinate_frame_key="r13-6-faithful-test-frame",
        module_characteristic_lengths=torch.full((batch, modules, 2), 0.12),
        environment_characteristic_lengths=torch.full((batch, environments, 2), 0.1),
    )


def _query() -> QueryBatch:
    torch.manual_seed(7602)
    batch, queries = 2, 4
    x = torch.linspace(0.05, 1.1, queries)
    coordinates = torch.stack((x, torch.linspace(-0.1, 0.2, queries)), dim=-1)
    return QueryBatch(
        coordinates=coordinates[None].expand(batch, -1, -1).clone(),
        query_mask=torch.ones(batch, queries, dtype=torch.bool),
        query_ids=torch.arange(500, 500 + queries)[None].expand(batch, -1).clone(),
        scene_key="r13-6-faithful-test-scene",
        coordinate_frame_key="r13-6-faithful-test-frame",
    )


def _models(*, address_mode: str = "ME", tile_size: int = 3, retained_prior_law: str = "none"):
    config = _base_config()
    original = JointMemberRoutingCore(
        config,
        address_mode=address_mode,
        input_dataset="synthetic-r13-6-faithful",
        query_tile_size=tile_size,
    )
    faithful = FaithfulMECore(
        config,
        address_mode=address_mode,
        input_dataset="synthetic-r13-6-faithful",
        query_tile_size=tile_size,
        retained_prior_law=retained_prior_law,
    )
    faithful.load_state_dict(original.state_dict(), strict=True)
    return original, faithful


class _LegacyConsistentReadHook(ConsistentJointMemberRoutingCore):
    """Explicit pre-hook Candidate A behavior used as a regression oracle."""

    def _role_read_contributions(
        self, marginal_contribution, member_prior, ell, candidate, *,
        full_candidate, proposal, retrieval_R, row_mass, module_count,
    ):
        del member_prior, ell, candidate, full_candidate, proposal, retrieval_R, row_mass, module_count
        return marginal_contribution, marginal_contribution, {}


def _assert_gradients_close(left, right, *, rtol: float = 5e-5, atol: float = 5e-6) -> None:
    left_parameters = dict(left.named_parameters())
    right_parameters = dict(right.named_parameters())
    assert left_parameters.keys() == right_parameters.keys()
    for name in left_parameters:
        left_gradient = left_parameters[name].grad
        right_gradient = right_parameters[name].grad
        assert (left_gradient is None) == (right_gradient is None), name
        if left_gradient is not None:
            torch.testing.assert_close(left_gradient, right_gradient, rtol=rtol, atol=atol, msg=name)


def test_faithful_config_is_strict_and_keeps_original_parameter_inventory() -> None:
    torch.manual_seed(13)
    original, faithful = _models()
    assert tuple(original.state_dict()) == tuple(faithful.state_dict())
    assert all(torch.equal(original.state_dict()[name], faithful.state_dict()[name])
               for name in original.state_dict())
    config = faithful.get_config()
    assert config["schema"] == "interaction_reused_faithful_joint_member_routing_v1"
    assert config["architecture"] == "R13-6-O-ME-FaithfulJoint95"
    assert config["retained_prior_law"] == "none"
    assert FaithfulMECore.from_config(config).get_config() == config
    k_model = FaithfulMECore(_base_config(), retained_prior_law="K")
    assert FaithfulMECore.from_config(k_model.get_config()).retained_prior_law == "K"

    bad = dict(config, unexpected_switch=True)
    with pytest.raises(ValueError, match="inventory/schema"):
        FaithfulMECore.validate_config(bad)


def test_new_role_hook_preserves_candidate_a_fields_and_gradients() -> None:
    config = _base_config()
    current = ConsistentJointMemberRoutingCore(
        config, input_dataset="synthetic-r13-6-faithful", query_tile_size=4,
    )
    legacy = _LegacyConsistentReadHook(
        config, input_dataset="synthetic-r13-6-faithful", query_tile_size=4,
    )
    legacy.load_state_dict(current.state_dict(), strict=True)
    current.eval()
    legacy.eval()
    scene, query = _scene(), _query()
    current_output = current.decode(current.prepare(scene), query, return_trace=True)
    legacy_output = legacy.decode(legacy.prepare(scene), query, return_trace=True)
    torch.testing.assert_close(current_output.joint_fields, legacy_output.joint_fields, rtol=0, atol=0)
    assert current_output.trace is not None and legacy_output.trace is not None
    torch.testing.assert_close(current_output.trace["member_joint_likelihood"],
                               legacy_output.trace["member_joint_likelihood"], rtol=0, atol=0)
    current_output.joint_fields.square().sum().backward()
    legacy_output.joint_fields.square().sum().backward()
    _assert_gradients_close(current, legacy, rtol=0, atol=0)


def test_original_role_law_and_retained_fraction_cover_empty_and_singleton_roles() -> None:
    prior = torch.tensor([[[0.1, 0.6, 0.3], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    ell = torch.tensor([[[0.2, -0.4, 0.8], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    candidate = torch.tensor([[[True, False, True], [False, False, False]]])
    actual = original_role_conditional_likelihood(prior, ell, candidate)
    cp = torch.tensor([[[0.25, 0.0, 0.75], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    raw = torch.where(candidate, cp * ell.exp(), torch.zeros_like(cp))
    expected = torch.where(raw.sum(-1, keepdim=True) > 0,
                           raw / raw.sum(-1, keepdim=True).clamp_min(torch.finfo(raw.dtype).tiny),
                           torch.zeros_like(raw))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(actual[:, 1], torch.zeros_like(actual[:, 1]))

    full = torch.tensor([[[True, True, True], [True, False, False]]])
    selected = torch.tensor([[[True, True, True], [True, False, False]]])
    fraction = retained_role_prior_fraction(prior, selected, full)
    torch.testing.assert_close(fraction, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    singleton_prior = torch.tensor([[[0.37]]], dtype=torch.float64)
    singleton = torch.ones_like(singleton_prior, dtype=torch.bool)
    singleton_fraction = retained_role_prior_fraction(singleton_prior, singleton, singleton)
    assert singleton_fraction.item() == 1.0


@pytest.mark.parametrize("read_mode", ["full", "restricted"])
def test_faithful_fields_traces_and_parameter_gradients_match_original_me(read_mode: str) -> None:
    original, faithful = _models(tile_size=4)
    original.eval()
    faithful.eval()
    scene, query = _scene(), _query()
    original_prepared = original.prepare(scene)
    faithful_prepared = faithful.prepare(scene)

    original_output = original.decode(
        original_prepared, query, intervention={"read_mode": read_mode}, return_trace=True,
    )
    faithful_output = faithful.decode(
        faithful_prepared, query, intervention={"read_mode": read_mode}, return_trace=True,
    )
    torch.testing.assert_close(faithful_output.joint_fields, original_output.joint_fields,
                               rtol=4e-5, atol=5e-6)
    assert faithful_output.trace is not None and original_output.trace is not None
    for key in (
        "proposal_weights_joint", "prior_P", "candidate_mask_joint", "candidate_ids_M",
        "candidate_ids_E", "member_relative_weights_M", "member_relative_weights_E",
        "member_weights_M", "member_weights_E", "member_read_strength_M",
        "member_read_strength_E", "context_M", "context_E",
    ):
        torch.testing.assert_close(faithful_output.trace[key], original_output.trace[key],
                                   rtol=5e-5, atol=6e-6, msg=key)

    original_output.joint_fields.square().sum().backward()
    faithful_output.joint_fields.square().sum().backward()
    _assert_gradients_close(original, faithful)
    assert faithful_output.work_counts["query_M_member_softmax_rows"] == int(
        faithful_output.trace["candidate_mask_M"].sum()
    )
    assert faithful_output.work_counts["query_E_member_softmax_rows"] == int(
        faithful_output.trace["candidate_mask_E"].sum()
    )


def test_faithful_paired_nonreentrant_replay_matches_separate_branches_and_gradients() -> None:
    _, paired = _models(tile_size=2)
    _, separate = _models(tile_size=2)
    separate.load_state_dict(paired.state_dict(), strict=True)
    paired.train()
    separate.train()
    scene, query = _scene(), _query()
    paired_prepared = paired.prepare(scene)
    separate_prepared = separate.prepare(scene)

    full, restricted, shared = paired.decode_paired(paired_prepared, query)
    full_separate = separate.decode(separate_prepared, query, intervention={"read_mode": "full"})
    restricted_separate = separate.decode(
        separate_prepared, query, intervention={"read_mode": "restricted"},
    )
    torch.testing.assert_close(full.joint_fields, full_separate.joint_fields, rtol=4e-5, atol=5e-6)
    torch.testing.assert_close(restricted.joint_fields, restricted_separate.joint_fields,
                               rtol=4e-5, atol=5e-6)
    assert shared["query_joint_address_score_rows"] > 0
    assert "query_M_member_softmax_rows" in full.work_counts
    assert "query_E_member_softmax_rows" in restricted.work_counts

    (full.joint_fields.square().sum() + restricted.joint_fields.square().sum()).backward()
    (full_separate.joint_fields.square().sum() + restricted_separate.joint_fields.square().sum()).backward()
    _assert_gradients_close(paired, separate)


def test_faithful_masks_empty_and_pure_roles_exclude_padded_members() -> None:
    original, faithful = _models(tile_size=4)
    del original
    faithful.eval()
    scene, query = _scene(), _query()
    prepared = faithful.prepare(scene)
    modules, environments = scene.num_modules, scene.num_environment_tokens
    full_output = faithful.decode(prepared, query, intervention={"read_mode": "full"}, return_trace=True)
    assert full_output.trace is not None
    full_eligible = full_output.trace["full_eligible_mask"]

    for role in ("M", "E", "empty"):
        candidate = torch.zeros_like(full_eligible)
        if role == "M":
            candidate[..., :modules] = full_eligible[..., :modules]
        elif role == "E":
            candidate[..., modules:] = full_eligible[..., modules:]
        output = faithful.decode(
            prepared, query,
            intervention={"fixed_candidate_mask_joint": candidate},
            return_trace=True,
        )
        assert output.trace is not None
        trace = output.trace
        assert torch.isfinite(output.joint_fields).all()
        assert torch.isfinite(trace["member_weights_M"]).all()
        assert torch.isfinite(trace["member_weights_E"]).all()
        if role in ("E", "empty"):
            assert not trace["candidate_mask_M"].any()
            assert not trace["member_weights_M"].any()
        if role in ("M", "empty"):
            assert not trace["candidate_mask_E"].any()
            assert not trace["member_weights_E"].any()
        assert not trace["candidate_mask_M"][~scene.module_mask[:, None, :].expand(-1, query.num_queries, -1)].any()
        eligible_e = scene.environment_mask & (scene.environment_measure > 0)
        assert not trace["candidate_mask_E"][~eligible_e[:, None, :].expand(-1, query.num_queries, -1)].any()
        assert (trace["candidate_ids_M"][:, -1] >= 0).all()
        assert (trace["candidate_ids_E"][:, -1] >= 0).all()


def test_k_full_read_is_identity_and_k_singleton_legal_role_is_unattenuated() -> None:
    _, original_law = _models(tile_size=4)
    k_model = FaithfulMECore(
        _base_config(), input_dataset="synthetic-r13-6-faithful", query_tile_size=4,
        retained_prior_law="K",
    )
    k_model.load_state_dict(original_law.state_dict(), strict=True)
    original_law.eval()
    k_model.eval()
    scene, query = _scene(), _query()
    reference = original_law.decode(original_law.prepare(scene), query,
                                    intervention={"read_mode": "full"}, return_trace=True)
    actual = k_model.decode(k_model.prepare(scene), query,
                            intervention={"read_mode": "full"}, return_trace=True)
    torch.testing.assert_close(actual.joint_fields, reference.joint_fields, rtol=0, atol=0)
    assert actual.trace is not None
    torch.testing.assert_close(actual.trace["member_retained_prior_fraction_M"],
                               torch.ones_like(actual.trace["member_retained_prior_fraction_M"]))
    torch.testing.assert_close(actual.trace["member_retained_prior_fraction_E"],
                               torch.ones_like(actual.trace["member_retained_prior_fraction_E"]))

    one_member_prior = torch.tensor([[[0.23]]])
    one_member = torch.ones_like(one_member_prior, dtype=torch.bool)
    p = original_role_conditional_likelihood(one_member_prior, torch.tensor([[[0.8]]]), one_member)
    fraction = retained_role_prior_fraction(one_member_prior, one_member, one_member)
    assert fraction.item() == 1.0
    assert (p * fraction[..., None]).item() == p.item() == 1.0
