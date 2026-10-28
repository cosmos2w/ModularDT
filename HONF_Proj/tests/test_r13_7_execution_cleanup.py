from __future__ import annotations

import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.content_hypergraph.query_chunks_v1 import slice_query_v1
from honf_forward_core.interaction_reused_member_routing.consistent import (
    ConsistentJointMemberRoutingCore,
    joint_likelihood_contribution,
)
from honf_forward_core.interaction_reused_member_routing.faithful import (
    FaithfulMECore,
    original_role_conditional_likelihood,
)
from honf_forward_core.interaction_reused_member_routing.typed_organization_core import (
    TypedOrganizationMemberRoutingCore,
)


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
    return SceneBatch(
        module_features=torch.randn(1, 2, 9),
        environment_features=torch.randn(1, 2, 6),
        module_coordinates=torch.tensor([[[0.0, 0.0], [0.5, 0.1]]]),
        environment_coordinates=torch.tensor([[[0.0, 0.4], [0.4, 0.4]]]),
        module_mask=torch.ones(1, 2, dtype=torch.bool),
        environment_mask=torch.ones(1, 2, dtype=torch.bool),
        module_measure=torch.ones(1, 2),
        environment_measure=torch.tensor([[0.4, 0.5]]),
        module_ids=torch.tensor([[27, 34]]),
        environment_ids=torch.tensor([[110, 114]]),
        local_radius=torch.full((1, 2), 0.2),
        operating_features=torch.randn(1, 12),
        coordinate_origin=torch.zeros(1, 2),
        coordinate_scale=torch.ones(1, 2),
        module_measure_reference=torch.ones(1),
        environment_measure_reference=torch.ones(1),
        scene_key="r13-7-execution-cleanup-scene",
        normalizer_key="r13-7-execution-cleanup-normalizer",
        coordinate_frame_key="r13-7-execution-cleanup-frame",
        module_characteristic_lengths=torch.full((1, 2, 2), 0.12),
        environment_characteristic_lengths=torch.full((1, 2, 2), 0.1),
    )


def _query() -> QueryBatch:
    x = torch.linspace(0.05, 1.1, 4)
    return QueryBatch(
        coordinates=torch.stack((x, torch.linspace(-0.1, 0.2, 4)), dim=-1)[None],
        query_mask=torch.ones(1, 4, dtype=torch.bool),
        query_ids=torch.arange(500, 504)[None],
        scene_key="r13-7-execution-cleanup-scene",
        coordinate_frame_key="r13-7-execution-cleanup-frame",
    )


def _faithful(*, tile_size: int = 3) -> FaithfulMECore:
    return FaithfulMECore(
        _base_config(), input_dataset="synthetic-r13-7-execution-cleanup", query_tile_size=tile_size,
    )


def test_faithful_skips_unused_joint_law_but_preserves_requested_diagnostics() -> None:
    model = _faithful(tile_size=4)
    model.eval()
    scene, query = _scene(), _query()
    prepared = model.prepare(scene)
    joint_calls = 0
    original_joint = model._joint_member_contribution

    def counted_joint(*args, **kwargs):
        nonlocal joint_calls
        joint_calls += 1
        return original_joint(*args, **kwargs)

    model._joint_member_contribution = counted_joint
    ordinary = model.decode(prepared, query)
    assert joint_calls == 0
    assert ordinary.work_counts["query_joint_likelihood_rows"] == 0
    assert ordinary.work_counts["query_joint_contribution_normalization_rows"] == 0
    assert all(type(value) is int for value in ordinary.work_counts.values())

    traced = model.decode(prepared, query, return_trace=True)
    assert joint_calls == 1
    assert traced.trace is not None
    _, expected_log_weight, expected_offset = joint_likelihood_contribution(
        traced.trace["prior_P"],
        torch.cat((traced.trace["member_ell_M"], traced.trace["member_ell_E"]), dim=-1),
        traced.trace["candidate_mask_joint"],
    )
    torch.testing.assert_close(traced.trace["member_joint_log_weight"], expected_log_weight, rtol=0, atol=0)
    torch.testing.assert_close(traced.trace["member_joint_log_weight_offset"], expected_offset, rtol=0, atol=0)
    expected_role_likelihood = torch.cat((
        original_role_conditional_likelihood(
            traced.trace["prior_P_M"], traced.trace["member_ell_M"], traced.trace["candidate_mask_M"],
        ),
        original_role_conditional_likelihood(
            traced.trace["prior_P_E"], traced.trace["member_ell_E"], traced.trace["candidate_mask_E"],
        ),
    ), dim=-1)
    torch.testing.assert_close(traced.trace["member_joint_likelihood"], expected_role_likelihood, rtol=0, atol=0)
    assert traced.work_counts["query_joint_likelihood_rows"] == int(
        traced.trace["candidate_mask_joint"].sum()
    )
    assert traced.work_counts["query_joint_contribution_normalization_rows"] > 0


def test_candidate_a_still_computes_its_default_joint_contribution() -> None:
    model = ConsistentJointMemberRoutingCore(
        _base_config(), input_dataset="synthetic-r13-7-execution-cleanup", query_tile_size=4,
    )
    model.eval()
    output = model.decode(model.prepare(_scene()), _query(), return_trace=True)
    assert output.trace is not None
    expected, _, _ = joint_likelihood_contribution(
        output.trace["prior_P"],
        torch.cat((output.trace["member_ell_M"], output.trace["member_ell_E"]), dim=-1),
        output.trace["candidate_mask_joint"],
    )
    torch.testing.assert_close(output.trace["member_joint_likelihood"], expected, rtol=0, atol=0)
    assert output.work_counts["query_joint_likelihood_rows"] > 0
    assert output.work_counts["query_joint_contribution_normalization_rows"] > 0
    assert all(type(value) is int for value in output.work_counts.values())


def test_h_distribution_helper_preserves_trace_and_paired_count_charging() -> None:
    model = TypedOrganizationMemberRoutingCore(
        _base_config(), input_dataset="synthetic-r13-7-execution-cleanup", query_tile_size=2,
    )
    model.eval()
    scene, query = _scene(), _query()
    prepared = model.prepare(scene)
    traced = model.decode(prepared, slice_query_v1(query, 0, 2), return_trace=True)
    assert traced.trace is not None
    ell = torch.cat((traced.trace["member_ell_M"], traced.trace["member_ell_E"]), dim=-1)
    distribution, helper_trace = model._typed_organization_distribution(
        ell,
        traced.trace["candidate_mask_joint"],
        traced.trace["proposal_weights_joint"],
        traced.trace["retrieval_R_read"],
        scene.num_modules,
    )
    torch.testing.assert_close(
        distribution,
        torch.cat((
            traced.trace["member_contribution_pre_strength_M"],
            traced.trace["member_contribution_pre_strength_E"],
        ), dim=-1),
        rtol=0,
        atol=0,
    )
    for role in ("M", "E"):
        torch.testing.assert_close(
            helper_trace[f"typed_organization_member_contribution_{role}"],
            traced.trace[f"typed_organization_member_contribution_{role}"],
            rtol=0,
            atol=0,
        )

    full, restricted, shared = model.decode_paired(prepared, query)
    full_separate = model.decode(prepared, query, intervention={"read_mode": "full"})
    restricted_separate = model.decode(prepared, query, intervention={"read_mode": "restricted"})
    assert all(type(value) is int for value in full.work_counts.values())
    assert all(type(value) is int for value in restricted.work_counts.values())
    assert all(type(value) is int for value in shared.values())
    for paired, separate in ((full, full_separate), (restricted, restricted_separate)):
        for name, value in separate.work_counts.items():
            assert paired.work_counts.get(name, 0) + shared.get(name, 0) == value, name
