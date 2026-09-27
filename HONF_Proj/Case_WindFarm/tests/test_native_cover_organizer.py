from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.adaptive_cover_field import AdaptiveCoverPairwiseField
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    INTERACTION_MECHANISMS,
    AdaptiveCoverPlan,
    CaseLocalReceiverTree,
    InteractionPermissionKey,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer
from honf_forward_core.interface_fields.types import EncodedInterfaceCase

from windfarm.workflows import native_cover_organizer as workflow
from windfarm.workflows import native_cover_organizer_fit as fit_workflow
from windfarm.workflows.native_cover_panel import TrainingLayout


@dataclass(frozen=True)
class _DeviceMarker:
    device: torch.device


@dataclass(frozen=True)
class _AnchorBatch:
    receiver_anchor_weights: torch.Tensor
    receiver_anchor_roles: torch.Tensor
    env_weights: torch.Tensor


def test_fixed_plan_replay_rejects_changed_source_order_and_anchor_geometry() -> None:
    environment = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    module = torch.tensor([[0.5, 1.0, 0.0]])
    weights = torch.ones(3)
    roles = torch.tensor([0, 0, 1], dtype=torch.long)
    scale = torch.ones(3)
    universe = ReceiverAnchorUniverse(torch.cat((environment, module)), weights, roles, scale)
    tree = CaseLocalReceiverTree.build(universe, max_nodes=1)
    plan = AdaptiveCoverPlan.full_access(tree, torch.ones(1), environment_count=2)
    policy = workflow._FixedPlanPolicy((plan,))
    encoded = SimpleNamespace(
        module_present=torch.ones((1, 1)), env_coords=environment.unsqueeze(0)
    )

    assert policy.plan_cases(encoded, None, (tree,))[0].tree is tree
    reordered = SimpleNamespace(
        module_present=encoded.module_present, env_coords=environment.flip(0).unsqueeze(0)
    )
    with pytest.raises(ValueError, match="environment source order"):
        policy.plan_cases(reordered, None, (tree,))

    moved_universe = ReceiverAnchorUniverse(
        torch.cat((environment, module + torch.tensor([0.1, 0.0, 0.0]))),
        weights, roles, scale,
    )
    moved_tree = CaseLocalReceiverTree.build(moved_universe, max_nodes=1)
    with pytest.raises(ValueError, match="receiver coordinates"):
        policy.plan_cases(encoded, None, (moved_tree,))


def test_case_batch_device_guard_checks_all_tensor_fields_at_cuda_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = torch.device("cuda:0")
    batch = SimpleNamespace(
        module_centers=_DeviceMarker(expected),
        query_xy=_DeviceMarker(expected),
        target_field=_DeviceMarker(expected),
        metadata=[{"case": "opaque host metadata"}],
    )
    monkeypatch.setattr(torch, "is_tensor", lambda value: isinstance(value, _DeviceMarker))

    workflow._assert_tensor_fields_on_device(batch, expected)

    batch.target_field = _DeviceMarker(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="target_field"):
        workflow._assert_tensor_fields_on_device(batch, expected)


def test_active_oracle_layouts_follow_the_frozen_train_order_not_sorted_panel_order() -> None:
    counts = {
        4: 14, 24: 6, 70: 27, 81: 29, 88: 7, 103: 6,
        107: 13, 115: 18, 119: 8, 174: 29, 177: 30, 196: 15,
    }
    layouts = tuple(
        TrainingLayout(
            layout_index=index,
            rows=(3 * index, 3 * index + 1, 3 * index + 2),
            turbine_count=count,
            feature_vector=(float(count), float(index), 1.0, float(count) / max(index, 1)),
        )
        for index, count in counts.items()
    )

    active = workflow._active_organizer_training_layouts(layouts, 8)

    assert [item.layout_index for item in active] == [4, 24, 70, 88, 115, 119, 174, 177]
    assert sum(len(item.rows) for item in active) == 24
    assert not {81, 103, 107, 196}.intersection(item.layout_index for item in active)

    same_m = fit_workflow._typed_same_module_count_holdout_summary(
        workflow.freeze_organizer_layout_split(layouts)
    )
    assert same_m["same_module_count_counts"] == [6, 29]
    assert same_m["training_layouts_with_repeated_module_counts"] == {}
    assert same_m["within_training_same_module_count_adaptation_comparison_available"] is False
    assert same_m["same_module_count_cross_layout_holdouts"]["6"]["training_layouts"][0][
        "layout_index"
    ] == 24
    assert same_m["same_module_count_cross_layout_holdouts"]["6"]["development_layouts"][0][
        "layout_index"
    ] == 103
    assert same_m["same_module_count_cross_layout_holdouts"]["29"]["training_layouts"][0][
        "layout_index"
    ] == 174
    assert same_m["same_module_count_cross_layout_holdouts"]["29"]["development_layouts"][0][
        "layout_index"
    ] == 81


def _typed_search_fixture() -> tuple[MechanismPlan, SimpleNamespace, dict[str, torch.Tensor]]:
    anchors = torch.tensor([
        [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0],
        [2.0, 0.0, 0.0], [2.5, 0.0, 0.0], [3.0, 0.0, 0.0], [3.5, 0.0, 0.0],
    ])
    universe = ReceiverAnchorUniverse(
        coordinates=anchors,
        weights=torch.ones(8),
        roles=torch.zeros(8, dtype=torch.long),
        coordinate_scale=torch.ones(3),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=3, min_leaf_anchors=1)
    present = torch.ones(3)
    plan = MechanismPlan.full_access(tree, present, environment_count=16).with_split(0, 1.0)
    env_xy = torch.cartesian_prod(torch.arange(4.0), torch.arange(4.0))
    encoded = SimpleNamespace(
        module_centers=torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.5, 0.0, 0.0]]]),
        module_present=present[None, :],
        coordinate_scale=torch.ones((1, 3)),
        env_coords=torch.cat((env_xy, torch.zeros((16, 1))), dim=1)[None, :, :],
        env_weights=torch.ones((1, 16)),
    )
    scores = {
        mechanism: torch.zeros_like(plan.permission_matrix(mechanism))
        for mechanism in INTERACTION_MECHANISMS
    }
    return plan, encoded, scores


def _typed_encoded_case(geometry: SimpleNamespace) -> EncodedInterfaceCase:
    return EncodedInterfaceCase(
        module_tokens=torch.ones((1, 3, 8)),
        env_tokens=torch.ones((1, 16, 8)),
        global_token=torch.ones((1, 8)),
        module_centers=geometry.module_centers,
        env_coords=geometry.env_coords,
        module_present=geometry.module_present,
        module_features=torch.ones((1, 3, 2)),
        env_features=torch.ones((1, 16, 2)),
        env_weights=geometry.env_weights,
        coordinate_scale=geometry.coordinate_scale,
    )


def test_typed_work_summary_matches_native_pair_ledgers_on_native_axes() -> None:
    plan, geometry, _scores = _typed_search_fixture()
    encoded = _typed_encoded_case(geometry)
    full = MechanismPlan.full_access(plan.tree, encoded.module_present[0], 16).with_split(0, 1.0)
    root = full.tree.nodes[0]
    assert root.left is not None and root.right is not None
    left, right = int(root.left), int(root.right)
    partial = full
    me = partial.permission_matrix("ME").clone()
    me[left, :8] = 0.0
    partial = partial.with_permission("ME", me)
    qe = partial.permission_matrix("QE").clone()
    qe[right, 8:] = 0.0
    partial = partial.with_permission("QE", qe)

    field = AdaptiveCoverPairwiseField(
        hidden_dim=8,
        message_hidden_dim=12,
        num_heads=2,
        fourier_frequencies=1,
        min_leaf_anchors=1,
        optional_native_policy=True,
    )
    field.set_cover_mode("external")
    field.set_cover_executor("packed")
    receivers = geometry.module_centers.new_tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0],
         [2.0, 0.0, 0.0], [2.5, 0.0, 0.0], [3.0, 0.0, 0.0], [3.5, 0.0, 0.0]]
    )
    receiver_batch = receivers[None, :, :]
    receiver_features = torch.ones((1, 8, 8))

    for candidate in (full, partial):
        summary = workflow._typed_work_summary(candidate, encoded, receivers)
        with torch.no_grad():
            prepared = field.prepare(
                encoded,
                encoded.module_tokens,
                cover_plans=(candidate,),
            )
            _result, query_ledger = field.read(
                prepared,
                encoded,
                receiver_batch,
                receiver_features,
            )
        for mechanism in ("MM", "ME", "EM"):
            key = f"cover_prepare_{mechanism.lower()}_unique_pairs"
            assert summary["mechanisms"][mechanism][
                f"cover_{mechanism.lower()}_unique_source_receiver_pairs"
            ] == int(prepared["cover_preparation_ledger"][key])
        for mechanism in ("QM", "QE"):
            key = f"cover_{mechanism.lower()}_unique_source_receiver_pairs"
            assert summary["mechanisms"][mechanism][key] == int(query_ledger[key])

    full_summary = workflow._typed_work_summary(full, encoded, receivers)
    partial_summary = workflow._typed_work_summary(partial, encoded, receivers)
    assert full_summary["mechanisms"]["MM"]["cover_mm_unique_source_receiver_pairs"] == 6
    assert full_summary["mechanisms"]["ME"]["cover_me_unique_source_receiver_pairs"] == 48
    assert full_summary["mechanisms"]["EM"]["cover_em_unique_source_receiver_pairs"] == 48
    assert full_summary["mechanisms"]["QM"]["cover_qm_unique_source_receiver_pairs"] == 24
    assert full_summary["mechanisms"]["QE"]["cover_qe_unique_source_receiver_pairs"] == 128
    assert partial_summary["mechanisms"]["ME"]["root_children_support_differs"] is True
    assert partial_summary["total_native_pair_reachable_distinct_packets_across_mechanisms"] > 0
    assert partial_summary["total_structural_nonredundant_packets_across_mechanisms"] > 0
    assert partial_summary["logical_work_score"] == pytest.approx(
        partial_summary["total_unique_source_receiver_pairs_across_mechanisms"]
        + 0.05
        * partial_summary["total_native_pair_reachable_distinct_packets_across_mechanisms"]
    )
    assert "not executed work or latency" in partial_summary[
        "logical_work_score_semantics"
    ]


def test_typed_differentiable_work_matches_native_pair_totals_and_keeps_gradient() -> None:
    template, geometry, _scores = _typed_search_fixture()
    encoded = _typed_encoded_case(geometry)
    encoded.module_present[0] = torch.tensor([1.0, 0.0, 1.0])
    queries = template.tree.universe.coordinates
    full = MechanismPlan.full_access(template.tree, encoded.module_present[0], 16).with_split(0, 1.0)

    root = full.tree.nodes[0]
    assert root.left is not None and root.right is not None
    left, right = int(root.left), int(root.right)
    me = full.permission_matrix("ME").clone()
    me[left, 1:5] = 0.0
    partial = full.with_permission("ME", me)
    qe = partial.permission_matrix("QE").clone()
    qe[right, 9:13] = 0.0
    partial = partial.with_permission("QE", qe)

    full_native_pairs = workflow._typed_work_summary(full, encoded, queries)[
        "total_unique_source_receiver_pairs_across_mechanisms"
    ]
    partial_native_pairs = workflow._typed_work_summary(partial, encoded, queries)[
        "total_unique_source_receiver_pairs_across_mechanisms"
    ]
    for mechanism in ("MM", "EM", "QM"):
        assert fit_workflow._typed_native_source_indices(
            full, encoded, queries, mechanism
        ).tolist() == [0, 2]
    for mechanism in ("ME", "QE"):
        assert fit_workflow._typed_native_source_indices(
            full, encoded, queries, mechanism
        ).tolist() == list(range(16))
    full_work = fit_workflow._typed_differentiable_work(full, encoded, queries)
    partial_work = fit_workflow._typed_differentiable_work(partial, encoded, queries)
    assert full_work.item() == pytest.approx(1.0)
    assert partial_work.item() == pytest.approx(partial_native_pairs / full_native_pairs)
    assert full_work.item() * full_native_pairs == pytest.approx(full_native_pairs)
    assert partial_work.item() * full_native_pairs == pytest.approx(partial_native_pairs)

    record = SimpleNamespace(
        encoded=encoded,
        trees=(template.tree,),
        full_prepared=SimpleNamespace(
            dense_prepared=SimpleNamespace(module_states=encoded.module_tokens)
        ),
    )
    organizer = InputOnlyCoverOrganizer(
        state_dim=8,
        module_feature_dim=2,
        environment_feature_dim=2,
        hidden_dim=12,
    ).train()
    fit_workflow._initialize_typed_organizer_all_access(organizer)
    scores = fit_workflow._typed_scores(organizer, record)
    straight_through_plan = organizer.plans_from_scores(
        (scores,), encoded, record.trees, hard=True, straight_through_hard=True
    )[0]
    straight_through_work = fit_workflow._typed_differentiable_work(
        straight_through_plan, encoded, queries
    )
    assert straight_through_work.item() == pytest.approx(1.0)
    pair_parameters = tuple(
        parameter
        for mechanism in INTERACTION_MECHANISMS
        for parameter in organizer.pair_scorers[mechanism].parameters()
    )
    gradients = torch.autograd.grad(straight_through_work, pair_parameters, allow_unused=True)
    assert any(value is not None and bool((value != 0).any()) for value in gradients)


def _typed_partial_permissions(
    template: MechanismPlan, geometry: SimpleNamespace
) -> MechanismPlan:
    permissions = {}
    for mechanism in INTERACTION_MECHANISMS:
        matrix = template.permission_matrix(mechanism).clone()
        matrix.zero_()
        if mechanism in {"MM", "EM", "QM"}:
            matrix[0, [0, 2]] = 1.0
        else:
            matrix[0, :8] = 1.0
        permissions[mechanism] = matrix
    return MechanismPlan(
        template.tree,
        torch.zeros_like(template.split_gates),
        geometry.module_present[0],
        int(geometry.env_coords.shape[1]),
        permissions,
    )


def test_typed_population_fixed_support_crops_training_max_axis_for_smaller_module_rows() -> None:
    anchor_coords = torch.stack(
        (torch.arange(30, dtype=torch.float32), torch.zeros(30), torch.zeros(30)), dim=-1
    )
    universe = ReceiverAnchorUniverse(
        coordinates=anchor_coords,
        weights=torch.ones(30),
        roles=torch.zeros(30, dtype=torch.long),
        coordinate_scale=torch.ones(3),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=3, min_leaf_anchors=1)
    env_coords = torch.stack(
        (torch.arange(16, dtype=torch.float32), torch.ones(16), torch.zeros(16)), dim=-1
    )
    records = []
    for row, module_count in ((12, 14), (72, 6)):
        present = torch.ones((1, module_count))
        encoded = SimpleNamespace(
            module_present=present,
            module_centers=anchor_coords[:module_count][None],
            env_coords=env_coords[None],
            env_weights=torch.ones((1, 16)),
        )
        oracle_plan = MechanismPlan.full_access(tree, present[0], environment_count=16)
        records.append(SimpleNamespace(
            case=SimpleNamespace(index=row),
            encoded=encoded,
            trees=(tree,),
            search_batch=SimpleNamespace(query_xy=anchor_coords[None]),
            oracle_plan=oracle_plan,
        ))

    maximum_support, report = fit_workflow._typed_population_fixed_support(records)

    assert maximum_support["MM"].shape == (14,)
    assert maximum_support["QM"].shape == (14,)
    assert report["target_source_count_by_mechanism"]["MM"] == 10
    assert report["target_source_count_by_mechanism"]["EM"] == 10
    assert report["target_source_count_by_mechanism"]["QM"] == 10
    assert report["target_source_count_by_mechanism"]["ME"] == 16
    assert report["target_source_count_by_mechanism"]["QE"] == 16

    for record in records:
        per_case_support = {
            mechanism: support[:int(fit_workflow._typed_valid_source_mask(record.encoded, mechanism).numel())]
            for mechanism, support in maximum_support.items()
        }
        plan = fit_workflow._typed_root_permissions_plan(record, per_case_support)
        module_count = int(record.encoded.module_present.shape[1])
        expected_source_counts = {
            "MM": module_count,
            "ME": 16,
            "EM": module_count,
            "QM": module_count,
            "QE": 16,
        }
        for mechanism in INTERACTION_MECHANISMS:
            assert plan.permission_matrix(mechanism).shape == (
                len(tree.nodes), expected_source_counts[mechanism]
            )
            assert torch.equal(plan.permission_matrix(mechanism)[0], per_case_support[mechanism])


def test_typed_g6_fixed_and_collapsed_controls_are_source_union_plans_from_train_only() -> None:
    template, geometry, _scores = _typed_search_fixture()
    encoded = _typed_encoded_case(geometry)
    query = template.tree.universe.coordinates
    oracle_a = _typed_partial_permissions(template, geometry)
    oracle_b_permissions = {
        mechanism: oracle_a.permission_matrix(mechanism).clone()
        for mechanism in INTERACTION_MECHANISMS
    }
    for mechanism in INTERACTION_MECHANISMS:
        if mechanism in {"MM", "EM", "QM"}:
            oracle_b_permissions[mechanism][0, 2] = 0.0
        else:
            oracle_b_permissions[mechanism][0, 7] = 0.0
    oracle_b = MechanismPlan(
        oracle_a.tree,
        oracle_a.split_gates,
        oracle_a.module_present,
        oracle_a.environment_count,
        oracle_b_permissions,
    )
    records = [
        SimpleNamespace(
            case=SimpleNamespace(index=12 + offset),
            encoded=encoded,
            trees=(template.tree,),
            search_batch=SimpleNamespace(query_xy=query[None]),
            oracle_plan=plan,
        )
        for offset, plan in enumerate((oracle_a, oracle_b))
    ]

    fixed_support, report = fit_workflow._typed_population_fixed_support(records)
    assert report["training_row_indices"] == [12, 13]
    assert report["development_rows_used"] == []
    assert report["support_indexing"] == (
        "training-maximum module/environment slot indices, cropped to each case's native source axis"
    )
    assert report["permutation_invariant_physical_support"] is False
    assert "reordering" in report["limitation"]
    assert set(fixed_support) == set(INTERACTION_MECHANISMS)
    fixed = fit_workflow._typed_root_permissions_plan(records[0], fixed_support)
    assert not bool((fixed.split_gates != 0).any())
    for mechanism in INTERACTION_MECHANISMS:
        assert torch.equal(
            fixed.permission_matrix(mechanism)[0] > 0,
            fixed_support[mechanism] > 0,
        )

    # The unchanged slot mask selects different physical environment cells
    # after a source-slot permutation, so this is an artificial slot control.
    environment_order = torch.arange(encoded.env_coords.shape[1])
    environment_order[0], environment_order[-1] = environment_order[-1].clone(), environment_order[0].clone()
    permuted_encoded = SimpleNamespace(
        module_present=encoded.module_present,
        module_centers=encoded.module_centers,
        env_coords=encoded.env_coords[:, environment_order],
        env_weights=encoded.env_weights[:, environment_order],
    )
    permuted_record = SimpleNamespace(encoded=permuted_encoded, trees=(template.tree,))
    fit_workflow._typed_root_permissions_plan(permuted_record, fixed_support)
    original_selected_cells = encoded.env_coords[0, fixed_support["ME"] > 0]
    permuted_selected_cells = permuted_encoded.env_coords[0, fixed_support["ME"] > 0]
    assert not torch.equal(original_selected_cells, permuted_selected_cells)

    collapsed = fit_workflow._typed_collapsed_root_plan(records[0], oracle_a)
    assert not bool((collapsed.split_gates != 0).any())
    for mechanism in INTERACTION_MECHANISMS:
        expected = fit_workflow._typed_native_source_indices(
            oracle_a, encoded, query, mechanism
        )
        actual = torch.nonzero(
            collapsed.permission_matrix(mechanism)[0] > 0, as_tuple=False
        ).flatten()
        assert actual.tolist() == expected.tolist()


def test_typed_g6_direct_pair_budget_is_deterministic_and_compiles_on_all_native_axes() -> None:
    template, geometry, _scores = _typed_search_fixture()
    encoded = _typed_encoded_case(geometry)
    query = template.tree.universe.coordinates
    learned = _typed_partial_permissions(template, geometry)
    record = SimpleNamespace(
        encoded=encoded,
        trees=(template.tree,),
        search_batch=SimpleNamespace(query_xy=query[None]),
    )

    direct, direct_report = fit_workflow._typed_direct_pair_plan(record, learned, query)
    repeated, repeated_report = fit_workflow._typed_direct_pair_plan(record, learned, query)
    learned_pairs = workflow._typed_work_summary(learned, encoded, query)[
        "total_unique_source_receiver_pairs_across_mechanisms"
    ]
    direct_work = workflow._typed_work_summary(
        direct, encoded, query, include_root_child_support=False
    )
    assert direct.canonical_hash() == repeated.canonical_hash()
    assert direct_report["ranking_rule"] == repeated_report["ranking_rule"]
    assert direct_report["selected_pair_count_exact_native_compiler"] == learned_pairs
    assert direct_work["total_unique_source_receiver_pairs_across_mechanisms"] == learned_pairs
    assert direct_report["teacher_oracle_or_reference_values_used_for_ranking"] is False
    assert direct_report["executor_requirement"].startswith("dense_masked reference")
    for mechanism in INTERACTION_MECHANISMS:
        matrix_count = int(direct.direct_pair_access[mechanism].sum())
        compiled_count = direct_work["mechanisms"][mechanism][
            f"cover_{mechanism.lower()}_unique_source_receiver_pairs"
        ]
        assert matrix_count == compiled_count == direct_report[
            "selected_pair_count_by_mechanism"
        ][mechanism]
        assert matrix_count > 0


def test_typed_g6_learned_plan_is_probe_order_chunk_and_target_independent_and_source_equivariant() -> None:
    torch.manual_seed(4409)
    template, geometry, _scores = _typed_search_fixture()
    encoded = _typed_encoded_case(geometry)
    organizer = InputOnlyCoverOrganizer(
        state_dim=8,
        module_feature_dim=2,
        environment_feature_dim=2,
        hidden_dim=12,
    ).eval()
    query = template.tree.universe.coordinates
    record = SimpleNamespace(
        encoded=encoded,
        trees=(template.tree,),
        full_prepared=SimpleNamespace(
            dense_prepared=SimpleNamespace(module_states=encoded.module_tokens)
        ),
        search_batch=SimpleNamespace(
            query_xy=query[None],
            target_field=torch.zeros((1, len(query), 3)),
        ),
        query_chunk_size=8,
    )
    reference = fit_workflow._typed_hard_plan(organizer, record)

    query_order = torch.tensor([7, 2, 5, 0, 6, 1, 4, 3])
    reordered_record = SimpleNamespace(
        **{
            **vars(record),
            "search_batch": SimpleNamespace(
                query_xy=query[query_order][None],
                target_field=torch.randn((1, len(query), 3)),
            ),
            "query_chunk_size": 3,
        }
    )
    reordered = fit_workflow._typed_hard_plan(organizer, reordered_record)
    assert reference.canonical_hash() == reordered.canonical_hash()
    for mechanism in INTERACTION_MECHANISMS:
        assert torch.equal(
            reference.permission_matrix(mechanism),
            reordered.permission_matrix(mechanism),
        )
    assert torch.equal(reference.split_gates, reordered.split_gates)

    module_order = torch.tensor([2, 0, 1])
    environment_order = torch.tensor([3, 1, 0, 2, 7, 5, 4, 6, 11, 9, 8, 10, 15, 13, 12, 14])
    permuted_encoded = EncodedInterfaceCase(
        module_tokens=encoded.module_tokens[:, module_order],
        env_tokens=encoded.env_tokens[:, environment_order],
        global_token=encoded.global_token,
        module_centers=encoded.module_centers[:, module_order],
        env_coords=encoded.env_coords[:, environment_order],
        module_present=encoded.module_present[:, module_order],
        module_features=encoded.module_features[:, module_order],
        env_features=encoded.env_features[:, environment_order],
        env_weights=encoded.env_weights[:, environment_order],
        coordinate_scale=encoded.coordinate_scale,
    )
    permuted_record = SimpleNamespace(
        encoded=permuted_encoded,
        trees=(template.tree,),
        full_prepared=SimpleNamespace(
            dense_prepared=SimpleNamespace(module_states=permuted_encoded.module_tokens)
        ),
    )
    permuted_plan = fit_workflow._typed_hard_plan(organizer, permuted_record)
    assert torch.equal(reference.split_gates, permuted_plan.split_gates)
    for mechanism in INTERACTION_MECHANISMS:
        order = module_order if mechanism in {"MM", "EM", "QM"} else environment_order
        torch.testing.assert_close(
            permuted_plan.permission_matrix(mechanism),
            reference.permission_matrix(mechanism)[:, order],
            rtol=0.0,
            atol=0.0,
        )


def test_typed_g6_direct_pair_plan_runs_through_dense_masked_complete_core_path() -> None:
    torch.manual_seed(3107)
    template, geometry, _scores = _typed_search_fixture()
    query = template.tree.universe.coordinates[:8][None]
    batch = BatchData(
        module_centers=geometry.module_centers,
        module_present=geometry.module_present,
        module_features=torch.randn(1, 3, 3),
        global_context=torch.randn(1, 5),
        query_xy=query,
        query_time=None,
        target_field=torch.randn(1, 8, 3),
        case_name="g6-direct-pair-control",
        env_coords=geometry.env_coords,
        env_features=torch.randn(1, 16, 2),
        query_features=torch.randn(1, 8, 2),
        env_weights=geometry.env_weights,
        metadata=[{}],
    )
    config = UnifiedForwardConfig(
        forward_architecture="dense_pairwise_field",
        field_dim=3,
        spatial_dim=3,
        coordinate_scale=[4.0, 4.0, 2.0],
        hidden_dim=16,
        dropout=0.0,
        geometry_mode="nonperiodic",
        boundary_feature_mode="none",
        interface_model=InterfaceFieldConfig(
            message_hidden_dim=8,
            attention_heads=2,
            coarse_latent_count=3,
            coarse_blocks=1,
            relative_fourier_frequencies=1,
            receiver_chunk_size=4,
        ),
    )
    core = InterfaceFieldCore(config).eval()
    core.backend = AdaptiveCoverPairwiseField(
        16,
        8,
        2,
        1,
        optional_native_policy=True,
    )
    core.backend.set_cover_mode("external")
    core.backend.set_cover_executor("dense_masked")
    encoded = core.encode_case(batch)
    tree = core.backend.build_case_trees(encoded)[0]
    full = MechanismPlan.full_access(tree, encoded.module_present[0], int(encoded.env_coords.shape[1]))
    learned_permissions = {}
    for mechanism in INTERACTION_MECHANISMS:
        matrix = full.permission_matrix(mechanism).clone()
        matrix.zero_()
        if mechanism in {"MM", "EM", "QM"}:
            matrix[0, :2] = 1.0
        else:
            matrix[0, :8] = 1.0
        learned_permissions[mechanism] = matrix
    learned = MechanismPlan(
        tree,
        torch.zeros_like(full.split_gates),
        encoded.module_present[0],
        int(encoded.env_coords.shape[1]),
        learned_permissions,
    )
    record = SimpleNamespace(
        encoded=encoded,
        trees=(tree,),
        search_batch=batch,
    )
    direct, _ = fit_workflow._typed_direct_pair_plan(record, learned, batch.query_xy[0])
    context = workflow.InteractionContext(receiver_role="query")
    with torch.inference_mode():
        dense_prepared = core.prepare(encoded, encoded.module_tokens)
        dense_output = core.decode_queries(
            dense_prepared, batch.query_xy, batch.query_features
        )["pred_field"]
        prepared = core.prepare(
            encoded,
            encoded.module_tokens,
            fixed_cover_plans=(direct,),
            interaction_context=context,
        )
        direct_output = core.decode_queries(
            prepared,
            batch.query_xy,
            batch.query_features,
            return_interaction_aux=True,
            interaction_context=context,
        )

    assert float((dense_output - direct_output["pred_field"]).abs().max()) > 1.0e-7
    summary = workflow._typed_work_summary(
        direct, encoded, batch.query_xy[0], include_root_child_support=False
    )
    preparation = prepared.backend_state["cover_preparation_ledger"]
    query_ledger = direct_output["_interaction_aux"]
    assert preparation["cover_prepare_executor_dense_masked"] == 1
    assert prepared.backend_state["cover_execution_views"] is None
    for mechanism in ("MM", "ME", "EM"):
        assert int(preparation[f"cover_prepare_{mechanism.lower()}_unique_pairs"]) == summary[
            "mechanisms"][mechanism][f"cover_{mechanism.lower()}_unique_source_receiver_pairs"]
    for mechanism in ("QM", "QE"):
        assert int(query_ledger[f"cover_{mechanism.lower()}_executor_dense_masked"]) == 1
        assert int(query_ledger[f"cover_{mechanism.lower()}_unique_source_receiver_pairs"]) == summary[
            "mechanisms"][mechanism][f"cover_{mechanism.lower()}_unique_source_receiver_pairs"]
    learned_pair_count = int(
        workflow._typed_work_summary(learned, encoded, batch.query_xy[0])[
            "total_unique_source_receiver_pairs_across_mechanisms"
        ]
    )
    assert summary["total_unique_source_receiver_pairs_across_mechanisms"] == learned_pair_count

    full_direct, _ = fit_workflow._typed_direct_pair_plan(record, full, batch.query_xy[0])
    sparse_access = dict(full_direct.direct_pair_access)
    sparse_access["QE"] = torch.zeros_like(sparse_access["QE"])
    sparse_qe = replace(full_direct, direct_pair_access=sparse_access)
    with torch.inference_mode():
        full_direct_prepared = core.prepare(
            encoded,
            encoded.module_tokens,
            fixed_cover_plans=(full_direct,),
            interaction_context=context,
        )
        full_direct_output = core.decode_queries(
            full_direct_prepared,
            batch.query_xy,
            batch.query_features,
            interaction_context=context,
        )["pred_field"]
        sparse_qe_prepared = core.prepare(
            encoded,
            encoded.module_tokens,
            fixed_cover_plans=(sparse_qe,),
            interaction_context=context,
        )
        sparse_qe_output = core.decode_queries(
            sparse_qe_prepared,
            batch.query_xy,
            batch.query_features,
            interaction_context=context,
        )["pred_field"]

    assert full_direct_prepared.backend_state["cover_execution_views"] is None
    assert sparse_qe_prepared.backend_state["cover_execution_views"] is None
    assert float((full_direct_output - sparse_qe_output).abs().max()) > 1.0e-7
    full_direct_work = workflow._typed_work_summary(
        full_direct, encoded, batch.query_xy[0], include_root_child_support=False
    )
    sparse_qe_work = workflow._typed_work_summary(
        sparse_qe, encoded, batch.query_xy[0], include_root_child_support=False
    )
    full_qe_pairs = int(
        full_direct_work["mechanisms"]["QE"]["cover_qe_unique_source_receiver_pairs"]
    )
    sparse_qe_pairs = int(
        sparse_qe_work["mechanisms"]["QE"]["cover_qe_unique_source_receiver_pairs"]
    )
    assert full_qe_pairs > 0
    assert sparse_qe_pairs == 0
    for mechanism in ("MM", "ME", "EM", "QM"):
        assert sparse_qe_work["mechanisms"][mechanism][
            f"cover_{mechanism.lower()}_unique_source_receiver_pairs"
        ] == full_direct_work["mechanisms"][mechanism][
            f"cover_{mechanism.lower()}_unique_source_receiver_pairs"
        ]
    assert (
        full_direct_work["total_unique_source_receiver_pairs_across_mechanisms"]
        - sparse_qe_work["total_unique_source_receiver_pairs_across_mechanisms"]
        == full_qe_pairs
    )


def test_typed_g6_direct_pair_executor_is_restored_before_next_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    template, geometry, _scores = _typed_search_fixture()
    encoded = _typed_encoded_case(geometry)
    query = template.tree.universe.coordinates[:6]
    learned = _typed_partial_permissions(template, geometry)
    record = SimpleNamespace(
        case=SimpleNamespace(index=12),
        encoded=encoded,
        trees=(template.tree,),
        search_batch=SimpleNamespace(query_xy=query[None]),
        verification_batch=SimpleNamespace(query_xy=query[None]),
        full_verification_prediction=torch.zeros((1, len(query), 3)),
        verification=SimpleNamespace(roles=()),
    )
    direct, metadata = fit_workflow._typed_direct_pair_plan(record, learned, query)
    backend = AdaptiveCoverPairwiseField(
        8, 12, 2, 1, optional_native_policy=True
    )
    backend.set_cover_executor("packed")
    model = SimpleNamespace(core=SimpleNamespace(backend=backend))
    observed_executor: list[str] = []

    def replay(_model, _batch, plan, **_kwargs):
        observed_executor.append(backend.cover_executor)
        prepared = SimpleNamespace(backend_state={
            "cover_preparation_ledger": {
                "cover_prepare_me_actual_rows": 24,
                "cover_prepare_me_unique_pairs": 6,
                "cover_prepare_me_padded_rows": 18,
            }
        })
        prediction = torch.zeros((1, len(query), 3))
        output = {"_interaction_aux": {
            "cover_qm_actual_rows": 18,
            "cover_qm_unique_source_receiver_pairs": 7,
            "cover_qm_padded_rows": 11,
            "cover_qm_executor_dense_masked": int(isinstance(plan, fit_workflow._TypedDirectPairPlan)),
        }}
        return prepared, prediction, output, 0.1, 0.2

    monkeypatch.setattr(fit_workflow.base, "_typed_replay_plan", replay)
    monkeypatch.setattr(
        fit_workflow.base,
        "_probe_observation",
        lambda *_args, **_kwargs: (SimpleNamespace(roles={}), {}),
    )
    monkeypatch.setattr(
        fit_workflow.base,
        "teacher_preservation_gate",
        lambda *_args, **_kwargs: (True, "teacher_preserved"),
    )
    monkeypatch.setattr(
        fit_workflow.base, "_typed_teacher_gate_frontier", lambda *_args, **_kwargs: {}
    )

    direct_result = fit_workflow._typed_evaluate_g6_variant(
        model,
        record,
        direct,
        name="ungrouped_direct_pair_matched_budget",
        teacher_checkpoint_id="test",
        normalizer=None,
        device=torch.device("cpu"),
        provenance={},
        direct_pair_metadata=metadata,
    )
    assert backend.cover_executor == "packed"
    direct_executor_work = direct_result["native_executor_work"]
    assert direct_executor_work["preparation"][
        "cover_prepare_me_rectangle_minus_selected_mask_rows"
    ] == 18
    assert direct_executor_work["query"][
        "cover_qm_rectangle_minus_selected_mask_rows"
    ] == 11
    assert not any(
        key.endswith("_padded_rows")
        for area in ("preparation", "query")
        for key in direct_executor_work[area]
    )
    fit_workflow._typed_evaluate_g6_variant(
        model,
        record,
        learned,
        name="learned_input_only",
        teacher_checkpoint_id="test",
        normalizer=None,
        device=torch.device("cpu"),
        provenance={},
    )
    assert observed_executor == ["dense_masked", "packed"]
    assert backend.cover_executor == "packed"


def test_typed_q1024_reference_rmse_is_reported_when_sufficiency_is_unresolved() -> None:
    observation = SimpleNamespace(roles={
        "near_turbine": SimpleNamespace(reference=0.123, teacher=0.04, resolved=False),
        "unmeasured": SimpleNamespace(reference=None, teacher=None, resolved=False),
    })

    reference_rmse = fit_workflow._typed_reference_rmse_by_role(observation)

    assert reference_rmse == {"near_turbine": 0.123}
    assert observation.roles["near_turbine"].resolved is False


def test_root_child_support_aggregates_active_descendant_permissions() -> None:
    seed_plan, geometry, _scores = _typed_search_fixture()
    encoded = _typed_encoded_case(geometry)
    tree = CaseLocalReceiverTree.build(
        seed_plan.tree.universe,
        max_nodes=7,
        min_leaf_anchors=1,
    )
    plan = MechanismPlan.full_access(tree, encoded.module_present[0], 16)
    root = plan.tree.nodes[0]
    assert root.left is not None and root.right is not None
    left, right = int(root.left), int(root.right)
    left_node = plan.tree.nodes[left]
    assert left_node.left is not None and left_node.right is not None
    left_leaf, right_leaf = int(left_node.left), int(left_node.right)

    plan = plan.with_split(0, 1.0).with_split(left, 1.0)
    me = plan.permission_matrix("ME").clone()
    me[left] = 0.0
    me[left_leaf] = 0.0
    me[left_leaf, :4] = 1.0
    me[right_leaf] = 0.0
    me[right_leaf, 4:8] = 1.0
    me[right] = 0.0
    me[right, 8:12] = 1.0
    plan = plan.with_permission("ME", me)

    summary = workflow._typed_work_summary(
        plan,
        encoded,
        geometry.module_centers[0],
    )["mechanisms"]["ME"]

    by_child = summary["root_children_source_indices"]
    assert by_child[str(left)] == list(range(8))
    assert by_child[str(right)] == list(range(8, 12))
    assert summary["root_children_support_differs"] is True
    assert summary["root_children_unique_source_receiver_pairs"][str(left)] > 0
    assert summary["root_children_unique_source_receiver_pairs"][str(right)] > 0


def test_equivalent_root_merge_preserves_bypass_and_phase_inheritance() -> None:
    anchors = torch.stack((torch.arange(8.0), torch.zeros(8), torch.zeros(8)), dim=1)
    universe = ReceiverAnchorUniverse(
        coordinates=anchors,
        weights=torch.ones(8),
        roles=torch.zeros(8, dtype=torch.long),
        coordinate_scale=torch.ones(3),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=7, min_leaf_anchors=1)
    assert tree.nodes[0].left is not None and tree.nodes[0].right is not None
    left, right = int(tree.nodes[0].left), int(tree.nodes[0].right)
    present = torch.ones(3)
    full = MechanismPlan.full_access(tree, present, environment_count=16)
    me = full.permission_matrix("ME").clone()
    me[left, :4] = me[right, :4] = 0.0
    qe_phase = full.permission_matrix("QE").clone()
    qe_phase[left, 8:] = qe_phase[right, 8:] = 0.0
    permissions = {
        InteractionPermissionKey("MM"): full.permission_matrix("MM"),
        InteractionPermissionKey("ME"): me,
        InteractionPermissionKey("QE", "P0"): qe_phase,
    }
    plan = MechanismPlan(tree, full.with_split(0, 1.0).split_gates, present, 16, permissions)

    merged = workflow._merge_equivalent_root_children(plan)

    assert merged.split_gates[0].item() == 0.0
    assert set(merged.permissions) == set(plan.permissions)
    assert merged.explicit_bypass_keys == plan.explicit_bypass_keys
    assert merged.explicit_phase_keys == plan.explicit_phase_keys
    assert merged.permission_status("ME", phase="P0") == "mechanism_permission_inherited"
    assert merged.permission_status("QE", phase="P0") == "phase_permission"
    for phase in (None, "P0"):
        for mechanism in INTERACTION_MECHANISMS:
            assert torch.equal(
                plan.access_for(mechanism, anchors, phase=phase),
                merged.access_for(mechanism, anchors, phase=phase),
            )


def test_teacher_gate_reason_keeps_complete_string() -> None:
    reason = "protected_role_worst_case_distortion_exceeds_limit"
    assert workflow._teacher_gate_reason_list(reason) == [reason]


def test_typed_local_proposals_change_one_child_and_one_mechanism_cumulatively() -> None:
    plan, encoded, scores = _typed_search_fixture()
    proposals = workflow._typed_child_local_proposals(
        plan,
        encoded,
        scores,
        train_evidence_id="train-row-test",
    )

    assert proposals
    for proposal in proposals:
        candidate = proposal["plan"]
        changed_mechanisms = []
        for mechanism in INTERACTION_MECHANISMS:
            before = plan.permission_matrix(mechanism)
            after = candidate.permission_matrix(mechanism)
            if not torch.equal(before, after):
                changed_mechanisms.append(mechanism)
                changed_rows = torch.nonzero((before != after).any(dim=1), as_tuple=False).flatten().tolist()
                assert changed_rows == [proposal["node"]]
                assert torch.all(after <= before)
        assert changed_mechanisms == [proposal["mechanism"]]
        assert proposal["node"] in {plan.tree.nodes[0].left, plan.tree.nodes[0].right}
        assert candidate.canonical_hash() != plan.canonical_hash()

    first = proposals[0]
    cumulative = first["plan"]
    second_candidates = workflow._typed_child_local_proposals(
        cumulative,
        encoded,
        scores,
        train_evidence_id="train-row-test",
    )
    assert second_candidates
    for second in second_candidates:
        for mechanism in INTERACTION_MECHANISMS:
            assert torch.all(
                second["plan"].permission_matrix(mechanism)
                <= cumulative.permission_matrix(mechanism)
            )
    environment_proposals = [item for item in proposals if item["mechanism"] in {"ME", "QE"}]
    assert environment_proposals
    assert all(
        item["kind"] == "one_child_flow_frame_global_environment_cell_remove"
        and item["flow_frame"]["partition"] == "4 by 4 flow-frame global quantile cells"
        for item in environment_proposals
    )


def test_typed_environment_quantile_cells_are_global_for_both_receiver_children() -> None:
    plan, encoded, _scores = _typed_search_fixture()
    root = plan.tree.nodes[0]
    assert root.left is not None and root.right is not None

    left = workflow._typed_flow_frame_environment_blocks(plan, encoded, root.left)
    right = workflow._typed_flow_frame_environment_blocks(plan, encoded, root.right)

    assert left == right
    assert sorted(source for cell in left for source in cell) == list(range(16))


def test_typed_supervision_marks_split_parent_reachable_and_collapses_equal_children() -> None:
    plan, _encoded, _scores = _typed_search_fixture()
    labels = workflow._typed_plan_supervision(plan)

    assert labels["split_observed"][0] is True
    assert labels["reachable_node_ids"] == [0, 1, 2]

    merged = workflow._merge_equivalent_root_children(plan)
    queries = plan.tree.universe.coordinates
    assert merged.split_gates[0].item() == 0.0
    for mechanism in INTERACTION_MECHANISMS:
        assert torch.equal(
            plan.access_for(mechanism, queries), merged.access_for(mechanism, queries)
        )


def test_typed_cache_key_uses_complete_plan_hash_and_experiment_identity() -> None:
    plan, _encoded, _scores = _typed_search_fixture()
    module = plan.permission_matrix("QM").clone()
    module[1, 0] = 0.0
    changed = plan.with_permission("QM", module)
    common = {
        "input_hash": "input-a",
        "checkpoint_hash": "checkpoint-a",
        "probe_hash": "probe-a",
        "mechanism": "QM",
    }

    first = workflow._typed_observation_cache_key(**common, plan_hash=plan.canonical_hash())
    changed_plan = workflow._typed_observation_cache_key(
        **common, plan_hash=changed.canonical_hash()
    )
    other_checkpoint = workflow._typed_observation_cache_key(
        **{**common, "checkpoint_hash": "checkpoint-b"}, plan_hash=plan.canonical_hash()
    )

    assert first != changed_plan
    assert first != other_checkpoint


def test_typed_fit_starts_from_all_access_and_uses_typed_case_state() -> None:
    target, geometry, _scores = _typed_search_fixture()
    encoded = EncodedInterfaceCase(
        module_tokens=torch.randn(1, 3, 8),
        env_tokens=torch.randn(1, 16, 8),
        global_token=torch.randn(1, 8),
        module_centers=geometry.module_centers,
        env_coords=geometry.env_coords,
        module_present=geometry.module_present,
        module_features=torch.ones(1, 3, 2),
        env_features=torch.ones(1, 16, 2),
        env_weights=torch.ones(1, 16),
        coordinate_scale=torch.ones(1, 3),
    )
    record = SimpleNamespace(
        encoded=encoded,
        trees=(target.tree,),
        full_prepared=SimpleNamespace(
            dense_prepared=SimpleNamespace(module_states=encoded.module_tokens)
        ),
    )
    organizer = InputOnlyCoverOrganizer(
        state_dim=8,
        module_feature_dim=2,
        environment_feature_dim=2,
        hidden_dim=12,
    ).eval()

    initialization = fit_workflow._initialize_typed_organizer_all_access(organizer)
    scores = fit_workflow._typed_scores(organizer, record)
    plan = organizer.plans_from_scores(
        (scores,), encoded, record.trees, hard=True
    )[0]

    assert "all active module/environment slots open independently" in initialization[
        "permission_hard_state"
    ]
    assert initialization["permission_initial_logit"] == 2.0
    assert initialization["split_initial_logit"] == -2.0
    assert initialization["permission_sigmoid_derivative_at_initial_logit"] > 0.10
    assert not bool((plan.split_gates != 0).any())
    assert plan.is_full_access()
    for mechanism in INTERACTION_MECHANISMS:
        assert bool((plan.permission_matrix(mechanism) == 1).all())


def test_typed_plan_formation_does_not_read_dense_prepared_state() -> None:
    target, geometry, _scores = _typed_search_fixture()
    encoded = _typed_encoded_case(geometry)

    class _PoisonDensePreparation:
        @property
        def dense_prepared(self):
            raise AssertionError("typed plan formation must not read dense pairwise states")

    record = SimpleNamespace(
        encoded=encoded,
        trees=(target.tree,),
        full_prepared=_PoisonDensePreparation(),
    )
    organizer = InputOnlyCoverOrganizer(
        state_dim=8,
        module_feature_dim=2,
        environment_feature_dim=2,
        hidden_dim=12,
    ).eval()
    fit_workflow._initialize_typed_organizer_all_access(organizer)

    scores = fit_workflow._typed_scores(organizer, record)
    plan = fit_workflow._typed_hard_plan(organizer, record)

    assert torch.isfinite(scores.split_logits).all()
    assert plan.is_full_access()
    assert fit_workflow._typed_organizer_input_provenance()[
        "dense_prepared_module_states_used_for_plan_formation"
    ] is False


def test_typed_fit_requires_merged_corrected_native_work_provenance() -> None:
    report = {"merged_stage_directories": ["stage_01", "stage_02"]}
    legacy_row = {
        "search_selection": {
            "selected_search_work": {
                "logical_work_score": 1.0,
                "total_unique_source_receiver_pairs_across_mechanisms": 1,
            }
        }
    }
    with pytest.raises(ValueError, match="corrected exact-native-work G2 provenance"):
        fit_workflow._validate_corrected_merged_typed_search_provenance(report, {12: legacy_row})

    corrected_row = {
        "search_selection": {
            "selected_search_work": {
                "pair_work_semantics": (
                    "exact valid pair rows from native core compilers: MM excludes self and padded modules; "
                    "ME uses present module receivers and positive environment weights; EM uses all environment "
                    "receiver slots and present module sources; QM/QE use Q query receivers and valid sources"
                ),
                "logical_work_score_semantics": (
                    "heuristic score equal to exact valid native pair rows plus 0.05 times the count of "
                    "receiver-conditioned permission signatures; packet term is not executed work or latency"
                ),
            }
        }
    }
    fit_workflow._validate_corrected_merged_typed_search_provenance(report, {12: corrected_row})

    with pytest.raises(ValueError, match="merged from corrected G2 stages"):
        fit_workflow._validate_corrected_merged_typed_search_provenance(
            {"merged_stage_directories": []}, {12: corrected_row}
        )


def test_typed_fit_binds_plan_documents_to_exact_report_paths_and_hashes(tmp_path) -> None:
    row = 12
    directory = tmp_path.resolve()
    path = directory / f"selected_typed_plan_row_{row:04d}.json"
    document = {row: {"plan_hash": "verified-plan-hash"}}
    report = {
        "rows": [{
            "row_index": row,
            "selected_plan_path": str(path),
            "selected_plan_hash": "verified-plan-hash",
        }]
    }
    fit_workflow._validate_typed_report_plan_bindings(report, directory, [row], document)

    wrong_hash = json.loads(json.dumps(report))
    wrong_hash["rows"][0]["selected_plan_hash"] = "stale-plan-hash"
    with pytest.raises(ValueError, match="selected plan hash differs"):
        fit_workflow._validate_typed_report_plan_bindings(wrong_hash, directory, [row], document)

    wrong_path = json.loads(json.dumps(report))
    wrong_path["rows"][0]["selected_plan_path"] = str(directory / "stale.json")
    with pytest.raises(ValueError, match="selected plan path differs"):
        fit_workflow._validate_typed_report_plan_bindings(wrong_path, directory, [row], document)

    wrong_rows = json.loads(json.dumps(report))
    wrong_rows["rows"][0]["row_index"] = row + 1
    with pytest.raises(ValueError, match="report rows do not exactly match"):
        fit_workflow._validate_typed_report_plan_bindings(wrong_rows, directory, [row], document)


def test_typed_fit_binds_merged_plan_document_bytes_to_unique_source_shard(tmp_path) -> None:
    row = 12
    merged = tmp_path / "merged"
    stage = tmp_path / "stage"
    merged.mkdir()
    stage.mkdir()
    stage_report = {
        "status": "stage_complete",
        "training_rows": [row],
    }
    (stage / "typed_search_report.json").write_text(json.dumps(stage_report), encoding="utf-8")
    source_plan = stage / f"selected_typed_plan_row_{row:04d}.json"
    merged_plan = merged / f"selected_typed_plan_row_{row:04d}.json"
    source_plan.write_text('{"plan_hash":"source"}\n', encoding="utf-8")
    merged_plan.write_bytes(source_plan.read_bytes())
    report = {"merged_stage_directories": [str(stage)]}

    fit_workflow._validate_typed_merged_document_bytes(report, merged, [row])

    source_plan.write_text('{"plan_hash":"edited-source"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="merged plan bytes differ"):
        fit_workflow._validate_typed_merged_document_bytes(report, merged, [row])


def test_typed_full_grid_can_add_predeclared_dev_row_only_after_clean_q1024_gate() -> None:
    split = {
        "training_layouts": [
            {"layout_index": 4, "rows_direction_order": [12, 13, 14], "turbine_count": 14},
            {"layout_index": 24, "rows_direction_order": [72, 73, 74], "turbine_count": 6},
        ],
        "development_layouts": [
            {"layout_index": 81, "rows_direction_order": [243, 244, 245], "turbine_count": 29},
            {"layout_index": 103, "rows_direction_order": [309, 310, 311], "turbine_count": 6},
        ],
    }
    q1024 = {
        72: {
            "status": "complete", "partition": "training", "layout_index": 24,
            "teacher_preservation": {"passed": True}, "artifact_path": "/tmp/train.json",
        },
        309: {
            "status": "complete", "partition": "development", "layout_index": 103,
            "teacher_preservation": {"passed": True}, "artifact_path": "/tmp/dev.json",
            "development_g2_labels_loaded": False, "independent_holdout": True,
        },
    }

    selected, provenance = fit_workflow._typed_select_full_grid_rows(
        split,
        q1024,
        native_validation_layouts=2,
        include_development_layout=True,
    )

    assert selected == [72, 309]
    assert [item["partition"] for item in provenance["selected_layout_rows"]] == [
        "training", "development"
    ]
    assert provenance["development_g2_labels_loaded"] is False

    q1024[309]["teacher_preservation"]["passed"] = False
    with pytest.raises(ValueError, match="development row 309 lacks a passing Q1024"):
        fit_workflow._typed_select_full_grid_rows(
            split,
            q1024,
            native_validation_layouts=2,
            include_development_layout=True,
        )

    q1024[309]["teacher_preservation"]["passed"] = True
    q1024[309]["development_g2_labels_loaded"] = True
    with pytest.raises(ValueError, match="must not load per-case G2 labels"):
        fit_workflow._typed_select_full_grid_rows(
            split,
            q1024,
            native_validation_layouts=2,
            include_development_layout=True,
        )


def test_typed_fit_recomputes_recursive_supervision_from_bound_plan() -> None:
    plan, _geometry, _scores = _typed_search_fixture()
    valid = workflow._typed_plan_supervision(plan)
    fit_workflow._validate_typed_plan_document_supervision(plan, valid, 12)

    edited = json.loads(json.dumps(valid))
    edited["mechanism_targets"]["QE"][0][0] = 1.0 - edited[
        "mechanism_targets"
    ]["QE"][0][0]
    with pytest.raises(ValueError, match="supervision differs from deterministic labels"):
        fit_workflow._validate_typed_plan_document_supervision(plan, edited, 12)


def test_typed_row_reference_status_uses_only_measured_native_grid_guard() -> None:
    rows = [
        {"row_index": 12, "reference_sufficient": "unknown"},
        {"row_index": 13, "reference_sufficient": "unknown"},
        {"row_index": 72, "reference_sufficient": "unknown"},
    ]
    grids = [
        {
            "row_index": 12,
            "layout_index": 4,
            "native_metrics": {
                "train_only_physical_reference_guard": {
                    "status": "passed",
                    "protected_roles": {"near_turbine": {"measured": True, "passed": True}},
                }
            },
        },
        {
            "row_index": 72,
            "layout_index": 24,
            "native_metrics": {
                "train_only_physical_reference_guard": {
                    "status": "failed",
                    "protected_roles": {"near_turbine": {"measured": True, "passed": False}},
                }
            },
        },
    ]

    fit_workflow._attach_typed_reference_status(rows, grids)

    assert [item["reference_sufficient"] for item in rows] == ["passed", "unknown", "failed"]
    assert rows[0]["reference_evidence"]["source"] == "frozen train/development layout native grid"
    assert rows[1]["reference_evidence"]["reason"]


def test_typed_search_stage_parser_accepts_row_groups_and_rejects_repeats() -> None:
    parser = workflow._parser()
    args = parser.parse_args([
        "--checkpoint", "best_field.pt",
        "--volume", "volume",
        "--compact", "compact.npz",
        "--output-dir", "diagnostics/generated/stage",
        "--typed-search",
        "--typed-search-rows", "12,13,14,72,73,74",
    ])

    assert workflow._parse_typed_search_rows(args.typed_search_rows) == [12, 13, 14, 72, 73, 74]
    with pytest.raises(ValueError, match="cannot repeat"):
        workflow._parse_typed_search_rows("12,13,12")


def test_typed_search_stage_merge_attests_exact_rows_and_plan_hashes(tmp_path) -> None:
    plan, _encoded, _scores = _typed_search_fixture()
    train_layouts = [4, 24, 70, 88, 115, 119, 174, 177]
    train_rows = [[12 + 3 * index, 13 + 3 * index, 14 + 3 * index] for index in range(8)]
    development_layouts = [81, 103, 107, 196]
    development_rows = [243, 244, 245, 309, 310, 311, 321, 322, 323, 588, 589, 590]
    expected_rows = [row for group in train_rows for row in group]
    split = {
        "split_sha256": "geometry-lock-test",
        "training_layout_indices": train_layouts,
        "development_layout_indices": development_layouts,
        "training_rows_direction_order": expected_rows,
        "development_rows_direction_order": development_rows,
        "training_layouts": [
            {"layout_index": layout, "rows_direction_order": rows}
            for layout, rows in zip(train_layouts, train_rows, strict=True)
        ],
    }
    checkpoint_hash = "checkpoint-test"
    stage_root = tmp_path / "diagnostics" / "generated" / "typed_stages"
    stage_directories = []
    for stage_index in range(4):
        directory = stage_root / f"stage_{stage_index + 1:02d}"
        directory.mkdir(parents=True)
        stage_directories.append(directory)
        stage_rows = [row for group in train_rows[stage_index * 2 : stage_index * 2 + 2] for row in group]
        lock_path = directory / "organizer_split_lock.json"
        lock_payload = json.dumps({"split_sha256": split["split_sha256"]}, sort_keys=True)
        lock_path.write_text(lock_payload)
        lock_hash = hashlib.sha256(lock_payload.encode()).hexdigest()
        row_summaries = []
        candidate_events = []
        for row in stage_rows:
            candidate_events.append({
                "row_index": row,
                "complete_plan_hash": plan.canonical_hash(),
                "executed_forward": True,
                "teacher_gate_frontier": {
                    "0.01": {"passed_all_protected_roles": False},
                    "0.05": {"passed_all_protected_roles": False},
                    "0.10": {"passed_all_protected_roles": True},
                },
                "teacher_preservation": {
                    "normalized_distortion_by_role": {"near_turbine": 0.05, "volume": 0.02}
                },
                "native_reference_rmse_mps_by_role": {"near_turbine": 0.1, "volume": 0.1},
            })
            selected = {
                "row_index": row,
                "plan_hash": plan.canonical_hash(),
                "plan": plan.to_dict(),
                "supervision": workflow._typed_plan_supervision(plan),
                "search_selection": {
                    "generated_candidate_plan_count": 1,
                    "skipped_without_query_pair_reduction": 0,
                    "selected_disjoint_teacher_verification": {"passed": True}
                },
            }
            (directory / f"selected_typed_plan_row_{row:04d}.json").write_text(
                json.dumps(selected, sort_keys=True)
            )
            (directory / f"typed_search_row_{row:04d}.json").write_text(json.dumps({
                "row_index": row,
                "selected_plan_hash": plan.canonical_hash(),
                "candidate_forward_count": 1,
                "generated_candidate_plan_count": 1,
                "skipped_without_query_pair_reduction": 0,
                "logical_work_frontier_by_teacher_gate": {
                    "0.01": {"qualifying_candidate_observation_count": 0},
                    "0.05": {"qualifying_candidate_observation_count": 0},
                    "0.10": {"qualifying_candidate_observation_count": 1},
                },
            }))
            row_summaries.append({
                "row_index": row,
                "selected_plan_path": str(directory / f"selected_typed_plan_row_{row:04d}.json"),
                "selected_plan_hash": plan.canonical_hash(),
                "candidate_forward_count": 1,
                "logical_work_frontier_by_teacher_gate": {
                    "0.01": {"qualifying_candidate_observation_count": 0},
                    "0.05": {"qualifying_candidate_observation_count": 0},
                    "0.10": {"qualifying_candidate_observation_count": 1},
                },
            })
        manifest = {
            "checkpoint_sha256": checkpoint_hash,
            "anchor_measure_variant": "raw",
            "organizer_split_frozen_before_new_outcomes": {"split_sha256": split["split_sha256"]},
        }
        (directory / "panel_manifest.json").write_text(json.dumps(manifest))
        report = {
            "status": "stage_complete",
            "workflow": "windfarm_receiver_local_incremental_typed_search",
            "checkpoint": "Run2103-best_field.pt",
            "checkpoint_sha256": checkpoint_hash,
            "teacher_checkpoint_id": "Run2103:test",
            "device": "cpu",
            "query_count_per_search_and_verification_probe": 1024,
            "anchor_measure_variant": "raw",
            "frozen_split": split,
            "split_lock_path": str(lock_path),
            "split_lock_sha256": lock_hash,
            "training_layout_indices": train_layouts,
            "development_layout_indices": development_layouts,
            "training_rows": stage_rows,
            "expected_training_rows": expected_rows,
            "stage_layout_indices": train_layouts[stage_index * 2 : stage_index * 2 + 2],
            "development_rows_used_for_search_or_fit": [],
            "protected_receiver_roles": ["near_turbine", "volume"],
            "teacher_distortion_limit_per_protected_role": 0.10,
            "proposal_rule": "one child and mechanism",
            "beam_width": 4,
            "candidate_evaluations_per_row_cap": 96,
            "rows": row_summaries,
            "total_candidate_forward_calls": len(stage_rows),
            "total_gradient_ranking_forward_calls": len(stage_rows),
            "total_identity_forward_calls": len(stage_rows),
            "total_disjoint_verification_forward_calls": len(stage_rows),
            "total_panel_reference_forward_calls": 4 * len(stage_rows),
            "materialization_forward_calls": 1,
            "total_complete_forward_calls": 1 + 4 * len(stage_rows) + 3 * len(stage_rows) + len(stage_rows),
            "total_prepare_case_calls": 1 + 2 * len(stage_rows) + 3 * len(stage_rows) + len(stage_rows),
            "total_decode_calls": 1 + 4 * len(stage_rows) + 3 * len(stage_rows) + len(stage_rows),
            "optimizer_updates": 0,
            "new_physical_solves": 0,
            "elapsed_seconds": 1.0,
        }
        (directory / "typed_search_report.json").write_text(json.dumps(report))
        (directory / "typed_candidate_observations.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in candidate_events)
        )

    output = stage_root / "merged"
    merged = workflow.merge_incremental_typed_search_stages(stage_directories, output)

    assert merged["status"] == "complete"
    assert merged["training_rows"] == expected_rows
    assert merged["development_rows_used_for_search_or_fit"] == []
    assert merged["total_candidate_forward_calls"] == 24
    assert merged["total_generated_candidate_plan_count"] == 24
    assert merged["total_complete_forward_calls"] == 196
    assert len(list(output.glob("selected_typed_plan_row_*.json"))) == 24
    assert merged["split_lock_sha256"] == hashlib.sha256(
        (output / "organizer_split_lock.json").read_bytes()
    ).hexdigest()


def test_checkpoint_only_evaluation_parser_is_distinct_and_defaults_to_disjoint() -> None:
    args = workflow._parser().parse_args([
        "--checkpoint", "best_field.pt",
        "--volume", "volume",
        "--compact", "compact.npz",
        "--output-dir", "diagnostics/generated/eval",
        "--oracle-dir", "diagnostics/generated/oracle",
        "--evaluate-organizer-state", "organizer_checkpoint_update_0100.pt",
    ])

    assert args.evaluate_organizer_state.endswith("update_0100.pt")
    assert args.evaluation_stage == "disjoint"
    assert args.query_count is None


def test_typed_checkpoint_parser_has_opt_in_development_native_grid() -> None:
    args = workflow._parser().parse_args([
        "--checkpoint", "best_field.pt",
        "--volume", "volume",
        "--compact", "compact.npz",
        "--output-dir", "diagnostics/generated/typed_eval",
        "--typed-search-dir", "diagnostics/generated/typed_search",
        "--evaluate-typed-organizer-state", "typed_state.pt",
        "--typed-evaluation-stage", "full-grid",
        "--typed-disjoint-results-dir", "diagnostics/generated/typed_q1024",
        "--typed-include-development-native-grid",
    ])

    assert args.typed_evaluation_stage == "full-grid"
    assert args.typed_include_development_native_grid is True


def test_typed_fit_cli_allows_q1024_only_without_full_grid_layouts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    def fake_fit(**kwargs):
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(fit_workflow, "run_typed_organizer_fit", fake_fit, raising=False)
    output_dir = tmp_path / "diagnostics" / "generated" / "typed_fit_cli"
    result = workflow.main([
        "--checkpoint", "Run2103-best_field.pt",
        "--volume", "wind-volume",
        "--compact", "wind-compact.npz",
        "--device", "cpu",
        "--output-dir", str(output_dir),
        "--typed-fit",
        "--typed-search-dir", "merged-typed-search",
        "--typed-fit-supervised-updates", "150",
        "--typed-fit-predictive-updates", "1",
        "--typed-native-validation-layouts", "0",
        "--query-count", "1024",
    ])

    assert result == 0
    assert captured["native_validation_layouts"] == 0
    assert captured["supervised_updates"] == 150
    assert captured["predictive_updates"] == 1


def test_checkpoint_only_evaluation_json_uses_atomic_replace(tmp_path) -> None:
    destination = tmp_path / "row.json"

    fit_workflow._atomic_json_write({"stage": "disjoint", "complete": True}, destination)

    assert json.loads(destination.read_text()) == {"complete": True, "stage": "disjoint"}
    assert not destination.with_name(".row.json.tmp").exists()


def test_tree_node_manifest_derives_parent_and_depth_from_candidate_links() -> None:
    universe = ReceiverAnchorUniverse(
        coordinates=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        weights=torch.ones(4),
        roles=torch.zeros(4, dtype=torch.long),
        coordinate_scale=torch.ones(3),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=3, min_leaf_anchors=1)

    nodes = workflow._tree_node_metadata(tree)

    assert len(nodes) == 3
    assert nodes[0]["parent"] is None
    assert nodes[0]["depth"] == 0
    assert nodes[1]["parent"] == nodes[2]["parent"] == 0
    assert nodes[1]["depth"] == nodes[2]["depth"] == 1


def test_plan_supervision_exports_one_coherent_plan_and_masks_unreached_children() -> None:
    universe = ReceiverAnchorUniverse(
        coordinates=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        weights=torch.ones(4),
        roles=torch.zeros(4, dtype=torch.long),
        coordinate_scale=torch.ones(3),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=3, min_leaf_anchors=1)
    present = torch.tensor([1.0, 0.0, 1.0])
    plan = workflow.AdaptiveCoverPlan.full_access(tree, present, environment_count=4)
    modules = plan.module_membership.clone()
    environment = plan.environment_membership.clone()
    environment[0, 1] = 0.0
    plan = workflow.AdaptiveCoverPlan(tree, plan.split_gates, modules, environment)

    labels = workflow._plan_supervision(plan, present)

    assert labels["environment_targets"][0] == [1.0, 0.0, 1.0, 1.0]
    assert labels["environment_observed"][0] == [True, True, True, True]
    assert labels["environment_observed"][1] == [False, False, False, False]
    assert labels["module_observed"][0] == [True, False, True]
    assert labels["split_observed"][0] is True
    assert labels["split_observed"][1] is False


def test_split_plan_supervision_observes_only_root_and_reached_children() -> None:
    universe = ReceiverAnchorUniverse(
        coordinates=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        weights=torch.ones(4),
        roles=torch.zeros(4, dtype=torch.long),
        coordinate_scale=torch.ones(3),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=3, min_leaf_anchors=1)
    present = torch.tensor([1.0, 1.0])
    plan = workflow.AdaptiveCoverPlan.full_access(tree, present, environment_count=2).with_split(0, 1.0)

    labels = workflow._plan_supervision(plan, present)

    left, right = tree.nodes[0].left, tree.nodes[0].right
    assert left is not None and right is not None
    assert labels["split_targets"][0] == 1.0
    assert labels["split_observed"][0] is True
    assert labels["module_observed"][left] == [True, True]
    assert labels["module_observed"][right] == [True, True]


def test_masked_hard_metrics_report_positive_and_negative_learning_separately() -> None:
    universe = ReceiverAnchorUniverse(
        coordinates=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        weights=torch.ones(2),
        roles=torch.zeros(2, dtype=torch.long),
        coordinate_scale=torch.ones(3),
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=1, min_leaf_anchors=1)
    plan = workflow.AdaptiveCoverPlan.full_access(tree, torch.ones(2), environment_count=2)
    environment = plan.environment_membership.clone()
    environment[0, 1] = 0.0
    plan = workflow.AdaptiveCoverPlan(tree, plan.split_gates, plan.module_membership, environment)
    labels = {
        "split_targets": torch.zeros_like(plan.split_gates),
        "split_observed": torch.ones_like(plan.split_gates, dtype=torch.bool),
        "module_targets": torch.ones_like(plan.module_membership),
        "module_observed": torch.ones_like(plan.module_membership, dtype=torch.bool),
        "environment_targets": environment.clone(),
        "environment_observed": torch.ones_like(environment, dtype=torch.bool),
    }

    metrics = fit_workflow._masked_hard_label_metrics(plan, labels)

    assert metrics["split"]["observed_negative_count"] == 1
    assert metrics["split"]["negative_accuracy"] == 1.0
    assert metrics["module"]["observed_positive_count"] == 2
    assert metrics["module"]["positive_accuracy"] == 1.0
    assert metrics["environment"]["observed_positive_count"] == 1
    assert metrics["environment"]["observed_negative_count"] == 1
    assert metrics["environment"]["masked_accuracy"] == 1.0


def test_role_balanced_anchor_variant_changes_only_tree_measure() -> None:
    batch = _AnchorBatch(
        receiver_anchor_weights=torch.tensor([[1.0, 3.0, 10.0, 10.0]]),
        receiver_anchor_roles=torch.tensor([[0, 0, 1, 2]]),
        env_weights=torch.tensor([[0.25, 0.75]]),
    )

    balanced, provenance = workflow._role_balanced_anchor_batch(batch)

    assert torch.equal(balanced.env_weights, batch.env_weights)
    assert torch.equal(balanced.receiver_anchor_roles, batch.receiver_anchor_roles)
    assert torch.equal(balanced.receiver_anchor_weights, torch.tensor([[0.25, 0.75, 0.5, 0.5]]))
    assert provenance["raw_mass"]["module_rotor_to_environment_mass_ratio"] == 5.0
    assert provenance["balanced_mass"]["module_rotor_to_environment_mass_ratio"] == 1.0
    assert provenance["source_quadrature_changed"] is False


def _panel_manifest(active_layouts: int) -> dict[str, object]:
    panel_layouts = (
        (4, 14),
        (24, 6),
        (70, 27),
        (81, 29),
        (88, 7),
        (103, 6),
        (107, 13),
        (115, 18),
        (119, 8),
        (174, 29),
        (177, 30),
        (196, 15),
    )
    return {
        "layout_count": 12,
        "layouts": [
            {
                "layout_index": layout,
                "training_rows_direction_order": [3 * layout, 3 * layout + 1, 3 * layout + 2],
                "turbine_count": turbine_count,
                "geometry_feature_vector": [float(turbine_count), float(layout), 1.0, float(turbine_count)],
                "activated_in_this_run": index < active_layouts,
            }
            for index, (layout, turbine_count) in enumerate(panel_layouts)
        ],
    }


def test_fit_gate_rejects_limited_sample_without_disjoint_verified_fast_partial_labels() -> None:
    rows = []
    baselines = []
    documents = {}
    for index in range(3):
        row_index = 12 + index
        rows.append({
            "row_index": row_index,
            "layout_index": 4,
            "candidate_observations": [
                {
                    "accepted_by_oracle": True,
                    "actual_synchronized_complete_ms": 300.0,
                    "proposal": {"kind": "environment_block_prune"},
                }
                for _ in range(32)
            ],
            "selected_plan_source": "full_access_retained_after_direct_paired_timing",
        })
        baselines.append({
            "row_index": row_index,
            "dense_prepare_ms": 4.0,
            "dense_search_decode_ms": 6.0,
        })
        documents[row_index] = {
            "primary_supervision": {
                "split_targets": [0.0, 0.0, 0.0],
                "module_targets": [[1.0, 1.0]],
                "environment_targets": [[1.0, 1.0, 1.0]],
            }
        }
    report = {
        "active_training_layouts": 1,
        "query_count_per_disjoint_search_and_verification_probe": 1024,
        "rows": rows,
        "baseline_timings": baselines,
    }

    assessment = fit_workflow._assess_fit_target(report, _panel_manifest(1), documents)

    assert assessment["status"] == "learning_ineligible"
    assert not assessment["learning_eligible"]
    assert assessment["partial_evidence"]["search_only_accepted_partial_observations_not_used_as_joint_labels"] == 96
    assert assessment["partial_evidence"]["search_only_accepted_candidates_faster_than_policy_none_dense"] == 0
    assert assessment["partial_evidence"]["directly_search_and_disjoint_verified_partial_rows"] == []
    assert assessment["optimizer"]["updates"] == 0
    assert assessment["controls"]["fixed_cover"] == "not_evaluated_learning_ineligible"


@pytest.mark.parametrize("hard_cover_k", [1, 2])
def test_fit_gate_accepts_two_layout_competitive_verified_partial_labels_without_full_sweep(
    hard_cover_k: int,
) -> None:
    rows = []
    baselines = []
    documents = {}
    for layout in range(2):
        layout_index = (4, 24)[layout]
        for direction in range(3):
            row_index = 3 * layout_index + direction
            row = {
                "row_index": row_index,
                "layout_index": layout_index,
                "candidate_observations": [],
            }
            if direction < 2:
                split_targets = [1.0, 0.0, 0.0] if hard_cover_k == 2 else [0.0, 0.0, 0.0]
                module_targets = [[1.0, 0.0]] if hard_cover_k == 2 else [[1.0, 1.0]]
                environment_targets = (
                    [[1.0, 0.0, 1.0]] if direction == 0 else [[1.0, 1.0, 0.0]]
                )
                row["teacher_oracle_search_winner"] = {
                    "proposal": {
                        "kind": "split_child_prune_nearest_child" if hard_cover_k == 2 else "environment_block_prune"
                    },
                    "teacher_search_gate_passed": True,
                    "teacher_adequate_for_primary_labels": True,
                    "actual_synchronized_complete_ms": 5.0,
                    "actual_hard_support": {"hard_cover_k": hard_cover_k},
                    "disjoint_verification": {"teacher_preservation": {"gate_0p10": True}},
                    "split_gates": split_targets,
                    "module_membership": module_targets,
                    "environment_membership": environment_targets,
                }
                row["candidate_observations"] = [{
                    "accepted_by_oracle": True,
                    "actual_synchronized_complete_ms": 5.0,
                    "proposal": row["teacher_oracle_search_winner"]["proposal"],
                }]
            rows.append(row)
            baselines.append({
                "row_index": row_index,
                "dense_prepare_ms": 4.0,
                "dense_search_decode_ms": 6.0,
            })
            documents[row_index] = {
                "primary_supervision": {
                    "split_targets": split_targets,
                    "module_targets": module_targets,
                    "environment_targets": environment_targets,
                }
            }
    report = {"active_training_layouts": 2, "rows": rows, "baseline_timings": baselines}

    assessment = fit_workflow._assess_fit_target(report, _panel_manifest(2), documents)

    assert assessment["status"] == "learning_eligible"
    assert assessment["learning_eligible"]
    assert assessment["reference_sufficient"] == "unknown"
    assert not assessment["deployment_eligible"]
    assert assessment["measured_speed_eligible"]
    assert assessment["partial_evidence"]["distinct_verified_primary_label_signatures"] > 1
    assert len(assessment["partial_evidence"]["measured_faster_partial_layouts"]) == 2
    assert assessment["partial_evidence"]["train_search_accepted_partial_observations"] == 4
    assert assessment["partial_evidence"]["search_only_accepted_partial_observations_not_used_as_joint_labels"] == 0
    assert assessment["optimizer"]["updates"] == 100
    assert fit_workflow._fit_row_ids_to_use([12, 13, 14, 72, 73, 74], assessment) == [12, 13, 72, 73]

    missing_dense = dict(report)
    missing_dense["baseline_timings"] = baselines[:3]
    without_two_measured_comparators = fit_workflow._assess_fit_target(
        missing_dense, _panel_manifest(2), documents
    )
    assert without_two_measured_comparators["status"] == "learning_eligible"
    assert without_two_measured_comparators["partial_evidence"]["measured_faster_partial_layouts"] == [4]


def test_slow_constant_coherent_label_remains_learning_eligible() -> None:
    rows = []
    baselines = []
    documents = {}
    for direction in range(3):
        row_index = 12 + direction
        winner = None
        document = None
        trials = []
        if direction == 0:
            split_targets = [0.0, 0.0, 0.0]
            module_targets = [[1.0, 1.0]]
            environment_targets = [[1.0, 0.0, 1.0]]
            winner = {
                "proposal": {"kind": "environment_block_prune"},
                "teacher_search_gate_passed": True,
                "teacher_adequate_for_primary_labels": True,
                "actual_synchronized_complete_ms": 300.0,
                "actual_hard_support": {"hard_cover_k": 1},
                "disjoint_verification": {"teacher_preservation": {"gate_0p10": True}},
                "split_gates": split_targets,
                "module_membership": module_targets,
                "environment_membership": environment_targets,
            }
            document = {"primary_supervision": {
                "split_targets": split_targets,
                "module_targets": module_targets,
                "environment_targets": environment_targets,
            }}
            trials = [{
                "accepted_by_oracle": True,
                "actual_synchronized_complete_ms": 300.0,
                "proposal": winner["proposal"],
            }]
            documents[row_index] = document
        row = {
            "row_index": row_index,
            "layout_index": 4,
            "candidate_observations": trials,
        }
        if winner is not None:
            row["teacher_oracle_search_winner"] = winner
        rows.append(row)
        baselines.append({
            "row_index": row_index,
            "dense_prepare_ms": 4.0,
            "dense_search_decode_ms": 6.0,
        })

    assessment = fit_workflow._assess_fit_target(
        {
            "active_training_layouts": 1,
            "rows": rows,
            "baseline_timings": baselines,
        },
        _panel_manifest(1),
        documents,
    )

    assert assessment["status"] == "learning_eligible"
    assert assessment["learning_eligible"]
    assert assessment["partial_evidence"]["distinct_verified_primary_label_signatures"] == 1
    assert assessment["partial_evidence"]["measured_faster_partial_layouts"] == []
    assert not assessment["measured_speed_eligible"]
    assert not assessment["deployment_eligible"]
    assert assessment["optimizer"]["updates"] == 100
    assert assessment["controls"]["fixed_cover"] == "evaluate_on_disjoint_queries"
    assert assessment["reference_sufficient"] == "unknown"


def test_fit_target_assessment_rejects_a_checkpoint_with_different_bytes(tmp_path) -> None:
    checkpoint = tmp_path / "best_field.pt"
    wrong_checkpoint = tmp_path / "wrong_best_field.pt"
    checkpoint.write_bytes(b"frozen teacher checkpoint")
    wrong_checkpoint.write_bytes(b"different checkpoint")
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {"checkpoint_sha256": checkpoint_hash, "rows": [], "baseline_timings": []}
    manifest = _panel_manifest(0)
    manifest["checkpoint_sha256"] = checkpoint_hash
    (oracle_dir / "oracle_benchmark.json").write_text(json.dumps(report), encoding="utf-8")
    (oracle_dir / "panel_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="requested fit checkpoint bytes differ"):
        fit_workflow.write_fit_target_assessment(
            checkpoint_path=wrong_checkpoint,
            oracle_dir=oracle_dir,
            output_dir=tmp_path / "report",
        )
