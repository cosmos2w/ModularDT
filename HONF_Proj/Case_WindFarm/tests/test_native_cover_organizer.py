from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CaseLocalReceiverTree,
    ReceiverAnchorUniverse,
)

from windfarm.workflows import native_cover_organizer as workflow
from windfarm.workflows import native_cover_organizer_fit as fit_workflow


@dataclass(frozen=True)
class _DeviceMarker:
    device: torch.device


@dataclass(frozen=True)
class _AnchorBatch:
    receiver_anchor_weights: torch.Tensor
    receiver_anchor_roles: torch.Tensor
    env_weights: torch.Tensor


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
    return {
        "layout_count": 12,
        "layouts": [
            {
                "layout_index": layout,
                "training_rows_direction_order": [3 * layout, 3 * layout + 1, 3 * layout + 2],
                "activated_in_this_run": layout < active_layouts,
            }
            for layout in range(12)
        ],
    }


def test_fit_gate_rejects_limited_sample_without_disjoint_verified_fast_k_gt_1_labels() -> None:
    rows = []
    baselines = []
    documents = {}
    for index in range(3):
        row_index = index
        rows.append({
            "row_index": row_index,
            "layout_index": 0,
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

    assert assessment["status"] == "target_unavailable"
    assert assessment["partial_evidence"]["search_only_accepted_partial_observations_not_used_as_joint_labels"] == 96
    assert assessment["partial_evidence"]["search_only_accepted_candidates_faster_than_policy_none_dense"] == 0
    assert assessment["partial_evidence"]["directly_search_and_disjoint_verified_partial_rows"] == []
    assert assessment["optimizer"]["updates"] == 0
    assert assessment["controls"]["fixed_cover"] == "not_evaluated_target_unavailable"


def test_fit_gate_accepts_two_layout_competitive_verified_split_labels_without_full_sweep() -> None:
    rows = []
    baselines = []
    documents = {}
    for layout in range(2):
        for direction in range(3):
            row_index = 3 * layout + direction
            row = {
                "row_index": row_index,
                "layout_index": layout,
                "candidate_observations": [],
            }
            if direction < 2:
                row["teacher_oracle_search_winner"] = {
                    "proposal": {"kind": "split_child_prune_nearest_child"},
                    "teacher_search_gate_passed": True,
                    "teacher_adequate_for_primary_labels": True,
                    "actual_synchronized_complete_ms": 5.0,
                    "actual_hard_support": {"hard_cover_k": 2},
                    "disjoint_verification": {"teacher_preservation": {"gate_0p10": True}},
                    "split_gates": [1.0, 0.0, 0.0],
                    "module_membership": [[1.0, 0.0]],
                    "environment_membership": (
                        [[1.0, 0.0, 1.0]] if direction == 0 else [[1.0, 1.0, 0.0]]
                    ),
                }
            rows.append(row)
            baselines.append({
                "row_index": row_index,
                "dense_prepare_ms": 4.0,
                "dense_search_decode_ms": 6.0,
            })
            documents[row_index] = {
                "primary_supervision": {
                    "split_targets": [1.0, 0.0, 0.0],
                    "module_targets": [[1.0, 0.0]],
                    "environment_targets": (
                        [[1.0, 0.0, 1.0]] if direction == 0 else [[1.0, 1.0, 0.0]]
                    ),
                }
            }
    report = {"active_training_layouts": 2, "rows": rows, "baseline_timings": baselines}

    assessment = fit_workflow._assess_fit_target(report, _panel_manifest(2), documents)

    assert assessment["status"] == "fit_eligible"
    assert assessment["partial_evidence"]["distinct_verified_primary_label_signatures"] > 1
    assert len(assessment["partial_evidence"]["K_gt_1_and_measured_faster_layouts"]) == 2
    assert assessment["optimizer"]["updates"] == 100
    assert fit_workflow._fit_row_ids_to_use(list(range(6)), assessment) == [0, 1, 3, 4]


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
