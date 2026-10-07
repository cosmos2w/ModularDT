import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from honf_forward_core.interface_fields.receiver_packets import (
    ActionDomain,
    InteractionBinding,
    ReceiverPacket,
    prepared_context_fingerprints,
)
from honf_forward_core.interface_fields.source_response_operator import SourceResponseOperator

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from receiver_packet_study import (
    FixedGeometryIncrementTask,
    IncrementRead,
    build_train_teacher_table,
    evaluate_patch_selectors,
    run_fixed_geometry_inverse_consumer,
    summarize_packet_catalog,
    validate_fixed25_train_authority,
)


def test_teacher_table_uses_only_primary_train_layouts_and_masks_padding():
    features = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    kernels = torch.ones(2, 2, 2, 3)
    weights = torch.full((2, 2, 2), 0.5)
    radii = torch.tensor([[0.1, 0.2, 0.0], [0.2, 0.1, 0.0]])
    table = build_train_teacher_table(
        pair_features=features,
        teacher_kernels=kernels,
        patch_weights=weights,
        action_radii=radii,
        layout_ids=("train-a", "train-b"),
        partition_labels=("TRAIN", "TRAIN"),
        source_ids_by_layout=(("train-a:s0", "train-a:s1", ""), ("train-b:s0", "train-b:s1", "")),
        source_present=torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        teacher_checkpoint_id="frozen-reconstruction-parent",
        manifest_sha256="fixed25-manifest-sha256",
        expected_layout_count=2,
    )
    assert table.pair_features.shape == (8, 4)
    assert table.raw_sensitivities.tolist() == pytest.approx([1.0] * 8)
    assert table.action_importance.tolist() == pytest.approx([0.1, 0.2] * 2 + [0.2, 0.1] * 2)
    assert set(table.layout_ids) == {"train-a", "train-b"}
    assert all(not source_id.endswith(":s2") for source_id in table.source_ids)

    with pytest.raises(ValueError, match="Only primary TRAIN"):
        build_train_teacher_table(
            pair_features=features,
            teacher_kernels=kernels,
            patch_weights=weights,
            action_radii=radii,
            layout_ids=("train-a", "train-b"),
            partition_labels=("TRAIN", "DEV"),
            source_ids_by_layout=(("train-a:s0", "train-a:s1", ""), ("train-b:s0", "train-b:s1", "")),
            source_present=torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            teacher_checkpoint_id="parent",
            manifest_sha256="manifest",
            expected_layout_count=2,
        )


def test_fixed25_authority_binds_membership_labels_rule_and_checkpoint_bytes(tmp_path):
    layout_ids = tuple(f"layout-{index:03d}" for index in range(150))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scope": "fixed_input_stratified_development_only",
                "partitions": {"train": {"case_ids": list(layout_ids)}},
            }
        )
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    selection_rule_path = tmp_path / "selection_rule.json"
    selection_rule_path.write_text('{"rule":"sealed"}\n')
    checkpoint_path = tmp_path / "teacher.pt"
    checkpoint_path.write_bytes(b"frozen teacher checkpoint bytes")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    receipt_path = tmp_path / "selection.json"
    receipt_path.write_text(
        json.dumps(
            {
                "selected_arm": "R-geom",
                "all_B4_promotion_guards_pass": False,
                "selected_from_saved_100_cadence_checkpoints": True,
                "selected_epoch": 1000,
                "selection_rule_sha256": hashlib.sha256(selection_rule_path.read_bytes()).hexdigest(),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
    )
    kwargs = {
        "manifest_path": manifest_path,
        "selection_receipt_path": receipt_path,
        "selection_rule_path": selection_rule_path,
        "checkpoint_path": checkpoint_path,
        "layout_ids": layout_ids,
        "partition_labels": ("TRAIN",) * 150,
        "teacher_checkpoint_id": f"thermal-response-checkpoint-sha256:{checkpoint_sha256}",
        "expected_manifest_sha256": manifest_sha256,
        "expected_selection_rule_sha256": hashlib.sha256(selection_rule_path.read_bytes()).hexdigest(),
    }
    authority = validate_fixed25_train_authority(**kwargs)
    assert authority["profile"] == "fixed25_v1"
    assert authority["train_layout_count"] == 150
    assert authority["checkpoint_sha256"] == checkpoint_sha256

    with pytest.raises(ValueError, match="authoritative fixed25_v1 TRAIN"):
        validate_fixed25_train_authority(**{**kwargs, "layout_ids": layout_ids[:-1] + ("dev-001",)})
    with pytest.raises(ValueError, match="Only authoritative primary TRAIN"):
        validate_fixed25_train_authority(**{**kwargs, "partition_labels": ("DEV",) + ("TRAIN",) * 149})
    original_rule_bytes = selection_rule_path.read_bytes()
    selection_rule_path.write_text('{"rule":"changed"}\n')
    with pytest.raises(ValueError, match="not the sealed fixed25_v1 rule"):
        validate_fixed25_train_authority(**kwargs)
    selection_rule_path.write_bytes(original_rule_bytes)
    checkpoint_path.write_bytes(b"replaced checkpoint")
    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        validate_fixed25_train_authority(**kwargs)


def test_patch_verifier_keeps_physical_error_separate_and_equal_k_controls():
    kernel = torch.tensor([[1.0, 0.0, 0.5], [0.5, 1.0, -0.25]])
    result = evaluate_patch_selectors(
        kernel=kernel,
        patch_weights=torch.tensor([0.5, 0.5]),
        action_radii=torch.tensor([0.2, 0.1, 0.1]),
        delta_control=torch.tensor([0.1, -0.05, 0.03]),
        source_ids=("a", "b", "c"),
        learned_source_ids=("a",),
        source_coordinates=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        receiver_center=torch.tensor([0.4, 0.4]),
        flow_direction=torch.tensor([1.0, 0.0]),
        budget_fraction=0.01,
        train_response_rms=0.4,
        reference_increment=torch.tensor([0.2, -0.1]),
        measured_receipts={"learned_proposal": {"fine_rows": 3, "latency_ms": 0.2}},
    )
    selectors = result["selectors"]
    assert selectors["learned_proposal"]["K_src"] == 1
    assert selectors["nearest_equal_k"]["K_src"] == 1
    assert selectors["upstream_equal_k"]["K_src"] == 1
    assert selectors["learned_proposal"]["teacher_distortion_rms"] >= 0
    assert selectors["learned_proposal"]["selected_physical_response_rmse"] is not None
    assert selectors["learned_proposal"]["actual_execution"]["fine_rows"] == 3
    assert "raw_sensitivity" in result and "radius_weighted_action_importance" in result
    assert result["scope"].startswith("offline_exact_teacher_verifier")


def test_packet_catalog_reports_nonredundant_k_packet_from_physical_source_lists():
    result = summarize_packet_catalog(
        (("s2", "s0"), ("s0", "s2"), ("s1",), ("s0", "s1")),
        ("s0", "s1", "s2"),
    )
    assert result["K_packet"] == 3
    assert result["K_src_by_receiver_packet"] == [2, 2, 1, 2]
    assert result["K_src_min"] == 1
    assert result["K_src_median"] == 2
    assert result["K_src_max"] == 2
    assert result["nonredundant_source_lists"] == [["s0", "s1"], ["s0", "s2"], ["s1"]]
    with pytest.raises(ValueError, match="repeat"):
        summarize_packet_catalog((("s0", "s0"),), ("s0", "s1"))
    with pytest.raises(ValueError, match="outside"):
        summarize_packet_catalog((("foreign",),), ("s0", "s1"))


def _binding(task_id, receiver_ids, receiver_coordinates, prepared_context):
    geometry_fingerprint, context_fingerprint = prepared_context_fingerprints(prepared_context)
    return InteractionBinding(
        dataset_id="thermal",
        scene_id=task_id,
        geometry_fingerprint=geometry_fingerprint,
        context_fingerprint=context_fingerprint,
        source_ids=tuple(f"{task_id}:source:{index}" for index in range(3)),
        source_slots=(0, 1, 2),
        prepared_source_ids=("0", "1", "2"),
        receiver_ids=tuple(receiver_ids),
        receiver_coordinates=tuple(tuple(value for value in row) for row in receiver_coordinates),
        environment_context_ids=("thermal-env-context",),
        output_roles=("fluid-temperature",),
        control_roles=("heat",),
        units=(("control", "W"), ("output", "K"), ("position", "m")),
        capability="affine_scalar_increment",
        ancestry=("all-source-context",),
    )


def _task(task_id):
    torch.manual_seed(sum(ord(value) for value in task_id))
    operator = SourceResponseOperator(
        2,
        1,
        2,
        mode="direct",
        spatial_dim=2,
        hidden=6,
        message=6,
        output_width=1,
        far_hidden=6,
    )
    centers = torch.tensor([[[0.1, 0.1], [0.9, 0.9], [0.5, 0.4], [0.0, 0.0]]])
    context = operator.prepare_context(
        torch.randn(1, 4, 2),
        torch.ones(1, 1),
        centers,
        torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        torch.ones(1, 2),
        torch.tensor([[0.05, 0.05, 0.05, 0.0]]),
        environment_tokens=torch.empty(1, 0, 2),
        environment_coords=torch.empty(1, 0, 2),
    )
    receivers = torch.tensor([[[0.05, 0.05], [0.4, 0.8]]])
    receiver_ids = (f"{task_id}:observed-0", f"{task_id}:observed-1")
    held_ids = (f"{task_id}:held-0",)
    binding = _binding(
        task_id,
        receiver_ids + held_ids,
        ((0.05, 0.05), (0.4, 0.8), (0.8, 0.2)),
        context,
    )
    action = ActionDomain("heat", "W", binding.source_ids, (0.2, 0.2, 0.2))
    packets = (
        ReceiverPacket(
            f"{task_id}:packet",
            binding.fingerprint,
            action.fingerprint,
            receiver_ids + held_ids,
            binding.source_ids[:1],
            binding.environment_context_ids,
            binding.ancestry,
            binding.output_roles,
            binding.control_roles,
            binding.units,
            residual_margin=0.1,
        ),
    )
    held_receivers = torch.tensor([[[0.8, 0.2]]])
    base_heat = torch.tensor([[0.4, 0.4, 0.4, 0.0]])
    observed_base = operator.apply_forcing(
        operator.prepare_receivers(context, receivers), base_heat, accumulation_dtype=torch.float64
    )
    held_base = operator.apply_forcing(
        operator.prepare_receivers(context, held_receivers), base_heat, accumulation_dtype=torch.float64
    )
    observed = IncrementRead(receiver_ids[:2], receivers, observed_base, packets)
    held = IncrementRead(held_ids, held_receivers, held_base, packets)
    return FixedGeometryIncrementTask(
        task_id,
        operator,
        context,
        torch.tensor([[0.05, -0.025, -0.025]]),
        binding,
        action,
        observed,
        held,
    )


def test_fixed_geometry_inverse_consumer_runs_four_train_tasks_on_observed_and_held_reads():
    result = run_fixed_geometry_inverse_consumer(tuple(_task(f"train-{index}") for index in range(4)))
    assert result["optimizer_updates"] == 0
    assert result["physical_reference_attempts"] == 0
    assert set(result["tasks"]) == {f"train-{index}" for index in range(4)}
    for task in result["tasks"].values():
        assert set(task) == {"observed", "held"}
        assert task["observed"]["same_exact_baseline"]
        assert task["held"]["same_exact_baseline"]
        assert task["observed"]["modes"]["packet_experimental"]["receipt"]["used_pre_mlp_subset_gather"]
        assert task["observed"]["modes"]["packet_experimental"]["receipt"][
            "increment_accumulation_dtype"
        ] == "torch.float64"
        for read in task.values():
            full_rows = read["modes"]["full"]["receipt"]["executed_subset_receipt"]["full_far_rows"]
            control = read["modes"]["padding_only_control"]
            control_receipt = control["receipt"]
            assert control_receipt["active_source_omissions"] == 0
            assert control_receipt["used_pre_mlp_subset_gather"]
            assert control_receipt["executed_subset_receipt"]["fine_rows"] < full_rows
            assert control_receipt["executed_subset_receipt"]["full_far_rows"] == full_rows
            assert read["padding_only_model_distortion_rms"] == pytest.approx(0.0, abs=1e-8)
