"""Negative admission and resource-boundary checks for the Round7 CLI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import native_retrieval_train as cli
import pytest

from honf_runtime.native_retrieval_protocol import canonical_sha256


def _round_receipts(tmp_path, *, storage_estimate=None, associated_seconds=0.0):
    round_root = tmp_path / "round"
    receipts = round_root / "receipts"
    receipts.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    budget = {
        "root": str(round_root.resolve()),
        "started_utc": (now - timedelta(minutes=10)).isoformat(),
        "mount": {"filesystems": [{"target": "/data2"}]},
        "persistent_hard_bytes": 1_000_000,
        "temporary_hard_bytes": 1_000_000,
        "peak_new_bytes": 2_000_000,
        "minimum_free_bytes": 1,
        "gpu_uuid": cli.GPU2_UUID,
        "gpu_associated_hours": 12,
        "elapsed_hours": 10,
        "closeout_reserved_hours": 1,
    }
    floor = {
        "schema": "round7_gpu_resource_floor_v1",
        "root": str(round_root.resolve()),
        "gpu_uuid": cli.GPU2_UUID,
        "minimum_free_gpu_bytes": 1024,
        "minimum_free_fraction": 0.20,
    }
    recipe = {"dataset": "thermal", "treatment": "G-native-M"}
    binding = {
        "recipe_sha256": canonical_sha256(recipe),
        "maximum_stop_after": 1000,
        "storage_estimate_bytes": storage_estimate or {"persistent": 1000, "temporary": 1000},
        "resource_estimate_hours": {"gpu_associated": 0.1, "elapsed": 0.1},
    }
    admission = {
        "schema": "round7_execution_admission_v1",
        "status": "completed",
        "primary_pairs_admitted": {"thermal": True},
        "optional_E_pilots_admitted": False,
        "full_TRAIN_authorized": False,
        "inverse_or_solver_authorized": False,
        "TEST_access": False,
        "solver_own_new_attempts": 0,
        "primary_pair_bindings": {"thermal": {"G-native-M": binding}},
    }
    (receipts / "storage_resource_budget.json").write_text(json.dumps(budget), encoding="utf-8")
    (receipts / "gpu_resource_floor.json").write_text(json.dumps(floor), encoding="utf-8")
    admission_path = receipts / "execution_admission_decision.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    if associated_seconds:
        (receipts / "qualification.json").write_text(json.dumps({
            "GPU_uuid": cli.GPU2_UUID,
            "associated_seconds": associated_seconds,
        }), encoding="utf-8")
    return round_root, admission_path, recipe, admission


def test_cli_rejects_unadmitted_pair_before_any_model_or_cuda_work(tmp_path):
    _root, receipt_path, recipe, receipt = _round_receipts(tmp_path)
    receipt["primary_pairs_admitted"]["thermal"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="not admitted"):
        cli._validate_admission(receipt_path, recipe, stop_after=600)


def test_cli_resource_check_uses_actual_elapsed_and_gpu_receipts_and_tree_sizes(tmp_path):
    round_root, receipt_path, recipe, _ = _round_receipts(tmp_path, associated_seconds=30.0)
    (round_root / "receipts" / "metadata_index.json").write_text(json.dumps(["figure-a", "figure-b"]))
    with (round_root / "receipts" / "fit_resource_usage.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "schema": "round7_fit_resource_usage_v1",
            "gpu_uuid": cli.GPU2_UUID,
            "associated_seconds": 15.0,
        }) + "\n")
    output = round_root / "runs" / "thermal_g_native"
    _checked_root, budgets = cli._validate_round_resources(
        receipt_path, output, recipe, stop_after=600,
    )
    assert _checked_root == round_root.resolve()
    assert budgets["gpu_associated_seconds_recorded"] == pytest.approx(45.0)
    assert budgets["gpu_associated_seconds_conservative_bound"] >= 10.0 * 60.0
    assert budgets["gpu_associated_hours_used"] == pytest.approx(
        budgets["gpu_associated_seconds_conservative_bound"] / 3600.0
    )
    assert budgets["gpu_associated_hours_remaining"] == pytest.approx(
        12.0 - budgets["gpu_associated_hours_used"]
    )
    assert "not measured GPU-busy time" in budgets["gpu_associated_time_accounting"]
    assert 8.7 < budgets["elapsed_hours_remaining_before_closeout"] < 8.9
    assert budgets["persistent_bytes_used"] > 0
    assert budgets["minimum_free_gpu_fraction"] == pytest.approx(0.20)
    assert budgets["projected_persistent_bytes"] == budgets["persistent_bytes_used"] + 1000


def test_gpu_receipt_scan_recognizes_nested_uuid_and_skips_unbound_metadata(tmp_path):
    round_root, _receipt_path, _recipe, _receipt = _round_receipts(tmp_path, associated_seconds=30.0)
    receipts = round_root / "receipts"
    (receipts / "qualification.json").write_text(json.dumps({
        "GPU": {"uuid": cli.GPU2_UUID},
        "associated_seconds": 7.5,
    }), encoding="utf-8")
    (receipts / "law_probe.json").write_text(json.dumps({
        "schema": "round7_current_native_laws_v1",
        "associated_seconds": 1000.0,
    }), encoding="utf-8")
    (receipts / "other_gpu.json").write_text(json.dumps({
        "GPU": {"uuid": "GPU-other"},
        "associated_seconds": 1000.0,
    }), encoding="utf-8")
    (receipts / "figure_index.json").write_text(json.dumps(["one.pdf", "two.pdf"]))

    assert cli._recorded_gpu2_seconds(receipts) == pytest.approx(7.5)


def test_cli_resource_check_rejects_projected_tree_over_hard_limit(tmp_path):
    round_root, receipt_path, recipe, _receipt = _round_receipts(
        tmp_path, storage_estimate={"persistent": 1_000_000, "temporary": 0},
    )
    # The current receipt tree itself consumes persistent budget, so an estimate
    # equal to the cap must be rejected before the model/provider is built.
    with pytest.raises(OSError, match="projected persistent"):
        cli._validate_round_resources(receipt_path, round_root / "runs" / "candidate", recipe, stop_after=600)


def test_cli_resource_check_rejects_estimate_that_reaches_closeout_reserve(tmp_path):
    round_root, receipt_path, recipe, receipt = _round_receipts(tmp_path)
    receipt["primary_pair_bindings"]["thermal"]["G-native-M"]["resource_estimate_hours"]["elapsed"] = 9.0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="closeout-reserved deadline"):
        cli._validate_round_resources(receipt_path, round_root / "runs" / "candidate", recipe, stop_after=600)


def test_cli_admission_binds_exact_recipe_and_stop_age(tmp_path):
    _root, receipt_path, recipe, _receipt = _round_receipts(tmp_path)
    cli._validate_admission(receipt_path, recipe, stop_after=600)
    with pytest.raises(ValueError, match="recipe hash"):
        cli._validate_admission(receipt_path, {**recipe, "native_backend": "dense_masked"}, stop_after=600)
    with pytest.raises(ValueError, match="absolute stop"):
        cli._validate_admission(receipt_path, recipe, stop_after=1500)
