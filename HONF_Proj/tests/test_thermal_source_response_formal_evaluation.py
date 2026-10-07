"""Focused binding tests for the opt-in formal 89/90 field evaluator."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import thermal_source_response_evaluate as evaluate


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


@pytest.fixture
def formal_checkpoint(tmp_path, monkeypatch):
    data_binding = {
        "dataset_path": str((tmp_path / "dataset.h5").resolve()),
        "dataset_id": "synthetic-dataset",
        "source_metadata_sha256": "source-sha",
        "training_case_count": 600,
    }
    compatibility_ids = [f"case-{index:02d}" for index in range(90)]
    compatibility_ids[27] = "0273"
    primary_ids = [case_id for case_id in compatibility_ids if case_id != "0273"]
    validation_binding = {
        "source_metadata_sha256": "source-sha",
        "primary_scope": "original_test_excluding_train_duplicate",
        "primary_case_ids": primary_ids,
        "primary_case_count": 89,
        "primary_case_ids_sha256": _sha(primary_ids),
        "compatibility_scope": "original_test_all_rows",
        "compatibility_case_ids": compatibility_ids,
        "compatibility_case_count": 90,
        "compatibility_case_ids_sha256": _sha(compatibility_ids),
        "excluded_training_duplicate_case_id": "0273",
    }
    normalization_binding = {
        "identity": "global_h5_original_train_only_v1",
        "fit_split": "train",
        "training_case_count": 600,
    }
    profile = {
        "profile_name": "thermal_source_response_r_direct_full5000_v1",
        "data": {
            "expected_train_case_count": 600,
            "formal_validation": {
                "primary_scope": "original_test_excluding_train_duplicate",
                "compatibility_scope": "original_test_all_rows",
                "expected_primary_case_count": 89,
                "expected_compatibility_case_count": 90,
                "excluded_training_duplicate_case_id": "0273",
            },
        },
    }
    recipe = {
        "identity": "thermal_source_response_r_direct_formal5000_v1",
        "preferred_response_family": "R-direct",
        "profile": profile,
    }
    checkpoint = {
        "epoch": 5000,
        "formal_workflow_scope": "formal_full_train_v1",
        "startup_benchmark": False,
        "formal_dataset_binding": data_binding,
        "formal_normalization_binding": normalization_binding,
        "formal_validation_binding": validation_binding,
        "fit_identity": {"recipe": recipe},
        "global_normalization_stats": {"field_mean_by_channel": np.zeros(5, dtype=np.float32)},
        "train_config": {"dataset": {
            "packed_h5_path": data_binding["dataset_path"],
            "formal_dataset_binding": data_binding,
            "formal_normalization_binding": normalization_binding,
            "formal_validation_binding": validation_binding,
        }},
    }
    monkeypatch.setattr(evaluate, "bind_original_train", lambda *args, **kwargs: (data_binding, []))
    monkeypatch.setattr(evaluate, "bind_formal_validation", lambda *args, **kwargs:
        (validation_binding, primary_ids, compatibility_ids))
    return checkpoint, data_binding, validation_binding, primary_ids, compatibility_ids


@pytest.mark.parametrize("panel,expected_count", (("canonical89", 89), ("original90", 90)))
def test_formal_panel_uses_checkpoint_order_and_duplicate_policy(formal_checkpoint, panel, expected_count):
    checkpoint, _, binding, primary_ids, compatibility_ids = formal_checkpoint

    scope = evaluate.validate_formal_evaluation_checkpoint(checkpoint, panel=panel)

    expected_ids = primary_ids if panel == "canonical89" else compatibility_ids
    assert scope["case_ids"] == expected_ids
    assert scope["case_count"] == expected_count
    assert ("0273" in scope["case_ids"]) is (panel == "original90")
    assert scope["formal_validation_binding_sha256"] == _sha(binding)


@pytest.mark.parametrize("change", ("epoch", "startup", "profile"))
def test_formal_evaluation_rejects_startup_or_nonendpoint_checkpoint(formal_checkpoint, change):
    checkpoint, *_ = formal_checkpoint
    malformed = json.loads(json.dumps(checkpoint, default=lambda value: value.tolist()
        if isinstance(value, np.ndarray) else value))
    if change == "epoch":
        malformed["epoch"] = 4999
    elif change == "startup":
        malformed["startup_benchmark"] = True
    else:
        malformed["fit_identity"]["recipe"]["profile"]["profile_name"] = "unknown"

    with pytest.raises(ValueError, match="e5000|profile"):
        evaluate.validate_formal_evaluation_checkpoint(malformed, panel="canonical89")


def test_formal_evaluation_rejects_current_catalog_mismatch(formal_checkpoint, monkeypatch):
    checkpoint, _, _, _, _ = formal_checkpoint
    changed = dict(checkpoint["formal_dataset_binding"])
    changed["source_metadata_sha256"] = "changed-source"
    monkeypatch.setattr(evaluate, "bind_original_train", lambda *args, **kwargs: (changed, []))

    with pytest.raises(ValueError, match="TRAIN inputs/metadata"):
        evaluate.validate_formal_evaluation_checkpoint(checkpoint, panel="canonical89")


def test_formal_evaluation_accepts_retained_milestone_profile_variant(formal_checkpoint):
    checkpoint, *_ = formal_checkpoint
    checkpoint["fit_identity"]["recipe"]["profile"]["profile_name"] += "_retained_milestones"

    scope = evaluate.validate_formal_evaluation_checkpoint(checkpoint, panel="canonical89")

    assert scope["panel"] == "canonical89"


def test_formal_panel_rejects_malformed_membership(formal_checkpoint):
    _, _, binding, _, _ = formal_checkpoint
    malformed = dict(binding)
    malformed["primary_case_ids"] = list(binding["primary_case_ids"])
    malformed["primary_case_ids"][0] = "0273"

    with pytest.raises(ValueError, match="panel memberships"):
        evaluate.formal_panel_case_ids(malformed, "canonical89")


def test_formal_native_reader_uses_sealed_ids_and_embedded_normalizer(formal_checkpoint, monkeypatch):
    checkpoint, _, _, primary_ids, _ = formal_checkpoint
    calls = []

    class FakeDataset:
        def __init__(self, path, **kwargs):
            calls.append((str(path), kwargs))
            self.selected_case_ids = kwargs["case_ids"]

    monkeypatch.setattr(evaluate, "GlobalChannelThermalDataset", FakeDataset)

    dataset, scope = evaluate.load_formal_native_cases(checkpoint, "canonical89")

    assert dataset.selected_case_ids == primary_ids
    assert scope["panel"] == "canonical89"
    assert calls[0][1]["split"] == "test"
    assert calls[0][1]["case_ids"] == primary_ids
    assert calls[0][1]["normalize_inputs"] is False
    assert calls[0][1]["normalize_targets"] is False
    assert calls[0][1]["normalizer"].stats is checkpoint["global_normalization_stats"]


def test_classic_formal_reader_requires_same_h5_and_uses_classic_transform(formal_checkpoint, monkeypatch):
    _, data_binding, _, primary_ids, _ = formal_checkpoint
    scope = {
        "dataset_path": data_binding["dataset_path"],
        "case_ids": primary_ids,
    }
    stats = {"field_mean_by_channel": np.arange(5, dtype=np.float32)}
    classic = {"global_normalization_stats": stats, "train_config": {"dataset": {
        "packed_h5_path": data_binding["dataset_path"],
        "normalize_inputs": True,
        "normalize_targets": True,
    }}}
    calls = []

    class FakeDataset:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)
            self.selected_case_ids = kwargs["case_ids"]

    from channelthermal.data import datasets
    monkeypatch.setattr(datasets, "GlobalChannelThermalDataset", FakeDataset)

    normalized, raw, _ = evaluate.load_classic_formal_native_cases(classic, scope)

    assert normalized.selected_case_ids == raw.selected_case_ids == primary_ids
    assert calls[0]["normalize_inputs"] is True and calls[0]["normalize_targets"] is True
    assert calls[1]["normalize_inputs"] is False and calls[1]["normalize_targets"] is False
    np.testing.assert_array_equal(calls[0]["normalizer"].stats["field_mean_by_channel"], stats["field_mean_by_channel"])
    with pytest.raises(ValueError, match="same packed H5"):
        evaluate.load_classic_formal_native_cases(classic, {**scope, "dataset_path": "/different/dataset.h5"})


def test_formal_classic_cli_rejects_development_and_formal_scope_mix():
    with pytest.raises(SystemExit, match="2"):
        evaluate.main([
            "--checkpoint", "/unused/classic.pt",
            "--output-dir", "/unused/output",
            "--mode", "classic-fields",
            "--classic-id", "Run1804",
            "--development-manifest", "/unused/manifest.json",
            "--formal-panel", "canonical89",
            "--formal-reference-checkpoint", "/unused/rdirect.pt",
        ])


def test_formal_physical_aggregates_keep_duplicate_out_of_primary_errors(formal_checkpoint):
    _, _, _, primary_ids, compatibility_ids = formal_checkpoint
    rows = []
    for case_id in compatibility_ids:
        error = 100.0 if case_id == "0273" else 1.0
        rows.append({"case_id": case_id, "metrics": {"fluid/temperature": {
            "finite": True, "count": 1, "rmse": error, "mae": error,
            "squared_error_sum": error ** 2, "absolute_error_sum": error}}})
    result = evaluate.formal_physical_aggregates(rows, "original90")
    primary = result["primary_excluding_0273"]["fluid/temperature"]
    compatibility = result["compatibility_including_0273"]["fluid/temperature"]
    assert primary["cases"] == 89 and primary["equal_case_rmse_mean"] == 1.0
    assert compatibility["cases"] == 90
    assert compatibility["equal_case_rmse_mean"] == pytest.approx(2.1)
    canonical = evaluate.formal_physical_aggregates(
        [row for row in rows if row["case_id"] in primary_ids], "canonical89")
    assert canonical["primary_excluding_0273"] == result["primary_excluding_0273"]
    assert "compatibility_including_0273" not in canonical
