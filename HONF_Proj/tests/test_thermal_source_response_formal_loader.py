"""Full-TRAIN source-response loader scopes preserve strict component binding."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from channelthermal.dependency_flow import CASE_CAPABILITY, DEPENDENCY_ID, ThermalFlowReader
from channelthermal.source_response import (
    FIELD_ORDER,
    SOURCE_RESPONSE_CAPABILITY,
    SOURCE_RESPONSE_ID,
    ThermalSourceResponse,
    load_source_response_model,
)
from channelthermal.training.checkpoints import atomic_save_checkpoint_payload

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from thermal_formal_dsep_flow_fit import _formal_validation_panels, _summarize_flow_rows
from thermal_formal_profile import initialize_formal_output


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _stats():
    return {
        "field_mean_by_channel": np.arange(5, dtype=np.float32),
        "field_std_by_channel": np.arange(2, 7, dtype=np.float32),
    }


def _formal_pair(tmp_path, *, thermal_epoch=250, flow_epoch=5000):
    stats = _stats()
    train_ids = [f"train-{index:03d}" for index in range(600)]
    train_hash = _digest(train_ids)
    data_binding = {
        "schema_version": 1, "scope": "all_original_train", "training_split": "train",
        "dataset_path": "/dataset.h5", "dataset_id": "test_dataset",
        "source_metadata": {}, "source_metadata_sha256": "source-metadata-sha",
        "training_case_ids": train_ids, "training_case_count": 600,
        "training_case_ids_sha256": train_hash,
    }
    normalization_binding = {
        "identity": "global_h5_original_train_only_v1", "fit_function": "test",
        "fit_split": "train", "training_case_ids_sha256": train_hash,
        "training_case_count": 600, "stats_sha256": _digest({key: value.tolist() for key, value in sorted(stats.items())}),
        "stat_names": sorted(stats),
    }
    compatibility_ids = [f"test-{index:02d}" for index in range(90)]
    compatibility_ids[0] = "0273"
    primary_ids = [case_id for case_id in compatibility_ids if case_id != "0273"]
    validation_binding = {
        "schema_version": 1, "source_metadata_sha256": "source-metadata-sha",
        "primary_scope": "original_test_excluding_train_duplicate", "primary_case_ids": primary_ids,
        "primary_case_count": 89, "primary_case_ids_sha256": _digest(primary_ids),
        "compatibility_scope": "original_test_all_rows", "compatibility_case_ids": compatibility_ids,
        "compatibility_case_count": 90, "compatibility_case_ids_sha256": _digest(compatibility_ids),
        "excluded_training_duplicate_case_id": "0273",
    }
    profile = {
        "profile_name": "thermal_source_response_r_direct_full5000_v1",
        "data": {
            "training_split": "train", "normalization_source": "all_original_train_only",
            "expected_train_case_count": 600,
            "startup_validation": {"scope": "fixed25_v1_DEV22_exposed", "expected_case_count": 22,
                "maximum_new_epochs": 3},
            "formal_validation": {"primary_scope": "original_test_excluding_train_duplicate",
                "compatibility_scope": "original_test_all_rows", "expected_primary_case_count": 89,
                "expected_compatibility_case_count": 90, "excluded_training_duplicate_case_id": "0273"},
        },
        "schedule": {"horizon_epochs": 5000},
    }
    flow_profile = {"profile_name": "thermal_source_response_d_sep_full5000_v1",
        "schedule": {"horizon_epochs": 5000}}
    dataset = {
        "packed_h5_path": "/dataset.h5", "formal_dataset_binding": data_binding,
        "formal_normalization_binding": normalization_binding,
        "formal_validation_binding": validation_binding,
    }
    flow_model = ThermalFlowReader("D-sep", {"hidden": 8, "message": 8})
    flow_identity = {
        "run_identity": "formal5000", "profile": flow_profile,
        "formal_dataset_binding": data_binding,
        "formal_normalization_binding": normalization_binding,
        "formal_validation_binding": validation_binding,
    }
    flow_checkpoint = {
        "checkpoint_schema_version": 1, "case_id": "ThermalChannel", "model_family": "honf_forward",
        "workflow": "forward", "stage": DEPENDENCY_ID, "dependency_identity": DEPENDENCY_ID,
        "dependency_policy": "D-sep", "case_capability": CASE_CAPABILITY,
        "formal_workflow_scope": "formal_full_train_v1", "startup_benchmark": False,
        "formal_profile": flow_profile,
        "flow_reader_config": flow_model.reader.config, "flow_state_dict": flow_model.state_dict(),
        "fit_identity": flow_identity, "epoch": flow_epoch, "current_epoch": flow_epoch,
        "global_normalization_stats": stats,
        "formal_dataset_binding": data_binding, "formal_normalization_binding": normalization_binding,
        "formal_validation_binding": validation_binding,
        "train_config": {"dataset": dataset},
    }
    flow_path = tmp_path / "flow.pt"
    atomic_save_checkpoint_payload(flow_path, flow_checkpoint)

    model = ThermalSourceResponse({"mode": "direct", "hidden": 8, "message": 8},
        nx=16, ny=8, environment_nx=3, environment_ny=2)
    recipe = {
        "identity": "thermal_source_response_r_direct_formal5000_v1",
        "workflow_scope": "formal_full_train_v1", "mode": "direct", "preferred_response_family": "R-direct",
        "startup_benchmark": False, "run_identity": "formal5000", "profile": profile,
        "core_configs": {"direct": model.core_config}, "adapter_config": model.adapter_config(),
        "formal_dataset_binding": data_binding, "formal_normalization_binding": normalization_binding,
        "formal_validation_binding": validation_binding,
    }
    thermal_checkpoint = {
        "checkpoint_schema_version": 1, "case_id": "ThermalChannel", "model_family": "honf_forward",
        "workflow": "forward", "source_response_identity": SOURCE_RESPONSE_ID,
        "case_capability": SOURCE_RESPONSE_CAPABILITY, "channel_order": list(FIELD_ORDER),
        "formal_workflow_scope": "formal_full_train_v1", "startup_benchmark": False,
        "source_response_config": {"core": model.core_config, "adapter": model.adapter_config()},
        "thermal_state_dict": model.state_dict(), "fit_identity": {"recipe": recipe, "mode": "direct"},
        "epoch": thermal_epoch, "current_epoch": thermal_epoch,
        "flow_checkpoint": str(flow_path), "flow_checkpoint_sha256": hashlib.sha256(flow_path.read_bytes()).hexdigest(),
        "global_normalization_stats": stats, "formal_dataset_binding": data_binding,
        "formal_normalization_binding": normalization_binding, "formal_validation_binding": validation_binding,
        "train_config": {"dataset": dataset},
    }
    thermal_path = tmp_path / "thermal.pt"
    atomic_save_checkpoint_payload(thermal_path, thermal_checkpoint)
    return thermal_path, thermal_checkpoint, flow_path, flow_checkpoint


def test_formal_loader_accepts_monitored_thermal_prefix_with_exact_flow_endpoint(tmp_path):
    thermal_path, _, _, _ = _formal_pair(tmp_path, thermal_epoch=250, flow_epoch=5000)
    model, metadata = load_source_response_model(thermal_path, device="cpu")
    assert metadata["epoch"] == 250
    assert model.thermal.core.config["mode"] == "direct"


@pytest.mark.parametrize("thermal_epoch", [0, 5001])
def test_formal_loader_rejects_thermal_age_outside_full_profile(tmp_path, thermal_epoch):
    thermal_path, _, _, _ = _formal_pair(tmp_path, thermal_epoch=thermal_epoch, flow_epoch=5000)
    with pytest.raises(ValueError, match="monitored thermal age"):
        load_source_response_model(thermal_path, device="cpu")


def test_formal_loader_rejects_malformed_full_train_count_even_when_bindings_match(tmp_path):
    thermal_path, thermal_checkpoint, flow_path, flow_checkpoint = _formal_pair(tmp_path)
    bad = copy.deepcopy(thermal_checkpoint)
    bad["formal_dataset_binding"]["training_case_count"] = 599
    bad["fit_identity"]["recipe"]["formal_dataset_binding"]["training_case_count"] = 599
    bad["train_config"]["dataset"]["formal_dataset_binding"]["training_case_count"] = 599
    bad_flow = copy.deepcopy(flow_checkpoint)
    bad_flow["formal_dataset_binding"]["training_case_count"] = 599
    bad_flow["fit_identity"]["formal_dataset_binding"]["training_case_count"] = 599
    bad_flow["train_config"]["dataset"]["formal_dataset_binding"]["training_case_count"] = 599
    atomic_save_checkpoint_payload(flow_path, bad_flow)
    bad["flow_checkpoint_sha256"] = hashlib.sha256(flow_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="full-TRAIN profile"):
        load_source_response_model(thermal_path, checkpoint=bad)


def test_formal_loader_requires_the_completed_flow_endpoint(tmp_path):
    thermal_path, _, _, _ = _formal_pair(tmp_path, thermal_epoch=250, flow_epoch=4999)
    with pytest.raises(ValueError, match="exact e5000 flow endpoint"):
        load_source_response_model(thermal_path, device="cpu")


def test_formal_loader_rejects_unknown_explicit_workflow_scope(tmp_path):
    thermal_path, checkpoint, _, _ = _formal_pair(tmp_path)
    bad = copy.deepcopy(checkpoint)
    bad["formal_workflow_scope"] = "quarter_data_disguised_as_formal"
    with pytest.raises(ValueError, match="unsupported explicit workflow scope"):
        load_source_response_model(thermal_path, checkpoint=bad)


def test_formal_prepare_only_refuses_nonempty_output_without_overwriting(tmp_path):
    output = tmp_path / "existing_formal_run"
    output.mkdir()
    retained = output / "latest_model.pt"
    retained.write_bytes(b"preserve existing checkpoint")
    with pytest.raises(ValueError, match="nonempty output directory"):
        initialize_formal_output(output, prepare_only=True)
    assert retained.read_bytes() == b"preserve existing checkpoint"
    fresh = initialize_formal_output(tmp_path / "fresh_formal_run", prepare_only=True)
    assert fresh.is_dir() and not list(fresh.iterdir())


def test_formal_flow_validation_splits_original90_into_canonical89_and_compatibility():
    compatibility_ids = ["0273"] + [f"test-{index:02d}" for index in range(89)]
    primary_ids = compatibility_ids[1:]
    rows = [{"case_id": case_id, "mse_by_flow_channel": ([100.0] * 4 if case_id == "0273" else [1.0] * 4)}
        for case_id in compatibility_ids]
    primary_rows, compatibility_rows = _formal_validation_panels(rows, primary_ids, compatibility_ids)
    assert len(primary_rows) == 89
    assert len(compatibility_rows) == 90
    assert "0273" not in {row["case_id"] for row in primary_rows}
    assert "0273" in {row["case_id"] for row in compatibility_rows}
    assert _summarize_flow_rows(primary_rows)["standardized_four_field_mse"] == 1.0
    assert _summarize_flow_rows(compatibility_rows)["standardized_four_field_mse"] > 1.0
