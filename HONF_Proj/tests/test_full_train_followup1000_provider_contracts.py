from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
for source in (ROOT / "Case_ThermalChannel/src", ROOT / "Case_WindFarm/src", ROOT / "src"):
    sys.path.insert(0, str(source))

from channelthermal.training import joint_task as thermal_provider
from full_train_followup_contract import DATA as FOLLOWUP_DATA
from full_train_followup_contract import _case_ids_sha256
from windfarm.training import joint_task as wind_provider


def test_canonical89_provider_binding_preserves_nul_encoding_separate_from_saved_panel() -> None:
    ids = [f"{index:04d}" for index in (*range(274, 303), *range(633, 693))]
    provider_hash = hashlib.sha256(b"".join(value.encode() + b"\0" for value in ids)).hexdigest()
    saved_panel_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    assert len(ids) == 89
    assert provider_hash == "1c33b4cc5ebddb1720a6ba6adcd9ad88beb988302008c58623cf55b2fdea41ab"
    assert saved_panel_hash == "51f0bea2278c26b24a994a29a52ad6b8af1e3e85941dc75d9b20bbebce5d8728"
    assert thermal_provider._case_ids_hash(ids) == _case_ids_sha256(ids) == provider_hash
    assert FOLLOWUP_DATA["thermal"]["validation_sha256"] == provider_hash
    assert provider_hash != saved_panel_hash


def test_provider_sealed_membership_constants_match_runner_contract() -> None:
    thermal = FOLLOWUP_DATA["thermal"]
    assert thermal_provider.FULL_TRAIN_FOLLOWUP_DATASET == thermal["dataset_protocol"]
    assert thermal_provider.FULL_TRAIN_FOLLOWUP_TRAIN_SHA256 == thermal["train_sha256"]
    assert thermal_provider.FULL_TRAIN_FOLLOWUP_VALIDATION_SHA256 == thermal["validation_sha256"]
    wind = FOLLOWUP_DATA["wind"]
    assert wind_provider.FULL_TRAIN_FOLLOWUP_DATASET == wind["dataset_protocol"]
    assert wind_provider.FULL_TRAIN_FOLLOWUP_TRAIN_SHA256 == wind["train_sha256"]
    assert wind_provider.FULL_TRAIN_FOLLOWUP_VALIDATION_SHA256 == wind["validation_sha256"]
    assert wind_provider.FULL_TRAIN_FOLLOWUP_TEST_SHA256 == wind["test_sha256"]


def test_thermal_full_provider_membership_checks_exact_population_before_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    train = [f"train-{index:04d}" for index in range(600)]
    validation = [f"valid-{index:04d}" for index in range(89)]
    monkeypatch.setattr(thermal_provider, "FULL_TRAIN_FOLLOWUP_TRAIN_SHA256",
                        thermal_provider._case_ids_hash(train))
    monkeypatch.setattr(thermal_provider, "FULL_TRAIN_FOLLOWUP_VALIDATION_SHA256",
                        thermal_provider._case_ids_hash(validation))
    binding = thermal_provider.validate_full_train_followup_membership(train, validation)
    assert binding["training_case_count"] == 600
    assert binding["validation_case_count"] == 89
    assert binding["unexposed_original_test_targets_read"] is False
    with pytest.raises(ValueError, match="canonical89"):
        thermal_provider.validate_full_train_followup_membership(train, [*validation, "0273"])
    changed_validation = [*validation[:-1], "other-validation-case"]
    with pytest.raises(ValueError, match="sealed full-population"):
        thermal_provider.validate_full_train_followup_membership(train, changed_validation)


def test_thermal_full_mask_catalog_reads_only_stored_module_masks(tmp_path: Path) -> None:
    train = [f"train-{index:04d}" for index in range(600)]
    validation = [f"valid-{index:04d}" for index in range(89)]
    path = tmp_path / "geometry_only.h5"
    with h5py.File(path, "w") as packed:
        cases = packed.create_group("cases")
        for case_id in (*train, *validation):
            group = cases.create_group(case_id)
            group.create_dataset("module_mask", data=np.zeros((2, 3), dtype=np.uint8))
    masks, binding = thermal_provider._read_fixed_native_mask_catalog(
        path, train, validation, expected_shape=(2, 3), full_train_followup1000=True,
    )
    assert len(masks) == 689
    assert binding["dataset_protocol"] == thermal_provider.FULL_TRAIN_FOLLOWUP_DATASET
    assert binding["target_data_keys_read"] == []
    assert set(binding["training_case_ids"]) == set(train)
    assert set(binding["validation_case_ids"]) == set(validation)


def test_wind_full_provider_binds_disjoint_train_valid_and_locked_test_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, validation, test = np.arange(420), np.arange(420, 510), np.arange(510, 600)
    monkeypatch.setattr(wind_provider, "FULL_TRAIN_FOLLOWUP_TRAIN_SHA256",
                        wind_provider._indices_sha256(train))
    monkeypatch.setattr(wind_provider, "FULL_TRAIN_FOLLOWUP_VALIDATION_SHA256",
                        wind_provider._indices_sha256(validation))
    monkeypatch.setattr(wind_provider, "FULL_TRAIN_FOLLOWUP_TEST_SHA256",
                        wind_provider._indices_sha256(test))
    binding = wind_provider.validate_full_train_followup_membership(
        train, validation, test,
        train_layout_count=140, validation_layout_count=30, test_layout_count=30,
    )
    assert binding["train_row_count"] == 420
    assert binding["validation_row_count"] == 90
    assert binding["test_row_count"] == 90
    assert binding["test_target_values_read"] is False
    overlapped_test = np.sort(np.r_[test[:-1], validation[-1]])
    monkeypatch.setattr(wind_provider, "FULL_TRAIN_FOLLOWUP_TEST_SHA256",
                        wind_provider._indices_sha256(overlapped_test))
    with pytest.raises(ValueError, match="disjoint and exhaustive"):
        wind_provider.validate_full_train_followup_membership(
            train, validation, overlapped_test,
            train_layout_count=140, validation_layout_count=30, test_layout_count=30,
        )
    monkeypatch.setattr(wind_provider, "FULL_TRAIN_FOLLOWUP_TEST_SHA256",
                        wind_provider._indices_sha256(test))
    with pytest.raises(ValueError, match="TEST must preserve exactly 30 layouts"):
        wind_provider.validate_full_train_followup_membership(
            train, validation, test,
            train_layout_count=140, validation_layout_count=30, test_layout_count=29,
        )
