"""Focused guards for maintained manual full-TRAIN profiles and bindings."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))

from thermal_formal_dsep_flow_fit import _validate_profile as validate_flow_profile
from thermal_formal_profile import (
    ensure_formal_resume_identity,
    formal_train_config,
    validate_formal_bindings,
)
from thermal_source_response_fit import validate_formal_profile as validate_thermal_profile

_PROFILE_DIR = Path(__file__).resolve().parents[1] / "src/config_core/forward/thermal_source_response"


def _profile(name):
    return json.loads((_PROFILE_DIR / name).read_text())


@pytest.mark.parametrize(("name", "validator"), [
    ("d-sep_full5000.json", validate_flow_profile),
    ("r-direct_full5000.json", validate_thermal_profile),
])
def test_formal_profiles_validate_and_require_best_field_checkpointing(name, validator):
    profile = _profile(name)
    assert validator(profile)["schedule"]["horizon_epochs"] == 5000
    bad = copy.deepcopy(profile)
    bad["checkpointing"]["save_best_field"] = False
    with pytest.raises(ValueError):
        validator(bad)


@pytest.mark.parametrize(("name", "validator"), [
    ("d-sep_full5000.json", validate_flow_profile),
    ("r-direct_full5000.json", validate_thermal_profile),
])
def test_formal_retention_subset_keeps_monitoring_and_rejects_invalid_ages(name, validator):
    profile = _profile(name)
    profile["checkpointing"]["milestone_epochs"] = [100, 500, 1000, 2500, 5000]
    assert validator(profile)["checkpointing"]["monitoring_interval_epochs"] == 100
    for ages in ([], [500, 5000], [100, 500], [100, 100, 5000],
                 [100, 250, 5000], [100, 5000, 5100], [100, 5000.0], [5000, 100]):
        invalid = copy.deepcopy(profile)
        invalid["checkpointing"]["milestone_epochs"] = ages
        with pytest.raises(ValueError, match="milestones"):
            validator(invalid)


def test_formal_train_config_drops_development_identity_and_binds_full_scope():
    train_ids = [f"train-{index:03d}" for index in range(600)]
    data_binding = {"scope": "all_original_train", "training_case_ids": train_ids,
        "training_case_count": len(train_ids)}
    normalization_binding = {"fit_split": "train", "training_case_ids_sha256": "ids-sha"}
    validation_binding = {"primary_scope": "canonical89", "compatibility_scope": "original90"}
    base = {"dataset": {"development_manifest": "fixed25", "development_manifest_sha256": "dev-sha",
        "development_subset": "quarter", "development_manifest_path": "/fixed25.json"}}
    result = formal_train_config(base, data_binding, normalization_binding, validation_binding)
    assert result["dataset"]["normalization_scope"] == "all_original_train_only"
    assert result["dataset"]["formal_dataset_binding"] == data_binding
    assert result["dataset"]["formal_normalization_binding"] == normalization_binding
    assert result["dataset"]["formal_validation_binding"] == validation_binding
    assert not {"development_manifest", "development_manifest_sha256", "development_subset",
        "development_manifest_path"}.intersection(result["dataset"])


def test_formal_flow_partner_rejects_changed_membership_or_normalizer_values():
    stats = {"field_mean_by_channel": np.arange(5, dtype=np.float32),
        "field_std_by_channel": np.arange(1, 6, dtype=np.float32)}
    data_binding = {"scope": "all_original_train", "training_case_ids": ["a", "b"],
        "training_case_count": 2}
    normalization_binding = {"fit_split": "train", "training_case_ids_sha256": "ids-sha"}
    validation_binding = {"primary_scope": "canonical89", "compatibility_scope": "original90"}
    checkpoint = {"formal_dataset_binding": data_binding,
        "formal_normalization_binding": normalization_binding,
        "formal_validation_binding": validation_binding,
        "global_normalization_stats": stats,
        "train_config": {"dataset": {"formal_dataset_binding": data_binding,
            "formal_normalization_binding": normalization_binding,
            "formal_validation_binding": validation_binding}}}
    validate_formal_bindings(checkpoint, data_binding, normalization_binding, validation_binding, stats)

    changed_membership = copy.deepcopy(checkpoint)
    changed_membership["formal_dataset_binding"]["training_case_ids"] = ["a", "c"]
    with pytest.raises(ValueError, match="dataset binding"):
        validate_formal_bindings(changed_membership, data_binding, normalization_binding, validation_binding, stats)

    changed_stats = copy.deepcopy(checkpoint)
    changed_stats["global_normalization_stats"]["field_mean_by_channel"][0] = 99
    with pytest.raises(ValueError, match="normalization values"):
        validate_formal_bindings(changed_stats, data_binding, normalization_binding, validation_binding, stats)


def test_formal_resume_binds_existing_output_identity_before_writes(tmp_path):
    identity = {"run_identity": "formal5000", "profile_sha256": "profile-a",
        "flow_reader_config": {"heat_columns": (4,)}}
    output = tmp_path / "empty_run"
    output.mkdir()
    ensure_formal_resume_identity(output, "fit_identity.json", identity)
    assert json.loads((output / "fit_identity.json").read_text()) == {
        **identity, "flow_reader_config": {"heat_columns": [4]}}
    ensure_formal_resume_identity(output, "fit_identity.json", identity)
    with pytest.raises(ValueError, match="different run identity"):
        ensure_formal_resume_identity(output, "fit_identity.json", {
            **identity, "flow_reader_config": {"heat_columns": (3,)}})

    retained = output / "latest_model.pt"
    retained.write_bytes(b"existing checkpoint")
    with pytest.raises(ValueError, match="different run identity"):
        ensure_formal_resume_identity(output, "fit_identity.json", {**identity, "profile_sha256": "profile-b"})
    assert retained.read_bytes() == b"existing checkpoint"

    orphan = tmp_path / "orphan_run"
    orphan.mkdir()
    (orphan / "history.json").write_text("[]")
    with pytest.raises(ValueError, match="without its prepared identity"):
        ensure_formal_resume_identity(orphan, "formal_recipe.json", identity)
