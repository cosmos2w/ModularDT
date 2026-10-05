"""Exercise native profile composition, rather than only inspecting helper JSON."""

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from channelthermal.plugin import create_plugin
from thermal_development import development_profiles

from honf_runtime.config_loader import load_config_bundle


@pytest.mark.parametrize("arm", ["B-native", "B-fine", "H-tree", "H-overlap", "H-local"])
def test_bound_core_profile_composes_exact_membership_into_native_case_config(tmp_path, arm):
    selected = {"manifest_sha256": "a" * 64, "partitions": {"train": {"case_ids": ["0348"]}}}
    profile = development_profiles(manifest=selected, manifest_path=str(tmp_path / "manifest.json"))[arm]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    bundle = load_config_bundle(path)
    create_plugin().validate_config(bundle)
    binding = profile["case"]["dataset"]
    assert all(bundle.effective["dataset"][key] == value for key, value in binding.items())
    assert all(bundle.case["dataset"][key] == value for key, value in binding.items())
    assert bundle.effective["dataset"]["batch_size"] == 48
    assert bundle.effective["dataset"]["val_split"] == "test"
    assert "dataset" not in bundle.effective["case"]["selection"]


@pytest.mark.parametrize("override", [{"packed_h5_path": "arbitrary.h5"}, {"val_split": "train"}, {"typo": 1}])
def test_core_dataset_binding_does_not_bypass_case_contract(tmp_path, override):
    profile = development_profiles()["H-tree"]
    profile["case"]["dataset"].update(override)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="Unknown core.case.dataset"):
        load_config_bundle(path)


@pytest.mark.parametrize("policy", ["packed", "train_only"])
def test_explicit_normalization_policy_composes_into_case_and_effective_dataset(tmp_path, policy):
    profile = development_profiles()["H-tree"]
    profile["case"]["dataset"]["normalization_policy"] = policy
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    bundle = load_config_bundle(path)
    create_plugin().validate_config(bundle)
    assert bundle.case["dataset"]["normalization_policy"] == policy
    assert bundle.effective["dataset"]["normalization_policy"] == policy


@pytest.mark.parametrize("origin", ["core", "case", "experiment"])
def test_merged_normalization_policy_rejects_invalid_values_from_each_source(tmp_path, origin):
    profile = development_profiles()["H-tree"]
    profile["case"]["dataset"]["normalization_policy"] = "packed"
    overlay = None
    if origin == "core":
        profile["case"]["dataset"]["normalization_policy"] = "typo"
    elif origin == "case":
        profile["case"]["dataset"].pop("normalization_policy")
        case_path = TOOLS.parent / "Case_ThermalChannel/configs/case_default.json"
        case = json.loads(case_path.read_text())
        case["dataset"]["normalization_policy"] = "typo"
        local_case = tmp_path / "case.json"
        local_case.write_text(json.dumps(case))
        profile["case"]["config"] = str(local_case)
    else:
        overlay = tmp_path / "experiment.json"
        overlay.write_text(json.dumps({"schema_version": 1, "case": {"dataset": {"normalization_policy": "typo"}}}))
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="normalization_policy"):
        load_config_bundle(path, experiment_overlay=overlay)
