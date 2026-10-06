"""Preparation recipes bind each own fit100 without launching or changing cohorts."""

import copy
import json

import pytest
from channelthermal.training.refinement_policy import POLICY_DECLARATION, REFINEMENT_KIND

from honf_runtime.config_loader import load_config_bundle
from tests.test_refinement_policy import fixture
from tools.thermal_response_refinement import TEMPLATE_ROOT, build_refinement_profile, main


def source_fixture():
    parent, _, _ = fixture()
    parent["model_config"]["channelthermal"]["fixed_heat_scale"] = 2.
    parent["train_config"]["model"] = copy.deepcopy(parent["model_config"])
    parent["train_config"]["dataset"]["development_manifest"] = "/data/fixed25_v1/manifest.json"
    source = {"schema_version": 1, "workflow": "forward", "model_family": "honf_forward",
              "case": {"id": "ThermalChannel", "dataset": {"primary_setting": "retained"}},
              "model": copy.deepcopy(parent["model_config"]), "training": copy.deepcopy(parent["train_config"]["training"]),
              "checkpointing": {}, "run": {"id": "3701"}}
    return parent, source


@pytest.mark.parametrize("arm,mode,run_id", [("H-add", "add", 3801), ("H-joint", "joint", 3802)])
def test_durable_templates_have_identical_policy_and_cadence(arm, mode, run_id):
    template = json.loads((TEMPLATE_ROOT / f"{arm.lower()}.json").read_text())
    campaign = template["training"]["campaign"]
    assert campaign["forward_refinement"] == POLICY_DECLARATION
    assert campaign["parent"] == {"kind": REFINEMENT_KIND, "epoch": 100, "checkpoint": None}
    assert template["model"]["core_honf"]["interface_model"]["hypergraph_options"] == {
        "query_interaction_mode": mode, "query_admission_mode": "soft"}
    assert template["run"]["id"] == str(run_id)
    assert template["checkpointing"]["save_epoch_milestones"] == list(range(100, 1001, 100))


def test_profile_preserves_source_data_primary_identity_and_records_new_response_exposure(tmp_path):
    parent, source = source_fixture()
    before = copy.deepcopy(parent)
    calibration = {"response_coefficient": .2, "null_coefficient": .3, "scope": "pooled initial train-only"}
    profile = build_refinement_profile(parent, source_profile=source, parent_path=tmp_path / "epoch_100_model.pt",
        arm="H-add", run_id=3801, output_root=tmp_path, atlas_directory=tmp_path / "atlas", calibration=calibration)
    assert "interface_fit" not in profile["training"]["campaign"]
    assert "heat_null_response" not in profile["training"]["campaign"]
    assert profile["training"]["campaign"]["response_refinement"]["calibration"] == calibration
    assert profile["training"]["campaign"]["response_addendum"]["fit_anchor_ids"] == ["0001", "0318", "0333", "0348"]
    assert profile["case"]["dataset"]["development_manifest"] == before["train_config"]["dataset"]["development_manifest"]
    assert profile["case"]["dataset"]["primary_setting"] == "retained"
    assert profile["training"]["learning_rate"] == 1e-5 and profile["training"]["epochs"] == 100
    assert parent["train_config"]["dataset"] == before["train_config"]["dataset"]
    assert "query_admission_mode" not in parent["model_config"]["core_honf"]["interface_model"]["hypergraph_options"]


@pytest.mark.parametrize("change", ["fit300", "wrong_arm", "negative_calibration"])
def test_profile_rejects_unmatched_parent_or_invalid_calibration(tmp_path, change):
    parent, source = source_fixture()
    calibration = {"response_coefficient": .2, "null_coefficient": .3}
    if change == "fit300": parent["epoch"] = 300
    elif change == "wrong_arm": parent["model_config"]["core_honf"]["interface_model"]["hypergraph_options"]["query_interaction_mode"] = "joint"
    else: calibration["null_coefficient"] = 0
    with pytest.raises(ValueError):
        build_refinement_profile(parent, source_profile=source, parent_path=tmp_path / "epoch_100_model.pt",
            arm="H-add", run_id=3801, output_root=tmp_path, atlas_directory=tmp_path, calibration=calibration)


def test_cli_rejects_missing_calibration_before_any_preparation_write(tmp_path):
    with pytest.raises(SystemExit): main(["--output-root", str(tmp_path)])
    assert not list(tmp_path.iterdir())


def test_resolved_runtime_training_identity_is_kept_out_of_core_profile(tmp_path):
    parent, source = source_fixture()
    source["case"]["dataset"].pop("primary_setting")
    parent["train_config"]["training"].update(Run_ID="3701", run_name="historical_fit")
    source["case"].update(config="project://Case_ThermalChannel/configs/case_source_local.json",
                          dataset_id="thermal_channel_global_v1")
    profile = build_refinement_profile(parent, source_profile=source, parent_path=tmp_path / "epoch_0100_model.pt",
        arm="H-add", run_id=3801, output_root=tmp_path, atlas_directory=tmp_path,
        calibration={"response_coefficient": .2, "null_coefficient": .3})
    assert "Run_ID" not in profile["training"] and "run_name" not in profile["training"]
    assert parent["train_config"]["training"]["Run_ID"] == "3701"
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    bundle = load_config_bundle(profile_path)
    assert bundle.effective["Run_ID"] == "3801"
