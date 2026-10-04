"""Manual long-run preparation preserves parents and never creates optimizers."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from honf_runtime.config_loader import load_config_bundle

_PATH = Path(__file__).resolve().parents[1] / "tools/thermal_campaign_long_run.py"
_SPEC = importlib.util.spec_from_file_location("thermal_campaign_long_run", _PATH)
launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launcher)
PROFILE = Path(__file__).resolve().parents[1] / "src/config_core/forward/thermal_campaign/h-tree_e1000.json"
PROFILE500 = PROFILE.with_name("h-tree_e500.json")


@pytest.fixture(autouse=True)
def no_optimizer_or_native_resources(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Launch preparation must never instantiate an optimizer.")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path / "fixture_project")
    monkeypatch.setattr(launcher.ThermalChannelPlugin, "inspect_launch", lambda *args: {"fixture": "resource facts"})
    def native_config(self, bundle, request, run_dir):
        config = copy.deepcopy(bundle.effective)
        config["dataset"]["packed_h5_path"] = "fixture.h5"
        return config
    monkeypatch.setattr(launcher.ThermalChannelPlugin, "_forward_config", native_config)
    class MetadataDataset:
        field_dim, material_param_dim, n_interface_points = 5, 6, 64
        selected_module_counts = (1, 2, 3, 4, 5, 6, 7, 9, 10, 12)
        def __init__(self, *args, **kwargs):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return {"structure": {"domain_length_x": [30.], "domain_length_y": [20.],
                                   "material_params": torch.tensor([0., 0., 0., 0., 0., 1.])}}
    monkeypatch.setattr(launcher, "GlobalChannelThermalDataset", MetadataDataset)


def _parent(tmp_path, completed_stage=1000):
    profile = PROFILE500 if completed_stage == 500 else PROFILE
    config = copy.deepcopy(load_config_bundle(profile).effective)
    config["dataset"]["packed_h5_path"] = "fixture.h5"
    checkpoint = {"epoch": completed_stage, "train_config": config, "model_config": launcher.resolved_native_model_config(config),
        "model_state_dict": {"weight": torch.tensor([1., 2.])},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(completed_stage * 13), "exp_avg": torch.tensor([.2])}},
                                 "param_groups": [{"params": [0], "lr": .0003}]},
        "rng_state": {key: {"fixture": key} for key in ("python", "numpy", "torch", "cuda")},
        "selection_state": {"epoch": completed_stage, "total_epochs": 5000},
        "campaign_training_state": _calibration()}
    parent = tmp_path / f"finished{completed_stage}"
    parent.mkdir()
    path = parent / f"epoch_{completed_stage}_model.pt"
    torch.save(checkpoint, path)
    (parent / "summary.json").write_text('{"finished": true}\n')
    return path, checkpoint


def _calibration():
    strata = [[1, 2], [3, 4], [5, 6], [7, 9], [10, 12]]
    return {"physical_loss_policy_amendment": {"physical_loss_policy_from": 1,
            "physical_loss_policy_to": 2, "checkpoint_epoch": 100, "activation_epoch": 101},
        "response_scale_samples": [.1] * 5, "response_scale": .1,
        "response_native_gradient_norm": 1., "response_gradient_norm": .1,
        "calibration_policy_version": 2, "structural_calibration_policy": "training_module_count_strata_v2",
        "structural_calibration_complete": True, "structural_calibration_planned_strata": strata,
        "structural_calibration_observed_strata": strata,
        "structural_calibration_samples": [{"stratum": value, "observed_module_counts": value,
            "epoch": 26, "native_batch": index + 1, "task_organizer_gradient_norm": 1.,
            "cost_organizer_gradient_norm": 1., "coefficient": .001} for index, value in enumerate(strata)],
        "structural_scale_samples": [.001] * 5, "structural_scale": .001,
        "last_task_organizer_gradient_norm": 1., "last_cost_organizer_gradient_norm": 1.}


@pytest.mark.parametrize("key,value", [
    ("response_scale_samples", [.1] * 4), ("response_scale_samples", [.1] * 4 + [float("nan")]),
    ("response_scale", .2), ("response_scale", .05), ("response_gradient_norm", float("inf")),
    ("response_native_gradient_norm", -1.), ("response_gradient_norm", 100.),
    ("physical_loss_policy_amendment", {}), ("physical_loss_policy_amendment", [2, 101]),
    ("structural_calibration_complete", False),
    ("calibration_policy_version", 1), ("structural_calibration_planned_strata", [[1, 2]] * 5),
    ("structural_calibration_samples", []), ("structural_scale_samples", [.001] * 4),
    ("structural_scale", .0001), ("last_cost_organizer_gradient_norm", .5),
])
def test_manual_parent_rejects_missing_or_inconsistent_calibration(tmp_path, key, value):
    _, checkpoint = _parent(tmp_path)
    checkpoint["campaign_training_state"][key] = value
    with pytest.raises(ValueError, match="calibration|policy2"):
        launcher.validate_parent_checkpoint(checkpoint, checkpoint["train_config"])


def test_manual_parent_missing_calibration_and_horizon_are_rejected(tmp_path):
    _, checkpoint = _parent(tmp_path)
    for state in (None, {}):
        changed = copy.deepcopy(checkpoint)
        changed["campaign_training_state"] = state
        with pytest.raises(ValueError, match="saved campaign calibration"):
            launcher.validate_parent_checkpoint(changed, changed["train_config"])
    for selection in ({"epoch": 1000, "total_epochs": 1000}, {"epoch": 999, "total_epochs": 5000}, None):
        changed = copy.deepcopy(checkpoint)
        changed["selection_state"] = selection
        with pytest.raises(ValueError, match="selection and horizon"):
            launcher.validate_parent_checkpoint(changed, changed["train_config"])


@pytest.mark.parametrize("field,value", [
    ("observed_module_counts", [{}]), ("observed_module_counts", [12]),
    ("observed_module_counts", []), ("stratum", {"wrong": [1, 2]}),
    ("task_organizer_gradient_norm", float("nan")), ("coefficient", .1),
    ("epoch", 0), ("native_batch", 0),
])
def test_invalid_structural_sample_is_a_validation_error(tmp_path, field, value):
    _, checkpoint = _parent(tmp_path)
    checkpoint["campaign_training_state"]["structural_calibration_samples"][0][field] = value
    with pytest.raises(ValueError, match="structural calibration"):
        launcher.validate_parent_checkpoint(checkpoint, checkpoint["train_config"])


def test_nondict_and_duplicate_structural_samples_are_rejected(tmp_path):
    _, checkpoint = _parent(tmp_path)
    records = checkpoint["campaign_training_state"]["structural_calibration_samples"]
    for value in (None, copy.deepcopy(records[1])):
        changed = copy.deepcopy(checkpoint)
        changed["campaign_training_state"]["structural_calibration_samples"][0] = value
        with pytest.raises(ValueError, match="structural calibration"):
            launcher.validate_parent_checkpoint(changed, changed["train_config"])


@pytest.mark.parametrize("architecture", ["dense_pairwise_field", "three_term_full_access_honf"])
def test_baselines_preserve_response_calibration_without_organizer_state(tmp_path, architecture):
    _, checkpoint = _parent(tmp_path)
    checkpoint["train_config"]["model"]["core_honf"]["forward_architecture"] = architecture
    checkpoint["selection_state"] = {"epoch": None, "total_epochs": None}
    checkpoint["campaign_training_state"] = {key: value for key, value in _calibration().items()
                                            if key.startswith("response_") or key == "physical_loss_policy_amendment"}
    launcher.validate_parent_calibration(checkpoint, checkpoint["train_config"])


@pytest.mark.parametrize("completed_stage", [500, 1000])
def test_exact_stage_preparation_uses_new_runstore_and_preserves_parent(tmp_path, completed_stage):
    parent, checkpoint = _parent(tmp_path, completed_stage)
    source_profile = PROFILE500 if completed_stage == 500 else PROFILE
    # Retain the existing default1000 API while opting into stage500 explicitly.
    stage_option = {"completed_stage": 500} if completed_stage == 500 else {}
    original = {p.name: p.read_bytes() for p in parent.parent.iterdir()}
    output = tmp_path / "data_results"
    dry = launcher.prepare_launch(source_profile, run_id="3203", output_root=output, physical_gpu=1,
                                  fresh=False, parent_checkpoint=parent, **stage_option)
    assert not output.exists() and not dry["prepared"]
    prepared = launcher.prepare_launch(source_profile, run_id="3203", output_root=output, physical_gpu=2,
                                       fresh=False, parent_checkpoint=parent, prepare=True, **stage_option)
    run = Path(prepared["new_workspace"])
    copied = run / f"continuation_parent_epoch_{completed_stage}_model.pt"
    assert copied.read_bytes() == parent.read_bytes()
    assert {p.name: p.read_bytes() for p in parent.parent.iterdir()} == original
    loaded = torch.load(copied, weights_only=False)
    assert loaded["campaign_training_state"] == checkpoint["campaign_training_state"]
    assert int(loaded["optimizer_state_dict"]["state"][0]["step"]) == completed_stage * 13
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["status"] == "created" and manifest["continuation"]["next_epoch"] == completed_stage + 1
    assert manifest["continuation"]["mode"] == f"exact{completed_stage}_continuation"
    assert prepared["completed_stage_epoch"] == completed_stage
    assert manifest["continuation"]["completed_stage_epoch"] == completed_stage
    assert manifest["continuation"]["parent_checkpoint"] == str(parent.resolve())
    profile = json.loads(Path(prepared["profile"]).read_text())
    assert profile["run"]["output_root"] == str(output.resolve())
    assert profile["training"]["epochs"] == 5000
    assert f"continue{completed_stage}_e5000" in profile["run"]["name"]
    assert "CUDA_VISIBLE_DEVICES=2" in prepared["launch_command"] and "--device cuda:0" in prepared["launch_command"]
    assert str(copied) in prepared["launch_command"] and "--yes" in prepared["launch_command"]
    assert "--dry-run" in prepared["validate_command"]


@pytest.mark.parametrize("completed_stage", [500, 1000])
def test_structural_calibration_cannot_postdate_the_reviewed_stage(tmp_path, completed_stage):
    _, checkpoint = _parent(tmp_path, completed_stage)
    checkpoint["campaign_training_state"]["structural_calibration_samples"][0]["epoch"] = completed_stage
    launcher.validate_parent_calibration(checkpoint, checkpoint["train_config"], completed_stage=completed_stage)
    checkpoint["campaign_training_state"]["structural_calibration_samples"][0]["epoch"] = completed_stage + 1
    with pytest.raises(ValueError, match="training-input provenance"):
        launcher.validate_parent_calibration(checkpoint, checkpoint["train_config"], completed_stage=completed_stage)


@pytest.mark.parametrize("epoch", [467, 500.5, 1000])
def test_stage500_requires_exact_parent_and_selection(tmp_path, epoch):
    _, checkpoint = _parent(tmp_path, 500)
    checkpoint["epoch"] = epoch
    with pytest.raises(ValueError, match="exact completed epoch500"):
        launcher.validate_parent_checkpoint(checkpoint, checkpoint["train_config"], completed_stage=500)


@pytest.mark.parametrize("selection", [
    {"epoch": 467, "total_epochs": 5000}, {"epoch": 1000, "total_epochs": 5000},
    {"epoch": 500, "total_epochs": 500}, None,
])
def test_stage500_selection_and_horizon_must_match(tmp_path, selection):
    _, checkpoint = _parent(tmp_path, 500)
    checkpoint["selection_state"] = selection
    with pytest.raises(ValueError, match="epoch500 selection and horizon5000"):
        launcher.validate_parent_checkpoint(checkpoint, checkpoint["train_config"], completed_stage=500)


@pytest.mark.parametrize("source_profile,completed_stage", [(PROFILE, 500), (PROFILE500, 1000)])
def test_stage_profile_mismatch_fails_before_checkpoint_loading(tmp_path, monkeypatch, source_profile, completed_stage):
    def forbidden(*args, **kwargs):
        raise AssertionError("Wrong source duration must fail before loading a parent.")
    monkeypatch.setattr(launcher, "load_trusted_checkpoint", forbidden)
    output = tmp_path / "data_results"
    with pytest.raises(ValueError, match=f"reviewed e{completed_stage} source profile"):
        launcher.prepare_launch(source_profile, run_id="3203", output_root=output, physical_gpu=1,
                                fresh=False, parent_checkpoint=tmp_path / "unread.pt", completed_stage=completed_stage)
    assert not output.exists()


@pytest.mark.parametrize("completed_stage", [467, 501, 750, 5000])
def test_unreviewed_stages_are_rejected(tmp_path, completed_stage):
    core = load_config_bundle(PROFILE).core
    with pytest.raises(ValueError, match="completed stage of 500 or 1000"):
        launcher.build_profile(core, run_id="3203", output_root=tmp_path, fresh=True,
                               completed_stage=completed_stage)


def test_stage500_fresh_recipe_is_still_seed0_without_parent(tmp_path):
    prepared = launcher.prepare_launch(PROFILE500, run_id="3303", output_root=tmp_path / "results",
                                       physical_gpu=1, fresh=True, prepare=True, completed_stage=500)
    profile = json.loads(Path(prepared["profile"]).read_text())
    assert prepared["lineage"]["mode"] == "fresh_seed0"
    assert prepared["lineage"]["completed_stage_epoch"] == 500
    assert "fresh_seed0_e5000" in profile["run"]["name"]
    assert profile["training"]["seed"] == 0
    assert "--resume-checkpoint" not in prepared["launch_command"]


@pytest.mark.parametrize("option,expected", [([], 1000), (["--completed-stage", "500"], 500)])
def test_cli_propagates_reviewed_stage_without_running_native(tmp_path, monkeypatch, capsys, option, expected):
    captured = {}
    def prepare(profile, **kwargs):
        captured.update(kwargs)
        return {"completed_stage_epoch": kwargs["completed_stage"], "training_launched": False}
    monkeypatch.setattr(launcher, "prepare_launch", prepare)
    monkeypatch.setattr(launcher.sys, "argv", [str(_PATH), "--profile", str(PROFILE), "--run-id", "3203",
        "--output-root", str(tmp_path), "--physical-gpu", "1", "--fresh", *option])
    assert launcher.main() == 0
    assert captured["completed_stage"] == expected
    assert json.loads(capsys.readouterr().out)["completed_stage_epoch"] == expected


def test_fresh_seed0_prepares_only_one_profile_without_a_run_or_optimizer(tmp_path):
    output = tmp_path / "data_results"
    prepared = launcher.prepare_launch(PROFILE, run_id="3303", output_root=output, physical_gpu=1, fresh=True, prepare=True)
    profile = json.loads(Path(prepared["profile"]).read_text())
    assert list(output.glob("launch_profiles/*.json")) == [Path(prepared["profile"])]
    assert not (output / "ThermalChannel").exists()
    assert profile["training"]["seed"] == 0
    assert profile["training"]["campaign"]["native_loss_denominators_start_epoch"] == 101
    assert "--resume-checkpoint" not in prepared["launch_command"]
    with pytest.raises(ValueError, match="seed0"):
        launcher.prepare_launch(PROFILE, run_id="3403", output_root=output, physical_gpu=1, fresh=True, seed=1)
    with pytest.raises(ValueError, match="unused run ID"):
        launcher.prepare_launch(PROFILE, run_id="2203", output_root=output, physical_gpu=1, fresh=True)
    ordinary = copy.deepcopy(load_config_bundle(PROFILE).core)
    ordinary["training"]["seed"] = 1
    with pytest.raises(ValueError, match="seed 0"):
        launcher.validate_campaign(ordinary)
    ordinary["training"]["seed"] = 0
    ordinary["training"]["campaign"]["arm"] = "extra-screen"
    with pytest.raises(ValueError, match="finite five-arm"):
        launcher.build_profile(ordinary, run_id="3403", output_root=output, fresh=True)


def test_invalid_parent_epoch_state_or_policy_is_rejected(tmp_path):
    _, checkpoint = _parent(tmp_path)
    native = copy.deepcopy(checkpoint["train_config"])
    native["training"]["epochs"] = 5000
    for epoch in (100, 999, 1001):
        changed = copy.deepcopy(checkpoint)
        changed["epoch"] = epoch
        with pytest.raises(ValueError, match="epoch1000"):
            launcher.validate_parent_checkpoint(changed, native)
    for key in ("optimizer_state_dict", "rng_state", "model_state_dict"):
        changed = copy.deepcopy(checkpoint)
        changed.pop(key)
        with pytest.raises(ValueError):
            launcher.validate_parent_checkpoint(changed, native)
    changed = copy.deepcopy(checkpoint)
    changed["rng_state"]["torch"] = None
    with pytest.raises(ValueError, match="non-null"):
        launcher.validate_parent_checkpoint(changed, native)
    changed = copy.deepcopy(native)
    changed["training"]["campaign"]["physical_loss_policy_version"] = 1
    with pytest.raises(ValueError, match="schedule/lineage"):
        launcher.validate_parent_checkpoint(checkpoint, changed)
    for section, key, value in (("channelthermal", "heat_scale", 2.),
                                ("local_coupling", "freeze_local_surrogate", False),
                                ("local_coupling", "local_surrogate_checkpoint_path", "another_stageA.pt"),
                                ("core_honf", "hidden_dim", 64)):
        changed = copy.deepcopy(native)
        changed["model"][section][key] = value
        with pytest.raises(ValueError, match="native model configuration or Stage-A"):
            launcher.validate_parent_checkpoint(checkpoint, changed)
    with pytest.raises(ValueError, match="GPU1 or GPU2"):
        launcher.launch_commands(tmp_path / "profile.json", physical_gpu=0)
