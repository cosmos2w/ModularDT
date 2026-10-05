"""Development selection/configuration guards without native models or optimizers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from channelthermal.data import development_split
from channelthermal.plugin import ThermalChannelPlugin
from channelthermal.training.campaign import CampaignMicrobatchLoader, validate_campaign_resume
from channelthermal.training.checkpoints import _validate_resume_checkpoint
from channelthermal.workflows import train_forward as training

from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.config_loader import load_config_bundle


def fixture_config():
    manifest = {"manifest_sha256": "a" * 64, "policy": {"seed": 7, "fraction": .25},
                "source": {"metadata_sha256": "b" * 64},
                "partitions": {"train": {"case_ids": ["train_a", "train_b"]},
                               "test": {"case_ids": ["test_a"]}}}
    config = {"dataset": {"packed_h5_path": "fixture.h5", "development_manifest": "fixed.json",
                "development_manifest_sha256": manifest["manifest_sha256"], "dataset_id": "fixture",
                "dataset_schema": "fixture1", "dataset_fingerprint": "c" * 64},
              "training": {"seed": 0, "epochs": 500, "campaign": {
                  "require_full_epoch": True, "schedule_total_epochs": 1000,
                  "response_stencils": ["fixture_train_response.npz"]}}, "checkpointing": {}}
    return config, manifest


@pytest.fixture
def fixed_manifest(monkeypatch):
    config, manifest = fixture_config()
    monkeypatch.setattr(development_split, "resolve_development_manifest", lambda *args: copy.deepcopy(manifest))
    return config, manifest


def test_development_defaults_and_manifest_are_bound_before_dataset_creation(fixed_manifest, monkeypatch):
    config, manifest = fixed_manifest
    bound = training.resolve_development_training(config)
    assert config["dataset"]["development_subset"] == manifest
    assert config["checkpointing"] == {"save_best": False, "save_best_field_mse": True,
        "save_best_temperature_mse": False, "save_best_predicted": False, "save_latest": True,
        "save_latest_every_epochs": 100,
        "save_best_every_epochs": 100, "save_epoch_milestones": [100, 200, 300, 400, 500]}
    assert config["training"]["plot_every_epochs"] == 100
    calls = []

    class Dataset:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))
            self.selected_case_ids = tuple(kwargs["case_ids"])
            self.normalizer = kwargs.get("normalizer", SimpleNamespace(fitted_case_ids=self.selected_case_ids))

    monkeypatch.setattr(training, "GlobalChannelThermalDataset", Dataset)
    train, val = training.build_training_datasets(config, bound)
    assert calls[0][1]["case_ids"] == ("train_a", "train_b")
    assert calls[1][1]["case_ids"] == ("test_a",)
    assert train.normalizer.fitted_case_ids == ("train_a", "train_b")
    assert val.normalizer is train.normalizer
    assert calls[1][1]["normalizer"] is train.normalizer
    config["dataset"]["val_split"] = "val"
    training.build_training_datasets(config, bound)
    assert calls[-1][1]["split"] == "test"
    assert calls[-1][1]["case_ids"] == ("test_a",)


def test_launch_inspection_validates_binding_and_reports_selected_native_coverage(fixed_manifest, monkeypatch):
    config, manifest = fixed_manifest
    config["dataset"]["batch_size"] = 48
    manifest["partitions"]["train"]["case_ids"] = [f"train_{index}" for index in range(150)]
    manifest["partitions"]["test"]["case_ids"] = [f"test_{index}" for index in range(22)]
    monkeypatch.setattr(development_split, "resolve_development_manifest", lambda *args: copy.deepcopy(manifest))
    resource = SimpleNamespace(dataset_id="fixture", path=Path("fixture.h5"),
        record={"schema": "fixture1", "num_cases": 690, "splits": {"train": 600, "test": 90}}, fingerprint="c" * 64)
    monkeypatch.setattr(ThermalChannelPlugin, "_resource", lambda *args: resource)
    bundle = SimpleNamespace(effective=config, case={"model": {"local_coupling": {"use_local_surrogate": False}}})
    before = copy.deepcopy(config)
    facts = ThermalChannelPlugin().inspect_launch(bundle, WorkflowRequest(workflow="forward"))
    assert facts["selected dataset splits"] == {"train": 150, "test": 22}
    assert facts["native optimizer batches per epoch"] == 4
    assert facts["development schedule epochs"] == 1000
    assert facts["development manifest sha256"] == manifest["manifest_sha256"]
    assert facts["dataset splits"] == {"train": 600, "test": 90}
    assert "selected training cases only" in facts["normalization scope"]
    assert config == before
    with pytest.raises(ValueError, match="fresh initialization"):
        ThermalChannelPlugin().inspect_launch(bundle, WorkflowRequest(workflow="forward", initialize_checkpoint=Path("old.pt")))
    config["checkpointing"]["save_best_every_epochs"] = 1
    with pytest.raises(ValueError, match="cadence"):
        ThermalChannelPlugin().inspect_launch(bundle, WorkflowRequest(workflow="forward"))
    config["checkpointing"].clear()

    def unbound(*args):
        raise ValueError("Development manifest SHA binding is required")

    monkeypatch.setattr(development_split, "resolve_development_manifest", unbound)
    with pytest.raises(ValueError, match="SHA binding"):
        ThermalChannelPlugin().inspect_launch(bundle, WorkflowRequest(workflow="forward"))


def test_full_training_configuration_and_dataset_defaults_remain_unchanged(monkeypatch):
    config = {"dataset": {"packed_h5_path": "fixture.h5"}, "training": {"seed": 0},
              "checkpointing": {"save_latest_every_epochs": 25}}
    before = copy.deepcopy(config)
    assert training.resolve_development_training(config) is None
    assert config == before
    calls = []

    class Dataset:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)
            self.normalizer = kwargs.get("normalizer", object())

    monkeypatch.setattr(training, "GlobalChannelThermalDataset", Dataset)
    training.build_training_datasets(config)
    assert all("case_ids" not in kwargs for kwargs in calls)
    assert training.should_save_best_checkpoint(1, 1000, {})
    assert training.should_save_latest_checkpoint(25, 1000, before["checkpointing"])


@pytest.mark.parametrize("policy,split", [("typo", "train"), (None, "train"), (["train_only"], "train"),
                                         ("train_only", "test"), ("train_only", "all")])
def test_normalization_policy_rejected_before_dataset_creation(monkeypatch, policy, split):
    config = {"dataset": {"normalization_policy": policy, "train_split": split}, "training": {}}
    monkeypatch.setattr(training, "GlobalChannelThermalDataset", lambda *args, **kwargs: pytest.fail("Dataset constructed."))
    with pytest.raises(ValueError, match="normalization|train split"):
        training.build_training_datasets(config)


@pytest.mark.parametrize("section,key,value", [
    ("checkpointing", "save_latest_every_epochs", 25),
    ("checkpointing", "save_best_every_epochs", 1),
    ("checkpointing", "save_epoch_milestones", [25, 100]),
    ("training", "plot_every_epochs", 50),
    ("training", "max_train_batches_per_epoch", 1),
    ("training", "max_val_batches", 1),
    ("dataset", "allow_train_as_validation", True),
    ("dataset", "val_split", "train"),
    ("dataset", "val_split", "all"),
    ("dataset", "train_split", "test"),
    ("training", "init_checkpoint_path", "old.pt"),
    ("checkpointing", "save_best", True),
    ("checkpointing", "save_best_temperature_mse", True),
    ("checkpointing", "save_best_predicted", True),
    ("checkpointing", "save_best_field_mse", False),
    ("checkpointing", "save_latest", False),
])
def test_development_rejects_inherited_io_or_partial_epoch_settings(fixed_manifest, section, key, value):
    config, _ = fixed_manifest
    config[section][key] = value
    with pytest.raises(ValueError):
        training.resolve_development_training(config)


@pytest.mark.parametrize("kwargs", [{"max_train_batches": 1}, {"max_val_batches": 1}])
def test_development_cli_batch_caps_are_rejected(fixed_manifest, kwargs):
    config, _ = fixed_manifest
    with pytest.raises(ValueError, match="batch caps"):
        training.resolve_development_training(config, **kwargs)


def test_development_response_pool_respects_effective_stop_and_size(fixed_manifest):
    config, _manifest = fixed_manifest
    config["training"]["campaign"]["response_stencils"] = []
    training.resolve_development_training(config, effective_epochs=100)
    with pytest.raises(ValueError, match="one to four"):
        training.resolve_development_training(config, effective_epochs=500)
    config["training"]["campaign"]["response_stencils"] = [f"{index}.npz" for index in range(5)]
    with pytest.raises(ValueError, match="one to four"):
        training.resolve_development_training(config, effective_epochs=100)


def test_development_response_callback_validates_actual_loaded_anchor_metadata(fixed_manifest, monkeypatch):
    from channelthermal.interaction_evidence.types import EvidenceSplit
    from channelthermal.training import campaign_response

    config, manifest = fixed_manifest
    config["dataset"]["development_subset"] = manifest
    dataset = SimpleNamespace(normalizer=SimpleNamespace(stats={}))
    loaded = [(SimpleNamespace(split=EvidenceSplit.TRAIN, physical_family_id="one"), {"anchor_id": "excluded"})]
    monkeypatch.setattr(campaign_response, "load_response_atlas_stencil", lambda path: loaded[int(path.stem)])
    with pytest.raises(ValueError, match="selected training anchors"):
        campaign_response.NativeCampaignResponse(None, dataset, config["dataset"], {"response_stencils": ["0.npz"]})
    loaded[0][1]["anchor_id"] = "train_a"
    monkeypatch.setattr(campaign_response, "_ResponseCall", lambda *args: "public wrapper fixture")

    class Dataset:
        normalizer = SimpleNamespace(stats={})

        def __getitem__(self, index):
            return {"case_id": "train_a"}

    callback = campaign_response.NativeCampaignResponse(None, Dataset(), config["dataset"], {"response_stencils": ["0.npz"]})
    assert callback.stencils == [loaded[0][0]]
    # Historical full campaigns retain their existing TRAIN-family scope.
    loaded[0][1]["anchor_id"] = "excluded"
    assert campaign_response.NativeCampaignResponse(None, Dataset(), {}, {"response_stencils": ["0.npz"]}).stencils


def test_development_resume_requires_exact_fixed_membership_and_dataset_identity(fixed_manifest):
    config, _ = fixed_manifest
    training.resolve_development_training(config)
    dataset = config["dataset"]
    checkpoint = {"train_config": copy.deepcopy(config)}
    training.validate_development_resume(checkpoint, dataset)
    for key in ("dataset_id", "dataset_schema", "dataset_fingerprint", "development_manifest_sha256"):
        changed = copy.deepcopy(checkpoint)
        changed["train_config"]["dataset"][key] = "wrong"
        with pytest.raises(ValueError, match="identity|digest"):
            training.validate_development_resume(changed, dataset)
    for changed_binding in (None, {}, copy.deepcopy(dataset["development_subset"])):
        changed = copy.deepcopy(checkpoint)
        if changed_binding:
            changed_binding["partitions"]["train"]["case_ids"][0] = "other_train"
        changed["train_config"]["dataset"]["development_subset"] = changed_binding
        with pytest.raises(ValueError, match="membership|identity"):
            training.validate_development_resume(changed, dataset)
    changed = copy.deepcopy(checkpoint)
    changed["train_config"]["dataset"]["development_subset"]["policy"]["seed"] += 1
    with pytest.raises(ValueError, match="membership"):
        training.validate_development_resume(changed, dataset)


def test_full_and_development_resumes_cannot_be_mixed(fixed_manifest):
    config, _ = fixed_manifest
    training.resolve_development_training(config)
    full = {"train_config": {"dataset": {"dataset_id": "fixture"}}}
    training.validate_development_resume(full, full["train_config"]["dataset"])
    with pytest.raises(ValueError, match="membership"):
        training.validate_development_resume(full, config["dataset"])
    with pytest.raises(ValueError, match="membership"):
        training.validate_development_resume({"train_config": config}, full["train_config"]["dataset"])


def test_development_normalization_inventory_cannot_be_missing_on_resume(fixed_manifest):
    config, _ = fixed_manifest
    training.resolve_development_training(config)
    stats = {"field_mean": np.array([2.]), "field_std": np.array([1.])}
    checkpoint = {"train_config": copy.deepcopy(config), "global_normalization_stats": copy.deepcopy(stats)}
    normalizer = SimpleNamespace(stats=stats)
    training.validate_development_resume(checkpoint, config["dataset"], normalizer=normalizer)
    for missing in (None, {}, {"field_mean": stats["field_mean"]}):
        checkpoint["global_normalization_stats"] = missing
        with pytest.raises(ValueError, match="normalization inventory"):
            training.validate_development_resume(checkpoint, config["dataset"], normalizer=normalizer)


def test_development_training_seed_remains_a_separate_strict_resume_policy(fixed_manifest):
    config, _ = fixed_manifest
    training.resolve_development_training(config)
    checkpoint = {"train_config": copy.deepcopy(config), "optimizer_state_dict": {"fixture": True},
                  "rng_state": {"fixture": True}}
    changed = copy.deepcopy(config)
    changed["training"]["seed"] = 1
    with pytest.raises(ValueError, match="training.seed"):
        validate_campaign_resume(checkpoint, changed)


def test_saved_normalizer_is_still_independently_validated_before_state_application():
    dataset = SimpleNamespace(channel_order=["T"], interface_condition_feature_names=["x"],
        interface_target_names=["T"], normalizer=SimpleNamespace(stats={"field_mean": np.array([2.])}))
    model_config = SimpleNamespace(to_dict=lambda: {"channelthermal": {"internal_prediction_mode": "local_surrogate"}})
    checkpoint = {"case_id": "ThermalChannel", "model_family": "honf_forward", "workflow": "forward",
                  "global_normalization_stats": {"field_mean": np.array([2.])}}
    _validate_resume_checkpoint(checkpoint, model=object(), model_config=model_config, dataset=dataset, dataset_config={})
    checkpoint["global_normalization_stats"]["field_mean"] = np.array([8.])
    with pytest.raises(ValueError, match="normalization mismatch"):
        _validate_resume_checkpoint(checkpoint, model=object(), model_config=model_config, dataset=dataset, dataset_config={})


def test_measure_cost_option_cannot_bypass_strict_model_resume():
    from channelthermal.config import ChannelThermalHONFConfig

    from honf_forward_core.config import InterfaceFieldConfig

    initial = ChannelThermalHONFConfig()
    initial.core_honf.forward_architecture = "faithful_receiver_hypergraph_honf"
    initial.core_honf.interface_model = InterfaceFieldConfig()
    initial.channelthermal.internal_prediction_mode = "global_head"
    payload = initial.to_dict()
    dataset = SimpleNamespace(channel_order=["T"], interface_condition_feature_names=["x"],
        interface_target_names=["T"], normalizer=SimpleNamespace(stats={}))
    checkpoint = {"case_id": "ThermalChannel", "model_family": "honf_forward", "workflow": "forward",
        "epoch": 100, "model_config": payload}
    current = ChannelThermalHONFConfig.from_dict(payload)
    current.core_honf.interface_model.hypergraph_options["structural_measure_policy_version"] = 1
    inputs = {"model": object(), "model_config": current, "dataset": dataset, "dataset_config": {}}
    _validate_resume_checkpoint(checkpoint, **inputs)
    current.core_honf.interface_model.hypergraph_options["structural_measure_policy_version"] = 2
    with pytest.raises(ValueError, match="configuration"):
        _validate_resume_checkpoint(checkpoint, **inputs)
    amendment = {"structural_measure_policy": {"from": 1, "to": 2, "activation_epoch": 101}}
    _validate_resume_checkpoint(checkpoint, campaign_amendment=amendment, **inputs)
    current.core_honf.hidden_dim += 1
    with pytest.raises(ValueError, match="configuration"):
        _validate_resume_checkpoint(checkpoint, campaign_amendment=amendment, **inputs)


def test_reviewed_measure_policy_cannot_enter_a_fresh_training_workflow(fixed_manifest):
    config, _ = fixed_manifest
    config["model"] = {"core_honf": {"forward_architecture": "faithful_receiver_hypergraph_honf",
        "interface_model": {"hypergraph_options": {"structural_measure_policy_version": 2}}}}
    config["training"]["campaign"].update(physical_loss_policy_version=2,
        native_loss_denominators_start_epoch=101, structural_measure_policy_version=2)
    args = SimpleNamespace(epochs=None, resume_checkpoint=None)
    with pytest.raises(ValueError, match="cannot start fresh"):
        training.run_from_config(config, args)


def test_selected150_cases_keep_four_native_boundaries_and_153600_primary_queries():
    ids = [f"selected_{i}" for i in range(150)]
    batches = [{"case_id": ids[start:start + 48], "field_targets": torch.zeros(min(48, 150 - start), 1024, 5)}
               for start in range(0, 150, 48)]

    class Loader:
        dataset = SimpleNamespace(selected_case_ids=ids)
        def __iter__(self):
            return iter(batches)
        def __len__(self):
            return len(batches)

    parts = list(CampaignMicrobatchLoader(Loader(), 8))
    assert [case for part in parts for case in part["case_id"]] == ids
    assert sum(part["_optimizer_boundary"] for part in parts) == 4
    assert len(parts) == 19
    assert sum(part["field_targets"].shape[0] * part["field_targets"].shape[1] for part in parts) == 153600


def test_best_aliases_and_latest_save_only_at_100_or_the_final_stop():
    checkpointing = {"save_best_every_epochs": 100, "save_latest_every_epochs": 100}
    assert [epoch for epoch in range(1, 251) if training.should_save_best_checkpoint(epoch, 250, checkpointing)] == [100, 200, 250]
    assert [epoch for epoch in range(1, 251) if training.should_save_latest_checkpoint(epoch, 250, checkpointing)] == [100, 200, 250]


@pytest.mark.parametrize("cadence", [0, -1, True, 1.5])
def test_core_schema_rejects_invalid_best_save_cadence(tmp_path, cadence):
    profile = Path(__file__).resolve().parents[1] / "src/config_core/forward/thermal_campaign/h-tree_e500.json"
    core = json.loads(profile.read_text())
    core["checkpointing"]["save_best_every_epochs"] = cadence
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(core))
    with pytest.raises(ValueError, match="save_best_every_epochs"):
        load_config_bundle(path)
