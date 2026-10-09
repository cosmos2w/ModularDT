"""Formal joint recipe contracts and epoch-one artifact behavior."""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import honf_runtime.compat as runtime_compat
import honf_runtime.unified_training as runtime
from honf_runtime.run_layout import RunLayout, resolve_checkpoint
from honf_runtime.unified_training import (
    EngineConfig,
    TrainingEngine,
    _engine_config_payload,
    _sampling_dataset_id,
)
from tests.test_unified_training import _config, _ToyProvider

ROOT = Path(__file__).parents[1]
FORMAL_RECIPES = ROOT / "src/config_core/forward/joint_regional"
FORMAL_CHECKPOINT_EPOCHS = [100, 500, 1000, 2000, 2500, 5000]


def _cli():
    path = ROOT / "tools/joint_regional_train.py"
    spec = importlib.util.spec_from_file_location("joint_formal_protocol_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("task", "validation_scope", "seed", "queries", "batch"),
    [
        ("thermal", "canonical89", 0, 1024, 48),
        ("wind", "fullVALID90", 42, 4096, 24),
    ],
)
def test_geometry_formal_cli_recipes_seal_baseline_protocol(
    task, validation_scope, seed, queries, batch,
):
    cli = _cli()
    recipe = cli.read_recipe(FORMAL_RECIPES / f"{task}_geometry_full5000_v1.json")

    assert recipe["task"] == task
    assert recipe["mode"] == "J-geometry"
    assert recipe["formal_full"] is True
    assert recipe["launch_policy"] == "manual_only"
    assert recipe["seed"] == seed
    assert recipe["primary_queries"] == queries
    assert recipe["microbatch_cases"] == recipe["effective_cases"] == batch
    assert recipe["validation_scope"] == validation_scope
    assert recipe["native_sampling_protocol"] == "baseline_formal_v1"
    assert recipe["checkpoint_epochs"] == FORMAL_CHECKPOINT_EPOCHS
    assert recipe["write_initial_artifacts"] is True
    assert recipe["total_epochs"] == 5000
    assert recipe["initialization"] == "fresh_all_trainable"
    assert recipe["dataset_protocol"] == (
        "original600_train" if task == "thermal" else "original420_train"
    )

    # The explicit formal boundary remains mandatory at the command line.
    with pytest.raises(SystemExit):
        cli.main([
            "start", "--recipe-json", str(FORMAL_RECIPES / f"{task}_geometry_full5000_v1.json"),
            "--output-dir", "/unused-formal-run", "--stop-after", "5000",
        ])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.update(formal_full=False, total_epochs=2500, dataset_protocol="fixed25_v1"),
         "Formal comparison controls"),
        (lambda r: r.update(validation_scope="DEV22"), "validation scope"),
        (lambda r: r.update(native_sampling_protocol="legacy"), "native sampling protocol"),
        (lambda r: r.update(checkpoint_epochs=[5000, 100]), "checkpoint epochs"),
        (lambda r: r.update(write_initial_artifacts=1), "Initial artifact control"),
        (lambda r: r.update(optimizer_schedule={"peak_lr": 1e-4}), "optimizer schedule"),
    ],
)
def test_formal_cli_rejects_crossed_or_malformed_protocol_controls(mutate, message):
    cli = _cli()
    recipe = cli.read_recipe(FORMAL_RECIPES / "thermal_geometry_full5000_v1.json")
    crossed = copy.deepcopy(recipe)
    mutate(crossed)
    with pytest.raises(ValueError, match=message):
        cli.validate_recipe(crossed)


def test_baseline_formal_sampling_identity_reproduces_full_train_key_and_rejects_bad_binding():
    train_ids = [f"case-{index:03d}" for index in range(6)]
    fingerprint = hashlib.sha256(
        json.dumps(train_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    legacy_identity = {
        "dataset": "ThermalChannel",
        "manifest_fingerprint": fingerprint,
    }
    formal_identity = {
        "dataset": "ThermalChannel",
        # The fixed development manifest remains provenance, while this
        # explicit field reproduces the old full-TRAIN sampling identity.
        "manifest_fingerprint": "f" * 64,
        "native_sampling_identity": {
            "protocol": "baseline_formal_v1",
            "dataset": "ThermalChannel",
            "train_membership_fingerprint": fingerprint,
        },
    }
    assert _sampling_dataset_id(formal_identity) == _sampling_dataset_id(legacy_identity)

    wrong_dataset = copy.deepcopy(formal_identity)
    wrong_dataset["native_sampling_identity"]["dataset"] = "WindFarm"
    with pytest.raises(ValueError, match="matching dataset and TRAIN fingerprint"):
        _sampling_dataset_id(wrong_dataset)
    malformed_fingerprint = copy.deepcopy(formal_identity)
    malformed_fingerprint["native_sampling_identity"]["train_membership_fingerprint"] = "F" * 64
    with pytest.raises(ValueError, match="matching dataset and TRAIN fingerprint"):
        _sampling_dataset_id(malformed_fingerprint)


def test_initial_artifact_engine_key_stays_out_of_legacy_identity_until_enabled():
    legacy = _engine_config_payload(_config())
    explicit_default = _engine_config_payload(replace(_config(), write_initial_artifacts=False))
    enabled = _engine_config_payload(replace(_config(), write_initial_artifacts=True))

    assert explicit_default == legacy
    assert "write_initial_artifacts" not in legacy
    assert enabled["write_initial_artifacts"] is True


@pytest.mark.parametrize(
    "write_initial_artifacts, expected_curve_histories, expected_latest_epochs",
    [
        (False, [(1, 2)], [2]),
        (True, [(1,), (1, 2)], [1, 2]),
    ],
)
def test_joint_initial_artifact_control_writes_epoch_one_curves_and_latest_only_when_enabled(
    tmp_path, monkeypatch, write_initial_artifacts, expected_curve_histories,
    expected_latest_epochs,
):
    curve_histories = []
    latest_epochs = []
    render = runtime._render_loss_curves
    atomic_save = runtime._atomic_torch_save

    def record_curve(history, *args, **kwargs):
        curve_histories.append(tuple(int(row["epoch"]) for row in history))
        return render(history, *args, **kwargs)

    def record_save(path, payload):
        atomic_save(path, payload)
        if path.name == "latest_model.pt":
            latest_epochs.append(int(payload["epoch"]))

    monkeypatch.setattr(runtime, "_render_loss_curves", record_curve)
    monkeypatch.setattr(runtime, "_atomic_torch_save", record_save)

    config = EngineConfig(
        seed=17,
        microbatch_cases=2,
        effective_cases=4,
        total_epochs=3,
        warmup_epochs=1,
        open_through_epoch=1,
        soft_through_epoch=2,
        monitor_every=100,
        gradient_clip=10.0,
        checkpoint_epochs=(3,),
        latest_every=100,
        curve_every=100,
        sampling_version="case_epoch_v1",
        training_mode="joint",
        write_initial_artifacts=write_initial_artifacts,
    )
    output = tmp_path / f"initial-{write_initial_artifacts}"
    TrainingEngine(config, device="cpu").fit(
        nn.Linear(1, 1), _ToyProvider(), output,
        identity={"workflow": "joint_initial_artifact_test"}, arm="J-geometry", stop_after=2,
    )

    assert curve_histories == expected_curve_histories
    assert latest_epochs == expected_latest_epochs
    artifacts = RunLayout(output)
    assert not artifacts.read_path("epoch_0001_model.pt").exists()
    assert not artifacts.read_path("validation_epoch_0001.json").exists()
    assert resolve_checkpoint(output, "latest").is_file()
    assert torch.load(resolve_checkpoint(output, "latest"), map_location="cpu", weights_only=False)["epoch"] == 2
    assert artifacts.read_path("loss_curves.png").stat().st_size > 0
    assert artifacts.read_path("loss_curves.pdf").stat().st_size > 0


def test_joint_checkpoint_epochs_keep_ordinary_reviews_and_curves_without_snapshots(
    tmp_path, monkeypatch,
):
    curve_histories = []
    latest_epochs = []
    render = runtime._render_loss_curves
    atomic_save = runtime._atomic_torch_save

    def record_curve(history, *args, **kwargs):
        curve_histories.append(tuple(int(row["epoch"]) for row in history))
        return render(history, *args, **kwargs)

    def record_save(path, payload):
        atomic_save(path, payload)
        if path.name == "latest_model.pt":
            latest_epochs.append(int(payload["epoch"]))

    monkeypatch.setattr(runtime, "_render_loss_curves", record_curve)
    monkeypatch.setattr(runtime, "_atomic_torch_save", record_save)

    config = EngineConfig(
        seed=17,
        microbatch_cases=2,
        effective_cases=4,
        total_epochs=5,
        warmup_epochs=1,
        open_through_epoch=1,
        soft_through_epoch=2,
        monitor_every=2,
        gradient_clip=10.0,
        checkpoint_epochs=(4,),
        latest_every=2,
        curve_every=2,
        sampling_version="case_epoch_v1",
        training_mode="joint",
    )
    output = tmp_path / "sparse-key-checkpoints"
    TrainingEngine(config, device="cpu").fit(
        nn.Linear(1, 1), _ToyProvider(), output,
        identity={"workflow": "joint_checkpoint_epoch_test"}, arm="J-geometry", stop_after=4,
    )

    artifacts = RunLayout(output)
    checkpoint_dir = artifacts.read_path("epoch_0004_model.pt").parent
    assert [path.name for path in checkpoint_dir.glob("epoch_*_model.pt")] == ["epoch_0004_model.pt"]
    assert not artifacts.read_path("epoch_0002_model.pt").exists()
    assert resolve_checkpoint(output, "latest").is_file()
    assert resolve_checkpoint(output, "best_field").is_file()
    history = json.loads(artifacts.read_path("history.json").read_text())
    assert [row["epoch"] for row in history if "validation" in row] == [2, 4]
    assert latest_epochs == [2, 4]
    assert curve_histories == [(1, 2), (1, 2, 3, 4)]


def test_thermal_canonical89_factory_excludes_duplicate_but_keeps_original90_indices(
    tmp_path, monkeypatch,
):
    # Load the production module through the same import path setup as its CLI.
    _cli()
    thermal = importlib.import_module("channelthermal.training.joint_task")

    data_path = tmp_path / "synthetic-thermal-catalog.bin"
    data_path.write_bytes(b"synthetic catalog binding")
    manifest = {"manifest_sha256": "a" * 64, "source": {"metadata_sha256": "b" * 64}}
    train_ids = tuple(f"train-{index:03d}" for index in range(600))
    original90 = [f"valid-{index:03d}" for index in range(90)]
    duplicate_index = 37
    original90[duplicate_index] = "0273"
    catalog = ([{"case_id": case_id, "split": "train"} for case_id in train_ids]
               + [{"case_id": case_id, "split": "test"} for case_id in original90])
    loaded_ids = {}

    monkeypatch.setattr(thermal, "_load_data_binding", lambda _manifest, _data_path: (manifest, data_path, None))
    monkeypatch.setattr(runtime_compat, "set_seed", lambda _seed: None)
    monkeypatch.setattr(thermal, "read_case_catalog", lambda _path: (None, catalog))
    monkeypatch.setattr(thermal, "development_case_ids", lambda _manifest, split: ("stale-dev-id",))
    monkeypatch.setattr(thermal, "fit_global_normalizer", lambda _path, ids: SimpleNamespace(stats={"x": [1.0]}))

    def read_selected(_path, case_ids, *, split, normalizer, include_grid):
        del normalizer, include_grid
        loaded_ids[split] = tuple(case_ids)
        return tuple({"case_id": case_id} for case_id in case_ids)

    monkeypatch.setattr(thermal, "_read_selected_cases", read_selected)
    monkeypatch.setattr(thermal, "_load_recipe_helpers", lambda: (lambda _cases: "balances", lambda _families: {}))
    monkeypatch.setattr(thermal, "_read_response_families", lambda _path, _ids: ("family",))

    class Model:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to(self, _device):
            return self

    class Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(thermal, "JointThermalRegionalAdapter", Model)
    monkeypatch.setattr(thermal, "JointThermalTask", Provider)
    _model, provider = thermal.build_thermal_joint_task(
        mode="J-geometry",
        device="cpu",
        data_path=data_path,
        formal_full=True,
        total_epochs=5000,
        validation_scope="canonical89",
        native_sampling_protocol="baseline_formal_v1",
    )

    expected_validation = tuple(case_id for case_id in original90 if case_id != "0273")
    assert len(loaded_ids["train"]) == 600
    assert loaded_ids["train"] == train_ids
    assert loaded_ids["test"] == expected_validation
    assert len(loaded_ids["test"]) == 89
    assert "0273" not in loaded_ids["test"]
    assert provider.kwargs["validation_scope"] == "canonical89"
    assert provider.validation_sampling_indices == {
        case_id: index for index, case_id in enumerate(original90) if case_id != "0273"
    }
    assert provider.validation_sampling_indices[original90[duplicate_index - 1]] == duplicate_index - 1
    assert provider.validation_sampling_indices[original90[duplicate_index + 1]] == duplicate_index + 1
    assert max(provider.validation_sampling_indices.values()) == 89
