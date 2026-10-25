from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from honf_runtime import research_training
from honf_runtime.contextual_frontier_training import (
    _link_parent_field_selector,
    _engine_config,
    _run_identity,
    continue_contextual_frontier,
    load_contextual_frontier_config,
    train_contextual_frontier,
)
from honf_runtime.unified_training import LossTerm, OptimizerGroupSpec, SamplingKey, ScheduleSpec, TaskBatch


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT / "configs" / "research"


@pytest.mark.parametrize(("suffix", "arm"), (("jd", "J-D"), ("jh", "J-H")))
def test_development_configs_bind_the_paired_fixed25_recipe(suffix: str, arm: str) -> None:
    config, _ = load_contextual_frontier_config(
        CONFIG_DIR / f"thermal_contextual_frontier_{suffix}_dev_v1.json"
    )

    assert config["contextual_frontier"]["arm"] == arm
    assert config["training"]["membership"] == "fixed25_train150_dev22"
    assert config["training"]["effective_batch_size"] == 48
    assert config["training"]["microbatch_cases"] == 24
    assert config["training"]["primary_queries"] == 1024
    assert config["training"]["seed"] == 0
    assert config["training"]["first_segment_epochs"] == 100
    assert config["training"]["local_horizon"] == 2000
    assert config["initialization"]["pretrained_weights"] is False
    assert config["storage"]["max_output_bytes"] == 4 * 1024**3
    assert config["storage"]["max_peak_temporary_bytes"] == 1 * 1024**3
    assert config["storage"]["minimum_free_headroom_bytes"] == 200 * 1024**3
    assert config["evaluation"]["max_output_bytes"] == 256 * 1024**2
    assert config["evaluation"]["representative_case_limit"] == 4
    assert config["training"]["wall_clock_budget_seconds"] == 12 * 60 * 60
    assert config["training"]["clean_save_reserve_seconds"] == 600


def test_config_rejects_unknown_arm_without_name_or_exact_budget_gate(tmp_path: Path) -> None:
    config, _ = load_contextual_frontier_config(CONFIG_DIR / "thermal_contextual_frontier_jd_dev_v1.json")
    config["name"] = "local-experiment-name"
    config["storage"]["max_output_bytes"] = 3 * 1024**3
    config["contextual_frontier"]["arm"] = "unknown"
    invalid = tmp_path / "config.json"
    invalid.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="arm"):
        load_contextual_frontier_config(invalid)


def test_engine_and_run_identity_preserve_declared_recipe() -> None:
    config, _ = load_contextual_frontier_config(CONFIG_DIR / "thermal_contextual_frontier_jh_dev_v1.json")
    engine_config = _engine_config(config)
    identity = _run_identity(config, SimpleNamespace())

    assert engine_config.total_epochs == 2000
    assert engine_config.checkpoint_epochs == tuple(range(100, 2001, 100))
    assert engine_config.latest_every == 10
    assert engine_config.curve_every == 10
    assert engine_config.progress_enabled is True
    assert engine_config.sampling_version == SamplingKey.CASE_EPOCH_VERSION
    assert engine_config.training_mode == "joint"
    assert identity["recipe"]["native_weights"] == config["objective"]["native_weights"]
    assert identity["recipe"]["absolute_horizon"] == 2000
    assert identity["recipe"]["wall_clock_budget_seconds"] == 12 * 60 * 60


def test_external_continuation_keeps_parent_best_checkpoint_by_reference(tmp_path, monkeypatch) -> None:
    import honf_runtime.compat as compat
    import honf_runtime.contextual_frontier_training as runner

    source_run = tmp_path / "source"
    checkpoint_dir = source_run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    source_latest = checkpoint_dir / "epoch_0100_model.pt"
    source_best = checkpoint_dir / "best_by_field_mse_model.pt"
    source_latest.touch()
    source_best.touch()
    identity = {"workflow": "spatial_interaction_groups_joint_training_v1"}
    source_payload = {"epoch": 100, "arm": "F13-0", "experiment_identity": identity}
    best_payload = {**source_payload, "epoch": 76}
    monkeypatch.setattr(runner, "resolve_checkpoint", lambda run, selector: source_best)
    monkeypatch.setattr(
        compat, "load_trusted_checkpoint",
        lambda path, map_location: best_payload,
    )

    output = tmp_path / "continuation"
    output.mkdir()
    selected = _link_parent_field_selector(output, source_latest, source_payload)

    assert selected.is_symlink()
    assert selected.resolve() == source_best
    assert selected.stat().st_ino == source_best.stat().st_ino


def test_external_continuation_dispatch_uses_literal_source_and_writes_no_history_early(
    tmp_path, monkeypatch,
) -> None:
    import honf_runtime.compat as compat
    import honf_runtime.contextual_frontier_training as runner

    config, config_path = load_contextual_frontier_config(
        CONFIG_DIR / "thermal_spatial_groups_f13_0_dev_mature_v1.json"
    )
    output_root = tmp_path / "runs"
    config["storage"]["output_root"] = str(output_root)
    source_run = tmp_path / "source"
    checkpoint_dir = source_run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    source = checkpoint_dir / "epoch_0100_model.pt"
    best = checkpoint_dir / "best_by_field_mse_model.pt"
    source.touch()
    best.touch()
    config["continuation"]["source_checkpoint"] = str(source)
    config["continuation"]["source_epoch"] = 100
    identity = {
        "workflow": "spatial_interaction_groups_joint_training_v1",
        "training_membership": "fixed25_train150_dev22",
        "provider_identity": {
            "workflow": "spatial_interaction_groups_joint_training_v1",
        },
    }
    payload = {"epoch": 100, "arm": "F13-0", "experiment_identity": identity,
               "history": [{"epoch": epoch} for epoch in range(1, 101)]}
    best_payload = {**payload, "epoch": 76}
    monkeypatch.setattr(runner, "load_contextual_frontier_config", lambda _path: (config, config_path))
    monkeypatch.setattr(runner, "_ensure_workspace_storage_bindings", lambda: {})
    monkeypatch.setattr(runner, "resolve_checkpoint", lambda _run, _selector: best)
    monkeypatch.setattr(
        compat, "load_trusted_checkpoint",
        lambda path, map_location: best_payload if Path(path) == best else payload,
    )
    monkeypatch.setattr(runner.research, "verify_storage", lambda *_args, **_kwargs: {
        "free_bytes_before": 10**13,
        "minimum_free_headroom_bytes": 1,
        "estimated_new_bytes": 0,
        "estimated_peak_temporary_bytes": 0,
    })
    monkeypatch.setattr(runner, "_render_loss_curves", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(
        "honf_runtime.unified_training._render_loss_curves",
        lambda *_args, **_kwargs: None,
    )
    calls = {}

    def fake_fit(model, provider, actual_config, actual_output, **kwargs):
        calls.update(kwargs)
        assert not (actual_output / "history.json").exists()
        assert not (actual_output / "metrics" / "history.json").exists()
        assert actual_config["training"]["membership"] == "fixed25_train150_dev22"
        return {"status": "dispatched"}

    monkeypatch.setattr(runner, "_fit", fake_fit)

    def task_factory(cfg, arm, device):
        del cfg
        model = _TinyModel(arm).to(device)
        return model, _TinyProvider(model)

    result = continue_contextual_frontier(
        CONFIG_DIR / "thermal_spatial_groups_f13_0_dev_mature_v1.json",
        device="cpu", devices=["cpu"], output=output_root / "continuation",
        task_factory=task_factory,
        verify_bindings=False,
    )

    assert result == {"status": "dispatched"}
    assert calls["resume_checkpoint"] == source
    assert calls["external_state_continuation"] is True


class _TinyModel(torch.nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        self.architecture = arm
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class _TinyProvider:
    def __init__(self, model: _TinyModel):
        self.model = model
        self.training_case_ids = ("train-a", "train-b")
        self.validation_case_ids = ("dev-a", "dev-b")
        self.wall_clock_budget = None
        self._features = {
            "train-a": (0.0, 0.25),
            "train-b": (1.0, 0.75),
            "dev-a": (0.5, 0.4),
            "dev-b": (1.5, 0.9),
        }

    def identity_payload(self):
        return {
            "schema": "tiny_contextual_runner_test_v1",
            "dataset": "thermal",
            "architecture": self.model.architecture,
            "training_case_ids": list(self.training_case_ids),
            "validation_case_ids": list(self.validation_case_ids),
        }

    def epoch_cases(self, epoch: int, seed: int):
        del epoch, seed
        return list(self.training_case_ids)

    def _batch(self, case_ids):
        pairs = [self._features[case_id] for case_id in case_ids]
        return TaskBatch(
            scene_inputs=torch.tensor([[row[0]] for row in pairs], dtype=torch.float32),
            receivers=None,
            targets=torch.tensor([[row[1]] for row in pairs], dtype=torch.float32),
            case_keys=tuple(case_ids),
        )

    def make_batch(self, case_keys, key):
        assert key.sampling_version == SamplingKey.CASE_EPOCH_VERSION
        return self._batch(case_keys)

    def loss_denominators(self, batches, phase, arm):
        del phase, arm
        return {"native": sum(len(batch.case_keys) for batch in batches)}

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, temperature=None):
        del receivers, execution_mode, phase, epoch, temperature
        return model(scene), {}

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase, auxiliary_state
        return {"native": LossTerm((predictions - targets).square().sum(), len(targets), 1.0)}

    def validation_batches(self):
        yield self._batch(self.validation_case_ids)

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del predictions, targets, auxiliary_state
        return {"validation_record": 1.0}

    def reduce_native_metrics(self, records):
        return {"field_score": float(len(records))}

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch, arm):
        del batch, arm
        return self.loss_terms(predictions, targets, "joint", auxiliary_state)

    def optimizer_groups(self, model, arm, stage):
        assert model is self.model and arm == model.architecture and stage == "joint"
        return (OptimizerGroupSpec(
            "joint", tuple(name for name, _ in model.named_parameters()),
            ScheduleSpec(peak_lr=3e-4, warmup_start_lr=3e-4, warmup_epochs=0,
                        hold_through_epoch=0, total_epochs=2, final_lr=3e-4),
            weight_decay=1e-5,
        ),)

    def work_counts(self, batch, predictions, auxiliary_state):
        del batch, predictions, auxiliary_state
        return {}

    def training_state_dict(self):
        return {}

    def load_training_state_dict(self, state):
        assert state == {}

    def bind_wall_clock_budget(self, *, output_dir, budget_seconds, saver_reserve_seconds=600):
        self.wall_clock_budget = (Path(output_dir), budget_seconds, saver_reserve_seconds)


def test_runner_uses_maintained_engine_and_saves_standard_runlayout(tmp_path: Path, monkeypatch) -> None:
    config, config_path = load_contextual_frontier_config(
        CONFIG_DIR / "thermal_contextual_frontier_jd_dev_v1.json"
    )
    config["name"] = "cpu-runner-integration-test"
    config["storage"]["output_root"] = str(tmp_path / "runs")
    config["storage"]["minimum_free_headroom_bytes"] = 1
    config["training"].update({
        "local_horizon": 1,
        "first_segment_epochs": 1,
        "effective_batch_size": 2,
        "microbatch_cases": 2,
        "monitor_every": 1,
        "checkpoint_epochs": [1],
        "latest_every": 1,
        "curve_every": 1,
        "progress_enabled": False,
    })
    monkeypatch.setattr(
        "honf_runtime.contextual_frontier_training.load_contextual_frontier_config",
        lambda path: (config, config_path),
    )

    def storage_snapshot(config, *, output=None, estimated_bytes=0, peak_temporary_bytes=0):
        del output
        return {
            "free_bytes_before": 10**13,
            "minimum_free_headroom_bytes": config["storage"]["minimum_free_headroom_bytes"],
            "estimated_new_bytes": estimated_bytes,
            "estimated_peak_temporary_bytes": peak_temporary_bytes,
        }

    monkeypatch.setattr(research_training, "verify_storage", storage_snapshot)

    providers = []

    def factory(task_config, *, arm, device):
        assert arm == "J-D" and device.type == "cpu"
        model = _TinyModel(arm).to(device)
        provider = _TinyProvider(model)
        providers.append(provider)
        return model, provider

    output = tmp_path / "runs" / "one-epoch"
    result = train_contextual_frontier(
        config_path,
        device="cpu",
        devices=["cpu"],
        output=output,
        task_factory=factory,
        verify_bindings=False,
    )

    assert result["completed_epoch"] == 1
    assert (output / "checkpoints" / "latest_model.pt").is_file()
    assert (output / "checkpoints" / "best_by_field_mse_model.pt").is_file()
    assert (output / "checkpoints" / "epoch_0001_model.pt").is_file()
    assert (output / "metrics" / "history.json").is_file()
    assert (output / "evaluations" / "validation" / "validation_epoch_0001.json").is_file()
    assert providers[0].wall_clock_budget == (output.resolve(), 43200, 600)


@pytest.mark.parametrize("command", ("train-contextual", "resume-contextual", "evaluate-contextual"))
def test_cli_exposes_contextual_frontier_commands(command: str) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT / "tools" / "honf_research.py"), command, "--help"],
        cwd=PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "contextual" in result.stdout.lower()
