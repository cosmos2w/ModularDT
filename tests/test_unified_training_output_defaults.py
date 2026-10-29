from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.run_layout import RunLayout
from honf_runtime.unified_training import EngineConfig, TrainingEngine, _engine_config_payload
from tests.test_unified_training import _ToyProvider, _config


def _rng_equal(left, right):
    assert left["python"] == right["python"]
    assert left["numpy"][0] == right["numpy"][0]
    np.testing.assert_array_equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][2:] == right["numpy"][2:]
    assert torch.equal(left["torch_cpu"], right["torch_cpu"])


def test_output_defaults_are_compatible_and_validation_sidecars_are_sparse(tmp_path, monkeypatch):
    config = _config()
    assert config.curve_every == 10
    assert config.progress_enabled is True
    assert "curve_every" not in _engine_config_payload(config)
    assert "progress_enabled" not in _engine_config_payload(config)
    assert _engine_config_payload(replace(config, curve_every=20))["curve_every"] == 20
    assert _engine_config_payload(replace(config, progress_enabled=False))["progress_enabled"] is False
    with pytest.raises(ValueError, match="Progress display control"):
        replace(config, progress_enabled=1)

    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    provider = _ToyProvider()
    output = Path(tmp_path) / "sparse-validation"
    short = replace(config, total_epochs=5, monitor_every=1, checkpoint_epochs=(2, 4), training_mode="joint")
    TrainingEngine(short, device="cpu").fit(
        nn.Linear(1, 1), provider, output,
        identity={"run": "sparse-validation"}, arm="adaptive_detail", stop_after=5,
    )

    artifacts = RunLayout(output)
    history = json.loads(artifacts.read_path("history.json").read_text())
    assert [row["epoch"] for row in history] == [1, 2, 3, 4, 5]
    assert all("validation" in row for row in history)
    assert len(provider.validation_phases) == 10  # two validation batches at each of five reviews
    assert sorted(path.name for path in artifacts.read_path("validation_epoch_0001.json").parent.glob("validation_epoch_*.json")) == [
        "validation_epoch_0002.json", "validation_epoch_0004.json", "validation_epoch_0005.json",
    ]
    assert sorted(path.name for path in artifacts.read_path("epoch_0002_model.pt").parent.glob("epoch_*_model.pt")) == [
        "epoch_0002_model.pt", "epoch_0004_model.pt", "epoch_0005_model.pt",
    ]
    assert artifacts.read_path("latest_model.pt").is_file()
    assert artifacts.read_path("best_by_field_mse_model.pt").is_file()


@pytest.mark.parametrize("progress_enabled, expected_disabled", [(True, False), (False, True)])
def test_progress_bar_default_is_enabled_without_tty_and_can_be_disabled(
    tmp_path, monkeypatch, progress_enabled, expected_disabled,
):
    captured = []

    class _NonTty:
        @staticmethod
        def isatty():
            return False

    class _Bar:
        def set_postfix(self, values, *, refresh=True):
            del values, refresh

        def update(self, amount):
            del amount

        def close(self):
            pass

    monkeypatch.setattr(runtime.sys, "stderr", _NonTty())

    def fake_tqdm(**options):
        captured.append(options)
        return _Bar()

    monkeypatch.setattr(runtime, "tqdm", fake_tqdm)
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    config = replace(_config(), total_epochs=2, training_mode="joint", progress_enabled=progress_enabled)
    output = Path(tmp_path) / f"progress-{progress_enabled}"
    TrainingEngine(config, device="cpu").fit(
        nn.Linear(1, 1), _ToyProvider(), output,
        identity={"run": f"progress-{progress_enabled}"}, arm="adaptive_detail", stop_after=1,
    )

    assert len(captured) == 1
    assert captured[0]["disable"] is expected_disabled
    assert captured[0]["file"] is runtime.sys.stderr


def test_default_ten_epoch_curves_do_not_change_weights_or_rng(tmp_path, monkeypatch):
    records = []

    def noisy_render(history, *args, **kwargs):
        del args, kwargs
        records.append(tuple(int(row["epoch"]) for row in history))
        torch.rand(5)
        np.random.random(3)
        random.random()

    monkeypatch.setattr(runtime, "_render_loss_curves", noisy_render)
    initial = nn.Linear(1, 1).state_dict()
    config = replace(_config(), total_epochs=10, monitor_every=100)
    initial_rng = runtime.capture_rng_state()

    def run(name, curve_every):
        runtime.restore_rng_state(initial_rng)
        model = nn.Linear(1, 1)
        model.load_state_dict(initial)
        output = Path(tmp_path) / name
        result = TrainingEngine(replace(config, curve_every=curve_every), device="cpu").fit(
            model, _ToyProvider(), output,
            identity={"run": name}, arm="adaptive_detail", stop_after=10,
        )
        assert result["completed_epoch"] == 10
        return model.state_dict(), runtime.capture_rng_state()

    torch.manual_seed(44)
    np.random.seed(44)
    random.seed(44)
    initial_rng = runtime.capture_rng_state()
    with_curves, rng_with_curves = run("default-curves", 10)
    default_records = list(records)
    records.clear()
    without_interval_curves, rng_without_interval = run("no-interval-curves", None)

    assert default_records == [(1,), tuple(range(1, 11))]
    assert records == []
    assert all(torch.equal(with_curves[name], without_interval_curves[name])
               for name in with_curves)
    _rng_equal(rng_with_curves, rng_without_interval)
