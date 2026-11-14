import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.run_layout import RunLayout
from honf_runtime.unified_training import LossTerm, TrainingEngine, read_training_history
from tests.test_unified_training import _ToyProvider, _config


class CsvToyProvider(_ToyProvider):
    training_output_policy = "csv_v3"
    training_stream_hash_history = False

    def __init__(self):
        super().__init__()
        self.completed = 0

    def completed_training_epoch(self, model, engine, epoch, output, row):
        self.completed += 1

    def training_state_dict(self):
        return {"completed": self.completed}

    def load_training_state_dict(self, state):
        self.completed = state["completed"]

    def training_history_columns(self):
        return ("train_losses.native", "work_counts.query_rows", "work_counts.gradient_norm")

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch, arm):
        return {"native": LossTerm((predictions - targets).square().sum(), targets.numel())}


def _config_for_test():
    return replace(_config(), total_epochs=4, monitor_every=1, checkpoint_epochs=(2, 4),
                   training_mode="joint", progress_enabled=False)


def _fit(model, path, stop, resume=None):
    return TrainingEngine(_config_for_test(), device="cpu").fit(
        model, CsvToyProvider(), path, identity={"run": "csv-toy"}, arm="adaptive_detail",
        stop_after=stop, resume_checkpoint=resume)


def test_csv_exact_resume_and_curves(tmp_path, monkeypatch):
    curves = []
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda history, *args, **kwargs:
                        curves.append((history[-1]["epoch"], kwargs.get("separate_totals"))))
    torch.manual_seed(123)
    model = nn.Linear(1, 1)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    _fit(model, tmp_path / "complete", 4)
    complete = torch.load(RunLayout(tmp_path / "complete").read_path("latest_model.pt"), weights_only=False)
    resumed = nn.Linear(1, 1); resumed.load_state_dict(initial)
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    _fit(resumed, tmp_path / "resume", 2)
    layout = RunLayout(tmp_path / "resume")
    _fit(resumed, tmp_path / "resume", 4, layout.read_path("latest_model.pt"))
    recovered = torch.load(layout.read_path("latest_model.pt"), weights_only=False)
    assert "history" not in recovered and recovered["history_path"] == "metrics/history.csv"
    assert all(torch.equal(complete["model_state_dict"][name], value)
               for name, value in recovered["model_state_dict"].items())
    for name, state in complete["optimizer_state_by_name"]["state_by_name"].items():
        for key, value in state.items():
            assert torch.equal(value, recovered["optimizer_state_by_name"]["state_by_name"][name][key])
    assert torch.equal(complete["rng_state"]["torch_cpu"], recovered["rng_state"]["torch_cpu"])
    assert complete["rng_state"]["python"] == recovered["rng_state"]["python"]
    np.testing.assert_array_equal(complete["rng_state"]["numpy"][1], recovered["rng_state"]["numpy"][1])
    assert complete["rng_state"]["numpy"][2:] == recovered["rng_state"]["numpy"][2:]
    assert complete["sampler_state"] == recovered["sampler_state"]
    assert complete["provider_training_state"] == recovered["provider_training_state"] == {"completed": 4}
    rows = read_training_history(layout.read_path("history.csv"))
    expected = read_training_history(RunLayout(tmp_path / "complete").read_path("history.csv"))
    assert [r["train_losses"] for r in rows] == [r["train_losses"] for r in expected]
    assert [r["validation_objective"]["terms"] for r in rows] == [r["validation_objective"]["terms"] for r in expected]
    assert curves == [(1, True), (2, True), (3, True), (4, True),
                      (1, True), (2, True), (3, True), (4, True)]
    assert not (tmp_path / "resume" / "comparisons").exists()
    assert layout.read_path("epoch_0004_model.pt").stat().st_ino == layout.read_path("latest_model.pt").stat().st_ino


def test_selector_interruption_repairs_from_committed_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    original = runtime._publish_csv_checkpoint
    def interrupted(layout, payload):
        if payload["epoch"] == 2:
            raise RuntimeError("simulated interruption after latest commit")
        original(layout, payload)
    monkeypatch.setattr(runtime, "_publish_csv_checkpoint", interrupted)
    path = tmp_path / "run"
    model = nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="simulated"):
        _fit(model, path, 2)
    layout = RunLayout(path)
    latest = torch.load(layout.read_path("latest_model.pt"), weights_only=False)
    previous = torch.load(layout.read_path("best_by_field_mse_model.pt"), weights_only=False)
    assert latest["epoch"] == 2 and previous["epoch"] == 1
    assert not layout.read_path("epoch_0002_model.pt").exists()
    monkeypatch.setattr(runtime, "_publish_csv_checkpoint", original)
    _fit(model, path, 3, layout.read_path("latest_model.pt"))
    assert torch.load(layout.read_path("epoch_0002_model.pt"), weights_only=False)["epoch"] == 2
    assert torch.load(layout.read_path("best_by_field_mse_model.pt"), weights_only=False)["epoch"] >= 2


def test_csv_curves_include_separate_pending_validation(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    captured = []
    save = Figure.savefig
    def capture(figure, *args, **kwargs):
        captured.append([(axis.get_title(), tuple(axis.get_position().bounds)) for axis in figure.axes])
        return save(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", capture)
    runtime._render_loss_curves([{"epoch": 1, "phase": "joint", "arm": "toy",
        "case_visits": 2, "train_losses": {"native": 1.0}}], tmp_path / "plots" / "training",
        "field_score", metadata_output=tmp_path / "metrics", separate_totals=True)
    titles = [title for title, _ in captured[0]]
    assert "Total TRAIN loss" in titles and "Total measurable validation loss" in titles
    sizes = [(round(bounds[2], 6), round(bounds[3], 6)) for _, bounds in captured[0]]
    assert len(set(sizes)) == 1
    assert (tmp_path / "plots" / "training" / "loss_curves_initial.png").is_file()
    assert (tmp_path / "plots" / "training" / "loss_curves_initial.pdf").is_file()


def test_optional_validation_paths_preserve_provider_rng_and_model_mode():
    class ValidationProvider(CsvToyProvider):
        state = 0
        def training_state_dict(self):
            return {"state": self.state}
        def load_training_state_dict(self, state):
            self.state = state["state"]
        def validation_objective_predictions(self, model, batch, *, arm, epoch):
            self.state += 1
            model.train()
            torch.rand(1)
            return model(batch.scene_inputs) + 1, None
        def validation_auxiliary_terms(self, model, *, arm, epoch):
            self.state += 1
            model.train()
            torch.rand(1)
            return {"pair": LossTerm(torch.tensor(6.0), 2, 0.5)}
    provider = ValidationProvider()
    model = nn.Linear(1, 1)
    model.train()
    rng = runtime.capture_rng_state()
    result = TrainingEngine(_config_for_test(), device="cpu")._evaluate(model, provider, "adaptive_detail", "joint", 1)
    assert result["validation_loss_terms"]["pair"] == 1.5
    assert provider.state == 0 and model.training
    assert torch.equal(rng["torch_cpu"], runtime.capture_rng_state()["torch_cpu"])


def test_resume_reconciles_uncommitted_and_torn_history(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    path = tmp_path / "run"
    model = nn.Linear(1, 1)
    _fit(model, path, 2)
    layout = RunLayout(path)
    history_path = layout.read_path("history.csv")
    original = history_path.read_bytes()
    with history_path.open("ab") as stream:
        stream.write(original.splitlines(keepends=True)[-1].replace(b"2,", b"3,", 1))
        stream.write(b"4,joi")
    _fit(model, path, 3, layout.read_path("latest_model.pt"))
    assert [row["epoch"] for row in read_training_history(history_path)] == [1, 2, 3]
    assert len(list(history_path.parent.glob("history.recovery_*.csv"))) == 1
    assert '"event": "scalar_recovery"' in layout.read_path("events.jsonl").read_text()


def test_csv_curve_cadence_includes_first_tenth_validation_and_stop(tmp_path, monkeypatch):
    curves = []
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda history, *args, **kwargs:
                        curves.append(history[-1]["epoch"]))
    config = replace(_config(), total_epochs=10, monitor_every=100, monitor_epochs=(5,),
                     training_mode="joint", progress_enabled=False)
    TrainingEngine(config, device="cpu").fit(nn.Linear(1, 1), CsvToyProvider(), tmp_path / "run",
        identity={"run": "cadence"}, arm="adaptive_detail", stop_after=10)
    assert curves == [1, 5, 10]


def test_missing_committed_validation_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    model = nn.Linear(1, 1)
    path = tmp_path / "run"
    _fit(model, path, 2)
    layout = RunLayout(path)
    validation = layout.read_path("validation.csv")
    lines = validation.read_bytes().splitlines(keepends=True)
    validation.write_bytes(b"".join(lines[:-1]))
    with pytest.raises(ValueError, match="missing committed measurements"):
        _fit(model, path, 3, layout.read_path("latest_model.pt"))


def test_optional_case_reordering_preserves_membership():
    class Reordered(CsvToyProvider):
        def reorder_epoch_cases(self, cases, *, epoch, seed):
            return tuple(reversed(cases))
    class Invalid(Reordered):
        def reorder_epoch_cases(self, cases, *, epoch, seed):
            return cases[:-1]
    engine = TrainingEngine(_config_for_test(), device="cpu")
    original, _ = engine.epoch_case_order(CsvToyProvider(), 1)
    reordered, _ = engine.epoch_case_order(Reordered(), 1)
    assert reordered == list(reversed(original))
    with pytest.raises(ValueError, match="exact epoch membership"):
        engine.epoch_case_order(Invalid(), 1)
