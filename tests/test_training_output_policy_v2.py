from __future__ import annotations

import json
from dataclasses import replace

import torch
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.run_layout import RunLayout
from honf_runtime.unified_training import TrainingEngine, read_training_history
from tests.test_unified_training import _ToyProvider, _config
from tests.test_unified_loss_curves import _capture_figures


class _LegacyLogProvider(_ToyProvider):
    training_history_in_logs = True
    training_stream_hash_history = False

    def response_epoch_family_ids(self, epoch):
        return ("family-a", "family-b")


def test_new_policy_separates_data_and_curves_and_resumes_exactly(tmp_path, monkeypatch, capsys):
    renders = []

    def render(history, output, field_metric, **kwargs):
        del field_metric
        renders.append((len(history), output, kwargs.get("metadata_output")))
        output.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            (output / f"loss_curves.{suffix}").write_bytes(b"curve")

    monkeypatch.setattr(runtime, "_render_loss_curves", render)
    config = replace(_config(), total_epochs=10, training_mode="joint", monitor_every=5,
                     progress_enabled=False, write_initial_artifacts=True)
    initial = nn.Linear(1, 1).state_dict()
    output = tmp_path / "new"
    model = nn.Linear(1, 1)
    model.load_state_dict(initial)
    engine = TrainingEngine(config, device="cpu")
    engine.fit(model, _LegacyLogProvider(), output, identity={"run": "same"},
               arm="adaptive_detail", stop_after=5)
    marker = json.loads((output / "artifact_layout.json").read_text())
    assert marker["training_output_policy"] == "categorized_v2"
    assert marker["compact_history"] is True
    assert (output / "metrics/history.json").is_file()
    assert (output / "metrics/history_metadata.json").is_file()
    assert not (output / "logs/history.json").exists()
    assert (output / "plots/training/loss_curves.png").is_file()
    assert not (output / "logs/loss_curves.png").exists()
    history = read_training_history(output / "metrics/history.json")
    assert all("case_order_sha256" not in row and "query_sampling_sha256" not in row for row in history)
    assert all("train_losses" in row for row in history)
    assert all(row["response_fit_family_ids"] == ["family-a", "family-b"] for row in history)
    assert "validation" in history[-1]
    console = capsys.readouterr().out
    assert "case_order_sha256" not in console
    assert '"response_fit_family_count": 2' in console
    assert "response_fit_family_ids" not in console
    result = engine.fit(nn.Linear(1, 1), _LegacyLogProvider(), output, identity={"run": "same"},
                        arm="adaptive_detail", stop_after=10,
                        resume_checkpoint=output / "checkpoints/latest_model.pt")
    assert result["completed_epoch"] == 10 and "identity" in result
    assert [row[0] for row in renders] == [1, 5, 10]
    assert all(row[1] == output / "plots/training" and row[2] == output / "metrics" for row in renders)
    checkpoint = torch.load(output / "checkpoints/latest_model.pt", map_location="cpu", weights_only=False)
    assert "experiment_identity" in checkpoint and "rng_state" in checkpoint
    assert "last_case_order_sha256" in checkpoint["sampler_state"]
    assert all("case_order_sha256" in row and "query_sampling_sha256" in row for row in checkpoint["history"])
    assert "identity" not in json.loads((output / "logs/active_process.json").read_text())
    uninterrupted = nn.Linear(1, 1)
    uninterrupted.load_state_dict(initial)
    engine.fit(uninterrupted, _LegacyLogProvider(), tmp_path / "full", identity={"run": "same"},
               arm="adaptive_detail", stop_after=10)
    assert all(torch.equal(checkpoint["model_state_dict"][key], value)
               for key, value in uninterrupted.state_dict().items())


def test_existing_log_layout_resume_preserves_serialization_and_stream_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    config = replace(_config(), total_epochs=4, training_mode="joint", progress_enabled=False)
    output = tmp_path / "legacy"
    provider = _LegacyLogProvider()
    provider.training_stream_hash_history = True
    engine = TrainingEngine(config, device="cpu")
    engine.fit(nn.Linear(1, 1), provider, output, identity={"run": "legacy"},
               arm="adaptive_detail", stop_after=2, compact_history=False)
    # Recreate the established v1 producer layout before resuming its checkpoint.
    marker_path = output / "artifact_layout.json"
    marker = json.loads(marker_path.read_text())
    marker.pop("training_output_policy")
    marker.pop("compact_history")
    marker_path.write_text(json.dumps(marker))
    (output / "metrics/history.json").rename(output / "logs/history.json")
    before = json.loads((output / "logs/history.json").read_text())
    engine.fit(nn.Linear(1, 1), provider, output, identity={"run": "legacy"},
               arm="adaptive_detail", stop_after=4,
               resume_checkpoint=output / "checkpoints/latest_model.pt")
    history = json.loads((output / "logs/history.json").read_text())
    assert history[:2] == before
    assert len(history) == 4 and all("query_sampling_sha256" in row for row in history)
    assert not (output / "metrics/history.json").exists()
    assert RunLayout(output).read_path("history.json") == output / "logs/history.json"


def test_curve_metadata_uses_data_folder_and_old_reader_precedence(tmp_path, monkeypatch):
    output, data = tmp_path / "plots/training", tmp_path / "metrics"
    _capture_figures(monkeypatch, output)
    runtime._render_loss_curves([{"epoch": 1, "arm": "adaptive_detail", "case_visits": 5,
                                  "train_losses": {"native": 1.0}}], output, "field_score",
                                metadata_output=data)
    assert (output / "loss_curves.png").is_file()
    assert (data / "loss_curves_metadata.json").is_file()
    assert (data / "loss_curves_initial_metadata.json").is_file()
    assert not (output / "loss_curves_metadata.json").exists()

    # A marked pre-policy run keeps the original canonical-first reader order.
    (tmp_path / "artifact_layout.json").write_text('{"artifact_layout_version": 1}')
    (tmp_path / "logs").mkdir()
    (data / "history.json").write_text("canonical")
    (tmp_path / "logs/history.json").write_text("old-log")
    (output / "loss_curves_metadata.json").write_text("old-plot")
    layout = RunLayout(tmp_path)
    assert layout.read_path("history.json") == data / "history.json"
    assert layout.read_path("loss_curves_metadata.json") == output / "loss_curves_metadata.json"
