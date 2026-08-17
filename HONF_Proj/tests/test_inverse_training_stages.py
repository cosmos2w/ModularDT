"""Explicit four-stage ownership for `R,c -> G -> D -> G_hat`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from honf_inverse_core.training.stages import TRAINING_STAGES, configure_stage
from honf_inverse_core.training.checkpointing import load_inverse_checkpoint
from honf_inverse_core.training.trainer import InverseTrainer


def _designer() -> HierarchicalInverseDesigner:
    return HierarchicalInverseDesigner.from_config(
        {
            "num_edges": 3, "max_modules": 5, "request_hidden_dim": 16,
            "plan_hidden_dim": 24, "plan_layers": 1, "layout_hidden_dim": 24,
            "layout_layers": 1, "corrector_enabled": True, "corrector_hidden_dim": 24,
            "corrector_blocks": 1, "dropout": 0.0,
        }
    )


def _provenance() -> dict:
    return {
        "forward_checkpoint_id": "best_predicted_model.pt",
        "inverse_dataset_version": 1,
        "inverse_dataset_hash": "dataset-sha",
        "request_schema_version": 1,
        "compact_plan_schema_version": 1,
        "normalization_stats": {"mean": [0.0]},
    }


def test_training_stages_freeze_only_intended_modules() -> None:
    designer = _designer()
    for stage in TRAINING_STAGES:
        policy = configure_stage(designer, stage)
        assert any(policy.values())
        assert any(parameter.requires_grad for parameter in designer.parameters())
    policy = configure_stage(designer, "stage_plan")
    assert policy["plan_flow"] and not policy["layout_flow"] and not policy["corrector"]
    teacher = configure_stage(designer, "stage_layout_teacher_plan")
    assert teacher["layout_flow"] and not teacher["request_encoder"] and not teacher["plan_flow"]
    mixed = configure_stage(designer, "stage_layout_mixed_plan")
    assert mixed["request_encoder"] and mixed["plan_flow"] and mixed["layout_flow"]


def test_trainer_publishes_live_metrics_plot_status_and_checkpoint_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = InverseTrainer(
        _designer(), device="cpu", run_dir=tmp_path, checkpoint_provenance=_provenance()
    )
    epoch_results = iter(
        [
            {"total": 2.0, "flow": 1.5},
            {"total": 1.8, "flow": 1.4},
            {"total": 1.2, "flow": 0.9},
            {"total": 1.0, "flow": 0.8},
        ]
    )

    def fake_epoch(*args, **kwargs):
        if kwargs["optimizer"] is not None:
            trainer.global_step += 3
        return next(epoch_results)

    monkeypatch.setattr(trainer, "_run_epoch", fake_epoch)
    result = trainer.train_stage(
        "stage_plan", [object()], [object()], epochs=2, learning_rate=1.0e-4
    )
    trainer.mark_training_complete([result.__dict__])

    assert result.best_epoch == 1
    assert (tmp_path / "latest_model.pt").is_file()
    assert (tmp_path / "best_plan_model.pt").is_file()
    assert (tmp_path / "loss_curve.png").stat().st_size > 0
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[-1]["is_best"] == "1"
    status = json.loads((tmp_path / "training_status.json").read_text())
    assert status["status"] == "complete"
    assert status["global_step"] == 6
    assert load_inverse_checkpoint(tmp_path / "latest_model.pt")["epoch"] == 1
    output = capsys.readouterr().out
    assert "[epoch] stage_plan 2/2" in output
    assert "[checkpoint:latest]" in output
    assert "[checkpoint:best]" in output
    assert "[training:complete]" in output


def test_trainer_records_failure_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = InverseTrainer(
        _designer(), device="cpu", run_dir=tmp_path, checkpoint_provenance=_provenance()
    )

    def fail_epoch(*args, **kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(trainer, "_run_epoch", fail_epoch)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        trainer.train_stage(
            "stage_plan", [object()], [object()], epochs=1, learning_rate=1.0e-4
        )
    status = json.loads((tmp_path / "training_status.json").read_text())
    assert status["status"] == "failed"
    assert status["stage"] == "stage_plan"
    assert status["epoch"] == 1
    assert status["error_type"] == "RuntimeError"
    assert "synthetic training failure" in status["traceback"]
