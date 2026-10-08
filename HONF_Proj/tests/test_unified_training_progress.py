from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import honf_runtime.unified_training as runtime
import torch
from honf_runtime.run_layout import RunLayout
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    TaskBatch,
    TrainingEngine,
)
from torch import nn


@dataclass
class _ProgressRecorder:
    options: dict[str, Any]
    events: list[tuple[str, Any]]
    total: int = 0
    current: int = 0
    postfix: dict[str, str] | None = None
    snapshots: list[dict[str, str]] | None = None
    closed: bool = False

    def __post_init__(self):
        self.total = self.options["total"]
        self.postfix = {}
        self.snapshots = []

    def set_postfix(self, values, *, refresh=True):
        del refresh
        self.postfix = dict(values)

    def update(self, amount):
        self.current += amount
        self.snapshots.append(dict(self.postfix))
        self.events.append(("progress", amount))

    def close(self):
        self.closed = True


class _ProgressProvider:
    def identity_payload(self):
        return {"dataset": "progress-test"}

    def epoch_cases(self, epoch, seed):
        del epoch, seed
        return tuple(range(5))

    def make_batch(self, case_keys, key: SamplingKey):
        del key
        values = torch.tensor(case_keys, dtype=torch.float32).reshape(-1, 1)
        return TaskBatch(values, None, 2.0 * values + 1.0, case_keys=tuple(case_keys))

    def loss_denominators(self, batches, phase, arm):
        del phase, arm
        return {"native": float(sum(batch.targets.numel() for batch in batches))}

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        del receivers, execution_mode, phase, epoch, temperature
        return model(scene), None

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase, auxiliary_state
        return {"native": LossTerm((predictions - targets).square().sum(), targets.numel())}

    def validation_batches(self):
        yield self.make_batch((0, 1), SamplingKey(3, 1, 0, 0, "warmup", "full_detail"))

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del auxiliary_state
        return {"squared_error": float((predictions - targets).square().sum()), "count": targets.numel()}

    def reduce_native_metrics(self, records):
        return {"field_score": sum(row["squared_error"] for row in records)
                / sum(row["count"] for row in records)}

    def optimizer_groups(self, model, arm, stage):
        del model, arm, stage
        return (OptimizerGroupSpec(
            name="predictor", parameter_names=("weight", "bias"),
            schedule=ScheduleSpec(peak_lr=0.02, warmup_start_lr=0.01, warmup_epochs=1,
                                 hold_through_epoch=1, total_epochs=4, final_lr=0.001),
            weight_decay=0.0,
        ),)

    def work_counts(self, batch, predictions, auxiliary_state):
        del predictions, auxiliary_state
        return {"query_rows": batch.targets.numel()}


def test_progress_is_tty_gated_and_reports_completed_macro_updates(tmp_path, monkeypatch):
    events: list[tuple[str, Any]] = []
    bars: list[_ProgressRecorder] = []

    class _Terminal:
        @staticmethod
        def isatty():
            return True

    monkeypatch.setattr(runtime.sys, "stderr", _Terminal())

    def make_bar(**options):
        bar = _ProgressRecorder(options, events)
        bars.append(bar)
        return bar

    monkeypatch.setattr(runtime, "tqdm", make_bar)
    engine = TrainingEngine(EngineConfig(
        seed=3, microbatch_cases=2, effective_cases=4, total_epochs=4,
        warmup_epochs=1, open_through_epoch=2, soft_through_epoch=3,
        monitor_every=2, gradient_clip=10.0,
    ), device="cpu")
    original_run_update = engine._run_update

    def record_update(*args, **kwargs):
        result = original_run_update(*args, **kwargs)
        events.append(("optimizer", None))
        return result

    monkeypatch.setattr(engine, "_run_update", record_update)
    output = Path(tmp_path) / "progress"
    engine.fit(nn.Linear(1, 1), _ProgressProvider(), output,
               identity={"run": "progress"}, arm="full_detail", stop_after=1)

    assert len(bars) == 1
    bar = bars[0]
    assert bar.options["file"] is runtime.sys.stderr
    assert bar.options["disable"] is False
    assert bar.options["unit"] == "case" and bar.options["leave"] is False
    assert bar.total == bar.current == 5 and bar.closed
    assert events == [("optimizer", None), ("progress", 4), ("optimizer", None), ("progress", 1)]
    assert [row["updates"] for row in bar.snapshots] == ["1/2", "2/2"]
    assert all(float(row["loss"]) >= 0.0 for row in bar.snapshots)
    assert all("predictor:" in row["lr"] and row["phase"] == "warmup" for row in bar.snapshots)
    assert RunLayout(output).read_path("progress.json") == output / "logs" / "progress.json"
    assert not (output / "progress.json").exists()
    assert (output / "logs" / "active_process.json").is_file()
    assert (output / "metrics" / "history.json").is_file()
    assert (output / "checkpoints" / "latest_model.pt").is_file()
    assert (output / "artifact_layout.json").is_file()
    assert (output / "configs" / "experiment_identity.json").is_file()
    assert (output / "environment" / "software.json").is_file()
    assert (output / "environment" / "source_state.json").is_file()
