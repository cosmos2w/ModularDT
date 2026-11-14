from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.run_layout import RunLayout
from honf_runtime.unified_training import TrainingEngine, read_training_history
from tests.test_unified_training import _ToyProvider, _config
from windfarm.training.unified_task import ROLE_NAMES, WindRefinementTask


class TinyWindLossProvider(_ToyProvider):
    """Train a tiny predictor through the real Wind loss and CSV inventory."""
    training_output_policy = "csv_v3"
    training_history_columns = WindRefinementTask.training_history_columns

    def __init__(self):
        super().__init__()
        self.role_scales = {role: 1. for role in ROLE_NAMES}
        self.recipe = None
        self.objective_weights = {}
        self._message_scale = 1.

    def loss_denominators(self, batches, phase, arm):
        count = sum(batch.targets.numel() for batch in batches) * 3
        return {**{f"native_role/{role}": count for role in ROLE_NAMES}, "base_loss": count}

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        count = targets.numel()
        values = predictions.reshape(count, 1, 1).expand(count, len(ROLE_NAMES), 3)
        expected = targets.reshape(count, 1, 1).expand_as(values)
        roles = torch.arange(len(ROLE_NAMES)).expand(count, -1)
        prediction = SimpleNamespace(main_mps=values, full_mps=values, role_ids=roles, execution_mode="all_fine")
        target = SimpleNamespace(velocity_mps=expected.detach().numpy())
        auxiliary = {"base_numerator": values[:, 0].square().sum(), "base_denominator": count * 3}
        return WindRefinementTask.loss_terms(self, prediction, target, phase, auxiliary)


def test_wind_loss_inventory_and_tiny_csv_fit_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    config = replace(_config(), total_epochs=2, monitor_every=1, training_mode="joint", progress_enabled=False)
    engine = TrainingEngine(config, device="cpu")
    run = tmp_path / "wind"
    engine.fit(nn.Linear(1, 1), TinyWindLossProvider(), run, identity={"case": "wind-fixture"},
               arm="adaptive_detail", stop_after=1)
    engine.fit(nn.Linear(1, 1), TinyWindLossProvider(), run, identity={"case": "wind-fixture"},
               arm="adaptive_detail", stop_after=2, resume_checkpoint=RunLayout(run).read_path("latest_model.pt"))
    rows = read_training_history(RunLayout(run).read_path("history.csv"))
    assert len(rows) == 2
    assert set(rows[-1]["train_losses"]) == {*(f"native_role/{role}" for role in ROLE_NAMES), "base_loss"}
    assert np.isfinite(rows[-1]["train_loss_total"])

    # Every scheduled adaptive term is declared before it first becomes active.
    provider = TinyWindLossProvider()
    provider._expected_work_calibration = {}
    provider._expected_work_weight = .1
    values = torch.ones((1, len(ROLE_NAMES), 3), requires_grad=True)
    prediction = SimpleNamespace(main_mps=values, full_mps=values,
        role_ids=torch.arange(len(ROLE_NAMES)).reshape(1, -1), execution_mode="adaptive")
    target = SimpleNamespace(velocity_mps=np.zeros(values.shape))
    aux = {f"{name}_{part}": values.sum() if part == "numerator" else 1
           for name in ("base", "router_importance", "expected_work") for part in ("numerator", "denominator")}
    declared = set(provider.training_history_columns())
    for phase in ("open", "soft", "hard"):
        terms = WindRefinementTask.loss_terms(provider, prediction, target, phase, aux)
        assert {f"train_losses.{name}" for name in terms} <= declared
