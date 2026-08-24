from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from channelthermal.training import epoch as epoch_module
from channelthermal.workflows.train_forward import (
    pack_scalar_metrics,
    reuses_primary_validation_for_predicted_mode,
    should_save_latest_checkpoint,
)


def test_predicted_validation_is_reused_independently_of_unused_mix_ratio() -> None:
    assert reuses_primary_validation_for_predicted_mode("predicted")
    assert reuses_primary_validation_for_predicted_mode("PREDICTED")
    assert not reuses_primary_validation_for_predicted_mode("mixed")
    assert not reuses_primary_validation_for_predicted_mode("teacher")


def test_latest_checkpoint_cadence_is_backward_compatible_and_saves_final_epoch() -> None:
    assert should_save_latest_checkpoint(3, 20, {"save_latest": True})
    cadence_config = {"save_latest": True, "save_latest_every_epochs": 10}
    assert not should_save_latest_checkpoint(9, 23, cadence_config)
    assert should_save_latest_checkpoint(10, 23, cadence_config)
    assert should_save_latest_checkpoint(23, 23, cadence_config)
    assert not should_save_latest_checkpoint(
        10,
        20,
        {"save_latest": False, "save_latest_every_epochs": 10},
    )


def test_scalar_metrics_are_packed_without_changing_values() -> None:
    metrics = pack_scalar_metrics(
        {
            "loss": torch.tensor(1.25),
            "field_mse": torch.tensor(0.03125),
            "temperature_mse": torch.tensor(-2.0),
        }
    )

    assert tuple(metrics) == ("loss", "field_mse", "temperature_mse")
    assert metrics == pytest.approx(
        {"loss": 1.25, "field_mse": 0.03125, "temperature_mse": -2.0}
    )


class _EpochFixtureModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            channelthermal=SimpleNamespace(field_names=["temperature"])
        )

    def train(self, training: bool) -> None:
        self.training = training

    def __call__(self, **_inputs):
        prediction = torch.ones((1, 2, 1), dtype=torch.float32)
        return {
            "pred_field": prediction,
            "pred_internal_temperature": prediction.new_empty((1, 0, 1)),
            "pred_interface": prediction.new_empty((1, 0, 1)),
            "pred_port_condition": prediction.new_empty((1, 0, 5)),
        }


def test_run_epoch_owns_runtime_metric_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        epoch_module,
        "channelthermal_field_mse",
        lambda prediction, target, *_args, **_kwargs: (prediction - target).square().mean(),
    )
    monkeypatch.setattr(
        epoch_module,
        "organizer_regularization",
        lambda output, _loss_cfg: output["pred_field"].new_zeros(()),
    )
    monkeypatch.setattr(epoch_module, "compute_honf_diagnostics", lambda *_args, **_kwargs: {})
    batch = {
        "field_targets": torch.zeros((1, 2, 1), dtype=torch.float32),
        "query_xy": torch.zeros((1, 2, 2), dtype=torch.float32),
        "structure": {"module_present": torch.ones((1, 1), dtype=torch.float32)},
    }
    common = {
        "model": _EpochFixtureModel(),
        "device": torch.device("cpu"),
        "loss_cfg": {},
        "optimizer": None,
        "scaler": None,
        "amp": False,
        "max_batches": None,
        "local_port_condition_mode": "predicted",
        "mixed_teacher_ratio": 0.0,
        "effective_internal_temperature_weight": 0.0,
        "effective_interface_weight": 0.0,
        "predicted_consistency_weight": 0.0,
    }

    metrics = epoch_module.run_epoch(loader=[batch], **common)
    empty_metrics = epoch_module.run_epoch(loader=[], **common)

    assert metrics["loss_total"] == pytest.approx(1.0)
    assert metrics["field_mse"] == pytest.approx(1.0)
    assert math.isnan(empty_metrics["loss_total"])
