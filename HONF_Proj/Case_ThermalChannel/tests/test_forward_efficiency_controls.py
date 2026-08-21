from __future__ import annotations

import pytest
import torch

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
