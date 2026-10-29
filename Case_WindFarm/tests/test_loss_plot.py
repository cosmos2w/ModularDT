"""Real numerical rendering check for WindFarm training loss history."""

from __future__ import annotations

from pathlib import Path

from windfarm.loss_plot import render_loss_history


def test_render_loss_history_writes_validation_objective_and_pdf_master(tmp_path: Path) -> None:
    history = [
        {
            "epoch": 1,
            "loss_total": 1.2,
            "val_loss_total": "",
            "train_volume_mse": 1.1,
            "val_volume_mse": "",
            "train_band_mse": 1.4,
            "val_band_mse": "",
        },
        {
            "epoch": 2,
            "loss_total": 0.7,
            "val_loss_total": 0.8,
            "train_volume_mse": 0.6,
            "val_volume_mse": 0.8,
            "train_band_mse": 0.9,
            "val_band_mse": 1.0,
        },
        {
            "epoch": 3,
            "loss_total": 0.4,
            "val_loss_total": 0.5,
            "train_volume_mse": 0.3,
            "val_volume_mse": 0.5,
            "train_band_mse": 0.6,
            "val_band_mse": 0.7,
        },
    ]
    output = render_loss_history(history, tmp_path / "plots" / "training" / "loss_history.png", title="numeric test")
    assert output.is_file()
    assert output.stat().st_size > 0
    assert output.with_suffix(".pdf").is_file()
    assert output.with_suffix(".pdf").stat().st_size > 0
    assert not list(output.parent.glob(".loss_history.*.tmp.png"))
