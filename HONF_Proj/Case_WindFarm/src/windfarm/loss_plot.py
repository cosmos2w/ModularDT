"""Small, reusable loss-history plots for WindFarm training runs."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _finite_points(rows: Sequence[Mapping[str, Any]], field: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for row in rows:
        try:
            epoch = int(float(row["epoch"]))
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0.0 and math.isfinite(value):
            epochs.append(epoch)
            values.append(value)
    return epochs, values


def _replace_figure(fig: Any, output: Path) -> None:
    """Save a figure through a same-directory temporary PNG replacement."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp{output.suffix or '.png'}")
    try:
        fig.savefig(temporary, dpi=150)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_loss_history(
    history: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Render separate objective, volume, and band loss panels.

    Validation values are absent on unscheduled epochs and are simply omitted
    from their panel.  All panels use a logarithmic y-axis because the study
    follows positive standardized MSE values over several orders of magnitude.
    The target is replaced atomically after the complete PNG is rendered.
    """

    if not history:
        raise ValueError("cannot render an empty WindFarm loss history")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output)
    latest_epoch = max(int(float(row["epoch"])) for row in history)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    try:
        panels = (
            (axes[0], (("loss_total", "Training objective"),), "Training objective"),
            (
                axes[1],
                (("train_volume_mse", "Train"), ("val_volume_mse", "Validation")),
                "Volume MSE",
            ),
            (
                axes[2],
                (("train_band_mse", "Train"), ("val_band_mse", "Validation")),
                "Rotor-height-band MSE",
            ),
        )
        for axis, series, panel_title in panels:
            for field, label in series:
                epochs, values = _finite_points(history, field)
                if epochs:
                    axis.plot(epochs, values, label=label, linewidth=1.2,
                              marker="o" if len(epochs) == 1 else None)
            axis.set(
                title=panel_title,
                xlabel="Epoch",
                ylabel="Standardized MSE (log scale)",
                yscale="log",
            )
            axis.grid(alpha=0.2)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=8)
        figure.suptitle(title or f"WindFarm loss history through epoch {latest_epoch}")
        _replace_figure(figure, output_path)
    finally:
        plt.close(figure)
    return output_path


__all__ = ["render_loss_history"]
