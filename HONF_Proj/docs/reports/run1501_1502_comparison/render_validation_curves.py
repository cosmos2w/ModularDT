"""Render the recorded 5000-epoch sampled-validation trajectories.

Run 1804's root metrics CSV has malformed values after its epoch-500 resume.
Its finite resume log is the source for epochs 501--5000.  No model inference
or training is performed by this script.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "Trained_Results/ThermalChannel/HONF_Forward_Runs"
OUTPUT = Path(__file__).resolve().parent / "figures"
RUNS = {
    "1404 classic HONF": ("Run_1404_*", "#607d8b"),
    "1804 dense": ("Run_1804_*", "#3f51b5"),
    "1501 entmax": ("Run_1501_*", "#da7c30"),
    "1502 E-sparsemax": ("Run_1502_*", "#1a8f78"),
}
LOG_PATTERN = re.compile(
    r"\[epoch\s+(\d+)\].*?val=([0-9.eE+-]+) "
    r"val_field=([0-9.eE+-]+) val_temp=([0-9.eE+-]+)"
)


def _finite(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def read_run(pattern: str) -> dict[int, dict[str, float]]:
    paths = list(RUN_ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one run for {pattern}, found {len(paths)}")
    run_dir = paths[0]
    values: dict[int, dict[str, float]] = {}
    with (run_dir / "metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            if "Run_1804_" in run_dir.name and epoch > 500:
                continue
            metrics = {
                "field": _finite(row.get("val_field_mse", "")),
                "temperature": _finite(row.get("val_temperature_mse", "")),
                "total": _finite(row.get("val_loss_total", "")),
            }
            if all(value is not None for value in metrics.values()):
                values[epoch] = metrics
    if "Run_1804_" in run_dir.name:
        logs = list((run_dir / "logs").glob("resume_to_5000_*.log"))
        if len(logs) != 1:
            raise RuntimeError(f"Expected one Run 1804 resume log, found {len(logs)}")
        for epoch, total, field, temperature in LOG_PATTERN.findall(
            logs[0].read_text(errors="replace")
        ):
            values[int(epoch)] = {
                "total": float(total),
                "field": float(field),
                "temperature": float(temperature),
            }
    expected = set(range(1, 5001))
    if set(values) != expected:
        raise RuntimeError(
            f"{run_dir.name}: missing {len(expected - set(values))} epochs, "
            f"extra {len(set(values) - expected)}"
        )
    return values


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), constrained_layout=True)
    x = np.arange(1, 5001)
    for label, (pattern, color) in RUNS.items():
        data = read_run(pattern)
        for axis, metric, title in zip(
            axes,
            ("field", "temperature"),
            ("Field MSE", "Temperature MSE"),
            strict=True,
        ):
            raw = np.array([data[epoch][metric] for epoch in x])
            smooth = np.array(
                [np.median(raw[max(0, i - 25) : min(len(raw), i + 26)])
                 for i in range(len(raw))]
            )
            axis.plot(x, raw, color=color, linewidth=0.38, alpha=0.13)
            axis.plot(x, smooth, color=color, linewidth=1.65, label=label)
            axis.set_yscale("log")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Sampled validation MSE")
            axis.grid(alpha=0.2, which="both")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Recorded validation trajectories · 51-epoch rolling median", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"validation_trajectories.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
