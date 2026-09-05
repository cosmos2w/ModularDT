#!/usr/bin/env python3
"""Plot and summarize aligned HONF training histories through one epoch budget."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-stage1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("loss_total", "field_mse", "val_loss_total", "val_field_mse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL=METRICS_CSV",
        help="Named history; repeat in desired display order.",
    )
    parser.add_argument("--max-epoch", type=int, default=500)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    return parser.parse_args()


def load_series(raw: str, max_epoch: int) -> tuple[str, list[dict[str, float]]]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=METRICS_CSV, got {raw!r}.")
    label, raw_path = raw.split("=", 1)
    path = Path(raw_path)
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            epoch = int(float(source["epoch"]))
            if epoch > max_epoch:
                continue
            row = {"epoch": float(epoch)}
            for metric in METRICS:
                value = float(source.get(metric, "nan"))
                row[metric] = value
            rows.append(row)
    if not rows or int(rows[-1]["epoch"]) != max_epoch:
        raise ValueError(f"{path} does not contain the requested epoch {max_epoch} endpoint.")
    return label, rows


def main() -> int:
    args = parse_args()
    if args.max_epoch < 1:
        raise ValueError("--max-epoch must be positive.")
    histories = [load_series(raw, args.max_epoch) for raw in args.series]
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    panels = (
        ("loss_total", "val_loss_total", "Total physical objective"),
        ("field_mse", "val_field_mse", "Field mean-squared error"),
    )
    for ax, (train_metric, val_metric, title) in zip(axes, panels):
        for label, rows in histories:
            epochs = [row["epoch"] for row in rows]
            line = ax.plot(epochs, [row[val_metric] for row in rows], label=f"{label} — development")[0]
            ax.plot(
                epochs,
                [row[train_metric] for row in rows],
                linestyle="--",
                alpha=0.55,
                color=line.get_color(),
                label=f"{label} — train",
            )
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f"Matched training histories through epoch {args.max_epoch}")
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)

    fields = ["model_label", "metric", "endpoint_epoch", "endpoint_value", "best_epoch", "best_value"]
    with args.output_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, rows in histories:
            for metric in METRICS:
                finite = [row for row in rows if math.isfinite(row[metric])]
                if not finite:
                    continue
                best = min(finite, key=lambda row: row[metric])
                writer.writerow(
                    {
                        "model_label": label,
                        "metric": metric,
                        "endpoint_epoch": int(rows[-1]["epoch"]),
                        "endpoint_value": rows[-1][metric],
                        "best_epoch": int(best["epoch"]),
                        "best_value": best[metric],
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
