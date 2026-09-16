"""Plot recorded Run-2000 learning, router updates, and effective pair retention."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def values(rows, key):
    return np.asarray([float(row.get(key, "nan")) for row in rows], dtype=float)


def render(run_dir: Path, output: Path):
    rows = read_rows(run_dir / "metrics.csv")
    routing = read_rows(run_dir / "routing_metrics.csv")
    epoch = values(rows, "epoch")
    if epoch.tolist() != list(range(1, 501)):
        raise ValueError("The final training figure requires the completed, contiguous 1..500 history")
    if values(routing, "epoch").tolist() != epoch.tolist():
        raise ValueError("Routing and ordinary histories must have the same epochs")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    models = [("Run 2000", rows, "#D2691E")]
    sources = {"Run 2000": str((run_dir / "metrics.csv").resolve())}
    for run, label, color in (("1401", "Legacy 1401", "#6B7280"),
                              ("1804", "Dense 1804", "#2563A6"),
                              ("1806", "Regional 1806", "#47845B")):
        matches = list(run_dir.parent.glob(f"Run_{run}_*/metrics.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected one maintained history for Run {run}, found {matches}")
        parent = [row for row in read_rows(matches[0]) if int(row["epoch"]) <= 500]
        models.append((label, parent, color))
        sources[label] = str(matches[0].resolve())
    for ax, key, title in zip(axes[0], ("val_field_mse", "val_temperature_mse"),
                              ("Sampled validation field MSE", "Sampled validation temperature MSE"), strict=True):
        for label, history, color in models:
            x, y = values(history, "epoch"), values(history, key)
            finite = np.isfinite(y) & (y > 0)
            if not finite.any():
                raise ValueError(f"No positive finite {key} observations for {label}")
            ax.plot(x[finite], y[finite], color=color, label=label, lw=1.1, alpha=.9)
        ax.set(title=title, xlabel="Epoch", ylabel="MSE", yscale="log")
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    ax = axes[1, 0]
    for key, label, color in (("routing_preclip_gradient_norm", "Router preclip norm", "#7B4BA0"),
                              ("routing_parameter_update_norm", "Router update norm", "#D2691E")):
        y = values(routing, key)
        finite = np.isfinite(y) & (y > 0)
        ax.plot(epoch[finite], y[finite], marker="o", ms=3, lw=1, color=color, label=label)
    ax.set(title="Recorded first-batch router diagnostics", xlabel="Epoch", ylabel="L2 norm", yscale="log")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    ax = axes[1, 1]
    active_m = values(routing, "routing_summary_routing_candidate_count")
    module_pairs = values(routing, "routing_summary_routing_module_fine_pair_count")
    env_pairs = values(routing, "routing_summary_routing_environment_fine_pair_count")
    module_retention = np.divide(module_pairs, active_m, out=np.full_like(module_pairs, np.nan), where=active_m > 0)
    env_retention = env_pairs / 192.0
    ax.plot(epoch, module_retention, lw=1.1, label="QM: mean pair count / mean active M", color="#2563A6")
    ax.plot(epoch, env_retention, lw=1.1, label="QE: mean pair count / 192", color="#47845B")
    ax.set(title="P2 effective fine-pair retention", xlabel="Epoch", ylabel="Ratio to active-source dense work", ylim=(0, 1.03))
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    for extension in ("png", "svg"):
        fig.savefig(output / f"training_trajectory.{extension}", dpi=180)
    plt.close(fig)
    field = values(rows, "val_field_mse")
    recorded = np.isfinite(values(rows, "preclip_gradient_norm"))
    def observed(row, keys):
        result = {}
        for key in keys:
            value = float(row.get(key) or "nan")
            result[key] = value if np.isfinite(value) else None
        return result

    learning_keys = ("field_mse", "temperature_mse", "val_field_mse", "val_temperature_mse")
    milestones = [
        {"epoch": step, **observed(rows[step - 1], learning_keys)}
        for step in (1, 10, 50, 100, 250, 500)
    ]
    gradient_observations = [
        {"epoch": int(epoch[index]),
         **observed(rows[index], ("preclip_gradient_norm", "gradient_clip_scale", "parameter_update_norm")),
         **observed(routing[index], ("routing_preclip_gradient_norm", "routing_parameter_update_norm"))}
        for index in np.flatnonzero(recorded)
    ]
    summary = {
        "status": "complete", "epochs": 500, "sources": sources,
        "best_sampled_validation_field_epoch": int(epoch[np.nanargmin(field)]),
        "best_sampled_validation_field_mse": float(np.nanmin(field)),
        "exact500_sampled_validation_field_mse": float(field[-1]),
        "validation_field_mean_epochs301_400": float(field[300:400].mean()),
        "validation_field_mean_epochs401_500": float(field[400:500].mean()),
        "recorded_gradient_epochs": epoch[recorded].astype(int).tolist(),
        "recorded_first_batch_gradients": gradient_observations,
        "milestones": milestones,
        "trailing_learning_means": {
            str(window): {key: float(values(rows[-window:], key).mean()) for key in learning_keys}
            for window in (50, 100)
        },
        "training_seconds": float(values(rows, "train_wall_seconds").sum()),
        "validation_seconds": float(values(rows, "val_wall_seconds").sum()),
        "peak_training_allocated_mib": float(values(rows, "peak_cuda_memory_mb").max()),
        "final_qm_ratio_of_means": float(module_retention[-1]),
        "final_qe_ratio_of_means": float(env_retention[-1]),
        "limitations": ["Sampled validation MSE differs from full-grid fluid L2.",
                        "Router norms are recorded first-batch observations at selected epochs, not every optimizer step.",
                        "Pair retention uses ratios of recorded batch-averaged counts; the endpoint ledger reports exact case/phase work.",
                        "Training and validation seconds exclude plot/checkpoint write overhead."]}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.run_dir, args.output)


if __name__ == "__main__":
    main()
