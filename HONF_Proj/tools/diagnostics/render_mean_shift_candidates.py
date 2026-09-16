"""Render measured candidate attraction beside actual fine support."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text())
    rows = payload["results"]
    if not rows:
        raise ValueError("At least one diagnostic case is required")
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.4 * len(rows)),
                             squeeze=False, constrained_layout=True)
    for row, panels in zip(rows, axes, strict=True):
        prep = next(p for p in row["preparations"] if p["phase"] == "p2_field")
        candidates = prep["candidates"]
        trajectory = np.asarray(candidates["trajectory_physical"])[0]
        valid = np.asarray(candidates["valid_candidate_mask"], dtype=bool)[0]
        trajectory = trajectory[:, valid]
        if trajectory.shape[0] != 4 or trajectory.shape[-1] != 2:
            raise ValueError("This figure expects three iterations in two physical dimensions")
        ax = panels[0]
        for index in range(trajectory.shape[1]):
            path = trajectory[:, index]
            ax.plot(path[:, 0], path[:, 1], color="#778899", lw=1, alpha=.7)
            for start, end in pairwise(path):
                ax.annotate("", xy=end, xytext=start,
                            arrowprops={"arrowstyle": "->", "color": "#778899", "lw": .8})
        ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], marker="o", s=35,
                   facecolors="none", edgecolors="#2563A6", label="Module seed")
        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], marker="x", s=40,
                   color="#D2691E", label="After three updates")
        ax.set(title=f"Case {row['case_id']}: P2 candidate paths", xlabel="Physical x",
               ylabel="Physical y", aspect="equal")
        ax.margins(.25)
        ax.legend(frameon=False, fontsize=8)
        ax = panels[1]
        physical = candidates["physical"]
        drift = [r["summary"]["mean"] for r in physical["drift_from_seed"]]
        separation = [r["summary"].get("min", np.nan)
                      for r in physical["minimum_separation"]["per_iteration"]]
        ax.plot(range(4), drift, "o-", color="#2563A6", label="Mean drift from seed")
        ax.plot(range(4), separation, "s--", color="#D2691E", label="Minimum separation")
        ax.set(title="Attraction in physical space", xlabel="Mean-shift iteration",
               ylabel="Physical distance", xticks=range(4))
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
        ax = panels[2]
        comparison = row["candidate_intervention_comparison"]["fine_support_statistics"]
        x = np.arange(2)
        for offset, strategy, color, label in (
            (-.18, "module_hubs", "#778899", "Same weights: module hubs"),
            (.18, "mean_shift", "#D2691E", "Mean shift"),
        ):
            support = comparison[strategy]
            incidence = support["source_and_candidate_occupancy"]
            fractions = [support[f"p2_field_{kind}"]["fine_pair_count"]["mean"]
                         / incidence[kind]["active_source_count"]["mean"]
                         for kind in ("module", "environment")]
            ax.bar(x + offset, fractions, .36, color=color, label=label)
        ax.set(title="Actual P2 fine work retained", xticks=x, xticklabels=["QM", "QE"],
               ylabel="Fraction of active-source dense pairs", ylim=(0, 1.35))
        ax.axhline(1, color="#777777", lw=.7, ls=":")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    epoch = payload["checkpoint"]["epoch"]
    fig.suptitle(f"Run 2100 · epoch {epoch} · {payload['query_count']} fixed field queries per case\n"
                 "Candidate attraction and fine execution are separate measurements", fontsize=12)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"candidate_attraction.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.source, args.output)


if __name__ == "__main__":
    main()
