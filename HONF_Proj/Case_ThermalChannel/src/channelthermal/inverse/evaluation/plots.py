"""Predicted-only plots for verified `R,c -> G -> D -> G_hat` populations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from honf_inverse_core.sampling.contracts import CandidateRecord, InverseSamplingResult


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_top_candidate(candidate: CandidateRecord, output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    centers = np.asarray(candidate.design["module_centers"])
    present = np.asarray(candidate.design["module_present"]) > 0.5
    heat = np.asarray(candidate.design["heat_powers"])
    environment = candidate.outputs["environment"]
    x = np.asarray(environment["x_grid"])
    y = np.asarray(environment["y_grid"])
    field = np.asarray(environment["pred_field_grid"])
    channel_order = list(environment["channel_order"])

    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.scatter(centers[present, 0], centers[present, 1], s=80 + 5 * np.abs(heat[present]), c=heat[present], cmap="inferno")
    for index in np.flatnonzero(present):
        axis.text(centers[index, 0], centers[index, 1], str(index), ha="center", va="center", color="white", fontsize=7)
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_aspect("equal")
    axis.set_title(f"Generated layout {candidate.candidate_id}")
    _save(figure, directory / "layout_plot.png")

    figure, axes = plt.subplots(2, 3, figsize=(13, 6))
    for index, name in enumerate(channel_order):
        image = axes.flat[index].pcolormesh(x, y, field[..., index], shading="auto")
        axes.flat[index].set_title(f"Predicted {name}")
        figure.colorbar(image, ax=axes.flat[index])
    axes.flat[-1].axis("off")
    _save(figure, directory / "predicted_fields.png")

    internal = np.asarray(candidate.outputs["internal"])
    if internal.shape[-1:] == (1,):
        internal = internal[..., 0]
    figure, axis = plt.subplots(figsize=(8, 4))
    for index in np.flatnonzero(present):
        axis.plot(internal[index], label=f"module {index}")
    axis.set_title("Predicted internal module temperature")
    axis.set_xlabel("local query index")
    axis.legend(fontsize=7)
    _save(figure, directory / "internal_module_temperature_plots.png")

    planned = candidate.planned_compact_raw
    figure, axis = plt.subplots(figsize=(8, 3.5))
    for edge in range(planned.shape[0]):
        axis.annotate(
            "",
            xy=planned[edge, 3:5],
            xytext=planned[edge, 1:3],
            arrowprops={"arrowstyle": "->", "alpha": 0.25 + 0.75 * planned[edge, 7], "lw": 1.5},
        )
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_title("Planned compact mechanism")
    _save(figure, directory / "planned_mechanism_diagram.png")

    full = candidate.realized_full_plan
    A_mh = np.asarray(full["A_mh"])
    regions = np.asarray(full["hyper_region_coords"])
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.scatter(centers[present, 0], centers[present, 1], c="black", label="modules")
    axis.scatter(regions[:, 0], regions[:, 1], marker="x", c=np.arange(regions.shape[0]), cmap="tab10", label="regions")
    for module in np.flatnonzero(present):
        edge = int(np.argmax(A_mh[module]))
        axis.plot([centers[module, 0], regions[edge, 0]], [centers[module, 1], regions[edge, 1]], alpha=0.55)
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_title("Realized final organization")
    axis.legend(fontsize=7)
    _save(figure, directory / "realized_organization_diagram.png")

    difference = candidate.planned_compact_normalized - candidate.realized_compact_normalized
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, values, title in zip(
        axes,
        (candidate.planned_compact_normalized, candidate.realized_compact_normalized, difference),
        ("planned G", "realized G_hat", "planned - realized"),
    ):
        image = axis.imshow(values, aspect="auto", cmap="coolwarm")
        axis.set_title(title)
        axis.set_xlabel("compact feature")
        axis.set_ylabel("canonical edge")
        figure.colorbar(image, ax=axis, fraction=0.046)
    _save(figure, directory / "planned_vs_realized_comparison.png")

    labels = [str(term["request_type"]).replace("_", "\n") for term in candidate.request_terms]
    contributions = [float(term["violation_normalized"]) * float(term["weight"]) for term in candidate.request_terms]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(range(len(labels)), contributions)
    axis.set_xticks(range(len(labels)), labels, fontsize=7)
    axis.set_title("Weighted request-term contributions")
    _save(figure, directory / "request_term_contribution_chart.png")


def plot_population(result: InverseSamplingResult, output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    raw = result.raw_unguided
    corrected = result.corrected
    accepted = result.accepted_one_pass
    groups = [raw] + ([corrected, accepted] if corrected else [])
    labels = ["raw"] + (["proposal", "accepted"] if corrected else [])

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, [np.mean([candidate.request_satisfied for candidate in group]) for group in groups])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Request success fraction")
    axes[1].bar(labels, [np.mean([candidate.geometry_valid for candidate in group]) for group in groups])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Geometry validity fraction")
    _save(figure, directory / "raw_vs_corrected_success.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    for group, label in zip(groups, labels):
        axis.hist([candidate.request_violation for candidate in group], bins=16, alpha=0.55, label=label)
    axis.set_title("Request score distribution")
    axis.legend()
    _save(figure, directory / "request_score_distribution.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    for group, label in zip(groups, labels):
        axis.hist([candidate.plan_distance for candidate in group], bins=16, alpha=0.55, label=label)
    axis.set_title("Planned-realized mismatch distribution")
    axis.legend()
    _save(figure, directory / "plan_mismatch_distribution.png")

    candidates = result.final_ranked
    matrix = np.zeros((len(candidates), len(candidates)), dtype=np.float32)
    for left in range(len(candidates)):
        for right in range(len(candidates)):
            lc = np.asarray(candidates[left].design["module_centers"])
            rc = np.asarray(candidates[right].design["module_centers"])
            matrix[left, right] = np.sqrt(np.mean(np.square(lc - rc)))
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="viridis")
    axis.set_title("Final candidate layout diversity")
    figure.colorbar(image, ax=axis)
    _save(figure, directory / "candidate_diversity.png")

    population = sorted(raw + corrected, key=lambda candidate: max(candidate.forward_call_indices))
    calls, best, running = [], [], float("inf")
    for candidate in population:
        running = min(running, candidate.request_violation)
        calls.append(max(candidate.forward_call_indices))
        best.append(running)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.step(calls, best, where="post")
    axis.set_xlabel("HONF forward calls")
    axis.set_ylabel("best request violation")
    axis.set_title("Best score vs HONF calls")
    _save(figure, directory / "best_score_vs_honf_calls.png")


__all__ = ["plot_population", "plot_top_candidate"]
