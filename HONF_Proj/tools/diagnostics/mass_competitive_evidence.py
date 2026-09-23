"""Run-1500 mass-competition population evidence and visualizations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import occupancy_adaptive_evidence as occupancy
except ModuleNotFoundError:  # imported as tools.diagnostics.mass_competitive_evidence
    from tools.diagnostics import occupancy_adaptive_evidence as occupancy

EXPECTED_CASE_COUNT = occupancy.EXPECTED_CASE_COUNT
DEFAULT_KMAX = occupancy.DEFAULT_KMAX
DEFAULT_QUERY_COUNT = occupancy.DEFAULT_QUERY_COUNT


def _exact(payload: Any, key: str) -> Any:
    for container in occupancy._containers(payload):
        if key in container:
            return container[key]
    return None


def _vector(
    payload: Any,
    key: str,
    *,
    kmax: int,
    case_index: int = 0,
) -> np.ndarray:
    value = _exact(payload, key)
    if value is None:
        raise occupancy.OccupancyEvidenceError(f"missing Run-1500 diagnostic {key!r}")
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim > 1:
        if case_index < 0 or case_index >= int(array.shape[0]):
            raise occupancy.OccupancyEvidenceError(
                f"{key} case index is outside batch size {array.shape[0]}"
            )
        array = array[case_index]
    array = array.reshape(-1)
    if array.size != int(kmax) or not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise occupancy.OccupancyEvidenceError(
            f"{key} must be finite nonnegative [Kmax] data"
        )
    return array


def _batched_vector(value: Any) -> np.ndarray:
    """Preserve the singleton case axis expected by the occupancy adapter."""

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array[None, ...]
    return array


def canonicalize_case(
    payload: Any,
    *,
    case_index: int = 0,
    query_count: int | None = None,
    kmax: int | None = None,
) -> dict[str, Any]:
    """Extend the established support audit with explicit Run-1500 pi/gamma."""

    if not isinstance(payload, Mapping):
        raise occupancy.OccupancyEvidenceError("Run-1500 payload must be a mapping")
    adapted = dict(payload)
    aliases = {
        "occupancy_group_active_mask": "mass_competitive_plan_active_mask",
        "occupancy_group_prototype_ids": "mass_competitive_plan_prototype_ids",
        "occupancy_group_packed_valid": "mass_competitive_plan_packed_valid",
        "occupancy_group_k_plan": "mass_competitive_k_case_per_case",
        "occupancy_group_kappa": "mass_competitive_kappa",
    }
    for target, source in aliases.items():
        value = _exact(payload, source)
        if value is not None:
            adapted[target] = _batched_vector(value)
    record = occupancy.canonicalize_case(
        adapted,
        case_index=case_index,
        query_count=query_count,
        kmax=kmax,
    )
    row = record["case"]
    inferred_kmax = int(row["kmax"])
    pi = _vector(
        payload,
        "mass_competitive_pi",
        kmax=inferred_kmax,
        case_index=case_index,
    )
    gamma = _vector(
        payload,
        "mass_competitive_gamma",
        kmax=inferred_kmax,
        case_index=case_index,
    )
    if not np.isclose(float(np.sum(pi)), 1.0, atol=2.0e-5, rtol=0.0):
        raise occupancy.OccupancyEvidenceError("Run-1500 pi must sum to one")
    if not np.isclose(float(np.sum(gamma)), 1.0, atol=2.0e-5, rtol=0.0):
        raise occupancy.OccupancyEvidenceError("Run-1500 gamma must sum to one")
    active = gamma > 0.0
    kcase = int(np.count_nonzero(active))
    plan_active = np.asarray(row["active_mask"], dtype=bool)
    if not np.array_equal(active, plan_active):
        raise occupancy.OccupancyEvidenceError(
            "positive gamma support differs from the P0 prototype-ID plan"
        )
    kappa = float(1.0 / np.sum(gamma * gamma))
    if not math.isclose(float(row["kappa"]), kappa, abs_tol=2.0e-5, rel_tol=2.0e-5):
        raise occupancy.OccupancyEvidenceError(
            "runtime kappa differs from 1/sum(gamma**2)"
        )
    module_count = int(row["active_module_count"])
    row.update(
        {
            "kcase": kcase,
            "kplan": kcase,
            "kcase_minus_active_module_count": kcase - module_count,
            "kcase_equals_active_module_count": kcase == module_count,
            "pi": pi.tolist(),
            "gamma": gamma.tolist(),
            "pi_max": float(np.max(pi)),
            "gamma_max": float(np.max(gamma)),
            "retained_precompetition_mass": float(np.sum(pi[active])),
            "kappa": kappa,
        }
    )
    record["maps"]["pi"] = pi
    record["maps"]["gamma"] = gamma
    return record


def _scalar_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_population(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int = EXPECTED_CASE_COUNT,
) -> dict[str, Any]:
    summary = occupancy.summarize_population(rows, expected_cases=expected_cases)
    kcase = np.asarray([int(row["kcase"]) for row in rows], dtype=np.int64)
    module_count = np.asarray(
        [int(row["active_module_count"]) for row in rows], dtype=np.int64
    )
    if len(kcase) < 2 or float(np.std(kcase)) == 0.0 or float(np.std(module_count)) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(kcase, module_count)[0, 1])
    summary.update(
        {
            "kcase_histogram": summary.pop("kplan_histogram"),
            "kcase_mean": summary.pop("kplan_mean"),
            "kcase_median": summary.pop("kplan_median"),
            "kcase_min": summary.pop("kplan_min"),
            "kcase_max": summary.pop("kplan_max"),
            "prototype_selection_frequency": summary.pop(
                "prototype_occupancy_frequency"
            ),
            "pi_max": _scalar_summary(rows, "pi_max"),
            "gamma_max": _scalar_summary(rows, "gamma_max"),
            "retained_precompetition_mass": _scalar_summary(
                rows, "retained_precompetition_mass"
            ),
            "kcase_equals_module_count_cases": int(np.sum(kcase == module_count)),
            "pearson_kcase_module_count": correlation,
        }
    )
    summary.pop("kplan", None)
    summary.pop("kplan_minus_active_module_count", None)
    gate = summary["continuation_gate"]
    universal_k1 = bool(np.all(kcase == 1))
    universal_kmax = bool(np.all(kcase == DEFAULT_KMAX))
    case_dependent = bool(np.min(kcase) != np.max(kcase))
    gate.update(
        {
            "healthy_nontrivial_hypergraph": bool(
                case_dependent and not (universal_k1 or universal_kmax)
            ),
            "case_dependent_kcase": case_dependent,
            "universal_k1": universal_k1,
            "universal_kmax": universal_kmax,
            "decision": (
                "review_accuracy_and_executor_evidence_before_continuing"
                if case_dependent
                else "stop_no_case_dependent_discrete_k"
            ),
        }
    )
    gate.pop("case_dependent_kplan", None)
    return summary


def save_case_arrays(path: Path, maps: Mapping[str, np.ndarray]) -> None:
    occupancy.save_case_arrays(path, maps)


def _dominant(values: np.ndarray) -> np.ndarray:
    positive = values > 0.0
    return np.where(
        positive.any(axis=-1),
        np.argmax(np.where(positive, values, -np.inf), axis=-1),
        -1,
    )


def render_case_board(record: Mapping[str, Any], path: Path) -> None:
    """Render pi/gamma, source geometry, query use, and logical support."""

    maps = record["maps"]
    row = record["case"]
    module = np.asarray(maps["module_assignment"])
    environment = np.asarray(maps["environment_assignment"])
    query = np.asarray(maps["query_assignment"])
    module_xy = np.asarray(maps["module_coords"])
    environment_xy = np.asarray(maps["environment_coords"])
    query_xy = np.asarray(maps["query_coords"])
    module_centres = np.asarray(maps["module_centres"])
    environment_centres = np.asarray(maps["environment_centres"])
    joint_centres = np.asarray(maps["joint_centres"])
    pi = np.asarray(maps["pi"])
    gamma = np.asarray(maps["gamma"])
    active = gamma > 0.0
    colours = plt.get_cmap("tab20")
    figure, axes = plt.subplots(2, 4, figsize=(15.0, 7.2), constrained_layout=True)

    module_valid = module.sum(axis=-1) > 0.0
    module_dom = _dominant(module)
    axes[0, 0].scatter(
        module_xy[module_valid, 0],
        module_xy[module_valid, 1],
        c=module_dom[module_valid],
        cmap=colours,
        vmin=0,
        vmax=19,
        s=44,
    )
    axes[0, 0].scatter(
        module_centres[active, 0],
        module_centres[active, 1],
        marker="x",
        color="black",
        s=58,
        label="module centres",
    )
    axes[0, 0].set_title("Modules · dominant learned group")
    axes[0, 0].legend(frameon=False, fontsize=8)

    environment_dom = _dominant(environment)
    axes[0, 1].scatter(
        environment_xy[:, 0],
        environment_xy[:, 1],
        c=environment_dom,
        cmap=colours,
        vmin=0,
        vmax=19,
        s=10,
        alpha=0.8,
    )
    axes[0, 1].scatter(
        environment_centres[active, 0],
        environment_centres[active, 1],
        marker="+",
        color="black",
        s=58,
        label="environment centres",
    )
    axes[0, 1].set_title("Environment · dominant learned group")
    axes[0, 1].legend(frameon=False, fontsize=8)

    ids = np.arange(len(pi))
    width = 0.38
    axes[0, 2].bar(ids - width / 2, pi, width, label=r"pre-competition $\pi$")
    axes[0, 2].bar(ids + width / 2, gamma, width, label=r"case prior $\gamma$")
    axes[0, 2].set_xticks(ids)
    axes[0, 2].set_title(r"Source mass $\pi$ and competition $\gamma$")
    axes[0, 2].legend(frameon=False, fontsize=8)

    axes[0, 3].scatter(
        joint_centres[active, 0],
        joint_centres[active, 1],
        c=ids[active],
        cmap=colours,
        vmin=0,
        vmax=19,
        s=90,
        marker="D",
    )
    for group_id, centre in zip(ids[active], joint_centres[active], strict=False):
        axes[0, 3].annotate(str(group_id), centre, xytext=(4, 4), textcoords="offset points")
    axes[0, 3].set_title("Final joint group centres")

    query_dom = _dominant(query)
    axes[1, 0].scatter(
        query_xy[:, 0],
        query_xy[:, 1],
        c=query_dom,
        cmap=colours,
        vmin=0,
        vmax=19,
        s=7,
    )
    axes[1, 0].set_title("Query dominant learned group")

    degree = np.count_nonzero(query > 0.0, axis=-1)
    support_plot = axes[1, 1].scatter(
        query_xy[:, 0], query_xy[:, 1], c=degree, cmap="viridis", s=7
    )
    figure.colorbar(support_plot, ax=axes[1, 1], label="positive group degree")
    axes[1, 1].set_title("Query support degree")

    query_index = int(np.argmin(np.sum((query_xy - np.mean(query_xy, axis=0)) ** 2, axis=-1)))
    query_groups = np.flatnonzero(query[query_index] > 0.0)
    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.5, f"q[{query_index}]", ha="center", va="center", weight="bold")
    for order, group_id in enumerate(query_groups):
        y = 0.1 + 0.8 * (order + 0.5) / max(len(query_groups), 1)
        module_count = int(np.count_nonzero(module[:, group_id] > 0.0))
        environment_count = int(np.count_nonzero(environment[:, group_id] > 0.0))
        axes[1, 2].plot([0.10, 0.40], [0.5, y], color="0.55", linewidth=0.8)
        axes[1, 2].text(0.45, y, f"g{group_id}", ha="center", va="center")
        axes[1, 2].plot([0.50, 0.72], [y, y], color="0.55", linewidth=0.8)
        axes[1, 2].text(
            0.75,
            y,
            f"M:{module_count}  E:{environment_count}",
            ha="left",
            va="center",
            fontsize=8,
        )
    axes[1, 2].set_title("One query → groups → all supported source counts")

    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"Kmax = {row['kmax']}",
                f"Kcase = {row['kcase']}",
                f"kappa = {float(row['kappa']):.3f}",
                f"active modules = {row['active_module_count']}",
                f"module logical support = {float(row['module_RM_support']):.3f}",
                f"environment logical support = {float(row['environment_RE_support']):.3f}",
                f"module actual rows = {row.get('p2_module_actual_rows')}",
                f"environment actual rows = {row.get('p2_environment_actual_rows')}",
                "",
                "Learned labels are permutation-ambiguous.",
                "Logical support is not physical causality",
                "and is not actual executed work.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=9,
    )
    for axis in axes.flat[:6]:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Run 1500 mass-competitive hypergraph organization", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def render_kplan_histogram(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    kcase = np.asarray([int(row["kcase"]) for row in rows], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    bins = np.arange(int(kcase.min()) - 0.5, int(kcase.max()) + 1.5)
    axis.hist(kcase, bins=bins, color="#32648e", edgecolor="white")
    axis.set_xticks(np.arange(int(kcase.min()), int(kcase.max()) + 1))
    axis.set_xlabel(r"$K_{case}$")
    axis.set_ylabel("Cases")
    axis.set_title("Run 1500 case-level active groups")
    axis.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def render_population_figures(
    rows: Sequence[Mapping[str, Any]], figure_dir: Path
) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    kcase = np.asarray([int(row["kcase"]) for row in rows])
    modules = np.asarray([int(row["active_module_count"]) for row in rows])
    kappa = np.asarray([float(row["kappa"]) for row in rows])
    environment_radius = np.asarray([float(row["environment_radius_mean"]) for row in rows])
    environment_shuffle = np.asarray(
        [float(row["environment_radius_shuffled_mass_preserving_mean"]) for row in rows]
    )
    specifications = (
        (
            "kcase_vs_active_module_count.png",
            modules,
            kcase,
            "Active modules",
            r"$K_{case}$",
            "Case capacity versus module count",
            None,
        ),
        (
            "kappa_vs_kcase.png",
            kcase,
            kappa,
            r"$K_{case}$",
            r"$\kappa=1/\sum\gamma^2$",
            "Continuous versus discrete case complexity",
            (np.arange(1, DEFAULT_KMAX + 1), np.arange(1, DEFAULT_KMAX + 1)),
        ),
        (
            "environment_compactness_vs_shuffled.png",
            environment_shuffle,
            environment_radius,
            "Mass-preserving shuffled RMS radius",
            "Learned weighted RMS radius",
            "Environment compactness",
            None,
        ),
    )
    outputs: dict[str, str] = {}
    for filename, x, y, xlabel, ylabel, title, reference in specifications:
        figure, axis = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
        axis.scatter(x, y, s=24, alpha=0.72, color="#32648e")
        if reference is not None:
            axis.plot(reference[0], reference[1], linestyle="--", color="0.4", label="equal-weight reference")
            axis.legend(frameon=False, fontsize=8)
        if "compactness" in filename:
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            axis.plot([lo, hi], [lo, hi], linestyle="--", color="0.4", label="equal radius")
            axis.legend(frameon=False, fontsize=8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
        path = figure_dir / filename
        figure.savefig(path, dpi=170)
        plt.close(figure)
        outputs[filename.removesuffix(".png")] = str(path)
    return outputs


__all__ = [
    "DEFAULT_KMAX",
    "DEFAULT_QUERY_COUNT",
    "EXPECTED_CASE_COUNT",
    "canonicalize_case",
    "render_case_board",
    "render_kplan_histogram",
    "render_population_figures",
    "save_case_arrays",
    "summarize_population",
]
