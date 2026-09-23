"""Evidence extraction and rendering for Run 1501 sparse incidence.

The registered prototype bank has fixed capacity K=12.  This module therefore
keeps three quantities separate throughout:

* registered capacity (always 12),
* occupied source groups in the phase-local source organization, and
* Kq, the positive sparsemax support degree of each query.

It deliberately does not report an occupancy ``Kplan`` because Run 1501 has no
case-level plan.  Logical paths, unique source pairs, dense denominators, and
actual rectangular executor rows are also retained as distinct measurements.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import occupancy_adaptive_evidence as occupancy

EXPECTED_CASE_COUNT = 90
DEFAULT_KMAX = 12
HISTOGRAM_FILENAME = "kq_histogram.png"
HISTOGRAM_FIGURE_KEY = "kq_histogram"
HISTOGRAM_SUMMARY_KEY = "kq_histogram"


class SparseIncidenceEvidenceError(ValueError):
    """Raised when Run-1501 evidence cannot be audited literally."""


def _augment_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SparseIncidenceEvidenceError("Run-1501 evidence payload must be a mapping")
    result = dict(payload)
    aliases = {
        "occupancy_group_kappa": ("sparse_incidence_kappa",),
        "occupancy_group_active_mask": (
            "sparse_incidence_phase_occupied",
            "sparse_incidence_active_mask",
        ),
        "occupancy_group_module_centres": ("sparse_incidence_module_centres",),
        "occupancy_group_environment_centres": ("sparse_incidence_environment_centres",),
        "occupancy_group_joint_centres": ("sparse_incidence_joint_centres",),
        "occupancy_group_query_routing": ("sparse_incidence_query_routing",),
    }
    for target, sources in aliases.items():
        if target in result:
            continue
        for source in sources:
            if source in result:
                result[target] = result[source]
                break
    # The shared canonicalizer also validates Run-1409 plan bookkeeping.
    # Run 1501 has no packed plan, so provide a neutral registered-bank view
    # solely to satisfy that internal shape audit; every plan-derived field is
    # discarded below.  This remains correct when some source groups are empty.
    result.setdefault("occupancy_group_prototype_ids", np.arange(DEFAULT_KMAX, dtype=np.int64))
    result.setdefault("occupancy_group_packed_prototype_ids", np.arange(DEFAULT_KMAX, dtype=np.int64))
    result.setdefault("occupancy_group_packed_valid", np.ones(DEFAULT_KMAX, dtype=bool))
    return result


def _case_vector(payload: Mapping[str, Any], names: Sequence[str], length: int) -> np.ndarray | None:
    for name in names:
        if name not in payload:
            continue
        value = payload[name]
        if hasattr(value, "detach") and callable(value.detach):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.ndim >= 2 and array.shape[0] == 1:
            array = array[0]
        array = array.reshape(-1)
        if array.size == length:
            return array
    return None


def _histogram(values: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in np.asarray(values).reshape(-1))
    return {str(key): int(counts[key]) for key in sorted(counts)}


def canonicalize_case(
    payload: Any,
    *,
    case_index: int = 0,
    query_count: int | None = None,
    kmax: int | None = None,
) -> dict[str, Any]:
    """Canonicalize one P2 map payload without fabricating a Kplan."""

    augmented = _augment_payload(payload)
    record = occupancy.canonicalize_case(
        augmented,
        case_index=case_index,
        query_count=query_count,
        kmax=kmax,
    )
    row = dict(record["case"])
    maps = dict(record["maps"])
    query = np.asarray(maps["query_assignment"], dtype=np.float64)
    module = np.asarray(maps["module_assignment"], dtype=np.float64)
    environment = np.asarray(maps["environment_assignment"], dtype=np.float64)
    registered_capacity = int(row["kmax"])
    if registered_capacity != DEFAULT_KMAX:
        raise SparseIncidenceEvidenceError(
            f"registered capacity must be {DEFAULT_KMAX}, got {registered_capacity}"
        )
    query_degree = (query > 0.0).sum(axis=-1).astype(np.int64)
    module_degree = (module > 0.0).sum(axis=-1).astype(np.int64)
    environment_degree = (environment > 0.0).sum(axis=-1).astype(np.int64)
    active = np.asarray(row["active_mask"], dtype=bool)
    pi = _case_vector(augmented, ("sparse_incidence_pi",), registered_capacity)
    if pi is None:
        pi = 0.5 * (
            np.asarray(row["module_mass"], dtype=np.float64)
            + np.asarray(row["environment_mass"], dtype=np.float64)
        )
    kappa_from_pi = float(1.0 / max(float(np.sum(pi * pi)), np.finfo(np.float64).tiny))
    row.update(
        {
            "registered_capacity": registered_capacity,
            "occupied_source_groups": int(active.sum()),
            "pi": pi.tolist(),
            "pi_sum": float(pi.sum()),
            "kappa_from_pi": kappa_from_pi,
            "kappa_formula_max_abs": abs(float(row["kappa"]) - kappa_from_pi),
            "query_count": int(query.shape[0]),
            "query_degree_mean": float(np.mean(query_degree)),
            "query_degree_median": float(np.median(query_degree)),
            "query_degree_p95": float(np.percentile(query_degree, 95.0)),
            "query_degree_min": int(np.min(query_degree)),
            "query_degree_max": int(np.max(query_degree)),
            "query_degree_histogram": _histogram(query_degree),
            "module_source_degree_histogram": _histogram(module_degree),
            "environment_source_degree_histogram": _histogram(environment_degree),
            "no_empty_query_support": bool(np.all(query_degree >= 1)),
        }
    )
    # These are Run-1409 occupancy-plan fields.  Keeping them would relabel
    # phase occupancy as a plan, which is explicitly false for Run 1501.
    for key in (
        "kplan",
        "kplan_minus_active_module_count",
        "kplan_equals_active_module_count",
        "prototype_ids",
        "packed_prototype_ids",
        "packed_valid",
        "proposal_ids_reused",
    ):
        row.pop(key, None)
    maps["query_degree"] = query_degree
    maps["pi"] = pi
    return {"case": row, "maps": maps}


def _summary(values: Sequence[float]) -> dict[str, float] | None:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return None
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def summarize_population(
    rows: Sequence[Mapping[str, Any]], *, expected_cases: int = EXPECTED_CASE_COUNT
) -> dict[str, Any]:
    if expected_cases > 0 and len(rows) != expected_cases:
        raise SparseIncidenceEvidenceError(
            f"expected {expected_cases} cases, found {len(rows)}"
        )
    if not rows:
        raise SparseIncidenceEvidenceError("population is empty")
    if any(int(row["registered_capacity"]) != DEFAULT_KMAX for row in rows):
        raise SparseIncidenceEvidenceError("population contains a non-K=12 candidate")

    pooled_kq = Counter()
    total_queries = 0
    for row in rows:
        total_queries += int(row["query_count"])
        for key, count in row["query_degree_histogram"].items():
            pooled_kq[int(key)] += int(count)
    fields = (
        "occupied_source_groups",
        "M_active",
        "M_padded",
        "kappa",
        "pi_sum",
        "kappa_formula_max_abs",
        "module_source_degree_mean",
        "environment_source_degree_mean",
        "query_degree_mean",
        "query_degree_median",
        "query_degree_p95",
        "query_degree_min",
        "query_degree_max",
        "module_effective_groups_mean",
        "environment_effective_groups_mean",
        "query_effective_groups_mean",
        "module_assignment_numerical_rank",
        "environment_assignment_numerical_rank",
        "query_assignment_numerical_rank",
        "joint_source_assignment_numerical_rank",
        "module_entropy_mean",
        "environment_entropy_mean",
        "query_entropy_mean",
        "module_radius_mean",
        "environment_radius_mean",
        "module_radius_shuffled_mass_preserving_mean",
        "environment_radius_shuffled_mass_preserving_mean",
        "joint_centre_separation_mean",
        "query_to_joint_centre_mean",
        "module_RM_support",
        "environment_RE_support",
        "module_multiplicity",
        "environment_multiplicity",
        "module_logical_paths",
        "environment_logical_paths",
        "module_unique_pairs",
        "environment_unique_pairs",
        "module_dense_valid_pairs",
        "environment_dense_valid_pairs",
        "p2_module_actual_rows",
        "p2_environment_actual_rows",
        "p2_module_padded_rows",
        "p2_environment_padded_rows",
        "p2_module_geometry_rows",
        "p2_environment_geometry_rows",
        "p2_module_content_rows",
        "p2_environment_content_rows",
    )
    summary: dict[str, Any] = {
        "case_count": len(rows),
        "registered_capacity": DEFAULT_KMAX,
        "query_count_per_case": sorted({int(row["query_count"]) for row in rows}),
        "query_source_count_per_case": sorted(
            {int(row["query_source_count"]) for row in rows if row.get("query_source_count") is not None}
        ),
        "query_selection_counts": dict(
            Counter(str(row.get("query_selection") or "recorded_query_grid") for row in rows)
        ),
        "total_queries": total_queries,
        "kq_histogram": {str(key): int(pooled_kq[key]) for key in sorted(pooled_kq)},
        "all_queries_have_support": bool(all(row["no_empty_query_support"] for row in rows)),
        "note": "K=12 is registered capacity; occupied source groups and Kq are measured separately. There is no Kplan.",
    }
    for field in fields:
        summary[field] = _summary(
            [float(row[field]) for row in rows if row.get(field) is not None]
        )
    module_compactness = [
        float(row["module_radius_mean"])
        / max(float(row["module_radius_shuffled_mass_preserving_mean"]), np.finfo(np.float64).tiny)
        for row in rows
    ]
    environment_compactness = [
        float(row["environment_radius_mean"])
        / max(float(row["environment_radius_shuffled_mass_preserving_mean"]), np.finfo(np.float64).tiny)
        for row in rows
    ]
    summary["module_radius_vs_shuffle_ratio"] = _summary(module_compactness)
    summary["environment_radius_vs_shuffle_ratio"] = _summary(environment_compactness)
    actual_rows_available = any(
        row.get("p2_module_actual_rows") is not None
        or row.get("p2_environment_actual_rows") is not None
        for row in rows
    )
    summary["continuation_gate"] = {
        "registered_capacity_fixed_at_12": True,
        "nonempty_query_support": summary["all_queries_have_support"],
        "query_support_nontrivial": bool(
            summary["query_degree_mean"] is not None
            and 1.0 < float(summary["query_degree_mean"]["mean"]) < DEFAULT_KMAX
        ),
        "support_is_less_than_dense": bool(
            float(summary["module_RM_support"]["mean"]) < 1.0
            and float(summary["environment_RE_support"]["mean"]) < 1.0
        ),
        "actual_rows_available": actual_rows_available,
        "decision": "review_accuracy_trends_and_rectangular_executor_cost_before_continuing",
    }
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


def _set_equal_domain(
    axis: Any, *coordinates: np.ndarray, bounds: Sequence[float] | None = None
) -> None:
    """Use one physical coordinate frame for source/query geometry panels."""

    if bounds is not None:
        values = np.asarray(bounds, dtype=np.float64).reshape(-1)
        if values.size == 4 and np.isfinite(values).all() and values[1] >= values[0] and values[3] >= values[2]:
            lower = values[[0, 2]]
            upper = values[[1, 3]]
        else:
            lower = upper = None
    else:
        lower = upper = None
    if lower is None or upper is None:
        blocks = [np.asarray(value, dtype=np.float64).reshape(-1, 2) for value in coordinates]
        finite_blocks = [block[np.isfinite(block).all(axis=1)] for block in blocks if block.size]
        if not finite_blocks:
            return
        points = np.concatenate(finite_blocks, axis=0)
        lower = points.min(axis=0)
        upper = points.max(axis=0)
    span = np.maximum(upper - lower, np.finfo(np.float64).eps)
    # A one-dimensional synthetic/fixed panel can have a zero physical span
    # in one coordinate.  A finite pixel-scale margin keeps Matplotlib's
    # transform nonsingular while preserving the measured domain.
    margin = np.maximum(0.03 * span, 1.0e-6)
    axis.set_xlim(float(lower[0] - margin[0]), float(upper[0] + margin[0]))
    axis.set_ylim(float(lower[1] - margin[1]), float(upper[1] + margin[1]))
    axis.set_aspect("equal", adjustable="box")


def render_case_board(record: Mapping[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    maps = record["maps"]
    row = record["case"]
    module_xy = np.asarray(maps["module_coords"])
    environment_xy = np.asarray(maps["environment_coords"])
    query_xy = np.asarray(maps["query_coords"])
    module = np.asarray(maps["module_assignment"])
    environment = np.asarray(maps["environment_assignment"])
    query = np.asarray(maps["query_assignment"])
    centres = np.asarray(maps["joint_centres"])
    active = np.asarray(row["active_mask"], dtype=bool)
    domain_bounds = row.get("query_domain_bounds")
    module_present = np.asarray(
        maps.get("module_present", np.any(module > 0.0, axis=1)), dtype=bool
    ).reshape(-1)
    if module_present.size != module_xy.shape[0]:
        raise SparseIncidenceEvidenceError(
            "module_present does not align with rendered module sources"
        )
    module_xy = module_xy[module_present]
    module = module[module_present]
    module_source_ids = np.asarray(
        maps.get("module_source_ids", np.arange(module_present.size)), dtype=np.int64
    ).reshape(-1)
    if module_source_ids.size != module_present.size:
        raise SparseIncidenceEvidenceError(
            "module_source_ids does not align with module sources"
        )
    module_source_ids = module_source_ids[module_present]
    module_ids_text = ",".join(str(int(value)) for value in module_source_ids)
    kq = np.asarray(maps["query_degree"])
    cmap = plt.get_cmap("tab20", DEFAULT_KMAX)
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)

    axes[0, 0].scatter(
        module_xy[:, 0], module_xy[:, 1], c=_dominant(module), cmap=cmap,
        vmin=0, vmax=DEFAULT_KMAX - 1, marker="s", s=34, edgecolors="black", linewidths=0.25,
    )
    axes[0, 0].scatter(
        centres[active, 0], centres[active, 1], c=np.flatnonzero(active), cmap=cmap,
        vmin=0, vmax=DEFAULT_KMAX - 1, marker="+", s=90,
    )
    axes[0, 0].set_title(f"Module sources + live centres (IDs {module_ids_text})")
    _set_equal_domain(axes[0, 0], module_xy, centres, query_xy, bounds=domain_bounds)

    axes[0, 1].scatter(
        environment_xy[:, 0], environment_xy[:, 1], c=_dominant(environment),
        cmap=cmap, vmin=0, vmax=DEFAULT_KMAX - 1, s=8, alpha=0.72,
    )
    axes[0, 1].scatter(
        centres[active, 0], centres[active, 1], c=np.flatnonzero(active), cmap=cmap,
        vmin=0, vmax=DEFAULT_KMAX - 1, marker="+", s=90,
    )
    axes[0, 1].set_title("Environment sources + live centres")
    _set_equal_domain(axes[0, 1], environment_xy, centres, query_xy, bounds=domain_bounds)

    incidence = np.concatenate((module, environment), axis=0).T
    image = axes[0, 2].imshow(incidence, aspect="auto", interpolation="nearest", cmap="viridis")
    figure.colorbar(image, ax=axes[0, 2], label="incidence weight")
    axes[0, 2].axvline(module.shape[0] - 0.5, color="white", linewidth=1.0)
    axes[0, 2].set_xlabel("module sources | environment sources")
    axes[0, 2].set_ylabel("registered prototype ID")
    axes[0, 2].set_title("Phase-local Aᴹ / Aᴱ (M active | E)")

    axes[1, 0].scatter(
        query_xy[:, 0], query_xy[:, 1], c=_dominant(query), cmap=cmap,
        vmin=0, vmax=DEFAULT_KMAX - 1, s=5, alpha=0.7,
    )
    axes[1, 0].set_title("Dominant query prototype (not Kq/confidence)")
    _set_equal_domain(axes[1, 0], query_xy, centres, bounds=domain_bounds)

    degree_image = axes[1, 1].scatter(
        query_xy[:, 0], query_xy[:, 1], c=kq, cmap="magma", vmin=1,
        vmax=max(int(kq.max()), 1), s=5,
    )
    figure.colorbar(degree_image, ax=axes[1, 1], label="Kq")
    axes[1, 1].set_title("Exact sparsemax support degree Kq")
    _set_equal_domain(axes[1, 1], query_xy, centres, bounds=domain_bounds)

    selected = int(np.argmax(kq))
    query_point = query_xy[selected]
    axes[1, 2].scatter(*query_point, marker="*", s=135, color="#d55e00", label=f"q={selected}, Kq={kq[selected]}")
    for group in np.flatnonzero(query[selected] > 0.0):
        axes[1, 2].plot(
            [query_point[0], centres[group, 0]], [query_point[1], centres[group, 1]],
            color=cmap(int(group)), linewidth=1.2,
        )
        axes[1, 2].scatter(*centres[group], marker="+", color=cmap(int(group)), s=85)
        module_sources = np.flatnonzero(module[:, group] > 0.0)
        environment_sources = np.flatnonzero(environment[:, group] > 0.0)
        for source in module_sources[:2]:
            axes[1, 2].plot(
                [centres[group, 0], module_xy[source, 0]], [centres[group, 1], module_xy[source, 1]],
                color="#0072B2", linewidth=0.5, alpha=0.45,
            )
        for source in environment_sources[:2]:
            axes[1, 2].plot(
                [centres[group, 0], environment_xy[source, 0]], [centres[group, 1], environment_xy[source, 1]],
                color="#009E73", linewidth=0.5, alpha=0.35,
            )
    axes[1, 2].legend(frameon=False, fontsize=8)
    axes[1, 2].set_title("Exact query → group → source support")
    for axis in axes.flat:
        if axis is not axes[0, 2]:
            axis.set_xlabel("x")
            axis.set_ylabel("y")
    _set_equal_domain(axes[1, 2], query_xy, centres, module_xy, environment_xy, bounds=domain_bounds)
    module_rows = row.get("p2_module_actual_rows")
    environment_rows = row.get("p2_environment_actual_rows")
    row_text = "n/a/n/a" if module_rows is None or environment_rows is None else f"{float(module_rows):.0f}/{float(environment_rows):.0f}"
    figure.suptitle(
        "Run 1501 phase-local sparse incidence\n"
        f"Q={query.shape[0]} · M={int(module_xy.shape[0])}/{int(row.get('M_padded', module_xy.shape[0]))} · E={environment.shape[0]} · "
        f"registered K=12 · occupied={row['occupied_source_groups']} · "
        f"mean Kq={row['query_degree_mean']:.2f} · κ={row['kappa']:.2f} · "
        f"RM/RE={row['module_RM_support']:.3f}/{row['environment_RE_support']:.3f} · "
        f"P2 actual rows M/E={row_text}\n"
        "Learned organization, not physical causality; logical support is distinct from rectangular execution",
        fontsize=10.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    if path.suffix.lower() == ".png":
        figure.savefig(path.with_suffix(".pdf"), format="pdf", facecolor="white")
    plt.close(figure)


def render_kplan_histogram(
    rows: Sequence[Mapping[str, Any]], path: Path, *, run_label: str = "Run 1501"
) -> None:
    """Compatibility entry point that renders pooled Kq, never Kplan.

    ``run_label`` keeps the historical entry point reusable by later
    sparse-incidence-compatible candidates without changing its default
    Run-1501 presentation.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    counts = Counter()
    for row in rows:
        for key, count in row["query_degree_histogram"].items():
            counts[int(key)] += int(count)
    x = np.asarray(sorted(counts), dtype=np.int64)
    y = np.asarray([counts[int(value)] for value in x], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    bars = axis.bar(x, y, color="#6A51A3")
    axis.set_xlabel("query sparse support Kq")
    axis.set_ylabel("queries across explicit population")
    axis.set_title(f"{run_label} query support; registered capacity K=12")
    for bar, count in zip(bars, y, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2.0, count, f"{int(count):,}", ha="center", va="bottom", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _query_degree_samples(row: Mapping[str, Any]) -> np.ndarray:
    """Expand one case's exact Kq histogram for quantile plotting."""

    histogram = row.get("query_degree_histogram", {})
    values: list[int] = []
    if isinstance(histogram, Mapping):
        for key, count in histogram.items():
            degree = int(key)
            values.extend([degree] * int(count))
    if not values:
        # This fallback keeps the renderer useful for hand-authored review
        # tables while never fabricating a distribution for normal evidence.
        mean = float(row.get("query_degree_mean", 0.0))
        return np.asarray([mean], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def render_kq_case_spread(
    rows: Sequence[Mapping[str, Any]], path: Path, *, run_label: str = "Run 1501"
) -> None:
    """Render exact per-case Kq intervals and histograms across the population.

    Case IDs are sorted by their measured mean Kq.  The figure is deliberately
    descriptive: it does not assign case IDs to physical regimes or infer a
    causal role for query support.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get("query_degree_mean", float("nan"))),
            str(row.get("case_id", "")),
        ),
    )
    if not ordered:
        return
    samples = [_query_degree_samples(row) for row in ordered]
    means = np.asarray([float(np.mean(sample)) for sample in samples], dtype=np.float64)
    q10 = np.asarray([float(np.percentile(sample, 10.0)) for sample in samples], dtype=np.float64)
    q90 = np.asarray([float(np.percentile(sample, 90.0)) for sample in samples], dtype=np.float64)
    fractions = np.zeros((len(ordered), DEFAULT_KMAX), dtype=np.float64)
    for index, sample in enumerate(samples):
        for degree in range(1, DEFAULT_KMAX + 1):
            fractions[index, degree - 1] = float(np.mean(sample == degree))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 7.0),
        gridspec_kw={"width_ratios": (1.35, 1.0)},
        constrained_layout=True,
    )
    figure.suptitle(
        f"{run_label} query support Kq across explicit cases\n"
        "Intervals are per-case query distributions; case IDs are sorted by mean Kq",
        fontsize=14,
        fontweight="bold",
    )
    x = np.arange(len(ordered), dtype=np.float64)
    axes[0].vlines(x, q10, q90, color="#4C78A8", linewidth=1.8, alpha=0.85)
    axes[0].scatter(x, means, s=24, color="#D2691E", zorder=3, label="case mean")
    axes[0].axhline(float(np.mean(means)), color="#555555", linestyle="--", linewidth=1.0, label="mean of case means")
    tick_step = max(1, len(ordered) // 12)
    tick_indices = np.arange(0, len(ordered), tick_step)
    if tick_indices[-1] != len(ordered) - 1:
        tick_indices = np.append(tick_indices, len(ordered) - 1)
    axes[0].set_xticks(tick_indices, [str(ordered[int(index)].get("case_id", "?")) for index in tick_indices])
    axes[0].tick_params(axis="x", rotation=55)
    axes[0].set_xlabel("case ID (sorted by mean Kq)")
    axes[0].set_ylabel("query support Kq; vertical range = q10–q90")
    axes[0].set_title("Across-case support spread")
    axes[0].legend(frameon=False, fontsize=8)

    image = axes[1].imshow(
        fractions,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0.5, DEFAULT_KMAX + 0.5, -0.5, len(ordered) - 0.5),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    figure.colorbar(image, ax=axes[1], label="fraction of queries in case")
    axes[1].set_xlabel("query support degree Kq")
    axes[1].set_ylabel("cases in same order as left panel")
    axes[1].set_xticks(np.arange(1, DEFAULT_KMAX + 1))
    axes[1].set_yticks(tick_indices, [str(ordered[int(index)].get("case_id", "?")) for index in tick_indices])
    axes[1].set_title("Per-case Kq histogram fractions")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _coordinate_region_masks(query_xy: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return descriptive normalized-coordinate regions for one case."""

    x_values = np.asarray(query_xy[:, 0], dtype=np.float64)
    y_values = np.asarray(query_xy[:, 1], dtype=np.float64)
    x_span = max(float(np.ptp(x_values)), np.finfo(np.float64).tiny)
    y_span = max(float(np.ptp(y_values)), np.finfo(np.float64).tiny)
    x_norm = (x_values - float(np.min(x_values))) / x_span
    y_norm = (y_values - float(np.min(y_values))) / y_span
    inlet = x_norm < 0.2
    outlet = x_norm > 0.8
    wall = (~inlet) & (~outlet) & ((y_norm < 0.2) | (y_norm > 0.8))
    interior = ~(inlet | outlet | wall)
    return inlet, wall, interior, outlet


def render_kq_coordinate_regions(
    rows: Sequence[Mapping[str, Any]], array_dir: Path, path: Path
) -> None:
    """Render per-case Kq means in descriptive normalized-coordinate bands."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    region_labels = ("inlet\nx/L < 0.2", "wall band", "interior", "outlet\nx/L > 0.8")
    records: list[tuple[str, float, np.ndarray]] = []
    for row in rows:
        case_id = str(row.get("case_id", ""))
        array_path = array_dir / f"{case_id}.npz"
        if not array_path.is_file():
            continue
        with np.load(array_path) as data:
            query_xy = np.asarray(data["query_coords"], dtype=np.float64)
            query_degree = np.asarray(data["query_degree"], dtype=np.float64)
        values = np.full(4, np.nan, dtype=np.float64)
        for index, mask in enumerate(_coordinate_region_masks(query_xy)):
            if np.any(mask):
                values[index] = float(np.mean(query_degree[mask]))
        records.append((case_id, float(np.mean(query_degree)), values))
    if not records:
        return
    records.sort(key=lambda item: (item[1], item[0]))
    case_ids = [item[0] for item in records]
    values = np.vstack([item[2] for item in records])
    figure, axis = plt.subplots(figsize=(8.8, 7.0), constrained_layout=True)
    image = axis.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    figure.colorbar(image, ax=axis, label="mean Kq in coordinate band")
    axis.set_xticks(np.arange(4), region_labels)
    tick_step = max(1, len(case_ids) // 12)
    tick_indices = np.arange(0, len(case_ids), tick_step)
    if tick_indices[-1] != len(case_ids) - 1:
        tick_indices = np.append(tick_indices, len(case_ids) - 1)
    axis.set_yticks(tick_indices, [case_ids[int(index)] for index in tick_indices])
    axis.set_xlabel("descriptive normalized-coordinate region")
    axis.set_ylabel("case ID (sorted by overall mean Kq)")
    axis.set_title(
        "Run 1501 Kq by case and coordinate region\n"
        "within-case normalized geometry; descriptive organization, not causality"
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def render_representative_hypergraph_overview(
    rows: Sequence[Mapping[str, Any]],
    array_dir: Path,
    path: Path,
    *,
    case_ids: Sequence[str] = ("0273", "0653"),
    run_label: str = "Run 1501",
) -> None:
    """Render compact source/query organization boards for representative cases.

    The overview keeps source geometry, query-local support, query-to-group
    weights, and source incidence visible in one figure.  It intentionally
    labels group IDs as learned/permutation-ambiguous organizations.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    row_by_case = {str(row.get("case_id", "")): row for row in rows}
    records: list[tuple[str, Mapping[str, Any], dict[str, np.ndarray]]] = []
    for case_id in case_ids:
        array_path = array_dir / f"{case_id}.npz"
        row = row_by_case.get(str(case_id))
        if row is None or not array_path.is_file():
            continue
        with np.load(array_path) as data:
            maps = {key: np.asarray(data[key]) for key in data.files}
        records.append((str(case_id), row, maps))
    if not records:
        return
    max_kq = max(int(np.max(maps["query_degree"])) for _, _, maps in records)
    cmap = plt.get_cmap("tab20", DEFAULT_KMAX)
    figure, axes = plt.subplots(
        len(records),
        4,
        figsize=(18.0, 4.7 * len(records)),
        squeeze=False,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.45, 1.45)},
        constrained_layout=True,
    )
    figure.suptitle(
        f"{run_label} representative hypergraph organizations\n"
        "Learned group labels are permutation-ambiguous; support maps do not imply physical causality",
        fontsize=14,
        fontweight="bold",
    )
    for row_index, (case_id, row, maps) in enumerate(records):
        module_xy = np.asarray(maps["module_coords"], dtype=np.float64)
        environment_xy = np.asarray(maps["environment_coords"], dtype=np.float64)
        query_xy = np.asarray(maps["query_coords"], dtype=np.float64)
        module = np.asarray(maps["module_assignment"], dtype=np.float64)
        environment = np.asarray(maps["environment_assignment"], dtype=np.float64)
        query = np.asarray(maps["query_assignment"], dtype=np.float64)
        centres = np.asarray(maps["joint_centres"], dtype=np.float64)
        kq = np.asarray(maps["query_degree"], dtype=np.float64)
        module_present = np.asarray(
            maps.get("module_present", np.any(module > 0.0, axis=1)), dtype=bool
        ).reshape(-1)
        if module_present.size != module_xy.shape[0]:
            raise SparseIncidenceEvidenceError(
                "module_present does not align with representative module sources"
            )
        module_xy = module_xy[module_present]
        module = module[module_present]
        module_source_ids = np.asarray(
            maps.get("module_source_ids", np.arange(module_present.size)), dtype=np.int64
        ).reshape(-1)
        if module_source_ids.size != module_present.size:
            raise SparseIncidenceEvidenceError(
                "module_source_ids does not align with representative module sources"
            )
        module_source_ids = module_source_ids[module_present]
        module_ids_text = ",".join(str(int(value)) for value in module_source_ids)
        query_count = int(query.shape[0])
        source_query_count = row.get("query_source_count")
        selection = str(row.get("query_selection") or "recorded_query_grid")
        domain_bounds = row.get("query_domain_bounds")
        if source_query_count is None:
            query_scope = f"recorded Q={query_count}; original-grid provenance unavailable"
        elif query_count == int(source_query_count) and selection == "full_original_grid":
            query_scope = f"full original Q={query_count} grid"
        else:
            query_scope = f"Q={query_count} subset of original Q={int(source_query_count)} grid"
        source_axis, query_axis, routing_axis, incidence_axis = axes[row_index]
        module_groups = _dominant(module)
        environment_groups = _dominant(environment)
        source_axis.scatter(
            environment_xy[:, 0],
            environment_xy[:, 1],
            c=[cmap(int(index)) if index >= 0 else "#bdbdbd" for index in environment_groups],
            s=8,
            alpha=0.55,
            label="environment sources",
        )
        source_axis.scatter(
            module_xy[:, 0],
            module_xy[:, 1],
            c=[cmap(int(index)) if index >= 0 else "#bdbdbd" for index in module_groups],
            marker="s",
            s=38,
            edgecolors="black",
            linewidths=0.25,
            label="module sources",
        )
        source_axis.scatter(
            centres[:, 0], centres[:, 1], marker="+", c=np.arange(DEFAULT_KMAX), cmap=cmap, s=105,
            linewidths=1.3, label="joint centres",
        )
        source_axis.set_title(
            f"Case {case_id} · source geometry\n"
            f"active M={module.shape[0]}/{int(row.get('M_padded', module.shape[0]))}; IDs {module_ids_text}\n"
            f"E={environment.shape[0]}; RM/RE={float(row['module_RM_support']):.3f}/{float(row['environment_RE_support']):.3f}",
            fontsize=9,
        )
        source_axis.legend(frameon=False, fontsize=7, loc="best")
        source_axis.set_aspect("equal", adjustable="datalim")
        _set_equal_domain(source_axis, module_xy, environment_xy, centres, query_xy, bounds=domain_bounds)

        query_image = query_axis.scatter(
            query_xy[:, 0], query_xy[:, 1], c=kq, cmap="magma", vmin=1, vmax=max(max_kq, 1), s=5,
        )
        query_axis.scatter(centres[:, 0], centres[:, 1], marker="+", c="white", s=55, linewidths=0.8)
        query_axis.set_title(
            f"Query support Kq\nmean={float(row['query_degree_mean']):.2f}, "
            f"q10–q90={np.percentile(kq, 10):.1f}–{np.percentile(kq, 90):.1f}\n{query_scope}"
        )
        query_axis.set_aspect("equal", adjustable="datalim")
        _set_equal_domain(query_axis, query_xy, centres, bounds=domain_bounds)
        figure.colorbar(query_image, ax=query_axis, label="Kq", fraction=0.046, pad=0.03)

        dominant = _dominant(query)
        order = np.lexsort((query_xy[:, 1], query_xy[:, 0], dominant))
        routing_image = routing_axis.imshow(
            query[order].T,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        routing_axis.set_title("Query → group weights (Q×K, sorted by dominant group)")
        routing_axis.set_xlabel("query index after descriptive sort")
        routing_axis.set_ylabel("learned group ID")
        routing_axis.set_yticks(np.arange(DEFAULT_KMAX))
        figure.colorbar(routing_image, ax=routing_axis, label="query assignment", fraction=0.046, pad=0.03)

        incidence = np.concatenate((module, environment), axis=0).T
        incidence_image = incidence_axis.imshow(
            incidence,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        incidence_axis.axvline(module.shape[0] - 0.5, color="white", linewidth=1.0)
        incidence_axis.set_title("Source incidence Aᴹ | Aᴱ (K×sources; M active | E)")
        incidence_axis.set_xlabel("module sources | environment sources")
        incidence_axis.set_ylabel("learned group ID")
        incidence_axis.set_yticks(np.arange(DEFAULT_KMAX))
        figure.colorbar(incidence_image, ax=incidence_axis, label="incidence weight", fraction=0.046, pad=0.03)

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    if path.suffix.lower() == ".png":
        figure.savefig(path.with_suffix(".pdf"), format="pdf", facecolor="white")
    plt.close(figure)


def render_population_figures(
    rows: Sequence[Mapping[str, Any]], figure_dir: Path, *, run_label: str = "Run 1501"
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    array_dir = figure_dir.parent / "arrays"
    array_paths = sorted(array_dir.glob("*.npz"))
    if array_paths:
        query_xy_blocks: list[np.ndarray] = []
        query_degree_blocks: list[np.ndarray] = []
        for array_path in array_paths:
            with np.load(array_path) as data:
                query_xy_blocks.append(np.asarray(data["query_coords"], dtype=np.float64))
                query_degree_blocks.append(np.asarray(data["query_degree"], dtype=np.float64))
        query_xy = np.concatenate(query_xy_blocks, axis=0)
        query_degree = np.concatenate(query_degree_blocks, axis=0)
        x_span = max(float(np.ptp(query_xy[:, 0])), np.finfo(np.float64).tiny)
        y_span = max(float(np.ptp(query_xy[:, 1])), np.finfo(np.float64).tiny)
        x_norm = (query_xy[:, 0] - float(np.min(query_xy[:, 0]))) / x_span
        y_norm = (query_xy[:, 1] - float(np.min(query_xy[:, 1]))) / y_span
        x_bin = np.clip((x_norm * 24).astype(np.int64), 0, 23)
        y_bin = np.clip((y_norm * 12).astype(np.int64), 0, 11)
        sums = np.zeros((12, 24), dtype=np.float64)
        counts = np.zeros((12, 24), dtype=np.int64)
        np.add.at(sums, (y_bin, x_bin), query_degree)
        np.add.at(counts, (y_bin, x_bin), 1)
        binned = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        inlet = x_norm < 0.2
        outlet = x_norm > 0.8
        wall = (~inlet) & (~outlet) & ((y_norm < 0.2) | (y_norm > 0.8))
        interior = ~(inlet | outlet | wall)
        region_masks = (inlet, wall, interior, outlet)
        region_labels = ("inlet x<0.2", "wall band", "interior", "outlet x>0.8")
        location_path = figure_dir / "kq_vs_query_location.png"
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
        image = axes[0].imshow(
            binned,
            origin="lower",
            aspect="auto",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="magma",
        )
        figure.colorbar(image, ax=axes[0], label="mean Kq")
        axes[0].set_xlabel("normalized x")
        axes[0].set_ylabel("normalized y")
        axes[0].set_title("Population mean Kq by query location (white = unsampled bin)")
        axes[1].boxplot(
            [query_degree[mask] for mask in region_masks],
            tick_labels=region_labels,
            showfliers=False,
        )
        axes[1].set_ylabel("Kq")
        axes[1].tick_params(axis="x", rotation=18)
        axes[1].set_title("Explicit coordinate-region classes")
        figure.savefig(location_path, dpi=170)
        plt.close(figure)
        paths["kq_vs_query_location"] = str(location_path)

    case_spread_path = figure_dir / "kq_case_spread.png"
    render_kq_case_spread(rows, case_spread_path, run_label=run_label)
    if case_spread_path.is_file():
        paths["kq_case_spread"] = str(case_spread_path)

    coordinate_path = figure_dir / "kq_case_coordinate_regions.png"
    render_kq_coordinate_regions(rows, array_dir, coordinate_path)
    if coordinate_path.is_file():
        paths["kq_case_coordinate_regions"] = str(coordinate_path)

    representative_path = figure_dir / "representative_hypergraph_organization__0273__0653.png"
    render_representative_hypergraph_overview(
        rows, array_dir, representative_path, run_label=run_label
    )
    if representative_path.is_file():
        paths["representative_hypergraph_organization"] = str(representative_path)

    compactness_path = figure_dir / "source_compactness_vs_shuffle.png"
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    for axis, source, color in zip(axes, ("module", "environment"), ("#0072B2", "#009E73"), strict=True):
        observed = np.asarray([float(row[f"{source}_radius_mean"]) for row in rows])
        shuffled = np.asarray([float(row[f"{source}_radius_shuffled_mass_preserving_mean"]) for row in rows])
        axis.scatter(shuffled, observed, s=22, alpha=0.7, color=color)
        limit = max(float(observed.max()), float(shuffled.max()), 1.0e-12)
        axis.plot([0, limit], [0, limit], linestyle="--", color="#555555", linewidth=1)
        axis.set_xlabel("mass-preserving shuffled radius")
        axis.set_ylabel("observed radius")
        axis.set_title(f"{source.capitalize()} source compactness")
    figure.savefig(compactness_path, dpi=170)
    plt.close(figure)
    paths["source_compactness_vs_shuffle"] = str(compactness_path)

    support_path = figure_dir / "support_and_rectangular_work.png"
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    labels = ["module RM", "environment RE"]
    means = [
        np.mean([float(row["module_RM_support"]) for row in rows]),
        np.mean([float(row["environment_RE_support"]) for row in rows]),
    ]
    axes[0].bar(labels, means, color=("#0072B2", "#009E73"))
    axes[0].axhline(1.0, linestyle="--", color="#555555", linewidth=1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("unique support / dense valid pairs")
    axes[0].set_title("Logical support ratios")
    row_fields = ("p2_module_actual_rows", "p2_environment_actual_rows")
    actual = [
        np.mean([float(row[field]) for row in rows if row.get(field) is not None])
        if any(row.get(field) is not None for row in rows)
        else np.nan
        for field in row_fields
    ]
    dense = [
        np.mean([float(row["module_dense_valid_pairs"]) for row in rows]),
        np.mean([float(row["environment_dense_valid_pairs"]) for row in rows]),
    ]
    x = np.arange(2)
    axes[1].bar(x - 0.18, dense, 0.36, label="dense valid pairs", color="#A6CEE3")
    axes[1].bar(x + 0.18, actual, 0.36, label="actual rectangular rows", color="#FB9A99")
    axes[1].set_xticks(x, ("module", "environment"))
    axes[1].set_ylabel("rows per case at recorded Q")
    axes[1].set_title("Support does not imply executor sparsity")
    axes[1].legend(frameon=False)
    figure.savefig(support_path, dpi=170)
    plt.close(figure)
    paths["support_and_rectangular_work"] = str(support_path)

    state_path = figure_dir / "continuous_state_summary.png"
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].hist([float(row["kappa"]) for row in rows], bins=14, color="#E69F00")
    axes[0].set_xlabel("κ = 1 / Σπ²")
    axes[0].set_ylabel("cases")
    axes[0].set_title("Continuous effective source scale")
    axes[1].scatter(
        [float(row["query_effective_groups_mean"]) for row in rows],
        [float(row["query_entropy_mean"]) for row in rows],
        c=[float(row["query_degree_mean"]) for row in rows],
        cmap="viridis", s=28,
    )
    axes[1].set_xlabel("mean effective query groups")
    axes[1].set_ylabel("mean normalized query entropy")
    axes[1].set_title("Query support shape; color = mean Kq")
    figure.savefig(state_path, dpi=170)
    plt.close(figure)
    paths["continuous_state_summary"] = str(state_path)
    return paths


__all__ = [
    "canonicalize_case",
    "render_case_board",
    "render_kplan_histogram",
    "render_kq_case_spread",
    "render_kq_coordinate_regions",
    "render_population_figures",
    "render_representative_hypergraph_overview",
    "save_case_arrays",
    "summarize_population",
]
