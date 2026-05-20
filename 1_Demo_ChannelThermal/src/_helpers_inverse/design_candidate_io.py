from __future__ import annotations

"""Reusable candidate export and visualization helpers."""

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-inverse")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


CANDIDATE_CSV_FIELDS = [
    "method",
    "rank",
    "total_score",
    "internal_total_score",
    "fair_objective_score",
    "ranking_score",
    "ranking_score_key",
    "hard_violation_score",
    "satisfied",
    "num_satisfied",
    "num_terms",
    "num_modules",
    "repair_distance",
    "geometry_penalty",
    "hypergraph_consistency_score",
    "hypergraph_realization_score",
    "hypergraph_extra_score",
    "planned_realized_hypergraph_score",
    "prior_energy",
    "mechanism_prior_score",
    "mechanism_cluster_id",
    "objective_score",
    "best_score",
    "runtime_seconds",
    "forward_calls",
    "diversity_cluster_id",
    "source",
]


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if torch is not None and torch.is_tensor(obj):
        return to_jsonable(obj.detach().cpu().numpy())
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _value(candidate: Mapping[str, Any], field: str) -> Any:
    if field in candidate:
        return candidate[field]
    objective = candidate.get("objective_result")
    if isinstance(objective, Mapping) and field in objective:
        return objective[field]
    layout = candidate.get("layout")
    if field == "num_modules" and isinstance(layout, Mapping):
        return layout.get("count", layout.get("num_modules", ""))
    return ""


def _csv_safe(value: Any) -> Any:
    value = to_jsonable(value)
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def write_candidates_csv(candidates: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    out = _ensure_parent(path)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_CSV_FIELDS)
        writer.writeheader()
        for row in candidates:
            writer.writerow({field: _csv_safe(_value(row, field)) for field in CANDIDATE_CSV_FIELDS})


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> None:
    out = _ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)


def plot_score_vs_calls(
    histories: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    score_key: str = "best_score",
    ylabel: str = "Best total score",
) -> None:
    out = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    if isinstance(histories, Mapping):
        items = histories.items()
    else:
        items = [("method", histories)]
    for name, history in items:
        calls = [float(row.get("num_forward_calls", idx + 1)) for idx, row in enumerate(history)]
        scores = [float(row.get(score_key, row.get("best_score", row.get("total_score", np.nan)))) for row in history]
        if calls and scores:
            ax.plot(calls, scores, marker="o", linewidth=1.5, label=str(name))
    ax.set_xlabel("Forward calls")
    ax.set_ylabel(ylabel)
    ax.set_title("Score vs forward calls")
    ax.grid(True, alpha=0.25)
    if isinstance(histories, Mapping) and len(histories) > 1:
        ax.legend()
    fig.savefig(str(out), dpi=170)
    plt.close(fig)


def plot_method_comparison(summary_by_method: Mapping[str, Any], path: str | Path) -> None:
    out = _ensure_parent(path)
    names = []
    scores = []
    for name, payload in summary_by_method.items():
        names.append(str(name))
        if isinstance(payload, Mapping):
            scores.append(float(payload.get("best_score", payload.get("total_score", np.nan))))
        elif isinstance(payload, Sequence) and payload:
            first = payload[0]
            scores.append(float(first.get("total_score", np.nan)) if isinstance(first, Mapping) else float("nan"))
        else:
            scores.append(float("nan"))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * max(len(names), 1)), 4.0), constrained_layout=True)
    ax.bar(np.arange(len(names)), scores)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Best total score")
    ax.set_title("Method comparison")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(str(out), dpi=170)
    plt.close(fig)


def _layout_payload(layout: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = layout.get("layout")
    return nested if isinstance(nested, Mapping) else layout


def plot_layout_candidate(
    layout: Mapping[str, Any],
    path: str | Path,
    *,
    title: str = "",
    field: Any = None,
    domain: Mapping[str, Any] | None = None,
    target_regions: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    out = _ensure_parent(path)
    payload = _layout_payload(layout)
    centers = np.asarray(payload.get("centers", payload.get("module_centers", [])), dtype=np.float64).reshape(-1, 2) if np.asarray(payload.get("centers", payload.get("module_centers", []))).size else np.zeros((0, 2), dtype=np.float64)
    count = int(payload.get("count", centers.shape[0]))
    centers = centers[: max(count, 0)]
    domain_payload = domain or payload.get("domain") if isinstance(payload.get("domain"), Mapping) else domain
    lx = float((domain_payload or {}).get("domain_length_x", (domain_payload or {}).get("lx", 12.0)))
    ly = float((domain_payload or {}).get("domain_length_y", (domain_payload or {}).get("ly", 4.0)))
    radius = float(payload.get("module_radius", (domain_payload or {}).get("module_radius", 0.45)))
    fig, ax = plt.subplots(figsize=(8.0, 3.2), constrained_layout=True)
    if field is not None:
        arr = np.asarray(field, dtype=np.float64).squeeze()
        if arr.ndim == 3:
            arr = arr[..., -1]
        if arr.ndim == 2:
            im = ax.imshow(arr, origin="lower", extent=[0.0, lx, 0.0, ly], aspect="auto", alpha=0.75)
            fig.colorbar(im, ax=ax, shrink=0.82, label="field")
    ax.add_patch(plt.Rectangle((0.0, 0.0), lx, ly, fill=False, linewidth=1.4, edgecolor="black"))
    if target_regions:
        for region in target_regions:
            if not isinstance(region, Mapping):
                continue
            try:
                x0, x1 = [float(v) for v in region.get("x_range", [])[:2]]
                y0, y1 = [float(v) for v in region.get("y_range", [])[:2]]
            except Exception:
                continue
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=1.5, linestyle="--", edgecolor="tab:cyan")
            ax.add_patch(rect)
            label = str(region.get("name", "target"))
            ax.text(x0, y1, label, ha="left", va="bottom", fontsize=7, color="tab:cyan", bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.5, "edgecolor": "none"})
    for idx, (cx, cy) in enumerate(centers):
        ax.add_patch(Circle((float(cx), float(cy)), radius, fill=False, linewidth=1.4))
        ax.text(float(cx), float(cy), str(idx + 1), ha="center", va="center", fontsize=8)
    ax.set_xlim(-0.05 * lx, 1.05 * lx)
    ax.set_ylim(-0.08 * ly, 1.08 * ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(str(title or "Layout candidate"))
    ax.grid(True, alpha=0.2)
    fig.savefig(str(out), dpi=170)
    plt.close(fig)


def _candidate_field(candidate: Mapping[str, Any]) -> Any:
    for source in (candidate, candidate.get("forward_prediction") if isinstance(candidate.get("forward_prediction"), Mapping) else {}):
        if not isinstance(source, Mapping):
            continue
        for key in ("pred_field_grid", "temperature", "T", "field", "field_grid"):
            if key in source:
                return source[key]
    return None


def write_candidate_artifacts(candidate: Mapping[str, Any], out_dir: str | Path, prefix: str) -> Mapping[str, str]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_prefix = str(prefix).strip() or "candidate"
    json_path = root / f"{safe_prefix}.json"
    layout_path = root / f"{safe_prefix}_layout.png"
    csv_path = root / f"{safe_prefix}_summary.csv"
    write_summary_json(candidate, json_path)
    write_candidates_csv([candidate], csv_path)
    title = f"{candidate.get('method', 'candidate')} rank {candidate.get('rank', '')}".strip()
    plot_layout_candidate(candidate.get("layout", candidate), layout_path, title=title, field=_candidate_field(candidate))
    artifacts = {"json": json_path.as_posix(), "csv": csv_path.as_posix(), "layout_plot": layout_path.as_posix()}
    return artifacts
