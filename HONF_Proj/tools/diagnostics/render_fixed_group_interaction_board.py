"""Render the compact Run-1405 fixed-group interaction evidence board.

The board consumes NPZ maps written by
``run_run1405_epoch50_comparison.py`` (or an equivalent map using the
documented fixed-group debug API).  It has four panels per model/case:

* module/environment geometry and both learned group-centre sets;
* exact ``A_m``/``A_e`` incidence maps;
* the complete ``alpha_qk`` query-to-group routing map;
* actual selected ``q -> group -> source`` triples for modules and environment
  sources.

Lines and memberships are labelled learned interaction routes, never physical
causality.  The board is CPU-only; no checkpoint is loaded and no model is
executed.  Accuracy, memory, and timing remain claims of the paired
comparison manifest; only the measured prepared-decode median is annotated on
the board when it is present in the map metadata.

Examples (run from ``HONF_Proj/``)::

    python tools/diagnostics/render_fixed_group_interaction_board.py \
        --comparison diagnostics/generated/run1405_epoch50/comparison.json \
        --output-dir diagnostics/generated/run1405_epoch50/board

Or point directly at a directory containing ``1405__0273.npz`` and
``1405__0653.npz``::

    python tools/diagnostics/render_fixed_group_interaction_board.py \
        --evidence 1405=diagnostics/generated/run1405_epoch50/maps \
        --output-dir diagnostics/generated/run1405_epoch50/board
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from fixed_group_evidence import FixedGroupArrays, FixedGroupMetrics, load_evidence_npz

DEFAULT_CASE_IDS = ("0273", "0653")


def _parse_label_path(raw: str) -> tuple[str, Path]:
    value = str(raw).strip()
    if "=" not in value:
        raise TypeError(f"expected LABEL=PATH, got {raw!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path or "/" in label or "\\" in label:
        raise TypeError(f"expected a simple non-empty LABEL=PATH, got {raw!r}")
    return label, Path(path).expanduser().resolve()


def _resolve_manifest_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _find_case_map(root: Path, label: str, case_id: str) -> Path:
    if root.is_file():
        if root.suffix.lower() != ".npz":
            raise TypeError(f"evidence path must be an NPZ or directory: {root}")
        return root
    candidates = (
        root / f"{label}__{case_id}.npz",
        root / f"{label}_{case_id}.npz",
        root / f"{case_id}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no fixed-group map for label={label!r}, case={case_id!r} beneath {root}"
    )


def _paths_from_comparison(
    comparison: Path,
    *,
    case_ids: Sequence[str],
) -> dict[str, dict[str, Path]]:
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"comparison manifest must be an object: {comparison}")
    map_records = payload.get("evidence_maps")
    if not isinstance(map_records, Mapping):
        raise TypeError("comparison manifest lacks evidence_maps")
    result: dict[str, dict[str, Path]] = {}
    for label, case_values in map_records.items():
        if not isinstance(case_values, Mapping):
            continue
        rows: dict[str, Path] = {}
        for case_id in case_ids:
            raw = case_values.get(str(case_id))
            if raw is None:
                continue
            if isinstance(raw, Mapping):
                raw = raw.get("path")
            if raw is not None:
                rows[str(case_id)] = _resolve_manifest_path(raw, comparison.parent)
        if rows:
            result[str(label)] = rows
    if not result:
        raise ValueError("comparison manifest has no requested fixed-group evidence maps")
    return result


def _paths_from_explicit(
    values: Sequence[str],
    *,
    case_ids: Sequence[str],
) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for raw in values:
        label, root = _parse_label_path(raw)
        if label in result:
            raise ValueError(f"duplicate evidence label={label!r}")
        result[label] = {
            str(case_id): _find_case_map(root, label, str(case_id)) for case_id in case_ids
        }
    return result


def _dominant(values: np.ndarray) -> np.ndarray:
    support = np.asarray(values) > 0.0
    result = np.argmax(np.where(support, values, -np.inf), axis=-1)
    return np.where(support.any(axis=-1), result, -1)


def _sample_indices(count: int, maximum: int = 3500) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _set_geometry_axes(ax: Any, arrays: FixedGroupArrays) -> None:
    points = np.concatenate(
        (arrays.module_coords, arrays.env_coords, arrays.module_group_centres, arrays.environment_group_centres),
        axis=0,
    )
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1.0e-8)
    ax.set_xlim(lo[0] - 0.06 * span[0], hi[0] + 0.06 * span[0])
    ax.set_ylim(lo[1] - 0.06 * span[1], hi[1] + 0.06 * span[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.16)


def _draw_geometry(ax: Any, arrays: FixedGroupArrays, *, label: str, case_id: str) -> None:
    import matplotlib.pyplot as plt

    k = arrays.group_count
    cmap = plt.get_cmap("turbo", max(k, 2))
    module_group = _dominant(arrays.module_incidence)
    environment_group = _dominant(arrays.environment_incidence)
    env_indices = _sample_indices(len(arrays.env_coords))
    ax.scatter(
        arrays.env_coords[env_indices, 0],
        arrays.env_coords[env_indices, 1],
        c=np.where(environment_group[env_indices] >= 0, environment_group[env_indices], -1),
        cmap=cmap,
        vmin=-0.5,
        vmax=max(k - 0.5, 0.5),
        s=7,
        alpha=0.30,
        marker=".",
        label="environment sources (dominant learned group)",
    )
    active = arrays.active_module_mask
    ax.scatter(
        arrays.module_coords[active, 0],
        arrays.module_coords[active, 1],
        c=np.where(module_group[active] >= 0, module_group[active], -1),
        cmap=cmap,
        vmin=-0.5,
        vmax=max(k - 0.5, 0.5),
        s=42,
        marker="s",
        edgecolors="black",
        linewidths=0.3,
        label="active modules (dominant learned group)",
    )
    ax.scatter(
        arrays.module_group_centres[:, 0],
        arrays.module_group_centres[:, 1],
        s=100,
        marker="D",
        facecolors="none",
        edgecolors=[cmap(group) for group in range(k)],
        linewidths=1.25,
        label="$r^M_k$ group centres",
    )
    ax.scatter(
        arrays.environment_group_centres[:, 0],
        arrays.environment_group_centres[:, 1],
        s=70,
        marker="o",
        facecolors="none",
        edgecolors=[cmap(group) for group in range(k)],
        linewidths=0.9,
        label="$r^E_k$ group centres",
    )
    for group in range(k):
        centre = arrays.module_group_centres[group]
        ax.text(float(centre[0]), float(centre[1]), str(group), fontsize=6, ha="center", va="center")
    _set_geometry_axes(ax, arrays)
    ax.set_title(f"A  Geometry and group centres\n{label} · case {case_id}", fontsize=10)
    ax.legend(fontsize=5.8, loc="best", frameon=True)


def _draw_incidence(ax: Any, arrays: FixedGroupArrays, *, label: str, case_id: str) -> None:
    module = arrays.module_incidence[arrays.active_module_mask]
    environment = arrays.environment_incidence
    matrix = np.vstack((module, environment))
    vmax = max(float(np.max(matrix)) if matrix.size else 0.0, 1.0e-12)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax)
    ax.axhline(len(module) - 0.5, color="white", linewidth=0.8)
    ax.text(-0.02, 0.25, "M", transform=ax.transAxes, ha="right", va="center", color="#8ecae6", weight="bold")
    ax.text(-0.02, 0.75, "E", transform=ax.transAxes, ha="right", va="center", color="#d9d9d9", weight="bold")
    ax.set_xlabel("fixed group k")
    ax.set_ylabel("active module / environment source rows")
    ax.set_title(f"B  Learned incidence $A^M$, $A^E$\n{label} · case {case_id}", fontsize=10)
    ax.figure.colorbar(image, ax=ax, pad=0.02, label="membership")


def _draw_query_routing(ax: Any, arrays: FixedGroupArrays, metrics: FixedGroupMetrics, *, label: str, case_id: str) -> None:
    image = ax.imshow(
        arrays.query_routing,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=max(float(np.max(arrays.query_routing)), 1.0e-12),
    )
    selected = metrics.selected_query_index
    ax.axhline(selected, color="#f94144", linewidth=0.7, alpha=0.9)
    ax.set_xlabel("fixed group k")
    ax.set_ylabel("query index q")
    ax.set_title(
        f"C  Query routing $\\alpha_{{qk}}$ (red = selected q={selected})\n{label} · case {case_id}",
        fontsize=10,
    )
    ax.figure.colorbar(image, ax=ax, pad=0.02, label="learned route weight")


def _draw_triples(
    ax: Any,
    arrays: FixedGroupArrays,
    metrics: FixedGroupMetrics,
    *,
    label: str,
    case_id: str,
) -> None:
    import matplotlib.pyplot as plt

    k = arrays.group_count
    cmap = plt.get_cmap("turbo", max(k, 2))
    query_index = metrics.selected_query_index
    query = arrays.query_xy[query_index]
    ax.scatter(query[0], query[1], marker="*", s=190, color="#f94144", edgecolors="black", linewidths=0.45, label=f"selected q={query_index}")
    env_indices = _sample_indices(len(arrays.env_coords), maximum=1800)
    ax.scatter(arrays.env_coords[env_indices, 0], arrays.env_coords[env_indices, 1], s=5, color="#9aa0a6", alpha=0.11, label="environment source backdrop")
    active = arrays.active_module_mask
    ax.scatter(arrays.module_coords[active, 0], arrays.module_coords[active, 1], marker="s", s=35, color="#264653", alpha=0.30, label="module source backdrop")
    for group in range(k):
        centre = arrays.module_group_centres[group]
        ax.scatter(centre[0], centre[1], marker="D", s=75, color=cmap(group), edgecolors="black", linewidths=0.35, zorder=4)
        ax.text(float(centre[0]), float(centre[1]), str(group), fontsize=6, ha="center", va="center", zorder=5)
        alpha = float(arrays.query_routing[query_index, group])
        if alpha <= 0.0:
            continue
        for _, _, source in metrics.selected_module_triples[metrics.selected_module_triples[:, 1] == group]:
            point = arrays.module_coords[int(source)]
            ax.plot(
                [query[0], centre[0], point[0]],
                [query[1], centre[1], point[1]],
                color=cmap(group),
                alpha=min(0.82, 0.12 + 0.72 * alpha),
                linewidth=0.65,
                zorder=2,
            )
        for _, _, source in metrics.selected_environment_triples[metrics.selected_environment_triples[:, 1] == group]:
            point = arrays.env_coords[int(source)]
            ax.plot(
                [query[0], arrays.environment_group_centres[group, 0], point[0]],
                [query[1], arrays.environment_group_centres[group, 1], point[1]],
                color=cmap(group),
                alpha=min(0.42, 0.07 + 0.36 * alpha),
                linewidth=0.38,
                zorder=1,
            )
    ax.scatter(
        arrays.environment_group_centres[:, 0],
        arrays.environment_group_centres[:, 1],
        marker="o",
        s=45,
        facecolors="none",
        edgecolors=[cmap(group) for group in range(k)],
        linewidths=0.9,
        label="$r^E_k$ centres",
    )
    _set_geometry_axes(ax, arrays)
    values = metrics.values
    decode = values.get("prepared_decode_median_ms")
    decode_text = "NA" if decode is None else f"{float(decode):.3f} ms"
    ax.set_title(
        "D  Actual learned interaction triples\n"
        f"q→group→source; nM={len(metrics.selected_module_triples)}, nE={len(metrics.selected_environment_triples)}",
        fontsize=10,
    )
    ax.text(
        0.01,
        0.01,
        "P_M={P_M}  P_E={P_E}  R_M={R_M:.3g}  R_E={R_E:.3g}\n"
        "sQ={sQ:.3g}  sM={sM:.3g}  sE={sE:.3g}  decode={decode}\n"
        "learned interaction routes; not physical causality".format(
            P_M=values["P_M"],
            P_E=values["P_E"],
            R_M=float(values["R_M"] or 0.0),
            R_E=float(values["R_E"] or 0.0),
            sQ=float(values["sQ"]),
            sM=float(values["sM"]),
            sE=float(values["sE"]),
            decode=decode_text,
        ),
        transform=ax.transAxes,
        fontsize=6.2,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.0},
    )
    ax.legend(fontsize=5.7, loc="best", frameon=True)


def render_board(
    evidence_maps: Mapping[str, Mapping[str, str | Path]],
    output_dir: str | Path,
    *,
    case_ids: Sequence[str] = DEFAULT_CASE_IDS,
) -> dict[str, Any]:
    """Render all supplied model labels and requested cases."""

    cases = tuple(str(case_id) for case_id in case_ids)
    if not {"0273", "0653"}.issubset(cases):
        raise ValueError("fixed-group board requires anchor cases 0273 and 0653")
    if not evidence_maps:
        raise ValueError("at least one fixed-group evidence label is required")
    loaded: dict[str, dict[str, tuple[FixedGroupArrays, FixedGroupMetrics, dict[str, Any], Path]]] = {}
    for label, paths in evidence_maps.items():
        loaded[str(label)] = {}
        for case_id in cases:
            raw_path = paths.get(str(case_id))
            if raw_path is None:
                raise KeyError(f"label={label!r} lacks case={case_id!r}")
            path = Path(raw_path).expanduser().resolve()
            arrays, metrics, metadata = load_evidence_npz(path)
            loaded[str(label)][str(case_id)] = (arrays, metrics, metadata, path)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(loaded) * len(cases)
    figure, axes_grid = plt.subplots(rows, 4, figsize=(19.0, max(6.4, 5.0 * rows)), squeeze=False, constrained_layout=True)
    manifest_rows: list[dict[str, Any]] = []
    row_index = 0
    for label, cases_data in loaded.items():
        for case_id in cases:
            arrays, metrics, metadata, path = cases_data[case_id]
            axes = axes_grid[row_index]
            _draw_geometry(axes[0], arrays, label=label, case_id=case_id)
            _draw_incidence(axes[1], arrays, label=label, case_id=case_id)
            _draw_query_routing(axes[2], arrays, metrics, label=label, case_id=case_id)
            _draw_triples(axes[3], arrays, metrics, label=label, case_id=case_id)
            row_index += 1
            manifest_rows.append(
                {
                    "label": label,
                    "case_id": case_id,
                    "path": str(path),
                    "metrics": metrics.values,
                    "metadata": metadata,
                }
            )
    figure.suptitle(
        "Run-1405 fixed-group interaction evidence board\n"
        "incidence · centres · query routes · actual q→group→source triples "
        "(learned interaction routes, not physical causality)",
        fontsize=14,
    )
    png_path = output / "fixed_group_interaction_board.png"
    pdf_path = output / "fixed_group_interaction_board.pdf"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "run1405_fixed_group_interaction_board",
        "status": "ok",
        "case_ids": list(cases),
        "rows": manifest_rows,
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "provenance": str(output / "fixed_group_interaction_board.json"),
        },
        "limitations": [
            "Memberships and lines are learned interaction routes, not physical-causality or field-value maps.",
            "P_M/P_E/R_M/R_E/sQ/sM/sE are computed from exact positive semantic supports in the NPZ map.",
            "The triple panel uses the selected far query stored by the comparison tool; it is not every q row.",
            "Measured decode time is copied from the paired timing artifact and is not inferred from plotted geometry.",
        ],
    }
    provenance = output / "fixed_group_interaction_board.json"
    provenance.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=None, help="comparison JSON with evidence_maps")
    parser.add_argument("--evidence", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = tuple(str(value) for value in (args.case_id or DEFAULT_CASE_IDS))
    if args.comparison is not None and args.evidence:
        raise ValueError("choose --comparison or --evidence, not both")
    if args.comparison is None and not args.evidence:
        raise ValueError("one --comparison or at least one --evidence LABEL=PATH is required")
    evidence = (
        _paths_from_comparison(args.comparison.expanduser().resolve(), case_ids=cases)
        if args.comparison is not None
        else _paths_from_explicit(args.evidence, case_ids=cases)
    )
    manifest = render_board(evidence, args.output_dir, case_ids=cases)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_CASE_IDS", "build_parser", "main", "render_board"]
