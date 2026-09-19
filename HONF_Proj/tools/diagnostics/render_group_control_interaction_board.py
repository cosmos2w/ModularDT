"""Render a CPU-only Run-1406 group-control interaction board.

The board makes the Run-1406 work distinction visible.  Dashed paths are
logical ``q -> group -> source`` support used by cheap control contractions;
solid paths are unique ``q -> source`` candidates for one fine interaction,
annotated with their overlap ``rho``.  The board never draws every logical
path as a separate fine execution and labels all routes as learned
interactions, not physical causality.

No checkpoint, model, CUDA device, or training process is loaded.  The input
NPZ files are produced by ``run_group_control_comparison.py``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import group_control_evidence as group_control
import numpy as np

DEFAULT_CASE_IDS = ("0273", "0653")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parse_label_path(raw: str) -> tuple[str, Path]:
    value = str(raw).strip()
    if "=" not in value:
        raise ValueError(f"expected LABEL=PATH, got {raw!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path or "/" in label or "\\" in label:
        raise ValueError(f"expected a simple LABEL=PATH, got {raw!r}")
    return label, Path(path).expanduser().resolve()


def _resolve_manifest_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _find_case_map(root: Path, label: str, case_id: str) -> Path:
    if root.is_file():
        if root.suffix.lower() != ".npz":
            raise TypeError(f"evidence path must be an NPZ or directory: {root}")
        return root
    candidates = [
        root / f"{label}__{case_id}.npz",
        root / f"{label}_{case_id}.npz",
        root / f"{case_id}.npz",
        root / "forward" / f"{label}__{case_id}.npz",
    ]
    candidates.extend(sorted(root.rglob(f"{label}__{case_id}.npz")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"no group-control map for label={label!r}, case={case_id!r} beneath {root}")


def _paths_from_comparison(comparison: Path, *, case_ids: Sequence[str]) -> dict[str, dict[str, Path]]:
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"comparison manifest must be an object: {comparison}")
    records = payload.get("evidence_maps")
    if not isinstance(records, Mapping):
        raise TypeError("comparison manifest lacks evidence_maps")
    result: dict[str, dict[str, Path]] = {}
    for label, case_values in records.items():
        if not isinstance(case_values, Mapping):
            continue
        rows: dict[str, Path] = {}
        for case_id in case_ids:
            raw = case_values.get(str(case_id))
            if isinstance(raw, Mapping):
                raw = raw.get("path")
            if raw is not None:
                rows[str(case_id)] = _resolve_manifest_path(raw, comparison.parent)
        if rows:
            result[str(label)] = rows
    if not result:
        raise ValueError("comparison manifest has no requested group-control maps")
    return result


def _paths_from_explicit(values: Sequence[str], *, case_ids: Sequence[str]) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for raw in values:
        label, root = _parse_label_path(raw)
        if label in result:
            raise ValueError(f"duplicate evidence label={label!r}")
        result[label] = {str(case_id): _find_case_map(root, label, str(case_id)) for case_id in case_ids}
    return result


def _dominant(values: np.ndarray) -> np.ndarray:
    support = np.asarray(values) > 0.0
    return np.where(support.any(axis=-1), np.argmax(np.where(support, values, -np.inf), axis=-1), -1)


def _sample_indices(count: int, maximum: int = 2500) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _finite_points(*arrays: np.ndarray) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1, 2) for value in arrays if np.asarray(value).size]
    values = [value[np.isfinite(value).all(axis=1)] for value in values]
    return np.concatenate(values, axis=0) if values else np.empty((0, 2), dtype=np.float64)


def _set_geometry_axes(ax: Any, arrays: group_control.GroupControlArrays) -> None:
    points = _finite_points(arrays.module_coords, arrays.env_coords, arrays.module_group_centres, arrays.environment_group_centres)
    if not len(points):
        points = np.zeros((1, 2), dtype=np.float64)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1.0e-8)
    ax.set_xlim(lo[0] - 0.08 * span[0], hi[0] + 0.08 * span[0])
    ax.set_ylim(lo[1] - 0.08 * span[1], hi[1] + 0.08 * span[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.16)


def _draw_geometry(ax: Any, arrays: group_control.GroupControlArrays, *, label: str, case_id: str) -> None:
    import matplotlib.pyplot as plt

    k = arrays.group_count
    cmap = plt.get_cmap("turbo", max(k, 2))
    env_group = _dominant(arrays.environment_incidence)
    module_group = _dominant(arrays.module_incidence)
    env_indices = _sample_indices(len(arrays.env_coords))
    ax.scatter(arrays.env_coords[env_indices, 0], arrays.env_coords[env_indices, 1], c=env_group[env_indices], cmap=cmap, vmin=-0.5, vmax=max(k - 0.5, 0.5), s=7, alpha=0.25, marker=".", label="environment sources")
    active = arrays.active_module_mask
    ax.scatter(arrays.module_coords[active, 0], arrays.module_coords[active, 1], c=module_group[active], cmap=cmap, vmin=-0.5, vmax=max(k - 0.5, 0.5), s=38, marker="s", edgecolors="black", linewidths=0.3, label="active modules")
    for group in range(k):
        module_centre = arrays.module_group_centres[group]
        environment_centre = arrays.environment_group_centres[group]
        if np.isfinite(module_centre).all():
            ax.scatter(module_centre[0], module_centre[1], marker="D", s=85, facecolors="none", edgecolors=[cmap(group)], linewidths=1.2)
            ax.text(float(module_centre[0]), float(module_centre[1]), f"M{group}", fontsize=5.5, ha="center", va="center")
        if np.isfinite(environment_centre).all():
            ax.scatter(environment_centre[0], environment_centre[1], marker="o", s=62, facecolors="none", edgecolors=[cmap(group)], linewidths=1.0)
            ax.text(float(environment_centre[0]), float(environment_centre[1]), f"E{group}", fontsize=5.5, ha="center", va="center")
    unavailable = int(np.count_nonzero(~np.isfinite(arrays.module_group_centres).all(axis=1))) + int(np.count_nonzero(~np.isfinite(arrays.environment_group_centres).all(axis=1)))
    _set_geometry_axes(ax, arrays)
    ax.set_title(f"A  Incidence, memberships and centres\n{label} · case {case_id}", fontsize=9)
    ax.text(0.01, 0.01, f"empty/unavailable centres: {unavailable}\ncentres are detached learned summaries", transform=ax.transAxes, fontsize=6, va="bottom", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"})
    ax.legend(fontsize=5.5, loc="best", frameon=True)


def _draw_incidence(ax: Any, arrays: group_control.GroupControlArrays, *, label: str, case_id: str) -> None:
    module = arrays.module_incidence[arrays.active_module_mask]
    environment = arrays.environment_incidence
    matrix = np.vstack((module, environment))
    vmax = max(float(np.max(matrix)) if matrix.size else 0.0, 1.0e-12)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax)
    ax.axhline(len(module) - 0.5, color="white", linewidth=0.8)
    ax.set_xlabel("group k")
    ax.set_ylabel("active module / environment row")
    ax.set_title(f"B  A_m / A_e membership\n{label} · case {case_id}", fontsize=9)
    ax.figure.colorbar(image, ax=ax, pad=0.02, label="membership")


def _draw_query(ax: Any, arrays: group_control.GroupControlArrays, metrics: group_control.GroupControlMetrics, *, label: str, case_id: str) -> None:
    image = ax.imshow(arrays.query_routing, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=max(float(np.max(arrays.query_routing)), 1.0e-12))
    ax.axhline(metrics.selected_query_index, color="#f94144", linewidth=0.7)
    ax.set_xlabel("group k")
    ax.set_ylabel("query q")
    ax.set_title(f"C  Query routing α_qk; red q={metrics.selected_query_index}\n{label} · case {case_id}", fontsize=9)
    ax.figure.colorbar(image, ax=ax, pad=0.02, label="learned control route")


def _draw_routes(ax: Any, arrays: group_control.GroupControlArrays, metrics: group_control.GroupControlMetrics, *, label: str, case_id: str) -> None:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("turbo", max(arrays.group_count, 2))
    query = arrays.query_xy[metrics.selected_query_index]
    ax.scatter(query[0], query[1], marker="*", s=180, color="#f94144", edgecolors="black", linewidths=0.4, zorder=7, label="selected query")
    env_indices = _sample_indices(len(arrays.env_coords), maximum=1400)
    ax.scatter(arrays.env_coords[env_indices, 0], arrays.env_coords[env_indices, 1], s=5, color="#9aa0a6", alpha=0.12, label="environment backdrop")
    active = arrays.active_module_mask
    ax.scatter(arrays.module_coords[active, 0], arrays.module_coords[active, 1], marker="s", s=32, color="#264653", alpha=0.25, label="module backdrop")

    def draw_kind(kind: str, triples: np.ndarray, unique: np.ndarray, coords: np.ndarray, centres: np.ndarray, color: str) -> None:
        del color
        alpha_row = arrays.query_routing[metrics.selected_query_index]
        # Dashed paths show cheap logical control incidence; they are capped only
        # for legibility, while the exact count remains in the ledger/table.
        for query_index, group, source in triples[:350]:
            group = int(group)
            source = int(source)
            centre = centres[group]
            point = coords[source]
            if not np.isfinite(centre).all():
                continue
            ax.plot([query[0], centre[0], point[0]], [query[1], centre[1], point[1]], color=cmap(group), alpha=min(0.55, 0.10 + 0.35 * float(alpha_row[group])), linewidth=0.55, linestyle="--", zorder=2)
        if len(unique):
            rho_max = max(float(np.max(unique[:, 2])), 1.0e-12)
            for unique_index, (_, source, rho) in enumerate(unique[:350]):
                point = coords[int(source)]
                ax.plot([query[0], point[0]], [query[1], point[1]], color="#8d1b3d" if kind == "module" else "#176f78", alpha=0.45, linewidth=0.65 + 2.0 * float(rho) / rho_max, linestyle="-", zorder=3)
                if unique_index < 12:
                    ax.text(float(point[0]), float(point[1]), f"ρ={float(rho):.2g}", fontsize=4.5, alpha=0.8)

    draw_kind("module", metrics.selected_module_logical_triples, metrics.selected_module_unique_pairs, arrays.module_coords, arrays.module_group_centres, "#8d1b3d")
    draw_kind("environment", metrics.selected_environment_logical_triples, metrics.selected_environment_unique_pairs, arrays.env_coords, arrays.environment_group_centres, "#176f78")
    _set_geometry_axes(ax, arrays)
    values = metrics.values
    decode = values.get("prepared_decode_median_ms")
    decode_text = "NA" if decode is None else f"{float(decode):.3f} ms"
    ax.set_title("D  Logical paths versus unique fine candidates", fontsize=9)
    ax.text(0.01, 0.01, f"dashed: logical q→group→source (cheap control)\nsolid: unique q→source (one fine candidate; rho-scaled)\nlogical M/E: {values.get('module_logical_path_count')}/{values.get('environment_logical_path_count')} · unique M/E: {values.get('module_unique_pair_count')}/{values.get('environment_unique_pair_count')}\nD multiplicity M/E: {values.get('module_multiplicity'):.3g}/{values.get('environment_multiplicity'):.3g} · prepared P2: {decode_text}\nlearned interaction routes; not physical causality", transform=ax.transAxes, fontsize=5.9, va="bottom", bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none", "pad": 2.0})
    ax.legend(fontsize=5.2, loc="best", frameon=True)


def _draw_ledger(ax: Any, ledger: Mapping[str, Any], *, label: str, case_id: str) -> None:
    ax.axis("off")
    rows: list[list[str]] = []
    for phase in group_control.PHASES:
        phase_value = ledger.get(phase, {})
        for source_kind in group_control.SOURCE_TYPES:
            source = phase_value.get(source_kind, {}) if isinstance(phase_value, Mapping) else {}
            if not isinstance(source, Mapping):
                source = {}
            rows.append(
                [
                    f"{phase}/{source_kind[0].upper()}",
                    str(source.get("logical_path_count", "NA")),
                    str(source.get("unique_pair_count", "NA")),
                    f"{float(source.get('multiplicity')):.2f}" if isinstance(source.get("multiplicity"), (float, int)) else "NA",
                    str(source.get("actual_fine_call_count", "NA")),
                    str(source.get("module_mlp_rows", "NA")),
                    str(source.get("environment_geometry_network_rows", "NA")),
                    str(source.get("environment_content_rows", "NA")),
                    str(source.get("scalar_control_rows", "NA")),
                    str(source.get("source_projection_rows", "NA")),
                    str(source.get("forward_call_count", "NA")),
                    str(source.get("checkpoint_recompute_count", "NA")),
                    f"{source.get('valid_pair_denominator', 'NA')}/{source.get('padded_pair_denominator', 'NA')}",
                ]
            )
    columns = ["phase/src", "logical", "unique", "D", "fine calls", "M-MLP", "E-geom", "E-content", "scalar", "source proj", "fwd", "recompute", "valid/pad"]
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(4.9)
    table.scale(1.0, 1.24)
    ax.set_title(f"E  P0/P1/P2 work ledger\n{label} · case {case_id}", fontsize=9, pad=5)
    ax.text(0.01, 0.01, "Counts are summed over receiver chunks before ratios. NA means the backend did not expose an actual phase record.", transform=ax.transAxes, fontsize=5.8, va="bottom")


def _draw_metrics(ax: Any, metrics: group_control.GroupControlMetrics, metadata: Mapping[str, Any], *, label: str, case_id: str) -> None:
    ax.axis("off")
    values = metrics.values
    module_moment_norm = "NA" if metrics.selected_module_moments is None or not len(metrics.selected_module_moments) else f"{float(np.linalg.norm(metrics.selected_module_moments, axis=-1).mean()):.4g}"
    environment_moment_norm = "NA" if metrics.selected_environment_moments is None or not len(metrics.selected_environment_moments) else f"{float(np.linalg.norm(metrics.selected_environment_moments, axis=-1).mean()):.4g}"
    lines = [
        f"F  Metrics and cost · {label} / {case_id}",
        f"K={values.get('K')}  D={values.get('D')}  Q={values.get('Q')}",
        f"active/padded M={values.get('M_active')}/{values.get('M_padded')}  E={values.get('E_active')}",
        f"selected q={values.get('selected_query_index')} at {values.get('selected_query_xy')}",
        f"logical paths M/E={values.get('module_logical_path_count')}/{values.get('environment_logical_path_count')}",
        f"unique q→source M/E={values.get('module_unique_pair_count')}/{values.get('environment_unique_pair_count')}",
        f"multiplicity D M/E={values.get('module_multiplicity'):.3g}/{values.get('environment_multiplicity'):.3g}",
        f"unique/dense valid M/E={values.get('module_unique_over_dense_valid')}/{values.get('environment_unique_over_dense_valid')}",
        f"selected moment ||n|| mean M/E={module_moment_norm}/{environment_moment_norm}",
        f"prepared P2 median={values.get('prepared_decode_median_ms', 'NA')} ms",
        "",
        "A logical path is not a fine call.",
        "Routes are learned interactions, not physical causality.",
        f"map: {metadata.get('label', label)} / {metadata.get('case_id', case_id)}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=7, family="monospace", bbox={"facecolor": "#f7f7f7", "edgecolor": "#cccccc", "pad": 5})


def render_board(evidence_maps: Mapping[str, Mapping[str, str | Path]], output_dir: str | Path, *, case_ids: Sequence[str] = DEFAULT_CASE_IDS) -> dict[str, Any]:
    """Render all supplied labels and cases without executing a model."""

    cases = tuple(str(case_id) for case_id in case_ids)
    if not {"0273", "0653"}.issubset(cases):
        raise ValueError("group-control board requires cases 0273 and 0653")
    if not evidence_maps:
        raise ValueError("at least one group-control evidence label is required")
    loaded: dict[str, dict[str, tuple[group_control.GroupControlArrays, group_control.GroupControlMetrics, dict[str, Any], dict[str, Any], Path]]] = {}
    for label, paths in evidence_maps.items():
        loaded[str(label)] = {}
        for case_id in cases:
            path = Path(paths[str(case_id)]).expanduser().resolve()
            arrays, metrics, ledger, metadata = group_control.load_group_control_npz(path)
            loaded[str(label)][str(case_id)] = (arrays, metrics, ledger, metadata, path)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(loaded) * len(cases)
    figure, axes_grid = plt.subplots(rows, 6, figsize=(26.0, max(7.2, 5.6 * rows)), squeeze=False, constrained_layout=True)
    manifest_rows: list[dict[str, Any]] = []
    row_index = 0
    for label, case_values in loaded.items():
        for case_id in cases:
            arrays, metrics, ledger, metadata, path = case_values[case_id]
            axes = axes_grid[row_index]
            _draw_geometry(axes[0], arrays, label=label, case_id=case_id)
            _draw_incidence(axes[1], arrays, label=label, case_id=case_id)
            _draw_query(axes[2], arrays, metrics, label=label, case_id=case_id)
            _draw_routes(axes[3], arrays, metrics, label=label, case_id=case_id)
            _draw_ledger(axes[4], ledger, label=label, case_id=case_id)
            _draw_metrics(axes[5], metrics, metadata, label=label, case_id=case_id)
            manifest_rows.append({"label": label, "case_id": case_id, "path": str(path), "metrics": metrics.values, "ledger": ledger, "metadata": metadata})
            row_index += 1
    figure.suptitle("Run-1406 group-control interaction board\nincidence · centres · query routing · logical paths versus unique q→source candidates (learned interactions, not physical causality)", fontsize=14)
    png_path = output / "group_control_interaction_board.png"
    pdf_path = output / "group_control_interaction_board.pdf"
    provenance = output / "group_control_interaction_board.json"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "run1406_group_control_interaction_board",
        "status": "ok",
        "case_ids": list(cases),
        "rows": manifest_rows,
        "outputs": {"png": str(png_path), "pdf": str(pdf_path), "provenance": str(provenance)},
        "semantics": {
            "logical_paths": "q->group->source support used by cheap control contractions",
            "unique_pairs": "one q->source candidate for the fine interaction, rho annotated",
            "causality": "not physical causality",
        },
        "limitations": [
            "The selected-query route panel is a readable slice; exact Q-wide counts are in the ledger and metrics JSON.",
            "Debug maps are untimed and cannot replace measured full-forward or prepared-decode timings.",
            "Missing backend phase fields remain NA rather than being inferred from map support.",
        ],
    }
    provenance.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    evidence = _paths_from_comparison(args.comparison.expanduser().resolve(), case_ids=cases) if args.comparison is not None else _paths_from_explicit(args.evidence, case_ids=cases)
    manifest = render_board(evidence, args.output_dir, case_ids=cases)
    print(json.dumps(_jsonable(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_CASE_IDS", "build_parser", "main", "render_board"]
