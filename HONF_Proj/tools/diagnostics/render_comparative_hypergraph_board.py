#!/usr/bin/env python3
"""Render a paired sparse-routing hypergraph board for two exact checkpoints.

The board is intentionally a compact comparison artifact rather than a new
model evaluator.  For each requested case it juxtaposes the P2 candidate
geometry, source-to-hub incidence, and one selected far-query pair view for
the two supplied ledgers.  It requires both ledgers to identify the requested
epoch (2500 by default), so an in-progress Run 2001 cannot silently become a
mixed-epoch comparison.

Run from ``HONF_Proj/``::

    python tools/diagnostics/render_comparative_hypergraph_board.py \
        --run2001-ledger path/to/run2001_epoch2500_routing_ledger.json \
        --run2101-ledger path/to/run2101_epoch2500_routing_ledger.json \
        --output-dir diagnostics/generated/.../comparative_board \
        --case-id 0273 --case-id 0283

The script is CPU-only and reads the selected NPZ maps emitted by the dynamic
routing ledger.  Route weights and hub coordinates are learned diagnostics;
they are not physical influence maps or field-value substitutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "tools" / "diagnostics") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools" / "diagnostics"))

from render_routing_diagnostics import (  # type: ignore[import-not-found]
    _canonical_arrays,
    _phase_view,
    _strategy_label,
)

DEFAULT_CASE_IDS = ("0273", "0283")
DEFAULT_EPOCH = 2500


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_number(value: Any) -> float | None:
    try:
        number = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return default
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text if text else default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(raw: Any, base: Path) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _strategy_from_payload(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("routing_strategy")
    if value is None and isinstance(payload.get("routing_strategy_metadata"), Mapping):
        value = payload["routing_strategy_metadata"].get("strategy")
    if value is None:
        rows = payload.get("rows")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            for row in rows:
                if isinstance(row, Mapping) and row.get("routing_strategy") is not None:
                    value = row["routing_strategy"]
                    break
    return _text(value)


def _row_map_path(payload: Mapping[str, Any], row: Mapping[str, Any], case_id: str, base: Path) -> Path:
    candidates: list[Any] = []
    if row.get("routing_maps") is not None:
        candidates.append(row.get("routing_maps"))
    top_maps = payload.get("routing_maps")
    if isinstance(top_maps, Sequence) and not isinstance(top_maps, (str, bytes)):
        candidates.extend(top_maps)
    for candidate in candidates:
        values = candidate if isinstance(candidate, (list, tuple)) else (candidate,)
        for value in values:
            path = _resolve_path(value, base)
            if path is not None and path.is_file() and (case_id in path.stem or row.get("routing_maps") is not None):
                return path
    raise FileNotFoundError(f"no selected routing NPZ map for case {case_id!r}")


def _active_arrays(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract active P2 geometry/incidence arrays from one selected map."""

    canonical = _canonical_arrays(data)
    module_valid = np.asarray(canonical.get("module_valid"), dtype=bool).reshape(-1)
    env_valid = np.asarray(canonical.get("env_valid"), dtype=bool).reshape(-1)
    hub_valid = np.asarray(canonical.get("hub_valid"), dtype=bool).reshape(-1)
    module_coords = np.asarray(canonical["module_coords"], dtype=float)[module_valid]
    env_coords = np.asarray(canonical["env_coords"], dtype=float)[env_valid]
    hubs = np.asarray(canonical["hub_coords"], dtype=float)[hub_valid]
    module_source = np.asarray(data["p2_field__module_source_A"], dtype=float)[module_valid][:, hub_valid]
    environment_source = np.asarray(data["p2_field__environment_source_A"], dtype=float)[env_valid][:, hub_valid]
    return {
        "module_coords": module_coords,
        "env_coords": env_coords,
        "hub_coords": hubs,
        "module_source_A": module_source,
        "environment_source_A": environment_source,
        "selected_query_xy": np.asarray(data.get("selected_far_query_xy", []), dtype=float).reshape(-1, 2),
    }


def _pair_view(data: Mapping[str, Any], source: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"p2_far_{source}_"
    pair_ids = np.asarray(data.get(f"{prefix}pair_ids", np.empty((0, 3))), dtype=int).reshape(-1, 3)
    prior = np.asarray(data.get(f"{prefix}pair_Pi", np.ones(len(pair_ids))), dtype=float).reshape(-1)
    if not len(pair_ids):
        return np.empty((0,), dtype=int), np.empty((0,), dtype=float)
    valid = (pair_ids[:, 0] == 0) & np.isfinite(prior) & (prior > 0.0)
    return pair_ids[valid, 2], prior[valid]


def _ratio_metrics(payload: Mapping[str, Any], row: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    phase_metrics = row.get("phase_metrics", {})
    module = phase_metrics.get("p2_field_module", {}) if isinstance(phase_metrics, Mapping) else {}
    environment = phase_metrics.get("p2_field_environment", {}) if isinstance(phase_metrics, Mapping) else {}
    module_raw = _finite_number(row.get("p2_field_module_raw_path_count", module.get("raw_path_count")))
    module_unique = _finite_number(row.get("p2_field_module_unique_pair_count", module.get("unique_pair_count")))
    env_raw = _finite_number(row.get("p2_field_environment_raw_path_count", environment.get("raw_path_count")))
    env_unique = _finite_number(row.get("p2_field_environment_unique_pair_count", environment.get("unique_pair_count")))
    m_active = int(_finite_number(row.get("M_active")) or np.count_nonzero(np.asarray(data.get("module_present", [])) > 0.5))
    e_active = int(_finite_number(row.get("environment_source_count")) or len(np.asarray(data.get("env_coords", []))))
    q = int(_finite_number(row.get("query_count")) or 0)
    fine = environment.get("fine_pair_count", {}) if isinstance(environment, Mapping) else {}
    complete = _finite_number(row.get("p2_field_environment_R_completeQE"))
    if complete is None and isinstance(fine, Mapping):
        minimum = _finite_number(fine.get("min"))
        maximum = _finite_number(fine.get("max"))
        count = _finite_number(fine.get("count"))
        if minimum is not None and maximum is not None and count is not None and minimum == e_active and maximum == e_active:
            complete = 1.0
    return {
        "M_active": m_active,
        "E_active": e_active,
        "Q": q,
        "R_duplicate_module": None if module_raw is None or not module_unique else module_raw / module_unique,
        "R_duplicate_environment": None if env_raw is None or not env_unique else env_raw / env_unique,
        "R_module": None if not (q and m_active and module_unique is not None) else module_unique / (q * m_active),
        "R_env": None if not (q and e_active and env_unique is not None) else env_unique / (q * e_active),
        "R_completeQE": complete,
        "R_completeQE_actual": _finite_number(row.get("p2_field_environment_R_completeQE_actual")),
    }


def _load_run(label: str, ledger_path: Path, case_ids: Sequence[str], expected_epoch: int) -> dict[str, Any]:
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"ledger must contain an object: {ledger_path}")
    checkpoint = payload.get("checkpoint", {})
    epoch = _finite_number(checkpoint.get("epoch")) if isinstance(checkpoint, Mapping) else None
    if epoch is None or int(epoch) != int(expected_epoch):
        raise ValueError(f"{label} ledger epoch {epoch!r} does not match required epoch {expected_epoch}")
    rows = payload.get("rows")
    rows_by_case = {
        str(row.get("case_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("case_id") is not None
    } if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else {}
    data_by_case: dict[str, dict[str, Any]] = {}
    strategy = _strategy_from_payload(payload)
    map_records: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        row = rows_by_case.get(str(case_id))
        if row is None:
            raise KeyError(f"{label} ledger lacks requested case {case_id}")
        map_path = _row_map_path(payload, row, str(case_id), ledger_path.parent)
        with np.load(map_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        map_strategy = _text(data.get("routing_strategy"))
        if strategy is None and map_strategy is not None:
            strategy = map_strategy
        data_by_case[str(case_id)] = data
        map_records[str(case_id)] = {
            "path": str(map_path),
            "sha256": _sha256(map_path),
            "strategy": map_strategy,
            "keys": sorted(data),
        }
    if strategy is None:
        strategy = "module_hubs"
    return {
        "label": str(label),
        "ledger": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "payload": payload,
        "rows": rows_by_case,
        "maps": data_by_case,
        "map_records": map_records,
        "checkpoint": dict(checkpoint) if isinstance(checkpoint, Mapping) else {},
        "epoch": int(epoch),
        "strategy": strategy,
        "strategy_label": _strategy_label(strategy),
    }


def _set_axes(ax: Any, points: Sequence[np.ndarray]) -> None:
    values = [point[:, :2] for point in points if point is not None and len(point)]
    if values:
        merged = np.concatenate(values, axis=0)
        lo = merged.min(axis=0)
        hi = merged.max(axis=0)
        span = np.maximum(hi - lo, 1.0e-8)
        ax.set_xlim(lo[0] - 0.06 * span[0], hi[0] + 0.06 * span[0])
        ax.set_ylim(lo[1] - 0.06 * span[1], hi[1] + 0.06 * span[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _draw_geometry(ax: Any, run: Mapping[str, Any], case_id: str, data: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    arrays = _active_arrays(data)
    ax.scatter(arrays["env_coords"][:, 0], arrays["env_coords"][:, 1], s=7, color="#a0a7ae", alpha=0.22, label="E sources")
    ax.scatter(arrays["module_coords"][:, 0], arrays["module_coords"][:, 1], s=42, marker="s", color="#315f7c", label="M sources")
    ax.scatter(arrays["hub_coords"][:, 0], arrays["hub_coords"][:, 1], s=76, marker="D", facecolors="none", edgecolors="#bf6b3c", linewidths=1.2, label="active hubs")
    selected_query = arrays["selected_query_xy"]
    if len(selected_query):
        ax.scatter(selected_query[0, 0], selected_query[0, 1], s=150, marker="*", color="#e53e3e", edgecolors="black", linewidths=0.4, label="selected P2 query")
    for index, xy in enumerate(arrays["hub_coords"]):
        ax.text(float(xy[0]), float(xy[1]), str(index), fontsize=6, ha="center", va="center")
    _set_axes(ax, [arrays["env_coords"], arrays["module_coords"], arrays["hub_coords"], selected_query])
    ax.set_title(f"{run['strategy_label']} · {case_id}\nM={metrics['M_active']} E={metrics['E_active']} K={len(arrays['hub_coords'])}", fontsize=10)
    ax.legend(fontsize=6, loc="best", frameon=True)


def _draw_incidence(ax: Any, run: Mapping[str, Any], case_id: str, data: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    arrays = _active_arrays(data)
    module = arrays["module_source_A"]
    environment = arrays["environment_source_A"]
    matrix = np.vstack((module, environment))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=max(float(matrix.max()), 1.0e-12))
    ax.axhline(len(module) - 0.5, color="white", linewidth=0.8)
    ax.text(-0.02, max(len(module) / max(len(matrix), 1) * 0.5, 0.02), "M", transform=ax.transAxes, ha="right", va="center", color="#315f7c", weight="bold")
    ax.text(-0.02, (len(module) + len(matrix)) / max(2 * len(matrix), 1), "E", transform=ax.transAxes, ha="right", va="center", color="#53606e", weight="bold")
    ax.set_xlabel("active candidate hub")
    ax.set_ylabel("source rows")
    ax.set_title(f"{run['strategy_label']} · {case_id}\nsource-to-hub incidence (M over E)", fontsize=10)
    ax.figure.colorbar(image, ax=ax, pad=0.02, label="membership")


def _draw_selected_pairs(ax: Any, run: Mapping[str, Any], case_id: str, data: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    arrays = _active_arrays(data)
    receiver = arrays["selected_query_xy"]
    receiver_xy = receiver[0] if len(receiver) else np.array([np.nan, np.nan])
    ax.scatter(arrays["env_coords"][:, 0], arrays["env_coords"][:, 1], s=7, color="#a0a7ae", alpha=0.12)
    ax.scatter(arrays["module_coords"][:, 0], arrays["module_coords"][:, 1], s=35, marker="s", color="#315f7c", alpha=0.35, label="M sources")
    ax.scatter(arrays["hub_coords"][:, 0], arrays["hub_coords"][:, 1], s=65, marker="D", facecolors="none", edgecolors="#bf6b3c", linewidths=1.0, label="active hubs")
    if np.isfinite(receiver_xy).all():
        ax.scatter(receiver_xy[0], receiver_xy[1], s=165, marker="*", color="#e53e3e", edgecolors="black", linewidths=0.4, label="selected P2 query")
    source_specs = (("module", arrays["module_coords"], "#315f7c", "s"), ("environment", arrays["env_coords"], "#7b8791", "."))
    for source, coordinates, color, marker in source_specs:
        indices, priors = _pair_view(data, source)
        valid = (indices >= 0) & (indices < len(coordinates))
        indices, priors = indices[valid], priors[valid]
        if not len(indices):
            continue
        points = coordinates[indices]
        ax.scatter(points[:, 0], points[:, 1], s=35 if marker == "s" else 10, marker=marker, color=color, alpha=0.82, label=f"{source} unique pairs ({len(indices)})")
        if np.isfinite(receiver_xy).all():
            scale = max(float(np.max(priors)), 1.0e-12)
            for point, prior in zip(points, priors):
                ax.plot([receiver_xy[0], point[0]], [receiver_xy[1], point[1]], color=color, alpha=min(0.65, 0.10 + 0.5 * float(prior) / scale), linewidth=0.45)
    _set_axes(ax, [arrays["env_coords"], arrays["module_coords"], arrays["hub_coords"], receiver])
    ax.set_title(f"{run['strategy_label']} · {case_id}\nselected P2 query unique pairs", fontsize=10)
    ax.legend(fontsize=6, loc="best", frameon=True)


def _metric_text(metrics: Mapping[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.3g}"

    return (
        f"Rdup M/E={fmt(metrics.get('R_duplicate_module'))}/{fmt(metrics.get('R_duplicate_environment'))} · "
        f"Rmodule/Renv={fmt(metrics.get('R_module'))}/{fmt(metrics.get('R_env'))} · "
        f"RcompleteQE={fmt(metrics.get('R_completeQE'))}"
    )


def render_board(
    run2001_ledger: str | Path,
    run2101_ledger: str | Path,
    output_dir: str | Path,
    *,
    case_ids: Sequence[str] = DEFAULT_CASE_IDS,
    expected_epoch: int = DEFAULT_EPOCH,
) -> dict[str, Any]:
    """Render and return the comparative board manifest."""

    cases = tuple(str(case) for case in case_ids)
    if not {"0273", "0283"}.issubset(cases):
        raise ValueError("comparative board requires anchor cases 0273 and 0283")
    runs = [
        _load_run("Run 2001", Path(run2001_ledger).expanduser().resolve(), cases, expected_epoch),
        _load_run("Run 2101", Path(run2101_ledger).expanduser().resolve(), cases, expected_epoch),
    ]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(cases) * len(runs), 3, figsize=(18.0, max(7.5, 5.4 * len(cases))), squeeze=False, constrained_layout=True)
    metric_manifest: dict[str, dict[str, Any]] = {}
    for case_index, case_id in enumerate(cases):
        metric_manifest[case_id] = {}
        for run_index, run in enumerate(runs):
            row_index = case_index * len(runs) + run_index
            data = run["maps"][case_id]
            metrics = _ratio_metrics(run["payload"], run["rows"][case_id], data)
            metric_manifest[case_id][run["label"]] = metrics
            _draw_geometry(axes[row_index, 0], run, case_id, data, metrics)
            _draw_incidence(axes[row_index, 1], run, case_id, data, metrics)
            _draw_selected_pairs(axes[row_index, 2], run, case_id, data, metrics)
            axes[row_index, 0].text(0.01, -0.17, _metric_text(metrics), transform=axes[row_index, 0].transAxes, fontsize=7, color="#53606e")
    figure.suptitle(
        "Comparative sparse-routing hypergraph board · exact epoch "
        f"{expected_epoch}\nGeometry · source incidence · selected P2 pair view (route weights are learned diagnostics)",
        fontsize=14,
    )
    png_path = output / "comparative_hypergraph_board.png"
    pdf_path = output / "comparative_hypergraph_board.pdf"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "comparative_sparse_routing_hypergraph_board",
        "status": "ok",
        "expected_epoch": int(expected_epoch),
        "case_ids": list(cases),
        "runs": [
            {
                "label": run["label"],
                "ledger": run["ledger"],
                "ledger_sha256": run["ledger_sha256"],
                "checkpoint": run["checkpoint"],
                "epoch": run["epoch"],
                "strategy": run["strategy"],
                "strategy_label": run["strategy_label"],
                "maps": run["map_records"],
            }
            for run in runs
        ],
        "metrics": metric_manifest,
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "provenance": str(output / "comparative_hypergraph_board.json"),
        },
        "limitations": [
            "The selected-pair panel uses the ledger's selected far P2 query, not every receiver/query row.",
            "Incidence and pair priors describe learned routing and compiler execution; they are not physical influence maps.",
            "Accuracy, memory, and timing claims belong to the paired evaluation/profile artifacts, not this visualization.",
        ],
    }
    provenance_path = output / "comparative_hypergraph_board.json"
    provenance_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run2001-ledger", type=Path, required=True)
    parser.add_argument("--run2101-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=None, help="repeatable anchor; defaults to 0273 and 0283")
    parser.add_argument("--expected-epoch", type=int, default=DEFAULT_EPOCH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = render_board(
        args.run2001_ledger,
        args.run2101_ledger,
        args.output_dir,
        case_ids=args.case_id or DEFAULT_CASE_IDS,
        expected_epoch=args.expected_epoch,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_CASE_IDS", "DEFAULT_EPOCH", "build_parser", "main", "render_board"]
