"""Validation, streaming field statistics, and report-friendly summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import WindFarmDataset
from .splits import make_group_split, write_split_outputs

SCHEMA_VERSION = 1
FIELD_NAMES: tuple[str, ...] = ("U", "p", "k", "epsilon")


def _json_value(value: Any) -> Any:
    """Convert NumPy scalars/arrays into stable JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _counts(values: Iterable[Any]) -> list[dict[str, Any]]:
    """Count scalar values or fixed-size vector/tuple values row-wise."""

    normalized: list[Any] = []
    for value in values:
        array = np.asarray(value)
        if array.ndim == 0:
            normalized.append(_json_value(array))
        else:
            normalized.append(tuple(_json_value(item) for item in array.reshape(-1)))
    counts = Counter(normalized)
    return [
        {"value": value, "count": int(count)} for value, count in sorted(counts.items(), key=lambda item: repr(item[0]))
    ]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {name: float("nan") for name in ("min", "q25", "median", "q75", "max")}
    q = np.percentile(finite, [0, 25, 50, 75, 100])
    return {
        "min": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "max": float(q[4]),
    }


@dataclass
class RunningStats:
    """Numerically stable streaming summary for one scalar field."""

    count: int = 0
    finite_count: int = 0
    nonfinite_count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    sum_value: float = 0.0
    sum_square: float = 0.0

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values).reshape(-1)
        self.count += int(flat.size)
        finite = np.asarray(flat[np.isfinite(flat)], dtype=np.float64)
        self.finite_count += int(finite.size)
        self.nonfinite_count += int(flat.size - finite.size)
        if finite.size == 0:
            return
        self.minimum = min(self.minimum, float(np.min(finite)))
        self.maximum = max(self.maximum, float(np.max(finite)))
        self.sum_value += float(np.sum(finite, dtype=np.float64))
        self.sum_square += float(np.sum(finite * finite, dtype=np.float64))

    def as_dict(self) -> dict[str, Any]:
        if self.finite_count == 0:
            mean = std = float("nan")
            minimum = maximum = float("nan")
        else:
            mean = self.sum_value / self.finite_count
            variance = max(0.0, self.sum_square / self.finite_count - mean * mean)
            std = float(np.sqrt(variance))
            minimum = self.minimum
            maximum = self.maximum
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "min": minimum,
            "max": maximum,
            "mean": float(mean),
            "std": std,
        }


def _sample_indices(start: int, stop: int, count: int) -> np.ndarray:
    if stop <= start or count <= 0:
        return np.empty((0,), dtype=np.int64)
    actual = min(int(count), stop - start)
    # Linspace gives deterministic coverage of both ends of every run and
    # avoids allocating a permutation proportional to the full field size.
    return np.unique(np.linspace(start, stop - 1, actual, dtype=np.int64))


def _field_stats(
    dataset: WindFarmDataset,
    *,
    sample_points_per_run: int,
    full_scan: bool,
    chunk_size: int,
) -> dict[str, dict[str, Any]]:
    stats = {name: RunningStats() for name in FIELD_NAMES}
    chunk = max(1, int(chunk_size))
    for index in range(dataset.n_runs):
        bounds = dataset.run_bounds(index)
        if full_scan:
            ranges = range(bounds.start, bounds.stop, chunk)
            slices = [slice(start, min(start + chunk, bounds.stop)) for start in ranges]
        else:
            indices = _sample_indices(bounds.start, bounds.stop, sample_points_per_run)
            slices = [indices]
        for selection in slices:
            for name in FIELD_NAMES:
                stats[name].update(dataset.array(name)[selection])
    return {name: value.as_dict() for name, value in stats.items()}


def _axis_summary(dataset: WindFarmDataset, index: int, axis: str) -> dict[str, Any]:
    values = np.asarray(dataset.axis(index, axis), dtype=np.float64)
    finite = values[np.isfinite(values)]
    monotonic = bool(finite.size < 2 or np.all(np.diff(finite) > 0))
    return {
        "count": int(values.size),
        "finite": int(finite.size),
        "strictly_increasing": monotonic,
        "min_m": float(np.min(finite)) if finite.size else float("nan"),
        "max_m": float(np.max(finite)) if finite.size else float("nan"),
    }


def case_rows(dataset: WindFarmDataset) -> list[dict[str, Any]]:
    """Build one compact, CSV-friendly metadata row per CFD run."""

    rows: list[dict[str, Any]] = []
    for index in range(dataset.n_runs):
        run = dataset.run(index)
        rows.append(
            {
                "index": index,
                "case": run.case,
                "layout_index": run.layout_index,
                "wind_direction_deg": run.wind_direction_deg,
                "source_time": run.source_time,
                "completed": int(dataset.array("completed")[index]),
                "nx": run.nx,
                "ny": run.ny,
                "nz": run.nz,
                "cell_count": run.cell_count,
                "x_min_m": float(run.x_m[0]) if len(run.x_m) else float("nan"),
                "x_max_m": float(run.x_m[-1]) if len(run.x_m) else float("nan"),
                "y_min_m": float(run.y_m[0]) if len(run.y_m) else float("nan"),
                "y_max_m": float(run.y_m[-1]) if len(run.y_m) else float("nan"),
                "z_min_m": float(run.z_m[0]) if len(run.z_m) else float("nan"),
                "z_max_m": float(run.z_m[-1]) if len(run.z_m) else float("nan"),
            }
        )
    return rows


def _array_inventory(dataset: WindFarmDataset) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for name in dataset.arrays:
        array = dataset.array(name)
        path = dataset.volume_root / f"{name}.npy"
        item: dict[str, Any] = {
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
            "file": str(path),
            "file_size_bytes": int(path.stat().st_size) if path.exists() else None,
            "memory_mapped": isinstance(array, np.memmap),
        }
        inventory.append(item)
    return inventory


def _validation(dataset: WindFarmDataset) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    n = dataset.n_runs
    shapes = np.asarray(dataset.array("run_shape"))
    if np.any(shapes <= 0):
        errors.append("run_shape contains non-positive dimensions")
    completed = np.asarray(dataset.array("completed"))
    incomplete = np.flatnonzero(completed != 1)
    if incomplete.size:
        errors.append(f"completed marker is not 1 for {incomplete.size} runs")
    for name in ("run_cell_offsets", "run_x_offsets", "run_y_offsets", "run_z_offsets"):
        offsets = np.asarray(dataset.array(name))
        if offsets[0] != 0:
            errors.append(f"{name} does not start at zero")
        if np.any(np.diff(offsets) < 0):
            errors.append(f"{name} is not monotone")
    cell_offsets = np.asarray(dataset.array("run_cell_offsets"))
    expected_cells = np.prod(shapes.astype(np.int64), axis=1)
    actual_cells = np.diff(cell_offsets)
    if not np.array_equal(expected_cells, actual_cells):
        errors.append("run_shape products do not agree with run_cell_offsets")
    if int(cell_offsets[-1]) != int(dataset.array("U").shape[0]):
        errors.append("run_cell_offsets total does not agree with U.npy")
    for name in ("p", "k", "epsilon"):
        if dataset.array(name).shape != dataset.array("U").shape[:1]:
            errors.append(f"{name}.npy does not share U.npy cell count")
    axis_pairs = (
        ("x", "run_x_offsets", "run_shape", 0),
        ("y", "run_y_offsets", "run_shape", 1),
        ("z", "run_z_offsets", "run_shape", 2),
    )
    for axis, offset_name, _, dimension in axis_pairs:
        lengths = np.diff(np.asarray(dataset.array(offset_name)))
        if not np.array_equal(lengths, shapes[:, dimension]):
            errors.append(f"{offset_name} lengths do not agree with run_shape axis {axis}")
    # Axis monotonicity is cheap (about 350k coordinates in the source copy).
    bad_axes: list[str] = []
    for index in range(n):
        for axis in "xyz":
            values = np.asarray(dataset.axis(index, axis))
            if len(values) > 1 and not np.all(np.diff(values) > 0):
                bad_axes.append(f"{index}:{axis}")
    if bad_axes:
        errors.append(f"non-increasing cell-centre axes ({len(bad_axes)} run/axis pairs)")
    _, group_row_counts = np.unique(dataset.array("layout_index"), return_counts=True)
    if np.any(~np.isin(group_row_counts, (2, 3))):
        warnings.append("some layout groups do not have two or three directional rows")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def profile_dataset(
    dataset: WindFarmDataset,
    *,
    sample_points_per_run: int = 256,
    full_scan: bool = False,
    chunk_size: int = 1_000_000,
) -> dict[str, Any]:
    """Validate and summarize a volume dataset with bounded memory use."""

    rows = case_rows(dataset)
    shapes = np.asarray(dataset.array("run_shape"))
    cells = np.prod(shapes.astype(np.int64), axis=1)
    layout = np.asarray(dataset.array("layout_index"))
    wind_direction = np.asarray(dataset.array("wd_deg"))
    completed = np.asarray(dataset.array("completed"))
    _, group_row_counts = np.unique(layout, return_counts=True)
    profile: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "volume_root": str(dataset.volume_root),
            "dataset_root": str(dataset.root),
            "family_tensor_bundle_present": (dataset.root / "family_tensor.npz").is_file(),
            "compact_individual_arrays_present": any(
                (dataset.root / "family_tensor" / f"{name}.npy").is_file()
                for name in ("case", "layout_index", "wd_deg")
            ),
        },
        "dataset": {
            "runs": dataset.n_runs,
            "total_cells": int(cells.sum()),
            "layout_groups": int(np.unique(layout).size),
            "directions": [float(value) for value in np.unique(wind_direction)],
            "completed_runs": int(np.count_nonzero(completed == 1)),
            "cell_count": _quantiles(cells),
            "shape_unique": int(np.unique(shapes, axis=0).shape[0]),
        },
        "categories": {
            "wind_direction_deg": _counts(wind_direction),
            "layout_index": _counts(layout),
            "layout_row_counts": _counts(group_row_counts),
            "completed": _counts(completed),
            "shape_nxyz": _counts(tuple(int(value) for value in row) for row in shapes),
            "source_time": _counts(dataset.array("source_time")),
        },
        "arrays": _array_inventory(dataset),
        "validation": _validation(dataset),
        "field_statistics": _field_stats(
            dataset,
            sample_points_per_run=sample_points_per_run,
            full_scan=full_scan,
            chunk_size=chunk_size,
        ),
        "sampling": {
            "mode": "full_scan" if full_scan else "deterministic_even_points_per_run",
            "sample_points_per_run": int(sample_points_per_run),
            "chunk_size": int(chunk_size),
        },
        "axis_extents_m": {
            axis: {
                "min": float(np.min(dataset.array(f"{axis}_cell_m"))),
                "max": float(np.max(dataset.array(f"{axis}_cell_m"))),
            }
            for axis in "xyz"
        },
        "case_rows": rows,
    }
    # Preserve a compact stable hash for report provenance without hashing 57 GB
    # of fields.  Metadata and offsets are sufficient to identify the view.
    digest = hashlib.sha256()
    for name in ("case", "layout_index", "wd_deg", "run_shape", "run_cell_offsets"):
        digest.update(np.asarray(dataset.array(name)).tobytes())
    profile["source"]["metadata_sha256"] = digest.hexdigest()
    return profile


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_value) + "\n",
        encoding="utf-8",
    )


def _write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_profile_outputs(
    profile: dict[str, Any],
    output_dir: str | Path,
    *,
    group_values: np.ndarray | None = None,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, Path]:
    """Write the stable JSON/CSV/NPZ inspection outputs.

    ``profile`` must be produced by :func:`profile_dataset`; ``group_values``
    is kept separate so callers can avoid re-opening a large dataset.
    """

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "windfarm_profile.json"
    cases_path = output / "windfarm_cases.csv"
    splits_path = output / "windfarm_splits.json"
    indices_path = output / "windfarm_split_indices.npz"
    _json_dump(profile_path, profile)
    _write_case_csv(cases_path, list(profile.get("case_rows", [])))

    if group_values is None:
        group_values = np.asarray([row["layout_index"] for row in profile["case_rows"]])
    split = make_group_split(group_values, seed=seed, fractions=fractions)
    split_json = split.metadata
    _json_dump(splits_path, split_json)
    write_split_outputs(indices_path, split)
    return {
        "profile": profile_path,
        "cases": cases_path,
        "splits": splits_path,
        "split_indices": indices_path,
    }
