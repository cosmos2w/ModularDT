"""Prepare the small native WindFarm forward-study view.

This command reads only volume metadata/axes and bounded sampled ``U`` rows.
It writes ignored index/statistics sidecars; the raw ragged arrays remain
external and mmap-backed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

from windfarm.data import WindFarmNativeView
from windfarm.geometry import (
    D_M,
    ENV_TOKEN_SHAPE,
    HUB_HEIGHT_M,
    U_REF_MPS,
    gather_native_sample,
)
from windfarm.normalization import (
    DEFAULT_PROFILE_BINS,
    DEFAULT_SAMPLES_PER_ROW,
    fit_velocity_statistics,
    write_normalization_json,
)
from windfarm.splits import make_group_split


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_value) + "\n",
        encoding="utf-8",
    )


def _case_rows(view: WindFarmNativeView) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    volume = view.volume
    for index in range(view.n_cases):
        run = volume.run(index)
        support = view.run(index).support
        bounds = volume.run_bounds(index)
        rows.append(
            {
                "source_index": index,
                "case": run.case,
                "layout": str(np.asarray(view.metadata["layout"])[index]),
                "layout_index": run.layout_index,
                "wind_direction_deg": run.wind_direction_deg,
                "n_turbines": int(np.asarray(view.metadata["n_turbines"])[index]),
                "nx": run.nx,
                "ny": run.ny,
                "nz": run.nz,
                "cell_count": run.cell_count,
                "run_cell_start": int(bounds.start),
                "run_cell_stop": int(bounds.stop),
                "x_min_m": float(run.x_m[0]),
                "x_max_m": float(run.x_m[-1]),
                "y_min_m": float(run.y_m[0]),
                "y_max_m": float(run.y_m[-1]),
                "z_min_m": float(run.z_m[0]),
                "z_max_m": float(run.z_m[-1]),
                "support_volume_m3": support.volume_m3,
                "support_volume_D3": support.volume_D3,
            }
        )
    return rows


def _write_case_index(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _geometry_metadata(path: Path, view: WindFarmNativeView) -> None:
    volume = view.volume
    support_lower = np.empty((view.n_cases, 3), dtype=np.float32)
    support_upper = np.empty((view.n_cases, 3), dtype=np.float32)
    environment_coords = np.empty((view.n_cases, int(np.prod(ENV_TOKEN_SHAPE)), 3), dtype=np.float32)
    environment_features = np.empty((view.n_cases, int(np.prod(ENV_TOKEN_SHAPE)), 7), dtype=np.float32)
    environment_weights = np.empty((view.n_cases, int(np.prod(ENV_TOKEN_SHAPE))), dtype=np.float32)
    for index in range(view.n_cases):
        case = view.run(index)
        support_lower[index] = case.support.lower_D
        support_upper[index] = case.support.upper_D
        environment_coords[index] = case.env_coords
        environment_features[index] = case.env_features
        environment_weights[index] = case.env_weights
    np.savez_compressed(
        path,
        support_lower_D=support_lower,
        support_upper_D=support_upper,
        environment_coords_D=environment_coords,
        environment_features=environment_features,
        environment_weights_D3=environment_weights,
        x_axis_offsets=np.asarray(volume.array("run_x_offsets"), dtype=np.int64),
        y_axis_offsets=np.asarray(volume.array("run_y_offsets"), dtype=np.int64),
        z_axis_offsets=np.asarray(volume.array("run_z_offsets"), dtype=np.int64),
        x_axis_values_m=np.asarray(volume.array("x_cell_m"), dtype=np.float32),
        y_axis_values_m=np.asarray(volume.array("y_cell_m"), dtype=np.float32),
        z_axis_values_m=np.asarray(volume.array("z_cell_m"), dtype=np.float32),
    )


def _access_checks(view: WindFarmNativeView) -> list[dict[str, Any]]:
    cells = np.asarray(view.volume.array("run_shape"), dtype=np.int64).prod(axis=1)
    order = np.argsort(cells, kind="stable")
    selected = (int(order[0]), int(order[len(order) // 2]), int(order[-1]))
    checks: list[dict[str, Any]] = []
    for row in selected:
        case = view.run(row)
        sample = gather_native_sample(
            case.run,
            8,
            np.random.default_rng(np.random.SeedSequence((42, 0, row, 909))),
            mode="volume",
        )
        direct = np.asarray(case.run.U[sample.flat_indices], dtype=np.float32)
        checks.append(
            {
                "source_index": row,
                "case": case.case,
                "shape_nxyz": list(case.shape_nxyz),
                "support_lower_D": case.support.lower_D,
                "support_upper_D": case.support.upper_D,
                "sample_count": int(sample.flat_indices.size),
                "native_gather_matches_direct_slice": bool(np.array_equal(sample.velocity_mps, direct)),
                "sample_coords_D_min": np.min(sample.coords_D, axis=0),
                "sample_coords_D_max": np.max(sample.coords_D, axis=0),
            }
        )
    return checks


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], np.dtype[Any], bool]:
    """Read a safe numeric NPY header from an uncompressed NPZ member."""

    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported compact NPY header version {version}")
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError("compact comparison refuses object-dtype members")
    return tuple(int(value) for value in shape), dtype, bool(fortran_order)


def _read_npz_member(path: Path, member: str, row: int | None = None) -> np.ndarray:
    """Read one compact member or one leading row without materializing NPZ fields."""

    with ZipFile(path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise FileNotFoundError(f"compact bundle is missing {member}") from exc
        with archive.open(info, "r") as stream:
            shape, dtype, fortran_order = _read_npy_header(stream)
            if fortran_order:
                raise ValueError(f"compact comparison expects C-order member {member}")
            if row is None:
                payload = stream.read(int(np.prod(shape, dtype=np.int64)) * dtype.itemsize)
                expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if len(payload) != expected:
                    raise ValueError(f"short compact member {member}: {len(payload)} != {expected} bytes")
                return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
            if len(shape) == 0 or row < 0 or row >= shape[0]:
                raise IndexError(f"row {row} is outside compact member {member} with shape {shape}")
            row_count = int(np.prod(shape[1:], dtype=np.int64))
            row_bytes = row_count * dtype.itemsize
            stream.seek(int(row) * row_bytes, 1)
            payload = stream.read(row_bytes)
            if len(payload) != row_bytes:
                raise ValueError(f"short compact row {row} in {member}")
            return np.frombuffer(payload, dtype=dtype).reshape(shape[1:]).copy()


def _nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Find nearest indices on a sorted native axis without a dense distance matrix."""

    positions = np.searchsorted(axis, values, side="left")
    right = np.clip(positions, 0, len(axis) - 1)
    left = np.clip(positions - 1, 0, len(axis) - 1)
    choose_left = np.abs(values - axis[left]) <= np.abs(axis[right] - values)
    return np.where(choose_left, left, right).astype(np.int64)


def _compact_native_comparisons(view: WindFarmNativeView, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare bounded compact hub rows with matching native hub-like planes."""

    bundle = view.volume.root / "family_tensor.npz"
    if not bundle.is_file():
        return [{"available": False, "reason": "family_tensor.npz is not present"}]
    compact_x_D = np.asarray(_read_npz_member(bundle, "x_D.npy"), dtype=np.float64)
    compact_y_D = np.asarray(_read_npz_member(bundle, "y_D.npy"), dtype=np.float64)
    results: list[dict[str, Any]] = []
    for item in rows:
        row = int(item["source_index"])
        compact_u = np.asarray(_read_npz_member(bundle, "U_hub.npy", row=row), dtype=np.float64)
        compact_valid = np.asarray(_read_npz_member(bundle, "valid_hub.npy", row=row), dtype=bool)
        case = view.run(row)
        z_index = int(np.argmin(np.abs(np.asarray(case.z_m, dtype=np.float64) - HUB_HEIGHT_M)))
        native_plane = np.asarray(case.run.U_structured[z_index, :, :, :2], dtype=np.float64)
        compact_iy, compact_ix = np.nonzero(compact_valid)
        compact_x_m = compact_x_D[compact_ix] * D_M
        compact_y_m = compact_y_D[compact_iy] * D_M
        inside = (
            (compact_x_m >= float(case.x_m[0]))
            & (compact_x_m <= float(case.x_m[-1]))
            & (compact_y_m >= float(case.y_m[0]))
            & (compact_y_m <= float(case.y_m[-1]))
        )
        compact_iy = compact_iy[inside]
        compact_ix = compact_ix[inside]
        compact_x_m = compact_x_m[inside]
        compact_y_m = compact_y_m[inside]
        native_ix = _nearest_axis_indices(np.asarray(case.x_m, dtype=np.float64), compact_x_m)
        native_iy = _nearest_axis_indices(np.asarray(case.y_m, dtype=np.float64), compact_y_m)
        compact_values = compact_u[compact_iy, compact_ix, :]
        native_values = native_plane[native_iy, native_ix, :]
        difference = native_values - compact_values
        results.append(
            {
                "available": True,
                "source_index": row,
                "case": case.case,
                "compact_valid_points": int(np.count_nonzero(compact_valid)),
                "matched_points_inside_native_support": len(native_values),
                "native_hub_axis_requested_m": HUB_HEIGHT_M,
                "native_hub_axis_actual_m": float(case.z_m[z_index]),
                "mean_abs_xy_nearest_offset_m": [
                    float(np.mean(np.abs(np.asarray(case.x_m)[native_ix] - compact_x_m))),
                    float(np.mean(np.abs(np.asarray(case.y_m)[native_iy] - compact_y_m))),
                ],
                "rmse_m_s": np.sqrt(np.mean(difference**2, axis=0)).tolist() if len(native_values) else None,
                "mae_m_s": np.mean(np.abs(difference), axis=0).tolist() if len(native_values) else None,
                "comparison_note": "supplementary nearest-centre comparison; compact and native views are not interchangeable labels",
            }
        )
    return results


def prepare_forward_view(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    samples_per_row: int = DEFAULT_SAMPLES_PER_ROW,
    profile_bins: int = DEFAULT_PROFILE_BINS,
) -> dict[str, Path]:
    """Create the ignored compact index, split, normalization, and geometry sidecars."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    view = WindFarmNativeView(dataset_root)
    if view.n_cases != 600:
        raise ValueError(f"the WindFarm forward view expects 600 rows, found {view.n_cases}")
    if not np.isclose(view.diameter_m, D_M) or not np.isclose(view.hub_height_m, HUB_HEIGHT_M):
        raise ValueError("source geometry constants differ from the documented WindFarm contract")
    if not np.isclose(view.reference_speed_mps, U_REF_MPS):
        raise ValueError("source reference speed differs from the documented WindFarm contract")
    groups = np.asarray(view.volume.array("layout_index"))
    split = make_group_split(groups, seed=int(seed))
    if (split.train.size, split.validation.size, split.test.size) != (420, 90, 90):
        raise ValueError("seed-42 grouped split did not produce 420/90/90 rows")
    rows = _case_rows(view)
    case_index = output / "case_index.csv"
    _write_case_index(case_index, rows)
    split_path = output / "split_indices.npz"
    np.savez_compressed(split_path, train=split.train, validation=split.validation, test=split.test)
    split_metadata = {
        "schema_version": 1,
        "seed": int(seed),
        "fractions": [0.70, 0.15, 0.15],
        "group_key": "layout_index",
        "rows": {"train": int(split.train.size), "validation": int(split.validation.size), "test": int(split.test.size)},
        "groups": {
            name: int(np.unique(groups[indices]).size)
            for name, indices in (("train", split.train), ("validation", split.validation), ("test", split.test))
        },
        "directions_kept_with_layout": True,
    }
    split_json = output / "split_metadata.json"
    _write_json(split_json, split_metadata)
    normalizer, profile = fit_velocity_statistics(
        view.volume,
        split.train,
        samples_per_row=int(samples_per_row),
        seed=int(seed),
        profile_bins=int(profile_bins),
    )
    normalization_path = output / "normalization.json"
    write_normalization_json(normalization_path, normalizer, profile)
    geometry_path = output / "geometry_metadata.npz"
    _geometry_metadata(geometry_path, view)
    access_checks = _access_checks(view)
    compact_native_comparisons = _compact_native_comparisons(view, access_checks)
    sampling_path = output / "sampling_metadata.json"
    _write_json(
        sampling_path,
        {
            "schema_version": 1,
            "source_volume_id": "wind_farm_volume_v1",
            "source_compact_id": "wind_farm_tensor_v1",
            "volume_root": view.volume.volume_root,
            "seed": int(seed),
            "split": "layout-grouped 70/15/15, 420/90/90 rows",
            "target_components": ["Ux", "Uy", "Uz"],
            "target_source": "native U.npy only",
            "target_normalization": "training rows only; equal 8192 volume-weighted samples per row",
            "samples_per_training_row": int(samples_per_row),
            "queries_per_case": 1024,
            "query_mixture": {"volume": 0.75, "hub_height_band": 0.25},
            "hub_height_m": HUB_HEIGHT_M,
            "rotor_diameter_m": D_M,
            "environment_token_shape": list(ENV_TOKEN_SHAPE),
            "environment_token_count": int(np.prod(ENV_TOKEN_SHAPE)),
            "environment_token_content": "geometry/support features only; no solved field or wake-loss target",
            "support_definition": "exact native cell-centre bounding box",
            "quadrature": "factorized cell-centre support weights; unknown outer CFD faces excluded",
            "full_volume_copy": False,
            "representative_native_access_checks": access_checks,
            "compact_native_hub_comparisons": compact_native_comparisons,
        },
    )
    return {
        "case_index": case_index,
        "split_indices": split_path,
        "split_metadata": split_json,
        "normalization": normalization_path,
        "sampling_metadata": sampling_path,
        "geometry_metadata": geometry_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Dataset" / "links" / "wind_farm",
        help="dataset root or family_volume directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Dataset" / "derived" / "forward_velocity_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-row", type=int, default=DEFAULT_SAMPLES_PER_ROW)
    parser.add_argument("--profile-bins", type=int, default=DEFAULT_PROFILE_BINS)
    args = parser.parse_args()
    paths = prepare_forward_view(
        args.dataset_root,
        args.output_dir,
        seed=args.seed,
        samples_per_row=args.samples_per_row,
        profile_bins=args.profile_bins,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
