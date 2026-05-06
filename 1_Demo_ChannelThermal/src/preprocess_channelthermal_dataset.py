"""Pack raw channel thermal cases into a steady/quasi-steady HDF5 dataset.

This first Demo 1 preprocessing pass is deliberately not phase-cycle based.
For each raw case it selects saved frames after heat activation, averages the
final window, stores an RMS field over that window, and adds point samples for
neural-field training.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("preprocess_channelthermal_dataset.py requires h5py.") from exc

from channelthermal_common import (
    SimulationConfig,
    build_uniform_grid,
    config_from_dict,
    find_case_dirs,
    kinematic_viscosity,
    read_json,
    resolve_data_path,
)


CHANNEL_ORDER = ("u", "v", "p", "omega", "temperature")
SAMPLED_POINT_FEATURES = ("x", "y", "u", "v", "p", "omega", "temperature")


@dataclass
class RawCase:
    split_hint: str
    case_dir: Path
    cfg: SimulationConfig
    cfg_payload: Dict[str, Any]
    frame_rows: List[Dict[str, str]]


@dataclass
class ProcessedCase:
    case_key: str
    split: str
    case_dir: Path
    cfg: SimulationConfig
    cfg_payload: Dict[str, Any]
    selected_times: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    steady_field: np.ndarray
    rms_field: np.ndarray
    sampled_points: np.ndarray
    module_internal_temperature: np.ndarray
    module_internal_mask: np.ndarray
    interface_response: np.ndarray
    interface_feature_names: Tuple[str, ...]
    module_centers: np.ndarray
    heat_powers: np.ndarray
    module_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess channel thermal cases into packed_dataset.h5.")
    parser.add_argument("--input-root", type=Path, default=Path("./Data_Saved"), help="Raw global case root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./Data_Saved/Processed_ChannelThermal_Dataset"),
        help="Processed output root.",
    )
    parser.add_argument("--final-window-frames", type=int, default=None, help="Override save.final_window_frames.")
    parser.add_argument("--points-per-case", type=int, default=4096, help="Global sampled points per case; <=0 keeps all cells.")
    parser.add_argument("--max-modules", type=int, default=8, help="Pad module arrays to at least this module count.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Train split fraction for unsplit raw folders.")
    parser.add_argument("--seed", type=int, default=123, help="Sampling and split RNG seed.")
    return parser.parse_args()


def read_frame_index(case_dir: Path) -> List[Dict[str, str]]:
    index_path = case_dir / "frame_index.csv"
    if not index_path.exists():
        return []
    with index_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def discover_raw_cases(input_root: Path) -> List[RawCase]:
    records: List[RawCase] = []
    for split_hint, case_dir in find_case_dirs(input_root):
        scene_dir = case_dir / "scene"
        if not scene_dir.exists() or not list(scene_dir.glob("frame_*.npz")):
            continue
        payload = read_json(case_dir / "case_config.json")
        cfg = config_from_dict(payload)
        rows = read_frame_index(case_dir)
        if not rows:
            continue
        records.append(RawCase(split_hint=split_hint, case_dir=case_dir, cfg=cfg, cfg_payload=payload, frame_rows=rows))
    return records


def select_final_window(raw: RawCase, final_window_override: int | None) -> List[Dict[str, str]]:
    """Select final frames after heat activation, with robust fallbacks."""
    heat_start = float(raw.cfg.thermal.heat_start_time)
    eligible: List[Dict[str, str]] = []
    for row in raw.frame_rows:
        time_value = float(row.get("time", "0.0"))
        heat_active = int(float(row.get("heat_active", "1"))) == 1
        if heat_active and time_value >= heat_start:
            eligible.append(row)
    if not eligible:
        eligible = [row for row in raw.frame_rows if int(float(row.get("warmup_complete", "1"))) == 1]
    if not eligible:
        eligible = list(raw.frame_rows)
    window = int(final_window_override or raw.cfg.save.final_window_frames)
    return eligible[-max(1, window) :]


def load_frame(case_dir: Path, row: Dict[str, str]) -> Dict[str, np.ndarray]:
    file_name = row.get("file") or f"frame_{int(row['saved_frame']):06d}.npz"
    frame_path = case_dir / "scene" / file_name
    with np.load(frame_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def stack_field_window(case_dir: Path, selected_rows: Sequence[Dict[str, str]]) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    frames: List[np.ndarray] = []
    last_payload: Dict[str, np.ndarray] = {}
    times: List[float] = []
    for row in selected_rows:
        payload = load_frame(case_dir, row)
        last_payload = payload
        channels = [payload[name].astype(np.float32) for name in CHANNEL_ORDER]
        frames.append(np.stack(channels, axis=-1))
        times.append(float(row.get("time", "0.0")))
    return np.stack(frames, axis=0), last_payload, np.asarray(times, dtype=np.float32)


def choose_split(case_index: int, raw: RawCase, split_assignments: Dict[Path, str]) -> str:
    if raw.split_hint in {"train", "test"}:
        return raw.split_hint
    return split_assignments.get(raw.case_dir, "train")


def assign_unsplit_cases(raw_cases: Sequence[RawCase], train_fraction: float, seed: int) -> Dict[Path, str]:
    unsplit = [raw.case_dir for raw in raw_cases if raw.split_hint not in {"train", "test"}]
    if not unsplit:
        return {}
    rng = np.random.default_rng(seed)
    order = list(unsplit)
    rng.shuffle(order)
    if len(order) == 1:
        train_count = 1
    else:
        train_count = int(round(np.clip(train_fraction, 0.0, 1.0) * len(order)))
        train_count = min(max(train_count, 1), len(order) - 1)
    return {path: ("train" if idx < train_count else "test") for idx, path in enumerate(order)}


def sample_global_points(
    steady_field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    cfg: SimulationConfig,
    module_mask: np.ndarray,
    points_per_case: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample uniform, near-module, and temperature-gradient-focused points."""
    h, w, _ = steady_field.shape
    yy, xx = np.indices((h, w))
    flat_indices = np.arange(h * w)
    if points_per_case <= 0 or points_per_case >= h * w:
        chosen = flat_indices
    else:
        n_uniform = points_per_case // 2
        n_near = points_per_case // 4
        n_grad = points_per_case - n_uniform - n_near

        chosen_parts: List[np.ndarray] = [rng.choice(flat_indices, size=n_uniform, replace=h * w < n_uniform)]

        near_mask = np.zeros((h, w), dtype=bool)
        radius = float(cfg.domain.module_radius) + 2.0 * float(cfg.domain.min_gap)
        for cx, cy in cfg.layout.centers or []:
            near_mask |= np.hypot(x_grid - float(cx), y_grid - float(cy)) <= radius
        near_candidates = np.flatnonzero(near_mask & ~module_mask)
        if len(near_candidates) > 0 and n_near > 0:
            chosen_parts.append(rng.choice(near_candidates, size=n_near, replace=len(near_candidates) < n_near))

        temp = steady_field[..., CHANNEL_ORDER.index("temperature")]
        grad_y, grad_x = np.gradient(temp)
        grad_mag = np.hypot(grad_x, grad_y)
        grad_candidates = np.flatnonzero(grad_mag >= np.quantile(grad_mag, 0.80))
        if len(grad_candidates) > 0 and n_grad > 0:
            weights = grad_mag.reshape(-1)[grad_candidates].astype(np.float64)
            weights = weights + 1e-8
            weights = weights / np.sum(weights)
            chosen_parts.append(rng.choice(grad_candidates, size=n_grad, replace=len(grad_candidates) < n_grad, p=weights))

        chosen = np.unique(np.concatenate(chosen_parts))
        if len(chosen) < points_per_case:
            fill = rng.choice(flat_indices, size=points_per_case - len(chosen), replace=h * w < points_per_case)
            chosen = np.concatenate([chosen, fill])
        chosen = chosen[:points_per_case]

    jj = yy.reshape(-1)[chosen]
    ii = xx.reshape(-1)[chosen]
    samples = np.zeros((len(chosen), len(SAMPLED_POINT_FEATURES)), dtype=np.float32)
    samples[:, 0] = x_grid[jj, ii]
    samples[:, 1] = y_grid[jj, ii]
    samples[:, 2:] = steady_field[jj, ii, :]
    return samples


def unique_case_key(base_key: str, existing: set[str]) -> str:
    key = base_key
    suffix = 1
    while key in existing:
        suffix += 1
        key = f"{base_key}_{suffix}"
    existing.add(key)
    return key


def process_case(
    raw: RawCase,
    case_key: str,
    split: str,
    final_window_override: int | None,
    points_per_case: int,
    seed: int,
) -> ProcessedCase:
    selected_rows = select_final_window(raw, final_window_override)
    tensor, last_payload, selected_times = stack_field_window(raw.case_dir, selected_rows)
    steady_field = np.mean(tensor, axis=0).astype(np.float32)
    rms_field = np.sqrt(np.mean((tensor - steady_field[None, ...]) ** 2, axis=0)).astype(np.float32)
    x_grid, y_grid = build_uniform_grid(raw.cfg)
    module_mask = last_payload["module_mask"].astype(np.uint8)
    rng = np.random.default_rng(seed)
    sampled_points = sample_global_points(steady_field, x_grid, y_grid, raw.cfg, module_mask.astype(bool), points_per_case, rng)

    internal_frames: List[np.ndarray] = []
    interface_frames: List[np.ndarray] = []
    for row in selected_rows:
        payload = load_frame(raw.case_dir, row)
        internal_frames.append(payload["module_internal_temperature"].astype(np.float32))
        interface_frames.append(payload["interface_response"].astype(np.float32))

    internal_temperature = np.mean(np.stack(internal_frames, axis=0), axis=0).astype(np.float32)
    interface_response = np.mean(np.stack(interface_frames, axis=0), axis=0).astype(np.float32)
    internal_mask = last_payload["module_internal_mask"].astype(np.uint8)
    feature_names = tuple(name.decode("utf-8") for name in last_payload["interface_feature_names"])
    centers = np.asarray(raw.cfg.layout.centers or [], dtype=np.float32)
    heat_powers = np.asarray(raw.cfg.layout.heat_powers or [], dtype=np.float32)
    return ProcessedCase(
        case_key=case_key,
        split=split,
        case_dir=raw.case_dir,
        cfg=raw.cfg,
        cfg_payload=raw.cfg_payload,
        selected_times=selected_times,
        x_grid=x_grid.astype(np.float32),
        y_grid=y_grid.astype(np.float32),
        steady_field=steady_field,
        rms_field=rms_field,
        sampled_points=sampled_points,
        module_internal_temperature=internal_temperature,
        module_internal_mask=internal_mask,
        interface_response=interface_response,
        interface_feature_names=feature_names,
        module_centers=centers,
        heat_powers=heat_powers,
        module_mask=module_mask,
    )


def pad_first_axis(array: np.ndarray, target: int, fill_value: float = 0.0) -> np.ndarray:
    shape = (target,) + tuple(array.shape[1:])
    output = np.full(shape, fill_value, dtype=array.dtype)
    output[: min(target, array.shape[0])] = array[:target]
    return output


def write_global_case_index(output_root: Path, processed: Sequence[ProcessedCase]) -> None:
    index_path = output_root / "global_case_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["case_key", "split", "case_dir", "num_modules", "re", "heat_power_min", "heat_power_max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in processed:
            writer.writerow(
                {
                    "case_key": item.case_key,
                    "split": item.split,
                    "case_dir": str(item.case_dir),
                    "num_modules": item.cfg.layout.num_modules,
                    "re": item.cfg.flow.re,
                    "heat_power_min": float(np.min(item.heat_powers)) if len(item.heat_powers) else 0.0,
                    "heat_power_max": float(np.max(item.heat_powers)) if len(item.heat_powers) else 0.0,
                }
            )


def write_h5(output_root: Path, processed: Sequence[ProcessedCase], max_modules_arg: int) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    h5_path = output_root / "packed_dataset.h5"
    max_modules = max(max_modules_arg, max((item.module_centers.shape[0] for item in processed), default=0))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["dataset_type"] = "channelthermal_steady"
        h5.attrs["field_dim"] = len(CHANNEL_ORDER)
        h5.attrs["max_modules"] = max_modules
        h5.attrs["state_id"] = "steady_final_window"
        h5.create_dataset("field_dim", data=np.asarray([len(CHANNEL_ORDER)], dtype=np.int32))
        h5.create_dataset("channel_order", data=np.asarray(CHANNEL_ORDER, dtype=string_dtype))
        h5.create_dataset("sampled_point_feature_names", data=np.asarray(SAMPLED_POINT_FEATURES, dtype=string_dtype))
        if processed:
            h5.create_dataset("interface_feature_names", data=np.asarray(processed[0].interface_feature_names, dtype=string_dtype))
        h5.create_dataset("case_ids", data=np.asarray([item.case_key for item in processed], dtype=string_dtype))
        h5.create_dataset("splits", data=np.asarray([item.split for item in processed], dtype=string_dtype))

        cases_group = h5.create_group("cases")
        for item in processed:
            group = cases_group.create_group(item.case_key)
            group.attrs["split"] = item.split
            group.attrs["source_case_dir"] = str(item.case_dir)
            group.attrs["field_dim"] = len(CHANNEL_ORDER)
            group.attrs["channel_order"] = ",".join(CHANNEL_ORDER)
            group.create_dataset("x_grid", data=item.x_grid, compression="gzip")
            group.create_dataset("y_grid", data=item.y_grid, compression="gzip")
            group.create_dataset("steady_field", data=item.steady_field, compression="gzip")
            group.create_dataset("rms_field", data=item.rms_field, compression="gzip")
            group.create_dataset("sampled_points", data=item.sampled_points, compression="gzip")
            group.create_dataset("selected_times", data=item.selected_times)
            group.create_dataset("steady_time", data=np.asarray([float(np.mean(item.selected_times))], dtype=np.float32))
            group.create_dataset("module_mask", data=item.module_mask, compression="gzip")
            group.create_dataset("module_internal_mask", data=item.module_internal_mask, compression="gzip")
            group.create_dataset(
                "module_internal_temperature",
                data=pad_first_axis(item.module_internal_temperature, max_modules),
                compression="gzip",
            )
            group.create_dataset(
                "interface_response",
                data=pad_first_axis(item.interface_response, max_modules),
                compression="gzip",
            )
            centers = pad_first_axis(item.module_centers.reshape((-1, 2)), max_modules)
            powers = pad_first_axis(item.heat_powers.reshape((-1, 1)), max_modules).reshape((max_modules,))
            present = np.zeros((max_modules,), dtype=np.uint8)
            present[: min(max_modules, item.module_centers.shape[0])] = 1
            group.create_dataset("module_centers", data=centers)
            group.create_dataset("heat_powers", data=powers)
            group.create_dataset("module_present", data=present)
            group.create_dataset("case_config_json", data=json.dumps(item.cfg_payload, indent=2), dtype=string_dtype)

            materials = group.create_group("material_parameters")
            materials.attrs["re"] = float(item.cfg.flow.re)
            materials.attrs["u_in"] = float(item.cfg.flow.u_in)
            materials.attrs["nu"] = float(kinematic_viscosity(item.cfg))
            materials.attrs["solid_alpha"] = float(item.cfg.thermal.solid_alpha)
            materials.attrs["fluid_alpha"] = float(item.cfg.thermal.fluid_alpha)
            materials.attrs["solid_k"] = float(item.cfg.thermal.solid_k)
            materials.attrs["fluid_k"] = float(item.cfg.thermal.fluid_k)
            materials.attrs["module_radius"] = float(item.cfg.domain.module_radius)
    write_global_case_index(output_root, processed)
    return h5_path


def main() -> int:
    args = parse_args()
    input_root = resolve_data_path(args.input_root)
    output_root = resolve_data_path(args.output_root)
    raw_cases = discover_raw_cases(input_root)
    if not raw_cases:
        tqdm.write(f"No raw channel thermal cases found under: {input_root}")
        return 1

    split_assignments = assign_unsplit_cases(raw_cases, args.train_fraction, args.seed)
    processed: List[ProcessedCase] = []
    existing_keys: set[str] = set()
    for idx, raw in enumerate(tqdm(raw_cases, desc="Preprocessing cases", unit="case", dynamic_ncols=True)):
        base_key = str(raw.cfg.save.case_id) or raw.case_dir.name
        case_key = unique_case_key(base_key, existing_keys)
        split = choose_split(idx, raw, split_assignments)
        processed.append(
            process_case(
                raw,
                case_key,
                split,
                args.final_window_frames,
                args.points_per_case,
                args.seed + idx,
            )
        )

    h5_path = write_h5(output_root, processed, args.max_modules)
    tqdm.write(f"Packed {len(processed)} channel thermal cases into: {h5_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
