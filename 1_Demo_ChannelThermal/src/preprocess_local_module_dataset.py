"""Pack raw local module conduction cases into a leakage-free HDF5 dataset.

Scope
-----
This script handles the **local module** data layer for Demo 1. It reads raw
cases produced by ``simulate_local_module_thermal.py`` from
``Data_Saved/LocalModule_Raw`` and writes the canonical Stage-A dataset:

``Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5``.

Generated HDF5 structure
------------------------
The root stores uniform, stacked arrays for direct surrogate training:

* ``case_ids`` and ``splits`` list the cases and train/test assignments.
* ``module_param_names`` names ``module_params``:
  ``q_internal, solid_k, solid_alpha, h_mean, h_std, T_env_mean, T_env_std``.
* ``port_input_feature_names`` names each row of ``port_tokens``:
  ``theta, cos_theta, sin_theta, T_env, h``.
* ``interface_target_names`` names each row of ``interface_targets``:
  ``T_surface, q_normal``.
* ``local_target_stat_names`` names diagnostic summaries in
  ``local_target_stats``.
* ``normalization/`` stores dataset-level means and standard deviations.

The same per-case arrays are also copied under ``cases/<case_key>/`` together
with ``case_config_json`` and source-path metadata.

Physical meaning
----------------
Each case is a steady 2-D conduction solve on a single circular solid module.
``q_internal`` is the known internal heat generation strength, while
``solid_k`` and ``solid_alpha`` are material properties. ``port_tokens``
describe Robin boundary conditions around the module perimeter: angular
location ``theta``, its sinusoidal embedding, outside/environment temperature
``T_env``, and local heat-transfer coefficient ``h``.

The learning targets are deliberately separate. ``internal_query_points`` are
coordinates inside the disk, and ``internal_temperature_targets`` are the solved
temperatures at those coordinates. ``interface_targets`` are the solved module
surface temperature ``T_surface`` and outward normal heat flux ``q_normal``.
``local_grid`` and ``local_mask`` preserve the full square grid and disk mask so
visualization and reconstruction use the same geometry as the solver.

Data contract
-------------
The packed file separates known inputs from solved targets:

* ``module_params`` contains only known-before-solve scalar inputs.
* ``port_tokens`` contains only boundary/interface condition inputs.
* ``internal_temperature_targets`` and ``interface_targets`` contain solved
  temperatures and fluxes.
* ``local_target_stats`` stores solved-temperature summaries for analysis, not
  model inputs.

Backward safety
---------------
Older raw files may contain ``T_surface`` and ``q_normal`` inside
``port_tokens``. The preprocessor detects that legacy schema, prints a warning,
and strips the target columns before writing the HDF5 file.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("preprocess_local_module_dataset.py requires h5py.") from exc

import _bootstrap_imports 
from channelthermal_common import config_from_dict, find_case_dirs, read_json, resolve_data_path


MODULE_PARAM_NAMES = (
    "q_internal",
    "solid_k",
    "solid_alpha",
    "h_mean",
    "h_std",
    "T_env_mean",
    "T_env_std",
)
PORT_INPUT_FEATURE_NAMES = ("theta", "cos_theta", "sin_theta", "T_env", "h")
INTERFACE_TARGET_NAMES = ("T_surface", "q_normal")
LOCAL_TARGET_STAT_NAMES = ("T_mean", "T_max", "T_min", "T_std")
LOCAL_TARGET_ROUGHNESS_NAMES = (
    "roughness_T_surface",
    "roughness_q_normal",
    "highfreq_ratio_T_surface",
    "highfreq_ratio_q_normal",
)


@dataclass
class LocalRawCase:
    """A raw local case discovered under the input root."""

    split_hint: str
    case_dir: Path
    cfg_payload: Dict[str, Any]
    payload: Dict[str, np.ndarray]


@dataclass
class LocalProcessedCase:
    """Processed arrays for one leakage-free local surrogate case."""

    case_key: str
    split: str
    case_dir: Path
    cfg_payload: Dict[str, Any]
    module_params: np.ndarray
    local_target_stats: np.ndarray
    port_tokens: np.ndarray
    internal_query_points: np.ndarray
    internal_temperature_targets: np.ndarray
    interface_targets: np.ndarray
    interface_targets_raw: np.ndarray | None
    local_target_roughness: np.ndarray
    local_grid: np.ndarray
    local_mask: np.ndarray
    solver_type: str
    n_active_modes: int
    effective_conductivity: float
    module_radius: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack local module thermal cases into HDF5.")
    parser.add_argument("--input-root", type=Path, default=Path("./Data_Saved/LocalModule_Raw"), help="Raw local case root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./Data_Saved/Processed_LocalModule_Dataset"),
        help="Processed local dataset root.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.90, help="Train split fraction for unsplit raw folders.")
    parser.add_argument("--seed", type=int, default=321, help="Split RNG seed.")
    parser.add_argument("--smooth-interface-targets", action="store_true", help="Optional ablation: smooth interface targets before writing interface_targets.")
    parser.add_argument("--interface-smooth-modes", type=int, default=6, help="Number of low Fourier modes to keep when smoothing is enabled.")
    return parser.parse_args()


def decode_names(values: np.ndarray) -> List[str]:
    """Decode HDF5/NPZ byte-string feature-name arrays."""
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def discover_local_cases(input_root: Path) -> List[LocalRawCase]:
    """Discover raw local cases and load their single-frame payloads."""
    cases: List[LocalRawCase] = []
    for split_hint, case_dir in find_case_dirs(input_root):
        solution_path = case_dir / "local_solution.npz"
        if not solution_path.exists():
            frame_path = case_dir / "scene" / "frame_000000.npz"
            if not frame_path.exists():
                continue
            solution_path = frame_path
        with np.load(solution_path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        cfg_payload = read_json(case_dir / "case_config.json")
        cases.append(LocalRawCase(split_hint=split_hint, case_dir=case_dir, cfg_payload=cfg_payload, payload=payload))
    return cases


def assign_splits(raw_cases: Sequence[LocalRawCase], train_fraction: float, seed: int) -> Dict[Path, str]:
    """Assign train/test splits for unsplit raw folders.

    If the input root already contains explicit ``train/`` or ``test/``
    subfolders, those labels are preserved. Otherwise, a deterministic random
    split is made from case directories.
    """
    unsplit = [raw.case_dir for raw in raw_cases if raw.split_hint not in {"train", "test"}]
    assignments: Dict[Path, str] = {}
    for raw in raw_cases:
        if raw.split_hint in {"train", "test"}:
            assignments[raw.case_dir] = raw.split_hint
    if unsplit:
        order = list(unsplit)
        random.Random(seed).shuffle(order)
        if len(order) == 1:
            train_count = 1
        else:
            train_count = int(round(np.clip(train_fraction, 0.0, 1.0) * len(order)))
            train_count = min(max(train_count, 1), len(order) - 1)
        for idx, path in enumerate(order):
            assignments[path] = "train" if idx < train_count else "test"
    return assignments


def unique_case_key(base_key: str, existing: set[str]) -> str:
    """Keep HDF5 case group names unique even when configs reuse case IDs."""
    key = base_key
    suffix = 1
    while key in existing:
        suffix += 1
        key = f"{base_key}_{suffix}"
    existing.add(key)
    return key


def read_port_tokens(payload: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Return leakage-free port input tokens and names.

    New raw files write ``port_input_feature_names``. Legacy raw files only have
    ``port_feature_names`` and may include solved target columns; those target
    columns are stripped here.
    """
    port_tokens = payload["port_tokens"].astype(np.float32)
    if "port_input_feature_names" in payload:
        feature_names = tuple(decode_names(payload["port_input_feature_names"]))
    else:
        feature_names = tuple(decode_names(payload.get("port_feature_names", np.asarray(PORT_INPUT_FEATURE_NAMES, dtype="S"))))

    target_columns = {"T_surface", "q_normal"}
    if any(name in target_columns for name in feature_names):
        print("Detected legacy port_tokens with target columns; stripping T_surface/q_normal from inputs.")
        keep_indices = [idx for idx, name in enumerate(feature_names) if name not in target_columns]
        port_tokens = port_tokens[:, keep_indices]
        feature_names = tuple(feature_names[idx] for idx in keep_indices)

    if feature_names != PORT_INPUT_FEATURE_NAMES:
        missing = [name for name in PORT_INPUT_FEATURE_NAMES if name not in feature_names]
        if missing:
            raise ValueError(f"port_tokens are missing required input features: {missing}")
        reorder = [feature_names.index(name) for name in PORT_INPUT_FEATURE_NAMES]
        port_tokens = port_tokens[:, reorder]
    return port_tokens.astype(np.float32), PORT_INPUT_FEATURE_NAMES


def read_interface_targets(payload: Dict[str, np.ndarray]) -> np.ndarray:
    """Return solved interface targets in canonical order."""
    if "interface_targets" in payload:
        targets = payload["interface_targets"].astype(np.float32)
        if "interface_target_names" in payload:
            names = tuple(decode_names(payload["interface_target_names"]))
            if names != INTERFACE_TARGET_NAMES:
                reorder = [names.index(name) for name in INTERFACE_TARGET_NAMES]
                targets = targets[:, reorder]
        return targets.astype(np.float32)

    return np.stack([payload["T_surface"], payload["q_normal"]], axis=-1).astype(np.float32)


def periodic_curve_roughness(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return 0.0
    diff = arr - np.roll(arr, 1)
    scale = max(float(np.std(arr)), 1.0e-8)
    return float(np.sqrt(np.mean(diff * diff)) / scale)


def highfreq_ratio(values: np.ndarray, keep_modes: int = 6) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        return 0.0
    coeff = np.fft.rfft(arr - np.mean(arr))
    power = np.abs(coeff) ** 2
    total = float(np.sum(power[1:]))
    if total <= 1.0e-20:
        return 0.0
    cutoff = min(max(int(keep_modes), 0), power.shape[0] - 1)
    high = float(np.sum(power[cutoff + 1 :]))
    return high / total


def interface_roughness_metrics(interface_targets: np.ndarray, smooth_modes: int = 6) -> np.ndarray:
    return np.asarray(
        [
            periodic_curve_roughness(interface_targets[:, 0]),
            periodic_curve_roughness(interface_targets[:, 1]),
            highfreq_ratio(interface_targets[:, 0], keep_modes=smooth_modes),
            highfreq_ratio(interface_targets[:, 1], keep_modes=smooth_modes),
        ],
        dtype=np.float32,
    )


def smooth_periodic_curve(values: np.ndarray, keep_modes: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    coeff = np.fft.rfft(arr)
    cutoff = min(max(int(keep_modes), 0), coeff.shape[0] - 1)
    coeff[cutoff + 1 :] = 0.0
    return np.fft.irfft(coeff, n=arr.shape[0]).astype(np.float32)


def smooth_interface_targets(interface_targets: np.ndarray, keep_modes: int) -> np.ndarray:
    smoothed = np.empty_like(interface_targets, dtype=np.float32)
    for idx in range(interface_targets.shape[-1]):
        smoothed[:, idx] = smooth_periodic_curve(interface_targets[:, idx], keep_modes)
    return smoothed


def payload_scalar(payload: Dict[str, np.ndarray], key: str, fallback: Any) -> Any:
    if key not in payload:
        return fallback
    value = payload[key]
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
    else:
        item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return item


def process_local_case(
    raw: LocalRawCase,
    case_key: str,
    split: str,
    *,
    smooth_interface: bool = False,
    smooth_modes: int = 6,
) -> LocalProcessedCase:
    """Convert one raw case into leakage-free arrays for HDF5 writing."""
    cfg = config_from_dict(raw.cfg_payload)
    payload = raw.payload
    local_x = payload["local_x"].astype(np.float32)
    local_y = payload["local_y"].astype(np.float32)
    local_grid = np.stack([local_x, local_y], axis=-1)
    local_mask = payload["disk_mask"].astype(np.uint8)
    temperature = payload["temperature"].astype(np.float32)
    mask_bool = local_mask.astype(bool)
    internal_query_points = local_grid[mask_bool].astype(np.float32)
    internal_temperature_targets = temperature[mask_bool].astype(np.float32)
    h = payload["h"].astype(np.float32)
    t_env = payload["T_env"].astype(np.float32)
    q_internal = float(payload["q_internal"][0])

    # Model input parameters: every value below is known before solving.
    module_params = np.asarray(
        [
            q_internal,
            float(cfg.thermal.solid_k),
            float(cfg.thermal.solid_alpha),
            float(np.mean(h)),
            float(np.std(h)),
            float(np.mean(t_env)),
            float(np.std(t_env)),
        ],
        dtype=np.float32,
    )

    # Target-derived summaries are useful diagnostics but must not be mixed with
    # the model input vector.
    local_target_stats = np.asarray(
        [
            float(np.mean(internal_temperature_targets)),
            float(np.max(internal_temperature_targets)),
            float(np.min(internal_temperature_targets)),
            float(np.std(internal_temperature_targets)),
        ],
        dtype=np.float32,
    )
    port_tokens, _ = read_port_tokens(payload)
    raw_interface_targets = read_interface_targets(payload)
    roughness = interface_roughness_metrics(raw_interface_targets, smooth_modes=smooth_modes)
    if smooth_interface:
        interface_targets = smooth_interface_targets(raw_interface_targets, keep_modes=smooth_modes)
        interface_targets_raw = raw_interface_targets
    else:
        interface_targets = raw_interface_targets
        interface_targets_raw = None

    local_solution = raw.cfg_payload.get("local_solution", {}) if isinstance(raw.cfg_payload, dict) else {}
    solver_type = str(payload_scalar(payload, "solver_type", local_solution.get("solver_type", cfg.local_module.solver_type)))
    n_active_modes = int(payload_scalar(payload, "n_active_modes", local_solution.get("n_active_modes", cfg.local_module.n_boundary_modes)))
    effective_conductivity = float(payload_scalar(payload, "effective_conductivity", local_solution.get("effective_conductivity", 0.0)))
    module_radius = float(payload_scalar(payload, "module_radius", local_solution.get("module_radius", cfg.domain.module_radius)))

    return LocalProcessedCase(
        case_key=case_key,
        split=split,
        case_dir=raw.case_dir,
        cfg_payload=raw.cfg_payload,
        module_params=module_params,
        local_target_stats=local_target_stats,
        port_tokens=port_tokens,
        internal_query_points=internal_query_points,
        internal_temperature_targets=internal_temperature_targets,
        interface_targets=interface_targets,
        interface_targets_raw=interface_targets_raw,
        local_target_roughness=roughness,
        local_grid=local_grid.astype(np.float32),
        local_mask=local_mask,
        solver_type=solver_type,
        n_active_modes=n_active_modes,
        effective_conductivity=effective_conductivity,
        module_radius=module_radius,
    )


def validate_uniform_shapes(processed: Sequence[LocalProcessedCase]) -> None:
    """Ensure root-level stacked arrays can be written without ragged storage."""
    first = processed[0]
    expected = {
        "module_params": first.module_params.shape,
        "local_target_stats": first.local_target_stats.shape,
        "port_tokens": first.port_tokens.shape,
        "internal_query_points": first.internal_query_points.shape,
        "internal_temperature_targets": first.internal_temperature_targets.shape,
        "interface_targets": first.interface_targets.shape,
        "local_target_roughness": first.local_target_roughness.shape,
        "local_grid": first.local_grid.shape,
        "local_mask": first.local_mask.shape,
    }
    for item in processed[1:]:
        for name, shape in expected.items():
            if getattr(item, name).shape != shape:
                raise ValueError(
                    f"Local cases must share shapes for packed root arrays. "
                    f"{item.case_key} has {name}{getattr(item, name).shape}, expected {shape}."
                )


def write_index_csv(output_root: Path, processed: Sequence[LocalProcessedCase]) -> None:
    """Write a human-readable local case index next to the HDF5 file."""
    with (output_root / "local_case_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_key",
                "split",
                "case_dir",
                "q_internal",
                "solver_type",
                "n_active_modes",
                "roughness_T_surface",
                "roughness_q_normal",
            ],
        )
        writer.writeheader()
        for item in processed:
            writer.writerow(
                {
                    "case_key": item.case_key,
                    "split": item.split,
                    "case_dir": str(item.case_dir),
                    "q_internal": float(item.module_params[0]),
                    "solver_type": item.solver_type,
                    "n_active_modes": int(item.n_active_modes),
                    "roughness_T_surface": float(item.local_target_roughness[0]),
                    "roughness_q_normal": float(item.local_target_roughness[1]),
                }
            )


def write_normalization_group(h5, processed: Sequence[LocalProcessedCase]) -> None:
    """Save dataset-level statistics used by later training scripts."""
    norm = h5.create_group("normalization")
    module_params = np.stack([item.module_params for item in processed])
    port_tokens = np.stack([item.port_tokens for item in processed])
    internal_targets = np.concatenate([item.internal_temperature_targets.reshape(-1) for item in processed])
    interface_targets = np.stack([item.interface_targets for item in processed])
    roughness = np.stack([item.local_target_roughness for item in processed])

    norm.create_dataset("module_params_mean", data=np.mean(module_params, axis=0).astype(np.float32))
    norm.create_dataset("module_params_std", data=np.std(module_params, axis=0).astype(np.float32))
    norm.create_dataset("port_tokens_mean", data=np.mean(port_tokens.reshape(-1, port_tokens.shape[-1]), axis=0).astype(np.float32))
    norm.create_dataset("port_tokens_std", data=np.std(port_tokens.reshape(-1, port_tokens.shape[-1]), axis=0).astype(np.float32))
    norm.create_dataset("internal_temperature_mean", data=np.asarray([np.mean(internal_targets)], dtype=np.float32))
    norm.create_dataset("internal_temperature_std", data=np.asarray([np.std(internal_targets)], dtype=np.float32))
    norm.create_dataset(
        "interface_targets_mean",
        data=np.mean(interface_targets.reshape(-1, interface_targets.shape[-1]), axis=0).astype(np.float32),
    )
    norm.create_dataset(
        "interface_targets_std",
        data=np.std(interface_targets.reshape(-1, interface_targets.shape[-1]), axis=0).astype(np.float32),
    )
    norm.create_dataset("local_target_roughness_mean", data=np.mean(roughness, axis=0).astype(np.float32))
    norm.create_dataset("local_target_roughness_std", data=np.std(roughness, axis=0).astype(np.float32))


def write_h5(output_root: Path, processed: Sequence[LocalProcessedCase]) -> Path:
    """Write the packed local module HDF5 dataset."""
    output_root.mkdir(parents=True, exist_ok=True)
    validate_uniform_shapes(processed)
    h5_path = output_root / "packed_dataset.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    local_grid_size = int(processed[0].local_mask.shape[0])
    n_interface_points = int(processed[0].port_tokens.shape[0])

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["dataset_type"] = "local_module_steady_conduction"
        h5.attrs["dataset_role"] = "local_module_surrogate"
        h5.attrs["target_kind"] = "steady_robin_conduction"
        h5.attrs["num_cases"] = len(processed)
        h5.attrs["local_grid_size"] = local_grid_size
        h5.attrs["n_interface_points"] = n_interface_points
        h5.create_dataset("case_ids", data=np.asarray([item.case_key for item in processed], dtype=string_dtype))
        h5.create_dataset("splits", data=np.asarray([item.split for item in processed], dtype=string_dtype))
        h5.create_dataset("module_param_names", data=np.asarray(MODULE_PARAM_NAMES, dtype=string_dtype))
        h5.create_dataset("port_input_feature_names", data=np.asarray(PORT_INPUT_FEATURE_NAMES, dtype=string_dtype))
        h5.create_dataset("port_feature_names", data=np.asarray(PORT_INPUT_FEATURE_NAMES, dtype=string_dtype))
        h5.create_dataset("interface_target_names", data=np.asarray(INTERFACE_TARGET_NAMES, dtype=string_dtype))
        h5.create_dataset("local_target_stat_names", data=np.asarray(LOCAL_TARGET_STAT_NAMES, dtype=string_dtype))
        h5.create_dataset("local_target_roughness_names", data=np.asarray(LOCAL_TARGET_ROUGHNESS_NAMES, dtype=string_dtype))
        h5.create_dataset("module_params", data=np.stack([item.module_params for item in processed]), compression="gzip")
        h5.create_dataset("local_target_stats", data=np.stack([item.local_target_stats for item in processed]), compression="gzip")
        h5.create_dataset("local_target_roughness", data=np.stack([item.local_target_roughness for item in processed]), compression="gzip")
        h5.create_dataset("port_tokens", data=np.stack([item.port_tokens for item in processed]), compression="gzip")
        h5.create_dataset(
            "internal_query_points",
            data=np.stack([item.internal_query_points for item in processed]),
            compression="gzip",
        )
        h5.create_dataset(
            "internal_temperature_targets",
            data=np.stack([item.internal_temperature_targets for item in processed]),
            compression="gzip",
        )
        h5.create_dataset("interface_targets", data=np.stack([item.interface_targets for item in processed]), compression="gzip")
        if any(item.interface_targets_raw is not None for item in processed):
            h5.attrs["interface_targets_smoothed"] = True
            h5.create_dataset(
                "interface_targets_raw",
                data=np.stack(
                    [
                        item.interface_targets_raw if item.interface_targets_raw is not None else item.interface_targets
                        for item in processed
                    ]
                ),
                compression="gzip",
            )
        h5.create_dataset("solver_type", data=np.asarray([item.solver_type for item in processed], dtype=string_dtype))
        h5.create_dataset("n_active_modes", data=np.asarray([item.n_active_modes for item in processed], dtype=np.int32))
        h5.create_dataset("effective_conductivity", data=np.asarray([item.effective_conductivity for item in processed], dtype=np.float32))
        h5.create_dataset("module_radius", data=np.asarray([item.module_radius for item in processed], dtype=np.float32))
        h5.create_dataset("local_grid", data=np.stack([item.local_grid for item in processed]), compression="gzip")
        h5.create_dataset("local_mask", data=np.stack([item.local_mask for item in processed]), compression="gzip")
        write_normalization_group(h5, processed)

        cases_group = h5.create_group("cases")
        for item in processed:
            group = cases_group.create_group(item.case_key)
            group.attrs["split"] = item.split
            group.attrs["source_case_dir"] = str(item.case_dir)
            group.create_dataset("module_params", data=item.module_params)
            group.create_dataset("local_target_stats", data=item.local_target_stats)
            group.create_dataset("local_target_roughness", data=item.local_target_roughness)
            group.create_dataset("port_tokens", data=item.port_tokens, compression="gzip")
            group.create_dataset("internal_query_points", data=item.internal_query_points, compression="gzip")
            group.create_dataset("internal_temperature_targets", data=item.internal_temperature_targets, compression="gzip")
            group.create_dataset("interface_targets", data=item.interface_targets, compression="gzip")
            if item.interface_targets_raw is not None:
                group.create_dataset("interface_targets_raw", data=item.interface_targets_raw, compression="gzip")
            group.create_dataset("local_grid", data=item.local_grid, compression="gzip")
            group.create_dataset("local_mask", data=item.local_mask, compression="gzip")
            group.create_dataset("case_config_json", data=json.dumps(item.cfg_payload, indent=2), dtype=string_dtype)
            group.attrs["solver_type"] = item.solver_type
            group.attrs["n_active_modes"] = int(item.n_active_modes)
            group.attrs["effective_conductivity"] = float(item.effective_conductivity)
            group.attrs["module_radius"] = float(item.module_radius)

    write_index_csv(output_root, processed)
    return h5_path


def main() -> int:
    args = parse_args()
    input_root = resolve_data_path(args.input_root)
    output_root = resolve_data_path(args.output_root)
    raw_cases = discover_local_cases(input_root)
    if not raw_cases:
        tqdm.write(f"No local module cases found under: {input_root}")
        return 1

    assignments = assign_splits(raw_cases, args.train_fraction, args.seed)
    processed: List[LocalProcessedCase] = []
    existing: set[str] = set()
    for raw in tqdm(raw_cases, desc="Preprocessing local cases", unit="case", dynamic_ncols=True):
        base_key = str(config_from_dict(raw.cfg_payload).save.case_id) or raw.case_dir.name
        case_key = unique_case_key(base_key, existing)
        processed.append(
            process_local_case(
                raw,
                case_key,
                assignments.get(raw.case_dir, "train"),
                smooth_interface=bool(args.smooth_interface_targets),
                smooth_modes=int(args.interface_smooth_modes),
            )
        )

    h5_path = write_h5(output_root, processed)
    tqdm.write(f"Packed {len(processed)} local module cases into: {h5_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
