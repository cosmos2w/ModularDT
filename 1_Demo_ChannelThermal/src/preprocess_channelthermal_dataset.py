"""Pack raw global channel thermal cases into a steady/quasi-steady HDF5 dataset.

Scope
-----
This script handles the **global channel** data layer for Demo 1. It reads raw
case folders produced by ``simulate_channelthermal.py`` under ``Data_Saved`` and
writes the canonical Stage-B dataset:

``Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5``.

Generated HDF5 structure
------------------------
The root records dataset metadata and feature names:

* ``case_ids`` and ``splits`` list the cases and train/test assignments.
* ``channel_order`` names the field channels in ``steady_field`` and
  ``rms_field``: ``u, v, p, omega, temperature``.
* ``sampled_point_feature_names`` names rows of ``sampled_points``:
  ``x, y, u, v, p, omega, temperature``.
* ``interface_condition_feature_names`` and ``interface_target_names`` define
  the clean input/target split for module-boundary coupling.
* ``normalization/`` stores dataset-level means and standard deviations.

Each case is stored under ``cases/<case_key>/`` with:

* ``x_grid`` and ``y_grid``: fixed Eulerian coordinates of the channel domain.
* ``steady_field``: final-window mean flow/thermal state on the grid.
* ``rms_field``: final-window fluctuation magnitude for the same channels.
* ``sampled_points``: sparse point supervision sampled from ``steady_field``;
  by default, these are fluid-domain points outside module interiors.
* ``module_mask``: channel cells occupied by solid modules.
* ``module_internal_temperature`` and ``module_internal_mask``: local solid
  temperature fields and valid disk masks, padded to ``max_modules``.
* ``interface_response``: legacy full interface array preserved for
  compatibility.
* ``interface_condition``: known boundary-condition inputs at module surfaces.
* ``interface_target``: solved coupling targets at module surfaces.
* ``module_centers``, ``heat_powers``, and ``module_present``: padded module
  layout and source metadata.
* ``material_parameters`` and ``case_config_json``: physical constants and the
  original simulation configuration.

Physical meaning
----------------
The processed case represents a quasi-steady forced-convection problem: fluid
flows through a channel around internally heated circular solid modules. The
field channels describe velocity, pressure, vorticity, and temperature in the
global channel. ``heat_powers`` are known internal heat generation strengths.
``interface_condition`` carries quantities known from the fluid side and local
geometry, while ``interface_target`` carries solved solid/fluid exchange values:
surface temperature ``T_surface`` and outward normal heat flux ``q_normal``.
This separation keeps training inputs free of target leakage.

Data flow
---------
For each case, frames after heat activation are filtered, the final window is
averaged into ``steady_field``, and an ``rms_field`` is computed over that same
window. Global point samples, module-internal temperature targets, and
interface arrays are then packed for future neural-field training. Global point
sampling excludes module interiors by default because solid temperatures are
supervised separately through ``module_internal_temperature``.

Leakage guard
-------------
The raw ``interface_response`` is preserved for compatibility, but the packed
HDF5 also writes ``interface_condition`` and ``interface_target`` so future
training scripts can use clean inputs and targets by default.
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
INTERFACE_CONDITION_FEATURE_NAMES = (
    "theta",
    "normal_x",
    "normal_y",
    "T_outside",
    "u_normal",
    "u_tangent",
    "h_proxy",
    "h_effective",
)
INTERFACE_TARGET_NAMES = ("T_surface", "q_normal")


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
    interface_condition: np.ndarray
    h_effective_valid_mask: np.ndarray
    interface_target: np.ndarray
    interface_feature_names: Tuple[str, ...]
    module_centers: np.ndarray
    heat_powers: np.ndarray
    module_mask: np.ndarray
    exclude_module_interior_from_global_points: bool
    converged: bool
    converged_time: float
    converged_step: int
    final_delta_inf: float
    final_delta_l2_rel: float
    selected_frame_ids: np.ndarray
    packed_unconverged: bool
    h_effective_valid_fraction: float
    h_effective_clipped_fraction: float
    h_effective_mean: float


@dataclass
class SelectionResult:
    selected_rows: List[Dict[str, str]]
    converged: bool
    converged_time: float
    converged_step: int
    final_delta_inf: float
    final_delta_l2_rel: float
    packed_unconverged: bool
    skip_reason: str = ""


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
    parser.add_argument(
        "--exclude-module-interior-from-global-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample global point targets from fluid cells only; use --no-exclude-module-interior-from-global-points for old behavior.",
    )
    convergence_group = parser.add_mutually_exclusive_group()
    convergence_group.add_argument(
        "--require-converged",
        dest="require_converged",
        action="store_true",
        default=True,
        help="Skip raw cases without a converged saved frame (default).",
    )
    convergence_group.add_argument(
        "--allow-unconverged",
        dest="require_converged",
        action="store_false",
        help="Pack unconverged raw cases using the target-mode fallback.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["converged_final", "converged_window_mean", "final_window_legacy"],
        default="converged_final",
        help="How to select steady training target frames.",
    )
    parser.add_argument("--min-final-window-frames", type=int, default=1, help="Minimum selected frames for window targets.")
    parser.add_argument("--h-effective-eps", type=float, default=1.0e-3, help="Minimum |T_surface - T_outside| used for h_effective.")
    parser.add_argument("--h-effective-max", type=float, default=1.0e4, help="Upper clip for derived h_effective.")
    return parser.parse_args()


def read_frame_index(case_dir: Path) -> List[Dict[str, str]]:
    index_path = case_dir / "frame_index.csv"
    if not index_path.exists():
        return []
    with index_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def discover_raw_cases(input_root: Path) -> List[RawCase]:
    """Find raw global cases with frames and config metadata."""
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


def _row_bool(row: Dict[str, str], key: str, default: bool = False) -> bool:
    value = row.get(key, "")
    if value == "":
        return bool(default)
    try:
        return int(float(value)) != 0
    except ValueError:
        return str(value).strip().lower() in {"true", "yes", "y"}


def _row_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _runtime_float(runtime: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = runtime.get(key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _runtime_int(runtime: Dict[str, Any], key: str, default: int = -1) -> int:
    value = runtime.get(key, default)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _legacy_final_window(raw: RawCase, final_window_override: int | None) -> List[Dict[str, str]]:
    """Select final frames using the historical heat-active fallback behavior."""
    heat_start = float(raw.cfg.thermal.heat_start_time)
    eligible: List[Dict[str, str]] = []
    for row in raw.frame_rows:
        time_value = float(row.get("time", "0.0"))
        heat_active = _row_bool(row, "heat_active", True)
        if heat_active and time_value >= heat_start:
            eligible.append(row)
    if not eligible:
        eligible = [row for row in raw.frame_rows if _row_bool(row, "warmup_complete", True)]
    if not eligible:
        eligible = list(raw.frame_rows)
    window = int(final_window_override or raw.cfg.save.final_window_frames)
    return eligible[-max(1, window) :]


def _case_convergence_metadata(raw: RawCase) -> Tuple[bool, float, int, float, float]:
    runtime = raw.cfg_payload.get("runtime", {}) if isinstance(raw.cfg_payload, dict) else {}
    converged_rows = [row for row in raw.frame_rows if _row_bool(row, "converged", False)]
    if converged_rows:
        row = converged_rows[0]
        converged = True
        converged_time = _row_float(row, "time")
        converged_step = int(_row_float(row, "step", -1))
    else:
        converged = bool(runtime.get("converged", False))
        converged_time = _runtime_float(runtime, "converged_time")
        converged_step = _runtime_int(runtime, "converged_step")
    final_row = raw.frame_rows[-1] if raw.frame_rows else {}
    final_delta_inf = _runtime_float(runtime, "final_delta_inf", _row_float(final_row, "delta_inf"))
    final_delta_l2_rel = _runtime_float(runtime, "final_delta_l2_rel", _row_float(final_row, "delta_l2_rel"))
    return converged, converged_time, converged_step, final_delta_inf, final_delta_l2_rel


def select_final_window(
    raw: RawCase,
    final_window_override: int | None,
    *,
    target_mode: str,
    require_converged: bool,
    min_final_window_frames: int,
) -> SelectionResult:
    """Select target frames using convergence metadata by default."""
    converged, converged_time, converged_step, final_delta_inf, final_delta_l2_rel = _case_convergence_metadata(raw)
    min_frames = max(1, int(min_final_window_frames))

    if target_mode == "final_window_legacy":
        selected = _legacy_final_window(raw, final_window_override)
        return SelectionResult(
            selected_rows=selected,
            converged=converged,
            converged_time=converged_time,
            converged_step=converged_step,
            final_delta_inf=final_delta_inf,
            final_delta_l2_rel=final_delta_l2_rel,
            packed_unconverged=not converged,
        )

    converged_indices = [idx for idx, row in enumerate(raw.frame_rows) if _row_bool(row, "converged", False)]
    if not converged_indices and converged:
        if converged_step >= 0:
            step_matches = [idx for idx, row in enumerate(raw.frame_rows) if int(_row_float(row, "step", -1)) == converged_step]
            converged_indices.extend(step_matches)
        if not converged_indices and np.isfinite(converged_time):
            time_matches = [idx for idx, row in enumerate(raw.frame_rows) if _row_float(row, "time") >= converged_time]
            if time_matches:
                converged_indices.append(time_matches[0])
    if not converged_indices:
        if require_converged:
            return SelectionResult([], converged, converged_time, converged_step, final_delta_inf, final_delta_l2_rel, False, "unconverged")
        return SelectionResult(
            selected_rows=[raw.frame_rows[-1]],
            converged=converged,
            converged_time=converged_time,
            converged_step=converged_step,
            final_delta_inf=final_delta_inf,
            final_delta_l2_rel=final_delta_l2_rel,
            packed_unconverged=True,
        )

    converged_idx = converged_indices[0]
    if target_mode == "converged_final":
        selected = [raw.frame_rows[converged_idx]]
    elif target_mode == "converged_window_mean":
        window = max(min_frames, int(final_window_override or raw.cfg.save.final_window_frames))
        start = max(0, converged_idx - window + 1)
        selected = list(raw.frame_rows[start : converged_idx + 1])
        if len(selected) < min_frames:
            return SelectionResult([], converged, converged_time, converged_step, final_delta_inf, final_delta_l2_rel, False, "insufficient_window")
        if not all(_row_bool(row, "heat_active", False) for row in selected):
            return SelectionResult([], converged, converged_time, converged_step, final_delta_inf, final_delta_l2_rel, False, "window_contains_heat_inactive")
    else:
        raise ValueError(f"Unsupported target_mode={target_mode!r}.")

    return SelectionResult(
        selected_rows=selected,
        converged=converged,
        converged_time=converged_time,
        converged_step=converged_step,
        final_delta_inf=final_delta_inf,
        final_delta_l2_rel=final_delta_l2_rel,
        packed_unconverged=False,
    )


def load_frame(case_dir: Path, row: Dict[str, str]) -> Dict[str, np.ndarray]:
    file_name = row.get("file") or f"frame_{int(row['saved_frame']):06d}.npz"
    frame_path = case_dir / "scene" / file_name
    with np.load(frame_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def stack_field_window(case_dir: Path, selected_rows: Sequence[Dict[str, str]]) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Load all field channels for the selected final window."""
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
    """Assign deterministic train/test labels when raw data is not pre-split."""
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
    *,
    exclude_module_interior_from_global_points: bool = True,
) -> np.ndarray:
    """Sample uniform, near-module, and temperature-gradient-focused points.

    These samples become point-supervision tuples for the global environment
    neural field. Module-interior and interface targets are stored separately,
    so the default candidate universe is fluid cells only.
    """
    h, w, _ = steady_field.shape
    yy, xx = np.indices((h, w))
    flat_indices = np.arange(h * w)
    candidate_mask = np.ones((h, w), dtype=bool)
    if exclude_module_interior_from_global_points:
        candidate_mask &= ~np.asarray(module_mask, dtype=bool)
    candidate_indices = np.flatnonzero(candidate_mask.reshape(-1))
    if len(candidate_indices) == 0:
        print("Warning: no fluid cells available for global point sampling; falling back to all grid cells.")
        candidate_indices = flat_indices

    if points_per_case <= 0 or points_per_case >= len(candidate_indices):
        chosen = candidate_indices
    else:
        n_uniform = points_per_case // 2
        n_near = points_per_case // 4
        n_grad = points_per_case - n_uniform - n_near

        # Uniform, near-module, and gradient-focused samples all draw from the
        # same candidate universe, so the global field loss does not silently
        # include solid interior cells unless the compatibility flag is off.
        chosen_parts: List[np.ndarray] = [
            rng.choice(candidate_indices, size=n_uniform, replace=len(candidate_indices) < n_uniform)
        ]

        near_mask = np.zeros((h, w), dtype=bool)
        radius = float(cfg.domain.module_radius) + 2.0 * float(cfg.domain.min_gap)
        for cx, cy in cfg.layout.centers or []:
            near_mask |= np.hypot(x_grid - float(cx), y_grid - float(cy)) <= radius
        near_candidates = np.intersect1d(np.flatnonzero(near_mask.reshape(-1)), candidate_indices, assume_unique=False)
        if len(near_candidates) > 0 and n_near > 0:
            chosen_parts.append(rng.choice(near_candidates, size=n_near, replace=len(near_candidates) < n_near))

        temp = steady_field[..., CHANNEL_ORDER.index("temperature")]
        grad_y, grad_x = np.gradient(temp)
        grad_mag = np.hypot(grad_x, grad_y)
        grad_candidates = np.intersect1d(
            np.flatnonzero((grad_mag >= np.quantile(grad_mag, 0.80)).reshape(-1)),
            candidate_indices,
            assume_unique=False,
        )
        if len(grad_candidates) > 0 and n_grad > 0:
            weights = grad_mag.reshape(-1)[grad_candidates].astype(np.float64)
            weights = weights + 1e-8
            weights = weights / np.sum(weights)
            chosen_parts.append(rng.choice(grad_candidates, size=n_grad, replace=len(grad_candidates) < n_grad, p=weights))

        chosen = np.unique(np.concatenate(chosen_parts))
        if len(chosen) < points_per_case:
            fill = rng.choice(
                candidate_indices,
                size=points_per_case - len(chosen),
                replace=len(candidate_indices) < points_per_case,
            )
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


def split_interface_response(
    interface_response: np.ndarray,
    feature_names: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Split full raw interface arrays into clean condition and target arrays."""
    raw_condition_names = tuple(name for name in INTERFACE_CONDITION_FEATURE_NAMES if name != "h_effective")
    condition_indices = [feature_names.index(name) for name in raw_condition_names]
    target_indices = [feature_names.index(name) for name in INTERFACE_TARGET_NAMES]
    return interface_response[..., condition_indices].astype(np.float32), interface_response[..., target_indices].astype(np.float32)


def append_h_effective(
    interface_condition: np.ndarray,
    interface_target: np.ndarray,
    *,
    eps: float,
    h_effective_max: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Append flux-consistent Robin coefficient h and its per-point validity."""
    t_outside = interface_condition[..., 3]
    t_surface = interface_target[..., 0]
    q_normal = interface_target[..., 1]
    delta_t = t_surface - t_outside
    eps_value = max(float(eps), 1.0e-12)

    if interface_condition.shape[-1] >= len(INTERFACE_CONDITION_FEATURE_NAMES):
        condition = interface_condition.astype(np.float32)
        h_values = condition[..., -1]
        valid_mask = (
            (np.abs(delta_t) >= eps_value)
            & np.isfinite(q_normal)
            & np.isfinite(h_values)
            & (h_values >= 0.0)
            & (h_values <= float(h_effective_max))
        ).astype(np.float32)
        return condition, valid_mask, {
            "valid_fraction": float(np.mean(valid_mask)) if valid_mask.size else 0.0,
            "clipped_fraction": 0.0,
            "mean": float(np.mean(h_values)) if h_values.size else 0.0,
        }
    sign = np.where(delta_t < 0.0, -1.0, 1.0).astype(np.float32)
    denom = np.where(np.abs(delta_t) < eps_value, sign * eps_value, delta_t).astype(np.float32)
    raw_h = q_normal / denom
    finite = np.isfinite(raw_h)
    clipped_h = np.nan_to_num(raw_h, nan=0.0, posinf=float(h_effective_max), neginf=0.0)
    clipped_h = np.clip(clipped_h, 0.0, float(h_effective_max)).astype(np.float32)
    condition = np.concatenate([interface_condition.astype(np.float32), clipped_h[..., None]], axis=-1)
    valid_mask = (
        (np.abs(delta_t) >= eps_value)
        & np.isfinite(q_normal)
        & np.isfinite(raw_h)
        & (raw_h >= 0.0)
        & (raw_h <= float(h_effective_max))
    ).astype(np.float32)
    total = max(int(raw_h.size), 1)
    diagnostics = {
        "valid_fraction": float(np.sum(valid_mask) / total),
        "clipped_fraction": float(np.sum(finite & ((raw_h < 0.0) | (raw_h > float(h_effective_max)))) / total),
        "mean": float(np.mean(clipped_h)) if clipped_h.size else 0.0,
    }
    return condition, valid_mask, diagnostics


def process_case(
    raw: RawCase,
    case_key: str,
    split: str,
    selection: SelectionResult,
    points_per_case: int,
    seed: int,
    exclude_module_interior_from_global_points: bool,
    h_effective_eps: float,
    h_effective_max: float,
) -> ProcessedCase:
    """Convert one raw case folder into packed steady-window arrays."""
    selected_rows = selection.selected_rows

    # Final-window averaging is the core steady/quasi-steady reduction. The raw
    # transient is not discarded on disk; only the packed dataset chooses this
    # first training target.
    tensor, last_payload, selected_times = stack_field_window(raw.case_dir, selected_rows)
    selected_frame_ids = np.asarray([int(float(row.get("saved_frame", idx))) for idx, row in enumerate(selected_rows)], dtype=np.int32)
    steady_field = np.mean(tensor, axis=0).astype(np.float32)
    rms_field = np.sqrt(np.mean((tensor - steady_field[None, ...]) ** 2, axis=0)).astype(np.float32)
    x_grid, y_grid = build_uniform_grid(raw.cfg)
    module_mask = last_payload["module_mask"].astype(np.uint8)
    rng = np.random.default_rng(seed)
    sampled_points = sample_global_points(
        steady_field,
        x_grid,
        y_grid,
        raw.cfg,
        module_mask.astype(bool),
        points_per_case,
        rng,
        exclude_module_interior_from_global_points=exclude_module_interior_from_global_points,
    )

    internal_frames: List[np.ndarray] = []
    interface_frames: List[np.ndarray] = []
    for row in selected_rows:
        payload = load_frame(raw.case_dir, row)
        internal_frames.append(payload["module_internal_temperature"].astype(np.float32))
        interface_frames.append(payload["interface_response"].astype(np.float32))

    internal_temperature = np.mean(np.stack(internal_frames, axis=0), axis=0).astype(np.float32)
    interface_response = np.mean(np.stack(interface_frames, axis=0), axis=0).astype(np.float32)
    internal_mask = last_payload["module_internal_mask"].astype(np.uint8)
    feature_names = tuple(name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in last_payload["interface_feature_names"])
    interface_condition, interface_target = split_interface_response(interface_response, feature_names)
    interface_condition, h_effective_valid_mask, h_diag = append_h_effective(
        interface_condition,
        interface_target,
        eps=float(h_effective_eps),
        h_effective_max=float(h_effective_max),
    )
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
        interface_condition=interface_condition,
        h_effective_valid_mask=h_effective_valid_mask,
        interface_target=interface_target,
        interface_feature_names=feature_names,
        module_centers=centers,
        heat_powers=heat_powers,
        module_mask=module_mask,
        exclude_module_interior_from_global_points=exclude_module_interior_from_global_points,
        converged=bool(selection.converged),
        converged_time=float(selection.converged_time),
        converged_step=int(selection.converged_step),
        final_delta_inf=float(selection.final_delta_inf),
        final_delta_l2_rel=float(selection.final_delta_l2_rel),
        selected_frame_ids=selected_frame_ids,
        packed_unconverged=bool(selection.packed_unconverged),
        h_effective_valid_fraction=float(h_diag["valid_fraction"]),
        h_effective_clipped_fraction=float(h_diag["clipped_fraction"]),
        h_effective_mean=float(h_diag["mean"]),
    )


def pad_first_axis(array: np.ndarray, target: int, fill_value: float = 0.0) -> np.ndarray:
    shape = (target,) + tuple(array.shape[1:])
    output = np.full(shape, fill_value, dtype=array.dtype)
    output[: min(target, array.shape[0])] = array[:target]
    return output


def write_global_case_index(output_root: Path, processed: Sequence[ProcessedCase]) -> None:
    """Write a small CSV summary next to the packed HDF5 file."""
    index_path = output_root / "global_case_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case_key",
            "split",
            "case_dir",
            "num_modules",
            "re",
            "heat_power_min",
            "heat_power_max",
            "converged",
            "converged_time",
            "final_delta_inf",
            "final_delta_l2_rel",
        ]
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
                    "converged": int(item.converged),
                    "converged_time": item.converged_time,
                    "final_delta_inf": item.final_delta_inf,
                    "final_delta_l2_rel": item.final_delta_l2_rel,
                }
            )


def write_quality_report(output_root: Path, rows: Sequence[Dict[str, object]]) -> Path:
    """Write per-case preprocessing convergence/packing status."""
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "preprocessing_case_quality.csv"
    fieldnames = [
        "case_key",
        "converged",
        "converged_time",
        "final_delta_inf",
        "final_delta_l2_rel",
        "packed",
        "skip_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _safe_mean_std(values: np.ndarray, axis: int | None = 0) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.asarray([0.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32)
    return np.asarray(np.mean(values, axis=axis), dtype=np.float32), np.asarray(np.std(values, axis=axis), dtype=np.float32)


def _scalar_stat(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def write_normalization_group(h5, processed: Sequence[ProcessedCase]) -> None:
    """Save dataset-level normalization statistics for future training."""
    norm = h5.create_group("normalization")
    fields = np.concatenate([item.steady_field.reshape(-1, len(CHANNEL_ORDER)) for item in processed], axis=0)
    samples = np.concatenate([item.sampled_points[:, 2:] for item in processed], axis=0)
    heat_arrays = [item.heat_powers.reshape(-1) for item in processed if item.heat_powers.size > 0]
    condition_arrays = [
        item.interface_condition.reshape(-1, item.interface_condition.shape[-1]) for item in processed if item.interface_condition.size > 0
    ]
    target_arrays = [item.interface_target.reshape(-1, item.interface_target.shape[-1]) for item in processed if item.interface_target.size > 0]
    heat_powers = np.concatenate(heat_arrays) if heat_arrays else np.asarray([], dtype=np.float32)
    interface_condition = (
        np.concatenate(condition_arrays, axis=0)
        if condition_arrays
        else np.zeros((0, len(INTERFACE_CONDITION_FEATURE_NAMES)), dtype=np.float32)
    )
    interface_target = np.concatenate(target_arrays, axis=0) if target_arrays else np.zeros((0, len(INTERFACE_TARGET_NAMES)), dtype=np.float32)
    internal_values = []
    for item in processed:
        if item.module_internal_temperature.size == 0:
            continue
        disk = item.module_internal_mask.astype(bool)
        internal_values.append(item.module_internal_temperature[:, disk].reshape(-1))
    internal_temperature = np.concatenate(internal_values) if internal_values else np.asarray([], dtype=np.float32)

    field_mean, field_std = _safe_mean_std(fields, axis=0)
    sample_mean, sample_std = _safe_mean_std(samples, axis=0)
    condition_mean, condition_std = _safe_mean_std(interface_condition, axis=0)
    target_mean, target_std = _safe_mean_std(interface_target, axis=0)
    internal_mean, internal_std = _safe_mean_std(internal_temperature, axis=None)

    norm.create_dataset("field_mean_by_channel", data=field_mean)
    norm.create_dataset("field_std_by_channel", data=field_std)
    norm.create_dataset("sampled_point_mean_by_channel", data=sample_mean)
    norm.create_dataset("sampled_point_std_by_channel", data=sample_std)
    norm.create_dataset("heat_power_mean", data=np.asarray([np.mean(heat_powers) if heat_powers.size else 0.0], dtype=np.float32))
    norm.create_dataset("heat_power_std", data=np.asarray([np.std(heat_powers) if heat_powers.size else 0.0], dtype=np.float32))
    norm.create_dataset("interface_condition_mean", data=condition_mean)
    norm.create_dataset("interface_condition_std", data=condition_std)
    norm.create_dataset("interface_target_mean", data=target_mean)
    norm.create_dataset("interface_target_std", data=target_std)
    norm.create_dataset("internal_temperature_mean", data=np.asarray([_scalar_stat(internal_mean)], dtype=np.float32))
    norm.create_dataset("internal_temperature_std", data=np.asarray([_scalar_stat(internal_std)], dtype=np.float32))


def write_h5(
    output_root: Path,
    processed: Sequence[ProcessedCase],
    max_modules_arg: int,
    exclude_module_interior_from_global_points: bool,
    *,
    target_mode: str,
    require_converged: bool,
) -> Path:
    """Write the packed global channel HDF5 file."""
    output_root.mkdir(parents=True, exist_ok=True)
    h5_path = output_root / "packed_dataset.h5"
    max_modules = max(max_modules_arg, max((item.module_centers.shape[0] for item in processed), default=0))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    local_grid_size = int(processed[0].module_internal_mask.shape[0]) if processed else 0
    n_interface_points = int(processed[0].interface_response.shape[1]) if processed and processed[0].interface_response.ndim >= 2 else 0
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["dataset_type"] = "channelthermal_steady"
        h5.attrs["dataset_role"] = "global_channelthermal"
        h5.attrs["target_kind"] = "steady_final_window"
        h5.attrs["target_mode"] = str(target_mode)
        h5.attrs["require_converged"] = bool(require_converged)
        h5.attrs["field_dim"] = len(CHANNEL_ORDER)
        h5.attrs["max_modules"] = max_modules
        h5.attrs["local_grid_size"] = local_grid_size
        h5.attrs["n_interface_points"] = n_interface_points
        h5.attrs["state_id"] = "steady_final_window"
        h5.attrs["exclude_module_interior_from_global_points"] = bool(exclude_module_interior_from_global_points)
        h5.attrs["interface_condition_valid_mask"] = "h_effective_valid_mask"
        h5.attrs["interface_condition_valid_mask_note"] = (
            "1 marks interface points whose h_effective target is finite, within configured bounds, "
            "and has |T_surface - T_outside| above h_effective_eps."
        )
        h5.create_dataset("field_dim", data=np.asarray([len(CHANNEL_ORDER)], dtype=np.int32))
        h5.create_dataset("channel_order", data=np.asarray(CHANNEL_ORDER, dtype=string_dtype))
        h5.create_dataset("sampled_point_feature_names", data=np.asarray(SAMPLED_POINT_FEATURES, dtype=string_dtype))
        h5.create_dataset("interface_condition_feature_names", data=np.asarray(INTERFACE_CONDITION_FEATURE_NAMES, dtype=string_dtype))
        h5.create_dataset("interface_condition_valid_mask_feature_names", data=np.asarray(["h_effective_valid_mask"], dtype=string_dtype))
        h5.create_dataset("interface_target_names", data=np.asarray(INTERFACE_TARGET_NAMES, dtype=string_dtype))
        if processed:
            h5.create_dataset("interface_feature_names", data=np.asarray(processed[0].interface_feature_names, dtype=string_dtype))
        h5.create_dataset("case_ids", data=np.asarray([item.case_key for item in processed], dtype=string_dtype))
        h5.create_dataset("splits", data=np.asarray([item.split for item in processed], dtype=string_dtype))
        write_normalization_group(h5, processed)

        cases_group = h5.create_group("cases")
        for item in processed:
            group = cases_group.create_group(item.case_key)
            group.attrs["split"] = item.split
            group.attrs["source_case_dir"] = str(item.case_dir)
            group.attrs["field_dim"] = len(CHANNEL_ORDER)
            group.attrs["channel_order"] = ",".join(CHANNEL_ORDER)
            group.attrs["converged"] = bool(item.converged)
            group.attrs["converged_time"] = float(item.converged_time)
            group.attrs["converged_step"] = int(item.converged_step)
            group.attrs["final_delta_inf"] = float(item.final_delta_inf)
            group.attrs["final_delta_l2_rel"] = float(item.final_delta_l2_rel)
            group.attrs["packed_unconverged"] = bool(item.packed_unconverged)
            group.attrs["target_mode"] = str(target_mode)
            group.attrs["h_effective_valid_fraction"] = float(item.h_effective_valid_fraction)
            group.attrs["h_effective_clipped_fraction"] = float(item.h_effective_clipped_fraction)
            group.attrs["h_effective_mean"] = float(item.h_effective_mean)
            group.create_dataset("x_grid", data=item.x_grid, compression="gzip")
            group.create_dataset("y_grid", data=item.y_grid, compression="gzip")
            group.create_dataset("steady_field", data=item.steady_field, compression="gzip")
            group.create_dataset("rms_field", data=item.rms_field, compression="gzip")
            group.create_dataset("sampled_points", data=item.sampled_points, compression="gzip")
            group.create_dataset("selected_times", data=item.selected_times)
            group.create_dataset("selected_frame_ids", data=item.selected_frame_ids)
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
            group.create_dataset(
                "interface_condition",
                data=pad_first_axis(item.interface_condition, max_modules),
                compression="gzip",
            )
            group.create_dataset(
                "interface_condition_valid_mask",
                data=pad_first_axis(item.h_effective_valid_mask, max_modules),
                compression="gzip",
            )
            group.create_dataset(
                "interface_target",
                data=pad_first_axis(item.interface_target, max_modules),
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


def ensure_processed_train_split(processed: Sequence[ProcessedCase]) -> None:
    """Keep a skipped-case preprocessing run from producing zero train cases."""
    if not processed:
        return
    if any(item.split == "train" for item in processed):
        return
    processed[0].split = "train"


def main() -> int:
    """CLI entry point for packing raw global cases."""
    args = parse_args()
    input_root = resolve_data_path(args.input_root)
    output_root = resolve_data_path(args.output_root)
    raw_cases = discover_raw_cases(input_root)
    if not raw_cases:
        tqdm.write(f"No raw channel thermal cases found under: {input_root}")
        return 1

    split_assignments = assign_unsplit_cases(raw_cases, args.train_fraction, args.seed)
    processed: List[ProcessedCase] = []
    quality_rows: List[Dict[str, object]] = []
    existing_keys: set[str] = set()
    skipped_unconverged = 0
    packed_unconverged_if_allowed = 0
    for idx, raw in enumerate(tqdm(raw_cases, desc="Preprocessing cases", unit="case", dynamic_ncols=True)):
        base_key = str(raw.cfg.save.case_id) or raw.case_dir.name
        case_key = unique_case_key(base_key, existing_keys)
        selection = select_final_window(
            raw,
            args.final_window_frames,
            target_mode=args.target_mode,
            require_converged=bool(args.require_converged),
            min_final_window_frames=int(args.min_final_window_frames),
        )
        if selection.skip_reason:
            if selection.skip_reason == "unconverged":
                skipped_unconverged += 1
            quality_rows.append(
                {
                    "case_key": case_key,
                    "converged": int(selection.converged),
                    "converged_time": selection.converged_time,
                    "final_delta_inf": selection.final_delta_inf,
                    "final_delta_l2_rel": selection.final_delta_l2_rel,
                    "packed": 0,
                    "skip_reason": selection.skip_reason,
                }
            )
            tqdm.write(f"Skipping case {case_key}: {selection.skip_reason}")
            continue

        if selection.packed_unconverged:
            packed_unconverged_if_allowed += 1
        split = choose_split(idx, raw, split_assignments)
        item = process_case(
            raw,
            case_key,
            split,
            selection,
            args.points_per_case,
            args.seed + idx,
            bool(args.exclude_module_interior_from_global_points),
            float(args.h_effective_eps),
            float(args.h_effective_max),
        )
        processed.append(item)
        quality_rows.append(
            {
                "case_key": case_key,
                "converged": int(item.converged),
                "converged_time": item.converged_time,
                "final_delta_inf": item.final_delta_inf,
                "final_delta_l2_rel": item.final_delta_l2_rel,
                "packed": 1,
                "skip_reason": "packed_unconverged" if item.packed_unconverged else "",
            }
        )

    quality_path = write_quality_report(output_root, quality_rows)
    if not processed:
        tqdm.write(
            "Preprocessing summary: "
            f"total_raw_cases={len(raw_cases)}, packed_cases=0, skipped_unconverged={skipped_unconverged}, "
            f"packed_unconverged_if_allowed={packed_unconverged_if_allowed}"
        )
        tqdm.write(f"Wrote case quality report: {quality_path}")
        tqdm.write("No cases were packed; no HDF5 file was written.")
        return 1

    ensure_processed_train_split(processed)
    h5_path = write_h5(
        output_root,
        processed,
        args.max_modules,
        bool(args.exclude_module_interior_from_global_points),
        target_mode=args.target_mode,
        require_converged=bool(args.require_converged),
    )
    tqdm.write(
        "Preprocessing summary: "
        f"total_raw_cases={len(raw_cases)}, packed_cases={len(processed)}, skipped_unconverged={skipped_unconverged}, "
        f"packed_unconverged_if_allowed={packed_unconverged_if_allowed}"
    )
    tqdm.write(f"Wrote case quality report: {quality_path}")
    tqdm.write(f"Packed {len(processed)} channel thermal cases into: {h5_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
