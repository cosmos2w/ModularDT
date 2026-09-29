"""Leakage-safe source data for conditional Thermal heat-allocation generation.

The clean heat vector is returned only as the supervised denoising target and
for evaluation. The inverse condition contains known geometry/material/context,
the supplied total heat, and a small fixed-grid set of fluid-temperature
observations. Sensor selection uses only the known solid mask and coordinates;
it never reads heat powers or field values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from honf_inverse_core.models.frozen_packet_diffusion import DiffusionCondition

MODULE_FEATURE_NAMES = (
    "x_fraction",
    "y_fraction",
    "re",
    "u_in",
    "domain_length_x",
    "domain_length_y",
    "nu",
    "solid_alpha",
    "fluid_alpha",
    "solid_k",
    "fluid_k",
    "module_radius",
    "supplied_total_heat",
)
SENSOR_FEATURE_NAMES = ("x_fraction", "y_fraction", "observed_temperature")
PHYSICAL_CONTEXT_NAMES = (
    "reynolds_number",
    "inlet_velocity",
    "domain_length_x",
    "domain_length_y",
    "kinematic_viscosity",
    "solid_thermal_diffusivity",
    "fluid_thermal_diffusivity",
    "solid_conductivity",
    "fluid_conductivity",
    "module_radius",
)


@dataclass(frozen=True)
class ThermalHeatTask:
    """One solved Thermal layout represented without exposing its hidden heats."""

    case_id: str
    split: str
    module_centers: np.ndarray
    module_features: np.ndarray
    module_valid: np.ndarray
    sensor_xy: np.ndarray
    sensor_features: np.ndarray
    physical_context: np.ndarray
    clean_heat: np.ndarray
    clean_state: np.ndarray
    total_heat: float


@dataclass(frozen=True)
class ThermalHeatBatch:
    """Padded tensors for the frozen-packet diffusion model and metrics."""

    condition: DiffusionCondition
    clean_state: torch.Tensor
    clean_heat: torch.Tensor
    total_heat: torch.Tensor
    module_centers: torch.Tensor
    sensor_xy: torch.Tensor
    physical_context: torch.Tensor
    case_ids: tuple[str, ...]


def centered_log_fractions(heat: np.ndarray, *, epsilon: float = 1.0e-8) -> np.ndarray:
    """Encode a positive allocation as centered log fractions."""

    values = np.asarray(heat, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("heat must be a nonempty finite vector.")
    if np.any(values < 0.0) or float(values.sum()) <= 0.0:
        raise ValueError("heat allocations must be nonnegative with positive total.")
    fractions = values / values.sum()
    fractions = np.maximum(fractions, float(epsilon))
    fractions /= fractions.sum()
    logits = np.log(fractions)
    return (logits - logits.mean()).astype(np.float32)


def _decode_text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def _case_metadata(group: h5py.Group) -> dict[str, object]:
    raw = group["case_config_json"][()] if "case_config_json" in group else "{}"
    try:
        value = json.loads(_decode_text(raw))
    except (json.JSONDecodeError, TypeError):
        value = {}
    return value if isinstance(value, dict) else {}


def _context(group: h5py.Group) -> tuple[float, ...]:
    config = _case_metadata(group)
    domain = config.get("domain", {})
    flow = config.get("flow", {})
    thermal = config.get("thermal", {})
    domain = domain if isinstance(domain, dict) else {}
    flow = flow if isinstance(flow, dict) else {}
    thermal = thermal if isinstance(thermal, dict) else {}
    material = group.get("material_parameters")

    def attr(name: str, fallback: float) -> float:
        return float(material.attrs.get(name, fallback)) if material is not None else float(fallback)

    reynolds = float(flow.get("re") if flow.get("re") is not None else attr("re", np.nan))
    inflow = float(flow.get("u_in") if flow.get("u_in") is not None else attr("u_in", np.nan))
    lx = float(domain.get("lx", np.nan))
    ly = float(domain.get("ly", np.nan))
    radius = float(
        domain.get("module_radius")
        if domain.get("module_radius") is not None
        else attr("module_radius", np.nan)
    )
    raw_nu = flow.get("nu")
    if raw_nu is None:
        raw_nu = attr("nu", np.nan)
    runtime = config.get("runtime", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    runtime_nu = runtime.get("nu")
    derived_nu = float(flow.get("viscosity_scale", 1.0)) * inflow * (2.0 * radius) / reynolds
    if raw_nu is None or not np.isfinite(float(raw_nu)):
        nu = float(runtime_nu) if runtime_nu is not None else derived_nu
    else:
        nu = float(raw_nu)
    if runtime_nu is not None and not math.isclose(nu, float(runtime_nu), rel_tol=1.0e-6, abs_tol=1.0e-12):
        raise ValueError("Configured/derived viscosity disagrees with the recorded runtime value.")
    values = (
        reynolds,
        inflow,
        lx,
        ly,
        nu,
        float(thermal.get("solid_alpha", attr("solid_alpha", np.nan))),
        float(thermal.get("fluid_alpha", attr("fluid_alpha", np.nan))),
        float(thermal.get("solid_k", attr("solid_k", np.nan))),
        float(thermal.get("fluid_k", attr("fluid_k", np.nan))),
        radius,
    )
    if not np.isfinite(values).all() or any(value <= 0.0 for value in values):
        raise ValueError("Case metadata must provide positive finite flow, material, and domain values.")
    return values


def _sensor_candidates(height: int, width: int) -> np.ndarray:
    """Return a fixed, spatially broad grid catalogue in row/column order."""

    rows = np.unique(np.linspace(3, height - 4, min(15, height - 6), dtype=np.int64))
    cols = np.unique(np.linspace(4, width - 5, min(25, width - 8), dtype=np.int64))
    grid = np.asarray([(row, col) for row in rows for col in cols], dtype=np.int64)
    # This order is fixed across cases; a later geometry-only farthest-point
    # selection spreads the selected sensors over this declared catalogue.
    return grid


def select_fluid_sensor_indices(
    module_mask: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    sensor_count: int = 12,
) -> np.ndarray:
    """Select fixed-grid fluid sensor cells using only coordinates and geometry."""

    mask = np.asarray(module_mask, dtype=bool)
    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    if mask.ndim != 2 or x.shape != mask.shape or y.shape != mask.shape:
        raise ValueError("module_mask, x_grid, and y_grid must share a 2-D shape.")
    if sensor_count <= 0:
        raise ValueError("sensor_count must be positive.")
    candidates = _sensor_candidates(*mask.shape)
    candidates = candidates[~mask[candidates[:, 0], candidates[:, 1]]]
    if candidates.shape[0] < sensor_count:
        raise ValueError("The declared sensor catalogue has too few fluid cells for this geometry.")
    coords = np.stack((x[candidates[:, 0], candidates[:, 1]], y[candidates[:, 0], candidates[:, 1]]), axis=-1)
    # Seed at the catalogue center, then select the farthest valid point from
    # the existing set. Ties resolve by the stable catalogue order. This uses
    # geometry only and yields exactly sensor_count known-coordinate probes.
    center = np.asarray([np.median(x), np.median(y)])
    selected: list[int] = [int(np.argmin(np.square(coords - center).sum(axis=-1)))]
    while len(selected) < sensor_count:
        distance = np.square(coords[:, None, :] - coords[np.asarray(selected)][None, :, :]).sum(axis=-1).min(axis=1)
        distance[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(distance)))
    return candidates[np.asarray(selected)]


def build_thermal_heat_tasks(
    dataset_path: str | Path,
    *,
    split: str,
    sensor_count: int = 12,
    case_ids: Sequence[str] | None = None,
    min_modules: int = 2,
) -> tuple[ThermalHeatTask, ...]:
    """Load compact heat-generation tasks from the packed global dataset.

    Sensor values are read from the stored temperature channel on only the
    grid rows touched by the geometry-selected sensor cells. Full fields are
    discarded as each case is read and never enter ``ThermalHeatTask``.
    """

    if split not in {"train", "test"}:
        raise ValueError("Heat-task source split must be exactly 'train' or 'test'.")
    if min_modules < 1:
        raise ValueError("min_modules must be positive.")
    path = Path(dataset_path).expanduser().resolve()
    requested = None if case_ids is None else {str(value) for value in case_ids}
    tasks: list[ThermalHeatTask] = []
    with h5py.File(path, "r") as handle:
        ids = [_decode_text(value) for value in handle["case_ids"][...]]
        splits = [_decode_text(value) for value in handle["splits"][...]]
        for case_id, case_split in zip(ids, splits):
            if case_split != split or (requested is not None and case_id not in requested):
                continue
            group = handle["cases"][case_id]
            if not bool(group.attrs.get("converged", False)):
                continue
            present = np.asarray(group["module_present"][...], dtype=bool)
            centers_all = np.asarray(group["module_centers"][...], dtype=np.float32)
            heat_all = np.asarray(group["heat_powers"][...], dtype=np.float32)
            centers = centers_all[present]
            heat = heat_all[present]
            if centers.shape[0] < min_modules:
                continue
            if centers.ndim != 2 or centers.shape[1] != 2 or heat.shape != (centers.shape[0],):
                raise ValueError(f"Case {case_id} has misaligned active module geometry and heat.")
            if centers.shape[0] == 0 or not np.isfinite(centers).all() or not np.isfinite(heat).all():
                raise ValueError(f"Case {case_id} has empty or nonfinite active design values.")
            total_heat = float(heat.sum())
            if np.any(heat < 0.0) or total_heat <= 0.0:
                raise ValueError(f"Case {case_id} needs a nonnegative positive-total heat allocation.")
            if "module_mask" not in group or "steady_field" not in group:
                raise ValueError(f"Case {case_id} lacks the geometry mask or stored steady field.")
            module_mask = np.asarray(group["module_mask"][...], dtype=bool)
            x_grid = np.asarray(group["x_grid"][...], dtype=np.float32)
            y_grid = np.asarray(group["y_grid"][...], dtype=np.float32)
            field = group["steady_field"]
            if field.ndim != 3 or field.shape[-1] < 5 or tuple(field.shape[:2]) != tuple(module_mask.shape):
                raise ValueError(f"Case {case_id} has an unsupported native field/mask shape.")
            indices = select_fluid_sensor_indices(
                module_mask, x_grid, y_grid, sensor_count=sensor_count
            )
            rows, cols = indices[:, 0], indices[:, 1]
            sensor_xy = np.stack((x_grid[rows, cols], y_grid[rows, cols]), axis=-1).astype(np.float32)
            # h5py requires monotonic indices for fancy indexing. Read only
            # the temperature channel and selected rows, never a whole field.
            selected_rows = {
                int(row): np.asarray(field[int(row), :, 4], dtype=np.float32)
                for row in np.unique(rows)
            }
            observed_temperature = np.asarray(
                [selected_rows[int(row)][int(col)] for row, col in zip(rows, cols, strict=True)],
                dtype=np.float32,
            )
            if not np.isfinite(observed_temperature).all():
                raise ValueError(f"Case {case_id} has nonfinite selected sensor observations.")
            re, u_in, lx, ly, nu, solid_alpha, fluid_alpha, solid_k, fluid_k, radius = _context(group)
            physical_context = np.asarray(
                (re, u_in, lx, ly, nu, solid_alpha, fluid_alpha, solid_k, fluid_k, radius),
                dtype=np.float32,
            )
            x_fraction = centers[:, 0] / lx
            y_fraction = centers[:, 1] / ly
            known = np.stack(
                (
                    x_fraction,
                    y_fraction,
                    np.full_like(x_fraction, re),
                    np.full_like(x_fraction, u_in),
                    np.full_like(x_fraction, lx),
                    np.full_like(x_fraction, ly),
                    np.full_like(x_fraction, nu),
                    np.full_like(x_fraction, solid_alpha),
                    np.full_like(x_fraction, fluid_alpha),
                    np.full_like(x_fraction, solid_k),
                    np.full_like(x_fraction, fluid_k),
                    np.full_like(x_fraction, radius),
                    np.full_like(x_fraction, total_heat),
                ),
                axis=-1,
            ).astype(np.float32)
            sensor_features = np.stack(
                (sensor_xy[:, 0] / lx, sensor_xy[:, 1] / ly, observed_temperature), axis=-1
            ).astype(np.float32)
            valid = np.ones((centers.shape[0],), dtype=bool)
            tasks.append(
                ThermalHeatTask(
                    case_id=case_id,
                    split=case_split,
                    module_centers=centers,
                    module_features=known,
                    module_valid=valid,
                    sensor_xy=sensor_xy,
                    sensor_features=sensor_features,
                    physical_context=physical_context,
                    clean_heat=heat.copy(),
                    clean_state=centered_log_fractions(heat)[:, None],
                    total_heat=total_heat,
                )
            )
    if requested is not None:
        missing = requested - {task.case_id for task in tasks}
        if missing:
            raise ValueError(f"Requested case IDs are absent from split {split!r}: {sorted(missing)}")
    return tuple(tasks)


@dataclass(frozen=True)
class ThermalHeatFeatureScaler:
    """Training-only standardization for known module and sensor features."""

    module_mean: np.ndarray
    module_std: np.ndarray
    sensor_mean: np.ndarray
    sensor_std: np.ndarray

    @classmethod
    def fit(cls, tasks: Iterable[ThermalHeatTask]) -> ThermalHeatFeatureScaler:
        rows = tuple(tasks)
        if not rows or any(task.split != "train" for task in rows):
            raise ValueError("Feature scaling may be fit on a nonempty train-only cohort.")
        modules = np.concatenate([task.module_features for task in rows], axis=0).astype(np.float64)
        sensors = np.concatenate([task.sensor_features for task in rows], axis=0).astype(np.float64)
        module_mean = modules.mean(axis=0)
        sensor_mean = sensors.mean(axis=0)
        module_std = modules.std(axis=0)
        sensor_std = sensors.std(axis=0)
        module_std = np.maximum(module_std, 1.0e-6)
        sensor_std = np.maximum(sensor_std, 1.0e-6)
        return cls(module_mean.astype(np.float32), module_std.astype(np.float32), sensor_mean.astype(np.float32), sensor_std.astype(np.float32))

    def transform_modules(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.module_mean) / self.module_std).astype(np.float32)

    def transform_sensors(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.sensor_mean) / self.sensor_std).astype(np.float32)


def collate_thermal_heat_tasks(
    tasks: Sequence[ThermalHeatTask],
    *,
    scaler: ThermalHeatFeatureScaler,
    device: torch.device | str = "cpu",
) -> ThermalHeatBatch:
    """Pad a variable-module task batch without placing clean heats in inputs."""

    if not tasks or len({task.sensor_features.shape[0] for task in tasks}) != 1:
        raise ValueError("A heat batch needs tasks with one shared nonempty sensor count.")
    batch = len(tasks)
    max_modules = max(task.module_valid.size for task in tasks)
    sensor_count = int(tasks[0].sensor_features.shape[0])
    module_features = np.zeros((batch, max_modules, len(MODULE_FEATURE_NAMES)), dtype=np.float32)
    module_valid = np.zeros((batch, max_modules), dtype=bool)
    clean_state = np.zeros((batch, max_modules, 1), dtype=np.float32)
    clean_heat = np.zeros((batch, max_modules), dtype=np.float32)
    centers = np.zeros((batch, max_modules, 2), dtype=np.float32)
    sensor_features = np.zeros((batch, sensor_count, len(SENSOR_FEATURE_NAMES)), dtype=np.float32)
    sensor_xy = np.zeros((batch, sensor_count, 2), dtype=np.float32)
    physical_context = np.zeros((batch, len(PHYSICAL_CONTEXT_NAMES)), dtype=np.float32)
    total_heat = np.zeros((batch, 1), dtype=np.float32)
    for row, task in enumerate(tasks):
        count = task.module_valid.size
        if task.split not in {"train", "test"}:
            raise ValueError(f"Unsupported source split {task.split!r}.")
        module_features[row, :count] = scaler.transform_modules(task.module_features)
        module_valid[row, :count] = task.module_valid
        clean_state[row, :count] = task.clean_state
        clean_heat[row, :count] = task.clean_heat
        centers[row, :count] = task.module_centers
        sensor_features[row] = scaler.transform_sensors(task.sensor_features)
        sensor_xy[row] = task.sensor_xy
        physical_context[row] = task.physical_context
        total_heat[row, 0] = task.total_heat
    valid_tensor = torch.as_tensor(module_valid, dtype=torch.bool, device=device)
    known_state = torch.zeros((batch, max_modules, 1), dtype=torch.float32, device=device)
    condition = DiffusionCondition(
        known_state=known_state,
        design_mask=valid_tensor.clone(),
        module_valid=valid_tensor,
        module_features=torch.as_tensor(module_features, device=device),
        sensor_features=torch.as_tensor(sensor_features, device=device),
        sensor_valid=torch.ones((batch, sensor_count), dtype=torch.bool, device=device),
    )
    return ThermalHeatBatch(
        condition=condition,
        clean_state=torch.as_tensor(clean_state, device=device),
        clean_heat=torch.as_tensor(clean_heat, device=device),
        total_heat=torch.as_tensor(total_heat, device=device),
        module_centers=torch.as_tensor(centers, device=device),
        sensor_xy=torch.as_tensor(sensor_xy, device=device),
        physical_context=torch.as_tensor(physical_context, device=device),
        case_ids=tuple(task.case_id for task in tasks),
    )


__all__ = [
    "MODULE_FEATURE_NAMES",
    "PHYSICAL_CONTEXT_NAMES",
    "SENSOR_FEATURE_NAMES",
    "ThermalHeatBatch",
    "ThermalHeatFeatureScaler",
    "ThermalHeatTask",
    "build_thermal_heat_tasks",
    "centered_log_fractions",
    "collate_thermal_heat_tasks",
    "select_fluid_sensor_indices",
]
