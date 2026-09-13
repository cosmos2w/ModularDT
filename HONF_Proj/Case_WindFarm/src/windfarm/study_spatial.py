"""Bounded native-volume reductions and physical-coordinate slice figures.

The caller owns model preparation, the target transform, and axis quadrature.
These routines retain at most one coordinate chunk or native plane, never a
full prediction volume or a native-cell by environmental-token routing map.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import RunView


def native_coordinates(run: RunView, flat: np.ndarray, diameter: float) -> np.ndarray:
    """Generate exact C-order coordinates in D units using index arithmetic."""
    flat = np.asarray(flat, dtype=np.int64)
    ix = flat % run.nx
    iy = (flat // run.nx) % run.ny
    iz = flat // (run.nx * run.ny)
    return np.column_stack((run.x_m[ix], run.y_m[iy], run.z_m[iz])).astype(np.float32) / diameter


class WeightedVelocityErrors:
    """Float64 integral sums with an explicit target-energy denominator."""

    def __init__(self) -> None:
        self.weight = 0.0
        self.count = 0
        self.squared = np.zeros(3, dtype=np.float64)
        self.absolute = np.zeros(3, dtype=np.float64)
        self.energy = np.zeros(3, dtype=np.float64)

    def add(self, prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if prediction.shape != target.shape or target.shape != (len(weights), 3):
            raise ValueError("Velocity predictions, targets, and integration weights must align.")
        if not (np.isfinite(prediction).all() and np.isfinite(target).all() and np.isfinite(weights).all()):
            raise FloatingPointError("Nonfinite native prediction, target, or integration weight.")
        if np.any(weights < 0):
            raise ValueError("Integration weights must be nonnegative.")
        difference = prediction - target
        self.weight += float(weights.sum())
        self.count += len(weights)
        self.squared += np.sum(weights[:, None] * difference**2, axis=0)
        self.absolute += np.sum(weights[:, None] * np.abs(difference), axis=0)
        self.energy += np.sum(weights[:, None] * target**2, axis=0)

    def result(self, physical_std: Sequence[float], reference_speed: float) -> dict[str, Any]:
        if self.weight <= 0:
            return {"count": self.count, "available": False}
        mse = self.squared / self.weight
        energy = float(self.energy.sum())
        return {
            "available": True,
            "count": self.count,
            "quadrature_volume_D3": self.weight,
            "rmse_m_s": np.sqrt(mse).tolist(),
            "mae_m_s": (self.absolute / self.weight).tolist(),
            "rmse_over_U_ref": (np.sqrt(mse) / reference_speed).tolist(),
            "standardized_mse": float(np.mean(mse / np.asarray(physical_std)**2)),
            "vector_relative_l2": float(np.sqrt(self.squared.sum() / energy)) if energy > 0 else None,
            "squared_error_integral": self.squared.tolist(),
            "target_energy_integral": self.energy.tolist(),
        }


def downstream_envelope(coords_D: np.ndarray, hubs_D: np.ndarray) -> np.ndarray:
    """Geometry diagnostic only; this is neither a true wake nor a solid mask."""
    result = np.zeros(len(coords_D), dtype=bool)
    for hub in hubs_D:
        relative = coords_D - hub
        result |= ((relative[:, 0] > 0) & (relative[:, 0] <= 10)
                   & (np.sum(relative[:, 1:]**2, axis=1) <= 1.5**2))
    return result


def stream_native_errors(
    run: RunView,
    predict: Callable[[np.ndarray], np.ndarray],
    background: Callable[[np.ndarray], np.ndarray],
    axis_weights_D: Sequence[np.ndarray],
    hubs_D: np.ndarray,
    physical_std: Sequence[float],
    *,
    diameter: float = 80.0,
    reference_speed: float = 9.0,
    hub_height: float = 70.0,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    """Read each native cell once and integrate under centre-support weights."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    errors = {name: WeightedVelocityErrors() for name in ("volume", "band", "downstream", "background")}
    residual_energy = 0.0
    wx, wy, wz = axis_weights_D
    for start in range(0, run.cell_count, chunk_size):
        flat = np.arange(start, min(start + chunk_size, run.cell_count), dtype=np.int64)
        coords = native_coordinates(run, flat, diameter)
        target = np.array(run.U[start:start + len(flat)], dtype=np.float64, copy=True)
        prediction = predict(coords)
        baseline = background(coords[:, 2])
        weights = wx[flat % run.nx] * wy[(flat // run.nx) % run.ny] * wz[flat // (run.nx * run.ny)]
        errors["volume"].add(prediction, target, weights)
        errors["background"].add(baseline, target, weights)
        residual_energy += float(np.sum(weights * (target[:, 0] - baseline[:, 0])**2))
        band = np.abs(coords[:, 2] - hub_height / diameter) <= 0.5
        wake = downstream_envelope(coords, hubs_D)
        for name, mask in (("band", band), ("downstream", wake)):
            if mask.any():
                errors[name].add(prediction[mask], target[mask], weights[mask])
    result = {name: value.result(physical_std, reference_speed) for name, value in errors.items()}
    result["Ux_background_residual"] = {
        "target_residual_energy_integral": residual_energy,
        "error_integral": float(errors["volume"].squared[0]),
        "relative_l2": float(np.sqrt(errors["volume"].squared[0] / residual_energy)) if residual_energy > 0 else None,
        "definition": "Compare (prediction_x - background_x) with (target_x - background_x).",
    }
    return result


def native_plane(run: RunView, fixed_axis: str, requested_m: float, diameter: float = 80.0) -> dict[str, Any]:
    """Extract a true native orthogonal plane and exact matching flat indices."""
    axes = {"x": run.x_m, "y": run.y_m, "z": run.z_m}
    if fixed_axis not in axes:
        raise ValueError("fixed_axis must be x, y, or z.")
    index = int(np.argmin(np.abs(axes[fixed_axis] - requested_m)))
    if fixed_axis == "z":
        first, second = "x", "y"
        iy, ix = np.meshgrid(np.arange(run.ny), np.arange(run.nx), indexing="ij")
        flat = ix + run.nx * (iy + run.ny * index)
    elif fixed_axis == "y":
        first, second = "x", "z"
        iz, ix = np.meshgrid(np.arange(run.nz), np.arange(run.nx), indexing="ij")
        flat = ix + run.nx * (index + run.ny * iz)
    else:
        first, second = "y", "z"
        iz, iy = np.meshgrid(np.arange(run.nz), np.arange(run.ny), indexing="ij")
        flat = index + run.nx * (iy + run.ny * iz)
    return {
        "fixed_axis": fixed_axis, "requested_m": requested_m,
        "actual_m": float(axes[fixed_axis][index]), "index": index,
        "horizontal_axis": first, "vertical_axis": second,
        "horizontal_D": np.asarray(axes[first]) / diameter,
        "vertical_D": np.asarray(axes[second]) / diameter,
        "coords_D": native_coordinates(run, flat.ravel(), diameter),
        "target": np.array(run.U[flat.ravel()], copy=True).reshape(*flat.shape, 3),
    }


def render_native_plane(plane: dict[str, Any], prediction: np.ndarray, hubs_D: np.ndarray,
                        title: str, destination: Path) -> None:
    """Use common physical scales across reference/prediction and equal aspect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = plane["target"]
    prediction = np.asarray(prediction).reshape(target.shape)
    difference = prediction - target
    fig, axes = plt.subplots(3, 3, figsize=(17, 10), layout="constrained")
    for channel, label in enumerate(("Ux", "Uy", "Uz")):
        low = float(min(target[..., channel].min(), prediction[..., channel].min()))
        high = float(max(target[..., channel].max(), prediction[..., channel].max()))
        if channel:
            high = max(abs(low), abs(high), 1e-6)
            low = -high
        error_scale = max(float(np.abs(difference[..., channel]).max()), 1e-6)
        for column, (values, name) in enumerate(((target, "Reference"), (prediction, "Prediction"), (difference, "Error"))):
            ax = axes[channel, column]
            limits = (-error_scale, error_scale) if column == 2 else (low, high)
            artist = ax.pcolormesh(plane["horizontal_D"], plane["vertical_D"], values[..., channel],
                                  shading="nearest", cmap="RdBu_r" if channel or column == 2 else "viridis",
                                  vmin=limits[0], vmax=limits[1], rasterized=True)
            first = "xyz".index(plane["horizontal_axis"])
            second = "xyz".index(plane["vertical_axis"])
            ax.scatter(hubs_D[:, first], hubs_D[:, second], s=12, facecolors="none", edgecolors="black", linewidths=0.5)
            ax.set(xlabel=f"{plane['horizontal_axis']} / D", ylabel=f"{plane['vertical_axis']} / D",
                   title=f"{name}: {label} [m/s]",
                   xlim=(plane["horizontal_D"][0], plane["horizontal_D"][-1]),
                   ylim=(plane["vertical_D"][0], plane["vertical_D"][-1]))
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(artist, ax=ax, shrink=0.65)
    fig.suptitle(f"{title}\nNative {plane['fixed_axis']} = {plane['actual_m']:.3f} m; turbine markers projected onto plane")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
