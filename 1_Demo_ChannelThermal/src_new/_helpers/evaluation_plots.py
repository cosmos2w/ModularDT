"""CHANNELTHERMAL-SPECIFIC evaluation plotting helpers.

Inputs are ChannelThermal dataset samples, predicted global fields, optional
internal/interface predictions, and channel names. Outputs are quicklook PNGs
and metric dictionaries. These helpers are specific to ChannelThermal geometry,
module masks, and legacy field-channel ordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def channel_cmap(name: str) -> str:
    return {"u": "coolwarm", "v": "coolwarm", "p": "magma", "omega": "RdBu_r", "temperature": "inferno"}.get(name, "viridis")


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    diff = pred - gt
    flat_diff = diff.reshape(-1)
    flat_gt = gt.reshape(-1)
    mse = float(np.mean(flat_diff * flat_diff)) if flat_diff.size else float("nan")
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
    mae = float(np.mean(np.abs(flat_diff))) if flat_diff.size else float("nan")
    l2_norm = float(np.linalg.norm(flat_diff, ord=2))
    gt_norm = float(np.linalg.norm(flat_gt, ord=2))
    relative_l2 = float(l2_norm / max(gt_norm, 1.0e-12))
    finite_gt = flat_gt[np.isfinite(flat_gt)]
    if finite_gt.size:
        span = float(np.max(finite_gt) - np.min(finite_gt))
        rms_scale = float(np.sqrt(np.mean(finite_gt * finite_gt)))
        normalizer = span if span > 1.0e-12 else max(rms_scale, 1.0e-12)
    else:
        normalizer = float("nan")
    return {
        "l2_norm": l2_norm,
        "mse": mse,
        "rmse": rmse,
        "nrmse": float(rmse / normalizer) if np.isfinite(rmse) and np.isfinite(normalizer) else float("nan"),
        "mae": mae,
        "relative_l2": relative_l2,
        "normalizer": normalizer,
        "num_values": float(flat_diff.size),
    }


def masked_error_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(prediction)
    gt = np.asarray(target)
    valid = np.asarray(mask, dtype=bool)
    if pred.ndim == valid.ndim + 1:
        pred = pred[valid, :]
        gt = gt[valid, :]
    else:
        pred = pred[valid]
        gt = gt[valid]
    return error_metrics(pred, gt)


def module_radius_from_sample(sample: Dict[str, Any], fallback: float = 0.45) -> float:
    material = np.asarray(sample["structure"].get("material_params", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    if material.size > 5 and float(material[5]) > 0.0:
        return float(material[5])
    return float(fallback)


def module_and_fluid_masks(sample: Dict[str, Any], pred_field: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if "module_mask" in sample:
        module_mask = np.asarray(sample["module_mask"], dtype=bool)
    else:
        x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
        y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
        module_mask = np.zeros(x_grid.shape, dtype=bool)
        centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
        present = np.asarray(sample["structure"]["module_present"] > 0.5)
        radius = module_radius_from_sample(sample)
        for module_idx in np.flatnonzero(present):
            cx, cy = centers[module_idx]
            module_mask |= np.hypot(x_grid - float(cx), y_grid - float(cy)) <= radius
    if pred_field is not None:
        module_mask = module_mask[: pred_field.shape[0], : pred_field.shape[1]]
    return module_mask, ~module_mask


def draw_module_outlines(ax: Any, sample: Dict[str, Any], color: str = "#d9d9d9", linewidth: float = 1.0) -> None:
    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"] > 0.5)
    radius = module_radius_from_sample(sample)
    for module_idx in np.flatnonzero(present):
        cx, cy = centers[module_idx]
        ax.add_patch(Circle((float(cx), float(cy)), radius, fill=False, color=color, lw=linewidth))


def _local_disk_image(values: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
    image = np.full(local_mask.shape, np.nan, dtype=np.float32)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    image[np.asarray(local_mask, dtype=bool)] = flat[: int(np.sum(local_mask))]
    return image


def composite_temperature_grid(sample: Dict[str, Any], global_temperature: np.ndarray, internal_temperature_points: np.ndarray) -> np.ndarray:
    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"] > 0.5)
    local_mask = np.asarray(sample["module_internal_mask"], dtype=bool)
    radius = module_radius_from_sample(sample)
    out = np.asarray(global_temperature, dtype=np.float32).copy()
    n = int(local_mask.shape[0])
    if n <= 1:
        return out
    for module_idx in np.flatnonzero(present):
        if module_idx >= internal_temperature_points.shape[0]:
            continue
        local_img = _local_disk_image(internal_temperature_points[module_idx], local_mask)
        cx, cy = centers[module_idx]
        inside = np.hypot(x_grid - float(cx), y_grid - float(cy)) <= radius
        xi = np.clip((x_grid[inside] - float(cx)) / max(radius, 1.0e-12), -1.0, 1.0)
        eta = np.clip((y_grid[inside] - float(cy)) / max(radius, 1.0e-12), -1.0, 1.0)
        ii = np.rint((xi + 1.0) * 0.5 * (n - 1)).astype(int)
        jj = np.rint((eta + 1.0) * 0.5 * (n - 1)).astype(int)
        values = local_img[jj, ii]
        valid = np.isfinite(values)
        inside_indices = np.flatnonzero(inside.reshape(-1))
        out.reshape(-1)[inside_indices[valid]] = values[valid]
    return out


def plot_field_quicklook(
    output_path: Path,
    sample: Dict[str, Any],
    pred_field: np.ndarray,
    channel_order: list[str],
    *,
    pred_internal_temperature: Optional[np.ndarray] = None,
    temperature_display_mode: str = "fluid_only",
) -> None:
    gt = sample["steady_field"][..., : pred_field.shape[-1]]
    preferred = [name for name in ["u", "v", "p", "omega", "temperature"] if name in channel_order]
    if not preferred:
        preferred = channel_order[: min(3, len(channel_order))]
    extent = (
        float(np.min(sample["x_grid"])),
        float(np.max(sample["x_grid"])),
        float(np.min(sample["y_grid"])),
        float(np.max(sample["y_grid"])),
    )
    fig, axes = plt.subplots(len(preferred), 3, figsize=(10.5, max(2.2, 2.2 * len(preferred))), constrained_layout=True)
    if len(preferred) == 1:
        axes = axes[None, :]
    _, fluid_mask = module_and_fluid_masks(sample, pred_field)
    for row, name in enumerate(preferred):
        idx = channel_order.index(name)
        gt_img = np.where(fluid_mask, gt[..., idx], np.nan)
        pred_img = np.where(fluid_mask, pred_field[..., idx], np.nan)
        if name == "temperature" and temperature_display_mode == "composite_internal" and pred_internal_temperature is not None and np.asarray(pred_internal_temperature).size:
            gt_img = composite_temperature_grid(sample, gt[..., idx], sample["module_internal_temperature_points"])
            pred_img = composite_temperature_grid(sample, pred_field[..., idx], pred_internal_temperature[..., 0])
        err_img = np.abs(pred_img - gt_img)
        channel_metrics = masked_error_metrics(pred_img, gt_img, fluid_mask)
        vmin = float(np.nanmin(gt_img))
        vmax = float(np.nanmax(gt_img))
        for col, (image, title, cmap) in enumerate(
            [
                (gt_img, f"GT {name}", channel_cmap(name)),
                (pred_img, f"Pred {name}", channel_cmap(name)),
                (err_img, f"Abs error {name}\nrelL2={channel_metrics['relative_l2']:.4e}", "magma"),
            ]
        ):
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#303030")
            im = axes[row, col].imshow(image, origin="lower", extent=extent, cmap=cm, vmin=vmin if col < 2 else None, vmax=vmax if col < 2 else None, aspect="auto")
            draw_module_outlines(axes[row, col], sample, color="#e6e6e6", linewidth=0.9)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.savefig(str(output_path), dpi=170)
    plt.close(fig)


def plot_internal(output_path: Path, sample: Dict[str, Any], pred_internal: np.ndarray) -> bool:
    if pred_internal is None or np.asarray(pred_internal).size == 0 or np.asarray(pred_internal).shape[-2] == 0:
        return False
    present = sample["structure"]["module_present"] > 0.5
    indices = np.flatnonzero(present)[: min(3, int(np.sum(present)))]
    if len(indices) == 0:
        return False
    mask = sample["module_internal_mask"].astype(bool)
    gt_points = sample["module_internal_temperature_points"]
    fig, axes = plt.subplots(len(indices), 3, figsize=(9.8, 3.0 * len(indices)), constrained_layout=True)
    if len(indices) == 1:
        axes = axes[None, :]
    for row, module_idx in enumerate(indices):
        gt_img = np.full(mask.shape, np.nan, dtype=np.float32)
        pred_img = np.full(mask.shape, np.nan, dtype=np.float32)
        gt_img[mask] = gt_points[module_idx].reshape(-1)
        pred_img[mask] = pred_internal[module_idx, :, 0].reshape(-1)
        err_img = np.abs(pred_img - gt_img)
        for col, (image, title, cmap) in enumerate([(gt_img, f"M{module_idx} GT", "inferno"), (pred_img, f"M{module_idx} Pred", "inferno"), (err_img, f"M{module_idx} Error", "magma")]):
            im = axes[row, col].imshow(image, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap)
            axes[row, col].set_title(title)
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.savefig(str(output_path), dpi=170)
    plt.close(fig)
    return True


def plot_interface(output_path: Path, sample: Dict[str, Any], pred_interface: np.ndarray) -> bool:
    if pred_interface is None or np.asarray(pred_interface).size == 0 or np.asarray(pred_interface).shape[-2] == 0:
        return False
    present = sample["structure"]["module_present"] > 0.5
    indices = np.flatnonzero(present)[: min(3, int(np.sum(present)))]
    if len(indices) == 0:
        return False
    theta = sample["teacher_port_tokens"][0, :, 0]
    gt = sample["interface_target"]
    fig, axes = plt.subplots(len(indices), 2, figsize=(10.0, 3.0 * len(indices)), constrained_layout=True)
    if len(indices) == 1:
        axes = axes[None, :]
    for row, module_idx in enumerate(indices):
        for col, label in enumerate(["T_surface", "q_normal"]):
            ax = axes[row, col]
            ax.plot(theta, gt[module_idx, :, col], color="black", lw=1.7, label="GT")
            ax.plot(theta, pred_interface[module_idx, :, col], color="#d95f02", lw=1.4, label="Pred")
            ax.set_title(f"M{module_idx} {label}")
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.legend(fontsize=8)
    fig.savefig(str(output_path), dpi=170)
    plt.close(fig)
    return True
