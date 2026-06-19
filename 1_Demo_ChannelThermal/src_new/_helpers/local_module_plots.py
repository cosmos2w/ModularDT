"""CHANNELTHERMAL-SPECIFIC local surrogate plotting helpers.

Inputs are Stage-A local-module samples and predictions. Outputs are PNG plots
for internal temperature disks and interface curves. This module is
ChannelThermal-specific because it assumes circular local coordinates,
theta-ordered port tokens, and two interface targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _helpers.evaluation_plots import error_metrics


def raster_from_points(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.full(mask.shape, np.nan, dtype=np.float32)
    image[mask.astype(bool)] = values.reshape(-1)
    return image


def plot_local_internal(output_path: Path, sample: Dict[str, Any], pred_internal: np.ndarray, metrics: Dict[str, float]) -> None:
    target = sample["internal_temperature_targets"].reshape(-1)
    pred = pred_internal.reshape(-1)
    mask = sample.get("local_mask")
    if mask is None:
        return
    gt_img = raster_from_points(target, mask)
    pred_img = raster_from_points(pred, mask)
    err_img = np.abs(pred_img - gt_img)
    vmin = float(np.nanmin(gt_img))
    vmax = float(np.nanmax(gt_img))
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    for col, (ax, image, title, cmap) in enumerate(
        [
            (axes[0], gt_img, "GT internal T", "inferno"),
            (axes[1], pred_img, "Pred internal T", "inferno"),
            (axes[2], err_img, f"Abs error\nRMSE={metrics['rmse']:.4e}", "magma"),
        ]
    ):
        im = ax.imshow(image, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, vmin=vmin if col < 2 else None, vmax=vmax if col < 2 else None)
        ax.set_title(title)
        ax.set_xlabel("xi")
        ax.set_ylabel("eta")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(str(output_path), dpi=170)
    plt.close(fig)


def plot_local_interface(output_path: Path, sample: Dict[str, Any], pred_interface: np.ndarray) -> None:
    theta = sample["port_tokens"][:, 0]
    target = sample["interface_targets"]
    rough = sample.get("local_target_roughness", np.zeros((4,), dtype=np.float32))
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(theta, sample["port_tokens"][:, 3], color="#4c78a8", lw=1.5, label="T_env")
    axes[0].set_ylabel("T_env")
    ax_h = axes[0].twinx()
    ax_h.plot(theta, sample["port_tokens"][:, 4], color="#f58518", lw=1.3, label="h")
    ax_h.set_ylabel("h")
    axes[0].grid(True, alpha=0.25)
    for idx, ax in enumerate(axes[1:]):
        label = ["T_surface", "q_normal"][idx]
        channel_metrics = error_metrics(pred_interface[:, idx], target[:, idx])
        ax.plot(theta, target[:, idx], color="black", lw=1.8, label="GT")
        ax.plot(theta, pred_interface[:, idx], color="#d95f02", lw=1.5, label="Pred")
        if "interface_targets_raw" in sample:
            ax.plot(theta, sample["interface_targets_raw"][:, idx], color="#7f7f7f", lw=1.0, alpha=0.65, label="raw target")
        ax.set_ylabel(label)
        ax.set_title(
            f"{label} RMSE={channel_metrics['rmse']:.4e}, relL2={channel_metrics['relative_l2']:.4e}, "
            f"rough={float(rough[idx]):.3f}"
        )
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("theta")
    axes[1].legend()
    fig.savefig(str(output_path), dpi=170)
    plt.close(fig)
