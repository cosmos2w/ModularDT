from __future__ import annotations

"""Evaluate a Stage B global Channel Thermal checkpoint on one processed case."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from channelthermal_datasets import CHANNEL_ORDER, GlobalChannelThermalDataset
from channelthermal_model_utils import (
    current_timestamp,
    load_trusted_checkpoint,
    recursive_to_device,
    resolve_demo_path,
    select_device,
    strip_module_prefix,
    write_json,
)
from model import GlobalChannelThermalModel, GlobalChannelThermalModelConfig, load_local_surrogate_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Demo 1 global Channel Thermal model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint selector: best, latest/lastest, or a direct .pt path.",
    )
    parser.add_argument("--Run_ID", dest="run_id", type=str, default=None, help="Numeric run serial used to find the latest matching saved model, e.g. 0001.")
    parser.add_argument("--saved-root", type=str, default="./Saved_Model", help="Root directory containing global saved-model runs.")
    parser.add_argument("--dataset", type=str, default=None, help="Override packed global HDF5 path.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split.")
    parser.add_argument("--case-id", type=str, default=None, help="Processed dataset case id.")
    parser.add_argument("--case-index", type=int, default=0, help="Index within selected split when case-id is omitted.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for evaluation outputs.")
    parser.add_argument("--query-batch-size", type=int, default=32768, help="Grid query chunk size.")
    parser.add_argument(
        "--local-port-condition-mode",
        choices=["teacher", "predicted", "mixed", "both"],
        default="both",
        help="Evaluate teacher-forced, model-predicted, mixed, or both teacher and predicted port conditions.",
    )
    parser.add_argument("--mixed-teacher-ratio", type=float, default=0.5, help="Teacher-token ratio for mixed port evaluation.")
    return parser.parse_args()


def checkpoint_file_name(selector: str) -> str:
    cleaned = str(selector).strip().lower()
    if cleaned == "best":
        return "best_model.pt"
    if cleaned in {"latest", "lastest"}:
        return "latest_model.pt"
    raise ValueError("--checkpoint must be 'best', 'latest'/'lastest', or a direct checkpoint path.")


def normalize_run_id(value: str) -> str:
    raw = str(value).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be a numeric serial such as '0001'; got {raw!r}.")
    return f"{int(raw):04d}"


def latest_run_dir(saved_root: Path, run_id: str) -> Path:
    normalized = normalize_run_id(run_id)
    patterns = (f"Run_{normalized}_*", f"{normalized}_*", f"{normalized}*")
    matches = sorted({path for pattern in patterns for path in saved_root.glob(pattern) if path.is_dir()})
    if not matches:
        raise FileNotFoundError(f"No saved global runs found under {saved_root} with Run_ID={normalized!r}.")
    return matches[-1]


def resolve_checkpoint_arg(args: argparse.Namespace) -> Path:
    selector = str(args.checkpoint)
    if args.run_id:
        saved_root = resolve_demo_path(args.saved_root)
        run_dir = latest_run_dir(saved_root, args.run_id)
        return (run_dir / checkpoint_file_name(selector)).resolve()
    candidate = resolve_demo_path(selector)
    if candidate.suffix == ".pt" or candidate.exists():
        return candidate
    if selector.strip().lower() in {"best", "latest", "lastest"}:
        raise ValueError("--Run_ID is required when --checkpoint is 'best' or 'latest'.")
    saved_root = resolve_demo_path(args.saved_root)
    return (latest_run_dir(saved_root, selector) / "best_model.pt").resolve()


def numpy_to_batched_tensor(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).unsqueeze(0)
    if isinstance(value, dict):
        return {key: numpy_to_batched_tensor(item) for key, item in value.items()}
    return value


def make_batch(sample: Dict[str, Any], query_xy: np.ndarray, device: torch.device) -> Dict[str, Any]:
    payload = {key: value for key, value in sample.items() if key not in {"x_grid", "y_grid", "steady_field", "rms_field", "case_id"}}
    payload["query_xy"] = query_xy.astype(np.float32)
    return recursive_to_device(numpy_to_batched_tensor(payload), device)


def attach_local_surrogate(model: GlobalChannelThermalModel, checkpoint: Dict[str, Any], device: torch.device) -> None:
    if not model.config.use_local_surrogate:
        return
    train_cfg = checkpoint.get("train_config", {})
    local_path = train_cfg.get("model", {}).get("local_surrogate_checkpoint_path")
    if not local_path:
        raise ValueError("Checkpoint config uses the local surrogate but does not include local_surrogate_checkpoint_path.")
    local_model, local_checkpoint = load_local_surrogate_from_checkpoint(resolve_demo_path(local_path), map_location=device)
    normalization_config = local_checkpoint.get("local_normalization_config")
    if not isinstance(normalization_config, dict):
        dataset_cfg = local_checkpoint.get("train_config", {}).get("dataset", {})
        normalization_config = {
            "normalize_inputs": bool(dataset_cfg.get("normalize_inputs", False)),
            "normalize_targets": bool(dataset_cfg.get("normalize_targets", False)),
        }
    normalization_stats = local_checkpoint.get("local_normalization_stats", {})
    if not isinstance(normalization_stats, dict):
        normalization_stats = {}
    if bool(normalization_config.get("normalize_inputs", False)):
        missing = [key for key in ("module_params_mean", "module_params_std", "port_tokens_mean", "port_tokens_std") if key not in normalization_stats]
        if missing:
            raise ValueError(f"Local surrogate checkpoint was trained with normalized inputs but is missing stats: {missing}")
    if bool(normalization_config.get("normalize_targets", False)):
        missing = [
            key
            for key in (
                "internal_temperature_mean",
                "internal_temperature_std",
                "interface_targets_mean",
                "interface_targets_std",
            )
            if key not in normalization_stats
        ]
        if missing:
            raise ValueError(f"Local surrogate checkpoint was trained with normalized targets but is missing stats: {missing}")
    local_model.to(device)
    model.set_local_surrogate(
        local_model,
        freeze=bool(model.config.freeze_local_surrogate),
        normalization_config=normalization_config,
        normalization_stats=normalization_stats,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GlobalChannelThermalModel, Dict[str, Any]]:
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    model_config = GlobalChannelThermalModelConfig.from_dict(checkpoint.get("model_config", {}))
    model = GlobalChannelThermalModel(model_config).to(device)
    global_norm_cfg = checkpoint.get("global_normalization_config", {})
    if not isinstance(global_norm_cfg, dict):
        global_norm_cfg = checkpoint.get("train_config", {}).get("dataset", {})
    model.set_global_target_normalization(
        checkpoint.get("global_normalization_stats", {}),
        normalize_targets=bool(global_norm_cfg.get("normalize_targets", False)),
    )
    attach_local_surrogate(model, checkpoint, device)
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=False)
    model.eval()
    return model, checkpoint


def select_sample(dataset: GlobalChannelThermalDataset, case_id: Optional[str], case_index: int) -> Dict[str, Any]:
    if len(dataset) == 0:
        raise RuntimeError("No global channel thermal cases are available for evaluation.")
    if case_id is not None:
        for idx, candidate in enumerate(dataset.selected_case_ids):
            if str(candidate) == str(case_id):
                return dataset[idx]
        raise KeyError(f"case_id={case_id!r} not found in split {dataset.split!r}.")
    return dataset[min(max(int(case_index), 0), len(dataset) - 1)]


def predict_case(
    model: GlobalChannelThermalModel,
    sample: Dict[str, Any],
    device: torch.device,
    *,
    query_batch_size: int,
    local_port_condition_mode: str,
    mixed_teacher_ratio: float,
) -> Dict[str, Any]:
    x_grid = sample["x_grid"]
    y_grid = sample["y_grid"]
    query_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1).astype(np.float32)
    pred_chunks = []
    first_outputs = None
    with torch.no_grad():
        for start in range(0, query_xy.shape[0], int(query_batch_size)):
            chunk = query_xy[start : start + int(query_batch_size)]
            batch = make_batch(sample, chunk, device)
            outputs = model(
                batch["structure"],
                batch["query_xy"],
                interface_condition=batch["interface_condition"],
                local_module_params=batch["local_module_params"],
                teacher_port_tokens=batch["teacher_port_tokens"],
                local_query_points=batch["module_internal_query_points"],
                # Teacher mode is useful for debugging the global field with
                # exact boundary tokens. Predicted mode is the autonomous
                # forward-design setting because no solved teacher tokens are
                # available at inference time.
                local_port_condition_mode=local_port_condition_mode,
                mixed_teacher_ratio=float(mixed_teacher_ratio),
            )
            pred_chunks.append(outputs["pred_field"].detach().cpu().numpy()[0])
            if first_outputs is None:
                first_outputs = outputs
    pred_field = np.concatenate(pred_chunks, axis=0).reshape(*x_grid.shape, model.config.field_dim)
    assert first_outputs is not None
    return {
        "pred_field_grid": pred_field,
        "pred_internal_temperature": first_outputs["pred_internal_temperature"].detach().cpu().numpy()[0],
        "pred_interface": first_outputs["pred_interface"].detach().cpu().numpy()[0],
        "pred_port_condition": first_outputs["pred_port_condition"].detach().cpu().numpy()[0],
        "organizer_aux": {
            key: value.detach().cpu().numpy()[0] if torch.is_tensor(value) and value.ndim > 0 else value
            for key, value in first_outputs["organizer_aux"].items()
        },
    }


def channel_cmap(name: str) -> str:
    return {"u": "coolwarm", "v": "coolwarm", "p": "magma", "omega": "RdBu_r", "temperature": "inferno"}.get(name, "viridis")


def l2_error(prediction: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(diff.reshape(-1), ord=2))


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Return aggregate and per-value error metrics for arrays."""
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    diff = pred - gt
    flat_diff = diff.reshape(-1)
    flat_gt = gt.reshape(-1)
    l2_norm = float(np.linalg.norm(flat_diff, ord=2))
    mse = float(np.mean(flat_diff * flat_diff)) if flat_diff.size else float("nan")
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
    mae = float(np.mean(np.abs(flat_diff))) if flat_diff.size else float("nan")
    gt_norm = float(np.linalg.norm(flat_gt, ord=2))
    relative_l2 = float(l2_norm / max(gt_norm, 1e-12))
    return {
        "l2_norm": l2_norm,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "relative_l2": relative_l2,
        "num_values": float(flat_diff.size),
    }


def safe_path_name(value: object) -> str:
    raw = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return safe or "case"


def evaluation_output_dir(base_dir_arg: str | None, checkpoint_path: Path, case_id: object) -> Path:
    base_dir = Path(base_dir_arg) if base_dir_arg else checkpoint_path.parent / "eval_global"
    return resolve_demo_path(base_dir) / f"{safe_path_name(case_id)}_{current_timestamp()}"


def plot_field_quicklook(output_path: Path, sample: Dict[str, Any], pred_field: np.ndarray, channel_order: list[str]) -> None:
    gt = sample["steady_field"][..., : pred_field.shape[-1]]
    preferred = [name for name in ["temperature", "u", "omega"] if name in channel_order]
    if not preferred:
        preferred = channel_order[: min(3, len(channel_order))]
    x_min = float(np.min(sample["x_grid"]))
    x_max = float(np.max(sample["x_grid"]))
    y_min = float(np.min(sample["y_grid"]))
    y_max = float(np.max(sample["y_grid"]))
    extent = (x_min, x_max, y_min, y_max)
    domain_aspect = max((x_max - x_min) / max(y_max - y_min, 1.0e-12), 1.0e-6)
    box_aspect = 1.0 / domain_aspect
    panel_width = 3.5
    panel_height = max(1.9, panel_width / domain_aspect)
    fig, axes = plt.subplots(len(preferred), 3, figsize=(3.0 * panel_width, panel_height * len(preferred)), constrained_layout=True)
    if len(preferred) == 1:
        axes = axes[None, :]
    for row, name in enumerate(preferred):
        idx = channel_order.index(name)
        gt_img = gt[..., idx]
        pred_img = pred_field[..., idx]
        err_img = np.abs(pred_img - gt_img)
        channel_metrics = error_metrics(pred_img, gt_img)
        vmin = float(np.nanmin(gt_img))
        vmax = float(np.nanmax(gt_img))
        for col, (image, title, cmap) in enumerate(
            [
                (gt_img, f"GT {name}", channel_cmap(name)),
                (pred_img, f"Pred {name}", channel_cmap(name)),
                (err_img, f"Abs error {name}\nrelL2={channel_metrics['relative_l2']:.4e}", "magma"),
            ]
        ):
            im = axes[row, col].imshow(
                image,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin if col < 2 else None,
                vmax=vmax if col < 2 else None,
                aspect="equal",
            )
            axes[row, col].set_aspect("equal", adjustable="box")
            if hasattr(axes[row, col], "set_box_aspect"):
                axes[row, col].set_box_aspect(box_aspect)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def raster_from_points(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.full(mask.shape, np.nan, dtype=np.float32)
    image[mask.astype(bool)] = values.reshape(-1)
    return image


def plot_internal(output_path: Path, sample: Dict[str, Any], pred_internal: np.ndarray) -> None:
    present = sample["structure"]["module_present"] > 0.5
    indices = np.flatnonzero(present)[: min(3, int(np.sum(present)))]
    if len(indices) == 0 or pred_internal.shape[-2] == 0:
        return
    mask = sample["module_internal_mask"]
    gt_points = sample["module_internal_temperature_points"]
    fig, axes = plt.subplots(len(indices), 3, figsize=(9.8, 3.0 * len(indices)), constrained_layout=True)
    if len(indices) == 1:
        axes = axes[None, :]
    for row, module_idx in enumerate(indices):
        gt_img = raster_from_points(gt_points[module_idx], mask)
        pred_img = raster_from_points(pred_internal[module_idx, :, 0], mask)
        err_img = np.abs(pred_img - gt_img)
        module_metrics = error_metrics(pred_internal[module_idx, :, 0], gt_points[module_idx])
        vmin = float(np.nanmin(gt_img))
        vmax = float(np.nanmax(gt_img))
        for col, (image, title, cmap) in enumerate(
            [
                (gt_img, f"M{module_idx} GT", "inferno"),
                (pred_img, f"M{module_idx} Pred", "inferno"),
                (err_img, f"M{module_idx} Error\nRMSE={module_metrics['rmse']:.4e}, relL2={module_metrics['relative_l2']:.4e}", "magma"),
            ]
        ):
            im = axes[row, col].imshow(image, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, vmin=vmin if col < 2 else None, vmax=vmax if col < 2 else None)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("xi")
            axes[row, col].set_ylabel("eta")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_interface(output_path: Path, sample: Dict[str, Any], pred_interface: np.ndarray) -> None:
    present = sample["structure"]["module_present"] > 0.5
    indices = np.flatnonzero(present)[: min(3, int(np.sum(present)))]
    if len(indices) == 0:
        return
    theta = sample["teacher_port_tokens"][0, :, 0]
    gt = sample["interface_target"]
    fig, axes = plt.subplots(len(indices), 2, figsize=(10.0, 3.0 * len(indices)), constrained_layout=True)
    if len(indices) == 1:
        axes = axes[None, :]
    for row, module_idx in enumerate(indices):
        for col, label in enumerate(["T_surface", "q_normal"]):
            ax = axes[row, col]
            curve_metrics = error_metrics(pred_interface[module_idx, :, col], gt[module_idx, :, col])
            ax.plot(theta, gt[module_idx, :, col], color="black", lw=1.7, label="GT")
            ax.plot(theta, pred_interface[module_idx, :, col], color="#d95f02", lw=1.4, label="Pred")
            ax.set_title(f"M{module_idx} {label} RMSE={curve_metrics['rmse']:.4e}, relL2={curve_metrics['relative_l2']:.4e}")
            ax.set_xlabel("theta")
            ax.grid(True, alpha=0.25)
            if row == 0 and col == 0:
                ax.legend()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_organizer(output_path: Path, sample: Dict[str, Any], aux: Dict[str, Any], model: GlobalChannelThermalModel) -> None:
    centers = sample["structure"]["module_centers"]
    present = sample["structure"]["module_present"] > 0.5
    heat = np.asarray(sample["structure"].get("heat_powers", np.zeros((centers.shape[0],))), dtype=np.float32)
    env_coords = np.asarray(aux.get("env_coords", model.env_coords.detach().cpu().numpy()), dtype=np.float32)
    A_eh = np.asarray(aux.get("A_eh", np.zeros((env_coords.shape[0], 1))), dtype=np.float32)
    A_mh = np.asarray(aux.get("A_mh", np.zeros((centers.shape[0], A_eh.shape[-1]))), dtype=np.float32)
    strength = np.asarray(aux.get("hyper_strength", np.ones((A_eh.shape[-1],), dtype=np.float32)), dtype=np.float32)
    module_mass = np.asarray(aux.get("hyper_module_mass", np.zeros_like(strength)), dtype=np.float32)
    env_mass = np.asarray(aux.get("hyper_env_mass", np.zeros_like(strength)), dtype=np.float32)
    src = np.asarray(aux.get("hyper_source_coords", np.zeros((strength.shape[0], 2))), dtype=np.float32)
    dst = np.asarray(aux.get("hyper_thermal_region_coords", np.zeros((strength.shape[0], 2))), dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)
    ax_overlay, ax_mh, ax_eh, ax_schema = axes.reshape(-1)
    temp_idx = CHANNEL_ORDER.index("temperature") if "temperature" in CHANNEL_ORDER else min(sample["steady_field"].shape[-1] - 1, 0)

    ax_overlay.imshow(
        sample["steady_field"][..., temp_idx],
        origin="lower",
        extent=(float(np.min(sample["x_grid"])), float(np.max(sample["x_grid"])), float(np.min(sample["y_grid"])), float(np.max(sample["y_grid"]))),
        cmap="inferno",
        alpha=0.78,
        aspect="auto",
    )
    num_h = max(A_eh.shape[-1], 1)
    cmap = plt.get_cmap("tab10", num_h)
    dominant_env = A_eh.argmax(axis=-1) if A_eh.size else np.zeros((env_coords.shape[0],), dtype=np.int64)
    ax_overlay.scatter(env_coords[:, 0], env_coords[:, 1], c=dominant_env, cmap=cmap, s=24, edgecolor="white", linewidth=0.25, alpha=0.9)
    heat_abs = np.abs(heat)
    heat_scale = heat_abs / max(float(np.nanmax(heat_abs)) if heat_abs.size else 0.0, 1.0e-6)
    for module_idx in np.flatnonzero(present):
        cx, cy = centers[module_idx]
        color = "#fdae61" if heat[module_idx] >= 0 else "#74add1"
        radius = float(model.config.module_radius)
        ax_overlay.add_patch(plt.Circle((float(cx), float(cy)), radius, fill=False, color=color, lw=1.2 + 1.4 * float(heat_scale[module_idx])))
        ax_overlay.text(float(cx), float(cy), f"M{module_idx}", ha="center", va="center", color="white", fontsize=8, weight="bold")
    for hidx in range(strength.shape[0]):
        alpha = float(np.clip(strength[hidx], 0.12, 1.0))
        color = cmap(hidx)
        ax_overlay.plot([src[hidx, 0], dst[hidx, 0]], [src[hidx, 1], dst[hidx, 1]], color=color, lw=1.0 + 2.0 * alpha, alpha=alpha)
        ax_overlay.scatter(src[hidx, 0], src[hidx, 1], marker="x", s=35 + 70 * alpha, color=color, linewidth=1.5)
        ax_overlay.scatter(dst[hidx, 0], dst[hidx, 1], marker="o", s=34 + 95 * alpha, color=color, edgecolor="black", linewidth=0.45)
        ax_overlay.text(dst[hidx, 0], dst[hidx, 1], f"H{hidx}\n{strength[hidx]:.2f}", color="white", fontsize=7, ha="center", va="center")
    ax_overlay.set_title("Physical organizer overlay")
    ax_overlay.set_xlabel("x")
    ax_overlay.set_ylabel("y")

    im1 = ax_mh.imshow(A_mh.T, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(A_mh)) if A_mh.size else 1.0, 1.0e-6))
    ax_mh.set_title("A_mh modules x hyperedges")
    ax_mh.set_xlabel("module")
    ax_mh.set_ylabel("hyperedge")
    ax_mh.set_xticks(np.arange(centers.shape[0]))
    ax_mh.set_xticklabels([f"M{i}" for i in range(centers.shape[0])], rotation=45, ha="right")
    fig.colorbar(im1, ax=ax_mh, fraction=0.046, pad=0.04)

    sort_idx = np.lexsort((np.arange(A_eh.shape[0]), dominant_env)) if A_eh.size else np.arange(0)
    im2 = ax_eh.imshow(A_eh[sort_idx].T if A_eh.size else A_eh.T, aspect="auto", cmap="viridis", vmin=0.0)
    ax_eh.set_title("A_eh env tokens sorted by dominant hyperedge")
    ax_eh.set_xlabel("sorted env token")
    ax_eh.set_ylabel("hyperedge")
    fig.colorbar(im2, ax=ax_eh, fraction=0.046, pad=0.04)

    ax_schema.set_title("Module/environment mass")
    y = np.arange(strength.shape[0])
    ax_schema.barh(y - 0.18, module_mass, height=0.34, color="#4c78a8", label="module mass")
    ax_schema.barh(y + 0.18, env_mass, height=0.34, color="#f58518", label="env mass")
    ax_schema.set_yticks(y)
    ax_schema.set_yticklabels([f"H{i}" for i in range(strength.shape[0])])
    ax_schema.set_xlabel("normalized mass")
    ax_schema.grid(True, axis="x", alpha=0.25)
    ax_schema.legend(fontsize=8)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_organizer_summary(output_dir: Path, aux: Dict[str, Any], sample: Dict[str, Any]) -> tuple[Path, Path]:
    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"] > 0.5)
    A_mh = np.asarray(aux.get("A_mh", np.zeros((centers.shape[0], 0))), dtype=np.float32)
    A_eh = np.asarray(aux.get("A_eh", np.zeros((0, A_mh.shape[-1] if A_mh.ndim == 2 else 0))), dtype=np.float32)
    module_mass = np.asarray(aux.get("hyper_module_mass", np.zeros((A_mh.shape[-1],))), dtype=np.float32)
    env_mass = np.asarray(aux.get("hyper_env_mass", np.zeros_like(module_mass)), dtype=np.float32)
    strength = np.asarray(aux.get("hyper_strength", np.zeros_like(module_mass)), dtype=np.float32)
    src = np.asarray(aux.get("hyper_source_coords", np.zeros((module_mass.shape[0], 2))), dtype=np.float32)
    dst = np.asarray(aux.get("hyper_thermal_region_coords", np.zeros((module_mass.shape[0], 2))), dtype=np.float32)
    dominant_env = A_eh.argmax(axis=-1) if A_eh.size else np.zeros((0,), dtype=np.int64)
    rows = []
    for hidx in range(module_mass.shape[0]):
        module_scores = A_mh[:, hidx] if A_mh.size else np.zeros((centers.shape[0],), dtype=np.float32)
        valid_scores = [(idx, float(module_scores[idx])) for idx in np.flatnonzero(present)]
        valid_scores.sort(key=lambda item: item[1], reverse=True)
        top_modules = [f"M{idx}:{score:.3f}" for idx, score in valid_scores[:4] if score > 1.0e-6]
        row = {
            "hyperedge_id": int(hidx),
            "module_mass": float(module_mass[hidx]),
            "env_mass": float(env_mass[hidx]),
            "hyper_strength": float(strength[hidx]),
            "source_x": float(src[hidx, 0]),
            "source_y": float(src[hidx, 1]),
            "thermal_region_x": float(dst[hidx, 0]),
            "thermal_region_y": float(dst[hidx, 1]),
            "top_modules": ";".join(top_modules),
            "top_env_token_count": int(np.sum(dominant_env == hidx)),
        }
        rows.append(row)
    csv_path = output_dir / "organization_summary.csv"
    json_path = output_dir / "organization_summary.json"
    fieldnames = [
        "hyperedge_id",
        "module_mass",
        "env_mass",
        "hyper_strength",
        "source_x",
        "source_y",
        "thermal_region_x",
        "thermal_region_y",
        "top_modules",
        "top_env_token_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, {"hyperedges": rows})
    return csv_path, json_path


def mode_suffix(mode: str) -> str:
    return "predicted" if str(mode).lower() == "predicted" else str(mode).lower()


def denormalize_predictions(
    predictions: Dict[str, Any],
    dataset: GlobalChannelThermalDataset,
    normalize_targets: bool,
) -> Dict[str, Any]:
    if not normalize_targets:
        return predictions
    out = dict(predictions)
    out["pred_field_grid"] = dataset.normalizer.denormalize_fields(out["pred_field_grid"])
    out["pred_internal_temperature"] = dataset.normalizer.denormalize_internal_temperature(out["pred_internal_temperature"])
    out["pred_interface"] = dataset.normalizer.denormalize_interface_targets(out["pred_interface"])
    return out


def summarize_prediction(
    checkpoint_path: Path,
    raw_sample: Dict[str, Any],
    predictions: Dict[str, Any],
    output_dir: Path,
    suffix: str,
    channel_order: list[str],
) -> Dict[str, Any]:
    pred_field = predictions["pred_field_grid"]
    gt_field = raw_sample["steady_field"][..., : pred_field.shape[-1]]
    pred_internal = predictions["pred_internal_temperature"]
    gt_internal = raw_sample["module_internal_temperature_points"]
    pred_interface = predictions["pred_interface"]
    gt_interface = raw_sample["interface_target"]
    npz_path = output_dir / f"evaluation_outputs_{suffix}.npz"
    np.savez_compressed(
        npz_path,
        pred_field_grid=pred_field.astype(np.float32),
        gt_field_grid=gt_field.astype(np.float32),
        pred_internal_temperature=pred_internal.astype(np.float32),
        gt_internal_temperature=gt_internal.astype(np.float32),
        pred_interface=pred_interface.astype(np.float32),
        gt_interface=gt_interface.astype(np.float32),
        pred_port_condition=predictions["pred_port_condition"].astype(np.float32),
    )
    outputs = {
        "global_field_quicklook": str(output_dir / f"global_field_quicklook_{suffix}.png"),
        "module_internal_temperature": str(output_dir / f"module_internal_temperature_{suffix}.png"),
        "interface_curves": str(output_dir / f"interface_curves_{suffix}.png"),
        "npz": str(npz_path),
    }
    field_channel_l2 = {
        str(name): l2_error(pred_field[..., idx], gt_field[..., idx])
        for idx, name in enumerate(channel_order[: pred_field.shape[-1]])
    }
    field_channel_rmse = {
        str(name): error_metrics(pred_field[..., idx], gt_field[..., idx])["rmse"]
        for idx, name in enumerate(channel_order[: pred_field.shape[-1]])
    }
    field_channel_relative_l2 = {
        str(name): error_metrics(pred_field[..., idx], gt_field[..., idx])["relative_l2"]
        for idx, name in enumerate(channel_order[: pred_field.shape[-1]])
    }
    interface_l2_by_target = {
        "T_surface": l2_error(pred_interface[..., 0], gt_interface[..., 0]),
        "q_normal": l2_error(pred_interface[..., 1], gt_interface[..., 1]),
    }
    interface_rmse_by_target = {
        "T_surface": error_metrics(pred_interface[..., 0], gt_interface[..., 0])["rmse"],
        "q_normal": error_metrics(pred_interface[..., 1], gt_interface[..., 1])["rmse"],
    }
    interface_relative_l2_by_target = {
        "T_surface": error_metrics(pred_interface[..., 0], gt_interface[..., 0])["relative_l2"],
        "q_normal": error_metrics(pred_interface[..., 1], gt_interface[..., 1])["relative_l2"],
    }
    field_metrics = error_metrics(pred_field, gt_field)
    temperature_metrics = error_metrics(pred_field[..., 4], gt_field[..., 4]) if pred_field.shape[-1] >= 5 else None
    internal_metrics = error_metrics(pred_internal.reshape(-1), gt_internal.reshape(-1))
    interface_metrics = error_metrics(pred_interface, gt_interface)
    return {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "metric_note": "l2_error is the aggregate Euclidean norm over all values; relative_l2 is l2_error divided by target L2 norm.",
        "field_l2_error": field_metrics["l2_norm"],
        "temperature_l2_error": temperature_metrics["l2_norm"] if temperature_metrics is not None else None,
        "internal_l2_error": internal_metrics["l2_norm"],
        "interface_l2_error": interface_metrics["l2_norm"],
        "field_rmse": field_metrics["rmse"],
        "temperature_rmse": temperature_metrics["rmse"] if temperature_metrics is not None else None,
        "internal_rmse": internal_metrics["rmse"],
        "interface_rmse": interface_metrics["rmse"],
        "field_relative_l2": field_metrics["relative_l2"],
        "temperature_relative_l2": temperature_metrics["relative_l2"] if temperature_metrics is not None else None,
        "internal_relative_l2": internal_metrics["relative_l2"],
        "interface_relative_l2": interface_metrics["relative_l2"],
        "field_channel_l2_error": field_channel_l2,
        "field_channel_rmse": field_channel_rmse,
        "field_channel_relative_l2": field_channel_relative_l2,
        "interface_l2_error_by_target": interface_l2_by_target,
        "interface_rmse_by_target": interface_rmse_by_target,
        "interface_relative_l2_by_target": interface_relative_l2_by_target,
        "field_metrics": field_metrics,
        "temperature_metrics": temperature_metrics,
        "internal_metrics": internal_metrics,
        "interface_metrics": interface_metrics,
        "channel_order": channel_order,
        "outputs": outputs,
    }


def evaluate_mode(
    *,
    mode: str,
    model: GlobalChannelThermalModel,
    sample: Dict[str, Any],
    raw_sample: Dict[str, Any],
    dataset: GlobalChannelThermalDataset,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    query_batch_size: int,
    mixed_teacher_ratio: float,
    normalize_targets: bool,
    channel_order: list[str],
) -> Dict[str, Any]:
    suffix = mode_suffix(mode)
    predictions = predict_case(
        model,
        sample,
        device,
        query_batch_size=query_batch_size,
        local_port_condition_mode=mode,
        mixed_teacher_ratio=mixed_teacher_ratio,
    )
    predictions = denormalize_predictions(predictions, dataset, normalize_targets)
    plot_field_quicklook(output_dir / f"global_field_quicklook_{suffix}.png", raw_sample, predictions["pred_field_grid"], channel_order)
    plot_internal(output_dir / f"module_internal_temperature_{suffix}.png", raw_sample, predictions["pred_internal_temperature"])
    plot_interface(output_dir / f"interface_curves_{suffix}.png", raw_sample, predictions["pred_interface"])
    summary = summarize_prediction(checkpoint_path, raw_sample, predictions, output_dir, suffix, channel_order)
    summary["_organizer_aux"] = predictions["organizer_aux"]
    return summary


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_arg(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = select_device(args.device)
    model, checkpoint = load_model(checkpoint_path, device)
    train_cfg = checkpoint.get("train_config", {})
    dataset_cfg = train_cfg.get("dataset", {})
    dataset_path = args.dataset or dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5")
    dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=args.split,
        points_per_case=1,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
    )
    raw_dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=args.split,
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=True,
    )
    if len(dataset) == 0:
        dataset = GlobalChannelThermalDataset(dataset_path, split="all", points_per_case=1, include_grid=True)
        raw_dataset = GlobalChannelThermalDataset(dataset_path, split="all", points_per_case=1, include_grid=True)
    sample = select_sample(dataset, args.case_id, args.case_index)
    raw_sample = select_sample(raw_dataset, str(sample["case_id"]), args.case_index)

    output_dir = evaluation_output_dir(args.output_dir, checkpoint_path, raw_sample["case_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_order = dataset.channel_order or list(CHANNEL_ORDER)
    normalize_targets = bool(dataset_cfg.get("normalize_targets", False))
    requested_modes = ["teacher", "predicted"] if args.local_port_condition_mode == "both" else [args.local_port_condition_mode]
    mode_summaries: Dict[str, Dict[str, Any]] = {}
    first_aux: Optional[Dict[str, Any]] = None
    for mode in requested_modes:
        mode_summary = evaluate_mode(
            mode=mode,
            model=model,
            sample=sample,
            raw_sample=raw_sample,
            dataset=dataset,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=device,
            query_batch_size=args.query_batch_size,
            mixed_teacher_ratio=float(args.mixed_teacher_ratio),
            normalize_targets=normalize_targets,
            channel_order=channel_order,
        )
        if first_aux is None:
            first_aux = mode_summary.pop("_organizer_aux", None)
        else:
            mode_summary.pop("_organizer_aux", None)
        mode_summaries[mode_suffix(mode)] = mode_summary

    if first_aux is not None:
        plot_organizer(output_dir / "organizer_visualization.png", raw_sample, first_aux, model)
        org_csv, org_json = save_organizer_summary(output_dir, first_aux, raw_sample)
    else:
        org_csv = output_dir / "organization_summary.csv"
        org_json = output_dir / "organization_summary.json"

    summary = {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "local_port_condition_mode": args.local_port_condition_mode,
        "mixed_teacher_ratio": float(args.mixed_teacher_ratio),
        "outputs": {
            "organizer_visualization": str(output_dir / "organizer_visualization.png"),
            "organization_summary_csv": str(org_csv),
            "organization_summary_json": str(org_json),
        },
        "modes": mode_summaries,
    }
    if args.local_port_condition_mode == "both":
        teacher = mode_summaries["teacher"]
        predicted = mode_summaries["predicted"]
        summary.update(
            {
                # "teacher_field_l2_error": teacher["field_l2_error"],
                # "predicted_field_l2_error": predicted["field_l2_error"],
                "teacher_field_relative_l2": teacher["field_relative_l2"],
                "predicted_field_relative_l2": predicted["field_relative_l2"],
                # "teacher_temperature_l2_error": teacher["temperature_l2_error"],
                # "predicted_temperature_l2_error": predicted["temperature_l2_error"],
                "teacher_temperature_relative_l2": teacher["temperature_relative_l2"],
                "predicted_temperature_relative_l2": predicted["temperature_relative_l2"],
                # "teacher_internal_l2_error": teacher["internal_l2_error"],
                # "predicted_internal_l2_error": predicted["internal_l2_error"],
                "teacher_internal_relative_l2": teacher["internal_relative_l2"],
                "predicted_internal_relative_l2": predicted["internal_relative_l2"],
                # "teacher_interface_l2_error": teacher["interface_l2_error"],
                # "predicted_interface_l2_error": predicted["interface_l2_error"],
                "teacher_interface_relative_l2": teacher["interface_relative_l2"],
                "predicted_interface_relative_l2": predicted["interface_relative_l2"],
            }
        )
    else:
        suffix = mode_suffix(args.local_port_condition_mode)
        only = mode_summaries[suffix]
        summary.update(
            {
                "field_l2_error": only["field_l2_error"],
                "field_relative_l2": only["field_relative_l2"],
                "temperature_l2_error": only["temperature_l2_error"],
                "temperature_relative_l2": only["temperature_relative_l2"],
                "internal_l2_error": only["internal_l2_error"],
                "internal_relative_l2": only["internal_relative_l2"],
                "interface_l2_error": only["interface_l2_error"],
                "interface_relative_l2": only["interface_relative_l2"],
                f"{suffix}_field_l2_error": only["field_l2_error"],
                f"{suffix}_field_relative_l2": only["field_relative_l2"],
                f"{suffix}_temperature_l2_error": only["temperature_l2_error"],
                f"{suffix}_temperature_relative_l2": only["temperature_relative_l2"],
                f"{suffix}_internal_l2_error": only["internal_l2_error"],
                f"{suffix}_internal_relative_l2": only["internal_relative_l2"],
                f"{suffix}_interface_l2_error": only["interface_l2_error"],
                f"{suffix}_interface_relative_l2": only["interface_relative_l2"],
            }
        )
    write_json(output_dir / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
