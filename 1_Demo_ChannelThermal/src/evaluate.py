from __future__ import annotations

"""Evaluate a Stage B global Channel Thermal checkpoint on one processed case."""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from organizer_viz_channelthermal import (
    render_channelthermal_organization_overview,
    render_channelthermal_organization_schematic_presentation,
    render_channelthermal_organization_summary_matrices,
)


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
    parser.add_argument(
        "--temperature-display-mode",
        choices=["fluid_only", "composite_internal"],
        default=None,
        help="Quicklook temperature display: mask module interiors or composite fluid field with module-internal temperatures.",
    )
    parser.add_argument(
        "--organization-view",
        choices=["all", "physical", "matrices", "schematic", "none"],
        default="all",
        help="Organizer diagnostics to render.",
    )
    parser.add_argument(
        "--organization-style",
        choices=["presentation", "debug", "both"],
        default="presentation",
        help="Organizer visualization style: clear presentation plots, dense raw debug plots, or both.",
    )
    parser.add_argument(
        "--organization-link-threshold",
        type=float,
        default=0.25,
        help="Minimum module-to-hyperedge assignment A_mh drawn as a visual link.",
    )
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
    result = {
        "pred_field_grid": pred_field,
        "pred_internal_temperature": first_outputs["pred_internal_temperature"].detach().cpu().numpy()[0],
        "pred_interface": first_outputs["pred_interface"].detach().cpu().numpy()[0],
        "pred_port_condition": first_outputs["pred_port_condition"].detach().cpu().numpy()[0],
        "interface_flux_mode": str(model.config.local_surrogate_flux_mode) if model.config.use_local_surrogate else "global_head",
        "organizer_aux": {
            key: value.detach().cpu().numpy()[0] if torch.is_tensor(value) and value.ndim > 0 else value
            for key, value in first_outputs["organizer_aux"].items()
        },
    }
    for key in ("pred_interface_surrogate_raw", "pred_interface_flux_physics", "pred_interface_delta_q"):
        if key in first_outputs:
            result[key] = first_outputs[key].detach().cpu().numpy()[0]
    return result


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


def masked_error_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Return metrics on cells selected by a boolean mask."""
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
    """Build evaluation masks so global field metrics ignore solid interiors."""
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
        ax.add_patch(plt.Circle((float(cx), float(cy)), radius, fill=False, color=color, lw=linewidth))


def _local_disk_image(values: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
    image = np.full(local_mask.shape, np.nan, dtype=np.float32)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    image[np.asarray(local_mask, dtype=bool)] = flat[: int(np.sum(local_mask))]
    return image


def composite_temperature_grid(
    sample: Dict[str, Any],
    global_temperature: np.ndarray,
    internal_temperature_points: np.ndarray,
) -> np.ndarray:
    """Composite fluid global temperature with module-internal disk values.

    The global neural field represents the fluid/channel environment. For this
    display we project each module's local solid-temperature disk back into the
    matching global module footprint and leave fluid cells from the global grid.
    """
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


def resolve_temperature_display_mode(requested: Optional[str], predictions: Dict[str, Any]) -> str:
    if requested is not None:
        return str(requested)
    pred_internal = predictions.get("pred_internal_temperature")
    if pred_internal is not None and np.asarray(pred_internal).size > 0:
        return "composite_internal"
    return "fluid_only"


def safe_path_name(value: object) -> str:
    raw = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return safe or "case"


def evaluation_output_dir(base_dir_arg: str | None, checkpoint_path: Path, case_id: object) -> Path:
    base_dir = Path(base_dir_arg) if base_dir_arg else checkpoint_path.parent / "eval_global"
    return resolve_demo_path(base_dir) / f"{safe_path_name(case_id)}_{current_timestamp()}"


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
    module_mask, fluid_mask = module_and_fluid_masks(sample, pred_field)
    for row, name in enumerate(preferred):
        idx = channel_order.index(name)
        gt_img = gt[..., idx]
        pred_img = pred_field[..., idx]
        if name == "temperature" and str(temperature_display_mode) == "composite_internal" and pred_internal_temperature is not None:
            gt_img = composite_temperature_grid(sample, gt_img, sample["module_internal_temperature_points"])
            pred_img = composite_temperature_grid(sample, pred_img, pred_internal_temperature[..., 0])
            metric_mask = np.ones_like(fluid_mask, dtype=bool)
        else:
            # The global neural field is not trained to predict arbitrary flow
            # values inside solid modules, so quicklooks mask those interiors.
            gt_img = np.where(fluid_mask, gt_img, np.nan)
            pred_img = np.where(fluid_mask, pred_img, np.nan)
            metric_mask = fluid_mask
        err_img = np.abs(pred_img - gt_img)
        channel_metrics = masked_error_metrics(pred_img, gt_img, metric_mask)
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
            im = axes[row, col].imshow(
                image,
                origin="lower",
                extent=extent,
                cmap=cm,
                vmin=vmin if col < 2 else None,
                vmax=vmax if col < 2 else None,
                aspect="equal",
            )
            draw_module_outlines(axes[row, col], sample, color="#e6e6e6", linewidth=0.9)
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


def plot_interface(
    output_path: Path,
    sample: Dict[str, Any],
    pred_interface: np.ndarray,
    pred_interface_surrogate_raw: Optional[np.ndarray] = None,
    pred_interface_flux_physics: Optional[np.ndarray] = None,
    interface_flux_mode: str = "unknown",
) -> None:
    present = sample["structure"]["module_present"] > 0.5
    indices = np.flatnonzero(present)[: min(3, int(np.sum(present)))]
    if len(indices) == 0:
        return
    theta = sample["teacher_port_tokens"][0, :, 0]
    gt = sample["interface_target"]
    fig, axes = plt.subplots(len(indices), 2, figsize=(10.0, 3.0 * len(indices)), constrained_layout=True)
    mode = str(interface_flux_mode)
    if mode == "physics_from_port":
        note = "Pred q_normal = physics_from_port"
    elif mode == "corrected_physics":
        note = "Pred q_normal = physics + learned delta_q"
    else:
        note = f"Pred q_normal mode = {mode}"
    fig.suptitle(note, fontsize=10)
    if len(indices) == 1:
        axes = axes[None, :]
    for row, module_idx in enumerate(indices):
        for col, label in enumerate(["T_surface", "q_normal"]):
            ax = axes[row, col]
            curve_metrics = error_metrics(pred_interface[module_idx, :, col], gt[module_idx, :, col])
            ax.plot(theta, gt[module_idx, :, col], color="black", lw=1.7, label="GT")
            ax.plot(theta, pred_interface[module_idx, :, col], color="#d95f02", lw=1.4, label="Pred")
            if col == 1 and pred_interface_surrogate_raw is not None:
                ax.plot(theta, pred_interface_surrogate_raw[module_idx, :, col], color="#7570b3", lw=1.0, alpha=0.85, label="surrogate raw")
            if col == 1 and pred_interface_flux_physics is not None:
                ax.plot(theta, pred_interface_flux_physics[module_idx, :, col], color="#1b9e77", lw=1.0, alpha=0.85, label="physics")
            ax.set_title(f"M{module_idx} {label} RMSE={curve_metrics['rmse']:.4e}, relL2={curve_metrics['relative_l2']:.4e}")
            ax.set_xlabel("theta")
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def extract_channelthermal_organization_arrays(
    sample: Dict[str, Any],
    aux: Dict[str, Any],
    model: GlobalChannelThermalModel,
) -> Dict[str, np.ndarray]:
    """Collect organizer arrays with defaults suitable for visualization."""
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
    return {
        "centers": np.asarray(centers, dtype=np.float32),
        "present": np.asarray(present, dtype=bool),
        "heat": heat,
        "env_coords": env_coords,
        "A_eh": A_eh,
        "A_mh": A_mh,
        "strength": strength,
        "module_mass": module_mass,
        "env_mass": env_mass,
        "src": src,
        "dst": dst,
    }


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)})
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float32)

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def compute_channelthermal_hyperedge_summary(arrays: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    A_mh = arrays["A_mh"]
    A_eh = arrays["A_eh"]
    present = arrays["present"]
    module_mass = arrays["module_mass"]
    env_mass = arrays["env_mass"]
    strength = arrays["strength"]
    src = arrays["src"]
    dst = arrays["dst"]
    dominant_env = A_eh.argmax(axis=-1) if A_eh.size else np.zeros((0,), dtype=np.int64)
    rows: List[Dict[str, Any]] = []
    for hidx in range(module_mass.shape[0]):
        module_scores = A_mh[:, hidx] if A_mh.size else np.zeros((present.shape[0],), dtype=np.float32)
        valid_scores = [(idx, float(module_scores[idx])) for idx in np.flatnonzero(present)]
        valid_scores.sort(key=lambda item: item[1], reverse=True)
        top = [(idx, score) for idx, score in valid_scores[:4] if score > 1.0e-6]
        rows.append(
            {
                "hyperedge_id": int(hidx),
                "module_mass": float(module_mass[hidx]),
                "env_mass": float(env_mass[hidx]),
                "hyper_strength": float(strength[hidx]),
                "top_modules": ";".join(f"M{idx}" for idx, _ in top),
                "top_module_weights": ";".join(f"{score:.4f}" for _, score in top),
                "env_token_count": int(np.sum(dominant_env == hidx)),
                "dominant_env_count": int(np.sum(dominant_env == hidx)),
                "source_x": float(src[hidx, 0]),
                "source_y": float(src[hidx, 1]),
                "thermal_region_x": float(dst[hidx, 0]),
                "thermal_region_y": float(dst[hidx, 1]),
                "active": bool(strength[hidx] >= 0.05),
                "low_strength": bool(strength[hidx] < 0.05),
            }
        )
    return rows


def render_channelthermal_physical_organization(
    output_path: Path,
    sample: Dict[str, Any],
    aux: Dict[str, Any],
    model: GlobalChannelThermalModel,
) -> None:
    arrays = extract_channelthermal_organization_arrays(sample, aux, model)
    centers = arrays["centers"]
    present = arrays["present"]
    heat = arrays["heat"]
    env_coords = arrays["env_coords"]
    A_eh = arrays["A_eh"]
    A_mh = arrays["A_mh"]
    strength = arrays["strength"]
    src = arrays["src"]
    dst = arrays["dst"]

    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    temp_idx = CHANNEL_ORDER.index("temperature") if "temperature" in CHANNEL_ORDER else min(sample["steady_field"].shape[-1] - 1, 0)
    ax.imshow(
        sample["steady_field"][..., temp_idx],
        origin="lower",
        extent=(float(np.min(sample["x_grid"])), float(np.max(sample["x_grid"])), float(np.min(sample["y_grid"])), float(np.max(sample["y_grid"]))),
        cmap="inferno",
        alpha=0.42,
        aspect="auto",
    )
    num_h = max(A_eh.shape[-1], 1)
    cmap = plt.get_cmap("tab10", num_h)
    dominant_env = A_eh.argmax(axis=-1) if A_eh.size else np.zeros((env_coords.shape[0],), dtype=np.int64)
    confidence = A_eh.max(axis=-1) if A_eh.size else np.ones((env_coords.shape[0],), dtype=np.float32)
    for hidx in range(num_h):
        pts = env_coords[dominant_env == hidx]
        if pts.shape[0] >= 3:
            hull = _convex_hull(pts)
            if hull.shape[0] >= 3:
                ax.fill(hull[:, 0], hull[:, 1], color=cmap(hidx), alpha=0.10, lw=0.0)
    ax.scatter(
        env_coords[:, 0],
        env_coords[:, 1],
        c=dominant_env,
        cmap=cmap,
        s=18.0 + 38.0 * confidence,
        edgecolor="white",
        linewidth=0.25,
        alpha=np.clip(0.25 + 0.75 * confidence, 0.25, 1.0),
    )
    heat_abs = np.abs(heat)
    heat_scale = heat_abs / max(float(np.nanmax(heat_abs)) if heat_abs.size else 0.0, 1.0e-6)
    for module_idx in np.flatnonzero(present):
        cx, cy = centers[module_idx]
        color = "#fdae61" if heat[module_idx] >= 0 else "#74add1"
        radius = float(model.config.module_radius)
        ax.add_patch(plt.Circle((float(cx), float(cy)), radius, fill=True, color=color, alpha=0.20 + 0.35 * float(heat_scale[module_idx]), lw=0.0))
        ax.add_patch(plt.Circle((float(cx), float(cy)), radius, fill=False, color=color, lw=1.2 + 1.4 * float(heat_scale[module_idx])))
        ax.text(float(cx), float(cy), f"M{module_idx}", ha="center", va="center", color="white", fontsize=8, weight="bold")
    for hidx in range(strength.shape[0]):
        alpha = float(np.clip(strength[hidx], 0.12, 1.0))
        color = cmap(hidx)
        ax.plot([src[hidx, 0], dst[hidx, 0]], [src[hidx, 1], dst[hidx, 1]], color=color, lw=1.0 + 2.0 * alpha, alpha=alpha)
        ax.scatter(src[hidx, 0], src[hidx, 1], marker="x", s=35 + 70 * alpha, color=color, linewidth=1.5)
        ax.scatter(dst[hidx, 0], dst[hidx, 1], marker="*", s=65 + 125 * alpha, color=color, edgecolor="black", linewidth=0.45)
        ax.text(dst[hidx, 0], dst[hidx, 1], f"H{hidx}\n{strength[hidx]:.2f}", color="white", fontsize=7, ha="center", va="center")
        if A_mh.size:
            for module_idx in np.flatnonzero(present):
                weight = float(A_mh[module_idx, hidx])
                if weight >= 0.35:
                    ax.plot([centers[module_idx, 0], src[hidx, 0]], [centers[module_idx, 1], src[hidx, 1]], color=color, lw=0.5 + 1.8 * weight, alpha=0.18 + 0.55 * weight)
    ax.set_title("Physical organizer overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_channelthermal_organization_matrices(
    output_path: Path,
    sample: Dict[str, Any],
    aux: Dict[str, Any],
    model: GlobalChannelThermalModel,
) -> None:
    arrays = extract_channelthermal_organization_arrays(sample, aux, model)
    centers = arrays["centers"]
    A_mh = arrays["A_mh"]
    A_eh = arrays["A_eh"]
    strength = arrays["strength"]
    module_mass = arrays["module_mass"]
    env_mass = arrays["env_mass"]
    dominant_env = A_eh.argmax(axis=-1) if A_eh.size else np.zeros((0,), dtype=np.int64)
    sort_idx = np.lexsort((np.arange(A_eh.shape[0]), dominant_env)) if A_eh.size else np.arange(0)

    fig = plt.figure(figsize=(12.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.08, 1.45])
    ax_mh = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[0, 1])
    ax_eh = fig.add_subplot(gs[0, 2])
    labels = [f"H{i}\nS={strength[i]:.2f}\nM={module_mass[i]:.2f} E={env_mass[i]:.2f}" for i in range(strength.shape[0])]
    im1 = ax_mh.imshow(A_mh.T, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(A_mh)) if A_mh.size else 1.0, 1.0e-6))
    ax_mh.set_title("A_mh modules x hyperedges")
    ax_mh.set_xlabel("module")
    ax_mh.set_ylabel("hyperedge")
    ax_mh.set_xticks(np.arange(centers.shape[0]))
    ax_mh.set_xticklabels([f"M{i}" for i in range(centers.shape[0])], rotation=45, ha="right")
    ax_mh.set_yticks(np.arange(strength.shape[0]))
    ax_mh.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im1, ax=ax_mh, fraction=0.046, pad=0.04)

    strip = dominant_env[sort_idx][:, None] if A_eh.size else np.zeros((0, 1), dtype=np.int64)
    ax_strip.imshow(strip, aspect="auto", cmap=plt.get_cmap("tab10", max(strength.shape[0], 1)))
    ax_strip.set_title("dom", fontsize=8)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    im2 = ax_eh.imshow(A_eh[sort_idx].T if A_eh.size else A_eh.T, aspect="auto", cmap="viridis", vmin=0.0)
    ax_eh.set_title("A_eh env tokens sorted by dominant hyperedge")
    ax_eh.set_xlabel("sorted env token")
    ax_eh.set_ylabel("hyperedge")
    ax_eh.set_yticks(np.arange(strength.shape[0]))
    ax_eh.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im2, ax=ax_eh, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_channelthermal_hypergraph_schematic(
    output_path: Path,
    sample: Dict[str, Any],
    aux: Dict[str, Any],
    model: GlobalChannelThermalModel,
) -> None:
    arrays = extract_channelthermal_organization_arrays(sample, aux, model)
    centers = arrays["centers"]
    present = arrays["present"]
    A_mh = arrays["A_mh"]
    strength = arrays["strength"]
    module_mass = arrays["module_mass"]
    env_mass = arrays["env_mass"]
    src = arrays["src"]
    dst = arrays["dst"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    num_h = max(strength.shape[0], 1)
    cmap = plt.get_cmap("tab10", num_h)
    for hidx in range(strength.shape[0]):
        color = cmap(hidx) if strength[hidx] >= 0.05 else (0.55, 0.55, 0.55, 1.0)
        alpha = float(np.clip(strength[hidx], 0.12, 0.85))
        cx = 0.5 * (src[hidx, 0] + dst[hidx, 0])
        cy = 0.5 * (src[hidx, 1] + dst[hidx, 1])
        ax.add_patch(plt.Circle((float(cx), float(cy)), 0.34 + 0.8 * float(env_mass[hidx]), color=color, alpha=0.12 + 0.18 * alpha, lw=0.0))
        ax.text(float(cx), float(cy), f"H{hidx}\nM={module_mass[hidx]:.2f} E={env_mass[hidx]:.2f}\nS={strength[hidx]:.2f}", ha="center", va="center", fontsize=8, color="black")
    for module_idx in np.flatnonzero(present):
        cx, cy = centers[module_idx]
        ax.scatter(cx, cy, s=150, color="#fdae61", edgecolor="black", linewidth=0.7, zorder=3)
        ax.text(cx, cy, f"M{module_idx}", ha="center", va="center", fontsize=8, color="white", weight="bold", zorder=4)
        if A_mh.size:
            for hidx in range(strength.shape[0]):
                weight = float(A_mh[module_idx, hidx])
                if weight < 0.20:
                    continue
                color = cmap(hidx) if strength[hidx] >= 0.05 else (0.55, 0.55, 0.55, 1.0)
                ax.plot([cx, src[hidx, 0]], [cy, src[hidx, 1]], color=color, lw=0.4 + 2.2 * weight, alpha=0.20 + 0.55 * weight)
    ax.scatter(src[:, 0], src[:, 1], marker="x", s=55, color="black", linewidth=1.3, label="source")
    ax.scatter(dst[:, 0], dst[:, 1], marker="*", s=95, color="black", linewidth=0.7, label="thermal region")
    ax.set_title("Conceptual organization schematic")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_organizer(output_path: Path, sample: Dict[str, Any], aux: Dict[str, Any], model: GlobalChannelThermalModel) -> None:
    """Backward-compatible combined organizer diagnostic."""
    render_channelthermal_physical_organization(output_path, sample, aux, model)


def save_organizer_summary(output_dir: Path, aux: Dict[str, Any], sample: Dict[str, Any], model: GlobalChannelThermalModel) -> tuple[Path, Path]:
    arrays = extract_channelthermal_organization_arrays(sample, aux, model)
    rows = compute_channelthermal_hyperedge_summary(arrays)
    csv_path = output_dir / "organization_summary.csv"
    json_path = output_dir / "organization_summary.json"
    fieldnames = [
        "hyperedge_id",
        "module_mass",
        "env_mass",
        "hyper_strength",
        "top_modules",
        "top_module_weights",
        "env_token_count",
        "dominant_env_count",
        "source_x",
        "source_y",
        "thermal_region_x",
        "thermal_region_y",
        "active",
        "low_strength",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        json_path,
        {
            "hyperedges": rows,
            "visual_encoding": {
                "env_token_color": "dominant hyperedge argmax(A_eh)",
                "env_token_alpha": "max assignment confidence max(A_eh)",
                "module_link_width": "A_mh",
                "hyperedge_strength": "sqrt(module_mass * env_mass)",
            },
        },
    )
    return csv_path, json_path


def copy_figure_alias(source: Path, alias: Path) -> None:
    if source.resolve() == alias.resolve():
        return
    shutil.copyfile(source, alias)


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
    for key in ("pred_interface_surrogate_raw", "pred_interface_flux_physics"):
        if key in out:
            out[key] = dataset.normalizer.denormalize_interface_targets(out[key])
    return out


def summarize_prediction(
    checkpoint_path: Path,
    raw_sample: Dict[str, Any],
    predictions: Dict[str, Any],
    output_dir: Path,
    suffix: str,
    channel_order: list[str],
    temperature_display_mode: str,
) -> Dict[str, Any]:
    pred_field = predictions["pred_field_grid"]
    gt_field = raw_sample["steady_field"][..., : pred_field.shape[-1]]
    pred_internal = predictions["pred_internal_temperature"]
    gt_internal = raw_sample["module_internal_temperature_points"]
    pred_interface = predictions["pred_interface"]
    gt_interface = raw_sample["interface_target"]
    module_mask, fluid_mask = module_and_fluid_masks(raw_sample, pred_field)
    pred_temp_display = None
    gt_temp_display = None
    composite_temperature_metrics = None
    if pred_field.shape[-1] >= 5 and str(temperature_display_mode) == "composite_internal":
        pred_temp_display = composite_temperature_grid(raw_sample, pred_field[..., 4], pred_internal[..., 0])
        gt_temp_display = composite_temperature_grid(raw_sample, gt_field[..., 4], gt_internal)
        composite_temperature_metrics = error_metrics(pred_temp_display, gt_temp_display)
    save_payload = {
        "pred_field_grid": pred_field.astype(np.float32),
        "gt_field_grid": gt_field.astype(np.float32),
        "module_mask": module_mask.astype(np.uint8),
        "fluid_mask": fluid_mask.astype(np.uint8),
        "temperature_display_mode": np.asarray(str(temperature_display_mode)),
        "pred_internal_temperature": pred_internal.astype(np.float32),
        "gt_internal_temperature": gt_internal.astype(np.float32),
        "pred_interface": pred_interface.astype(np.float32),
        "gt_interface": gt_interface.astype(np.float32),
        "pred_port_condition": predictions["pred_port_condition"].astype(np.float32),
    }
    if pred_temp_display is not None and gt_temp_display is not None:
        save_payload["pred_temperature_display"] = pred_temp_display.astype(np.float32)
        save_payload["gt_temperature_display"] = gt_temp_display.astype(np.float32)
    if "pred_interface_surrogate_raw" in predictions:
        save_payload["pred_interface_surrogate_raw"] = predictions["pred_interface_surrogate_raw"].astype(np.float32)
    if "pred_interface_flux_physics" in predictions:
        save_payload["pred_interface_flux_physics"] = predictions["pred_interface_flux_physics"].astype(np.float32)
    if "pred_interface_delta_q" in predictions:
        save_payload["pred_interface_delta_q"] = predictions["pred_interface_delta_q"].astype(np.float32)
    npz_path = output_dir / f"evaluation_outputs_{suffix}.npz"
    np.savez_compressed(npz_path, **save_payload)
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
    field_channel_mse_fluid = {
        str(name): masked_error_metrics(pred_field[..., idx], gt_field[..., idx], fluid_mask)["mse"]
        for idx, name in enumerate(channel_order[: pred_field.shape[-1]])
    }
    field_channel_rmse_fluid = {
        str(name): masked_error_metrics(pred_field[..., idx], gt_field[..., idx], fluid_mask)["rmse"]
        for idx, name in enumerate(channel_order[: pred_field.shape[-1]])
    }
    field_channel_relative_l2_fluid = {
        str(name): masked_error_metrics(pred_field[..., idx], gt_field[..., idx], fluid_mask)["relative_l2"]
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
    target_port = raw_sample["teacher_port_tokens"][..., 3:5]
    pred_port = predictions["pred_port_condition"][..., 3:5]
    port_t_env_rmse = error_metrics(pred_port[..., 0], target_port[..., 0])["rmse"]
    port_h_rmse = error_metrics(pred_port[..., 1], target_port[..., 1])["rmse"]
    q_surrogate_raw_rmse = None
    if "pred_interface_surrogate_raw" in predictions:
        q_surrogate_raw_rmse = error_metrics(predictions["pred_interface_surrogate_raw"][..., 1], gt_interface[..., 1])["rmse"]
    q_physics_rmse = None
    if "pred_interface_flux_physics" in predictions:
        q_physics_rmse = error_metrics(predictions["pred_interface_flux_physics"][..., 1], gt_interface[..., 1])["rmse"]
    interface_relative_l2_by_target = {
        "T_surface": error_metrics(pred_interface[..., 0], gt_interface[..., 0])["relative_l2"],
        "q_normal": error_metrics(pred_interface[..., 1], gt_interface[..., 1])["relative_l2"],
    }
    field_metrics = error_metrics(pred_field, gt_field)
    temperature_metrics = error_metrics(pred_field[..., 4], gt_field[..., 4]) if pred_field.shape[-1] >= 5 else None
    field_metrics_fluid = masked_error_metrics(pred_field, gt_field, fluid_mask)
    temperature_metrics_fluid = masked_error_metrics(pred_field[..., 4], gt_field[..., 4], fluid_mask) if pred_field.shape[-1] >= 5 else None
    internal_metrics = error_metrics(pred_internal.reshape(-1), gt_internal.reshape(-1))
    interface_metrics = error_metrics(pred_interface, gt_interface)
    return {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "metric_note": "l2_error is the aggregate Euclidean norm over all values; relative_l2 is l2_error divided by target L2 norm.",
        "metric_mask_note": "Fluid-only global metrics exclude solid module interiors; module internal temperature is evaluated separately.",
        "field_l2_error": field_metrics["l2_norm"],
        "temperature_l2_error": temperature_metrics["l2_norm"] if temperature_metrics is not None else None,
        "internal_l2_error": internal_metrics["l2_norm"],
        "interface_l2_error": interface_metrics["l2_norm"],
        "field_rmse": field_metrics["rmse"],
        "temperature_rmse": temperature_metrics["rmse"] if temperature_metrics is not None else None,
        "field_mse_fluid": field_metrics_fluid["mse"],
        "field_rmse_fluid": field_metrics_fluid["rmse"],
        "field_relative_l2_fluid": field_metrics_fluid["relative_l2"],
        "temperature_mse_fluid": temperature_metrics_fluid["mse"] if temperature_metrics_fluid is not None else None,
        "temperature_rmse_fluid": temperature_metrics_fluid["rmse"] if temperature_metrics_fluid is not None else None,
        "temperature_relative_l2_fluid": temperature_metrics_fluid["relative_l2"] if temperature_metrics_fluid is not None else None,
        "u_mse_fluid": field_channel_mse_fluid.get("u"),
        "omega_mse_fluid": field_channel_mse_fluid.get("omega"),
        "temperature_display_mode": str(temperature_display_mode),
        "temperature_composite_mse": composite_temperature_metrics["mse"] if composite_temperature_metrics is not None else None,
        "temperature_composite_rmse": composite_temperature_metrics["rmse"] if composite_temperature_metrics is not None else None,
        "temperature_composite_relative_l2": composite_temperature_metrics["relative_l2"] if composite_temperature_metrics is not None else None,
        "internal_rmse": internal_metrics["rmse"],
        "interface_rmse": interface_metrics["rmse"],
        "interface_flux_mode": str(predictions.get("interface_flux_mode", "unknown")),
        "T_surface_rmse": interface_rmse_by_target["T_surface"],
        "q_normal_rmse": interface_rmse_by_target["q_normal"],
        "port_T_env_rmse": port_t_env_rmse,
        "port_h_rmse": port_h_rmse,
        "q_surrogate_raw_rmse": q_surrogate_raw_rmse,
        "q_physics_rmse": q_physics_rmse,
        "field_relative_l2": field_metrics["relative_l2"],
        "temperature_relative_l2": temperature_metrics["relative_l2"] if temperature_metrics is not None else None,
        "internal_relative_l2": internal_metrics["relative_l2"],
        "interface_relative_l2": interface_metrics["relative_l2"],
        "field_channel_l2_error": field_channel_l2,
        "field_channel_rmse": field_channel_rmse,
        "field_channel_relative_l2": field_channel_relative_l2,
        "field_channel_mse_fluid": field_channel_mse_fluid,
        "field_channel_rmse_fluid": field_channel_rmse_fluid,
        "field_channel_relative_l2_fluid": field_channel_relative_l2_fluid,
        "interface_l2_error_by_target": interface_l2_by_target,
        "interface_rmse_by_target": interface_rmse_by_target,
        "interface_relative_l2_by_target": interface_relative_l2_by_target,
        "field_metrics": field_metrics,
        "temperature_metrics": temperature_metrics,
        "field_metrics_fluid": field_metrics_fluid,
        "temperature_metrics_fluid": temperature_metrics_fluid,
        "composite_temperature_metrics": composite_temperature_metrics,
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
    temperature_display_mode_arg: Optional[str],
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
    temperature_display_mode = resolve_temperature_display_mode(temperature_display_mode_arg, predictions)
    plot_field_quicklook(
        output_dir / f"global_field_quicklook_{suffix}.png",
        raw_sample,
        predictions["pred_field_grid"],
        channel_order,
        pred_internal_temperature=predictions.get("pred_internal_temperature"),
        temperature_display_mode=temperature_display_mode,
    )
    plot_internal(output_dir / f"module_internal_temperature_{suffix}.png", raw_sample, predictions["pred_internal_temperature"])
    plot_interface(
        output_dir / f"interface_curves_{suffix}.png",
        raw_sample,
        predictions["pred_interface"],
        predictions.get("pred_interface_surrogate_raw"),
        predictions.get("pred_interface_flux_physics"),
        str(predictions.get("interface_flux_mode", "unknown")),
    )
    summary = summarize_prediction(checkpoint_path, raw_sample, predictions, output_dir, suffix, channel_order, temperature_display_mode)
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
            temperature_display_mode_arg=args.temperature_display_mode,
        )
        if first_aux is None:
            first_aux = mode_summary.pop("_organizer_aux", None)
        else:
            mode_summary.pop("_organizer_aux", None)
        mode_summaries[mode_suffix(mode)] = mode_summary

    if first_aux is not None:
        org_outputs: Dict[str, str] = {}
        view = str(args.organization_view)
        style = str(args.organization_style)
        link_threshold = float(args.organization_link_threshold)
        org_arrays = extract_channelthermal_organization_arrays(raw_sample, first_aux, model)
        org_module_radius = module_radius_from_sample(raw_sample, fallback=float(getattr(model.config, "module_radius", 0.45)))
        render_presentation = style in {"presentation", "both"}
        render_debug = style in {"debug", "both"}
        if view in {"all", "physical"}:
            if render_presentation:
                overview_path = output_dir / "organization_overview.png"
                render_channelthermal_organization_overview(
                    overview_path,
                    raw_sample,
                    org_arrays,
                    module_radius=org_module_radius,
                    channel_order=channel_order,
                    link_threshold=link_threshold,
                )
                org_outputs["organization_overview"] = str(overview_path)
                legacy_path = output_dir / "organizer_visualization.png"
                copy_figure_alias(overview_path, legacy_path)
                org_outputs["organizer_visualization"] = str(legacy_path)
                org_outputs["organization_physical"] = str(legacy_path)
            if render_debug:
                debug_physical_path = output_dir / "organization_physical_debug.png"
                render_channelthermal_physical_organization(debug_physical_path, raw_sample, first_aux, model)
                org_outputs["organization_physical_debug"] = str(debug_physical_path)
                if not render_presentation:
                    legacy_path = output_dir / "organizer_visualization.png"
                    copy_figure_alias(debug_physical_path, legacy_path)
                    org_outputs["organizer_visualization"] = str(legacy_path)
                    org_outputs["organization_physical"] = str(legacy_path)
        if view in {"all", "matrices"}:
            if render_presentation:
                summary_matrix_path = output_dir / "organization_summary_matrices.png"
                render_channelthermal_organization_summary_matrices(
                    summary_matrix_path,
                    raw_sample,
                    org_arrays,
                    module_radius=org_module_radius,
                    channel_order=channel_order,
                )
                org_outputs["organization_summary_matrices"] = str(summary_matrix_path)
                legacy_matrix_path = output_dir / "organization_matrices.png"
                copy_figure_alias(summary_matrix_path, legacy_matrix_path)
                org_outputs["organization_matrices"] = str(legacy_matrix_path)
            if render_debug:
                debug_matrix_path = output_dir / "organization_matrices_debug.png"
                render_channelthermal_organization_matrices(debug_matrix_path, raw_sample, first_aux, model)
                org_outputs["organization_matrices_debug"] = str(debug_matrix_path)
                if not render_presentation:
                    legacy_matrix_path = output_dir / "organization_matrices.png"
                    copy_figure_alias(debug_matrix_path, legacy_matrix_path)
                    org_outputs["organization_matrices"] = str(legacy_matrix_path)
        if view in {"all", "schematic"}:
            if render_presentation:
                schematic_path = output_dir / "organization_schematic.png"
                render_channelthermal_organization_schematic_presentation(
                    schematic_path,
                    raw_sample,
                    org_arrays,
                    link_threshold=link_threshold,
                )
                org_outputs["organization_schematic"] = str(schematic_path)
            if render_debug:
                debug_schematic_path = output_dir / "organization_schematic_debug.png"
                render_channelthermal_hypergraph_schematic(debug_schematic_path, raw_sample, first_aux, model)
                org_outputs["organization_schematic_debug"] = str(debug_schematic_path)
                if not render_presentation:
                    legacy_schematic_path = output_dir / "organization_schematic.png"
                    copy_figure_alias(debug_schematic_path, legacy_schematic_path)
                    org_outputs["organization_schematic"] = str(legacy_schematic_path)
        org_csv, org_json = save_organizer_summary(output_dir, first_aux, raw_sample, model)
    else:
        org_outputs = {}
        org_csv = output_dir / "organization_summary.csv"
        org_json = output_dir / "organization_summary.json"

    summary = {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "local_port_condition_mode": args.local_port_condition_mode,
        "mixed_teacher_ratio": float(args.mixed_teacher_ratio),
        "outputs": {
            "organization_summary_csv": str(org_csv),
            "organization_summary_json": str(org_json),
            **org_outputs,
        },
        "modes": mode_summaries,
    }
    if args.local_port_condition_mode == "both":
        teacher = mode_summaries["teacher"]
        predicted = mode_summaries["predicted"]
        extra_metric_keys = [
            "interface_flux_mode",
            "T_surface_rmse",
            "q_normal_rmse",
            "port_T_env_rmse",
            "port_h_rmse",
            "q_surrogate_raw_rmse",
            "q_physics_rmse",
        ]
        summary.update(
            {
                # "teacher_field_l2_error": teacher["field_l2_error"],
                # "predicted_field_l2_error": predicted["field_l2_error"],
                "teacher_field_relative_l2": teacher["field_relative_l2"],
                "predicted_field_relative_l2": predicted["field_relative_l2"],
                "teacher_field_mse_fluid": teacher["field_mse_fluid"],
                "predicted_field_mse_fluid": predicted["field_mse_fluid"],
                "teacher_temperature_mse_fluid": teacher["temperature_mse_fluid"],
                "predicted_temperature_mse_fluid": predicted["temperature_mse_fluid"],
                "teacher_u_mse_fluid": teacher["u_mse_fluid"],
                "predicted_u_mse_fluid": predicted["u_mse_fluid"],
                "teacher_omega_mse_fluid": teacher["omega_mse_fluid"],
                "predicted_omega_mse_fluid": predicted["omega_mse_fluid"],
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
        for key in extra_metric_keys:
            summary[f"teacher_{key}"] = teacher.get(key)
            summary[f"predicted_{key}"] = predicted.get(key)
    else:
        suffix = mode_suffix(args.local_port_condition_mode)
        only = mode_summaries[suffix]
        extra_metric_keys = [
            "interface_flux_mode",
            "T_surface_rmse",
            "q_normal_rmse",
            "port_T_env_rmse",
            "port_h_rmse",
            "q_surrogate_raw_rmse",
            "q_physics_rmse",
        ]
        summary.update(
            {
                "field_l2_error": only["field_l2_error"],
                "field_relative_l2": only["field_relative_l2"],
                "field_mse_fluid": only["field_mse_fluid"],
                "field_relative_l2_fluid": only["field_relative_l2_fluid"],
                "temperature_l2_error": only["temperature_l2_error"],
                "temperature_relative_l2": only["temperature_relative_l2"],
                "temperature_mse_fluid": only["temperature_mse_fluid"],
                "temperature_relative_l2_fluid": only["temperature_relative_l2_fluid"],
                "u_mse_fluid": only["u_mse_fluid"],
                "omega_mse_fluid": only["omega_mse_fluid"],
                "internal_l2_error": only["internal_l2_error"],
                "internal_relative_l2": only["internal_relative_l2"],
                "interface_l2_error": only["interface_l2_error"],
                "interface_relative_l2": only["interface_relative_l2"],
                f"{suffix}_field_l2_error": only["field_l2_error"],
                f"{suffix}_field_relative_l2": only["field_relative_l2"],
                f"{suffix}_field_mse_fluid": only["field_mse_fluid"],
                f"{suffix}_temperature_mse_fluid": only["temperature_mse_fluid"],
                f"{suffix}_u_mse_fluid": only["u_mse_fluid"],
                f"{suffix}_omega_mse_fluid": only["omega_mse_fluid"],
                f"{suffix}_temperature_l2_error": only["temperature_l2_error"],
                f"{suffix}_temperature_relative_l2": only["temperature_relative_l2"],
                f"{suffix}_internal_l2_error": only["internal_l2_error"],
                f"{suffix}_internal_relative_l2": only["internal_relative_l2"],
                f"{suffix}_interface_l2_error": only["interface_l2_error"],
                f"{suffix}_interface_relative_l2": only["interface_relative_l2"],
            }
        )
        for key in extra_metric_keys:
            summary[key] = only.get(key)
            summary[f"{suffix}_{key}"] = only.get(key)
    write_json(output_dir / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
