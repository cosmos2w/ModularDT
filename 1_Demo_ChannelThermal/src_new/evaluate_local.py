"""CHANNELTHERMAL-SPECIFIC Stage-A local surrogate evaluator.

Inputs are a copied or legacy local surrogate checkpoint plus a packed local
HDF5 dataset. Outputs are internal-temperature and interface quicklook plots
plus JSON metrics. This executable is ChannelThermal-specific and uses the
copied local architecture with strict state-dict loading.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _bootstrap_imports
from _data.local_module_datasets import LocalModuleDataset
from _helpers.model_utils import (
    current_timestamp,
    load_trusted_checkpoint,
    recursive_to_device,
    resolve_demo_path,
    select_device,
    strip_module_prefix,
    write_json,
)
from _models_local.model_local import LocalModuleConfig, LocalModuleSurrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local module surrogate checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint selector: best, latest/lastest, or a direct .pt path.",
    )
    parser.add_argument("--Run_ID", dest="run_id", type=str, default=None, help="Numeric run serial used to find the latest matching saved model, e.g. 0001.")
    parser.add_argument("--saved-root", type=str, default="./Saved_Model_LocalModule", help="Root directory containing local saved-model runs.")
    parser.add_argument("--dataset", type=str, default=None, help="Override packed local HDF5 path.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument("--case-index", type=int, default=0, help="Index within the selected split.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for quicklook outputs.")
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
        raise FileNotFoundError(f"No saved local runs found under {saved_root} with Run_ID={normalized!r}.")
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


def tensorize_sample(sample: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            out[key] = torch.from_numpy(value).unsqueeze(0)
        else:
            out[key] = value
    return recursive_to_device(out, device)


def raster_from_points(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.full(mask.shape, np.nan, dtype=np.float32)
    image[mask.astype(bool)] = values.reshape(-1)
    return image


def l2_error(prediction: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(diff.reshape(-1), ord=2))


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Return aggregate and scale-normalized error metrics."""
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
    base_dir = Path(base_dir_arg) if base_dir_arg else checkpoint_path.parent / "eval_local"
    return resolve_demo_path(base_dir) / f"{safe_path_name(case_id)}_{current_timestamp()}"


def plot_internal(output_path: Path, sample: Dict, pred_internal: np.ndarray, metrics: Dict[str, float]) -> None:
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
    for col, (ax, image, title, cmap) in enumerate([
        (axes[0], gt_img, "GT internal T", "inferno"),
        (axes[1], pred_img, "Pred internal T", "inferno"),
        (axes[2], err_img, f"Abs error\nRMSE={metrics['rmse']:.4e}", "magma"),
    ]):
        im = ax.imshow(image, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, vmin=vmin if col < 2 else None, vmax=vmax if col < 2 else None)
        ax.set_title(title)
        ax.set_xlabel("xi")
        ax.set_ylabel("eta")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(str(output_path), dpi=170)
    plt.close(fig)


def plot_interface(output_path: Path, sample: Dict, pred_interface: np.ndarray) -> None:
    theta = sample["port_tokens"][:, 0]
    target = sample["interface_targets"]
    rough = sample.get("local_target_roughness", np.zeros((4,), dtype=np.float32))
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(theta, sample["port_tokens"][:, 3], color="#4c78a8", lw=1.5, label="T_env")
    axes[0].set_ylabel("T_env")
    ax_h = axes[0].twinx()
    ax_h.plot(theta, sample["port_tokens"][:, 4], color="#f58518", lw=1.3, label="h")
    ax_h.set_ylabel("h")
    axes[0].set_title(
        f"Boundary inputs | solver={sample.get('solver_type', 'unknown')} "
        f"modes={int(np.asarray(sample.get('n_active_modes', [-1])).reshape(-1)[0])}"
    )
    axes[0].grid(True, alpha=0.25)
    labels = ["T_surface", "q_normal"]
    for idx, ax in enumerate(axes[1:]):
        channel_metrics = error_metrics(pred_interface[:, idx], target[:, idx])
        ax.plot(theta, target[:, idx], color="black", lw=1.8, label="GT")
        ax.plot(theta, pred_interface[:, idx], color="#d95f02", lw=1.5, label="Pred")
        if "interface_targets_raw" in sample:
            ax.plot(theta, sample["interface_targets_raw"][:, idx], color="#7f7f7f", lw=1.0, alpha=0.65, label="raw target")
        ax.set_ylabel(labels[idx])
        ax.set_title(
            f"{labels[idx]} RMSE={channel_metrics['rmse']:.4e}, relL2={channel_metrics['relative_l2']:.4e}, "
            f"rough={float(rough[idx]):.3f}"
        )
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("theta")
    axes[1].legend()
    fig.savefig(str(output_path), dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_arg(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    train_cfg = checkpoint.get("train_config", {})
    dataset_cfg = train_cfg.get("dataset", {})
    dataset_path = args.dataset or dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5")
    normalize_inputs = bool(dataset_cfg.get("normalize_inputs", False))
    normalize_targets = bool(dataset_cfg.get("normalize_targets", False))
    dataset = LocalModuleDataset(dataset_path, split=args.split, normalize_inputs=normalize_inputs, normalize_targets=normalize_targets)
    raw_dataset = LocalModuleDataset(dataset_path, split=args.split, normalize_inputs=False, normalize_targets=False)
    if len(dataset) == 0:
        dataset = LocalModuleDataset(dataset_path, split="all", normalize_inputs=normalize_inputs, normalize_targets=normalize_targets)
        raw_dataset = LocalModuleDataset(dataset_path, split="all", normalize_inputs=False, normalize_targets=False)
    if len(dataset) == 0:
        raise RuntimeError("No local module cases are available for evaluation.")

    device = select_device(args.device)
    model_config = LocalModuleConfig.from_dict(checkpoint.get("model_config", {}))
    model = LocalModuleSurrogate(model_config).to(device)
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    model.eval()

    index = min(max(int(args.case_index), 0), len(dataset) - 1)
    sample = dataset[index]
    raw_sample = raw_dataset[index]
    batch = tensorize_sample(sample, device)
    with torch.no_grad():
        outputs = model(batch["module_params"], batch["port_tokens"], batch["internal_query_points"])

    pred_internal = outputs["internal_temperature"][0].detach().cpu().numpy()
    pred_interface = outputs["interface_pred"][0].detach().cpu().numpy()
    target_internal = sample["internal_temperature_targets"]
    target_interface = sample["interface_targets"]
    if normalize_targets:
        pred_internal = dataset.normalizer.denormalize_internal_temperature(pred_internal)
        pred_interface = dataset.normalizer.denormalize_interface_targets(pred_interface)
        target_internal = raw_sample["internal_temperature_targets"]
        target_interface = raw_sample["interface_targets"]
    internal_metrics = error_metrics(pred_internal.reshape(-1), target_internal.reshape(-1))
    interface_metrics = error_metrics(pred_interface, target_interface)
    t_surface_metrics = error_metrics(pred_interface[:, 0], target_interface[:, 0])
    q_normal_metrics = error_metrics(pred_interface[:, 1], target_interface[:, 1])
    roughness = raw_sample.get("local_target_roughness", np.zeros((4,), dtype=np.float32))
    n_active_modes = int(np.asarray(raw_sample.get("n_active_modes", [-1])).reshape(-1)[0])

    output_dir = evaluation_output_dir(args.output_dir, checkpoint_path, raw_sample["case_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_internal(
        output_dir / "internal_temperature_comparison.png",
        {**raw_sample, "internal_temperature_targets": target_internal},
        pred_internal,
        internal_metrics,
    )
    plot_interface(
        output_dir / "interface_curve_comparison.png",
        {**raw_sample, "interface_targets": target_interface},
        pred_interface,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "metric_note": "l2_error is the aggregate Euclidean norm over all values; rmse is usually better for visual comparison.",
        "internal_l2_error": internal_metrics["l2_norm"],
        "interface_l2_error": interface_metrics["l2_norm"],
        "T_surface_rmse": t_surface_metrics["rmse"],
        "T_surface_relative_l2": t_surface_metrics["relative_l2"],
        "q_normal_rmse": q_normal_metrics["rmse"],
        "q_normal_relative_l2": q_normal_metrics["relative_l2"],
        "internal_rmse": internal_metrics["rmse"],
        "interface_rmse": interface_metrics["rmse"],
        "internal_mae": internal_metrics["mae"],
        "interface_mae": interface_metrics["mae"],
        "internal_relative_l2": internal_metrics["relative_l2"],
        "interface_relative_l2": interface_metrics["relative_l2"],
        "internal_num_values": int(internal_metrics["num_values"]),
        "interface_num_values": int(interface_metrics["num_values"]),
        "internal_metrics": internal_metrics,
        "interface_metrics": interface_metrics,
        "T_surface_metrics": t_surface_metrics,
        "q_normal_metrics": q_normal_metrics,
        "solver_type": str(raw_sample.get("solver_type", "unknown")),
        "n_active_modes": n_active_modes,
        "roughness_metrics": {
            "roughness_T_surface": float(roughness[0]) if roughness.size > 0 else None,
            "roughness_q_normal": float(roughness[1]) if roughness.size > 1 else None,
            "highfreq_ratio_T_surface": float(roughness[2]) if roughness.size > 2 else None,
            "highfreq_ratio_q_normal": float(roughness[3]) if roughness.size > 3 else None,
        },
        "interface_targets_smoothed": bool(getattr(raw_dataset, "interface_targets_smoothed", False)),
        "outputs": {
            "internal_temperature_comparison": str(output_dir / "internal_temperature_comparison.png"),
            "interface_curve_comparison": str(output_dir / "interface_curve_comparison.png"),
        },
    }
    write_json(output_dir / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
