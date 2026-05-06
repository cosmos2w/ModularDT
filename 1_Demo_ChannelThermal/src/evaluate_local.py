from __future__ import annotations

"""Evaluate a Stage A local module surrogate checkpoint."""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from channelthermal_datasets import LocalModuleDataset
from channelthermal_model_utils import recursive_to_device, resolve_demo_path, select_device, strip_module_prefix, write_json
from model_local import LocalModuleConfig, LocalModuleSurrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local module surrogate checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt or latest_model.pt.")
    parser.add_argument("--dataset", type=str, default=None, help="Override packed local HDF5 path.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument("--case-index", type=int, default=0, help="Index within the selected split.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for quicklook outputs.")
    return parser.parse_args()


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


def plot_internal(output_path: Path, sample: Dict, pred_internal: np.ndarray) -> None:
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
    for ax, image, title, cmap in [
        (axes[0], gt_img, "GT internal T", "inferno"),
        (axes[1], pred_img, "Pred internal T", "inferno"),
        (axes[2], err_img, "Abs error", "magma"),
    ]:
        im = ax.imshow(image, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, vmin=vmin if title != "Abs error" else None, vmax=vmax if title != "Abs error" else None)
        ax.set_title(title)
        ax.set_xlabel("xi")
        ax.set_ylabel("eta")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_interface(output_path: Path, sample: Dict, pred_interface: np.ndarray) -> None:
    theta = sample["port_tokens"][:, 0]
    target = sample["interface_targets"]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True, constrained_layout=True)
    labels = ["T_surface", "q_normal"]
    for idx, ax in enumerate(axes):
        ax.plot(theta, target[:, idx], color="black", lw=1.8, label="GT")
        ax.plot(theta, pred_interface[:, idx], color="#d95f02", lw=1.5, label="Pred")
        ax.set_ylabel(labels[idx])
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("theta")
    axes[0].legend()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_demo_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "eval_local"
    output_dir = resolve_demo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_internal(
        output_dir / "internal_temperature_comparison.png",
        {**raw_sample, "internal_temperature_targets": target_internal},
        pred_internal,
    )
    plot_interface(
        output_dir / "interface_curve_comparison.png",
        {**raw_sample, "interface_targets": target_interface},
        pred_interface,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "internal_mse": float(np.mean((pred_internal.reshape(-1) - target_internal.reshape(-1)) ** 2)),
        "interface_mse": float(np.mean((pred_interface - target_interface) ** 2)),
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

