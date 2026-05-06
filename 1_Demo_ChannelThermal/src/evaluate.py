from __future__ import annotations

"""Evaluate a Stage B global Channel Thermal checkpoint on one processed case."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from channelthermal_datasets import CHANNEL_ORDER, GlobalChannelThermalDataset
from channelthermal_model_utils import recursive_to_device, resolve_demo_path, select_device, strip_module_prefix, write_json
from model import GlobalChannelThermalModel, GlobalChannelThermalModelConfig, load_local_surrogate_from_checkpoint

LOCAL_SURROGATE_NORMALIZATION_ERROR = (
    "Local surrogate normalization is not yet supported inside Stage-B. "
    "Re-train local surrogate with normalize_inputs=false and normalize_targets=false, "
    "or implement local normalizer application."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Demo 1 global Channel Thermal model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt or latest_model.pt.")
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
    dataset_cfg = local_checkpoint.get("train_config", {}).get("dataset", {})
    if bool(dataset_cfg.get("normalize_inputs", False)) or bool(dataset_cfg.get("normalize_targets", False)):
        raise ValueError(LOCAL_SURROGATE_NORMALIZATION_ERROR)
    local_model.to(device)
    model.set_local_surrogate(local_model, freeze=bool(model.config.freeze_local_surrogate))


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GlobalChannelThermalModel, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = GlobalChannelThermalModelConfig.from_dict(checkpoint.get("model_config", {}))
    model = GlobalChannelThermalModel(model_config).to(device)
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


def plot_field_quicklook(output_path: Path, sample: Dict[str, Any], pred_field: np.ndarray, channel_order: list[str]) -> None:
    gt = sample["steady_field"][..., : pred_field.shape[-1]]
    preferred = [name for name in ["temperature", "u", "omega"] if name in channel_order]
    if not preferred:
        preferred = channel_order[: min(3, len(channel_order))]
    fig, axes = plt.subplots(len(preferred), 3, figsize=(10.5, 3.0 * len(preferred)), constrained_layout=True)
    if len(preferred) == 1:
        axes = axes[None, :]
    for row, name in enumerate(preferred):
        idx = channel_order.index(name)
        gt_img = gt[..., idx]
        pred_img = pred_field[..., idx]
        err_img = np.abs(pred_img - gt_img)
        vmin = float(np.nanmin(gt_img))
        vmax = float(np.nanmax(gt_img))
        for col, (image, title, cmap) in enumerate(
            [(gt_img, f"GT {name}", channel_cmap(name)), (pred_img, f"Pred {name}", channel_cmap(name)), (err_img, f"Abs error {name}", "magma")]
        ):
            im = axes[row, col].imshow(
                image,
                origin="lower",
                extent=(float(np.min(sample["x_grid"])), float(np.max(sample["x_grid"])), float(np.min(sample["y_grid"])), float(np.max(sample["y_grid"]))),
                cmap=cmap,
                vmin=vmin if col < 2 else None,
                vmax=vmax if col < 2 else None,
                aspect="auto",
            )
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
        vmin = float(np.nanmin(gt_img))
        vmax = float(np.nanmax(gt_img))
        for col, (image, title, cmap) in enumerate(
            [(gt_img, f"M{module_idx} GT", "inferno"), (pred_img, f"M{module_idx} Pred", "inferno"), (err_img, f"M{module_idx} Error", "magma")]
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
            ax.plot(theta, gt[module_idx, :, col], color="black", lw=1.7, label="GT")
            ax.plot(theta, pred_interface[module_idx, :, col], color="#d95f02", lw=1.4, label="Pred")
            ax.set_title(f"M{module_idx} {label}")
            ax.set_xlabel("theta")
            ax.grid(True, alpha=0.25)
            if row == 0 and col == 0:
                ax.legend()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_organizer(output_path: Path, sample: Dict[str, Any], aux: Dict[str, Any], model: GlobalChannelThermalModel) -> None:
    centers = sample["structure"]["module_centers"]
    present = sample["structure"]["module_present"] > 0.5
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)

    axes[0].imshow(
        sample["steady_field"][..., CHANNEL_ORDER.index("temperature")],
        origin="lower",
        extent=(float(np.min(sample["x_grid"])), float(np.max(sample["x_grid"])), float(np.min(sample["y_grid"])), float(np.max(sample["y_grid"]))),
        cmap="inferno",
        alpha=0.78,
        aspect="auto",
    )
    for idx, (cx, cy) in enumerate(centers[present]):
        axes[0].add_patch(plt.Circle((float(cx), float(cy)), float(model.config.module_radius), fill=False, color="white", lw=1.4))
        axes[0].text(float(cx), float(cy), f"M{idx}", ha="center", va="center", color="white", fontsize=8)
    if "hyper_source_coords" in aux and "hyper_thermal_region_coords" in aux:
        src = aux["hyper_source_coords"]
        dst = aux["hyper_thermal_region_coords"]
        strength = aux.get("hyper_strength", np.ones((src.shape[0],), dtype=np.float32))
        for hidx in range(src.shape[0]):
            alpha = float(np.clip(strength[hidx], 0.15, 1.0))
            axes[0].plot([src[hidx, 0], dst[hidx, 0]], [src[hidx, 1], dst[hidx, 1]], color="#66c2a5", lw=1.0 + 2.0 * alpha, alpha=alpha)
            axes[0].scatter(dst[hidx, 0], dst[hidx, 1], s=30 + 70 * alpha, color="#66c2a5", edgecolor="black", linewidth=0.4)
    axes[0].set_title("Physical organizer overlay")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im1 = axes[1].imshow(aux.get("A_mh", np.zeros((1, 1))).T, aspect="auto", cmap="viridis")
    axes[1].set_title("A_mh module-to-hyper")
    axes[1].set_xlabel("module")
    axes[1].set_ylabel("hyperedge")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(aux.get("A_eh", np.zeros((1, 1))).T, aspect="auto", cmap="viridis")
    axes[2].set_title("A_eh env-to-hyper")
    axes[2].set_xlabel("env token")
    axes[2].set_ylabel("hyperedge")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


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
    return {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "field_mse": float(np.mean((pred_field - gt_field) ** 2)),
        "temperature_mse": float(np.mean((pred_field[..., 4] - gt_field[..., 4]) ** 2)) if pred_field.shape[-1] >= 5 else None,
        "internal_mse": float(np.mean((pred_internal.reshape(-1) - gt_internal.reshape(-1, 1)) ** 2)),
        "interface_mse": float(np.mean((pred_interface - gt_interface) ** 2)),
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
    checkpoint_path = resolve_demo_path(args.checkpoint)
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

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "eval_global"
    output_dir = resolve_demo_path(output_dir)
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

    summary = {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "local_port_condition_mode": args.local_port_condition_mode,
        "mixed_teacher_ratio": float(args.mixed_teacher_ratio),
        "outputs": {"organizer_visualization": str(output_dir / "organizer_visualization.png")},
        "modes": mode_summaries,
    }
    if args.local_port_condition_mode == "both":
        teacher = mode_summaries["teacher"]
        predicted = mode_summaries["predicted"]
        summary.update(
            {
                "teacher_field_mse": teacher["field_mse"],
                "predicted_field_mse": predicted["field_mse"],
                "teacher_temperature_mse": teacher["temperature_mse"],
                "predicted_temperature_mse": predicted["temperature_mse"],
                "teacher_internal_mse": teacher["internal_mse"],
                "predicted_internal_mse": predicted["internal_mse"],
                "teacher_interface_mse": teacher["interface_mse"],
                "predicted_interface_mse": predicted["interface_mse"],
            }
        )
    else:
        suffix = mode_suffix(args.local_port_condition_mode)
        only = mode_summaries[suffix]
        summary.update(
            {
                "field_mse": only["field_mse"],
                "temperature_mse": only["temperature_mse"],
                "internal_mse": only["internal_mse"],
                "interface_mse": only["interface_mse"],
                f"{suffix}_field_mse": only["field_mse"],
                f"{suffix}_temperature_mse": only["temperature_mse"],
                f"{suffix}_internal_mse": only["internal_mse"],
                f"{suffix}_interface_mse": only["interface_mse"],
            }
        )
    write_json(output_dir / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
