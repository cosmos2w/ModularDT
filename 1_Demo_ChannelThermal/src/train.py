from __future__ import annotations

"""Train the Stage B global Channel Thermal neural-field model."""

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from channelthermal_datasets import GlobalChannelThermalDataset
from channelthermal_model_utils import (
    autocast_context,
    count_parameters,
    current_timestamp,
    ensure_dir,
    make_grad_scaler,
    masked_mse,
    read_json,
    recursive_to_device,
    resolve_demo_path,
    save_loss_curve,
    select_device,
    set_seed,
    write_json,
)
from model import GlobalChannelThermalModel, GlobalChannelThermalModelConfig, load_local_surrogate_from_checkpoint


DEFAULT_CONFIG_PATH = "./Configs/train_global_config_template.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Demo 1 global Channel Thermal model.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="JSON config file path.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Stop each epoch after this many train batches.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Stop validation after this many batches.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional output run directory name.")
    return parser.parse_args()


def resolve_config_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return resolve_demo_path(path)


def _auto_int(value: Any, fallback: int) -> int:
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return int(fallback)
    return int(value)


def _auto_float(value: Any, fallback: float) -> float:
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return float(fallback)
    return float(value)


def _first_domain_and_radius(dataset: GlobalChannelThermalDataset) -> tuple[float, float, float]:
    if len(dataset) == 0:
        return 12.0, 4.0, 0.45
    sample = dataset[0]
    structure = sample["structure"]
    lx = float(structure["domain_length_x"][0])
    ly = float(structure["domain_length_y"][0])
    material = structure["material_params"]
    radius = float(material[5]) if material.shape[0] > 5 and float(material[5]) > 0.0 else 0.45
    return lx, ly, radius


def build_model_config(payload: Dict[str, Any], dataset: GlobalChannelThermalDataset) -> GlobalChannelThermalModelConfig:
    model_cfg = dict(payload.get("model", {}))
    lx, ly, radius = _first_domain_and_radius(dataset)
    model_cfg["field_dim"] = _auto_int(model_cfg.get("field_dim"), dataset.field_dim)
    model_cfg["max_num_modules"] = _auto_int(model_cfg.get("max_num_modules"), dataset.max_num_modules)
    model_cfg["domain_length_x"] = _auto_float(model_cfg.get("domain_length_x"), lx)
    model_cfg["domain_length_y"] = _auto_float(model_cfg.get("domain_length_y"), ly)
    model_cfg["module_radius"] = _auto_float(model_cfg.get("module_radius"), radius)
    model_cfg["default_num_interface_points"] = _auto_int(
        model_cfg.get("default_num_interface_points"), dataset.n_interface_points or 64
    )
    model_cfg["material_param_dim"] = _auto_int(model_cfg.get("material_param_dim"), dataset.material_param_dim)
    return GlobalChannelThermalModelConfig.from_dict(model_cfg)


def field_loss(pred: torch.Tensor, target: torch.Tensor, loss_cfg: Dict[str, Any]) -> torch.Tensor:
    weights = torch.ones(pred.shape[-1], device=pred.device, dtype=pred.dtype)
    temperature_weight = float(loss_cfg.get("temperature_weight", 1.0))
    if pred.shape[-1] >= 5:
        weights[4] = temperature_weight
    channel_weights = loss_cfg.get("field_channel_weights")
    if channel_weights is not None:
        custom = torch.as_tensor(channel_weights, device=pred.device, dtype=pred.dtype)
        weights[: custom.numel()] = custom[: pred.shape[-1]]
    return ((pred - target).square() * weights).mean()


def compute_losses(outputs: Dict[str, Any], batch: Dict[str, Any], loss_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    structure = batch["structure"]
    module_present = structure["module_present"]
    losses: Dict[str, torch.Tensor] = {}
    losses["loss_field"] = field_loss(outputs["pred_field"], batch["field_targets"], loss_cfg)

    if outputs.get("pred_internal_temperature") is not None and outputs["pred_internal_temperature"].numel() > 0:
        target_internal = batch["module_internal_temperature_points"].unsqueeze(-1)
        losses["loss_internal_temperature"] = masked_mse(
            outputs["pred_internal_temperature"],
            target_internal,
            module_present[:, :, None],
        )
    else:
        losses["loss_internal_temperature"] = outputs["pred_field"].new_tensor(0.0)

    if outputs.get("pred_interface") is not None and outputs["pred_interface"].numel() > 0:
        losses["loss_interface"] = masked_mse(outputs["pred_interface"], batch["interface_target"], module_present[:, :, None])
    else:
        losses["loss_interface"] = outputs["pred_field"].new_tensor(0.0)

    pred_port = outputs["pred_port_condition"]
    target_port = batch["teacher_port_tokens"]
    losses["loss_port_condition"] = masked_mse(pred_port[..., 3:5], target_port[..., 3:5], module_present[:, :, None])

    aux = outputs.get("organizer_aux", {})
    if isinstance(aux, dict) and "hyper_strength" in aux:
        losses["loss_organizer_strength"] = aux["hyper_strength"].mean()
    else:
        losses["loss_organizer_strength"] = pred_port.new_tensor(0.0)

    total = (
        float(loss_cfg.get("field_mse_weight", 1.0)) * losses["loss_field"]
        + float(loss_cfg.get("internal_temperature_weight", 1.0)) * losses["loss_internal_temperature"]
        + float(loss_cfg.get("interface_weight", 0.2)) * losses["loss_interface"]
        + float(loss_cfg.get("port_condition_weight", 0.1)) * losses["loss_port_condition"]
        + float(loss_cfg.get("organizer_strength_weight", 0.0)) * losses["loss_organizer_strength"]
    )
    losses["loss_total"] = total
    return losses


def run_epoch(
    model: GlobalChannelThermalModel,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler=None,
    amp: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if model.local_surrogate is not None and model.config.freeze_local_surrogate:
        model.local_surrogate.eval()
    sums: Dict[str, float] = {}
    count = 0
    iterator = tqdm(loader, desc="train" if training else "val", unit="batch", dynamic_ncols=True, leave=False)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = recursive_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context(device, amp):
            outputs = model(
                batch["structure"],
                batch["query_xy"],
                interface_condition=batch["interface_condition"],
                local_module_params=batch["local_module_params"],
                teacher_port_tokens=batch["teacher_port_tokens"],
                local_query_points=batch["module_internal_query_points"],
                local_port_condition_mode=training_cfg.get("local_port_condition_mode", "teacher"),
                mixed_teacher_ratio=float(training_cfg.get("mixed_teacher_ratio", 0.5)),
            )
            losses = compute_losses(outputs, batch, loss_cfg)
        if training:
            clip_norm = float(training_cfg.get("gradient_clip_norm", 1.0))
            if scaler is not None and scaler.is_enabled():
                scaler.scale(losses["loss_total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
        batch_size = int(batch["query_xy"].shape[0])
        count += batch_size
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        iterator.set_postfix(loss=f"{float(losses['loss_total'].detach().cpu()):.3e}")
    if count == 0:
        return {
            key: float("nan")
            for key in [
                "loss_total",
                "loss_field",
                "loss_internal_temperature",
                "loss_interface",
                "loss_port_condition",
            ]
        }
    return {key: value / count for key, value in sums.items()}


def save_checkpoint(
    path: Path,
    *,
    model: GlobalChannelThermalModel,
    model_config: GlobalChannelThermalModelConfig,
    train_config: Dict[str, Any],
    epoch: int,
    best_metric: float,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "model_config": model_config.to_dict(),
            "model_state_dict": model.state_dict(),
            "train_config": train_config,
        },
        path,
    )


def attach_local_surrogate_if_needed(
    model: GlobalChannelThermalModel,
    model_config: GlobalChannelThermalModelConfig,
    cfg: Dict[str, Any],
    device: torch.device,
) -> None:
    if not model_config.use_local_surrogate:
        return
    checkpoint_path = cfg.get("model", {}).get("local_surrogate_checkpoint_path")
    if not checkpoint_path:
        raise ValueError(
            "model.use_local_surrogate=true requires model.local_surrogate_checkpoint_path. "
            "Set use_local_surrogate=false for the global-only baseline."
        )
    local_model, _ = load_local_surrogate_from_checkpoint(resolve_demo_path(checkpoint_path), map_location=device)
    local_model.to(device)
    model.set_local_surrogate(local_model, freeze=bool(model_config.freeze_local_surrogate))


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = read_json(config_path)
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    set_seed(int(training_cfg.get("seed", 42)))
    device = select_device(args.device or training_cfg.get("device"))

    train_dataset = GlobalChannelThermalDataset(
        dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"),
        split=dataset_cfg.get("train_split", "train"),
        points_per_case=dataset_cfg.get("points_per_case", 4096),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=bool(dataset_cfg.get("random_point_sampling", True)),
        seed=int(training_cfg.get("seed", 42)),
    )
    val_dataset = GlobalChannelThermalDataset(
        dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"),
        split=dataset_cfg.get("val_split", "test"),
        points_per_case=dataset_cfg.get("val_points_per_case", dataset_cfg.get("points_per_case", 4096)),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        seed=int(training_cfg.get("seed", 42)) + 1000,
    )
    if len(val_dataset) == 0:
        val_dataset = train_dataset

    model_config = build_model_config(cfg, train_dataset)
    model = GlobalChannelThermalModel(model_config).to(device)
    attach_local_surrogate_if_needed(model, model_config, cfg, device)
    print(f"[setup] device={device}, train_cases={len(train_dataset)}, val_cases={len(val_dataset)}")
    print(f"[setup] model parameters={count_parameters(model):,}")

    batch_size = int(dataset_cfg.get("batch_size", training_cfg.get("batch_size", 4)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(dataset_cfg.get("val_batch_size", batch_size)),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(training_cfg.get("learning_rate", 2.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-5)),
    )
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 200))
    scaler = make_grad_scaler(device, bool(training_cfg.get("amp", False)))
    saved_root = ensure_dir(resolve_demo_path(cfg.get("paths", {}).get("saved_model_dir", "./Saved_Model")))
    run_name = args.run_name or training_cfg.get("run_name") or f"ChannelThermal_{current_timestamp()}"
    run_dir = ensure_dir(saved_root / run_name)
    write_json(run_dir / "resolved_train_config.json", cfg)
    history_path = run_dir / "loss_history.csv"
    fieldnames = [
        "epoch",
        "loss_total",
        "loss_field",
        "loss_internal_temperature",
        "loss_interface",
        "loss_port_condition",
        "val_loss_total",
        "val_loss_field",
        "val_loss_internal_temperature",
        "val_loss_interface",
        "val_loss_port_condition",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    max_train_batches = args.max_train_batches
    if max_train_batches is None and training_cfg.get("max_train_batches_per_epoch") is not None:
        max_train_batches = int(training_cfg["max_train_batches_per_epoch"])
    max_val_batches = args.max_val_batches
    if max_val_batches is None and training_cfg.get("max_val_batches") is not None:
        max_val_batches = int(training_cfg["max_val_batches"])

    best_metric = math.inf
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            loss_cfg,
            training_cfg,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(training_cfg.get("amp", False)),
            max_batches=max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            loss_cfg,
            training_cfg,
            optimizer=None,
            scaler=None,
            amp=bool(training_cfg.get("amp", False)),
            max_batches=max_val_batches,
        )
        row = {
            "epoch": epoch,
            "loss_total": train_metrics.get("loss_total", math.nan),
            "loss_field": train_metrics.get("loss_field", math.nan),
            "loss_internal_temperature": train_metrics.get("loss_internal_temperature", math.nan),
            "loss_interface": train_metrics.get("loss_interface", math.nan),
            "loss_port_condition": train_metrics.get("loss_port_condition", math.nan),
            "val_loss_total": val_metrics.get("loss_total", math.nan),
            "val_loss_field": val_metrics.get("loss_field", math.nan),
            "val_loss_internal_temperature": val_metrics.get("loss_internal_temperature", math.nan),
            "val_loss_interface": val_metrics.get("loss_interface", math.nan),
            "val_loss_port_condition": val_metrics.get("loss_port_condition", math.nan),
        }
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        metric = float(row["val_loss_total"])
        if math.isfinite(metric) and metric < best_metric:
            best_metric = metric
            save_checkpoint(run_dir / "best_model.pt", model=model, model_config=model_config, train_config=cfg, epoch=epoch, best_metric=best_metric)
        save_checkpoint(run_dir / "latest_model.pt", model=model, model_config=model_config, train_config=cfg, epoch=epoch, best_metric=best_metric)
        save_loss_curve(history_path, run_dir / "loss_curve.png", title="Global Channel Thermal Loss")
        print(
            f"[epoch {epoch:04d}] loss={row['loss_total']:.4e} field={row['loss_field']:.4e} "
            f"internal={row['loss_internal_temperature']:.4e} interface={row['loss_interface']:.4e} "
            f"port={row['loss_port_condition']:.4e} val={row['val_loss_total']:.4e}"
        )

    print(f"[done] saved global model run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

