from __future__ import annotations

"""Train the Stage A local module thermal surrogate."""

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from channelthermal_datasets import LocalModuleDataset
from channelthermal_model_utils import (
    autocast_context,
    count_parameters,
    current_timestamp,
    ensure_dir,
    make_grad_scaler,
    read_json,
    recursive_to_device,
    resolve_demo_path,
    save_loss_curve,
    select_device,
    set_seed,
    write_json,
)
from model_local import LocalModuleConfig, LocalModuleSurrogate


DEFAULT_CONFIG_PATH = "./Configs/train_local_config_template.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Demo 1 local module surrogate.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="JSON config file path.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cpu or cuda:0.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Stop each epoch after this many train batches.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Stop validation after this many batches.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional descriptive suffix after the numeric Run_ID.")
    parser.add_argument("--Run_ID", dest="run_id", type=str, default=None, help="Numeric run serial, e.g. 0001.")
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


def build_model_config(payload: Dict[str, Any], dataset: LocalModuleDataset) -> LocalModuleConfig:
    model_cfg = dict(payload.get("model", {}))
    model_cfg["module_param_dim"] = _auto_int(model_cfg.get("module_param_dim"), dataset.module_param_dim)
    model_cfg["port_token_dim"] = _auto_int(model_cfg.get("port_token_dim"), dataset.port_token_dim)
    model_cfg["interface_target_dim"] = _auto_int(model_cfg.get("interface_target_dim"), dataset.interface_target_dim)
    return LocalModuleConfig.from_dict(model_cfg)


def normalize_run_id(value: Any, fallback: str = "0001") -> str:
    raw = str(value or fallback).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be a numeric serial such as '0001'; got {raw!r}.")
    return f"{int(raw):04d}"


def sanitize_run_suffix(value: Any) -> str:
    raw = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw).strip("_")


def resolve_run_id(args: argparse.Namespace, cfg: Dict[str, Any], fallback: str) -> str:
    training_cfg = cfg.get("training", {})
    return normalize_run_id(
        args.run_id
        or cfg.get("Run_ID")
        or cfg.get("run_id")
        or training_cfg.get("Run_ID")
        or training_cfg.get("run_id")
        or fallback,
        fallback,
    )


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], loss_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    internal_loss = F.mse_loss(outputs["internal_temperature"], batch["internal_temperature_targets"])
    interface_loss = F.mse_loss(outputs["interface_pred"], batch["interface_targets"])
    smoothness_weight = float(loss_cfg.get("smoothness_weight", 0.0))
    smoothness_loss = outputs["interface_pred"].new_tensor(0.0)
    if smoothness_weight > 0.0 and outputs["interface_pred"].shape[-2] > 1:
        diff = outputs["interface_pred"] - torch.roll(outputs["interface_pred"], shifts=1, dims=-2)
        smoothness_loss = diff.square().mean()
    total = (
        float(loss_cfg.get("internal_mse_weight", 1.0)) * internal_loss
        + float(loss_cfg.get("interface_mse_weight", 1.0)) * interface_loss
        + smoothness_weight * smoothness_loss
    )
    return {
        "loss_total": total,
        "loss_internal": internal_loss,
        "loss_interface": interface_loss,
        "loss_smoothness": smoothness_loss,
    }


def run_epoch(
    model: LocalModuleSurrogate,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: Dict[str, Any],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler=None,
    amp: bool = False,
    max_batches: Optional[int] = None,
    clip_norm: float = 1.0,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
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
            outputs = model(batch["module_params"], batch["port_tokens"], batch["internal_query_points"])
            losses = compute_losses(outputs, batch, loss_cfg)
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(losses["loss_total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
                optimizer.step()
        batch_size = int(batch["module_params"].shape[0])
        count += batch_size
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        iterator.set_postfix(loss=f"{float(losses['loss_total'].detach().cpu()):.3e}")
    if count == 0:
        return {key: float("nan") for key in ["loss_total", "loss_internal", "loss_interface", "loss_smoothness"]}
    return {key: value / count for key, value in sums.items()}


def save_checkpoint(
    path: Path,
    *,
    model: LocalModuleSurrogate,
    model_config: LocalModuleConfig,
    train_config: Dict[str, Any],
    dataset: LocalModuleDataset,
    epoch: int,
    best_metric: float,
) -> None:
    dataset_cfg = train_config.get("dataset", {})
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "model_config": model_config.to_dict(),
            "model_state_dict": model.state_dict(),
            "train_config": train_config,
            "dataset_feature_names": {
                "module_param_names": list(dataset.module_param_names),
                "port_input_feature_names": list(dataset.port_input_feature_names),
                "interface_target_names": list(dataset.interface_target_names),
            },
            "local_normalization_config": {
                "normalize_inputs": bool(dataset_cfg.get("normalize_inputs", False)),
                "normalize_targets": bool(dataset_cfg.get("normalize_targets", False)),
            },
            "local_normalization_stats": {name: value.copy() for name, value in dataset.normalizer.stats.items()},
        },
        path,
    )


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = read_json(config_path)
    training_cfg = cfg.get("training", {})
    dataset_cfg = cfg.get("dataset", {})
    loss_cfg = cfg.get("loss", {})
    set_seed(int(training_cfg.get("seed", 42)))
    device = select_device(args.device or training_cfg.get("device"))

    train_dataset = LocalModuleDataset(
        dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5"),
        split=dataset_cfg.get("train_split", "train"),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
    )
    val_split = dataset_cfg.get("val_split", "test")
    val_dataset = LocalModuleDataset(
        dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5"),
        split=val_split,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
    )
    if len(val_dataset) == 0:
        val_dataset = train_dataset

    model_config = build_model_config(cfg, train_dataset)
    model = LocalModuleSurrogate(model_config).to(device)
    print(f"[setup] device={device}, train_cases={len(train_dataset)}, val_cases={len(val_dataset)}")
    print(
        "[setup] normalization: "
        f"inputs={bool(dataset_cfg.get('normalize_inputs', False))}, "
        f"targets={bool(dataset_cfg.get('normalize_targets', False))}"
    )
    if train_dataset.normalizer.has("internal_temperature_mean", "internal_temperature_std"):
        internal_mean = train_dataset.normalizer.stats["internal_temperature_mean"].reshape(-1)[0]
        internal_std = train_dataset.normalizer.stats["internal_temperature_std"].reshape(-1)[0]
        interface_mean = train_dataset.normalizer.stats.get("interface_targets_mean")
        interface_std = train_dataset.normalizer.stats.get("interface_targets_std")
        print(f"[setup] internal T mean/std={float(internal_mean):.6g}/{float(internal_std):.6g}")
        if interface_mean is not None and interface_std is not None:
            print(
                "[setup] interface target mean/std="
                f"{interface_mean.astype(float).tolist()}/{interface_std.astype(float).tolist()}"
            )
    print(f"[setup] model parameters={count_parameters(model):,}")

    batch_size = int(dataset_cfg.get("batch_size", training_cfg.get("batch_size", 16)))
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
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 2.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-5)),
    )
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 200))
    scaler = make_grad_scaler(device, bool(training_cfg.get("amp", False)))

    saved_root = ensure_dir(resolve_demo_path(cfg.get("paths", {}).get("saved_model_dir", "./Saved_Model_LocalModule")))
    run_id = resolve_run_id(args, cfg, "0001")
    cfg["Run_ID"] = run_id
    run_suffix = sanitize_run_suffix(args.run_name or training_cfg.get("run_name"))
    run_name = f"Run_{run_id}_{run_suffix}_{current_timestamp()}" if run_suffix else f"Run_{run_id}_{current_timestamp()}"
    run_dir = ensure_dir(saved_root / run_name)
    write_json(run_dir / "resolved_train_config.json", cfg)
    history_path = run_dir / "loss_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "loss_total",
                "loss_internal",
                "loss_interface",
                "val_loss_total",
                "val_loss_internal",
                "val_loss_interface",
            ],
        )
        writer.writeheader()

    best_metric = math.inf
    max_train_batches = args.max_train_batches
    if max_train_batches is None and training_cfg.get("max_train_batches_per_epoch") is not None:
        max_train_batches = int(training_cfg["max_train_batches_per_epoch"])
    max_val_batches = args.max_val_batches
    if max_val_batches is None and training_cfg.get("max_val_batches") is not None:
        max_val_batches = int(training_cfg["max_val_batches"])

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            loss_cfg,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(training_cfg.get("amp", False)),
            max_batches=max_train_batches,
            clip_norm=float(training_cfg.get("gradient_clip_norm", 1.0)),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            loss_cfg,
            optimizer=None,
            scaler=None,
            amp=bool(training_cfg.get("amp", False)),
            max_batches=max_val_batches,
            clip_norm=float(training_cfg.get("gradient_clip_norm", 1.0)),
        )
        row = {
            "epoch": epoch,
            "loss_total": train_metrics.get("loss_total", math.nan),
            "loss_internal": train_metrics.get("loss_internal", math.nan),
            "loss_interface": train_metrics.get("loss_interface", math.nan),
            "val_loss_total": val_metrics.get("loss_total", math.nan),
            "val_loss_internal": val_metrics.get("loss_internal", math.nan),
            "val_loss_interface": val_metrics.get("loss_interface", math.nan),
        }
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
        metric = float(row["val_loss_total"])
        if math.isfinite(metric) and metric < best_metric:
            best_metric = metric
            save_checkpoint(
                run_dir / "best_model.pt",
                model=model,
                model_config=model_config,
                train_config=cfg,
                dataset=train_dataset,
                epoch=epoch,
                best_metric=best_metric,
            )
            print("Improved! Saving best model.")
        save_checkpoint(
            run_dir / "latest_model.pt",
            model=model,
            model_config=model_config,
            train_config=cfg,
            dataset=train_dataset,
            epoch=epoch,
            best_metric=best_metric,
        )
        save_loss_curve(history_path, run_dir / "loss_curve.png", title="Local Module Surrogate Loss")
        print(
            f"[epoch {epoch:04d}] loss={row['loss_total']:.4e} "
            f"internal={row['loss_internal']:.4e} interface={row['loss_interface']:.4e} "
            f"val={row['val_loss_total']:.4e}"
        )

    print(f"[done] saved local surrogate run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
