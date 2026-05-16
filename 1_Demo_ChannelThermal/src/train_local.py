from __future__ import annotations

"""Train the Stage A local module thermal surrogate."""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import _bootstrap_imports  # noqa: F401
from channelthermal_datasets import GlobalModuleAlignmentDataset, LocalModuleDataset
from channelthermal_model_utils import (
    autocast_context,
    count_parameters,
    current_timestamp,
    ensure_dir,
    load_trusted_checkpoint,
    make_grad_scaler,
    read_json,
    recursive_to_device,
    resolve_demo_path,
    select_device,
    set_seed,
    strip_module_prefix,
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
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Optional Stage-A checkpoint to initialize from before fine-tuning.")
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


def build_model_config(payload: Dict[str, Any], dataset: Dataset) -> LocalModuleConfig:
    model_cfg = dict(payload.get("model", {}))
    model_cfg["module_param_dim"] = _auto_int(model_cfg.get("module_param_dim"), dataset.module_param_dim)
    model_cfg["port_token_dim"] = _auto_int(model_cfg.get("port_token_dim"), dataset.port_token_dim)
    model_cfg["interface_target_dim"] = _auto_int(model_cfg.get("interface_target_dim"), dataset.interface_target_dim)
    return LocalModuleConfig.from_dict(model_cfg)


class MixedLocalDataset(Dataset):
    """Simple concatenation with local-surrogate metadata exposed for training."""

    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = [dataset for dataset in datasets if len(dataset) > 0]
        if not self.datasets:
            raise ValueError("MixedLocalDataset requires at least one non-empty dataset.")
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative = np.cumsum(self.lengths).tolist()
        first = self.datasets[0]
        self.module_param_names = getattr(first, "module_param_names", [])
        self.port_input_feature_names = getattr(first, "port_input_feature_names", [])
        self.interface_target_names = getattr(first, "interface_target_names", [])
        self.module_param_dim = int(getattr(first, "module_param_dim"))
        self.port_token_dim = int(getattr(first, "port_token_dim"))
        self.interface_target_dim = int(getattr(first, "interface_target_dim"))
        self.n_interface_points = int(getattr(first, "n_interface_points"))
        self.num_internal_points = int(getattr(first, "num_internal_points"))
        self.normalizer = getattr(first, "normalizer", None)

    def __len__(self) -> int:
        return int(sum(self.lengths))

    def __getitem__(self, item: int) -> Dict[str, Any]:
        idx = int(item)
        for dataset_idx, end in enumerate(self.cumulative):
            start = 0 if dataset_idx == 0 else self.cumulative[dataset_idx - 1]
            if idx < end:
                return self.datasets[dataset_idx][idx - start]
        return self.datasets[-1][self.lengths[-1] - 1]


def build_stage_a_dataset(dataset_cfg: Dict[str, Any], *, split: str, source: str) -> Dataset:
    normalize_inputs = bool(dataset_cfg.get("normalize_inputs", False))
    normalize_targets = bool(dataset_cfg.get("normalize_targets", False))
    source = str(source).lower()
    if source == "local":
        return LocalModuleDataset(
            dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5"),
            split=split,
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
        )
    if source == "global_alignment":
        return GlobalModuleAlignmentDataset(
            dataset_cfg.get("global_alignment_packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"),
            split=split,
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
        )
    if source == "mixed":
        local = LocalModuleDataset(
            dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5"),
            split=split,
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
        )
        global_split = dataset_cfg.get("global_alignment_split", split) if split == dataset_cfg.get("train_split", "train") else dataset_cfg.get("global_alignment_val_split", split)
        global_align = GlobalModuleAlignmentDataset(
            dataset_cfg.get("global_alignment_packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"),
            split=global_split,
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
        )
        return MixedLocalDataset([local, global_align])
    raise ValueError("dataset.source must be 'local', 'global_alignment', or 'mixed'.")


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
    if loss_cfg.get("interface_target_weights", None) is not None:
        weights = torch.as_tensor(loss_cfg["interface_target_weights"], device=outputs["interface_pred"].device, dtype=outputs["interface_pred"].dtype)
        if weights.numel() < outputs["interface_pred"].shape[-1]:
            weights = F.pad(weights, (0, outputs["interface_pred"].shape[-1] - weights.numel()), value=1.0)
        weights = weights[: outputs["interface_pred"].shape[-1]].clamp_min(0.0)
        mse_by_target = (outputs["interface_pred"] - batch["interface_targets"]).square().mean(dim=(0, 1))
        interface_loss = (mse_by_target * weights).sum() / weights.sum().clamp_min(1.0e-6)
    else:
        interface_loss = F.mse_loss(outputs["interface_pred"], batch["interface_targets"])
    smoothness_weight = float(loss_cfg.get("interface_smoothness_weight", loss_cfg.get("smoothness_weight", 0.0)))
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
    dataset: Dataset,
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
            "local_normalization_stats": {
                name: value.copy()
                for name, value in getattr(getattr(dataset, "normalizer", None), "stats", {}).items()
            },
        },
        path,
    )


def load_init_checkpoint_if_requested(model: LocalModuleSurrogate, init_path: Optional[str], device: torch.device) -> None:
    if not init_path:
        return
    checkpoint = load_trusted_checkpoint(resolve_demo_path(init_path), map_location=device)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    missing, unexpected = model.load_state_dict(strip_module_prefix(state), strict=False)
    print(
        f"[setup] initialized from {resolve_demo_path(init_path)} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


def save_local_loss_curve(
    history_path: Path,
    output_path: Path,
    *,
    include_smoothness: bool = False,
) -> None:
    """Plot the primary Stage-A losses without crowding the figure.

    Smoothness is a small auxiliary regularizer and remains in loss_history.csv
    for diagnostics, but it is omitted from the figure unless explicitly
    requested with loss.plot_smoothness_loss=true.
    """
    if not history_path.exists():
        return
    rows = np.genfromtxt(history_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.size == 0:
        return
    if rows.ndim == 0:
        rows = np.asarray([rows])
    names = rows.dtype.names or ()
    if "epoch" not in names:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [
        "loss_total",
        "val_loss_total",
        "loss_internal",
        "val_loss_internal",
        "loss_interface",
        "val_loss_interface",
    ]
    if include_smoothness:
        keys.extend(["loss_smoothness", "val_loss_smoothness"])

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for key in keys:
        if key not in names:
            continue
        values = np.asarray(rows[key], dtype=float)
        if np.any(np.isfinite(values)):
            ax.plot(rows["epoch"], values, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Local Module Surrogate Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = read_json(config_path)
    training_cfg = cfg.get("training", {})
    dataset_cfg = cfg.get("dataset", {})
    loss_cfg = cfg.get("loss", {})
    set_seed(int(training_cfg.get("seed", 42)))
    device = select_device(args.device or training_cfg.get("device"))

    dataset_source = str(dataset_cfg.get("source", "local")).lower()
    train_split = (
        dataset_cfg.get("global_alignment_split", "train")
        if dataset_source == "global_alignment"
        else dataset_cfg.get("train_split", "train")
    )
    train_dataset = build_stage_a_dataset(dataset_cfg, split=train_split, source=dataset_source)
    val_split = dataset_cfg.get("val_split", "test")
    if dataset_source == "global_alignment":
        val_split = dataset_cfg.get("global_alignment_val_split", dataset_cfg.get("global_alignment_split", "train"))
    val_dataset = build_stage_a_dataset(dataset_cfg, split=val_split, source=dataset_source)
    if len(val_dataset) == 0:
        val_dataset = train_dataset

    model_config = build_model_config(cfg, train_dataset)
    model = LocalModuleSurrogate(model_config).to(device)
    init_checkpoint = args.init_checkpoint or training_cfg.get("init_checkpoint_path")
    load_init_checkpoint_if_requested(model, init_checkpoint, device)
    print(f"[setup] device={device}, source={dataset_source}, train_cases={len(train_dataset)}, val_cases={len(val_dataset)}")
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
                "loss_smoothness",
                "val_loss_total",
                "val_loss_internal",
                "val_loss_interface",
                "val_loss_smoothness",
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
            "loss_smoothness": train_metrics.get("loss_smoothness", math.nan),
            "val_loss_total": val_metrics.get("loss_total", math.nan),
            "val_loss_internal": val_metrics.get("loss_internal", math.nan),
            "val_loss_interface": val_metrics.get("loss_interface", math.nan),
            "val_loss_smoothness": val_metrics.get("loss_smoothness", math.nan),
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
        save_local_loss_curve(
            history_path,
            run_dir / "loss_curve.png",
            include_smoothness=bool(loss_cfg.get("plot_smoothness_loss", False)),
        )
        print(
            f"[epoch {epoch:04d}] loss={row['loss_total']:.4e} "
            f"internal={row['loss_internal']:.4e} interface={row['loss_interface']:.4e} "
            f"val={row['val_loss_total']:.4e}"
        )

    print(f"[done] saved local surrogate run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
