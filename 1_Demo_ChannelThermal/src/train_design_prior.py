from __future__ import annotations

"""Train the target-agnostic ChannelThermal latent design prior."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-design-prior")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

try:
    from channelthermal_model_utils import current_timestamp, ensure_dir, resolve_demo_path, select_device, set_seed, write_json
    from design_prior_dataset import DesignPriorDataset, collate_fn
    from model_design_prior import DesignPriorConfig, LatentModularDesignPrior
except Exception:  # pragma: no cover
    from .channelthermal_model_utils import current_timestamp, ensure_dir, resolve_demo_path, select_device, set_seed, write_json
    from .design_prior_dataset import DesignPriorDataset, collate_fn
    from .model_design_prior import DesignPriorConfig, LatentModularDesignPrior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ChannelThermal latent design prior.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def read_json(path: str | Path) -> Dict[str, Any]:
    with resolve_demo_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def mean_rows(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        vals = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        out[key] = float(np.mean(finite)) if finite.size else float("nan")
    return out


def run_epoch(
    model: LatentModularDesignPrior,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: float = 1.0,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows = []
    for batch in tqdm(loader, desc="train" if training else "val", leave=False, dynamic_ncols=True):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            metrics = model.training_loss(batch)
            loss = metrics["loss_total"]
        if training:
            loss.backward()
            if float(grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
        rows.append({key: scalar(value) for key, value in metrics.items()})
    return mean_rows(rows)


def write_history(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not history:
        return
    keys = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in keys})


def plot_loss_curves(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    epochs = [int(row["epoch"]) for row in history]
    for key in ("train_loss_total", "val_loss_total", "train_layout_recon_loss", "val_layout_recon_loss"):
        vals = [float(row.get(key, np.nan)) for row in history]
        if any(math.isfinite(v) for v in vals):
            ax.plot(epochs, vals, marker="o", markersize=3, label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_latent_norm_hist(model: LatentModularDesignPrior, loader: DataLoader, device: torch.device, path: Path) -> None:
    norms = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= 8:
                break
            batch = move_batch(batch, device)
            mu, logvar = model.encode(batch["design_vec"], batch.get("hypergraph_vec"), batch.get("behavior_vec"), batch.get("context_vec"))
            norms.extend(torch.linalg.norm(mu, dim=-1).detach().cpu().numpy().tolist())
    if not norms:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    ax.hist(norms, bins=24)
    ax.set_xlabel("latent ||mu||")
    ax.set_ylabel("count")
    ax.set_title("Latent norm distribution")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_generated_layout_examples(model: LatentModularDesignPrior, device: torch.device, path: Path, *, num_examples: int = 8) -> None:
    model.eval()
    with torch.no_grad():
        samples = model.sample(None, int(num_examples), device=device)["design_vec"].detach().cpu().numpy()
    m = int(model.cfg.max_num_modules)
    cols = min(4, int(num_examples))
    rows = int(math.ceil(num_examples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.2 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(axes_arr):
        ax.set_xlim(0.0, float(model.cfg.domain_length_x))
        ax.set_ylim(0.0, float(model.cfg.domain_length_y))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= samples.shape[0]:
            ax.axis("off")
            continue
        vec = samples[idx]
        centers = vec[: 2 * m].reshape(m, 2)
        mask = vec[2 * m : 3 * m] > 0.5
        centers[:, 0] *= float(model.cfg.domain_length_x)
        centers[:, 1] *= float(model.cfg.domain_length_y)
        ax.add_patch(plt.Rectangle((0.0, 0.0), float(model.cfg.domain_length_x), float(model.cfg.domain_length_y), fill=False, lw=1.0))
        for cx, cy in centers[mask]:
            ax.add_patch(plt.Circle((float(cx), float(cy)), float(model.cfg.module_radius), fill=False, lw=1.0))
        ax.set_title(f"sample {idx + 1}", fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_checkpoint(path: Path, *, model: LatentModularDesignPrior, cfg: Mapping[str, Any], dataset_stats: Mapping[str, Any], epoch: int, best_metric: float) -> None:
    torch.save(
        {
            "stage": "channelthermal_design_prior_vae",
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "model_config": model.cfg.to_dict(),
            "model_state_dict": model.state_dict(),
            "dataset_stats": dict(dataset_stats),
            "train_config": dict(cfg),
        },
        path,
    )


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), Mapping) else {}
    model_cfg = dict(cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {})
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), Mapping) else {}
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), Mapping) else {}
    set_seed(int(training_cfg.get("seed", 0)))
    device_arg = training_cfg.get("device", "auto")
    device = select_device(None if str(device_arg).lower() == "auto" else str(device_arg))
    dataset = DesignPriorDataset(data_cfg.get("library_path", "Data_Saved/DesignPrior_Library/design_library.h5"), behavior_dim=model_cfg.get("behavior_dim"), generate_heat_power=bool(model_cfg.get("generate_heat_power", False)))
    model_cfg["design_dim"] = int(dataset.design_vec.shape[1])
    model_cfg["context_dim"] = int(dataset.context_vec.shape[1])
    if model_cfg.get("hypergraph_dim") is None:
        model_cfg["hypergraph_dim"] = int(dataset.hypergraph_vec.shape[1])
    if int(model_cfg.get("behavior_dim", 0) or 0) <= 0 or dataset.behavior_vec.shape[1] == 0:
        model_cfg["behavior_dim"] = int(dataset.behavior_vec.shape[1])
    model_config = DesignPriorConfig.from_dict(model_cfg)
    model = LatentModularDesignPrior(model_config).to(device)
    if args.resume:
        checkpoint = torch.load(resolve_demo_path(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    val_len = int(round(len(dataset) * val_fraction)) if len(dataset) > 1 else 0
    val_len = min(max(val_len, 0), max(len(dataset) - 1, 0))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(int(training_cfg.get("seed", 0)))
    if val_len > 0:
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)
    else:
        train_dataset, val_dataset = dataset, dataset
    train_loader = DataLoader(train_dataset, batch_size=int(training_cfg.get("batch_size", 128)), shuffle=True, num_workers=int(data_cfg.get("num_workers", 0)), collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=int(training_cfg.get("batch_size", 128)), shuffle=False, num_workers=int(data_cfg.get("num_workers", 0)), collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get("learning_rate", 5.0e-4)), weight_decay=float(training_cfg.get("weight_decay", 1.0e-6)))
    run_dir = ensure_dir(resolve_demo_path(output_cfg.get("run_dir", f"Data_Saved/DesignPrior_Runs/Run_{current_timestamp()}")))
    write_json(run_dir / "resolved_train_design_prior_config.json", cfg)
    write_json(run_dir / "dataset_stats.json", dataset.stats)
    epochs = int(args.epochs or training_cfg.get("epochs", 200))
    history = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)))
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        row: Dict[str, Any] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        metric = float(row.get("val_loss_total", row.get("train_loss_total", float("inf"))))
        if metric < best:
            best = metric
            save_checkpoint(run_dir / "best_model.pt", model=model, cfg=cfg, dataset_stats=dataset.stats, epoch=epoch, best_metric=best)
        if epoch % max(int(training_cfg.get("save_every", 10)), 1) == 0 or epoch == epochs:
            save_checkpoint(run_dir / "latest_model.pt", model=model, cfg=cfg, dataset_stats=dataset.stats, epoch=epoch, best_metric=best)
        write_history(history, run_dir / "loss_history.csv")
        plot_loss_curves(history, run_dir / "loss_curves.png")
        print(f"[epoch {epoch:04d}] train={row.get('train_loss_total', float('nan')):.6g} val={row.get('val_loss_total', float('nan')):.6g} best={best:.6g}")
    plot_latent_norm_hist(model, val_loader, device, run_dir / "latent_norm_hist.png")
    plot_generated_layout_examples(model, device, run_dir / "generated_layout_examples.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
