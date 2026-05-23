"""Training entry point for the unified forward-model sandbox.

The script keeps the original dry-run smoke path and adds real multi-epoch
ChannelThermal training for forward-core ablations.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

from case_adapters import ChannelThermalAdapter, MultiCylinderAdapter, describe_batch, make_synthetic_batch
from channelthermal_dataset import ChannelThermalPointDataset, collate_batchdata
from diagnostics import (
    compute_basic_field_metrics,
    compute_channelthermal_region_metrics,
    compute_hypergraph_diagnostics,
    plot_organization_overview,
    save_diagnostics_json,
)
from unified_model_core import UnifiedHypergraphNeuralField
from unified_types import BatchData, UnifiedForwardConfig


SANDBOX_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    payload = load_json(args.config)
    if args.device is not None:
        payload.setdefault("training", {})["device"] = args.device
    if args.dry_run:
        run_dry_run(payload, args.case, args.inspect_data, args.max_steps)
        return
    train_one_run(
        payload,
        case=args.case,
        run_name=args.run_name,
        resume=args.resume,
        max_steps=args.max_steps,
    )


def train_one_run(
    config_payload: Dict[str, Any],
    *,
    case: str = "channelthermal",
    run_name: Optional[str] = None,
    resume: Optional[str] = None,
    dry_run: bool = False,
    max_steps: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    use_output_dir_as_run_dir: bool = False,
) -> Dict[str, Any]:
    """Train one unified forward run and return its summary."""
    if dry_run:
        return run_dry_run(config_payload, case=case, inspect_data=False, max_steps=max_steps)
    if case != "channelthermal":
        raise ValueError("Real multi-epoch training is currently implemented only for --case channelthermal.")

    training_cfg = config_payload.get("training", {})
    torch.manual_seed(int(training_cfg.get("seed", 0)))
    device = resolve_device(training_cfg.get("device", "auto"))
    model_cfg = UnifiedForwardConfig.from_dict(config_payload.get("model", {}))
    if case == "channelthermal":
        model_cfg = adapt_model_config_to_channelthermal_dataset(config_payload, model_cfg)
    run_dir = Path(output_dir) if use_output_dir_as_run_dir and output_dir is not None else resolve_run_dir(run_name, output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, target_stats = build_channelthermal_datasets(config_payload, model_cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batchdata,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg.get("val_batch_size", training_cfg.get("batch_size", 16))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batchdata,
    )

    model = UnifiedHypergraphNeuralField(model_cfg).to(device)
    initialize_lazy_model(model, train_loader, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_field_mse_physical = float("inf")
    best_val_temperature_mse = float("inf")
    best_by_loss_metrics: Dict[str, Any] = {}
    best_by_field_mse_metrics: Dict[str, Any] = {}
    best_by_temperature_mse_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_val_field_mse_physical = float(checkpoint.get("best_val_field_mse_physical", float("inf")))
        best_val_temperature_mse = float(checkpoint.get("best_val_temperature_mse", float("inf")))

    config_resolved = dict(config_payload)
    config_resolved["model"] = model_cfg.to_dict()
    config_resolved.setdefault("training", {}).update(
        {
            "target_stats": stats_to_json(target_stats),
            "run_dir": str(run_dir),
        }
    )
    write_json(run_dir / "config_resolved.json", config_resolved)

    metrics_csv = run_dir / "metrics.csv"
    latest_path = run_dir / "latest_model.pt"
    best_path = run_dir / "best_model.pt"
    best_by_loss_path = run_dir / "best_by_loss_model.pt"
    best_by_field_mse_path = run_dir / "best_by_field_mse_model.pt"
    best_by_temperature_mse_path = run_dir / "best_by_temperature_mse_model.pt"
    total_steps = 0
    epochs = int(training_cfg.get("epochs", 1))
    eval_every = max(1, int(training_cfg.get("eval_every", 1)))
    save_every = max(1, int(training_cfg.get("save_every", 1)))
    max_steps_int = int(max_steps) if max_steps is not None and int(max_steps) > 0 else None

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, train_steps = run_epoch(
            model,
            train_loader,
            device,
            training_cfg,
            target_stats,
            optimizer=optimizer,
            max_batches=training_cfg.get("max_train_batches_per_epoch"),
            max_steps=max_steps_int - total_steps if max_steps_int is not None else None,
        )
        total_steps += train_steps
        should_eval = epoch == 1 or epoch % eval_every == 0 or (max_steps_int is not None and total_steps >= max_steps_int)
        val_metrics: Dict[str, Any] = {}
        last_val_output: Optional[Dict[str, Any]] = None
        if should_eval:
            val_metrics, last_val_output = evaluate_loader(
                model,
                val_loader,
                device,
                training_cfg,
                target_stats,
                model_cfg,
                max_batches=training_cfg.get("max_val_batches"),
            )
            if last_val_output is not None:
                plot_organization_overview(last_val_output, run_dir / "val_organization_latest.png")
                save_temperature_visualizations(last_val_output, run_dir, target_stats)

        row = {"epoch": epoch, "train_steps": total_steps, **prefix_keys(train_metrics, "train_"), **val_metrics}
        history.append(row)
        append_metrics_csv(metrics_csv, row)
        save_loss_curves(history, run_dir / "loss_curves.png")

        val_loss = float(val_metrics.get("val_loss", float("nan")))
        val_field_mse = float(val_metrics.get("val_field_mse_physical", float("nan")))
        val_temperature_mse = float(val_metrics.get("val_temperature_mse", float("nan")))
        is_best_loss = math.isfinite(val_loss) and val_loss < best_val_loss
        is_best_field_mse = math.isfinite(val_field_mse) and val_field_mse < best_val_field_mse_physical
        is_best_temperature_mse = math.isfinite(val_temperature_mse) and val_temperature_mse < best_val_temperature_mse
        if is_best_loss:
            best_val_loss = val_loss
            best_by_loss_metrics = dict(val_metrics)
        if is_best_field_mse:
            best_val_field_mse_physical = val_field_mse
            best_by_field_mse_metrics = dict(val_metrics)
        if is_best_temperature_mse:
            best_val_temperature_mse = val_temperature_mse
            best_by_temperature_mse_metrics = dict(val_metrics)
        if is_best_loss or is_best_field_mse or is_best_temperature_mse or epoch % save_every == 0 or max_steps_int is not None:
            checkpoint = make_checkpoint(
                epoch,
                model,
                optimizer,
                config_resolved,
                model_cfg,
                target_stats,
                best_val_loss,
                val_metrics,
                best_val_field_mse_physical=best_val_field_mse_physical,
                best_val_temperature_mse=best_val_temperature_mse,
            )
            torch.save(checkpoint, latest_path)
            if is_best_loss:
                torch.save(checkpoint, best_by_loss_path)
                torch.save(checkpoint, best_path)
            if is_best_field_mse:
                torch.save(checkpoint, best_by_field_mse_path)
            if is_best_temperature_mse:
                torch.save(checkpoint, best_by_temperature_mse_path)

        print(
            f"[epoch] {epoch:03d}/{epochs:03d} train_loss={train_metrics.get('loss', float('nan')):.4e} "
            f"val_loss={val_metrics.get('val_loss', float('nan')):.4e} steps={total_steps}"
        )
        if max_steps_int is not None and total_steps >= max_steps_int:
            break

    if not latest_path.exists():
        final_val = history[-1] if history else {}
        torch.save(
            make_checkpoint(
                epoch,
                model,
                optimizer,
                config_resolved,
                model_cfg,
                target_stats,
                best_val_loss,
                final_val,
                best_val_field_mse_physical=best_val_field_mse_physical,
                best_val_temperature_mse=best_val_temperature_mse,
            ),
            latest_path,
        )
        if not best_path.exists():
            torch.save(
                make_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    config_resolved,
                    model_cfg,
                    target_stats,
                    best_val_loss,
                    final_val,
                    best_val_field_mse_physical=best_val_field_mse_physical,
                    best_val_temperature_mse=best_val_temperature_mse,
                ),
                best_path,
            )
        if not best_by_loss_path.exists():
            torch.save(torch.load(best_path, map_location="cpu"), best_by_loss_path)
        if not best_by_field_mse_path.exists():
            torch.save(torch.load(latest_path, map_location="cpu"), best_by_field_mse_path)
        if not best_by_temperature_mse_path.exists():
            torch.save(torch.load(latest_path, map_location="cpu"), best_by_temperature_mse_path)

    summary = {
        "run_dir": str(run_dir),
        "best_val_loss": best_val_loss,
        "best_val_field_mse_physical": best_val_field_mse_physical,
        "best_val_temperature_mse": best_val_temperature_mse,
        "best_metrics": best_by_loss_metrics or (history[-1] if history else {}),
        "best_by_loss_metrics": best_by_loss_metrics or (history[-1] if history else {}),
        "best_by_field_mse_metrics": best_by_field_mse_metrics or (history[-1] if history else {}),
        "best_by_temperature_mse_metrics": best_by_temperature_mse_metrics or (history[-1] if history else {}),
        "latest_epoch": history[-1]["epoch"] if history else 0,
        "latest_metrics": history[-1] if history else {},
        "target_stats": stats_to_json(target_stats),
    }
    write_json(run_dir / "summary.json", summary)
    train_dataset.close()
    val_dataset.close()
    return summary


def run_dry_run(
    config_payload: Dict[str, Any],
    case: str,
    inspect_data: bool = False,
    max_steps: Optional[int] = 1,
) -> Dict[str, Any]:
    training_cfg = config_payload.get("training", {})
    torch.manual_seed(int(training_cfg.get("seed", 0)))
    device = resolve_device(training_cfg.get("device", "auto"))
    model_cfg = UnifiedForwardConfig.from_dict(config_payload.get("model", {}))
    if case == "channelthermal":
        model_cfg = adapt_model_config_to_channelthermal_dataset(config_payload, model_cfg)
    batch = load_batch(case, config_payload, batch_size=int(training_cfg.get("batch_size", 1)))
    if inspect_data:
        print(json.dumps({"batch": describe_batch(batch)}, indent=2, sort_keys=True))
    model = UnifiedHypergraphNeuralField(model_cfg).to(device)
    batch = batch.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get("learning_rate", 3e-4)))
    steps = max(int(max_steps or 1), 1)
    last_output: Dict[str, Any] = {}
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        pred = output["pred_field"]
        if inspect_data:
            print(json.dumps({"pred_field_shape": list(pred.shape)}, indent=2, sort_keys=True))
        if batch.target_field is not None and pred.shape == batch.target_field.shape:
            loss = torch.mean((pred - batch.target_field.float()) ** 2)
            loss.backward()
        last_output = output

    metrics = compute_basic_field_metrics(last_output["pred_field"], batch.target_field)
    metrics.update(compute_hypergraph_diagnostics(last_output))
    metrics.update({"case": batch.case_name, "dry_run": True, "max_steps": steps})
    out_dir = SANDBOX_ROOT / "results" / "dry_run"
    save_diagnostics_json({"metrics": metrics, "batch": batch.to_dict(), "config": model_cfg.to_dict()}, out_dir / "train_metrics.json")
    plot_organization_overview(last_output, out_dir / "train_organization.png")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def build_channelthermal_datasets(
    config_payload: Dict[str, Any],
    model_cfg: UnifiedForwardConfig,
) -> tuple[ChannelThermalPointDataset, ChannelThermalPointDataset, Dict[str, torch.Tensor | bool]]:
    data_cfg = config_payload.get("data", {})
    training_cfg = config_payload.get("training", {})
    path = data_cfg.get("channelthermal_dataset_path", "../1_Demo_ChannelThermal/Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5")
    normalize_targets = bool(training_cfg.get("normalize_targets", True))
    heat_feature_mode = str(training_cfg.get("module_heat_feature_mode", "both"))
    stats_probe = ChannelThermalPointDataset(
        path,
        split="train",
        max_cases=training_cfg.get("max_train_cases"),
        points_per_case=int(training_cfg.get("points_per_case", 1024)),
        max_num_modules=int(model_cfg.max_num_modules),
        field_dim=int(model_cfg.field_dim),
        normalize_targets=False,
        module_heat_feature_mode=heat_feature_mode,
        seed=int(training_cfg.get("seed", 0)),
    )
    computed_stats = stats_probe.compute_target_stats(max_cases=training_cfg.get("max_train_cases"))
    dataset_heat_scale = float(stats_probe.dataset_heat_scale or 1.0)
    stats_probe.close()
    target_mean = computed_stats["mean"]
    target_std = computed_stats["std"]
    train_dataset = ChannelThermalPointDataset(
        path,
        split="train",
        max_cases=training_cfg.get("max_train_cases"),
        points_per_case=int(training_cfg.get("points_per_case", 1024)),
        max_num_modules=int(model_cfg.max_num_modules),
        field_dim=int(model_cfg.field_dim),
        normalize_targets=normalize_targets,
        target_mean=target_mean,
        target_std=target_std,
        module_heat_feature_mode=heat_feature_mode,
        dataset_heat_scale=dataset_heat_scale,
        seed=int(training_cfg.get("seed", 0)),
    )
    val_dataset = ChannelThermalPointDataset(
        path,
        split="test",
        max_cases=training_cfg.get("max_val_cases"),
        points_per_case=int(training_cfg.get("val_points_per_case", training_cfg.get("points_per_case", 1024))),
        max_num_modules=int(model_cfg.max_num_modules),
        field_dim=int(model_cfg.field_dim),
        normalize_targets=normalize_targets,
        target_mean=target_mean,
        target_std=target_std,
        module_heat_feature_mode=heat_feature_mode,
        dataset_heat_scale=dataset_heat_scale,
        seed=int(training_cfg.get("seed", 0)) + 999,
    )
    return train_dataset, val_dataset, {
        "mean": target_mean,
        "std": target_std,
        "normalize_targets": normalize_targets,
        "dataset_heat_scale": dataset_heat_scale,
        "module_heat_feature_mode": heat_feature_mode,
    }


def adapt_model_config_to_channelthermal_dataset(
    config_payload: Dict[str, Any],
    model_cfg: UnifiedForwardConfig,
) -> UnifiedForwardConfig:
    data_cfg = config_payload.get("data", {})
    path = Path(data_cfg.get("channelthermal_dataset_path", "../1_Demo_ChannelThermal/Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"))
    if not path.is_absolute():
        path = (SANDBOX_ROOT / path).resolve()
    try:
        import h5py  # type: ignore
        import numpy as np

        with h5py.File(path, "r") as h5:
            first_case = next(iter(h5["cases"].keys()))
            group = h5["cases"][first_case]
            x_grid = np.asarray(group["x_grid"], dtype=np.float32)
            y_grid = np.asarray(group["y_grid"], dtype=np.float32)
            dx = float(np.mean(np.diff(x_grid[0]))) if x_grid.ndim == 2 and x_grid.shape[1] > 1 else 0.0
            dy = float(np.mean(np.diff(y_grid[:, 0]))) if y_grid.ndim == 2 and y_grid.shape[0] > 1 else 0.0
            lx = float(x_grid.max() - x_grid.min() + abs(dx))
            ly = float(y_grid.max() - y_grid.min() + abs(dy))
        if abs(float(model_cfg.domain_length_x) - lx) > 1e-5 or abs(float(model_cfg.domain_length_y) - ly) > 1e-5:
            print(
                "[setup] adapting ChannelThermal domain from dataset: "
                f"({model_cfg.domain_length_x}, {model_cfg.domain_length_y}) -> ({lx}, {ly})"
            )
            payload = {**model_cfg.to_dict(), "domain_length_x": lx, "domain_length_y": ly}
            return UnifiedForwardConfig.from_dict(payload)
    except Exception as exc:
        print(f"[setup] could not infer ChannelThermal domain from dataset ({exc}); using config values.")
    return model_cfg


def run_epoch(
    model: UnifiedHypergraphNeuralField,
    loader: DataLoader,
    device: torch.device,
    training_cfg: Dict[str, Any],
    target_stats: Dict[str, Any],
    *,
    optimizer: Optional[torch.optim.Optimizer],
    max_batches: Optional[Any] = None,
    max_steps: Optional[int] = None,
) -> tuple[Dict[str, float], int]:
    model.train(optimizer is not None)
    sums: Dict[str, float] = {}
    count = 0
    steps = 0
    max_batches_int = int(max_batches) if max_batches is not None else None
    for batch_idx, batch in enumerate(loader):
        if max_batches_int is not None and batch_idx >= max_batches_int:
            break
        if max_steps is not None and steps >= max_steps:
            break
        batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(batch)
            loss = weighted_mse_loss(output["pred_field"], batch.target_field, training_cfg)
        if optimizer is not None:
            loss.backward()
            clip = float(training_cfg.get("grad_clip_norm", training_cfg.get("gradient_clip_norm", 1.0)))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        batch_size = int(batch.query_xy.shape[0])
        count += batch_size
        steps += 1
        sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu()) * batch_size
        for key, value in compute_hypergraph_diagnostics(output).items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
    return ({key: value / max(count, 1) for key, value in sums.items()}, steps)


def evaluate_loader(
    model: UnifiedHypergraphNeuralField,
    loader: DataLoader,
    device: torch.device,
    training_cfg: Dict[str, Any],
    target_stats: Dict[str, Any],
    model_cfg: UnifiedForwardConfig,
    *,
    max_batches: Optional[Any] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    model.eval()
    sums: Dict[str, float] = {}
    count = 0
    last_output: Optional[Dict[str, Any]] = None
    max_batches_int = int(max_batches) if max_batches is not None else None
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches_int is not None and batch_idx >= max_batches_int:
                break
            batch = batch.to(device)
            output = model(batch)
            loss = weighted_mse_loss(output["pred_field"], batch.target_field, training_cfg)
            physical_pred = denormalize_field(output["pred_field"], target_stats)
            physical_target = denormalize_field(batch.target_field, target_stats)
            last_output = dict(output)
            last_output["_query_xy"] = batch.query_xy.detach()
            if physical_pred is not None:
                last_output["_physical_pred_field"] = physical_pred.detach()
            if physical_target is not None:
                last_output["_physical_target_field"] = physical_target.detach()
            metrics = compute_basic_field_metrics(physical_pred, physical_target)
            if batch.case_name == "channelthermal":
                metrics.update(compute_channelthermal_region_metrics(physical_pred, physical_target, batch, model_cfg))
            metrics.update(compute_hypergraph_diagnostics(output))
            metrics["val_loss"] = float(loss.cpu())
            batch_size = int(batch.query_xy.shape[0])
            count += batch_size
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    sums[key] = sums.get(key, 0.0) + float(value) * batch_size
    averaged = {f"val_{key}" if not str(key).startswith("val_") else str(key): value / max(count, 1) for key, value in sums.items()}
    if "val_field_mse" in averaged:
        averaged["val_field_mse_physical"] = averaged["val_field_mse"]
    return averaged, last_output


def weighted_mse_loss(pred: torch.Tensor, target: Optional[torch.Tensor], training_cfg: Dict[str, Any]) -> torch.Tensor:
    if target is None:
        return pred.new_tensor(0.0)
    target = target.to(device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        return pred.new_tensor(0.0)
    diff = (pred - target).square()
    weights = training_cfg.get("field_channel_weights")
    if weights is None:
        channel_weights = torch.ones(pred.shape[-1], device=pred.device, dtype=pred.dtype)
    else:
        channel_weights = torch.as_tensor(weights, device=pred.device, dtype=pred.dtype)
        if channel_weights.numel() != pred.shape[-1]:
            raise ValueError(f"field_channel_weights length {channel_weights.numel()} must match field_dim={pred.shape[-1]}.")
    if pred.shape[-1] >= 5:
        channel_weights = channel_weights.clone()
        channel_weights[4] = channel_weights[4] * float(training_cfg.get("temperature_weight", 1.0))
    return (diff * channel_weights.view(*([1] * (diff.ndim - 1)), -1)).mean()


def denormalize_field(values: Optional[torch.Tensor], target_stats: Dict[str, Any]) -> Optional[torch.Tensor]:
    if values is None:
        return None
    if not bool(target_stats.get("normalize_targets", False)):
        return values
    mean = target_stats["mean"].to(device=values.device, dtype=values.dtype)
    std = target_stats["std"].to(device=values.device, dtype=values.dtype)
    return values * std.view(*([1] * (values.ndim - 1)), -1) + mean.view(*([1] * (values.ndim - 1)), -1)


def initialize_lazy_model(model: UnifiedHypergraphNeuralField, loader: DataLoader, device: torch.device) -> None:
    sample = next(iter(loader)).to(device)
    model.eval()
    with torch.no_grad():
        model(sample)


def make_checkpoint(
    epoch: int,
    model: UnifiedHypergraphNeuralField,
    optimizer: torch.optim.Optimizer,
    config_resolved: Dict[str, Any],
    model_cfg: UnifiedForwardConfig,
    target_stats: Dict[str, Any],
    best_val_loss: float,
    val_metrics: Dict[str, Any],
    *,
    best_val_field_mse_physical: float = float("inf"),
    best_val_temperature_mse: float = float("inf"),
) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config_resolved": config_resolved,
        "model_config": model_cfg.to_dict(),
        "target_stats": stats_to_json(target_stats),
        "best_val_loss": float(best_val_loss),
        "best_val_field_mse_physical": float(best_val_field_mse_physical),
        "best_val_temperature_mse": float(best_val_temperature_mse),
        "val_metrics": val_metrics,
    }


def load_batch(case: str, payload: Dict[str, Any], batch_size: int = 1) -> BatchData:
    data_cfg = payload.get("data", {})
    training_cfg = payload.get("training", {})
    if case == "synthetic":
        return make_synthetic_batch("channel", batch_size=batch_size, points_per_case=192)
    if case == "channelthermal":
        dataset = ChannelThermalPointDataset(
            data_cfg.get("channelthermal_dataset_path", "../1_Demo_ChannelThermal/Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"),
            split="train",
            max_cases=batch_size,
            points_per_case=int(training_cfg.get("points_per_case", 192)),
            max_num_modules=int(payload.get("model", {}).get("max_num_modules", 12)),
            field_dim=int(payload.get("model", {}).get("field_dim", 5)),
            normalize_targets=False,
            module_heat_feature_mode=str(training_cfg.get("module_heat_feature_mode", "both")),
            seed=int(training_cfg.get("seed", 0)),
        )
        items = [dataset[idx] for idx in range(min(batch_size, len(dataset)))]
        dataset.close()
        return collate_batchdata(items)
    return MultiCylinderAdapter(data_cfg.get("multicylinder_dataset_path", MultiCylinderAdapter().dataset_path)).load_one_batch(
        batch_size=batch_size,
        points_per_case=192,
    )


def save_temperature_visualizations(output: Dict[str, Any], run_dir: Path, target_stats: Dict[str, Any]) -> None:
    """Save lightweight scatter plots for one validation batch's temperature field."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    query_xy = output.get("_query_xy")
    pred = output.get("_physical_pred_field")
    target = output.get("_physical_target_field")
    if not (torch.is_tensor(query_xy) and torch.is_tensor(pred) and torch.is_tensor(target)):
        return
    if pred.ndim < 3 or target.ndim < 3 or pred.shape[-1] < 5 or target.shape[-1] < 5:
        return
    xy = query_xy[0].detach().cpu()
    pred_temp = pred[0, :, 4].detach().cpu()
    target_temp = target[0, :, 4].detach().cpu()
    error_temp = (pred_temp - target_temp).abs()
    for filename, values, title in [
        ("val_temperature_target_latest.png", target_temp, "Temperature Target"),
        ("val_temperature_pred_latest.png", pred_temp, "Temperature Prediction"),
        ("val_temperature_error_latest.png", error_temp, "Temperature Absolute Error"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=8, cmap="inferno")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(run_dir / filename)
        plt.close(fig)


def save_loss_curves(history: List[Dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    for key, label in [("train_loss", "train"), ("val_loss", "val"), ("val_field_mse_physical", "val physical")]:
        values = [row.get(key, float("nan")) for row in history]
        xs = [x for x, y in zip(epochs, values) if isinstance(y, (int, float)) and math.isfinite(float(y)) and float(y) > 0.0]
        ys = [float(y) for y in values if isinstance(y, (int, float)) and math.isfinite(float(y)) and float(y) > 0.0]
        if xs and ys:
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def append_metrics_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_fields: List[str] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing_fields = next(reader, [])
    fields = sorted(set(existing_fields) | set(row.keys()))
    rewrite = bool(existing_fields and existing_fields != fields)
    rows: List[Dict[str, Any]] = []
    if rewrite:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows.append(row)
    mode = "w" if rewrite or not path.exists() else "a"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer.writerow(row)


def prefix_keys(payload: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in payload.items()}


def stats_to_json(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mean": stats["mean"].detach().cpu().tolist(),
        "std": stats["std"].detach().cpu().tolist(),
        "normalize_targets": bool(stats.get("normalize_targets", False)),
        "dataset_heat_scale": float(stats.get("dataset_heat_scale", 1.0)),
        "module_heat_feature_mode": str(stats.get("module_heat_feature_mode", "both")),
    }


def stats_from_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mean": torch.as_tensor(payload["mean"], dtype=torch.float32),
        "std": torch.as_tensor(payload["std"], dtype=torch.float32).clamp_min(1e-6),
        "normalize_targets": bool(payload.get("normalize_targets", False)),
    }


def resolve_run_dir(run_name: Optional[str], output_dir: Optional[str | Path]) -> Path:
    root = Path(output_dir) if output_dir is not None else SANDBOX_ROOT / "results" / "runs"
    if root.name.startswith("Run_") and re.match(r"^Run_\d{4}_\d{8}_\d{6}", root.name):
        return root
    root.mkdir(parents=True, exist_ok=True)
    run_id = next_run_id(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = sanitize_run_label(run_name)
    name = f"Run_{run_id:04d}_{timestamp}" + (f"_{suffix}" if suffix else "")
    return root / name


def next_run_id(root: Path) -> int:
    max_id = 0
    pattern = re.compile(r"^Run_(\d{4})_\d{8}_\d{6}")
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def sanitize_run_label(label: Optional[str]) -> str:
    if not label:
        return ""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
    return text.strip("_")[:80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified_forward_config_template.json")
    parser.add_argument("--case", choices=["channelthermal", "multicylinder", "synthetic"], default="synthetic")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None, help="Override training.device, for example cpu, cuda, or cuda:0.")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = SANDBOX_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_device(value: Any) -> torch.device:
    if value in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(value))


if __name__ == "__main__":
    main()
