"""Train external naive ChannelThermal neural-field baselines.

These baselines predict ``U`` from ``D, c, q`` without learned hypergraph
organization ``H`` or any HONF decoder internals.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-unified-forward")

from case_adapters import describe_batch
from channelthermal_dataset import ChannelThermalPointDataset, collate_batchdata
from diagnostics import compute_basic_field_metrics, compute_channelthermal_region_metrics
from naive_field_baselines import NaiveFieldBaseline, NaiveFieldBaselineConfig
from train_unified import (
    SANDBOX_ROOT,
    append_metrics_csv,
    denormalize_field,
    jsonable,
    prefix_keys,
    resolve_device,
    save_loss_curves,
    save_temperature_visualizations,
    stats_to_json,
    weighted_mse_loss,
    write_json,
)


def main() -> None:
    args = parse_args()
    payload = load_json(args.config)
    if args.device is not None:
        payload.setdefault("training", {})["device"] = args.device
    if args.dry_run:
        run_dry_run(payload, inspect_data=bool(args.inspect_data), max_steps=args.max_steps)
        return
    train_one_run(payload, run_name=args.run_name, resume=args.resume, max_steps=args.max_steps)


def train_one_run(
    config_payload: Dict[str, Any],
    *,
    run_name: Optional[str] = None,
    resume: Optional[str] = None,
    max_steps: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    use_output_dir_as_run_dir: bool = False,
) -> Dict[str, Any]:
    training_cfg = config_payload.get("training", {})
    torch.manual_seed(int(training_cfg.get("seed", 0)))
    device = resolve_device(training_cfg.get("device", "auto"))
    model_cfg = adapt_model_config_to_channelthermal_dataset(config_payload, NaiveFieldBaselineConfig.from_dict(config_payload.get("model", {})))
    run_dir = Path(output_dir) if use_output_dir_as_run_dir and output_dir is not None else resolve_naive_run_dir(config_payload, run_name, output_dir)
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

    model = NaiveFieldBaseline(model_cfg).to(device)
    initialize_model(model, train_loader, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-5)),
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
        checkpoint = torch.load(resolve_path(resume), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_val_field_mse_physical = float(checkpoint.get("best_val_field_mse_physical", float("inf")))
        best_val_temperature_mse = float(checkpoint.get("best_val_temperature_mse", float("inf")))

    config_resolved = dict(config_payload)
    config_resolved["model"] = model_cfg.to_dict()
    config_resolved.setdefault("training", {}).update({"target_stats": stats_to_json(target_stats), "run_dir": str(run_dir)})
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

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics, train_steps = run_epoch(
                model,
                train_loader,
                device,
                training_cfg,
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
                    save_temperature_visualizations(last_val_output, run_dir, target_stats)

            row = {
                "epoch": epoch,
                "train_steps": total_steps,
                "epoch_runtime_sec": time.perf_counter() - epoch_start,
                **prefix_keys(train_metrics, "train_"),
                **val_metrics,
            }
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
    finally:
        train_dataset.close()
        val_dataset.close()

    if not latest_path.exists():
        checkpoint = make_checkpoint(
            max(start_epoch - 1, 0),
            model,
            optimizer,
            config_resolved,
            model_cfg,
            target_stats,
            best_val_loss,
            {},
            best_val_field_mse_physical=best_val_field_mse_physical,
            best_val_temperature_mse=best_val_temperature_mse,
        )
        torch.save(checkpoint, latest_path)

    summary = {
        "run_dir": str(run_dir),
        "model_type": model_cfg.model_type,
        "best_val_loss": best_val_loss,
        "best_val_field_mse_physical": best_val_field_mse_physical,
        "best_val_temperature_mse": best_val_temperature_mse,
        "best_metrics": best_by_loss_metrics or (history[-1] if history else {}),
        "best_by_loss_metrics": best_by_loss_metrics or (history[-1] if history else {}),
        "best_by_field_mse_metrics": best_by_field_mse_metrics or (history[-1] if history else {}),
        "best_by_temperature_mse_metrics": best_by_temperature_mse_metrics or (history[-1] if history else {}),
        "latest_epoch": history[-1]["epoch"] if history else 0,
        "latest_metrics": history[-1] if history else {},
        "no_hypergraph_baseline": True,
        "loss_semantics": {
            "train_loss": "normalized weighted MSE plus org_reg_loss; org_reg_loss is always 0 for naive baselines.",
            "val_loss": "normalized weighted MSE, matching train_unified.py validation loss.",
            "val_field_mse_physical": "unweighted physical-space MSE after target denormalization.",
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return summary


def run_dry_run(config_payload: Dict[str, Any], *, inspect_data: bool = False, max_steps: Optional[int] = None) -> Dict[str, Any]:
    training_cfg = config_payload.get("training", {})
    torch.manual_seed(int(training_cfg.get("seed", 0)))
    device = resolve_device(training_cfg.get("device", "auto"))
    model_cfg = adapt_model_config_to_channelthermal_dataset(config_payload, NaiveFieldBaselineConfig.from_dict(config_payload.get("model", {})))
    dataset_path = resolve_dataset_path(config_payload)
    if not dataset_path.exists():
        raise FileNotFoundError(f"ChannelThermal dataset not found: {dataset_path}")
    dataset = ChannelThermalPointDataset(
        dataset_path,
        split="train",
        max_cases=1,
        points_per_case=int(training_cfg.get("points_per_case", 192)),
        max_num_modules=int(model_cfg.max_num_modules),
        field_dim=int(model_cfg.field_dim),
        normalize_targets=False,
        module_heat_feature_mode=str(training_cfg.get("module_heat_feature_mode", "both")),
        seed=int(training_cfg.get("seed", 0)),
    )
    try:
        batch = collate_batchdata([dataset[0]])
    finally:
        dataset.close()
    if inspect_data:
        print(json.dumps({"batch": describe_batch(batch)}, indent=2, sort_keys=True))
    batch = batch.to(device)
    model = NaiveFieldBaseline(model_cfg).to(device)
    model.eval()
    with torch.no_grad():
        output = model(batch)
    loss = weighted_mse_loss(output["pred_field"], batch.target_field, training_cfg)
    metrics = compute_basic_field_metrics(output["pred_field"], batch.target_field)
    metrics.update(compatibility_metrics(output))
    result = {
        "case": batch.case_name,
        "dry_run": True,
        "max_steps": max_steps,
        "pred_field_shape": list(output["pred_field"].shape),
        "target_field_shape": None if batch.target_field is None else list(batch.target_field.shape),
        "loss": float(loss.detach().cpu()),
        "metrics": metrics,
        "model_config": model_cfg.to_dict(),
    }
    print(json.dumps(jsonable(result), indent=2, sort_keys=True))
    return result


def build_channelthermal_datasets(
    config_payload: Dict[str, Any],
    model_cfg: NaiveFieldBaselineConfig,
) -> tuple[ChannelThermalPointDataset, ChannelThermalPointDataset, Dict[str, Any]]:
    data_path = resolve_dataset_path(config_payload)
    if not data_path.exists():
        raise FileNotFoundError(f"ChannelThermal dataset not found: {data_path}")
    training_cfg = config_payload.get("training", {})
    normalize_targets = bool(training_cfg.get("normalize_targets", True))
    heat_feature_mode = str(training_cfg.get("module_heat_feature_mode", "both"))
    stats_probe = ChannelThermalPointDataset(
        data_path,
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
        data_path,
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
        data_path,
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


def run_epoch(
    model: NaiveFieldBaseline,
    loader: DataLoader,
    device: torch.device,
    training_cfg: Dict[str, Any],
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
            field_loss = weighted_mse_loss(output["pred_field"], batch.target_field, training_cfg)
            org_reg_loss = output["pred_field"].new_tensor(0.0)
            loss = field_loss + org_reg_loss
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
        sums["field_loss"] = sums.get("field_loss", 0.0) + float(field_loss.detach().cpu()) * batch_size
        sums["org_reg_loss"] = sums.get("org_reg_loss", 0.0) + float(org_reg_loss.detach().cpu()) * batch_size
        for key, value in compatibility_metrics(output).items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
    return ({key: value / max(count, 1) for key, value in sums.items()}, steps)


def evaluate_loader(
    model: NaiveFieldBaseline,
    loader: DataLoader,
    device: torch.device,
    training_cfg: Dict[str, Any],
    target_stats: Dict[str, Any],
    model_cfg: NaiveFieldBaselineConfig,
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
            field_loss = weighted_mse_loss(output["pred_field"], batch.target_field, training_cfg)
            org_reg_loss = output["pred_field"].new_tensor(0.0)
            loss = field_loss + org_reg_loss
            physical_pred = denormalize_field(output["pred_field"], target_stats)
            physical_target = denormalize_field(batch.target_field, target_stats)
            last_output = dict(output)
            last_output["_query_xy"] = batch.query_xy.detach()
            if physical_pred is not None:
                last_output["_physical_pred_field"] = physical_pred.detach()
            if physical_target is not None:
                last_output["_physical_target_field"] = physical_target.detach()
            metrics = compute_basic_field_metrics(physical_pred, physical_target)
            metrics.update(compute_channelthermal_region_metrics(physical_pred, physical_target, batch, model_cfg))
            metrics.update(compatibility_metrics(output))
            metrics["field_loss"] = float(field_loss.detach().cpu())
            metrics["org_reg_loss"] = float(org_reg_loss.detach().cpu())
            metrics["val_loss"] = float(loss.detach().cpu())
            batch_size = int(batch.query_xy.shape[0])
            count += batch_size
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    sums[key] = sums.get(key, 0.0) + float(value) * batch_size
    averaged = {f"val_{key}" if not str(key).startswith("val_") else str(key): value / max(count, 1) for key, value in sums.items()}
    if "val_field_mse" in averaged:
        averaged["val_field_mse_physical"] = averaged["val_field_mse"]
    return averaged, last_output


def compatibility_metrics(output: Dict[str, Any]) -> Dict[str, float]:
    metrics = {
        "uses_hyper_context": 0.0,
        "active_edge_count": float("nan"),
        "A_mh_entropy": float("nan"),
        "A_eh_entropy": float("nan"),
        "org_reg_loss": 0.0,
    }
    value = output.get("pooled_module_summary_norm")
    if torch.is_tensor(value):
        metrics["pooled_module_summary_norm"] = float(value.detach().cpu())
    elif isinstance(value, (int, float)):
        metrics["pooled_module_summary_norm"] = float(value)
    return metrics


def initialize_model(model: NaiveFieldBaseline, loader: DataLoader, device: torch.device) -> None:
    sample = next(iter(loader)).to(device)
    model.eval()
    with torch.no_grad():
        model(sample)


def make_checkpoint(
    epoch: int,
    model: NaiveFieldBaseline,
    optimizer: torch.optim.Optimizer,
    config_resolved: Dict[str, Any],
    model_cfg: NaiveFieldBaselineConfig,
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
        "baseline_family": "naive_no_hypergraph",
    }


def adapt_model_config_to_channelthermal_dataset(
    config_payload: Dict[str, Any],
    model_cfg: NaiveFieldBaselineConfig,
) -> NaiveFieldBaselineConfig:
    path = resolve_dataset_path(config_payload)
    try:
        import h5py
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
        if abs(float(model_cfg.domain_length_x) - lx) > 1.0e-5 or abs(float(model_cfg.domain_length_y) - ly) > 1.0e-5:
            print(f"[setup] adapting ChannelThermal domain from dataset: ({model_cfg.domain_length_x}, {model_cfg.domain_length_y}) -> ({lx}, {ly})")
            return NaiveFieldBaselineConfig.from_dict({**model_cfg.to_dict(), "domain_length_x": lx, "domain_length_y": ly})
    except Exception as exc:
        print(f"[setup] could not infer ChannelThermal domain from dataset ({exc}); using config values.")
    return model_cfg


def resolve_dataset_path(config_payload: Dict[str, Any]) -> Path:
    raw = config_payload.get("data", {}).get(
        "channelthermal_dataset_path",
        "../1_Demo_ChannelThermal/Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5",
    )
    return resolve_path(raw)


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return (SANDBOX_ROOT / value).resolve()


def resolve_naive_run_dir(config_payload: Dict[str, Any], run_name: Optional[str], output_dir: Optional[str | Path]) -> Path:
    root = Path(output_dir) if output_dir is not None else resolve_path(config_payload.get("output", {}).get("saved_root", "results/naive_baselines"))
    if root.name.startswith("Run_") and re.match(r"^Run_\d{4}_\d{8}_\d{6}", root.name):
        return root
    root.mkdir(parents=True, exist_ok=True)
    run_id = next_run_id(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = sanitize_run_label(run_name)
    return root / (f"Run_{run_id:04d}_{timestamp}" + (f"_{suffix}" if suffix else ""))


def next_run_id(root: Path) -> int:
    max_id = 0
    pattern = re.compile(r"^Run_(\d{4})_\d{8}_\d{6}")
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir():
            match = pattern.match(child.name)
            if match:
                max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def sanitize_run_label(label: Optional[str]) -> str:
    if not label:
        return ""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
    return text.strip("_")[:80]


def load_json(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = SANDBOX_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/naive_baseline_channelthermal_template.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None, help="Override training.device, for example cpu, cuda, or cuda:0.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
