"""Evaluate a trained external naive ChannelThermal baseline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-unified-forward")

from channelthermal_dataset import ChannelThermalPointDataset, collate_batchdata
from naive_field_baselines import NaiveFieldBaseline, NaiveFieldBaselineConfig
from train_naive_baseline import compatibility_metrics, load_json, resolve_dataset_path
from train_unified import SANDBOX_ROOT, denormalize_field, jsonable, resolve_device, stats_from_json, write_json
from train_unified import weighted_mse_loss
from diagnostics import compute_basic_field_metrics, compute_channelthermal_region_metrics


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(args)


def evaluate_checkpoint(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_payload = load_json(args.config) if args.config else {}
    payload = checkpoint.get("config_resolved", config_payload)
    training_cfg = payload.get("training", {})
    if args.device is not None:
        training_cfg = {**training_cfg, "device": args.device}
    device = resolve_device(training_cfg.get("device", "auto"))
    model_cfg = NaiveFieldBaselineConfig.from_dict(checkpoint.get("model_config", payload.get("model", {})))
    target_stats = stats_from_json(checkpoint["target_stats"])
    target_stats["dataset_heat_scale"] = float(checkpoint.get("target_stats", {}).get("dataset_heat_scale", 1.0))
    target_stats["module_heat_feature_mode"] = str(checkpoint.get("target_stats", {}).get("module_heat_feature_mode", training_cfg.get("module_heat_feature_mode", "both")))
    output_dir = Path(args.output_dir) if args.output_dir else make_evaluation_dir(checkpoint_path.parent)
    if not output_dir.is_absolute():
        output_dir = (SANDBOX_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = NaiveFieldBaseline(model_cfg).to(device)
    dataset = ChannelThermalPointDataset(
        resolve_dataset_path(payload),
        split=args.split,
        max_cases=args.max_cases if args.max_cases is not None else training_cfg.get("max_val_cases"),
        points_per_case=int(args.points_per_case or training_cfg.get("val_points_per_case", training_cfg.get("points_per_case", 1024))),
        max_num_modules=int(model_cfg.max_num_modules),
        field_dim=int(model_cfg.field_dim),
        normalize_targets=bool(target_stats.get("normalize_targets", False)),
        target_mean=target_stats["mean"],
        target_std=target_stats["std"],
        module_heat_feature_mode=str(target_stats.get("module_heat_feature_mode", training_cfg.get("module_heat_feature_mode", "both"))),
        dataset_heat_scale=float(target_stats.get("dataset_heat_scale", 1.0)),
        seed=int(training_cfg.get("seed", 0)) + 999,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or training_cfg.get("val_batch_size", training_cfg.get("batch_size", 16))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batchdata,
    )
    try:
        initialize_and_load(model, checkpoint, loader, device)
        metrics, last_output = evaluate_loader(
            model,
            loader,
            device,
            training_cfg,
            target_stats,
            model_cfg,
            max_batches=args.num_batches,
        )
        if last_output is not None:
            save_eval_temperature_visualizations(last_output, output_dir)
        write_metrics_csv(output_dir / "eval_metrics.csv", metrics)
        summary = {
            "checkpoint": str(checkpoint_path),
            "model_type": model_cfg.model_type,
            "metrics": metrics,
            "outputs": {
                "eval_metrics_csv": str(output_dir / "eval_metrics.csv"),
                "eval_temperature_target": str(output_dir / "eval_temperature_target.png"),
                "eval_temperature_pred": str(output_dir / "eval_temperature_pred.png"),
                "eval_temperature_error": str(output_dir / "eval_temperature_error.png"),
            },
            "no_hypergraph_baseline": True,
        }
        write_json(output_dir / "eval_summary.json", summary)
        print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
        return summary
    finally:
        dataset.close()


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
            metrics.update(compute_channelthermal_region_metrics(physical_pred, physical_target, batch, model_cfg))
            metrics.update(compatibility_metrics(output))
            metrics["eval_loss"] = float(loss.detach().cpu())
            batch_size = int(batch.query_xy.shape[0])
            count += batch_size
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    sums[key] = sums.get(key, 0.0) + float(value) * batch_size
    averaged = {f"eval_{key}" if not str(key).startswith("eval_") else str(key): value / max(count, 1) for key, value in sums.items()}
    if "eval_field_mse" in averaged:
        averaged["eval_field_mse_physical"] = averaged["eval_field_mse"]
    return averaged, last_output


def initialize_and_load(
    model: NaiveFieldBaseline,
    checkpoint: Dict[str, Any],
    loader: DataLoader,
    device: torch.device,
) -> None:
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        sample = next(iter(loader)).to(device)
        with torch.no_grad():
            model(sample)
        model.load_state_dict(checkpoint["model_state_dict"])


def save_eval_temperature_visualizations(output: Dict[str, Any], output_dir: Path) -> None:
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
        ("eval_temperature_target.png", target_temp, "Temperature Target"),
        ("eval_temperature_pred.png", pred_temp, "Temperature Prediction"),
        ("eval_temperature_error.png", error_temp, "Temperature Absolute Error"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=8, cmap="inferno")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)


def write_metrics_csv(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    cwd_candidate = (Path.cwd() / value).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (SANDBOX_ROOT / value).resolve()


def make_evaluation_dir(run_dir: Path) -> Path:
    eval_root = run_dir / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)
    run_id = next_eval_id(eval_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return eval_root / f"Run_{run_id:04d}_{timestamp}"


def next_eval_id(root: Path) -> int:
    max_id = 0
    pattern = re.compile(r"^Run_(\d{4})_\d{8}_\d{6}")
    for child in root.iterdir():
        if child.is_dir():
            match = pattern.match(child.name)
            if match:
                max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/naive_baseline_channelthermal_template.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--points-per-case", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
