"""Evaluation entry point for the unified forward-model sandbox."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from case_adapters import describe_batch
from channelthermal_dataset import ChannelThermalPointDataset, collate_batchdata
from diagnostics import (
    compute_basic_field_metrics,
    compute_hypergraph_diagnostics,
    plot_organization_overview,
    save_diagnostics_json,
)
from train_unified import (
    SANDBOX_ROOT,
    denormalize_field,
    load_batch,
    load_json,
    resolve_device,
    stats_from_json,
)
from unified_model_core import UnifiedHypergraphNeuralField
from unified_types import UnifiedForwardConfig


def main() -> None:
    args = parse_args()
    if args.checkpoint or not args.dry_run:
        if not args.checkpoint:
            args.checkpoint = str(find_latest_checkpoint(args.runs_root, use_latest=bool(args.latest)))
            print(f"[eval] using latest checkpoint: {args.checkpoint}")
        evaluate_checkpoint(args)
        return
    run_dry_eval(args)


def run_dry_eval(args: argparse.Namespace) -> None:
    payload = load_json(args.config)
    training_cfg = payload.get("training", {})
    torch.manual_seed(int(training_cfg.get("seed", 0)))
    device = resolve_device(training_cfg.get("device", "auto"))
    config = UnifiedForwardConfig.from_dict(payload.get("model", {}))

    batch = load_batch(args.case, payload, batch_size=int(training_cfg.get("batch_size", 1)))
    if args.inspect_data:
        print(json.dumps({"batch": describe_batch(batch)}, indent=2, sort_keys=True))
    batch = batch.to(device)
    model = UnifiedHypergraphNeuralField(config).to(device)
    model.eval()
    with torch.no_grad():
        output = model(batch)
    if args.inspect_data:
        pred_shape = list(output["pred_field"].shape)
        print(json.dumps({"pred_field_shape": pred_shape}, indent=2, sort_keys=True))
        if batch.target_field is not None and output["pred_field"].shape != batch.target_field.shape:
            print(
                "[warning] pred_field shape does not match target_field shape; "
                f"skipping MSE. pred={pred_shape} target={list(batch.target_field.shape)}"
            )

    metrics = compute_basic_field_metrics(output["pred_field"], batch.target_field)
    metrics.update(compute_hypergraph_diagnostics(output))
    metrics.update({"case": batch.case_name, "dry_run": bool(args.dry_run)})
    out_dir = SANDBOX_ROOT / "results" / "dry_run"
    save_diagnostics_json({"metrics": metrics, "batch": batch.to_dict(), "config": config.to_dict()}, out_dir / "eval_metrics.json")
    plot_organization_overview(output, out_dir / "eval_organization.png")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def evaluate_checkpoint(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else make_evaluation_dir(checkpoint_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    payload = checkpoint.get("config_resolved", {})
    training_cfg = payload.get("training", {})
    device = resolve_device(training_cfg.get("device", "auto"))
    model_cfg = UnifiedForwardConfig.from_dict(checkpoint.get("model_config", payload.get("model", {})))
    target_stats = stats_from_json(checkpoint["target_stats"])

    model = UnifiedHypergraphNeuralField(model_cfg).to(device)
    dataset = ChannelThermalPointDataset(
        payload.get("data", {}).get("channelthermal_dataset_path", "../1_Demo_ChannelThermal/Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"),
        split="test",
        max_cases=training_cfg.get("max_val_cases"),
        points_per_case=int(training_cfg.get("val_points_per_case", training_cfg.get("points_per_case", 1024))),
        max_num_modules=int(model_cfg.max_num_modules),
        field_dim=int(model_cfg.field_dim),
        normalize_targets=bool(target_stats.get("normalize_targets", False)),
        target_mean=target_stats["mean"],
        target_std=target_stats["std"],
        seed=int(training_cfg.get("seed", 0)) + 999,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("val_batch_size", training_cfg.get("batch_size", 16))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batchdata,
    )
    initialize_and_load(model, checkpoint, loader, device)
    metrics, last_output = evaluate_model_batches(model, loader, device, target_stats, num_batches=args.num_batches)
    if last_output is not None:
        plot_organization_overview(last_output, output_dir / "eval_organization.png")
    write_metrics_csv(output_dir / "eval_metrics.csv", metrics)
    save_diagnostics_json({"metrics": metrics, "checkpoint": str(checkpoint_path)}, output_dir / "eval_summary.json")
    dataset.close()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def initialize_and_load(
    model: UnifiedHypergraphNeuralField,
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


def evaluate_model_batches(
    model: UnifiedHypergraphNeuralField,
    loader: DataLoader,
    device: torch.device,
    target_stats: Dict[str, Any],
    *,
    num_batches: Optional[int] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    model.eval()
    sums: Dict[str, float] = {}
    count = 0
    last_output: Optional[Dict[str, Any]] = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if num_batches is not None and batch_idx >= int(num_batches):
                break
            batch = batch.to(device)
            output = model(batch)
            last_output = output
            pred = denormalize_field(output["pred_field"], target_stats)
            target = denormalize_field(batch.target_field, target_stats)
            metrics = compute_basic_field_metrics(pred, target)
            metrics.update(compute_hypergraph_diagnostics(output))
            batch_size = int(batch.query_xy.shape[0])
            count += batch_size
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    sums[key] = sums.get(key, 0.0) + float(value) * batch_size
    return ({key: value / max(count, 1) for key, value in sums.items()}, last_output)


def write_metrics_csv(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified_forward_config_template.json")
    parser.add_argument("--case", choices=["channelthermal", "multicylinder", "synthetic"], default="synthetic")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--runs-root", default=str(SANDBOX_ROOT / "results" / "runs"))
    parser.add_argument("--latest", action="store_true", help="Use latest_model.pt instead of best_model.pt when auto-selecting.")
    return parser.parse_args()


def find_latest_checkpoint(runs_root: str | Path, *, use_latest: bool = False) -> Path:
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(f"Runs root not found: {root}")
    candidates = []
    pattern = re.compile(r"^Run_\d{4}_\d{8}_\d{6}")
    checkpoint_name = "latest_model.pt" if use_latest else "best_model.pt"
    for run_dir in root.rglob("Run_*"):
        if not run_dir.is_dir() or not pattern.match(run_dir.name):
            continue
        checkpoint = run_dir / checkpoint_name
        if checkpoint.exists():
            candidates.append((run_dir.stat().st_mtime, run_dir.name, checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No {checkpoint_name} found under {root} in Run_####_YYYYMMDD_HHMMSS directories.")
    candidates.sort()
    return candidates[-1][2]


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


if __name__ == "__main__":
    main()
