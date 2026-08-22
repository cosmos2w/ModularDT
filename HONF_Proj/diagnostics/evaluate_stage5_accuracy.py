#!/usr/bin/env python3
"""Evaluate frozen HONF checkpoints with common spatial/channel denominators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "diagnostics"))

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer  # noqa: E402
from channelthermal.evaluation_tools.plots import module_and_fluid_masks, module_radius_from_sample  # noqa: E402
from channelthermal.workflows.evaluate_forward import load_model, make_batch  # noqa: E402
from frozen_override_cli import (  # noqa: E402
    add_frozen_override_arguments,
    apply_label_frozen_overrides,
    resolve_frozen_overrides,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument(
        "--edge-capacity-override",
        action="append",
        default=[],
        metavar="LABEL=CAPACITY",
        help="Evaluation-only runtime capacity override for exchangeable checkpoints.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--query-batch-size", type=int, default=65536)
    parser.add_argument("--max-cases", type=int, default=None)
    add_frozen_override_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "diagnostics" / "stage5_accuracy",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def region_masks(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    _, fluid = module_and_fluid_masks(sample)
    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"], dtype=np.float32) > 0.5
    radius = module_radius_from_sample(sample)
    if np.any(present):
        surface = np.min(
            np.stack(
                [np.hypot(x_grid - cx, y_grid - cy) - radius for cx, cy in centers[present]]
            ),
            axis=0,
        )
    else:
        surface = np.full(x_grid.shape, np.inf, dtype=np.float32)
    fluid = np.asarray(fluid, dtype=bool)
    return {
        "whole": np.ones(x_grid.shape, dtype=bool),
        "fluid": fluid,
        "near_interface_fluid": fluid & (surface >= 0.0) & (surface <= 0.25),
        "far_field_fluid": fluid & (surface >= 1.0),
    }


def predict_case(
    model: Any,
    sample: dict[str, Any],
    device: torch.device,
    query_batch_size: int,
) -> np.ndarray:
    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    queries = np.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
    prepared = None
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(queries), query_batch_size):
            query = queries[start : start + query_batch_size]
            if prepared is None:
                batch = make_batch(sample, query, device)
                output = model(
                    batch["structure"],
                    batch["query_xy"],
                    interface_condition=batch.get("interface_condition"),
                    local_module_params=batch.get("local_module_params"),
                    teacher_port_tokens=batch.get("teacher_port_tokens"),
                    local_query_points=batch.get("module_internal_query_points"),
                    local_port_condition_mode="predicted",
                    return_prepared_state=True,
                )
                prepared = output.pop("prepared_state")
            else:
                tensor = torch.from_numpy(query).unsqueeze(0).to(device=device)
                output = model.decode_prepared(prepared, tensor)
            chunks.append(output["pred_field"][0].detach().cpu().numpy())
    return np.concatenate(chunks).reshape(*x_grid.shape, -1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
    label: str,
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, checkpoint = load_model(path, device)
    frozen_overrides = apply_label_frozen_overrides(model, label, args)
    runtime_capacity = args.capacity_overrides.get(label)
    if runtime_capacity is not None:
        model.set_edge_capacity(runtime_capacity)
    model.eval()
    dataset_cfg = checkpoint["train_config"]["dataset"]
    stats = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    normalizer = H5Normalizer(stats)
    dataset = GlobalChannelThermalDataset(
        dataset_cfg["packed_h5_path"],
        split=args.split,
        points_per_case=1,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=False,
        normalizer=normalizer,
    )
    count = len(dataset) if args.max_cases is None else min(len(dataset), int(args.max_cases))
    field_names = list(model.config.channelthermal.field_names)
    if field_names != list(dataset.channel_order):
        raise RuntimeError(f"{label}: checkpoint and dataset channel orders differ")
    rows: list[dict[str, Any]] = []
    for index in range(count):
        sample = dataset[index]
        prediction_normalized = predict_case(model, sample, device, int(args.query_batch_size))
        target_physical = np.asarray(sample["steady_field"], dtype=np.float32)
        target_normalized = (
            normalizer.normalize_fields(target_physical)
            if bool(dataset_cfg.get("normalize_targets", False))
            else target_physical
        )
        prediction_physical = (
            normalizer.denormalize_fields(prediction_normalized)
            if bool(dataset_cfg.get("normalize_targets", False))
            else prediction_normalized
        )
        for region, mask in region_masks(sample).items():
            channels = [(field_index, name) for field_index, name in enumerate(field_names)]
            channels.append((None, "__all__"))
            for field_index, field_name in channels:
                if field_index is None:
                    pred_n = prediction_normalized[mask]
                    target_n = target_normalized[mask]
                    pred_p = prediction_physical[mask]
                    target_p = target_physical[mask]
                else:
                    pred_n = prediction_normalized[..., field_index][mask]
                    target_n = target_normalized[..., field_index][mask]
                    pred_p = prediction_physical[..., field_index][mask]
                    target_p = target_physical[..., field_index][mask]
                normalized_error = np.asarray(pred_n - target_n, dtype=np.float64)
                physical_error = np.asarray(pred_p - target_p, dtype=np.float64)
                rows.append(
                    {
                        "checkpoint": label,
                        "case_id": str(sample["case_id"]),
                        "region": region,
                        "channel": field_name,
                        "value_count": int(normalized_error.size),
                        "normalized_sse": float(np.sum(normalized_error * normalized_error)),
                        "normalized_target_ss": float(
                            np.sum(np.asarray(target_n, dtype=np.float64) ** 2)
                        ),
                        "normalized_mse": float(np.mean(normalized_error * normalized_error)),
                        "physical_sse": float(np.sum(physical_error * physical_error)),
                        "physical_target_ss": float(
                            np.sum(np.asarray(target_p, dtype=np.float64) ** 2)
                        ),
                        "physical_mse": float(np.mean(physical_error * physical_error)),
                    }
                )
        print(f"[{label}] {index + 1}/{count} case={sample['case_id']}", flush=True)
    core_cfg = checkpoint.get("model_config", {}).get("core_honf", {})
    info = {
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "organizer_mode": core_cfg.get("organizer_mode"),
        "field_assembly_mode": core_cfg.get("field_assembly_mode"),
        "runtime_edge_capacity": runtime_capacity,
        "frozen_overrides": frozen_overrides,
        "field_names": field_names,
        "dataset_fingerprint": checkpoint.get(
            "dataset_fingerprint", dataset_cfg.get("dataset_fingerprint")
        ),
        "evaluated_case_count": count,
    }
    dataset.close()
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info, rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], row["region"], row["channel"])].append(row)
    result: dict[str, Any] = {}
    for group, members in sorted(groups.items()):
        label, region, channel = group
        count = sum(int(member["value_count"]) for member in members)
        normalized_sse = sum(float(member["normalized_sse"]) for member in members)
        physical_sse = sum(float(member["physical_sse"]) for member in members)
        normalized_target_ss = sum(
            float(member["normalized_target_ss"]) for member in members
        )
        physical_target_ss = sum(float(member["physical_target_ss"]) for member in members)
        case_values = np.asarray([float(member["normalized_mse"]) for member in members])
        result[f"{label}/{region}/{channel}"] = {
            "pooled_normalized_mse": normalized_sse / max(count, 1),
            "pooled_normalized_relative_l2": float(
                np.sqrt(normalized_sse / max(normalized_target_ss, 1.0e-15))
            ),
            "pooled_physical_mse": physical_sse / max(count, 1),
            "pooled_physical_relative_l2": float(
                np.sqrt(physical_sse / max(physical_target_ss, 1.0e-15))
            ),
            "case_normalized_mse_mean": float(case_values.mean()),
            "case_normalized_mse_median": float(np.median(case_values)),
            "case_normalized_mse_p05": float(np.quantile(case_values, 0.05)),
            "case_normalized_mse_p95": float(np.quantile(case_values, 0.95)),
            "case_count": len(members),
            "value_count": count,
        }
    return result


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoints: dict[str, Path] = {}
    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        checkpoints[label] = Path(raw_path).expanduser().resolve()
    args.capacity_overrides = {}
    for value in args.edge_capacity_override:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=CAPACITY, got {value!r}")
        label, raw_capacity = value.split("=", 1)
        capacity = int(raw_capacity)
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        args.capacity_overrides[label] = capacity
    unknown_overrides = sorted(set(args.capacity_overrides) - set(checkpoints))
    if unknown_overrides:
        raise ValueError(f"Capacity overrides reference unknown checkpoint labels: {unknown_overrides}")
    resolve_frozen_overrides(args, checkpoints)
    provenance: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for label, path in checkpoints.items():
        info, checkpoint_rows = evaluate_checkpoint(label, path, args, device)
        provenance[label] = info
        rows.extend(checkpoint_rows)
    csv_path = args.output_dir / "accuracy_per_case_channel_region.csv"
    summary_path = args.output_dir / "accuracy_summary.json"
    write_csv(csv_path, rows)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "device": str(device),
        "split": args.split,
        "provenance": provenance,
        "summary": summarize(rows),
        "artifacts": {"per_case_csv": str(csv_path)},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
