#!/usr/bin/env python3
"""Compare dense decoding with evaluation-only retained-mass gathering.

The checkpoint and its state dict remain unchanged.  The script prepares each
physical case once, decodes identical query chunks through the dense reference
and gathered paths, and restores the checkpoint-owned routing configuration
after every comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-honf-pruning-diagnostics")

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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--query-mass-floor", type=float, default=0.98)
    parser.add_argument("--module-mass-floor", type=float, default=0.95)
    parser.add_argument("--minimum-query-routes", type=int, default=1)
    parser.add_argument("--minimum-module-routes", type=int, default=1)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    add_frozen_override_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "diagnostics" / "retained_mass_pruning",
    )
    return parser.parse_args()


def parse_specs(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        label, value = item.split("=", 1)
        result[label.strip()] = Path(value).expanduser().resolve()
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def routing_configs(model: Any) -> list[Any]:
    candidates = [
        model.config.core_honf,
        model.core.config,
        model.core.decoder.config,
        model.core.decoder.pairwise_kernel.config,
        model.core.organizer.config,
        getattr(getattr(model.core.organizer, "exchangeable", None), "config", None),
    ]
    result: list[Any] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            result.append(candidate)
    return result


def routing_state(model: Any) -> dict[str, Any]:
    cfg = model.config.core_honf
    return {
        key: getattr(cfg, key)
        for key in (
            "routing_execution", "query_edge_limit", "query_module_limit",
            "query_edge_retained_mass_floor", "module_incidence_retained_mass_floor",
        )
    }


def set_routing_state(model: Any, values: dict[str, Any]) -> None:
    for config in routing_configs(model):
        for key, value in values.items():
            setattr(config, key, value)


def region_masks(sample: dict[str, Any], near_width: float = 0.25, far_distance: float = 1.0) -> dict[str, np.ndarray]:
    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    _, fluid = module_and_fluid_masks(sample)
    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"], dtype=np.float32) > 0.5
    radius = module_radius_from_sample(sample)
    if np.any(present):
        distances = [np.hypot(x_grid - cx, y_grid - cy) - radius for cx, cy in centers[present]]
        surface = np.min(np.stack(distances), axis=0)
    else:
        surface = np.full(x_grid.shape, np.inf)
    fluid = np.asarray(fluid, dtype=bool)
    return {
        "whole": np.ones(x_grid.shape, dtype=bool),
        "fluid": fluid,
        "near_interface_fluid": fluid & (surface >= 0.0) & (surface <= near_width),
        "far_field_fluid": fluid & (surface >= far_distance),
    }


def prepare_case(model: Any, sample: dict[str, Any], device: torch.device) -> Any:
    query = np.stack(
        (np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)),
        axis=-1,
    ).astype(np.float32)[:1]
    batch = make_batch(sample, query, device)
    with torch.inference_mode():
        output = model(
            batch["structure"], batch["query_xy"],
            interface_condition=batch.get("interface_condition"),
            local_module_params=batch.get("local_module_params"),
            teacher_port_tokens=batch.get("teacher_port_tokens"),
            local_query_points=batch.get("module_internal_query_points"),
            local_port_condition_mode="predicted",
            return_prepared_state=True,
        )
    return output["prepared_state"]


def tensor_scalar(output: dict[str, Any], key: str) -> float:
    value = output.get(key)
    if not torch.is_tensor(value):
        return 0.0
    return float(value.detach().cpu().reshape(-1)[0])


def decode_case(
    model: Any,
    prepared: Any,
    sample: dict[str, Any],
    device: torch.device,
    query_batch_size: int,
    mode: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    set_routing_state(model, state)
    reference = prepared.organizer["hyper_state"]
    prepared.organizer["routing_execution_gathered"] = reference.new_tensor(
        float(mode == "gathered")
    )
    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    queries = np.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
    predictions: list[np.ndarray] = []
    query_retention: list[np.ndarray] = []
    module_retention: list[np.ndarray] = []
    query_routes = 0.0
    dense_query_routes = 0.0
    module_routes = 0.0
    dense_module_routes = 0.0
    with torch.inference_mode():
        for start in range(0, queries.shape[0], query_batch_size):
            query = torch.from_numpy(queries[start : start + query_batch_size]).unsqueeze(0).to(device=device)
            output = model.decode_prepared(prepared, query, return_routing_maps=True)
            predictions.append(output["pred_field"][0].detach().cpu().numpy())
            retained_query = output["query_edge_retained_probability_mass"][0].detach().cpu().numpy()
            query_retention.append(retained_query.reshape(-1))
            retained_module = output["retained_module_incidence_mass"][0].detach().cpu().numpy()
            routed = output["routed_query_edge_pair_mask"][0].detach().cpu().numpy().astype(bool)
            module_retention.append(retained_module[routed])
            attention = output["query_hyper_attention"]
            query_routes += float((attention > 0).sum().detach().cpu())
            active_edge_mask = prepared.organizer.get(
                "effective_edge_mask", prepared.organizer.get("edge_active_mask")
            )
            active_edges = (
                float((active_edge_mask > 0).sum().detach().cpu())
                if torch.is_tensor(active_edge_mask)
                else float(attention.shape[0] * attention.shape[2])
            )
            dense_query_routes += float(attention.shape[1]) * active_edges
            module_present = prepared.organizer.get("module_present")
            active_modules = (
                float((module_present > 0).sum().detach().cpu())
                if torch.is_tensor(module_present)
                else tensor_scalar(output, "pairwise_available_modules")
            )
            dense_module_count = float(attention.shape[1]) * active_modules
            if mode == "dense":
                module_routes += dense_module_count
            else:
                module_routes += tensor_scalar(output, "pairwise_gathered_route_count")
            dense_module_routes += dense_module_count
    return {
        "prediction": np.concatenate(predictions).reshape(*x_grid.shape, -1),
        "query_retention": np.concatenate(query_retention),
        "module_retention": np.concatenate(module_retention) if module_retention else np.zeros((0,)),
        "query_routes": query_routes,
        "dense_query_routes": dense_query_routes,
        "module_routes": module_routes,
        "dense_module_routes": dense_module_routes,
    }


def benchmark(
    model: Any,
    prepared: Any,
    sample: dict[str, Any],
    device: torch.device,
    states: dict[str, dict[str, Any]],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    queries = np.stack(
        (np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)),
        axis=-1,
    ).astype(np.float32)
    query = torch.from_numpy(queries).unsqueeze(0).to(device=device)
    result: dict[str, Any] = {}
    for label, state in states.items():
        set_routing_state(model, state)
        reference = prepared.organizer["hyper_state"]
        prepared.organizer["routing_execution_gathered"] = reference.new_tensor(
            float(state["routing_execution"] == "gathered")
        )
        for _ in range(warmup):
            with torch.inference_mode():
                model.decode_prepared(prepared, query)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        elapsed: list[float] = []
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                model.decode_prepared(prepared, query)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - started)
        result[label] = {
            "median_seconds": float(statistics.median(elapsed)),
            "mean_seconds": float(statistics.mean(elapsed)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        }
    return result


def distribution(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)) if arr.size else 0.0,
        "p05": float(np.quantile(arr, 0.05)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(label: str, path: Path, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, checkpoint = load_model(path, device)
    frozen_overrides = apply_label_frozen_overrides(model, label, args)
    if model.config.core_honf.field_assembly_mode != "edge_additive":
        raise ValueError(f"{label}: gathered retained-mass comparison requires edge_additive mode")
    original = routing_state(model)
    dense_state = {**original, "routing_execution": "dense", "query_edge_limit": 0, "query_module_limit": 0}
    pruned_state = {
        **original,
        "routing_execution": "gathered",
        "query_edge_limit": int(args.minimum_query_routes),
        "query_module_limit": int(args.minimum_module_routes),
        "query_edge_retained_mass_floor": float(args.query_mass_floor),
        "module_incidence_retained_mass_floor": float(args.module_mass_floor),
    }
    full_state = {
        **original,
        "routing_execution": "gathered",
        "query_edge_limit": max(int(model.config.core_honf.num_hyperedges), int(model.config.core_honf.edge_capacity)),
        "query_module_limit": 10_000,
        "query_edge_retained_mass_floor": 0.0,
        "module_incidence_retained_mass_floor": 0.0,
    }
    dataset_cfg = checkpoint.get("train_config", {}).get("dataset", {})
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint.get("global_normalization_stats", {}).items()}
    dataset = GlobalChannelThermalDataset(
        dataset_cfg["packed_h5_path"], split=args.split, points_per_case=1,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False, include_grid=True, include_structure_targets=False,
        normalizer=H5Normalizer(stats),
    )
    count = len(dataset) if args.max_cases is None else min(len(dataset), int(args.max_cases))
    field_names = list(model.config.channelthermal.field_names)
    rows: list[dict[str, Any]] = []
    query_values: list[np.ndarray] = []
    module_values: list[np.ndarray] = []
    runtime = None
    state_keys_before = tuple(model.state_dict())
    for index in range(count):
        sample = dataset[index]
        prepared = prepare_case(model, sample, device)
        dense = decode_case(model, prepared, sample, device, args.query_batch_size, "dense", dense_state)
        pruned = decode_case(model, prepared, sample, device, args.query_batch_size, "gathered", pruned_state)
        full = decode_case(model, prepared, sample, device, args.query_batch_size, "gathered", full_state)
        query_values.append(pruned["query_retention"])
        module_values.append(pruned["module_retention"])
        target = np.asarray(sample["steady_field"], dtype=np.float64)
        if bool(dataset_cfg.get("normalize_targets", False)):
            target = dataset.normalizer.normalize_fields(target)
        masks = region_masks(sample)
        for region, mask in masks.items():
            channel_specs = [(index, name) for index, name in enumerate(field_names)] + [(None, "__all__")]
            for channel_index, channel in channel_specs:
                if channel_index is None:
                    dense_values = dense["prediction"][mask]
                    pruned_values = pruned["prediction"][mask]
                    target_values = target[mask]
                else:
                    dense_values = dense["prediction"][..., channel_index][mask]
                    pruned_values = pruned["prediction"][..., channel_index][mask]
                    target_values = target[..., channel_index][mask]
                dense_error = float(np.mean((dense_values - target_values) ** 2))
                pruned_error = float(np.mean((pruned_values - target_values) ** 2))
                rows.append({
                    "checkpoint": label, "case_id": str(sample["case_id"]), "region": region, "channel": channel,
                    "dense_mse": dense_error, "pruned_mse": pruned_error,
                    "relative_mse_degradation": (pruned_error - dense_error) / max(dense_error, 1.0e-15),
                    "dense_vs_pruned_max_abs": float(np.max(np.abs(dense["prediction"] - pruned["prediction"]))),
                    "dense_vs_full_limit_gathered_max_abs": float(np.max(np.abs(dense["prediction"] - full["prediction"]))),
                    "query_route_reduction": 1.0 - pruned["query_routes"] / max(pruned["dense_query_routes"], 1.0),
                    "module_route_reduction": 1.0 - pruned["module_routes"] / max(pruned["dense_module_routes"], 1.0),
                })
        if index == 0:
            runtime = benchmark(
                model, prepared, sample, device,
                {"dense": dense_state, "retained_mass_pruned": pruned_state},
                args.benchmark_warmup, args.benchmark_iterations,
            )
        print(f"[{label}] {index + 1}/{count} case={sample['case_id']}", flush=True)
    set_routing_state(model, original)
    state_keys_after = tuple(model.state_dict())
    if state_keys_before != state_keys_after:
        raise RuntimeError("Evaluation-only routing override changed state_dict structure")
    query_stats = distribution(np.concatenate(query_values))
    module_stats = distribution(np.concatenate(module_values))
    info = {
        "checkpoint": str(path), "sha256": sha256_file(path),
        "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "field_names": field_names, "evaluated_case_count": count,
        "dense_state": dense_state, "pruned_state": pruned_state,
        "query_retained_mass": query_stats, "routed_module_retained_mass": module_stats,
        "runtime": runtime, "state_dict_key_count": len(state_keys_before),
        "state_dict_structure_unchanged": state_keys_before == state_keys_after,
        "frozen_overrides": frozen_overrides,
    }
    dataset.close()
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info, rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups = sorted({(row["checkpoint"], row["region"], row["channel"]) for row in rows})
    for checkpoint, region, channel in groups:
        members = [row for row in rows if (row["checkpoint"], row["region"], row["channel"]) == (checkpoint, region, channel)]
        dense = np.asarray([row["dense_mse"] for row in members])
        pruned = np.asarray([row["pruned_mse"] for row in members])
        result[f"{checkpoint}/{region}/{channel}"] = {
            "dense_mse_mean": float(dense.mean()),
            "pruned_mse_mean": float(pruned.mean()),
            "pooled_relative_mse_degradation": float((pruned.mean() - dense.mean()) / max(float(dense.mean()), 1.0e-15)),
            "max_no_prune_output_difference": float(max(row["dense_vs_full_limit_gathered_max_abs"] for row in members)),
            "query_route_reduction_mean": float(np.mean([row["query_route_reduction"] for row in members])),
            "module_route_reduction_mean": float(np.mean([row["module_route_reduction"] for row in members])),
        }
    return result


def main() -> int:
    args = parse_args()
    specs = parse_specs(args.checkpoint)
    resolve_frozen_overrides(args, specs)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoints: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for label, path in specs.items():
        if not path.exists():
            raise FileNotFoundError(path)
        info, current = evaluate_checkpoint(label, path, args, device)
        checkpoints[label] = info
        rows.extend(current)
    csv_path = args.output_dir / "retained_mass_pruning_per_case_channel_region.csv"
    json_path = args.output_dir / "retained_mass_pruning_summary.json"
    write_csv(csv_path, rows)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "device": str(device), "split": args.split,
        "checkpoints": checkpoints, "summary": summarize(rows),
        "promotion_gates": {
            "aggregate_mse_degradation_max": 0.02,
            "channel_mse_degradation_max": 0.05,
            "query_retained_mass_p05_min": 0.98,
            "module_retained_mass_p05_min": 0.95,
            "route_reduction_min": 0.20,
        },
        "artifacts": {"per_case_csv": str(csv_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
