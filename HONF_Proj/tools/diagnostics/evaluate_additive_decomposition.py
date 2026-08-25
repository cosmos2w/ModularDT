#!/usr/bin/env python3
"""Diagnostics-only decomposition of ChannelThermal edge-additive checkpoints.

This script requests the decoder's existing ``pred_field_background`` and
``pred_field_by_edge`` tensors.  It does not alter model parameters, routing,
selection, schedules, or persisted checkpoint state.  The temporary monkey
patch used by the timing benchmark is process-local, is restored immediately,
and exists only to measure the already-defined edge path without evaluating
the background branch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-honf-additive-diagnostics")

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer  # noqa: E402
from channelthermal.evaluation_tools.plots import module_and_fluid_masks, module_radius_from_sample  # noqa: E402
from channelthermal.workflows.evaluate_forward import load_model, make_batch  # noqa: E402


RUN_ROOT = PROJECT_ROOT / "Trained_Results" / "ThermalChannel" / "HONF_Forward_Runs"
DEFAULT_CHECKPOINTS = {
    "1102_best_field": RUN_ROOT
    / "Run_1102_20260820_002237_adaptive_sparse_additive_formal"
    / "best_by_field_mse_model.pt",
    "1102_latest": RUN_ROOT
    / "Run_1102_20260820_002237_adaptive_sparse_additive_formal"
    / "latest_model.pt",
    "1103_best_field": RUN_ROOT
    / "Run_1103_20260820_095337_adaptive_sparse_additive_LR_Check"
    / "best_by_field_mse_model.pt",
    "1103_latest": RUN_ROOT
    / "Run_1103_20260820_095337_adaptive_sparse_additive_LR_Check"
    / "latest_model.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--near-interface-width", type=float, default=0.25)
    parser.add_argument("--far-surface-distance", type=float, default=1.0)
    parser.add_argument("--benchmark-iterations", type=int, default=30)
    parser.add_argument("--benchmark-warmup", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=None, help="Diagnostics smoke-test limit.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "diagnostics" / "generated" / "additive_decomposition",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Override defaults; may be repeated.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_snapshot(source: Path, destination: Path, retries: int = 5) -> Path:
    """Copy an actively-written checkpoint only when source metadata is stable."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        before = source.stat()
        shutil.copy2(source, destination)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            return destination
        if attempt + 1 < retries:
            time.sleep(0.5)
    raise RuntimeError(f"Checkpoint changed repeatedly while snapshotting: {source}")


def parse_checkpoint_specs(raw: Iterable[str]) -> dict[str, Path]:
    if not raw:
        return {key: path.resolve() for key, path in DEFAULT_CHECKPOINTS.items()}
    result: dict[str, Path] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        label, value = item.split("=", 1)
        result[label.strip()] = Path(value).expanduser().resolve()
    return result


def scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().reshape(-1)[0])
        return value.detach().cpu().numpy()
    return value


def prediction_chunks(
    model: Any,
    sample: dict[str, Any],
    device: torch.device,
    query_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, float]:
    """Return full/background/per-edge normalized fields and prepared state."""

    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    query_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1)
    full_chunks: list[np.ndarray] = []
    background_chunks: list[np.ndarray] = []
    edge_chunks: list[np.ndarray] = []
    prepared = None
    max_closure_error = 0.0
    with torch.inference_mode():
        for start in range(0, query_xy.shape[0], query_batch_size):
            chunk = query_xy[start : start + query_batch_size]
            if prepared is None:
                batch = make_batch(sample, chunk, device)
                output = model(
                    batch["structure"],
                    batch["query_xy"],
                    interface_condition=batch.get("interface_condition"),
                    local_module_params=batch.get("local_module_params"),
                    teacher_port_tokens=batch.get("teacher_port_tokens"),
                    local_query_points=batch.get("module_internal_query_points"),
                    local_port_condition_mode="predicted",
                    mixed_teacher_ratio=0.5,
                    return_edge_fields=True,
                    return_prepared_state=True,
                )
                prepared = output.pop("prepared_state")
            else:
                query = torch.from_numpy(chunk).unsqueeze(0).to(device=device)
                output = model.decode_prepared(
                    prepared,
                    query,
                    return_routing_maps=False,
                    return_edge_fields=True,
                )
            full = output["pred_field"]
            background = output["pred_field_background"]
            by_edge = output["pred_field_by_edge"]
            closure = (full - background - by_edge.sum(dim=2)).abs().amax()
            max_closure_error = max(max_closure_error, float(closure.detach().cpu()))
            full_chunks.append(full[0].detach().cpu().numpy())
            background_chunks.append(background[0].detach().cpu().numpy())
            edge_chunks.append(by_edge[0].detach().cpu().numpy())
    if prepared is None:
        raise RuntimeError("No query chunks were decoded.")
    shape = (*x_grid.shape, int(model.config.field_dim))
    full_np = np.concatenate(full_chunks, axis=0).reshape(shape)
    background_np = np.concatenate(background_chunks, axis=0).reshape(shape)
    edges_flat = np.concatenate(edge_chunks, axis=0)
    edges_np = edges_flat.reshape(*x_grid.shape, edges_flat.shape[-2], edges_flat.shape[-1])
    return full_np, background_np, edges_np, prepared, max_closure_error


def region_masks(
    sample: dict[str, Any],
    near_width: float,
    far_distance: float,
) -> dict[str, np.ndarray]:
    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    _, fluid = module_and_fluid_masks(sample)
    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"], dtype=np.float32) > 0.5
    radius = module_radius_from_sample(sample)
    if np.any(present):
        distances = [
            np.hypot(x_grid - float(cx), y_grid - float(cy)) - float(radius)
            for cx, cy in centers[present]
        ]
        surface_distance = np.min(np.stack(distances, axis=0), axis=0)
    else:
        surface_distance = np.full(x_grid.shape, np.inf, dtype=np.float32)
    return {
        "whole": np.ones(x_grid.shape, dtype=bool),
        "fluid": np.asarray(fluid, dtype=bool),
        "near_interface_fluid": np.asarray(fluid, dtype=bool)
        & (surface_distance >= 0.0)
        & (surface_distance <= float(near_width)),
        "far_field_fluid": np.asarray(fluid, dtype=bool)
        & (surface_distance >= float(far_distance)),
    }


def rms(values: np.ndarray) -> float:
    values64 = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values64 * values64))) if values64.size else float("nan")


def decomposition_metrics(
    background: np.ndarray,
    edge_sum: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    norm_mean: float,
    norm_std: float,
) -> dict[str, float]:
    b = np.asarray(background, dtype=np.float64).reshape(-1)
    e = np.asarray(edge_sum, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if not b.size:
        return {"num_points": 0.0}
    rb, re, rp, ry = rms(b), rms(e), rms(p), rms(y)
    eb, ee = rb * rb, re * re
    amp_denom = max(rb + re, 1.0e-15)
    energy_denom = max(eb + ee, 1.0e-15)
    b_mean, e_mean, p_mean, y_mean = map(float, (b.mean(), e.mean(), p.mean(), y.mean()))
    b_fluct = rms(b - b_mean)
    e_fluct = rms(e - e_mean)
    p_fluct = rms(p - p_mean)
    y_fluct = rms(y - y_mean)
    full_error = p - y
    edge_error = e - y
    background_error = b - y
    full_mse = float(np.mean(full_error * full_error))
    edge_mse = float(np.mean(edge_error * edge_error))
    background_mse = float(np.mean(background_error * background_error))
    target_energy = max(float(np.sum(y * y)), 1.0e-15)
    cross = float(np.mean(b * e))
    cosine = cross / max(rb * re, 1.0e-15)
    cancellation = max(0.0, (rb + re - rp) / amp_denom)
    std = max(float(norm_std), 1.0e-15)
    offset = float(norm_mean)
    p_phys = offset + std * p
    e_phys = offset + std * e
    b_phys = offset + std * b
    y_phys = offset + std * y
    full_phys_mse = float(np.mean((p_phys - y_phys) ** 2))
    edge_phys_mse = float(np.mean((e_phys - y_phys) ** 2))
    background_phys_mse = float(np.mean((b_phys - y_phys) ** 2))
    return {
        "num_points": float(b.size),
        "background_rms": rb,
        "edge_rms": re,
        "prediction_rms": rp,
        "target_rms": ry,
        "background_amplitude_share": rb / amp_denom,
        "edge_amplitude_share": re / amp_denom,
        "background_to_prediction_rms": rb / max(rp, 1.0e-15),
        "edge_to_prediction_rms": re / max(rp, 1.0e-15),
        "background_energy_share": eb / energy_denom,
        "edge_energy_share": ee / energy_denom,
        "background_mean": b_mean,
        "edge_mean": e_mean,
        "prediction_mean": p_mean,
        "target_mean": y_mean,
        "background_fluctuation_rms": b_fluct,
        "edge_fluctuation_rms": e_fluct,
        "prediction_fluctuation_rms": p_fluct,
        "target_fluctuation_rms": y_fluct,
        "full_mse": full_mse,
        "full_rmse": math.sqrt(full_mse),
        "full_relative_l2": math.sqrt(float(np.sum(full_error * full_error)) / target_energy),
        "edge_only_mse": edge_mse,
        "edge_only_rmse": math.sqrt(edge_mse),
        "edge_only_relative_l2": math.sqrt(float(np.sum(edge_error * edge_error)) / target_energy),
        "background_only_mse": background_mse,
        "background_only_rmse": math.sqrt(background_mse),
        "background_only_relative_l2": math.sqrt(float(np.sum(background_error * background_error)) / target_energy),
        "edge_only_mse_ratio_to_full": edge_mse / max(full_mse, 1.0e-15),
        "edge_only_mse_degradation_percent": 100.0 * (edge_mse / max(full_mse, 1.0e-15) - 1.0),
        "background_correction_fraction_of_edge_error": (edge_mse - full_mse) / max(edge_mse, 1.0e-15),
        "cancellation_ratio": cancellation,
        "background_edge_cosine": cosine,
        "background_edge_cross_energy": cross,
        "physical_normalization_offset": offset,
        "physical_background_mean_increment": std * b_mean,
        "physical_edge_mean_increment": std * e_mean,
        "physical_prediction_mean": float(p_phys.mean()),
        "physical_target_mean": float(y_phys.mean()),
        "physical_full_mse": full_phys_mse,
        "physical_edge_only_mse": edge_phys_mse,
        "physical_background_only_mse": background_phys_mse,
    }


def per_edge_metrics(fields: np.ndarray) -> list[dict[str, float]]:
    """Return metrics for ``[N,K]`` edge contributions for one channel/region."""

    values = np.asarray(fields, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected [N,K], got {values.shape}")
    energy = np.mean(values * values, axis=0)
    total = max(float(np.sum(energy)), 1.0e-15)
    rows = []
    for edge in range(values.shape[1]):
        edge_values = values[:, edge]
        mean = float(np.mean(edge_values)) if edge_values.size else float("nan")
        rows.append(
            {
                "edge": float(edge),
                "energy": float(energy[edge]),
                "energy_fraction": float(energy[edge] / total),
                "mean": mean,
                "mean_energy": mean * mean,
                "fluctuation_rms": rms(edge_values - mean),
                "rms": rms(edge_values),
                "abs_mean": float(np.mean(np.abs(edge_values))) if edge_values.size else float("nan"),
            }
        )
    return rows


def _bypass_edge_additive_output(self: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
    """Process-local timing-only version of the existing edge path."""

    query_xy = kwargs["query_xy"]
    query_state = kwargs["query_state"]
    hyper_state = kwargs["hyper_state"]
    hyper_attention = kwargs["hyper_attention"]
    edge_pair_context = kwargs["edge_pair_context"]
    organizer_output = kwargs["organizer_output"]
    gathered_execution = bool(kwargs["gathered_execution"])
    geometry_features = self._hyper_geometry_features(query_xy, organizer_output)
    edge_active_mask = organizer_output.get("effective_edge_mask")
    if not torch.is_tensor(edge_active_mask):
        edge_active_mask = organizer_output.get("edge_active_mask")
    if not torch.is_tensor(edge_active_mask):
        edge_active_mask = torch.ones_like(hyper_attention[:, 0, :])
    additive_gate = torch.sigmoid(self.additive_edge_gate)
    if gathered_execution:
        edge_sum = self._gathered_edge_execution(
            query_state,
            hyper_state,
            geometry_features,
            edge_pair_context,
            hyper_attention,
            edge_active_mask,
            additive_gate,
            return_edge_fields=False,
        )[0]
    else:
        edge_sum = self._dense_edge_execution(
            query_state,
            hyper_state,
            geometry_features,
            edge_pair_context,
            hyper_attention,
            edge_active_mask,
            additive_gate,
            return_edge_fields=False,
        )[0]
    return {"pred_field": edge_sum}


def benchmark_decoder(
    model: Any,
    prepared: Any,
    query_xy: np.ndarray,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Benchmark normal decode and a restored-after-use background bypass."""

    query = torch.from_numpy(np.asarray(query_xy, dtype=np.float32)).unsqueeze(0).to(device)
    decoder = model.core.decoder
    original = decoder._edge_additive_output
    state_keys_before = tuple(model.state_dict().keys())

    def run_mode(background_bypass: bool) -> tuple[list[float], float, float, np.ndarray]:
        decoder._edge_additive_output = (
            MethodType(_bypass_edge_additive_output, decoder) if background_bypass else original
        )
        elapsed: list[float] = []
        with torch.inference_mode():
            for _ in range(warmup):
                output = model.decode_prepared(prepared, query, return_edge_fields=False)
                del output
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                baseline_allocated = float(torch.cuda.memory_allocated(device))
                baseline_reserved = float(torch.cuda.memory_reserved(device))
            else:
                baseline_allocated = baseline_reserved = float("nan")
            last = None
            for _ in range(iterations):
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = model.decode_prepared(prepared, query, return_edge_fields=False)
                    end.record()
                    torch.cuda.synchronize(device)
                    elapsed.append(float(start.elapsed_time(end)))
                else:
                    tick = time.perf_counter()
                    output = model.decode_prepared(prepared, query, return_edge_fields=False)
                    elapsed.append(1000.0 * (time.perf_counter() - tick))
                last = output["pred_field"].detach().cpu().numpy()
                del output
            if device.type == "cuda":
                peak_allocated = float(torch.cuda.max_memory_allocated(device)) - baseline_allocated
                peak_reserved = float(torch.cuda.max_memory_reserved(device)) - baseline_reserved
            else:
                peak_allocated = peak_reserved = float("nan")
        assert last is not None
        return elapsed, peak_allocated, peak_reserved, last

    try:
        full_times, full_alloc, full_res, full_before = run_mode(False)
        bypass_times, bypass_alloc, bypass_res, edge_only = run_mode(True)
        normal_after_times, _, _, full_after = run_mode(False)
    finally:
        decoder._edge_additive_output = original
    state_keys_after = tuple(model.state_dict().keys())
    full_ms = float(np.median(full_times + normal_after_times))
    bypass_ms = float(np.median(bypass_times))
    return {
        "query_count": int(query.shape[1]),
        "warmup_iterations_per_mode": int(warmup),
        "timed_iterations_full_total": int(len(full_times) + len(normal_after_times)),
        "timed_iterations_bypass": int(len(bypass_times)),
        "full_decoder_median_ms": full_ms,
        "background_bypass_median_ms": bypass_ms,
        "background_time_difference_ms": full_ms - bypass_ms,
        "background_fraction_of_full_time": (full_ms - bypass_ms) / max(full_ms, 1.0e-12),
        "full_incremental_peak_allocated_bytes": full_alloc,
        "bypass_incremental_peak_allocated_bytes": bypass_alloc,
        "full_incremental_peak_reserved_bytes": full_res,
        "bypass_incremental_peak_reserved_bytes": bypass_res,
        "normal_prediction_max_abs_difference_after_restore": float(np.max(np.abs(full_before - full_after))),
        "edge_only_differs_from_full_rms": rms(edge_only - full_before),
        "state_dict_keys_unchanged": state_keys_before == state_keys_after,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        key
        for key in ("checkpoint", "case_id", "region", "channel", "edge")
        if key in columns
    ]
    columns = preferred + [key for key in columns if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    result: dict[str, Any] = {}
    excluded = set(group_keys) | {"case_id", "edge"}
    for group, members in groups.items():
        cursor = result
        for part in group[:-1]:
            cursor = cursor.setdefault(str(part), {})
        leaf: dict[str, Any] = {"case_count": len({row.get("case_id") for row in members})}
        numeric_keys = sorted(set.intersection(*(set(row) for row in members)) - excluded)
        for key in numeric_keys:
            try:
                values = np.asarray([float(row[key]) for row in members], dtype=np.float64)
            except (TypeError, ValueError):
                continue
            finite = values[np.isfinite(values)]
            if finite.size:
                leaf[key] = {
                    "mean": float(np.mean(finite)),
                    "median": float(np.median(finite)),
                    "p05": float(np.quantile(finite, 0.05)),
                    "p95": float(np.quantile(finite, 0.95)),
                }
        cursor[str(group[-1])] = leaf
    return result


def pooled_edge_summary(edge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        grouped[(row["checkpoint"], row["region"], row["channel"], int(row["edge"]))].append(row)
    raw: dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for (checkpoint, region, channel, edge), members in grouped.items():
        raw[checkpoint][region][channel][edge] = {
            "mean_energy": float(np.mean([float(row["energy"]) for row in members])),
            "mean_abs_mean": float(np.mean([abs(float(row["mean"])) for row in members])),
            "mean_fluctuation_rms": float(np.mean([float(row["fluctuation_rms"]) for row in members])),
            "dominant_case_fraction": float(
                np.mean([float(row.get("is_dominant", 0.0)) for row in members])
            ),
        }
    result: dict[str, Any] = {}
    for checkpoint, regions in raw.items():
        result[checkpoint] = {}
        for region, channels in regions.items():
            result[checkpoint][region] = {}
            for channel, edges in channels.items():
                total = max(sum(item["mean_energy"] for item in edges.values()), 1.0e-15)
                result[checkpoint][region][channel] = {
                    str(edge): {**item, "pooled_energy_fraction": item["mean_energy"] / total}
                    for edge, item in sorted(edges.items())
                }
    return result


def evaluate_checkpoint(
    label: str,
    source_path: Path,
    snapshot_path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stable_snapshot(source_path, snapshot_path)
    digest = sha256_file(snapshot_path)
    model, checkpoint = load_model(snapshot_path, device)
    dataset_cfg = checkpoint.get("train_config", {}).get("dataset", {})
    dataset_path = Path(dataset_cfg["packed_h5_path"])
    stats = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    normalizer = H5Normalizer(stats)
    dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=args.split,
        points_per_case=1,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=False,
        normalizer=normalizer,
    )
    channel_order = list(model.config.channelthermal.field_names)
    if channel_order != list(dataset.channel_order):
        raise RuntimeError(
            f"Checkpoint/dataset channel mismatch: {channel_order} != {dataset.channel_order}"
        )
    norm_means = np.asarray(stats["field_mean_by_channel"], dtype=np.float64)
    norm_stds = np.asarray(stats["field_std_by_channel"], dtype=np.float64)
    checkpoint_info: dict[str, Any] = {
        "label": label,
        "source_path": str(source_path),
        "snapshot_sha256": digest,
        "snapshot_size_bytes": snapshot_path.stat().st_size,
        "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "selection_state": checkpoint.get("selection_state"),
        "best_metric": checkpoint.get("best_metric"),
        "channel_order": channel_order,
        "dataset_path": str(dataset_path),
        "dataset_fingerprint": dataset_cfg.get("dataset_fingerprint"),
        "split": args.split,
        "case_count": len(dataset),
        "normalization_mean": norm_means.tolist(),
        "normalization_std": norm_stds.tolist(),
        "max_additive_closure_error": 0.0,
        "state_dict_key_count": len(checkpoint["model_state_dict"]),
        "state_dict_key_digest": hashlib.sha256(
            "\n".join(sorted(checkpoint["model_state_dict"])).encode("utf-8")
        ).hexdigest(),
    }
    metric_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    benchmark_payload = None
    case_count = len(dataset) if args.max_cases is None else min(len(dataset), int(args.max_cases))
    checkpoint_info["evaluated_case_count"] = case_count
    for case_index in range(case_count):
        sample = dataset[case_index]
        prediction, background, by_edge, prepared, closure = prediction_chunks(
            model, sample, device, args.query_batch_size
        )
        checkpoint_info["max_additive_closure_error"] = max(
            float(checkpoint_info["max_additive_closure_error"]), closure
        )
        target = normalizer.normalize_fields(np.asarray(sample["steady_field"], dtype=np.float32))
        edge_sum = by_edge.sum(axis=2)
        masks = region_masks(sample, args.near_interface_width, args.far_surface_distance)
        case_id = str(sample["case_id"])
        for region, mask in masks.items():
            for channel_index, channel in enumerate(channel_order):
                row = {
                    "checkpoint": label,
                    "case_id": case_id,
                    "region": region,
                    "channel": channel,
                }
                row.update(
                    decomposition_metrics(
                        background[..., channel_index][mask],
                        edge_sum[..., channel_index][mask],
                        prediction[..., channel_index][mask],
                        target[..., channel_index][mask],
                        norm_means[channel_index],
                        norm_stds[channel_index],
                    )
                )
                metric_rows.append(row)
                edge_values = by_edge[..., channel_index][mask]
                edge_metrics = per_edge_metrics(edge_values)
                dominant = int(np.argmax([item["energy"] for item in edge_metrics]))
                for item in edge_metrics:
                    edge_rows.append(
                        {
                            "checkpoint": label,
                            "case_id": case_id,
                            "region": region,
                            "channel": channel,
                            **item,
                            "is_dominant": float(int(item["edge"]) == dominant),
                        }
                    )
        if case_index == 0 and label == "1102_best_field":
            x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
            y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
            query_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1)
            benchmark_payload = benchmark_decoder(
                model,
                prepared,
                query_xy,
                device,
                args.benchmark_warmup,
                args.benchmark_iterations,
            )
            benchmark_payload["case_id"] = case_id
            benchmark_payload["checkpoint"] = label
        if (case_index + 1) % 10 == 0 or case_index + 1 == case_count:
            print(f"[{label}] evaluated {case_index + 1}/{case_count} cases", flush=True)
    checkpoint_info["runtime_benchmark"] = benchmark_payload
    dataset.close()
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_info, metric_rows, edge_rows


def main() -> int:
    args = parse_args()
    specs = parse_checkpoint_specs(args.checkpoint)
    for path in specs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_infos: dict[str, Any] = {}
    all_metric_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="honf_additive_checkpoint_snapshot_") as temporary:
        temp_root = Path(temporary)
        for index, (label, source) in enumerate(specs.items()):
            print(f"[{label}] snapshotting {source}", flush=True)
            info, metric_rows, edge_rows = evaluate_checkpoint(
                label,
                source,
                temp_root / f"{index}_{source.name}",
                device,
                args,
            )
            checkpoint_infos[label] = info
            all_metric_rows.extend(metric_rows)
            all_edge_rows.extend(edge_rows)

    metric_csv = output_prefix.with_name(output_prefix.name + "_per_case_channel_region.csv")
    edge_csv = output_prefix.with_name(output_prefix.name + "_per_edge_case_channel_region.csv")
    summary_json = output_prefix.with_name(output_prefix.name + "_summary.json")
    write_csv(metric_csv, all_metric_rows)
    write_csv(edge_csv, all_edge_rows)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "script": str(Path(__file__).resolve()),
        "device": args.device,
        "region_definitions": {
            "whole": "all 64x128 evaluation-grid points",
            "fluid": "checkpoint evaluation grid excluding the packed module mask",
            "near_interface_fluid": (
                f"fluid points with distance to the nearest module surface <= {args.near_interface_width}"
            ),
            "far_field_fluid": (
                f"fluid points with distance to every module surface >= {args.far_surface_distance}"
            ),
        },
        "checkpoints": checkpoint_infos,
        "metric_summary": summarize_rows(
            all_metric_rows, ("checkpoint", "region", "channel")
        ),
        "per_edge_summary": pooled_edge_summary(all_edge_rows),
        "artifacts": {
            "per_case_channel_region_csv": str(metric_csv),
            "per_edge_case_channel_region_csv": str(edge_csv),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] wrote {summary_json}")
    print(f"[done] wrote {metric_csv}")
    print(f"[done] wrote {edge_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
