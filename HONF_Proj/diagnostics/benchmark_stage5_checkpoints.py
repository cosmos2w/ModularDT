#!/usr/bin/env python3
"""Benchmark frozen HONF checkpoints through identical evaluation-only paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "diagnostics"))

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer  # noqa: E402
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
    parser.add_argument("--case-id", default="0273")
    parser.add_argument("--queries", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "diagnostics" / "stage5_final_checkpoint_benchmark.json",
    )
    add_frozen_override_arguments(parser)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_call(
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        with torch.inference_mode():
            output = function()
        del output
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    elapsed: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = function()
        torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - started)
        del output
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "median_seconds": float(statistics.median(elapsed)),
        "mean_seconds": float(statistics.mean(elapsed)),
        "p05_seconds": float(np.quantile(elapsed, 0.05)),
        "p95_seconds": float(np.quantile(elapsed, 0.95)),
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
        "incremental_peak_reserved_bytes": peak_reserved - baseline_reserved,
    }


def evaluate_checkpoint(
    label: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model, checkpoint = load_model(checkpoint_path, device)
    model.eval()
    frozen_overrides = apply_label_frozen_overrides(model, label, args)
    dataset_cfg = checkpoint["train_config"]["dataset"]
    stats = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    dataset = GlobalChannelThermalDataset(
        dataset_cfg["packed_h5_path"],
        split=args.split,
        points_per_case=1,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=False,
        normalizer=H5Normalizer(stats),
    )
    matches = [
        index
        for index, case_id in enumerate(dataset.selected_case_ids)
        if str(case_id) == str(args.case_id)
    ]
    if not matches:
        raise KeyError(f"case_id={args.case_id!r} is absent from split {args.split!r}")
    sample = dataset[matches[0]]
    all_queries = np.stack(
        (np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)),
        axis=-1,
    ).astype(np.float32)
    queries = all_queries[: min(int(args.queries), len(all_queries))]
    batch = make_batch(sample, queries, device)

    def full_forward() -> dict[str, Any]:
        return model(
            batch["structure"],
            batch["query_xy"],
            interface_condition=batch.get("interface_condition"),
            local_module_params=batch.get("local_module_params"),
            teacher_port_tokens=batch.get("teacher_port_tokens"),
            local_query_points=batch.get("module_internal_query_points"),
            local_port_condition_mode="predicted",
        )

    with torch.inference_mode():
        prepared_output = model(
            batch["structure"],
            batch["query_xy"][:, :1],
            interface_condition=batch.get("interface_condition"),
            local_module_params=batch.get("local_module_params"),
            teacher_port_tokens=batch.get("teacher_port_tokens"),
            local_query_points=batch.get("module_internal_query_points"),
            local_port_condition_mode="predicted",
            return_prepared_state=True,
        )
    prepared = prepared_output["prepared_state"]
    del prepared_output

    def prepared_decode() -> dict[str, Any]:
        return model.decode_prepared(prepared, batch["query_xy"])

    state_keys_before = tuple(model.state_dict())
    decoder = benchmark_call(
        prepared_decode,
        device=device,
        warmup=int(args.warmup),
        iterations=int(args.iterations),
    )
    full = benchmark_call(
        full_forward,
        device=device,
        warmup=int(args.warmup),
        iterations=int(args.iterations),
    )
    state_keys_after = tuple(model.state_dict())
    if state_keys_before != state_keys_after:
        raise RuntimeError(f"{label}: benchmark changed state_dict structure")
    parameters = list(model.parameters())
    core_cfg = checkpoint.get("model_config", {}).get("core_honf", {})
    result = {
        "checkpoint": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "case_id": str(sample["case_id"]),
        "query_count": int(len(queries)),
        "organizer_mode": core_cfg.get("organizer_mode"),
        "field_assembly_mode": core_cfg.get("field_assembly_mode"),
        "total_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "state_dict_key_count": len(state_keys_before),
        "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
        "prepared_decoder": decoder,
        "full_forward": full,
        "state_dict_structure_unchanged": state_keys_before == state_keys_after,
        "frozen_overrides": frozen_overrides,
    }
    dataset.close()
    del prepared, batch, model, checkpoint
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The Stage-5 checkpoint benchmark requires CUDA.")
    checkpoints: dict[str, Path] = {}
    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        checkpoints[label] = Path(raw_path).expanduser().resolve()
    resolve_frozen_overrides(args, checkpoints)
    results = {
        label: evaluate_checkpoint(label, path, args, device)
        for label, path in checkpoints.items()
    }
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "split": args.split,
        "case_id": str(args.case_id),
        "query_count": int(args.queries),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
