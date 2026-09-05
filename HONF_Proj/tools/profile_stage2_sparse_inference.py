#!/usr/bin/env python3
"""Measure synchronized sparse-interface preparation and prepared decoding."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.evaluation.loading import load_model, make_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    return parser.parse_args()


def measure(
    function: Callable[[], Any],
    device: torch.device,
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        with torch.inference_mode():
            value = function()
        del value
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    samples_ms: list[float] = []
    for _ in range(repetitions):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            value = function()
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        del value
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "median_ms": float(statistics.median(samples_ms)),
        "mean_ms": float(statistics.mean(samples_ms)),
        "p05_ms": float(np.quantile(samples_ms, 0.05)),
        "p95_ms": float(np.quantile(samples_ms, 0.95)),
        "samples_ms": samples_ms,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(peak_allocated - baseline_allocated, 0),
        "incremental_peak_reserved_bytes": max(peak_reserved - baseline_reserved, 0),
    }


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("--warmups must be nonnegative and --repetitions must be positive.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This utility records synchronized CUDA timings; select a CUDA device.")

    checkpoint_path = args.checkpoint.resolve()
    model, checkpoint = load_model(checkpoint_path, device)
    model.eval()
    dataset_config = checkpoint.get("train_config", {}).get("dataset", {})
    statistics_payload = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    dataset = GlobalChannelThermalDataset(
        dataset_config["packed_h5_path"],
        split="test",
        points_per_case=1,
        normalize_inputs=bool(dataset_config.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_config.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
        normalizer=H5Normalizer(statistics_payload) if statistics_payload else None,
    )
    case_ids = args.case_id or ["0273", "0653"]
    case_indices = {
        str(case_id): index for index, case_id in enumerate(dataset.selected_case_ids)
    }
    missing = [case_id for case_id in case_ids if case_id not in case_indices]
    if missing:
        raise KeyError(f"Cases are absent from the test split: {missing}")

    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        sample = dataset[case_indices[case_id]]
        query_xy = np.stack(
            (sample["x_grid"].reshape(-1), sample["y_grid"].reshape(-1)), axis=-1
        ).astype(np.float32)
        batch = make_batch(sample, query_xy, device)
        forward_kwargs = {
            "interface_condition": batch.get("interface_condition"),
            "local_module_params": batch.get("local_module_params"),
            "teacher_port_tokens": batch.get("teacher_port_tokens"),
            "local_query_points": batch.get("module_internal_query_points"),
            "local_port_condition_mode": "predicted",
            "mixed_teacher_ratio": 0.0,
        }

        def full_forward() -> Any:
            return model(batch["structure"], batch["query_xy"], **forward_kwargs)

        def prepare_with_one_query() -> Any:
            return model(
                batch["structure"],
                batch["query_xy"][:, :1],
                return_prepared_state=True,
                **forward_kwargs,
            )

        for phase, query_count, function in (
            ("full_forward", int(query_xy.shape[0]), full_forward),
            ("physical_preparation_plus_one_query", 1, prepare_with_one_query),
        ):
            result = measure(
                function,
                device,
                warmups=int(args.warmups),
                repetitions=int(args.repetitions),
            )
            rows.append(
                {
                    "case_id": case_id,
                    "phase": phase,
                    "query_count": query_count,
                    **result,
                }
            )

        with torch.inference_mode():
            prepared = prepare_with_one_query()["prepared_state"]

        def prepared_decode() -> Any:
            return model.decode_prepared(prepared, batch["query_xy"])

        result = measure(
            prepared_decode,
            device,
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        )
        rows.append(
            {
                "case_id": case_id,
                "phase": "prepared_decode",
                "query_count": int(query_xy.shape[0]),
                **result,
            }
        )
        del prepared, batch

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "warmups": int(args.warmups),
        "repetitions": int(args.repetitions),
        "synchronized": True,
        "scope": (
            "physical_preparation_plus_one_query includes encoding, layout construction, "
            "P0/P1/P2 preparation, the frozen local operator, one refinement, and one decode"
        ),
        "rows": rows,
    }
    json_path = output_dir / "gpu0_phase_timing.json"
    csv_path = output_dir / "gpu0_phase_timing.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "case_id",
            "phase",
            "query_count",
            "median_ms",
            "mean_ms",
            "p05_ms",
            "p95_ms",
            "baseline_allocated_bytes",
            "baseline_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "incremental_peak_allocated_bytes",
            "incremental_peak_reserved_bytes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{key: value for key, value in row.items() if key != "samples_ms"} for row in rows]
        )
    dataset.close()
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
