#!/usr/bin/env python3
"""Stream exact Kq histograms for all 90 test cases without saving route maps."""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DATASET = Path("/data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5")
DEVICE = "cuda:2"
QUERY_BATCH_SIZE = 2048
RUNS = {
    "Run 1501": REPO / "Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/best_by_field_mse_model.pt",
    "Run 1502": REPO / "Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1502_20260923_004751_sparse_incidence_environment_sparsemax/best_by_field_mse_model.pt",
}


def main() -> None:
    diagnostics = REPO / "tools" / "diagnostics"
    for path in (REPO / "src", REPO / "Case_ThermalChannel" / "src", diagnostics):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model
    from channelthermal.evaluation.prepared import predict_case
    from run_run1409_occupancy_population import _dataset_for_checkpoint, _query_sample

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    device = torch.device(DEVICE)
    all_case_rows: list[dict[str, object]] = []
    aggregate: dict[str, object] = {}
    for run_name, checkpoint_path in RUNS.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        start_time = time.monotonic()
        model, checkpoint = load_model(checkpoint_path, device)
        model.eval()
        dataset, resolved_dataset, _dataset_config = _dataset_for_checkpoint(
            checkpoint,
            dataset_path=str(DATASET),
            split="test",
            GlobalChannelThermalDataset=GlobalChannelThermalDataset,
            H5Normalizer=H5Normalizer,
        )
        case_ids = [str(value) for value in dataset.selected_case_ids]
        if len(case_ids) != 90:
            raise RuntimeError(f"expected 90 test cases, found {len(case_ids)}")
        index_by_case = {case_id: index for index, case_id in enumerate(case_ids)}
        pooled = Counter()
        for order, case_id in enumerate(case_ids, start=1):
            sample = dataset[index_by_case[case_id]]
            selected_sample, query_xy = _query_sample(sample, None)
            if query_xy.shape[0] != 8192:
                raise RuntimeError(f"case {case_id} has Q={query_xy.shape[0]}, expected 8192")
            with torch.inference_mode():
                prediction = predict_case(
                    model,
                    selected_sample,
                    device,
                    query_batch_size=QUERY_BATCH_SIZE,
                    local_port_condition_mode="predicted",
                    mixed_teacher_ratio=0.0,
                    return_routing_maps=True,
                    return_prepared_state=False,
                )
            maps = prediction.get("routing_maps", {})
            assignment = maps.get("sparse_incidence_query_routing", maps.get("group_control_query_routing"))
            if assignment is None:
                raise RuntimeError(f"no sparse/group-control query assignment for case {case_id}")
            assignment = np.asarray(assignment)
            if assignment.shape != (8192, 12):
                raise RuntimeError(f"unexpected route shape for {case_id}: {assignment.shape}")
            if not np.isfinite(assignment).all():
                raise RuntimeError(f"non-finite query assignment for case {case_id}")
            degree = np.count_nonzero(assignment > 0.0, axis=-1)
            if np.any((degree < 1) | (degree > 12)):
                raise RuntimeError(f"invalid Kq support range for case {case_id}")
            local_hist = np.bincount(degree, minlength=13)[1:13]
            pooled.update({k: int(v) for k, v in enumerate(local_hist, start=1) if v})
            all_case_rows.append(
                {
                    "run": run_name,
                    "checkpoint": str(checkpoint_path),
                    "case_id": case_id,
                    "query_count": int(degree.size),
                    "Kq_min": int(degree.min()),
                    "Kq_max": int(degree.max()),
                    "Kq_mean": float(degree.mean()),
                    "Kq_q10": float(np.percentile(degree, 10)),
                    "Kq_q90": float(np.percentile(degree, 90)),
                    "Kq_histogram": json.dumps({str(k): int(v) for k, v in enumerate(local_hist, start=1) if v}),
                }
            )
            del prediction, maps, assignment, degree, sample, selected_sample
            if order % 10 == 0 or order == len(case_ids):
                print(f"[{run_name}] {order}/90 complete", flush=True)
        total = sum(pooled.values())
        mean = sum(k * v for k, v in pooled.items()) / total
        aggregate[run_name] = {
            "checkpoint": str(checkpoint_path),
            "architecture": str(model.config.core_honf.forward_architecture),
            "registered_capacity": 12,
            "dataset": str(resolved_dataset),
            "split": "test",
            "case_count": len(case_ids),
            "queries_per_case": 8192,
            "total_queries": total,
            "query_batch_size": QUERY_BATCH_SIZE,
            "device": DEVICE,
            "Kq_min": min(pooled),
            "Kq_max": max(pooled),
            "Kq_mean": mean,
            "Kq_histogram": {str(k): int(pooled.get(k, 0)) for k in range(1, 13)},
            "elapsed_seconds": time.monotonic() - start_time,
            "note": "Exact count of query assignment entries > 0; supports the full original 8192-point grid per case. Route tensors were reduced in memory and not written to disk.",
        }
        del model
        torch.cuda.empty_cache()
    out_dir = HERE
    (out_dir / "fullgrid_kq_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    with (out_dir / "fullgrid_kq_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_case_rows[0]))
        writer.writeheader()
        writer.writerows(all_case_rows)
    histogram_rows = []
    for run_name, summary in aggregate.items():
        for k, count in summary["Kq_histogram"].items():
            histogram_rows.append(
                {
                    "run": run_name,
                    "Kq": int(k),
                    "query_count": int(count),
                    "fraction": int(count) / int(summary["total_queries"]),
                    "percent": 100.0 * int(count) / int(summary["total_queries"]),
                    "total_query_assignments": int(summary["total_queries"]),
                }
            )
    with (out_dir / "fullgrid_kq_histogram.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histogram_rows[0]))
        writer.writeheader()
        writer.writerows(histogram_rows)
    print(f"Wrote full-grid exact Kq summaries to {out_dir}")


if __name__ == "__main__":
    main()
