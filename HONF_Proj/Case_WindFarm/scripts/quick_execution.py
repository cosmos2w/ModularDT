"""Run the bounded, disposable real-data WindFarm learning checks.

This script deliberately bypasses the managed run allocator.  It creates fresh
models for each architecture, trains only on a fixed four-row training batch,
and writes compact evidence under the ignored study store.  The resulting
weights are never used by the formal runs.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.config import UnifiedForwardConfig
from honf_runtime.compat import select_device, set_seed
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.registry import load_case_plugin
from torch.utils.data import DataLoader

from windfarm.data import WindFarmNativeDataset, WindFarmNativeView, collate_windfarm
from windfarm.model import WindFarmForwardModel
from windfarm.normalization import read_normalization_json
from windfarm.splits import make_group_split
from windfarm.workflows.evaluate_forward import _compact_metadata
from windfarm.workflows.train_forward import _as_device_batch, _run_loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classic-config", default="project://src/config_core/forward/windfarm_classic_k6.json")
    parser.add_argument("--dense-config", default="project://src/config_core/forward/windfarm_dense_pairwise.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--output-dir", default="project://Case_WindFarm/diagnostics/generated/forward_velocity_study/quick_execution")
    return parser.parse_args()


def _resolved_case(config_path: str, output_dir: Path) -> tuple[dict[str, Any], Any]:
    bundle = load_config_bundle(config_path)
    plugin = load_case_plugin(bundle.case["plugin"])
    plugin.validate_config(bundle)
    return plugin._forward_config(bundle, output_dir), bundle


def _select_probe_rows(view: WindFarmNativeView, train_rows: np.ndarray) -> list[int]:
    cases = [view.run(int(row)) for row in train_rows]
    m_values = np.asarray([case.n_turbines for case in cases], dtype=np.float64)
    volume_values = np.asarray([case.support.volume_D3 for case in cases], dtype=np.float64)
    # Retain the two support-volume extrema and then force both turbine-count
    # extrema.  This makes the geometry coverage explicit even when the two
    # descriptors are correlated in the source layouts.
    chosen: list[int] = []
    for index in (
        int(np.argmin(volume_values)),
        int(np.argmax(volume_values)),
        int(np.argmin(m_values)),
        int(np.argmax(m_values)),
    ):
        row = int(cases[index].index)
        if row not in chosen:
            chosen.append(row)
    if len(chosen) != 4:
        raise RuntimeError("Could not select four distinct real training rows for the probe.")
    return chosen


def _prediction_summary(model: WindFarmForwardModel, raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch = _as_device_batch(raw, device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        prepared = model.prepare_case(batch)
        standardized = model.predict_standardized(
            prepared,
            batch.query_xy,
            batch.query_features,
            receiver_chunk_size=128,
        )
        physical = model.predict_physical(
            prepared,
            batch.query_xy,
            batch.query_features,
            receiver_chunk_size=128,
        )
    model.train(was_training)
    return {
        "standardized_mean": standardized.mean(dim=(0, 1)).cpu().tolist(),
        "standardized_std": standardized.std(dim=(0, 1), unbiased=False).cpu().tolist(),
        "physical_mean_mps": physical.mean(dim=(0, 1)).cpu().tolist(),
        "physical_first_query_mps": physical[0, 0].cpu().tolist(),
    }


def _fresh_model(config: Mapping[str, Any], normalizer: Any, device: torch.device) -> WindFarmForwardModel:
    core_payload = dict(config["model"]["core_honf"])
    return WindFarmForwardModel(UnifiedForwardConfig.from_dict(core_payload), velocity_transform=normalizer).to(device)


def _run_architecture(
    name: str,
    config: Mapping[str, Any],
    view: WindFarmNativeView,
    normalizer: Any,
    train_rows: np.ndarray,
    probe_rows: list[int],
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    set_seed(0)
    dataset = WindFarmNativeDataset(
        view,
        probe_rows,
        normalizer=normalizer,
        queries_per_case=512,
        volume_fraction=0.75,
        seed=42,
        fixed_sampling=True,
    )
    raw = collate_windfarm([dataset[index] for index in range(len(dataset))])
    batch = _as_device_batch(raw, device)
    model = _fresh_model(config, normalizer, device)
    model.materialize(batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-5)
    channel_weights = torch.ones(3, device=device)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_windfarm)
    trajectory: list[dict[str, Any]] = []
    receiver_chunk_size = int(config["model"]["core_honf"].get("interface_model", {}).get("receiver_chunk_size", 128))
    for step in range(1, min(int(steps), 60) + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        metrics = _run_loader(
            model,
            loader,
            device,
            optimizer=optimizer,
            channel_weights=channel_weights,
            receiver_chunk_size=receiver_chunk_size,
            max_batches=1,
            gradient_clip_norm=1.0,
            volume_queries=dataset.volume_queries,
        )
        prediction = _prediction_summary(model, raw, device)
        trajectory.append(
            {
                "step": step,
                "loss": metrics["loss"],
                "volume_mse": metrics["volume_mse"],
                "band_mse": metrics["band_mse"],
                "gradient_norm_preclip": metrics["gradient_norm_preclip"],
                "clip_scale": metrics["clip_scale"],
                "sampled_update_norm": metrics["sampled_update_norm"],
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_memory_mb": (
                    float(torch.cuda.max_memory_allocated(device) / 2**20)
                    if device.type == "cuda" else 0.0
                ),
                "prediction": prediction,
            }
        )
    del model, optimizer, loader, batch, raw
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # A fresh instance measures the complete logical B=8 path at Q=1024 with
    # the largest-M training rows.  It does not reuse disposable probe state.
    largest = sorted((int(row) for row in train_rows), key=lambda row: (-view.run(row).n_turbines, row))[:8]
    large_dataset = WindFarmNativeDataset(
        view,
        largest,
        normalizer=normalizer,
        queries_per_case=1024,
        volume_fraction=0.75,
        seed=42,
        fixed_sampling=True,
    )
    large_loader = DataLoader(large_dataset, batch_size=8, shuffle=False, collate_fn=collate_windfarm)
    large_raw = next(iter(large_loader))
    large_batch = _as_device_batch(large_raw, device)
    set_seed(0)
    large_model = _fresh_model(config, normalizer, device)
    large_model.materialize(large_batch)
    large_optimizer = torch.optim.AdamW(large_model.parameters(), lr=3.0e-4, weight_decay=1.0e-5)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    large_started = time.perf_counter()
    large_metrics = _run_loader(
        large_model,
        large_loader,
        device,
        optimizer=large_optimizer,
        channel_weights=channel_weights,
        receiver_chunk_size=receiver_chunk_size,
        max_batches=1,
        gradient_clip_norm=1.0,
        volume_queries=large_dataset.volume_queries,
    )
    large_record = {
        "rows": largest,
        "max_n_turbines": max(view.run(row).n_turbines for row in largest),
        "queries_per_case": 1024,
        "batch_size": 8,
        "loss": large_metrics["loss"],
        "gradient_norm_preclip": large_metrics["gradient_norm_preclip"],
        "sampled_update_norm": large_metrics["sampled_update_norm"],
        "wall_seconds": time.perf_counter() - large_started,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0
        ),
    }
    del large_model, large_optimizer, large_loader, large_batch, large_raw
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "architecture": name,
        "probe_rows": [
            {
                "row": int(row),
                "case": view.run(int(row)).case,
                "n_turbines": view.run(int(row)).n_turbines,
                "support_volume_D3": view.run(int(row)).support.volume_D3,
            }
            for row in probe_rows
        ],
        "queries_per_case": 512,
        "environment_tokens": 512,
        "max_steps": min(int(steps), 60),
        "trajectory": trajectory,
        "large_batch": large_record,
    }


def main() -> int:
    args = _parse_args()
    output_value = str(args.output_dir)
    if output_value.startswith("project://"):
        from honf_runtime.paths import resolve_path

        output = resolve_path(output_value)
    else:
        output = Path(output_value).expanduser()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    classic_config, _bundle = _resolved_case(args.classic_config, output)
    dense_config, _ = _resolved_case(args.dense_config, output)
    dataset_cfg = classic_config["dataset"]
    view = WindFarmNativeView(
        dataset_cfg["volume_path"],
        compact_metadata=_compact_metadata(dataset_cfg["compact_path"]),
        token_shape=tuple(dataset_cfg.get("env_token_shape", (16, 8, 4))),
    )
    split = make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    normalizer, _ = read_normalization_json(dataset_cfg["normalization"])
    probe_rows = _select_probe_rows(view, split.train)
    device = select_device(args.device)
    result: dict[str, Any] = {
        "scope": "disposable real-data fixed-sample probe; weights are discarded",
        "device": str(device),
        "split_seed": 42,
        "train_rows": int(split.train.size),
        "probe_rows": probe_rows,
        "models": [],
    }
    destination = output / "quick_execution.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, config in (("classic_k6", classic_config), ("dense_pairwise", dense_config)):
        try:
            result["models"].append(
                _run_architecture(name, config, view, normalizer, split.train, probe_rows, device, args.steps)
            )
        except BaseException as exc:
            result["failure"] = {"architecture": name, "type": type(exc).__name__, "message": str(exc)}
            destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[windfarm-quick] wrote {output / 'quick_execution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
