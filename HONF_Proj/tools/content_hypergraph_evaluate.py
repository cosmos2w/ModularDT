#!/usr/bin/env python3
"""Evaluate retained content-hypergraph checkpoints on their native panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "tools", PROJECT / "Case_ThermalChannel/src", PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from content_hypergraph_train import build, canonical_sha256, engine_for, validate_recipe

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.content_hypergraph import load_content_hypergraph_checkpoint


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    return value


def observation_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Keep numeric observations, without serializing live prepared graphs."""
    if torch.is_tensor(value):
        return {prefix: value.detach().cpu().numpy()}
    if isinstance(value, np.ndarray):
        return {prefix: value}
    if is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in fields(value)}
    result = {}
    if isinstance(value, Mapping):
        for name, child in value.items():
            if name in {"prepared", "organization", "scene", "query", "trace", "auxiliary_state"} or name.endswith("_prepared"):
                continue
            key = f"{prefix}__{name}" if prefix else str(name)
            if torch.is_tensor(child) or isinstance(child, (np.ndarray, Mapping)):
                result.update(observation_arrays(child, key))
    return result


def load_evaluation_components(checkpoint: Path, device: str):
    """Reopen only the checkpoint's authorized native evaluation panel."""
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
    recipe = validate_recipe(payload["experiment_identity"]["recipe"])
    model, provider = build(recipe, device=device)
    expected = payload["experiment_identity"]["provider_identity"]
    observed = dict(provider.identity_payload())
    if canonical_sha256(expected) != canonical_sha256(observed):
        raise ValueError("Native evaluation data/scale/provider binding differs from the saved run.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    # Exercise the self-contained path separately; it must reconstruct exactly
    # the same tensor inventory without relying on provider/dataset state.
    standalone, adapter, _ = load_content_hypergraph_checkpoint(checkpoint, device=device)
    for name, value in standalone.state_dict().items():
        if not torch.equal(value, model.state_dict()[name]):
            raise ValueError(f"Standalone native load differs at {name}.")
    del standalone, adapter
    return model, provider, payload


def fixed_monitor_batches(provider, recipe: dict[str, Any], epoch: int, split: str):
    """Replay declared monitoring queries at one fixed checkpoint, without updates."""
    if split == "validation":
        return provider.validation_batches(), None
    engine = engine_for(recipe, str(provider.device))
    ordered, dataset_id = engine.epoch_case_order(provider, epoch)

    def batches():
        for update_index, selection in enumerate(engine.epoch_update_slices(len(ordered))):
            yield from engine.prepare_update_microbatches(
                provider, arm=recipe["architecture"], phase="joint", epoch=epoch,
                update_index=update_index, update_cases=ordered[selection], sampling_dataset_id=dataset_id)

    return batches(), list(ordered)


def label_monitor_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Replace native-endpoint labels with counts measured from monitor rows."""
    result = dict(metrics)
    result["evaluation_panel"] = "fixed_checkpoint_monitor"
    if "query_count_per_row" in result:
        counts = {
            sum(int(role["query_count"]) for role in row["roles"].values())
            for row in result["rows"]
        }
        if len(counts) != 1:
            raise ValueError("Fixed Wind monitoring must have one observed query count per row.")
        result["query_count_per_row"] = counts.pop()
        result["monitor_rows"] = result.pop("endpoint_rows")
        result.pop("endpoint_selection", None)
        result["monitor_selection"] = "complete declared cohort; saved epoch sampling and original packing"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("validation", "train"), default="validation")
    parser.add_argument("--panel", choices=("native", "monitor"), default="native")
    parser.add_argument("--case-keys", help="Comma-separated physical case IDs or native Wind row IDs")
    parser.add_argument("--export-case", action="append", default=[])
    parser.add_argument("--microbatch-cases", type=int, help="Native evaluation packing; exports default to one case")
    args = parser.parse_args(argv)
    packing = args.microbatch_cases if args.microbatch_cases is not None else (1 if args.export_case else None)
    if packing is not None and packing < 1:
        raise ValueError("Native evaluation packing must be positive.")
    if args.export_case and packing != 1:
        raise ValueError("Detailed exports require packing one so only selected cases are materialized.")
    if args.panel == "monitor" and (args.case_keys or args.export_case or args.microbatch_cases):
        raise ValueError("Fixed monitoring replays the complete declared cohort and original packing; use native for details.")
    if args.output.exists():
        raise FileExistsError("Evaluation evidence is immutable; choose a fresh output directory.")
    started = time.perf_counter()
    model, provider, payload = load_evaluation_components(args.checkpoint, args.device)
    recipe = payload["experiment_identity"]["recipe"]
    selected = None if not args.case_keys else args.case_keys.split(",")
    if selected is not None and recipe["dataset"] == "wind":
        selected = [int(x) for x in selected]
    args.output.mkdir(parents=True)
    records, work, exports = [], [], []
    if args.panel == "monitor":
        batches, selected = fixed_monitor_batches(provider, recipe, int(payload["epoch"]), args.split)
    else:
        batches = provider.native_evaluation_batches(split=args.split, case_keys=selected, microbatch_size=packing)
    with torch.no_grad():
        for index, batch in enumerate(batches):
            scene = provider.make_scene(batch.scene_inputs)
            prediction, auxiliary = provider.predict_native(
                model, scene, batch.receivers, recipe["architecture"], "joint",
                epoch=int(payload["epoch"]), temperature=1.0,
            )
            records.append(provider.validation_metrics(prediction, batch.targets, auxiliary))
            work.append(dict(provider.work_counts(batch, prediction, auxiliary)))
            keys = tuple(str(x) for x in batch.case_keys)
            if any(key in args.export_case for key in keys):
                arrays = {}
                for prefix, value in (("scene", scene), ("prediction", prediction), ("target", batch.targets),
                                      ("receiver", batch.receivers)):
                    arrays.update(observation_arrays(value, prefix))
                path = args.output / f"observations_{index:03d}_{'-'.join(keys)}.npz"
                np.savez_compressed(path, **arrays)
                exports.append({"path": str(path), "case_keys": list(keys),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "array_shapes": {k: list(v.shape) for k, v in arrays.items()}})
    reducer = getattr(provider, "reduce_native_evaluation_metrics", None)
    metrics = (reducer(records, split=args.split, case_keys=selected) if callable(reducer)
               else provider.reduce_native_metrics(records))
    if args.panel == "monitor":
        metrics = label_monitor_metrics(metrics)
    result = {
        "schema": "content_hypergraph_native_evaluation_v1", "status": "completed",
        "dataset": recipe["dataset"], "architecture": recipe["architecture"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "completed_epoch": int(payload["epoch"]), "split": args.split,
        "panel": args.panel,
        "receiver_scope": ("full native endpoint support" if args.panel == "native" else
                           "fixed checkpoint; declared primary monitoring queries and native auxiliary samples"),
        "evaluation_microbatch_cases": packing,
        "provider_binding_sha256": canonical_sha256(payload["experiment_identity"]["provider_identity"]),
        "metrics": metrics, "records": records, "work_counts": work, "exports": exports,
        "elapsed_seconds": time.perf_counter() - started, "optimizer_updates": 0,
        "new_solver_attempts": 0, "WindTEST": "locked",
    }
    destination = args.output / "result.json"
    destination.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_safe({k: v for k, v in result.items() if k not in {"records", "exports"}}),
                     sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
