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
from content_hypergraph_train import build, canonical_sha256, validate_recipe

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("validation", "train"), default="validation")
    parser.add_argument("--case-keys", help="Comma-separated physical case IDs or native Wind row IDs")
    parser.add_argument("--export-case", action="append", default=[])
    args = parser.parse_args(argv)
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
    with torch.no_grad():
        for index, batch in enumerate(provider.native_evaluation_batches(split=args.split, case_keys=selected)):
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
                for prefix, value in (("prediction", prediction), ("target", batch.targets),
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
    result = {
        "schema": "content_hypergraph_native_evaluation_v1", "status": "completed",
        "dataset": recipe["dataset"], "architecture": recipe["architecture"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "completed_epoch": int(payload["epoch"]), "split": args.split,
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
