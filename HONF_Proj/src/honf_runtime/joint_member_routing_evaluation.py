"""Maintained all-VAL89 and fixed-detail evaluation for R13-5 joint95 checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .contextual_frontier_evaluation import load_contextual_checkpoint
from .member_routing_evaluation import (
    _evaluate_native_split,
    _jsonable,
    _storage_snapshot,
)


TRAIN_DETAIL_ID = "0003"
VAL_DETAIL_IDS = ("0274", "0293", "0678")


def evaluate(checkpoint: str | Path, device: str, output: str | Path) -> dict:
    started = time.perf_counter()
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError("Keep prior scientific evaluations; select a fresh R13-5 evaluation output.")
    model, provider, payload = load_contextual_checkpoint(checkpoint, device)
    identity = payload["experiment_identity"]
    provider_identity = identity.get("provider_identity", {})
    lineage = provider_identity.get("loss_lineage", {}) if isinstance(provider_identity, dict) else {}
    source_parent_epoch = int(lineage.get("parent_completed_epoch", 0))
    additional_reader_epochs = max(0, int(payload["epoch"]) - source_parent_epoch)
    if (identity.get("workflow") != "interaction_reused_joint_member_routing_joint_training_v1"
            or model.architecture not in ("R13-5-M-Joint95", "R13-5-ME-Joint95")):
        raise ValueError("R13-5 evaluation requires a maintained joint95 checkpoint.")
    if len(provider.training_case_ids) != 600 or len(provider.validation_case_ids) != 89:
        raise ValueError("R13-5 all-VAL89 native evaluation requires full TRAIN600 and canonical VAL89.")
    if TRAIN_DETAIL_ID not in provider.training_case_ids or any(
            case_id not in provider.validation_case_ids for case_id in VAL_DETAIL_IDS):
        raise ValueError("The fixed R13-5 detail panel is not present in the checkpoint's exposed partitions.")

    recipe = provider.recipe
    output_root = Path(recipe["storage"]["output_root"]).expanduser().resolve(strict=True)
    if output == output_root or not output.is_relative_to(output_root):
        raise ValueError(f"R13-5 evaluation output must be below saved DATA2 root {output_root}.")
    storage_options = {
        "storage_output_root": output_root,
        "output_budget_bytes": int(recipe["evaluation"]["max_output_bytes"]),
        "payload_budget_bytes": int(recipe["storage"]["max_output_bytes"]),
        "temporary_budget_bytes": int(recipe["storage"]["max_peak_temporary_bytes"]),
        "minimum_free_headroom_bytes": int(recipe["storage"]["minimum_free_headroom_bytes"]),
    }
    # Reserve against the existing DATA2 root before materializing any new directory.
    _storage_snapshot(output_root, **storage_options)
    output.parent.mkdir(parents=True, exist_ok=True)
    storage = _storage_snapshot(output.parent, **storage_options)
    output.mkdir(parents=True)
    modes = {}
    for mode in ("full", "restricted"):
        mode_dir = output / mode
        mode_dir.mkdir()
        with model.core.read_execution(mode):
            validation = _evaluate_native_split(
                model, provider, payload, "validation", provider.validation_case_ids,
                export_ids=set(VAL_DETAIL_IDS),
            )
            train = _evaluate_native_split(
                model, provider, payload, "train", (TRAIN_DETAIL_ID,),
                export_ids={TRAIN_DETAIL_ID},
            )
        for split in (validation, train):
            for case_id, arrays in split.pop("representative_arrays").items():
                np.savez_compressed(mode_dir / f"case_{case_id}.npz", **arrays)
        modes[mode] = {"validation": validation, "train_detail": train}

    result = {
        "schema": "thermal_joint_member_routing_native_evaluation_v1",
        "status": "completed",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "arm": model.architecture,
        "address_mode": model.core.address_mode,
        "joint_member_coverage": model.core.joint_member_coverage,
        "training_membership": identity.get("training_membership"),
        "source_parent_completed_epoch": source_parent_epoch,
        "additional_joint_member_reader_epochs": additional_reader_epochs,
        "partition_scope": provider.validation_scope,
        "validation_panel": "all 89 canonical VAL cases; nine native observables report mean, p90, worst and per-M strata",
        "executions": modes,
        "fixed_detail_panel": {
            "TRAIN": [TRAIN_DETAIL_ID],
            "VAL": list(VAL_DETAIL_IDS),
            "format": "compressed NPZ in each independent mode directory",
        },
        "fit_epochs": 0,
        "optimizer_updates": 0,
        "solver_attempts": 0,
        "TEST_access": False,
        "inverse_tested": False,
        "storage": storage,
        "process_seconds": time.perf_counter() - started,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    actual_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if actual_bytes > int(recipe["evaluation"]["max_output_bytes"]):
        raise OSError("R13-5 native evaluation exceeded its declared output budget.")
    return {
        "status": "completed",
        "output": str(output),
        "actual_bytes": actual_bytes,
        "checkpoint_epoch": int(payload["epoch"]),
        "process_seconds": result["process_seconds"],
    }


__all__ = ["evaluate", "TRAIN_DETAIL_ID", "VAL_DETAIL_IDS"]
