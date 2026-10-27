"""All-VAL89 native evaluation for maintained R13-6 C/H checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .contextual_frontier_evaluation import load_contextual_checkpoint
from .member_routing_evaluation import _evaluate_native_split, _jsonable, _storage_snapshot


TRAIN_DETAIL_ID = "0003"
VAL_DETAIL_IDS = ("0274", "0293", "0678")
_VARIANTS = {
    "R13-6-O-ME-FaithfulJoint95": {
        "workflow": "interaction_reused_faithful_member_routing_training_v1",
        "candidate": "C",
    },
    "R13-6-H-ME-TypedOrganizationJoint95": {
        "workflow": "interaction_reused_typed_organization_member_routing_training_v1",
        "candidate": "H",
    },
}


def evaluate(checkpoint: str | Path, device: str, output: str | Path) -> dict:
    """Evaluate paired full/restricted native fields on canonical VAL89."""
    started = time.perf_counter()
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError("Keep prior scientific evaluations; select a fresh R13-6 evaluation output.")
    model, provider, payload = load_contextual_checkpoint(checkpoint_path, device)
    identity = payload["experiment_identity"]
    variant = _VARIANTS.get(str(model.architecture))
    if (variant is None or identity.get("workflow") != variant["workflow"]
            or identity.get("training_membership") != "full_train600"):
        raise ValueError("R13-6 evaluation requires a maintained full-TRAIN600 C or H checkpoint.")
    if (len(provider.training_case_ids) != 600 or len(provider.validation_case_ids) != 89
            or provider.validation_scope != "canonical_VAL89_exposed"):
        raise ValueError("R13-6 all-VAL89 evaluation requires full TRAIN600 and canonical exposed VAL89.")
    if TRAIN_DETAIL_ID not in provider.training_case_ids or any(
            case_id not in provider.validation_case_ids for case_id in VAL_DETAIL_IDS):
        raise ValueError("The fixed R13-6 detail panel is absent from the saved partitions.")

    recipe = provider.recipe
    output_root = Path(recipe["storage"]["output_root"]).expanduser().resolve(strict=True)
    if output_path == output_root or not output_path.is_relative_to(output_root):
        raise ValueError(f"R13-6 evaluation output must be below saved DATA2 root {output_root}.")
    storage_options = {
        "storage_output_root": output_root,
        "output_budget_bytes": int(recipe["evaluation"]["max_output_bytes"]),
        "payload_budget_bytes": int(recipe["storage"]["max_output_bytes"]),
        "temporary_budget_bytes": int(recipe["storage"]["max_peak_temporary_bytes"]),
        "minimum_free_headroom_bytes": int(recipe["storage"]["minimum_free_headroom_bytes"]),
    }
    _storage_snapshot(output_root, **storage_options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    storage = _storage_snapshot(output_path.parent, **storage_options)
    output_path.mkdir(parents=True)

    executions = {}
    for mode in ("full", "restricted"):
        mode_dir = output_path / mode
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
        executions[mode] = {"validation": validation, "train_detail": train}

    result = {
        "schema": "thermal_r13_6_member_routing_native_evaluation_v1",
        "status": "completed",
        "candidate": variant["candidate"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "arm": model.architecture,
        "core_schema": model.core.schema,
        "address_mode": model.core.address_mode,
        "joint_member_coverage": model.core.joint_member_coverage,
        "retained_prior_law": model.core.retained_prior_law,
        "training_membership": identity.get("training_membership"),
        "partition_scope": provider.validation_scope,
        "validation_panel": "all 89 canonical VAL cases; nine native observables report mean, p90, worst and per-M strata",
        "executions": executions,
        "fixed_detail_panel": {
            "TRAIN": [TRAIN_DETAIL_ID],
            "VAL": list(VAL_DETAIL_IDS),
            "format": "compressed NPZ in each independent read-mode directory",
        },
        "fit_epochs": 0,
        "optimizer_updates": 0,
        "solver_attempts": 0,
        "TEST_access": False,
        "inverse_tested": False,
        "storage": storage,
        "process_seconds": time.perf_counter() - started,
    }
    (output_path / "result.json").write_text(
        json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    actual_bytes = sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file())
    if actual_bytes > int(recipe["evaluation"]["max_output_bytes"]):
        raise OSError("R13-6 native evaluation exceeded its declared output budget.")
    return {
        "status": "completed",
        "output": str(output_path),
        "actual_bytes": actual_bytes,
        "checkpoint_epoch": int(payload["epoch"]),
        "process_seconds": result["process_seconds"],
    }


__all__ = ["evaluate", "TRAIN_DETAIL_ID", "VAL_DETAIL_IDS"]
