"""Checkpoint loading and native evaluation for the R13-5A/B supplement."""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from channelthermal.supplement_member_routing import (
    ThermalOrganizationConditionedMemberRoutingAdapter,
    ThermalSupplementMemberRoutingAdapter,
)
from channelthermal.training.supplement_member_routing_task import (
    ARM,
    B_ARM,
    B_WORKFLOW,
    WORKFLOW,
)
from .compat import load_trusted_checkpoint
from .member_routing_evaluation import _evaluate_native_split, _jsonable, _storage_snapshot


SUPPLEMENT_PROVIDER_SCHEMA = "thermal_r13_5_supplement_member_routing_provider_v1"
_VARIANTS_BY_CORE_SCHEMA = {
    ThermalSupplementMemberRoutingAdapter.CORE_SCHEMA: {
        "adapter": ThermalSupplementMemberRoutingAdapter,
        "arm": ARM,
        "workflow": WORKFLOW,
        "candidate": "A",
    },
    ThermalOrganizationConditionedMemberRoutingAdapter.CORE_SCHEMA: {
        "adapter": ThermalOrganizationConditionedMemberRoutingAdapter,
        "arm": B_ARM,
        "workflow": B_WORKFLOW,
        "candidate": "B",
    },
}


def _checkpoint_parts(payload: Mapping[str, object]):
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Checkpoint is not an R13-5 supplement training checkpoint.")
    provider = identity.get("provider_identity")
    if not isinstance(provider, Mapping) or provider.get("schema") != SUPPLEMENT_PROVIDER_SCHEMA:
        raise ValueError("R13-5 supplement checkpoint provider identity does not match its workflow.")
    adapter_state = provider.get("adapter_state")
    recipe = provider.get("recipe")
    arm = str(payload.get("arm", provider.get("architecture", "")))
    if not isinstance(adapter_state, Mapping) or not isinstance(recipe, Mapping):
        raise ValueError("R13-5 supplement checkpoint is missing its adapter state or saved recipe.")
    core = adapter_state.get("core")
    core_schema = core.get("schema") if isinstance(core, Mapping) else None
    variant = _VARIANTS_BY_CORE_SCHEMA.get(str(core_schema))
    if variant is None:
        raise ValueError("R13-5 supplement checkpoint has an unsupported A/B core schema.")
    adapter_type = variant["adapter"]
    expected_arm = str(variant["arm"])
    if (identity.get("workflow") != variant["workflow"]
            or adapter_state.get("schema") != adapter_type.SCHEMA
            or provider.get("member_routing_schema") != adapter_type.CORE_SCHEMA
            or provider.get("architecture") != expected_arm or arm != expected_arm
            or not isinstance(core, Mapping) or core.get("architecture") != expected_arm
            or core.get("address_mode") != "ME"):
        raise ValueError("R13-5 supplement checkpoint adapter, workflow, core, and architecture identities disagree.")
    return identity, provider, adapter_state, dict(recipe), arm, variant


def _bind_loaded_model(provider, model, device: torch.device) -> None:
    provider.model = model
    provider.adapter = model
    provider.device = device
    organizer = getattr(provider, "organizer", None)
    if organizer is not None:
        organizer.model = model
        organizer.core = getattr(model, "core", model)
        organizer.provider = provider


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load an R13-5A or R13-5B checkpoint and its selected-TRAIN provider."""
    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Maintained checkpoint loader did not return a mapping payload.")
    _identity, _provider_identity, adapter_state, recipe, arm, variant = _checkpoint_parts(payload)
    from channelthermal.training.supplement_member_routing_task import (
        build_thermal_supplement_member_routing_task,
    )

    evaluation_device = torch.device(device)
    _factory_model, provider = build_thermal_supplement_member_routing_task(
        recipe, arm=arm, device="cpu",
    )
    if _jsonable(provider.identity_payload()) != _jsonable(_provider_identity):
        raise ValueError("Saved supplement dataset, normalization or provider identity differs from its rebuilt recipe.")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("R13-5 supplement checkpoint has no model_state_dict.")
    adapter_type = variant["adapter"]
    model = adapter_type.from_state_payload(adapter_state, device="cpu")
    model.load_state_dict(model_state, strict=True, assign=True)
    model.to(evaluation_device)
    _bind_loaded_model(provider, model, evaluation_device)
    model.eval()
    payload = dict(payload)
    payload["checkpoint_path"] = str(checkpoint_path)
    payload["candidate"] = str(variant["candidate"])
    return model, provider, payload


def evaluate(checkpoint: str | Path, device: str, output: str | Path) -> dict:
    """Evaluate full/restricted native readers on canonical VAL89 and fixed detail cases."""
    started = time.perf_counter()
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError("Keep prior scientific evaluations; select a fresh supplement evaluation output.")
    model, provider, payload = load_checkpoint(checkpoint_path, device)
    if (len(provider.training_case_ids) != 600 or len(provider.validation_case_ids) != 89
            or provider.validation_scope != "canonical_VAL89_exposed"):
        raise ValueError("R13-5 endpoint evaluation requires full TRAIN600 and canonical exposed VAL89.")
    if "0003" not in provider.training_case_ids or any(
            case_id not in provider.validation_case_ids for case_id in ("0274", "0293", "0678")):
        raise ValueError("The fixed R13-5 detailed evaluation panel is absent from the saved partitions.")

    storage_recipe = provider.recipe
    output_root = Path(storage_recipe["storage"]["output_root"]).expanduser().resolve(strict=True)
    if output_path == output_root or not output_path.is_relative_to(output_root):
        raise ValueError(f"R13-5 evaluation output must be below saved DATA2 root {output_root}.")
    storage_options = {
        "storage_output_root": output_root,
        "output_budget_bytes": int(storage_recipe["evaluation"]["max_output_bytes"]),
        "payload_budget_bytes": int(storage_recipe["storage"]["max_output_bytes"]),
        "temporary_budget_bytes": int(storage_recipe["storage"]["max_peak_temporary_bytes"]),
        "minimum_free_headroom_bytes": int(storage_recipe["storage"]["minimum_free_headroom_bytes"]),
    }
    _storage_snapshot(output_root, **storage_options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    storage = _storage_snapshot(output_path.parent, **storage_options)
    output_path.mkdir(parents=True)
    modes = {}
    for mode in ("full", "restricted"):
        mode_dir = output_path / mode
        mode_dir.mkdir()
        with model.core.read_execution(mode):
            validation = _evaluate_native_split(
                model, provider, payload, "validation", provider.validation_case_ids,
                export_ids={"0274", "0293", "0678"},
            )
            train = _evaluate_native_split(
                model, provider, payload, "train", ("0003",), export_ids={"0003"},
            )
        for split in (validation, train):
            for case_id, arrays in split.pop("representative_arrays").items():
                np.savez_compressed(mode_dir / f"case_{case_id}.npz", **arrays)
        modes[mode] = {"validation": validation, "train_detail": train}

    result = {
        "schema": "thermal_r13_5_supplement_native_evaluation_v1",
        "status": "completed",
        "candidate": str(payload["candidate"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "arm": model.architecture,
        "retrieval_calibration": model.core.retrieval_calibration,
        "score_calibration": model.core.score_calibration,
        "joint_member_coverage": model.core.joint_member_coverage,
        "training_membership": provider.recipe["training"]["membership"],
        "training_case_count": len(provider.training_case_ids),
        "partition_scope": provider.validation_scope,
        "validation_panel": "all 89 exposed canonical VAL cases; nine native observables report mean, p90, worst and per-M strata",
        "executions": modes,
        "fixed_detail_panel": {"TRAIN": ["0003"], "VAL": ["0274", "0293", "0678"]},
        "fit_epochs": 0,
        "optimizer_updates": 0,
        "solver_attempts": 0,
        "TEST_access": False,
        "inverse_tested": False,
        "storage": storage,
        "process_seconds": time.perf_counter() - started,
    }
    result_path = output_path / "result.json"
    result_path.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    actual_bytes = sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file())
    if actual_bytes > int(storage_recipe["evaluation"]["max_output_bytes"]):
        raise RuntimeError("R13-5 evaluation output exceeded its declared byte budget.")
    result["storage"]["actual_output_bytes"] = actual_bytes
    result_path.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


__all__ = [
    "SUPPLEMENT_PROVIDER_SCHEMA", "load_checkpoint", "evaluate",
]
