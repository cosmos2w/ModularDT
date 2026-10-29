"""Native full-reference and independently executed restricted R13-4 evidence."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .guided_member_routing_training import load_checkpoint
from .member_routing_evaluation import (
    _evaluate_native_split, _jsonable, _query_indices, _storage_snapshot,
)


def evaluate(checkpoint, device, output):
    started = time.perf_counter()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Keep prior scientific evaluations; select a fresh output name.")
    model, provider, payload = load_checkpoint(checkpoint, device)
    recipe = provider.recipe
    storage = recipe["storage"]
    limits = _storage_snapshot(
        output.parent, storage_output_root=storage["output_root"],
        output_budget_bytes=recipe["evaluation"]["max_output_bytes"],
        payload_budget_bytes=storage["max_output_bytes"],
        temporary_budget_bytes=storage["max_peak_temporary_bytes"],
        minimum_free_headroom_bytes=storage["minimum_free_headroom_bytes"])
    if len(provider.training_case_ids) != 600 or len(provider.validation_case_ids) != 89:
        raise ValueError("Formal guided evaluation needs the original TRAIN600/VAL89 membership.")
    output.mkdir()
    modes = {}
    for mode in ("full", "restricted"):
        directory = output / mode
        directory.mkdir()
        with model.core.read_execution(mode):
            validation = _evaluate_native_split(
                model, provider, payload, "validation", provider.validation_case_ids,
                export_ids={"0274", "0293", "0678"})
            train = _evaluate_native_split(
                model, provider, payload, "train", ("0003",), export_ids={"0003"})
        for split in (validation, train):
            for case_id, arrays in split.pop("representative_arrays").items():
                np.savez_compressed(directory / f"case_{case_id}.npz", **arrays)
        modes[mode] = {"validation": validation, "train_detail": train}
    access = []
    for case_id in ("0003", "0274", "0293", "0678"):
        split = "train" if case_id == "0003" else "validation"
        batch = next(provider.native_evaluation_batches(split=split, case_keys=(case_id,), microbatch_size=1))
        with torch.no_grad(), model.core.read_execution("full"):
            prepared = provider._prepare_for_probe(model, batch)
            native, work, _ = model.apply_native_with_work_counts(prepared, batch.receivers.heat)
            from channelthermal.training.content_hypergraph_task import ThermalContentPredictions
            prediction = ThermalContentPredictions(prepared, native, work=work)
            indices = _query_indices(batch.receivers, prediction)
            queries = provider._query_subset(prepared.query_batch, indices)
            full = model.core.capture_interaction_routing_trace(prepared.organization, queries)
        with torch.no_grad(), model.core.read_execution("restricted"):
            restricted = model.core.capture_interaction_routing_trace(prepared.organization, queries)
        captured = (full["conditional_preference_E"] * restricted["chosen_candidate_mask_E"]).sum(-1)
        access.append({"case_id": case_id, "partition": split,
                       "native_query_indices": indices, "query_coordinates": queries.coordinates,
                       "full": full, "restricted": restricted,
                       "captured_full_conditional_E_mass": captured,
                       "interpretation": "Restricted trace independently recomputes candidates and member scores; full scores never enter its prediction."})
    identity = payload["experiment_identity"]
    result = {"schema": "thermal_guided_member_access_evaluation_v1", "status": "completed",
              "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_epoch": payload["epoch"],
              "arm": model.architecture, "parent_epoch": identity["initialization"]["source_parent_age"],
              "additional_epochs": payload["epoch"] - identity["initialization"]["source_parent_age"],
              "scientific_reconstruction_reference": "full", "deployed_guide_execution": "restricted",
              "executions": modes, "original_member_access": access,
              "no_old_absolute_mass_truncation": True, "fit_epochs": 0,
              "TEST_access": False, "new_solver_calls": 0, "inverse_tested": False,
              "storage": limits, "process_seconds": time.perf_counter() - started}
    (output / "result.json").write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n")
    actual = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if actual > recipe["evaluation"]["max_output_bytes"]:
        raise OSError("Native evidence exceeded the declared evaluation byte budget.")
    return {"status": "completed", "output": str(output), "actual_bytes": actual,
            "checkpoint_epoch": payload["epoch"], "process_seconds": result["process_seconds"]}
