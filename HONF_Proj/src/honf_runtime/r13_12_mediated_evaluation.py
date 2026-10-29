"""Native, original-record evidence for the opt-in R13-12 P/G readers.

The maintained Thermal provider owns normalization, native extraction, masks,
objectives and the nine-observable reduction. This module only chooses fixed
workloads, invokes that path and saves its actual tensors and measurements.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from .contextual_frontier_evaluation import (
    _known_module_counts, _summarize_nine_observables, load_contextual_checkpoint,
)
from .member_routing_evaluation import _jsonable, _native_arrays_with_residuals


READ_MODES = ("full", "all_e95", "source95_all", "restricted")
DETAIL_CASES = ("0274", "0293", "0678")
ORGANIZATION_CASES = ("0003", "0678", "0274", "0293")
QUERY_KEY = "thermal_native_fluid_grid_linspace16x32_v1"
_yy, _xx = np.meshgrid(np.linspace(0, 63, 16, dtype=np.int64),
                      np.linspace(0, 127, 32, dtype=np.int64), indexing="ij")
QUERY_IDS = (_yy * 128 + _xx).reshape(-1)
LOW_HIGH_CASES = {"low_M": "0006", "high_M": "0231"}
# Original R13-8 native_24case_memory_qualification panel, retained unchanged.
TRAIN24_CASES = ("0006", "0051", "0002", "0097", "0322", "0142", "0337", "0185",
                 "0348", "0231", "0015", "0054", "0303", "0099", "0327", "0146",
                 "0342", "0186", "0357", "0236", "0016", "0060", "0310", "0102")
_MAX_OUTPUT_BYTES = 512 * 1024**2


def _array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _stats(values: Any) -> dict[str, Any]:
    array = _array(values).astype(np.float64, copy=False).reshape(-1)
    if not array.size:
        return {"count": 0, "mean": None, "min": None, "median": None, "p90": None, "max": None}
    return {"count": int(array.size), "mean": float(array.mean()), "min": float(array.min()),
            "median": float(np.median(array)), "p90": float(np.quantile(array, .9)), "max": float(array.max())}


def _modes(model: Any, modes: Sequence[str]) -> tuple[str, ...]:
    result = tuple(modes)
    permitted = READ_MODES if getattr(model.core, "mediated_mode", "none") != "none" else ("full", "restricted")
    if not result or len(result) != len(set(result)) or any(mode not in permitted for mode in result):
        raise ValueError(f"Unique execution modes must be drawn from {permitted}.")
    return tuple(mode for mode in permitted if mode in result)


def _output_directory(output: str | Path) -> Path:
    path = Path(output).expanduser().resolve()
    root = Path("/data2/wanglz/ModularDT").resolve()
    if not os.path.ismount("/data2") or not path.is_relative_to(root):
        raise ValueError("New evaluation payloads must use the mounted /data2/wanglz/ModularDT root.")
    if path.exists():
        raise FileExistsError(path)
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_dev != Path("/data2").stat().st_dev:
        raise ValueError("Evaluation output resolves outside the DATA2 filesystem.")
    if shutil.disk_usage(parent).free - _MAX_OUTPUT_BYTES < 100 * 1024**3:
        raise OSError("Evaluation reservation would violate 100 GiB DATA2 headroom.")
    path.mkdir(parents=True)
    return path


def _identity(checkpoint: Any, payload: Mapping[str, Any], model: Any, device: Any) -> dict[str, Any]:
    return {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_epoch": int(payload["epoch"]),
            "architecture": model.architecture, "device": str(device), "fit_epochs": 0,
            "normalization": "literal checkpoint selected-TRAIN adapter state; no refit",
            "mediated_mode": getattr(model.core, "mediated_mode", "none"),
            "source_retained_mass": getattr(model.core, "source_retained_mass", None),
            "environment_retained_mass": getattr(model.core, "environment_retained_mass", None)}


def _save_result(directory: Path, result: Mapping[str, Any]) -> None:
    result = dict(result)
    result["storage"] = {"payload_bytes_before_result": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
                         "free_after_bytes": shutil.disk_usage(directory).free,
                         "maximum_output_bytes": _MAX_OUTPUT_BYTES}
    if result["storage"]["payload_bytes_before_result"] > _MAX_OUTPUT_BYTES:
        raise OSError("Bounded evaluation exceeded its declared output allowance.")
    temporary = directory / "result.json.tmp"
    temporary.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "result.json")


def _native_prediction(model: Any, prepared: Any, heat: Any, mode: str, **controls: Any):
    from channelthermal.training.content_hypergraph_task import ThermalContentPredictions
    native, counts, _ = model.apply_native_with_work_counts(
        prepared, heat, intervention={"read_mode": mode, **controls},
    )
    return ThermalContentPredictions(prepared, native, work=counts)


def _reduce(provider: Any, records: Sequence[Mapping[str, Any]], case_ids: Sequence[str]) -> dict[str, Any]:
    reduced = provider.reduce_native_evaluation_metrics(records, split="validation", case_keys=case_ids)
    rows = reduced["per_case"]
    if tuple(str(row["case_id"]) for row in rows) != tuple(case_ids):
        raise ValueError("Native reduction changed the fixed case order.")
    return {"case_ids": list(case_ids), "case_count": len(case_ids),
            "metrics": _summarize_nine_observables(rows, known_module_counts=_known_module_counts(provider)),
            "maintained_native_reduction": reduced}


def _fixed_detail_batches(provider: Any, case_ids: Sequence[str]):
    from channelthermal.training.content_hypergraph_task import THERMAL_CONTENT_THETA_COUNT
    for case in provider._load_native_cases(case_ids):
        cid = str(case["case_id"])
        yield provider._native_case_batch([case], (0,), split="fixed_detail_only", ntheta=THERMAL_CONTENT_THETA_COUNT)


def evaluate_native(checkpoint: str | Path, device: str, output: str | Path, *,
                    modes: Sequence[str] = READ_MODES, case_ids: Sequence[str] | None = None,
                    detail_case_ids: Sequence[str] = DETAIL_CASES) -> dict[str, Any]:
    """All nine native means/p90 on exactly the checkpoint VAL22 or VAL89.

    Detailed scenes outside that membership are exported and reported separately;
    they never enter the statistical validation aggregate.
    """
    model, provider, payload = load_contextual_checkpoint(checkpoint, device)
    modes = _modes(model, modes)
    ids = tuple(map(str, provider.validation_case_ids if case_ids is None else case_ids))
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(map(str, provider.validation_case_ids)):
        raise ValueError("Statistical native cases must be unique checkpoint validation members.")
    details = tuple(map(str, detail_case_ids))
    if any(provider.case_split_by_id.get(cid) != "test" for cid in details):
        raise ValueError("Detailed validation illustrations must retain their original test partition.")
    directory = _output_directory(output)
    result = {"schema": "honf_r13_12_native_evaluation_v1", "status": "running",
              **_identity(checkpoint, payload, model, device), "executions": {},
              "partition": "checkpoint_validation_membership" if case_ids is None else "explicit_validation_subset",
              "detail_panel": list(details), "complete_same_weight_modes": list(modes)}
    records = {mode: [] for mode in modes}
    counts = {mode: {} for mode in modes}
    started = time.perf_counter()
    model.eval()
    model.core.set_organization_clock(int(payload["epoch"]), 0)
    for mode in modes:
        (directory / mode).mkdir()
    with torch.no_grad():
        for batch in provider.native_evaluation_batches(split="validation", case_keys=ids, microbatch_size=1):
            cid = str(batch.case_keys[0])
            prepared = provider._prepare_for_probe(model, batch)
            for mode in modes:
                prediction = _native_prediction(model, prepared, batch.receivers.heat, mode)
                records[mode].append(provider.validation_metrics(prediction, batch.targets, None))
                for name, value in {**prepared.organization.work_counts, **prediction.work}.items():
                    counts[mode][name] = counts[mode].get(name, 0) + int(value)
                if cid in details:
                    np.savez_compressed(directory / mode / f"case_{cid}.npz",
                                        **_native_arrays_with_residuals(prediction, batch, cid))
            print(json.dumps({"task": "native", "case": cid, "completed_cases": len(records[modes[0]]),
                              "total_cases": len(ids)}), flush=True)
        for mode in modes:
            result["executions"][mode] = _reduce(provider, records[mode], ids)
            result["executions"][mode]["executed_work_counts"] = counts[mode]
        missing = tuple(cid for cid in details if cid not in ids)
        if missing:
            extra = {mode: [] for mode in modes}
            for batch in _fixed_detail_batches(provider, missing):
                cid = str(batch.case_keys[0])
                prepared = provider._prepare_for_probe(model, batch)
                for mode in modes:
                    prediction = _native_prediction(model, prepared, batch.receivers.heat, mode)
                    extra[mode].append(provider.validation_metrics(prediction, batch.targets, None))
                    np.savez_compressed(directory / mode / f"case_{cid}.npz",
                                        **_native_arrays_with_residuals(prediction, batch, cid))
            for mode in modes:
                result["executions"][mode]["detail_only_outside_validation_membership"] = _reduce(provider, extra[mode], missing)
    result.update(status="completed", seconds=time.perf_counter() - started)
    _save_result(directory, result)
    return result


def _flatten_arrays(value: Any, name: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    """Save ragged trace tuples with explicit stable paths, never object arrays."""
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        arrays[name] = _array(value)
    elif isinstance(value, Mapping):
        for key, part in value.items():
            _flatten_arrays(part, f"{name}_{key}", arrays, metadata)
    elif isinstance(value, (tuple, list)):
        metadata[name + "_length"] = len(value)
        for index, part in enumerate(value):
            _flatten_arrays(part, f"{name}_{index}", arrays, metadata)
    elif value is None or isinstance(value, (str, bool, int, float)):
        metadata[name] = value


def summarize_mediated_trace(trace: Mapping[str, Any], *, module_count: int,
                             full_environment_mass: Any | None = None) -> dict[str, Any]:
    """Separate source mass, admitted path mass, executed values and raw beta."""
    rid = _array(trace["anchor_read_member_read_index"]).astype(int)
    member = _array(trace["anchor_read_member_index"]).astype(int)
    reads = _array(trace["anchor_read_indices"]).astype(int)
    own = _array(trace["anchor_read_E_slots"]).astype(int)
    prior = _array(trace["anchor_read_access_prior"])
    prob = _array(trace["anchor_read_probabilities"])
    strength = _array(trace["anchor_read_strength"])
    is_module = member < module_count
    count = len(reads)
    counts_m = np.bincount(rid[is_module], minlength=count)
    counts_e = np.bincount(rid[~is_module], minlength=count)
    retained_e_mass = np.bincount(rid[~is_module], weights=prior[~is_module], minlength=count)
    own_hit = member == module_count + own[rid]
    own_beta = np.bincount(rid[own_hit], weights=(prob * strength[rid, 1])[own_hit], minlength=count)
    out = {"anchor_read_count": count, "admitted_M_count_per_anchor_read": _stats(counts_m),
           "admitted_E_count_per_anchor_read": _stats(counts_e),
           "zero_M_source_fraction": float(np.mean(counts_m == 0)),
           "zero_M_value_strength_fraction": float(np.mean(strength[:, 0] == 0)),
           "M_value_strength": _stats(strength[:, 0]), "E_value_strength": _stats(strength[:, 1]),
           "final_own_E_beta_before_locator_blend": _stats(own_beta),
           "retained_E_prior_mass": _stats(retained_e_mass),
           "beta_semantics": "rho times member probability, with locator blend in coalesced coefficients; not causal attribution"}
    for role in ("M", "E"):
        if f"member_weights_{role}" in trace:
            beta = _array(trace[f"member_weights_{role}"])
            out[f"executed_positive_value_{role}_count_per_query"] = _stats((beta > 0).sum(-1))
            out[f"raw_beta_{role}_sum_per_query"] = _stats(beta.sum(-1))
    if "mediated_source_alpha" in trace:
        alpha = _array(trace["mediated_source_alpha"])
        selected = _array(trace["mediated_source_selected"]).astype(bool)
        environment = _array(trace["mediated_environment_prior"])
        module_mass = alpha[:, 1:].sum(-1)
        retained_m = (alpha[:, 1:] * selected[:, 1:]).sum(-1)
        empty_mass = np.zeros(count)
        selected_empty_mass = np.zeros(count)
        raw_rows = []
        slots = trace.get("mediated_conditional_module_slots")
        for group_index, (group_reads, conditional, raw_mass) in enumerate(trace["mediated_conditional_rows"]):
            group_reads = _array(group_reads).astype(int)
            raw_mass = _array(raw_mass)
            raw_rows.extend(raw_mass.tolist())
            columns = np.arange(len(raw_mass)) if slots is None else _array(slots[group_index]).astype(int)
            empty_mass[group_reads] = (alpha[np.ix_(group_reads, columns + 1)] * (raw_mass <= 0)).sum(-1)
            selected_empty_mass[group_reads] = (alpha[np.ix_(group_reads, columns + 1)] *
                selected[np.ix_(group_reads, columns + 1)] * (raw_mass <= 0)).sum(-1)
        out.update(source_alpha0=_stats(alpha[:, 0]),
                   source_alpha_retained=_stats((alpha * selected).sum(-1)),
                   conditional_M_mass_retained=_stats(np.divide(retained_m, module_mass, out=np.zeros_like(retained_m), where=module_mass > 0)),
                   source_mass_assigned_to_empty_ME_rows=_stats(empty_mass), raw_ME_row_mass=_stats(raw_rows),
                   selected_source_mass_assigned_to_empty_ME_rows=_stats(selected_empty_mass),
                   composed_E_mass_after_source_masking=_stats(environment.sum(-1)),
                   source_alpha0_vs_own_E_beta_correlation=(float(np.corrcoef(alpha[:, 0], own_beta)[0, 1])
                       if count > 1 and np.std(alpha[:, 0]) > 0 and np.std(own_beta) > 0 else None))
        if full_environment_mass is not None:
            full_mass = _array(full_environment_mass)
            out["omitted_E_path_mass_due_to_source_masking"] = _stats(full_mass - environment.sum(-1))
            out["retained_E_path_mass_relative_to_full_composition"] = _stats(
                np.divide(retained_e_mass, full_mass, out=np.zeros_like(retained_e_mass), where=full_mass > 0))
    return out


def _scene_arrays(scene: Any) -> dict[str, np.ndarray]:
    names = ("module_ids", "module_coordinates", "module_mask", "module_measure",
             "environment_ids", "environment_coordinates", "environment_mask", "environment_measure",
             "coordinate_origin", "coordinate_scale", "module_features", "environment_features")
    return {name: _array(getattr(scene, name))[0] for name in names if getattr(scene, name, None) is not None}


def evaluate_organization(checkpoint: str | Path, device: str, output: str | Path, *,
                          case_ids: Sequence[str] = ORGANIZATION_CASES,
                          modes: Sequence[str] = READ_MODES) -> dict[str, Any]:
    """Save both actual contextual cycles and the same fixed 512 original queries."""
    from channelthermal.e_anchor_locator import attach_thermal_e_anchor_query
    from honf_forward_core.content_hypergraph.contracts import QueryBatch
    from honf_forward_core.content_hypergraph.query_chunks_v1 import slice_query_v1
    model, provider, payload = load_contextual_checkpoint(checkpoint, device)
    modes = _modes(model, modes)
    directory = _output_directory(output)
    result = {"schema": "honf_r13_12_original_member_evidence_v1", "status": "running",
              **_identity(checkpoint, payload, model, device), "query_key": QUERY_KEY,
              "query_ids": QUERY_IDS.tolist(), "cases": [],
              "semantics": "MM/EM/EE/ME are actual directed exchange stages; alpha D/B are later routing compositions; geometry is an explicit prior"}
    model.eval()
    model.core.set_organization_clock(int(payload["epoch"]), 0)
    with torch.no_grad():
        for case in provider._load_native_cases(case_ids):
            cid = str(case["case_id"])
            if provider.case_split_by_id.get(cid) != ("train" if cid == "0003" else "test"):
                raise ValueError("Fixed organization illustration changed original partition.")
            structure = {key: torch.as_tensor(value, device=device, dtype=torch.float32)[None]
                         for key, value in case["structure"].items()}
            scene = model.scene_batch(structure, scene_key=(cid,))
            x, y = np.asarray(case["x_grid"]), np.asarray(case["y_grid"])
            if x.ndim == 1:
                x, y = np.meshgrid(x, y, indexing="xy")
            if x.shape != (64, 128) or y.shape != x.shape:
                raise ValueError("Fixed 512-query panel requires the original 64 by 128 grid.")
            xy = np.stack((x.reshape(-1), y.reshape(-1)), -1)[QUERY_IDS]
            query = QueryBatch(coordinates=torch.as_tensor(xy, dtype=torch.float32, device=device)[None],
                               query_ids=torch.as_tensor(QUERY_IDS, device=device)[None],
                               query_mask=torch.ones((1, len(QUERY_IDS)), dtype=torch.bool, device=device),
                               scene_key=scene.scene_key, coordinate_frame_key=scene.coordinate_frame_key,
                               receiver_key=(QUERY_KEY, cid))
            query = attach_thermal_e_anchor_query(query, scene, model.core.locator_mode,
                         environment_nx=model.environment_nx, environment_ny=model.environment_ny)
            requests = [f"cycle{cycle}/{route}_attention" for cycle in range(model.core.context_depth)
                        for route in ("mm", "em", "ee", "me")]
            organization = model.core.prepare(scene, return_trace=True,
                         trace_request={"include": requests, "max_elements": 2_000_000})
            arrays = {**_scene_arrays(scene), "query_ids": QUERY_IDS.copy(), "query_xy": xy,
                      "anchor_indices": _array(query.anchor_indices)[0], "anchor_weights": _array(query.anchor_weights)[0]}
            metadata = {"case_id": cid, "original_partition": provider.case_split_by_id[cid],
                        "scope": "TRAIN M1 routing illustration, not held-out fidelity" if cid == "0003" else "input-selected validation illustration",
                        "module_count": int(scene.module_mask.sum()), "stages": [], "modes": {}}
            stages = getattr(organization, "applied_context_stages", None) or getattr(organization, "context_stage_traces", ())
            for cycle, stage in enumerate(stages):
                for route in ("mm", "em", "ee", "me"):
                    snapshot = stage.get(route + "_native_route_snapshot")
                    prefix = f"cycle{cycle}_{route}"
                    _flatten_arrays(stage.get(route + "_applied_incidence"), prefix + "_applied", arrays, metadata)
                    if snapshot is not None:
                        metadata["stages"].append({"cycle": cycle, "route": route,
                                                  "stage_identity": snapshot.stage_identity})
                        for name in ("raw_scores", "geometry_bias", "allowed", "donor_measure",
                                     "applied_message_mass", "donor_ids", "destination_ids"):
                            _flatten_arrays(getattr(snapshot, name, None), prefix + "_" + name, arrays, metadata)
            _flatten_arrays(organization.trace or {}, "prepare", arrays, metadata)
            _flatten_arrays(getattr(organization, "anchored_route_provenance", {}), "route_provenance", arrays, metadata)
            # Original identities and measures above are the crosswalk for every
            # stage's receiver/donor slot axes, including genuine final ME.
            for start in range(0, len(QUERY_IDS), model.core.query_tile_size):
                stop = min(start + model.core.query_tile_size, len(QUERY_IDS))
                tile = slice_query_v1(query, start, stop)
                full_mass = None
                for mode in modes:
                    decoded = model.core.decode(organization, tile, intervention={"read_mode": mode}, return_trace=True)
                    trace = decoded.trace
                    prefix = f"{mode}_tile{start}"
                    _flatten_arrays(trace, prefix, arrays, metadata)
                    if mode == "full" and "mediated_environment_prior" in trace:
                        full_mass = _array(trace["mediated_environment_prior"]).sum(-1)
                    metadata["modes"].setdefault(mode, []).append({"query_start": start, "query_stop": stop,
                        "work_counts": decoded.work_counts, "summary": summarize_mediated_trace(trace,
                              module_count=scene.num_modules, full_environment_mass=full_mass)})
            np.savez_compressed(directory / f"case_{cid}.npz", **arrays)
            (directory / f"case_{cid}.json").write_text(json.dumps(_jsonable(metadata), indent=2, allow_nan=False) + "\n")
            result["cases"].append({"case_id": cid, "arrays": f"case_{cid}.npz", "metadata": f"case_{cid}.json"})
            print(json.dumps({"task": "organization", "case": cid, "arrays": len(arrays)}), flush=True)
    result["status"] = "completed"
    _save_result(directory, result)
    return result


def _observation_vectors(prepared: Any, batch: Any, native: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    targets = batch.targets
    field = native["pred_field"]
    fluid = (targets.point_weights.to(field) > 0) & prepared.stencils["fluid"].valid
    out = {name: field[..., i][fluid & torch.isfinite(targets.field_targets[..., i])]
           for i, name in enumerate(("u", "v", "p", "omega", "temperature_fluid"))}
    present = prepared.source_present > 0
    interface = native["pred_interface"]
    shape = interface.shape[:-1]
    surface = prepared.stencils["surface"].valid.reshape(shape)
    outside = prepared.stencils["outside"].valid.reshape(shape)
    observed = targets.interface_valid_mask > .5
    smask = present[..., None] & surface & observed & torch.isfinite(targets.interface_target[..., 0])
    qmask = present[..., None] & surface & outside & observed & torch.isfinite(targets.interface_target[..., 1])
    out["temperature_surface"] = interface[..., 0][smask]
    out["q_proxy"] = interface[..., 1][qmask]
    material = native["pred_internal_temperature"][..., 0]
    valid = present[..., None] & prepared.stencils["material"].valid.reshape(targets.material_targets.shape)
    out["temperature_material"] = material[valid & torch.isfinite(targets.material_targets)]
    return out


def _prediction_delta(base: Mapping[str, torch.Tensor], trial: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {name: {"elements": int(value.numel()),
                   "rms": float(value.square().mean().sqrt()) if value.numel() else None,
                   "max_abs": float(value.abs().max()) if value.numel() else None}
            for name in base for value in (trial[name] - base[name],)}


def evaluate_interventions(checkpoint: str | Path, device: str, output: str | Path, *,
                           case_ids: Sequence[str] = DETAIL_CASES,
                           modes: Sequence[str] = ("full", "restricted"), collective: bool = True) -> dict[str, Any]:
    """Fixed-context route/value removals and a fixed region set/member probe."""
    from .r13_8_anchored_organization_evaluation import _native_metric_summary
    model, provider, payload = load_contextual_checkpoint(checkpoint, device)
    if getattr(model.core, "mediated_mode", "none") == "none":
        raise ValueError("M-mediated route interventions require a P/G checkpoint.")
    modes = _modes(model, modes)
    if any(provider.case_split_by_id.get(str(cid)) != "test" for cid in case_ids):
        raise ValueError("Intervention scenes must retain their original validation/test partition.")
    directory = _output_directory(output)
    result = {"schema": "honf_r13_12_fixed_context_interventions_v1", "status": "running",
              **_identity(checkpoint, payload, model, device), "cases": [],
              "scope": "Native read interventions with contextual states fixed and legal probabilities/strengths recomputed; not physical causality",
              "collective_region": "positive-measure original E coordinates in inclusive [0.375,0.625] of each domain span; structural own E retained"}
    model.eval()
    with torch.no_grad():
        for batch in _fixed_detail_batches(provider, case_ids):
            cid = str(batch.case_keys[0])
            prepared = provider._prepare_for_probe(model, batch)
            scene = prepared.scene_batch
            unit = (scene.environment_coordinates[0] - scene.coordinate_origin[0]) / scene.coordinate_scale[0]
            mask = scene.environment_mask[0] & (scene.environment_measure[0] > 0)
            mask &= (unit >= .375).all(-1) & (unit <= .625).all(-1)
            region = tuple(map(int, scene.environment_ids[0][mask].cpu().tolist()))
            row = {"case_id": cid, "region_original_E_ids": list(region), "modes": {}}
            for mode in modes:
                base = _native_prediction(model, prepared, batch.receivers.heat, mode)
                metrics = _native_metric_summary(provider, batch, base, None)
                base_means = {name: float(value["mean"]) for name, value in metrics.items()}
                vectors = _observation_vectors(prepared, batch, base.native_main)
                def trial(controls):
                    prediction = _native_prediction(model, prepared, batch.receivers.heat, mode, **controls)
                    measured = _native_metric_summary(provider, batch, prediction, None)
                    means = {name: float(value["mean"]) for name, value in measured.items()}
                    vec = _observation_vectors(prepared, batch, prediction.native_main)
                    return {"rmse": means, "delta_rmse": {name: means[name] - base_means[name] for name in means},
                            "prediction_change": _prediction_delta(vectors, vec)}, vec
                route, _ = trial({"remove_module_mediated_environment": True})
                values, _ = trial({"suppress_module_values": True})
                item = {"baseline_rmse": base_means, "remove_M_mediated_E": route, "suppress_M_values": values}
                if collective:
                    singles, deltas = [], {name: torch.zeros_like(value) for name, value in vectors.items()}
                    for identity in region:
                        measurement, vec = trial({"exclude_environment_ids": [identity]})
                        singles.append({"original_E_id": identity, **measurement})
                        for name in deltas:
                            deltas[name] += vec[name] - vectors[name]
                    group, vec = trial({"exclude_environment_ids": region})
                    group_delta = {name: vec[name] - vectors[name] for name in vectors}
                    zeros = {name: torch.zeros_like(value) for name, value in vectors.items()}
                    residual = {name: group_delta[name] - deltas[name] for name in deltas}
                    item["collective_probe"] = {"individual": singles, "set": group,
                        "sum_individual_prediction_change": _prediction_delta(zeros, deltas),
                        "set_minus_sum_individual_prediction_change": _prediction_delta(zeros, residual)}
                row["modes"][mode] = item
            result["cases"].append(row)
            _save_result(directory, result)
            print(json.dumps({"task": "interventions", "case": cid}), flush=True)
    result["status"] = "completed"
    _save_result(directory, result)
    return result


def _move(value: Any, device: Any) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if is_dataclass(value):
        return replace(value, **{field.name: _move(getattr(value, field.name), device) for field in fields(value)})
    if isinstance(value, Mapping):
        return {key: _move(part, device) for key, part in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(part, device) for part in value)
    if isinstance(value, list):
        return [_move(part, device) for part in value]
    return value


def _fixed_training_batch(provider: Any):
    """Use original benchmark inputs without changing fitted membership/state."""
    from .unified_training import SamplingKey
    ids = TRAIN24_CASES
    if any(provider.case_split_by_id.get(cid) != "train" for cid in ids):
        raise ValueError("The inherited cost panel must retain original TRAIN ownership.")
    cases = provider._load_native_cases(ids)
    old_cases, old_indices = provider.training_cases, provider._train_index_by_id
    temporary = list(old_cases)
    if len(temporary) < len(ids):
        raise ValueError("The fixed TRAIN24 benchmark requires a fitted population of at least 24 cases.")
    temporary[:len(ids)] = cases
    provider.training_cases = temporary
    provider._train_index_by_id = {cid: index for index, cid in enumerate(ids)}
    key = SamplingKey(seed=0, epoch=1, update_index=0, microbatch_index=0,
                      stage="joint", arm=provider.training_sampling_arm,
                      sampling_version=provider.sampling_version, dataset_id=provider.sampling_dataset_id)
    try:
        return provider.make_batch(ids, key)
    finally:
        provider.training_cases, provider._train_index_by_id = old_cases, old_indices


def _workload_description(batch: Any) -> dict[str, Any]:
    present = batch.scene_inputs.structure["module_present"]
    return {"case_ids": list(map(str, batch.case_keys)),
            "fluid_xy_shape": list(batch.receivers.fluid_xy.shape),
            "local_query_points_shape": list(batch.receivers.local_query_points.shape),
            "physical_heat_shape": list(batch.receivers.heat.shape),
            "physical_module_counts": _array((present > .5).sum(-1)).astype(int).tolist()}


def benchmark_native(checkpoint: str | Path, device: str, output: str | Path, *,
                     warmups: int = 2, repeats: int = 5,
                     modes: Sequence[str] = ("full", "restricted"), cpu_smoke: bool = False) -> dict[str, Any]:
    """Complete host-to-host native, prepared decode, and paired backward cost.

    Model/provider load, disk reads, JSON writing and optimizer updates are not
    timed. Complete inference includes host/device movement, preparation,
    organization, decoding, native extraction and output movement to CPU.
    """
    from .unified_training import capture_rng_state, restore_rng_state
    if warmups < 0 or repeats < 1:
        raise ValueError("Cost measurement needs nonnegative warmups and positive repeats.")
    resolved_device = torch.device(device)
    if cpu_smoke and resolved_device.type != "cpu":
        raise ValueError("CPU smoke is only valid on CPU.")
    model, provider, payload = load_contextual_checkpoint(checkpoint, "cpu")
    modes = _modes(model, modes)
    directory = _output_directory(output)
    hosts = {}
    for name, cid in LOW_HIGH_CASES.items():
        if provider.case_split_by_id.get(cid) != "train":
            raise ValueError("Native cost cases must remain original TRAIN inputs.")
        batch = next(_fixed_detail_batches(provider, (cid,)))
        if cpu_smoke:
            batch = replace(batch, receivers=replace(batch.receivers, fluid_xy=batch.receivers.fluid_xy[:, :8]))
        hosts[name] = batch
    if cpu_smoke:
        # Preserve the numerical path; only this explicitly labelled API smoke
        # uses one sampled case rather than the complete measured workload.
        cid = str(provider.training_case_ids[0])
        from .unified_training import SamplingKey
        key = SamplingKey(seed=0, epoch=1, update_index=0, microbatch_index=0,
                          stage="joint", arm=provider.training_sampling_arm,
                          sampling_version=provider.sampling_version, dataset_id=provider.sampling_dataset_id)
        hosts["train_sampled_smoke"] = provider.make_batch((cid,), key)
    else:
        hosts["train24_sampled"] = _fixed_training_batch(provider)
    model.to(resolved_device)
    provider.device = resolved_device
    result = {"schema": "honf_r13_12_native_cost_v1", "status": "running",
              **_identity(checkpoint, payload, model, device), "warmups": warmups, "repeats": repeats,
              "cpu_smoke_not_benchmark": cpu_smoke, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "scope": "complete host-to-host inference includes H2D, prepare, decode, extraction and D2H; prepared decode reuses resident preparation; paired backward includes prepare, both native physical objectives and backward, with no optimizer update",
              "low_high_cases": LOW_HIGH_CASES, "training_case_ids": list(TRAIN24_CASES),
              "workload_inputs": {name: _workload_description(batch) for name, batch in hosts.items()},
              "training_query_recipe": {"sampling_version": provider.sampling_version,
                  "sampling_dataset_id": provider.sampling_dataset_id, "seed": 0,
                  "checkpoint_training_seed": provider.seed,
                  "sampling_arm": provider.training_sampling_arm, "epoch": 1, "primary_queries": provider.primary_queries},
              "precision": str(next(model.parameters()).dtype), "scenarios": {}}
    rng = capture_rng_state()
    def sync():
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
    def measured(host, mode, *, prepared=None, backward=False):
        model.train(backward)
        model.zero_grad(set_to_none=True)
        sync()
        if resolved_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved_device)
            allocated = torch.cuda.memory_allocated(resolved_device)
            reserved = torch.cuda.memory_reserved(resolved_device)
        else:
            allocated = reserved = None
        start = time.perf_counter()
        with torch.set_grad_enabled(backward):
            batch = _move(host, resolved_device) if prepared is None else host
            state = provider._prepare_for_probe(model, batch) if prepared is None else prepared
            if backward:
                pair = provider._paired_predictions(state, batch.receivers.heat)
                loss = sum(sum(float(provider.native_weights[name]) * values.mean()
                               for name, values in provider._role_losses(prediction, batch.targets).items())
                           for prediction in (pair.full, pair.restricted)) * .5
                loss.backward()
                cpu_outputs = _move((pair.full.native_main, pair.restricted.native_main, loss.detach()), "cpu")
                work = pair.work
            else:
                native, work, _ = model.apply_native_with_work_counts(state, batch.receivers.heat,
                                                                     intervention={"read_mode": mode})
                if prepared is None:
                    work = {**{f"core_prepare_{name}": value for name, value in state.organization.work_counts.items()},
                            **work}
                cpu_outputs = _move(native, "cpu") if prepared is None else native
        sync()
        seconds = time.perf_counter() - start
        del cpu_outputs
        memory = ({"allocated_baseline_bytes": allocated,
                   "allocated_peak_bytes": torch.cuda.max_memory_allocated(resolved_device),
                   "allocated_increment_bytes": torch.cuda.max_memory_allocated(resolved_device) - allocated,
                   "reserved_baseline_bytes": reserved,
                   "reserved_peak_bytes": torch.cuda.max_memory_reserved(resolved_device),
                   "reserved_increment_bytes": torch.cuda.max_memory_reserved(resolved_device) - reserved}
                  if resolved_device.type == "cuda" else {"scope": "CUDA allocator memory unavailable on CPU"})
        model.zero_grad(set_to_none=True)
        return {"seconds": seconds, "memory": memory, "work_counts": work}
    def series(function):
        for _ in range(warmups):
            function()
        samples = [function() for _ in range(repeats)]
        return {"seconds": _stats([row["seconds"] for row in samples]), "samples": samples}
    try:
        for name, host in hosts.items():
            if name.startswith("train"):
                result["scenarios"][name] = {"paired_forward_backward": series(lambda: measured(host, "restricted", backward=True))}
            else:
                rows = {}
                for mode in modes:
                    rows[mode] = {"complete_native": series(lambda: measured(host, mode))}
                model.eval()
                with torch.no_grad():
                    batch = _move(host, resolved_device)
                    prepared = provider._prepare_for_probe(model, batch)
                    for mode in modes:
                        rows[mode]["prepared_decode_and_native_extraction"] = series(lambda: measured(batch, mode, prepared=prepared))
                    del prepared, batch
                result["scenarios"][name] = rows
            _save_result(directory, result)
            print(json.dumps({"task": "cost", "scenario": name}), flush=True)
        result["status"] = "completed"
        _save_result(directory, result)
    finally:
        restore_rng_state(rng)
        model.eval()
        model.zero_grad(set_to_none=True)
    return result
