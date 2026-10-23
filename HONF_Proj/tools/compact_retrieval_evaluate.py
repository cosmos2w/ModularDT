#!/usr/bin/env python3
"""Saved Thermal response replay and exact-panel summaries for compact retrieval."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "tools", PROJECT / "Case_ThermalChannel/src"):
    sys.path.insert(0, str(directory))

VAL89_IDS = tuple([f"{index:04d}" for index in range(274, 303)] +
                  [f"{index:04d}" for index in range(633, 693)])
TRAIN24_IDS = (
    "0006", "0020", "0060", "0066", "0142", "0162", "0186", "0207", "0254", "0303", "0310", "0329",
    "0345", "0348", "0358", "0382", "0416", "0463", "0464", "0474", "0529", "0558", "0601", "0610",
)
REPRESENTATIVE_IDS = {
    "validation": ("0274", "0658", "0293", "0692"),
    "train": ("0006", "0254"),
}
RESPONSE_FAMILY_IDS = ("0001", "0318", "0333", "0348")
W6_CANDIDATE_IDS = tuple(f"size_matched_random_u00_c{index:02d}" for index in range(6))
FIELD_PATHS = {
    "u": ("roles", "u", "rmse"),
    "v": ("roles", "v", "rmse"),
    "p": ("roles", "p", "rmse"),
    "omega": ("roles", "omega", "rmse"),
    "temperature_fluid": ("roles", "temperature_fluid", "rmse"),
    "temperature_surface": ("roles", "temperature_surface", "rmse"),
    "temperature_material": ("roles", "temperature_material", "rmse"),
    "q_proxy": ("q_proxy_rmse",),
}
CONFIG_ARM_LABELS = {
    "thermal_u_h_full5000_v1": "U-H",
    "thermal_u_s_full5000_v1": "U-S",
}


def validate_case_panel(case_ids: Sequence[Any], expected: Sequence[str] = VAL89_IDS) -> tuple[str, ...]:
    observed = tuple(map(str, case_ids))
    expected_ids = tuple(map(str, expected))
    if len(observed) != len(set(observed)):
        raise ValueError("A fixed evaluation panel cannot contain duplicate case IDs.")
    if len(expected_ids) != len(set(expected_ids)) or set(observed) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(observed))
        unexpected = sorted(set(observed) - set(expected_ids))
        raise ValueError(f"Evaluation panel differs from its sealed case set: missing={missing}, unexpected={unexpected}.")
    return observed


def _nested_value(row: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Metric row is missing the declared field path {tuple(path)}.")
        value = value[key]
    result = float(value)
    if not (result >= 0.0 and result < float("inf")):
        raise ValueError(f"Metric path {tuple(path)} must contain a finite nonnegative error.")
    return result


def _module_peak_rmse(row: Mapping[str, Any]) -> float:
    import numpy as np

    modules = row.get("module_peaks")
    if modules is not None:
        errors = [float(module["prediction"]) - float(module["target"]) for module in modules]
    else:
        endpoint = row.get("roles", {}).get("temperature_material_endpoint_maxima", {})
        modules = endpoint.get("per_module", ())
        errors = [float(module["predicted_max"]) - float(module["target_max"]) for module in modules]
    if not errors or not np.isfinite(errors).all():
        raise ValueError("A module-peak panel row requires aligned finite predicted and reference maxima.")
    return float(np.sqrt(np.mean(np.square(errors))))


def _field_value(row: Mapping[str, Any], name: str) -> float:
    if name == "module_peak_endpoint":
        return _module_peak_rmse(row)
    fields = row.get("fields")
    if isinstance(fields, Mapping) and name in fields:
        return _nested_value(fields[name], ("rmse",))
    return _nested_value(row, FIELD_PATHS[name])


def summarize_native_case_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Sequence[str] = VAL89_IDS,
) -> dict[str, Any]:
    """Summarize nine physical quantities by equal case and by input M stratum."""
    import numpy as np

    ids = validate_case_panel([str(row["case_id"]) for row in rows], expected_case_ids)
    by_id = {str(row["case_id"]): row for row in rows}
    ordered = [by_id[case_id] for case_id in ids]
    module_counts = {case_id: int(by_id[case_id]["module_count"]) for case_id in ids}
    if any(value < 1 for value in module_counts.values()):
        raise ValueError("Every native case must retain at least one physically present module.")

    summaries: dict[str, Any] = {}
    for field in (*FIELD_PATHS, "module_peak_endpoint"):
        values = [_field_value(row, field) for row in ordered]
        worst_index = max(range(len(values)), key=lambda index: (values[index], ids[index]))
        summaries[field] = {
            "case_count": len(values),
            "equal_case_mean": sum(values) / len(values),
            "equal_case_p90": float(np.quantile(values, 0.9)),
            "worst": values[worst_index],
            "worst_case_id": ids[worst_index],
            "unit": "native physical error; temperature and q-proxy use packed dataset native scales",
            "by_module_count": {},
        }
        for module_count in sorted(set(module_counts.values())):
            selected = [row for row in ordered if int(row["module_count"]) == module_count]
            selected_values = [_field_value(row, field) for row in selected]
            selected_ids = [str(row["case_id"]) for row in selected]
            local_worst = max(range(len(selected_values)), key=lambda index: (selected_values[index], selected_ids[index]))
            summaries[field]["by_module_count"][f"M{module_count}"] = {
                "case_count": len(selected_values),
                "equal_case_mean": sum(selected_values) / len(selected_values),
                "equal_case_p90": float(np.quantile(selected_values, 0.9)),
                "worst": selected_values[local_worst],
                "worst_case_id": selected_ids[local_worst],
            }
    return {
        "schema": "round12_thermal_native_case_summary_v1",
        "case_count": len(ids),
        "case_ids": list(ids),
        "module_count_strata": {
            f"M{count}": sum(value == count for value in module_counts.values())
            for count in sorted(set(module_counts.values()))
        },
        "fields": summaries,
    }


def _as_numpy(value: Any):
    import numpy as np
    import torch

    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _compact_group_version_stamp(compact: Any):
    """Refresh the maintained prepared-state stamp after a local intervention."""
    from honf_forward_core.native_retrieval.contracts import tensor_versions

    tensors = [
        compact.membership, compact.group_mask, compact.group_types,
        compact.controls, compact.centroids, compact.lengths, compact.extents,
        compact.pair_logits, compact.pair_mask, compact.pair_slots,
    ]
    for metadata in (compact.bank_group_indices, compact.bank_group_masks):
        if metadata is not None:
            tensors.extend(metadata)
    return tensor_versions(*tensors)


def prepare_copy_with_disabled_group(prepared: Any, *, batch_index: int, group_index: int):
    """Create a per-scene, copy-only one-group intervention.

    The prepared model state and its live compact groups remain untouched. The
    returned copy uses the normal compact-state version check and normal typed
    query scorer, so the remaining entmax weights are freshly normalized.
    """
    from dataclasses import replace

    organization = prepared.organization
    compact = getattr(organization, "compact_groups", None)
    if compact is None:
        raise TypeError("A compact prepared organization is required for a group intervention.")
    if compact.bank_group_indices is None or compact.bank_group_masks is None:
        raise ValueError("A compact group intervention requires the prepared typed bank indices and masks.")
    live = compact.group_mask
    if live.ndim != 2 or not 0 <= batch_index < live.shape[0]:
        raise ValueError("Intervention batch index is outside the compact group batch.")
    if not 0 <= group_index < live.shape[1] or not bool(live[batch_index, group_index].detach().cpu()):
        raise ValueError("The selected intervention group is not live in its scene.")
    if int(live[batch_index].sum().detach().cpu()) <= 1:
        raise ValueError("Disabling the only live compact group would leave no query bank.")

    changed_group_mask = live.clone()
    changed_group_mask[batch_index, group_index] = False
    changed_bank_masks = []
    for indices, bank_mask in zip(compact.bank_group_indices, compact.bank_group_masks, strict=True):
        changed_bank_masks.append(bank_mask.clone() & changed_group_mask.gather(1, indices))
    changed_compact = replace(
        compact,
        group_mask=changed_group_mask,
        bank_group_masks=tuple(changed_bank_masks),
    )
    changed_organization = copy.copy(organization)
    changed_organization.compact_groups = changed_compact
    changed_organization.compact_group_versions = _compact_group_version_stamp(changed_compact)
    changed_prepared = replace(prepared, organization=changed_organization)
    return changed_prepared, {
        "batch_index": int(batch_index),
        "disabled_group_index": int(group_index),
        "live_K_before": int(live[batch_index].sum().detach().cpu()),
        "live_K_after": int(changed_group_mask[batch_index].sum().detach().cpu()),
        "allocated_K_before": int(live.shape[1]),
        "allocated_K_after": int(changed_group_mask.shape[1]),
        "transition": "per-scene group disabled; allocated slots retained",
    }


def summarize_role_group_mass(trace: Mapping[str, Any], prepared: Any) -> dict[str, Any]:
    """Measure query-to-group mass through each maintained native role stencil."""
    import numpy as np

    group_weights = _as_numpy(trace["compact_group_weights"])
    group_mask = _as_numpy(trace["compact_group_mask"]).astype(bool)
    query_valid = _as_numpy(trace.get("query_valid_mask", np.ones(group_weights.shape[:2], bool))).astype(bool)
    if group_weights.ndim != 3 or group_mask.shape != (group_weights.shape[0], group_weights.shape[2]):
        raise ValueError("Role mass needs aligned [B,Q,K] group weights and [B,K] group masks.")
    if query_valid.shape != group_weights.shape[:2]:
        raise ValueError("Role mass query-valid mask differs from the query-group weights.")
    if not np.isfinite(group_weights).all() or (group_weights < 0).any():
        raise ValueError("Role mass requires finite nonnegative query-group weights.")
    batch_size, query_count, group_count = group_weights.shape
    if group_count != group_mask.shape[1]:
        raise ValueError("Role mass group axis differs from the live-group mask.")

    role_summaries: dict[str, Any] = {}
    role_means = []
    total_group_mass = np.zeros((group_count,), dtype=np.float64)
    total_receiver_count = 0
    for role in ("fluid", "surface", "outside", "material"):
        stencil = getattr(prepared, "stencils", {}).get(role)
        if stencil is None:
            continue
        indices = _as_numpy(stencil.indices).astype(np.int64)
        interpolation = _as_numpy(stencil.weights).astype(np.float64)
        stencil_valid = _as_numpy(stencil.valid).astype(bool)
        if indices.ndim != 3 or indices.shape[-1] != 4 or interpolation.shape != indices.shape:
            raise ValueError(f"Native role {role!r} must retain flattened [B,R,4] interpolation stencils.")
        if stencil_valid.shape != indices.shape[:-1] or indices.shape[0] != batch_size:
            raise ValueError(f"Native role {role!r} stencil batch/query axes do not align.")
        if ((indices < 0) | (indices >= query_count)).any():
            raise ValueError(f"Native role {role!r} stencil references a receiver outside the query union.")
        batch_indices = np.arange(batch_size)[:, None, None]
        cells_valid = query_valid[batch_indices, indices]
        valid = stencil_valid & (((interpolation == 0.0) | cells_valid).all(axis=-1))
        if role != "fluid":
            present = _as_numpy(prepared.source_present).astype(bool)
            if present.ndim != 2 or present.shape[0] != batch_size:
                raise ValueError("Module-associated role stencils require the prepared source-presence axis.")
            receivers_per_module = indices.shape[1] // present.shape[1]
            if receivers_per_module * present.shape[1] != indices.shape[1]:
                raise ValueError(f"Native role {role!r} cannot be mapped to original module rows.")
            valid &= np.repeat(present, receivers_per_module, axis=1)
        gathered = group_weights[batch_indices, indices]
        role_weights = np.sum(gathered * interpolation[..., None], axis=-2)
        role_weights *= valid[..., None]
        mass = role_weights.sum(axis=(0, 1), dtype=np.float64)
        count = int(valid.sum())
        if count:
            expected = float(count)
            observed = float(mass.sum())
            if not np.isclose(observed, expected, rtol=2e-4, atol=2e-3):
                raise ValueError(f"Valid {role} receiver group mass sums to {observed}, expected {expected}.")
            mean_mass = mass / count
            role_means.append(mean_mass)
            total_group_mass += mass
            total_receiver_count += count
        else:
            mean_mass = np.zeros_like(mass)
        role_summaries[role] = {
            "valid_native_receivers": count,
            "total_query_group_mass": mass.tolist(),
            "mean_group_mass_per_native_receiver": mean_mass.tolist(),
        }
    if not role_means:
        raise ValueError("No valid native role receivers contributed query-group mass.")
    role_balanced_mass = np.sum(np.stack(role_means, axis=0), axis=0)
    pooled_mean_mass = total_group_mass / total_receiver_count
    selected_group = int(np.argmax(role_balanced_mass))
    return {
        "schema": "round12_compact_native_role_group_mass_v1",
        "roles": role_summaries,
        "role_count": len(role_means),
        "role_balanced_group_mass": role_balanced_mass.tolist(),
        "query_count_weighted_group_mass": pooled_mean_mass.tolist(),
        "selected_group_index": selected_group,
        "selection_rule": "largest sum of per-role mean pi mass, giving each valid native role equal weight",
        "total_valid_native_receivers_across_roles": int(total_receiver_count),
    }


def summarize_compact_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Report exact-positive memberships, query mass and final source support."""
    import numpy as np

    required = ("compact_group_weights", "compact_membership", "compact_group_mask", "gates",
                "local_module_envelope", "original_record_present", "original_record_eligible")
    absent = [name for name in required if name not in trace]
    if absent:
        raise ValueError(f"Compact evaluation trace is missing required arrays: {absent}.")
    weights = _as_numpy(trace["compact_group_weights"])
    membership = _as_numpy(trace["compact_membership"])
    group_mask = _as_numpy(trace["compact_group_mask"]).astype(bool)
    final_access = _as_numpy(trace["gates"])
    local_envelope = _as_numpy(trace["local_module_envelope"])
    record_present = _as_numpy(trace["original_record_present"]).astype(bool)
    record_eligible = _as_numpy(trace["original_record_eligible"]).astype(bool)
    if weights.ndim != 3 or membership.ndim != 3 or final_access.ndim != 3:
        raise ValueError("Compact weights, membership and final access must retain [B,Q,K]/[B,K,N]/[B,Q,N] axes.")
    if weights.shape[0] != membership.shape[0] or weights.shape[2] != membership.shape[1]:
        raise ValueError("Compact query weights do not align with the declared group axis.")
    if weights.shape[0] != final_access.shape[0] or weights.shape[1] != final_access.shape[1] or membership.shape[2] != final_access.shape[2]:
        raise ValueError("Compact membership or final access does not align with query/source axes.")
    if group_mask.shape != (weights.shape[0], weights.shape[2]):
        raise ValueError("Compact live-group mask differs from the query group axis.")
    if local_envelope.ndim != 3 or local_envelope.shape[:2] != weights.shape[:2]:
        raise ValueError("Local module envelope must retain the [B,Q,M] axes.")
    if local_envelope.shape[2] > membership.shape[2]:
        raise ValueError("Local module envelope exceeds the original record axis.")
    if record_present.shape != (weights.shape[0], membership.shape[2]):
        raise ValueError("Original-record presence mask differs from the compact donor axis.")
    if record_eligible.shape != record_present.shape:
        raise ValueError("Original-record eligibility mask differs from the compact donor axis.")
    query_valid_value = trace.get("query_valid_mask")
    query_valid = (np.ones(weights.shape[:2], dtype=bool) if query_valid_value is None
                   else _as_numpy(query_valid_value).astype(bool))
    if query_valid.shape != weights.shape[:2]:
        raise ValueError("Query-valid mask differs from the query axis.")
    declared_counts = trace.get("original_record_count")
    if declared_counts is not None:
        declared = _as_numpy(declared_counts).reshape(-1)
        if declared.shape != (weights.shape[0],) or not np.array_equal(declared.astype(int), record_present.sum(axis=1)):
            raise ValueError("Declared original-record count differs from the present-record mask.")
    if (not np.isfinite(weights).all() or not np.isfinite(membership).all()
            or not np.isfinite(final_access).all() or not np.isfinite(local_envelope).all()):
        raise ValueError("Compact evaluation trace contains nonfinite routing values.")

    raw_positive_membership = (membership > 0) & record_present[:, None, :]
    record_accessible = record_present & record_eligible
    positive_membership = (membership > 0) & record_accessible[:, None, :]
    positive_weights = (weights > 0) & group_mask[:, None, :]
    represented_support = np.einsum("bqk,bkn->bqn", positive_weights.astype(np.int64),
                                     positive_membership.astype(np.int64)) > 0
    positive_final = (final_access > 0) & record_accessible[:, None, :]
    local_support = np.zeros_like(positive_final)
    module_count = local_envelope.shape[2]
    local_support[:, :, :module_count] = (
        (local_envelope > 0) & record_accessible[:, None, :module_count]
    )
    composed_support = represented_support | local_support
    type_values = trace.get("compact_group_types")
    types = None if type_values is None else _as_numpy(type_values)
    rows = []
    for batch_index in range(weights.shape[0]):
        valid_queries = query_valid[batch_index]
        present = record_present[batch_index]
        live = group_mask[batch_index]
        if "original_record_count" in trace:
            present_count = int(_as_numpy(trace["original_record_count"]).reshape(-1)[batch_index])
        else:
            present_count = int(present.sum())
        live_types: dict[str, int] = {}
        if types is not None:
            for value in types[batch_index][live].reshape(-1):
                key = str(value.item() if hasattr(value, "item") else value)
                live_types[key] = live_types.get(key, 0) + 1
        query_group_mass = weights[batch_index][valid_queries].sum(axis=0)
        member_counts = positive_membership[batch_index].sum(axis=1)
        group_query_counts = positive_weights[batch_index][valid_queries].sum(axis=0)
        group_count = int(live.sum())
        valid_pi_support = represented_support[batch_index][valid_queries]
        valid_local_support = local_support[batch_index][valid_queries]
        valid_composed_support = composed_support[batch_index][valid_queries]
        valid_final_support = positive_final[batch_index][valid_queries]
        rows.append({
            "batch_index": batch_index,
            "source_count_N": present_count,
            "allocated_source_slots_N": int(membership.shape[2]),
            "source_eligible_count": int(record_accessible[batch_index].sum()),
            "valid_query_count": int(valid_queries.sum()),
            "allocated_query_slots": int(weights.shape[1]),
            "live_group_count_K": group_count,
            "allocated_group_count_K": int(weights.shape[2]),
            "group_types_live_counts": live_types,
            "positive_C_memberships": int(positive_membership[batch_index][live].sum()),
            "positive_C_memberships_on_ineligible_present_records": int(
                (raw_positive_membership[batch_index] & ~record_eligible[batch_index, None, :])[live].sum()
            ),
            "positive_C_memberships_per_live_group": member_counts[live].astype(int).tolist(),
            "query_positive_group_entries": int(positive_weights[batch_index][valid_queries][:, live].sum()),
            "group_query_positive_counts": group_query_counts[live].astype(int).tolist(),
            "group_query_mass": query_group_mass[live].astype(float).tolist(),
            "piC_positive_support_entries": int(valid_pi_support.sum()),
            "local_module_envelope_positive_support_entries": int(valid_local_support.sum()),
            "support_added_by_local_module_envelope": int((valid_local_support & ~valid_pi_support).sum()),
            "represented_positive_support_entries": int(valid_composed_support.sum()),
            "positive_final_access_entries": int(valid_final_support.sum()),
            "positive_final_access_M_entries": int(valid_final_support[:, :module_count].sum()),
            "positive_final_access_E_entries": int(valid_final_support[:, module_count:].sum()),
            "represented_support_equals_final_positive_support": bool(
                np.array_equal(valid_composed_support, valid_final_support)
            ),
            "piC_equals_final_positive_support_without_local_envelope": bool(
                np.array_equal(valid_pi_support, valid_final_support)
            ),
            "final_positive_access_per_valid_query": valid_final_support.sum(axis=-1).astype(int).tolist(),
        })
    return {
        "schema": "round12_compact_trace_support_summary_v1",
        "semantics": "exact positive values only; no threshold is applied",
        "batch_count": len(rows),
        "query_count_Q": int(query_valid.sum()),
        "allocated_query_slots_Q": int(weights.shape[1]),
        "rows": rows,
    }


def response_payload_for_arm(payload: Mapping[str, Any], arm_label: str | None = None) -> dict[str, Any]:
    """Add a derived result label without rewriting sealed checkpoint identity."""
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Checkpoint is missing its sealed experiment identity.")
    config_name = str(identity.get("config_name", ""))
    derived = CONFIG_ARM_LABELS.get(config_name)
    if arm_label is not None and derived is not None and arm_label != derived:
        raise ValueError(f"Requested arm label {arm_label!r} conflicts with sealed config {config_name!r}.")
    label = derived if arm_label is None else arm_label
    if not label:
        recipe = identity.get("recipe", {})
        label = identity.get("treatment") or (recipe.get("arm") if isinstance(recipe, Mapping) else None)
    if not isinstance(label, str) or not label:
        raise ValueError("A response replay needs an explicit arm label or a recognized sealed research config.")
    result = dict(payload)
    result["experiment_identity"] = copy.deepcopy(dict(identity))
    result["experiment_identity"]["treatment"] = label
    return result


def evaluate_saved_response_cli(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from honf_runtime.joint_native_evaluation import write_json_new
    from honf_runtime.joint_native_physics import evaluate_saved_responses
    from honf_runtime.research_training import (
        _load_run_manifest,
        _model_from_run_payload,
        _read_run_checkpoint,
    )

    run_dir = args.run_dir.resolve(strict=True)
    run_manifest, _ = _load_run_manifest(run_dir)
    payload, checkpoint = _read_run_checkpoint(run_dir, args.checkpoint)
    if payload.get("workflow") != "ordinary_honf_research_train_v1":
        raise ValueError("Parent saved-response replay requires a literal ordinary Thermal research checkpoint.")
    if args.expected_epoch is not None and int(payload.get("epoch", -1)) != args.expected_epoch:
        raise ValueError("Literal checkpoint epoch differs from the requested evidence label.")
    config_name = str(run_manifest["config"].get("config_name", ""))
    identity_name = str(payload["experiment_identity"].get("config_name", ""))
    if config_name and identity_name and config_name != identity_name:
        raise ValueError("Run manifest and selected checkpoint name different sealed research configs.")
    arm_payload = response_payload_for_arm(payload, args.arm_label)
    inventory_path = args.inventory.resolve(strict=True)
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema") != "round9_existing_physical_response_inventory_v1":
        raise ValueError("Response replay requires the inventoried Round9 saved physical references.")
    if tuple(source["family_id"] for source in inventory.get("thermal_families", ())) != RESPONSE_FAMILY_IDS:
        raise ValueError("Physical response inventory differs from the four Round12 frozen TRAIN families.")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = _model_from_run_payload(payload, torch.device(args.device))
    options = arm_payload["experiment_identity"].get("execution_options")
    if isinstance(options, Mapping):
        setter = getattr(model.core, "set_execution_options", None)
        if callable(setter):
            setter(**dict(options))
    model.eval()
    result = evaluate_saved_responses(
        model, None, arm_payload, inventory, args.output, family_ids=RESPONSE_FAMILY_IDS,
    )
    write_json_new(args.output / "provenance.json", {
        "schema": "thermal_saved_response_replay_provenance_v1",
        "run_directory": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "sealed_research_config": arm_payload["experiment_identity"].get("config_name"),
        "derived_arm_label": arm_payload["experiment_identity"]["treatment"],
        "response_inventory": str(inventory_path),
        "response_family_ids": list(RESPONSE_FAMILY_IDS),
        "optimizer_updates": 0,
        "new_solver_attempts": 0,
        "TEST_or_Wind_access": False,
        "device": str(args.device),
    })
    return {"status": "completed", "arm": arm_payload["experiment_identity"]["treatment"],
            "epoch": int(payload["epoch"]), "output": str(args.output.resolve())}


def _module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ranking dependency from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_w6_ranking_reference(rows: Sequence[Mapping[str, Any]], saved: Mapping[str, Any],
                                  study: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Retain the original six M3 candidates and exact decision controls."""
    if len(rows) != len(W6_CANDIDATE_IDS):
        raise ValueError("Saved ranking reference must contain exactly six candidates.")
    observed = tuple(str(row.get("candidate_id", "")) for row in rows)
    if observed != W6_CANDIDATE_IDS:
        raise ValueError("Saved ranking candidates differ from the original ordered six-ID pool.")
    constraints = saved.get("original_constraints")
    if not isinstance(constraints, Mapping):
        raise ValueError("Saved ranking reference lacks original physical/objective constraints.")
    tau = float(constraints.get("tau_temperature", float("nan")))
    pressure_limit = float(study.get("pressure_drop_limit", float("nan")))
    if not math.isfinite(tau) or tau != 2.0:
        raise ValueError("Saved ranking must use the original smooth module-peak objective at tau=2.0.")
    if not math.isfinite(pressure_limit) or abs(pressure_limit - 0.1364936370) > 5e-11:
        raise ValueError("Saved ranking pressure limit differs from the original 0.1364936370 control.")
    return tuple(dict(row) for row in rows)


def summarize_cached_heat_reuse_arrays(arrays: Mapping[str, Any], family_id: str) -> dict[str, Any]:
    """Compare saved fixed-layout K contractions with fresh endpoint deltas by physical channel."""
    import numpy as np

    rows = []
    component_names = {
        "fluid_fields": ("u", "v", "p", "omega", "temperature"),
        "interface": ("temperature", "q_proxy"),
        "solid_temperature": ("temperature",),
    }
    component_units = {
        "u": "dataset velocity",
        "v": "dataset velocity",
        "p": "dataset pressure",
        "omega": "dataset vorticity",
        "temperature": "dataset temperature",
        "q_proxy": "dataset heat-flux proxy",
    }

    def errors(cached_values, fresh_values, reference_values, mask):
        valid_values = int(mask.sum())
        if not valid_values:
            raise ValueError("Fixed-layout K comparison has no valid component values.")
        disagreement = cached_values[mask] - fresh_values[mask]
        cached_error = cached_values[mask] - reference_values[mask]
        fresh_error = fresh_values[mask] - reference_values[mask]

        def stats(values):
            return {
                "absolute_rmse": float(np.sqrt(np.mean(np.square(values)))),
                "absolute_mae": float(np.mean(np.abs(values))),
                "maximum_absolute_error": float(np.max(np.abs(values))),
            }

        return {
            "valid_values": valid_values,
            "cached_K_vs_fresh_native": stats(disagreement),
            "cached_K_vs_physical_reference": stats(cached_error),
            "fresh_native_vs_physical_reference": stats(fresh_error),
        }

    for label in ("heat_transfer_plus", "heat_transfer_minus"):
        for role in ("fluid_fields", "interface", "solid_temperature"):
            prefix = f"{label}__{role}__"
            required = (prefix + "promoted_increment", prefix + "prediction_delta",
                        prefix + "reference_delta", "heat__" + role + "__valid")
            absent = [name for name in required if name not in arrays]
            if absent:
                raise ValueError(f"Saved response arrays lack fixed-layout K evidence: {absent}.")
            cached = np.asarray(arrays[prefix + "promoted_increment"], dtype=np.float64)
            fresh = np.asarray(arrays[prefix + "prediction_delta"], dtype=np.float64)
            reference = np.asarray(arrays[prefix + "reference_delta"], dtype=np.float64)
            valid = np.asarray(arrays["heat__" + role + "__valid"], dtype=bool)
            if cached.shape != fresh.shape or cached.shape != reference.shape:
                raise ValueError("Cached contraction, fresh prediction and physical reference shapes differ.")
            if valid.shape == cached.shape[:-1]:
                valid = valid[..., None]
            try:
                valid = np.broadcast_to(valid, cached.shape)
            except ValueError as error:
                raise ValueError("Saved physical-response mask does not broadcast to its role values.") from error
            finite_values = np.isfinite(cached) & np.isfinite(fresh) & np.isfinite(reference)
            if np.any(valid & ~finite_values):
                raise ValueError("Fixed-layout K comparison contains nonfinite values on its physical-valid support.")
            finite = valid
            if not finite.any():
                raise ValueError("Fixed-layout K comparison needs nonempty physical-valid saved response values.")
            channels = component_names[role]
            if cached.ndim < 1 or cached.shape[-1] != len(channels):
                raise ValueError(f"Saved {role} response has {cached.shape[-1] if cached.ndim else 0} channels; expected {len(channels)}.")
            aggregate = errors(cached, fresh, reference, finite)
            hetero = role in ("fluid_fields", "interface")
            rows.append({
                "family_id": str(family_id),
                "heat_vector_label": label,
                "role": role,
                "valid_values": aggregate["valid_values"],
                "aggregate_interpretation": (
                    "heterogeneous channel units; descriptive aggregate only, not a scientific scalar metric"
                    if hetero else "single temperature channel in dataset-native units"
                ),
                "cached_K_vs_fresh_native": aggregate["cached_K_vs_fresh_native"],
                "cached_K_vs_physical_reference": aggregate["cached_K_vs_physical_reference"],
                "fresh_native_vs_physical_reference": aggregate["fresh_native_vs_physical_reference"],
                "component_comparisons": {
                    channel: {
                        "units": component_units[channel],
                        **errors(
                            cached[..., index], fresh[..., index], reference[..., index], finite[..., index],
                        ),
                    }
                    for index, channel in enumerate(channels)
                },
            })
    return {"schema": "round12_fixed_layout_K_reuse_components_v2", "family_id": str(family_id),
            "comparison_count": len(rows), "comparisons": rows}


def prepare_cached_native_response(model: Any, prepared: Any) -> tuple[Any, dict[str, Any]]:
    """Promote one baseline native preparation to retained affine response factors."""
    import torch

    promote = getattr(model, "prepare_affine_application", None)
    if not callable(promote):
        raise TypeError("Thermal adapter does not expose retained affine response preparation.")
    cached = promote(prepared)
    affine = getattr(cached, "affine_query", None)
    if affine is None:
        raise RuntimeError("Affine promotion returned no retained response factors.")

    def tensor_inventory(value: Any) -> dict[str, Any]:
        if torch.is_tensor(value):
            return {"type": "Tensor", "shape": list(value.shape), "entries": int(value.numel())}
        return {"type": type(value).__name__, "shape": None, "entries": None}

    local_pairs = getattr(affine, "local_pairs", None)
    factor_work = getattr(affine, "source_read_work", None)
    if factor_work is not None:
        if not isinstance(factor_work, Mapping) or any(
            not isinstance(name, str) or type(value) is not int or value < 0
            for name, value in factor_work.items()
        ):
            raise RuntimeError("Affine response preparation returned invalid source-read work counts.")
        factor_work = dict(factor_work)
    return cached, {
        "cache_present": True,
        "promotion_method": "prepare_affine_application",
        "prepared_affine_type": type(affine).__name__,
        "basis": tensor_inventory(getattr(affine, "basis", None)),
        "direct_coefficients": tensor_inventory(getattr(affine, "direct_coefficients", None)),
        "local_coefficients": tensor_inventory(getattr(affine, "local_coefficients", None)),
        "local_pair_count": int(getattr(local_pairs, "count", 0)),
        "factor_preparation_source_read_work": factor_work,
    }


def apply_cached_native_response(
    model: Any,
    prepared: Any,
    physical_heat: Any,
    *,
    increment: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Apply controls through retained factors and validate same-call work counters."""
    import torch
    from honf_runtime.joint_native_physics import _response_role_arrays

    if getattr(prepared, "affine_query", None) is None:
        raise ValueError("Cached Thermal response application requires promoted affine factors.")
    apply = getattr(model, "apply_native_with_work_counts", None)
    if not callable(apply):
        raise TypeError("Thermal v2 adapter must expose apply_native_with_work_counts for cache accounting.")
    output, work = apply(
        prepared,
        physical_heat,
        increment=bool(increment),
        accumulation_dtype=torch.float64,
    )
    if not isinstance(work, Mapping):
        raise RuntimeError("Cached native response application omitted work counters.")
    normalized_work: dict[str, int] = {}
    for name, value in work.items():
        if not isinstance(name, str) or type(value) is not int or value < 0:
            raise RuntimeError("Cached native response returned invalid work counters.")
        normalized_work[name] = value
    # `direct_dense_reference_entries` describes retained A2-D coefficients
    # consumed by the contraction. It is cached-factor application work, not
    # a reader pass, and is expected to be nonzero for this architecture.
    reader_keys = ("edge_rows", "edge_message_rows", "edge_access_rows", "direct_source_rows",
                   "direct_source_message_entries", "direct_temperature_coefficient_entries",
                   "field_head_rows")
    unexpected_reader_work = {name: normalized_work.get(name, 0) for name in reader_keys
                              if normalized_work.get(name, 0) != 0}
    if unexpected_reader_work:
        raise RuntimeError(f"Retained affine application unexpectedly repeated reader work: {unexpected_reader_work}.")
    return _response_role_arrays(output), normalized_work


def _absolute_error_statistics(left: Any, right: Any, valid: Any) -> dict[str, Any]:
    import numpy as np

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("Absolute cached endpoint arrays must share the same role axes.")
    mask = np.asarray(valid, dtype=bool)
    if mask.shape == left_array.shape[:-1]:
        mask = mask[..., None]
    try:
        mask = np.broadcast_to(mask, left_array.shape)
    except ValueError as error:
        raise ValueError("Absolute endpoint validity mask does not broadcast to role values.") from error
    finite_values = np.isfinite(left_array) & np.isfinite(right_array)
    if np.any(mask & ~finite_values):
        raise ValueError("Absolute endpoint comparison contains nonfinite values on physical-valid support.")
    selected = left_array[mask] - right_array[mask]
    if selected.size == 0:
        raise ValueError("Absolute endpoint comparison has no valid finite physical entries.")
    return {
        "valid_values": int(selected.size),
        "absolute_rmse": float(np.sqrt(np.mean(np.square(selected)))),
        "absolute_mae": float(np.mean(np.abs(selected))),
        "maximum_absolute_error": float(np.max(np.abs(selected))),
    }


def summarize_actual_cached_endpoint_arrays(arrays: Mapping[str, Any], family_id: str) -> dict[str, Any]:
    """Summarize cached absolute endpoints against same-checkpoint and physical evidence."""
    import numpy as np

    component_names = {
        "fluid_fields": ("u", "v", "p", "omega", "temperature"),
        "interface": ("temperature", "q_proxy"),
        "solid_temperature": ("temperature",),
    }
    component_units = {
        "u": "dataset velocity", "v": "dataset velocity", "p": "dataset pressure",
        "omega": "dataset vorticity", "temperature": "dataset temperature",
        "q_proxy": "dataset heat-flux proxy",
    }
    comparisons = []
    for heat_label in ("heat_transfer_plus", "heat_transfer_minus"):
        for role, channels in component_names.items():
            prefix = f"{heat_label}__{role}__"
            required = (
                prefix + "actual_cache_absolute",
                prefix + "fresh_native_absolute_endpoint",
                prefix + "physical_endpoint_reference",
                "heat__" + role + "__valid",
            )
            absent = [name for name in required if name not in arrays]
            if absent:
                raise ValueError(f"Actual cached endpoint evidence is missing arrays: {absent}.")
            cached = np.asarray(arrays[prefix + "actual_cache_absolute"], dtype=np.float64)
            fresh = np.asarray(arrays[prefix + "fresh_native_absolute_endpoint"], dtype=np.float64)
            reference = np.asarray(arrays[prefix + "physical_endpoint_reference"], dtype=np.float64)
            valid = np.asarray(arrays["heat__" + role + "__valid"], dtype=bool)
            if cached.shape != fresh.shape or cached.shape != reference.shape:
                raise ValueError("Cached, fresh native, and physical endpoint axes differ.")
            if cached.shape[-1] != len(channels):
                raise ValueError(f"Actual cached {role} endpoint has an unexpected component axis.")
            if valid.shape == cached.shape[:-1]:
                component_valid = np.broadcast_to(valid[..., None], cached.shape)
            else:
                try:
                    component_valid = np.broadcast_to(valid, cached.shape)
                except ValueError as error:
                    raise ValueError("Absolute endpoint validity mask does not broadcast to role values.") from error
            component_results = {}
            for index, channel in enumerate(channels):
                channel_valid = component_valid[..., index]
                component_results[channel] = {
                    "units": component_units[channel],
                    "cached_K_vs_fresh_native_absolute_endpoint": _absolute_error_statistics(
                        cached[..., index], fresh[..., index], channel_valid,
                    ),
                    "cached_K_vs_physical_reference_endpoint": _absolute_error_statistics(
                        cached[..., index], reference[..., index], channel_valid,
                    ),
                    "fresh_native_vs_physical_reference_endpoint": _absolute_error_statistics(
                        fresh[..., index], reference[..., index], channel_valid,
                    ),
                }
            aggregate_interpretation = (
                "heterogeneous channel units; descriptive aggregate only, not a scientific scalar metric"
                if role in ("fluid_fields", "interface")
                else "single temperature channel in dataset-native units"
            )
            comparisons.append({
                "family_id": str(family_id),
                "heat_vector_label": heat_label,
                "role": role,
                "aggregate_interpretation": aggregate_interpretation,
                "component_comparisons": component_results,
            })
    return {
        "schema": "round12_actual_cached_K_absolute_endpoint_components_v1",
        "family_id": str(family_id),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def evaluate_actual_cached_k_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse a promoted native affine cache at saved TRAIN heat endpoints only."""
    import numpy as np
    import torch
    from honf_runtime.joint_native_evaluation import file_sha256, write_arrays, write_json_new
    from honf_runtime.joint_native_physics import _response_role_arrays, _response_structure
    from honf_runtime.research_training import (
        _load_run_manifest,
        _model_from_run_payload,
        _read_run_checkpoint,
    )

    run_dir = args.run_dir.resolve(strict=True)
    run_manifest, _ = _load_run_manifest(run_dir)
    payload, checkpoint = _read_run_checkpoint(run_dir, args.checkpoint)
    if payload.get("workflow") != "ordinary_honf_research_train_v1":
        raise ValueError("Actual cached-K evaluation requires a literal Thermal research checkpoint.")
    if args.expected_epoch is not None and int(payload.get("epoch", -1)) != args.expected_epoch:
        raise ValueError("Literal checkpoint epoch differs from the requested evidence label.")
    config_name = str(run_manifest["config"].get("config_name", ""))
    identity_name = str(payload["experiment_identity"].get("config_name", ""))
    if config_name and identity_name and config_name != identity_name:
        raise ValueError("Run manifest and selected checkpoint name different sealed research configs.")
    arm_payload = response_payload_for_arm(payload, args.arm_label)
    arm = str(arm_payload["experiment_identity"]["treatment"])

    response_result_path = args.response_result.resolve(strict=True)
    response_result = json.loads(response_result_path.read_text())
    response_provenance_path = response_result_path.parent / "provenance.json"
    if not response_provenance_path.is_file():
        raise FileNotFoundError("Saved response replay lacks its existing provenance.json binding.")
    response_provenance = json.loads(response_provenance_path.read_text())
    if (response_result.get("schema") != "round9_saved_physical_response_replay_v1"
            or response_result.get("dataset") != "thermal"
            or int(response_result.get("completed_epoch", -1)) != int(payload["epoch"])
            or int(response_result.get("optimizer_updates", -1)) != 0
            or int(response_result.get("new_solver_attempts", -1)) != 0):
        raise ValueError("Saved response results must be read-only Thermal arrays at this checkpoint epoch.")
    if (response_provenance.get("schema") != "thermal_saved_response_replay_provenance_v1"
            or Path(str(response_provenance.get("run_directory", ""))).resolve() != run_dir
            or Path(str(response_provenance.get("checkpoint", ""))).resolve() != checkpoint.resolve()
            or int(response_provenance.get("checkpoint_epoch", -1)) != int(payload["epoch"])
            or response_provenance.get("derived_arm_label") != arm):
        raise ValueError("Saved response provenance is not bound to this run, checkpoint, and arm.")

    inventory_path = args.inventory.resolve(strict=True)
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema") != "round9_existing_physical_response_inventory_v1":
        raise ValueError("Actual cached-K evaluation requires the inventoried Round9 physical responses.")
    inventory_families = tuple(inventory.get("thermal_families", ()))
    if tuple(str(row.get("family_id")) for row in inventory_families) != RESPONSE_FAMILY_IDS:
        raise ValueError("Physical response inventory differs from the exact four-family Round12 panel.")
    result_families = {str(row.get("family_id")): row for row in response_result.get("families", ())}
    if set(result_families) != set(RESPONSE_FAMILY_IDS) or len(response_result.get("families", ())) != len(RESPONSE_FAMILY_IDS):
        raise ValueError("Current response replay must contain exactly the four original TRAIN families.")

    output = args.output
    if output.exists():
        raise FileExistsError("Actual cached-K output already exists; choose a fresh evidence path.")
    output.mkdir(parents=True)
    model = _model_from_run_payload(payload, torch.device(args.device))
    options = arm_payload["experiment_identity"].get("execution_options")
    if isinstance(options, Mapping):
        setter = getattr(model.core, "set_execution_options", None)
        if callable(setter):
            setter(**dict(options))
    model.eval()

    def inference_call(function, *call_args, **call_kwargs):
        with torch.no_grad():
            return function(*call_args, **call_kwargs)

    family_summaries = []
    for inventory_source in inventory_families:
        family_id = str(inventory_source["family_id"])
        replay_family = result_families[family_id]
        source_path = Path(str(inventory_source["path"])).resolve(strict=True)
        expected_sha = str(inventory_source["sha256"])
        if file_sha256(source_path) != expected_sha:
            raise ValueError(f"Physical response family {family_id} differs from its inventoried bytes.")
        if (Path(str(replay_family.get("source_path", ""))).resolve() != source_path
                or replay_family.get("source_sha256") != expected_sha):
            raise ValueError(f"Saved replay family {family_id} differs from the physical inventory binding.")
        arrays_path = (response_result_path.parent / str(replay_family["arrays"])).resolve(strict=True)
        if file_sha256(arrays_path) != str(replay_family.get("arrays_sha256", "")):
            raise ValueError(f"Saved response arrays for family {family_id} changed after the current replay.")
        with np.load(source_path, allow_pickle=False) as handle:
            atlas = {name: handle[name] for name in handle.files}
        with np.load(arrays_path, allow_pickle=False) as handle:
            saved = {name: handle[name] for name in handle.files}
        labels = atlas["all_labels"].tolist()
        if saved.get("labels", np.asarray(())).tolist() != labels:
            raise ValueError(f"Saved response labels for family {family_id} do not match its physical atlas.")
        if not {"baseline", "heat_transfer_plus", "heat_transfer_minus"} <= set(labels):
            raise ValueError(f"Family {family_id} lacks an existing admissible baseline and plus/minus heat pair.")
        baseline_index = labels.index("baseline")
        structure, fluid, local, _ = _response_structure(atlas, baseline_index, torch.device(args.device))
        prepared = inference_call(model.prepare_native,
            structure, fluid, local_query_points=local, ntheta=64,
            scene_key=(family_id, "baseline"),
        )
        if getattr(prepared, "affine_query", None) is not None:
            raise RuntimeError("Fresh native reference preparation was unexpectedly pre-promoted.")
        baseline_controls = torch.as_tensor(
            np.asarray(atlas["heating"][baseline_index:baseline_index + 1], dtype=np.float64),
            device=torch.device(args.device),
        )
        fresh_baseline = inference_call(
            lambda: _response_role_arrays(model.apply_native(
                prepared, baseline_controls, accumulation_dtype=torch.float64,
            )),
        )
        cached_prepared, cache_evidence = inference_call(
            prepare_cached_native_response, model, prepared,
        )
        cached_baseline, baseline_work = inference_call(
            apply_cached_native_response, model, cached_prepared, baseline_controls, increment=False,
        )
        output_arrays: dict[str, Any] = {
            "labels": np.asarray(labels),
            "native_module_ids": np.asarray(atlas["active_module_ids"]),
            "module_centers_xy": np.asarray(atlas["module_centers_xy"][baseline_index]),
            "baseline_heat": baseline_controls.detach().cpu().numpy(),
        }
        for role, value in cached_baseline.items():
            output_arrays[f"baseline__{role}__actual_cache_absolute"] = np.asarray(value, dtype=np.float64)
            output_arrays[f"baseline__{role}__fresh_native_absolute"] = np.asarray(
                fresh_baseline[role], dtype=np.float64,
            )

        endpoint_rows = []
        cached_apply_work: dict[str, Any] = {}
        for heat_label in ("heat_transfer_plus", "heat_transfer_minus"):
            endpoint_index = labels.index(heat_label)
            endpoint_controls = torch.as_tensor(
                np.asarray(atlas["heating"][endpoint_index:endpoint_index + 1], dtype=np.float64),
                device=torch.device(args.device),
            )
            delta_controls = endpoint_controls - baseline_controls
            fresh_absolute = inference_call(
                lambda: _response_role_arrays(model.apply_native(
                    prepared, endpoint_controls, accumulation_dtype=torch.float64,
                )),
            )
            cached_absolute, absolute_work = inference_call(
                apply_cached_native_response, model, cached_prepared, endpoint_controls, increment=False,
            )
            cached_delta, delta_work = inference_call(
                apply_cached_native_response, model, cached_prepared, delta_controls, increment=True,
            )
            cached_apply_work[heat_label] = {
                "absolute_endpoint": absolute_work,
                "delta_endpoint": delta_work,
            }
            for role in ("fluid_fields", "interface", "solid_temperature"):
                prefix = f"{heat_label}__{role}__"
                required = (
                    prefix + "prediction_delta",
                    prefix + "reference_delta",
                    "heat__" + role + "__valid",
                )
                absent = [name for name in required if name not in saved]
                if absent:
                    raise ValueError(f"Current response replay for {family_id} is missing arrays {absent}.")
                fresh_delta = np.asarray(saved[prefix + "prediction_delta"], dtype=np.float64)
                reference_delta = np.asarray(saved[prefix + "reference_delta"], dtype=np.float64)
                valid = np.asarray(saved["heat__" + role + "__valid"], dtype=bool)
                physical_endpoint = np.asarray(atlas[role + "_values"][endpoint_index], dtype=np.float64)
                cached_absolute_values = np.asarray(cached_absolute[role], dtype=np.float64)
                cached_delta_values = np.asarray(cached_delta[role], dtype=np.float64)
                fresh_absolute_values = np.asarray(fresh_absolute[role], dtype=np.float64)
                if not (fresh_delta.shape == reference_delta.shape == cached_delta_values.shape
                        == cached_absolute_values.shape == fresh_absolute_values.shape == physical_endpoint.shape):
                    raise ValueError(f"Actual cache endpoint {family_id}/{heat_label}/{role} has mismatched axes.")
                output_arrays[prefix + "promoted_increment"] = cached_delta_values
                output_arrays[prefix + "actual_cache_absolute"] = cached_absolute_values
                output_arrays[prefix + "fresh_native_absolute_endpoint"] = fresh_absolute_values
                output_arrays[prefix + "prediction_delta"] = fresh_delta
                output_arrays[prefix + "reference_delta"] = reference_delta
                output_arrays[prefix + "physical_endpoint_reference"] = physical_endpoint
                output_arrays["heat__" + role + "__valid"] = valid
            endpoint_rows.append({
                "label": heat_label,
                "physical_heating": endpoint_controls.detach().cpu().numpy().reshape(-1).tolist(),
                "delta_heating": delta_controls.detach().cpu().numpy().reshape(-1).tolist(),
                "scope": "existing atlas heat vector at this fixed original TRAIN layout",
                "fresh_native_reference_application_count": 1,
                "cached_absolute_application_count": 1,
                "cached_delta_application_count": 1,
            })

        family_arrays_path = output / f"family_{family_id}.npz"
        write_arrays(family_arrays_path, output_arrays)
        delta_summary = summarize_cached_heat_reuse_arrays(output_arrays, family_id)
        absolute_summary = summarize_actual_cached_endpoint_arrays(output_arrays, family_id)
        family_summaries.append({
            "family_id": family_id,
            "source_path": str(source_path),
            "source_sha256": expected_sha,
            "saved_response_arrays": str(arrays_path),
            "actual_cache_arrays": family_arrays_path.name,
            "cache": cache_evidence,
            "controls": endpoint_rows,
            "fresh_native_baseline_reference_application_count": 1,
            "fresh_native_total_reference_application_count": 3,
            "affine_cache_promotion_count": 1,
            "baseline_cached_apply_work_counts": baseline_work,
            "cached_apply_work_counts": cached_apply_work,
            "reader_recompute_work_counts": {
                "baseline": {key: int(baseline_work.get(key, 0)) for key in (
                    "edge_rows", "edge_message_rows", "edge_access_rows", "direct_source_rows",
                    "direct_source_message_entries", "direct_temperature_coefficient_entries", "field_head_rows",
                )},
                **{
                    label: {
                        apply_kind: {key: int(counts.get(key, 0)) for key in (
                            "edge_rows", "edge_message_rows", "edge_access_rows", "direct_source_rows",
                            "direct_source_message_entries", "direct_temperature_coefficient_entries", "field_head_rows",
                        )}
                        for apply_kind, counts in row.items()
                    }
                    for label, row in cached_apply_work.items()
                },
            },
            "retained_factor_application_work_counts": {
                "baseline": {key: int(baseline_work.get(key, 0)) for key in (
                    "direct_dense_reference_entries", "spatial_factor_entries", "local_contraction_rows",
                    "direct_response_contraction_rows", "edge_response_contraction_rows",
                )},
                **{
                    label: {
                        apply_kind: {key: int(counts.get(key, 0)) for key in (
                            "direct_dense_reference_entries", "spatial_factor_entries", "local_contraction_rows",
                            "direct_response_contraction_rows", "edge_response_contraction_rows",
                        )}
                        for apply_kind, counts in row.items()
                    }
                    for label, row in cached_apply_work.items()
                },
            },
            "cached_affine_application_count": 5,
            "cached_k_delta_comparisons": delta_summary["comparisons"],
            "cached_absolute_endpoint_comparisons": absolute_summary["comparisons"],
        })

    result = {
        "schema": "round12_actual_cached_native_K_reuse_v1",
        "status": "completed_actual_retained_affine_cache_reuse",
        "arm": arm,
        "checkpoint_epoch": int(payload["epoch"]),
        "run_directory": str(run_dir),
        "checkpoint": str(checkpoint),
        "response_result": str(response_result_path),
        "response_inventory": str(inventory_path),
        "response_family_ids": list(RESPONSE_FAMILY_IDS),
        "families": family_summaries,
        "fresh_native_reference_application_count": 12,
        "affine_cache_promotion_count": 4,
        "cached_affine_application_count": 20,
        "interpretation": "Each baseline native preparation is promoted once with the maintained affine-cache API. Three fresh native reference reads per family retain baseline and plus/minus absolute endpoints; five baseline, absolute-endpoint, and delta applications per family then use retained factors. Saved fresh endpoint deltas and physical atlas endpoints are also reused for comparisons. This demonstrates fixed-layout contraction reuse only, not speedup, generated-design validity, or physical causality.",
        "optimizer_updates": 0,
        "new_solver_attempts": 0,
        "TEST_or_Wind_access": False,
        "device": str(args.device),
    }
    write_json_new(output / "result.json", result)
    write_json_new(output / "provenance.json", {
        "schema": "round12_actual_cached_native_K_reuse_provenance_v1",
        "run_directory": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "arm": arm,
        "response_result": str(response_result_path),
        "response_inventory": str(inventory_path),
        "optimizer_updates": 0,
        "new_solver_attempts": 0,
        "TEST_or_Wind_access": False,
        "device": str(args.device),
    })
    return {"status": "completed", "arm": arm, "epoch": int(payload["epoch"]),
            "family_count": len(family_summaries), "output": str(output.resolve())}


def summarize_parent_heat_reuse_cli(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from honf_runtime.joint_native_evaluation import write_json_new

    arms: dict[str, Any] = {}
    expected_families = list(RESPONSE_FAMILY_IDS)
    if len(args.response_result) < 2:
        raise ValueError("Fixed-layout K reuse summary needs at least two saved response results.")
    for binding in args.response_result:
        label, separator, raw_path = binding.partition("=")
        label = label.strip()
        if not separator or not label or label in arms:
            raise ValueError("Provide unique nonempty ARM=PATH bindings for saved response results.")
        result_path = Path(raw_path).resolve(strict=True)
        result = json.loads(result_path.read_text())
        try:
            checkpoint_epoch = int(result.get("completed_epoch", -1))
        except (TypeError, ValueError) as error:
            raise ValueError("K reuse source must record its completed checkpoint epoch.") from error
        if (result.get("schema") != "round9_saved_physical_response_replay_v1"
                or result.get("dataset") != "thermal" or checkpoint_epoch < 0):
            raise ValueError("K reuse source must be a completed saved Thermal response replay with an epoch.")
        if int(result.get("optimizer_updates", -1)) != 0 or int(result.get("new_solver_attempts", -1)) != 0:
            raise ValueError("K reuse evidence must remain a saved-model read-only evaluation.")
        family_map = {str(row["family_id"]): row for row in result.get("families", ())}
        if set(family_map) != set(expected_families):
            raise ValueError("K reuse source must contain the four original TRAIN response families.")
        family_summaries = []
        for family_id in RESPONSE_FAMILY_IDS:
            family = family_map[family_id]
            array_path = result_path.parent / str(family["arrays"])
            with np.load(array_path, allow_pickle=False) as archive:
                family_summaries.append(summarize_cached_heat_reuse_arrays(archive, family_id))
        arms[label] = {
            "response_result": str(result_path),
            "checkpoint_epoch": checkpoint_epoch,
            "families": family_summaries,
        }
    output = args.output
    write_json_new(output / "result.json", {
        "schema": "round12_fixed_layout_K_reuse_summary_v2",
        "status": "completed_saved_response_comparison",
        "arms": arms,
        "actual_cache_demonstrated": False,
        "interpretation": "These precision-promoted increments use saved response states whose baseline preparation was not promoted with prepare_affine_application; apply_increment therefore recomputes source readers. This summary is not evidence of retained affine-cache reuse. Use actual-cache-k results for cache-reuse evidence. Component comparisons retain physical units; heterogeneous aggregate rows are descriptive only, and absolute errors are reported because small response references make relative errors unstable.",
        "optimizer_updates": 0,
        "new_solver_attempts": 0,
        "TEST_or_Wind_access": False,
    })
    return {"status": "completed", "arms": sorted(arms), "output": str(output.resolve())}


def resolve_native_board_case_ids(
    split: str,
    requested: Sequence[str] | None,
    *,
    representative_panel: bool = False,
) -> tuple[str, ...]:
    split = str(split).lower()
    if split not in ("train", "validation"):
        raise ValueError("Native Thermal board split must be train or validation.")
    if representative_panel:
        expected = REPRESENTATIVE_IDS[split]
        if requested is not None and tuple(map(str, requested)) != expected:
            raise ValueError(f"Representative {split} probe must use the exact ordered panel {expected}.")
        return expected
    if split == "validation":
        if requested is not None and tuple(map(str, requested)) != VAL89_IDS:
            raise ValueError("Round12 validation board must retain the canonical ordered VAL89 panel.")
        return VAL89_IDS
    ids = TRAIN24_IDS if requested is None else tuple(map(str, requested))
    if len(ids) != len(set(ids)):
        raise ValueError("TRAIN native diagnostic panel contains duplicate case IDs.")
    if ids != TRAIN24_IDS:
        raise ValueError("Round12 TRAIN board must retain the existing ordered fixed TRAIN24 panel.")
    return ids


def native_response_coefficient_arrays(model: Any, prepared: Any) -> dict[str, Any]:
    """Recompute original per-source native heat-response columns on one preparation."""
    import torch

    apply = getattr(model, "apply_native_with_work_counts", None)
    if not callable(apply):
        raise TypeError("Response coefficients require the maintained native increment adapter.")
    present = prepared.source_present
    if present.ndim != 2 or present.shape[0] != 1:
        raise ValueError("Round12 response coefficients require one original case per preparation.")
    module_count = int(present.shape[1])
    columns: dict[str, list[Any]] = {
        "fluid_temperature": [], "pred_interface": [], "pred_internal_temperature": [],
    }
    work: list[dict[str, int]] = []
    device = prepared.scene_batch.module_features.device
    for source_index in range(module_count):
        if not bool(present[0, source_index].detach().cpu()):
            continue
        unit_heat = torch.zeros_like(present, dtype=prepared.scene_batch.module_features.dtype, device=device)
        unit_heat[0, source_index] = 1.0
        output, counts = apply(prepared, unit_heat, increment=True)
        for name in columns:
            if name not in output:
                raise ValueError(f"Native increment omitted response coefficient field {name!r}.")
            columns[name].append(output[name].detach())
        work.append(dict(counts))
    if not work:
        raise ValueError("Response coefficient probe found no physically present module sources.")
    result = {"response_K_" + name: torch.stack(values, dim=0).cpu().numpy()
              for name, values in columns.items()}
    result["response_K_source_module_indices"] = (
        present[0].nonzero(as_tuple=False).flatten().detach().cpu().numpy()
    )
    result["response_K_column_work_counts"] = work
    return result


def compare_native_board_pair(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Compare matched representative cut boards using physical outputs and saved arrays."""
    import numpy as np

    if tuple(before.get("case_ids", ())) != tuple(after.get("case_ids", ())):
        raise ValueError("Cut-boundary boards must have identical ordered case IDs.")
    case_ids = tuple(map(str, before["case_ids"]))
    expected = REPRESENTATIVE_IDS.get(str(before.get("split")))
    if expected is None or case_ids != expected or str(after.get("split")) != str(before.get("split")):
        raise ValueError("Cut-boundary comparison requires one exact representative panel in one partition.")
    fields = ("u", "v", "p", "omega", "temperature_fluid", "temperature_surface",
              "temperature_material", "q_proxy", "module_peak_endpoint")
    before_rows = {str(row["case_id"]): row for row in before.get("records", ())}
    after_rows = {str(row["case_id"]): row for row in after.get("records", ())}
    if set(before_rows) != set(case_ids) or set(after_rows) != set(case_ids):
        raise ValueError("Cut-boundary field records do not cover the exact representative panel.")
    field_delta = {}
    for field in fields:
        rows = []
        for case_id in case_ids:
            left = _field_value(before_rows[case_id], field)
            right = _field_value(after_rows[case_id], field)
            rows.append({"case_id": case_id, "before_error": left, "after_error": right,
                         "error_delta_after_minus_before": right - left})
        field_delta[field] = rows
    before_traces, after_traces = before.get("organization_traces", {}), after.get("organization_traces", {})
    support_changes = {}
    for case_id in case_ids:
        left_summary = before_traces.get(case_id, {}).get("summary", {}).get("rows", ())
        right_summary = after_traces.get(case_id, {}).get("summary", {}).get("rows", ())
        if len(left_summary) != 1 or len(right_summary) != 1:
            raise ValueError(f"Both boundary boards must retain one full compact trace for {case_id}.")
        keys = ("live_group_count_K", "allocated_group_count_K", "positive_C_memberships", "piC_positive_support_entries",
                "positive_final_access_entries", "support_added_by_local_module_envelope")
        support_changes[case_id] = {
            key: {"before": left_summary[0].get(key), "after": right_summary[0].get(key),
                  "delta": (None if left_summary[0].get(key) is None or right_summary[0].get(key) is None
                            else right_summary[0][key] - left_summary[0][key])}
            for key in keys
        }
        support_changes[case_id]["lineage_before"] = before_traces.get(case_id, {}).get("lineage")
        support_changes[case_id]["lineage_after"] = after_traces.get(case_id, {}).get("lineage")
    result = {
        "schema": "round12_compact_cut_boundary_pair_comparison_v1",
        "case_ids": list(case_ids),
        "before_checkpoint": before.get("checkpoint"),
        "before_epoch": before.get("checkpoint_epoch"),
        "after_checkpoint": after.get("checkpoint"),
        "after_epoch": after.get("checkpoint_epoch"),
        "field_error_changes": field_delta,
        "organization_support_changes": support_changes,
        "response_K_array_comparison_required": True,
        "response_K_comparison_note": "Use response_K_* arrays from paired case NPZ exports; columns are per-present-module unit-heat native increments.",
    }
    before_counts = {str(row["case_id"]): row for row in before.get("organization_counts", ())}
    after_counts = {str(row["case_id"]): row for row in after.get("organization_counts", ())}
    if before_counts and after_counts:
        if set(before_counts) != set(case_ids) or set(after_counts) != set(case_ids):
            raise ValueError("Boundary organization counts must cover the exact representative panel.")
        count_fields = tuple(key for key in before_counts[case_ids[0]] if key != "case_id")
        result["organization_count_changes"] = {
            case_id: {
                key: {"before": int(before_counts[case_id][key]),
                      "after": int(after_counts[case_id][key]),
                      "delta": int(after_counts[case_id][key]) - int(before_counts[case_id][key])}
                for key in count_fields
            }
            for case_id in case_ids
        }
    return result


def _delta_statistics(
    before: Any,
    after: Any,
    *,
    label: str,
    valid_mask: Any | None = None,
    interpretation: str | None = None,
) -> dict[str, Any]:
    import numpy as np

    left, right = _as_numpy(before).astype(np.float64), _as_numpy(after).astype(np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Boundary {label} axes differ: {left.shape} versus {right.shape}.")
    finite = np.isfinite(left) & np.isfinite(right)
    if valid_mask is not None:
        try:
            finite &= np.broadcast_to(np.asarray(valid_mask, dtype=bool), left.shape)
        except ValueError as error:
            raise ValueError(f"Boundary {label} validity mask does not broadcast to its output axes.") from error
    if not finite.any():
        raise ValueError(f"Boundary {label} has no finite shared values.")
    delta = (right - left)[finite]
    result = {"shape": list(left.shape), "finite_values": int(finite.sum()),
              "delta_rmse": float(np.sqrt(np.mean(np.square(delta)))),
              "delta_mae": float(np.mean(np.abs(delta))),
              "delta_max_abs": float(np.max(np.abs(delta)))}
    if interpretation is not None:
        result["interpretation"] = interpretation
    return result


def _paired_validity_mask(before: Mapping[str, Any], after: Mapping[str, Any], name: str) -> Any:
    import numpy as np

    if name not in before or name not in after:
        raise ValueError(f"Boundary arrays need the saved receiver mask {name!r}.")
    left, right = np.asarray(before[name], dtype=bool), np.asarray(after[name], dtype=bool)
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(f"Boundary receiver support {name!r} changed across the cut comparison.")
    return left


def _paired_interface_validity(before: Mapping[str, Any], after: Mapping[str, Any]) -> Any:
    import numpy as np

    surface = _paired_validity_mask(before, after, "interface_valid")
    outside = _paired_validity_mask(before, after, "outside_valid")
    if surface.shape != outside.shape:
        raise ValueError("Boundary surface and outside receiver masks have different axes.")
    return np.stack((surface, surface & outside), axis=-1)


def _component_delta_statistics(
    before: Any,
    after: Any,
    *,
    label: str,
    component_names: Sequence[str],
    valid_mask: Any,
) -> dict[str, Any]:
    import numpy as np

    left, right = _as_numpy(before), _as_numpy(after)
    if left.shape != right.shape or left.ndim < 1:
        raise ValueError(f"Boundary {label} component axes differ.")
    if left.shape[-1] != len(component_names):
        raise ValueError(f"Boundary {label} has {left.shape[-1]} channels, expected {len(component_names)}.")
    try:
        channel_mask = np.broadcast_to(np.asarray(valid_mask, dtype=bool), left.shape)
    except ValueError as error:
        raise ValueError(f"Boundary {label} component mask does not broadcast to its output axes.") from error
    return {
        name: _delta_statistics(
            left[..., index], right[..., index], label=f"{label}.{name}",
            valid_mask=channel_mask[..., index],
            interpretation=f"component {name}; dataset-native output units",
        )
        for index, name in enumerate(component_names)
    }


def compare_native_board_pair_arrays(
    before: Mapping[str, Any], after: Mapping[str, Any], before_dir: Path, after_dir: Path,
) -> dict[str, Any]:
    """Add component-aware physical-output and recomputed source-response K comparisons."""
    import numpy as np

    result = compare_native_board_pair(before, after)
    result["schema"] = "round12_compact_cut_boundary_pair_arrays_v2"
    result["prediction_delta_metric_note"] = (
        "Per-component prediction and response-K deltas use the saved native validity masks. "
        "Mixed-channel aggregate rows are retained for compatibility and are descriptive only; "
        "they are not scientific scalar metrics."
    )
    case_deltas: dict[str, Any] = {}
    for case_id in result["case_ids"]:
        left_path, right_path = before_dir / f"case_{case_id}.npz", after_dir / f"case_{case_id}.npz"
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(f"Boundary comparison needs both exported arrays for case {case_id}.")
        with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
            row: dict[str, Any] = {"physical_outputs": {}, "response_K": {}}
            for name in ("field_prediction", "interface_prediction", "material_prediction",
                         "response_K_fluid_temperature", "response_K_pred_interface",
                         "response_K_pred_internal_temperature"):
                if name not in left or name not in right:
                    group = "physical output" if name.endswith("prediction") else "recomputed coefficient"
                    raise ValueError(f"Boundary case {case_id} lacks {group} array {name!r}.")
            fluid_valid = _paired_validity_mask(left, right, "fluid_valid")
            interface_valid = _paired_interface_validity(left, right)
            material_valid = _paired_validity_mask(left, right, "material_valid")
            physical = row["physical_outputs"]
            physical["field_prediction"] = _delta_statistics(
                left["field_prediction"], right["field_prediction"], label="field_prediction",
                valid_mask=fluid_valid[..., None],
                interpretation="heterogeneous channel units; aggregate is descriptive only, not a scientific scalar metric",
            )
            physical["field_prediction_by_component"] = _component_delta_statistics(
                left["field_prediction"], right["field_prediction"], label="field_prediction",
                component_names=("u", "v", "p", "omega", "temperature"),
                valid_mask=fluid_valid[..., None],
            )
            physical["interface_prediction"] = _delta_statistics(
                left["interface_prediction"], right["interface_prediction"], label="interface_prediction",
                valid_mask=interface_valid,
                interpretation="temperature and q_proxy units differ; aggregate is descriptive only, not a scientific scalar metric",
            )
            physical["interface_prediction_by_component"] = _component_delta_statistics(
                left["interface_prediction"], right["interface_prediction"], label="interface_prediction",
                component_names=("temperature", "q_proxy"), valid_mask=interface_valid,
            )
            physical["material_prediction"] = _delta_statistics(
                left["material_prediction"], right["material_prediction"], label="material_prediction",
                valid_mask=material_valid,
                interpretation="material temperature; dataset-native output units",
            )

            response = row["response_K"]
            response["response_K_fluid_temperature"] = _delta_statistics(
                left["response_K_fluid_temperature"], right["response_K_fluid_temperature"],
                label="response_K_fluid_temperature", valid_mask=fluid_valid[None, None, :, None],
                interpretation="fluid temperature response coefficient; dataset-native temperature per unit heat",
            )
            response["response_K_pred_interface"] = _delta_statistics(
                left["response_K_pred_interface"], right["response_K_pred_interface"],
                label="response_K_pred_interface", valid_mask=interface_valid[None, None, ...],
                interpretation="temperature and q_proxy units differ; aggregate is descriptive only, not a scientific scalar metric",
            )
            response["response_K_pred_interface_by_component"] = _component_delta_statistics(
                left["response_K_pred_interface"], right["response_K_pred_interface"],
                label="response_K_pred_interface", component_names=("temperature", "q_proxy"),
                valid_mask=interface_valid[None, None, ...],
            )
            response["response_K_pred_internal_temperature"] = _delta_statistics(
                left["response_K_pred_internal_temperature"], right["response_K_pred_internal_temperature"],
                label="response_K_pred_internal_temperature", valid_mask=material_valid[None, None, ..., None],
                interpretation="internal temperature response coefficient; dataset-native temperature per unit heat",
            )
            for name in ("response_K_source_module_indices",):
                if name not in left or name not in right or not np.array_equal(left[name], right[name]):
                    raise ValueError(f"Boundary case {case_id} response K source module axes differ.")
            case_deltas[case_id] = row
    result["per_case_prediction_and_response_K_deltas"] = case_deltas
    return result


def choose_legal_module_translation(structure: Mapping[str, Any]) -> dict[str, Any]:
    """Choose a non-overlapping horizontal move with positive wall clearance."""
    import numpy as np

    centers = _as_numpy(structure["module_centers"]).astype(np.float64)
    present = _as_numpy(structure["module_present"]).astype(bool)
    material = _as_numpy(structure["material_params"]).astype(np.float64)
    lx = float(_as_numpy(structure["domain_length_x"]).reshape(-1)[0])
    if centers.ndim != 3 or centers.shape[0] != 1 or centers.shape[-1] != 2:
        raise ValueError("Geometry probe requires one [1,M,2] Thermal module-center scene.")
    if present.shape != centers.shape[:2] or material.reshape(-1, material.shape[-1]).shape[-1] < 6:
        raise ValueError("Geometry probe presence/material axes differ from module centers.")
    radius = float(material.reshape(-1, material.shape[-1])[0, 5])
    active = np.flatnonzero(present[0])
    if not len(active) or lx <= 2 * radius:
        raise ValueError("Geometry probe requires a present module and a nonempty physical x-domain.")
    candidates = []
    for module_index in active:
        x, y = centers[0, module_index]
        for sign in (-1, 1):
            wall_limit = (x - radius) if sign < 0 else (lx - radius - x)
            collision_limit = float("inf")
            for other_index in active:
                if int(other_index) == int(module_index):
                    continue
                dx = sign * (centers[0, other_index, 0] - x)
                dy = centers[0, other_index, 1] - y
                contact = math.sqrt(max((2.0 * radius) ** 2 - dy * dy, 0.0))
                if dx >= contact:
                    collision_limit = min(collision_limit, dx - contact)
                elif dx > -contact:
                    collision_limit = 0.0
            maximum = max(0.0, min(float(wall_limit), collision_limit))
            if maximum > 1.0e-7 * lx:
                candidates.append((maximum, int(module_index), int(sign)))
    if not candidates:
        raise ValueError("No nontrivial legal horizontal module translation exists for this scene.")
    maximum, module_index, sign = max(candidates)
    nominal = min(0.002 * lx, 0.25 * maximum)
    return {
        "module_index": module_index,
        "axis": "x",
        "sign": sign,
        "nominal_shift": float(sign * nominal),
        "maximum_legal_shift_magnitude": float(maximum),
        "selection_rule": "largest one-module horizontal travel before wall contact or pairwise circle overlap",
    }


def translated_structure(structure: Mapping[str, Any], module_index: int, axis: int, amount: Any):
    """Copy structure inputs and translate one present module, preserving autograd."""
    import torch

    if axis not in (0, 1):
        raise ValueError("A module translation axis must be x=0 or y=1.")
    moved = dict(structure)
    centers = torch.as_tensor(structure["module_centers"])
    if centers.ndim != 3 or centers.shape[0] != 1 or not 0 <= module_index < centers.shape[1]:
        raise ValueError("Translated module slot is outside the one-case geometry.")
    mask = torch.zeros_like(centers)
    mask[0, module_index, axis] = 1.0
    if torch.is_tensor(amount):
        shift = amount.to(device=centers.device, dtype=centers.dtype)
    else:
        shift = torch.as_tensor(amount, device=centers.device, dtype=centers.dtype)
    moved["module_centers"] = centers + shift * mask
    return moved


def _candidate_leaf_identities(organization: Any) -> list[tuple[Any, ...]] | None:
    """Recover stable source identities in the coordinate-sorted compact leaf order."""
    import numpy as np

    scene = getattr(organization, "scene", None)
    records = getattr(organization, "joint_records", None)
    if scene is None or records is None:
        return None
    origin = getattr(organization, "frame_origin", None)
    scale = getattr(organization, "frame_scale", None)
    if origin is None or scale is None:
        return None
    result: list[tuple[Any, ...]] = []
    for bank, present_name, coordinates_name, ids_name, offset in (
        (0, "module_mask", "module_coordinates", "module_ids", 0),
        (1, "environment_mask", "environment_coordinates", "environment_ids", int(records.module_count)),
    ):
        present = _as_numpy(getattr(scene, present_name))[0].astype(bool)
        coordinates = _as_numpy(getattr(records, coordinates_name))[0].astype(np.float64)
        native_ids = _as_numpy(getattr(records, ids_name))[0]
        frame_origin = _as_numpy(origin)[0].astype(np.float64)
        frame_scale = _as_numpy(scale)[0].astype(np.float64)
        indices = np.flatnonzero(present)
        normalized = (coordinates - frame_origin) / frame_scale
        ordered = sorted(indices.tolist(), key=lambda index: (tuple(normalized[index].tolist()), index))
        id_counts: dict[int, int] = {}
        for index in indices:
            identity = int(native_ids[index])
            id_counts[identity] = id_counts.get(identity, 0) + 1
        for index in ordered:
            identity = int(native_ids[index])
            # Include the original typed row only if a malformed/repeated ID
            # would otherwise make two distinct leaves indistinguishable.
            suffix = (int(index + offset),) if id_counts[identity] > 1 else ()
            result.append((int(bank), identity, *suffix))
    return result


def _topology_from_organization(organization: Any) -> dict[str, Any]:
    import numpy as np

    compact = getattr(organization, "compact_groups", None)
    if compact is None:
        raise TypeError("Geometry diagnostics require a compact prepared organization.")
    lineage = compact.topology[0].as_dict()
    candidate = tuple(tuple(map(int, row)) for row in lineage["candidate_pairs"])
    accepted = tuple(tuple(map(int, row)) for row in lineage["accepted_pairs"])
    groups = tuple(tuple(map(int, row)) for row in lineage["group_members"])
    group_types = tuple(map(int, lineage.get("group_types", (0,) * len(groups))))
    active_keys = tuple((int(row[0]), tuple(map(int, row[1]))) for row in lineage["active_group_keys"])
    leaf_identities = _candidate_leaf_identities(organization)
    if leaf_identities is not None:
        def pair_identity(row: Sequence[int]) -> tuple[Any, ...]:
            left, right, bank = map(int, row)
            if max(left, right) >= len(leaf_identities):
                raise ValueError("Compact candidate pair references a leaf outside the typed source identity axis.")
            endpoints = tuple(sorted((leaf_identities[left], leaf_identities[right])))
            return (int(bank), endpoints)

        candidate_identity_order = tuple(pair_identity(row) for row in candidate)
        accepted_identity_order = tuple(pair_identity(row) for row in accepted)
    else:
        candidate_identity_order = tuple(tuple(row) for row in candidate)
        accepted_identity_order = tuple(tuple(row) for row in accepted)
    canonical_groups = tuple(sorted(
        (int(group_types[index]), tuple(sorted(map(int, active_keys[index][1]))))
        for index in range(min(len(groups), len(group_types), len(active_keys)))
    ))
    logits = _as_numpy(compact.pair_logits[0]).astype("float64")
    used_logits = logits[:len(candidate)]
    b_positive = tuple((candidate_identity_order[index], bool(used_logits[index] > 0))
                       for index in range(min(len(candidate_identity_order), len(used_logits))))
    order_signature = (candidate_identity_order, active_keys)
    ancestry_signature = (canonical_groups, tuple(sorted(accepted_identity_order)))
    b_signature = tuple(sorted(b_positive, key=repr))
    return {
        # The derivative interval requires actual candidate ordering, hard-cut
        # ancestry, and b>0 decisions to agree. The seam scanner below reports
        # order-only events separately from changes to accepted ancestry.
        "signature": (order_signature, ancestry_signature, b_signature),
        "ordering_signature": order_signature,
        "ancestry_signature": ancestry_signature,
        "b_positive_signature": b_signature,
        "candidate_pairs": [list(row) for row in candidate],
        "accepted_pairs": [list(row) for row in accepted],
        "group_members": [list(row) for row in groups],
        "group_types": list(group_types),
        "pair_logits": used_logits.tolist(),
        "pair_logit_positive": (used_logits > 0).tolist(),
        "candidate_pair_identities": _json_safe(candidate_identity_order),
        "accepted_pair_identities": _json_safe(accepted_identity_order),
        "canonical_group_ancestry": _json_safe(canonical_groups),
        "minimum_abs_pair_logit": None if not len(used_logits) else float(np.min(np.abs(used_logits))),
    }


def _geometry_topology(model: Any, structure: Mapping[str, Any], scene_key: Any) -> dict[str, Any]:
    core = getattr(model, "core", model)
    scene = model.scene_batch(structure, scene_key=scene_key)
    return _topology_from_organization(core.prepare(scene))


def fixed_exterior_fluid_receiver_mask(
    structure: Mapping[str, Any], fluid_xy: Any, translation: Mapping[str, Any], offsets: Sequence[float],
    *, translation_interval: tuple[float, float] | None = None,
):
    """Select one fixed physical-fluid receiver support exterior to every probe geometry."""
    import numpy as np

    points = _as_numpy(fluid_xy)[0].astype(np.float64)
    centers = _as_numpy(structure["module_centers"])[0].astype(np.float64)
    present = _as_numpy(structure["module_present"])[0].astype(bool)
    material = _as_numpy(structure["material_params"]).reshape(-1, 6)
    if points.ndim != 2 or points.shape[-1] != 2 or not present.any():
        raise ValueError("A geometry probe requires fixed two-dimensional fluid receivers and a present module.")
    if not np.isfinite(points).all() or not np.isfinite(centers[present]).all():
        raise ValueError("Geometry-probe receivers and present module centers must be finite.")
    radius = float(material[0, 5])
    moved_index = int(translation["module_index"])
    if not 0 <= moved_index < len(present) or not present[moved_index]:
        raise ValueError("The legal geometry translation must select a physically present module.")
    mask = np.ones((points.shape[0],), dtype=bool)
    margin = max(1.0e-7 * radius, 1.0e-10)
    if translation_interval is None:
        translation_interval = (0.0, 0.0)
    lo, hi = sorted(map(float, translation_interval))
    for module_index in np.flatnonzero(present):
        if int(module_index) == moved_index:
            center_x = centers[module_index, 0]
            closest_x = np.clip(points[:, 0], center_x + lo, center_x + hi)
            dx = points[:, 0] - closest_x
            dy = points[:, 1] - centers[module_index, 1]
        else:
            dx = points[:, 0] - centers[module_index, 0]
            dy = points[:, 1] - centers[module_index, 1]
        mask &= np.hypot(dx, dy) > radius + margin
    # Every queried endpoint is still checked explicitly; the interval check
    # covers all intervening positions for continuous finite differences.
    for offset in offsets:
        geometry = centers[moved_index].copy()
        geometry[0] += float(offset)
        delta = points - geometry
        mask &= np.hypot(delta[:, 0], delta[:, 1]) > radius + margin
        for other_index in np.flatnonzero(present):
            if int(other_index) == moved_index:
                continue
            delta_other = points - centers[other_index]
            mask &= np.hypot(delta_other[:, 0], delta_other[:, 1]) > radius + margin
    if not mask.any():
        raise ValueError("The selected bounded geometry probe has no fixed exterior-fluid native receivers.")
    return mask


def _geometry_native_scalar(
    model: Any,
    batch: Any,
    translation: Mapping[str, Any],
    offset: Any,
    *,
    fixed_fluid_mask: Any,
    expected_fluid_support: Any | None = None,
    need_gradient: bool = False,
):
    import torch
    import numpy as np

    structure = translated_structure(
        batch.scene_inputs.structure, int(translation["module_index"]), 0, offset,
    )
    prepared = model.prepare_native(
        structure,
        batch.receivers.fluid_xy,
        local_query_points=batch.receivers.local_query_points,
        ntheta=64,
        scene_key=batch.scene_inputs.case_ids,
    )
    output = model.apply_native(prepared, batch.receivers.heat)
    valid = _as_numpy(prepared.stencils["fluid"].valid).astype(bool)[0]
    if valid.shape != tuple(fixed_fluid_mask.shape):
        raise ValueError("Fixed exterior-fluid support differs from the native receiver-validity axis.")
    support = valid & np.asarray(fixed_fluid_mask, dtype=bool)
    if expected_fluid_support is not None and not np.array_equal(support, expected_fluid_support):
        raise ValueError("Native fluid support changed across geometry shifts; the finite difference is not comparable.")
    if not support.any():
        raise ValueError("Fixed exterior-fluid geometry observable has no valid native receiver support.")
    values = output["fluid_temperature"][..., 0]
    scalar = values[torch.as_tensor(support, device=values.device)[None]].mean()
    if not bool(torch.isfinite(scalar).detach().cpu()):
        raise ValueError("Translated fixed exterior-fluid mean-temperature probe is nonfinite.")
    grad = None
    if need_gradient:
        grad = torch.autograd.grad(scalar, offset, allow_unused=False)[0]
        if not bool(torch.isfinite(grad).detach().all().cpu()):
            raise ValueError("Translated native geometry autodiff is nonfinite.")
    return scalar, _topology_from_organization(prepared.organization), grad, prepared, support


def _numeric_delta(left: Any, right: Any) -> dict[str, Any] | None:
    import numpy as np

    before, after = _as_numpy(left).astype(np.float64), _as_numpy(right).astype(np.float64)
    if before.shape != after.shape:
        raise ValueError(f"Aligned geometry arrays have different shapes: {before.shape} versus {after.shape}.")
    finite = np.isfinite(before) & np.isfinite(after)
    if not finite.any():
        return None
    delta = after[finite] - before[finite]
    return {"shape": list(before.shape), "finite_values": int(finite.sum()),
            "delta_rmse": float(np.sqrt(np.mean(np.square(delta)))),
            "delta_mae": float(np.mean(np.abs(delta))),
            "delta_max_abs": float(np.max(np.abs(delta)))}


def _trace_group_keys(trace: Mapping[str, Any]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    lineage = trace.get("compact_lineage")
    if lineage is None or len(lineage) != 1:
        raise ValueError("Geometry routing comparison needs one case's compact group lineage.")
    rows = lineage[0]["active_group_keys"]
    return tuple((int(row[0]), tuple(map(int, row[1]))) for row in rows)


def _fluid_receiver_trace_values(trace: Mapping[str, Any], prepared: Any, name: str):
    """Pull query-plane diagnostics back to the fixed native fluid receivers."""
    import numpy as np

    values = _as_numpy(trace[name])
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError(f"Geometry routing field {name!r} must have [1,Q,D] axes.")
    stencil = prepared.stencils["fluid"]
    indices = _as_numpy(stencil.indices).astype(np.int64)
    weights = _as_numpy(stencil.weights).astype(np.float64)
    if indices.shape != weights.shape or indices.ndim != 3 or indices.shape[0] != 1:
        raise ValueError("Native fluid pullback must retain one-case [1,R,4] interpolation axes.")
    if ((indices < 0) | (indices >= values.shape[1])).any():
        raise ValueError("Native fluid interpolation stencil references an absent compact-trace query.")
    query_valid = _as_numpy(trace["query_valid_mask"]).astype(bool)
    if query_valid.shape != values.shape[:2]:
        raise ValueError("Geometry routing query validity differs from the captured query axis.")
    if not query_valid[0, indices[0]][(weights[0] != 0)].all():
        raise ValueError("Native fluid interpolation uses a padded query from the captured compact trace.")
    gathered = values[0, indices[0]]
    pulled = (gathered * weights[0][..., None]).sum(axis=1)
    query_ids = _as_numpy(trace["query_ids"])
    if query_ids.shape[:2] != values.shape[:2]:
        raise ValueError("Geometry query IDs differ from the captured query axis.")
    neighbor_ids = query_ids[0, indices[0]]
    return pulled, neighbor_ids, weights[0]


def _routing_geometry_comparison(
    before_trace: Mapping[str, Any], before_prepared: Any,
    after_trace: Mapping[str, Any], after_prepared: Any,
    fixed_fluid_support: Any,
) -> dict[str, Any]:
    """Compare live group state and access only after explicit ancestry alignment."""
    import numpy as np

    before_keys, after_keys = _trace_group_keys(before_trace), _trace_group_keys(after_trace)
    before_mask = _as_numpy(before_trace["compact_group_mask"])[0].astype(bool)
    after_mask = _as_numpy(after_trace["compact_group_mask"])[0].astype(bool)
    if before_mask.sum() != len(before_keys) or after_mask.sum() != len(after_keys):
        raise ValueError("Compact lineage keys do not align with the live group masks.")
    before_index = {key: index for index, key in enumerate(before_keys)}
    after_index = {key: index for index, key in enumerate(after_keys)}
    common = sorted(set(before_index) & set(after_index), key=repr)
    before_c = _as_numpy(before_trace["compact_membership"])[0]
    after_c = _as_numpy(after_trace["compact_membership"])[0]
    before_controls = _as_numpy(before_trace["compact_group_controls"])[0]
    after_controls = _as_numpy(after_trace["compact_group_controls"])[0]
    before_centroids = _as_numpy(before_trace["compact_group_centroids"])[0]
    after_centroids = _as_numpy(after_trace["compact_group_centroids"])[0]
    before_lengths = _as_numpy(before_trace["compact_group_lengths"])[0]
    after_lengths = _as_numpy(after_trace["compact_group_lengths"])[0]

    def state_summary(trace: Mapping[str, Any], group_count: int) -> dict[str, Any]:
        membership = _as_numpy(trace["compact_membership"])[0][:group_count]
        types = _as_numpy(trace["compact_group_types"])[0][:group_count]
        return {
            "live_K": int(group_count),
            "allocated_K": int(_as_numpy(trace["compact_group_mask"]).shape[-1]),
            "groups_by_type": {str(int(kind)): int((types == kind).sum()) for kind in sorted(set(types.tolist()))},
            "positive_C_memberships": int((membership > 0).sum()),
            "mean_control_L2": float(np.linalg.norm(_as_numpy(trace["compact_group_controls"])[0][:group_count], axis=-1).mean()),
            "mean_centroid": _as_numpy(trace["compact_group_centroids"])[0][:group_count].mean(0).astype(float).tolist(),
        }

    before_pi, before_query_ids, before_stencil_weights = _fluid_receiver_trace_values(
        before_trace, before_prepared, "compact_group_weights",
    )
    after_pi, after_query_ids, after_stencil_weights = _fluid_receiver_trace_values(
        after_trace, after_prepared, "compact_group_weights",
    )
    before_gates, before_gate_ids, before_gate_weights = _fluid_receiver_trace_values(
        before_trace, before_prepared, "gates",
    )
    after_gates, after_gate_ids, after_gate_weights = _fluid_receiver_trace_values(
        after_trace, after_prepared, "gates",
    )
    support = np.asarray(fixed_fluid_support, dtype=bool)
    checks = {
        "same_fixed_fluid_receiver_query_count": before_pi.shape[0] == after_pi.shape[0] == len(support),
        "same_grid_neighbor_ids": np.array_equal(before_query_ids, after_query_ids),
        "same_final_access_neighbor_ids": np.array_equal(before_gate_ids, after_gate_ids),
        "same_interpolation_weights": np.array_equal(before_stencil_weights, after_stencil_weights)
            and np.array_equal(before_gate_weights, after_gate_weights),
    }
    if not all(checks.values()):
        raise ValueError(f"Geometry routing comparison lacks fixed receiver-grid correspondence: {checks}.")

    aligned_pi_before = np.stack([before_pi[:, before_index[key]] for key in common], axis=-1) if common else None
    aligned_pi_after = np.stack([after_pi[:, after_index[key]] for key in common], axis=-1) if common else None
    before_ids = np.concatenate((_as_numpy(before_trace["module_ids"])[0],
                                 _as_numpy(before_trace["environment_ids"])[0]))
    after_ids = np.concatenate((_as_numpy(after_trace["module_ids"])[0],
                                _as_numpy(after_trace["environment_ids"])[0]))
    same_record_axis = np.array_equal(before_ids, after_ids)
    if before_gates.shape != after_gates.shape or not same_record_axis:
        raise ValueError("Geometry final-access traces do not retain the same original typed record axis.")
    valid_values = before_gates[support], after_gates[support]
    return {
        "observable_scope": "same fixed exterior-fluid native receiver points; query-plane values interpolated through each fresh native stencil",
        "receiver_correspondence_checks": checks,
        "fixed_exterior_fluid_receiver_count": int(support.sum()),
        "input_only_hard_cut": {
            "same_canonical_group_ancestry": set(before_keys) == set(after_keys),
            "before_group_keys": _json_safe(before_keys),
            "after_group_keys": _json_safe(after_keys),
            "cut_policy": "positive compatibility b>0 and coordinate-sorted adjacent-pair topology regenerated from each original scene",
        },
        "compact_state_before": state_summary(before_trace, len(before_keys)),
        "compact_state_after": state_summary(after_trace, len(after_keys)),
        "group_ancestry_alignment": {
            "matched_group_count": len(common),
            "before_only_group_count": len(set(before_keys) - set(after_keys)),
            "after_only_group_count": len(set(after_keys) - set(before_keys)),
            "matched_group_keys": _json_safe(common),
            "C_membership_delta": None if not common else _numeric_delta(
                np.stack([before_c[before_index[key]] for key in common]),
                np.stack([after_c[after_index[key]] for key in common]),
            ),
            "control_delta": None if not common else _numeric_delta(
                np.stack([before_controls[before_index[key]] for key in common]),
                np.stack([after_controls[after_index[key]] for key in common]),
            ),
            "centroid_delta": None if not common else _numeric_delta(
                np.stack([before_centroids[before_index[key]] for key in common]),
                np.stack([after_centroids[after_index[key]] for key in common]),
            ),
            "length_delta": None if not common else _numeric_delta(
                np.stack([before_lengths[before_index[key]] for key in common]),
                np.stack([after_lengths[after_index[key]] for key in common]),
            ),
            "pi_query_group_access_delta_on_matched_ancestry": None if not common else _numeric_delta(
                aligned_pi_before[support], aligned_pi_after[support],
            ),
        },
        "final_access_delta_on_fixed_exterior_fluid_receivers": _numeric_delta(
            valid_values[0], valid_values[1],
        ),
        "final_access_source_axis_ids": _json_safe(before_ids),
    }


def _geometry_probe_one_case(model: Any, batch: Any, case_id: str) -> dict[str, Any]:
    import numpy as np
    import torch

    structure = batch.scene_inputs.structure
    translation = choose_legal_module_translation(structure)
    nominal = float(translation["nominal_shift"])
    base_centers = _as_numpy(structure["module_centers"])
    lx = float(_as_numpy(structure["domain_length_x"]).reshape(-1)[0])
    h_max = min(0.004 * lx, 0.20 * abs(nominal),
                0.20 * max(translation["maximum_legal_shift_magnitude"] - abs(nominal), 0.0))
    if h_max <= 1.0e-7 * lx:
        raise ValueError("Legal geometry movement leaves no finite-difference interval around the probe point.")
    fixed_mask = fixed_exterior_fluid_receiver_mask(
        structure, batch.receivers.fluid_xy, translation,
        (0.0, nominal, nominal - h_max, nominal + h_max),
        translation_interval=(min(0.0, nominal - h_max), max(0.0, nominal + h_max)),
    )
    with torch.no_grad():
        baseline, initial, _, base_prepared, valid_support = _geometry_native_scalar(
            model, batch, translation, 0.0, fixed_fluid_mask=fixed_mask,
        )
    offset = torch.tensor(nominal, device=batch.receivers.fluid_xy.device,
                          dtype=batch.receivers.fluid_xy.dtype, requires_grad=True)
    with torch.enable_grad():
        scalar, topology, autodiff, translated_prepared, support_check = _geometry_native_scalar(
            model, batch, translation, offset, fixed_fluid_mask=fixed_mask,
            expected_fluid_support=valid_support, need_gradient=True,
        )
    h_sweep = []
    for fraction in (1.0, 0.5, 0.25, 0.125):
        epsilon = h_max * fraction
        with torch.no_grad():
            plus, plus_topology, _, _, _ = _geometry_native_scalar(
                model, batch, translation, nominal + epsilon, fixed_fluid_mask=fixed_mask,
                expected_fluid_support=valid_support,
            )
            minus, minus_topology, _, _, _ = _geometry_native_scalar(
                model, batch, translation, nominal - epsilon, fixed_fluid_mask=fixed_mask,
                expected_fluid_support=valid_support,
            )
        same_order = (plus_topology["ordering_signature"] == topology["ordering_signature"]
                      and minus_topology["ordering_signature"] == topology["ordering_signature"])
        same_ancestry = (plus_topology["ancestry_signature"] == topology["ancestry_signature"]
                         and minus_topology["ancestry_signature"] == topology["ancestry_signature"])
        same_b = (plus_topology["b_positive_signature"] == topology["b_positive_signature"]
                  and minus_topology["b_positive_signature"] == topology["b_positive_signature"])
        if not (same_order and same_ancestry and same_b):
            continue
        fd = float(((plus - minus) / (2.0 * epsilon)).detach().cpu())
        h_sweep.append({"h": epsilon, "central_finite_difference": fd,
                        "autodiff_minus_fd": float(autodiff.detach().cpu()) - fd,
                        "absolute_derivative_difference": abs(float(autodiff.detach().cpu()) - fd),
                        "actual_ancestry_same_on_both_sides": same_ancestry,
                        "candidate_pair_order_same_on_both_sides": same_order,
                        "b_positive_cut_same_on_both_sides": same_b})
    if not h_sweep:
        raise ValueError("No fixed-topology finite-difference interval found around the legal translation.")
    ad = float(autodiff.detach().cpu())
    # Quote the largest fixed-topology step as the primary represented FD;
    # retain the full decreasing h sweep to expose FP32 cancellation behavior.
    primary_fd = h_sweep[0]
    base_trace = capture_compact_trace_tiled(model, base_prepared)
    translated_trace = capture_compact_trace_tiled(model, translated_prepared)
    routing_comparison = _routing_geometry_comparison(
        base_trace, base_prepared, translated_trace, translated_prepared, valid_support,
    )
    selected = int(translation["module_index"])
    original = base_centers[0, selected].tolist()
    return {
        "case_id": str(case_id),
        "module_count": int(_as_numpy(structure["module_present"]).astype(bool).sum()),
        "translation": {**translation, "original_module_xy": original,
                        "target_module_xy": [original[0] + nominal, original[1]]},
        "observable": "mean native physical temperature on one fixed set of exterior-fluid grid receivers; no solid-site neural probes",
        "fixed_exterior_fluid_receiver_count": int(valid_support.sum()),
        "native_fluid_support_verified_unchanged_across_all_finite_difference_shifts": True,
        "autodiff": ad,
        "represented_central_finite_difference": primary_fd["central_finite_difference"],
        "finite_difference_epsilon": primary_fd["h"],
        "finite_difference_h_sweep": h_sweep,
        "autodiff_minus_finite_difference": primary_fd["autodiff_minus_fd"],
        "absolute_derivative_difference": primary_fd["absolute_derivative_difference"],
        "relative_derivative_difference": primary_fd["absolute_derivative_difference"] / max(
            abs(ad), abs(primary_fd["central_finite_difference"]), 1.0e-12),
        "fixed_topology_confirmed_for_derivative": True,
        "topology_at_original_geometry": {key: value for key, value in initial.items() if key not in {
            "signature", "ordering_signature", "ancestry_signature", "b_positive_signature"}},
        "topology_at_translated_geometry": {key: value for key, value in topology.items() if key not in {
            "signature", "ordering_signature", "ancestry_signature", "b_positive_signature"}},
        "original_to_translated_live_routing": routing_comparison,
        "native_scalar_at_original_geometry": float(baseline.detach().cpu()),
        "native_scalar_at_translated_geometry": float(scalar.detach().cpu()),
    }


def _probe_topology_seam(model: Any, batch: Any, translation: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    sign = int(translation["sign"])
    maximum = float(translation["maximum_legal_shift_magnitude"])
    module_index = int(translation["module_index"])
    scan_limit = 0.99 * maximum
    base = batch.scene_inputs.structure
    reference = _geometry_topology(model, base, batch.scene_inputs.case_ids)
    previous_t, previous = 0.0, reference
    bracket = None
    first_ordering_bracket = None
    scan_count = 12
    with torch.no_grad():
        for index in range(1, scan_count + 1):
            distance = scan_limit * index / scan_count
            structure = translated_structure(base, module_index, 0, sign * distance)
            current = _geometry_topology(model, structure, batch.scene_inputs.case_ids)
            if (first_ordering_bracket is None
                    and current["ordering_signature"] != previous["ordering_signature"]):
                first_ordering_bracket = (previous_t, distance, previous, current)
            if current["ancestry_signature"] != previous["ancestry_signature"]:
                bracket = (previous_t, distance, previous, current)
                break
            previous_t, previous = distance, current
    if bracket is None:
        if first_ordering_bracket is not None:
            left, right = first_ordering_bracket[2], first_ordering_bracket[3]
            return {
                "status": "ordering_change_without_accepted_ancestry_change",
                "scan_points": scan_count + 1,
                "legal_translation_scan_limit": scan_limit,
                "strictly_inside_legal_boundary": True,
                "ordering_change": {
                    "candidate_pair_order_changed": left["candidate_pairs"] != right["candidate_pairs"],
                    "active_group_slot_order_changed": left["group_members"] != right["group_members"],
                    "accepted_ancestry_unchanged": left["ancestry_signature"] == right["ancestry_signature"],
                    "before_candidate_pair_identities": left["candidate_pair_identities"],
                    "after_candidate_pair_identities": right["candidate_pair_identities"],
                },
                "shrinking_gap_outputs": [],
                "function_jump_claim": False,
                "global_C1_claim": False,
            }
        return {
            "status": "no_ordering_or_accepted_ancestry_change_found_in_bounded_legal_scan",
            "scan_points": scan_count + 1,
            "legal_translation_scan_limit": scan_limit,
            "strictly_inside_legal_boundary": True,
            "base_topology": {key: value for key, value in reference.items() if key not in {
                "signature", "ordering_signature", "ancestry_signature", "b_positive_signature"}},
            "shrinking_gap_outputs": [],
            "function_jump_claim": False,
            "global_C1_claim": False,
        }
    low, high, low_topology, high_topology = bracket
    initial_left, initial_right = low_topology, high_topology
    snapshots = []
    initial_kind = {
        "candidate_pair_order_changed": low_topology["candidate_pairs"] != high_topology["candidate_pairs"],
        "candidate_pair_identity_order_changed": low_topology["candidate_pair_identities"] != high_topology["candidate_pair_identities"],
        "accepted_ancestry_changed": low_topology["ancestry_signature"] != high_topology["ancestry_signature"],
        "b_positive_cut_changed": low_topology["b_positive_signature"] != high_topology["b_positive_signature"],
        "ordering_event_was_seen_before_cut_change": first_ordering_bracket is not None,
    }
    for iteration in range(1, 13):
        midpoint = 0.5 * (low + high)
        structure = translated_structure(base, module_index, 0, sign * midpoint)
        with torch.no_grad():
            middle = _geometry_topology(model, structure, batch.scene_inputs.case_ids)
        if middle["ancestry_signature"] == low_topology["ancestry_signature"]:
            low, low_topology = midpoint, middle
        else:
            high, high_topology = midpoint, middle
        if iteration not in (4, 8, 12):
            continue
        left_offset, right_offset = sign * low, sign * high
        fixed_mask = fixed_exterior_fluid_receiver_mask(
            base, batch.receivers.fluid_xy, translation, (left_offset, right_offset),
            translation_interval=(min(left_offset, right_offset), max(left_offset, right_offset)),
        )
        with torch.no_grad():
            left_scalar, left_live_topology, _, _, valid_support = _geometry_native_scalar(
                model, batch, translation, left_offset, fixed_fluid_mask=fixed_mask,
            )
            right_scalar, right_live_topology, _, _, _ = _geometry_native_scalar(
                model, batch, translation, right_offset, fixed_fluid_mask=fixed_mask,
                expected_fluid_support=valid_support,
            )
        snapshots.append({
            "refinement_iteration": iteration,
            "translation_gap": high - low,
            "left_translation": left_offset,
            "right_translation": right_offset,
            "fixed_exterior_fluid_receiver_count": int(valid_support.sum()),
            "left_mean_fluid_temperature": float(left_scalar.detach().cpu()),
            "right_mean_fluid_temperature": float(right_scalar.detach().cpu()),
            "left_to_right_output_jump": float((right_scalar - left_scalar).detach().cpu()),
            "left_topology": {key: value for key, value in left_live_topology.items() if key not in {
                "signature", "ordering_signature", "ancestry_signature", "b_positive_signature"}},
            "right_topology": {key: value for key, value in right_live_topology.items() if key not in {
                "signature", "ordering_signature", "ancestry_signature", "b_positive_signature"}},
            "actual_ancestry_differs": left_live_topology["ancestry_signature"] != right_live_topology["ancestry_signature"],
            "candidate_pair_order_differs": left_live_topology["ordering_signature"] != right_live_topology["ordering_signature"],
            "candidate_pair_rows_differ": left_live_topology["candidate_pairs"] != right_live_topology["candidate_pairs"],
            "accepted_hard_cut_differs": left_live_topology["ancestry_signature"] != right_live_topology["ancestry_signature"],
            "b_positive_cut_differs": left_live_topology["b_positive_signature"] != right_live_topology["b_positive_signature"],
        })
    return {
        "status": "accepted_ancestry_change_found",
        "scan_points": scan_count + 1,
        "legal_translation_scan_limit": scan_limit,
        "strictly_inside_legal_boundary": True,
        "change_kind": initial_kind,
        "left_initial_topology": {key: value for key, value in initial_left.items() if key not in {
            "signature", "ordering_signature", "ancestry_signature", "b_positive_signature"}},
        "right_initial_topology": {key: value for key, value in initial_right.items() if key not in {
            "signature", "ordering_signature", "ancestry_signature", "b_positive_signature"}},
        "shrinking_gap_outputs": snapshots,
        "function_jump_claim": False,
        "global_C1_claim": False,
    }


def _json_safe(value: Any) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(child) for child in value]
    return value


def _trace_subset(value: Any, *, local_points: int):
    if value is None:
        return None
    shape = tuple(value.shape)
    if len(shape) == 4:
        return value[..., :min(local_points, shape[-2]), :]
    if len(shape) == 3:
        return value[:, :min(local_points, shape[1]), :]
    raise ValueError("Thermal local material points must retain [B,M,Q,2] or [B,Q,2] axes.")


def capture_compact_trace_tiled(model: Any, prepared: Any) -> dict[str, Any]:
    """Capture the complete prepared native query union within the core tile bound."""
    import torch
    from honf_forward_core.content_hypergraph.query_chunks_v1 import slice_query_v1

    core = getattr(model, "core", model)
    capture = getattr(core, "capture_continuous_trace", None)
    query = prepared.query_batch
    if not callable(capture) or not hasattr(prepared.organization, "compact_groups"):
        raise TypeError("Tiled compact trace requires the maintained compact continuous trace API.")
    tile_size = int(getattr(core, "query_tile_size", query.num_queries))
    if tile_size < 1:
        raise ValueError("Compact trace query tile must be positive.")
    query_axis = {
        "compact_group_weights", "gates", "local_module_envelope", "query_valid_mask", "query_ids",
    }
    static_names = (
        "compact_membership", "compact_group_mask", "compact_group_types",
        "compact_group_controls", "compact_group_centroids", "compact_group_lengths",
        "compact_group_extents", "compact_live_K", "compact_allocated_K",
        "compact_lineage", "original_record_present", "original_record_eligible",
        "original_record_count", "module_ids", "environment_ids",
    )
    chunks: dict[str, list[Any]] = {name: [] for name in query_axis}
    static: dict[str, Any] = {}
    for start in range(0, query.num_queries, tile_size):
        tile = slice_query_v1(query, start, min(start + tile_size, query.num_queries))
        current = capture(prepared.organization, tile)
        if not static:
            static = {name: current[name] for name in static_names if name in current}
        for name in query_axis:
            if name not in current:
                raise ValueError(f"Compact trace tile omitted required query-axis field {name!r}.")
            chunks[name].append(current[name])
    if not chunks["compact_group_weights"]:
        raise ValueError("Cannot capture compact routing from an empty native query union.")
    if "compact_membership" not in static or "compact_group_mask" not in static:
        raise ValueError("Compact trace lacks its scene-owned group membership or mask.")
    return {
        **static,
        **{name: torch.cat(values, dim=1) for name, values in chunks.items()},
    }


def _compact_trace_sample_arrays(
    trace: Mapping[str, Any], prepared: Any, *, fluid_receiver_count: int, local_receiver_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retain a bounded set of actual query rows from the existing full trace."""
    import numpy as np

    query_count = _as_numpy(trace["compact_group_weights"]).shape[1]
    valid_query = _as_numpy(trace["query_valid_mask"])[0].astype(bool)
    query_ids = _as_numpy(trace["query_ids"])[0]
    query_coordinates = _as_numpy(prepared.query_batch.coordinates)[0]
    if len(valid_query) != query_count or len(query_ids) != query_count or len(query_coordinates) != query_count:
        raise ValueError("Selected compact-trace queries do not share one prepared native query axis.")
    selected: list[tuple[int, str, int, float]] = []

    def add_role(role: str, maximum: int, *, module_associated: bool) -> None:
        stencil = prepared.stencils.get(role)
        if stencil is None or maximum <= 0:
            return
        indices = _as_numpy(stencil.indices).astype(np.int64)[0]
        weights = _as_numpy(stencil.weights).astype(np.float64)[0]
        valid = _as_numpy(stencil.valid).astype(bool)[0]
        if indices.shape != weights.shape or indices.ndim != 2 or indices.shape[-1] != 4 or len(valid) != len(indices):
            raise ValueError(f"Native {role} trace selection requires aligned [R,4] stencils.")
        candidate_rows = np.flatnonzero(valid)
        if module_associated:
            present = _as_numpy(prepared.source_present)[0].astype(bool)
            if not len(present) or len(indices) % len(present):
                raise ValueError(f"Native {role} receiver rows do not map to present module rows.")
            per_module = len(indices) // len(present)
            row_module = np.arange(len(indices)) // max(per_module, 1)
            candidate_rows = candidate_rows[present[row_module[candidate_rows]]]
        if not len(candidate_rows):
            return
        positions = np.linspace(0, len(candidate_rows) - 1,
                                num=min(maximum, len(candidate_rows)), dtype=np.int64)
        for receiver_index in candidate_rows[positions]:
            neighbor = int(np.argmax(weights[receiver_index]))
            query_index = int(indices[receiver_index, neighbor])
            if not 0 <= query_index < query_count or not valid_query[query_index]:
                continue
            selected.append((query_index, role, int(receiver_index), float(weights[receiver_index, neighbor])))

    add_role("fluid", fluid_receiver_count, module_associated=False)
    for role in ("surface", "outside", "material"):
        add_role(role, local_receiver_count, module_associated=True)
    if not selected:
        raise ValueError("The bounded compact-trace sample did not select any valid native query rows.")
    query_indices = np.asarray([row[0] for row in selected], dtype=np.int64)
    native_ids = query_ids[query_indices]
    solid = _as_numpy(getattr(prepared, "native_solid_mask", np.zeros((1, 1, 1), dtype=bool)))[0].reshape(-1)
    inside_solid = np.zeros((len(selected),), dtype=bool)
    if native_ids.ndim == 1 and np.issubdtype(native_ids.dtype, np.integer):
        in_range = (native_ids >= 0) & (native_ids < len(solid))
        inside_solid[in_range] = solid[native_ids[in_range].astype(np.int64)]
    sample_arrays = {
        "trace_sample_query_indices": query_indices,
        "trace_sample_query_ids": native_ids,
        "trace_sample_query_coordinates": query_coordinates[query_indices],
        "trace_sample_role_labels": np.asarray([row[1] for row in selected], dtype=np.str_),
        "trace_sample_native_receiver_indices": np.asarray([row[2] for row in selected], dtype=np.int64),
        "trace_sample_selected_neighbor_interpolation_weights": np.asarray([row[3] for row in selected], dtype=np.float64),
        "trace_sample_query_valid": valid_query[query_indices],
        "trace_sample_inside_module_solid": inside_solid,
        "trace_sample_compact_group_weights": _as_numpy(trace["compact_group_weights"])[0, query_indices],
        "trace_sample_final_access_gates": _as_numpy(trace["gates"])[0, query_indices],
        "trace_sample_local_module_envelope": _as_numpy(trace["local_module_envelope"])[0, query_indices],
    }
    manifest = [{
        "sample_index": index,
        "query_index_in_prepared_union": int(query_indices[index]),
        "query_id": _json_safe(native_ids[index]),
        "role": row[1],
        "native_receiver_index": row[2],
        "selected_neighbor_interpolation_weight": row[3],
        "query_valid": bool(valid_query[query_indices[index]]),
        "inside_module_solid_neural_probe": bool(inside_solid[index]),
        "physical_fluid_error_receiver": False,
    } for index, row in enumerate(selected)]
    return sample_arrays, manifest


def _native_prediction_validity(native_masks: Mapping[str, Any], name: str, output_shape: Sequence[int]) -> Any:
    import numpy as np

    if name == "pred_field":
        mask_names = ("fluid_valid",)
    elif name == "pred_interface":
        mask_names = ("interface_valid", "outside_valid")
    elif name == "pred_internal_temperature":
        mask_names = ("material_valid",)
    else:
        raise ValueError(f"No physical receiver validity is defined for {name!r}.")
    axes = tuple(output_shape[:-1])
    masks = []
    for mask_name in mask_names:
        if mask_name not in native_masks:
            raise ValueError(f"Native field deltas need the {mask_name!r} physical receiver mask.")
        mask = _as_numpy(native_masks[mask_name]).astype(bool)
        try:
            mask = mask.reshape(axes)
        except ValueError as error:
            raise ValueError(f"Native {mask_name} mask does not match {name} axes.") from error
        masks.append(mask)
    if name == "pred_interface":
        return np.stack((masks[0], masks[0] & masks[1]), axis=-1)
    return masks[0][..., None]


def _prediction_difference(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    before_masks: Mapping[str, Any],
    after_masks: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    result = {}
    component_names = {
        "pred_field": ("u", "v", "p", "omega", "temperature"),
        "pred_interface": ("temperature", "q_proxy"),
        "pred_internal_temperature": ("temperature",),
    }
    for name in ("pred_field", "pred_interface", "pred_internal_temperature"):
        if name not in before or name not in after:
            continue
        left, right = _as_numpy(before[name]).astype(np.float64), _as_numpy(after[name]).astype(np.float64)
        if left.shape != right.shape:
            raise ValueError(f"Native intervention output {name!r} changed its physical axes.")
        before_mask = _native_prediction_validity(before_masks, name, left.shape)
        after_mask = _native_prediction_validity(after_masks, name, right.shape)
        if not np.array_equal(before_mask, after_mask):
            raise ValueError(f"Native intervention output {name!r} changed its physical receiver support.")
        channel_labels = component_names[name]
        if len(channel_labels) != left.shape[-1]:
            raise ValueError(f"Native intervention output {name!r} channel count differs from its physical schema.")
        interpretation = (
            "heterogeneous channel units; aggregate is descriptive only, not a scientific scalar metric"
            if len(channel_labels) > 1 else f"{channel_labels[0]}; dataset-native output units"
        )
        result[name] = _delta_statistics(
            left, right, label=name, valid_mask=before_mask, interpretation=interpretation,
        )
        result[f"{name}_by_component"] = _component_delta_statistics(
            left, right, label=name, component_names=channel_labels, valid_mask=before_mask,
        )
    if not result:
        raise ValueError("Native intervention produced no comparable Thermal field outputs.")
    return result


def run_single_group_intervention(
    model: Any,
    prepared: Any,
    physical_heat: Any,
    trace: Mapping[str, Any],
    role_mass: Mapping[str, Any],
    native_masks: Mapping[str, Any],
    *,
    baseline_output: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Disable one role-used group on a copied preparation and rerun native decode."""
    import numpy as np
    import torch

    batch_index = 0
    group_index = int(role_mass["selected_group_index"])
    changed_prepared, transition = prepare_copy_with_disabled_group(
        prepared, batch_index=batch_index, group_index=group_index,
    )
    apply = getattr(model, "apply_native_with_work_counts", None)
    if not callable(apply):
        raise TypeError("Group intervention needs the maintained same-call native work-count adapter.")
    device = prepared.scene_batch.module_features.device
    heat = torch.as_tensor(physical_heat, device=device)
    if heat.shape != prepared.scene_batch.module_mask.shape:
        raise ValueError("Group intervention heat vector differs from the original physical module axis.")
    with torch.no_grad():
        baseline, baseline_work = apply(prepared, heat)
        changed, changed_work = apply(changed_prepared, heat)
    reference = baseline if baseline_output is None else baseline_output
    repeated_difference = _prediction_difference(
        reference, baseline, before_masks=native_masks, after_masks=native_masks,
    )
    changed_trace = capture_compact_trace_tiled(model, changed_prepared)
    changed_support = summarize_compact_trace(changed_trace)
    weights = _as_numpy(changed_trace["compact_group_weights"])
    valid = _as_numpy(changed_trace["query_valid_mask"]).astype(bool)
    has_disabled_mass = bool((weights[..., group_index] != 0).any())
    if has_disabled_mass:
        raise ValueError("Disabled group retained nonzero query mass after copy-only intervention.")
    valid_weights = weights[valid]
    if valid_weights.size and not np.allclose(valid_weights.sum(axis=-1), 1.0, rtol=2e-4, atol=2e-4):
        raise ValueError("Remaining query-to-group weights were not freshly normalized after disabling a group.")
    compact = prepared.organization.compact_groups
    lineage = trace.get("compact_lineage", ())
    lineage_row = lineage[batch_index] if len(lineage) > batch_index else {}
    members = lineage_row.get("group_members", ()) if isinstance(lineage_row, Mapping) else ()
    keys = lineage_row.get("active_group_keys", ()) if isinstance(lineage_row, Mapping) else ()
    group_types = _as_numpy(compact.group_types)[batch_index]
    return {
        "schema": "round12_compact_single_group_intervention_v1",
        "selection": dict(role_mass),
        "selected_group": {
            "index": group_index,
            "type_code": int(group_types[group_index]),
            "original_record_members": list(map(int, members[group_index])) if group_index < len(members) else [],
            "lineage_key": keys[group_index] if group_index < len(keys) else None,
        },
        "transition": transition,
        "remaining_group_mass_normalization_max_abs_error": (
            0.0 if not valid_weights.size else float(np.abs(valid_weights.sum(axis=-1) - 1.0).max())
        ),
        "disabled_group_max_abs_query_mass": float(np.abs(weights[..., group_index]).max()),
        "native_field_deltas_vs_baseline": _prediction_difference(
            baseline, changed, before_masks=native_masks, after_masks=native_masks,
        ),
        "baseline_repeat_deltas_vs_existing_native_prediction": repeated_difference,
        "logical_final_support_after_intervention": changed_support["rows"],
        "baseline_native_work_counts": dict(baseline_work),
        "intervened_native_work_counts": dict(changed_work),
        "model_state_or_original_preparation_mutated": False,
    }


def _capture_compact_case_trace(model, batch, case_id: str, *, fluid_query_count: int,
                                local_query_count: int, prepared: Any | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch

    if prepared is None:
        receivers = batch.receivers
        fluid = receivers.fluid_xy
        if fluid.shape[1] < 1:
            raise ValueError("Compact trace needs at least one native fluid query.")
        indices = torch.linspace(0, fluid.shape[1] - 1, steps=min(fluid_query_count, fluid.shape[1]),
                                 device=fluid.device).round().long().unique(sorted=True)
        sampled_fluid = fluid.index_select(1, indices)
        sampled_local = _trace_subset(receivers.local_query_points, local_points=local_query_count)
        prepared = model.prepare_native(
            batch.scene_inputs.structure,
            sampled_fluid,
            local_query_points=sampled_local,
            ntheta=16,
            scene_key=batch.scene_inputs.case_ids,
        )
        selection = "bounded evenly spaced fluid queries plus bounded local material points"
    else:
        selection = "complete maintained native query union from the same prepared state used for physical prediction"
    trace = capture_compact_trace_tiled(model, prepared)
    summary = summarize_compact_trace(trace)
    role_mass = summarize_role_group_mass(trace, prepared)
    arrays: dict[str, Any] = {}
    trace_names = (
        "compact_membership", "compact_group_mask", "compact_group_types",
        "compact_group_controls", "compact_group_centroids", "compact_group_lengths",
        "compact_group_extents", "compact_live_K", "compact_allocated_K",
        "original_record_present", "original_record_eligible", "original_record_count",
        "module_ids", "environment_ids",
    )
    for name in trace_names:
        if name in trace:
            value = trace[name]
            if torch.is_tensor(value):
                arrays[name] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                arrays[name] = value
    sample_arrays, sample_manifest = _compact_trace_sample_arrays(
        trace, prepared, fluid_receiver_count=fluid_query_count,
        local_receiver_count=local_query_count,
    )
    arrays.update(sample_arrays)
    arrays["role_balanced_group_mass"] = np.asarray(role_mass["role_balanced_group_mass"], dtype=np.float64)
    arrays["query_count_weighted_group_mass"] = np.asarray(
        role_mass["query_count_weighted_group_mass"], dtype=np.float64,
    )
    for role, values in role_mass["roles"].items():
        arrays[f"role_{role}_mean_group_mass"] = np.asarray(
            values["mean_group_mass_per_native_receiver"], dtype=np.float64,
        )
    scene = prepared.organization.scene
    for name in ("module_coordinates", "module_mask", "module_characteristic_lengths",
                 "environment_coordinates", "environment_mask", "environment_measure"):
        value = getattr(scene, name, None)
        if value is not None:
            arrays["scene_" + name] = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    lineage = trace.get("compact_lineage", ())
    return arrays, {
        "case_id": case_id,
        "query_selection": selection,
        "trace_query_ids": _as_numpy(trace["query_ids"])[0].tolist() if "query_ids" in trace else [],
        "summary": summary,
        "role_group_mass": role_mass,
        "bounded_query_access_sample": {
            "sample_count": len(sample_manifest),
            "selected_from_same_prepared_trace": True,
            "coefficients_are_query_plane_values_not_interpolated_role_coefficients": True,
            "samples": sample_manifest,
        },
        "lineage": _json_safe(lineage),
    }, trace


def compact_organization_case_counts(prepared: Any, case_id: str) -> dict[str, Any]:
    """Count compact storage and typed group populations without a routing trace."""
    organization = getattr(prepared, "organization", None)
    compact = None if organization is None else getattr(organization, "compact_groups", None)
    if compact is None:
        raise TypeError("Organization counts require a prepared compact Thermal organization.")
    mask = _as_numpy(compact.group_mask)[0].astype(bool)
    types = _as_numpy(compact.group_types)[0]
    records = organization.joint_records
    topology = compact.topology[0].as_dict()
    return {
        "case_id": str(case_id),
        "live_K_M": int((mask & (types == 0)).sum()),
        "live_K_E": int((mask & (types == 1)).sum()),
        "live_K_total": int(mask.sum()),
        "allocated_K": int(mask.size),
        "present_N": int(_as_numpy(records.present)[0].astype(bool).sum()),
        "eligible_N": int(_as_numpy(records.eligible)[0].astype(bool).sum()),
        "allocated_N": int(compact.membership.shape[-1]),
        "candidate_pair_count": len(topology["candidate_pairs"]),
        "accepted_pair_count": len(topology["accepted_pairs"]),
        "query_scorer_slot_count": int(compact.score_slot_count),
    }


def summarize_organization_case_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    metrics = ("live_K_M", "live_K_E", "live_K_total", "allocated_K", "present_N", "eligible_N",
               "allocated_N", "candidate_pair_count", "accepted_pair_count", "query_scorer_slot_count")
    return {
        "case_count": len(rows),
        "case_ids": [str(row["case_id"]) for row in rows],
        "distributions": {
            key: {
                "mean": float(np.mean([int(row[key]) for row in rows])),
                "median": float(np.median([int(row[key]) for row in rows])),
                "minimum": int(min(int(row[key]) for row in rows)),
                "maximum": int(max(int(row[key]) for row in rows)),
            }
            for key in metrics
        },
    }


def evaluate_native_board_cli(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from honf_runtime.joint_native_evaluation import (
        reduce_thermal_records,
        thermal_record_and_arrays,
        typed_work_summary,
        write_arrays,
        write_json_new,
    )
    from honf_runtime.research_training import (
        _build_provider_for_resume,
        _load_run_manifest,
        _read_run_checkpoint,
    )

    run_dir = args.run_dir.resolve(strict=True)
    manifest, _ = _load_run_manifest(run_dir)
    payload, checkpoint = _read_run_checkpoint(run_dir, args.checkpoint)
    if args.expected_epoch is not None and int(payload.get("epoch", -1)) != args.expected_epoch:
        raise ValueError("Literal checkpoint epoch differs from the requested board label.")
    if payload.get("workflow") != "ordinary_honf_research_train_v1":
        raise ValueError("Native compact board requires a literal ordinary Thermal research checkpoint.")
    config = manifest["config"]
    optimizer_contract = manifest.get("optimizer_schedule_contract")
    if not isinstance(optimizer_contract, list) or not optimizer_contract:
        raise ValueError("Research run is missing its ordinary optimizer schedule contract.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model, provider = _build_provider_for_resume(
        config,
        payload,
        optimizer_contract,
        manifest.get("run_case_ids"),
        int(config["training"]["local_horizon"]),
        bool(manifest.get("diagnostic_only")),
        device,
    )
    model.eval()
    compact_core = getattr(model, "core", model)
    cut_flag = getattr(compact_core, "compact_cut_active", None)
    cut_active = None if cut_flag is None else bool(cut_flag.detach().cpu())
    requested_ids = args.case_id
    if args.case_id_file is not None:
        if requested_ids:
            raise ValueError("Choose either --case-id entries or one --case-id-file.")
        panel_value = json.loads(args.case_id_file.resolve(strict=True).read_text())
        if isinstance(panel_value, list):
            requested_ids = panel_value
        elif isinstance(panel_value, Mapping):
            requested_ids = next((panel_value[key] for key in ("selected_case_ids", "case_ids", "train24_case_ids")
                                  if isinstance(panel_value.get(key), list)), None)
        if not isinstance(requested_ids, list):
            raise ValueError("Case-ID file must contain a case-ID list or selected_case_ids/case_ids field.")
    case_ids = resolve_native_board_case_ids(
        args.split, requested_ids, representative_panel=bool(args.representative_panel),
    )
    trace_cases = tuple(map(str, args.trace_case or ()))
    export_cases = tuple(map(str, args.export_case or ()))
    intervention_case = None if args.intervention_case is None else str(args.intervention_case)
    if args.response_coefficients and set(export_cases) != set(case_ids):
        raise ValueError("Response-coefficient evidence requires exporting every case in the exact panel.")
    if len(set(trace_cases)) != len(trace_cases) or len(set(export_cases)) != len(export_cases):
        raise ValueError("Trace and array-export case selections cannot contain duplicate IDs.")
    if set(trace_cases) - set(case_ids) or set(export_cases) - set(case_ids):
        raise ValueError("Trace/export cases must belong to the exact native board panel.")
    if intervention_case is not None and (trace_cases.count(intervention_case) != 1 or intervention_case not in case_ids):
        raise ValueError("The one-group intervention case must be included exactly once in --trace-case.")
    output = args.output
    if output.exists():
        raise FileExistsError("Native board output directory already exists.")
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    organization_counts: list[dict[str, Any]] = []
    work: list[dict[str, Any]] = []
    trace_summaries: dict[str, Any] = {}
    group_interventions: dict[str, Any] = {}
    response_coefficient_work: dict[str, Any] = {}
    exported: list[dict[str, Any]] = []
    with torch.no_grad():
        batches = provider.native_evaluation_batches(split=args.split, case_keys=case_ids, microbatch_size=1)
        for batch in batches:
            if len(batch.case_keys) != 1:
                raise ValueError("Native physical panels require one original case per evaluation batch.")
            prediction, auxiliary = provider.predict_native(
                model, provider.make_scene(batch.scene_inputs), batch.receivers,
                str(getattr(model, "architecture", "A2-D")),
                "joint", epoch=int(payload["epoch"]), temperature=1.0,
            )
            case_id = str(batch.case_keys[0])
            row, arrays = thermal_record_and_arrays(prediction, batch)
            rows.append(row)
            organization_counts.append(compact_organization_case_counts(prediction.native_prepared, str(batch.case_keys[0])))
            work.append(dict(provider.work_counts(batch, prediction, auxiliary)))
            response_arrays: dict[str, Any] = {}
            if args.response_coefficients:
                response_arrays = native_response_coefficient_arrays(
                    model, prediction.native_prepared,
                )
                response_coefficient_work[case_id] = response_arrays.pop("response_K_column_work_counts")
            trace_arrays: dict[str, Any] = {}
            if case_id in trace_cases:
                trace_arrays, trace_summaries[case_id], full_trace = _capture_compact_case_trace(
                    model, batch, case_id, fluid_query_count=args.trace_fluid_queries,
                    local_query_count=args.trace_local_queries, prepared=prediction.native_prepared,
                )
                if case_id == intervention_case:
                    group_interventions[case_id] = run_single_group_intervention(
                        model,
                        prediction.native_prepared,
                        batch.receivers.heat,
                        full_trace,
                        trace_summaries[case_id]["role_group_mass"],
                        {name: arrays[name] for name in
                         ("fluid_valid", "interface_valid", "outside_valid", "material_valid")},
                        baseline_output=prediction.native_main,
                    )
            if case_id in export_cases:
                arrays.update(response_arrays)
                arrays.update({"trace__" + name: value for name, value in trace_arrays.items()})
                array_path = output / f"case_{case_id}.npz"
                write_arrays(array_path, arrays)
                exported.append({"case_id": case_id, "array_path": array_path.name})
            elif trace_arrays:
                array_path = output / f"case_{case_id}_organization.npz"
                write_arrays(array_path, trace_arrays)
                trace_summaries[case_id]["array_path"] = array_path.name
    observed = tuple(str(row["case_id"]) for row in rows)
    if observed != case_ids:
        raise ValueError("Native board did not cover the exact requested case order.")
    if args.split == "validation" and not args.representative_panel:
        validate_case_panel(observed, VAL89_IDS)
    result = {
        "schema": "round12_thermal_native_compact_board_v1",
        "status": "completed",
        "run_directory": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "cut_active": cut_active,
        "sealed_research_config": payload["experiment_identity"].get("config_name"),
        "split": args.split,
        "representative_panel": bool(args.representative_panel),
        "evidence_status": ("representative_boundary_or_probe_panel_not_terminal_population_evaluation"
                            if args.representative_panel else "full_terminal_panel"),
        "case_ids": list(case_ids),
        "case_count": len(rows),
        "physical_receiver_policy": "maintained native Thermal full receiver extraction for fluid8192, surface/outside native64 and every stored material point",
        "records": rows,
        "metrics": reduce_thermal_records(rows),
        "organization_counts": organization_counts,
        "organization_count_summary": summarize_organization_case_counts(organization_counts),
        "typed_work": typed_work_summary(work),
        "work_counts": work,
        "organization_traces": trace_summaries,
        "group_interventions": group_interventions,
        "response_coefficient_work_counts": response_coefficient_work,
        "response_coefficients_exported": bool(args.response_coefficients),
        "exports": exported,
        "optimizer_updates": 0,
        "new_solver_attempts": 0,
        "TEST_or_Wind_access": False,
        "device": str(device),
    }
    write_json_new(output / "result.json", result)
    return {"status": "completed", "split": args.split, "case_count": len(rows),
            "trace_case_count": len(trace_summaries), "output": str(output.resolve())}


def compare_native_boards_cli(args: argparse.Namespace) -> dict[str, Any]:
    from honf_runtime.joint_native_evaluation import write_json_new

    before_dir, after_dir = args.before_dir.resolve(strict=True), args.after_dir.resolve(strict=True)
    before_path, after_path = before_dir / "result.json", after_dir / "result.json"
    before, after = json.loads(before_path.read_text()), json.loads(after_path.read_text())
    result = compare_native_board_pair_arrays(before, after, before_dir, after_dir)
    if args.output.exists():
        raise FileExistsError("Boundary comparison output already exists.")
    write_json_new(args.output, result)
    return {"status": "completed", "case_count": len(result["case_ids"]),
            "output": str(args.output.resolve())}


def evaluate_geometry_probes_cli(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from honf_runtime.joint_native_evaluation import write_json_new
    from honf_runtime.research_training import (
        _build_provider_for_resume,
        _load_run_manifest,
        _read_run_checkpoint,
    )

    run_dir = args.run_dir.resolve(strict=True)
    manifest, _ = _load_run_manifest(run_dir)
    payload, checkpoint = _read_run_checkpoint(run_dir, args.checkpoint)
    if args.expected_epoch is not None and int(payload.get("epoch", -1)) != args.expected_epoch:
        raise ValueError("Literal checkpoint epoch differs from the requested geometry-probe label.")
    if payload.get("workflow") != "ordinary_honf_research_train_v1":
        raise ValueError("Geometry probes require a literal ordinary Thermal research checkpoint.")
    config = manifest["config"]
    optimizer_contract = manifest.get("optimizer_schedule_contract")
    if not isinstance(optimizer_contract, list) or not optimizer_contract:
        raise ValueError("Research run is missing its ordinary optimizer schedule contract.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model, provider = _build_provider_for_resume(
        config,
        payload,
        optimizer_contract,
        manifest.get("run_case_ids"),
        int(config["training"]["local_horizon"]),
        bool(manifest.get("diagnostic_only")),
        device,
    )
    model.eval()
    compact_core = getattr(model, "core", model)
    cut_flag = getattr(compact_core, "compact_cut_active", None)
    cut_active = False if cut_flag is None else bool(cut_flag.detach().cpu())
    if not cut_active:
        raise ValueError("Geometry derivative/seam probes require the literal post-cut compact checkpoint.")
    expected_ids = ("0006", "0254")
    if tuple(args.case_id or expected_ids) != expected_ids:
        raise ValueError("Round12 legal geometry probes use the fixed low/high-M TRAIN IDs 0006 and 0254.")
    rows = []
    batches = provider.native_evaluation_batches(split="train", case_keys=expected_ids, microbatch_size=1)
    for batch in batches:
        case_id = str(batch.case_keys[0])
        if len(batch.case_keys) != 1:
            raise ValueError("Geometry probes require one original TRAIN scene per batch.")
        row = _geometry_probe_one_case(model, batch, case_id)
        row["topology_seam_scan"] = _probe_topology_seam(
            model, batch, row["translation"],
        )
        rows.append(row)
    if tuple(row["case_id"] for row in rows) != expected_ids:
        raise ValueError("Geometry probes did not cover the exact ordered low/high-M TRAIN pair.")
    result = {
        "schema": "round12_compact_live_geometry_derivative_and_seam_probe_v1",
        "status": "completed",
        "run_directory": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "cut_active": cut_active,
        "sealed_research_config": payload["experiment_identity"].get("config_name"),
        "partition": "train",
        "case_ids": list(expected_ids),
        "cases": rows,
        "protocol": "Each derivative evaluation shifts one legal original module coordinate, rebuilds the Thermal scene/context and full native receiver preparation, then compares autograd with a represented FP32 central-difference h sweep only where actual candidate ordering, ancestry and b>0 cut agree on both sides. A separate bounded legal scan rebuilds the topology and reports any crossing with shrinking-gap native outputs.",
        "optimizer_updates": 0,
        "new_solver_attempts": 0,
        "TEST_or_Wind_access": False,
        "device": str(device),
    }
    output = args.output
    write_json_new(output, result)
    return {"status": "completed", "case_count": len(rows), "output": str(output.resolve())}


def evaluate_saved_ranking_cli(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from honf_runtime.joint_native_evaluation import write_json_new
    from honf_runtime.research_training import (
        _load_run_manifest,
        _model_from_run_payload,
        _read_run_checkpoint,
    )

    run_dir = args.run_dir.resolve(strict=True)
    run_manifest, _ = _load_run_manifest(run_dir)
    payload, checkpoint = _read_run_checkpoint(run_dir, args.checkpoint)
    if payload.get("workflow") != "ordinary_honf_research_train_v1":
        raise ValueError("Saved-candidate ranking requires a literal Thermal research checkpoint.")
    if args.expected_epoch is not None and int(payload.get("epoch", -1)) != args.expected_epoch:
        raise ValueError("Literal checkpoint epoch differs from the requested evidence label.")
    config_name = str(run_manifest["config"].get("config_name", ""))
    identity_name = str(payload["experiment_identity"].get("config_name", ""))
    if config_name and identity_name and config_name != identity_name:
        raise ValueError("Run manifest and selected checkpoint name different sealed research configs.")
    arm_payload = response_payload_for_arm(payload, args.arm_label)

    ranking_loader = args.ranking_loader.resolve(strict=True)
    diagnostic = _module_from_path("round12_saved_ranking_diagnostic", ranking_loader)
    reference = diagnostic.load_w6_ranking_reference()
    consumer, old, _inventory, saved, _availability, study, _proposals, _references, raw_rows, controls = reference
    rows = validate_w6_ranking_reference(raw_rows, saved, study)
    if saved.get("_baseline_total_heat") != controls["baseline_total_heat"] or saved.get("_pressure_limit") != controls["pressure"]["limit"]:
        raise ValueError("Saved ranking inputs differ from the original physical control binding.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = _model_from_run_payload(payload, torch.device(args.device))
    options = arm_payload["experiment_identity"].get("execution_options")
    if isinstance(options, Mapping):
        setter = getattr(model.core, "set_execution_options", None)
        if callable(setter):
            setter(**dict(options))
    model.eval()

    predictions: dict[str, Any] = {}
    reader_work: dict[str, Any] = {}
    candidate_seconds: dict[str, float] = {}
    started = time.monotonic()
    for candidate in rows:
        candidate_id = str(candidate["candidate_id"])
        with consumer.measured_native_reader_hooks(model) as hooks:
            tick = time.perf_counter()
            predicted = old.predict_rows(model, [candidate], saved, lambda: None, str(args.device))
            candidate_seconds[candidate_id] = time.perf_counter() - tick
        if set(predicted) != {candidate_id}:
            raise ValueError("Saved-candidate native prediction changed its original candidate ID.")
        result = predicted[candidate_id]
        pressure = float(result["pressure_drop"])
        objective = float(result["objective"])
        geometry_valid = result["geometry_valid"]
        feasible = result["feasible"]
        if (not math.isfinite(pressure) or not math.isfinite(objective)
                or type(geometry_valid) is not bool or type(feasible) is not bool):
            raise ValueError("Saved-candidate native prediction has invalid physical decision fields.")
        if feasible != (geometry_valid and pressure <= controls["pressure"]["limit"]):
            raise ValueError("Predicted feasibility no longer follows geometry plus the original pressure limit.")
        predictions.update(predicted)
        reader_work[candidate_id] = dict(hooks)

    reference_rows = {str(row["candidate_id"]): row["reference"] for row in rows}
    if set(predictions) != set(reference_rows) or len(predictions) != 6:
        raise ValueError("Predictions and original physical references must cover the same six candidate IDs.")
    arm = str(arm_payload["experiment_identity"]["treatment"])
    summary = consumer.summarize({arm: predictions}, reference_rows, float(study["pressure_drop_limit"]))[arm]
    confusion = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    for candidate_id, prediction in predictions.items():
        predicted_feasible = bool(prediction["feasible"])
        reference_feasible = bool(reference_rows[candidate_id]["feasible"])
        if predicted_feasible and reference_feasible:
            confusion["true_positive"] += 1
        elif predicted_feasible:
            confusion["false_positive"] += 1
        elif reference_feasible:
            confusion["false_negative"] += 1
        else:
            confusion["true_negative"] += 1
    summary["feasibility_confusion"] = confusion
    summary["candidate_count"] = 6
    summary["candidate_ids"] = list(W6_CANDIDATE_IDS)
    output = args.output
    write_json_new(output / "result.json", {
        "schema": "round12_thermal_saved_candidate_ranking_v1",
        "status": "completed_same_checkpoint_replay",
        "checkpoint_epoch": int(payload["epoch"]),
        "arm": arm,
        "candidate_ids": list(W6_CANDIDATE_IDS),
        "controls": controls,
        "ranking": summary,
        "references": reference_rows,
        "predictions": predictions,
        "reader_work": reader_work,
        "complete_candidate_seconds": candidate_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "six already-solved Thermal M3 candidates; existing native receiver/objective extraction; no new candidate generation, inverse optimization, optimizer update, or physical solver",
        "optimizer_updates": 0,
        "new_solver_attempts": 0,
        "TEST_or_Wind_access": False,
    })
    write_json_new(output / "provenance.json", {
        "schema": "thermal_saved_candidate_ranking_replay_provenance_v1",
        "run_directory": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "sealed_research_config": identity_name,
        "derived_arm_label": arm,
        "ranking_loader": str(ranking_loader),
        "candidate_ids": list(W6_CANDIDATE_IDS),
        "tau_temperature": 2.0,
        "pressure_limit": float(study["pressure_drop_limit"]),
        "optimizer_updates": 0,
        "new_solver_attempts": 0,
        "TEST_or_Wind_access": False,
        "device": str(args.device),
    })
    return {"status": "completed", "arm": arm, "epoch": int(payload["epoch"]),
            "candidate_count": 6, "output": str(output.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    response = subparsers.add_parser("responses", help="Replay existing saved TRAIN physical responses.")
    response.add_argument("--run-dir", type=Path, required=True)
    response.add_argument("--checkpoint", required=True, help="Run-relative selector or checkpoint filename.")
    response.add_argument("--expected-epoch", type=int)
    response.add_argument("--arm-label")
    response.add_argument("--inventory", type=Path, required=True)
    response.add_argument("--output", type=Path, required=True)
    response.add_argument("--device", default="cpu")
    ranking = subparsers.add_parser("rankings", help="Replay the original six saved Thermal M3 candidates.")
    ranking.add_argument("--run-dir", type=Path, required=True)
    ranking.add_argument("--checkpoint", required=True, help="Run-relative selector or checkpoint filename.")
    ranking.add_argument("--expected-epoch", type=int)
    ranking.add_argument("--arm-label")
    ranking.add_argument("--ranking-loader", type=Path, required=True,
                         help="Existing Round10 diagnostic that authenticates and loads the saved W6 reference.")
    ranking.add_argument("--output", type=Path, required=True)
    ranking.add_argument("--device", default="cpu")
    heat_reuse = subparsers.add_parser(
        "heat-reuse-summary",
        help="Summarize precision-promoted saved response increments; this does not demonstrate retained-cache reuse.",
    )
    heat_reuse.add_argument("--response-result", action="append", required=True, metavar="ARM=PATH")
    heat_reuse.add_argument("--output", type=Path, required=True)
    actual_cache = subparsers.add_parser(
        "actual-cache-k",
        help="Promote one saved baseline preparation and reuse retained affine factors at its existing heat endpoints.",
    )
    actual_cache.add_argument("--run-dir", type=Path, required=True)
    actual_cache.add_argument("--checkpoint", required=True, help="Run-relative selector or checkpoint filename.")
    actual_cache.add_argument("--expected-epoch", type=int)
    actual_cache.add_argument("--arm-label", required=True)
    actual_cache.add_argument("--response-result", type=Path, required=True)
    actual_cache.add_argument("--inventory", type=Path, required=True)
    actual_cache.add_argument("--output", type=Path, required=True)
    actual_cache.add_argument("--device", default="cpu")
    board = subparsers.add_parser("native-board", help="Run full native field evaluation on VAL89 or an explicit fixed panel.")
    board.add_argument("--run-dir", type=Path, required=True)
    board.add_argument("--checkpoint", required=True)
    board.add_argument("--expected-epoch", type=int)
    board.add_argument("--split", choices=("train", "validation"), required=True)
    board.add_argument("--representative-panel", action="store_true",
                       help="Use only the exact four VAL or two TRAIN boundary/probe cases; label as nonterminal evidence.")
    board.add_argument("--case-id", action="append")
    board.add_argument("--case-id-file", type=Path)
    board.add_argument("--trace-case", action="append")
    board.add_argument("--export-case", action="append")
    board.add_argument("--intervention-case", help="Run one copy-only most-used-group native intervention; include the same ID in --trace-case.")
    board.add_argument("--response-coefficients", action="store_true",
                       help="Recompute one-hot native incremental heat-response K columns and export them with every panel case.")
    board.add_argument("--trace-fluid-queries", type=int, default=16)
    board.add_argument("--trace-local-queries", type=int, default=8)
    board.add_argument("--output", type=Path, required=True)
    board.add_argument("--device", default="cpu")
    compare = subparsers.add_parser("compare-boundary", help="Compare matched pre/post-cut representative native boards and K arrays.")
    compare.add_argument("--before-dir", type=Path, required=True)
    compare.add_argument("--after-dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    geometry = subparsers.add_parser("geometry-probes", help="Compare live AD/FD on two fixed TRAIN scenes and scan legal topology seams.")
    geometry.add_argument("--run-dir", type=Path, required=True)
    geometry.add_argument("--checkpoint", required=True)
    geometry.add_argument("--expected-epoch", type=int)
    geometry.add_argument("--case-id", action="append", help="Optional exact pair: 0006 then 0254.")
    geometry.add_argument("--output", type=Path, required=True)
    geometry.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "responses":
        result = evaluate_saved_response_cli(args)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.command == "rankings":
        result = evaluate_saved_ranking_cli(args)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.command == "heat-reuse-summary":
        result = summarize_parent_heat_reuse_cli(args)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.command == "actual-cache-k":
        result = evaluate_actual_cached_k_cli(args)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.command == "native-board":
        if args.trace_fluid_queries < 1 or args.trace_local_queries < 1:
            raise ValueError("Compact trace fluid/local query selections must be positive.")
        result = evaluate_native_board_cli(args)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.command == "compare-boundary":
        result = compare_native_boards_cli(args)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.command == "geometry-probes":
        result = evaluate_geometry_probes_cli(args)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    raise ValueError(f"Unknown evaluator command {args.command!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
