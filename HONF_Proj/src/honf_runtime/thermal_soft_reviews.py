"""Bounded, in-process evidence for the Thermal soft-only compact arm.

Organizer reviews use a fixed input-selected TRAIN panel and retain only JSON
metrics. Final native and retained-response validation runs after fitting and
uses exact checkpoint aliases. Neither path changes model weights or launches
training, a solver, or an external monitor.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REVIEW_EPOCHS = (100, 300, 600, 1000)
P_THRESHOLD = 0.8
NORMALIZED_C_THRESHOLD = 0.1
INTERVENTION_RMSE_FRACTION = 0.1
NONDEGENERATE_MASS = 1.0e-8
INTERNAL_OUTPUT_BUDGET = 2 * 1024**2
FINAL_OUTPUT_BUDGET = 1024**3
MINIMUM_FREE_HEADROOM = 200 * 1024**3
PRIORITY_FIELDS = (
    "temperature_fluid",
    "temperature_surface",
    "temperature_material",
    "module_peak_endpoint",
)
THERMAL_FIELD_UNITS = {
    "u": "dataset velocity units",
    "v": "dataset velocity units",
    "p": "dataset pressure units",
    "omega": "dataset vorticity units",
    "temperature_fluid": "dataset temperature units",
    "temperature_surface": "dataset temperature units",
    "temperature_material": "dataset temperature units",
    "q_proxy": "dataset heat-flux proxy units",
    "module_peak_endpoint": "dataset temperature units",
}
THERMAL_MASK_POLICY = {
    "u_v_p_omega_temperature_fluid": "fluid stencil valid AND positive native point weight AND finite target",
    "temperature_surface": "surface stencil valid AND physically present module AND saved interface-valid mask",
    "q_proxy": "surface mask AND outside stencil valid",
    "temperature_material": "material stencil valid AND physically present module AND finite target",
    "module_peak_endpoint": "maximum prediction and target recomputed separately over valid material points for each present module",
}
TRAIN_RESPONSE_IDS = ("0001", "0318", "0333", "0348")
DEVELOPMENT_RESPONSE_IDS = ("0304", "0320", "0335", "0350")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _weights_sha256(model: torch.nn.Module) -> str:
    """Hash named parameters only, excluding mutable execution counters/buffers."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda row: row[0]):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(repr((tuple(value.shape), str(value.dtype))).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_snapshot(model: torch.nn.Module) -> dict[str, Any]:
    from .research_training import _cpu_state_snapshot
    return _cpu_state_snapshot(model.state_dict())


def _state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(repr((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, Mapping):
            digest.update(b"mapping\0")
            for key, item in sorted(value.items(), key=lambda row: str(row[0])):
                digest.update(str(key).encode("utf-8") + b"\0")
                visit(item)
        elif isinstance(value, (tuple, list)):
            digest.update(repr((type(value).__name__, len(value))).encode("ascii"))
            for item in value:
                visit(item)
        else:
            digest.update(json.dumps(_jsonable(value), sort_keys=True).encode("utf-8") + b"\0")
    visit(state)
    return digest.hexdigest()


def _same_value(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys()
                and all(_same_value(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (isinstance(left, (tuple, list)) and isinstance(right, (tuple, list))
                and len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right)))
    return left == right


def _select_train_panel(training_cases: Sequence[Mapping[str, Any]], limit: int = 4) -> tuple[str, ...]:
    """Choose min/next/median/max module-count strata using input fields only."""
    if not training_cases or not 1 <= limit <= 4:
        raise ValueError("The organizer review needs a nonempty TRAIN cohort and a panel limit from one to four.")
    counts: dict[str, int] = {}
    for case in training_cases:
        case_id = str(case["case_id"])
        present = np.asarray(case["structure"]["module_present"])
        if present.ndim != 1 or not np.isfinite(present).all():
            raise ValueError(f"TRAIN input {case_id} has invalid module_present metadata.")
        counts[case_id] = int((present > 0).sum())
    if len(counts) != len(training_cases):
        raise ValueError("TRAIN review inputs contain duplicate case IDs.")
    strata = sorted(set(counts.values()))
    indices = (0, min(1, len(strata) - 1), len(strata) // 2, len(strata) - 1)
    selected: list[str] = []
    for index in indices:
        candidates = sorted((key for key, count in counts.items() if count == strata[index]))
        if candidates and candidates[0] not in selected:
            selected.append(candidates[0])
    for case_id in sorted(counts):
        if len(selected) >= limit:
            break
        if case_id not in selected:
            selected.append(case_id)
    return tuple(selected[:limit])


def _normalized_pair_discrepancy(
    left: np.ndarray,
    right: np.ndarray,
    eligible: np.ndarray,
    module_count: int,
    *,
    epsilon: float = NONDEGENERATE_MASS,
) -> dict[str, Any]:
    """Compare typed donor distributions after removing each row's L1 scale.

    For each nonempty M/E donor slice, normalize left and right by their own
    L1 mass and return total-variation distance (0.5 * L1). Available types are
    equally weighted. Amplitude and degenerate/empty evidence remain explicit.
    """
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    eligible = np.asarray(eligible, dtype=bool).reshape(-1)
    if left.shape != right.shape or left.shape != eligible.shape:
        raise ValueError("Pair rows and eligible donor mask must align exactly.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Pair membership rows must be finite on the full native record axis.")
    if (left < -1.0e-12).any() or (right < -1.0e-12).any():
        raise ValueError("Compact C membership rows must be nonnegative routing masses.")
    module_count = int(module_count)
    if not 0 <= module_count <= left.size:
        raise ValueError("module_count is outside the membership donor axis.")
    left = np.where(eligible, left, 0.0)
    right = np.where(eligible, right, 0.0)
    left = np.maximum(left, 0.0)
    right = np.maximum(right, 0.0)
    left_l1 = float(np.abs(left).sum())
    right_l1 = float(np.abs(right).sum())
    left_l2 = float(np.linalg.norm(left))
    right_l2 = float(np.linalg.norm(right))
    typed: dict[str, Any] = {}
    discrepancies = []
    skipped = []
    degenerate_eligible_types = []
    for name, selection in (("M", slice(0, module_count)), ("E", slice(module_count, left.size))):
        mask = eligible[selection]
        if not mask.any():
            typed[name] = {"available": False, "reason": "no eligible donors in this type"}
            skipped.append(name)
            continue
        lvalues = left[selection][mask]
        rvalues = right[selection][mask]
        lmass = float(np.abs(lvalues).sum())
        rmass = float(np.abs(rvalues).sum())
        if lmass <= epsilon or rmass <= epsilon:
            typed[name] = {"available": False, "reason": "one or both typed row masses are degenerate",
                           "left_l1": lmass, "right_l1": rmass}
            skipped.append(name)
            degenerate_eligible_types.append(name)
            continue
        distance = 0.5 * float(np.abs(lvalues / lmass - rvalues / rmass).sum())
        discrepancies.append(distance)
        typed[name] = {"available": True, "normalized_total_variation": distance,
                       "left_l1": lmass, "right_l1": rmass,
                       "eligible_donor_count": int(mask.sum())}
    total_mass_nondegenerate = left_l1 > epsilon and right_l1 > epsilon
    nondegenerate = bool(total_mass_nondegenerate and bool(discrepancies)
                         and not degenerate_eligible_types)
    result = {
        "normalized_rowdiff": float(np.mean(discrepancies)) if discrepancies else None,
        "normalized_rowdiff_definition": "equal mean over available M/E donor types of 0.5*L1 between rows normalized by each typed L1 mass",
        "typed_donors": typed,
        "typed_empty_or_degenerate_skipped": skipped,
        "degenerate_eligible_types": degenerate_eligible_types,
        "nondegenerate": nondegenerate,
        "left_row_l1_mass": left_l1,
        "right_row_l1_mass": right_l1,
        "left_row_l2_magnitude": left_l2,
        "right_row_l2_magnitude": right_l2,
        "raw_row_l1_difference": float(np.abs(left - right).sum()),
        "raw_row_l2_difference": float(np.linalg.norm(left - right)),
        "degenerate_epsilon": float(epsilon),
    }
    return result


def _candidate_is_promising(probability: float, discrepancy: Mapping[str, Any]) -> bool:
    value = discrepancy.get("normalized_rowdiff")
    return bool(
        math.isfinite(float(probability))
        and float(probability) >= P_THRESHOLD
        and value is not None
        and math.isfinite(float(value))
        and float(value) <= NORMALIZED_C_THRESHOLD
        and discrepancy.get("nondegenerate") is True
    )


def _select_intervention_candidates(candidates: Sequence[Mapping[str, Any]], *, limit_per_bank: int = 2
                                   ) -> list[tuple[Mapping[str, Any], str]]:
    """Rank a bounded panel globally, preferring qualified pairs then p and stable IDs."""
    if limit_per_bank < 0:
        raise ValueError("Intervention candidate limit cannot be negative.")
    selected: list[tuple[Mapping[str, Any], str]] = []
    for bank in ("M", "E"):
        eligible = [row for row in candidates if row.get("bank") == bank]
        eligible.sort(key=lambda row: (
            -int(bool(row.get("promising"))),
            -float(row.get("probability", float("-inf"))),
            str(row.get("case_id", "")),
            tuple(map(int, row.get("slots", ()))),
            int(row.get("candidate_index", -1)),
        ))
        for row in eligible[:limit_per_bank]:
            basis = "promising_threshold" if bool(row.get("promising")) else "highest_probability_unqualified_probe"
            selected.append((row, basis))
    return selected


def _pair_key(case_id: str, bank: str, slots: Sequence[int]) -> str:
    left, right = sorted(map(int, slots))
    return f"{case_id}|{bank}|{left}:{right}"


def _stable_candidate_key(row: Mapping[str, Any]) -> str:
    siblings = row.get("sibling_native_ids")
    if siblings:
        identities = sorted((str(item["record_type"]), str(item["native_id"])) for item in siblings)
        return f"{row['case_id']}|{row['bank']}|" + ";".join(
            f"{kind}:{native_id}" for kind, native_id in identities)
    return _pair_key(str(row["case_id"]), str(row["bank"]), row["slots"])


def _candidate_stability(current: Sequence[Mapping[str, Any]], previous_path: Path | None,
                         current_panel: Sequence[str]) -> dict[str, Any]:
    if previous_path is None or not previous_path.is_file():
        return {"available": False, "reason": "no previous scheduled organizer review"}
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    old_rows = previous.get("candidates", [])
    old = {_stable_candidate_key(row): row for row in old_rows}
    new = {_stable_candidate_key(row): row for row in current}
    old_promising = {key for key, row in old.items() if row.get("promising")}
    new_promising = {key for key, row in new.items() if row.get("promising")}
    union = old_promising | new_promising
    shared = sorted(set(old) & set(new))
    changes = [{"pair": key,
                "probability_delta": float(new[key]["probability"] - old[key]["probability"]),
                "normalized_rowdiff_delta": (
                    None if old[key].get("normalized_rowdiff") is None or new[key].get("normalized_rowdiff") is None
                    else float(new[key]["normalized_rowdiff"] - old[key]["normalized_rowdiff"])),
                "nondegenerate_both_stages": bool(old[key].get("nondegenerate") and new[key].get("nondegenerate"))}
               for key in shared]
    return {
        "available": True,
        "previous_epoch": previous.get("epoch"),
        "previous_panel_case_ids": previous.get("panel_case_ids", []),
        "same_input_panel": previous.get("panel_case_ids") == list(current_panel),
        "eligible_pair_key_overlap": len(shared),
        "eligible_pair_keys_added": sorted(set(new) - set(old)),
        "eligible_pair_keys_removed": sorted(set(old) - set(new)),
        "promising_pair_jaccard": float(len(old_promising & new_promising) / len(union)) if union else 1.0,
        "promising_pairs_added": sorted(new_promising - old_promising),
        "promising_pairs_removed": sorted(old_promising - new_promising),
        "matching_pair_changes": changes,
    }


def _review_storage(output: Path, *, estimated_bytes: int, peak_temporary_bytes: int,
                    config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data2 = Path("/data2")
    if not data2.is_dir() or not os.path.ismount(data2):
        raise RuntimeError("/data2 is not mounted; refusing to write Thermal review evidence.")
    resolved = output.resolve(strict=True)
    if not resolved.is_relative_to(data2.resolve(strict=True)):
        raise ValueError("Thermal review output must resolve inside the mounted /data2 workspace.")
    if os.stat(resolved).st_dev != os.stat(data2).st_dev:
        raise RuntimeError("Thermal review output is not on the /data2 filesystem.")
    usage = os.statvfs(data2)
    free = int(usage.f_bavail * usage.f_frsize)
    headroom = (int(config["storage"]["minimum_free_headroom_bytes"])
                if config is not None else MINIMUM_FREE_HEADROOM)
    if free - estimated_bytes < headroom:
        raise RuntimeError("Thermal review output would breach the fixed /data2 free-space headroom.")
    configured = None
    if config is not None:
        from .research_training import verify_storage
        configured = verify_storage(config, output=resolved, estimated_bytes=estimated_bytes,
                                    peak_temporary_bytes=peak_temporary_bytes)
    return {"data2_mount": str(data2.resolve()), "output_path": str(resolved),
            "filesystem_device": int(os.stat(data2).st_dev), "free_bytes_before": free,
            "hard_new_output_budget_bytes": int(estimated_bytes),
            "estimated_peak_temporary_bytes": int(peak_temporary_bytes),
            "minimum_free_headroom_bytes": headroom,
            "configured_storage_check": configured}


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _module_count(case: Mapping[str, Any]) -> int:
    return int((np.asarray(case["structure"]["module_present"]) > 0).sum())


def _candidate_rows(organization: Any, case_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compact = organization.compact_groups
    records = organization.joint_records
    membership = compact.membership[0].detach().double().cpu().numpy()
    pair_mask = compact.pair_mask[0].detach().bool().cpu().numpy()
    pair_slots = compact.pair_slots[0].detach().long().cpu().numpy()
    logits = compact.pair_logits[0].detach().double().cpu().numpy()
    pair_types = compact.pair_types[0].detach().long().cpu().numpy()
    eligible = records.eligible[0].detach().bool().cpu().numpy()
    native_ids = records.ids[0].detach().cpu().tolist()
    record_types = records.types[0].detach().cpu().tolist()
    group_members = compact.topology[0].group_members
    module_count = int(records.module_count)
    candidates = []
    for index in np.flatnonzero(pair_mask):
        left, right = map(int, pair_slots[index])
        bank_id = int(pair_types[index])
        bank = "M" if bank_id == 0 else "E" if bank_id == 1 else None
        if bank is None:
            raise ValueError(f"Unknown compact candidate bank type {bank_id}.")
        probability = float(1.0 / (1.0 + math.exp(-float(np.clip(logits[index], -80, 80)))))
        discrepancy = _normalized_pair_discrepancy(membership[left], membership[right], eligible, module_count)
        sibling_native_ids = []
        for slot in (left, right):
            for record_index in group_members[slot]:
                sibling_native_ids.append({"record_type": "M" if int(record_types[record_index]) == 0 else "E",
                                           "native_id": int(native_ids[record_index])})
        row = {"case_id": case_id, "candidate_index": int(index), "bank": bank,
               "slots": [left, right], "sibling_native_ids": sibling_native_ids,
               "probability": probability,
               "normalized_rowdiff": discrepancy["normalized_rowdiff"],
               "nondegenerate": discrepancy["nondegenerate"],
               "promising": _candidate_is_promising(probability, discrepancy),
               "membership_evidence": discrepancy}
        candidates.append(row)
    live_k = int(compact.group_mask[0].sum().detach().cpu())
    present_n = int(records.present[0].sum().detach().cpu())
    topology = compact.topology[0]
    singleton_groups = all(len(members) == 1 for members in topology.group_members)
    audit = {"live_K": live_k, "allocated_K": int(compact.group_mask.shape[-1]),
             "present_native_N": present_n, "membership_donor_N": int(membership.shape[-1]),
             "actual_K_equals_N": bool(live_k == present_n and singleton_groups),
             "singleton_native_lineage": bool(singleton_groups),
             "candidate_count": int(pair_mask.sum()),
             "candidate_count_by_bank": {bank: int(sum(row["bank"] == bank for row in candidates))
                                          for bank in ("M", "E")}}
    return candidates, audit


def _query_access_summary(model: torch.nn.Module, organization: Any, query: Any) -> dict[str, Any]:
    core = model.core
    compact = organization.compact_groups
    qmask = query.query_mask
    if qmask is None:
        qmask = torch.ones(query.query_ids.shape, dtype=torch.bool, device=query.query_ids.device)
    tile_size = int(getattr(core, "query_tile_size", 0))
    if tile_size < 1:
        raise ValueError("Bounded query-access review requires a positive core.query_tile_size.")
    from honf_forward_core.content_hypergraph.query_chunks_v1 import slice_query_v1
    from honf_forward_core.content_hypergraph.decoding import pack_local_pairs
    accumulator = _new_query_access_accumulator()
    query_count = int(query.query_ids.shape[1])
    for start in range(0, query_count, tile_size):
        stop = min(start + tile_size, query_count)
        tile = slice_query_v1(query, start, stop)
        tile_mask = qmask[:, start:stop]
        query_states = core._query_states(organization, tile)
        local = pack_local_pairs(organization.scene, tile, organization.frame_scale)
        envelope = query_states.new_zeros(query_states.shape[0], query_states.shape[1], organization.scene.num_modules)
        if local.count:
            envelope = envelope.index_put(
                (local.batch_index, local.query_index, local.module_index), local.support_weight,
            )
        gates, weights, logits = core.compact_retrieval.query_access(
            compact, organization, tile, query_states, envelope, tile_mask,
        )
        _accumulate_query_access(accumulator, gates, weights, logits, tile_mask,
                                 organization.joint_records.eligible, compact.group_mask)
        del tile, query_states, envelope, gates, weights, logits
    q_count = accumulator["query_count"]
    donor_count = accumulator["eligible_query_record_pairs"]
    logit_count = accumulator["live_query_group_pairs"]
    return {
        "available": True,
        "query_count": q_count,
        "eligible_query_record_pairs": donor_count,
        "positive_logical_access_pairs": accumulator["positive_logical_access_pairs"],
        "positive_logical_access_fraction": (
            accumulator["positive_logical_access_pairs"] / donor_count if donor_count else None),
        "mean_logical_access_value": accumulator["logical_access_sum"] / donor_count if donor_count else None,
        "mean_group_weight_entropy_nats": accumulator["entropy_sum"] / q_count if q_count else None,
        "mean_group_logit": accumulator["logit_sum"] / logit_count if logit_count else None,
        "query_tile_size": tile_size,
        "query_tile_count": math.ceil(query_count / tile_size),
        "interpretation": "logical differentiable access only; executed sparse work and executor savings were not measured",
        "executor_savings_measured": False,
    }


def _new_query_access_accumulator() -> dict[str, float | int]:
    return {"query_count": 0, "eligible_query_record_pairs": 0,
            "positive_logical_access_pairs": 0, "logical_access_sum": 0.0,
            "entropy_sum": 0.0, "live_query_group_pairs": 0, "logit_sum": 0.0}


def _accumulate_query_access(accumulator: dict[str, float | int], gates: torch.Tensor,
                             weights: torch.Tensor, logits: torch.Tensor,
                             query_mask: torch.Tensor, eligible: torch.Tensor,
                             group_mask: torch.Tensor) -> None:
    donor_mask = eligible[:, None, :] & query_mask[..., None]
    gate_values = gates[donor_mask]
    query_total = int(query_mask.sum().detach().cpu())
    positive_count = int((gate_values > 0).sum().detach().cpu())
    accumulator["query_count"] += query_total
    accumulator["eligible_query_record_pairs"] += int(donor_mask.sum().detach().cpu())
    accumulator["positive_logical_access_pairs"] += positive_count
    accumulator["logical_access_sum"] += float(gate_values.sum().detach().cpu())
    if query_total:
        group_values = weights[query_mask]
        tiny = torch.finfo(group_values.dtype).tiny
        entropy = -(group_values * group_values.clamp_min(tiny).log()).sum(-1)
        accumulator["entropy_sum"] += float(entropy.sum().detach().cpu())
    live_query_group_mask = group_mask[:, None, :] & query_mask[..., None]
    selected_logits = logits[live_query_group_mask]
    accumulator["live_query_group_pairs"] += int(live_query_group_mask.sum().detach().cpu())
    accumulator["logit_sum"] += float(selected_logits.sum().detach().cpu())


def _priority_effects(base_arrays: Mapping[str, Any], changed_arrays: Mapping[str, Any],
                      native_record: Mapping[str, Any]) -> dict[str, Any]:
    base_field = np.asarray(base_arrays["field_prediction"], dtype=np.float64)
    changed_field = np.asarray(changed_arrays["field_prediction"], dtype=np.float64)
    fluid_mask = (np.asarray(base_arrays["fluid_valid"], dtype=bool)
                  & np.isfinite(np.asarray(base_arrays["field_target"], dtype=np.float64)[:, 4]))
    base_surface = np.asarray(base_arrays["interface_prediction"], dtype=np.float64)[..., 0]
    changed_surface = np.asarray(changed_arrays["interface_prediction"], dtype=np.float64)[..., 0]
    surface_mask = (np.asarray(base_arrays["interface_valid"], dtype=bool)
                    & np.isfinite(np.asarray(base_arrays["interface_target"], dtype=np.float64)[..., 0]))
    base_material = np.asarray(base_arrays["material_prediction"], dtype=np.float64)
    changed_material = np.asarray(changed_arrays["material_prediction"], dtype=np.float64)
    material_mask = np.asarray(base_arrays["material_valid"], dtype=bool)
    base_peaks = np.where(material_mask, base_material, -np.inf).max(axis=-1)
    changed_peaks = np.where(material_mask, changed_material, -np.inf).max(axis=-1)
    peak_mask = (np.asarray(base_arrays["module_present"], dtype=bool) & material_mask.any(axis=-1)
                 & np.isfinite(base_peaks) & np.isfinite(changed_peaks))
    values = {
        "temperature_fluid": (changed_field[:, 4] - base_field[:, 4], fluid_mask),
        "temperature_surface": (changed_surface - base_surface, surface_mask),
        "temperature_material": (changed_material - base_material, material_mask),
        "module_peak_endpoint": (changed_peaks - base_peaks, peak_mask),
    }
    results = {}
    for name, (delta, mask) in values.items():
        selected = np.asarray(delta, dtype=np.float64)[mask]
        intervention_rms = float(np.sqrt(np.mean(selected * selected))) if selected.size else None
        native_rmse = native_record["fields"][name]["rmse"]
        ratio = (intervention_rms / float(native_rmse)
                 if intervention_rms is not None and native_rmse is not None and float(native_rmse) > 0 else None)
        results[name] = {"intervention_rms_native_units": intervention_rms,
                         "native_rmse_native_units": native_rmse,
                         "intervention_rms_over_native_rmse": ratio,
                         "native_units": THERMAL_FIELD_UNITS[name],
                         "valid_entries": int(selected.size),
                         "passes_0.1_fieldwise_threshold": bool(ratio is not None and ratio <= INTERVENTION_RMSE_FRACTION)}
    return results


def _reset_inference_auxiliary(model: torch.nn.Module) -> None:
    core = getattr(model, "core", model)
    reset = getattr(core, "reset_auxiliary", None)
    if callable(reset):
        reset()
    for name in ("_joint_entries", "_context_entries", "_compact_loss_entries"):
        value = getattr(core, name, None)
        if isinstance(value, list):
            value.clear()


def _capture_inference_auxiliary(model: torch.nn.Module) -> dict[str, Any]:
    core = getattr(model, "core", model)
    saved = {}
    for name in ("_last_compact_topology", "_joint_entries", "_context_entries", "_compact_loss_entries"):
        if hasattr(core, name):
            value = getattr(core, name)
            saved[name] = copy.deepcopy(value) if name == "_last_compact_topology" else list(value)
    return saved


def _restore_inference_auxiliary(model: torch.nn.Module, saved: Mapping[str, Any]) -> None:
    core = getattr(model, "core", model)
    _reset_inference_auxiliary(model)
    for name, value in saved.items():
        setattr(core, name, copy.deepcopy(value) if name == "_last_compact_topology" else list(value))


def _gradient_competition_snapshot(provider: Any) -> dict[str, Any]:
    """Expose the latest maintained additive-gradient telemetry without replay."""
    runtime = getattr(provider, "soft_organization", None)
    if runtime is None:
        return {"available": False, "reason": "provider has no soft organization runtime"}
    state = getattr(runtime, "state", {})
    last = getattr(runtime, "last_record", {})
    return {
        "available": bool(last),
        "source": "last completed training macro telemetry; no gradient recomputation",
        "primary_updates": int(state.get("primary_updates", 0)),
        "calibrated_coefficients": copy.deepcopy(state.get("coefficients", {})),
        "geometry_coefficient": state.get("geometry_coefficient"),
        "latest_macro_gradient_balance": copy.deepcopy(last) if last else None,
    }


def _predict(provider: Any, model: torch.nn.Module, batch: Any, epoch: int):
    scene = provider.make_scene(batch.scene_inputs)
    return provider.predict_native(model, scene, batch.receivers, model.architecture,
                                   "validation", epoch=int(epoch), temperature=1.0)


def _measure_full_shared_intervention(model: torch.nn.Module, provider: Any, core: Any,
                                      row: Mapping[str, Any], pending: Mapping[str, Any], epoch: int
                                      ) -> dict[str, Any]:
    batch = pending["batch"]
    left, right = map(int, row["slots"])
    with core.compact_soft_intervention_scope("full_shared", pair_slots=[(left, right)]):
        changed, _ = _predict(provider, model, batch, epoch)
    from .joint_native_evaluation import thermal_record_and_arrays
    changed_record, changed_arrays = thermal_record_and_arrays(changed, batch)
    effects = _priority_effects(pending["baseline_arrays"], changed_arrays, pending["native_record"])
    organization = pending["organization"]
    changed_org = changed.native_prepared.organization
    changed_compact = changed_org.compact_groups
    base_compact = organization.compact_groups
    topology_unchanged = bool(
        torch.equal(base_compact.pair_slots, changed_compact.pair_slots)
        and torch.equal(base_compact.pair_mask, changed_compact.pair_mask)
        and torch.equal(base_compact.pair_types, changed_compact.pair_types)
        and torch.equal(organization.joint_records.ids, changed_org.joint_records.ids)
    )
    pair_probabilities_unchanged = torch.equal(base_compact.pair_logits, changed_compact.pair_logits)
    controls_equal = torch.equal(changed_compact.controls[0, left], changed_compact.controls[0, right])
    changed_live_k = int(changed_compact.group_mask[0].sum().cpu())
    cardinality_preserved = bool(
        changed_live_k == pending["cardinality"]["present_native_N"]
        and changed_live_k == int(base_compact.group_mask[0].sum().cpu())
    )
    invariants = bool(topology_unchanged and pair_probabilities_unchanged and controls_equal
                      and cardinality_preserved)
    effects_pass = bool(all(effects[name]["passes_0.1_fieldwise_threshold"] for name in PRIORITY_FIELDS))
    intervention = {
        "mode": "two-slot full_shared descriptor intervention",
        "slots": [left, right],
        "same_native_batch_and_receivers": True,
        "actual_K_equals_N_after_intervention": cardinality_preserved,
        "topology_and_native_records_unchanged": topology_unchanged,
        "pair_probabilities_unchanged": bool(pair_probabilities_unchanged),
        "selected_pair_controls_equal": bool(controls_equal),
        "intervention_invariants_pass": invariants,
        "priority_field_effects": effects,
        "all_priority_fields_pass": bool(invariants and effects_pass),
        "physical_prediction_metrics_after_intervention": changed_record["fields"],
        "optimizer_updates": 0,
        "causal_claim": False,
        "compaction_claim": False,
        "speed_claim": False,
    }
    del changed, changed_arrays, changed_record
    return {"intervention": intervention, "invariants_pass": invariants,
            "all_priority_fields_pass": effects_pass}


def run_internal_review(model: torch.nn.Module, provider: Any, epoch: int,
                        output: str | Path, *, budget_seconds: float = 120.0) -> dict[str, Any]:
    """Run a bounded TRAIN organizer review at the four prespecified ages.

    The wall-clock guard is checked between native cases and pair interventions;
    a native forward already in flight is allowed to finish. The model, provider
    training state, RNG streams, and train/eval mode are restored afterward.
    """
    epoch = int(epoch)
    if epoch not in REVIEW_EPOCHS:
        return {"status": "not_scheduled", "epoch": epoch, "scheduled_epochs": list(REVIEW_EPOCHS)}
    if budget_seconds <= 0 or not math.isfinite(float(budget_seconds)):
        raise ValueError("Organizer review budget_seconds must be finite and positive.")
    config = getattr(provider, "_research_config", None)
    if isinstance(config, Mapping) and config.get("round12", {}).get("compact_arm") != "soft_only":
        raise ValueError("Thermal soft reviews are restricted to the soft_only compact arm.")
    root = Path(output).expanduser().resolve(strict=True)
    storage = _review_storage(root, estimated_bytes=INTERNAL_OUTPUT_BUDGET,
                              peak_temporary_bytes=2 * 1024**2,
                              config=config if isinstance(config, Mapping) else None)
    review_dir = root / "evaluations" / "thermal_soft_reviews" / "organizer"
    destination = review_dir / f"epoch_{epoch:04d}.json"
    weight_hash = _weights_sha256(model)
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous.get("epoch") == epoch and previous.get("model_weights_sha256") == weight_hash:
            return previous
        raise FileExistsError(f"Organizer review destination is immutable and has different weights: {destination}")

    panel = _select_train_panel(provider.training_cases, limit=4)
    input_counts = {str(case["case_id"]): _module_count(case) for case in provider.training_cases}
    selection = {"case_ids": list(panel),
                 "module_count_by_case": {case_id: input_counts[case_id] for case_id in panel},
                 "policy": "input-only min/next/median/max module-count strata, stable lexical ID tie-break",
                 "targets_or_model_errors_used": False,
                 "panel_sha256": _canonical_sha256({case_id: input_counts[case_id] for case_id in panel})}
    provider_identity = provider.identity_payload()
    core = model.core
    alpha = getattr(core, "compact_alpha", None)
    if getattr(core, "compact_mode", None) != "soft_only":
        raise RuntimeError("Organizer review model is not the soft_only compact core.")
    if not hasattr(core, "compact_soft_intervention_scope"):
        raise RuntimeError("Soft-only core lacks the agreed full_shared intervention scope.")

    from .unified_training import capture_rng_state, restore_rng_state
    from .research_training import _cpu_state_snapshot
    rng_snapshot = _cpu_state_snapshot(capture_rng_state())
    provider_state = _cpu_state_snapshot(provider.training_state_dict()) if hasattr(provider, "training_state_dict") else None
    state_snapshot = _state_snapshot(model)
    state_digest = _state_sha256(state_snapshot)
    inference_auxiliary = _capture_inference_auxiliary(model)
    gradient_snapshot = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                         for name, parameter in model.named_parameters()}
    prior_mode = model.training
    started = time.monotonic()
    deadline = started + float(budget_seconds)
    case_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    selected_count = {bank: 0 for bank in ("M", "E")}
    attempted_candidate_count = {bank: 0 for bank in ("M", "E")}
    all_pair_count = {bank: 0 for bank in ("M", "E")}
    nondegenerate_count = {bank: 0 for bank in ("M", "E")}
    failures: list[dict[str, Any]] = []
    records_by_id = {str(case["case_id"]): case for case in provider.training_cases}
    pending_cases: dict[str, dict[str, Any]] = {}
    generator = iter(provider.native_evaluation_batches(split="train", case_keys=panel, microbatch_size=1))
    model.eval()
    attempted_cases = 0
    status = "complete"
    try:
        with torch.no_grad():
            for case_id in panel:
                if time.monotonic() >= deadline:
                    status = "partial_budget"
                    break
                attempted_cases += 1
                try:
                    batch = next(generator)
                    if tuple(map(str, batch.case_keys)) != (case_id,):
                        raise ValueError("Native TRAIN panel loader changed the fixed organizer case order.")
                    baseline, _ = _predict(provider, model, batch, epoch)
                    native_record, baseline_arrays = __import__(
                        "honf_runtime.joint_native_evaluation", fromlist=["thermal_record_and_arrays"]
                    ).thermal_record_and_arrays(baseline, batch)
                    prepared = baseline.native_prepared
                    organization, query = prepared.organization, prepared.query_batch
                    case_candidates, cardinality = _candidate_rows(organization, case_id)
                    access = _query_access_summary(model, organization, query)
                    candidates.extend(case_candidates)
                    for bank in ("M", "E"):
                        bank_rows = [row for row in case_candidates if row["bank"] == bank]
                        all_pair_count[bank] += len(bank_rows)
                        nondegenerate_count[bank] += sum(bool(row["nondegenerate"]) for row in bank_rows)
                    case_rows.append({"case_id": case_id,
                                      "module_count": _module_count(records_by_id[case_id]),
                                      "native_physical_metrics": native_record["fields"],
                                      "K_N_audit": cardinality,
                                      "query_access": access,
                                      "eligible_candidates": len(case_candidates),
                                      "promising_candidates_by_bank": {
                                          bank: sum(row["bank"] == bank and row["promising"] for row in case_candidates)
                                          for bank in ("M", "E")},
                                      "interventions_measured": [],
                                      "intervention_selection_cap_per_bank_across_panel": 2,
                                      "data_partition": "TRAIN",
                                      "query_sampling_sha256": batch.auxiliary.get("query_sampling_sha256")})
                    pending_cases[case_id] = {"batch": batch, "baseline_arrays": baseline_arrays,
                                              "native_record": native_record, "organization": organization,
                                              "cardinality": cardinality,
                                              "case_row": case_rows[-1]}
                    del baseline
                except StopIteration:
                    status = "partial_failure"
                    failures.append({"case_id": case_id, "stage": "native_train_loader",
                                     "error": "native evaluation panel ended before the declared case"})
                    break
                except Exception as error:
                    failures.append({"case_id": case_id, "stage": "organizer_case",
                                     "error_type": type(error).__name__, "error": str(error)})
                    status = "partial_failure"
                    if time.monotonic() >= deadline:
                        break
            for row in candidates:
                row["intervention_status"] = "not_selected_review_quota"
                row["selected_for_intervention"] = False
            for row, selection_basis in _select_intervention_candidates(candidates, limit_per_bank=2):
                bank = str(row["bank"])
                case_id = str(row["case_id"])
                row["selected_for_intervention"] = True
                row["selection_mode"] = "promising_or_unqualified_probe"
                row["selection_basis"] = selection_basis
                attempted_candidate_count[bank] += 1
                pending = pending_cases.get(case_id)
                if pending is None:
                    row["intervention_status"] = "failed_missing_baseline"
                    row["promising_after_intervention"] = False
                    failures.append({"case_id": case_id, "stage": "candidate_intervention",
                                     "candidate": _stable_candidate_key(row),
                                     "error": "selected candidate has no completed baseline review record"})
                    status = "partial_failure"
                    continue
                if time.monotonic() >= deadline:
                    row["intervention_status"] = "not_run_budget"
                    status = "partial_budget"
                    continue
                try:
                    measured = _measure_full_shared_intervention(model, provider, core, row, pending, epoch)
                    row["intervention_status"] = "measured"
                    row["intervention"] = measured["intervention"]
                    row["promising_after_intervention"] = bool(
                        row["promising"] and measured["intervention"]["all_priority_fields_pass"])
                    pending["case_row"]["interventions_measured"].append(_stable_candidate_key(row))
                    selected_count[bank] += 1
                    if not measured["invariants_pass"]:
                        status = "partial_failure"
                        failures.append({"case_id": case_id, "stage": "intervention_invariant",
                            "candidate": _stable_candidate_key(row),
                            "error": "full_shared intervention changed candidate/record identity, K, probability, or paired controls"})
                except Exception as error:
                    row["intervention_status"] = "failed"
                    row["intervention_failure"] = {"error_type": type(error).__name__, "error": str(error)}
                    row["promising_after_intervention"] = False
                    failures.append({"case_id": case_id, "stage": "candidate_intervention",
                        "candidate": _stable_candidate_key(row),
                        "error_type": type(error).__name__, "error": str(error)})
                    status = "partial_failure"
    finally:
        current_state = _state_snapshot(model)
        if _state_sha256(current_state) != state_digest:
            model.load_state_dict(state_snapshot, strict=True)
        for name, parameter in model.named_parameters():
            saved_gradient = gradient_snapshot[name]
            parameter.grad = None if saved_gradient is None else saved_gradient.to(parameter.device)
        if provider_state is not None:
            current_provider_state = provider.training_state_dict()
            if not _same_value(current_provider_state, provider_state):
                provider.load_training_state_dict(provider_state)
        restore_rng_state(rng_snapshot)
        model.train(prior_mode)
        _restore_inference_auxiliary(model, inference_auxiliary)
        restored_state = _state_snapshot(model)
        if _state_sha256(restored_state) != state_digest:
            model.load_state_dict(state_snapshot, strict=True)

    previous_path = next((review_dir / f"epoch_{older:04d}.json" for older in reversed(REVIEW_EPOCHS)
                          if older < epoch and (review_dir / f"epoch_{older:04d}.json").is_file()), None)
    candidate_stability = _candidate_stability(candidates, previous_path, panel)
    banks = {}
    for bank in ("M", "E"):
        bank_rows = [row for row in candidates if row["bank"] == bank]
        measured_rows = [row for row in bank_rows if row.get("intervention_status") == "measured"]
        probabilities = np.asarray([row["probability"] for row in bank_rows], dtype=np.float64)
        discrepancies = np.asarray([row["normalized_rowdiff"] for row in bank_rows
                                    if row["normalized_rowdiff"] is not None], dtype=np.float64)
        banks[bank] = {
            "eligible_candidates": int(all_pair_count[bank]),
            "nondegenerate_candidates": int(nondegenerate_count[bank]),
            "promising_candidates": int(sum(row["promising"] for row in bank_rows)),
            "qualified_candidates_after_fieldwise_intervention": int(
                sum(bool(row.get("promising_after_intervention")) for row in bank_rows)),
            "probability_summary": ({"mean": float(probabilities.mean()), "median": float(np.median(probabilities)),
                                     "p10": float(np.quantile(probabilities, .10)),
                                     "p90": float(np.quantile(probabilities, .90))}
                                    if probabilities.size else None),
            "normalized_rowdiff_summary": ({"mean": float(discrepancies.mean()),
                                            "median": float(np.median(discrepancies)),
                                            "p90": float(np.quantile(discrepancies, .90))}
                                           if discrepancies.size else None),
            "interventions_measured": int(selected_count[bank]),
            "interventions_selected_for_review": int(attempted_candidate_count[bank]),
            "selected_promising_pairs": int(sum(
                row.get("selected_for_intervention") and row["promising"] for row in bank_rows)),
            "selected_unqualified_probe_pairs": int(sum(
                row.get("selection_basis") == "highest_probability_unqualified_probe" for row in bank_rows)),
            "selected_candidate_coverage_of_eligible": (
                float(attempted_candidate_count[bank] / all_pair_count[bank]) if all_pair_count[bank] else None),
            "measured_candidate_coverage_of_eligible": (
                float(selected_count[bank] / all_pair_count[bank]) if all_pair_count[bank] else None),
            "measured_promising_candidate_coverage": (
                float(sum(row.get("intervention_status") == "measured" and row["promising"]
                          for row in bank_rows)
                      / sum(bool(row["promising"]) for row in bank_rows))
                if any(row["promising"] for row in bank_rows) else None),
            "measured_intervention_fieldwise_pass_count": {
                name: sum(bool(row["intervention"]["priority_field_effects"][name]["passes_0.1_fieldwise_threshold"])
                          for row in measured_rows) for name in PRIORITY_FIELDS},
        }
    expected_cases = len(panel)
    completed_ids = [row["case_id"] for row in case_rows]
    if len(completed_ids) != expected_cases and status == "complete":
        status = "partial_failure"
    after_hash = _weights_sha256(model)
    after_state = _state_snapshot(model)
    weight_state_restored = after_hash == weight_hash and _state_sha256(after_state) == state_digest
    result = {
        "schema": "thermal_softonly_internal_organizer_review_v2",
        "status": status,
        "epoch": epoch,
        "model_weights_sha256": weight_hash,
        "weights_unchanged_after_review": weight_state_restored,
        "checkpoint_sha256": None,
        "checkpoint_note": "completed_epoch hook precedes checkpoint persistence; this is the live model-weight digest",
        "provider_identity_sha256": _canonical_sha256(provider_identity),
        "compact_alpha": None if alpha is None else float(alpha),
        "selection": selection,
        "panel_case_ids": list(panel),
        "panel_case_count": len(panel),
        "completed_case_ids": completed_ids,
        "completed_case_count": len(case_rows),
        "case_coverage": float(len(case_rows) / expected_cases) if expected_cases else None,
        "selection_is_input_only": True,
        "native_physical_fields": {
            "units": THERMAL_FIELD_UNITS,
            "metrics": "per-case native masked RMSE/MAE/bias/max and relative L2",
            "mask_policy": THERMAL_MASK_POLICY,
        },
        "K_N_audit": {
            "case_count": len(case_rows),
            "all_actual_K_equals_N": (all(row["K_N_audit"]["actual_K_equals_N"] for row in case_rows)
                                       if case_rows else None),
            "per_case": [{"case_id": row["case_id"], **row["K_N_audit"]} for row in case_rows],
            "N_definition": "present original native M plus E records, including zero-measure present E",
        },
        "query_access": {"per_case": [{"case_id": row["case_id"], **row["query_access"]} for row in case_rows],
                         "executor_savings_measured": False},
        "thresholds": {"pair_probability_minimum": P_THRESHOLD,
                       "normalized_C_rowdiff_maximum": NORMALIZED_C_THRESHOLD,
                       "intervention_fieldwise_rms_over_native_rmse_maximum": INTERVENTION_RMSE_FRACTION,
                       "nondegenerate_row_l1_mass_minimum": NONDEGENERATE_MASS,
                       "intervention_priority_fields": list(PRIORITY_FIELDS)},
        "candidate_coverage_by_bank": banks,
        "candidate_intervention_sampling": {
            "maximum_selected_pairs_per_bank_across_panel": 2,
            "strategy": "globally prioritize promising candidates, then probability descending, then stable case/slot IDs; fill remaining quota with highest-probability unqualified probes",
            "selection_mode": "promising_or_unqualified_probe",
            "selected_counts_by_bank": dict(attempted_candidate_count),
            "measured_counts_by_bank": dict(selected_count),
        },
        "candidate_stability": candidate_stability,
        "candidate_pair_count": len(candidates),
        "failure_count": len(failures),
        "training_gradient_competition_snapshot": _gradient_competition_snapshot(provider),
        "candidates": candidates,
        "failures": failures,
        "budget": {"hard_seconds": float(budget_seconds), "elapsed_seconds": time.monotonic() - started,
                   "guard_granularity": "between complete native cases and interventions; no in-flight forward is interrupted"},
        "storage_snapshot": storage,
        "optimizer_updates": 0,
        "causal_claim": False,
        "compaction_claim": False,
        "speed_claim": False,
        "executor_savings_measured": False,
    }
    if not weight_state_restored:
        raise RuntimeError("Organizer review changed model parameters despite the no-update contract.")
    _write_new_json(destination, result)
    result["output_path"] = str(destination)
    return result


def _read_jsonl(path: Path, checkpoint_sha256: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError(f"Native validation receipt {path}:{line_number} belongs to another checkpoint.")
            case_id = str(row["case_id"])
            if case_id in rows:
                raise ValueError(f"Native validation receipt contains duplicate case {case_id}.")
            rows[case_id] = row
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _canonical_val89_ids() -> tuple[str, ...]:
    import sys
    tools_dir = Path(__file__).resolve().parents[2] / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from compact_retrieval_evaluate import VAL89_IDS
    return tuple(map(str, VAL89_IDS))


@torch.no_grad()
def _native_checkpoint_validation(model: torch.nn.Module, provider: Any, *, selector: str,
                                 checkpoint_path: Path, checkpoint_sha256: str,
                                 epoch: int, output: Path, deadline: float,
                                 global_deadline: float) -> dict[str, Any]:
    from .joint_native_evaluation import reduce_thermal_records, thermal_record_and_arrays
    expected = _canonical_val89_ids()
    exposed = tuple(map(str, provider.validation_case_ids))
    if exposed != expected:
        return {"status": "unavailable_wrong_exposed_validation_identity",
                "selector": selector, "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256, "epoch": epoch,
                "expected_val89_ids": list(expected), "exposed_validation_case_ids": list(exposed),
                "locked_test_targets_read": False, "records": [], "failures": []}
    record_path = output / f"native_{selector}.jsonl"
    receipts = _read_jsonl(record_path, checkpoint_sha256)
    if any(case_id not in expected for case_id in receipts):
        raise ValueError("Native validation receipt includes a case outside the exact canonical VAL89 panel.")
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    model.eval()
    checkpoint_weights_sha256 = _weights_sha256(model)
    pending = [case_id for case_id in expected if case_id not in receipts]
    generator = iter(provider.native_evaluation_batches(split="validation", case_keys=pending, microbatch_size=1))
    for case_id in pending:
        if time.monotonic() >= min(deadline, global_deadline):
            break
        try:
            batch = next(generator)
            if tuple(map(str, batch.case_keys)) != (case_id,):
                raise ValueError("VAL89 native iterator changed the canonical case order.")
            prediction, _ = _predict(provider, model, batch, epoch)
            row, arrays = thermal_record_and_arrays(prediction, batch)
            row.update({"checkpoint_sha256": checkpoint_sha256,
                        "weights_sha256": checkpoint_weights_sha256,
                        "partition": "exposed canonical VAL89",
                        "response_validation_targets_read": False})
            _append_jsonl(record_path, row)
            receipts[case_id] = row
            del arrays, prediction, batch
        except StopIteration:
            failures.append({"stage": "native_validation_loader",
                             "error": "VAL89 native iterator ended before its fixed exposed panel"})
            break
        except Exception as error:
            failures.append({"case_id": case_id, "error_type": type(error).__name__, "error": str(error)})
            if time.monotonic() >= min(deadline, global_deadline):
                break
    ordered_rows = [receipts[case_id] for case_id in expected if case_id in receipts]
    complete = len(ordered_rows) == len(expected)
    status = "complete" if complete else ("partial_budget" if time.monotonic() >= min(deadline, global_deadline)
                                           else "partial_failure")
    result = {
        "selector": selector,
        "status": status,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "weights_sha256": ordered_rows[0]["weights_sha256"] if ordered_rows else checkpoint_weights_sha256,
        "checkpoint_epoch": int(epoch),
        "partition": "exposed canonical VAL89 only",
        "expected_case_count": len(expected),
        "exposed_validation_case_ids": list(expected),
        "exposed_validation_membership_sha256": _canonical_sha256(list(expected)),
        "completed_case_count": len(ordered_rows),
        "completed_case_ids": [row["case_id"] for row in ordered_rows],
        "missing_case_ids": [case_id for case_id in expected if case_id not in receipts],
        "metrics_complete": bool(complete),
        "metrics": reduce_thermal_records(ordered_rows) if ordered_rows else None,
        "field_units": THERMAL_FIELD_UNITS,
        "field_mask_policy": THERMAL_MASK_POLICY,
        "case_metrics": [{"case_id": row["case_id"], "module_count": row["module_count"],
                          "fields": row["fields"]} for row in ordered_rows],
        "failures": failures,
        "locked_test_targets_read": False,
        "unexposed_original_test_targets_read": False,
        "per_case_guard_elapsed_seconds": time.monotonic() - started,
        "budget_deadline_unix_monotonic": min(deadline, global_deadline),
        "receipt_jsonl": str(record_path),
        "scope_note": "full native receivers per exposed case; partial case metrics are not labeled a full VAL89 result",
    }
    return result


def _response_inventory(provider: Any, family_id: str, *, response_group: str,
                        base_inventory: Mapping[str, Any]) -> dict[str, Any]:
    atlas_directory = Path(provider.atlas_directory).expanduser().resolve(strict=True)
    path = atlas_directory / f"train_{family_id}_responses.npz"
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Retained Thermal response reference is missing for family {family_id}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_physical = "duplicate_family:0001+0273" if family_id == "0001" else f"stored_family:{family_id}"
    if (metadata.get("split") != "train" or metadata.get("anchor_id") != family_id
            or metadata.get("family_id") != expected_physical):
        raise ValueError(f"Thermal response external metadata {family_id} has changed physical identity.")
    with np.load(path, allow_pickle=False) as handle:
        embedded = json.loads(str(handle["family_metadata_json"].item()))
        if (embedded.get("source_dataset_split") != "train" or embedded.get("split") != "train"
                or embedded.get("anchor_id") != family_id
                or embedded.get("physical_family_id") != expected_physical):
            raise ValueError(f"Thermal response embedded NPZ metadata {family_id} has changed physical identity.")
        labels = list(map(str, handle["all_labels"].tolist()))
        required = {"baseline", "i_minus", "i_plus", "j_minus", "j_plus", "mm", "mp", "pm", "pp",
                    "heat_transfer_plus", "heat_transfer_minus"}
        if not required.issubset(set(labels)):
            raise ValueError(f"Thermal response reference {family_id} lacks the fixed response endpoints.")
        heating = np.asarray(handle["heating"], dtype=np.float64)
        module_count = int(handle["heating"].shape[1])
        if int(embedded.get("module_count", -1)) != module_count:
            raise ValueError(f"Thermal response reference {family_id} module count differs from embedded metadata.")
        role_contract = {}
        for role in ("fluid_fields", "interface", "solid_temperature"):
            role_contract[role] = {
                "channel_names": list(map(str, handle[role + "_channel_names"].tolist())),
                "channel_units": list(map(str, handle[role + "_channel_units"].tolist())),
                "reference_shape": list(handle[role + "_values"].shape),
                "valid_mask_shape": list(handle[role + "_valid_mask"].shape),
            }
    source = {
        "family_id": family_id,
        "path": str(path),
        "sha256": _file_sha256(path),
        "source_split": "original TRAIN",
        "external_metadata_identity": {key: metadata.get(key) for key in ("anchor_id", "family_id", "split")},
        "embedded_npz_metadata_identity": {key: embedded.get(key) for key in (
            "anchor_id", "physical_family_id", "source_dataset_split", "split", "module_count")},
        "all_labels": labels,
        "response_training_exposure": (
            "original TRAIN response supervision" if response_group == "train"
            else "held out from fit; separately identified response-development family"),
        "stored_heating_rank_at_absolute_tolerance_1e-6": int(np.linalg.matrix_rank(
            heating - heating[0], tol=1.0e-6)),
        "module_count": module_count,
        "reference_json_path": str(metadata_path),
        "reference_json_sha256": _file_sha256(metadata_path),
        "role_contract": role_contract,
    }
    inventory = {"schema": "round9_existing_physical_response_inventory_v1",
                 "thermal_families": [source],
                 "independent_heat_null": base_inventory["independent_heat_null"] if family_id == "0001" else None}
    return inventory


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _retained_physical_inventory() -> dict[str, Any]:
    path = (Path("/data2/wanglz/ModularDT/diagnostics/joint_native_round9_20261017T210035Z")
            / "receipts" / "physical_response_inventory.json")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if inventory.get("schema") != "round9_existing_physical_response_inventory_v1":
        raise ValueError("Retained original physical response inventory schema changed.")
    return inventory


def _response_family_review(model: torch.nn.Module, provider: Any, payload: Mapping[str, Any],
                            family_id: str, group: str, output: Path,
                            base_inventory: Mapping[str, Any]) -> dict[str, Any]:
    from .joint_native_physics import evaluate_saved_responses
    payload = _response_payload_for_softonly(payload)
    inventory = _response_inventory(provider, family_id, response_group=group,
                                    base_inventory=base_inventory)
    path = output / group / family_id
    result = evaluate_saved_responses(model, provider, payload, inventory, path, family_ids=[family_id])
    family = result["families"][0]
    mixed = [{"family_id": family_id, "quartet": item["quartet"],
              "roles": item["roles"], "fixed_route_roles": item["fixed_route_roles"],
              "validation_partition": "mixed geometry response labels; validation-only labels"}
             for item in family.get("geometry_geometry_interactions", [])]
    return {
        "family_id": family_id,
        "response_group": group,
        "physical_reference_partition": "original TRAIN",
        "response_training_exposure": inventory["thermal_families"][0]["response_training_exposure"],
        "reference_npz_path": inventory["thermal_families"][0]["path"],
        "reference_npz_sha256": inventory["thermal_families"][0]["sha256"],
        "reference_json_path": inventory["thermal_families"][0]["reference_json_path"],
        "reference_json_sha256": inventory["thermal_families"][0]["reference_json_sha256"],
        "response_role_contract": inventory["thermal_families"][0]["role_contract"],
        "absolute_states": family["absolute_states"],
        "finite_responses": family["finite_responses"],
        "mixed_geometry_validation": mixed,
        "arrays_path": str(path / family["arrays"]),
        "arrays_sha256": family["arrays_sha256"],
        "new_solver_attempts": 0,
        "learned_routing_is_physical_causality": False,
    }


def _response_payload_for_softonly(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the response-evaluator label without changing sealed identity."""
    import sys
    tools_dir = Path(__file__).resolve().parents[2] / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from compact_retrieval_evaluate import response_payload_for_arm
    return response_payload_for_arm(payload, "thermal_compact_softonly_full1000_scratch_v2")


def run_final_validation(run: str | Path, device: str | torch.device,
                         budget_seconds: float = 14400) -> dict[str, Any]:
    """Evaluate exact terminal and selected-best checkpoints after fitting.

    The total wall-clock budget is shared by two full exposed VAL89 boards and
    retained-response replays. Guards are checked between cases/families; an
    individual native case or saved response family already in progress is
    allowed to finish. Partial records retain explicit missing-case status.
    """
    if not math.isfinite(float(budget_seconds)) or budget_seconds <= 0:
        raise ValueError("Final validation budget_seconds must be finite and positive.")
    from . import research_training as research
    root = Path(run).expanduser().resolve(strict=True)
    manifest, _ = research._load_run_manifest(root)
    run_identity = research.read_json(research.RunLayout(root).read_path("experiment_identity.json"))
    run_identity_sha256 = _canonical_sha256(run_identity)
    config = copy.deepcopy(manifest["config"])
    if config.get("round12", {}).get("compact_arm") != "soft_only":
        raise ValueError("Final Thermal soft validation is restricted to the soft_only arm.")
    if int(config["training"].get("local_horizon", 0)) != 1000:
        raise ValueError("Final soft-only validation requires the prespecified 1,000-epoch horizon.")
    storage = research.verify_storage(config, output=root, estimated_bytes=FINAL_OUTPUT_BUDGET,
                                      peak_temporary_bytes=256 * 1024**2)
    destination = root / "evaluations" / "thermal_soft_final_validation"
    result_path = destination / "result.json"
    latest_payload, latest_path = research._read_run_checkpoint(root, "latest")
    if int(latest_payload.get("epoch", -1)) != 1000:
        raise ValueError("Terminal validation requires the exact completed local epoch-1000 latest checkpoint.")
    best_payload, best_path = research._read_run_checkpoint(root, "best_by_field_mse")
    selectors = (("latest", latest_payload, latest_path), ("best_by_field_mse", best_payload, best_path))
    for selector, payload, _path in selectors:
        if payload.get("experiment_identity") != latest_payload.get("experiment_identity"):
            raise ValueError(f"Checkpoint alias {selector} has another experiment identity.")
    checkpoint_hashes = {selector: _file_sha256(checkpoint_path)
                          for selector, _payload, checkpoint_path in selectors}
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        saved_hashes = {selector: existing.get("selectors", {}).get(selector, {}).get("checkpoint_sha256")
                        for selector in ("latest", "best_by_field_mse")}
        if (existing.get("run_identity_sha256") == run_identity_sha256
                and saved_hashes == checkpoint_hashes):
            return existing
        raise FileExistsError("Final validation result exists with different run or checkpoint-alias identity.")
    destination.mkdir(parents=True, exist_ok=True)
    budget_receipt_path = destination / "storage_budget.json"
    if budget_receipt_path.is_file():
        budget_receipt = json.loads(budget_receipt_path.read_text(encoding="utf-8"))
        if budget_receipt.get("run_identity_sha256") != run_identity_sha256:
            raise ValueError("Partial final-validation output belongs to another run identity.")
        destination_bytes_before = int(budget_receipt["initial_output_bytes"])
    else:
        destination_bytes_before = research._tree_bytes(destination)
        _write_new_json(budget_receipt_path, {
            "schema": "thermal_softonly_final_validation_storage_budget_v1",
            "run_identity_sha256": run_identity_sha256,
            "initial_output_bytes": destination_bytes_before,
            "hard_new_output_budget_bytes": FINAL_OUTPUT_BUDGET,
            "estimated_peak_temporary_bytes": 256 * 1024**2,
            "minimum_free_headroom_bytes": storage["minimum_free_headroom_bytes"],
            "free_bytes_before": storage["free_bytes_before"],
        })

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the final-validation training process.")
    if requested_device.type == "cuda" and (requested_device.index or 0) >= torch.cuda.device_count():
        raise ValueError("Requested final-validation CUDA ordinal is outside the visible device set.")
    deadline = time.monotonic() + float(budget_seconds)
    started = time.monotonic()
    selector_deadlines = {"latest": started + float(budget_seconds) * 0.35,
                          "best_by_field_mse": started + float(budget_seconds) * 0.85}
    output_by_selector: dict[str, Any] = {}
    mixed_results: dict[str, Any] = {}
    base_inventory = None
    response_families_processed: list[dict[str, str]] = []
    runtime_execution_options = research._execution_options_for_run(root, config)
    optimizer_contract = manifest.get("optimizer_schedule_contract")
    run_case_ids = manifest.get("run_case_ids")
    local_horizon = int(config["training"]["local_horizon"])
    diagnostic = bool(manifest.get("diagnostic_only"))
    response_quota_endpoints = {"latest": started + float(budget_seconds) * 0.50,
                                "best_by_field_mse": deadline}

    for selector, payload, checkpoint_path in selectors:
        if time.monotonic() >= deadline:
            output_by_selector[selector] = {"status": "not_started_budget", "selector": selector,
                                            "checkpoint_path": str(checkpoint_path),
                                            "checkpoint_sha256": checkpoint_hashes[selector],
                                            "checkpoint_epoch": int(payload["epoch"])}
            continue
        runtime_config = copy.deepcopy(config)
        runtime_config["training"]["effective_batch_size"] = int(
            manifest.get("effective_batch_size", config["training"]["effective_batch_size"]))
        runtime_config["training"]["microbatch_cases"] = int(
            manifest.get("microbatch_cases", config["training"]["microbatch_cases"]))
        model, provider = research._build_provider_for_resume(
            runtime_config, payload, optimizer_contract, run_case_ids, local_horizon, diagnostic,
            requested_device, runtime_execution_options=runtime_execution_options)
        provider._identity_microbatch_size = int(
            manifest.get("identity_microbatch_cases", config["training"]["microbatch_cases"]))
        checkpoint_sha = checkpoint_hashes[selector]
        selector_root = destination / selector
        selector_root.mkdir(parents=True, exist_ok=True)
        native_result = _native_checkpoint_validation(
            model, provider, selector=selector, checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha, epoch=int(payload["epoch"]), output=selector_root,
            deadline=selector_deadlines[selector], global_deadline=deadline)
        output_by_selector[selector] = {"selector_role": (
                "terminal completed epoch 1000" if selector == "latest" else "separately selected best_by_field_mse alias"),
            "declared_best_field_score": payload.get("best_field_score"),
            "native_val89": native_result,
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": int(payload["epoch"]), "weights_sha256": _weights_sha256(model),
            "provider_identity_sha256": _canonical_sha256(provider.identity_payload())}
        if base_inventory is None:
            base_inventory = _retained_physical_inventory()

        response_results = []
        response_status = "complete"
        group_specs = (("train", TRAIN_RESPONSE_IDS), ("development", DEVELOPMENT_RESPONSE_IDS))
        for group, family_ids in group_specs:
            for family_id in family_ids:
                if time.monotonic() >= deadline or time.monotonic() >= response_quota_endpoints[selector]:
                    response_status = "partial_budget"
                    break
                try:
                    source_path = Path(provider.atlas_directory) / f"train_{family_id}_responses.npz"
                    projected = int(source_path.stat().st_size * 8 + 8 * 1024**2)
                    already_written = research._tree_bytes(destination) - destination_bytes_before
                    if already_written + projected > FINAL_OUTPUT_BUDGET:
                        response_status = "partial_storage_budget"
                        response_results.append({"family_id": family_id, "response_group": group,
                            "status": "not_started_storage_budget", "projected_bytes": projected,
                            "new_output_bytes_before": already_written})
                        break
                    family_result = _response_family_review(
                        model, provider, payload, family_id, group, selector_root / "responses", base_inventory)
                    response_results.append(family_result)
                    response_families_processed.append({"selector": selector, "family_id": family_id,
                                                        "group": group})
                    mixed_results.setdefault(selector, []).extend(family_result["mixed_geometry_validation"])
                except Exception as error:
                    response_status = "partial_failure"
                    response_results.append({"family_id": family_id, "response_group": group,
                        "status": "failed", "error_type": type(error).__name__, "error": str(error)})
            if response_status in ("partial_storage_budget", "partial_budget"):
                break
        output_by_selector[selector]["retained_response_families"] = {
            "status": ("complete" if len([row for row in response_results if row.get("arrays_path")]) == 8
                       else response_status),
            "expected_train4_ids": list(TRAIN_RESPONSE_IDS),
            "expected_heldout_response_dev4_ids": list(DEVELOPMENT_RESPONSE_IDS),
            "families": response_results,
            "mixed_geometry_validation": mixed_results.get(selector, []),
            "mixed_labels_are_separate_from_unary_response_metrics": True,
            "new_solver_attempts": 0,
        }
        del model, provider

    result = {
        "schema": "thermal_softonly_terminal_and_best_validation_v2",
        "status": ("complete" if all(
            output_by_selector.get(name, {}).get("native_val89", {}).get("metrics_complete")
            and output_by_selector.get(name, {}).get("retained_response_families", {}).get("status") == "complete"
            for name in ("latest", "best_by_field_mse")) else "partial"),
        "run": str(root),
        "run_identity_sha256": run_identity_sha256,
        "arm": "soft_only",
        "local_horizon": 1000,
        "selectors": output_by_selector,
        "response_family_coverage": response_families_processed,
        "mixed_geometry_validation_separately_reported": True,
        "field_units": THERMAL_FIELD_UNITS,
        "budget": {"hard_seconds": float(budget_seconds), "elapsed_seconds": time.monotonic() - started,
                   "native_arm_deadline_fractions": {"latest": 0.35, "best_by_field_mse": 0.85},
                   "retained_response_deadline_fractions": {"latest": 0.50, "best_by_field_mse": 1.0},
                   "hard_new_output_budget_bytes": FINAL_OUTPUT_BUDGET,
                   "guard_granularity": "between full native cases and retained response families"},
        "storage_snapshot": storage,
        "new_output_bytes_written": 0,
        "locked_test_targets_read": False,
        "unexposed_original_test_targets_read": False,
        "validation_used_for_model_selection": False,
        "new_solver_attempts": 0,
        "causal_claim": False,
        "compaction_claim": False,
        "speed_claim": False,
        "executor_savings_measured": False,
        "output_path": str(result_path),
    }
    existing_new_bytes = max(0, research._tree_bytes(destination) - destination_bytes_before)
    for _ in range(8):
        summary_bytes = len(json.dumps(_jsonable(result), indent=2, sort_keys=True,
                                       allow_nan=False).encode("utf-8")) + 1
        total_new_bytes = existing_new_bytes + summary_bytes
        if result["new_output_bytes_written"] == total_new_bytes:
            break
        result["new_output_bytes_written"] = total_new_bytes
    if result["new_output_bytes_written"] > FINAL_OUTPUT_BUDGET:
        raise RuntimeError("Final validation output exceeds its hard DATA2 byte budget.")
    _write_new_json(result_path, result)
    return result


__all__ = [
    "run_internal_review", "run_final_validation", "_normalized_pair_discrepancy",
    "_candidate_is_promising", "_select_train_panel",
]
