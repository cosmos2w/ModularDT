"""Measure the Wind G cut family at one immutable selected checkpoint.

Only G actions enter the flat action table. Retained W-full and same-G full
predictions, plus root-union, count-matched P, geometry, rewire, fixed-feature,
selected P direct, and P full controls, are measured beside those actions and
kept outside selector training rows. Planner features use current inputs,
realized G permissions, and native geometry; reference targets are used only
for the reported metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "HONF_Proj"
for _path in (
    str(PROJECT / "src"),
    str(PROJECT / "Case_WindFarm" / "src"),
    str(Path(__file__).resolve().parent),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import evaluate_controlled_maturation_panel as panel
import run_active_packet_reuse as runner
import run_controlled_maturation as maturation
from honf_forward_core.interface_fields.action_aware_frontier import (
    describe_realized_plan,
    receiver_role_descriptors,
)
from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan
from honf_runtime.compat import load_trusted_checkpoint

from windfarm.workflows.maturation import WIND_ACTION_PATHS, available_frontier_for_paths

RUN_ID = "2112"
ACTION_ORDER = ("root", "two_packet", "four_packet", "full_access")
PRIMARY_CAPACITY = {"QE": 0.95, "MM": 0.90}
FIXED_QUERY_SEED = 2_112_291
REPEAT_QUERY_SEED = 2_112_929
Q512_ROLE_COUNTS = {
    "volume": 256, "hub_slab": 64, "downstream_envelope": 64,
    "near_turbine": 64, "background": 64,
}
Q2048_ROLE_COUNTS = {
    "volume": 1024, "hub_slab": 256, "downstream_envelope": 256,
    "near_turbine": 256, "background": 256,
}
Q5_FEATURE_ROLE_COUNTS = {role: 1 for role in runner.ROLE_NAMES}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_binding_fields(
    arm_prefix: str, record: Mapping[str, Any], record_sha256: str, source: str,
) -> dict[str, Any]:
    """Serialize the verified source without relabeling a durability pointer as a review."""
    if arm_prefix not in {"g", "p"}:
        raise ValueError(f"Unknown checkpoint-binding arm prefix: {arm_prefix}")
    if source not in {"append_only_scheduled_review", "current_durability_checkpoint"}:
        raise ValueError(f"Unsupported checkpoint-binding source: {source}")
    digest = str(record_sha256).lower()
    if not isinstance(record, Mapping) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Checkpoint binding needs a record and a SHA-256 digest")
    if source == "current_durability_checkpoint" and record.get("checkpoint_kind") != "durability_only":
        raise ValueError("Current durability binding must retain checkpoint_kind=durability_only")
    is_review = source == "append_only_scheduled_review"
    return {
        f"{arm_prefix}_checkpoint_binding_source": source,
        f"{arm_prefix}_checkpoint_binding_record": dict(record),
        f"{arm_prefix}_checkpoint_binding_record_sha256": digest,
        f"{arm_prefix}_checkpoint_review_record": dict(record) if is_review else None,
        f"{arm_prefix}_checkpoint_review_record_line_sha256": digest if is_review else None,
    }


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _role_vector(values: Mapping[str, float], roles: Sequence[str]) -> list[float]:
    if set(values) != set(roles):
        raise ValueError("Role metric keys differ from the declared Wind role order")
    result = [float(values[role]) for role in roles]
    if not all(math.isfinite(value) and value >= 0.0 for value in result):
        raise ValueError("Role metric contains a negative or nonfinite value")
    return result


def _role_delta_rmse(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    role_slices: Mapping[str, slice],
    normalizer: Any,
) -> dict[str, float]:
    physical_delta = runner._standardized_delta_to_mps(prediction - reference, normalizer)
    result: dict[str, float] = {}
    for role in runner.ROLE_NAMES:
        part = physical_delta[0, role_slices[role]]
        result[role] = float(part.square().mean().sqrt().detach().cpu())
    return result


def _canonical_packet_features(
    raw_packet_features: np.ndarray,
    module_masks: np.ndarray,
    environment_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Quotient exact duplicate typed support rows and average input features."""
    features = np.asarray(raw_packet_features, dtype=np.float32)
    mm = np.asarray(module_masks, dtype=np.uint8)
    qe = np.asarray(environment_masks, dtype=np.uint8)
    if features.ndim != 2 or mm.ndim != 2 or qe.ndim != 2:
        raise ValueError("Packet features and MM/QE masks must be rank two")
    if not (features.shape[0] == mm.shape[0] == qe.shape[0]):
        raise ValueError("Packet features and typed masks have different packet counts")
    grouped: dict[tuple[bytes, bytes], list[int]] = {}
    for index in range(features.shape[0]):
        if not bool(mm[index].any() or qe[index].any()):
            continue
        signature = (
            np.ascontiguousarray(mm[index]).tobytes(),
            np.ascontiguousarray(qe[index]).tobytes(),
        )
        grouped.setdefault(signature, []).append(index)
    if not grouped:
        raise ValueError("Realized action has no source-bearing packet")
    groups = list(grouped.values())
    packet_rows = np.stack([features[indexes].mean(axis=0) for indexes in groups]).astype(
        np.float32, copy=False
    )
    canonical_mm = np.stack([mm[indexes[0]] for indexes in groups]).astype(np.uint8, copy=False)
    canonical_qe = np.stack([qe[indexes[0]] for indexes in groups]).astype(np.uint8, copy=False)
    signatures = [hashlib.sha256(left + b"\0" + right).hexdigest() for left, right in grouped]
    return packet_rows, canonical_mm, canonical_qe, signatures


def _summarize_action_exposure(
    records: Sequence[Mapping[str, Any]],
    *,
    training_rows: Sequence[int],
    selected_update: int,
    archived_optimizer_attempt_keys: Sequence[str] = (),
    required_passes: int = 2,
) -> dict[str, dict[str, Any]]:
    """Audit complete primary passes and the realized path/K exposure within them."""
    expected = set(map(int, training_rows))
    archived = set(map(str, archived_optimizer_attempt_keys))
    if not expected:
        raise ValueError("Exposure audit requires the exact eligible training rows")
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {
        action: defaultdict(list) for action in ("root", "two_packet", "four_packet")
    }
    eligible_updates = Counter()
    invalid_realized_path_updates = Counter()
    invalid_realized_k_updates = Counter()
    invalid_realized_execution_updates = Counter()
    for record in records:
        if int(record.get("update_count", -1)) > int(selected_update):
            continue
        attempt_key = str(record.get("optimizer_attempt_key", ""))
        if not attempt_key:
            raise ValueError("G update record is missing its optimizer-attempt lineage key")
        if attempt_key in archived:
            continue
        action = str(record.get("requested_action", ""))
        if action not in grouped or record.get("full_access_replay", False):
            continue
        if str(record.get("phase", "")) != "action_family":
            continue
        if list(record.get("requested_cut_paths", [])) != list(WIND_ACTION_PATHS[action]):
            continue
        capacity = record.get("capacity_vector", {})
        if (
            not isinstance(capacity, Mapping)
            or float(capacity.get("MM", -1.0)) != PRIMARY_CAPACITY["MM"]
            or float(capacity.get("QE", -1.0)) != PRIMARY_CAPACITY["QE"]
        ):
            continue
        if "primary_pass" not in record:
            raise ValueError("G update record is missing its primary-pass identity")
        eligible_updates[action] += 1
        realized_paths = _valid_realized_cut_paths(
            requested_paths=WIND_ACTION_PATHS[action],
            realized_paths=record.get("realized_cut_paths"),
        )
        if realized_paths is None:
            invalid_realized_path_updates[action] += 1
        raw_realized_k = record.get("realized_nonredundant_k")
        realized_k = (
            int(raw_realized_k)
            if isinstance(raw_realized_k, int) and not isinstance(raw_realized_k, bool)
            and raw_realized_k >= 1 else None
        )
        if realized_k is None:
            invalid_realized_k_updates[action] += 1
        if realized_paths is None or realized_k is None:
            invalid_realized_execution_updates[action] += 1
            continue
        grouped[action][int(record["primary_pass"])].append({
            "row_id": int(record["row"]),
            "realized_cut_paths": list(realized_paths),
            "realized_nonredundant_k": realized_k,
        })

    result: dict[str, dict[str, Any]] = {}
    for action, passes in grouped.items():
        complete = []
        pass_audits: dict[str, Any] = {}
        row_path_exposure: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for pass_id, row_ids in sorted(passes.items()):
            observed_rows = [int(entry["row_id"]) for entry in row_ids]
            complete_pass = len(observed_rows) == len(expected) and set(observed_rows) == expected
            if complete_pass:
                complete.append(pass_id)
            path_families: dict[str, dict[str, Any]] = {}
            for entry in row_ids:
                path_key = json.dumps(entry["realized_cut_paths"], separators=(",", ":"))
                family = path_families.setdefault(path_key, {
                    "realized_cut_paths": entry["realized_cut_paths"],
                    "row_ids": [],
                    "realized_nonredundant_k_counts": Counter(),
                })
                family["row_ids"].append(int(entry["row_id"]))
                if entry["realized_nonredundant_k"] is not None:
                    family["realized_nonredundant_k_counts"][str(entry["realized_nonredundant_k"])] += 1
                if complete_pass:
                    row_path_exposure[str(entry["row_id"])][path_key].append({
                        "primary_pass": int(pass_id),
                        "realized_nonredundant_k": entry["realized_nonredundant_k"],
                    })
            for family in path_families.values():
                family["row_count"] = len(family["row_ids"])
                family["row_ids"] = sorted(family["row_ids"])
                family["realized_nonredundant_k_counts"] = dict(sorted(
                    family["realized_nonredundant_k_counts"].items(), key=lambda item: int(item[0])
                ))
            pass_audits[str(pass_id)] = {
                "eligible_row_count": len(observed_rows),
                "distinct_row_count": len(set(observed_rows)),
                "complete_primary_pass": bool(complete_pass),
                "expected_row_count": len(expected),
                "resolved_path_families": dict(sorted(path_families.items())),
            }
        row_path_records: dict[str, list[dict[str, Any]]] = {}
        for row_id, by_path in sorted(row_path_exposure.items(), key=lambda item: int(item[0])):
            row_path_records[row_id] = []
            for path_key, observations in sorted(by_path.items()):
                paths = json.loads(path_key)
                row_path_records[row_id].append({
                    "realized_cut_paths": paths,
                    "primary_pass_observations": sorted(
                        observations, key=lambda item: int(item["primary_pass"])
                    ),
                    "primary_pass_ids": [int(item["primary_pass"]) for item in observations],
                    "realized_nonredundant_k_counts": dict(sorted(Counter(
                        str(item["realized_nonredundant_k"])
                        for item in observations if item["realized_nonredundant_k"] is not None
                    ).items(), key=lambda item: int(item[0]))),
                })
        result[action] = {
            "completed_primary_action_passes": len(complete),
            "completed_primary_pass_ids": complete,
            "required_complete_passes": int(required_passes),
            "trained_action": len(complete) >= int(required_passes),
            "eligible_action_family_update_count": int(eligible_updates[action]),
            "invalid_realized_cut_path_update_count": int(invalid_realized_path_updates[action]),
            "invalid_realized_nonredundant_k_update_count": int(invalid_realized_k_updates[action]),
            "invalid_realized_execution_update_count": int(invalid_realized_execution_updates[action]),
            "training_row_count": len(expected),
            "primary_capacity": {"MM": PRIMARY_CAPACITY["MM"], "QE": PRIMARY_CAPACITY["QE"]},
            "phase": "action_family",
            "scaffold_passes_excluded": True,
            "scheduled_full_access_replays_excluded": True,
            "exposure_unit": "resolved_path_family_and_measured_nonredundant_K_within_a_complete_primary_action_pass",
            "realized_cut_paths_by_primary_pass": pass_audits,
            "resolved_path_exposure_by_row": row_path_records,
        }
    return result


def _valid_realized_cut_paths(
    *, requested_paths: Sequence[str], realized_paths: Any,
) -> tuple[str, ...] | None:
    """Validate that ledger paths are a complete cut produced by ancestor collapse."""
    if not isinstance(realized_paths, (list, tuple)) or not realized_paths:
        return None
    paths = tuple(str(path) for path in realized_paths)
    if paths != tuple(sorted(set(paths))) or any(
        len(path) > 3 or set(path) - {"L", "R"} for path in paths
    ):
        return None
    if any(
        left != right and (left.startswith(right) or right.startswith(left))
        for index, left in enumerate(paths)
        for right in paths[index + 1:]
    ):
        return None
    kraft_sum = sum(2.0 ** (-len(path)) for path in paths)
    if not math.isclose(kraft_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return None
    requested = tuple(str(path) for path in requested_paths)
    if any(not any(requested_path.startswith(path) for path in paths) for requested_path in requested):
        return None
    if any(not any(requested_path.startswith(path) for requested_path in requested) for path in paths):
        return None
    return paths


def _trained_sparse_action(
    action: str, *, exact_work: float, full_work: float,
    exposure_record: Mapping[str, Any], row_id: int | None = None,
    realized_cut_paths: Sequence[str] = (), nonredundant_k: int | None = None,
) -> bool:
    sparse_success = float(exact_work) < float(full_work) - max(
        1.0e-9, 1.0e-12 * float(full_work)
    )
    required = int(exposure_record.get("required_complete_passes", 2))
    complete_pass_ids = set(map(int, exposure_record.get("completed_primary_pass_ids", [])))
    row_families = exposure_record.get("resolved_path_exposure_by_row", {})
    row_pass_ids: set[int] = set()
    if row_id is not None and isinstance(row_families, Mapping):
        for family in row_families.get(str(int(row_id)), []):
            for item in family.get("primary_pass_observations", []):
                pass_id = int(item["primary_pass"])
                if pass_id in complete_pass_ids:
                    row_pass_ids.add(pass_id)
    valid_current_execution = (
        action in WIND_ACTION_PATHS
        and _valid_realized_cut_paths(
            requested_paths=WIND_ACTION_PATHS[action], realized_paths=list(realized_cut_paths)
        ) is not None
        and nonredundant_k is not None
        and int(nonredundant_k) >= 1
    )
    return bool(
        action != "full_access"
        and exposure_record.get("trained_action", False)
        and len(complete_pass_ids) >= required
        and len(row_pass_ids & complete_pass_ids) >= required
        and valid_current_execution
        and sparse_success
    )


def _row_action_exposure(
    exposure_record: Mapping[str, Any], *, action: str, row_id: int,
    realized_cut_paths: Sequence[str], nonredundant_k: int | None,
) -> dict[str, Any]:
    """Keep the full exposure audit once at table level and bind only this row here."""
    same_path_passes: set[int] = set()
    same_path_k_passes: set[int] = set()
    for family in exposure_record.get("resolved_path_exposure_by_row", {}).get(str(int(row_id)), []):
        if list(family.get("realized_cut_paths", [])) != list(realized_cut_paths):
            continue
        for observation in family.get("primary_pass_observations", []):
            same_path_passes.add(int(observation["primary_pass"]))
            if observation.get("realized_nonredundant_k") == nonredundant_k:
                same_path_k_passes.add(int(observation["primary_pass"]))
    return {
        "action_key": action,
        "required_complete_passes": int(exposure_record.get("required_complete_passes", 2)),
        "completed_primary_action_passes": int(exposure_record.get("completed_primary_action_passes", 0)),
        "completed_primary_pass_ids": list(exposure_record.get("completed_primary_pass_ids", [])),
        "trained_action": bool(exposure_record.get("trained_action", False)),
        "training_row_count": int(exposure_record.get("training_row_count", 0)),
        "invalid_realized_cut_path_update_count": int(
            exposure_record.get("invalid_realized_cut_path_update_count", 0)
        ),
        "invalid_realized_nonredundant_k_update_count": int(
            exposure_record.get("invalid_realized_nonredundant_k_update_count", 0)
        ),
        "invalid_realized_execution_update_count": int(
            exposure_record.get("invalid_realized_execution_update_count", 0)
        ),
        "resolved_path_exposure_for_case": list(
            exposure_record.get("resolved_path_exposure_by_row", {}).get(str(int(row_id)), [])
        ),
        "same_realized_path_primary_pass_ids": sorted(same_path_passes),
        "same_realized_path_and_k_primary_pass_ids": sorted(same_path_k_passes),
    }


def _action_budget_vector(action: str) -> np.ndarray:
    if action not in ACTION_ORDER:
        raise ValueError(f"Unknown G action budget {action!r}")
    values = {"MM": 1.0, "QE": 1.0} if action == "full_access" else PRIMARY_CAPACITY
    return np.asarray([values["MM"], values["QE"]], dtype=np.float32)


def _family_role_medians(
    records: Sequence[Mapping[str, Any]],
    *,
    role_field: str,
    roles: Sequence[str],
) -> np.ndarray:
    by_family: dict[str, list[list[float]]] = defaultdict(list)
    for record in records:
        by_family[str(record["family_key"])].append(_role_vector(record[role_field], roles))
    if not by_family:
        raise ValueError(f"No family records available for {role_field}")
    return np.stack([
        np.median(np.asarray(values, dtype=np.float64), axis=0)
        for _family, values in sorted(by_family.items())
    ])


def _numerical_floor_role_mps(
    wfull_records: Sequence[Mapping[str, Any]], *, roles: Sequence[str]
) -> list[float]:
    family_medians = _family_role_medians(
        wfull_records, role_field="wfull_role_rmse_mps", roles=roles
    )
    train_family_median = np.median(family_medians, axis=0)
    return np.maximum(1.0e-8, 1.0e-6 * train_family_median).tolist()


def _sampling_allowance_role_mps(
    q512_vs_q2048: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    roles: Sequence[str],
) -> tuple[list[float], dict[str, Any]]:
    """Use family medians then a 95th percentile of same-G-full query-size deltas."""
    if len(q512_vs_q2048) < 2:
        raise ValueError("Q512/Q2048 allowance needs at least two train-fit layout families")
    family_medians = []
    details = {}
    for family, values in sorted(q512_vs_q2048.items()):
        if set(values) != set(roles):
            raise ValueError("Q512/Q2048 calibration roles differ from the declared order")
        role_medians = []
        for role in roles:
            deltas = np.asarray(values[role], dtype=np.float64)
            if deltas.size == 0 or not np.isfinite(deltas).all() or bool((deltas < 0).any()):
                raise ValueError("Q512/Q2048 calibration deltas must be finite and nonnegative")
            role_medians.append(float(np.median(deltas)))
        family_medians.append(role_medians)
        details[str(family)] = {
            role: role_medians[index] for index, role in enumerate(roles)
        }
    matrix = np.asarray(family_medians, dtype=np.float64)
    allowance = np.quantile(matrix, 0.95, axis=0)
    return allowance.tolist(), {
        "method": "per-layout median absolute same-G-full RMSE delta, then 95th percentile across train_fit layouts",
        "family_median_abs_delta_role_mps": details,
        "absolute_allowance_role_mps": allowance.tolist(),
        "train_fit_family_count": len(family_medians),
        "q512_role_query_counts": dict(Q512_ROLE_COUNTS),
        "q2048_role_query_counts": dict(Q2048_ROLE_COUNTS),
    }


def _validate_action_rows(rows: Sequence[Mapping[str, Any]], *, expected_sha: str) -> None:
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    allowed = set(ACTION_ORDER)
    for row in rows:
        action = str(row.get("action_key", ""))
        if action not in allowed:
            raise ValueError(f"Non-G action {action!r} was placed in the selector table")
        if row.get("forward_checkpoint_sha256") != expected_sha:
            raise ValueError("An action row is not bound to the selected G checkpoint")
        if str(row.get("family_key")) != str(row.get("layout_key")):
            raise ValueError("Physical family key must identify the Wind layout")
        if action == "full_access":
            if row.get("nonredundant_k") is not None or row.get("full_access") is not True:
                raise ValueError("Explicit full action must use null K and full_access=true")
        elif row.get("nonredundant_k") is None or int(row["nonredundant_k"]) < 1:
            raise ValueError("Every sparse G action needs measured nonredundant K")
        for field in ("exact_work", "full_work"):
            if not math.isfinite(float(row[field])):
                raise ValueError("Action work must be finite raw canonical work")
        if (
            float(row["full_work"]) <= 0
            or not 0 <= float(row["exact_work"]) <= float(row["full_work"]) * (1 + 1e-6)
        ):
            raise ValueError("Action work lies outside its same-case canonical full-access bound")
        by_case[str(row["case_key"])].append(row)
    if not by_case:
        raise ValueError("Measured action table contains no cases")
    for case_key, case_rows in by_case.items():
        actions = [str(row["action_key"]) for row in case_rows]
        if len(actions) != len(set(actions)) or set(actions) != allowed:
            raise ValueError(f"Case {case_key!r} does not have exactly four G actions")
        identities = {
            (row["split"], row["family_key"], row["query_panel"], row["row_id"])
            for row in case_rows
        }
        if len(identities) != 1:
            raise ValueError(f"Case {case_key!r} changes identity across G actions")


def _split_selected_layouts(layouts: Sequence[int], *, seed: int) -> dict[int, str]:
    unique = sorted(set(map(int, layouts)))
    if len(unique) < 6:
        raise ValueError("Selected action table needs at least six distinct training layouts")
    rng = np.random.default_rng(int(seed))
    shuffled = [int(value) for value in rng.permutation(unique)]
    train_count = max(2, round(0.5 * len(shuffled)))
    dev_count = max(2, round(0.25 * len(shuffled)))
    if train_count + dev_count >= len(shuffled):
        dev_count = len(shuffled) - train_count - 1
    if train_count < 2 or dev_count < 1 or len(shuffled) - train_count - dev_count < 1:
        raise ValueError("Unable to form disjoint train_fit/dev/held layout groups")
    return {
        **{layout: "train_fit" for layout in shuffled[:train_count]},
        **{layout: "dev" for layout in shuffled[train_count:train_count + dev_count]},
        **{layout: "held_family_audit" for layout in shuffled[train_count + dev_count:]},
    }


def _effective_action_features(
    *, scores: Any, plan: Any, encoded: Any, cut: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    raw = describe_realized_plan(scores, plan, encoded, cut, case_index=0)
    raw_features = raw.detach().cpu().numpy().astype(np.float32, copy=False)
    source_masks = panel._packet_source_masks(plan, encoded, cut)
    packet_rows, mm, qe, signatures = _canonical_packet_features(
        raw_features, source_masks["MM"], source_masks["QE"]
    )
    measured_k = panel._action_union_k(plan, encoded, cut)
    if packet_rows.shape[0] != measured_k:
        raise RuntimeError("Exact action quotient differs from the shared realized-plan K count")
    return packet_rows, mm, qe, signatures


def _metric_row(
    *,
    prediction: torch.Tensor,
    full_prediction: torch.Tensor,
    batch: Any,
    sample: Any,
    normalizer: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    _loss, _mse, role_rmse = runner._role_objective(
        prediction, batch.target_field, sample.role_slices, normalizer,
        {role: 1.0 for role in runner.ROLE_NAMES},
    )
    distortion = _role_delta_rmse(prediction, full_prediction, sample.role_slices, normalizer)
    return role_rmse, distortion


def _nearest_geometry_membership(
    packet_centers: np.ndarray,
    source_coordinates: np.ndarray,
    source_validity: np.ndarray,
    packet_sizes: np.ndarray,
    coordinate_scale: np.ndarray,
) -> np.ndarray:
    """Choose each packet's nearest valid sources while preserving its size."""
    centers = np.asarray(packet_centers, dtype=np.float64)
    sources = np.asarray(source_coordinates, dtype=np.float64)
    valid = np.asarray(source_validity, dtype=bool)
    sizes = np.asarray(packet_sizes, dtype=np.int64)
    scale = np.asarray(coordinate_scale, dtype=np.float64)
    if centers.ndim != 2 or sources.ndim != 2 or centers.shape[1] != sources.shape[1]:
        raise ValueError("Packet centers and source coordinates must share [count,dimension]")
    if valid.shape != (sources.shape[0],) or sizes.shape != (centers.shape[0],):
        raise ValueError("Geometry support validity and packet sizes do not align")
    if scale.shape != (centers.shape[1],) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("Geometry support scale must be finite and positive")
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(sources)):
        raise ValueError("Geometry support coordinates must be finite")
    if np.any(sizes < 0) or np.any(sizes > int(valid.sum())):
        raise ValueError("Geometry packet size exceeds the valid source population")
    membership = np.zeros((centers.shape[0], sources.shape[0]), dtype=np.uint8)
    eligible = np.flatnonzero(valid)
    scaled_sources = sources[eligible] / scale[None, :]
    for row, (center, size) in enumerate(zip(centers, sizes, strict=True)):
        count = int(size)
        if count == 0:
            continue
        squared_distance = ((scaled_sources - center[None, :] / scale[None, :]) ** 2).sum(axis=1)
        # Source index is the deterministic tie break after geometric distance.
        order = np.lexsort((eligible, squared_distance))
        membership[row, eligible[order[:count]]] = 1
    return membership


def _geometry_action_masks(
    *, plan: Any, encoded: Any, tree: Any, cut: tuple[int, ...]
) -> dict[str, np.ndarray]:
    indices = torch.as_tensor(cut, device=plan.split_gates.device, dtype=torch.long)
    universe = tree.universe
    coordinates = universe.coordinates.detach().cpu().numpy()
    weights = universe.weights.detach().cpu().numpy()
    centers = np.stack([
        (coordinates[list(tree.nodes[node].anchor_indices)]
         * weights[list(tree.nodes[node].anchor_indices), None]).sum(axis=0)
        / weights[list(tree.nodes[node].anchor_indices)].sum()
        for node in cut
    ])
    scale = universe.coordinate_scale.detach().cpu().numpy()
    route_data = {
        "MM": (encoded.module_centers[0], encoded.module_present[0] > 0.5),
        "QE": (encoded.env_coords[0], encoded.env_weights[0] > 0.0),
    }
    result: dict[str, np.ndarray] = {}
    for route, (source_coordinates, validity) in route_data.items():
        valid = validity.detach().cpu().numpy().astype(bool, copy=False)
        realized = (plan.permission_matrix(route)[indices] > 0.0).detach().cpu().numpy()
        realized &= valid[None, :]
        result[route] = _nearest_geometry_membership(
            centers,
            source_coordinates.detach().cpu().numpy(),
            valid,
            realized.sum(axis=1),
            scale,
        )
    return result


def _degree_size_preserving_rewire(
    membership: np.ndarray, *, seed: int, max_swaps: int = 31
) -> tuple[np.ndarray, int, int]:
    """Apply valid bipartite 2-switches and report actual changed links."""
    values = np.asarray(membership, dtype=np.uint8)
    if values.ndim != 2 or not np.isin(values, (0, 1)).all():
        raise ValueError("Packet membership must be a binary packet/source matrix")
    values = values.copy()
    edges = np.argwhere(values > 0)
    if len(edges) < 2 or max_swaps < 1:
        return values, 0, 0
    original_rows = (values > 0).sum(axis=1)
    original_columns = (values > 0).sum(axis=0)
    rng = np.random.default_rng(int(seed))
    swaps = 0
    target_swaps = min(int(max_swaps), max(1, len(edges) // 4))
    for _ in range(min(10 * len(edges), 2_000)):
        first, second = rng.choice(len(edges), size=2, replace=False)
        row_a, source_a = map(int, edges[first])
        row_b, source_b = map(int, edges[second])
        if row_a == row_b or source_a == source_b:
            continue
        if values[row_a, source_b] or values[row_b, source_a]:
            continue
        values[row_a, source_a] = values[row_b, source_b] = 0
        values[row_a, source_b] = values[row_b, source_a] = 1
        edges[first] = (row_a, source_b)
        edges[second] = (row_b, source_a)
        swaps += 1
        if swaps >= target_swaps:
            break
    if not np.array_equal((values > 0).sum(axis=1), original_rows) or not np.array_equal(
        (values > 0).sum(axis=0), original_columns
    ):
        raise AssertionError("Degree-preserving rewire changed packet size or source degree")
    changed_links = int(np.count_nonzero(values != np.asarray(membership, dtype=np.uint8)))
    return values, swaps, changed_links


def _plan_with_packet_masks(
    plan: Any, *, cut: tuple[int, ...], masks: Mapping[str, np.ndarray]
) -> Any:
    indices = torch.as_tensor(cut, device=plan.split_gates.device, dtype=torch.long)
    replacement = plan
    for route in ("MM", "QE"):
        current = plan.permission_matrix(route)
        values = np.asarray(masks[route], dtype=np.uint8)
        if values.shape != (len(cut), int(current.shape[1])):
            raise ValueError(f"{route} support does not match the selected packet/source axes")
        membership = torch.zeros_like(current)
        membership[indices] = torch.as_tensor(values, device=current.device, dtype=current.dtype)
        replacement = replacement.with_permission(route, membership)
    return replacement


def _top_count_pair_mask(
    scores: torch.Tensor, eligible_pairs: torch.Tensor, target_count: int
) -> torch.Tensor:
    """Select exactly N scored pairs with a stable row-major tie break."""
    if scores.ndim != 2 or eligible_pairs.shape != scores.shape or eligible_pairs.dtype != torch.bool:
        raise ValueError("Pair scores and eligibility must share a boolean [receiver,source] shape")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Pair scores must be finite before count matching")
    if isinstance(target_count, bool) or int(target_count) != target_count or int(target_count) < 0:
        raise ValueError("Count-matched target must be a nonnegative integer")
    flat_indices = torch.nonzero(eligible_pairs.reshape(-1), as_tuple=False).reshape(-1)
    count = int(target_count)
    if count > int(flat_indices.numel()):
        raise ValueError("Count-matched target exceeds the eligible live-pair population")
    selected = torch.zeros_like(eligible_pairs)
    if count:
        eligible_scores = scores.reshape(-1).index_select(0, flat_indices)
        order = torch.argsort(eligible_scores, descending=True, stable=True)
        chosen = flat_indices.index_select(0, order[:count])
        selected.reshape(-1)[chosen] = True
    return selected


def _count_matched_direct_plan(
    *, encoded: Any, tree: Any, batch: Any, scorer: Any,
    target_pairs_by_route: Mapping[str, int],
) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    """Build a query-bound direct P replay at exact G live pair counts."""
    tables = runner._direct_feature_tables(encoded)
    plan = MechanismPlan.full_access(
        tree, encoded.module_present[0], int(encoded.env_coords.shape[1])
    )
    masks: dict[str, np.ndarray] = {}
    route_records: dict[str, Any] = {}
    for route in ("MM", "QE"):
        catalog = runner.canonical_pair_catalog(encoded, tree, route)
        source_features, source_coordinates, source_validity = runner._direct_source_for_route(
            route, tables
        )
        if route == "MM":
            receivers = catalog.receiver_coordinates
            receiver_validity = catalog.receiver_validity
            eligible = catalog.pair_validity
        else:
            receivers = batch.query_xy[0]
            receiver_validity = torch.ones(
                receivers.shape[0], device=receivers.device, dtype=torch.bool
            )
            eligible = receiver_validity[:, None] & source_validity[None, :]
        receiver_features = runner._receiver_features_for_route(route, receivers, tables)
        scores = scorer(
            receiver_features,
            source_features,
            receivers,
            source_coordinates,
            mechanism=route,
            budget_fraction=PRIMARY_CAPACITY[route],
        )
        desired = int(target_pairs_by_route[route])
        selected = _top_count_pair_mask(scores, eligible, desired)
        if int(selected.sum().detach().cpu()) != desired:
            raise RuntimeError(f"{route} direct reprojection did not preserve the requested pair count")
        plan = plan.with_direct_pair_access(
            route,
            receivers,
            selected.to(dtype=scores.dtype),
            receiver_validity=receiver_validity,
        )
        masks[route] = selected.detach().cpu().numpy().astype(np.uint8, copy=False)
        route_records[route] = {
            "target_pair_count_from_g_four_packet": desired,
            "actual_selected_pair_count": desired,
            "eligible_live_pair_count": int(eligible.sum().detach().cpu()),
            "pair_count_basis": (
                "native module receiver/source panel" if route == "MM"
                else "sampled live query panel"
            ),
            "selected_mask_shape": list(masks[route].shape),
            "stable_score_tie_break": "descending score, then row-major receiver/source index",
            "selected_mask_sha256": _array_sha256(masks[route]),
        }
    return plan, masks, route_records


def _training_population_feature_means(
    records: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Compute equal-case means from input-only train_fit organizer tokens."""
    if not records:
        raise ValueError("Fixed training-population organizer inputs need at least one train_fit case")
    module_case_means = []
    environment_case_means = []
    global_case_means = []
    reference_device = records[0]["module_states"].device
    reference_dtype = records[0]["module_states"].dtype
    for record in records:
        modules = record["module_states"]
        module_valid = record["module_valid"].to(device=modules.device, dtype=torch.bool)
        environments = record["environment_states"]
        weights = record["environment_weights"].to(device=environments.device, dtype=environments.dtype)
        global_state = record["global_state"]
        if modules.ndim != 2 or module_valid.shape != (modules.shape[0],):
            raise ValueError("Training-population module tokens and validity do not align")
        if environments.ndim != 2 or weights.shape != (environments.shape[0],):
            raise ValueError("Training-population environment tokens and measure do not align")
        if global_state.numel() < 1 or modules.shape[1] != environments.shape[1]:
            raise ValueError("Training-population organizer token widths do not agree")
        if not bool(module_valid.any()) or not bool((weights > 0).any()):
            raise ValueError("Training-population case has no valid typed source tokens")
        module_case_means.append(modules[module_valid].mean(dim=0))
        positive = weights > 0
        normalized_weights = weights[positive] / weights[positive].sum()
        environment_case_means.append(
            (environments[positive] * normalized_weights[:, None]).sum(dim=0)
        )
        global_case_means.append(global_state.reshape(-1, modules.shape[1]).mean(dim=0))
    means = {
        "module_state": torch.stack(module_case_means).mean(dim=0),
        "environment_state": torch.stack(environment_case_means).mean(dim=0),
        "global_state": torch.stack(global_case_means).mean(dim=0),
    }
    if any(value.device != reference_device or value.dtype != reference_dtype for value in means.values()):
        means = {key: value.to(device=reference_device, dtype=reference_dtype) for key, value in means.items()}
    if any(not bool(torch.isfinite(value).all()) for value in means.values()):
        raise ValueError("Training-population organizer means must be finite")
    return means


def _fixed_population_context(encoded: Any, means: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Broadcast typed population means over the current case's source rows."""
    states = {}
    for name, key, target in (
        ("module_states", "module_state", encoded.module_tokens),
        ("environment_states", "environment_state", encoded.env_tokens),
        ("global_state", "global_state", encoded.global_token),
    ):
        mean = means[key].to(device=target.device, dtype=target.dtype).reshape(
            (1,) + (1,) * max(0, target.ndim - 2) + (-1,)
        )
        states[name] = mean.expand_as(target)
    return states


def _case_record(
    *,
    split: str,
    row_id: int,
    query_panel: str,
    query_seed: int,
    query_counts: Mapping[str, int],
    view: Any,
    sampler: Any,
    normalizer: Any,
    role_scales: Mapping[str, float],
    source_model: Any,
    g_model: Any,
    g_control: Any,
    p_model: Any,
    p_control: Any,
    device: torch.device,
    feature_arrays: dict[str, np.ndarray],
    mask_arrays: dict[str, np.ndarray],
    exposure: Mapping[str, Mapping[str, Any]],
    training_population_features: Mapping[str, torch.Tensor] | None = None,
    training_population_feature_sha256: str | None = None,
    prepared_inputs: tuple[Any, Any, Any] | None = None,
    query_seed_retry_count: int = 0,
    fixed_query_sample_index_sha256: str | None = None,
    query_repeat_overlap_count: int | None = None,
    query_repeat_candidate_attempt_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if prepared_inputs is None:
        case, sample, batch = maturation._case_batch(
            view=view, row_id=int(row_id), query_seed=int(query_seed), role_counts=query_counts,
            sampler=sampler, normalizer=normalizer, device=device,
        )
    else:
        case, sample, batch = prepared_inputs
    layout_id = int(view.metadata["layout_index"][int(row_id)])
    layout_key = str(layout_id)
    case_key = f"layout{layout_id:03d}_row{int(row_id):03d}_{query_panel}"
    query_sample_index_sha256 = hashlib.sha256(
        np.ascontiguousarray(sample.flat_indices, dtype=np.int64).tobytes()
    ).hexdigest()
    roles = tuple(runner.ROLE_NAMES)

    with torch.inference_mode():
        source_model.core.backend.set_cover_mode("external")
        source_encoded = source_model.core.encode_case(batch)
        source_tree = source_model.core.backend.build_case_trees(source_encoded)[0]
        source_full = MechanismPlan.full_access(
            source_tree, source_encoded.module_present[0], int(source_encoded.env_coords.shape[1])
        )
        wfull_prediction, _wfull_aux = runner._prediction_with_plan(
            source_model, source_encoded, batch, source_full
        )
        _wloss, _wmse, wfull_rmse = runner._role_objective(
            wfull_prediction, batch.target_field, sample.role_slices, normalizer, role_scales
        )

        g_model.core.backend.set_cover_mode("external")
        encoded = g_model.core.encode_case(batch)
        tree = g_model.core.backend.build_case_trees(encoded)[0]
        scores = g_control.score_cases(
            encoded,
            {
                "module_states": encoded.module_tokens,
                "environment_states": encoded.env_tokens,
                "global_state": encoded.global_token,
            },
            (tree,),
            budgets=PRIMARY_CAPACITY,
        )
        g_full_plan = MechanismPlan.full_access(
            tree, encoded.module_present[0], int(encoded.env_coords.shape[1])
        )
        g_full_prediction, g_full_aux = runner._prediction_with_plan(
            g_model, encoded, batch, g_full_plan, return_interaction_aux=True
        )
        _gfloss, _gfmse, same_g_full_rmse = runner._role_objective(
            g_full_prediction, batch.target_field, sample.role_slices, normalizer, role_scales
        )
        g_full_work = panel._plan_work(
            plan=g_full_plan, encoded=encoded, tree=tree, batch=batch, sample=sample
        )
        g_action_metrics: dict[str, dict[str, Any]] = {}
        g_features: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
        g_plans: dict[str, Any] = {}
        g_cuts: dict[str, tuple[int, ...]] = {}
        for action in ("root", "two_packet", "four_packet"):
            cut, resolved_paths = available_frontier_for_paths(tree, WIND_ACTION_PATHS[action])
            plan = g_control.plans_from_scores(
                scores, encoded, (tree,), hard=True, frontier_cuts=(cut,),
                budget_fractions=PRIMARY_CAPACITY,
            )[0]
            prediction, aux = runner._prediction_with_plan(
                g_model, encoded, batch, plan, return_interaction_aux=True
            )
            candidate_rmse, distortion = _metric_row(
                prediction=prediction, full_prediction=g_full_prediction, batch=batch,
                sample=sample, normalizer=normalizer,
            )
            work = panel._plan_work(
                plan=plan, encoded=encoded, tree=tree, batch=batch, sample=sample
            )
            packet_rows, mm, qe, signatures = _effective_action_features(
                scores=scores, plan=plan, encoded=encoded, cut=cut
            )
            g_action_metrics[action] = {
                "candidate_role_rmse_mps": candidate_rmse,
                "same_g_full_role_rmse_mps": same_g_full_rmse,
                "same_g_full_distortion_role_rmse_mps": distortion,
                "work": work,
                "executor_rows": runner._summarize_executor_rows(aux),
                "resolved_cut_paths": list(resolved_paths),
                "nonredundant_k": int(panel._action_union_k(plan, encoded, cut)),
                "permission_status": {route: plan.permission_status(route) for route in ("MM", "QE")},
            }
            g_features[action] = (packet_rows, mm, qe, signatures)
            g_plans[action] = plan
            g_cuts[action] = tuple(cut)

        full_rows, full_mm, full_qe, full_signatures = _effective_action_features(
            scores=scores, plan=g_full_plan, encoded=encoded, cut=(0,)
        )
        g_action_metrics["full_access"] = {
            "candidate_role_rmse_mps": same_g_full_rmse,
            "same_g_full_role_rmse_mps": same_g_full_rmse,
            "same_g_full_distortion_role_rmse_mps": {role: 0.0 for role in roles},
            "work": g_full_work,
            "executor_rows": runner._summarize_executor_rows(g_full_aux),
            "resolved_cut_paths": [],
            "nonredundant_k": None,
            "permission_status": {
                route: g_full_plan.permission_status(route) for route in ("MM", "QE")
            },
        }
        g_features["full_access"] = (full_rows, full_mm, full_qe, full_signatures)
        g_plans["full_access"] = g_full_plan
        g_cuts["full_access"] = (0,)

        # A matched root-union control uses the four-packet action's source
        # union as one packet. It is not a selector row or a trained root cut.
        four_plan = g_plans["four_packet"]
        four_cut = torch.as_tensor(g_cuts["four_packet"], device=four_plan.split_gates.device)
        root_union_plan = MechanismPlan.full_access(
            tree, encoded.module_present[0], int(encoded.env_coords.shape[1])
        )
        for route in ("MM", "QE"):
            packet_support = four_plan.permission_matrix(route)[four_cut] > 0.0
            source_union = packet_support.any(dim=0)
            root_membership = source_union[None, :].expand_as(
                root_union_plan.permission_matrix(route)
            ).to(dtype=root_union_plan.permission_matrix(route).dtype)
            root_union_plan = root_union_plan.with_permission(route, root_membership)
        root_union_prediction, root_union_aux = runner._prediction_with_plan(
            g_model, encoded, batch, root_union_plan, return_interaction_aux=True
        )
        root_union_rmse, root_union_distortion = _metric_row(
            prediction=root_union_prediction, full_prediction=g_full_prediction, batch=batch,
            sample=sample, normalizer=normalizer,
        )
        root_union_work = panel._plan_work(
            plan=root_union_plan, encoded=encoded, tree=tree, batch=batch, sample=sample
        )
        root_union_features = _effective_action_features(
            scores=scores, plan=root_union_plan, encoded=encoded, cut=(0,)
        )
        root_union_key = hashlib.sha256(f"{case_key}\0g_root_union".encode()).hexdigest()[:20]
        root_union_mm_key = f"g_root_union_mm_{root_union_key}"
        root_union_qe_key = f"g_root_union_qe_{root_union_key}"
        mask_arrays[root_union_mm_key] = np.ascontiguousarray(root_union_features[1], dtype=np.uint8)
        mask_arrays[root_union_qe_key] = np.ascontiguousarray(root_union_features[2], dtype=np.uint8)

        same_weight_controls: dict[str, dict[str, Any]] = {}
        if query_panel == "fixed":
            four_cut_tuple = tuple(g_cuts["four_packet"])
            four_packet_masks = panel._packet_source_masks(four_plan, encoded, four_cut_tuple)
            four_g_support_mm = runner._canonical_route_support(
                four_plan, encoded, tree, "MM"
            )
            four_g_support_qe = runner._query_qe_support(
                four_plan, encoded, sample, batch
            )
            target_pair_counts = {
                "MM": int(four_g_support_mm["actual_live_unique_pairs"]),
                "QE": int(four_g_support_qe["actual_selected_pairs"]),
            }

            # Reuse the selected P scorer on the selected G physical weights.
            # The exact G live pair counts make this a query-bound diagnostic,
            # not a production or query-independent P policy.
            count_matched_plan, count_matched_masks, count_matched_routes = (
                _count_matched_direct_plan(
                    encoded=encoded,
                    tree=tree,
                    batch=batch,
                    scorer=p_control,
                    target_pairs_by_route=target_pair_counts,
                )
            )
            count_matched_prediction, count_matched_aux = panel._predict_direct_pair_full_panel(
                model=g_model, encoded=encoded, batch=batch, plan=count_matched_plan
            )
            count_matched_rmse, count_matched_distortion = _metric_row(
                prediction=count_matched_prediction,
                full_prediction=g_full_prediction,
                batch=batch,
                sample=sample,
                normalizer=normalizer,
            )
            count_matched_mm = runner._canonical_route_support(
                count_matched_plan, encoded, tree, "MM"
            )
            count_matched_qe = runner._query_qe_support(
                count_matched_plan, encoded, sample, batch
            )
            if (
                int(count_matched_mm["actual_live_unique_pairs"]) != target_pair_counts["MM"]
                or int(count_matched_qe["actual_selected_pairs"]) != target_pair_counts["QE"]
            ):
                raise RuntimeError("Count-matched P diagnostic changed its declared live pair counts")
            count_matched_key = hashlib.sha256(
                f"{case_key}\0g_direct_p_count_matched_four_packet".encode()
            ).hexdigest()[:20]
            count_matched_npz_keys = {}
            for route, mask in count_matched_masks.items():
                key = f"g_direct_p_count_matched_{route.lower()}_{count_matched_key}"
                mask_arrays[key] = np.ascontiguousarray(mask, dtype=np.uint8)
                count_matched_npz_keys[f"{route.lower()}_mask_npz_key"] = key
            same_weight_controls["g_direct_p_count_matched_four_packet"] = {
                "definition": (
                    "P scorer applied to selected G encoded inputs and G physical weights; "
                    "top-N direct pairs match the same-case four-packet G MM native-panel and QE live-query counts"
                ),
                "action_key": "four_packet",
                "realized_cut_paths": list(g_action_metrics["four_packet"]["resolved_cut_paths"]),
                "g_four_packet_nonredundant_k": int(g_action_metrics["four_packet"]["nonredundant_k"]),
                "diagnostic_only": True,
                "production_p_policy": False,
                "same_physical_forward_weights_as_grouped_action": True,
                "direct_policy_source": "selected P checkpoint scorer on current G encoded inputs",
                "budget_feature_for_p_scorer": dict(PRIMARY_CAPACITY),
                "selection_rule": "descending P score with stable row-major tie break at the exact G pair count",
                "count_matched_live_pair_counts": count_matched_routes,
                "role_rmse_mps": count_matched_rmse,
                "same_g_full_role_rmse_mps": dict(same_g_full_rmse),
                "same_g_full_distortion_role_rmse_mps": count_matched_distortion,
                "MM_native_panel_support": count_matched_mm,
                "QE_live_query_support": count_matched_qe,
                "native_executor_rows": runner._summarize_executor_rows(count_matched_aux),
                "selected_mask_sha256_by_route": {
                    route: _array_sha256(mask) for route, mask in count_matched_masks.items()
                },
                "selected_mask_shape_by_route": {
                    route: list(mask.shape) for route, mask in count_matched_masks.items()
                },
                **count_matched_npz_keys,
            }

            # Geometry selection preserves each realized G packet's typed
            # source count, then takes nearest sources to that packet's native
            # receiver-anchor centroid. It uses no output values.
            geometry_masks = _geometry_action_masks(
                plan=four_plan, encoded=encoded, tree=tree, cut=four_cut_tuple
            )
            geometry_plan = _plan_with_packet_masks(
                four_plan, cut=four_cut_tuple, masks=geometry_masks
            )
            geometry_prediction, geometry_aux = runner._prediction_with_plan(
                g_model, encoded, batch, geometry_plan, return_interaction_aux=True
            )
            geometry_rmse, geometry_distortion = _metric_row(
                prediction=geometry_prediction,
                full_prediction=g_full_prediction,
                batch=batch,
                sample=sample,
                normalizer=normalizer,
            )
            geometry_work = panel._plan_work(
                plan=geometry_plan, encoded=encoded, tree=tree, batch=batch, sample=sample
            )
            geometry_key = hashlib.sha256(
                f"{case_key}\0g_geometry_selected_four_packet".encode()
            ).hexdigest()[:20]
            geometry_npz_keys = {}
            for route, mask in geometry_masks.items():
                key = f"g_geometry_{route.lower()}_{geometry_key}"
                mask_arrays[key] = np.ascontiguousarray(mask, dtype=np.uint8)
                geometry_npz_keys[f"{route.lower()}_mask_npz_key"] = key
            same_weight_controls["g_geometry_selected_four_packet"] = {
                "definition": "four-packet G cut with per-packet typed source counts held fixed",
                "action_key": "four_packet",
                "realized_cut_paths": list(g_action_metrics["four_packet"]["resolved_cut_paths"]),
                "g_four_packet_nonredundant_k": int(g_action_metrics["four_packet"]["nonredundant_k"]),
                "selection_rule": "nearest valid native sources to weighted packet anchor centroids",
                "role_rmse_mps": geometry_rmse,
                "same_g_full_role_rmse_mps": dict(same_g_full_rmse),
                "same_g_full_distortion_role_rmse_mps": geometry_distortion,
                "canonical_exact_work": float(geometry_work["canonical_total_work"]),
                "canonical_full_work": float(geometry_work["canonical_full_access_total_work"]),
                "work": geometry_work,
                "nonredundant_k": int(panel._action_union_k(geometry_plan, encoded, four_cut_tuple)),
                "native_executor_rows": runner._summarize_executor_rows(geometry_aux),
                "effective_support_shapes": {
                    route: list(mask.shape) for route, mask in geometry_masks.items()
                },
                **geometry_npz_keys,
            }

            # Rewire each typed incidence matrix independently. A zero-switch
            # result is explicitly inactive and is not run as an identical
            # prediction or interpreted as evidence of robustness.
            rewired_masks: dict[str, np.ndarray] = {}
            rewire_swaps: dict[str, int] = {}
            rewire_changed_links: dict[str, int] = {}
            rewire_seed_hashes: dict[str, str] = {}
            for route in ("MM", "QE"):
                stable_seed_bytes = hashlib.sha256(
                    f"{case_key}\0g_four_packet_rewire\0{route}".encode()
                ).digest()
                rewire_seed = int.from_bytes(stable_seed_bytes[:4], "little")
                rewired_masks[route], rewire_swaps[route], rewire_changed_links[route] = (
                    _degree_size_preserving_rewire(
                        four_packet_masks[route], seed=rewire_seed
                    )
                )
                rewire_seed_hashes[route] = hashlib.sha256(stable_seed_bytes).hexdigest()
            total_changed_links = sum(rewire_changed_links.values())
            rewire_record: dict[str, Any] = {
                "definition": "four-packet G support with packet sizes and source degrees preserved per typed route",
                "action_key": "four_packet",
                "realized_cut_paths": list(g_action_metrics["four_packet"]["resolved_cut_paths"]),
                "g_four_packet_nonredundant_k": int(g_action_metrics["four_packet"]["nonredundant_k"]),
                "status": "active" if total_changed_links else "inactive_zero_effective_switches",
                "diagnostic_only": True,
                "seed_sha256_by_route": rewire_seed_hashes,
                "successful_switches_by_route": rewire_swaps,
                "changed_link_count_by_route": rewire_changed_links,
                "actual_changed_link_count": total_changed_links,
                "packet_sizes_preserved_by_route": {
                    route: np.asarray(rewired_masks[route].sum(axis=1), dtype=int).tolist()
                    == np.asarray(four_packet_masks[route].sum(axis=1), dtype=int).tolist()
                    for route in ("MM", "QE")
                },
                "source_degrees_preserved_by_route": {
                    route: np.asarray(rewired_masks[route].sum(axis=0), dtype=int).tolist()
                    == np.asarray(four_packet_masks[route].sum(axis=0), dtype=int).tolist()
                    for route in ("MM", "QE")
                },
            }
            rewire_key = hashlib.sha256(
                f"{case_key}\0g_degree_size_rewire_four_packet".encode()
            ).hexdigest()[:20]
            rewire_npz_keys = {}
            for route, mask in rewired_masks.items():
                key = f"g_rewire_{route.lower()}_{rewire_key}"
                mask_arrays[key] = np.ascontiguousarray(mask, dtype=np.uint8)
                rewire_npz_keys[f"{route.lower()}_mask_npz_key"] = key
            rewire_record.update(rewire_npz_keys)
            if total_changed_links:
                rewire_plan = _plan_with_packet_masks(
                    four_plan, cut=four_cut_tuple, masks=rewired_masks
                )
                rewire_prediction, rewire_aux = runner._prediction_with_plan(
                    g_model, encoded, batch, rewire_plan, return_interaction_aux=True
                )
                rewire_rmse, rewire_distortion = _metric_row(
                    prediction=rewire_prediction,
                    full_prediction=g_full_prediction,
                    batch=batch,
                    sample=sample,
                    normalizer=normalizer,
                )
                rewire_work = panel._plan_work(
                    plan=rewire_plan, encoded=encoded, tree=tree, batch=batch, sample=sample
                )
                rewire_record.update({
                    "role_rmse_mps": rewire_rmse,
                    "same_g_full_role_rmse_mps": dict(same_g_full_rmse),
                    "same_g_full_distortion_role_rmse_mps": rewire_distortion,
                    "canonical_exact_work": float(rewire_work["canonical_total_work"]),
                    "canonical_full_work": float(rewire_work["canonical_full_access_total_work"]),
                    "work": rewire_work,
                    "native_executor_rows": runner._summarize_executor_rows(rewire_aux),
                })
            else:
                rewire_record["inactive_reason"] = (
                    "the bounded bipartite 2-switch search found no valid incidence exchange"
                )
            same_weight_controls["g_degree_size_rewire_four_packet"] = rewire_record

            if training_population_features is None or not training_population_feature_sha256:
                raise ValueError("Selected action table lacks train_fit organizer population features")
            fixed_context = _fixed_population_context(encoded, training_population_features)
            fixed_scores = g_control.score_cases(
                encoded, fixed_context, (tree,), budgets=PRIMARY_CAPACITY
            )
            fixed_population_plan = g_control.plans_from_scores(
                fixed_scores,
                encoded,
                (tree,),
                hard=True,
                frontier_cuts=(four_cut_tuple,),
                budget_fractions=PRIMARY_CAPACITY,
            )[0]
            fixed_population_prediction, fixed_population_aux = runner._prediction_with_plan(
                g_model, encoded, batch, fixed_population_plan, return_interaction_aux=True
            )
            fixed_population_rmse, fixed_population_distortion = _metric_row(
                prediction=fixed_population_prediction,
                full_prediction=g_full_prediction,
                batch=batch,
                sample=sample,
                normalizer=normalizer,
            )
            fixed_population_work = panel._plan_work(
                plan=fixed_population_plan, encoded=encoded, tree=tree,
                batch=batch, sample=sample,
            )
            fixed_population_features = _effective_action_features(
                scores=fixed_scores,
                plan=fixed_population_plan,
                encoded=encoded,
                cut=four_cut_tuple,
            )
            fixed_population_key = hashlib.sha256(
                f"{case_key}\0g_fixed_train_population_four_packet".encode()
            ).hexdigest()[:20]
            fixed_population_npz_keys = {}
            for route, mask in zip(
                ("MM", "QE"), (fixed_population_features[1], fixed_population_features[2]),
                strict=True,
            ):
                key = f"g_fixed_population_{route.lower()}_{fixed_population_key}"
                mask_arrays[key] = np.ascontiguousarray(mask, dtype=np.uint8)
                fixed_population_npz_keys[f"{route.lower()}_mask_npz_key"] = key
            same_weight_controls["g_fixed_population_features_four_packet"] = {
                "definition": "same selected G model/organizer weights with train_fit population-mean organizer tokens",
                "action_key": "four_packet",
                "realized_cut_paths": list(g_action_metrics["four_packet"]["resolved_cut_paths"]),
                "g_four_packet_nonredundant_k": int(g_action_metrics["four_packet"]["nonredundant_k"]),
                "organizer_population_feature_sha256": training_population_feature_sha256,
                "replaced_organizer_inputs": [
                    "module_states", "environment_states", "global_state"
                ],
                "current_case_inputs_retained": [
                    "native module/environment features and coordinates",
                    "receiver tree geometry and role descriptors",
                ],
                "case_varying_action_role_rmse_mps": g_action_metrics["four_packet"]["candidate_role_rmse_mps"],
                "fixed_population_action_role_rmse_mps": fixed_population_rmse,
                "fixed_population_same_g_full_role_rmse_mps": dict(same_g_full_rmse),
                "fixed_population_same_g_full_distortion_role_rmse_mps": fixed_population_distortion,
                "canonical_exact_work": float(fixed_population_work["canonical_total_work"]),
                "canonical_full_work": float(fixed_population_work["canonical_full_access_total_work"]),
                "work": fixed_population_work,
                "nonredundant_k": int(panel._action_union_k(
                    fixed_population_plan, encoded, four_cut_tuple
                )),
                "native_executor_rows": runner._summarize_executor_rows(fixed_population_aux),
                "effective_mm_mask_shape": list(fixed_population_features[1].shape),
                "effective_qe_mask_shape": list(fixed_population_features[2].shape),
                **fixed_population_npz_keys,
            }

        p_model.core.backend.set_cover_mode("external")
        p_encoded = p_model.core.encode_case(batch)
        p_tree = p_model.core.backend.build_case_trees(p_encoded)[0]
        p_full_plan = MechanismPlan.full_access(
            p_tree, p_encoded.module_present[0], int(p_encoded.env_coords.shape[1])
        )
        p_full_prediction, p_full_aux = runner._prediction_with_plan(
            p_model, p_encoded, batch, p_full_plan, return_interaction_aux=True
        )
        _pfloss, _pfmse, p_full_rmse = runner._role_objective(
            p_full_prediction, batch.target_field, sample.role_slices, normalizer, role_scales
        )
        p_full_work = panel._plan_work(
            plan=p_full_plan, encoded=p_encoded, tree=p_tree, batch=batch, sample=sample
        )
        p_direct, _p_soft, p_projections, _p_actual = maturation._direct_plan_pair_by_route(
            model=p_model, encoded=p_encoded, batch=batch, scorer=p_control,
            route_fractions=PRIMARY_CAPACITY,
        )
        p_prediction, p_direct_aux = panel._predict_direct_pair_full_panel(
            model=p_model, encoded=p_encoded, batch=batch, plan=p_direct
        )
        p_projection_map = {name: projection for name, projection in p_projections.items()}
        p_direct_metric = panel._metric(
            action_key="p_direct_primary", prediction=p_prediction, encoded=p_encoded, tree=p_tree,
            plan=p_direct, cut=(0,), paths=("direct_pair_plan",), batch=batch, sample=sample,
            normalizer=normalizer,
            role_scales={role: 1.0 for role in roles}, same_student_full_rmse=p_full_rmse,
            incumbent_rmse=wfull_rmse, executor_aux=dict(p_direct_aux), capacity=PRIMARY_CAPACITY,
            direct_projection=p_projection_map,
        )
        controls = {
            "p": {
                "full_access": {
                    "role_rmse_mps": p_full_rmse,
                    "canonical_exact_work": float(p_full_work["canonical_total_work"]),
                    "canonical_full_work": float(p_full_work["canonical_full_access_total_work"]),
                    "executor_rows": runner._summarize_executor_rows(p_full_aux),
                },
                "direct_primary": {
                    "role_rmse_mps": dict(p_direct_metric["role_rmse_mps"]),
                    "same_p_full_role_rmse_mps": dict(p_direct_metric["same_student_full_role_rmse_mps"]),
                    "canonical_exact_work": float(p_direct_metric["work"]["canonical_total_work"]),
                    "canonical_full_work": float(p_direct_metric["work"]["canonical_full_access_total_work"]),
                    "work": p_direct_metric["work"],
                    "executor_rows": p_direct_metric["native_executor_rows"],
                },
            },
            "g_root_union": {
                "definition": "four-packet typed source union applied as one root packet",
                "role_rmse_mps": root_union_rmse,
                "same_g_full_role_rmse_mps": dict(same_g_full_rmse),
                "same_g_full_distortion_role_rmse_mps": root_union_distortion,
                "nonredundant_k": 1,
                "canonical_exact_work": float(root_union_work["canonical_total_work"]),
                "canonical_full_work": float(root_union_work["canonical_full_access_total_work"]),
                "work": root_union_work,
                "executor_rows": runner._summarize_executor_rows(root_union_aux),
                "mm_mask_npz_key": root_union_mm_key,
                "qe_mask_npz_key": root_union_qe_key,
                "effective_mm_mask_shape": list(root_union_features[1].shape),
                "effective_qe_mask_shape": list(root_union_features[2].shape),
            },
            "g_same_weight_controls": same_weight_controls,
        }
        role_descriptors = receiver_role_descriptors(tree, role_count=len(roles)).detach().cpu().numpy().astype(
            np.float32, copy=False
        )

    base_info = {
        "case_key": case_key,
        "row_id": int(row_id),
        "layout_id": layout_id,
        "layout_key": layout_key,
        "family_key": layout_key,
        "module_count": int(case.n_turbines),
        "wind_direction_deg": float(case.wind_direction_deg),
        "source_partition": "native_training",
        "split": str(split),
        "query_panel": str(query_panel),
        "query_seed": int(query_seed),
        "query_seed_retry_count": int(query_seed_retry_count),
        "query_repeat_disjoint_from_fixed": (
            None if query_panel != "query_repeat" else int(query_repeat_overlap_count or 0) == 0
        ),
        "query_repeat_overlap_count": query_repeat_overlap_count,
        "query_repeat_candidate_attempt_count": query_repeat_candidate_attempt_count,
        "query_sample_index_sha256": query_sample_index_sha256,
        "fixed_query_sample_index_sha256": (
            query_sample_index_sha256 if query_panel == "fixed" else fixed_query_sample_index_sha256
        ),
        "selector_primary_fit_row": query_panel == "fixed",
        "role_query_counts": {str(key): int(value) for key, value in query_counts.items()},
        "role_names": list(roles),
    }

    action_rows: list[dict[str, Any]] = []
    for action in ACTION_ORDER:
        metric = g_action_metrics[action]
        packet_rows, mm, qe, signatures = g_features[action]
        key_suffix = hashlib.sha256(f"{case_key}\0{action}".encode()).hexdigest()[:20]
        packet_key = f"packet_{key_suffix}"
        budget_key = f"budget_{key_suffix}"
        roles_key = f"receiver_roles_{key_suffix}"
        mm_key = f"mm_mask_{key_suffix}"
        qe_key = f"qe_mask_{key_suffix}"
        feature_arrays[packet_key] = np.ascontiguousarray(packet_rows, dtype=np.float32)
        feature_arrays[budget_key] = _action_budget_vector(action)
        feature_arrays[roles_key] = np.ascontiguousarray(role_descriptors, dtype=np.float32)
        mask_arrays[mm_key] = np.ascontiguousarray(mm, dtype=np.uint8)
        mask_arrays[qe_key] = np.ascontiguousarray(qe, dtype=np.uint8)
        exposure_row = dict(exposure[action]) if action != "full_access" else {
            "trained_action": False,
            "reason": "full access is a control, not a trained sparse cut",
            "required_complete_passes": 2,
            "completed_primary_action_passes": 0,
        }
        exact_work = float(metric["work"]["canonical_total_work"])
        full_work = float(metric["work"]["canonical_full_access_total_work"])
        trained_sparse = _trained_sparse_action(
            action, exact_work=exact_work, full_work=full_work,
            exposure_record=exposure_row,
            row_id=int(row_id),
            realized_cut_paths=metric.get("resolved_cut_paths", []),
            nonredundant_k=(None if action == "full_access" else int(metric["nonredundant_k"])),
        )
        candidate = metric["candidate_role_rmse_mps"]
        action_rows.append({
            **base_info,
            "action_key": action,
            "forward_checkpoint_sha256": "__SET_G_SHA256__",
            "full_access": action == "full_access",
            "trained_sparse": trained_sparse,
            "trained_action_after_two_complete_passes": bool(exposure_row.get("trained_action", False)),
            "training_exposure": _row_action_exposure(
                exposure_row, action=action, row_id=int(row_id),
                realized_cut_paths=metric.get("resolved_cut_paths", []),
                nonredundant_k=(None if action == "full_access" else int(metric["nonredundant_k"])),
            ) if action != "full_access" else exposure_row,
            "requested_cut_paths": list(WIND_ACTION_PATHS[action]) if action != "full_access" else [],
            "realized_cut_paths": list(metric.get("resolved_cut_paths", [])),
            "nonredundant_k": None if action == "full_access" else int(metric["nonredundant_k"]),
            "packet_mask_signature_sha256": signatures,
            "permission_status": metric["permission_status"],
            "candidate_role_error": {role: float(candidate[role]) for role in roles},
            "candidate_role_rmse_mps": {role: float(candidate[role]) for role in roles},
            "incumbent_role_error": {role: float(wfull_rmse[role]) for role in roles},
            "retained_wfull_role_rmse_mps": {role: float(wfull_rmse[role]) for role in roles},
            "same_g_full_role_rmse_mps": {
                role: float(metric["same_g_full_role_rmse_mps"][role]) for role in roles
            },
            "same_g_full_distortion_role_rmse_mps": {
                role: float(metric["same_g_full_distortion_role_rmse_mps"][role]) for role in roles
            },
            "exact_work": exact_work,
            "full_work": full_work,
            "canonical_work_by_route": metric["work"]["routes"],
            "live_query_qe_work": metric["work"]["qe_live_query_panel"],
            "executed_work": {
                "executor": "dense_masked native executor",
                "native_executor_rows": metric["executor_rows"],
                "measured": True,
            },
            "packet_rows_npz_key": packet_key,
            "budget_vector_npz_key": budget_key,
            "receiver_role_features_npz_key": roles_key,
            "mm_mask_npz_key": mm_key,
            "qe_mask_npz_key": qe_key,
            "feature_packet_rows_shape": list(packet_rows.shape),
            "effective_mm_mask_shape": list(mm.shape),
            "effective_qe_mask_shape": list(qe.shape),
        })
    controls["p"].update(base_info)
    controls["g_root_union"].update(base_info)
    for control in controls["g_same_weight_controls"].values():
        control.update(base_info)
    return action_rows, controls, {
        "wfull_role_rmse_mps": dict(wfull_rmse),
        "same_g_full_role_rmse_mps": dict(same_g_full_rmse),
        "same_g_full_work": g_full_work,
    }


def _layout_rows(
    train_rows: Sequence[int], view: Any, *, layout_limit: int, seed: int
) -> dict[int, list[int]]:
    by_layout: dict[int, list[int]] = defaultdict(list)
    for row in train_rows:
        by_layout[int(view.metadata["layout_index"][int(row)])].append(int(row))
    layouts = sorted(by_layout)
    if layout_limit < 1 or layout_limit > 24:
        raise ValueError("The selected native action table is bounded to 1–24 layouts")
    layout_limit = min(layout_limit, len(layouts))
    rng = np.random.default_rng(int(seed))
    chosen = sorted(map(int, rng.choice(layouts, size=layout_limit, replace=False)))
    result = {
        layout: sorted(
            by_layout[layout],
            key=lambda row: (float(view.run(row).wind_direction_deg), row),
        )
        for layout in chosen
    }
    for layout, rows in result.items():
        if not rows:
            raise ValueError(f"Selected Wind layout {layout} has no eligible training row")
    return result


def _read_exposure(
    path: Path, *, training_rows: Sequence[int], update_count: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"G training update ledger is missing: {path}")
    ledger_records, ledger_bytes = _read_jsonl_snapshot(path)
    recovery_path = path.parent / "recovery_events.jsonl"
    recovery_records, recovery_bytes = (
        _read_jsonl_snapshot(recovery_path) if recovery_path.is_file() else ([], None)
    )
    archived_keys = sorted({
        str(key)
        for event in recovery_records
        for field in ("orphaned_update_attempt_ids", "orphaned_attempt_keys")
        for key in event.get(field, [])
        if key
    })
    exposure = _summarize_action_exposure(
        ledger_records,
        training_rows=training_rows,
        selected_update=update_count,
        archived_optimizer_attempt_keys=archived_keys,
    )
    return exposure, {
        "update_ledger_path": str(path),
        "update_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "update_ledger_snapshot_bytes": len(ledger_bytes),
        "update_ledger_snapshot_record_count": len(ledger_records),
        "recovery_ledger_path": str(recovery_path) if recovery_path.is_file() else None,
        "recovery_ledger_sha256": (
            hashlib.sha256(recovery_bytes).hexdigest() if recovery_bytes is not None else None
        ),
        "recovery_ledger_snapshot_bytes": len(recovery_bytes) if recovery_bytes is not None else None,
        "recovery_ledger_snapshot_record_count": (
            len(recovery_records) if recovery_bytes is not None else None
        ),
        "archived_optimizer_attempt_key_count": len(archived_keys),
        "selected_update_count": int(update_count),
    }


def _read_jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    """Parse and hash the same complete append-only ledger snapshot."""
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Ledger snapshot ends with a partial JSONL record: {path}")
    records = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    return records, raw


def _query_repeat_overlap_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize measured fixed/repeat overlap once per physical query panel."""
    by_case: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("query_panel", "")) != "query_repeat":
            continue
        case_key = str(row["case_key"])
        identity = {
            "row_id": int(row["row_id"]),
            "overlap_count": row.get("query_repeat_overlap_count"),
            "disjoint": row.get("query_repeat_disjoint_from_fixed"),
            "attempt_count": row.get("query_repeat_candidate_attempt_count"),
        }
        prior = by_case.setdefault(case_key, identity)
        if prior != identity:
            raise ValueError(f"Query-repeat overlap metadata changes across actions for {case_key!r}")
    records = list(by_case.values())
    if any(
        item["overlap_count"] is None
        or int(item["overlap_count"]) < 0
        or bool(item["disjoint"]) != (int(item["overlap_count"]) == 0)
        for item in records
    ):
        raise ValueError("Query-repeat disjointness must agree with measured sample-index overlap")
    disjoint_count = sum(int(item["overlap_count"]) == 0 for item in records)
    overlap_count = len(records) - disjoint_count
    overlaps = [int(item["overlap_count"]) for item in records]
    return {
        "measured_repeat_case_count": len(records),
        "disjoint_repeat_case_count": disjoint_count,
        "overlapping_repeat_case_count": overlap_count,
        "total_intersecting_query_indices": sum(overlaps),
        "maximum_intersection_count": max(overlaps) if overlaps else None,
        "all_measured_repeats_disjoint": (disjoint_count == len(records)) if records else None,
        "rows": [{
            "row_id": int(item["row_id"]),
            "overlap_count": int(item["overlap_count"]),
            "disjoint": bool(item["disjoint"]),
            "candidate_attempt_count": int(item["attempt_count"]),
        } for item in records],
    }


def _evaluate(
    *,
    run_dir: Path,
    update_count: int,
    p_update_count: int,
    output_dir: Path,
    layout_limit: int,
    repeat_layout_limit: int,
    selection_seed: int,
) -> dict[str, Any]:
    config_path = Path(runner.DEFAULT_CONFIG).resolve()
    config = runner._load_config(config_path)
    view, _canonical, train_rows, split_record = runner._native_inputs(config)
    source_path, source_payload, normalizer, source_sha = runner._load_source(config)
    if source_sha != runner.EXPECTED_SOURCE_SHA256:
        raise ValueError("Retained W-full checkpoint differs from the audited Run2110 source")
    role_scales = {
        str(key): float(value)
        for key, value in config["forward"]["stage_a"]["role_loss_scales_mps"].items()
    }
    if set(role_scales) != set(runner.ROLE_NAMES):
        raise ValueError("Native physical role scales differ from the Wind role order")

    manifest_path = run_dir / "run_manifest.json"
    manifest_sha = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != RUN_ID:
        raise ValueError("Selected Wind run manifest is not Run2112")
    g_path, g_binding_record, g_binding_record_sha, g_binding_source = panel._verified_checkpoint_binding(
        run_dir, arm="g_packet", update_count=update_count
    )
    p_path, p_binding_record, p_binding_record_sha, p_binding_source = panel._verified_checkpoint_binding(
        run_dir, arm="direct_pair", update_count=p_update_count
    )
    g_sha, p_sha = _sha256(g_path), _sha256(p_path)
    g_payload = load_trusted_checkpoint(g_path, map_location="cpu")
    p_payload = load_trusted_checkpoint(p_path, map_location="cpu")
    expected_train_rows_sha = split_record["student_train_rows_sha256"]
    for arm, payload, digest, expected_arm, binding_record in (
        ("G", g_payload, g_sha, "g_packet", g_binding_record),
        ("P", p_payload, p_sha, "direct_pair", p_binding_record),
    ):
        expected_update = update_count if arm == "G" else p_update_count
        if (
            not isinstance(payload, Mapping)
            or payload.get("run_id") != RUN_ID
            or payload.get("arm") != expected_arm
            or int(payload.get("update_count", -1)) != int(expected_update)
            or payload.get("source_checkpoint_sha256") != source_sha
            or payload.get("train_rows_sha256") != expected_train_rows_sha
            or digest != binding_record.get("checkpoint_sha256")
        ):
            raise ValueError(f"{arm} checkpoint does not match selected update, source, split, or verified binding")

    layout_rows = _layout_rows(
        train_rows, view, layout_limit=layout_limit, seed=selection_seed
    )
    split_by_layout = _split_selected_layouts(list(layout_rows), seed=selection_seed + 1)
    repeat_rng = np.random.default_rng(selection_seed + 2)
    repeat_count = min(max(0, int(repeat_layout_limit)), len(layout_rows))
    repeat_layouts = (
        sorted(map(int, repeat_rng.choice(sorted(layout_rows), size=repeat_count, replace=False)))
        if repeat_count else []
    )

    first_row = next(iter(layout_rows.values()))[0]
    init_case, _init_sample, init_batch = maturation._case_batch(
        view=view, row_id=first_row, query_seed=FIXED_QUERY_SEED + first_row,
        role_counts=Q2048_ROLE_COUNTS, sampler=runner.NativeRoleCatalogueCache(),
        normalizer=normalizer, device=torch.device("cuda:0"),
    )
    del init_case
    source_model = runner._new_model_from_source(
        source_payload, normalizer, init_batch, torch.device("cuda:0")
    )
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    g_model, g_control = panel._load_student(
        arm="g", payload=g_payload, source_payload=source_payload, normalizer=normalizer,
        batch=init_batch, forward_config=config["forward"], device=torch.device("cuda:0"),
    )
    p_model, p_control = panel._load_student(
        arm="p", payload=p_payload, source_payload=source_payload, normalizer=normalizer,
        batch=init_batch, forward_config=config["forward"], device=torch.device("cuda:0"),
    )
    exposure, exposure_lineage = _read_exposure(
        run_dir / "arms" / "g_packet" / "updates.jsonl",
        training_rows=train_rows, update_count=update_count,
    )

    sampler = runner.NativeRoleCatalogueCache()
    population_input_records: list[dict[str, torch.Tensor]] = []
    training_population_rows: list[int] = []
    training_population_layouts: list[int] = []
    for layout, row_ids in sorted(layout_rows.items()):
        if split_by_layout[layout] != "train_fit":
            continue
        for row_id in row_ids:
            _population_case, _population_sample, population_batch = maturation._case_batch(
                view=view,
                row_id=int(row_id),
                query_seed=int(FIXED_QUERY_SEED + int(row_id) * 1_009),
                role_counts=Q5_FEATURE_ROLE_COUNTS,
                sampler=sampler,
                normalizer=normalizer,
                device=torch.device("cuda:0"),
            )
            with torch.inference_mode():
                g_model.core.backend.set_cover_mode("external")
                population_encoded = g_model.core.encode_case(population_batch)
            population_input_records.append({
                "module_states": population_encoded.module_tokens[0].detach(),
                "module_valid": population_encoded.module_present[0].detach() > 0.5,
                "environment_states": population_encoded.env_tokens[0].detach(),
                "environment_weights": population_encoded.env_weights[0].detach(),
                "global_state": population_encoded.global_token[0].detach(),
            })
            training_population_rows.append(int(row_id))
            training_population_layouts.append(int(layout))
    training_population_features = _training_population_feature_means(population_input_records)
    population_feature_arrays = {
        f"control_training_population_{name}": value.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        for name, value in training_population_features.items()
    }
    training_population_feature_sha256 = hashlib.sha256(b"".join(
        name.encode() + b"\0" + np.ascontiguousarray(value).tobytes()
        for name, value in sorted(population_feature_arrays.items())
    )).hexdigest()
    training_population_audit = {
        "population_basis": "selected action-table train_fit layout families, fixed native rows only",
        "source_partition": "native_training",
        "split": "train_fit",
        "selected_g_checkpoint_sha256": g_sha,
        "fixed_query_panel_only": True,
        "reference_targets_used": False,
        "case_count": len(population_input_records),
        "row_ids": training_population_rows,
        "layout_ids_by_row": training_population_layouts,
        "row_layout_identities": [
            {"row_id": row_id, "layout_id": layout_id}
            for row_id, layout_id in zip(
                training_population_rows, training_population_layouts, strict=True
            )
        ],
        "query_role_counts_used_to_materialize_rows": Q5_FEATURE_ROLE_COUNTS,
        "organizer_inputs_query_independent": True,
        "aggregation": (
            "equal row weight; valid-module arithmetic mean; environment-token mean weighted by "
            "that row's environment measure; one global token mean per row"
        ),
        "replaced_inputs": ["module_states", "environment_states", "global_state"],
        "current_case_geometry_and_adapter_features_retained": True,
        "organizer_population_feature_sha256": training_population_feature_sha256,
        "feature_npz_keys": sorted(population_feature_arrays),
    }

    feature_arrays: dict[str, np.ndarray] = {}
    feature_arrays.update(population_feature_arrays)
    mask_arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    p_controls: list[dict[str, Any]] = []
    g_root_union_controls: list[dict[str, Any]] = []
    g_direct_p_count_matched_controls: list[dict[str, Any]] = []
    g_geometry_controls: list[dict[str, Any]] = []
    g_rewire_controls: list[dict[str, Any]] = []
    g_fixed_population_controls: list[dict[str, Any]] = []
    measured_wfull: list[dict[str, Any]] = []
    q512_deltas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {role: [] for role in runner.ROLE_NAMES}
    )
    fixed_query_indices_by_row: dict[int, set[int]] = {}
    evaluations: list[tuple[int, str, int, Mapping[str, int]]] = []
    for layout, row_ids in sorted(layout_rows.items()):
        split = split_by_layout[layout]
        for row_id in row_ids:
            fixed_seed = int(FIXED_QUERY_SEED + int(row_id) * 1_009)
            evaluations.append((row_id, "fixed", fixed_seed, Q2048_ROLE_COUNTS))
            if layout in repeat_layouts:
                repeat_seed = int(REPEAT_QUERY_SEED + int(row_id) * 1_009)
                evaluations.append((row_id, "query_repeat", repeat_seed, Q2048_ROLE_COUNTS))
    fixed_query_sample_index_sha_by_row: dict[int, str] = {}

    fixed_panel_count = sum(query_panel == "fixed" for _row, query_panel, _seed, _counts in evaluations)
    repeat_panel_count = len(evaluations) - fixed_panel_count
    q512_calibration_panel_count = sum(
        query_panel == "fixed" and split_by_layout[int(view.metadata["layout_index"][row_id])] == "train_fit"
        for row_id, query_panel, _seed, _counts in evaluations
    )
    prediction_calls_per_primary_panel = 8
    extra_fixed_diagnostic_prediction_upper_bound = 4
    predicted_native_prediction_calls_upper_bound = (
        len(evaluations) * prediction_calls_per_primary_panel
        + fixed_panel_count * extra_fixed_diagnostic_prediction_upper_bound
        + q512_calibration_panel_count
    )
    call_forecast = {
        "measured": False,
        "basis": "eight primary native predictions per listed panel; up to four same-weight fixed-panel controls plus one Q512 G-full calibration prediction per train_fit row",
        "fixed_q2048_panel_count": fixed_panel_count,
        "query_repeat_q2048_panel_count": repeat_panel_count,
        "same_weight_control_predictions_per_fixed_panel_upper_bound": extra_fixed_diagnostic_prediction_upper_bound,
        "same_weight_control_prediction_upper_bound": fixed_panel_count * extra_fixed_diagnostic_prediction_upper_bound,
        "train_fit_organizer_population_encode_case_count": len(training_population_rows),
        "train_fit_organizer_population_materialization_query_points": (
            len(training_population_rows) * sum(Q5_FEATURE_ROLE_COUNTS.values())
        ),
        "q512_train_fit_calibration_panel_count": q512_calibration_panel_count,
        "q2048_role_query_counts": dict(Q2048_ROLE_COUNTS),
        "q512_role_query_counts": dict(Q512_ROLE_COUNTS),
        "q2048_native_prediction_calls": len(evaluations) * prediction_calls_per_primary_panel,
        "q512_native_prediction_calls": q512_calibration_panel_count,
        "estimated_native_prediction_calls_upper_bound": predicted_native_prediction_calls_upper_bound,
        "q2048_query_points_total": (fixed_panel_count + repeat_panel_count) * sum(Q2048_ROLE_COUNTS.values()),
        "q512_query_points_total": q512_calibration_panel_count * sum(Q512_ROLE_COUNTS.values()),
        "query_points_total": (
            (fixed_panel_count + repeat_panel_count) * sum(Q2048_ROLE_COUNTS.values())
            + q512_calibration_panel_count * sum(Q512_ROLE_COUNTS.values())
        ),
        "estimated_decoder_query_evaluations_upper_bound": (
            (len(evaluations) * prediction_calls_per_primary_panel
             + fixed_panel_count * extra_fixed_diagnostic_prediction_upper_bound)
            * sum(Q2048_ROLE_COUNTS.values())
            + q512_calibration_panel_count * sum(Q512_ROLE_COUNTS.values())
        ),
    }
    print(json.dumps({"preflight_native_call_forecast": call_forecast}, sort_keys=True), flush=True)

    for row_id, query_panel, query_seed, query_counts in evaluations:
        layout = int(view.metadata["layout_index"][row_id])
        split = split_by_layout[layout]
        case_sample = maturation._case_batch(
            view=view, row_id=row_id, query_seed=query_seed, role_counts=query_counts,
            sampler=sampler, normalizer=normalizer, device=torch.device("cuda:0"),
        )
        retry_count = 0
        repeat_overlap_count: int | None = None
        repeat_attempt_count: int | None = None
        if query_panel == "fixed":
            fixed_indices = set(map(int, np.asarray(case_sample[1].flat_indices).tolist()))
            fixed_query_indices_by_row[row_id] = fixed_indices
            fixed_query_sample_index_sha_by_row[row_id] = hashlib.sha256(
                np.ascontiguousarray(case_sample[1].flat_indices, dtype=np.int64).tobytes()
            ).hexdigest()
        elif query_panel == "query_repeat":
            fixed_indices = fixed_query_indices_by_row[row_id]
            initial_seed = query_seed
            best_sample = case_sample
            best_seed = query_seed
            best_indices = set(map(int, np.asarray(case_sample[1].flat_indices).tolist()))
            best_overlap = len(best_indices & fixed_indices)
            repeat_attempt_count = 0
            for attempt in range(1024):
                candidate_indices = set(map(int, np.asarray(case_sample[1].flat_indices).tolist()))
                overlap = len(candidate_indices & fixed_indices)
                repeat_attempt_count = attempt + 1
                if overlap < best_overlap:
                    best_sample, best_seed, best_indices, best_overlap = (
                        case_sample, query_seed, candidate_indices, overlap
                    )
                if overlap == 0:
                    best_sample, best_seed, best_indices, best_overlap = (
                        case_sample, query_seed, candidate_indices, overlap
                    )
                    break
                if attempt + 1 < 1024:
                    query_seed += 1
                    case_sample = maturation._case_batch(
                        view=view, row_id=row_id, query_seed=query_seed, role_counts=query_counts,
                        sampler=sampler, normalizer=normalizer, device=torch.device("cuda:0"),
                    )
            case_sample = best_sample
            query_seed = best_seed
            retry_count = query_seed - initial_seed
            repeat_overlap_count = int(best_overlap)
        action_rows, controls, baseline_record = _case_record(
            split=split, row_id=row_id, query_panel=query_panel, query_seed=query_seed,
            query_counts=query_counts, view=view, sampler=sampler, normalizer=normalizer,
            role_scales=role_scales, source_model=source_model, g_model=g_model,
            g_control=g_control, p_model=p_model, p_control=p_control, device=torch.device("cuda:0"),
            feature_arrays=feature_arrays, mask_arrays=mask_arrays, exposure=exposure,
            training_population_features=training_population_features,
            training_population_feature_sha256=training_population_feature_sha256,
            prepared_inputs=case_sample, query_seed_retry_count=retry_count,
            fixed_query_sample_index_sha256=fixed_query_sample_index_sha_by_row.get(row_id),
            query_repeat_overlap_count=repeat_overlap_count,
            query_repeat_candidate_attempt_count=repeat_attempt_count,
        )
        for row in action_rows:
            row["forward_checkpoint_sha256"] = g_sha
        rows.extend(action_rows)
        controls["p"]["p_control_checkpoint_sha256"] = p_sha
        controls["p"]["control_checkpoint_path"] = str(p_path)
        controls["p"]["p_control_update_count"] = int(p_update_count)
        controls["p"]["g_p_update_counts_equal"] = int(update_count) == int(p_update_count)
        controls["p"]["g_p_comparison_class"] = (
            "matched_checkpoint_pair" if int(update_count) == int(p_update_count)
            else "unequal_update_diagnostic"
        )
        controls["g_root_union"]["forward_checkpoint_sha256"] = g_sha
        p_controls.append(controls["p"])
        g_root_union_controls.append(controls["g_root_union"])
        if query_panel == "fixed":
            for key, target in (
                ("g_direct_p_count_matched_four_packet", g_direct_p_count_matched_controls),
                ("g_geometry_selected_four_packet", g_geometry_controls),
                ("g_degree_size_rewire_four_packet", g_rewire_controls),
                ("g_fixed_population_features_four_packet", g_fixed_population_controls),
            ):
                record = controls["g_same_weight_controls"][key]
                record["forward_checkpoint_sha256"] = g_sha
                record["direct_policy_checkpoint_sha256"] = p_sha if key.startswith("g_direct_p") else None
                record["selected_g_update_count"] = int(update_count)
                record["selected_direct_policy_update_count"] = (
                    int(p_update_count) if key.startswith("g_direct_p") else None
                )
                record["direct_policy_checkpoint_path"] = (
                    str(p_path) if key.startswith("g_direct_p") else None
                )
                target.append(record)

        if query_panel != "fixed":
            continue
        measured_wfull.append({
            "family_key": str(layout),
            "row_id": row_id,
            "wfull_role_rmse_mps": baseline_record["wfull_role_rmse_mps"],
        })
        if split != "train_fit":
            continue

        # Compare same selected G full weights at Q512 and Q2048 on the same
        # physical training row. This calibration is never used as a fit row.
        _case512, sample512, batch512 = maturation._case_batch(
            view=view, row_id=row_id, query_seed=query_seed, role_counts=Q512_ROLE_COUNTS,
            sampler=sampler, normalizer=normalizer, device=torch.device("cuda:0"),
        )
        with torch.inference_mode():
            encoded512 = g_model.core.encode_case(batch512)
            tree512 = g_model.core.backend.build_case_trees(encoded512)[0]
            full512 = MechanismPlan.full_access(
                tree512, encoded512.module_present[0], int(encoded512.env_coords.shape[1])
            )
            prediction512, _aux512 = runner._prediction_with_plan(
                g_model, encoded512, batch512, full512
            )
            _l512, _m512, rmse512 = runner._role_objective(
                prediction512, batch512.target_field, sample512.role_slices, normalizer, role_scales
            )
        same_g_full = baseline_record["same_g_full_role_rmse_mps"]
        for role in runner.ROLE_NAMES:
            q512_deltas[str(layout)][role].append(
                abs(float(same_g_full[role]) - float(rmse512[role]))
            )

    roles = tuple(runner.ROLE_NAMES)
    wfull_train_records = [
        record for record in measured_wfull
        if split_by_layout[int(record["family_key"])] == "train_fit"
    ]
    numerical_floor = _numerical_floor_role_mps(wfull_train_records, roles=roles)
    sampling_sensitivity, sampling_sensitivity_meta = _sampling_allowance_role_mps(
        q512_deltas, roles=roles
    )
    feature_hashes = {key: _array_sha256(value) for key, value in feature_arrays.items()}
    mask_hashes = {key: _array_sha256(value) for key, value in mask_arrays.items()}
    _validate_action_rows(rows, expected_sha=g_sha)
    query_repeat_overlap = _query_repeat_overlap_summary(rows)
    call_forecast["same_weight_control_prediction_count_executed"] = (
        len(g_direct_p_count_matched_controls)
        + len(g_geometry_controls)
        + sum("role_rmse_mps" in record for record in g_rewire_controls)
        + len(g_fixed_population_controls)
    )
    call_forecast["rewire_predictions_skipped_inactive_count"] = sum(
        record.get("status") == "inactive_zero_effective_switches"
        for record in g_rewire_controls
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "action_features.npz"
    mask_path = output_dir / "action_masks.npz"
    np.savez_compressed(feature_path, **feature_arrays)
    np.savez_compressed(mask_path, **mask_arrays)

    dataset_manifest_path = runner.resolve_path(str(config["dataset"]["manifest"]))
    table = {
        "format_version": 2,
        "study": "Run2112 selected-checkpoint Wind native action table",
        "run_id": RUN_ID,
        "selected_update_count": int(update_count),
        "p_control_update_count": int(p_update_count),
        "g_p_update_counts_equal": int(update_count) == int(p_update_count),
        "g_p_comparison_class": (
            "matched_checkpoint_pair" if int(update_count) == int(p_update_count)
            else "unequal_update_diagnostic"
        ),
        "forward_checkpoint_path": str(g_path),
        "forward_checkpoint_sha256": g_sha,
        "p_control_checkpoint_path": str(p_path),
        "p_control_checkpoint_sha256": p_sha,
        "retained_wfull_checkpoint_path": str(source_path),
        "retained_wfull_checkpoint_sha256": source_sha,
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": manifest_sha,
        "evaluation_driver_path": str(Path(__file__).resolve()),
        "evaluation_driver_sha256": _sha256(Path(__file__).resolve()),
        **_checkpoint_binding_fields(
            "g", g_binding_record, g_binding_record_sha, g_binding_source
        ),
        **_checkpoint_binding_fields(
            "p", p_binding_record, p_binding_record_sha, p_binding_source
        ),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "native_dataset_manifest_path": str(dataset_manifest_path),
        "native_dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "training_split_identity": split_record,
        "roles": list(roles),
        "numerical_floor_role_mps": numerical_floor,
        # Every candidate and retained reference is scored on identical Q2048
        # receivers. Q512-vs-Q2048 sampling variation is useful uncertainty
        # evidence, but it is not a software-numerical or reference-discrepancy
        # allowance for declaring a sparse action physically adequate.
        "absolute_allowance_role_mps": [0.0 for _ in roles],
        "absolute_allowance_calibration": {
            "calibration_kind": "zero extra physical-risk allowance for paired same-query comparison",
            "method": "candidate and retained W-full errors use identical native query receivers; no measured software-numerical or reference-discrepancy allowance is added",
        },
        "query_sampling_sensitivity_role_mps": sampling_sensitivity,
        "query_sampling_sensitivity": sampling_sensitivity_meta,
        "g_action_exposure_lineage": exposure_lineage,
        "numerical_floor_method": (
            "max(1e-8 m/s, 1e-6 times median train-family retained W-full role RMSE)"
        ),
        "feature_npz_path": str(feature_path),
        "mask_npz_path": str(mask_path),
        "feature_npz_sha256": _sha256(feature_path),
        "mask_npz_sha256": _sha256(mask_path),
        "feature_sha256_by_npz_key": feature_hashes,
        "mask_sha256_by_npz_key": mask_hashes,
        "capacity_vector_order": ["MM", "QE"],
        "primary_capacity": {"MM": PRIMARY_CAPACITY["MM"], "QE": PRIMARY_CAPACITY["QE"]},
        "action_order": list(ACTION_ORDER),
        "action_exposure_by_key": exposure,
        "training_population_organizer_features": training_population_audit,
        "panel_design": {
            "candidate_measurements_use_native_training_rows_only": True,
            "selected_layout_count": len(layout_rows),
            "split_layout_count": {
                split: len({family for family, assigned in split_by_layout.items() if assigned == split})
                for split in ("train_fit", "dev", "held_family_audit")
            },
            "all_native_directions_for_selected_layouts": True,
            "primary_query_panel": "fixed",
            "query_repeat_panel": "query_repeat",
            "query_repeat_layouts": repeat_layouts,
            "fixed_role_query_counts": dict(Q2048_ROLE_COUNTS),
            "query_repeat_role_query_counts": dict(Q2048_ROLE_COUNTS),
            "q512_calibration_role_query_counts": dict(Q512_ROLE_COUNTS),
            "query_repeat_is_disjoint": query_repeat_overlap["all_measured_repeats_disjoint"],
            "query_repeat_overlap_summary": query_repeat_overlap,
            "hidden_test_rows_opened": False,
            "query_rows_are_split_by_layout_family": True,
            "same_weight_controls_run_on_fixed_query_panels_only": True,
            "same_weight_controls_action_key": "four_packet",
            "same_weight_control_prediction_upper_bound_per_fixed_panel": extra_fixed_diagnostic_prediction_upper_bound,
        },
        "p_controls_are_excluded_from_selector_rows": True,
        "all_same_weight_controls_are_excluded_from_selector_rows": True,
        "same_weight_control_interpretation_note": (
            "These selected-weight interventions compare computations and source support; they do not identify physical causal edges. "
            "The count-matched P reprojection is query-bound and is not the production P policy."
        ),
        "g_p_pairing_note": (
            "G/P checkpoint comparison uses equal selected update counts."
            if int(update_count) == int(p_update_count)
            else "Unequal-update diagnostic only; do not interpret as a matched G/P comparison."
        ),
        "native_call_forecast": call_forecast,
        "p_controls": p_controls,
        "g_root_union_controls": g_root_union_controls,
        "g_direct_p_count_matched_controls": g_direct_p_count_matched_controls,
        "g_geometry_controls": g_geometry_controls,
        "g_rewire_controls": g_rewire_controls,
        "g_fixed_population_feature_controls": g_fixed_population_controls,
        "rows": rows,
        "case_count": len({row["case_key"] for row in rows}),
        "selector_action_row_count": len(rows),
        "physical_grid_calls": 0,
        "reference": "stored native OpenFOAM CFD velocity fields; no new physical solve",
    }
    table_path = output_dir / "action_table.json"
    _write_json(table_path, table)
    return {
        "status": "completed",
        "action_table": str(table_path),
        "action_table_sha256": _sha256(table_path),
        "feature_npz": str(feature_path),
        "feature_npz_sha256": table["feature_npz_sha256"],
        "mask_npz": str(mask_path),
        "mask_npz_sha256": table["mask_npz_sha256"],
        "selected_g_sha256": g_sha,
        "selected_p_sha256": p_sha,
        "case_count": table["case_count"],
        "action_rows": len(rows),
        "output_dir": str(output_dir),
    }


def main() -> None:
    default_run_dir = (
        PROJECT
        / "Trained_Results/WindFarm/HONF_Forward_Runs/Run_2112_controlled_maturation_20260929"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--update-count", type=int, required=True)
    parser.add_argument(
        "--p-update-count", type=int,
        help="selected P control checkpoint update; defaults to the G update",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--layout-count", type=int, default=24,
        help="number of distinct eligible training layouts (bounded to 24)",
    )
    parser.add_argument("--query-repeat-layout-count", type=int, default=6)
    parser.add_argument("--selection-seed", type=int, default=21_122_471)
    args = parser.parse_args()
    if (
        args.update_count <= 100
        or (args.p_update_count is not None and args.p_update_count <= 100)
        or not 6 <= args.layout_count <= 24
        or args.query_repeat_layout_count < 0
        or args.query_repeat_layout_count > args.layout_count
    ):
        raise ValueError("update-count must exceed 100; layout counts must be nonnegative")
    if (
        Path(sys.executable).resolve()
        != Path("/home/wanglz/miniconda3/envs/ModularDT/bin/python").resolve()
    ):
        raise RuntimeError("Run native selected-action evaluation with the ModularDT interpreter")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "0":
        raise RuntimeError("Wind selected-action evaluation requires CUDA_VISIBLE_DEVICES=0")
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "NVIDIA RTX 6000 Ada Generation"
    ):
        raise RuntimeError("Authorized Wind CUDA device is unavailable")
    if maturation._gpu_uuid() != maturation.DEVICE_UUID:
        raise RuntimeError("CUDA_VISIBLE_DEVICES=0 did not resolve to the authorized physical GPU")
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir
        or PROJECT
        / "Case_WindFarm/diagnostics/generated/matured_action_table"
        / f"run2112_u{args.update_count:06d}"
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing action-table evidence: {output_dir}")
    started = time.monotonic()
    result = _evaluate(
        run_dir=run_dir,
        update_count=args.update_count,
        p_update_count=(args.update_count if args.p_update_count is None else args.p_update_count),
        output_dir=output_dir,
        layout_limit=args.layout_count,
        repeat_layout_limit=args.query_repeat_layout_count,
        selection_seed=args.selection_seed,
    )
    result["active_wall_seconds"] = time.monotonic() - started
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
