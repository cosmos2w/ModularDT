"""Fit and audit a layout-grouped WindFarm packet-action risk selector.

The native action table is produced at one frozen forward checkpoint by
``evaluate_matured_action_table.py``. Only planner features enter either fit;
measured role errors are used for train labels and post-selection audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from honf_forward_core.interface_fields.action_aware_frontier import incumbent_role_log_limit
from honf_forward_core.interface_fields.action_risk_fit import (
    ActionEvidenceRow,
    crossfit_action_risk,
    evaluate_action_policy,
    fit_action_risk_head,
    fit_ridge_action_baseline,
)

ACTION_KEYS = ("root", "two_packet", "four_packet", "full_access")
ACTION_PATHS = {
    "root": ("",),
    "two_packet": ("L", "R"),
    "four_packet": ("LL", "LR", "RL", "RR"),
}
RELATIVE_ALLOWANCE = 0.10
MAX_FIT_UPDATES = 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _float_tensor(values: Any) -> torch.Tensor:
    result = torch.as_tensor(np.asarray(values).copy(), dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Action table contains nonfinite features or role errors.")
    return result


def _checked_features(
    npz: Mapping[str, np.ndarray], row: Mapping[str, Any], digests: Mapping[str, str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = []
    for name in ("packet_rows", "budget_vector", "receiver_role_features"):
        key = str(row[f"{name}_npz_key"])
        if key not in npz:
            raise ValueError(f"Action feature {key!r} is absent from the bound NPZ.")
        value = np.asarray(npz[key])
        expected = str(digests[key])
        actual = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Action feature {key!r} fails its SHA-256 binding.")
        tensors.append(_float_tensor(value))
    packet, budget, roles = tensors
    if packet.ndim != 2 or budget.ndim != 1 or roles.ndim != 2:
        raise ValueError("Packet, budget, and receiver-role feature ranks changed.")
    return packet, budget, roles


def _role_tensor(values: Mapping[str, Any] | Sequence[float], roles: Sequence[str]) -> torch.Tensor:
    if isinstance(values, Mapping):
        if set(values) != set(roles):
            raise ValueError("Physical role error map differs from the bound Wind role order.")
        values = [values[role] for role in roles]
    return _float_tensor(values)


def _valid_action_paths(action_key: str, row: Mapping[str, Any]) -> tuple[str, ...]:
    requested = tuple(str(value) for value in row.get("requested_cut_paths", ()))
    realized_raw = row.get("realized_cut_paths")
    if requested != ACTION_PATHS[action_key] or not isinstance(realized_raw, (list, tuple)):
        raise ValueError(f"Action {action_key!r} lacks its requested and realized path identity.")
    realized = tuple(str(value) for value in realized_raw)
    if not realized or realized != tuple(sorted(set(realized))) or any(
        len(path) > 3 or set(path) - {"L", "R"} for path in realized
    ):
        raise ValueError("A sparse action has an invalid realized cut-path family.")
    if any(
        left != right and (left.startswith(right) or right.startswith(left))
        for index, left in enumerate(realized)
        for right in realized[index + 1:]
    ) or not math.isclose(
        sum(2.0 ** (-len(path)) for path in realized), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Realized cut paths do not identify one complete native cut.")
    if any(not any(path.startswith(ancestor) for ancestor in realized) for path in requested):
        raise ValueError("Realized paths cannot be obtained by resolving the requested action.")
    if any(not any(path.startswith(realized_path) for path in requested) for realized_path in realized):
        raise ValueError("Realized paths contain a node outside the requested action family.")
    return realized


def _selected_checkpoint_binding(
    payload: Mapping[str, Any], *, selected_update: int,
    expected_sha256: str, checkpoint_path: Path,
) -> tuple[str, Mapping[str, Any], str]:
    """Read legacy scheduled-review or explicit v2 scheduled/durability identity."""
    table_format = payload.get("format_version")
    if table_format == 1:
        source = "append_only_scheduled_review"
        record = payload.get("g_review_record")
        record_sha256 = payload.get("g_review_record_line_sha256")
        review_record, review_line_sha = record, record_sha256
    elif table_format == 2:
        source = payload.get("g_checkpoint_binding_source")
        record = payload.get("g_checkpoint_binding_record")
        record_sha256 = payload.get("g_checkpoint_binding_record_sha256")
        review_record = payload.get("g_checkpoint_review_record")
        review_line_sha = payload.get("g_checkpoint_review_record_line_sha256")
    else:
        raise ValueError("Unknown Wind action table format.")
    if (
        source not in {"append_only_scheduled_review", "current_durability_checkpoint"}
        or not isinstance(record, Mapping)
        or not _is_sha256(record_sha256)
        or record.get("arm") != "g_packet"
        or not isinstance(record.get("update_count"), int)
        or isinstance(record.get("update_count"), bool)
        or record.get("update_count") != selected_update
        or record.get("checkpoint_sha256") != expected_sha256
        or not isinstance(record.get("checkpoint"), str)
        or Path(record["checkpoint"]).resolve() != Path(checkpoint_path).resolve()
        or source == "current_durability_checkpoint"
        and (
            table_format != 2
            or record.get("checkpoint_kind") != "durability_only"
            or record.get("run_id") != payload.get("run_id")
            or review_record is not None
            or review_line_sha is not None
        )
        or source == "append_only_scheduled_review"
        and (
            not isinstance(review_record, Mapping)
            or review_record != record
            or review_line_sha != record_sha256
        )
    ):
        raise ValueError("Action table does not bind an explicit G checkpoint source, record, and identity.")
    return str(source), record, str(record_sha256)


def _validated_training_exposure(
    action_key: str, row: Mapping[str, Any], *, row_id: int,
    split: str,
    realized_paths: Sequence[str], nonredundant_k: int,
    exact_work: float, full_work: float,
    action_exposure_by_key: Mapping[str, Any],
) -> tuple[bool, Mapping[str, Any]]:
    row_lineage = row.get("training_exposure")
    exposure = action_exposure_by_key.get(action_key)
    if not isinstance(row_lineage, Mapping) or not isinstance(exposure, Mapping):
        raise TypeError(f"Sparse action {action_key!r} lacks nested training exposure evidence.")
    required = exposure.get("required_complete_passes")
    pass_ids = exposure.get("completed_primary_pass_ids")
    invalid_path_count = exposure.get("invalid_realized_cut_path_update_count")
    invalid_k_count = exposure.get("invalid_realized_nonredundant_k_update_count")
    invalid_execution_count = exposure.get("invalid_realized_execution_update_count")
    if (
        required != 2
        or not isinstance(pass_ids, list)
        or any(not isinstance(value, int) for value in pass_ids)
        or len(pass_ids) != len(set(pass_ids))
        or int(exposure.get("completed_primary_action_passes", -1)) != len(pass_ids)
        or bool(exposure.get("trained_action")) != (len(pass_ids) >= required)
        or not all(isinstance(value, int) and value >= 0 for value in (
            invalid_path_count, invalid_k_count, invalid_execution_count
        ))
        or invalid_execution_count < max(invalid_path_count, invalid_k_count)
        or invalid_execution_count > invalid_path_count + invalid_k_count
    ):
        raise ValueError(f"Sparse action {action_key!r} has inconsistent complete-pass lineage.")
    pass_audits = exposure.get("realized_cut_paths_by_primary_pass")
    if not isinstance(pass_audits, Mapping):
        raise TypeError(f"Sparse action {action_key!r} lacks realized-path pass audits.")
    for pass_id in pass_ids:
        audit = pass_audits.get(str(pass_id))
        if (
            not isinstance(audit, Mapping)
            or audit.get("complete_primary_pass") is not True
            or int(audit.get("distinct_row_count", -1)) != int(exposure.get("training_row_count", -2))
            or int(audit.get("expected_row_count", -1)) != int(exposure.get("training_row_count", -2))
        ):
            raise ValueError(f"Sparse action {action_key!r} completed pass {pass_id} is not row-complete.")
    row_path_map = exposure.get("resolved_path_exposure_by_row")
    if not isinstance(row_path_map, Mapping):
        raise TypeError(f"Sparse action {action_key!r} lacks per-row realized-path exposure.")
    if (
        row.get("trained_action_after_two_complete_passes") != exposure.get("trained_action")
        or row_lineage.get("action_key") != action_key
        or row_lineage.get("completed_primary_pass_ids") != pass_ids
        or row_lineage.get("completed_primary_action_passes") != len(pass_ids)
        or row_lineage.get("trained_action") != exposure.get("trained_action")
        or row_lineage.get("training_row_count") != exposure.get("training_row_count")
        or row_lineage.get("invalid_realized_cut_path_update_count")
        != exposure.get("invalid_realized_cut_path_update_count")
        or row_lineage.get("invalid_realized_nonredundant_k_update_count")
        != exposure.get("invalid_realized_nonredundant_k_update_count")
        or row_lineage.get("invalid_realized_execution_update_count")
        != exposure.get("invalid_realized_execution_update_count")
        or row_lineage.get("resolved_path_exposure_for_case")
        != row_path_map.get(str(int(row_id)), [])
    ):
        raise ValueError(f"Sparse action {action_key!r} row lineage differs from its table-level exposure audit.")
    row_pass_ids: set[int] = set()
    same_path_pass_ids: set[int] = set()
    same_path_k_pass_ids: set[int] = set()
    for family in row_lineage.get("resolved_path_exposure_for_case", []):
        if not isinstance(family, Mapping):
            continue
        observations = family.get("primary_pass_observations", [])
        if not isinstance(observations, list):
            continue
        for observation in observations:
            if not isinstance(observation, Mapping) or observation.get("primary_pass") not in pass_ids:
                continue
            pass_id = int(observation["primary_pass"])
            row_pass_ids.add(pass_id)
            if list(family.get("realized_cut_paths", [])) == list(realized_paths):
                same_path_pass_ids.add(pass_id)
                if observation.get("realized_nonredundant_k") == int(nonredundant_k):
                    same_path_k_pass_ids.add(pass_id)
    if (
        row_lineage.get("same_realized_path_primary_pass_ids") != sorted(same_path_pass_ids)
        or row_lineage.get("same_realized_path_and_k_primary_pass_ids") != sorted(same_path_k_pass_ids)
    ):
        raise ValueError(f"Sparse action {action_key!r} overstates its realized path/K overlap audit.")
    path_key = json.dumps(list(realized_paths), separators=(",", ":"))
    path_k_pass_ids = set()
    for pass_id in pass_ids:
        audited_count = int(pass_audits[str(pass_id)].get("resolved_path_families", {})
                            .get(path_key, {}).get("realized_nonredundant_k_counts", {})
                            .get(str(nonredundant_k), 0))
        observed_count = sum(
            observation.get("primary_pass") == pass_id
            and observation.get("realized_nonredundant_k") == nonredundant_k
            for families in row_path_map.values()
            for family in families
            if list(family.get("realized_cut_paths", [])) == list(realized_paths)
            for observation in family.get("primary_pass_observations", [])
        )
        if audited_count != observed_count:
            raise ValueError(f"Sparse action {action_key!r} path/K pass audit differs from its training rows")
        if audited_count > 0:
            path_k_pass_ids.add(int(pass_id))
    if split == "train_fit":
        # A seen native row needs its own resolved cut and K in both passes.
        exposed_pass_ids = same_path_k_pass_ids & path_k_pass_ids
    elif split in {"dev", "held_family_audit"}:
        # Held layouts have no training row ID; use exact cut/K exposure
        # transferred from complete training passes, not a fictitious visit.
        if row_pass_ids:
            raise ValueError("Held Wind case unexpectedly has training-row exposure")
        exposed_pass_ids = path_k_pass_ids
    else:
        raise ValueError(f"Unknown Wind action-table split: {split}")
    strict_work_saving = exact_work < full_work - max(1.0e-9, 1.0e-12 * full_work)
    qualified = bool(
        exposure.get("trained_action") is True
        and len(exposed_pass_ids & set(pass_ids)) >= required
        and strict_work_saving
    )
    if bool(row.get("trained_action_after_two_complete_passes")) != bool(exposure.get("trained_action")):
        raise ValueError(f"Sparse action {action_key!r} disagrees with its nested pass-exposure flag.")
    if bool(row.get("trained_sparse")) != qualified:
        raise ValueError(
            f"Sparse action {action_key!r} trained_sparse flag disagrees with complete-pass exposure, valid row execution, or strict work saving."
        )
    return qualified, exposure


def _load_action_table(
    table_path: Path, feature_path: Path, mask_path: Path, checkpoint_path: Path,
) -> tuple[list[ActionEvidenceRow], dict[str, Any]]:
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    table_format = payload.get("format_version")
    if table_format not in (1, 2):
        raise ValueError("Unknown Wind action table format.")
    expected_forward_sha = str(payload["forward_checkpoint_sha256"])
    selected_update = payload.get("selected_update_count")
    g_binding_source, g_binding_record, g_binding_record_sha = _selected_checkpoint_binding(
        payload, selected_update=selected_update, expected_sha256=expected_forward_sha,
        checkpoint_path=checkpoint_path,
    )
    exposure_lineage = payload.get("g_action_exposure_lineage")
    run_manifest_path = payload.get("run_manifest_path")
    if (
        payload.get("run_id") != "2112"
        or not isinstance(selected_update, int) or isinstance(selected_update, bool) or selected_update <= 100
        or not isinstance(exposure_lineage, Mapping)
        or not isinstance(exposure_lineage.get("selected_update_count"), int)
        or isinstance(exposure_lineage.get("selected_update_count"), bool)
        or exposure_lineage.get("selected_update_count") != selected_update
        or not _is_sha256(exposure_lineage.get("update_ledger_sha256"))
        or not isinstance(exposure_lineage.get("update_ledger_snapshot_bytes"), int)
        or exposure_lineage.get("update_ledger_snapshot_bytes") < 0
        or not isinstance(exposure_lineage.get("update_ledger_snapshot_record_count"), int)
        or exposure_lineage.get("update_ledger_snapshot_record_count") < 0
        or not isinstance(run_manifest_path, str) or not run_manifest_path
        or Path(str(exposure_lineage.get("update_ledger_path", ""))).resolve()
        != Path(run_manifest_path).resolve().parent / "arms" / "g_packet" / "updates.jsonl"
    ):
        raise ValueError("Action table does not bind the selected G checkpoint binding, update, and exposure ledger snapshot.")
    if _sha256(checkpoint_path) != expected_forward_sha:
        raise ValueError("Selected G checkpoint differs from the measured action table.")
    if _sha256(feature_path) != payload["feature_npz_sha256"]:
        raise ValueError("Planner feature archive differs from the measured action table.")
    if _sha256(mask_path) != payload["mask_npz_sha256"]:
        raise ValueError("Realized permission-mask archive differs from the measured action table.")
    exposure_by_action = payload.get("action_exposure_by_key")
    if not isinstance(exposure_by_action, Mapping):
        raise TypeError("Wind action table lacks its action-level realized exposure audit.")
    observed_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in payload["rows"]:
        observed_splits[str(row["split"])].add(str(row["family_key"]))
    split_families = {key: tuple(sorted(value)) for key, value in sorted(observed_splits.items())}
    if len(split_families.get("train_fit", ())) < 2:
        raise ValueError("Layout grouped risk fitting needs at least two train layouts.")
    split_sets = [set(values) for values in split_families.values()]
    if any(split_sets[i] & split_sets[j] for i in range(len(split_sets)) for j in range(i + 1, len(split_sets))):
        raise ValueError("Physical layout families overlap across splits.")
    roles = tuple(payload["roles"])
    if len(roles) != 5:
        raise ValueError("Wind role order must have five physical roles.")
    floor = _role_tensor(payload["numerical_floor_role_mps"], roles)
    allowance = _role_tensor(payload["absolute_allowance_role_mps"], roles)
    if floor.shape != (5,) or not bool((floor > 0).all()) or allowance.shape != (5,) or bool((allowance < 0).any()):
        raise ValueError("Role floor and physical allowance must be finite and aligned.")
    if (
        bool((allowance != 0).any())
        or payload.get("absolute_allowance_calibration", {}).get("calibration_kind")
        != "zero extra physical-risk allowance for paired same-query comparison"
    ):
        raise ValueError(
            "Paired same-query Wind adequacy cannot add query-sampling variability as a physical-risk allowance."
        )
    case_metadata: dict[str, dict[str, Any]] = {}
    rows: list[ActionEvidenceRow] = []
    with np.load(feature_path, allow_pickle=False) as npz:
        for action in payload["rows"]:
            case_key = str(action["case_key"])
            family_key = str(action["family_key"])
            split = str(action["split"])
            if split not in split_families or family_key not in split_families[split]:
                raise ValueError(f"Case {case_key!r} has an unbound split or layout family.")
            if action["forward_checkpoint_sha256"] != expected_forward_sha:
                raise ValueError(f"Case {case_key!r} uses stale G physical or scorer weights.")
            identity = {"family_key": family_key, "split": split,
                        "query_panel": str(action["query_panel"]),
                        "selector_primary_fit_row": action.get("selector_primary_fit_row"),
                        "row_id": int(action["row_id"]), "layout_id": int(action["layout_id"]),
                        "module_count": int(action["module_count"]),
                        "query_repeat_overlap_count": action.get("query_repeat_overlap_count"),
                        "query_repeat_disjoint_from_fixed": action.get("query_repeat_disjoint_from_fixed")}
            if (
                identity["query_panel"] not in {"fixed", "query_repeat"}
                or not isinstance(identity["selector_primary_fit_row"], bool)
                or identity["selector_primary_fit_row"] != (identity["query_panel"] == "fixed")
            ):
                raise ValueError(f"Case {case_key!r} does not explicitly separate primary and repeat panels.")
            if case_key in case_metadata and case_metadata[case_key] != identity:
                raise ValueError(f"Case {case_key!r} changes identity across actions.")
            case_metadata[case_key] = identity
            action_key = str(action["action_key"])
            if action_key not in ACTION_KEYS:
                raise ValueError(f"Unknown Wind action {action_key!r}.")
            packet, budget, descriptors = _checked_features(
                npz, action, payload["feature_sha256_by_npz_key"]
            )
            incumbent = _role_tensor(action["incumbent_role_error"], roles)
            candidate = _role_tensor(action["candidate_role_error"], roles)
            if any(value.shape != (5,) or bool((value < 0).any()) for value in (candidate, incumbent)):
                raise ValueError("Candidate or W-full reference physical role errors are invalid.")
            is_full = action_key == "full_access"
            if bool(action["full_access"]) != is_full:
                raise ValueError("Only the explicit full-access row may bypass packet selection.")
            count = action["nonredundant_k"]
            if is_full and count is not None:
                raise ValueError("Full access must have undefined external packet K.")
            if not is_full and (not isinstance(count, int) or count < 1):
                raise ValueError("Sparse packet action needs measured nonredundant K.")
            exact_work = float(action["exact_work"])
            full_work = float(action["full_work"])
            if not math.isfinite(exact_work) or not math.isfinite(full_work) or full_work <= 0 or not 0 <= exact_work <= full_work * (1 + 1e-6):
                raise ValueError("Canonical action and full work must be raw finite work units.")
            if is_full:
                if bool(action.get("trained_sparse")) or action.get("realized_cut_paths") != []:
                    raise ValueError("Explicit full access must remain outside sparse exposure and cut-path K.")
                trained_sparse = False
            else:
                realized_paths = _valid_action_paths(action_key, action)
                trained_sparse, _exposure = _validated_training_exposure(
                    action_key, action, row_id=identity["row_id"],
                    split=split,
                    realized_paths=realized_paths, nonredundant_k=int(count),
                    exact_work=exact_work, full_work=full_work,
                    action_exposure_by_key=exposure_by_action,
                )
            repeat_overlap = action.get("query_repeat_overlap_count")
            repeat_disjoint = action.get("query_repeat_disjoint_from_fixed")
            if identity["query_panel"] == "fixed":
                if repeat_overlap is not None or repeat_disjoint is not None:
                    raise ValueError("Fixed primary-fit rows cannot carry repeat-overlap measurements.")
            elif (
                not isinstance(repeat_overlap, int) or repeat_overlap < 0
                or not isinstance(repeat_disjoint, bool)
                or repeat_disjoint != (repeat_overlap == 0)
            ):
                raise ValueError("Query-repeat overlap flags must match the measured sample-index intersection.")
            rows.append(ActionEvidenceRow(
                family_key=family_key,
                case_key=case_key,
                action_key=action_key,
                forward_sha256=expected_forward_sha,
                packet_rows=packet,
                budget_vector=budget,
                receiver_role_features=descriptors,
                candidate_role_error=candidate,
                incumbent_role_error=incumbent,
                numerical_floor=floor,
                exact_work=exact_work,
                nonredundant_k=1 if is_full else count,
                trained_sparse=trained_sparse,
                full_access=is_full,
            ))
    actual_actions: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.action_key in actual_actions[row.case_key]:
            raise ValueError(f"Repeated action {row.action_key!r} in {row.case_key!r}.")
        actual_actions[row.case_key].add(row.action_key)
    if any(actions != set(ACTION_KEYS) for actions in actual_actions.values()):
        raise ValueError("Every Wind case must have one exact four-action frontier.")
    return rows, {
        "source": payload,
        "case_metadata": case_metadata,
        "split_families": split_families,
        "roles": roles,
        "allowance": allowance,
        "checkpoint_sha256": expected_forward_sha,
        "checkpoint_binding_source": g_binding_source,
        "checkpoint_binding_record": g_binding_record,
        "checkpoint_binding_record_sha256": g_binding_record_sha,
        "action_exposure_by_key": exposure_by_action,
    }


def _train_fixed_limits(
    rows: Sequence[ActionEvidenceRow], train_families: Sequence[str],
    case_metadata: Mapping[str, Mapping[str, Any]], allowance: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    calibration = []
    for family in sorted(train_families):
        primary = [
            row for row in rows if row.family_key == family and row.full_access
            and case_metadata[row.case_key]["query_panel"] == "fixed"
        ]
        if not primary:
            raise ValueError(f"Train layout {family!r} needs a fixed full-access panel.")
        incumbent_median = torch.stack([row.incumbent_role_error.double() for row in primary]).median(dim=0).values
        floor = primary[0].numerical_floor.double()
        if any(not torch.equal(row.numerical_floor.double(), floor) for row in primary[1:]):
            raise ValueError(f"Train layout {family!r} changes the role numerical floor across directions.")
        limit = incumbent_role_log_limit(
            incumbent_median, floor,
            allowance.double(), relative_allowance=RELATIVE_ALLOWANCE,
        )
        calibration.append({"family_key": family, "case_keys": [row.case_key for row in primary],
                            "fixed_direction_count": len(primary),
                            "incumbent_role_rmse_mps_median": incumbent_median.tolist(),
                            "role_log_limit": limit.tolist()})
    limits = torch.tensor([record["role_log_limit"] for record in calibration]).amin(dim=0)
    return limits, calibration


def _query_repeat_fit_audit(case_metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    repeats = [
        (case_key, item) for case_key, item in case_metadata.items()
        if item["query_panel"] == "query_repeat"
    ]
    disjoint = sum(item["query_repeat_overlap_count"] == 0 for _, item in repeats)
    overlaps = [int(item["query_repeat_overlap_count"]) for _, item in repeats]
    return {
        "query_repeat_case_count": len(repeats),
        "disjoint_repeat_case_count": disjoint,
        "overlapping_repeat_case_count": len(repeats) - disjoint,
        "total_intersecting_query_indices": sum(overlaps),
        "maximum_intersection_count": max(overlaps) if overlaps else None,
        "used_for_primary_selector_fit": False,
        "used_for_primary_threshold_calibration": False,
        "used_for_primary_split_evaluation": False,
        "cases": [{
            "case_key": case_key,
            "row_id": int(item["row_id"]),
            "overlap_count": int(item["query_repeat_overlap_count"]),
            "disjoint": bool(item["query_repeat_disjoint_from_fixed"]),
        } for case_key, item in repeats],
    }


def _predictions(
    rows: Sequence[ActionEvidenceRow], crossfit: Any, neural_fit: Any,
    ridge_fit: Any, train_families: set[str],
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[tuple[str, str], torch.Tensor]]:
    train_neural = dict(zip(crossfit.row_keys, crossfit.neural, strict=True))
    train_ridge = dict(zip(crossfit.row_keys, crossfit.ridge, strict=True))
    neural: dict[tuple[str, str], torch.Tensor] = {}
    ridge: dict[tuple[str, str], torch.Tensor] = {}
    with torch.no_grad():
        for row in rows:
            if not row.trained_sparse and not row.full_access:
                continue
            key = (row.case_key, row.action_key)
            if row.family_key in train_families:
                neural[key] = train_neural[key]
                ridge[key] = train_ridge[key]
            else:
                neural[key] = neural_fit.model(
                    row.packet_rows, row.budget_vector, row.receiver_role_features,
                    nonredundant_k=row.nonredundant_k,
                ).detach().cpu()
                ridge[key] = ridge_fit.predict(row).detach().cpu()
    return neural, ridge


def _final_model_predictions(
    rows: Sequence[ActionEvidenceRow], neural_model: Any, ridge_fit: Any,
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[tuple[str, str], torch.Tensor]]:
    """Score rows with the final fit; used only for the separate repeat diagnostic."""
    neural: dict[tuple[str, str], torch.Tensor] = {}
    ridge: dict[tuple[str, str], torch.Tensor] = {}
    with torch.no_grad():
        for row in rows:
            if not row.trained_sparse and not row.full_access:
                continue
            key = (row.case_key, row.action_key)
            neural[key] = neural_model(
                row.packet_rows, row.budget_vector, row.receiver_role_features,
                nonredundant_k=row.nonredundant_k,
            ).detach().cpu()
            ridge[key] = ridge_fit.predict(row).detach().cpu()
    return neural, ridge


def _paired_repeat_stability(
    primary_results: Sequence[Any], repeat_results: Sequence[Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare final-fit selection on fixed/repeat panels of the same native row."""
    primary_by_row: dict[tuple[str, int], Any] = {}
    repeat_by_row: dict[tuple[str, int], Any] = {}
    for destination, results, panel in (
        (primary_by_row, primary_results, "fixed"),
        (repeat_by_row, repeat_results, "query_repeat"),
    ):
        for result in results:
            identity = metadata["case_metadata"][result.case_key]
            key = (str(result.family_key), int(identity["row_id"]))
            if identity["query_panel"] != panel:
                raise ValueError("Paired query-repeat stability received a row from the wrong panel.")
            if key in destination:
                raise ValueError(f"Multiple {panel} cases map to the same native row {key!r}.")
            destination[key] = result

    pairs = []
    for key in sorted(set(primary_by_row) & set(repeat_by_row)):
        fixed = primary_by_row[key]
        repeat = repeat_by_row[key]
        repeat_identity = metadata["case_metadata"][repeat.case_key]

        def jaccard(left: Sequence[str], right: Sequence[str]) -> float | None:
            union = set(left) | set(right)
            return len(set(left) & set(right)) / len(union) if union else None

        oracle_agrees = (
            fixed.measured_oracle_action_key == repeat.measured_oracle_action_key
            if fixed.measured_oracle_action_key is not None
            and repeat.measured_oracle_action_key is not None else None
        )
        pairs.append({
            "family_key": key[0],
            "row_id": key[1],
            "fixed_case_key": fixed.case_key,
            "query_repeat_case_key": repeat.case_key,
            "query_repeat_overlap_count": int(repeat_identity["query_repeat_overlap_count"]),
            "query_repeat_disjoint_from_fixed": bool(repeat_identity["query_repeat_disjoint_from_fixed"]),
            "fixed_selected_action_key": fixed.selected_action_key,
            "query_repeat_selected_action_key": repeat.selected_action_key,
            "selected_action_agrees": fixed.selected_action_key == repeat.selected_action_key,
            "fixed_measured_oracle_action_key": fixed.measured_oracle_action_key,
            "query_repeat_measured_oracle_action_key": repeat.measured_oracle_action_key,
            "measured_oracle_action_agrees": oracle_agrees,
            "measured_adequate_sparse_jaccard": jaccard(
                fixed.measured_adequate_sparse, repeat.measured_adequate_sparse
            ),
            "predicted_safe_sparse_jaccard": jaccard(
                fixed.predicted_safe_sparse, repeat.predicted_safe_sparse
            ),
        })
    oracle_pairs = [pair for pair in pairs if pair["measured_oracle_action_agrees"] is not None]
    measured_jaccards = [
        pair["measured_adequate_sparse_jaccard"] for pair in pairs
        if pair["measured_adequate_sparse_jaccard"] is not None
    ]
    predicted_jaccards = [
        pair["predicted_safe_sparse_jaccard"] for pair in pairs
        if pair["predicted_safe_sparse_jaccard"] is not None
    ]
    return {
        "paired_case_count": len(pairs),
        "unpaired_query_repeat_case_count": len(set(repeat_by_row) - set(primary_by_row)),
        "selected_action_agreement_count": sum(bool(pair["selected_action_agrees"]) for pair in pairs),
        "selected_action_agreement_rate": (
            sum(bool(pair["selected_action_agrees"]) for pair in pairs) / len(pairs) if pairs else None
        ),
        "measured_oracle_comparable_pair_count": len(oracle_pairs),
        "measured_oracle_action_agreement_rate": (
            sum(bool(pair["measured_oracle_action_agrees"]) for pair in oracle_pairs) / len(oracle_pairs)
            if oracle_pairs else None
        ),
        "mean_measured_adequate_sparse_jaccard": (
            float(np.mean(measured_jaccards)) if measured_jaccards else None
        ),
        "mean_predicted_safe_sparse_jaccard": (
            float(np.mean(predicted_jaccards)) if predicted_jaccards else None
        ),
        "per_pair": pairs,
    }


def _query_repeat_diagnostic(
    primary_rows: Sequence[ActionEvidenceRow], repeat_rows: Sequence[ActionEvidenceRow],
    metadata: Mapping[str, Any], *, neural_model: Any, ridge_fit: Any,
    fixed_limits: torch.Tensor, neural_margin: torch.Tensor, ridge_margin: torch.Tensor,
    repeat_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate query repeats after fitting, without feeding repeat labels into fit or calibration."""
    common = {
        "evaluation_scope": (
            "separate query-repeat stability diagnostic; repeat labels are post-fit audit only and do not "
            "enter primary calibration, fitting, or headline split results"
        ),
        "used_for_primary_selector_fit": False,
        "used_for_primary_threshold_calibration": False,
        "used_for_primary_split_evaluation": False,
        "overlap_cases": list(repeat_audit["cases"]),
    }
    if not repeat_rows:
        return {"status": "no_query_repeat_rows", **common}

    primary_neural, primary_ridge = _final_model_predictions(
        primary_rows, neural_model, ridge_fit
    )
    repeat_neural, repeat_ridge = _final_model_predictions(
        repeat_rows, neural_model, ridge_fit
    )
    primary_allowance = {row.case_key: metadata["allowance"] for row in primary_rows}
    repeat_allowance = {row.case_key: metadata["allowance"] for row in repeat_rows}
    evaluations = {}
    for name, primary_predictions, repeat_predictions, margin in (
        ("neural", primary_neural, repeat_neural, neural_margin),
        ("ridge", primary_ridge, repeat_ridge, ridge_margin),
    ):
        primary_results = evaluate_action_policy(
            primary_rows, primary_predictions, current_forward_sha256=metadata["checkpoint_sha256"],
            fixed_role_log_limits=fixed_limits, absolute_allowance_by_case=primary_allowance,
            empirical_margin=margin, relative_allowance=RELATIVE_ALLOWANCE,
        )
        repeat_results = evaluate_action_policy(
            repeat_rows, repeat_predictions, current_forward_sha256=metadata["checkpoint_sha256"],
            fixed_role_log_limits=fixed_limits, absolute_allowance_by_case=repeat_allowance,
            empirical_margin=margin, relative_allowance=RELATIVE_ALLOWANCE,
        )
        evaluations[name] = {
            **_policy_summary(repeat_results, repeat_rows, metadata["case_metadata"]),
            "paired_fixed_repeat_stability": _paired_repeat_stability(
                primary_results, repeat_results, metadata
            ),
        }
    repeat_predictions = [
        {
            "case_key": row.case_key,
            "family_key": row.family_key,
            "action_key": row.action_key,
            "trained_sparse": row.trained_sparse,
            "full_access": row.full_access,
            "nonredundant_k": None if row.full_access else row.nonredundant_k,
            "measured_signed_log_role_risk": row.target_log_risk.tolist(),
            "neural_predicted_log_role_risk": (
                repeat_neural[(row.case_key, row.action_key)].tolist()
                if (row.case_key, row.action_key) in repeat_neural else None
            ),
            "ridge_predicted_log_role_risk": (
                repeat_ridge[(row.case_key, row.action_key)].tolist()
                if (row.case_key, row.action_key) in repeat_ridge else None
            ),
        }
        for row in repeat_rows
    ]
    return {
        "status": "evaluated",
        **common,
        "prediction_source": "final neural and ridge fits; repeat families did not update either fit",
        "threshold_source": "fixed primary train_fit full-access calibration and primary-only crossfit residual margin",
        "neural": evaluations["neural"],
        "ridge": evaluations["ridge"],
        "action_predictions": repeat_predictions,
    }


def _ranking(
    rows: Sequence[ActionEvidenceRow], prediction: Mapping[tuple[str, str], torch.Tensor],
) -> dict[str, Any]:
    grouped: dict[str, list[ActionEvidenceRow]] = defaultdict(list)
    for row in rows:
        if row.trained_sparse:
            grouped[row.case_key].append(row)
    concordant = discordant = tied = 0
    for case_rows in grouped.values():
        for left_index, left in enumerate(case_rows):
            for right in case_rows[left_index + 1:]:
                actual = float(left.target_log_risk.max() - right.target_log_risk.max())
                if abs(actual) <= 1e-3:
                    continue
                predicted = float(prediction[(left.case_key, left.action_key)].max()
                                  - prediction[(right.case_key, right.action_key)].max())
                if abs(predicted) <= 1e-3:
                    tied += 1
                elif predicted * actual > 0:
                    concordant += 1
                else:
                    discordant += 1
    return {"comparable_action_pairs": concordant + discordant + tied,
            "concordant": concordant, "discordant": discordant, "tied": tied,
            "accuracy_excluding_ties": concordant / (concordant + discordant)
            if concordant + discordant else None}


def _policy_summary(
    results: Sequence[Any], rows: Sequence[ActionEvidenceRow],
    case_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    full_work = {row.case_key: row.exact_work for row in rows if row.full_access}
    oracle = [row for row in results if row.measured_oracle_action_key is not None]
    selected_false_safe = [
        row for row in results
        if not row.unsupported_at_budget and not row.selected_supported_sparse
    ]
    full_fallback_with_adequate_sparse = [
        row for row in results
        if row.unsupported_at_budget and row.measured_oracle_action_key is not None
    ]
    by_m: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for row in results:
        if row.selected_nonredundant_k is not None:
            by_m[int(case_metadata[row.case_key]["module_count"])][int(row.selected_nonredundant_k)] += 1
    return {
        "case_count": len(results),
        "measured_oracle_available_count": len(oracle),
        "supported_sparse_deployment_count": sum(row.selected_supported_sparse for row in results),
        "explicit_full_fallback_count": sum(row.unsupported_at_budget for row in results),
        "false_safe_choice_count": len(selected_false_safe),
        "false_reject_choice_count": len(full_fallback_with_adequate_sparse),
        "candidate_false_safe_count": sum(len(row.false_safe_sparse) for row in results),
        "candidate_false_reject_count": sum(len(row.false_reject_sparse) for row in results),
        "mean_selected_exact_work": float(np.mean([row.selected_work for row in results])) if results else None,
        "mean_selected_work_over_same_case_full": float(np.mean([
            row.selected_work / full_work[row.case_key] for row in results
        ])) if results else None,
        "mean_oracle_sparse_work_when_available": float(np.mean([
            row.measured_oracle_sparse_work for row in oracle
        ])) if oracle else None,
        "selected_nonredundant_k_counts": dict(sorted(Counter(
            row.selected_nonredundant_k for row in results if row.selected_nonredundant_k is not None
        ).items())),
        "selected_nonredundant_k_by_module_count": {
            str(module_count): dict(sorted(counts.items())) for module_count, counts in sorted(by_m.items())
        },
        "selected_action_counts": dict(sorted(Counter(row.selected_action_key for row in results).items())),
        "per_case": [{
            "case_key": row.case_key,
            "family_key": row.family_key,
            "selected_action_key": row.selected_action_key,
            "measured_oracle_action_key": row.measured_oracle_action_key,
            "unsupported_at_budget": row.unsupported_at_budget,
            "selected_supported_sparse": row.selected_supported_sparse,
            "measured_adequate_sparse": list(row.measured_adequate_sparse),
            "predicted_safe_sparse": list(row.predicted_safe_sparse),
            "false_safe_sparse": list(row.false_safe_sparse),
            "false_reject_sparse": list(row.false_reject_sparse),
            "selected_false_safe": (
                not row.unsupported_at_budget and not row.selected_supported_sparse
            ),
            "full_fallback_with_adequate_sparse": (
                row.unsupported_at_budget and row.measured_oracle_action_key is not None
            ),
            "selected_exact_work": row.selected_work,
            "selected_nonredundant_k": row.selected_nonredundant_k,
            "module_count": case_metadata[row.case_key]["module_count"],
        } for row in results],
    }


def fit_and_evaluate(
    rows: Sequence[ActionEvidenceRow], metadata: Mapping[str, Any], *,
    updates: int = 120, seed: int = 2112,
) -> tuple[dict[str, Any], Any | None]:
    if not 1 <= updates <= MAX_FIT_UPDATES:
        raise ValueError("Risk fit update count exceeds the bounded diagnostic allocation.")
    forward_sha = metadata["checkpoint_sha256"]
    splits = metadata["split_families"]
    train_families = tuple(splits["train_fit"])
    train_set = set(train_families)
    allowance = metadata["allowance"].double()
    primary_rows = [
        row for row in rows if metadata["case_metadata"][row.case_key]["selector_primary_fit_row"]
    ]
    repeat_rows = [
        row for row in rows if not metadata["case_metadata"][row.case_key]["selector_primary_fit_row"]
    ]
    if not primary_rows:
        raise ValueError("The primary selector fit requires fixed query-panel action rows.")
    repeat_audit = _query_repeat_fit_audit(metadata["case_metadata"])
    limits, calibration = _train_fixed_limits(
        primary_rows, train_families, metadata["case_metadata"], allowance
    )
    full_work = {row.case_key: row.exact_work for row in rows if row.full_access}
    result: dict[str, Any] = {
        "forward_checkpoint_sha256": forward_sha,
        "role_order": list(metadata["roles"]),
        "split_families": {key: list(value) for key, value in splits.items()},
        "absolute_allowance_role_mps": allowance.tolist(),
        "fixed_train_role_log_limits": limits.tolist(),
        "train_gate_calibration": calibration,
        "query_repeat_audit": repeat_audit,
        "primary_fit_action_row_count": len(primary_rows),
        "query_repeat_action_row_count_excluded_from_primary_fit": len(repeat_rows),
        "relative_physical_allowance": RELATIVE_ALLOWANCE,
        "selector_fit": "unavailable_no_exposed_train_sparse_action",
        "optimizer_calls": 0,
        "split_results": {},
        "query_repeat_diagnostic": {
            "status": "not_evaluated_no_fitted_selector",
            "evaluation_scope": "repeat rows are post-fit diagnostics only",
            "used_for_primary_selector_fit": False,
            "used_for_primary_threshold_calibration": False,
            "used_for_primary_split_evaluation": False,
            "overlap_cases": list(repeat_audit["cases"]),
        },
        "action_prediction_rows": [{
            "case_key": row.case_key,
            "family_key": row.family_key,
            "split": metadata["case_metadata"][row.case_key]["split"],
            "query_panel": metadata["case_metadata"][row.case_key]["query_panel"],
            "selector_primary_fit_row": metadata["case_metadata"][row.case_key]["selector_primary_fit_row"],
            "module_count": metadata["case_metadata"][row.case_key]["module_count"],
            "action_key": row.action_key,
            "trained_sparse": row.trained_sparse,
            "full_access": row.full_access,
            "nonredundant_k": None if row.full_access else row.nonredundant_k,
            "exact_work": row.exact_work,
            "exact_work_over_full": row.exact_work / full_work[row.case_key],
            "candidate_role_rmse_mps": row.candidate_role_error.tolist(),
            "wfull_role_rmse_mps": row.incumbent_role_error.tolist(),
            "measured_signed_log_role_risk": row.target_log_risk.tolist(),
            "neural_predicted_log_role_risk": None,
            "ridge_predicted_log_role_risk": None,
        } for row in rows],
    }
    for split, families in splits.items():
        split_rows = [row for row in primary_rows if row.family_key in set(families)]
        result["split_results"][split] = {
            "case_count": len({row.case_key for row in split_rows}),
            "fixed_action": {action: {
                "measured_rows": sum(row.action_key == action for row in split_rows),
                "exposed_sparse_rows": sum(row.action_key == action and row.trained_sparse for row in split_rows),
                "adequate_sparse_rows": sum(
                    row.action_key == action and row.trained_sparse and bool((
                        row.candidate_role_error.double() <=
                        (1 + RELATIVE_ALLOWANCE) * row.incumbent_role_error.double() + allowance
                    ).all()) for row in split_rows
                ),
                "nonredundant_k_counts": dict(sorted(Counter(
                    row.nonredundant_k for row in split_rows if row.action_key == action and row.trained_sparse
                ).items())),
                "nonredundant_k_by_module_count": {
                    str(module_count): dict(sorted(Counter(
                        row.nonredundant_k for row in split_rows
                        if row.action_key == action and row.trained_sparse
                        and int(metadata["case_metadata"][row.case_key]["module_count"]) == module_count
                    ).items()))
                    for module_count in sorted({
                        int(metadata["case_metadata"][row.case_key]["module_count"])
                        for row in split_rows if row.action_key == action and row.trained_sparse
                    })
                },
            } for action in ACTION_KEYS[:-1]}
        }
    if not any(row.trained_sparse for row in primary_rows if row.family_key in train_set):
        return result, None
    folds = min(5, len(train_families))
    crossfit = crossfit_action_risk(
        primary_rows, current_forward_sha256=forward_sha, train_families=train_families,
        folds=folds, updates=updates, seed=seed,
    )
    neural_fit = fit_action_risk_head(
        primary_rows, current_forward_sha256=forward_sha, train_families=train_families,
        updates=updates, seed=seed + 1997,
    )
    ridge_fit = fit_ridge_action_baseline(
        primary_rows, current_forward_sha256=forward_sha, train_families=train_families,
    )
    neural_map, ridge_map = _predictions(primary_rows, crossfit, neural_fit, ridge_fit, train_set)
    for record in result["action_prediction_rows"]:
        key = (record["case_key"], record["action_key"])
        if key in neural_map:
            record["neural_predicted_log_role_risk"] = neural_map[key].tolist()
            record["ridge_predicted_log_role_risk"] = ridge_map[key].tolist()
    train_in_sample = {}
    with torch.no_grad():
        for row in primary_rows:
            if row.family_key in train_set and row.trained_sparse:
                train_in_sample[(row.case_key, row.action_key)] = neural_fit.model(
                    row.packet_rows, row.budget_vector, row.receiver_role_features,
                    nonredundant_k=row.nonredundant_k,
                ).detach().cpu()
    result.update({
        "selector_fit": "fitted",
        "optimizer_calls": updates * (folds + 1),
        "fit_updates_per_fold": updates,
        "crossfit_folds": folds,
        "neural_train_first_loss": neural_fit.first_loss,
        "neural_train_last_loss": neural_fit.last_loss,
        "ridge_train_feature_mean": ridge_fit.mean.tolist(),
        "ridge_train_feature_scale": ridge_fit.scale.tolist(),
        "ridge_train_coefficients": ridge_fit.coefficients.tolist(),
        "neural_empirical_upper_margin_by_role": crossfit.neural_upper_margin.tolist(),
        "ridge_empirical_upper_margin_by_role": crossfit.ridge_upper_margin.tolist(),
        "train_in_sample_ranking": _ranking(
            [row for row in primary_rows if row.family_key in train_set], train_in_sample
        ),
    })
    for split, families in splits.items():
        split_rows = [row for row in primary_rows if row.family_key in set(families)]
        case_keys = {row.case_key for row in split_rows}
        allowances = {case_key: allowance for case_key in case_keys}
        for name, predictions, margin in (
            ("neural", neural_map, crossfit.neural_upper_margin),
            ("ridge", ridge_map, crossfit.ridge_upper_margin),
        ):
            measured = evaluate_action_policy(
                split_rows, predictions, current_forward_sha256=forward_sha,
                fixed_role_log_limits=limits,
                absolute_allowance_by_case=allowances,
                empirical_margin=margin,
                relative_allowance=RELATIVE_ALLOWANCE,
            )
            result["split_results"][split][name] = {
                **_policy_summary(measured, split_rows, metadata["case_metadata"]),
                "ranking": _ranking(split_rows, predictions),
            }
    result["query_repeat_diagnostic"] = _query_repeat_diagnostic(
        primary_rows, repeat_rows, metadata, neural_model=neural_fit.model,
        ridge_fit=ridge_fit, fixed_limits=limits,
        neural_margin=crossfit.neural_upper_margin,
        ridge_margin=crossfit.ridge_upper_margin,
        repeat_audit=repeat_audit,
    )
    return result, neural_fit.model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-json", type=Path, required=True)
    parser.add_argument("--feature-npz", type=Path, required=True)
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument("--selected-g-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-updates", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2112)
    args = parser.parse_args()
    rows, metadata = _load_action_table(
        args.table_json.resolve(), args.feature_npz.resolve(), args.mask_npz.resolve(),
        args.selected_g_checkpoint.resolve()
    )
    result, model = fit_and_evaluate(rows, metadata, updates=args.fit_updates, seed=args.seed)
    result["input_identity"] = {
        "table_json": str(args.table_json.resolve()),
        "table_sha256": _sha256(args.table_json.resolve()),
        "feature_npz": str(args.feature_npz.resolve()),
        "feature_npz_sha256": _sha256(args.feature_npz.resolve()),
        "mask_npz": str(args.mask_npz.resolve()),
        "mask_npz_sha256": _sha256(args.mask_npz.resolve()),
        "selected_g_checkpoint": str(args.selected_g_checkpoint.resolve()),
        "selected_g_checkpoint_sha256": _sha256(args.selected_g_checkpoint.resolve()),
        "driver_sha256": _sha256(Path(__file__).resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if model is not None:
        model_path = args.output_dir / "wind_action_risk_head.pt"
        torch.save({"format_version": 1, "forward_checkpoint_sha256": metadata["checkpoint_sha256"],
                    "role_order": list(metadata["roles"]), "fit_updates": args.fit_updates,
                    "train_families": list(metadata["split_families"]["train_fit"]),
                    "state_dict": model.state_dict()}, model_path)
        result["risk_head_checkpoint"] = str(model_path.resolve())
        result["risk_head_checkpoint_sha256"] = _sha256(model_path)
    _write_json(args.output_dir / "wind_action_selector_results.json", result)
    print(json.dumps({"selector_fit": result["selector_fit"],
                      "optimizer_calls": result["optimizer_calls"],
                      "splits": {key: value["case_count"] for key, value in result["split_results"].items()},
                      "output": str(args.output_dir / "wind_action_selector_results.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
