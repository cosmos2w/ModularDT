#!/usr/bin/env python3
"""Matched fixed-panel review of a Run2112 WindFarm G/P endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "HONF_Proj"
for _path in (str(PROJECT / "src"), str(PROJECT / "Case_WindFarm" / "src"), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_active_packet_reuse as runner
import run_controlled_maturation as maturation
from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan
from honf_forward_core.interface_fields.budgeted_frontier import canonical_pair_catalog
from honf_runtime.compat import load_trusted_checkpoint
from windfarm.workflows.maturation import WIND_ACTION_PATHS, available_frontier_for_paths


RUN_ID = "2112"
DEVICE_UUID = maturation.DEVICE_UUID
ALL_MECHANISMS = ("MM", "ME", "EM", "QM", "QE")
ROLE_NAMES = tuple(runner.ROLE_NAMES)
PRIMARY_CAPACITY = {"QE": 0.95, "MM": 0.90}
FROZEN_090_PROBE = {"QE": 0.90, "MM": 0.90}
FIXED_QUERY_SEED = 2_112_291
REFRESH_QUERY_SEED = 2_112_929


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _action_union_k(plan: Any, encoded: Any, cut: tuple[int, ...]) -> int:
    indices = torch.as_tensor(cut, device=plan.split_gates.device, dtype=torch.long)
    module = (plan.permission_matrix("MM")[indices] > 0.0) & (encoded.module_present[0] > 0.5)[None, :]
    environment = (plan.permission_matrix("QE")[indices] > 0.0) & (encoded.env_weights[0] > 0.0)[None, :]
    signatures = torch.cat((module, environment), dim=1)
    bearing = signatures.any(dim=1)
    return len({tuple(row) for row in signatures[bearing].detach().cpu().tolist()})


def _packet_source_masks(plan: Any, encoded: Any, cut: tuple[int, ...]) -> dict[str, np.ndarray]:
    indices = torch.as_tensor(cut, device=plan.split_gates.device, dtype=torch.long)
    mm = (plan.permission_matrix("MM")[indices] > 0.0) & (encoded.module_present[0] > 0.5)[None, :]
    qe = (plan.permission_matrix("QE")[indices] > 0.0) & (encoded.env_weights[0] > 0.0)[None, :]
    return {
        "MM": mm.detach().cpu().numpy().astype(np.uint8, copy=False),
        "QE": qe.detach().cpu().numpy().astype(np.uint8, copy=False),
        "bearing": torch.cat((mm, qe), dim=1).any(dim=1).detach().cpu().numpy().astype(np.uint8, copy=False),
    }


def _plan_work(
    *,
    plan: Any,
    encoded: Any,
    tree: Any,
    batch: Any,
    sample: Any,
    direct_projection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    routes: dict[str, Any] = {}
    total_actual = 0.0
    total_full = 0.0
    for mechanism in ALL_MECHANISMS:
        if direct_projection is not None and mechanism in direct_projection:
            projection = direct_projection[mechanism]
            projection_summary = {
                "achieved_work": float(projection.achieved_work),
                "full_access_work": float(projection.full_access_work),
                "work_fraction": (
                    float(projection.achieved_work / projection.full_access_work)
                    if float(projection.full_access_work) > 0.0 else 0.0
                ),
                "selected_unique_pairs": int(projection.selected_unique_pairs),
                "full_unique_pairs": int(projection.full_unique_pairs),
            }
            if mechanism == "MM":
                item = runner._canonical_route_support(plan, encoded, tree, mechanism)
                item["canonical_projection"] = projection_summary
                item["work_basis"] = "hard support on the native canonical panel; direct projection recorded separately"
            else:
                item = {
                    "work_basis": "direct-plan canonical projection; live query support is recorded separately",
                    "permission_status": plan.permission_status(mechanism),
                    "canonical_achieved_work": projection_summary["achieved_work"],
                    "canonical_full_access_work": projection_summary["full_access_work"],
                    "canonical_work_fraction": projection_summary["work_fraction"],
                    "selected_unique_pairs": projection_summary["selected_unique_pairs"],
                    "full_unique_pairs": projection_summary["full_unique_pairs"],
                }
        else:
            item = runner._canonical_route_support(plan, encoded, tree, mechanism)
            item["work_basis"] = "native canonical receiver/source panel"
        total_actual += float(item.get("canonical_actual_work", item.get("canonical_achieved_work", 0.0)))
        total_full += float(item.get("canonical_full_work", item.get("canonical_full_access_work", 0.0)))
        routes[mechanism] = item
    qe_live = runner._query_qe_support(plan, encoded, sample, batch)
    if direct_projection is not None and "QE" in direct_projection:
        routes["QE"]["actual_live_query_panel"] = qe_live
    return {
        "routes": routes,
        "canonical_total_work": total_actual,
        "canonical_full_access_total_work": total_full,
        "canonical_work_fraction": total_actual / total_full if total_full > 0 else 0.0,
        "qe_live_query_panel": qe_live,
    }


def _metric(
    *,
    action_key: str,
    prediction: torch.Tensor,
    encoded: Any,
    tree: Any,
    plan: Any,
    cut: tuple[int, ...],
    paths: tuple[str, ...],
    batch: Any,
    sample: Any,
    normalizer: Any,
    role_scales: dict[str, float],
    same_student_full_rmse: dict[str, float],
    incumbent_rmse: dict[str, float],
    executor_aux: dict[str, Any],
    capacity: dict[str, float],
    direct_projection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _loss, mse, rmse = runner._role_objective(
        prediction, batch.target_field, sample.role_slices, normalizer, role_scales
    )
    work = _plan_work(
        plan=plan,
        encoded=encoded,
        tree=tree,
        batch=batch,
        sample=sample,
        direct_projection=direct_projection,
    )
    is_packet_action = direct_projection is None
    k = _action_union_k(plan, encoded, cut) if is_packet_action else None
    canonical_hard_sparse: dict[str, bool | None] = {}
    canonical_projection_sparse: dict[str, bool] = {}
    for mechanism in ("MM", "QE"):
        route = work["routes"].get(mechanism, {})
        if direct_projection is not None:
            projection = direct_projection.get(mechanism)
            if projection is not None:
                canonical_projection_sparse[mechanism] = bool(
                    int(projection.selected_unique_pairs) > 0
                    and int(projection.selected_unique_pairs) < int(projection.full_unique_pairs)
                )
            if mechanism == "QE":
                canonical_hard_sparse[mechanism] = None
                continue
        selected = route.get("actual_live_unique_pairs")
        eligible = route.get("eligible_unique_pairs")
        canonical_hard_sparse[mechanism] = (
            bool(0 < int(selected) < int(eligible))
            if selected is not None and eligible is not None
            else None
        )
    live_qe = work["qe_live_query_panel"]
    live_query_qe_sparse = bool(
        int(live_qe["actual_selected_pairs"]) > 0
        and int(live_qe["actual_selected_pairs"]) < int(live_qe["actual_full_pairs"])
    )
    support_bearing = bool(
        (not is_packet_action or int(k or 0) > 0)
        and (any(value is True for value in canonical_hard_sparse.values()) or live_query_qe_sparse)
    )
    if is_packet_action:
        collapsed_cut: bool | None = bool(k < len(cut))
        realized_cut_node_count: int | None = len(cut)
    else:
        # P's direct projection has no packet-tree cut, so packet K is
        # undefined. Record whether the actual projected mechanism support is
        # nonempty and sparse without inventing a cut cardinality.
        collapsed_cut = None
        realized_cut_node_count = None
    return {
        "action_key": action_key,
        "capacity_vector": dict(capacity),
        "requested_cut_paths": list(paths),
        "realized_cut_paths": list(paths),
        "realized_cut_node_count": realized_cut_node_count,
        "realized_nonredundant_k": k,
        "collapsed_cut": collapsed_cut,
        "support_bearing_sparse_action": support_bearing,
        "canonical_hard_sparse_support_by_route": canonical_hard_sparse,
        "canonical_projection_sparse_by_route": canonical_projection_sparse,
        "sampled_query_qe_hard_sparse_success": live_query_qe_sparse,
        "role_mse_mps2": mse,
        "role_rmse_mps": rmse,
        "role_rmse_over_fixed_wfull": {
            role: float(rmse[role] / max(incumbent_rmse[role], 1.0e-12)) for role in ROLE_NAMES
        },
        "role_rmse_over_same_student_full": {
            role: float(rmse[role] / max(same_student_full_rmse[role], 1.0e-12)) for role in ROLE_NAMES
        },
        "same_student_full_role_rmse_mps": dict(same_student_full_rmse),
        "fixed_run2110_wfull_role_rmse_mps": dict(incumbent_rmse),
        "work": work,
        "native_executor_rows": runner._summarize_executor_rows(executor_aux),
    }


def _load_student(
    *,
    arm: str,
    payload: Mapping[str, Any],
    source_payload: dict[str, Any],
    normalizer: Any,
    batch: Any,
    forward_config: dict[str, Any],
    device: torch.device,
) -> tuple[Any, Any]:
    arm_name = "g_packet" if arm == "g" else "direct_pair"
    model, _physical, control, _route, _optimizer = maturation._new_student(
        arm=arm_name,
        warm=payload,
        source_payload=source_payload,
        normalizer=normalizer,
        batch=batch,
        forward_config=forward_config,
        device=device,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if arm == "g":
        control.load_state_dict(payload["organizer_state_dict"], strict=True)
    else:
        control.load_state_dict(payload["direct_scorer_state_dict"], strict=True)
    model.eval()
    control.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in control.parameters():
        parameter.requires_grad_(False)
    return model, control


def _select_layout_rows(
    rows: list[int], view: Any, *, seed: int, layout_count: int = 0
) -> list[int]:
    by_layout: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_layout[int(view.metadata["layout_index"][int(row)])].append(int(row))
    rng = np.random.default_rng(seed)
    selected = []
    layouts = sorted(by_layout)
    if layout_count > 0 and layout_count < len(layouts):
        layouts = sorted(map(int, rng.choice(layouts, size=layout_count, replace=False)))
    for layout in layouts:
        candidates = sorted(by_layout[layout])
        selected.append(candidates[int(rng.integers(0, len(candidates)))])
    return selected


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in rows:
        split = str(case["split"])
        panel = str(case["query_panel"])
        for condition, metric in case["conditions"].items():
            grouped[(split, panel, condition)].append({
                "layout_index": case["layout_index"],
                "row": case["row"],
                "role_rmse_mps": metric["role_rmse_mps"],
                "role_rmse_over_fixed_wfull": metric["role_rmse_over_fixed_wfull"],
                "role_rmse_over_same_student_full": metric["role_rmse_over_same_student_full"],
            })
    summary: dict[str, Any] = {}
    for (split, panel, condition), cases in sorted(grouped.items()):
        role_summary: dict[str, Any] = {}
        for role in ROLE_NAMES:
            values = np.asarray([item["role_rmse_mps"][role] for item in cases], dtype=np.float64)
            ratios = np.asarray([item["role_rmse_over_fixed_wfull"][role] for item in cases], dtype=np.float64)
            full_ratios = np.asarray([item["role_rmse_over_same_student_full"][role] for item in cases], dtype=np.float64)
            worst = int(np.argmax(values))
            role_summary[role] = {
                "rmse_mps": {
                    "median": float(np.median(values)),
                    "p90": float(np.quantile(values, 0.90)),
                    "p95": float(np.quantile(values, 0.95)),
                    "maximum": float(values[worst]),
                    "worst_layout_index": int(cases[worst]["layout_index"]),
                    "worst_row": int(cases[worst]["row"]),
                },
                "ratio_to_fixed_wfull_rmse": {
                    "median": float(np.median(ratios)),
                    "p90": float(np.quantile(ratios, 0.90)),
                    "maximum": float(ratios.max()),
                },
                "ratio_to_same_student_full_rmse": {
                    "median": float(np.median(full_ratios)),
                    "p90": float(np.quantile(full_ratios, 0.90)),
                    "maximum": float(full_ratios.max()),
                },
            }
        summary[f"{split}/{panel}/{condition}"] = {
            "case_count": len(cases),
            "layout_count": len({item["layout_index"] for item in cases}),
            "roles": role_summary,
        }
    return summary


def _summarize_training_exposure(
    updates_path: Path,
    *,
    checkpoint_update: int,
    train_rows: list[int],
    layout_indices: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    update_rows = [
        json.loads(line)
        for line in updates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sparse_rows = [
        row for row in update_rows
        if not row.get("full_access_replay", False)
        and int(row.get("update_count", -1)) <= int(checkpoint_update)
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sparse_rows:
        key = (
            str(row.get("requested_action", "unknown")),
            json.dumps(row.get("capacity_vector", {}), sort_keys=True, separators=(",", ":")),
            json.dumps(row.get("requested_cut_paths", []), separators=(",", ":")),
        )
        grouped[key].append(row)

    groups: dict[str, Any] = {}
    detailed_rows: list[dict[str, Any]] = []
    for (action, capacity_json, requested_paths_json), records in sorted(grouped.items()):
        row_counts = Counter(int(row["row"]) for row in records)
        visits_all_rows = [int(row_counts.get(int(row), 0)) for row in train_rows]
        cut_histogram = Counter(
            json.dumps({
                "realized_cut_paths": row.get("realized_cut_paths", []),
                "realized_nonredundant_k": row.get("realized_nonredundant_k"),
            }, sort_keys=True, separators=(",", ":"))
            for row in records
        )
        group_key = f"{action}|capacity={capacity_json}|requested_paths={requested_paths_json}"
        groups[group_key] = {
            "requested_action": action,
            "capacity_vector": json.loads(capacity_json),
            "requested_cut_paths": json.loads(requested_paths_json),
            "sparse_optimizer_visits": len(records),
            "training_rows_total": len(train_rows),
            "unique_rows_visited": sum(count > 0 for count in visits_all_rows),
            "unique_layouts_visited": len({int(layout_indices[int(row)]) for row in row_counts}),
            "sparse_visits_per_408_row_pass": len(records) / len(train_rows),
            "complete_shuffled_passes_lower_bound": min(visits_all_rows, default=0),
            "minimum_visits_per_train_row": min(visits_all_rows, default=0),
            "maximum_visits_per_train_row": max(visits_all_rows, default=0),
            "rows_with_at_least_two_visits": sum(count >= 2 for count in visits_all_rows),
            "realized_cut_k_histogram": {
                key: int(value) for key, value in sorted(cut_histogram.items())
            },
        }
        for row in records:
            row_id = int(row["row"])
            detailed_rows.append({
                "update_count": int(row["update_count"]),
                "row": row_id,
                "layout_index": int(layout_indices[row_id]),
                "requested_action": action,
                "requested_cut_paths": list(row.get("requested_cut_paths", [])),
                "realized_cut_paths": list(row.get("realized_cut_paths", [])),
                "capacity_vector": dict(row.get("capacity_vector", {})),
                "realized_nonredundant_k": row.get("realized_nonredundant_k"),
                "route_work": row.get("route_work", {}),
                "permission_status": row.get("permission_status", {}),
                "physical_optimizer_step": True,
                "route_optimizer_step_applied": bool(row.get("route_optimizer_step_applied", False)),
            })
    return groups, sorted(detailed_rows, key=lambda row: int(row["update_count"]))


def _summarize_paired_training_loss_tail(run_dir: Path, *, checkpoint_update: int) -> dict[str, Any]:
    """Keep optimizer-stream loss tails separate from fixed physical-panel metrics."""

    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for arm, directory in (("g_packet", "g_packet"), ("direct_pair", "direct_pair")):
        path = run_dir / "arms" / directory / "updates.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_arm[arm] = {
            int(record["update_count"]): record
            for record in records
            if 100 < int(record.get("update_count", -1)) <= int(checkpoint_update)
        }
    g_updates = by_arm["g_packet"]
    p_updates = by_arm["direct_pair"]
    if set(g_updates) != set(p_updates):
        raise ValueError("G/P training loss tails do not contain identical optimizer update IDs")
    identity_fields = (
        "row", "query_seed", "requested_action", "requested_cut_paths",
        "capacity_vector", "full_access_replay",
    )
    for update_count in sorted(g_updates):
        for field in identity_fields:
            left = json.dumps(g_updates[update_count].get(field), sort_keys=True)
            right = json.dumps(p_updates[update_count].get(field), sort_keys=True)
            if left != right:
                raise ValueError(f"G/P training stream differs at u{update_count} field {field}")

    def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        values = np.asarray([float(row["combined_optimizer_loss"]) for row in records], dtype=np.float64)
        finite = np.isfinite(values)
        if not bool(finite.all()):
            raise FloatingPointError("non-finite value found in saved training loss tail")
        return {
            "update_count": len(records),
            "finite_loss_count": int(finite.sum()),
            "combined_optimizer_loss": {
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": float(values.max()),
            },
            "loss_gt_10_count": int((values > 10.0).sum()),
            "loss_gt_100_count": int((values > 100.0).sum()),
            "loss_gt_10_update_ids": [int(row["update_count"]) for row in records if float(row["combined_optimizer_loss"]) > 10.0],
            "loss_gt_100_update_ids": [int(row["update_count"]) for row in records if float(row["combined_optimizer_loss"]) > 100.0],
        }

    arms: dict[str, Any] = {}
    for arm, updates in by_arm.items():
        records = [updates[index] for index in sorted(updates)]
        arms[arm] = {
            "all_updates": _summary(records),
            "sparse_updates": _summary([row for row in records if not row.get("full_access_replay", False)]),
            "full_replay_updates": _summary([row for row in records if row.get("full_access_replay", False)]),
        }

    outlier_updates = sorted({
        update_count
        for updates in by_arm.values()
        for update_count, row in updates.items()
        if float(row["combined_optimizer_loss"]) > 10.0
    })
    outliers = []
    for update_count in outlier_updates:
        pair = {}
        for arm, updates in by_arm.items():
            row = updates[update_count]
            pair[arm] = {
                "combined_optimizer_loss": float(row["combined_optimizer_loss"]),
                "role_rmse_mps": dict(row.get("role_rmse_mps", {})),
                "route_gradient_preclip_l2": float(row.get("route_gradient_preclip_l2", 0.0)),
                "route_parameter_update_l2": float(row.get("route_parameter_update_l2", 0.0)),
            }
        exemplar = g_updates[update_count]
        outliers.append({
            "update_count": update_count,
            "row": int(exemplar["row"]),
            "query_seed": int(exemplar["query_seed"]),
            "requested_action": exemplar.get("requested_action"),
            "capacity_vector": exemplar.get("capacity_vector"),
            "full_access_replay": bool(exemplar.get("full_access_replay", False)),
            "paired_arm_values": pair,
        })
    return {
        "update_interval": [101, int(checkpoint_update)],
        "matched_update_count": len(g_updates),
        "identical_row_query_action_capacity_stream": True,
        "loss_definition": "combined_optimizer_loss from the predeclared physical update ledger",
        "thresholds_are_counts_of_updates_strictly_above": {"10": 10.0, "100": 100.0},
        "arms": arms,
        "paired_outliers_gt_10": outliers,
    }


def _checkpoint_review_binding(
    run_dir: Path, *, arm: str, update_count: int
) -> tuple[Path, dict[str, Any], str]:
    """Bind an endpoint to its append-only review record, even after later resumes."""

    review_log = run_dir / "reviews.jsonl"
    matches: list[tuple[dict[str, Any], str]] = []
    for line in review_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record.get("arm")) == arm and int(record.get("update_count", -1)) == int(update_count):
            matches.append((record, hashlib.sha256(line.encode("utf-8")).hexdigest()))
    if not matches:
        raise FileNotFoundError(f"no immutable review record for {arm} u{update_count} in {review_log}")
    records = {str(item[0].get("checkpoint_sha256")) for item in matches}
    paths = {str(item[0].get("checkpoint")) for item in matches}
    if len(records) != 1 or len(paths) != 1:
        raise ValueError(f"conflicting append-only checkpoint records for {arm} u{update_count}")
    record, line_sha256 = matches[-1]
    checkpoint = Path(str(record["checkpoint"]))
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"review-bound checkpoint is missing: {checkpoint}")
    if runner._file_sha256(checkpoint) != record.get("checkpoint_sha256"):
        raise ValueError(f"review-bound checkpoint bytes do not match {arm} u{update_count} record")
    return checkpoint, record, line_sha256


def _source_mask_turnover(
    *,
    run_dir: Path,
    update_count: int,
    current_index: list[dict[str, Any]],
    current_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    prior_reviews = []
    for path in run_dir.glob("u*_fixed_panel_review/fixed_train_dev_review.json"):
        try:
            prior_payload = json.loads(path.read_text(encoding="utf-8"))
            prior_update = int(prior_payload.get("update_count", -1))
        except (OSError, ValueError, TypeError):
            continue
        if 100 < prior_update < update_count:
            prior_reviews.append((prior_update, path, prior_payload))
    if not prior_reviews:
        return {"status": "no_prior_review", "paired_masks": 0, "records": [], "summary": {}}

    prior_update, prior_json, prior_payload = max(prior_reviews, key=lambda item: item[0])
    prior_archive = Path(prior_payload["packet_source_mask_archive"])
    if not prior_archive.is_absolute():
        prior_archive = (prior_json.parent / prior_archive).resolve()
    prior_index = prior_payload.get("packet_source_mask_index", [])
    prior_by_key = {
        (
            str(item["split"]), int(item["row"]), str(item["query_panel"]), str(item["condition"])
        ): item
        for item in prior_index
    }
    records = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with np.load(prior_archive, allow_pickle=False) as prior_npz:
        for current in current_index:
            key = (
                str(current["split"]), int(current["row"]),
                str(current["query_panel"]), str(current["condition"]),
            )
            prior = prior_by_key.get(key)
            if prior is None or list(prior.get("realized_cut_paths", [])) != list(current.get("realized_cut_paths", [])):
                continue
            current_prefix = str(current["array_prefix"])
            prior_prefix = str(prior["array_prefix"])
            for mechanism, suffix in (("MM", "mm"), ("QE", "qe")):
                left = np.asarray(prior_npz[f"{prior_prefix}_{suffix}"], dtype=bool)
                right = np.asarray(current_arrays[f"{current_prefix}_{suffix}"], dtype=bool)
                if left.shape != right.shape:
                    continue
                changed = left ^ right
                left_union = left.any(axis=0)
                right_union = right.any(axis=0)
                union_count = int(np.logical_or(left, right).sum())
                jaccard = (
                    float(np.logical_and(left, right).sum() / union_count)
                    if union_count else 1.0
                )
                source_union = int(np.logical_or(left_union, right_union).sum())
                source_jaccard = (
                    float(np.logical_and(left_union, right_union).sum() / source_union)
                    if source_union else 1.0
                )
                record = {
                    "split": key[0],
                    "row": key[1],
                    "layout_index": int(current["layout_index"]),
                    "query_panel": key[2],
                    "condition": key[3],
                    "mechanism": mechanism,
                    "packet_source_shape": list(right.shape),
                    "changed_packet_source_entries": int(changed.sum()),
                    "changed_sources_after_packet_union": int(np.logical_xor(left_union, right_union).sum()),
                    "packet_source_jaccard": jaccard,
                    "source_union_jaccard": source_jaccard,
                }
                records.append(record)
                grouped[(key[0], key[2], key[3], mechanism)].append(record)
    summary: dict[str, Any] = {}
    for key, values in sorted(grouped.items()):
        changed_bits = np.asarray([item["changed_packet_source_entries"] for item in values], dtype=np.float64)
        changed_sources = np.asarray([item["changed_sources_after_packet_union"] for item in values], dtype=np.float64)
        jaccard = np.asarray([item["packet_source_jaccard"] for item in values], dtype=np.float64)
        union_jaccard = np.asarray([item["source_union_jaccard"] for item in values], dtype=np.float64)
        summary["/".join(key)] = {
            "paired_case_count": len(values),
            "changed_packet_source_entries": {
                "median": float(np.median(changed_bits)),
                "p90": float(np.quantile(changed_bits, 0.90)),
                "maximum": int(changed_bits.max()),
            },
            "changed_sources_after_packet_union": {
                "median": float(np.median(changed_sources)),
                "p90": float(np.quantile(changed_sources, 0.90)),
                "maximum": int(changed_sources.max()),
            },
            "packet_source_jaccard": {
                "median": float(np.median(jaccard)),
                "p10": float(np.quantile(jaccard, 0.10)),
                "minimum": float(jaccard.min()),
            },
            "source_union_jaccard": {
                "median": float(np.median(union_jaccard)),
                "p10": float(np.quantile(union_jaccard, 0.10)),
                "minimum": float(union_jaccard.min()),
            },
        }
    return {
        "status": "compared_with_prior_endpoint",
        "prior_update_count": prior_update,
        "prior_review_json": str(prior_json),
        "prior_g_checkpoint_sha256": prior_payload.get("g_checkpoint_sha256"),
        "paired_masks": len(records),
        "records": records,
        "summary": summary,
        "comparison_note": "same split, row, query panel, action, and packet path; reports packet×source support turnover for G masks",
    }


def _predict_direct_pair_full_panel(
    *,
    model: Any,
    encoded: Any,
    batch: Any,
    plan: MechanismPlan,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Read explicit P permissions with the exact full panel they were built for."""

    core = model.core
    prepared = core.prepare(encoded, encoded.module_tokens, fixed_cover_plans=(plan,))
    output = core.decode_queries(
        prepared,
        batch.query_xy,
        query_features=batch.query_features,
        receiver_chunk_size=int(batch.query_xy.shape[1]),
        return_interaction_aux=True,
    )
    return output["pred_field"], dict(output.get("_interaction_aux", {}))


def _evaluate_case_panel(
    *,
    split: str,
    row_id: int,
    panel: str,
    query_seed: int,
    view: Any,
    sampler: Any,
    query_counts: dict[str, int],
    normalizer: Any,
    role_scales: dict[str, float],
    source_model: Any,
    g_model: Any,
    g_organizer: Any,
    p_model: Any,
    p_scorer: Any,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, dict[str, dict[str, np.ndarray]]]:
    case, sample, batch = maturation._case_batch(
        view=view,
        row_id=row_id,
        query_seed=query_seed,
        role_counts=query_counts,
        sampler=sampler,
        normalizer=normalizer,
        device=device,
    )
    layout_index = int(view.metadata["layout_index"][row_id])
    conditions: dict[str, Any] = {}
    source_masks: dict[str, dict[str, np.ndarray]] = {}
    with torch.inference_mode():
        # The same immutable query panel is used for W-full, both students,
        # every sparse action, and the frozen 0.90/0.90 probe.
        source_model.core.backend.set_cover_mode("external")
        source_encoded = source_model.core.encode_case(batch)
        source_tree = source_model.core.backend.build_case_trees(source_encoded)[0]
        source_full_plan = MechanismPlan.full_access(
            source_tree,
            source_encoded.module_present[0],
            int(source_encoded.env_coords.shape[1]),
        )
        source_pred, source_aux = runner._prediction_with_plan(
            source_model, source_encoded, batch, source_full_plan, return_interaction_aux=True
        )
        _loss, _mse, source_rmse = runner._role_objective(
            source_pred, batch.target_field, sample.role_slices, normalizer, role_scales
        )

        for arm, model, control in (("g", g_model, g_organizer), ("p", p_model, p_scorer)):
            model.core.backend.set_cover_mode("external")
            encoded = model.core.encode_case(batch)
            tree = model.core.backend.build_case_trees(encoded)[0]
            full_plan = MechanismPlan.full_access(
                tree, encoded.module_present[0], int(encoded.env_coords.shape[1])
            )
            full_pred, full_aux = runner._prediction_with_plan(
                model, encoded, batch, full_plan, return_interaction_aux=True
            )
            _loss, _mse, full_rmse = runner._role_objective(
                full_pred, batch.target_field, sample.role_slices, normalizer, role_scales
            )
            conditions[f"{arm}_full"] = {
                "action_key": f"{arm}_full",
                "capacity_vector": {"QE": 1.0, "MM": 1.0},
                "requested_cut_paths": [],
                "realized_cut_paths": [],
                "realized_cut_node_count": None,
                "realized_nonredundant_k": None,
                "collapsed_cut": None,
                "support_bearing_sparse_action": False,
                "role_rmse_mps": full_rmse,
                "role_mse_mps2": {role: full_rmse[role] ** 2 for role in ROLE_NAMES},
                "role_rmse_over_fixed_wfull": {
                    role: float(full_rmse[role] / max(source_rmse[role], 1.0e-12)) for role in ROLE_NAMES
                },
                "role_rmse_over_same_student_full": {role: 1.0 for role in ROLE_NAMES},
                "same_student_full_role_rmse_mps": dict(full_rmse),
                "fixed_run2110_wfull_role_rmse_mps": dict(source_rmse),
                "work": _plan_work(
                    plan=full_plan, encoded=encoded, tree=tree, batch=batch, sample=sample
                ),
                "native_executor_rows": runner._summarize_executor_rows(full_aux),
            }
            if arm == "g":
                source_masks["g_full"] = _packet_source_masks(full_plan, encoded, (0,))

            if arm == "g":
                score_inputs = {
                    "module_states": encoded.module_tokens,
                    "environment_states": encoded.env_tokens,
                    "global_state": encoded.global_token,
                }
                for capacity_name, capacity in (("primary", PRIMARY_CAPACITY), ("probe_090", FROZEN_090_PROBE)):
                    scores = control.score_cases(encoded, score_inputs, (tree,), budgets=capacity)
                    for action, requested_paths in WIND_ACTION_PATHS.items():
                        cut, actual_paths = available_frontier_for_paths(tree, requested_paths)
                        plan = control.plans_from_scores(
                            scores, encoded, (tree,), hard=True,
                            frontier_cuts=(cut,), budget_fractions=capacity,
                        )[0]
                        prediction, aux = runner._prediction_with_plan(
                            model, encoded, batch, plan, return_interaction_aux=True
                        )
                        key = f"g_{capacity_name}_{action}"
                        source_masks[key] = _packet_source_masks(plan, encoded, cut)
                        metric = _metric(
                            action_key=key,
                            prediction=prediction,
                            encoded=encoded,
                            tree=tree,
                            plan=plan,
                            cut=cut,
                            paths=actual_paths,
                            batch=batch,
                            sample=sample,
                            normalizer=normalizer,
                            role_scales=role_scales,
                            same_student_full_rmse=full_rmse,
                            incumbent_rmse=source_rmse,
                            executor_aux=dict(aux),
                            capacity=capacity,
                        )
                        conditions[key] = metric
            else:
                for capacity_name, capacity in (("primary", PRIMARY_CAPACITY), ("probe_090", FROZEN_090_PROBE)):
                    hard_plan, _soft_plan, projections, _actual = maturation._direct_plan_pair_by_route(
                        model=model,
                        encoded=encoded,
                        batch=batch,
                        scorer=control,
                        route_fractions=capacity,
                    )
                    prediction, aux = _predict_direct_pair_full_panel(
                        model=model, encoded=encoded, batch=batch, plan=hard_plan
                    )
                    direct_projection = {
                        mechanism: projection for mechanism, projection in projections.items()
                    }
                    key = f"p_{capacity_name}_direct"
                    cut = (0,)
                    conditions[key] = _metric(
                        action_key=key,
                        prediction=prediction,
                        encoded=encoded,
                        tree=tree,
                        plan=hard_plan,
                        cut=cut,
                        paths=("direct_pair_plan",),
                        batch=batch,
                        sample=sample,
                        normalizer=normalizer,
                        role_scales=role_scales,
                        same_student_full_rmse=full_rmse,
                        incumbent_rmse=source_rmse,
                        executor_aux=dict(aux),
                        capacity=capacity,
                        direct_projection=direct_projection,
                    )

    row_record = {
        "split": split,
        "row": int(row_id),
        "layout_index": layout_index,
        "wind_direction_deg": float(case.wind_direction_deg),
        "query_panel": panel,
        "query_seed": int(query_seed),
        "role_query_counts": dict(query_counts),
        "role_names": list(ROLE_NAMES),
        "fixed_run2110_wfull_role_rmse_mps": dict(source_rmse),
        "conditions": conditions,
    }
    return row_record, np.asarray(sample.flat_indices, dtype=np.int64), source_masks


def main() -> None:
    run_dir = (PROJECT / "Trained_Results/WindFarm/HONF_Forward_Runs/Run_2112_controlled_maturation_20260929").resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=run_dir)
    parser.add_argument("--update-count", type=int, default=200)
    parser.add_argument(
        "--layouts-per-split", type=int, default=4,
        help="number of distinct layouts per train/development panel; use 0 for every layout",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    update_count = int(args.update_count)
    layouts_per_split = int(args.layouts_per_split)
    if update_count < 101 or layouts_per_split < 0:
        raise ValueError("update-count must exceed 100 and layouts-per-split must be nonnegative")
    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    attempt_id = f"u{update_count}_review_{started_at.replace(':', '').replace('-', '')}_{os.getpid()}"
    script_path = Path(__file__).resolve()
    command_argv = [sys.executable, str(script_path), *sys.argv[1:]]
    attempt_log = run_dir / "attempts.jsonl"
    _append_jsonl(attempt_log, {
        "event": "fixed_panel_evaluation_started",
        "run_id": RUN_ID,
        "attempt_id": attempt_id,
        "arm": "g_packet+direct_pair+Run2110_W_full",
        "interpreter": sys.executable,
        "driver_script": str(script_path),
        "driver_source_sha256": runner._file_sha256(script_path),
        "maturation_helper_path": str(maturation.MATURATION_HELPER_PATH),
        "maturation_helper_source_sha256": runner._file_sha256(maturation.MATURATION_HELPER_PATH),
        "command_argv": command_argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at_utc": started_at,
        "optimizer_attempts": 0,
    })
    failure: dict[str, str] | None = None
    summary: dict[str, Any] | None = None
    try:
        if Path(sys.executable).resolve() != Path("/home/wanglz/miniconda3/envs/ModularDT/bin/python").resolve():
            raise RuntimeError("Use /home/wanglz/miniconda3/envs/ModularDT/bin/python for Wind evaluation")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "0":
            raise RuntimeError("Wind fixed-panel review requires CUDA_VISIBLE_DEVICES=0")
        if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
            raise RuntimeError("authorized physical GPU 0 is unavailable")
        if torch.cuda.get_device_name(0) != "NVIDIA RTX 6000 Ada Generation" or maturation._gpu_uuid() != DEVICE_UUID:
            raise RuntimeError("CUDA_VISIBLE_DEVICES=0 did not resolve to the authorized physical GPU")

        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != RUN_ID or float(manifest.get("gpu_active_seconds", 0.0)) >= maturation.LANE_BUDGET_SECONDS:
            raise ValueError("Run2112 identity is invalid or its 14 GPU-hour lane budget is exhausted")
        output_dir = run_dir / f"u{update_count}_fixed_panel_review"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = output_dir / "fixed_train_dev_review.json"
        if output_json.exists():
            raise FileExistsError(f"refusing to overwrite measured review {output_json}")

        config_path = Path(runner.DEFAULT_CONFIG).resolve()
        config = runner._load_config(config_path)
        view, canonical, train_rows, split_record = runner._native_inputs(config)
        source_path, source_payload, normalizer, source_sha = runner._load_source(config)
        if source_sha != runner.EXPECTED_SOURCE_SHA256:
            raise ValueError("fixed review incumbent changed from audited Run2110 W-full")
        role_scales = {str(k): float(v) for k, v in config["forward"]["stage_a"]["role_loss_scales_mps"].items()}
        query_counts = {str(k): int(v) for k, v in config["forward"]["stage_b"]["role_query_counts"].items()}
        layout_indices = np.asarray(view.metadata["layout_index"], dtype=np.int64)
        train_panel_rows = _select_layout_rows(
            train_rows, view, seed=212_119, layout_count=layouts_per_split
        )
        development_rows = [int(row) for row in canonical.validation]
        dev_panel_rows = _select_layout_rows(
            development_rows, view, seed=212_229, layout_count=layouts_per_split
        )
        if (
            len(train_panel_rows) != len(set(layout_indices[train_panel_rows].tolist()))
            or len(dev_panel_rows) != len(set(layout_indices[dev_panel_rows].tolist()))
            or (layouts_per_split > 0 and len(train_panel_rows) != layouts_per_split)
            or (layouts_per_split > 0 and len(dev_panel_rows) != layouts_per_split)
        ):
            raise ValueError("fixed review must contain the requested number of unique train/development layouts")

        checkpoint_bindings = {
            arm: _checkpoint_review_binding(run_dir, arm=arm, update_count=update_count)
            for arm in ("g_packet", "direct_pair")
        }
        checkpoint_paths = {arm: value[0] for arm, value in checkpoint_bindings.items()}
        checkpoint_payloads: dict[str, Mapping[str, Any]] = {}
        checkpoint_hashes: dict[str, str] = {}
        for arm, path in checkpoint_paths.items():
            _bound_path, checkpoint_record, _record_sha = checkpoint_bindings[arm]
            digest = runner._file_sha256(path)
            latest = json.loads((path.parents[1] / "latest_review.json").read_text(encoding="utf-8"))
            latest_update = int(latest.get("update_count", -1))
            if latest_update < update_count:
                raise ValueError(f"u{update_count} {arm} checkpoint is newer than the arm's latest pointer")
            if (
                latest.get("checkpoint_sha256") != digest
                and latest_update == update_count
            ):
                raise ValueError(f"u{update_count} {arm} latest pointer conflicts with its immutable review record")
            payload = load_trusted_checkpoint(path, map_location="cpu")
            if (
                not isinstance(payload, Mapping)
                or payload.get("run_id") != RUN_ID
                or payload.get("arm") != arm
                or int(payload.get("update_count", -1)) != update_count
                or payload.get("source_checkpoint_sha256") != source_sha
                or payload.get("train_rows_sha256") != split_record["student_train_rows_sha256"]
                or digest != checkpoint_record.get("checkpoint_sha256")
            ):
                raise ValueError(f"Run2112 {arm} u{update_count} checkpoint identity does not match this data/source")
            checkpoint_payloads[arm] = payload
            checkpoint_hashes[arm] = digest

        device = torch.device("cuda:0")
        probe_row = train_panel_rows[0]
        _probe_case, _probe_sample, init_batch = maturation._case_batch(
            view=view,
            row_id=probe_row,
            query_seed=FIXED_QUERY_SEED + probe_row,
            role_counts=query_counts,
            sampler=runner.NativeRoleCatalogueCache(),
            normalizer=normalizer,
            device=device,
        )
        source_model = runner._new_model_from_source(source_payload, normalizer, init_batch, device)
        source_model.eval()
        for parameter in source_model.parameters():
            parameter.requires_grad_(False)
        g_model, g_control = _load_student(
            arm="g", payload=checkpoint_payloads["g_packet"], source_payload=source_payload,
            normalizer=normalizer, batch=init_batch, forward_config=config["forward"], device=device,
        )
        p_model, p_control = _load_student(
            arm="p", payload=checkpoint_payloads["direct_pair"], source_payload=source_payload,
            normalizer=normalizer, batch=init_batch, forward_config=config["forward"], device=device,
        )
        role_rows: list[dict[str, Any]] = []
        source_mask_entries: list[dict[str, Any]] = []
        panel_indices: list[np.ndarray] = []
        panel_index_metadata: list[dict[str, Any]] = []
        sampler = runner.NativeRoleCatalogueCache()
        row_groups = [("train", row) for row in train_panel_rows] + [("development", row) for row in dev_panel_rows]
        for split, row_id in row_groups:
            for panel, seed_base in (("fixed", FIXED_QUERY_SEED), ("refreshed", REFRESH_QUERY_SEED)):
                query_seed = int(seed_base + int(row_id) * 1_009)
                row_record, query_indices, source_masks = _evaluate_case_panel(
                    split=split,
                    row_id=int(row_id),
                    panel=panel,
                    query_seed=query_seed,
                    view=view,
                    sampler=sampler,
                    query_counts=query_counts,
                    normalizer=normalizer,
                    role_scales=role_scales,
                    source_model=source_model,
                    g_model=g_model,
                    g_organizer=g_control,
                    p_model=p_model,
                    p_scorer=p_control,
                    device=device,
                )
                role_rows.append(row_record)
                for condition, masks in sorted(source_masks.items()):
                    array_prefix = f"mask_{len(source_mask_entries):04d}"
                    mask_metric = row_record["conditions"].get(condition, {})
                    source_mask_entries.append({
                        "array_prefix": array_prefix,
                        "split": split,
                        "row": int(row_id),
                        "layout_index": int(layout_indices[row_id]),
                        "query_panel": panel,
                        "condition": condition,
                        "realized_cut_paths": list(mask_metric.get("realized_cut_paths", [])),
                        "MM_shape": list(masks["MM"].shape),
                        "QE_shape": list(masks["QE"].shape),
                        "packet_bearing": masks["bearing"].tolist(),
                        "MM": masks["MM"],
                        "QE": masks["QE"],
                        "bearing": masks["bearing"],
                    })
                panel_indices.append(query_indices)
                panel_index_metadata.append({
                    "split": split,
                    "row": int(row_id),
                    "layout_index": int(layout_indices[row_id]),
                    "query_panel": panel,
                    "query_seed": query_seed,
                })

        exposure_path = run_dir / "arms/g_packet/updates.jsonl"
        exposure_by_action, exposure_rows = _summarize_training_exposure(
            exposure_path,
            checkpoint_update=update_count,
            train_rows=[int(row) for row in train_rows],
            layout_indices=layout_indices,
        )
        paired_training_loss_tail = _summarize_paired_training_loss_tail(
            run_dir, checkpoint_update=update_count
        )
        summary = {
            "run_id": RUN_ID,
            "update_count": update_count,
            "review_identity": f"same saved train/development role panels at matched G/P u{update_count} physical/scorer states",
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": source_sha,
            "maturation_helper_path": str(maturation.MATURATION_HELPER_PATH),
            "maturation_helper_source_sha256": runner._file_sha256(maturation.MATURATION_HELPER_PATH),
            "maturation_helper_resume_note": (
                "u101-u355 exercised scaffold L/R only; their behavior is unchanged by the L/R path-order correction."
            ),
            "config_path": str(config_path),
            "config_sha256": runner._file_sha256(config_path),
            "train_rows_sha256": split_record["student_train_rows_sha256"],
            "g_checkpoint": str(checkpoint_paths["g_packet"]),
            "g_checkpoint_sha256": checkpoint_hashes["g_packet"],
            "g_checkpoint_review_record": checkpoint_bindings["g_packet"][1],
            "g_checkpoint_review_record_line_sha256": checkpoint_bindings["g_packet"][2],
            "p_checkpoint": str(checkpoint_paths["direct_pair"]),
            "p_checkpoint_sha256": checkpoint_hashes["direct_pair"],
            "p_checkpoint_review_record": checkpoint_bindings["direct_pair"][1],
            "p_checkpoint_review_record_line_sha256": checkpoint_bindings["direct_pair"][2],
            "physical_gpu_uuid": DEVICE_UUID,
            "query_panel_design": {
                "role_query_counts": query_counts,
                "train_rows": len(train_panel_rows),
                "train_layouts": len(set(layout_indices[train_panel_rows].tolist())),
                "development_rows": len(dev_panel_rows),
                "development_layouts": len(set(layout_indices[dev_panel_rows].tolist())),
                "requested_layouts_per_split": layouts_per_split,
                "one_direction_row_per_layout": True,
                "test_rows_opened": False,
                "fixed_query_seed_base": FIXED_QUERY_SEED,
                "refreshed_query_seed_base": REFRESH_QUERY_SEED,
                "same_sample_indices_across_G_P_actions_and_reference": True,
            },
            "capacity_design": {
                "trained_primary": dict(PRIMARY_CAPACITY),
                "frozen_090_probe_only": dict(FROZEN_090_PROBE),
                "no_075_probe_in_review": True,
                "G_actions": ["root", "two_packet", "four_packet", "explicit_full"],
                "P_actions": ["direct_pair_primary", "direct_pair_090_probe", "explicit_full"],
                "P_direct_QE_projection": "one plan per full live query panel; no per-chunk budget reprojection",
                "P_direct_decode_receiver_chunk_size": "full query panel",
            },
            "G_sparse_exposure_by_requested_action_capacity_and_cut": exposure_by_action,
            "G_sparse_exposure_update_rows": exposure_rows,
            "paired_training_stream_loss_tail": paired_training_loss_tail,
            "action_exposure_scope": {
                key: {
                    "trained_sparse_candidate_requires_at_least_two_complete_shuffled_passes": (
                        int(value["complete_shuffled_passes_lower_bound"]) >= 2
                    ),
                    "complete_shuffled_passes_lower_bound": int(value["complete_shuffled_passes_lower_bound"]),
                    "sparse_visits_per_train_pass": float(value["sparse_visits_per_408_row_pass"]),
                    "unique_rows_visited": int(value["unique_rows_visited"]),
                    "training_rows_total": int(value["training_rows_total"]),
                }
                for key, value in exposure_by_action.items()
            },
            "evaluation_rows": role_rows,
            "role_layout_tail_summary": _summarize(role_rows),
            "panel_sample_index_archive": str(output_dir / "role_panel_native_indices.npz"),
            "panel_sample_index_metadata": panel_index_metadata,
            "model_native_role_catalogue_cache": sampler.summary(),
            "optimizer_calls": 0,
            "role_labels_used": "train and development only; hidden test rows unopened",
            "fixed_incumbent_is_reference_not_physical_truth": True,
        }
        np.savez_compressed(
            output_dir / "role_panel_native_indices.npz",
            flat_native_indices=np.stack(panel_indices),
            panel_metadata_json=np.asarray(json.dumps(panel_index_metadata)),
        )
        mask_arrays: dict[str, np.ndarray] = {}
        source_mask_index = []
        for entry in source_mask_entries:
            prefix = str(entry["array_prefix"])
            mask_arrays[f"{prefix}_mm"] = entry.pop("MM")
            mask_arrays[f"{prefix}_qe"] = entry.pop("QE")
            mask_arrays[f"{prefix}_bearing"] = entry.pop("bearing")
            source_mask_index.append(entry)
        source_mask_path = output_dir / "packet_source_masks.npz"
        np.savez_compressed(source_mask_path, **mask_arrays)
        summary["packet_source_mask_archive"] = str(source_mask_path)
        summary["packet_source_mask_index"] = source_mask_index
        summary["packet_source_mask_turnover_vs_prior_review"] = _source_mask_turnover(
            run_dir=run_dir,
            update_count=update_count,
            current_index=source_mask_index,
            current_arrays=mask_arrays,
        )
        _write_json(output_json, summary)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        elapsed = time.monotonic() - started_monotonic
        stopped_at = datetime.now(timezone.utc).isoformat()
        stop_record = {
            "event": "fixed_panel_evaluation_stopped",
            "run_id": RUN_ID,
            "attempt_id": attempt_id,
            "arm": "g_packet+direct_pair+Run2110_W_full",
            "stopped_at_utc": stopped_at,
            "active_wall_seconds": elapsed,
            "optimizer_attempts": 0,
            "status": "failed" if failure else "completed",
            "failure": failure,
        }
        _append_jsonl(attempt_log, stop_record)
        # Evaluation also occupies the allocated GPU lane. Add each stopped
        # evaluator invocation once to the shared Run2112 active-time total so
        # later training preflights see the same 14 h bound.
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            usage = manifest.setdefault("gpu_evaluation_attempts", [])
            if not any(item.get("attempt_id") == attempt_id for item in usage):
                usage.append({
                    "attempt_id": attempt_id,
                    "started_at_utc": started_at,
                    "stopped_at_utc": stopped_at,
                    "active_wall_seconds": elapsed,
                    "optimizer_attempts": 0,
                    "status": "failed" if failure else "completed",
                    "failure": failure,
                    "driver_script": str(script_path),
                    "driver_source_sha256": runner._file_sha256(script_path),
                })
                manifest["gpu_active_evaluation_seconds"] = float(
                    manifest.get("gpu_active_evaluation_seconds", 0.0)
                ) + elapsed
                manifest["gpu_active_seconds"] = float(
                    manifest.get("gpu_active_seconds", 0.0)
                ) + elapsed
                manifest["evaluation_active_seconds"] = float(
                    manifest.get("evaluation_active_seconds", 0.0)
                ) + elapsed
                manifest["last_evaluation_attempt"] = stop_record
                temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
                temporary.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(manifest_path)
    print(json.dumps({"status": "completed", "review": str(output_json), "active_wall_seconds": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
