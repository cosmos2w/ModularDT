"""Synchronized selected-checkpoint latency panel for Wind G/P and action selection.

The timer covers one native inference wrapper after its query batch is already
materialized on device. The learned G policy includes encoding, organizer
scoring, all three action plans and input-only descriptions, exact canonical
work, risk selection with explicit full fallback, preparation, and decode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
import fit_matured_action_selector as fit_driver
import run_active_packet_reuse as runner
import run_controlled_maturation as maturation
from honf_forward_core.interface_fields.action_aware_frontier import (
    ActionAwareRiskHead,
    describe_realized_plan,
    receiver_role_descriptors,
    select_action_by_risk,
)
from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan
from honf_runtime.compat import load_trusted_checkpoint

from windfarm.workflows.maturation import WIND_ACTION_PATHS, available_frontier_for_paths

RUN_ID = "2112"
ACTION_ORDER = ("root", "two_packet", "four_packet", "full_access")
MODE_ORDER = ("g_action_policy", "g_full", "p_direct", "retained_wfull")
PRIMARY_CAPACITY = {"QE": 0.95, "MM": 0.90}
ROLE_COUNTS = {
    64: {"volume": 32, "hub_slab": 8, "downstream_envelope": 8, "near_turbine": 8, "background": 8},
    1024: {"volume": 512, "hub_slab": 128, "downstream_envelope": 128, "near_turbine": 128, "background": 128},
    8192: {"volume": 4096, "hub_slab": 1024, "downstream_envelope": 1024, "near_turbine": 1024, "background": 1024},
}
DEFAULT_WARMUPS = 1
DEFAULT_REPEATS = 3
MAX_WARMUPS = 2
MAX_REPEATS = 5


@dataclass(frozen=True)
class Workload:
    name: str
    row_id: int
    module_count: int
    query_count: int
    role_counts: Mapping[str, int]
    query_seed: int
    case: Any
    sample: Any
    batch: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_table_path(table_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (table_path.parent / path).resolve()


def interleaved_workload_order(
    workloads: Sequence[str], modes: Sequence[str], *, rounds: int, seed: int,
) -> list[list[tuple[str, str]]]:
    """Return a deterministic independent shuffle of every workload/mode per round."""
    if not workloads or not modes or rounds < 1 or seed < 0:
        raise ValueError("Latency interleaving needs workloads, modes, positive rounds, and a nonnegative seed.")
    keys = [(str(workload), str(mode)) for workload in workloads for mode in modes]
    rng = np.random.default_rng(int(seed))
    return [
        [keys[int(index)] for index in rng.permutation(len(keys))]
        for _ in range(int(rounds))
    ]


def summarize_latencies(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        elapsed = float(row["elapsed_seconds"])
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Synchronized latency samples must be finite and nonnegative.")
        grouped[(str(row["workload"]), str(row["mode"]))].append(elapsed)
    return {
        f"{workload}/{mode}": {
            "measured_repeats": len(values),
            "p50_seconds": float(np.median(values)),
            "p90_seconds": float(np.quantile(values, 0.90)),
            "mean_seconds": float(np.mean(values)),
            "samples_seconds": list(values),
        }
        for (workload, mode), values in sorted(grouped.items())
    }


def _query_counts(query_count: int) -> dict[str, int]:
    try:
        return dict(ROLE_COUNTS[int(query_count)])
    except KeyError as exc:
        raise ValueError("Latency panel query counts are fixed to Q64, Q1024, and Q8192.") from exc


def _selector_mode_contract(selector_fit: str) -> tuple[bool, list[str]]:
    if selector_fit == "fitted":
        return True, list(MODE_ORDER)
    if selector_fit == "unavailable_no_exposed_train_sparse_action":
        return False, list(MODE_ORDER[1:])
    raise ValueError(f"Unsupported selector result state {selector_fit!r}.")


def _fixed_rows(table: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    by_row: dict[int, Mapping[str, Any]] = {}
    for row in table["rows"]:
        if row.get("query_panel") != "fixed" or row.get("selector_primary_fit_row") is not True:
            continue
        row_id = int(row["row_id"])
        prior = by_row.setdefault(row_id, row)
        if int(prior["module_count"]) != int(row["module_count"]):
            raise ValueError(f"Wind table changes module count across actions for row {row_id}.")
    return by_row


def choose_workload_rows(
    table: Mapping[str, Any], *, row_id: int | None = None,
    large_row_id: int | None = None,
) -> tuple[int, int]:
    fixed_rows = _fixed_rows(table)
    if len(fixed_rows) < 2:
        raise ValueError("Latency panel needs at least two fixed primary-fit native rows.")
    ordered = sorted(fixed_rows, key=lambda value: (int(fixed_rows[value]["module_count"]), value))
    base = int(row_id) if row_id is not None else ordered[0]
    if base not in fixed_rows:
        raise ValueError(f"Base row {base} is absent from the fixed primary-fit action panel.")
    large = int(large_row_id) if large_row_id is not None else max(
        ordered, key=lambda value: (int(fixed_rows[value]["module_count"]), -value)
    )
    if large not in fixed_rows:
        raise ValueError(f"Large row {large} is absent from the fixed primary-fit action panel.")
    if int(fixed_rows[large]["module_count"]) <= int(fixed_rows[base]["module_count"]):
        raise ValueError("Large-shape row must have strictly more modules than the Q-panel base row.")
    return base, large


def validate_latency_inputs(
    *, table_path: Path, run_dir: Path, update_count: int,
    selector_json_path: Path, risk_head_path: Path | None,
    row_id: int | None = None, large_row_id: int | None = None,
) -> dict[str, Any]:
    """CPU-only binding and workload preflight; performs no CUDA operation."""
    table_path = table_path.resolve()
    table = json.loads(table_path.read_text(encoding="utf-8"))
    if (
        table.get("format_version") != 1
        or table.get("run_id") != RUN_ID
        or int(table.get("selected_update_count", -1)) != int(update_count)
    ):
        raise ValueError("Action table is not the requested selected Run2112 checkpoint.")
    selected_g_path, selected_g_review, g_line_sha = panel._checkpoint_review_binding(
        run_dir.resolve(), arm="g_packet", update_count=int(update_count)
    )
    p_update = int(table.get("p_control_update_count", -1))
    selected_p_path, selected_p_review, p_line_sha = panel._checkpoint_review_binding(
        run_dir.resolve(), arm="direct_pair", update_count=p_update
    )
    g_sha, p_sha = _sha256(selected_g_path), _sha256(selected_p_path)
    if (
        table.get("forward_checkpoint_sha256") != g_sha
        or table.get("g_review_record_line_sha256") != g_line_sha
        or table.get("p_control_checkpoint_sha256") != p_sha
        or table.get("p_review_record_line_sha256") != p_line_sha
    ):
        raise ValueError("Action table no longer matches the immutable selected G/P review records.")
    selector_json_path = selector_json_path.resolve()
    selector = json.loads(selector_json_path.read_text(encoding="utf-8"))
    table_sha = _sha256(table_path)
    selector_fit = str(selector.get("selector_fit", ""))
    selector_available, modes = _selector_mode_contract(selector_fit)
    risk_head_sha = _sha256(risk_head_path.resolve()) if risk_head_path is not None else None
    if (
        selector.get("forward_checkpoint_sha256") != g_sha
        or selector.get("input_identity", {}).get("table_sha256") != table_sha
        or selector.get("input_identity", {}).get("selected_g_checkpoint_sha256") != g_sha
        or selector_available and (
            risk_head_path is None
            or selector.get("risk_head_checkpoint_sha256") != risk_head_sha
        )
    ):
        raise ValueError("Fitted selector result is not bound to this action table and selected G checkpoint.")
    if selector_available:
        risk_payload = load_trusted_checkpoint(risk_head_path.resolve(), map_location="cpu")
        if (
            not isinstance(risk_payload, Mapping)
            or int(risk_payload.get("format_version", -1)) != 1
            or risk_payload.get("forward_checkpoint_sha256") != g_sha
            or list(risk_payload.get("role_order", [])) != list(table.get("roles", []))
            or list(risk_payload.get("train_families", []))
            != list(selector.get("split_families", {}).get("train_fit", []))
        ):
            raise ValueError("Risk-head checkpoint lineage or role schema differs from its fitted report.")
    selected_g_payload = load_trusted_checkpoint(selected_g_path, map_location="cpu")
    selected_p_payload = load_trusted_checkpoint(selected_p_path, map_location="cpu")
    source_path, _source_payload, _normalizer, source_sha = runner._load_source(
        runner._load_config(Path(runner.DEFAULT_CONFIG).resolve())
    )
    expected_source_sha = str(table.get("retained_wfull_checkpoint_sha256", ""))
    split = table.get("training_split_identity", {})
    for arm, payload, expected_arm, review in (
        ("G", selected_g_payload, "g_packet", selected_g_review),
        ("P", selected_p_payload, "direct_pair", selected_p_review),
    ):
        expected_update = update_count if arm == "G" else p_update
        if (
            not isinstance(payload, Mapping)
            or payload.get("run_id") != RUN_ID
            or payload.get("arm") != expected_arm
            or int(payload.get("update_count", -1)) != expected_update
            or payload.get("source_checkpoint_sha256") != source_sha
            or payload.get("train_rows_sha256") != split.get("student_train_rows_sha256")
            or payload.get("checkpoint_sha256", review.get("checkpoint_sha256")) != review.get("checkpoint_sha256")
        ):
            raise ValueError(f"Selected {arm} checkpoint fails run/source/split/review identity.")
    if source_sha != expected_source_sha or str(source_path) != str(table.get("retained_wfull_checkpoint_path")):
        raise ValueError("Retained W-full comparison checkpoint differs from the action table.")
    feature_path = _resolve_table_path(table_path, str(table["feature_npz_path"]))
    mask_path = _resolve_table_path(table_path, str(table["mask_npz_path"]))
    _fit_rows, fit_metadata = fit_driver._load_action_table(
        table_path, feature_path, mask_path, selected_g_path
    )
    if list(fit_metadata["roles"]) != list(selector.get("role_order", [])):
        raise ValueError("Action-table and fitted selector role orders differ.")
    base_row, large_row = choose_workload_rows(table, row_id=row_id, large_row_id=large_row_id)
    fixed_rows = _fixed_rows(table)
    workloads = []
    for name, selected_row, query_count in (
        ("q64", base_row, 64),
        ("q1024", base_row, 1024),
        ("q8192", base_row, 8192),
        ("large_m_q8192", large_row, 8192),
    ):
        workloads.append({
            "name": name,
            "row_id": int(selected_row),
            "module_count": int(fixed_rows[selected_row]["module_count"]),
            "query_count": query_count,
            "role_query_counts": _query_counts(query_count),
        })
    return {
        "run_id": RUN_ID,
        "selected_update_count": int(update_count),
        "selected_p_update_count": p_update,
        "selected_g_checkpoint": str(selected_g_path),
        "selected_g_checkpoint_sha256": g_sha,
        "selected_p_checkpoint": str(selected_p_path),
        "selected_p_checkpoint_sha256": p_sha,
        "retained_wfull_checkpoint": str(source_path),
        "retained_wfull_checkpoint_sha256": source_sha,
        "action_table": str(table_path),
        "action_table_sha256": table_sha,
        "selector_results": str(selector_json_path),
        "selector_fit": selector_fit,
        "selector_available": selector_available,
        "g_action_policy_status": "available" if selector_available else selector_fit,
        "risk_head_checkpoint": str(risk_head_path.resolve()) if risk_head_path is not None else None,
        "risk_head_checkpoint_sha256": risk_head_sha,
        "base_row_id": base_row,
        "large_row_id": large_row,
        "workloads": workloads,
        "modes": modes,
        "input_materialization_and_host_to_device_transfer_included": False,
        "labels_or_target_fields_used_by_inference_wrapper": False,
    }


def _training_action_eligible(table: Mapping[str, Any], *, action: str, row_id: int) -> bool:
    if action == "full_access":
        return False
    summary = table["action_exposure_by_key"][action]
    if not bool(summary.get("trained_action")):
        return False
    complete_passes = set(map(int, summary.get("completed_primary_pass_ids", [])))
    row_entries = summary.get("resolved_path_exposure_by_row", {}).get(str(int(row_id)), [])
    covered: set[int] = set()
    for family in row_entries:
        for observation in family.get("primary_pass_observations", []):
            pass_id = int(observation["primary_pass"])
            if pass_id in complete_passes:
                covered.add(pass_id)
    return len(covered & complete_passes) >= int(summary.get("required_complete_passes", 2))


def _encode(model: Any, batch: Any) -> tuple[Any, Any]:
    core = model.core
    core.backend.set_cover_mode("external")
    encoded = core.encode_case(batch)
    tree = core.backend.build_case_trees(encoded)[0]
    return encoded, tree


def _g_action_policy_wrapper(
    *, workload: Workload, table: Mapping[str, Any],
    model: Any, organizer: Any, risk_head: ActionAwareRiskHead,
    selector: Mapping[str, Any],
) -> dict[str, Any]:
    batch, sample = workload.batch, workload.sample
    encoded, tree = _encode(model, batch)
    scores = organizer.score_cases(
        encoded,
        {
            "module_states": encoded.module_tokens,
            "environment_states": encoded.env_tokens,
            "global_state": encoded.global_token,
        },
        (tree,),
        budgets=PRIMARY_CAPACITY,
    )
    receiver_features = receiver_role_descriptors(tree, len(runner.ROLE_NAMES))
    role_limits = torch.as_tensor(
        selector["fixed_train_role_log_limits"], device=batch.query_xy.device, dtype=torch.float32
    )
    margin = torch.as_tensor(
        selector["neural_empirical_upper_margin_by_role"],
        device=batch.query_xy.device,
        dtype=torch.float32,
    )
    plans: list[Any] = []
    risks: list[torch.Tensor] = []
    exact_work: list[float] = []
    packet_counts: list[int] = []
    trained: list[bool] = []
    valid_execution: list[bool] = []
    paths_by_action: dict[str, list[str]] = {}
    for action in ACTION_ORDER:
        if action == "full_access":
            cut = (0,)
            paths: tuple[str, ...] = ()
            plan = MechanismPlan.full_access(
                tree, encoded.module_present[0], int(encoded.env_coords.shape[1])
            )
            action_k = 1
            budget = fit_driver._float_tensor([1.0, 1.0]).to(batch.query_xy.device)
            action_valid = True
        else:
            cut, paths = available_frontier_for_paths(tree, WIND_ACTION_PATHS[action])
            plan = organizer.plans_from_scores(
                scores, encoded, (tree,), hard=True, frontier_cuts=(cut,),
                budget_fractions=PRIMARY_CAPACITY,
            )[0]
            action_k = panel._action_union_k(plan, encoded, cut)
            budget = fit_driver._float_tensor(
                [PRIMARY_CAPACITY["MM"], PRIMARY_CAPACITY["QE"]]
            ).to(batch.query_xy.device)
            action_valid = action_k >= 1
        plans.append(plan)
        paths_by_action[action] = list(paths)
        metrics = panel._plan_work(
            plan=plan, encoded=encoded, tree=tree, batch=batch, sample=sample
        )
        work = float(metrics["canonical_total_work"])
        full_work = float(metrics["canonical_full_access_total_work"])
        if action != "full_access":
            action_valid = action_valid and full_work > 0.0 and work <= full_work * (1.0 + 1e-6)
            action_valid = action_valid and bool(paths)
            action_trained = _training_action_eligible(table, action=action, row_id=workload.row_id)
            strict_saving = work < full_work - max(1.0e-9, 1.0e-12 * full_work)
            trained.append(bool(action_valid and action_trained and strict_saving))
        else:
            trained.append(False)
        exact_work.append(work)
        packet_counts.append(int(action_k))
        action_features = describe_realized_plan(scores, plan, encoded, tuple(cut))
        if action_valid:
            risk = risk_head(
                action_features, budget, receiver_features, nonredundant_k=int(action_k)
            )
        else:
            risk = action_features.new_full((len(runner.ROLE_NAMES),), 1.0e6)
        risks.append(risk)
        valid_execution.append(bool(action_valid))
    predicted = torch.stack(risks)
    selection = select_action_by_risk(
        predicted,
        torch.tensor(exact_work, device=batch.query_xy.device, dtype=torch.float64),
        torch.tensor(packet_counts, device=batch.query_xy.device, dtype=torch.long),
        torch.tensor(trained, device=batch.query_xy.device, dtype=torch.bool),
        role_limits,
        full_access_index=ACTION_ORDER.index("full_access"),
        empirical_margin=margin,
    )
    chosen = ACTION_ORDER[selection.index]
    runner._prediction_with_plan(model, encoded, batch, plans[selection.index])
    return {
        "selected_action_key": chosen,
        "explicit_full_fallback": chosen == "full_access",
        "unsupported_at_budget": bool(selection.unsupported_at_budget),
        "selected_nonredundant_k": None if chosen == "full_access" else packet_counts[selection.index],
        "selected_exact_work": exact_work[selection.index],
        "selected_work_over_full": exact_work[selection.index] / max(exact_work[-1], 1.0e-12),
        "trained_sparse_action_mask": trained,
        "valid_execution_by_action": dict(zip(ACTION_ORDER, valid_execution, strict=True)),
        "resolved_paths_by_action": paths_by_action,
        "safe_sparse_action_indices": list(selection.safe_sparse_indices),
    }


def _run_wrapper(
    mode: str, *, workload: Workload, table: Mapping[str, Any],
    g_model: Any, g_control: Any, p_model: Any, p_control: Any,
    retained_model: Any, risk_head: ActionAwareRiskHead, selector: Mapping[str, Any],
) -> dict[str, Any]:
    batch = workload.batch
    if mode == "g_action_policy":
        return _g_action_policy_wrapper(
            workload=workload, table=table, model=g_model, organizer=g_control,
            risk_head=risk_head, selector=selector,
        )
    selected_model = {
        "g_full": g_model,
        "retained_wfull": retained_model,
    }.get(mode)
    if mode in {"g_full", "retained_wfull"}:
        encoded, tree = _encode(selected_model, batch)
        plan = MechanismPlan.full_access(
            tree, encoded.module_present[0], int(encoded.env_coords.shape[1])
        )
        runner._prediction_with_plan(selected_model, encoded, batch, plan)
        return {"selected_action_key": "full_access", "explicit_full_fallback": False}
    if mode == "p_direct":
        encoded, _tree = _encode(p_model, batch)
        plan, _soft, projections, _actual = maturation._direct_plan_pair_by_route(
            model=p_model, encoded=encoded, batch=batch, scorer=p_control,
            route_fractions=PRIMARY_CAPACITY,
        )
        runner._prediction_with_plan(p_model, encoded, batch, plan)
        return {
            "selected_action_key": "p_direct_primary",
            "direct_projected_pairs_by_route": {
                route: int(item.selected_unique_pairs) for route, item in projections.items()
            },
            "explicit_full_fallback": False,
        }
    raise ValueError(f"Unknown latency wrapper mode {mode!r}.")


def _prepare_workloads(
    *, plan: Mapping[str, Any], sampler: Any,
    view: Any, normalizer: Any, device: torch.device, seed: int,
) -> list[Workload]:
    workloads = []
    for index, item in enumerate(plan["workloads"]):
        row_id = int(item["row_id"])
        query_count = int(item["query_count"])
        case, sample, batch = maturation._case_batch(
            view=view,
            row_id=row_id,
            query_seed=int(seed + index * 1_000_003 + row_id),
            role_counts=_query_counts(query_count),
            sampler=sampler,
            normalizer=normalizer,
            device=device,
        )
        if int(batch.query_xy.shape[1]) != query_count:
            raise ValueError(f"Prepared {item['name']} batch does not contain exactly Q{query_count} queries.")
        workloads.append(Workload(
            name=str(item["name"]), row_id=row_id,
            module_count=int(item["module_count"]), query_count=query_count,
            role_counts=_query_counts(query_count), query_seed=int(seed + index * 1_000_003 + row_id),
            case=case, sample=sample, batch=batch,
        ))
    if not workloads or len({item.name for item in workloads}) != len(workloads):
        raise ValueError("Latency workload names must be unique and nonempty.")
    return workloads


def measure_latency_panel(
    *, workloads: Sequence[Workload], wrapper: Callable[[str, Workload], Mapping[str, Any]],
    device: torch.device, warmups: int = DEFAULT_WARMUPS, repeats: int = DEFAULT_REPEATS,
    seed: int = 2112, modes: Sequence[str] = MODE_ORDER,
) -> dict[str, Any]:
    if not 0 <= warmups <= MAX_WARMUPS or not 1 <= repeats <= MAX_REPEATS:
        raise ValueError("Synchronized latency repeats exceed the bounded diagnostic allocation.")
    if device.type != "cuda":
        raise ValueError("The selected-checkpoint native latency panel requires synchronized CUDA timing.")
    if not modes or any(mode not in MODE_ORDER for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("Latency panel modes must be a nonempty unique subset of the supported wrappers.")
    workload_names = [item.name for item in workloads]
    orders = interleaved_workload_order(
        workload_names, modes, rounds=warmups + repeats, seed=seed
    )
    samples = []
    all_observations = []
    for round_index, order in enumerate(orders):
        round_records = []
        for workload_name, mode in order:
            workload = next(item for item in workloads if item.name == workload_name)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = wrapper(mode, workload)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            record = {
                "round_index": round_index,
                "workload": workload_name,
                "mode": mode,
                "row_id": workload.row_id,
                "module_count": workload.module_count,
                "query_count": workload.query_count,
                "elapsed_seconds": elapsed,
                "synchronized_before_and_after": True,
                **dict(output),
            }
            round_records.append(record)
        all_observations.extend(round_records)
        if round_index >= warmups:
            samples.extend(round_records)
    return {
        "warmup_rounds": int(warmups),
        "measured_interleaved_rounds": int(repeats),
        "interleaved_order_by_round": [
            [{"workload": workload, "mode": mode} for workload, mode in order]
            for order in orders
        ],
        "latency_summary": summarize_latencies(samples),
        "records": samples,
        "warmup_records_excluded": sum(len(order) for order in orders[:warmups]),
        "total_wrapper_calls_including_warmup": len(all_observations),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_cuda_panel(
    *, plan: Mapping[str, Any], table_path: Path, selector_json_path: Path,
    risk_head_path: Path | None, selected_g_path: Path, selected_p_path: Path,
    run_dir: Path, update_count: int, device: torch.device, seed: int,
) -> dict[str, Any]:
    config = runner._load_config(Path(runner.DEFAULT_CONFIG).resolve())
    view, _canonical, _train_rows, split_record = runner._native_inputs(config)
    _source_path, source_payload, normalizer, source_sha = runner._load_source(config)
    table = json.loads(table_path.resolve().read_text(encoding="utf-8"))
    selector = json.loads(selector_json_path.resolve().read_text(encoding="utf-8"))
    risk_payload = (
        load_trusted_checkpoint(risk_head_path.resolve(), map_location="cpu")
        if plan["selector_available"] and risk_head_path is not None else None
    )
    g_payload = load_trusted_checkpoint(selected_g_path, map_location="cpu")
    p_payload = load_trusted_checkpoint(selected_p_path, map_location="cpu")
    sampler = runner.NativeRoleCatalogueCache()
    workloads = _prepare_workloads(
        plan=plan, sampler=sampler,
        view=view, normalizer=normalizer, device=device, seed=seed,
    )
    first_batch = workloads[0].batch
    retained_model = runner._new_model_from_source(source_payload, normalizer, first_batch, device)
    retained_model.eval()
    retained_model.core.backend.set_cover_mode("external")
    retained_model.core.backend.set_cover_executor("dense_masked")
    for parameter in retained_model.parameters():
        parameter.requires_grad_(False)
    g_model, g_control = panel._load_student(
        arm="g", payload=g_payload, source_payload=source_payload, normalizer=normalizer,
        batch=first_batch, forward_config=config["forward"], device=device,
    )
    p_model, p_control = panel._load_student(
        arm="p", payload=p_payload, source_payload=source_payload, normalizer=normalizer,
        batch=first_batch, forward_config=config["forward"], device=device,
    )
    for model in (g_model, p_model):
        model.core.backend.set_cover_mode("external")
        model.core.backend.set_cover_executor("dense_masked")
    evidence = fit_driver._load_action_table(
        table_path.resolve(),
        _resolve_table_path(table_path.resolve(), str(table["feature_npz_path"])),
        _resolve_table_path(table_path.resolve(), str(table["mask_npz_path"])),
        selected_g_path,
    )[0]
    risk_head = None
    if risk_payload is not None:
        first = evidence[0]
        risk_head = ActionAwareRiskHead(
            packet_feature_dim=first.packet_rows.shape[1],
            budget_dim=first.budget_vector.shape[0],
            receiver_role_dim=first.receiver_role_features.shape[1],
        ).to(device=device, dtype=torch.float32)
        risk_head.load_state_dict(risk_payload["state_dict"], strict=True)
        risk_head.eval()
        for parameter in risk_head.parameters():
            parameter.requires_grad_(False)
    wrapped = lambda mode, workload: _run_wrapper(
        mode, workload=workload, table=table, g_model=g_model, g_control=g_control,
        p_model=p_model, p_control=p_control, retained_model=retained_model,
        risk_head=risk_head, selector=selector,
    )
    with torch.inference_mode():
        measurement = measure_latency_panel(
            workloads=workloads, wrapper=wrapped, device=device,
            warmups=DEFAULT_WARMUPS, repeats=DEFAULT_REPEATS, seed=seed,
            modes=plan["modes"],
        )
    return {
        "status": "completed",
        "run_id": RUN_ID,
        "selected_update_count": int(update_count),
        "selected_p_update_count": int(table["p_control_update_count"]),
        "selected_g_checkpoint_sha256": _sha256(selected_g_path),
        "selected_p_checkpoint_sha256": _sha256(selected_p_path),
        "retained_wfull_checkpoint_sha256": source_sha,
        "action_table_sha256": _sha256(table_path.resolve()),
        "selector_json_sha256": _sha256(selector_json_path.resolve()),
        "risk_head_sha256": _sha256(risk_head_path.resolve()) if risk_head_path is not None else None,
        "selector_fit": plan["selector_fit"],
        "g_action_policy_status": plan["g_action_policy_status"],
        "comparison_modes": list(plan["modes"]),
        "run_dir": str(run_dir.resolve()),
        "training_rows_sha256": split_record["student_train_rows_sha256"],
        "device_name": torch.cuda.get_device_name(device),
        "device_uuid": maturation._gpu_uuid(),
        "role_order": list(runner.ROLE_NAMES),
        "workloads": [{
            "name": item.name, "row_id": item.row_id,
            "layout_index": int(view.metadata["layout_index"][item.row_id]),
            "wind_direction_deg": float(view.run(item.row_id).wind_direction_deg),
            "module_count": item.module_count, "query_count": item.query_count,
            "role_query_counts": dict(item.role_counts), "query_seed": item.query_seed,
        } for item in workloads],
        "timing_contract": {
            "clock": "time.perf_counter with CUDA synchronization immediately before and after every wrapper call",
            "g_action_policy_status": plan["g_action_policy_status"],
            "g_action_policy_timed": bool(plan["selector_available"]),
            "input_batch_creation_included": False,
            "host_to_device_transfer_included": False,
            "included_for_g_action_policy": [
                "encode_case", "organizer_score_cases", "action_path_resolution_and_plan_construction",
                "input_only_action_descriptions", "canonical_work_for_all_actions",
                "risk_head_and_calibrated_selection", "explicit_full_fallback", "selected_plan_prepare_and_decode",
            ],
            "g_full_includes": ["encode_case", "full_access_plan", "prepare_and_decode"],
            "p_direct_includes": ["encode_case", "direct_feature_preparation", "direct_pair_scoring_and_projection", "prepare_and_decode"],
            "retained_wfull_includes": ["encode_case", "full_access_plan", "prepare_and_decode"],
            "dense_masked_executor": True,
            "labels_or_targets_read": False,
        },
        **measurement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--update-count", type=int, required=True)
    parser.add_argument("--action-table", type=Path, required=True)
    parser.add_argument("--selector-results", type=Path, required=True)
    parser.add_argument("--risk-head-checkpoint", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--row-id", type=int)
    parser.add_argument("--large-row-id", type=int)
    parser.add_argument("--seed", type=int, default=2112)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.update_count <= 100 or args.seed < 0:
        raise ValueError("Latency panel needs a selected Run2112 endpoint above u100 and a nonnegative seed.")
    plan = validate_latency_inputs(
        table_path=args.action_table,
        run_dir=args.run_dir,
        update_count=args.update_count,
        selector_json_path=args.selector_results,
        risk_head_path=args.risk_head_checkpoint,
        row_id=args.row_id,
        large_row_id=args.large_row_id,
    )
    plan.update({"preflight_only": bool(args.preflight_only), "seed": int(args.seed)})
    if args.preflight_only:
        print(json.dumps(plan, sort_keys=True))
        return
    if (
        Path(sys.executable).resolve()
        != Path("/home/wanglz/miniconda3/envs/ModularDT/bin/python").resolve()
    ):
        raise RuntimeError("Selected Wind latency measurement requires the ModularDT interpreter")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "0":
        raise RuntimeError("Selected Wind latency measurement requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA RTX 6000 Ada Generation":
        raise RuntimeError("Authorized Wind RTX 6000 Ada is unavailable")
    if maturation._gpu_uuid() != maturation.DEVICE_UUID:
        raise RuntimeError("CUDA_VISIBLE_DEVICES=0 did not resolve to the authorized physical GPU")
    selected_g_path, _g_review, _g_line_sha = panel._checkpoint_review_binding(
        args.run_dir.resolve(), arm="g_packet", update_count=args.update_count
    )
    table = json.loads(args.action_table.resolve().read_text(encoding="utf-8"))
    selected_p_path, _p_review, _p_line_sha = panel._checkpoint_review_binding(
        args.run_dir.resolve(), arm="direct_pair", update_count=int(table["p_control_update_count"])
    )
    result = _run_cuda_panel(
        plan=plan, table_path=args.action_table, selector_json_path=args.selector_results,
        risk_head_path=args.risk_head_checkpoint, selected_g_path=selected_g_path,
        selected_p_path=selected_p_path, run_dir=args.run_dir,
        update_count=args.update_count, device=torch.device("cuda:0"), seed=args.seed,
    )
    output = args.output_json or args.action_table.resolve().parent / "synchronized_latency.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing latency evidence: {output}")
    _write_json(output, result)
    print(json.dumps({"status": result["status"], "output_json": str(output.resolve()),
                      "summary_count": len(result["latency_summary"])}, sort_keys=True))


if __name__ == "__main__":
    main()
