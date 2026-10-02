"""Preflight and benchmark exact selected Thermal G/P wrapper endpoints.

``preflight`` reads only saved artifacts on CPU and writes exact identity bindings.
``run`` rechecks those bindings, maps logical cuda:0 to physical GPU2, and measures
interleaved sparse G/P calls, their full-access controls, and the retained Run1804 source.
No solver, training loop, or endpoint selection is performed here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

CASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CASE_ROOT / "scripts"))

import evaluate_matured_action_selector as action_eval
import run_active_packet_stage_c as stage_c
import run_selected_weight_controls as selected_controls
from channelthermal.response_control.active_packet import ThermalCoverPlanBuilder
from channelthermal.response_control.contracts import DesignInput, RoleQuery, context_inputs
from channelthermal.response_control.latency import (
    QUERY_BATCH_SIZE,
    QUERY_COUNTS,
    RENDERER_LATENCY_FIELDS,
    RENDERER_LATENCY_NAME,
    RENDERER_MODEL_BY_VARIANT,
    capture_complete_wrapper_diagnostics,
    complete_executor_row_telemetry,
    native_dense_feature_rows,
    write_renderer_latency_csv,
)
from channelthermal.response_control.latency import (
    check_sha as _check_sha,
)
from channelthermal.response_control.latency import (
    hash_json as _hash_json,
)
from channelthermal.response_control.latency import (
    input_digest as _input_digest,
)
from channelthermal.response_control.latency import (
    interleaved as _interleaved,
)
from channelthermal.response_control.latency import (
    renderer_latency_fields as _renderer_latency_fields,
)
from channelthermal.response_control.latency import (
    role_candidates as _role_candidates,
)
from channelthermal.response_control.latency import (
    rows_by_role as _rows_by_role,
)
from channelthermal.response_control.latency import (
    scenario as _scenario,
)
from channelthermal.response_control.latency import (
    selected_ids as _selected_ids,
)
from channelthermal.response_control.latency import (
    sha256_file as _sha,
)
from channelthermal.response_control.latency import (
    sync as _sync,
)
from channelthermal.response_control.maturation import THERMAL_ACTION_PATHS, available_frontier_for_paths

FORMAT_VERSION = 1
PREFLIGHT_NAME = "latency_preflight.json"
RESULT_NAME = "latency_benchmark.json"


def _load_panel(path: Path, family_id: str, *, seed: int) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stencil, _ = stage_c.load_response_atlas_stencil(path)
    record = stencil.baseline
    actual_family = str(record.design.physical_family_id)
    if actual_family != str(family_id):
        raise ValueError(f"Requested panel family {family_id!r}; saved baseline is {actual_family!r}.")
    active_m = sum(bool(module.active) for module in record.design.modules)
    if active_m < 1:
        raise ValueError("Saved panel has no active Thermal modules.")
    candidates = _role_candidates(record, seed=seed)
    query_cap = candidates["query_cap"]
    if query_cap < 64:
        raise ValueError("Saved panel has fewer than 64 balanced valid native query rows.")
    full_role_rows = _rows_by_role(candidates, query_cap)
    return {
        "record": record,
        "path": str(path),
        "source_sha256": _sha(path),
        "family_id": actual_family,
        "active_m": active_m,
        "eligible_rows_by_role": candidates["eligible_rows_by_role"],
        "unique_eligible_rows_by_role": candidates["unique_eligible_rows_by_role"],
        "eligible_fluid_rows": candidates["eligible_rows_by_role"]["fluid_fields"],
        "unique_eligible_fluid_rows": candidates["unique_eligible_rows_by_role"]["fluid_fields"],
        "query_cap": query_cap,
        "selection_seed": int(seed),
        "candidates": candidates,
        "full_role_rows": full_role_rows,
        "rows_sha256": _hash_json([[name, int(row)] for name, rows in full_role_rows.items() for row in rows]),
        "query_ids_sha256": _hash_json(_selected_ids(record, full_role_rows)),
        "panel_sha256": _input_digest(record, full_role_rows),
    }


def _verify_preflight(preflight: Mapping[str, Any]) -> None:
    if preflight.get("format_version") != FORMAT_VERSION or preflight.get("status") != "cpu_preflight_complete":
        raise ValueError("Unsupported or incomplete Thermal latency preflight JSON.")
    reserve = int(preflight.get("run_reserve_seconds", 0))
    if not 1 <= reserve <= 7200:
        raise ValueError("Preflight run reserve must be between 1 and the 7200-second benchmark reserve.")
    scenarios = preflight.get("workload", {}).get("panels", {}).get("standard", {}).get("scenarios", ())
    if tuple(int(row["query_count"]) for row in scenarios) != QUERY_COUNTS:
        raise ValueError("Preflight must bind exact standard Q64, Q1024, and Q8192 workloads.")


def _require_large_m_panel(base_modules: int, large_modules: int) -> None:
    if large_modules != 10 or large_modules <= base_modules:
        raise ValueError("Large-M panel must have exactly 10 active modules and exceed the standard panel.")


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.controlled_manifest.expanduser().resolve()
    source_path = args.source_checkpoint.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    g_manifest, g_payload, g_sha = action_eval._load_checkpoint_binding(
        manifest_path, args.g_checkpoint.expanduser().resolve(), expected_arm="G"
    )
    p_manifest, p_payload, p_sha = action_eval._load_checkpoint_binding(
        manifest_path, args.p_checkpoint.expanduser().resolve(), expected_arm="P"
    )
    _check_sha(g_sha, args.g_checkpoint_sha256, "G endpoint")
    _check_sha(p_sha, args.p_checkpoint_sha256, "P endpoint")
    if g_manifest != p_manifest or int(g_payload["actual_optimizer_updates"]) != int(
        p_payload["actual_optimizer_updates"]
    ):
        raise ValueError("G/P endpoints must share the same controlled manifest and completed-update count.")
    source_sha = _sha(source_path)
    if source_sha != g_manifest.get("reference_checkpoint_sha256"):
        raise ValueError("Source checkpoint SHA256 differs from the controlled-run manifest.")
    gpu_index = g_manifest.get("physical_gpu_index", g_manifest.get("physical_gpu"))
    gpu_uuid = str(g_manifest.get("physical_gpu_uuid", ""))
    extra_route = str(g_manifest.get("extra_route", "")).upper()
    if gpu_index != 2 or not gpu_uuid or g_manifest.get("cuda_visible_devices") != "2":
        raise ValueError("Controlled manifest must bind physical GPU2 index, UUID, and CUDA_VISIBLE_DEVICES=2.")
    if extra_route not in {"MM", "ME"}:
        raise ValueError("Controlled manifest must bind the selected MM or ME route.")
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats must be positive.")
    if not 1 <= int(args.run_reserve_seconds) <= 7200:
        raise ValueError("run-reserve-seconds must be between 1 and the 7200-second benchmark reserve.")
    base = _load_panel(args.base_panel, args.base_family_id, seed=args.panel_seed)
    large = _load_panel(args.large_m_panel, args.large_m_family_id, seed=args.panel_seed + 1)
    _require_large_m_panel(base["active_m"], large["active_m"])
    large_q = int(args.large_m_query_count)
    if large_q not in QUERY_COUNTS:
        raise ValueError("large-m-query-count must match 64, 1024, or 8192.")
    if large_q > large["query_cap"]:
        raise ValueError(
            f"Large-M panel can supply only Q{large['query_cap']} across eligible unique native roles; "
            f"Q{large_q} would require duplicated or fabricated rows."
        )
    if base["query_cap"] < 8192:
        raise ValueError(
            "Standard panel cannot supply total Q8192 across unique valid native roles: "
            f"cap=Q{base['query_cap']}, unique_rows_by_role={base['unique_eligible_rows_by_role']}. "
            "Choose another saved panel; do not pad or duplicate rows."
        )
    standard_counts = QUERY_COUNTS
    panels = {}
    for key, panel, counts in (
        ("standard", base, standard_counts),
        ("large_m", large, (large_q,)),
    ):
        panels[key] = {
            "path": panel["path"],
            "source_sha256": panel["source_sha256"],
            "family_id": panel["family_id"],
            "active_module_count": panel["active_m"],
            "eligible_fluid_rows": panel["eligible_fluid_rows"],
            "unique_eligible_fluid_rows": panel["unique_eligible_fluid_rows"],
            "eligible_rows_by_role": panel["eligible_rows_by_role"],
            "unique_eligible_rows_by_role": panel["unique_eligible_rows_by_role"],
            "total_unique_eligible_native_rows": sum(panel["unique_eligible_rows_by_role"].values()),
            "query_cap": panel["query_cap"],
            "requested_high_query_count": 8192,
            "high_query_count_shortfall": max(0, 8192 - panel["query_cap"]),
            "selection_seed": panel["selection_seed"],
            "selected_rows_sha256": panel["rows_sha256"],
            "selected_query_ids_sha256": panel["query_ids_sha256"],
            "max_feasible_input_query_panel_digest": panel["panel_sha256"],
            "scenarios": [_scenario(panel, key, q) for q in counts],
        }
    preflight: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "cpu_preflight_complete",
        "run_reserve_seconds": int(args.run_reserve_seconds),
        "controlled_run": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha(manifest_path),
            "run_id": g_manifest.get("run_id"),
            "extra_route": extra_route,
            "source_checkpoint_path": str(source_path),
            "source_checkpoint_sha256": source_sha,
            "G": {
                "path": str(args.g_checkpoint.expanduser().resolve()),
                "sha256": g_sha,
                "update": int(g_payload["actual_optimizer_updates"]),
            },
            "P": {
                "path": str(args.p_checkpoint.expanduser().resolve()),
                "sha256": p_sha,
                "update": int(p_payload["actual_optimizer_updates"]),
            },
        },
        "gpu2": {
            "physical_index": 2,
            "uuid": gpu_uuid,
            "name": g_manifest.get("physical_gpu_name"),
            "visible_devices": "2",
        },
        "timing": {
            "precision": "float32",
            "query_batch_size": QUERY_BATCH_SIZE,
            "warmups_per_variant": int(args.warmups),
            "repeats_per_variant": int(args.repeats),
            "seed": int(args.seed),
            "interleaving": "seeded shuffled G/P sparse/full plus retained Run1804 source full-access round per repeat",
            "synchronize_before_and_after_each_call": True,
            "latency_statistic": "p50/p90 linear quantiles over measured samples",
        },
        "workload": {
            "q_definition": "total eligible unique native query rows summed across fluid_fields, interface, and solid_temperature",
            "role_selection": (
                "deterministic fluid-first rows up to the largest native-valid count; fill remaining Q with "
                "balanced interface rows by active module and shared-order solid rows across active modules"
            ),
            "composition_warning": "role counts and quadrature weights are reported per shape; Q comparisons also change role composition",
            "panels": panels,
            "budgets": {"QE": 0.90, extra_route: 0.90},
            "cut": "trained four_packet path action resolved on each native receiver tree",
            "no_padding_or_duplicate_query_coordinates": True,
        },
        "scope": {
            "solver_calls": 0,
            "training_updates": 0,
            "complete_call": "complete operator call from prebuilt device tensors: encoding + G/P plan scoring and hard organization + native preparation + decode; retained Run1804 source is measured as the native full-access control",
            "input_tensor_setup_timed": False,
            "full_access": "separate direct native calls for G/P endpoints and the retained Run1804 source with no organizer or soft-plan branch",
            "support_work_and_executor_rows_reported_separately": True,
            "risk_head_decision": "not measured unless a separately fitted selector is bound at final endpoint review",
        },
    }
    return preflight


def _load_preflight_panel(item: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(item["path"])).expanduser().resolve()
    if _sha(path) != item["source_sha256"]:
        raise ValueError(f"Saved panel changed after preflight: {path}")
    case = _load_panel(path, str(item["family_id"]), seed=int(item["selection_seed"]))
    if (
        case["active_m"] != int(item["active_module_count"])
        or case["rows_sha256"] != item["selected_rows_sha256"]
        or case["query_ids_sha256"] != item["selected_query_ids_sha256"]
        or case["panel_sha256"] != item["max_feasible_input_query_panel_digest"]
        or case["query_cap"] != int(item["query_cap"])
    ):
        raise ValueError(f"Saved panel contents no longer match the CPU preflight record: {path}")
    return case


def _preflight_checkpoints(preflight: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    control = preflight["controlled_run"]
    manifest_path = Path(control["manifest_path"])
    if _sha(manifest_path) != control["manifest_sha256"]:
        raise ValueError("Controlled-run manifest changed after preflight.")
    if _sha(Path(control["source_checkpoint_path"])) != control["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed after preflight.")
    gm, gp, gh = action_eval._load_checkpoint_binding(manifest_path, Path(control["G"]["path"]), expected_arm="G")
    pm, pp, ph = action_eval._load_checkpoint_binding(manifest_path, Path(control["P"]["path"]), expected_arm="P")
    if gm != pm or gh != control["G"]["sha256"] or ph != control["P"]["sha256"]:
        raise ValueError("G/P checkpoint binding changed after preflight.")
    updates = int(control["G"]["update"])
    if int(gp["actual_optimizer_updates"]) != updates or int(pp["actual_optimizer_updates"]) != updates:
        raise ValueError("G/P completed-update binding changed after preflight.")
    if str(gm.get("physical_gpu_uuid", "")) != preflight["gpu2"]["uuid"]:
        raise ValueError("Manifest physical GPU2 UUID changed after preflight.")
    return gm, gp, pp


def _role_queries(record: Any, role_rows: Mapping[str, Sequence[int]], device: torch.device) -> Mapping[str, RoleQuery]:
    index_by_id = {module.module_id: i for i, module in enumerate(record.design.modules)}
    result: dict[str, RoleQuery] = {}
    for name, role in record.output.roles.items():
        indices = np.asarray(role_rows.get(name, ()), dtype=np.int64)
        features = torch.as_tensor(np.array(role.query_features[indices], dtype=np.float32, copy=True), device=device)
        slots = None
        if role.receiver_module_ids is not None:
            slots = tuple(index_by_id[str(role.receiver_module_ids[int(row)])] for row in indices)
        result[name] = RoleQuery(
            role=name,
            query_features=features,
            channel_names=tuple(role.channel_names),
            channel_units=tuple(role.channel_units),
            receiver_slots=slots,
            coordinate_kind=role.coordinate_kind,
        )
    return result


def _sparse_builder(arm: str, model: Any, route: Any, extra_route: str) -> Any:
    def select(_case: int, _encoded: Any, _state: torch.Tensor, tree: Any) -> tuple[int, ...]:
        cut, _resolved = available_frontier_for_paths(tree, THERMAL_ACTION_PATHS["four_packet"])
        return cut

    builder = ThermalCoverPlanBuilder(
        core=model.core,
        budget_fractions={"QE": 0.90, extra_route: 0.90},
        mode=arm,
        organizer=route if arm == "G" else None,
        direct_scorer=route if arm == "P" else None,
        extra_route=extra_route,
        frontier_selector=select,
    )
    builder.hard = True
    return builder


def _route_metrics(record: Mapping[str, Any], *, full: bool) -> dict[str, Any]:
    # Reuse the selected-control serializer, then split support and work fields.
    canonical = selected_controls._canonical_work(record)
    routes = canonical["by_mechanism"]
    support = {
        name: {
            "selected_unique_pairs": row["selected_unique_pairs"],
            "eligible_unique_pairs": row["eligible_canonical_pairs"],
            "support_fraction": row["canonical_pair_density"],
            "executor": row["executor_label"],
        }
        for name, row in routes.items()
    }
    work = {
        name: {
            "canonical_work_fraction": row["canonical_work_fraction"],
            "selected_unique_pairs": row["selected_unique_pairs"],
            "eligible_unique_pairs": row["eligible_canonical_pairs"],
        }
        for name, row in routes.items()
    }
    total = canonical["all_mechanisms"]
    return {
        "support": support,
        "canonical_work": work,
        "all_mechanisms_support": {
            "selected_unique_pairs": total["selected_unique_pairs"],
            "eligible_unique_pairs": total["eligible_canonical_pairs"],
            "support_fraction": (
                total["selected_unique_pairs"] / total["eligible_canonical_pairs"]
                if total["eligible_canonical_pairs"]
                else 0.0
            ),
        },
        "all_mechanisms_canonical_work": {
            "achieved_work": total["canonical_achieved_work"],
            "full_access_work": total["canonical_full_access_work"],
            "canonical_work_fraction": total["canonical_work_fraction"],
        },
        "full_access_fallback": full,
    }


def _latency_packet_support_summary(
    builder: Any,
    state_capture: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize grouped packets only when the selected plan has node masks.

    P stores exact MM receiver/source access and a dynamic QE direct policy.
    Its own route record marks packet K as not applicable; asking the grouped
    G helper to rebuild MM node memberships correctly raises for this plan.
    """

    if getattr(builder, "mode", None) == "P":
        return {
            "raw_packet_cut_size": int(record.get("raw_frontier_k", len(record.get("frontier", ())))),
            "nonredundant_packet_count": record.get("nonredundant_k"),
            "nonredundant_packet_count_status": str(
                record.get("nonredundant_k_status", "not_applicable_to_direct_pair_receiver_axis")
            ),
            "nonredundant_packet_count_definition": (
                "P uses exact direct-pair access and has no node-membership packet K"
            ),
        }
    return selected_controls._packet_support_summary(state_capture, record)


def _diagnostics(
    label: str,
    op: Any,
    call: Callable[[], Any],
    builder: Any | None,
    device: torch.device,
    *,
    expected_field_chunks: int,
    identity_binding: Mapping[str, Any],
) -> dict[str, Any]:
    hook = stage_c._capture_cover_diagnostics(op)
    capture_before = bool(op.capture_packet_inputs)
    op.capture_packet_inputs = builder is None
    try:
        with capture_complete_wrapper_diagnostics(op) as wrapper_trace:
            _sync(device)
            with torch.inference_mode():
                output = call()
            _sync(device)
            del output
        diagnostic = dict(getattr(op, "_active_packet_latest_diagnostics", {}))
        first_call_rows = selected_controls._route_live_rows(diagnostic)
        executor_rows = complete_executor_row_telemetry(
            wrapper_trace,
            expected_field_chunks=expected_field_chunks,
            full_access=builder is None,
        )
        feature_rows = native_dense_feature_rows(
            wrapper_trace,
            identity_binding=identity_binding,
            expected_field_chunks=expected_field_chunks,
            full_access=builder is None,
        )
        if builder is None:
            capture = op.last_packet_inputs
            if not isinstance(capture, Mapping) or not capture.get("trees"):
                raise RuntimeError("Full-access capture omitted its receiver tree.")
            record = stage_c._full_work_record(capture["encoded"], capture["trees"][0])
            packet = selected_controls._full_access_packet_count_fields()
        else:
            record = builder.last_records[0]
            state_capture = {
                "baseline": {
                    "encoded": builder.last_encoded,
                    "trees": builder.last_trees,
                    "plans": builder.last_plans,
                    "runtime_diagnostics": diagnostic,
                }
            }
            packet = _latency_packet_support_summary(builder, state_capture, record)
        return {
            "variant": label,
            "packet_support": packet,
            **_route_metrics(record, full=builder is None),
            "actual_executor_rows": executor_rows["by_mechanism"],
            "executor_row_telemetry": executor_rows,
            "observed_native_dense_feature_rows": feature_rows,
            "first_native_call_executor_rows": first_call_rows,
            "full_access_bypass_routes": list(record.get("full_access_bypass_routes", ())),
        }
    finally:
        op.capture_packet_inputs = capture_before
        op.last_packet_inputs = None
        hook.remove()


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _append_run_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "kernel_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
            "pid": os.getpid(),
            **event,
        }
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_identity(preflight_path: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preflight_path": str(preflight_path),
        "run_id": preflight["controlled_run"]["run_id"],
        "source_checkpoint_sha256": preflight["controlled_run"]["source_checkpoint_sha256"],
        "endpoint_checkpoint_sha256": {arm: preflight["controlled_run"][arm]["sha256"] for arm in ("G", "P")},
        "panels": {
            key: {
                "path": item["path"],
                "source_sha256": item["source_sha256"],
                "family_id": item["family_id"],
                "active_module_count": item["active_module_count"],
                "scenarios": [
                    {
                        "shape_id": shape["shape_id"],
                        "query_count": shape["query_count"],
                        "input_query_panel_digest": shape["input_query_panel_digest"],
                        "role_query_counts": shape["role_query_counts"],
                    }
                    for shape in item["scenarios"]
                ],
            }
            for key, item in preflight["workload"]["panels"].items()
        },
    }


def _telemetry_identity_binding(
    preflight: Mapping[str, Any], *, panel_key: str, panel: Mapping[str, Any], shape: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    controlled = preflight["controlled_run"]
    endpoint_field = {
        "G_sparse": "G", "G_full_access_fallback": "G",
        "P_sparse": "P", "P_full_access_fallback": "P",
        "B_retained_reference": None,
    }
    if variant not in endpoint_field:
        raise ValueError(f"Unknown latency variant for identity binding: {variant!r}.")
    endpoint = endpoint_field[variant]
    return {
        "run_id": controlled["run_id"],
        "controlled_manifest_sha256": controlled["manifest_sha256"],
        "source_checkpoint_sha256": controlled["source_checkpoint_sha256"],
        "endpoint_checkpoint_sha256": controlled["source_checkpoint_sha256"] if endpoint is None else controlled[endpoint]["sha256"],
        "panel_role": panel_key,
        "panel_path": panel["path"],
        "panel_sha256": panel["source_sha256"],
        "shape_id": shape["shape_id"],
        "query_count": int(shape["query_count"]),
        "selected_rows_sha256": shape["selected_rows_sha256"],
        "selected_query_ids_sha256": shape["selected_query_ids_sha256"],
        "input_query_panel_digest": shape["input_query_panel_digest"],
        "variant": variant,
    }

def _execute_run(
    preflight_path: Path,
    preflight: Mapping[str, Any],
    *,
    started_monotonic: float,
) -> dict[str, Any]:
    output_path = preflight_path.parent / RESULT_NAME
    renderer_csv_path = preflight_path.parent / RENDERER_LATENCY_NAME
    for artifact in (output_path, renderer_csv_path):
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite benchmark evidence {artifact}.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "2" or not torch.cuda.is_available():
        raise RuntimeError("Run requires CUDA_VISIBLE_DEVICES=2 and available logical cuda:0.")
    device = torch.device("cuda:0")
    manifest, g_payload, p_payload = _preflight_checkpoints(preflight)
    gpu = selected_controls._validate_cuda_gpu_binding(
        manifest,
        device,
        cuda_visible_devices=visible,
        queried_identity=selected_controls._physical_gpu2_identity(),
        lane_rows=selected_controls._load_lane_execution_rows(Path(preflight["controlled_run"]["manifest_path"])),
    )
    if gpu["physical_gpu_uuid"] != preflight["gpu2"]["uuid"]:
        raise ValueError("Runtime GPU2 UUID differs from the CPU preflight JSON.")
    panels = {key: _load_preflight_panel(item) for key, item in preflight["workload"]["panels"].items()}
    _require_large_m_panel(panels["standard"]["active_m"], panels["large_m"]["active_m"])
    standard = panels["standard"]
    probe_record = standard["record"]
    probe_role_rows = _rows_by_role(standard["candidates"], 64)
    source_path = Path(preflight["controlled_run"]["source_checkpoint_path"])
    probe_model, source_payload, _dataset_path, template = action_eval._load_source_and_template(source_path, device)
    probe_op = action_eval._new_operator(probe_model, source_payload, template, capture=True)
    with torch.inference_mode():
        probe_op(
            DesignInput.from_state(probe_record.design, device=device),
            context_inputs(probe_record.context),
            _role_queries(probe_record, probe_role_rows, device),
        )
    encoded_probe = probe_op.last_packet_inputs["encoded"]
    source_model, _ = stage_c.load_model(source_path, device)
    stage_c._configure_native_expanded_response_interface_scope(source_model)
    source_model.eval().requires_grad_(False)
    source_op = action_eval._new_operator(source_model, source_payload, template, capture=False)
    g_model, g_route, g_op = selected_controls._load_arm_runtime(
        "G",
        g_payload,
        source_payload=source_payload,
        model=probe_model,
        encoded_probe=encoded_probe,
        template=template,
        device=device,
    )
    p_model, p_route, p_op = selected_controls._load_native_arm_state(
        "P",
        p_payload,
        source_path=source_path,
        source_payload=source_payload,
        encoded_probe=encoded_probe,
        template=template,
        device=device,
    )
    runtimes = {
        "G": {"model": g_model, "route": g_route, "operator": g_op},
        "P": {"model": p_model, "route": p_route, "operator": p_op},
    }
    initial_hashes = {
        "B": {"physical": stage_c._state_hash(source_model.state_dict())},
        **{
            arm: {
                "physical": stage_c._state_hash(row["model"].state_dict()),
                "route": stage_c._state_hash(row["route"].state_dict()),
            }
            for arm, row in runtimes.items()
        },
    }
    for row in runtimes.values():
        hook = getattr(row["operator"], "_selected_weight_diagnostics_hook", None)
        if hook is not None:
            hook.remove()
    extra = str(preflight["controlled_run"]["extra_route"])
    measurements = []
    for panel_key in ("standard", "large_m"):
        panel = panels[panel_key]
        record = panel["record"]
        for shape in preflight["workload"]["panels"][panel_key]["scenarios"]:
            q = int(shape["query_count"])
            role_rows = _rows_by_role(panel["candidates"], q)
            digest = _input_digest(record, role_rows)
            if digest != shape["input_query_panel_digest"]:
                raise ValueError(f"Input/query digest changed for {shape['shape_id']}.")
            design = DesignInput.from_state(record.design, device=device)
            context = context_inputs(record.context)
            queries = _role_queries(record, role_rows, device)
            expected_field_chunks = (
                len(queries["fluid_fields"].query_features) + QUERY_BATCH_SIZE - 1
            ) // QUERY_BATCH_SIZE
            builders = {
                arm: _sparse_builder(arm, runtime["model"], runtime["route"], extra)
                for arm, runtime in runtimes.items()
            }
            calls: dict[str, Callable[[], Any]] = {}
            for arm, runtime in runtimes.items():
                op = runtime["operator"]
                builder = builders[arm]
                calls[f"{arm}_sparse"] = (
                    lambda op=op, builder=builder, design=design, context=context, queries=queries: op(
                        design, context, queries, cover_plan_builder=builder
                    )
                )
                calls[f"{arm}_full_access_fallback"] = lambda op=op, design=design, context=context, queries=queries: (
                    op(design, context, queries)
                )
            calls["B_retained_reference"] = lambda op=source_op, design=design, context=context, queries=queries: op(
                design, context, queries
            )
            telemetry = {}
            for arm, runtime in runtimes.items():
                telemetry[f"{arm}_sparse"] = _diagnostics(
                    f"{arm}_sparse",
                    runtime["operator"],
                    calls[f"{arm}_sparse"],
                    builders[arm],
                    device,
                    expected_field_chunks=expected_field_chunks,
                    identity_binding=_telemetry_identity_binding(
                        preflight, panel_key=panel_key, panel=panel, shape=shape, variant=f"{arm}_sparse"
                    ),
                )
                telemetry[f"{arm}_full_access_fallback"] = _diagnostics(
                    f"{arm}_full_access_fallback",
                    runtime["operator"],
                    calls[f"{arm}_full_access_fallback"],
                    None,
                    device,
                    expected_field_chunks=expected_field_chunks,
                    identity_binding=_telemetry_identity_binding(
                        preflight, panel_key=panel_key, panel=panel, shape=shape, variant=f"{arm}_full_access_fallback"
                    ),
                )
            telemetry["B_retained_reference"] = _diagnostics(
                "B_retained_reference",
                source_op,
                calls["B_retained_reference"],
                None,
                device,
                expected_field_chunks=expected_field_chunks,
                identity_binding=_telemetry_identity_binding(
                    preflight, panel_key=panel_key, panel=panel, shape=shape, variant="B_retained_reference"
                ),
            )
            timing = _interleaved(
                calls,
                device=device,
                warmups=int(preflight["timing"]["warmups_per_variant"]),
                repeats=int(preflight["timing"]["repeats_per_variant"]),
                seed=int(preflight["timing"]["seed"]) + len(measurements) * 991,
                deadline_monotonic=started_monotonic + int(preflight["run_reserve_seconds"]),
            )
            sync_group = f"Thermal-{preflight['controlled_run']['run_id']}-{shape['shape_id']}-{digest[:16]}"
            for variant, stats in timing["summary"].items():
                model_label, access_mode = RENDERER_MODEL_BY_VARIANT[variant]
                arm = variant[0]
                checkpoint_sha = (
                    preflight["controlled_run"]["source_checkpoint_sha256"]
                    if variant == "B_retained_reference"
                    else preflight["controlled_run"][arm]["sha256"]
                )
                measurements.append(
                    {
                        **_renderer_latency_fields(
                            model=model_label,
                            checkpoint_sha256=checkpoint_sha,
                            access_mode=access_mode,
                            stats=stats,
                            input_panel_sha256=digest,
                            sync_group=sync_group,
                        ),
                        "shape_id": shape["shape_id"],
                        "panel_role": panel_key,
                        "active_module_count": int(shape["active_module_count"]),
                        "query_count": q,
                        "role_query_counts": shape["role_query_counts"],
                        "role_quadrature_weights": shape["role_quadrature_weights"],
                        "input_query_panel_digest": digest,
                        "variant": variant,
                        "arm": arm,
                        "execution_mode": "sparse_hard_plan" if access_mode == "sparse" else "full_access",
                        "latency": stats,
                        "support_work_and_executor_rows": telemetry[variant],
                        "interleaved_order_trace": timing["order_trace"],
                    }
                )
    final_hashes = {
        "B": {"physical": stage_c._state_hash(source_model.state_dict())},
        **{
            arm: {
                "physical": stage_c._state_hash(row["model"].state_dict()),
                "route": stage_c._state_hash(row["route"].state_dict()),
            }
            for arm, row in runtimes.items()
        },
    }
    if initial_hashes != final_hashes:
        raise RuntimeError("Benchmark changed a frozen G/P physical or route state.")
    if time.monotonic() >= started_monotonic + int(preflight["run_reserve_seconds"]):
        raise TimeoutError("Thermal latency run exceeded its reserved wall-clock limit.")
    representative_shape = next(
        shape["shape_id"]
        for shape in preflight["workload"]["panels"]["standard"]["scenarios"]
        if int(shape["query_count"]) == 1024
    )
    write_renderer_latency_csv(renderer_csv_path, measurements, representative_shape)
    result = {
        "format_version": FORMAT_VERSION,
        "status": "completed_synchronized_selected_weight_latency",
        "preflight_path": str(preflight_path),
        "gpu_binding": gpu,
        "checkpoint_bindings": preflight["controlled_run"],
        "panels": preflight["workload"]["panels"],
        "timing": preflight["timing"],
        "weight_state_hashes": {"before": initial_hashes, "after": final_hashes, "unchanged": True},
        "measurements": measurements,
        "renderer_latency_csv_columns": list(RENDERER_LATENCY_FIELDS),
        "renderer_latency_csv": {
            "path": str(renderer_csv_path),
            "sha256": _sha(renderer_csv_path),
            "representative_shape_id": representative_shape,
        },
        "interpretation": {
            "support_and_weighted_work_are_separate": True,
            "full_access_is_reported_as_a_separate_fallback": True,
            "executor_rows_are_measured_or_explicitly_unavailable": True,
            "no_solver_or_optimizer_update": True,
            "risk_head_decision": preflight["scope"]["risk_head_decision"],
        },
    }
    _write_json(output_path, result)
    return result


def _run(preflight_path: Path) -> dict[str, Any]:
    preflight_path = preflight_path.expanduser().resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    _verify_preflight(preflight)
    ledger_path = preflight_path.parent / "latency_run_ledger.jsonl"
    started = time.monotonic()
    identity = _run_identity(preflight_path, preflight)
    _append_run_event(
        ledger_path,
        {
            "event": "start",
            "status": "running",
            "elapsed_monotonic_seconds": 0.0,
            "reserve_seconds": int(preflight["run_reserve_seconds"]),
            "solver_calls": 0,
            "optimizer_updates": 0,
            **identity,
        },
    )
    try:
        result = _execute_run(preflight_path, preflight, started_monotonic=started)
    except BaseException as exc:
        _append_run_event(
            ledger_path,
            {
                "event": "failure",
                "status": "failed",
                "elapsed_monotonic_seconds": float(time.monotonic() - started),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "solver_calls": 0,
                "optimizer_updates": 0,
                **identity,
            },
        )
        raise
    _append_run_event(
        ledger_path,
        {
            "event": "stop",
            "status": "completed",
            "elapsed_monotonic_seconds": float(time.monotonic() - started),
            "solver_calls": 0,
            "optimizer_updates": 0,
            **identity,
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="CPU-only identity/workload record; no CUDA initialization or model call.")
    pre.add_argument("--controlled-manifest", type=Path, required=True)
    pre.add_argument("--source-checkpoint", type=Path, required=True)
    pre.add_argument("--g-checkpoint", type=Path, required=True)
    pre.add_argument("--g-checkpoint-sha256", required=True)
    pre.add_argument("--p-checkpoint", type=Path, required=True)
    pre.add_argument("--p-checkpoint-sha256", required=True)
    pre.add_argument("--base-panel", type=Path, required=True)
    pre.add_argument("--base-family-id", required=True)
    pre.add_argument("--large-m-panel", type=Path, required=True)
    pre.add_argument("--large-m-family-id", required=True)
    pre.add_argument("--large-m-query-count", type=int, choices=QUERY_COUNTS, default=1024)
    pre.add_argument("--output-dir", type=Path, required=True)
    pre.add_argument("--panel-seed", type=int, default=2317)
    pre.add_argument("--warmups", type=int, default=3)
    pre.add_argument("--repeats", type=int, default=30)
    pre.add_argument("--seed", type=int, default=7019)
    pre.add_argument("--run-reserve-seconds", type=int, default=7200)
    run = sub.add_parser("run", help="Execute one preflight-bound benchmark on physical GPU2.")
    run.add_argument("--preflight-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing benchmark evidence: {output_dir}")
        preflight = _preflight(args)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / PREFLIGHT_NAME
        _write_json(path, preflight)
        print(
            json.dumps(
                {
                    "status": preflight["status"],
                    "preflight_json": str(path),
                    "physical_gpu2_uuid": preflight["gpu2"]["uuid"],
                    "checkpoint_shas": {arm: preflight["controlled_run"][arm]["sha256"] for arm in ("G", "P")},
                    "shapes": [
                        row["shape_id"]
                        for panel in preflight["workload"]["panels"].values()
                        for row in panel["scenarios"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = _run(args.preflight_json)
    print(
        json.dumps(
            {
                "status": result["status"],
                "results_path": str(Path(args.preflight_json).expanduser().resolve().parent / RESULT_NAME),
                "measurement_rows": len(result["measurements"]),
                "physical_gpu2_uuid": result["gpu_binding"]["physical_gpu_uuid"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
