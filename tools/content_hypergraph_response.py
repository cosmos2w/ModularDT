"""Replay saved counted Thermal heat transfers through a trained content model.

This is a forward-only, dataset-free checkpoint evaluation. It reads the exact
saved fixed-panel request and records, prepares each native geometry once, and
uses the retained factorized affine response for the baseline and signed heat
transfers. It never runs a physical solver or an optimizer.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

DEFAULT_REQUEST = Path(
    "/data/wanglz/ModularDT/thermal_development/tree_faithfulness_20261004/"
    "follow_on_reference_request_fixed4.json"
)
DEFAULT_RECORDS = Path(
    "/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/reference/records"
)
CASE_IDS = ("0291", "0294", "0687")
STATE_LABELS = ("baseline", "transfer_minus", "transfer_plus")
ROLE_OUTPUTS = {
    "fluid_fields": "pred_field",
    "interface": "pred_interface",
    "solid_temperature": "pred_internal_temperature",
}
TEMPERATURE_CHANNELS = {
    "fluid_fields": "temperature",
    "interface": "T_surface",
    "solid_temperature": "temperature",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _tensor(value: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


def _typed_native_input(record: Any, *, device: torch.device) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor,
                                                                         dict[str, np.ndarray]]:
    """Build geometry/context and native query roles without reading targets."""
    from channelthermal.content_hypergraph import THERMAL_CONTENT_NATIVE_THETA_COUNT

    modules = record.design.modules
    context = record.context.values
    module_ids = tuple(record.design.active_module_ids)
    if tuple(module.module_id for module in modules) != module_ids:
        raise ValueError("Thermal source slots and active module IDs are not in a stable typed order.")
    centers_np = np.asarray([module.position_xy for module in modules], dtype=np.float32)
    if centers_np.shape != (len(module_ids), 2):
        raise ValueError("Thermal module centers do not match the typed active-module catalogue.")

    # The scene whitelist contains geometry and operating context only. Heat is
    # passed independently to the affine response application below.
    boundary_values = {name: float(context.get(name, 0.0)) for name in ("t_in", "t_wall")}
    if any(value != 0.0 for value in boundary_values.values()):
        raise ValueError("The affine Thermal response replay requires the saved inlet and wall temperatures to be zero.")
    structure = {
        "module_centers": _tensor(centers_np[None], device=device),
        "module_present": _tensor(np.ones((1, len(module_ids)), dtype=np.float32), device=device),
        "module_source_ids": module_ids,
        "material_params": _tensor(
            [[context[name] for name in ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius")]],
            device=device,
        ),
        "re": _tensor([[context["re"]]], device=device),
        "u_in": _tensor([[context["u_in"]]], device=device),
        "domain_length_x": _tensor([[context["domain_length_x"]]], device=device),
        "domain_length_y": _tensor([[context["domain_length_y"]]], device=device),
        "t_in": _tensor([[boundary_values["t_in"]]], device=device),
        "t_wall": _tensor([[boundary_values["t_wall"]]], device=device),
    }
    roles = record.output.roles
    fluid_role = roles["fluid_fields"]
    fluid_xy_np = np.asarray(fluid_role.query_features[:, :2], dtype=np.float32)
    fluid_xy = _tensor(fluid_xy_np[None], device=device)

    surface_role = roles["interface"]
    surface_receivers = np.asarray(surface_role.receiver_module_ids)
    interface_rows = [np.flatnonzero(surface_receivers == module_id) for module_id in module_ids]
    if len({len(rows) for rows in interface_rows}) != 1 or len(interface_rows[0]) != THERMAL_CONTENT_NATIVE_THETA_COUNT:
        raise ValueError("Counted response surfaces must retain all 64 native angular receivers per module.")
    expected_theta = np.arange(THERMAL_CONTENT_NATIVE_THETA_COUNT, dtype=np.float64) * (
        2.0 * np.pi / THERMAL_CONTENT_NATIVE_THETA_COUNT
    )
    for rows in interface_rows:
        if not np.allclose(surface_role.query_features[rows, 0], expected_theta, rtol=0.0, atol=2.0e-6):
            raise ValueError("Counted response surface receivers do not preserve the native angle order.")

    material_role = roles["solid_temperature"]
    material_receivers = np.asarray(material_role.receiver_module_ids)
    material_rows = [np.flatnonzero(material_receivers == module_id) for module_id in module_ids]
    if len({len(rows) for rows in material_rows}) != 1 or any(len(rows) == 0 for rows in material_rows):
        raise ValueError("Counted response material grids require equal nonempty per-module receiver counts.")
    material_xy = np.stack(
        [np.asarray(material_role.query_features[rows, :2], dtype=np.float32) for rows in material_rows]
    )
    local_query_points = _tensor(material_xy[None], device=device)

    row_order = {
        "fluid_fields": np.arange(len(fluid_role.query_features), dtype=np.int64),
        "interface": np.concatenate(interface_rows).astype(np.int64, copy=False),
        "solid_temperature": np.concatenate(material_rows).astype(np.int64, copy=False),
    }
    role_ids = {
        "fluid_fields": None,
        "interface": surface_receivers,
        "solid_temperature": material_receivers,
    }
    if any(len(np.unique(indices)) != len(indices) for indices in row_order.values()):
        raise ValueError("Native receiver ordering contains repeated row ownership.")
    for role_name, order in row_order.items():
        role = roles[role_name]
        if len(order) != len(role.query_features) or (role_ids[role_name] is not None and
                                                       len(role_ids[role_name]) != len(order)):
            raise ValueError(f"Typed native receiver join is incomplete for {role_name}.")

    surface_xy_grouped = np.concatenate([
        centers_np[index][None, :] + float(context["module_radius"]) * np.stack((
            np.cos(surface_role.query_features[rows, 0]),
            np.sin(surface_role.query_features[rows, 0]),
        ), axis=-1)
        for index, rows in enumerate(interface_rows)
    ], axis=0)
    surface_xy = np.empty_like(surface_xy_grouped)
    surface_xy[row_order["interface"]] = surface_xy_grouped
    material_xy_grouped = np.concatenate([
        centers_np[index][None, :] + float(context["module_radius"]) * material_role.query_features[rows, :2]
        for index, rows in enumerate(material_rows)
    ], axis=0)
    material_xy = np.empty_like(material_xy_grouped)
    material_xy[row_order["solid_temperature"]] = material_xy_grouped
    metadata = {
        "module_ids": module_ids,
        "centers_xy": centers_np,
        "fluid_xy": fluid_xy_np,
        "interface_rows_in_model_order": row_order["interface"],
        "material_rows_in_model_order": row_order["solid_temperature"],
        "interface_angles": np.asarray(surface_role.query_features[:, 0], dtype=np.float64),
        "surface_xy": surface_xy,
        "material_local_xy": np.asarray(material_role.query_features[:, :2], dtype=np.float32),
        "material_xy": material_xy,
        "role_row_order": row_order,
        "material_row_counts": [len(rows) for rows in material_rows],
    }
    return structure, fluid_xy, local_query_points, metadata


def _native_role_arrays(output: dict[str, torch.Tensor], metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for role_name, output_name in ROLE_OUTPUTS.items():
        values = output[output_name].detach().cpu().numpy()
        if role_name == "fluid_fields":
            result[role_name] = np.asarray(values[0], dtype=np.float64)
        else:
            flat = np.asarray(values[0], dtype=np.float64).reshape(-1, values.shape[-1])
            order_key = "interface_rows_in_model_order" if role_name == "interface" else "material_rows_in_model_order"
            order = metadata[order_key]
            original = np.empty_like(flat)
            original[order] = flat
            result[role_name] = original
    return result


def _weighted_change_metrics(prediction: np.ndarray, reference: np.ndarray, role: Any, channel: int) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    # Typed saved records expose immutable arrays. Refining this local metric
    # mask must not mutate their shared validity contract.
    valid = np.array(role.valid_mask[:, channel], dtype=bool, copy=True)
    weights = np.asarray(role.quadrature_weights, dtype=np.float64)
    valid &= np.isfinite(weights) & (weights > 0)
    valid &= np.isfinite(prediction) & np.isfinite(reference)
    if not valid.any():
        raise ValueError(f"No finite common native support for channel {role.channel_names[channel]}.")
    weights = weights[valid]
    weights = weights / weights.sum()
    pred = prediction[valid]
    truth = reference[valid]
    residual = pred - truth
    return {
        "native_weighted_rmse": float(np.sqrt(np.dot(weights, residual * residual))),
        "prediction_signed_mean": float(np.dot(weights, pred)),
        "reference_signed_mean": float(np.dot(weights, truth)),
        "prediction_rms": float(np.sqrt(np.dot(weights, pred * pred))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "valid_receiver_count": int(valid.sum()),
        "positive_quadrature_weight_sum": float(np.asarray(role.quadrature_weights, dtype=np.float64)[valid].sum()),
    }


def _difference_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(delta))), "rms": float(np.sqrt(np.mean(delta * delta)))}


def _precision_comparison(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("Precision comparison requires identical native response axes.")
    close = np.abs(left - right) <= (atol + rtol * np.abs(right))
    return {
        **_difference_metrics(left, right),
        "rtol": rtol,
        "atol": atol,
        "passed": bool(close.all()),
        "failed_elements": int((~close).sum()),
        "compared_elements": int(close.size),
    }


def module_peak_changes(
    baseline: np.ndarray,
    endpoint: np.ndarray,
    valid_mask: np.ndarray,
    receiver_module_ids: tuple[str, ...] | np.ndarray,
    module_ids: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Compute max(endpoint) minus max(baseline), joined by typed module ID."""
    base = np.asarray(baseline, dtype=np.float64).reshape(-1)
    end = np.asarray(endpoint, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    receivers = np.asarray(receiver_module_ids)
    if not (base.shape == end.shape == valid.shape == receivers.shape):
        raise ValueError("Material peak endpoint arrays and receiver IDs must have one common row axis.")
    peaks: dict[str, dict[str, float]] = {}
    for module_id in module_ids:
        selected = (receivers == module_id) & valid
        if not selected.any() or not np.isfinite(base[selected]).all() or not np.isfinite(end[selected]).all():
            raise ValueError(f"Missing or nonfinite material endpoint support for {module_id}.")
        base_peak = float(base[selected].max())
        endpoint_peak = float(end[selected].max())
        peaks[str(module_id)] = {
            "baseline_peak": base_peak,
            "endpoint_peak": endpoint_peak,
            "predicted_peak_change": endpoint_peak - base_peak,
        }
    return peaks


def _reference_module_peaks(record: Any) -> dict[str, float]:
    role = record.output.roles["solid_temperature"]
    return {
        module_id: result["baseline_peak"]
        for module_id, result in module_peak_changes(
            role.values[:, 0], role.values[:, 0], role.valid_mask[:, 0],
            role.receiver_module_ids, tuple(record.design.active_module_ids),
        ).items()
    }


def _percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    absolute = np.abs(array)
    return {
        "mean_signed": float(array.mean()),
        "mean_absolute": float(absolute.mean()),
        "p90_absolute": float(np.percentile(absolute, 90)),
        "p95_absolute": float(np.percentile(absolute, 95)),
        "worst_absolute": float(absolute.max()),
    }


def _source_hash_inventory(request_path: Path, records_dir: Path, families: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    paths: set[Path] = {request_path.resolve()}
    metadata_dir = records_dir
    for case_id, records in families:
        for label in records:
            metadata_path = metadata_dir / f"receiver_interaction_{case_id}_{label}.json"
            paths.add(metadata_path.resolve())
            metadata = json.loads(metadata_path.read_text())
            paths.add(Path(metadata["output_arrays"]).resolve())
    attempt_ledger = records_dir.parent / "reference_attempts.jsonl"
    if attempt_ledger.exists():
        paths.add(attempt_ledger.resolve())
    inventory = []
    for path in sorted(paths):
        inventory.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return inventory


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    request_path: Path = DEFAULT_REQUEST,
    records_dir: Path = DEFAULT_RECORDS,
    output_path: Path,
    device_name: str = "cpu",
    source_read_policy: str = "checkpoint",
    source_read_executor: str | None = None,
    source_read_operating_point: int | None = None,
) -> dict[str, Any]:
    from thermal_response_refinement_evaluation import load_counted_families, validate_heat_records

    from honf_runtime.content_hypergraph import load_content_hypergraph_checkpoint

    if source_read_policy not in ("checkpoint", "full", "fitted", "attn95", "attn99"):
        raise ValueError("Unknown read-only source policy for saved response replay.")
    if source_read_executor not in (None, "packed", "packet", "dense_masked"):
        raise ValueError("Unknown source executor for saved response replay.")
    if source_read_operating_point is not None and source_read_operating_point not in (0, 1):
        raise ValueError("Saved response operating point must be zero or one.")
    if source_read_policy == "checkpoint" and (
            source_read_executor is not None or source_read_operating_point is not None):
        raise ValueError("An executor or operating-point override requires an explicit source policy.")
    checkpoint_path = checkpoint_path.resolve()
    request_path = request_path.resolve()
    records_dir = records_dir.resolve()
    output_path = output_path.resolve()
    npz_path = output_path.with_suffix(".npz")
    temporary_outputs = (output_path, npz_path, output_path.with_suffix(output_path.suffix + ".tmp"),
                         npz_path.with_suffix(npz_path.suffix + ".tmp"))
    existing_outputs = [str(path) for path in temporary_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"Response outputs are immutable; refusing to overwrite {existing_outputs}.")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable.")
    if not checkpoint_path.is_file() or not request_path.is_file() or not records_dir.is_dir():
        raise FileNotFoundError("Checkpoint, exact counted request, or saved-record directory is missing.")

    start = perf_counter()
    model, adapter, checkpoint = load_content_hypergraph_checkpoint(checkpoint_path, device=device)
    if adapter is not model or not hasattr(adapter, "prepare_affine_application"):
        raise TypeError("This replay accepts a self-contained Thermal content-hypergraph checkpoint only.")
    model.eval()
    policy_context = nullcontext()
    if source_read_policy != "checkpoint":
        core = getattr(model, "core", model)
        override = getattr(core, "source_read_override_v1", None)
        if not callable(override):
            raise ValueError("This checkpoint runtime has no versioned source-policy override.")
        policy_context = override(source_read_policy, executor=source_read_executor,
                                  operating_point=source_read_operating_point)

    families, source_dataset = load_counted_families(request_path, records_dir)
    request = json.loads(request_path.read_text())
    by_case = {case_id: records for case_id, records in families}
    request_cases = {case["case_id"]: case for case in request["cases"]}
    if tuple(case_id for case_id, _ in families) != ("0277", *CASE_IDS):
        raise ValueError("The counted source must retain the exact fixed4 order and missing 0277 baseline.")
    if set(by_case) != {"0277", *CASE_IDS} or "baseline" in by_case["0277"]:
        raise ValueError("The 0277 baseline absence must remain visible; it is never synthesized.")
    if set(by_case["0277"]) != {"transfer_minus", "transfer_plus"}:
        raise ValueError("0277 must retain its two saved signed transfers and no baseline record.")

    array_payload: dict[str, np.ndarray] = {}
    cases_report: list[dict[str, Any]] = []
    new_solver_attempts = 0
    response_application_count = 0
    response_storage = []
    with torch.no_grad(), policy_context:
        for case_id in CASE_IDS:
            records = by_case[case_id]
            if tuple(records) != STATE_LABELS:
                raise ValueError(f"{case_id} requires exactly saved baseline/minus/plus states in canonical order.")
            validate_heat_records(records)
            base_record = records["baseline"]
            structure, fluid_xy, material_local, input_meta = _typed_native_input(base_record, device=device)
            prepared = adapter.prepare_native(
                structure,
                fluid_xy,
                local_query_points=material_local,
                ntheta=64,
                scene_key=("saved_counted_response_v1", case_id),
            )
            prepared = adapter.prepare_affine_application(prepared)
            direct = getattr(prepared.affine_query, "direct_coefficients", None)
            response_storage.append({
                "case_id": case_id,
                "direct_representation": "dense_scalar_kernel" if isinstance(direct, torch.Tensor)
                else "selected_affine_factors" if direct is not None else "other_factorization",
                "dense_scalar_kernel_entries": direct.numel() if isinstance(direct, torch.Tensor) else 0,
                "selected_coefficient_entries": int(direct.residual_coefficients.numel())
                if direct is not None and hasattr(direct, "residual_coefficients") else 0,
                "completion_factor_entries": sum(value.numel() for value in (
                    getattr(direct, "completion_query", None), getattr(direct, "completion_source", None),
                    getattr(direct, "thermal_rank_weight", None)) if isinstance(value, torch.Tensor)),
            })

            request_states = {
                state["state_id"]: state for state in request_cases[case_id]["states"]
            }
            controls64 = {
                label: _tensor(state["active_heat_powers"], device=device, dtype=torch.float64)[None]
                for label, state in request_states.items()
            }
            controls32 = {label: value.to(torch.float32) for label, value in controls64.items()}
            for label, record in records.items():
                recorded_heating = np.asarray([module.heating for module in record.design.modules], dtype=np.float64)
                request_heating = controls64[label].detach().cpu().numpy()[0]
                if not np.array_equal(recorded_heating, request_heating):
                    raise ValueError(f"Request/record heat-control values differ for {case_id}/{label}.")
            reference_native_peaks = {
                label: _reference_module_peaks(record) for label, record in records.items()
            }
            reference_peak_consistency = {
                label: {
                    module_id: {
                        "native_receiver_peak": float(reference_native_peaks[label][module_id]),
                        "saved_record_quantity_peak": float(record.output.module_peak_temperature[module_id]),
                        "saved_quantity_minus_native_receiver_peak": float(
                            record.output.module_peak_temperature[module_id] - reference_native_peaks[label][module_id]
                        ),
                    }
                    for module_id in input_meta["module_ids"]
                }
                for label, record in records.items()
            }
            baseline_fp32 = adapter.apply_native(prepared, controls32["baseline"])
            baseline_fp64 = adapter.apply_native(
                prepared, controls64["baseline"], accumulation_dtype=torch.float64
            )
            response_application_count += 2
            native_baseline32 = _native_role_arrays(baseline_fp32, input_meta)
            native_baseline64 = _native_role_arrays(baseline_fp64, input_meta)
            case_prefix = f"case_{case_id}__"
            array_payload[case_prefix + "module_ids"] = np.asarray(input_meta["module_ids"], dtype="U64")
            array_payload[case_prefix + "module_centers_xy"] = input_meta["centers_xy"]
            array_payload[case_prefix + "fluid_xy"] = input_meta["fluid_xy"]
            array_payload[case_prefix + "surface_xy"] = input_meta["surface_xy"]
            array_payload[case_prefix + "surface_angles"] = input_meta["interface_angles"]
            array_payload[case_prefix + "material_xy"] = input_meta["material_xy"]
            array_payload[case_prefix + "material_local_xy"] = input_meta["material_local_xy"]
            array_payload[case_prefix + "baseline_predicted_fluid_fields_fp32"] = native_baseline32["fluid_fields"]
            array_payload[case_prefix + "baseline_predicted_fluid_fields_fp64"] = native_baseline64["fluid_fields"]
            array_payload[case_prefix + "baseline_predicted_interface_fp64"] = native_baseline64["interface"]
            array_payload[case_prefix + "baseline_predicted_material_fp64"] = native_baseline64["solid_temperature"]

            base_native_material_role = base_record.output.roles["solid_temperature"]
            base_model_peaks = {
                module_id: peak["baseline_peak"]
                for module_id, peak in module_peak_changes(
                    native_baseline64["solid_temperature"][:, 0],
                    native_baseline64["solid_temperature"][:, 0],
                    base_native_material_role.valid_mask[:, 0],
                    base_native_material_role.receiver_module_ids,
                    input_meta["module_ids"],
                ).items()
            }
            direction_reports = []
            all_peak_errors: list[float] = []
            for label in ("transfer_minus", "transfer_plus"):
                request_state = request_states[label]
                endpoint32 = adapter.apply_native(prepared, controls32[label])
                endpoint64 = adapter.apply_native(
                    prepared, controls64[label], accumulation_dtype=torch.float64
                )
                delta_controls = controls64[label] - controls64["baseline"]
                delta_controls32 = controls32[label] - controls32["baseline"]
                increment64 = adapter.apply_increment(
                    prepared, delta_controls, accumulation_dtype=torch.float64
                )
                increment32 = adapter.apply_increment(prepared, delta_controls32)
                response_application_count += 4
                native_endpoint32 = _native_role_arrays(endpoint32, input_meta)
                native_endpoint64 = _native_role_arrays(endpoint64, input_meta)
                native_increment64 = _native_role_arrays(increment64, input_meta)
                native_increment32 = _native_role_arrays(increment32, input_meta)
                endpoint_delta32 = {
                    name: native_endpoint32[name].astype(np.float32) - native_baseline32[name].astype(np.float32)
                    for name in ROLE_OUTPUTS
                }
                endpoint_delta64 = {
                    name: native_endpoint64[name] - native_baseline64[name]
                    for name in ROLE_OUTPUTS
                }
                physical_delta = {
                    name: np.asarray(records[label].output.roles[name].values, dtype=np.float64)
                    - np.asarray(base_record.output.roles[name].values, dtype=np.float64)
                    for name in ROLE_OUTPUTS
                }

                delta_metrics: dict[str, Any] = {}
                endpoint_precision: dict[str, Any] = {}
                for role_name in ROLE_OUTPUTS:
                    role = base_record.output.roles[role_name]
                    direct = native_increment64[role_name]
                    legacy = np.asarray(endpoint_delta32[role_name], dtype=np.float64)
                    legacy_increment = np.asarray(native_increment32[role_name], dtype=np.float64)
                    precise_endpoint = endpoint_delta64[role_name]
                    endpoint_precision[role_name] = {
                        "legacy_fp32_endpoint_difference_vs_legacy_fp32_direct_increment": _precision_comparison(
                            legacy, legacy_increment, rtol=2.0e-5, atol=2.0e-6
                        ),
                        "legacy_fp32_endpoint_difference_vs_fp64_direct_increment": _difference_metrics(legacy, direct),
                        "fp64_endpoint_difference_vs_fp64_direct_increment": _precision_comparison(
                            precise_endpoint, direct, rtol=1.0e-7, atol=1.0e-9
                        ),
                    }
                    channel = role.channel_names.index(TEMPERATURE_CHANNELS[role_name])
                    delta_metrics[role_name] = _weighted_change_metrics(
                        direct[:, channel] if role_name != "solid_temperature" else direct[:, 0],
                        physical_delta[role_name][:, channel] if role_name != "solid_temperature" else physical_delta[role_name][:, 0],
                        role,
                        channel,
                    )
                    delta_metrics[role_name]["temperature_units"] = role.channel_units[channel]
                    array_payload[case_prefix + label + f"__{role_name}_predicted_fp32_endpoint_difference"] = endpoint_delta32[
                        role_name
                    ]
                    array_payload[case_prefix + label + f"__{role_name}_predicted_fp64_endpoint_difference"] = endpoint_delta64[
                        role_name
                    ]
                    array_payload[case_prefix + label + f"__{role_name}_direct_increment_fp64"] = native_increment64[
                        role_name
                    ]
                    array_payload[case_prefix + label + f"__{role_name}_direct_increment_fp32"] = native_increment32[
                        role_name
                    ]
                    array_payload[case_prefix + label + f"__{role_name}_reference_endpoint_difference"] = physical_delta[
                        role_name
                    ]

                fluid_flow_difference = endpoint_delta32["fluid_fields"][:, :4]
                fluid_flow_difference64 = endpoint_delta64["fluid_fields"][:, :4]
                flow_null = {
                    "legacy_fp32_endpoint_max_abs": float(np.max(np.abs(fluid_flow_difference))),
                    "fp64_endpoint_max_abs": float(np.max(np.abs(fluid_flow_difference64))),
                    "direct_increment_max_abs": float(np.max(np.abs(native_increment64["fluid_fields"][:, :4]))),
                    "saved_reference_endpoint_max_abs_by_u_v_p_omega": np.max(
                        np.abs(physical_delta["fluid_fields"][:, :4]), axis=0
                    ).tolist(),
                }

                material_role = base_record.output.roles["solid_temperature"]
                predicted_peaks = module_peak_changes(
                    native_baseline64["solid_temperature"][:, 0],
                    native_endpoint64["solid_temperature"][:, 0],
                    material_role.valid_mask[:, 0],
                    material_role.receiver_module_ids,
                    input_meta["module_ids"],
                )
                physical_endpoint_peaks = records[label].output.module_peak_temperature
                physical_base_peaks = base_record.output.module_peak_temperature
                peak_comparison: dict[str, Any] = {}
                for module_id in input_meta["module_ids"]:
                    reference_change = float(
                        reference_native_peaks[label][module_id] - reference_native_peaks["baseline"][module_id]
                    )
                    predicted_change = predicted_peaks[module_id]["predicted_peak_change"]
                    peak_error = predicted_change - reference_change
                    all_peak_errors.append(peak_error)
                    record_quantity_change = float(physical_endpoint_peaks[module_id] - physical_base_peaks[module_id])
                    peak_comparison[module_id] = {
                        **predicted_peaks[module_id],
                        "reference_baseline_native_receiver_peak": float(reference_native_peaks["baseline"][module_id]),
                        "reference_endpoint_native_receiver_peak": float(reference_native_peaks[label][module_id]),
                        "reference_peak_change": reference_change,
                        "saved_record_quantity_baseline_peak": float(physical_base_peaks[module_id]),
                        "saved_record_quantity_endpoint_peak": float(physical_endpoint_peaks[module_id]),
                        "saved_record_quantity_peak_change": record_quantity_change,
                        "peak_change_error": peak_error,
                    }
                direction_reports.append({
                    "state_id": label,
                    "signed_transfer_amplitude": float(request_state["signed_heat_transfer"]),
                    "selected_pair_source_slots": request_cases[case_id]["selected_pair_source_slots"],
                    "delta_heat_by_source": delta_controls.detach().cpu().numpy()[0].astype(np.float64).tolist(),
                    "heat_quantization_error_fp32_minus_saved_fp64": (
                        controls32[label].to(torch.float64) - controls64[label]
                    ).detach().cpu().numpy()[0].tolist(),
                    "heat_quantization_error_max_abs": float((
                        controls32[label].to(torch.float64) - controls64[label]
                    ).abs().max().cpu()),
                    "delta_heat_quantization_error_fp32_minus_saved_fp64": (
                        delta_controls32.to(torch.float64) - delta_controls
                    ).detach().cpu().numpy()[0].tolist(),
                    "delta_heat_quantization_error_max_abs": float((
                        delta_controls32.to(torch.float64) - delta_controls
                    ).abs().max().cpu()),
                    "native_temperature_response": delta_metrics,
                    "endpoint_difference_precision": endpoint_precision,
                    "heat_null_flow": flow_null,
                    "module_peak_endpoint_differences": peak_comparison,
                    "module_peak_error_summary": _percentile_summary([
                        item["peak_change_error"] for item in peak_comparison.values()
                    ]),
                })

            cases_report.append({
                "case_id": case_id,
                "active_module_count": len(input_meta["module_ids"]),
                "active_module_ids": list(input_meta["module_ids"]),
                "role_channels_and_units": {
                    role_name: {
                        "channel_names": list(base_record.output.roles[role_name].channel_names),
                        "channel_units": list(base_record.output.roles[role_name].channel_units),
                    }
                    for role_name in ROLE_OUTPUTS
                },
                "receiver_counts": {
                    "fluid_fields": len(base_record.output.roles["fluid_fields"].query_features),
                    "surface_per_module": 64,
                    "material_per_module": input_meta["material_row_counts"],
                },
                "native_surface_ntheta": prepared.ntheta,
                "prepared_once_affine_factors_reused": prepared.affine_query is not None,
                "module_peak_baseline_native": base_model_peaks,
                "reference_saved_peak_quantity_consistency": reference_peak_consistency,
                "reference_peak_source": "Common stored native material-role receiver support; saved module_peak_temperature quantity reported separately.",
                "signed_directions": direction_reports,
                "module_peak_error_summary_all_signed_directions": _percentile_summary(all_peak_errors),
            })

    source_files = _source_hash_inventory(request_path, records_dir, families)
    peak_errors_by_module_count: dict[str, list[float]] = {}
    for case_result in cases_report:
        module_count = str(case_result["active_module_count"])
        bucket = peak_errors_by_module_count.setdefault(module_count, [])
        for direction in case_result["signed_directions"]:
            bucket.extend(
                item["peak_change_error"]
                for item in direction["module_peak_endpoint_differences"].values()
            )
    peak_count_summaries = {}
    for module_count in (3, 5, 7, 10):
        values = peak_errors_by_module_count.get(str(module_count), [])
        peak_count_summaries[str(module_count)] = (
            {"status": "measured", "case_count": sum(
                case["active_module_count"] == module_count for case in cases_report
            ), "signed_direction_module_count": len(values), **_percentile_summary(values)}
            if values else {
                "status": "not_replayed",
                "reason": "The response replay is restricted to 0291/0294/0687; 0277 has no saved baseline.",
            }
        )
    identity = checkpoint.get("experiment_identity", {})
    recipe = identity.get("recipe", {})
    provider_identity = identity.get("provider_identity", {})
    adapter_state = provider_identity.get("adapter_state", {})
    source_binding = provider_identity.get("source_binding", {})
    dataset_binding = {
        "dataset": provider_identity.get("dataset"),
        "task": provider_identity.get("task"),
        "dataset_profile": provider_identity.get("dataset_profile"),
        "subset_manifest_sha256": provider_identity.get("subset_manifest_sha256"),
        "base_fixed25_manifest_sha256": provider_identity.get("base_fixed25_manifest_sha256"),
        "training_case_ids_sha256": provider_identity.get("training_case_ids_sha256"),
        "validation_case_ids_sha256": provider_identity.get("validation_case_ids_sha256"),
        "normalization_stats_sha256": provider_identity.get("normalization_stats_sha256"),
        "source_binding": source_binding,
    }
    payload = {
        "schema": "thermal_content_hypergraph_saved_response_v1",
        "scope": "Saved baseline and signed transfer replay for 0291, 0294, 0687 only; zero optimizer and zero solver calls.",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "epoch": int(checkpoint.get("epoch", -1)),
            "architecture": checkpoint.get("arm"),
            "recipe_sha256": identity.get("recipe_sha256"),
            "recipe_schema": recipe.get("schema"),
            "dataset_binding": dataset_binding,
            "model_config": provider_identity.get("model_config"),
            "adapter_normalizer_key": adapter_state.get("normalizer_key"),
        },
        "device": str(device),
        "request": {
            "path": str(request_path),
            "sha256": _sha256(request_path),
            "manifest_sha256": request.get("manifest_sha256"),
            "source_dataset_declared": source_dataset,
            "source_data_interpretation": "Saved analytic-wake/shared-grid reference; not CFD.",
            "selected_case_ids": list(CASE_IDS),
            "loaded_fixed4_case_ids": [case_id for case_id, _ in families],
            "missing_0277_baseline_preserved": "baseline" not in by_case["0277"],
            "0277_signed_transfer_records_loaded_but_not_replayed": sorted(by_case["0277"]),
            "states_per_replayed_case": list(STATE_LABELS),
            "counted_physical_records_total": sum(len(records) for _, records in families),
            "independent_transfer_families": 3,
            "signed_transfer_replays": 6,
            "input_hashes": source_files,
        },
        "response_method": {
            "geometry_context": "Saved typed centers, material/flow context, and role receiver metadata only; physical heat controls remain separate.",
            "native_surface_points_per_module": 64,
            "native_material_points_per_module": "all stored receivers; module-specific count recorded per case",
            "geometry_units": "saved Thermal dataset geometry units",
            "heat_control_units": "saved request active_heat_powers units, unchanged",
            "q_normal_interpretation": "saved receiver quantity is a heat-flux proxy; no new sign or calibration claim is made",
            "prepared_scene_and_affine_factors_per_case": 1,
            "factorized_response": True,
            "dense_source_kernel_materialized": any(row["dense_scalar_kernel_entries"] for row in response_storage),
            "explicit_dense_source_kernel_export_requested": False,
            "observed_response_storage": response_storage,
            "prepared_scene_count": len(CASE_IDS),
            "prepared_affine_factor_sets": len(CASE_IDS),
            "cached_response_application_calls": response_application_count,
            "neural_context_recomputed_per_source_or_transfer": False,
            "endpoint_precisions": ["FP32 endpoint subtraction", "FP64 endpoint subtraction", "FP64 direct factorized increment"],
            "legacy_fp32_strict_tolerance": {"rtol": 2.0e-5, "atol": 2.0e-6},
            "software_fp64_law_tolerance": {"rtol": 1.0e-7, "atol": 1.0e-9},
            "module_peak_definition": "max(T(h+delta_h)) - max(T(h)) on each module's native material rows; never max(delta_T).",
        },
        "physical_reference": {
            "solver_attempts_cumulative": 326,
            "solver_attempt_allowance": 326,
            "new_solver_attempts": new_solver_attempts,
            "new_physical_reference_calls": 0,
            "new_optimizer_updates": 0,
            "inverse_optimization": False,
        },
        "model_response_applications": response_application_count,
        "module_count_peak_error_summaries": peak_count_summaries,
        "elapsed_seconds": perf_counter() - start,
        "cases": cases_report,
        "numeric_arrays": str(output_path.with_suffix(".npz")),
    }
    if source_read_policy != "checkpoint":
        payload["read_only_source_policy"] = {
            "mode": source_read_policy, "executor_override": source_read_executor,
            "operating_point_override": source_read_operating_point,
            "same_policy_for_baseline_and_endpoints": True,
            "fit_or_threshold_change": False,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_path.with_suffix(".npz")
    temporary_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, **array_payload)
    temporary_npz.replace(npz_path)
    payload["numeric_arrays_sha256"] = _sha256(npz_path)
    _write_json(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, required=True, help="JSON receipt path; a neighboring NPZ stores selected arrays.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-read-policy", default="checkpoint",
                        choices=("checkpoint", "full", "fitted", "attn95", "attn99"))
    parser.add_argument("--source-read-executor", choices=("packed", "packet", "dense_masked"))
    parser.add_argument("--source-read-operating-point", type=int, choices=(0, 1))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        request_path=args.request,
        records_dir=args.records_dir,
        output_path=args.output,
        device_name=args.device,
        source_read_policy=args.source_read_policy,
        source_read_executor=args.source_read_executor,
        source_read_operating_point=args.source_read_operating_point,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "numeric_arrays": payload["numeric_arrays"],
        "architecture": payload["checkpoint"]["architecture"],
        "epoch": payload["checkpoint"]["epoch"],
        "cases": payload["request"]["selected_case_ids"],
        "model_response_applications": payload["model_response_applications"],
        "new_solver_attempts": payload["physical_reference"]["new_solver_attempts"],
        "new_optimizer_updates": payload["physical_reference"]["new_optimizer_updates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
