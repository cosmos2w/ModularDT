"""Native evidence and fixed-checkpoint diagnostics for Round13-3 routing."""
from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contextual_frontier_evaluation as native_evaluation


SCHEMA = "thermal_interaction_reused_member_routing_evaluation_v1"
RELATIVE_SCHEMA = "thermal_relative_member_routing_evaluation_v2"
PROBE_NAMES = (
    "b_fixed_candidates_scores",
    "geometry_b_comparable_counts",
    "geometry_member_scores_fixed_b_pi",
    "all_candidates_open",
    "remove_contextual_message_recompute",
)
_DETAIL_MODULE_COUNTS = (3, 7, 10)


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _module_count(case: Mapping[str, Any]) -> int:
    structure = case.get("structure")
    if not isinstance(structure, Mapping) or "module_present" not in structure:
        raise ValueError("Thermal input case has no original module_present vector.")
    return int((_as_numpy(structure["module_present"]) > 0.5).sum())


def select_low_m_train_cases(provider: Any, limit: int = 4) -> tuple[str, ...]:
    """Select a small input-only TRAIN panel spanning M1 and M2 first."""
    if not 1 <= int(limit) <= 4:
        raise ValueError("Round13-3 low-M TRAIN evaluation is limited to one through four cases.")
    cases = tuple(getattr(provider, "training_cases", ()))
    case_ids = tuple(map(str, getattr(provider, "training_case_ids", ())))
    if not cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("The rebuilt provider must expose unique selected-TRAIN case IDs and inputs.")
    by_id = {str(case["case_id"]): case for case in cases}
    if set(by_id) != set(case_ids):
        raise ValueError("Selected-TRAIN input cases do not match the provider's declared membership.")
    counts = {case_id: _module_count(by_id[case_id]) for case_id in case_ids}
    selected: list[str] = []
    for count in (1, 2):
        selected.extend(sorted(case_id for case_id in case_ids if counts[case_id] == count)[:2])
    remaining = sorted(
        (case_id for case_id in case_ids if case_id not in selected),
        key=lambda case_id: (counts[case_id], case_id),
    )
    selected.extend(remaining[:max(0, min(int(limit), len(case_ids)) - len(selected))])
    return tuple(selected[:int(limit)])


def _case_counts(provider: Any, split: str) -> dict[str, int]:
    attribute = "training_cases" if split == "train" else "validation_cases"
    cases = tuple(getattr(provider, attribute, ()))
    result = {str(case["case_id"]): _module_count(case) for case in cases}
    if len(result) != len(cases):
        raise ValueError(f"Thermal {split} input case IDs are not unique.")
    return result


def select_validation_representatives(provider: Any, limit: int = 4) -> tuple[str, ...]:
    """Select fixed VAL M3/M7/M10 layouts by input structure and stable ID."""
    if not 3 <= int(limit) <= 4:
        raise ValueError("Round13-3 VAL detail panel must include three and is limited to four cases.")
    case_ids = tuple(map(str, getattr(provider, "validation_case_ids", ())))
    if not case_ids:
        raise ValueError("The rebuilt provider has no exposed validation membership.")
    counts = _case_counts(provider, "validation")
    selected = tuple(
        min((case_id for case_id in case_ids if counts[case_id] == count), default=None)
        for count in _DETAIL_MODULE_COUNTS
    )
    if any(case_id is None for case_id in selected):
        fallback = native_evaluation._representative_cases(provider, case_ids, limit)
        ordered_fallback = [case_id for case_id in case_ids
                            if case_id in fallback and counts[case_id] != 1]
        selected_ids = [case_id for case_id in selected if case_id is not None]
        selected_ids.extend(case_id for case_id in ordered_fallback if case_id not in selected_ids)
        selected = tuple(selected_ids[:limit])
    ordered = tuple(case_id for case_id in selected if case_id is not None)
    if not ordered or len(ordered) > limit:
        raise ValueError("Input-selected validation representative panel is invalid.")
    return ordered


def _route_interventions_enabled(model: Any) -> bool:
    architecture = str(getattr(model, "architecture", ""))
    if architecture not in ("R13-3-Full", "R13-3-Route", "R13-3-Relative"):
        raise ValueError(f"Unsupported Round13-3 model architecture: {architecture!r}.")
    return architecture in ("R13-3-Route", "R13-3-Relative")


def _query_indices(receivers: Any, prediction: Any) -> tuple[int, ...]:
    """Choose three fixed rows from the prepared native query union."""
    del receivers
    native_prepared = getattr(prediction, "native_prepared", None)
    query_batch = getattr(native_prepared, "query_batch", None)
    coordinates = getattr(query_batch, "coordinates", None)
    if coordinates is None:
        raise TypeError("Prepared native query batch must expose its union coordinates.")
    xy = _as_numpy(coordinates)
    if xy.ndim == 3:
        if xy.shape[0] != 1:
            raise ValueError("Round13-3 route traces require one native case per batch.")
        xy = xy[0]
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 3:
        raise ValueError("Prepared native query union must contain at least three 2D coordinates.")
    valid = np.ones(xy.shape[0], dtype=bool)
    query_mask = getattr(query_batch, "query_mask", None)
    if query_mask is not None:
        mask = _as_numpy(query_mask)
        if mask.ndim == 2:
            if mask.shape[0] != 1:
                raise ValueError("Round13-3 route traces require one native query-mask row.")
            mask = mask[0]
        if mask.shape != valid.shape:
            raise ValueError("Prepared native query mask does not match the query-union coordinates.")
        valid &= mask.astype(bool)
    indexes = np.flatnonzero(valid)
    if indexes.size < 3:
        raise ValueError("Fewer than three valid prepared native query-union rows are available.")
    points = xy[indexes]
    xlo, xhi = float(points[:, 0].min()), float(points[:, 0].max())
    ymid = float(np.median(points[:, 1]))
    chosen: list[int] = []
    for fraction in (0.2, 0.5, 0.8):
        target_x = xlo + fraction * (xhi - xlo)
        distances = ((points[:, 0] - target_x) / max(xhi - xlo, 1e-12)) ** 2
        distances += ((points[:, 1] - ymid) / max(float(np.ptp(points[:, 1])), 1e-12)) ** 2
        for local in np.argsort(distances, kind="stable"):
            candidate = int(indexes[local])
            if candidate not in chosen:
                chosen.append(candidate)
                break
    if len(chosen) != 3:
        raise RuntimeError("Could not select three unique fixed query rows from the native union.")
    return tuple(chosen)


def _query_coordinates(prediction: Any, query_indices: Sequence[int]) -> list[list[float]]:
    query_batch = getattr(getattr(prediction, "native_prepared", None), "query_batch", None)
    coordinates = getattr(query_batch, "coordinates", None)
    if coordinates is None:
        raise TypeError("Prepared native query batch must expose its union coordinates.")
    xy = _as_numpy(coordinates)
    if xy.ndim == 3:
        if xy.shape[0] != 1:
            raise ValueError("Round13-3 route traces require one native case per batch.")
        xy = xy[0]
    rows = [int(index) for index in query_indices]
    if any(index < 0 or index >= len(xy) for index in rows):
        raise IndexError("Route trace query index is outside the prepared native query union.")
    return np.asarray(xy[rows], dtype=np.float64).tolist()


def _native_arrays_with_residuals(prediction: Any, batch: Any, case_id: str) -> dict[str, np.ndarray]:
    arrays = native_evaluation._native_export_arrays(prediction, batch, case_id)
    scene = getattr(prediction.native_prepared, "scene_batch", None)
    for name in ("module_coordinates", "environment_coordinates", "module_ids", "environment_ids",
                 "module_mask", "environment_mask", "module_measure", "environment_measure",
                 "module_characteristic_lengths", "environment_characteristic_lengths",
                 "coordinate_origin", "coordinate_scale"):
        value = getattr(scene, name, None)
        if value is not None:
            arrays[name] = _as_numpy(value)[0]
    pairs = (
        ("field", "pred_field", "target_field"),
        ("surface_temperature", "pred_surface_temperature", "target_surface_temperature"),
        ("q_proxy", "pred_q_proxy", "target_q_proxy"),
        ("material_temperature", "pred_material_temperature", "target_material_temperature"),
    )
    for name, prediction_key, target_key in pairs:
        arrays[f"residual_{name}"] = np.asarray(arrays[prediction_key]) - np.asarray(arrays[target_key])
    arrays["fluid_xy"] = _as_numpy(batch.receivers.fluid_xy)[0]
    query_batch = getattr(getattr(prediction, "native_prepared", None), "query_batch", None)
    if query_batch is not None and getattr(query_batch, "coordinates", None) is not None:
        arrays["native_union_query_coordinates"] = _as_numpy(query_batch.coordinates)[0]
        query_mask = getattr(query_batch, "query_mask", None)
        if query_mask is not None:
            arrays["native_union_query_mask"] = _as_numpy(query_mask)[0]
    arrays["units_json"] = np.asarray(json.dumps({
        "coordinates": "ThermalChannel native coordinates",
        "field_channels": ["u", "v", "p", "omega", "temperature"],
        "surface_temperature": "native ThermalChannel temperature units",
        "material_temperature": "native ThermalChannel temperature units",
        "q_proxy": "maintained native interface target units",
    }, sort_keys=True))
    return arrays


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    return value


def _storage_snapshot(output_parent: Path, *, storage_output_root: str | Path,
                      output_budget_bytes: int, payload_budget_bytes: int,
                      temporary_budget_bytes: int,
                      minimum_free_headroom_bytes: int) -> dict[str, Any]:
    data2 = Path("/data2")
    if not data2.is_dir() or not os.path.ismount(data2):
        raise RuntimeError("/data2 is not mounted; refusing to write Round13-3 evaluation evidence.")
    data2_root = (data2 / "wanglz/ModularDT").resolve(strict=True)
    root = Path(storage_output_root).expanduser().resolve(strict=True)
    parent = output_parent.resolve(strict=True)
    if not root.is_relative_to(data2_root) or root == data2_root:
        raise ValueError(f"Saved storage.output_root must be beneath {data2_root}.")
    if not parent.is_relative_to(root):
        raise ValueError(f"Evaluation output must remain beneath saved storage.output_root {root}.")
    data2_device = data2.stat().st_dev
    if root.stat().st_dev != data2_device or parent.stat().st_dev != data2_device:
        raise RuntimeError("Round13-3 output root is not on the mounted /data2 filesystem.")
    used_before = sum(
        path.stat().st_size for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if used_before + output_budget_bytes > payload_budget_bytes:
        raise OSError("Round13-3 evaluation reservation would exceed the saved DATA2 payload budget.")
    free_before = int(shutil.disk_usage(parent).free)
    if free_before - output_budget_bytes - temporary_budget_bytes < minimum_free_headroom_bytes:
        raise OSError("Evaluation output and peak temporary reservation would violate DATA2 headroom.")
    return {
        "output_root": str(root),
        "round_payload_budget_bytes": int(payload_budget_bytes),
        "round_payload_bytes_before": int(used_before),
        "round_payload_bytes_remaining_before": int(payload_budget_bytes - used_before),
        "max_output_bytes": int(output_budget_bytes),
        "estimated_peak_temporary_bytes": int(temporary_budget_bytes),
        "minimum_free_headroom_bytes": int(minimum_free_headroom_bytes),
        "free_before_bytes": free_before,
        "dataset_copies": 0,
        "new_persistent_caches": 0,
    }


def _accumulate_work(total: dict[str, float], values: Mapping[str, Any]) -> None:
    for name, value in values.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            number = float(value)
            if np.isfinite(number):
                total[str(name)] = total.get(str(name), 0.0) + number


def _evaluate_native_split(model: Any, provider: Any, payload: Mapping[str, Any], split: str,
                           case_ids: Sequence[str], *, export_ids: set[str] = frozenset(),
                           collect_routes: bool = False) -> dict[str, Any]:
    if split not in ("validation", "train") or not case_ids:
        raise ValueError("Native routing evaluation needs a nonempty validation or fixed TRAIN panel.")
    if len(case_ids) != len(set(map(str, case_ids))):
        raise ValueError("Native routing evaluation panel contains duplicate case IDs.")
    records: list[Mapping[str, Any]] = []
    rows: list[Mapping[str, Any]] = []
    exports: dict[str, dict[str, np.ndarray]] = {}
    traces: dict[str, Any] = {}
    work_totals: dict[str, float] = {}
    observed: list[str] = []
    epoch = int(payload.get("epoch", payload.get("current_epoch", 0)))
    batches = provider.native_evaluation_batches(split=split, case_keys=case_ids, microbatch_size=1)
    model.eval()
    with torch.no_grad():
        for batch in batches:
            batch_ids = tuple(map(str, batch.targets.case_ids))
            if len(batch_ids) != 1 or tuple(map(str, batch.case_keys)) != batch_ids:
                raise ValueError("Round13-3 native evaluation requires exactly one complete case per batch.")
            case_id = batch_ids[0]
            expected_index = len(observed)
            if expected_index >= len(case_ids) or case_id != str(case_ids[expected_index]):
                raise ValueError("Native routing provider changed the fixed case order or membership.")
            prediction, auxiliary = provider.predict_native(
                model, batch.scene_inputs, batch.receivers, model.architecture, "validation", epoch=epoch,
            )
            record = provider.validation_metrics(prediction, batch.targets, auxiliary)
            if not isinstance(record, Mapping):
                raise TypeError("Native Thermal validation_metrics must return a mapping.")
            records.append(record)
            case_rows = record.get("case_rows")
            if not isinstance(case_rows, Sequence) or len(case_rows) != 1:
                raise ValueError("Native metric records must contain exactly one per-case row.")
            rows.extend(case_rows)
            counter = getattr(provider, "work_counts", None)
            counts = counter(batch, prediction, auxiliary) if callable(counter) else {}
            if isinstance(counts, Mapping):
                _accumulate_work(work_totals, counts)
            if case_id in export_ids:
                exports[case_id] = _native_arrays_with_residuals(prediction, batch, case_id)
            if collect_routes and case_id in export_ids:
                trace_fn = getattr(provider, "member_routing_trace", None)
                if not callable(trace_fn):
                    raise TypeError("Member-routing provider must expose member_routing_trace for selected cases.")
                query_indices = _query_indices(batch.receivers, prediction)
                trace_rows = trace_fn(model, prediction, query_indices)
                if not isinstance(trace_rows, Sequence) or len(trace_rows) != 3:
                    raise ValueError("member_routing_trace must return one trace for each of three fixed queries.")
                traces[case_id] = {
                    "query_indices": list(query_indices),
                    "query_coordinates": _query_coordinates(prediction, query_indices),
                    "queries": _jsonable(trace_rows),
                }
            observed.append(case_id)
    if tuple(observed) != tuple(map(str, case_ids)):
        raise ValueError("Native routing evaluation did not cover its fixed case panel exactly once.")
    reducer = getattr(provider, "reduce_native_evaluation_metrics", None)
    if not callable(reducer):
        raise TypeError("Native routing provider must expose reduce_native_evaluation_metrics().")
    reduced = dict(reducer(records, split=split, case_keys=case_ids))
    reduced_rows = reduced.get("per_case")
    if not isinstance(reduced_rows, Sequence) or tuple(str(row.get("case_id")) for row in reduced_rows) != tuple(observed):
        raise ValueError("Native metric reducer changed the exact fixed-panel membership or order.")
    metrics = native_evaluation._summarize_nine_observables(
        reduced_rows, known_module_counts=native_evaluation._known_module_counts(provider),
    )
    return {
        "split": split,
        "case_ids": observed,
        "case_count": len(observed),
        "metrics": metrics,
        "maintained_native_reduction": reduced,
        "executed_work_counts": {
            key: int(value) if float(value).is_integer() else float(value)
            for key, value in sorted(work_totals.items())
        },
        "representative_exports": sorted(exports),
        "representative_arrays": exports,
        "route_traces": traces,
    }


def _fixed_probe_evidence(model: Any, provider: Any, batches: Mapping[str, Any],
                          base_predictions: Mapping[str, Any], query_indices: Mapping[str, Sequence[int]],
                          base_traces: Mapping[str, Any]) -> dict[str, Any]:
    probe = getattr(provider, "member_routing_probe", None)
    if not callable(probe):
        raise TypeError("Member-routing provider must expose member_routing_probe for fixed-checkpoint interventions.")
    result: dict[str, Any] = {}
    for case_id, batch in batches.items():
        case_rows = {}
        for probe_name in PROBE_NAMES:
            with torch.no_grad():
                measured = probe(model, batch, base_predictions[case_id], query_indices[case_id], probe_name)
            if not isinstance(measured, Mapping):
                raise TypeError(f"member_routing_probe({probe_name}) must return a mapping.")
            prediction = measured.get("prediction")
            work = measured.get("work_counts")
            if prediction is None or not isinstance(work, Mapping):
                raise ValueError(f"Fixed-checkpoint probe {probe_name} must return native prediction and actual work counts.")
            arrays = _native_arrays_with_residuals(prediction, batch, case_id)
            baseline = _native_arrays_with_residuals(base_predictions[case_id], batch, case_id)
            field_delta = np.asarray(arrays["pred_field"]) - np.asarray(baseline["pred_field"])
            component_deltas = {}
            for name in ("pred_field", "pred_surface_temperature", "pred_q_proxy", "pred_material_temperature"):
                delta = np.asarray(arrays[name]) - np.asarray(baseline[name])
                component_deltas[name.removeprefix("pred_")] = {
                    "rmse": float(np.sqrt(np.mean(np.square(delta, dtype=np.float64)))) if delta.size else 0.0,
                    "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
                }
            baseline_counts = measured.get("baseline_selected_member_counts")
            selected_counts = measured.get("selected_member_counts")
            if not isinstance(baseline_counts, Mapping) or not isinstance(selected_counts, Mapping):
                raise ValueError(f"Fixed-checkpoint probe {probe_name} must expose actual baseline and selected M/E member counts.")
            counts_matched = dict(baseline_counts) == dict(selected_counts)
            routing_changes = measured.get("route_weight_changes")
            if not isinstance(routing_changes, Mapping):
                raise ValueError(f"Fixed-checkpoint probe {probe_name} must report route-weight changes.")
            probe_trace = measured.get("trace")
            if not isinstance(probe_trace, Sequence) or len(probe_trace) != 3:
                raise ValueError(f"Fixed-checkpoint probe {probe_name} must preserve all three original-query route traces.")
            if probe_name == "all_candidates_open":
                max_delta = max(
                    (float(values["max_abs"]) for values in component_deltas.values()),
                    default=0.0,
                )
                dense_control = {
                    "expected_same_prediction": True,
                    "max_abs_native_component_delta": max_delta,
                    "within_1e-5_absolute_tolerance": bool(max_delta <= 1e-5),
                }
            else:
                dense_control = None
            case_rows[probe_name] = {
                "field_delta_rmse": float(np.sqrt(np.mean(np.square(field_delta, dtype=np.float64)))) if field_delta.size else 0.0,
                "field_delta_max_abs": float(np.max(np.abs(field_delta))) if field_delta.size else 0.0,
                "field_delta_mean_abs": float(np.mean(np.abs(field_delta))) if field_delta.size else 0.0,
                "native_component_deltas": component_deltas,
                "baseline_selected_member_counts": _jsonable(baseline_counts),
                "selected_member_counts": _jsonable(selected_counts),
                "selected_member_counts_match_baseline": counts_matched,
                "count_match_interpretation": (
                    "matched" if counts_matched else
                    "reported unmatched; geometry control remains a changed-work comparison"
                ),
                "route_weight_changes": _jsonable(routing_changes),
                "geometry_count_calibration": _jsonable(measured.get("geometry_count_calibration")),
                "full_native_selected_member_counts": _jsonable(measured.get("full_native_selected_member_counts")),
                "dense_execution_control": dense_control,
                "baseline_trace": _jsonable(base_traces[case_id]),
                "work_counts": _jsonable(work),
                "trace": _jsonable(probe_trace),
            }
        result[case_id] = case_rows
    return result


def _validate_benchmark(result: Any, *, case_id: str) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("member_routing_benchmark must return a mapping.")
    if int(result.get("warmup_count", -1)) != 1 or int(result.get("repetitions", -1)) != 3:
        raise ValueError("Round13-3 timing requires exactly one warmup and three measured repeats.")
    phases = result.get("phases")
    required = ("prepare", "cached_decode", "complete_native", "training_macro")
    if not isinstance(phases, Mapping) or any(name not in phases for name in required):
        raise ValueError(f"Round13-3 cost probe must report {required} separately.")
    checked = {}
    for phase in required:
        values = phases[phase]
        if isinstance(values, Mapping):
            samples = values.get("samples_seconds")
            summary = {key: value for key, value in values.items() if key != "samples_seconds"}
        else:
            samples, summary = values, {}
        if not isinstance(samples, Sequence) or len(samples) != 3:
            raise ValueError(f"Timing phase {phase} must preserve its three measured samples.")
        samples = [float(value) for value in samples]
        if any(not np.isfinite(value) or value < 0 for value in samples):
            raise ValueError(f"Timing phase {phase} contains invalid elapsed seconds.")
        checked[phase] = {
            **_jsonable(summary),
            "samples_seconds": samples,
            "mean_seconds": float(np.mean(samples)),
            "p90_seconds": float(np.percentile(samples, 90)),
        }
    return {
        "case_id": case_id,
        "warmup_count": 1,
        "repetitions": 3,
        "phases": checked,
        "work_counts": _jsonable(result.get("work_counts", {})),
        "peak_memory_bytes": _jsonable(result.get("peak_memory_bytes")),
        "peak_memory_increment_bytes": _jsonable(result.get("peak_memory_increment_bytes")),
        "selection_value_rows": _jsonable(result.get("selection_value_rows", {})),
        "context_score_work": _jsonable(result.get("context_score_work", {})),
        "training_macro_case_ids": _jsonable(result.get("training_macro_case_ids", [])),
        "training_macro_microbatch_sizes": _jsonable(result.get("training_macro_microbatch_sizes", [])),
        "training_macro_optimizer_steps": _jsonable(result.get("training_macro_optimizer_steps")),
        "training_macro_optimizer_evaluations": _jsonable(result.get("training_macro_optimizer_evaluations")),
        "training_macro_optimizer_updates_discarded": _jsonable(result.get("training_macro_optimizer_updates_discarded")),
        "training_macro_path": str(result.get("training_macro_path", "")),
    }


def _state_snapshot(model: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def _assert_model_state_unchanged(model: Any, before: Mapping[str, torch.Tensor], where: str) -> None:
    after = dict(model.named_parameters())
    changed = [name for name, value in before.items() if name not in after or not torch.equal(value, after[name].detach().cpu())]
    if changed:
        raise RuntimeError(f"Disposable Round13-3 {where} modified checkpoint model state: {changed[:8]}.")


def _write_result(output: Path, result: Mapping[str, Any], array_exports: Mapping[str, Mapping[str, np.ndarray]],
                  *, max_output_bytes: int, free_before: int) -> int:
    output.mkdir(parents=False)
    for case_id, arrays in array_exports.items():
        safe_id = "".join(character if character.isalnum() or character in "_.-" else "_" for character in case_id)
        with (output / f"case_{safe_id}.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    json_path = output / "result.json"
    result_jsonable = _jsonable(result)
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(result_jsonable, stream, indent=2, allow_nan=False)
        stream.write("\n")
    actual = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if actual > max_output_bytes:
        raise OSError(f"Round13-3 evaluation evidence used {actual} bytes, above its {max_output_bytes}-byte budget.")
    result_with_storage = dict(result_jsonable)
    for _ in range(3):
        result_with_storage["storage_budget"].update({
            "actual_output_bytes": actual,
            "free_before_bytes": int(free_before),
            "free_after_bytes": int(shutil.disk_usage(output).free),
        })
        with json_path.open("w", encoding="utf-8") as stream:
            json.dump(result_with_storage, stream, indent=2, allow_nan=False)
            stream.write("\n")
        updated_actual = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if updated_actual == actual:
            break
        actual = updated_actual
    else:
        raise OSError("Could not settle the final Round13-3 evidence size receipt.")
    actual = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if actual > max_output_bytes:
        raise OSError(f"Final Round13-3 evaluation evidence used {actual} bytes, above its {max_output_bytes}-byte budget.")
    return actual


def evaluate_member_routing(checkpoint: str | Path, device: str | torch.device,
                            output: str | Path) -> dict[str, Any]:
    """Evaluate VAL89, a fixed low-M TRAIN panel, route interventions, and disposable cost.

    The provider is rebuilt from the trusted checkpoint recipe. It performs no
    retained fitting, solver calls, inverse fitting, or cache writes. Its
    explicitly disposable training-macro timing benchmark executes optimizer
    steps, then restores the checkpoint model, provider progress, and RNG state.
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"Round13-3 evaluation output already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Evaluation output parent must exist before writing: {output_path.parent}")

    from channelthermal.training.member_routing_task import load_member_routing_checkpoint

    model, provider, payload = load_member_routing_checkpoint(checkpoint_path, device)
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Round13-3 checkpoint is missing experiment_identity.")
    workflow = str(identity.get("workflow", ""))
    if "interaction_reused_member_routing" not in workflow:
        raise ValueError("Checkpoint does not identify the maintained interaction-reused member-routing workflow.")
    provider_identity = identity.get("provider_identity", {})
    if not isinstance(provider_identity, Mapping):
        raise ValueError("Round13-3 checkpoint provider_identity must be a mapping.")
    recipe = provider_identity.get("recipe", identity.get("recipe", {}))
    if not isinstance(recipe, Mapping):
        raise ValueError("Round13-3 checkpoint recipe must be a mapping.")
    evaluation_config = recipe.get("evaluation", {})
    storage_config = recipe.get("storage", {})
    if not isinstance(evaluation_config, Mapping) or not isinstance(storage_config, Mapping):
        raise TypeError("Saved Round13-3 evaluation and storage settings must be mappings.")
    try:
        configured_output_budget = int(evaluation_config["max_output_bytes"])
        payload_budget = int(storage_config["max_output_bytes"])
        temporary_budget = int(storage_config["max_peak_temporary_bytes"])
        minimum_headroom = int(storage_config["minimum_free_headroom_bytes"])
        storage_output_root = str(storage_config["output_root"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Saved Round13-3 recipe must declare output_root and all storage/evaluation byte budgets.") from exc
    if min(configured_output_budget, payload_budget, temporary_budget) <= 0 or minimum_headroom < 0:
        raise ValueError("Saved Round13-3 storage/evaluation byte budgets must be positive, with nonnegative headroom.")
    max_output_bytes = min(configured_output_budget, payload_budget)
    storage = _storage_snapshot(
        output_path.parent,
        storage_output_root=storage_output_root,
        output_budget_bytes=max_output_bytes,
        payload_budget_bytes=payload_budget,
        temporary_budget_bytes=temporary_budget,
        minimum_free_headroom_bytes=minimum_headroom,
    )

    validation_ids = tuple(map(str, getattr(provider, "validation_case_ids", ())))
    training_ids = tuple(map(str, getattr(provider, "training_case_ids", ())))
    if not validation_ids or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Round13-3 provider must expose unique VAL89 case IDs.")
    if len(validation_ids) != 89 or len(training_ids) != 600:
        raise ValueError("Round13-3 native evidence requires the full selected TRAIN600 and exposed VAL89 memberships.")
    if set(validation_ids) & set(training_ids):
        raise ValueError("Thermal TRAIN and validation memberships must remain disjoint.")
    validation_counts = _case_counts(provider, "validation")
    training_counts = _case_counts(provider, "train")
    if set(validation_counts) != set(validation_ids) or set(training_counts) != set(training_ids):
        raise ValueError("Thermal input catalogs do not match provider TRAIN/validation memberships.")
    if str(provider_identity.get("validation_scope", "")) not in ("canonical_VAL89_exposed", "VAL89_exposed"):
        raise ValueError("Round13-3 native predictor evaluation requires the exposed canonical VAL89 scope.")

    validation_representatives = select_validation_representatives(provider, 3)
    low_m_train_ids = select_low_m_train_cases(provider, 4)
    cost_cases = {
        1: min((case_id for case_id in training_ids if training_counts[case_id] == 1), default=None),
        10: min((case_id for case_id in training_ids if training_counts[case_id] == 10), default=None),
    }
    if cost_cases[1] is None:
        raise ValueError("Round13-3 timing requires an input-selected TRAIN case with exactly M1.")
    if cost_cases[10] is None:
        raise ValueError("Round13-3 timing requires an input-selected TRAIN case with exactly M10.")
    route_arm = _route_interventions_enabled(model)
    detail_case_ids = (str(cost_cases[1]), *validation_representatives)
    validation_export_ids = set(validation_representatives)
    state_before = _state_snapshot(model)
    validation = _evaluate_native_split(
        model, provider, payload, "validation", validation_ids,
        export_ids=validation_export_ids, collect_routes=route_arm,
    )
    if route_arm and set(validation["route_traces"]) != validation_export_ids:
        raise ValueError("Native route traces did not cover every fixed VAL M3/M7/M10 representative case.")

    low_m_train = _evaluate_native_split(
        model, provider, payload, "train", low_m_train_ids,
        export_ids={str(cost_cases[1])}, collect_routes=route_arm,
    )
    if str(cost_cases[1]) not in low_m_train_ids:
        raise ValueError("Balanced low-M TRAIN panel must retain its selected M1 detail case.")
    if route_arm and set(low_m_train["route_traces"]) != {str(cost_cases[1])}:
        raise ValueError("Native route trace did not cover the fixed TRAIN M1 detail case.")
    _assert_model_state_unchanged(model, state_before, "native evaluation")

    selected_batches: dict[str, Any] = {}
    selected_predictions: dict[str, Any] = {}
    selected_query_indices: dict[str, tuple[int, ...]] = {}
    if route_arm:
        probe_cases = [("train", str(cost_cases[1]))]
        probe_cases.extend(("validation", case_id) for case_id in validation_representatives)
        for split, case_id in probe_cases:
            batch = next(provider.native_evaluation_batches(
                split=split, case_keys=(case_id,), microbatch_size=1,
            ))
            with torch.no_grad():
                prediction, _auxiliary = provider.predict_native(
                    model, batch.scene_inputs, batch.receivers, model.architecture, "validation",
                    epoch=int(payload.get("epoch", payload.get("current_epoch", 0))),
                )
            selected_batches[case_id] = batch
            selected_predictions[case_id] = prediction
            selected_query_indices[case_id] = _query_indices(batch.receivers, prediction)
        if set(selected_batches) != set(detail_case_ids):
            raise ValueError("Could not rebuild the fixed TRAIN M1 + VAL M3/M7/M10 probe panel.")

        expected_trace_rows = {
            str(cost_cases[1]): low_m_train["route_traces"][str(cost_cases[1])],
            **{case_id: validation["route_traces"][case_id]
               for case_id in validation_representatives},
        }
        for case_id, prediction in selected_predictions.items():
            query_indices = selected_query_indices[case_id]
            expected = expected_trace_rows[case_id]
            if query_indices != tuple(expected["query_indices"]):
                raise RuntimeError(f"Prepared union query rows changed while rebuilding fixed case {case_id}.")
            if not np.allclose(_query_coordinates(prediction, query_indices), expected["query_coordinates"]):
                raise RuntimeError(f"Prepared union query coordinates changed while rebuilding fixed case {case_id}.")
        base_route_traces = {
            str(cost_cases[1]): low_m_train["route_traces"][str(cost_cases[1])]["queries"],
            **{case_id: validation["route_traces"][case_id]["queries"]
               for case_id in validation_representatives},
        }
        state_before_probes = _state_snapshot(model)
        probe_results = _fixed_probe_evidence(
            model, provider, selected_batches, selected_predictions, selected_query_indices,
            base_route_traces,
        )
        _assert_model_state_unchanged(model, state_before_probes, "fixed-checkpoint probes")
    else:
        probe_results = {}
    state_before_benchmark = _state_snapshot(model)
    benchmark_fn = getattr(provider, "member_routing_benchmark", None)
    if not callable(benchmark_fn):
        raise TypeError("Member-routing provider must expose member_routing_benchmark for same-device costs.")
    cost_rows = {}
    for module_count, case_id in cost_cases.items():
        batch = next(provider.native_evaluation_batches(
            split="train", case_keys=(case_id,), microbatch_size=1,
        ))
        model.zero_grad(set_to_none=True)
        benchmark_result = benchmark_fn(model, batch, warmups=1, repetitions=3)
        model.zero_grad(set_to_none=True)
        cost_rows[f"M{module_count}"] = _validate_benchmark(benchmark_result, case_id=case_id)
        _assert_model_state_unchanged(model, state_before_benchmark, "cost benchmark")

    array_exports = {
        **validation.pop("representative_arrays"),
        **low_m_train.pop("representative_arrays"),
    }
    route_probe_result = {
        "status": "completed" if route_arm else "skipped",
        "case_ids": list(detail_case_ids),
        "query_indices": {key: list(value) for key, value in selected_query_indices.items()},
        "query_coordinates": {
            key: _query_coordinates(selected_predictions[key], selected_query_indices[key])
            for key in selected_query_indices
        },
        "probes": probe_results,
        "skip_reason": None if route_arm else (
            "R13-3-Full does not exercise route-only B/support interventions; compare native fields and costs "
            "against the paired R13-3-Route checkpoint."
        ),
        "probe_definitions": {
            "b_fixed_candidates_scores": "change B coefficients while freezing actual candidate original IDs, M proposals pi, and cheap member scores; compare beta, fields, and executed work",
            "geometry_b_comparable_counts": "geometry-only B/control with actual selected M/E counts matched and reported by the probe",
            "geometry_member_scores_fixed_b_pi": "replace member logits with geometry-only scores while fixed B and M proposals stay in force",
            "all_candidates_open": "dense-value execution control that evaluates every relation-eligible candidate value, including beta=0, under the same scores, priors, and normalization; prediction should agree with selected execution",
            "remove_contextual_message_recompute": "remove one genuinely applied contextual message through preparation and recompute dependent stages before decoding",
        } if route_arm else {},
        "fit_updates": 0,
        "solver_attempts": 0,
    }
    result = {
        "schema": RELATIVE_SCHEMA if model.architecture == "R13-3-Relative" else SCHEMA,
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", payload.get("current_epoch", 0))),
        "arm": str(payload.get("arm", model.architecture)),
        "workflow": workflow,
        "experiment_identity": {
            "config_name": identity.get("config_name"),
            "training_membership": identity.get("training_membership"),
            "validation_scope": provider_identity.get("validation_scope"),
            "normalization_fit_scope": provider_identity.get("normalization_fit_scope"),
        },
        "validation": validation,
        "low_m_train": {
            **{key: value for key, value in low_m_train.items() if key != "representative_arrays"},
            "selection": "first two stable case IDs at M1 and M2, then lowest remaining counts; input structure only",
            "selected_module_counts": {case_id: training_counts[case_id] for case_id in low_m_train_ids},
        },
        "route_probe_interventions": route_probe_result,
        "same_device_costs": {
            "device": str(torch.device(device)),
            "cost_cases_selected_by_input_module_count": {f"M{key}": value for key, value in cost_cases.items()},
            "case_measurements": cost_rows,
            "selection_value_context_work_semantics": {
                "selection_rows": "cheap query-to-original-member score rows",
                "value_rows": "actual selected original-member query/value executions after exact-positive selection and ID deduplication",
                "context_score_rows": "upstream directed contextual scoring, including E-to-E O(E^2) work",
            },
        },
        "evidence_policy": {
            "all_nine_native_observables": True,
            "per_case_equal_weight_and_module_count_strata": True,
            "saved_field_and_residual_arrays": sorted(array_exports),
            "three_original_id_route_traces_per_representative_case": route_arm,
            "route_trace_detail_case_ids": list(detail_case_ids) if route_arm else [],
            "route_trace_scope": "decoder member-read support is not the full input influence graph; upstream contextual dependencies, including E-to-E exchange, remain separate",
            "response_reference_scope": "any separate response diagnostic must use existing saved heat/geometry references only; this evaluator does not create or solve response references",
            "new_solver_calls": 0,
            "inverse_fit": False,
            "training_updates": 0,
            "training_updates_semantics": "retained fitting updates only; disposable training-macro optimizer updates are counted separately",
            "disposable_benchmark_optimizer_updates_discarded": sum(
                int(row.get("training_macro_optimizer_updates_discarded", 0)) for row in cost_rows.values()
            ),
            "disposable_benchmark_checkpoint_state_restored": True,
            "test_access": False,
            "dataset_copies": 0,
            "new_persistent_caches": 0,
        },
        "storage_budget": storage,
    }
    output_bytes = _write_result(
        output_path, result, array_exports,
        max_output_bytes=max_output_bytes, free_before=storage["free_before_bytes"],
    )
    return {"status": "completed", "output": str(output_path),
            "validation_case_count": len(validation_ids), "low_m_train_case_count": len(low_m_train_ids),
            "checkpoint_epoch": result["checkpoint_epoch"], "output_bytes": output_bytes}


__all__ = [
    "PROBE_NAMES",
    "evaluate_member_routing",
    "select_low_m_train_cases",
    "select_validation_representatives",
]
