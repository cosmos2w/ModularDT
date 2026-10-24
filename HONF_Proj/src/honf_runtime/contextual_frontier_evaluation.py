"""Checkpoint loading and bounded native evaluation for contextual-frontier runs."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .compat import load_trusted_checkpoint


_EVALUABLE_ARMS = ("J-D", "J-H")
_MAX_OUTPUT_BYTES = 256 * 1024**2
_OBSERVABLES = (
    "u", "v", "p", "omega", "temperature_fluid", "temperature_surface",
    "temperature_material", "q_proxy", "module_peak_endpoint_rmse",
)


def _contextual_checkpoint_parts(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping) or identity.get("workflow") != "contextual_frontier_joint_training_v1":
        raise ValueError("Checkpoint is not a maintained contextual-frontier joint-training checkpoint.")
    provider_identity = identity.get("provider_identity")
    if not isinstance(provider_identity, Mapping):
        raise ValueError("Contextual-frontier checkpoint is missing its provider identity.")
    if provider_identity.get("schema") != "thermal_contextual_frontier_provider_v1":
        raise ValueError("Checkpoint provider identity is not the Thermal contextual-frontier provider.")
    adapter_state = provider_identity.get("adapter_state")
    recipe = provider_identity.get("recipe")
    if not isinstance(adapter_state, Mapping) or not isinstance(recipe, Mapping):
        raise ValueError("Checkpoint must contain the adapter state payload and the original provider recipe.")
    arm = str(payload.get("arm", provider_identity.get("architecture", "")))
    if arm not in _EVALUABLE_ARMS or provider_identity.get("architecture") != arm:
        raise ValueError("Checkpoint arm and provider architecture must both be J-D or J-H and agree.")
    return identity, provider_identity, {"adapter_state": adapter_state, "recipe": recipe, "arm": arm}


def _bind_loaded_model(provider: Any, model: torch.nn.Module, device: torch.device) -> None:
    """Replace the factory's fresh model while keeping its selected-TRAIN provider."""
    provider.model = model
    provider.adapter = model
    provider.device = device
    organizer = getattr(provider, "organizer", None)
    if organizer is not None:
        organizer.model = model
        organizer.core = getattr(model, "core", model)
        organizer.provider = provider


def load_contextual_checkpoint(
    path: str | Path,
    device: str | torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Load a saved J-D/J-H model and rebuild its matching native data provider.

    The provider factory is run from the recipe embedded in the trusted engine
    checkpoint, which reuses the selected-TRAIN membership and normalization
    procedure. Its randomly initialized model is replaced by the saved adapter
    and exact checkpoint state before any evaluation batch is built.
    """
    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Maintained checkpoint loader did not return a mapping payload.")
    _identity, _provider_identity, saved = _contextual_checkpoint_parts(payload)

    # Import through the maintained runner so model/provider reconstruction
    # uses the same task factory as training and resume.
    from .contextual_frontier_training import _build_task
    from channelthermal.contextual_joint import ThermalContextualJointAdapter

    evaluation_device = torch.device(device)
    recipe = dict(saved["recipe"])
    arm = str(saved["arm"])
    _factory_model, provider = _build_task(recipe, arm, torch.device("cpu"))

    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("Contextual-frontier checkpoint has no model_state_dict.")
    model = ThermalContextualJointAdapter.from_state_payload(saved["adapter_state"], device="cpu")
    # assign=True preserves each checkpoint tensor's saved dtype instead of
    # silently copying it into the factory constructor's default float32.
    model.load_state_dict(model_state, strict=True, assign=True)
    model.to(evaluation_device)
    _bind_loaded_model(provider, model, evaluation_device)
    model.eval()
    payload["checkpoint_path"] = str(checkpoint_path)
    return model, provider, payload


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _representative_cases(provider: Any, case_ids: Sequence[str], limit: int) -> set[str]:
    """Choose input-only validation representatives, spread over module-count strata."""
    cases = getattr(provider, "validation_cases", ())
    module_count_by_id: dict[str, int] = {}
    for case in cases:
        case_id = str(case["case_id"])
        structure = case["structure"]
        present = np.asarray(structure["module_present"]) > 0.5
        module_count_by_id[case_id] = int(present.sum())
    if not module_count_by_id:
        return set(map(str, case_ids[:limit]))

    first_by_stratum: dict[int, str] = {}
    for case_id in map(str, case_ids):
        if case_id in module_count_by_id:
            first_by_stratum.setdefault(module_count_by_id[case_id], case_id)
    strata = sorted(first_by_stratum)
    if len(strata) > limit:
        indexes = np.linspace(0, len(strata) - 1, num=limit).round().astype(int)
        strata = [strata[int(index)] for index in indexes]
    return {first_by_stratum[count] for count in strata}


def _native_export_arrays(prediction: Any, batch: Any, case_id: str) -> dict[str, np.ndarray]:
    output = prediction.native_main
    prepared = prediction.native_prepared
    targets = batch.targets
    fluid_valid = prepared.stencils["fluid"].valid
    interface_shape = output["pred_interface"].shape[:-1]
    surface_valid = prepared.stencils["surface"].valid.reshape(interface_shape)
    outside_valid = prepared.stencils["outside"].valid.reshape(interface_shape)
    material_valid = prepared.stencils["material"].valid.reshape(targets.material_targets.shape)
    values = {
        "case_id": np.asarray(case_id),
        "pred_field": _as_numpy(output["pred_field"])[0],
        "target_field": _as_numpy(targets.field_targets)[0],
        "fluid_target_weight": _as_numpy(targets.point_weights)[0],
        "fluid_valid": _as_numpy(fluid_valid)[0],
        "pred_surface_temperature": _as_numpy(output["pred_interface"])[0, ..., 0],
        "target_surface_temperature": _as_numpy(targets.interface_target)[0, ..., 0],
        "pred_q_proxy": _as_numpy(output["pred_interface"])[0, ..., 1],
        "target_q_proxy": _as_numpy(targets.interface_target)[0, ..., 1],
        "surface_valid": _as_numpy(surface_valid)[0],
        "outside_valid": _as_numpy(outside_valid)[0],
        "interface_target_valid": _as_numpy(targets.interface_valid_mask)[0],
        "pred_material_temperature": _as_numpy(output["pred_internal_temperature"])[0, ..., 0],
        "target_material_temperature": _as_numpy(targets.material_targets)[0],
        "material_valid": _as_numpy(material_valid)[0],
        "module_present": _as_numpy(prepared.source_present)[0],
        "physical_heat": _as_numpy(batch.receivers.heat)[0],
    }
    return {name: np.asarray(value) for name, value in values.items()}


def _metric_value(row: Mapping[str, Any], name: str) -> float:
    if name in ("u", "v", "p", "omega"):
        value = row["flow_rmse"][name]
    elif name.startswith("temperature_"):
        role = name.removeprefix("temperature_")
        value = row["roles"][f"temperature_{role}"]["rmse"]
    elif name == "q_proxy":
        value = row["q_proxy_rmse"]
    elif name == "module_peak_endpoint_rmse":
        endpoint = row["roles"]["temperature_material_endpoint_maxima"]["per_module"]
        errors = np.asarray([float(item["error"]) for item in endpoint], dtype=np.float64)
        value = math.sqrt(float(np.mean(np.square(errors)))) if errors.size else math.nan
    else:
        raise KeyError(name)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Native evaluation returned a non-finite {name} metric for {row.get('case_id')}.")
    return result


def _summary(
    values: Sequence[tuple[str, int, float]],
    *,
    unit: str,
    value_definition: str,
    known_module_counts: Sequence[int] = (),
) -> dict[str, Any]:
    if not values:
        return {"case_count": 0, "unit": unit, "value_definition": value_definition}
    metric_values = np.asarray([value for _case_id, _module_count, value in values], dtype=np.float64)
    worst_index = int(np.argmax(metric_values))

    def aggregate(rows: Sequence[tuple[str, int, float]]) -> dict[str, Any]:
        numbers = np.asarray([value for _case_id, _module_count, value in rows], dtype=np.float64)
        local_worst = int(np.argmax(numbers))
        return {
            "case_count": int(len(rows)),
            "mean": float(numbers.mean()),
            "p90": float(np.percentile(numbers, 90)),
            "worst": float(numbers[local_worst]),
            "worst_case_id": str(rows[local_worst][0]),
        }

    by_stratum: dict[str, Any] = {
        f"M{module_count}": {
            "case_count": 0,
            "mean": None,
            "p90": None,
            "worst": None,
            "worst_case_id": None,
        }
        for module_count in sorted(set(map(int, known_module_counts)) | {item[1] for item in values})
    }
    for module_count in sorted({item[1] for item in values}):
        selected = [item for item in values if item[1] == module_count]
        by_stratum[f"M{module_count}"] = aggregate(selected)
    return {
        **aggregate(values),
        "unit": unit,
        "value_definition": value_definition,
        "by_module_count": by_stratum,
        "per_case": [
            {"case_id": case_id, "module_count": int(module_count), "value": float(value)}
            for case_id, module_count, value in values
        ],
    }


def _known_module_counts(provider: Any) -> tuple[int, ...]:
    counts = set()
    for attribute in ("training_cases", "validation_cases"):
        for case in getattr(provider, attribute, ()):
            present = np.asarray(case["structure"]["module_present"]) > 0.5
            counts.add(int(present.sum()))
    return tuple(sorted(counts))


def _summarize_nine_observables(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_module_counts: Sequence[int] = (),
) -> dict[str, Any]:
    units = {
        "u": "packed_dataset_native", "v": "packed_dataset_native", "p": "packed_dataset_native",
        "omega": "packed_dataset_native", "temperature_fluid": "packed_dataset_native_temperature",
        "temperature_surface": "packed_dataset_native_temperature",
        "temperature_material": "packed_dataset_native_temperature", "q_proxy": "packed_dataset_native_q_proxy",
        "module_peak_endpoint_rmse": "packed_dataset_native_temperature",
    }
    definitions = {
        name: "equal-case native RMSE" for name in _OBSERVABLES
    }
    definitions["module_peak_endpoint_rmse"] = (
        "per case, RMSE over signed errors of recomputed active-module endpoint maxima; ")
    definitions["module_peak_endpoint_rmse"] += (
        "per-module signed errors and worst modules remain in maintained_native_reduction"
    )
    result = {}
    for name in _OBSERVABLES:
        values = [(str(row["case_id"]), int(row["module_count"]), _metric_value(row, name)) for row in rows]
        result[name] = _summary(
            values,
            unit=units[name],
            value_definition=definitions[name],
            known_module_counts=known_module_counts,
        )
    return result


def _checkpoint_selection(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    name = path.name.lower()
    epoch = int(payload.get("epoch", payload.get("current_epoch", -1)))
    if name in ("best_by_field_mse_model.pt", "best_field.pt"):
        return {"kind": "best_field", "basis": "maintained best-field checkpoint alias", "epoch": epoch}
    run_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    summaries = (run_dir / "fit_summary.json", run_dir / "logs" / "fit_summary.json")
    summary = None
    for summary_path in summaries:
        if summary_path.is_file():
            try:
                candidate = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(candidate, Mapping):
                summary = candidate
                break
    if name in ("latest_model.pt", "latest.pt"):
        is_terminal = (
            summary
            and int(summary.get("completed_epoch", -2)) == epoch
            and summary.get("status") in ("completed", "clean_stopped")
        )
        if is_terminal:
            basis = f"latest alias matches fit_summary status={summary['status']}"
            return {"kind": "terminal", "basis": basis, "epoch": epoch}
        basis = "maintained latest checkpoint alias; terminal status not established"
        return {"kind": "latest", "basis": basis, "epoch": epoch}
    is_terminal = (
        summary
        and int(summary.get("completed_epoch", -2)) == epoch
        and summary.get("status") in ("completed", "clean_stopped")
    )
    if is_terminal:
        basis = f"saved epoch matches fit_summary status={summary['status']}"
        return {"kind": "terminal", "basis": basis, "epoch": epoch}
    return {"kind": "saved_epoch", "basis": "literal saved checkpoint path", "epoch": epoch}


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


def _filesystem_free_bytes(output_parent: Path) -> int:
    data2 = Path("/data2")
    if not data2.is_dir() or not os.path.ismount(data2):
        raise RuntimeError("/data2 is not mounted; refusing to write native evaluation evidence.")
    storage_root = (data2 / "wanglz/ModularDT").resolve(strict=True)
    resolved_parent = output_parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(storage_root):
        raise ValueError("Native evaluation evidence must be written beneath /data2/wanglz/ModularDT.")
    data2_device = data2.stat().st_dev
    if resolved_parent.stat().st_dev != data2_device:
        raise RuntimeError("Native evaluation output parent is not on the mounted /data2 filesystem.")
    return int(shutil.disk_usage(resolved_parent).free)


def evaluate_native(
    checkpoint: str | Path,
    device: str | torch.device,
    output: str | Path,
) -> dict[str, Any]:
    """Evaluate every exposed native validation case one at a time and save bounded evidence."""
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"Contextual-frontier evaluation output already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Evaluation output parent must exist before writing: {output_path.parent}")

    model, provider, payload = load_contextual_checkpoint(checkpoint_path, device)
    identity, provider_identity, saved = _contextual_checkpoint_parts(payload)
    recipe = saved["recipe"]
    evaluation = recipe.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise TypeError("Saved contextual recipe evaluation settings must be a mapping.")
    max_output_bytes = int(evaluation.get("max_output_bytes", _MAX_OUTPUT_BYTES))
    representative_limit = int(evaluation.get("representative_case_limit", 4))
    if not 1 <= max_output_bytes <= _MAX_OUTPUT_BYTES:
        raise ValueError("Contextual native evaluation output budget must be in (0, 256 MiB].")
    if not 0 <= representative_limit <= 8:
        raise ValueError("At most eight input-selected native representative cases may be exported.")
    storage = recipe.get("storage", {})
    if not isinstance(storage, Mapping):
        raise TypeError("Saved contextual recipe storage settings must be a mapping.")
    minimum_headroom = int(storage.get("minimum_free_headroom_bytes", 0))
    free_before = _filesystem_free_bytes(output_path.parent)
    if free_before - max_output_bytes < minimum_headroom:
        raise OSError("Native evaluation reservation would violate the configured filesystem free-space headroom.")

    case_ids = tuple(map(str, getattr(provider, "validation_case_ids", ())))
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("The rebuilt provider must expose a nonempty unique validation membership.")
    representative_ids = _representative_cases(provider, case_ids, representative_limit)
    representative_arrays: dict[str, dict[str, np.ndarray]] = {}
    metric_records: list[Mapping[str, Any]] = []
    per_case_rows: list[Mapping[str, Any]] = []
    work_totals: dict[str, float] = {}
    observed: list[str] = []
    epoch = int(payload.get("epoch", payload.get("current_epoch", 0)))
    arm = str(payload.get("arm", provider_identity.get("architecture", "")))
    model.core.set_organization_clock(epoch, 0)
    model.eval()

    with torch.no_grad():
        batches = provider.native_evaluation_batches(
            split="validation", case_keys=case_ids, microbatch_size=1,
        )
        for batch in batches:
            batch_ids = tuple(map(str, batch.targets.case_ids))
            if len(batch_ids) != 1 or tuple(map(str, batch.case_keys)) != batch_ids:
                raise ValueError("Native contextual validation must yield exactly one complete case per batch.")
            case_id = batch_ids[0]
            if len(observed) >= len(case_ids) or case_id != case_ids[len(observed)]:
                raise ValueError("Native validation case order or membership differs from the fixed provider panel.")
            prediction, auxiliary = provider.predict_native(
                model, batch.scene_inputs, batch.receivers, arm, "validation", epoch=epoch,
            )
            record = provider.validation_metrics(prediction, batch.targets, auxiliary)
            if not isinstance(record, Mapping):
                raise TypeError("Native validation metrics must return a record mapping.")
            metric_records.append(record)
            rows = record.get("case_rows")
            if isinstance(rows, Sequence):
                per_case_rows.extend(rows)
            elif isinstance(record.get("case_id"), str):
                per_case_rows.append(record)
            else:
                raise ValueError("Native validation metrics returned no per-case rows.")

            count_work = getattr(provider, "work_counts", None)
            counter = count_work(batch, prediction, auxiliary) if callable(count_work) else {}
            for name, value in dict(counter or {}).items():
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                    work_totals[str(name)] = work_totals.get(str(name), 0.0) + float(value)
            if case_id in representative_ids:
                representative_arrays[case_id] = _native_export_arrays(prediction, batch, case_id)
            observed.append(case_id)

    if tuple(observed) != case_ids:
        raise ValueError("Native contextual validation did not cover the complete exposed panel exactly once.")
    reducer = getattr(provider, "reduce_native_evaluation_metrics", None)
    if not callable(reducer):
        raise TypeError("The maintained contextual provider must expose reduce_native_evaluation_metrics().")
    reduced = dict(reducer(metric_records, split="validation", case_keys=case_ids))
    reduced_rows = reduced.get("per_case")
    if isinstance(reduced_rows, Sequence) and reduced_rows:
        per_case_rows = list(reduced_rows)
    if tuple(str(row.get("case_id")) for row in per_case_rows) != case_ids:
        raise ValueError("Native reducer did not preserve the exact validation case order.")
    metrics = _summarize_nine_observables(
        per_case_rows,
        known_module_counts=_known_module_counts(provider),
    )

    array_bytes = sum(int(value.nbytes) for arrays in representative_arrays.values() for value in arrays.values())
    result = {
        "schema": "thermal_contextual_frontier_native_evaluation_v1",
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": _checkpoint_selection(checkpoint_path, payload),
        "checkpoint_epoch": epoch,
        "arm": arm,
        "experiment_identity": {
            "workflow": identity.get("workflow"),
            "config_name": identity.get("config_name"),
            "training_membership": identity.get("training_membership"),
            "validation_scope": provider_identity.get("validation_scope"),
            "subset_manifest_sha256": provider_identity.get("subset_manifest_sha256"),
        },
        "split": "validation",
        "partition_scope": provider_identity.get("validation_scope"),
        "case_count": len(observed),
        "case_ids": observed,
        "metrics": metrics,
        "maintained_native_reduction": reduced,
        "executed_work_counts": {
            name: int(value) if value.is_integer() else value
            for name, value in sorted(work_totals.items())
        },
        "optimizer_updates": 0,
        "solver_attempts": 0,
        "test_access": False,
        "device": str(torch.device(device)),
        "representative_exports": [
            {"case_id": case_id, "module_count": int(next(
                int(row["module_count"]) for row in per_case_rows if str(row["case_id"]) == case_id
            )), "path": f"case_{re.sub(r'[^A-Za-z0-9_.-]', '_', case_id)}.npz"}
            for case_id in observed if case_id in representative_arrays
        ],
        "representative_selection": evaluation.get(
            "representative_selection",
            "first_validation_case_per_evenly_spaced_module_count_stratum",
        ),
        "representative_case_limit": representative_limit,
        "storage_budget": {
            "max_output_bytes": max_output_bytes,
            "representative_uncompressed_bytes": array_bytes,
            "peak_temporary_disk_bytes": 0,
        },
    }
    json_bytes = len(
        json.dumps(_jsonable(result), allow_nan=False, separators=(",", ":")).encode("utf-8")
    )
    estimated_bytes = array_bytes + json_bytes + 1024 * max(1, len(representative_arrays))
    if estimated_bytes > max_output_bytes:
        raise OSError(
            f"Native evaluation evidence estimate {estimated_bytes} exceeds its {max_output_bytes}-byte budget."
        )
    if free_before - estimated_bytes < minimum_headroom:
        raise OSError("Native evaluation output would violate the configured filesystem free-space headroom.")
    result["storage_budget"].update({
        "estimated_output_bytes": estimated_bytes,
        "free_before_bytes": free_before,
        "minimum_free_headroom_bytes": minimum_headroom,
    })

    output_path.mkdir(parents=False)
    try:
        for case_id, arrays in representative_arrays.items():
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", case_id)
            with (output_path / f"case_{safe_id}.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
        compressed_array_bytes = sum(
            path.stat().st_size for path in output_path.iterdir()
            if path.is_file() and path.suffix == ".npz"
        )
        result["storage_budget"]["representative_compressed_bytes"] = compressed_array_bytes
        result["storage_budget"]["free_after_representatives_bytes"] = _filesystem_free_bytes(output_path)
        with (output_path / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(_jsonable(result), stream, indent=2, allow_nan=False)
            stream.write("\n")
        for _ in range(3):
            actual_bytes = sum(path.stat().st_size for path in output_path.iterdir() if path.is_file())
            if result["storage_budget"].get("actual_output_bytes") == actual_bytes:
                break
            result["storage_budget"]["actual_output_bytes"] = actual_bytes
            result["storage_budget"]["free_after_bytes"] = _filesystem_free_bytes(output_path)
            with (output_path / "result.json").open("w", encoding="utf-8") as stream:
                json.dump(_jsonable(result), stream, indent=2, allow_nan=False)
                stream.write("\n")
    except BaseException:
        # This directory is created by this fresh evaluation call only.
        shutil.rmtree(output_path, ignore_errors=True)
        raise
    actual_bytes = sum(path.stat().st_size for path in output_path.iterdir() if path.is_file())
    if actual_bytes > max_output_bytes:
        shutil.rmtree(output_path, ignore_errors=True)
        raise OSError("Compressed native evaluation evidence exceeded its configured output budget.")
    return {"status": "completed", "output": str(output_path), "case_count": len(observed),
            "checkpoint_epoch": epoch, "arm": arm, "output_bytes": actual_bytes}


__all__ = ["evaluate_native", "load_contextual_checkpoint"]
