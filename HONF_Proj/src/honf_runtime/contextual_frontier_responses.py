"""Bounded physical response evaluation for contextual-frontier checkpoints.

The evaluator reuses the saved response-atlas stencils and the maintained
Thermal native adapter. It does not fit a response model or invoke a solver.
Response-fit, TRAIN-reference-only, and held-out response-validation families
are kept in separate summaries.
"""

from __future__ import annotations

import io
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.interaction_evidence.response_dataset import ResponseBlock

from .contextual_frontier_evaluation import (
    _checkpoint_selection,
    _filesystem_free_bytes,
    load_contextual_checkpoint,
)
from .thermal_geometry_supervision import (
    DEVELOPMENT_GEOMETRY_FAMILIES,
    MIXED_GEOMETRY_LABELS,
    NATIVE_ATLAS_THETA,
    SINGLE_GEOMETRY_LABELS,
    TRAIN_GEOMETRY_FAMILIES,
    _native_inputs,
    _sample_valid_rows,
    _validate_role_layout,
)

_MAX_OUTPUT_BYTES = 128 * 1024**2
_FLUID_QUERIES = 256
_SURFACE_THETA = 16
_FIT_FAMILY = "0348"
_TRAIN_REFERENCE_ONLY = ("0001", "0318", "0333")
_RESPONSE_VALIDATION = tuple(DEVELOPMENT_GEOMETRY_FAMILIES)
_ALL_FAMILIES = (*TRAIN_GEOMETRY_FAMILIES, *DEVELOPMENT_GEOMETRY_FAMILIES)
_ROLES = ("fluid_fields", "interface", "solid_temperature")
_HEAT_LABELS = ("heat_transfer_plus", "heat_transfer_minus")
_ZERO_HEAT_LABELS = ("zero_heat", "heat_zero", "zero_heat_control")
_PLOT_COMPONENTS = (
    ("fluid_fields", "u"),
    ("fluid_fields", "v"),
    ("fluid_fields", "p"),
    ("fluid_fields", "omega"),
    ("fluid_fields", "temperature"),
    ("interface", "T_surface"),
    ("interface", "q_normal"),
    ("solid_temperature", "temperature"),
    ("module_peak_endpoint", "temperature"),
)


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _atlas_paths(provider: Any, family_ids: Sequence[str] = _ALL_FAMILIES) -> tuple[Path, ...]:
    source = getattr(provider, "atlas_directory", None)
    if source is None:
        raise ValueError("The rebuilt contextual provider does not expose its response-atlas directory.")
    root = Path(source).expanduser().resolve(strict=True)
    family_root = root / "families" if (root / "families").is_dir() else root
    return tuple(
        (family_root / f"train_{family_id}_responses.npz").resolve(strict=True)
        for family_id in family_ids
    )


def _validate_atlas_family(stencil: Any, metadata: Mapping[str, Any], family_id: str) -> None:
    expected_physical_family = (
        "duplicate_family:0001+0273" if family_id == "0001" else f"stored_family:{family_id}"
    )
    if metadata.get("anchor_id") != family_id or stencil.anchor_id != family_id:
        raise ValueError(f"Response atlas anchor identity differs for {family_id}.")
    if metadata.get("family_id") != expected_physical_family or stencil.physical_family_id != expected_physical_family:
        raise ValueError(f"Response atlas physical family identity differs for {family_id}.")
    if metadata.get("split") != "train" or _value(stencil.split) != "train":
        raise ValueError(f"Response atlas {family_id} is not from the recorded source TRAIN partition.")
    if _value(stencil.source) != "reference_solver":
        raise ValueError(f"Response atlas {family_id} is not saved reference-solver evidence.")
    missing = sorted({*SINGLE_GEOMETRY_LABELS, *MIXED_GEOMETRY_LABELS, *_HEAT_LABELS} - set(stencil.variants))
    if missing:
        raise ValueError(f"Response atlas {family_id} lacks required saved variants {missing}.")
    module_ids = tuple(stencil.baseline.design.active_module_ids)
    _validate_role_layout(stencil, module_ids)
    if len(module_ids) != len(set(module_ids)) or not module_ids:
        raise ValueError(f"Response atlas {family_id} does not preserve unique original module IDs.")
    _saved_control_details(stencil)


def _saved_control_details(stencil: Any) -> dict[str, Any]:
    module_ids = tuple(stencil.baseline.design.active_module_ids)

    def design_inputs(label: str | None) -> tuple[np.ndarray, np.ndarray]:
        record = stencil.baseline if label is None else stencil.variants[label]
        modules = record.design.active_modules
        if tuple(record.design.active_module_ids) != module_ids:
            raise ValueError(f"Atlas {stencil.anchor_id}/{label} changed original module identities.")
        centers = np.asarray([module.position_xy for module in modules], dtype=np.float64)
        heat = np.asarray([module.heating for module in modules], dtype=np.float64)
        if centers.shape != (len(module_ids), 2) or heat.shape != (len(module_ids),):
            raise ValueError(f"Atlas {stencil.anchor_id}/{label} inputs do not align with module IDs.")
        return centers, heat

    baseline_centers, baseline_heat = design_inputs(None)
    geometry_details: dict[str, Any] = {}
    axis_slots: dict[str, int] = {}
    axis_indices: dict[str, int] = {}
    for label, family_axis in (("i_minus", "i"), ("i_plus", "i"), ("j_minus", "j"), ("j_plus", "j")):
        centers, heat = design_inputs(label)
        changed = np.flatnonzero(np.any(centers != baseline_centers, axis=1))
        if changed.size != 1 or not np.array_equal(heat, baseline_heat):
            raise ValueError(f"Atlas {stencil.anchor_id}/{label} must move one module without changing heat.")
        slot = int(changed[0])
        delta = centers[slot] - baseline_centers[slot]
        changed_axes = np.flatnonzero(delta != 0.0)
        if changed_axes.size != 1:
            raise ValueError(f"Atlas {stencil.anchor_id}/{label} must move along exactly one saved coordinate axis.")
        axis_index = int(changed_axes[0])
        if family_axis in axis_indices and axis_indices[family_axis] != axis_index:
            raise ValueError(f"Atlas {stencil.anchor_id} {family_axis}-axis signs move different coordinate axes.")
        if family_axis in axis_slots and axis_slots[family_axis] != slot:
            raise ValueError(f"Atlas {stencil.anchor_id} {family_axis}-axis signs move different module slots.")
        axis_indices[family_axis] = axis_index
        axis_slots[family_axis] = slot
        geometry_details[label] = {
            "moved_module_id": module_ids[slot],
            "axis": "x" if axis_index == 0 else "y",
            "signed_step": float(delta[axis_index]),
        }
    if geometry_details["i_minus"]["moved_module_id"] != geometry_details["i_plus"]["moved_module_id"]:
        raise ValueError(f"Atlas {stencil.anchor_id} i-axis signs move different original module slots.")
    if geometry_details["j_minus"]["moved_module_id"] != geometry_details["j_plus"]["moved_module_id"]:
        raise ValueError(f"Atlas {stencil.anchor_id} j-axis signs move different original module slots.")
    if geometry_details["i_minus"]["signed_step"] >= 0 or geometry_details["i_plus"]["signed_step"] <= 0:
        raise ValueError(f"Atlas {stencil.anchor_id} i-axis labels do not preserve their signed directions.")
    if geometry_details["j_minus"]["signed_step"] >= 0 or geometry_details["j_plus"]["signed_step"] <= 0:
        raise ValueError(f"Atlas {stencil.anchor_id} j-axis labels do not preserve their signed directions.")

    mixed_details: dict[str, Any] = {}
    mixed_components = {
        "mm": ("i_minus", "j_minus"), "mp": ("i_minus", "j_plus"),
        "pm": ("i_plus", "j_minus"), "pp": ("i_plus", "j_plus"),
    }
    for corner, (i_label, j_label) in mixed_components.items():
        centers, heat = design_inputs(corner)
        i_delta = design_inputs(i_label)[0] - baseline_centers
        j_delta = design_inputs(j_label)[0] - baseline_centers
        expected = baseline_centers + i_delta + j_delta
        if not np.array_equal(centers, expected) or not np.array_equal(heat, baseline_heat):
            raise ValueError(f"Atlas {stencil.anchor_id}/{corner} is not the declared two-axis corner.")
        mixed_details[corner] = {
            "i_label": i_label,
            "j_label": j_label,
            "i_axis": "x" if axis_indices["i"] == 0 else "y",
            "j_axis": "x" if axis_indices["j"] == 0 else "y",
        }

    heat_details: dict[str, Any] = {}
    for label in _HEAT_LABELS:
        centers, heat = design_inputs(label)
        if not np.array_equal(centers, baseline_centers) or np.array_equal(heat, baseline_heat):
            raise ValueError(f"Atlas {stencil.anchor_id}/{label} must change heat while holding geometry fixed.")
        heat_details[label] = {
            "per_module_heat_delta": {
                module_id: float(delta)
                for module_id, delta in zip(module_ids, heat - baseline_heat, strict=True)
            },
        }

    zero_heat_details: dict[str, Any] = {}
    for label in stencil.variants:
        if label not in _ZERO_HEAT_LABELS:
            continue
        centers, heat = design_inputs(label)
        if not np.array_equal(centers, baseline_centers) or not np.all(heat == 0.0):
            raise ValueError(f"Atlas {stencil.anchor_id}/{label} is not a saved zero-heat, fixed-geometry control.")
        zero_heat_details[label] = {"all_module_heating_is_zero": True}
    return {
        "single_geometry": geometry_details,
        "mixed_geometry_corners": mixed_details,
        "heat_transfer_controls": heat_details,
        "zero_heat_controls": zero_heat_details,
    }


def _load_stencils(provider: Any, family_ids: Sequence[str] = _ALL_FAMILIES) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    family_ids = tuple(map(str, family_ids))
    if not family_ids or len(family_ids) != len(set(family_ids)):
        raise ValueError("Physical response evaluation requires a fixed unique atlas family panel.")
    paths = _atlas_paths(provider, family_ids)
    stencils: dict[str, Any] = {}
    metadata_by_family: dict[str, dict[str, Any]] = {}
    seen_physical_families: set[str] = set()
    for family_id, path in zip(family_ids, paths, strict=True):
        stencil, metadata = load_response_atlas_stencil(path)
        _validate_atlas_family(stencil, metadata, family_id)
        if stencil.physical_family_id in seen_physical_families:
            raise ValueError("Response atlas families must preserve distinct physical-family identities.")
        seen_physical_families.add(stencil.physical_family_id)
        stencils[family_id] = stencil
        metadata_by_family[family_id] = dict(metadata)
    return stencils, metadata_by_family, str(paths[0].parent)


def _panel_for_family(stencil: Any, family_id: str) -> dict[str, Any]:
    baseline_roles = stencil.baseline.output.roles
    fluid = baseline_roles["fluid_fields"]
    interface = baseline_roles["interface"]
    material = baseline_roles["solid_temperature"]
    common_fluid = np.asarray(stencil.common_mask("fluid_fields"), dtype=bool)
    common_interface = np.asarray(stencil.common_mask("interface"), dtype=bool)
    common_material = np.asarray(stencil.common_mask("solid_temperature"), dtype=bool)
    module_ids = tuple(stencil.baseline.design.active_module_ids)
    module_count = len(module_ids)

    fluid_indices = _sample_valid_rows(
        common_fluid,
        _FLUID_QUERIES,
        seed=_panel_seed(family_id, "fluid"),
        all_channels=True,
    )
    if interface.values.shape[0] != module_count * NATIVE_ATLAS_THETA:
        raise ValueError(f"Response atlas {family_id} does not retain 64 surface rows per module.")
    if NATIVE_ATLAS_THETA % _SURFACE_THETA:
        raise ValueError("The native response surface panel must divide the saved 64-angle atlas.")
    stride = NATIVE_ATLAS_THETA // _SURFACE_THETA
    surface_indices = (
        np.arange(module_count * NATIVE_ATLAS_THETA)
        .reshape(module_count, NATIVE_ATLAS_THETA)[:, ::stride]
        .reshape(-1)
    )

    if material.values.shape[0] % module_count:
        raise ValueError(f"Response atlas {family_id} material rows are not module-major.")
    local_count = material.values.shape[0] // module_count
    material_features = material.query_features.reshape(module_count, local_count, 2)
    if not np.array_equal(material_features, np.broadcast_to(material_features[:1], material_features.shape)):
        raise ValueError(f"Response atlas {family_id} changed module-local material query coordinates.")
    # Use every saved material point so predicted and reference endpoint maxima
    # use the same common-valid original module-local receiver set.
    local_indices = np.arange(local_count, dtype=np.int64)
    material_indices = (
        np.arange(module_count, dtype=np.int64)[:, None] * local_count + local_indices[None, :]
    ).reshape(-1)
    common_material_by_module = common_material.reshape(module_count, local_count, -1)

    if fluid.receiver_module_ids is not None:
        raise ValueError("Eulerian fluid query rows must not be assigned to source module identities.")
    if interface.receiver_module_ids is None or material.receiver_module_ids is None:
        raise ValueError("Surface and material response rows must preserve original module identities.")

    return {
        "module_ids": module_ids,
        "module_count": module_count,
        "fluid_indices": fluid_indices,
        "surface_indices": surface_indices,
        "local_indices": local_indices,
        "material_indices": material_indices,
        "common_masks": {
            "fluid_fields": common_fluid,
            "interface": common_interface,
            "solid_temperature": common_material,
        },
        "material_common_by_module": common_material_by_module,
        "fluid_query_features": np.asarray(fluid.query_features[fluid_indices, :2], dtype=np.float32),
        "surface_query_features": np.asarray(interface.query_features[surface_indices], dtype=np.float32),
        "local_query_features": np.asarray(material_features[0, local_indices], dtype=np.float32),
        "fluid_query_ids": tuple(fluid.query_ids[int(index)] for index in fluid_indices),
        "surface_query_ids": tuple(interface.query_ids[int(index)] for index in surface_indices),
        "material_query_ids": tuple(material.query_ids[int(index)] for index in material_indices),
        "surface_receiver_module_ids": tuple(interface.receiver_module_ids[int(index)] for index in surface_indices),
        "material_receiver_module_ids": tuple(material.receiver_module_ids[int(index)] for index in material_indices),
    }


def _panel_seed(family_id: str, role: str) -> int:
    # Input-only and stable across arms; avoid selecting rows from model errors.
    import hashlib

    digest = hashlib.sha256(f"contextual_frontier_response:{family_id}:{role}:seed0".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _prediction_for_record(
    model: Any,
    record: Any,
    panel: Mapping[str, Any],
    *,
    family_id: str,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    structure, heat = _native_inputs(record, tuple(panel["module_ids"]), device, dtype)
    fluid_xy = torch.as_tensor(panel["fluid_query_features"][None], device=device, dtype=dtype)
    local_xy = torch.as_tensor(panel["local_query_features"][None], device=device, dtype=dtype)
    prepared = model.prepare_native(
        structure,
        fluid_xy,
        local_query_points=local_xy,
        ntheta=_SURFACE_THETA,
        scene_key=("contextual_frontier_response", family_id, label),
        physical_heat=heat,
    )
    output = model.apply_native(prepared, heat)
    if not isinstance(output, Mapping):
        raise TypeError("Thermal contextual native apply_native must return role outputs as a mapping.")
    required = ("pred_field", "pred_interface", "pred_internal_temperature")
    absent = [name for name in required if name not in output]
    if absent:
        raise ValueError(f"Thermal contextual adapter omitted response roles {absent}.")
    values = {
        "fluid_fields": _as_numpy(output["pred_field"])[0],
        "interface": _as_numpy(output["pred_interface"])[0].reshape(-1, 2),
        "solid_temperature": _as_numpy(output["pred_internal_temperature"])[0].reshape(-1, 1),
    }
    expected_shapes = {
        "fluid_fields": (len(panel["fluid_indices"]), 5),
        "interface": (len(panel["surface_indices"]), 2),
        "solid_temperature": (len(panel["material_indices"]), 1),
    }
    for role, expected in expected_shapes.items():
        if values[role].shape != expected:
            raise ValueError(
                f"Native response role {role} returned {values[role].shape}, expected {expected}."
            )
    return values


def _native_floors(model: Any, first_roles: Mapping[str, Any]) -> dict[str, float]:
    field_std = _as_numpy(model.field_std).reshape(-1).astype(np.float64)
    interface_std = _as_numpy(model.interface_target_std).reshape(-1).astype(np.float64)
    material_std = _as_numpy(model.internal_temperature_std).reshape(-1).astype(np.float64)
    if field_std.shape != (5,) or interface_std.shape != (2,) or material_std.size < 1:
        raise ValueError("Saved checkpoint normalization does not define the full Thermal native response scales.")
    native_std = {
        ("fluid_fields", "u"): field_std[0],
        ("fluid_fields", "v"): field_std[1],
        ("fluid_fields", "p"): field_std[2],
        ("fluid_fields", "omega"): field_std[3],
        ("fluid_fields", "temperature"): field_std[4],
        ("interface", "T_surface"): interface_std[0],
        ("interface", "q_normal"): interface_std[1],
        ("solid_temperature", "temperature"): material_std[0],
    }
    floors: dict[str, float] = {}
    for role in _ROLES:
        for name in first_roles[role].channel_names:
            std = float(native_std[(role, name)])
            if not math.isfinite(std) or std <= 0.0:
                raise ValueError(f"Checkpoint TRAIN-native scale is invalid for {role}/{name}.")
            floors[f"{role}/{name}"] = max(1.0e-8, 1.0e-3 * std)
    floors["module_peak_endpoint/temperature"] = max(1.0e-8, 1.0e-3 * float(material_std[0]))
    return floors


def _response_native_floors(
    model: Any,
    provider: Any,
    first_roles: Mapping[str, Any],
) -> tuple[dict[str, float], tuple[str, ...], str]:
    floors = _native_floors(model, first_roles)
    response_supervision = getattr(provider, "responses", None)
    admitted_floors = getattr(response_supervision, "floors", None)
    admitted_families = tuple(map(str, getattr(response_supervision, "family_ids", ())))
    if isinstance(admitted_floors, Mapping) and admitted_floors:
        expected_components = set(floors) - {"module_peak_endpoint/temperature"}
        if set(admitted_floors) != expected_components or not admitted_families:
            raise ValueError("Provider response-native floors must cover all native fields and declare their TRAIN fit families.")
        if not set(admitted_families).issubset(TRAIN_GEOMETRY_FAMILIES):
            raise ValueError("Response-native floors may use only the declared original-TRAIN response families.")
        if getattr(provider, "protocol", None) == "development" and admitted_families != (_FIT_FAMILY,):
            raise ValueError("Development response floors must come from the selected-TRAIN family 0348 only.")
        for component, floor in admitted_floors.items():
            if component in floors:
                value = float(floor)
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"Admitted TRAIN response native floor is invalid for {component}.")
                floors[component] = value
        source = "maintained provider.responses.floors from admitted TRAIN response families"
    else:
        source = "selected-TRAIN checkpoint normalization buffers"
    return floors, admitted_families, source


def _component_metric(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    *,
    unit: str,
    native_floor: float,
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    physical = np.asarray(valid, dtype=bool).reshape(-1)
    if pred.shape != ref.shape or physical.shape != pred.shape:
        raise ValueError("Physical response metric requires aligned predictions, references, and valid rows.")
    reference_mask = physical & np.isfinite(ref)
    nonfinite_reference_count = int((physical & ~np.isfinite(ref)).sum())
    nonfinite_prediction_count = int((reference_mask & ~np.isfinite(pred)).sum())
    if not reference_mask.any():
        return {
            "status": "invalid_no_finite_reference",
            "valid_scalar_count": 0,
            "nonfinite_prediction_count": 0,
            "nonfinite_reference_count": nonfinite_reference_count,
            "unit": unit,
            "absolute_mae": None,
            "absolute_rmse": None,
            "reference_rms": None,
            "prediction_rms": None,
            "train_native_floor": float(native_floor),
            "relative_rmse_to_reference_rms": None,
            "relative_metric_applicable": False,
            "relative_metric_reason": "no_finite_reference_on_physical_support",
        }
    selected_ref = ref[reference_mask]
    reference_rms = float(np.sqrt(np.mean(np.square(selected_ref))))
    if nonfinite_prediction_count:
        return {
            "status": "invalid_nonfinite_prediction",
            "valid_scalar_count": int(reference_mask.sum()),
            "nonfinite_prediction_count": nonfinite_prediction_count,
            "nonfinite_reference_count": nonfinite_reference_count,
            "unit": unit,
            "absolute_mae": None,
            "absolute_rmse": None,
            "reference_rms": reference_rms,
            "prediction_rms": None,
            "train_native_floor": float(native_floor),
            "relative_rmse_to_reference_rms": None,
            "relative_metric_applicable": False,
            "relative_metric_reason": "nonfinite_prediction_on_finite_reference_support",
        }
    selected_pred = pred[reference_mask]
    error = selected_pred - selected_ref
    prediction_rms = float(np.sqrt(np.mean(np.square(selected_pred))))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    relative_allowed = reference_rms > native_floor
    return {
        "status": "valid",
        "valid_scalar_count": int(reference_mask.sum()),
        "nonfinite_prediction_count": 0,
        "nonfinite_reference_count": nonfinite_reference_count,
        "unit": unit,
        "absolute_mae": float(np.mean(np.abs(error))),
        "absolute_rmse": rmse,
        "reference_rms": reference_rms,
        "prediction_rms": prediction_rms,
        "train_native_floor": float(native_floor),
        "relative_rmse_to_reference_rms": rmse / reference_rms if relative_allowed else None,
        "relative_metric_applicable": bool(relative_allowed),
        "relative_metric_reason": None if relative_allowed else "reference_rms_not_above_train_native_floor",
    }


def _response_kind(label: str) -> str:
    if label in SINGLE_GEOMETRY_LABELS:
        return "single_geometry_finite_change"
    if label in _HEAT_LABELS:
        return "heat_transfer_finite_control"
    if label in _ZERO_HEAT_LABELS:
        return "saved_zero_heat_control"
    if label == "mixed_raw_four_corner_contrast":
        return "centered_mixed_geometry_four_corner_contrast"
    return "centered_mixed_geometry_derivative"


def _axis_step(stencil: Any, minus_label: str, plus_label: str) -> tuple[float, str]:
    baseline = np.asarray([module.position_xy for module in stencil.baseline.design.active_modules], dtype=np.float64)
    minus = np.asarray([module.position_xy for module in stencil.variants[minus_label].design.active_modules], dtype=np.float64)
    plus = np.asarray([module.position_xy for module in stencil.variants[plus_label].design.active_modules], dtype=np.float64)
    minus_slots = np.flatnonzero(np.any(minus != baseline, axis=1))
    plus_slots = np.flatnonzero(np.any(plus != baseline, axis=1))
    if minus_slots.size != 1 or plus_slots.size != 1 or minus_slots[0] != plus_slots[0]:
        raise ValueError(f"Atlas {stencil.anchor_id} {minus_label}/{plus_label} do not move one shared module slot.")
    slot = int(minus_slots[0])
    minus_axes = np.flatnonzero(minus[slot] != baseline[slot])
    plus_axes = np.flatnonzero(plus[slot] != baseline[slot])
    if minus_axes.size != 1 or plus_axes.size != 1 or minus_axes[0] != plus_axes[0]:
        raise ValueError(f"Atlas {stencil.anchor_id} {minus_label}/{plus_label} do not use one shared coordinate axis.")
    axis = int(minus_axes[0])
    h = float((plus[slot, axis] - minus[slot, axis]) / 2.0)
    if minus[slot, axis] >= baseline[slot, axis] or plus[slot, axis] <= baseline[slot, axis]:
        raise ValueError(f"Atlas {stencil.anchor_id} {minus_label}/{plus_label} labels have inconsistent signs.")
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"Atlas {stencil.anchor_id} has invalid centered steps.")
    return h, "x" if axis == 0 else "y"


def _mixed_steps(stencil: Any) -> tuple[float, float]:
    h_i, _axis_i = _axis_step(stencil, "i_minus", "i_plus")
    h_j, _axis_j = _axis_step(stencil, "j_minus", "j_plus")
    if not math.isfinite(h_i) or not math.isfinite(h_j) or h_i <= 0.0 or h_j <= 0.0:
        raise ValueError(f"Atlas {stencil.anchor_id} has invalid centered mixed-difference steps.")
    return h_i, h_j


def _mixed_axes(stencil: Any) -> tuple[str, str]:
    return _axis_step(stencil, "i_minus", "i_plus")[1], _axis_step(stencil, "j_minus", "j_plus")[1]


def _raw_mixed_block(stencil: Any, role: str) -> ResponseBlock:
    baseline = stencil.baseline.output.roles[role]
    corners = [stencil.variants[label].output.roles[role] for label in ("pp", "pm", "mp", "mm")]
    valid = np.asarray(stencil.common_mask(role), dtype=bool)
    delta = np.full(baseline.values.shape, np.nan, dtype=np.float64)
    pp, pm, mp, mm = (corner.values for corner in corners)
    delta[valid] = pp[valid] - pm[valid] - mp[valid] + mm[valid]
    return ResponseBlock(
        role=role,
        label="mixed_raw_four_corner_contrast",
        query_features=baseline.query_features,
        delta=delta,
        valid_mask=valid,
        quadrature_weights=baseline.quadrature_weights,
        channel_names=baseline.channel_names,
        channel_units=baseline.channel_units,
        query_ids=baseline.query_ids,
        receiver_module_ids=baseline.receiver_module_ids,
    )


def _target_block(stencil: Any, label: str, role: str, steps: tuple[float, float] | None) -> ResponseBlock:
    if label == "mixed_raw_four_corner_contrast":
        return _raw_mixed_block(stencil, role)
    if label == "mixed_central_derivative":
        if steps is None:
            raise ValueError("Centered mixed response requires the saved physical axis steps.")
        return stencil.centered_mixed_difference(
            role=role,
            joint_pp="pp",
            joint_pm="pm",
            joint_mp="mp",
            joint_mm="mm",
            first_step=steps[0],
            second_step=steps[1],
        )
    return stencil.finite_change(label, role)


def _response_specs(stencil: Any) -> tuple[tuple[str, str], ...]:
    labels = [*SINGLE_GEOMETRY_LABELS, *_HEAT_LABELS]
    labels.extend(label for label in stencil.variants if label in _ZERO_HEAT_LABELS)
    has_mixed = set(MIXED_GEOMETRY_LABELS).issubset(stencil.variants)
    specs = [(label, label) for label in labels]
    if has_mixed:
        specs.append(("mixed_central_derivative", "mixed_central_derivative"))
        specs.append(("mixed_raw_four_corner_contrast", "mixed_raw_four_corner_contrast"))
    return tuple(specs)


def _difference(
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    response_label: str,
    role: str,
    steps: tuple[float, float] | None,
) -> np.ndarray:
    if response_label not in ("mixed_central_derivative", "mixed_raw_four_corner_contrast"):
        return predictions[response_label][role] - predictions["baseline"][role]
    assert steps is not None
    contrast = (
        predictions["pp"][role]
        - predictions["pm"][role]
        - predictions["mp"][role]
        + predictions["mm"][role]
    )
    if response_label == "mixed_raw_four_corner_contrast":
        return contrast
    return contrast / (4.0 * steps[0] * steps[1])


def _reference_endpoint_by_state(stencil: Any, panel: Mapping[str, Any]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    local_count = len(panel["local_indices"])
    module_count = int(panel["module_count"])
    common = np.asarray(panel["material_common_by_module"], dtype=bool)[..., 0]
    for label, record in (("baseline", stencil.baseline), *stencil.variants.items()):
        role = record.output.roles["solid_temperature"]
        role_values = np.asarray(role.values, dtype=np.float64).reshape(module_count, local_count, -1)[..., 0]
        peaks = []
        for module_index in range(module_count):
            mask = common[module_index]
            if not mask.any():
                raise ValueError(
                    f"Atlas {stencil.anchor_id}/{label} has no common valid material points for module "
                    f"{panel['module_ids'][module_index]}."
                )
            selected = role_values[module_index, mask]
            peaks.append(float(np.max(selected)) if np.isfinite(selected).all() else float("nan"))
        values[label] = np.asarray(peaks, dtype=np.float64)
    return values


def _prediction_endpoint_by_state(
    predictions: Mapping[str, Mapping[str, np.ndarray]], panel: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    module_count = int(panel["module_count"])
    local_count = len(panel["local_indices"])
    common = np.asarray(panel["material_common_by_module"], dtype=bool)[..., 0]
    for label, role_values in predictions.items():
        material = np.asarray(role_values["solid_temperature"], dtype=np.float64).reshape(module_count, local_count, -1)[..., 0]
        peaks = []
        for module_index in range(module_count):
            mask = common[module_index]
            if not mask.any():
                raise ValueError(
                    f"Prediction {label} has no common valid material points for module "
                    f"{panel['module_ids'][module_index]}."
                )
            selected = material[module_index, mask]
            peaks.append(float(np.max(selected)) if np.isfinite(selected).all() else float("nan"))
        values[label] = np.asarray(peaks, dtype=np.float64)
    return values


def _endpoint_difference(
    endpoint_by_state: Mapping[str, np.ndarray],
    label: str,
    steps: tuple[float, float] | None,
) -> np.ndarray:
    if label not in ("mixed_central_derivative", "mixed_raw_four_corner_contrast"):
        return endpoint_by_state[label] - endpoint_by_state["baseline"]
    assert steps is not None
    contrast = (
        endpoint_by_state["pp"] - endpoint_by_state["pm"] - endpoint_by_state["mp"] + endpoint_by_state["mm"]
    )
    if label == "mixed_raw_four_corner_contrast":
        return contrast
    return contrast / (4.0 * steps[0] * steps[1])


def _evaluate_family(
    model: Any,
    stencil: Any,
    family_id: str,
    *,
    fit_families: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    floors: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    panel = _panel_for_family(stencil, family_id)
    steps = _mixed_steps(stencil)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    records = {"baseline": stencil.baseline, **dict(stencil.variants)}
    used_labels = {"baseline"}
    for response_label, _target_label in _response_specs(stencil):
        if response_label == "mixed_central_derivative":
            used_labels.update(MIXED_GEOMETRY_LABELS)
        else:
            used_labels.add(response_label)
    for label in ("baseline", *stencil.variants.keys()):
        if label not in used_labels:
            continue
        predictions[label] = _prediction_for_record(
            model, records[label], panel,
            family_id=family_id, label=label, device=device, dtype=dtype,
        )

    reference_endpoints = _reference_endpoint_by_state(stencil, panel)
    prediction_endpoints = _prediction_endpoint_by_state(predictions, panel)
    family_rows: dict[str, Any] = {
        "family_id": family_id,
        "physical_family_id": stencil.physical_family_id,
        "source_partition": _value(stencil.split),
        "source": _value(stencil.source),
        "response_evaluation_cohort": _cohort_for_family(family_id, fit_families),
        "module_ids": list(panel["module_ids"]),
        "module_count": panel["module_count"],
        "saved_controls": _saved_control_details(stencil),
        "query_panel": {
            "selection_basis": "fixed input-only common-valid atlas receiver coordinates; seed=0",
            "fluid_query_count": len(panel["fluid_indices"]),
            "fluid_query_ids": list(panel["fluid_query_ids"]),
            "surface_theta_per_module": _SURFACE_THETA,
            "surface_query_count": len(panel["surface_indices"]),
            "surface_query_ids": list(panel["surface_query_ids"]),
            "material_local_points_per_module": len(panel["local_indices"]),
            "material_query_count": len(panel["material_indices"]),
            "material_endpoint_basis": "all saved local material receiver points on common valid original-module rows",
            "material_query_ids": list(panel["material_query_ids"]),
            "surface_receiver_module_ids": list(panel["surface_receiver_module_ids"]),
            "material_receiver_module_ids": list(panel["material_receiver_module_ids"]),
        },
        "mixed_geometry_steps": {
            "i_axis_half_step": steps[0],
            "j_axis_half_step": steps[1],
            "i_coordinate_axis": _mixed_axes(stencil)[0],
            "j_coordinate_axis": _mixed_axes(stencil)[1],
        },
        "responses": {},
    }
    panel_arrays: dict[str, Any] = {
        "panel": panel,
        "response_data": {},
    }
    family_records: dict[str, dict[str, Any]] = {}
    for response_label, target_label in _response_specs(stencil):
        label = response_label
        details: dict[str, Any] = {
            "kind": _response_kind(label),
            "response_definition": (
                "(pp - pm - mp + mm) / (4 * i_axis_half_step * j_axis_half_step)"
                if label == "mixed_central_derivative"
                else "pp - pm - mp + mm (unnormalized four-corner contrast)"
                if label == "mixed_raw_four_corner_contrast"
                else "saved trial native output minus saved baseline native output"
            ),
            "components": {},
        }
        if label in ("mixed_central_derivative", "mixed_raw_four_corner_contrast"):
            i_axis, j_axis = _mixed_axes(stencil)
            details["mixed_step_denominator"] = {
                "formula": "4 * i_axis_half_step * j_axis_half_step",
                "value": 4.0 * steps[0] * steps[1],
                "i_axis_half_step": steps[0],
                "j_axis_half_step": steps[1],
                "i_coordinate_axis": i_axis,
                "j_coordinate_axis": j_axis,
                "position_coordinate_units": "saved atlas position-coordinate units",
                "derivative_output_units": "native output units per saved position-coordinate unit squared",
                "raw_contrast_output_units": "native output units",
            }
        response_panel: dict[str, Any] = {}
        for role in _ROLES:
            is_mixed = label in ("mixed_central_derivative", "mixed_raw_four_corner_contrast")
            block = _target_block(stencil, target_label, role, steps if is_mixed else None)
            indices = {
                "fluid_fields": panel["fluid_indices"],
                "interface": panel["surface_indices"],
                "solid_temperature": panel["material_indices"],
            }[role]
            predicted_full = _difference(predictions, label, role, steps if is_mixed else None)
            predicted = np.asarray(predicted_full, dtype=np.float64)
            reference = np.asarray(block.delta[indices], dtype=np.float64)
            common = np.asarray(panel["common_masks"][role][indices], dtype=bool)
            valid = common & np.asarray(block.valid_mask[indices], dtype=bool)
            if predicted.shape != reference.shape or predicted.shape != valid.shape:
                raise ValueError(f"Response {family_id}/{label}/{role} outputs lost native receiver alignment.")
            role_panel = {
                "prediction_delta": predicted.astype(np.float32),
                "reference_delta": reference.astype(np.float32),
                "valid_mask": valid,
            }
            response_panel[role] = role_panel
            for channel_index, channel_name in enumerate(block.channel_names):
                component = f"{role}/{channel_name}"
                metrics = _component_metric(
                    predicted[:, channel_index],
                    reference[:, channel_index],
                    valid[:, channel_index],
                    unit=(
                        f"{block.channel_units[channel_index]} per saved position-coordinate unit squared"
                        if label == "mixed_central_derivative"
                        else block.channel_units[channel_index]
                    ),
                    native_floor=float(floors[component]),
                )
                details["components"][component] = metrics
                family_records.setdefault(label, {})[component] = metrics

        endpoint_reference = _endpoint_difference(
            reference_endpoints, label,
            steps if label in ("mixed_central_derivative", "mixed_raw_four_corner_contrast") else None,
        )
        endpoint_prediction = _endpoint_difference(
            prediction_endpoints, label,
            steps if label in ("mixed_central_derivative", "mixed_raw_four_corner_contrast") else None,
        )
        endpoint_rows = [
            {
                "module_id": module_id,
                "reference_delta": float(reference_value),
                "prediction_delta": float(prediction_value),
                "error": float(prediction_value - reference_value),
            }
            for module_id, reference_value, prediction_value in zip(
                panel["module_ids"], endpoint_reference, endpoint_prediction, strict=True,
            )
        ]
        endpoint_metrics = _component_metric(
            endpoint_prediction,
            endpoint_reference,
            np.ones_like(endpoint_reference, dtype=bool),
            unit=(
                "dataset temperature units per saved position-coordinate unit squared"
                if label == "mixed_central_derivative"
                else "dataset temperature units"
            ),
            native_floor=float(floors["module_peak_endpoint/temperature"]),
        )
        endpoint_metrics["endpoint_basis"] = "max over all common-valid saved local material points per original module"
        endpoint_metrics["prediction_endpoint_source"] = "native adapter output over the same local material coordinates"
        endpoint_metrics["reference_endpoint_source"] = "saved reference-solver local material field over the same coordinates"
        endpoint_metrics["per_module"] = endpoint_rows
        details["components"]["module_peak_endpoint/temperature"] = endpoint_metrics
        family_records.setdefault(label, {})["module_peak_endpoint/temperature"] = endpoint_metrics
        response_panel["module_peak_endpoint"] = {
            "prediction_delta": endpoint_prediction.astype(np.float32),
            "reference_delta": endpoint_reference.astype(np.float32),
            "valid_mask": np.ones_like(endpoint_reference, dtype=bool),
        }
        family_rows["responses"][label] = details
        panel_arrays["response_data"][label] = response_panel

    return family_rows, family_records, panel_arrays


def _cohort_plan(fit_families: Sequence[str]) -> dict[str, list[str]]:
    fit_set = set(map(str, fit_families))
    fit = tuple(family_id for family_id in TRAIN_GEOMETRY_FAMILIES if family_id in fit_set)
    if fit:
        fit_key = "fit_train_0348" if fit == (_FIT_FAMILY,) else "fit_train"
        cohorts: dict[str, list[str]] = {fit_key: list(fit)}
        train_reference_only = tuple(family_id for family_id in TRAIN_GEOMETRY_FAMILIES if family_id not in fit_set)
        if train_reference_only:
            cohorts["train_reference_only"] = list(train_reference_only)
    else:
        # A native-only checkpoint may be evaluated against the saved physical
        # response atlas without fitting any response roles. Keep every source
        # TRAIN family explicitly in the reference-only cohort; do not expand
        # this fixed panel into held-out response-validation atlases.
        cohorts = {"train_reference_only": list(TRAIN_GEOMETRY_FAMILIES)}
        cohorts["response_validation_only"] = []
    if fit:
        cohorts["response_validation_only"] = list(_RESPONSE_VALIDATION)
    return cohorts


def _cohort_for_family(family_id: str, fit_families: Sequence[str]) -> str:
    for cohort, family_ids in _cohort_plan(fit_families).items():
        if family_id in family_ids:
            return cohort
    raise ValueError(f"Unassigned response atlas family {family_id}.")


def _summarize_families(
    family_results: Mapping[str, Mapping[str, Any]],
    family_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    fit_families: Sequence[str],
) -> dict[str, Any]:
    cohorts: dict[str, Any] = {}
    by_cohort = _cohort_plan(fit_families)
    for cohort, family_ids in by_cohort.items():
        label_components: dict[str, dict[str, list[tuple[str, Mapping[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
        for family_id in family_ids:
            for label, components in family_metrics[family_id].items():
                for component, metrics in components.items():
                    label_components[label][component].append((family_id, metrics))
        label_summary: dict[str, Any] = {}
        for label, components in label_components.items():
            component_summary: dict[str, Any] = {}
            for component, family_values in components.items():
                metric_names = (
                    "absolute_mae", "absolute_rmse", "reference_rms", "prediction_rms",
                    "relative_rmse_to_reference_rms",
                )
                summary: dict[str, Any] = {"family_count": len(family_values), "aggregation": "equal-family mean and p90"}
                for metric_name in metric_names:
                    valid_rows = [
                        (family_id, float(metrics[metric_name]))
                        for family_id, metrics in family_values
                        if metrics.get(metric_name) is not None
                    ]
                    values = np.asarray([row[1] for row in valid_rows], dtype=np.float64)
                    summary[metric_name] = {
                        "count": int(values.size),
                        "mean": float(values.mean()) if values.size else None,
                        "p90": float(np.percentile(values, 90)) if values.size else None,
                        "family_ids": [row[0] for row in valid_rows],
                    }
                summary["relative_rmse_excluded_family_ids"] = [
                    family_id for family_id, metrics in family_values
                    if not metrics.get("relative_metric_applicable", False)
                ]
                summary["invalid_family_ids"] = [
                    family_id for family_id, metrics in family_values
                    if metrics.get("status") != "valid"
                ]
                component_summary[component] = summary
            label_summary[label] = component_summary
        cohorts[cohort] = {
            "family_ids": family_ids,
            "responses": label_summary,
        }
    return cohorts


def _representative_families(
    family_results: Mapping[str, Mapping[str, Any]],
    fit_families: Sequence[str],
    limit: int = 3,
) -> tuple[str, ...]:
    if limit < 1:
        return ()
    cohorts = _cohort_plan(fit_families)
    fit_key = next((key for key in cohorts if key.startswith("fit_train")), None)
    source_key = fit_key or "train_reference_only"
    source_by_module_count: dict[int, str] = {}
    for family_id in cohorts[source_key]:
        source_by_module_count.setdefault(int(family_results[family_id]["module_count"]), family_id)
    source_counts = sorted(source_by_module_count)
    preferred = [source_by_module_count[source_counts[0]]] if source_counts else []
    if len(source_counts) > 1:
        preferred.append(source_by_module_count[source_counts[-1]])
    if fit_key is not None and "train_reference_only" in cohorts:
        preferred.append(cohorts["train_reference_only"][0])
    if cohorts.get("response_validation_only"):
        preferred.append(cohorts["response_validation_only"][0])
    selected = [family_id for family_id in preferred if family_id in family_results][:limit]
    by_module_count: dict[int, str] = {}
    for family_id, row in family_results.items():
        by_module_count.setdefault(int(row["module_count"]), family_id)
    module_counts = sorted(by_module_count)
    for module_count in module_counts:
        family_id = by_module_count[module_count]
        if family_id not in selected and len(selected) < limit:
            selected.append(family_id)
    return tuple(selected)


def _panel_export_bytes(
    representative_ids: Sequence[str],
    panels: Mapping[str, Mapping[str, Any]],
    family_results: Mapping[str, Mapping[str, Any]],
) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    for family_id in representative_ids:
        prefix = f"family_{family_id}__"
        panel = panels[family_id]["panel"]
        arrays[prefix + "module_ids"] = np.asarray(panel["module_ids"], dtype="U")
        arrays[prefix + "fluid_query_features"] = np.asarray(panel["fluid_query_features"], dtype=np.float32)
        arrays[prefix + "fluid_query_ids"] = np.asarray(panel["fluid_query_ids"], dtype="U")
        arrays[prefix + "surface_query_features"] = np.asarray(panel["surface_query_features"], dtype=np.float32)
        arrays[prefix + "surface_query_ids"] = np.asarray(panel["surface_query_ids"], dtype="U")
        arrays[prefix + "surface_receiver_module_ids"] = np.asarray(panel["surface_receiver_module_ids"], dtype="U")
        arrays[prefix + "local_query_features"] = np.asarray(panel["local_query_features"], dtype=np.float32)
        arrays[prefix + "material_query_ids"] = np.asarray(panel["material_query_ids"], dtype="U")
        arrays[prefix + "material_receiver_module_ids"] = np.asarray(panel["material_receiver_module_ids"], dtype="U")
        for label, roles in panels[family_id]["response_data"].items():
            response_prefix = prefix + f"response_{_safe_name(label)}__"
            for role, fields in roles.items():
                safe_role = _safe_name(role)
                for field_name in ("prediction_delta", "reference_delta", "valid_mask"):
                    arrays[response_prefix + f"{safe_role}_{field_name}"] = np.asarray(fields[field_name])
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _render_response_plots(
    representative_ids: Sequence[str],
    panels: Mapping[str, Mapping[str, Any]],
) -> dict[str, bytes]:
    if not representative_ids:
        return {}
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output: dict[str, bytes] = {}
    plot_specs = (
        ("i_plus", "finite_i_plus", "Saved single-geometry i+ finite response"),
        ("mixed_central_derivative", "mixed_central", "Saved centered mixed-geometry derivative"),
        ("mixed_raw_four_corner_contrast", "mixed_raw_contrast", "Saved mixed-geometry four-corner contrast"),
    )
    for family_id in representative_ids:
        for response_label, file_stem, title in plot_specs:
            response_data = panels[family_id]["response_data"].get(response_label)
            if response_data is None:
                continue
            fig, axes = plt.subplots(3, 3, figsize=(12.0, 10.0), constrained_layout=True)
            axes_flat = axes.reshape(-1)
            for axis, (role, channel_name) in zip(axes_flat, _PLOT_COMPONENTS, strict=True):
                values = response_data.get(role)
                if values is None:
                    axis.set_visible(False)
                    continue
                if role == "module_peak_endpoint":
                    channel_index = 0
                else:
                    channel_names = {
                        "fluid_fields": ("u", "v", "p", "omega", "temperature"),
                        "interface": ("T_surface", "q_normal"),
                        "solid_temperature": ("temperature",),
                    }[role]
                    channel_index = channel_names.index(channel_name)
                valid = np.asarray(values["valid_mask"], dtype=bool)
                pred = np.asarray(values["prediction_delta"], dtype=np.float64)
                ref = np.asarray(values["reference_delta"], dtype=np.float64)
                if role != "module_peak_endpoint":
                    valid = valid[:, channel_index]
                    pred = pred[:, channel_index]
                    ref = ref[:, channel_index]
                finite_reference = valid & np.isfinite(ref)
                nonfinite_prediction_count = int((finite_reference & ~np.isfinite(pred)).sum())
                nonfinite_reference_count = int((valid & ~np.isfinite(ref)).sum())
                finite_pair = finite_reference & np.isfinite(pred)
                selected_pred = pred[finite_pair].reshape(-1)
                selected_ref = ref[finite_pair].reshape(-1)
                if selected_ref.size > 4000:
                    indices = np.linspace(0, selected_ref.size - 1, num=4000).astype(np.int64)
                    selected_pred = selected_pred[indices]
                    selected_ref = selected_ref[indices]
                axis.scatter(selected_ref, selected_pred, s=3, alpha=0.28, rasterized=True)
                if nonfinite_prediction_count or nonfinite_reference_count:
                    axis.text(
                        0.03, 0.97,
                        f"nonfinite predictions: {nonfinite_prediction_count}; references: {nonfinite_reference_count}",
                        transform=axis.transAxes, va="top", color="red",
                    )
                if selected_ref.size:
                    low = float(min(selected_ref.min(), selected_pred.min()))
                    high = float(max(selected_ref.max(), selected_pred.max()))
                    span = max(high - low, 1.0e-12)
                    axis.plot([low - 0.03 * span, high + 0.03 * span],
                              [low - 0.03 * span, high + 0.03 * span], "k--", linewidth=0.8)
                axis.set_title(f"{role}/{channel_name}")
                axis.set_xlabel("Reference response")
                axis.set_ylabel("Predicted response")
                axis.grid(alpha=0.18)
            fig.suptitle(f"{title} | family {family_id}")
            stream = io.BytesIO()
            fig.savefig(stream, format="png", dpi=130)
            plt.close(fig)
            output[f"{file_stem}_{family_id}.png"] = stream.getvalue()
    return output


def _write_artifacts(
    output_path: Path,
    summary: dict[str, Any],
    panel_bytes: bytes,
    plot_bytes: Mapping[str, bytes],
    *,
    max_output_bytes: int,
) -> dict[str, Any]:
    summary["representative_numerical_evidence"] = {
        "path": "response_delta_panels.npz",
        "selected_family_ids": summary["representative_family_ids"],
        "compressed_bytes": len(panel_bytes),
    }
    summary["plots"] = [
        {"path": name, "bytes": len(value)} for name, value in sorted(plot_bytes.items())
    ]
    summary_bytes = json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    planned_bytes = len(summary_bytes) + len(panel_bytes) + sum(len(value) for value in plot_bytes.values())
    if planned_bytes > max_output_bytes:
        raise OSError(f"Contextual physical-response artifacts need {planned_bytes} bytes; cap is {max_output_bytes}.")
    output_path.mkdir(parents=False, exist_ok=False)
    (output_path / "response_summary.json").write_bytes(summary_bytes)
    (output_path / "response_delta_panels.npz").write_bytes(panel_bytes)
    for name, payload in plot_bytes.items():
        (output_path / name).write_bytes(payload)
    actual_bytes = sum(path.stat().st_size for path in output_path.iterdir() if path.is_file())
    if actual_bytes != planned_bytes:
        raise OSError("Physical-response artifact bytes changed between preflight and save.")
    return {"output_path": str(output_path), "output_bytes": actual_bytes, "max_output_bytes": max_output_bytes}


def evaluate_responses(
    checkpoint: str | Path,
    device: str | torch.device,
    output: str | Path,
) -> dict[str, Any]:
    """Evaluate saved physical response stencils and write bounded numerical evidence.

    The evaluator uses fixed input-only Eulerian/surface query panels and every
    common-valid saved material point. Response-fit cohorts follow the saved
    provider's admitted TRAIN response families; remaining anchors are reported
    as reference-only or response-validation-only as appropriate.
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"Contextual physical-response output already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Physical-response output parent must exist: {output_path.parent}")

    model, provider, payload = load_contextual_checkpoint(checkpoint_path, device)
    identity = payload.get("experiment_identity", {})
    provider_identity = identity.get("provider_identity", {}) if isinstance(identity, Mapping) else {}
    recipe = provider_identity.get("recipe", {}) if isinstance(provider_identity, Mapping) else {}
    workflow = identity.get("workflow") if isinstance(identity, Mapping) else None
    reference_only_member_routing = workflow == "interaction_reused_member_routing_joint_training_v1"
    atlas_family_ids = tuple(TRAIN_GEOMETRY_FAMILIES) if reference_only_member_routing else _ALL_FAMILIES
    storage = recipe.get("storage", {}) if isinstance(recipe, Mapping) else {}
    minimum_headroom = int(storage.get("minimum_free_headroom_bytes", 0)) if isinstance(storage, Mapping) else 0
    free_before = _filesystem_free_bytes(output_path.parent)
    if free_before - _MAX_OUTPUT_BYTES < minimum_headroom:
        raise OSError("Physical-response output reservation would violate configured free-space headroom.")

    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("Contextual response checkpoint model has no parameters for native evaluation.")
    evaluation_device = torch.device(device)
    model.eval()
    core = getattr(model, "core", None)
    if core is not None and callable(getattr(core, "set_organization_clock", None)):
        epoch = int(payload.get("epoch", payload.get("current_epoch", 0)))
        core.set_organization_clock(epoch, 0)
    stencils, atlas_metadata, atlas_root = _load_stencils(provider, atlas_family_ids)
    first_roles = stencils[atlas_family_ids[0]].baseline.output.roles
    floors, response_fit_families, floor_source = _response_native_floors(model, provider, first_roles)
    cohort_plan = _cohort_plan(response_fit_families)

    family_results: dict[str, Any] = {}
    family_metrics: dict[str, dict[str, Any]] = {}
    panels: dict[str, Any] = {}
    with torch.no_grad():
        for family_id in atlas_family_ids:
            rows, metrics, panel_data = _evaluate_family(
                model,
                stencils[family_id],
                family_id,
                fit_families=response_fit_families,
                device=evaluation_device,
                dtype=parameter.dtype,
                floors=floors,
            )
            family_results[family_id] = rows
            family_metrics[family_id] = metrics
            panels[family_id] = panel_data

    representatives = _representative_families(family_results, response_fit_families, limit=3)
    panel_bytes = _panel_export_bytes(representatives, panels, family_results)
    plot_bytes = _render_response_plots(representatives, panels)
    epoch = int(payload.get("epoch", payload.get("current_epoch", 0)))
    arm = str(payload.get("arm", provider_identity.get("architecture", "")))
    invalid_components = [
        {
            "family_id": family_id,
            "response": label,
            "component": component,
            "status": metrics.get("status"),
            "nonfinite_prediction_count": metrics.get("nonfinite_prediction_count", 0),
            "nonfinite_reference_count": metrics.get("nonfinite_reference_count", 0),
        }
        for family_id, labels in family_metrics.items()
        for label, components in labels.items()
        for component, metrics in components.items()
        if metrics.get("status") != "valid"
    ]
    summary = {
        "schema": "thermal_contextual_frontier_physical_response_evaluation_v1",
        "status": "completed_with_invalid_component_metrics" if invalid_components else "completed",
        "metric_validity": {
            "invalid_component_count": len(invalid_components),
            "invalid_components": invalid_components,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": _checkpoint_selection(checkpoint_path, payload),
        "checkpoint_epoch": epoch,
        "arm": arm,
        "experiment_identity": {
            "workflow": identity.get("workflow") if isinstance(identity, Mapping) else None,
            "config_name": identity.get("config_name") if isinstance(identity, Mapping) else None,
            "subset_manifest_sha256": provider_identity.get("subset_manifest_sha256") if isinstance(provider_identity, Mapping) else None,
            "normalization_fit_scope": "selected TRAIN only",
        },
        "atlas_root": atlas_root,
        "response_cohort_contract": {
            "fit_train_family_ids": list(response_fit_families),
            "train_reference_only_family_ids": list(cohort_plan.get("train_reference_only", ())),
            "response_validation_only_family_ids": list(cohort_plan["response_validation_only"]),
            "atlas_family_ids_evaluated": list(atlas_family_ids),
            "response_fit_performed": bool(response_fit_families),
            "source_train_reference_use": (
                "saved physical references only; not response-training data"
                if reference_only_member_routing else "cohort follows saved provider response supervision"
            ),
            "cohort_labels": {cohort: list(family_ids) for cohort, family_ids in cohort_plan.items()},
            "original_source_partition": (
                "the four evaluated atlas anchors are recorded original source TRAIN families"
                if reference_only_member_routing else "all eight atlas anchors are recorded as original source TRAIN families"
            ),
            "response_validation_scope": (
                "not evaluated for this reference-only native checkpoint"
                if reference_only_member_routing else
                "held out from response fitting; not relabeled as original source DEV or TEST"
            ),
            "mixed_geometry": "reported separately as centered four-corner mixed derivatives, never pooled with single finite changes",
            "physical_scope_limit": (
                "Measurements are limited to saved source TRAIN atlas heat and geometry controls, common-valid receiver supports, "
                "and the recorded finite-change stencils; they do not establish physical causality or generalization beyond those supports."
            ),
        },
        "query_policy": {
            "fluid_fields": "input-selected 256 common-valid Eulerian points (or all if fewer)",
            "surface": "fixed 16 of the saved 64 native angles per original module",
            "material": "all saved module-local material points on common-valid original module rows",
            "endpoint": "per-state maximum over identical common-valid saved material points, then finite/central differencing",
        },
        "native_train_floors": floors,
        "native_floor_basis": {
            "source": floor_source,
            "admitted_response_fit_family_ids": list(response_fit_families),
            "validation_response_atlases_used_for_floor": False,
        },
        "response_summaries": _summarize_families(family_results, family_metrics, response_fit_families),
        "families": family_results,
        "atlas_families": {
            family_id: {
                "atlas_anchor_id": atlas_metadata[family_id].get("anchor_id"),
                "physical_family_id": stencils[family_id].physical_family_id,
                "source_partition": _value(stencils[family_id].split),
                "response_evaluation_cohort": _cohort_for_family(family_id, response_fit_families),
                "npz_path": str(Path(atlas_root) / f"train_{family_id}_responses.npz"),
            }
            for family_id in atlas_family_ids
        },
        "representative_family_ids": list(representatives),
        "optimizer_updates": 0,
        "solver_attempts": 0,
        "test_access": False,
        "storage_reservation": {
            "max_output_bytes": _MAX_OUTPUT_BYTES,
            "minimum_free_headroom_bytes": minimum_headroom,
            "free_bytes_before": free_before,
        },
    }
    safe_summary = _jsonable(summary)
    artifact_summary = _write_artifacts(output_path, safe_summary, panel_bytes, plot_bytes, max_output_bytes=_MAX_OUTPUT_BYTES)
    return {**safe_summary, **artifact_summary}


__all__ = ["evaluate_responses"]
