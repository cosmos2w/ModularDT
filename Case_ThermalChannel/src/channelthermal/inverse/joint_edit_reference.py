"""Native physical-reference helpers for bounded Thermal joint edits.

Eligibility is checked against original dataset membership before the atlas is read.
These helpers do not start solves or manage experiment budgets.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from channelthermal.interaction_evidence.quantities import smooth_peak_temperature
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.interaction_evidence.types import DesignState, SolveStatus


def load_eligible_atlas(dataset_path: str | Path, case_id: str, train_ids: Sequence[str], atlas_path: str | Path):
    """Require original TRAIN membership and HDF5 source split before atlas loading."""
    case_id = str(case_id)
    ids = tuple(str(item) for item in train_ids)
    if len(ids) != len(set(ids)) or case_id not in ids or case_id == "0273":
        raise ValueError(f"Case {case_id!r} is not an eligible unique original TRAIN member.")
    with h5py.File(dataset_path, "r") as handle:
        group = handle["cases"][case_id]
        split = group.attrs.get("split")
        if isinstance(split, bytes):
            split = split.decode("utf-8")
        if str(split).lower() != "train":
            raise ValueError(f"Case {case_id!r} has original source split {split!r}, not train.")
    stencil, metadata = load_response_atlas_stencil(atlas_path)
    if stencil.baseline.design.anchor_id != case_id or str(metadata.get("split", "")).lower() != "train":
        raise ValueError("Atlas family does not match the checked original TRAIN source.")
    return stencil, metadata


def edited_design(design: DesignState, centers: np.ndarray) -> DesignState:
    """Change at most two present original slots and preserve every other field."""
    xy = np.asarray(centers, dtype=np.float64)
    if xy.shape != (len(design.modules), 2) or not np.isfinite(xy).all():
        raise ValueError("Centers must contain one finite x/y pair per original slot.")
    changed = [i for i, module in enumerate(design.modules) if not np.array_equal(xy[i], module.position_xy)]
    if len(changed) > 2 or any(not design.modules[i].active for i in changed):
        raise ValueError("At most two present modules may move.")
    return replace(design, modules=tuple(replace(module, position_xy=tuple(xy[i])) for i, module in enumerate(design.modules)))


def native_reference_quantities(record, tau: float, *, fluid_stencil_valid=None, material_stencil_valid=None) -> dict[str, Any]:
    """Reduce native reference on the evaluator's geometry and receiver support.

    Optional stencil masks are from the fresh predictor preparation for the same
    design. Source and stencil masks are intersected and mismatch counts retained.
    """
    if record.status != SolveStatus.CONVERGED or record.output is None:
        raise ValueError("Native reference requires a converged physical record.")
    roles = record.output.roles
    solid = roles["solid_temperature"]
    if len(solid.channel_names) != 1:
        raise ValueError("Expected one material-temperature channel.")
    source_solid = solid.valid_mask[:, 0]
    solid_stencil = np.ones_like(source_solid) if material_stencil_valid is None else np.asarray(material_stencil_valid, dtype=bool).reshape(-1)
    if solid_stencil.shape != source_solid.shape:
        raise ValueError("Material stencil does not align with native receiver IDs.")
    selected = source_solid & solid_stencil
    if not selected.any() or not np.isfinite(tau) or tau <= 0:
        raise ValueError("Valid material samples and positive temperature scale required.")
    values = solid.values[selected, 0]
    fluid = roles["fluid_fields"]
    pindex = fluid.channel_names.index("p")
    xy = fluid.query_features[:, :2]
    centers = np.asarray([m.position_xy for m in record.design.active_modules])
    radius = float(record.context.values["module_radius"])
    geometry = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1).min(axis=1) >= radius ** 2
    source_fluid = fluid.valid_mask[:, pindex]
    stencil = np.ones_like(source_fluid) if fluid_stencil_valid is None else np.asarray(fluid_stencil_valid, dtype=bool).reshape(-1)
    if stencil.shape != source_fluid.shape:
        raise ValueError("Fluid stencil does not align with native receiver IDs.")
    mask = geometry & source_fluid & stencil
    length = float(record.context.values["domain_length_x"])
    left, right = mask & (xy[:, 0] <= 0.08 * length), mask & (xy[:, 0] >= 0.92 * length)
    if not left.any() or not right.any():
        raise ValueError("Native pressure bands require valid fluid samples.")
    p = fluid.values[:, pindex]
    return {"objective": smooth_peak_temperature(values, beta=1.0 / tau), "sampled_max": float(values.max()),
            "pressure_drop": float(p[left].mean() - p[right].mean()), "material_count": int(selected.sum()),
            "inlet_count": int(left.sum()), "outlet_count": int(right.sum()),
            "fluid_geometry_source_mismatch": int(np.count_nonzero(geometry != source_fluid)),
            "fluid_stencil_source_mismatch": int(np.count_nonzero(stencil != source_fluid)),
            "material_stencil_source_mismatch": int(np.count_nonzero(solid_stencil != source_solid)),
            "temperature_units": solid.channel_units[0], "pressure_units": fluid.channel_units[pindex]}


def aligned_field_metrics(role, prediction: np.ndarray, *, query_ids: Sequence[str], query_features: np.ndarray,
                          prediction_valid: np.ndarray | None = None) -> dict[str, dict[str, float | int | None]]:
    """Score same-ID, same-coordinate receiver arrays only, using their common mask."""
    pred = np.asarray(prediction, dtype=np.float64).reshape(role.values.shape)
    if tuple(query_ids) != role.query_ids or not np.array_equal(np.asarray(query_features), role.query_features):
        raise ValueError("Prediction and reference receiver identities/coordinates differ.")
    mask = role.valid_mask.copy()
    if prediction_valid is not None:
        valid = np.asarray(prediction_valid, dtype=bool)
        if valid.ndim == 1:
            valid = np.broadcast_to(valid[:, None], pred.shape)
        if valid.shape != pred.shape:
            raise ValueError("Prediction mask does not align with native values.")
        mask &= valid
    output = {}
    for i, name in enumerate(role.channel_names):
        use = mask[:, i]
        if not np.isfinite(pred[use, i]).all():
            raise ValueError("Prediction contains a nonfinite valid value.")
        diff = pred[use, i] - role.values[use, i]
        norm = np.linalg.norm(role.values[use, i])
        output[name] = {"count": int(use.sum()), "rmse": float(np.sqrt(np.mean(diff ** 2))) if use.any() else None,
                        "relative_l2": float(np.linalg.norm(diff) / norm) if norm > 0 else None,
                        "units": role.channel_units[i]}
    return output


def compatible_reference(record, expected_design, expected_context, stored_config: Mapping[str, Any], expected_config: Mapping[str, Any], *, expected_roles: Mapping[str, Any]) -> bool:
    """Conservative exact reuse comparison; transient output/runtime fields are excluded."""
    if record.status != SolveStatus.CONVERGED or record.output is None or record.design != expected_design:
        return False
    if dict(record.context.values) != dict(expected_context.values):
        return False
    def scientific(config):
        return {k: v for k, v in config.items() if k not in {"save", "execution", "runtime"}}
    if scientific(stored_config) != scientific(expected_config):
        return False
    if set(record.output.roles) != set(expected_roles):
        return False
    for name, expected in expected_roles.items():
        actual = record.output.roles[name]
        if (actual.query_ids != expected.query_ids or actual.receiver_module_ids != expected.receiver_module_ids
            or actual.channel_names != expected.channel_names or actual.channel_units != expected.channel_units
            or actual.coordinate_kind != expected.coordinate_kind
            or not np.array_equal(actual.query_features, expected.query_features)
            or not np.array_equal(actual.valid_mask, expected.valid_mask)
            or not np.array_equal(actual.quadrature_weights, expected.quadrature_weights)):
            return False
    return True
