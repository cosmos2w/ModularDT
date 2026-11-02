"""Saved TRAIN-origin geometry/heat response packs for Thermal adaptation.

The atlas is read in place. Corners and development families never enter
forward-fitting scales or samples. Each pack retains original member IDs and
separate per-channel physical masks, including the signed native q proxy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ..interaction_evidence.response_atlas import load_response_atlas_stencil
from ..interaction_evidence.response_dataset import ResponseStencil
from ..interaction_evidence.types import EvidenceSplit


FIT_FAMILIES = ("0001", "0318", "0333", "0348")
SINGLE_EDITS = ("i_minus", "i_plus", "j_minus", "j_plus")
HEAT_EDITS = ("heat_transfer_plus", "heat_transfer_minus")
ROLE_CHANNELS = {
    "fluid_temperature": ("fluid_fields", "temperature"),
    "surface_temperature": ("interface", "T_surface"),
    "material_temperature": ("solid_temperature", "temperature"),
    "q": ("interface", "q_normal"),
}


@dataclass(frozen=True)
class FitResponsePacks:
    families: Mapping[str, ResponseStencil]
    response_scales: Mapping[str, Mapping[str, float]]
    null_flow_scales: np.ndarray


def record_structure(record) -> dict:
    """Target-free adapter inputs from this saved state's original IDs/controls."""
    design, context = record.design, record.context.values
    modules = design.active_modules
    ids = [int(module.module_id.rsplit(":module:", 1)[1]) for module in modules]
    if len(set(ids)) != len(ids):
        raise ValueError("Response design repeats an original member ID.")
    return {
        "module_centers": np.asarray([module.position_xy for module in modules], dtype=np.float32),
        "module_present": np.ones(len(modules), dtype=np.float32),
        "module_source_ids": np.asarray(ids, dtype=np.int64),
        "heat_powers": np.asarray([module.heating for module in modules], dtype=np.float32),
        "material_params": np.asarray([context[name] for name in
            ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius")], dtype=np.float32),
        "re": np.asarray([context["re"]], dtype=np.float32),
        "u_in": np.asarray([context["u_in"]], dtype=np.float32),
        "domain_length_x": np.asarray([context["domain_length_x"]], dtype=np.float32),
        "domain_length_y": np.asarray([context["domain_length_y"]], dtype=np.float32),
    }


def _pair_delta(base, trial):
    if (base.channel_names != trial.channel_names or base.query_ids != trial.query_ids
            or base.receiver_module_ids != trial.receiver_module_ids
            or not np.array_equal(base.query_features, trial.query_features)):
        raise ValueError("Response receivers must align by original ID and physical coordinates.")
    valid = base.valid_mask & trial.valid_mask
    delta = np.asarray(trial.values, float) - np.asarray(base.values, float)
    if not np.isfinite(delta[valid]).all():
        raise ValueError("Nonfinite saved response on a physically valid receiver.")
    return delta, valid


def _role_mean_square(delta, valid, receiver_ids):
    """One physical channel: equal active module mean for attached roles."""
    if receiver_ids is None:
        return float(np.mean(np.square(delta[valid]))) if valid.any() else None
    ids = np.asarray(receiver_ids)
    means = [float(np.mean(np.square(delta[valid & (ids == member)])))
             for member in dict.fromkeys(receiver_ids) if (valid & (ids == member)).any()]
    return float(np.mean(means)) if means else None


def load_fit_packs(atlas_directory: str | Path, absolute_scales: Mapping) -> FitResponsePacks:
    """Fit class/role RMS scales on four declared FIT families, without DEV."""
    scales = {name: float(absolute_scales[name]) for name in ROLE_CHANNELS}
    flow = np.asarray(absolute_scales["flow"], dtype=float)
    if any(not np.isfinite(v) or v <= 0 for v in scales.values()) or flow.shape != (4,) or not np.isfinite(flow).all() or np.any(flow <= 0):
        raise ValueError("Inherited absolute scales must be finite and positive.")
    families = {}
    for family in FIT_FAMILIES:
        stencil, metadata = load_response_atlas_stencil(Path(atlas_directory) / f"train_{family}_responses.npz")
        if stencil.anchor_id != family or stencil.split is not EvidenceSplit.TRAIN:
            raise ValueError("Forward response fitting admits only declared TRAIN-origin FIT families.")
        base = stencil.baseline.design
        base_ids = base.active_module_ids
        for label in (*SINGLE_EDITS, *HEAT_EDITS):
            trial = stencil.variants[label].design
            if trial.active_module_ids != base_ids:
                raise ValueError("Saved edit changed module identities.")
            pairs = tuple(zip(base.active_modules, trial.active_modules))
            if label in SINGLE_EDITS:
                changed = sum(a.position_xy != b.position_xy for a, b in pairs)
                if changed != 1 or any(a.heating != b.heating for a, b in pairs):
                    raise ValueError("Geometry fitting requires one saved move with fixed heat.")
            else:
                if any(a.position_xy != b.position_xy for a, b in pairs) or not np.isclose(sum(a.heating for a, _ in pairs), sum(b.heating for _, b in pairs), atol=1e-6, rtol=1e-6):
                    raise ValueError("Heat-null fitting requires unchanged geometry and balanced heat.")
        families[family] = stencil
    response_scales = {}
    for kind, labels in (("geometry", SINGLE_EDITS), ("heat", HEAT_EDITS)):
        response_scales[kind] = {}
        for role, (native_role, channel_name) in ROLE_CHANNELS.items():
            family_mse = []
            for stencil in families.values():
                baseline = stencil.baseline.output.roles[native_role]
                channel = baseline.channel_names.index(channel_name)
                mse = []
                for label in labels:
                    delta, valid = _pair_delta(baseline, stencil.variants[label].output.roles[native_role])
                    value = _role_mean_square(delta[:, channel], valid[:, channel], baseline.receiver_module_ids)
                    if value is not None:
                        mse.append(value)
                if mse:
                    family_mse.append(float(np.mean(mse)))
            if not family_mse:
                raise ValueError(f"No FIT physical response observations for {kind}/{role}.")
            response_scales[kind][role] = max(float(np.sqrt(np.mean(family_mse))), .01 * scales[role])
    return FitResponsePacks(families, response_scales, .05 * flow)


def sample_fit_pack(store: FitResponsePacks, family_id: str, epoch: int, *, seed: int = 0) -> dict:
    """One baseline/single/heat pack with an independent deterministic draw."""
    if family_id not in FIT_FAMILIES or epoch < 1:
        raise ValueError("A response pack requires a declared FIT family and positive epoch.")
    stencil = store.families[family_id]
    geometry_label = SINGLE_EDITS[(epoch - 1) % len(SINGLE_EDITS)]
    heat_label = HEAT_EDITS[(epoch - 1) % len(HEAT_EDITS)]
    records = {"baseline": stencil.baseline, "geometry": stencil.variants[geometry_label],
               "heat": stencil.variants[heat_label]}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(family_id), 15664]))
    fluid_roles = [record.output.roles["fluid_fields"] for record in records.values()]
    common = np.logical_and.reduce([role.valid_mask.all(axis=1) for role in fluid_roles])
    eligible = np.flatnonzero(common)
    if len(eligible) < 512:
        raise ValueError("Saved pack lacks 512 shared valid Eulerian fluid receivers.")
    fluid_indices = np.sort(rng.choice(eligible, 512, replace=False))
    solid = stencil.baseline.output.roles["solid_temperature"]
    attached = np.asarray(solid.receiver_module_ids)
    module_rows = [np.flatnonzero(attached == member) for member in stencil.baseline.design.active_module_ids]
    canonical_local = solid.query_features[module_rows[0]]
    if any(not np.array_equal(solid.query_features[rows], canonical_local) for rows in module_rows):
        raise ValueError("Saved material receivers do not share an ID-aligned local coordinate catalogue.")
    material_valid = np.logical_and.reduce([solid.valid_mask[rows].all(axis=1) for rows in module_rows])
    eligible_material = np.flatnonzero(material_valid)
    if len(eligible_material) < 32:
        raise ValueError("Saved pack lacks 32 common module-attached material receivers.")
    local_indices = np.sort(rng.choice(eligible_material, 32, replace=False))
    material_indices = np.concatenate([rows[local_indices] for rows in module_rows])
    interface = stencil.baseline.output.roles["interface"]
    if interface.receiver_module_ids is None:
        raise ValueError("Saved interface must identify original members.")
    for member in stencil.baseline.design.active_module_ids:
        if interface.receiver_module_ids.count(member) != 64:
            raise ValueError("Response packs require complete 64-angle interface receivers.")
    indices = {"fluid_fields": fluid_indices, "solid_temperature": material_indices,
               "interface": np.arange(len(interface.values), dtype=np.int64)}
    roles = {}
    for scene, record in records.items():
        roles[scene] = {}
        for role_name, selected in indices.items():
            role = record.output.roles[role_name]
            _pair_delta(stencil.baseline.output.roles[role_name], role)
            roles[scene][role_name] = {
                "values": np.asarray(role.values)[selected],
                "valid_mask": np.asarray(role.valid_mask)[selected],
                "query_features": np.asarray(role.query_features)[selected],
                "receiver_module_ids": None if role.receiver_module_ids is None else tuple(role.receiver_module_ids[i] for i in selected),
                "channel_names": role.channel_names,
            }
    deltas = {}
    for kind in ("geometry", "heat"):
        deltas[kind] = {}
        for role_name in indices:
            base, trial = roles["baseline"][role_name], roles[kind][role_name]
            deltas[kind][role_name] = {"values": trial["values"] - base["values"],
                                      "valid_mask": trial["valid_mask"] & base["valid_mask"]}
    return {"family_id": family_id, "epoch": epoch, "geometry_label": geometry_label,
            "heat_label": heat_label, **records, "fluid_indices": fluid_indices,
            "material_indices": material_indices, "interface_indices": indices["interface"],
            "material_local": canonical_local[local_indices],
            "roles": roles, "deltas": deltas,
            "structures": {name: record_structure(record) for name, record in records.items()}}
