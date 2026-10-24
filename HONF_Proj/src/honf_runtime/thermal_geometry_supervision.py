"""Bounded, train-only geometry-response supervision for the Thermal adapter.

The loader consumes only the four declared original-TRAIN families and exposes
only their single-module ``i-/i+/j-/j+`` finite changes to the training cycle.
Each loss call uses the maintained adapter ``prepare_native`` / ``apply_native``
path for the baseline and moved geometry; this module owns no optimizer and
never runs a reference solver.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.interaction_evidence.response_dataset import ResponseBlock, ResponseStencil
from channelthermal.interaction_evidence.types import EvidenceSource, EvidenceSplit


TRAIN_GEOMETRY_FAMILIES = ("0001", "0318", "0333", "0348")
DEVELOPMENT_GEOMETRY_FAMILIES = ("0304", "0320", "0335", "0350")
SINGLE_GEOMETRY_LABELS = ("i_minus", "i_plus", "j_minus", "j_plus")
MIXED_GEOMETRY_LABELS = ("mm", "mp", "pm", "pp")
GEOMETRY_RESPONSE_PERIOD = 4
NATIVE_ATLAS_THETA = 64
SUPPORTED_NATIVE_THETA = (16, 64)
_ROLE_KEYS = ("fluid_fields", "interface", "solid_temperature")
_GEOMETRY_PAIR_COUNT = len(TRAIN_GEOMETRY_FAMILIES) * len(SINGLE_GEOMETRY_LABELS)
_MIN_SCALE = 1.0e-8


def geometry_response_due(primary_update: int, *, every: int = GEOMETRY_RESPONSE_PERIOD) -> bool:
    """Whether a one-based primary optimizer update receives a geometry pair."""

    if type(primary_update) is not int or primary_update < 1:
        raise ValueError("primary_update must be a positive one-based optimizer update.")
    if type(every) is not int or every < 1:
        raise ValueError("every must be a positive integer.")
    return primary_update % every == 0


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_physical_family(anchor_id: str) -> str:
    if anchor_id == "0001":
        return "duplicate_family:0001+0273"
    return f"stored_family:{anchor_id}"


def validate_training_family_metadata(metadata: Mapping[str, Any], expected_anchor: str) -> None:
    """Reject non-fit families and split/source mismatches before loading arrays."""

    if expected_anchor not in TRAIN_GEOMETRY_FAMILIES:
        raise ValueError(f"Geometry gradients are restricted to {TRAIN_GEOMETRY_FAMILIES}.")
    expected = {
        "anchor_id": expected_anchor,
        "family_id": _expected_physical_family(expected_anchor),
        "split": "train",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"TRAIN geometry atlas {expected_anchor} has invalid {key}: expected {value!r}.")


def validate_module_major_receivers(
    module_ids: Sequence[str], receiver_module_ids: Sequence[str] | None, rows_per_module: int,
) -> None:
    """Check that local native receivers retain their original module-major rows."""

    ids = tuple(map(str, module_ids))
    if not ids or len(ids) != len(set(ids)) or type(rows_per_module) is not int or rows_per_module < 1:
        raise ValueError("Native module identities and per-module receiver counts must be nonempty and unique.")
    if receiver_module_ids is None:
        raise ValueError("Module-local native receivers require explicit receiver module IDs.")
    observed = tuple(map(str, receiver_module_ids))
    expected = tuple(module_id for module_id in ids for _ in range(rows_per_module))
    if observed != expected:
        raise ValueError("Native receiver module IDs no longer match the original module-major local axis.")


@dataclass(frozen=True)
class GeometryResponsePair:
    """One baseline-to-single-geometry finite-change example."""

    family_id: str
    physical_family_id: str
    label: str
    module_ids: tuple[str, ...]
    baseline: Any
    trial: Any
    roles: Mapping[str, ResponseBlock]
    identity: Mapping[str, Any]


def _source_column(module_id: str) -> int:
    parts = str(module_id).split(":")
    if len(parts) == 3 and parts[1] == "module":
        raw = parts[2]
    elif len(parts) == 2 and parts[0] == "M":
        raw = parts[1]
    else:
        raise ValueError(f"Module ID {module_id!r} does not preserve an original integer source column.")
    try:
        result = int(raw)
    except ValueError as error:
        raise ValueError(f"Module ID {module_id!r} has a non-integer original source column.") from error
    if result < 0:
        raise ValueError("Original module source columns must be nonnegative.")
    return result


def _design_payload(record: Any) -> dict[str, Any]:
    modules = record.design.active_modules
    return {
        "module_ids": list(record.design.active_module_ids),
        "centers_xy_float32": np.asarray([module.position_xy for module in modules], dtype=np.float32).tolist(),
        "heating_float32": np.asarray([module.heating for module in modules], dtype=np.float32).tolist(),
        "context": {str(key): float(value) for key, value in record.context.values.items()},
    }


def _validate_pair_geometry(stencil: ResponseStencil, label: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    baseline = stencil.baseline
    trial = stencil.variants[label]
    if baseline.design.active_module_ids != trial.design.active_module_ids:
        raise ValueError(f"Geometry response {stencil.anchor_id}/{label} changes original module identities.")
    before = np.asarray([module.position_xy for module in baseline.design.active_modules], dtype=np.float32)
    after = np.asarray([module.position_xy for module in trial.design.active_modules], dtype=np.float32)
    base_heat = np.asarray([module.heating for module in baseline.design.active_modules], dtype=np.float32)
    trial_heat = np.asarray([module.heating for module in trial.design.active_modules], dtype=np.float32)
    panel = metadata.get("position_panel", {})
    moved_ids = tuple(map(str, panel.get("module_ids", ())))
    slots = tuple(int(value) for value in panel.get("module_slot_ids", ()))
    if len(slots) != 2 or len(moved_ids) != 2 or len(set(slots)) != 2:
        raise ValueError(f"Geometry atlas {stencil.anchor_id} lacks the declared two-module position identity.")
    if tuple(baseline.design.active_module_ids[index] for index in slots) != moved_ids:
        raise ValueError(f"Geometry atlas {stencil.anchor_id} position module IDs do not match original slots.")
    axis, sign, slot = {
        "i_minus": (0, -1, slots[0]), "i_plus": (0, 1, slots[0]),
        "j_minus": (1, -1, slots[1]), "j_plus": (1, 1, slots[1]),
    }[label]
    change = after - before
    changed_modules = np.flatnonzero(np.any(change != 0.0, axis=1)).tolist()
    if changed_modules != [slot] or change[slot, axis] * sign <= 0.0:
        raise ValueError(f"Geometry label {stencil.anchor_id}/{label} is not its declared single-module axis move.")
    other_axes = [index for index in range(2) if index != axis]
    if np.any(change[:, other_axes] != 0.0) or not np.array_equal(base_heat, trial_heat):
        raise ValueError(f"Geometry label {stencil.anchor_id}/{label} changes heat or a second geometry coordinate.")
    return {
        "moved_module_id": baseline.design.active_module_ids[slot],
        "moved_original_source_column": _source_column(baseline.design.active_module_ids[slot]),
        "axis": "x" if axis == 0 else "y",
        "signed_step": float(change[slot, axis]),
        "baseline_input_geometry_sha256": _json_digest(_design_payload(baseline)),
        "trial_input_geometry_sha256": _json_digest(_design_payload(trial)),
        "geometry_pair_sha256": _json_digest({"baseline": _design_payload(baseline), "trial": _design_payload(trial)}),
    }


def _validate_role_layout(pair: ResponseStencil, module_ids: tuple[str, ...]) -> None:
    baseline = pair.baseline.output
    if baseline is None:
        raise ValueError("Geometry supervision requires stored native receiver outputs.")
    fluid = baseline.roles["fluid_fields"]
    interface = baseline.roles["interface"]
    solid = baseline.roles["solid_temperature"]
    if fluid.query_features.ndim != 2 or fluid.query_features.shape[1] < 2:
        raise ValueError("Fluid native queries must preserve their two physical coordinates.")
    if fluid.receiver_module_ids is not None:
        raise ValueError("Fluid receiver rows must remain Eulerian and unassigned to module slots.")
    if interface.values.shape[1] != 2 or interface.values.shape[0] % len(module_ids):
        raise ValueError("Interface role arrays must retain module-major [module, angle, channel] axes.")
    interface_per_module = interface.values.shape[0] // len(module_ids)
    validate_module_major_receivers(module_ids, interface.receiver_module_ids, interface_per_module)
    if interface.query_features.shape[1] < 1:
        raise ValueError("Interface native receivers must preserve their local angle coordinate.")
    theta = interface.query_features[:, 0].reshape(len(module_ids), interface_per_module)
    expected_theta = np.arange(interface_per_module, dtype=np.float64) * (2.0 * math.pi / interface_per_module)
    if not np.allclose(theta, expected_theta[None, :], atol=3.0e-7, rtol=0.0):
        raise ValueError("Interface angle rows no longer match the maintained native module-local order.")
    if solid.values.shape[0] % len(module_ids) or solid.query_features.shape[1] != 2:
        raise ValueError("Material role arrays must retain module-major local coordinate axes.")
    solid_per_module = solid.values.shape[0] // len(module_ids)
    validate_module_major_receivers(module_ids, solid.receiver_module_ids, solid_per_module)
    local_xy = solid.query_features.reshape(len(module_ids), solid_per_module, 2)
    if not np.array_equal(local_xy, np.broadcast_to(local_xy[:1], local_xy.shape)):
        raise ValueError("Material receiver local coordinates differ between original module slots.")
    for role in _ROLE_KEYS:
        pair.common_mask(role)


def _role_query_digest(block: ResponseBlock) -> str:
    return _json_digest({
        "role": block.role,
        "query_features_sha256": _array_digest(block.query_features),
        "query_ids": list(block.query_ids),
        "receiver_module_ids": None if block.receiver_module_ids is None else list(block.receiver_module_ids),
    })


def _make_pair(
    stencil: ResponseStencil,
    label: str,
    metadata: Mapping[str, Any],
    embedded: Mapping[str, Any],
    atlas_path: Path,
    metadata_path: Path,
    atlas_sha256: str,
    metadata_sha256: str,
) -> GeometryResponsePair:
    if label not in SINGLE_GEOMETRY_LABELS:
        raise ValueError("Training geometry supervision accepts only i-/j-axis single-difference labels.")
    pair_stencil = ResponseStencil(stencil.baseline, {label: stencil.variants[label]})
    module_ids = tuple(pair_stencil.baseline.design.active_module_ids)
    _validate_role_layout(pair_stencil, module_ids)
    geometry = _validate_pair_geometry(stencil, label, embedded)
    roles = {role: pair_stencil.finite_change(label, role) for role in _ROLE_KEYS}
    for role, block in roles.items():
        if not block.valid_mask.any():
            raise ValueError(f"Geometry response {stencil.anchor_id}/{label}/{role} has no common receiver support.")
    identity = {
        "family_id": stencil.anchor_id,
        "physical_family_id": stencil.physical_family_id,
        "label": label,
        "module_ids": list(module_ids),
        "module_count": len(module_ids),
        "moved_module_id": geometry["moved_module_id"],
        "moved_original_source_column": geometry["moved_original_source_column"],
        "axis": geometry["axis"],
        "signed_step": geometry["signed_step"],
        "baseline_input_geometry_sha256": geometry["baseline_input_geometry_sha256"],
        "trial_input_geometry_sha256": geometry["trial_input_geometry_sha256"],
        "geometry_pair_sha256": geometry["geometry_pair_sha256"],
        "atlas_path": str(atlas_path),
        "atlas_sha256": atlas_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha256,
        "receiver_query_sha256": {role: _role_query_digest(block) for role, block in roles.items()},
        "reference_delta_sha256": {role: _array_digest(block.delta) for role, block in roles.items()},
        "common_mask_sha256": {role: _array_digest(block.valid_mask) for role, block in roles.items()},
        "common_receiver_counts": {role: int(block.valid_mask.any(axis=1).sum()) for role, block in roles.items()},
    }
    return GeometryResponsePair(
        family_id=stencil.anchor_id,
        physical_family_id=stencil.physical_family_id,
        label=label,
        module_ids=module_ids,
        baseline=pair_stencil.baseline,
        trial=pair_stencil.variants[label],
        roles=roles,
        identity=identity,
    )


def _component_names(roles: Mapping[str, ResponseBlock]) -> tuple[str, ...]:
    names = []
    for role in _ROLE_KEYS:
        names.extend(f"{role}/{name}" for name in roles[role].channel_names)
    return tuple(names)


def _component_scales(pairs: Sequence[GeometryResponsePair], family_ids=TRAIN_GEOMETRY_FAMILIES) -> tuple[dict[str, float], dict[str, Any]]:
    expected = Counter({family: len(SINGLE_GEOMETRY_LABELS) for family in family_ids})
    observed = Counter(pair.family_id for pair in pairs)
    if observed != expected or len(pairs) != len(family_ids) * len(SINGLE_GEOMETRY_LABELS):
        raise ValueError("Geometry scales require exactly four single-difference labels per declared TRAIN family.")
    expected_pairs = {(family, label) for family in family_ids for label in SINGLE_GEOMETRY_LABELS}
    if {(pair.family_id, pair.label) for pair in pairs} != expected_pairs:
        raise ValueError("Geometry scales cannot include substituted, repeated, mixed, or development labels.")
    first_roles = pairs[0].roles
    names = _component_names(first_roles)
    for pair in pairs[1:]:
        if _component_names(pair.roles) != names:
            raise ValueError("TRAIN geometry families must share one output role/channel schema.")
    pair_squared_means: dict[str, list[float]] = {name: [] for name in names}
    pair_counts: dict[str, list[int]] = {name: [] for name in names}
    for pair in pairs:
        for role in _ROLE_KEYS:
            block = pair.roles[role]
            for channel, channel_name in enumerate(block.channel_names):
                name = f"{role}/{channel_name}"
                mask = block.valid_mask[:, channel] & np.isfinite(block.delta[:, channel])
                if not mask.any():
                    raise ValueError(f"TRAIN geometry response has no valid {name} examples for {pair.family_id}/{pair.label}.")
                selected = np.asarray(block.delta[mask, channel], dtype=np.float64)
                pair_squared_means[name].append(float(np.mean(np.square(selected))))
                pair_counts[name].append(int(mask.sum()))
    scales = {}
    details = {}
    for name in names:
        rms = math.sqrt(float(np.mean(pair_squared_means[name])))
        if not math.isfinite(rms):
            raise ValueError(f"TRAIN-derived geometry response scale for {name} is non-finite.")
        scale = max(rms, _MIN_SCALE)
        scales[name] = scale
        details[name] = {
            "scale": scale,
            "equal_family_direction_rms": rms,
            "floor": _MIN_SCALE,
            "pair_count": len(pair_squared_means[name]),
            "valid_scalar_counts_by_pair": pair_counts[name],
            "basis_families": list(family_ids),
            "basis_labels": list(SINGLE_GEOMETRY_LABELS),
        }
    return scales, details


def _normalize_atlas_source(source: Any) -> Any:
    if not isinstance(source, (str, Path, Sequence, Mapping)):
        source = getattr(source, "atlas_directory", None)
        if source is None:
            raise TypeError("Geometry atlas input must be an atlas path, ordered family paths, or provider.atlas_directory.")
        if not isinstance(source, (str, Path, Sequence, Mapping)):
            raise TypeError("provider.atlas_directory must be an atlas path, ordered family paths, or family mapping.")
    return source


def _resolve_atlas_paths(source: Any, family_ids=TRAIN_GEOMETRY_FAMILIES) -> tuple[Path, ...]:
    source = _normalize_atlas_source(source)
    if isinstance(source, Mapping):
        if tuple(source.keys()) != family_ids:
            raise ValueError(f"Geometry atlas mappings must use exactly the ordered keys {family_ids}.")
        paths = tuple(Path(source[family]) for family in family_ids)
    elif isinstance(source, (str, Path)):
        root = Path(source).expanduser()
        family_root = root / "families" if (root / "families").is_dir() else root
        paths = tuple(family_root / f"train_{family}_responses.npz" for family in family_ids)
    else:
        paths = tuple(Path(path).expanduser() for path in source)
        if len(paths) != len(family_ids):
            raise ValueError("Geometry supervision requires exactly four existing TRAIN atlas NPZ files.")
    resolved = tuple(path.resolve(strict=True) for path in paths)
    if len(set(resolved)) != len(family_ids):
        raise ValueError("Each TRAIN geometry family must have a distinct atlas file.")
    return resolved


def _read_embedded_family_metadata(atlas_path: Path) -> dict[str, Any]:
    with np.load(atlas_path, allow_pickle=False) as archive:
        raw = np.asarray(archive["family_metadata_json"]).item()
    value = json.loads(str(raw))
    if not isinstance(value, dict):
        raise ValueError(f"Embedded physical family metadata must be a JSON object: {atlas_path}.")
    return value


def _build_pair_index(source: str | Path | Sequence[str | Path] | Mapping[str, str | Path], family_ids=TRAIN_GEOMETRY_FAMILIES):
    if not family_ids or len(set(family_ids)) != len(family_ids) or not set(family_ids).issubset(TRAIN_GEOMETRY_FAMILIES):
        raise ValueError("Response gradients require a nonempty subset of original TRAIN anchors.")
    paths = _resolve_atlas_paths(source, family_ids)
    pairs: list[GeometryResponsePair] = []
    family_rows = []
    physical_families = set()
    for expected_anchor, atlas_path in zip(family_ids, paths, strict=True):
        metadata_path = atlas_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Geometry atlas metadata companion is missing: {metadata_path}.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Geometry atlas metadata must be a JSON object: {metadata_path}.")
        validate_training_family_metadata(metadata, expected_anchor)
        embedded = _read_embedded_family_metadata(atlas_path)
        if (embedded.get("anchor_id") != expected_anchor
                or embedded.get("physical_family_id") != _expected_physical_family(expected_anchor)
                or embedded.get("source_dataset_split") != "train"
                or embedded.get("split") != "train"):
            raise ValueError(f"Embedded original-source partition or family identity differs for {expected_anchor}.")
        atlas_sha256 = _file_digest(atlas_path)
        metadata_sha256 = _file_digest(metadata_path)
        stencil, loaded_metadata = load_response_atlas_stencil(atlas_path)
        if loaded_metadata != metadata:
            raise ValueError(f"Geometry atlas metadata changed while family {expected_anchor} was loading.")
        if (stencil.anchor_id != expected_anchor
                or stencil.physical_family_id != _expected_physical_family(expected_anchor)
                or stencil.split is not EvidenceSplit.TRAIN
                or stencil.source is not EvidenceSource.REFERENCE_SOLVER):
            raise ValueError(f"Geometry atlas typed source identity differs for TRAIN family {expected_anchor}.")
        if stencil.physical_family_id in physical_families:
            raise ValueError("Geometry training families must have distinct physical-family identities.")
        physical_families.add(stencil.physical_family_id)
        labels = set(stencil.variants)
        if not set(SINGLE_GEOMETRY_LABELS).issubset(labels):
            raise ValueError(f"Geometry atlas {expected_anchor} lacks all four single-difference labels.")
        if not set(MIXED_GEOMETRY_LABELS).issubset(labels):
            raise ValueError(f"Geometry atlas {expected_anchor} lacks the declared mixed validation labels.")
        if tuple(stencil.variants[label].design.split.value for label in SINGLE_GEOMETRY_LABELS) != ("train",) * 4:
            raise ValueError(f"Geometry atlas {expected_anchor} contains a non-TRAIN single-difference record.")
        records_meta = metadata.get("records", {})
        for label in ("baseline", *SINGLE_GEOMETRY_LABELS, *MIXED_GEOMETRY_LABELS):
            row = records_meta.get(label, {})
            if row.get("status") != "converged" or row.get("source") != "reference_solver":
                raise ValueError(f"Stored physical-reference validity is incomplete for {expected_anchor}/{label}.")
        for label in SINGLE_GEOMETRY_LABELS:
            pairs.append(_make_pair(
                stencil, label, metadata, embedded, atlas_path, metadata_path, atlas_sha256, metadata_sha256,
            ))
        thermal = embedded.get("case_config", {}).get("thermal", {})
        if float(thermal.get("t_in", math.nan)) != 0.0 or float(thermal.get("t_wall", math.nan)) != 0.0:
            raise ValueError(f"Geometry response atlas {expected_anchor} must preserve zero boundary/background heat.")
        family_rows.append({
            "family_id": expected_anchor,
            "physical_family_id": stencil.physical_family_id,
            "split": "train",
            "module_ids": list(stencil.baseline.design.active_module_ids),
            "module_count": len(stencil.baseline.design.active_module_ids),
            "atlas_path": str(atlas_path),
            "atlas_sha256": atlas_sha256,
            "metadata_path": str(metadata_path),
            "metadata_sha256": metadata_sha256,
            "trained_labels": list(SINGLE_GEOMETRY_LABELS),
            "mixed_labels_validation_only": list(MIXED_GEOMETRY_LABELS),
            "duplicate_case_ids": list(embedded.get("duplicate_case_ids", ())),
        })
    return tuple(pairs), tuple(family_rows)


def _stable_seed(seed: int, index: int, pair: GeometryResponsePair, role: str) -> int:
    payload = f"{seed}:{index}:{pair.family_id}:{pair.label}:{role}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _sample_valid_rows(mask: np.ndarray, count: int, *, seed: int, all_channels: bool) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Receiver panel masks must be [query, channel].")
    eligible = mask.all(axis=1) if all_channels else mask.any(axis=1)
    indices = np.flatnonzero(eligible)
    if not indices.size:
        raise ValueError("The selected geometry query panel has no common valid receiver rows.")
    sample_count = min(int(count), int(indices.size))
    if sample_count < 1:
        raise ValueError("Geometry receiver query counts must be positive.")
    if sample_count == indices.size:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=sample_count, replace=False))


def _native_inputs(record: Any, module_ids: tuple[str, ...], device: torch.device, dtype: torch.dtype):
    modules = record.design.active_modules
    if tuple(record.design.active_module_ids) != module_ids:
        raise ValueError("Native adapter inputs changed the original module source order.")
    centers = np.asarray([module.position_xy for module in modules], dtype=np.float32)
    heating = np.asarray([module.heating for module in modules], dtype=np.float32)
    columns = [_source_column(module_id) for module_id in module_ids]
    if len(columns) != len(set(columns)):
        raise ValueError("Original module source columns must be unique in the native structure.")
    context = {str(key): float(value) for key, value in record.context.values.items()}
    required = ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius",
                "re", "u_in", "domain_length_x", "domain_length_y")
    absent = [key for key in required if key not in context]
    if absent:
        raise ValueError(f"Saved geometry operating context is missing native fields {absent}.")

    def tensor(value: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value, dtype=np.float32), device=device, dtype=dtype)

    structure = {
        "module_centers": tensor(centers[None]),
        "module_present": tensor(np.ones((1, len(module_ids)), dtype=np.float32)),
        "module_source_ids": torch.as_tensor(columns, device=device, dtype=torch.long)[None],
        "material_params": tensor([[context[name] for name in required[:6]]]),
        "re": tensor([[context["re"]]]),
        "u_in": tensor([[context["u_in"]]]),
        "domain_length_x": tensor([[context["domain_length_x"]]]),
        "domain_length_y": tensor([[context["domain_length_y"]]]),
        "t_in": tensor([[0.0]]),
        "t_wall": tensor([[0.0]]),
    }
    heat = tensor(heating[None])
    return structure, heat


def _component_sum(
    prediction: torch.Tensor,
    target: np.ndarray,
    valid_mask: np.ndarray,
    scales: Mapping[str, float],
    component_prefix: str,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    target_tensor = torch.as_tensor(np.array(target, copy=True), device=prediction.device, dtype=prediction.dtype)
    valid = torch.as_tensor(np.array(valid_mask, copy=True), device=prediction.device, dtype=torch.bool)
    if prediction.shape != target_tensor.shape or valid.shape != target_tensor.shape:
        raise ValueError(f"Native {component_prefix} predictions, targets, and common masks must align exactly.")
    if not bool(torch.isfinite(prediction[valid]).all()):
        raise ValueError(f"Native {component_prefix} prediction is non-finite on observed receivers.")
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    # Scale keys are inserted in physical role/channel order by _component_scales.
    channel_names = tuple(name for name in scales if name.startswith(component_prefix + "/"))
    for channel, component_name in enumerate(channel_names):
        mask = valid[:, channel] & torch.isfinite(target_tensor[:, channel])
        count = int(mask.sum().detach().cpu())
        if count < 1:
            raise ValueError(f"Native {component_prefix} component {component_name} has no valid panel rows.")
        selected_prediction = torch.where(mask, prediction[:, channel], torch.zeros_like(prediction[:, channel]))
        selected_target = torch.where(mask, target_tensor[:, channel], torch.zeros_like(target_tensor[:, channel]))
        error = (selected_prediction - selected_target) / float(scales[component_name])
        sums[component_name] = error.square().sum()
        counts[component_name] = count
    return sums, counts


def _pair_panel_loss(
    model: Any,
    pair: GeometryResponsePair,
    scales: Mapping[str, float],
    *,
    schedule_index: int,
    device: torch.device,
    dtype: torch.dtype,
    fluid_queries: int,
    local_queries_per_module: int,
    ntheta: int,
    fluid_chunk_size: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int], dict[str, Any]]:
    fluid = pair.roles["fluid_fields"]
    interface = pair.roles["interface"]
    solid = pair.roles["solid_temperature"]
    modules = len(pair.module_ids)
    fluid_indices = _sample_valid_rows(
        fluid.valid_mask,
        fluid_queries,
        seed=_stable_seed(seed, schedule_index, pair, "fluid"),
        all_channels=True,
    )
    local_features = solid.query_features.reshape(modules, -1, 2)
    local_mask = solid.valid_mask.reshape(modules, -1, len(solid.channel_names))
    common_local = np.logical_and.reduce(local_mask.all(axis=-1), axis=0)
    local_indices = _sample_valid_rows(
        common_local[:, None],
        local_queries_per_module,
        seed=_stable_seed(seed, schedule_index, pair, "solid_temperature"),
        all_channels=True,
    )
    if interface.delta.shape[0] != modules * NATIVE_ATLAS_THETA:
        raise ValueError("Geometry supervision requires the existing 64-point-per-module atlas interface rows.")
    if ntheta not in SUPPORTED_NATIVE_THETA:
        raise ValueError(f"Thermal native adapter supports only ntheta in {SUPPORTED_NATIVE_THETA}.")
    if NATIVE_ATLAS_THETA % ntheta:
        raise ValueError("The requested native surface query count must be an exact atlas-angle subsampling.")
    angle_stride = NATIVE_ATLAS_THETA // ntheta
    interface_indices = np.arange(modules * NATIVE_ATLAS_THETA).reshape(modules, NATIVE_ATLAS_THETA)[:, ::angle_stride].reshape(-1)
    if interface_indices.size != modules * ntheta:
        raise RuntimeError("Native surface receiver selection changed its module/local axes.")

    fluid_xy = fluid.query_features[fluid_indices, :2]
    local_xy = local_features[0, local_indices]
    local_flat_indices = (np.arange(modules)[:, None] * local_features.shape[1] + local_indices[None, :]).reshape(-1)
    fluid_chunks = tuple(fluid_indices[offset:offset + fluid_chunk_size]
                         for offset in range(0, len(fluid_indices), fluid_chunk_size))
    component_sums: dict[str, torch.Tensor] = {}
    component_counts: dict[str, int] = {}
    native_role_rows = {role: 0 for role in _ROLE_KEYS}
    prepared_scenes = 0
    model_calls = 0
    interface_target = interface.delta[interface_indices]
    interface_mask = interface.valid_mask[interface_indices]
    solid_target = solid.delta[local_flat_indices]
    solid_panel_mask = solid.valid_mask[local_flat_indices]

    for chunk_index, chunk_indices in enumerate(fluid_chunks):
        chunk_xy = fluid.query_features[chunk_indices, :2]
        base_structure, base_heat = _native_inputs(pair.baseline, pair.module_ids, device, dtype)
        trial_structure, trial_heat = _native_inputs(pair.trial, pair.module_ids, device, dtype)
        prepared_outputs = []
        for label, structure, heat in (("baseline", base_structure, base_heat), ("trial", trial_structure, trial_heat)):
            local_tensor = torch.as_tensor(local_xy[None], device=device, dtype=dtype)
            fluid_tensor = torch.as_tensor(chunk_xy[None], device=device, dtype=dtype)
            prepared = model.prepare_native(
                structure,
                fluid_tensor,
                local_query_points=local_tensor,
                ntheta=ntheta,
                scene_key=("thermal_geometry_response", pair.family_id, pair.label, schedule_index, chunk_index, label),
                **({"physical_heat": heat} if getattr(model, "requires_control_in_prepare", False) else {}),
            )
            output = model.apply_native(prepared, heat)
            if not isinstance(output, Mapping):
                raise TypeError("Thermal adapter apply_native must return a role mapping.")
            prepared_outputs.append(output)
            prepared_scenes += 1
            model_calls += 1
            native_role_rows["fluid_fields"] += len(chunk_indices)
            native_role_rows["interface"] += modules * ntheta
            native_role_rows["solid_temperature"] += modules * len(local_indices)
        base_output, trial_output = prepared_outputs
        base_field = base_output.get("pred_field")
        trial_field = trial_output.get("pred_field")
        if base_field is None or trial_field is None:
            raise ValueError("Thermal native adapter omitted the predicted fluid field role.")
        fluid_prediction = (trial_field - base_field)[0]
        fluid_target = fluid.delta[chunk_indices]
        fluid_mask = fluid.valid_mask[chunk_indices]
        fluid_sums, fluid_counts = _component_sum(
            fluid_prediction, fluid_target, fluid_mask, scales, "fluid_fields",
        )
        for key, value in fluid_sums.items():
            component_sums[key] = component_sums.get(key, value.new_zeros(())) + value
            component_counts[key] = component_counts.get(key, 0) + fluid_counts[key]

        if chunk_index == 0:
            base_interface = base_output.get("pred_interface")
            trial_interface = trial_output.get("pred_interface")
            if base_interface is None or trial_interface is None:
                raise ValueError("Thermal native adapter omitted the native interface response role.")
            interface_prediction = (trial_interface - base_interface)[0].reshape(-1, len(interface.channel_names))
            selected_interface = interface_prediction
            interface_sums, interface_counts = _component_sum(
                selected_interface, interface_target, interface_mask, scales, "interface",
            )
            for key, value in interface_sums.items():
                component_sums[key] = component_sums.get(key, value.new_zeros(())) + value
                component_counts[key] = component_counts.get(key, 0) + interface_counts[key]

            base_solid = base_output.get("pred_internal_temperature")
            trial_solid = trial_output.get("pred_internal_temperature")
            if base_solid is None or trial_solid is None:
                raise ValueError("Thermal native adapter omitted native module-local temperature receivers.")
            solid_prediction = (trial_solid - base_solid)[0].reshape(-1, len(solid.channel_names))
            solid_sums, solid_counts = _component_sum(
                solid_prediction, solid_target, solid_panel_mask, scales, "solid_temperature",
            )
            for key, value in solid_sums.items():
                component_sums[key] = component_sums.get(key, value.new_zeros(())) + value
                component_counts[key] = component_counts.get(key, 0) + solid_counts[key]

    component_losses = {key: component_sums[key] / component_counts[key] for key in component_sums}
    if set(component_losses) != set(scales):
        missing = sorted(set(scales) - set(component_losses))
        raise RuntimeError(f"Native geometry response omitted component losses {missing}.")
    pair_loss = torch.stack(tuple(component_losses.values())).mean()
    detail = {
        "family_id": pair.family_id,
        "physical_family_id": pair.physical_family_id,
        "label": pair.label,
        "identity": dict(pair.identity),
        "panel": {
            "fluid_query_count": int(len(fluid_indices)),
            "fluid_indices_sha256": _array_digest(fluid_indices),
            "fluid_query_ids_sha256": _json_digest([fluid.query_ids[int(index)] for index in fluid_indices]),
            "interface_query_count": int(len(interface_indices)),
            "interface_indices_sha256": _array_digest(interface_indices),
            "interface_query_ids_sha256": _json_digest([interface.query_ids[int(index)] for index in interface_indices]),
            "local_queries_per_module": int(len(local_indices)),
            "local_indices_sha256": _array_digest(local_indices),
            "local_query_ids_sha256": _json_digest([solid.query_ids[int(index)] for index in local_flat_indices]),
            "native_theta": int(ntheta),
            "fluid_chunk_count": int(len(fluid_chunks)),
        },
        "component_losses": {key: float(value.detach().cpu()) for key, value in component_losses.items()},
        "component_valid_scalar_counts": component_counts,
        "exposure": {
            "unique_geometry_pairs": 1,
            "unique_scene_inputs": 2,
            "prepared_scenes": int(prepared_scenes),
            "native_forward_calls": int(model_calls),
            "supervised_receiver_rows_per_difference": {
                "fluid_fields": int(len(fluid_indices)),
                "interface": int(len(interface_indices)),
                "solid_temperature": int(len(local_flat_indices)),
            },
            "native_receiver_rows_evaluated_across_calls": native_role_rows,
            "supervised_scalar_targets": int(sum(component_counts.values())),
        },
    }
    return pair_loss, component_losses, component_counts, detail


class GeometrySupervision:
    """TRAIN-only response atlas and bounded differentiable native loss provider.

    ``index`` passed to :meth:`loss` is the zero-based geometry schedule index.
    Normal training cycles over the fixed 16-family/direction pairs. Calibration
    evaluates one pair for each fit family with its direction offset by the
    family index, covering all four directions, and returns the equal-family mean without taking an optimizer
    step.
    """

    def __init__(
        self,
        atlas_directory: Any,
        *,
        fluid_queries: int = 512,
        local_queries_per_module: int = 64,
        material_queries_per_module: int | None = None,
        ntheta: int = 16,
        fluid_chunk_size: int | None = None,
        seed: int = 0,
        family_ids: Sequence[str] = TRAIN_GEOMETRY_FAMILIES,
    ) -> None:
        if type(fluid_queries) is not int or fluid_queries < 1:
            raise ValueError("fluid_queries must be a positive bounded receiver count.")
        if material_queries_per_module is not None:
            if type(material_queries_per_module) is not int or material_queries_per_module < 1:
                raise ValueError("material_queries_per_module must be a positive integer.")
            if local_queries_per_module != 64 and local_queries_per_module != material_queries_per_module:
                raise ValueError("local_queries_per_module and material_queries_per_module disagree.")
            local_queries_per_module = material_queries_per_module
        if type(local_queries_per_module) is not int or local_queries_per_module < 1:
            raise ValueError("local_queries_per_module must be positive.")
        if ntheta not in SUPPORTED_NATIVE_THETA:
            raise ValueError(f"ntheta must be one of {SUPPORTED_NATIVE_THETA}.")
        if fluid_chunk_size is None:
            fluid_chunk_size = fluid_queries
        if type(fluid_chunk_size) is not int or fluid_chunk_size < 1:
            raise ValueError("fluid_chunk_size must be a positive integer.")
        if type(seed) is not int:
            raise ValueError("seed must be an integer.")
        self.family_ids = tuple(family_ids)
        self._pairs, family_rows = _build_pair_index(atlas_directory, self.family_ids)
        self._pair_by_key = {(pair.family_id, pair.label): pair for pair in self._pairs}
        self.scales, scale_metadata = _component_scales(self._pairs, self.family_ids)
        self.fluid_queries = int(fluid_queries)
        self.local_queries_per_module = int(local_queries_per_module)
        self.ntheta = int(ntheta)
        self.fluid_chunk_size = int(fluid_chunk_size)
        self.seed = int(seed)
        self._metadata = {
            "schema": "thermal_geometry_response_supervision_v1",
            "source": "existing original-TRAIN physical response atlas NPZ/JSON; no solver calls",
            "training_families": list(self.family_ids),
            "development_families_excluded_from_gradients": list(DEVELOPMENT_GEOMETRY_FAMILIES),
            "physical_reference_family_ids": [row["physical_family_id"] for row in family_rows],
            "family_inventory": list(family_rows),
            "pair_identities": [dict(pair.identity) for pair in self._pairs],
            "training_variant_labels": list(SINGLE_GEOMETRY_LABELS),
            "mixed_geometry_labels_validation_only": list(MIXED_GEOMETRY_LABELS),
            "heat_transfer_labels_excluded": ["heat_transfer_plus", "heat_transfer_minus"],
            "pair_count": len(self._pairs),
            "cycle_order": [f"{pair.family_id}/{pair.label}" for pair in self._pairs],
            "component_scales": scale_metadata,
            "scale_fingerprint": _json_digest(scale_metadata),
            "scale_basis": "equal-family/equal-direction RMS of masked single-geometry reference deltas; no DEV or mixed targets",
            "loss_reduction": {
                "normalized_squared_error": "sum(valid * ((native_trial_delta - reference_delta) / train_component_scale)^2) / valid_scalar_count",
                "pair_reduction": "equal mean over selected family/direction pairs",
                "component_reduction": "equal mean over all eight role/channel components, including each of the five fluid channels separately",
                "calibration_pairs": "one pair per TRAIN family, direction offset by family index to cover i_minus/i_plus/j_minus/j_plus",
            },
            "query_panel": {
                "fluid_queries": self.fluid_queries,
                "local_queries_per_module": self.local_queries_per_module,
                "native_theta": self.ntheta,
                "atlas_theta": NATIVE_ATLAS_THETA,
                "surface_subsampling_stride": NATIVE_ATLAS_THETA // self.ntheta,
                "fluid_chunk_size": self.fluid_chunk_size,
                "seed": self.seed,
            },
            "optimizer_ownership": "caller-owned existing optimizer; this class returns a loss and never steps or zeroes an optimizer",
            "native_path": "adapter.prepare_native + adapter.apply_native for both baseline and moved geometry",
            "autograd": "enabled; adapter is placed in eval mode only during response extraction and its prior mode is restored",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """A defensive JSON-safe copy of validated source, identity, and scale metadata."""

        return copy.deepcopy(self._metadata)

    @staticmethod
    def due(primary_update: int, *, every: int = GEOMETRY_RESPONSE_PERIOD) -> bool:
        return geometry_response_due(primary_update, every=every)

    def _selected_pairs(self, index: int, calibration: bool) -> tuple[GeometryResponsePair, ...]:
        if type(index) is not int or index < 0:
            raise ValueError("Geometry schedule index must be a nonnegative zero-based integer.")
        if calibration:
            return tuple(self._pair_by_key[(family, SINGLE_GEOMETRY_LABELS[
                (offset + index) % len(SINGLE_GEOMETRY_LABELS)])]
                for offset, family in enumerate(self.family_ids))
        return (self._pairs[index % len(self._pairs)],)

    def loss(
        self,
        model: Any,
        index: int,
        device: str | torch.device | None = None,
        *,
        calibration: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return normalized finite-response loss and exposure metadata.

        ``model`` must expose the actual Thermal native adapter methods. The
        returned tensor remains connected to model parameters for the caller's
        ordinary backward/optimizer update. Calibration performs forward and
        backward-capable calls only; no optimizer state is inspected or changed.
        """

        if not callable(getattr(model, "prepare_native", None)) or not callable(getattr(model, "apply_native", None)):
            raise TypeError("Geometry supervision requires the maintained prepare_native/apply_native adapter API.")
        if not callable(getattr(model, "eval", None)) or not callable(getattr(model, "train", None)):
            raise TypeError("Geometry supervision requires a mode-aware torch adapter.")
        try:
            parameter = next(model.parameters())
        except (AttributeError, StopIteration, TypeError):
            parameter = None
        if device is None:
            target_device = parameter.device if parameter is not None else torch.device("cpu")
        else:
            target_device = torch.device(device)
        dtype = parameter.dtype if parameter is not None and parameter.is_floating_point() else torch.float32
        if dtype not in (torch.float32, torch.float64):
            dtype = torch.float32
        selected = self._selected_pairs(index, calibration)
        was_training = bool(getattr(model, "training", False))
        model_modules = tuple(model.modules()) if callable(getattr(model, "modules", None)) else (model,)
        module_training_modes = tuple(bool(getattr(module, "training", False)) for module in model_modules)
        component_rows: dict[str, list[torch.Tensor]] = {name: [] for name in self.scales}
        component_counts: dict[str, int] = {name: 0 for name in self.scales}
        pair_details = []
        pair_losses = []
        core = getattr(model, "core", None)
        previous_topology = copy.deepcopy(getattr(core, "_last_compact_topology", None))
        model.eval()
        try:
            with torch.enable_grad():
                for pair in selected:
                    loss, components, counts, detail = _pair_panel_loss(
                        model,
                        pair,
                        self.scales,
                        schedule_index=index,
                        device=target_device,
                        dtype=dtype,
                        fluid_queries=self.fluid_queries,
                        local_queries_per_module=self.local_queries_per_module,
                        ntheta=self.ntheta,
                        fluid_chunk_size=self.fluid_chunk_size,
                        seed=self.seed,
                    )
                    pair_losses.append(loss)
                    for name, value in components.items():
                        component_rows[name].append(value)
                        component_counts[name] += counts[name]
                    pair_details.append(detail)
        finally:
            if core is not None and hasattr(core, "_last_compact_topology"):
                core._last_compact_topology = previous_topology
            model.train(was_training)
            for module, training_mode in zip(model_modules[1:], module_training_modes[1:], strict=True):
                module.training = training_mode
        if not pair_losses:
            raise RuntimeError("Geometry response selection returned no eligible TRAIN pairs.")
        component_means = {name: torch.stack(values).mean() for name, values in component_rows.items()}
        total = torch.stack(tuple(component_means.values())).mean()
        exposure = {
            "pair_count": len(selected),
            "scene_count": sum(int(row["exposure"]["prepared_scenes"]) for row in pair_details),
            "prepared_scenes": sum(int(row["exposure"]["prepared_scenes"]) for row in pair_details),
            "unique_scene_inputs": sum(int(row["exposure"]["unique_scene_inputs"]) for row in pair_details),
            "native_forward_calls": sum(int(row["exposure"]["native_forward_calls"]) for row in pair_details),
            "supervised_receiver_rows": {
                role: sum(int(row["exposure"]["supervised_receiver_rows_per_difference"][role]) for row in pair_details)
                for role in _ROLE_KEYS
            },
            "native_receiver_rows_evaluated_across_calls": {
                role: sum(int(row["exposure"]["native_receiver_rows_evaluated_across_calls"][role]) for row in pair_details)
                for role in _ROLE_KEYS
            },
            "native_query_rows": sum(
                int(value)
                for row in pair_details
                for value in row["exposure"]["native_receiver_rows_evaluated_across_calls"].values()
            ),
            "supervised_scalar_targets": int(sum(component_counts.values())),
        }
        detail = {
            "schema": "thermal_geometry_response_loss_v1",
            "schedule_index": int(index),
            "calibration": bool(calibration),
            "training": bool(not calibration),
            "family_ids": [pair.family_id for pair in selected],
            "variant_labels": [pair.label for pair in selected],
            "loss": float(total.detach().cpu()),
            "components": {name: float(value.detach().cpu()) for name, value in component_means.items()},
            "component_valid_scalar_counts": component_counts,
            "component_scales": dict(self.scales),
            "pairs": pair_details,
            "exposure": exposure,
            "scene_visits": exposure["prepared_scenes"],
            "query_rows": int(sum(exposure["supervised_receiver_rows"].values())),
            "native_query_rows": exposure["native_query_rows"],
            "loss_reduction": {
                "pair_denominator": len(selected),
                "component_denominator": len(component_means),
                "components": list(component_means),
                "per_pair_component_valid_scalar_denominators": {
                    f"{row['family_id']}/{row['label']}": dict(row["component_valid_scalar_counts"])
                    for row in pair_details
                },
                "formula": "mean_pairs(mean_components(sum(valid * normalized_squared_error) / valid_scalar_count))",
            },
            "optimizer_steps": 0,
            "model_training_mode_restored": bool(getattr(model, "training", False) == was_training),
            "autograd_enabled": True,
            "reference_boundary": "only original-TRAIN families 0001/0318/0333/0348; mixed and DEV labels are excluded from this loss",
        }
        return total, detail


__all__ = [
    "DEVELOPMENT_GEOMETRY_FAMILIES",
    "GEOMETRY_RESPONSE_PERIOD",
    "GeometryResponsePair",
    "GeometrySupervision",
    "MIXED_GEOMETRY_LABELS",
    "SINGLE_GEOMETRY_LABELS",
    "TRAIN_GEOMETRY_FAMILIES",
    "geometry_response_due",
    "validate_module_major_receivers",
    "validate_training_family_metadata",
]
