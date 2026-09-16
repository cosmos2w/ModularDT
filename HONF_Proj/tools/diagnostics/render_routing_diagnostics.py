#!/usr/bin/env python3
"""Render the shared module-hub routing diagnostic figures.

The routing ledger deliberately stores only selected route arrays.  This
renderer turns those arrays into ordinary, offline PNG figures and a small
HTML index.  Field reference/prediction/error arrays are not loaded or copied
here: the index links to the endpoint figures produced by the maintained
endpoint evaluator.

The public entry point is :func:`render_routing_diagnostics`.  It accepts a
ledger JSON path (or its decoded mapping), finds one selected NPZ per anchor,
and writes the geometry, incidence, selected-pair, probability, occupancy,
and support-transition panels.  A strict call validates the complete five
anchor contract; a non-strict call is useful for a CPU fixture or an
in-progress ledger and records unavailable panels in the returned manifest.

The NPZ contract is intentionally explicit.  It keeps learned route weights
separate from physical field values and gives the renderer enough information
to distinguish M and E source incidences and to show actual deduplicated
fine-pair indices.  See ``ROUTING_MAP_KEYS`` and ``ROUTING_MAP_METADATA``
below.  Extra keys remain forward-compatible and are ignored.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

DEFAULT_ANCHOR_CASE_IDS = ("0273", "0653", "0283", "0298", "0302")
ROUTING_WEIGHT_SEMANTICS = "learned routing weights; not physical influence or field-value substitutes"

# These are the canonical names written by the ledger exporter.  The reader
# also accepts the aliases below so old/current selected maps can be rendered
# while the exporter is being migrated to the explicit contract.
ROUTING_MAP_KEYS = {
    "geometry": ("module_coords", "env_coords", "hub_coords", "hub_valid"),
    "incidence": ("module_membership", "env_membership"),
    "preparation": (
        "p0_port__hub_coords",
        "p0_port__module_source_A",
        "p0_port__environment_source_A",
        "p1_refinement__hub_coords",
        "p1_refinement__module_source_A",
        "p1_refinement__environment_source_A",
        "p2_field__hub_coords",
        "p2_field__module_source_A",
        "p2_field__environment_source_A",
    ),
    "selected_exporter": (
        "selected_port_indices",
        "selected_port_xy",
        "selected_far_query_xy",
        "p0_port_module_query_hub_alpha",
        "p0_port_module_pair_ids",
        "p0_port_module_pair_Pi",
        "p0_port_environment_query_hub_alpha",
        "p0_port_environment_pair_ids",
        "p0_port_environment_pair_Pi",
        "p2_far_module_query_hub_alpha",
        "p2_far_module_pair_ids",
        "p2_far_module_pair_Pi",
        "p2_far_environment_query_hub_alpha",
        "p2_far_environment_pair_ids",
        "p2_far_environment_pair_Pi",
    ),
    "phase": (
        "p0_selected_receiver",
        "p0_selected_hubs",
        "p0_pair_receiver",
        "p0_pair_source",
        "p0_pair_prior",
        "p0_query_hub_probability",
        "p0_query_source_prior",
        "p2_selected_receiver",
        "p2_selected_hubs",
        "p2_pair_receiver",
        "p2_pair_source",
        "p2_pair_prior",
        "p2_query_hub_probability",
        "p2_query_source_prior",
    ),
    "counts": (
        "dk_module",
        "dk_environment",
        "p0_raw_path_count",
        "p0_unique_pair_count",
        "p2_raw_path_count",
        "p2_unique_pair_count",
    ),
    "support": ("support_transition_h", "support_transition_support"),
}

ROUTING_MAP_METADATA = (
    "case_id",
    "routing_strategy",
    "source_kind",
    "coordinate_dim",
    "route_weight_semantics",
)

# An endpoint plot is owned by the endpoint evaluator.  It is accepted as an
# optional link, but is not required to be duplicated into every routing NPZ.
ROUTING_MAP_OPTIONAL_METADATA = ("field_plot_index",)

_ALIASES: dict[str, tuple[str, ...]] = {
    "module_coords": (
        "module_coords",
        "routing_module_coords",
        "module_centers",
        "routing_module_centers",
        "routing_index_module_centers",
    ),
    "module_valid": (
        "module_valid",
        "module_present",
        "routing_module_valid",
        "routing_module_present",
    ),
    "env_coords": (
        "env_coords",
        "routing_env_coords",
        "environment_coords",
        "routing_environment_coords",
        "fine_env_coords",
    ),
    "env_valid": (
        "env_valid",
        "environment_valid",
        "routing_env_valid",
        "routing_environment_valid",
    ),
    "hub_coords": (
        "hub_coords",
        "routing_hub_coords",
        "candidate_coords",
        "candidate_hub_coords",
        "routing_candidate_coords",
        "routing_index_candidate_coords",
    ),
    "hub_valid": (
        "hub_valid",
        "routing_hub_valid",
        "candidate_hub_valid",
        "routing_candidate_hub_valid",
    ),
    "module_membership": (
        "module_membership",
        "routing_module_membership",
        "routing_module_source_membership",
        "module_source_membership",
        "module_incidence_membership",
        "routing_module_incidence_membership",
    ),
    "env_membership": (
        "env_membership",
        "environment_membership",
        "routing_environment_membership",
        "routing_environment_source_membership",
        "environment_source_membership",
        "environment_incidence_membership",
        "routing_environment_incidence_membership",
    ),
    "receiver_coords": (
        "receiver_coords",
        "query_coords",
        "receivers",
        "routing_receiver_coords",
        "routing_query_coords",
    ),
    "support_h": (
        "support_transition_h",
        "support_transition_step",
        "support_transition_steps",
        "support_h",
    ),
    "support_values": (
        "support_transition_support",
        "support_transition_active_count",
        "support_transition_support_count",
        "support_transition_values",
        "support_support",
    ),
}

# The exporter keeps the preparation phase names (`p0_port`, `p1_refinement`,
# and `p2_field`) in the NPZ keys.  Selected maps use the user-facing route
# names (`p0_port` and `p2_far`).  Keeping these aliases here lets the figure
# reader consume both forms while retaining the phase-specific names in the
# manifest and figure titles.
_PHASE_ROLES: dict[str, tuple[str, ...]] = {
    "p0": ("p0", "p0_port"),
    "p1": ("p1", "p1_refinement"),
    "p2": ("p2", "p2_field", "p2_far"),
}

_PHASE_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "selected_receiver": ("selected_receiver", "receiver_index"),
    "selected_hubs": ("selected_hubs", "active_hubs"),
    "query_hub_probability": (
        "query_hub_probability",
        "query_probability",
        "query_hub_alpha",
        "alpha",
    ),
    "query_hub_support": ("query_hub_support", "hub_support", "support"),
    "query_hub_density": ("query_hub_density", "query_density", "density"),
    "query_source_prior": (
        "query_source_prior",
        "source_prior",
        "effective_Pi",
        "effective_pi",
        "effective_prior",
    ),
    "pair_ids": ("pair_ids", "pair_id"),
    "pair_receiver": ("pair_receiver", "receiver"),
    "pair_source": ("pair_source", "source"),
    "pair_prior": ("pair_prior", "pair_Pi", "pair_pi", "Pi"),
    "receiver_coords": ("receiver_coords", "query_coords"),
    "raw_path_count": ("raw_path_count", "raw_paths"),
    "unique_pair_count": ("unique_pair_count", "unique_pairs"),
    "duplicate_count": ("duplicate_count", "duplicates"),
    "duplicate_expansion": ("duplicate_expansion", "path_expansion"),
}


class RoutingFigureError(ValueError):
    """Raised when a strict routing figure contract is incomplete."""


def _aliases(name: str) -> tuple[str, ...]:
    return _ALIASES.get(name, (name,))


def _pick(data: Mapping[str, Any], *names: str) -> Any | None:
    """Return the first present value, including an explicitly empty array."""

    for name in names:
        if name in data:
            return data[name]
    return None


def _pick_named(data: Mapping[str, Any], name: str, *extra: str) -> tuple[str | None, Any | None]:
    names = (*_aliases(name), *extra)
    for candidate in names:
        if candidate in data:
            return candidate, data[candidate]
    return None, None


def _phase_names(phase: str, field: str, source: str | None = None) -> tuple[str, ...]:
    """Return canonical and exporter-era names for one phase array."""

    phase = str(phase).lower()
    roles = _PHASE_ROLES.get(phase, (phase,))
    fields = _PHASE_FIELD_ALIASES.get(field, (field,))
    names: list[str] = []
    for role in roles:
        source_token = f"_{source}" if source else ""
        for token in fields:
            names.extend(
                (
                    f"{role}{source_token}_{token}",
                    f"routing_{role}{source_token}_{token}",
                    f"routing_{role}_{token}",
                )
            )
            if source:
                names.extend(
                    (
                        f"{role}_{source}_{token}",
                        f"{source}_{role}_{token}",
                    )
                )
        # Keep the old phase prefix available for maps whose source is encoded
        # in a separate key rather than in the phase role.
        if role != phase:
            for token in fields:
                names.extend(
                    (
                        f"{phase}{source_token}_{token}",
                        f"routing_{phase}{source_token}_{token}",
                    )
                )
    return tuple(dict.fromkeys(names))


def _phase_pick(data: Mapping[str, Any], phase: str, field: str, source: str | None = None) -> Any | None:
    names = _phase_names(phase, field, source)
    for name in names:
        if name in data:
            return data[name]
    return None


def _as_array(value: Any, *, dtype: Any | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None
    return array


def _first_batch(array: np.ndarray | None) -> np.ndarray | None:
    """Remove only an unambiguous leading batch dimension."""

    if array is None:
        return None
    if array.ndim >= 3 and array.shape[0] == 1:
        return array[0]
    return array


def _points(value: Any | None) -> np.ndarray | None:
    array = _first_batch(_as_array(value, dtype=float))
    if array is None:
        return None
    if array.ndim == 1 and array.size >= 2:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    if not np.isfinite(array[:, : min(array.shape[1], 3)]).all():
        return None
    return array


def _matrix(value: Any | None) -> np.ndarray | None:
    array = _first_batch(_as_array(value, dtype=float))
    if array is None or array.ndim != 2:
        return None
    if not np.isfinite(array).all():
        return None
    return array


def _route_matrix(value: Any | None) -> np.ndarray | None:
    """Normalize a selected route vector to a receiver-by-target matrix."""

    array = _first_batch(_as_array(value, dtype=float))
    if array is None:
        return None
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return _matrix(array)


def _flat(value: Any | None, *, dtype: Any | None = None) -> np.ndarray | None:
    array = _as_array(value, dtype=dtype)
    if array is None:
        return None
    return array.reshape(-1)


def _finite_number(value: Any | None) -> float | None:
    if value is None:
        return None
    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        number = float(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any | None, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    array = np.asarray(value)
    if array.size == 1:
        item = array.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_ledger(source: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        return dict(source), None
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"routing ledger does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutingFigureError(f"unable to parse routing ledger: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RoutingFigureError("routing ledger JSON must contain an object")
    return dict(payload), path


def _resolve_path(value: Any, base: Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text or "://" in text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _case_from_path(path: Path) -> str | None:
    match = re.search(r"(?:__|case[_-]?)([A-Za-z0-9-]+)$", path.stem)
    return match.group(1) if match else None


def _map_candidates(payload: Mapping[str, Any], ledger_path: Path | None) -> dict[str, list[Path]]:
    """Collect NPZ paths from both top-level and per-row ledger records."""

    base = ledger_path.parent if ledger_path is not None else None
    result: dict[str, list[Path]] = {}

    def add(value: Any, case_id: Any | None = None) -> None:
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            path = _resolve_path(item, base)
            if path is None or path.suffix.lower() != ".npz":
                continue
            case = str(case_id) if case_id is not None else _case_from_path(path)
            if not case:
                continue
            result.setdefault(case, [])
            if path not in result[case]:
                result[case].append(path)

    add(payload.get("routing_maps"))
    add(payload.get("routing_map_paths"))
    for row in payload.get("rows", ()) if isinstance(payload.get("rows"), Sequence) else ():
        if not isinstance(row, Mapping):
            continue
        case = row.get("case_id")
        add(row.get("routing_maps", row.get("routing_map")), case)
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"routing map does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {str(key): np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise RoutingFigureError(f"unable to read routing map without pickle: {path}") from exc


def _sparse_membership(data: Mapping[str, Any], source: str, source_count: int, hub_count: int) -> np.ndarray | None:
    token = "module" if source == "module" else "environment"
    source_index = _pick(
        data,
        f"{source}_incidence_source_index",
        f"routing_{source}_incidence_source_index",
        f"{token}_source_index",
        f"routing_{token}_source_index",
    )
    hub_index = _pick(
        data,
        f"{source}_incidence_hub_index",
        f"routing_{source}_incidence_hub_index",
        f"{token}_hub_index",
        f"routing_{token}_hub_index",
    )
    values = _pick(
        data,
        f"{source}_incidence_value",
        f"{source}_incidence_values",
        f"routing_{source}_incidence_values",
        f"{token}_membership_values",
    )
    if source_index is None or hub_index is None:
        return None
    src = _flat(source_index, dtype=int)
    hub = _flat(hub_index, dtype=int)
    if src is None or hub is None or src.size != hub.size:
        return None
    if values is None:
        val = np.ones(src.size, dtype=float)
    else:
        val = _flat(values, dtype=float)
        if val is None or val.size != src.size:
            return None
    if np.any(src < 0) or np.any(src >= source_count) or np.any(hub < 0) or np.any(hub >= hub_count):
        return None
    membership = np.zeros((source_count, hub_count), dtype=float)
    np.add.at(membership, (src, hub), val)
    return membership


def _phase_preparation_pick(
    data: Mapping[str, Any], field: str, source: str | None = None
) -> Any | None:
    """Pick an explicitly phase-scoped preparation array.

    Preparation keys use a double underscore separator in the current
    exporter (for example ``p2_field__module_source_A``).  A single underscore
    spelling remains accepted for hand-authored fixtures.
    """

    source_token = f"_{source}" if source else ""
    roles = ("p2_field", "p0_port", "p1_refinement")
    for role in roles:
        for separator in ("__", "_"):
            for token in (field, field.lower()):
                name = f"{role}{separator}{token}{source_token}"
                if name in data:
                    return data[name]
                # The exporter places the source token before the field.
                if source:
                    name = f"{role}{separator}{source}{'_' + token}"
                    if name in data:
                        return data[name]
    return None


def _canonical_arrays(data: Mapping[str, Any]) -> dict[str, Any]:
    module_coords = _points(_pick_named(data, "module_coords")[1])
    env_coords = _points(_pick_named(data, "env_coords")[1])
    hub_coords = _points(_pick_named(data, "hub_coords")[1])
    module_membership = _matrix(_pick_named(data, "module_membership")[1])
    env_membership = _matrix(_pick_named(data, "env_membership")[1])
    if hub_coords is None:
        hub_coords = _points(_phase_preparation_pick(data, "hub_coords"))
    if module_membership is None:
        module_membership = _matrix(_phase_preparation_pick(data, "module_source_A"))
    if env_membership is None:
        env_membership = _matrix(_phase_preparation_pick(data, "environment_source_A"))
    if module_membership is None and module_coords is not None and hub_coords is not None:
        module_membership = _sparse_membership(data, "module", len(module_coords), len(hub_coords))
    if env_membership is None and env_coords is not None and hub_coords is not None:
        env_membership = _sparse_membership(data, "environment", len(env_coords), len(hub_coords))
    hub_valid = _flat(_pick_named(data, "hub_valid")[1], dtype=bool)
    if hub_valid is None:
        hub_valid = _flat(_phase_preparation_pick(data, "hub_valid"), dtype=bool)
    module_valid = _flat(_pick_named(data, "module_valid")[1], dtype=bool)
    if module_valid is None and module_coords is not None:
        module_valid = np.ones(len(module_coords), dtype=bool)
    env_valid = _flat(_pick_named(data, "env_valid")[1], dtype=bool)
    if env_valid is None:
        env_weights = _flat(_pick(data, "env_weights", "environment_weights"), dtype=float)
        if env_weights is not None and env_coords is not None and env_weights.size == len(env_coords):
            env_valid = env_weights > 0.0
    if env_valid is None and env_coords is not None:
        env_valid = np.ones(len(env_coords), dtype=bool)
    return {
        "module_coords": module_coords,
        "env_coords": env_coords,
        "hub_coords": hub_coords,
        "module_membership": module_membership,
        "env_membership": env_membership,
        "module_valid": module_valid,
        "env_valid": env_valid,
        "hub_valid": hub_valid,
    }


def _infer_metadata(data: Mapping[str, Any], payload: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    """Derive only metadata fixed by the ledger/model contract.

    Current run-turnover exports predate embedded NPZ metadata.  Their
    architecture, phase-specific preparation arrays, and coordinate shapes
    still determine these values exactly.  Endpoint plot ownership remains
    external and is deliberately not inferred here.
    """

    arrays = _canonical_arrays(data)
    inferred: dict[str, Any] = {}
    if "case_id" not in data:
        inferred["case_id"] = case_id
    architecture = _text(payload.get("architecture"))
    phase_keys = tuple(str(key) for key in data)
    if "routing_strategy" not in data and architecture == "routed_pairwise_honf" and any(
        key.startswith(("p0_port__", "p2_field__", "p0_port_", "p2_far_")) for key in phase_keys
    ):
        inferred["routing_strategy"] = "module_hubs"
    if "source_kind" not in data and arrays["module_coords"] is not None and arrays["env_coords"] is not None:
        inferred["source_kind"] = "module_and_environment"
    if "coordinate_dim" not in data:
        coordinate_shapes = [
            value.shape[1]
            for value in (arrays["module_coords"], arrays["env_coords"], arrays["hub_coords"])
            if value is not None and value.ndim == 2 and value.shape[1] >= 2
        ]
        if coordinate_shapes and len(set(coordinate_shapes)) == 1:
            inferred["coordinate_dim"] = int(coordinate_shapes[0])
    if "route_weight_semantics" not in data:
        inferred["route_weight_semantics"] = ROUTING_WEIGHT_SEMANTICS
    return inferred


def _phase_view(data: Mapping[str, Any], phase: str, source: str | None = None) -> dict[str, Any]:
    """Normalize one P0/P2 map while retaining whether fields were derived."""

    phase = str(phase).lower()
    derived: list[str] = []
    receiver = _phase_pick(data, phase, "selected_receiver", source)
    if receiver is None:
        receiver = _pick(
            data,
            f"{phase}_selected_receiver",
            f"{phase}_selected_receiver_index",
            f"{phase}_receiver_index",
        )
    hubs = _phase_pick(data, phase, "selected_hubs", source)
    if hubs is None:
        hubs = _phase_pick(data, phase, "active_hubs", source)
    if hubs is None:
        hubs = _pick(data, f"{phase}_selected_hubs", f"{phase}_active_hubs")

    # The current selected-map exporter stores one global port index and the
    # selected support vector rather than repeating either value per source.
    selected_indices = _as_array(_pick(data, "selected_port_indices"), dtype=int)
    selected_indices = None if selected_indices is None else selected_indices.reshape(-1)
    if receiver is None and selected_indices is not None and phase == "p0" and selected_indices.size >= 3:
        receiver = selected_indices[2:3]
        derived.append("selected_receiver_from_selected_port_indices")

    receiver_coords = _phase_pick(data, phase, "receiver_coords", source)
    if receiver_coords is None:
        receiver_coords = _pick(data, f"{phase}_receiver_coords", f"{phase}_query_coords")
    if receiver_coords is None:
        receiver_coords = _pick_named(data, "receiver_coords")[1]
    if receiver_coords is None:
        receiver_coords = _pick(data, "selected_port_xy" if phase == "p0" else "selected_far_query_xy")
        if receiver_coords is not None:
            derived.append("receiver_coords_from_selected_query_coordinate")

    hub_probability = _phase_pick(data, phase, "query_hub_probability", source)
    # Existing final maps use routing_module_query_probability and
    # routing_environment_query_probability.  They are P2 maps by contract.
    if hub_probability is None and phase == "p2" and source:
        hub_probability = _pick(
            data,
            f"routing_{source}_query_probability",
            f"routing_{source}_query_hub_probability",
            f"{source}_query_probability",
        )
    source_prior = _phase_pick(data, phase, "query_source_prior", source)
    pair_receiver = _phase_pick(data, phase, "pair_receiver", source)
    pair_source = _phase_pick(data, phase, "pair_source", source)
    pair_prior = _phase_pick(data, phase, "pair_prior", source)

    pair_ids = _phase_pick(data, phase, "pair_ids", source)
    pair_ids_array = _as_array(pair_ids, dtype=int)
    if pair_ids_array is not None:
        pair_ids_array = pair_ids_array.reshape(-1, pair_ids_array.shape[-1]) if pair_ids_array.ndim == 2 else None
        if pair_ids_array is not None and pair_ids_array.shape[1] >= 3:
            if pair_receiver is None:
                pair_receiver = pair_ids_array[:, 1]
                derived.append("pair_receiver_from_batch_receiver_source_ids")
            if pair_source is None:
                pair_source = pair_ids_array[:, 2]
                derived.append("pair_source_from_batch_receiver_source_ids")
        elif pair_ids_array is not None and pair_ids_array.shape[1] == 2:
            if pair_receiver is None:
                pair_receiver = pair_ids_array[:, 0]
                derived.append("pair_receiver_from_receiver_source_ids")
            if pair_source is None:
                pair_source = pair_ids_array[:, 1]
                derived.append("pair_source_from_receiver_source_ids")

    # Selected maps call the denominator-cancelled prior ``pair_Pi`` and
    # ``effective_Pi``.  The phase aliases above resolve those names before
    # this point; retain a direct legacy fallback for old P2 maps.
    if phase == "p2" and source:
        pair_receiver = pair_receiver if pair_receiver is not None else _pick(data, f"routing_{source}_pair_receiver")
        pair_source = pair_source if pair_source is not None else _pick(data, f"routing_{source}_pair_source")
        pair_prior = pair_prior if pair_prior is not None else _pick(data, f"routing_{source}_pair_prior")

    support = _phase_pick(data, phase, "query_hub_support", source)
    support_array = _flat(support)
    if hubs is None and support_array is not None:
        hubs = np.flatnonzero(support_array > 0.0).astype(int)
        derived.append("selected_hubs_from_query_hub_support")

    if receiver is None and pair_receiver is not None:
        pair_receiver_array = _flat(pair_receiver, dtype=int)
        if pair_receiver_array is not None and pair_receiver_array.size:
            finite_receivers = pair_receiver_array[pair_receiver_array >= 0]
            if finite_receivers.size:
                receiver = np.asarray([finite_receivers[0]], dtype=int)
                derived.append("selected_receiver_from_pair_ids")

    result: dict[str, Any] = {
        "phase": phase,
        "source": source or "combined",
        "selected_receiver": _flat(receiver, dtype=int),
        "selected_hubs": _flat(hubs, dtype=int),
        "receiver_coords": _points(receiver_coords),
        "query_hub_probability": _route_matrix(hub_probability),
        "query_source_prior": _route_matrix(source_prior),
        "pair_receiver": _flat(pair_receiver, dtype=int),
        "pair_source": _flat(pair_source, dtype=int),
        "pair_prior": _flat(pair_prior, dtype=float),
        "pair_ids": pair_ids_array,
        "derived": derived,
    }
    port_index = _phase_pick(data, phase, "selected_port_index", source)
    far_index = _phase_pick(data, phase, "far_receiver_index", source)
    if port_index is None:
        port_index = _pick(
            data,
            f"{phase}_selected_port_index",
            f"{phase}_port_receiver_index",
            "selected_port_index",
            "port_receiver_index",
            "port_index",
        )
    if far_index is None:
        far_index = _pick(
            data,
            f"{phase}_far_receiver_index",
            f"{phase}_far_field_receiver_index",
            "far_receiver_index",
            "far_field_receiver_index",
            "far_index",
        )
    if port_index is None and phase == "p0" and selected_indices is not None and selected_indices.size >= 3:
        port_index = selected_indices[2]
        derived.append("port_index_from_selected_port_indices")
    if far_index is None and phase == "p2" and pair_receiver is not None:
        pair_receiver_array = _flat(pair_receiver, dtype=int)
        if pair_receiver_array is not None and pair_receiver_array.size:
            finite_receivers = pair_receiver_array[pair_receiver_array >= 0]
            if finite_receivers.size:
                far_index = finite_receivers[0]
                derived.append("far_index_from_pair_ids")
    result["port_index"] = _finite_number(port_index)
    result["far_index"] = _finite_number(far_index)
    raw = _phase_pick(data, phase, "raw_path_count", source)
    unique = _phase_pick(data, phase, "unique_pair_count", source)
    duplicate = _phase_pick(data, phase, "duplicate_count", source)
    expansion = _phase_pick(data, phase, "duplicate_expansion", source)
    if phase == "p2" and source:
        raw = raw if raw is not None else _pick(data, f"routing_{source}_raw_path_count")
        unique = unique if unique is not None else _pick(data, f"routing_{source}_unique_pair_count")
        expansion = expansion if expansion is not None else _pick(data, f"routing_{source}_duplicate_expansion")
    raw = raw if raw is not None else _pick(data, f"{phase}_raw_path_count")
    unique = unique if unique is not None else _pick(data, f"{phase}_unique_pair_count")
    duplicate = duplicate if duplicate is not None else _pick(data, f"{phase}_duplicate_count")
    expansion = expansion if expansion is not None else _pick(data, f"{phase}_duplicate_expansion")
    # Selected NPZs intentionally contain the already compiled unique pairs;
    # derive that count only when the ledger did not provide the scalar.  Raw
    # path and duplicate counts stay missing unless the exporter recorded
    # them, because the selected array cannot recover pre-dedup multiplicity.
    if unique is None and result["pair_receiver"] is not None and result["pair_source"] is not None:
        pair_receiver_array = result["pair_receiver"]
        pair_source_array = result["pair_source"]
        count = min(pair_receiver_array.size, pair_source_array.size)
        if count:
            unique_pairs = np.unique(np.column_stack((pair_receiver_array[:count], pair_source_array[:count])), axis=0)
            unique = np.asarray(float(len(unique_pairs)))
            derived.append("unique_pair_count_from_exported_pair_ids")
    result["raw_path_count"] = _finite_number(raw)
    result["unique_pair_count"] = _finite_number(unique)
    result["duplicate_count"] = _finite_number(duplicate)
    result["duplicate_expansion"] = _finite_number(expansion)
    return result


def _derive_selected_hubs(view: dict[str, Any], receiver_index: int | None = None) -> np.ndarray:
    hubs = view.get("selected_hubs")
    if hubs is not None and hubs.size:
        return np.unique(hubs[hubs >= 0])
    probabilities = view.get("query_hub_probability")
    if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[0]:
        row = 0 if receiver_index is None else int(receiver_index)
        if 0 <= row < probabilities.shape[0]:
            return np.flatnonzero(probabilities[row] > 0.0).astype(int)
    return np.empty((0,), dtype=int)


def _derive_source_prior(view: dict[str, Any], source_count: int) -> np.ndarray | None:
    direct = view.get("query_source_prior")
    if direct is not None:
        if direct.ndim != 2:
            return None
        return direct
    receiver = view.get("pair_receiver")
    source = view.get("pair_source")
    prior = view.get("pair_prior")
    if receiver is None or source is None or prior is None:
        return None
    if not (receiver.size == source.size == prior.size) or source_count <= 0:
        return None
    valid = (receiver >= 0) & (source >= 0) & (source < source_count) & np.isfinite(prior)
    if not valid.any():
        return None
    q_count = int(receiver[valid].max()) + 1
    result = np.zeros((q_count, source_count), dtype=float)
    np.add.at(result, (receiver[valid], source[valid]), prior[valid])
    view.setdefault("derived", []).append("query_source_prior_from_deduplicated_pairs")
    return result


def _derive_dk(membership: np.ndarray | None) -> np.ndarray | None:
    if membership is None or membership.ndim != 2:
        return None
    return np.sum(membership > 0.0, axis=0, dtype=int)


def validate_routing_map(
    data: Mapping[str, Any],
    *,
    expected_phases: Sequence[str] = ("p0", "p2"),
    require_support_transition: bool = True,
) -> dict[str, Any]:
    """Validate the selected-map contract without importing plotting code.

    The returned lists are suitable for the manifest and are intentionally
    based on metadata/array presence rather than on a model-specific shape.
    ``strict=True`` in :func:`render_routing_diagnostics` raises when these
    lists are nonempty.  ``require_support_transition=False`` is used when a
    validated external run_turnover trace supplies the support evidence.
    """

    arrays = _canonical_arrays(data)
    missing: list[str] = []
    for key in ("module_coords", "env_coords", "hub_coords", "module_membership", "env_membership"):
        if arrays[key] is None:
            missing.append(key)
    phase_missing: dict[str, list[str]] = {}
    for phase in expected_phases:
        phase_missing[str(phase)] = []
        views = [_phase_view(data, str(phase), source) for source in ("module", "environment")]
        for source, view in zip(("module", "environment"), views):
            if view["query_hub_probability"] is None:
                phase_missing[str(phase)].append(f"{source}.query_hub_probability")
            if view["query_source_prior"] is None:
                phase_missing[str(phase)].append(f"{source}.query_source_prior")
            if not (
                view["pair_receiver"] is not None
                and view["pair_source"] is not None
                and view["pair_prior"] is not None
            ):
                phase_missing[str(phase)].append(f"{source}.deduplicated_pairs")
        if not any(view["selected_receiver"] is not None for view in views) and _pick(data, f"{phase}_selected_receiver") is None:
            phase_missing[str(phase)].append("selected_receiver")
        if not any(view["selected_hubs"] is not None for view in views) and _pick(data, f"{phase}_selected_hubs") is None:
            phase_missing[str(phase)].append("selected_hubs")
        if not any(view["receiver_coords"] is not None for view in views):
            phase_missing[str(phase)].append("receiver_coords")
        if not any(view["raw_path_count"] is not None for view in views):
            phase_missing[str(phase)].append("raw_path_count")
        if not any(view["unique_pair_count"] is not None for view in views):
            phase_missing[str(phase)].append("unique_pair_count")
    for phase, values in phase_missing.items():
        missing.extend(f"{phase}.{value}" for value in values)
    if require_support_transition and _support_trace(data) is None:
        missing.append("support_transition")
    metadata_missing = [
        key
        for key in ROUTING_MAP_METADATA
        if key not in data or not _text(data[key]).strip()
    ]
    missing.extend(f"metadata.{key}" for key in metadata_missing)
    return {
        "complete": not missing,
        "missing": missing,
        "phase_missing": phase_missing,
        "metadata_missing": metadata_missing,
        "arrays": {key: None if value is None else list(value.shape) for key, value in arrays.items()},
    }


def _mpl() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _placeholder(path: Path, title: str, lines: Sequence[str]) -> None:
    _matplotlib, plt = _mpl()
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    axis.axis("off")
    axis.text(0.02, 0.92, title, fontsize=13, weight="bold", va="top")
    axis.text(0.02, 0.80, "\n".join(lines), fontsize=10, va="top", color="#53606e")
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _display_matrix(matrix: np.ndarray, *, max_rows: int = 512, max_cols: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.linspace(0, matrix.shape[0] - 1, min(matrix.shape[0], max_rows), dtype=int)
    cols = np.linspace(0, matrix.shape[1] - 1, min(matrix.shape[1], max_cols), dtype=int)
    return matrix[np.ix_(np.unique(rows), np.unique(cols))], np.unique(rows), np.unique(cols)


def _plot_points(axis: Any, points: np.ndarray | None, *, label: str, color: str, marker: str, size: float) -> None:
    if points is None or not len(points):
        return
    coordinates = points[:, :2]
    kwargs: dict[str, Any] = {"s": size, "marker": marker, "label": label, "edgecolors": "none"}
    if points.shape[1] >= 3:
        kwargs.update({"c": points[:, 2], "cmap": "viridis", "edgecolors": "black", "linewidths": 0.2})
    else:
        kwargs["color"] = color
    scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], **kwargs)
    if points.shape[1] >= 3:
        scatter.figure.colorbar(scatter, ax=axis, pad=0.02, label="z coordinate")


def _set_spatial_axis(axis: Any, points: Sequence[np.ndarray | None]) -> None:
    finite_parts = [value[:, :2] for value in points if value is not None and len(value)]
    if finite_parts:
        values = np.concatenate(finite_parts, axis=0)
        lo = np.min(values, axis=0)
        hi = np.max(values, axis=0)
        span = np.maximum(hi - lo, 1.0e-8)
        axis.set_xlim(lo[0] - 0.05 * span[0], hi[0] + 0.05 * span[0])
        axis.set_ylim(lo[1] - 0.05 * span[1], hi[1] + 0.05 * span[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x coordinate")
    axis.set_ylabel("y coordinate")
    axis.grid(alpha=0.18)


def _render_geometry(path: Path, case_id: str, arrays: Mapping[str, Any], views: Mapping[str, dict[str, Any]]) -> None:
    _matplotlib, plt = _mpl()
    figure, axis = plt.subplots(figsize=(8.8, 6.0), constrained_layout=True)
    _plot_points(axis, arrays.get("env_coords"), label="environment samples (E)", color="#9aa0a6", marker=".", size=20)
    _plot_points(axis, arrays.get("module_coords"), label="physical modules (M)", color="#315f7c", marker="s", size=44)
    _plot_points(axis, arrays.get("hub_coords"), label="candidate hub descriptors", color="#bf6b3c", marker="D", size=62)
    selected: set[int] = set()
    for view in views.values():
        selected.update(int(value) for value in _derive_selected_hubs(view))
    hubs = arrays.get("hub_coords")
    if hubs is not None and selected:
        indices = np.asarray(sorted(selected), dtype=int)
        indices = indices[(indices >= 0) & (indices < len(hubs))]
        if len(indices):
            axis.scatter(hubs[indices, 0], hubs[indices, 1], facecolors="none", edgecolors="#e53e3e", linewidths=1.4, s=150, label="active selected hubs")
    _set_spatial_axis(axis, [arrays.get("module_coords"), arrays.get("env_coords"), arrays.get("hub_coords")])
    axis.set_title(f"Module-hub routing geometry · case {case_id}\nHub coordinates are descriptors, not field values")
    axis.legend(loc="best", frameon=True, fontsize=8)
    figure.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(figure)


def _render_incidence(path: Path, case_id: str, arrays: Mapping[str, Any]) -> None:
    _matplotlib, plt = _mpl()
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    for axis, key, source_label, validity_key in zip(
        axes,
        ("module_membership", "env_membership"),
        ("module source", "environment source"),
        ("module_valid", "env_valid"),
    ):
        matrix = arrays.get(key)
        title = "M source incidence" if source_label == "module source" else "E source incidence"
        if matrix is None:
            axis.axis("off")
            axis.text(0.02, 0.92, f"{title}\nmissing selected membership", va="top", color="#53606e")
            continue
        shown, rows, cols = _display_matrix(matrix)
        source_valid = arrays.get(validity_key)
        hub_valid = arrays.get("hub_valid")
        source_mask = np.ones(matrix.shape[0], dtype=bool) if source_valid is None or len(source_valid) != matrix.shape[0] else np.asarray(source_valid, dtype=bool)
        hub_mask = np.ones(matrix.shape[1], dtype=bool) if hub_valid is None or len(hub_valid) != matrix.shape[1] else np.asarray(hub_valid, dtype=bool)
        shown_mask = (~source_mask[rows, None]) | (~hub_mask[cols][None, :])
        shown = np.ma.array(shown, mask=shown_mask)
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#d7dde2")
        valid_values = shown.compressed()
        vmax = max(float(np.max(valid_values)) if len(valid_values) else 0.0, 1.0e-12)
        image = axis.imshow(shown, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
        source_count = int(np.count_nonzero(source_mask))
        hub_count = int(np.count_nonzero(hub_mask))
        source_token = "M" if source_label == "module source" else "E"
        axis.set_title(
            f"{title}\nactive {source_token} {source_count} / packed {matrix.shape[0]} source rows; "
            f"active K {hub_count} / packed {matrix.shape[1]} hubs"
        )
        axis.set_xlabel("candidate hub index")
        axis.set_ylabel(f"{source_label} index")
        axis.set_xticks(np.linspace(0, len(cols) - 1, min(len(cols), 10), dtype=int))
        axis.set_yticks(np.linspace(0, len(rows) - 1, min(len(rows), 10), dtype=int))
        figure.colorbar(image, ax=axis, pad=0.02, label="ordinary source membership")
    figure.suptitle(f"M/E source incidences after ordinary source sparsemax · case {case_id}")
    figure.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(figure)


def _receiver_index(view: Mapping[str, Any], kind: str, receiver_count: int) -> int | None:
    value = view.get("port_index" if kind == "port" else "far_index")
    if value is None:
        selected = view.get("selected_receiver")
        if selected is not None and selected.size:
            value = float(selected[0] if kind == "port" else selected[-1])
    if value is None:
        # A fixture without explicit labels can still be inspected, but this
        # fallback is reported in the manifest as derived selection metadata.
        value = 0.0 if kind == "port" else float(max(receiver_count - 1, 0))
    index = int(value)
    if 0 <= index < receiver_count:
        return index
    # Selected exporter maps carry one coordinate (`selected_port_xy` or
    # `selected_far_query_xy`) but pair IDs retain the global receiver index.
    # Return that global index so pair filtering remains correct; the drawing
    # code maps it to the sole displayed coordinate below.
    return index if receiver_count == 1 and index >= 0 else None


def _pair_rows(view: Mapping[str, Any], receiver_index: int, source_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    receiver = view.get("pair_receiver")
    source = view.get("pair_source")
    prior = view.get("pair_prior")
    if receiver is None or source is None:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)
    if prior is None:
        prior = np.ones(receiver.size, dtype=float)
    count = min(receiver.size, source.size, prior.size)
    receiver = receiver[:count]
    source = source[:count]
    prior = prior[:count]
    valid = (receiver == receiver_index) & (source >= 0) & (source < source_count) & np.isfinite(prior) & (prior > 0.0)
    receiver = receiver[valid]
    source = source[valid]
    prior = prior[valid]
    if not len(source):
        return receiver, source, prior
    # Exported pairs are expected to be deduplicated already.  Coalescing the
    # plotting view as well keeps the claim true for an older map that still
    # contains repeated paths and preserves the summed live prior.
    unique_source, inverse = np.unique(source, return_inverse=True)
    unique_prior = np.zeros(len(unique_source), dtype=float)
    np.add.at(unique_prior, inverse, prior)
    return np.full(len(unique_source), receiver_index, dtype=int), unique_source, unique_prior


def _render_selected(path: Path, case_id: str, arrays: Mapping[str, Any], phase_views: Mapping[str, Mapping[str, dict[str, Any]]]) -> None:
    _matplotlib, plt = _mpl()
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.9), constrained_layout=True)
    hubs = arrays.get("hub_coords")
    source_styles = (("module", "module_coords", "#315f7c", "s"), ("environment", "env_coords", "#9aa0a6", "."))
    for axis, kind in zip(axes, ("port", "far")):
        drawn = False
        labels: list[str] = []
        annotation_count = 0
        phase = "p0" if kind == "port" else "p2"
        source_views = phase_views.get(phase, {})
        # Both M and E views describe the same selected receiver.  Draw the
        # receiver and active hubs once, then draw each source's deduplicated
        # pairs once; doing this per source view would repeat every edge and
        # overstate occupancy visually.
        view = next((candidate for candidate in source_views.values() if candidate.get("receiver_coords") is not None), None)
        if view is not None:
            receiver_coords = view.get("receiver_coords")
            if receiver_coords is not None and len(receiver_coords):
                index = _receiver_index(view, kind, len(receiver_coords))
                if index is not None:
                    display_index = index if index < len(receiver_coords) else (0 if len(receiver_coords) == 1 else None)
                    if display_index is not None:
                        receiver_xy = receiver_coords[display_index, :2]
                        axis.scatter([receiver_xy[0]], [receiver_xy[1]], marker="*", s=180, color="#e53e3e", edgecolors="black", linewidths=0.45, zorder=6, label=f"selected {phase.upper()} receiver")
                        labels.append(f"selected {phase.upper()} receiver")
                        active_hubs = np.unique(
                            np.concatenate(
                                [_derive_selected_hubs(candidate, display_index) for candidate in source_views.values() if candidate.get("query_hub_probability") is not None or candidate.get("selected_hubs") is not None]
                                or [np.empty((0,), dtype=int)]
                            )
                        )
                        if hubs is not None and len(active_hubs):
                            active_hubs = active_hubs[(active_hubs >= 0) & (active_hubs < len(hubs))]
                            if len(active_hubs):
                                axis.scatter(hubs[active_hubs, 0], hubs[active_hubs, 1], marker="D", s=58, facecolors="none", edgecolors="#bf6b3c", linewidths=1.1, zorder=5, label="active selected hubs")
                                for hub_index in active_hubs:
                                    axis.text(float(hubs[hub_index, 0]), float(hubs[hub_index, 1]), str(int(hub_index)), fontsize=6, ha="center", va="center")
                        for source_name, coord_key, color, marker in source_styles:
                            source_coords = arrays.get(coord_key)
                            if source_coords is None:
                                continue
                            pair_view = source_views.get(source_name)
                            if pair_view is None:
                                continue
                            _recv, source_index, prior = _pair_rows(pair_view, index, len(source_coords))
                            if len(source_index):
                                # A selected pair panel may contain a large Q.
                                # Keep every unique pair count in the annotation
                                # but draw at most 512 highest-prior edges.
                                keep = np.arange(len(source_index))
                                if len(keep) > 512:
                                    keep = np.argsort(prior)[-512:]
                                points = source_coords[source_index]
                                label = f"{source_name} unique fine sources"
                                axis.scatter(points[:, 0], points[:, 1], marker=marker, s=30 if marker == "." else 45, color=color, alpha=0.8, label=label, zorder=3)
                                for row in keep:
                                    point = points[row]
                                    weight = float(prior[row])
                                    axis.plot([receiver_xy[0], point[0]], [receiver_xy[1], point[1]], color=color, alpha=float(min(0.75, 0.12 + 0.5 * weight / max(float(np.max(prior)), 1.0e-12))), linewidth=0.55, zorder=1)
                                drawn = True
                                axis.text(0.02, 0.98 - 0.06 * annotation_count, f"{phase.upper()} {source_name}: {len(source_index)} unique pairs", transform=axis.transAxes, va="top", fontsize=8)
                                annotation_count += 1
        if hubs is not None and not drawn:
            # Keep the geometry context visible even when selected pair arrays
            # are absent; validation/manifest explains the missing contract.
            _plot_points(axis, arrays.get("env_coords"), label="environment samples", color="#9aa0a6", marker=".", size=18)
            _plot_points(axis, arrays.get("module_coords"), label="physical modules", color="#315f7c", marker="s", size=40)
            _plot_points(axis, hubs, label="candidate hubs", color="#bf6b3c", marker="D", size=56)
        _set_spatial_axis(axis, [arrays.get("module_coords"), arrays.get("env_coords"), hubs])
        axis.set_title(f"Selected {kind} receiver · {phase.upper()} active hubs and unique fine pairs")
        if labels:
            handles, handle_labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, handle_labels, fontsize=7, frameon=True, loc="best")
    figure.suptitle(f"Selected receiver routing paths · case {case_id}\nLeft: P0 port · right: P2 far query · lines show deduplicated receiver–source pairs")
    figure.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(figure)


def _probability_image(
    axis: Any,
    value: np.ndarray | None,
    title: str,
    *,
    xlabel: str,
    ylabel: str,
    valid_columns: np.ndarray | None = None,
    valid_column_label: str = "target",
) -> None:
    _matplotlib, plt = _mpl()
    if value is None or value.ndim != 2:
        axis.axis("off")
        axis.text(0.02, 0.92, f"{title}\nmissing selected map", va="top", color="#53606e")
        return
    shown, _rows, cols = _display_matrix(value)
    if valid_columns is not None and len(valid_columns) == value.shape[1]:
        valid_columns = np.asarray(valid_columns, dtype=bool)
        shown = np.ma.array(shown, mask=~valid_columns[cols][None, :])
        title = f"{title}\nactive {valid_column_label} {int(np.count_nonzero(valid_columns))} / packed {value.shape[1]}"
    valid_values = shown.compressed() if np.ma.isMaskedArray(shown) else shown.reshape(-1)
    positive = valid_values[np.isfinite(valid_values) & (valid_values > 0.0)]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d7dde2")
    if not len(positive):
        image = axis.imshow(shown, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    elif float(np.max(positive)) / max(float(np.min(positive)), np.finfo(float).tiny) > 100.0:
        from matplotlib.colors import LogNorm

        masked = np.ma.masked_where(shown <= 0.0, shown)
        image = axis.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive))))
    else:
        image = axis.imshow(shown, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=max(float(np.max(positive)), 1.0e-12))
    axis.set_title(f"{title}\nshape {value.shape}")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xticks(np.linspace(0, shown.shape[1] - 1, min(shown.shape[1], 10), dtype=int))
    axis.set_yticks(np.linspace(0, shown.shape[0] - 1, min(shown.shape[0], 8), dtype=int))
    axis.figure.colorbar(image, ax=axis, pad=0.02, label="route weight")


def _render_probabilities(path: Path, case_id: str, arrays: Mapping[str, Any], phase_views: Mapping[str, Mapping[str, dict[str, Any]]]) -> None:
    _matplotlib, plt = _mpl()
    figure, axes = plt.subplots(2, 4, figsize=(17.0, 8.4), constrained_layout=True, squeeze=False)
    for row_index, phase in enumerate(("p0", "p2")):
        source_views = phase_views.get(phase, {})
        for source_index, source in enumerate(("module", "environment")):
            view = source_views.get(source, {})
            probability = view.get("query_hub_probability")
            _probability_image(
                axes[row_index, source_index],
                probability,
                f"{phase.upper()} {source} query–hub probability",
                xlabel="hub",
                ylabel="receiver",
                valid_columns=arrays.get("hub_valid"),
                valid_column_label="K",
            )
            source_coords = arrays.get("module_coords") if source == "module" else arrays.get("env_coords")
            source_count = 0 if source_coords is None else len(source_coords)
            source_prior = _derive_source_prior(view, source_count)
            _probability_image(
                axes[row_index, source_index + 2],
                source_prior,
                f"{phase.upper()} {source} final query–source prior",
                xlabel=f"{source} source",
                ylabel="receiver",
                valid_columns=arrays.get("module_valid" if source == "module" else "env_valid"),
                valid_column_label=source,
            )
    figure.suptitle(f"Query–hub probabilities and final query–source priors · case {case_id}\nRoute weights are learned routing quantities; quadrature is already included in the final prior")
    figure.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(figure)


def _render_occupancy(path: Path, case_id: str, arrays: Mapping[str, Any], phase_views: Mapping[str, Mapping[str, dict[str, Any]]], data: Mapping[str, Any]) -> None:
    _matplotlib, plt = _mpl()
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True, squeeze=False)
    from matplotlib.patches import Patch

    hub_valid = arrays.get("hub_valid")
    if hub_valid is not None:
        hub_valid = _flat(hub_valid, dtype=bool)
    for axis, source, key in zip(axes[0], ("module", "environment"), ("module_membership", "env_membership")):
        dk = _flat(_pick(data, "dk_module" if source == "module" else "dk_environment"), dtype=float)
        derived = False
        if dk is None:
            dk = _derive_dk(arrays.get(key))
            derived = True
        if dk is None:
            axis.axis("off")
            axis.text(0.02, 0.92, f"{source} Dₖ\nmissing membership", va="top", color="#53606e")
            continue
        if hub_valid is None or hub_valid.size != dk.size:
            valid = np.ones(dk.size, dtype=bool)
            validity_note = "validity metadata unavailable"
        else:
            valid = hub_valid
            validity_note = "valid zero occupancy and occupied hubs distinguished"
        occupied = valid & (dk > 0.0)
        valid_zero = valid & ~occupied
        invalid = ~valid
        base_color = "#315f7c" if source == "module" else "#9aa0a6"
        colors = np.full(dk.size, "#d7dde2", dtype=object)
        colors[occupied] = base_color
        colors[valid_zero] = "#f2c28f"
        bars = axis.bar(np.arange(len(dk)), dk, color=colors)
        for index, bar in enumerate(bars):
            if invalid[index]:
                bar.set_hatch("//")
                bar.set_edgecolor("#7a8791")
                axis.axvspan(index - 0.48, index + 0.48, facecolor="#d7dde2", alpha=0.65, hatch="//", edgecolor="#7a8791", linewidth=0.35, zorder=0)
            elif valid_zero[index]:
                axis.plot(index, 0.0, marker="o", markersize=5.5, color="#f2c28f", markeredgecolor="#8b4513", markeredgewidth=0.55, zorder=5)
        legend_handles = [
            Patch(facecolor=base_color, label="valid occupied hubs (Dₖ > 0)"),
            Patch(facecolor="#f2c28f", label="valid zero occupancy (Dₖ = 0)"),
        ]
        if invalid.any():
            legend_handles.append(Patch(facecolor="#d7dde2", edgecolor="#7a8791", hatch="//", label="invalid/padded slots"))
        axis.legend(handles=legend_handles, fontsize=7, frameon=True, loc="best")
        axis.set_title(f"True Dₖ occupancy · P2 source incidence · {source}{' (derived from incidence)' if derived else ''}\n{validity_note}")
        axis.set_xlabel("hub index")
        axis.set_ylabel("active positive source rows")
        axis.grid(axis="y", alpha=0.22)
    phases = ("p0", "p2")
    for axis, source in zip(axes[1], ("module", "environment")):
        raw_values: list[float] = []
        unique_values: list[float] = []
        for phase in phases:
            view = phase_views.get(phase, {}).get(source, {})
            raw = view.get("raw_path_count")
            unique = view.get("unique_pair_count")
            if raw is None:
                raw = _finite_number(_pick(data, f"{phase}_{source}_raw_path_count", f"{phase}_raw_path_count"))
            if unique is None:
                unique = _finite_number(_pick(data, f"{phase}_{source}_unique_pair_count", f"{phase}_unique_pair_count"))
            raw_values.append(float(raw) if raw is not None else float("nan"))
            unique_values.append(float(unique) if unique is not None else float("nan"))
        x = np.arange(len(phases))
        width = 0.36
        axis.bar(x - width / 2, raw_values, width, label="raw two-hop paths", color="#bf6b3c")
        axis.bar(x + width / 2, unique_values, width, label="unique receiver–source pairs", color="#315f7c")
        axis.set_xticks(x, [phase.upper() for phase in phases])
        axis.set_title(f"Actual path duplication counts · {source}")
        axis.set_ylabel("count")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(fontsize=8, frameon=True)
    figure.suptitle(f"Source occupancy and two-hop deduplication · case {case_id}\nDₖ counts positive source rows per valid candidate hub; invalid padded slots are masked")
    figure.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(figure)


def _support_trace(data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    h = _as_array(_pick_named(data, "support_h")[1], dtype=float)
    values = _as_array(_pick_named(data, "support_values")[1], dtype=float)
    if h is None or values is None:
        return None
    h = h.reshape(-1)
    if values.ndim > 1:
        values = np.sum(values > 0.0, axis=tuple(range(1, values.ndim)), dtype=int)
    values = values.reshape(-1)
    count = min(len(h), len(values))
    if count == 0:
        return None
    finite = np.isfinite(h[:count]) & np.isfinite(values[:count])
    if not finite.any():
        return None
    return h[:count][finite], values[:count][finite]


def _render_support(path: Path, traces: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> None:
    _matplotlib, plt = _mpl()
    figure, axis = plt.subplots(figsize=(9.8, 5.4), constrained_layout=True)
    if not traces:
        axis.axis("off")
        axis.text(0.02, 0.92, "Support transition trace unavailable\nSelected maps contain no support_transition_h/support_transition_support arrays.", va="top", color="#53606e")
    else:
        for case_id, (h, values) in traces.items():
            order = np.argsort(h, kind="stable")
            axis.step(h[order], values[order], where="mid", marker="o", markersize=3.5, linewidth=1.1, label=case_id)
        axis.axvline(0.0, color="#53606e", linewidth=0.8, alpha=0.7)
        axis.set_xlabel("support-transition perturbation h")
        axis.set_ylabel("positive support count")
        axis.set_title("Sparsemax support transition trace")
        axis.grid(alpha=0.22)
        axis.legend(title="anchor case", fontsize=8, frameon=True)
    figure.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(figure)


_TURNOVER_PHASES = ("p0_port", "p1_refinement", "p2_field")
_TURNOVER_METRICS = ("query_support_jaccard", "source_support_jaccard")
_TURNOVER_SOURCES = ("module", "environment")


def _turnover_source_items(
    turnover: Any,
    *,
    ledger_path: Path | None,
    output: Path,
) -> list[tuple[str, Mapping[str, Any], str]]:
    """Load repeatable run_turnover JSON inputs and preserve provenance."""

    if turnover is None:
        return []
    values: Sequence[Any]
    if isinstance(turnover, (str, Path, Mapping)):
        values = (turnover,)
    elif isinstance(turnover, Sequence):
        values = turnover
    else:
        raise RoutingFigureError("turnover must be a JSON path, mapping, or repeatable sequence of either")
    base = ledger_path.parent if ledger_path is not None else Path.cwd()
    records: list[tuple[str, Mapping[str, Any], str]] = []
    for index, item in enumerate(values):
        provenance = f"turnover[{index}]"
        if isinstance(item, Mapping):
            payload = dict(item)
            source = "<mapping>"
        else:
            path = _resolve_path(item, base)
            if (path is None or not path.is_file()) and output != base:
                path = _resolve_path(item, output)
            if path is None:
                raise RoutingFigureError(f"{provenance} must be a local JSON path")
            if not path.is_file():
                raise RoutingFigureError(f"{provenance} JSON file does not exist: {path}")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RoutingFigureError(f"unable to parse {provenance} JSON: {path}") from exc
            source = str(path)
        if not isinstance(payload, Mapping):
            raise RoutingFigureError(f"{provenance} JSON must contain an object")
        case_id = _text(payload.get("case_id")).strip()
        if not case_id:
            raise RoutingFigureError(f"{provenance} is missing case_id")
        records.append((case_id, payload, source))
    return records


def _validated_turnover(
    turnover: Any,
    *,
    expected_case_ids: Sequence[str],
    ledger_path: Path | None,
    output: Path,
) -> dict[str, dict[str, Any]]:
    """Validate and normalize one run_turnover artifact per anchor case."""

    records = _turnover_source_items(turnover, ledger_path=ledger_path, output=output)
    if not records:
        return {}
    expected = {str(case) for case in expected_case_ids}
    normalized: dict[str, dict[str, Any]] = {}
    for case_id, payload, source in records:
        if case_id not in expected:
            raise RoutingFigureError(f"turnover artifact {source} has unexpected case_id {case_id!r}")
        if case_id in normalized:
            raise RoutingFigureError(f"multiple turnover artifacts supplied for case {case_id!r}")
        if _text(payload.get("task")) not in {"dynamic_sparse_routing_turnover", "routing_turnover"}:
            raise RoutingFigureError(f"turnover artifact {source} is not a run_turnover result")
        if _text(payload.get("status"), "ok") not in {"ok", "available", "complete"}:
            raise RoutingFigureError(f"turnover artifact {source} has non-success status {payload.get('status')!r}")
        for count_name in ("failure_count", "unavailable_count"):
            count = _finite_number(payload.get(count_name))
            if count is not None and count != 0.0:
                raise RoutingFigureError(f"turnover artifact {source} has {count_name}={count}")
        comparisons = payload.get("adjacent_comparisons")
        if not isinstance(comparisons, Sequence) or isinstance(comparisons, (str, bytes)) or not comparisons:
            raise RoutingFigureError(f"turnover artifact {source} has no adjacent_comparisons")
        pairs: list[dict[str, Any]] = []
        values: dict[str, dict[str, dict[str, list[float]]]] = {
            phase: {metric: {source_name: [] for source_name in _TURNOVER_SOURCES} for metric in _TURNOVER_METRICS}
            for phase in _TURNOVER_PHASES
        }
        for comparison_index, comparison in enumerate(comparisons):
            if not isinstance(comparison, Mapping):
                raise RoutingFigureError(f"{source} adjacent_comparisons[{comparison_index}] is not an object")
            before = comparison.get("from")
            after = comparison.get("to")
            if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                raise RoutingFigureError(f"{source} adjacent_comparisons[{comparison_index}] lacks from/to checkpoints")
            before_epoch = _finite_number(before.get("epoch"))
            after_epoch = _finite_number(after.get("epoch"))
            if before_epoch is None or after_epoch is None or after_epoch < before_epoch:
                raise RoutingFigureError(f"{source} adjacent_comparisons[{comparison_index}] has invalid epochs")
            before_label = _text(before.get("label"), str(int(before_epoch)))
            after_label = _text(after.get("label"), str(int(after_epoch)))
            pair = {
                "from_epoch": int(before_epoch),
                "to_epoch": int(after_epoch),
                "from_label": before_label,
                "to_label": after_label,
                "label": f"{before_label} → {after_label}",
            }
            phases = comparison.get("phases")
            if not isinstance(phases, Mapping):
                raise RoutingFigureError(f"{source} adjacent_comparisons[{comparison_index}] lacks phases")
            for phase in _TURNOVER_PHASES:
                phase_data = phases.get(phase)
                if not isinstance(phase_data, Mapping):
                    raise RoutingFigureError(f"{source} comparison {comparison_index} lacks phase {phase}")
                for metric in _TURNOVER_METRICS:
                    metric_data = phase_data.get(metric)
                    if not isinstance(metric_data, Mapping):
                        raise RoutingFigureError(f"{source} comparison {comparison_index} lacks {phase}.{metric}")
                    for source_name in _TURNOVER_SOURCES:
                        summary = metric_data.get(source_name)
                        if not isinstance(summary, Mapping):
                            raise RoutingFigureError(f"{source} comparison {comparison_index} lacks {phase}.{metric}.{source_name}")
                        jaccard = _finite_number(summary.get("jaccard"))
                        if jaccard is None or not 0.0 <= jaccard <= 1.0:
                            raise RoutingFigureError(f"{source} comparison {comparison_index} has invalid {phase}.{metric}.{source_name}.jaccard")
                        values[phase][metric][source_name].append(jaccard)
            pairs.append(pair)
        normalized[case_id] = {
            "source": source,
            "query_count": _finite_number(payload.get("query_count")),
            "pairs": pairs,
            "values": values,
            "interpretation": payload.get("interpretation", {}),
        }
    return normalized


def _render_turnover(path: Path, case_id: str, record: Mapping[str, Any]) -> None:
    _matplotlib, plt = _mpl()
    figure, axes = plt.subplots(3, 2, figsize=(13.4, 11.0), constrained_layout=True, squeeze=False)
    pairs = list(record.get("pairs", ()))
    labels = [str(pair.get("label", index + 1)) for index, pair in enumerate(pairs)]
    values = record.get("values", {})
    colors = {"module": "#315f7c", "environment": "#9aa0a6"}
    for row_index, phase in enumerate(_TURNOVER_PHASES):
        for column_index, metric in enumerate(_TURNOVER_METRICS):
            axis = axes[row_index, column_index]
            metric_values = values.get(phase, {}).get(metric, {})
            positions = np.arange(len(pairs), dtype=float)
            for source_name in _TURNOVER_SOURCES:
                series = np.asarray(metric_values.get(source_name, ()), dtype=float)
                if series.size != positions.size:
                    continue
                axis.plot(positions, series, marker="o", linewidth=1.2, markersize=4.0, color=colors[source_name], label=source_name)
            axis.set_ylim(-0.03, 1.03)
            axis.set_ylabel("Jaccard")
            axis.set_title(f"{phase}: {metric.replace('_', ' ')}")
            axis.set_xticks(positions, labels, rotation=35, ha="right")
            axis.grid(alpha=0.22)
            axis.legend(fontsize=8, frameon=True, loc="best")
    figure.suptitle(f"Support turnover by epoch pair · case {case_id}\nJaccard = intersection / union; 1 − Jaccard is changed fraction", fontsize=14)
    figure.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(figure)


def _relative_href(target: Any, base: Path) -> str:
    text = str(target)
    if "://" in text or text.startswith("#"):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    try:
        return str(path.relative_to(base.resolve()))
    except ValueError:
        try:
            return str(Path(os.path.relpath(path, base)))
        except (OSError, ValueError):
            return str(path)


_ENDPOINT_LINK_KEYS = ("endpoint_links", "endpoint_figures", "field_plot_links", "field_figures")


def _validated_link_target(value: Any, *, bases: Sequence[Path], context: str) -> str:
    """Validate one endpoint link as an existing file or an HTTP(S) URL."""

    if isinstance(value, (str, Path)) and not isinstance(value, bool):
        text = str(value).strip()
    else:
        text = ""
    if not text:
        raise RoutingFigureError(f"{context} must be a non-empty existing file path or HTTP(S) URL")
    parsed = urlparse(text)
    if parsed.scheme:
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return text
        raise RoutingFigureError(f"{context} has unsupported URL scheme: {text!r}")
    candidate = Path(text).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [base / candidate for base in bases]
    for path in candidates:
        try:
            if path.is_file():
                return str(path.resolve())
        except OSError:
            continue
    raise RoutingFigureError(f"{context} does not name an existing file or HTTP(S) URL: {text!r}")


def _endpoint_links(
    payload: Mapping[str, Any],
    explicit: Mapping[str, Any] | None,
    *,
    bases: Sequence[Path] = (),
) -> dict[str, dict[str, str]]:
    """Collect only explicit, validated endpoint figure links.

    Numeric ledger metrics often contain words such as ``field`` or
    ``endpoint`` in their names.  Row-level discovery therefore accepts only
    the four explicit link-field names above and never scans arbitrary row
    keys for substrings.
    """

    result: dict[str, dict[str, str]] = {}

    def consume(value: Any, *, context: str) -> None:
        if not isinstance(value, Mapping):
            raise RoutingFigureError(f"{context} must be a case-to-link mapping")
        for case, links in value.items():
            case_id = str(case)
            if isinstance(links, Mapping):
                for kind, target in links.items():
                    result.setdefault(case_id, {})[str(kind)] = _validated_link_target(
                        target,
                        bases=bases,
                        context=f"{context}[{case_id!r}][{kind!r}]",
                    )
            elif isinstance(links, (list, tuple)) and not isinstance(links, (str, bytes)):
                for index, target in enumerate(links):
                    result.setdefault(case_id, {})[f"field_{index + 1}"] = _validated_link_target(
                        target,
                        bases=bases,
                        context=f"{context}[{case_id!r}][{index}]",
                    )
            else:
                result.setdefault(case_id, {})["endpoint"] = _validated_link_target(
                    links,
                    bases=bases,
                    context=f"{context}[{case_id!r}]",
                )

    if explicit is not None:
        consume(explicit, context="explicit endpoint_links")
    for key in _ENDPOINT_LINK_KEYS:
        if key in payload:
            consume(payload[key], context=f"ledger.{key}")
    rows = payload.get("rows", ())
    for row_index, row in enumerate(rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else ()):
        if not isinstance(row, Mapping) or row.get("case_id") is None:
            continue
        case = str(row["case_id"])
        for key in _ENDPOINT_LINK_KEYS:
            if key in row:
                consume({case: row[key]}, context=f"ledger.rows[{row_index}].{key}")
    return result


def _write_index(path: Path, manifest: Mapping[str, Any]) -> None:
    links = manifest.get("endpoint_links", {})
    cards: list[str] = []
    figure_names = ("geometry", "incidence", "selected_pairs", "probabilities", "occupancy", "support_turnover")
    for anchor in manifest.get("anchors", ()):
        case = str(anchor.get("case_id", ""))
        figures = anchor.get("figures", {})
        body = [f'<section class="card"><h2>Anchor {html.escape(case)}</h2>']
        body.append('<div class="grid">')
        for name in figure_names:
            target = figures.get(name)
            if target:
                body.append(f'<figure><img src="{html.escape(str(target), quote=True)}" alt="{html.escape(name)} for case {html.escape(case)}"><figcaption>{html.escape(name.replace("_", " "))}</figcaption></figure>')
        body.append("</div>")
        case_links = links.get(case, {}) if isinstance(links, Mapping) else {}
        if case_links:
            body.append("<h3>Existing endpoint field figures</h3><ul>")
            for label, target in case_links.items():
                body.append(f'<li><a href="{html.escape(str(target), quote=True)}">{html.escape(str(label))}</a></li>')
            body.append("</ul>")
        else:
            body.append('<p class="muted">No endpoint field links were supplied; field arrays are intentionally not duplicated here.</p>')
        missing = anchor.get("validation", {}).get("missing", [])
        if missing:
            body.append(f'<p class="warning">Unavailable route metadata: {html.escape(", ".join(map(str, missing)))}</p>')
        body.append("</section>")
        cards.append("\n".join(body))
    support = manifest.get("support_transition_figure")
    support_block = f'<section class="card"><h2>Support transition</h2><img src="{html.escape(str(support), quote=True)}" alt="support transition trace"></section>' if support else ""
    text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HONF module-hub routing diagnostics</title>
<style>
:root {{ color-scheme: light; --ink:#23313d; --muted:#60717e; --line:#dce4e9; --page:#f4f7f9; --panel:#fff; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--page); color:var(--ink); font-family:Inter,Arial,sans-serif; line-height:1.4; }}
main {{ width:min(1500px,96vw); margin:auto; padding:28px 0 60px; }} header,.card {{ background:var(--panel); border:1px solid var(--line); padding:18px 20px; margin-bottom:18px; }}
h1 {{ margin:0 0 8px; font-size:26px; }} h2 {{ margin:0 0 12px; font-size:18px; }} h3 {{ font-size:14px; margin:14px 0 6px; }}
p {{ color:var(--muted); }} .notice {{ border-left:4px solid #315f7c; background:#eef5f8; padding:10px 13px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }} figure {{ margin:0; }} img {{ display:block; width:100%; height:auto; border:1px solid var(--line); background:white; }} figcaption {{ color:var(--muted); font-size:12px; padding:4px 2px; text-transform:capitalize; }}
.muted {{ color:var(--muted); }} .warning {{ color:#8b4513; background:#fff7ed; border-left:3px solid #bf6b3c; padding:7px 10px; }} ul {{ margin:6px 0 8px 20px; }} code,pre {{ font-size:12px; }}
</style></head><body><main>
<header><h1>Module-hub routing diagnostics</h1>
<p>Shared selected-anchor figures for ordinary source memberships, source-measure query probabilities, deduplicated two-hop priors, and sparse work occupancy.</p>
<div class="notice"><strong>Interpretation.</strong> Hub coordinates are routing descriptors. Route weights describe learned routing and are not physical influence or field values. Reference/prediction/error field plots remain endpoint-owned and are linked below without copying their arrays.</div>
<p>Status: <strong>{html.escape(str(manifest.get("status", "unknown")))}</strong> · anchors rendered: {html.escape(str(manifest.get("rendered_anchor_count", 0)))}</p></header>
{support_block}
{''.join(cards)}
<details class="card"><summary>Manifest and contract</summary><pre>{html.escape(json.dumps(_jsonable(manifest), indent=2, sort_keys=True))}</pre></details>
</main></body></html>
"""
    path.write_text(text, encoding="utf-8")


def render_routing_diagnostics(
    ledger: Mapping[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    endpoint_links: Mapping[str, Any] | None = None,
    turnover: Any | None = None,
    expected_case_ids: Sequence[str] = DEFAULT_ANCHOR_CASE_IDS,
    strict: bool = False,
) -> dict[str, Any]:
    """Render shared routing panels and return a JSON-safe manifest.

    ``ledger`` may be the JSON path emitted by ``ledger`` or a decoded
    mapping.  ``endpoint_links`` is a case-to-link mapping; links are written
    into the index only, so this helper never duplicates normal/reference/error
    endpoint arrays.  ``turnover`` accepts one or more run_turnover JSON paths
    (or decoded mappings), one artifact per anchor case.  In strict mode every
    expected anchor must have complete geometry, M/E incidence, P0/P2 selected
    maps, and pair metadata; supplied turnover inputs must cover every anchor.
    """

    payload, ledger_path = _load_ledger(ledger)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected = tuple(str(case) for case in expected_case_ids)
    link_bases = [output]
    if ledger_path is not None:
        link_bases.append(ledger_path.parent)
    # Validate external endpoint links before selecting or strict-checking a
    # route map.  The resulting links are reused verbatim in the index.
    endpoint = _endpoint_links(payload, endpoint_links, bases=tuple(link_bases))
    turnover_records = _validated_turnover(
        turnover,
        expected_case_ids=expected,
        ledger_path=ledger_path,
        output=output,
    )
    missing_turnover = [case for case in expected if case not in turnover_records]
    if strict and turnover is not None and missing_turnover:
        raise RoutingFigureError(f"turnover inputs are missing expected anchor cases: {missing_turnover}")
    candidates = _map_candidates(payload, ledger_path)
    rows_by_case: dict[str, Mapping[str, Any]] = {}
    rows = payload.get("rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if isinstance(row, Mapping) and row.get("case_id") is not None:
                rows_by_case[str(row["case_id"])] = row
    selected: dict[str, Path] = {}
    validation: dict[str, dict[str, Any]] = {}
    loaded: dict[str, dict[str, np.ndarray]] = {}
    inferred_metadata: dict[str, list[str]] = {}
    for case_id in expected:
        paths = candidates.get(case_id, [])
        if not paths:
            validation[case_id] = {"complete": False, "missing": ["routing_map"], "phase_missing": {}, "metadata_missing": []}
            continue
        # Prefer the candidate with the most complete explicit contract; keep
        # selection deterministic when several phase/map files are present.
        scored: list[tuple[int, str, Path, dict[str, np.ndarray], dict[str, Any]]] = []
        for path in paths:
            data = _load_npz(path)
            # The exporter stores scalar duplicate counts in the per-case
            # ledger row while the NPZ carries the actual pair IDs.  Merge
            # only those scalar contract fields; this keeps the route arrays
            # phase-specific and avoids copying endpoint field data.
            row = rows_by_case.get(case_id)
            if row is not None:
                for key, value in row.items():
                    token = str(key)
                    if (
                        token in ROUTING_MAP_METADATA
                        or token.endswith(("_raw_path_count", "_unique_pair_count", "_duplicate_count", "_duplicate_expansion"))
                    ) and token not in data:
                        data[token] = np.asarray(value)
                selection = row.get("routing_map_selection")
                if isinstance(selection, Mapping):
                    if "port_flat_receiver_index" in selection and "p0_selected_receiver" not in data:
                        data["p0_selected_receiver"] = np.asarray([selection["port_flat_receiver_index"]])
                    if "far_query_index" in selection and "p2_far_receiver_index" not in data:
                        data["p2_far_receiver_index"] = np.asarray([selection["far_query_index"]])
            inferred = _infer_metadata(data, payload, case_id)
            for key, value in inferred.items():
                data[key] = np.asarray(value)
            if inferred:
                inferred_metadata[case_id] = sorted(inferred)
            report = validate_routing_map(data, require_support_transition=turnover is None)
            score = int(report.get("complete", False)) * 1000 + sum(value is not None for value in _canonical_arrays(data).values())
            scored.append((score, str(path), path, data, report))
        _score, _name, path, data, report = min(scored, key=lambda item: (-item[0], item[1]))
        selected[case_id] = path
        loaded[case_id] = data
        validation[case_id] = report
    missing_cases = [case for case in expected if case not in selected]
    incomplete = {case: report for case, report in validation.items() if not report.get("complete", False)}
    if strict and (missing_cases or incomplete):
        details = {"missing_cases": missing_cases, "incomplete": incomplete}
        raise RoutingFigureError(f"routing figure contract is incomplete: {json.dumps(_jsonable(details), sort_keys=True)}")

    anchors: list[dict[str, Any]] = []
    support_traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case_id in expected:
        if case_id not in loaded:
            continue
        data = loaded[case_id]
        arrays = _canonical_arrays(data)
        phase_views = {
            phase: {source: _phase_view(data, phase, source) for source in ("module", "environment")}
            for phase in ("p0", "p2")
        }
        figure_paths = {
            "geometry": output / f"routing__{case_id}__geometry.png",
            "incidence": output / f"routing__{case_id}__incidence.png",
            "selected_pairs": output / f"routing__{case_id}__selected_pairs.png",
            "probabilities": output / f"routing__{case_id}__probabilities.png",
            "occupancy": output / f"routing__{case_id}__occupancy.png",
        }
        if arrays["module_coords"] is not None and arrays["env_coords"] is not None and arrays["hub_coords"] is not None:
            _render_geometry(figure_paths["geometry"], case_id, arrays, {f"{phase}_{source}": view for phase, source_views in phase_views.items() for source, view in source_views.items()})
        else:
            _placeholder(figure_paths["geometry"], f"Module-hub routing geometry · case {case_id}", validation[case_id].get("missing", []))
        _render_incidence(figure_paths["incidence"], case_id, arrays)
        _render_selected(figure_paths["selected_pairs"], case_id, arrays, phase_views)
        _render_probabilities(figure_paths["probabilities"], case_id, arrays, phase_views)
        _render_occupancy(figure_paths["occupancy"], case_id, arrays, phase_views, data)
        turnover_path = output / f"routing__{case_id}__support_turnover.png"
        turnover_record = turnover_records.get(case_id)
        if turnover_record is None:
            _placeholder(
                turnover_path,
                f"Support turnover by epoch pair · case {case_id}",
                ["No run_turnover artifact was supplied for this anchor."],
            )
        else:
            _render_turnover(turnover_path, case_id, turnover_record)
        trace = _support_trace(data)
        if trace is not None:
            support_traces[case_id] = trace
        anchors.append(
            {
                "case_id": case_id,
                "map": str(selected[case_id]),
                "metadata": {key: _text(data[key]) for key in ROUTING_MAP_METADATA if key in data},
                "metadata_inferred": inferred_metadata.get(case_id, []),
                "validation": validation[case_id],
                "derived": sorted({item for source_views in phase_views.values() for view in source_views.values() for item in view.get("derived", [])}),
                "figures": {
                    **{key: str(path.relative_to(output)) for key, path in figure_paths.items()},
                    "support_turnover": str(turnover_path.relative_to(output)),
                },
                "turnover": None
                if turnover_record is None
                else {
                    "source": turnover_record.get("source"),
                    "comparison_count": len(turnover_record.get("pairs", ())),
                    "query_count": turnover_record.get("query_count"),
                },
            }
        )

    support_path = output / "routing__support_transition.png"
    _render_support(support_path, support_traces)
    endpoint_relative = {
        case: {kind: _relative_href(target, output) for kind, target in links.items()}
        for case, links in endpoint.items()
    }
    turnover_incomplete = turnover is not None and bool(missing_turnover)
    status = "ok" if not missing_cases and not incomplete and not turnover_incomplete else "partial"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "render_routing_diagnostics",
        "status": status,
        "strict": bool(strict),
        "expected_case_ids": list(expected),
        "rendered_anchor_count": len(anchors),
        "missing_case_ids": missing_cases,
        "output_dir": str(output),
        "ledger": str(ledger_path) if ledger_path is not None else None,
        "anchors": anchors,
        "support_transition_figure": str(support_path.relative_to(output)),
        "support_transition_cases": sorted(support_traces),
        "support_transition_source": (
            "routing-map support-transition arrays"
            if support_traces
            else "external run_turnover epoch-pair Jaccard"
            if turnover_records
            else "unavailable"
        ),
        "turnover_cases": sorted(turnover_records),
        "turnover_missing_case_ids": missing_turnover if turnover is not None else [],
        "turnover_inputs": {
            case: {
                "source": record.get("source"),
                "comparison_count": len(record.get("pairs", ())),
                "query_count": record.get("query_count"),
            }
            for case, record in turnover_records.items()
        },
        "endpoint_links": endpoint_relative,
        "field_arrays_reused": True,
        "field_plot_policy": "existing endpoint reference/prediction/error figures are linked in index.html; arrays are not copied",
        "route_weight_semantics": ROUTING_WEIGHT_SEMANTICS,
        "metadata_inferred": inferred_metadata,
        "contract": {
            "canonical_keys": _jsonable(ROUTING_MAP_KEYS),
            "metadata_keys": list(ROUTING_MAP_METADATA),
            "optional_metadata_keys": list(ROUTING_MAP_OPTIONAL_METADATA),
            "validation": {case: report for case, report in validation.items()},
        },
    }
    manifest_path = output / "routing_manifest.json"
    index_path = output / "index.html"
    manifest["manifest"] = str(manifest_path)
    manifest["index"] = str(index_path)
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_index(index_path, manifest)
    return _jsonable(manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--endpoint-links", type=Path, default=None, help="optional JSON case-to-field-figure link mapping")
    parser.add_argument(
        "--turnover",
        type=Path,
        action="append",
        default=None,
        metavar="JSON",
        help="repeatable run_turnover JSON artifact, one case per file",
    )
    parser.add_argument("--case-id", action="append", default=None, help="expected anchor; defaults to the five prescribed anchors")
    parser.add_argument("--strict", action="store_true", help="fail if every expected anchor lacks the full route map contract")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    links: Mapping[str, Any] | None = None
    if args.endpoint_links is not None:
        links = json.loads(args.endpoint_links.read_text(encoding="utf-8"))
        if not isinstance(links, Mapping):
            raise SystemExit("--endpoint-links JSON must contain an object")
    manifest = render_routing_diagnostics(
        args.ledger,
        output_dir=args.figure_dir,
        endpoint_links=links,
        turnover=args.turnover,
        expected_case_ids=args.case_id or DEFAULT_ANCHOR_CASE_IDS,
        strict=args.strict,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ANCHOR_CASE_IDS",
    "ROUTING_MAP_KEYS",
    "ROUTING_MAP_METADATA",
    "ROUTING_MAP_OPTIONAL_METADATA",
    "ROUTING_WEIGHT_SEMANTICS",
    "RoutingFigureError",
    "build_parser",
    "main",
    "render_routing_diagnostics",
    "validate_routing_map",
]
