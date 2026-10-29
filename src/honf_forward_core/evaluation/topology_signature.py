"""Unordered, query-referenced HONF topology signatures.

The signature is a diagnostic/export object.  Canonical ordering is applied
only when serializing or displaying it; comparison matches active edges as a
set and never treats the serialized position as learned edge identity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_NAME = "honf_topology_signature"
SCHEMA_VERSION = 3
EPS = 1.0e-8

EDGE_FEATURE_NAMES = (
    "active_mask",
    "quality",
    "source_x",
    "source_y",
    "source_scale_x",
    "source_scale_y",
    "region_x",
    "region_y",
    "region_scale_x",
    "region_scale_y",
    "source_region_dx",
    "source_region_dy",
    "source_region_distance",
    "module_mass",
    "environment_mass",
    "module_assignment_purity",
    "environment_assignment_purity",
    "effective_module_count",
    "mean_query_routing",
    "rms_query_routing",
    "mean_channel_contribution_rms",
    "mean_channel_contribution_fraction",
)
QUERY_SUMMARY_NAMES = (
    "mean",
    "rms",
    "maximum",
    "nonzero_fraction",
)
CONTRIBUTION_SUMMARY_NAMES = (
    "signed_mean",
    "rms",
    "energy_fraction",
)
RELATION_FEATURE_NAMES = (
    "module_overlap",
    "environment_overlap",
    "query_corouting",
    "source_dx",
    "source_dy",
    "source_distance",
    "region_dx",
    "region_dy",
    "region_distance",
)


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _case_array(value: Any, rank: int, batch_index: int) -> np.ndarray:
    array = _numpy(value)
    if array.ndim == rank + 1:
        if not 0 <= int(batch_index) < array.shape[0]:
            raise IndexError(f"batch_index={batch_index} is outside batch size {array.shape[0]}.")
        array = array[int(batch_index)]
    if array.ndim != rank:
        raise ValueError(f"Expected rank {rank} (or batched rank {rank + 1}), got shape {array.shape}.")
    return array


def _optional_case_array(
    payload: Mapping[str, Any],
    names: Sequence[str],
    *,
    rank: int,
    batch_index: int,
    default: np.ndarray,
) -> np.ndarray:
    for name in names:
        if name in payload and payload[name] is not None:
            return _case_array(payload[name], rank, batch_index).astype(np.float64, copy=False)
    return np.asarray(default, dtype=np.float64)


def _decoder_value(payload: Mapping[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    routing = payload.get("routing_aux", {})
    if isinstance(routing, Mapping):
        return routing.get(key)
    return None


def _query_chunks(
    decoder_outputs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    key: str,
    *,
    rank: int,
    batch_index: int,
) -> list[np.ndarray]:
    if decoder_outputs is None:
        return []
    outputs = [decoder_outputs] if isinstance(decoder_outputs, Mapping) else list(decoder_outputs)
    chunks: list[np.ndarray] = []
    for output in outputs:
        value = _decoder_value(output, key)
        if value is not None:
            chunks.append(_case_array(value, rank, batch_index).astype(np.float64, copy=False))
    return chunks


def _minimum_image(delta: np.ndarray, domain_lengths: np.ndarray, periodic_axes: Sequence[int]) -> np.ndarray:
    result = np.asarray(delta, dtype=np.float64).copy()
    for axis in periodic_axes:
        length = float(domain_lengths[int(axis)])
        if length > 0:
            result[..., int(axis)] = (result[..., int(axis)] + 0.5 * length) % length - 0.5 * length
    return result


def _cosine_overlap(incidence: np.ndarray) -> np.ndarray:
    numerator = incidence.T @ incidence
    norms = np.sqrt(np.maximum(np.diag(numerator), 0.0))
    return numerator / np.maximum(norms[:, None] * norms[None, :], EPS)


def _string_array(values: Sequence[str]) -> np.ndarray:
    width = max((len(str(value)) for value in values), default=1)
    return np.asarray([str(value) for value in values], dtype=f"<U{width}")


def _reference_digest(reference_query_xy: Any | None, batch_index: int) -> tuple[int, str]:
    if reference_query_xy is None:
        return 0, ""
    query = _case_array(reference_query_xy, 2, batch_index).astype("<f4", copy=False)
    return int(query.shape[0]), hashlib.sha256(query.tobytes(order="C")).hexdigest()


def extract_topology_signature(
    organizer_output: Mapping[str, Any],
    module_present: Any,
    *,
    decoder_outputs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    reference_query_xy: Any | None = None,
    reference_measure: str = "unspecified",
    field_names: Sequence[str] | None = None,
    domain_length_x: float = 0.0,
    domain_length_y: float = 0.0,
    periodic_axes: Sequence[int] = (),
    case_id: str = "",
    forward_checkpoint_sha256: str = "",
    batch_index: int = 0,
    canonicalize: bool = True,
) -> dict[str, np.ndarray]:
    """Extract one design's static and reference-query topology signature."""

    module_incidence = _case_array(organizer_output["A_mh"], 2, batch_index).astype(np.float64)
    environment_incidence = _case_array(organizer_output["A_eh"], 2, batch_index).astype(np.float64)
    present = _case_array(module_present, 1, batch_index).astype(np.float64)
    if module_incidence.shape[0] != present.shape[0]:
        raise ValueError("module_present does not align with A_mh rows.")
    if module_incidence.shape[1] != environment_incidence.shape[1]:
        raise ValueError("A_mh and A_eh must share the candidate-edge axis.")
    capacity = int(module_incidence.shape[1])
    module_incidence = module_incidence * (present > 0.5)[:, None]

    active = _optional_case_array(
        organizer_output,
        ("edge_active_mask", "active_hyperedge_mask"),
        rank=1,
        batch_index=batch_index,
        default=np.ones(capacity),
    )
    if active.shape != (capacity,):
        raise ValueError("Active-edge mask does not align with the incidence edge axis.")
    active = (active > 0.5).astype(np.float64)
    module_incidence *= active[None, :]
    environment_incidence *= active[None, :]

    zeros_2 = np.zeros((capacity, 2), dtype=np.float64)
    zeros_1 = np.zeros(capacity, dtype=np.float64)
    source = _optional_case_array(
        organizer_output, ("hyper_source_coords",), rank=2, batch_index=batch_index, default=zeros_2
    )
    source_scale = _optional_case_array(
        organizer_output, ("hyper_source_scale",), rank=2, batch_index=batch_index, default=zeros_2
    )
    region = _optional_case_array(
        organizer_output,
        ("hyper_region_coords", "hyper_thermal_region_coords"),
        rank=2,
        batch_index=batch_index,
        default=zeros_2,
    )
    region_scale = _optional_case_array(
        organizer_output, ("hyper_region_scale",), rank=2, batch_index=batch_index, default=zeros_2
    )
    quality = _optional_case_array(
        organizer_output, ("edge_quality", "hyper_strength"), rank=1, batch_index=batch_index, default=zeros_1
    )
    module_mass = _optional_case_array(
        organizer_output, ("hyper_module_mass",), rank=1, batch_index=batch_index, default=zeros_1
    )
    environment_mass = _optional_case_array(
        organizer_output, ("hyper_env_mass",), rank=1, batch_index=batch_index, default=zeros_1
    )
    module_purity = _optional_case_array(
        organizer_output, ("hyper_module_purity",), rank=1, batch_index=batch_index, default=zeros_1
    )
    environment_purity = _optional_case_array(
        organizer_output, ("hyper_env_purity",), rank=1, batch_index=batch_index, default=zeros_1
    )

    route_chunks = _query_chunks(
        decoder_outputs, "query_hyper_attention", rank=2, batch_index=batch_index
    )
    routes = np.concatenate(route_chunks, axis=0) if route_chunks else np.zeros((0, capacity), dtype=np.float64)
    if routes.shape[1:] != (capacity,):
        raise ValueError("Query routing does not align with the incidence edge axis.")
    routes *= active[None, :]
    if routes.shape[0]:
        query_summary = np.stack(
            (
                routes.mean(axis=0),
                np.sqrt(np.mean(np.square(routes), axis=0)),
                routes.max(axis=0),
                (routes > 0).mean(axis=0),
            ),
            axis=-1,
        )
        query_corouting = routes.T @ routes / float(routes.shape[0])
    else:
        query_summary = np.zeros((capacity, len(QUERY_SUMMARY_NAMES)), dtype=np.float64)
        query_corouting = np.zeros((capacity, capacity), dtype=np.float64)

    contribution_chunks = _query_chunks(
        decoder_outputs, "pred_field_by_edge", rank=3, batch_index=batch_index
    )
    if contribution_chunks:
        contributions = np.concatenate(contribution_chunks, axis=0)
        if contributions.shape[1] != capacity:
            raise ValueError("Per-edge fields do not align with the incidence edge axis.")
        contributions *= active[None, :, None]
        signed_mean = contributions.mean(axis=0)
        squared_energy = np.square(contributions).sum(axis=0)
        rms = np.sqrt(np.mean(np.square(contributions), axis=0))
        fraction = squared_energy / np.maximum(squared_energy.sum(axis=0, keepdims=True), EPS)
        field_summary = np.stack((signed_mean, rms, fraction), axis=-1)
        num_fields = int(contributions.shape[-1])
        contribution_available = 1
    else:
        num_fields = len(field_names or ())
        field_summary = np.zeros(
            (capacity, num_fields, len(CONTRIBUTION_SUMMARY_NAMES)), dtype=np.float64
        )
        contribution_available = 0
    names = list(field_names or [f"field_{index}" for index in range(num_fields)])
    if len(names) != num_fields:
        raise ValueError("field_names length does not match the per-edge field width.")

    source_weights = module_incidence / np.maximum(module_incidence.sum(axis=0, keepdims=True), EPS)
    effective_modules = 1.0 / np.maximum(np.square(source_weights).sum(axis=0), EPS)
    effective_modules *= (module_incidence.sum(axis=0) > EPS) * active
    domain_lengths = np.asarray([float(domain_length_x), float(domain_length_y)], dtype=np.float64)
    displacement = _minimum_image(region - source, domain_lengths, periodic_axes)
    source_region_distance = np.linalg.norm(displacement, axis=-1)
    if num_fields:
        mean_contribution_rms = field_summary[:, :, 1].mean(axis=-1)
        mean_contribution_fraction = field_summary[:, :, 2].mean(axis=-1)
    else:
        mean_contribution_rms = zeros_1
        mean_contribution_fraction = zeros_1
    edge_features = np.column_stack(
        (
            active,
            quality,
            source,
            source_scale,
            region,
            region_scale,
            displacement,
            source_region_distance,
            module_mass,
            environment_mass,
            module_purity,
            environment_purity,
            effective_modules,
            query_summary[:, 0],
            query_summary[:, 1],
            mean_contribution_rms,
            mean_contribution_fraction,
        )
    )
    edge_features *= active[:, None]

    source_pair = _minimum_image(source[None, :, :] - source[:, None, :], domain_lengths, periodic_axes)
    region_pair = _minimum_image(region[None, :, :] - region[:, None, :], domain_lengths, periodic_axes)
    relations = np.stack(
        (
            _cosine_overlap(module_incidence),
            _cosine_overlap(environment_incidence),
            query_corouting,
            source_pair[..., 0],
            source_pair[..., 1],
            np.linalg.norm(source_pair, axis=-1),
            region_pair[..., 0],
            region_pair[..., 1],
            np.linalg.norm(region_pair, axis=-1),
        ),
        axis=-1,
    )
    relations *= active[:, None, None] * active[None, :, None]

    candidate_module = organizer_output.get("candidate_A_mh")
    candidate_environment = organizer_output.get("candidate_A_eh")
    reference_count, reference_digest = _reference_digest(reference_query_xy, batch_index)
    if reference_count == 0 and routes.shape[0]:
        reference_count = int(routes.shape[0])
    signature: dict[str, np.ndarray] = {
        "schema_name": np.asarray(SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "edge_feature_names": _string_array(EDGE_FEATURE_NAMES),
        "relation_feature_names": _string_array(RELATION_FEATURE_NAMES),
        "query_summary_names": _string_array(QUERY_SUMMARY_NAMES),
        "contribution_summary_names": _string_array(CONTRIBUTION_SUMMARY_NAMES),
        "field_names": _string_array(names),
        "edge_mask": active.astype(np.float32),
        "edge_features": edge_features.astype(np.float32),
        "edge_relations": relations.astype(np.float32),
        "module_incidence": module_incidence.astype(np.float32),
        "environment_incidence": environment_incidence.astype(np.float32),
        "module_present": (present > 0.5).astype(np.float32),
        "query_route_summary": query_summary.astype(np.float32),
        "field_contribution_summary": field_summary.astype(np.float32),
        "num_module_slots": np.asarray(present.shape[0], dtype=np.int32),
        "active_module_count": np.asarray((present > 0.5).sum(), dtype=np.int32),
        "candidate_edge_count": np.asarray(capacity, dtype=np.int32),
        "active_edge_count": np.asarray(active.sum(), dtype=np.int32),
        "reference_query_count": np.asarray(reference_count, dtype=np.int32),
        "reference_query_digest": np.asarray(reference_digest),
        "reference_measure": np.asarray(str(reference_measure)),
        "domain_lengths": domain_lengths.astype(np.float32),
        "periodic_axes": np.asarray(tuple(int(axis) for axis in periodic_axes), dtype=np.int32),
        "field_contribution_available": np.asarray(contribution_available, dtype=np.int32),
        "case_id": np.asarray(str(case_id)),
        "forward_checkpoint_sha256": np.asarray(str(forward_checkpoint_sha256)),
        "serialization_permutation": np.arange(capacity, dtype=np.int64),
    }
    if candidate_module is not None:
        signature["candidate_module_incidence"] = _case_array(
            candidate_module, 2, batch_index
        ).astype(np.float32)
    if candidate_environment is not None:
        signature["candidate_environment_incidence"] = _case_array(
            candidate_environment, 2, batch_index
        ).astype(np.float32)
    signature = canonicalize_topology_signature(signature) if canonicalize else signature
    validate_topology_signature(signature)
    return signature


def _canonical_permutation(signature: Mapping[str, Any]) -> np.ndarray:
    mask = np.asarray(signature["edge_mask"], dtype=np.float64)
    feature = np.asarray(signature["edge_features"], dtype=np.float64)
    module = np.asarray(signature["module_incidence"], dtype=np.float64)
    environment = np.asarray(signature["environment_incidence"], dtype=np.float64)
    query = np.asarray(signature["query_route_summary"], dtype=np.float64)
    contribution = np.asarray(signature["field_contribution_summary"], dtype=np.float64)
    candidate_module = np.asarray(signature.get("candidate_module_incidence", module), dtype=np.float64)
    candidate_environment = np.asarray(
        signature.get("candidate_environment_incidence", environment), dtype=np.float64
    )
    rows = []
    for index in range(mask.shape[0]):
        fingerprint = np.concatenate(
            (
                feature[index],
                module[:, index],
                environment[:, index],
                candidate_module[:, index],
                candidate_environment[:, index],
                query[index],
                contribution[index].reshape(-1),
            )
        )
        rows.append((0 if mask[index] > 0.5 else 1, *fingerprint.tolist(), index))
    return np.asarray(sorted(range(mask.shape[0]), key=rows.__getitem__), dtype=np.int64)


def canonicalize_topology_signature(signature: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return a deterministic serialization copy and retain its source permutation."""

    result = {key: np.asarray(value).copy() for key, value in signature.items()}
    permutation = _canonical_permutation(result)
    for key in ("edge_mask", "edge_features", "query_route_summary", "field_contribution_summary"):
        result[key] = result[key][permutation]
    for key in (
        "module_incidence",
        "environment_incidence",
        "candidate_module_incidence",
        "candidate_environment_incidence",
    ):
        if key in result:
            result[key] = result[key][:, permutation]
    result["edge_relations"] = result["edge_relations"][permutation][:, permutation]
    prior = np.asarray(
        result.get("serialization_permutation", np.arange(permutation.shape[0])), dtype=np.int64
    )
    result["serialization_permutation"] = prior[permutation]
    return result


def save_topology_signature(path: str | Path, signature: Mapping[str, Any]) -> None:
    """Validate and save a canonically ordered compressed NPZ signature."""

    canonical = canonicalize_topology_signature(signature)
    validate_topology_signature(canonical)
    np.savez_compressed(path, **canonical)


def load_topology_signature(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a topology signature without pickle support."""

    with np.load(path, allow_pickle=False) as payload:
        signature = {key: payload[key] for key in payload.files}
    validate_topology_signature(signature)
    return signature


def validate_topology_signature(signature: Mapping[str, Any]) -> None:
    """Validate schema identity, dimensions, counts, masks, and finite values."""

    required = {
        "schema_name",
        "schema_version",
        "edge_feature_names",
        "relation_feature_names",
        "query_summary_names",
        "contribution_summary_names",
        "field_names",
        "edge_mask",
        "edge_features",
        "edge_relations",
        "module_incidence",
        "environment_incidence",
        "module_present",
        "query_route_summary",
        "field_contribution_summary",
        "num_module_slots",
        "active_module_count",
        "candidate_edge_count",
        "active_edge_count",
        "serialization_permutation",
    }
    missing = sorted(required - set(signature))
    if missing:
        raise ValueError(f"Topology signature missing keys: {missing}")
    if str(np.asarray(signature["schema_name"]).item()) != SCHEMA_NAME:
        raise ValueError("Topology signature schema_name is not supported.")
    if int(np.asarray(signature["schema_version"])) != SCHEMA_VERSION:
        raise ValueError("Topology signature schema_version is not supported.")
    mask = np.asarray(signature["edge_mask"])
    capacity = int(np.asarray(signature["candidate_edge_count"]))
    if mask.shape != (capacity,) or not np.isin(mask, [0, 1]).all():
        raise ValueError("edge_mask must be binary with candidate_edge_count entries.")
    feature = np.asarray(signature["edge_features"])
    relations = np.asarray(signature["edge_relations"])
    module = np.asarray(signature["module_incidence"])
    environment = np.asarray(signature["environment_incidence"])
    query = np.asarray(signature["query_route_summary"])
    contribution = np.asarray(signature["field_contribution_summary"])
    if feature.shape != (capacity, len(np.asarray(signature["edge_feature_names"]))):
        raise ValueError("edge_features shape does not match its names or edge capacity.")
    if relations.shape != (capacity, capacity, len(np.asarray(signature["relation_feature_names"]))):
        raise ValueError("edge_relations shape does not match its names or edge capacity.")
    if module.ndim != 2 or module.shape[1] != capacity:
        raise ValueError("module_incidence must have shape [M,K_cap].")
    if environment.ndim != 2 or environment.shape[1] != capacity:
        raise ValueError("environment_incidence must have shape [E,K_cap].")
    if query.shape != (capacity, len(np.asarray(signature["query_summary_names"]))):
        raise ValueError("query_route_summary has an invalid shape.")
    if contribution.ndim != 3 or contribution.shape[0] != capacity:
        raise ValueError("field_contribution_summary must have shape [K_cap,F,summary].")
    if contribution.shape[1] != len(np.asarray(signature["field_names"])):
        raise ValueError("field_contribution_summary does not match field_names.")
    if contribution.shape[2] != len(np.asarray(signature["contribution_summary_names"])):
        raise ValueError("field_contribution_summary does not match its summary names.")
    present = np.asarray(signature["module_present"])
    if present.shape != (module.shape[0],):
        raise ValueError("module_present does not align with module_incidence.")
    if int(np.asarray(signature["num_module_slots"])) != module.shape[0]:
        raise ValueError("num_module_slots does not match module_incidence.")
    if int(np.asarray(signature["active_module_count"])) != int((present > 0.5).sum()):
        raise ValueError("active_module_count does not match module_present.")
    if int(np.asarray(signature["active_edge_count"])) != int(mask.sum()):
        raise ValueError("active_edge_count does not match edge_mask.")
    permutation = np.asarray(signature["serialization_permutation"])
    if permutation.shape != (capacity,) or sorted(permutation.tolist()) != list(range(capacity)):
        raise ValueError("serialization_permutation must cover every candidate edge exactly once.")
    for key, value in signature.items():
        array = np.asarray(value)
        if array.dtype.kind in "fiu" and not np.isfinite(array).all():
            raise ValueError(f"Topology signature key {key!r} contains non-finite values.")
    inactive = mask <= 0.5
    if inactive.any():
        if not np.allclose(module[:, inactive], 0.0, atol=1.0e-6):
            raise ValueError("Inactive edges must have zero module incidence.")
        if not np.allclose(environment[:, inactive], 0.0, atol=1.0e-6):
            raise ValueError("Inactive edges must have zero environment incidence.")


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """Return the minimum-cost column for each row of a finite square matrix."""

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Hungarian matching requires a square cost matrix.")
    size = matrix.shape[0]
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=np.int64)
    way = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        minimum = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = np.full(size, -1, dtype=np.int64)
    for column in range(1, size + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    return assignment


def compare_topology_signatures(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    relation_weight: float = 0.25,
    unmatched_cost: float = 1.0,
) -> dict[str, Any]:
    """Compare active edge sets with Hungarian token matching and relation error."""

    validate_topology_signature(first)
    validate_topology_signature(second)
    first_indices = np.flatnonzero(np.asarray(first["edge_mask"]) > 0.5)
    second_indices = np.flatnonzero(np.asarray(second["edge_mask"]) > 0.5)
    first_tokens = _comparison_tokens(first, first_indices)
    second_tokens = _comparison_tokens(second, second_indices)
    size = max(len(first_indices), len(second_indices))
    if size == 0:
        return {
            "topology_distance": 0.0,
            "matched_feature_cost": 0.0,
            "relation_cost": 0.0,
            "unmatched_edge_count": 0,
            "matches": [],
        }
    scale = np.maximum(
        np.maximum(
            np.max(np.abs(first_tokens), axis=0) if len(first_tokens) else 0.0,
            np.max(np.abs(second_tokens), axis=0) if len(second_tokens) else 0.0,
        ),
        1.0,
    )
    pair_cost = np.mean(np.square(first_tokens[:, None, :] / scale - second_tokens[None, :, :] / scale), axis=-1)
    padded = np.full((size, size), float(unmatched_cost), dtype=np.float64)
    if pair_cost.size:
        padded[: len(first_indices), : len(second_indices)] = pair_cost
    if len(first_indices) < size and len(second_indices) < size:
        padded[len(first_indices) :, len(second_indices) :] = 0.0
    assignment = _hungarian(padded)
    matched_local = [
        (row, int(column))
        for row, column in enumerate(assignment[: len(first_indices)])
        if column < len(second_indices)
    ]
    feature_sum = sum(float(pair_cost[row, column]) for row, column in matched_local)
    unmatched = size - len(matched_local)
    matched_feature_cost = (feature_sum + float(unmatched_cost) * unmatched) / float(size)
    if matched_local:
        rows = np.asarray([row for row, _ in matched_local], dtype=np.int64)
        columns = np.asarray([column for _, column in matched_local], dtype=np.int64)
        first_relations = np.asarray(first["edge_relations"], dtype=np.float64)[first_indices[rows]][:, first_indices[rows]]
        second_relations = np.asarray(second["edge_relations"], dtype=np.float64)[second_indices[columns]][:, second_indices[columns]]
        relation_scale = np.maximum(
            np.maximum(np.abs(first_relations).max(axis=(0, 1)), np.abs(second_relations).max(axis=(0, 1))),
            1.0,
        )
        relation_cost = float(np.mean(np.square(first_relations / relation_scale - second_relations / relation_scale)))
    else:
        relation_cost = 0.0
    return {
        "topology_distance": float(matched_feature_cost + float(relation_weight) * relation_cost),
        "matched_feature_cost": float(matched_feature_cost),
        "relation_cost": relation_cost,
        "unmatched_edge_count": int(unmatched),
        "matches": [
            {
                "first_edge": int(first_indices[row]),
                "second_edge": int(second_indices[column]),
                "feature_cost": float(pair_cost[row, column]),
            }
            for row, column in matched_local
        ],
    }


def _comparison_tokens(signature: Mapping[str, Any], indices: np.ndarray) -> np.ndarray:
    edge = np.asarray(signature["edge_features"], dtype=np.float64)[indices]
    query = np.asarray(signature["query_route_summary"], dtype=np.float64)[indices]
    contribution = np.asarray(signature["field_contribution_summary"], dtype=np.float64)[indices]
    return np.concatenate((edge, query, contribution.reshape(len(indices), -1)), axis=-1)


def reconstruct_module_affinity(module_incidence: Any, edge_mask: Any) -> np.ndarray:
    """Reconstruct an unordered module relation from selected incidence."""

    incidence = np.asarray(module_incidence, dtype=np.float64)
    active = np.asarray(edge_mask, dtype=np.float64).reshape(-1)
    selected = incidence * active[None, :]
    return selected @ (selected / np.maximum(selected.sum(axis=0, keepdims=True), EPS)).T


def reconstruct_environment_module_influence(
    query_hyper_attention: Any,
    module_incidence: Any,
    edge_mask: Any,
) -> np.ndarray:
    """Reconstruct query-to-module influence without assigning edge labels."""

    attention = np.asarray(query_hyper_attention, dtype=np.float64)
    incidence = np.asarray(module_incidence, dtype=np.float64)
    active = np.asarray(edge_mask, dtype=np.float64).reshape(-1)
    normalized = incidence * active[None, :]
    normalized = normalized / np.maximum(normalized.sum(axis=0, keepdims=True), EPS)
    return attention @ normalized.T


def evaluate_structure_relations(
    signature: Mapping[str, Any],
    *,
    module_affinity_target: Any | None = None,
    module_affinity_mask: Any | None = None,
    query_hyper_attention: Any | None = None,
    environment_module_target: Any | None = None,
    environment_module_mask: Any | None = None,
    active_edge_count_target: Any | None = None,
    has_solved_targets: bool = False,
) -> dict[str, Any]:
    """Evaluate optional case-owned relation targets without edge supervision."""

    validate_topology_signature(signature)
    result: dict[str, Any] = {
        "target_source": "solved" if has_solved_targets else "fallback",
        "has_solved_targets": bool(has_solved_targets),
    }
    incidence = np.asarray(signature["module_incidence"], dtype=np.float64)
    edge_mask = np.asarray(signature["edge_mask"], dtype=np.float64)
    if module_affinity_target is not None:
        prediction = reconstruct_module_affinity(incidence, edge_mask)
        result.update(_masked_errors("module_affinity", prediction, module_affinity_target, module_affinity_mask))
    if query_hyper_attention is not None and environment_module_target is not None:
        prediction = reconstruct_environment_module_influence(
            query_hyper_attention, incidence, edge_mask
        )
        result.update(
            _masked_errors(
                "environment_module_influence",
                prediction,
                environment_module_target,
                environment_module_mask,
            )
        )
    if active_edge_count_target is not None:
        target = float(np.asarray(active_edge_count_target).reshape(-1)[0])
        prediction = float(np.asarray(signature["active_edge_count"]))
        result["active_edge_count_absolute_error"] = abs(prediction - target)
    return result


def _masked_errors(name: str, prediction: Any, target: Any, mask: Any | None) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    if predicted.shape != expected.shape:
        raise ValueError(f"{name} prediction shape {predicted.shape} does not match target {expected.shape}.")
    weights = np.ones_like(expected) if mask is None else np.asarray(mask, dtype=np.float64)
    if weights.shape != expected.shape:
        weights = np.broadcast_to(weights, expected.shape)
    valid = weights > 0.5
    if not valid.any():
        return {f"{name}_mse": 0.0, f"{name}_relative_l2": 0.0}
    difference = predicted[valid] - expected[valid]
    return {
        f"{name}_mse": float(np.mean(np.square(difference))),
        f"{name}_relative_l2": float(
            np.linalg.norm(difference) / max(float(np.linalg.norm(expected[valid])), EPS)
        ),
    }


def summarize_topology_signature(signature: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-compatible schema and sparsity summary."""

    validate_topology_signature(signature)
    module = np.asarray(signature["module_incidence"])
    environment = np.asarray(signature["environment_incidence"])
    query = np.asarray(signature["query_route_summary"])
    return {
        "schema_name": str(np.asarray(signature["schema_name"]).item()),
        "schema_version": int(np.asarray(signature["schema_version"])),
        "case_id": str(np.asarray(signature.get("case_id", "")).item()),
        "num_module_slots": int(np.asarray(signature["num_module_slots"])),
        "active_module_count": int(np.asarray(signature["active_module_count"])),
        "candidate_edge_count": int(np.asarray(signature["candidate_edge_count"])),
        "active_edge_count": int(np.asarray(signature["active_edge_count"])),
        "reference_measure": str(np.asarray(signature.get("reference_measure", "")).item()),
        "reference_query_count": int(np.asarray(signature.get("reference_query_count", 0))),
        "reference_query_digest": str(np.asarray(signature.get("reference_query_digest", "")).item()),
        "field_contribution_available": bool(int(np.asarray(signature.get("field_contribution_available", 0)))),
        "module_incidence_nonzero_fraction": float(np.mean(module > 0)) if module.size else 0.0,
        "environment_incidence_nonzero_fraction": float(np.mean(environment > 0)) if environment.size else 0.0,
        "mean_query_routing_nonzero_fraction": float(query[:, 3].mean()) if query.size else 0.0,
        "field_names": [str(value) for value in np.asarray(signature["field_names"]).tolist()],
        "serialization_permutation": np.asarray(signature["serialization_permutation"], dtype=np.int64).tolist(),
    }
