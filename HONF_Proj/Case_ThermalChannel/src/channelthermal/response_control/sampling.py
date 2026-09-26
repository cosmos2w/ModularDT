"""Deterministic, role-aware receiver sampling for fixed response panels."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from channelthermal.interaction_evidence.response_dataset import ResponseStencil
from channelthermal.interaction_evidence.types import EvidenceSplit, RoleOutput, SolveRecord

from .thermal import (
    NEAR_INTERFACE_DISTANCE,
    PRESSURE_INLET_BAND_FRACTION,
    PRESSURE_OUTLET_BAND_FRACTION,
)


@dataclass(frozen=True)
class SamplingSummary:
    """Stored query budget and protected receiver coverage for one family."""

    physical_family_id: str
    split: str
    original_counts: dict[str, int]
    sampled_counts: dict[str, int]
    protected_counts: dict[str, int]
    solid_peak_query_coverage: dict[str, bool]
    inverse_probability_weighting: bool

    def __post_init__(self) -> None:
        for name in ("original_counts", "sampled_counts", "protected_counts"):
            object.__setattr__(self, name, MappingProxyType({str(key): int(value) for key, value in getattr(self, name).items()}))
        object.__setattr__(
            self,
            "solid_peak_query_coverage",
            MappingProxyType({str(key): bool(value) for key, value in self.solid_peak_query_coverage.items()}),
        )


@dataclass(frozen=True)
class SampledResponseStencil:
    stencil: ResponseStencil
    summary: SamplingSummary


@dataclass(frozen=True)
class ReceiverSamplingConfig:
    """Per-role query budgets; pressure, near-interface, and ports are protected.

    ``max_fluid_queries`` is a target budget for ordinary fluid rows. Protected
    rows are always retained, so the final fluid count can exceed it when the
    protected union is larger.
    """

    max_fluid_queries: int = 3072
    solid_queries_per_module: int = 128
    hot_solid_points_per_module: int = 16
    random_seed: int = 2317

    def __post_init__(self) -> None:
        if min(self.max_fluid_queries, self.solid_queries_per_module) <= 0:
            raise ValueError("Receiver query budgets must be positive.")
        if self.hot_solid_points_per_module < 0:
            raise ValueError("hot_solid_points_per_module must be nonnegative.")


def _sample_rows(
    n_rows: int,
    protected: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    protected = np.unique(np.asarray(protected, dtype=np.int64))
    if np.any((protected < 0) | (protected >= n_rows)):
        raise ValueError("Protected receiver index is outside the role array.")
    other = np.setdiff1d(np.arange(n_rows, dtype=np.int64), protected, assume_unique=True)
    sample_count = min(int(other.size), max(0, int(budget) - int(protected.size)))
    sampled_other = (
        np.sort(rng.choice(other, size=sample_count, replace=False))
        if sample_count
        else np.zeros((0,), dtype=np.int64)
    )
    indices = np.sort(np.concatenate([protected, sampled_other]))
    inclusion_probability = np.ones((indices.size,), dtype=np.float64)
    if sample_count and other.size:
        selected_other = np.isin(indices, sampled_other, assume_unique=True)
        inclusion_probability[selected_other] = sample_count / float(other.size)
    return indices, inclusion_probability, int(protected.size)


def _role_channels(role: RoleOutput) -> tuple[int, ...]:
    if "temperature" in role.channel_names:
        return (role.channel_names.index("temperature"),)
    if len(role.channel_names) == 1:
        return (0,)
    raise ValueError(f"Role {role.role!r} has no temperature channel for hot-point protection.")


def _protected_solid_local_indices(stencil: ResponseStencil, per_module_count: int) -> np.ndarray:
    baseline = stencil.baseline.output.roles["solid_temperature"]  # type: ignore[union-attr]
    if baseline.receiver_module_ids is None:
        raise ValueError("Solid material receivers must declare their module IDs.")
    modules = tuple(dict.fromkeys(baseline.receiver_module_ids))
    temperature_channel = _role_channels(baseline)[0]
    local_features = baseline.query_features[
        [index for index, current in enumerate(baseline.receiver_module_ids) if current == modules[0]], :2
    ]
    protected: set[int] = set()
    for module_id in modules:
        rows = np.asarray(
            [index for index, current in enumerate(baseline.receiver_module_ids) if current == module_id],
            dtype=np.int64,
        )
        features = baseline.query_features[rows, :2]
        if features.shape != local_features.shape or not np.allclose(features, local_features, atol=1.0e-7, rtol=0.0):
            raise ValueError("Solid training roles must share one material-local query grid across modules.")
        for record in stencil.records:
            role = record.output.roles["solid_temperature"]  # type: ignore[union-attr]
            if role.receiver_module_ids != baseline.receiver_module_ids:
                raise ValueError("Solid receiver module ordering changed within the stencil.")
            valid = role.valid_mask[rows, temperature_channel]
            candidates = np.flatnonzero(valid)
            keep_count = min(per_module_count, candidates.size)
            if keep_count:
                scores = role.values[rows[candidates], temperature_channel]
                order = candidates[np.argsort(scores, kind="stable")[-keep_count:]]
                protected.update(int(value) for value in order)
    return np.asarray(sorted(protected), dtype=np.int64)


def _protected_near_interface_fluid_indices(stencil: ResponseStencil) -> np.ndarray:
    """Union all valid near-interface Eulerian rows across the stencil states."""

    baseline = stencil.baseline.output.roles["fluid_fields"]  # type: ignore[union-attr]
    if baseline.coordinate_kind != "eulerian" or baseline.query_features.shape[1] < 2:
        raise ValueError("Near-interface fluid protection requires Eulerian x/y coordinates.")
    baseline_ids = baseline.query_ids
    baseline_xy = np.asarray(baseline.query_features[:, :2], dtype=np.float64)
    protected: set[int] = set()
    for record in stencil.records:
        if record.output is None:
            raise ValueError("Near-interface protection requires targets for valid-row masks.")
        role = record.output.roles["fluid_fields"]
        if role.coordinate_kind != "eulerian":
            raise ValueError("Near-interface fluid protection requires Eulerian coordinates.")
        if role.query_ids != baseline_ids or not np.allclose(
            role.query_features[:, :2], baseline_xy, atol=1.0e-7, rtol=0.0
        ):
            raise ValueError("Fluid stencil states must share the same Eulerian query rows and order.")
        try:
            radius = float(record.context.values["module_radius"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Near-interface fluid protection requires module_radius context.") from exc
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("Near-interface module radius must be positive and finite.")
        centers = np.asarray(
            [module.position_xy for module in record.design.modules if module.active],
            dtype=np.float64,
        )
        if centers.size == 0:
            raise ValueError("Near-interface fluid protection needs an active module.")
        surface_distance = np.linalg.norm(
            baseline_xy[:, None, :] - centers[None, :, :], axis=-1
        ).min(axis=1) - radius
        near = (surface_distance >= 0.0) & (surface_distance <= NEAR_INTERFACE_DISTANCE)
        valid = np.asarray(role.valid_mask, dtype=bool).any(axis=1)
        protected.update(np.flatnonzero(near & valid).tolist())
    return np.asarray(sorted(protected), dtype=np.int64)


def _subsample_role(
    role: RoleOutput,
    indices: np.ndarray,
    inclusion_probability: np.ndarray,
) -> RoleOutput:
    if indices.shape != inclusion_probability.shape or np.any(inclusion_probability <= 0.0):
        raise ValueError("Receiver sampler produced invalid inclusion probabilities.")
    weights = role.quadrature_weights[indices] / inclusion_probability
    floor = None if role.noise_floor is None else role.noise_floor[indices]
    return RoleOutput(
        role=role.role,
        query_features=role.query_features[indices],
        values=role.values[indices],
        channel_names=role.channel_names,
        channel_units=role.channel_units,
        valid_mask=role.valid_mask[indices],
        quadrature_weights=weights,
        query_ids=tuple(role.query_ids[index] for index in indices),
        receiver_module_ids=(
            None
            if role.receiver_module_ids is None
            else tuple(role.receiver_module_ids[index] for index in indices)
        ),
        coordinate_kind=role.coordinate_kind,
        noise_floor=floor,
    )


def _replace_record_roles(record: SolveRecord, roles: dict[str, RoleOutput]) -> SolveRecord:
    if record.output is None:
        raise ValueError("Receiver sampling requires converged physical outputs.")
    output = replace(record.output, roles=roles)
    return replace(record, output=output)


def sample_training_stencil(
    stencil: ResponseStencil,
    *,
    config: ReceiverSamplingConfig | None = None,
) -> SampledResponseStencil:
    """Apply one family-shared sample so all finite/mixed rows stay aligned.

    Inlet/outlet rows use the maintained exact 8% pressure bands, and the
    union of valid fluid rows within 0.25 of any module surface over every
    state is protected. Interface ports are retained in full. Solid points
    include each module's highest target temperature rows across every train
    stencil state plus deterministic inverse-probability-weighted samples for
    the remaining material mesh. Validation stencils remain full-resolution.
    """

    if stencil.split is not EvidenceSplit.TRAIN:
        raise ValueError("Only train-split stencils may be sampled for fitting.")
    config = ReceiverSamplingConfig() if config is None else config
    family_suffix = zlib.crc32(stencil.physical_family_id.encode("utf-8"))
    rng = np.random.default_rng(int(config.random_seed) ^ int(family_suffix))
    roles = stencil.baseline.output.roles  # type: ignore[union-attr]
    fluid = roles["fluid_fields"]
    context = stencil.baseline.context.values
    length_x = float(context["domain_length_x"])
    if fluid.coordinate_kind != "eulerian":
        raise ValueError("Fluid training receivers must be Eulerian.")
    x = fluid.query_features[:, 0]
    protected_fluid = np.flatnonzero(
        (x <= PRESSURE_INLET_BAND_FRACTION * length_x)
        | (x >= (1.0 - PRESSURE_OUTLET_BAND_FRACTION) * length_x)
    )
    protected_near_interface = _protected_near_interface_fluid_indices(stencil)
    protected_fluid_union = np.union1d(protected_fluid, protected_near_interface)
    fluid_indices, fluid_probability, fluid_protected_count = _sample_rows(
        fluid.values.shape[0], protected_fluid_union, config.max_fluid_queries, rng
    )
    solid = roles["solid_temperature"]
    if solid.receiver_module_ids is None:
        raise ValueError("Solid training receivers must declare module IDs.")
    module_order = tuple(dict.fromkeys(solid.receiver_module_ids))
    hot_local_indices = _protected_solid_local_indices(stencil, config.hot_solid_points_per_module)
    local_count = sum(solid.receiver_module_ids[index] == module_order[0] for index in range(len(solid.receiver_module_ids)))
    local_indices, local_probability, local_protected_count = _sample_rows(
        local_count, hot_local_indices, config.solid_queries_per_module, rng
    )
    solid_rows_by_module = {
        module_id: np.asarray(
            [index for index, current in enumerate(solid.receiver_module_ids) if current == module_id],
            dtype=np.int64,
        )
        for module_id in module_order
    }
    if any(rows.size != local_count for rows in solid_rows_by_module.values()):
        raise ValueError("Solid material roles need equal receiver counts per module.")
    solid_indices_unsorted = np.concatenate(
        [rows[local_indices] for rows in solid_rows_by_module.values()]
    )
    solid_probability_unsorted = np.tile(local_probability, len(module_order))
    permutation = np.argsort(solid_indices_unsorted, kind="stable")
    solid_indices = solid_indices_unsorted[permutation]
    solid_probability = solid_probability_unsorted[permutation]
    interface = roles["interface"]
    interface_indices = np.arange(interface.values.shape[0], dtype=np.int64)
    interface_probability = np.ones((interface_indices.size,), dtype=np.float64)

    selections = {
        "fluid_fields": (fluid_indices, fluid_probability),
        "interface": (interface_indices, interface_probability),
        "solid_temperature": (solid_indices, solid_probability),
    }
    sampled_records = []
    for record in stencil.records:
        sampled_roles = {
            name: _subsample_role(record.output.roles[name], *selections[name])  # type: ignore[union-attr]
            for name in record.output.roles  # type: ignore[union-attr]
        }
        sampled_records.append(_replace_record_roles(record, sampled_roles))
    sampled = ResponseStencil(sampled_records[0], dict(zip(stencil.variants, sampled_records[1:])))
    selected_ids = set(sampled.baseline.output.roles["solid_temperature"].query_ids)  # type: ignore[union-attr]
    coverage: dict[str, bool] = {}
    temperature_channel = _role_channels(solid)[0]
    for record in stencil.records:
        role = record.output.roles["solid_temperature"]  # type: ignore[union-attr]
        for module_id in module_order:
            rows = np.asarray(
                [index for index, current in enumerate(role.receiver_module_ids) if current == module_id],
                dtype=np.int64,
            )
            valid = role.valid_mask[rows, temperature_channel]
            if not np.any(valid):
                coverage[f"{record.record_id}:{module_id}"] = False
                continue
            peak_row = int(rows[np.flatnonzero(valid)[np.argmax(role.values[rows[valid], temperature_channel])]])
            coverage[f"{record.record_id}:{module_id}"] = role.query_ids[peak_row] in selected_ids
    if not all(coverage.values()):
        missing = [key for key, value in coverage.items() if not value]
        raise ValueError(f"Sampled solid support omits exact reference peak queries: {missing}.")
    summary = SamplingSummary(
        physical_family_id=stencil.physical_family_id,
        split=stencil.split.value,
        original_counts={name: roles[name].values.shape[0] for name in selections},
        sampled_counts={name: selections[name][0].size for name in selections},
        protected_counts={
            "fluid_pressure_bands": int(protected_fluid.size),
            "fluid_near_interface_union": int(protected_near_interface.size),
            "fluid_protected_union": fluid_protected_count,
            "interface_ports": interface_indices.size,
            "solid_hot_local_locations": local_protected_count,
        },
        solid_peak_query_coverage=coverage,
        inverse_probability_weighting=True,
    )
    return SampledResponseStencil(sampled, summary)


def sample_training_panel(
    stencils: Sequence[ResponseStencil],
    *,
    config: ReceiverSamplingConfig | None = None,
) -> tuple[SampledResponseStencil, ...]:
    if not stencils:
        raise ValueError("A fixed response panel must contain at least one stencil.")
    if any(stencil.split is not EvidenceSplit.TRAIN for stencil in stencils):
        raise ValueError("Training panels may contain train-split stencils only.")
    config = ReceiverSamplingConfig() if config is None else config
    return tuple(sample_training_stencil(stencil, config=config) for stencil in stencils)


__all__ = [
    "ReceiverSamplingConfig",
    "SampledResponseStencil",
    "SamplingSummary",
    "sample_training_panel",
    "sample_training_stencil",
]
