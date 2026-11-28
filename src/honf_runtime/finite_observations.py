"""Finite physical observations for scene anchors and response contrasts.

Adapters select existing source rows before constructing these compact objects.
Each scene is reduced on its own rows; no member or point correspondence is
implied between scenes. The fixed objective is exact for the selected rows,
not an unbiased estimate of a full-support squared observation loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from collections.abc import Hashable, Mapping, Sequence
from typing import Any
import math

import numpy as np
import torch

from .observation_blocks import FieldComponent, LinearExtraction


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _identity(value: Any, label: str, *, nullable: bool = False) -> None:
    if value is None and not nullable:
        raise ValueError(f'{label} must be declared.')
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f'{label} must be hashable.') from exc


def select_source_rows(valid: Any, *, count: int = 64, seed: int) -> np.ndarray:
    """Select source-valid row indices uniformly, without replacement.

    The caller's mask must incorporate physical validity and positive measures;
    this function never sees reference values, model errors or edit outcomes.
    Rows are returned in original source order after selection. Passing the same
    mask and seed gives the same selection, including for shared populations.
    """
    valid = _numpy(valid)
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError('Source validity must be a one-dimensional boolean mask.')
    if type(count) is not int or count <= 0 or type(seed) is not int or seed < 0:
        raise ValueError('The row count must be positive and the seed a nonnegative integer.')
    candidates = np.flatnonzero(valid)
    if len(candidates) <= count:
        return candidates.astype(np.int64, copy=False)
    return np.sort(np.random.default_rng(seed).choice(candidates, size=count, replace=False)).astype(np.int64)


@dataclass(frozen=True)
class AffineFrame:
    """One common input-defined physical frame, shared across all scenes."""

    identity: Hashable
    lower: Sequence[float]
    upper: Sequence[float]

    def __post_init__(self) -> None:
        _identity(self.identity, 'Frame identity')
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if (lower.ndim != 1 or not len(lower) or upper.shape != lower.shape or
                not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= lower)):
            raise ValueError('Frame bounds must be finite vectors with strictly positive extent.')
        object.__setattr__(self, 'lower', tuple(float(value) for value in lower))
        object.__setattr__(self, 'upper', tuple(float(value) for value in upper))


@dataclass(frozen=True)
class FiniteObservationFamily:
    """Selected rows of one configured field and physical support.

    Native importance belongs to the support family, not to an output column.
    The linear extraction has one observation component (index zero), one row
    per selected source ID and all dependencies, including complete stencils.
    Families with no valid rows are omitted from the scene rather than filled
    with zero labels. Coordinates may have zero columns for constant-only data.
    """

    name: str
    component: FieldComponent
    support: Hashable
    frame: Hashable | None
    time: Hashable
    extraction_id: Hashable
    native_weight: float
    row_ids: Sequence[Hashable]
    coordinates: Any
    measures: Any
    references: Any
    extraction: LinearExtraction

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Each family needs a nonempty name.')
        if not isinstance(self.component, FieldComponent):
            raise ValueError('Each family needs a configured FieldComponent.')
        for value, label, nullable in ((self.support, 'Support', False), (self.frame, 'Frame', True),
                                        (self.time, 'Time convention', False),
                                        (self.extraction_id, 'Extraction identity', False)):
            _identity(value, label, nullable=nullable)
        if not np.isfinite(self.native_weight) or self.native_weight <= 0:
            raise ValueError('Each configured family needs positive finite native importance.')
        rows = tuple(self.row_ids)
        if not rows:
            raise ValueError('Omit an unavailable family instead of creating an empty one.')
        for row in rows:
            _identity(row, 'Source row identity')
        coordinates, measures, references = (_numpy(self.coordinates), _numpy(self.measures), _numpy(self.references))
        if (coordinates.ndim != 2 or coordinates.shape[0] != len(rows) or
                measures.shape != (len(rows),) or references.shape != (len(rows),)):
            raise ValueError('Family coordinates, measures and references must align with selected rows.')
        for values in (coordinates, measures, references):
            if not np.issubdtype(values.dtype, np.number) or not np.isrealobj(values) or not np.isfinite(values).all():
                raise ValueError('Selected coordinates, measures and references must be finite real numbers.')
        if np.any(measures <= 0):
            raise ValueError('Selected physical measures must be strictly positive.')
        if not isinstance(self.extraction, LinearExtraction):
            raise ValueError('Each selected family needs a linear extraction map.')
        extraction = self.extraction
        obs, comps = _numpy(extraction.observation_indices), _numpy(extraction.component_indices)
        if (np.any(obs >= len(rows)) or np.any(comps != 0) or
                set(obs.tolist()) != set(range(len(rows)))):
            raise ValueError('Every selected row needs extraction terms for family component zero.')
        object.__setattr__(self, 'row_ids', rows)
        object.__setattr__(self, 'coordinates', coordinates)
        object.__setattr__(self, 'measures', measures)
        object.__setattr__(self, 'references', references)
        object.__setattr__(self, 'native_weight', float(self.native_weight))


@dataclass(frozen=True)
class FiniteObservationScene:
    scene_id: Hashable
    families: Sequence[FiniteObservationFamily]

    def __post_init__(self) -> None:
        _identity(self.scene_id, 'Scene identity')
        families = tuple(self.families)
        if any(not isinstance(family, FiniteObservationFamily) for family in families):
            raise ValueError('A scene must contain finite-observation families.')
        if len({family.name for family in families}) != len(families):
            raise ValueError('Family names must be unique within a scene.')
        if any(dependency.scene_id != self.scene_id for family in families
               for dependency in family.extraction.dependencies):
            raise ValueError('Every extraction dependency must belong to its scene.')
        object.__setattr__(self, 'families', families)


def _positive_weights(configured_weights: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(configured_weights, Mapping) or not configured_weights:
        raise ValueError('Declare the positive native weights for all configured families.')
    weights = {}
    for name, weight in configured_weights.items():
        if not isinstance(name, str) or not name or not np.isfinite(weight) or weight <= 0:
            raise ValueError('Configured family names and native weights must be positive and finite.')
        weights[name] = float(weight)
    if not math.isfinite(sum(weights.values())):
        raise ValueError('The sum of native family weights must be finite.')
    return weights


def extract_family_predictions(family: FiniteObservationFamily, predictions: torch.Tensor, *,
                               dependency_indices: Any | None = None, output_scale: Any | None = None,
                               output_offset: Any | None = None) -> torch.Tensor:
    """Apply a selected sparse extraction to physical joint-field predictions.

    ``predictions`` is [..., dependency, output]. With ``dependency_indices``,
    each family dependency instead indexes a deduplicated union in that axis.
    Scale and offset, when supplied, convert the joint fields to physical units
    *before* the extraction (including signed or derivative coefficients).
    """
    if not isinstance(family, FiniteObservationFamily):
        raise ValueError('Expected a selected finite-observation family.')
    if not torch.is_tensor(predictions) or not predictions.is_floating_point() or predictions.ndim < 2:
        raise ValueError('Predictions must be floating [..., dependency, output] tensors.')
    extraction = family.extraction
    num_dependencies = len(extraction.dependencies)
    if dependency_indices is None:
        if predictions.shape[-2] != num_dependencies:
            raise ValueError('Predictions must align with the family dependencies or provide union indices.')
        dependency_indices = np.arange(num_dependencies, dtype=np.int64)
    else:
        dependency_indices = _numpy(dependency_indices)
        if (dependency_indices.shape != (num_dependencies,) or
                not np.issubdtype(dependency_indices.dtype, np.integer) or
                np.any(dependency_indices < 0) or np.any(dependency_indices >= predictions.shape[-2])):
            raise ValueError('Union indices must map every family dependency to a prediction.')
    outputs = _numpy(extraction.output_indices)
    if len(outputs) and np.max(outputs) >= predictions.shape[-1]:
        raise ValueError('An extraction output index exceeds the predicted joint fields.')
    physical = predictions
    for value, operation in ((output_scale, 'scale'), (output_offset, 'offset')):
        if value is not None:
            value = torch.as_tensor(value, device=predictions.device)
            if value.ndim > 1 or (value.ndim == 1 and value.shape[0] != predictions.shape[-1]):
                raise ValueError(f'Output {operation} must be scalar or match the joint output count.')
            physical = physical * value if operation == 'scale' else physical + value
    indices = torch.as_tensor(dependency_indices[_numpy(extraction.dependency_indices)],
                              device=predictions.device, dtype=torch.long)
    outputs = torch.as_tensor(outputs, device=predictions.device, dtype=torch.long)
    rows = torch.as_tensor(_numpy(extraction.observation_indices), device=predictions.device, dtype=torch.long)
    coefficients = torch.as_tensor(_numpy(extraction.coefficients), device=predictions.device)
    terms = physical[..., indices, outputs] * coefficients
    result = terms.new_zeros((*terms.shape[:-1], len(family.row_ids)))
    return result.index_add(-1, rows, terms)


@dataclass(frozen=True)
class FamilyObservations:
    family: FiniteObservationFamily
    anchor: torch.Tensor
    moments: torch.Tensor
    basis: str
    frame_definition: AffineFrame | None
    out_of_frame_count: int


@dataclass(frozen=True)
class SceneObservations:
    scene: FiniteObservationScene
    anchor: torch.Tensor | None
    families: Mapping[str, FamilyObservations]
    missing_families: tuple[str, ...]
    configured_weights: Mapping[str, float]


@dataclass(frozen=True)
class SceneContrast:
    loss: torch.Tensor | None
    eligible_families: tuple[str, ...]
    missing_families: tuple[str, ...]


@dataclass(frozen=True)
class SceneEligibility:
    eligible_families: tuple[str, ...]
    missing_families: tuple[str, ...]


def _compatible_descriptors(first: FiniteObservationFamily, second: FiniteObservationFamily) -> None:
    first_component, second_component = first.component, second.component
    if (first.name != second.name or first.support != second.support or first.frame != second.frame or
            first.time != second.time or first.extraction_id != second.extraction_id or
            (first_component.name, first_component.unit, first_component.output_index, first_component.scale) !=
            (second_component.name, second_component.unit, second_component.output_index, second_component.scale) or
            first.native_weight != second.native_weight):
        raise ValueError(f'Incompatible units, support, frame, time, field or extraction for family {first.name!r}.')


def compatible_scene_families(left: FiniteObservationScene, right: FiniteObservationScene, *,
                              configured_weights: Mapping[str, float]) -> SceneEligibility:
    """Predeclare eligible scene pairs without reading predictions or matching rows.

    A pair with no eligible families is excluded from the fixed auxiliary pair
    set; its scenes can still contribute their anchors. An incompatible family
    present in both scenes is an error rather than a silently dropped label.
    """
    if not isinstance(left, FiniteObservationScene) or not isinstance(right, FiniteObservationScene):
        raise ValueError('Eligibility needs two finite-observation scenes.')
    weights = _positive_weights(configured_weights)
    by_scene = []
    for scene in (left, right):
        available = {family.name: family for family in scene.families}
        for name, family in available.items():
            if name not in weights or not math.isclose(family.native_weight, weights[name], rel_tol=1e-12):
                raise ValueError(f'Native importance for family {name!r} differs from its configured weight.')
        by_scene.append(available)
    first, second = by_scene
    eligible = tuple(name for name in weights if name in first and name in second)
    for name in eligible:
        _compatible_descriptors(first[name], second[name])
    return SceneEligibility(eligible, tuple(name for name in weights if name not in eligible))


def _family_basis(family: FiniteObservationFamily, frame: AffineFrame | None, basis: str,
                  *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, str, int]:
    size = len(family.row_ids)
    if basis == 'constant' or frame is None:
        return torch.ones((size, 1), device=device, dtype=dtype), 'constant', 0
    if frame.identity != family.frame or len(frame.lower) != family.coordinates.shape[1]:
        raise ValueError('Coordinate frame identity and dimension must match the family.')
    # Bounds and selected coordinates are source metadata. Count outside rows
    # there, avoiding a GPU synchronization for every family during training.
    coordinates = family.coordinates.astype(np.float64, copy=False)
    lower, upper = np.asarray(frame.lower), np.asarray(frame.upper)
    affine_coordinates = 2 * (coordinates - lower) / (upper - lower) - 1
    out_of_frame = int(np.count_nonzero(((affine_coordinates < -1) | (affine_coordinates > 1)).any(axis=1)))
    xi = torch.as_tensor(affine_coordinates, device=device, dtype=dtype)
    if basis == 'coordinate':
        return torch.cat((torch.ones((size, 1), device=device, dtype=dtype), xi), dim=-1), basis, out_of_frame
    # Standard three-node hats on [0, 1]. They are not clamped to the frame;
    # outside coordinates can leave the partition of unity and are reported.
    eta = (xi + 1) / 2
    nodes = torch.as_tensor((0., .5, 1.), device=device, dtype=dtype)
    hats = torch.clamp(1 - 2 * torch.abs(eta[..., None] - nodes), min=0)
    columns = [torch.prod(torch.stack([hats[:, dim, index] for dim, index in enumerate(indices)], dim=-1), dim=-1)
               for indices in product(range(3), repeat=xi.shape[1])]
    return torch.stack(columns, dim=-1), basis, out_of_frame


def evaluate_observation_scene(scene: FiniteObservationScene, predictions: Mapping[str, torch.Tensor], *,
                               configured_weights: Mapping[str, float],
                               frames: Mapping[Hashable, AffineFrame] | None = None,
                               basis: str = 'coordinate') -> SceneObservations:
    """Reduce independently extracted predictions to point anchors and moments.

    Each prediction has shape [..., selected row]. Native family weights are
    normalized by the *configured* total, including families unavailable in
    this scene. A missing frame gives a constant-only observation and is exposed
    by the per-family ``basis`` and ``frame_definition`` fields.
    """
    if not isinstance(scene, FiniteObservationScene) or not isinstance(predictions, Mapping):
        raise ValueError('A scene and its selected-family predictions are required.')
    if basis not in ('constant', 'coordinate', 'tensor_hats'):
        raise ValueError('Basis must be constant, coordinate, or tensor_hats.')
    weights = _positive_weights(configured_weights)
    frames = {} if frames is None else frames
    if not isinstance(frames, Mapping) or any(not isinstance(value, AffineFrame) or key != value.identity
                                              for key, value in frames.items()):
        raise ValueError('Frames must map frame identities to their common affine definitions.')
    scene_names = {family.name for family in scene.families}
    if set(predictions) != scene_names:
        raise ValueError('Predictions must contain exactly the available scene families.')
    results: dict[str, FamilyObservations] = {}
    anchor = None
    leading_shape = None
    device = None
    total_weight = sum(weights.values())
    for family in scene.families:
        if family.name not in weights or not math.isclose(family.native_weight, weights[family.name], rel_tol=1e-12):
            raise ValueError(f'Native importance for family {family.name!r} differs from its configured weight.')
        predicted = predictions[family.name]
        if (not torch.is_tensor(predicted) or not predicted.is_floating_point() or predicted.ndim < 1 or
                predicted.shape[-1] != len(family.row_ids)):
            raise ValueError(f'Predictions for family {family.name!r} must align with selected rows.')
        if leading_shape is None:
            leading_shape, device = predicted.shape[:-1], predicted.device
        elif predicted.shape[:-1] != leading_shape or predicted.device != device:
            raise ValueError('All family predictions must have matching leading dimensions and device.')
        reference = torch.as_tensor(family.references, device=predicted.device)
        measure = torch.as_tensor(family.measures, device=predicted.device)
        coords_dtype = torch.as_tensor(family.coordinates).dtype
        dtype = torch.promote_types(torch.promote_types(predicted.dtype, reference.dtype),
                                    torch.promote_types(measure.dtype, coords_dtype))
        if not dtype.is_floating_point:
            dtype = predicted.dtype
        reference = reference.to(dtype=dtype)
        measure = measure.to(dtype=dtype)
        residual = (predicted.to(dtype=dtype) - reference) / family.component.scale
        frame = frames.get(family.frame) if family.frame is not None else None
        values, actual_basis, out_of_frame = _family_basis(family, frame, basis, device=device, dtype=dtype)
        normalized = measure / measure.sum()
        family_anchor = (normalized * residual.square()).sum(dim=-1)
        moments = (residual[..., :, None] * normalized[:, None] * values).sum(dim=-2)
        results[family.name] = FamilyObservations(family, family_anchor, moments, actual_basis,
                                                 frame if actual_basis != 'constant' else None, out_of_frame)
        contribution = family_anchor * (weights[family.name] / total_weight)
        anchor = contribution if anchor is None else anchor + contribution
    return SceneObservations(scene, anchor, results, tuple(name for name in weights if name not in scene_names), weights)


def _compatible(left: FamilyObservations, right: FamilyObservations) -> None:
    _compatible_descriptors(left.family, right.family)
    if (left.basis != right.basis or left.frame_definition != right.frame_definition or
            left.moments.shape != right.moments.shape or left.moments.device != right.moments.device):
        raise ValueError(f'Incompatible basis, common affine frame or prediction shape for family {left.family.name!r}.')


def finite_scene_contrast(left: SceneObservations, right: SceneObservations, *,
                          configured_weights: Mapping[str, float]) -> SceneContrast:
    """Compare residual observations, never individual rows or local members."""
    if not isinstance(left, SceneObservations) or not isinstance(right, SceneObservations):
        raise ValueError('Contrast needs two evaluated scenes.')
    weights = _positive_weights(configured_weights)
    if weights != left.configured_weights or weights != right.configured_weights:
        raise ValueError('Both scenes must use the same complete configured family weights.')
    eligibility = compatible_scene_families(left.scene, right.scene, configured_weights=weights)
    eligible, missing = eligibility.eligible_families, eligibility.missing_families
    loss = None
    denominator = sum(weights.values())
    for name in eligible:
        first, second = left.families[name], right.families[name]
        _compatible(first, second)
        contribution = (second.moments - first.moments).square().mean(dim=-1) * (weights[name] / denominator)
        loss = contribution if loss is None else loss + contribution
    return SceneContrast(loss, eligible, missing)
