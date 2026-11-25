"""Couple native row samples and reuse their predictions for response loss.

The population here is the adapter's actual native row population. Distinct
rows at the same physical position remain distinct. The only supported panel
law is uniform sampling without replacement; unsupported laws are reported
without changing the caller's ordinary native sampler. All physical matching
and dependency eligibility are declared before sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import numpy as np
import torch

from .observation_blocks import FieldComponent


@dataclass(frozen=True)
class ObservationKey:
    """Exact shared physical identity of a native observation row.

    ``correspondence`` is a declared common frame or persistent member
    identity, never an unrelated local member index. ``extraction`` declares
    the physical extraction meaning shared by the row's components. A row
    without a valid correspondence uses ``None`` instead of a key.
    """

    frame: Hashable
    time: Hashable
    coordinates: tuple[float, ...]
    correspondence: Hashable
    extraction: Hashable

    def __post_init__(self) -> None:
        coordinates = tuple(self.coordinates)
        if not coordinates or not np.isfinite(coordinates).all():
            raise ValueError('Observation coordinates must be finite and nonempty.')
        if self.frame is None or self.time is None or self.correspondence is None or self.extraction is None:
            raise ValueError('A matching key needs declared frame, time, correspondence and extraction identities.')
        object.__setattr__(self, 'coordinates', coordinates)
        try:
            hash((self.frame, self.time, coordinates, self.correspondence, self.extraction))
        except TypeError as exc:
            raise ValueError('Observation identities must be hashable.') from exc


def _array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class NativeObservationPopulation:
    """A scene's full native rows and predeclared extraction eligibility.

    ``references``, ``measures``, ``valid`` and ``dependency_complete`` have
    [row, component] shape. ``dependencies[row][component]`` is the nonempty
    tuple of native execution dependency IDs required for that extraction.
    The completeness mask means the native sampler/executor guarantees those
    dependencies whenever the row is selected, not that they happened to be
    available in one realized sample. Unsupported components use False (or
    empty dependencies) and retain their ordinary native supervision.

    ``extraction_ids`` can distinguish component-specific physical operators;
    matching requires equal identities in both scenes. None means unavailable.
    By default each FieldComponent supplies the component identity, alongside
    the extraction identity already present in the ObservationKey.
    """

    keys: Sequence[ObservationKey | None]
    components: Sequence[FieldComponent]
    references: Any
    measures: Any
    valid: Any
    dependency_complete: Any
    dependencies: Sequence[Sequence[Sequence[Hashable] | None]]
    extraction_ids: Sequence[Sequence[Hashable | None]] | None = None

    def __post_init__(self) -> None:
        keys, components = tuple(self.keys), tuple(self.components)
        n, c = len(keys), len(components)
        if not n or not c or any(key is not None and not isinstance(key, ObservationKey) for key in keys):
            raise ValueError('Native populations need observation keys and configured components.')
        if any(not isinstance(component, FieldComponent) for component in components):
            raise ValueError('Every component must be a FieldComponent.')
        if len({component.name for component in components}) != c:
            raise ValueError('Native component names must be unique.')
        reference, measure = _array(self.references), _array(self.measures)
        valid, complete = _array(self.valid), _array(self.dependency_complete)
        if any(array.shape != (n, c) for array in (reference, measure, valid, complete)):
            raise ValueError('Native reference, measure and masks must be [row, component].')
        if valid.dtype != np.bool_ or complete.dtype != np.bool_:
            raise ValueError('Native validity and dependency completeness masks must be boolean.')
        if not np.issubdtype(reference.dtype, np.number) or not np.issubdtype(measure.dtype, np.number):
            raise ValueError('Native references and measures must be numerical.')
        if (not np.isfinite(reference[valid]).all() or not np.isfinite(measure[valid]).all()
                or np.any(measure[valid] < 0)):
            raise ValueError('Valid references and physical measures must be finite, with nonnegative measures.')
        if len(self.dependencies) != n or any(len(row) != c for row in self.dependencies):
            raise ValueError('Dependency declarations must align with the native row/component population.')
        # Native populations often repeat the same position and several
        # components use the same query. Share immutable identities inside the
        # population instead of retaining a separate Python tuple per use.
        key_pool: dict[ObservationKey, ObservationKey] = {}
        keys = tuple(None if key is None else key_pool.setdefault(key, key) for key in keys)
        dependency_pool: dict[tuple[Hashable, ...], tuple[Hashable, ...]] = {}
        dependency_row_pool: dict[tuple[tuple[Hashable, ...], ...], tuple[tuple[Hashable, ...], ...]] = {}
        dependencies = []
        for row in self.dependencies:
            declared = []
            for values in row:
                items = () if values is None else tuple(values)
                try:
                    for item in items:
                        hash(item)
                except TypeError as exc:
                    raise ValueError('Native dependency IDs must be hashable.') from exc
                declared.append(dependency_pool.setdefault(items, items))
            row = tuple(declared)
            dependencies.append(dependency_row_pool.setdefault(row, row))
        if self.extraction_ids is None:
            extraction_ids = (components,) * n
        else:
            if len(self.extraction_ids) != n or any(len(row) != c for row in self.extraction_ids):
                raise ValueError('Extraction identities must align with native rows and components.')
            extraction_rows = (tuple(row) for row in self.extraction_ids)
            try:
                extraction_pool: dict[tuple[Hashable | None, ...], tuple[Hashable | None, ...]] = {}
                extraction_ids_list = []
                for row in extraction_rows:
                    for item in row:
                        hash(item)
                    extraction_ids_list.append(extraction_pool.setdefault(row, row))
            except TypeError as exc:
                raise ValueError('Component extraction identities must be hashable.') from exc
            extraction_ids = tuple(extraction_ids_list)
        object.__setattr__(self, 'keys', keys)
        object.__setattr__(self, 'components', components)
        object.__setattr__(self, 'references', reference)
        object.__setattr__(self, 'measures', measure)
        object.__setattr__(self, 'valid', valid)
        object.__setattr__(self, 'dependency_complete', complete)
        object.__setattr__(self, 'dependencies', tuple(dependencies))
        object.__setattr__(self, 'extraction_ids', extraction_ids)


@dataclass(frozen=True)
class NativeCouplingPlan:
    left: NativeObservationPopulation
    right: NativeObservationPopulation
    available: bool
    reason: str | None
    left_buckets: tuple[np.ndarray, ...] = ()
    right_buckets: tuple[np.ndarray, ...] = ()
    left_bucket_probabilities: Any = None
    right_bucket_probabilities: Any = None
    common_bucket_probabilities: Any = None
    common_probability: float = 0.0
    left_groups: Any = None
    right_groups: Any = None
    left_fractions: Any = None
    right_fractions: Any = None
    group_masses: Any = None
    eligible_mass: Any = None
    dependency_ineligible_rows: Any = None


def prepare_native_coupling(left: NativeObservationPopulation, right: NativeObservationPopulation, *,
                            sampling_law: str = 'uniform_without_replacement') -> NativeCouplingPlan:
    """Prepare common proposals and fixed physical masses, without row products."""
    if left.components != right.components:
        raise ValueError('Coupled scenes must declare identical component semantics, scales and weights.')
    if sampling_law != 'uniform_without_replacement':
        return NativeCouplingPlan(left, right, False, f'Unsupported native sampling law: {sampling_law}')
    buckets: dict[Hashable, tuple[list[int], list[int]]] = {}
    for side, population in enumerate((left, right)):
        for row, key in enumerate(population.keys):
            # Distinct uncorresponded rows cannot accidentally match each other.
            bucket_key = key if key is not None else ('uncorresponded', side, row)
            if bucket_key not in buckets:
                buckets[bucket_key] = ([], [])
            buckets[bucket_key][side].append(row)
    left_buckets = tuple(np.asarray(value[0], dtype=np.int64) for value in buckets.values())
    right_buckets = tuple(np.asarray(value[1], dtype=np.int64) for value in buckets.values())
    pl = np.asarray([len(rows) / len(left.keys) for rows in left_buckets], dtype=np.float64)
    pr = np.asarray([len(rows) / len(right.keys) for rows in right_buckets], dtype=np.float64)
    common = np.minimum(pl, pr)
    c = len(left.components)
    lg = np.full((len(left.keys), c), -1, dtype=np.int64)
    rg = np.full((len(right.keys), c), -1, dtype=np.int64)
    lf = np.zeros_like(lg, dtype=np.float64)
    rf = np.zeros_like(rg, dtype=np.float64)
    masses: list[float] = []
    eligible_mass = np.zeros(c, dtype=np.float64)
    ineligible = np.zeros((2, c), dtype=np.int64)
    for side, population in enumerate((left, right)):
        for row in range(len(population.keys)):
            for component in range(c):
                if (population.valid[row, component] and population.measures[row, component] > 0 and
                        (not population.dependency_complete[row, component] or
                         not population.dependencies[row][component] or
                         population.extraction_ids[row][component] is None)):
                    ineligible[side, component] += 1
    for lrows, rrows, probability in zip(left_buckets, right_buckets, common):
        if probability == 0:
            continue
        for component in range(c):
            grouped: list[dict[Hashable, list[int]]] = [{}, {}]
            for side, (population, rows) in enumerate(((left, lrows), (right, rrows))):
                for row_index in rows:
                    row = int(row_index)
                    identity = population.extraction_ids[row][component]
                    if (population.valid[row, component] and population.measures[row, component] > 0 and
                            population.dependency_complete[row, component] and
                            population.dependencies[row][component] and identity is not None):
                        grouped[side].setdefault(identity, []).append(row)
            for identity, lvalues in grouped[0].items():
                rvalues = grouped[1].get(identity)
                if not rvalues:
                    continue
                lm = np.asarray(left.measures[lvalues, component], dtype=np.float64)
                rm = np.asarray(right.measures[rvalues, component], dtype=np.float64)
                lsum, rsum = float(lm.sum()), float(rm.sum())
                mass = min(lsum, rsum)
                if not np.isfinite(lsum) or not np.isfinite(rsum):
                    raise ValueError('Physical bucket mass overflowed.')
                group = len(masses)
                masses.append(mass)
                eligible_mass[component] += mass
                lg[lvalues, component], rg[rvalues, component] = group, group
                lf[lvalues, component], rf[rvalues, component] = lm / lsum, rm / rsum
    return NativeCouplingPlan(
        left, right, True, None, left_buckets, right_buckets, pl, pr, common,
        float(common.sum()), lg, rg, lf, rf, np.asarray(masses, dtype=np.float64),
        eligible_mass, ineligible,
    )


@dataclass(frozen=True)
class NativeCoupledSample:
    plan: NativeCouplingPlan
    left_rows: np.ndarray
    right_rows: np.ndarray
    left_anchor_rows: np.ndarray
    right_anchor_rows: np.ndarray
    left_anchor_positions: np.ndarray
    right_anchor_positions: np.ndarray
    common: np.ndarray
    common_row_probability: np.ndarray


def _choice(rng: np.random.Generator, probabilities: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError('Cannot draw from an empty probability distribution.')
    return rng.choice(len(probabilities), size=count, p=probabilities / total)


def _complete_panel(population: int, budget: int, anchors: np.ndarray, *,
                    completion_rng: np.random.Generator, order_rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(anchors)
    remaining = np.setdiff1d(np.arange(population, dtype=np.int64), unique, assume_unique=True)
    completion = completion_rng.choice(remaining, size=budget - len(unique), replace=False)
    panel = np.concatenate((unique, completion))
    panel = panel[order_rng.permutation(len(panel))]
    positions = np.empty(population, dtype=np.int64)
    positions[panel] = np.arange(budget, dtype=np.int64)
    return panel, positions[anchors]


def sample_native_coupling(plan: NativeCouplingPlan, left_budget: int, right_budget: int, *,
                           anchor_rng: np.random.Generator, left_completion_rng: np.random.Generator,
                           right_completion_rng: np.random.Generator, left_order_rng: np.random.Generator,
                           right_order_rng: np.random.Generator, anchor_count: int = 128) -> NativeCoupledSample:
    """Draw maximally coupled IID row anchors, then complete/shuffle each panel.

    The five generators must be distinct independent sampling namespaces. The
    caller owns their reproducible seeds and saved state. Anchor occurrences
    and their inverse panel positions retain duplicate draws.
    """
    if not plan.available:
        raise ValueError(plan.reason or 'Native coupling is unavailable.')
    generators = (anchor_rng, left_completion_rng, right_completion_rng, left_order_rng, right_order_rng)
    if (any(not isinstance(rng, np.random.Generator) for rng in generators) or
            len({id(rng) for rng in generators}) != len(generators)):
        raise ValueError('Coupling needs five independent NumPy Generator namespaces.')
    if (type(left_budget) is not int or type(right_budget) is not int or type(anchor_count) is not int or
            not 1 <= left_budget <= len(plan.left.keys) or not 1 <= right_budget <= len(plan.right.keys) or
            anchor_count < 1):
        raise ValueError('Native budgets must be within each population; anchor_count must be positive.')
    k = min(anchor_count, left_budget, right_budget)
    common_probability = min(1.0, max(0.0, plan.common_probability))
    common = anchor_rng.random(k) < common_probability
    common_count = int(common.sum())
    lb = np.empty(k, dtype=np.int64)
    rb = np.empty(k, dtype=np.int64)
    common_bucket = _choice(anchor_rng, plan.common_bucket_probabilities, common_count)
    lb[common] = common_bucket
    rb[common] = common_bucket
    remaining = k - common_count
    left_residual = np.maximum(plan.left_bucket_probabilities - plan.common_bucket_probabilities, 0.0)
    right_residual = np.maximum(plan.right_bucket_probabilities - plan.common_bucket_probabilities, 0.0)
    lb[~common] = _choice(anchor_rng, left_residual, remaining)
    rb[~common] = _choice(anchor_rng, right_residual, remaining)
    la = np.asarray([rows[int(anchor_rng.integers(len(rows)))] for rows in
                     (plan.left_buckets[index] for index in lb)], dtype=np.int64)
    ra = np.asarray([rows[int(anchor_rng.integers(len(rows)))] for rows in
                     (plan.right_buckets[index] for index in rb)], dtype=np.int64)
    q = np.zeros(k, dtype=np.float64)
    for occurrence in np.flatnonzero(common):
        bucket = int(lb[occurrence])
        q[occurrence] = (plan.common_bucket_probabilities[bucket] /
                         (len(plan.left_buckets[bucket]) * len(plan.right_buckets[bucket])))
    left_rows, left_positions = _complete_panel(len(plan.left.keys), left_budget, la,
        completion_rng=left_completion_rng, order_rng=left_order_rng)
    right_rows, right_positions = _complete_panel(len(plan.right.keys), right_budget, ra,
        completion_rng=right_completion_rng, order_rng=right_order_rng)
    return NativeCoupledSample(plan, left_rows, right_rows, la, ra, left_positions, right_positions, common, q)


@dataclass(frozen=True)
class CoupledResponseEstimate:
    loss: torch.Tensor | None
    view_loss: torch.Tensor | None
    per_component: dict[str, torch.Tensor | None]
    eligible_mass: dict[str, float]
    common_probability: float
    realized_common_draws: int
    eligible_draws: dict[str, int]
    inverse_probability_weights: dict[str, np.ndarray]


def coupled_response_loss(plan: NativeCouplingPlan, sample: NativeCoupledSample,
                          left_predictions: torch.Tensor, right_predictions: torch.Tensor, *,
                          left_dependencies: Sequence[Hashable] | None = None,
                          right_dependencies: Sequence[Hashable] | None = None) -> CoupledResponseEstimate:
    """Estimate fixed-measure response from existing native extraction outputs.

    Predictions are [panel, component] or [view, panel, component] and retain
    the graph used by the native loss. A missing promised dependency is an
    execution error; querying it here or dropping the sampled row would change
    the native computation or bias the fixed-population estimate.
    """
    if not plan.available or sample.plan is not plan:
        raise ValueError('Response loss needs a sample from an available matching plan.')
    if (not torch.is_tensor(left_predictions) or not torch.is_tensor(right_predictions) or
            not left_predictions.is_floating_point() or not right_predictions.is_floating_point() or
            left_predictions.ndim not in (2, 3) or left_predictions.ndim != right_predictions.ndim or
            left_predictions.device != right_predictions.device):
        raise ValueError('Predictions need same-device floating [panel, component] or [view, panel, component] tensors.')
    c = len(plan.left.components)
    if (left_predictions.shape[-2:] != (len(sample.left_rows), c) or
            right_predictions.shape[-2:] != (len(sample.right_rows), c) or
            left_predictions.shape[:-2] != right_predictions.shape[:-2]):
        raise ValueError('Native extracted predictions must align with both sampled panels and component identities.')
    left_rows, right_rows = sample.left_anchor_rows, sample.right_anchor_rows
    left_group, right_group = plan.left_groups[left_rows], plan.right_groups[right_rows]
    active = ((left_group >= 0) & (left_group == right_group) & sample.common[:, None])
    group = np.maximum(left_group, 0)
    weight = np.zeros(active.shape, dtype=np.float64)
    if active.any():
        mass = plan.group_masses[group[active]]
        raw = mass * plan.left_fractions[left_rows][active] * plan.right_fractions[right_rows][active]
        q = np.broadcast_to(sample.common_row_probability[:, None], active.shape)[active]
        weight[active] = raw / q
        for population, rows, available in ((plan.left, left_rows, left_dependencies),
                                            (plan.right, right_rows, right_dependencies)):
            if available is None:
                raise ValueError('Actual native execution dependencies must accompany supported response predictions.')
            present = set(available)
            for occurrence, component in np.argwhere(active):
                missing = set(population.dependencies[int(rows[occurrence])][int(component)]) - present
                if missing:
                    raise ValueError(f'Native execution omitted declared extraction dependencies: {missing!r}')
    device = left_predictions.device
    source_l = torch.as_tensor(np.where(active, plan.left.references[left_rows], 0), device=device)
    source_r = torch.as_tensor(np.where(active, plan.right.references[right_rows], 0), device=device)
    dtype = torch.promote_types(torch.promote_types(left_predictions.dtype, right_predictions.dtype),
                                torch.promote_types(source_l.dtype, source_r.dtype))
    mask = torch.as_tensor(active, device=device)
    left_pred = left_predictions[..., torch.as_tensor(sample.left_anchor_positions, device=device), :].to(dtype)
    right_pred = right_predictions[..., torch.as_tensor(sample.right_anchor_positions, device=device), :].to(dtype)
    left_pred = torch.where(mask, left_pred, torch.zeros_like(left_pred))
    right_pred = torch.where(mask, right_pred, torch.zeros_like(right_pred))
    scale = torch.as_tensor([field.scale for field in plan.left.components], dtype=dtype, device=device)
    contrast = ((right_pred - source_r.to(dtype)) - (left_pred - source_l.to(dtype))) / scale
    weighted = contrast.square() * torch.as_tensor(weight, dtype=dtype, device=device)
    numerator = weighted.sum(dim=-2) / len(sample.common)
    denominator = torch.as_tensor(np.where(plan.eligible_mass > 0, plan.eligible_mass, 1.),
                                  dtype=dtype, device=device)
    estimates = numerator / denominator
    available_components = plan.eligible_mass > 0
    component_weights = np.asarray([field.weight for field in plan.left.components], dtype=np.float64)
    total_weight = float(component_weights[available_components].sum())
    view_loss = None
    loss = None
    if total_weight > 0:
        weights = torch.as_tensor(np.where(available_components, component_weights, 0.), dtype=dtype, device=device)
        view_loss = (estimates * weights).sum(dim=-1) / total_weight
        loss = view_loss.mean()
    return CoupledResponseEstimate(
        loss, view_loss,
        {field.name: estimates[..., index] if available_components[index] else None
         for index, field in enumerate(plan.left.components)},
        {field.name: float(plan.eligible_mass[index]) for index, field in enumerate(plan.left.components)},
        plan.common_probability, int(sample.common.sum()),
        {field.name: int(active[:, index].sum()) for index, field in enumerate(plan.left.components)},
        {field.name: weight[:, index].copy() for index, field in enumerate(plan.left.components)},
    )
