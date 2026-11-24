"""Case-neutral physical observations, query reuse and sampled responses.

An adapter describes existing measurements and linear extraction maps.  This
module makes no assumptions about the physical meaning or number of outputs.
All sampling normalizers are computed from the declared population, before
any draws, and repeated draws remain repeated statistical contributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import numpy as np
import torch


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class FieldComponent:
    """One configured output component and its TRAIN-only normalization."""

    name: str
    unit: str
    output_index: int
    scale: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.unit or type(self.output_index) is not int or self.output_index < 0:
            raise ValueError('Each component needs a name, unit and nonnegative output index.')
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Component scales must be finite and positive.')
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError('Component weights must be finite and nonnegative.')


@dataclass(frozen=True)
class QueryDependency:
    """A single joint-field query with an explicit original native identity."""

    scene_id: Hashable
    frame: Hashable
    time: Hashable
    native_id: Hashable
    coordinates: tuple[float, ...]

    def __post_init__(self) -> None:
        coords = tuple(self.coordinates)
        if not coords or not np.isfinite(coords).all():
            raise ValueError('A dependency needs finite physical coordinates.')
        object.__setattr__(self, 'coordinates', coords)
        try:
            hash((self.scene_id, self.frame, self.time, self.native_id, coords))
        except TypeError as exc:
            raise ValueError('Dependency identities and coordinates must be hashable.') from exc


@dataclass(frozen=True)
class LinearExtraction:
    """Sparse terms mapping joint-field queries to observation components.

    All five term arrays have the same length.  Multiple terms may contribute
    to one observation, and coefficients may be signed (for example, a finite
    difference).  The dependency list need not be unique.
    """

    dependencies: tuple[QueryDependency, ...]
    dependency_indices: Any
    observation_indices: Any
    component_indices: Any
    output_indices: Any
    coefficients: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, 'dependencies', tuple(self.dependencies))
        arrays = [_numpy(value) for value in (self.dependency_indices, self.observation_indices,
                  self.component_indices, self.output_indices, self.coefficients)]
        if any(array.ndim != 1 or array.shape != arrays[0].shape for array in arrays):
            raise ValueError('Extraction term arrays must be equally sized vectors.')
        for array in arrays[:4]:
            if (array.size and not np.issubdtype(array.dtype, np.integer)) or np.any(array < 0):
                raise ValueError('Extraction indices must be nonnegative integers.')
        for name, array in zip(('dependency_indices', 'observation_indices', 'component_indices', 'output_indices'),
                               arrays[:4]):
            if not array.size:
                object.__setattr__(self, name, np.empty(0, dtype=np.int64))
        if arrays[0].size and np.max(arrays[0]) >= len(self.dependencies):
            raise ValueError('An extraction dependency index is out of range.')
        if not np.isfinite(arrays[4]).all():
            raise ValueError('Extraction coefficients must be finite.')


@dataclass(frozen=True)
class ObservationBlock:
    """Existing aligned observations for two scenes, without fabricated labels.

    ``references`` and ``absolute_mask`` are [scene, receiver, component];
    ``pair_mask`` and ``edge_mask`` are [receiver, component].  Coordinates
    express the declared common physical or transported member frame.  The
    adapter supplies neighbor groups to prevent edges between unrelated objects
    or disconnected support.  Two independent masks remain available even if
    no response correspondence exists.

    Absolute measures are [2,R,C] and pair measures [R,C].  Measures are
    physical masses for the seed position, not sampling-normalized weights.
    Each extraction must be linear in the decoded joint fields; absent scenes
    can use an empty extraction.
    """

    name: str
    components: tuple[FieldComponent, ...]
    scene_ids: tuple[Hashable, Hashable]
    partition: str
    frame: Hashable
    time: Hashable
    receiver_ids: tuple[str, ...]
    coordinates: Any
    references: Any
    absolute_mask: Any
    pair_mask: Any
    edge_mask: Any
    absolute_measures: Any
    pair_measures: Any
    extractions: tuple[LinearExtraction, LinearExtraction]
    block_weight: float = 1.0
    neighbor_groups: tuple[Hashable, ...] | None = None
    member_ids: tuple[Hashable | None, ...] | None = None
    domain_tags: tuple[Hashable | None, ...] | None = None
    correspondence: str = 'fixed_frame'
    coordinate_transport: str | None = None
    source_times: tuple[Hashable | None, Hashable | None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'components', tuple(self.components))
        object.__setattr__(self, 'scene_ids', tuple(self.scene_ids))
        object.__setattr__(self, 'receiver_ids', tuple(self.receiver_ids))
        object.__setattr__(self, 'extractions', tuple(self.extractions))
        n, c = len(self.receiver_ids), len(self.components)
        if not self.name or not self.partition or not n or not c:
            raise ValueError('A block needs a name, partition, receivers and components.')
        if len(self.scene_ids) != 2 or self.scene_ids[0] == self.scene_ids[1] or len(self.extractions) != 2:
            raise ValueError('A block needs two distinct scenes and two extraction maps.')
        if any(not isinstance(i, str) or not i for i in self.receiver_ids) or len(set(self.receiver_ids)) != n:
            raise ValueError('Receiver identities must be unique nonempty strings within a block.')
        if len({field.name for field in self.components}) != c:
            raise ValueError('Component names must be unique within a block.')
        if not np.isfinite(self.block_weight) or self.block_weight < 0:
            raise ValueError('Block weight must be finite and nonnegative.')
        coords = _numpy(self.coordinates)
        if coords.ndim != 2 or coords.shape[0] != n or not np.isfinite(coords).all():
            raise ValueError('Observation coordinates must be a finite [receiver, dimension] array.')
        if coords.shape[1] < 1:
            raise ValueError('Physical coordinates need at least one dimension.')
        ref, absolute = _numpy(self.references), _numpy(self.absolute_mask)
        pair, edge = _numpy(self.pair_mask), _numpy(self.edge_mask)
        amass, pmass = _numpy(self.absolute_measures), _numpy(self.pair_measures)
        if ref.shape != (2, n, c) or absolute.shape != ref.shape or amass.shape != ref.shape:
            raise ValueError('References, absolute masks and measures must be [2, receiver, component].')
        if pair.shape != (n, c) or edge.shape != pair.shape or pmass.shape != pair.shape:
            raise ValueError('Pair and edge masks and measures must be [receiver, component].')
        if any(value.dtype != np.bool_ for value in (absolute, pair, edge)):
            raise ValueError('Observation validity masks must be boolean.')
        if np.any(pair & ~(absolute[0] & absolute[1])) or np.any(edge & ~pair):
            raise ValueError('Pair support needs both references; edge support needs pair support.')
        if self.correspondence not in ('fixed_frame', 'member_attached', 'none'):
            raise ValueError('Correspondence must be fixed_frame, member_attached or none.')
        if self.source_times is not None:
            if len(self.source_times) != 2:
                raise ValueError('Source-time metadata must align with the two scenes.')
            object.__setattr__(self, 'source_times', tuple(self.source_times))
        if self.correspondence == 'none' and (pair.any() or edge.any()):
            raise ValueError('Observations without correspondence cannot provide pair or edge support.')
        for name in ('member_ids', 'domain_tags'):
            value = getattr(self, name)
            if value is not None:
                value = tuple(value)
                if len(value) != n:
                    raise ValueError(f'{name} must align with receiver identities.')
                try:
                    for item in value:
                        hash(item)
                except TypeError as exc:
                    raise ValueError(f'{name} values must be hashable.') from exc
                object.__setattr__(self, name, value)
        if self.correspondence == 'member_attached':
            if not self.coordinate_transport or self.member_ids is None:
                raise ValueError('Member-attached correspondence needs original IDs and a declared coordinate transport.')
            if any(self.member_ids[i] is None for i in np.flatnonzero(pair.any(axis=1))):
                raise ValueError('Every paired member-attached observation needs a persistent member ID.')
        if not np.isfinite(ref[absolute]).all():
            raise ValueError('References must be finite on individual-scene support.')
        if any(not np.isfinite(value).all() or np.any(value < 0) for value in (amass, pmass)):
            raise ValueError('Observation measures must be finite and nonnegative.')
        groups = (None,) * n if self.neighbor_groups is None else tuple(self.neighbor_groups)
        if len(groups) != n:
            raise ValueError('Neighbor groups must align with receiver identities.')
        try:
            for group in groups:
                hash(group)
        except TypeError as exc:
            raise ValueError('Neighbor groups must be hashable.') from exc
        object.__setattr__(self, 'neighbor_groups', groups)
        for scene, extraction in enumerate(self.extractions):
            dep_indices = _numpy(extraction.dependency_indices)
            obs_indices = _numpy(extraction.observation_indices)
            component_indices = _numpy(extraction.component_indices)
            if obs_indices.size and (obs_indices.max() >= n or component_indices.max() >= c):
                raise ValueError('An extraction observation or component index is out of range.')
            if any(dep.scene_id != self.scene_ids[scene] for dep in extraction.dependencies):
                raise ValueError('Each extraction dependency must belong to its declared scene.')
            if dep_indices.size and any(len(dep.coordinates) != coords.shape[1] for dep in extraction.dependencies):
                raise ValueError('Extraction and observation coordinate dimensions must agree.')
            # A missing map must not silently turn a valid measured value into zero.
            populated = np.zeros((n, c), dtype=bool)
            populated[obs_indices, component_indices] = True
            if np.any(absolute[scene] & ~populated):
                raise ValueError('Every valid observation component needs extraction terms.')


@dataclass(frozen=True)
class DependencyUnion:
    dependencies: tuple[QueryDependency, ...]
    # Per block, per scene: local dependency index -> union index, or -1.
    mappings: tuple[tuple[np.ndarray, np.ndarray], ...]
    requested_dependencies: int
    requested_terms: int


@dataclass(frozen=True)
class ObservationDraw:
    block_index: int
    seed_index: int
    neighbor_index: int | None


@dataclass(frozen=True)
class _BlockPopulation:
    eligible: np.ndarray
    neighbors: tuple[tuple[int, ...], ...]
    absolute_mass: np.ndarray  # [2,C]
    point_mass: np.ndarray  # [C]
    spatial_mass: np.ndarray  # [C]


@dataclass(frozen=True)
class ObservationSamplingPlan:
    blocks: tuple[ObservationBlock, ...]
    populations: tuple[_BlockPopulation, ...]
    probabilities: np.ndarray
    max_neighbors: int


@dataclass(frozen=True)
class SamplingBatch:
    plan: ObservationSamplingPlan
    draws: tuple[ObservationDraw, ...]

    def identities(self) -> tuple[tuple[str, str, str | None], ...]:
        """Return reproducible observation identities in draw order."""
        return tuple((self.plan.blocks[draw.block_index].name,
                      self.plan.blocks[draw.block_index].receiver_ids[draw.seed_index],
                      None if draw.neighbor_index is None else
                      self.plan.blocks[draw.block_index].receiver_ids[draw.neighbor_index])
                     for draw in self.draws)


def _stable_neighbors(block: ObservationBlock, eligible: np.ndarray, limit: int) -> tuple[tuple[int, ...], ...]:
    """Nearest eligible positions within groups, resolving ties by native ID."""
    coords = _numpy(block.coordinates)
    neighbors: list[tuple[int, ...]] = [()] * len(block.receiver_ids)
    grouped: dict[Hashable, list[int]] = {}
    for row in eligible:
        row = int(row)
        domain = None if block.domain_tags is None else block.domain_tags[row]
        member = block.member_ids[row] if block.correspondence == 'member_attached' else None
        grouped.setdefault((block.neighbor_groups[row], domain, member), []).append(row)
    # scipy is an optional project dependency. The NumPy path has identical
    # selection semantics on installations that have only core dependencies.
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    for members in grouped.values():
        if len(members) < 2:
            continue
        values = coords[members].astype(np.float64, copy=False)
        tree = cKDTree(values) if cKDTree is not None else None
        k = min(limit, len(members) - 1)
        queried_distances, queried_indices = (tree.query(values, k=min(k + 2, len(members)))
                                               if tree is not None else (None, None))
        for local, original in enumerate(members):
            if tree is None:
                candidates = range(len(members))
            else:
                candidates = [int(i) for i in np.atleast_1d(queried_indices[local]) if int(i) != local]
                distances = [float(d) for i, d in zip(np.atleast_1d(queried_indices[local]),
                                                       np.atleast_1d(queried_distances[local])) if int(i) != local]
                if len(candidates) > k and distances[k-1] == distances[k]:
                    # Include the whole distance tie before choosing by native ID.
                    radius = np.nextafter(distances[k-1], np.inf)
                    candidates = tree.query_ball_point(values[local], radius)
            ranked = []
            for candidate in candidates:
                if candidate != local:
                    delta = values[candidate] - values[local]
                    ranked.append((float(np.dot(delta, delta)), block.receiver_ids[members[candidate]],
                                   members[candidate]))
            ranked.sort(key=lambda item: (item[0], item[1]))
            neighbors[original] = tuple(item[2] for item in ranked[:k])
    return tuple(neighbors)


def _population(block: ObservationBlock, max_neighbors: int) -> _BlockPopulation:
    absolute = _numpy(block.absolute_mask)
    pair, edge = _numpy(block.pair_mask), _numpy(block.edge_mask)
    amass = _numpy(block.absolute_measures).astype(np.float64, copy=False)
    pmass = _numpy(block.pair_measures).astype(np.float64, copy=False)
    eligible = np.flatnonzero(np.any(absolute & (amass > 0), axis=(0, 2)) |
                              np.any(pair & (pmass > 0), axis=1))
    neighbors = _stable_neighbors(block, eligible, max_neighbors)
    c = len(block.components)
    if not len(eligible):
        return _BlockPopulation(eligible, neighbors, np.zeros((2, c)), np.zeros(c), np.zeros(c))
    adjacent = np.zeros((len(eligible), max_neighbors), dtype=np.int64)
    present = np.zeros_like(adjacent, dtype=bool)
    for row, seed in enumerate(eligible):
        chosen = neighbors[int(seed)]
        adjacent[row, :len(chosen)] = chosen
        present[row, :len(chosen)] = True
    counts = present.sum(axis=1)
    seed_weight = np.where(counts > 0, .5, 1.)
    neighbor_weight = np.divide(.5 * present, counts[:, None], out=np.zeros_like(adjacent, dtype=float),
                                where=counts[:, None] > 0)
    abs_valid = (absolute[:, eligible] * seed_weight[None, :, None] +
                 (absolute[:, adjacent] * neighbor_weight[None, :, :, None]).sum(axis=2))
    point_valid = (pair[eligible] * seed_weight[:, None] +
                   (pair[adjacent] * neighbor_weight[:, :, None]).sum(axis=1))
    edge_probability = np.divide(present, counts[:, None], out=np.zeros_like(adjacent, dtype=float),
                                 where=counts[:, None] > 0)
    spatial_valid = (edge[eligible] & pair[eligible]) * (
        (edge[adjacent] & pair[adjacent]) * edge_probability[:, :, None]).sum(axis=1)
    abs_z = (amass[:, eligible] * abs_valid).sum(axis=1)
    point_z = (pmass[eligible] * point_valid).sum(axis=0)
    spatial_z = (pmass[eligible] * spatial_valid).sum(axis=0)
    return _BlockPopulation(eligible, neighbors, abs_z, point_z, spatial_z)


def prepare_observation_sampling(blocks: Sequence[ObservationBlock], *, max_neighbors: int = 4) -> ObservationSamplingPlan:
    """Prepare the fixed geometry, eligible masses and block proposal once."""
    blocks = tuple(blocks)
    if not blocks or len({block.name for block in blocks}) != len(blocks):
        raise ValueError('Sampling requires distinctly named observation blocks.')
    if type(max_neighbors) is not int or max_neighbors < 1:
        raise ValueError('At least one nearest-neighbor slot is required.')
    populations = tuple(_population(block, max_neighbors) for block in blocks)
    weights = np.asarray([block.block_weight if len(pop.eligible) else 0.0
                          for block, pop in zip(blocks, populations)], dtype=np.float64)
    if not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError('At least one positive-weight block needs eligible observations.')
    return ObservationSamplingPlan(blocks, populations, weights / weights.sum(), max_neighbors)


def sample_observation_blocks(plan: ObservationSamplingPlan, count: int,
                              rng: np.random.Generator) -> SamplingBatch:
    """Draw blocks and seeds with replacement, then one compatible neighbor."""
    if type(count) is not int or count < 1 or not isinstance(rng, np.random.Generator):
        raise ValueError('Sampling requires a positive count and a NumPy Generator.')
    draws = []
    for block_index in rng.choice(len(plan.blocks), size=count, p=plan.probabilities):
        population = plan.populations[int(block_index)]
        seed = int(rng.choice(population.eligible))
        adjacent = population.neighbors[seed]
        neighbor = int(rng.choice(adjacent)) if adjacent else None
        draws.append(ObservationDraw(int(block_index), seed, neighbor))
    return SamplingBatch(plan, tuple(draws))


def union_observation_dependencies(blocks: Sequence[ObservationBlock],
                                   batch: SamplingBatch | None = None) -> DependencyUnion:
    """Select and deduplicate exact joint-field dependencies within this update."""
    blocks = tuple(blocks)
    if batch is not None and (len(blocks) != len(batch.plan.blocks) or
                              any(left is not right for left, right in zip(blocks, batch.plan.blocks))):
        raise ValueError('Sampling batch and extraction blocks must be the same objects.')
    selected = [set() for _ in blocks]
    if batch is not None:
        for draw in batch.draws:
            selected[draw.block_index].add(draw.seed_index)
            if draw.neighbor_index is not None:
                selected[draw.block_index].add(draw.neighbor_index)
    unique, lookup, mappings = [], {}, []
    requested_dependencies = requested_terms = 0
    for index, block in enumerate(blocks):
        block_maps = []
        for scene, extraction in enumerate(block.extractions):
            dep_indices = _numpy(extraction.dependency_indices)
            observations = _numpy(extraction.observation_indices)
            components = _numpy(extraction.component_indices)
            term_mask = (np.ones(len(dep_indices), dtype=bool) if batch is None else
                         np.isin(observations, list(selected[index])))
            term_mask &= _numpy(block.absolute_mask)[scene, observations, components]
            requested_terms += int(term_mask.sum())
            local = np.unique(dep_indices[term_mask])
            requested_dependencies += len(local)
            mapping = np.full(len(extraction.dependencies), -1, dtype=np.int64)
            for dep_index in local:
                dep = extraction.dependencies[int(dep_index)]
                if dep not in lookup:
                    lookup[dep] = len(unique)
                    unique.append(dep)
                mapping[int(dep_index)] = lookup[dep]
            block_maps.append(mapping)
        mappings.append(tuple(block_maps))
    return DependencyUnion(tuple(unique), tuple(mappings), requested_dependencies, requested_terms)


def extract_observations(block: ObservationBlock, scene_index: int, predictions: torch.Tensor,
                         mapping: np.ndarray) -> torch.Tensor:
    """Apply one sparse extraction to union predictions [view, query, output]."""
    if scene_index not in (0, 1) or predictions.ndim != 3 or not predictions.is_floating_point():
        raise ValueError('Extraction needs a scene index and floating [view, query, output] predictions.')
    extraction = block.extractions[scene_index]
    mapping = _numpy(mapping)
    if mapping.shape != (len(extraction.dependencies),):
        raise ValueError('Union mapping must align with extraction dependencies.')
    dep = _numpy(extraction.dependency_indices)
    chosen = mapping[dep]
    obs_indices = _numpy(extraction.observation_indices)
    comp_indices = _numpy(extraction.component_indices)
    active = (chosen >= 0) & _numpy(block.absolute_mask)[scene_index, obs_indices, comp_indices]
    if np.any(chosen[active] >= predictions.shape[1]):
        raise ValueError('Union prediction axis is too short for the mapping.')
    out = _numpy(extraction.output_indices)[active]
    if out.size and out.max() >= predictions.shape[2]:
        raise ValueError('Extraction requests an unavailable joint-field output.')
    device = predictions.device
    # A higher-precision saved interpolation/difference coefficient is part of
    # the observation definition; do not silently round it to model precision.
    coefficients_source = torch.as_tensor(_numpy(extraction.coefficients)[active], device=device)
    dtype = torch.promote_types(predictions.dtype, coefficients_source.dtype)
    idx = torch.as_tensor(chosen[active], dtype=torch.long, device=device)
    output = torch.as_tensor(out, dtype=torch.long, device=device)
    observation = obs_indices[active]
    component = comp_indices[active]
    destination = torch.as_tensor(observation * len(block.components) + component, dtype=torch.long, device=device)
    coefficients = coefficients_source.to(dtype=dtype)
    terms = predictions[:, idx, output].to(dtype=dtype) * coefficients
    result = torch.zeros((predictions.shape[0], len(block.receiver_ids) * len(block.components)),
                         dtype=dtype, device=device)
    result = result.index_add(1, destination, terms)
    return result.reshape(predictions.shape[0], len(block.receiver_ids), len(block.components))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape), dtype=np.float64)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def observation_response_losses(blocks: Sequence[ObservationBlock], predictions: Sequence[torch.Tensor],
                                batch: SamplingBatch, *, include_component_estimates: bool = False) -> dict[str, Any]:
    """Estimate extra absolute, point and spatial response objectives.

    Each seed and its chosen neighbor contribute equally to point/absolute
    terms; an isolated seed contributes alone. Spatial loss uses the signed
    difference of paired residuals and requires both edge endpoints. Each
    scene/component/term uses its fixed population eligible mass, calculated
    over the complete conditional-neighbor law, not the sampled mass.
    """
    blocks, predictions = tuple(blocks), tuple(predictions)
    if (len(blocks) != len(batch.plan.blocks) or len(predictions) != len(blocks) or
            any(left is not right for left, right in zip(blocks, batch.plan.blocks)) or not batch.draws):
        raise ValueError('Sampling, blocks and predictions must be aligned and nonempty.')
    first = predictions[0]
    if not torch.is_tensor(first) or not first.is_floating_point():
        raise ValueError('Predictions must be floating tensors.')
    zero = first.new_zeros(())
    total_abs, total_point, total_spatial = zero, zero, zero
    block_coverage, component_estimates = {}, {}
    for b, (block, pred, population) in enumerate(zip(blocks, predictions, batch.plan.populations)):
        r, c = len(block.receiver_ids), len(block.components)
        if (not torch.is_tensor(pred) or not pred.is_floating_point() or pred.ndim != 4 or pred.shape[0] < 1 or
                pred.shape[1:] != (2, r, c) or pred.device != first.device):
            raise ValueError('Each prediction needs aligned [view, 2, receiver, component] axes.')
        rows = [draw for draw in batch.draws if draw.block_index == b]
        if not rows:
            block_coverage[block.name] = {'draws': 0, 'seeds': 0, 'endpoints': 0, 'neighbor_draws': 0, 'edges': 0,
                                          'absolute_components': (), 'point_components': (),
                                          'spatial_components': ()}
            if include_component_estimates:
                component_estimates[block.name] = {
                    field.name: {'unit': field.unit, 'mse_unit': f'({field.unit})^2',
                        'scale': field.scale, 'component_weight': field.weight,
                        'block_weight': block.block_weight, 'proposal_probability': float(batch.plan.probabilities[b]),
                        'absolute_eligible_mass': population.absolute_mass[:, index].tolist(),
                        'point_eligible_mass': float(population.point_mass[index]),
                        'spatial_eligible_mass': float(population.spatial_mass[index]),
                        'absolute_draws': 0, 'point_draws': 0, 'spatial_draws': 0,
                        'absolute_mse_normalized': 0., 'point_mse_normalized': 0.,
                        'spatial_mse_normalized': 0., 'absolute_mse_physical': 0.,
                        'point_mse_physical': 0., 'spatial_mse_physical': 0.}
                    for index, field in enumerate(block.components)}
            continue
        device = pred.device
        seeds = np.asarray([draw.seed_index for draw in rows], dtype=np.int64)
        adjacent = np.asarray([draw.seed_index if draw.neighbor_index is None else draw.neighbor_index
                               for draw in rows], dtype=np.int64)
        has_neighbor = np.asarray([draw.neighbor_index is not None for draw in rows], dtype=bool)
        # Rows without a neighbor have one contribution instead of a duplicated half.
        point_factor = np.where(has_neighbor, 0.5, 1.0)
        neighbor_factor = np.where(has_neighbor, 0.5, 0.0)
        absolute = _numpy(block.absolute_mask)
        pair, edge = _numpy(block.pair_mask), _numpy(block.edge_mask)
        source_reference = torch.as_tensor(block.references, device=device)
        dtype = torch.promote_types(pred.dtype, source_reference.dtype)
        reference = source_reference.to(dtype=dtype)
        pred = pred.to(dtype=dtype)
        scale = torch.as_tensor([field.scale for field in block.components], device=device, dtype=dtype)
        component_weights = torch.as_tensor([field.weight for field in block.components], device=device, dtype=dtype)
        mask = torch.as_tensor(absolute, dtype=torch.bool, device=device)
        if not bool(torch.isfinite(pred[:, mask]).all()):
            # Only selected rows are required: a provider may leave others unfilled.
            used = np.unique(np.concatenate((seeds, adjacent)))
            used_pred = pred[:, :, torch.as_tensor(used, device=device), :]
            used_mask = mask[:, torch.as_tensor(used, device=device), :]
            if not bool(torch.isfinite(used_pred[:, used_mask]).all()):
                raise ValueError('Selected predictions must be finite on valid observation support.')
        safe_pred = torch.where(mask[None], pred, torch.zeros_like(pred))
        safe_ref = torch.where(mask, reference, torch.zeros_like(reference))
        residual = (safe_pred - safe_ref[None]) / scale
        seed_idx = torch.as_tensor(seeds, dtype=torch.long, device=device)
        next_idx = torch.as_tensor(adjacent, dtype=torch.long, device=device)
        error_seed, error_next = residual[:, :, seed_idx, :], residual[:, :, next_idx, :]
        endpoint_seed = torch.as_tensor(point_factor, dtype=dtype, device=device)[None, None, :, None]
        endpoint_next = torch.as_tensor(neighbor_factor, dtype=dtype, device=device)[None, None, :, None]
        seed_absolute = torch.as_tensor(absolute[:, seeds], dtype=dtype, device=device)[None]
        next_absolute = torch.as_tensor(absolute[:, adjacent], dtype=dtype, device=device)[None]
        abs_error = (endpoint_seed * seed_absolute * error_seed.square() +
                     endpoint_next * next_absolute * error_next.square()).mean(dim=0)
        pair_seed = torch.as_tensor(pair[seeds], dtype=dtype, device=device)[None]
        pair_next = torch.as_tensor(pair[adjacent], dtype=dtype, device=device)[None]
        delta_seed = error_seed[:, 1] - error_seed[:, 0]
        delta_next = error_next[:, 1] - error_next[:, 0]
        point_error = (endpoint_seed[:, 0] * pair_seed * delta_seed.square() +
                       endpoint_next[:, 0] * pair_next * delta_next.square()).mean(dim=0)
        edge_valid = edge[seeds] & edge[adjacent] & pair[seeds] & pair[adjacent] & has_neighbor[:, None]
        spatial_error = ((delta_seed - delta_next).square() *
                         torch.as_tensor(edge_valid, dtype=dtype, device=device)[None]).mean(dim=0)
        draw_weight = len(population.eligible) / (batch.plan.probabilities[b] * len(batch.draws))
        abs_measure = _numpy(block.absolute_measures)[:, seeds, :].astype(np.float64, copy=False)
        pair_measure = _numpy(block.pair_measures)[seeds].astype(np.float64, copy=False)
        active_scenes = (population.absolute_mass > 0).sum(axis=0)
        abs_weight = _safe_divide(abs_measure, population.absolute_mass[:, None, :])
        abs_weight = _safe_divide(abs_weight, active_scenes[None, None, :])
        pweight = _safe_divide(pair_measure, population.point_mass[None])
        sweight = _safe_divide(pair_measure, population.spatial_mass[None])
        abs_parts = draw_weight * (abs_error * torch.as_tensor(abs_weight, device=device, dtype=dtype)).sum(dim=(0, 1))
        point_parts = draw_weight * (point_error * torch.as_tensor(pweight, device=device, dtype=dtype)).sum(dim=0)
        spatial_parts = draw_weight * (spatial_error * torch.as_tensor(sweight, device=device, dtype=dtype)).sum(dim=0)
        total_abs = total_abs + block.block_weight * (abs_parts * component_weights).sum()
        total_point = total_point + block.block_weight * (point_parts * component_weights).sum()
        total_spatial = total_spatial + block.block_weight * (spatial_parts * component_weights).sum()
        abs_valid_draws = ((absolute[:, seeds] | (absolute[:, adjacent] & has_neighbor[None, :, None])) &
                           (abs_measure > 0)).any(axis=0)
        point_valid_draws = ((pair[seeds] | (pair[adjacent] & has_neighbor[:, None])) & (pair_measure > 0))
        spatial_valid_draws = edge_valid & (pair_measure > 0)
        abs_seen, point_seen, spatial_seen = (value.any(axis=0) for value in
                                              (abs_valid_draws, point_valid_draws, spatial_valid_draws))
        block_coverage[block.name] = {
            'draws': len(rows), 'seeds': len(set(seeds.tolist())),
            'endpoints': len(set(seeds.tolist() + adjacent.tolist())),
            'neighbor_draws': int(has_neighbor.sum()), 'edges': int(spatial_valid_draws.any(axis=1).sum()),
            'absolute_components': tuple(field.name for field, valid in zip(block.components, abs_seen) if valid),
            'point_components': tuple(field.name for field, valid in zip(block.components, point_seen) if valid),
            'spatial_components': tuple(field.name for field, valid in zip(block.components, spatial_seen) if valid),
        }
        if include_component_estimates:
            estimates = torch.stack((abs_parts, point_parts, spatial_parts)).detach().cpu().numpy()
            abs_contributions = abs_valid_draws.sum(axis=0)
            point_contributions = point_valid_draws.sum(axis=0)
            spatial_contributions = spatial_valid_draws.sum(axis=0)
            component_estimates[block.name] = {}
            for index, field in enumerate(block.components):
                absolute_value, point_value, spatial_value = map(float, estimates[:, index])
                component_estimates[block.name][field.name] = {
                    'unit': field.unit, 'mse_unit': f'({field.unit})^2', 'scale': field.scale,
                    'component_weight': field.weight,
                    'block_weight': block.block_weight, 'proposal_probability': float(batch.plan.probabilities[b]),
                    'absolute_eligible_mass': population.absolute_mass[:, index].tolist(),
                    'point_eligible_mass': float(population.point_mass[index]),
                    'spatial_eligible_mass': float(population.spatial_mass[index]),
                    'absolute_draws': int(abs_contributions[index]),
                    'point_draws': int(point_contributions[index]),
                    'spatial_draws': int(spatial_contributions[index]),
                    'absolute_mse_normalized': absolute_value, 'point_mse_normalized': point_value,
                    'spatial_mse_normalized': spatial_value,
                    'absolute_mse_physical': absolute_value * field.scale ** 2,
                    'point_mse_physical': point_value * field.scale ** 2,
                    'spatial_mse_physical': spatial_value * field.scale ** 2,
                }
    result = {'extra_absolute': total_abs, 'point_response': total_point, 'spatial_response': total_spatial,
              'coverage': block_coverage, 'draw_count': len(batch.draws)}
    if include_component_estimates:
        result['component_estimates'] = component_estimates
    return result
