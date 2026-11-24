"""Wind native-volume observation metadata for the shared response runtime.

The source rows contain steady solutions in a wind-relative coordinate frame.
Pair only rows with the same declared wind direction so the saved frame has the
same meaning. Positions are a reproducible finite sample of the first row's
native centres within both centre boxes. Each saved velocity reference and
each model observation use the same explicit trilinear extraction from the
respective row's native centres. The finite candidate population is not a
claim of full-volume evaluation or an Earth-fixed turbine intervention.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np
import torch

from honf_runtime.observation_blocks import FieldComponent, LinearExtraction, ObservationBlock, QueryDependency

from ..geometry import cell_centre_weights
from .relation_anchored_task import build_wind_relation_anchored_task


_FRAME = 'wind_relative_downstream_crosswind_vertical_D'
_TIME = 'steady_final_solution'


def _groups(rows, directions):
    by_direction = defaultdict(list)
    for row in rows:
        by_direction[float(directions[int(row)])].append(str(int(row)))
    return tuple((values[index], values[index + 1])
                 for _, values in sorted(by_direction.items())
                 for index in range(0, len(values) - 1, 2))


def _common_positions(first, second, count, seed):
    """Select native first-scene centres inside both represented centre boxes."""
    axes_first = tuple(np.asarray(getattr(first, f'{axis}_m')) for axis in 'xyz')
    axes_second = tuple(np.asarray(getattr(second, f'{axis}_m')) for axis in 'xyz')
    eligible_axes = []
    for left, right in zip(axes_first, axes_second):
        if np.any(np.diff(left) <= 0) or np.any(np.diff(right) <= 0):
            raise ValueError('Wind native coordinate axes must increase strictly.')
        eligible_axes.append(np.flatnonzero((left >= right[0]) & (left <= right[-1])))
    sizes = tuple(len(indices) for indices in eligible_axes)
    population = int(np.prod(sizes, dtype=np.int64))
    if not population:
        raise ValueError('These Wind scenes have no common physical centre-box support.')
    chosen = np.random.default_rng(seed).choice(population, size=min(count, population), replace=False)
    # Native flattened field order is z, y, x, with x varying fastest.
    iz, iy, ix = np.unravel_index(chosen, sizes[::-1])
    indices = tuple(eligible_axes[axis][index] for axis, index in enumerate((ix, iy, iz)))
    coordinates_m = np.stack([axis[index] for axis, index in zip(axes_first, indices)], axis=-1)
    axis_weights = tuple(cell_centre_weights(axis) for axis in axes_first)
    masses_m3 = np.prod(np.stack([weight[index] for weight, index in zip(axis_weights, indices)], axis=-1), axis=-1)
    return coordinates_m, masses_m3


def _linear_extraction(run, positions_m, diameter_m, scene_id):
    """Return the explicit native-corner map and source-precision references."""
    axes = tuple(np.asarray(getattr(run, f'{axis}_m')) for axis in 'xyz')
    lower, upper, fraction = [], [], []
    for dimension, axis in enumerate(axes):
        q = np.asarray(positions_m[:, dimension], dtype=np.float64)
        values = np.asarray(axis, dtype=np.float64)
        if len(values) < 2 or np.any(q < values[0]) or np.any(q > values[-1]):
            raise ValueError('A Wind interpolation position lies outside its saved centre support.')
        high = np.searchsorted(values, q, side='right').clip(1, len(values) - 1)
        low = high - 1
        lower.append(low)
        upper.append(high)
        fraction.append((q - values[low]) / (values[high] - values[low]))
    n = len(positions_m)
    result = np.zeros((n, 3), dtype=np.float64)
    dependencies, lookup = [], {}
    dep_indices, observation_indices, component_indices, output_indices, coefficients = [], [], [], [], []
    nx, ny, _ = run.shape_nxyz
    for corner in product((0, 1), repeat=3):
        ix, iy, iz = [np.where(corner[dim], upper[dim], lower[dim]) for dim in range(3)]
        coefficient = np.prod(np.stack([fraction[dim] if corner[dim] else 1. - fraction[dim]
                                         for dim in range(3)]), axis=0)
        flat = (iz * ny + iy) * nx + ix
        active = np.flatnonzero(coefficient != 0.)
        result[active] += coefficient[active, None] * np.asarray(run.U[flat[active]], dtype=np.float64)
        for observation in active:
            native = int(flat[observation])
            if native not in lookup:
                lookup[native] = len(dependencies)
                coordinates = tuple(float(axes[dim][index[observation]]) / diameter_m
                                    for dim, index in enumerate((ix, iy, iz)))
                dependencies.append(QueryDependency(scene_id, _FRAME, _TIME, native, coordinates))
            for component in range(3):
                dep_indices.append(lookup[native]); observation_indices.append(int(observation))
                component_indices.append(component); output_indices.append(component)
                coefficients.append(float(coefficient[observation]))
    extraction = LinearExtraction(tuple(dependencies), np.asarray(dep_indices, dtype=np.int64),
        np.asarray(observation_indices, dtype=np.int64), np.asarray(component_indices, dtype=np.int64),
        np.asarray(output_indices, dtype=np.int64), np.asarray(coefficients, dtype=np.float64))
    return extraction, result


class WindObservationBinding:
    """Describe existing three-component native Wind observations and queries."""

    def __init__(self, provider, *, population_size=4096):
        if type(population_size) is not int or population_size < 1:
            raise ValueError('Wind observation population size must be positive.')
        self.provider = provider
        self.population_size = population_size
        self._train = set(map(int, provider.train_rows))
        self._validation = set(map(int, provider.validation_rows))
        if self._train & self._validation:
            raise ValueError('Wind observation populations must preserve their dataset split.')
        normalizer = provider.adapter.normalizer
        self.scales = np.asarray(normalizer.safe_std, dtype=np.float64) * float(normalizer.u_ref_mps)
        if self.scales.shape != (3,) or not np.isfinite(self.scales).all() or np.any(self.scales <= 0):
            raise ValueError('Wind observations need the existing TRAIN-fitted component scales.')

    def identity_payload(self):
        return {'schema': 'wind_volume_observations_v1', 'components': ['Ux', 'Uy', 'Uz'], 'units': 'm/s',
                'block_names': list(self.block_names()),
                'frame': _FRAME, 'time': _TIME, 'scales_mps': self.scales.tolist(),
                'pairing': 'same prescribed wind direction, different input layouts within one split',
                'population': 'input-only sampled first-scene native centres inside both centre boxes',
                'maximum_population_points': self.population_size,
                'measure': 'first-scene centre-box quadrature at the selected positions',
                'extraction': 'identical per-scene linear native-centre interpolation for references and predictions'}

    def block_names(self):
        return ('wind_velocity_common_volume',)

    def training_pairs(self):
        return _groups(sorted(self._train), self.provider.view.metadata['wd_deg'])

    def validation_pairs(self):
        return _groups(sorted(self._validation), self.provider.view.metadata['wd_deg'])

    def calibration_case_ids(self):
        return tuple(str(row) for row in sorted(self._train)[:15])

    def native_case_key(self, case_id):
        return int(case_id)

    def validate_pair(self, pair):
        if len(pair) != 2:
            raise ValueError('A Wind observation pair needs two source rows.')
        rows = tuple(map(int, pair))
        if rows[0] == rows[1] or not (set(rows) <= self._train or set(rows) <= self._validation):
            raise ValueError('Wind observation pairs must be distinct and stay within one declared split.')
        directions = self.provider.view.metadata['wd_deg']
        if float(directions[rows[0]]) != float(directions[rows[1]]):
            raise ValueError('Wind fixed-frame observations require the same prescribed wind direction.')

    def observation_blocks(self, pair):
        self.validate_pair(pair)
        pair = tuple(map(str, pair))
        runs = tuple(self.provider.view.volume.run(int(row)) for row in pair)
        diameter = float(self.provider.view.diameter_m)
        seed = np.random.SeedSequence([int(pair[0]), int(pair[1]), int(self.provider.seed)])
        coordinates_m, mass = _common_positions(*runs, self.population_size, seed)
        maps, references = zip(*[_linear_extraction(run, coordinates_m, diameter, scene)
                                 for run, scene in zip(runs, pair)])
        refs = np.stack(references)
        n = len(coordinates_m)
        valid = np.isfinite(refs)
        pair_mask = valid[0] & valid[1]
        mass = np.broadcast_to((mass / diameter ** 3)[:, None], (n, 3)).copy()
        components = tuple(FieldComponent(name, 'm/s', index, float(self.scales[index]), 1. / 3.)
                           for index, name in enumerate(('Ux', 'Uy', 'Uz')))
        return (ObservationBlock(name='wind_velocity_common_volume', components=components, scene_ids=pair,
            partition='train' if int(pair[0]) in self._train else 'validation', frame=_FRAME, time=_TIME,
            receiver_ids=tuple(f'common_position_{i}' for i in range(n)),
            coordinates=coordinates_m.astype(np.float64) / diameter,
            references=refs, absolute_mask=valid, pair_mask=pair_mask, edge_mask=pair_mask.copy(),
            absolute_measures=np.broadcast_to(mass, (2, n, 3)).copy(), pair_measures=mass,
            extractions=tuple(maps), neighbor_groups=('common_volume',) * n,
            source_times=tuple(run.source_time for run in runs)),)

    def field_request(self, model, case_id, dependencies):
        """Build one native scene/query and its existing physical field transform."""
        parameter = next(model.parameters())
        if not dependencies or any(str(dependency.scene_id) != str(case_id) for dependency in dependencies):
            raise ValueError('Wind field requests must contain dependencies from the declared scene.')
        case = self.provider.view.run(int(case_id))
        scene = model.make_scene(case)
        coordinates = torch.as_tensor([dependency.coordinates for dependency in dependencies],
                                      dtype=parameter.dtype, device=parameter.device)
        query_ids = torch.as_tensor([dependency.native_id for dependency in dependencies],
                                    dtype=torch.long, device=parameter.device)[None]
        query = model.make_query(scene, coordinates, query_ids=query_ids)
        scale = torch.as_tensor(self.scales, dtype=parameter.dtype, device=parameter.device)
        offset = model.transforms.profile_at(query.coordinates)
        return scene, query, scale, offset


def build_observation_capability(recipe, *, device, resume_payload=None):
    del resume_payload  # TrainingEngine restores the model and provider state.
    model, provider = build_wind_relation_anchored_task(recipe, device)
    size = recipe.get('observation_response', {}).get('wind_population_size', 4096)
    return model, provider, WindObservationBinding(provider, population_size=size)
