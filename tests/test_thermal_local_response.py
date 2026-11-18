import copy

import numpy as np
import torch

from channelthermal.training.thermal_local_response import (
    sample_thermal_neighborhood_observations, scheduled_thermal_neighborhoods,
    thermal_chart, thermal_compatibility_key,
)


def config():
    return {'domain': {'lx': 3., 'ly': 3., 'module_radius': .2},
            'flow': {'re': 100., 'u_in': 1., 'nu': None},
            'thermal': {'heat_power_min': .5, 'heat_power_max': 2.},
            'local_module': {'local_grid_size': 4, 'n_interface_points': 8},
            'layout': {'num_modules': 3, 'centers': [[.5, .7], [1.3, 1.], [2., 2.]],
                       'heat_powers': [.8, 1.2, 1.9]}}


def test_geometry_assignment_carries_every_heat_and_member_permutation():
    anchor = config()
    candidate = copy.deepcopy(anchor)
    candidate['layout']['centers'] = [[.6, .7], [1.4, 1.], [2.1, 2.]]
    candidate['layout']['heat_powers'] = [1.8, .7, 1.3]
    candidate['flow']['re'] = 130.
    chart, error, _, gap = thermal_chart(anchor, candidate, reynolds_range=180.)
    permutation = [2, 0, 1]
    permuted = copy.deepcopy(candidate)
    for key in ('centers', 'heat_powers'):
        permuted['layout'][key] = [candidate['layout'][key][i] for i in permutation]
    other, other_error, _, other_gap = thermal_chart(anchor, permuted, reynolds_range=180.)
    np.testing.assert_array_equal(chart, other)
    np.testing.assert_array_equal(error, other_error)
    assert gap > 0 and gap == other_gap
    assert np.count_nonzero(error) > 0
    assert len(chart) == 10


def test_new_compatibility_excludes_only_reviewed_local_fields():
    original = config()
    modified = copy.deepcopy(original)
    modified['flow']['re'] = 120.
    modified['local_module']['solver_iterations'] = 99
    assert thermal_compatibility_key(original) == thermal_compatibility_key(modified)
    modified['thermal']['fluid_k'] = 1.5
    assert thermal_compatibility_key(original) != thermal_compatibility_key(modified)
    modified = copy.deepcopy(original)
    modified['flow']['nu'] = 123.
    try:
        thermal_compatibility_key(modified)
        assert False
    except ValueError:
        pass


def test_common_fluid_fixed_mass_missing_channel_and_float64_coordinates():
    field = np.arange(3 * 3 * 5, dtype=np.float64).reshape(3, 3, 5)
    rows = np.array([[.500000000001, .5], [.500000000001, .5], [1., 1.], [2.5, 2.5]])
    groups = [dict(steady_field=field + i, module_mask=np.zeros((3, 3), bool),
                   sampled_points=rows, sampled_point_weights=np.array([1., 2., 3., 4.])) for i in range(4)]
    groups[1]['steady_field'] = groups[1]['steady_field'].copy()
    groups[1]['steady_field'][..., 2] = np.nan
    groups[2]['module_mask'][1, 1] = True
    result = sample_thermal_neighborhood_observations(groups, (3., 3.), seed=3, total_positions=12)
    assert result.eligible_counts == (3, 3, 3, 3)
    assert result.coordinates.dtype == torch.float64
    assert (result.coordinates[:, 0] == rows[0, 0]).any()
    assert not result.mask[:, 2].any()
    assert not result.masses[:, 2].any()
    assert not (result.coordinates == torch.tensor([1., 1.])).all(-1).any()
    torch.testing.assert_close(result.masses.sum(0)[[0, 1, 3, 4]], torch.ones(4, dtype=torch.float64))
    assert (result.coordinates == torch.tensor(rows[0])).all(-1).sum() == 8


def test_seeded_deck_has_no_dropped_tail():
    definitions = {'neighborhoods': [{'center_id': str(i)} for i in range(7)]}
    selected = [x['center_id'] for epoch in range(11, 18)
                for x in scheduled_thermal_neighborhoods(definitions, epoch, 4, 9)]
    assert all(len(set(selected[i:i+7])) == 7 for i in range(0, len(selected), 7))
    assert selected == [x['center_id'] for epoch in range(11, 18)
                        for x in scheduled_thermal_neighborhoods(definitions, epoch, 4, 9)]
