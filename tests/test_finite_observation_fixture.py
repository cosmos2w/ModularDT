"""The non-Thermal finite-only fixture has a real independent native sampler."""
from unittest.mock import patch

import numpy as np
import torch

from honf_runtime.finite_observations import compatible_scene_families
from honf_runtime.observation_response_fixture import build_observation_capability
from honf_runtime.unified_training import SamplingKey


def _fixture():
    recipe = {'dataset': 'analytic_fixture', 'seed': 7, 'architecture': 'relation_anchored', 'total_epochs': 2,
              'fixture': {'output_dim': 7, 'finite_only_binding': True,
                          'native_sampling_law': 'weighted_with_replacement', 'native_draw_count': 8}}
    return build_observation_capability(recipe, device='cpu')


def test_finite_fixture_covers_ragged_nonmatching_rows_and_absent_family():
    _, _, binding = _fixture()
    assert not hasattr(binding, 'coupled_native_population')
    assert not hasattr(binding, 'coupled_native_sampling_law')
    frame = binding.finite_observation_frame()
    assert frame.identity == 'analytic_fixture_xyz' and frame.lower == (0., 0., 0.) and frame.upper == (1., 1., 1.)
    weights = binding.finite_observation_family_weights()
    assert weights == {f'field_{index}': float(index + 1) for index in range(7)}
    first = binding.finite_observation_scene('fixture_0', count=64, seed=4)
    second = binding.finite_observation_scene('fixture_1', count=64, seed=4)
    assert len(first.families) == 7 and len(second.families) == 6
    assert len(first.families[0].row_ids) == 4 and len(second.families[0].row_ids) == 5
    assert set(first.families[0].row_ids).isdisjoint(second.families[0].row_ids)
    assert first.families[0].row_ids[0] != first.families[0].row_ids[1]
    np.testing.assert_array_equal(first.families[0].coordinates[0], first.families[0].coordinates[1])
    assert np.all(first.families[0].measures > 0) and first.families[0].measures[0] != first.families[0].measures[1]
    assert all(family.component.output_index == index for index, family in enumerate(first.families))
    eligible = compatible_scene_families(first, second, configured_weights=weights)
    assert eligible.eligible_families == tuple(f'field_{index}' for index in range(6))
    assert eligible.missing_families == ('field_6',)
    # An independent closed-form value, from the first scene and first field.
    point = first.families[0].coordinates[0]
    expected = np.sin(point[0] + .15 * point[1]) + .1 * point[1] * point[2] + .15 * (point[0] + point[2])
    assert first.families[0].references[0] == expected
    selected = binding.finite_observation_scene('fixture_0', count=2, seed=41)
    replay = binding.finite_observation_scene('fixture_0', count=2, seed=41)
    assert [family.row_ids for family in selected.families] == [family.row_ids for family in replay.families]
    assert selected.families[0].row_ids == selected.families[2].row_ids
    assert all(len(family.row_ids) == 2 for family in selected.families)


def test_native_sampler_is_weighted_with_replacement_and_independent_of_binding():
    model, provider, _ = _fixture()
    key = SamplingKey(provider.seed, 1, 0, 0, 'joint', model.architecture,
                      sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=provider.sampling_dataset_id)
    assert not hasattr(provider, 'sample_coupled_native_rows')
    assert provider.identity_payload()['native_sampling_law'] == 'weighted_with_replacement'
    first = provider.make_batch(('fixture_0', 'fixture_1'), key)
    repeat = provider.make_batch(('fixture_0', 'fixture_1'), key)
    assert not first.auxiliary
    for actual, replay in zip(first.receivers, repeat.receivers):
        np.testing.assert_array_equal(actual, replay)
        assert len(actual) == 8 and len(np.unique(actual, axis=0)) < len(actual)
    # The documented increasing source-row weights are used directly by the
    # keyed NumPy sampler, with repeated indices retained in the native batch.
    generator = key.numpy_rng('fixture_native_weighted_rows', 'fixture_0')
    expected_indices = generator.choice(4, 8, replace=True, p=np.array([.1, .2, .3, .4]))
    all_rows = np.array([[.1 + .11 * index, .15 + .06 * (index % 3), .2 + .05 * (index % 2)]
                         for index in range(4)])
    np.testing.assert_allclose(first.receivers[0], all_rows[expected_indices])


def test_fixture_prefers_finite_reader_then_falls_back_to_historical_reader():
    torch.set_num_threads(1)
    model, provider, _ = _fixture()
    key = SamplingKey(provider.seed, 1, 0, 0, 'joint', model.architecture)
    batch = provider.make_batch(('fixture_0',), key)
    model.train()
    provider.finite_observation_reader_mode = 'full'
    provider.coupled_native_reader_mode = 'restricted'
    with patch.object(model, 'predict', wraps=model.predict) as predict:
        fields, _ = provider.predict_native(model, batch.scene_inputs, batch.receivers, model.architecture,
                                             'joint', epoch=1, temperature=1.)
        assert predict.call_args.kwargs['read_mode'] == 'full' and fields[0].shape[0] == 1
        provider.finite_observation_reader_mode = None
        fields, _ = provider.predict_native(model, batch.scene_inputs, batch.receivers, model.architecture,
                                             'joint', epoch=1, temperature=1.)
        assert predict.call_args.kwargs['read_mode'] == 'restricted' and fields[0].shape[0] == 1
        model.eval()
        fields, _ = provider.predict_native(model, batch.scene_inputs, batch.receivers, model.architecture,
                                             'joint', epoch=1, temperature=1.)
        assert predict.call_args.kwargs['read_mode'] is None and fields[0].shape[0] == 2
