"""Finite review uses existing decoded values and input-defined row panels."""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

import finite_observation_evaluate as review
from honf_runtime.finite_observations import AffineFrame, FiniteObservationFamily, FiniteObservationScene
from honf_runtime.observation_blocks import FieldComponent, LinearExtraction, QueryDependency


def _scene(case_id, references):
    deps = tuple(QueryDependency(case_id, 'physical', 'steady', index, coordinate)
                 for index, coordinate in enumerate(((.5, .5), (1.5, 1.5))))
    extraction = LinearExtraction(deps, np.array([0, 1]), np.array([0, 1]), np.array([0, 0]),
                                  np.array([6, 6]), np.array([1., 1.]))
    family = FiniteObservationFamily(name='temperature', component=FieldComponent('temperature', 'K', 6, 2., 1.),
        support='observed material', frame='physical', time='steady', extraction_id='direct', native_weight=1.,
        row_ids=('first-row', 'second-row'), coordinates=np.array([[.5, .5], [1.5, 1.5]], dtype=np.float64),
        measures=np.array([1., 3.]), references=np.asarray(references, dtype=np.float64), extraction=extraction)
    return FiniteObservationScene(case_id, (family,))


def _lookup(values, missing=()):
    def get(dependencies):
        if any(dep.native_id in missing for dep in dependencies):
            return None
        raw = torch.zeros((1, len(dependencies), 7), dtype=torch.float64)
        raw[0, :, 6] = torch.tensor([values[dep.native_id] for dep in dependencies], dtype=torch.float64)
        return raw
    return get


def test_generic_reduction_retains_weighted_reference_prediction_and_unavailable_support():
    scene = _scene('first', [2., 4.])
    weights = {'temperature': 1., 'unavailable': 3.}
    frame = AffineFrame('physical', (0., 0.), (2., 2.))
    result = review.summarize_decoded_scene(scene, _lookup([4., 4.]), configured_weights=weights,
                                             frames={'physical': frame})
    family = result['families']['temperature']
    assert family['normalized_point_mse'] == [pytest.approx(.25)]
    assert result['normalized_weighted_anchor_mse'] == [pytest.approx(.0625)]
    np.testing.assert_allclose(family['normalized_residual_moments'], [[.25, -.125, -.125]])
    np.testing.assert_allclose(family['normalized_predicted_moments'], [[2., .5, .5]])
    np.testing.assert_allclose(family['normalized_reference_moments'], [[1.75, .625, .625]])
    assert family['source_row_ids'] == ['first-row', 'second-row']
    assert family['source_references'] == [2., 4.] and family['reference_dtype'] == 'float64'
    assert result['missing_families'] == ['unavailable']
    missing = review.summarize_decoded_scene(scene, _lookup([4., 4.], missing={1}),
                                              configured_weights=weights, frames={'physical': frame})
    assert missing['families']['temperature']['available'] is False
    assert missing['normalized_weighted_anchor_mse'] is None


def test_ragged_scene_changes_compare_moments_without_row_correspondence():
    first = _scene('first', [2., 4.])
    second = _scene('second', [4., 6.])
    second_family = replace(second.families[0], row_ids=('unrelated-third', 'unrelated-fourth'))
    second = FiniteObservationScene('second', (second_family,))
    weights = {'temperature': 1.}
    frame = AffineFrame('physical', (0., 0.), (2., 2.))
    a = review.summarize_decoded_scene(first, _lookup([4., 4.]), configured_weights=weights, frames={'physical': frame})
    b = review.summarize_decoded_scene(second, _lookup([6., 6.]), configured_weights=weights, frames={'physical': frame})
    pair = review.summarize_scene_changes(a, b, configured_weights=weights)
    np.testing.assert_allclose(pair['normalized_weighted_change_mse'], [0.], atol=1.e-15)
    np.testing.assert_allclose(pair['families']['temperature']['normalized_predicted_change'], [[1., .25, .25]])
    assert pair['families']['temperature']['physical_reference_change'][0][0] == pytest.approx(2.)
    b['families']['temperature']['unit'] = 'different unit'
    with pytest.raises(ValueError, match='Incompatible'):
        review.summarize_scene_changes(a, b, configured_weights=weights)


def test_thermal_decoded_lookup_uses_only_available_native_cells():
    model = SimpleNamespace(nx=2, ny=2)
    batch = SimpleNamespace(receivers=SimpleNamespace(fluid_xy=torch.tensor([[[.5, .5], [1.5, .5], [.5, 1.5]]])))
    values = torch.tensor([[[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.]]])
    prediction = SimpleNamespace(native_main={'pred_field': values}, native_prepared=SimpleNamespace(
        stencils={'fluid': SimpleNamespace(valid=torch.tensor([[True, False, True]]))}, lengths=torch.tensor([[2., 2.]])))
    decoded = review.ThermalDecodedFields(model, prediction, batch)
    first = QueryDependency('first', 'physical', 'steady', 0, (.5, .5))
    hidden = QueryDependency('first', 'physical', 'steady', 1, (1.5, .5))
    absent = QueryDependency('first', 'physical', 'steady', 3, (1.5, 1.5))
    torch.testing.assert_close(decoded((first,)), torch.tensor([[[1., 2., 3., 5.]]]))
    assert decoded((hidden,)) is None and decoded((absent,)) is None


def _saved_normalization_example():
    stats = {'field_mean_by_channel': [.1, .2, .3, .4, .5],
             'field_std_by_channel': [1., 2., 3., 4., 5.],
             'internal_temperature_std': [6.], 'interface_targets_std': [7., 8.]}
    model = SimpleNamespace(field_mean=torch.tensor(stats['field_mean_by_channel']),
        field_std=torch.tensor(stats['field_std_by_channel']),
        internal_temperature_std=torch.tensor(stats['internal_temperature_std']),
        interface_target_std=torch.tensor(stats['interface_targets_std']))
    saved = {'experiment_identity': {'provider_identity': {
        'adapter_state': {'normalization_stats': stats},
        'finite_observations': {'family_weights': {'flow/u': 1., 'port/outside_temperature': .05},
                                'frame': {'identity': 'test_frame', 'lower': [0., 0.], 'upper': [1., 1.]}}}}}
    return model, saved


def test_saved_identity_checks_weights_normalization_and_emitted_family_scale():
    model, saved = _saved_normalization_example()
    binding = SimpleNamespace(provider=SimpleNamespace(model=model),
        finite_observation_family_weights=lambda: {'flow/u': 1., 'port/outside_temperature': .05},
        finite_observation_frame=lambda: AffineFrame('test_frame', (0., 0.), (1., 1.)))
    checked = review.verify_saved_thermal_observation_identity(binding, saved)
    assert checked['family_scales']['temperature/material'] == 6.
    assert checked['family_scales']['q_proxy'] == 8.
    assert checked['selected_row_and_extraction_identity'] == 'not compared'
    saved['experiment_identity']['provider_identity']['finite_observations']['family_weights']['flow/u'] = 2.
    with pytest.raises(ValueError, match='family weights'):
        review.verify_saved_thermal_observation_identity(binding, saved)
    saved['experiment_identity']['provider_identity']['finite_observations']['family_weights']['flow/u'] = 1.
    saved_frame = saved['experiment_identity']['provider_identity']['finite_observations']['frame']
    saved_frame['upper'] = [2., 1.]
    with pytest.raises(ValueError, match='frame'):
        review.verify_saved_thermal_observation_identity(binding, saved)
    saved_frame['upper'] = [1., 1.]
    model.field_mean[0] += 1.
    with pytest.raises(ValueError, match='field_mean'):
        review.verify_saved_thermal_observation_identity(binding, saved)
    observer = object.__new__(review.ThermalFiniteObservationObserver)
    observer.expected_family_scales = {'temperature': 3.}
    with pytest.raises(ValueError, match='source scale'):
        observer._validate_family_scales(_scene('first', [2., 4.]))
    observer.expected_family_scales = {'temperature': 2.}
    observer._validate_family_scales(_scene('first', [2., 4.]))


def test_cli_reuses_one_existing_review_per_reader_and_preserves_declared_panel(monkeypatch, tmp_path):
    import channelthermal.training.native_checkpoint as checkpoint_module
    from channelthermal.training.observation_response import ThermalObservationBinding
    import honf_runtime.compat as compat
    import thermal_field_response_evaluate as native_review

    cases = tuple(f'case_{index:03d}' for index in range(150))
    chosen = cases[:16]
    pairs = tuple((chosen[index], chosen[index+1]) for index in range(0, 16, 2))
    panel = tmp_path / 'panels.json'
    panel.write_text(json.dumps({'development': {'train_case_ids': chosen, 'train_pairs': pairs}}))
    model, saved = _saved_normalization_example()
    saved['epoch'] = 100
    saved['experiment_identity']['recipe'] = {}
    saved['experiment_identity']['provider_identity']['finite_observations'].update(
        {'probe_count': 64, 'probe_seed': 130013, 'basis': 'tensor_hats'})
    model.core, model.eval = SimpleNamespace(query_tile_size=0), lambda: None
    provider = SimpleNamespace(training_case_ids=cases, validation_case_ids=(), data_path='/saved/native.h5',
        case_split_by_id={case: 'train' for case in cases}, recipe={'evaluation': {}}, model=model,
        native_weights={'flow/u': 1.}, outside_temperature_weight=.05)
    monkeypatch.setattr(review, '_verify_review_storage', lambda output: {'free_bytes_before': 1})
    monkeypatch.setattr(compat, 'load_trusted_checkpoint', lambda *args, **kwargs: saved)
    monkeypatch.setattr(checkpoint_module, 'build_native_checkpoint_inference', lambda *args, **kwargs: (model, provider))
    monkeypatch.setattr(ThermalObservationBinding, 'finite_observation_frame',
                        lambda self: AffineFrame('test_frame', (0., 0.), (1., 1.)))
    class Observer:
        def __init__(self, binding, actual_model, actual_cases, actual_pairs, **kwargs):
            assert actual_model is model and actual_cases == chosen and actual_pairs == pairs
            assert kwargs['basis'] == 'tensor_hats'
            assert kwargs['expected_family_scales']['temperature/fluid'] == 5.
        def __call__(self, *args): pass
        def finish(self): return {'received': True}
    monkeypatch.setattr(review, 'ThermalFiniteObservationObserver', Observer)
    calls = []
    def evaluate(model_arg, provider_arg, **kwargs):
        assert model_arg is model and provider_arg is provider and callable(kwargs['case_observer'])
        assert kwargs['include_thermal_functionals'] is True
        assert kwargs['local_contrast_diagnostics'] is True
        assert kwargs['include_triple_contrasts'] is False
        assert kwargs['case_ids'] == chosen and kwargs['pairs'] == pairs
        calls.append(kwargs['mode'])
        return {'field_summary': True}
    monkeypatch.setattr(native_review, 'evaluate_field_response_panel', evaluate)
    written = {}
    monkeypatch.setattr(review, '_atomic_json', lambda path, payload: written.update(payload))
    assert review.main(['--checkpoint', str(tmp_path / 'model.pt'), '--panels', str(panel),
                        '--stage', 'development', '--output', str(tmp_path / 'out.json')]) == 0
    assert calls == ['full', 'restricted']
    assert written['case_count'] == 16 and len(written['pairs']) == 8
    assert written['saved_training_probe_recipe_matches_fixed'] and written['saved_training_basis_matches_review']
    assert written['saved_observation_identity_verification']['selected_row_and_extraction_identity'] == 'not compared'
    assert set(written['views']) == {'full', 'restricted'}
    with pytest.raises(ValueError, match='bounded'):
        review._validate_train_panel(cases[:16], (), cases)
