"""Focused entry-point and native-prefix contracts for local response."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from honf_runtime.native_prefix_branch import _native_recipe
from tools.relation_anchored_train import build
from tests.test_native_prefix_branch import _checkpoint, _equal
from honf_runtime.run_layout import RunLayout


@pytest.mark.parametrize('variant', ['multisecant_response', 'targeted_reynolds_response',
                                    'trace_matched_uniform_response'])
def test_local_response_native_recipe_preserves_native_content(variant):
    parent = {'training_profile': 'field_response',
              'thermal_field_response': {'variant': 'field_only', 'pairs': [['1', '2']]},
              'training': {'microbatch_cases': 24, 'effective_batch_size': 48},
              'model': {'depth': 3}, 'dataset': {'name': 'thermal'}}
    child = copy.deepcopy(parent)
    child.update(training_profile='local_multisecant_response',
                 thermal_local_response={'variant': variant, 'definition': 'shared.npz',
                                         'state_priority': True})
    assert _native_recipe(parent) == _native_recipe(child)
    child['model']['depth'] = 4
    assert _native_recipe(parent) != _native_recipe(child)


def test_local_response_dispatches_to_thermal_adapter(monkeypatch):
    module_name = 'channelthermal.training.thermal_local_response'
    module = types.ModuleType(module_name)
    observed = {}
    def adapter(recipe, *, device, resume_payload):
        observed.update(recipe=recipe, device=device, resume_payload=resume_payload)
        return 'model', 'provider'
    module.build_thermal_local_response_task = adapter
    monkeypatch.setitem(sys.modules, module_name, module)
    recipe = {'dataset': {'name': 'thermal'}, 'initialization': {'seed': 1},
              'training_profile': 'local_multisecant_response'}
    checkpoint = {'epoch': 10}
    assert build(recipe, device='cpu', resume_payload=checkpoint) == ('model', 'provider')
    assert observed == {'recipe': recipe, 'device': 'cpu', 'resume_payload': checkpoint}
    recipe['dataset'] = 'wind'
    recipe['seed'] = 1
    with pytest.raises(ValueError, match='Thermal adapter'):
        build(recipe, device='cpu')


def test_local_response_curve_labels_from_producer_identity(tmp_path, monkeypatch):
    from honf_runtime.unified_training import _curve_provider_identity, _render_loss_curves
    from tests.test_unified_loss_curves import _capture_figures
    identity = {'local_response': {'variant': 'multisecant_response'}}
    assert _curve_provider_identity(identity) == identity
    assert _curve_provider_identity({'dataset': 'thermal'}) is None
    captured = _capture_figures(monkeypatch, tmp_path)
    history = [{'epoch': 11, 'phase': 'joint', 'arm': 'adaptive_detail', 'case_visits': 150,
                'train_losses': {'flow/u': 1., 'local_response/response': .2},
                'validation': {'field_score': .5},
                'validation_objective': {'terms': {'flow/u': .8}}}]
    _render_loss_curves(history, tmp_path, 'field_score',
                        provider_identity=_curve_provider_identity(identity), separate_totals=True)
    for figure in captured:
        assert 'Native-only validation subtotal' in figure['titles']
        assert any('VALIDATION: native-only subtotal' in labels for labels in figure['labels'])
        assert any('Full validation objective unavailable' in texts for texts in figure['texts'])
    metadata = json.loads((tmp_path / 'loss_curves_metadata.json').read_text())
    assert 'native-only subtotal' in metadata['panel_definitions']['weighted_loss_totals']


def test_fresh_process_local_response_resume(tmp_path):
    """One ten-epoch CPU prefix and eight new response-active fixture epochs."""
    root = Path(__file__).resolve().parents[1]
    worker = root / 'tests/local_response_prefix_worker.py'
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONPATH': os.pathsep.join(
        (str(root / 'src'), str(root), os.environ.get('PYTHONPATH', '')))}
    def run(*args):
        subprocess.run([sys.executable, str(worker), *map(str, args)], check=True, env=env)
    recipe = {'training_profile': 'field_response', 'thermal_field_response': {'variant': 'field_only'},
              'terms': {'zeta': {'weight': 1., 'unit': 'scaled', 'meaning': 'zeta error'},
                        'alpha': {'weight': .5, 'unit': 'scaled', 'meaning': 'alpha error'}}}
    recipe_file = tmp_path / 'parent.json'
    recipe_file.write_text(json.dumps(recipe))
    parent = tmp_path / 'parent'
    run('parent', recipe_file, parent, 10)
    source = RunLayout(parent).read_path('epoch_0010_model.pt')
    for variant in ('uniform_response', 'multisecant_response'):
        child_recipe = {**recipe, 'training_profile': 'local_multisecant_response',
                        'thermal_local_response': {'variant': variant, 'input_definitions': 'shared'}}
        child_file = tmp_path / (variant + '.json')
        child_file.write_text(json.dumps(child_recipe))
        control, resumed = tmp_path / (variant + '_control'), tmp_path / (variant + '_resumed')
        for path in (control, resumed):
            run('branch', child_file, path, source)
        run('fit', RunLayout(control).read_path('resolved_config.json'), control, 12)
        run('fit', RunLayout(resumed).read_path('resolved_config.json'), resumed, 11)
        run('fit', RunLayout(resumed).read_path('resolved_config.json'), resumed, 12)
        expected, actual = _checkpoint(control), _checkpoint(resumed)
        for key in ('model_state_dict', 'optimizer_state_by_name', 'rng_state', 'sampler_state',
                    'provider_training_state', 'best_field_score'):
            _equal(expected[key], actual[key])
        assert len(actual['provider_training_state']['local_response']['episodes']) == 2
        layouts = [RunLayout(control), RunLayout(resumed)]
        assert json.loads(layouts[0].read_path('metric_definitions.json', category='configs').read_text()) == \
               json.loads(layouts[1].read_path('metric_definitions.json', category='configs').read_text())
        histories = []
        for layout in layouts:
            with layout.read_path('history.csv').open(newline='') as stream:
                reader = csv.DictReader(stream)
                histories.append((reader.fieldnames, list(reader)))
        assert histories[0][0] == histories[1][0]
        for left, right in zip(histories[0][1], histories[1][1]):
            assert {k: v for k, v in left.items() if k != 'train_seconds'} == \
                   {k: v for k, v in right.items() if k != 'train_seconds'}


def test_fresh_process_priority_multiplier_native_prefix_resume(tmp_path):
    """Ten native epochs and four response-active fixture epochs, across real processes."""
    root = Path(__file__).resolve().parents[1]
    worker = root / 'tests/local_response_prefix_worker.py'
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONPATH': os.pathsep.join(
        (str(root / 'src'), str(root), os.environ.get('PYTHONPATH', '')))}

    def run(*args):
        subprocess.run([sys.executable, str(worker), *map(str, args)], check=True, env=env)

    recipe = {'training_profile': 'field_response', 'thermal_field_response': {'variant': 'field_only'},
              'terms': {'zeta': {'weight': 1., 'unit': 'scaled', 'meaning': 'zeta error'},
                        'alpha': {'weight': .5, 'unit': 'scaled', 'meaning': 'alpha error'}}}
    recipe_file = tmp_path / 'parent.json'
    recipe_file.write_text(json.dumps(recipe))
    parent = tmp_path / 'parent'
    run('parent', recipe_file, parent, 10)
    source = RunLayout(parent).read_path('epoch_0010_model.pt')
    child = {**recipe, 'training_profile': 'local_multisecant_response',
             'thermal_local_response': {'variant': 'uniform_response', 'input_definitions': 'shared',
                                        'state_priority': True, 'objective': {'response_weight': .5}}}
    child_file = tmp_path / 'child.json'
    child_file.write_text(json.dumps(child))
    uninterrupted, resumed = tmp_path / 'uninterrupted', tmp_path / 'resumed'
    for directory in (uninterrupted, resumed):
        run('branch', child_file, directory, source)
        saved_recipe = json.loads(RunLayout(directory).read_path('resolved_config.json').read_text())
        assert saved_recipe['thermal_local_response'] == child['thermal_local_response']
    run('fit', RunLayout(uninterrupted).read_path('resolved_config.json'), uninterrupted, 12)
    run('fit', RunLayout(resumed).read_path('resolved_config.json'), resumed, 11)
    run('fit', RunLayout(resumed).read_path('resolved_config.json'), resumed, 12)
    expected, actual = _checkpoint(uninterrupted), _checkpoint(resumed)
    for key in ('model_state_dict', 'optimizer_state_by_name', 'rng_state', 'sampler_state',
                'provider_training_state', 'best_field_score'):
        _equal(expected[key], actual[key])
    assert len(actual['provider_training_state']['local_response']['episodes']) == 2
    layouts = [RunLayout(uninterrupted), RunLayout(resumed)]
    definitions = [json.loads(layout.read_path('metric_definitions.json', category='configs').read_text())
                   for layout in layouts]
    assert definitions[0] == definitions[1]
    assert definitions[0]['training_loss_order'] == ['zeta', 'alpha', 'local_response/absolute',
                                                      'local_response/response']
    assert definitions[0]['training_total_order'] == definitions[0]['training_loss_order']
    assert 'work_counts.state_priority_active_count' in definitions[0]['training_columns']
    histories = []
    for layout in layouts:
        with layout.read_path('history.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            histories.append((reader.fieldnames, list(reader)))
    assert histories[0][0] == histories[1][0]
    assert histories[0][0] == definitions[0]['training_columns']
    assert len(histories[0][1]) == len(histories[1][1])
    assert all(float(row['work_counts.state_priority_active_count']) > 0 for row in histories[1][1][-2:])
    for left, right in zip(histories[0][1], histories[1][1]):
        assert {k: v for k, v in left.items() if k != 'train_seconds'} == \
               {k: v for k, v in right.items() if k != 'train_seconds'}


def test_local_response_curve_metadata_refreshes_both_locations(tmp_path):
    from honf_runtime.unified_training import _render_loss_curves

    plots, metrics = tmp_path / 'plots', tmp_path / 'metrics'
    identity = {'local_response': {'variant': 'multisecant_response'}}
    terms = {'local_response/response': {'panel': 'physics', 'label': 'Local response',
                                         'formula': 'saved response definition', 'weight': 'calibrated',
                                         'unit': 'scaled squared error'}}
    first = {'epoch': 10, 'phase': 'joint', 'arm': 'R13-8-A-EAnchor',
             'train_losses': {'flow/u': 1.},
             'validation_objective': {'terms': {'flow/u': 1.}, 'term_weights': {'flow/u': 1.}}}
    second = {'epoch': 11, 'phase': 'joint', 'arm': 'R13-8-A-EAnchor',
              'train_losses': {'flow/u': 0.9, 'local_response/response': 0.2},
              'validation_objective': {'terms': {'flow/u': 0.8}, 'term_weights': {'flow/u': 0.75}}}
    _render_loss_curves([first], plots, 'field_score', terms, identity, separate_totals=True)
    _render_loss_curves([first, second], plots, 'field_score', terms, identity,
                        metadata_output=metrics, separate_totals=True)
    plot_meta = json.loads((plots / 'loss_curves_metadata.json').read_text())
    metric_meta = json.loads((metrics / 'loss_curves_metadata.json').read_text())
    assert plot_meta == metric_meta
    assert plot_meta['term_definitions']['local_response/response'] == terms['local_response/response']
    assert plot_meta['validation_objective_weights_by_epoch']['flow/u'] == {'10': 1.0, '11': 0.75}
    assert 'local-neighborhood' in plot_meta['panel_definitions']['physics']
    assert 'No separately fitted' not in plot_meta['panel_definitions']['physics']
    assert 'native-only subtotal' in plot_meta['panel_definitions']['weighted_loss_totals']
