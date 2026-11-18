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


def test_local_response_native_recipe_preserves_native_content():
    parent = {'training_profile': 'field_response',
              'thermal_field_response': {'variant': 'field_only', 'pairs': [['1', '2']]},
              'training': {'microbatch_cases': 24, 'effective_batch_size': 48},
              'model': {'depth': 3}, 'dataset': {'name': 'thermal'}}
    child = copy.deepcopy(parent)
    child.update(training_profile='local_multisecant_response',
                 thermal_local_response={'variant': 'multisecant_response', 'definition': 'shared.npz'})
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
