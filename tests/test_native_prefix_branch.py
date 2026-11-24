"""Real subprocess save, branch, saved-recipe reload, and resume checks."""
import csv
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from honf_runtime.run_layout import RunLayout


WORKER = Path(__file__).with_name('field_response_prefix_worker.py')


def _run(*args):
    root = WORKER.parent.parent
    subprocess.run([sys.executable, str(WORKER), *map(str, args)], check=True,
                   env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONPATH': os.pathsep.join(
                       (str(root / 'src'), str(root), os.environ.get('PYTHONPATH', '')))}, text=True)


def _checkpoint(path):
    return torch.load(RunLayout(path).read_path('latest_model.pt'), map_location='cpu', weights_only=False)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _equal(first, second)
    else:
        assert left == right


def _hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


def test_prefix_rejects_legacy_sampling_explicitly(tmp_path, monkeypatch):
    from honf_runtime import native_prefix_branch
    source = tmp_path / 'epoch_0010_model.pt'
    source.touch()
    monkeypatch.setattr(native_prefix_branch, 'load_trusted_checkpoint', lambda *args, **kwargs: {})
    engine = SimpleNamespace(config=SimpleNamespace(sampling_version='legacy'))
    with pytest.raises(ValueError, match='requires case_epoch_v1'):
        native_prefix_branch.initialize_native_prefix_child(source, tmp_path / 'child', {}, None, None, engine)


def test_observation_prefix_writes_initial_curve_metadata_to_metrics(tmp_path):
    """Branch from a synthetic saved prefix without running a training epoch."""
    from honf_runtime.native_prefix_branch import initialize_native_prefix_child
    from honf_runtime.scalar_history import VALIDATION_COLUMNS, append_scalar_rows, training_recording_definitions
    from honf_runtime.state_priority import recording_columns_and_definitions
    from honf_runtime.unified_training import (SamplingKey, TrainingEngine, _engine_config_payload,
                                              _optimizer_spec_payload, _sampling_dataset_id)
    from tests.field_response_prefix_worker import PrefixProvider
    from tests.test_unified_training import _config

    parent_recipe = {'training_profile': 'field_response',
                     'thermal_field_response': {'variant': 'field_only'},
                     'terms': {'zeta': {'weight': 1., 'unit': 'scaled', 'meaning': 'zeta error'},
                               'alpha': {'weight': .5, 'unit': 'scaled', 'meaning': 'alpha error'}}}
    child_recipe = {**parent_recipe, 'training_profile': 'observation_response',
                    'observation_response': {'variant': 'anchors'}}

    class ObservationProvider(PrefixProvider):
        def identity_payload(self):
            return {**super().identity_payload(),
                    'observation_response': {'validation_objective_complete': True}}

        def training_history_columns(self):
            return (*super().training_history_columns(), 'train_losses.observation_response/extra_absolute')

        def loss_metadata(self):
            return {**super().loss_metadata(), 'observation_response/extra_absolute':
                    {'unit': 'scaled', 'meaning': 'extra observation error'}}

        def validate_observation_calibration(self, calibration, *, source_checkpoint):
            assert source_checkpoint.name == 'epoch_0010_model.pt'
            return calibration

        def initialize_observation_response_state(self, native_state, checked):
            return {**native_state, 'observation_response': {'calibration': checked}}

    settings = replace(_config(), total_epochs=12, training_mode='joint',
                       sampling_version=SamplingKey.CASE_EPOCH_VERSION)
    engine = TrainingEngine(settings, device='cpu')
    model, parent_provider = nn.Linear(1, 1), PrefixProvider(parent_recipe)
    specs = parent_provider.optimizer_groups(model, 'adaptive_detail', engine.config.stage_for_epoch(11))
    schedules = [_optimizer_spec_payload(spec) for spec in specs]
    provider_identity = parent_provider.identity_payload()
    identity = {'workflow': 'prefix-toy', 'recipe': parent_recipe, 'provider_identity': provider_identity,
                'engine_config': _engine_config_payload(engine.config), 'selection_policy': asdict(engine.selection),
                'optimizer_schedule_contract': schedules, 'sampling_dataset_id': _sampling_dataset_id(provider_identity)}
    if engine.sampler_epoch_contract() is not None:
        identity['sampler_epoch_mapping'] = dict(engine.sampler_epoch_contract())

    base = ('epoch', 'phase', 'arm', 'temperature', 'case_visits', 'optimizer_updates',
            'microbatches', 'partial_update_cases', 'train_seconds', 'train_loss_total')
    columns, overrides = recording_columns_and_definitions(parent_provider)
    columns = (*base, *(f'learning_rates.{spec.name}' for spec in specs), *columns)
    definitions = training_recording_definitions(columns, parent_provider.loss_metadata(), overrides)
    parent = RunLayout(tmp_path / 'parent')
    definitions_path = parent.write_path('metric_definitions.json', category='configs')
    definitions_path.parent.mkdir(parents=True)
    definitions_path.write_text(json.dumps(definitions))
    rows = [{'epoch': epoch, 'phase': 'joint', 'arm': 'adaptive_detail', 'temperature': 1.,
             'case_visits': 5, 'optimizer_updates': 2, 'microbatches': 3, 'partial_update_cases': 1,
             'train_seconds': .1, 'train_loss_total': 1.5, 'learning_rates.predictor': .01,
             'train_losses.zeta': 1., 'train_losses.alpha': .5, 'work_counts.query_rows': 5}
            for epoch in range(1, 11)]
    append_scalar_rows(parent.write_path('history.csv'), columns, rows)
    append_scalar_rows(parent.write_path('validation.csv'), VALIDATION_COLUMNS,
                       [{'epoch': 10, 'partition': 'VALIDATION', 'metric': 'validation.field_score', 'value': .5}])
    source = parent.write_path('epoch_0010_model.pt')
    source.parent.mkdir(parents=True)
    torch.save({'epoch': 10, 'experiment_identity': identity, 'model_state_dict': model.state_dict(),
                'arm': 'adaptive_detail', 'optimizer_group_names': [spec.name for spec in specs],
                'optimizer_schedule_contract': schedules, 'best_field_score': .5, 'committed_validation_rows': 1,
                'provider_training_state': {'native': {'completed': 10, 'auxiliary_calls': 0},
                                            'field_response': {'calibration': None}},
                'sampler_state': {'completed_epoch': 10, 'next_epoch': 11, 'next_case_cursor': 0,
                                  'case_visits_this_epoch': 5, 'optimizer_updates_this_epoch': 2}}, source)
    parent_before = _hashes(parent.run_dir)

    child = tmp_path / 'child'
    initialize_native_prefix_child(source, child, child_recipe, nn.Linear(1, 1),
                                   ObservationProvider(child_recipe), engine,
                                   calibration={'source_epoch': 10})
    metadata = child / 'metrics/loss_curves_metadata.json'
    assert json.loads(metadata.read_text())['panel_definitions']['weighted_loss_totals']
    assert (child / 'plots/training/loss_curves.pdf').is_file()
    assert (child / 'plots/training/loss_curves.png').is_file()
    assert not (child / 'plots/training/loss_curves_metadata.json').exists()
    assert _hashes(parent.run_dir) == parent_before


def test_prefix_rejects_mismatched_earlier_visit_count(tmp_path):
    recipe = {'training_profile': 'field_response', 'thermal_field_response': {'variant': 'field_only'},
              'terms': {'zeta': {'weight': 1., 'unit': 'scaled', 'meaning': 'zeta error'},
                        'alpha': {'weight': .5, 'unit': 'scaled', 'meaning': 'alpha error'}}}
    recipe_file = tmp_path / 'recipe.json'
    recipe_file.write_text(json.dumps(recipe))
    parent = tmp_path / 'parent'
    _run('parent', recipe_file, parent, 10)
    history = RunLayout(parent).read_path('history.csv')
    with history.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    rows[0]['case_visits'] = str(int(rows[0]['case_visits']) + 1)
    with history.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    recipe['thermal_field_response']['variant'] = 'field_response'
    recipe_file.write_text(json.dumps(recipe))
    with pytest.raises(subprocess.CalledProcessError):
        _run('branch', recipe_file, tmp_path / 'child', RunLayout(parent).read_path('epoch_0010_model.pt'))


@pytest.mark.parametrize('variant', ['field_anchors', 'field_response'])
def test_prefix_children_and_fresh_process_resume(tmp_path, variant):
    recipe = {'training_profile': 'field_response', 'thermal_field_response': {'variant': 'field_only'},
              'terms': {'zeta': {'weight': 1., 'unit': 'scaled', 'meaning': 'zeta error'},
                        'alpha': {'weight': .5, 'unit': 'scaled', 'meaning': 'alpha error'}}}
    recipe_file = tmp_path / 'parent_recipe.json'
    recipe_file.write_text(json.dumps(recipe))
    parent = tmp_path / 'parent'
    _run('parent', recipe_file, parent, 12)
    parent_before = _hashes(parent)
    source = RunLayout(parent).read_path('epoch_0010_model.pt')
    recipe['thermal_field_response']['variant'] = variant
    child_recipe = tmp_path / 'child_recipe.json'
    child_recipe.write_text(json.dumps(recipe))
    control, resumed = tmp_path / 'control', tmp_path / 'resumed'
    for child in (control, resumed):
        _run('branch', child_recipe, child, source)
        layout = RunLayout(child)
        with layout.read_path('history.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row['epoch']) for row in rows] == list(range(1, 11))
        added = [name for name in rows[0] if name.startswith('train_losses.field_response/')]
        assert len(added) == (1 if variant == 'field_anchors' else 2)
        assert all(row[name] == '' for row in rows for name in added)
        assert torch.load(layout.read_path('best_by_field_mse_model.pt'), weights_only=False)['epoch'] == 10
    _run('fit', RunLayout(control).read_path('resolved_config.json'), control, 12)
    _run('fit', RunLayout(resumed).read_path('resolved_config.json'), resumed, 11)
    _run('fit', RunLayout(resumed).read_path('resolved_config.json'), resumed, 12)
    expected, actual = _checkpoint(control), _checkpoint(resumed)
    for key in ('model_state_dict', 'optimizer_state_by_name', 'rng_state', 'sampler_state',
                'provider_training_state', 'best_field_score'):
        _equal(expected[key], actual[key])
    assert actual['provider_training_state']['native'] == {'completed': 12, 'auxiliary_calls': 2}
    assert actual['provider_training_state']['field_response']['calibration']['source_epoch'] == 10
    layouts = [RunLayout(control), RunLayout(resumed)]
    assert [json.loads(layout.read_path('metric_definitions.json', category='configs').read_text()) for layout in layouts][0] == \
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
    assert _hashes(parent) == parent_before
