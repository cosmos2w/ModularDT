"""Real subprocess save, branch, saved-recipe reload, and resume checks."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
