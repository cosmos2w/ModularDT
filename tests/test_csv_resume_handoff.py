"""Real cross-interpreter handoff preserves the saved CSV and training state."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


WORKER = Path(__file__).with_name('csv_resume_handoff_worker.py')
ROOT = WORKER.parents[1]


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_csv_resume_across_interpreters_preserves_saved_schema_and_loss_order(tmp_path):
    config = {'terms': {
        'zeta': {'weight': 0.7, 'unit': 'scaled squared error', 'meaning': 'Original residual'},
        'middle': {'weight': 0.2, 'unit': 'scaled squared error', 'meaning': 'Offset residual'},
        'alpha': {'weight': 0.1, 'unit': 'scaled squared error', 'meaning': 'Shifted residual'},
    }}
    config_path = tmp_path / 'recipe.json'
    config_path.write_text(json.dumps(config))
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT), str(ROOT / 'src'),
                                         str(ROOT / 'Case_ThermalChannel/src'),
                                         env.get('PYTHONPATH', '')))
    def run(output, stop, resume='-', recipe_path=config_path):
        subprocess.run([sys.executable, str(WORKER), str(recipe_path), str(output), str(stop), str(resume)],
                       cwd=ROOT, env=env, check=True, capture_output=True, text=True)

    complete, resumed = tmp_path / 'complete', tmp_path / 'resumed'
    run(complete, 4)
    run(resumed, 2)
    history = resumed / 'metrics/history.csv'
    definitions_path = resumed / 'configs/metric_definitions.json'
    prefix, definition_bytes = history.read_bytes(), definitions_path.read_bytes()
    definitions = json.loads(definition_bytes)
    with history.open(newline='') as f:
        header = next(csv.reader(f))
    assert header == definitions['training_columns']
    assert definitions['training_loss_order'] == ['zeta', 'middle', 'alpha']
    for name, spec in config['terms'].items():
        column = definitions['training_column_definitions']['train_losses.' + name]
        assert column['dtype'] == 'float'
        assert column['unit'] == spec['unit']
        assert column['meaning'] == spec['meaning']
    query = definitions['training_column_definitions']['work_counts.query_rows']
    assert query == {'dtype': 'float', 'unit': 'rows', 'meaning': 'Native query rows processed in the epoch'}
    assert all(set(entry) >= {'dtype', 'unit', 'meaning'}
               for entry in definitions['training_column_definitions'].values())

    # This preserves values but changes mapping insertion order in a new interpreter.
    saved_config = resumed / 'configs/resolved_config.json'
    assert json.loads(saved_config.read_text()) == config
    saved_config.write_text(json.dumps(json.loads(saved_config.read_text()), sort_keys=True, indent=4) + '\n')
    assert list(json.loads(saved_config.read_text())['terms']) == ['alpha', 'middle', 'zeta']
    run(resumed, 4, resumed / 'checkpoints/latest_model.pt', recipe_path=saved_config)
    assert history.read_bytes().startswith(prefix)
    assert definitions_path.read_bytes() == definition_bytes
    with history.open(newline='') as f:
        assert next(csv.reader(f)) == header
    def checkpoint(directory):
        return torch.load(directory / 'checkpoints/latest_model.pt', map_location='cpu', weights_only=False)
    left, right = checkpoint(complete), checkpoint(resumed)
    for key in ('model_state_dict', 'optimizer_state_by_name', 'rng_state', 'sampler_state',
                'provider_training_state'):
        assert_tree_equal(left[key], right[key])
    with (complete / 'metrics/history.csv').open(newline='') as f:
        full_rows = list(csv.DictReader(f))
    with history.open(newline='') as f:
        resumed_rows = list(csv.DictReader(f))
    scalar_columns = ['train_loss_total'] + ['train_losses.' + name for name in config['terms']]
    assert len(full_rows) == len(resumed_rows) == 4
    assert [[row[name] for name in scalar_columns] for row in full_rows] == [
        [row[name] for name in scalar_columns] for row in resumed_rows]

    with (complete / 'metrics/validation.csv').open(newline='') as f:
        full_validation = list(csv.DictReader(f))
    with (resumed / 'metrics/validation.csv').open(newline='') as f:
        resumed_validation = list(csv.DictReader(f))
    assert full_validation
    # Validation stores independent named values as rows; their physical row
    # order can follow mapping order, while each saved numeric value is exact.
    def validation_values(rows):
        return {tuple((key, value) for key, value in row.items() if key != 'value'): row['value']
                for row in rows}
    assert len(validation_values(full_validation)) == len(full_validation)
    assert len(validation_values(resumed_validation)) == len(resumed_validation)
    assert validation_values(full_validation) == validation_values(resumed_validation)
