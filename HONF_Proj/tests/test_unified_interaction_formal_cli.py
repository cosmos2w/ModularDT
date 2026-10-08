"""Boundary checks for manual recipes, without a formal optimizer or dataset fit."""
from __future__ import annotations

import copy
import json

import pytest
import unified_interaction_formal as cli


def _recipe():
    return {'workflow': cli.FORMAL_WORKFLOW, 'task': 'wind',
            'execution_arm': 'adaptive_detail',
            'dataset_recipe': {'ready_for_training': False},
            'ready_for_training': False, 'horizon_epochs': 5000,
            'fine_hold_through_epoch': 2000, 'development_weights_loaded': False,
            'optimizer_started': False, 'new_solver_attempts': 0}


def _save(tmp_path, value):
    value = copy.deepcopy(value)
    value['recipe_sha256'] = cli._canonical_sha(value)
    path = tmp_path / 'manual_recipe.json'
    path.write_text(json.dumps(value))
    return path


def test_metadata_dry_run_never_constructs_task(tmp_path, monkeypatch):
    class Factory:
        def validate_recipe(self, native):
            return {'ready_for_training': native['ready_for_training']}

        def create_task(self, native):
            raise AssertionError('Metadata dry run constructed a formal model.')

    monkeypatch.setattr(cli, '_factory', lambda task: Factory())
    path = _save(tmp_path, _recipe())
    args = cli.build_parser().parse_args(['dry-run', '--recipe', str(path)])
    result = args.handler(args)
    assert result['optimizer_created'] is False
    assert result['training_started'] is False
    assert result['ready_for_training'] is False
    assert result['engine_config']['total_epochs'] == 5000


@pytest.mark.parametrize('key,value', [
    ('ready_for_training', True), ('horizon_epochs', 2500),
    ('development_weights_loaded', True), ('new_solver_attempts', 1),
])
def test_resealed_crossed_contract_rejected(tmp_path, monkeypatch, key, value):
    recipe = _recipe()
    recipe[key] = value
    path = _save(tmp_path, recipe)
    monkeypatch.setattr(cli, '_factory', lambda task: pytest.fail('Crossed recipe reached its provider.'))
    with pytest.raises(ValueError, match='contract'):
        cli._load_recipe(path)


def test_recipe_edit_without_reseal_is_rejected(tmp_path):
    path = _save(tmp_path, _recipe())
    value = json.loads(path.read_text())
    value['task'] = 'thermal'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='hash'):
        cli._load_recipe(path)


def test_manual_lifecycle_parser_requires_resume_checkpoint():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['resume', '--recipe', 'recipe.json', '--run-dir', 'future_run'])
    args = parser.parse_args(['start', '--recipe', 'recipe.json', '--run-dir', 'startup',
                              '--startup-benchmark', '--stop-after', '3'])
    assert args.startup_benchmark and args.stop_after == 3
