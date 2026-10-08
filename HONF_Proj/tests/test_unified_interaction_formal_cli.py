"""Boundary checks for manual recipes, without a formal optimizer or dataset fit."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import unified_interaction_formal as cli
import unified_interaction_train as development_cli


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


@pytest.mark.parametrize("wind_recipe_id", [
    'wind_w1_component_q1024_v1',
    'wind_w3_component_q1024_h128_v1',
])
def test_manual_wind_pair_shares_one_prepared_recipe_and_case_epoch_binding(tmp_path, monkeypatch, wind_recipe_id):
    class Factory:
        def prepare_recipe(self, config, *, metadata_only):
            assert config['wind_recipe_id'] == wind_recipe_id
            assert metadata_only is True
            return {'recipe_id': f'prepared-{wind_recipe_id}', 'recipe_sha256': 'native-recipe-sha',
                    'ready_for_training': False,
                    'wind_training_recipe': {'recipe_id': wind_recipe_id}}

        def validate_recipe(self, native):
            return {'ready_for_training': native['ready_for_training']}

        def create_task(self, native):
            raise AssertionError('Manual recipe dry run constructed a model.')

    monkeypatch.setattr(cli, '_factory', lambda task: Factory())
    path = tmp_path / 'wind_pair.json'
    args = cli.build_parser().parse_args([
        'prepare', '--task', 'wind', '--arm', 'both', '--recipe', str(path),
        '--wind-recipe-id', wind_recipe_id, '--metadata-only',
    ])
    prepared = cli.prepare(args)
    assert prepared['optimizer_started'] is False
    assert len(prepared['matched_recipes']) == 2
    recipes = [json.loads(Path(item['recipe']).read_text()) for item in prepared['matched_recipes']]
    assert {item['execution_arm'] for item in recipes} == {'full_detail', 'adaptive_detail'}
    assert recipes[0]['dataset_recipe'] == recipes[1]['dataset_recipe']
    assert {item['horizon_epochs'] for item in recipes} == {5000}
    result = cli.dry_run(cli.build_parser().parse_args(['dry-run', '--recipe', prepared['matched_recipes'][0]['recipe']]))
    assert result['engine_config']['microbatch_cases'] == 24
    assert result['engine_config']['sampling_version'] == 'case_epoch_v1'
    assert result['optimizer_created'] is result['training_started'] is False


def test_versioned_development_profile_uses_measured_effective24_packing():
    legacy, _, _ = development_cli._profile('wind', 42)
    versioned, _, _ = development_cli._profile('wind', 42, versioned_wind=True)
    assert (legacy.microbatch_cases, legacy.effective_cases, legacy.sampling_version) == (
        4, 24, 'legacy_packed_v1')
    assert (versioned.microbatch_cases, versioned.effective_cases, versioned.sampling_version) == (
        24, 24, 'case_epoch_v1')


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


def test_sparse_monitoring_is_sealed_and_used_by_formal_dry_run(tmp_path, monkeypatch):
    recipe = _recipe()
    recipe['monitoring'] = {'validation_every': 500, 'validation_epochs': [100],
        'checkpoint_epochs': [100, 500, 1000, 2000, 2500, 5000],
        'latest_every': 100, 'curve_every': 100}
    monkeypatch.setattr(cli, '_factory', lambda task: type('Factory', (), {
        'validate_recipe': staticmethod(lambda native: {'ready_for_training': False})})())
    path = _save(tmp_path, recipe)
    result = cli.dry_run(cli.build_parser().parse_args(['dry-run', '--recipe', str(path)]))
    config = result['engine_config']
    assert config['monitor_every'] == 500
    assert config['monitor_epochs'] == (100,)
    assert config['checkpoint_epochs'] == (100, 500, 1000, 2000, 2500, 5000)
    assert config['latest_every'] == config['curve_every'] == 100


@pytest.mark.parametrize('overrides', [
    {'validation_every': 0}, {'checkpoint_epochs': [100, 500]},
    {'checkpoint_epochs': [5000, 100]}, {'latest_every': -1},
    {'validation_epochs': [5001]}, {'curve_every': 0},
])
def test_invalid_sparse_formal_schedule_is_rejected(overrides):
    monitoring = {'validation_every': 500, 'validation_epochs': [100],
        'checkpoint_epochs': [100, 500, 1000, 2000, 2500, 5000],
        'latest_every': 100, 'curve_every': 100}
    monitoring.update(overrides)
    with pytest.raises(ValueError):
        cli._profile('wind', monitoring=monitoring)


def test_invalid_monitoring_fails_before_full_data_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, '_factory', lambda task: pytest.fail('Invalid options reached full-data preparation.'))
    args = cli.build_parser().parse_args(['prepare', '--task', 'wind', '--arm', 'adaptive_detail',
        '--recipe', str(tmp_path / 'invalid.json'), '--curve-every', '0'])
    with pytest.raises(ValueError, match='interval'):
        cli.prepare(args)
    assert not list(tmp_path.iterdir())


def test_explicit_resume_microbatch_changes_only_accumulation_chunks():
    args = cli.build_parser().parse_args(['resume', '--recipe', 'recipe.json', '--run-dir', 'run',
        '--checkpoint', 'run/latest_model.pt', '--microbatch-cases', '8'])
    assert args.microbatch_cases == 8
    before, _ = cli._profile('wind')
    after, _ = cli._profile('wind', microbatch_cases=args.microbatch_cases)
    assert before.microbatch_cases == 4 and after.microbatch_cases == 8
    before_values, after_values = before.__dict__.copy(), after.__dict__.copy()
    before_values.pop('microbatch_cases'); after_values.pop('microbatch_cases')
    assert before_values == after_values
    with pytest.raises(ValueError, match='microbatch size'):
        cli._profile('wind', microbatch_cases=25)
