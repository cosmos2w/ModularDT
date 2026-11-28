"""Recovery preparation uses committed artifacts without another optimizer update."""
import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from honf_runtime import unified_training as runtime
from honf_runtime.run_layout import RunLayout


spec = importlib.util.spec_from_file_location(
    'coupled_recovery', Path(__file__).resolve().parents[1] / 'tools/verify_coupled_native_recovery.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_epoch_one_curve_reads_canonical_saved_definitions(tmp_path, monkeypatch):
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'metrics').mkdir()
    (tmp_path / 'configs/metric_definitions.json').write_text(json.dumps({'terms': {'native': {'unit': '1'}}}))
    (tmp_path / 'metrics/history.csv').write_text('epoch\n1\n')
    history = [{'epoch': 1}]
    monkeypatch.setattr(runtime, 'read_training_history', lambda path: history)
    monkeypatch.setattr(runtime, 'capture_rng_state', lambda: 'saved-rng')
    restored, rendered = [], []
    monkeypatch.setattr(runtime, 'restore_rng_state', restored.append)
    monkeypatch.setattr(runtime, '_render_loss_curves', lambda *args, **kwargs: rendered.append((args, kwargs)))
    payload = {'experiment_identity': {'selection_policy': {'field_metric': 'field_score'},
                                      'provider_identity': {'dataset': 'fixture'}}}
    recovery._render_epoch_one(RunLayout(tmp_path), payload)
    args, kwargs = rendered[0]
    assert args == (history, tmp_path / 'plots/training', 'field_score')
    assert kwargs['loss_metadata'] == {'native': {'unit': '1'}}
    assert kwargs['metadata_output'] == tmp_path / 'metrics'
    assert restored == ['saved-rng']


def test_prepare_committed_state_preserves_checkpoint_and_saved_config(tmp_path, monkeypatch):
    run = tmp_path / 'Interrupted_Sampled_Reader_Recovery'
    (run / 'configs').mkdir(parents=True)
    (run / 'checkpoints').mkdir()
    saved = {'first': {'native': 1, 'response': 2}, 'second': 3}
    (run / 'configs/resolved_config.json').write_text(json.dumps(saved))
    checkpoint = run / 'checkpoints/latest_model.pt'
    checkpoint.write_bytes(b'committed checkpoint')
    payload = {'epoch': 1, 'sampler_state': {'next_epoch': 2},
               'optimizer_state_by_name': {'state_by_name': {'weight': {}}},
               'experiment_identity': {'recipe': saved}}
    monkeypatch.setattr(recovery, 'load_trusted_checkpoint', lambda *args, **kwargs: payload)
    plotted = []
    monkeypatch.setattr(recovery, '_render_epoch_one', lambda layout, actual: plotted.append((layout.run_dir, actual)))
    recovery._prepare_interrupted(tmp_path)
    preserved = tmp_path / 'Interrupted_Epoch_One_Model.pt'
    assert preserved.samefile(checkpoint)
    reordered = json.loads((tmp_path / 'reordered_saved_recipe.json').read_text())
    assert reordered == saved and list(reordered) == ['second', 'first']
    assert list(reordered['first']) == ['response', 'native']
    assert plotted == [(run, payload)]


def _finite_run(run, *, elapsed='1', query_count='12'):
    for directory in ('configs', 'metrics', 'plots/training'):
        (run / directory).mkdir(parents=True, exist_ok=True)
    objective = {'variant': 'contrasts', 'probe_seed': 130013, 'frame': {'lower': [0., 0.]}}
    calibration = {'coefficients': {'anchor': .1, 'contrast': .05}, 'calibration_case_ids': ['one', 'two']}
    source = {'schema': 'finite_observation_source_definitions_v1', 'scenes': [
        {'scene_id': 'scene', 'families': [{'row_ids': ['row-1'], 'extraction': {'coefficients': [1.]}}]}]}
    (run / 'configs/source.json').write_text(json.dumps(source))
    pointer = {'schema': 'finite_observation_run_definition_v1',
               'source_definitions': 'source.json', 'objective': objective,
               'calibration': calibration, 'metadata_bytes': 100}
    (run / 'configs/finite_observation_objective.json').write_text(json.dumps(pointer))
    columns = ['epoch', 'train_seconds', 'work_counts.finite_observations_planning_seconds',
               'work_counts.finite_observations_prediction_seconds',
               'work_counts.finite_observations_neural_queries', 'train_losses.native']
    definitions = {'training_columns': columns, 'training_loss_order': ['native', 'anchor'],
                   'training_total_order': ['anchor', 'native']}
    (run / 'configs/metric_definitions.json').write_text(json.dumps(definitions))
    with (run / 'metrics/history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for epoch in (1, 2):
            writer.writerow(dict(zip(columns, (str(epoch), elapsed, elapsed, elapsed, query_count, '0.25'))))
    (run / 'metrics/validation.csv').write_text('epoch,metric,value\n2,native,0.5\n')
    for name in ('loss_curves_initial.png', 'loss_curves_initial.pdf', 'loss_curves.png', 'loss_curves.pdf'):
        (run / 'plots/training' / name).touch()
    checkpoint_objective = copy.deepcopy(objective)
    checkpoint_objective['frame']['lower'] = (0., 0.)
    checkpoint_calibration = copy.deepcopy(calibration)
    checkpoint_calibration['calibration_case_ids'] = ('one', 'two')
    return {'epoch': 2, 'model_state_dict': {'weight': 1},
        'optimizer_state_by_name': {'state_by_name': {'weight': {'step': 2}}},
        'rng_state': {'seed': 1}, 'sampler_state': {'next_epoch': 3},
        'provider_training_state': {'finite_observations': {
            'schema': 'finite_observation_training_state_v1', 'calibration': checkpoint_calibration, 'last_reader': 'full'}},
        'best_field_score': .5, 'best_response_guarded_score': .6, 'optimizer_group_names': ['joint'],
        'optimizer_schedule_contract': {'horizon': 1000},
        'experiment_identity': {'recipe': {'finite_observations': {'variant': 'contrasts'}},
                                'provider_identity': {'finite_observations': checkpoint_objective}}}


def _comparison(tmp_path, monkeypatch):
    whole, resumed = tmp_path / 'whole', tmp_path / 'resumed'
    left = _finite_run(whole, elapsed='1')
    right = _finite_run(resumed, elapsed='9')
    interrupted = {'epoch': 1, 'sampler_state': {'next_epoch': 2},
                   'optimizer_state_by_name': {'state_by_name': {'weight': {'step': 1}}}}
    def load(path, **_):
        if Path(path).name == 'interrupted.pt':
            return interrupted
        return left if Path(path).parents[1] == whole else right
    monkeypatch.setattr(recovery, 'load_trusted_checkpoint', load)
    return whole, resumed, left, right


def test_finite_recovery_compares_state_calibration_and_deterministic_work(tmp_path, monkeypatch):
    whole, resumed, _, right = _comparison(tmp_path, monkeypatch)
    receipt = recovery._check_runs(whole, resumed, tmp_path / 'interrupted.pt')
    assert receipt['finite_source_definitions_equal'] and receipt['finite_objective_and_calibration_equal']
    assert 'work_counts.finite_observations_prediction_seconds' in receipt['elapsed_and_cache_work_excluded']
    assert 'work_counts.finite_observations_neural_queries' not in receipt['elapsed_and_cache_work_excluded']

    history = resumed / 'metrics/history.csv'
    history.write_text(history.read_text().replace(',12,', ',13,'))
    with pytest.raises(AssertionError, match='neural_queries'):
        recovery._check_runs(whole, resumed, tmp_path / 'interrupted.pt')
    history.write_text(history.read_text().replace(',13,', ',12,'))
    right['provider_training_state']['finite_observations']['last_reader'] = 'restricted'
    with pytest.raises(AssertionError, match='provider_training_state'):
        recovery._check_runs(whole, resumed, tmp_path / 'interrupted.pt')


@pytest.mark.parametrize('changed', ['probe_rows', 'extraction', 'coefficient', 'loss_order'])
def test_finite_recovery_rejects_changed_saved_semantics(tmp_path, monkeypatch, changed):
    whole, resumed, _, _ = _comparison(tmp_path, monkeypatch)
    path = resumed / ('configs/metric_definitions.json' if changed == 'loss_order' else
                      'configs/finite_observation_objective.json' if changed == 'coefficient' else
                      'configs/source.json')
    value = json.loads(path.read_text())
    if changed == 'probe_rows':
        value['scenes'][0]['families'][0]['row_ids'] = ['different']
    elif changed == 'extraction':
        value['scenes'][0]['families'][0]['extraction']['coefficients'] = [-1.]
    elif changed == 'coefficient':
        value['calibration']['coefficients']['contrast'] = .07
    else:
        value['training_loss_order'].reverse()
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        recovery._check_runs(whole, resumed, tmp_path / 'interrupted.pt')


def test_generic_finite_only_next_draw_uses_native_sampler_and_rechecks_saved_rows(tmp_path, monkeypatch):
    from honf_runtime.finite_observation_task import attach_finite_observation_task
    from honf_runtime.observation_response_fixture import build_observation_capability

    recipe = {'dataset': 'analytic_fixture', 'seed': 7, 'architecture': 'relation_anchored',
              'total_epochs': 2, 'microbatch_size': 2, 'effective_batch_size': 2,
              'fixture': {'output_dim': 7, 'finite_only_binding': True,
                          'native_sampling_law': 'weighted_with_replacement', 'native_draw_count': 8},
              'finite_observations': {'variant': 'contrasts', 'native_sampling': 'native',
                                      'calibration_case_ids': ['fixture_0', 'fixture_1']}}
    model, provider, binding = build_observation_capability(recipe, device='cpu')
    model, provider = attach_finite_observation_task(recipe, model, provider, binding)
    calibration = {'schema': 'finite_observation_calibration_v1', 'epoch': 0,
                   'calibration_case_ids': ['fixture_0', 'fixture_1'],
                   'coefficients': {'anchor': .1, 'contrast': .05},
                   'gradient_diagnostics': {'groups': {'global': {'norms': {'native': 1., 'anchor': 1., 'contrast': 1.}}}}}
    provider.set_finite_observation_calibration(calibration)
    run = tmp_path / 'run'
    provider.write_finite_observation_definitions(run)
    saved_state = copy.deepcopy(provider.training_state_dict())
    checkpoint = {'experiment_identity': {'recipe': recipe}, 'sampler_state': {'next_epoch': 2},
                  'provider_training_state': saved_state}
    monkeypatch.setitem(sys.modules, 'relation_anchored_train',
                        SimpleNamespace(build=lambda *args, **kwargs: (model, provider)))
    monkeypatch.setattr(recovery, 'load_trusted_checkpoint', lambda *args, **kwargs: checkpoint)
    recovery._sample_worker(run, tmp_path / 'draws.json')
    draws = json.loads((tmp_path / 'draws.json').read_text())
    assert draws['native_sampling'] == 'native_make_batch_receivers'
    assert len(draws['sample']) == 2 and all(len(rows) == 8 for rows in draws['sample'])
    assert draws['finite_calibration_coefficients'] == {'anchor': .1, 'contrast': .05}
    assert draws['finite_definitions']['saved_and_rebuilt_definitions_equal']
    assert draws['finite_reader_draws'] == draws['reader_draws'] and len(draws['reader_draws']) == 4
    source = run / 'configs/finite_observations.json'
    original = json.loads(source.read_text())
    for field, bad in (('row_ids', ['wrong']), ('actual_basis', 'constant'),
                       ('frame_definition', None), ('out_of_frame_count', 999)):
        changed = copy.deepcopy(original)
        changed['scenes'][0]['families'][0][field] = bad
        source.write_text(json.dumps(changed))
        with pytest.raises(AssertionError, match='Rebuilt finite probe IDs'):
            recovery._sample_worker(run, tmp_path / 'different.json')


def test_recovery_recipe_supports_legacy_and_finite_only_horizons():
    assert recovery._supported_recipe({'observation_response': {
        'mode': 'coupled_native', 'variant': 'sampled_reader_response'}})
    assert recovery._supported_recipe({'finite_observations': {'native_sampling': 'native'}})
    assert not recovery._supported_recipe({'observation_response': {'variant': 'anchors'}})
    assert recovery._recipe_horizon({'training': {'local_horizon': 1000}}) == 1000
    assert recovery._recipe_horizon({'total_epochs': 2}) == 2
