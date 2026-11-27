"""Fixed saved-edit selection, physical changes and inverse choice ordering."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal import saved_response_evaluation as old
from tools import thermal_saved_response_benchmark as benchmark


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def test_explicit_fit_panel_does_not_filter_by_current_native_membership(tmp_path, monkeypatch):
    from channelthermal.training import thermal_state_response_task, r16_8_corpus

    original_dirs = {}
    for family in ('0101', '0102'):
        folder = tmp_path / family
        write(folder / 'case_config.json', {'layout': {'centers': [[0, 0]]}})
        original_dirs[family] = str(folder)

    class Original:
        def __init__(self, *_):
            self.cases = {family: {'source_case_dir': folder, 'module_count': 1,
                                   'selection': {'actions': {'edit': [[1, 0]]}}}
                          for family, folder in original_dirs.items()}
        def family_ids(self, role):
            return ('0101', '0102') if role == 'fit' else ()
        def action_entry(self, family, action, *, role):
            return {'raw_case_dir': family + action}

    class New:
        def __init__(self, *_):
            self.cases = {family: {'source_config': {'layout': {'num_modules': 1}},
                                   'source_case_dir': family} for family in ('0201', '0202')}
        def family_ids(self, role):
            return ('0201', '0202')
        def actions_for_family(self, family):
            return (family + '_edit',)
        def action_entry(self, family, action, *, role):
            return {'centers': [[1, 0]], 'raw_case_dir': family + action}

    monkeypatch.setattr(thermal_state_response_task, 'ThermalResponseCorpus', Original)
    monkeypatch.setattr(thermal_state_response_task, 'ACTIONS', ('edit',))
    monkeypatch.setattr(r16_8_corpus, 'ThermalR168Corpus', New)
    recipe = {'r16_10_family': {'original_input_recipe': 'unused', 'original_ledger': 'unused',
                                'new_fit_manifest': 'unused'}}
    selected, _ = old.family_board_inputs(recipe, 'fit', ('0101', '0201'),
                                          selected_families=('0102', '0202'))
    assert [(item['family'], item['source']) for item in selected] == [('0102', 'original'), ('0202', 'new')]
    default, population = old.family_board_inputs(recipe, 'fit', ('0101', '0201'))
    assert population == 2 and {item['family'] for item in default} <= {'0101', '0201'}
    with pytest.raises(ValueError, match='unique'):
        old.family_board_inputs(recipe, 'fit', ('0101',), selected_families=('0102', '0102'))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_physical_change_reports_zero_reference_direction_and_new_fluid(dtype):
    baseline = (np.array([2., 8., 3.], dtype=dtype), np.array([1., 7., 5.], dtype=dtype), np.array([1, 1, 0]))
    edited = (np.array([5., 9., 10.], dtype=dtype), np.array([3., 8., 8.], dtype=dtype), np.array([1, 1, 1]))
    measured = old.physical_response_error(baseline, edited, detailed=True, newly_valid_state=True)
    assert measured['rmse'] == pytest.approx(np.sqrt(.5))
    assert measured['zero_change_reference_rmse'] == pytest.approx(np.sqrt(2.5))
    assert measured['relative_to_zero_change_rmse'] == pytest.approx(np.sqrt(.2))
    assert measured['magnitude_ratio'] == pytest.approx(np.sqrt(10 / 5))
    assert measured['direction_cosine'] == pytest.approx(7 / np.sqrt(50))
    assert measured['newly_valid_edited_state'] == {'count': 1, 'rmse': 2.}
    assert type(measured['reference_resolved']) is bool
    assert type(measured['prediction_resolved']) is bool
    assert json.loads(json.dumps(measured, allow_nan=False)) == measured
    zero = old.physical_response_error(baseline, baseline, detailed=True)
    assert zero['relative_to_zero_change_rmse'] is None and zero['direction_cosine'] is None
    assert old.physical_response_error(baseline, edited) == {'rmse': pytest.approx(np.sqrt(.5)), 'count': 2}


def make_fixed_reference(tmp_path, monkeypatch):
    role_families = {'fit': [f'fit_{i}' for i in range(14)], 'original_withheld': ['old'],
                     'historical_new_withheld': ['new1', 'new2'], 'native_seen': ['native'],
                     'fresh_layout': ['fresh']}
    actions = {family: [f'{family}_action_{index}' for index in range(6 if role == 'historical_new_withheld' else 4)]
               for role, families in role_families.items() for family in families}
    for role, families in role_families.items():
        rows = [{'family': family, 'action': action} for family in families for action in actions[family]]
        write(tmp_path / 'boards' / f'{role}.json', {'role': role, 'family_ids': families,
            'population_count': len(families), 'views': {'restricted': {'actions': rows}, 'full': {'actions': rows}}})
    request = {'request_id': 'fixed', 'case_id': '0010', 'objective': 'J', 'constraint': 'P',
               'constraint_max_delta': 0., 'candidate_ids': ['base'] + [f'edit_{i}' for i in range(6)],
               'baseline_action_id': 'base'}
    write(tmp_path / 'boards/strict.json', {'request': request})
    recipe = write(tmp_path / 'recipe.json', {'r16_10_family': {}})
    inputs = write(tmp_path / 'inputs.json', {})
    scales = write(tmp_path / 'scales.json', {'provider_identity': {'response_scales': {'J': {'scale': 2., 'floor': .1}}}})
    manifest = write(tmp_path / 'strict_actions.json', {'cases': [{'case_id': '0010',
        'source_config': {'layout': {'centers': [[0, 0]], 'heat_powers': [1]}},
        'baseline_centers': [[0, 0]], 'baseline_heat_powers': [1], 'member_ids': [0]}],
        'actions': [{'case_id': '0010', 'action': f'edit_{i}', 'centers': [[i + 1, 0]],
                     'physical_delta': {'J': -float(i), 'P': .1 if i == 5 else 0.}} for i in range(6)]})
    protocol = write(tmp_path / 'protocol.json', {'requests': [request], 'action_manifest': str(manifest), 'tau': .4})
    sources = {role: write(tmp_path / f'{role}.json', {})
               for role in ('historical_new_withheld', 'native_seen', 'fresh_layout')}
    def cases(_recipe, role, native_ids, *, selected_families, source_paths=None):
        assert native_ids == ()
        return ([{'family': family, 'source': role, 'baseline_path': family + '_baseline',
                  'actions': [{'action': action, 'raw_path': action + '_raw'} for action in actions[family]]}
                 for family in selected_families], len(role_families[role]))
    monkeypatch.setattr(benchmark, 'family_board_inputs', cases)
    panel = benchmark.prepare_panel(tmp_path / 'boards', recipe, inputs, protocol, scales, source_paths=sources)
    return panel, role_families, actions, manifest


def test_panel_keeps_historical_actions_and_reports_native_exposure(tmp_path, monkeypatch):
    panel, roles, actions, _ = make_fixed_reference(tmp_path, monkeypatch)
    assert (panel['family_count'], panel['edit_count']) == (19, 80)
    assert panel['roles']['fit']['families'][0]['actions'] == actions[roles['fit'][0]]
    exposure = benchmark.native_exposure(panel, (roles['fit'][0], 'native'))['families']
    assert exposure[0]['baseline_in_current_native_training'] is True
    assert next(row for row in exposure if row['family'] == 'fresh')['baseline_in_current_native_training'] is None
    board_path = tmp_path / 'boards/fit.json'
    board = json.loads(board_path.read_text())
    board['views']['full']['actions'][0]['action'] = 'changed'
    write(board_path, board)
    with pytest.raises(ValueError, match='reader panels differ'):
        benchmark.prepare_panel(tmp_path / 'boards', panel['historical_recipe'], panel['training_inputs'],
                                panel['strict_protocol'], panel['strict_scales_identity'], source_paths=panel['source_paths'])


def test_fixed_actions_reject_drift_before_prediction_files(tmp_path, monkeypatch):
    monkeypatch.setattr(old, 'family_board_inputs', lambda *args, **kwargs:
                        ([{'family': 'case', 'actions': [{'action': 'other'}]}], 1))
    with pytest.raises(ValueError, match='family/action'):
        old.family_board(None, SimpleNamespace(training_case_ids=[]), {}, 'fit', None, tmp_path / 'result.json',
                         selected_families=('case',), expected_actions={'case': ['expected']})
    assert not list(tmp_path.iterdir())


def test_strict_choice_is_persisted_before_physical_join(tmp_path, monkeypatch):
    from channelthermal import native_design
    panel, _, _, manifest = make_fixed_reference(tmp_path, monkeypatch)
    output = tmp_path / 'result' / 'strict.json'
    original_read = benchmark._read
    reads = []
    def read(path):
        if Path(path) == manifest:
            reads.append(1)
            if len(reads) == 2:
                assert output.with_name('strict_predictions.json').is_file()
        return original_read(path)
    monkeypatch.setattr(benchmark, '_read', read)
    monkeypatch.setattr(native_design, 'build_native_design', lambda config, *args, **kwargs: config)
    monkeypatch.setattr(native_design, 'score_native_design', lambda model, design, **kwargs: {
        'functionals': {'J': -float(design['layout']['centers'][0][0]), 'P': 0.}, 'work': {}})
    class Model:
        core = SimpleNamespace(query_tile_size=0)
        def eval(self):
            return self
    result = benchmark.strict_request(Model(), 'checkpoint.pt', 500, panel, output, device='cpu')
    assert len(reads) == 2
    assert result['views']['full']['selected_action'] == 'edit_5'
    assert result['views']['full']['raw_constraint_violation']
    assert result['views']['full']['accepted_action'] == 'base'
    assert result['views']['full']['raw_numerically_resolved_improvement']
    assert result['views']['full']['offline_truth_based_fallback_used']
    assert result['views']['full']['raw_feasible_regret'] is None
    assert result['views']['full']['accepted_normalized_regret'] == pytest.approx(2.)


def test_evaluation_dispatch_reuses_exact_panel_for_both_cohorts(tmp_path, monkeypatch):
    panel, roles, actions, _ = make_fixed_reference(tmp_path, monkeypatch)
    panel_path = write(tmp_path / 'panel.json', panel)
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.touch()
    monkeypatch.setattr(benchmark, 'load_trusted_checkpoint', lambda *a, **k: {'epoch': 500})
    monkeypatch.setattr(benchmark, 'original_heldout_case_ids', lambda saved: ('heldout',))
    calls = []
    monkeypatch.setattr(benchmark, 'family_board', lambda *args, **kwargs: (
        calls.append((args[3], kwargs['selected_families'], kwargs['expected_actions'])) or {}))
    monkeypatch.setattr(benchmark, 'strict_request', lambda *args, **kwargs: {})
    for cohort, train_ids in (('small', ('fit_1',)), ('full', ('fit_2', 'fit_3'))):
        monkeypatch.setattr(benchmark, 'build_native_checkpoint_inference', lambda *a, ids=train_ids, **k:
                            (None, SimpleNamespace(training_case_ids=ids)))
        benchmark.evaluate(panel_path, checkpoint, tmp_path / cohort, device='cpu')
    assert len(calls) == 10
    assert calls[:5] == calls[5:]
    assert calls[0] == ('fit', tuple(roles['fit']), {family: actions[family] for family in roles['fit']})
    small = json.loads((tmp_path / 'small/native_exposure.json').read_text())['families']
    full = json.loads((tmp_path / 'full/native_exposure.json').read_text())['families']
    assert [row['baseline_in_current_native_training'] for row in small] != [row['baseline_in_current_native_training'] for row in full]
    benchmark.evaluate(panel_path, checkpoint, tmp_path / 'withheld', device='cpu',
                       roles=('original_withheld', 'historical_new_withheld'), include_strict=False)
    partial = json.loads((tmp_path / 'withheld/evaluation_summary.json').read_text())
    assert partial['family_count'] == 3 and partial['edit_count'] == 16 and partial['strict_request'] is None
