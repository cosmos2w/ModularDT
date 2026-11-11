"""Sampled formal evaluator retains complete source-order family inventories."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from thermal_r16_10_evaluate import family_board_inputs, sampled_ids


def test_reserved_roles_select_input_sample_and_all_actions(monkeypatch):
    import channelthermal.training.thermal_state_response_task as old
    import channelthermal.training.r16_8_corpus as newer
    monkeypatch.setattr(old, 'ThermalResponseCorpus', lambda *args: object())

    class Corpus:
        def __init__(self, path):
            self.scope = Path(path).stem.removesuffix('_corpus')
            self.preview = False
            self.cases = {f'family{i:02}': {'source_config': {'layout': {'num_modules': 2}},
                                           'source_case_dir': f'/baseline/{i}'} for i in range(12)}
        def actions_for_family(self, family):
            return tuple(f'{family}_a{i}' for i in range(4))
        def action_entry(self, family, action, *, role):
            assert role == self.scope
            return {'centers': [[1, 2]], 'raw_case_dir': f'/edit/{action}'}
    monkeypatch.setattr(newer, 'ThermalR168Corpus', Corpus)
    recipe = {'r16_10_family': {'original_input_recipe': 'input', 'original_ledger': 'ledger'}}
    expected = sampled_ids([f'family{i:02}' for i in range(12)])
    for role in ('native_seen', 'fresh_layout'):
        rows, total = family_board_inputs(recipe, role, ())
        assert total == 12
        assert [row['family'] for row in rows] == list(expected)
        assert rows[0]['source'] == role and len(rows[0]['actions']) == 4
        assert [row['action'] for row in rows[0]['actions']] == [f'{expected[0]}_a{i}' for i in range(4)]
