"""R16 matched recipes reach the paired-response entry point."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def recipes():
    return {arm: json.loads((ROOT / f'configs/research/thermal_r16_1_{arm}_train150.json').read_text())
            for arm in ('h0', 'q0', 'h1')}


def test_matched_r16_configs_differ_only_by_name_and_reader():
    configs = recipes()
    assert [configs[a]['member_routing']['model']['reader_capability'] for a in configs] == [
        'anchored', 'all_native', 'evidence_strength']
    for config in configs.values():
        config.pop('name')
        config['member_routing']['model'].pop('reader_capability')
    assert configs['h0'] == configs['q0'] == configs['h1']


@pytest.mark.parametrize('arm', ['h0', 'q0', 'h1'])
def test_r16_recipe_dispatches_to_response_provider(monkeypatch, arm):
    spec = importlib.util.spec_from_file_location('r16_entry', ROOT / 'tools/relation_anchored_train.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    calls = []
    module = types.ModuleType('channelthermal.training.thermal_state_response_task')
    module.build_thermal_state_response_task = lambda recipe, *, arm, device: calls.append(
        (recipe, arm, device)) or ('model', 'provider')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    recipe = recipes()[arm]
    assert entry.build(recipe, device='cpu') == ('model', 'provider')
    assert calls == [(recipe, recipe['member_routing']['arm'], 'cpu')]
    engine = entry.engine_for(recipe, device='cpu')
    assert engine.config.total_epochs == 1000
    assert tuple(engine.config.checkpoint_epochs) == tuple(range(100, 1001, 100))
