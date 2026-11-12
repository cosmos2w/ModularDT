"""Native evaluation preserves the complete panel and nine field summaries."""
from contextlib import nullcontext
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('native_response_eval', ROOT / 'tools/thermal_state_response_native_evaluate.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_complete_native_panel_reuses_nine_observable_reduction(tmp_path):
    class Provider:
        training_cases = validation_cases = ()
        def native_evaluation_batches(self, *, split, case_keys, microbatch_size):
            assert split == 'validation' and microbatch_size == 1
            for case in case_keys:
                yield SimpleNamespace(targets=SimpleNamespace(case_ids=(case,)), scene_inputs=None, receivers=None)
        def predict_native(self, *args, **kwargs): return None, None
        def validation_metrics(self, prediction, targets, auxiliary):
            value = {'a': 1., 'b': 3.}[targets.case_ids[0]]
            roles = {f'temperature_{r}': {'rmse': value} for r in ('fluid', 'surface', 'material')}
            roles['temperature_material_endpoint_maxima'] = {'per_module': [{'error': value}]}
            return {'case_rows': [{'case_id': targets.case_ids[0], 'module_count': 1,
                      'flow_rmse': dict.fromkeys(('u', 'v', 'p', 'omega'), value),
                      'roles': roles, 'q_proxy_rmse': value}]}
        def work_counts(self, *args): return {'native_fluid_queries': 8192}
        def reduce_native_evaluation_metrics(self, records, *, split, case_keys):
            rows = [x['case_rows'][0] for x in records]
            assert tuple(row['case_id'] for row in rows) == case_keys
            return {'per_case': rows}
    model = torch.nn.Linear(1, 1)
    model.core = SimpleNamespace(read_execution=lambda mode: nullcontext())
    model.architecture = 'fixture'
    result = entry.evaluate_native_panel(model, Provider(), epoch=1000, split='validation',
        case_ids=('a', 'b'), mode='restricted', export_cases=(), output=tmp_path / 'result.json')
    assert len(result['nine_observables']) == 9
    assert result['nine_observables']['u']['mean'] == 2.
    assert result['nine_observables']['u']['p90'] == pytest.approx(2.8)
    assert result['nine_observables']['u']['worst_case_id'] == 'b'
    assert result['work_counts']['native_fluid_queries'] == 16384


@pytest.mark.parametrize('profile,split,all_cases,available_count,expected_count', [
    (None, 'train', False, 100, 10),
    (None, 'validation', False, 100, 10),
    ('state_response', 'validation', False, 9, 9),
    (None, 'validation', True, 100, 100),
])
def test_native_cli_selects_sample_or_preserved_validation_panel(monkeypatch, tmp_path,
        profile, split, all_cases, available_count, expected_count):
    recipe = {} if profile is None else {'training_profile': profile}
    recipe_path = tmp_path/'recipe.json'
    recipe_path.write_text(json.dumps(recipe))
    available = tuple(str(i) for i in range(available_count))
    provider = SimpleNamespace(training_case_ids=available, validation_case_ids=available)
    model = SimpleNamespace(core=SimpleNamespace(reader_capability='anchored'))
    monkeypatch.setattr(entry, 'restore_model', lambda *args: (model, provider, {'epoch': 1}, tmp_path/'saved.pt'))
    seen = []
    def evaluate(model_arg, provider_arg, **kwargs):
        seen.extend(kwargs['case_ids'])
        return {}
    monkeypatch.setattr(entry, 'evaluate_native_panel', evaluate)
    args = ['--recipe', str(recipe_path), '--run', 'run', '--output', str(tmp_path/'out.json'), '--split', split]
    if all_cases:
        args.append('--all-cases')
    entry.main(args)
    assert tuple(seen) == available[:expected_count]
