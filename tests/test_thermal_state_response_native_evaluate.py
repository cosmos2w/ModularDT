"""Native evaluation preserves the complete panel and nine field summaries."""
from contextlib import nullcontext
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


def test_train_evaluation_requires_explicit_membership_panel():
    with pytest.raises(SystemExit):
        entry.main(['--recipe','recipe.json','--run','run','--output','out.json','--split','train'])
