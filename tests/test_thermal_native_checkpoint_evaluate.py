from types import SimpleNamespace

import numpy as np
import pytest
import torch

import thermal_native_checkpoint_evaluate as command
from thermal_state_response_native_evaluate import near_object_errors


def test_near_object_errors_use_physical_support_and_finite_reference():
    prediction_values = torch.tensor([1., 3., 20., 50.])[None, :, None].repeat(1, 1, 5)
    target = torch.zeros_like(prediction_values)
    target[0, 0, 2] = float('nan')
    prediction = SimpleNamespace(
        native_main={'pred_field': prediction_values},
        native_prepared=SimpleNamespace(stencils={
            'fluid': SimpleNamespace(valid=torch.tensor([[True, True, True, False]]))}),
    )
    batch = SimpleNamespace(
        targets=SimpleNamespace(field_targets=target, point_weights=torch.ones((1, 4))),
        receivers=SimpleNamespace(fluid_xy=torch.tensor([[[1., 0.], [2., 0.], [3.1, 0.], [1.5, 0.]]])),
    )
    config = {'domain': {'lx': 10., 'ly': 10., 'nx': 10, 'ny': 10, 'module_radius': 1.},
              'layout': {'centers': [[0., 0.]]}}

    result = near_object_errors(prediction, batch, config)
    assert result['u']['valid_positions'] == 2
    assert result['u']['rmse'] == pytest.approx(np.sqrt(5.))
    assert result['u']['absolute_error_p95'] == pytest.approx(2.9)
    assert result['p']['valid_positions'] == 1
    assert result['p']['rmse'] == pytest.approx(3.)
    assert result['p']['unit'] == 'dataset pressure'


def test_explicit_population_runs_both_readers_and_applies_tile(monkeypatch, tmp_path):
    cases = tuple(f'{number:04d}' for number in range(100, 189))
    saved = {'epoch': 3000}
    model = SimpleNamespace(core=SimpleNamespace(query_tile_size=None))
    provider = SimpleNamespace(recipe={'evaluation': {}})
    monkeypatch.setattr(command, 'load_trusted_checkpoint', lambda *args, **kwargs: saved)
    monkeypatch.setattr(command, 'original_heldout_case_ids', lambda value: cases)
    monkeypatch.setattr(command, 'build_native_checkpoint_inference',
                        lambda value, validation_case_ids, device: (model, provider))
    calls = []

    def evaluate(model_value, provider_value, **kwargs):
        assert model_value is model and provider_value is provider
        assert provider.recipe['evaluation']['inference_query_tile_size'] == 512
        calls.append(kwargs)
        return {'case_ids': list(kwargs['case_ids'])}

    monkeypatch.setattr(command, 'evaluate_native_panel', evaluate)
    written = {}
    monkeypatch.setattr(command, '_atomic_json', lambda path, value: written.update(value))
    output = tmp_path / 'native.json'
    assert command.main(['--checkpoint', str(tmp_path / 'model.pt'), '--all-original-held-out',
                         '--read-mode', 'both', '--query-tile-size', '512',
                         '--export-cases', cases[0], '--output', str(output)]) == 0
    assert [item['mode'] for item in calls] == ['restricted', 'full']
    assert all(item['case_ids'] == cases for item in calls)
    assert calls[0]['output'] != calls[1]['output']
    assert written['case_count'] == 89 and set(written['views']) == {'restricted', 'full'}


def test_full_population_requires_explicit_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(command, 'load_trusted_checkpoint', lambda *args, **kwargs: {'epoch': 1})
    monkeypatch.setattr(command, 'build_native_checkpoint_inference',
                        lambda *args, **kwargs: pytest.fail('unexpected full-population build'))
    with pytest.raises(SystemExit):
        command.main(['--checkpoint', str(tmp_path / 'model.pt'), '--cases', ','.join(map(str, range(10))),
                      '--output', str(tmp_path / 'result.json')])
