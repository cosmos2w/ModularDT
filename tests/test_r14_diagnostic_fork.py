"""Explicit R14 subset continuation preserves the source and saved identity."""
import json
from pathlib import Path

import pytest
import torch

from honf_runtime import contextual_frontier_training as training
from honf_runtime.run_layout import RunLayout


@pytest.mark.parametrize('compact_history', [False, True])
def test_historical_diagnostic_fork_is_explicit_and_preserves_state(tmp_path, monkeypatch, compact_history):
    source = tmp_path / 'source'
    layout = RunLayout(source)
    layout.ensure()
    config = {'member_routing': {'arm': training.R14_ARM},
              'training': {'membership': 'fixed25_train150_dev22', 'local_horizon': 1000,
                           'checkpoint_epochs': list(range(100, 1001, 100)),
                           'wall_clock_budget_seconds': 10000, 'clean_save_reserve_seconds': 600},
              'execution': {}, 'storage': {}}
    manifest = {'schema': 'honf_r14_comparison_run_v1', 'config': config}
    layout.write_path('run_manifest.json').write_text(json.dumps(manifest))
    identity = {'schema': 'honf_r14_comparison_training_v1', 'workflow': training.R14_WORKFLOW,
                'provider_identity': {'sentinel': 1}, 'engine_config': {},
                'selection_policy': {}, 'optimizer_schedule_contract': [],
                'sampling_dataset_id': 'same', 'recipe': {'unchanged': True}}
    state = {'epoch': 1000, 'arm': training.R14_ARM, 'experiment_identity': identity,
             'model_state_dict': {'weight': torch.tensor([1.])},
             'optimizer_state_by_name': {'state_by_name': {}}, 'history': list(range(1000)),
             'provider_training_state': {'training_elapsed_seconds': 1200.0}}
    source_latest = layout.write_path('latest_model.pt')
    torch.save(state, source_latest)
    before = source_latest.read_bytes()
    history = [{'history_metadata_version': 1}] if compact_history else [{'epoch': 1}]
    history_path = layout.write_path('history.json')
    history_path.write_text(json.dumps(history))
    for name in ('loss_curves.pdf', 'loss_curves.png'):
        layout.write_path('plots/training/' + name).write_bytes(b'inherited curve')
    monkeypatch.setattr(training, '_validate_loaded_contextual_config', lambda config: None)
    monkeypatch.setattr(training.research, 'verify_storage', lambda *a, **k: {})
    monkeypatch.setattr(training.research, '_permitted_order', lambda *a: ['cpu'])
    monkeypatch.setattr(training.research, 'resolve_device', lambda *a, **k: (torch.device('cpu'), {}))
    monkeypatch.setattr(training, '_build_task', lambda *a, **k: (object(), object()))
    monkeypatch.setattr(training, '_storage_estimate', lambda *a, **k: (0, 0))
    monkeypatch.setattr(training, '_prepare_storage', lambda *a, **k: {})
    rendered = []
    monkeypatch.setattr(training, '_refresh_diagnostic_fork_curves',
                        lambda history, output, provider: rendered.append((len(history), output)))
    observed = {}
    def fake_fit(model, provider, config, output, **kwargs):
        observed.update(output=output, **kwargs)
        return {'status': 'prepared'}
    monkeypatch.setattr(training, '_fit', fake_fit)
    target = tmp_path / 'fork'
    with pytest.raises(ValueError):
        training.resume_contextual_frontier(source, device='cpu', stop_after_epoch=2000,
                                            diagnostic_fork_output=target, verify_bindings=False)
    assert not target.exists()
    result = training.resume_contextual_frontier(source, device='cpu', stop_after_epoch=2000,
        extend_horizon=True, diagnostic_fork_output=target, verify_bindings=False,
        wall_clock_budget_seconds=5000)
    assert result['status'] == 'prepared'
    assert source_latest.read_bytes() == before
    assert (target / 'checkpoints/latest_model.pt').read_bytes() == before
    fork = json.loads((target / 'run_manifest.json').read_text())['diagnostic_fork']
    assert fork['source_epoch'] == 1000
    assert fork['source_elapsed_seconds'] == 1200
    assert fork['provider_budget_seconds'] == 5000
    assert (target / 'plots/training/loss_curves.pdf').read_bytes() == b'inherited curve'
    assert (target / 'plots/training/loss_curves.png').read_bytes() == b'inherited curve'
    assert rendered == [(1000, target)]
    assert observed['wall_clock_budget_seconds'] == 5000
    assert observed['compact_history'] is compact_history
    assert json.loads(history_path.read_text()) == history
    assert json.loads((target / 'metrics/history.json').read_text()) == history
    assert observed['identity_override']['recipe'] == {'unchanged': True}
    assert 'provider_identity' not in observed['identity_override']
    assert observed['allow_horizon_extension'] is True
    assert observed['engine_checkpoint_epochs'] == tuple(range(100, 2001, 100))
    with pytest.raises(ValueError):
        training.resume_contextual_frontier(source, device='cpu', stop_after_epoch=2001,
            extend_horizon=True, diagnostic_fork_output=tmp_path / 'too_far', verify_bindings=False)

    training.resume_contextual_frontier(target, device='cpu', stop_after_epoch=2000,
        extend_horizon=True, verify_bindings=False, wall_clock_budget_seconds=6000)
    amended_manifest = json.loads((target / 'run_manifest.json').read_text())
    assert amended_manifest['diagnostic_fork'] == fork
    assert amended_manifest['execution_budget'] == {
        'provider_budget_seconds': 6000, 'previous_provider_budget_seconds': 5000}
    assert observed['wall_clock_budget_seconds'] == 6000
    training.resume_contextual_frontier(target, device='cpu', stop_after_epoch=2000,
        extend_horizon=True, verify_bindings=False)
    assert observed['wall_clock_budget_seconds'] == 6000
    assert json.loads((target / 'run_manifest.json').read_text())['execution_budget'] == amended_manifest['execution_budget']
    with pytest.raises(ValueError):
        training._resume_execution_budget(config, amended_manifest, source_latest, state, 1200 + 24 * 3600 + 1, diagnostic_fork=fork)
    with pytest.raises(ValueError):
        training._resume_execution_budget(config, amended_manifest, source_latest, state, 1800, diagnostic_fork=fork)
    with pytest.raises(ValueError):
        training._resume_execution_budget(config, {}, source_latest, state, 5000)
