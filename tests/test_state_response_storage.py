"""State-response storage planning uses the maintained DATA2 budget policy."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from honf_runtime import state_response_storage as storage

MIB = 1024 ** 2


@pytest.fixture
def preflight(tmp_path, monkeypatch):
    data2 = tmp_path / 'data2'
    data2.mkdir()
    root = data2 / 'runs'
    root.mkdir()
    config = {'storage': {'output_root': str(root), 'max_output_bytes': 256 * MIB,
                          'max_peak_temporary_bytes': 64 * MIB,
                          'minimum_free_headroom_bytes': 128 * MIB}, 'training': {}}
    monkeypatch.setattr(storage, '_DATA2', data2)
    monkeypatch.setattr(storage.storage_policy, '_ensure_workspace_storage_bindings',
                        lambda: {'workspace/Trained_Results': 'data2/Trained_Results'})
    calls = []
    def verify(recipe, *, output, estimated_bytes, peak_temporary_bytes):
        calls.append((recipe, Path(output), estimated_bytes, peak_temporary_bytes))
        limits = recipe['storage']
        if estimated_bytes > limits['max_output_bytes']:
            raise RuntimeError('output budget exceeded')
        if peak_temporary_bytes > limits['max_peak_temporary_bytes']:
            raise RuntimeError('temporary budget exceeded')
        return {'free_bytes_before': 1024 * MIB, 'output_bytes_before': 0,
                'minimum_free_headroom_bytes': limits['minimum_free_headroom_bytes'],
                'output_budget_bytes': limits['max_output_bytes'],
                'maximum_peak_temporary_bytes': limits['max_peak_temporary_bytes'],
                'estimated_new_bytes': estimated_bytes,
                'estimated_peak_temporary_bytes': peak_temporary_bytes}
    monkeypatch.setattr(storage.storage_policy, 'verify_storage', verify)
    return config, root, calls


def test_fresh_preflight_estimates_checkpoints_and_peak_without_creating_run(preflight):
    config, root, calls = preflight
    output = root / 'fresh'
    model = torch.nn.Linear(1, 1)
    result = storage.prepare_state_response_storage(config, output, model, completed_epoch=0,
                                                   stop_after=200, checkpoint_epochs=(100, 200))
    state_bytes = sum(p.numel() * p.element_size() for p in model.state_dict().values())
    payload = 4 * state_bytes + 8 * MIB
    assert result['estimated_new_bytes'] == 4 * payload + 64 * MIB
    assert result['estimated_peak_temporary_bytes'] == payload + 32 * MIB
    assert result['workspace_bindings'] == {'workspace/Trained_Results': 'data2/Trained_Results'}
    assert result['checkpoint_epochs'] == [100, 200]
    assert result['run_output'] == str(output)
    assert len(calls) == 2 and calls[0][2:] == (0, 0)
    assert not output.exists()


def test_resume_counts_only_new_checkpoints_and_preserves_existing_run(preflight):
    config, root, calls = preflight
    output = root / 'existing'
    output.mkdir()
    (output / 'history.json').write_text('preserved')
    result = storage.prepare_state_response_storage(config, output, torch.nn.Linear(1, 1),
        completed_epoch=100, stop_after=200, checkpoint_epochs=(100, 200))
    payload = 4 * 8 + 8 * MIB
    assert result['estimated_new_bytes'] == payload + 64 * MIB
    assert result['completed_epoch'] == 100
    assert (output / 'history.json').read_text() == 'preserved'
    assert calls[1][2] == result['estimated_new_bytes']


def test_budget_and_headroom_fail_before_run_creation(preflight):
    config, root, calls = preflight
    output = root / 'too-large'
    config['storage']['max_output_bytes'] = 32 * MIB
    with pytest.raises(RuntimeError, match='output budget'):
        storage.prepare_state_response_storage(config, output, torch.nn.Linear(1, 1),
            completed_epoch=0, stop_after=100, checkpoint_epochs=(100,))
    assert len(calls) == 2 and not output.exists()
    config['storage']['max_output_bytes'] = 256 * MIB
    config['storage']['minimum_free_headroom_bytes'] = 1000 * MIB
    with pytest.raises(RuntimeError, match='headroom'):
        storage.prepare_state_response_storage(config, output, torch.nn.Linear(1, 1),
            completed_epoch=0, stop_after=100, checkpoint_epochs=(100,))
    assert not output.exists()


def test_refuses_nonempty_fresh_output_and_escaped_or_missing_binding(preflight, monkeypatch, tmp_path):
    config, root, calls = preflight
    output = root / 'existing'
    output.mkdir()
    (output / 'keep').write_text('user data')
    with pytest.raises(FileExistsError):
        storage.prepare_state_response_storage(config, output, torch.nn.Linear(1, 1),
            completed_epoch=0, stop_after=100, checkpoint_epochs=(100,))
    assert (output / 'keep').read_text() == 'user data' and not calls
    outside = tmp_path / 'outside'
    outside.mkdir()
    with pytest.raises(ValueError, match='inside'):
        storage.prepare_state_response_storage(config, outside / 'run', torch.nn.Linear(1, 1),
            completed_epoch=0, stop_after=100, checkpoint_epochs=(100,))
    (root / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='inside'):
        storage.prepare_state_response_storage(config, root / 'escape' / 'run', torch.nn.Linear(1, 1),
            completed_epoch=0, stop_after=100, checkpoint_epochs=(100,))
    def missing():
        raise RuntimeError('required DATA2 binding is missing')
    monkeypatch.setattr(storage.storage_policy, '_ensure_workspace_storage_bindings', missing)
    with pytest.raises(RuntimeError, match='binding is missing'):
        storage.prepare_state_response_storage(config, root / 'new', torch.nn.Linear(1, 1),
            completed_epoch=0, stop_after=100, checkpoint_epochs=(100,))
    assert not (root / 'new').exists()


def test_storage_module_imports_without_thermal_case(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / 'src'
    environment = {**os.environ, 'PYTHONPATH': str(source_root), 'PYTHONDONTWRITEBYTECODE': '1',
                   'CUDA_VISIBLE_DEVICES': ''}
    script = """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'channelthermal' or name.startswith('channelthermal.'):
        raise AssertionError(f'unexpected case dependency: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from honf_runtime.state_response_storage import prepare_state_response_storage
assert callable(prepare_state_response_storage)
assert 'honf_runtime.research_training' not in sys.modules
assert 'honf_runtime.contextual_frontier_training' not in sys.modules
assert not any(name == 'channelthermal' or name.startswith('channelthermal.') for name in sys.modules)
"""
    subprocess.run([sys.executable, '-B', '-c', script], env=environment, cwd=tmp_path,
                   check=True, capture_output=True, text=True)


def test_legacy_storage_entry_points_preserve_injected_helpers(tmp_path, monkeypatch):
    from honf_runtime import contextual_frontier_training as legacy

    calls = []
    def verify(config, *, output, estimated_bytes, peak_temporary_bytes):
        calls.append((config, output, estimated_bytes, peak_temporary_bytes))
        return {'free_bytes_before': 100, 'minimum_free_headroom_bytes': 75}
    monkeypatch.setattr(legacy.research, 'verify_storage', verify)
    with pytest.raises(RuntimeError, match='headroom'):
        legacy._prepare_storage({}, tmp_path, estimated_bytes=20, peak_temporary_bytes=10)
    assert calls == [({}, tmp_path, 20, 10)]
    monkeypatch.setattr(legacy, '_model_state_bytes', lambda model: 3)
    estimate, peak = legacy._storage_estimate(None, completed_epoch=0, stop_after=100,
        config={'training': {'checkpoint_epochs': [100]}})
    assert estimate == 3 * (4 * 3 + 8 * MIB) + 64 * MIB
    assert peak == 4 * 3 + 40 * MIB
