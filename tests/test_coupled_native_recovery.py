"""Recovery preparation uses committed artifacts without another optimizer update."""
import importlib.util
import json
from pathlib import Path

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
