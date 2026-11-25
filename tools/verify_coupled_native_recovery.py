#!/usr/bin/env python3
"""Check a coupled recipe with uninterrupted and fresh-process recovery.

Runs two epochs uninterrupted and the same two epochs with an intentional
process interruption immediately after the first committed checkpoint. The
recipe's scientific horizon and optimizer schedule are left unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src', PROJECT / 'Case_WindFarm/src'):
    sys.path.insert(0, str(directory))

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.coupled_initialization import _same
from honf_runtime.storage_policy import _ensure_workspace_storage_bindings, verify_storage


def _reverse_keys(value):
    if isinstance(value, dict):
        return {key: _reverse_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


def _rows(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def _render_epoch_one(layout, payload):
    """Render already-committed scalars without running model inference."""
    import honf_runtime.unified_training as runtime

    history_path = layout.read_path('history.csv')
    history = runtime.read_training_history(history_path)
    if len(history) != 1 or int(history[0]['epoch']) != 1:
        raise AssertionError('Expected exactly one committed epoch in scalar history.')
    definitions_path = layout.read_path('configs/metric_definitions.json')
    definitions = json.loads(definitions_path.read_text(encoding='utf-8'))
    identity = payload['experiment_identity']
    field_metric = identity['selection_policy']['field_metric']
    rng = runtime.capture_rng_state()
    try:
        runtime._render_loss_curves(history, layout.write_path('loss_curves.png').parent, field_metric,
            loss_metadata=definitions['terms'], provider_identity=identity['provider_identity'],
            metadata_output=history_path.parent, separate_totals=True)
    finally:
        runtime.restore_rng_state(rng)


def _interrupt_worker(recipe, output, device):
    """Interrupt only after the epoch-one checkpoint and scalar rows commit."""
    import honf_runtime.unified_training as runtime

    original = runtime._publish_csv_checkpoint

    def publish_and_interrupt(layout, payload):
        original(layout, payload)
        if int(payload['epoch']) != 1:
            return
        _render_epoch_one(layout, payload)
        print('Intentional interruption after committed epoch one and its first curves.', flush=True)
        sys.stderr.flush()
        os._exit(73)

    runtime._publish_csv_checkpoint = publish_and_interrupt
    launcher = PROJECT / 'tools/relation_anchored_train.py'
    sys.argv = [str(launcher), 'fit', '--recipe', str(recipe), '--output', str(output),
                '--stop-after', '2', '--device', device]
    runpy.run_path(str(launcher), run_name='__main__')
    raise AssertionError('The intended epoch-one interruption did not occur.')


def _prepare_interrupted(root):
    """Preserve and plot an already committed epoch-one interrupted state."""
    from honf_runtime.run_layout import RunLayout

    run = root / 'Interrupted_Sampled_Reader_Recovery'
    layout = RunLayout(run)
    latest = layout.read_path('latest_model.pt')
    payload = load_trusted_checkpoint(latest, map_location='cpu')
    if (int(payload['epoch']) != 1 or payload['sampler_state']['next_epoch'] != 2
            or not payload['optimizer_state_by_name']['state_by_name']):
        raise ValueError('Expected a committed epoch-one checkpoint with optimizer state and next epoch two.')
    saved = json.loads(layout.read_path('resolved_config.json').read_text(encoding='utf-8'))
    if saved != payload['experiment_identity']['recipe']:
        raise ValueError('Saved configuration differs from the committed checkpoint recipe.')
    _render_epoch_one(layout, payload)
    preserved = root / 'Interrupted_Epoch_One_Model.pt'
    if preserved.exists():
        if not os.path.samefile(latest, preserved):
            raise FileExistsError('The preserved epoch-one checkpoint belongs to a different state.')
    else:
        os.link(latest, preserved)
    reordered = root / 'reordered_saved_recipe.json'
    reversed_saved = _reverse_keys(saved)
    if reordered.exists():
        if json.loads(reordered.read_text(encoding='utf-8')) != reversed_saved:
            raise FileExistsError('The existing reordered saved configuration differs.')
    else:
        reordered.write_text(json.dumps(reversed_saved, indent=2) + '\n', encoding='utf-8')


def _sample_worker(run, destination):
    """Record the next named native and reader draws without model inference."""
    sys.path.insert(0, str(PROJECT / 'tools'))
    from relation_anchored_train import build
    from honf_runtime.unified_training import SamplingKey

    checkpoint = load_trusted_checkpoint(run / 'checkpoints/latest_model.pt', map_location='cpu')
    recipe = checkpoint['experiment_identity']['recipe']
    model, provider = build(recipe, device='cpu', resume_payload=checkpoint)
    provider.load_training_state_dict(checkpoint['provider_training_state'])
    pair = provider.coupled_native_pairs[0]
    epoch = int(checkpoint['sampler_state']['next_epoch'])
    key = SamplingKey(recipe['training']['seed'], epoch, 0, 0, 'joint', 'reader',
        sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=provider.sampling_dataset_id)
    rows, carriers = provider.sample_coupled_native_rows(pair, key)
    if len(carriers) != 1 or set(rows) != set(pair):
        raise AssertionError('Expected one coupled sample for the next paired native draw.')
    sample = carriers[0][1]
    fields = ('left_rows', 'right_rows', 'left_anchor_rows', 'right_anchor_rows',
              'left_anchor_positions', 'right_anchor_positions', 'common', 'common_row_probability')
    result = {'epoch': epoch, 'pair': list(pair),
              'sample': {name: getattr(sample, name).tolist() for name in fields},
              'reader_draws': []}
    for index in range(4):
        provider.begin_native_update(epoch=epoch, update_index=index, update_count=4)
        result['reader_draws'].append(provider.coupled_native_reader_mode)
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


def _check_runs(uninterrupted, resumed, interrupted_checkpoint):
    left = load_trusted_checkpoint(uninterrupted / 'checkpoints/latest_model.pt', map_location='cpu')
    right = load_trusted_checkpoint(resumed / 'checkpoints/latest_model.pt', map_location='cpu')
    checkpoint_one = load_trusted_checkpoint(interrupted_checkpoint, map_location='cpu')
    if checkpoint_one['epoch'] != 1 or checkpoint_one['sampler_state']['next_epoch'] != 2:
        raise AssertionError('The interruption must preserve a complete epoch-one recovery state.')
    if not checkpoint_one['optimizer_state_by_name']['state_by_name']:
        raise AssertionError('The interrupted checkpoint must contain the actual optimizer moments.')
    keys = ('epoch', 'model_state_dict', 'optimizer_state_by_name', 'rng_state', 'sampler_state',
            'provider_training_state', 'best_field_score', 'best_response_guarded_score',
            'optimizer_group_names', 'optimizer_schedule_contract')
    for key in keys:
        if not _same(left[key], right[key]):
            raise AssertionError(f'Fresh-process continuation changed {key}.')
    definitions = []
    for run in (uninterrupted, resumed):
        definitions.append(json.loads((run / 'configs/metric_definitions.json').read_text(encoding='utf-8')))
    if definitions[0] != definitions[1]:
        raise AssertionError('Ordered recording or loss definitions changed on resume.')
    columns = definitions[0]['training_columns']
    if not definitions[0]['training_loss_order'] or not definitions[0]['training_total_order']:
        raise AssertionError('Loss accumulation orders must be explicitly saved.')
    left_rows = _rows(uninterrupted / 'metrics/history.csv')
    right_rows = _rows(resumed / 'metrics/history.csv')
    if [row['epoch'] for row in left_rows] != ['1', '2'] or [row['epoch'] for row in right_rows] != ['1', '2']:
        raise AssertionError('Both lineages must contain precisely two completed epoch rows.')
    volatile = {'train_seconds', 'work_counts.coupled_plan_build_seconds', 'work_counts.coupled_plan_builds'}
    for before, after in zip(left_rows, right_rows):
        if list(before) != columns or list(after) != columns:
            raise AssertionError('Saved CSV header differs from the ordered recording definitions.')
        for key in columns:
            if key not in volatile and before[key] != after[key]:
                raise AssertionError(f'Recorded {key} differs at epoch {before["epoch"]}.')
    if _rows(uninterrupted / 'metrics/validation.csv') != _rows(resumed / 'metrics/validation.csv'):
        raise AssertionError('The two dual-reader validation histories differ.')
    for run in (uninterrupted, resumed):
        for name in ('loss_curves_initial.png', 'loss_curves_initial.pdf', 'loss_curves.png', 'loss_curves.pdf'):
            if not (run / 'plots/training' / name).is_file():
                raise AssertionError(f'Missing training curve {run / "plots/training" / name}.')
    return {'completed_epochs_per_lineage': 2, 'charged_training_epochs': 4,
            'checkpoint_state_equal': list(keys), 'interrupted_epoch_one_checkpoint_preserved': True,
            'recorded_columns_and_losses_equal': True, 'validation_equal': True,
            'elapsed_and_cache_work_excluded': sorted(volatile)}


def _run(command, *, log, env, expected=0):
    with log.open('w', encoding='utf-8') as stream:
        completed = subprocess.run(command, cwd=PROJECT, env=env, text=True,
                                   stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != expected:
        raise RuntimeError(f'Expected exit {expected}, received {completed.returncode}; inspect {log}.')


def _finish(root, recipe, env, *, interruption):
    whole = root / 'Uninterrupted_Sampled_Reader_Recovery'
    resumed = root / 'Interrupted_Sampled_Reader_Recovery'
    receipt = _check_runs(whole, resumed, root / 'Interrupted_Epoch_One_Model.pt')
    sample_env = dict(env, CUDA_VISIBLE_DEVICES='')
    for name, run in (('uninterrupted', whole), ('resumed', resumed)):
        _run([sys.executable, str(Path(__file__).resolve()), '--output-root', str(run),
              '--sample-worker', str(root / f'{name}_next_draws.json')],
             log=root / f'{name}_next_draws.log', env=sample_env)
    original_draws = json.loads((root / 'uninterrupted_next_draws.json').read_text(encoding='utf-8'))
    resumed_draws = json.loads((root / 'resumed_next_draws.json').read_text(encoding='utf-8'))
    if original_draws != resumed_draws:
        raise AssertionError('Next native pair, query, completion, ordering or reader streams differ.')
    receipt.update(reordered_saved_config_reloaded=True, next_native_and_reader_draws_equal=True,
                   next_draw_coverage='First supported pair native panel and anchor draws, plus four reader draws.',
                   interruption=interruption, scientific_horizon=recipe['training']['local_horizon'],
                   optimizer_schedule_horizon=recipe['training'].get('optimizer_schedule_horizon'))
    (root / 'recovery_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--continue-interrupted', action='store_true',
                        help='Continue an existing committed epoch-one interruption through epoch two.')
    parser.add_argument('--interrupt-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--prepare-interrupted-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--sample-worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.sample_worker:
        _sample_worker(args.output_root, args.sample_worker)
        return
    if args.prepare_interrupted_worker:
        _prepare_interrupted(args.output_root)
        return
    if args.continue_interrupted:
        root = args.output_root.expanduser().resolve(strict=True)
        resumed = root / 'Interrupted_Sampled_Reader_Recovery'
        saved_path = resumed / 'configs/resolved_config.json'
        recipe = json.loads(saved_path.read_text(encoding='utf-8'))
        if (recipe.get('observation_response', {}).get('mode') != 'coupled_native'
                or recipe['observation_response'].get('variant') != 'sampled_reader_response'):
            raise ValueError('The saved interrupted recipe must be sampled-reader coupled native.')
        _ensure_workspace_storage_bindings()
        verify_storage(recipe, output=root, estimated_bytes=1024**3, peak_temporary_bytes=1024**3)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        _run([sys.executable, str(Path(__file__).resolve()), '--output-root', str(root),
              '--prepare-interrupted-worker'], log=root / 'interrupted_recovery_preparation.log',
             env=dict(env, CUDA_VISIBLE_DEVICES=''))
        launcher = PROJECT / 'tools/relation_anchored_train.py'
        _run([sys.executable, str(launcher), 'fit', '--recipe', str(root / 'reordered_saved_recipe.json'),
              '--output', str(resumed), '--resume', 'latest', '--stop-after', '2', '--device', args.device],
             log=root / 'resumed_after_publication_failure.log', env=env)
        _finish(root, recipe, env, interruption='Publication completed; the original curve helper then failed. '
                'Its failure log and committed epoch-one checkpoint are preserved.')
        return
    if args.recipe is None:
        parser.error('--recipe is required')
    if args.interrupt_worker:
        _interrupt_worker(args.recipe, args.output_root, args.device)
        return
    root = args.output_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Choose a new empty recovery root; existing artifacts are protected.')
    recipe = json.loads(args.recipe.read_text(encoding='utf-8'))
    response = recipe.get('observation_response', {})
    if response.get('mode') != 'coupled_native' or response.get('variant') != 'sampled_reader_response':
        raise ValueError('Recovery qualification requires a sampled-reader coupled native recipe.')
    if int(recipe['training']['local_horizon']) < 2:
        raise ValueError('The saved scientific training horizon must include at least two epochs.')
    _ensure_workspace_storage_bindings()
    verify_storage(recipe, output=root, estimated_bytes=1024**3, peak_temporary_bytes=1024**3)
    root.mkdir(parents=True, exist_ok=True)
    whole = root / 'Uninterrupted_Sampled_Reader_Recovery'
    resumed = root / 'Interrupted_Sampled_Reader_Recovery'
    recipe_path = args.recipe.expanduser().resolve()
    launcher = PROJECT / 'tools/relation_anchored_train.py'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    base = [sys.executable, str(launcher), 'fit', '--recipe', str(recipe_path),
            '--stop-after', '2', '--device', args.device]
    _run([*base, '--output', str(whole)], log=root / 'uninterrupted.log', env=env)
    _run([sys.executable, str(Path(__file__).resolve()), '--interrupt-worker', '--recipe', str(recipe_path),
          '--output-root', str(resumed), '--device', args.device], log=root / 'interrupted.log', env=env, expected=73)
    first = resumed / 'checkpoints/latest_model.pt'
    preserved = root / 'Interrupted_Epoch_One_Model.pt'
    os.link(first, preserved)
    saved = json.loads((resumed / 'configs/resolved_config.json').read_text(encoding='utf-8'))
    reordered = root / 'reordered_saved_recipe.json'
    reordered.write_text(json.dumps(_reverse_keys(saved), indent=2) + '\n', encoding='utf-8')
    _run([sys.executable, str(launcher), 'fit', '--recipe', str(reordered), '--output', str(resumed),
          '--resume', 'latest', '--stop-after', '2', '--device', args.device], log=root / 'resumed.log', env=env)
    _finish(root, recipe, env, interruption='Intentional exit after epoch-one publication completed.')


if __name__ == '__main__':
    main()
