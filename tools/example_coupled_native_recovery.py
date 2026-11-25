#!/usr/bin/env python3
"""Verify coupled native training and saved-config resume in fresh processes.

The seven-field analytic fixture is software evidence only. It fits two
two-epoch lineages, one uninterrupted and one resumed after epoch one.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.coupled_initialization import _same
from honf_runtime.storage_policy import _ensure_workspace_storage_bindings, verify_storage


def recipe_for(root):
    return {'schema': 'coupled_native_recovery_example_v1', 'name': 'Sampled_Reader_Seven_Field_Recovery',
        'dataset': 'analytic_fixture', 'architecture': 'relation_anchored', 'seed': 7, 'sampling_seed': 7,
        'total_epochs': 2, 'microbatch_size': 2, 'effective_batch_size': 2, 'fixture': {'output_dim': 7},
        'training_profile': 'observation_response', 'observation_response': {
            'adapter': 'analytic_fixture', 'mode': 'coupled_native', 'variant': 'sampled_reader_response',
            'pair_seed': 11, 'reader_seed': 13, 'anchor_count': 4,
            'epoch_zero_checkpoint': str(root / 'common_epoch_zero.pt')},
        'execution': {'monitor_every': 1, 'monitor_epochs': [1, 2], 'checkpoint_epochs': [1, 2],
                      'latest_every': 1, 'curve_every': 10, 'progress_enabled': False},
        'storage': {'output_root': str(root), 'max_output_bytes': 2 * 1024**3,
                    'max_peak_temporary_bytes': 1024**3, 'minimum_free_headroom_bytes': 300 * 1024**3}}


def _rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def _reverse_keys(value):
    if isinstance(value, dict):
        return {key: _reverse_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


def check_runs(uninterrupted, resumed):
    left = load_trusted_checkpoint(uninterrupted / 'checkpoints/latest_model.pt', map_location='cpu')
    right = load_trusted_checkpoint(resumed / 'checkpoints/latest_model.pt', map_location='cpu')
    keys = ('epoch', 'model_state_dict', 'optimizer_state_by_name', 'rng_state', 'sampler_state',
            'provider_training_state', 'best_field_score', 'optimizer_group_names', 'optimizer_schedule_contract')
    for key in keys:
        if not _same(left[key], right[key]):
            raise AssertionError(f'Fresh-process continuation changed {key}.')
    left_def = json.loads((uninterrupted / 'configs/metric_definitions.json').read_text())
    right_def = json.loads((resumed / 'configs/metric_definitions.json').read_text())
    if left_def != right_def:
        raise AssertionError('Ordered recording or loss definitions changed on resume.')
    left_rows, right_rows = _rows(uninterrupted / 'metrics/history.csv'), _rows(resumed / 'metrics/history.csv')
    if len(left_rows) != 2 or len(right_rows) != 2:
        raise AssertionError('Both lineages must contain two complete epochs.')
    volatile = {'train_seconds', 'work_counts.coupled_plan_build_seconds', 'work_counts.coupled_plan_builds'}
    for before, after in zip(left_rows, right_rows):
        for key in left_def['training_columns']:
            if key not in volatile and before[key] != after[key]:
                raise AssertionError(f'Recorded {key} differs at epoch {before["epoch"]}.')
    if _rows(uninterrupted / 'metrics/validation.csv') != _rows(resumed / 'metrics/validation.csv'):
        raise AssertionError('The two dual-reader validation histories differ.')
    for run in (uninterrupted, resumed):
        for suffix in ('png', 'pdf'):
            if not list((run / 'plots/training').glob(f'loss_curves*.{suffix}')):
                raise AssertionError(f'Training curves were not saved under {run}.')
    return {'completed_epochs_per_lineage': 2, 'charged_fixture_epochs': 4,
            'checkpoint_state_equal': list(keys), 'recorded_columns_and_losses_equal': True,
            'validation_equal': True, 'elapsed_and_cache_work_excluded': sorted(volatile)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Choose a new empty fixture root; existing artifacts are protected.')
    recipe = recipe_for(root)
    _ensure_workspace_storage_bindings()
    verify_storage(recipe, output=root, estimated_bytes=1024**3, peak_temporary_bytes=1024**3)
    root.mkdir(parents=True, exist_ok=True)
    recipe_path = root / 'fixture_recipe.json'
    recipe_path.write_text(json.dumps(recipe, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    prepare, launcher = PROJECT / 'tools/observation_response_prepare.py', PROJECT / 'tools/relation_anchored_train.py'
    whole, resumed = root / 'Uninterrupted_Coupled_Fixture', root / 'Resumed_Coupled_Fixture'
    commands = [('prepare', [prepare, '--fresh-epoch-zero', '--recipe', recipe_path,
                             '--output', root / 'common_epoch_zero.pt']),
                ('uninterrupted', [launcher, 'fit', '--recipe', recipe_path, '--output', whole, '--stop-after', '2']),
                ('initial', [launcher, 'fit', '--recipe', recipe_path, '--output', resumed, '--stop-after', '1'])]
    for name, command in commands:
        completed = subprocess.run([sys.executable, *map(str, command)], cwd=PROJECT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (root / f'{name}.log').write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f'{name} failed; inspect {root / (name + ".log")}')
    saved = json.loads((resumed / 'configs/resolved_config.json').read_text())
    reordered = root / 'reordered_saved_recipe.json'
    reordered.write_text(json.dumps(_reverse_keys(saved), indent=1) + '\n')
    command = [sys.executable, str(launcher), 'fit', '--recipe', str(reordered), '--output', str(resumed),
               '--resume', 'latest', '--stop-after', '2']
    completed = subprocess.run(command, cwd=PROJECT, env=env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    (root / 'resume.log').write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f'Resume failed; inspect {root / "resume.log"}')
    receipt = check_runs(whole, resumed)
    receipt['reordered_saved_config_reloaded'] = True
    (root / 'recovery_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
