#!/usr/bin/env python3
"""Check an observation recipe with uninterrupted and fresh-process recovery.

Runs two epochs uninterrupted and the same two epochs with an intentional
process interruption immediately after the first committed checkpoint. The
recipe's scientific horizon and optimizer schedule are left unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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


def _sampled_reader_recipe(response):
    """Accept both the legacy variant and the independent reader policy."""
    return (response.get('mode') == 'coupled_native'
            and (response.get('reader_policy') == 'sampled_equal_probability'
                 or (response.get('reader_policy') is None
                     and response.get('variant') == 'sampled_reader_response')))


def _supported_recipe(recipe):
    return (_sampled_reader_recipe(recipe.get('observation_response', {}))
            or isinstance(recipe.get('finite_observations'), dict))


def _recipe_horizon(recipe):
    training = recipe.get('training', {})
    return int(training['local_horizon'] if 'local_horizon' in training else recipe['total_epochs'])


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


def _finite_definitions(run):
    """Read a run's saved objective and the referenced finite source definitions."""
    path = run / 'configs/finite_observation_objective.json'
    pointer = json.loads(path.read_text(encoding='utf-8'))
    if pointer.get('schema') != 'finite_observation_run_definition_v1':
        raise AssertionError(f'Invalid saved finite objective definition: {path}')
    source = Path(pointer['source_definitions']).expanduser()
    if not source.is_absolute():
        source = path.parent / source
    definition = json.loads(source.read_text(encoding='utf-8'))
    if definition.get('schema') != 'finite_observation_source_definitions_v1':
        raise AssertionError(f'Invalid saved finite source definition: {source}')
    return pointer, definition


def _check_finite_artifacts(runs, checkpoints):
    """Check all saved rows, extraction maps, objective settings and calibration."""
    from honf_runtime.finite_observation_task import _json_value

    artifacts = [_finite_definitions(run) for run in runs]
    comparable = [{key: value for key, value in pointer.items()
                   if key not in ('source_definitions', 'metadata_bytes')}
                  for pointer, _ in artifacts]
    if not _same(comparable[0], comparable[1]):
        raise AssertionError('Saved finite objectives or calibration coefficients changed on resume.')
    if not _same(artifacts[0][1], artifacts[1][1]):
        raise AssertionError('Saved finite probe IDs, measures or extraction definitions changed on resume.')
    for (pointer, _), checkpoint in zip(artifacts, checkpoints):
        state = checkpoint['provider_training_state']['finite_observations']
        if state.get('schema') != 'finite_observation_training_state_v1':
            raise AssertionError('Missing finite-observation provider recovery state.')
        if not _same(pointer['calibration'], _json_value(state['calibration'])):
            raise AssertionError('Saved finite calibration differs from the checkpoint provider state.')
        identity = checkpoint['experiment_identity']['provider_identity']['finite_observations']
        if not _same(pointer['objective'], _json_value(identity)):
            raise AssertionError('Saved finite objective differs from the checkpoint provider identity.')
    return {'finite_objective_and_calibration_equal': True,
            'finite_source_definitions_equal': True,
            'finite_provider_and_reader_state_equal': True}


def _check_rebuilt_finite_scenes(provider, definition):
    """Compare every saved scene's compact source rows to the rebuilt provider."""
    import numpy as np
    from honf_runtime.finite_observation_task import _json_value

    frame_fn = getattr(provider.observation_binding, 'finite_observation_frame', None)
    frame = frame_fn() if callable(frame_fn) else None
    basis = provider.finite_observation_spec.get('basis', 'coordinate')
    actual = []
    for scene in provider.finite_observation_scenes.values():
        families = []
        for family in scene.families:
            actual_frame = frame if frame is not None and frame.identity == family.frame else None
            actual_basis = basis if actual_frame is not None and basis != 'constant' else 'constant'
            out_of_frame = 0
            if actual_frame is not None and actual_basis != 'constant':
                coordinates = np.asarray(family.coordinates, dtype=np.float64)
                out_of_frame = int(np.count_nonzero(((coordinates < np.asarray(actual_frame.lower)) |
                                                    (coordinates > np.asarray(actual_frame.upper))).any(axis=1)))
            families.append({'name': family.name, 'component': asdict(family.component),
                'support': family.support, 'frame': family.frame, 'time': family.time,
                'extraction_id': family.extraction_id, 'native_weight': family.native_weight,
                'row_ids': family.row_ids, 'coordinates': family.coordinates, 'measures': family.measures,
                'actual_basis': actual_basis, 'frame_definition': None if actual_frame is None else asdict(actual_frame),
                'out_of_frame_count': out_of_frame,
                'reference': 'existing adapter source arrays indexed by these row_ids; no duplicate field arrays',
                'extraction': asdict(family.extraction)})
        actual.append({'scene_id': scene.scene_id, 'families': families})
    actual = _json_value(actual)
    if actual != definition['scenes']:
        raise AssertionError('Rebuilt finite probe IDs, measures or extraction definitions differ from the saved source.')
    return {'scene_count': len(actual), 'family_count': sum(len(scene['families']) for scene in actual),
            'saved_and_rebuilt_definitions_equal': True}


def _sample_worker(run, destination):
    """Record the next named native and reader draws without model inference."""
    sys.path.insert(0, str(PROJECT / 'tools'))
    from relation_anchored_train import build
    from honf_runtime.unified_training import SamplingKey

    checkpoint = load_trusted_checkpoint(run / 'checkpoints/latest_model.pt', map_location='cpu')
    recipe = checkpoint['experiment_identity']['recipe']
    model, provider = build(recipe, device='cpu', resume_payload=checkpoint)
    provider.load_training_state_dict(checkpoint['provider_training_state'])
    pairs = getattr(provider, 'coupled_native_pairs', None)
    if pairs is None:
        pairs = provider.finite_observation_pairs
    pair = pairs[0]
    epoch = int(checkpoint['sampler_state']['next_epoch'])
    key = SamplingKey(provider.seed, epoch, 0, 0, 'joint', 'reader',
        sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=provider.sampling_dataset_id)
    result = {'epoch': epoch, 'pair': list(pair), 'reader_draws': []}
    if 'finite_observations' in recipe:
        pointer, definition = _finite_definitions(run)
        result['finite_definitions'] = _check_rebuilt_finite_scenes(provider, definition)
        result['finite_calibration_coefficients'] = pointer['calibration']['coefficients']
        result['finite_state_before_draws'] = {
            key: value for key, value in provider.training_state_dict()['finite_observations'].items()
            if key != 'calibration'}
        result['finite_reader_draws'] = []
    sampler = getattr(provider, 'sample_coupled_native_rows', None)
    if callable(sampler):
        rows, carriers = sampler(pair, key)
        if len(carriers) != 1 or set(rows) != set(pair):
            raise AssertionError('Expected one coupled sample for the next paired native draw.')
        sample = carriers[0][1]
        fields = ('left_rows', 'right_rows', 'left_anchor_rows', 'right_anchor_rows',
                  'left_anchor_positions', 'right_anchor_positions', 'common', 'common_row_probability')
        result['sample'] = {name: getattr(sample, name).tolist() for name in fields}
        result['native_sampling'] = 'coupled_native_rows'
    else:
        from honf_runtime.finite_observation_task import _json_value
        batch = provider.make_batch(pair, key)
        result['sample'] = _json_value(batch.receivers)
        result['native_sampling'] = 'native_make_batch_receivers'
    for index in range(4):
        provider.begin_native_update(epoch=epoch, update_index=index, update_count=4)
        mode = getattr(provider, 'coupled_native_reader_mode', None)
        if mode is None:
            mode = getattr(provider, 'finite_observation_reader_mode', None)
        result['reader_draws'].append(mode)
        if 'finite_observations' in recipe:
            result['finite_reader_draws'].append(provider.finite_observation_reader_mode)
    if 'finite_observations' in recipe:
        result['finite_state_after_draws'] = {
            key: value for key, value in provider.training_state_dict()['finite_observations'].items()
            if key != 'calibration'}
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
    recipes = [checkpoint['experiment_identity']['recipe'] for checkpoint in (left, right)]
    finite = ['finite_observations' in recipe for recipe in recipes]
    if finite[0] != finite[1]:
        raise AssertionError('Finite-observation attachment changed on resume.')
    finite_results = {}
    if finite[0]:
        volatile.update({'work_counts.finite_observations_planning_seconds',
                         'work_counts.finite_observations_prediction_seconds'})
        finite_results = _check_finite_artifacts((uninterrupted, resumed), (left, right))
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
            'elapsed_and_cache_work_excluded': sorted(volatile), **finite_results}


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
    sampler = original_draws.get('native_sampling')
    coverage = ('First supported pair native and anchor rows' if sampler == 'coupled_native_rows'
                else 'First eligible pair native make_batch receiver rows')
    if 'finite_observations' in recipe:
        coverage += ', all rebuilt finite source definitions and four reader draws.'
    else:
        coverage += ' and four reader draws.'
    receipt.update(reordered_saved_config_reloaded=True, next_native_and_reader_draws_equal=True,
                   next_draw_coverage=coverage,
                   interruption=interruption, scientific_horizon=_recipe_horizon(recipe),
                   optimizer_schedule_horizon=recipe.get('training', {}).get('optimizer_schedule_horizon'))
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
        if not _supported_recipe(recipe):
            raise ValueError('The saved interrupted recipe must have sampled coupled readers or finite observations.')
        _ensure_workspace_storage_bindings()
        verify_storage(recipe, output=root, estimated_bytes=1024**3, peak_temporary_bytes=1024**3)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        _run([sys.executable, str(Path(__file__).resolve()), '--output-root', str(root),
              '--prepare-interrupted-worker'], log=root / 'interrupted_recovery_preparation.log',
             env=dict(env, CUDA_VISIBLE_DEVICES=''))
        launcher = PROJECT / 'tools/relation_anchored_train.py'
        _run([sys.executable, str(launcher), 'fit', '--recipe', str(root / 'reordered_saved_recipe.json'),
              '--output', str(resumed), '--resume', 'latest', '--stop-after', '2', '--device', args.device],
             log=root / 'resumed_after_repair.log', env=env)
        _finish(root, recipe, env, interruption='Fresh-process continuation of an existing committed epoch-one '
                'interruption; prior attempt logs and the committed checkpoint are preserved.')
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
    if not _supported_recipe(recipe):
        raise ValueError('Recovery qualification requires sampled coupled readers or finite observations.')
    if _recipe_horizon(recipe) < 2:
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
