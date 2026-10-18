#!/usr/bin/env python3
"""Run an unchanged pinned HONF CLI with an explicitly authorized GPU reservation.

This operational entry point stays outside HONF_Proj so it does not alter the
native model-source inventory. It changes only the CLI's device constant and
its read of the exact historical reservation; the original CLI owns training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError('Expected a JSON object.')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def bound_object(binding, label):
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ValueError(f'{label} needs an exact path/SHA binding.')
    path = Path(binding['path']).resolve(strict=True)
    if str(path) != binding['path'] or file_sha256(path) != binding['sha256']:
        raise ValueError(f'{label} path or SHA differs.')
    return path, read_object(path)


def validate_execution_reservation(recipe, execution_path, execution_sha256, stop_after):
    """Validate metadata only; never initialize CUDA, construct a model or draw RNG."""
    original_path = Path(recipe['resource_reservation']).resolve(strict=True)
    original = read_object(original_path)
    execution_path = Path(execution_path).resolve(strict=True)
    if file_sha256(execution_path) != execution_sha256:
        raise ValueError('Execution reservation SHA differs.')
    effective = read_object(execution_path)
    metadata = effective.get('execution_override')
    if not isinstance(metadata, dict) or set(metadata) != {
        'schema', 'original_reservation', 'authorization', 'terminal_epoch'
    } or metadata['schema'] != 'round9_execution_reservation_v1':
        raise ValueError('Execution override metadata differs.')
    bound_original_path, bound_original = bound_object(metadata['original_reservation'], 'Original reservation')
    if bound_original_path != original_path or canonical(bound_original) != canonical(original):
        raise ValueError('Override does not bind the recipe reservation.')
    if original.get('schema') != 'round9_resource_reservation_v1' or 'execution_override' in original:
        raise ValueError('Historical reservation schema differs.')
    protected_original = {k: v for k, v in original.items() if k not in ('gpu_uuid', 'physical_gpu')}
    protected_effective = {k: v for k, v in effective.items()
                           if k not in ('gpu_uuid', 'physical_gpu', 'execution_override')}
    if canonical(protected_original) != canonical(protected_effective):
        raise ValueError('An original resource field, cap, start, mount or type changed.')
    authorization_path, authorization = bound_object(metadata['authorization'], 'User GPU authorization')
    if authorization.get('schema') != 'round9_user_GPU_lane_authorization_v1':
        raise ValueError('User GPU authorization schema differs.')
    if authorization.get('original_reservation') != metadata['original_reservation']:
        raise ValueError('Authorization original-reservation binding differs.')
    if (type(metadata['terminal_epoch']) is not int or metadata['terminal_epoch'] != 3500
            or authorization.get('terminal_epoch') != 3500
            or authorization.get('training_after_3500_authorized') is not False
            or type(stop_after) is not int or not 0 < stop_after <= 3500):
        raise ValueError('Only the explicit literal e3500 decision stop is authorized.')
    amendment_path, amendment = bound_object(
        authorization.get('matched_decision_stop_amendment'), 'Matched decision-stop amendment')
    if (amendment.get('terminal_epoch') != 3500 or amendment.get('scientific_updates') != 27000
            or amendment.get('training_after_3500_authorized') is not False
            or amendment.get('original_reservation_path') != str(original_path)
            or amendment.get('original_reservation_sha256') != file_sha256(original_path)):
        raise ValueError('Decision-stop amendment differs.')
    allowed = authorization.get('allowed_devices')
    if not isinstance(allowed, list) or not allowed:
        raise ValueError('User authorization has no allowed devices.')
    pairs = []
    for device in allowed:
        if (not isinstance(device, dict) or set(device) != {'physical_gpu', 'gpu_uuid'}
                or type(device['physical_gpu']) is not int or device['physical_gpu'] < 0
                or not isinstance(device['gpu_uuid'], str) or not device['gpu_uuid'].startswith('GPU-')):
            raise ValueError('Malformed authorized physical GPU/UUID pair.')
        pairs.append((device['physical_gpu'], device['gpu_uuid']))
    if len(set(pairs)) != len(pairs) or len({p[0] for p in pairs}) != len(pairs) or len({p[1] for p in pairs}) != len(pairs):
        raise ValueError('Ambiguous authorized device mapping.')
    selected = (effective.get('physical_gpu'), effective.get('gpu_uuid'))
    if type(selected[0]) is not int or selected not in pairs:
        raise ValueError('Execution GPU/UUID pair is not user-authorized.')
    root = original_path.parents[1]
    if any(not path.is_relative_to(root) for path in (execution_path, authorization_path, amendment_path)):
        raise ValueError('Execution authorization must remain in the original reserved round root.')
    return effective, {
        'original_reservation': {'path': str(original_path), 'sha256': file_sha256(original_path)},
        'execution_reservation': {'path': str(execution_path), 'sha256': execution_sha256},
        'authorization': {'path': str(authorization_path), 'sha256': file_sha256(authorization_path)},
        'decision_stop_amendment': {'path': str(amendment_path), 'sha256': file_sha256(amendment_path)},
        'gpu_uuid': selected[1], 'physical_gpu': selected[0], 'terminal_epoch': 3500,
        'training_after_3500_authorized': False,
    }


def validate_snapshot(snapshot, manifest_path, manifest_sha256):
    snapshot = Path(snapshot).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    if file_sha256(manifest_path) != manifest_sha256:
        raise ValueError('Pinned source snapshot manifest SHA differs.')
    manifest = read_object(manifest_path)
    declared = manifest.get('snapshot', manifest.get('root'))
    if declared is None or Path(declared).resolve(strict=True) != snapshot:
        raise ValueError('Pinned snapshot root differs.')
    files = manifest['files']
    seen = set()
    for entry in files:
        relative = entry['path']
        if relative == '.git':
            continue
        path = (snapshot / relative).resolve(strict=True)
        if not path.is_relative_to(snapshot) or relative in seen or file_sha256(path) != entry['sha256']:
            raise ValueError('Pinned source bytes/path differ: ' + relative)
        seen.add(relative)
    cli = snapshot / 'HONF_Proj/tools/joint_native_train.py'
    if 'HONF_Proj/tools/joint_native_train.py' not in seen:
        raise ValueError('Snapshot does not pin the native CLI.')
    return cli, {'snapshot': str(snapshot), 'manifest': str(manifest_path),
                 'manifest_sha256': manifest_sha256, 'cli': str(cli), 'cli_sha256': file_sha256(cli),
                 'verified_source_files': len(seen)}


@contextmanager
def pinned_cli_override(*, snapshot, snapshot_manifest, snapshot_manifest_sha256,
                        recipe, execution_reservation, execution_reservation_sha256, stop_after):
    """Expose the real pinned CLI alias; useful for independent CPU resume preflights."""
    effective, binding = validate_execution_reservation(
        recipe, execution_reservation, execution_reservation_sha256, stop_after)
    cli_path, source_binding = validate_snapshot(snapshot, snapshot_manifest, snapshot_manifest_sha256)
    project = cli_path.parents[1]
    for name, loaded in tuple(sys.modules.items()):
        if name == 'honf_runtime' or name.startswith('honf_runtime.'):
            filename = getattr(loaded, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(project):
                raise ValueError('A different HONF runtime is already imported.')
    alias = '_honf_joint_device_pinned_cli'
    if alias in sys.modules:
        raise ValueError('Pinned CLI override is already active.')
    spec = importlib.util.spec_from_file_location(alias, cli_path)
    module = importlib.util.module_from_spec(spec)
    previous_path = list(sys.path)
    original_read = original_uuid = None
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
        if Path(module.__file__).resolve() != cli_path or Path(module.PROJECT).resolve() != project:
            raise ValueError('Native CLI import root differs.')
        original_read, original_uuid = module.read, module.GPU_UUID

        def execution_read(path):
            if Path(path).resolve() != Path(binding['original_reservation']['path']):
                return original_read(path)
            current, current_binding = validate_execution_reservation(
                recipe, execution_reservation, execution_reservation_sha256, stop_after)
            if current_binding != binding:
                raise ValueError('Execution authorization changed during the operation.')
            return copy.deepcopy(current)

        module.read = execution_read
        module.GPU_UUID = effective['gpu_uuid']
        yield module, {'execution': binding, 'source': source_binding}
    finally:
        if original_read is not None:
            module.read, module.GPU_UUID = original_read, original_uuid
        if sys.modules.get(alias) is module:
            del sys.modules[alias]
        sys.path[:] = previous_path


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--snapshot-manifest', type=Path, required=True)
    parser.add_argument('--snapshot-manifest-sha256', required=True)
    parser.add_argument('--execution-reservation', type=Path, required=True)
    parser.add_argument('--execution-reservation-sha256', required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('native_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    native = args.native_args[1:] if args.native_args[:1] == ['--'] else args.native_args
    if not native or native[0] not in ('prepare', 'fit', 'resume'):
        raise ValueError('Only original native prepare/fit/resume is supported.')
    native_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    native_parser.add_argument('--recipe', type=Path, required=True)
    native_parser.add_argument('--output', type=Path, required=True)
    native_parser.add_argument('--stop-after', type=int, required=True)
    operation, _ = native_parser.parse_known_args(native[1:])
    recipe = read_object(operation.recipe)
    with pinned_cli_override(snapshot=args.snapshot, snapshot_manifest=args.snapshot_manifest,
            snapshot_manifest_sha256=args.snapshot_manifest_sha256, recipe=recipe,
            execution_reservation=args.execution_reservation,
            execution_reservation_sha256=args.execution_reservation_sha256,
            stop_after=operation.stop_after) as (module, binding):
        root = Path(binding['execution']['original_reservation']['path']).parents[1]
        provenance = args.provenance.resolve()
        if not provenance.is_relative_to(root) or not provenance.parent.is_dir():
            raise ValueError('Provenance must use an existing directory under the reserved round root.')
        if provenance.exists():
            raise FileExistsError('Refuse to overwrite operation provenance.')
        if native[0] != 'prepare':
            if os.environ.get('CUDA_VISIBLE_DEVICES') != binding['execution']['gpu_uuid']:
                raise ValueError('Visible CUDA lane differs from the authorized execution GPU.')
            inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                                '--format=csv,noheader,nounits'], text=True)
            pairs = [(int(fields[0].strip()), fields[1].strip())
                     for line in inventory.splitlines() if line.strip()
                     for fields in [line.split(',')]]
            if (binding['execution']['physical_gpu'], binding['execution']['gpu_uuid']) not in pairs:
                raise ValueError('Physical index/UUID does not match the authorized execution lane.')
        started = time.time()
        usage_path = root / 'receipts/round9_usage.jsonl'
        before = usage_path.read_bytes() if usage_path.exists() else b''
        source_hashes = {str(operation.recipe.resolve()): file_sha256(operation.recipe),
                         binding['source']['cli']: binding['source']['cli_sha256'],
                         **{item['path']: item['sha256'] for item in binding['execution'].values()
                            if isinstance(item, dict) and set(item) == {'path', 'sha256'}}}
        start_receipt = provenance.with_suffix('.started.json')
        write_new(start_receipt, {'schema': 'round9_explicit_GPU_execution_start_v1', 'unix': started,
            'PID': os.getpid(), 'native_args': native, 'binding': binding,
            'wrapper': str(Path(__file__).resolve()), 'wrapper_sha256': file_sha256(__file__)})
        error = None
        result = None
        status = 'failed'
        try:
            result = module.main(native)
            if result not in (None, 0):
                raise RuntimeError(f'Native CLI returned {result}.')
            status = 'completed'
            return 0
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            after = usage_path.read_bytes() if usage_path.exists() else b''
            intact = after.startswith(before)
            new_rows = ([json.loads(line) for line in after[len(before):].splitlines() if line.strip()]
                        if intact else [])
            write_new(provenance, {'schema': 'round9_explicit_GPU_execution_provenance_v1',
                'status': status, 'error': error, 'native_returncode': result,
                'started_unix': started, 'finished_unix': time.time(), 'binding': binding,
                'native_args': native, 'recipe_unchanged': file_sha256(operation.recipe) == source_hashes[str(operation.recipe.resolve())],
                'protected_files_unchanged': {path: file_sha256(path) == expected
                                               for path, expected in source_hashes.items()},
                'native_usage_journal': str(usage_path), 'prior_usage_prefix_unchanged': intact,
                'new_native_usage_rows': new_rows, 'new_native_usage_bytes_sha256': hashlib.sha256(after[len(before):]).hexdigest(),
                'all_new_native_usage_UUIDs_match': all(row['gpu_uuid'] == binding['execution']['gpu_uuid'] for row in new_rows),
                'original_CLI_training_resource_monitor_and_usage_code_executed': True,
                'no_native_source_or_recipe_modified': True, 'wrapper_sha256': file_sha256(__file__)})


if __name__ == '__main__':
    raise SystemExit(main())
