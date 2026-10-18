"""CPU-only resource authorization and unchanged source-inventory boundaries."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'tools/honf_joint_device_continue.py'
SPEC = importlib.util.spec_from_file_location('joint_device_wrapper_test', PATH)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return {'path': str(path.resolve()), 'sha256': wrapper.file_sha256(path)}


@pytest.fixture
def files(tmp_path):
    receipts = tmp_path / 'round/receipts'
    receipts.mkdir(parents=True)
    original = receipts / 'original.json'
    base = {'schema': 'round9_resource_reservation_v1', 'gpu_uuid': 'GPU-two', 'physical_gpu': 2,
            'start_unix': 1000.25, 'elapsed_limit_seconds': 86400, 'scientific_updates': 81000,
            'engineering_updates_limit': 600, 'disk_min_free_bytes': 30000,
            'mount': {'target': '/data2', 'source': '/dev/example', 'fstype': 'ext4'}}
    original_binding = write(original, base)
    amendment = receipts / 'amendment.json'
    amendment_binding = write(amendment, {'terminal_epoch': 3500, 'scientific_updates': 27000,
        'training_after_3500_authorized': False, 'original_reservation_path': str(original),
        'original_reservation_sha256': original_binding['sha256']})
    authorization = receipts / 'authorization.json'
    authorization_binding = write(authorization, {'schema': 'round9_user_GPU_lane_authorization_v1',
        'original_reservation': original_binding, 'terminal_epoch': 3500,
        'training_after_3500_authorized': False, 'matched_decision_stop_amendment': amendment_binding,
        'allowed_devices': [{'physical_gpu': 1, 'gpu_uuid': 'GPU-one'},
                            {'physical_gpu': 2, 'gpu_uuid': 'GPU-two'}]})
    effective = copy.deepcopy(base)
    effective.update(gpu_uuid='GPU-one', physical_gpu=1, execution_override={
        'schema': 'round9_execution_reservation_v1', 'original_reservation': original_binding,
        'authorization': authorization_binding, 'terminal_epoch': 3500})
    override = receipts / 'execution.json'
    override_binding = write(override, effective)
    snapshot = tmp_path / 'snapshot'
    cli = snapshot / 'HONF_Proj/tools/joint_native_train.py'
    cli.parent.mkdir(parents=True)
    cli.write_text("from pathlib import Path\nimport json\n"
                   "PROJECT=Path(__file__).resolve().parents[1]\nGPU_UUID='GPU-two'\n"
                   "def read(path):return json.loads(Path(path).read_text())\n")
    manifest = receipts / 'snapshot.json'
    manifest_binding = write(manifest, {'root': str(snapshot), 'files': [
        {'path': 'HONF_Proj/tools/joint_native_train.py', 'sha256': wrapper.file_sha256(cli)}]})
    return {'recipe': {'resource_reservation': str(original)}, 'original': original,
            'override': override, 'effective': effective, 'override_sha': override_binding['sha256'],
            'authorization': authorization, 'amendment': amendment, 'snapshot': snapshot,
            'manifest': manifest, 'manifest_sha': manifest_binding['sha256'], 'cli': cli}


def validate(files, stop=3500):
    return wrapper.validate_execution_reservation(files['recipe'], files['override'],
                                                  files['override_sha'], stop)


def context(files):
    return wrapper.pinned_cli_override(snapshot=files['snapshot'], snapshot_manifest=files['manifest'],
        snapshot_manifest_sha256=files['manifest_sha'], recipe=files['recipe'],
        execution_reservation=files['override'], execution_reservation_sha256=files['override_sha'],
        stop_after=3500)


def test_gpu1_override_is_execution_metadata_only(files):
    before = files['original'].read_bytes()
    effective, binding = validate(files)
    assert effective['gpu_uuid'] == 'GPU-one' and effective['physical_gpu'] == 1
    assert binding['authorization']['sha256'] == wrapper.file_sha256(files['authorization'])
    assert binding['terminal_epoch'] == 3500 and not binding['training_after_3500_authorized']
    assert files['original'].read_bytes() == before
    assert files['recipe'] == {'resource_reservation': str(files['original'])}


@pytest.mark.parametrize('key,value', [
    ('engineering_updates_limit', 601), ('scientific_updates', 27000),
    ('start_unix', 2000.25), ('elapsed_limit_seconds', 86401),
    ('disk_min_free_bytes', 1), ('mount', {'target': '/other'}),
    ('new_unapproved_field', True), ('engineering_updates_limit', 600.0),
])
def test_rejects_any_protected_resource_change(files, key, value):
    changed = copy.deepcopy(files['effective'])
    changed[key] = value
    files['override_sha'] = write(files['override'], changed)['sha256']
    with pytest.raises(ValueError, match='original resource field'):
        validate(files)


@pytest.mark.parametrize('key,value', [('physical_gpu', 3), ('gpu_uuid', 'GPU-unapproved'),
                                      ('physical_gpu', True)])
def test_rejects_unapproved_physical_uuid_pair(files, key, value):
    changed = copy.deepcopy(files['effective'])
    changed[key] = value
    files['override_sha'] = write(files['override'], changed)['sha256']
    with pytest.raises(ValueError, match='not user-authorized'):
        validate(files)


@pytest.mark.parametrize('target', ['original', 'authorization', 'amendment', 'override'])
def test_rejects_changed_bound_file_bytes(files, target):
    files[target].write_text(files[target].read_text() + ' ')
    with pytest.raises(ValueError, match='SHA'):
        validate(files)


def test_rejects_after3500_before_any_native_import(files):
    with pytest.raises(ValueError, match='e3500 decision stop'):
        validate(files, 3501)
    assert '_honf_joint_device_pinned_cli' not in sys.modules


def test_read_override_is_exact_and_all_globals_restore_on_exception(files):
    original_bytes = files['cli'].read_bytes()
    previous_path = list(sys.path)
    ordinary = files['original'].parent / 'ordinary.json'
    write(ordinary, {'untouched': True})
    with pytest.raises(RuntimeError, match='injected'):
        with context(files) as (native, binding):
            assert native.GPU_UUID == 'GPU-one'
            assert native.read(files['original'])['gpu_uuid'] == 'GPU-one'
            assert native.read(ordinary) == {'untouched': True}
            returned = native.read(files['original'])
            returned['scientific_updates'] = -1
            assert native.read(files['original'])['scientific_updates'] == 81000
            assert binding['source']['cli_sha256'] == wrapper.file_sha256(files['cli'])
            raise RuntimeError('injected')
    assert native.GPU_UUID == 'GPU-two'
    assert native.read(files['original'])['gpu_uuid'] == 'GPU-two'
    assert '_honf_joint_device_pinned_cli' not in sys.modules
    assert sys.path == previous_path and files['cli'].read_bytes() == original_bytes


def test_rejects_pinned_cli_source_change(files):
    files['cli'].write_text(files['cli'].read_text() + '\n# changed\n')
    with pytest.raises(ValueError, match='Pinned source bytes'):
        with context(files):
            pass


def test_real_source_inventory_uses_original_path_not_module_alias(files):
    # Exercise the maintained inventory scanner, not an imitation of its rule.
    previous = set(sys.modules)
    try:
        with context(files) as (native, _):
            from honf_runtime.native_budgeted_factory import _loaded_source_binding
            inventory = _loaded_source_binding(native.PROJECT)
            assert inventory['files'] == {'tools/joint_native_train.py': wrapper.file_sha256(files['cli'])}
            assert all('honf_joint_device_continue' not in key for key in inventory['files'])
    finally:
        for name in set(sys.modules) - previous:
            if name == 'honf_runtime' or name.startswith('honf_runtime.'):
                del sys.modules[name]
