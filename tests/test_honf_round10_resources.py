"""CPU checks for shared campaign admission and protected GPU ownership."""
from datetime import datetime, timezone
import errno
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC=importlib.util.spec_from_file_location('round10_resources_test',Path(__file__).resolve().parents[1]/'tools/honf_round10_resources.py')
resources=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resources)


@pytest.fixture
def campaign(tmp_path,monkeypatch):
    (tmp_path/'receipts').mkdir()
    master={'schema':'honf_round10_master_reservation_v1','root':str(tmp_path),
        'created_utc':datetime.now(timezone.utc).isoformat(),'authorized_gpus':{'1':'GPU-one','2':'GPU-two'},
        'limits':{'main_scientific_updates':10,'discarded_engineering_updates':4,
                  'gpu_associated_seconds':1000,'gpu_associated_seconds_reserved_for_evaluation':100,
                  'wall_seconds':10000}}
    (tmp_path/'receipts/master_reservation_v1.json').write_text(json.dumps(master))
    value=resources.Campaign(tmp_path)
    monkeypatch.setattr(value,'check_storage',lambda:{})
    monkeypatch.setattr(resources, 'device_memory', lambda uuid: {
        'gpu_free_bytes': 40 * 1024**3, 'gpu_total_bytes': 48 * 1024**3,
        'gpu_floor_bytes': 10 * 1024**3, 'preventive_margin_bytes': 6 * 1024**3})
    return value


def reserve(campaign,name,**kwargs):
    return campaign.reserve(name,track='A',kind='main',updates=kwargs.get('updates',6),
                            seconds=kwargs.get('seconds',400),physical_gpu=kwargs.get('gpu',1))


def test_active_devices_share_one_update_and_time_cap(campaign):
    reserve(campaign,'gpu1')
    with pytest.raises(RuntimeError,match='update reservation'):
        reserve(campaign,'gpu2',gpu=2)
    with pytest.raises(RuntimeError,match='process-time'):
        reserve(campaign,'time_overflow',updates=1,seconds=501,gpu=2)
    campaign.finish('gpu1',updates=4,status='completed')
    reserve(campaign,'gpu2_after_release',gpu=2)
    assert resources.read(campaign.ledger_path)['operations']['gpu1']['actual_updates']==4


def test_operation_ids_and_unauthorized_device_cannot_be_reused(campaign):
    reserve(campaign,'one')
    with pytest.raises(ValueError,match='already exists'):
        reserve(campaign,'one',updates=1)
    with pytest.raises(ValueError,match='authorization'):
        reserve(campaign,'gpu0',updates=1,gpu=0)


def test_evaluation_reserve_is_not_charged_twice_against_training(campaign,monkeypatch):
    clock=[1000.]
    monkeypatch.setattr(resources.time,'time',lambda:clock[0])
    campaign.reserve('evaluation',track='W6',kind='evaluation',updates=0,seconds=100,physical_gpu=1)
    clock[0]+=80
    campaign.finish('evaluation',updates=0,status='completed')
    with pytest.raises(RuntimeError,match='process-time'):
        campaign.reserve('evaluation_allowance_overflow',track='W6',kind='evaluation',updates=0,seconds=21,physical_gpu=1)
    # 900 training seconds plus80 evaluation seconds obey both900/1000 caps.
    reserve(campaign,'training',updates=6,seconds=900,gpu=2)
    with pytest.raises(RuntimeError,match='process-time'):
        campaign.reserve('evaluation_overflow',track='W6',kind='evaluation',updates=0,seconds=21,physical_gpu=1)


def test_live_elapsed_monitoring_uses_separate_evaluation_allowance(campaign,monkeypatch,tmp_path):
    monkeypatch.setattr(resources.time,'time',lambda:1000.)
    campaign.reserve('evaluation',track='W6',kind='evaluation',updates=0,seconds=100,physical_gpu=1)
    ledger=resources.read(campaign.ledger_path)
    assert campaign.time_accounting(ledger,now=1080.,running_as_reserved=False)=={
        'total_seconds':80.,'non_evaluation_seconds':0.,'evaluation_seconds':80.}
    monkeypatch.setattr(resources.time,'time',lambda:1101.)
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *args,**kwargs:'GPU-one, 40000, 49140\n')
    monkeypatch.setattr(resources,'device_processes',lambda uuid:[])
    output=tmp_path/'monitored';output.mkdir()
    sample=resources.Monitor(campaign,'evaluation',output,'GPU-one').sample()
    assert sample['actual_time_accounting']['evaluation_seconds']==101.
    assert 'evaluation_process_time_budget' in sample['stop_reasons']
    assert (output/'CLEAN_STOP_REQUEST.json').exists()


@pytest.mark.parametrize('exhausted_kind,current_kind,exhausted_seconds,forbidden_reason',[
    ('main','evaluation',900.,'non_evaluation_process_time_budget'),
    ('evaluation','main',100.,'evaluation_process_time_budget'),
])
def test_completed_allowance_does_not_block_other_time_category(campaign,monkeypatch,tmp_path,
        exhausted_kind,current_kind,exhausted_seconds,forbidden_reason):
    clock=[1000.]
    monkeypatch.setattr(resources.time,'time',lambda:clock[0])
    campaign.reserve('old',track='A',kind=exhausted_kind,updates=0,seconds=exhausted_seconds,physical_gpu=1)
    clock[0]+=exhausted_seconds
    campaign.finish('old',updates=0,status='completed')
    remaining=100. if current_kind=='evaluation' else 500.
    campaign.reserve('current',track='W6',kind=current_kind,updates=0,seconds=remaining,physical_gpu=1)
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *args,**kwargs:'GPU-one, 40000, 49140\n')
    monkeypatch.setattr(resources,'device_processes',lambda uuid:[])
    output=tmp_path/'current';output.mkdir()
    sample=resources.Monitor(campaign,'current',output,'GPU-one').sample()
    assert forbidden_reason not in sample['stop_reasons']


def test_competing_physical_process_prevents_device_admission(campaign,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-one')
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *args,**kwargs:'1, GPU-one\n2, GPU-two\n')
    monkeypatch.setattr(resources,'device_processes',lambda uuid:[{'pid':987654,'gpu_uuid':uuid}])
    with pytest.raises(RuntimeError,match='competing compute process'):
        with campaign.device(1):
            pytest.fail('A competing process must reject the lease before allocation.')


def test_device_lock_is_exclusive(campaign,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-one')
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *args,**kwargs:'1, GPU-one\n')
    monkeypatch.setattr(resources,'device_processes',lambda uuid:[])
    with campaign.device(1):
        with pytest.raises(BlockingIOError):
            with campaign.device(1):
                pytest.fail('A second lease must not acquire the physical device.')


def shared_device_inventory(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-one')
    monkeypatch.setattr(resources.subprocess, 'check_output', lambda *args, **kwargs: '1, GPU-one\n')
    monkeypatch.setattr(resources, 'device_processes', lambda uuid: [{'pid': 987654, 'gpu_uuid': uuid}])


def test_shared_training_keeps_external_processes_as_telemetry(campaign, monkeypatch, tmp_path):
    shared_device_inventory(monkeypatch)
    with campaign.device(1, access='shared_training'):
        reserve(campaign, 'shared')
        with pytest.raises(BlockingIOError):
            with campaign.device(1, access='shared_training'):
                pytest.fail('One campaign job per lane still applies.')
    assert campaign.device_access_mode == 'exclusive'
    row = resources.read(campaign.ledger_path)['operations']['shared']
    assert row['device_lease_mode'] == 'shared_training'
    output = tmp_path / 'shared'; output.mkdir()
    sample = resources.Monitor(campaign, 'shared', output, 'GPU-one').sample()
    assert sample['unexpected_compute_processes'][0]['pid'] == 987654
    assert sample['device_lease_mode'] == 'shared_training'
    assert sample['stop_reasons'] == []
    assert not (output / 'CLEAN_STOP_REQUEST.json').exists()


def test_shared_training_still_rejects_memory_pressure_at_admission(campaign, monkeypatch):
    shared_device_inventory(monkeypatch)
    monkeypatch.setattr(resources, 'device_memory', lambda uuid: {
        'gpu_free_bytes': 16, 'gpu_floor_bytes': 10, 'preventive_margin_bytes': 6})
    with pytest.raises(RuntimeError, match='no capacity'):
        with campaign.device(1, access='shared_training'):
            pytest.fail('No allocator capacity remains.')
    assert not campaign.ledger_path.exists()


@pytest.mark.parametrize('kind', ['engineering', 'evaluation'])
def test_shared_lease_cannot_admit_measurements(campaign, monkeypatch, kind):
    shared_device_inventory(monkeypatch)
    with campaign.device(1, access='shared_training'):
        with pytest.raises(ValueError, match='main training only'):
            campaign.reserve('measurement', track='W6', kind=kind, updates=0, seconds=300, physical_gpu=1)
    assert not campaign.ledger_path.exists()


@pytest.mark.parametrize('mode,pressure,expected', [
    ('shared_training', True, 'gpu_floor_plus_margin'),
    ('exclusive', False, 'unexpected_gpu_competitor'),
])
def test_monitor_keeps_pressure_and_exclusive_measurement_stops(campaign, monkeypatch, tmp_path,
                                                              mode, pressure, expected):
    shared_device_inventory(monkeypatch)
    campaign.device_access_mode = mode
    reserve(campaign, 'monitored')
    if pressure:
        monkeypatch.setattr(resources, 'device_memory', lambda uuid: {
            'gpu_free_bytes': 15, 'gpu_floor_bytes': 10, 'preventive_margin_bytes': 6})
    output = tmp_path / 'monitored'; output.mkdir()
    sample = resources.Monitor(campaign, 'monitored', output, 'GPU-one').sample()
    assert expected in sample['stop_reasons']
    assert (output / 'CLEAN_STOP_REQUEST.json').exists()
    if mode == 'shared_training':
        assert 'unexpected_gpu_competitor' not in sample['stop_reasons']


def test_storage_scan_counts_regular_paths_and_tmp_components(tmp_path,monkeypatch):
    (tmp_path/'visible').write_bytes(b'1234567')
    (tmp_path/'.hidden').write_bytes(b'abc')
    (tmp_path/'tmp').write_bytes(b'12345')
    (tmp_path/'nested'/'tmp').mkdir(parents=True)
    (tmp_path/'nested'/'tmp'/'payload').write_bytes(b'12345678901')
    (tmp_path/'nested2'/'tmp'/'data').parent.mkdir(parents=True)
    (tmp_path/'nested2'/'tmp'/'data').write_bytes(b'1234567890123')
    os.link(tmp_path/'visible',tmp_path/'visible-hardlink')
    (tmp_path/'outside').write_bytes(b'x'*101)
    (tmp_path/'file-link').symlink_to(tmp_path/'outside')
    (tmp_path/'directory-link').symlink_to(tmp_path/'nested',target_is_directory=True)

    real_scandir=os.scandir; entry_paths=[]; stat_calls=[]
    class WrappedEntry:
        def __init__(self,entry):
            self.entry=entry;self.name=entry.name;self.path=entry.path
        def stat(self,*,follow_symlinks=True):
            stat_calls.append((self.path,follow_symlinks))
            return self.entry.stat(follow_symlinks=follow_symlinks)
    class WrappedScan:
        def __init__(self,path):self.context=real_scandir(path)
        def __enter__(self):
            entries=self.context.__enter__()
            def wrapped_entries():
                for entry in entries:
                    entry_paths.append(entry.path)
                    yield WrappedEntry(entry)
            return wrapped_entries()
        def __exit__(self,*args):return self.context.__exit__(*args)
    monkeypatch.setattr(resources.os,'scandir',WrappedScan)

    assert resources._scan_storage(tmp_path)==(118,29)
    assert stat_calls==[(path,False) for path in entry_paths]


def test_scan_storage_restarts_complete_walk_on_enoent(monkeypatch):
    calls=[]; sleeps=[]
    def scan(root):
        calls.append(root)
        if len(calls)<3:
            raise FileNotFoundError(errno.ENOENT,'entry disappeared')
        return 17,29
    monkeypatch.setattr(resources,'_scan_storage_once',scan)
    monkeypatch.setattr(resources.time,'sleep',sleeps.append)

    assert resources._scan_storage('/scan/root')==(17,29)
    assert calls==['/scan/root']*3
    assert sleeps==[0.05,0.05]


def test_scan_storage_propagates_retry_exhaustion_and_unexpected_errors(monkeypatch):
    calls=[]
    def vanished(root):
        calls.append(root)
        raise FileNotFoundError(errno.ENOENT,'entry disappeared')
    monkeypatch.setattr(resources,'_scan_storage_once',vanished)
    monkeypatch.setattr(resources.time,'sleep',lambda _:None)
    with pytest.raises(FileNotFoundError):
        resources._scan_storage('/scan/root')
    assert len(calls)==3

    calls.clear()
    def denied(root):
        calls.append(root)
        raise PermissionError(errno.EACCES,'scan denied')
    monkeypatch.setattr(resources,'_scan_storage_once',denied)
    with pytest.raises(PermissionError):
        resources._scan_storage('/scan/root')
    assert len(calls)==1


def test_campaign_storage_preserves_mount_headroom_and_payload_caps(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir()
    (root/'payload').write_bytes(b'1234')
    campaign=resources.Campaign.__new__(resources.Campaign)
    campaign.root=root
    campaign.master={'mount':'/data2 /dev/mock ext4'}
    campaign.limits={'minimum_free_data2_bytes':100,'retained_payload_bytes':4,
                     'temporary_payload_bytes':0,'peak_new_payload_bytes':4}
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *args,**kwargs:'/data2 /dev/mock ext4\n')
    monkeypatch.setattr(resources.shutil,'disk_usage',lambda path:SimpleNamespace(free=100+128*1024**2))

    assert campaign.check_storage()=={'retained_bytes':4,'temporary_bytes':0,'free_bytes':100+128*1024**2}
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *args,**kwargs:'/data2 /dev/changed ext4\n')
    with pytest.raises(RuntimeError,match='mount differs'):
        campaign.check_storage()
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *args,**kwargs:'/data2 /dev/mock ext4\n')
    monkeypatch.setattr(resources.shutil,'disk_usage',lambda path:SimpleNamespace(free=100+128*1024**2-1))
    with pytest.raises(OSError,match='headroom exhausted'):
        campaign.check_storage()
    monkeypatch.setattr(resources.shutil,'disk_usage',lambda path:SimpleNamespace(free=100+128*1024**2))
    campaign.limits['retained_payload_bytes']=3
    with pytest.raises(OSError,match='payload budget'):
        campaign.check_storage()
