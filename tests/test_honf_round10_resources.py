"""CPU checks for shared campaign admission and protected GPU ownership."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

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
