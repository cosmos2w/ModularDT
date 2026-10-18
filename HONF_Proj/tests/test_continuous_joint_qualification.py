"""Qualification scheduling/resource boundaries without scientific optimizer steps."""
from types import SimpleNamespace
import pytest
import torch
from honf_runtime import continuous_joint_qualification as qualification
from honf_runtime.unified_training import TrainingEngine


def task_fixture(monkeypatch):
    monkeypatch.setattr(qualification,'_allowed_dropped_names',lambda *args:[])
    monkeypatch.setattr(qualification,'filtered_continuous_optimizer_seed',lambda *args:{})
    monkeypatch.setattr(qualification,'verify_continuous_joint_optimizer',lambda *args:None)
    monkeypatch.setattr('honf_runtime.unified_training.restore_rng_state',lambda state:None)
    for name in ('synchronize','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ('max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda:0)
    parameter=SimpleNamespace(device=SimpleNamespace(type='cuda'))
    model=SimpleNamespace(parameters=lambda:iter([parameter]),named_parameters=lambda:[],train=lambda:None)
    optimizer=SimpleNamespace(param_groups=[{'group_name':'physical','lr':-1.}])
    # Use the maintained native engine schedule setter, with a nonconstant
    # absolute-age schedule so an omitted call cannot accidentally pass.
    specs=[SimpleNamespace(name='physical',schedule=SimpleNamespace(value=lambda age:age/100000.))]
    calls=[]
    def update(*args,epoch,update_index,successful_step_recorder,**kwargs):
        assert optimizer.param_groups[0]['lr']==epoch/100000.
        calls.append((epoch,update_index));successful_step_recorder(epoch,update_index,1)
        return {},{},{},str(calls[-1])
    engine=SimpleNamespace(_make_optimizer=lambda *args:(optimizer,specs),
        _set_schedules=TrainingEngine._set_schedules,epoch_case_order=lambda *args:([0,1],'named_fixture'),
        epoch_update_slices=lambda count:[slice(0,1),slice(1,2)],
        prepare_update_microbatches=lambda *args,**kwargs:[SimpleNamespace(case_keys=['named'])],_run_update=update)
    return dict(model=model,provider=SimpleNamespace(),engine=engine,identity={'recipe':dict(dataset='thermal',arm='U10-D',source_checkpoint_sha256='a'*64)},
        source_payload={'optimizer_state_by_name':{'groups':[]},'rng_state':{}}),calls


def test_qualification_uses_native_absolute_schedule_each_epoch(monkeypatch):
    task,calls=task_fixture(monkeypatch)
    result=qualification.qualify_complete_native_epochs(task,epochs=2)
    assert calls==[(3001,0),(3001,1),(3002,0),(3002,1)]
    assert result['completed_native_epochs']==2 and result['discarded_optimizer_updates']==4


def test_qualification_honors_resource_stop_before_next_macro(monkeypatch):
    task,calls=task_fixture(monkeypatch)
    def boundary():
        if calls:raise RuntimeError('fixture Monitor clean stop')
    with pytest.raises(RuntimeError,match='Monitor clean stop'):
        qualification.qualify_complete_native_epochs(task,epochs=2,boundary=boundary)
    assert calls==[(3001,0)]
