"""Typed macro chain rule, native engine capability and stage boundaries."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
import torch
from honf_runtime.joint_native_objective import JointMacroObjective
from honf_runtime.joint_native_protocol import validate_joint_native_stop, STAGES
from honf_forward_core.joint_native.model import JointNativeCore
from honf_forward_core.joint_native.budget import JointBudgetState
from tests.test_content_hypergraph_v2_core import _core
from tests.test_native_budgeted_protocol import _Provider, _engine


def test_typed_macro_hinge_is_after_equal_case_mean():
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.,dtype=torch.double));self.budget=JointBudgetState(.3,.7,dtype=torch.double)
        def joint_begin_update(self,epoch,index):return {'absolute_epoch':epoch}
        def joint_begin_microbatch(self):pass
        def joint_constraint_duals(self):return self.budget.duals
        def joint_microbatch_result(self):
            return {'constraints':{kind:{'residual_numerator':value,'denominator':self.count,'controller_numerator':value.detach()} for kind,value in self.values.items()}}
        def joint_after_update(self,*args):self.observed=args[-1]
    model=Core();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    objective=JointMacroObjective(model,optimizer,3201,0)
    for scale,count in [(1.,1),(-.25,3)]:
        objective.begin_microbatch();model.count=count;model.values={'M':model.weight*scale,'E':model.weight*(-scale)};objective.observe()
    objective.apply()
    ref=model.weight.detach().clone().requires_grad_();m=(ref-ref*.25)/4;e=-m
    penalty=model.budget.duals['M'].penalty_residual(m)+model.budget.duals['E'].penalty_residual(e)
    expected=torch.autograd.grad(penalty,ref)[0]
    assert torch.equal(model.weight.grad,expected)
    assert objective.work['joint_case_requests_M']==4
    objective.after_update();assert model.observed['M']==m.detach()


@pytest.mark.parametrize('mode',['dense','joint','direct'])
def test_joint_capability_keeps_native_loss_on_training_engine(mode):
    provider=_Provider();model=JointNativeCore.from_a2d(_core('A2-D'),routing_mode=mode).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    steps=[]
    losses,work,_,_=_engine()._run_update(model,provider,optimizer,(provider.batch,),phase='joint',arm='A2-D',epoch=3001,update_index=0,
        successful_step_recorder=lambda epoch,index,cases:steps.append((epoch,index,cases)))
    assert steps==[(3001,0,1)]
    assert losses['field']>0 and work['fixture_native_calls']==1
    assert ('joint_work_penalty_M' in losses)==(mode!='dense')
    assert any(p.grad is not None for p in model.environment_reader.parameters())
    reconstructed=JointNativeCore.from_config(model.get_config());reconstructed.load_state_dict(model.state_dict(),strict=True)
    assert reconstructed._joint_epoch==3001
    assert reconstructed._joint_macro_clock==model._joint_macro_clock


def test_round9_stages_reject_silent_horizon_changes():
    assert STAGES['D9-M']==(2500,5000) and STAGES['D9-ME-warm']==(2500,3000)
    validate_joint_native_stop({'arm':'D9-ME-warm'},2750)
    with pytest.raises(ValueError):validate_joint_native_stop({'arm':'D9-ME-warm'},3200)
    with pytest.raises(ValueError):validate_joint_native_stop({'arm':'H9-ME'},3025)


@pytest.mark.parametrize("routing_mode,arm", [("joint","H9-ME"),("direct","S9-ME")])
def test_fixed_train_calibration_does_not_update_weights_rng_or_clock(routing_mode,arm):
    from honf_runtime.joint_native_calibration import calibrate_joint_native_task
    from honf_runtime.unified_training import capture_rng_state
    provider=_Provider();model=JointNativeCore.from_a2d(_core('A2-D'),routing_mode=routing_mode).train()
    with torch.no_grad():model.environment_conditioning.weight.fill_(.02)
    engine=_engine()
    engine.epoch_case_order=lambda provider,epoch: (['fixture-case'],'fixture-dataset')
    engine.epoch_update_slices=lambda count: (slice(0,count),)
    engine.prepare_update_microbatches=lambda *args,**kwargs: [provider.batch]
    task={'model':model,'provider':provider,'engine':engine,'identity':{'recipe':{
        'arm':arm,'dataset':'thermal','source_epoch':3000,'source_checkpoint_sha256':'a'*64,'microbatch_cases':1}}}
    before={name:value.detach().clone() if torch.is_tensor(value) else value for name,value in model.state_dict().items()}
    rng=capture_rng_state()
    receipt=calibrate_joint_native_task(task,max_batches=1)
    assert receipt['optimizer_updates']==0 and len(receipt['batches'])==1
    assert set(receipt['typed_gradient'])=={'M','E'}
    assert all(row['status']=='calibrated' for row in receipt['membership'].values())
    for name,value in before.items():
        actual=model.state_dict()[name]
        assert torch.equal(value,actual) if torch.is_tensor(value) else value==actual
    assert torch.equal(rng['torch_cpu'],capture_rng_state()['torch_cpu'])


def test_standalone_export_reopens_without_parent_receipt_or_data(tmp_path):
    from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
    from tests.test_native_budgeted_postfit import _thermal_stats, _thermal_budgeted_config
    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
    from honf_runtime.joint_native_protocol import (JOINT_NATIVE_RECIPE_SCHEMA,JOINT_NATIVE_FAMILY,
        JOINT_NATIVE_WORKFLOW,JOINT_NATIVE_PROTOCOL,canonical_sha256)
    from honf_runtime.joint_native_runtime import load_joint_native_checkpoint,export_joint_native_checkpoint
    parent=ContentHypergraphCoreV2.from_config({**_thermal_budgeted_config(),'context_depth':2})
    model=ThermalContentHypergraphAdapterV2(_thermal_stats(),core=JointNativeCore.from_a2d(parent))
    model.core.set_training_clock(2600,400)
    recipe=dict(schema=JOINT_NATIVE_RECIPE_SCHEMA,family=JOINT_NATIVE_FAMILY,workflow=JOINT_NATIVE_WORKFLOW,
        protocol=JOINT_NATIVE_PROTOCOL,dataset='thermal',arm='D9-ME-warm',source_checkpoint='/unavailable/parent.pt',
        source_checkpoint_sha256='a'*64,source_epoch=2500,source_recipe_sha256='b'*64,
        source_provider_identity_sha256='c'*64,base_recipe={},microbatch_cases=8,effective_cases=48,
        final_epoch=3000,sampling_arm='A2-D',sampling_version='case_epoch_v1',model_options={},calibration=None,
        resource_reservation='/unavailable/resource.json',schedule_extension='retain_exact_source_terminal_lr_through_e5000',
        execution=dict(checkpoint_every=100,curve_every=10,progress_enabled=True))
    identity=dict(family=JOINT_NATIVE_FAMILY,workflow=JOINT_NATIVE_WORKFLOW,recipe=recipe,
        recipe_sha256=canonical_sha256(recipe),provider_identity=dict(model_config=model.get_config(),adapter_state=model.state_payload()))
    payload=dict(workflow=JOINT_NATIVE_WORKFLOW,epoch=2600,arm='A2-D',experiment_identity=identity,
        model_state_dict=model.state_dict(),optimizer_state_by_name={'fixture':'deliberately excluded'})
    source=tmp_path/'training.pt';target=tmp_path/'inference.pt';torch.save(payload,source)
    export_joint_native_checkpoint(source,target);source.unlink()
    loaded,adapter,exported=load_joint_native_checkpoint(target,device='cpu')
    assert adapter is loaded and loaded.core._joint_epoch==2600 and loaded.core._joint_macro_clock==400
    assert 'optimizer_state_by_name' not in exported
    for name,value in model.state_dict().items():
        actual=loaded.state_dict()[name]
        assert torch.equal(actual,value) if torch.is_tensor(value) else actual==value
