"""Standalone continuous native state and strict common optimizer boundaries."""
from pathlib import Path
import copy
import pytest
import torch
from honf_runtime.continuous_joint_protocol import PARENT_SHA,canonical_sha256
from tests.test_native_budgeted_postfit import _thermal_stats,_thermal_budgeted_config,_thermal_native_inputs


@pytest.mark.parametrize('mode,arm',[('hypergraph','C10-H'),('direct','C10-S')])
def test_continuous_export_predictions_need_no_parent_or_dataset(tmp_path,mode,arm):
    from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
    from honf_forward_core.joint_native.model import JointNativeCore
    from honf_forward_core.continuous_joint.model import ContinuousJointCore
    from honf_runtime.continuous_joint_runtime import export_continuous_joint_checkpoint,load_continuous_joint_checkpoint
    parent=JointNativeCore.from_a2d(ContentHypergraphCoreV2.from_config({**_thermal_budgeted_config(),'context_depth':2}))
    core=ContinuousJointCore.from_parent(parent,routing_mode=mode,query_tile_size=2)
    model=ThermalContentHypergraphAdapterV2(_thermal_stats(),core=core).eval();core.set_training_clock(3100,400)
    recipe=dict(schema='round10_continuous_joint_recipe_v1',family=core.schema,workflow='continuous_joint_train_v1',
        protocol='round10_continuous_joint_v1',dataset='thermal',arm=arm,source_checkpoint='/unavailable/parent.pt',
        source_checkpoint_sha256=PARENT_SHA['thermal'],source_epoch=3000,source_recipe_sha256='b'*64,
        source_provider_identity_sha256='c'*64,base_recipe={},microbatch_cases=8,effective_cases=48,
        final_epoch=5000,sampling_arm='A2-D',sampling_version='case_epoch_v1',model_options={},calibration=None,
        resource_reservation='/unavailable/resource.json',schedule_extension='retain_exact_source_terminal_lr_through_e5000',
        execution=dict(checkpoint_every=100,curve_every=10,progress_enabled=True))
    identity=dict(family=core.schema,workflow='continuous_joint_train_v1',recipe=recipe,recipe_sha256=canonical_sha256(recipe),
        provider_identity=dict(model_config=model.get_config(),adapter_state=model.state_payload()))
    source=tmp_path/'train.pt';target=tmp_path/'inference.pt'
    torch.save(dict(workflow='continuous_joint_train_v1',epoch=3100,arm='A2-D',experiment_identity=identity,
        model_state_dict=model.state_dict(),optimizer_state_by_name={'excluded':True}),source)
    export_continuous_joint_checkpoint(source,target);source.unlink()
    loaded,adapter,payload=load_continuous_joint_checkpoint(target)
    scene,query=_thermal_native_inputs();heat=torch.tensor([[1.,-.5,2.]])
    with torch.no_grad():
        expected=core.decode(core.prepare(scene),query,heat)
        actual=loaded.core.decode(loaded.core.prepare(scene),query,heat)
    torch.testing.assert_close(expected.nonlinear_field,actual.nonlinear_field,rtol=0,atol=0)
    torch.testing.assert_close(expected.thermal_temperature,actual.thermal_temperature,rtol=0,atol=0)
    assert adapter is loaded and 'optimizer_state_by_name' not in payload
    assert loaded.normalizer_key==model.normalizer_key
    torch.testing.assert_close(loaded.field_std,model.field_std,rtol=0,atol=0)
