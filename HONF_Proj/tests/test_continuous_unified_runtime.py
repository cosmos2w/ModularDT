"""Explicit summary/Adam drop inventory and ancestor-free unified reconstruction."""
import copy
import pytest
import torch

from honf_runtime.continuous_joint_child import filtered_continuous_optimizer_seed, _allowed_dropped_names


def test_only_declared_summary_state_can_be_dropped_and_source_is_unchanged():
    source={'groups':[{'parameter_names':['core.scene_summary_net.0.weight','core.module_encoder.0.weight']}],
            'state_by_name':{'core.scene_summary_net.0.weight':{'step':torch.tensor(3.)},
                             'core.module_encoder.0.weight':{'step':torch.tensor(7.),'exp_avg':torch.tensor([.2])}}}
    before=copy.deepcopy(source)
    amendment={'dropped_optimizer_parameter_names':['core.scene_summary_net.0.weight']}
    filtered=filtered_continuous_optimizer_seed(source,amendment)
    assert filtered['groups'][0]['parameter_names']==['core.module_encoder.0.weight']
    assert set(filtered['state_by_name'])=={'core.module_encoder.0.weight'}
    assert torch.equal(filtered['state_by_name']['core.module_encoder.0.weight']['step'],torch.tensor(7.))
    assert source['groups']==before['groups']
    for name in source['state_by_name']:
        for key in source['state_by_name'][name]:
            assert torch.equal(source['state_by_name'][name][key],before['state_by_name'][name][key])
    with pytest.raises(ValueError,match='nonsummary'):
        filtered_continuous_optimizer_seed(source,{'dropped_optimizer_parameter_names':['core.module_encoder.0.weight']})
    with pytest.raises(ValueError,match='undeclared'):
        _allowed_dropped_names(['core.module_encoder.0.weight'],[],{'arm':'U10-H'})
    with pytest.raises(ValueError,match='undeclared'):
        _allowed_dropped_names(['core.scene_summary_net.0.weight'],[],{'arm':'C10-H'})


def test_unified_standalone_export_loads_without_parent_data_or_receipt(tmp_path):
    from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
    from tests.test_native_budgeted_postfit import _thermal_stats,_thermal_budgeted_config
    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
    from honf_forward_core.joint_native.model import JointNativeCore
    from honf_forward_core.continuous_joint.unified import UnifiedJointCore
    from honf_runtime.continuous_joint_protocol import PARENT_SHA,canonical_sha256
    from honf_runtime.continuous_joint_runtime import load_continuous_joint_checkpoint,export_continuous_joint_checkpoint
    parent=JointNativeCore.from_a2d(ContentHypergraphCoreV2.from_config({**_thermal_budgeted_config(),'context_depth':2}))
    core=UnifiedJointCore.from_parent(parent,input_dataset='thermal',routing_mode='dense')
    model=ThermalContentHypergraphAdapterV2(_thermal_stats(),core=core)
    core.set_training_clock(3100,400)
    recipe=dict(schema='round10_unified_joint_recipe_v1',family=core.schema,workflow='continuous_joint_train_v1',
        protocol='round10_unified_joint_v1',dataset='thermal',arm='U10-D',source_checkpoint='/unavailable/parent.pt',
        source_checkpoint_sha256=PARENT_SHA['thermal'],source_epoch=3000,source_recipe_sha256='b'*64,
        source_provider_identity_sha256='c'*64,base_recipe={},microbatch_cases=8,effective_cases=48,
        final_epoch=5000,sampling_arm='A2-D',sampling_version='case_epoch_v1',model_options={},calibration=None,
        resource_reservation='/unavailable/resource.json',schedule_extension='retain_exact_source_terminal_lr_through_e5000',
        execution=dict(checkpoint_every=100,curve_every=10,progress_enabled=True),
        input_capability=core.input_capability,input_feature_contract=core.local_input_contract,unified_admission=None)
    identity=dict(family=core.schema,workflow='continuous_joint_train_v1',recipe=recipe,
        recipe_sha256=canonical_sha256(recipe),provider_identity=dict(model_config=model.get_config(),adapter_state=model.state_payload()))
    source=tmp_path/'training_fixture.pt';target=tmp_path/'standalone_fixture.pt'
    torch.save(dict(workflow='continuous_joint_train_v1',epoch=3100,arm='A2-D',experiment_identity=identity,
                    model_state_dict=model.state_dict(),optimizer_state_by_name={'fixture':'excluded'}),source)
    export_continuous_joint_checkpoint(source,target)
    loaded,adapter,payload=load_continuous_joint_checkpoint(target)
    assert loaded is adapter
    assert 'optimizer_state_by_name' not in payload
    assert loaded.core.input_capability=='local_records_context_v1'
    assert not any('scene_summary' in name for name in loaded.state_dict())
    for name,value in model.state_dict().items():
        actual=loaded.state_dict()[name]
        assert torch.equal(actual,value) if torch.is_tensor(value) else actual==value
