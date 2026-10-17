"""Scientific joins, input-only panels, physical units and explicit E controls."""
from types import SimpleNamespace
import json
import copy

import numpy as np
import pytest
import torch

from honf_runtime.joint_native_evaluation import (canonical_sha256,freeze_native_panel,validate_native_panel,
    metric,select_input_representatives,summarize,write_arrays,compare_physical_boards,typed_work_summary,THERMAL_FIELDS,BOARD_SCHEMA)
from honf_runtime.joint_native_physics import (environment_read_intervention,mixed_finite_difference,_response_structure,
    _continuous_query,_continuous_values,_fixed_read_fingerprint,_effect_magnitudes,_effect_validity,
    _kernel_effect_magnitudes,evaluate_seams)
from honf_forward_core.joint_native.model import JointNativeCore
from tests.test_content_hypergraph_v2_core import _core,_scene,_query
from tests.test_budgeted_native_core import _double


def test_panel_selection_uses_input_strata_and_rejects_membership_or_normalizer_drift(tmp_path):
    cases=[{'case_id':str(i),'structure':{'module_present':np.ones(m)},'target':object()}
           for i,m in enumerate((1,2,3,7,10,12))]
    provider=SimpleNamespace(training_case_ids=tuple(str(i) for i in range(4)),validation_case_ids=('4','5'),
        training_cases=cases[:4],validation_cases=cases[4:],identity_payload=lambda:{'normalization_stats_sha256':'a'*64})
    checkpoint=tmp_path/'source.pt';checkpoint.write_bytes(b'input-only constructor identity')
    panel=freeze_native_panel(provider,'thermal',source_checkpoint=checkpoint)
    assert panel['representatives']['train']==['0','1','2','3']
    assert not panel['selection_reads_prediction_or_target_values']
    validate_native_panel(panel,provider,'thermal')
    provider.identity_payload=lambda:{'normalization_stats_sha256':'b'*64}
    with pytest.raises(ValueError,match='normalizer'):validate_native_panel(panel,provider,'thermal')
    assert select_input_representatives({'M1b':1,'M1a':1,'M2':2,'M12':12})[:2]==['M1a','M2']


def test_physical_metrics_apply_masks_without_hiding_zero_denominators():
    result=metric(np.array([1.,1000.,3.]),np.array([0.,np.nan,0.]),np.array([True,False,True]))
    assert result['rmse']==pytest.approx(np.sqrt(5))
    assert result['relative_l2'] is None and result['valid_entries']==2
    tails=summarize([1.,2.,10.],['a','b','c'])
    assert tails['worst_case_id']=='c' and tails['p90']==pytest.approx(8.4)
    with pytest.raises(ValueError,match='axes'):metric(np.zeros(3),np.zeros(4),True)


def test_mixed_physical_response_requires_complete_four_state_quartet():
    values=np.asarray([[1.],[2.],[3.],[7.]])
    labels=['baseline','a','b','ab']
    assert mixed_finite_difference(values,labels,labels)==3
    with pytest.raises(ValueError,match='four compatible'):mixed_finite_difference(values,labels,['baseline','a','b','missing'])
    # Field-linear heating has zero mixed response, while an endpoint maximum
    # can change its maximizing location and is not a heat-linearity proof.
    affine=np.asarray([[1.,2.],[2.,2.],[1.,3.],[2.,3.]])
    np.testing.assert_array_equal(mixed_finite_difference(affine,labels,labels),0)


def test_saved_response_input_builder_preserves_joins_without_reading_targets():
    config={'runtime':{'nu':.1},'thermal':{'t_in':0.,'t_wall':0.,'solid_alpha':.1,'fluid_alpha':.2,'solid_k':1.,'fluid_k':1.},
            'domain':{'module_radius':.5,'lx':12.,'ly':6.},'flow':{'re':50.,'u_in':1.}}
    atlas={'family_metadata_json':np.asarray(json.dumps({'source_dataset_split':'train','case_config':config})),
        'heating':np.array([[1.,2.]],dtype=np.float32),'module_centers_xy':np.array([[[2.,2.],[5.,3.]]],dtype=np.float32),
        'fluid_fields_query_features':np.ones((4,2),dtype=np.float32),
        'solid_temperature_query_features':np.tile(np.array([[0.,0.],[.1,.2]]),(2,1)),
        'active_module_ids':np.array(['M:0','M:1']),'interface_receiver_module_ids':np.repeat(['M:0','M:1'],2),
        'solid_temperature_receiver_module_ids':np.repeat(['M:0','M:1'],2),'fluid_fields_values':object()}
    structure,fluid,local,heat=_response_structure(atlas,0,'cpu')
    assert structure['module_source_ids'].tolist()==[[0,1]] and fluid.shape==(1,4,2) and local.shape==(1,2,2)
    assert heat.tolist()==[[1.,2.]]
    atlas['active_module_ids']=np.array(['0001:module:3','0001:module:7'])
    for role in ('interface','solid_temperature'):atlas[role+'_receiver_module_ids']=np.repeat(['0001:module:3','0001:module:7'],2)
    assert _response_structure(atlas,0,'cpu')[0]['module_source_ids'].tolist()==[[3,7]]
    atlas['solid_temperature_receiver_module_ids']=np.array(['M:0','M:1','M:0','M:1'])
    with pytest.raises(ValueError,match='source IDs'):_response_structure(atlas,0,'cpu')


@pytest.mark.parametrize('mode',['zero_context','zero_projected_values'])
def test_fixed_prepared_E_read_effect_restores_and_changes_native_K(mode):
    model=JointNativeCore.from_a2d(_core('A2-D').double()).eval()
    with torch.no_grad():model.environment_conditioning.weight.copy_(torch.eye(model.context_dim,dtype=torch.double)*.2)
    scene,query=_double(_scene(1)),_double(_query(1))
    with torch.no_grad():
        prepared=model.prepare(scene);baseline=model.export_source_kernels(prepared,query)
        fingerprint=_fixed_read_fingerprint(model,prepared,query)
        original=prepared.joint_environment_values.clone()
        with environment_read_intervention(model,mode):
            changed=model.export_source_kernels(prepared,query)
            assert _fixed_read_fingerprint(model,prepared,query)==fingerprint
        restored=model.export_source_kernels(prepared,query)
        assert not torch.equal(baseline,changed)
        assert torch.equal(restored,baseline)
        assert torch.equal(prepared.joint_environment_values,original)
        assert _fixed_read_fingerprint(model,prepared,query)==fingerprint
        model.validate_prepared_state(prepared,query)


def test_numerical_exports_respect_budget_and_immutability(tmp_path):
    destination=tmp_path/'evidence.npz'
    with pytest.raises(OSError,match='budget'):write_arrays(destination,{'large':np.zeros(100)},budget_bytes=4)
    assert not destination.exists()
    write_arrays(destination,{'small':np.zeros(4)})
    with pytest.raises(FileExistsError):write_arrays(destination,{'small':np.zeros(4)})


def test_comparison_requires_same_age_receivers_and_keeps_small_denominators_unresolved():
    fields={name:{'mean':1.,'p90':1.,'worst':1.} for name in THERMAL_FIELDS};fields['q_proxy']={'mean':0.,'p90':0.,'worst':0.}
    base={'schema':BOARD_SCHEMA,'dataset':'thermal','split':'validation','completed_epoch':4000,
        'panel_sha256':'a'*64,'query_receipts':{'a':'b'*64},'metrics':{'overall':{'fields':fields},
        'by_module_count':{'M1':{'case_count':1,'fields':fields}}},'TRAIN_role_scales':dict.fromkeys(THERMAL_FIELDS,1.),
        'typed_work':{'M':{'available':True,'equal_case_selected_fraction':.3},'E':{'available':False}}}
    boards={name:copy.deepcopy(base) for name in ('D9-M','D9-ME','H9-ME','S9-ME')}
    for name,board in boards.items():board['arm']=name
    boards['H9-ME']['typed_work']['M']['equal_case_selected_fraction']=.2
    result=compare_physical_boards(boards)
    unresolved=result['comparisons']['H9-ME_versus_D9-M']['q_proxy']['mean']
    assert unresolved['relative_change'] is None and unresolved['within_band'] is None
    assert result['trained_H_S_work_match']['M']['unmatched_by_more_than_five_percentage_points']
    boards['H9-ME']['completed_epoch']=5000
    with pytest.raises(ValueError,match='literal age'):compare_physical_boards(boards)
    boards['H9-ME']['completed_epoch']=4000
    boards['H9-ME']['TRAIN_role_scales']['u']=0
    with pytest.raises(ValueError,match='strictly positive'):compare_physical_boards(boards)


def test_field_effects_retain_individual_units_and_native_validity():
    result=_effect_magnitudes(np.array([[3.,4.],[100.,100.]]),np.array([[True],[False]]),['u','temperature'])
    assert result['per_channel']['u']['rms']==3 and result['per_channel']['temperature']['rms']==4
    assert result['valid_entries']==2 and result['pooled_units_warning']
    arrays={'interface_valid':np.array([[True,True]]),'outside_valid':np.array([[True,False]])}
    mask=_effect_validity('interface_prediction',arrays,(1,2,2))
    result=_effect_magnitudes(np.array([[[2.,3.],[4.,100.]]]),mask,['temperature','q_proxy'])
    assert result['per_channel']['q_proxy']['rms']==3
    assert result['per_channel']['temperature']['valid_entries']==2


def test_native_K_effects_exclude_padding_but_keep_zero_measure_sources():
    query=SimpleNamespace(query_ids=torch.tensor([[7,8]]),query_mask=torch.tensor([[True,False]]))
    scene=SimpleNamespace(module_mask=torch.tensor([[True,True,False]]),module_ids=torch.tensor([[10,11,-1]]),
        module_measure=torch.tensor([[1.,0.,0.]]))
    summary,mask=_kernel_effect_magnitudes(torch.tensor([[[1.,3.,0.],[100.,100.,0.]]]),query,scene)
    assert summary['rms']==pytest.approx(np.sqrt(5)) and summary['valid_entries']==2
    assert summary['per_native_source'][1]['native_module_id']==11
    assert summary['per_native_source'][1]['rms']==3 and summary['per_native_source'][1]['valid_entries']==1
    assert mask.tolist()==[[[True,True,False],[False,False,False]]]


def test_dense_seams_replay_exact_sparse_crossing_coordinates(monkeypatch,tmp_path):
    import honf_runtime.joint_native_physics as physics
    core=JointNativeCore.from_a2d(_core('A2-D').double()).eval()
    scene,query=_double(_scene(1)),_double(_query(1));organization=core.prepare(scene)
    provider=SimpleNamespace(adapter=SimpleNamespace(query_features=lambda coordinates,support:coordinates*2,
        physical_velocity=lambda field,coordinates:(field,None,None)))
    batch=SimpleNamespace(case_keys=('A',),scene_inputs=(SimpleNamespace(support=object()),),receivers=object())
    monkeypatch.setattr(physics,'validate_native_panel',lambda *args:None)
    monkeypatch.setattr(physics,'native_batches',lambda *args,**kwargs:iter([batch]))
    monkeypatch.setattr(physics,'predict_physical',lambda *args:(None,{}))
    monkeypatch.setattr(physics,'_prepared_native',lambda *args:(organization,query))
    source={'schema':'round9_spatial_seam_probe_v1','dataset':'wind','panel_sha256':'a'*64,'completed_epoch':4000,
        'cases':[{'case_id':'A','available':True,'crossing_center':query.coordinates[0,:1].tolist(),
            'query_id':int(query.query_ids[0,0]),'shrinking_steps_native_coordinate_units':[.001,.0001],
            'crossing_bracket_width':1e-8}]}
    result=evaluate_seams(core,provider,{'epoch':4000},{'dataset':'wind','panel_sha256':'a'*64},tmp_path/'seams',crossings=source)
    assert result['shared_crossings_sha256']==canonical_sha256(source)
    case=result['cases'][0]
    assert case['available'] and case['crossing_scope']=='replay of exact shared crossing coordinates'
    assert len(case['shrinking_probes'])==2
    saved=np.load(tmp_path/'seams'/case['arrays'])
    np.testing.assert_array_equal(saved['crossing_center'],query.coordinates[0,:1])


def test_continuous_seam_helper_rebuilds_query_features_and_keeps_typed_support():
    core=JointNativeCore.from_a2d(_core('A2-D').double(),routing_mode='joint').eval();core.set_training_clock(4001,0)
    scene,template=_double(_scene(1)),_double(_query(1))
    prepared=core.prepare(scene)
    provider=SimpleNamespace(adapter=SimpleNamespace(query_features=lambda coordinates,support:coordinates*2))
    batch=SimpleNamespace(scene_inputs=(SimpleNamespace(support=object()),),receivers=SimpleNamespace(heat=torch.ones(1,3,dtype=torch.double)))
    coordinates=template.coordinates[:,:2]+.001
    query=_continuous_query(provider,batch,prepared,template,coordinates,template.query_ids[:,:2])
    torch.testing.assert_close(query.features,coordinates*2)
    model=SimpleNamespace(core=core,prepare_native=object(),field_std=torch.ones(5,dtype=torch.double),field_mean=torch.zeros(5,dtype=torch.double))
    with torch.no_grad():values,support,dispatch=_continuous_values(provider,model,batch,prepared,query)
    assert values.shape==(2,4) and support.shape==(2,7) and dispatch.shape==(2,)


def test_present_pair_work_denominator_does_not_count_padding_as_learned_sparsity():
    result=typed_work_summary([{'source_read_joint_M_selected_rows':2,'source_read_joint_M_candidate_rows':12,
        'source_read_joint_M_value_rows':12,'joint_M_available_present_rows':2}])
    assert result['M']['equal_case_selected_fraction']==1.
    assert result['M']['executed_nonlinear_rows']==12
    assert result['E']['available'] is False
