import numpy as np
import pytest
import torch
from types import SimpleNamespace

from channelthermal.training.thermal_finite_evaluation import field_metrics, summarize_actions, intermodule_fluid_mask, evaluate_native_panel


def test_physical_mask_is_shared_and_empty_or_misaligned_is_rejected():
    result = field_metrics(np.array([2., 100., 5.]), np.array([1., -100., 3.]), [True, False, True])
    assert result['count'] == 2
    assert result['rmse'] == pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError):
        field_metrics([1.], [1.], [False])
    with pytest.raises(ValueError):
        field_metrics([1.], [1., 2.], [True, True])


def test_equal_family_resolved_coverage_and_ordering():
    def row(case, physical, model):
        return {'case_id': case, 'delta': {k: {'physical': physical, 'model': model} for k in ('J', 'P')}}
    rows = [row('one', 2., 3.), row('one', -2., -3.), row('two', .1, 4.)]
    result = summarize_actions(rows, {'J': 1., 'P': 1.})['J']
    assert result['all_action_equal_family_mae'] == pytest.approx(2.45)
    assert result['resolved_family_count'] == 1 and result['unresolved_family_count'] == 1
    assert result['resolved_equal_family_mae'] == 1.
    assert result['resolved_equal_family_mean_abs_effect'] == 2.
    assert (result['resolved_sign_correct'], result['resolved_sign_count']) == (2, 2)
    assert (result['resolved_order_wrong'], result['resolved_order_count']) == (0, 1)


def test_unavailable_families_and_all_action_denominators_are_reported():
    def row(case, physical, model):
        return {'case_id': case, 'delta': {k: {'physical': physical, 'model': model} for k in ('J', 'P')}}
    rows = [row('one', 2., 1.), row('one', .1, -.1)]
    result = summarize_actions(rows, {'J': 1., 'P': 1.}, family_ids=['one', 'missing'])['J']
    assert result['family_count'] == 2 and result['families_without_actions'] == 1
    assert result['unresolved_family_count'] == 1
    assert (result['all_sign_correct'], result['all_action_count']) == (1, 2)
    assert (result['all_order_wrong'], result['all_order_count']) == (0, 1)
    assert result['resolved_order_count'] == 0


def test_representative_query_trace_uses_requested_read_mode(monkeypatch):
    from tools import thermal_finite_response_evaluate as evaluator
    from pathlib import Path

    table = SimpleNamespace(offsets=torch.tensor([[0,2]]), member_index=torch.tensor([0,1]),
        original_id=torch.tensor([1,2]),is_module=torch.tensor([True,False]),
        access_weight=torch.tensor([.3,.7]),applied_exchange=torch.tensor([.3,.4]),
        structural_self=torch.tensor([0.,.3]),measure=torch.tensor([1.,1.]))
    scene=SimpleNamespace(module_ids=torch.tensor([[1]]),environment_ids=torch.tensor([[2]]),
        module_coordinates=torch.tensor([[[1.,2.]]]),environment_coordinates=torch.tensor([[[2.,3.]]]),
        module_mask=torch.tensor([[True]]),environment_mask=torch.tensor([[True]]),
        module_measure=torch.tensor([[1.]]),environment_measure=torch.tensor([[2.]]),
        module_characteristic_lengths=torch.ones((1,1,2)),environment_characteristic_lengths=torch.ones((1,1,2)))
    org = SimpleNamespace(e_access_directory=SimpleNamespace(full=table, restricted=table),
                          context_stage_traces=[], scene=scene)
    query = SimpleNamespace(num_queries=1, coordinates=torch.zeros(1, 1, 2), query_ids=torch.tensor([[3]]))
    structure={'domain_length_x':torch.tensor([12.]),'domain_length_y':torch.tensor([6.]),
               'material_params':torch.tensor([[0.,0.,0.,0.,0.,.45]])}
    result = {'prepared': SimpleNamespace(organization=org, query_batch=query,structure=structure), 'prediction': {},
              'reference': {}, 'mask': {}}
    seen = []
    core = SimpleNamespace(decode=lambda org, q, return_trace, intervention:
                           (seen.append(intervention['read_mode']) or SimpleNamespace(trace={})))
    monkeypatch.setattr(evaluator, 'select_query_rows_v1', lambda query, rows: query)
    exports=[]
    monkeypatch.setattr(evaluator.np, 'savez_compressed', lambda *args, **kwargs: exports.append(kwargs))
    monkeypatch.setattr(Path, 'write_text', lambda self, text: len(text))
    evaluator.save_organization('unused', result, SimpleNamespace(core=core), 'full')
    evaluator.save_organization('unused', result, SimpleNamespace(core=core), 'restricted')
    assert seen == ['full', 'restricted']
    assert np.allclose(exports[0]['C_restricted'], [[.3,.7]])
    assert exports[0]['module_xy'].tolist() == [[1.,2.]]
    assert exports[0]['domain_bounds'].tolist() == [0.,12.,0.,6.]
    # Exercise the maintained saved-scene loader without writing an artifact.
    from honf_viz import organization_data
    class Arrays(dict):
        @property
        def files(self): return list(self)
        def __enter__(self): return self
        def __exit__(self, *args): return False
    monkeypatch.setattr(organization_data.np, 'load', lambda *args,**kwargs: Arrays(exports[0]))
    monkeypatch.setattr(Path, 'read_text', lambda self: '{"case":"1","epoch":100}')
    loaded=organization_data.load_scene('case',Path('unused.npz'))
    assert loaded.m_ids == ('M1',) and loaded.e_ids == ('E2',)
    assert np.allclose(loaded.C,[[.3,.7]]) and loaded.bounds == (0.,12.,0.,6.)


def test_guarded_route_coefficient_handles_zero_measure_empty_and_closed_rows():
    from tools.thermal_finite_response_evaluate import _guarded_context_coefficients
    from honf_forward_core.continuous_joint.preparation import contextual_value_update
    snap=SimpleNamespace(raw_scores=torch.tensor([[[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]]]),
        donor_measure=torch.tensor([[1.,3.,0.]]),
        allowed=torch.tensor([[[True,True,True],[False,False,False],[True,True,True]]]))
    gate=torch.tensor([[[.4,.8,1.],[0.,0.,0.],[0.,0.,0.]]])
    result=_guarded_context_coefficients(snap,gate)
    weighted=torch.tensor([.1,.6,0.]); z=weighted.sum()
    assert torch.allclose(result[0,0,0],weighted/(z+.001*(1-z).square()))
    assert torch.equal(result[0,0,1:],torch.zeros_like(result[0,0,1:]))
    attention=SimpleNamespace(heads=1,head_width=1,width=1,output=torch.nn.Linear(1,1,bias=False))
    query=torch.zeros((1,1,3,1)); key=torch.zeros((1,1,3,1)); value=torch.ones((1,1,3,1))
    _,actual,_,_=contextual_value_update(attention,query,key,value,torch.zeros((1,1,3,3)),
        snap.donor_measure,snap.allowed,torch.ones((1,3),dtype=torch.bool),lambda *args,**kwargs:gate,
        route_name='mm',stage_identity='test',destination_ids=torch.tensor([[0,1,2]]),donor_ids=torch.tensor([[0,1,2]]))
    assert torch.equal(result,actual)


def test_unpacked_directory_preserves_original_id_and_rejects_mismatch():
    from tools.thermal_finite_response_evaluate import _dense_directory
    table=SimpleNamespace(offsets=torch.tensor([[0,2]]),member_index=torch.tensor([0,1]),
        original_id=torch.tensor([7,11]),is_module=torch.tensor([True,False]),access_weight=torch.tensor([.2,.8]))
    assert np.allclose(_dense_directory(table,np.array([7]),np.array([11])),[[.2,.8]])
    table.original_id=torch.tensor([8,11])
    with pytest.raises(ValueError,match='original IDs'):
        _dense_directory(table,np.array([7]),np.array([11]))


def test_intermodule_hull_is_input_defined_and_degenerate_cases_unavailable():
    xy=np.array([[.2,.2],[.9,.9],[.1,.1]])
    centers=np.array([[0.,0.],[1.,0.],[0.,1.]])
    mask,reason=intermodule_fluid_mask(xy,[True,True,False],centers)
    assert reason is None and mask.tolist()==[True,False,False]
    mask,reason=intermodule_fluid_mask(xy,[True]*3,centers[:2])
    assert not mask.any() and 'noncollinear' in reason
    mask,reason=intermodule_fluid_mask(xy,[True]*3,np.array([[0.,0.],[1.,0.],[2.,0.]]))
    assert not mask.any() and 'noncollinear' in reason


def test_native_panel_preserves_direct_local_distinction_and_source_clock(monkeypatch):
    from honf_runtime import joint_native_evaluation as native, contextual_frontier_evaluation as frontier
    from honf_runtime import member_routing_evaluation as routing
    t=lambda x: torch.as_tensor(x,dtype=torch.float32)
    names=('u','v','p','omega','temperature_fluid','temperature_surface','temperature_material','q_proxy')
    direct={'case_id':'1','module_count':3,'fields':{n:{'rmse':1.} for n in names},
            'module_peaks':[{'error':1.}]}
    monkeypatch.setattr(native,'thermal_record_and_arrays',lambda pred,batch:(direct,{}))
    monkeypatch.setattr(routing,'_native_arrays_with_residuals',lambda pred,batch,case:{'direct':np.array([1])})
    monkeypatch.setattr(frontier,'_known_module_counts',lambda provider:(3,))
    monkeypatch.setattr(frontier,'_summarize_nine_observables',lambda rows,**kwargs:{'u':rows[0]['flow_rmse']['u']})
    stencil=lambda x: SimpleNamespace(valid=torch.as_tensor(x,dtype=torch.bool))
    prep=SimpleNamespace(source_present=t([[1,1,1]]),stencils={'fluid':stencil([[1,1]]),
        'outside':stencil([[[1,1],[1,1],[1,1]]]),'surface':stencil([[[1,1],[1,1],[1,1]]]),
        'material':stencil([[[1,1],[1,1],[1,1]]])},
        scene_batch=SimpleNamespace(module_coordinates=t([[[0,0],[1,0],[0,1]]])))
    target=SimpleNamespace(case_ids=('1',),point_weights=t([[1,1]]),field_targets=t([[[0]*5,[0]*5]]),
        teacher_ports=t(np.zeros((1,3,2,5))),outside_valid=t(np.ones((1,3,2))),
        interface_valid64=t(np.ones((1,3,2))),interface_target64=t(np.zeros((1,3,2,2))),
        material_targets=t(np.zeros((1,3,2))))
    batch=SimpleNamespace(case_keys=('1',),targets=target,scene_inputs=object(),
                          receivers=SimpleNamespace(fluid_xy=t([[[.1,.1],[.9,.9]]])))
    prediction=SimpleNamespace(native_prepared=prep,native_main={'pred_field':t(np.ones((1,2,5)))},
        outside64={'restricted':t(np.ones((1,3,2)))},local1={'restricted':{
        'interface_pred':t(np.full((1,3,2,2),2.)), 'internal_temperature':t(np.full((1,3,2,1),2.))}})
    epochs=[]
    assembled={'flow_rmse':{'u':2.}}
    provider=SimpleNamespace(native_evaluation_batches=lambda **kwargs:iter([batch]),
        predict_native=lambda *args,**kwargs:(epochs.append(kwargs['epoch']) or (prediction,{})),
        validation_metrics=lambda *args:{'case_rows':[assembled]},
        reduce_native_evaluation_metrics=lambda *args,**kwargs:{'per_case':[assembled]},
        work_counts=lambda *args:{'pairs':3})
    model=SimpleNamespace(eval=lambda:None,architecture='test')
    result=evaluate_native_panel(model,provider,{'epoch':100,'experiment_identity':{'recipe':{}}},['1'],['1'])
    assert epochs==[0] and result['metrics']['u']==1. and result['assembled_local_metrics']['u']==2.
    assert result['additional_metrics']['outside_T']['mean']==1.
    assert result['additional_metrics']['local_q']['mean']==2.
    assert result['additional_metrics']['intermodule_fluid_T']['mean']==1.
    assert result['representative_arrays']['1']['outside_target'].shape==(3,2)


def test_pair_disable_restores_parameters_after_failure():
    from tools.thermal_finite_response_evaluate import disabled_pair_content
    pair=torch.nn.Module();pair.U=torch.nn.Parameter(torch.tensor([3.]))
    core=torch.nn.Module();core.pair_content=pair
    model=SimpleNamespace(core=core)
    with pytest.raises(RuntimeError):
        with disabled_pair_content(model):
            assert pair.U.item()==0.
            raise RuntimeError('diagnostic failed')
    assert pair.U.item()==3.


def test_holdout_retry_keeps_failed_attempt_and_uses_success(monkeypatch):
    import json
    from pathlib import Path
    from tools.thermal_finite_response_evaluate import ledger_paths
    lines=[{'event':'started','attempt':1,'role':'response_holdout','allocation':'edited','case_id':'5','action':'single_plus'},
        {'event':'finished','attempt':1,'status':'failed','serialization_status':None},
        {'event':'started','attempt':2,'role':'response_holdout','allocation':'edited','case_id':'5','action':'single_plus'},
        {'event':'finished','attempt':2,'status':'converged','serialization_status':'stored','raw_case_dir':'existing'}]
    monkeypatch.setattr(Path,'read_text',lambda self:'\n'.join(map(json.dumps,lines)))
    entry=ledger_paths('unused')[('5','single_plus')]
    assert entry['path']=='existing' and [row['status'] for row in entry['attempts']]==['failed','converged']
