import numpy as np
import pytest
from channelthermal.response_control.transfer_probe import FIT, DEV, atlas_edits, organization_features, fit_three_arms, evaluate_probe, useful_transfer, request_distribution


def test_permutation_is_consistent_and_single_has_no_pair_features():
    kw = dict(member_ids=['a','b'], source_ids=['a','b','c'], query_m=np.array([[.1,.2,.7]]),
              objective_weights=np.array([1.]), anchor_indices=np.array([[0,1]]), anchor_weights=np.array([[.5,.5]]),
              e_measures=np.array([1.,1.]), directory_m=np.array([[.2,.3,.5],[.4,.1,.5]]),
              mm=np.arange(9).reshape(3,3), em=np.array([[1,2,3],[4,5,6]]), displacement=np.ones((2,2)))
    actual = organization_features(**kw, permutation=[2,0,1])
    assert np.allclose(actual[:2], [.7,.1])
    assert np.allclose(actual[3:5], [6,2])
    kw['displacement'][1] = 0
    assert np.all(organization_features(**kw)[2:5] == 0)


def test_fit_only_and_padded_controls():
    rng = np.random.default_rng(5); x = rng.normal(size=(16,24)); xp=x.copy(); xp[:,18:]=xp[::-1,18:]
    y = np.stack((x[:,0],x[:,1]), axis=1); fam = list(FIT)*4
    probes, losses = fit_three_arms(x,xp,y,fam,[1,1])
    assert set(probes) == set(losses) == {'no_organization','aligned','permuted'}
    assert np.all(probes['no_organization'].coefficients[18:] == 0)
    with pytest.raises(ValueError): fit_three_arms(x,xp,y,[DEV[0]]*16,[1,1])
    assert len(evaluate_probe(probes['aligned'], x, y, fam)) == 4


def test_transfer_criterion_and_no_op():
    def families(v): return {f:{'normalized_mae':[v, 0]} for f in DEV}
    results={'aligned':families(.5), 'no_organization':families(1.), 'permuted':families(1.)}
    assert useful_transfer(results, pressure_usable=True)
    with pytest.raises(ValueError): request_distribution(useful=False, fit_temperature_changes=[-1], predicted_changes=np.zeros((9,2)), predicted_baseline_pressure=1, physical_changes=np.zeros((9,2)), physical_baseline_pressure=1, displacement_cost=np.zeros(9), fit_temperature_scale=1, inherited_pressure_scale=1)
    pred=np.ones((9,2)); actual=np.zeros((9,2))
    rows=request_distribution(useful=True, fit_temperature_changes=[-1,-2], predicted_changes=pred, predicted_baseline_pressure=1, physical_changes=actual, physical_baseline_pressure=1, displacement_cost=np.zeros(9), fit_temperature_scale=1, inherited_pressure_scale=1)
    assert len(rows)==4 and all(np.isclose(sum(r['probabilities']),1) for r in rows)
    assert all(len(r['draws']) == 3 for r in rows)


def test_loss_history_includes_unpenalized_intercept_and_ridge():
    from channelthermal.response_control.transfer_probe import RidgeProbe, probe_history
    probe = RidgeProbe(np.zeros(2),np.ones(2),np.ones(2),np.array([1.,2.]),np.array([[1.,2.],[0.,1.]]))
    y=np.array([[1.,2.],[2.,3.],[3.,4.]])
    pred=np.array([[1.,2.],[1.,2.],[3.,3.]])
    history=probe_history(probe,pred,y,[True,True,False],[False,False,True])
    assert history['fit_ridge_l2'] == [0.,3.]
    assert history['fit_total_mse_plus_l2'][1] == history['fit_mse'][1] + 3.
    assert history['dev_mse'] == [4.,.5]


def test_pressure_ranking_and_common_request_baseline():
    from channelthermal.response_control.transfer_probe import RidgeProbe, evaluate_probe, request_distributions
    probe=RidgeProbe(np.zeros(2),np.ones(2),np.ones(2),np.zeros(2),np.eye(2))
    x=np.array([[0.,3.],[1.,1.],[2.,2.]])
    y=np.array([[0.,1.],[2.,2.],[1.,3.]])
    row=evaluate_probe(probe,x,y,['f']*3)['f']
    assert row['pressure_ranking']['pairwise_order_error_fraction'] == 2/3
    assert row['pressure_ranking']['top1_regret'] == 1.
    p=np.zeros((9,2))
    arms=request_distributions(predictions_by_arm={'a':p,'b':p}, predicted_baseline_pressure=10., useful=True,
        fit_temperature_changes=[-1.,-2.], physical_changes=p, physical_baseline_pressure=10.,
        displacement_cost=np.zeros(9),fit_temperature_scale=1., inherited_pressure_scale=1.)
    assert arms['a'] == arms['b']
    assert arms['a'][0]['pressure_limit'] == 10.1


def test_driver_dispatch_and_signed_pressure_trace(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    path=Path(__file__).resolve().parents[2]/'tools/thermal_organization_transfer_probe.py'
    spec=importlib.util.spec_from_file_location('thermal_transfer_driver_test',path)
    driver=importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    monkeypatch.setattr(driver,'load_trusted_checkpoint',lambda *a,**k: {'experiment_identity':{'recipe':{}}})
    monkeypatch.setattr(driver,'load_contextual_checkpoint',lambda *a: ('legacy','provider','payload'))
    assert driver.load_probe_model('path','cpu') == ('legacy','payload')
    monkeypatch.setattr(driver,'load_trusted_checkpoint',lambda *a,**k: {'experiment_identity':{'recipe':{'thermal_port_response':{'arm':'C0'}}}})
    monkeypatch.setitem(sys.modules,'channelthermal.training.thermal_port_response_task',SimpleNamespace(
        load_thermal_port_response_checkpoint=lambda *a,**k: ('port','local','payload')))
    assert driver.load_probe_model('path','cpu') == ('port','payload')
    monkeypatch.setattr(driver,'select_query_rows_v1',lambda query,ids: SimpleNamespace(ids=ids))
    class Core:
        query_tile_size=2
        def decode(self, organization, query, **kwargs):
            assert kwargs['intervention'] == {'read_mode':'full'}
            return SimpleNamespace(trace={'member_weights_M':torch.ones(1,len(query.ids),2),
                                          'member_weights_E':torch.ones(1,len(query.ids),3)})
    prepared=SimpleNamespace(query_batch='queries',organization='organization')
    weights=torch.tensor([0.,-.25,0.,.5])
    selected,_,trace=driver.capture_pressure_trace(SimpleNamespace(core=Core()),prepared,weights)
    assert selected.tolist()==[1,3] and weights[selected].tolist()==[-.25,.5]
    assert trace['member_weights_M'].shape == (1,2,2)
    with pytest.raises(ValueError): driver.capture_pressure_trace(SimpleNamespace(core=Core()),prepared,torch.zeros(4))


def test_objective_query_capture_chunks_both_roles_in_requested_order(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    path=Path(__file__).resolve().parents[2]/'tools/thermal_organization_transfer_probe.py'
    spec=importlib.util.spec_from_file_location('thermal_transfer_chunk_test',path)
    driver=importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    monkeypatch.setattr(driver,'select_query_rows_v1',lambda query,ids: SimpleNamespace(ids=ids))
    class Core:
        query_tile_size=3
        def __init__(self): self.chunks=[]
        def decode(self, organization, query, **kwargs):
            assert len(query.ids) <= self.query_tile_size
            assert kwargs['intervention'] == {'read_mode':'full'}
            self.chunks.append(query.ids)
            values=torch.tensor(query.ids, dtype=torch.float32)[None,:,None]
            return SimpleNamespace(trace={'member_weights_M':values, 'member_weights_E':values+100})
    core=Core();model=SimpleNamespace(core=core);prepared=SimpleNamespace(query_batch='queries',organization='org')
    selected=torch.tensor([8,2,9,0,6,4,1])
    query,trace=driver.capture_query_trace(model,prepared,selected)
    assert query.ids==selected.tolist()
    assert core.chunks==[[8,2,9],[0,6,4],[1]]
    assert trace['member_weights_M'][0,:,0].tolist()==selected.tolist()
    assert trace['member_weights_E'][0,:,0].tolist()==[108,102,109,100,106,104,101]
    core.chunks.clear()
    weights=torch.tensor([-.2,0.,.4,.8,0.,-.5,1.])
    chosen,_,pressure=driver.capture_pressure_trace(model,prepared,weights)
    assert chosen.tolist()==[0,2,3,5,6]
    assert core.chunks==[[0,2,3],[5,6]]
    assert pressure['member_weights_M'][0,:,0].tolist()==chosen.tolist()
