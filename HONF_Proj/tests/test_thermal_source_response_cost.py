"""Actual native cost scopes and learned-matrix approximation bounds on CPU."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
if str(TOOLS) not in sys.path: sys.path.insert(0, str(TOOLS))

from channelthermal.dependency_flow import ThermalFlowReader
from channelthermal.source_response import SourceResponseThermalModel, ThermalSourceResponse
from thermal_source_response_cost import (
    NATIVE_ROLES,
    balanced_radii,
    candidate_work,
    complete_call,
    compression_probe,
    incumbent_physical,
    low_rank_response,
    native_temperature_vector,
    physical_heat,
    prepare_candidate,
    prepared_forcing_cost,
)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def case(mode):
    torch.manual_seed(0)
    thermal = ThermalSourceResponse({'hidden':8,'message':8,'mode':mode}, nx=16,ny=8,
        environment_nx=3,environment_ny=2)
    stats = {'field_mean_by_channel':np.arange(5,dtype=np.float32),
        'field_std_by_channel':np.arange(1,6,dtype=np.float32)}
    model = SourceResponseThermalModel(thermal,ThermalFlowReader('D-sep',{'hidden':8,'message':8}),stats).eval().requires_grad_(False)
    structure = {'module_centers':torch.tensor([[[3.1,2.1],[7.2,3.2]]]),
        'module_present':torch.ones(1,2),'material_params':torch.tensor([[.018,.01,.02,1.,2.,.45]]),
        'heat_powers':torch.tensor([[100.,200.]]),'re':torch.tensor([[50.]]),'u_in':torch.ones(1,1),
        'domain_length_x':torch.tensor([[12.]]),'domain_length_y':torch.tensor([[6.]])}
    arguments = {'structure':structure,'query_xy':torch.tensor([[[.375,.375],[11.625,5.625],[3.,2.]]]),
        'local_query_points':torch.tensor([[[-.5,0.],[.5,0.]]]),
        'local_module_params':torch.tensor([[[1.,.01,1.,.45],[2.,.01,1.,.45]]]),
        'interface_condition':torch.zeros(1,2,4,5)}
    saved = {'train_config':{'dataset':{'normalize_inputs':True}},'global_normalization_stats':stats}
    return model,arguments,saved


def test_physical_heat_and_incumbent_units_preserve_actual_derivatives():
    heat = torch.tensor([[.2,-.1]],requires_grad=True)
    metadata = {'train_config':{'dataset':{'normalize_inputs':True,'normalize_targets':True}},
        'global_normalization_stats':{'heat_power_std':3.,'heat_power_mean':10.,
        'field_mean_by_channel':np.arange(5),'field_std_by_channel':np.arange(1,6),
        'internal_temperature_mean':2.,'internal_temperature_std':4.,
        'interface_target_mean':np.array([3.,5.]),'interface_target_std':np.array([2.,7.])}}
    actual = physical_heat({'structure':{'heat_powers':heat}},metadata)
    torch.testing.assert_close(actual,torch.tensor([[10.6,9.7]]))
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),heat)[0],torch.full_like(heat,3.))
    outputs = {'pred_field':torch.ones(1,2,5,requires_grad=True),
        'pred_internal_temperature':torch.ones(1,2,3,1,requires_grad=True),
        'pred_interface':torch.ones(1,2,4,2,requires_grad=True),
        'pred_port_condition':torch.ones(1,2,4,5,requires_grad=True)}
    converted = incumbent_physical(outputs,metadata)
    gradient, = torch.autograd.grad(converted['pred_interface'].sum(),outputs['pred_interface'])
    torch.testing.assert_close(gradient,torch.tensor([2.,7.]).expand_as(gradient))
    assert converted['pred_port_condition'] is outputs['pred_port_condition']
    plural=dict(metadata['global_normalization_stats'])
    plural['interface_targets_mean']=plural.pop('interface_target_mean')
    plural['interface_targets_std']=plural.pop('interface_target_std')
    plural_metadata={**metadata,'global_normalization_stats':plural}
    torch.testing.assert_close(incumbent_physical(outputs,plural_metadata)['pred_interface'],converted['pred_interface'])


@pytest.mark.parametrize('mode',['direct','group'])
def test_prepared_cost_uses_real_native_complete_calls_and_affine_roles(mode):
    model,arguments,saved = case(mode)
    with torch.no_grad(): output = complete_call('R-'+mode,model,saved,arguments)
    assert all(name in output for name in NATIVE_ROLES)
    assert output['pred_internal_temperature'].shape == (1,2,2,1)
    receipt = {'complete_calls':0}
    rows = prepared_forcing_cost(model,saved,arguments,'synthetic',torch.device('cpu'),receipt=receipt)
    assert len(rows)==10 and receipt['complete_calls']==40
    assert receipt['thermal_increment_calls']==1
    for row in rows:
        assert row['instrumented_prepared_scopes']['native_role_extraction_and_validation_seconds']>=0
        assert all(error['passed'] for error in row['cold_prepared_full_output_errors'].values())
        assert all(error['passed'] for error in row['cold_prepared_increment_errors'].values())
    work = candidate_work(model,arguments,saved)
    assert work['old_thermal_wrapper_calls']==0 and work['near_coefficient_rows']>0
    assert work['source_rows']==2


def test_predeclared_radii_and_svd_learned_response_bound():
    heat=torch.tensor([[1.,5.,9.,50.]],dtype=torch.float64)
    radii=balanced_radii(heat,torch.tensor([[1.,1.,1.,0.]],dtype=torch.float64),0.,10.)
    torch.testing.assert_close(radii,torch.tensor([[1.,1.,1.,0.]],dtype=torch.float64))
    generator=torch.Generator().manual_seed(0)
    matrix=torch.randn(19,3,generator=generator,dtype=torch.float64)
    delta=torch.tensor([.3,-.2,-.1],dtype=torch.float64)
    for budget in (0.,.1,1.,100.):
        factors=low_rank_response(matrix,radii[0,:3],budget)
        error=(factors['left']@(factors['right']@delta)-matrix@delta).abs().max()
        assert error <= factors['omission_bound']+1e-13
        assert factors['omission_bound']<=budget+1e-13
        assert factors['physical_operator_factorizations']==0
    with pytest.raises(ValueError,match='nonnegative'): low_rank_response(matrix,radii[0,:3],-1.)


@pytest.mark.parametrize('mode',['direct','group'])
def test_native_compression_prices_same_temperature_catalogue_and_keeps_baseline(mode):
    model,arguments,saved = case(mode)
    heat=physical_heat(arguments,saved)
    with torch.no_grad(): prepared=prepare_candidate(model,arguments)
    radii=balanced_radii(heat,arguments['structure']['module_present'],0.,3.)
    increments=torch.tensor([[[.05,-.05]],[[.4,-.4]]])
    native=prepared['thermal']
    # Explicit synthetic finite responses test the post-selection join only.
    with torch.no_grad():
        synthetic_truth=torch.stack([native_temperature_vector(model.thermal,native,
            model.thermal.core.apply_increment(native.response,delta))+.01 for delta in increments])
    mask=torch.ones_like(synthetic_truth[0],dtype=torch.bool);mask[:,0]=False
    synthetic_truth[:,:,0]=float('nan')
    rows=compression_probe(model,prepared,heat,increments,radii,.1,torch.device('cpu'),
        physical_reference_increments=synthetic_truth,physical_receiver_mask=mask)
    assert [row['budget_fraction'] for row in rows]==[0.,.005,.01,.02]
    for row in rows:
        assert row['baseline_unchanged'] and row['physical_factorizations']==0
        assert len(row['samples']['full_response'])==len(row['samples']['compressed_response'])==5
        assert row['errors'][0]['inside_declared_balanced_box']
        assert row['errors'][1]['fallback_full_response']
        assert all(error['bound_holds'] for error in row['errors'])
        assert all(error['physical_error_triangle_holds'] for error in row['errors'])
        assert all(error['physical_receiver_count']==mask.sum() for error in row['errors'])
        if mode=='group': assert not row['details']['actual_sparse_application_savings']
