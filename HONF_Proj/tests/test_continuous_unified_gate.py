"""Exact all-TRAIN catastrophic-window policy and accepted provenance contracts."""
import copy
import json
import pytest
from honf_runtime.continuous_unified_gate import (AGES,THERMAL_PRIORITY,catastrophic_review,
    validate_board_pair,bound,load_provenance_board,register_board,board_registry,process_time_headroom)


def test_triplet_forecast_keeps_evaluation_reserve_separate():
    hour=3600
    limits={'gpu_associated_seconds':72*hour,
            'gpu_associated_seconds_reserved_for_evaluation':8*hour}
    rows=[{'kind':'main','status':'completed','actual_seconds':20*hour},
          {'kind':'evaluation','status':'completed','actual_seconds':3*hour},
          {'kind':'main','status':'running','reserved_seconds':2*hour}]
    result=process_time_headroom(limits,rows,4*hour)
    assert result['total_remaining_seconds']==43*hour
    assert result['non_evaluation_remaining_seconds']==38*hour
    assert result['charged_evaluation_seconds']==3*hour
    assert result['evaluation_allowance_ok']
    rows[1]['actual_seconds']=9*hour
    assert not process_time_headroom(limits,rows,4*hour)['evaluation_allowance_ok']


def boards():
    result={}
    for epoch in AGES:
        result[epoch]={}
        for arm in ('D9-M','C10-H'):
            result[epoch][arm]=dict(schema='round9_native_physical_board_v1',dataset='thermal',split='train',
                completed_epoch=epoch,arm=arm,panel_sha256='p'*64,records=[{} for _ in range(150)],
                query_receipts={str(index):'a'*64 for index in range(150)},optimizer_updates=0,new_solver_attempts=0,
                WindTEST='locked',TRAIN_role_scales={name:1. for name in THERMAL_PRIORITY},
                metrics={'overall':{'case_count':150,'fields':{name:{'mean':1.} for name in THERMAL_PRIORITY}}})
    return result


def test_adjacent_same_field_threshold_and_missing_ages_are_not_substituted():
    data=boards()
    data[3100]['C10-H']['metrics']['overall']['fields']['temperature_fluid']['mean']=2.1
    data[3200]['C10-H']['metrics']['overall']['fields']['temperature_surface']['mean']=2.1
    assert catastrophic_review(data,dataset='thermal',panel_sha256='p'*64)['status']=='passed'
    data[3200]['C10-H']['metrics']['overall']['fields']['temperature_fluid']['mean']=2.1
    failure=catastrophic_review(data,dataset='thermal',panel_sha256='p'*64)
    assert failure['status']=='catastrophic_failure'
    assert failure['consecutive_failures']==[{'field':'temperature_fluid','window_endpoints':[3100,3200],'threshold_ratio':2.}]
    del data[3200]
    pending=catastrophic_review(data,dataset='thermal',panel_sha256='p'*64)
    assert pending['status']=='pending' and pending['missing_epochs']==[3200]
    # Exactly2x is outside the strict exceeds2x trigger.
    data=boards()
    for age in (3100,3200):data[age]['C10-H']['metrics']['overall']['fields']['q_proxy']['mean']=2.
    assert catastrophic_review(data,dataset='thermal',panel_sha256='p'*64)['status']=='passed'


def test_DEV_scalar_or_mismatched_receivers_cannot_fill_TRAIN_gate():
    pair=boards()[3100]
    pair['C10-H']['split']='validation'
    with pytest.raises(ValueError,match='full selected-TRAIN'):
        validate_board_pair(pair['C10-H'],pair['D9-M'],dataset='thermal',epoch=3100,panel_sha256='p'*64)
    pair=boards()[3100];pair['C10-H']['query_receipts']['0']='b'*64
    with pytest.raises(ValueError,match='receiver draws'):
        validate_board_pair(pair['C10-H'],pair['D9-M'],dataset='thermal',epoch=3100,panel_sha256='p'*64)


@pytest.mark.parametrize('producer',['round9','round10'])
def test_bound_historical_and_root_round10_provenance_fixture(tmp_path,producer):
    checkpoint=tmp_path/'epoch_3500_model.pt';checkpoint.write_bytes(b'fixed_checkpoint_fixture')
    board=tmp_path/'result.json';board.write_text(json.dumps({'completed_epoch':3500}))
    provenance=tmp_path/('round10_provenance.json' if producer=='round10' else 'provenance.json')
    value=dict(checkpoint=str(checkpoint),checkpoint_sha256=bound(checkpoint)['sha256'])
    value['literal_epoch' if producer=='round10' else 'completed_epoch']=3500
    provenance.write_text(json.dumps(value))
    binding=bound(board)
    assert load_provenance_board(binding,checkpoint=checkpoint)['completed_epoch']==3500
    binding['provenance']=bound(provenance)
    assert load_provenance_board(binding,checkpoint=checkpoint)['completed_epoch']==3500
    index=tmp_path/'index.jsonl'
    register_board(index,epoch=3500,arm='D9-M',result_path=board,checkpoint_sha256=bound(checkpoint)['sha256'],provenance_path=provenance)
    registry=board_registry({'train_board_index':str(index),'train_boards':{}})
    assert registry[3500]['D9-M']['provenance']==bound(provenance)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes/age'):load_provenance_board(binding,checkpoint=checkpoint)


def test_gate_registry_rejects_an_undeclared_line_format(tmp_path):
    index=tmp_path/'index.jsonl';index.write_text(json.dumps({'epoch':3500,'arm':'D9-M','result':{}})+'\n')
    with pytest.raises(ValueError,match='registry contract'):
        board_registry({'train_board_index':str(index)})


def test_root_W6_index_accepts_bound_historical_reuse_without_mutating_old_board(tmp_path):
    checkpoint=tmp_path/'epoch_3500_model.pt';checkpoint.write_bytes(b'exact_old_dense_checkpoint')
    panel=tmp_path/'panel.json';panel.write_text(json.dumps({'panel_sha256':'p'*64}))
    result=tmp_path/'result.json';result.write_text(json.dumps({'completed_epoch':3500,'panel_sha256':'p'*64}))
    row=dict(schema='round10_native_board_index_row_v1',name='thermal_D9M_e3500_TRAIN',dataset='thermal',arm='D9-M',epoch=3500,split='train',
        checkpoint=bound(checkpoint),panel=bound(panel),result=bound(result),provenance=None,historical_reuse=True)
    index=tmp_path/'root_index.jsonl';index.write_text(json.dumps(row)+'\n')
    registry=board_registry({'dataset':'thermal','train_board_index':str(index)})
    assert load_provenance_board(registry[3500]['D9-M'],checkpoint=checkpoint)['completed_epoch']==3500
    assert not (tmp_path/'provenance.json').exists()


def test_automatic_pending_gate_does_not_duplicate_root_W6_queue(tmp_path,monkeypatch):
    import continuous_unified_gate as driver
    from types import SimpleNamespace
    campaign=SimpleNamespace(root=tmp_path,master={'authorized_gpus':{'1':'GPU-fixture'}})
    spec={'dataset':'thermal','board_producer':'root_w6'}
    monkeypatch.setattr(driver,'evaluate_unified_gate',lambda *args:dict(status='pending',missing_conditions=['full_selected_TRAIN_adjacent100_epoch_boards']))
    monkeypatch.setattr(driver,'collect_boards',lambda *args:pytest.fail('Root W6 queue must not be duplicated'))
    result=driver.run_automatic(spec,campaign,1,100.)
    assert result['status']=='pending'
    assert result['missing']==['full_selected_TRAIN_adjacent100_epoch_boards']
