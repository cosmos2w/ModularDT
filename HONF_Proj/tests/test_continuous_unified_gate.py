"""Exact all-TRAIN catastrophic-window policy and accepted provenance contracts."""
import copy
import json
from pathlib import Path
import pytest
from honf_runtime.continuous_unified_gate import (AGES,THERMAL_PRIORITY,catastrophic_review,
    validate_board_pair,bound,load_provenance_board,register_board,board_registry,process_time_headroom,
    _validate_unified_qualification,storage_projection_headroom)
from honf_runtime.continuous_unified_gate import process_update_headroom
from honf_runtime.continuous_joint_protocol import PARENT_SHA,canonical_sha256,file_sha256


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


def test_typed_future_reservations_charge_evaluation_only_to_its_protected_pool():
    hour=3600
    limits={'gpu_associated_seconds':72*hour,
            'gpu_associated_seconds_reserved_for_evaluation':8*hour}
    rows=[{'kind':'main','status':'completed','actual_seconds':20*hour},
          {'kind':'evaluation','status':'completed','actual_seconds':3*hour}]
    future=[{'kind':'evaluation','seconds':5*hour,'updates':0},
            {'kind':'engineering','seconds':2*hour,'updates':400}]
    result=process_time_headroom(limits,rows,future)
    assert result['total_remaining_seconds']==42*hour
    assert result['non_evaluation_remaining_seconds']==42*hour
    assert result['reserved_evaluation_remaining_seconds']==0
    assert result['evaluation_allowance_ok']
    assert result['forecast_other_main_updates']==0
    assert result['forecast_other_engineering_updates']==400
    over=process_time_headroom(limits,rows,[*future,{'kind':'evaluation','seconds':1,'updates':0}])
    assert not over['evaluation_allowance_ok']
    assert over['reserved_evaluation_remaining_seconds']==-1


def test_legacy_future_reservation_defaults_to_non_evaluation_and_main_updates():
    limits={'gpu_associated_seconds':72*3600,
            'gpu_associated_seconds_reserved_for_evaluation':8*3600}
    result=process_time_headroom(limits,[],[{'seconds':120,'updates':7}])
    assert result['forecast_other_non_evaluation_seconds']==120
    assert result['forecast_other_evaluation_seconds']==0
    assert result['forecast_other_main_updates']==7


def test_future_engineering_updates_remain_in_their_own_budget():
    limits={'main_scientific_updates':100,'discarded_engineering_updates':12}
    operations=[{'kind':'engineering','status':'completed','actual_updates':10},
                {'kind':'main','status':'running','reserved_updates':20}]
    headroom=process_update_headroom(limits,operations,
        [{'kind':'engineering','seconds':10,'updates':2},{'kind':'main','seconds':10,'updates':5}],proposed_main_updates=70)
    assert headroom['main_update_budget_ok']
    assert headroom['engineering_update_budget_ok']
    over=process_update_headroom(limits,operations,
        [{'kind':'engineering','seconds':10,'updates':3}],proposed_main_updates=81)
    assert not over['main_update_budget_ok']
    assert not over['engineering_update_budget_ok']


def test_triplet_storage_forecast_enforces_temporary_and_peak_caps():
    limits={'retained_payload_bytes':10,'temporary_payload_bytes':10,'peak_new_payload_bytes':20,
        'minimum_free_data2_bytes':100}
    temporary_over=storage_projection_headroom(limits,{'retained_bytes':0,'temporary_bytes':3},0,8,200)
    assert not temporary_over['temporary_budget_ok'] and not temporary_over['disk_ok']
    peak_limits=dict(limits,peak_new_payload_bytes=10)
    peak_over=storage_projection_headroom(peak_limits,{'retained_bytes':7,'temporary_bytes':2},1,1,200)
    assert peak_over['retained_budget_ok'] and peak_over['temporary_budget_ok']
    assert not peak_over['peak_budget_ok'] and not peak_over['disk_ok']
    with pytest.raises(ValueError,match='nonnegative integers'):
        storage_projection_headroom(limits,{'retained_bytes':0,'temporary_bytes':0},-1,0,200)


def qualification_fixture(tmp_path,monkeypatch,*,physical_gpu=1):
    from types import SimpleNamespace
    import honf_runtime.continuous_joint_protocol as protocol
    monkeypatch.setattr(protocol,'validate_continuous_joint_recipe',lambda recipe,verify_source=False:recipe)
    root=tmp_path/'campaign';(root/'receipts').mkdir(parents=True)
    master_path=root/'receipts/master_reservation_v1.json';master_path.write_text('{"schema":"master"}\n')
    source=tmp_path/'source_manifest.json';source.write_text('{"files":[]}\n')
    source_binding={'path':str(source.resolve()),'sha256':file_sha256(source)}
    recipe_path=tmp_path/'recipe.json'
    recipe=dict(schema='round10_unified_joint_recipe_v1',family='unified_continuous_joint_native_v1',
        protocol='round10_unified_joint_v1',arm='U10-H',dataset='thermal',source_checkpoint_sha256=PARENT_SHA['thermal'],
        input_capability='local_records_context_v1',input_feature_contract={'schema':'local_records_context_v1','sha256':'c'*64})
    recipe_path.write_text(json.dumps(recipe,sort_keys=True))
    operation='continuous_thermal_U10-H_fixture';uuid='GPU-fixture-'+str(physical_gpu)
    q_path=tmp_path/'qualification.json'
    q=dict(schema='round10_complete_native_qualification_v1',status='completed_engineering_epochs',dataset='thermal',arm='U10-H',
        source_checkpoint_sha256=PARENT_SHA['thermal'],scientific_training_epochs=0,completed_native_epochs=1,
        discarded_optimizer_updates=4,exact_inherited_adam_verified=True,gpu_uuid=uuid,physical_gpu=physical_gpu,operation=operation,
        epoch_records=[dict(epoch=3001,cases=150,optimizer_updates=4,elapsed_seconds=1.25,
            case_query_sampling_sha256='d'*64,native_macros=[dict(macro_index=index) for index in range(4)])])
    q_path.write_text(json.dumps(q,sort_keys=True))
    launch_path=tmp_path/'launch.json';completion_path=tmp_path/'completion.json'
    launch=dict(schema='round10_unified_qualification_launch_receipt_v1',resource_operation=operation,
        physical_gpu=physical_gpu,gpu_uuid=uuid,environment={'CUDA_VISIBLE_DEVICES':uuid},
        exact_argv=['python','continuous_joint_train.py','qualify','--recipe',str(recipe_path.resolve()),'--output',str(tmp_path.resolve()),
            '--epochs','1','--physical-gpu',str(physical_gpu),'--reserve-seconds','600','--device','cuda:0',
            '--source-manifest',source_binding['path'],'--source-manifest-sha256',source_binding['sha256']],
        recipe_binding={'path':str(recipe_path.resolve()),'sha256':file_sha256(recipe_path)},
        source_closure={'manifest_path':source_binding['path'],'manifest_sha256':source_binding['sha256'],'source_root':'frozen-v8'})
    if physical_gpu==0:
        sidecar_path=root/'receipts/gpu0_continuation_authorization_v2.json'
        sidecar=dict(schema='round10_gpu0_continuation_authorization_v2',status='authorized_by_user',root=str(root.resolve()),
            authorization_basis={'instruction':'User explicitly authorized continuing Round 10 on physical GPU 0.',
                'scope':'Only the four Track-B recipe/source/checkpoint bindings listed in recipes are covered.'},
            authorized_gpus={'0':uuid},physical_gpu0=0,gpu_uuid=uuid,cuda_visible_devices=uuid,
            master_reference={'path':str(master_path.resolve()),'sha256':file_sha256(master_path)},
            source_manifest=source_binding,recipes={name:{} for name in ('thermal_H','wind_H','thermal_S','wind_S')})
        sidecar_path.write_text(json.dumps(sidecar,sort_keys=True))
        launch['device_authorization_binding']={'path':str(sidecar_path.resolve()),'sha256':file_sha256(sidecar_path)}
    launch_path.write_text(json.dumps(launch,sort_keys=True))
    completion=dict(schema='round10_unified_qualification_completion_receipt_v1',status='completed_engineering_epochs',exit_code=0,
        actual_discarded_optimizer_updates=4,source_manifest_sha256=source_binding['sha256'],
        recipe_sha256=canonical_sha256(recipe),resource_operation=operation,
        launch_receipt_binding={'path':str(launch_path.resolve()),'sha256':file_sha256(launch_path)},
        qualification_binding={'path':str(q_path.resolve()),'sha256':file_sha256(q_path)})
    completion_path.write_text(json.dumps(completion,sort_keys=True))
    authorized={'1':uuid} if physical_gpu==1 else {'1':'GPU-original'}
    campaign=SimpleNamespace(root=root.resolve(),master={'authorized_gpus':authorized},
        ledger_path=root/'receipts/execution_ledger_v1.json')
    campaign.ledger_path.write_text(json.dumps({'operations':{operation:dict(status='completed',track='C',kind='engineering',
        physical_gpu=physical_gpu,gpu_uuid=uuid,reserved_updates=4,actual_updates=4,reserved_seconds=600.,actual_seconds=1.5)}}))
    spec=dict(qualification={'U10-H':str(q_path)},source_manifest=source_binding,source_root='frozen-v8',
        external_hash_bound_launch_completion_receipts_required=True,
        external_qualification_receipts={'U10-H':{'launch_receipt':str(launch_path),'completion_receipt':str(completion_path)}},
        recipes={'U10-H':str(recipe_path)},native_qualification_epochs=1,qualification_reserve_seconds=600)
    return spec,campaign


@pytest.mark.parametrize('physical_gpu',[1,0])
def test_strict_native_qualification_binds_source_recipe_epoch_and_authorized_device(tmp_path,monkeypatch,physical_gpu):
    spec,campaign=qualification_fixture(tmp_path,monkeypatch,physical_gpu=physical_gpu)
    seconds,binding=_validate_unified_qualification(spec,campaign,arm='U10-H',dataset='thermal',case_count=150,updates=4)
    assert seconds==1.25
    assert binding['source_manifest_sha256']==spec['source_manifest']['sha256']
    assert len(binding['recipe_sha256'])==64


def test_strict_native_qualification_rejects_two_epochs_even_when_each_is_complete(tmp_path,monkeypatch):
    spec,campaign=qualification_fixture(tmp_path,monkeypatch)
    path=Path(spec['qualification']['U10-H']);value=json.loads(path.read_text())
    value['completed_native_epochs']=2;value['epoch_records'].append(dict(value['epoch_records'][0]))
    value['discarded_optimizer_updates']=8;path.write_text(json.dumps(value,sort_keys=True))
    completion_path=Path(spec['external_qualification_receipts']['U10-H']['completion_receipt'])
    completion=json.loads(completion_path.read_text())
    completion['qualification_binding']['sha256']=file_sha256(path)
    completion_path.write_text(json.dumps(completion,sort_keys=True))
    with pytest.raises(ValueError,match='exactly one complete native epoch'):
        _validate_unified_qualification(spec,campaign,arm='U10-H',dataset='thermal',case_count=150,updates=4)


def reseal_launch(spec,mutate):
    launch_path=Path(spec['external_qualification_receipts']['U10-H']['launch_receipt'])
    completion_path=Path(spec['external_qualification_receipts']['U10-H']['completion_receipt'])
    launch=json.loads(launch_path.read_text());mutate(launch);launch_path.write_text(json.dumps(launch,sort_keys=True))
    completion=json.loads(completion_path.read_text())
    completion['launch_receipt_binding']['sha256']=file_sha256(launch_path)
    completion_path.write_text(json.dumps(completion,sort_keys=True))


def test_strict_native_qualification_accepts_and_verifies_existing_bound_path_shape(tmp_path,monkeypatch):
    spec,campaign=qualification_fixture(tmp_path,monkeypatch)
    q_path=Path(spec['qualification']['U10-H'])
    spec['qualification']['U10-H']=bound(q_path)
    seconds,_=_validate_unified_qualification(spec,campaign,arm='U10-H',dataset='thermal',case_count=150,updates=4)
    assert seconds==1.25
    spec['qualification']['U10-H']['sha256']='0'*64
    with pytest.raises(ValueError,match='bytes changed'):
        _validate_unified_qualification(spec,campaign,arm='U10-H',dataset='thermal',case_count=150,updates=4)


def test_strict_native_qualification_rejects_recipe_capability_or_source_closure_mismatch(tmp_path,monkeypatch):
    spec,campaign=qualification_fixture(tmp_path,monkeypatch)
    recipe_path=Path(spec['recipes']['U10-H'])
    recipe=json.loads(recipe_path.read_text());recipe['input_capability']='global_records'
    recipe_path.write_text(json.dumps(recipe,sort_keys=True))
    def reseal_recipe(launch):launch['recipe_binding']['sha256']=file_sha256(recipe_path)
    reseal_launch(spec,reseal_recipe)
    completion_path=Path(spec['external_qualification_receipts']['U10-H']['completion_receipt'])
    completion=json.loads(completion_path.read_text());completion['recipe_sha256']=canonical_sha256(recipe)
    completion_path.write_text(json.dumps(completion,sort_keys=True))
    with pytest.raises(ValueError,match='local-input capability'):
        _validate_unified_qualification(spec,campaign,arm='U10-H',dataset='thermal',case_count=150,updates=4)

    spec,campaign=qualification_fixture(tmp_path/'closure',monkeypatch)
    reseal_launch(spec,lambda launch:launch['source_closure'].__setitem__('manifest_sha256','f'*64))
    with pytest.raises(ValueError,match='exact v8 source'):
        _validate_unified_qualification(spec,campaign,arm='U10-H',dataset='thermal',case_count=150,updates=4)


def test_strict_native_qualification_rejects_unapproved_device_and_wrong_ledger_reserve(tmp_path,monkeypatch):
    spec,campaign=qualification_fixture(tmp_path,monkeypatch)
    def change_uuid(launch):
        launch['gpu_uuid']='GPU-not-authorized';launch['environment']['CUDA_VISIBLE_DEVICES']='GPU-not-authorized'
    reseal_launch(spec,change_uuid)
    with pytest.raises(ValueError,match='UUID differs'):
        _validate_unified_qualification(spec,campaign,arm='U10-H',dataset='thermal',case_count=150,updates=4)

    spec,campaign=qualification_fixture(tmp_path/'ledger',monkeypatch)
    ledger=json.loads(campaign.ledger_path.read_text())
    ledger['operations']['continuous_thermal_U10-H_fixture']['reserved_seconds']=599.
    campaign.ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError,match='ledger operation'):
        _validate_unified_qualification(spec,campaign,arm='U10-H',dataset='thermal',case_count=150,updates=4)


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
