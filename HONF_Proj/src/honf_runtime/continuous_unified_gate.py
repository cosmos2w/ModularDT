"""Prospective Round10 unified-triplet admission from exact native evidence."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
from .continuous_joint_protocol import PARENT_SHA,canonical_sha256,file_sha256

AGES=tuple(range(3100,5001,100))
THERMAL_PRIORITY=('temperature_fluid','temperature_surface','temperature_material','q_proxy','module_peak_endpoint')


def _future_reservation_totals(reservations):
    """Return typed future time/update totals; old rows charge non-evaluation."""
    if isinstance(reservations,(int,float)) and not isinstance(reservations,bool):
        if not math.isfinite(reservations) or reservations<0:
            raise ValueError('Future reservation seconds must be finite and nonnegative.')
        rows=[dict(seconds=float(reservations),updates=0)]
    elif reservations is None:
        rows=[]
    else:
        rows=list(reservations)
    result=dict(total_seconds=0.,non_evaluation_seconds=0.,evaluation_seconds=0.,main_updates=0,engineering_updates=0)
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Future reservations must be typed mappings or legacy seconds.')
        seconds=row.get('seconds',0.)
        updates=row.get('updates',0)
        kind=row.get('kind')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
            raise ValueError('Future reservation seconds must be finite and nonnegative.')
        if type(updates) is not int or updates<0:
            raise ValueError('Future reservation updates must be nonnegative integers.')
        if kind not in (None,'main','engineering','evaluation'):
            raise ValueError('Future reservation kind is not declared.')
        result['total_seconds']+=seconds
        if kind=='evaluation':result['evaluation_seconds']+=seconds
        else:result['non_evaluation_seconds']+=seconds
        # An untyped legacy update count is conservatively charged to main.
        if kind in (None,'main'):result['main_updates']+=updates
        elif kind=='engineering':result['engineering_updates']+=updates
    return result


def process_time_headroom(limits, operations, other_reservations):
    """Keep total, non-evaluation and evaluation reservations separate.

    A scalar retains the old API and is charged as non-evaluation. Typed future
    rows can include evaluation reservations without spending their allowance.
    """
    charged=lambda row:row['reserved_seconds'] if row['status']=='running' else row['actual_seconds']
    total=sum(charged(row) for row in operations)
    non_evaluation=sum(charged(row) for row in operations if row['kind']!='evaluation')
    evaluation=total-non_evaluation
    future=_future_reservation_totals(other_reservations)
    total_remaining=limits['gpu_associated_seconds']-total-future['total_seconds']
    non_evaluation_remaining=(limits['gpu_associated_seconds']
        -limits['gpu_associated_seconds_reserved_for_evaluation']-non_evaluation-future['non_evaluation_seconds'])
    evaluation_remaining=(limits['gpu_associated_seconds_reserved_for_evaluation']
        -evaluation-future['evaluation_seconds'])
    evaluation_allowance_ok=evaluation+future['evaluation_seconds']<=limits['gpu_associated_seconds_reserved_for_evaluation']
    return dict(charged_total_seconds=total,charged_non_evaluation_seconds=non_evaluation,
        charged_evaluation_seconds=evaluation,total_remaining_seconds=total_remaining,
        non_evaluation_remaining_seconds=non_evaluation_remaining,
        reserved_evaluation_remaining_seconds=evaluation_remaining,
        forecast_other_total_seconds=future['total_seconds'],
        forecast_other_non_evaluation_seconds=future['non_evaluation_seconds'],
        forecast_other_evaluation_seconds=future['evaluation_seconds'],
        forecast_other_main_updates=future['main_updates'],
        forecast_other_engineering_updates=future['engineering_updates'],
        evaluation_allowance_ok=evaluation_allowance_ok)


def process_update_headroom(limits,operations,other_reservations,proposed_main_updates):
    """Check scientific and discarded-engineering update pools independently."""
    future=_future_reservation_totals(other_reservations)
    charged=lambda row:row['reserved_updates'] if row['status']=='running' else row['actual_updates']
    main=sum(charged(row) for row in operations if row['kind']=='main')
    engineering=sum(charged(row) for row in operations if row['kind']=='engineering')
    return dict(charged_main_updates=main,charged_engineering_updates=engineering,
        forecast_main_updates=future['main_updates'],forecast_engineering_updates=future['engineering_updates'],
        proposed_main_updates=proposed_main_updates,
        main_update_budget_ok=main+future['main_updates']+proposed_main_updates<=limits['main_scientific_updates'],
        engineering_update_budget_ok=engineering+future['engineering_updates']<=limits['discarded_engineering_updates'])


def storage_projection_headroom(limits,storage,projected_retained_bytes,projected_temporary_bytes,free_bytes):
    """Include the entire projected closeout payload in every storage ceiling."""
    projected=(projected_retained_bytes,projected_temporary_bytes,free_bytes,
        storage.get('retained_bytes'),storage.get('temporary_bytes'))
    if any(type(value) is not int or value<0 for value in projected):
        raise ValueError('Projected and measured storage bytes must be nonnegative integers.')
    retained=storage['retained_bytes']+projected_retained_bytes
    temporary=storage['temporary_bytes']+projected_temporary_bytes
    peak=retained+temporary
    free_after=free_bytes-projected_retained_bytes-projected_temporary_bytes
    retained_ok=retained<=limits['retained_payload_bytes']
    temporary_ok=temporary<=limits['temporary_payload_bytes']
    peak_ok=peak<=limits['peak_new_payload_bytes']
    free_ok=free_after>=limits['minimum_free_data2_bytes']
    return dict(retained_after_bytes=retained,temporary_after_bytes=temporary,
        peak_after_bytes=peak,free_after_bytes=free_after,retained_budget_ok=retained_ok,
        temporary_budget_ok=temporary_ok,peak_budget_ok=peak_ok,free_headroom_ok=free_ok,
        disk_ok=retained_ok and temporary_ok and peak_ok and free_ok)


def read(path):return json.loads(Path(path).read_text())


def bound(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=file_sha256(path))


def read_bound(binding):
    if file_sha256(binding['path'])!=binding['sha256']:raise ValueError('Unified gate evidence bytes changed.')
    return read(binding['path'])


def _bound_receipt(path):
    if isinstance(path,dict):
        if set(path)!={'path','sha256'} or not isinstance(path.get('path'),str) or not isinstance(path.get('sha256'),str):
            raise ValueError('Unified external receipt binding must contain exactly path and sha256.')
        resolved=Path(path['path']).resolve(strict=True)
        actual=file_sha256(resolved)
        if path['sha256']!=actual:
            raise ValueError('Unified external receipt bytes changed.')
        return dict(path=str(resolved),sha256=actual)
    resolved=Path(path).resolve(strict=True)
    return dict(path=str(resolved),sha256=file_sha256(resolved))


def _external_receipt_path(value):
    if isinstance(value,str):return value
    if isinstance(value,os.PathLike):
        path=os.fspath(value)
        return path if isinstance(path,str) else None
    if isinstance(value,dict) and set(value)=={'path','sha256'} and isinstance(value.get('path'),str):
        return value['path']
    return None


def _validate_gpu0_device_authorization(binding,*,campaign,source_manifest,gpu_uuid):
    """Validate the explicit amendment without rewriting the original master."""
    if not isinstance(binding,dict) or set(binding)!={'path','sha256'}:
        raise ValueError('GPU0 qualification needs a hash-bound continuation authorization.')
    amendment=read_bound(binding)
    master_path=campaign.root/'receipts/master_reservation_v1.json'
    if (amendment.get('schema')!='round10_gpu0_continuation_authorization_v2'
        or amendment.get('status')!='authorized_by_user' or amendment.get('root')!=str(campaign.root)
        or amendment.get('physical_gpu0')!=0 or amendment.get('gpu_uuid')!=gpu_uuid
        or amendment.get('authorized_gpus')!={'0':gpu_uuid}
        or amendment.get('cuda_visible_devices')!=gpu_uuid):
        raise ValueError('GPU0 amendment does not authorize this physical index/UUID.')
    basis=amendment.get('authorization_basis',{})
    if any(not isinstance(basis.get(name),str) or not basis[name].strip()
        for name in ('instruction','scope')):
        raise ValueError('GPU0 amendment lacks the recorded user scope.')
    master=amendment.get('master_reference',{})
    if (Path(master.get('path','')).resolve()!=master_path.resolve()
        or master.get('sha256')!=file_sha256(master_path)):
        raise ValueError('GPU0 amendment is not bound to this unchanged campaign master.')
    source=amendment.get('source_manifest',{})
    if (Path(source.get('path','')).resolve()!=Path(source_manifest['path']).resolve()
        or source.get('sha256')!=source_manifest['sha256']):
        raise ValueError('GPU0 amendment source closure differs from the gate source manifest.')
    if set(amendment.get('recipes',{}))!={'thermal_H','wind_H','thermal_S','wind_S'}:
        raise ValueError('GPU0 amendment recipe inventory is incomplete.')
    return amendment


def _validate_qualification_device(*,physical_gpu,gpu_uuid,spec,campaign,source_manifest,launch):
    """Original master devices stay valid; GPU0 needs its bound amendment."""
    key=str(physical_gpu)
    authorized=campaign.master.get('authorized_gpus',{})
    if key in authorized:
        if authorized[key]!=gpu_uuid:raise ValueError('Qualification UUID differs from the original master reservation.')
        return dict(authorization='original_master',physical_gpu=physical_gpu,gpu_uuid=gpu_uuid)
    if physical_gpu!=0:raise ValueError('Qualification device is outside the campaign authorization.')
    binding=spec.get('device_authorization')
    launch_binding=launch.get('device_authorization_binding')
    if binding is not None and launch_binding is not None and binding!=launch_binding:
        raise ValueError('Launch and gate GPU0 amendment bindings differ.')
    binding=binding or launch_binding
    amendment=_validate_gpu0_device_authorization(binding,campaign=campaign,source_manifest=source_manifest,gpu_uuid=gpu_uuid)
    return dict(authorization='gpu0_amendment',physical_gpu=physical_gpu,gpu_uuid=gpu_uuid,
        amendment_sha256=binding['sha256'],source_manifest_sha256=amendment['source_manifest']['sha256'])


def _validate_unified_qualification(spec,campaign,*,arm,dataset,case_count,updates):
    """Validate a complete disposable epoch and its external frozen-v8 binding."""
    qualification_spec=spec['qualification'][arm]
    if isinstance(qualification_spec,dict):
        if not {'path','sha256'}.issubset(qualification_spec):
            raise ValueError('Unified qualification path binding is incomplete.')
        path=Path(qualification_spec['path']).resolve(strict=True)
        q=read_bound(qualification_spec)
        qualification_source=dict(path=str(path),sha256=qualification_spec['sha256'])
    else:
        path=Path(qualification_spec).resolve(strict=True)
        qualification_source=bound(path)
        q=read_bound(qualification_source)
    source_manifest=spec.get('source_manifest')
    strict=spec.get('external_hash_bound_launch_completion_receipts_required') is True
    if strict:
        if (not isinstance(source_manifest,dict) or set(source_manifest)!={'path','sha256'}
            or file_sha256(source_manifest['path'])!=source_manifest['sha256']):
            raise ValueError('Unified qualification source manifest is not byte-bound.')
        external=spec.get('external_qualification_receipts',{}).get(arm)
        if not isinstance(external,dict) or set(external)!={'launch_receipt','completion_receipt'}:
            raise ValueError('Unified qualification needs hash-bound external launch/completion receipts.')
        launch_binding=_bound_receipt(external['launch_receipt'])
        completion_binding=_bound_receipt(external['completion_receipt'])
        launch=read_bound(launch_binding);completion=read_bound(completion_binding)
        recipe_path=Path(spec['recipes'][arm]).resolve(strict=True)
        recipe=read(recipe_path)
        from .continuous_joint_protocol import validate_continuous_joint_recipe
        recipe=validate_continuous_joint_recipe(recipe,verify_source=True)
        recipe_file_sha=file_sha256(recipe_path);recipe_sha=canonical_sha256(recipe)
        if (recipe['arm']!=arm or recipe['dataset']!=dataset
            or recipe.get('input_capability')!='local_records_context_v1'
            or recipe.get('input_feature_contract',{}).get('schema')!='local_records_context_v1'
            or not recipe.get('input_feature_contract',{}).get('sha256')):
            raise ValueError('Unified qualification recipe lacks its exact local-input capability contract.')
        completion_qualification=completion.get('qualification_binding',{})
        completion_launch=completion.get('launch_receipt_binding',{})
        launch_recipe=launch.get('recipe_binding',{})
        source_closure=launch.get('source_closure',{})
        expected_epochs=spec.get('native_qualification_epochs')
        if (completion.get('schema')!='round10_unified_qualification_completion_receipt_v1'
            or completion.get('status')!='completed_engineering_epochs' or completion.get('exit_code')!=0
            or completion.get('source_manifest_sha256')!=source_manifest['sha256']
            or completion.get('recipe_sha256')!=recipe_sha
            or completion.get('resource_operation')!=q.get('operation')
            or completion.get('actual_discarded_optimizer_updates')!=updates
            or Path(completion_qualification.get('path','')).resolve()!=path
            or completion_qualification.get('sha256')!=file_sha256(path)
            or Path(completion_launch.get('path','')).resolve()!=Path(launch_binding['path']).resolve()
            or completion_launch.get('sha256')!=launch_binding['sha256']):
            raise ValueError('External completion does not bind the exact recipe, source, launch and one-epoch output.')
        physical_gpu=launch.get('physical_gpu');gpu_uuid=launch.get('gpu_uuid')
        if (launch.get('schema')!='round10_unified_qualification_launch_receipt_v1'
            or launch.get('resource_operation')!=q.get('operation')
            or Path(launch_recipe.get('path','')).resolve()!=recipe_path
            or launch_recipe.get('sha256')!=recipe_file_sha
            or launch.get('environment',{}).get('CUDA_VISIBLE_DEVICES')!=gpu_uuid
            or Path(source_closure.get('manifest_path','')).resolve()!=Path(source_manifest['path']).resolve()
            or source_closure.get('manifest_sha256')!=source_manifest['sha256']
            or (spec.get('source_root') is not None and source_closure.get('source_root')!=spec['source_root'])):
            raise ValueError('External launch does not bind the exact v8 source, recipe and device.')
        argv=launch.get('exact_argv')
        if not isinstance(argv,list) or not all(isinstance(value,str) for value in argv):
            raise ValueError('External launch command is not retained as an exact argv vector.')
        required_argv=(('--epochs',str(expected_epochs)),('--physical-gpu',str(physical_gpu)),
            ('--recipe',str(recipe_path)),('--source-manifest',str(Path(source_manifest['path']).resolve())),
            ('--source-manifest-sha256',source_manifest['sha256']),
            ('--reserve-seconds',str(spec['qualification_reserve_seconds'])),
            ('--device','cuda:0'),('--output',str(path.parent)))
        for option,value in required_argv:
            if not any(argv[index:index+2]==[option,value] for index in range(len(argv)-1)):
                raise ValueError('External launch argv differs from the one-epoch frozen-source qualification contract.')
        if 'qualify' not in argv:
            raise ValueError('External launch did not invoke native engineering qualification.')
        _validate_qualification_device(physical_gpu=physical_gpu,gpu_uuid=gpu_uuid,spec=spec,
            campaign=campaign,source_manifest=source_manifest,launch=launch)
        ledger=read(campaign.ledger_path) if campaign.ledger_path.exists() else {'operations':{}}
        operation=ledger.get('operations',{}).get(q.get('operation'))
        actual_seconds=operation.get('actual_seconds') if isinstance(operation,dict) else None
        reserved_seconds=operation.get('reserved_seconds') if isinstance(operation,dict) else None
        if (not isinstance(operation,dict) or operation.get('status')!='completed'
            or operation.get('track')!='C' or operation.get('kind')!='engineering'
            or operation.get('physical_gpu')!=physical_gpu or operation.get('gpu_uuid')!=gpu_uuid
            or operation.get('reserved_updates')!=updates or operation.get('actual_updates')!=updates
            or reserved_seconds!=spec['qualification_reserve_seconds']
            or type(actual_seconds) not in (int,float) or not math.isfinite(actual_seconds)
            or actual_seconds<0 or actual_seconds>reserved_seconds
            or operation.get('budget_violation') is True):
            raise ValueError('Qualification lacks its exact completed native engineering ledger operation.')
        if (expected_epochs!=1 or completion.get('actual_discarded_optimizer_updates')!=updates
            or q.get('completed_native_epochs')!=1 or len(q.get('epoch_records',()))!=1):
            raise ValueError('Unified feasibility must be exactly one complete native epoch.')
        receipt_binding=dict(path=completion_binding['path'],sha256=completion_binding['sha256'],
            launch_receipt=launch_binding,qualification=completion_qualification,
            recipe_sha256=recipe_sha,source_manifest_sha256=source_manifest['sha256'])
    else:
        # Older accepted inputs remain valid on original master-reserved devices.
        physical_gpu=q.get('physical_gpu')
        gpu_uuid=q.get('gpu_uuid')
        if physical_gpu is not None:
            _validate_qualification_device(physical_gpu=physical_gpu,gpu_uuid=gpu_uuid,
                spec=spec,campaign=campaign,source_manifest=source_manifest or {},launch={})
        elif gpu_uuid not in campaign.master.get('authorized_gpus',{}).values():
            raise ValueError('Legacy qualification UUID is outside the original master reservation.')
        receipt_binding=qualification_source
    if (q.get('schema')!='round10_complete_native_qualification_v1' or q.get('arm')!=arm or q.get('dataset')!=dataset
        or q.get('status')!='completed_engineering_epochs' or q.get('completed_native_epochs',0)<1
        or q.get('scientific_training_epochs')!=0 or q.get('source_checkpoint_sha256')!=PARENT_SHA[dataset]
        or not q.get('gpu_uuid') or q.get('exact_inherited_adam_verified') is not True):
        raise ValueError('Unified native feasibility needs complete real-objective GPU epochs with exact inherited Adam.')
    if strict and (q.get('physical_gpu')!=physical_gpu or q.get('gpu_uuid')!=gpu_uuid
        or q.get('operation')!=completion.get('resource_operation')):
        raise ValueError('Native epoch payload device/operation differs from its external bound receipts.')
    if strict and (q.get('completed_native_epochs')!=1 or len(q.get('epoch_records',()))!=1):
        raise ValueError('Unified feasibility must contain exactly one complete native epoch.')
    rows=q.get('epoch_records',())
    if (strict and len(rows)!=q.get('completed_native_epochs')) or any(row.get('cases')!=case_count
        or row.get('optimizer_updates')!=updates for row in rows):
        raise ValueError('Unified qualification omitted native cases or a partial macro.')
    if strict and any(type(row.get('elapsed_seconds')) not in (int,float) or not math.isfinite(row['elapsed_seconds'])
        or row['elapsed_seconds']<=0 for row in rows):
        raise ValueError('Unified qualification needs finite positive synchronized epoch timings.')
    if strict:
        row=rows[0]
        macros=row.get('native_macros',())
        if (row.get('epoch')!=3001 or len(macros)!=updates
            or [item.get('macro_index') for item in macros]!=list(range(updates))
            or q.get('discarded_optimizer_updates')!=updates
            or not isinstance(row.get('case_query_sampling_sha256'),str)
            or len(row['case_query_sampling_sha256'])!=64):
            raise ValueError('Unified qualification did not record one complete native epoch and every discarded macro.')
    return max(row['elapsed_seconds'] for row in rows),receipt_binding


def priority_means(board):
    if board['dataset']=='thermal':
        return {name:board['metrics']['overall']['fields'][name]['mean'] for name in THERMAL_PRIORITY}
    return {f'{role}/{name}':values['vector_rmse_mps']['mean'] if name=='vector' else values['component_rmse_mps']['Ux']['mean']
        for role,values in board['metrics']['overall']['roles'].items() for name in ('Ux','vector')}


def validate_board_pair(h,d,*,dataset,epoch,panel_sha256):
    expected=150 if dataset=='thermal' else 72
    for arm,board in (('C10-H',h),('D9-M',d)):
        if (board.get('schema')!='round9_native_physical_board_v1' or board.get('dataset')!=dataset
            or board.get('split')!='train' or board.get('completed_epoch')!=epoch or board.get('arm')!=arm
            or board.get('panel_sha256')!=panel_sha256 or len(board.get('records',()))!=expected
            or len(board.get('query_receipts',{}))!=expected or board.get('optimizer_updates')!=0
            or board.get('new_solver_attempts')!=0 or board.get('WindTEST')!='locked'):
            raise ValueError('Catastrophic review requires exact full selected-TRAIN physical boards at each literal age.')
        count=board['metrics']['overall'].get('case_count' if dataset=='thermal' else 'row_count')
        if count!=expected:raise ValueError('Catastrophic board omitted selected-TRAIN cases.')
    if h['query_receipts']!=d['query_receipts'] or h['TRAIN_role_scales']!=d['TRAIN_role_scales']:
        raise ValueError('Catastrophic board cases, receiver draws or physical TRAIN scales differ.')
    a,b=priority_means(h),priority_means(d)
    if set(a)!=set(b):raise ValueError('Catastrophic priority fields differ.')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (*a.values(),*b.values())):
        raise ValueError('Catastrophic physical means must be finite and nonnegative.')
    return {name:dict(continuous_mean=a[name],dense_mean=b[name],ratio=a[name]/b[name] if b[name]>0 else None,
        dense_zero_denominator=b[name]==0,exceeds_twice=a[name]>2*b[name]) for name in a}


def catastrophic_review(boards,*,dataset,panel_sha256):
    """Same priority field must exceed twice dense at both adjacent endpoints."""
    ratios={};missing=[];failures=[]
    for epoch in AGES:
        pair=boards.get(epoch,{})
        if set(pair)!={'C10-H','D9-M'}:
            missing.append(epoch);continue
        ratios[epoch]=validate_board_pair(pair['C10-H'],pair['D9-M'],dataset=dataset,epoch=epoch,panel_sha256=panel_sha256)
        previous=ratios.get(epoch-100)
        if previous is not None:
            for field,row in ratios[epoch].items():
                if row['exceeds_twice'] and previous[field]['exceeds_twice']:
                    failures.append(dict(field=field,window_endpoints=[epoch-100,epoch],threshold_ratio=2.))
    return dict(status='catastrophic_failure' if failures else 'pending' if missing else 'passed',
        definition='same priority physical mean exceeds twice same-age D9-M at both adjacent saved100-epoch endpoints',
        missing_epochs=missing,ratios=ratios,consecutive_failures=failures)


def register_board(index,*,epoch,arm,result_path,checkpoint_sha256,provenance_path):
    """Append one explicit immutable board/provenance binding to the local JSONL index."""
    import fcntl
    if epoch not in AGES or arm not in ('C10-H','D9-M'):
        raise ValueError('Invalid catastrophic-board registry age/arm.')
    result=bound(result_path);result['provenance']=bound(provenance_path)
    row=dict(schema='round10_catastrophic_board_registry_entry_v1',epoch=epoch,arm=arm,
        result=result,checkpoint_sha256=checkpoint_sha256)
    index=Path(index);index.parent.mkdir(parents=True,exist_ok=True)
    with index.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush()
    return row


def board_registry(spec):
    rows={int(age):dict(pair) for age,pair in spec.get('train_boards',{}).items()}
    index=Path(spec['train_board_index'])
    if index.exists():
        for line in index.read_text().splitlines():
            row=json.loads(line)
            if row.get('schema')=='round10_catastrophic_board_registry_entry_v1':
                if row.get('epoch') not in AGES or row.get('arm') not in ('C10-H','D9-M') or 'provenance' not in row.get('result',{}):
                    raise ValueError('Unknown catastrophic-board registry contract.')
                rows.setdefault(row['epoch'],{})[row['arm']]=row['result']
                continue
            required={'name','dataset','arm','epoch','split','checkpoint','panel','result','provenance','historical_reuse'}
            if row.get('schema')!='round10_native_board_index_row_v1' or not required.issubset(row):raise ValueError('Unknown catastrophic-board registry contract.')
            if row['dataset']!=spec['dataset'] or row['split']!='train' or row['arm'] not in ('C10-H','D9-M'):continue
            if row['epoch'] not in AGES:raise ValueError('Root catastrophic registry contains an undeclared review age.')
            if row['historical_reuse'] and (row['arm']!='D9-M' or row['epoch']!=3500):
                raise ValueError('Historical gate reuse is restricted to the exact dense e3500 board.')
            result=dict(row['result'],provenance=row['provenance'],registry_checkpoint=row['checkpoint'],
                registry_panel=row['panel'],registry_epoch=row['epoch'],historical_reuse=row['historical_reuse'])
            rows.setdefault(row['epoch'],{})[row['arm']]=result
    return rows


def load_provenance_board(binding,*,checkpoint=None):
    """Accept bound historical Round9 or the root's declared Round10 producer."""
    result=read_bound(binding)
    registered=binding.get('registry_checkpoint')
    if registered is not None:
        if file_sha256(registered['path'])!=registered['sha256']:
            raise ValueError('Registered catastrophic checkpoint bytes changed.')
        panel=read_bound(binding['registry_panel'])
        if panel.get('panel_sha256')!=result.get('panel_sha256'):
            raise ValueError('Registered catastrophic panel differs from its board.')
    if binding.get('provenance') is not None:
        provenance=read_bound(binding['provenance'])
    elif binding.get('historical_reuse') is True and registered is not None:
        provenance=dict(checkpoint=registered['path'],checkpoint_sha256=registered['sha256'],
                        literal_epoch=binding['registry_epoch'])
    else:
        directory=Path(binding['path']).parent
        path=directory/'round10_provenance.json'
        if not path.is_file():path=directory/'provenance.json'
        provenance=read(path)
    source=Path(provenance['checkpoint'])
    epoch=provenance.get('literal_epoch',provenance.get('completed_epoch'))
    if checkpoint is not None and source.resolve()!=Path(checkpoint).resolve():
        raise ValueError('Catastrophic board source differs from the checkpoint index.')
    if file_sha256(source)!=provenance['checkpoint_sha256'] or epoch!=result['completed_epoch']:
        raise ValueError('Catastrophic board checkpoint bytes/age changed.')
    return result


def evaluate_unified_gate(spec,campaign):
    """Read-only decision; missing physical or resource evidence stays pending."""
    from .continuous_joint_protocol import _source,CONTINUOUS_FAMILY
    dataset=spec['dataset'];panel=read_bound(spec['panel']);missing=[];sources={}
    if dataset not in ('thermal','wind') or panel.get('dataset')!=dataset:
        raise ValueError('Unified gate dataset/panel differs.')
    endpoint=Path(spec['track_b_endpoint_checkpoint'])
    endpoint_hash=None
    if not endpoint.is_file():missing.append('literal_C10_H_e5000_endpoint')
    else:
        saved=_source(endpoint);identity=saved.get('experiment_identity',{});recipe=identity.get('recipe',{})
        if (saved.get('epoch')!=5000 or identity.get('family')!=CONTINUOUS_FAMILY
            or recipe.get('arm')!='C10-H' or recipe.get('dataset')!=dataset
            or recipe.get('source_checkpoint_sha256')!=PARENT_SHA[dataset]):
            raise ValueError('Unified gate source is not the matched literal C-H endpoint.')
        endpoint_hash=file_sha256(endpoint)
    checkpoint_index=read_bound(spec['dense_checkpoint_index'])
    dense={row['epoch']:row['path'] for row in checkpoint_index['rows'] if row['dataset']==dataset}
    registry=board_registry(spec);boards={}
    for epoch,pair in registry.items():
        for arm,binding in pair.items():
            expected=(dense[epoch] if arm=='D9-M' else str(Path(spec['track_b_run'])/'checkpoints'/f'epoch_{epoch}_model.pt'))
            boards.setdefault(epoch,{})[arm]=load_provenance_board(binding,checkpoint=expected)
    review=catastrophic_review(boards,dataset=dataset,panel_sha256=panel['panel_sha256'])
    if review['missing_epochs']:missing.append('full_selected_TRAIN_adjacent100_epoch_boards')
    safety=spec.get('software_qualification')
    categories=('source_target_isolation','continuous_value_gradient','native_law','fresh_state','all_open_equations')
    if safety is None:missing.append('software_qualification')
    else:
        value=read_bound(safety);sources['software_qualification']=safety
        if value.get('schema')!='round10_unified_software_qualification_v1' or any(value.get(name) is not True for name in categories):
            raise ValueError('Unified source/native/equation safety qualification did not pass.')
    times={};qualification_sources={}
    for arm in ('U10-D','U10-H','U10-S'):
        binding=spec.get('qualification',{}).get(arm)
        if isinstance(binding,str):binding=bound(binding) if Path(binding).is_file() else {'path':binding,'sha256':None}
        if binding is None or not Path(binding['path']).is_file():missing.append(arm+'_complete_native_GPU_epoch');continue
        if spec.get('external_hash_bound_launch_completion_receipts_required') is True:
            receipts=spec.get('external_qualification_receipts',{}).get(arm,{})
            required=('launch_receipt','completion_receipt')
            if any((receipt_path:=_external_receipt_path(receipts.get(name))) is None
                   or not Path(receipt_path).is_file() for name in required):
                missing.append(arm+'_hash_bound_external_qualification_receipts');continue
        case_count=150 if dataset=='thermal' else 72;updates=4 if dataset=='thermal' else 5
        times[arm],qualification_sources[arm]=_validate_unified_qualification(spec,campaign,arm=arm,
            dataset=dataset,case_count=case_count,updates=updates)
    forecast=None
    if spec.get('other_required_reservations_complete') is not True:
        missing.append('complete_remaining_A_B_resource_obligations')
    if len(times)==3 and spec.get('other_required_reservations_complete') is True:
        ledger=read(campaign.ledger_path) if campaign.ledger_path.exists() else {'operations':{}}
        operations=list(ledger['operations'].values())
        other_reservations=spec['other_required_reservations']
        new_updates=2000*(4 if dataset=='thermal' else 5)*3
        headroom=process_time_headroom(campaign.limits,operations,other_reservations)
        update_headroom=process_update_headroom(campaign.limits,operations,other_reservations,new_updates)
        other_seconds=headroom['forecast_other_total_seconds']
        remaining=min(headroom['total_remaining_seconds'],headroom['non_evaluation_remaining_seconds'])
        proposed=2000*sum(times.values())*spec['forecast_overhead_factor']+spec['validation_checkpoint_io_seconds']+360
        if spec['forecast_overhead_factor']<1 or spec['validation_checkpoint_io_seconds']<0:
            raise ValueError('Triplet forecast cannot reduce the measured native epoch cost.')
        storage=campaign.check_storage()
        from shutil import disk_usage
        storage_projection=storage_projection_headroom(campaign.limits,storage,
            spec['projected_retained_bytes'],spec['projected_temporary_bytes'],disk_usage('/data2').free)
        disk_ok=storage_projection['disk_ok']
        forecast=dict(native_epoch_seconds=times,complete_triplet_seconds=proposed,reserved_remaining_process_seconds=remaining,
            process_time_headroom=headroom,
            process_update_headroom=update_headroom,
            triplet_updates=new_updates,other_required_reservations=other_reservations,
            projected_retained_bytes=spec['projected_retained_bytes'],projected_temporary_bytes=spec['projected_temporary_bytes'],
            storage_projection=storage_projection,disk_ok=disk_ok,
            process_time_ok=remaining>=proposed and headroom['evaluation_allowance_ok'],
            wall_time_ok=__import__('time').time()+proposed+other_seconds<=campaign.start+campaign.limits['wall_seconds'],
            update_budget_ok=update_headroom['main_update_budget_ok'],
            engineering_update_budget_ok=update_headroom['engineering_update_budget_ok'])
        if not all(forecast[name] for name in ('disk_ok','process_time_ok','wall_time_ok','update_budget_ok','engineering_update_budget_ok')):missing.append('whole_triplet_resource_reservation')
    status='not_admitted_catastrophic_failure' if review['consecutive_failures'] else 'pending' if missing else 'admitted'
    result=dict(schema='round10_unified_triplet_admission_v1',dataset=dataset,status=status,track_b_epoch=5000,
        track_b_endpoint_checkpoint=str(endpoint),track_b_endpoint_checkpoint_sha256=endpoint_hash,
        source_checkpoint_sha256=PARENT_SHA[dataset],resource_reservation_sha256=file_sha256(spec['resource_reservation']),
        source_target_isolation_passed=safety is not None,continuous_value_gradient_passed=safety is not None,
        native_law_passed=safety is not None,fresh_state_passed=safety is not None,
        track_b_catastrophic_collapse=bool(review['consecutive_failures']),catastrophic_review=review,
        complete_triplet_reserved=status=='admitted',missing_conditions=missing,qualification_sources=qualification_sources,
        software_qualification=safety,diagnostic_limitations=spec.get('diagnostic_limitations',[]))
    if forecast is not None:result.update(forecast)
    return result
