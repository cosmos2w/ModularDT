"""Prospective Round10 unified-triplet admission from exact native evidence."""
from __future__ import annotations
import json
import math
from pathlib import Path
from .continuous_joint_protocol import PARENT_SHA,canonical_sha256,file_sha256

AGES=tuple(range(3100,5001,100))
THERMAL_PRIORITY=('temperature_fluid','temperature_surface','temperature_material','q_proxy','module_peak_endpoint')


def process_time_headroom(limits, operations, other_seconds):
    """Keep the total, non-evaluation and reserved evaluation pools distinct."""
    charged=lambda row:row['reserved_seconds'] if row['status']=='running' else row['actual_seconds']
    total=sum(charged(row) for row in operations)
    non_evaluation=sum(charged(row) for row in operations if row['kind']!='evaluation')
    evaluation=total-non_evaluation
    total_remaining=limits['gpu_associated_seconds']-total-other_seconds
    non_evaluation_remaining=(limits['gpu_associated_seconds']
        -limits['gpu_associated_seconds_reserved_for_evaluation']-non_evaluation-other_seconds)
    evaluation_allowance_ok=evaluation<=limits['gpu_associated_seconds_reserved_for_evaluation']
    return dict(charged_total_seconds=total,charged_non_evaluation_seconds=non_evaluation,
        charged_evaluation_seconds=evaluation,total_remaining_seconds=total_remaining,
        non_evaluation_remaining_seconds=non_evaluation_remaining,
        evaluation_allowance_ok=evaluation_allowance_ok)


def read(path):return json.loads(Path(path).read_text())


def bound(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=file_sha256(path))


def read_bound(binding):
    if file_sha256(binding['path'])!=binding['sha256']:raise ValueError('Unified gate evidence bytes changed.')
    return read(binding['path'])


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
        q=read_bound(binding)
        if (q.get('schema')!='round10_complete_native_qualification_v1' or q.get('arm')!=arm or q.get('dataset')!=dataset
            or q.get('status')!='completed_engineering_epochs' or q.get('completed_native_epochs',0)<1
            or q.get('scientific_training_epochs')!=0 or q.get('source_checkpoint_sha256')!=PARENT_SHA[dataset]
            or not q.get('gpu_uuid') or q.get('exact_inherited_adam_verified') is not True):
            raise ValueError('Unified native feasibility needs complete real-objective GPU epochs with exact inherited Adam.')
        case_count=150 if dataset=='thermal' else 72;updates=4 if dataset=='thermal' else 5
        if any(row['cases']!=case_count or row['optimizer_updates']!=updates for row in q['epoch_records']):
            raise ValueError('Unified qualification omitted native cases or a partial macro.')
        times[arm]=max(row['elapsed_seconds'] for row in q['epoch_records'])
        qualification_sources[arm]=binding
    forecast=None
    if spec.get('other_required_reservations_complete') is not True:
        missing.append('complete_remaining_A_B_resource_obligations')
    if len(times)==3 and spec.get('other_required_reservations_complete') is True:
        ledger=read(campaign.ledger_path) if campaign.ledger_path.exists() else {'operations':{}}
        operations=list(ledger['operations'].values())
        used_updates=sum(row['reserved_updates'] if row['status']=='running' else row['actual_updates'] for row in operations if row['kind']=='main')
        other_seconds=sum(row['seconds'] for row in spec['other_required_reservations'])
        other_updates=sum(row['updates'] for row in spec['other_required_reservations'])
        headroom=process_time_headroom(campaign.limits,operations,other_seconds)
        remaining=min(headroom['total_remaining_seconds'],headroom['non_evaluation_remaining_seconds'])
        proposed=2000*sum(times.values())*spec['forecast_overhead_factor']+spec['validation_checkpoint_io_seconds']+360
        new_updates=2000*(4 if dataset=='thermal' else 5)*3
        if spec['forecast_overhead_factor']<1 or spec['validation_checkpoint_io_seconds']<0:
            raise ValueError('Triplet forecast cannot reduce the measured native epoch cost.')
        storage=campaign.check_storage()
        from shutil import disk_usage
        actual_retained=sum(p.stat().st_size for p in campaign.root.rglob('*') if p.is_file() and not p.is_symlink() and 'tmp' not in p.relative_to(campaign.root).parts)
        disk_ok=(actual_retained+spec['projected_retained_bytes']<=campaign.limits['retained_payload_bytes']
            and disk_usage('/data2').free-spec['projected_retained_bytes']-spec['projected_temporary_bytes']>=campaign.limits['minimum_free_data2_bytes'])
        forecast=dict(native_epoch_seconds=times,complete_triplet_seconds=proposed,reserved_remaining_process_seconds=remaining,
            process_time_headroom=headroom,
            triplet_updates=new_updates,other_required_reservations=spec['other_required_reservations'],
            projected_retained_bytes=spec['projected_retained_bytes'],projected_temporary_bytes=spec['projected_temporary_bytes'],disk_ok=disk_ok,
            process_time_ok=remaining>=proposed and headroom['evaluation_allowance_ok'],wall_time_ok=__import__('time').time()+proposed+other_seconds<=campaign.start+campaign.limits['wall_seconds'],update_budget_ok=used_updates+other_updates+new_updates<=campaign.limits['main_scientific_updates'])
        if not all(forecast[name] for name in ('disk_ok','process_time_ok','wall_time_ok','update_budget_ok')):missing.append('whole_triplet_resource_reservation')
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
