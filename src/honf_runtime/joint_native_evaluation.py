"""Round9 physical boards on frozen original native receivers and exposed cohorts.

Five-role Wind diagnostics remain separate from the inherited two-region loss.
Only input-selected representative arrays are retained; all selected cases keep
compact per-field metrics, tails, strata and literal checkpoint provenance.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


PANEL_SCHEMA = "round9_fixed_native_physical_panel_v1"
BOARD_SCHEMA = "round9_native_physical_board_v1"
WIND_ROLE_COUNTS = (1640,1640,1640,1640,1632)
THERMAL_FIELDS = ("u","v","p","omega","temperature_fluid","temperature_surface",
                  "temperature_material","q_proxy","module_peak_endpoint")


def checkpoint_arm(payload):
    """Resolve the literal arm across inherited and continuous identities."""
    identity = payload.get('experiment_identity')
    if not isinstance(identity, Mapping):
        raise ValueError('Native checkpoint lacks experiment identity.')
    treatment = identity.get('treatment')
    recipe = identity.get('recipe')
    continuous = recipe.get('arm') if isinstance(recipe, Mapping) else None
    if treatment is not None and continuous is not None and treatment != continuous:
        raise ValueError('Native checkpoint has conflicting arm identities.')
    arm = treatment if treatment is not None else continuous
    if not isinstance(arm, str) or not arm:
        raise ValueError('Native checkpoint has no literal arm.')
    return arm


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def array_sha256(*values):
    digest=hashlib.sha256()
    for value in values:
        a=np.ascontiguousarray(array(value));digest.update(repr((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def write_json_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def write_arrays(path,values,*,budget_bytes=512*1024**2):
    """Bound retained evidence before writing; no temporary dataset or cache."""
    path=Path(path)
    if path.exists():raise FileExistsError('Native numerical evidence is immutable.')
    actual=sum(item.stat().st_size for item in path.parent.rglob('*') if item.is_file())
    projected=sum(np.asarray(value).nbytes for value in values.values())+4096*len(values)
    if actual+projected>budget_bytes:raise OSError('This physical-evidence invocation would exceed its512MiB hard budget.')
    np.savez_compressed(path,**values)


def select_input_representatives(counts,limit=4):
    """Minimum, next, median and maximum M; stable input-ID tie breaks."""
    if not counts or not 1<=limit<=4:raise ValueError('A representative panel needs input counts and limit1..4.')
    strata=sorted(set(counts.values()))
    indices=(0,min(1,len(strata)-1),len(strata)//2,len(strata)-1)
    chosen=[]
    for index in indices:
        candidates=sorted((key for key,value in counts.items() if value==strata[index]),key=lambda key:str(key))
        if candidates[0] not in chosen:chosen.append(candidates[0])
    for key in sorted(counts,key=lambda key:str(key)):
        if len(chosen)>=limit:break
        if key not in chosen:chosen.append(key)
    return chosen[:limit]


def input_catalogue(provider,dataset):
    if dataset=='thermal':
        train=list(map(str,provider.training_case_ids));validation=list(map(str,provider.validation_case_ids))
        counts={str(case['case_id']):int(np.asarray(case['structure']['module_present']).sum())
                for cases in (provider.training_cases,provider.validation_cases) for case in cases}
    elif dataset=='wind':
        train=list(map(int,provider.train_rows));validation=list(map(int,provider.validation_rows))
        counts={row:int(provider.view.metadata['n_turbines'][row]) for row in train+validation}
    else:raise ValueError('Only Thermal/Wind source cohorts are permitted.')
    if set(train)&set(validation):raise ValueError('Native physical cohorts must remain disjoint.')
    return {'train':train,'validation':validation,
            'module_counts':{str(key):counts[key] for key in train+validation}}


def freeze_native_panel(provider,dataset,*,source_checkpoint,source_checkpoint_sha256=None):
    catalogue=input_catalogue(provider,dataset)
    representatives={split:select_input_representatives({key:catalogue['module_counts'][str(key)] for key in catalogue[split]})
                     for split in ('train','validation')}
    identity=provider.identity_payload()
    value={'schema':PANEL_SCHEMA,'dataset':dataset,'source_checkpoint':str(Path(source_checkpoint).resolve()),
        'source_checkpoint_sha256':source_checkpoint_sha256 or file_sha256(source_checkpoint),
        'catalogue':catalogue,'catalogue_sha256':canonical_sha256(catalogue),'representatives':representatives,
        'normalization_sha256':identity.get('normalization_stats_sha256',identity.get('normalizer_sha256')),
        'selection':'input module-count minimum/next/median/maximum strata and stable native ID; no prediction or error inspected',
        'selection_reads_prediction_or_target_values':False,
        'physical_receiver_policy':('all stored native Thermal fluid8192/surface64/full-material records'
          if dataset=='thermal' else 'maintained five-role native Q8192 (1640,1640,1640,1640,1632); native flat IDs and case_epoch_v1 source streams'),
        'scientific_monitor_policy_unchanged':True,'WindTEST':'locked','new_solver_attempts':0}
    value['panel_sha256']=canonical_sha256(value)
    return value


def validate_native_panel(panel,provider,dataset):
    if panel.get('schema')!=PANEL_SCHEMA or panel.get('dataset')!=dataset:raise ValueError('Native panel schema/dataset mismatch.')
    body={key:value for key,value in panel.items() if key!='panel_sha256'}
    if canonical_sha256(body)!=panel.get('panel_sha256'):raise ValueError('Frozen native panel seal changed.')
    if canonical_sha256(input_catalogue(provider,dataset))!=panel.get('catalogue_sha256'):
        raise ValueError('Frozen native panel membership or input strata changed.')
    identity=provider.identity_payload()
    current=identity.get('normalization_stats_sha256',identity.get('normalizer_sha256'))
    if current!=panel['normalization_sha256']:raise ValueError('Frozen TRAIN normalizer changed.')
    return panel


def metric(prediction,target,valid):
    p,t=np.asarray(prediction,dtype=np.float64),np.asarray(target,dtype=np.float64)
    if p.shape!=t.shape:raise ValueError('Native prediction/target axes differ.')
    mask=np.broadcast_to(np.asarray(valid,dtype=bool),t.shape)&np.isfinite(t)
    if not mask.any() or not np.isfinite(p[mask]).all():raise ValueError('Physical metrics require finite nonempty native support.')
    delta=(p-t)[mask];truth=t[mask];norm=float(np.linalg.norm(truth))
    return {'rmse':float(np.sqrt(np.mean(delta**2))),'mae':float(np.mean(np.abs(delta))),
            'bias':float(np.mean(delta)),'max_abs':float(np.max(np.abs(delta))),
            'relative_l2':float(np.linalg.norm(delta)/norm) if norm>0 else None,'valid_entries':int(mask.sum())}


def summarize(values,ids):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Metric summary requires finite case values.')
    worst=int(a.argmax())
    return {'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),
            'worst':float(a[worst]),'worst_case_id':ids[worst],'case_count':len(a)}


def thermal_record_and_arrays(predictions,batch):
    if len(batch.case_keys)!=1:raise ValueError('Physical arrays and peak joins require one native case per batch.')
    output=predictions.native_main;prepared=predictions.native_prepared;target=batch.targets
    case=str(batch.case_keys[0]);present=array(prepared.source_present[0])>0
    pred=array(output['pred_field'][0]);truth=array(target.field_targets[0]);fluid=array(prepared.stencils['fluid'].valid[0])&(array(target.point_weights[0])>0)
    interface=array(output['pred_interface'][0]);interface_target=array(target.interface_target[0])
    surface=array(prepared.stencils['surface'].valid[0]).reshape(interface.shape[:-1])&present[:,None]&(array(target.interface_valid_mask[0])>.5)
    outside=array(prepared.stencils['outside'].valid[0]).reshape(interface.shape[:-1])
    material=array(output['pred_internal_temperature'][0,...,0]);material_target=array(target.material_targets[0])
    material_valid=array(prepared.stencils['material'].valid[0]).reshape(material_target.shape)&present[:,None]&np.isfinite(material_target)
    fields={name:metric(pred[:,channel],truth[:,channel],fluid) for channel,name in enumerate(THERMAL_FIELDS[:5])}
    fields['temperature_surface']=metric(interface[...,0],interface_target[...,0],surface)
    fields['q_proxy']=metric(interface[...,1],interface_target[...,1],surface&outside)
    fields['temperature_material']=metric(material,material_target,material_valid)
    live=present&material_valid.any(-1)
    predicted_peaks=np.where(material_valid,material,-np.inf).max(-1)[live]
    actual_peaks=np.where(material_valid,material_target,-np.inf).max(-1)[live]
    fields['module_peak_endpoint']=metric(predicted_peaks,actual_peaks,np.ones_like(actual_peaks,dtype=bool))
    ids=array(prepared.scene_batch.module_ids[0])[live]
    record={'case_id':case,'module_count':int(present.sum()),'fields':fields,
        'module_peaks':[{'native_module_id':int(native),'prediction':float(p),'target':float(t),'error':float(p-t)}
                        for native,p,t in zip(ids,predicted_peaks,actual_peaks)]}
    arrays={'fluid_xy':array(prepared.fluid_xy[0]),'field_prediction':pred,'field_target':truth,
        'field_error':pred-truth,'fluid_valid':fluid,'interface_prediction':interface,'interface_target':interface_target,
        'interface_error':interface-interface_target,'interface_valid':surface,'outside_valid':outside,
        'material_prediction':material,'material_target':material_target,'material_error':material-material_target,
        'material_valid':material_valid,'material_local_xy':array(batch.receivers.local_query_points[0]),
        'module_present':present,'module_ids':array(prepared.scene_batch.module_ids[0]),
        'module_coordinates':array(prepared.scene_batch.module_coordinates[0]),
        'environment_coordinates':array(prepared.scene_batch.environment_coordinates[0]),
        'fluid_native_ids':np.arange(len(pred),dtype=np.int64),
        'native_neural_union_ids':array(prepared.query_batch.query_ids[0]),
        'native_neural_union_xy':array(prepared.query_batch.coordinates[0]),
        'receiver_coordinate_export_policy':np.asarray('fluid and neural union xy are direct prepared receivers; surface/material xy are not retained by the native prepared object, so only original theta/local definitions and geometry are exported'),
        'module_characteristic_lengths':array(prepared.scene_batch.module_characteristic_lengths[0]),
        'surface_theta':np.arange(interface.shape[1],dtype=np.float64)*2*np.pi/interface.shape[1],
        'interface_native_module_ids':np.broadcast_to(array(prepared.scene_batch.module_ids[0])[:,None],interface.shape[:-1]),
        'material_native_module_ids':np.broadcast_to(array(prepared.scene_batch.module_ids[0])[:,None],material.shape),
        'units_json':np.asarray(json.dumps({'coordinates':'dataset native xy','field_channels':
            ['dataset velocity','dataset velocity','dataset pressure','dataset vorticity','dataset temperature'],
            'interface_channels':['dataset temperature','dataset heat-flux proxy'],'material':'dataset temperature'})),
        'predicted_module_peaks':predicted_peaks,'target_module_peaks':actual_peaks}
    return record,arrays


def reduce_thermal_records(rows):
    if not rows or len({row['case_id'] for row in rows})!=len(rows):raise ValueError('Thermal board cases must be nonempty and unique.')
    def group(items):
        ids=[row['case_id'] for row in items]
        return {'case_count':len(items),'fields':{field:summarize([row['fields'][field]['rmse'] for row in items],ids)
                                               for field in THERMAL_FIELDS}}
    by_m=defaultdict(list)
    for row in rows:by_m[row['module_count']].append(row)
    return {'overall':group(rows),'by_module_count':{f'M{m}':group(items) for m,items in sorted(by_m.items())},
            'small_M_visible':{f'M{m}':group(by_m[m]) if m in by_m else {'case_count':0,'available':False} for m in (1,2)}}


def wind_role_batch(provider,row,split,batch_index=0):
    from windfarm.training.native_role_evaluation import make_native_role_panel_batch
    batch=make_native_role_panel_batch(provider,int(row),split='validation' if split=='validation' else 'train-fixture',batch_index=batch_index)
    expected=np.repeat(np.arange(5,dtype=np.int8),WIND_ROLE_COUNTS)
    if not np.array_equal(np.asarray(batch.receivers.role_ids),expected[None]):raise ValueError('Wind physical five-role quotas/order changed.')
    return batch


def native_batches(provider,dataset,panel,split,*,representatives_only=False):
    validate_native_panel(panel,provider,dataset)
    keys=panel['representatives'][split] if representatives_only else panel['catalogue'][split]
    if dataset=='thermal':
        yield from provider.native_evaluation_batches(split=split,case_keys=keys,microbatch_size=1)
    else:
        for row in keys:yield wind_role_batch(provider,row,split,panel['catalogue'][split].index(row))


def typed_work_summary(work):
    """Sum executed work, using the physical panel's single pair denominator.

    Providers may also export per-decode available-pair counters. The unprefixed
    counter is the independently counted query/present-record inventory for the
    case, so adding those aliases would count availability more than once.
    """
    result={}
    for kind in ('M','E'):
        entries=[]
        for row in work:
            def lookup(suffix):
                canonical='joint_'+kind+'_'+suffix
                if suffix=='available_present_rows' and canonical in row:
                    return row[canonical]
                selected=[value for name,value in row.items() if name.endswith('joint_'+kind+'_'+suffix)]
                return sum(selected) if selected else None
            selected,candidates,executed,present=(lookup(name) for name in ('selected_rows','candidate_rows','value_rows','available_present_rows'))
            if selected is None or candidates is None or executed is None or present is None:continue
            entries.append((selected,candidates,executed,present))
        result[kind]=({'available':False,'reason':'this original model has no typed reader counter'} if not entries else
            {'available':True,'case_count':len(entries),'selected_rows':sum(v[0] for v in entries),
             'candidate_rows':sum(v[1] for v in entries),'executed_nonlinear_rows':sum(v[2] for v in entries),
             'available_present_rows':sum(v[3] for v in entries),
             'equal_case_selected_fraction':float(np.mean([v[0]/v[3] for v in entries if v[3]>0])) if any(v[3]>0 for v in entries) else None,
             'denominator':'actual valid query and present M / positive-quadrature E pairs; raw padding candidates excluded'})
    return result


def predict_physical(provider,model,batch,epoch):
    scene=provider.make_scene(batch.scene_inputs)
    return provider.predict_native(model,scene,batch.receivers,'A2-D','joint',epoch=epoch,temperature=1.)


@torch.no_grad()
def evaluate_board(model,provider,payload,panel,split,output,*,exports='representatives'):
    from time import perf_counter
    if split not in ('train','validation') or exports not in ('none','representatives'):raise ValueError('Invalid physical board scope.')
    dataset=panel['dataset'];validate_native_panel(panel,provider,dataset)
    output=Path(output)
    if output.exists():raise FileExistsError('Physical board evidence is immutable.')
    output.mkdir(parents=True);started=perf_counter();model.eval();rows=[];exported=[];query_receipts={};work=[]
    for batch in native_batches(provider,dataset,panel,split):
        prediction,auxiliary=predict_physical(provider,model,batch,int(payload['epoch']))
        key=batch.case_keys[0]
        if dataset=='thermal':
            row,arrays=thermal_record_and_arrays(prediction,batch)
            query_hash=array_sha256(batch.receivers.fluid_xy,batch.receivers.local_query_points,
                                    prediction.native_prepared.query_batch.query_ids)
        else:
            from windfarm.training.native_role_evaluation import physical_role_metrics
            target=np.asarray(batch.targets.velocity_mps[0]);pred=array(prediction.main_mps[0]);roles=array(prediction.role_ids[0])
            row=physical_role_metrics(batch.targets.case_metadata[0],pred,target,roles)
            arrays={'query_coordinates':np.asarray(batch.receivers.coordinates_D[0]),
                'native_flat_indices':np.asarray(batch.receivers.flat_indices[0]),'role_ids':roles,
                'velocity_prediction_mps':pred,'velocity_target_mps':target,'velocity_error_mps':pred-target,
                'profile_mps':array(prediction.profile_mps[0]),'normalized_residual':array(prediction.normalized_residual[0])}
            scene=auxiliary['organization'].scene
            arrays.update(module_coordinates_D=array(scene.module_coordinates[0]),module_ids=array(scene.module_ids[0]),
                module_present=array(scene.module_mask[0]),module_characteristic_lengths_D=array(scene.module_characteristic_lengths[0]),
                environment_coordinates_D=array(scene.environment_coordinates[0]),
                units_json=np.asarray(json.dumps({'coordinates':'rotor diameters, original wind-aligned frame','velocity':'m/s'})))
            query_hash=batch.auxiliary['query_sampling_sha256']
        row['query_sampling_sha256']=query_hash;rows.append(row);query_receipts[str(key)]=query_hash
        measured=dict(provider.work_counts(batch,prediction,auxiliary))
        if dataset=='thermal':
            prepared=prediction.native_prepared;scene=prepared.scene_batch;query=prepared.query_batch
        else:
            scene=auxiliary['organization'].scene;query=provider.adapter.make_query_batch(scene,batch.receivers)
        queries=int(query.query_mask.sum()) if query.query_mask is not None else query.num_queries
        measured['joint_M_available_present_rows']=queries*int(scene.module_mask.sum())
        measured['joint_E_available_present_rows']=queries*int((scene.environment_mask & (scene.environment_measure>0)).sum())
        work.append(measured)
        if exports=='representatives' and key in panel['representatives'][split]:
            path=output/f'case_{key}.npz';write_arrays(path,arrays)
            exported.append({'case_id':key,'path':path.name,'sha256':file_sha256(path),
                'array_shapes':{name:list(value.shape) for name,value in arrays.items()}})
        del prediction,auxiliary,arrays,batch
    expected=panel['catalogue'][split]
    observed=[row['case_id'] if dataset=='thermal' else row['row_index'] for row in rows]
    if observed!=expected:raise ValueError('Physical board did not visit exact frozen cohort order.')
    if dataset=='thermal':metrics=reduce_thermal_records(rows)
    else:
        from windfarm.training.native_role_evaluation import summarize_physical_role_metrics
        metrics=summarize_physical_role_metrics(rows)
    result={'schema':BOARD_SCHEMA,'dataset':dataset,'split':split,'completed_epoch':int(payload['epoch']),
        'arm':checkpoint_arm(payload),'panel_sha256':panel['panel_sha256'],
        'scope':'all selected original TRAIN endpoint audit' if split=='train' else 'complete exposed native DEV board',
        'receiver_policy':panel['physical_receiver_policy'],'scientific_monitor_policy_unchanged':True,
        'parameter_count':sum(parameter.numel() for parameter in model.parameters()),
        'query_receipts':query_receipts,'records':rows,'metrics':metrics,'work_counts':work,'typed_work':typed_work_summary(work),'exports':exported,
        'elapsed_seconds':perf_counter()-started,'optimizer_updates':0,'new_solver_attempts':0,'WindTEST':'locked'}
    if dataset=='thermal':
        result['TRAIN_role_scales']={**{name:float(model.field_std[i]) for i,name in enumerate(THERMAL_FIELDS[:5])},
            'temperature_surface':float(model.interface_target_std[0]),'temperature_material':float(model.internal_temperature_std[0]),
            'q_proxy':float(model.interface_target_std[1]),'module_peak_endpoint':float(model.internal_temperature_std[0])}
    else:
        scales=np.asarray(provider.adapter.normalizer.safe_std)*provider.adapter.normalizer.u_ref_mps
        result['TRAIN_role_scales']={**dict(zip(('Ux','Uy','Uz'),map(float,scales))),'vector':float(np.linalg.norm(scales))}
    write_json_new(output/'result.json',result)
    return result


def compare_physical_boards(boards):
    """Same-age role bands and E utility, preserving unresolved denominators."""
    required={'D9-M','D9-ME','H9-ME','S9-ME'}
    if set(boards)!=required:raise ValueError('A comparison needs all four literal arms, including the shared all-open boundary at3000.')
    first=boards['D9-M']
    for label,board in boards.items():
        allowed={label}
        if label=='D9-ME' and board['completed_epoch']==3000:allowed.add('D9-ME-warm')
        if board.get('arm') not in allowed:raise ValueError('Physical board label differs from its literal trained arm.')
        if any(board[key]!=first[key] for key in ('schema','dataset','split','completed_epoch','panel_sha256','query_receipts')):
            raise ValueError('Physical comparisons require identical literal age, cohort, receivers and query IDs.')
        scales=np.asarray(list(board['TRAIN_role_scales'].values()),dtype=np.float64)
        if not scales.size or not np.all(np.isfinite(scales)&(scales>0)):
            raise ValueError('Frozen TRAIN physical scales must be finite and strictly positive.')
    dataset=first['dataset'];comparisons={}
    if dataset=='thermal':
        priorities={'temperature_fluid','temperature_surface','temperature_material','q_proxy','module_peak_endpoint'}
        def metrics(board):return board['metrics']['overall']['fields']
        fields=list(metrics(first))
        def scale(board,name):return board['TRAIN_role_scales'][name]
    else:
        fields=[f'{role}/{component}' for role in first['metrics']['overall']['roles'] for component in ('Ux','Uy','Uz','vector')]
        priorities={field for field in fields if field.rsplit('/',1)[-1] in ('Ux','vector')}
        def metrics(board):
            return {f'{role}/{component}':values['vector_rmse_mps'] if component=='vector' else values['component_rmse_mps'][component]
                    for role,values in board['metrics']['overall']['roles'].items() for component in ('Ux','Uy','Uz','vector')}
        def scale(board,name):return board['TRAIN_role_scales'][name.rsplit('/',1)[-1]]
    pairs=(('H9-ME','D9-ME'),('H9-ME','D9-M'),('D9-ME','D9-M'),('H9-ME','S9-ME'))
    for numerator,denominator in pairs:
        a,b=metrics(boards[numerator]),metrics(boards[denominator]);table={}
        for name in fields:
            floor=(1e-4 if name in ('omega','q_proxy') else 1e-5)*scale(boards[denominator],name)
            table[name]={}
            for statistic in ('mean','p90','worst'):
                old,new=b[name][statistic],a[name][statistic];resolved=old>floor
                band=(.10 if statistic=='mean' else .15 if statistic=='p90' and name in priorities else None)
                table[name][statistic]={'prediction':new,'reference':old,'absolute_change':new-old,
                    'change_over_frozen_TRAIN_scale':(new-old)/scale(boards[denominator],name),
                    'resolved_denominator':resolved,'numerical_resolution_in_native_units':floor,
                    'relative_change':new/old-1 if resolved else None,
                    'prospective_band':band,
                    'within_band':new<=(1+band)*old if resolved and band is not None else None,
                    'priority':name in priorities}
        comparisons[numerator+'_versus_'+denominator]=table
    strata={}
    for reference in ('D9-M','D9-ME'):
        key='by_module_count' if dataset=='thermal' else 'by_active_module_count_M'
        aa,bb=boards['H9-ME']['metrics'][key],boards[reference]['metrics'][key]
        if set(aa)!=set(bb):raise ValueError('Physical comparison lost a populated module-count stratum.')
        strata[reference]={}
        for m in aa:
            if dataset=='thermal':a,b=aa[m]['fields'],bb[m]['fields']
            else:
                def flat(group):
                    return {f'{role}/{component}':values['vector_rmse_mps'] if component=='vector' else values['component_rmse_mps'][component]
                            for role,values in group['roles'].items() for component in ('Ux','Uy','Uz','vector')}
                a,b=flat(aa[m]),flat(bb[m])
            strata[reference][m]={}
            for name in sorted(priorities):
                old,new=b[name]['mean'],a[name]['mean'];floor=(1e-4 if name=='q_proxy' else 1e-5)*scale(boards[reference],name)
                strata[reference][m][name]={'H_mean':new,'reference_mean':old,'reference_arm':reference,
                    'absolute_change':new-old,'relative_change':new/old-1 if old>floor else None,
                    'exceeds_prospective_25_percent_stratum_band':new>1.25*old if old>floor else None,
                    'case_count':aa[m].get('case_count',aa[m].get('row_count'))}
    work={}
    for kind in ('M','E'):
        a=boards['H9-ME'].get('typed_work',{}).get(kind,{});b=boards['S9-ME'].get('typed_work',{}).get(kind,{})
        delta=(a['equal_case_selected_fraction']-b['equal_case_selected_fraction']) if a.get('available') and b.get('available') and a['equal_case_selected_fraction'] is not None and b['equal_case_selected_fraction'] is not None else None
        work[kind]={'H_minus_S_fraction':delta,'unmatched_by_more_than_five_percentage_points':abs(delta)>.05 if delta is not None else None}
    return {'schema':'round9_matched_physical_board_comparison_v1','dataset':dataset,'split':first['split'],
        'completed_epoch':first['completed_epoch'],'panel_sha256':first['panel_sha256'],'comparisons':comparisons,
        'phase':'shared_all_open_sparse_boundary' if first['completed_epoch']==3000 else 'trained_same_age_arms',
        'priority_module_count_strata':strata,
        'trained_H_S_work_match':work,'scope':'same-age native physical-role comparison; E architectural utility is not independent environmental information',
        'unresolved_denominator_policy':'retain absolute numbers and frozen TRAIN-scale changes; no ratio or automatic band pass',
        'organizer_superiority_is_causal_claim':False}
