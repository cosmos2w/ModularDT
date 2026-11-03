#!/usr/bin/env python3
"""Evaluate fixed R15-8 response holdouts from strict checkpoints and raw solves."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'Case_ThermalChannel/src')]
import numpy as np
import torch

from channelthermal.data.datasets import _local_disk_query_points
from channelthermal.thermal_coupling import local_response_from_outside, outside_from_native_output
from channelthermal.training.thermal_finite_corpus import load_raw_endpoint
from channelthermal.training.thermal_finite_evaluation import field_metrics, summarize_actions
from channelthermal.training.thermal_finite_response_task import load_finite_response_checkpoint, load_finite_response_source
from channelthermal.training.thermal_native_functionals import native_design_functionals
from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1
from honf_forward_core.content_hypergraph.staged_attention_v2 import safe_masked_softmax

ACTIONS = ('single_plus', 'single_minus', 'joint_plus', 'joint_minus')
CHANNELS = ('u', 'v', 'p', 'omega', 'fluid_T', 'surface_T', 'material_T', 'q')


def ledger_paths(ledger):
    started, finished = {}, {}
    for line in Path(ledger).read_text().splitlines():
        row = json.loads(line)
        if row['event'] == 'started': started[row['attempt']] = row
        elif row['event'] == 'finished': finished[row['attempt']] = row
    result = {}
    for attempt, start in started.items():
        end = finished.get(attempt, {})
        if start.get('role') != 'response_holdout' or start.get('allocation') != 'edited': continue
        key = (str(start['case_id']), start['action'])
        if not end: raise ValueError(f'Unfinished holdout physical action: {key}')
        entry = result.setdefault(key, {'path': None, 'attempts': []})
        success = end.get('status') == 'converged' and end.get('serialization_status') == 'stored'
        entry['attempts'].append({'attempt': attempt, 'status': end.get('status'),
                                  'serialization_status': end.get('serialization_status')})
        if success:
            if entry['path'] is not None: raise ValueError(f'Duplicate successful fixed action: {key}')
            entry['path'] = end['raw_case_dir']
    return result


def as_numpy(value):
    return value.detach().cpu().numpy()


def native_inputs(model, raw):
    cfg = raw['config']; domain = cfg['domain']; thermal = cfg['thermal']; flow = cfg['flow']
    n = len(cfg['layout']['centers']); device = next(model.parameters()).device
    tensor = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32).copy(), device=device)
    nu = cfg.get('runtime', {}).get('nu')
    if nu is None: nu = flow.get('nu')
    if nu is None: raise ValueError('Physical endpoint does not declare its actual viscosity.')
    structure = {'module_centers': tensor(cfg['layout']['centers'])[None],
        'module_present': tensor(np.ones(n))[None], 'module_source_ids': tuple(str(i) for i in range(n)),
        'material_params': tensor([nu, thermal['solid_alpha'], thermal['fluid_alpha'], thermal['solid_k'], thermal['fluid_k'], domain['module_radius']])[None],
        'heat_powers': tensor(cfg['layout']['heat_powers'])[None]}
    for key, value in {'re':flow['re'], 'u_in':flow['u_in'], 'domain_length_x':domain['lx'],
                       'domain_length_y':domain['ly'], 't_in':thermal['t_in'], 't_wall':thermal['t_wall']}.items():
        structure[key] = tensor([value])[None]
    x = (np.arange(domain['nx'], dtype=np.float32) + .5) * (domain['lx']/domain['nx'])
    y = (np.arange(domain['ny'], dtype=np.float32) + .5) * (domain['ly']/domain['ny'])
    xx, yy = np.meshgrid(x, y)
    fluid = tensor(np.stack((xx, yy), -1).reshape(-1, 2))[None]
    local = tensor(_local_disk_query_points(raw['material_mask']))[None]
    return structure, fluid, local


def infer(model, local_model, raw, *, mode='restricted', trace=False):
    structure, fluid, local_points = native_inputs(model, raw)
    device = next(model.parameters()).device
    if device.type == 'cuda': torch.cuda.synchronize(device)
    started = time.perf_counter()
    prep = model.prepare_native(structure, fluid, physical_heat=structure['heat_powers'],
        local_query_points=local_points, ntheta=64, capture_context_trace=trace)
    native, work, core = model.apply_native_with_work_counts(prep, structure['heat_powers'], intervention={'read_mode': mode})
    outside = outside_from_native_output(model, prep, core)
    local = local_response_from_outside(model, local_model, prep, structure['heat_powers'], local_points, outside)
    if device.type == 'cuda': torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    material = native['pred_internal_temperature'][..., 0]; pressure = native['pred_field'][..., 2]
    matmask = prep.stencils['material'].valid.reshape(material.shape) & prep.source_present[..., None]
    fluidmask = torch.as_tensor(raw['fluid_mask'].reshape(1,-1), device=pressure.device) & prep.stencils['fluid'].valid.reshape(pressure.shape)
    j,p = native_design_functionals(material,matmask,pressure,fluidmask,fluid[...,0],raw['config']['domain']['lx'],.05*float(model.internal_temperature_std[0]))
    # Apply the identical predicted/reference physical and stencil masks to functionals.
    physical_material = torch.as_tensor(raw['material'][:,raw['material_mask']][None], device=pressure.device)
    physical_pressure = torch.as_tensor(raw['field'][...,2].reshape(1,-1),device=pressure.device)
    jr,pr = native_design_functionals(physical_material,matmask,physical_pressure,fluidmask,fluid[...,0],raw['config']['domain']['lx'],.05*float(model.internal_temperature_std[0]))
    pred = {'field':as_numpy(native['pred_field'][0]), 'interface':as_numpy(native['pred_interface'][0]),
        'material':as_numpy(material[0]), 'outside':as_numpy(outside[0]),
        'local_interface':as_numpy(local['interface_pred'][0]), 'local_material':as_numpy(local['internal_temperature'][0,...,0])}
    physical = {'field':raw['field'].reshape(-1,5), 'interface':raw['interface_target'],
        'material':raw['material'][:,raw['material_mask']], 'outside':raw['teacher_ports'][...,3],
        'local_interface':raw['interface_target'], 'local_material':raw['material'][:,raw['material_mask']]}
    surface = as_numpy(prep.stencils['surface'].valid).reshape(pred['interface'].shape[:-1])
    outside_mask = as_numpy(prep.stencils['outside'].valid).reshape(surface.shape) & raw['outside_valid']
    masks = {'field':as_numpy(fluidmask[0]), 'interface':surface, 'material':as_numpy(matmask[0]),
        'outside':outside_mask, 'local_interface':surface & outside_mask, 'local_material':as_numpy(matmask[0])}
    return {'prediction':pred,'reference':physical,'mask':masks,'model':{'J':j.item(),'P':p.item()},
            'physical':{'J':jr.item(),'P':pr.item()},'work':{'decode':work,
            'preparation':{str(k):int(v) for k,v in prep.organization.work_counts.items()}},
            'seconds':seconds,'prepared':prep,'core':core}


def metrics(one, other=None):
    result = {}
    specs = [('field', i, CHANNELS[i]) for i in range(5)] + [('interface',0,'surface_T'),('material',None,'material_T'),('interface',1,'q'),('outside',None,'outside_T'),('local_interface',0,'local_surface_T'),('local_interface',1,'local_q'),('local_material',None,'local_material_T')]
    for role, channel, name in specs:
        p=one['prediction'][role]; t=one['reference'][role]; mask=one['mask'][role]
        if channel is not None: p=p[...,channel]; t=t[...,channel]
        if role in ('interface','local_interface') and channel == 1: mask=mask & one['mask']['outside']
        if other is not None:
            op=other['prediction'][role]; ot=other['reference'][role]; om=other['mask'][role]
            if channel is not None: op=op[...,channel]; ot=ot[...,channel]
            if role in ('interface','local_interface') and channel == 1: om=om & other['mask']['outside']
            p=p-op; t=t-ot; mask=mask & om
        mask=mask & np.isfinite(t)
        result[name]=field_metrics(p,t,mask)
    return result


def _guarded_context_coefficients(snapshot, gate):
    """Reconstruct contextual coefficients from a captured actual route."""
    measure = snapshot.donor_measure.to(snapshot.raw_scores.dtype)
    valid = measure > 0
    safe = torch.where(valid, measure, torch.ones_like(measure))
    allowed = snapshot.allowed & valid[:, None, :]
    base = safe_masked_softmax(snapshot.raw_scores + safe.log()[:, None, None, :],
                               allowed[:, None], dim=-1)
    weighted = base * gate[:, None]
    z = weighted.sum(-1, keepdim=True)
    return weighted / (z + 1.e-3 * (1 - z).square())


def _dense_directory(table, module_ids, environment_ids):
    """Unpack the single-scene E rows without changing member identity."""
    offsets = as_numpy(table.offsets)
    members = as_numpy(table.member_index)
    original = as_numpy(table.original_id)
    roles = as_numpy(table.is_module).astype(bool)
    weights = as_numpy(table.access_weight)
    nm, ne = len(module_ids), len(environment_ids)
    if offsets.shape != (1, ne + 1) or offsets[0, 0] != 0 or offsets[0, -1] != len(members):
        raise ValueError('Organization export requires one complete packed E directory.')
    result = np.zeros((ne, nm + ne), dtype=weights.dtype)
    for row in range(ne):
        start, stop = offsets[0, row:row+2]
        index = members[start:stop]
        if (np.any(index < 0) or np.any(index >= nm + ne) or len(np.unique(index)) != len(index)
                or not np.array_equal(roles[start:stop], index < nm)):
            raise ValueError('Packed E directory has invalid original-member slots.')
        expected = np.concatenate((module_ids, environment_ids))[index]
        if not np.array_equal(original[start:stop], expected):
            raise ValueError('Packed E directory original IDs differ from the scene.')
        result[row, index] = weights[start:stop]
    return result


def save_organization(outpath, result, model, mode, *, metadata=None):
    prep=result['prepared']; org=prep.organization; scene=org.scene; values={}
    if scene.module_coordinates.shape[0] != 1 or scene.environment_coordinates.shape[0] != 1:
        raise ValueError('Organization figures require one original scene per export.')
    module_ids=as_numpy(scene.module_ids[0]); environment_ids=as_numpy(scene.environment_ids[0])
    values.update(module_xy=as_numpy(scene.module_coordinates[0]), environment_xy=as_numpy(scene.environment_coordinates[0]),
        module_ids=module_ids, environment_ids=environment_ids,
        module_mask=as_numpy(scene.module_mask[0]), environment_mask=as_numpy(scene.environment_mask[0]),
        module_measure=as_numpy(scene.module_measure[0]), environment_measure=as_numpy(scene.environment_measure[0]),
        module_characteristic_lengths=as_numpy(scene.module_characteristic_lengths[0]),
        environment_characteristic_lengths=as_numpy(scene.environment_characteristic_lengths[0]))
    structure=prep.structure
    lx=float(structure['domain_length_x'].reshape(-1)[0]); ly=float(structure['domain_length_y'].reshape(-1)[0])
    values['domain_bounds']=np.asarray([0., lx, 0., ly])
    values['radius']=np.full(len(module_ids),float(structure['material_params'][0,5]))
    for directory_mode in ('full','restricted'):
        table=getattr(org.e_access_directory,directory_mode)
        for key in ('offsets','member_index','original_id','is_module','access_weight','applied_exchange','structural_self','measure'):
            values[directory_mode+'_'+key]=as_numpy(getattr(table,key))
        values['C_'+directory_mode]=_dense_directory(table,module_ids,environment_ids)
    for cycle,stage in enumerate(org.context_stage_traces):
        for route in ('mm','em','ee','me'):
            snap=stage[route+'_native_route_snapshot']; prefix=f'cycle{cycle}_{route}_'
            gate=stage[route+'_applied_incidence']
            values[prefix+'G']=as_numpy(gate); values[prefix+'B']=as_numpy(_guarded_context_coefficients(snap,gate))
            values[prefix+'raw_scores']=as_numpy(snap.raw_scores); values[prefix+'donor_measure']=as_numpy(snap.donor_measure)
            values['context_cycle'+str(cycle)+'_'+route]=as_numpy(gate[0])
            values[prefix+'receiver_ids']=as_numpy(snap.destination_ids)
            values[prefix+'donor_ids']=as_numpy(snap.donor_ids); values[prefix+'allowed']=as_numpy(snap.allowed)
    query=prep.query_batch
    selected=select_query_rows_v1(query,list(range(min(16,query.num_queries))))
    trace=model.core.decode(org,selected,return_trace=True,intervention={'read_mode':mode}).trace
    values['query_coordinates']=as_numpy(selected.coordinates)
    if selected.query_ids is not None: values['query_ids']=as_numpy(selected.query_ids)
    for key in ('anchor_indices','anchor_weights','anchor_read_coalesced_member_coefficients',
                'anchor_read_coalesced_union_keys','anchor_read_original_id','context_M','context_E','joint_fields'):
        if key in trace: values['query_'+key]=as_numpy(trace[key])
    for key, value in result['prediction'].items(): values['prediction_'+key]=value
    for key, value in result['reference'].items(): values['reference_'+key]=value
    for key, value in result['mask'].items(): values['mask_'+key]=value
    np.savez_compressed(outpath,**values)
    provenance={'partition':'TRAIN-origin response-holdout', 'coordinate_units':'dataset length units',
        'read_mode':mode, 'stage_provenance':'actual sequential contextual route gates, receiver <- donor',
        'coefficient_provenance':'B reconstructed diagnostically from captured raw route scores, donor measures, allowed support and actual applied gates using guarded reduction; not a capture of native kernel arithmetic',
        'route_provenance':'actual prepared E directory; C matrices are exact unpackings of packed original-member rows',
        **(metadata or {})}
    Path(outpath).with_suffix('.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')


@contextmanager
def disabled_pair_content(model):
    params=[p for n,p in model.core.named_parameters() if n.endswith('pair_content.U')]
    before=[p.detach().clone() for p in params]
    try:
        with torch.no_grad():
            for p in params: p.zero_()
        yield
    finally:
        with torch.no_grad():
            for p,v in zip(params,before): p.copy_(v)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-checkpoint',type=Path,required=True);p.add_argument('--checkpoint',action='append',default=[],metavar='ARM=PATH')
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--resolution',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Floors are read and validated before any holdout outcome or corpus ledger.
    floors=json.loads(args.resolution.read_text())
    if not floors.get('source') or any(not math.isfinite(float(floors.get(k,0))) or float(floors[k]) <= 0 for k in ('J','P')):
        raise ValueError('A fixed TRAIN-derived source and positive J/P resolution floors are required.')
    recipe=json.loads(args.recipe.read_text()); paths=ledger_paths(args.ledger)
    selected=[c for c in recipe['cases'] if c['role']=='response_holdout']
    if len(selected) != 12 or len({str(c['case_id']) for c in selected}) != 12:
        raise ValueError('Expected the 12 distinct fixed response-holdout families.')
    cases=[c for c in selected if c['feasible']]
    if not cases: raise ValueError('No geometry-feasible holdout family remains.')
    for case in cases:
        for action in ACTIONS:
            if (str(case['case_id']),action) not in paths: raise ValueError(f'Holdout action not yet attempted: {case["case_id"]}/{action}')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    checkpoints={'P64':args.source_checkpoint}
    for item in args.checkpoint:
        arm,path=item.split('=',1)
        if arm not in ('A','B','C') or arm in checkpoints: raise ValueError('Checkpoints must be uniquely named A/B/C.')
        checkpoints[arm]=Path(path)
    start=time.perf_counter(); cpu_started=time.process_time()
    source,local,source_payload=load_finite_response_source(args.source_checkpoint,device=args.device); source.eval();local.eval()
    training_ids=set(map(str,source_payload['experiment_identity']['provider_identity']['training_case_ids']))
    if any(str(c['case_id']) not in training_ids for c in selected):
        raise ValueError('Response-holdout family is outside the saved original TRAIN membership.')
    representatives={str(c['case_id']) for c in (min(cases,key=lambda c:(c['module_count'],c['case_id'])),max(cases,key=lambda c:(c['module_count'],c['case_id'])))}
    report={'recipe':str(args.recipe),'ledger':str(args.ledger),'floors':floors,'floor_path':str(args.resolution),'partition':'TRAIN-origin response-holdout; baselines are TRAIN600; edited labels excluded from fitting', 'representatives':sorted(representatives),
        'units':'native dataset velocity, pressure, vorticity, temperature and heat-flux-proxy units',
        'masks':'native source and stencil validity; delta uses both endpoint masks; attached material/angles use original module order; q also requires outside validity',
        'trace_scope':'separate diagnostic preparations; G is actual applied gates; B is diagnostically reconstructed from captured scores/measures/gates with guarded reduction, not native kernel identity; full/restricted access rows retain original IDs; first 16 original union queries; no causal interpretation',
        'cost_scope':'native timing includes preparation, field decode and frozen local call; arm wall includes input loading, scoring and diagnostics, and excludes final JSON serialization; source load is included in total wall',
        'infeasible_families':[str(c['case_id']) for c in selected if not c['feasible']],
        'failed_attempts':[{'case_id':k[0],'action':k[1],**attempt} for k,v in paths.items() for attempt in v['attempts']
                           if attempt['status'] != 'converged' or attempt['serialization_status'] != 'stored'],
        'unavailable_actions':[{'case_id':k[0],'action':k[1]} for k,v in paths.items() if v['path'] is None], 'models':{}}
    matched_epoch=None
    for arm,path in checkpoints.items():
        for mode in ('full', 'restricted'):
            arm_start=time.perf_counter(); cpu_start=time.process_time()
            if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            if arm=='P64': model=source; saved_epoch=100
            else:
                model,saved=load_finite_response_checkpoint(path,device=args.device);saved_epoch=saved['epoch']
                spec=saved['experiment_identity']['recipe']['thermal_finite_response']
                if spec['arm'] != arm or Path(spec['source_checkpoint']).resolve() != args.source_checkpoint.resolve():
                    raise ValueError('Checkpoint arm or common source identity differs.')
                if matched_epoch is None: matched_epoch=saved_epoch
                elif saved_epoch != matched_epoch: raise ValueError('Study arms must have the same new-epoch horizon.')
            model.eval(); rows=[]; endpoint_rows=[]; timings=[]
            with torch.inference_mode():
                for case in cases:
                    case_id=str(case['case_id']); braw=load_raw_endpoint(case['source_case_dir'])
                    if not np.allclose(braw['config']['layout']['centers'],case['centers'],rtol=0,atol=1e-6): raise ValueError('Baseline original geometry differs from recipe.')
                    if case['original_ids'] != list(range(len(case['original_ids']))): raise ValueError('Original module ID order is unsupported.')
                    base=infer(model,local,braw,mode=mode); timings.append(base['seconds']); endpoint_rows.append({'case_id':case_id,'action':'baseline','model':base['model'],'physical':base['physical'],'fields':metrics(base)})
                    if case_id in representatives:
                        traced_base=infer(model,local,braw,mode=mode,trace=True)
                        save_organization(args.output/f'{arm}_{mode}_{case_id}_baseline_trace.npz',traced_base,model,mode,
                            metadata={'case':case_id,'action':'baseline','arm':arm,'checkpoint':str(path),
                                      'epoch':saved_epoch,'learned_age':5100+(0 if arm=='P64' else saved_epoch)})
                        endpoint_rows[-1]['trace_minus_ordinary']={k:traced_base['model'][k]-base['model'][k] for k in ('J','P')}
                        repeated=infer(model,local,braw,mode=mode)
                        heat_raw=copy.copy(braw);heat_raw['config']=copy.deepcopy(braw['config'])
                        heat_raw['config']['layout']['heat_powers'][0] += .01
                        heat=infer(model,local,heat_raw,mode=mode)
                        endpoint_rows[-1]['diagnostics']={'zero_edit_max_abs_field_difference':float(np.max(np.abs(repeated['prediction']['field']-base['prediction']['field']))),
                            'heat_only_delta_first_module':.01,'heat_only_max_abs_flow_difference':float(np.max(np.abs(heat['prediction']['field'][...,:4]-base['prediction']['field'][...,:4]))),
                            'meaning':'same-model no-op and heat-to-flow-null diagnostics, without physical labels'}
                    for action in ACTIONS:
                        path_info=paths[(case_id,action)]
                        if path_info['path'] is None: continue
                        raw=load_raw_endpoint(path_info['path'])
                        if not np.allclose(raw['config']['layout']['centers'],case['selection']['actions'][action],rtol=0,atol=1e-6): raise ValueError('Edited geometry differs from fixed recipe.')
                        if raw['config']['layout']['heat_powers'] != braw['config']['layout']['heat_powers']:
                            raise ValueError('Edited endpoint heat inputs differ from baseline.')
                        if any(raw['config'][part][key] != braw['config'][part][key] for part,key in
                               (('domain','lx'),('domain','ly'),('domain','nx'),('domain','ny'),('domain','module_radius'),('flow','re'),('flow','u_in'),('thermal','solid_k'),('thermal','fluid_k'),('thermal','solid_alpha'),('thermal','fluid_alpha'),('thermal','t_in'),('thermal','t_wall'))):
                            raise ValueError('Edited endpoint physics or native geometry differs from baseline.')
                        if not np.array_equal(raw['material_mask'],braw['material_mask']): raise ValueError('Material catalogue changed.')
                        edited=infer(model,local,raw,mode=mode);timings.append(edited['seconds'])
                        row={'case_id':case_id,'action':action,'amplitude':case['amplitude'],'joint_kind':case['joint_kind'],
                            'delta':{k:{'model':edited['model'][k]-base['model'][k], 'physical':edited['physical'][k]-base['physical'][k]} for k in ('J','P')},'fields':metrics(edited,base),'work':edited['work']}
                        rows.append(row);endpoint_rows.append({'case_id':case_id,'action':action,'model':edited['model'],'physical':edited['physical'],'fields':metrics(edited)})
                        if case_id in representatives and action == ACTIONS[0]:
                            # Traced preparation is separately labelled and never used for scored results.
                            traced=infer(model,local,raw,mode=mode,trace=True); save_organization(args.output/f'{arm}_{mode}_{case_id}_{action}_trace.npz',traced,model,mode,
                                metadata={'case':case_id,'action':action,'arm':arm,'checkpoint':str(path),
                                          'epoch':saved_epoch,'learned_age':5100+(0 if arm=='P64' else saved_epoch)})
                            row['trace_minus_ordinary']={k:traced['model'][k]-edited['model'][k] for k in ('J','P')}
                            if arm=='C':
                                with disabled_pair_content(model):
                                    disabled_base=infer(model,local,braw,mode=mode)
                                    disabled=infer(model,local,raw,mode=mode)
                                row['residual_disabled_intervention']={'scope':'out-of-training inference intervention',
                                    'endpoint_difference':{k:disabled['model'][k]-edited['model'][k] for k in ('J','P')},
                                    'response_difference':{k:(disabled['model'][k]-disabled_base['model'][k])-(edited['model'][k]-base['model'][k]) for k in ('J','P')}}
            report['models'].setdefault(arm,{})[mode]={'checkpoint':str(path),'epoch':saved_epoch,'actions':rows,'endpoints':endpoint_rows,'summary':summarize_actions(rows,floors,family_ids=[str(c['case_id']) for c in cases]),
                'native_preparation_decode_local_seconds':{'count':len(timings),'total':sum(timings),'mean':float(np.mean(timings))},
                'complete_arm_wall_seconds':time.perf_counter()-arm_start,'complete_arm_cpu_seconds':time.process_time()-cpu_start,
                'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(torch.device(args.device)) if args.device.startswith('cuda') else None}
            (args.output/'response_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    report['elapsed_seconds']=time.perf_counter()-start
    report['total_cpu_seconds']=time.process_time()-cpu_started
    (args.output/'response_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
