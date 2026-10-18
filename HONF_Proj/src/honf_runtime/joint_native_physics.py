"""Read-path effects, saved physical finite responses and hard-route seams.

These are explicit evaluation interventions. Saved geometry and heat endpoints
are never interpolated into invented physical targets; independent environment
and geometry-plus-heat identification remain unavailable without four states.
"""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from .joint_native_evaluation import (array,array_sha256,checkpoint_arm,file_sha256,metric,native_batches,
    predict_physical,thermal_record_and_arrays,validate_native_panel,write_json_new,write_arrays,canonical_sha256)


@contextmanager
def environment_read_intervention(core,mode):
    """Change only final E values after public prepared-state validation.

    No prepared tensor, model parameter, geometry, score or support is mutated.
    Original projections are supplied to all scoring. The projected-value
    intervention changes the E value argument of the pure reader explicitly.
    """
    if mode not in ('zero_context','zero_projected_values') or core.training:
        raise ValueError('E read interventions require evaluation mode and a declared value control.')
    if not hasattr(core,'environment_reader'):raise ValueError('This model has no direct E value path.')
    original=core._read_type;had='_read_type' in core.__dict__;previous=core.__dict__.get('_read_type')
    def read(self,kind,organization,query,qstates,gate,backend,return_messages=False):
        if kind=='environment' and mode=='zero_projected_values':
            supplied=copy.copy(organization)
            supplied.joint_environment_values=torch.zeros_like(organization.joint_environment_values)
        else:supplied=organization
        result=original(kind,supplied,query,qstates,gate,backend,return_messages)
        return replace(result,context=torch.zeros_like(result.context)) if kind=='environment' and mode=='zero_context' else result
    core._read_type=MethodType(read,core)
    try:yield
    finally:
        if had:core._read_type=previous
        else:del core._read_type


def _prepared_native(provider,model,batch,prediction,auxiliary):
    if hasattr(prediction,'native_prepared'):
        return prediction.native_prepared.organization,prediction.native_prepared.query_batch
    scene=provider.make_scene(batch.scene_inputs)
    return auxiliary['organization'],provider.adapter.make_query_batch(scene,batch.receivers)


def _replay_prepared(provider,model,batch,baseline,auxiliary):
    if hasattr(baseline,'native_prepared'):
        output=model.apply_native(baseline.native_prepared,batch.receivers.heat)
        return replace(baseline,native_main=output)
    organization,query=_prepared_native(provider,model,batch,baseline,auxiliary)
    output=model.decode(organization,query)
    physical,profile,residual=provider.adapter.physical_velocity(output.nonlinear_field,query.coordinates)
    return replace(baseline,main_mps=physical,normalized_residual=output.nonlinear_field,
                   residual_mps=residual,profile_mps=profile)


def _effect_arrays_and_record(dataset,prediction,batch):
    if dataset=='thermal':return thermal_record_and_arrays(prediction,batch)
    from windfarm.training.native_role_evaluation import physical_role_metrics
    p=array(prediction.main_mps[0]);t=np.asarray(batch.targets.velocity_mps[0]);r=array(prediction.role_ids[0])
    record=physical_role_metrics(batch.targets.case_metadata[0],p,t,r)
    return record,{'velocity_prediction_mps':p,'velocity_target_mps':t,'role_ids':r,
                   'query_coordinates':np.asarray(batch.receivers.coordinates_D[0]),
                   'native_flat_indices':np.asarray(batch.receivers.flat_indices[0])}


def _fixed_read_fingerprint(core,organization,query):
    routes=core.capture_read_routes(organization,query)
    route_hashes={name:array_sha256(getattr(routes,name)) for name in
        ('module_ids','environment_ids','query_ids','module_present','environment_present','query_present','support')}
    state_hashes={name:array_sha256(getattr(organization,name)) for name in
        ('module_states','environment_states','direct_source_keys','direct_source_values','joint_environment_keys','joint_environment_values')}
    return {'typed_ids_and_support':route_hashes,'prepared_states_and_projections':state_hashes,
            'combined_sha256':canonical_sha256({'routes':route_hashes,'states':state_hashes})}


def _effect_magnitudes(delta,mask,names=None):
    delta=np.asarray(delta,dtype=np.float64);mask=np.broadcast_to(mask,delta.shape)
    def values(selected):
        return {'rms':float(np.sqrt(np.mean(selected**2))),'max_abs':float(np.max(np.abs(selected))),
                'valid_entries':int(selected.size)} if selected.size else {'rms':None,'max_abs':None,'valid_entries':0}
    result=values(delta[mask]);result['pooled_units_warning']='mixed channel units; use per_channel for physical interpretation' if names else None
    if names:
        result['per_channel']={name:values(delta[...,i][mask[...,i]]) for i,name in enumerate(names)}
    return result


def _effect_validity(name,arrays,shape):
    if name=='field_prediction':return arrays['fluid_valid'][:,None]
    if name=='interface_prediction':
        return np.stack((arrays['interface_valid'],arrays['interface_valid']&arrays['outside_valid']),axis=-1)
    if name=='material_prediction':return arrays['material_valid']
    return np.ones(shape,dtype=bool)


def _kernel_effect_magnitudes(delta,query,scene):
    """K keeps physically present zero-measure M sources, excluding padding."""
    qmask=(torch.ones_like(query.query_ids,dtype=torch.bool) if query.query_mask is None else query.query_mask)
    valid=array(qmask[:,:,None]&scene.module_mask[:,None,:])
    values=array(delta);summary=_effect_magnitudes(values,valid)
    summary['units']='core normalized temperature per native physical heat control; multiply by adapter.field_std[4] for physical temperature/heat'
    summary['denominator']='valid native query requests times physically present M; zero-measure M retained'
    summary['per_native_source']=[{'batch_index':b,'native_module_id':int(scene.module_ids[b,m]),
        **_effect_magnitudes(values[b,:,m],valid[b,:,m])}
        for b in range(values.shape[0]) for m in range(values.shape[-1]) if bool(scene.module_mask[b,m])]
    return summary,valid


@torch.no_grad()
def evaluate_effects(model,provider,payload,panel,split,output):
    dataset=panel['dataset'];validate_native_panel(panel,provider,dataset);model.eval()
    output=Path(output)
    if output.exists():raise FileExistsError('Read-path evidence is immutable.')
    output.mkdir(parents=True);core=getattr(model,'core',model);rows=[]
    if not hasattr(core,'environment_reader'):
        result={'schema':'round9_E_read_effects_v1','available':False,'reason':'D-M has no direct E value reader',
                'completed_epoch':int(payload['epoch']),'optimizer_updates':0}
        write_json_new(output/'result.json',result);return result
    for batch in native_batches(provider,dataset,panel,split,representatives_only=True):
        baseline,auxiliary=predict_physical(provider,model,batch,int(payload['epoch']))
        organization,query=_prepared_native(provider,model,batch,baseline,auxiliary)
        fixed_fingerprint=_fixed_read_fingerprint(core,organization,query)
        reference,arrays=_effect_arrays_and_record(dataset,baseline,batch)
        saved={'baseline__'+name:value for name,value in arrays.items()};variants={}
        baseline_kernel=core.export_source_kernels(organization,query) if core.thermal_response_rank else None
        if baseline_kernel is not None:
            _,kernel_valid=_kernel_effect_magnitudes(torch.zeros_like(baseline_kernel),query,organization.scene)
            saved.update(baseline__native_signed_K=array(baseline_kernel[0]),native_signed_K_valid=kernel_valid[0],
                native_signed_K_query_ids=array(query.query_ids[0]),native_signed_K_module_ids=array(organization.scene.module_ids[0]),
                native_signed_K_units=np.asarray('core normalized temperature per native physical heat control'),
                native_signed_K_physical_temperature_scale=array(model.field_std[4]))
        for mode in ('zero_context','zero_projected_values'):
            with environment_read_intervention(core,mode):
                control_fingerprint=_fixed_read_fingerprint(core,organization,query)
                if control_fingerprint!=fixed_fingerprint:raise ValueError('E value control changed original IDs, support or prepared states.')
                changed=_replay_prepared(provider,model,batch,baseline,auxiliary)
                changed_record,changed_arrays=_effect_arrays_and_record(dataset,changed,batch)
                kernel=core.export_source_kernels(organization,query) if baseline_kernel is not None else None
            differences={}
            for name,value in changed_arrays.items():
                if 'prediction' in name:
                    delta=np.asarray(value,dtype=np.float64)-np.asarray(arrays[name],dtype=np.float64)
                    mask=_effect_validity(name,arrays,delta.shape)
                    names=({'field_prediction':['u','v','p','omega','temperature'],
                            'interface_prediction':['temperature','q_proxy'],
                            'velocity_prediction_mps':['Ux','Uy','Uz']}).get(name)
                    differences[name]=_effect_magnitudes(delta,mask,names)
                    saved[mode+'__'+name]=value;saved[mode+'__delta_'+name]=delta
            if kernel is not None:
                differences['native_signed_K'],_=_kernel_effect_magnitudes(kernel-baseline_kernel,query,organization.scene)
                saved[mode+'__native_signed_K']=array(kernel[0])
            variants[mode]={'physical_reference_metrics':changed_record,'field_effect':differences,
                            'measured_fixed_read_fingerprint':control_fingerprint}
        restored=_replay_prepared(provider,model,batch,baseline,auxiliary)
        _,restored_arrays=_effect_arrays_and_record(dataset,restored,batch)
        restored_fingerprint=_fixed_read_fingerprint(core,organization,query)
        if restored_fingerprint!=fixed_fingerprint:raise ValueError('E control restoration changed native IDs, support or prepared states.')
        restoration={name:float(np.max(np.abs(np.asarray(value,dtype=np.float64)-np.asarray(arrays[name],dtype=np.float64))))
                     for name,value in restored_arrays.items() if 'prediction' in name}
        case=batch.case_keys[0];path=output/f'case_{case}.npz';write_arrays(path,saved)
        rows.append({'case_id':case,'baseline_physical_reference_metrics':reference,'variants':variants,
            'restoration_max_abs':restoration,'support_and_prepared_states_held_fixed':True,
            'baseline_fixed_read_fingerprint':fixed_fingerprint,'restored_fixed_read_fingerprint':restored_fingerprint,
            'source_column_correspondence_unchanged':True,'arrays':path.name,'arrays_sha256':file_sha256(path)})
        del baseline,auxiliary,organization,query,baseline_kernel,saved,restored
    result={'schema':'round9_E_read_effects_v1','available':True,'dataset':dataset,'split':split,
        'completed_epoch':int(payload['epoch']),'panel_sha256':panel['panel_sha256'],'cases':rows,
        'scope':'trained same-weight final-read interventions at fixed prepared M/E states; native-reference errors retained',
        'independent_environment_physics_evidence':'untested; geometry-derived E value changes are off-manifold controls',
        'physical_causality_claim':False,'optimizer_updates':0,'new_solver_attempts':0,'WindTEST':'locked'}
    write_json_new(output/'result.json',result);return result


def mixed_finite_difference(values,labels,quartet):
    if len(quartet)!=4 or any(label not in labels for label in quartet):
        raise ValueError('A physical mixed response requires all four compatible saved states.')
    baseline,a,b,combined=(np.asarray(values[labels.index(label)],dtype=np.float64) for label in quartet)
    return combined-a-b+baseline


def _response_structure(atlas,index,device):
    metadata=json.loads(str(atlas['family_metadata_json']));config=metadata['case_config'];count=atlas['heating'].shape[1]
    if metadata.get('source_dataset_split')!='train':raise ValueError('Saved response replay is restricted to inventoried original TRAIN.')
    if any(float(config['thermal'][name])!=0 for name in ('t_in','t_wall')):raise ValueError('Thermal response requires zero boundary/background heat.')
    def tensor(value):return torch.as_tensor(np.asarray(value,dtype=np.float32),device=device)
    material=[[config['runtime']['nu'],config['thermal']['solid_alpha'],config['thermal']['fluid_alpha'],
               config['thermal']['solid_k'],config['thermal']['fluid_k'],config['domain']['module_radius']]]
    labels=np.asarray(atlas['active_module_ids']).tolist()
    def original_column(label):
        parts=str(label).split(':')
        if not ((len(parts)==2 and parts[0]=='M') or (len(parts)==3 and parts[1]=='module')):
            raise ValueError('Unrecognized original module ID namespace.')
        return int(parts[-1])
    try:source_ids=[original_column(label) for label in labels]
    except ValueError as error:raise ValueError('Saved module IDs must preserve their original integer source columns.') from error
    if len(source_ids)!=count or len(set(source_ids))!=count or min(source_ids)<0:
        raise ValueError('Saved active module IDs must be unique original source columns.')
    structure={'module_centers':tensor(atlas['module_centers_xy'][index:index+1]),
        'module_present':tensor(np.ones((1,count))),'module_source_ids':torch.tensor(source_ids,device=device)[None],
        'material_params':tensor(material),'re':tensor([[config['flow']['re']]]),'u_in':tensor([[config['flow']['u_in']]]),
        'domain_length_x':tensor([[config['domain']['lx']]]),'domain_length_y':tensor([[config['domain']['ly']]]),
        't_in':tensor([[0.]]),'t_wall':tensor([[0.]])}
    fluid=tensor(atlas['fluid_fields_query_features'][None,:,:2])
    flat=atlas['solid_temperature_query_features'][:,:2];per_module=len(flat)//count
    if len(flat)%count or not all(np.array_equal(flat[:per_module],flat[i*per_module:(i+1)*per_module]) for i in range(count)):
        raise ValueError('Saved material receivers lost native module/local-coordinate correspondence.')
    for role in ('interface','solid_temperature'):
        ids=atlas[role+'_receiver_module_ids'];expected=np.repeat(atlas['active_module_ids'],len(ids)//count)
        if not np.array_equal(ids,expected):raise ValueError('Saved role source IDs changed order.')
    return structure,fluid,tensor(flat[:per_module][None]),tensor(atlas['heating'][index:index+1])


def _response_role_arrays(output):
    return {'fluid_fields':array(output['pred_field'][0]),
            'interface':array(output['pred_interface'][0]).reshape(-1,2),
            'solid_temperature':array(output['pred_internal_temperature'][0]).reshape(-1,1)}


def _per_channel_metrics(prediction,target,valid,names):
    mask=np.asarray(valid,dtype=bool)
    if mask.ndim==1:mask=mask[:,None]
    mask=np.broadcast_to(mask,np.asarray(target).shape)
    result={}
    for i,name in enumerate(names):
        row=metric(prediction[:,i],target[:,i],mask[:,i])
        selected=mask[:,i]&np.isfinite(target[:,i]);p=np.asarray(prediction[:,i],dtype=np.float64)[selected];t=np.asarray(target[:,i],dtype=np.float64)[selected]
        pn,tn=float(np.linalg.norm(p)),float(np.linalg.norm(t));dot=float(np.dot(p,t))
        row.update(prediction_RMS=float(np.sqrt(np.mean(p*p))),reference_RMS=float(np.sqrt(np.mean(t*t))),
            cosine_alignment=dot/(pn*tn) if pn>0 and tn>0 else None,
            least_squares_response_gain=dot/(tn*tn) if tn>0 else None,
            signed_mean_prediction=float(p.mean()),signed_mean_reference=float(t.mean()),
            sign_agreement_nonzero_reference=float(np.mean(np.sign(p[t!=0])==np.sign(t[t!=0]))) if np.any(t!=0) else None)
        result[str(name)]=row
    return result


@torch.no_grad()
def evaluate_saved_responses(model,provider,payload,inventory,output,*,family_ids=None):
    if not hasattr(model,'prepare_native'):raise ValueError('Only the maintained Thermal native affine capability supports this saved replay.')
    if inventory.get('schema')!='round9_existing_physical_response_inventory_v1':raise ValueError('Expected verified Round9 physical inventory.')
    output=Path(output)
    if output.exists():raise FileExistsError('Saved physical-response evidence is immutable.')
    output.mkdir(parents=True);model.eval();device=next(model.parameters()).device;core=model.core;families=[]
    for source in inventory['thermal_families']:
        if family_ids is not None and source['family_id'] not in family_ids:continue
        if file_sha256(source['path'])!=source['sha256']:raise ValueError('Saved response source checksum changed.')
        with np.load(source['path'],allow_pickle=False) as handle:atlas={name:handle[name] for name in handle.files}
        labels=atlas['all_labels'].tolist();predictions={};frozen_predictions={};baseline_routes=None;baseline_prepared=None
        saved={'labels':np.asarray(labels),'native_module_ids':atlas['active_module_ids'],
               'module_centers_xy':atlas['module_centers_xy'],'heating':atlas['heating']}
        absolute=[]
        for index,label in enumerate(labels):
            structure,fluid,local,heat=_response_structure(atlas,index,device)
            prepared=model.prepare_native(structure,fluid,local_query_points=local,ntheta=64,scene_key=(source['family_id'],label))
            prediction=_response_role_arrays(model.apply_native(prepared,heat))
            predictions[label]=prediction
            if label=='baseline':
                baseline_prepared=prepared
                if hasattr(core,'capture_read_routes'):baseline_routes=core.capture_read_routes(prepared.organization,prepared.query_batch)
            fixed=prediction
            if baseline_routes is not None and label in atlas['position_labels'].tolist():
                fixed_output=core.decode_counterfactual_routes(prepared.scene_batch,prepared.query_batch,baseline_routes,heat)
                fixed=_response_role_arrays(model._extract_native_roles(prepared,fixed_output.thermal_temperature,fixed_output.nonlinear_field))
            frozen_predictions[label]=fixed
            record={'label':label,'roles':{}}
            for role,value in prediction.items():
                target=atlas[role+'_values'][index];valid=atlas[role+'_valid_mask'][index]
                names=atlas[role+'_channel_names'].tolist()
                record['roles'][role]=_per_channel_metrics(value,target,valid,names)
            count=atlas['heating'].shape[1];solid=prediction['solid_temperature'].reshape(count,-1)
            reference=atlas['solid_temperature_values'][index].reshape(count,-1)
            valid=atlas['solid_temperature_valid_mask'][index].reshape(count,-1)
            pp=np.where(valid,solid,-np.inf).max(-1);tt=np.where(valid,reference,-np.inf).max(-1)
            record['recomputed_module_peaks']={'prediction':pp.tolist(),'target':tt.tolist(),
                'saved_scalar_reference':atlas['module_peak_temperature'][index].tolist(),
                'error':(pp-tt).tolist(),'rmse':float(np.sqrt(np.mean((pp-tt)**2)))}
            absolute.append(record)
        responses=[];interactions=[]
        for label in labels:
            if label=='baseline':continue
            scope='heat' if label.startswith('heat_') else 'geometry'
            record={'label':label,'perturbation':scope,'roles':{},'fixed_route_roles':{}}
            for role in predictions[label]:
                reference=atlas[role+'_values'][labels.index(label)].astype(np.float64)-atlas[role+'_values'][labels.index('baseline')].astype(np.float64)
                valid=atlas[('heat' if scope=='heat' else 'position')+'_common_'+{'fluid_fields':'fluid','interface':'interface','solid_temperature':'solid'}[role]+'_mask']
                delta=predictions[label][role].astype(np.float64)-predictions['baseline'][role].astype(np.float64)
                fixed_delta=frozen_predictions[label][role].astype(np.float64)-frozen_predictions['baseline'][role].astype(np.float64)
                names=atlas[role+'_channel_names'].tolist()
                record['roles'][role]=_per_channel_metrics(delta,reference,valid,names)
                record['fixed_route_roles'][role]=_per_channel_metrics(fixed_delta,reference,valid,names)
                saved[label+'__'+role+'__reference_delta']=reference
                saved[label+'__'+role+'__prediction_delta']=delta
                saved[label+'__'+role+'__fixed_route_delta']=fixed_delta
                saved[scope+'__'+role+'__valid']=valid
            endpoint=absolute[labels.index(label)]['recomputed_module_peaks'];base_peaks=absolute[labels.index('baseline')]['recomputed_module_peaks']
            peak_delta=np.asarray(endpoint['prediction'])-np.asarray(base_peaks['prediction'])
            peak_reference=np.asarray(endpoint['target'])-np.asarray(base_peaks['target'])
            record['recomputed_module_peak_changes']={'module_ids':atlas['active_module_ids'].tolist(),
                'prediction':peak_delta.tolist(),'reference':peak_reference.tolist(),
                'metrics':metric(peak_delta,peak_reference,np.ones_like(peak_reference,dtype=bool)),
                'definition':'difference of maxima recomputed independently at each endpoint, not max of a difference'}
            responses.append(record)
        for quartet in (('baseline','i_plus','j_plus','pp'),('baseline','i_minus','j_minus','mm')):
            item={'quartet':list(quartet),'roles':{},'fixed_route_roles':{}}
            for role in predictions['baseline']:
                actual=mixed_finite_difference(atlas[role+'_values'],labels,quartet)
                rebuilt=mixed_finite_difference(np.stack([predictions[label][role] for label in labels]),labels,quartet)
                fixed=mixed_finite_difference(np.stack([frozen_predictions[label][role] for label in labels]),labels,quartet)
                valid=atlas['position_common_'+{'fluid_fields':'fluid','interface':'interface','solid_temperature':'solid'}[role]+'_mask']
                names=atlas[role+'_channel_names'].tolist()
                item['roles'][role]=_per_channel_metrics(rebuilt,actual,valid,names)
                item['fixed_route_roles'][role]=_per_channel_metrics(fixed,actual,valid,names)
                prefix=quartet[-1]+'__mixed__'+role
                saved[prefix+'__reference']=actual;saved[prefix+'__prediction']=rebuilt;saved[prefix+'__fixed_route']=fixed
            interactions.append(item)
        # Promoted affine increments isolate finite heat response from FP32 endpoint cancellation.
        promoted=[]
        for label in ('heat_transfer_plus','heat_transfer_minus'):
            index=labels.index(label);base=labels.index('baseline')
            increment=torch.as_tensor(atlas['heating'][index:index+1].astype(np.float64)-atlas['heating'][base:base+1].astype(np.float64),device=device)
            values=_response_role_arrays(model.apply_increment(baseline_prepared,increment,accumulation_dtype=torch.float64))
            item={'label':label,'roles':{}}
            for role,value in values.items():
                target=atlas[role+'_values'][index].astype(np.float64)-atlas[role+'_values'][base].astype(np.float64)
                valid=atlas['heat_common_'+{'fluid_fields':'fluid','interface':'interface','solid_temperature':'solid'}[role]+'_mask']
                item['roles'][role]=_per_channel_metrics(value,target,valid,atlas[role+'_channel_names'].tolist())
                saved[label+'__'+role+'__promoted_increment']=value
            promoted.append(item)
        path=output/f'family_{source["family_id"]}.npz';write_arrays(path,saved)
        families.append({'family_id':source['family_id'],'source_path':source['path'],'source_sha256':source['sha256'],
            'source_split':source['source_split'],'response_training_exposure':source['response_training_exposure'],
            'independent_heating_rank':source['stored_heating_rank_at_absolute_tolerance_1e-6'],
            'absolute_states':absolute,'finite_responses':responses,'geometry_geometry_interactions':interactions,
            'promoted_affine_heat_increments':promoted,'fixed_route_available':baseline_routes is not None,
            'arrays':path.name,'arrays_sha256':file_sha256(path)})
        if source['family_id']=='0001':
            independent=_evaluate_independent_heat_null(model,atlas,inventory['independent_heat_null'],output,device)
        del predictions,frozen_predictions,saved,baseline_prepared,baseline_routes
    result={'schema':'round9_saved_physical_response_replay_v1','dataset':'thermal','completed_epoch':int(payload['epoch']),
        'arm':checkpoint_arm(payload),
        'families':families,'independent_heat_null':locals().get('independent'),
        'geometry_plus_heat_interaction':{'available':False,'reason':'compatible combined physical endpoint missing'},
        'independent_environment_physics_evidence':{'available':False,'reason':'no independently excited E physical field in current inventory'},
        'scope':'existing exposed original TRAIN solver endpoints; baseline-to-plus heat delta supervised',
        'learned_routing_is_physical_causality':False,'optimizer_updates':0,'new_solver_attempts':0,'WindTEST':'locked'}
    write_json_new(output/'result.json',result);return result


def _evaluate_independent_heat_null(model,atlas,source,output,device):
    if file_sha256(source['array_path'])!=source['array_sha256'] or file_sha256(source['json_path'])!=source['json_sha256']:
        raise ValueError('Independent heat-null physical source checksum changed.')
    labels=source['labels'];inputs=[]
    for label in labels:
        config=json.loads((Path(source['full_four_state_raw_paths'][label])/'case_config.json').read_text())
        inputs.append((np.asarray(config['layout']['centers'],dtype=np.float32),np.asarray(config['layout']['heat_powers'],dtype=np.float64)))
    base=atlas['all_labels'].tolist().index('baseline')
    if any(not np.array_equal(centers,atlas['module_centers_xy'][base]) for centers,_ in inputs):
        raise ValueError('Independent heat-null geometry does not match its native baseline.')
    structure,fluid,local,_=_response_structure(atlas,base,device)
    prepared=model.prepare_affine_application(model.prepare_native(structure,fluid,local_query_points=local,ntheta=64,
                                                                   scene_key=('independent_heat_null','0001')))
    predictions=[_response_role_arrays(model.apply_native(prepared,torch.as_tensor(heat[None],device=device),
                     accumulation_dtype=torch.float64)) for _,heat in inputs]
    with np.load(source['array_path'],allow_pickle=False) as handle:reference={name:handle[name] for name in handle.files}
    arrays={};metrics={}
    for role in predictions[0]:
        mixed=mixed_finite_difference(np.stack([value[role] for value in predictions]),labels,labels)
        actual=reference[role+'_mixed_delta'];valid=reference[role+'_common_mask'];names=reference[role+'_channel_names'].tolist()
        metrics[role]=_per_channel_metrics(mixed,actual,valid,names)
        arrays[role+'__prediction_mixed_delta']=mixed;arrays[role+'__reference_mixed_delta']=actual;arrays[role+'__valid']=valid
    path=output/'independent_heat_null.npz';write_arrays(path,arrays)
    return {'labels':labels,'independent_heating_rank':int(np.linalg.matrix_rank(np.stack([heat-inputs[0][1] for _,heat in inputs]),tol=1e-6)),
        'physical_reference_interpretation':source['physical_interpretation'],
        'prediction_scope':'promoted float64 affine contraction of fixed float32 learned coefficients; not ordinary FP32 endpoint subtraction',
        'roles':metrics,'arrays':path.name,'arrays_sha256':file_sha256(path)}


def _continuous_query(provider,batch,organization,template,coordinates,ids):
    """Recompute geometry-owned query features and local support at new coordinates."""
    from honf_forward_core.content_hypergraph.contracts import QueryBatch
    features=None
    if template.features is not None:
        # Target-free Wind scene inputs retain bounds, not the source case's
        # SupportGeometry object. Reuse the native feature transform unchanged.
        scene_input=batch.scene_inputs[0]
        support=SimpleNamespace(lower_D=scene_input.support_lower_D,
            upper_D=scene_input.support_upper_D,extent_D=scene_input.support_extent_D)
        values=provider.adapter.query_features(array(coordinates[0]),support)
        features=torch.as_tensor(values[None],device=coordinates.device,dtype=coordinates.dtype)
    return QueryBatch(coordinates,features=features,query_mask=torch.ones(coordinates.shape[:2],dtype=torch.bool,device=coordinates.device),
        query_ids=ids,scene_key=organization.scene.scene_key,coordinate_frame_key=organization.scene.coordinate_frame_key,
        receiver_key=('round9_continuous_coordinate_seam_v1',))


def _continuous_values(provider,model,batch,organization,query):
    core=getattr(model,'core',model)
    controls=batch.receivers.heat if hasattr(model,'prepare_native') else None
    output=core.decode(organization,query,controls,trace_request={'include':['native_dispatch_seed','joint_hard_support']})
    if hasattr(model,'prepare_native'):
        flow=output.nonlinear_field*model.field_std[:3]+model.field_mean[:3]
        physical=torch.cat((flow,output.thermal_temperature[...,None]*model.field_std[4]),-1)
    else:physical=provider.adapter.physical_velocity(output.nonlinear_field,query.coordinates)[0]
    trace=output.trace or {}
    return (array(physical[0]),None if 'joint_hard_support' not in trace else array(trace['joint_hard_support'][0]),
            None if 'native_dispatch_seed' not in trace else array(trace['native_dispatch_seed'][0]))


def _literal_grid_probes(model,batch,point):
    """A geometry-selected x interpolation-stencil crossing, independent of routing."""
    length=float(model.scene_batch(batch.scene_inputs.structure,scene_key=batch.scene_inputs.case_ids).coordinate_scale[0,0])
    dx=length/model.nx;center=point.clone()
    center[...,0]=(torch.round(center[...,0]/dx-.5)+.5)*dx
    probes=[];saved={'native_grid_crossing_center':array(center[0])}
    for fraction in (1e-3,1e-4,1e-5,1e-6):
        step=fraction*dx;offset=torch.zeros_like(center);offset[...,0]=step
        coordinates=torch.cat((center-offset,center+offset),1)
        try:
            prepared=model.prepare_native(batch.scene_inputs.structure,coordinates,local_query_points=None,
                ntheta=batch.receivers.ntheta,scene_key=batch.scene_inputs.case_ids)
        except ValueError as error:
            if 'Native curl is defined only on native cell-centred receivers' not in str(error):raise
            # Preserve the adapter's physical boundary. Adjacent native cells
            # are a supported diagnostic, but are not a shrinking seam limit.
            adjacent=torch.cat((point,point+point.new_tensor([dx,0.])[None,None]),1)
            prepared=model.prepare_native(batch.scene_inputs.structure,adjacent,local_query_points=None,
                ntheta=batch.receivers.ntheta,scene_key=batch.scene_inputs.case_ids)
            values=model.apply_native(prepared,batch.receivers.heat)['pred_field'][0]
            saved.update(supported_adjacent_cell_coordinates=array(adjacent[0]),supported_adjacent_cell_fields=array(values))
            return {'available':False,'reason':str(error),'public_boundary_rejection_checked':True,
                'scope':'literal native-grid shrinking crossing unavailable through the maintained curl API',
                'supported_adjacent_cell_diagnostic':{'coordinate_gap':dx,'physical_field_difference':array(values[1]-values[0]).tolist(),
                    'channels':['u','v','p','omega','temperature'],'valid':array(prepared.stencils['fluid'].valid[0]).tolist(),
                    'interpretation':'difference between adjacent legal native cell centres; not a continuity limit'}},saved
        values=model.apply_native(prepared,batch.receivers.heat)['pred_field'][0]
        stencil=prepared.stencils['fluid'];native_ids=prepared.union.grid_indices[0][stencil.indices[0]]
        probes.append({'requested_step':step,'step_over_native_dx':fraction,
            'actual_coordinate_gap':float((coordinates[:,1]-coordinates[:,0]).abs().max()),
            'coordinates_distinct_in_fp32':not torch.equal(coordinates[:,0],coordinates[:,1]),
            'stencil_native_ids':array(native_ids).tolist(),'stencil_ids_differ':not torch.equal(native_ids[0],native_ids[1]),
            'physical_field_jump':array(values[1]-values[0]).tolist(),'valid':array(stencil.valid[0]).tolist()})
        saved[f'native_grid_fraction_{fraction:g}__coordinates']=array(coordinates[0])
        saved[f'native_grid_fraction_{fraction:g}__physical_fields']=array(values)
    return {'channels':['u','v','p','omega','temperature'],'native_dx':dx,'probes':probes,
            'scope':'literal native interpolation-stencil crossing; no target assigned to moved receivers'},saved


@torch.no_grad()
def evaluate_seams(model,provider,payload,panel,output,*,crossings=None):
    """Input-only hard-read crossings, distinct from Thermal native interpolation."""
    dataset=panel['dataset'];validate_native_panel(panel,provider,dataset);model.eval();core=getattr(model,'core',model)
    if crossings is not None:
        if (crossings.get('schema')!='round9_spatial_seam_probe_v1' or crossings.get('dataset')!=dataset or
            crossings.get('panel_sha256')!=panel['panel_sha256'] or crossings.get('completed_epoch')!=int(payload['epoch'])):
            raise ValueError('Shared crossings require the same literal epoch, dataset and frozen panel.')
        shared={str(row['case_id']):row for row in crossings['cases']}
    else:shared=None
    output=Path(output)
    if output.exists():raise FileExistsError('Spatial seam evidence is immutable.')
    output.mkdir(parents=True);rows=[]
    for batch in native_batches(provider,dataset,panel,'train',representatives_only=True):
        prediction,auxiliary=predict_physical(provider,model,batch,int(payload['epoch']))
        organization,template=_prepared_native(provider,model,batch,prediction,auxiliary)
        scale=organization.frame_scale[0];origin=organization.frame_origin[0]
        coordinates=template.coordinates;valid=(template.query_mask[0] if template.query_mask is not None else torch.ones(coordinates.shape[1],dtype=torch.bool,device=coordinates.device))
        margin=.02*scale[0]
        valid=valid & (coordinates[0,:,0]>origin[0]+margin) & (coordinates[0,:,0]<origin[0]+scale[0]-margin)
        eligible=valid.nonzero(as_tuple=False).flatten()
        selected=eligible[torch.linspace(0,len(eligible)-1,min(64,len(eligible)),device=eligible.device).long()] if len(eligible) else eligible
        row={'case_id':batch.case_keys[0],'selection':'first64 evenly spaced valid input receiver IDs with x margin; no target/errors used',
             'candidate_query_ids':array(template.query_ids[0,selected]).tolist(),'probe_count':len(selected)}
        if not len(selected):row['available']=False;row['reason']='no valid frozen probe';rows.append(row);continue
        grid_saved={}
        if dataset=='thermal':
            row['literal_native_grid_stencil_crossing'],grid_saved=_literal_grid_probes(model,batch,coordinates[:,selected[:1]])
            gridpath=output/f'case_{batch.case_keys[0]}_native_grid.npz';write_arrays(gridpath,grid_saved)
            row['literal_native_grid_arrays']={'path':gridpath.name,'sha256':file_sha256(gridpath)}
        if shared is not None:
            source=shared.get(str(batch.case_keys[0]))
            if source is None:raise ValueError('Shared crossing receipt lost a frozen representative case.')
            if not source['available']:
                row.update(available=False,reason='source crossing unavailable',source_reason=source['reason']);rows.append(row);continue
            center=coordinates.new_tensor(source['crossing_center'])[None]
            qid=template.query_ids.new_tensor([[source['query_id']]])
            steps=np.asarray(source['shrinking_steps_native_coordinate_units'],dtype=np.float64)
            if center.shape!=(1,1,template.coordinates.shape[-1]) or not np.all(np.isfinite(steps)&(steps>0)):
                raise ValueError('Shared crossing coordinates or shrinking steps are invalid.')
            bracket_width=source['crossing_bracket_width']
        else:
            centers=coordinates[:,selected];ids=template.query_ids[:,selected]
            offset=torch.zeros_like(centers);offset[...,0]=margin
            left_query=_continuous_query(provider,batch,organization,template,centers-offset,ids)
            right_query=_continuous_query(provider,batch,organization,template,centers+offset,ids)
            _,ls,ld=_continuous_values(provider,model,batch,organization,left_query)
            _,rs,rd=_continuous_values(provider,model,batch,organization,right_query)
            if ls is None or rs is None:
                row.update(available=False,reason='dense core has no discrete typed read switch; replay an H/S receipt with --crossings')
                rows.append(row);continue
            changed=np.any(ls!=rs,axis=-1)
            if ld is not None:changed |= ld!=rd
            switches=np.flatnonzero(changed)
            if not len(switches):
                row.update(available=False,reason='no support/dispatch switch on frozen64 probes; shared H/S crossings can be replayed')
                rows.append(row);continue
            index=int(switches[0]);lo=(centers-offset)[:,index:index+1].clone();hi=(centers+offset)[:,index:index+1].clone();qid=ids[:,index:index+1]
            left_support=ls[index].copy();left_seed=None if ld is None else int(ld[index])
            for _ in range(24):
                mid=(lo+hi)*.5
                _,support,dispatch=_continuous_values(provider,model,batch,organization,
                    _continuous_query(provider,batch,organization,template,mid,qid))
                same=np.array_equal(support[0],left_support) and (dispatch is None or int(dispatch[0])==left_seed)
                if same:lo=mid
                else:hi=mid
            center=(lo+hi)*.5;steps=np.asarray((1e-3,1e-4,1e-5,1e-6,1e-7))*float(scale[0])
            bracket_width=float((hi-lo).abs().max())
        saved={'crossing_center':array(center[0]),'shrinking_steps_native_coordinate_units':steps,'query_id':array(qid[0])}
        probes=[]
        for step in steps:
            delta=torch.zeros_like(center);delta[...,0]=float(step)
            points=torch.cat((center-delta,center+delta),1);query=_continuous_query(provider,batch,organization,template,points,qid.expand(-1,2))
            values,support,dispatch=_continuous_values(provider,model,batch,organization,query)
            probes.append({'step':float(step),'coordinate_gap':2*float(step),'physical_field_jump':(values[1]-values[0]).tolist(),
                'pooled_mixed_unit_jump_norm':float(np.linalg.norm(values[1]-values[0])),
                'coordinates_distinct_in_fp32':not torch.equal(points[:,0],points[:,1]),
                'supports_differ':None if support is None else not np.array_equal(support[0],support[1]),
                'dispatch_seeds':None if dispatch is None else dispatch.tolist()})
            saved[f'step_{step:.9g}__coordinates']=array(points[0]);saved[f'step_{step:.9g}__physical_fields']=values
            if support is not None:saved[f'step_{step:.9g}__support']=support
        row.update(available=True,query_id=int(qid[0,0]),crossing_kind='query dispatch or final support/protection switch',
            continuous_core_channels=['u','v','p','temperature'] if dataset=='thermal' else ['Ux','Uy','Uz'],
            continuous_core_units=['dataset velocity','dataset velocity','dataset pressure','dataset temperature'] if dataset=='thermal' else ['m/s']*3,
            crossing_center=array(center[0]).tolist(),shrinking_steps_native_coordinate_units=steps.tolist(),
            crossing_bracket_width=bracket_width,shrinking_probes=probes,
            crossing_scope='replay of exact shared crossing coordinates' if shared is not None else 'input-only switch discovery',
            literal_Thermal_native_grid=({'available':False,
                'reason':'maintained physical curl API accepts native cell-centred receivers only; continuous-core probes are a separate scope'} if dataset=='thermal' else None))
        path=output/f'case_{batch.case_keys[0]}.npz';write_arrays(path,saved)
        row.update(arrays=path.name,arrays_sha256=file_sha256(path));rows.append(row)
        del prediction,auxiliary,organization,template,saved
    result={'schema':'round9_spatial_seam_probe_v1','dataset':dataset,'completed_epoch':int(payload['epoch']),
        'panel_sha256':panel['panel_sha256'],'cases':rows,'scope':'input-selected model discontinuities; no physical target assigned to perturbed receivers',
        'shared_crossings_sha256':None if crossings is None else canonical_sha256(crossings),
        'globally_smooth_inverse_claim':False,'optimizer_updates':0,'new_solver_attempts':0,'WindTEST':'locked'}
    write_json_new(output/'result.json',result);return result
