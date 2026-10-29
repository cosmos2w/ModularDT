"""Input-only composed-access birth, followed by fixed-TRAIN gradient calibration."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import torch
from .continuous_joint_protocol import file_sha256
from .joint_native_calibration import _capture_calibration_rng_state,_restore_calibration_rng_state


def apply_continuous_calibration(core,recipe):
    binding=recipe['calibration']
    if file_sha256(binding['path'])!=binding['sha256']:raise ValueError('Continuous calibration receipt changed.')
    value=json.loads(Path(binding['path']).read_text())
    for key,expected in dict(schema='round10_continuous_calibration_v1',dataset=recipe['dataset'],arm=recipe['arm'],
        source_checkpoint_sha256=recipe['source_checkpoint_sha256'],optimizer_updates=0).items():
        if value.get(key)!=expected:raise ValueError('Continuous calibration source/family differs.')
    if recipe['arm'].startswith('U10-') and value.get('input_feature_contract_sha256')!=core.local_input_contract['sha256']:
        raise ValueError('Unified calibration local-input capability differs.')
    if recipe['arm']=='U10-S' and value.get('context_initialization_sha256')!=(recipe.get('context_initialization') or {}).get('sha256'):
        raise ValueError('U-S calibration shared contextual initialization differs.')
    if not 1<=len(value['batches'])<=8 or value['target_composed_access']!=.98:
        raise ValueError('Continuous birth calibration exposure/target differs.')
    if set(value.get('birth',{}))!={'M','E'} or any(row.get('status')!='calibrated' or abs(row.get('mean_composed_access',0)-.98)>1e-4 for row in value['birth'].values()):
        raise ValueError('Continuous composed-access birth was not qualified near open.')
    core.router.entmax_temperature.fill_(value['temperature']['selected'])
    named=dict(core.named_parameters())
    expected={name for name,p in named.items() if name.startswith('router.') and name.endswith('.bias')}
    if set(value['biases'])!=expected:raise ValueError('Continuous bias inventory differs.')
    with torch.no_grad():
        for name,bias in value['biases'].items():named[name].fill_(bias)
        if set(value['typed_gradient'])!=set(core.joint_constraint_kinds()):raise ValueError('Continuous dual inventory differs.')
        for kind,row in value['typed_gradient'].items():
            if row['status']!='calibrated' or not all(math.isfinite(row[k]) and row[k]>0 for k in ('lambda0','native_gradient_norm','work_gradient_norm')):
                raise ValueError('Unsupported/nonfinite continuous gradient calibration.')
            if not math.isclose(row['lambda0'],.05*row['native_gradient_norm']/row['work_gradient_norm'],rel_tol=1e-10):
                raise ValueError('Continuous gradient balance formula differs.')
            core.joint_constraint_duals()[kind].set_calibrated_lambda0(row['lambda0'])
    return value


def calibrate_continuous_joint_task(task,*,max_batches=2):
    """No optimizer updates; restore every model clock, RNG and provider state."""
    if type(max_batches) is not int or not 1<=max_batches<=8:raise ValueError('Birth calibration permits <=8 fixed TRAIN batches.')
    model,provider,engine=task['model'],task['provider'],task['engine'];core=getattr(model,'core',model)
    recipe=task['identity']['recipe']
    if recipe['arm']=='U10-S' and recipe.get('context_initialization') is None:
        raise ValueError('U-S birth must reuse its matched U-H contextual initialization.')
    state=copy.deepcopy(core.state_dict());rng=_capture_calibration_rng_state()
    modes={module:module.training for module in model.modules()}
    provider_state=copy.deepcopy(provider.training_state_dict()) if hasattr(provider,'training_state_dict') else None
    hooks=[];samples=[];latest={}
    try:
        cases,dataset_id=engine.epoch_case_order(provider,3001);batches=[];macro_indices=[]
        for index,sl in enumerate(engine.epoch_update_slices(len(cases))):
            additions=engine.prepare_update_microbatches(provider,arm='A2-D',phase='joint',epoch=3001,
                update_index=index,update_cases=cases[sl],sampling_dataset_id=dataset_id)
            batches.extend(additions);macro_indices.extend([index]*len(additions))
            if len(batches)>=max_batches:break
        batches=batches[:max_batches];macro_indices=macro_indices[:max_batches]
        receipt=dict(schema='round10_continuous_calibration_v1',dataset=recipe['dataset'],arm=recipe['arm'],
            source_checkpoint_sha256=recipe['source_checkpoint_sha256'],optimizer_updates=0,target_composed_access=.98,
            batches=[dict(macro_index_zero_based=i,case_keys=[repr(k) for k in b.case_keys],
                query_sampling_sha256=b.auxiliary.get('query_sampling_sha256')) for i,b in zip(macro_indices,batches)],
            temperature={},biases={},birth={},typed_gradient={})
        if recipe['arm']=='U10-S':receipt['context_initialization_sha256']=recipe['context_initialization']['sha256']
        from honf_forward_core.continuous_joint.primitives import positive_quadratic_gate,entmax15
        from honf_forward_core.joint_native.routing import typed_query_logits
        def incidence_capture(name):
            def hook(module,inputs,output):latest[name]=(output.detach()-module.bias.detach(),inputs[1].detach())
            return hook
        for name,calibration in core.router.incidence.routes.items():
            hooks.append(calibration.register_forward_hook(incidence_capture(name)))
        def router_capture(module,inputs):
            rm,re,records,envelope,qmask,incidence=inputs
            samples.append((rm.detach(),re.detach(),records,envelope.detach(),qmask.detach(),
                {key:(a.clone(),allowed.clone()) for key,(a,allowed) in latest.items()}))
        hooks.append(core.router.register_forward_pre_hook(router_capture))
        model.eval()
        with torch.no_grad():
            for index,batch in zip(macro_indices,batches):
                core.joint_begin_update(3001,index);scene=provider.make_scene(batch.scene_inputs)
                provider.predict_native(model,scene,batch.receivers,'A2-D','joint',epoch=3001,
                    temperature=engine.config.temperature_for_epoch(3001))
        for h in hooks:h.remove()
        hooks=[]
        # Temperature depends solely on the sealed pre-E scores and typed masks.
        candidates=[]
        for temperature in (.5,1.,2.,4.):
            counts=[];types=[];entropies=[]
            for rm,re,records,envelope,qmask,blocks in samples:
                logits=typed_query_logits(rm,re,records,core.router.dispatch_parameters,qmask)
                p=entmax15(logits/temperature,records.present[:,None]&qmask[...,None])
                active=p>0;counts.extend(active.sum(-1)[qmask].cpu().tolist())
                types.extend(active[...,:records.module_count].sum(-1)[qmask].cpu().tolist())
                entropies.extend((-(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1))[qmask].cpu().tolist())
            median=float(torch.tensor(counts,dtype=torch.double).median()) if counts else 0.
            candidates.append(dict(temperature=temperature,median_positive_seeds=median,positive_seed_counts=counts,
                module_seed_counts=types,entropy=entropies))
        selected=min(candidates,key=lambda row:(abs(row['median_positive_seeds']-4),abs(math.log2(row['temperature']))))
        selection='closest_median_to_four_then_log2_distance_to_one'
        if recipe['arm'].startswith('U10-'):
            frozen=recipe['model_options'].get('entmax_temperature')
            if frozen not in (.5,1.,2.,4.):
                raise ValueError('Unified calibration requires the already sealed per-dataset Track-B entmax temperature.')
            selected=next(row for row in candidates if row['temperature']==frozen)
            selection='inherited_fixed_dataset_track_b_calibration'
            receipt['input_feature_contract_sha256']=core.local_input_contract['sha256']
        core.router.entmax_temperature.fill_(selected['temperature'])
        receipt['temperature']=dict(selected=selected['temperature'],candidates=candidates,selection=selection)
        # Each donor type has one shared additive birth adjustment across its
        # existing typed calibrators. Composed g, including local envelopes,
        # determines the fitted mean; no target/field value enters this search.
        def type_values(kind,shift):
            values=[];positive=total=0
            for rm,re,records,envelope,qmask,blocks in samples:
                m=records.module_count;sl=slice(0,m) if kind=='M' else slice(m,None)
                eligible=records.eligible[:,None,sl]&qmask[...,None]
                if core.routing_mode=='direct':
                    calibration=core.router.direct_parameters.module if kind=='M' else core.router.direct_parameters.environment
                    raw=rm if kind=='M' else re
                    g=positive_quadratic_gate(calibration(raw,eligible)+shift)*eligible
                else:
                    b={}
                    for route in ('mm','me','em','ee'):
                        pre,allowed=blocks[route]
                        bias=core.router.incidence.routes[route].bias.detach()
                        adjustment=shift if (route[1]=='m')==(kind=='M') else 0
                        gate=positive_quadratic_gate(pre+bias+adjustment)*allowed
                        if route[0]==route[1]:
                            donor=records.eligible[:,:m] if route[1]=='m' else records.eligible[:,m:]
                            present=records.present[:,:m] if route[0]=='m' else records.present[:,m:]
                            eye=torch.eye(gate.shape[-1],device=gate.device,dtype=torch.bool)[None]
                            gate=torch.where(eye & present[:,:,None]&donor[:,None],1.,gate)
                        b[route]=gate
                    matrix=torch.cat((torch.cat((b['mm'],b['me']),-1),torch.cat((b['em'],b['ee']),-1)),1)
                    logits=typed_query_logits(rm,re,records,core.router.dispatch_parameters,qmask)
                    p=entmax15(logits/core.router.entmax_temperature,records.present[:,None]&qmask[...,None])
                    g=torch.bmm(p,matrix)[...,sl]*eligible
                if kind=='M':g=envelope+(1-envelope)*g
                selected_g=g[eligible]
                values.append(selected_g)
                positive+=int((selected_g>0).sum());total+=selected_g.numel()
            values=torch.cat(values)
            return float(values.mean()),positive,total
        with torch.no_grad():
            for kind in ('M','E'):
                lower,upper=-128.,128.
                for _ in range(36):
                    midpoint=(lower+upper)/2
                    if type_values(kind,midpoint)[0]<.98:lower=midpoint
                    else:upper=midpoint
                shift=(lower+upper)/2
                mean,positive,total=type_values(kind,shift)
                if core.routing_mode=='direct':
                    calibration=core.router.direct_parameters.module if kind=='M' else core.router.direct_parameters.environment
                    calibration.bias.add_(shift)
                else:
                    for name,calibration in core.router.incidence.routes.items():
                        if (name[1]=='m')==(kind=='M'):calibration.bias.add_(shift)
                receipt['birth'][kind]=dict(mean_composed_access=mean,positive_rows=positive,possible_rows=total,
                    status='calibrated' if abs(mean-.98)<1e-4 else 'floor_or_range_limited',shared_bias_shift=shift)
            receipt['biases']={name:float(p) for name,p in core.named_parameters() if name.startswith('router.') and name.endswith('.bias')}
        del samples
        if recipe['arm'].startswith('U10-'):
            _verify_unified_birth(model,provider,engine,batches,macro_indices,receipt)
        # Calibrate the unpressured support proxy using the same fixed TRAIN
        # rows. e3001 pressure is zero by design, so remove only that scalar
        # ramp for the numerical norm; no scientific optimization is performed.
        receipt['gradient_clock']=dict(input_epoch=3001,temporary_work_clock_epoch=3002,
            reason='unpressure_support_proxy_norm_only',pressure_divided_out=True,
            model_provider_rng_and_clocks_restored=True)
        model.train();planned=provider.loss_denominators(batches,'joint','A2-D')
        named=tuple(core.router.named_parameters());parameters=tuple(p for _,p in named)
        native=[torch.zeros_like(p) for p in parameters]
        kinds=core.joint_constraint_kinds();work={k:[torch.zeros_like(p) for p in parameters] for k in kinds};counts={k:0 for k in kinds}
        for index,batch in zip(macro_indices,batches):
            core.joint_begin_update(3002,index);scene=provider.make_scene(batch.scene_inputs)
            prediction,aux=provider.predict_native(model,scene,batch.receivers,'A2-D','joint',epoch=3001,
                temperature=engine.config.temperature_for_epoch(3001))
            constraints=core.joint_microbatch_result()['constraints']
            terms=provider.loss_terms(prediction,batch.targets,'joint',aux)
            objective=sum(term.numerator*(float(term.weight)/float(planned[name])) for name,term in terms.items())
            grad=torch.autograd.grad(objective,parameters,retain_graph=True,allow_unused=True)
            for target,g in zip(native,grad):
                if g is not None:target.add_(g.detach())
            for kind in kinds:
                counts[kind]+=constraints[kind]['denominator']
                value=constraints[kind]['residual_numerator']
                if kind in ('M','E'):value=value/core.joint_budget.schedule()['pressure']
                elif kind=='context':
                    from honf_forward_core.continuous_joint.unified import context_schedule
                    value=value/context_schedule(3002)[1]
                grad=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                for target,g in zip(work[kind],grad):
                    if g is not None:target.add_(g.detach())
            del prediction,aux,scene,objective,constraints,terms
        for kind in kinds:
            def belongs(name):
                return (('incidence.' in name) if kind=='context' else ('dispatch_parameters' in name
                    or kind=='M' and any(s in name for s in ('.mm.','.em.','.module.'))
                    or kind=='E' and any(s in name for s in ('.me.','.ee.','.environment.'))))
            indices=[i for i,(name,p) in enumerate(named) if belongs('.'+name)]
            native_norm=math.sqrt(sum(float(native[i].double().square().sum()) for i in indices))
            work_norm=math.sqrt(sum(float((work[kind][i]/max(counts[kind],1)).double().square().sum()) for i in indices))
            supported=all(math.isfinite(x) and x>0 for x in (native_norm,work_norm))
            receipt['typed_gradient'][kind]=dict(status='calibrated' if supported else 'unsupported_zero_or_nonfinite_gradient',
                native_gradient_norm=native_norm,work_gradient_norm=work_norm,
                lambda0=.05*native_norm/work_norm if supported else None,parameter_names=[named[i][0] for i in indices])
        return receipt
    finally:
        for hook in hooks:hook.remove()
        core.load_state_dict(state,strict=True);core._joint_entries=[]
        if hasattr(core,'_context_entries'):core._context_entries=[]
        for module,training in modes.items():module.training=training
        if provider_state is not None:provider.load_training_state_dict(provider_state)
        _restore_calibration_rng_state(rng)


def _verify_unified_birth(model,provider,engine,batches,macro_indices,receipt):
    """Remeasure actual new preparation after bias calibration, on the same inputs."""
    core=getattr(model,'core',model);passes=[]
    corrections={'M':0.,'E':0.}
    model.eval()
    for attempt in range(6):
        access={kind:[0.,0,0] for kind in ('M','E')}
        context={route:[0.,0,0] for route in ('mm','em','ee','me')}
        def observe_router(module,args,result):
            gates,_=result;records=args[2];qmask=args[4]
            for kind,sl in (('M',slice(0,records.module_count)),('E',slice(records.module_count,None))):
                valid=records.eligible[:,None,sl]&qmask[...,None];values=gates[...,sl][valid]
                access[kind][0]+=float(values.double().sum());access[kind][1]+=values.numel()
                access[kind][2]+=int((values>0).sum())
        def observe_context(module,args,kwargs,result):
            if kwargs.get('stage_identity') is None:return
            route=args[2];values=result[args[1]]
            context[route][0]+=float(values.double().sum());context[route][1]+=values.numel()
            context[route][2]+=int((values>0).sum())
        handles=[core.router.register_forward_hook(observe_router),
                 core.router.incidence.register_forward_hook(observe_context,with_kwargs=True)]
        try:
            with torch.no_grad():
                for index,batch in zip(macro_indices,batches):
                    core.joint_begin_update(3001,index);scene=provider.make_scene(batch.scene_inputs)
                    provider.predict_native(model,scene,batch.receivers,'A2-D','joint',epoch=3001,
                        temperature=engine.config.temperature_for_epoch(3001))
        finally:
            for handle in handles:handle.remove()
        means={kind:(row[0]/row[1] if row[1] else 0.) for kind,row in access.items()}
        context_means={route:(row[0]/row[1] if row[1] else None) for route,row in context.items()}
        passes.append(dict(index=attempt,mean_composed_access=means,context_mean=context_means))
        context_ready=True
        if core.routing_mode=='direct' and not receipt.get('context_initialization_sha256'):
            with torch.no_grad():
                for route,mean in context_means.items():
                    if mean is None:continue
                    if abs(mean-.98)>1e-4:
                        context_ready=False
                        derivative=max(2*math.sqrt(max(mean,0.))*(max(1-mean,0.))**1.5,1e-6)
                        shift=max(-2.,min(2.,(.98-mean)/derivative))
                        core.router.incidence.routes[route].bias.add_(shift)
        if all(abs(mean-.98)<=1e-4 for mean in means.values()) and context_ready:break
        if attempt==5:break
        with torch.no_grad():
            for kind,mean in means.items():
                if abs(mean-.98)<=1e-4:continue
                derivative=max(2*math.sqrt(max(mean,0.))*(max(1-mean,0.))**1.5,1e-6)
                shift=max(-2.,min(2.,(.98-mean)/derivative));corrections[kind]+=shift
                if core.routing_mode=='direct':
                    calibration=core.router.direct_parameters.module if kind=='M' else core.router.direct_parameters.environment
                    calibration.bias.add_(shift)
                else:
                    for route,calibration in core.router.incidence.routes.items():
                        if (route[1]=='m')==(kind=='M'):calibration.bias.add_(shift)
    for kind,row in access.items():
        receipt['birth'][kind].update(mean_composed_access=means[kind],positive_rows=row[2],possible_rows=row[1],
            status='calibrated' if abs(means[kind]-.98)<=1e-4 else 'fresh_context_composition_miss',
            fresh_state_shared_bias_correction=corrections[kind])
    receipt['context_birth']={route:dict(mean_incidence=context_means[route],positive_edges=row[2],eligible_edges=row[1])
                              for route,row in context.items()}
    receipt['fresh_state_composed_access_verification']=dict(passes=passes,
        fixed_input_batches_reused=True,target_outputs_not_read=True)
    receipt['biases']={name:float(p) for name,p in core.named_parameters() if name.startswith('router.') and name.endswith('.bias')}
