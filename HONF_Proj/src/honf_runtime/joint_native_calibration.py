"""Fixed-TRAIN birth and typed gradient calibration with no optimizer update."""
from __future__ import annotations
import copy
import json
import math
from collections.abc import Mapping
from types import MethodType
from pathlib import Path
import torch
from .joint_native_protocol import canonical_sha256,file_sha256


def apply_joint_native_calibration(core,recipe):
    binding=recipe['calibration'];path=Path(binding['path'])
    if file_sha256(path)!=binding['sha256']:raise ValueError('TRAIN calibration receipt changed.')
    receipt=json.loads(path.read_text())
    expected={'schema':'round9_train_calibration_v1','dataset':recipe['dataset'],'arm':recipe['arm'],
        'source_checkpoint_sha256':recipe['source_checkpoint_sha256'],'microbatch_cases':recipe['microbatch_cases']}
    if any(receipt.get(key)!=value for key,value in expected.items()):
        raise ValueError('TRAIN calibration belongs to another source, arm or microbatch.')
    if not 1<=len(receipt.get('batches',()))<=8 or receipt.get('optimizer_updates')!=0:
        raise ValueError('TRAIN calibration exceeds its fixed eight-batch/no-update contract.')
    from honf_forward_core.joint_native.routing import ScoreCalibration
    expected_biases={name+'.bias' for name,module in core.named_modules()
        if isinstance(module,ScoreCalibration) and module.bias is not None}
    if set(receipt.get('membership',{}))!=expected_biases:
        raise ValueError('TRAIN calibration omitted or added a typed membership family.')
    if receipt.get('target_nonmandatory_probability')!=.98 or receipt.get('fraction_per_type')!=.05:
        raise ValueError('TRAIN calibration changed declared birth or gradient-balance targets.')
    named=dict(core.named_parameters())
    with torch.no_grad():
        for name,row in receipt['membership'].items():
            if name not in named or not name.startswith('router.') or not name.endswith('.bias'):
                raise ValueError('TRAIN calibration attempts to alter a non-membership parameter.')
            if row['status']!='calibrated':raise ValueError('Sparse launch requires supported membership calibration.')
            named[name].fill_(row['bias'])
        for kind in ('M','E'):
            row=receipt['typed_gradient'][kind]
            if (row['status']!='calibrated' or not all(math.isfinite(row[k]) and row[k]>0
                    for k in ('native_gradient_norm','work_gradient_norm','lambda0'))):
                raise ValueError('Sparse launch requires supported finite native/work gradient calibration.')
            expected=.05*row['native_gradient_norm']/row['work_gradient_norm']
            if not math.isclose(row['lambda0'],expected,rel_tol=1e-10):raise ValueError('Typed calibration coefficient formula changed.')
            core.joint_budget.duals[kind].set_calibrated_lambda0(row['lambda0'])
    return receipt


def calibrate_joint_native_task(task, *, max_batches=8):
    """Use first fixed absolute-e3001 native batches, preserving RNG and parent weights.

    The returned receipt records unsupported zero-gradient cases explicitly;
    applying such a receipt refuses a scientific launch.
    """
    if type(max_batches) is not int or not 1<=max_batches<=8:raise ValueError('Calibration permits one through eight TRAIN batches.')
    from honf_forward_core.joint_native.routing import ScoreCalibration
    from honf_forward_core.joint_native.graph import calibrate_bias
    from honf_forward_core.joint_native.budget import calibrated_dual_weight
    from .unified_training import capture_rng_state,restore_rng_state
    model,provider,engine=task['model'],task['provider'],task['engine'];core=getattr(model,'core',model)
    recipe=task['identity']['recipe']
    if recipe['arm'] not in ('H9-ME','S9-ME') or recipe['source_epoch']!=3000:
        raise ValueError('Calibration requires a fresh sparse child of the literal shared e3000 warm-up.')
    original_state=copy.deepcopy(core.state_dict());rng=capture_rng_state();previous_mode=model.training
    core.set_training_clock(3001,0);core.joint_begin_update(3001,0);model.train()
    cases,dataset_id=engine.epoch_case_order(provider,3001)
    batches=[];macro_indices=[]
    for index,sl in enumerate(engine.epoch_update_slices(len(cases))):
        additions=engine.prepare_update_microbatches(provider,arm='A2-D',phase='joint',epoch=3001,
            update_index=index,update_cases=cases[sl],sampling_dataset_id=dataset_id)
        batches.extend(additions);macro_indices.extend([index]*len(additions))
        if len(batches)>=max_batches:break
    batches=batches[:max_batches];macro_indices=macro_indices[:max_batches]
    membership={name:module for name,module in core.named_modules() if isinstance(module,ScoreCalibration) and module.bias is not None}
    values={name:[] for name in membership};context={};handles=[]
    original_route=core.router.route
    def route(_self,directory,raw_M,raw_E,protected_M,protected_E,**options):
        context['module']=protected_M;context['environment']=protected_E;context['records']=directory
        try:return original_route(directory,raw_M,raw_E,protected_M,protected_E,**options)
        finally:context.clear()
    core.router.route=MethodType(route,core.router)
    def hook(name):
        def capture(module,inputs,output):
            allowed=inputs[1].detach().clone();logits=(output-module.bias).detach()
            suffix=name.rsplit('.',1)[-1]
            if suffix in ('module','environment') and suffix not in context:
                # The detached deterministic controller replays these modules;
                # calibrate only the actual request with its exact protection union.
                return
            if suffix in ('mm','ee'):
                eye=torch.eye(allowed.shape[-1],device=allowed.device,dtype=torch.bool)[None]
                allowed &= ~eye
            else:
                # Cross-type memberships and direct gates each retain one best donor.
                candidate=output.detach().masked_fill(~allowed,-torch.inf)
                if candidate.shape[-1]:
                    if suffix in ('module','environment'):
                        from honf_forward_core.joint_native.routing import canonical_winner
                        from honf_forward_core.joint_native.contracts import CanonicalGeometry
                        records=context['records'];m=records.module_count
                        sl=slice(None,m) if suffix=='module' else slice(m,None)
                        geometry=CanonicalGeometry.build(records.geometry.keys[:,sl])
                        winner,_,unresolved=canonical_winner(candidate,allowed,geometry)
                        if bool(unresolved.any()):raise ValueError('Indeterminate direct calibration best-donor tie.')
                        floor=torch.zeros_like(allowed).scatter_(2,winner.clamp_min(0)[...,None],(winner>=0)[...,None])
                    else:
                        winner=candidate.argmax(-1,keepdim=True)
                        floor=torch.zeros_like(allowed).scatter_(2,winner,True)
                    allowed &= ~floor
                if suffix in context:allowed &= ~context[suffix]
            values[name].append(logits[allowed].detach().cpu())
        return capture
    receipt={'schema':'round9_train_calibration_v1','dataset':recipe['dataset'],'arm':recipe['arm'],
        'source_checkpoint_sha256':recipe['source_checkpoint_sha256'],'microbatch_cases':recipe['microbatch_cases'],
        'epoch':3001,'optimizer_updates':0,'target_nonmandatory_probability':.98,'fraction_per_type':.05,
        'batches':[{'macro_index_zero_based':index,'case_keys':[repr(k) for k in b.case_keys],'query_sampling_sha256':b.auxiliary.get('query_sampling_sha256') if isinstance(b.auxiliary,Mapping) else None} for index,b in zip(macro_indices,batches)],
        'membership':{},'typed_gradient':{}}
    try:
        handles=[module.register_forward_hook(hook(name)) for name,module in membership.items()]
        for index,batch in zip(macro_indices,batches):
            core.joint_begin_update(3001,index)
            core.joint_begin_microbatch();scene=provider.make_scene(batch.scene_inputs)
            prediction,aux=provider.predict_native(model,scene,batch.receivers,'A2-D','joint',epoch=3001,temperature=engine.config.temperature_for_epoch(3001))
            core._joint_entries=[];del prediction,aux,scene
        for handle in handles:handle.remove()
        handles=[];core.router.route=original_route
        with torch.no_grad():
            for name,module in membership.items():
                selected=torch.cat(values[name]) if values[name] else torch.empty(0)
                if not selected.numel():receipt['membership'][name+'.bias']={'status':'unsupported_no_nonmandatory_pairs'};continue
                bias=float(calibrate_bias(selected,torch.ones_like(selected,dtype=torch.bool)))
                module.bias.fill_(bias)
                receipt['membership'][name+'.bias']={'status':'calibrated','bias':bias,'eligible_pairs':selected.numel(),
                    'mean_birth_probability':float((selected+bias).sigmoid().mean())}
        planned=provider.loss_denominators(batches,'joint','A2-D')
        named=tuple(core.router.named_parameters());parameters=tuple(p for _,p in named)
        native=[torch.zeros_like(p) for p in parameters];work={kind:[torch.zeros_like(p) for p in parameters] for kind in ('M','E')};counts={'M':0,'E':0}
        for index,batch in zip(macro_indices,batches):
            core.joint_begin_update(3001,index)
            core.joint_begin_microbatch();scene=provider.make_scene(batch.scene_inputs)
            prediction,aux=provider.predict_native(model,scene,batch.receivers,'A2-D','joint',epoch=3001,temperature=engine.config.temperature_for_epoch(3001))
            constraints=core.joint_microbatch_result()['constraints'];terms=provider.loss_terms(prediction,batch.targets,'joint',aux)
            objective=sum(term.numerator*(float(term.weight)/float(planned[name])) for name,term in terms.items())
            gradient=torch.autograd.grad(objective,parameters,retain_graph=True,allow_unused=True)
            for target,g in zip(native,gradient):
                if g is not None:target.add_(g.detach())
            for kind in ('M','E'):
                counts[kind]+=constraints[kind]['denominator']
                gradient=torch.autograd.grad(constraints[kind]['residual_numerator'],parameters,retain_graph=True,allow_unused=True)
                for target,g in zip(work[kind],gradient):
                    if g is not None:target.add_(g.detach())
            del prediction,aux,scene,objective,constraints,terms
        for kind in ('M','E'):
            # Dispatch is shared. Membership/direct parameters are attributed by consumed donor type.
            def belongs(name):
                return ('dispatch_parameters' in name or (kind=='M' and any(x in name for x in ('.mm.','.em.','.module.')))
                    or (kind=='E' and any(x in name for x in ('.me.','.ee.','.environment.'))))
            indices=[i for i,(name,p) in enumerate(named) if belongs('.'+name)]
            native_norm=math.sqrt(sum(float(native[i].double().square().sum()) for i in indices))
            work_norm=math.sqrt(sum(float((work[kind][i]/max(counts[kind],1)).double().square().sum()) for i in indices))
            row={'native_gradient_norm':native_norm,'work_gradient_norm':work_norm,'parameter_names':[named[i][0] for i in indices]}
            try:row.update(status='calibrated',lambda0=calibrated_dual_weight(native_norm,work_norm,.05))
            except ValueError:row.update(status='unsupported_zero_or_nonfinite_gradient',lambda0=None)
            receipt['typed_gradient'][kind]=row
        return receipt
    finally:
        for handle in handles:handle.remove()
        core.router.route=original_route;core._joint_entries=[];core.load_state_dict(original_state,strict=True)
        model.train(previous_mode);restore_rng_state(rng)
