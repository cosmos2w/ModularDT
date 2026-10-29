"""Bounded real-objective native engineering epochs; no scientific trajectory output."""
from __future__ import annotations
import hashlib
import time
import torch
from .continuous_joint_child import (_allowed_dropped_names,filtered_continuous_optimizer_seed,
                                      verify_continuous_joint_optimizer)


def qualify_complete_native_epochs(task,*,epochs=1,successful_step_recorder=None,boundary=None):
    """Use the maintained engine's exact native macro update for complete epochs.

    The caller must reserve an engineering GPU lane. Every optimizer update is
    discarded and counted; this never produces a resumable scientific member.
    """
    if type(epochs) is not int or not 1<=epochs<=40:
        raise ValueError('Qualification permits one to forty complete native engineering epochs.')
    model,provider,engine=task['model'],task['provider'],task['engine']
    recipe=task['identity']['recipe'];source=task['source_payload']
    if next(model.parameters()).device.type!='cuda':
        raise ValueError('Real native epoch qualification requires a reserved CUDA lane.')
    names=dict(model.named_parameters())
    old_names={name for group in source['optimizer_state_by_name']['groups'] for name in group['parameter_names']}
    amendment=dict(dropped_optimizer_parameter_names=_allowed_dropped_names(old_names,names,recipe),
                   new_parameter_names=sorted(set(names)-old_names))
    seed=filtered_continuous_optimizer_seed(source['optimizer_state_by_name'],amendment)
    optimizer,specs=engine._make_optimizer(model,provider,'A2-D',3001,seed)
    verify_continuous_joint_optimizer(model,optimizer,source['optimizer_state_by_name'],amendment)
    restore=getattr(provider,'load_training_state_dict',None)
    if callable(restore):restore(source.get('provider_training_state',{}))
    from .unified_training import restore_rng_state
    restore_rng_state(source['rng_state'])
    model.train();rows=[];steps=[]
    def record(epoch,index,cases):
        item=dict(epoch=epoch,macro_index=index,cases=cases);steps.append(item)
        if successful_step_recorder is not None:successful_step_recorder(epoch,index,cases)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    for epoch in range(3001,3001+epochs):
        if boundary is not None:boundary()
        engine._set_schedules(optimizer,specs,epoch)
        torch.cuda.synchronize();epoch_started=time.monotonic()
        cases,namespace=engine.epoch_case_order(provider,epoch)
        actual_visits=0;loss_rows=[];query_hashes=[]
        for index,sl in enumerate(engine.epoch_update_slices(len(cases))):
            if boundary is not None:boundary()
            batches=engine.prepare_update_microbatches(provider,arm='A2-D',phase='joint',epoch=epoch,
                update_index=index,update_cases=cases[sl],sampling_dataset_id=namespace)
            actual_visits+=sum(len(batch.case_keys) for batch in batches)
            losses,work,_,signature=engine._run_update(model,provider,optimizer,batches,phase='joint',arm='A2-D',
                epoch=epoch,update_index=index,successful_step_recorder=record)
            loss_rows.append(dict(macro_index=index,losses=losses,work=work))
            query_hashes.append(signature)
            if boundary is not None:boundary()
        torch.cuda.synchronize()
        if actual_visits!=len(cases):raise ValueError('Engineering epoch did not visit every selected native case exactly once.')
        rows.append(dict(epoch=epoch,cases=actual_visits,optimizer_updates=len(loss_rows),
            elapsed_seconds=time.monotonic()-epoch_started,native_macros=loss_rows,
            case_query_sampling_sha256=hashlib.sha256(repr(query_hashes).encode()).hexdigest()))
    return dict(schema='round10_complete_native_qualification_v1',status='completed_engineering_epochs',
        dataset=recipe['dataset'],arm=recipe['arm'],source_checkpoint_sha256=recipe['source_checkpoint_sha256'],
        scientific_training_epochs=0,discarded_optimizer_updates=len(steps),completed_native_epochs=epochs,
        elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),epoch_records=rows,
        exact_inherited_adam_verified=True,dropped_optimizer_parameter_names=amendment['dropped_optimizer_parameter_names'])
