#!/usr/bin/env python3
"""Continue frozen Round9 scientific state under a new Round10 reservation.

The frozen factory, TrainingEngine, native losses and exact-resume guards are
executed unchanged. Only closed Round9 operational admission is superseded by
the separate Round10 mount/device/storage/time/update admission and monitor.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from honf_round10_resources import Campaign, Monitor, atomic, read
from honf_joint_device_continue import validate_snapshot


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def equal(a,b,path=''):
    import numpy as np
    import torch
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a.detach().cpu(),b.detach().cpu()):
            raise ValueError('Tensor mismatch at '+path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('Array mismatch at '+path)
    elif isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Key mismatch at '+path)
        for k,v in a.items():equal(v,b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):raise ValueError('Sequence mismatch at '+path)
        for i,(v,w) in enumerate(zip(a,b)):equal(v,w,path+'/'+str(i))
    elif a!=b:raise ValueError('Value mismatch at '+path)


def load_task(args):
    cli,source=validate_snapshot(args.snapshot,args.snapshot_manifest,args.snapshot_manifest_sha256)
    spec=importlib.util.spec_from_file_location('_round10_frozen_native_cli',cli)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    recipe=module.validate_joint_native_recipe(read(args.recipe),verify_source=True)
    module.validate_joint_native_stop(recipe,args.stop_after)
    from honf_runtime.source_read_protocol import configure_source_read_numerics
    configure_source_read_numerics(recipe['base_recipe'])
    from honf_runtime.joint_native_factory import build_joint_native_task
    # Match the frozen CLI's import inventory at factory capture time.
    from honf_runtime.joint_native_resources import JointResourceMonitor
    from honf_runtime.compat import load_trusted_checkpoint
    from honf_runtime.run_layout import resolve_checkpoint
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cpu' if args.command=='prepare' else 'cuda:0'
    if device!='cpu':
        inventory=subprocess_inventory(args.physical_gpu)
        floor=max(8*1024**3,.2*inventory['total']);margin=6*1024**3
        usable=inventory['free']-floor-margin
        if usable<1024**3:raise RuntimeError('Insufficient device allocation headroom.')
        torch.cuda.set_per_process_memory_fraction(min(usable/inventory['total'],.70),device=0)
    task=build_joint_native_task(recipe,device=device)
    payload=load_trusted_checkpoint(args.parent_checkpoint,map_location='cpu')
    if payload['epoch']!=3500 or sha(args.parent_checkpoint)!=args.parent_sha256:
        raise ValueError('Track A requires the literal authenticated e3500 scientific checkpoint.')
    if payload['experiment_identity']['recipe']!=recipe or recipe['microbatch_cases']!=8:
        raise ValueError('Continuation recipe or microbatch differs from original.')
    expected=payload['experiment_identity']['loaded_source']['files']
    actual=task['identity']['loaded_source']['files']
    if expected!=actual:
        raise ValueError('Frozen loaded source differs: '+json.dumps({p:[expected.get(p),actual.get(p)] for p in set(expected)|set(actual) if expected.get(p)!=actual.get(p)},sort_keys=True))
    return task,payload,source


def subprocess_inventory(index):
    import subprocess
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
    for line in rows.splitlines():
        physical,free,total=[int(v.strip()) for v in line.split(',')]
        if physical==index:return {'free':free*1024**2,'total':total*1024**2}
    raise RuntimeError('Device disappeared.')


def seed(args):
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'parent_reference_v1.json').exists():
        previous=read(output/'parent_reference_v1.json')
        if previous['sha256']!=args.parent_sha256:raise ValueError('Run parent mapping changed.')
        return
    if any(output.iterdir()):raise ValueError('New Track A output must be empty.')
    (output/'checkpoints').mkdir()
    parent=args.parent_checkpoint.resolve(strict=True)
    if sha(parent)!=args.parent_sha256:raise ValueError('Parent checksum differs.')
    aliases=[p for p in parent.parent.glob('*model.pt') if p.name.startswith('best')]
    for target in aliases+[parent]:
        (output/'checkpoints'/target.name).symlink_to(target)
    (output/'checkpoints/latest_model.pt').symlink_to(parent)
    atomic(output/'artifact_layout.json',{'artifact_layout_version':1,'run_dir':str(output)})
    atomic(output/'parent_reference_v1.json',{'schema':'round10_literal_continuation_parent_v1','path':str(parent),'sha256':args.parent_sha256,
        'recipe':str(args.recipe.resolve()),'recipe_sha256':sha(args.recipe),'snapshot':str(args.snapshot.resolve()),
        'snapshot_manifest':str(args.snapshot_manifest.resolve()),'snapshot_manifest_sha256':args.snapshot_manifest_sha256,
        'inherited_absolute_epoch':3500,'new_absolute_epochs':[3501,5000],
        'original_recipe_and_resource_files_unchanged':True,'original_directory_write_policy':'read_only',
        'superseded_operational_guard':'closed Round9 root/device/start/time/update admission; new Round10 master reservation and atomic ledger apply',
        'scientific_resume_guards':'unchanged frozen TrainingEngine exact model/namedAdam/RNG/provider/identity checks'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','parity','run'))
    for name in ('root','snapshot','snapshot-manifest','recipe','parent-checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--snapshot-manifest-sha256',required=True);parser.add_argument('--parent-sha256',required=True)
    parser.add_argument('--physical-gpu',type=int,required=True);parser.add_argument('--stop-after',type=int,choices=(4000,4500,5000),required=True)
    parser.add_argument('--operation',required=True);parser.add_argument('--seconds',type=float,default=10800)
    args=parser.parse_args();campaign=Campaign(args.root)
    output=args.output.resolve()
    if not output.is_relative_to(campaign.root/'runs/mature_round9'):raise ValueError('Track A output outside declared root.')
    if args.command=='prepare':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation must hide CUDA.')
        task,payload,source=load_task(args);seed(args)
        import torch
        if torch.cuda.is_initialized():raise RuntimeError('CPU preparation unexpectedly initialized CUDA.')
        cases,_=task['engine'].epoch_case_order(task['provider'],3501)
        receipt={'schema':'round10_track_a_cpu_preparation_v1','source':source,'parent_sha256':args.parent_sha256,
            'cases':len(cases),'updates_per_epoch':len(task['engine'].epoch_update_slices(len(cases))),
            'loaded_source_exact':True,'recipe_unchanged':True,'cuda_initialized':torch.cuda.is_initialized(),'scientific_updates':0}
        atomic(campaign.root/f'receipts/{args.operation}.json',receipt);print(json.dumps(receipt,sort_keys=True));return
    with campaign.device(args.physical_gpu) as uuid:
        seed(args)
        from honf_joint_device_continue import read_object
        # Read only metadata before reserving the complete bounded operation.
        recipe=read(args.recipe)
        summary=output/'metrics/fit_summary.json'
        reserved_start=int(read(summary)['completed_epoch']) if summary.exists() else 3500
        updates=2 if args.command=='parity' else (args.stop_after-reserved_start)*(4 if recipe['dataset']=='thermal' else 5)
        campaign.reserve(args.operation,track='A',kind='engineering' if args.command=='parity' else 'main',updates=updates,seconds=args.seconds,physical_gpu=args.physical_gpu)
        completed=0;status='failed';error=None
        try:
            with Monitor(campaign,args.operation,output,uuid):
                task,payload,source=load_task(args)
                from honf_runtime.run_layout import resolve_checkpoint
                from honf_runtime.compat import load_trusted_checkpoint
                checkpoint=resolve_checkpoint(output,'latest')
                current=load_trusted_checkpoint(checkpoint,map_location='cpu')
                start=current['epoch'];journal=output/'logs/joint_optimizer_steps.jsonl'
                if args.command=='run' and start!=reserved_start:raise ValueError('Saved epoch differs from reserved update interval.')
                before=len(journal.read_text().splitlines()) if journal.exists() else 0
                if args.command=='parity':
                    receipt=parity(task,payload,args);completed=2
                    atomic(campaign.root/f'receipts/{args.operation}.json',receipt)
                else:
                    atomic(campaign.root/f'receipts/{args.operation}.started.json',{'PID':os.getpid(),'unix':time.time(),'source':source,'parent_sha256':args.parent_sha256,
                        'recipe_sha256':sha(args.recipe),'wrapper_sha256':sha(__file__),'start_epoch':start,'stop_after':args.stop_after,'physical_gpu':args.physical_gpu,'gpu_uuid':uuid})
                    result=task['engine'].fit(task['model'],task['provider'],output,identity=task['identity'],arm='A2-D',stop_after=args.stop_after,
                        resume_checkpoint=checkpoint,joint_native_child=task['joint_native_child'])
                    completed=(len(journal.read_text().splitlines()) if journal.exists() else 0)-before
                    atomic(campaign.root/f'receipts/{args.operation}.json',{'result':result,'source':source,'parent_sha256_unchanged':sha(args.parent_checkpoint)==args.parent_sha256,
                        'scientific_updates':completed,'wrapper_sha256':sha(__file__)})
                status='completed'
        except BaseException as exc:
            error=repr(exc)
            if args.command=='parity':completed=getattr(args,'actual_updates',0)
            if args.command=='run' and 'journal' in locals() and journal.exists():completed=len(journal.read_text().splitlines())-before
            raise
        finally:campaign.finish(args.operation,updates=completed,status=status,error=error)


def parity(task,payload,args):
    """Repeat the real first complete native macro from the literal saved state."""
    import torch
    from honf_runtime.unified_training import named_optimizer_state,restore_rng_state,capture_rng_state
    model=task['model'];provider=task['provider'];engine=task['engine'];rows=[]
    final_state=None;final_optimizer=None;final_provider=None;final_rng=None;comparisons={};args.actual_updates=0
    for repetition in range(2):
        model.load_state_dict(payload['model_state_dict'],strict=True)
        equal(payload['model_state_dict'],model.state_dict(),'initial_model')
        loader=getattr(provider,'load_training_state_dict',None)
        if loader:loader(copy.deepcopy(payload.get('provider_training_state',{})))
        optimizer,specs=engine._make_optimizer(model,provider,'A2-D',3501,payload['optimizer_state_by_name'])
        equal(payload['optimizer_state_by_name'],named_optimizer_state(model,optimizer),'initial_namedAdam')
        restore_rng_state(payload['rng_state']);equal(payload['rng_state'],capture_rng_state(),'initial_rng');model.train(True)
        if callable(getattr(getattr(model,'core',model),'set_native_training_epoch',None)):
            engine._set_native_retrieval_epoch(model,3501)
        cases,dataset=engine.epoch_case_order(provider,3501);slices=engine.epoch_update_slices(len(cases))
        batches=engine.prepare_update_microbatches(provider,arm='A2-D',phase='joint',epoch=3501,update_index=0,update_cases=cases[slices[0]],sampling_dataset_id=dataset)
        def record_step(*unused):args.actual_updates+=1
        row=engine._run_update(model,provider,optimizer,batches,phase='joint',arm='A2-D',epoch=3501,update_index=0,successful_step_recorder=record_step)
        torch.cuda.synchronize();rows.append(row)
        state={k:v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in model.state_dict().items()}
        optimizer_state=copy.deepcopy(named_optimizer_state(model,optimizer))
        provider_state=copy.deepcopy(provider.training_state_dict()) if hasattr(provider,'training_state_dict') else {}
        rng=capture_rng_state()
        if repetition==0:final_state=state;final_optimizer=optimizer_state;final_provider=provider_state;final_rng=rng
        else:
            def differences(a,b,path=''):
                if torch.is_tensor(a):
                    if not torch.equal(a.detach().cpu(),b.detach().cpu()):
                        delta=(a.detach().cpu().to(torch.float64)-b.detach().cpu().to(torch.float64)).abs()
                        comparisons[path]=float(delta.max()) if delta.numel() else 0.
                elif isinstance(a,dict):
                    if set(a)!=set(b):raise ValueError('Replay keys changed at '+path)
                    for k,v in a.items():differences(v,b[k],path+'/'+str(k))
                elif isinstance(a,(list,tuple)):
                    for i,(v,w) in enumerate(zip(a,b)):differences(v,w,path+'/'+str(i))
                elif isinstance(a,(float,int)) and a!=b:comparisons[path]=abs(float(a)-float(b))
                elif a!=b:raise ValueError('Replay metadata changed at '+path)
            differences(final_state,state,'model');differences(final_optimizer,optimizer_state,'namedAdam')
            equal(final_provider,provider_state,'post_step_provider');equal(final_rng,rng,'post_step_rng')
            differences(rows[0],row,'native')
        del batches,optimizer
    return {'schema':'round10_track_a_same_source_first_macro_parity_v1','strict_bitwise_passed':not comparisons,'parent_sha256':args.parent_sha256,
        'actual_discarded_optimizer_updates':2,'scientific_optimizer_updates':0,'epoch':3501,'macro_index':0,
        'scope':'same frozen fitting implementation, real complete native macro, two literal state restores; exact initial model/namedAdam and final provider/RNG; numerical replay differences retained',
        'maximum_absolute_differences':{prefix:max((value for key,value in comparisons.items() if key.startswith(prefix)),default=0.) for prefix in ('model','namedAdam','native')},
        'all_nonzero_differences':comparisons,'rows':rows,'source_recipe_unchanged':True,'wrapper_sha256':sha(__file__)}


if __name__=='__main__':main()
