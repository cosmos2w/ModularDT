#!/usr/bin/env python3
"""Sealed Round10 continuous training through the maintained TrainingEngine."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[1]
for directory in (PROJECT/'src',PROJECT/'Case_ThermalChannel/src',PROJECT/'Case_WindFarm/src',PROJECT.parent/'tools'):
    sys.path.insert(0,str(directory))

def import_protocol():
    from honf_runtime import continuous_joint_protocol as protocol
    globals().update({name:getattr(protocol,name) for name in dir(protocol) if not name.startswith('_')})


def read(path):return json.loads(Path(path).read_text())
def write_new(path,value):
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan')
    plan.add_argument('--source-checkpoint',type=Path,required=True);plan.add_argument('--dataset',choices=('thermal','wind'),required=True)
    plan.add_argument('--arm',choices=('C10-H','C10-S','U10-D','U10-H','U10-S'),required=True);plan.add_argument('--resource-reservation',type=Path,required=True)
    plan.add_argument('--context-initialization-receipt',type=Path);plan.add_argument('--admission-receipt',type=Path);plan.add_argument('--model-options',type=Path);plan.add_argument('--calibration-receipt',type=Path);plan.add_argument('--output',type=Path,required=True)
    for command in ('prepare','calibrate','qualify','fit','resume'):
        p=sub.add_parser(command);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--stop-after',type=int,default=5000);p.add_argument('--device',default='cpu' if command in ('prepare','calibrate') else 'cuda:0')
        p.add_argument('--physical-gpu',type=int,choices=(1,2));p.add_argument('--reserve-seconds',type=float)
        p.add_argument('--source-manifest',type=Path);p.add_argument('--source-manifest-sha256')
        if command=='calibrate':p.add_argument('--max-batches',type=int,default=2)
        if command=='qualify':p.add_argument('--epochs',type=int,default=1)
    s=sub.add_parser('status');s.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('export');e.add_argument('--checkpoint',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    source_closure=None
    if args.command in ('qualify','fit','resume'):
        if args.source_manifest is None or args.source_manifest_sha256 is None:
            raise ValueError('GPU qualification/fit/resume requires an immutable complete source manifest and its SHA256.')
        from honf_runtime.source_closure import validate_source_manifest
        source_closure=validate_source_manifest(args.source_manifest,args.source_manifest_sha256,PROJECT.parent)
    import_protocol()
    if args.command=='plan':
        value=make_continuous_joint_recipe(args.source_checkpoint,dataset=args.dataset,arm=args.arm,
            resource_reservation=args.resource_reservation,model_options=read(args.model_options) if args.model_options else {},calibration_receipt=args.calibration_receipt,admission_receipt=args.admission_receipt,
            context_initialization_receipt=args.context_initialization_receipt)
        write_new(args.output,value);print(json.dumps(dict(status='sealed_plan',sha256=canonical_sha256(value))));return 0
    if args.command=='status':
        from honf_runtime.run_layout import RunLayout
        layout=RunLayout(args.output)
        for name in ('progress.json','fit_summary.json'):
            path=layout.read_path(name)
            if path.exists():print(json.dumps({name:read(path)}))
        return 0
    if args.command=='export':
        from honf_runtime.continuous_joint_runtime import export_continuous_joint_checkpoint
        print(export_continuous_joint_checkpoint(args.checkpoint,args.output));return 0
    recipe=validate_continuous_joint_recipe(read(args.recipe),verify_source=True)
    validate_continuous_joint_stop(recipe,args.stop_after)
    from honf_round10_resources import Campaign,Monitor
    resource=Path(recipe['resource_reservation']).resolve(strict=True)
    campaign=Campaign(resource.parents[1]);campaign.check_storage()
    if not args.output.resolve().is_relative_to(campaign.root):raise ValueError('Continuous artifacts must remain under their reserved payload root.')
    from honf_runtime.source_read_protocol import configure_source_read_numerics
    configure_source_read_numerics(recipe['base_recipe'])
    from honf_runtime.continuous_joint_factory import build_continuous_joint_task
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.command in ('prepare','calibrate'):
        if args.device!='cpu':raise ValueError('CPU preparation/calibration never opens CUDA.')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation/calibration requires CUDA_VISIBLE_DEVICES empty.')
        task=build_continuous_joint_task(recipe,device='cpu')
        if args.command=='calibrate':
            from honf_runtime.continuous_joint_calibration import calibrate_continuous_joint_task
            value=calibrate_continuous_joint_task(task,max_batches=args.max_batches)
            value['source_state']='live_unsealed_CPU_calibration_no_optimizer_updates'
            write_new(args.output,value)
            print(json.dumps(dict(status='calibrated_without_updates',path=str(args.output))));return 0
        cases,_=task['engine'].epoch_case_order(task['provider'],3001)
        core=getattr(task['model'],'core',task['model'])
        prepared=dict(status='prepared_without_updates',cases=len(cases),
            updates_per_epoch=len(task['engine'].epoch_update_slices(len(cases))),final_epoch=5000,source_epoch=3000,
            recipe_sha256=canonical_sha256(recipe),source_checkpoint_sha256=recipe['source_checkpoint_sha256'],
            cuda_initialized=torch.cuda.is_initialized(),optimizer_updates=0,
            provider_identity_sha256=canonical_sha256(task['provider_identity']),
            initialization_inventory=getattr(core,'parent_initialization_inventory',None),source_state='live_unsealed_CPU_preparation_only')
        write_new(args.output,prepared)
        print(json.dumps({key:prepared[key] for key in ('status','cases','updates_per_epoch','recipe_sha256','cuda_initialized','optimizer_updates')}));return 0
    if args.device!='cuda:0' or args.physical_gpu is None or args.reserve_seconds is None:
        raise ValueError('Fit requires an explicit physical device and bounded process-time reservation.')
    if recipe['arm']!='U10-D' and recipe['calibration'] is None:raise ValueError('Continuous scientific fit requires composed-access and typed-gradient calibration.')
    if recipe['arm'].startswith('U10-') and args.command!='qualify':validate_unified_admission(recipe)
    resume=None;start=3000
    if args.command=='resume':
        from honf_runtime.run_layout import resolve_checkpoint
        from honf_runtime.compat import load_trusted_checkpoint
        resume=resolve_checkpoint(args.output,'latest');start=load_trusted_checkpoint(resume,map_location='cpu')['epoch']
    updates=(args.epochs if args.command=='qualify' else args.stop_after-start)*(4 if recipe['dataset']=='thermal' else 5)
    operation='continuous_'+recipe['dataset']+'_'+recipe['arm']+'_'+str(time.time_ns())
    access = 'exclusive' if args.command == 'qualify' else 'shared_training'
    with campaign.device(args.physical_gpu, access=access) as uuid:
        campaign.reserve(operation,track='C' if recipe['arm'].startswith('U10-') else 'B',kind='engineering' if args.command=='qualify' else 'main',updates=updates,seconds=args.reserve_seconds,physical_gpu=args.physical_gpu)
        status='failed';error=None;actual=0
        journal=args.output.resolve()/'logs/joint_optimizer_steps.jsonl'
        before=len(journal.read_text().splitlines()) if journal.exists() else 0
        try:
            # Derive allocator ceiling from the actual current device occupancy,
            # required free floor, and preventive margin before model transfer.
            inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
            free,total=next((int(f)*1024**2,int(t)*1024**2) for u,f,t in (line.split(',') for line in inventory.splitlines()) if u.strip()==uuid)
            floor=max(8*1024**3,.2*total);available=free-floor-6*1024**3
            if available<=0:raise RuntimeError('No safe allocator reservation remains.')
            torch.cuda.set_per_process_memory_fraction(min(.95,available/total),0)
            with Monitor(campaign,operation,args.output,uuid) as monitor:
                task=build_continuous_joint_task(recipe,device=args.device)
                task['identity']['source_closure']=source_closure
                if args.command=='qualify':
                    from honf_runtime.continuous_joint_qualification import qualify_complete_native_epochs
                    journal.parent.mkdir(parents=True,exist_ok=True)
                    def record(epoch,index,cases):
                        with journal.open('a') as stream:
                            stream.write(json.dumps(dict(epoch=epoch,macro_index=index,cases=cases,kind='discarded_engineering'))+'\n')
                    def boundary():
                        if (args.output/'CLEAN_STOP_REQUEST.json').exists() or not monitor.thread.is_alive():
                            raise RuntimeError('Native qualification resource/time stop; journaled discarded updates retained.')
                    result=qualify_complete_native_epochs(task,epochs=args.epochs,successful_step_recorder=record,boundary=boundary)
                    result.update(operation=operation,physical_gpu=args.physical_gpu,gpu_uuid=uuid,loaded_source=task['identity']['loaded_source'])
                    write_new(args.output/'qualification.json',result)
                    print(json.dumps(dict(status=result['status'],epochs=result['completed_native_epochs'],updates=result['discarded_optimizer_updates'])))
                    status='completed'
                    return 0
                result=task['engine'].fit(task['model'],task['provider'],args.output,identity=task['identity'],arm='A2-D',
                    stop_after=args.stop_after,branch_from_checkpoint=task['source_checkpoint'] if resume is None else None,
                    resume_checkpoint=resume,joint_native_child=task['joint_native_child'])
                print(json.dumps(result,default=str));status='completed'
        except Exception as exc:error=repr(exc);raise
        finally:
            if journal.exists():actual=len(journal.read_text().splitlines())-before
            campaign.finish(operation,updates=actual,status=status,error=error)
    return 0

if __name__=='__main__':raise SystemExit(main())
