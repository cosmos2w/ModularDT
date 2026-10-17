#!/usr/bin/env python3
"""Managed Round9 stages on the maintained TrainingEngine; no startup speed gate."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
for directory in (PROJECT/'src',PROJECT/'Case_ThermalChannel/src',PROJECT/'Case_WindFarm/src'):
    sys.path.insert(0,str(directory))
from honf_runtime.joint_native_protocol import *


def read(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict):raise TypeError('Expected a JSON object.')
    return value


def write_new(path,value):
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def check_resources(recipe,output,stop_after,*,start_after=None,cuda=False):
    """Check actual mount, byte caps, time and all recorded scientific updates before allocation."""
    reservation_path=Path(recipe['resource_reservation']).resolve(strict=True)
    resources=read(reservation_path);root=reservation_path.parents[1]
    if resources.get('schema')!='round9_resource_reservation_v1' or resources.get('gpu_uuid')!=GPU_UUID:
        raise ValueError('Round9 reservation schema/device differs.')
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T','/data2'],text=True))['filesystems'][0]
    expected=resources['mount']
    if any(mount.get(k)!=expected[k] for k in ('target','source','fstype')):
        raise RuntimeError('Actual /data2 mount differs from the reserved filesystem.')
    output=Path(output).resolve()
    if not output.is_relative_to(root):raise ValueError('Round9 output must stay within its reserved root.')
    retained=temporary=0
    for path in root.rglob('*'):
        if path.is_file() and not path.is_symlink():
            size=path.stat().st_size
            if path.is_relative_to(root/'tmp'):temporary+=size
            else:retained+=size
    if retained>resources['retained_disk_limit_bytes'] or temporary>resources['temporary_disk_limit_bytes']:
        raise OSError('Round9 actual payload exceeds retained/temporary caps.')
    # Reserve two full model+Adam atomic checkpoint payloads before any writer starts.
    projected=max(2*Path(recipe['source_checkpoint']).stat().st_size,128*1024**2)
    free=shutil.disk_usage('/data2').free
    if (retained+temporary+projected>resources['peak_new_disk_limit_bytes']
            or retained+projected>resources['retained_disk_limit_bytes']
            or temporary+projected>resources['temporary_disk_limit_bytes']
            or free-projected<resources['disk_min_free_bytes']):
        raise OSError('Round9 projected writes violate storage budget/headroom.')
    elapsed=time.time()-resources['start_unix']
    if elapsed+resources['closeout_reserved_seconds']>=resources['elapsed_limit_seconds']:
        raise RuntimeError('Round9 reached its reserved closeout time.')
    usage=root/'receipts/round9_usage.jsonl';rows=[]
    if usage.exists():rows=[json.loads(line) for line in usage.read_text().splitlines() if line.strip()]
    used_seconds=sum(row['associated_seconds'] for row in rows)
    updates_by_output={}
    for row in rows:updates_by_output[row['output']]=updates_by_output.get(row['output'],0)+row['optimizer_updates']
    # Successful-step journals also survive abrupt interpreter/OS termination,
    # including work repeated after an older exact checkpoint is restored.
    for journal in root.rglob('joint_optimizer_steps.jsonl'):
        run=str(journal.parent.parent.resolve())
        observed=sum(1 for line in journal.read_text().splitlines() if line.strip())
        updates_by_output[run]=max(updates_by_output.get(run,0),observed)
    used_updates=sum(updates_by_output.values())
    start=recipe['source_epoch'] if start_after is None else start_after
    updates=(stop_after-start)*(4 if recipe['dataset']=='thermal' else 5)
    if used_updates+updates>resources['scientific_updates']:
        raise RuntimeError('Round9 proposed stage exceeds the remaining 81000 scientific updates.')
    if max(elapsed,used_seconds)+resources['closeout_reserved_seconds']>=resources['gpu_associated_limit_seconds']:
        raise RuntimeError('Round9 GPU-associated time budget exhausted.')
    if cuda:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:
            raise ValueError('Round9 requires CUDA_VISIBLE_DEVICES mapped to physical GPU2 UUID.')
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
        found=None
        for line in inventory.splitlines():
            fields=[x.strip() for x in line.split(',')]
            if fields[0]==GPU_UUID:found=(int(fields[1])*1024**2,int(fields[2])*1024**2)
        floor=max(resources['gpu_min_free_bytes'],8*1024**3,math.ceil(.2*found[1]) if found else 0)
        if found is None or found[0]<floor+resources['preventive_margin_bytes']:
            raise RuntimeError('Round9 device lacks the declared free floor and preventive margin.')
    return root,resources


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    make=sub.add_parser('make-recipe');make.add_argument('--source-checkpoint',type=Path,required=True)
    make.add_argument('--dataset',choices=('thermal','wind'),required=True);make.add_argument('--arm',choices=JOINT_NATIVE_ARMS,required=True)
    make.add_argument('--microbatch-cases',type=int,required=True);make.add_argument('--resource-reservation',type=Path,required=True)
    make.add_argument('--model-options',type=Path);make.add_argument('--calibration-receipt',type=Path);make.add_argument('--output',type=Path,required=True)
    for command in ('prepare','fit','resume'):
        p=sub.add_parser(command);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--stop-after',type=int,required=True);p.add_argument('--device',default='cpu' if command=='prepare' else 'cuda:0')
    status=sub.add_parser('status');status.add_argument('--output',type=Path,required=True)
    export=sub.add_parser('export');export.add_argument('--checkpoint',type=Path,required=True);export.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='make-recipe':
        recipe=make_joint_native_recipe(args.source_checkpoint,dataset=args.dataset,arm=args.arm,microbatch_cases=args.microbatch_cases,
            resource_reservation=args.resource_reservation,calibration_receipt=args.calibration_receipt,model_options={} if args.model_options is None else read(args.model_options))
        write_new(args.output,recipe);print(json.dumps({'status':'recipe_written','path':str(args.output),'recipe_sha256':canonical_sha256(recipe)}));return 0
    if args.command=='status':
        from honf_runtime.run_layout import RunLayout
        layout=RunLayout(args.output)
        for name in ('progress.json','fit_summary.json'):
            path=layout.read_path(name)
            if path.exists():print(json.dumps({name:read(path)},sort_keys=True))
        return 0
    if args.command=='export':
        from honf_runtime.joint_native_runtime import export_joint_native_checkpoint
        from honf_runtime.compat import load_trusted_checkpoint
        payload=load_trusted_checkpoint(args.checkpoint,map_location='cpu')
        recipe=validate_joint_native_recipe(payload['experiment_identity']['recipe'])
        check_resources(recipe,args.output,payload['epoch'],start_after=payload['epoch'])
        print(export_joint_native_checkpoint(args.checkpoint,args.output));return 0
    recipe=validate_joint_native_recipe(read(args.recipe),verify_source=True);validate_joint_native_stop(recipe,args.stop_after)
    resume=None;start=recipe['source_epoch']
    if args.command=='resume':
        from honf_runtime.run_layout import resolve_checkpoint
        from honf_runtime.compat import load_trusted_checkpoint
        resume=resolve_checkpoint(args.output,'latest');payload=load_trusted_checkpoint(resume,map_location='cpu');start=payload['epoch']
    if args.command!='prepare' and args.device!='cuda:0':raise ValueError('Round9 scientific fit requires the physical GPU2 lane.')
    if args.command!='prepare' and recipe['arm'] in ('H9-ME','S9-ME') and recipe['calibration'] is None:
        raise ValueError('Sparse scientific launch requires measured fixed-TRAIN membership and typed-gradient calibration.')
    root,resources=check_resources(recipe,args.output,args.stop_after,start_after=start,cuda=args.command!='prepare')
    from honf_runtime.source_read_protocol import configure_source_read_numerics
    configure_source_read_numerics(recipe['base_recipe'])
    from honf_runtime.joint_native_factory import build_joint_native_task
    if args.command=='prepare':
        if args.device!='cpu':raise ValueError('Preparation is CPU-only.')
        task=build_joint_native_task(recipe,device='cpu');cases,_=task['engine'].epoch_case_order(task['provider'],start+1)
        print(json.dumps({'status':'prepared_without_updates','arm':recipe['arm'],'cases':len(cases),
            'updates_per_epoch':len(task['engine'].epoch_update_slices(len(cases))),
            'recipe_sha256':canonical_sha256(recipe),'microbatch_cases':recipe['microbatch_cases']}));return 0
    # Exclusive campaign ledger, including failure counts, prevents concurrent budget oversubscription.
    import fcntl
    lock=(root/'receipts/runtime.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_resources(recipe,args.output,args.stop_after,start_after=start,cuda=True)
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.70,device=0)
    from honf_runtime.joint_native_resources import JointResourceMonitor
    monitor=JointResourceMonitor(root,args.output,resources)
    step_journal=args.output.resolve()/'logs/joint_optimizer_steps.jsonl'
    initial_journal_steps=sum(1 for line in step_journal.read_text().splitlines() if line.strip()) if step_journal.exists() else 0
    task=None;result=None;status='failed';started=time.perf_counter()
    monitor.__enter__()
    try:
        task=build_joint_native_task(recipe,device=args.device)
        result=task['engine'].fit(task['model'],task['provider'],args.output,identity=task['identity'],arm='A2-D',
            stop_after=args.stop_after,branch_from_checkpoint=task['source_checkpoint'] if resume is None else None,
            resume_checkpoint=resume,joint_native_child=task['joint_native_child'])
        status='completed';print(json.dumps(result,default=str,sort_keys=True))
    finally:
        monitor.__exit__(None,None,None)
        from honf_runtime.run_layout import RunLayout
        summary=RunLayout(args.output).read_path('fit_summary.json');completed=start
        if summary.exists():completed=max(start,int(read(summary).get('completed_epoch',start)))
        progress=RunLayout(args.output).read_path('progress.json')
        observed_updates=(completed-start)*(4 if recipe['dataset']=='thermal' else 5)
        if progress.exists():
            current=read(progress)
            inherited=0
            if resume is not None:inherited=sum(int(row['optimizer_updates']) for row in payload.get('history',()))
            progress_updates=int(current.get('completed_optimizer_updates',0))+int(current.get('updates_this_epoch',0))
            observed_updates=max(observed_updates,progress_updates-inherited)
        if step_journal.exists():
            observed_updates=sum(1 for line in step_journal.read_text().splitlines() if line.strip())-initial_journal_steps
        row={'schema':'round9_usage_v1','gpu_uuid':GPU_UUID,'arm':recipe['arm'],'dataset':recipe['dataset'],
            'output':str(args.output.resolve()),'start_epoch':start,'completed_epoch':completed,
            'optimizer_updates':observed_updates,
            'associated_seconds':time.perf_counter()-started,'status':status,'recipe_sha256':canonical_sha256(recipe)}
        with (root/'receipts/round9_usage.jsonl').open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
    return 0

if __name__=='__main__':raise SystemExit(main())
