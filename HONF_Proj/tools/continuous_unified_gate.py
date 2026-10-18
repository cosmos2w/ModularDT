#!/usr/bin/env python3
"""Collect exact TRAIN reviews, admit, and run a matched unified triplet."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
for directory in (PROJECT/'src',PROJECT/'tools',PROJECT/'Case_ThermalChannel/src',PROJECT/'Case_WindFarm/src',PROJECT.parent/'tools'):
    sys.path.insert(0,str(directory))
from honf_runtime.continuous_unified_gate import *
from honf_runtime.continuous_joint_protocol import validate_continuous_joint_recipe


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def check_spec(spec,campaign):
    if spec.get('schema')!='round10_unified_gate_spec_v1' or spec['dataset'] not in ('thermal','wind'):
        raise ValueError('Unknown automatic unified gate specification.')
    if Path(spec['resource_reservation']).resolve()!=campaign.root/'receipts/master_reservation_v1.json':
        raise ValueError('Unified gate must use the exact shared Round10 reservation.')
    if set(spec['recipes'])!={'U10-D','U10-H','U10-S'}:raise ValueError('Unified admission requires the complete matched triplet.')
    if spec.get('board_producer','root_w6') not in ('root_w6','gate_collector'):
        raise ValueError('Unified gate board producer must be explicitly declared.')
    for arm,path in spec['recipes'].items():
        recipe=validate_continuous_joint_recipe(read(path),verify_source=True)
        if recipe['arm']!=arm or recipe['dataset']!=spec['dataset']:raise ValueError('Unified gate recipe/arm/dataset mismatch.')
    for path in (spec['train_board_index'],spec['board_output_root'],*spec['qualification'].values()):
        if not Path(path).resolve().is_relative_to(campaign.root):raise ValueError('New gate evidence must remain under the reserved root.')


def collect_boards(spec,campaign,physical_gpu,reserve_seconds):
    """Complete missing whole-TRAIN boards; reuse verified existing results."""
    import torch
    from honf_round10_resources import Monitor
    from content_hypergraph_evaluate import load_evaluation_components
    from honf_runtime.joint_native_evaluation import evaluate_board
    from joint_native_evaluate import evaluation_boundaries
    registry=board_registry(spec)
    dense={row['epoch']:row['path'] for row in read_bound(spec['dense_checkpoint_index'])['rows'] if row['dataset']==spec['dataset']}
    work=[]
    for epoch in AGES:
        for arm in ('D9-M','C10-H'):
            if arm in registry.get(epoch,{}):continue
            checkpoint=dense[epoch] if arm=='D9-M' else str(Path(spec['track_b_run'])/'checkpoints'/f'epoch_{epoch}_model.pt')
            if Path(checkpoint).is_file():work.append((epoch,arm,checkpoint))
    if not work:return dict(status='no_available_missing_boards',completed=0)
    operation=f"unified_review_{spec['dataset']}_{time.time_ns()}";status='failed';error=None
    control=campaign.root/'receipts'/operation;control.mkdir()
    with campaign.device(physical_gpu) as uuid:
        campaign.reserve(operation,track='C',kind='evaluation',updates=0,seconds=reserve_seconds,physical_gpu=physical_gpu)
        try:
            torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
            free,total=next((int(f)*1024**2,int(t)*1024**2) for u,f,t in (line.split(',') for line in inventory.splitlines()) if u.strip()==uuid)
            available=free-max(8*1024**3,.2*total)-6*1024**3
            if available<=0:raise RuntimeError('No safe allocation margin for selected-TRAIN reviews.')
            torch.cuda.set_per_process_memory_fraction(min(.95,available/total),0)
            with Monitor(campaign,operation,control,uuid) as monitor:
                def boundary():
                    if (control/'CLEAN_STOP_REQUEST.json').exists() or not monitor.thread.is_alive():
                        raise RuntimeError('Unified review resource/time stop; completed evidence retained.')
                with evaluation_boundaries(boundary):
                    for epoch,arm,checkpoint in work:
                        boundary();model,provider,payload=load_evaluation_components(checkpoint,'cuda:0')
                        output=Path(spec['board_output_root'])/f'e{epoch}'/f'{arm}_{time.time_ns()}'
                        result=evaluate_board(model,provider,payload,read_bound(spec['panel']),'train',output,exports='none')
                        source_hash=file_sha256(checkpoint)
                        write_new(output/'round10_provenance.json',dict(schema='round10_native_board_provenance_v1',checkpoint=str(Path(checkpoint).resolve()),
                            checkpoint_sha256=source_hash,literal_epoch=epoch,arm=arm,
                            provider_binding_sha256=canonical_sha256(payload['experiment_identity']['provider_identity']),
                            panel_path=spec['panel']['path'],round10_operation=operation,gpu_uuid=uuid))
                        register_board(spec['train_board_index'],epoch=epoch,arm=arm,result_path=output/'result.json',
                            checkpoint_sha256=source_hash,provenance_path=output/'round10_provenance.json')
                        del model,provider,payload,result;torch.cuda.empty_cache();boundary()
            status='completed';return dict(status=status,completed=len(work),operation=operation)
        except BaseException as exc:error=repr(exc);raise
        finally:campaign.finish(operation,updates=0,status=status,error=error)


def run_command(command,log,*,gpu_uuid):
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid,PYTHONDONTWRITEBYTECODE='1')
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as stream:
        completed=subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError(f'Unified subprocess failed ({completed.returncode}); see {log}')


def run_automatic(spec,campaign,physical_gpu,review_seconds):
    """All conditions already authorize the complete triplet; no per-arm selection."""
    dataset=spec['dataset'];manifest_path=campaign.root/'receipts'/f'{dataset}_unified_admitted_triplet_v1.json'
    uuid=campaign.master['authorized_gpus'][str(physical_gpu)]
    if not manifest_path.exists():
        decision=evaluate_unified_gate(spec,campaign)
        # Whole native reviews can be accumulated before the endpoint. The
        # conditional U epoch probes wait for its actual complete horizon.
        if decision['missing_conditions'] and spec.get('board_producer','root_w6')=='gate_collector':
            collect_boards(spec,campaign,physical_gpu,review_seconds)
            decision=evaluate_unified_gate(spec,campaign)
        blocked=[name for name in decision['missing_conditions'] if not name.endswith('_complete_native_GPU_epoch')]
        if decision['status']=='not_admitted_catastrophic_failure' or blocked:
            path=campaign.root/'receipts'/f'{dataset}_unified_gate_pending_{time.time_ns()}.json'
            write_new(path,decision);return dict(status=decision['status'],decision=str(path),missing=blocked)
        for arm in ('U10-D','U10-H','U10-S'):
            qualification=Path(spec['qualification'][arm])
            if qualification.exists():continue
            output=qualification.parent
            command=[sys.executable,str(PROJECT/'tools/continuous_joint_train.py'),'qualify','--recipe',spec['recipes'][arm],
                '--output',str(output),'--epochs',str(spec['native_qualification_epochs']),
                '--physical-gpu',str(physical_gpu),'--reserve-seconds',str(spec['qualification_reserve_seconds']),'--device','cuda:0',
                '--source-manifest',spec['source_manifest']['path'],'--source-manifest-sha256',spec['source_manifest']['sha256']]
            run_command(command,campaign.root/'receipts'/f'{dataset}_{arm}_qualification_{time.time_ns()}.log',gpu_uuid=uuid)
        decision=evaluate_unified_gate(spec,campaign)
        admission=campaign.root/'receipts'/f'{dataset}_unified_gate_{time.time_ns()}.json';write_new(admission,decision)
        if decision['status']!='admitted':return dict(status=decision['status'],decision=str(admission))
        recipes={}
        for arm in ('U10-D','U10-H','U10-S'):
            recipe=read(spec['recipes'][arm]);recipe['unified_admission']=bound(admission)
            validate_continuous_joint_recipe(recipe,verify_source=True)
            target=campaign.root/'recipes'/f'{dataset}_{arm.replace("-","_")}_admitted_v1.json'
            write_new(target,recipe);recipes[arm]=str(target)
        manifest=dict(schema='round10_unified_admitted_triplet_v1',dataset=dataset,admission=bound(admission),recipes=recipes,
            run_root=str(campaign.root/'runs/unified'/dataset),native_epoch_seconds=decision['native_epoch_seconds'],
            forecast_overhead_factor=spec['forecast_overhead_factor'],validation_checkpoint_io_seconds=spec['validation_checkpoint_io_seconds'])
        write_new(manifest_path,manifest)
    manifest=read(manifest_path);admission=read_bound(manifest['admission'])
    completed=[]
    for arm in ('U10-D','U10-H','U10-S'):
        output=Path(manifest['run_root'])/arm[-1]
        endpoint=output/'checkpoints/epoch_5000_model.pt'
        if endpoint.is_file():
            from honf_runtime.continuous_joint_runtime import load_continuous_joint_checkpoint
            model,adapter,payload=load_continuous_joint_checkpoint(endpoint)
            if payload['epoch']!=5000 or payload['experiment_identity']['recipe']['arm']!=arm:raise ValueError('Unified endpoint belongs to another arm.')
            del model,adapter,payload;completed.append(dict(arm=arm,checkpoint=bound(endpoint)));continue
        latest=output/'checkpoints/latest_model.pt';start=3000
        if latest.is_file():
            from honf_runtime.compat import load_trusted_checkpoint
            start=load_trusted_checkpoint(latest,map_location='cpu')['epoch']
        seconds=manifest['native_epoch_seconds'][arm]*(5000-start)*manifest['forecast_overhead_factor']+manifest['validation_checkpoint_io_seconds']/3+120
        command=[sys.executable,str(PROJECT/'tools/continuous_joint_train.py'),'resume' if latest.is_file() else 'fit',
            '--recipe',manifest['recipes'][arm],'--output',str(output),'--stop-after','5000','--physical-gpu',str(physical_gpu),
            '--reserve-seconds',str(seconds),'--device','cuda:0',
            '--source-manifest',spec['source_manifest']['path'],'--source-manifest-sha256',spec['source_manifest']['sha256']]
        run_command(command,campaign.root/'receipts'/f'{dataset}_{arm}_full_horizon_{time.time_ns()}.log',gpu_uuid=uuid)
        if not endpoint.is_file():raise RuntimeError('Unified fit stopped resumably before its literal endpoint; controls remain admitted.')
        completed.append(dict(arm=arm,checkpoint=bound(endpoint)))
    result=dict(schema='round10_unified_complete_triplet_v1',status='completed_literal_e5000_triplet',dataset=dataset,
                admission=manifest['admission'],endpoints=completed)
    path=campaign.root/'receipts'/f'{dataset}_unified_complete_{time.time_ns()}.json';write_new(path,result)
    return dict(status=result['status'],receipt=str(path))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','collect','run'))
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--physical-gpu',type=int,choices=(1,2));parser.add_argument('--review-reserve-seconds',type=float,default=7200)
    args=parser.parse_args(argv);spec=read(args.spec)
    from honf_round10_resources import Campaign
    campaign=Campaign(Path(spec['resource_reservation']).resolve().parents[1]);check_spec(spec,campaign)
    if args.command=='check':
        result=evaluate_unified_gate(spec,campaign)
        if args.output is not None:write_new(args.output,result)
    else:
        if args.physical_gpu is None:raise ValueError('Collect/run requires a reserved physical GPU lane.')
        if args.command=='collect' and spec.get('board_producer','root_w6')!='gate_collector':
            raise ValueError('Root W6 owns the board queue; this gate is a registry consumer.')
        result=(collect_boards(spec,campaign,args.physical_gpu,args.review_reserve_seconds) if args.command=='collect'
                else run_automatic(spec,campaign,args.physical_gpu,args.review_reserve_seconds))
    print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
