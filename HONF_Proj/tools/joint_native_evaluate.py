#!/usr/bin/env python3
"""Frozen native Round9 physical boards and saved-reference response replay."""
from __future__ import annotations
import argparse,hashlib,json,math,os,shutil,subprocess,sys,threading,time
from contextlib import contextmanager
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
for directory in (PROJECT/'src',PROJECT/'tools',PROJECT/'Case_ThermalChannel/src',PROJECT/'Case_WindFarm/src'):
    sys.path.insert(0,str(directory))


def storage_guard(reservation,output,projected_bytes):
    reservation=Path(reservation).resolve(strict=True);root=reservation.parents[1]
    value=json.loads(reservation.read_text());output=Path(output).resolve()
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T','/data2'],text=True))['filesystems'][0]
    if any(mount.get(key)!=value['mount'][key] for key in ('target','source','fstype')):
        raise ValueError('Physical evidence mount differs from the Round9 reservation.')
    if not output.is_relative_to(root) or output.exists():raise ValueError('Evidence must be a fresh reserved-root output.')
    actual=sum(path.stat().st_size for path in root.rglob('*') if path.is_file() and not path.is_symlink())
    free=shutil.disk_usage('/data2').free
    if actual+projected_bytes>value['retained_disk_limit_bytes'] or free-projected_bytes<value['disk_min_free_bytes']:
        raise OSError('Physical evidence would exceed retained budget or filesystem headroom.')
    if value['estimated_peak_temporary_bytes']>value['temporary_disk_limit_bytes'] or actual+projected_bytes+value['estimated_peak_temporary_bytes']>value['peak_new_disk_limit_bytes']:
        raise OSError('Physical evidence would exceed the shared temporary/peak disk reservation.')
    return {'reservation':str(reservation),'projected_max_new_bytes':projected_bytes,
        'estimated_peak_temporary_disk_bytes':0,'free_before_bytes':free,'retained_actual_before_bytes':actual,
        'minimum_free_bytes':value['disk_min_free_bytes'],'dataset_copies':0,'new_persistent_caches':0}


def gpu_admission(resources):
    """Fail before CUDA allocation unless the reserved physical device is free."""
    now=time.time();elapsed=now-resources['start_unix']
    if elapsed+resources['closeout_reserved_seconds']>=resources['elapsed_limit_seconds']:
        raise RuntimeError('Evaluation would consume reserved round closeout time.')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=resources['gpu_uuid']:
        raise ValueError('Round9 evaluation requires the reserved GPU UUID mapping.')
    text=subprocess.check_output(['nvidia-smi',f"--id={int(resources['physical_gpu'])}",
        '--query-gpu=index,uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5)
    fields=[part.strip() for part in text.strip().split(',')]
    if len(fields)!=4 or int(fields[0])!=resources['physical_gpu'] or fields[1]!=resources['gpu_uuid']:
        raise RuntimeError('Reserved physical GPU index/UUID binding differs.')
    free,total=int(fields[2])*1024**2,int(fields[3])*1024**2
    floor=max(resources['gpu_min_free_bytes'],8*1024**3,math.ceil(.2*total))
    if free<floor+resources['preventive_margin_bytes']:
        raise RuntimeError('Physical GPU free memory is below its floor plus preventive margin.')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True,timeout=5)
    occupied=[]
    for line in apps.splitlines():
        if not line.strip():continue
        uuid,pid=(part.strip() for part in line.split(','))
        if uuid==resources['gpu_uuid'] and int(pid)!=os.getpid():occupied.append(int(pid))
    if occupied:raise RuntimeError(f'Reserved evaluation GPU has other compute PIDs: {occupied}')
    return {'checked_unix':now,'gpu_uuid':resources['gpu_uuid'],'physical_gpu':resources['physical_gpu'],
        'gpu_free_bytes':free,'gpu_floor_bytes':floor,'preventive_margin_bytes':resources['preventive_margin_bytes'],
        'other_compute_pids':occupied,'remaining_before_closeout_seconds':resources['elapsed_limit_seconds']-elapsed-resources['closeout_reserved_seconds']}


@contextmanager
def evaluation_resources(args,storage):
    if not args.device.startswith('cuda'):
        yield lambda:None;return
    if args.device not in ('cuda','cuda:0'):raise ValueError('The reserved UUID is exposed only as logical cuda:0.')
    reservation=Path(args.resource_reservation).resolve();root=reservation.parents[1]
    resources=json.loads(reservation.read_text());admission=gpu_admission(resources)
    from honf_runtime.joint_native_resources import JointResourceMonitor
    class EvaluationMonitor(JointResourceMonitor):
        def __init__(self,*values,**kwargs):
            super().__init__(*values,**kwargs);self.abort=threading.Event();self.reasons=()
        def request_stop(self,reasons):
            # Evaluation has no resumable training epoch. Keep the immutable
            # evidence directory untouched; the CLI raises at the next boundary.
            self.reasons=tuple(reasons);self.abort.set()
    token=hashlib.sha256(str(Path(args.output).resolve()).encode()).hexdigest()[:16]
    control=root/'receipts'/f'evaluation_resources_{token}'
    if control.exists():raise FileExistsError('Resource-control receipts are immutable.')
    monitor=EvaluationMonitor(root,control,resources,interval=1.)
    status='failed';last=admission;failure=None
    def check():
        nonlocal last
        if monitor.abort.is_set():raise RuntimeError(f'Evaluation resource stop requested: {monitor.reasons}')
        last_sample=monitor.last_sample if monitor.last_sample is not None else admission['checked_unix']
        if not monitor.thread.is_alive() or time.time()-last_sample>8.:
            raise RuntimeError('Evaluation resource monitor is unavailable or stale.')
        last=gpu_admission(resources)
    storage['gpu_resource_guard']={'admission':admission,'control_receipts':str(control),'sample_interval_seconds':1.,
        'stop_boundaries':'native case batches and saved-response input states; not inside an individual CUDA operation'}
    with monitor:
        try:
            check();yield check;check();status='completed'
        except BaseException as error:
            failure=f'{type(error).__name__}: {error}';raise
        finally:
            control.mkdir(parents=True,exist_ok=True)
            with (control/'result.json').open('x') as stream:
                json.dump({'status':status,'evaluation_output':str(Path(args.output).resolve()),'admission':admission,
                    'last_boundary_admission':last,'monitor_stop_reasons':monitor.reasons,
                    'last_sample_unix':monitor.last_sample,'error':failure,'CLEAN_STOP_written':False},stream,indent=2);stream.write('\n')


@contextmanager
def evaluation_boundaries(check):
    """Install evaluation-only callbacks without changing core/training code."""
    import honf_runtime.joint_native_evaluation as boards
    import honf_runtime.joint_native_physics as physics
    originals=[(module,'native_batches',module.native_batches) for module in (boards,physics)]
    originals.append((physics,'_response_structure',physics._response_structure))
    for module,name,original in originals:
        if name=='native_batches':
            def wrapped(*args,_original=original,**kwargs):
                check()
                for batch in _original(*args,**kwargs):check();yield batch;check()
        else:
            def wrapped(*args,_original=original,**kwargs):check();return _original(*args,**kwargs)
        setattr(module,name,wrapped)
    try:yield
    finally:
        for module,name,original in originals:setattr(module,name,original)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    compare=sub.add_parser('compare');compare.add_argument('--board',action='append',required=True,metavar='ARM=RESULT_JSON')
    compare.add_argument('--output',type=Path,required=True);compare.add_argument('--resource-reservation',type=Path,required=True)
    for name in ('panel','board','effects','responses','seams'):
        p=sub.add_parser(name);p.add_argument('--checkpoint',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True);p.add_argument('--resource-reservation',type=Path,required=True)
        p.add_argument('--device',default='cpu')
        if name in ('board','effects','seams'):p.add_argument('--panel',type=Path,required=True)
        if name in ('board','effects'):p.add_argument('--split',choices=('train','validation'),default='validation')
        if name=='board':p.add_argument('--exports',choices=('none','representatives'),default='representatives')
        if name=='responses':p.add_argument('--inventory',type=Path,required=True)
        if name=='seams':p.add_argument('--crossings',type=Path,help='Replay exact coordinates from an H/S seam result.json on this arm.')
    args=parser.parse_args(argv)
    projected=1024**2 if args.command in ('panel','compare') else 512*1024**2
    storage=storage_guard(args.resource_reservation,args.output,projected)
    if args.command=='compare':
        from honf_runtime.joint_native_evaluation import compare_physical_boards,write_json_new,file_sha256
        boards={};sources={}
        for item in args.board:
            label,separator,path=item.partition('=')
            if not separator or label in boards:raise ValueError('Comparison boards need unique ARM=RESULT_JSON entries.')
            boards[label]=json.loads(Path(path).read_text());sources[label]={'path':str(Path(path).resolve()),'sha256':file_sha256(path)}
        result=compare_physical_boards(boards);result.update(sources=sources,storage=storage);write_json_new(args.output,result)
        print(json.dumps({'command':'compare','output':str(args.output),'status':'completed'}));return 0
    with evaluation_resources(args,storage) as check:
        with evaluation_boundaries(check):
            completed=run_evaluation(args,storage,check)
    print(json.dumps(completed));return 0


def run_evaluation(args,storage,check):
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.70)
    from content_hypergraph_evaluate import load_evaluation_components
    from honf_runtime.joint_native_evaluation import freeze_native_panel,write_json_new,evaluate_board,file_sha256
    check()
    model,provider,payload=load_evaluation_components(args.checkpoint,args.device)
    check()
    execution={'device':args.device,'torch_num_threads_after_native_builder':torch.get_num_threads(),
        'torch_num_interop_threads_after_native_builder':torch.get_num_interop_threads(),
        'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS'),
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    recipe=payload['experiment_identity']['recipe'];dataset=recipe['dataset']
    if args.command=='panel':
        result=freeze_native_panel(provider,dataset,source_checkpoint=args.checkpoint)
        write_json_new(args.output,result)
    elif args.command=='board':
        panel=json.loads(args.panel.read_text())
        result=evaluate_board(model,provider,payload,panel,args.split,args.output,exports=args.exports)
    else:
        from honf_runtime.joint_native_physics import evaluate_effects,evaluate_saved_responses,evaluate_seams
        if args.command=='responses':
            result=evaluate_saved_responses(model,provider,payload,json.loads(args.inventory.read_text()),args.output)
        else:
            panel=json.loads(args.panel.read_text())
            result=(evaluate_effects(model,provider,payload,panel,args.split,args.output) if args.command=='effects'
                    else evaluate_seams(model,provider,payload,panel,args.output,
                        crossings=None if args.crossings is None else json.loads(args.crossings.read_text())))
    if args.command!='panel':
        from honf_runtime.joint_native_evaluation import canonical_sha256
        write_json_new(args.output/'provenance.json',{'checkpoint':str(args.checkpoint.resolve()),
            'checkpoint_sha256':file_sha256(args.checkpoint),'completed_epoch':int(payload['epoch']),
            'arm':payload['experiment_identity'].get('treatment'),
            'provider_binding_sha256':canonical_sha256(payload['experiment_identity']['provider_identity']),
            'panel_path':str(args.panel.resolve()) if hasattr(args,'panel') else None,'storage':storage,
            'measured_execution_after_native_builder':execution,
            'crossings_source':({'path':str(args.crossings.resolve()),'sha256':file_sha256(args.crossings)}
                if getattr(args,'crossings',None) is not None else None),
            'free_after_bytes':shutil.disk_usage('/data2').free})
    return {'command':args.command,'output':str(args.output),'dataset':dataset,
            'epoch':int(payload['epoch']),'status':'completed'}

if __name__=='__main__':raise SystemExit(main())
