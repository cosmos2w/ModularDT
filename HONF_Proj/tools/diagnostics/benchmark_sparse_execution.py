"""Same-checkpoint full-model exact-executor measurements in isolated processes.

Run each executor/backend in a fresh process with the same explicit checkpoint.
Uses real data, the full physical loop and maintained training loss/optimizer.
Never allocates a run or saves model weights. Timings exclude diagnostic hooks.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_routing_optimization as training
import run_dynamic_sparse_routing_study as study
from channelthermal.evaluation.prepared import select_sample


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(fn, device, warmups, repetitions):
    sync(device)
    start = time.perf_counter()
    for _ in range(warmups):
        fn()
    sync(device)
    warmup_seconds = time.perf_counter() - start
    samples = []
    for _ in range(repetitions):
        sync(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        wall_start = time.time()
        start = time.perf_counter()
        result = fn()
        sync(device)
        elapsed = time.perf_counter() - start
        samples.append({'seconds': elapsed, 'start_unix_seconds': wall_start,
                            'end_unix_seconds': time.time(), 'baseline_allocated_mib': baseline/2**20,
                            'peak_allocated_mib': torch.cuda.max_memory_allocated(device)/2**20,
                            'incremental_allocated_mib': (torch.cuda.max_memory_allocated(device)-baseline)/2**20,
                            'peak_reserved_mib': torch.cuda.max_memory_reserved(device)/2**20})
        del result
        samples[-1]['post_call_live_mib'] = torch.cuda.memory_allocated(device)/2**20
    return {'warmups': warmups, 'repetitions': repetitions, 'warmup_seconds': warmup_seconds,
                'median_seconds': float(np.median([s['seconds'] for s in samples])), 'samples': samples}


def set_mode(model, execution, backend):
    field = getattr(model.core, 'backend', None)
    if hasattr(field, 'routing_execution'):
        field.routing_execution = execution
        field.dense_environment_fast_path = execution != 'gathered'
        field.qe_backend = backend
        # Compatibility with the maintained integration name, if present.
        if hasattr(field, 'routing_qe_backend'):
            field.routing_qe_backend = backend


def qe_observation(model):
    """Return the last QE dispatch observation without changing model state."""
    field = getattr(getattr(model, 'core', None), 'backend', None)
    if field is None:
        return {'actual_qe_backend': None, 'qe_backend_reason': 'field_unavailable'}
    return {
        'actual_qe_backend': getattr(field, 'last_qe_backend', None),
        'qe_backend_reason': getattr(field, 'last_qe_backend_reason', None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execution', choices=['optimized_exact', 'compiled_exact'], required=True)
    parser.add_argument('--backend', choices=['torch', 'triton'], default='torch')
    parser.add_argument('--task', choices=['inference', 'train'], default='inference')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--receiver-chunk-size', type=int, default=2048)
    parser.add_argument('--query-count', type=int, default=8192)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device)
    spec = study.parse_checkpoint_specs([args.checkpoint])[0]
    model, checkpoint = study._load_model_spec(spec, device)
    set_mode(model, args.execution, args.backend)
    payload = {'checkpoint': str(spec.path), 'epoch': checkpoint['epoch'], 'execution': args.execution,
                   'requested_qe_backend': args.backend, 'task': args.task,
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                   'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(device),
                   'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                   'receiver_chunk_size': args.receiver_chunk_size, 'query_count': args.query_count, 'rows': []}
    if args.task == 'inference':
        dataset, _ = study._load_dataset(checkpoint, SimpleNamespace(dataset=None, split='test'))
        with torch.inference_mode(), study._runtime_receiver_chunk_size(model, args.receiver_chunk_size):
            for case in ('0273', '0653'):
                sample = select_sample(dataset, case, 0)
                batch = study.make_batch(dict(sample), study._query_points(sample, args.query_count), device)
                kwargs = study._profile_forward_kwargs(batch)
                def full(batch=batch, kwargs=kwargs):
                    return model(batch['structure'], batch['query_xy'], return_routing_maps=False, **kwargs)
                def prepare(batch=batch, kwargs=kwargs):
                    return model(batch['structure'], batch['query_xy'][:, :1], return_prepared_state=True,
                                 return_routing_maps=False, **kwargs)
                phases = {'full_forward': measure(full, device, args.warmups, args.repetitions)}
                phases['full_forward'].update(qe_observation(model))
                phases['prepare_plus_one'] = measure(prepare, device, args.warmups, args.repetitions)
                phases['prepare_plus_one'].update(qe_observation(model))
                prepared = prepare()['prepared_state']
                def decode(prepared=prepared, batch=batch):
                    return model.decode_prepared(prepared, batch['query_xy'], return_routing_maps=False,
                                                 receiver_chunk_size=args.receiver_chunk_size)
                phases['prepared_decode'] = measure(decode, device, args.warmups, args.repetitions)
                phases['prepared_decode'].update(qe_observation(model))
                del decode, prepared
                row = {'case': case, 'phases': phases}
                if args.trace:
                    row['profile'] = study._profiler_trace(full, args.output.with_name(args.output.stem+'_'+case+'_trace.json'),
                                                          'compiled_exact.full_forward', device)
                payload['rows'].append(row)
                study.write_json(args.output, payload)
        dataset.close()
    else:
        dataset, _ = study._load_dataset(checkpoint, SimpleNamespace(dataset=None, split='train'),
                         points_per_case_override=args.query_count, random_point_sampling_override=False)
        dataset.include_grid = False
        buckets = training._make_buckets([dataset], batch_size=48, requested='both')
        del model
        gc.collect()
        for bucket in buckets:
            torch.manual_seed(42)
            np.random.seed(42)
            model, checkpoint = study._load_model_spec(spec, device)
            set_mode(model, args.execution, args.backend)
            model.train()
            loader = training._build_loader(dataset, checkpoint, bucket, batch_size=48)
            schedule = training._training_schedule(checkpoint)
            optimizer, inventory = training.build_forward_optimizer(model, schedule[0])
            state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))
            policy = 'fresh_parent_policy'
            if state is not None:
                training._validate_optimizer_resume_compatibility(checkpoint, inventory)
                optimizer.load_state_dict(state)
                policy = 'restored_parent'
            metrics = []
            def step(model=model, loader=loader, checkpoint=checkpoint, optimizer=optimizer, schedule=schedule, metrics=metrics):
                value = training._run_step(model, loader, device, checkpoint, optimizer=optimizer, training_schedule=schedule)
                metrics.append(value)
                return value
            measured = measure(step, device, args.warmups, args.repetitions)
            measured.update(qe_observation(model))
            payload['rows'].append({'bucket': bucket.label, 'case_ids': bucket.case_ids, 'batch_size': 48,
                                        'optimizer_policy': policy, 'measurement': measured, 'metrics': metrics})
            study.write_json(args.output, payload)
            del step, optimizer, model, loader
            gc.collect()
        dataset.close()
    print(args.output)


if __name__ == '__main__':
    main()
