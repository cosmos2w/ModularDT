"""Actual-checkpoint full physical forward/first-derivative executor comparison."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dynamic_sparse_routing_study as study
from benchmark_sparse_execution import set_mode
from channelthermal.evaluation.prepared import select_sample


def compare(a, b):
    a, b = a.double(), b.double()
    difference = torch.linalg.vector_norm(a-b).item()
    reference = torch.linalg.vector_norm(a).item()
    return {'relative_norm': difference/max(reference, 1e-30), 'absolute_norm': difference,
                'max_absolute': (a-b).abs().max().item() if a.numel() else 0., 'reference_norm': reference,
                'finite': bool(torch.isfinite(b).all())}


def run(spec, device, case, execution, backend):
    model, checkpoint = study._load_model_spec(spec, device)
    set_mode(model, execution, backend)
    # eval leaves dropout disabled while maintaining all physical derivatives.
    model.eval()
    dataset, _ = study._load_dataset(checkpoint, SimpleNamespace(dataset=None, split='test'))
    sample = select_sample(dataset, case, 0)
    batch = study.make_batch(dict(sample), study._query_points(sample, 32), device)
    query = batch['query_xy'].detach().requires_grad_(True)
    centers = batch['structure']['module_centers'].detach().requires_grad_(True)
    batch['structure']['module_centers'] = centers
    output = model(batch['structure'], query, return_routing_maps=False, **study._profile_forward_kwargs(batch))
    tensors = {k: v for k, v in output.items() if torch.is_tensor(v) and
               (k.startswith('pred_') or 'port_tokens' in k) and v.is_floating_point()}
    # One fixed scalar exercises the complete physical field and all exposed
    # prediction heads, without being mislabeled as the supervised train loss.
    loss = sum(v.square().mean() for v in tensors.values() if v.numel() and v.requires_grad)
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    gradients = torch.autograd.grad(loss, [query, centers]+[p for _, p in params], allow_unused=True)
    result = {'outputs': {k: v.detach().cpu() for k, v in tensors.items()}, 'loss': loss.detach().cpu(),
                  'gradients': {k: (v.detach().cpu() if v is not None else None)
                             for k, v in zip(['query_coordinates', 'module_coordinates']+[n for n, _ in params], gradients)}}
    dataset.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['torch', 'triton'], default='torch')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    spec = study.parse_checkpoint_specs([args.checkpoint])[0]
    device = torch.device(args.device)
    result = {'checkpoint': str(spec.path), 'backend': args.backend, 'rows': []}
    for case in ('0273', '0653'):
        reference = run(spec, device, case, 'optimized_exact', 'torch')
        gc.collect()
        candidate = run(spec, device, case, 'compiled_exact', args.backend)
        row = {'case': case, 'outputs': {k: compare(v, candidate['outputs'][k]) for k, v in reference['outputs'].items()},
                   'loss': compare(reference['loss'], candidate['loss']), 'gradients': {}, 'disconnected_changes': []}
        for name, gradient in reference['gradients'].items():
            other = candidate['gradients'][name]
            if gradient is None or other is None:
                if (gradient is None) != (other is None):
                    row['disconnected_changes'].append(name)
            else:
                row['gradients'][name] = compare(gradient, other)
        result['rows'].append(row)
        study.write_json(args.output, result)
    # Softmax key biases have an analytically zero gradient; FP32 cancellation
    # leaves reference norms ~1e-6 for this loss. Use an explicit 1e-6 absolute
    # check only below reference norm 1e-5; all other gradients use rtol1e-4.
    failed = any(r['disconnected_changes'] or
                 any(not v['finite'] or (v['relative_norm'] > 2e-5 and v['max_absolute'] > 1e-7)
                     for v in r['outputs'].values()) or
                 any(not v['finite'] or (v['relative_norm'] > 1e-4 and not (v['reference_norm'] < 1e-5 and v['max_absolute'] < 1e-6))
                     for v in r['gradients'].values()) for r in result['rows'])
    result['status'] = 'failed' if failed else 'passed'
    result['tolerances'] = {'output_relative_norm': 2e-5, 'gradient_relative_norm': 1e-4,
                            'near_zero_reference_gradient_norm': 1e-5, 'near_zero_gradient_absolute': 1e-6}
    study.write_json(args.output, result)
    print(args.output, 'FAIL' if failed else 'PASS')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
