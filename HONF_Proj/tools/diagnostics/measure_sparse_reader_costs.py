"""Fixed-support marginal reader timing; negative/unstable slopes imply equal weights."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dynamic_sparse_routing_study as study
from benchmark_sparse_execution import measure
from channelthermal.evaluation.prepared import select_sample

from honf_forward_core.interface_fields.routing_index import PackedPairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    device = torch.device('cuda:0')
    spec = study.parse_checkpoint_specs([args.checkpoint])[0]
    model, checkpoint = study._load_model_spec(spec, device)
    dataset, _ = study._load_dataset(checkpoint, SimpleNamespace(dataset=None, split='test'))
    field = model.core.backend
    records = []
    with torch.inference_mode():
        for case in ('0273', '0653'):
            saved = []
            original = field.read
            def capture(*a, saved=saved, original=original, **kw):
                if getattr(model.core, '_interface_read_role', None) == 'p2_field':
                    saved.append(a)
                return original(*a, **kw)
            field.read = capture
            sample = select_sample(dataset, case, 0)
            try:
                study._forward_batch(model, sample, study._query_points(sample, 128), device)
            finally:
                field.read = original
            state, encoded, receivers, features = saved[0][:4]
            q = receivers.shape[1]
            for kind in ('module', 'environment'):
                valid = (encoded.module_present[0] > .5 if kind == 'module'
                         else encoded.env_weights.reshape(-1) > 0)
                indices = torch.nonzero(valid).flatten()
                for count in sorted({max(1, len(indices)//2), len(indices)}):
                    sources = indices[:count].repeat(q)
                    receivers_index = torch.arange(q, device=device).repeat_interleave(count)
                    pairs = PackedPairs(torch.zeros_like(sources), receivers_index, sources,
                                        torch.full((q*count,), 1/count, dtype=torch.float64, device=device),
                                        q*count, q*count)
                    if kind == 'module':
                        def call(pairs=pairs, state=state, encoded=encoded, receivers=receivers):
                            return field.read_module_pairs(state, encoded, receivers, pairs)
                    else:
                        def call(pairs=pairs, state=state, encoded=encoded, receivers=receivers, features=features):
                            return field.read_environment_pairs(state, encoded, receivers, features, pairs)
                    records.append({'case': case, 'type': kind, 'sources': count, 'queries': q,
                                    'measurement': measure(call, device, 2, 5)})
    slopes = {'module': [], 'environment': []}
    for case in ('0273', '0653'):
        for kind, values in slopes.items():
            rows = sorted([r for r in records if r['case'] == case and r['type'] == kind], key=lambda r:r['sources'])
            if len(rows) == 2:
                values.append((rows[1]['measurement']['median_seconds']-rows[0]['measurement']['median_seconds'])/
                                    ((rows[1]['sources']-rows[0]['sources'])*rows[0]['queries']))
    stable = all(len(v) == 2 and min(v) > 0 and max(v)/min(v) <= 2 for v in slopes.values())
    costs = {'module': float(np.mean(slopes['module'])/np.mean(slopes['environment'])) if stable else 1.,
             'environment': 1.}
    study.write_json(args.output, {'checkpoint': str(spec.path), 'rows': records, 'marginal_seconds_per_pair': slopes,
        'stable_positive_slopes': stable, 'fixed_costs': costs,
        'interpretation': 'marginal reader cost' if stable else 'equal-weight pair-count objective; marginal timing estimates unstable',
        'scope': 'fixed supports through historical exact Torch reader, not full-model speedup'})
    dataset.close()
    print(args.output)


if __name__ == '__main__':
    main()
