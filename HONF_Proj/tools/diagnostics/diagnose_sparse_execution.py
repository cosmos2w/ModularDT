"""Bounded, read-only source-measure margin and actual Boolean-union audit.

Query-temperature probes change only query projection; source temperatures stay
at one. The parent checkpoint is loaded through the maintained trusted loader.
No checkpoint, run, or baseline copy is written.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dynamic_sparse_routing_study as study
from benchmark_routing_optimization import _build_loader, _make_buckets
from channelthermal.evaluation.prepared import select_sample

from honf_forward_core.interface_fields.routing_index.sparse_projection import source_measure_sparsemax


def summary(x):
    x = x.detach().double().reshape(-1)
    return {"count": x.numel(), "min": x.min().item(), "mean": x.mean().item(),
            "max": x.max().item()} if x.numel() else {"count": 0}


@contextlib.contextmanager
def capture(model, rows, temperature):
    backend = model.core.backend
    original = backend._query_density

    def query(encoded, receivers, features, candidates, incidence, source_type,
              query_descriptors=None):
        descriptors = (backend._query_descriptors(encoded, receivers, features)
                       if query_descriptors is None else query_descriptors)
        logits = backend._query_logits(encoded, receivers, features, candidates,
                                       source_type, query_descriptors=descriptors)
        mu = incidence.hub_measure.double()
        valid = (candidates.valid[:, None, :] & (mu[:, None, :] > 0)).expand_as(logits)
        u = logits.double() / temperature
        projection = source_measure_sparsemax(u, mu, valid)
        total = mu.sum(-1, keepdim=True)
        tau = ((u * mu[:, None]).sum(-1) - 1) / total.clamp_min(1e-300)
        minimum = u.masked_fill(~valid, torch.inf).amin(-1)
        margin = minimum - tau
        active = projection.density > 0
        source = incidence.membership > 0
        positive = incidence.source_weights > 0
        source = source & positive[..., None]
        union = torch.zeros((*u.shape[:2], source.shape[1]), dtype=torch.bool, device=u.device)
        # Bounded probe only: one Boolean QxN plane, never QxKxN paths.
        for k in range(u.shape[-1]):
            union |= active[:, :, k, None] & source[:, None, :, k]
        counts = union.sum(-1)
        complete = counts == positive.sum(-1)[:, None]
        raw = (active.sum(1) * source.sum(1)).sum()
        content = backend.routing_content_scale * torch.einsum('bqd,bkd->bqk', descriptors, candidates.descriptors)
        delta = (receivers[:, :, None] - candidates.coords[:, None]) / backend._length_scale(encoded, receivers)
        geometry = -backend.routing_geometry_scale * torch.log1p(delta.square().sum(-1))
        propensity = (backend.routing_propensity_scale * candidates.propensity[:, None]).expand_as(logits)
        resistance = logits - content - geometry - propensity
        parts = {"content": content, "geometry": geometry, "propensity": propensity,
                 "negative_resistance_roundoff": resistance}
        spreads = {name: summary(part.masked_fill(~valid, -torch.inf).amax(-1)
                                  - part.masked_fill(~valid, torch.inf).amin(-1))
                   for name, part in parts.items()}
        rows.append({"phase": str(getattr(model.core, '_interface_read_role', 'unscoped')),
                     "type": source_type, "batch": u.shape[0], "queries_per_case": u.shape[1],
                     "valid_sources_per_case": positive.sum(-1).tolist(),
                     "mu": mu.tolist(), "margin_min_u_minus_tau_all": summary(margin[total.expand_as(margin) > 0]),
                     "all_occupied_active_receivers": int((active == valid).all(-1).sum()),
                     "receiver_denominator": complete.numel(), "complete_receivers": int(complete.sum()),
                     "complete_cases_this_chunk": int(complete.all(-1).sum()), "case_chunk_denominator": u.shape[0],
                     "unique_pairs": int(counts.sum()), "dense_valid_pairs": int(positive.sum() * u.shape[1]),
                     "logical_raw_paths": int(raw), "query_support": summary(active.sum(-1)),
                     "source_support": summary(source.sum(-1)[positive]),
                     "density": summary(projection.density[valid]), "probability": summary(projection.probability[valid]),
                     "affinity_across_hub_spreads": spreads,
                     "routing_length_scale": backend._length_scale(encoded, receivers).tolist()})
        return logits, projection

    backend._query_density = query
    try:
        yield
    finally:
        backend._query_density = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--query-count', type=int, default=32)
    args = p.parse_args()
    device = torch.device(args.device)
    model, checkpoint = study._load_model_spec(study.parse_checkpoint_specs([args.checkpoint])[0], device)
    payload = {"checkpoint": args.checkpoint, "epoch": checkpoint['epoch'],
               "query_count": args.query_count, "anchors": [], "training_batches": [],
               "scope": "projection diagnosis; no full-population fidelity evaluation"}
    dataset, _ = study._load_dataset(checkpoint, SimpleNamespace(dataset=None, split='test'))
    with torch.no_grad():
        for case in study.ANCHOR_CASE_IDS:
            sample = select_sample(dataset, case, 0)
            baseline = None
            for temperature in (1., .5, .25, .125):
                rows = []
                with capture(model, rows, temperature):
                    outputs = study._forward_batch(model, sample, study._query_points(sample, args.query_count), device)
                field = outputs['pred_field']
                if baseline is None:
                    baseline = field.clone()
                payload['anchors'].append({"case": case, "query_temperature": temperature,
                    "full_loop_field_relative_change": float(torch.linalg.vector_norm(field-baseline) / torch.linalg.vector_norm(baseline)),
                    "reads": rows})
        dataset.close()
        dataset, _ = study._load_dataset(checkpoint, SimpleNamespace(dataset=None, split='train'),
                         points_per_case_override=args.query_count, random_point_sampling_override=False)
        dataset.include_grid = False
        for bucket in _make_buckets([dataset], batch_size=48, requested='both'):
            batch = next(iter(_build_loader(dataset, checkpoint, bucket, batch_size=48)))
            def move(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device)
                if isinstance(x, dict):
                    return {k: move(v) for k, v in x.items()}
                return x
            batch = move(batch)
            rows = []
            with capture(model, rows, 1.):
                model(batch['structure'], batch['query_xy'], **study._profile_forward_kwargs(batch))
            payload['training_batches'].append({"bucket": bucket.label, "case_ids": bucket.case_ids, "reads": rows})
        dataset.close()
    study.write_json(args.output, payload)
    print(args.output)


if __name__ == '__main__':
    main()
