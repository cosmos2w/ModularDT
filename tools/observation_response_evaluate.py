#!/usr/bin/env python3
"""Evaluate shared observation responses on an explicitly declared pair panel."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .relation_anchored_train import build, provider_identity_matches_for_inference
except ImportError:
    from relation_anchored_train import build, provider_identity_matches_for_inference

import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.observation_blocks import observation_response_losses
from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.storage_policy import _ensure_workspace_storage_bindings, verify_storage
from honf_runtime.unified_training import _atomic_json


def _numbers(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numbers(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def evaluate_observation_panel(model, provider, pairs, *, view_names=('full', 'restricted')):
    """Reuse each pair prediction for separate physical observation views.

    The fixed draw namespace is independent of model errors and checkpoint age.
    Results average pairs; callers retain family grouping for correlated edits.
    This routine never selects a larger population than its supplied panel.
    """
    pairs = tuple(tuple(map(str, pair)) for pair in pairs)
    if not pairs:
        raise ValueError('Observation evaluation requires an explicitly declared nonempty pair panel.')
    bind = getattr(provider, 'with_observation_evaluation_pairs', None)
    if callable(bind):
        provider = bind(pairs)
    rows = []
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for pair in pairs:
                _, work, details = provider.observation_components(model, tuple(pair), epoch=10, details=True)
                predictions = details['predictions']
                if any(pred.shape[0] != len(view_names) for pred in predictions):
                    raise ValueError('View labels differ from the executed observation prediction views.')
                views = {}
                for index, name in enumerate(view_names):
                    result = observation_response_losses(details['blocks'],
                        [prediction[index:index+1] for prediction in predictions], details['draws'],
                        include_component_estimates=True)
                    views[name] = _numbers(result)
                rows.append({'pair': list(pair), 'views': views, 'work': _numbers(work)})
    finally:
        model.train(was_training)
    names = ('extra_absolute', 'point_response', 'spatial_response')
    summary = {view: {name: sum(row['views'][view][name] for row in rows) / len(rows)
                      for name in names} for view in view_names}
    return {'definition': 'Equal-pair mean of fixed-mass block estimators; sampled physical observations with replacement. '
                          'Response differences compare the entire observed input change. '
                          'Full and restricted readers are measured separately; no training coefficient enters these raw errors.',
            'sampling_epoch_namespace': 10, 'pair_count': len(rows), 'summary': summary, 'pairs': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='latest')
    parser.add_argument('--pair-panel', type=Path, required=True,
                        help='JSON containing an explicit nonempty pairs list and its panel name')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    recipe = json.loads(args.recipe.read_text())
    if recipe.get('training_profile') != 'observation_response':
        raise ValueError('This command evaluates a shared observation-response training profile.')
    panel = json.loads(args.pair_panel.read_text())
    pairs = tuple(tuple(map(str, pair)) for pair in panel.get('pairs', ()))
    if (not panel.get('name') or not pairs or len(set(pairs)) != len(pairs)
            or any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs)):
        raise ValueError('Declare a named panel with distinct pairs of two different case IDs.')
    if args.output.exists():
        raise FileExistsError('Retain the existing evaluation instead of repeating it.')
    _ensure_workspace_storage_bindings()
    verify_storage(recipe, output=args.output, estimated_bytes=16*1024**2, peak_temporary_bytes=16*1024**2)
    checkpoint = resolve_checkpoint(args.run, args.checkpoint)
    saved = load_trusted_checkpoint(checkpoint, map_location='cpu')
    identity = saved['experiment_identity']
    if recipe != identity['recipe']:
        raise ValueError('The evaluation recipe differs from the checkpoint recipe.')
    model, provider = build(identity['recipe'], device=args.device, resume_payload=saved)
    if not provider_identity_matches_for_inference(recipe, identity['provider_identity'], dict(provider.identity_payload())):
        raise ValueError('The observation provider differs from its saved scientific definition.')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    provider.load_training_state_dict(saved['provider_training_state'])
    model.to(args.device)
    result = evaluate_observation_panel(model, provider, pairs)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.synchronize(args.device)
    result.update(panel=panel, checkpoint=str(checkpoint.resolve()), epoch=int(saved['epoch']),
                  process_seconds=time.perf_counter()-started, device=str(args.device))
    _atomic_json(args.output, result)
    print(json.dumps({'output': str(args.output.resolve()), 'epoch': result['epoch'],
                      'summary': result['summary'], 'process_seconds': result['process_seconds']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
