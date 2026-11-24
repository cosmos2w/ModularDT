#!/usr/bin/env python3
"""Compare within-role access priors on an explicit saved observation panel."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from .relation_anchored_train import build, provider_identity_matches_for_inference
except ImportError:
    from relation_anchored_train import build, provider_identity_matches_for_inference
from honf_runtime.access_prior_evaluation import evaluate_access_prior_panel
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.storage_policy import _ensure_workspace_storage_bindings, verify_storage
from honf_runtime.unified_training import _atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='latest')
    parser.add_argument('--pair-panel', type=Path, required=True,
                        help='JSON containing a panel name and an explicit list of scene pairs')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-output-mib', type=int, default=64,
                        help='Hard output and peak temporary disk budget in MiB')
    args = parser.parse_args(argv)
    started = time.perf_counter()
    recipe = json.loads(args.recipe.read_text())
    if recipe.get('training_profile') != 'observation_response':
        raise ValueError('This command requires a saved observation-response profile.')
    panel = json.loads(args.pair_panel.read_text())
    raw_pairs = panel.get('pairs', ())
    if not isinstance(raw_pairs, (list, tuple)) or any(not isinstance(pair, (list, tuple)) for pair in raw_pairs):
        raise ValueError('Declare a named panel with distinct pairs of two different case IDs.')
    pairs = tuple(tuple(map(str, pair)) for pair in raw_pairs)
    if (not panel.get('name') or not pairs or len(set(pairs)) != len(pairs)
            or any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs)):
        raise ValueError('Declare a named panel with distinct pairs of two different case IDs.')
    if args.max_output_mib < 1:
        raise ValueError('The output budget must be a positive number of MiB.')
    if args.output.exists():
        raise FileExistsError('Retain the existing evaluation instead of repeating it.')
    _ensure_workspace_storage_bindings()
    budget = args.max_output_mib * 1024**2
    storage = verify_storage(recipe, output=args.output, estimated_bytes=budget, peak_temporary_bytes=budget)
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
    result = evaluate_access_prior_panel(model, provider, pairs)
    result.update(panel=panel, checkpoint=str(checkpoint.resolve()), epoch=int(saved['epoch']),
                  storage=storage, command_process_seconds=time.perf_counter() - started)
    size = len((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))
    if size > budget:
        raise ValueError(f'The evaluation record requires {size} bytes, exceeding the declared {budget}-byte budget.')
    _atomic_json(args.output, result)
    print(json.dumps({'output': str(args.output.resolve()), 'epoch': result['epoch'],
                      'scene_preparations': result['scene_preparations'], 'summary': result['summary'],
                      'command_process_seconds': result['command_process_seconds']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
