#!/usr/bin/env python3
"""Prepare shared epoch-zero coupled training or historical epoch-ten calibration."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .relation_anchored_train import build
except ImportError:
    from relation_anchored_train import build
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.native_prefix_branch import _native_recipe
from honf_runtime.storage_policy import _ensure_workspace_storage_bindings, verify_storage
from honf_runtime.unified_training import _atomic_json, capture_rng_state, restore_rng_state


def state_equal(left, right):
    """Compare actual recovery values, including arrays and ordered sequences."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(state_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(state_equal(a, b) for a, b in zip(left, right)))
    return left == right


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fresh-epoch-zero', action='store_true')
    args = parser.parse_args(argv)
    if args.fresh_epoch_zero:
        if args.source is not None:
            raise ValueError('A fresh epoch-zero state cannot consume a source checkpoint.')
        recipe = json.loads(args.recipe.read_text())
        if (recipe.get('training_profile') != 'observation_response'
                or (recipe.get('observation_response', {}).get('mode') != 'coupled_native'
                    and 'finite_observations' not in recipe)):
            raise ValueError('Fresh preparation requires a coupled native or finite-observation recipe.')
        if args.output.exists():
            raise FileExistsError('Preserve and reuse the existing shared epoch-zero state.')
        if 'storage' in recipe:
            _ensure_workspace_storage_bindings()
            storage = verify_storage(recipe, output=args.output, estimated_bytes=1024**3,
                                     peak_temporary_bytes=1024**3)
        else:
            storage = {}
        from relation_anchored_train import engine_for
        from honf_runtime.coupled_initialization import prepare_epoch_zero
        from honf_runtime.unified_training import _atomic_torch_save

        model, provider = build(recipe, device=args.device)
        engine = engine_for(recipe, device=args.device)
        payload = prepare_epoch_zero(model, provider, engine)
        _atomic_torch_save(args.output, payload)
        print(json.dumps({'output': str(args.output.resolve()), 'epoch': 0, 'optimizer_updates': 0,
                          'coefficient': (payload['calibration']['coefficient']
                                          if payload['calibration'] is not None else None),
                          'finite_coefficients': payload.get('finite_calibration', {}).get('coefficients'),
                          'free_bytes_before': storage.get('free_bytes_before')}))
        return 0
    if args.source is None:
        raise ValueError('Historical observation calibration requires --source.')
    source_path = args.source.expanduser().resolve(strict=True)
    source = load_trusted_checkpoint(source_path, map_location='cpu')
    parent = source['experiment_identity']['recipe']
    recipe = json.loads(args.recipe.read_text())
    if (source_path.name != 'epoch_0010_model.pt' or source.get('epoch') != 10
            or parent.get('training_profile') != 'field_response'
            or parent.get('thermal_field_response', {}).get('variant') != 'field_only'
            or source.get('provider_training_state', {}).get('field_response', {}).get('calibration') is not None):
        raise ValueError('Use the literal completed native-only epoch-ten checkpoint.')
    if (recipe.get('training_profile') != 'observation_response'
            or _native_recipe(recipe) != _native_recipe(parent)):
        raise ValueError('The observation recipe changed native data, model, objective, sampling or schedule.')
    if args.output.exists():
        raise FileExistsError('Preserve and reuse the existing common calibration record.')
    _ensure_workspace_storage_bindings()
    storage = verify_storage(recipe, output=args.output, estimated_bytes=1024**2,
                             peak_temporary_bytes=1024**2)
    model, provider = build(recipe, device=args.device, resume_payload=source)
    model.load_state_dict(source['model_state_dict'], strict=True)
    model.to(args.device)
    provider.load_training_state_dict(provider.initialize_observation_response_state(
        source.get('provider_training_state', {}), None))
    restore_rng_state(source['rng_state'])
    rng_before = capture_rng_state()
    provider_before = copy.deepcopy(provider.training_state_dict())
    model_before = copy.deepcopy(model.state_dict())
    record = provider.calibrate_observation_response(model, source_checkpoint=source_path)
    if (not state_equal(rng_before, capture_rng_state())
            or not state_equal(provider_before, provider.training_state_dict())
            or not state_equal(model_before, model.state_dict())):
        raise RuntimeError('Calibration changed native recovery state.')
    provider.validate_observation_calibration(record, source_checkpoint=source_path)
    _atomic_json(args.output, record)
    print(json.dumps({'output': str(args.output.resolve()), 'optimizer_updates': 0,
                      'free_bytes_before': storage['free_bytes_before']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
