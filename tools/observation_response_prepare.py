#!/usr/bin/env python3
"""Calibrate shared observation losses at an unchanged native epoch-ten state."""
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
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
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
