#!/usr/bin/env python3
"""Create shared TRAIN input definitions and calibrate one native prefix."""
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
from honf_runtime.unified_training import _atomic_json, capture_rng_state, restore_rng_state


def _state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right))
    return left == right


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('definitions', 'recipe', 'calibrate'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, help='Local-response child recipe, required for calibration')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--input-definitions', type=Path)
    parser.add_argument('--variant', choices=('state_anchors', 'uniform_response', 'multisecant_response',
                                              'soft_inverse_response'))
    parser.add_argument('--name')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    source_path = args.source.expanduser().resolve(strict=True)
    source = load_trusted_checkpoint(source_path, map_location='cpu')
    parent = source['experiment_identity']['recipe']
    if (source_path.name != 'epoch_0010_model.pt' or source.get('epoch') != 10
            or parent.get('training_profile') != 'field_response'
            or parent.get('thermal_field_response', {}).get('variant') != 'field_only'
            or source.get('provider_training_state', {}).get('field_response', {}).get('calibration') is not None):
        raise ValueError('A literal native-only epoch-ten checkpoint is required.')
    if args.output.exists():
        raise ValueError('Output already exists; retain and reuse the original shared definition or calibration.')
    if args.operation == 'definitions':
        from channelthermal.training.thermal_local_response import (build_thermal_input_definitions,
                                                                      save_thermal_input_definitions)
        _, provider = build(parent, device='cpu', resume_payload=source)
        definitions = build_thermal_input_definitions(provider.data_path, provider.training_case_ids)
        save_thermal_input_definitions(args.output, definitions)
    elif args.operation == 'recipe':
        if not all((args.input_definitions, args.variant, args.name, args.output_root)):
            parser.error('recipe requires --input-definitions, --variant, --name and --output-root')
        child = copy.deepcopy(parent)
        child['name'] = args.name
        child['storage']['output_root'] = str(args.output_root.expanduser().resolve())
        child['training_profile'] = 'local_multisecant_response'
        child['thermal_local_response'] = {'input_definitions': str(args.input_definitions.expanduser().resolve(strict=True)),
                                           'variant': args.variant, 'seed': args.seed}
        _atomic_json(args.output, child)
    else:
        if args.recipe is None:
            parser.error('calibrate requires --recipe')
        recipe = json.loads(args.recipe.read_text())
        if (recipe.get('training_profile') != 'local_multisecant_response'
                or _native_recipe(recipe) != _native_recipe(parent)):
            raise ValueError('The local-response recipe must preserve the source native definition.')
        model, provider = build(recipe, device=args.device, resume_payload=source)
        model.load_state_dict(source['model_state_dict'], strict=True)
        model.to(args.device)
        provider.load_training_state_dict(provider.initialize_local_response_state(
            source.get('provider_training_state', {}), None))
        restore_rng_state(source['rng_state'])
        rng_before = capture_rng_state()
        state_before = copy.deepcopy(provider.training_state_dict())
        record = provider.calibrate_local_response(model, source_checkpoint=source_path)
        if not _state_equal(rng_before, capture_rng_state()) or not _state_equal(
                state_before, provider.training_state_dict()):
            raise RuntimeError('Calibration changed the restored source RNG or provider state.')
        provider.validate_local_calibration(record, source_checkpoint=source_path)
        _atomic_json(args.output, record)
    print(json.dumps({'operation': args.operation, 'output': str(args.output.resolve())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
