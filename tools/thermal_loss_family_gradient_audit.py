#!/usr/bin/env python3
"""Measure occasional fixed-TRAIN-panel gradients from an existing checkpoint."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in ('src', 'Case_ThermalChannel/src', 'Case_WindFarm/src', 'tools'):
    sys.path.insert(0, str(ROOT / directory))

import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.coupled_initialization import _same
from honf_runtime.loss_family_gradient_audit import audit_loss_families
from relation_anchored_train import build, provider_identity_matches_for_inference


def restore_checkpoint(path, *, device):
    """Restore the saved model/provider and verify the common calibration."""
    checkpoint = Path(path).expanduser().resolve(strict=True)
    saved = load_trusted_checkpoint(checkpoint, map_location='cpu')
    identity = saved['experiment_identity']
    recipe = identity['recipe']
    observation = recipe.get('observation_response', {})
    if recipe.get('training_profile') != 'observation_response' or observation.get('mode') != 'coupled_native':
        raise ValueError('The saved checkpoint must use the coupled-native observation-response profile.')
    if observation.get('reader_policy') != 'sampled_equal_probability':
        raise ValueError('The saved checkpoint must use the sampled full/restricted reader policy.')
    epoch = saved.get('epoch')
    if type(epoch) is not int or epoch < 1:
        raise ValueError('The saved checkpoint must represent a completed positive epoch.')
    initialization = Path(observation['epoch_zero_checkpoint']).expanduser()
    if not initialization.is_absolute():
        initialization = ROOT / initialization
    initialization = initialization.resolve(strict=True)
    common = load_trusted_checkpoint(initialization, map_location='cpu')
    calibration = saved['provider_training_state']['coupled_native']['calibration']
    if (not isinstance(common, dict) or common.get('schema') != 'coupled_native_epoch_zero_v1'
            or common.get('epoch') != 0 or not _same(common.get('calibration'), calibration)
            or not _same(common.get('provider_training_state', {}).get('coupled_native', {}).get('calibration'),
                         calibration)):
        raise ValueError('Checkpoint calibration does not match its saved common epoch-zero state.')
    model, provider = build(recipe, device=device, resume_payload=saved)
    if not provider_identity_matches_for_inference(recipe, identity['provider_identity'], dict(provider.identity_payload())):
        raise ValueError('The reconstructed provider differs from the checkpoint identity.')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.to(device)
    provider.load_training_state_dict(saved['provider_training_state'])
    return model, provider, saved, checkpoint, initialization


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Existing saved model checkpoint')
    parser.add_argument('--panels', required=True, help='Saved input-selected TRAIN15 panel JSON')
    parser.add_argument('--output', required=True, help='New JSON output path')
    parser.add_argument('--device', default='cpu', help='Runtime device, for example cpu or cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f'The requested output already exists: {output}')
    if args.cpu_threads < 1:
        raise ValueError('--cpu-threads must be positive.')
    torch.set_num_threads(args.cpu_threads)
    panel_path = Path(args.panels).expanduser().resolve(strict=True)
    panel = json.loads(panel_path.read_text())['development']['train_case_ids']
    from channelthermal.training.observation_response import _TRAIN15
    if not isinstance(panel, list) or tuple(map(str, panel)) != _TRAIN15:
        raise ValueError('The panel must contain the established input-selected TRAIN15 case order.')
    restore_started = time.perf_counter()
    model, provider, saved, checkpoint, initialization = restore_checkpoint(args.checkpoint, device=args.device)
    restore_seconds = time.perf_counter() - restore_started
    recipe = saved['experiment_identity']['recipe']
    if not isinstance(recipe.get('dataset'), dict) or recipe['dataset'].get('name') != 'thermal':
        raise ValueError('This entry point requires the Thermal case adapter.')
    result = audit_loss_families(model, provider, panel_case_ids=panel, epoch=saved['epoch'],
                                 read_modes=('restricted', 'full'), sampling_epoch=1)
    device = next(model.parameters()).device
    result['provenance'] = {'checkpoint': str(checkpoint), 'recipe_name': recipe.get('name'),
                            'panels': str(panel_path), 'common_epoch_zero': str(initialization),
                            'response_update_scope': recipe['observation_response'].get('response_update_scope'),
                            'device': str(device), 'torch_version': str(torch.__version__),
                            'python_version': platform.python_version(), 'cpu_threads': torch.get_num_threads(),
                            'checkpoint_restore_seconds': restore_seconds}
    if device.type == 'cuda':
        result['provenance']['device_name'] = torch.cuda.get_device_name(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
