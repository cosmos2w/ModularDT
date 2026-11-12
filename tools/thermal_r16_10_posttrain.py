#!/usr/bin/env python3
"""Launch or exactly resume an independently named R16-10 B0-to-R child."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / 'src', PROJECT / 'Case_ThermalChannel/src'):
    sys.path.insert(0, str(directory))

import torch
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.unified_training import (
    EngineConfig, SamplingKey, SelectionPolicy, named_optimizer_state, restore_rng_state,
)
from channelthermal.training.r16_10_posttrain import (
    POSTTRAIN_EPOCHS, PosttrainEngine, build_posttrain_task,
)


def assert_same_tree(left, right):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or not torch.equal(left.cpu(), right.cpu()):
            raise ValueError('Inherited optimizer tensor changed before post-training.')
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError('Inherited optimizer keys differ.')
        for key in left:
            assert_same_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise ValueError('Inherited optimizer groups differ.')
        for a, b in zip(left, right):
            assert_same_tree(a, b)
    elif left != right:
        raise ValueError('Inherited optimizer scalar changed.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--stop-after', type=int, default=POSTTRAIN_EPOCHS)
    p.add_argument('--resume', help='Use latest for an exact resume of this child only')
    p.add_argument('--cuda-memory-fraction', type=float, default=.5)
    args = p.parse_args(argv)
    if not 1 <= args.stop_after <= POSTTRAIN_EPOCHS:
        p.error('This child has exactly 1000 authorized post-training epochs.')
    if not 0 < args.cuda_memory_fraction <= 1:
        p.error('The CUDA allocator fraction must be in (0, 1].')
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recipe = json.loads(args.recipe.read_text())
    expected_output = Path(recipe['storage']['output_root']) / recipe['name']
    if args.output.resolve() != expected_output.resolve():
        p.error('Output must equal the separately named recipe lineage.')
    source_path = Path(recipe['initialization']['source_checkpoint']).resolve()
    if args.output.resolve() == source_path.parents[1]:
        p.error('A post-training child cannot overwrite the parent output.')
    saved = (load_trusted_checkpoint(resolve_checkpoint(args.output, args.resume), map_location='cpu')
             if args.resume else None)
    source = None if saved is not None else load_trusted_checkpoint(source_path, map_location='cpu')
    model, provider = build_posttrain_task(recipe, device=args.device, source=source, resume_payload=saved)
    policy = recipe['training']
    provider.bind_wall_clock_budget(
        output_dir=args.output, budget_seconds=policy['wall_clock_budget_seconds'],
        saver_reserve_seconds=policy['clean_save_reserve_seconds'])
    engine = PosttrainEngine(EngineConfig(
        seed=policy['seed'], microbatch_cases=policy['microbatch_cases'],
        effective_cases=policy['effective_batch_size'], total_epochs=POSTTRAIN_EPOCHS,
        monitor_every=policy['monitor_every'], checkpoint_epochs=tuple(policy['checkpoint_epochs']),
        latest_every=policy['latest_every'], curve_every=policy['curve_every'],
        progress_enabled=policy['progress_enabled'], gradient_clip=recipe['optimizer']['gradient_clip'],
        sampling_version=SamplingKey.CASE_EPOCH_VERSION, training_mode='joint',
        write_initial_artifacts=True), device=args.device, selection=SelectionPolicy(field_metric='field_score'))
    identity = {'workflow': 'thermal_r16_10_b0_to_r_posttraining_v1', 'recipe': recipe,
                'parent_checkpoint': str(source_path), 'parent_epoch': 3000}
    optimizer_seed = None
    if source is not None:
        optimizer_seed = source['optimizer_state_by_name']
        # Verify actual named moments before any training; the ordinary engine
        # will create the optimizer again from this same supported seed input.
        optimizer, _ = engine._make_optimizer(model, provider, model.architecture, 1, optimizer_seed)
        assert_same_tree(named_optimizer_state(model, optimizer), optimizer_seed)
        del optimizer
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'configs').mkdir(exist_ok=True)
        receipt = {'source_checkpoint': str(source_path), 'source_epoch': 3000,
                   'posttraining_epochs': POSTTRAIN_EPOCHS, 'exact_model_state_loaded': True,
                   'exact_named_adamw_state_loaded': True, 'native_sampler_first_epoch': 3001,
                   'family_sampler_first_epoch': 11, 'auxiliary_ramp_first_epoch': .025,
                   'learning_rate': 3e-5, 'normalization': 'inherited TRAIN600-only',
                   'new_auxiliary_sampler_seed': 0, 'source_history_retained_at_parent': True}
        (args.output / 'configs/posttraining_initialization.json').write_text(json.dumps(receipt, indent=2)+'\n')
        restore_rng_state(source['rng_state'])
    result = engine.fit(model, provider, args.output, identity=identity, arm=model.architecture,
                        stop_after=args.stop_after, optimizer_seed=optimizer_seed,
                        resume_checkpoint=args.resume)
    print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
