#!/usr/bin/env python3
"""Run case-owned relation-anchored tasks with the shared training engine."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src", PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.unified_training import EngineConfig, SamplingKey, SelectionPolicy, TrainingEngine


def build(recipe, *, device):
    """Construct the dataset's own model, transforms, and native task."""
    thermal = isinstance(recipe.get("dataset"), dict) and recipe["dataset"].get("name") == "thermal"
    seed = int(recipe["initialization"]["seed"] if thermal else recipe["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if recipe["dataset"] == "wind":
        if recipe.get("schema") == "wind_relation_anchored_r15_3_v1":
            from windfarm.training.relation_anchored_r15_3_task import build_wind_r15_3_task

            return build_wind_r15_3_task(recipe, device=device)
        from windfarm.training.relation_anchored_task import build_wind_relation_anchored_task

        return build_wind_relation_anchored_task(recipe, device=device)
    if thermal:
        arm = recipe["member_routing"]["arm"]
        if recipe.get("thermal_finite_response", {}).get("enabled", False):
            from channelthermal.training.thermal_finite_response_task import build_thermal_finite_response_task

            return build_thermal_finite_response_task(recipe, arm=arm, device=device)
        if recipe.get("thermal_port_response", {}).get("enabled", False):
            from channelthermal.training.thermal_port_response_task import build_thermal_port_response_task

            return build_thermal_port_response_task(recipe, arm=arm, device=device)
        if recipe.get("thermal_feedback", {}).get("enabled", False):
            from channelthermal.training.thermal_response_feedback_task import build_thermal_response_feedback_task

            return build_thermal_response_feedback_task(recipe, arm=arm, device=device)
        if recipe.get("local_expert", {}).get("enabled", False):
            from channelthermal.training.local_expert_task import build_thermal_local_expert_task

            return build_thermal_local_expert_task(recipe, arm=arm, device=device)
        if recipe.get("local_ports", {}).get("enabled", False):
            from channelthermal.training.local_ports_task import build_thermal_local_ports_task

            return build_thermal_local_ports_task(recipe, arm=arm, device=device)
        from channelthermal.training.r13_8_anchored_task import build_thermal_r13_8_anchored_member_routing_task

        return build_thermal_r13_8_anchored_member_routing_task(recipe, arm=arm, device=device)
    raise ValueError(f"Unsupported dataset {recipe['dataset']!r}.")


def engine_for(recipe, *, device, execution_horizon=None):
    thermal = isinstance(recipe.get("dataset"), dict) and recipe["dataset"].get("name") == "thermal"
    policy = recipe["training"] if thermal else recipe.get("execution", {})
    horizon = int(policy["local_horizon"] if thermal else recipe["total_epochs"])
    checkpoints = policy.get("checkpoint_epochs", range(100, horizon + 1, 100))
    if execution_horizon is not None:
        wind_extension = (recipe.get("dataset") == "wind"
                          and recipe.get("schema") == "wind_relation_anchored_r15_2_v2")
        thermal_extension = (thermal and recipe.get("thermal_port_response", {}).get("schema") == "thermal_port_response_v1"
                             and recipe["thermal_port_response"].get("arm") in ("P64", "R64")
                             and policy.get("membership") == "full_train600"
                             and horizon == 100 and execution_horizon == 500)
        if (not (wind_extension or thermal_extension)
                or type(execution_horizon) is not int
                or execution_horizon <= horizon or execution_horizon % 100):
            raise ValueError("Unsupported explicit same-lineage horizon extension.")
        horizon = execution_horizon
        checkpoints = range(100, horizon + 1, 100)
    config = EngineConfig(
        seed=int(policy["seed"] if thermal else recipe.get("sampling_seed", recipe["seed"])),
        microbatch_cases=int(policy["microbatch_cases"] if thermal else recipe["microbatch_size"]),
        effective_cases=int(policy["effective_batch_size"] if thermal else recipe["effective_batch_size"]),
        total_epochs=horizon,
        monitor_every=int(policy.get("monitor_every", 100)),
        checkpoint_epochs=tuple(checkpoints),
        latest_every=int(policy.get("latest_every", 1)),
        curve_every=int(policy.get("curve_every", 10)),
        progress_enabled=bool(policy.get("progress_enabled", True)),
        gradient_clip=float(recipe.get("gradient_clip", 1.0)),
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        training_mode="joint",
        write_initial_artifacts=bool(policy.get("write_initial_artifacts", True)),
    )
    return TrainingEngine(config, device=device, selection=SelectionPolicy(
        field_metric=policy.get("field_metric", "field_score")))


def calibrate_finite_response(model, provider, engine, output):
    """Measure twelve actual accumulated macros without an optimizer update."""
    spec = provider.finite_spec
    if spec['arm'] != 'B' or float(spec['lambda_delta']) != 1.:
        raise ValueError('Initial balance measurement requires B with unweighted delta coefficient one.')
    if len(provider.training_case_ids) != 600:
        raise ValueError('Balance calibration uses the formal TRAIN600 macro objective.')
    output.mkdir(parents=True, exist_ok=True)
    sampler = provider.pair_sampler
    families = [row for row in sampler.cases if row['qualification']]
    if len(families) < 12:
        raise ValueError('Twelve fixed fitting families are required for this calibration.')
    cases, dataset_id = engine.epoch_case_order(provider, 1)
    slices = engine.epoch_update_slices(len(cases))
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    records, ratios = [], []
    def inner(left, right):
        return sum(float((left[name].detach().double() * right[name].detach().double()).sum())
                   for name in left.keys() & right.keys())
    try:
        for index, family in enumerate(families[:12]):
            action = sampler.ACTIONS[index % len(sampler.ACTIONS)]
            kind = action.split('_')[0]
            def chosen(_key, family=family, action=action, kind=kind, index=index):
                return sampler.pair(family, action, sampler.scales[kind], seed=15808+index)
            provider.pair_sampler = chosen
            batches = engine.prepare_update_microbatches(provider, arm=model.architecture,
                phase='joint', epoch=1, update_index=index, update_cases=cases[slices[index]],
                sampling_dataset_id=dataset_id)
            terms = provider.loss_denominators(batches, 'joint', model.architecture)
            groups = {'primary': [name for name in terms if not name.startswith('finite/')],
                      'endpoint': ['finite/endpoint'], 'delta': ['finite/delta']}
            result = engine.diagnose_macro_update(model, provider, batches, phase='joint',
                arm=model.architecture, epoch=1, term_groups=groups, include_gradient_tensors=True)
            gradients = result.pop('gradient_tensors')
            result.pop('combined_gradient_tensors')
            absolute = {name: gradients['primary'].get(name, 0) + gradients['endpoint'].get(name, 0)
                        for name in gradients['primary'].keys() | gradients['endpoint'].keys()}
            delta = gradients['delta']
            na, nd = math.sqrt(inner(absolute, absolute)), math.sqrt(inner(delta, delta))
            if not all(math.isfinite(v) for v in (na, nd)):
                raise FloatingPointError('Nonfinite accumulated balance gradient.')
            result['balance_fp64'] = {'absolute_norm': na, 'delta_norm': nd,
                'cosine': inner(absolute, delta)/(na*nd) if na and nd else None,
                'absolute_zero': na == 0, 'delta_zero': nd == 0}
            result['pair'] = {'family': family['case_id'], 'action': action, 'role': 'fit'}
            ratios.append(na/max(nd, 1e-12))
            (output/f'example_{index+1:02d}.json').write_text(json.dumps(result, indent=2)+'\n')
            records.append({'example': index+1, **result['pair'], **result['balance_fp64']})
            print(json.dumps({'calibration_example': index+1, **result['balance_fp64']}), flush=True)
            del batches, result, gradients, absolute, delta
    finally:
        provider.pair_sampler = sampler
    if any(not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items()):
        raise RuntimeError('Read-only balance calibration changed model state.')
    coefficient = min(1., .25*float(np.median(ratios)))
    result = {'schema': 'thermal_finite_response_initial_balance_v1', 'lambda_delta': coefficient,
        'alpha_endpoint': .25, 'examples': records, 'norm_precision': 'FP64 arithmetic',
        'parameters': 'all inherited trainable parameters; common P64 source; no new C parameters',
        'optimizer_updates': 0, 'model_state_unchanged': True,
        'reduction': 'actual accumulated primary macro plus one alpha-weighted endpoint pair; pack denominator one'}
    (output/'balance.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def native_finite_review(model, provider, saved, output):
    """Use the maintained complete native reducer on the saved exposed panel."""
    from honf_runtime.member_routing_evaluation import select_validation_representatives
    from channelthermal.training.thermal_finite_evaluation import evaluate_native_panel
    ids = tuple(provider.validation_case_ids)
    if len(ids) != 89 or '0273' in ids:
        raise ValueError('Formal native review requires the existing canonical exposed VAL89.')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    representatives = set(select_validation_representatives(provider))
    result = evaluate_native_panel(model, provider, saved, ids, export_ids=representatives)
    for case, arrays in result.pop('representative_arrays').items():
        np.savez_compressed(output/f'case_{case}.npz', **arrays)
    result['checkpoint_epoch'] = saved['epoch']
    result['clock'] = 5100 + int(saved['epoch'])
    result['partition'] = 'canonical exposed VAL89; source test-labelled; no new training labels'
    (output/'native.json').write_text(json.dumps(result, indent=2, default=str, allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "fit", "evaluate", "calibrate", "native"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", help="Same-run checkpoint selector, typically latest")
    parser.add_argument("--extend-horizon", action="store_true",
                        help="Explicit supported same-run execution extension; preserve the saved recipe and optimizer schedule")
    parser.add_argument("--checkpoint", default="latest")
    args = parser.parse_args(argv)
    if args.extend_horizon and (args.command != "fit" or not args.resume or args.stop_after is None):
        parser.error("--extend-horizon requires fit, --resume and --stop-after")
    recipe = json.loads(args.recipe.read_text())
    finite = bool(recipe.get('thermal_finite_response', {}).get('enabled', False))
    if args.command in ('calibrate', 'native') and (not finite or args.output is None):
        parser.error('calibrate/native require a finite-response recipe and --output')
    started, cpu_started = time.perf_counter(), time.process_time()
    if finite:
        torch.set_num_threads(1)
    model, provider = build(recipe, device=args.device)
    engine = (engine_for(recipe, device=args.device, execution_horizon=args.stop_after)
              if args.extend_horizon else engine_for(recipe, device=args.device))
    arm = (recipe["member_routing"]["arm"] if isinstance(recipe.get("dataset"), dict)
           else recipe["architecture"])
    identity = {"workflow": "relation_anchored_case_training_v1", "recipe": recipe}
    if args.command == "inspect":
        cases, dataset_id = engine.epoch_case_order(provider, 1)
        result = {"provider": dict(provider.identity_payload()), "sampling_dataset_id": dataset_id,
                  "train_cases": len(cases), "parameters": sum(p.numel() for p in model.parameters()),
                  "optimizer_groups": [spec.name for spec in provider.optimizer_groups(model, arm, "joint")]}
    elif args.command == 'calibrate':
        result = calibrate_finite_response(model, provider, engine, args.output)
    elif args.command == 'native':
        if args.checkpoint == 'source':
            saved = {'epoch': 0, 'source_checkpoint': recipe['thermal_finite_response']['source_checkpoint']}
        else:
            from channelthermal.training.thermal_finite_response_task import load_finite_response_checkpoint
            model, saved = load_finite_response_checkpoint(Path(args.checkpoint), device=args.device)
            if saved['experiment_identity']['recipe'] != recipe:
                raise ValueError('Native checkpoint and recipe differ.')
            from honf_runtime.contextual_frontier_evaluation import _bind_loaded_model
            _bind_loaded_model(provider, model, torch.device(args.device))
        result = native_finite_review(model, provider, saved, args.output)
    elif args.command == "fit":
        if args.output is None or args.stop_after is None:
            parser.error("fit requires --output and --stop-after")
        if recipe["dataset"] == "wind":
            print(json.dumps({"event": "wind_preparation_complete",
                              "preparation": provider.preparation_summary()},
                             sort_keys=True, default=str, allow_nan=False), flush=True)
        result = engine.fit(model, provider, args.output, identity=identity, arm=arm,
                            stop_after=args.stop_after, resume_checkpoint=args.resume,
                            allow_horizon_extension=args.extend_horizon)
    else:
        if args.output is None:
            parser.error("evaluate requires --output")
        saved = load_trusted_checkpoint(resolve_checkpoint(args.output, args.checkpoint), map_location="cpu")
        saved_identity = saved["experiment_identity"]
        if saved_identity["recipe"] != recipe or saved_identity["provider_identity"] != dict(provider.identity_payload()):
            raise ValueError("The recipe or native provider differs from the saved run.")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        model.to(args.device)
        restore = getattr(provider, "load_training_state_dict", None)
        if callable(restore):
            restore(saved.get("provider_training_state", {}))
        result = engine._evaluate(model, provider, arm, "joint", int(saved["epoch"]))
    if finite and args.output is not None:
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        receipt = {'command': args.command, 'requested_stop_epoch': args.stop_after, 'resume': args.resume,
            'device': args.device, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'wall_seconds': time.perf_counter()-started, 'cpu_seconds': time.process_time()-cpu_started,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0}
        directory = args.output/'metrics' if args.command == 'fit' else args.output
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/'execution_sessions.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt)+'\n')
    print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
