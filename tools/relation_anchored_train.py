#!/usr/bin/env python3
"""Run case-owned relation-anchored tasks with the shared training engine."""

from __future__ import annotations

import argparse
import json
import random
import sys
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
        from windfarm.training.relation_anchored_task import build_wind_relation_anchored_task

        return build_wind_relation_anchored_task(recipe, device=device)
    if thermal:
        arm = recipe["member_routing"]["arm"]
        if recipe.get("local_expert", {}).get("enabled", False):
            from channelthermal.training.local_expert_task import build_thermal_local_expert_task

            return build_thermal_local_expert_task(recipe, arm=arm, device=device)
        if recipe.get("local_ports", {}).get("enabled", False):
            from channelthermal.training.local_ports_task import build_thermal_local_ports_task

            return build_thermal_local_ports_task(recipe, arm=arm, device=device)
        from channelthermal.training.r13_8_anchored_task import build_thermal_r13_8_anchored_member_routing_task

        return build_thermal_r13_8_anchored_member_routing_task(recipe, arm=arm, device=device)
    raise ValueError(f"Unsupported dataset {recipe['dataset']!r}.")


def engine_for(recipe, *, device):
    thermal = isinstance(recipe.get("dataset"), dict) and recipe["dataset"].get("name") == "thermal"
    policy = recipe["training"] if thermal else recipe.get("execution", {})
    horizon = int(policy["local_horizon"] if thermal else recipe["total_epochs"])
    config = EngineConfig(
        seed=int(policy["seed"] if thermal else recipe.get("sampling_seed", recipe["seed"])),
        microbatch_cases=int(policy["microbatch_cases"] if thermal else recipe["microbatch_size"]),
        effective_cases=int(policy["effective_batch_size"] if thermal else recipe["effective_batch_size"]),
        total_epochs=horizon,
        monitor_every=int(policy.get("monitor_every", 100)),
        checkpoint_epochs=tuple(policy.get("checkpoint_epochs", range(100, horizon + 1, 100))),
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "fit", "evaluate"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", help="Same-run checkpoint selector, typically latest")
    parser.add_argument("--checkpoint", default="latest")
    args = parser.parse_args(argv)
    recipe = json.loads(args.recipe.read_text())
    model, provider = build(recipe, device=args.device)
    engine = engine_for(recipe, device=args.device)
    arm = (recipe["member_routing"]["arm"] if isinstance(recipe.get("dataset"), dict)
           else recipe["architecture"])
    identity = {"workflow": "relation_anchored_case_training_v1", "recipe": recipe}
    if args.command == "inspect":
        cases, dataset_id = engine.epoch_case_order(provider, 1)
        result = {"provider": dict(provider.identity_payload()), "sampling_dataset_id": dataset_id,
                  "train_cases": len(cases), "parameters": sum(p.numel() for p in model.parameters()),
                  "optimizer_groups": [spec.name for spec in provider.optimizer_groups(model, arm, "joint")]}
    elif args.command == "fit":
        if args.output is None or args.stop_after is None:
            parser.error("fit requires --output and --stop-after")
        if recipe["dataset"] == "wind":
            print(json.dumps({"event": "wind_preparation_complete",
                              "preparation": provider.preparation_summary()},
                             sort_keys=True, default=str, allow_nan=False), flush=True)
        result = engine.fit(model, provider, args.output, identity=identity, arm=arm,
                            stop_after=args.stop_after, resume_checkpoint=args.resume)
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
    print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
