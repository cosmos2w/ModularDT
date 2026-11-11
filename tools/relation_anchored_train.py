#!/usr/bin/env python3
"""Train, resume and evaluate native relation-anchored case adapters."""
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


def configure_r16_5_execution(recipe, *, device):
    """Select the declared replay-safe policy before any CUDA initialization."""
    if recipe.get("r16_5_warm_fork") is None:
        return
    from channelthermal.training.r16_5_warm_fork import configure_r16_5_execution as configure

    configure(recipe, device=device)


def build(recipe, *, device, response_scales=None, saved_adapter_state=None, saved_normalization_source=None,
          resume_payload=None):
    """Dispatch supported adapters while preserving their saved recipe schemas."""
    configure_r16_5_execution(recipe, device=device)
    thermal = isinstance(recipe.get("dataset"), dict) and recipe["dataset"].get("name") == "thermal"
    unsupported = ("thermal_finite_response", "thermal_feedback", "local_expert", "local_ports")
    if any(recipe.get(name, {}).get("enabled", False) for name in unsupported):
        raise ValueError("This entry supports native fields and one-pass ports; use the research branch for this recipe.")
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
        if recipe.get("r16_10_family") is not None:
            from channelthermal.training.r16_10_family_task import build_thermal_r16_10_task

            model, provider = build_thermal_r16_10_task(
                recipe, arm=arm, device=device, resume_payload=resume_payload)
            from channelthermal.training.r16_10_health import attach_update_diagnostics

            attach_update_diagnostics(provider, recipe)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return model, provider
        if recipe.get("r16_8_content") is not None:
            from channelthermal.training.r16_8_content_task import build_thermal_r16_8_task
            model, provider = build_thermal_r16_8_task(recipe, arm=arm, device=device,
                resume_payload=resume_payload, response_scales=response_scales)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return model, provider
        if recipe.get("r16_7_prescribed") is not None:
            from channelthermal.training.r16_7_prescribed_task import build_thermal_r16_7_task
            model, provider = build_thermal_r16_7_task(recipe, arm=arm, device=device,
                resume_payload=resume_payload, response_scales=response_scales)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return model, provider
        if recipe.get("r16_6_balanced") is not None:
            from channelthermal.training.r16_6_balanced_task import build_thermal_r16_6_task
            model, provider = build_thermal_r16_6_task(recipe, arm=arm, device=device,
                resume_payload=resume_payload, response_scales=response_scales)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return model, provider
        if recipe.get("r16_5_warm_fork") is not None:
            from channelthermal.training.r16_5_warm_fork import build_thermal_r16_5_warm_fork_task

            model, provider = build_thermal_r16_5_warm_fork_task(
                recipe, arm=arm, device=device, resume_payload=resume_payload, response_scales=response_scales)
            # A fresh fork owns its own RNG; a resume restores its saved RNG in fit.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return model, provider
        if recipe.get("r16_4_warm_fork") is not None:
            from channelthermal.training.r16_4_warm_fork import build_thermal_r16_4_warm_fork_task

            model, provider = build_thermal_r16_4_warm_fork_task(recipe, arm=arm, device=device)
            if response_scales is not None and provider.response_scales != response_scales:
                raise ValueError("Saved run response scales differ from the literal source FIT scales.")
            # Each independent arm starts the same non-sampler streams. Exact
            # resumes subsequently restore that lineage's own saved RNG state.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return model, provider
        if recipe.get("thermal_state_response", {}).get("enabled", False):
            from channelthermal.training.thermal_state_response_task import build_thermal_state_response_task

            options = {"arm": arm, "device": device}
            if response_scales is not None:
                options["response_scales"] = response_scales
            if saved_adapter_state is not None:
                options["saved_adapter_state"] = saved_adapter_state
                options["saved_normalization_source"] = saved_normalization_source
            return build_thermal_state_response_task(recipe, **options)
        if recipe.get("thermal_port_response", {}).get("enabled", False):
            from channelthermal.training.thermal_port_response_task import build_thermal_port_response_task

            return build_thermal_port_response_task(recipe, arm=arm, device=device)
        from channelthermal.training.r13_8_anchored_task import build_thermal_r13_8_anchored_member_routing_task

        return build_thermal_r13_8_anchored_member_routing_task(recipe, arm=arm, device=device)
    raise ValueError(f"Unsupported dataset {recipe['dataset']!r}.")


def engine_for(recipe, *, device, execution_horizon=None):
    """Keep the scientific horizon separate from a requested execution stop."""
    thermal = isinstance(recipe.get("dataset"), dict) and recipe["dataset"].get("name") == "thermal"
    policy = recipe["training"] if thermal else recipe.get("execution", {})
    horizon = int(policy["local_horizon"] if thermal else recipe["total_epochs"])
    checkpoints = policy.get("checkpoint_epochs", range(100, horizon + 1, 100))
    if execution_horizon is not None:
        wind_extension = (recipe.get("dataset") == "wind" and recipe.get("schema") in
                          ("wind_relation_anchored_r15_2_v2", "wind_relation_anchored_v1"))
        port = recipe.get("thermal_port_response", {})
        r16_5_arm = recipe.get("r16_5_decision", {}).get("arm")
        thermal_extension = (thermal and port.get("schema") == "thermal_port_response_v1"
                             and port.get("arm") == "P64" and policy.get("membership") == "full_train600"
                             and horizon == 100 and execution_horizon == 500
                             and (recipe.get("r16_6_balanced", {}).get("arm") == "M"
                                  or (recipe.get("r16_6_balanced") is None and
                                      (recipe.get("r16_5_warm_fork") is None or r16_5_arm == "PDeltaD"))))
        family = recipe.get("r16_10_family", {})
        r16_10_extension = (thermal and family.get("schema") == "thermal_r16_10_family_v1"
                            and family.get("population") == 600 and family.get("arm") in ("B0", "R")
                            and policy.get("membership") == "full_train600" and horizon == 2000
                            and execution_horizon == 3000
                            and recipe.get("member_routing", {}).get("model", {}).get("access_measure") == "inventory_unit_total")
        if (not (wind_extension or thermal_extension or r16_10_extension) or type(execution_horizon) is not int
                or execution_horizon <= horizon or execution_horizon % 100):
            raise ValueError("Unsupported explicit same-lineage horizon extension.")
        horizon = execution_horizon
        checkpoints = range(100, horizon + 1, 100)
    config = EngineConfig(
        seed=int(policy["seed"] if thermal else recipe.get("sampling_seed", recipe["seed"])),
        microbatch_cases=int(policy["microbatch_cases"] if thermal else recipe["microbatch_size"]),
        effective_cases=int(policy["effective_batch_size"] if thermal else recipe["effective_batch_size"]),
        total_epochs=horizon, monitor_every=int(policy.get("monitor_every", 100)),
        checkpoint_epochs=tuple(checkpoints), latest_every=int(policy.get("latest_every", 1)),
        curve_every=int(policy.get("curve_every", 10)), progress_enabled=bool(policy.get("progress_enabled", True)),
        gradient_clip=float(recipe.get("gradient_clip", 1.0)), sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        training_mode="joint", write_initial_artifacts=bool(policy.get("write_initial_artifacts", True)),
    )
    engine_type = TrainingEngine
    if thermal and recipe.get("r16_5_warm_fork") is not None:
        from channelthermal.training.r16_5_warm_fork import ThermalR165WarmForkEngine

        engine_type = ThermalR165WarmForkEngine
    elif thermal and recipe.get("r16_4_warm_fork") is not None:
        from channelthermal.training.r16_4_warm_fork import ThermalR164WarmForkEngine

        engine_type = ThermalR164WarmForkEngine
    return engine_type(config, device=device, selection=SelectionPolicy(
        field_metric=policy.get("field_metric", "field_score")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "fit", "evaluate"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", help="Same-run checkpoint selector, typically latest")
    parser.add_argument("--extend-horizon", action="store_true",
                        help="Explicit same-run extension; retain the saved recipe and optimizer schedule")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--cuda-memory-fraction", type=float,
                        help="Optional process-local CUDA allocator limit; does not change training populations")
    args = parser.parse_args(argv)
    if args.extend_horizon and (args.command != "fit" or not args.resume or args.stop_after is None):
        parser.error("--extend-horizon requires fit, --resume and --stop-after")
    if args.command == "fit" and (args.output is None or args.stop_after is None):
        parser.error("fit requires --output and --stop-after")
    if args.command == "evaluate" and args.output is None:
        parser.error("evaluate requires --output")
    if (args.cuda_memory_fraction is not None
            and (not 0 < args.cuda_memory_fraction <= 1 or torch.device(args.device).type != "cuda")):
        parser.error("--cuda-memory-fraction requires a CUDA device and a fraction in (0, 1]")
    recipe = json.loads(args.recipe.read_text())
    configure_r16_5_execution(recipe, device=args.device)
    if args.cuda_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=args.device)
    source_payload = None
    if ((recipe.get("thermal_state_response", {}).get("enabled")
         or recipe.get("r16_10_family") is not None)
            and (args.command == "evaluate" or (args.command == "fit" and args.resume))):
        selector = args.checkpoint if args.command == "evaluate" else args.resume
        source_payload = load_trusted_checkpoint(resolve_checkpoint(args.output, selector), map_location="cpu")
        if source_payload["experiment_identity"]["recipe"] != recipe:
            raise ValueError("The recipe differs from the saved run.")
        if (args.extend_horizon and recipe.get("r16_10_family") is not None
                and source_payload["experiment_identity"]["engine_config"]["total_epochs"] == 2000
                and (int(source_payload["epoch"]) != 2000
                     or Path(args.resume).name != "epoch_2000_model.pt")):
            raise ValueError("R16-10 extension requires the literal completed epoch 2000 checkpoint.")
    options = {}
    if source_payload is not None and recipe.get("r16_10_family") is not None:
        options["resume_payload"] = source_payload
    if source_payload is not None and recipe.get("thermal_state_response", {}).get("enabled"):
        # Objective denominators are saved scientific state. Recomputing their
        # last bits on another device must not change the restored objective.
        options["response_scales"] = source_payload["experiment_identity"]["provider_identity"]["response_scales"]
        if recipe.get("r16_5_warm_fork") is not None:
            options["resume_payload"] = source_payload
        elif recipe.get("r16_4_warm_fork") is None:
            native_identity = source_payload["experiment_identity"]["provider_identity"]
            options["saved_adapter_state"] = native_identity["adapter_state"]
            options["saved_normalization_source"] = native_identity["source_binding"]["normalization_source"]
    model, provider = build(recipe, device=args.device, **options)
    engine = (engine_for(recipe, device=args.device, execution_horizon=args.stop_after)
              if args.extend_horizon else engine_for(recipe, device=args.device))
    arm = recipe["member_routing"]["arm"] if isinstance(recipe.get("dataset"), dict) else recipe["architecture"]
    identity = {"workflow": "relation_anchored_case_training_v1", "recipe": recipe}
    if args.command == "inspect":
        cases, dataset_id = engine.epoch_case_order(provider, 1)
        result = {"provider": dict(provider.identity_payload()), "sampling_dataset_id": dataset_id,
                  "train_cases": len(cases), "parameters": sum(p.numel() for p in model.parameters()),
                  "optimizer_groups": [spec.name for spec in provider.optimizer_groups(model, arm, "joint")]}
    elif args.command == "fit":
        if recipe["dataset"] == "wind":
            print(json.dumps({"event": "wind_preparation_complete", "preparation": provider.preparation_summary()},
                             sort_keys=True, default=str, allow_nan=False), flush=True)
        result = engine.fit(model, provider, args.output, identity=identity, arm=arm,
                            stop_after=args.stop_after, resume_checkpoint=args.resume,
                            allow_horizon_extension=args.extend_horizon)
    else:
        saved = (source_payload if source_payload is not None else
                 load_trusted_checkpoint(resolve_checkpoint(args.output, args.checkpoint), map_location="cpu"))
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
