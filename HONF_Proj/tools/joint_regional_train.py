#!/usr/bin/env python
"""Fresh joint field development and explicitly manual full-TRAIN recipes.

All optimization uses the maintained TrainingEngine. Preparation never takes
an optimizer step; dry-run uses a disposable CPU model. Formal optimization
requires the explicit --manual-formal-launch switch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "Case_ThermalChannel/src", ROOT / "Case_WindFarm/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

MODES = ("J-direct", "J-geometry", "J-H")
RECIPE_KEYS = {
    "schema_version", "task", "mode", "seed", "hidden", "message", "regional_anchors",
    "depth", "receiver_tile", "primary_queries", "microbatch_cases", "effective_cases",
    "formal_full", "total_epochs", "initialization", "launch_policy", "dataset_protocol",
    "response_coefficient", "operator_coefficient",
    "auxiliary_calibration",
}


def read_recipe(path: str | Path) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text())
    return validate_recipe(recipe)


def validate_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(recipe, dict) or set(recipe) - RECIPE_KEYS:
        raise ValueError("Joint recipe contains unsupported fields.")
    if recipe.get("schema_version") != 1 or recipe.get("task") not in ("thermal", "wind"):
        raise ValueError("Joint recipe schema/task is invalid.")
    if recipe.get("mode") not in MODES or recipe.get("initialization") != "fresh_all_trainable":
        raise ValueError("Joint recipes require a declared mode and fresh trainable weights.")
    if recipe.get("formal_full") and recipe.get("launch_policy") != "manual_only":
        raise ValueError("Formal recipes must be manual-only.")
    for name in ("hidden", "message", "regional_anchors", "depth", "receiver_tile", "primary_queries",
                 "microbatch_cases", "effective_cases", "total_epochs"):
        if type(recipe.get(name)) is not int or recipe[name] < 1:
            raise ValueError(f"Recipe {name} must be a positive integer.")
    effective = 48 if recipe["task"] == "thermal" else 24
    if recipe["effective_cases"] != effective or recipe["microbatch_cases"] > effective:
        raise ValueError("Joint recipes preserve the dataset's effective batch.")
    if recipe["total_epochs"] != (5000 if recipe.get("formal_full") else 2500):
        raise ValueError("Joint recipe horizon must match the development/formal identity.")
    if type(recipe.get("seed")) is not int or type(recipe.get("formal_full")) is not bool:
        raise ValueError("Joint seed/formal identity has invalid types.")
    expected = ("original600_train" if recipe["task"] == "thermal" else "original420_train") if recipe["formal_full"] else (
        "fixed25_v1" if recipe["task"] == "thermal" else "wind_shared_fixed24_v1")
    if recipe.get("dataset_protocol") != expected:
        raise ValueError("Joint recipe dataset protocol differs from its population identity.")
    for name in ("response_coefficient", "operator_coefficient"):
        if name in recipe and (not isinstance(recipe[name], (float, int)) or
                               not math.isfinite(recipe[name]) or recipe[name] < 0):
            raise ValueError(f"Recipe {name} must be finite and nonnegative.")
    return recipe


def resolved_recipe(args: argparse.Namespace) -> dict[str, Any]:
    if args.recipe_json:
        return read_recipe(args.recipe_json)
    task = args.task
    if task is None:
        raise ValueError("Provide --task or --recipe-json.")
    return validate_recipe({
        "schema_version": 1, "task": task, "mode": args.mode,
        "seed": args.seed if args.seed is not None else (0 if task == "thermal" else 42),
        "hidden": args.hidden, "message": args.message,
        "regional_anchors": args.regional_anchors or (16 if task == "thermal" else 32),
        "depth": args.depth, "receiver_tile": args.receiver_tile,
        "primary_queries": args.primary_queries or (1024 if task == "thermal" else 4096),
        "microbatch_cases": args.microbatch_cases or (4 if task == "thermal" else 1),
        "effective_cases": 48 if task == "thermal" else 24,
        "formal_full": False, "total_epochs": 2500,
        "initialization": "fresh_all_trainable", "launch_policy": "bounded_development",
        "dataset_protocol": "fixed25_v1" if task == "thermal" else "wind_shared_fixed24_v1",
    })


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build(recipe: dict[str, Any], *, device: str):
    from honf_runtime.unified_training import EngineConfig, SelectionPolicy, TrainingEngine
    seed_all(recipe["seed"])
    if recipe["task"] == "thermal":
        from channelthermal.training.joint_task import build_thermal_joint_task as factory
    else:
        from windfarm.training.joint_task import build_wind_joint_task as factory
    task_options = {name: recipe[name] for name in ("response_coefficient", "operator_coefficient", "auxiliary_calibration")
                    if name in recipe}
    model, provider = factory(
        mode=recipe["mode"], device=device, seed=recipe["seed"], hidden=recipe["hidden"],
        message=recipe["message"], regional_anchors=recipe["regional_anchors"], depth=recipe["depth"],
        receiver_tile=recipe["receiver_tile"], primary_queries=recipe["primary_queries"],
        microbatch_size=recipe["microbatch_cases"], effective_batch_size=recipe["effective_cases"],
        formal_full=recipe["formal_full"], total_epochs=recipe["total_epochs"],
        **task_options,
    )
    config = EngineConfig(
        seed=recipe["seed"], microbatch_cases=recipe["microbatch_cases"],
        effective_cases=recipe["effective_cases"], total_epochs=recipe["total_epochs"],
        training_mode="joint", sampling_version="case_epoch_v1", monitor_every=100,
        warmup_epochs=20, open_through_epoch=20, soft_through_epoch=20,
        latest_every=100, curve_every=100,
    )
    engine = TrainingEngine(config, device=device, selection=SelectionPolicy(field_metric="field_score"))
    identity = {
        "workflow": "joint_regional_fields_v1", "recipe": recipe,
        "initial_model_state_sha256": model_digest(model),
        "initialization": "fresh_all_trainable", "external_learned_field_files": [],
        "solver_attempts": 0, "WindTEST_targets": "locked",
    }
    return model, provider, engine, identity


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
    os.replace(temporary, path)


def summary(model: Any, provider: Any, engine: Any, identity: dict[str, Any]) -> dict[str, Any]:
    from dataclasses import asdict
    groups = provider.optimizer_groups(model, identity["recipe"]["mode"], "joint")
    covered = [name for group in groups for name in group.parameter_names]
    active = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if len(covered) != len(set(covered)) or set(covered) != active:
        raise ValueError("Declared joint optimizer groups omit or duplicate an active parameter.")
    return {
        "identity": identity, "provider_identity": provider.identity_payload(),
        "preparation": provider.preparation_summary(),
        "engine_config": asdict(engine.config), "selection_policy": asdict(engine.selection),
        "active_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "optimizer_parameter_coverage": "all active parameters exactly once",
        "optimizer_schedule_contract": [asdict(group) for group in groups],
        "optimizer_started": False, "formal_training_started": False,
    }


def validate_manual_recipe(model: Any, provider: Any, engine: Any, identity: dict[str, Any]) -> dict[str, Any]:
    """Inert fullTRAIN identity/head validation; no optimizer is constructed."""
    import torch

    from honf_runtime.unified_training import SamplingKey, _sampling_dataset_id
    result = summary(model, provider, engine, identity)
    cases = tuple(provider.epoch_cases(1, identity["recipe"]["seed"]))
    if not cases:
        raise ValueError("Manual recipe has no TRAIN cases.")
    dataset_id = _sampling_dataset_id(provider.identity_payload())
    key = SamplingKey(identity["recipe"]["seed"], 1, 0, 0, "joint", identity["recipe"]["mode"],
                      sampling_version="case_epoch_v1", dataset_id=dataset_id)
    checks = []
    for case in dict.fromkeys((cases[0], cases[-1])):
        batch = provider.make_batch((case,), key)
        model.eval()
        with torch.no_grad():
            predictions, auxiliary = provider.predict_native(
                model, provider.make_scene(batch.scene_inputs), batch.receivers,
                identity["recipe"]["mode"], "joint", epoch=1, temperature=1.0)
            terms = provider.validation_loss_terms(predictions, batch.targets, auxiliary,
                                                   batch=batch, arm=identity["recipe"]["mode"])
            if not terms or not all(bool(torch.isfinite(term.numerator)) for term in terms.values()):
                raise FloatingPointError("Manual recipe native physical head check is nonfinite.")
        checks.append({"case_key": case, "finite_physical_head_terms": sorted(terms)})
    result["native_forward_checks"] = checks
    result["optimizer_instantiated"] = False
    result["optimizer_updates"] = 0
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "dry-run", "start", "resume"))
    parser.add_argument("--task", choices=("thermal", "wind"))
    parser.add_argument("--mode", choices=MODES, default="J-H")
    parser.add_argument("--recipe-json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--message", type=int, default=128)
    parser.add_argument("--regional-anchors", type=int)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--receiver-tile", type=int, default=512)
    parser.add_argument("--primary-queries", type=int)
    parser.add_argument("--microbatch-cases", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--manual-formal-launch", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    recipe = resolved_recipe(args)
    if args.command in ("start", "resume"):
        if not args.output_dir or args.stop_after is None:
            parser.error("Optimization requires --output-dir and --stop-after.")
        if recipe["formal_full"] and not args.manual_formal_launch:
            parser.error("Formal optimization requires explicit --manual-formal-launch; preparation is inert.")
        if not 1 <= args.stop_after <= recipe["total_epochs"]:
            parser.error("Requested stop exceeds the declared horizon.")
    if args.command in ("prepare", "dry-run"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        args.device = "cpu"
    import torch
    torch.set_num_threads(args.threads)
    started = time.time()
    model, provider, engine, identity = build(recipe, device=args.device)
    setup_seconds = time.time() - started
    if args.command == "prepare":
        payload = {"status": "prepared_only", "cold_setup_seconds": setup_seconds,
                   **summary(model, provider, engine, identity)}
        if args.output_dir:
            write_json(Path(args.output_dir) / "preparation.json", payload)
    elif args.command == "dry-run":
        if recipe["formal_full"]:
            # FullTRAIN preparation is deliberately inert. No formal optimizer
            # is instantiated, even for a disposable update.
            payload = {"status": "manual_formal_recipe_validated_inert",
                       **validate_manual_recipe(model, provider, engine, identity)}
        else:
            payload = {"status": "disposable_cpu_joint_update", "cold_setup_seconds": setup_seconds,
                       "update": engine.preflight_one_update(model, provider, arm=recipe["mode"]),
                       "formal_training_started": False, "checkpoint_written": False}
    else:
        from honf_runtime.run_layout import resolve_checkpoint
        output = Path(args.output_dir).resolve()
        if args.command == "start":
            write_json(output / "resolved_recipe.json", recipe)
            write_json(output / "preparation.json", {"cold_setup_seconds": setup_seconds,
                       **summary(model, provider, engine, identity)})
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        payload = engine.fit(
            model, provider, output, identity=identity, arm=recipe["mode"],
            stop_after=args.stop_after,
            resume_checkpoint=resolve_checkpoint(output, "latest") if args.command == "resume" else None,
        )
        payload["cold_setup_seconds"] = setup_seconds
        payload["outer_elapsed_seconds"] = time.time() - started
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize()
            payload["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            payload["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(output / f"invocation_{args.command}_e{args.stop_after:04d}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
