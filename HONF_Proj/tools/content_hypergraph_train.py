#!/usr/bin/env python3
"""Train the opt-in Oct13 family with the maintained joint TrainingEngine.

Dry-run resolves the physical/data contract without an optimizer update or run
allocation. A stop is separate from the immutable learning-rate horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src", PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch

from honf_runtime.unified_training import (
    EngineConfig,
    SamplingKey,
    SelectionPolicy,
    TrainingEngine,
)

SCHEMA = "content_hypergraph_recipe_v1"
WORKFLOW = "content_hypergraph_joint_v1"
ALLOWED_UUIDS = {
    "GPU-233fcd85-5c6a-6212-3f44-655252afec70",
    "GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e",
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def loaded_source_binding() -> dict[str, Any]:
    """Bind actual imported project code once, outside the training loop."""
    files = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(PROJECT) or not path.is_file():
            continue
        files[path.relative_to(PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files": dict(sorted(files.items())), "sha256": canonical_sha256(files)}


def validate_recipe(value: dict[str, Any]) -> dict[str, Any]:
    """Keep this family's schema separate from sealed historical recipes."""
    recipe = dict(value)
    if recipe.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}.")
    if recipe.get("dataset") not in {"thermal", "wind"}:
        raise ValueError("dataset must be thermal or wind.")
    if recipe.get("architecture") not in {"A-D", "A-H6"}:
        raise ValueError("The primary family has only A-D and A-H6 architectures.")
    if recipe.get("protocol") not in {"development", "full_train1000"}:
        raise ValueError("Use an explicitly bound development or full_train1000 protocol.")
    thermal = recipe["dataset"] == "thermal"
    fixed = {"seed": 0 if thermal else 42,
             "primary_queries": 1024 if thermal else 4096,
             "effective_batch_size": 48 if thermal else 24}
    for name, expected in fixed.items():
        if type(recipe.get(name)) is not int or recipe[name] != expected:
            raise ValueError(f"The primary recipe requires {name}={expected}.")
    micro = recipe.get("microbatch_size")
    if type(micro) is not int or not 1 <= micro <= fixed["effective_batch_size"]:
        raise ValueError("Invalid microbatch_size.")
    full = recipe["protocol"] == "full_train1000"
    horizon = 5000 if full else 2500
    if recipe.get("total_epochs") != horizon:
        raise ValueError(f"The immutable schedule horizon must be {horizon}.")
    if full and (recipe.get("approved_stop_after") != 1000 or recipe["architecture"] != "A-H6"):
        raise ValueError("A full-TRAIN follow-up requires an organized model and the explicit 1000 stop.")
    if recipe.get("precision", "fp32") != "fp32":
        raise ValueError("Unqualified precision policy; use the FP32 reference recipe.")
    if recipe.get("edge_access", "full") != "full":
        raise ValueError("Primary scientific fits require full edge access.")
    return recipe


def validate_stop(recipe: dict[str, Any], stop_after: int) -> None:
    maximum = 1000 if recipe["protocol"] == "full_train1000" else 2500
    if type(stop_after) is not int or stop_after < 1 or stop_after > maximum:
        raise ValueError(f"Execution stop must lie in 1..{maximum}; the schedule is not launch authorization.")


def build(recipe: dict[str, Any], *, device: str):
    """Delegate physical construction to the dataset-owned provider."""
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if recipe["dataset"] == "thermal":
        from channelthermal.training.content_hypergraph_task import build_thermal_content_hypergraph_task
        result = build_thermal_content_hypergraph_task(recipe, device=device)
    else:
        from windfarm.training.content_hypergraph_task import build_wind_content_task
        result = build_wind_content_task(recipe, device=device)
    model, provider = result[:2]
    return model, provider


def engine_for(recipe: dict[str, Any], device: str) -> TrainingEngine:
    maximum = 1000 if recipe["protocol"] == "full_train1000" else 2500
    config = EngineConfig(
        seed=recipe["seed"], microbatch_cases=recipe["microbatch_size"],
        effective_cases=recipe["effective_batch_size"], total_epochs=recipe["total_epochs"],
        monitor_every=100, checkpoint_epochs=tuple(range(100, maximum + 1, 100)),
        latest_every=100, curve_every=100, gradient_clip=1.0,
        sampling_version=SamplingKey.CASE_EPOCH_VERSION, training_mode="joint",
    )
    return TrainingEngine(config, device=device, selection=SelectionPolicy(field_metric="field_score"))


def validate_qualification(path: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    qualification = json.loads(path.read_text())
    required = {"correctness_pass", "edge_content_used", "nontrivial_train_dev_fields", "budget_forecast_fits"}
    if qualification.get("schema") != "content_hypergraph_full1000_qualification_v1":
        raise ValueError("Missing this family's measured full-TRAIN qualification.")
    if qualification.get("dataset") != recipe["dataset"] or not all(qualification.get(k) is True for k in required):
        raise ValueError("The dataset has not qualified for a fresh full-TRAIN follow-up.")
    if int(qualification.get("common_development_epoch", 0)) < 1000:
        raise ValueError("Full-TRAIN qualification requires a common development age >=1000.")
    for key, maximum in (("field_score_ratio", 1.25), ("priority_field_ratio", 1.25),
                         ("maximum_resolved_component_ratio", 2.0), ("high_m_q8192_time_ratio", 2.0)):
        value = qualification.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0 or value > maximum:
            raise ValueError(f"Full-TRAIN qualification failed {key}.")
    return qualification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    recipe = validate_recipe(json.loads(args.recipe.read_text()))
    validate_stop(recipe, args.stop_after)
    if args.dry_run and args.device != "cpu":
        raise ValueError("Dry-run must be CUDA-hidden CPU preparation.")
    if not args.dry_run:
        if not args.yes or args.output is None:
            raise ValueError("A scientific start requires --yes and an explicit fresh output directory.")
        if args.device.startswith("cuda") and (
            args.device not in {"cuda", "cuda:0"} or os.getenv("CUDA_VISIBLE_DEVICES") not in ALLOWED_UUIDS
        ):
            raise ValueError("Expose exactly physical GPU0 or GPU1 by UUID and use logical cuda:0.")
    qualification = None
    if recipe["protocol"] == "full_train1000" and not args.dry_run:
        if args.qualification is None:
            raise ValueError("A full-TRAIN start requires --qualification with measured Section13 evidence.")
        qualification = validate_qualification(args.qualification, recipe)
    model, provider = build(recipe, device=args.device)
    engine = engine_for(recipe, args.device)
    identity = {
        "workflow": WORKFLOW, "model_family": "honf_content_hypergraph",
        "recipe": recipe, "recipe_sha256": canonical_sha256(recipe),
        "design_sha256": "6887c5436f6006fa8b6451cca5abe753e08a2d27ca3fed08cbbd336caeb63a44",
        "loaded_source": loaded_source_binding(),
        "data_locks": {"WindTEST": "locked", "solver_usage": "326/326", "new_solver_attempts": 0},
    }
    if qualification is not None:
        identity["full1000_qualification"] = qualification
    summary = {"recipe": recipe, "provider": dict(provider.identity_payload()),
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "engine": vars(engine.config), "requested_stop": args.stop_after,
               "optimizer_updates": 0, "run_allocated": False}
    print(json.dumps(summary, sort_keys=True, default=str, allow_nan=False), flush=True)
    if args.dry_run:
        return 0
    result = engine.fit(model, provider, args.output, identity=identity, arm=recipe["architecture"],
                        stop_after=args.stop_after, resume_checkpoint=args.resume)
    print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
