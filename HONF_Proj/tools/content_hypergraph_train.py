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
SCHEMA_V2 = "content_hypergraph_recipe_v2"
WORKFLOW_V2 = "content_hypergraph_joint_v2"
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
    version = {SCHEMA: 1, SCHEMA_V2: 2}.get(recipe.get("schema"))
    if version is None:
        raise ValueError(f"Expected {SCHEMA} or {SCHEMA_V2}.")
    if recipe.get("dataset") not in {"thermal", "wind"}:
        raise ValueError("dataset must be thermal or wind.")
    arms = {1: {"A-D", "A-H6"}, 2: {"A2-D", "A2-H6"}}[version]
    if recipe.get("architecture") not in arms:
        raise ValueError("Architecture does not belong to the declared recipe schema.")
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
    organized = "A-H6" if version == 1 else "A2-H6"
    if full and (recipe.get("approved_stop_after") != 1000 or recipe["architecture"] != organized):
        raise ValueError("A full-TRAIN follow-up requires an organized model and the explicit 1000 stop.")
    if recipe.get("precision", "fp32") != "fp32":
        raise ValueError("Unqualified precision policy; use the FP32 reference recipe.")
    if recipe.get("edge_access", "full") != "full":
        raise ValueError("Primary scientific fits require full edge access.")
    if version == 2:
        model = recipe.get("model", {})
        fixed_model = {"context_dim": 128, "num_heads": 4, "context_depth": 2,
                       "num_hyperedges": 6, "thermal_response_rank": 4 if thermal else 0,
                       "local_hidden_dim": 128, "dropout": 0.0}
        for name, expected in fixed_model.items():
            if model.get(name) != expected:
                raise ValueError(f"Round2 requires model.{name}={expected}.")
        if recipe.get("optimizer_bias_norm_exclusion") is not True:
            raise ValueError("Fresh Round2 recipes require explicit optimizer_bias_norm_exclusion.")
        if "query_edge_geometry_mode" in model:
            mode = model["query_edge_geometry_mode"]
            if mode != "typed_w_centroids_v1":
                raise ValueError("Unsupported model.query_edge_geometry_mode.")
            if recipe["architecture"] != "A2-H6":
                raise ValueError("Typed centroid query-edge geometry is available only for A2-H6.")
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
        if recipe["schema"] == SCHEMA_V2:
            from windfarm.training.content_hypergraph_task import build_wind_content_task_v2
            result = build_wind_content_task_v2(recipe, device=device)
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
    if recipe.get("schema") == SCHEMA_V2:
        from honf_runtime.content_hypergraph_qualification import validate_round2_qualification
        return validate_round2_qualification(path, recipe)
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


def load_numerical_pooling_amendment(path: Path) -> dict[str, Any]:
    """Load the engine-validated, separately evidenced resume amendment."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("Numerical pooling amendment must be a JSON object.")
    if value.get("schema") != "content_hypergraph_pooling_numerical_repair_v1":
        raise ValueError("Unsupported numerical pooling amendment schema.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--numerical-pooling-amendment", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    recipe = validate_recipe(json.loads(args.recipe.read_text()))
    validate_stop(recipe, args.stop_after)
    numerical_pooling_amendment = None
    if args.numerical_pooling_amendment is not None:
        if args.dry_run:
            raise ValueError("A numerical pooling amendment is only valid for an exact scientific resume.")
        if (recipe["dataset"], recipe["architecture"], recipe["protocol"]) != (
            "wind",
            "A-H6",
            "full_train1000",
        ):
            raise ValueError("Numerical pooling amendment is limited to Wind A-H6 full_train1000.")
        if args.resume is None:
            raise ValueError("A numerical pooling amendment requires --resume.")
        numerical_pooling_amendment = load_numerical_pooling_amendment(args.numerical_pooling_amendment)
    if args.resume is None and recipe["dataset"] == "wind" and recipe.get("optimizer_bias_norm_exclusion") is not True:
        raise ValueError("Fresh Wind recipes must explicitly enable optimizer_bias_norm_exclusion.")
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
    if qualification is not None and recipe["schema"] == SCHEMA_V2:
        if canonical_sha256(model.get_config()) != qualification["architecture_config_sha256"]:
            raise ValueError("Full constructor differs from the qualified Round2 architecture.")
        current_sources = loaded_source_binding()["files"]
        if any(current_sources.get(name) != digest for name, digest in qualification["development_loaded_source"]["files"].items()):
            raise ValueError("Imported runtime changed since the qualified Round2 development fit.")
        binding = qualification["partition_binding"]
        physical = dict(provider.identity_payload())
        from honf_runtime.content_hypergraph_qualification import physical_output_law
        if canonical_sha256(physical_output_law(physical, recipe["dataset"])) != qualification["physical_output_law_sha256"]:
            raise ValueError("Full native output law differs from the qualified Round2 model.")
        train_key = "training_case_ids_sha256" if recipe["dataset"] == "thermal" else "train_row_indices_sha256"
        validation_key = "validation_case_ids_sha256" if recipe["dataset"] == "thermal" else "validation_row_indices_sha256"
        if (binding.get("training_membership_sha256") != physical.get(train_key)
                or binding.get("validation_membership_sha256") != physical.get(validation_key)):
            raise ValueError("Full TRAIN/validation membership differs from its qualification binding.")
    engine = engine_for(recipe, args.device)
    identity = {
        "workflow": WORKFLOW_V2 if recipe["schema"] == SCHEMA_V2 else WORKFLOW,
        "model_family": "honf_content_hypergraph",
        "recipe": recipe, "recipe_sha256": canonical_sha256(recipe),
        "design_sha256": "6887c5436f6006fa8b6451cca5abe753e08a2d27ca3fed08cbbd336caeb63a44",
        "loaded_source": loaded_source_binding(),
        "data_locks": {"WindTEST": "locked", "solver_usage": "326/326", "new_solver_attempts": 0},
    }
    if recipe["schema"] == SCHEMA_V2:
        identity["round2_plan_sha256"] = "4138ca37dbf51fe9a28368e27a51b1e1d7fcc5f31ff327075f9d3e18d4b4234f"
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
    fit_options = {"stop_after": args.stop_after, "resume_checkpoint": args.resume}
    if numerical_pooling_amendment is not None:
        fit_options["numerical_pooling_amendment"] = numerical_pooling_amendment
    result = engine.fit(
        model,
        provider,
        args.output,
        identity=identity,
        arm=recipe["architecture"],
        **fit_options,
    )
    if recipe["schema"] == SCHEMA_V2 and hasattr(provider, "catalogue_cache"):
        cache = provider.catalogue_cache.summary()
        receipt = {"schema": "content_hypergraph_provider_runtime_v2", "status": "completed",
                   "completed_epoch": result["completed_epoch"], "architecture": recipe["architecture"],
                   "native_catalogue_cache": cache, "scope": "this segment's process and shared cache",
                   "WindTEST": "locked", "new_solver_attempts": 0}
        destination = args.output / "metrics" / f"provider_runtime_epoch_{result['completed_epoch']:04d}.json"
        if destination.exists():
            raise FileExistsError("Provider runtime receipts are immutable; an epoch receipt already exists.")
        destination.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
