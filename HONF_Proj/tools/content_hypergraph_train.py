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
from dataclasses import asdict
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
from honf_runtime.content_hypergraph_formal import (
    FORMAL_PROTOCOL,
    validate_formal_d_recipe,
    validate_formal_d_stop,
)
from honf_runtime.training_output_policy import (
    OUTPUT_POLICY_SCHEMA,
    validate_output_policy_amendment,
)
from honf_runtime.source_read_protocol import (
    SOURCE_READ_DEVELOPMENT, SOURCE_READ_FULL, SOURCE_READ_PROTOCOLS,
    validate_source_read_recipe, validate_source_read_stop,
)
from honf_runtime.source_read_qualification import validate_source_read_qualification

SCHEMA = "content_hypergraph_recipe_v1"
WORKFLOW = "content_hypergraph_joint_v1"
SCHEMA_V2 = "content_hypergraph_recipe_v2"
WORKFLOW_V2 = "content_hypergraph_joint_v2"
ALLOWED_UUIDS = {
    "GPU-233fcd85-5c6a-6212-3f44-655252afec70",
    "GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e",
}
SOURCE_READ_ALLOWED_UUIDS = {
    "GPU-233fcd85-5c6a-6212-3f44-655252afec70",
    "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39",
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
    if recipe.get("protocol") in SOURCE_READ_PROTOCOLS:
        return validate_source_read_recipe(recipe)
    if recipe.get("protocol") == FORMAL_PROTOCOL:
        return validate_formal_d_recipe(recipe)
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
    if recipe["protocol"] in SOURCE_READ_PROTOCOLS:
        validate_source_read_stop(recipe, stop_after)
        return
    if recipe["protocol"] == FORMAL_PROTOCOL:
        validate_formal_d_stop(recipe, stop_after)
        return
    maximum = 1000 if recipe["protocol"] == "full_train1000" else 2500
    if type(stop_after) is not int or stop_after < 1 or stop_after > maximum:
        raise ValueError(f"Execution stop must lie in 1..{maximum}; the schedule is not launch authorization.")


def _is_sparse_source_read_recipe(recipe: dict[str, Any]) -> bool:
    return recipe.get("source_reader_variant") in ("S-Utility", "S-Complete")


def _load_source_read_child_modules() -> None:
    """Import the complete sparse runtime before sealing child source identity."""
    import honf_forward_core.content_hypergraph.query_chunks_v1  # noqa: F401
    import honf_forward_core.content_hypergraph.source_read_decoder_v1  # noqa: F401
    import honf_forward_core.content_hypergraph.source_read_response  # noqa: F401
    import honf_forward_core.content_hypergraph.source_read_sparse  # noqa: F401
    import honf_runtime.source_read_auxiliary  # noqa: F401
    import honf_runtime.source_read_child  # noqa: F401


def _source_read_child_auxiliary_callback(variant: str):
    if variant not in ("S-Utility", "S-Complete"):
        raise ValueError("Auxiliary fitting is available only for S-Utility/S-Complete children.")
    from honf_runtime.source_read_auxiliary import fit_source_read_child_auxiliary

    def callback(model, provider, optimizer, **kwargs):
        return fit_source_read_child_auxiliary(
            model, provider, optimizer, variant=variant, **kwargs,
        )

    return callback


def source_read_child_fit_options(
    recipe: dict[str, Any],
    declaration: Any,
    parent_checkpoint: Path | None,
    *,
    resume_checkpoint: Path | None = None,
    qualification: Path | None = None,
    numerical_pooling_amendment: Path | None = None,
    output_policy_amendment: Path | None = None,
) -> dict[str, Any]:
    """Resolve only the audited development e500 sparse-child transition."""
    sparse = _is_sparse_source_read_recipe(recipe)
    if recipe.get("protocol") == SOURCE_READ_FULL and sparse:
        raise ValueError(
            "Full sparse source-read training is not wired; the only full1000 template is D-scale."
        )
    if declaration is None:
        if parent_checkpoint is not None:
            raise ValueError("--branch-from-checkpoint requires --source-read-child-declaration.")
        if sparse and recipe.get("protocol") == SOURCE_READ_DEVELOPMENT and resume_checkpoint is None:
            raise ValueError(
                "A fresh sparse development run requires its audited dense e500 child declaration."
            )
        if sparse:
            _load_source_read_child_modules()
        return {}
    if (recipe.get("protocol") != SOURCE_READ_DEVELOPMENT or not sparse
            or resume_checkpoint is not None or parent_checkpoint is None
            or qualification is not None or numerical_pooling_amendment is not None
            or output_policy_amendment is not None):
        raise ValueError(
            "A source-read child declaration requires a fresh S-Utility/S-Complete development branch "
            "with one explicit parent and no resume or unrelated amendment."
        )
    if not isinstance(declaration, dict):
        raise TypeError("Source-read child declaration must be a JSON object.")
    declared_path = declaration.get("source_checkpoint")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError("Source-read child declaration must bind its exact source_checkpoint path.")
    try:
        parent_path = parent_checkpoint.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("Source-read dense parent checkpoint does not exist.") from exc
    if not parent_path.is_file() or declared_path != str(parent_path):
        raise ValueError(
            "--branch-from-checkpoint must resolve to the exact canonical path in the child declaration."
        )
    if declaration.get("source_reader_variant") != recipe.get("source_reader_variant"):
        raise ValueError("Source-read child declaration and recipe variants differ.")
    _load_source_read_child_modules()
    return {
        "branch_from_checkpoint": parent_path,
        "source_read_child": declaration,
        "source_read_child_auxiliary_fit": _source_read_child_auxiliary_callback(
            recipe["source_reader_variant"]
        ),
    }


def preflight_source_read_child(
    model,
    provider,
    engine: TrainingEngine,
    *,
    identity: dict[str, Any],
    recipe: dict[str, Any],
    declaration: dict[str, Any],
    parent_checkpoint: Path,
    output: Path | None,
    stop_after: int,
) -> dict[str, Any]:
    """Validate and load the exact parent in memory before any output allocation."""
    from honf_runtime.compat import load_trusted_checkpoint
    from honf_runtime.run_layout import resolve_checkpoint
    from honf_runtime.source_read_child import prepare_source_read_child
    from honf_runtime.unified_training import (
        _engine_config_payload, _optimizer_spec_payload, _sampling_dataset_id,
    )

    source_path = resolve_checkpoint(parent_checkpoint.parent, parent_checkpoint)
    if not source_path.is_file():
        raise ValueError("Source-read e500 parent checkpoint is not a file.")
    if output is not None:
        output_path = output.expanduser().resolve()
        try:
            source_path.resolve().relative_to(output_path)
        except ValueError:
            pass
        else:
            raise ValueError("A source-read child must use an output directory outside its dense parent.")
        if output_path.exists() and any(output_path.iterdir()):
            raise ValueError("A source-read child requires a distinct empty output directory.")

    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if declaration.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError("Dense e500 checkpoint SHA256 differs from the audited child declaration.")
    source_payload = load_trusted_checkpoint(source_path, map_location="cpu")
    provider_identity = dict(provider.identity_payload())
    schedules = tuple(provider.optimizer_groups(model, recipe["architecture"], "joint"))
    sealed_identity = {
        **identity,
        "provider_identity": provider_identity,
        "engine_config": _engine_config_payload(engine.config),
        "selection_policy": asdict(engine.selection),
        "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in schedules],
    }
    if engine.config.sampling_version == SamplingKey.CASE_EPOCH_VERSION:
        sealed_identity["sampling_dataset_id"] = _sampling_dataset_id(provider_identity)
    amendment = prepare_source_read_child(
        model,
        source_payload,
        sealed_identity,
        declaration,
        source_sha256=source_sha256,
        source_path=source_path,
        case_keys=tuple(provider.epoch_cases(501, engine.config.seed)),
        stop_after=stop_after,
    )
    return {
        "status": "preflight_passed",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "source_epoch": 500,
        "source_reader_variant": recipe["source_reader_variant"],
        "train_case_count": len(provider.epoch_cases(501, engine.config.seed)),
        "exact_parent_state_copied": amendment["exact_parent_state_copied"],
        "exact_loaded_source_changes": amendment["source_changes"],
        "auxiliary_fit_mode": declaration["auxiliary_fit_mode"],
        "auxiliary_update_limit": declaration["auxiliary_update_limit"],
    }


def build(recipe: dict[str, Any], *, device: str):
    """Delegate physical construction to the dataset-owned provider."""
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if recipe["protocol"] in SOURCE_READ_PROTOCOLS:
        if recipe["dataset"] == "thermal":
            from channelthermal.training.content_hypergraph_task_v2 import build_thermal_source_read_task_v1
            result = build_thermal_source_read_task_v1(recipe, device=device)
        else:
            from windfarm.training.content_hypergraph_task import build_wind_source_read_task_v1
            result = build_wind_source_read_task_v1(recipe, device=device)
    elif recipe["protocol"] == FORMAL_PROTOCOL:
        if recipe["dataset"] == "thermal":
            from channelthermal.training.content_hypergraph_task_v2 import build_thermal_formal_content_task_v1
            result = build_thermal_formal_content_task_v1(recipe, device=device)
        else:
            from windfarm.training.content_hypergraph_task import build_wind_formal_content_task_v1
            result = build_wind_formal_content_task_v1(recipe, device=device)
    elif recipe["dataset"] == "thermal":
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
    if recipe["protocol"] == FORMAL_PROTOCOL or recipe["protocol"] in SOURCE_READ_PROTOCOLS:
        policy = recipe["execution"]
        config = EngineConfig(
            seed=recipe["sampling_seed"], microbatch_cases=recipe["microbatch_size"],
            effective_cases=recipe["effective_batch_size"], total_epochs=recipe["total_epochs"],
            monitor_every=policy["monitor_every"],
            checkpoint_epochs=tuple(policy["checkpoint_epochs"]),
            latest_every=policy["latest_every"], curve_every=policy["curve_every"],
            gradient_clip=1.0, sampling_version=SamplingKey.CASE_EPOCH_VERSION,
            training_mode="joint",
        )
        return TrainingEngine(config, device=device,
                              selection=SelectionPolicy(field_metric=policy["field_metric"]))
    maximum = 1000 if recipe["protocol"] == "full_train1000" else 2500
    config = EngineConfig(
        seed=recipe["seed"], microbatch_cases=recipe["microbatch_size"],
        effective_cases=recipe["effective_batch_size"], total_epochs=recipe["total_epochs"],
        monitor_every=100, checkpoint_epochs=tuple(range(100, maximum + 1, 100)),
        latest_every=100, curve_every=10, gradient_clip=1.0,
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
    parser.add_argument("--source-read-child-declaration", type=Path,
                        help="Audited Round3 sparse-child declaration for a dense development e500 parent.")
    parser.add_argument("--branch-from-checkpoint", type=Path,
                        help="Exact e500 checkpoint named by --source-read-child-declaration.")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--numerical-pooling-amendment", type=Path)
    parser.add_argument("--output-policy-amendment", type=Path,
                        help="Explicit audited output-only continuation of these formal D histories.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    recipe = validate_recipe(json.loads(args.recipe.read_text()))
    validate_stop(recipe, args.stop_after)
    child_declaration = (
        json.loads(args.source_read_child_declaration.read_text())
        if args.source_read_child_declaration is not None else None
    )
    source_child_fit_options = source_read_child_fit_options(
        recipe,
        child_declaration,
        args.branch_from_checkpoint,
        resume_checkpoint=args.resume,
        qualification=args.qualification,
        numerical_pooling_amendment=args.numerical_pooling_amendment,
        output_policy_amendment=args.output_policy_amendment,
    )
    output_policy_amendment = None
    if args.output_policy_amendment is not None:
        if (recipe["protocol"] != FORMAL_PROTOCOL or args.resume is None or args.output is None
                or args.qualification is not None or args.numerical_pooling_amendment is not None):
            raise ValueError("An output-policy amendment requires only a same-run formal D --resume and --output.")
        output_policy_amendment = json.loads(args.output_policy_amendment.read_text())
        if output_policy_amendment.get("schema") != OUTPUT_POLICY_SCHEMA:
            raise ValueError("Unsupported output-policy amendment schema.")
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
    if (recipe["protocol"] != FORMAL_PROTOCOL and recipe["protocol"] not in SOURCE_READ_PROTOCOLS
            and args.resume is None
            and recipe["dataset"] == "wind" and recipe.get("optimizer_bias_norm_exclusion") is not True):
        raise ValueError("Fresh Wind recipes must explicitly enable optimizer_bias_norm_exclusion.")
    if recipe["protocol"] == FORMAL_PROTOCOL:
        if args.qualification is not None or args.numerical_pooling_amendment is not None:
            raise ValueError("Formal D authorization is separate from H6 qualification and pooling repairs.")
        if not args.dry_run:
            if args.output is None:
                raise ValueError("A formal D run requires a distinct output directory.")
            if args.resume is None:
                if args.output.exists() and any(args.output.iterdir()):
                    raise ValueError("Fresh formal D initialization requires an empty new run directory.")
            else:
                from honf_runtime.run_layout import resolve_checkpoint
                if resolve_checkpoint(args.output, args.resume) != resolve_checkpoint(args.output, "latest"):
                    raise ValueError("Formal D can resume only this run's current latest state.")
    if args.dry_run and args.device != "cpu":
        raise ValueError("Dry-run must be CUDA-hidden CPU preparation.")
    if not args.dry_run:
        if not args.yes or args.output is None:
            raise ValueError("A scientific start requires --yes and an explicit fresh output directory.")
        allowed = SOURCE_READ_ALLOWED_UUIDS if recipe["protocol"] in SOURCE_READ_PROTOCOLS else ALLOWED_UUIDS
        if args.device.startswith("cuda") and (
            args.device not in {"cuda", "cuda:0"} or os.getenv("CUDA_VISIBLE_DEVICES") not in allowed
        ):
            lanes = "GPU0 or GPU2" if recipe["protocol"] in SOURCE_READ_PROTOCOLS else "GPU0 or GPU1"
            raise ValueError(f"Expose exactly physical {lanes} by UUID and use logical cuda:0.")
    qualification = None
    if recipe["protocol"] == "full_train1000" and not args.dry_run:
        if args.qualification is None:
            raise ValueError("A full-TRAIN start requires --qualification with measured Section13 evidence.")
        qualification = validate_qualification(args.qualification, recipe)
    if recipe["protocol"] == SOURCE_READ_FULL and not args.dry_run:
        if args.qualification is None:
            raise ValueError("A source-read full1000 start requires its measured Round3 qualification.")
        qualification = validate_source_read_qualification(args.qualification, recipe, output_dir=args.output)
    model, provider = build(recipe, device=args.device)
    if qualification is not None and recipe["protocol"] == SOURCE_READ_FULL:
        if canonical_sha256(model.get_config()) != qualification.get("full_model_config_sha256"):
            raise ValueError("Full source-reader constructor differs from its qualified dry-run binding.")
        if canonical_sha256(dict(provider.identity_payload())) != qualification.get("full_provider_identity_sha256"):
            raise ValueError("Full source-reader data/transform/provider differs from its qualification.")
        observed_sources = loaded_source_binding()["files"]
        if observed_sources != qualification["full_loaded_source"]["files"]:
            raise ValueError("Imported full runtime differs from the qualified source-reader binding.")
    if (qualification is not None and recipe["schema"] == SCHEMA_V2
            and recipe["protocol"] not in SOURCE_READ_PROTOCOLS):
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
    if recipe["protocol"] in SOURCE_READ_PROTOCOLS:
        plan = PROJECT / "UpgradePlan/HONF_A2D_Fidelity_and_Sparse_Execution_Round3_Plan.md"
        identity["source_read_plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
        identity["source_read_authorization"] = {
            "protocol": recipe["protocol"], "variant": recipe["source_reader_variant"],
            "hard_stop": recipe["approved_stop_after"],
            "gpu_scope": "physical0/2, separately authorized Round3 study",
            "full_data_launch_requires_measured_qualification": True,
        }
    if recipe["protocol"] == FORMAL_PROTOCOL:
        identity["formal_d_authorization"] = {
            "scope": "fresh full-TRAIN A2-D, independent of the H6 full1000 protocol",
            "dataset": recipe["dataset"], "approved_stop_after": recipe["approved_stop_after"],
            "initialization": "fresh model weights and empty AdamW state; exact same-run latest resume only",
        }
    if qualification is not None:
        identity["full1000_qualification"] = qualification
    summary = {"recipe": recipe, "provider": dict(provider.identity_payload()),
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "engine": vars(engine.config), "requested_stop": args.stop_after,
               "optimizer_updates": 0, "run_allocated": False,
               "loaded_source": identity["loaded_source"]}
    if child_declaration is not None:
        summary["source_read_child_preflight"] = preflight_source_read_child(
            model,
            provider,
            engine,
            identity=identity,
            recipe=recipe,
            declaration=child_declaration,
            parent_checkpoint=source_child_fit_options["branch_from_checkpoint"],
            output=args.output,
            stop_after=args.stop_after,
        )
    if args.dry_run and output_policy_amendment is not None:
        from honf_runtime.compat import load_trusted_checkpoint
        from honf_runtime.run_layout import resolve_checkpoint
        from honf_runtime.unified_training import (
            _engine_config_payload, _optimizer_spec_payload, _sampling_dataset_id,
            named_optimizer_state,
        )
        source = resolve_checkpoint(args.output, args.resume)
        if source != resolve_checkpoint(args.output, "latest"):
            raise ValueError("Output-policy continuation requires this run's current latest checkpoint.")
        saved = load_trusted_checkpoint(source, map_location="cpu")
        specs = tuple(provider.optimizer_groups(model, recipe["architecture"], "joint"))
        sealed = {**identity, "provider_identity": dict(provider.identity_payload()),
                  "engine_config": _engine_config_payload(engine.config),
                  "selection_policy": asdict(engine.selection),
                  "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in specs],
                  "sampling_dataset_id": _sampling_dataset_id(provider.identity_payload())}
        amendment = validate_output_policy_amendment(
            saved["experiment_identity"], sealed, output_policy_amendment,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_epoch=int(saved["epoch"]),
            stop_after=args.stop_after, run_directory=str(args.output.resolve()),
        )
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer, _ = engine._make_optimizer(model, provider, recipe["architecture"],
                                            int(saved["epoch"]) + 1, saved["optimizer_state_by_name"],
                                            group_specs=specs)
        restored = named_optimizer_state(model, optimizer)
        def equal_tree(left, right):
            if torch.is_tensor(left):
                return torch.equal(left.cpu(), right.cpu())
            if isinstance(left, dict):
                return left.keys() == right.keys() and all(equal_tree(value, right[name]) for name, value in left.items())
            if isinstance(left, (list, tuple)):
                return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
            return left == right
        if not equal_tree(saved["optimizer_state_by_name"], restored):
            raise ValueError("Output-only resume did not preserve the exact named optimizer state.")
        summary["output_policy_resume_preflight"] = {
            "source_epoch": saved["epoch"], "source_checkpoint_sha256": output_policy_amendment["source_checkpoint_sha256"],
            "preserved_case_visits": sum(row["case_visits"] for row in saved["history"]),
            "preserved_optimizer_updates": sum(row["optimizer_updates"] for row in saved["history"]),
            "model_state_loaded_strictly": True, "named_optimizer_state_exact": True,
            "amendment": amendment,
        }
    print(json.dumps(summary, sort_keys=True, default=str, allow_nan=False), flush=True)
    if args.dry_run:
        return 0
    fit_options = {"stop_after": args.stop_after, "resume_checkpoint": args.resume}
    fit_options.update(source_child_fit_options)
    if output_policy_amendment is not None:
        fit_options["output_policy_amendment"] = output_policy_amendment
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
