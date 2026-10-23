#!/usr/bin/env python3
"""Create recipes and run bounded Round5 moment-hyperedge IAG-C continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src", PROJECT / "Case_WindFarm/src", PROJECT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.moment_hyperedge_factory import (
    build_moment_hyperedge_task,
    configure_runtime_root,
    loaded_source_binding,
    moment_hyperedge_attachment_declaration,
)
# Bind the exact attachment/resume guard implementation before the CLI seals
# loaded_source. TrainingEngine imports this helper at branch time otherwise,
# which would make a same-process resume observe a different source inventory.
from honf_runtime.moment_hyperedge_child import (
    prepare_moment_hyperedge_child as _prepare_moment_hyperedge_child,
)
from honf_runtime.moment_hyperedge_protocol import (
    MOMENT_FAMILY,
    MOMENT_PROTOCOL,
    MOMENT_VARIANTS,
    MOMENT_WORKFLOW,
    canonical_sha256,
    make_moment_hyperedge_recipe,
    validate_moment_hyperedge_recipe,
)
from honf_runtime.unified_training import (
    SamplingKey,
    _engine_config_payload,
    _optimizer_spec_payload,
    _sampling_dataset_id,
)

ROUND5_PLAN = PROJECT / "UpgradePlan/ThemalChannel/20261016_Round05_Moment_Hyperedges/20261016_135059Z_HONF_Round5_Scene_Prepared_Moment_Hyperedges_Plan.md"


def _object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return value


def _write_json(path: Path, value: Any) -> None:
    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing recipe/output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _make_recipe(args: argparse.Namespace) -> int:
    parent_path = args.parent_checkpoint.expanduser().resolve(strict=True)
    payload = load_trusted_checkpoint(parent_path, map_location="cpu")
    identity = payload.get("experiment_identity")
    if not isinstance(identity, dict):
        raise ValueError("Parent checkpoint does not carry an experiment identity.")
    recipe = make_moment_hyperedge_recipe(
        identity.get("recipe", {}), parent_path, args.variant,
    )
    value = validate_moment_hyperedge_recipe(recipe)
    if (payload.get("epoch") != 500 or payload.get("arm") != "IAG-C"
            or identity.get("recipe_sha256") != value["source_recipe_sha256"]):
        raise ValueError("Recipe generation accepts only the exact own IAG-C soft e500 parent.")
    _write_json(args.output, value)
    print(json.dumps({
        "status": "recipe_written",
        "recipe": str(args.output.expanduser().resolve()),
        "dataset": value["dataset"],
        "variant": value["variant"],
        "source_checkpoint": value["source_checkpoint"],
        "source_checkpoint_sha256": value["source_checkpoint_sha256"],
        "source_recipe_sha256": value["source_recipe_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


def _run_paths(args: argparse.Namespace, recipe: dict[str, Any]) -> tuple[Path | None, Path]:
    if args.output is None:
        if not args.dry_run or args.resume is not None:
            raise ValueError("Training requires an explicit output directory; resume requires its existing run directory.")
        source = Path(recipe["source_checkpoint"]).expanduser().resolve(strict=True)
        if args.branch_from_checkpoint is not None:
            branch = args.branch_from_checkpoint.expanduser().resolve(strict=True)
            if branch != source:
                raise ValueError("Branch checkpoint differs from the recipe's exact pinned e500 source.")
        return None, source
    output = args.output.expanduser().resolve()
    if args.resume is not None and args.branch_from_checkpoint is not None:
        raise ValueError("Round5 cannot branch and resume in one invocation.")
    if args.resume is not None:
        from honf_runtime.run_layout import resolve_checkpoint

        if not output.is_dir():
            raise ValueError("Exact resume requires the existing Round5 run directory.")
        source = resolve_checkpoint(output, args.resume)
        if source != resolve_checkpoint(output, "latest"):
            raise ValueError("Round5 exact resume requires that run's current latest checkpoint.")
    else:
        source = Path(recipe["source_checkpoint"]).expanduser().resolve(strict=True)
        if args.branch_from_checkpoint is None:
            raise ValueError("A fresh Round5 continuation must name its exact pinned e500 parent.")
        branch = args.branch_from_checkpoint.expanduser().resolve(strict=True)
        if branch != source:
            raise ValueError("Branch checkpoint differs from the recipe's exact own-dataset e500 source.")
        try:
            source.relative_to(output)
        except ValueError:
            pass
        else:
            raise ValueError("Round5 continuation output must be distinct from its source run.")
        if output.exists() and any(output.iterdir()):
            raise ValueError("A fresh Round5 continuation requires a distinct empty output directory.")
    return output, source


def _train(args: argparse.Namespace) -> int:
    recipe = validate_moment_hyperedge_recipe(_object(args.recipe, "Round5 recipe"))
    if not 501 <= args.stop_after <= 2500:
        raise ValueError("Round5 absolute stop must advance beyond parent e500 and remain at or below e2500.")
    if args.dry_run and (args.device != "cpu" or args.yes):
        raise ValueError("Round5 dry-run is CUDA-hidden CPU preparation and cannot accept --yes.")
    if not args.dry_run and (not args.yes or not args.device.startswith("cuda")):
        raise ValueError("A Round5 scientific start requires --yes and the root-authorized CUDA lane.")
    if not args.dry_run and args.output is None:
        raise ValueError("A Round5 scientific start requires a distinct explicit output directory.")
    output, source_path = _run_paths(args, recipe)
    if args.dry_run and args.branch_from_checkpoint is None and args.resume is None:
        # Recipe validation and provider build still establish the exact native contract;
        # no run path or checkpoint is allocated in this preparation-only mode.
        source_path = Path(recipe["source_checkpoint"])

    configure_runtime_root(PROJECT)
    source_payload = load_trusted_checkpoint(source_path, map_location="cpu")
    parent_identity = source_payload.get("experiment_identity")
    if not isinstance(parent_identity, dict):
        raise ValueError("Pinned C-soft e500 parent has no experiment identity.")
    if args.resume is not None:
        resume_payload = load_trusted_checkpoint(source_path, map_location="cpu")
        parent_binding = resume_payload.get("branch_parent")
        if not isinstance(parent_binding, dict) or not isinstance(parent_binding.get("identity"), dict):
            raise ValueError("Round5 resume checkpoint has no original e500 parent identity.")
        parent_identity = parent_binding["identity"]
    model, provider, provider_identity, provider_binding = build_moment_hyperedge_task(
        recipe,
        device=args.device,
        parent_provider_identity=parent_identity.get("provider_identity", {}),
    )
    from inherited_attention_train import engine_for
    from honf_runtime.moment_hyperedge_protocol import native_recipe_for_moment_hyperedges

    native_recipe = native_recipe_for_moment_hyperedges(recipe)
    engine = engine_for(native_recipe, args.device)
    sampling_dataset_id = _sampling_dataset_id(provider_identity)
    if sampling_dataset_id != getattr(provider, "sampling_dataset_id", sampling_dataset_id):
        raise ValueError("Round5 provider sampler namespace differs from its exact native identity.")
    identity: dict[str, Any] = {
        "family": MOMENT_FAMILY,
        "model_family": MOMENT_FAMILY,
        "workflow": MOMENT_WORKFLOW,
        "protocol": MOMENT_PROTOCOL,
        "variant": recipe["variant"],
        "recipe": recipe,
        "recipe_sha256": canonical_sha256(recipe),
        "round5_plan_sha256": _sha256(ROUND5_PLAN),
        "sampling_identity": {
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "sampling_dataset_id": sampling_dataset_id,
            "matches_frozen_native_reference": True,
        },
        "loaded_source": loaded_source_binding(PROJECT),
        "continuation_basis": {
            "family": "inherited_attention_v1",
            "workflow": "inherited_attention_joint_v1",
            "architecture": "IAG-C",
            "source_epoch": 500,
            "source_checkpoint_sha256": recipe["source_checkpoint_sha256"],
            "source_recipe_sha256": recipe["source_recipe_sha256"],
            "common_state": "exact by-name C-soft weights and named AdamW moments",
            "born_parameter_state": "new moments absent until first optimizer update",
        },
        "data_locks": {"WindTEST": "locked", "solver_usage": "326/326", "new_solver_attempts": 0},
    }
    declaration = moment_hyperedge_attachment_declaration(recipe)

    attachment_preview = None
    if args.resume is None:
        case_keys = tuple(provider.epoch_cases(501, engine.config.seed))
        attachment_preview = _prepare_moment_hyperedge_child(
            model,
            source_payload,
            {
                **identity,
                "provider_identity": provider_identity,
                "engine_config": _engine_config_payload(engine.config),
                "selection_policy": asdict(engine.selection),
                "optimizer_schedule_contract": [
                    _optimizer_spec_payload(spec)
                    for spec in provider.optimizer_groups(model, "IAG-C", "joint")
                ],
                "sampling_dataset_id": sampling_dataset_id,
            },
            declaration,
            source_path=source_path,
            case_keys=case_keys,
            stop_after=args.stop_after,
        )
    summary = {
        "status": "prepared_only",
        "family": MOMENT_FAMILY,
        "workflow": MOMENT_WORKFLOW,
        "protocol": MOMENT_PROTOCOL,
        "variant": recipe["variant"],
        "dataset": recipe["dataset"],
        "recipe_sha256": identity["recipe_sha256"],
        "native_provider_binding": provider_binding,
        "provider": provider_identity,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "engine": vars(engine.config),
        "engine_payload": _engine_config_payload(engine.config),
        "requested_absolute_stop": args.stop_after,
        "optimizer_updates": 0,
        "run_allocated": False,
        "device": args.device,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": recipe["source_checkpoint_sha256"],
        "attachment_preview": attachment_preview,
    }
    print(json.dumps(summary, sort_keys=True, default=str, allow_nan=False), flush=True)
    if args.dry_run:
        return 0
    if output is None:
        raise RuntimeError("Round5 training output path was not resolved.")
    fit_options: dict[str, Any] = {"stop_after": args.stop_after, "moment_hyperedge_child": declaration}
    if args.resume is not None:
        fit_options["resume_checkpoint"] = args.resume
    else:
        fit_options["branch_from_checkpoint"] = source_path
    result = engine.fit(
        model,
        provider,
        output,
        identity=identity,
        arm="IAG-C",
        **fit_options,
    )
    print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("make-recipe", help="Derive one C0/M/H recipe from an exact own C-soft e500 parent.")
    make.add_argument("--parent-checkpoint", type=Path, required=True)
    make.add_argument("--variant", choices=sorted(MOMENT_VARIANTS), required=True)
    make.add_argument("--output", type=Path, required=True)
    make.set_defaults(handler=_make_recipe)

    train = subparsers.add_parser("train", help="Prepare, branch, or exact-resume one native Round5 run.")
    train.add_argument("--recipe", type=Path, required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--output", type=Path)
    train.add_argument("--stop-after", type=int, required=True)
    train.add_argument("--branch-from-checkpoint", type=Path)
    train.add_argument("--resume", type=Path, help="Resume the current latest checkpoint in this run.")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--yes", action="store_true")
    train.set_defaults(handler=_train)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
