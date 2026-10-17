#!/usr/bin/env python3
"""Prepare, fit, resume and inspect the bounded Round6 H readout comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src", PROJECT / "Case_WindFarm/src", PROJECT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from honf_runtime.hyperedge_readout_factory import (
    build_hyperedge_readout_task,
    configure_runtime_root,
    load_hyperedge_readout_components,
    make_hyperedge_readout_engine,
    verify_frozen_state,
)
from honf_runtime.hyperedge_readout_protocol import (
    READOUT_FAMILY,
    READOUT_HEALTH_REVIEW_AGE,
    READOUT_PROTOCOL,
    READOUT_WORKFLOW,
    canonical_sha256,
    make_hyperedge_readout_recipe,
    readout_attachment_declaration,
    validate_hyperedge_readout_recipe,
    verify_frozen_extension_decision,
)
from honf_runtime.unified_training import (
    SamplingKey,
    _pending_clean_stop_request,
    _sampling_dataset_id,
)

ROUND6_PLAN = PROJECT / "UpgradePlan/HONF_Round6_Hyperedge_Readout_Diagnosis_Work_Plan.md"
ROUND6_CLEAN_STOP_REQUEST = "CLEAN_STOP_REQUEST.json"


def _json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> Path:
    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing Round6 recipe: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def _make_recipe(args: argparse.Namespace) -> int:
    gate = None if args.joint_gate_binding is None else _json_object(args.joint_gate_binding, "joint gate binding")
    recipe = make_hyperedge_readout_recipe(
        args.parent_checkpoint,
        receiver_reduction=args.reduction,
        receiver_access=args.access,
        phase=args.phase,
        joint_gate_binding=gate,
    )
    destination = _write_json(args.output, recipe)
    print(json.dumps({
        "status": "recipe_written",
        "recipe": str(destination),
        "dataset": recipe["dataset"],
        "phase": recipe["phase"],
        "reduction": recipe["receiver_reduction"],
        "access": recipe["receiver_access"],
        "source_checkpoint": recipe["source_checkpoint"],
        "source_checkpoint_sha256": recipe["source_checkpoint_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


def _check_run_paths(args: argparse.Namespace, recipe: dict[str, Any]) -> tuple[Path | None, Path | None]:
    if args.resume is not None and args.branch_from_checkpoint is not None:
        raise ValueError("Round6 cannot branch and exact-resume in one invocation.")
    if args.dry_run:
        if args.device != "cpu" or args.yes:
            raise ValueError("Round6 dry-run is CUDA-hidden CPU preparation and cannot accept --yes.")
        if args.output is not None and args.resume is not None:
            raise ValueError("CPU preparation does not write or resume a run directory.")
        if args.branch_from_checkpoint is not None:
            branch = args.branch_from_checkpoint.expanduser().resolve(strict=True)
            if branch != Path(recipe["source_checkpoint"]):
                raise ValueError("Dry-run branch checkpoint differs from the recipe's exact H parent.")
        return None, None
    if args.output is None:
        raise ValueError("A Round6 fit requires an explicit distinct output directory.")
    if not args.yes or not args.device.startswith("cuda"):
        raise ValueError("A Round6 scientific start requires --yes and its root-authorized CUDA lane.")
    output = args.output.expanduser().resolve()
    if args.resume is not None:
        if not output.is_dir():
            raise ValueError("Exact Round6 resume requires its existing run directory.")
        from honf_runtime.run_layout import resolve_checkpoint

        source = resolve_checkpoint(output, args.resume)
        if source != resolve_checkpoint(output, "latest"):
            raise ValueError("Exact Round6 resume must use its current latest checkpoint.")
        return output, source
    source = Path(recipe["source_checkpoint"]).expanduser().resolve(strict=True)
    if recipe["phase"] == "joint_confirmation":
        if args.branch_from_checkpoint is None:
            raise ValueError("A fresh joint confirmation must name its exact pinned H e1000 parent.")
        branch = args.branch_from_checkpoint.expanduser().resolve(strict=True)
        if branch != source:
            raise ValueError("Joint branch checkpoint differs from the exact H e1000 parent in the recipe.")
    elif args.branch_from_checkpoint is not None:
        raise ValueError("Frozen probes use the H e2500 parent binding without a TrainingEngine branch.")
    try:
        source.relative_to(output)
    except ValueError:
        pass
    else:
        raise ValueError("A Round6 fit output must be distinct from the H source run.")
    if output.exists() and any(output.iterdir()):
        existing = {path.name for path in output.iterdir()}
        pending_clean_stop = (
            existing == {ROUND6_CLEAN_STOP_REQUEST}
            and _pending_clean_stop_request(output, ROUND6_CLEAN_STOP_REQUEST) is not None
        )
        if not pending_clean_stop:
            raise ValueError("A fresh Round6 fit requires a distinct empty output directory or its single pending clean-stop request.")
    return output, source if recipe["phase"] == "joint_confirmation" else None


def _fit_identity(recipe: dict[str, Any], task: dict[str, Any], engine: Any) -> dict[str, Any]:
    provider_identity = task["provider_identity"]
    sampling_dataset_id = _sampling_dataset_id(provider_identity)
    if sampling_dataset_id != getattr(task["native_provider"], "sampling_dataset_id", sampling_dataset_id):
        raise ValueError("Round6 provider sampler namespace differs from its exact native identity.")
    return {
        "family": READOUT_FAMILY,
        "model_family": READOUT_FAMILY,
        "workflow": READOUT_WORKFLOW,
        "protocol": READOUT_PROTOCOL,
        "phase": recipe["phase"],
        "receiver_reduction": recipe["receiver_reduction"],
        "receiver_access": recipe["receiver_access"],
        "recipe": recipe,
        "recipe_sha256": canonical_sha256(recipe),
        "round6_plan_sha256": _sha256(ROUND6_PLAN),
        "sampling_identity": {
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "source_sampling_dataset_id": sampling_dataset_id,
            "sampler_epoch_mapping": recipe["sampler_epoch_mapping"],
            "matches_parent_native_namespace": True,
        },
        "sampler_epoch_mapping": recipe["sampler_epoch_mapping"],
        "branch_binding": task["branch_binding"],
        "attachment": task["attachment"],
        "native_provider_binding": task["native_provider_binding"],
        "fit_scope": task["fit_scope"],
        "loaded_source": task["identity"]["loaded_source"],
        "engine_plan": {
            "seed": engine.config.seed,
            "microbatch_cases": engine.config.microbatch_cases,
            "effective_cases": engine.config.effective_cases,
            "sampling_version": engine.config.sampling_version,
            "deterministic_case_order": engine.config.deterministic_case_order,
        },
    }


def _check_round6_training_device_scope(recipe: dict[str, Any], device: str) -> dict[str, Any] | None:
    if not device.startswith("cuda"):
        return None
    from inherited_attention_train import _check_device_scope

    from honf_runtime.inherited_attention_protocol import configure_inherited_attention_numerics

    configure_inherited_attention_numerics(recipe["native_recipe"]["numerical_policy"])
    receipt = _check_device_scope(device, None)
    if receipt is None:
        raise RuntimeError("Round6 CUDA training did not produce an authorized physical-device binding.")
    return {
        "schema": "round6_authorized_gpu_scope_v1",
        "device": device,
        **receipt,
    }


def _prepare_only(args: argparse.Namespace, recipe: dict[str, Any], task: dict[str, Any], engine: Any,
                  identity: dict[str, Any]) -> dict[str, Any]:
    if args.resume is not None:
        raise ValueError("Round6 CPU preparation does not resume or write a run.")
    source = task["source_payload"]
    start_age = (1 if recipe["phase"] == "frozen_probe"
                 else recipe["source_epoch"] + 1)
    ordered, dataset_id = engine.epoch_case_order(task["training_provider"], start_age)
    update_slices = engine.epoch_update_slices(len(ordered))
    if recipe["phase"] == "frozen_probe":
        if args.stop_after != READOUT_HEALTH_REVIEW_AGE or args.extension_decision is not None:
            raise ValueError("Frozen CPU preparation must target the declared fit-age-100 health review without an extension receipt.")
        frozen_audit = verify_frozen_state(
            task["model"], task["frozen_state"], task["trainable_parameter_names"],
        )
        branch_preview = None
        first_sampler_epoch = engine.sampler_epoch(1)
    else:
        if args.stop_after != recipe["fit_age_ceiling"]:
            raise ValueError("Joint CPU preparation must target the absolute e2500 endpoint.")
        if args.extension_decision is not None:
            raise ValueError("A frozen extension decision cannot bind joint confirmation.")
        from honf_runtime.hyperedge_readout_child import prepare_hyperedge_readout_child

        declaration = readout_attachment_declaration(recipe)
        branch_preview = prepare_hyperedge_readout_child(
            task["model"], source, identity | {"provider_identity": task["provider_identity"]},
            declaration, source_path=task["source_path"],
            case_keys=tuple(task["native_provider"].epoch_cases(start_age, engine.config.seed)),
            stop_after=args.stop_after,
        )
        frozen_audit = None
        first_sampler_epoch = engine.sampler_epoch(start_age)
    return {
        "status": "prepared_only",
        "family": READOUT_FAMILY,
        "workflow": READOUT_WORKFLOW,
        "phase": recipe["phase"],
        "dataset": recipe["dataset"],
        "receiver_reduction": recipe["receiver_reduction"],
        "receiver_access": recipe["receiver_access"],
        "recipe_sha256": canonical_sha256(recipe),
        "source_checkpoint": recipe["source_checkpoint"],
        "source_checkpoint_sha256": recipe["source_checkpoint_sha256"],
        "native_provider_binding": task["native_provider_binding"],
        "sampling_dataset_id": dataset_id,
        "fit_start_age": start_age,
        "first_sampler_epoch": first_sampler_epoch,
        "sampler_epoch_mapping": recipe["sampler_epoch_mapping"],
        "case_count": len(ordered),
        "optimizer_updates_per_epoch": len(update_slices),
        "microbatch_cases": engine.config.microbatch_cases,
        "effective_cases": engine.config.effective_cases,
        "model_parameter_count": sum(parameter.numel() for parameter in task["model"].parameters()),
        "trainable_parameter_names": list(task["trainable_parameter_names"]),
        "engine_config": vars(engine.config),
        "frozen_state_verification": frozen_audit,
        "joint_attachment_preview": branch_preview,
        "optimizer_updates": 0,
        "run_allocated": False,
        "device": args.device,
    }


def _train(args: argparse.Namespace) -> int:
    recipe = validate_hyperedge_readout_recipe(_json_object(args.recipe, "Round6 recipe"))
    output, source_path = _check_run_paths(args, recipe)
    if args.dry_run:
        if args.extension_decision is not None:
            raise ValueError("Extension decisions are accepted only on an exact fit-500 resume.")
    elif recipe["phase"] == "frozen_probe":
        if args.stop_after not in (READOUT_HEALTH_REVIEW_AGE, 500, 1000):
            raise ValueError("Frozen fit stop must target the fit-age-100 review, fit age 500, or gated 1000 extension.")
        if args.stop_after == 500 and args.resume is None:
            raise ValueError(
                "The frozen primary horizon requires a completed fit-age-100 health review; "
                "a fresh fit must stop at the declared fit-age-100 health review."
            )
        if args.stop_after == READOUT_HEALTH_REVIEW_AGE and args.resume is not None:
            from honf_runtime.compat import load_trusted_checkpoint

            saved = load_trusted_checkpoint(args.resume.expanduser().resolve(strict=True), map_location="cpu")
            if int(saved.get("epoch", -1)) >= READOUT_HEALTH_REVIEW_AGE:
                raise ValueError("Fit-age-100 health review stop is only valid for a fresh or pre-100 interrupted fit.")
        if args.stop_after == 1000 and (args.resume is None or args.extension_decision is None):
            raise ValueError("Fit 1000 requires an exact fit-500 resume and a TRAIN extension decision receipt.")
        if args.stop_after != 1000 and args.extension_decision is not None:
            raise ValueError("An extension decision cannot be used before the fit-500 review.")
    elif args.stop_after != 2500:
        raise ValueError("Joint confirmation has one sealed absolute stop at e2500.")
    if args.extension_decision is not None:
        verify_frozen_extension_decision(args.extension_decision, recipe=recipe)
    if args.dry_run and args.device != "cpu":
        raise ValueError("Round6 dry-run must hide CUDA and use CPU.")
    configure_runtime_root(PROJECT)
    device_scope = _check_round6_training_device_scope(recipe, args.device)
    task = build_hyperedge_readout_task(recipe, device=args.device, runtime_root=PROJECT)
    engine = make_hyperedge_readout_engine(recipe, device=args.device)
    identity = _fit_identity(recipe, task, engine)
    if device_scope is not None:
        identity["device_scope"] = device_scope
    summary = _prepare_only(args, recipe, task, engine, identity) if args.dry_run else {
        "status": "fit_ready",
        "family": READOUT_FAMILY,
        "workflow": READOUT_WORKFLOW,
        "phase": recipe["phase"],
        "dataset": recipe["dataset"],
        "reduction": recipe["receiver_reduction"],
        "access": recipe["receiver_access"],
        "source_checkpoint": recipe["source_checkpoint"],
        "source_checkpoint_sha256": recipe["source_checkpoint_sha256"],
        "output": str(output),
        "stop_after": args.stop_after,
        "device": args.device,
        "device_scope": device_scope,
        "trainable_parameter_names": list(task["trainable_parameter_names"]),
        "optimizer_state_policy": task["fit_scope"]["optimizer_state"],
    }
    print(json.dumps(summary, sort_keys=True, default=str, allow_nan=False), flush=True)
    if args.dry_run:
        return 0
    if output is None:
        raise RuntimeError("Round6 training output path was not resolved.")
    fit_options: dict[str, Any] = {"stop_after": args.stop_after}
    if args.resume is not None:
        fit_options["resume_checkpoint"] = args.resume
    elif recipe["phase"] == "joint_confirmation":
        if source_path is None:
            raise RuntimeError("Joint branch source path was not resolved.")
        fit_options["branch_from_checkpoint"] = source_path
        fit_options["hyperedge_readout_child"] = readout_attachment_declaration(recipe)
    if args.extension_decision is not None:
        fit_options["hyperedge_readout_extension"] = str(args.extension_decision.expanduser().resolve(strict=True))
    result = engine.fit(
        task["model"], task["training_provider"], output,
        identity=identity, arm="IAG-C", **fit_options,
    )
    print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


def _inspect(args: argparse.Namespace) -> int:
    if args.device != "cpu":
        raise ValueError("Round6 CLI inspection is CPU-only; GPU callers must enforce the authorized UUID scope through the API.")
    loaded = load_hyperedge_readout_components(
        args.checkpoint, runtime_root=PROJECT, device=args.device,
    )
    model = loaded["model"]
    print(json.dumps({
        "status": "loaded_without_provider",
        "checkpoint": loaded["checkpoint"],
        "dataset": loaded["recipe"]["dataset"],
        "phase": loaded["recipe"]["phase"],
        "receiver_reduction": loaded["recipe"]["receiver_reduction"],
        "receiver_access": loaded["recipe"]["receiver_access"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "native_adapter_type": type(loaded["native_adapter"]).__name__,
    }, sort_keys=True, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-recipe", help="Pin a reduction/access mode to exact native H parent weights.")
    make.add_argument("--parent-checkpoint", type=Path, required=True)
    make.add_argument("--phase", choices=("frozen_probe", "joint_confirmation"), default="frozen_probe")
    make.add_argument("--reduction", choices=("pool_first", "edge_first"), required=True)
    make.add_argument("--access", choices=("local_cover", "full_bank_diagnostic"), default="local_cover")
    make.add_argument("--joint-gate-binding", type=Path)
    make.add_argument("--output", type=Path, required=True)
    make.set_defaults(handler=_make_recipe)

    train = subparsers.add_parser("train", help="Dry-run, start or exact-resume one native Round6 fit.")
    train.add_argument("--recipe", type=Path, required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--output", type=Path)
    train.add_argument("--stop-after", type=int, required=True)
    train.add_argument("--branch-from-checkpoint", type=Path)
    train.add_argument("--resume", type=Path, help="Resume the current latest checkpoint in this run.")
    train.add_argument("--extension-decision", type=Path)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--yes", action="store_true")
    train.set_defaults(handler=_train)

    inspect = subparsers.add_parser("inspect-checkpoint", help="Load one model and adapter without building a provider.")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.add_argument("--device", default="cpu")
    inspect.set_defaults(handler=_inspect)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
