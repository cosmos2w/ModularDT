#!/usr/bin/env python3
"""Train the opt-in dataset-native inherited-attention family.

This entry point intentionally has no Round2 or Round3 recipe dispatch. Full
TRAIN1000 runs require both a measured e2500 qualification receipt and a
separate one-GPU resource-budget receipt before native data is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src", PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch

from honf_runtime.inherited_attention_protocol import (
    IAG_AUTHORIZED_PHYSICAL_GPUS,
    IAG_FAMILY,
    IAG_FULL_PROTOCOL,
    IAG_PROTOCOL,
    IAG_Q_ARCHITECTURE,
    configure_inherited_attention_numerics,
    validate_inherited_attention_full_train_authorization,
    validate_inherited_attention_recipe,
    validate_inherited_attention_representation_review,
    validate_inherited_attention_stop,
)
from honf_runtime.unified_training import (
    EngineConfig,
    SamplingKey,
    SelectionPolicy,
    TrainingEngine,
    _engine_config_payload,
    _optimizer_spec_payload,
    _sampling_dataset_id,
)

WORKFLOW = "inherited_attention_joint_v1"
SPARSE_CHILD_SCHEMA = "inherited_attention_sparse_child_v1"
ROUND4_PLAN = PROJECT / "UpgradePlan/HONF_Round4_Attention_Inherited_Gathering_Plan.md"
ALLOWED_STOPS = (100, 500, 1000, 2500)
EXPECTED_SAMPLING_DATASET_IDS = {
    ("thermal", IAG_PROTOCOL): "ThermalChannel:ee96482a86b5c2c5f421121319e7c812f0d0b3998ba830b5fdf2429ef50df111",
    ("wind", IAG_PROTOCOL): "WindFarm:1322db58917ec1d4aea5b64c6d07481541051e879fcc087f0f533b1d545171f0",
    ("thermal", IAG_FULL_PROTOCOL): "ThermalChannel:0fded1b9f3e0d4979ede681685ddd8becc20cfb38eec5dd3d6d59c7f415cfe87",
    ("wind", IAG_FULL_PROTOCOL): "WindFarm:f02a1163f9da6b0e7e9b95888135949bc720a9a1a695291a11717b47cbec521e",
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def loaded_source_binding() -> dict[str, Any]:
    """Bind the actual imported IAG runtime, adapters, and provider code."""
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


def build(recipe: dict[str, Any], *, device: str):
    """Build only the declared Thermal or Wind native IAG provider."""
    recipe = validate_inherited_attention_recipe(recipe)
    configure_inherited_attention_numerics(recipe["numerical_policy"])
    random.seed(int(recipe["seed"]))
    np.random.seed(int(recipe["seed"]))
    torch.manual_seed(int(recipe["seed"]))
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if recipe["dataset"] == "thermal":
        from channelthermal.training.inherited_attention_task import build_thermal_inherited_attention_task_v1

        model, provider, _ = build_thermal_inherited_attention_task_v1(recipe, device=device)
    else:
        from windfarm.training.inherited_attention_task import build_wind_inherited_attention_task_v1

        model, provider = build_wind_inherited_attention_task_v1(recipe, device=device)
    return model, provider


def engine_for(recipe: dict[str, Any], device: str) -> TrainingEngine:
    policy = recipe["execution"]
    config = EngineConfig(
        seed=int(recipe["sampling_seed"]),
        microbatch_cases=int(recipe["microbatch_size"]),
        effective_cases=int(recipe["effective_batch_size"]),
        total_epochs=int(recipe["total_epochs"]),
        monitor_every=int(policy["monitor_every"]),
        checkpoint_epochs=tuple(int(epoch) for epoch in policy["checkpoint_epochs"]),
        latest_every=int(policy["latest_every"]),
        curve_every=int(policy["curve_every"]),
        gradient_clip=1.0,
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        training_mode="joint",
    )
    return TrainingEngine(
        config,
        device=device,
        selection=SelectionPolicy(field_metric=str(policy["field_metric"])),
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return value


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_qualified_checkpoint(qualification: dict[str, Any]) -> None:
    checkpoint = qualification["selected_checkpoint"]
    path = Path(checkpoint["path"]).expanduser().resolve(strict=True)
    if not path.is_file() or _checkpoint_digest(path) != checkpoint["sha256"]:
        raise ValueError("Qualified inherited-attention e2500 checkpoint is missing or differs from its receipt SHA256.")


def _representation_review_binding(
    recipe: dict[str, Any], receipt_path: Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Verify the supplied Q review receipt and produce its compact identity binding."""
    if recipe["architecture"] != IAG_Q_ARCHITECTURE:
        if receipt_path is not None:
            validate_inherited_attention_representation_review(recipe, {})
        return None, None
    if receipt_path is None:
        raise ValueError("IAG-Q requires --representation-review on every CLI invocation.")
    path = receipt_path.expanduser().resolve(strict=True)
    receipt = validate_inherited_attention_representation_review(
        recipe, _read_json_object(path, "IAG-Q representation-review receipt"),
    )
    assert receipt is not None
    for path_key, digest_key in (
        ("source_checkpoint", "source_checkpoint_sha256"),
        ("capacity_diagnosis", "capacity_diagnosis_sha256"),
    ):
        evidence_path = Path(receipt[path_key]).expanduser().resolve(strict=True)
        if not evidence_path.is_file() or _checkpoint_digest(evidence_path) != receipt[digest_key]:
            raise ValueError(f"IAG-Q review evidence {path_key} is missing or differs from its receipt SHA256.")
    identity = {
        "schema": receipt["schema"],
        "path": str(path),
        "sha256": _checkpoint_digest(path),
        "dataset": receipt["dataset"],
        "decision": receipt["decision"],
        "fresh_run_dir": str(Path(receipt["fresh_run_dir"]).expanduser().resolve()),
    }
    return receipt, identity


def _validate_review_output_binding(
    recipe: dict[str, Any], receipt: dict[str, Any] | None, output: Path | None,
) -> None:
    if recipe["architecture"] != IAG_Q_ARCHITECTURE or receipt is None or output is None:
        return
    if recipe["protocol"] == IAG_FULL_PROTOCOL:
        return
    key = (
        "future_sparse_child_run_dir"
        if recipe["normalization"] == "measure_entmax15" else "fresh_run_dir"
    )
    expected = Path(receipt[key]).expanduser().resolve()
    observed = output.expanduser().resolve()
    if observed != expected:
        raise ValueError(f"IAG-Q output must match the review-bound {key}.")


def _verify_cuda_uuid_mapping(expected_uuid: str) -> int:
    """Require the caller's single visible UUID to map to the authorized index."""
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if visible != expected_uuid:
        raise ValueError("Expose exactly the authorized physical GPU UUID through CUDA_VISIBLE_DEVICES.")
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Could not verify current physical GPU index-to-UUID mapping.") from exc
    mapping: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        index, separator, uuid = line.strip().partition(",")
        if separator:
            mapping[index.strip()] = uuid.strip()
    matches = [int(index) for index, uuid in mapping.items() if uuid == expected_uuid]
    if len(matches) != 1 or mapping.get(str(matches[0])) != expected_uuid:
        raise ValueError("Authorized GPU UUID does not map to exactly one current physical device.")
    if matches[0] not in IAG_AUTHORIZED_PHYSICAL_GPUS:
        raise ValueError("Inherited-attention GPU dispatch is limited to the authorized physical GPU1/GPU2 lanes.")
    if IAG_AUTHORIZED_PHYSICAL_GPUS[matches[0]] != expected_uuid:
        raise ValueError("Current GPU index and UUID do not match the frozen Round4 lane binding.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Inherited-attention dispatch requires exactly one visible CUDA device.")
    return matches[0]


def _check_device_scope(device: str, budget: dict[str, Any] | None) -> dict[str, Any] | None:
    if not device.startswith("cuda"):
        return None
    if device != "cuda:0":
        raise ValueError("Round4 CUDA dispatch exposes one physical GPU as logical cuda:0 only.")
    if budget is None:
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
        if visible not in set(IAG_AUTHORIZED_PHYSICAL_GPUS.values()):
            raise ValueError("Round4 development dispatch is limited to physical GPU1/GPU2 UUIDs.")
        physical_index = _verify_cuda_uuid_mapping(visible)
        return {"physical_gpu_index": physical_index, "physical_gpu_uuid": visible}
    physical_index = int(budget["physical_gpu_index"])
    expected_uuid = str(budget["physical_gpu_uuids"][0])
    observed_index = _verify_cuda_uuid_mapping(expected_uuid)
    if observed_index != physical_index:
        raise ValueError("Current device mapping differs from the explicitly approved full-TRAIN budget.")
    return {"physical_gpu_index": observed_index, "physical_gpu_uuid": expected_uuid}


def _validate_sparse_declaration(
    recipe: dict[str, Any], declaration: dict[str, Any] | None, checkpoint: Path | None,
    resume: Path | None, representation_review: dict[str, Any] | None,
) -> dict[str, Any]:
    architecture = recipe["architecture"]
    sparse = recipe["protocol"] == IAG_PROTOCOL and recipe["normalization"] == "measure_entmax15"
    if not sparse:
        if declaration is not None or checkpoint is not None:
            raise ValueError("The e500 child declaration is reserved for segmented sparse development.")
        return {}
    if resume is not None:
        if declaration is not None or checkpoint is not None:
            raise ValueError("A same-run sparse resume cannot attach another e500 parent.")
        return {}
    if declaration is None or checkpoint is None:
        raise ValueError(f"Sparse development {architecture} requires an exact soft e500 parent declaration and checkpoint.")
    base_required = {
        "schema", "source_checkpoint", "source_checkpoint_sha256", "source_recipe_sha256",
        "target_recipe_sha256", "source_epoch",
    }
    required = base_required | (
        {"representation_review_sha256"} if architecture == IAG_Q_ARCHITECTURE else set()
    )
    if set(declaration) != required or declaration.get("schema") != SPARSE_CHILD_SCHEMA:
        raise ValueError(f"Unsupported or incomplete {architecture} sparse-child declaration.")
    path = checkpoint.expanduser().resolve(strict=True)
    if not path.is_file() or declaration.get("source_checkpoint") != str(path):
        raise ValueError("Sparse-child checkpoint must be the exact canonical file named by the declaration.")
    if declaration.get("source_checkpoint_sha256") != _checkpoint_digest(path):
        raise ValueError("Sparse-child parent checkpoint differs from the declared SHA256.")
    if declaration.get("source_epoch") != 500:
        raise ValueError(f"Sparse development {architecture} attaches only at absolute epoch500.")
    if declaration.get("target_recipe_sha256") != canonical_sha256(recipe):
        raise ValueError("Sparse-child declaration targets a different recipe.")
    if (architecture == IAG_Q_ARCHITECTURE
            and (representation_review is None or declaration.get("representation_review_sha256") != representation_review.get("sha256"))):
        raise ValueError("IAG-Q sparse-child declaration must bind the supplied representation-review receipt SHA.")
    return {"branch_from_checkpoint": path, "declaration": declaration}


def _load_parent_payload(
    path: Path, declaration: dict[str, Any], recipe: dict[str, Any],
    representation_review: dict[str, Any] | None,
) -> dict[str, Any]:
    from honf_runtime.compat import load_trusted_checkpoint

    payload = load_trusted_checkpoint(path, map_location="cpu")
    identity = payload.get("experiment_identity", {})
    parent_recipe = identity.get("recipe", {})
    architecture = recipe["architecture"]
    if (payload.get("workflow") != WORKFLOW or payload.get("arm") != architecture
            or payload.get("epoch") != 500 or identity.get("model_family") != IAG_FAMILY
            or parent_recipe.get("dataset") != recipe.get("dataset")
            or parent_recipe.get("architecture") != architecture
            or parent_recipe.get("representation") != recipe.get("representation")
            or parent_recipe.get("protocol") != IAG_PROTOCOL
            or parent_recipe.get("normalization") != "soft"
            or parent_recipe.get("trajectory") != "fresh_soft"):
        raise ValueError(f"Sparse {architecture} parent must be its literal same-architecture soft e500 checkpoint.")
    if (architecture == IAG_Q_ARCHITECTURE
            and (parent_recipe.get("dataset") != "wind" or identity.get("representation_review") != representation_review)):
        raise ValueError("IAG-Q parent must preserve the exact Wind representation-review identity binding.")
    if declaration.get("source_recipe_sha256") != identity.get("recipe_sha256"):
        raise ValueError("Sparse-child declaration does not bind the parent's recipe checksum.")
    return payload


def _validate_run_paths(
    *, output: Path | None, resume: Path | None, branch: Path | None, dry_run: bool,
) -> tuple[Path | None, Path | None]:
    if output is None:
        if not dry_run:
            raise ValueError("A scientific start requires an explicit output directory.")
        if resume is not None:
            raise ValueError("Same-run resume requires its existing output directory, including in dry-run.")
        source = None if branch is None else branch.expanduser().resolve(strict=True)
        return None, source
    output = output.expanduser().resolve()
    source = None
    if resume is not None and branch is not None:
        raise ValueError("An inherited-attention invocation cannot resume and branch at once.")
    if resume is not None:
        from honf_runtime.run_layout import resolve_checkpoint

        if not output.is_dir():
            raise ValueError("Same-run resume requires the existing run output directory.")
        source = resolve_checkpoint(output, resume)
        if source != resolve_checkpoint(output, "latest"):
            raise ValueError("Exact resume requires this run's current latest checkpoint.")
    elif branch is not None:
        source = branch.expanduser().resolve(strict=True)
        try:
            source.relative_to(output)
        except ValueError:
            pass
        else:
            raise ValueError("A sparse child must write outside its soft parent's run directory.")
        if output.exists() and any(output.iterdir()):
            raise ValueError("A sparse child requires a distinct empty output directory.")
    elif not dry_run and output.exists() and any(output.iterdir()):
        raise ValueError("A fresh fit requires an empty new run directory.")
    return output, source


def _preflight_sparse_child(
    model: torch.nn.Module,
    provider: Any,
    engine: TrainingEngine,
    identity: dict[str, Any],
    recipe: dict[str, Any],
    declaration: dict[str, Any],
    parent_path: Path,
    stop_after: int,
) -> dict[str, Any]:
    from honf_runtime.compat import load_trusted_checkpoint
    from honf_runtime.inherited_attention_child import prepare_inherited_attention_sparse_child

    source_hash = _checkpoint_digest(parent_path)
    source_payload = load_trusted_checkpoint(parent_path, map_location="cpu")
    provider_identity = dict(provider.identity_payload())
    architecture = recipe["architecture"]
    groups = tuple(provider.optimizer_groups(model, architecture, "joint"))
    sealed_identity = {
        **identity,
        "provider_identity": provider_identity,
        "engine_config": _engine_config_payload(engine.config),
        "selection_policy": asdict(engine.selection),
        "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in groups],
        "sampling_dataset_id": _sampling_dataset_id(provider_identity),
    }
    if declaration.get("source_checkpoint_sha256") != source_hash:
        raise ValueError("Sparse-child parent SHA changed during preflight.")
    amendment = prepare_inherited_attention_sparse_child(
        model,
        source_payload,
        sealed_identity,
        declaration,
        source_path=parent_path,
        case_keys=tuple(provider.epoch_cases(501, engine.config.seed)),
        stop_after=stop_after,
    )
    return {
        "status": "preflight_passed",
        "source_checkpoint": str(parent_path),
        "source_checkpoint_sha256": source_hash,
        "source_epoch": 500,
        "train_case_count": len(provider.epoch_cases(501, engine.config.seed)),
        "exact_parameter_layout_verified": amendment["exact_parameter_layout_verified"],
        "exact_optimizer_group_contract_verified": amendment["exact_optimizer_group_contract_verified"],
        "exact_optimizer_moments_pending_engine_restore": True,
        "scene_executor_transition": amendment["scene_executor_transition"],
        "target_recipe_sha256": canonical_sha256(recipe),
    }


def _restore_normalization_correction_identity(
    saved_identity: dict[str, Any],
    current_identity: dict[str, Any],
) -> dict[str, Any] | None:
    binding = saved_identity.get("normalization_correction")
    if binding is None:
        return None
    from honf_runtime.inherited_attention_child import (
        validate_inherited_attention_normalization_correction_resume,
    )

    validate_inherited_attention_normalization_correction_resume(binding, current_identity)
    current_identity["normalization_correction"] = binding
    return binding


def _validate_normalization_correction_branch_binding(
    identity: dict[str, Any],
    checkpoint_payload: dict[str, Any],
) -> None:
    binding = identity.get("normalization_correction")
    branch = checkpoint_payload.get("branch_parent")
    child_contract = branch.get("inherited_attention_sparse_child") if isinstance(branch, dict) else None
    observed = child_contract.get("normalization_correction") if isinstance(child_contract, dict) else None
    expected = None
    if binding is not None:
        receipt = json.loads(binding["receipt_json"])
        saved_identity = checkpoint_payload.get("experiment_identity")
        if not isinstance(saved_identity, dict):
            raise ValueError("Corrected sparse checkpoint is missing its sealed child experiment identity.")
        if not isinstance(identity, dict):
            raise TypeError("Current corrected sparse-child identity must be a dictionary.")
        if any(key not in saved_identity or saved_identity[key] != value for key, value in identity.items()):
            raise ValueError("Current sparse-child identity differs from its sealed checkpoint identity.")
        parent_identity = branch.get("identity") if isinstance(branch, dict) else None
        if not isinstance(parent_identity, dict):
            raise ValueError("Corrected sparse checkpoint is missing its sealed soft e500 parent identity.")
        if (receipt.get("source_checkpoint") != branch.get("path")
                or receipt.get("source_checkpoint_sha256") != branch.get("sha256")):
            raise ValueError("Normalization-correction receipt differs from its checkpoint branch parent binding.")
        from honf_runtime.inherited_attention_child import _require_same_soft_identity

        _require_same_soft_identity(parent_identity, saved_identity)
        expected = {
            "scope": receipt["scope"],
            "receipt_sha256": binding["receipt_sha256"],
            "target_loaded_source_sha256": receipt["target_loaded_source_sha256"],
        }
    if observed != expected:
        raise ValueError("Sparse child checkpoint branch contract differs from its normalization-correction identity binding.")


def _preflight_same_run_resume(
    model: torch.nn.Module,
    provider: Any,
    engine: TrainingEngine,
    identity: dict[str, Any],
    path: Path,
    stop_after: int,
) -> dict[str, Any]:
    """Check exact identity, state layout, and named AdamW state without writing."""
    from honf_runtime.compat import load_trusted_checkpoint
    from honf_runtime.unified_training import (
        _resume_identity_amendment,
        named_optimizer_state,
    )

    payload = load_trusted_checkpoint(path, map_location="cpu")
    _validate_normalization_correction_branch_binding(identity, payload)
    architecture = identity["recipe"]["architecture"]
    if payload.get("workflow") != WORKFLOW or payload.get("arm") != architecture:
        raise ValueError("Resume checkpoint belongs to a different workflow or architecture.")
    completed_epoch = payload.get("epoch")
    if type(completed_epoch) is not int or completed_epoch >= stop_after:
        raise ValueError("Exact resume stop must advance the current saved epoch.")
    groups = tuple(provider.optimizer_groups(model, architecture, "joint"))
    provider_identity = dict(provider.identity_payload())
    sealed_identity = {
        **identity,
        "provider_identity": provider_identity,
        "engine_config": _engine_config_payload(engine.config),
        "selection_policy": asdict(engine.selection),
        "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in groups],
        "sampling_dataset_id": _sampling_dataset_id(provider_identity),
    }
    _resume_identity_amendment(
        payload.get("experiment_identity", {}), sealed_identity, allow_microbatch_change=False,
    )
    if (recipe := identity.get("recipe")) and (
        recipe.get("protocol") == IAG_PROTOCOL and recipe.get("normalization") == "measure_entmax15"
    ):
        from honf_runtime.inherited_attention_child import verify_inherited_attention_sparse_child_continuation

        branch = payload.get("branch_parent")
        contract = branch.get("inherited_attention_sparse_child") if isinstance(branch, dict) else None
        verify_inherited_attention_sparse_child_continuation(
            contract, completed_epoch=completed_epoch, stop_after=stop_after,
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer, _ = engine._make_optimizer(
        model, provider, architecture, completed_epoch + 1,
        payload.get("optimizer_state_by_name"), group_specs=groups,
    )
    restored = named_optimizer_state(model, optimizer)
    if canonical_sha256(_tree_jsonable(restored)) != canonical_sha256(_tree_jsonable(payload["optimizer_state_by_name"])):
        raise ValueError("Exact resume failed the named AdamW state round-trip.")
    return {
        "status": "preflight_passed",
        "source_checkpoint": str(path),
        "source_epoch": completed_epoch,
        "strict_model_state_loaded": True,
        "exact_named_optimizer_state_verified": True,
    }


def _tree_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"tensor_dtype": str(value.dtype), "shape": list(value.shape),
                "values": value.detach().cpu().tolist()}
    if isinstance(value, dict):
        return {str(key): _tree_jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tree_jsonable(child) for child in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after", type=int, choices=ALLOWED_STOPS, default=100)
    parser.add_argument("--resume", type=Path, help="Resume this run's exact current latest checkpoint.")
    parser.add_argument("--branch-from-checkpoint", type=Path,
                        help="Exact same-architecture soft epoch500 parent for a segmented sparse-development child.")
    parser.add_argument("--sparse-child-declaration", type=Path,
                        help="Declaration binding the exact soft epoch500 checkpoint and sparse recipe.")
    parser.add_argument("--normalization-correction", type=Path,
                        help="Exact receipt for the audited entmax prefix-boundary numerical correction at e500 attachment.")
    parser.add_argument("--representation-review", type=Path,
                        help="Measured representation-review receipt required by Wind IAG-Q.")
    parser.add_argument("--qualification", type=Path,
                        help="Measured same-architecture e2500 qualification receipt for full_train1000.")
    parser.add_argument("--budget-approval", type=Path,
                        help="Current GPU/time budget receipt bound to the selected qualification checkpoint.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    recipe = validate_inherited_attention_recipe(_read_json_object(args.recipe, "inherited-attention recipe"))
    representation_receipt, representation_identity = _representation_review_binding(
        recipe, args.representation_review,
    )
    configure_inherited_attention_numerics(recipe["numerical_policy"])
    validate_inherited_attention_stop(recipe, args.stop_after)
    if args.dry_run and (args.device != "cpu" or args.yes):
        raise ValueError("Inherited-attention dry-run is CUDA-hidden CPU preparation and does not accept --yes.")
    if not args.dry_run and (not args.yes or args.output is None):
        raise ValueError("A scientific start requires --yes and a distinct explicit output directory.")

    full = recipe["protocol"] == IAG_FULL_PROTOCOL
    if full:
        if args.qualification is None or args.budget_approval is None:
            raise ValueError("Full-TRAIN requires both a qualified e2500 receipt and a separate current budget receipt before data construction.")
        qualification = _read_json_object(args.qualification, "inherited-attention qualification receipt")
        budget_receipt = _read_json_object(args.budget_approval, "full-TRAIN budget receipt")
        qualification, budget = validate_inherited_attention_full_train_authorization(
            recipe, qualification, budget_receipt,
        )
        _validate_qualified_checkpoint(qualification)
    else:
        if args.qualification is not None or args.budget_approval is not None:
            raise ValueError("Full-TRAIN qualification and budget receipts are reserved for full_train1000.")
        qualification, budget = None, None
    if args.dry_run and args.output is None and args.resume is not None:
        raise ValueError("A resume preflight must name its existing output directory.")
    if not args.dry_run and not args.device.startswith("cuda"):
        raise ValueError("Round4 scientific training dispatch requires an explicitly authorized GPU1/GPU2 lane.")
    if args.resume is not None and args.branch_from_checkpoint is not None:
        raise ValueError("Inherited-attention cannot resume and branch in one invocation.")
    if full and args.branch_from_checkpoint is not None:
        raise ValueError("Full-TRAIN always starts fresh; it cannot branch from a development checkpoint.")

    declaration = (
        _read_json_object(args.sparse_child_declaration, "sparse-child declaration")
        if args.sparse_child_declaration is not None else None
    )
    sparse_options = _validate_sparse_declaration(
        recipe, declaration, args.branch_from_checkpoint, args.resume, representation_identity,
    )
    if args.normalization_correction is not None and args.branch_from_checkpoint is None:
        raise ValueError("--normalization-correction is reserved for a fresh sparse e500 child attachment.")
    output, source_path = _validate_run_paths(
        output=args.output, resume=args.resume, branch=sparse_options.get("branch_from_checkpoint"),
        dry_run=args.dry_run,
    )
    _validate_review_output_binding(recipe, representation_receipt, output)
    parent_payload = None
    if sparse_options.get("branch_from_checkpoint") is not None:
        parent_payload = _load_parent_payload(
            sparse_options["branch_from_checkpoint"], declaration, recipe, representation_identity,
        )
    device_receipt = _check_device_scope(args.device, budget)
    # Import the common exact-branch validator on both soft and sparse arms so
    # their sealed loaded-source inventories match across the e500 boundary.
    import honf_runtime.inherited_attention_child  # noqa: F401

    model, provider = build(recipe, device=args.device)
    engine = engine_for(recipe, args.device)
    provider_identity = dict(provider.identity_payload())
    sampling_dataset_id = _sampling_dataset_id(provider_identity)
    expected_sampling_id = EXPECTED_SAMPLING_DATASET_IDS[(recipe["dataset"], recipe["protocol"])]
    if (sampling_dataset_id != expected_sampling_id
            or getattr(provider, "sampling_dataset_id", None) != sampling_dataset_id):
        raise ValueError(
            "case_epoch_v1 namespace differs from the frozen segmented or established full-TRAIN reference."
        )
    plan_digest = _checkpoint_digest(ROUND4_PLAN) if ROUND4_PLAN.is_file() else None
    if plan_digest is None:
        raise FileNotFoundError(f"The frozen Round4 plan is unavailable: {ROUND4_PLAN}")
    identity: dict[str, Any] = {
        "workflow": WORKFLOW,
        "model_family": IAG_FAMILY,
        "recipe": recipe,
        "recipe_sha256": canonical_sha256(recipe),
        "round4_plan_sha256": plan_digest,
        "sampling_identity": {
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "sampling_dataset_id": sampling_dataset_id,
            "matches_frozen_native_reference": True,
        },
        "loaded_source": loaded_source_binding(),
        "data_locks": {"WindTEST": "locked", "solver_usage": "326/326", "new_solver_attempts": 0},
    }
    if representation_identity is not None:
        identity["representation_review"] = representation_identity
    normalization_correction = None
    if args.normalization_correction is not None:
        from honf_runtime.inherited_attention_child import bind_inherited_attention_normalization_correction

        if parent_payload is None or source_path is None:
            raise ValueError("Normalization correction requires the loaded exact soft e500 parent.")
        normalization_correction = bind_inherited_attention_normalization_correction(
            args.normalization_correction,
            parent_payload.get("experiment_identity", {}),
            identity,
            source_path=source_path,
        )
        identity["normalization_correction"] = normalization_correction
    if args.resume is not None:
        from honf_runtime.compat import load_trusted_checkpoint

        resume_payload = load_trusted_checkpoint(source_path, map_location="cpu")
        saved_identity = resume_payload.get("experiment_identity", {})
        _restore_normalization_correction_identity(saved_identity, identity)
        _validate_normalization_correction_branch_binding(identity, resume_payload)
    if qualification is not None and budget is not None:
        identity["full_train1000_authorization"] = {
            "qualification": qualification,
            "budget": budget,
            "launch_gpu": device_receipt,
            "initialization": "fresh weights and empty AdamW state; same-run exact latest resume only",
            "full_train_only_transforms": True,
        }

    summary: dict[str, Any] = {
        "status": "prepared_only",
        "workflow": WORKFLOW,
        "family": IAG_FAMILY,
        "recipe": recipe,
        "provider": provider_identity,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "engine": vars(engine.config),
        "requested_stop": args.stop_after,
        "optimizer_updates": 0,
        "run_allocated": False,
        "device": args.device,
        "device_scope": device_receipt,
        "loaded_source": identity["loaded_source"],
        "WindTEST": "locked",
        "new_solver_attempts": 0,
    }
    if representation_identity is not None:
        summary["representation_review"] = representation_identity
    if normalization_correction is not None:
        summary["normalization_correction"] = {
            "scope": json.loads(normalization_correction["receipt_json"])["scope"],
            "receipt_sha256": normalization_correction["receipt_sha256"],
        }
    elif "normalization_correction" in identity:
        summary["normalization_correction"] = {
            "scope": json.loads(identity["normalization_correction"]["receipt_json"])["scope"],
            "receipt_sha256": identity["normalization_correction"]["receipt_sha256"],
            "restored_from_checkpoint_identity": True,
        }
    if source_path is not None and sparse_options.get("declaration") is not None:
        summary["sparse_child_preflight"] = _preflight_sparse_child(
            model, provider, engine, identity, recipe, declaration,
            source_path, args.stop_after,
        )
    if args.dry_run and args.resume is not None:
        summary["same_run_resume_preflight"] = _preflight_same_run_resume(
            model, provider, engine, identity, source_path, args.stop_after,
        )
    print(json.dumps(summary, sort_keys=True, default=str, allow_nan=False), flush=True)
    if args.dry_run:
        return 0

    fit_options: dict[str, Any] = {"stop_after": args.stop_after}
    if args.resume is not None:
        fit_options["resume_checkpoint"] = args.resume
    if sparse_options.get("declaration") is not None:
        fit_options["branch_from_checkpoint"] = sparse_options["branch_from_checkpoint"]
        fit_options["inherited_attention_sparse_child"] = sparse_options["declaration"]
    result = engine.fit(
        model,
        provider,
        output,
        identity=identity,
        arm=recipe["architecture"],
        **fit_options,
    )
    print(json.dumps(result, sort_keys=True, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
