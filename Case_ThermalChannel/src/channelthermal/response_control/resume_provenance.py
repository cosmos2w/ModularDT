"""Strict provenance checks for paired response-control checkpoint resumes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_config_mapping(config: Any, *, arm: str) -> dict[str, Any]:
    """Serialize the fields that must match a saved optimizer/sampler resume."""

    return {
        "arm": arm,
        "max_optimizer_updates": int(config.max_optimizer_updates),
        "max_epochs": int(config.max_epochs),
        "total_optimizer_update_ceiling": int(config.total_optimizer_update_ceiling),
        "checkpoint_every_updates": int(config.checkpoint_every_updates),
        "max_wall_seconds": config.max_wall_seconds,
        "review_updates": list(config.review_updates),
        "random_seed": int(config.random_seed),
        "deterministic_eval_mode": bool(config.deterministic_eval_mode),
        "deterministic_algorithms": bool(config.deterministic_algorithms),
        "project_response_gradient_blockwise": bool(config.project_response_gradient_blockwise),
        "response_ramp_start_update": config.response_ramp_start_update,
        "response_ramp_end_update": config.response_ramp_end_update,
        "response_ramp_terms": list(config.response_ramp_terms),
        "required_response_terms": list(config.required_response_terms),
        "required_control_terms": list(config.required_control_terms),
        "include_feasibility_bce": bool(config.include_feasibility_bce),
        "stages": [
            {
                "name": stage.name,
                "start_update": int(stage.start_update),
                "stop_update": int(stage.stop_update),
                "active_terms": list(stage.active_terms),
            }
            for stage in config.stages
        ],
    }


def _canonical(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(value) for value in item]
        if isinstance(item, Path):
            return str(item.expanduser().resolve())
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(
            left.detach().cpu(), right.detach().cpu()
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(
            left, right
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_same(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_same(a, b) for a, b in zip(left, right, strict=True))
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _read_manifest(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object.")
    return source, payload


def _require_equal(actual: Any, expected: Any, reason: str) -> None:
    if _canonical(actual) != _canonical(expected):
        raise ValueError(f"Resume provenance mismatch: {reason}.")


def _validate_fit_arm_gate_accounting(
    fit_arm: Mapping[str, Any], *, arm: str, required_update: int
) -> None:
    """Bind a resume checkpoint to the cumulative gate across fit segments."""

    initial_update = int(fit_arm.get("initial_update", -1))
    segment_updates = int(fit_arm.get("actual_optimizer_updates", -1))
    segment_attempts = int(fit_arm.get("attempted_optimizer_steps", -1))
    cumulative_attempts = int(fit_arm.get("total_attempted_optimizer_steps", -1))
    final_update = int(fit_arm.get("final_update", -1))
    if (
        initial_update < 0
        or segment_updates < 0
        or initial_update + segment_updates != required_update
        or final_update != required_update
    ):
        raise ValueError(f"Resume provenance mismatch: fit manifest does not end at u{required_update}.")
    if segment_attempts != segment_updates or cumulative_attempts != required_update:
        raise ValueError(
            f"Resume provenance mismatch: {arm} attempt accounting differs from its completed updates."
        )


def validate_paired_resume_provenance(
    *,
    fit_manifest_path: str | Path,
    replay_manifest_path: str | Path,
    source_checkpoint_path: str | Path,
    train_atlas_paths: Sequence[str | Path],
    development_atlas_paths: Sequence[str | Path],
    train_family_ids: Sequence[str],
    train_sampling: Sequence[Mapping[str, Any]],
    frozen_scales_path: str | Path,
    loss_scales: Mapping[str, Any],
    refit_config: Mapping[str, Any],
    native_trainable_scope: Mapping[str, Any] | None = None,
    native_parameter_inventory: Mapping[str, Any] | None = None,
    historical_case_order: Sequence[str] = (),
    historical_dataset_path: str | Path | None = None,
    resume_checkpoint_paths: Mapping[str, str | Path],
    resume_payloads: Mapping[str, Mapping[str, Any]],
    training_config: Any,
    required_update: int = 100,
) -> dict[str, Any]:
    """Validate an exact paired gate before resuming either arm.

    The fit manifest binds the source model, training panel, sampling summary,
    calibration, schedule, and response weights. Its paired read-only replay
    manifest binds the current atlas bytes and the exact resumable checkpoint
    bytes. Checkpoint payloads then carry the optimizer, RNG, and sampler state.
    """

    fit_path, fit = _read_manifest(fit_manifest_path, label="Resume fit manifest")
    replay_path, replay = _read_manifest(replay_manifest_path, label="Resume replay manifest")
    if fit.get("status") != "passed" or fit.get("mode") != "paired_staged_fit":
        raise ValueError("Resume provenance mismatch: source fit manifest is not a passed paired fit.")
    if replay.get("status") != "passed" or replay.get("mode") != "read_only_full_grid_train_replay":
        raise ValueError("Resume provenance mismatch: checkpoint replay manifest is not passed/read-only.")
    _require_equal(
        replay.get("refit_config"),
        fit.get("refit_config"),
        "read-only replay refit configuration differs from the paired fit",
    )
    replay_activity_counters = (
        "optimizer_instances_created",
        "optimizer_calls",
        "optimizer_updates",
        "reference_solver_calls",
        "reference_solves",
    )
    for counter in replay_activity_counters:
        if int(replay.get(counter, -1)) != 0:
            raise ValueError(
                "Resume provenance mismatch: source replay must be strictly read-only; "
                f"{counter} must be zero."
            )

    source_checkpoint = Path(source_checkpoint_path).expanduser().resolve()
    source_digest = file_sha256(source_checkpoint)
    for manifest, path_key, hash_key in (
        (fit, "checkpoint", "checkpoint_sha256"),
        (replay, "source_checkpoint", "source_checkpoint_sha256"),
    ):
        if Path(manifest.get(path_key, "")).expanduser().resolve() != source_checkpoint:
            raise ValueError("Resume provenance mismatch: source checkpoint path differs.")
        if manifest.get(hash_key) != source_digest:
            raise ValueError("Resume provenance mismatch: source checkpoint SHA-256 differs.")

    train_paths = [str(Path(path).expanduser().resolve()) for path in train_atlas_paths]
    development_paths = [str(Path(path).expanduser().resolve()) for path in development_atlas_paths]
    _require_equal(fit.get("train_atlas_paths"), train_paths, "train atlas path/order changed")
    _require_equal(fit.get("train_family_ids"), list(train_family_ids), "train family IDs changed")
    _require_equal(fit.get("train_sampling"), list(train_sampling), "sampled training panel changed")
    _require_equal(fit.get("development_paths"), development_paths, "development panel changed")
    _require_equal(fit.get("refit_config"), dict(refit_config), "full-access refit configuration changed")

    saved_scales = fit.get("loss_scales_source")
    if not isinstance(saved_scales, Mapping):
        raise TypeError("Resume source fit loss_scales_source must be an object.")
    scales_path = Path(frozen_scales_path).expanduser().resolve()
    scales_digest = file_sha256(scales_path)
    if Path(saved_scales.get("path", "")).expanduser().resolve() != scales_path:
        raise ValueError("Resume provenance mismatch: frozen scale source path differs.")
    if saved_scales.get("sha256") != scales_digest:
        raise ValueError("Resume provenance mismatch: frozen scale source SHA-256 differs.")
    _require_equal(fit.get("loss_scales"), loss_scales, "frozen loss-scale snapshot changed")
    if fit.get("initialization_mode") == "native_checkpoint":
        if native_trainable_scope is None or native_parameter_inventory is None:
            raise ValueError("Resume provenance mismatch: native scope inventory is required.")
        _require_equal(fit.get("native_trainable_scope"), native_trainable_scope, "native trainable scope changed")
        saved_inventory = fit.get("parameter_inventory") or {}
        for key in ("trainable_parameter_names", "trainable_parameter_shapes"):
            _require_equal(
                saved_inventory.get(key), native_parameter_inventory.get(key),
                f"native {key} changed",
            )
        saved_replay = fit.get("historical_value_replay") or {}
        if historical_dataset_path is None:
            raise ValueError("Resume provenance mismatch: historical train dataset is required.")
        current_dataset = Path(historical_dataset_path).expanduser().resolve()
        if Path(saved_replay.get("dataset", "")).expanduser().resolve() != current_dataset:
            raise ValueError("Resume provenance mismatch: historical train dataset path changed.")
        current_stat = current_dataset.stat()
        for key, value in (
            ("dataset_size_bytes", current_stat.st_size),
            ("dataset_mtime_ns", current_stat.st_mtime_ns),
        ):
            _require_equal(saved_replay.get(key), value, f"historical train dataset {key} changed")
        _require_equal(
            saved_replay.get("train_case_order"), list(historical_case_order),
            "historical train cohort/order changed",
        )

    replay_atlases = replay.get("train_stencils")
    if not isinstance(replay_atlases, list):
        raise TypeError("Resume replay manifest train_stencils must be a list.")
    if len(replay_atlases) != len(train_paths):
        raise ValueError("Resume provenance mismatch: replay atlas inventory is incomplete.")
    atlas_hashes: list[dict[str, str]] = []
    for current_path, entry in zip(train_paths, replay_atlases, strict=True):
        if not isinstance(entry, Mapping):
            raise TypeError("Resume replay atlas entries must be objects.")
        if Path(entry.get("path", "")).expanduser().resolve() != Path(current_path):
            raise ValueError("Resume provenance mismatch: replay atlas path/order differs.")
        atlas_path = Path(current_path)
        metadata_path = atlas_path.with_suffix(".json")
        atlas_digest = file_sha256(atlas_path)
        metadata_digest = file_sha256(metadata_path)
        if entry.get("sha256") != atlas_digest or entry.get("json_sha256") != metadata_digest:
            raise ValueError(f"Resume provenance mismatch: train atlas bytes changed since u{required_update}.")
        atlas_hashes.append(
            {"path": current_path, "sha256": atlas_digest, "json_sha256": metadata_digest}
        )

    response_arm = str(training_config.arm)
    if not response_arm.endswith("_response"):
        raise ValueError("Resume provenance requires a paired response-arm training config.")
    value_arm = f"{response_arm[0]}_value"
    expected_config = {
        arm: training_config_mapping(training_config, arm=arm)
        for arm in (value_arm, response_arm)
    }
    if set(resume_payloads) != set(expected_config) or set(resume_checkpoint_paths) != set(
        expected_config
    ):
        raise ValueError("Resume provenance mismatch: both paired arms are required.")
    replay_checkpoints = replay.get("arm_checkpoints")
    if not isinstance(replay_checkpoints, Mapping):
        raise TypeError("Resume replay manifest arm_checkpoints must be an object.")
    fit_arms = fit.get("arms")
    if not isinstance(fit_arms, Mapping):
        raise TypeError("Resume source fit manifest arms must be an object.")
    fit_checkpoint_paths = fit.get("checkpoint_paths")
    if not isinstance(fit_checkpoint_paths, Mapping):
        raise TypeError("Resume source fit checkpoint_paths must be an object.")
    if int(fit.get("review_cap", -1)) != required_update:
        raise ValueError("Resume provenance mismatch: source review gate is not the required update.")
    if fit.get("review_decisions", {}).get(str(required_update)) != "stop":
        raise ValueError("Resume provenance mismatch: source checkpoint was not stopped at its review gate.")

    checkpoint_evidence: dict[str, dict[str, Any]] = {}
    for arm in (value_arm, response_arm):
        path = Path(resume_checkpoint_paths[arm]).expanduser().resolve()
        digest = file_sha256(path)
        saved_checkpoint = replay_checkpoints.get(arm)
        if not isinstance(saved_checkpoint, Mapping):
            raise TypeError(f"Resume replay checkpoint for {arm} must be an object.")
        if Path(saved_checkpoint.get("path", "")).expanduser().resolve() != path:
            raise ValueError(f"Resume provenance mismatch: {arm} checkpoint path differs.")
        if saved_checkpoint.get("sha256") != digest:
            raise ValueError(f"Resume provenance mismatch: {arm} checkpoint SHA-256 differs.")
        arm_checkpoint_list = fit_checkpoint_paths.get(arm)
        if not isinstance(arm_checkpoint_list, list):
            raise TypeError(f"Resume source fit checkpoint path list for {arm} must be a list.")
        normalized_paths = [str(Path(saved).expanduser().resolve()) for saved in arm_checkpoint_list]
        if str(path) not in normalized_paths:
            raise ValueError(f"Resume provenance mismatch: {arm} path is absent from the source fit manifest.")
        payload = resume_payloads[arm]
        fit_arm = fit_arms.get(arm)
        if not isinstance(fit_arm, Mapping):
            raise TypeError(f"Resume source fit metadata for {arm} must be an object.")
        if payload.get("arm") != arm or int(payload.get("actual_optimizer_updates", -1)) != required_update:
            raise ValueError(f"Resume provenance mismatch: {arm} checkpoint update/arm differs.")
        if int(payload.get("attempted_optimizer_steps", -1)) != int(
            fit_arm.get("total_attempted_optimizer_steps", -2)
        ):
            raise ValueError(f"Resume provenance mismatch: {arm} attempted-step count differs.")
        _validate_fit_arm_gate_accounting(fit_arm, arm=arm, required_update=required_update)
        saved_training_config = dict(payload.get("training_config") or {})
        if arm.startswith("B_"):
            # Older B checkpoints omit additive response-control metadata.
            # Interpret those omissions as the unchanged historical defaults.
            for key, legacy_default in (
                ("deterministic_eval_mode", False),
                ("deterministic_algorithms", False),
                ("project_response_gradient_blockwise", False),
                ("response_ramp_start_update", None),
                ("response_ramp_end_update", None),
                ("response_ramp_terms", []),
                ("required_response_terms", []),
                ("required_control_terms", []),
                ("include_feasibility_bce", True),
            ):
                saved_training_config.setdefault(key, legacy_default)
        _require_equal(
            saved_training_config,
            expected_config[arm],
            f"{arm} checkpoint training schedule changed",
        )
        if not isinstance(payload.get("model"), Mapping):
            raise TypeError(f"Resume checkpoint model state for {arm} must be an object.")
        if not payload["model"]:
            raise ValueError(f"Resume provenance mismatch: {arm} checkpoint has an empty model state.")
        optimizer = payload.get("optimizer")
        if not isinstance(optimizer, Mapping):
            raise TypeError(f"Resume checkpoint optimizer state for {arm} must be an object.")
        if not isinstance(optimizer.get("state"), Mapping):
            raise TypeError(f"Resume checkpoint optimizer state entries for {arm} must be an object.")
        if not isinstance(optimizer.get("param_groups"), list):
            raise TypeError(f"Resume checkpoint optimizer param_groups for {arm} must be a list.")
        for key in (
            "python_rng_state",
            "numpy_rng_state",
            "torch_rng_state",
            "sampler_rng_state",
            "sampler_remaining_order",
            *(("historical_case_order", "historical_next_index") if fit.get("initialization_mode") == "native_checkpoint" else ()),
        ):
            if key not in payload or payload[key] is None:
                raise ValueError(f"Resume provenance mismatch: {arm} checkpoint lacks {key}.")
        if fit.get("initialization_mode") == "native_checkpoint" and arm.startswith("R_"):
            cuda_rng = payload.get("cuda_rng_state_by_model_device")
            if not isinstance(cuda_rng, Mapping) or not cuda_rng:
                raise ValueError(
                    f"Resume provenance mismatch: {arm} checkpoint lacks device-specific CUDA RNG state."
                )
        checkpoint_evidence[arm] = {"path": str(path), "sha256": digest}

    value_payload = resume_payloads[value_arm]
    response_payload = resume_payloads[response_arm]
    matched_sampler_fields = ["sampler_rng_state", "sampler_remaining_order"]
    if fit.get("initialization_mode") == "native_checkpoint":
        matched_sampler_fields.extend(("historical_case_order", "historical_next_index"))
    for key in matched_sampler_fields:
        if not _same(value_payload.get(key), response_payload.get(key)):
            raise ValueError(f"Resume provenance mismatch: paired RNG/sampler field {key} differs.")
    expected_weights = fit.get("calibrated_response_weights")
    if not isinstance(expected_weights, Mapping):
        raise TypeError("Resume source fit calibrated_response_weights must be an object.")
    _require_equal(
        response_payload.get("calibrated_loss_weights"),
        expected_weights,
        f"{response_arm} checkpoint weights differ from the frozen u{required_update} snapshot",
    )

    return {
        "fit_manifest": {"path": str(fit_path), "sha256": file_sha256(fit_path)},
        "read_only_replay_manifest": {
            "path": str(replay_path),
            "sha256": file_sha256(replay_path),
            "optimizer_instances_created": int(replay["optimizer_instances_created"]),
            "optimizer_calls": int(replay["optimizer_calls"]),
            "optimizer_updates": int(replay["optimizer_updates"]),
            "reference_solver_calls": int(replay["reference_solver_calls"]),
            "reference_solves": int(replay["reference_solves"]),
        },
        "source_checkpoint": {"path": str(source_checkpoint), "sha256": source_digest},
        "resume_gate_update": required_update,
        "resume_checkpoints": checkpoint_evidence,
        "train_atlas_hashes": atlas_hashes,
        "frozen_loss_scales": {
            "path": str(scales_path),
            "sha256": scales_digest,
            "snapshot": dict(loss_scales),
        },
        "calibrated_response_weights": dict(expected_weights),
        "source_code_identity_attested": False,
        "source_code_identity_caveat": (
            "The source fit manifest has no launch-time Git, worktree, or source-file identity; "
            "checkpoint, recipe, data, scales, and replay checks do not retroactively attest "
            "the exact source revision used for that fit."
        ),
        "per_arm_rng_state_fields": {
            arm: {
                "python_rng_state": True,
                "numpy_rng_state": True,
                "torch_rng_state": True,
                "cuda_rng_state_by_model_device": resume_payloads[arm].get(
                    "cuda_rng_state_by_model_device"
                )
                is not None,
                "sampler_rng_state": True,
                "sampler_remaining_order": True,
            }
            for arm in (value_arm, response_arm)
        },
        "validated_contracts": [
            "source checkpoint identity",
            "train/development panel and family identity",
            "sampled receiver contract",
            "train atlas and metadata bytes",
            "frozen loss-scale source and numeric snapshot",
            "full-access refit configuration",
            "u100 arm checkpoint bytes and optimizer state",
            "training schedule, per-arm RNG state, and shared sampler continuity",
            f"frozen {response_arm} loss multipliers",
        ],
    }


def validate_checkpoint_only_review_provenance(
    *,
    failed_manifest_path: str | Path,
    source_checkpoint_path: str | Path,
    recipe_path: str | Path,
    r0_diagnostic_path: str | Path,
    frozen_scales_path: str | Path,
    train_atlas_paths: Sequence[str | Path],
    development_atlas_paths: Sequence[str | Path],
    arm_checkpoint_paths: Mapping[str, str | Path],
    arm_payloads: Mapping[str, Mapping[str, Any]],
    training_config: Any,
    review_update: int = 200,
) -> dict[str, Any]:
    """Attest saved u200 pair inputs for evaluation without optimizer updates."""

    if review_update != 200:
        raise ValueError("This checkpoint-only R1 review entrypoint is restricted to u200.")
    expected_arms = {"R_value", "R_response"}
    if set(arm_checkpoint_paths) != expected_arms or set(arm_payloads) != expected_arms:
        raise ValueError("Checkpoint-only review requires exactly R_value and R_response checkpoints.")
    if len(train_atlas_paths) != 8 or len(development_atlas_paths) != 4:
        raise ValueError("Checkpoint-only R1 review requires eight train and four Re90 development atlases.")

    manifest_path, manifest = _read_manifest(failed_manifest_path, label="Source fit manifest")
    if manifest.get("status") != "failed" or manifest.get("mode") != "paired":
        raise ValueError("Checkpoint-only review requires the preserved failed paired-run manifest.")
    if manifest.get("error_type") != "ValueError" or "quadrature weights differ" not in str(manifest.get("error", "")):
        raise ValueError("Source manifest does not identify the known fixed-heat quadrature evaluation failure.")
    if manifest.get("initialization_mode") != "native_checkpoint":
        raise ValueError("Checkpoint-only R1 review requires native-checkpoint initialization.")

    source_path = Path(source_checkpoint_path).expanduser().resolve()
    recipe = Path(recipe_path).expanduser().resolve()
    r0_path = Path(r0_diagnostic_path).expanduser().resolve()
    scales_path = Path(frozen_scales_path).expanduser().resolve()
    for path in (source_path, recipe, r0_path, scales_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    _require_equal(manifest.get("checkpoint"), str(source_path), "source checkpoint identity changed")
    _require_equal(manifest.get("recipe_config"), str(recipe), "training recipe path changed")
    _require_equal(manifest.get("r0_diagnostic_json"), str(r0_path), "R0 diagnostic path changed")
    _require_equal(
        [str(Path(path).expanduser().resolve()) for path in manifest.get("train_stencils", ())],
        [str(Path(path).expanduser().resolve()) for path in train_atlas_paths],
        "train atlas order or identity changed",
    )
    recorded_dev = manifest.get("development_stencils", manifest.get("development_paths"))
    if recorded_dev is not None:
        _require_equal(
            [str(Path(path).expanduser().resolve()) for path in recorded_dev],
            [str(Path(path).expanduser().resolve()) for path in development_atlas_paths],
            "development atlas order or identity changed",
        )

    source_digest = file_sha256(source_path)
    recipe_digest = file_sha256(recipe)
    r0_digest = file_sha256(r0_path)
    scales_digest = file_sha256(scales_path)
    scales_payload = json.loads(scales_path.read_text(encoding="utf-8"))
    frozen_scales = scales_payload.get("frozen_scales")
    if not isinstance(frozen_scales, Mapping):
        raise TypeError("Frozen scales file has no frozen_scales mapping.")

    atlas_hashes: list[dict[str, str | None]] = []
    for path_text in (*train_atlas_paths, *development_atlas_paths):
        path = Path(path_text).expanduser().resolve()
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(path if not path.is_file() else metadata_path)
        atlas_hashes.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "metadata_sha256": file_sha256(metadata_path),
        })

    config_mappings = {
        arm: training_config_mapping(training_config, arm=arm)
        for arm in ("R_value", "R_response")
    }
    checkpoint_digests: dict[str, dict[str, Any]] = {}
    model_states: dict[str, Mapping[str, Any]] = {}
    active_scopes: dict[str, Mapping[str, Any]] = {}
    for arm in ("R_value", "R_response"):
        path = Path(arm_checkpoint_paths[arm]).expanduser().resolve()
        expected_name = f"response_control_{arm}_training_checkpoint_u00200.pt"
        if path.name != expected_name or path.parent != manifest_path.parent:
            raise ValueError("Review checkpoints must be the exact u200 files in the failed run directory.")
        payload = arm_payloads[arm]
        if payload.get("arm") != arm:
            raise ValueError(f"Checkpoint arm identity differs for {arm}.")
        completed = int(payload.get("actual_optimizer_updates", -1))
        attempted = int(payload.get("attempted_optimizer_steps", -1))
        if completed != review_update or attempted != review_update:
            raise ValueError(f"{arm} must have exactly {review_update} attempted and completed updates.")
        if dict(payload.get("training_config") or {}) != config_mappings[arm]:
            raise ValueError(f"{arm} saved training schedule differs from the declared R1 recipe.")
        state = payload.get("model")
        if not isinstance(state, Mapping):
            raise TypeError(f"{arm} checkpoint model state is not a mapping.")
        model_states[arm] = state
        provenance = payload.get("response_control_calibration_provenance")
        if not isinstance(provenance, Mapping):
            raise TypeError(f"{arm} checkpoint lacks saved calibration provenance.")
        if Path(str(provenance.get("source_checkpoint", ""))).expanduser().resolve() != source_path:
            raise ValueError(f"{arm} saved source checkpoint path differs.")
        if provenance.get("source_checkpoint_sha256") != source_digest:
            raise ValueError(f"{arm} saved source checkpoint digest differs.")
        if Path(str(provenance.get("training_recipe", ""))).expanduser().resolve() != recipe:
            raise ValueError(f"{arm} saved training recipe path differs.")
        if provenance.get("training_recipe_sha256") != recipe_digest:
            raise ValueError(f"{arm} saved training recipe digest differs.")
        scale_source = provenance.get("loss_scales_source")
        if not isinstance(scale_source, Mapping):
            raise TypeError(f"{arm} checkpoint has no frozen-scale provenance.")
        if Path(str(scale_source.get("path", ""))).expanduser().resolve() != scales_path:
            raise ValueError(f"{arm} frozen-scale source path differs.")
        if scale_source.get("sha256") != scales_digest:
            raise ValueError(f"{arm} frozen-scale digest differs.")
        if not _same(provenance.get("loss_scales"), frozen_scales):
            raise ValueError(f"{arm} saved frozen-scale values differ from the source file.")
        audit = provenance.get("frozen_buffer_checkpoint_audit")
        if not isinstance(audit, Mapping) or audit.get("passed") is not True:
            raise ValueError(f"{arm} saved frozen-buffer audit did not pass.")
        scope = provenance.get("active_scope")
        if not isinstance(scope, Mapping) or scope.get("name") != "native_nonlinear_interface":
            raise ValueError(f"{arm} saved active trainable scope is not native_nonlinear_interface.")
        active_scopes[arm] = scope
        weights = payload.get("calibrated_loss_weights")
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError(f"{arm} has no frozen objective weights.")
        if any(not np.isfinite(float(value)) or float(value) <= 0.0 for value in weights.values()):
            raise ValueError(f"{arm} objective weights must be positive and finite.")
        checkpoint_digests[arm] = {"path": str(path), "sha256": file_sha256(path)}

    if set(model_states["R_value"]) != set(model_states["R_response"]):
        raise ValueError("Paired u200 model state names differ.")
    for arm in expected_arms:
        if set(active_scopes[arm].get("trainable_parameter_names", ())) != set(
            active_scopes["R_value"].get("trainable_parameter_names", ())
        ):
            raise ValueError("Paired u200 checkpoints have different trainable scopes.")
    for field in ("sampler_rng_state", "sampler_remaining_order", "historical_case_order", "historical_next_index"):
        if not _same(arm_payloads["R_value"].get(field), arm_payloads["R_response"].get(field)):
            raise ValueError(f"Paired u200 checkpoints differ in {field}.")

    return {
        "status": "passed",
        "mode": "checkpoint_only_review",
        "review_update": review_update,
        "optimizer_calls": 0,
        "optimizer_instances_created": 0,
        "source_failure_manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "error": manifest["error"],
        },
        "source_checkpoint": {"path": str(source_path), "sha256": source_digest},
        "recipe": {"path": str(recipe), "sha256": recipe_digest},
        "r0_diagnostic": {"path": str(r0_path), "sha256": r0_digest},
        "r0_content_hash_attested_by_source_run": False,
        "frozen_scales": {"path": str(scales_path), "sha256": scales_digest},
        "train_and_development_atlas_hashes": atlas_hashes,
        "development_panel_attestation": (
            "matched to source failure manifest"
            if recorded_dev is not None
            else "explicit calibration/Re90 panel; source failure manifest did not persist development paths"
        ),
        "checkpoints": checkpoint_digests,
        "active_scope": {
            arm: list(active_scopes[arm].get("trainable_parameter_names", ()))
            for arm in ("R_value", "R_response")
        },
        "validated_contracts": [
            "known post-fit evaluation failure preserved in source manifest",
            "source checkpoint, recipe, and frozen-scale hashes; R0 path and current bytes are recorded without claiming source-run hash attestation",
            "exact eight train and four development atlas bytes",
            "both exact u200 arm checkpoint files and finite objective weights",
            "saved training schedules and native nonlinear trainable scope",
            "paired sampler and historical-cohort cursors",
            "zero optimizer construction and calls in review mode",
        ],
    }
