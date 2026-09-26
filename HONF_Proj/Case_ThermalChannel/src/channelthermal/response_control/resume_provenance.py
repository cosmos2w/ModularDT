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
    if int(replay.get("optimizer_updates", -1)) != 0 or int(replay.get("reference_solves", -1)) != 0:
        raise ValueError("Resume provenance mismatch: source replay must make zero updates and solves.")

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
            raise ValueError("Resume provenance mismatch: train atlas bytes changed since u100.")
        atlas_hashes.append(
            {"path": current_path, "sha256": atlas_digest, "json_sha256": metadata_digest}
        )

    expected_config = {
        arm: training_config_mapping(training_config, arm=arm)
        for arm in ("B_value", "B_response")
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
    for arm in ("B_value", "B_response"):
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
        if int(fit_arm.get("actual_optimizer_updates", -1)) != required_update or int(
            fit_arm.get("final_update", -1)
        ) != required_update:
            raise ValueError(f"Resume provenance mismatch: fit manifest does not end at u{required_update}.")
        _require_equal(
            payload.get("training_config"),
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
        if not isinstance(optimizer.get("param_groups"), list):
            raise TypeError(f"Resume checkpoint optimizer param_groups for {arm} must be a list.")
        for key in (
            "python_rng_state",
            "numpy_rng_state",
            "torch_rng_state",
            "sampler_rng_state",
            "sampler_remaining_order",
        ):
            if key not in payload or payload[key] is None:
                raise ValueError(f"Resume provenance mismatch: {arm} checkpoint lacks {key}.")
        checkpoint_evidence[arm] = {"path": str(path), "sha256": digest}

    value_payload = resume_payloads["B_value"]
    response_payload = resume_payloads["B_response"]
    for key in ("sampler_rng_state", "sampler_remaining_order"):
        if not _same(value_payload.get(key), response_payload.get(key)):
            raise ValueError(f"Resume provenance mismatch: paired RNG/sampler field {key} differs.")
    expected_weights = fit.get("calibrated_response_weights")
    if not isinstance(expected_weights, Mapping):
        raise TypeError("Resume source fit calibrated_response_weights must be an object.")
    _require_equal(
        response_payload.get("calibrated_loss_weights"),
        expected_weights,
        "B_response checkpoint weights differ from the frozen u100 snapshot",
    )

    return {
        "fit_manifest": {"path": str(fit_path), "sha256": file_sha256(fit_path)},
        "read_only_replay_manifest": {
            "path": str(replay_path),
            "sha256": file_sha256(replay_path),
            "optimizer_updates": int(replay["optimizer_updates"]),
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
            for arm in ("B_value", "B_response")
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
            "frozen B_response loss multipliers",
        ],
    }
