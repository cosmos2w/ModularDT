"""Shared data-binding helpers for manually launched full-TRAIN recipes."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from channelthermal.data.datasets import H5Normalizer, fit_global_normalizer
from channelthermal.data.development_split import read_case_catalog

from honf_runtime.compat import resolve_demo_path
from honf_runtime.run_store import atomic_write_json

FORMAL_TRAIN_SCOPE = "formal_full_train_v1"
NORMALIZATION_ID = "global_h5_original_train_only_v1"


def validate_formal_milestones(checkpointing: Mapping[str, Any], horizon: int) -> None:
    """Allow explicitly bound retention milestones without changing monitoring."""
    milestones = checkpointing["milestone_epochs"]
    interval = checkpointing["monitoring_interval_epochs"]
    if (not isinstance(milestones, list) or not milestones
            or any(type(epoch) is not int or epoch <= 0 or epoch > horizon
                   or epoch % interval for epoch in milestones)
            or milestones != sorted(set(milestones))
            or milestones[0] != interval or milestones[-1] != horizon):
        raise ValueError("Formal milestones must be ordered unique monitoring ages including the first review and endpoint.")


def initialize_formal_output(path: str | Path, *, prepare_only: bool) -> Path:
    """Create a formal run directory without overwriting prepared or saved state."""
    output = Path(path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("Formal output path already exists as a non-directory file.")
    if prepare_only and output.is_dir() and next(output.iterdir(), None) is not None:
        raise ValueError("Formal prepare-only refuses a nonempty output directory; use a fresh path.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def ensure_formal_resume_identity(path: str | Path, filename: str,
                                  expected: Mapping[str, Any]) -> None:
    """Bind a resumed run to its prepared output directory before any writes."""
    if filename not in {"fit_identity.json", "formal_recipe.json"}:
        raise ValueError("Unsupported formal output identity filename.")
    output = Path(path)
    identity_path = output / filename
    # JSON round-trips tuple-valued reader settings as lists. Compare the exact
    # serialized identity while checkpoint-to-checkpoint checks keep their types.
    serialized_expected = json.loads(json.dumps(dict(expected)))
    if identity_path.exists():
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != serialized_expected:
            raise ValueError("Formal resume rejects an output directory with a different run identity.")
        return
    if next(output.iterdir(), None) is not None:
        raise ValueError("Formal resume refuses a nonempty output directory without its prepared identity.")
    atomic_write_json(identity_path, dict(expected))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def stats_sha256(stats: Mapping[str, Any]) -> str:
    """Hash the exact named float32 transform values saved in checkpoints."""
    payload = {name: np.asarray(value, dtype=np.float32).tolist() for name, value in sorted(stats.items())}
    return _canonical_sha256(payload)


def bind_original_train(dataset_path: str | Path, expected_count: int | None = None,
                        dataset_id: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """Resolve exact original TRAIN membership from packed-H5 metadata."""
    source, records = read_case_catalog(dataset_path)
    train_ids = [record["case_id"] for record in records if record["split"] == "train"]
    if not train_ids or len(train_ids) != len(set(train_ids)):
        raise ValueError("Formal full-TRAIN binding requires nonempty unique original TRAIN membership.")
    if expected_count is not None and len(train_ids) != int(expected_count):
        raise ValueError(f"Formal profile expected {int(expected_count)} original TRAIN cases; found {len(train_ids)}.")
    source_metadata = {key: source.get(key) for key in ("dataset_path", "dataset_id", "catalog_checksum", "metadata_sha256", "stat")}
    binding = {
        "schema_version": 1,
        "scope": "all_original_train",
        "training_split": "train",
        "dataset_path": str(resolve_demo_path(dataset_path).resolve()),
        "dataset_id": source.get("dataset_id") or dataset_id,
        "source_metadata": source_metadata,
        "source_metadata_sha256": source["metadata_sha256"],
        "training_case_ids": train_ids,
        "training_case_count": len(train_ids),
        "training_case_ids_sha256": _canonical_sha256(train_ids),
    }
    return binding, train_ids


def bind_formal_validation(dataset_path: str | Path, *, expected_primary_count: int,
                           expected_compatibility_count: int, duplicate_case_id: str) -> tuple[dict[str, Any], list[str], list[str]]:
    """Bind canonical89 and original90 from source split metadata without resampling."""
    source, records = read_case_catalog(dataset_path)
    compatibility_ids = [record["case_id"] for record in records if record["split"] == "test"]
    if len(compatibility_ids) != int(expected_compatibility_count) or duplicate_case_id not in compatibility_ids:
        raise ValueError("Original-test metadata does not match the declared compatibility panel.")
    primary_ids = [case_id for case_id in compatibility_ids if case_id != duplicate_case_id]
    if len(primary_ids) != int(expected_primary_count):
        raise ValueError("Duplicate-excluded canonical validation membership differs from its declared count.")
    binding = {
        "schema_version": 1,
        "source_metadata_sha256": source["metadata_sha256"],
        "primary_scope": "original_test_excluding_train_duplicate",
        "primary_case_ids": primary_ids,
        "primary_case_count": len(primary_ids),
        "primary_case_ids_sha256": _canonical_sha256(primary_ids),
        "compatibility_scope": "original_test_all_rows",
        "compatibility_case_ids": compatibility_ids,
        "compatibility_case_count": len(compatibility_ids),
        "compatibility_case_ids_sha256": _canonical_sha256(compatibility_ids),
        "excluded_training_duplicate_case_id": duplicate_case_id,
    }
    return binding, primary_ids, compatibility_ids


def fit_formal_normalizer(dataset_path: str | Path, dataset_binding: Mapping[str, Any]) -> tuple[H5Normalizer, dict[str, Any]]:
    """Fit native transforms on every bound original TRAIN case only."""
    path = resolve_demo_path(dataset_path).resolve()
    before = path.stat()
    normalizer = fit_global_normalizer(path, dataset_binding["training_case_ids"])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Packed H5 changed while fitting the formal TRAIN normalizer.")
    binding = {
        "identity": NORMALIZATION_ID,
        "fit_function": "channelthermal.data.datasets.fit_global_normalizer",
        "fit_split": "train",
        "training_case_ids_sha256": dataset_binding["training_case_ids_sha256"],
        "training_case_count": dataset_binding["training_case_count"],
        "stats_sha256": stats_sha256(normalizer.stats),
        "stat_names": sorted(normalizer.stats),
    }
    return normalizer, binding


def formal_train_config(base: Mapping[str, Any], dataset_binding: Mapping[str, Any],
                        normalization_binding: Mapping[str, Any],
                        validation_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Copy native config while removing the quarter-data workflow identity."""
    result = json.loads(json.dumps(base))
    dataset = result.setdefault("dataset", {})
    for key in ("development_manifest", "development_manifest_sha256", "development_subset", "development_manifest_path"):
        dataset.pop(key, None)
    dataset.update({
        "train_split": "train",
        "val_split": None,
        "normalization_scope": "all_original_train_only",
        "formal_dataset_binding": dict(dataset_binding),
        "formal_normalization_binding": dict(normalization_binding),
        "formal_validation_binding": dict(validation_binding),
    })
    return result


def validate_formal_bindings(flow_checkpoint: Mapping[str, Any], dataset_binding: Mapping[str, Any],
                             normalization_binding: Mapping[str, Any], validation_binding: Mapping[str, Any],
                             stats: Mapping[str, Any]) -> None:
    """Reject a flow partner trained on another membership or transform."""
    if flow_checkpoint.get("formal_dataset_binding") != dict(dataset_binding):
        raise ValueError("Formal flow partner has a different full-TRAIN dataset binding.")
    if flow_checkpoint.get("formal_normalization_binding") != dict(normalization_binding):
        raise ValueError("Formal flow partner has a different full-TRAIN normalization binding.")
    if flow_checkpoint.get("formal_validation_binding") != dict(validation_binding):
        raise ValueError("Formal flow partner has a different formal validation panel binding.")
    flow_stats = flow_checkpoint.get("global_normalization_stats")
    if not isinstance(flow_stats, Mapping) or stats_sha256(flow_stats) != stats_sha256(stats):
        raise ValueError("Formal flow partner normalization values differ from the full-TRAIN fit.")
    train_config = flow_checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {}) if isinstance(train_config, Mapping) else {}
    if dataset_config.get("formal_dataset_binding") != dict(dataset_binding):
        raise ValueError("Formal flow train_config omits or changes the full-TRAIN data binding.")
    if dataset_config.get("formal_normalization_binding") != dict(normalization_binding):
        raise ValueError("Formal flow train_config omits or changes the full-TRAIN normalization binding.")
    if dataset_config.get("formal_validation_binding") != dict(validation_binding):
        raise ValueError("Formal flow train_config omits or changes the canonical89/original90 validation binding.")
