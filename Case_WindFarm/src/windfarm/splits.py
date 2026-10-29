"""Deterministic group-safe train/validation/test split metadata.

The wind-farm rows are ordered by layout and direction.  All directions of a
physical layout must stay together; random row splits leak geometry across
partitions.  This module implements the documented 70/15/15 split without a
scikit-learn dependency and records the exact groups and row indices.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _partition_counts(group_count: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if group_count < 3:
        raise ValueError("At least three groups are required for train/validation/test splits")
    if len(fractions) != 3 or any(float(value) <= 0 for value in fractions):
        raise ValueError("fractions must contain three positive values")
    total = float(sum(fractions))
    normalized = np.asarray(fractions, dtype=float) / total
    raw = normalized * group_count
    counts = np.floor(raw).astype(int)
    remainder = int(group_count - counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    # Keep all partitions non-empty even for small synthetic fixtures.
    for index in range(3):
        if counts[index] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                raise ValueError("Could not create non-empty group partitions")
            counts[donor] -= 1
            counts[index] += 1
    return tuple(int(value) for value in counts)  # type: ignore[return-value]


def _hash_indices(indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class GroupSplit:
    """Exact row indices and machine-readable metadata for one split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    metadata: dict[str, Any]


def make_group_split(
    groups: np.ndarray,
    *,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> GroupSplit:
    """Create a deterministic, group-disjoint split.

    Groups are shuffled using ``seed``; row indices within each partition are
    sorted in source order for reproducible downstream iteration.  The group
    labels themselves are included in the metadata and no row is dropped.
    """

    values = np.asarray(groups)
    if values.ndim != 1:
        raise ValueError(f"groups must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("groups cannot be empty")
    if not np.all(np.isfinite(values.astype(float))):
        raise ValueError("groups must contain finite values")
    unique = np.unique(values)
    counts = _partition_counts(len(unique), fractions)
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(len(unique))]
    cut_train = counts[0]
    cut_validation = cut_train + counts[1]
    group_sets = {
        "train": shuffled[:cut_train],
        "validation": shuffled[cut_train:cut_validation],
        "test": shuffled[cut_validation:],
    }
    indices = {name: np.flatnonzero(np.isin(values, labels)).astype(np.int64) for name, labels in group_sets.items()}
    memberships = [set(map(int, labels.tolist())) for labels in group_sets.values()]
    if memberships[0] & memberships[1] or memberships[0] & memberships[2] or memberships[1] & memberships[2]:
        raise RuntimeError("Internal group split overlap")
    if np.unique(np.concatenate(tuple(indices.values()))).size != values.size:
        raise RuntimeError("Internal group split dropped or duplicated rows")
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "group_key": "layout_index",
        "seed": int(seed),
        "fractions_requested": [float(value) for value in fractions],
        "groups_total": int(unique.size),
        "rows_total": int(values.size),
        "partitions": {},
    }
    for name in ("train", "validation", "test"):
        rows = indices[name]
        labels = group_sets[name]
        metadata["partitions"][name] = {
            "rows": int(rows.size),
            "groups": int(labels.size),
            "group_values": [
                int(value) if np.issubdtype(values.dtype, np.integer) else float(value) for value in labels
            ],
            "row_indices_sha256": _hash_indices(rows),
        }
    return GroupSplit(
        train=indices["train"],
        validation=indices["validation"],
        test=indices["test"],
        metadata=metadata,
    )


def write_split_outputs(path: str | Path, split: GroupSplit) -> Path:
    """Write exact indices to a compressed NPZ file."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        train=split.train.astype(np.int64),
        validation=split.validation.astype(np.int64),
        test=split.test.astype(np.int64),
    )
    return destination


def split_json(path: str | Path, split: GroupSplit) -> Path:
    """Convenience helper for callers that need only the split JSON."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(split.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
