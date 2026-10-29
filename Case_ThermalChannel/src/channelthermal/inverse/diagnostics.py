"""Reread-based diagnostics for a completed inverse dataset.

The artifact stores physical designs ``D``, contexts ``c``, augmented requests
``R``, compact targets ``G``, and provenance for later realized ``G_hat``.
Diagnostics never depend on private builder memory and therefore validate the
public storage contract while reporting it.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from honf_inverse_core.request_schema import RELATION_NAMES

from .compact_plan import COMPACT_PLAN_FEATURE_NAMES
from .dataset_io import validate_inverse_hdf5
from .vocabulary import REQUEST_TYPES


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def generate_dataset_diagnostics(dataset_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Validate and summarize one finalized artifact, then write three plots."""

    dataset_path = Path(dataset_path).expanduser().resolve()
    diagnostics_dir = Path(output_dir).expanduser().resolve()
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    basic = validate_inverse_hdf5(dataset_path)
    with h5py.File(dataset_path, "r") as h5:
        active = h5["requests/active_mask"][...].astype(bool)
        type_id = h5["requests/type_id"][...]
        relation_id = h5["requests/relation_id"][...]
        plan = h5["plan/compact_raw"][...]
        global_values = h5["functionals/global_raw"][...]
        regional_values = h5["requests/realized_value_raw"][...]
        type_counts = Counter(int(value) for value in type_id[active])
        relation_counts = Counter(int(value) for value in relation_id[active])
        split_hashes = json.loads(str(h5.attrs["split_hashes"]))
        summary = {
            **basic,
            "artifact": str(dataset_path),
            "artifact_size": int(dataset_path.stat().st_size),
            "artifact_sha256": artifact_sha256(dataset_path),
            "partial_debug": bool(h5.attrs.get("partial_debug", False)),
            "schema_versions": {
                "dataset": int(h5.attrs["schema_version"]),
                "request": int(h5.attrs["request_schema_version"]),
                "compact_plan": int(h5.attrs["compact_plan_schema_version"]),
                "canonical_full_plan": int(h5.attrs["canonical_full_plan_schema_version"]),
            },
            "active_token_count_distribution": dict(
                sorted(Counter(active.sum(axis=-1).reshape(-1).tolist()).items())
            ),
            "request_type_counts": {
                REQUEST_TYPES[index]: int(type_counts.get(index, 0)) for index in range(len(REQUEST_TYPES))
            },
            "relation_counts": {
                RELATION_NAMES[index]: int(relation_counts.get(index, 0)) for index in range(len(RELATION_NAMES))
            },
            "geometry_valid_fraction": float(np.mean(h5["geometry/valid"][...])),
            "plan_feature_min": dict(zip(COMPACT_PLAN_FEATURE_NAMES, np.min(plan, axis=(0, 1)).tolist())),
            "plan_feature_max": dict(zip(COMPACT_PLAN_FEATURE_NAMES, np.max(plan, axis=(0, 1)).tolist())),
            "split_case_id_hashes": split_hashes,
        }

        figure, axes = plt.subplots(3, 3, figsize=(13, 10))
        for index, name in enumerate(REQUEST_TYPES):
            axis = axes.flat[index]
            if index < 5:
                values = global_values[:, index]
            else:
                selected = active & (type_id == index)
                values = regional_values[selected]
            axis.hist(np.asarray(values).reshape(-1), bins=24, color="#3567a8", alpha=0.85)
            axis.set_title(name.replace("_", " "), fontsize=9)
        for axis in axes.flat[len(REQUEST_TYPES):]:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(diagnostics_dir / "functional_histograms.png", dpi=160)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].bar(range(len(REQUEST_TYPES)), [type_counts.get(i, 0) for i in range(len(REQUEST_TYPES))])
        axes[0].set_xticks(range(len(REQUEST_TYPES)), [name.replace("_", "\n") for name in REQUEST_TYPES], fontsize=7)
        axes[0].set_title("Request type counts")
        axes[1].bar(range(len(RELATION_NAMES)), [relation_counts.get(i, 0) for i in range(len(RELATION_NAMES))])
        axes[1].set_xticks(range(len(RELATION_NAMES)), RELATION_NAMES, rotation=20, ha="right")
        axes[1].set_title("Relation counts")
        figure.tight_layout()
        figure.savefig(diagnostics_dir / "request_type_counts.png", dpi=160)
        plt.close(figure)

        figure, axes = plt.subplots(3, 4, figsize=(14, 9))
        for index, name in enumerate(COMPACT_PLAN_FEATURE_NAMES):
            axes.flat[index].hist(plan[..., index].reshape(-1), bins=24, color="#248f6b", alpha=0.85)
            axes.flat[index].set_title(name.replace("_", " "), fontsize=8)
        figure.tight_layout()
        figure.savefig(diagnostics_dir / "compact_plan_feature_histograms.png", dpi=160)
        plt.close(figure)

    _write_json(diagnostics_dir / "dataset_summary.json", summary)
    _write_json(diagnostics_dir / "split_case_id_hashes.json", summary["split_case_id_hashes"])
    return summary


__all__ = ["artifact_sha256", "generate_dataset_diagnostics"]
