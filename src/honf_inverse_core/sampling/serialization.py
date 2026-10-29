"""Atomic JSON/CSV/NPZ writers for inverse candidate populations."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import CandidateRecord


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, destination)
    return destination


def candidate_csv_row(candidate: CandidateRecord) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "group": candidate.group,
        "source_plan_index": candidate.source_plan_index,
        "source_layout_index": candidate.source_layout_index,
        "request_violation": candidate.request_violation,
        "request_satisfied": int(candidate.request_satisfied),
        "request_term_satisfaction_fraction": candidate.request_term_satisfaction_fraction,
        "geometry_valid": int(candidate.geometry_valid),
        "plan_distance": candidate.plan_distance,
        "correction_used": int(candidate.correction_used),
        "correction_magnitude": candidate.correction_magnitude,
        "forward_call_count": candidate.forward_call_count,
        "module_count": int(np.sum(np.asarray(candidate.design["module_present"]) > 0.5)),
        "geometry_total_violation": float(candidate.geometry.get("total_violation", 0.0)),
        "forward_call_indices_json": json.dumps(list(candidate.forward_call_indices)),
        "functional_values_json": json.dumps(candidate.functional_values, sort_keys=True),
        "request_terms_json": json.dumps(list(candidate.request_terms), sort_keys=True),
        "design_json": json.dumps(candidate.design, sort_keys=True),
    }


def write_candidates_csv(path: str | Path, candidates: Sequence[CandidateRecord]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    rows = [candidate_csv_row(candidate) for candidate in candidates]
    fieldnames = list(rows[0]) if rows else list(candidate_csv_row.__annotations__)
    if not rows:
        fieldnames = [
            "candidate_id", "group", "source_plan_index", "source_layout_index",
            "request_violation", "request_satisfied", "request_term_satisfaction_fraction", "geometry_valid", "plan_distance",
            "correction_used", "correction_magnitude", "forward_call_count", "module_count",
            "geometry_total_violation", "forward_call_indices_json", "functional_values_json",
            "request_terms_json", "design_json",
        ]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)
    return destination


def write_candidate_arrays(path: str | Path, candidates: Sequence[CandidateRecord]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        raise ValueError("Complete candidate array export requires at least one candidate.")
    arrays = {
        "candidate_id": np.asarray([candidate.candidate_id for candidate in candidates]),
        "planned_compact": np.stack([candidate.planned_compact_normalized for candidate in candidates]),
        "realized_compact": np.stack([candidate.realized_compact_normalized for candidate in candidates]),
        "module_centers": np.stack([candidate.design["module_centers"] for candidate in candidates]),
        "module_present": np.stack([candidate.design["module_present"] for candidate in candidates]),
        "heat_powers": np.stack([candidate.design["heat_powers"] for candidate in candidates]),
        "request_violation": np.asarray([candidate.request_violation for candidate in candidates], dtype=np.float32),
        "plan_distance": np.asarray([candidate.plan_distance for candidate in candidates], dtype=np.float32),
    }
    full_keys = (
        "A_mh", "A_eh", "hyper_source_coords", "hyper_region_coords",
        "hyper_module_mass", "hyper_env_mass", "hyper_strength",
        "active_hyperedge_mask", "env_coords", "edge_permutation",
    )
    for key in full_keys:
        if all(key in candidate.realized_full_plan for candidate in candidates):
            values = [np.asarray(candidate.realized_full_plan[key]) for candidate in candidates]
            if len({value.shape for value in values}) == 1:
                arrays[f"realized_full_{key}"] = np.stack(values)
    np.savez_compressed(destination, **arrays)
    return destination


__all__ = ["candidate_csv_row", "write_candidate_arrays", "write_candidates_csv", "write_json_atomic"]
