"""Versioned serializable records for `R,c -> G -> D -> G_hat` candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from honf_inverse_core.contracts import jsonable


CANDIDATE_SCHEMA_NAME = "honf_hierarchical_inverse_candidate"
CANDIDATE_SCHEMA_VERSION = 1


@dataclass
class CandidateRecord:
    candidate_id: str
    group: str
    source_plan_index: int
    source_layout_index: int
    plan_seed: int
    layout_seed: int
    planned_compact_raw: np.ndarray
    planned_compact_normalized: np.ndarray
    design: Mapping[str, Any]
    geometry: Mapping[str, Any]
    realized_full_plan: Mapping[str, Any]
    realized_compact_raw: np.ndarray
    realized_compact_normalized: np.ndarray
    plan_distance: float
    functional_values: Mapping[str, Any]
    request_terms: Sequence[Mapping[str, Any]]
    request_violation: float
    request_satisfied: bool
    correction_used: bool
    correction_magnitude: float
    forward_call_count: int
    forward_call_indices: Sequence[int]
    outputs: Mapping[str, Any] = field(default_factory=dict)
    immutable_base_plan_distance: float | None = None

    def __post_init__(self) -> None:
        self.planned_compact_raw = np.asarray(self.planned_compact_raw, dtype=np.float32)
        self.planned_compact_normalized = np.asarray(self.planned_compact_normalized, dtype=np.float32)
        self.realized_compact_raw = np.asarray(self.realized_compact_raw, dtype=np.float32)
        self.realized_compact_normalized = np.asarray(self.realized_compact_normalized, dtype=np.float32)
        numeric = [self.plan_distance, self.request_violation, self.correction_magnitude]
        if not np.isfinite(numeric).all():
            raise ValueError("Candidate scalar metrics must be finite.")
        if self.group not in {"raw_unguided", "corrected"}:
            raise ValueError(f"Unsupported candidate group: {self.group!r}")
        if int(self.forward_call_count) != len(self.forward_call_indices):
            raise ValueError("Candidate forward_call_count must match its explicit call ledger.")

    @property
    def geometry_valid(self) -> bool:
        return bool(self.geometry.get("valid", False))

    @property
    def request_term_satisfaction_fraction(self) -> float:
        return float(np.mean([bool(term.get("satisfied", False)) for term in self.request_terms])) if self.request_terms else 0.0

    def layout_descriptor(self) -> np.ndarray:
        centers = np.asarray(self.design["module_centers"], dtype=np.float32)
        present = np.asarray(self.design["module_present"], dtype=np.float32) > 0.5
        heat = np.asarray(self.design["heat_powers"], dtype=np.float32)
        selected = np.concatenate([centers[present].reshape(-1), heat[present].reshape(-1)])
        return selected.astype(np.float32)

    def to_dict(self, *, include_dense: bool = False) -> dict[str, Any]:
        payload = {
            "schema_name": CANDIDATE_SCHEMA_NAME,
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "group": self.group,
            "source_plan_index": int(self.source_plan_index),
            "source_layout_index": int(self.source_layout_index),
            "plan_seed": int(self.plan_seed),
            "layout_seed": int(self.layout_seed),
            "design": jsonable(self.design),
            "geometry": jsonable(self.geometry),
            "plan_distance": float(self.plan_distance),
            "immutable_base_plan_distance": (
                None if self.immutable_base_plan_distance is None else float(self.immutable_base_plan_distance)
            ),
            "functional_values": jsonable(self.functional_values),
            "request_terms": jsonable(self.request_terms),
            "request_violation": float(self.request_violation),
            "request_satisfied": bool(self.request_satisfied),
            "request_term_satisfaction_fraction": self.request_term_satisfaction_fraction,
            "correction_used": bool(self.correction_used),
            "correction_magnitude": float(self.correction_magnitude),
            "forward_call_count": int(self.forward_call_count),
            "forward_call_indices": list(map(int, self.forward_call_indices)),
        }
        if include_dense:
            payload.update(
                planned_compact_raw=self.planned_compact_raw.tolist(),
                planned_compact_normalized=self.planned_compact_normalized.tolist(),
                realized_full_plan=jsonable(self.realized_full_plan),
                realized_compact_raw=self.realized_compact_raw.tolist(),
                realized_compact_normalized=self.realized_compact_normalized.tolist(),
                outputs=jsonable(self.outputs),
            )
        return payload


@dataclass
class InverseSamplingResult:
    request_summary: Mapping[str, Any]
    metadata: Mapping[str, Any]
    raw_unguided: list[CandidateRecord]
    corrected: list[CandidateRecord]
    accepted_one_pass: list[CandidateRecord]
    final_ranked: list[CandidateRecord]

    def to_dict(self, *, include_dense: bool = False) -> dict[str, Any]:
        return {
            "schema_name": "honf_hierarchical_inverse_sampling_result",
            "schema_version": 2,
            "request_summary": jsonable(self.request_summary),
            "metadata": jsonable(self.metadata),
            "generated": [candidate.to_dict(include_dense=include_dense) for candidate in self.raw_unguided],
            "corrected": [candidate.to_dict(include_dense=include_dense) for candidate in self.corrected],
            "accepted_one_pass": [
                candidate.to_dict(include_dense=include_dense) for candidate in self.accepted_one_pass
            ],
            "verified": [candidate.to_dict(include_dense=include_dense) for candidate in self.final_ranked],
            "raw_unguided_candidates": [candidate.to_dict(include_dense=include_dense) for candidate in self.raw_unguided],
            "correction_proposals": [
                candidate.to_dict(include_dense=include_dense) for candidate in self.corrected
            ],
            "accepted_one_pass_candidates": [
                candidate.to_dict(include_dense=include_dense) for candidate in self.accepted_one_pass
            ],
            "final_ranked_candidates": [candidate.to_dict(include_dense=include_dense) for candidate in self.final_ranked],
        }


__all__ = [
    "CANDIDATE_SCHEMA_NAME",
    "CANDIDATE_SCHEMA_VERSION",
    "CandidateRecord",
    "InverseSamplingResult",
]
