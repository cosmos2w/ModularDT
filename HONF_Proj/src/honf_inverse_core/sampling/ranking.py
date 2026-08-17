"""Transparent request-first, geometry, plan, and diversity ranking.

Each record already captures planned `G`, design `D`, realized `G_hat`, request
`R`, and context provenance. Ranking is post-generation reporting; success is
always measured on the full raw/corrected populations before this selection.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .contracts import CandidateRecord


def _descriptor_distance(left: CandidateRecord, right: CandidateRecord) -> float:
    left_centers = np.asarray(left.design["module_centers"], dtype=np.float64)
    right_centers = np.asarray(right.design["module_centers"], dtype=np.float64)
    left_present = np.asarray(left.design["module_present"], dtype=np.float64)
    right_present = np.asarray(right.design["module_present"], dtype=np.float64)
    left_heat = np.asarray(left.design["heat_powers"], dtype=np.float64)
    right_heat = np.asarray(right.design["heat_powers"], dtype=np.float64)
    center = np.mean(np.square(left_centers - right_centers))
    presence = np.mean(np.abs(left_present - right_present))
    heat_scale = max(np.std(np.concatenate([left_heat, right_heat])), 1.0)
    heat = np.mean(np.square((left_heat - right_heat) / heat_scale))
    return float(np.sqrt(center + presence + heat))


def rank_candidates(
    candidates: Sequence[CandidateRecord],
    *,
    top_k: int = 8,
    request_near_tie: float = 0.02,
    plan_near_tie: float = 0.02,
) -> list[CandidateRecord]:
    """Greedily diversify only within practical lexicographic near ties."""

    remaining = list(candidates)
    selected: list[CandidateRecord] = []
    while remaining and len(selected) < int(top_k):
        remaining.sort(
            key=lambda candidate: (
                candidate.request_violation,
                0 if candidate.geometry_valid else 1,
                candidate.plan_distance,
                candidate.candidate_id,
            )
        )
        best = remaining[0]
        near = [
            candidate
            for candidate in remaining
            if candidate.request_violation <= best.request_violation + request_near_tie
            and candidate.geometry_valid == best.geometry_valid
            and candidate.plan_distance <= best.plan_distance + plan_near_tie
        ]
        if selected and len(near) > 1:
            best = max(
                near,
                key=lambda candidate: min(_descriptor_distance(candidate, prior) for prior in selected),
            )
        selected.append(best)
        remaining.remove(best)
    return selected


__all__ = ["rank_candidates"]
