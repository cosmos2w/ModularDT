"""Request-first ranking of verified `G,D,G_hat` candidates under `R,c`."""

from __future__ import annotations

import numpy as np

from honf_inverse_core.sampling.contracts import CandidateRecord
from honf_inverse_core.sampling.ranking import rank_candidates


def _candidate(name: str, request: float, valid: bool, plan: float, x: float) -> CandidateRecord:
    compact = np.zeros((2, 12), dtype=np.float32)
    design = {
        "module_centers": [[x, 0.5], [0.0, 0.0]],
        "module_present": [1, 0],
        "heat_powers": [2.0, 0.0],
    }
    return CandidateRecord(
        name, "raw_unguided", 0, 0, 1, 2,
        compact, compact, design, {"valid": valid}, {}, compact, compact,
        plan, {}, [], request, request == 0.0, False, 0.0, 1, [1], {},
    )


def test_ranking_is_request_first_then_geometry_plan_and_diversity() -> None:
    lower_request_invalid = _candidate("a", 0.01, False, 0.5, 0.0)
    higher_request_valid = _candidate("b", 0.20, True, 0.01, 0.1)
    assert rank_candidates([higher_request_valid, lower_request_invalid], top_k=2)[0] is lower_request_invalid
    near_center = _candidate("c", 0.011, False, 0.51, 0.01)
    near_far = _candidate("d", 0.012, False, 0.51, 4.0)
    ranked = rank_candidates([lower_request_invalid, near_center, near_far], top_k=3)
    assert ranked[0] is lower_request_invalid
    assert ranked[1] is near_far
