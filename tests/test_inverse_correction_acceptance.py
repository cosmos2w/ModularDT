"""One verified bounded correction proposal is accepted only when beneficial."""

from __future__ import annotations

import numpy as np
import pytest

from honf_inverse_core.sampling.contracts import CandidateRecord
from channelthermal.inverse.evaluation.candidate_evaluator import select_one_pass_representatives


def _candidate(name: str, group: str, violation: float, *, valid: bool = True) -> CandidateRecord:
    compact = np.zeros((2, 12), dtype=np.float32)
    corrected = group == "corrected"
    return CandidateRecord(
        name, group, 0, 0, 1, 2, compact, compact,
        {"module_centers": [[0.5, 0.5]], "module_present": [1], "heat_powers": [1.0]},
        {"valid": valid}, {}, compact, compact, 0.1, {}, [], violation, False,
        corrected, 0.01 if corrected else 0.0, 2 if corrected else 1,
        [1, 2] if corrected else [1], {},
    )


def test_one_pass_acceptance_keeps_only_verified_improvements() -> None:
    raw = [_candidate("a", "raw_unguided", 0.3), _candidate("b", "raw_unguided", 0.2)]
    proposals = [_candidate("a_C", "corrected", 0.1), _candidate("b_C", "corrected", 0.3)]
    accepted = select_one_pass_representatives(raw, proposals)
    assert accepted == [proposals[0], raw[1]]


def test_one_pass_acceptance_rejects_invalid_and_mismatched_proposals() -> None:
    raw = [_candidate("a", "raw_unguided", 0.3)]
    invalid = [_candidate("a_C", "corrected", 0.1, valid=False)]
    assert select_one_pass_representatives(raw, invalid) == raw
    with pytest.raises(ValueError, match="exactly one proposal"):
        select_one_pass_representatives(raw, invalid * 2)
