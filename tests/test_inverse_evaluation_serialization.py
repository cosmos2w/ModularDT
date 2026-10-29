"""Strict JSON/CSV/NPZ serialization for verified inverse candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from honf_inverse_core.sampling.contracts import CandidateRecord
from honf_inverse_core.sampling.serialization import write_candidate_arrays, write_candidates_csv, write_json_atomic


def _candidate() -> CandidateRecord:
    compact = np.zeros((2, 12), dtype=np.float32)
    return CandidateRecord(
        "P000_L000", "raw_unguided", 0, 0, 3, 5,
        compact, compact,
        {"module_centers": [[1.0, 1.0], [0.0, 0.0]], "module_present": [1, 0], "heat_powers": [2.0, 0.0]},
        {"valid": True}, {}, compact, compact, 0.2, {}, [], 0.1, False,
        False, 0.0, 1, [1], {},
    )


def test_evaluation_serialization_artifacts_exist_and_are_pickle_free(tmp_path: Path) -> None:
    candidate = _candidate()
    write_json_atomic(tmp_path / "summary.json", candidate.to_dict())
    write_candidates_csv(tmp_path / "candidates.csv", [candidate])
    write_candidate_arrays(tmp_path / "candidates.npz", [candidate])
    assert (tmp_path / "summary.json").is_file()
    assert "P000_L000" in (tmp_path / "candidates.csv").read_text()
    with np.load(tmp_path / "candidates.npz", allow_pickle=False) as values:
        assert values["planned_compact"].shape == (1, 2, 12)
