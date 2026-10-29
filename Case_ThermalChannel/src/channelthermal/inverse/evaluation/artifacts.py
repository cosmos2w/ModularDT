"""Durable outputs for verified `R,c -> G -> D -> G_hat` evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from honf_inverse_core.sampling.contracts import InverseSamplingResult
from honf_inverse_core.sampling.serialization import (
    write_candidate_arrays,
    write_candidates_csv,
    write_json_atomic,
)

from .plots import plot_population, plot_top_candidate


def write_evaluation_artifacts(result: InverseSamplingResult, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_candidates = result.raw_unguided + result.corrected
    summary = {
        "status": "complete",
        "metadata": dict(result.metadata),
        "raw_unguided": {
            "count": len(result.raw_unguided),
            "request_success_fraction": float(np.mean([candidate.request_satisfied for candidate in result.raw_unguided])),
            "geometry_valid_fraction": float(np.mean([candidate.geometry_valid for candidate in result.raw_unguided])),
        },
        "corrected": None if not result.corrected else {
            "count": len(result.corrected),
            "role": "verified one-pass proposals before trust-region acceptance",
            "request_success_fraction": float(np.mean([candidate.request_satisfied for candidate in result.corrected])),
            "geometry_valid_fraction": float(np.mean([candidate.geometry_valid for candidate in result.corrected])),
            "mean_correction_magnitude": float(np.mean([candidate.correction_magnitude for candidate in result.corrected])),
        },
        "accepted_one_pass": {
            "count": len(result.accepted_one_pass),
            "accepted_correction_count": int(np.sum([
                candidate.correction_used for candidate in result.accepted_one_pass
            ])),
            "request_success_fraction": float(np.mean([
                candidate.request_satisfied for candidate in result.accepted_one_pass
            ])),
            "request_term_satisfaction_fraction": float(np.mean([
                candidate.request_term_satisfaction_fraction
                for candidate in result.accepted_one_pass
            ])),
            "geometry_valid_fraction": float(np.mean([
                candidate.geometry_valid for candidate in result.accepted_one_pass
            ])),
            "mean_request_violation": float(np.mean([
                candidate.request_violation for candidate in result.accepted_one_pass
            ])),
        },
        "final_ranked": {
            "count": len(result.final_ranked),
            "candidate_ids": [candidate.candidate_id for candidate in result.final_ranked],
        },
        "success_claim_basis": (
            "full raw, correction-proposal, and accepted-one-pass populations; "
            "final reranking is not counted as generation success"
        ),
    }
    write_json_atomic(output / "evaluation_summary.json", summary)
    write_candidates_csv(output / "candidates_all.csv", all_candidates)
    write_candidates_csv(output / "candidates_accepted.csv", result.accepted_one_pass)
    write_candidates_csv(output / "candidates_top.csv", result.final_ranked)
    write_json_atomic(output / "request_summary.json", result.request_summary)
    write_json_atomic(
        output / "plan_generation_summary.json",
        {
            "num_plans": result.metadata["num_plans"],
            "layouts_per_plan": result.metadata["layouts_per_plan"],
            "raw_plan_distance_mean": float(np.mean([candidate.plan_distance for candidate in result.raw_unguided])),
            "raw_plan_distance_median": float(np.median([candidate.plan_distance for candidate in result.raw_unguided])),
            "plan_seeds": sorted({candidate.plan_seed for candidate in result.raw_unguided}),
        },
    )
    write_candidate_arrays(output / "candidates_arrays.npz", all_candidates)
    plot_population(result, output / "comparison_plots")
    for rank, candidate in enumerate(result.final_ranked):
        directory = output / "top_candidates" / f"rank_{rank + 1:02d}_{candidate.candidate_id}"
        write_json_atomic(directory / "candidate.json", candidate.to_dict(include_dense=False))
        plot_top_candidate(candidate, directory)
    return summary


__all__ = ["write_evaluation_artifacts"]
