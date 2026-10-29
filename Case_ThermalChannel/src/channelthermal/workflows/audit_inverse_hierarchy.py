"""Bounded held-out audit of the five hierarchical inverse milestones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from honf_inverse_core.training.checkpointing import load_inverse_checkpoint

from channelthermal.inverse.context import CONTEXT_FEATURE_NAMES, CONTEXT_SCHEMA_NAME, parse_context
from channelthermal.inverse.compact_plan import COMPACT_PLAN_FEATURE_NAMES
from channelthermal.inverse.evaluation.candidate_evaluator import ThermalChannelCandidateEvaluator
from channelthermal.inverse.evaluation.scoring import compact_plan_distance
from channelthermal.inverse.request import make_request_codec
from channelthermal.inverse.verifier import FrozenThermalChannelVerifier, file_sha256
from channelthermal.workflows.evaluate_inverse_hierarchical import normalizers_from_checkpoint


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _mean_pair_distance(records: list[Any]) -> float:
    values = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            left_centers = np.asarray(records[left].design["module_centers"], dtype=np.float64)
            right_centers = np.asarray(records[right].design["module_centers"], dtype=np.float64)
            values.append(float(np.sqrt(np.mean(np.square(left_centers - right_centers)))))
    return float(np.mean(values)) if values else 0.0


def audit_hierarchy(
    *,
    inverse_checkpoint: str | Path,
    forward_checkpoint: str | Path,
    inverse_dataset: str | Path,
    device: str = "cuda:0",
    split: str = "test",
    num_requests: int = 5,
    num_plans: int = 2,
    layouts_per_plan: int = 2,
    seed: int = 20260813,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = load_inverse_checkpoint(inverse_checkpoint)
    expected = checkpoint["provenance"].get("forward_checkpoint_sha256")
    actual = file_sha256(forward_checkpoint)
    if expected and expected != actual:
        raise ValueError("Milestone audit forward checkpoint SHA mismatch.")
    normalizers = normalizers_from_checkpoint(checkpoint)
    codec = make_request_codec(normalizers.functional)
    designer = HierarchicalInverseDesigner.load(inverse_checkpoint, device=device)
    frozen = FrozenThermalChannelVerifier(forward_checkpoint, device=device)
    evaluator = ThermalChannelCandidateEvaluator(designer, frozen, normalizers)
    dataset_path = Path(inverse_dataset).expanduser().resolve()
    inputs = []
    with h5py.File(dataset_path, "r") as h5:
        splits = [_decode(value) for value in h5["inverse_split"][...]]
        candidates = [index for index, name in enumerate(splits) if name == split]
        if not candidates:
            raise ValueError(f"Audit dataset has no {split!r} cases.")
        for request_index in range(int(num_requests)):
            case_index = candidates[request_index % len(candidates)]
            variant = request_index % int(h5.attrs["variants_per_case"])
            request = codec.parse(json.loads(_decode(h5["requests/json"][case_index, variant])))
            vector = h5["context/vector"][case_index].astype(np.float32)
            payload = {
                "schema_name": CONTEXT_SCHEMA_NAME,
                "schema_version": 1,
                **{name: float(vector[index]) for index, name in enumerate(CONTEXT_FEATURE_NAMES)},
            }
            inputs.append((request, parse_context(payload)))

    results = [
        evaluator.sample_candidates(
            request=request,
            context=context,
            num_plans=num_plans,
            layouts_per_plan=layouts_per_plan,
            correct_once=True,
            top_k=min(8, num_plans * layouts_per_plan),
            seed=seed,
        )
        for request, context in inputs
    ]
    raw = [candidate for result in results for candidate in result.raw_unguided]
    corrected = [candidate for result in results for candidate in result.corrected]
    plan_representatives = [result.raw_unguided[0].planned_compact_normalized for result in results]
    request_sensitive_distance = (
        float(np.mean([
            np.sqrt(np.mean(np.square(plan_representatives[index] - plan_representatives[0])))
            for index in range(1, len(plan_representatives))
        ]))
        if len(plan_representatives) > 1 else 0.0
    )
    raw_term_success = float(np.mean([candidate.request_term_satisfaction_fraction for candidate in raw]))
    raw_partial_success = float(np.mean([
        candidate.geometry_valid and candidate.request_term_satisfaction_fraction >= 0.5
        for candidate in raw
    ]))
    raw_full_success = float(np.mean([
        candidate.geometry_valid and candidate.request_satisfied for candidate in raw
    ]))
    corrected_term_success = float(np.mean([candidate.request_term_satisfaction_fraction for candidate in corrected]))
    raw_violation = float(np.mean([candidate.request_violation for candidate in raw]))
    corrected_violation = float(np.mean([candidate.request_violation for candidate in corrected]))
    correction_improvement = (raw_violation - corrected_violation) / max(raw_violation, 1.0e-8)
    accepted = [candidate for result in results for candidate in result.accepted_one_pass]
    accepted_violation = float(np.mean([candidate.request_violation for candidate in accepted]))
    accepted_term_success = float(np.mean([
        candidate.request_term_satisfaction_fraction for candidate in accepted
    ]))
    accepted_improvement = (raw_violation - accepted_violation) / max(raw_violation, 1.0e-8)
    accepted_corrections = [candidate.correction_used for candidate in accepted]
    accepted_center_displacements = []
    accepted_heat_changes = []
    for result, (_, context) in zip(results, inputs):
        values = context.as_mapping()
        domain_diagonal = float(np.hypot(values["domain_length_x"], values["domain_length_y"]))
        for base, selected in zip(result.raw_unguided, result.accepted_one_pass):
            if not selected.correction_used:
                continue
            base_present = np.asarray(base.design["module_present"], dtype=np.float64) > 0.5
            selected_present = np.asarray(selected.design["module_present"], dtype=np.float64) > 0.5
            common = base_present & selected_present
            if np.any(common):
                base_centers = np.asarray(base.design["module_centers"], dtype=np.float64)
                selected_centers = np.asarray(selected.design["module_centers"], dtype=np.float64)
                accepted_center_displacements.extend(
                    (np.linalg.norm(selected_centers[common] - base_centers[common], axis=-1)
                     / max(domain_diagonal, 1.0e-8)).tolist()
                )
                base_heat = np.asarray(base.design["heat_powers"], dtype=np.float64)
                selected_heat = np.asarray(selected.design["heat_powers"], dtype=np.float64)
                accepted_heat_changes.extend(
                    (np.abs(selected_heat[common] - base_heat[common])
                     / max(normalizers.active_heat.std, 1.0e-8)).tolist()
                )
    median_center_displacement = (
        float(np.median(accepted_center_displacements)) if accepted_center_displacements else 0.0
    )
    median_heat_change = float(np.median(accepted_heat_changes)) if accepted_heat_changes else 0.0
    diversity = float(np.mean([_mean_pair_distance(result.raw_unguided) for result in results]))
    plan_median = float(np.median([candidate.plan_distance for candidate in raw]))
    matched_plan_medians = {
        mode: float(np.median([
            compact_plan_distance(
                candidate.planned_compact_normalized,
                candidate.realized_compact_normalized,
                matching_mode=mode,
            )
            for candidate in raw
        ]))
        for mode in ("canonical", "hungarian", "sinkhorn")
    }
    per_feature_rms = np.sqrt(np.mean(np.stack([
        np.square(
            candidate.planned_compact_normalized
            - candidate.realized_compact_normalized
        )
        for candidate in raw
    ]), axis=(0, 1)))
    correction_magnitude = float(np.mean([candidate.correction_magnitude for candidate in corrected]))
    gates = {
        # Candidate construction validates every sampled compact plan before it
        # is admitted to ``raw``; an invalid plan raises and no passing audit is
        # written. Keep the explicit fraction in the persisted evidence.
        "plan_valid_and_request_sensitive": bool(len(raw) > 0 and request_sensitive_distance > 1.0e-3),
        "layout_diverse_and_geometry_valid": bool(
            np.mean([candidate.geometry_valid for candidate in raw]) >= 0.70 and diversity > 1.0e-3
        ),
        "realized_plan_close": bool(plan_median <= 0.15),
        "raw_partial_request_success": bool(raw_partial_success >= 0.20),
        "one_pass_correction_improves_bounded": bool(
            accepted_improvement >= 0.10
            and np.mean(accepted_corrections) >= 0.10
            and correction_magnitude <= 0.08
            and median_center_displacement <= 0.05
            and median_heat_change <= 0.10
        ),
    }
    summary = {
        "status": "passed" if all(gates.values()) else "diagnostic_incomplete",
        "gate_thresholds": {
            "minimum_request_sensitive_plan_rms": 1.0e-3,
            "minimum_geometry_valid_fraction": 0.70,
            "minimum_layout_diversity": 1.0e-3,
            "maximum_median_matched_plan_distance": 0.15,
            "minimum_raw_candidates_satisfying_half_request_terms": 0.20,
            "minimum_accepted_relative_request_violation_improvement": 0.10,
            "minimum_accepted_correction_fraction": 0.10,
            "maximum_mean_correction_magnitude": 0.08,
            "maximum_median_center_displacement_domain_diagonal": 0.05,
            "maximum_median_active_heat_change_standard_deviations": 0.10,
        },
        "gates": gates,
        "metrics": {
            "request_sensitive_plan_rms": request_sensitive_distance,
            "sampled_plan_valid_fraction": 1.0,
            "raw_geometry_valid_fraction": float(np.mean([candidate.geometry_valid for candidate in raw])),
            "mean_within_request_layout_diversity": diversity,
            "median_planned_realized_distance": plan_median,
            "median_plan_distance_by_matching": matched_plan_medians,
            "planned_realized_rms_by_feature": {
                name: float(per_feature_rms[index])
                for index, name in enumerate(COMPACT_PLAN_FEATURE_NAMES)
            },
            "raw_request_term_satisfaction_fraction": raw_term_success,
            "raw_candidates_satisfying_at_least_half_request_terms": raw_partial_success,
            "raw_full_request_success_fraction": raw_full_success,
            "corrected_request_term_satisfaction_fraction": corrected_term_success,
            "raw_mean_request_violation": raw_violation,
            "corrected_mean_request_violation": corrected_violation,
            "relative_request_violation_improvement": correction_improvement,
            "accepted_corrected_request_term_satisfaction_fraction": accepted_term_success,
            "accepted_corrected_mean_request_violation": accepted_violation,
            "accepted_relative_request_violation_improvement": accepted_improvement,
            "accepted_correction_fraction": float(np.mean(accepted_corrections)),
            "mean_correction_magnitude": correction_magnitude,
            "accepted_median_center_displacement_domain_diagonal": median_center_displacement,
            "accepted_median_active_heat_change_standard_deviations": median_heat_change,
        },
        "counts": {
            "requests": len(results),
            "raw_candidates": len(raw),
            "corrected_candidates": len(corrected),
            "forward_calls": sum(result.metadata["forward_call_count"] for result in results),
        },
        "provenance": {
            "inverse_checkpoint": str(Path(inverse_checkpoint).resolve()),
            "forward_checkpoint_sha256": actual,
            "inverse_dataset": str(dataset_path),
            "split": split,
            "seed": seed,
        },
    }
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inverse-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    parser.add_argument("--inverse-dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--num-plans", type=int, default=2)
    parser.add_argument("--layouts-per-plan", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    values = vars(args).copy()
    output = values.pop("output")
    print(json.dumps(audit_hierarchy(**values, output_path=output), indent=2, sort_keys=True))
    return 0


__all__ = ["audit_hierarchy", "main"]
