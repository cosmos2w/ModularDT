"""One bounded, reference-checked comparison of native inverse proposals.

Every policy sees the same intact absolute predictor, reference baseline,
number of two-coordinate layout candidates, geometry rules, pressure limit,
and one charged reference trial. A policy chooses by its corrected native
scores before the independent reference result is revealed. This experiment
does not turn a physically nominated pair into a learned causal hyperedge.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from honf_inverse_core.contracts import NamedContext, PhysicalDesign
from honf_inverse_core.request_schema import GeometryConstraints

from .geometry import evaluate_geometry
from .interaction_guided import DecisionEstimate, DecisionObservation
from .native_corrected import NativeBaselineCorrection, NativeQuantityPredictor, TopologyFactory


@dataclass(frozen=True)
class NativeCandidate:
    label: str
    design: PhysicalDesign


@dataclass(frozen=True)
class NativePolicySelection:
    policy: str
    candidate_labels: tuple[str, ...]
    predicted: tuple[DecisionEstimate, ...]
    selected_index: int
    selected_predicted_feasible: bool
    predicted_improvement: float


@dataclass(frozen=True)
class NativePolicyResult:
    policy: str
    candidate_labels: tuple[str, ...]
    predicted: tuple[DecisionEstimate, ...]
    selected_index: int
    selected_predicted_feasible: bool
    reference: DecisionObservation | None
    reference_error: str | None
    reference_attempts: int
    predicted_improvement: float
    actual_improvement: float | None
    accepted: bool
    actual_to_predicted_ratio: float | None
    suggested_next_radius: float


def _objective(values: DecisionEstimate | DecisionObservation, module_id: Hashable | None) -> float:
    if module_id is None:
        return float(max(values.module_temperature_by_id.values()))
    return float(values.module_temperature_by_id[module_id])


def _validate_matched_candidates(
    baseline: PhysicalDesign,
    context: NamedContext,
    constraints: GeometryConstraints,
    candidates_by_policy: Mapping[str, Sequence[NativeCandidate]],
    *,
    trust_radius: float,
    changed_coordinates: int,
) -> int:
    if not candidates_by_policy:
        raise ValueError("At least one inverse policy is required.")
    if not evaluate_geometry(baseline, context, constraints).valid:
        raise ValueError("The measured baseline violates declared geometry constraints.")
    counts = {len(candidates) for candidates in candidates_by_policy.values()}
    if len(counts) != 1 or next(iter(counts)) == 0:
        raise ValueError("All matched policies need the same nonzero candidate count.")
    if not np.isfinite(trust_radius) or trust_radius <= 0.0:
        raise ValueError("Trust radius must be positive and finite.")
    if changed_coordinates <= 0:
        raise ValueError("Changed-coordinate budget must be positive.")
    baseline_centers = np.asarray(baseline.module_centers)
    for policy, candidates in candidates_by_policy.items():
        if not policy:
            raise ValueError("Policy names must be explicit.")
        if len({item.label for item in candidates}) != len(candidates):
            raise ValueError(f"Policy {policy!r} has duplicate candidate labels.")
        for item in candidates:
            trial = item.design
            if not item.label or trial.module_family_id != baseline.module_family_id:
                raise ValueError("Every candidate needs a label and the same module family.")
            if not np.array_equal(trial.module_present, baseline.module_present):
                raise ValueError("Position trials cannot change module presence.")
            if not np.array_equal(trial.heat_powers, baseline.heat_powers):
                raise ValueError("Position trials cannot change heating.")
            delta = np.asarray(trial.module_centers) - baseline_centers
            changed = np.abs(delta) > 1.0e-6
            if int(changed.sum()) != changed_coordinates:
                raise ValueError("Every matched candidate must change the declared coordinate count.")
            if np.max(np.abs(delta)) > trust_radius + 1.0e-6:
                raise ValueError("A candidate exceeds the shared physical trust radius.")
            if not evaluate_geometry(trial, context, constraints).valid:
                raise ValueError("A proposed candidate violates exact disk geometry.")
    return next(iter(counts))


def run_matched_native_round(
    *,
    predictor: NativeQuantityPredictor,
    baseline_design: PhysicalDesign,
    reference_baseline: DecisionObservation,
    context: NamedContext,
    module_ids_by_slot: Sequence[Hashable | None],
    candidates_by_policy: Mapping[str, Sequence[NativeCandidate]],
    reference_trial: Callable[[str, NativeCandidate], DecisionObservation],
    constraints: GeometryConstraints,
    pressure_limit: float,
    trust_radius: float,
    changed_coordinates: int = 2,
    objective_module_id: Hashable | None = None,
    decision_resolution: float = 1.0e-4,
    topology_factory_by_policy: Mapping[str, TopologyFactory] | None = None,
    selection_callback: Callable[[Mapping[str, NativePolicySelection]], None] | None = None,
) -> dict[str, NativePolicyResult]:
    """Score equal-sized candidate sets; independently verify one per arm.

    If a policy predicts no feasible candidate, its least-violating candidate
    is still sent to reference as a charged *audit-only* trial. It cannot be
    accepted by this policy, even when the independent result is feasible.
    The predicted denominator is used for a trust ratio only when it exceeds
    the declared decision resolution.
    """

    _validate_matched_candidates(
        baseline_design, context, constraints, candidates_by_policy,
        trust_radius=trust_radius, changed_coordinates=changed_coordinates,
    )
    if not np.isfinite(pressure_limit) or pressure_limit <= 0.0:
        raise ValueError("Pressure limit must be positive and finite.")
    if not np.isfinite(decision_resolution) or decision_resolution <= 0.0:
        raise ValueError("Decision resolution must be positive and finite.")
    if reference_baseline.pressure_drop > pressure_limit + 1.0e-8:
        raise ValueError("The independently measured baseline is pressure-infeasible.")
    if objective_module_id is not None and objective_module_id not in reference_baseline.module_temperature_by_id:
        raise ValueError("The nominated objective module is absent from the measured baseline.")
    baseline_objective = _objective(reference_baseline, objective_module_id)
    selected: dict[str, NativePolicySelection] = {}
    for policy, candidates in candidates_by_policy.items():
        topology_factory = (topology_factory_by_policy or {}).get(policy)
        anchor = NativeBaselineCorrection(
            predictor=predictor,
            baseline_design=baseline_design,
            reference_baseline=reference_baseline,
            context=context,
            module_ids_by_slot=module_ids_by_slot,
            topology_factory=topology_factory,
        )
        predicted = tuple(anchor.evaluate(item.design).baseline_corrected for item in candidates)
        feasible = [index for index, value in enumerate(predicted) if value.pressure_drop <= pressure_limit]
        if feasible:
            selected_index = min(feasible, key=lambda index: (_objective(predicted[index], objective_module_id), index))
            predicted_feasible = True
        else:
            selected_index = min(
                range(len(predicted)),
                key=lambda index: (
                    max(0.0, predicted[index].pressure_drop - pressure_limit),
                    _objective(predicted[index], objective_module_id), index,
                ),
            )
            predicted_feasible = False
        chosen_prediction = predicted[selected_index]
        predicted_improvement = baseline_objective - _objective(chosen_prediction, objective_module_id)
        selected[policy] = NativePolicySelection(
            policy=policy,
            candidate_labels=tuple(item.label for item in candidates),
            predicted=predicted,
            selected_index=selected_index,
            selected_predicted_feasible=predicted_feasible,
            predicted_improvement=predicted_improvement,
        )
    if selection_callback is not None:
        selection_callback(selected)

    results: dict[str, NativePolicyResult] = {}
    for policy, candidates in candidates_by_policy.items():
        selection = selected[policy]
        chosen = candidates[selection.selected_index]
        measured: DecisionObservation | None = None
        error: str | None = None
        try:
            measured = reference_trial(policy, chosen)
            if measured.evidence_source not in {"reference_solver", "stored_reference"}:
                raise ValueError("A trial acceptance needs an independent reference outcome.")
            if set(measured.module_temperature_by_id) != set(reference_baseline.module_temperature_by_id):
                raise ValueError("Reference trial changed physical module identities.")
            if (
                measured.pressure_drop_units != reference_baseline.pressure_drop_units
                or measured.pressure_drop_definition != reference_baseline.pressure_drop_definition
            ):
                raise ValueError("Reference trial pressure definition or units changed.")
        except Exception as exc:  # noqa: BLE001 - reference failure is charged and reported.
            error = f"{type(exc).__name__}: {exc}"
            measured = None
        actual_improvement = (
            None if measured is None else baseline_objective - _objective(measured, objective_module_id)
        )
        accepted = bool(
            selection.selected_predicted_feasible and measured is not None
            and measured.pressure_drop <= pressure_limit
            and actual_improvement is not None
            and actual_improvement > decision_resolution
        )
        ratio = (
            actual_improvement / selection.predicted_improvement
            if actual_improvement is not None and selection.predicted_improvement > decision_resolution
            else None
        )
        if not accepted or (ratio is not None and ratio < 0.25):
            next_radius = trust_radius * 0.5
        elif ratio is not None and ratio > 0.75:
            next_radius = trust_radius * 1.5
        else:
            next_radius = trust_radius
        results[policy] = NativePolicyResult(
            policy=policy,
            candidate_labels=selection.candidate_labels,
            predicted=selection.predicted,
            selected_index=selection.selected_index,
            selected_predicted_feasible=selection.selected_predicted_feasible,
            reference=measured,
            reference_error=error,
            reference_attempts=1,
            predicted_improvement=selection.predicted_improvement,
            actual_improvement=actual_improvement,
            accepted=accepted,
            actual_to_predicted_ratio=ratio,
            suggested_next_radius=next_radius,
        )
    return results


__all__ = [
    "NativeCandidate", "NativePolicyResult", "NativePolicySelection", "run_matched_native_round",
]
