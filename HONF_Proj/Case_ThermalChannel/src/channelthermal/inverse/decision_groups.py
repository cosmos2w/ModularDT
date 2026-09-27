"""Finite, receiver-aware coordinate groups for one corrected native anchor.

This selector observes only baseline-corrected model outputs from fresh finite
trials. A group may coordinate a temperature objective and pressure constraint;
that does not assert a nonlinear physical mixed response.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from honf_inverse_core.contracts import PhysicalDesign
from honf_inverse_core.request_schema import GeometryConstraints

from .geometry import evaluate_geometry
from .local_interface_contract import DecisionGroup, ResponseAssessment
from .native_corrected import NativeBaselineCorrection


@dataclass(frozen=True)
class CoordinateResponse:
    coordinate: tuple[int, int]
    physical_module_id: Hashable
    radius: float
    temperature_plus: dict[Hashable, float]
    temperature_minus: dict[Hashable, float]
    pressure_plus: float
    pressure_minus: float

    def increments(self, sign: int) -> tuple[dict[Hashable, float], float]:
        if sign not in (-1, 1):
            raise ValueError("A finite move sign must be -1 or +1.")
        return (
            self.temperature_plus if sign > 0 else self.temperature_minus,
            self.pressure_plus if sign > 0 else self.pressure_minus,
        )


def _moved(design: PhysicalDesign, slot: int, axis: int, delta: float) -> PhysicalDesign:
    centers = np.array(design.module_centers, copy=True)
    centers[slot, axis] += float(delta)
    return PhysicalDesign(
        module_centers=centers,
        module_present=np.array(design.module_present, copy=True),
        heat_powers=np.array(design.heat_powers, copy=True),
        module_family_id=design.module_family_id,
    )


def finite_coordinate_responses(
    anchor: NativeBaselineCorrection,
    constraints: GeometryConstraints,
    *,
    radius: float,
) -> tuple[CoordinateResponse, ...]:
    """Probe valid +/- moves with the anchor's one frozen forward topology."""

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Finite coordinate radius must be positive and finite.")
    baseline = anchor.baseline_design
    baseline_model = anchor.model_baseline
    rows: list[CoordinateResponse] = []
    for slot, active in enumerate(baseline.module_present > 0.5):
        if not active:
            continue
        module_id = anchor.module_ids_by_slot[slot]
        if module_id is None:
            raise ValueError("An active coordinate needs a physical module ID.")
        for axis in (0, 1):
            plus = _moved(baseline, slot, axis, radius)
            minus = _moved(baseline, slot, axis, -radius)
            if not (evaluate_geometry(plus, anchor.context, constraints).valid
                    and evaluate_geometry(minus, anchor.context, constraints).valid):
                continue
            plus_out = anchor.evaluate(plus).native_absolute
            minus_out = anchor.evaluate(minus).native_absolute
            rows.append(CoordinateResponse(
                coordinate=(slot, axis),
                physical_module_id=module_id,
                radius=float(radius),
                temperature_plus={
                    key: plus_out.module_temperature_by_id[key] - baseline_model.module_temperature_by_id[key]
                    for key in baseline_model.module_temperature_by_id
                },
                temperature_minus={
                    key: minus_out.module_temperature_by_id[key] - baseline_model.module_temperature_by_id[key]
                    for key in baseline_model.module_temperature_by_id
                },
                pressure_plus=plus_out.pressure_drop - baseline_model.pressure_drop,
                pressure_minus=minus_out.pressure_drop - baseline_model.pressure_drop,
            ))
    return tuple(rows)


def _score(
    baseline_temperature: dict[Hashable, float],
    baseline_pressure: float,
    responses: Sequence[tuple[dict[Hashable, float], float]],
    pressure_limit: float,
) -> tuple[int, float, float]:
    if not responses:
        raise ValueError("A candidate needs at least one finite response.")
    pressure = baseline_pressure + sum(item[1] for item in responses)
    peak = max(
        value + sum(item[0][module_id] for item in responses)
        for module_id, value in baseline_temperature.items()
    )
    return (int(pressure > pressure_limit), max(0.0, pressure - pressure_limit), peak)


def select_receiver_response_groups(
    responses: Sequence[CoordinateResponse],
    anchor: NativeBaselineCorrection,
    *,
    pressure_limit: float,
    near_hot_band: float,
) -> tuple[DecisionGroup | None, DecisionGroup | None, tuple[ResponseAssessment, ...]]:
    """Return best coordinated pair, independent-coordinate control and audit.

    Both selectors see the same finite model response matrix. The coordinated
    pair ranks the *combined* protected outputs. The control takes the best
    single coordinate, then the next distinct coordinate by its own score.
    A mixed-physics label is never inferred from these additive estimates.
    """

    if not np.isfinite(pressure_limit) or pressure_limit <= 0:
        raise ValueError("The original pressure limit must be positive and finite.")
    if not np.isfinite(near_hot_band) or near_hot_band < 0:
        raise ValueError("Near-hot protection band must be nonnegative and finite.")
    baseline = dict(anchor.reference_baseline.module_temperature_by_id)
    baseline_pressure = float(anchor.reference_baseline.pressure_drop)
    hottest = max(baseline.values())
    protected = tuple(str(key) for key, value in baseline.items() if value >= hottest - near_hot_band)
    signed: list[tuple[CoordinateResponse, int, dict[Hashable, float], float, tuple[int, float, float]]] = []
    assessment: list[ResponseAssessment] = []
    for row in responses:
        for sign in (-1, 1):
            changes, pressure = row.increments(sign)
            score = _score(baseline, baseline_pressure, ((changes, pressure),), pressure_limit)
            signed.append((row, sign, changes, pressure, score))
            suffix = "+" if sign > 0 else "-"
            for module_id, value in changes.items():
                assessment.append(ResponseAssessment(
                    receiver_quantity=f"material_peak:{module_id}",
                    changed_coordinates=(f"slot{row.coordinate[0]}:{'xy'[row.coordinate[1]]}:{suffix}",),
                    predicted_increment=float(value),
                    numerical_discrepancy=None,
                    model_error_indicator=None,
                    evidence_scope="model_response",
                    supported_move_scale=row.radius,
                ))
            assessment.append(ResponseAssessment(
                receiver_quantity="pressure_drop",
                changed_coordinates=(f"slot{row.coordinate[0]}:{'xy'[row.coordinate[1]]}:{suffix}",),
                predicted_increment=float(pressure),
                numerical_discrepancy=None,
                model_error_indicator=None,
                evidence_scope="model_response",
                supported_move_scale=row.radius,
            ))
    if len({row.coordinate for row in responses}) < 2:
        return None, None, tuple(assessment)
    by_coordinate = {}
    for item in signed:
        coordinate = item[0].coordinate
        if coordinate not in by_coordinate or item[4] < by_coordinate[coordinate][4]:
            by_coordinate[coordinate] = item
    independently_ranked = sorted(by_coordinate.values(), key=lambda item: (item[4], item[0].coordinate))
    direct = independently_ranked[:2]
    direct_group = DecisionGroup(
        changed_coordinates=tuple(f"slot{item[0].coordinate[0]}:{'xy'[item[0].coordinate[1]]}" for item in direct),
        protected_receivers=protected,
        protected_constraints=("pressure_drop",),
        rationale="direct_sensitivity_control",
        supporting_response_ids=tuple(
            f"finite_model:{item[0].coordinate[0]}:{item[0].coordinate[1]}:{item[1]}" for item in direct
        ),
    )
    pairs = (
        (left, right)
        for left, right in combinations(signed, 2)
        if left[0].coordinate != right[0].coordinate
    )
    best = min(
        pairs,
        key=lambda pair: (
            _score(baseline, baseline_pressure, ((pair[0][2], pair[0][3]), (pair[1][2], pair[1][3])), pressure_limit),
            pair[0][0].coordinate,
            pair[1][0].coordinate,
            pair[0][1],
            pair[1][1],
        ),
    )
    pressure_temperature_complement = any(
        best[index][3] < 0
        and any(best[1 - index][2][module_id] < 0 for module_id in baseline)
        for index in (0, 1)
    )
    shared_receiver = any(
        abs(best[0][2][module_id]) > 1e-6 and abs(best[1][2][module_id]) > 1e-6
        for module_id in baseline if str(module_id) in protected
    )
    rationale = (
        "constraint_coordination" if pressure_temperature_complement
        else "shared_receiver" if shared_receiver
        else "direct_sensitivity_control"
    )
    coordinated = DecisionGroup(
        changed_coordinates=tuple(f"slot{item[0].coordinate[0]}:{'xy'[item[0].coordinate[1]]}" for item in best),
        protected_receivers=protected,
        protected_constraints=("pressure_drop",),
        rationale=rationale,
        supporting_response_ids=tuple(
            f"finite_model:{item[0].coordinate[0]}:{item[0].coordinate[1]}:{item[1]}" for item in best
        ),
    )
    return coordinated, direct_group, tuple(assessment)


__all__ = ["CoordinateResponse", "finite_coordinate_responses", "select_receiver_response_groups"]
