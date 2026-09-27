"""Decision coordination is distinct from a measured mixed physical term."""

from __future__ import annotations

from types import SimpleNamespace

from channelthermal.inverse.decision_groups import (
    CoordinateResponse,
    select_receiver_response_groups,
)


def test_pressure_enabling_coordinate_can_pair_with_peak_reducing_coordinate() -> None:
    anchor = SimpleNamespace(reference_baseline=SimpleNamespace(
        module_temperature_by_id={"hot": 100.0, "cool": 95.0},
        pressure_drop=9.5,
    ))
    responses = (
        CoordinateResponse(
            coordinate=(0, 0), physical_module_id="hot", radius=0.1,
            temperature_plus={"hot": -5.0, "cool": 0.0},
            temperature_minus={"hot": 2.0, "cool": 0.0},
            pressure_plus=1.0, pressure_minus=0.0,
        ),
        CoordinateResponse(
            coordinate=(1, 0), physical_module_id="cool", radius=0.1,
            temperature_plus={"hot": 0.0, "cool": 0.0},
            temperature_minus={"hot": 0.0, "cool": 0.0},
            pressure_plus=-1.0, pressure_minus=0.5,
        ),
        CoordinateResponse(
            coordinate=(1, 1), physical_module_id="cool", radius=0.1,
            temperature_plus={"hot": -1.0, "cool": 0.0},
            temperature_minus={"hot": 1.0, "cool": 0.0},
            pressure_plus=0.0, pressure_minus=0.0,
        ),
    )
    grouped, direct, assessments = select_receiver_response_groups(
        responses, anchor, pressure_limit=10.0, near_hot_band=5.0,
    )
    assert grouped is not None and direct is not None
    assert grouped.changed_coordinates == ("slot0:x", "slot1:x")
    assert grouped.rationale == "constraint_coordination"
    assert direct.changed_coordinates != grouped.changed_coordinates
    assert all(item.evidence_scope == "model_response" for item in assessments)
    assert all(item.numerical_discrepancy is None for item in assessments)
