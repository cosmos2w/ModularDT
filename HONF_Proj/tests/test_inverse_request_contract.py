from __future__ import annotations

import copy

import numpy as np
import pytest

from honf_inverse_core.normalization import ScalarStats, VectorStats, fit_scalar
from honf_inverse_core.request_schema import RELATION_TO_ID
from channelthermal.inverse.context import CONTEXT_FEATURE_NAMES, parse_context
from channelthermal.inverse.request import make_request_codec
from channelthermal.inverse.vocabulary import REQUEST_TYPES


def _stats() -> dict[str, ScalarStats]:
    return {name: ScalarStats(mean=100.0 + index, std=10.0 + index, count=20) for index, name in enumerate(REQUEST_TYPES)}


def _request() -> dict:
    return {
        "schema_name": "thermalchannel_inverse_request",
        "schema_version": 1,
        "tokens": [
            {
                "request_type": "pressure_drop",
                "relation": "upper_bound",
                "target": 12.5,
                "tolerance": 0.4,
                "priority": 2,
                "region": None,
                "active": True,
            },
            {
                "request_type": "regional_temperature_max",
                "relation": "target_range",
                "target": 310.0,
                "target_range": [305.0, 315.0],
                "tolerance": 1.5,
                "priority": 3,
                "weight": 2.5,
                "region": [0.6, 0.1, 0.9, 0.8],
                "active": True,
            },
        ],
        "geometry_constraints": {
            "module_count_min": 2,
            "module_count_max": 8,
            "minimum_center_distance": 1.1,
            "wall_clearance": 0.05,
            "inlet_clearance": 0.25,
            "outlet_clearance": 0.25,
            "total_heat_range": [80.0, 120.0],
        },
    }


def _context() -> dict:
    return {
        "schema_name": "thermalchannel_inverse_context",
        "schema_version": 1,
        "re": 100.0,
        "u_in": 1.0,
        "nu": 1.0e-5,
        "solid_alpha": 1.0e-5,
        "fluid_alpha": 2.0e-5,
        "solid_k": 10.0,
        "fluid_k": 0.6,
        "module_radius": 0.45,
        "domain_length_x": 12.0,
        "domain_length_y": 4.0,
    }


def test_request_json_materialization_and_tensor_round_trip() -> None:
    codec = make_request_codec(_stats())
    request = codec.parse(_request())
    reparsed = codec.parse(request.to_dict())
    assert reparsed == request
    assert request.tokens[0].weight == 1.0  # Derived once from priority; not multiplied again.
    tensors = codec.tensorize(request)
    assert tensors.type_id.shape == (4,)
    assert tensors.relation_id[0] == RELATION_TO_ID["upper_bound"]
    assert tensors.range_mask.tolist() == [0, 1, 0, 0]
    assert tensors.region_mask.tolist() == [0, 1, 0, 0]
    assert tensors.active_mask.tolist() == [1, 1, 0, 0]
    summary = codec.summarize(request, context={"domain_length_x": 12.0, "domain_length_y": 4.0})
    assert "pressure_drop" in summary
    assert "region=(7.2,0.4)-(10.8,3.2)" in summary


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(unknown=1), "Unknown request keys"),
        (lambda value: value["tokens"][0].update(request_type="cooling"), "unsupported"),
        (lambda value: value["tokens"][0].update(relation="maximize"), "unsupported"),
        (lambda value: value["tokens"][0].update(target=None), "requires target"),
        (lambda value: value["tokens"][0].update(region=[0, 0, 1, 1]), "nonregional"),
        (lambda value: value["tokens"].append(copy.deepcopy(value["tokens"][0])), "repeat"),
        (lambda value: value["tokens"][0].update(tolerance=float("nan")), "finite"),
    ],
)
def test_request_validation_rejects_malformed_payloads(mutation, match: str) -> None:
    payload = _request()
    mutation(payload)
    with pytest.raises((ValueError, TypeError), match=match):
        make_request_codec(_stats()).parse(payload)


def test_request_normalized_values_are_checked() -> None:
    payload = _request()
    payload["tokens"][0]["normalized_target"] = 99.0
    with pytest.raises(ValueError, match="inconsistent"):
        make_request_codec(_stats()).parse(payload)


def test_context_and_normalization_contracts() -> None:
    context = parse_context(_context())
    assert context.feature_names == CONTEXT_FEATURE_NAMES
    assert context.as_mapping()["module_radius"] == pytest.approx(0.45)
    vector = VectorStats(CONTEXT_FEATURE_NAMES, np.zeros(10), np.ones(10), 4)
    assert np.allclose(vector.denormalize(vector.normalize(context.vector)), context.vector)
    scalar = fit_scalar([1.0, 1.5, 2.0])
    assert scalar.std == pytest.approx(np.std([1.0, 1.5, 2.0]))


def test_context_rejects_invalid_physical_values() -> None:
    payload = _context()
    payload["nu"] = 0.0
    with pytest.raises(ValueError, match="nu must be positive"):
        parse_context(payload)
