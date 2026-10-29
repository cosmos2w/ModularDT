from __future__ import annotations

import numpy as np
import pytest

from honf_inverse_core.contracts import PhysicalDesign
from honf_inverse_core.request_schema import GeometryConstraints
from channelthermal.inverse.context import parse_context
from channelthermal.inverse.functionals import (
    evaluate_nonregional_functionals,
    evaluate_regional_functional,
    module_and_fluid_masks,
)
from channelthermal.inverse.geometry import canonicalize_design, decode_generated_design, evaluate_geometry


def _context():
    return parse_context(
        {
            "schema_name": "thermalchannel_inverse_context",
            "schema_version": 1,
            "re": 100.0,
            "u_in": 1.0,
            "nu": 1.0e-5,
            "solid_alpha": 1.0e-5,
            "fluid_alpha": 2.0e-5,
            "solid_k": 10.0,
            "fluid_k": 0.6,
            "module_radius": 0.1,
            "domain_length_x": 4.0,
            "domain_length_y": 2.0,
        }
    )


def _design() -> PhysicalDesign:
    return PhysicalDesign(
        module_centers=np.asarray([[3.0, 1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        module_present=np.asarray([1, 0, 1], dtype=np.float32),
        heat_powers=np.asarray([20.0, 99.0, 10.0], dtype=np.float32),
    )


def _constraints() -> GeometryConstraints:
    return GeometryConstraints(
        module_count_min=2,
        module_count_max=3,
        minimum_center_distance=1.0,
        wall_clearance=0.2,
        inlet_clearance=0.5,
        outlet_clearance=0.5,
        total_heat_range=(25.0, 35.0),
    )


def test_canonical_design_is_active_first_and_reversible() -> None:
    source = _design()
    view = canonicalize_design(source, _context())
    assert view.canonical_to_source.tolist() == [2, 0, 1]
    assert view.design.module_present.tolist() == [1.0, 1.0, 0.0]
    assert np.allclose(view.design.module_centers[:2], [[1.0, 1.0], [3.0, 1.0]])
    assert np.allclose(view.design.module_centers[2], 0.0)
    recovered_centers = np.zeros_like(source.module_centers)
    recovered_centers[view.canonical_to_source] = view.design.module_centers
    assert np.allclose(recovered_centers[source.module_present > 0.5], source.module_centers[source.module_present > 0.5])


def test_exact_geometry_values_and_invalid_margin() -> None:
    result = evaluate_geometry(_design(), _context(), _constraints())
    assert result.valid
    assert result.pair_distance_defined
    assert result.actual_raw.tolist() == pytest.approx([2.0, 2.0, 0.9, 0.9, 0.9, 30.0])
    invalid = GeometryConstraints(
        module_count_min=2,
        module_count_max=3,
        minimum_center_distance=2.5,
        wall_clearance=0.2,
        inlet_clearance=0.5,
        outlet_clearance=0.5,
        total_heat_range=(25.0, 35.0),
    )
    failed = evaluate_geometry(_design(), _context(), invalid)
    assert not failed.valid
    assert failed.violation_raw[2] == pytest.approx(0.5)


def test_exact_supported_functionals_on_hand_computed_grid() -> None:
    x_line = np.arange(5, dtype=np.float32)
    y_line = np.arange(3, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y_line, x_line, indexing="ij")
    field = np.zeros((3, 5, 5), dtype=np.float32)
    field[..., 2] = x_grid
    field[..., 4] = x_grid + y_grid
    internal = np.asarray([[[2.0], [5.0]], [[0.0], [0.0]], [[1.0], [3.0]]], dtype=np.float32)
    values = evaluate_nonregional_functionals(
        pred_field_grid=field,
        x_grid=x_grid,
        y_grid=y_grid,
        channel_order=["u", "v", "p", "omega", "temperature"],
        design=_design(),
        context=_context(),
        pred_internal_temperature=internal,
    )
    assert values["environment_temperature_max"].value == pytest.approx(6.0)
    assert values["pressure_drop"].value == pytest.approx(-4.0)
    assert values["outlet_temperature_nonuniformity"].value == pytest.approx(np.std([4.0, 5.0, 6.0]))
    assert values["internal_temperature_max"].value == pytest.approx(5.0)
    assert values["internal_temperature_spread"].value == pytest.approx(2.0)
    regional = evaluate_regional_functional(
        "regional_temperature_mean",
        [0.5, 0.0, 1.0, 1.0],
        pred_field_grid=field,
        x_grid=x_grid,
        y_grid=y_grid,
        channel_order=["u", "v", "p", "omega", "temperature"],
        design=_design(),
        context=_context(),
    )
    _, fluid = module_and_fluid_masks(x_grid, y_grid, _design(), _context())
    selection = fluid & (x_grid >= 2.0)
    assert regional.value == pytest.approx(float(np.mean((x_grid + y_grid)[selection])))


def test_generated_endpoint_decode_is_analytically_geometry_valid() -> None:
    constraints = GeometryConstraints(
        module_count_min=3,
        module_count_max=3,
        minimum_center_distance=0.8,
        wall_clearance=0.15,
        inlet_clearance=0.25,
        outlet_clearance=0.25,
        total_heat_range=(5.0, 7.0),
    )
    layout = np.asarray(
        [[0.9, 0.0, -5.0], [0.0, 1.0, 2.0], [0.5, 0.4, 1.0]], dtype=np.float32
    )
    design = decode_generated_design(
        layout,
        [1, 1, 1],
        _context(),
        constraints,
        heat_mean=2.0,
        heat_std=1.0,
    )
    result = evaluate_geometry(design, _context(), constraints)
    assert result.valid
    assert 5.0 <= result.actual_raw[-1] <= 7.0


def test_valid_normalized_training_layout_passes_through_decode() -> None:
    layout = np.asarray(
        [[0.25, 0.5, 10.0], [0.75, 0.5, 20.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    design = decode_generated_design(
        layout,
        [1, 1, 0],
        _context(),
        _constraints(),
        heat_mean=0.0,
        heat_std=1.0,
    )
    np.testing.assert_allclose(design.module_centers[:2], [[1.0, 1.0], [3.0, 1.0]])
    np.testing.assert_allclose(design.heat_powers[:2], [10.0, 20.0])
