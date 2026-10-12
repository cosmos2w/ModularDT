"""Public Thermal native routing trace and rebuild API contracts."""

from __future__ import annotations

import numpy as np
import torch
from channelthermal.joint_regional import InteractionPreservingThermalAdapter


def _stats():
    return {
        "field_mean_by_channel": np.asarray([1.0, -2.0, 0.5, 3.0, 10.0], dtype=np.float32),
        "field_std_by_channel": np.asarray([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "interface_targets_mean": np.asarray([10.0, 0.0], dtype=np.float32),
        "interface_targets_std": np.asarray([4.0, 5.0], dtype=np.float32),
        "internal_temperature_std": np.asarray([7.0], dtype=np.float32),
    }


def _sample():
    x_axis = (np.arange(8, dtype=np.float32) + 0.5) * (12.0 / 8)
    y_axis = (np.arange(4, dtype=np.float32) + 0.5) * (6.0 / 4)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    return {
        "structure": {
            "module_centers": np.asarray([[2.25, 2.25], [8.25, 3.75]], dtype=np.float32),
            "module_present": np.ones(2, dtype=np.float32),
            "material_params": np.asarray([0.02, 0.01, 0.02, 1.0, 2.0, 0.22], dtype=np.float32),
            "re": np.asarray([50.0], dtype=np.float32),
            "u_in": np.asarray([1.0], dtype=np.float32),
            "domain_length_x": np.asarray([12.0], dtype=np.float32),
            "domain_length_y": np.asarray([6.0], dtype=np.float32),
            "module_source_ids": ("source-a", "source-b"),
            "heat_powers": np.asarray([1.0, 2.0], dtype=np.float32),
        },
        "x_grid": x_grid,
        "y_grid": y_grid,
        "module_internal_query_points": np.asarray([[0.0, 0.0], [0.25, -0.25]], dtype=np.float32),
    }


def _model():
    torch.manual_seed(91)
    return InteractionPreservingThermalAdapter(
        mode="P-R2",
        normalization_stats=_stats(),
        hidden=12,
        message=128,
        regional_anchors=0,
        receiver_tile=16,
        nx=8,
        ny=4,
        environment_nx=4,
        environment_ny=2,
        seed=91,
    )


def test_predict_native_sample_exposes_rebuilt_trace_and_exact_fluid_mask():
    model = _model()
    sample = _sample()

    ordinary = model.predict_native_sample(sample, device="cpu")
    traced = model.predict_native_sample(
        sample,
        device="cpu",
        routing_trace=True,
        native_route_intervention={"kind": "density_reset", "route_types": ["MM"]},
    )

    assert set(ordinary) == {
        "pred_field_grid",
        "pred_interface",
        "pred_internal_temperature",
        "pred_port_condition",
        "initial_port_status",
    }
    assert set(traced) == set(ordinary) | {
        "native_routing_trace",
        "native_routing_trace_metadata",
        "fluid_stencil_valid",
    }
    np.testing.assert_array_equal(traced["pred_field_grid"], ordinary["pred_field_grid"])
    assert traced["fluid_stencil_valid"].shape == sample["x_grid"].shape
    assert traced["fluid_stencil_valid"].all()

    trace = traced["native_routing_trace"]
    assert len(trace) == 6
    assert {entry["round"] for entry in trace} == {1, 2}
    assert all(entry["view_ids"].tolist() == [0, 1] for entry in trace)
    mm_trace = [entry for entry in trace if entry["legacy_route"] == "MM"]
    other_trace = [entry for entry in trace if entry["legacy_route"] != "MM"]
    assert len(mm_trace) == 2
    assert len(other_trace) == 4
    assert all(entry["intervention"] == "density_reset" for entry in mm_trace)
    assert all(entry["intervention"] is None for entry in other_trace)
    assert mm_trace[0]["weighted_message_content"].shape[-2:] == (2, 64)
    metadata = traced["native_routing_trace_metadata"]
    assert metadata["configuration"]["native_routing"]["views"] == 2
    assert metadata["active_routing_semantics"][0] == "native_interaction_routing_semantics_v1"
    assert metadata["output_law_metadata"]["response_output_law"] == "affine"


def test_prepare_native_forwards_trace_and_intervention_to_shared_core():
    model = _model()
    sample = _sample()
    structure = {}
    for key, value in sample["structure"].items():
        if key == "module_source_ids":
            structure[key] = value
        elif key != "heat_powers":
            structure[key] = torch.as_tensor(value, dtype=torch.float32)[None]
    fluid = torch.as_tensor(
        np.stack((sample["x_grid"].reshape(-1), sample["y_grid"].reshape(-1)), axis=-1),
        dtype=torch.float32,
    )[None]

    prepared = model.prepare_native(
        structure,
        fluid,
        routing_trace=True,
        native_route_intervention={"kind": "density_reset", "route_types": ["ME"], "view": 1},
    )

    assert len(prepared.context.native_routing_trace) == 6
    assert prepared.context.native_route_intervention == ("density_reset", ("ME",))
    assert prepared.context.native_route_intervention_view == 1
    me_trace = [
        entry for entry in prepared.context.native_routing_trace
        if entry["legacy_route"] == "ME"
    ]
    assert len(me_trace) == 2
    assert all(entry["intervention"] == "density_reset" for entry in me_trace)
    assert all(entry["intervention_view"] == 1 for entry in me_trace)
