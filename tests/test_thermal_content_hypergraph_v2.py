"""CPU contracts for the versioned Thermal v2 adapter and physical laws."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import pytest
import torch
from channelthermal.training import content_hypergraph_task_v2 as thermal_task_v2

from channelthermal.training.content_hypergraph_task import (
    THERMAL_CONTENT_PROVIDER_SCHEMA,
    THERMAL_CONTENT_PROVIDER_V2_SCHEMA,
    ThermalContentHypergraphTask,
)
from channelthermal.training.content_hypergraph_task_v2 import ThermalContentHypergraphTaskV2
from channelthermal.content_hypergraph import (
    THERMAL_CONTENT_ADAPTER_V2_SCHEMA,
    THERMAL_CONTENT_COORDINATE_FRAME_V2,
    ThermalContentHypergraphAdapter,
    ThermalContentHypergraphAdapterV2,
)
from channelthermal.flow_curl import physical_uvp_native_curl
from channelthermal.native_observations import interpolate_native


def _stats() -> dict[str, np.ndarray]:
    return {
        "field_mean_by_channel": np.asarray([1.0, -2.0, 0.5, 3.0, 10.0], dtype=np.float32),
        "field_std_by_channel": np.asarray([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "interface_targets_mean": np.asarray([10.0, 0.0], dtype=np.float32),
        "interface_targets_std": np.asarray([4.0, 5.0], dtype=np.float32),
        "internal_temperature_std": np.asarray([7.0], dtype=np.float32),
    }


def _core_config(architecture: str, query_edge_geometry_mode: str | None = None) -> dict[str, object]:
    result = {
        "schema": "content_hypergraph_core_v2",
        "architecture": architecture,
        "module_input_dim": 8,
        "environment_input_dim": 8,
        "coordinate_dim": 2,
        "query_input_dim": 0,
        "operating_input_dim": 14,
        "output_dim": 3,
        "context_dim": 32,
        "num_heads": 4,
        "context_depth": 1,
        "num_hyperedges": 6,
        "thermal_response_rank": 4,
        "local_hidden_dim": 32,
        "dropout": 0.0,
        "init_seed": 7,
    }
    if query_edge_geometry_mode is not None:
        result["query_edge_geometry_mode"] = query_edge_geometry_mode
    return result


def _structure() -> dict[str, torch.Tensor]:
    return {
        "module_centers": torch.tensor([[[2.0, 2.0], [7.5, 4.0]]]),
        "module_present": torch.ones(1, 2),
        "module_source_ids": torch.tensor([[31, 47]], dtype=torch.long),
        "material_params": torch.tensor([[0.02, 0.01, 0.02, 1.0, 2.0, 0.22]]),
        "re": torch.tensor([[50.0]]),
        "u_in": torch.ones(1, 1),
        "domain_length_x": torch.tensor([[12.0]]),
        "domain_length_y": torch.tensor([[6.0]]),
    }


def _receivers() -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(6, dtype=torch.float32) + 0.5,
        (torch.arange(8, dtype=torch.float32) + 0.5) * 1.5,
        indexing="ij",
    )
    fluid = torch.stack((xx.reshape(-1), yy.reshape(-1)), -1)[None]
    local = torch.tensor([[[[0.0, 0.0], [0.35, 0.0]], [[0.0, 0.0], [0.35, 0.0]]]])
    return fluid, local


def _model(
    architecture: str = "A2-H6", query_edge_geometry_mode: str | None = None,
) -> ThermalContentHypergraphAdapterV2:
    torch.manual_seed(11)
    return ThermalContentHypergraphAdapterV2(
        _stats(), core_config=_core_config(architecture, query_edge_geometry_mode), nx=8, ny=6,
    )


def _prepared(model: ThermalContentHypergraphAdapterV2):
    fluid, local = _receivers()
    return model.prepare_native(
        _structure(), fluid, local_query_points=local, ntheta=16, scene_key=("thermal-v2-test",)
    )


def test_v1_payload_and_config_remain_independent_of_v2_construction():
    v1 = ThermalContentHypergraphAdapter(
        _stats(),
        nx=8,
        ny=6,
        core_config={
            "architecture": "A-H6",
            "context_dim": 32,
            "num_heads": 4,
            "context_depth": 1,
            "num_hyperedges": 6,
            "thermal_response_rank": 4,
            "local_hidden_dim": 16,
            "init_seed": 7,
        },
    )
    config_before = v1.get_config()
    payload_before = v1.state_payload()
    assert config_before["schema"] == "thermal_content_hypergraph_adapter_v1"
    assert config_before["coordinate_frame"] == "thermal_physical_rectangular_xy_v1"
    _model("A2-D")
    assert v1.get_config() == config_before
    assert v1.state_payload() == payload_before
    restored = ThermalContentHypergraphAdapter.from_state_payload(
        payload_before, state_dict=v1.state_dict(), device="cpu"
    )
    assert restored.get_config() == config_before


def test_v2_provider_keeps_the_fixed_sampling_and_supervision_methods():
    assert THERMAL_CONTENT_PROVIDER_SCHEMA == "thermal_content_hypergraph_provider_v1"
    assert THERMAL_CONTENT_PROVIDER_V2_SCHEMA == "thermal_content_hypergraph_provider_v2"
    for name in (
        "epoch_cases",
        "_response_sample",
        "make_batch",
        "_role_losses",
        "_response_loss",
        "loss_terms",
        "validation_batches",
        "native_evaluation_batches",
    ):
        assert getattr(ThermalContentHypergraphTaskV2, name) is getattr(ThermalContentHypergraphTask, name)


def test_thermal_builder_allows_only_the_typed_h6_mode_before_data_binding(monkeypatch):
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "src/config_core/forward/content_hypergraph/thermal_a2-h6_r1_centroids_development.json"
    )
    recipe = json.loads(recipe_path.read_text())

    def stop_at_binding(*args, **kwargs):
        raise RuntimeError("reached data binding")

    monkeypatch.setattr(thermal_task_v2, "_load_data_binding", stop_at_binding)
    with pytest.raises(RuntimeError, match="reached data binding"):
        thermal_task_v2.build_thermal_content_hypergraph_task_v2(recipe)

    invalid_mode = {**recipe, "model": {**recipe["model"], "query_edge_geometry_mode": "unknown"}}
    with pytest.raises(ValueError, match="typed centroid geometry"):
        thermal_task_v2.build_thermal_content_hypergraph_task_v2(invalid_mode)

    direct = {**recipe, "architecture": "A2-D"}
    with pytest.raises(ValueError, match="only for A2-H6"):
        thermal_task_v2.build_thermal_content_hypergraph_task_v2(direct)


def test_v2_scene_has_explicit_frame_object_and_environment_lengths():
    model = _model("A2-D")
    scene = model.scene_batch(_structure(), scene_key=("geometry",))
    torch.testing.assert_close(scene.coordinate_origin, torch.zeros((1, 2)))
    torch.testing.assert_close(scene.coordinate_scale, torch.tensor([[12.0, 6.0]]))
    torch.testing.assert_close(
        scene.module_characteristic_lengths,
        torch.full((1, 2, 2), 0.22),
    )
    torch.testing.assert_close(
        scene.environment_characteristic_lengths,
        torch.tensor([0.5, 0.75]).view(1, 1, 2).expand(1, 192, 2),
    )
    torch.testing.assert_close(scene.local_radius, torch.full((1, 2), 0.66))
    torch.testing.assert_close(scene.module_features[..., 2], torch.full((1, 2), 0.22 / 12.0))
    torch.testing.assert_close(scene.module_features[..., 3], torch.full((1, 2), 0.22 / 6.0))
    assert scene.coordinate_frame_key == THERMAL_CONTENT_COORDINATE_FRAME_V2
    assert scene.environment_features.shape == (1, 192, 8)


def test_typed_centroids_are_valid_native_coordinates_inside_the_thermal_frame():
    model = _model("A2-H6", "typed_w_centroids_v1")
    scene = model.scene_batch(_structure(), scene_key=("typed-centroid-native",))
    organization = model.core.prepare(scene)
    extent = scene.coordinate_scale[:, None, :]
    for name in ("module", "environment"):
        centroids = getattr(organization, f"{name}_content_centroids")
        valid = getattr(organization, f"{name}_content_valid")
        assert torch.isfinite(centroids).all()
        assert valid.dtype == torch.bool and valid.all()
        assert torch.all(centroids >= 0.0)
        assert torch.all(centroids <= extent)


@pytest.mark.parametrize("query_edge_geometry_mode", [None, "typed_w_centroids_v1"])
def test_v2_native_flow_temperature_curl_and_q_follow_thermal_laws(query_edge_geometry_mode):
    model = _model("A2-H6", query_edge_geometry_mode).eval()
    prepared = _prepared(model)
    zero_heat = torch.zeros((1, 2))
    heat = torch.tensor([[0.4, 0.8]], requires_grad=True)
    zero = model.apply_native(prepared, zero_heat)
    prediction = model.apply_native(prepared, heat)
    doubled = model.apply_native(prepared, 2.0 * heat)

    assert torch.count_nonzero(zero["pred_field"][..., 4]) == 0
    assert torch.count_nonzero(zero["pred_interface"]) == 0
    torch.testing.assert_close(prediction["pred_field"][..., :4], zero["pred_field"][..., :4], rtol=0, atol=0)
    torch.testing.assert_close(
        doubled["pred_field"][..., 4], 2.0 * prediction["pred_field"][..., 4], rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        prediction["pred_interface"][..., 0],
        0.5 * doubled["pred_interface"][..., 0],
        rtol=2e-6,
        atol=2e-6,
    )

    raw = model.core.decode(prepared.organization, prepared.query_batch, controls=heat)
    expected_flow = physical_uvp_native_curl(
        raw.nonlinear_field, prepared.curl_stencil, model.field_mean, model.field_std
    )
    torch.testing.assert_close(prediction["pred_field"][..., :4], expected_flow, rtol=0, atol=0)
    torch.testing.assert_close(
        prediction["pred_field"][..., 4], prediction["fluid_temperature"][..., 0], rtol=0, atol=0
    )

    physical_temperature = (raw.thermal_temperature[..., None] * model.field_std[4]).detach()
    surface = interpolate_native(physical_temperature, prepared.stencils["surface"]).reshape(1, 2, 16, 1)
    outside = interpolate_native(physical_temperature, prepared.stencils["outside"]).reshape(1, 2, 16, 1)
    delta = prepared.interface_delta[:, None, :, None]
    conductivity = prepared.interface_conductivity[:, None, :, None]
    expected_q = -conductivity / delta * (outside - surface)
    torch.testing.assert_close(prediction["pred_interface"][..., 1:2], expected_q, rtol=0, atol=0)

    flow_heat_gradient = torch.autograd.grad(
        prediction["pred_field"][..., :4].square().sum(), heat, retain_graph=True, allow_unused=True
    )[0]
    assert flow_heat_gradient is None or torch.count_nonzero(flow_heat_gradient) == 0
    temperature_gradient = torch.autograd.grad(
        prediction["fluid_temperature"].square().sum(), heat, allow_unused=True
    )[0]
    assert temperature_gradient is not None
    assert torch.isfinite(temperature_gradient).all()
    assert torch.count_nonzero(temperature_gradient) > 0


@pytest.mark.parametrize("query_edge_geometry_mode", [None, "typed_w_centroids_v1"])
def test_v2_checkpoint_payload_replays_all_adapter_transforms(query_edge_geometry_mode):
    model = _model("A2-H6", query_edge_geometry_mode).eval()
    prepared = _prepared(model)
    expected = model.apply_native(prepared, torch.tensor([[0.4, 0.8]]))
    payload = model.state_payload()
    assert payload["schema"] == THERMAL_CONTENT_ADAPTER_V2_SCHEMA
    assert payload["core"]["schema"] == "content_hypergraph_core_v2"
    if query_edge_geometry_mode is None:
        assert "query_edge_geometry_mode" not in payload["core"]
    else:
        assert payload["core"]["query_edge_geometry_mode"] == query_edge_geometry_mode
    restored = ThermalContentHypergraphAdapterV2.from_state_payload(
        payload, state_dict=model.state_dict(), device="cpu"
    ).eval()
    actual = restored.apply_native(_prepared(restored), torch.tensor([[0.4, 0.8]]))
    torch.testing.assert_close(actual["pred_field"], expected["pred_field"], rtol=0, atol=0)
    torch.testing.assert_close(actual["pred_interface"], expected["pred_interface"], rtol=0, atol=0)
    torch.testing.assert_close(
        actual["pred_internal_temperature"], expected["pred_internal_temperature"], rtol=0, atol=0
    )


def test_v2_native_work_counters_come_from_the_executed_decode_and_increment():
    model = _model("A2-H6").eval()
    prepared = _prepared(model)
    heat = torch.tensor([[0.4, 0.8]])

    expected = model.apply_native(prepared, heat)
    actual, primary_counts = model.apply_native_with_work_counts(prepared, heat)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert primary_counts["receiver_rows"] == prepared.query_batch.num_queries
    assert primary_counts["edge_rows"] > 0
    assert primary_counts["local_contraction_rows"] > 0

    response, response_counts = model.apply_native_with_work_counts(prepared, heat, increment=True)
    assert response_counts["receiver_rows"] == prepared.query_batch.num_queries
    assert response_counts["edge_rows"] > 0
    assert response_counts["field_head_rows"] == 0
    assert torch.count_nonzero(response["pred_field"][..., :4]) == 0
