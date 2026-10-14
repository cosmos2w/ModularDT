from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2

from windfarm.content_hypergraph import (
    WIND_CONTENT_COORDINATE_FRAME,
    WIND_CONTENT_ENVIRONMENT_COUNT,
    WindContentHypergraphAdapter,
    WindContentHypergraphAdapterV3,
    WindContentReceivers,
    WindContentSceneInputs,
)
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.training.content_hypergraph_task import (
    WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES,
    WIND_CONTENT_ROLE_CACHE_FULL_BYTES,
    WindContentHypergraphTaskV2,
    _validate_recipe_v2,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import wind_native_reference_evaluate as native_evaluator


class _TargetReadForbidden:
    @property
    def U(self):
        raise AssertionError("Wind v2 scene construction must not read native velocity targets")


def _transforms() -> tuple[VelocityNormalizer, VerticalProfileBaseline]:
    normalizer = VelocityNormalizer(
        mean=np.asarray([100.0, -50.0, 8.0]),
        std=np.asarray([0.3, 0.2, 0.4]),
        safe_std=np.asarray([0.3, 0.2, 0.4]),
        u_ref_mps=9.0,
        source_rows=72,
        seed=42,
    )
    profile = VerticalProfileBaseline(
        bin_centers_D=np.asarray([0.0, 1.0, 2.0]),
        values_mps=np.asarray([[2.0, 0.0, 1.0], [4.0, 1.0, 2.0], [6.0, 2.0, 3.0]]),
        counts=np.asarray([10, 10, 10]),
    )
    return normalizer, profile


def _scene_inputs() -> WindContentSceneInputs:
    count = 2
    environment_coordinates = np.column_stack(
        (np.arange(64), np.arange(64) * 0.1, np.arange(64) * 0.01)
    ).astype(np.float32)
    environment_features = np.arange(64 * 7, dtype=np.float32).reshape(64, 7)
    environment_measure = np.linspace(0.01, 0.64, 64, dtype=np.float64)
    case = SimpleNamespace(
        index=5,
        case="wind-layout-2-wd285",
        layout_index=2,
        wind_direction_deg=285.0,
        module_centers=np.asarray([[0.0, 0.0, 0.875], [10.0, 2.0, 0.875]], dtype=np.float32),
        module_present=np.asarray([1.0, 0.0], dtype=np.float32),
        module_features=np.asarray([[0.5, 0.875], [0.5, 0.875]], dtype=np.float32),
        global_context=np.asarray([0.0, 1.0, 0.0, 2.0 / 30.0, 1.0, -1.0, -1.0, 0.0, 4.0, 2.0, 2.0]),
        support=SimpleNamespace(
            lower_D=np.asarray([-1.0, -2.0, 0.0]),
            upper_D=np.asarray([4.0, 3.0, 2.0]),
            extent_D=np.asarray([5.0, 5.0, 2.0]),
        ),
        environment=SimpleNamespace(
            token_shape=(4, 4, 4),
            coords_D=environment_coordinates,
            features=environment_features,
            weights_D3=environment_measure,
        ),
        run=_TargetReadForbidden(),
    )
    result = WindContentSceneInputs.from_case(case)
    assert result.module_centers_D.shape == (count, 3)
    return result


@pytest.mark.parametrize("architecture", ["A2-D", "A2-H6"])
def test_native_plane_tiles_validate_complete_request_once_and_match_full_decode(monkeypatch, architecture):
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapterV3(normalizer, profile)
    inputs = _scene_inputs()
    scene = adapter.make_scene_batch((inputs,))
    model = ContentHypergraphCoreV2(
        module_input_dim=2, environment_input_dim=7, coordinate_dim=3,
        query_input_dim=7, operating_input_dim=11, output_dim=3,
        architecture=architecture, context_dim=32, num_heads=4, context_depth=2,
        thermal_response_rank=0, local_hidden_dim=32,
    ).eval()
    coordinates = np.array([[.1, .1, .875], [.8, .2, .9], [1.7, .3, 1.], [3., .1, .875]], dtype=np.float32)
    flat = np.array([5, 8, 13, 21], dtype=np.int64)
    support = SimpleNamespace(lower_D=inputs.support_lower_D, upper_D=inputs.support_upper_D,
                              extent_D=inputs.support_extent_D)
    receivers = native_evaluator._receivers_for_tile(adapter, inputs, support, coordinates, flat)
    query = adapter.make_query_batch(scene, receivers)
    with torch.no_grad():
        organization = model.prepare(scene)
        expected = adapter.physical_velocity(model.decode(organization, query).nonlinear_field, query.coordinates)[0][0].numpy()
    validate = model.validate_prepared_state
    checked = []

    def counted_validate(prepared, request):
        checked.append(request.num_queries)
        return validate(prepared, request)

    monkeypatch.setattr(model, "validate_prepared_state", counted_validate)
    actual = native_evaluator.decode_content_query_tiles(
        model, adapter, inputs, scene, organization, coordinates, flat, tile_size=2, support=support,
    )
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
    assert checked == [len(coordinates)]
    with torch.no_grad():
        next(model.parameters()).add_(.01)
    with pytest.raises(ValueError, match="(?i)stale|weight|parameter"):
        native_evaluator.decode_content_query_tiles(
            model, adapter, inputs, scene, organization, coordinates, flat, tile_size=2, support=support,
        )


def test_v3_wind_scene_adds_diameter_and_e64_cell_lengths_without_changing_v1_inputs():
    inputs = _scene_inputs()
    normalizer, profile = _transforms()
    v1 = WindContentHypergraphAdapter(normalizer, profile)
    v3 = WindContentHypergraphAdapterV3(normalizer, profile)
    old_scene = v1.make_scene_batch((inputs,))
    scene = v3.make_scene_batch((inputs,))

    assert scene.coordinate_frame_key == WIND_CONTENT_COORDINATE_FRAME
    torch.testing.assert_close(scene.coordinate_origin, torch.tensor([[-1.0, -2.0, 0.0]]))
    torch.testing.assert_close(scene.coordinate_scale, torch.tensor([[5.0, 5.0, 2.0]]))
    torch.testing.assert_close(scene.local_radius, torch.full((1, 2), 1.5))
    torch.testing.assert_close(
        scene.module_characteristic_lengths,
        torch.ones((1, 2, 3)),
    )
    expected_environment_lengths = torch.tensor([1.25, 1.25, 0.5]).reshape(1, 1, 3).expand(
        1, WIND_CONTENT_ENVIRONMENT_COUNT, 3
    )
    torch.testing.assert_close(scene.environment_characteristic_lengths, expected_environment_lengths)
    for name in (
        "module_features",
        "environment_features",
        "module_coordinates",
        "environment_coordinates",
        "module_mask",
        "environment_mask",
        "module_measure",
        "environment_measure",
        "module_ids",
        "environment_ids",
        "local_radius",
        "coordinate_origin",
        "coordinate_scale",
    ):
        torch.testing.assert_close(getattr(scene, name), getattr(old_scene, name))


def test_v3_physical_transform_uses_profile_plus_scaled_residual_only():
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapterV3(normalizer, profile)
    coords = torch.tensor([[[0.0, 0.0, 0.5], [2.0, 1.0, 1.5]]], dtype=torch.float32)
    normalized = torch.tensor([[[1.0, -2.0, 0.5], [0.0, 0.25, -1.0]]], dtype=torch.float32)
    physical, profile_mps, residual_mps = adapter.physical_velocity(normalized, coords)
    expected_profile = adapter.profile_at(coords)
    expected_residual = normalized * torch.tensor([0.3, 0.2, 0.4]) * 9.0

    torch.testing.assert_close(profile_mps, expected_profile)
    torch.testing.assert_close(residual_mps, expected_residual)
    torch.testing.assert_close(physical, expected_profile + expected_residual)
    assert not torch.allclose(physical, physical + torch.tensor([100.0, -50.0, 8.0]))


def test_v2_provider_dispatches_to_versioned_core_with_bounded_prepare_trace():
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapterV3(normalizer, profile)
    scene = adapter.make_scene_batch((_scene_inputs(),))
    receivers = WindContentReceivers(
        coordinates_D=np.asarray([[[0.0, 0.0, 0.875], [10.0, 2.0, 0.875]]], dtype=np.float32),
        features=np.zeros((1, 2, 7), dtype=np.float32),
        flat_indices=np.asarray([[5, 6]], dtype=np.int64),
        role_ids=np.asarray([[0, 1]], dtype=np.int8),
        scene_keys=scene.scene_key,
        local_batch_indices=np.asarray([0, 0], dtype=np.int64),
        local_query_indices=np.asarray([0, 1], dtype=np.int64),
        local_module_indices=np.asarray([0, 1], dtype=np.int64),
    )
    model = ContentHypergraphCoreV2.from_config(
        {
            "schema": "content_hypergraph_core_v2",
            "module_input_dim": 2,
            "environment_input_dim": 7,
            "coordinate_dim": 3,
            "query_input_dim": 7,
            "operating_input_dim": 11,
            "output_dim": 3,
            "architecture": "A2-H6",
            "context_dim": 128,
            "num_heads": 4,
            "context_depth": 2,
            "num_hyperedges": 6,
            "thermal_response_rank": 0,
            "local_hidden_dim": 128,
            "dropout": 0.0,
            "init_seed": 42,
        }
    )
    # Exercise prediction dispatch without opening a dataset or reading targets.
    provider = object.__new__(WindContentHypergraphTaskV2)
    provider.adapter = adapter
    provider.recipe = {
        "schema": "content_hypergraph_recipe_v2",
        "dataset": "wind",
        "architecture": "A2-H6",
        "seed": 42,
        "model": {
            "context_dim": 128,
            "num_heads": 4,
            "context_depth": 2,
            "num_hyperedges": 6,
            "thermal_response_rank": 0,
            "local_hidden_dim": 128,
            "dropout": 0.0,
        },
    }
    request = {"include": ["module_states"], "max_elements": 512}

    predictions, auxiliary = provider.predict_native(
        model, scene, receivers, execution_mode="A2-H6", trace_request=request
    )

    assert predictions.main_mps.shape == (1, 2, 3)
    assert torch.isfinite(predictions.main_mps).all()
    assert auxiliary["prepared_hyperedge_count"] == 6
    assert auxiliary["organization"].trace["module_states"].shape == (1, 2, 128)
    assert auxiliary["trace_request"] == request
    with pytest.raises(ValueError, match="max_elements"):
        provider.predict_native(
            model,
            scene,
            receivers,
            execution_mode="A2-H6",
            trace_request={"include": ["module_states"], "max_elements": 1},
        )


def test_typed_h6_native_prediction_preserves_law_roundtrip_and_rejects_provider_mode_mismatch():
    normalizer, profile = _transforms()
    adapter = WindContentHypergraphAdapterV3(normalizer, profile)
    inputs = _scene_inputs()
    scene = adapter.make_scene_batch((inputs,))
    coordinates = np.asarray([[[0.0, 0.0, 0.5], [2.0, 1.0, 1.5]]], dtype=np.float32)
    support = SimpleNamespace(
        lower_D=inputs.support_lower_D,
        upper_D=inputs.support_upper_D,
        extent_D=inputs.support_extent_D,
    )
    receivers = WindContentReceivers(
        coordinates_D=coordinates,
        features=adapter.query_features(coordinates[0], support)[None],
        flat_indices=np.asarray([[5, 6]], dtype=np.int64),
        role_ids=np.asarray([[0, 1]], dtype=np.int8),
        scene_keys=scene.scene_key,
        local_batch_indices=np.asarray([0, 0], dtype=np.int64),
        local_query_indices=np.asarray([0, 1], dtype=np.int64),
        local_module_indices=np.asarray([0, 1], dtype=np.int64),
    )
    model_config = {
        "schema": "content_hypergraph_core_v2",
        "module_input_dim": 2,
        "environment_input_dim": 7,
        "coordinate_dim": 3,
        "query_input_dim": 7,
        "operating_input_dim": 11,
        "output_dim": 3,
        "architecture": "A2-H6",
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6,
        "thermal_response_rank": 0,
        "local_hidden_dim": 128,
        "dropout": 0.0,
        "init_seed": 42,
        "query_edge_geometry_mode": "typed_w_centroids_v1",
    }
    model = ContentHypergraphCoreV2.from_config(model_config).eval()
    recipe = {
        "schema": "content_hypergraph_recipe_v2",
        "dataset": "wind",
        "architecture": "A2-H6",
        "seed": 42,
        "model": {
            "context_dim": 128,
            "num_heads": 4,
            "context_depth": 2,
            "num_hyperedges": 6,
            "thermal_response_rank": 0,
            "local_hidden_dim": 128,
            "dropout": 0.0,
            "query_edge_geometry_mode": "typed_w_centroids_v1",
        },
    }
    provider = object.__new__(WindContentHypergraphTaskV2)
    provider.adapter = adapter
    provider.recipe = recipe

    query = adapter.make_query_batch(scene, receivers)
    with torch.no_grad():
        organization = model.prepare(scene)
        decoded = model.decode(organization, query)
        expected_physical = adapter.physical_velocity(decoded.nonlinear_field, query.coordinates)[0]
        predictions, _ = provider.predict_native(model, scene, receivers, execution_mode="A2-H6")
    torch.testing.assert_close(predictions.main_mps, expected_physical, atol=1e-6, rtol=1e-6)

    restored_adapter = WindContentHypergraphAdapterV3.from_state_payload(adapter.state_payload())
    assert restored_adapter.state_payload() == adapter.state_payload()
    restored_model = ContentHypergraphCoreV2.from_config(model.get_config())
    restored_model.load_state_dict(model.state_dict(), strict=True)
    assert restored_model.get_config() == model.get_config()
    assert all(torch.equal(value, restored_model.state_dict()[name]) for name, value in model.state_dict().items())

    default_model_config = dict(model_config)
    default_model_config.pop("query_edge_geometry_mode")
    default_model = ContentHypergraphCoreV2.from_config(default_model_config)
    with pytest.raises(ValueError, match="missing or unsupported fields"):
        provider.predict_native(default_model, scene, receivers, execution_mode="A2-H6")

    default_recipe = dict(recipe)
    default_recipe["model"] = dict(recipe["model"])
    default_recipe["model"].pop("query_edge_geometry_mode")
    provider.recipe = default_recipe
    with pytest.raises(ValueError, match="missing or unsupported fields"):
        provider.predict_native(model, scene, receivers, execution_mode="A2-H6")


def test_v3_state_is_versioned_and_cannot_be_loaded_by_the_v1_adapter():
    normalizer, profile = _transforms()
    v1 = WindContentHypergraphAdapter(normalizer, profile)
    v3 = WindContentHypergraphAdapterV3(normalizer, profile)
    old_payload = v1.state_payload()
    payload = v3.state_payload()

    assert old_payload["schema"] == "wind_content_hypergraph_adapter_v2"
    assert "geometry_contract" not in old_payload
    assert payload["schema"] == "wind_content_hypergraph_adapter_v3"
    assert payload["geometry_contract"]["object_length_is_local_reader_radius"] is False
    assert WindContentHypergraphAdapterV3.from_state_payload(payload).state_payload() == payload
    with pytest.raises(ValueError, match="adapter state schema"):
        WindContentHypergraphAdapter.from_state_payload(payload)
    with pytest.raises(ValueError, match="adapter state schema"):
        WindContentHypergraphAdapterV3.from_state_payload(old_payload)


def _read_recipe(name: str) -> dict:
    path = (
        Path(__file__).parents[2]
        / "src"
        / "config_core"
        / "forward"
        / "content_hypergraph"
        / name
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("name", "architecture", "protocol", "cache_bytes"),
    [
        ("wind_a2-d_development.json", "A2-D", "development", WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES),
        ("wind_a2-h6_development.json", "A2-H6", "development", WIND_CONTENT_ROLE_CACHE_DEVELOPMENT_BYTES),
        ("wind_a2-h6_full1000.json", "A2-H6", "full_train1000", WIND_CONTENT_ROLE_CACHE_FULL_BYTES),
    ],
)
def test_v2_wind_recipe_files_resolve_the_sealed_protocol(name, architecture, protocol, cache_bytes):
    recipe, resolved_architecture, microbatch = _validate_recipe_v2(_read_recipe(name))
    assert recipe["schema"] == "content_hypergraph_recipe_v2"
    assert resolved_architecture == architecture
    assert recipe["protocol"] == protocol
    assert recipe["seed"] == 42 and recipe["primary_queries"] == 4096
    assert recipe["effective_batch_size"] == 24 and microbatch == 12
    assert recipe["catalogue_cache_ram_bytes"] == cache_bytes
    assert recipe["optimizer_bias_norm_exclusion"] is True
    if protocol == "full_train1000":
        assert recipe["total_epochs"] == 5000 and recipe["approved_stop_after"] == 1000
    else:
        assert recipe["total_epochs"] == 2500


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema": "content_hypergraph_recipe_v1"}, "recipe_v2"),
        ({"architecture": "A-H6"}, "A2-D and A2-H6"),
        ({"optimizer_bias_norm_exclusion": False}, "optimizer_bias_norm_exclusion"),
        ({"primary_queries": 8192}, "4096"),
        ({"microbatch_size": 0}, "microbatch_size"),
    ],
)
def test_v2_wind_recipe_rejects_version_or_contract_drift(change, message):
    recipe = _read_recipe("wind_a2-d_development.json")
    recipe.update(change)
    with pytest.raises(ValueError, match=message):
        _validate_recipe_v2(recipe)
