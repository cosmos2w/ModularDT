from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from honf_runtime.unified_training import SamplingKey, _sampling_dataset_id
from windfarm.content_hypergraph import WindContentSceneInputs
from windfarm.training.content_hypergraph_task import (
    WindContentHypergraphTask,
    WindContentHypergraphTaskV2,
    WindFormalA2DTaskV1,
    WindSourceReadTaskV1,
    _validate_source_read_wind_recipe_v1,
    _validate_wind_v2_core_binding,
)


RECIPE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src/config_core/forward/content_hypergraph"
)


def _recipe(name: str) -> dict:
    return json.loads((RECIPE_ROOT / name).read_text(encoding="utf-8"))


def _scene_inputs(row: int = 7) -> WindContentSceneInputs:
    return WindContentSceneInputs(
        row_index=row,
        case_name=f"wind-{row}",
        layout_index=2,
        wind_direction_deg=285.0,
        module_centers_D=np.zeros((1, 3), dtype=np.float32),
        module_present=np.ones(1, dtype=np.float32),
        module_features=np.ones((1, 2), dtype=np.float32),
        module_ids=np.zeros(1, dtype=np.int64),
        operating_features=np.zeros(11, dtype=np.float32),
        support_lower_D=np.zeros(3, dtype=np.float32),
        support_upper_D=np.ones(3, dtype=np.float32),
        support_extent_D=np.ones(3, dtype=np.float32),
        environment_coordinates_D=np.zeros((64, 3), dtype=np.float32),
        environment_features=np.zeros((64, 7), dtype=np.float32),
        environment_measure_D3=np.ones(64, dtype=np.float32),
    )


class _SampleCase:
    def __init__(self, row: int) -> None:
        self.index = row
        self.case = f"wind-{row}"
        self.layout_index = 2
        self.wind_direction_deg = 285.0
        self.module_present = np.ones(1, dtype=np.float32)
        self.support = object()
        self.sample_calls: list[tuple[int, str, int]] = []

    def sample_queries(self, count: int, rng: np.random.Generator, *, mode: str):
        marker = int(rng.integers(0, 2**31))
        self.sample_calls.append((count, mode, marker))
        coordinates = np.zeros((count, 3), dtype=np.float32)
        coordinates[:, 0] = marker % 1000
        return SimpleNamespace(
            flat_indices=np.full(count, marker, dtype=np.int64),
            coords_D=coordinates,
            velocity_mps=np.full((count, 3), marker / 1000.0, dtype=np.float32),
        )


def _provider_without_data(provider_type, protocol: str):
    provider = object.__new__(provider_type)
    row = 7
    manifest_sha = "a" * 64
    identity = {"dataset": "WindFarm", "subset_manifest_sha256": manifest_sha}
    expected_dataset_id = _sampling_dataset_id(identity)
    case = _SampleCase(row)
    scene_inputs = _scene_inputs(row)
    provider.view = SimpleNamespace(run=lambda selected: case)
    provider.adapter = SimpleNamespace(
        scene_inputs=lambda ignored: scene_inputs,
        query_features=lambda coordinates, support: np.zeros((len(coordinates), 7), dtype=np.float32),
        packed_local_pair_indices=lambda inputs, coordinates: (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        ),
    )
    provider.recipe = {"protocol": protocol}
    provider.sampling_seed = 42
    provider.seed = 42
    provider.primary_queries = 8192
    provider.validation_query_count = provider.VALIDATION_QUERY_COUNT
    provider.max_microbatch_cases = 8
    provider.validation_rows = np.asarray([row], dtype=np.int64)
    provider._validation_cache = {}
    provider.sampling_version = SamplingKey.CASE_EPOCH_VERSION
    provider.sampling_dataset_id = expected_dataset_id
    provider._train_row_set = {row}
    provider._validation_row_set = {row}
    provider.training_samples = 0
    provider.validation_samples = 0
    provider.role_query_counts = {"volume": 6144, "hub_band": 2048}
    provider.validation_role_query_counts = {"volume": 8192, "hub_band": 2048}
    provider.identity_payload = lambda: identity
    provider._test_expected_dataset_id = expected_dataset_id
    provider._test_case = case
    return provider


def test_wind_source_read_recipes_bind_formal_objective_and_distinct_scopes():
    d0 = _recipe("wind_a2-d_source_read_development_d0_v1.json")
    dscale = _recipe("wind_a2-d_source_read_development_d_scale_v1.json")
    full = _recipe("wind_a2-d_source_read_full1000_d_scale_v1.json")

    assert _validate_source_read_wind_recipe_v1(d0)[0] == d0
    assert _validate_source_read_wind_recipe_v1(dscale)[0] == dscale
    assert _validate_source_read_wind_recipe_v1(full)[0] == full
    assert d0["seed"] == dscale["seed"] == full["seed"] == 0
    assert d0["sampling_seed"] == dscale["sampling_seed"] == full["sampling_seed"] == 42
    assert d0["primary_queries"] == dscale["primary_queries"] == full["primary_queries"] == 8192
    assert dscale["model"]["source_value_conditioning"] == "source_scale"
    assert full["approved_stop_after"] == 1000 < full["total_epochs"]
    assert full["execution"]["monitor_every"] == 5
    assert full["execution"]["checkpoint_epochs"] == list(range(100, 1001, 100))
    assert dscale["execution"]["monitor_every"] == 100
    assert dscale["execution"]["checkpoint_epochs"] == list(range(100, 2501, 100))


@pytest.mark.parametrize(
    ("recipe_name", "expected_conditioning"),
    [
        ("wind_a2-d_source_read_development_d0_v1.json", "none"),
        ("wind_a2-d_source_read_development_d_scale_v1.json", "source_scale"),
    ],
)
def test_source_read_recipe_matches_core_config_without_serializing_defaults(
    recipe_name: str, expected_conditioning: str
):
    recipe = _recipe(recipe_name)
    model_values = recipe["model"]
    config = {
        "schema": "content_hypergraph_core_v2",
        "module_input_dim": 2,
        "environment_input_dim": 7,
        "coordinate_dim": 3,
        "query_input_dim": 7,
        "operating_input_dim": 11,
        "output_dim": 3,
        "architecture": "A2-D",
        "context_dim": model_values["context_dim"],
        "num_heads": model_values["num_heads"],
        "context_depth": model_values["context_depth"],
        "num_hyperedges": model_values["num_hyperedges"],
        "thermal_response_rank": 0,
        "local_hidden_dim": model_values["local_hidden_dim"],
        "dropout": model_values["dropout"],
        "init_seed": recipe["seed"],
    }
    if model_values.get("reader_execution_backend") != "legacy":
        config["reader_execution_backend"] = model_values["reader_execution_backend"]
    if model_values.get("source_value_conditioning") != "none":
        config["source_value_conditioning"] = model_values["source_value_conditioning"]

    model = ContentHypergraphCoreV2.from_config(config)
    model_config = model.get_config()
    _validate_wind_v2_core_binding(recipe, model_config)

    assert model_config.get("source_value_conditioning", "none") == expected_conditioning
    if expected_conditioning == "none":
        assert "reader_execution_backend" not in model_config
        assert "source_value_conditioning" not in model_config


@pytest.mark.parametrize(
    ("recipe_name", "mutate"),
    [
        (
            "wind_a2-d_source_read_development_d0_v1.json",
            lambda value: value.update(volume_queries=4096),
        ),
        (
            "wind_a2-d_source_read_development_d_scale_v1.json",
            lambda value: value["model"].update(source_value_conditioning="none"),
        ),
        (
            "wind_a2-d_source_read_development_d0_v1.json",
            lambda value: value.update(sampling_version="case_random_v0"),
        ),
        (
            "wind_a2-d_source_read_full1000_d_scale_v1.json",
            lambda value: value.update(approved_stop_after=2500),
        ),
        (
            "wind_a2-d_source_read_full1000_d_scale_v1.json",
            lambda value: value.update(source_reader_variant="D0"),
        ),
    ],
)
def test_wind_source_read_recipe_rejects_scope_or_sampler_drift(recipe_name, mutate):
    recipe = copy.deepcopy(_recipe(recipe_name))
    mutate(recipe)
    with pytest.raises(ValueError):
        _validate_source_read_wind_recipe_v1(recipe)


def test_source_read_provider_reuses_formal_two_region_draws_and_selector():
    formal = _provider_without_data(WindFormalA2DTaskV1, "formal_d_full_train_v1")
    source = _provider_without_data(WindSourceReadTaskV1, "source_read_development_v1")
    train_key = SamplingKey(
        42,
        17,
        0,
        0,
        "joint",
        "A2-D",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        dataset_id=source._test_expected_dataset_id,
    )
    formal_batch = formal._make_batch((7,), train_key, validation=False, query_count=8192)
    source_batch = source._make_batch((7,), train_key, validation=False, query_count=8192)
    np.testing.assert_array_equal(source_batch.receivers.coordinates_D, formal_batch.receivers.coordinates_D)
    np.testing.assert_array_equal(source_batch.receivers.flat_indices, formal_batch.receivers.flat_indices)
    assert source_batch.receivers.role_ids.shape == (1, 8192)
    assert np.count_nonzero(source_batch.receivers.role_ids[0] == 0) == 6144
    assert np.count_nonzero(source_batch.receivers.role_ids[0] == 1) == 2048
    assert source.loss_denominators((source_batch,)) == {source.LOSS_NAME: 1.0}

    validation_key = SamplingKey(
        42,
        0,
        0,
        0,
        "validation",
        "A2-D",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        dataset_id=source._test_expected_dataset_id,
    )
    validation = source._make_batch((7,), validation_key, validation=True, query_count=10240)
    assert validation.receivers.role_ids.shape == (1, 10240)
    assert np.count_nonzero(validation.receivers.role_ids[0] == 0) == 8192
    assert np.count_nonzero(validation.receivers.role_ids[0] == 1) == 2048
    assert tuple(batch.receivers.role_ids.shape[1] for batch in source.validation_batches()) == (10240,)
    assert source._test_case.sample_calls[0][2] == formal._test_case.sample_calls[0][2]


def test_source_read_identity_keeps_protocol_variant_and_fixed_cohort_explicit(monkeypatch):
    base_identity = {
        "schema": "wind_content_hypergraph_provider_v2",
        "dataset": "WindFarm",
        "training": {"sampling_dataset_id": "wind:fixed"},
        "native_role_catalogue_cache": {"roles": ["five roles"]},
        "normalizer_and_background_source": {"kind": "verified_fixed24_train_fit"},
        "train_layout_indices": list(range(24)),
        "validation_layout_indices": list(range(8)),
    }
    monkeypatch.setattr(WindContentHypergraphTaskV2, "identity_payload", lambda self: copy.deepcopy(base_identity))
    provider = object.__new__(WindSourceReadTaskV1)
    provider.recipe = _recipe("wind_a2-d_source_read_development_d0_v1.json")
    provider.source_reader_variant = "D0"
    provider.profile_id = "wind_shared_fixed24_v1"
    provider.manifest = {
        "subset_id": "wind_shared_fixed24_v1",
        "train_layout_selection": {"uncovered_turbine_count_strata_by_limited_selection": [18]},
    }
    provider.train_rows = np.arange(72)
    provider.validation_rows = np.arange(24)
    provider.role_query_counts = {"volume": 6144, "hub_band": 2048}
    provider.validation_role_query_counts = {"volume": 8192, "hub_band": 2048}
    provider.initialization_seed = 0
    provider.sampling_seed = 42
    provider.effective_batch_size = 16
    provider.max_microbatch_cases = 8
    provider.primary_queries = 8192
    provider.total_epochs = 2500

    identity = provider.identity_payload()

    assert identity["protocol"] == "source_read_development_v1"
    assert identity["source_read_scope"] == "source_read_development_v1"
    assert identity["source_reader_variant"] == "D0"
    assert "formal_protocol" not in identity
    assert "test_row_indices_sha256" not in identity
    assert identity["normalizer_and_background_source"]["fit_rows"] == "the fixed 72-row TRAIN cohort only"
    assert identity["objective"]["role_component_scales_mps_are_training_objective_inputs"] is False
    assert "component_scales_mps" not in identity["objective"]
    assert "scale_calibration" not in identity["objective"]
    assert "scale_cache_identity" not in identity["objective"]
    assert identity["development_cohort_coverage"]["uncovered_train_module_count_strata"] == [18]
    assert identity["training"]["initialization_seed"] == 0
    assert identity["training"]["sampling_seed"] == 42


def test_source_read_provider_refuses_historical_five_role_normalized_paths():
    provider = object.__new__(WindContentHypergraphTask)
    provider.role_component_scales = None

    with pytest.raises(RuntimeError, match="Five-role normalized training loss is unavailable"):
        WindContentHypergraphTask._loss_terms(provider, None, None)
    with pytest.raises(RuntimeError, match="Five-role normalized validation metrics are unavailable"):
        WindContentHypergraphTask.validation_metrics(provider, None, None, {})
