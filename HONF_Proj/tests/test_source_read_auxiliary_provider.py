"""Constructed CPU contracts for bounded source-read provider hooks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.training.content_hypergraph_task_v2 import ThermalSourceReadTaskV1
from honf_runtime import source_read_auxiliary
from honf_runtime.unified_training import SamplingKey
from windfarm.training.content_hypergraph_task import (
    WindSourceReadTaskV1,
    _apply_wind_source_read_core_options,
    _validate_wind_v2_core_binding,
)
from windfarm.workflows.joint_forward import ROLE_NAMES


def test_thermal_source_read_heat_envelope_uses_only_ordered_train_sources():
    provider = object.__new__(ThermalSourceReadTaskV1)
    provider.training_case_ids = ("train-a", "train-b")
    provider.training_cases = [
        {"structure": {"heat_powers": torch.tensor([2.0, -4.0, 1000.0]),
                        "module_present": torch.tensor([True, True, False])}},
        {"structure": {"heat_powers": torch.tensor([-3.0, 5.0]),
                        "module_present": torch.tensor([True, True])}},
    ]
    assert provider.source_read_positive_train_envelope(("train-a", "train-b")) == 10.0
    with pytest.raises(ValueError, match="exact ordered sealed TRAIN cohort"):
        provider.source_read_positive_train_envelope(("train-b", "train-a"))


def test_thermal_teacher_trace_is_target_free_bounded_and_restores_counters(monkeypatch):
    provider = object.__new__(ThermalSourceReadTaskV1)
    provider.model = None
    provider.sampling_dataset_id = "thermal-train-fixture"
    provider._train_index_by_id = {"train-a": 0}
    provider.physical_training_queries = 7

    class Batch:
        scene_inputs = object()
        receivers = SimpleNamespace(
            fluid_xy=torch.zeros(1, 2, 2),
            local_query_points=torch.zeros(1, 2, 1, 2),
            ntheta=8,
        )

        @property
        def targets(self):
            raise AssertionError("the target-free source-read teacher accessed labels")

    scene = SimpleNamespace(structure={}, case_ids=("train-a",))
    organization = object()
    query = object()
    received = {}

    def make_batch(case_keys, key):
        provider.physical_training_queries += 12
        assert case_keys == ("train-a",)
        return Batch()

    class Core:
        def set_source_read_training_epoch(self, epoch):
            received["epoch"] = epoch

        def source_read_teacher_trace_v1(self, actual_organization, actual_query):
            received["call"] = (actual_organization, actual_query)
            return {"thermal_k": torch.zeros(1, 2, 1)}

    class Model:
        core = Core()

        def prepare_native(self, *args, **kwargs):
            return SimpleNamespace(organization=organization, query_batch=query)

    model = Model()
    provider.model = model
    provider.make_batch = make_batch
    provider.make_scene = lambda _: scene
    monkeypatch.setattr(source_read_auxiliary, "select_evenly_spaced_teacher_rows", lambda value, cap: value)
    initial_key = SamplingKey(
        0, 500, 0, 0, "source_read_auxiliary_initial", "S-Utility",
        SamplingKey.CASE_EPOCH_VERSION, provider.sampling_dataset_id,
    )
    result = provider.source_read_teacher_trace(model, "train-a", initial_key, 32)
    refresh_key = SamplingKey(
        0, 601, 8, 0, "source_read_auxiliary_refresh", "S-Utility",
        SamplingKey.CASE_EPOCH_VERSION, provider.sampling_dataset_id,
    )
    refreshed = provider.source_read_teacher_trace(model, "train-a", refresh_key, 32)
    assert result["thermal_k"].shape == (1, 2, 1)
    assert refreshed["thermal_k"].shape == (1, 2, 1)
    assert received["call"] == (organization, query)
    assert received["epoch"] == 601
    assert provider.physical_training_queries == 7

    for epoch, stage in (
        (500, "source_read_auxiliary_refresh"),
        (501, "source_read_auxiliary_initial"),
        (2501, "source_read_auxiliary_refresh"),
    ):
        invalid_key = SamplingKey(
            0, epoch, 8, 0, stage, "S-Utility",
            SamplingKey.CASE_EPOCH_VERSION, provider.sampling_dataset_id,
        )
        with pytest.raises(ValueError, match="e500 initial fit or a bounded"):
            provider.source_read_teacher_trace(model, "train-a", invalid_key, 32)


def test_thermal_source_read_prediction_sets_epoch_before_native_prepare(monkeypatch):
    from channelthermal.training.content_hypergraph_task_v2 import ThermalContentHypergraphTaskV2

    observed = []

    class Core:
        def set_source_read_training_epoch(self, epoch):
            observed.append(("epoch", epoch))

    model = SimpleNamespace(core=Core())
    provider = object.__new__(ThermalSourceReadTaskV1)
    provider.model = model

    def parent_predict(self, actual_model, scene, receivers, execution_mode, phase, *, epoch, temperature):
        assert self is provider
        assert actual_model is model
        observed.append(("prepare", epoch))
        return "predictions", {}

    monkeypatch.setattr(ThermalContentHypergraphTaskV2, "predict_native", parent_predict)
    assert provider.predict_native(model, object(), object(), "A2-D", "joint", epoch=501) == ("predictions", {})
    assert observed == [("epoch", 501), ("prepare", 501)]


def test_wind_teacher_receiver_sampling_uses_geometry_without_reading_velocity_targets():
    class Case:
        case = "wind-case"
        layout_index = 4
        wind_direction_deg = 225.0
        support = object()

        @property
        def U(self):
            raise AssertionError("geometry-only source-read sampling loaded wind targets")

    case = Case()
    coordinates = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    catalogue = SimpleNamespace(
        role_cdf={role: np.asarray([1.0], dtype=np.float64) for role in ROLE_NAMES},
        role_indices={role: None for role in ROLE_NAMES},
        coordinates_D=coordinates,
    )

    class Adapter:
        def scene_inputs(self, actual_case):
            assert actual_case is case
            return SimpleNamespace(scene_key=("wind-case", 3, 4, 225.0))

        def query_features(self, coords, support):
            assert support is case.support
            return np.zeros((coords.shape[0], 7), dtype=np.float32)

        def packed_local_pair_indices(self, scene_inputs, coords):
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    provider = object.__new__(WindSourceReadTaskV1)
    provider._train_row_set = {3}
    provider.view = SimpleNamespace(run=lambda row: case)
    provider.adapter = Adapter()
    provider.catalogue_cache = SimpleNamespace(get=lambda actual_case: catalogue)
    provider.primary_queries = 5
    _, receivers = provider._source_read_target_free_receivers(
        3,
        SamplingKey(42, 500, 0, 0, "initial", "S-Utility",
                    SamplingKey.CASE_EPOCH_VERSION, "wind-train-fixture"),
    )
    assert receivers.coordinates_D.shape == (1, 5, 3)
    assert set(receivers.role_ids.reshape(-1).tolist()) == set(range(5))
    assert np.all(receivers.features == 0.0)


def test_wind_source_read_prediction_sets_epoch_before_native_prepare(monkeypatch):
    from windfarm.training.content_hypergraph_task import WindFormalA2DTaskV1

    observed = []

    class Model:
        def set_source_read_training_epoch(self, epoch):
            observed.append(("epoch", epoch))

    model = Model()
    provider = object.__new__(WindSourceReadTaskV1)
    provider.model = model

    def parent_predict(self, actual_model, scene, receivers, execution_mode, phase, *, epoch,
                       temperature, trace_request):
        assert self is provider
        assert actual_model is model
        observed.append(("prepare", epoch))
        return "predictions", {}

    monkeypatch.setattr(WindFormalA2DTaskV1, "predict_native", parent_predict)
    assert provider.predict_native(model, object(), object(), "A2-D", "joint", epoch=501) == ("predictions", {})
    assert observed == [("epoch", 501), ("prepare", 501)]


@pytest.mark.parametrize(
    "epoch,stage,accepted",
    [
        (500, "source_read_auxiliary_initial", True),
        (601, "source_read_auxiliary_refresh", True),
        (500, "source_read_auxiliary_refresh", False),
        (501, "source_read_auxiliary_initial", False),
        (2501, "source_read_auxiliary_refresh", False),
    ],
)
def test_wind_teacher_trace_accepts_only_declared_initial_and_current_model_refresh_keys(
    monkeypatch, epoch, stage, accepted,
):
    provider = object.__new__(WindSourceReadTaskV1)
    provider.model = None
    provider.sampling_dataset_id = "wind-train-fixture"
    provider._train_row_set = {3}
    seen = []
    scene_inputs, receivers, scene, query, organization = object(), object(), object(), object(), object()

    class Adapter:
        def make_scene_batch(self, inputs):
            assert inputs == (scene_inputs,)
            return scene

        def make_query_batch(self, actual_scene, actual_receivers):
            assert actual_scene is scene
            assert actual_receivers is receivers
            return query

    class Model:
        def set_source_read_training_epoch(self, actual_epoch):
            seen.append(("epoch", actual_epoch))

        def prepare(self, actual_scene):
            assert actual_scene is scene
            seen.append(("prepare", epoch))
            return organization

        def source_read_teacher_trace_v1(self, actual_organization, actual_query):
            assert actual_organization is organization
            assert actual_query is query
            return {"query_states": torch.zeros(1, 1, 4)}

    model = Model()
    provider.model = model
    provider.adapter = Adapter()
    provider._source_read_target_free_receivers = lambda case_key, key: (scene_inputs, receivers)
    monkeypatch.setattr(source_read_auxiliary, "select_evenly_spaced_teacher_rows", lambda value, cap: value)
    key = SamplingKey(
        0, epoch, 8 if epoch != 500 else 0, 0, stage, "S-Utility",
        SamplingKey.CASE_EPOCH_VERSION, provider.sampling_dataset_id,
    )
    if accepted:
        assert provider.source_read_teacher_trace(model, 3, key, 128)["query_states"].shape == (1, 1, 4)
        assert seen == [("epoch", epoch), ("prepare", epoch)]
    else:
        with pytest.raises(ValueError, match="e500 initial fit or a bounded"):
            provider.source_read_teacher_trace(model, 3, key, 128)
        assert seen == []


def test_wind_sparse_source_read_recipe_fields_reach_core_without_changing_dense_defaults():
    base = {
        "schema": "content_hypergraph_core_v2",
        "module_input_dim": 2,
        "environment_input_dim": 7,
        "coordinate_dim": 3,
        "query_input_dim": 7,
        "operating_input_dim": 11,
        "output_dim": 3,
        "architecture": "A2-D",
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 0,
        "thermal_response_rank": 0,
        "local_hidden_dim": 128,
        "dropout": 0.0,
        "init_seed": 0,
    }
    dense = dict(base)
    _apply_wind_source_read_core_options(
        dense,
        {"source_read_policy": "full"},
        schema_version=4,
        source_reader_variant="D0",
    )
    assert dense == base

    recipe = {
        "schema": "content_hypergraph_recipe_v2",
        "dataset": "wind",
        "architecture": "A2-D",
        "protocol": "source_read_development_v1",
        "seed": 0,
        "model": {
            "reader_execution_backend": "legacy",
            "source_value_conditioning": "none",
            "source_read_policy": "utility",
            "source_read_executor": "packed",
            "source_read_rank": 16,
            "source_read_scorer_width": 32,
        },
    }
    sparse = dict(base)
    _apply_wind_source_read_core_options(
        sparse, recipe["model"], schema_version=4, source_reader_variant="S-Utility",
    )
    _validate_wind_v2_core_binding(recipe, sparse)
    # The frozen dense controls saved whole-query Wind constructors despite
    # this inert recipe field. A child must preserve that actual execution.
    recipe["model"]["reader_query_chunk_size"] = 8192
    inherited = dict(base)
    _apply_wind_source_read_core_options(
        inherited, recipe["model"], schema_version=4, source_reader_variant="S-Utility")
    assert "reader_query_chunk_size" not in inherited
    _validate_wind_v2_core_binding(recipe, inherited)
    with pytest.raises(ValueError, match="missing or unsupported"):
        _validate_wind_v2_core_binding(recipe, {**inherited, "reader_query_chunk_size": 8192})
    assert {name: sparse[name] for name in (
        "source_read_policy", "source_read_executor", "source_read_rank", "source_read_scorer_width",
    )} == {
        "source_read_policy": "utility", "source_read_executor": "packed",
        "source_read_rank": 16, "source_read_scorer_width": 32,
    }
    with pytest.raises(ValueError, match="must set model.source_read_policy explicitly"):
        _apply_wind_source_read_core_options(
            {}, {}, schema_version=4, source_reader_variant="S-Utility",
        )
