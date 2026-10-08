from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch
from channelthermal.training.unified_task import (
    ThermalPredictions,
    ThermalRefinementTask,
    ThermalTargets,
    _sample_primary,
)
from torch import nn
from windfarm.training.unified_task import (
    ROLE_NAMES,
    WindPredictions,
    WindRefinementTask,
    WindTargets,
)

from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    SamplingKey,
    TaskBatch,
    TrainingEngine,
)


def _thermal_sampling_case():
    fluid_count, material_count = 80, 40
    return {
        "case_id": "thermal-case-A",
        "structure": {
            "module_centers": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "module_present": np.asarray([1.0], dtype=np.float32),
            "material_params": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        },
        "query_xy": np.arange(fluid_count * 2, dtype=np.float32).reshape(fluid_count, 2),
        "field_targets": np.arange(fluid_count * 5, dtype=np.float32).reshape(fluid_count, 5),
        "point_weights": np.ones((fluid_count, 1), dtype=np.float32),
        "interface_target": np.arange(1 * 8 * 2, dtype=np.float32).reshape(1, 8, 2),
        "interface_condition_valid_mask": np.ones((1, 8, 2), dtype=np.float32),
        "module_internal_temperature_points": np.arange(material_count, dtype=np.float32)[None],
        "module_internal_query_points": np.arange(material_count * 2, dtype=np.float32).reshape(material_count, 2),
    }


def _thermal_response_validation_family():
    fluid_count, material_count = 80, 40
    return {
        "family_id": "0304",
        "structure": {
            "module_centers": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "module_present": np.asarray([1.0], dtype=np.float32),
            "material_params": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        },
        "fluid_xy": np.arange(fluid_count * 2, dtype=np.float32).reshape(fluid_count, 2),
        "fluid_valid": np.ones(fluid_count, dtype=bool),
        "material_local": np.arange(material_count * 2, dtype=np.float32).reshape(material_count, 2),
        "heat_increment": np.ones((1,), dtype=np.float32),
        "deltas": {
            "fluid_fields": np.zeros((fluid_count, 5), dtype=np.float32),
            "interface": np.zeros((8, 1), dtype=np.float32),
            "solid_temperature": np.zeros((1, material_count), dtype=np.float32),
        },
        "module_count": 1,
        "interface_valid": np.ones((8, 1), dtype=bool),
        "material_valid": np.ones((1, material_count), dtype=bool),
    }


def test_thermal_case_epoch_sampler_is_packing_independent_and_prefix_stable():
    case = _thermal_sampling_case()
    short_budget = {"fluid_queries": 12, "material_queries_per_module": 8, "surface_stride": 2}
    long_budget = {"fluid_queries": 24, "material_queries_per_module": 16, "surface_stride": 2}
    full_key = SamplingKey(
        0, 5, 1, 0, "warmup", "full_detail",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id="ThermalChannel:fixed25_v1",
    )
    repacked_key = SamplingKey(
        0, 5, 8, 3, "hard", "adaptive_detail",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id="ThermalChannel:fixed25_v1",
    )

    short, short_hash = _sample_primary(
        (case,), (0,), epoch=5, key=full_key, training=True,
        budget=short_budget, device=torch.device("cpu"),
    )
    long, long_hash = _sample_primary(
        (case,), (0,), epoch=5, key=repacked_key, training=True,
        budget=long_budget, device=torch.device("cpu"),
    )
    assert short_hash != long_hash  # Receipts bind the full sampled panel.
    np.testing.assert_array_equal(short["query_xy"].numpy(), long["query_xy"][:, :12].numpy())
    np.testing.assert_array_equal(
        short["module_internal_query_points"].numpy(),
        long["module_internal_query_points"][:, :8].numpy(),
    )
    assert short["field_targets"].shape[1] == 12
    assert long["field_targets"].shape[1] == 24
    assert short["module_internal_temperature_points"].shape[-1] == 8
    assert long["module_internal_temperature_points"].shape[-1] == 16


def test_thermal_legacy_sampler_retains_the_historical_packed_draws():
    case = _thermal_sampling_case()
    budget = {"fluid_queries": 12, "material_queries_per_module": 8, "surface_stride": 2}
    key = SamplingKey(0, 5, 1, 2, "warmup", "full_detail")
    sampled, _receipt = _sample_primary(
        (case,), (0,), epoch=5, key=key, training=True,
        budget=budget, device=torch.device("cpu"),
    )
    rng = key.numpy_rng("thermal_primary_native_queries", case["case_id"])
    fluid = rng.choice(len(case["query_xy"]), budget["fluid_queries"], replace=False)
    material = rng.choice(len(case["module_internal_query_points"]), budget["material_queries_per_module"], replace=False)
    np.testing.assert_array_equal(sampled["query_xy"].numpy()[0], case["query_xy"][fluid])
    np.testing.assert_array_equal(sampled["module_internal_query_points"].numpy()[0],
                                  case["module_internal_query_points"][material])


class _ValidationObjectiveProvider:
    def __init__(self):
        self.validation_visits = 0
        self.training_marker = 7
        self.predict_modes = []

    def validation_batches(self):
        for numerator, denominator in ((4.0, 2), (8.0, 4)):
            self.validation_visits += 1
            yield TaskBatch(
                scene_inputs=torch.tensor([[float(self.validation_visits)]]),
                receivers=None,
                targets=torch.tensor([numerator, denominator]),
            )

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        del receivers, epoch, temperature
        self.predict_modes.append((execution_mode, phase, model.training))
        random.random()
        np.random.random()
        torch.rand(1)
        return model(scene), None

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del predictions, auxiliary_state
        return {"count": int(targets[1])}

    def reduce_native_metrics(self, records):
        return {"field_score": float(sum(row["count"] for row in records))}

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch, arm):
        del predictions, auxiliary_state, batch, arm
        self.training_marker = -100
        numerator, denominator = float(targets[0]), int(targets[1])
        return {"reconstruction": LossTerm(torch.tensor(numerator), denominator, weight=0.5)}

    def training_state_dict(self):
        return {"validation_visits": self.validation_visits, "training_marker": self.training_marker}

    def load_training_state_dict(self, state):
        self.validation_visits = int(state["validation_visits"])
        self.training_marker = int(state["training_marker"])


def test_evaluate_records_hard_validation_loss_and_preserves_rng_and_task_state():
    config = EngineConfig(
        seed=1,
        microbatch_cases=1,
        effective_cases=1,
        total_epochs=10,
        warmup_epochs=2,
        open_through_epoch=4,
        soft_through_epoch=6,
    )
    engine = TrainingEngine(config, device="cpu")
    model = nn.Linear(1, 1)
    model.train(True)
    provider = _ValidationObjectiveProvider()
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    expected_torch = torch.get_rng_state().clone()

    result = engine._evaluate(model, provider, "adaptive_detail", "open", 3)

    assert model.training is True
    assert provider.predict_modes == [
        ("adaptive_detail", "hard", False),
        ("adaptive_detail", "hard", False),
    ]
    assert provider.training_marker == 7
    assert provider.validation_visits == 2
    assert result["validation_loss_terms"] == {"reconstruction": 1.0}
    assert result["validation_loss_term_weights"] == {"reconstruction": 0.5}
    assert result["validation_route_phase"] == "hard"
    assert "sum of numerators" in result["validation_loss_aggregation"]
    assert random.getstate() == expected_python
    actual_numpy = np.random.get_state()
    assert np.array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    assert torch.equal(torch.get_rng_state(), expected_torch)


def test_thermal_validation_terms_use_native_hard_output_and_available_expected_work():
    provider = object.__new__(ThermalRefinementTask)
    provider._expected_work_calibration = {"stage": "train_calibration"}
    provider._expected_work_weight = 0.4
    provider._case_reconstruction = lambda output, prepared, targets: {
        "reconstruction": torch.tensor([2.0, 4.0])
    }
    probability = torch.tensor([[[0.5, 0.25], [0.2, 0.8]]])
    protected = torch.tensor([[[True, False], [False, False]]])
    prepared = SimpleNamespace(
        response=SimpleNamespace(refinement_aux={"probability": probability, "protected": protected}),
        source_present=torch.ones((1, 2)),
    )
    predictions = ThermalPredictions(
        model=None,
        execution_mode="adaptive_detail",
        phase="hard",
        native_prepared=prepared,
        native_main={},
        native_full=None,
        response_prepared=None,
        response_main=None,
        response_full=None,
        auxiliary={},
        work={},
    )
    targets = ThermalTargets(
        case_ids=("dev-a", "dev-b"), case_indices=(), epoch=1,
        field_targets=torch.empty(0), point_weights=torch.empty(0),
        interface_target=torch.empty(0), interface_valid_mask=torch.empty(0),
        material_targets=torch.empty(0),
    )

    terms = provider.validation_loss_terms(
        predictions, targets, {}, batch=TaskBatch(None, None, None), arm="adaptive_detail")

    assert set(terms) == {"reconstruction", "expected_work"}
    assert float(terms["reconstruction"].numerator) == 6.0
    assert terms["reconstruction"].denominator == 2
    assert float(terms["expected_work"].numerator) == 2.25
    assert float(terms["expected_work"].denominator) == 4.0
    assert terms["expected_work"].weight == 0.4


def test_training_query_override_keeps_real_primary_and_response_validation_at_baseline_q():
    baseline_budget = {"fluid_queries": 12, "material_queries_per_module": 8,
                       "surface_stride": 2, "microbatch_cases": 1}
    candidate_budget = {**baseline_budget, "fluid_queries": 24, "material_queries_per_module": 16}
    case = _thermal_sampling_case()
    case["structure"]["heat_powers"] = np.ones((1,), dtype=np.float32)
    family = _thermal_response_validation_family()

    def provider_for(training_budget):
        provider = object.__new__(ThermalRefinementTask)
        provider.training_cases = (case,)
        provider.validation_cases = (case,)
        provider.budget = dict(training_budget)
        provider.validation_budget = dict(baseline_budget)
        provider.calibration_budget = dict(baseline_budget)
        provider.recipe = {"calibration": {"calibration_case_ids": [case["case_id"]]}}
        provider._train_index_by_id = {case["case_id"]: 0}
        provider.device = torch.device("cpu")
        provider._validation_samples = 0
        provider._training_samples = 0
        return provider

    baseline = provider_for(baseline_budget)
    candidate = provider_for(candidate_budget)
    baseline_batch = next(baseline.validation_batches())
    candidate_batch = next(candidate.validation_batches())
    assert baseline_batch.receivers.fluid_xy.shape == candidate_batch.receivers.fluid_xy.shape == (1, 12, 2)
    assert (baseline_batch.receivers.local_query_points.shape
            == candidate_batch.receivers.local_query_points.shape == (1, 8, 2))
    torch.testing.assert_close(baseline_batch.receivers.fluid_xy, candidate_batch.receivers.fluid_xy)
    torch.testing.assert_close(baseline_batch.targets.field_targets, candidate_batch.targets.field_targets)
    torch.testing.assert_close(baseline_batch.targets.material_targets, candidate_batch.targets.material_targets)

    baseline_calibration = baseline._calibration_batches()
    candidate_calibration = candidate._calibration_batches()
    assert baseline_calibration.receivers.fluid_xy.shape == candidate_calibration.receivers.fluid_xy.shape == (1, 12, 2)
    assert (baseline_calibration.receivers.local_query_points.shape
            == candidate_calibration.receivers.local_query_points.shape == (1, 8, 2))
    torch.testing.assert_close(baseline_calibration.receivers.fluid_xy, candidate_calibration.receivers.fluid_xy)
    torch.testing.assert_close(baseline_calibration.targets.field_targets, candidate_calibration.targets.field_targets)

    baseline_response = baseline._response_sample(family, training=False, key=None)
    candidate_response = candidate._response_sample(family, training=False, key=None)
    assert candidate_response["fluid_ids"].shape == baseline_response["fluid_ids"].shape == (12,)
    assert candidate_response["material_ids"].shape == baseline_response["material_ids"].shape == (8,)
    np.testing.assert_array_equal(candidate_response["fluid_ids"], baseline_response["fluid_ids"])
    np.testing.assert_array_equal(candidate_response["material_ids"], baseline_response["material_ids"])


def test_wind_validation_terms_use_role_scales_and_measured_route_probability():
    provider = object.__new__(WindRefinementTask)
    provider.role_scales = {name: 2.0 for name in ROLE_NAMES}
    provider._expected_work_calibration = {"stage": "train_calibration"}
    provider._expected_work_weight = 0.4
    probability = torch.full((1, 5, 3), 0.5)
    protected = torch.zeros((1, 5, 3), dtype=torch.bool)
    predictions = WindPredictions(
        main_mps=torch.zeros((1, 5, 3)),
        full_mps=None,
        role_ids=torch.arange(5).reshape(1, 5),
        auxiliary={"probability": probability, "protected": protected},
        execution_mode="adaptive",
    )
    targets = WindTargets(
        velocity_mps=np.ones((1, 5, 3), dtype=np.float32),
        row_indices=(2,), case_metadata=({},),
    )
    batch = TaskBatch(
        scene_inputs=(SimpleNamespace(module_present=np.asarray([1.0, 1.0, 0.0])),),
        receivers=None,
        targets=targets,
    )

    terms = provider.validation_loss_terms(
        predictions, targets, {}, batch=batch, arm="adaptive_detail")

    assert set(terms) == {*(f"native_role/{name}" for name in ROLE_NAMES), "expected_work"}
    assert terms[f"native_role/{ROLE_NAMES[0]}"].numerator.item() == 0.75
    assert terms[f"native_role/{ROLE_NAMES[0]}"].denominator == 3
    assert terms[f"native_role/{ROLE_NAMES[0]}"].weight == 0.2
    assert terms["expected_work"].numerator.item() == 5.0
    assert terms["expected_work"].denominator.item() == 10.0
    assert terms["expected_work"].weight == 0.4
