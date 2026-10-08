from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch
from channelthermal.training.unified_task import (
    ThermalPredictions,
    ThermalRefinementTask,
    ThermalTargets,
)
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    TaskBatch,
    TrainingEngine,
)
from torch import nn
from windfarm.training.unified_task import (
    ROLE_NAMES,
    WindPredictions,
    WindRefinementTask,
    WindTargets,
)


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
