"""Formal A2-D full-TRAIN Thermal recipe and optimizer contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from channelthermal.training import content_hypergraph_task_v2 as thermal_task_v2
from channelthermal.training.content_hypergraph_task_v2 import (
    ThermalFormalContentHypergraphTaskV1,
    build_thermal_formal_content_task_v1,
)
from honf_runtime.content_hypergraph_formal import validate_formal_d_recipe


_RECIPE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/config_core/forward/content_hypergraph/thermal_a2-d_formal_full_train_v1.json"
)


def _change_to_wind(value):
    value["dataset"] = "wind"
    value["sampling_seed"] = 42
    value["primary_queries"] = 8192
    value["microbatch_size"] = 8
    value["effective_batch_size"] = 16
    value["total_epochs"] = 2500
    value["approved_stop_after"] = 2500
    value["execution"] = {
        "monitor_every": 5,
        "latest_every": 5,
        "curve_every": 100,
        "checkpoint_epochs": [100, 500, 1000, 1500, 2000, 2500],
        "field_metric": "val_volume_mse",
    }
    value["model"] = {
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6,
        "thermal_response_rank": 0,
        "local_hidden_dim": 128,
        "dropout": 0.0,
    }


def _recipe() -> dict:
    return json.loads(_RECIPE_PATH.read_text())


def test_thermal_formal_recipe_matches_shared_contract_and_reaches_data_binding(monkeypatch):
    recipe = validate_formal_d_recipe(_recipe())
    assert recipe["dataset"] == "thermal"
    assert recipe["architecture"] == "A2-D"
    assert recipe["sampling_seed"] == 0
    assert recipe["execution"]["field_metric"] == "field_score"

    def stop_at_binding(*args, **kwargs):
        raise RuntimeError("reached Thermal data binding")

    monkeypatch.setattr(thermal_task_v2, "_load_data_binding", stop_at_binding)
    with pytest.raises(RuntimeError, match="reached Thermal data binding"):
        build_thermal_formal_content_task_v1(recipe)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_change_to_wind, "Thermal formal builder requires dataset=thermal"),
        (lambda value: value.update(architecture="A2-H6"), "limited to A2-D"),
        (lambda value: value.update(approved_stop_after=1000), "approved_stop_after=5000"),
        (lambda value: value.update(optimizer_bias_norm_exclusion=True), "optimizer_bias_norm_exclusion=false"),
    ],
)
def test_thermal_formal_builder_rejects_crossed_or_changed_profiles_before_data(mutate, message, monkeypatch):
    recipe = _recipe()
    mutate(recipe)
    monkeypatch.setattr(
        thermal_task_v2,
        "_load_data_binding",
        lambda *args, **kwargs: pytest.fail("invalid formal recipe reached data binding"),
    )
    with pytest.raises(ValueError, match=message):
        build_thermal_formal_content_task_v1(recipe)


def test_thermal_formal_optimizer_is_one_constant_legacy_adamw_group():
    class TinyA2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = "A2-D"
            self.linear = nn.Linear(4, 4)
            self.norm = nn.LayerNorm(4)

    model = TinyA2D()
    provider = object.__new__(ThermalFormalContentHypergraphTaskV1)
    provider.model = model
    groups = provider.optimizer_groups(model, "A2-D", "joint")
    assert len(groups) == 1
    assert set(groups[0].parameter_names) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert groups[0].weight_decay == 1.0e-5
    assert groups[0].schedule.value(1) == 3.0e-4
    assert groups[0].schedule.value(5000) == 3.0e-4


def test_development_recipe_optimizer_contract_stays_versioned_and_separate():
    development = json.loads(
        (_RECIPE_PATH.parent / "thermal_a2-d_development.json").read_text()
    )
    assert development["protocol"] == "development"
    assert development["optimizer_bias_norm_exclusion"] is True
    assert development["optimizer"]["weight_decay"] == 1.0e-4
