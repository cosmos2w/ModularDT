"""CPU contracts for the bounded Round3 Thermal source-read provider."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
from channelthermal.training import content_hypergraph_task_v2 as thermal_task_v2
from channelthermal.training.content_hypergraph_task_v2 import (
    ThermalContentHypergraphTaskV2,
    ThermalSourceReadTaskV1,
    build_thermal_source_read_task_v1,
)
from torch import nn

from honf_runtime.source_read_protocol import (
    SOURCE_READ_DEVELOPMENT,
    SOURCE_READ_FULL,
    validate_source_read_recipe,
)

_CONFIG_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src/config_core/forward/content_hypergraph"
)


def _recipe(name: str) -> dict:
    return json.loads((_CONFIG_ROOT / name).read_text())


def test_thermal_source_read_recipes_are_aligned_and_keep_full_scope_conditional():
    d0 = _recipe("thermal_a2-d_source_read_development_d0_v1.json")
    dscale = _recipe("thermal_a2-d_source_read_development_d_scale_v1.json")
    full = _recipe("thermal_a2-d_source_read_full1000_d_scale_v1.json")

    for recipe in (d0, dscale, full):
        assert validate_source_read_recipe(recipe) == recipe
        assert recipe["architecture"] == "A2-D"
        assert recipe["sampling_seed"] == recipe["seed"] == 0
        assert recipe["primary_queries"] == 1024
        assert recipe["microbatch_size"] == 24
        assert recipe["effective_batch_size"] == 48
        assert recipe["optimizer"]["lr"] == 3.0e-4
        assert recipe["optimizer"]["weight_decay"] == 1.0e-5
        assert recipe["optimizer"]["bias_and_norm_weight_decay"] == 1.0e-5
        assert recipe["model"]["context_dim"] == 128
        assert recipe["model"]["num_hyperedges"] == 6
        assert recipe["model"]["reader_execution_backend"] == "legacy"

    assert d0["protocol"] == dscale["protocol"] == SOURCE_READ_DEVELOPMENT
    assert d0["execution"]["checkpoint_epochs"] == dscale["execution"]["checkpoint_epochs"]
    assert d0["execution"]["checkpoint_epochs"] == list(range(100, 2501, 100))
    assert d0["execution"]["monitor_every"] == d0["execution"]["latest_every"] == 100
    assert d0["execution"]["curve_every"] == 10
    assert d0["source_reader_variant"] == "D0"
    assert d0["model"]["source_value_conditioning"] == "none"
    assert dscale["source_reader_variant"] == "D-scale"
    assert dscale["model"]["source_value_conditioning"] == "source_scale"
    assert full["protocol"] == SOURCE_READ_FULL
    assert full["source_reader_variant"] == "D-scale"
    assert full["total_epochs"] == 5000
    assert full["approved_stop_after"] == 1000
    assert full["execution"]["monitor_every"] == full["execution"]["latest_every"] == 10
    assert full["execution"]["checkpoint_epochs"] == list(range(100, 1001, 100))


def test_thermal_source_read_builder_routes_only_validated_recipes_to_data_binding(monkeypatch):
    valid = _recipe("thermal_a2-d_source_read_development_d0_v1.json")

    def stop_at_binding(*args, **kwargs):
        raise RuntimeError("reached Thermal source-read data binding")

    monkeypatch.setattr(thermal_task_v2, "_load_data_binding", stop_at_binding)
    with pytest.raises(RuntimeError, match="reached Thermal source-read data binding"):
        build_thermal_source_read_task_v1(valid)

    invalid = _recipe("thermal_a2-d_source_read_development_d0_v1.json")
    invalid["model"]["source_value_conditioning"] = "source_scale"
    with pytest.raises(ValueError, match="Source-scale conditioning"):
        build_thermal_source_read_task_v1(invalid)

    forbidden = _recipe("thermal_a2-d_source_read_development_d0_v1.json")
    forbidden["protocol"] = SOURCE_READ_FULL
    forbidden["total_epochs"] = 5000
    forbidden["approved_stop_after"] = 1000
    forbidden["source_reader_variant"] = "D0"
    forbidden["execution"] = {
        "monitor_every": 10,
        "latest_every": 10,
        "curve_every": 10,
        "checkpoint_epochs": list(range(100, 1001, 100)),
        "field_metric": "field_score",
    }
    with pytest.raises(ValueError, match="Unchanged D0 does not qualify"):
        build_thermal_source_read_task_v1(forbidden)


@pytest.mark.parametrize("total_epochs", [2500, 5000])
def test_thermal_source_read_optimizer_is_constant_all_parameter_adamw(total_epochs):
    class TinyA2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = "A2-D"
            self.linear = nn.Linear(4, 4)
            self.norm = nn.LayerNorm(4)

    model = TinyA2D()
    provider = object.__new__(ThermalSourceReadTaskV1)
    provider.model = model
    provider.total_epochs = total_epochs
    groups = provider.optimizer_groups(model, "A2-D", "joint")
    assert len(groups) == 1
    assert set(groups[0].parameter_names) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert groups[0].weight_decay == 1.0e-5
    assert groups[0].betas == (0.9, 0.999)
    assert groups[0].eps == 1.0e-8
    assert groups[0].schedule.value(1) == 3.0e-4
    assert groups[0].schedule.value(total_epochs) == 3.0e-4


def test_source_read_identity_replaces_legacy_scope_and_preserves_variant(monkeypatch):
    base_identity = {
        "protocol": "development",
        "dataset_profile": "fixed25_v1",
        "source_binding": {"normalization_fit_scope": "selected original TRAIN150 only"},
        "training": {"optimizer_schedule": {"kind": "old"}, "weight_decay": 1.0e-4},
        "recipe": {},
    }
    monkeypatch.setattr(
        ThermalContentHypergraphTaskV2,
        "identity_payload",
        lambda self: dict(base_identity),
    )
    provider = object.__new__(ThermalSourceReadTaskV1)
    provider.source_read_protocol = SOURCE_READ_DEVELOPMENT
    provider.source_reader_variant = "D-scale"
    provider.reader_execution_backend = "reduced_projection"
    provider.source_value_conditioning = "source_scale"
    provider.total_epochs = 2500
    provider.recipe = _recipe("thermal_a2-d_source_read_development_d_scale_v1.json")

    identity = provider.identity_payload()
    assert identity["protocol"] == SOURCE_READ_DEVELOPMENT
    assert identity["dataset_profile"] == "fixed25_v1_TRAIN150_DEV22"
    assert identity["source_reader_variant"] == "D-scale"
    assert identity["source_read_contract"]["reader_execution_backend"] == "reduced_projection"
    assert identity["training"]["weight_decay"] == 1.0e-5
    assert identity["training"]["gradient_clip"] == 1.0
    assert identity["training"]["optimizer_schedule"]["kind"] == "constant"
    assert identity["recipe"]["source_reader_variant"] == "D-scale"


def _stats() -> dict[str, np.ndarray]:
    return {
        "field_mean_by_channel": np.asarray([1.0, -2.0, 0.5, 3.0, 10.0], dtype=np.float32),
        "field_std_by_channel": np.asarray([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "interface_targets_mean": np.asarray([10.0, 0.0], dtype=np.float32),
        "interface_targets_std": np.asarray([4.0, 5.0], dtype=np.float32),
        "internal_temperature_std": np.asarray([7.0], dtype=np.float32),
    }


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


@pytest.mark.parametrize("conditioning", ["none", "source_scale"])
def test_source_read_thermal_adapter_preserves_native_heat_laws_and_roundtrip(conditioning):
    core_config = {
        "schema": "content_hypergraph_core_v2",
        "architecture": "A2-D",
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
        "init_seed": 0,
        "reader_execution_backend": "legacy",
        "source_value_conditioning": conditioning,
    }
    torch.manual_seed(13)
    model = ThermalContentHypergraphAdapterV2(_stats(), core_config=core_config, nx=8, ny=6).eval()
    yy, xx = torch.meshgrid(
        torch.arange(6, dtype=torch.float32) + 0.5,
        (torch.arange(8, dtype=torch.float32) + 0.5) * 1.5,
        indexing="ij",
    )
    fluid = torch.stack((xx.reshape(-1), yy.reshape(-1)), -1)[None]
    local = torch.tensor([[[[0.0, 0.0], [0.35, 0.0]], [[0.0, 0.0], [0.35, 0.0]]]])
    prepared = model.prepare_native(
        _structure(), fluid, local_query_points=local, ntheta=16, scene_key=("source-read-law",)
    )
    zero = model.apply_native(prepared, torch.zeros((1, 2)))
    heat = model.apply_native(prepared, torch.tensor([[0.4, 0.8]]))
    doubled = model.apply_native(prepared, torch.tensor([[0.8, 1.6]]))
    assert torch.count_nonzero(zero["pred_field"][..., 4]) == 0
    assert torch.count_nonzero(zero["pred_interface"]) == 0
    torch.testing.assert_close(heat["pred_field"][..., :4], zero["pred_field"][..., :4], rtol=0, atol=0)
    torch.testing.assert_close(
        doubled["pred_field"][..., 4], 2.0 * heat["pred_field"][..., 4], rtol=2e-6, atol=2e-6
    )
    payload = model.state_payload()
    if conditioning == "none":
        assert "source_value_conditioning" not in payload["core"]
    else:
        assert payload["core"]["source_value_conditioning"] == conditioning
    assert "reader_execution_backend" not in payload["core"]
    restored = ThermalContentHypergraphAdapterV2.from_state_payload(
        payload, state_dict=model.state_dict(), device="cpu"
    ).eval()
    actual = restored.apply_native(
        restored.prepare_native(
            _structure(), fluid, local_query_points=local, ntheta=16, scene_key=("source-read-law",)
        ),
        torch.tensor([[0.4, 0.8]]),
    )
    torch.testing.assert_close(actual["pred_field"], heat["pred_field"], rtol=0, atol=0)
    torch.testing.assert_close(actual["pred_interface"], heat["pred_interface"], rtol=0, atol=0)
    torch.testing.assert_close(
        actual["pred_internal_temperature"], heat["pred_internal_temperature"], rtol=0, atol=0
    )
