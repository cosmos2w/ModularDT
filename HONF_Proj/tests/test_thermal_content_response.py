"""Native transpose products and explicit factor reuse use the same operator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from channelthermal.content_hypergraph import ThermalContentHypergraphAdapter


def _fixture_module():
    path = Path(__file__).with_name("test_thermal_content_hypergraph.py")
    spec = importlib.util.spec_from_file_location("_thermal_response_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("architecture", ["A-D", "A-H6"])
def test_retained_factors_apply_native_and_transpose_without_neural_reads(architecture, monkeypatch):
    fixture = _fixture_module()
    config = fixture._model().core.get_config()
    config["architecture"] = architecture
    config.pop("final_environment_update", None)
    model = ThermalContentHypergraphAdapter(
        fixture._stats(), nx=8, ny=6, core_config=config,
    ).eval()
    heat = torch.tensor([[0.4, -0.7]], dtype=torch.float64)
    with torch.no_grad():
        prepared = fixture._prepared(model)
        expected = model.apply_increment(prepared, heat, accumulation_dtype=torch.float64)
        retained = model.prepare_affine_application(prepared)

    def forbidden(*args, **kwargs):
        raise AssertionError("Repeated affine application re-entered a neural reader or dense K export.")

    monkeypatch.setattr(model.core, "_decode_features", forbidden)
    monkeypatch.setattr(model.core, "export_source_kernels", forbidden)
    generator = torch.Generator().manual_seed(93)
    cotangents = {
        name: torch.randn(values.shape, generator=generator, dtype=torch.float64)
        for name, values in expected.items()
    }
    with torch.no_grad():
        actual = model.apply_increment(retained, heat, accumulation_dtype=torch.float64)
        transpose = model.heat_vjp_native(retained, cotangents, accumulation_dtype=torch.float64)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    lhs = sum((actual[name] * cotangents[name]).sum() for name in actual)
    rhs = (heat * transpose).sum()
    torch.testing.assert_close(lhs, rhs, rtol=2e-12, atol=2e-12)
    assert torch.count_nonzero(actual["pred_field"][..., :4]) == 0


def test_native_heat_transpose_rejects_wrong_observation_axes():
    fixture = _fixture_module()
    model = fixture._model().eval()
    prepared = fixture._prepared(model)
    with pytest.raises(ValueError, match="observation axes"):
        model.heat_vjp_native(prepared, {"pred_interface": torch.zeros(1, 2, 15, 2)})
    with pytest.raises(ValueError, match="actual extracted"):
        model.heat_vjp_native(prepared, {"unsupported": torch.zeros(1)})


def test_integer_physical_controls_are_converted_before_native_contraction():
    fixture = _fixture_module()
    model = fixture._model().eval()
    prepared = fixture._prepared(model)
    actual = model.apply_increment(prepared, torch.tensor([[1, 0]]))
    expected = model.apply_increment(prepared, torch.tensor([[1.0, 0.0]]))
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
