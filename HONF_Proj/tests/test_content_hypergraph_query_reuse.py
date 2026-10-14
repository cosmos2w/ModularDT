"""Retained query factors implement the same affine map without new reads."""

import importlib.util
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location("query_reuse_fixtures", Path(__file__).with_name("test_content_hypergraph_core.py"))
_FIXTURES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURES)
_core, _query, _scene = _FIXTURES._core, _FIXTURES._query, _FIXTURES._scene


def double_tree(value):
    return replace(value, **{f.name: getattr(value, f.name).double() for f in fields(value)
                             if torch.is_tensor(getattr(value, f.name)) and getattr(value, f.name).is_floating_point()})


@pytest.mark.parametrize("architecture", ["A-D", "A-H6"])
def test_cached_affine_fields_and_transpose_need_no_neural_reader(architecture, monkeypatch):
    core = _core(architecture).double()
    organization = core.prepare(double_tree(_scene()))
    query = double_tree(_query())
    heat = torch.tensor([[0.2, -0.1, 0.9]], dtype=torch.float64)
    tangent = heat * 1e-4
    expected = core.decode(organization, query, heat)
    kernel = core.export_source_kernels(organization, query, accumulation_dtype=torch.float64)
    cached = core.prepare_affine_query(organization, query)
    for reader in (core.local_reader, core.edge_reader, core.direct_reader):
        if reader is not None:
            monkeypatch.setattr(reader, "forward", lambda *a, **k: pytest.fail("Repeated neural read"))
    monkeypatch.setattr(core, "export_source_kernels", lambda *a, **k: pytest.fail("Dense K export"))
    result = core.apply_prepared_affine(cached, heat, accumulation_dtype=torch.float64)
    increment = core.apply_prepared_affine(cached, tangent, accumulation_dtype=torch.float64)
    torch.testing.assert_close(result.thermal_temperature, expected.thermal_temperature, atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(increment.thermal_temperature, torch.einsum("bqm,bm->bq", kernel, tangent), atol=1e-9, rtol=1e-7)
    assert torch.equal(result.nonlinear_field, expected.nonlinear_field)
    cotangent = torch.tensor([[0.3, 0.2, -0.4, 0.7]], dtype=torch.float64)
    gradient = core.heat_vjp_prepared_affine(cached, cotangent, accumulation_dtype=torch.float64)
    torch.testing.assert_close(gradient, torch.einsum("bq,bqm->bm", cotangent, kernel), atol=1e-9, rtol=1e-7)
    assert result.work_counts["edge_rows"] == result.work_counts["direct_source_rows"] == 0
    if architecture == "A-H6":
        assert cached.direct_coefficients is None
        assert cached.basis.shape[-2:] == (6, 4)


@pytest.mark.parametrize("mutation", ["coordinates", "query_ids", "factor", "weights", "coefficients"])
def test_cached_query_rejects_receiver_factor_and_support_mutation(mutation):
    core = _core()
    organization = core.prepare(_scene())
    query = _query()
    cached = core.prepare_affine_query(organization, query)
    with torch.no_grad():
        if mutation == "factor":
            cached.basis.add_(0.01)
        elif mutation == "weights":
            cached.local_pairs.support_weight.mul_(0)
        elif mutation == "coefficients":
            organization.source_coefficients.add_(0.01)
        else:
            getattr(query, mutation).add_(1)
    with pytest.raises(ValueError, match="changed"):
        core.apply_prepared_affine(cached, torch.ones(1, 3))
