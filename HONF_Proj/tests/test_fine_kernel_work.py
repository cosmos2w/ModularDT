"""Actual native MLP rows, physical gradient parity and recorder lifetime."""

import importlib.util
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from honf_forward_core.evaluation.fine_kernel_work import FineKernelWork
from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _case(dimension=2, modules=4):
    generator = torch.Generator().manual_seed(172)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(2, modules, 8, generator=generator),
        env_tokens=torch.randn(2, 7, 8, generator=generator),
        global_token=torch.randn(2, 8, generator=generator),
        module_centers=torch.rand(2, modules, dimension, generator=generator),
        env_coords=torch.rand(2, 7, dimension, generator=generator),
        module_present=torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])[:, :modules],
        module_features=torch.randn(2, modules, 3, generator=generator), env_features=None,
        env_weights=torch.ones(2, 7), coordinate_scale=torch.ones(1, 1, dimension),
    )


def _backend(kind, dimension=2):
    if kind == "dense":
        return DensePairwiseField(8, 12, 2, 2).eval()
    backend = TypedHypergraphField(
        8, 12, 2, 2, architecture=kind, spatial_dim=dimension,
        module_characteristic_length=.03,
        options={"organizer_dim": 16} if kind == "adaptive_receiver_hypergraph_honf" else {},
    ).eval()
    backend.organizer.set_epoch(301)
    if kind == "adaptive_receiver_hypergraph_honf":
        with torch.no_grad():
            backend.organizer.geometry_strength["QE"].fill_(8)
            backend.organizer.geometry_strength["QM"].fill_(4)
    return backend


def _pass(backend, encoded, query, features):
    state = backend.prepare(encoded, encoded.module_tokens)
    result, auxiliary = backend.read(state, encoded, query, features)
    return result, {**state.get("hypergraph_ledger", {}), **auxiliary}


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("kind,executor", [
    ("dense", None),
    ("adaptive_receiver_hypergraph_honf", "dense_masked_reference"),
    ("overlap_control_hypergraph_honf", "dense_masked_reference"),
    ("local_overlap_hypergraph_honf", "dense_masked_reference"),
    ("adaptive_receiver_hypergraph_honf", "rectangular_subset"),
    ("overlap_control_hypergraph_honf", "rectangular_subset"),
    ("local_overlap_hypergraph_honf", "rectangular_subset"),
])
def test_real_backends_preserve_outputs_first_gradients_and_count_actual_rows(dimension, kind, executor):
    encoded = _case(dimension)
    query, features = torch.rand(2, 5, dimension), torch.rand(2, 5, 6)
    backend = _backend(kind, dimension)
    with torch.no_grad():
        _pass(backend, encoded, query, features)  # materialize lazy physical modules
    if executor:
        backend.set_execution_mode(executor, receiver_chunk_size=2)
    observed = deepcopy(backend)
    outputs, gradients, input_gradients = {}, {}, {}
    for label, model in (("reference", backend), ("recorded", observed)):
        record = replace(encoded, module_tokens=encoded.module_tokens.clone().requires_grad_(),
                         env_tokens=encoded.env_tokens.clone().requires_grad_())
        receiver = query.clone().requires_grad_()
        if label == "recorded":
            with FineKernelWork(model) as counted:
                value, ledger = _pass(model, record, receiver, features)
        else:
            value, _ = _pass(model, record, receiver, features)
        value.square().mean().backward()  # recorder removed before backward
        outputs[label] = value.detach()
        gradients[label] = {name: parameter.grad for name, parameter in model.named_parameters()}
        input_gradients[label] = (record.module_tokens.grad, record.env_tokens.grad, receiver.grad)
    torch.testing.assert_close(outputs["recorded"], outputs["reference"], rtol=0, atol=0)
    assert input_gradients["reference"][-1].abs().sum() > 0
    for observed_grad, reference_grad in zip(input_gradients["recorded"], input_gradients["reference"], strict=True):
        torch.testing.assert_close(observed_grad, reference_grad, rtol=0, atol=0)
    assert any(value is not None and value.abs().sum() > 0 for value in gradients["reference"].values())
    for name, gradient in gradients["reference"].items():
        if gradient is None:
            assert gradients["recorded"][name] is None
        else:
            torch.testing.assert_close(gradients["recorded"][name], gradient, rtol=0, atol=0)
    counts = counted.snapshot()["routes"]
    if kind == "dense":
        expected = {"MM": 32, "ME": 56, "EM": 56, "QM": 40, "QE": 70}
        assert counts == {route: {"padded_input_rows": rows, "calls": 1} for route, rows in expected.items()}
    else:
        for route, entry in counts.items():
            assert entry["padded_input_rows"] == int(ledger[f"hypergraph_{route}_executed_rows"])
            assert entry["calls"] == int(ledger[f"hypergraph_{route}_fine_calls"])
        if executor == "rectangular_subset":
            assert counts["MM"]["padded_input_rows"] < 32  # actual omitted padded rectangles


def test_checkpoint_backward_is_outside_counts_and_exception_restores_existing_hooks():
    encoded = _case()
    query, features = torch.rand(2, 5, 2), torch.rand(2, 5, 6)
    backend = DensePairwiseField(8, 12, 2, 2, activation_checkpointing=True).train()
    _pass(backend, encoded, query, features)
    existing_calls = []
    handle = backend.mm_message.register_forward_hook(lambda *_: existing_calls.append("existing"))
    before = set(backend.mm_message._forward_hooks)
    with FineKernelWork(backend) as counted:
        value, _ = _pass(backend, encoded, query, features)
    snapshot = counted.snapshot()
    value.square().mean().backward()
    assert counted.snapshot() == snapshot
    assert set(backend.mm_message._forward_hooks) == before
    with pytest.raises(RuntimeError, match="proposal failed"), FineKernelWork(backend):
        _pass(backend, encoded, query, features)
        raise RuntimeError("proposal failed")
    assert set(backend.mm_message._forward_hooks) == before
    assert existing_calls
    handle.remove()


def test_zero_module_axis_counts_successful_calls_without_fabricated_rows():
    backend = DensePairwiseField(8, 12, 2, 2).eval()
    with FineKernelWork(backend) as counted:
        output, _ = _pass(backend, _case(modules=0), torch.rand(2, 3, 2), torch.rand(2, 3, 6))
    assert torch.isfinite(output).all()
    for route in ("MM", "ME", "EM", "QM"):
        assert counted.records[route] == {"padded_input_rows": 0, "calls": 1}
    assert counted.records["QE"] == {"padded_input_rows": 42, "calls": 1}


def test_reentry_snapshot_independence_and_unsupported_backend():
    backend = _backend("dense")
    encoded = _case()
    recorder = FineKernelWork(backend)
    with recorder:
        _pass(backend, encoded, torch.rand(2, 5, 2), torch.rand(2, 5, 6))
        with pytest.raises(RuntimeError, match="concurrently"):
            recorder.__enter__()
    saved = recorder.snapshot()
    with recorder:
        _pass(backend, encoded, torch.rand(2, 3, 2), torch.rand(2, 3, 6))
    assert saved["routes"]["QE"]["padded_input_rows"] == 70
    assert recorder.records["QE"]["padded_input_rows"] == 42
    with pytest.raises(TypeError, match="five"):
        FineKernelWork(nn.Linear(2, 2))


def test_benchmark_fallback_keeps_unsupported_historical_backend_available():
    tool = Path(__file__).resolve().parents[1] / "tools/thermal_campaign_benchmark.py"
    spec = importlib.util.spec_from_file_location("fine_work_fallback_benchmark", tool)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    backend = nn.Linear(2, 2)
    inputs = torch.rand(3, 2)
    reference = backend(inputs)
    with benchmark.optional_fine_work(backend) as work:
        observed = backend(inputs)
    torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert work["measured"] is False and work["routes"] is None
    assert "unmeasured" in work["scope"] and "five" in work["reason"]
    assert not backend._forward_hooks


@pytest.mark.skipif(not os.environ.get("HONF_ALIGNMENT_THERMAL_CHECKPOINT"), reason="Needs retained native Thermal resources")
def test_native_benchmark_counts_only_untimed_passes(monkeypatch):
    tool = Path(__file__).resolve().parents[1] / "tools/thermal_campaign_benchmark.py"
    spec = importlib.util.spec_from_file_location("fine_work_benchmark", tool)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    model, _, normalized, raw, _ = benchmark.load_native(
        Path(os.environ["HONF_ALIGNMENT_THERMAL_CHECKPOINT"]), None, torch.device("cpu"),
    )
    original = benchmark.latency_samples

    def uninstrumented(call, device, **settings):
        assert all(not module._forward_hooks for module in FineKernelWork(model.core.backend).modules.values())
        return original(call, device, **settings)

    monkeypatch.setattr(benchmark, "latency_samples", uninstrumented)
    rows = benchmark.benchmark_model_case(model, normalized[0], raw[0], torch.device("cpu"), repeats=1, warmup=0)
    assert len(rows) == 4
    for row in rows:
        routes = row["fine_kernel_work"]["routes"]
        for route in ("MM", "ME", "EM"):
            if row["scope"].startswith("prepared"):
                assert routes[route] == {"padded_input_rows": 0, "calls": 0}
            else:
                assert routes[route]["padded_input_rows"] > 0 and routes[route]["calls"] == 3
        assert routes["QM"]["padded_input_rows"] > 0 and routes["QE"]["padded_input_rows"] > 0
    assert all(not module._forward_hooks for module in FineKernelWork(model.core.backend).modules.values())
