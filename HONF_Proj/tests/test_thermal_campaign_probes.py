"""Independent timing scopes and correctly defined numerical probe controls."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"

def _tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

benchmark = _tool("thermal_campaign_benchmark")
topology = _tool("thermal_campaign_topology")
heat_inference = _tool("thermal_campaign_heat_inference")


def _frozen_forward():
    model = torch.nn.Linear(2, 1)
    model.register_buffer("counter", torch.tensor(0, dtype=torch.int64))
    model.register_buffer("special_values", torch.tensor([0., float("nan")]))
    return model.eval().requires_grad_(False)


def test_forward_freeze_snapshot_covers_loaded_parameters_and_persistent_buffers():
    model = _frozen_forward()
    snapshot = heat_inference.snapshot_forward_state(model)
    checked = heat_inference.verify_frozen_forward(model, snapshot)
    assert checked["passed"] and checked["state_dict_unchanged_bitwise"]
    assert checked["state_dict_tensors_checked"] == 4
    assert checked["state_dict_scalars_checked"] == 6
    assert checked["trainable_forward_parameter_tensors"] == 0
    assert checked["forward_parameter_gradient_tensors"] == 0
    for name, value in model.state_dict().items():
        assert snapshot[name].device.type == "cpu" and not snapshot[name].requires_grad
        assert snapshot[name].data_ptr() != value.data_ptr()


@pytest.mark.parametrize("name", ["weight", "counter"])
def test_forward_freeze_rejects_changed_parameter_or_buffer(name):
    model = _frozen_forward()
    snapshot = heat_inference.snapshot_forward_state(model)
    with torch.no_grad():
        getattr(model, name).add_(1)
    with pytest.raises(RuntimeError, match=f"state tensor changed: {name}"):
        heat_inference.verify_frozen_forward(model, snapshot)


def test_forward_freeze_compares_bit_patterns_including_signed_zero_and_nan():
    model = _frozen_forward()
    snapshot = heat_inference.snapshot_forward_state(model)
    assert heat_inference.verify_frozen_forward(model, snapshot)["passed"]
    model.special_values[0] = -0.
    with pytest.raises(RuntimeError, match="state tensor changed: special_values"):
        heat_inference.verify_frozen_forward(model, snapshot)


@pytest.mark.parametrize("mutation", ["requires_grad", "gradient", "training_mode"])
def test_forward_freeze_rejects_trainable_parameters_gradients_or_training_mode(mutation):
    model = _frozen_forward()
    snapshot = heat_inference.snapshot_forward_state(model)
    if mutation == "requires_grad":
        model.weight.requires_grad_(True)
    elif mutation == "gradient":
        model.weight.grad = torch.ones_like(model.weight)
    else:
        model.train()
    with pytest.raises(RuntimeError, match="not frozen|left evaluation mode"):
        heat_inference.verify_frozen_forward(model, snapshot)


def test_repeats_have_one_scope_and_median_p90(monkeypatch):
    calls = []
    values = iter((0., .02, 1., 1.04, 2., 2.06))
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(values))
    result = benchmark.latency_samples(lambda: calls.append(1), torch.device("cpu"), repeats=3, warmup=2)
    assert len(calls) == 5
    assert result["median_seconds"] == pytest.approx(.04)
    assert result["p90_seconds"] == pytest.approx(.056)
    assert all(sample["cuda_peak_allocated_bytes"] is None for sample in result["samples"])
    assert "not a call peak" in result["memory_scope"]


def test_low_high_m_selection_and_gpu_mapping(monkeypatch):
    ids = ["0001", "0002", "0003", "0273"]
    dataset = SimpleNamespace(selected_case_ids=ids,
        h5={"cases": {name: {"module_present": np.ones(count)} for name, count in zip(ids, (4, 7, 3, 1))}})
    assert benchmark.low_high_indices(dataset) == [2, 1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert benchmark.allowed_device("cuda:0") == torch.device("cuda:0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(ValueError, match="physical GPU"):
        benchmark.allowed_device("cuda:0")


def test_finite_differences_are_per_physical_channel_and_do_not_invent_topology():
    def forward(parameter):
        return {"fields": torch.arange(6, dtype=parameter.dtype).reshape(2, 3) * parameter,
                "interface": torch.ones(2, 2) * parameter.square(), "solid": torch.ones(2, 2) * parameter}, None
    arrays, rows = topology.finite_difference_probe(forward)
    assert arrays["autograd_derivative_vector"].shape == (14,)
    assert len(rows) == 3
    assert rows[0]["within_same_discrete_topology"] is None
    assert len(rows[0]["derivative_error"]["fields"]["rmse_by_channel"]) == 3
    assert max(rows[0]["derivative_error"]["fields"]["max_abs_by_channel"]) < 1e-5


@pytest.mark.parametrize("architecture", ["adaptive_receiver_hypergraph_honf", "overlap_control_hypergraph_honf", "local_overlap_hypergraph_honf"])
def test_fine_invariants_and_physical_atom_split_definition(architecture):
    from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
    from honf_forward_core.interface_fields.types import EncodedInterfaceCase
    torch.manual_seed(93)
    encoded = EncodedInterfaceCase(torch.randn(1, 4, 8), torch.randn(1, 7, 8), torch.randn(1, 8),
        torch.rand(1, 4, 2), torch.rand(1, 7, 2), torch.ones(1, 4), torch.randn(1, 4, 3),
        None, torch.ones(1, 7), torch.ones(2), env_characteristic_lengths=torch.full((1, 7), .03))
    backend = TypedHypergraphField(8, 12, 2, 2, architecture=architecture, spatial_dim=2,
        module_characteristic_length=.03).eval()
    query, features = torch.rand(1, 5, 2), torch.rand(1, 5, 6)
    with torch.no_grad():
        backend.read(backend.prepare(encoded, encoded.module_tokens), encoded, query, features)
        for head in backend.organizer.control_heads.values():
            final = head[-1] if architecture.startswith("adaptive") else head.net[-1]
            final.weight.normal_(0, .2)
        for gain in backend.control_gain.values():
            gain.weight.normal_(0, .2)
        checks, arrays = topology.fine_core_invariance(backend, encoded, encoded.module_tokens, query, features)
    for key in ("module_permutation", "inactive_module_padding", "quadrature_atom_split_fixed_prepared_action"):
        assert checks[key]["passed"], (key, checks[key])
    assert "quadrature_atom_split_with_organizer_rebuild" in checks
    assert arrays["original"].shape == (1, 5, 8)


@pytest.mark.skipif(not os.environ.get("HONF_ALIGNMENT_THERMAL_CHECKPOINT"), reason="Needs retained native Thermal resources")
def test_native_complete_wrapper_and_prepared_decode_are_separate_cpu_scopes():
    model, _, normalized, raw, _ = benchmark.load_native(Path(os.environ["HONF_ALIGNMENT_THERMAL_CHECKPOINT"]), None, torch.device("cpu"))
    rows = benchmark.benchmark_model_case(model, normalized[0], raw[0], torch.device("cpu"), repeats=1, warmup=0)
    assert len(rows) == 4
    for row in rows:
        phases = row["forward_work"]["hard"].values()
        prepares = sum(phase["prepare_calls"] for phase in phases)
        assert prepares == (3 if row["scope"].startswith("complete") else 0)
        assert row["repeats"] == 1
