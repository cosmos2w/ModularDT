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


def _executor_probe_fixture(mutation=None):
    """Small three-phase wrapper using the actual typed physical executors.

    This verifies the probe itself; retained native checkpoint evidence is a
    separate opt-in test and an admitted GPU measurement, not this fixture.
    """
    from honf_forward_core.interface_fields.adaptive_interaction_cover import InteractionContext
    from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
    from honf_forward_core.interface_fields.types import EncodedInterfaceCase, InterfaceRead

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = TypedHypergraphField(8, 12, 2, 2,
                architecture="local_overlap_hypergraph_honf", spatial_dim=2, module_characteristic_length=.03)

        def prepare(self, encoded, phase):
            state = self.backend.prepare(encoded, encoded.module_tokens,
                interaction_context=InteractionContext(phase=phase))
            return SimpleNamespace(backend_state=state, encoded=encoded, module_states=state["module_tokens"],
                coarse_state=state["env_tokens"], interaction_aux=self.backend.preparation_aux(state))

        def read(self, prepared, query):
            context, auxiliary = self.backend.read(prepared.backend_state, prepared.encoded, query,
                torch.cat((query, query.square(), query.sin()), -1))
            return InterfaceRead(context, auxiliary)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core, self.encoder = Core(), torch.nn.Linear(2, 8)
            self.stage_a = torch.nn.Linear(8, 2).requires_grad_(False)
            self.head = torch.nn.Linear(8, 4)
            self.unused = torch.nn.Parameter(torch.randn(2))
            self.register_buffer("env", torch.randn(1, 5, 8))
            self.register_buffer("centres", torch.rand(1, 3, 2))
            self.register_buffer("env_coords", torch.rand(1, 5, 2))

        def forward(self, structure, query_xy, local_module_params):
            heat = structure["heat_powers"]
            tokens = self.encoder(torch.stack((heat, local_module_params[..., 0]), -1))
            encoded = EncodedInterfaceCase(tokens, self.env, tokens.mean(1), self.centres, self.env_coords,
                structure["module_present"], torch.cat((self.centres, heat[..., None]), -1),
                None, torch.ones(1, 5), torch.ones(2), env_characteristic_lengths=torch.full((1, 5), .03))
            contexts = [self.core.read(self.core.prepare(encoded, f"P{phase}"), query_xy).context for phase in range(3)]
            field = self.head(contexts[2])
            if self.core.backend.execution_mode == "rectangular_subset":
                if mutation == "output":
                    field = field + .01
                if mutation == "gradient":
                    field = field + .01 * (field - field.detach())
                if mutation == "buffer":
                    self.env.add_(.01)
                if mutation == "raise":
                    raise RuntimeError("intentional probe failure")
            ports0, ports1 = self.head(contexts[0]), self.head(contexts[1])
            return {"pred_field": field, "pred_interface": self.head(contexts[1]),
                "pred_internal_temperature": self.stage_a(contexts[1]),
                "pred_port_condition_raw": ports0, "pred_port_condition": ports1,
                "local_port_condition_used": ports0, "module_response_latent": contexts[1],
                **{name: field[..., :0] for name in benchmark.PARITY_OUTPUTS[7:]}}

    torch.manual_seed(414)
    model = Wrapper().eval()
    arguments = {"structure": {"heat_powers": torch.tensor([[2., 3., 0.]]),
        "module_present": torch.tensor([[1., 1., 0.]])}, "query_xy": torch.rand(1, 7, 2),
        "local_module_params": torch.cat((torch.tensor([[[2.], [3.], [0.]]]), torch.zeros(1, 3, 6)), -1)}
    with torch.no_grad():
        model(**arguments)
    flags = {name: value.requires_grad for name, value in model.named_parameters()}
    model.requires_grad_(False)
    return model, flags, arguments


def test_executor_parity_actual_typed_first_gradients_and_restored_state():
    torch.set_num_threads(1)
    model, flags, arguments = _executor_probe_fixture()
    model.core.backend.set_execution_mode("rectangular_subset", receiver_chunk_size=3)
    rng = torch.get_rng_state().clone()
    checked = benchmark.verify_executor_parity(model, arguments, {}, flags, receiver_chunk=2)
    assert checked["passed"]
    assert checked["tolerance"]["output"] == {"rtol": 2e-5, "atol": 2e-5}
    assert checked["tolerance"]["first_gradient"] == {"rtol": 2e-5, "atol": 1e-6}
    assert checked["active_parameter_gradient_tensors"] > 20
    assert checked["active_input_gradient_names"] == ["input.physical_heat", "input.query_xy"]
    assert checked["unused_gradient_names"]["dense_masked_reference"] == checked["unused_gradient_names"]["rectangular_subset"]
    assert "unused" in checked["unused_gradient_names"]["dense_masked_reference"]
    assert checked["native_frozen_parameter_names"] == ["stage_a.weight", "stage_a.bias"]
    assert set(checked["adjoint"]["roles"]) == set(benchmark.ADJOINT_OUTPUTS)
    assert checked["output_roles"]["pred_port_global_temperature_target"] == "non_predictive_port_target"
    assert checked["output_roles"]["pred_port_global_consistency_mask"] == "non_predictive_port_mask"
    assert {name.split(".")[0] for name in checked["output_checks"] if name.startswith("P")} == {"P0", "P1", "P2"}
    assert checked["restored_frozen_state"]["state_dict_unchanged_bitwise"]
    assert all(value["passed"] for value in checked["applied_support_inventory_checks"].values())
    assert len(checked["applied_support_inventory_checks"]) == 15
    assert model.core.backend.execution_mode == "rectangular_subset" and model.core.backend.execution_receiver_chunk == 3
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
    assert torch.equal(rng, torch.get_rng_state())
    assert "prepare" not in model.core.__dict__ and "read" not in model.core.__dict__
    assert "prepare" not in model.core.backend.__dict__ and "_access" not in model.core.backend.__dict__
    for execution in checked["forward_support_and_work"].values():
        assert set(execution["forward_work"]["hard"]) == {"P0", "P1", "P2"}
        assert execution["fine_kernel_work"]["measured"]


@pytest.mark.parametrize("mutation", ["output", "gradient"])
def test_executor_parity_detects_value_and_gradient_only_mismatch(mutation):
    model, flags, arguments = _executor_probe_fixture(mutation)
    checked = benchmark.verify_executor_parity(model, arguments, {}, flags, receiver_chunk=2)
    assert not checked["passed"]
    if mutation == "gradient":
        assert all(value["passed"] for value in checked["output_checks"].values())
        assert not all(value["passed"] for value in checked["first_gradient_checks"].values())
    else:
        assert not checked["output_checks"]["pred_field"]["passed"]
    assert checked["restored_frozen_state"]["passed"]


@pytest.mark.parametrize("mutation", ["buffer", "raise"])
def test_executor_parity_restores_hooks_flags_executor_on_exception(mutation):
    model, flags, arguments = _executor_probe_fixture(mutation)
    expected = "state tensor changed: env" if mutation == "buffer" else "intentional probe failure"
    with pytest.raises(RuntimeError, match=expected):
        benchmark.verify_executor_parity(model, arguments, {}, flags, receiver_chunk=2)
    assert model.core.backend.execution_mode == "dense_masked_reference"
    assert all(not value.requires_grad and value.grad is None for value in model.parameters())
    assert "prepare" not in model.core.__dict__ and "read" not in model.core.__dict__


def test_parity_heat_leaves_preserve_values_and_physical_normalization():
    _, _, arguments = _executor_probe_fixture()
    checkpoint = {"train_config": {"dataset": {"normalize_inputs": True}},
        "global_normalization_stats": {"heat_power_mean": [7.], "heat_power_std": [2.]}}
    live, leaves = benchmark.differentiable_arguments(arguments, checkpoint)
    assert torch.equal(live["structure"]["heat_powers"], arguments["structure"]["heat_powers"])
    assert torch.equal(live["local_module_params"], arguments["local_module_params"])
    assert torch.equal(live["query_xy"], arguments["query_xy"])
    gradient = torch.autograd.grad(live["structure"]["heat_powers"].sum() + live["local_module_params"][..., 0].sum(),
        leaves["input.physical_heat"])[0]
    torch.testing.assert_close(gradient, torch.tensor([[1.5, 1.5, .5]]))


def test_parity_adjoint_treats_material_positions_as_one_temperature_channel():
    outputs = {name: torch.ones(1, 2, 3) for name in benchmark.ADJOINT_OUTPUTS}
    outputs["pred_internal_temperature"] = torch.tensor([[[2., 4., 6.], [1., 3., 5.]]])
    weights, scales = benchmark.common_adjoint(outputs)
    assert len(scales["pred_internal_temperature"]) == 1
    assert scales["pred_internal_temperature"][0] == pytest.approx((91. / 6.) ** .5)
    assert weights["pred_internal_temperature"].shape == (1, 2, 3)
    assert not any(value.requires_grad for value in weights.values())


def test_parity_nonfinite_values_fail_with_serializable_evidence():
    checked = benchmark.tensor_error(torch.tensor([0., 1.]), torch.tensor([float("nan"), 1.]), rtol=2e-5, atol=1e-6)
    assert not checked["passed"] and not checked["finite"]
    import json
    json.dumps(checked, allow_nan=False)


def test_failed_executor_parity_never_starts_timing(monkeypatch, tmp_path):
    model, flags, arguments = _executor_probe_fixture("gradient")
    for name, value in model.named_parameters():
        value.requires_grad_(flags[name])
    reference = {"case_id": "0001", "structure": {"module_present": np.ones(2)}}
    monkeypatch.setattr(benchmark, "load_native", lambda *_args, **_kwargs: (model, {}, [reference], [reference], tmp_path))
    monkeypatch.setattr(benchmark, "low_high_indices", lambda *_args, **_kwargs: [0])
    monkeypatch.setattr(benchmark, "query_panels", lambda *_args: [("small_inverse", np.zeros((7, 2)))])
    monkeypatch.setattr(benchmark, "native_arguments", lambda *_args: arguments)
    def timing_must_not_run(*_args, **_kwargs):
        pytest.fail("Timing started after failed parity")
    monkeypatch.setattr(benchmark, "benchmark_model_case", timing_must_not_run)
    args = benchmark.parse_args(["--checkpoint", str(tmp_path / "model.pt"), "--output-dir", str(tmp_path), "--verify-executor-parity"])
    with pytest.raises(RuntimeError, match="parity failed"):
        benchmark.evaluate(args)
    import json
    saved = json.loads((tmp_path / "executor_parity.json").read_text())
    assert not saved["passed"] and not (tmp_path / "timing.json").exists()


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


@pytest.mark.parametrize("variant", ["original", "permutation", "split_equal", "split_unequal", "split_multiple"])
def test_whole_wrapper_environment_equivalence_preserves_explicit_measure_and_length(variant):
    from channelthermal.environment import ChannelThermalEnvironment
    coords = torch.tensor([[[0., 0.], [1., .2], [3., .4]]], dtype=torch.float64)
    features = torch.arange(21, dtype=torch.float64).reshape(1, 3, 7)
    mass = torch.tensor([[.4, 1.5, 2.3]], dtype=torch.float64)
    lengths = torch.tensor([[.7, .3, 1.1]], dtype=torch.float64)
    environment = ChannelThermalEnvironment(coords, features, env_weights=mass, env_characteristic_lengths=lengths)
    builder = topology.RefinedEnvironmentBuilder(lambda **kwargs: environment, variant, gradients=True)
    actual = builder(batch_size=1)
    torch.testing.assert_close(actual.env_weights.sum(1), mass.sum(1), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(actual.env_characteristic_lengths, lengths[:, builder.parents], atol=0, rtol=0)
    torch.testing.assert_close(actual.env_features, features[:, builder.parents], atol=0, rtol=0)
    integral = (actual.env_features[..., 0] * actual.env_weights).sum()
    grad_feature, grad_mass = torch.autograd.grad(integral, (builder.base.env_features, builder.base.env_weights))
    torch.testing.assert_close(grad_feature[..., 0], mass, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(grad_mass, features[..., 0], atol=1e-14, rtol=1e-14)
    mapped = topology._pullback_atom_axis(actual.env_features, builder.parents, builder.fractions, 1, 3)
    torch.testing.assert_close(mapped, features, atol=1e-14, rtol=1e-14)


def test_whole_wrapper_environment_refuses_measure_fallback_and_nonfinite_evidence_is_serializable():
    from channelthermal.environment import ChannelThermalEnvironment
    builder = topology.RefinedEnvironmentBuilder(lambda **kwargs: ChannelThermalEnvironment(
        torch.zeros(1, 2, 2), torch.zeros(1, 2, 7)), "split_equal")
    with pytest.raises(ValueError, match="adapter-owned masses"):
        builder()
    import json
    measured = topology.numerical_check(torch.ones(2), torch.tensor([torch.nan, torch.inf]))
    assert not measured["passed"] and not measured["finite"]
    json.dumps(measured, allow_nan=False)


def test_control_locality_probe_discloses_manipulated_exclusion_and_restores_phase_hooks(monkeypatch):
    from honf_forward_core.interface_fields.adaptive_receiver_hypergraph import AdaptiveReceiverHypergraph

    from .test_adaptive_receiver_hypergraph import _case

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = torch.nn.Module()
            self.backend.organizer = AdaptiveReceiverHypergraph(8, organizer_dim=16, faithful_controls=True,
                measure_consistent=True).eval()

        def decode_queries(self, prepared, query, **kwargs):
            return {"pred_field": query}

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Core()
            self.encoded = _case()

        def forward(self, query_xy):
            for phase in range(3):
                self.core.backend.organizer.prepare(self.encoded, self.encoded.module_tokens, phase=phase)
            return {"pred_field": query_xy}

    model = Wrapper().eval()
    query = torch.ones(1, 7, 2)
    monkeypatch.setattr(topology, "sensor_panel", lambda sample: (query[0].numpy(), None, None, None, None))
    monkeypatch.setattr(topology, "native_arguments", lambda *args: {"query_xy": query})
    measured, arrays = topology.trained_control_locality(model, {}, {"case_id": "unit_fixture"})
    assert measured["excluded_control_derivative_zero"]
    assert measured["exclusion_origin"] == "explicitly manipulated conditional diagnostic"
    assert set(measured["ancestry_and_actual_donor_inventory"]) == {"P0", "P1", "P2"}
    assert "P2/QE/control_membership_E" in arrays and "P0/QM/control_membership_M" in arrays
    assert "prepare" not in model.core.backend.organizer.__dict__
    assert "access" not in model.core.backend.organizer.__dict__
    assert "decode_queries" not in model.core.__dict__


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
