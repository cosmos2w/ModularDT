"""Numerical identity, honest executor accounting and isolated soft gradients."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase
from honf_forward_core.training.hypergraph_shadow import hard_value_soft_hypergraph_forward


def _encoded(dimension=2):
    generator = torch.Generator().manual_seed(731)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(2, 4, 8, generator=generator),
        env_tokens=torch.randn(2, 7, 8, generator=generator),
        global_token=torch.randn(2, 8, generator=generator),
        module_centers=torch.rand(2, 4, dimension, generator=generator),
        env_coords=torch.rand(2, 7, dimension, generator=generator),
        module_present=torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]]),
        module_features=torch.randn(2, 4, 3, generator=generator), env_features=None,
        env_weights=torch.tensor([[1., .5, 2., 1., 3., .75, 1.]]).expand(2, -1),
        coordinate_scale=torch.ones(1, 1, dimension))


def _typed(dimension=2):
    return TypedHypergraphField(8, 12, 2, 2, architecture="adaptive_receiver_hypergraph_honf",
                                spatial_dim=dimension, module_characteristic_length=0.1,
                                options={"organizer_dim": 16})


@pytest.mark.parametrize("dimension", [2, 3])
def test_all_access_identity_matches_b_fine_outputs_and_first_gradients(dimension):
    encoded = _encoded(dimension)
    query = torch.rand(2, 5, dimension)
    features = torch.rand(2, 5, 6)
    dense = DensePairwiseField(8, 12, 2, 2).eval()
    candidate = _typed(dimension).eval()
    dense.read(dense.prepare(encoded, encoded.module_tokens), encoded, query, features)
    candidate.read(candidate.prepare(encoded, encoded.module_tokens), encoded, query, features)
    compatible = dense.state_dict()
    candidate.load_state_dict({**candidate.state_dict(), **compatible})
    for model in (dense, candidate):
        model.zero_grad(set_to_none=True)
    physical_inputs = {}
    predictions = {}
    preparations = {}
    for name, model in (("dense", dense), ("candidate", candidate)):
        states = encoded.module_tokens.clone().requires_grad_()
        env = encoded.env_tokens.clone().requires_grad_()
        coord = query.clone().requires_grad_()
        record = replace(encoded, module_tokens=states, env_tokens=env)
        prepared = model.prepare(record, states)
        value, _ = model.read(prepared, record, coord, features)
        value.square().sum().backward()
        physical_inputs[name] = (states, env, coord)
        predictions[name], preparations[name] = value, prepared
    torch.testing.assert_close(predictions["candidate"], predictions["dense"], rtol=2e-5, atol=2e-6)
    for source, target in zip(physical_inputs["candidate"], physical_inputs["dense"]):
        torch.testing.assert_close(source.grad, target.grad, rtol=3e-5, atol=3e-6)
    dense_parameters = dict(dense.named_parameters())
    for name, parameter in candidate.named_parameters():
        if name in dense_parameters:
            torch.testing.assert_close(parameter.grad, dense_parameters[name].grad, rtol=3e-5, atol=3e-6)
    for key in ("module_tokens", "env_tokens"):
        torch.testing.assert_close(preparations["candidate"][key], preparations["dense"][key])


@pytest.mark.parametrize("dimension", [2, 3])
def test_registered_core_multi_chunk_backward_and_exact_row_ledger(dimension):
    encoded = _encoded(dimension)
    config = UnifiedForwardConfig(spatial_dim=dimension, coordinate_scale=[1.] * dimension,
        boundary_feature_mode="none", periodic_axes=[],
        field_dim=3, hidden_dim=8, forward_architecture="adaptive_receiver_hypergraph_honf",
        interface_model=InterfaceFieldConfig(message_hidden_dim=12, attention_heads=2,
            relative_fourier_frequencies=2, receiver_chunk_size=3,
            hypergraph_options={"organizer_dim": 16}))
    core = InterfaceFieldCore(config).train()
    batch = BatchData(module_centers=encoded.module_centers, module_present=encoded.module_present,
        module_features=encoded.module_features, global_context=encoded.global_token,
        query_xy=torch.rand(2, 11, dimension), target_field=None, query_time=None,
        case_name="typed-hypergraph-test", metadata={},
        env_coords=encoded.env_coords, env_features=torch.randn(2, 7, 3), env_weights=encoded.env_weights)
    inputs = core.encode_case(batch)
    prepared = core.prepare(inputs, inputs.module_tokens)
    output = core.decode_queries(prepared, batch.query_xy, receiver_chunk_size=3, return_routing_maps=True)
    output["pred_field"].square().mean().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in core.parameters())
    complete = core.decode_queries(prepared, batch.query_xy, receiver_chunk_size=30)
    torch.testing.assert_close(complete["pred_field"], output["pred_field"])
    aux = output
    assert aux["hypergraph_QM_executed_rows"].item() == 2 * 11 * 4
    assert aux["hypergraph_QE_executed_rows"].item() == 2 * 11 * 7
    assert aux["hypergraph_QM_fine_calls"].item() == 4
    assert aux["hypergraph_QE_unique_pairs"].item() <= aux["hypergraph_QE_executed_rows"].item()
    for tau in ("MM", "ME", "EM"):
        key = f"hypergraph_{tau}_unique_pairs"
        assert key not in aux  # preparation is not repeated for each read tile
        assert prepared.interaction_aux[key].item() == prepared.backend_state["hypergraph_accesses"][tau].support.sum().item()
    wide = core.decode_queries(prepared, batch.query_xy, receiver_chunk_size=30, return_routing_maps=True)
    for tau in ("QM", "QE"):
        for suffix in ("unique_pairs", "eligible_pairs", "executed_rows"):
            key = f"hypergraph_{tau}_{suffix}"
            assert aux[key].item() == wide[key].item()


class _TinyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.backend = _typed()
        self.physical = nn.Linear(8, 1)

    def forward(self, encoded, query, features):
        backend = self.core.backend
        prepared = backend.prepare(encoded, encoded.module_tokens)
        context, aux = backend.read(prepared, encoded, query, features)
        return {"pred_field": self.physical(context), "interaction_aux": aux}


def test_shadow_preserves_hard_physical_gradients_and_trains_omitted_source():
    encoded = _encoded()
    model = _TinyWrapper().train()
    model.core.backend.organizer.set_epoch(301)
    model.core.backend.organizer.eval()  # hard deterministic main values
    query, features = torch.rand(2, 5, 2), torch.rand(2, 5, 6)
    model(encoded, query, features)  # materialize native lazy physical kernels
    with torch.no_grad():
        model.core.backend.organizer.geometry_strength["QE"].fill_(8)
        model.core.backend.organizer.geometry_strength["QM"].fill_(2)
    reference = deepcopy(model)
    hard = reference(encoded, query, features)
    hard["pred_field"].square().sum().backward()
    result = hard_value_soft_hypergraph_forward(model, encoded, query, features)
    torch.testing.assert_close(result["pred_field"], hard["pred_field"], rtol=0, atol=0)
    result["pred_field"].square().sum().backward()
    hard_parameters = dict(reference.named_parameters())
    useful = 0.0
    for name, parameter in model.named_parameters():
        if ".organizer." in name:
            useful += 0 if parameter.grad is None else float(parameter.grad.abs().sum())
        else:
            torch.testing.assert_close(parameter.grad, hard_parameters[name].grad, rtol=2e-5, atol=2e-6)
    assert useful > 1e-6
    assert result["campaign_shadow_calls"] == 1
    assert model.core.backend.permission_mode == "hard"


def test_control_identity_intervention_preserves_access():
    encoded = _encoded()
    model = _typed().eval()
    with torch.no_grad():
        model.organizer.geometry_strength["QE"].fill_(2)
        model.organizer.control_heads["QE"][-1].weight.normal_(0, 0.2)
        model.control_score.weight.normal_(0, 0.2)
    prepared = model.prepare(encoded, encoded.module_tokens)
    query = torch.rand(2, 4, 2)
    normal = model._access(prepared["hypergraph_plan"], query, "QE")
    model.plan_intervention = "control_identity"
    identity = model._access(prepared["hypergraph_plan"], query, "QE")
    torch.testing.assert_close(identity.weight, normal.weight)
    torch.testing.assert_close(identity.control, torch.zeros_like(identity.control))


def test_full_access_intervention_preserves_normal_pair_controls():
    encoded = _encoded()
    model = _typed().eval()
    with torch.no_grad():
        model.organizer.geometry_strength["QE"].fill_(5)
        model.organizer.control_heads["QE"][-1].weight.normal_(0, 0.2)
    prepared = model.prepare(encoded, encoded.module_tokens)
    query = torch.rand(2, 4, 2)
    normal = model._access(prepared["hypergraph_plan"], query, "QE")
    assert normal.control.abs().sum() > 0
    assert not normal.support.all()
    model.set_plan_intervention("full_access")
    full = model._access(prepared["hypergraph_plan"], query, "QE")
    assert full.support.all()
    torch.testing.assert_close(full.weight, torch.ones_like(full.weight))
    torch.testing.assert_close(full.control, normal.control)


@pytest.mark.parametrize("mode", ["full_access", "root_union"])
def test_exported_access_interventions_preserve_mm_self_and_receiver_validity(mode):
    encoded = _encoded()
    model = _typed().eval()
    model.set_plan_intervention(mode)
    prepared = model.prepare(encoded, encoded.module_tokens)
    exported = model.export_typed_state(prepared)
    access = exported["receiver_access"](encoded.module_centers, "MM")
    expected = ((encoded.module_present[:, :, None] > .5) & (encoded.module_present[:, None] > .5)
                & ~torch.eye(4, dtype=torch.bool)[None])
    torch.testing.assert_close(access.support, expected)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("architecture", [
    "adaptive_receiver_hypergraph_honf", "overlap_control_hypergraph_honf", "local_overlap_hypergraph_honf",
])
def test_rectangular_executor_preserves_values_first_gradients_and_measures_actual_rows(dimension, architecture):
    encoded = _encoded(dimension)
    encoded = replace(encoded, env_characteristic_lengths=torch.full_like(encoded.env_weights, .03))
    dense = TypedHypergraphField(8, 12, 2, 2, architecture=architecture, spatial_dim=dimension,
                                module_characteristic_length=.03).double().eval()
    encoded = replace(encoded, **{name: value.double() for name, value in vars(encoded).items()
                                  if torch.is_tensor(value) and value.is_floating_point()})
    query = torch.rand(2, 7, dimension, dtype=torch.float64)
    features = torch.rand(2, 7, 6, dtype=torch.float64)
    with torch.no_grad():
        dense.read(dense.prepare(encoded, encoded.module_tokens), encoded, query, features)
        for gain in dense.control_gain.values():
            gain.weight.normal_(0, .2)
        dense.control_score.weight.normal_(0, .2)
    subset = deepcopy(dense)
    subset.set_execution_mode("rectangular_subset", receiver_chunk_size=3)
    results, input_gradients, ledgers = {}, {}, {}
    for name, model in (("dense", dense), ("subset", subset)):
        # A controlled valid sparse typed plan exercises actual omitted rows.
        original_prepare = model.organizer.prepare

        def sparse_prepare(*args, _prepare=original_prepare, **kwargs):
            plan = _prepare(*args, **kwargs)
            memberships = {}
            for mechanism, value in plan.memberships.items():
                mask = torch.ones_like(value)
                mask[..., 1::2] = 0
                memberships[mechanism] = value * mask
            return replace(plan, memberships=memberships)

        model.organizer.prepare = sparse_prepare
        module = encoded.module_tokens.clone().requires_grad_()
        environment = encoded.env_tokens.clone().requires_grad_()
        coordinates = query.clone().requires_grad_()
        record = replace(encoded, module_tokens=module, env_tokens=environment)
        state = model.prepare(record, module)
        value, work = model.read(state, record, coordinates, features)
        value.square().sum().backward()
        results[name] = value
        input_gradients[name] = (module.grad, environment.grad, coordinates.grad)
        ledgers[name] = {**state["hypergraph_ledger"], **work}
    torch.testing.assert_close(results["subset"], results["dense"], rtol=1e-9, atol=1e-10)
    for actual, expected in zip(input_gradients["subset"], input_gradients["dense"]):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
    for (_, actual), (_, expected) in zip(subset.named_parameters(), dense.named_parameters()):
        if actual.grad is not None or expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-8, atol=1e-9)
    saved = 0
    for mechanism in ("MM", "ME", "EM", "QM", "QE"):
        work = ledgers["subset"]
        rows = work[f"hypergraph_{mechanism}_executed_rows"]
        allocated = work[f"hypergraph_{mechanism}_allocated_rows"]
        assert rows <= allocated
        assert work[f"hypergraph_{mechanism}_padded_rows"] >= 0
        assert work[f"hypergraph_{mechanism}_skipped_eligible_pairs"] >= 0
        assert work[f"hypergraph_{mechanism}_unique_pairs"] <= rows
        saved += int(allocated - rows)
    assert saved > 0
    assert ledgers["subset"]["hypergraph_QE_attention_cells"] <= ledgers["dense"]["hypergraph_QE_attention_cells"]


def test_collective_controls_receive_task_gradients_after_gain_learns():
    encoded = _encoded()
    model = _TinyWrapper().train()
    model.core.backend.organizer.eval()
    query, features = torch.rand(2, 5, 2), torch.rand(2, 5, 6)
    model(encoded, query, features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    gradients = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        result = hard_value_soft_hypergraph_forward(model, encoded, query, features)
        (result["pred_field"] - query[..., :1]).square().mean().backward()
        control_head = model.core.backend.organizer.control_heads["QE"][-1]
        gradients.append(0 if control_head.weight.grad is None else float(control_head.weight.grad.abs().sum()))
        optimizer.step()
    assert max(gradients[1:]) > 1e-8


def test_export_keeps_prepared_intervention_when_live_backend_mode_changes():
    encoded = _encoded()
    model = _typed().eval()
    model.set_plan_intervention("control_identity")
    prepared = model.prepare(encoded, encoded.module_tokens)
    exported = model.export_typed_state(prepared)
    query = torch.rand(2, 4, 2)
    features = torch.rand(2, 4, 6)
    before = exported["receiver_access"](query, "QE")
    prediction_before, work_before = model.read(prepared, encoded, query, features)
    model.set_plan_intervention("normal")
    model.set_execution_mode("rectangular_subset", receiver_chunk_size=2)
    after = exported["receiver_access"](query, "QE")
    delayed_export = model.export_typed_state(prepared)
    delayed_access = delayed_export["receiver_access"](query, "QE")
    prediction_after, work_after = model.read(prepared, encoded, query, features)
    torch.testing.assert_close(before.weight, after.weight)
    torch.testing.assert_close(after.control, torch.zeros_like(after.control))
    torch.testing.assert_close(delayed_access.control, after.control)
    torch.testing.assert_close(prediction_after, prediction_before, rtol=0, atol=0)
    assert delayed_export["executor"] == "dense_masked_reference"
    for mechanism in ("QM", "QE"):
        key = f"hypergraph_{mechanism}_executed_rows"
        torch.testing.assert_close(work_after[key], work_before[key])
    live = model._access(prepared["hypergraph_plan"], query, "QE")
    assert not torch.allclose(live.control, after.control)
