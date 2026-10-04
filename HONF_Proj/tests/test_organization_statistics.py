"""Scalar statistics preserve native reads and distinguish eligibility/work/actions."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from honf_forward_core.evaluation.organization_statistics import TypedOrganizationStatistics
from honf_forward_core.evaluation.typed_work_evidence import TypedWorkEvidenceRecorder
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def case():
    generator = torch.Generator().manual_seed(593)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(1, 3, 8, generator=generator),
        env_tokens=torch.randn(1, 5, 8, generator=generator),
        global_token=torch.randn(1, 8, generator=generator),
        module_centers=torch.rand(1, 3, 2, generator=generator),
        env_coords=torch.rand(1, 5, 2, generator=generator),
        module_present=torch.tensor([[1., 1., 0.]]),
        module_features=torch.randn(1, 3, 4, generator=generator), env_features=None,
        env_weights=torch.tensor([[1., 2., .5, 0., 1.]]),
        coordinate_scale=torch.ones(1, 1, 2))


def backend():
    return TypedHypergraphField(8, 12, 2, 2, architecture="faithful_receiver_hypergraph_honf",
                                spatial_dim=2, module_characteristic_length=.1,
                                options={"organizer_dim": 16}).eval()


def test_actual_numerical_stream_sufficient_statistics_with_no_extra_reads_or_mutation():
    encoded = case()
    model = backend()
    query, features = torch.rand(1, 6, 2, requires_grad=True), torch.rand(1, 6, 6)
    state = model.prepare(encoded, encoded.module_tokens)
    expected, _ = model.read(state, encoded, query, features)
    calls = []
    handles = [getattr(model, name).register_forward_pre_hook(lambda module, args: calls.append(module))
               for name in ("mm_message", "me_message", "em_message", "query_module_message", "env_geometry_bias")]
    initial = deepcopy(model.state_dict())
    with TypedOrganizationStatistics(model) as recorder, TypedWorkEvidenceRecorder(model) as detailed:
        state = model.prepare(encoded, encoded.module_tokens)
        actual, _ = model.read(state, encoded, query, features)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == 5
    differentiable = tuple(model.parameters()) + (query,)
    actual_grad = torch.autograd.grad(actual.square().sum(), differentiable, allow_unused=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), differentiable, allow_unused=True)
    for got, reference in zip(actual_grad, expected_grad):
        if reference is None:
            assert got is None
        else:
            torch.testing.assert_close(got, reference, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    summary = recorder.summary()
    json.dumps(summary, allow_nan=False)
    assert set(summary["native_routes"]) == {"P0/" + tau for tau in ("MM", "ME", "EM", "QM", "QE")}
    for key, row in summary["native_routes"].items():
        assert row["access_calls"] == 1
        assert row["positive_measure_support_fraction"] == 1
        assert row["positive_measure_value_pairs"] == row["positive_measure_eligible_pairs"]
        assert row["distinct_receiver_count_exact"]
        assert row["receiver_group_participation_mass"] == pytest.approx(row["receiver_measure_sum"])
        assert row["repeated_far_group_path_access_calls"] == row["access_calls"]
        assert row["repeated_far_group_paths_removed"] >= 0
        source_count = 3 if key.rsplit("/", 1)[-1] in {"MM", "EM", "QM"} else 5
        assert row["projected_numerical_pair_rows"] == row["receiver_rows"] * source_count
    assert summary["native_routes"]["P0/QE"]["eligible_pairs"] == 24
    assert summary["native_routes"]["P0/QE"]["native_support_fraction"] == 1
    assert summary["native_routes"]["P0/QE"]["executed_work"]["hypergraph_QE_executed_rows"] == 30
    assert summary["native_routes"]["P0/MM"]["eligible_pairs"] == 2
    assert summary["native_routes"]["P0/MM"]["receiver_measure_sum"] == 2
    row = summary["native_routes"]["P0/QE"]
    expected_hhi = float((encoded.env_weights / encoded.env_weights.sum()).square().sum())
    assert row["receiver_weighted_source_hhi_sum"] / row["source_bearing_receiver_measure_sum"] == pytest.approx(expected_hhi)
    assert row["source_participation_equivalent"] == pytest.approx(1. / expected_hhi)
    participation = np.asarray(row["per_group_receiver_participation_mass"])
    assert row["receiver_group_participation_equivalent"] == pytest.approx(participation.sum() ** 2 / np.square(participation).sum())
    # Detailed recording still observes exactly the same native calls.
    assert len([key for key in detailed.arrays if key.endswith("/support")]) == 5
    assert all(name not in model.__dict__ for name in ("prepare", "_access", "_numerical_access", "_ledger"))


def test_bounded_complete_action_signatures_report_lower_bounds_without_digest_collisions():
    recorder = TypedOrganizationStatistics(backend(), signature_tolerance=.01, signature_capacity=2)
    recorder._distinct("P0/QE", np.asarray([[1., 0.], [1.001, 0.], [2., 0.]]))
    assert len(recorder._signatures["P0/QE"]) == 2
    assert "P0/QE" not in recorder._signature_saturated
    recorder._distinct("P0/QE", np.asarray([[3., 0.]]))
    assert "P0/QE" in recorder._signature_saturated
    assert len(recorder._signatures["P0/QE"]) == 2


def test_statistics_restore_methods_on_failure_and_reject_training():
    model = backend()
    with pytest.raises(RuntimeError), TypedOrganizationStatistics(model):
        raise RuntimeError("interrupted diagnostic")
    assert all(name not in model.__dict__ for name in ("prepare", "_access", "_numerical_access", "_ledger"))
    with pytest.raises(ValueError, match="frozen-evaluation"):
        TypedOrganizationStatistics(model.train())
    with pytest.raises(ValueError, match="tolerance"):
        TypedOrganizationStatistics(model.eval(), signature_tolerance=0)


def test_one_group_zero_control_rebind_preserves_physical_sources_access_and_affine_bias():
    model, encoded = backend(), case()
    query, features = torch.rand(1, 6, 2), torch.rand(1, 6, 6)
    with torch.no_grad():
        state = model.prepare(encoded, encoded.module_tokens)
        plan = state["hypergraph_plan"]
        controls = dict(plan.controls)
        controls["QE"] = torch.randn_like(controls["QE"])
        plan = replace(plan, controls=controls)
        state["hypergraph_plan"] = plan
        model.read(state, encoded, query, features)
        original = model._numerical_access(plan, query, "QE", state["hypergraph_actions"])
        participation = original.edge_access.sum((0, 1))
        group = int(participation.argmax())
        controls = dict(plan.controls)
        controls["QE"] = controls["QE"].clone()
        controls["QE"][0, group] = 0
        ablated_plan = replace(plan, controls=controls)
        ablated_state = dict(state, hypergraph_plan=ablated_plan)
        model.read(ablated_state, encoded, query, features)
        assert ablated_state["hypergraph_action_plan"] is ablated_plan
        for key in ("module_tokens", "env_tokens", "projected_key", "projected_value"):
            assert ablated_state[key] is state[key]
        changed = model._numerical_access(ablated_plan, query, "QE", ablated_state["hypergraph_actions"])
        for name in ("density", "weight", "support", "edge_access"):
            torch.testing.assert_close(getattr(changed, name), getattr(original, name), rtol=0, atol=0)
        weight = torch.cat((model.control_score.weight, model.control_gain["QE"].weight))
        linear_control = torch.nn.functional.linear(plan.controls["QE"][0, group], weight)
        ratio = original.edge_access[:, :, group, None] * plan.memberships["QE"][:, group, None]
        ratio = ratio / torch.where(original.density > 0, original.density, torch.ones_like(original.density))
        torch.testing.assert_close(original.projected - changed.projected, ratio[..., None] * linear_control,
                                   rtol=2e-5, atol=2e-6)


def test_streaming_query_chunks_preserve_measure_weighted_sufficient_statistics():
    model, encoded = backend(), case()
    query, features = torch.rand(1, 11, 2), torch.rand(1, 11, 6)
    with torch.no_grad():
        with TypedOrganizationStatistics(model, signature_tolerance=1e-3) as whole:
            state = model.prepare(encoded, encoded.module_tokens)
            model.read(state, encoded, query, features)
        with TypedOrganizationStatistics(model, signature_tolerance=1e-3) as chunks:
            state = model.prepare(encoded, encoded.module_tokens)
            for start in range(0, 11, 4):
                model.read(state, encoded, query[:, start:start + 4], features[:, start:start + 4])
    first, second = whole.summary()["native_routes"], chunks.summary()["native_routes"]
    for tau in ("QM", "QE"):
        before, after = first[f"P0/{tau}"], second[f"P0/{tau}"]
        assert before["access_calls"] == 1
        assert after["access_calls"] == 3
        for name in ("receiver_rows", "eligible_pairs", "positive_measure_value_pairs", "receiver_measure_sum",
                     "receiver_weighted_source_hhi_sum", "source_bearing_receiver_measure_sum",
                     "eligible_pair_measure_sum", "supported_projected_square_sum",
                     "supported_projected_scalar_measure_sum", "supported_gain_scalar_measure_sum",
                     "repeated_far_group_paths_removed", "projected_numerical_pair_rows",
                     "source_participation_equivalent", "receiver_group_participation_equivalent"):
            assert after[name] == pytest.approx(before[name], rel=2e-6, abs=1e-6)
        np.testing.assert_allclose(after["per_group_receiver_participation_mass"],
                                   before["per_group_receiver_participation_mass"], rtol=2e-6, atol=1e-6)
        assert after["distinct_quantized_receiver_actions"] == before["distinct_quantized_receiver_actions"]
