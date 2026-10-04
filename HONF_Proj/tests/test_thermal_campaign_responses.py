"""Finite-response magnitudes and failures must survive reporting."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "tools/thermal_campaign_responses.py"
_SPEC = importlib.util.spec_from_file_location("thermal_campaign_responses", _PATH)
responses = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(responses)


@pytest.mark.parametrize("supported", [True, False])
def test_stencil_work_retains_every_absolute_state_without_repeating_or_changing_predictions(supported):
    from channelthermal.interaction_evidence import DesignState, EvidenceSplit, ModuleState, OperatingContext
    from channelthermal.response_control.algebra import predict_stencil
    from channelthermal.response_control.contracts import AbsolutePrediction, RoleQuery

    from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
    from honf_forward_core.interface_fields.types import EncodedInterfaceCase

    torch.manual_seed(41)
    backend = DensePairwiseField(8, 12, 2, 2).eval()
    query = RoleQuery("fluid_fields", torch.rand(2, 2), ("temperature",), ("benchmark_T",), None, "eulerian")

    def record(heat):
        return SimpleNamespace(design=DesignState(
            anchor_id="synthetic-test", physical_family_id="synthetic-test", split=EvidenceSplit.TRAIN,
            modules=(ModuleState("m0", (.2, .3), heat), ModuleState("m1", (.8, .7), 1.))),
            context=OperatingContext({"re": 50.}))

    stencil = SimpleNamespace(baseline=record(1.), variants={f"heat_{index}": record(1. + index / 10) for index in range(1, 11)})
    calls = []

    def operator(design, context, queries):
        calls.append(design.module_heating.clone())
        assert set(context) == {"re"} and not hasattr(design, "output")
        encoded = EncodedInterfaceCase(
            module_tokens=design.module_heating[None, :, None].expand(1, 2, 8), env_tokens=torch.ones(1, 3, 8),
            global_token=torch.ones(1, 8), module_centers=design.module_positions[None],
            env_coords=torch.tensor([[[.1, .1], [.5, .5], [.9, .9]]]), module_present=design.module_present[None],
            module_features=torch.ones(1, 2, 3), env_features=None,
            env_weights=torch.ones(1, 3), coordinate_scale=torch.ones(1, 1, 2))
        prepared = backend.prepare(encoded, encoded.module_tokens)
        values, _ = backend.read(prepared, encoded, queries["fluid_fields"].query_features[None], torch.ones(1, 2, 6))
        return AbsolutePrediction({"fluid_fields": values[0, :, :1]})

    reference = predict_stencil(operator, stencil, role_queries={"fluid_fields": query})
    calls.clear()
    prediction, work = responses.predict_stencil_with_fine_work(
        backend if supported else SimpleNamespace(), operator, stencil, role_queries={"fluid_fields": query})
    assert len(calls) == 11
    assert list(work["states"]) == ["baseline", *stencil.variants]
    for label in prediction.values:
        torch.testing.assert_close(prediction.values[label].role_values["fluid_fields"],
                                   reference.values[label].role_values["fluid_fields"], rtol=0, atol=0)
    if supported:
        per_state = {route: {"padded_input_rows": rows, "calls": 1}
                     for route, rows in {"MM": 4, "ME": 6, "EM": 6, "QM": 4, "QE": 6}.items()}
        assert all(state["routes"] == per_state for state in work["states"].values())
        assert work["routes"] == {route: {"padded_input_rows": rows["padded_input_rows"] * 11, "calls": 11}
                                  for route, rows in per_state.items()}
    else:
        assert not work["measured"] and work["routes"] is None and work["reason"]
        assert all(not state["measured"] and state["routes"] is None for state in work["states"].values())
    assert not backend.mm_message._forward_hooks

    def failed(*_args):
        raise RuntimeError("absolute prediction failed")

    with pytest.raises(RuntimeError, match="absolute prediction failed"):
        responses.predict_stencil_with_fine_work(backend, failed, stencil, role_queries={"fluid_fields": query})
    assert not backend.mm_message._forward_hooks


def test_weighted_response_and_zero_control_retain_physical_units_without_unresolved_ratio():
    target = np.asarray([[1., 0.], [3., 0.]])
    prediction = np.asarray([[2., 2.], [1., 2.]])
    rows = responses.response_channel_metrics(prediction, target, np.ones(2, bool),
        np.asarray([1., 3.]), ["temperature", "p"], ["benchmark_T", "benchmark_p"])
    assert rows["temperature"]["rmse"] == pytest.approx(np.sqrt(3.25))
    assert rows["temperature"]["reference_rms"] == pytest.approx(np.sqrt(7.))
    assert rows["temperature"]["zero_change_rmse"] == rows["temperature"]["reference_rms"]
    assert rows["p"]["rmse"] == 2 and rows["p"]["reference_rms"] == 0
    assert rows["p"]["relative_accuracy"] is None
    assert rows["temperature"]["unit"] == "benchmark_T"


def test_nonfinite_and_empty_responses_are_reported_and_bad_quadrature_rejected():
    target = np.zeros((2, 1))
    rows = responses.response_channel_metrics(np.asarray([[np.nan], [0.]]), target,
        np.ones_like(target, bool), np.ones(2), ["T"], ["K"])
    assert not rows["T"]["finite"] and rows["T"]["rmse"] is None
    rows = responses.response_channel_metrics(target, target, np.zeros_like(target, bool),
        np.ones(2), ["T"], ["K"])
    assert rows["T"]["count"] == 0 and rows["T"]["rmse"] is None
    with pytest.raises(ValueError, match="quadrature"):
        responses.response_channel_metrics(target, target, np.ones_like(target, bool),
            np.asarray([1., -1.]), ["T"], ["K"])


def test_input_selected_transfers_preserve_totals_range_and_padding():
    positions = np.asarray([[1., 0.], [2., 1.], [3., 2.], [0., 0.]])
    heat = np.asarray([1., 1.3, 1.6, 0.])
    variants = responses.fixed_total_transfers(positions, heat, [1, 1, 1, 0], lower=.5, upper=2.)
    assert len(variants) == 8
    assert {tuple(row["donors"]) for row in variants} == {(0, 2), (0, 1)}
    for variant in variants:
        assert variant["heat"].sum() == pytest.approx(heat.sum())
        assert .5 <= variant["heat"][:3].min() <= variant["heat"][:3].max() <= 2.
        assert variant["heat"][3] == 0
    assert responses.fixed_total_transfers(positions[:1], heat[:1], [1], lower=.5, upper=2.) == []


def test_subset_response_audit_reads_selected_inputs_only_and_separates_duplicates(tmp_path):
    import h5py

    path = tmp_path / "cases.h5"
    with h5py.File(path, "w") as handle:
        cases = handle.create_group("cases")
        for case_id, heat in (("a", [1., 2.]), ("b", [1.2, 1.8]), ("duplicate", [1., 2.])):
            group = cases.create_group(case_id)
            group["module_present"] = [1, 1]
            group["module_centers"] = [[1., 1.], [2., 2.]]
            group["heat_powers"] = heat
            group["case_config_json"] = json.dumps({"domain": {"lx": 3.}, "flow": {"re": 90.},
                "thermal": {"fluid_k": 1., "heat_power_max": 2.}}).encode()
            group.attrs["converged"] = True
        # Invalid excluded inputs must never be inspected.
        cases.create_group("excluded")
    manifest = {"partitions": {"train": {"case_ids": ["a"]}, "test": {"case_ids": ["b"]}},
                "manifest_sha256": "test-manifest"}
    result = responses.audit_subset_response_pairs(path, manifest)
    assert result["held_response_family_count"] == 1
    assert result["excluded_cases_read"] == 0
    assert result["matched_nonzero_response_families"][0]["different_heat_pairs"][0]["partitions"] == ["train", "test"]
    manifest["partitions"]["test"]["case_ids"] = ["duplicate"]
    duplicate = responses.audit_subset_response_pairs(path, manifest)
    assert duplicate["held_response_family_count"] == 0
    assert duplicate["identical_heat_layout_context_duplicates"]
