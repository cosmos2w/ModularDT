"""Focused Round8 post-fit loader, telemetry, and lifecycle regressions."""

from __future__ import annotations

import gc
import json
import time
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from honf_forward_core.content_hypergraph.contracts import QueryBatch, SceneBatch
from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore
from honf_forward_core.native_retrieval.budgeted_reader import BudgetedNativeReader
from tests.test_content_hypergraph_v2_core import _core, _query, _scene
from tests.test_native_budgeted_reader import _fixture


def _thermal_stats():
    return {
        "field_mean_by_channel": np.zeros(5, dtype=np.float32),
        "field_std_by_channel": np.ones(5, dtype=np.float32),
        "interface_targets_mean": np.zeros(2, dtype=np.float32),
        "interface_targets_std": np.ones(2, dtype=np.float32),
        "internal_temperature_std": np.ones(1, dtype=np.float32),
    }


def _thermal_budgeted_config():
    return {
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
        "init_seed": 71,
    }


def _thermal_native_inputs():
    torch.manual_seed(710)
    batch, modules, environments, queries = 1, 3, 4, 5
    scene = SceneBatch(
        module_features=torch.randn(batch, modules, 8),
        environment_features=torch.randn(batch, environments, 8),
        module_coordinates=torch.rand(batch, modules, 2),
        environment_coordinates=torch.rand(batch, environments, 2),
        module_mask=torch.ones(batch, modules, dtype=torch.bool),
        environment_mask=torch.ones(batch, environments, dtype=torch.bool),
        module_measure=torch.rand(batch, modules) + 0.2,
        environment_measure=torch.rand(batch, environments) + 0.2,
        module_ids=torch.tensor([[11, 17, 23]]),
        environment_ids=torch.tensor([[31, 37, 41, 43]]),
        local_radius=torch.full((batch, modules), 0.4),
        operating_features=torch.randn(batch, 14),
        coordinate_origin=torch.zeros(batch, 2),
        coordinate_scale=torch.ones(batch, 2),
        module_measure_reference=torch.ones(batch),
        environment_measure_reference=torch.ones(batch),
        scene_key=("thermal-postfit",),
        normalizer_key="postfit-normalizer",
        coordinate_frame_key="thermal-native-v2",
        module_characteristic_lengths=torch.full((batch, modules, 2), 0.2),
        environment_characteristic_lengths=torch.full((batch, environments, 2), 0.1),
    )
    query = QueryBatch(
        coordinates=torch.rand(batch, queries, 2),
        query_mask=torch.ones(batch, queries, dtype=torch.bool),
        query_ids=torch.arange(queries)[None],
        scene_key=("thermal-postfit",),
        coordinate_frame_key="thermal-native-v2",
        receiver_key="postfit-query-panel",
    )
    return scene, query


def test_nested_thermal_adapter_load_restores_round8_clocks_without_rewriting_dual_state():
    base = ContentHypergraphCoreV2.from_config(_thermal_budgeted_config())
    core = BudgetedNativeCore.from_a2d(base, native_backend="flat_selected_reference", lambda0=0.2)
    from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2

    adapter = ThermalContentHypergraphAdapterV2(_thermal_stats(), core=core, nx=8, ny=6)
    core.native_training_epoch.fill_(1733)
    core.budgeted_macro_step.fill_(23)
    core._native_epoch = 500
    core._budgeted_macro_clock = 0
    dual = core.budgeted_router.dual
    dual.macro_step.fill_(19)
    dual.window_residual_sum.fill_(0.125)
    dual.window_step_count.fill_(3)
    dual.dual_update_count.fill_(7)
    state = {key: value.clone() if isinstance(value, torch.Tensor) else value
             for key, value in adapter.state_dict().items()}

    nested_core = BudgetedNativeCore.from_config(core.get_config())
    restored = ThermalContentHypergraphAdapterV2(_thermal_stats(), core=nested_core, nx=8, ny=6)
    restored.load_state_dict(state, strict=True)
    assert nested_core.native_epoch == 1733
    assert nested_core._budgeted_macro_clock == 23
    assert int(dual.macro_step) == 19
    assert int(nested_core.budgeted_router.dual.macro_step) == 19
    assert int(nested_core.budgeted_router.dual.window_step_count) == 3
    assert float(nested_core.budgeted_router.dual.window_residual_sum) == pytest.approx(0.125)
    assert int(nested_core.budgeted_router.dual.dual_update_count) == 7
    nested_core.train()
    assert nested_core._router_mode()[2] is True  # next global macro is 24

    direct = BudgetedNativeCore.from_config(core.get_config())
    direct.load_state_dict(core.state_dict(), strict=True)
    direct.eval()
    nested_core.eval()
    scene, query = _thermal_native_inputs()
    with torch.no_grad():
        direct_output = direct.decode(direct.prepare(scene), query, torch.ones(1, 3))
        nested_output = nested_core.decode(nested_core.prepare(scene), query, torch.ones(1, 3))
    torch.testing.assert_close(nested_output.nonlinear_field, direct_output.nonlinear_field, rtol=0, atol=0)
    torch.testing.assert_close(nested_output.thermal_temperature, direct_output.thermal_temperature, rtol=0, atol=0)

    boundary = BudgetedNativeCore.from_config(core.get_config())
    boundary.set_native_training_epoch(500)
    assert boundary.native_epoch == 500
    assert not boundary._uses_native_retrieval()


def test_budgeted_store_hook_does_not_retain_prepared_store_cycle():
    model = BudgetedNativeCore.from_a2d(_core("A2-D").double(), membership_bias=0.0).train()
    model.budgeted_begin_update(701, 0)
    prepared = model.prepare(_scene(1))
    query = _query(1)
    store = prepared.native_token_store
    store_ref = weakref.ref(store)
    environment_ref = weakref.ref(prepared.environment_states)
    output = model.decode(prepared, query)
    output.nonlinear_field.square().sum().backward()
    assert store.backward_consumed
    with pytest.raises(ValueError, match="consum"):
        model.decode(prepared, query)

    del store, output, prepared
    model._budgeted_entries.clear()
    model._budgeted_observations.clear()
    model.zero_grad(set_to_none=True)
    gc.collect()
    assert store_ref() is None
    assert environment_ref() is None


def test_dense_and_flat_protection_exception_telemetry_is_measured_or_explicitly_unavailable():
    reader, organization, query, query_states, _ = _fixture()
    batch, queries, modules = 2, 5, 4
    membership = torch.zeros((batch, modules, modules), dtype=torch.bool)
    membership[:, torch.arange(modules), torch.arange(modules)] = True
    membership[:, 0, 1] = True
    dispatch = torch.arange(queries)[None].expand(batch, -1).remainder(modules)
    chosen = membership.gather(1, dispatch[..., None].expand(batch, queries, modules))
    protected = torch.zeros((batch, queries, modules), dtype=torch.bool)
    protected[0, 0, 3] = True
    protected[1, 4, 2] = True
    support = chosen | protected
    graph = SimpleNamespace(
        membership_hard=membership,
        seed_valid=torch.ones((batch, modules), dtype=torch.bool),
    )
    dispatch_plan = SimpleNamespace(dispatch_seed=dispatch)
    expected = int((support & protected & ~chosen).sum())

    dense = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states, support.to(query_states.dtype),
        backend="dense_hard_reference", graph=graph, dispatch_plan=dispatch_plan,
        protected_mask=protected,
    )
    flat = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states, support,
        backend="flat_selected_reference", graph=graph, dispatch_plan=dispatch_plan,
        protected_mask=protected,
    )
    for result in (dense, flat):
        assert result.counts["native_protection_exception_rows"] == expected
        assert expected == 2
        assert result.counts["native_protection_exception_rows_available"] == 1
        assert result.counts["native_request_syncs"] == 0  # CPU fixture; CUDA transfer charges are tested by endpoint probes.

    unavailable = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states, support,
        backend="flat_selected_reference",
    )
    assert unavailable.counts["native_protection_exception_rows"] == 0
    assert unavailable.counts["native_protection_exception_rows_available"] == 0


def test_core_gate_override_marks_route_relative_protection_counter_unavailable():
    model = BudgetedNativeCore.from_a2d(_core("A2-D").double(), membership_bias=0.0).eval()
    model.set_native_training_epoch(2500)
    model._budgeted_gate_override = lambda gate, *_args: gate
    with torch.no_grad():
        output = model.decode(model.prepare(_scene(1)), _query(1))
    assert output.work_counts["source_read_native_protection_exception_rows"] >= 0
    assert output.work_counts["source_read_native_protection_exception_rows_available"] == 0


def test_gpu_usage_append_uses_segment_fit_summary_only_at_matching_stop(tmp_path):
    from tools.native_budgeted_train import _append_gpu_usage

    output = tmp_path / "run"
    metrics = output / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "fit_summary.json").write_text(json.dumps({
        "completed_epoch": 600,
        "start_epoch": 501,
        "optimizer_updates": 400,
        "case_visits": 15000,
    }), encoding="utf-8")
    root = tmp_path / "root"
    _append_gpu_usage(root, {"dataset": "thermal", "treatment": "D-control"},
                      output, time.perf_counter(), "completed", {"completed_epoch": 600})
    row = json.loads((root / "receipts" / "round8_gpu_usage.jsonl").read_text(encoding="utf-8"))
    assert row["optimizer_updates"] == 400
    assert row["case_visits"] == 15000
    assert row["segment_start_epoch"] == 501
    assert row["usage_counts_source"] == "metrics/fit_summary.json"

    _append_gpu_usage(root, {"dataset": "thermal", "treatment": "D-control"},
                      output, time.perf_counter(), "failed", {"completed_epoch": 599})
    rows = [json.loads(line) for line in (root / "receipts" / "round8_gpu_usage.jsonl").read_text().splitlines()]
    assert rows[1]["optimizer_updates"] is None
    assert rows[1]["case_visits"] is None
    assert rows[1]["usage_counts_source"] is None
