from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from channelthermal.response_control.latency import MECHANISMS, query_role_counts
from channelthermal.response_control.maturation import THERMAL_ACTION_PATHS, available_frontier_for_paths

from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CaseLocalReceiverTree,
    DirectPairAccess,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.budgeted_frontier import enumerate_frontier_cuts

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "scripts"))

import benchmark_selected_weight_latency as benchmark


def test_latency_sparse_builder_uses_trained_four_packet_action_on_deep_tree(monkeypatch) -> None:
    coordinates = torch.arange(8, dtype=torch.float32)[:, None]
    universe = ReceiverAnchorUniverse(coordinates, torch.ones(8), torch.zeros(8, dtype=torch.long), torch.tensor([8.0]))
    tree = CaseLocalReceiverTree.build(universe, max_nodes=15, min_leaf_anchors=1)
    monkeypatch.setattr(
        benchmark,
        "ThermalCoverPlanBuilder",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    builder = benchmark._sparse_builder("G", SimpleNamespace(core=object()), object(), "MM")
    selected = builder.frontier_selector(0, None, torch.empty(0), tree)
    expected, _ = available_frontier_for_paths(tree, THERMAL_ACTION_PATHS["four_packet"])

    assert selected == expected
    assert len(selected) == 4
    assert len(enumerate_frontier_cuts(tree, max_depth=3)[-1]) == 8
    assert builder.hard is True


def test_latency_packet_summary_preserves_p_direct_pair_nonapplicability(monkeypatch) -> None:
    coordinates = torch.arange(8, dtype=torch.float32)[:, None]
    universe = ReceiverAnchorUniverse(
        coordinates, torch.ones(8), torch.zeros(8, dtype=torch.long), torch.tensor([8.0])
    )
    tree = CaseLocalReceiverTree.build(universe, max_nodes=15, min_leaf_anchors=1)
    module_present = torch.ones(3)
    direct_plan = MechanismPlan(
        tree=tree,
        split_gates=torch.zeros(len(tree.nodes)),
        module_present=module_present,
        environment_count=2,
        permissions={"QE": torch.ones(len(tree.nodes), 2)},
        direct_access={
            "MM": DirectPairAccess(coordinates, torch.ones(8, 3)),
        },
    )
    with pytest.raises(ValueError, match="cannot be represented as node memberships"):
        direct_plan.permission_matrix("MM")

    def reject_grouped_reconstruction(*_args, **_kwargs):
        raise AssertionError("P direct-pair support must not be rebuilt as node memberships")

    monkeypatch.setattr(benchmark.selected_controls, "_packet_support_summary", reject_grouped_reconstruction)
    record = {
        "frontier": [0, 1, 2, 3],
        "raw_frontier_k": 4,
        "nonredundant_k": None,
        "nonredundant_k_status": "not_applicable_to_direct_pair_receiver_axis",
    }
    result = benchmark._latency_packet_support_summary(
        SimpleNamespace(mode="P"), {"baseline": {"plans": (direct_plan,)}}, record
    )

    assert result["raw_packet_cut_size"] == 4
    assert result["nonredundant_packet_count"] is None
    assert result["nonredundant_packet_count_status"] == "not_applicable_to_direct_pair_receiver_axis"


@pytest.mark.parametrize("large_modules", [9, 11])
def test_large_m_panel_requires_exactly_ten_active_modules(large_modules: int) -> None:
    with pytest.raises(ValueError, match="exactly 10 active modules"):
        benchmark._require_large_m_panel(4, large_modules)

    benchmark._require_large_m_panel(4, 10)


def _role(name: str, features: np.ndarray, *, ids: tuple[str, ...] | None = None):
    count = len(features)
    return SimpleNamespace(
        role=name,
        query_features=np.asarray(features, dtype=np.float64),
        values=np.zeros((count, 1), dtype=np.float64),
        channel_names=("value",),
        channel_units=("unit",),
        valid_mask=np.ones((count, 1), dtype=bool),
        quadrature_weights=np.ones(count, dtype=np.float64),
        query_ids=tuple(f"{name}:{index}" for index in range(count)),
        receiver_module_ids=ids,
        coordinate_kind={
            "fluid_fields": "eulerian",
            "interface": "interface_material_angle",
            "solid_temperature": "solid_material_normalized_xy",
        }[name],
    )


def _record(fluid_count: int = 100):
    modules = (
        SimpleNamespace(
            module_id="module-0",
            position_xy=(0.25, 0.75),
            heating=1.5,
            active=True,
        ),
    )
    fluid_xy = np.stack((np.arange(fluid_count), np.zeros(fluid_count)), axis=1)
    roles = {
        "fluid_fields": _role("fluid_fields", fluid_xy),
        "interface": _role("interface", np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]), ids=("module-0",) * 2),
        "solid_temperature": _role("solid_temperature", np.asarray([[0.0, 0.0], [0.5, 0.5]]), ids=("module-0",) * 2),
    }
    return SimpleNamespace(
        design=SimpleNamespace(physical_family_id="family-a", modules=modules),
        context=SimpleNamespace(values={"u_in": 2.0, "re": 80.0}),
        output=SimpleNamespace(roles=roles),
    )


def test_fluid_rows_use_valid_positive_weight_unique_saved_coordinates() -> None:
    record = _record()
    fluid = record.output.roles["fluid_fields"]
    fluid.query_features[1] = fluid.query_features[0]
    fluid.valid_mask[2] = False
    fluid.quadrature_weights[3] = 0.0

    panel_candidates = benchmark._role_candidates(record, seed=17)
    candidates = panel_candidates["fluid_fields"]

    assert panel_candidates["eligible_rows_by_role"]["fluid_fields"] == 98
    assert 1 not in candidates
    assert 2 not in candidates
    assert 3 not in candidates
    selected = np.random.default_rng(17).permutation(candidates)[:20]
    assert len({tuple(fluid.query_features[index, :2]) for index in selected}) == 20


def test_panel_digest_binds_model_inputs_and_eligibility_but_excludes_targets() -> None:
    record = _record()
    candidates = benchmark._role_candidates(record, seed=17)
    rows = benchmark._rows_by_role(candidates, 64)

    first = benchmark._input_digest(record, rows)
    record.output.roles["fluid_fields"].values[rows["fluid_fields"][0], 0] = 1.0e9
    assert benchmark._input_digest(record, rows) == first

    record.output.roles["fluid_fields"].query_features[rows["fluid_fields"][0], 0] += 0.25
    assert benchmark._input_digest(record, rows) != first


def test_scenario_digest_uses_nested_query_prefixes() -> None:
    record = _record()
    candidates = benchmark._role_candidates(record, seed=31)
    panel = {"record": record, "candidates": candidates, "active_m": 1}

    short = benchmark._scenario(panel, "standard", 64)
    long = benchmark._scenario(panel, "standard", 96)

    assert short["query_count"] == 64
    assert long["query_count"] == 96
    assert short["input_query_panel_digest"] != long["input_query_panel_digest"]
    assert sum(short["role_query_counts"].values()) == 64
    assert sum(long["role_query_counts"].values()) == 96


def test_mixed_native_role_panel_is_deterministic_valid_and_unique() -> None:
    record = _record()
    candidates = benchmark._role_candidates(record, seed=41)
    second = benchmark._role_candidates(record, seed=41)
    selected = benchmark._rows_by_role(candidates, 96)

    assert candidates["fluid_fields"] == second["fluid_fields"]
    assert candidates["interface"] == second["interface"]
    assert candidates["solid_temperature"] == second["solid_temperature"]
    assert sum(map(len, selected.values())) == 96
    for name, rows in selected.items():
        role = record.output.roles[name]
        assert len(rows) == len(set(rows))
        assert all(np.all(role.valid_mask[row]) for row in rows)
        assert all(role.quadrature_weights[row] > 0.0 for row in rows)


def test_q8192_balances_native_material_rows_per_active_module() -> None:
    candidates = {
        "active_slots": (0, 1, 2),
        "fluid_fields": tuple(range(7976)),
        "interface": {slot: tuple(range(1000 * slot, 1000 * slot + 64)) for slot in range(3)},
        "solid_temperature": {slot: tuple(range(10000 * slot, 10000 * slot + 3000)) for slot in range(3)},
        "query_cap": 8192,
    }

    assert query_role_counts(candidates, 8192) == (7976, 192, 24)
    selected = benchmark._rows_by_role(candidates, 8192)
    assert {name: len(rows) for name, rows in selected.items()} == {
        "fluid_fields": 7976,
        "interface": 192,
        "solid_temperature": 24,
    }
    for name in ("interface", "solid_temperature"):
        assert len(selected[name]) % 3 == 0


def test_cpu_interleaving_gives_each_variant_one_sample_per_round() -> None:
    calls = {
        name: (lambda value=name: value)
        for name in (
            "G_sparse",
            "P_sparse",
            "G_full_access_fallback",
            "P_full_access_fallback",
            "B_retained_reference",
        )
    }

    result = benchmark._interleaved(calls, device=torch.device("cpu"), warmups=2, repeats=5, seed=21)

    assert {name: row["sample_count"] for name, row in result["summary"].items()} == {name: 5 for name in calls}
    for repeat in range(1, 6):
        assert {row["variant"] for row in result["order_trace"] if row["round"] == repeat} == set(calls)
    assert all(
        row["p50_seconds"] > 0.0 and row["p90_seconds"] >= row["p50_seconds"] for row in result["summary"].values()
    )


def test_support_and_weighted_work_remain_separate() -> None:
    route_work = {
        mechanism: {
            "achieved_work": 7.0,
            "full_access_work": 28.0,
            "selected_unique_pairs": 5,
            "full_unique_pairs": 20,
            "executor": "hard_sparse",
        }
        for mechanism in MECHANISMS
    }
    record = {
        "all_mechanism_route_work": route_work,
        "all_mechanisms_total": {
            "achieved_work": 35.0,
            "full_access_work": 140.0,
            "selected_unique_pairs": 25,
            "full_unique_pairs": 100,
            "work_fraction": 0.25,
        },
        "full_access_bypass_routes": ["ME", "EM", "QM"],
    }

    result = benchmark._route_metrics(record, full=False)

    assert result["support"]["MM"]["selected_unique_pairs"] == 5
    assert result["support"]["MM"]["support_fraction"] == pytest.approx(0.25)
    assert result["canonical_work"]["MM"]["canonical_work_fraction"] == pytest.approx(0.25)
    assert result["support"]["MM"]["eligible_unique_pairs"] == 20
    assert result["full_access_fallback"] is False


def test_preflight_json_requires_completed_status_and_known_version() -> None:
    preflight = {
        "format_version": benchmark.FORMAT_VERSION,
        "status": "cpu_preflight_complete",
        "run_reserve_seconds": 7200,
        "workload": {"panels": {"standard": {"scenarios": [{"query_count": q} for q in benchmark.QUERY_COUNTS]}}},
    }
    benchmark._verify_preflight(preflight)

    preflight["status"] = "cpu_preflight_incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        benchmark._verify_preflight(preflight)

    preflight["format_version"] = -1
    with pytest.raises(ValueError, match="incomplete"):
        benchmark._verify_preflight(preflight)

    preflight["format_version"] = benchmark.FORMAT_VERSION
    preflight["status"] = "cpu_preflight_complete"
    preflight["workload"]["panels"]["standard"]["scenarios"].pop()
    with pytest.raises(ValueError, match="Q64, Q1024, and Q8192"):
        benchmark._verify_preflight(preflight)


def test_expected_checkpoint_sha_is_required_and_exact() -> None:
    expected = hashlib.sha256(b"checkpoint").hexdigest()
    benchmark._check_sha(expected, expected, "G endpoint")
    with pytest.raises(ValueError, match="mismatch"):
        benchmark._check_sha(expected, "0" * 64, "G endpoint")
    with pytest.raises(ValueError, match="explicit 64-character"):
        benchmark._check_sha(expected, "unbound", "G endpoint")


def test_renderer_latency_csv_matches_report_schema_and_bindings(tmp_path: Path) -> None:
    digest = "b" * 64
    sync_group = "Thermal-1804-standard_M4_Q1024-" + digest[:16]
    measurements = []
    expected_shas = {}
    for index, (variant, (model, access_mode)) in enumerate(benchmark.RENDERER_MODEL_BY_VARIANT.items()):
        checkpoint_sha = hashlib.sha256(f"checkpoint-{index}".encode()).hexdigest()
        expected_shas[model] = checkpoint_sha
        row = benchmark._renderer_latency_fields(
            model=model,
            checkpoint_sha256=checkpoint_sha,
            access_mode=access_mode,
            stats={"p50_seconds": 0.012, "p90_seconds": 0.016, "sample_count": 30},
            input_panel_sha256=digest,
            sync_group=sync_group,
        )
        assert row["lane"] == "thermal"
        assert row["model"] == model
        assert row["access_mode"] == access_mode
        assert row["p50_ms"] == pytest.approx(12.0)
        assert row["p90_ms"] == pytest.approx(16.0)
        assert row["n_cases"] == 1
        assert row["n_repeats"] == row["sample_count"] == 30
        assert row["complete_wrapper"] is True
        measurements.append({**row, "shape_id": "standard_M4_Q1024"})

    path = tmp_path / benchmark.RENDERER_LATENCY_NAME
    benchmark.write_renderer_latency_csv(path, measurements, "standard_M4_Q1024")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == list(benchmark.RENDERER_LATENCY_FIELDS)
    assert {row["model"] for row in rows} == {model for model, _ in benchmark.RENDERER_MODEL_BY_VARIANT.values()}
    assert {row["lane"] for row in rows} == {"thermal"}
    assert {row["checkpoint_sha256"] for row in rows} == set(expected_shas.values())
    assert all(row["input_panel_sha256"] == digest for row in rows)
    assert all(row["sync_group"] == sync_group for row in rows)
    assert all(row["complete_wrapper"] == "True" and row["n_cases"] == "1" for row in rows)
    assert all(float(row["p50_ms"]) == pytest.approx(12.0) for row in rows)
    assert all(float(row["p90_ms"]) == pytest.approx(16.0) for row in rows)


def _fake_complete_wrapper(*, full_access: bool):
    class Context:
        def __init__(self, phase: str, receiver_role: str):
            self.phase, self.receiver_role = phase, receiver_role

    class FakeBackend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("mm_message", "me_message", "em_message", "query_module_message", "env_geometry_bias"):
                setattr(self, name, torch.nn.Identity())

        def prepare(self, *, interaction_context=None):
            self.mm_message(torch.ones((1, 12, 12, 530)))
            self.me_message(torch.ones((1, 12, 192, 530)))
            self.em_message(torch.ones((1, 192, 12, 530)))
            state = {}
            if interaction_context is not None:
                state["cover_interaction_context"] = interaction_context
            if not full_access:
                state["cover_preparation_ledger"] = {
                    "cover_prepare_mm_executed_rows": 144,
                    "cover_prepare_me_executed_rows": 2304,
                    "cover_prepare_em_executed_rows": 2304,
                }
            return state

        def read(self, _state, _encoded, receivers, _features, *, interaction_context=None):
            rows = int(receivers.shape[1])
            self.query_module_message(torch.ones((1, rows, 12, 530)))
            self.env_geometry_bias(torch.ones((1, rows, 192, 18)))
            aux = {} if full_access else {
                "cover_qm_executed_rows": 12 * rows,
                "cover_qe_executed_rows": 192 * rows,
            }
            return receivers[..., :1] + 1.0, aux

    class FakeCore:
        def __init__(self):
            self.backend = FakeBackend()

        def prepare(self, *, interaction_context):
            context = interaction_context
            if full_access:
                return self.backend.prepare()
            return self.backend.prepare(interaction_context=context)

        def read(self, state, receivers, *, interaction_context):
            outputs = []
            for start in range(0, receivers.shape[1], 128):
                chunk = receivers[:, start : start + 128]
                if full_access:
                    value, _aux = self.backend.read(state, None, chunk, None)
                else:
                    value, _aux = self.backend.read(state, None, chunk, None, interaction_context=interaction_context)
                outputs.append(value)
            return torch.cat(outputs, dim=1)

    core = FakeCore()
    operator = SimpleNamespace(model=SimpleNamespace(core=core))

    def invoke() -> torch.Tensor:
        outputs = []
        for phase, role, rows in (("P0", "p0_port", 768), ("P1", "p1_refinement", 768), ("P2", "p2_field", 32)):
            context = Context(phase, role)
            state = core.prepare(interaction_context=context)
            receivers = torch.full((1, rows, 2), float(rows))
            outputs.append(core.read(state, receivers, interaction_context=context))
        return torch.cat(outputs, dim=1)

    return operator, invoke


def test_complete_wrapper_trace_groups_chunked_reads_and_sums_every_role() -> None:
    operator, invoke = _fake_complete_wrapper(full_access=False)
    before = invoke()
    with benchmark.capture_complete_wrapper_diagnostics(operator) as trace:
        after = invoke()
    assert torch.equal(before, after)

    telemetry = benchmark.complete_executor_row_telemetry(trace, expected_field_chunks=1, full_access=False)
    dense = benchmark.native_dense_feature_rows(
        trace,
        identity_binding={
            "endpoint_checkpoint_sha256": "a" * 64,
            "selected_rows_sha256": "b" * 64,
            "selected_query_ids_sha256": "c" * 64,
            "input_query_panel_digest": "d" * 64,
        },
        expected_field_chunks=1,
        full_access=False,
    )
    assert telemetry["status"] == "measured_complete"
    assert telemetry["prepare_calls_observed"] == 3
    assert telemetry["read_outer_requests_observed"] == 3
    assert telemetry["read_backend_chunks_observed"] == 13
    assert telemetry["by_mechanism"] == {"MM": 432, "ME": 6912, "EM": 6912, "QM": 18816, "QE": 301056}
    assert [len(request["backend_chunks"]) for request in trace["core_read_requests"]] == [6, 6, 1]
    assert [sum(chunk["receiver_rows"] for chunk in request["backend_chunks"]) for request in trace["core_read_requests"]] == [768, 768, 32]
    assert dense["by_mechanism"] == telemetry["by_mechanism"]
    assert dense["module_call_counts_match_expected"] is True


def test_full_access_binds_context_from_outer_core_and_keeps_cover_rows_unavailable() -> None:
    operator, invoke = _fake_complete_wrapper(full_access=True)
    before = invoke()
    with benchmark.capture_complete_wrapper_diagnostics(operator) as trace:
        after = invoke()
    assert torch.equal(before, after)
    assert [row["context_source"] for row in trace["prepare_calls"]] == ["outer_core_prepare"] * 3
    assert all(chunk["context_source"] == "outer_core_read" for chunk in trace["read_calls"])

    telemetry = benchmark.complete_executor_row_telemetry(trace, expected_field_chunks=1, full_access=True)
    dense = benchmark.native_dense_feature_rows(
        trace,
        identity_binding={"endpoint_checkpoint_sha256": "c" * 64, "input_query_panel_digest": "e" * 64},
        expected_field_chunks=1,
        full_access=True,
    )
    assert telemetry["status"] == "unavailable" and telemetry["by_mechanism"] is None
    assert telemetry["reason"] == "full_access_native_bypass_has_no_cover_executor_ledger_or_read_aux"
    assert dense["by_mechanism"] == {"MM": 432, "ME": 6912, "EM": 6912, "QM": 18816, "QE": 301056}
    assert dense["module_call_counts_match_expected"] is True


def test_complete_wrapper_trace_rejects_chunk_sum_or_role_mismatch() -> None:
    operator, invoke = _fake_complete_wrapper(full_access=True)
    with benchmark.capture_complete_wrapper_diagnostics(operator) as trace:
        invoke()
    trace["core_read_requests"][0]["backend_chunks"][0]["receiver_rows"] += 1
    with pytest.raises(RuntimeError, match="do not sum to outer request"):
        benchmark.complete_executor_row_telemetry(trace, expected_field_chunks=1, full_access=True)

    operator, invoke = _fake_complete_wrapper(full_access=True)
    with benchmark.capture_complete_wrapper_diagnostics(operator) as trace:
        invoke()
    trace["core_read_requests"][2]["receiver_role"] = "p1_refinement"
    with pytest.raises(RuntimeError, match="Outer core.read roles"):
        benchmark.complete_executor_row_telemetry(trace, expected_field_chunks=1, full_access=True)


@pytest.mark.parametrize("failure", ["missing_input", "missing_call"])
def test_native_feature_rows_marks_incomplete_module_telemetry_partial(failure: str) -> None:
    operator, invoke = _fake_complete_wrapper(full_access=True)
    with benchmark.capture_complete_wrapper_diagnostics(operator) as trace:
        invoke()
    if failure == "missing_input":
        trace["feature_module_calls"]["me_message"][0] = {"feature_input_rows": None, "input_shape": None}
    else:
        trace["feature_module_calls"]["em_message"].pop()
    result = benchmark.native_dense_feature_rows(
        trace,
        identity_binding={"endpoint_checkpoint_sha256": "d" * 64},
        expected_field_chunks=1,
        full_access=True,
    )
    assert result["status"] == "partially_unavailable"
    assert result["by_mechanism"] is None and result["total_feature_input_rows"] is None
    assert result["module_call_counts_match_expected"] is (failure != "missing_call")
    if failure == "missing_input":
        assert result["missing_input_modules"] == ["me_message"]
    else:
        assert result["missing_input_modules"] == []


def test_telemetry_identity_binding_uses_exact_preflight_schema_and_baseline_checkpoint() -> None:
    preflight = {"controlled_run": {
        "run_id": "run",
        "manifest_sha256": "1" * 64,
        "source_checkpoint_sha256": "2" * 64,
        "G": {"sha256": "3" * 64},
        "P": {"sha256": "4" * 64},
    }}
    panel = {"path": "/panel.npz", "source_sha256": "5" * 64}
    shape = {
        "shape_id": "standard_M5_Q64", "query_count": 64,
        "selected_rows_sha256": "6" * 64,
        "selected_query_ids_sha256": "7" * 64,
        "input_query_panel_digest": "8" * 64,
    }
    bound = benchmark._telemetry_identity_binding(
        preflight, panel_key="standard", panel=panel, shape=shape, variant="B_retained_reference"
    )
    assert bound["selected_rows_sha256"] == shape["selected_rows_sha256"]
    assert bound["selected_query_ids_sha256"] == shape["selected_query_ids_sha256"]
    assert bound["input_query_panel_digest"] == shape["input_query_panel_digest"]
    assert bound["endpoint_checkpoint_sha256"] == preflight["controlled_run"]["source_checkpoint_sha256"]


def test_run_ledger_appends_jsonl_events(tmp_path: Path) -> None:
    ledger = tmp_path / "latency_run_ledger.jsonl"
    benchmark._append_run_event(ledger, {"event": "start", "elapsed_monotonic_seconds": 0.0})
    benchmark._append_run_event(ledger, {"event": "stop", "elapsed_monotonic_seconds": 0.5})

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["start", "stop"]
    assert rows[-1]["elapsed_monotonic_seconds"] >= rows[0]["elapsed_monotonic_seconds"]
    assert all(row["timestamp_utc"].endswith("+00:00") for row in rows)
    assert all(row["kernel_boot_id"] and row["pid"] > 0 for row in rows)


def test_interleaved_run_honors_expired_cpu_deadline() -> None:
    called = False

    def call() -> None:
        nonlocal called
        called = True

    with pytest.raises(TimeoutError, match="reserved wall-clock limit"):
        benchmark._interleaved(
            {"G_sparse": call},
            device=torch.device("cpu"),
            warmups=0,
            repeats=1,
            seed=1,
            deadline_monotonic=time.monotonic() - 1,
        )
    assert called is False
