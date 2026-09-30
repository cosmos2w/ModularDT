"""CPU-only contracts for the bounded synchronized Wind latency panel."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_matured_action_selector.py"
SPEC = importlib.util.spec_from_file_location("wind_action_latency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LATENCY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LATENCY
SPEC.loader.exec_module(LATENCY)


def _table() -> dict[str, object]:
    return {
        "rows": [
            {
                "query_panel": "fixed", "selector_primary_fit_row": True,
                "row_id": row_id, "module_count": modules,
            }
            for row_id, modules in ((4, 6), (5, 8), (6, 12))
        ]
    }


def test_interleaving_has_every_mode_and_query_shape_once_per_round() -> None:
    order = LATENCY.interleaved_workload_order(
        ["q64", "q1024", "q8192", "large_m_q8192"],
        LATENCY.MODE_ORDER,
        rounds=4,
        seed=37,
    )
    expected = {
        (workload, mode)
        for workload in ("q64", "q1024", "q8192", "large_m_q8192")
        for mode in LATENCY.MODE_ORDER
    }
    assert all(len(round_order) == len(expected) and set(round_order) == expected for round_order in order)
    assert order == LATENCY.interleaved_workload_order(
        ["q64", "q1024", "q8192", "large_m_q8192"],
        LATENCY.MODE_ORDER,
        rounds=4,
        seed=37,
    )


def test_unavailable_selector_keeps_full_comparison_modes_schedulable() -> None:
    available, modes = LATENCY._selector_mode_contract(
        "unavailable_no_exposed_train_sparse_action"
    )
    assert available is False
    assert modes == ["g_full", "p_direct", "retained_wfull"]
    with pytest.raises(ValueError, match="Unsupported selector result state"):
        LATENCY._selector_mode_contract("unknown")
    available_modes = LATENCY.MODE_ORDER[1:]
    order = LATENCY.interleaved_workload_order(
        ["q64", "q1024", "q8192", "large_m_q8192"],
        available_modes,
        rounds=2,
        seed=91,
    )
    expected = {
        (workload, mode)
        for workload in ("q64", "q1024", "q8192", "large_m_q8192")
        for mode in available_modes
    }
    assert all(len(round_order) == 12 and set(round_order) == expected for round_order in order)
    assert "g_action_policy" not in {mode for _workload, mode in order[0]}
    assert {"g_full", "p_direct", "retained_wfull"} <= {
        mode for _workload, mode in order[0]
    }


def test_workload_selection_requires_large_m_and_primary_fixed_rows() -> None:
    assert LATENCY.choose_workload_rows(_table()) == (4, 6)
    assert LATENCY.choose_workload_rows(_table(), row_id=5, large_row_id=6) == (5, 6)
    with pytest.raises(ValueError, match="strictly more modules"):
        LATENCY.choose_workload_rows(_table(), row_id=6, large_row_id=5)
    with pytest.raises(ValueError, match="fixed primary-fit"):
        LATENCY.choose_workload_rows({"rows": []})


def test_query_counts_and_latency_summary_are_explicit_and_bounded() -> None:
    assert sum(LATENCY._query_counts(64).values()) == 64
    assert sum(LATENCY._query_counts(1024).values()) == 1024
    assert sum(LATENCY._query_counts(8192).values()) == 8192
    with pytest.raises(ValueError, match="fixed to Q64"):
        LATENCY._query_counts(512)
    summary = LATENCY.summarize_latencies([
        {"workload": "q64", "mode": "g_full", "elapsed_seconds": value}
        for value in (0.1, 0.2, 0.3)
    ])
    assert summary["q64/g_full"]["measured_repeats"] == 3
    assert summary["q64/g_full"]["p50_seconds"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="bounded diagnostic allocation"):
        LATENCY.measure_latency_panel(
            workloads=[], wrapper=lambda _mode, _workload: {},
            device=__import__("torch").device("cpu"), warmups=0, repeats=LATENCY.MAX_REPEATS + 1,
        )


def test_table_binding_accepts_durability_source_without_synthetic_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "g_u1750.pt"
    checkpoint.write_bytes(b"durability checkpoint fixture")
    checkpoint_sha = LATENCY._sha256(checkpoint)
    binding_sha = "a" * 64
    record = {
        "run_id": "2112", "arm": "g_packet", "update_count": 1750,
        "checkpoint_kind": "durability_only", "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
    }
    monkeypatch.setattr(
        LATENCY.panel, "_verified_checkpoint_binding",
        lambda *_args, **_kwargs: (checkpoint, record, binding_sha, "current_durability_checkpoint"),
    )
    table = {
        "format_version": 2,
        "forward_checkpoint_sha256": checkpoint_sha,
        "g_checkpoint_binding_source": "current_durability_checkpoint",
        "g_checkpoint_binding_record": record,
        "g_checkpoint_binding_record_sha256": binding_sha,
        "g_checkpoint_review_record": None,
        "g_checkpoint_review_record_line_sha256": None,
    }

    path, bound, digest, source = LATENCY._verified_table_checkpoint_binding(
        table, run_dir=tmp_path, arm="g_packet", update_count=1750
    )
    assert path == checkpoint
    assert bound["checkpoint_kind"] == "durability_only"
    assert digest == binding_sha
    assert source == "current_durability_checkpoint"

    table["g_checkpoint_review_record"] = dict(record)
    table["g_checkpoint_review_record_line_sha256"] = binding_sha
    with pytest.raises(ValueError, match="verified g_packet checkpoint binding source"):
        LATENCY._verified_table_checkpoint_binding(
            table, run_dir=tmp_path, arm="g_packet", update_count=1750
        )
