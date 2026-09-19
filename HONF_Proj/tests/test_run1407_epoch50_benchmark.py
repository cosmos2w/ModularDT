"""CPU/static contracts for the Run-1407 matched benchmark tool."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools" / "diagnostics"))

from run_run1407_epoch50_benchmark import (
    COMPARISON_LABELS,
    DEFAULT_CASE_IDS,
    _measurement_summary,
    _normal_executor_counters,
    build_parser,
    build_plan,
)


def _args(tmp_path: Path, *extra: str):
    paths = []
    for label in COMPARISON_LABELS:
        path = tmp_path / f"{label}.pt"
        path.write_bytes(b"placeholder")
        paths.append(path)
    return build_parser().parse_args(
        [
            "--checkpoint-1407",
            str(paths[0]),
            "--checkpoint-1406",
            str(paths[1]),
            "--checkpoint-1804",
            str(paths[2]),
            "--output",
            str(tmp_path / "benchmark.json"),
            *extra,
        ]
    )


def test_plan_requires_three_explicit_epoch50_inputs_and_records_protocol(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))

    assert plan["status"] == "plan_only"
    assert list(plan["checkpoint_policy"]) == ["1407", "1406", "1804"]
    assert plan["protocol"]["case_ids"] == list(DEFAULT_CASE_IDS)
    assert plan["protocol"]["query_count"] == 8192
    assert plan["protocol"]["receiver_chunk_size"] == 2048
    assert plan["protocol"]["inference_warmups"] == 2
    assert plan["protocol"]["inference_repetitions"] == 5
    assert plan["protocol"]["training_batch_size"] == 48
    assert plan["protocol"]["training_query_count"] == 1024
    assert plan["protocol"]["training_buckets"] == ["M1", "M12"]
    assert plan["protocol"]["training_warmups"] == 1
    assert plan["protocol"]["training_repetitions"] == 3
    assert plan["protocol"]["timed_maps"] is False
    assert plan["protocol"]["optimizer_state_policy"] == {
        "kind": "fresh_disposable_optimizer",
        "reference_hyperparameters": "Run-1804 checkpoint training config",
        "restore_checkpoint_optimizer_state": False,
        "save_updated_checkpoint": False,
    }
    assert plan["counter_contract"]["fields"] == [
        "support_pairs",
        "executed_rows",
        "padded_rows",
        "recomputed_rows",
    ]


def test_plan_rejects_protocol_changes(tmp_path: Path) -> None:
    args = _args(tmp_path, "--query-count", "1024")
    with pytest.raises(ValueError, match="query_count=8192"):
        build_plan(args)

    args = _args(tmp_path, "--case-id", "0273")
    with pytest.raises(ValueError, match="exactly cases 0273 and 0653"):
        build_plan(args)


def test_normal_executor_counters_keep_support_executed_padded_and_recompute_distinct() -> None:
    ledger = {
        "P2": {
            "module": {
                "support_pair_count": 10,
                "executed_rows": 12,
                "padded_rows": 3,
                "recomputed_rows": 12,
                "logical_path_count": 18,
                "valid_pair_denominator": 20,
                "padded_pair_denominator": 4,
            },
            "environment": {
                "unique_pair_count": 9,
                "actual_fine_call_count": 9,
                "fine_rows_padded": 2,
                "checkpoint_recompute_count": 9,
            },
        }
    }
    counters = _normal_executor_counters(ledger)
    module = counters["P2"]["module"]
    assert module["status"] == "ok"
    assert module["support_pairs"] == 10
    assert module["executed_rows"] == 12
    assert module["padded_rows"] == 3
    assert module["recomputed_rows"] == 12
    assert module["valid_pair_denominator"] == 20
    assert module["padded_pair_denominator"] == 4

    environment = counters["P2"]["environment"]
    assert environment["support_pairs"] == 9
    assert environment["executed_rows"] == 9
    assert environment["padded_rows"] == 2
    assert environment["recomputed_rows"] == 9
    assert environment["status"] == "ok"

    assert counters["P0"]["module"]["status"] == "unavailable"
    assert counters["P1"]["environment"]["executed_rows"] is None


def test_counter_normalization_does_not_relabel_dense_denominator_as_padded_rows() -> None:
    counters = _normal_executor_counters(
        {
            "P2": {
                "module": {
                    "unique_pair_count": 5,
                    "actual_fine_call_count": 5,
                    "padded_pair_denominator": 11,
                    "checkpoint_recompute_count": 0,
                }
            }
        }
    )
    row = counters["P2"]["module"]
    assert row["support_pairs"] == 5
    assert row["executed_rows"] == 5
    assert row["padded_rows"] is None
    assert row["padded_pair_denominator"] == 11
    assert row["status"] == "unavailable"


def test_measurement_summary_retains_baseline_and_peak_memory_fields() -> None:
    summary = _measurement_summary(
        {
            "status": "complete",
            "median_seconds": 0.011,
            "samples": [
                {
                    "status": "complete",
                    "elapsed_seconds": 0.010,
                    "baseline_allocated_bytes": 100,
                    "baseline_reserved_bytes": 200,
                    "peak_allocated_bytes": 140,
                    "peak_reserved_bytes": 260,
                    "incremental_peak_allocated_bytes": 40,
                    "incremental_peak_reserved_bytes": 60,
                },
                {
                    "status": "complete",
                    "elapsed_seconds": 0.012,
                    "baseline_allocated_bytes": 120,
                    "baseline_reserved_bytes": 220,
                    "peak_allocated_bytes": 160,
                    "peak_reserved_bytes": 280,
                    "incremental_peak_allocated_bytes": 40,
                    "incremental_peak_reserved_bytes": 60,
                },
            ],
        }
    )
    assert summary["median_seconds"] == pytest.approx(0.011)
    assert summary["median_baseline_allocated_bytes"] == pytest.approx(110.0)
    assert summary["median_baseline_reserved_bytes"] == pytest.approx(210.0)
    assert summary["median_peak_allocated_bytes"] == pytest.approx(150.0)
    assert summary["median_peak_reserved_bytes"] == pytest.approx(270.0)
    assert summary["timed_maps"] is False
