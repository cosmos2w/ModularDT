"""CPU/static tests for the Run-1406 group-control evidence tools."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "tools" / "diagnostics"), str(_ROOT / "src")]

from group_control_evidence import (
    build_group_control_phase_records,
    canonicalize_group_control_arrays,
    phase_ledger,
    save_group_control_npz,
    semantic_metrics,
)
from render_group_control_interaction_board import render_board
from run_group_control_comparison import (
    build_parser,
    build_plan,
    decision_evidence,
)


def _fixture_payload() -> dict[str, np.ndarray | dict[str, object]]:
    return {
        "module_coords": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "module_present": np.ones((2,), dtype=np.float32),
        "env_coords": np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "env_weights": np.ones((2,), dtype=np.float32),
        "query_xy": np.asarray([[0.1, 0.2], [0.9, 0.2]], dtype=np.float32),
        "group_control_module_incidence": np.asarray(
            [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "group_control_environment_incidence": np.asarray(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "group_control_query_routing": np.asarray(
            [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "group_control_module_control_moment": np.ones((2, 2, 16), dtype=np.float32),
        "group_control_environment_control_moment": np.ones((2, 2, 16), dtype=np.float32),
        "group_control_phase_ledger": {
            "P0": {
                "module": {
                    "chunks": [
                        {"logical_path_count": 2, "unique_pair_count": 1, "actual_fine_call_count": 1, "valid_pair_denominator": 2, "padded_pair_denominator": 3},
                        {"logical_path_count": 3, "unique_pair_count": 2, "actual_fine_call_count": 2, "valid_pair_denominator": 4, "padded_pair_denominator": 5},
                    ],
                    "module_mlp_rows": 7,
                },
                "environment": {
                    "logical_path_count": 4,
                    "unique_pair_count": 3,
                    "actual_fine_call_count": 3,
                    "environment_geometry_network_rows": 8,
                    "environment_content_rows": 9,
                    "scalar_control_rows": 10,
                    "source_projection_rows": 11,
                    "forward_call_count": 12,
                    "checkpoint_recompute_count": 13,
                    "valid_pair_denominator": 14,
                    "padded_pair_denominator": 15,
                },
            },
            "P1": {"module": {"actual_fine_call_count": 2}, "environment": {"actual_fine_call_count": 3}},
            "P2": {"module": {"actual_fine_call_count": 4}, "environment": {"actual_fine_call_count": 4}},
        },
    }


def _arrays_metrics_ledger() -> tuple[object, object, dict[str, object]]:
    arrays, records = canonicalize_group_control_arrays(_fixture_payload())
    metrics = semantic_metrics(arrays, selected_query_index=0, prepared_decode_median_ms=4.25)
    ledger = phase_ledger(arrays, records)
    return arrays, metrics, ledger


def test_logical_paths_are_distinct_from_unique_pairs_and_moments() -> None:
    arrays, metrics, _ = _arrays_metrics_ledger()
    assert metrics.values["module_logical_path_count"] > metrics.values["module_unique_pair_count"]
    assert metrics.values["module_multiplicity"] > 1.0
    assert metrics.values["D"] == 16
    assert metrics.selected_module_logical_triples.tolist() == [[0, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert metrics.selected_module_unique_pairs.shape == (2, 3)
    assert arrays.module_group_centres.shape == (6, 2)


def test_group_control_h_derives_exact_d_wide_pair_moments() -> None:
    payload = _fixture_payload()
    payload.pop("group_control_module_control_moment")
    payload.pop("group_control_environment_control_moment")
    payload["group_control_h"] = np.ones((1, 6, 16), dtype=np.float32)
    arrays, _ = canonicalize_group_control_arrays(payload)
    metrics = semantic_metrics(arrays, selected_query_index=0)
    assert arrays.group_control is not None
    assert arrays.module_control_moment is not None
    assert arrays.environment_control_moment is not None
    assert metrics.values["D"] == 16
    assert np.allclose(metrics.selected_module_moments[0], 2.0)
    assert np.allclose(metrics.selected_module_moments[1], 1.0)


def test_phase_ledger_sums_receiver_chunk_numerators_without_inventing_calls() -> None:
    _, _, ledger = _arrays_metrics_ledger()
    module = ledger["P0"]["module"]
    assert module["logical_path_count"] == 5
    assert module["unique_pair_count"] == 3
    assert module["backend_logical_path_count"] == 5
    assert module["backend_unique_pair_count"] == 3
    assert module["actual_fine_call_count"] == 3
    assert module["valid_pair_denominator"] == 6
    assert module["padded_pair_denominator"] == 8
    assert module["receiver_chunk_aggregation"]["actual_fine_call_count"]["policy"] == "sum_over_receiver_chunks"
    assert ledger["P1"]["module"]["status"] == "ok"
    assert ledger["P2"]["environment"]["actual_fine_call_count"] == 4


def test_raw_thermal_phase_prefixes_map_only_explicit_p0_p1_p2_records() -> None:
    payload = _fixture_payload()
    payload.pop("group_control_phase_ledger")
    payload.update(
        {
            "initial_port_group_control_module_logical_paths": np.asarray(5.0),
            "initial_port_group_control_module_unique_pairs": np.asarray(3.0),
            "initial_port_group_control_module_fine_rows": np.asarray(3.0),
            "initial_port_group_control_module_checkpoint_recomputations": np.asarray(1.0),
            "provisional_group_control_environment_logical_paths": np.asarray(6.0),
            "provisional_group_control_environment_unique_pairs": np.asarray(4.0),
            "provisional_group_control_environment_fine_rows": np.asarray(4.0),
            "provisional_group_control_environment_geometry_rows_forward": np.asarray(4.0),
            "provisional_group_control_environment_content_dot_rows_forward": np.asarray(8.0),
            "port_global_group_control_module_logical_paths": np.asarray(2.0),
            "port_global_group_control_module_unique_pairs": np.asarray(2.0),
            "port_global_group_control_module_fine_rows": np.asarray(2.0),
            "group_control_environment_logical_paths": np.asarray(7.0),
            "group_control_environment_unique_pairs": np.asarray(5.0),
            "group_control_environment_fine_rows": np.asarray(5.0),
        }
    )
    records = build_group_control_phase_records(payload)
    assert records["P0"]["module"]["logical_path_count"] == 5.0
    assert records["P1"]["environment"]["environment_geometry_network_rows"] == 4.0
    assert records["P2"]["environment"]["actual_fine_call_count"] == 5.0
    assert records["P2_consistency"]["module"]["unique_pair_count"] == 2.0
    assert "P0" in records and "P1" in records
    _, canonical_records = canonicalize_group_control_arrays(payload)
    assert canonical_records is not None
    assert canonical_records["P0"]["module"]["actual_fine_call_count"] == 3.0

    p2_only = {key: value for key, value in payload.items() if not str(key).startswith(("initial_port_", "provisional_", "port_global_"))}
    p2_records = build_group_control_phase_records(p2_only)
    assert "P2" in p2_records
    assert "P0" not in p2_records
    assert "P1" not in p2_records


def test_board_renders_logical_and_unique_work_panels(tmp_path: Path) -> None:
    arrays, metrics, ledger = _arrays_metrics_ledger()
    maps = tmp_path / "maps"
    paths = {}
    for case_id in ("0273", "0653"):
        path = save_group_control_npz(
            maps / f"1406__{case_id}.npz",
            arrays,
            metrics,
            ledger,
            metadata={"label": "1406", "case_id": case_id},
        )
        paths[case_id] = path
    manifest = render_board({"1406": paths}, tmp_path / "board")
    assert manifest["status"] == "ok"
    assert len(manifest["rows"]) == 2
    assert manifest["semantics"]["unique_pairs"].startswith("one q")
    for name in ("group_control_interaction_board.png", "group_control_interaction_board.pdf", "group_control_interaction_board.json"):
        assert (tmp_path / "board" / name).is_file()


def _comparison_payload() -> dict[str, object]:
    inference = []
    for label, scale in (("1406", 0.9), ("1804", 1.0)):
        for case_id in ("0273", "0653"):
            inference.append(
                {
                    "label": label,
                    "case_id": case_id,
                    "phases": {"full_physical_forward": {"median_seconds": scale, "peak_allocated_bytes": 100.0 * scale}},
                    "semantic": {"R_M": 1.4, "R_E": 1.7},
                    "phase_ledger": {
                        "P2": {
                            "module": {"unique_pair_count": 10, "actual_fine_call_count": 10},
                            "environment": {"unique_pair_count": 20, "actual_fine_call_count": 20},
                        }
                    },
                }
            )
    return {
        "inference": inference,
        "training": [
            {"label": "1406", "bucket": {"label": "M12"}, "measurement": {"median_seconds": 0.9, "peak_allocated_bytes": 100.0}},
            {"label": "1804", "bucket": {"label": "M12"}, "measurement": {"median_seconds": 1.0, "peak_allocated_bytes": 100.0}},
        ],
        "learning_evidence": {"1406": {"status": "pass", "checks": {"improving_validation_median": True}}},
    }


def test_decision_evidence_keeps_fixed_thresholds_without_an_r_gate() -> None:
    evidence = decision_evidence(_comparison_payload())
    assert evidence["status"] == "pass"
    assert evidence["criteria"]["mean_full_forward_speed"]["status"] == "pass"
    assert evidence["criteria"]["execution_semantics"]["status"] == "pass"
    assert evidence["criteria"]["peak_allocated_memory"]["status"] == "pass"
    assert "R_M/R_E below one" in evidence["note"]


def test_plan_has_only_explicit_1406_and_1804_inputs(tmp_path: Path) -> None:
    checkpoint_1406 = tmp_path / "1406.pt"
    checkpoint_1804 = tmp_path / "1804.pt"
    checkpoint_1406.write_bytes(b"placeholder")
    checkpoint_1804.write_bytes(b"placeholder")
    args = build_parser().parse_args(
        [
            "--checkpoint-1406",
            str(checkpoint_1406),
            "--checkpoint-1804",
            str(checkpoint_1804),
            "--output",
            str(tmp_path / "comparison.json"),
            "--plan-only",
        ]
    )
    plan = build_plan(args)
    assert plan["status"] == "plan_only"
    assert set(plan["checkpoint_policy"]) == {"1406", "1804"}
    assert plan["protocol"]["query_count"] == 8192
    assert plan["protocol"]["receiver_chunk_size"] == 2048
    assert plan["protocol"]["training_labels"] == ["1406", "1804"]
    assert plan["protocol"]["timed_maps"] is False


def test_plan_rejects_noncontrolled_query_or_chunk_count(tmp_path: Path) -> None:
    paths = []
    for label in ("1406", "1804"):
        path = tmp_path / f"{label}.pt"
        path.write_bytes(b"placeholder")
        paths.append(path)
    args = build_parser().parse_args(
        [
            "--checkpoint-1406",
            str(paths[0]),
            "--checkpoint-1804",
            str(paths[1]),
            "--query-count",
            "1024",
            "--output",
            str(tmp_path / "comparison.json"),
            "--plan-only",
        ]
    )
    with pytest.raises(ValueError, match="query_count=8192"):
        build_plan(args)
