"""CPU/static tests for the Run-1405 evidence and board contracts."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "tools" / "diagnostics"), str(_ROOT / "src")]

from fixed_group_evidence import (
    FixedGroupEvidenceError,
    canonicalize_fixed_group_arrays,
    collect_debug_payload,
    load_evidence_npz,
    save_evidence_npz,
    semantic_metrics,
)
from render_fixed_group_interaction_board import render_board
from run_run1405_epoch50_comparison import (
    TRAIN_HEALTH_CORE_COLUMNS,
    TRAIN_HEALTH_DIAGNOSTIC_COLUMNS,
    _markdown_report,
    build_parser,
    build_plan,
    continuation_criteria,
    fidelity_gate,
    training_health_gate,
)


def _fixture_payload() -> dict[str, np.ndarray]:
    return {
        "module_coords": np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        "module_present": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "env_coords": np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32),
        "env_weights": np.ones((4,), dtype=np.float32),
        "module_incidence": np.asarray(
            [
                [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "environment_incidence": np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "module_group_centres": np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            dtype=np.float32,
        ),
        "environment_group_centres": np.asarray(
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]],
            dtype=np.float32,
        ),
        "query_xy": np.asarray([[0.1, 0.2], [1.5, 0.2], [4.5, 0.2]], dtype=np.float32),
        "query_routing": np.asarray(
            [
                [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }


def _write_training_health_csv(
    path: Path,
    *,
    missing_epoch: int | None = None,
    zero_group: str | None = None,
    worsening: bool = False,
    catastrophic: bool = False,
) -> Path:
    fieldnames = ["epoch", *TRAIN_HEALTH_CORE_COLUMNS, *TRAIN_HEALTH_DIAGNOSTIC_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, 51):
            if epoch == missing_epoch:
                continue
            validation = 1.0 - 0.01 * epoch
            if worsening:
                validation = 0.5 if epoch <= 10 else 0.6
            if catastrophic and epoch in (41, 42):
                validation = 2.0
            row = {
                "epoch": epoch,
                "loss_total": 2.0 / (epoch + 1),
                "loss_field": 1.5 / (epoch + 1),
                "field_mse": 1.25 / (epoch + 1),
                "val_loss_total": 1.8 / (epoch + 1),
                "val_loss_field": 1.4 / (epoch + 1),
                "val_field_mse": validation,
            }
            row.update({column: "nan" for column in TRAIN_HEALTH_DIAGNOSTIC_COLUMNS})
            if epoch in (1, 10, 20, 50):
                for column in TRAIN_HEALTH_DIAGNOSTIC_COLUMNS:
                    row[column] = 1.0
                if zero_group is not None:
                    row[f"preclip_gradient_norm_{zero_group}"] = 0.0
            writer.writerow(row)
    return path


def test_semantic_metrics_use_actual_positive_triples_and_skip_inactive_modules() -> None:
    arrays = canonicalize_fixed_group_arrays(_fixture_payload())
    metrics = semantic_metrics(arrays, selected_query_index=0, prepared_decode_median_ms=1.25)

    # q0 selects groups 0 and 2, each carrying module 0; group 0 and group 2
    # each carry one environment source.  Inactive module 2 never counts.
    assert metrics.values["P_M"] == 3
    assert metrics.values["P_E"] == 3
    assert metrics.values["R_M"] == pytest.approx(3.0 / 6.0)
    assert metrics.values["R_E"] == pytest.approx(3.0 / 12.0)
    assert metrics.values["sQ"] == pytest.approx(4.0 / 3.0)
    assert metrics.values["sM"] == pytest.approx(1.5)
    assert metrics.values["sE"] == pytest.approx(1.0)
    assert metrics.values["prepared_decode_median_ms"] == pytest.approx(1.25)
    assert metrics.selected_module_triples.tolist() == [[0, 0, 0], [0, 2, 0]]
    assert metrics.selected_environment_triples.tolist() == [[0, 0, 0], [0, 2, 1]]


def test_debug_payload_accepts_model_containers_and_round_trips_npz(tmp_path: Path) -> None:
    payload = _fixture_payload()
    outputs = {"interaction_aux": {key: value for key, value in payload.items() if key not in {"module_coords", "module_present", "env_coords", "env_weights", "query_xy"}}, "prepared_state": object()}
    # Geometry is supplied by the prepared-state-compatible top-level payload
    # in this CPU fixture, matching the extractor's intended adapter contract.
    outputs["interaction_aux"].update({key: payload[key] for key in ("module_coords", "module_present", "env_coords", "env_weights")})
    extracted = collect_debug_payload(outputs, query_xy=payload["query_xy"])
    arrays = canonicalize_fixed_group_arrays(extracted)
    metrics = semantic_metrics(arrays)
    path = save_evidence_npz(tmp_path / "1405__0273.npz", arrays, metrics, metadata={"case_id": "0273"})
    loaded_arrays, loaded_metrics, metadata = load_evidence_npz(path)
    assert loaded_arrays.query_routing.shape == (3, 6)
    assert loaded_metrics.values["P_E"] == metrics.values["P_E"]
    assert metadata["case_id"] == "0273"


def test_board_renders_exact_maps_and_provenance(tmp_path: Path) -> None:
    arrays = canonicalize_fixed_group_arrays(_fixture_payload())
    metrics = semantic_metrics(arrays, selected_query_index=0, prepared_decode_median_ms=2.5)
    maps = tmp_path / "maps"
    for case_id in ("0273", "0653"):
        save_evidence_npz(maps / f"1405__{case_id}.npz", arrays, metrics, metadata={"case_id": case_id})
    manifest = render_board({"1405": {case: maps / f"1405__{case}.npz" for case in ("0273", "0653")}}, tmp_path / "board")
    assert manifest["status"] == "ok"
    assert len(manifest["rows"]) == 2
    for name in ("fixed_group_interaction_board.png", "fixed_group_interaction_board.pdf", "fixed_group_interaction_board.json"):
        assert (tmp_path / "board" / name).is_file()


def test_board_requires_both_anchors(tmp_path: Path) -> None:
    arrays = canonicalize_fixed_group_arrays(_fixture_payload())
    metrics = semantic_metrics(arrays)
    path = save_evidence_npz(tmp_path / "1405__0273.npz", arrays, metrics)
    with pytest.raises(ValueError, match="requires anchor cases"):
        render_board({"1405": {"0273": path}}, tmp_path / "board", case_ids=("0273",))


def test_fidelity_gate_has_no_silent_metric_fallback() -> None:
    assert fidelity_gate({"status": "ok", "metric": 0.19}, {"status": "ok", "metric": 0.10})["status"] == "pass"
    assert fidelity_gate({"status": "ok", "metric": 0.21}, {"status": "ok", "metric": 0.10})["status"] == "fail"
    assert fidelity_gate({"status": "unavailable"}, {"status": "ok", "metric": 0.10})["status"] == "unavailable"


def test_plan_requires_explicit_1401_and_records_policy(tmp_path: Path) -> None:
    checkpoints = []
    for label in ("1405", "1804", "1401"):
        path = tmp_path / f"{label}.pt"
        path.write_bytes(b"placeholder")
        checkpoints.append(path)
    metrics_path = tmp_path / "run1405_metrics.csv"
    metrics_path.write_text("epoch\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--checkpoint-1405",
            str(checkpoints[0]),
            "--checkpoint-1804",
            str(checkpoints[1]),
            "--checkpoint-1401",
            str(checkpoints[2]),
            "--metrics-1405",
            str(metrics_path),
            "--output",
            str(tmp_path / "plan.json"),
            "--plan-only",
        ]
    )
    plan = build_plan(args)
    assert plan["status"] == "plan_only"
    assert plan["checkpoint_policy"]["1401"]["selection_policy"] == "explicit_cli_checkpoint"
    assert "no matched epoch_50 assumption" in plan["checkpoint_policy"]["1401"]["epoch_role"]
    assert plan["protocol"]["query_count"] == 8192
    assert plan["protocol"]["receiver_chunk_size"] == 2048
    assert plan["protocol"]["training_labels"] == ["1405", "1804", "1401"]
    assert plan["protocol"]["training_buckets"] == ["M1", "M12"]
    assert plan["training_health"]["metrics_csv"] == str(metrics_path.resolve())
    assert "last-10 val_field_mse median < first-10 median" in plan["training_health"]["required_checks"]


def test_missing_debug_array_is_explicit() -> None:
    payload = _fixture_payload()
    payload.pop("query_routing")
    with pytest.raises(FixedGroupEvidenceError, match="missing query_routing"):
        canonicalize_fixed_group_arrays(payload)


def test_training_health_gate_requires_epoch50_diagnostics_and_improving_windows(tmp_path: Path) -> None:
    path = _write_training_health_csv(tmp_path / "metrics.csv")
    health = training_health_gate(path)

    assert health["status"] == "pass"
    assert health["checks"]["rows_through_epoch_50"]["status"] == "pass"
    assert health["checks"]["finite_core_train_validation_metrics"]["status"] == "pass"
    assert health["checks"]["finite_gradient_update_diagnostics"]["status"] == "pass"
    assert health["checks"]["positive_major_group_gradient_update_evidence"]["status"] == "pass"
    assert health["checks"]["last10_median_below_first10_median"]["status"] == "pass"
    assert health["checks"]["catastrophic_last10_instability"]["status"] == "pass"
    assert 50 in health["checks"]["finite_gradient_update_diagnostics"]["diagnostic_evidence_epochs"]


def test_training_health_gate_rejects_missing_rows_and_nonpositive_major_group(tmp_path: Path) -> None:
    missing = training_health_gate(_write_training_health_csv(tmp_path / "missing.csv", missing_epoch=25))
    assert missing["status"] == "fail"
    assert missing["checks"]["rows_through_epoch_50"]["status"] == "fail"
    assert 25 in missing["checks"]["rows_through_epoch_50"]["missing_epochs"]

    nonpositive = training_health_gate(
        _write_training_health_csv(tmp_path / "nonpositive.csv", zero_group="backend")
    )
    assert nonpositive["status"] == "fail"
    assert nonpositive["checks"]["positive_major_group_gradient_update_evidence"]["status"] == "fail"
    assert "backend" in nonpositive["checks"]["positive_major_group_gradient_update_evidence"]["nonpositive_or_nonfinite_groups"]


def test_training_health_gate_rejects_worsening_or_repeated_catastrophic_tail(tmp_path: Path) -> None:
    worsening = training_health_gate(_write_training_health_csv(tmp_path / "worsening.csv", worsening=True))
    assert worsening["status"] == "fail"
    assert worsening["checks"]["last10_median_below_first10_median"]["status"] == "fail"

    catastrophic = training_health_gate(
        _write_training_health_csv(tmp_path / "catastrophic.csv", catastrophic=True)
    )
    assert catastrophic["status"] == "fail"
    check = catastrophic["checks"]["catastrophic_last10_instability"]
    assert check["status"] == "fail"
    assert check["exceeding_count"] == 2


def test_markdown_report_includes_training_health_gate() -> None:
    report = _markdown_report(
        {
            "checkpoint_policy": {},
            "protocol": {},
            "inference": [],
            "training": [],
            "fidelity_gate": {"status": "unavailable"},
            "training_health": {
                "status": "fail",
                "path": "/tmp/run1405/metrics.csv",
                "checks": {
                    "rows_through_epoch_50": {"status": "fail"},
                    "finite_core_train_validation_metrics": {"status": "fail"},
                    "finite_gradient_update_diagnostics": {"status": "fail"},
                    "positive_major_group_gradient_update_evidence": {"status": "fail"},
                    "last10_median_below_first10_median": {
                        "status": "fail",
                        "first10_median_val_field_mse": 1.0,
                        "last10_median_val_field_mse": 1.1,
                    },
                    "catastrophic_last10_instability": {
                        "status": "fail",
                        "exceeding_count": 2,
                    },
                },
            },
            "continuation_criteria": {"status": "fail"},
        }
    )
    assert "## Epoch-50 training health" in report
    assert "Catastrophic rule" in report
    assert "/tmp/run1405/metrics.csv" in report


def test_continuation_criteria_keep_fixed_thresholds() -> None:
    def inference_row(label: str, case_id: str, seconds: float, peak: int) -> dict[str, object]:
        return {
            "label": label,
            "case_id": case_id,
            "phases": {"full_physical_forward": {"median_seconds": seconds, "peak_allocated_bytes": peak}},
            "semantic": {"status": "ok", "M_active": 2, "R_M": 0.5, "R_E": 0.5},
        }

    payload = {
        "inference": [
            inference_row("1405", "0273", 0.9, 105),
            inference_row("1405", "0653", 0.9, 105),
            inference_row("1804", "0273", 1.0, 100),
            inference_row("1804", "0653", 1.0, 100),
        ],
        "training": [
            {
                "label": "1405",
                "bucket": {"label": "M12"},
                "measurement": {"median_seconds": 0.9, "peak_allocated_bytes": 105},
            },
            {
                "label": "1804",
                "bucket": {"label": "M12"},
                "measurement": {"median_seconds": 1.0, "peak_allocated_bytes": 100},
            },
        ],
        "fidelity_gate": {"status": "pass"},
        "training_health": {"status": "pass"},
    }
    result = continuation_criteria(payload)
    assert result["status"] == "pass"
    assert result["criteria"]["mean_full_forward_speed"]["status"] == "pass"
    assert result["criteria"]["m12_step_speed"]["status"] == "pass"
    assert result["criteria"]["peak_allocated_memory"]["status"] == "pass"
    assert result["criteria"]["semantic_grouped_work"]["status"] == "pass"
    assert result["criteria"]["training_health"]["status"] == "pass"

    # One slower case must move the two-case mean beyond the fixed 0.95 gate.
    payload["inference"][0]["phases"]["full_physical_forward"]["median_seconds"] = 1.1
    assert continuation_criteria(payload)["criteria"]["mean_full_forward_speed"]["status"] == "fail"
