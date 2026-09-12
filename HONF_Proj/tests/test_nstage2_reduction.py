"""Scientific selection and sparse-history regressions for NStage2 tables."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "diagnostics"))

import nstage2_reduction as reduction


def test_completed_run_history_alias_is_not_a_second_run(tmp_path: Path) -> None:
    run_dir = tmp_path / reduction.RUN_ROOT_RELATIVE / "Run_1808_fixture"
    (run_dir / "metrics").mkdir(parents=True)
    primary = run_dir / "metrics.csv"
    primary.write_text("epoch,val_field_mse\n500,0.1\n")
    (run_dir / "metrics" / "metrics.csv").hardlink_to(primary)

    path, _ = reduction._history_source(tmp_path, tmp_path / "study", "1808")

    assert path == primary


def _endpoint_reuse_fixture(tmp_path: Path) -> tuple[Path, reduction.TableSet, Path]:
    run_dir = tmp_path / "Run_1808_fixture"
    run_dir.mkdir()
    endpoint_path = run_dir / "epoch_0500_model.pt"
    best_path = run_dir / "best_by_field_mse_model.pt"
    endpoint_path.write_bytes(b"endpoint")
    best_path.write_bytes(b"best")
    endpoint_table_dir = tmp_path / "endpoint500" / "tables"
    rows = [{"case_id": f"case-{index:03d}", "checkpoint": str(endpoint_path)} for index in range(90)]
    endpoint = reduction.TableSet(
        "1808",
        reduction.RUN_LABELS["1808"],
        "exact500",
        endpoint_table_dir,
        "available",
        rows=rows,
        summary={"checkpoint": str(endpoint_path)},
        checkpoint=str(endpoint_path),
        checkpoint_epoch=500,
    )
    return run_dir, endpoint, tmp_path / "best_field" / "tables"


def test_saved_best_reuses_equal_epoch500_endpoint_tables(tmp_path: Path, monkeypatch) -> None:
    _, endpoint, best_table_dir = _endpoint_reuse_fixture(tmp_path)
    state = {"weight": np.asarray([1.0, 2.0], dtype=np.float32)}
    payloads = {
        endpoint.checkpoint: {"epoch": 500, "model_state_dict": state},
        str(tmp_path / "Run_1808_fixture" / "best_by_field_mse_model.pt"): {
            "epoch": 500,
            "model_state_dict": {"weight": state["weight"].copy()},
        },
    }
    monkeypatch.setattr(reduction, "_load_trusted_checkpoint", lambda path: payloads[str(path)])

    reused = reduction._reuse_exact_endpoint_for_best(
        tmp_path, "1808", endpoint, best_table_dir
    )

    assert reused.available
    assert reused.phase == "best_field"
    assert reused.table_dir == endpoint.table_dir
    assert reused.rows is endpoint.rows
    assert reused.summary is endpoint.summary
    assert reused.summary["checkpoint"] == endpoint.checkpoint
    assert reused.checkpoint == str(tmp_path / "Run_1808_fixture" / "best_by_field_mse_model.pt")
    assert reused.checkpoint_epoch == 500
    assert "reused exact500 endpoint evaluation tables" in (reused.reason or "")
    assert "all 1 model_state_dict tensors equal" in (reused.reason or "")


@pytest.mark.parametrize(
    "best_payload,reason",
    [
        (
            {"epoch": 490, "model_state_dict": {"weight": np.asarray([1.0], dtype=np.float32)}},
            "not both epoch500",
        ),
        (
            {"epoch": 500, "model_state_dict": {"weight": np.asarray([9.0], dtype=np.float32)}},
            "model weights differ",
        ),
    ],
)
def test_saved_best_does_not_reuse_different_epoch_or_weights(
    tmp_path: Path, monkeypatch, best_payload, reason: str
) -> None:
    _, endpoint, best_table_dir = _endpoint_reuse_fixture(tmp_path)
    best_path = str(tmp_path / "Run_1808_fixture" / "best_by_field_mse_model.pt")
    payloads = {
        str(endpoint.checkpoint): {"epoch": 500, "model_state_dict": {"weight": np.asarray([1.0], dtype=np.float32)}},
        best_path: best_payload,
    }
    monkeypatch.setattr(reduction, "_load_trusted_checkpoint", lambda path: payloads[str(path)])

    refused = reduction._reuse_exact_endpoint_for_best(
        tmp_path, "1808", endpoint, best_table_dir
    )

    assert not refused.available
    assert refused.phase == "best_field"
    assert reason in (refused.reason or "")


def test_exact_endpoint_does_not_relabel_an_available_earlier_checkpoint() -> None:
    rows = [
        {"checkpoint": "Run_1807_example/epoch_0100_model.pt", "epoch": "100"},
        {"checkpoint": "Run_1807_example/epoch_0500_model.pt", "epoch": "500"},
    ]
    for select in (reduction._filter_table_rows, reduction._summary_rows):
        assert select(rows[:1], "1807", "exact500") == []
        assert select(rows, "1807", "exact500") == rows[1:]


def test_sparse_gradient_history_is_not_zero_filled_or_counted_twice(tmp_path: Path) -> None:
    study = tmp_path / "diagnostics/generated/interface_operator_study/nstage2"
    comparison = study / "comparison"
    comparison.mkdir(parents=True)
    run = tmp_path / reduction.RUN_ROOT_RELATIVE / "Run_1807_example"
    run.mkdir(parents=True)
    with (run / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "field_mse", "preclip_gradient_norm", "parameter_update_norm_group_prepare"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"epoch": 499, "field_mse": 0.2, "preclip_gradient_norm": "nan", "parameter_update_norm_group_prepare": "nan"},
                {"epoch": 500, "field_mse": 0.1, "preclip_gradient_norm": 3.0, "parameter_update_norm_group_prepare": 0.02},
            ]
        )
    reduction._history_outputs(tmp_path, study, comparison)
    updates = reduction.read_csv(comparison / "gradient_updates.csv")
    assert len(updates) == 2
    assert {row["epoch"] for row in updates} == {"500"}
    window = reduction.read_csv(comparison / "last50_window.csv")
    total = [row for row in window if row["metric"] == "preclip_gradient_norm"]
    assert len(total) == 1
    assert float(total[0]["mean"]) == 3.0
