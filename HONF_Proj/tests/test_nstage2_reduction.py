"""Scientific selection and sparse-history regressions for NStage2 tables."""

from __future__ import annotations

import csv
import json
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


def test_mature_exact_endpoint_selects_epoch5000_and_uses_mature_paths(tmp_path: Path) -> None:
    rows = [
        {"checkpoint": "Run_1807_example/epoch_0500_model.pt", "epoch": "500"},
        {"checkpoint": "Run_1807_example/epoch_2500_model.pt", "epoch": "2500"},
        {"checkpoint": "Run_1807_example/epoch_5000_model.pt", "epoch": "5000"},
    ]
    assert reduction._filter_table_rows(rows, "1807", "exact5000") == [rows[-1]]
    assert reduction._summary_rows(rows, "1807", "exact5000") == [rows[-1]]
    assert reduction._candidate_table_dir(
        tmp_path / "maturity5000", "1807", "exact5000", 5000
    ) == tmp_path / "maturity5000/track_a/endpoint5000/tables"
    project = tmp_path / "project"
    assert reduction._parent_table_dir(project, "1801", 5000) == (
        project / reduction.STUDY_RELATIVE / "five_model_epoch5000/evaluation/tables"
    )


def test_mature_root_resolution_never_falls_back_to_historical_nstage2(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "diagnostics").mkdir()
    nstage2 = project / reduction.STUDY_RELATIVE / "nstage2"
    maturity = nstage2 / "maturity5000"
    maturity.mkdir(parents=True)
    parent = project / reduction.STUDY_RELATIVE / "five_model_epoch5000"
    parent.mkdir(parents=True)

    assert reduction._resolve_roots(project, 5000)[1] == maturity
    assert reduction._resolve_roots(nstage2, 5000)[1] == maturity
    assert reduction._resolve_roots(parent, 5000)[1] == maturity
    assert reduction._resolve_roots(maturity, 5000)[1] == maturity
    assert reduction._resolve_roots(project)[1] == nstage2

    default_project_mature = reduction.PROJECT_ROOT / reduction.STUDY_RELATIVE / "nstage2/maturity5000"
    assert reduction._resolve_roots(None, 5000)[1] == default_project_mature


def test_history_endpoint5000_keeps_milestones_and_larger_windows(
    tmp_path: Path, monkeypatch
) -> None:
    history = tmp_path / "history.csv"
    with history.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "epoch", "field_mse"])
        writer.writeheader()
        for epoch in (499, 500, 1000, 2500, 4951, 5000):
            writer.writerow({"run": "1807", "epoch": epoch, "field_mse": epoch / 10000})

    monkeypatch.setattr(
        reduction,
        "_history_source",
        lambda project, study, run: (history, "fixture history"),
    )
    comparison = tmp_path / "comparison"
    reduction._history_outputs(tmp_path, tmp_path / "study", comparison, 5000)

    curves = reduction.read_csv(comparison / "learning_curves.csv")
    assert max(int(row["epoch"]) for row in curves) == 5000
    windows = reduction.read_csv(comparison / "history_windows.csv")
    assert {int(row["requested_window_epochs"]) for row in windows} == set(
        reduction.HISTORY_WINDOW_SIZES
    )
    milestones = reduction.read_csv(comparison / "history_milestones.csv")
    assert {int(row["milestone_epoch"]) for row in milestones} == {500, 1000, 2500, 5000}
    manifest = json.loads((comparison / "history_manifest.json").read_text())
    assert manifest["budget_epoch"] == 5000
    assert manifest["milestones"] == [500, 1000, 2500, 5000]


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
