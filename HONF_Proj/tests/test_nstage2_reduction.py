"""Scientific selection and sparse-history regressions for NStage2 tables."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "diagnostics"))

import nstage2_reduction as reduction


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
