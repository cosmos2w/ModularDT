from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools/diagnostics/analyze_run1406_run1407_best5000_comparison.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("mature_comparison", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_reader_deduplicates_and_recovers_expanded_rows(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "metrics.csv"
    header = [
        "epoch",
        "val_loss_total",
        "val_field_mse",
        "val_temperature_mse",
        "peak_cuda_memory_mb",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow([1, 4.0, 3.0, 2.0, 100.0])
        writer.writerow([1, 4.0, 3.0, 2.0, 101.0])
        writer.writerow([2, "nan", "nan", "nan", "nan", 9.0, 8.0, 7.0])

    rows = module.read_training_rows(path)

    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[0]["peak_cuda_memory_mb"] == 101.0
    assert rows[1]["val_loss_total"] == 9.0
    assert rows[1]["val_field_mse"] == 8.0
    assert rows[1]["val_temperature_mse"] == 7.0
    assert rows[1]["expanded_schema_row"] == 1


def test_accuracy_summary_uses_pooled_sse_and_paired_cases() -> None:
    module = _module()
    rows = []
    for label, error in (("left", 1.0), ("right", 2.0)):
        for case_index in range(90):
            row = {
                "model_label": label,
                "case_id": f"{case_index:04d}",
                "global_field_fluid_norm_l2": str(error),
            }
            for prefix in module.COMPONENTS.values():
                row[f"{prefix}_sse"] = str(error**2)
                row[f"{prefix}_target_sse"] = "1.0"
            rows.append(row)

    summaries, pairs, components = module.accuracy_summary(
        rows, {"Left": "left", "Right": "right"}
    )

    assert summaries[0]["pooled_fluid_relative_l2"] == 1.0
    assert summaries[1]["pooled_fluid_relative_l2"] == 2.0
    assert pairs[0]["candidate_wins"] == 90
    assert pairs[0]["baseline_wins"] == 0
    assert components[0]["Near interface"] == 1.0
