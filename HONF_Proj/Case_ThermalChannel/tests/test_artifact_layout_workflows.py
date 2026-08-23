from __future__ import annotations

import csv
from pathlib import Path

from channelthermal.workflows import evaluate_forward, evaluate_local
from channelthermal.workflows.train_forward import save_global_loss_plots


def test_forward_and_local_evaluators_share_canonical_run_root(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "Run_0001_fixture"
    checkpoint = run_dir / "checkpoints" / "best_total.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaluate_forward, "current_timestamp", lambda: "20260823_120000")
    monkeypatch.setattr(evaluate_local, "current_timestamp", lambda: "20260823_120000")

    expected = run_dir / "evaluations" / "single_case" / "0653_20260823_120000"
    assert evaluate_forward.evaluation_output_dir(None, checkpoint, "0653") == expected
    assert evaluate_local.evaluation_output_dir(None, checkpoint, "0653") == expected


def test_managed_forward_training_writes_only_canonical_plot_tree(tmp_path) -> None:
    run_dir = tmp_path / "Run_0001_fixture"
    (run_dir / "plots" / "training").mkdir(parents=True)
    (run_dir / "plots" / "diagnostics").mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    metrics_path = run_dir / "metrics.csv"
    fields = [
        "epoch",
        "loss_total",
        "val_loss_total",
        "loss_field",
        "val_loss_field",
        "field_mse",
        "val_field_mse",
        "temperature_mse",
        "val_temperature_mse",
        "selected_edge_count",
        "val_selected_edge_count",
        "functional_edge_count",
        "val_functional_edge_count",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: 1 if field == "epoch" else 0.5 for field in fields})

    save_global_loss_plots(metrics_path, run_dir)

    assert (run_dir / "plots" / "training" / "loss_curve.png").is_file()
    assert (run_dir / "plots" / "diagnostics" / "loss_total_curve.png").is_file()
    assert not (run_dir / "loss_curve.png").exists()
    assert not (run_dir / "diagnostic_plots").exists()
