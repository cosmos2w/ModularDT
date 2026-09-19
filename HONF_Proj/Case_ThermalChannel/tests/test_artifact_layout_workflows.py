from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from channelthermal.evaluation.results import summarize
from channelthermal.training.reporting import _read_metric_history
from channelthermal.workflows import evaluate_forward, evaluate_local
from channelthermal.workflows.train_forward import save_global_loss_plots

from honf_runtime.artifact_layout import EvaluationArtifactLayout


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


def test_group_control_activity_history_stays_aligned_across_resume(tmp_path) -> None:
    run_dir = tmp_path / "Run_1406_fixture"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "config_resolved.json").write_text(
        '{"model":{"core_honf":{"forward_architecture":"group_control_pairwise_honf"}}}\n',
        encoding="utf-8",
    )
    metrics_path = run_dir / "metrics.csv"
    fields = [
        "epoch",
        "loss_total",
        "val_loss_total",
        "interaction_module_group_incidence_count",
        "val_interaction_module_group_incidence_count",
        "interaction_environment_group_incidence_count",
        "val_interaction_environment_group_incidence_count",
        "interaction_group_read_degree_mean",
        "val_interaction_group_read_degree_mean",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 50,
                "loss_total": 1.0,
                "val_loss_total": 1.0,
                **{key: "nan" for key in fields[3:]},
            }
        )
        writer.writerow(
            {
                "epoch": 51,
                "loss_total": 0.9,
                "val_loss_total": 0.9,
                "interaction_module_group_incidence_count": 18,
                "val_interaction_module_group_incidence_count": 17,
                "interaction_environment_group_incidence_count": 420,
                "val_interaction_environment_group_incidence_count": 410,
                "interaction_group_read_degree_mean": 2.5,
                "val_interaction_group_read_degree_mean": 2.4,
            }
        )

    history = _read_metric_history(metrics_path)
    assert history["epoch"] == [50.0, 51.0]
    assert np.isnan(history["interaction_group_read_degree_mean"][0])
    assert history["interaction_group_read_degree_mean"][1] == 2.5

    save_global_loss_plots(metrics_path, run_dir)
    assert (run_dir / "plots" / "training" / "loss_curve.png").is_file()


def test_forward_summary_writes_metrics_csv_and_arrays(tmp_path) -> None:
    layout = EvaluationArtifactLayout.at(tmp_path / "evaluation")
    prediction = np.ones((2, 3, 5), dtype=np.float32)
    summary = summarize(
        {
            "case_id": "0653",
            "steady_field": np.zeros_like(prediction),
            "module_mask": np.zeros(prediction.shape[:2], dtype=bool),
        },
        {
            "suffix": "predicted",
            "pred_field_grid": prediction,
            "pred_internal_temperature": np.empty((0,), dtype=np.float32),
            "pred_interface": np.empty((0,), dtype=np.float32),
            "pred_port_condition": np.empty((0,), dtype=np.float32),
        },
        Path("best_by_field_mse_model.pt"),
        layout,
        ["u", "v", "p", "omega", "temperature"],
    )

    metrics_path = layout.metrics / "metrics_predicted.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "0653"
    assert rows[0]["field_mse"] == "1.0"
    assert (layout.arrays / "evaluation_outputs_predicted.npz").is_file()
    assert summary["outputs"]["metrics_csv"] == str(metrics_path)
