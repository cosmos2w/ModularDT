from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from channelthermal.model_support import ChannelThermalModelSupportMixin
from channelthermal.training.reporting import save_global_loss_plots

from honf_forward_core.training.diagnostics import compute_honf_diagnostics


def _residual_organizer_fixture() -> dict[str, torch.Tensor]:
    return {
        "hyper_strength": torch.tensor([[0.20, 0.01, 0.01], [0.20, 0.01, 0.01]]),
        "case_adaptive_edge_count": torch.tensor([1.0, 2.0]),
        "case_adaptive_edge_cap": torch.tensor([3.0, 3.0]),
        "case_adaptive_soft_edge_count": torch.tensor([1.3, 2.0]),
        "edge_survival_weight": torch.tensor([[1.0, 0.2, 0.1], [1.0, 0.8, 0.2]]),
        "hard_case_edge_mask": torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        "residual_mechanism_strength": torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]]),
        "residual_module_factor": torch.ones(2, 1, 3),
        "residual_environment_factor": torch.ones(2, 1, 3),
        "residual_coupling_row_mass": torch.ones(2, 1),
        "residual_fraction_trace": torch.tensor([[1.0, 0.1, 0.01, 0.0], [1.0, 0.5, 0.2, 0.1]]),
        "residual_marginal_explained_fraction": torch.tensor([[0.9, 0.09, 0.01], [0.5, 0.3, 0.1]]),
        "case_adaptive_stop_reached": torch.tensor([1.0, 0.0]),
        "case_adaptive_cap_hit": torch.tensor([0.0, 1.0]),
        "case_adaptive_stop_margin": torch.tensor([0.01, -0.02]),
        "residual_monotonic_violation_max": torch.tensor([0.0, 0.001]),
    }


def test_residual_diagnostics_use_explicit_count_and_support() -> None:
    organizer = _residual_organizer_fixture()
    diagnostics = compute_honf_diagnostics(
        {
            "pred_field": torch.zeros(2, 1, 5),
            "organizer_aux": organizer,
            "routing_aux": {},
        }
    )

    assert diagnostics["case_adaptive_edge_count_mean"] == 1.5
    assert diagnostics["case_adaptive_edge_cap_mean"] == 3.0
    assert diagnostics["case_adaptive_soft_edge_count_mean"] == pytest.approx(1.65)
    assert diagnostics["case_adaptive_stop_reached_fraction"] == 0.5
    assert diagnostics["case_adaptive_cap_hit_fraction"] == 0.5
    # The final residual is sampled at each case's hard count, not at a fixed
    # hyper-strength threshold or necessarily at the packed-width endpoint.
    assert diagnostics["residual_fraction_final_mean"] == pytest.approx(0.15)
    assert diagnostics["residual_last_active_marginal_mean"] == pytest.approx(0.6)


def test_thermal_compatibility_view_uses_effective_residual_mask() -> None:
    mixin = ChannelThermalModelSupportMixin()
    mixin.config = SimpleNamespace(
        core_honf=SimpleNamespace(organizer_mode="case_adaptive_residual")
    )
    adapter = SimpleNamespace(
        module_centers=torch.zeros(1, 1, 2),
        module_present=torch.ones(1, 1),
        heat_powers=torch.ones(1, 1),
    )
    core_output = _residual_organizer_fixture()
    core_output.update(
        {
            "edge_active_mask": torch.ones(2, 3),
            "effective_edge_mask": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.5, 0.0]]),
            "module_env_context": torch.zeros(2, 1, 4),
        }
    )

    aux = mixin._legacy_organizer_aux(core_output, adapter, torch.zeros(2, 1, 2))

    assert torch.equal(aux["active_hyperedge_mask"], core_output["effective_edge_mask"])
    for key in (
        "residual_fraction_trace",
        "residual_marginal_explained_fraction",
        "residual_mechanism_strength",
        "residual_module_factor",
        "residual_environment_factor",
        "residual_coupling_row_mass",
        "edge_survival_weight",
        "hard_case_edge_mask",
        "case_adaptive_edge_count",
        "case_adaptive_edge_cap",
        "case_adaptive_soft_edge_count",
        "case_adaptive_cap_hit",
    ):
        assert key in aux


def test_training_reporting_writes_compact_organizer_health_plot(tmp_path: Path) -> None:
    run_dir = tmp_path / "Run_0001_fixture"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    metrics_path = run_dir / "metrics.csv"
    fields = [
        "epoch",
        "loss_total",
        "val_loss_total",
        "case_adaptive_edge_count_mean",
        "val_case_adaptive_edge_count_mean",
        "case_adaptive_soft_edge_count_mean",
        "val_case_adaptive_soft_edge_count_mean",
        "residual_fraction_final_mean",
        "val_residual_fraction_final_mean",
        "case_adaptive_stop_reached_fraction",
        "val_case_adaptive_stop_reached_fraction",
        "case_adaptive_cap_hit_fraction",
        "val_case_adaptive_cap_hit_fraction",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: 1.0 for field in fields})

    save_global_loss_plots(metrics_path, run_dir)

    assert (run_dir / "plots" / "diagnostics" / "honf_organizer_health_curve.png").is_file()
