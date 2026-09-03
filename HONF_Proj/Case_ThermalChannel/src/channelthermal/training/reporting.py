"""Metrics CSV lifecycle and behavior-neutral training plots."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable

from honf_forward_core.training.diagnostics import HONF_DIAGNOSTIC_KEYS
from honf_runtime.compat import read_json


def write_metrics_row(path: Path, fieldnames: Iterable[str], row: Dict[str, Any]) -> None:
    """Write metrics row."""

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def repair_metrics_csv_for_append(path: Path) -> None:
    """Repair metrics csv for append."""

    if not path.exists():
        return
    raw = path.read_bytes()
    repaired = raw.replace(b"\x00", b"")
    if repaired and not repaired.endswith(b"\n"):
        repaired += b"\n"
    if repaired != raw:
        path.write_bytes(repaired)


def best_metrics_payload(row: Dict[str, Any], best_total: float, best_field: float, best_temperature: float, best_predicted: float) -> Dict[str, float]:
    """Perform the best metrics payload operation used by this module."""

    payload = {
        "best_val_loss_total": float(best_total),
        "best_val_field_mse": float(best_field),
        "best_val_temperature_mse": float(best_temperature),
        "best_val_predicted_loss_total": float(best_predicted),
    }
    for key in HONF_DIAGNOSTIC_KEYS:
        val_key = f"val_{key}"
        if val_key in row:
            payload[val_key] = float(row[val_key])
    return payload


def _read_metric_history(metrics_path: Path) -> Dict[str, list[float]]:
    """Read numeric metric columns from `metrics.csv` for compact plotting."""

    if not metrics_path.exists():
        return {}
    columns: Dict[str, list[float]] = {}
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(parsed):
                    columns.setdefault(key, []).append(parsed)
    # Read-only aliases keep plots usable for historical metrics.csv files.
    aliases = {
        "selected_edge_count": "active_edge_count",
        "functional_edge_count": "active_edge_count",
        "soft_functional_edge_count": "soft_active_edge_count",
    }
    for new_name, old_name in aliases.items():
        for prefix in ("", "val_"):
            new_key = prefix + new_name
            old_key = prefix + old_name
            if new_key not in columns and old_key in columns:
                columns[new_key] = list(columns[old_key])
    return columns


def _plot_metric_group(
    ax: Any,
    history: Dict[str, list[float]],
    keys: tuple[str, ...],
    *,
    title: str,
    ylabel: str = "value",
    log_scale: bool = True,
    y_min_zero: bool = False,
    reference_lines: tuple[tuple[float, str], ...] = (),
) -> None:
    """Perform the plot metric group operation used by this module."""

    epochs = history.get("epoch", [])
    for key in keys:
        values = history.get(key)
        if not values:
            continue
        label = key
        for prefix in ("val_", "loss_"):
            label = label.removeprefix(prefix)
        if key.startswith("val_"):
            label = f"val {label}"
        ax.plot(epochs[: len(values)], values, label=label)
    if epochs:
        for index, (reference_y, reference_label) in enumerate(reference_lines):
            ax.axhline(
                float(reference_y),
                color="black" if index == 0 else "#666666",
                linestyle="--" if index == 0 else ":",
                linewidth=1.0,
                alpha=0.60,
                label=reference_label,
            )
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    if y_min_zero:
        _, top = ax.get_ylim()
        reference_max = max((value for value, _ in reference_lines), default=0.0)
        ax.set_ylim(bottom=0.0, top=max(float(top), float(reference_max) * 1.10, 1.0))
    ax.grid(True, alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=8)


def _resolved_active_edge_references(run_dir: Path) -> tuple[tuple[float, str], ...]:
    """Return configured candidate/selection references for activity plots."""

    config_path = run_dir / "config_resolved.json"
    if not config_path.exists():
        return ()
    try:
        payload = read_json(config_path)
        core = payload.get("model", {}).get("core_honf", {})
        if core.get("organizer_mode", "fixed_projection") == "exchangeable_slots":
            references = []
            for key in ("edge_capacity", "initial_active_edges"):
                value = core.get(key)
                if value is not None:
                    references.append((float(value), f"{key}={int(value)}"))
            return tuple(references)
        value = core.get("num_hyperedges")
        return () if value is None else ((float(value), f"num_hyperedges={int(value)}"),)
    except (OSError, TypeError, ValueError):
        return ()


def _save_organizer_health_plot(history: Dict[str, list[float]], diagnostics_dir: Path) -> None:
    """Write one compact case-adaptive residual-organizer health figure.

    The figure is intentionally separate from the loss overview: its panels
    show the hard/soft mechanism support, residual progress, and stop/cap
    outcomes without duplicating any prediction or physical-loss curves.
    Historical fixed-organizer metrics do not contain these columns, so no
    empty/not-applicable image is emitted for those runs.
    """

    health_keys = (
        "case_adaptive_edge_count_mean",
        "case_adaptive_soft_edge_count_mean",
        "residual_fraction_final_mean",
        "case_adaptive_stop_reached_fraction",
        "case_adaptive_cap_hit_fraction",
    )
    if not any(
        any(abs(float(value)) > 0.0 for value in history.get(key, []))
        or any(abs(float(value)) > 0.0 for value in history.get(f"val_{key}", []))
        for key in health_keys
    ):
        return

    import matplotlib.pyplot as plt

    panels = [
        (
            "Hard mechanism count",
            ("case_adaptive_edge_count_mean", "val_case_adaptive_edge_count_mean"),
            "count",
            False,
            True,
        ),
        (
            "Soft effective mechanism count",
            ("case_adaptive_soft_edge_count_mean", "val_case_adaptive_soft_edge_count_mean"),
            "count",
            False,
            True,
        ),
        (
            "Final residual fraction",
            ("residual_fraction_final_mean", "val_residual_fraction_final_mean"),
            "fraction",
            True,
            False,
        ),
        (
            "Stop reached / cap hit",
            (
                "case_adaptive_stop_reached_fraction",
                "val_case_adaptive_stop_reached_fraction",
                "case_adaptive_cap_hit_fraction",
                "val_case_adaptive_cap_hit_fraction",
            ),
            "fraction",
            False,
            True,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.8), constrained_layout=True)
    for ax, (title, keys, ylabel, log_scale, y_min_zero) in zip(axes.reshape(-1), panels):
        _plot_metric_group(
            ax,
            history,
            keys,
            title=title,
            ylabel=ylabel,
            log_scale=log_scale,
            y_min_zero=y_min_zero,
        )
    fig.suptitle("Case-Adaptive Organizer Health", fontsize=13)
    fig.savefig(str(diagnostics_dir / "honf_organizer_health_curve.png"), dpi=160)
    plt.close(fig)


def save_global_loss_plots(metrics_path: Path, run_dir: Path) -> None:
    """Save readable grouped plots instead of one overcrowded metric figure."""

    history = _read_metric_history(metrics_path)
    if not history or not history.get("epoch"):
        return
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-honf_cl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Managed runs write directly to the canonical tree.  The legacy fallback
    # keeps direct case-workflow invocations compatible without mirroring one
    # plot into both ``diagnostic_plots`` and ``plots/diagnostics``.
    managed_run = (run_dir / "run_manifest.json").is_file()
    training_dir = run_dir / "plots" / "training" if managed_run else run_dir
    diagnostics_dir = run_dir / "plots" / "diagnostics" if managed_run else run_dir / "diagnostic_plots"
    training_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    active_edge_references = _resolved_active_edge_references(run_dir)
    stale_root_plots = [
        "loss_curves.png",
        "loss_total_curve.png",
        "loss_field_curve.png",
        "loss_local_coupling_curve.png",
        "loss_port_condition_curve.png",
        "honf_entropy_activity_curve.png",
        "honf_context_curve.png",
    ]
    for filename in stale_root_plots:
        stale_path = training_dir / filename
        if stale_path.exists():
            stale_path.unlink()

    panels = [
        ("Total", ("loss_total", "val_loss_total", "val_predicted_loss_total"), "loss", True, False),
        ("Global Field", ("loss_field", "val_loss_field", "field_mse", "val_field_mse"), "loss / mse", True, False),
        (
            "Local/Internal Coupling",
            ("loss_internal_temperature", "val_loss_internal_temperature", "loss_interface", "val_loss_interface"),
            "loss",
            True,
            False,
        ),
        (
            "Port and Consistency",
            ("loss_port_condition", "val_loss_port_condition", "loss_port_global_consistency", "val_loss_port_global_consistency"),
            "loss",
            True,
            False,
        ),
        ("Temperature", ("temperature_mse", "val_temperature_mse"), "mse", True, False),
        ("Selected and Functional H-Edge Activity", ("selected_edge_count", "val_selected_edge_count", "functional_edge_count", "val_functional_edge_count", "soft_functional_edge_count", "val_soft_functional_edge_count"), "edges", False, True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.4), constrained_layout=True)
    for ax, (title, keys, ylabel, log_scale, y_min_zero) in zip(axes.reshape(-1), panels):
        _plot_metric_group(
            ax,
            history,
            keys,
            title=title,
            ylabel=ylabel,
            log_scale=log_scale,
            y_min_zero=y_min_zero,
            reference_lines=active_edge_references if y_min_zero else (),
        )
    fig.suptitle("HONF-CL Training Overview", fontsize=13)
    fig.savefig(str(training_dir / "loss_curve.png"), dpi=160)
    plt.close(fig)

    focused = {
        "loss_total_curve.png": ("Total Loss", ("loss_total", "val_loss_total", "val_predicted_loss_total"), "loss", True, False),
        "loss_field_curve.png": ("Field Loss and MSE", ("loss_field", "val_loss_field", "field_mse", "val_field_mse"), "loss / mse", True, False),
        "loss_local_coupling_curve.png": (
            "Internal Temperature and Interface Loss",
            ("loss_internal_temperature", "val_loss_internal_temperature", "loss_interface", "val_loss_interface"),
            "loss",
            True,
            False,
        ),
        "loss_port_condition_curve.png": (
            "Port and Consistency Loss",
            (
                "loss_port_condition",
                "val_loss_port_condition",
                "loss_port_global_consistency",
                "val_loss_port_global_consistency",
                "loss_predicted_consistency",
                "val_loss_predicted_consistency",
            ),
            "loss",
            True,
            False,
        ),
        "honf_entropy_activity_curve.png": (
            "HONF Entropy and Selected H-Edge Count",
            (
                "A_mh_entropy",
                "val_A_mh_entropy",
                "A_eh_entropy",
                "val_A_eh_entropy",
                "selected_edge_count",
                "val_selected_edge_count",
                "functional_edge_count",
                "val_functional_edge_count",
            ),
            "value",
            False,
            True,
        ),
        "honf_candidate_mass_purity_curve.png": (
            "Candidate Mass Fractions and Purities",
            (
                "candidate_module_mass_fraction_min",
                "val_candidate_module_mass_fraction_min",
                "candidate_environment_mass_fraction_min",
                "val_candidate_environment_mass_fraction_min",
                "candidate_module_purity_mean",
                "val_candidate_module_purity_mean",
                "candidate_environment_purity_mean",
                "val_candidate_environment_purity_mean",
            ),
            "fraction",
            False,
            True,
        ),
        "honf_candidate_scale_curve.png": (
            "Candidate Source and Region Scales",
            (
                "candidate_source_scale_mean",
                "val_candidate_source_scale_mean",
                "candidate_region_scale_mean",
                "val_candidate_region_scale_mean",
            ),
            "scale",
            False,
            True,
        ),
        "honf_edge_contribution_fraction_curve.png": (
            "Additive Edge Contribution Fractions",
            (
                "edge_contribution_fraction_min",
                "val_edge_contribution_fraction_min",
                "edge_contribution_fraction_max",
                "val_edge_contribution_fraction_max",
            ),
            "fraction",
            False,
            True,
        ),
        "honf_context_curve.png": (
            "HONF Context Norms and Pairwise Gate",
            (
                "pairwise_kernel_gate",
                "val_pairwise_kernel_gate",
                "pairwise_context_norm",
                "val_pairwise_context_norm",
                "total_hyper_context_norm",
                "val_total_hyper_context_norm",
                "nonhyper_context_norm",
                "val_nonhyper_context_norm",
            ),
            "value",
            True,
            False,
        ),
    }
    for filename, (title, keys, ylabel, log_scale, y_min_zero) in focused.items():
        fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
        _plot_metric_group(
            ax,
            history,
            keys,
            title=title,
            ylabel=ylabel,
            log_scale=log_scale,
            y_min_zero=y_min_zero,
            reference_lines=active_edge_references if y_min_zero else (),
        )
        fig.savefig(str(diagnostics_dir / filename), dpi=160)
        plt.close(fig)

    _save_organizer_health_plot(history, diagnostics_dir)
