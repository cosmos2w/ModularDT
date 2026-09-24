"""Rebuild the Run 1404/1804/1501/1502 accuracy comparison from saved evals.

No model inference is performed. Inputs are the September 20 baseline evaluation
and the fresh 90-case Run 1501/1502 evaluation under this report directory.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "diagnostics/generated/run1404_1406_1407_1804_best5000_accuracy_20260920"
NEW = OUT / "evaluation"
OLD_TABLES = OLD / "tables"
NEW_TABLES = NEW / "tables"
SEED = 20260923
BOOTSTRAPS = 10_000

MODELS = {
    "Run1404": {"label": "Run1404_best_field", "source": OLD_TABLES, "run_glob": "Run_1404_*", "checkpoint": "best_by_field_mse_model.pt"},
    "Run1804": {"label": "Run1804_best_field", "source": OLD_TABLES, "run_glob": "Run_1804_*", "checkpoint": "best_by_field_mse_model.pt"},
    "Run1501": {"label": "Run1501_best_field_e4689", "source": NEW_TABLES, "run_glob": "Run_1501_*", "checkpoint": "best_by_field_mse_model.pt"},
    "Run1502": {"label": "Run1502_best_field_e4794", "source": NEW_TABLES, "run_glob": "Run_1502_*", "checkpoint": "best_by_field_mse_model.pt"},
}
EXPLORATORY = {"Run1501": "Run1501_best_total_e4877"}

METRICS = {
    "fluid_global": ("global_field_fluid_norm", "global_field_fluid_norm_l2"),
    "fluid_near_interface": ("global_field_near_interface_norm", "global_field_near_interface_norm_l2"),
    "fluid_far": ("global_field_far_fluid_norm", "global_field_far_fluid_norm_l2"),
    "u_fluid": ("field_u_fluid_physical", "field_u_fluid_physical_relative_l2"),
    "v_fluid": ("field_v_fluid_physical", "field_v_fluid_physical_relative_l2"),
    "pressure_fluid": ("field_p_fluid_physical", "field_p_fluid_physical_relative_l2"),
    "vorticity_fluid": ("field_omega_fluid_physical", "field_omega_fluid_physical_relative_l2"),
    "temperature_fluid": ("field_temperature_fluid_physical", "field_temperature_fluid_physical_relative_l2"),
    "temperature_internal": ("internal_temperature_physical", "internal_temperature_physical_relative_l2"),
    "temperature_surface": ("interface_t_surface_physical", "interface_t_surface_physical_relative_l2"),
    "heat_flux_interface": ("interface_q_normal_physical", "interface_q_normal_physical_relative_l2"),
    "port_temperature_final": ("port_t_env_final_physical", "port_t_env_final_physical_relative_l2"),
    "port_effective_h_final": ("port_h_effective_final_physical", "port_h_effective_final_physical_relative_l2"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict, key: str) -> float:
    value = row.get(key, "")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def run_dir_for(pattern: str) -> Path:
    paths = sorted((ROOT / "Trained_Results/ThermalChannel/HONF_Forward_Runs").glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one run directory for {pattern}, got {paths}")
    return paths[0]


def load_checkpoint_meta(path: Path) -> dict:
    import torch
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    return obj


def find_architecture(value) -> str | None:
    if isinstance(value, dict):
        if "forward_architecture" in value:
            return str(value["forward_architecture"])
        for nested in value.values():
            found = find_architecture(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = find_architecture(nested)
            if found:
                return found
    return None


def candidate_checkpoints() -> list[dict]:
    candidates = []
    variants = [
        ("best_total", "best_model.pt", "validation_total_loss"),
        ("best_field", "best_by_field_mse_model.pt", "validation_field_mse"),
        ("best_temperature", "best_by_temperature_mse_model.pt", "validation_temperature_mse"),
        ("best_predicted", "best_predicted_model.pt", "validation_predicted_total_loss"),
    ]
    for run, spec in MODELS.items():
        rdir = run_dir_for(spec["run_glob"])
        summary = json.loads((rdir / "summary.json").read_text())
        for variant, filename, selection_metric in variants:
            ckpt = rdir / filename
            if not ckpt.exists():
                continue
            meta = load_checkpoint_meta(ckpt)
            candidates.append({
                "run": run,
                "architecture": find_architecture(meta.get("model_config", {})) or ("legacy_honf (default config)" if run == "Run1404" else "unresolved"),
                "variant": variant,
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "epoch": meta.get("epoch", meta.get("current_epoch", "")),
                "selection_metric": selection_metric,
                "saved_best_metric": meta.get("best_metric", ""),
                "run_summary_best_val_total": summary.get("best_val_loss_total", ""),
                "run_summary_best_val_field_mse": summary.get("best_val_field_mse", ""),
                "run_summary_best_val_temperature_mse": summary.get("best_val_temperature_mse", ""),
                "run_summary_best_val_predicted_total": summary.get("best_val_predicted_loss_total", ""),
                "selected_for_headline": variant == "best_field",
                "overlapping_test_validation_caveat": "90-case test evaluation is treated as an established held-out/development comparison set; it overlaps the validation/development split used in project checkpoint selection and is not an independent final test.",
            })
    return candidates


def load_cases() -> tuple[dict[str, list[dict]], list[str]]:
    all_cases = {}
    for run, spec in MODELS.items():
        rows = read_csv(spec["source"] / "per_case_metrics.csv")
        selected = [r for r in rows if r.get("model_label") == spec["label"]]
        if len(selected) != 90:
            raise RuntimeError(f"{run}: expected 90 cases for {spec['label']}, got {len(selected)}")
        all_cases[run] = selected
    extra_label = EXPLORATORY["Run1501"]
    extra_rows = read_csv(NEW_TABLES / "per_case_metrics.csv")
    extra = [r for r in extra_rows if r.get("model_label") == extra_label]
    if len(extra) != 90:
        raise RuntimeError(f"Expected 90 exploratory rows for {extra_label}, got {len(extra)}")
    all_cases["Run1501_best_total_exploratory"] = extra
    reference = [r["case_id"] for r in all_cases["Run1404"]]
    if len(set(reference)) != 90:
        raise RuntimeError("Run1404 case IDs are not unique")
    for run, rows in all_cases.items():
        ids = [r["case_id"] for r in rows]
        if ids != reference:
            raise RuntimeError(f"Case IDs/order mismatch for {run}")
    return all_cases, reference


def summarize_metric(rows: list[dict], stem: str, case_column: str) -> dict:
    sse = np.asarray([f(r, stem + "_sse") for r in rows], dtype=float)
    target = np.asarray([f(r, stem + "_target_sse") for r in rows], dtype=float)
    counts = np.asarray([f(r, stem + "_num_values") for r in rows], dtype=float)
    cases = np.asarray([f(r, case_column) for r in rows], dtype=float)
    valid = np.isfinite(sse) & np.isfinite(target) & np.isfinite(counts) & np.isfinite(cases)
    if not valid.all() or (target <= 0).any():
        raise RuntimeError(f"Invalid metric values in {stem}")
    return {
        "pooled_relative_l2": float(np.sqrt(sse.sum() / target.sum())),
        "case_mean": float(cases.mean()),
        "case_median": float(np.median(cases)),
        "case_p95": float(np.quantile(cases, 0.95)),
        "pooled_sse": float(sse.sum()),
        "pooled_target_sse": float(target.sum()),
        "pooled_num_values": round(float(counts.sum())),
    }


def build_accuracy_outputs(all_cases: dict, case_ids: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    selected_runs = ["Run1404", "Run1804", "Run1501", "Run1502"]
    summary_rows = []
    for run in selected_runs:
        rows = all_cases[run]
        out = {"run": run, "model_label": MODELS[run]["label"], "num_cases": len(rows), "checkpoint": str((run_dir_for(MODELS[run]["run_glob"]) / MODELS[run]["checkpoint"]).relative_to(ROOT))}
        ckpt = load_checkpoint_meta(run_dir_for(MODELS[run]["run_glob"]) / MODELS[run]["checkpoint"])
        out["checkpoint_epoch"] = ckpt.get("epoch", "")
        for label, (stem, case_column) in METRICS.items():
            summary = summarize_metric(rows, stem, case_column)
            for key, value in summary.items():
                out[f"{label}_{key}"] = value
        summary_rows.append(out)
    case_metric_cols = ["global_field_fluid_norm_l2", "global_field_fluid_norm_sse", "global_field_fluid_norm_target_sse", "global_field_fluid_norm_num_values", "global_field_near_interface_norm_l2", "global_field_far_fluid_norm_l2"]
    for stem, case_column in METRICS.values():
        case_metric_cols.extend([case_column, stem + "_sse", stem + "_target_sse", stem + "_num_values"])
    case_metric_cols = list(dict.fromkeys(case_metric_cols))
    per_case = []
    for run, rows in all_cases.items():
        for row in rows:
            out = {"run": run, "model_label": row.get("model_label"), "case_id": row.get("case_id"), "split": row.get("split"), "dataset_index": row.get("dataset_index"), "case_order": row.get("case_order")}
            out.update({key: row.get(key, "") for key in case_metric_cols})
            per_case.append(out)
    wins = []
    for i, left in enumerate(selected_runs):
        for right in selected_runs[i + 1:]:
            a = np.asarray([f(r, "global_field_fluid_norm_l2") for r in all_cases[left]])
            b = np.asarray([f(r, "global_field_fluid_norm_l2") for r in all_cases[right]])
            wins.append({"candidate": left, "baseline": right, "metric": "global_field_fluid_norm_l2", "num_cases": len(a), "candidate_wins_lower": int((a < b).sum()), "baseline_wins_lower": int((b < a).sum()), "ties": int((a == b).sum()), "mean_case_delta_candidate_minus_baseline": float((a - b).mean()), "median_case_delta_candidate_minus_baseline": float(np.median(a - b))})
    return summary_rows, per_case, wins


def bootstrap_outputs(all_cases: dict) -> list[dict]:
    rng = np.random.default_rng(SEED)
    pairs = [("Run1502", "Run1501"), ("Run1501", "Run1804"), ("Run1502", "Run1804")]
    results = []
    n = 90
    idx = rng.integers(0, n, size=(BOOTSTRAPS, n))
    for candidate, baseline in pairs:
        ca, ba = all_cases[candidate], all_cases[baseline]
        cv = np.asarray([f(r, "global_field_fluid_norm_l2") for r in ca])
        bv = np.asarray([f(r, "global_field_fluid_norm_l2") for r in ba])
        cs = np.asarray([f(r, "global_field_fluid_norm_sse") for r in ca]); ct = np.asarray([f(r, "global_field_fluid_norm_target_sse") for r in ca])
        bs = np.asarray([f(r, "global_field_fluid_norm_sse") for r in ba]); bt = np.asarray([f(r, "global_field_fluid_norm_target_sse") for r in ba])
        case_delta = cv - bv
        pooled_delta = np.sqrt(cs[idx].sum(axis=1) / ct[idx].sum(axis=1)) - np.sqrt(bs[idx].sum(axis=1) / bt[idx].sum(axis=1))
        case_boot = case_delta[idx].mean(axis=1)
        for metric, point, dist in [
            ("case_mean_fluid_relative_l2", float(case_delta.mean()), case_boot),
            ("pooled_fluid_relative_l2", float(np.sqrt(cs.sum() / ct.sum()) - np.sqrt(bs.sum() / bt.sum())), pooled_delta),
        ]:
            lo, hi = np.quantile(dist, [0.025, 0.975])
            results.append({"candidate": candidate, "baseline": baseline, "metric": metric, "difference_candidate_minus_baseline": point, "ci95_low": float(lo), "ci95_high": float(hi), "bootstrap_replicates": BOOTSTRAPS, "seed": SEED, "resampling_unit": "paired case_id"})
    return results


def target_reconciliation(all_cases: dict) -> list[dict]:
    runs = ["Run1404", "Run1804", "Run1501", "Run1502"]
    rows = []
    for metric, (stem, _) in METRICS.items():
        stats = {}
        for run in runs:
            rs = all_cases[run]
            stats[run] = {
                "target_num_values": round(sum(f(r, stem + "_num_values") for r in rs)),
                "target_sse": float(sum(f(r, stem + "_target_sse") for r in rs)),
                "prediction_sse": float(sum(f(r, stem + "_sse") for r in rs)),
            }
        ref = stats["Run1404"]
        for run in runs:
            rows.append({"metric": metric, "run": run, **stats[run], "count_delta_vs_1404": stats[run]["target_num_values"] - ref["target_num_values"], "target_sse_delta_vs_1404": stats[run]["target_sse"] - ref["target_sse"], "target_matches_1404_exactly": stats[run]["target_num_values"] == ref["target_num_values"] and stats[run]["target_sse"] == ref["target_sse"]})
    return rows


def query_reconciliation() -> list[dict]:
    sources = {
        "Run1404": OLD_TABLES / "evaluation_cost_case_metrics.csv",
        "Run1804": OLD_TABLES / "evaluation_cost_case_metrics.csv",
        "Run1501": NEW_TABLES / "evaluation_cost_case_metrics.csv",
        "Run1502": NEW_TABLES / "evaluation_cost_case_metrics.csv",
    }
    labels = {run: spec["label"] for run, spec in MODELS.items()}
    rows_out = []
    for run, path in sources.items():
        rows = [r for r in read_csv(path) if r.get("model_label") == labels[run]]
        counts = [int(float(r["evaluation_grid_query_count"])) for r in rows]
        if len(rows) != 90 or set(counts) != {8192}:
            raise RuntimeError(f"Query counts unexpected for {run}: {len(rows)}, {set(counts)}")
        rows_out.append({"run": run, "num_cases": len(rows), "queries_per_case": 8192, "total_grid_queries": sum(counts), "all_case_ids_match": True})
    return rows_out


def endpoint_context() -> list[dict]:
    rows_out = []
    for run, spec in MODELS.items():
        rdir = run_dir_for(spec["run_glob"])
        source = rdir / "metrics.csv"
        endpoint = None
        if run != "Run1804":
            try:
                matches = [r for r in read_csv(source) if int(float(r.get("epoch", -1))) == 5000]
                if matches:
                    endpoint = matches[-1]
            except (ValueError, TypeError):
                endpoint = None
        else:
            logs = list(rdir.glob("logs/resume_to_5000_*.log")) + list(rdir.glob("resume_to_5000_*.log"))
            if logs:
                text = logs[-1].read_text(errors="replace")
                lines = [line for line in text.splitlines() if "epoch 5000" in line.lower()]
                if lines:
                    line = lines[-1]
                    vals = {}
                    for key, value in re.findall(r"(val_field|val_temp|val_pred)=([0-9.eE+-]+)", line):
                        vals[key] = value
                    endpoint = {"epoch": "5000", "val_field_mse": vals.get("val_field"), "val_temperature_mse": vals.get("val_temp"), "val_predicted_loss_total": vals.get("val_pred")}
                    source = logs[-1]
        if endpoint is None:
            rows_out.append({"run": run, "epoch": 5000, "source": str(source.relative_to(ROOT)), "status": "unavailable or non-finite"})
        else:
            rows_out.append({"run": run, "epoch": 5000, "source": str(source.relative_to(ROOT)), "val_loss_total": endpoint.get("val_loss_total"), "val_field_mse": endpoint.get("val_field_mse"), "val_temperature_mse": endpoint.get("val_temperature_mse"), "val_predicted_loss_total": endpoint.get("val_predicted_loss_total"), "status": "context only; not headline"})
    return rows_out


def make_figure(summary_rows: list[dict]) -> None:
    names = [r["run"] for r in summary_rows]
    colors = ["#6b7280", "#2563eb", "#e67e22", "#0f766e"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    ax = axes[0]
    pooled = [r["fluid_global_pooled_relative_l2"] for r in summary_rows]
    means = [r["fluid_global_case_mean"] for r in summary_rows]
    x = np.arange(len(names)); width = 0.35
    ax.bar(x - width / 2, pooled, width, color=colors, alpha=0.95, label="Pooled full-grid")
    ax.bar(x + width / 2, means, width, color=colors, alpha=0.35, edgecolor=colors, linewidth=1.0, label="Mean of 90 case L2")
    ax.set_xticks(x, names, rotation=0)
    ax.set_ylabel("Fluid relative L2 (lower is better)")
    ax.set_title("Primary fluid accuracy")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    channel_names = ["u", "v", "p", "vorticity", "fluid T", "internal T", "surface T", "heat flux", "port T", "effective h"]
    keys = ["u_fluid", "v_fluid", "pressure_fluid", "vorticity_fluid", "temperature_fluid", "temperature_internal", "temperature_surface", "heat_flux_interface", "port_temperature_final", "port_effective_h_final"]
    xx = np.arange(len(keys))
    for run_i, row in enumerate(summary_rows):
        vals = [row[f"{key}_pooled_relative_l2"] for key in keys]
        ax.plot(xx, vals, marker="o", linewidth=1.5, markersize=3.5, color=colors[run_i], label=row["run"])
    ax.set_xticks(xx, channel_names, rotation=40, ha="right")
    ax.set_ylabel("Pooled relative L2")
    ax.set_title("Fields and thermal/interface outputs")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Best validation-field checkpoints · 90-case evaluation", fontsize=13)
    fig.savefig(OUT / "accuracy_comparison.png", dpi=180)
    fig.savefig(OUT / "accuracy_comparison.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_cases, case_ids = load_cases()
    summary_rows, per_case, wins = build_accuracy_outputs(all_cases, case_ids)
    candidates = candidate_checkpoints()
    bootstrap = bootstrap_outputs(all_cases)
    targets = target_reconciliation(all_cases)
    queries = query_reconciliation()
    endpoints = endpoint_context()
    write_csv(OUT / "accuracy_summary.csv", summary_rows)
    write_csv(OUT / "per_case_accuracy.csv", per_case)
    write_csv(OUT / "paired_case_wins.csv", wins)
    write_csv(OUT / "paired_bootstrap_ci.csv", bootstrap)
    write_csv(OUT / "target_count_reconciliation.csv", targets)
    write_csv(OUT / "query_count_reconciliation.csv", queries)
    write_csv(OUT / "checkpoint_selection.csv", candidates)
    write_csv(OUT / "validation_endpoint_context.csv", endpoints)
    make_figure(summary_rows)
    manifest = {
        "schema_version": 1,
        "analysis": "90-case accuracy comparison; best validation-field checkpoint per run is primary",
        "evaluation_case_ids": case_ids,
        "n_cases": len(case_ids),
        "split": "test label in evaluation artifacts; overlaps project validation/development split and is not independent final test evidence",
        "checkpoint_policy": "best_by_field_mse_model.pt selected using sampled logged validation field MSE before evaluating this 90-case set",
        "exploratory_sensitivity": "Run1501 best_model.pt / best-total e4877 is included in per_case_accuracy.csv only; it does not replace the preselected headline checkpoint",
        "bootstrap": {"replicates": BOOTSTRAPS, "seed": SEED, "unit": "paired case_id"},
        "sources": {"baseline_eval": str(OLD.relative_to(ROOT)), "new_eval": str(NEW.relative_to(ROOT))},
        "target_counts_match": all(r["target_matches_1404_exactly"] for r in targets),
        "query_counts_match": all(r["total_grid_queries"] == 737280 for r in queries),
    }
    (OUT / "comparison.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "pooled_fluid_relative_l2": {r["run"]: r["fluid_global_pooled_relative_l2"] for r in summary_rows},
        "paired_wins": wins,
        "bootstrap": bootstrap,
        "target_counts_match": manifest["target_counts_match"],
        "query_counts_match": manifest["query_counts_match"],
    }, indent=2))


if __name__ == "__main__":
    main()
