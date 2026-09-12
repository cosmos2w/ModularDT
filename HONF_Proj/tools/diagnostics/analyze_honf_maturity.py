#!/usr/bin/env python3
"""Reduce the existing five-model, three-endpoint development comparison.

Reads evaluator tables in place; writes computed summaries and paired deltas.
No model execution, training, or replacement of historical evaluation tables.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
STUDIES = PROJECT / "diagnostics/generated/interface_operator_study"
MODELS = {"1401": "Legacy", "1801": "Latent", "1804": "Dense", "1805": "Reader", "1806": "Regional"}
EPOCHS = (500, 2500, 5000)
STRATA = ("module_count_stratum", "spacing_stratum", "wall_proximity_stratum", "heating_heterogeneity_stratum")
CHANNELS = ("u", "v", "p", "omega", "temperature")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def source_directory(root, run, epoch):
    if epoch == 500:
        if run == "1805":
            return STUDIES / "group_reader_recovery/endpoint/tables"
        if run == "1806":
            return STUDIES / "regional_response/endpoint500/tables"
        return PROJECT / "Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage3_Run1401_1804_1801_1802_Epoch500_90Case/tables"
    if run in ("1801", "1806"):
        return root / "evaluation/tables"
    return STUDIES / f"epoch{epoch}_comparison/evaluation/tables"


def selected(rows, run, epoch):
    return [row for row in rows if f"Run_{run}_" in row.get("checkpoint", "") and f"epoch_{epoch:04d}_model.pt" in row.get("checkpoint", "")]


def pooled(rows, base):
    triples = [[number(row.get(base + suffix)) for suffix in ("_sse", "_target_sse", "_num_values")] for row in rows]
    if any(value is None for triple in triples for value in triple):
        return None
    sse, target, count = np.asarray(triples).sum(axis=0)
    if target <= 0 or count <= 0:
        return None
    return {"sse": float(sse), "target_sse": float(target), "num_values": int(count), "mse": float(sse / count), "relative_l2": float(np.sqrt(sse / target))}


def geometry_summaries(root):
    """Describe actual routing exports, without inventing topology targets."""
    rows, attention_rows = [], []
    for run in MODELS:
        directory = source_directory(root, run, 5000)
        for table in ("interaction_case_metrics", "hypergraph_case_metrics"):
            selected_rows = selected(read_csv(directory / (table + ".csv")), run, 5000)
            if len(selected_rows) != 90:
                raise ValueError(f"Expected 90 organization rows: {run}/{table}")
            for key in selected_rows[0]:
                if not any(part in key for part in ("fraction_mean", "entropy", "support_", "regional_", "module_affinity", "shape_mismatch", "active_edge_count")):
                    continue
                values = [number(row[key]) for row in selected_rows]
                values = [value for value in values if value is not None]
                if values:
                    rows.append({"run": run, "model": MODELS[run], "metric": key, "table": table, "n": len(values), "equal_case_mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))})
        debug = directory.parent / "debug_npz"
        for case in ("0273", "0653", "0298", "0302"):
            files = [p for p in debug.glob(f"*{run}*5000*{case}.npz")]
            if len(files) != 1:
                raise ValueError(f"Expected one anchor NPZ: {run}/{case}, found {files}")
            with np.load(files[0], allow_pickle=False) as arrays:
                for key, role in (("latent_query_attention", "P2_field"), ("regional_environment_attention", "P2_field"), ("interaction__initial_port_latent_query_attention", "P0_ports"), ("interaction__initial_port_regional_environment_attention", "P0_ports")):
                    if key not in arrays:
                        continue
                    weights = np.asarray(arrays[key], dtype=np.float64)
                    if weights.ndim == 3:
                        weights = weights.mean(axis=0)
                    if weights.ndim != 2:
                        raise ValueError(f"Unexpected attention shape: {files[0]}, {key}, {weights.shape}")
                    if role == "P0_ports":
                        present = np.asarray(arrays["module_present"], dtype=bool).reshape(-1)
                        if weights.shape[0] % len(present):
                            raise ValueError("Port attention cannot align with module-presence mask")
                        weights = weights[np.repeat(present, weights.shape[0] // len(present))]
                    elif "fluid_mask" in arrays:
                        weights = weights[np.asarray(arrays["fluid_mask"], dtype=bool).reshape(-1)]
                    mass = weights.sum(axis=1)
                    conditional = np.divide(weights, mass[:, None], out=np.zeros_like(weights), where=mass[:, None] > 0)
                    entropy = -np.sum(conditional * np.log(np.maximum(conditional, np.finfo(float).tiny)), axis=1)
                    attention_rows.append({"run": run, "model": MODELS[run], "case_id": case, "role": role, "sources": weights.shape[1], "receivers": weights.shape[0], "row_mass_mean": float(mass.mean()), "entropy_over_log_sources_mean": float((entropy / np.log(weights.shape[1])).mean()), "exp_entropy_mean": float(np.exp(entropy).mean()), "maximum_weight_mean": float(conditional.max(axis=1).mean()), "source": str(files[0]), "key": key})
    output = root / "geometry"
    write_csv(output / "organization_summary.csv", rows)
    write_csv(output / "attention_summary.csv", attention_rows)
    (output / "organization.json").write_text(json.dumps({"organization": rows, "attention": attention_rows, "limitations": ["Attention concentration and context norms are descriptions, not causal contribution or physical organization accuracy.", "P0 port rows exclude padded modules; P2 rows use the stored fluid mask when available.", "Main context in Dense and Regional includes direct module plus environmental/regional reads.", "Legacy historical affinity targets and active-edge targets have different semantics from newer families; no cross-family topology accuracy is inferred.", "Regional membership is deterministic; learned states and receiver attention differ from membership."]}, indent=2) + "\n")


def best_selected_summaries(root, reference, target_keys):
    """Keep validation-selected checkpoints separate from exact endpoints."""
    directory = root / "best_field_evaluation/tables"
    if not directory.exists():
        return
    from honf_runtime.compat import load_trusted_checkpoint

    all_rows = read_csv(directory / "per_case_metrics.csv")
    summaries = read_csv(directory / "model_summary_metrics.csv")
    headline, pools, cases = [], [], {}
    checks = 0
    for run, model in MODELS.items():
        rows = [row for row in all_rows if f"Run_{run}_" in row["checkpoint"]]
        summary = [row for row in summaries if f"Run_{run}_" in row["checkpoint"]]
        if len(rows) != 90 or {row["case_id"] for row in rows} != set(reference) or len(summary) != 1:
            raise ValueError(f"Best-selected case set mismatch: {run}")
        for row in rows:
            for key in target_keys:
                actual, expected = number(row.get(key)), number(reference[row["case_id"]].get(key))
                checks += 1
                if actual is None or expected is None or not np.isclose(actual, expected, rtol=1e-7, atol=1e-7):
                    raise ValueError(f"Best-selected target mismatch: {run}/{row['case_id']}/{key}")
        checkpoint_path = Path(summary[0]["checkpoint"])
        if checkpoint_path.name != "best_by_field_mse_model.pt":
            raise ValueError(f"Unexpected selection policy: {checkpoint_path}")
        checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
        actual_epoch = int(checkpoint["epoch"])
        if not 1 <= actual_epoch <= 5000:
            raise ValueError(f"Best checkpoint is outside the studied training budget: {actual_epoch}")
        identity = {"run": run, "model": model, "checkpoint_epoch": actual_epoch,
                    "selection": "best_logged_validation_field_mse_through5000", "checkpoint": str(checkpoint_path)}
        errors = np.asarray([float(row["global_field_fluid_norm_l2"]) for row in rows])
        result = pooled(rows, "global_field_fluid_norm")
        if not np.isclose(result["relative_l2"], float(summary[0]["global_field_fluid_pooled_relative_l2"]), rtol=1e-10, atol=1e-11):
            raise ValueError(f"Best-selected pooled mismatch: {run}")
        headline.append({**identity, **result, "equal_case_mean": float(errors.mean()),
                         "median": float(np.median(errors)), "p95": float(np.quantile(errors, .95)), "max": float(errors.max())})
        for key in rows[0]:
            if key.endswith("_sse") and not key.endswith("_target_sse") and key[:-4] + "_target_sse" in rows[0]:
                if (result := pooled(rows, key[:-4])) is not None:
                    pools.append({**identity, "base": key[:-4], **result})
        cases[run] = {row["case_id"]: row for row in rows}
        del checkpoint
    pairs = []
    for baseline, candidate in itertools.combinations(MODELS, 2):
        delta = np.asarray([float(cases[candidate][case]["global_field_fluid_norm_l2"]) - float(cases[baseline][case]["global_field_fluid_norm_l2"]) for case in sorted(reference)])
        pairs.append({"candidate": candidate, "baseline": baseline, "n": len(delta),
                      "wins": int((delta < 0).sum()), "losses": int((delta > 0).sum()), "mean_delta": float(delta.mean())})
    output = root / "reduction"
    write_csv(output / "best_selected_headline.csv", headline)
    write_csv(output / "best_selected_pooled.csv", pools)
    write_csv(output / "best_selected_pairs.csv", pairs)
    (output / "best_selected.json").write_text(json.dumps({"target_count_checks": checks, "headline": headline,
        "paired": pairs, "limitation": "Validation-selected on the same development split; descriptive selection sensitivity, not independent testing."}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDIES / "five_model_epoch5000")
    args = parser.parse_args()
    root = args.study_root.resolve()
    out = root / "reduction"
    cases, summaries, sources, cache = {}, {}, [], {}
    for epoch, run in itertools.product(EPOCHS, MODELS):
        source = source_directory(root, run, epoch)
        if source not in cache:
            cache[source] = {name: read_csv(source / (name + ".csv")) for name in ("per_case_metrics", "model_summary_metrics")}
        rows = selected(cache[source]["per_case_metrics"], run, epoch)
        summary = selected(cache[source]["model_summary_metrics"], run, epoch)
        if len(rows) != 90 or len({row["case_id"] for row in rows}) != 90 or len(summary) != 1:
            raise ValueError(f"Expected 90 unique cases and one summary: {run}@{epoch}")
        cases[epoch, run] = rows
        summaries[epoch, run] = summary[0]
        sources.append({"run": run, "epoch": epoch, "tables": str(source), "checkpoint": summary[0]["checkpoint"]})

    reference = {row["case_id"]: row for row in cases[500, "1401"]}
    common = set.intersection(*(set(rows[0]) for rows in cases.values()))
    target_keys = sorted(key for key in common if key.endswith(("_target_sse", "_num_values")))
    target_checks = 0
    missing_target_pairs = 0
    for (epoch, run), rows in cases.items():
        if {row["case_id"] for row in rows} != set(reference):
            raise ValueError(f"Case sets differ: {run}@{epoch}")
        for row in rows:
            baseline = reference[row["case_id"]]
            for key in target_keys:
                actual, expected = number(row.get(key)), number(baseline.get(key))
                target_checks += 1
                if actual is None and expected is None:
                    missing_target_pairs += 1
                elif actual is None or expected is None or not np.isclose(actual, expected, rtol=1e-7, atol=1e-7):
                    raise ValueError(f"Target/count mismatch: {run}@{epoch}, {row['case_id']}, {key}")
            for key in STRATA:
                if row[key] != baseline[key]:
                    raise ValueError(f"Stratum mismatch: {run}@{epoch}, {row['case_id']}, {key}")

    pool_bases = sorted(key[:-4] for key in common if key.endswith("_sse") and not key.endswith("_target_sse") and key[:-4] + "_target_sse" in common)
    pools, metrics, headline, strata, worst = [], [], [], [], []
    reconciliations = 0
    for (epoch, run), rows in cases.items():
        identity = {"epoch": epoch, "run": run, "model": MODELS[run]}
        summary = summaries[epoch, run]
        for key, value in summary.items():
            if (converted := number(value)) is not None and key not in ("model_index", "num_cases"):
                metrics.append({**identity, "metric": key, "value": converted})
        for base in pool_bases:
            result = pooled(rows, base)
            if result is None:
                continue
            pools.append({**identity, "base": base, **result})
            summary_base = base.removesuffix("_norm") if base.startswith("global_field_") else base
            for suffix, computed in (("_pooled_relative_l2", result["relative_l2"]), ("_pooled_mse", result["mse"])):
                recorded = number(summary.get(summary_base + suffix))
                if recorded is not None:
                    reconciliations += 1
                    if not np.isclose(recorded, computed, rtol=1e-10, atol=1e-11):
                        raise ValueError(f"Pooled mismatch: {run}@{epoch}, {summary_base + suffix}")
        errors = np.asarray([float(row["global_field_fluid_norm_l2"]) for row in rows])
        headline.append({**identity, **pooled(rows, "global_field_fluid_norm"), "equal_case_mean": float(errors.mean()), "median": float(np.median(errors)), "p95": float(np.quantile(errors, .95)), "max": float(errors.max())})
        for axis in STRATA:
            for label in sorted({row[axis] for row in rows}):
                subset = [row for row in rows if row[axis] == label]
                for base in ("global_field_fluid_norm", "global_field_near_interface_norm", "global_field_far_fluid_norm", "internal_temperature_physical", "interface_q_normal_physical"):
                    result = pooled(subset, base)
                    if result:
                        strata.append({**identity, "axis": axis, "stratum": label, "num_cases": len(subset), "base": base, **result})
        for rank, row in enumerate(sorted(rows, key=lambda row: float(row["global_field_fluid_norm_l2"]), reverse=True)[:10], 1):
            worst.append({**identity, "rank": rank, "case_id": row["case_id"], "fluid_l2": float(row["global_field_fluid_norm_l2"]), **{key: row[key] for key in STRATA}})

    pair_keys = sorted(key for key in common if key.endswith(("_relative_l2", "_norm_l2", "_physical_abs_error")))
    comparisons = [(epoch, candidate, epoch, baseline) for epoch in EPOCHS for baseline, candidate in itertools.combinations(MODELS, 2)]
    comparisons += [(5000, run, old, run) for run in MODELS for old in (500, 2500)]
    pairs, pair_summary, contributions = [], [], []
    for epoch, candidate, before, baseline in comparisons:
        candidate_rows = {row["case_id"]: row for row in cases[epoch, candidate]}
        baseline_rows = {row["case_id"]: row for row in cases[before, baseline]}
        identity = {"candidate_epoch": epoch, "candidate": candidate, "baseline_epoch": before, "baseline": baseline}
        for key in pair_keys:
            values = [(case, number(candidate_rows[case].get(key)), number(baseline_rows[case].get(key))) for case in sorted(reference)]
            if any(x is None or y is None for _, x, y in values):
                continue
            delta = np.asarray([x - y for _, x, y in values])
            for case, x, y in values:
                pairs.append({**identity, "metric": key, "case_id": case, "candidate_value": x, "baseline_value": y, "delta": x - y})
            pair_summary.append({**identity, "metric": key, "n": len(delta), "wins": int((delta < 0).sum()), "losses": int((delta > 0).sum()), "ties": int((delta == 0).sum()), "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)), "p05_delta": float(np.quantile(delta, .05)), "p95_delta": float(np.quantile(delta, .95))})
        count = pooled(cases[epoch, candidate], "global_field_fluid_norm")["num_values"]
        component_sum = 0.0
        for channel in CHANNELS:
            key = f"field_{channel}_fluid_norm_sse"
            difference = (sum(float(row[key]) for row in candidate_rows.values()) - sum(float(row[key]) for row in baseline_rows.values())) / count
            component_sum += difference
            contributions.append({**identity, "channel": channel, "global_mse_delta_contribution": difference})
        total_difference = pooled(cases[epoch, candidate], "global_field_fluid_norm")["mse"] - pooled(cases[before, baseline], "global_field_fluid_norm")["mse"]
        if not np.isclose(component_sum, total_difference, rtol=1e-10, atol=1e-11):
            raise ValueError(f"Channel contributions do not reconcile: {identity}")

    outputs = {"headline": headline, "metrics_long": metrics, "pooled_metrics": pools, "strata": strata, "worst_cases": worst, "paired_case_deltas": pairs, "paired_summary": pair_summary, "channel_mse_contributions": contributions}
    for name, rows in outputs.items():
        write_csv(out / (name + ".csv"), rows)
    payload = {"sources": sources, "validation": {"datasets": len(cases), "unique_cases_each": 90, "common_target_columns": len(target_keys), "target_count_checks": target_checks, "both_missing_target_pairs": missing_target_pairs, "pooled_reconciliations": reconciliations, "channel_decompositions": len(comparisons)}, "headline": headline, "outputs": {name: str(out / (name + ".csv")) for name in outputs}, "definitions": {"pooled_l2": "sqrt(sum case SSE / sum case target SSE)", "pooled_mse": "sum case SSE / sum scalar values", "equal_case_l2": "individual case relative L2 summarized without size weighting", "paired_delta": "candidate minus baseline; lower error is better", "channel_contribution": "normalized channel SSE difference / common global fluid scalar-value count", "population": "90-case development holdout; one training seed, not independent physical-reference validation"}}
    (out / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    geometry_summaries(root)
    best_selected_summaries(root, reference, target_keys)
    print(json.dumps(payload["validation"]))
    for row in headline:
        if row["epoch"] == 5000:
            print(row)


if __name__ == "__main__":
    main()
