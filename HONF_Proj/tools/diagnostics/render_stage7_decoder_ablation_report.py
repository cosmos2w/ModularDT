#!/usr/bin/env python3
"""Render the mature Stage-7 decoder-context ablation report.

The script consumes only reduced CSVs and evaluator debug exports. It does not
load models or modify run artifacts. The HTML is self-contained for local use.
"""
from __future__ import annotations

import base64
import csv
import html
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
STUDY = PROJECT / "diagnostics/generated/interface_operator_study/stage7_decoder_ablation"
REDUCTION = STUDY / "reduction"
REPORT_MD = PROJECT / "docs/reports/HONF_Stage7_Decoder_Context_Ablation_Report.md"
REPORT_HTML = REPORT_MD.with_suffix(".html")
RUNS = ["1401", "1402", "1403", "1804"]
COLORS = {"1401": "#326b8c", "1402": "#c75b39", "1403": "#d69b2d", "1804": "#287271"}


def rows(name: str) -> list[dict[str, str]]:
    with (REDUCTION / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def by_run(data: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["run"]: row for row in data}


def fmt(value: str | float, digits: int = 5) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: float) -> str:
    return f"{100 * value:+.1f}%"


def table(headers: list[str], body: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
        + ["| " + " | ".join(row) + " |" for row in body]
    )


def figure_data_uri(fig: plt.Figure) -> str:
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def accuracy_figure(headline: dict[str, dict[str, str]], selected: dict[str, dict[str, str]]) -> str:
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    x = np.arange(4)
    width = 0.34
    exact = [float(headline[r]["relative_l2"]) for r in RUNS]
    best = [float(selected[r]["relative_l2"]) for r in RUNS]
    ax.bar(x - width / 2, exact, width, color=[COLORS[r] for r in RUNS], label="Exact epoch 5000")
    ax.bar(x + width / 2, best, width, facecolor="white", edgecolor=[COLORS[r] for r in RUNS], linewidth=2, label="Best field checkpoint")
    ax.set_xticks(x, ["1401\nLegacy", "1402\n−global", "1403\n−global/near", "1804\nDense"])
    ax.set_ylabel("Pooled fluid relative L2 (lower is better)")
    ax.set_ylim(0, max(exact) * 1.2)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    return figure_data_uri(fig)


def trajectory_figure(data: list[dict[str, str]]) -> str:
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for run in RUNS:
        subset = sorted((r for r in data if r["run"] == run), key=lambda r: int(r["epoch"]))
        ax.plot([int(r["epoch"]) for r in subset], [float(r["pooled_relative_l2"]) for r in subset], marker="o", lw=2.2, color=COLORS[run], label=run)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Pooled fluid relative L2")
    ax.set_xticks([500, 2500, 5000])
    ax.grid(alpha=.2)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    return figure_data_uri(fig)


def timing_figure(data: list[dict[str, str]]) -> str:
    values: dict[str, list[float]] = {run: [] for run in RUNS}
    for run in RUNS:
        anchors = [float(r["median_ms"]) for r in data if r["run"] == run and r["kind"] == "real_anchor" and r["phase"] == "full_forward"]
        small = next(float(r["median_ms"]) for r in data if r["run"] == run and r["shape"] == "M32_E768_Q65536")
        large = next(float(r["median_ms"]) for r in data if r["run"] == run and r["shape"] == "M128_E3072_Q262144")
        values[run] = [float(np.mean(anchors)), small, large]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    x = np.arange(3)
    width = .19
    for index, run in enumerate(RUNS):
        ax.bar(x + (index - 1.5) * width, values[run], width, color=COLORS[run], label=run)
    ax.set_yscale("log")
    ax.set_xticks(x, ["Two-anchor mean", "M32 / E768 / Q65k", "M128 / E3072 / Q262k"])
    ax.set_ylabel("Median full-forward time (ms, log scale)")
    ax.grid(axis="y", which="both", alpha=.2)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    return figure_data_uri(fig)


def routing_figure(data: list[dict[str, str]]) -> str:
    metrics = [
        ("module_row_effective_edges", "Module\neffective edges"),
        ("environment_row_effective_edges", "Environment\neffective edges"),
        ("environment_spatial_smoothness", "Environment\nsmoothness"),
        ("region_separation_normalized", "Region\nseparation"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.7))
    for ax, (metric, label) in zip(axes, metrics):
        subset = {r["run"]: float(r["equal_case_mean"]) for r in data if r["metric"] == metric}
        ax.bar(range(3), [subset[r] for r in RUNS[:3]], color=[COLORS[r] for r in RUNS[:3]])
        ax.set_xticks(range(3), RUNS[:3], rotation=35)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Hyperedge organization summaries (Dense 1804 has no learned partition)", fontsize=11)
    fig.tight_layout()
    return figure_data_uri(fig)


def error_map_figure() -> str:
    paths = {
        "1401 Legacy": STUDY / "historical_evaluation/debug_npz/Legacy_1401__5000_replay__0298.npz",
        "1402 −global": STUDY / "evaluation/debug_npz/No-global_1402__5000__0298.npz",
        "1403 −global/near": STUDY / "evaluation/debug_npz/No-global-near_1403__5000__0298.npz",
        "1804 Dense": STUDY / "historical_evaluation/debug_npz/Dense_1804__5000_replay__0298.npz",
    }
    maps: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for label, path in paths.items():
        with np.load(path) as data:
            scale = np.sqrt(np.mean(np.square(data["gt_field_grid"][data["fluid_mask"]]), axis=0))
            scale = np.where(scale > 1e-8, scale, 1.0)
            error = np.sqrt(np.mean(np.square((data["pred_field_grid"] - data["gt_field_grid"]) / scale), axis=-1))
            error = np.where(data["fluid_mask"], error, np.nan)
            maps.append((label, error, data["x_grid"].copy(), data["y_grid"].copy()))
    vmax = float(np.nanquantile(np.concatenate([m[1].ravel() for m in maps]), .98))
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.7), constrained_layout=True)
    image = None
    for ax, (label, error, x, y) in zip(axes.ravel(), maps):
        image = ax.pcolormesh(x, y, error, shading="nearest", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=.82, label="Five-channel normalized error magnitude (common scale)")
    fig.suptitle("Case 0298: spatial error structure at exact epoch 5000", fontsize=12)
    return figure_data_uri(fig)


def html_table(headers: list[str], body: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(x)}</th>" for x in headers)
    cells = "".join("<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>" for row in body)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{cells}</tbody></table></div>"


def main() -> None:
    headline = by_run(rows("headline"))
    selected = by_run(rows("best_selected"))
    trajectory = rows("trajectory")
    training = by_run(rows("training"))
    timing = rows("timing")
    routing = rows("routing")
    topology = rows("topology_quality")
    components = rows("components")
    anchors = rows("anchors")

    models = {r: headline[r]["model"] for r in RUNS}
    accuracy_body = [[r, models[r], fmt(headline[r]["relative_l2"]), fmt(headline[r]["equal_case_mean"]), fmt(headline[r]["median"]), fmt(headline[r]["p95"]), fmt(headline[r]["maximum"]), headline[r]["worst_case"]] for r in RUNS]
    component_names = ["Near-interface field", "Far-fluid field", "pressure", "vorticity", "internal temperature", "surface temperature", "normal heat flux", "final outside temperature"]
    component_by = {(r["run"], r["metric"]): r for r in components}
    component_body = [[name] + [fmt(component_by[(run, name)]["relative_l2"]) for run in RUNS] for name in component_names]
    trajectory_by = {(r["run"], r["epoch"]): r["pooled_relative_l2"] for r in trajectory}
    trajectory_body = [[epoch] + [fmt(trajectory_by[(run, epoch)]) for run in RUNS] for epoch in ["500", "2500", "5000"]]
    training_body = [[r, f"{int(float(training[r]['trainable_parameters'])):,}", fmt(training[r]["final_val_field_mse"], 6), fmt(training[r]["best_val_field_mse"], 6), training[r]["first_epoch_val_field_le_0p01"], (fmt(float(training[r]["train_wall_hours"]) + float(training[r]["val_wall_hours"]), 2) if training[r]["train_wall_hours"] else "not logged"), (fmt(training[r]["peak_cuda_memory_mib"], 0) if training[r]["peak_cuda_memory_mib"] else "not logged")] for r in RUNS]
    threshold_body = [
        ["≤ 0.02", "461", "510", "409", "357"],
        ["≤ 0.01", "831", "898", "805", "651"],
        ["≤ 0.005", "1,431", "1,732", "1,597", "1,109"],
        ["≤ 0.003", "2,013", "4,486", "3,542", "1,764"],
    ]

    def time_row(run: str) -> list[str]:
        real = [float(x["median_ms"]) for x in timing if x["run"] == run and x["kind"] == "real_anchor" and x["phase"] == "full_forward"]
        decode = [float(x["median_ms"]) for x in timing if x["run"] == run and x["kind"] == "real_anchor" and x["phase"] == "prepared_decode"]
        small = next(x for x in timing if x["run"] == run and x["shape"] == "M32_E768_Q65536")
        large = next(x for x in timing if x["run"] == run and x["shape"] == "M128_E3072_Q262144")
        return [run, fmt(np.mean(real), 2), fmt(np.mean(decode), 2), fmt(small["median_ms"], 1), fmt(large["median_ms"], 1), fmt(large["incremental_peak_allocated_mib"], 0)]
    timing_body = [time_row(r) for r in RUNS]

    route_map = {(r["run"], r["metric"]): r for r in routing if r["status"] == "measured"}
    routing_metrics = [
        ("module_affinity_norm_l2", "Affinity target relative L2 ↓"),
        ("static_organization_env_mass_entropy_norm", "Environment mass entropy / log K"),
        ("static_organization_env_mass_max", "Largest environment mass"),
        ("base_vs_final_module_mass_shift", "Module mass shift"),
        ("base_vs_final_env_mass_shift", "Environment mass shift"),
        ("routing_query_attention_effective_edges", "Effective query hyperedges"),
        ("routing_query_attention_max", "Largest query weight"),
        ("routing_pairwise_edge_contribution_mean", "Pairwise edge contribution norm"),
    ]
    routing_body = [[label] + [fmt(route_map[(run, key)]["equal_case_mean"]) for run in RUNS[:3]] for key, label in routing_metrics]
    topology_map = {(r["run"], r["metric"]): r for r in topology}
    topology_metrics = [
        ("module_row_entropy_norm", "Module row entropy / log K"),
        ("module_row_effective_edges", "Module effective edges"),
        ("module_row_max", "Largest module-row weight"),
        ("module_largest_dominant_occupancy", "Largest module dominant occupancy"),
        ("environment_row_entropy_norm", "Environment row entropy / log K"),
        ("environment_row_effective_edges", "Environment effective edges"),
        ("environment_row_max", "Largest environment-row weight"),
        ("environment_largest_dominant_occupancy", "Largest environment dominant occupancy"),
        ("environment_neighbor_dominant_agreement", "Environment neighbor agreement"),
        ("environment_neighbor_l1", "Environment neighbor L1"),
        ("environment_spatial_smoothness", "Environment spatial smoothness"),
        ("source_separation_normalized", "Source separation"),
        ("region_separation_normalized", "Region separation"),
        ("base_to_final_alignment_cost", "Base→final alignment cost"),
        ("base_to_final_module_assignment_l1", "Base→final module assignment L1"),
        ("base_to_final_environment_assignment_l1", "Base→final environment assignment L1"),
    ]
    topology_body = [[label] + [fmt(topology_map[(run, key)]["equal_case_mean"]) for run in RUNS[:3]] for key, label in topology_metrics]
    anchor_map = {(r["case_id"], r["run"]): r["field_relative_l2"] for r in anchors}
    anchor_body = [[case] + [fmt(anchor_map[(case, run)]) for run in RUNS] for case in ["0273", "0653", "0298", "0302"]]

    r1401, r1402, r1403, r1804 = (float(headline[r]["relative_l2"]) for r in RUNS)
    markdown = f"""# HONF Stage-7 Decoder-Context Ablation Report

**Runs:** 1401, 1402, 1403, and 1804  
**Primary checkpoint policy:** exact epoch 5000  
**Evaluation population:** the same 90-case development holdout, all 8,192 grid queries per case  
**Report date:** 2026-09-16

## Decision summary

The global decoder context is necessary for the Stage-7 model represented by Run 1401. Removing it in Run 1402 raises pooled fluid relative L2 from **{r1401:.5f} to {r1402:.5f} ({pct(r1402 / r1401 - 1)})**, and Run 1402 loses all 90 paired cases. Removing near-module context as well (Run 1403) recovers much of that loss relative to Run 1402, but still ends **{pct(r1403 / r1401 - 1)} worse than 1401** in pooled L2.

Run 1403 has a slightly better case median than 1401 and wins 49/90 pairs, but its p95 and maximum errors are much worse. Its favorable central cases therefore do not support removing both contexts. Run 1804 remains the accuracy leader at **{r1804:.5f}**, while the Stage-7 family is much cheaper at inference on the measured shapes.

The routing diagnostics do not rescue either ablation. Runs 1402 and 1403 achieve lower module-affinity target error and more diffuse environment mass than 1401, yet reconstruct the field less accurately. All three models retain six fixed organizer edges and route queries over roughly 3.7–3.9 effective edges. These geometry summaries show organized, non-collapsed routing; they do not show useful physical coupling by themselves.

## Models and controlled comparison

- **1401 / Legacy:** original Stage-7 structured-context reference.
- **1402 / No global:** Run 1401 configuration with global decoder context disabled.
- **1403 / No global/near:** global and near-module decoder contexts disabled.
- **1804 / Dense:** dense pairwise field-adaptation reference. It has no learned hyperedge partition, so hyperedge clustering statistics are not applicable.

The exact checkpoints were evaluated with predicted port conditions and routing-map export. Current replay reproduces the historical Run 1401 result exactly; Run 1804 differs from the prior aggregate only at approximately 6e-10 in pooled relative L2. Every model uses the same case IDs, targets, masks, and query count. This is a one-seed development-holdout comparison, not an independent physical-reference validation.

## Reconstruction accuracy

{table(["Run", "Model", "Pooled L2", "Equal-case mean", "Median", "p95", "Maximum", "Worst case"], accuracy_body)}

Run 1403 beats Run 1402 on 80/90 exact-endpoint cases and reduces pooled L2 by **{100 * (1 - r1403 / r1402):.2f}%**. That conditional recovery means the near-module branch is counterproductive or redundant after global context is removed. It does not establish that near context is harmful when global context is present; a global-retained, near-disabled run would be required for that claim.

Run 1403's tail is the main concern: its median is 0.02928 versus 0.03162 for 1401, while its p95 is 0.10829 versus 0.06024. Difficult cases include 0284, 0286, 0283, 0295, 0297, and 0285. Dense 1804 beats 1403 in 82/90 cases and 1402 in all 90.

### Component and region errors

{table(["Metric"] + RUNS, component_body)}

Run 1402 degrades every reported fluid channel; pressure more than doubles relative to 1401. Run 1403 nearly restores vorticity and is competitive on internal and interface temperature, but pressure and far-fluid reconstruction remain poor. Dense 1804 is strongest on velocity, pressure, internal module temperature, surface temperature, and normal heat flux. These results point to a field-communication failure rather than merely a port-read failure.

### Four established anchors

{table(["Case"] + RUNS, anchor_body)}

The anchors expose the heterogeneity hidden by averages. Run 1403 is best among the Stage-7 models on 0273 and 0653 and is close to 1401 on 0298, but degrades sharply on 0302. The corresponding HTML includes a common-scale spatial error map for case 0298; it is demonstrative rather than a substitute for the 90-case statistics.

## Convergence and checkpoint sensitivity

{table(["Epoch"] + RUNS, trajectory_body)}

{table(["Run", "Parameters", "Final val field MSE", "Best val field MSE", "First val MSE ≤ 0.01", "Recorded wall h", "Peak CUDA MiB"], training_body)}

{table(["First validation field-MSE crossing"] + RUNS, threshold_body)}

Run 1403 crosses the first 0.01 validation-field threshold earlier than 1401 and 1402, but its exact endpoint and last-100-epoch validation behavior do not reach 1401. Run 1402 improves little after epoch 2500. Dense is already best at epoch 500 and becomes clearly best by epoch 5000.

Validation-selected best-field checkpoints give pooled L2 values of **0.03210 (1401, epoch 4585), 0.05099 (1402, epoch 4936), 0.05066 (1403, epoch 4588), and 0.02896 (1804, epoch 4738)**. These values use the same development holdout for selection and measurement, so they are sensitivity evidence. Exact epoch 5000 remains the primary policy.

The historical training wall records are not controlled across runs. Run 1804 includes continuation accounting and different execution conditions; the old 1401 history does not log the same decomposed timing and memory columns. They describe operational history, while the synchronized timing below supports architecture cost comparisons.

## Controlled computation cost

{table(["Run", "Anchor full ms", "Anchor prepared decode ms", "M32/E768/Q65k ms", "M128/E3072/Q262k ms", "Largest allocated MiB"], timing_body)}

The two context switches leave parameter count and measured allocation unchanged within the Stage-7 family. They also provide no repeatable speed benefit: small latency differences are comparable to run-to-run timing variation because the disabled contributions do not remove the surrounding model structure.

Dense 1804 is about **{float(timing_body[3][4]) / float(timing_body[0][4]):.2f}×** slower than 1401 on the largest synthetic shape and about **{float(timing_body[3][1]) / float(timing_body[0][1]):.2f}×** slower on the two-anchor mean. Its largest-shape incremental allocated memory is lower in this chunked benchmark, despite higher latency and parameter count. Allocation is architecture- and chunk-schedule-dependent and is not total process residency.

Only Dense 1804 currently exports operation rows in this benchmark. At the largest synthetic shape it reports 855,641,088 environment-geometry-bias operations, 35,651,712 query-module messages, 1,179,648 messages for each EM and ME direction, and 49,152 MM messages. The Stage-7 operation dictionaries are empty because that legacy wrapper is not instrumented; they must not be interpreted as zero work.

## Hyperedge routing and clustering quality

{table(["Metric", "1401", "1402", "1403"], routing_body)}

All values are equal-case means over 90 cases. Lower affinity-target relative L2 means closer agreement with the organizer target; entropy near one means mass is distributed relatively evenly across the fixed six edges. The active-edge-count target relative L2 is identical at 0.44405 for all three runs because this experiment retains the same fixed six edges. It does not test edge-count selection.

### Full-holdout assignment geometry

{table(["Metric", "1401", "1402", "1403"], topology_body)}

Run 1402 sharpens module assignments and concentrates dominant modules: module effective edges fall from 5.75 to 4.78 and the largest dominant occupancy rises from 0.475 to 0.608. Run 1403 remains sharper than 1401 but distributes dominant modules more evenly. Both ablations make environmental assignments more diffuse and spatially smoother, while reducing region separation. Their base-to-final assignment changes are also larger, so decoder context affects the organization recomputed through the physical loop.

Run 1402 moves aggregate module mass more than twice as much as 1401, while Run 1403 returns near the 1401 shift. Query routing remains multi-edge rather than collapsing to one edge. The pairwise edge-contribution norm falls from 2.2366 to 1.4335 and then 0.4366, yet 1403 reconstructs better than 1402. Contribution magnitude, entropy, clustering sharpness, and target affinity therefore cannot be read as direct evidence of useful field mediation.

Dense 1804 receives “not applicable” for these columns because it uses dense environment attention and direct query-module communication instead of a learned hyperedge partition. Missing values are not zeros and do not imply inferior organization.

## Conclusions and next research decision

1. **Retain global decoder context in the Stage-7 baseline.** Run 1402 is uniformly worse than 1401 and provides no measured cost reduction.
2. **Do not adopt Run 1403 as a replacement.** It fixes much of Run 1402's damage and improves many central cases, but pooled error and tail robustness remain substantially worse than 1401.
3. **Use Run 1804 as the accuracy reference and Run 1401 as the lower-cost structured-routing reference.** Dense gives the best reconstruction at higher latency; Stage-7 remains materially cheaper on the measured workloads.
4. **Treat organization metrics as diagnostics, not success criteria.** Better target affinity and diffuse hyperedge mass did not imply better reconstruction. Future changes should be judged through paired ground-truth field error and intervention-based influence.
5. If isolating the near-module branch remains scientifically important, the minimal next ablation is global context retained with only near context disabled. It should be run only to answer that specific interaction question; Runs 1402/1403 already rule out the proposed decoder simplifications as replacements.

## Reproducibility and artifacts

Primary reduced tables and machine-readable summary:

`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/reduction/`

Exact candidate evaluation:

`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/evaluation/`

Matched historical replay:

`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/historical_evaluation/`

Trajectory, selected-checkpoint, and timing evidence:

`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/trajectory_evaluation/`  
`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/best_field_evaluation/`  
`diagnostics/generated/interface_operator_study/stage7_decoder_ablation/timing/four_model_timing.json`

Reducer and renderer:

`tools/diagnostics/analyze_stage7_decoder_ablation.py`  
`tools/diagnostics/render_stage7_decoder_ablation_report.py`

The exact evaluator command and checkpoint paths are recorded in each evaluation directory's `comparison_manifest.json`; execution logs are retained beside the output directories. The report does not claim CFD or new physical-reference validation.
"""
    REPORT_MD.write_text(markdown, encoding="utf-8")

    figures = {
        "accuracy": accuracy_figure(headline, selected),
        "trajectory": trajectory_figure(trajectory),
        "timing": timing_figure(timing),
        "routing": routing_figure(topology),
        "error": error_map_figure(),
    }
    css = """
    :root{--ink:#17232c;--muted:#5d6b74;--paper:#f5f2eb;--card:#fff;--line:#d9d4c9;--accent:#287271}
    *{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.58 system-ui,-apple-system,Segoe UI,sans-serif}
    main{max-width:1120px;margin:auto;padding:38px 28px 80px} h1{font:700 clamp(32px,5vw,58px)/1.03 Georgia,serif;max-width:900px;margin:.25em 0}
    h2{font:700 28px/1.2 Georgia,serif;margin:2.1em 0 .55em;border-top:1px solid var(--line);padding-top:1em} h3{font:700 20px Georgia,serif}
    .eyebrow{letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:700;font-size:13px}.lede{font-size:20px;max-width:900px;color:#33434d}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}.card{background:var(--card);padding:18px;border:1px solid var(--line);border-radius:14px}.card b{display:block;font-size:27px}.card span{color:var(--muted);font-size:13px}
    .callout{background:#e7f0ed;border-left:5px solid var(--accent);padding:18px 22px;border-radius:6px;margin:24px 0}.figure{background:white;border:1px solid var(--line);border-radius:14px;padding:14px;margin:24px 0}.figure img{display:block;width:100%;height:auto}.caption{font-size:13px;color:var(--muted);padding:4px 10px 8px}
    .table-wrap{overflow:auto;margin:18px 0 26px;background:white;border:1px solid var(--line);border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#263943;color:white;text-align:left}th,td{padding:10px 12px;border-bottom:1px solid #e8e4dc;white-space:nowrap}td:not(:first-child),th:not(:first-child){text-align:right}
    code{background:#ece8df;padding:2px 5px;border-radius:4px}ul,ol{padding-left:24px}.note{color:var(--muted);font-size:14px}@media(max-width:760px){.cards{grid-template-columns:1fr 1fr}main{padding:24px 14px}.lede{font-size:17px}}
    """
    html_doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HONF Stage-7 Decoder-Context Ablation</title><style>{css}</style></head><body><main>
    <div class="eyebrow">HONF · mature checkpoint comparison · 2026-09-16</div><h1>Decoder context is useful; routing neatness is not enough</h1>
    <p class="lede">A matched 90-case comparison of Stage-7 Runs 1401, 1402, and 1403 against Dense Run 1804, covering exact reconstruction, convergence, synchronized GPU cost, and hyperedge organization.</p>
    <div class="cards"><div class="card"><b>0.03740</b><span>Run 1401 pooled L2</span></div><div class="card"><b>+77.7%</b><span>Run 1402 vs 1401</span></div><div class="card"><b>+40.4%</b><span>Run 1403 vs 1401</span></div><div class="card"><b>0.02966</b><span>Dense 1804 pooled L2</span></div></div>
    <div class="callout"><b>Decision.</b> Retain global context. Removing near context after global removal recovers accuracy, but does not repair the tail or match 1401. Dense remains the accuracy reference; 1401 remains the lower-cost structured-routing reference.</div>
    <h2>Exact epoch-5000 reconstruction</h2>{html_table(["Run","Model","Pooled L2","Equal-case mean","Median","p95","Maximum","Worst"],accuracy_body)}
    <div class="figure"><img src="{figures['accuracy']}" alt="Exact and selected pooled relative L2 comparison"><div class="caption">Exact epoch 5000 is primary. Best-field checkpoints are shown only as development-holdout sensitivity evidence.</div></div>
    <p>Run 1402 loses all 90 paired cases to 1401. Run 1403 beats 1402 in 80/90 cases and beats 1401 in 49/90, but its severe tail raises pooled error by 40.4% relative to 1401. Dense beats 1403 in 82/90 cases.</p>
    <h3>Component and region errors</h3>{html_table(["Metric"]+RUNS,component_body)}
    <p>Pressure and far-fluid reconstruction carry the clearest ablation damage. Run 1403 nearly restores vorticity and performs well on internal temperature, but that does not translate into robust global field reconstruction.</p>
    <h3>Four established anchors</h3>{html_table(["Case"]+RUNS,anchor_body)}
    <div class="figure"><img src="{figures['error']}" alt="Spatial normalized error maps for case 0298"><div class="caption">Case 0298, five-channel normalized error magnitude, common color scale. This spatial example is descriptive; the decision uses all 90 cases.</div></div>
    <h2>Convergence</h2>{html_table(["Epoch"]+RUNS,trajectory_body)}
    <div class="figure"><img src="{figures['trajectory']}" alt="Pooled relative L2 by epoch"><div class="caption">Matched full-grid evaluation at epochs 500, 2500, and 5000.</div></div>
    {html_table(["Run","Parameters","Final val MSE","Best val MSE","First ≤.01","Recorded h","Peak MiB"],training_body)}
    {html_table(["First validation field-MSE crossing"]+RUNS,threshold_body)}
    <p>Run 1403 crosses an early validation threshold quickly, yet does not reach 1401 at maturity. Run 1402 largely stalls after epoch 2500. Historical wall-time rows are operational records rather than a controlled architecture benchmark.</p>
    <h2>Controlled GPU cost</h2>{html_table(["Run","Anchor full ms","Prepared decode ms","M32/E768/Q65k ms","M128/E3072/Q262k ms","Largest MiB"],timing_body)}
    <div class="figure"><img src="{figures['timing']}" alt="Controlled full-forward latency"><div class="caption">Median synchronized GPU full-forward time. Synthetic timing uses one warmup and three measured repeats; anchors use two warmups and five repeats.</div></div>
    <p>The decoder switches leave Stage-7 parameters and allocation unchanged and show no practical speedup. Dense is about 3× slower on the largest synthetic shape, while using less incremental allocated memory under the measured chunk schedule. Operation counters are available only for Dense in this harness; empty Stage-7 dictionaries mean uninstrumented, not zero work.</p>
    <h2>Hyperedge organization</h2>{html_table(["Metric","1401","1402","1403"],routing_body)}
    <h3>Full-holdout assignment geometry</h3>{html_table(["Metric","1401","1402","1403"],topology_body)}
    <div class="figure"><img src="{figures['routing']}" alt="Hyperedge organization metrics"><div class="caption">Equal-case means over 90 cases. Dense 1804 has no learned hyperedge partition, so these quantities are not applicable.</div></div>
    <p>All three Stage-7 runs preserve multi-edge query attention. The ablations improve affinity-target agreement and diffuse environmental mass, but worsen field reconstruction. The pairwise edge-contribution norm also falls sharply without predicting accuracy. These properties describe organization; paired ground-truth error and causal interventions are needed to establish useful mediation.</p>
    <h2>Research handoff</h2><ol><li>Retain global decoder context in the Stage-7 reference.</li><li>Do not replace 1401 with either ablation.</li><li>Use 1804 as the accuracy reference and 1401 as the lower-cost routed reference.</li><li>If the near branch must be isolated, test only a global-retained, near-disabled configuration; 1402/1403 cannot answer that interaction cleanly.</li></ol>
    <p class="note">Scope: one seed and the established 90-case development holdout. Best-checkpoint results reuse that holdout and are sensitivity evidence. No new CFD or physical-reference validation is claimed. Full commands, manifests, logs, and reduced CSVs are retained under <code>diagnostics/generated/interface_operator_study/stage7_decoder_ablation/</code>.</p>
    </main></body></html>"""
    REPORT_HTML.write_text(html_doc, encoding="utf-8")
    print(REPORT_MD)
    print(REPORT_HTML)


if __name__ == "__main__":
    main()
