#!/usr/bin/env python3
"""Render the mature Stage-7 decoder-context and routing-only comparison."""
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
RUNS = ["1401", "1402", "1403", "1404", "1804"]
ROUTED = RUNS[:4]
COLORS = {"1401": "#326b8c", "1402": "#c75b39", "1403": "#d69b2d", "1404": "#755da8", "1804": "#287271"}


def rows(name: str) -> list[dict[str, str]]:
    with (REDUCTION / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def by_run(data: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["run"]: row for row in data}


def fmt(value: str | float | None, digits: int = 5) -> str:
    return "not logged" if value in {None, ""} else f"{float(value):.{digits}f}"


def integer(value: str | float | None) -> str:
    return "—" if value in {None, ""} else f"{int(float(value)):,}"


def pct(value: float) -> str:
    return f"{100 * value:+.1f}%"


def md_table(headers: list[str], body: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
        + ["| " + " | ".join(row) + " |" for row in body]
    )


def html_table(headers: list[str], body: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(x)}</th>" for x in headers)
    cells = "".join("<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>" for row in body)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{cells}</tbody></table></div>"


def data_uri(fig: plt.Figure) -> str:
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def accuracy_figure(headline: dict[str, dict[str, str]], selected: dict[str, dict[str, str]]) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(5)
    exact = [float(headline[r]["relative_l2"]) for r in RUNS]
    best = [float(selected[r]["relative_l2"]) for r in RUNS]
    ax.bar(x - .17, exact, .34, color=[COLORS[r] for r in RUNS], label="Exact epoch 5000")
    ax.bar(x + .17, best, .34, facecolor="white", edgecolor=[COLORS[r] for r in RUNS], linewidth=2, label="Best field checkpoint")
    ax.set_xticks(x, ["1401\nLegacy", "1402\n−global", "1403\n−global/near", "1404\nRouting-only", "1804\nDense"])
    ax.set_ylabel("Pooled fluid relative L2")
    ax.set_ylim(0, max(exact) * 1.2)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return data_uri(fig)


def trajectory_figure(data: list[dict[str, str]]) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for run in RUNS:
        subset = sorted((r for r in data if r["run"] == run), key=lambda r: int(r["epoch"]))
        ax.plot([int(r["epoch"]) for r in subset], [float(r["pooled_relative_l2"]) for r in subset], marker="o", lw=2.2, color=COLORS[run], label=run)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Pooled fluid relative L2")
    ax.set_xticks([500, 2500, 5000])
    ax.grid(alpha=.2)
    ax.legend(frameon=False, ncol=5)
    fig.tight_layout()
    return data_uri(fig)


def timing_figure(data: list[dict[str, str]]) -> str:
    values = {}
    for run in RUNS:
        anchors = [float(r["median_ms"]) for r in data if r["run"] == run and r["kind"] == "real_anchor" and r["phase"] == "full_forward"]
        small = next(float(r["median_ms"]) for r in data if r["run"] == run and r["shape"] == "M32_E768_Q65536")
        large = next(float(r["median_ms"]) for r in data if r["run"] == run and r["shape"] == "M128_E3072_Q262144")
        values[run] = [float(np.mean(anchors)), small, large]
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    x = np.arange(3)
    for index, run in enumerate(RUNS):
        ax.bar(x + (index - 2) * .16, values[run], .16, color=COLORS[run], label=run)
    ax.set_yscale("log")
    ax.set_xticks(x, ["Two-anchor mean", "M32/E768/Q65k", "M128/E3072/Q262k"])
    ax.set_ylabel("Median full-forward time (ms, log scale)")
    ax.grid(axis="y", which="both", alpha=.2)
    ax.legend(frameon=False, ncol=5)
    fig.tight_layout()
    return data_uri(fig)


def topology_figure(data: list[dict[str, str]]) -> str:
    metrics = [("module_row_effective_edges", "Module\neffective edges"), ("environment_row_effective_edges", "Environment\neffective edges"), ("environment_spatial_smoothness", "Environment\nsmoothness"), ("region_separation_normalized", "Region\nseparation")]
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.8))
    for ax, (metric, label) in zip(axes, metrics):
        subset = {r["run"]: float(r["equal_case_mean"]) for r in data if r["metric"] == metric}
        ax.bar(range(4), [subset[r] for r in ROUTED], color=[COLORS[r] for r in ROUTED])
        ax.set_xticks(range(4), ROUTED, rotation=35)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Hyperedge organization (Dense has no learned partition)", fontsize=11)
    fig.tight_layout()
    return data_uri(fig)


def usefulness_figure(data: list[dict[str, str]]) -> str:
    order = ["normal", "uniform_query_to_edge", "uniform_module_to_edge", "base_pair_module_token", "suppress_pair_context"]
    labels = ["Normal", "Uniform query→edge", "Uniform module→edge", "Base module token", "No pair context"]
    indexed = {r["variant"]: r for r in data}
    values = [float(indexed[key]["fluid_mse_mean"]) for key in order]
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    ax.bar(range(5), values, color=["#8b95a1"] + [COLORS["1404"]] * 4)
    ax.set_yscale("log")
    ax.set_xticks(range(5), labels, rotation=18, ha="right")
    ax.set_ylabel("Four-anchor mean fluid MSE (log scale)")
    ax.grid(axis="y", which="both", alpha=.2)
    fig.tight_layout()
    return data_uri(fig)


def error_map_figure() -> str:
    paths = {
        "1401 Legacy": STUDY / "historical_evaluation/debug_npz/Legacy_1401__5000_replay__0298.npz",
        "1402 −global": STUDY / "evaluation/debug_npz/No-global_1402__5000__0298.npz",
        "1403 −global/near": STUDY / "evaluation/debug_npz/No-global-near_1403__5000__0298.npz",
        "1404 Routing-only": STUDY / "run1404_evaluation/debug_npz/Routing-only_1404__5000__0298.npz",
        "1804 Dense": STUDY / "historical_evaluation/debug_npz/Dense_1804__5000_replay__0298.npz",
    }
    maps = []
    for label, path in paths.items():
        with np.load(path) as data:
            scale = np.sqrt(np.mean(np.square(data["gt_field_grid"][data["fluid_mask"]]), axis=0))
            scale = np.where(scale > 1e-8, scale, 1.0)
            error = np.sqrt(np.mean(np.square((data["pred_field_grid"] - data["gt_field_grid"]) / scale), axis=-1))
            maps.append((label, np.where(data["fluid_mask"], error, np.nan), data["x_grid"].copy(), data["y_grid"].copy()))
    vmax = float(np.nanquantile(np.concatenate([m[1].ravel() for m in maps]), .98))
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6), constrained_layout=True)
    image = None
    for ax, (label, error, x, y) in zip(axes.ravel(), maps):
        image = ax.pcolormesh(x, y, error, shading="nearest", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    axes.ravel()[-1].axis("off")
    fig.colorbar(image, ax=axes, shrink=.82, label="Five-channel normalized error magnitude")
    fig.suptitle("Case 0298 at exact epoch 5000 (common scale)", fontsize=12)
    return data_uri(fig)


def main() -> None:
    headline, selected, training = map(by_run, [rows("headline"), rows("best_selected"), rows("training")])
    trajectory, timing, routing, topology = rows("trajectory"), rows("timing"), rows("routing"), rows("topology_quality")
    components, anchors, paired = rows("components"), rows("anchors"), rows("paired")
    usefulness, gradients, retained = rows("run1404_usefulness"), rows("run1404_gradients"), rows("run1404_retained_mass")
    models = {r: headline[r]["model"] for r in RUNS}
    accuracy_body = [[r, models[r], fmt(headline[r]["relative_l2"]), fmt(headline[r]["equal_case_mean"]), fmt(headline[r]["median"]), fmt(headline[r]["p95"]), fmt(headline[r]["maximum"]), headline[r]["worst_case"]] for r in RUNS]
    names = ["Near-interface field", "Far-fluid field", "u", "v", "pressure", "vorticity", "field temperature", "internal temperature", "surface temperature", "normal heat flux", "final outside temperature"]
    component_map = {(r["run"], r["metric"]): r for r in components}
    component_body = [[name] + [fmt(component_map[(run, name)]["relative_l2"]) for run in RUNS] for name in names]
    trajectory_map = {(r["run"], r["epoch"]): r["pooled_relative_l2"] for r in trajectory}
    trajectory_body = [[epoch] + [fmt(trajectory_map[(run, epoch)]) for run in RUNS] for epoch in ["500", "2500", "5000"]]
    training_body = [[r, f"{int(float(training[r]['trainable_parameters'])):,}", fmt(training[r]["final_val_field_mse"], 6), fmt(training[r]["best_val_field_mse"], 6), integer(training[r]["first_epoch_val_field_le_0p01"]), (fmt(float(training[r]["train_wall_hours"]) + float(training[r]["val_wall_hours"]), 2) if training[r]["train_wall_hours"] else "not logged"), (fmt(training[r]["peak_cuda_memory_mib"], 0) if training[r]["peak_cuda_memory_mib"] else "not logged")] for r in RUNS]
    threshold_body = [[label] + [integer(training[r][key]) for r in RUNS] for label, key in [("≤ 0.02", "first_epoch_val_field_le_0p02"), ("≤ 0.01", "first_epoch_val_field_le_0p01"), ("≤ 0.005", "first_epoch_val_field_le_0p005"), ("≤ 0.003", "first_epoch_val_field_le_0p003")]]
    selected_body = [[r, integer(selected[r]["checkpoint_epoch"]), fmt(selected[r]["relative_l2"]), fmt(selected[r]["equal_case_mean"]), fmt(selected[r]["p95"]), fmt(selected[r]["maximum"])] for r in RUNS]

    def time_row(run: str) -> list[str]:
        real = [float(r["median_ms"]) for r in timing if r["run"] == run and r["kind"] == "real_anchor" and r["phase"] == "full_forward"]
        decode = [float(r["median_ms"]) for r in timing if r["run"] == run and r["kind"] == "real_anchor" and r["phase"] == "prepared_decode"]
        small = next(r for r in timing if r["run"] == run and r["shape"] == "M32_E768_Q65536")
        large = next(r for r in timing if r["run"] == run and r["shape"] == "M128_E3072_Q262144")
        return [run, fmt(np.mean(real), 2), fmt(np.mean(decode), 2), fmt(small["median_ms"], 1), fmt(large["median_ms"], 1), fmt(large["incremental_peak_allocated_mib"], 0)]

    timing_body = [time_row(r) for r in RUNS]
    route_map = {(r["run"], r["metric"]): r for r in routing if r["status"] == "measured"}
    route_metrics = [("module_affinity_norm_l2", "Affinity target relative L2 ↓"), ("static_organization_env_mass_entropy_norm", "Environment mass entropy / log K"), ("static_organization_env_mass_max", "Largest environment mass"), ("base_vs_final_module_mass_shift", "Module mass shift"), ("base_vs_final_env_mass_shift", "Environment mass shift"), ("routing_query_attention_effective_edges", "Effective query hyperedges"), ("routing_query_attention_max", "Largest query weight"), ("routing_pairwise_edge_contribution_mean", "Pairwise edge contribution norm")]
    routing_body = [[label] + [fmt(route_map[(r, key)]["equal_case_mean"]) for r in ROUTED] for key, label in route_metrics]
    topology_map = {(r["run"], r["metric"]): r for r in topology}
    topology_metrics = [("module_row_entropy_norm", "Module row entropy / log K"), ("module_row_effective_edges", "Module effective edges"), ("module_row_max", "Largest module-row weight"), ("module_largest_dominant_occupancy", "Largest module dominant occupancy"), ("environment_row_entropy_norm", "Environment row entropy / log K"), ("environment_row_effective_edges", "Environment effective edges"), ("environment_row_max", "Largest environment-row weight"), ("environment_largest_dominant_occupancy", "Largest environment dominant occupancy"), ("environment_neighbor_dominant_agreement", "Environment neighbor agreement"), ("environment_neighbor_l1", "Environment neighbor L1"), ("environment_spatial_smoothness", "Environment spatial smoothness"), ("source_separation_normalized", "Source separation"), ("region_separation_normalized", "Region separation"), ("base_to_final_alignment_cost", "Base→final alignment cost"), ("base_to_final_module_assignment_l1", "Base→final module assignment L1"), ("base_to_final_environment_assignment_l1", "Base→final environment assignment L1")]
    topology_body = [[label] + [fmt(topology_map[(r, key)]["equal_case_mean"]) for r in ROUTED] for key, label in topology_metrics]
    anchor_map = {(r["case_id"], r["run"]): r["field_relative_l2"] for r in anchors}
    anchor_body = [[case] + [fmt(anchor_map[(case, r)]) for r in RUNS] for case in ["0273", "0653", "0298", "0302"]]
    pair_map = {(r["baseline"], r["candidate"]): r for r in paired}
    paired_body = [[r, pair_map[("1401", r)]["candidate_wins"], pair_map[("1401", r)]["candidate_losses"], fmt(pair_map[("1401", r)]["mean_case_l2_delta"])] for r in ["1402", "1403", "1404", "1804"]]
    labels = {"normal": "Normal", "uniform_query_to_edge": "Uniform query→edge", "uniform_module_to_edge": "Uniform module→edge", "base_pair_module_token": "Base module token", "suppress_pair_context": "Suppress pair context"}
    usefulness_body = [[labels[r["variant"]], fmt(r["fluid_mse_mean"], 6), fmt(r["fluid_mse_delta_from_normal_mean"], 6), fmt(r["prediction_rms_difference_from_normal_mean"], 5)] for r in usefulness]
    gradient_body = [[r["group"], integer(r["gradient_tensor_count"]), fmt(r["grad_norm"], 6), r["finite"]] for r in gradients]
    retained_body = [[r["policy"].replace("beta_", ""), fmt(r["selected_module_count_mean"], 3), f"{integer(r['selected_module_count_min'])}–{integer(r['selected_module_count_max'])}", fmt(r["retained_beta_mass_mean"], 8), integer(r["pair_mlp_rows_mean"])] for r in retained]
    r1401, r1402, r1403, r1404, r1804 = [float(headline[r]["relative_l2"]) for r in RUNS]
    gain = 1 - r1404 / r1401
    dense_gap = r1404 / r1804 - 1
    best_gain = 1 - float(selected["1404"]["relative_l2"]) / r1404

    markdown = f"""# HONF Stage-7 Decoder-Context Ablation Report

**Runs:** 1401, 1402, 1403, 1404, and 1804

**Primary checkpoint policy:** exact epoch 5000

**Evaluation population:** the same 90-case development holdout, all 8,192 grid queries per case

**Report date:** 2026-09-16

## Decision summary

The mature results strengthen the case for useful routed computation without overturning the decoder-context ablation. Removing global decoder context remains harmful: Run 1402 is **{pct(r1402 / r1401 - 1)}** worse than Run 1401 in pooled fluid relative L2 and loses all 90 paired cases. Removing near context as well recovers part of that damage, but Run 1403 remains **{pct(r1403 / r1401 - 1)}** worse than 1401 with a much heavier error tail.

Run 1404 changes a different mechanism. It retains global and near context, removes direct hyperedge-value context, and routes contextualized module-conditioned pair responses. At exact epoch 5000 it reaches **{r1404:.5f} pooled L2**, **{100 * gain:.1f}% better than 1401**, winning 58/90 paired cases. Its p95 is essentially unchanged and its maximum error is lower. This is a meaningful mature recovery from its poor epoch-500 result, not evidence that global context can be removed.

Dense Run 1804 remains the accuracy leader at **{r1804:.5f}** and beats Run 1404 in 84/90 cases. Run 1404 is **{100 * dense_gap:.1f}%** worse in pooled L2 but retains Stage-7-class inference cost. The practical frontier is **1804 for maximum accuracy, 1404 for the best mature routed accuracy, and 1401 as the simpler historical routed baseline.**

## Models and comparison boundary

- **1401 / Legacy:** original Stage-7 structured-context reference.
- **1402 / No global:** Run 1401 with global decoder context disabled.
- **1403 / No global/near:** global and near-module decoder contexts disabled.
- **1404 / Routing-only:** global and near context retained; direct hyperedge-value context disabled; contextualized module tokens feed the fused query-module pair path.
- **1804 / Dense:** dense pairwise field-adaptation accuracy reference, without a learned hyperedge partition.

Runs 1402 and 1403 are controlled decoder-context ablations. Run 1404 is an adjacent routing-path redesign, so its gains cannot be attributed to one context switch. Every exact checkpoint uses predicted port conditions and identical cases, masks, targets, and query counts. This is a one-seed development-holdout comparison, not independent physical-reference validation.

## Reconstruction accuracy at exact epoch 5000

{md_table(["Run", "Model", "Pooled L2", "Equal-case mean", "Median", "p95", "Maximum", "Worst case"], accuracy_body)}

Run 1404 improves over 1401 in pooled error, equal-case mean, median, and maximum. Its main gains are pressure (**0.03723 vs 0.04915**), far-fluid error (**0.03928 vs 0.04178**), and field temperature (**0.04913 vs 0.05229**). It regresses on both velocity components, normal heat flux, and final outside temperature. The aggregate gain is real but component-specific.

Dense remains stronger on pooled error and most major components. Run 1404 is slightly better on near-interface field error and vorticity, but these exceptions do not offset Dense's global advantage.

### Component and region errors

{md_table(["Metric"] + RUNS, component_body)}

### Paired outcomes against Run 1401

{md_table(["Candidate", "Wins", "Losses", "Mean case-L2 delta"], paired_body)}

Run 1404 wins 58/90 cases against 1401; Dense wins 80/90. Run 1403's 49/90 wins coexist with worse pooled error because its losses concentrate in difficult cases.

### Four established anchors

{md_table(["Case"] + RUNS, anchor_body)}

Run 1404 is worse than Run 1401 on each of these four established anchors even though it wins 58/90 cases across the full holdout. The anchors expose difficult behavior but are not representative of the population ranking. The common-scale case-0298 maps in the HTML are demonstrative rather than a substitute for the 90-case statistics.

## Convergence and checkpoint sensitivity

{md_table(["Epoch"] + RUNS, trajectory_body)}

Run 1404 is the slowest starter: its epoch-500 pooled L2 is **0.15073**, versus 0.11715 for 1401 and 0.09874 for Dense. By epoch 2500 it reaches 0.05198, still 14.8% behind 1401. Between epochs 2500 and 5000 it improves by 31.7%, overtakes 1401, and approaches the Dense frontier. Judging it at epoch 500 would therefore have produced the wrong mature ranking.

{md_table(["Run", "Parameters", "Final val field MSE", "Best val field MSE", "First val MSE ≤ 0.01", "Recorded wall h", "Peak CUDA MiB"], training_body)}

{md_table(["First validation field-MSE crossing"] + RUNS, threshold_body)}

Run 1404 reaches early validation thresholds later than 1401 and reaches 0.003 at epoch 2516, but its mature full-grid reconstruction is better. Its 5,004 raw history rows contain four duplicate epochs (851–854) from an interrupted resume; the reducer retains the final row for each epoch, yielding a consecutive 5,000-epoch series without modifying the source file.

### Validation-selected checkpoint sensitivity

{md_table(["Run", "Selected epoch", "Pooled L2", "Equal-case mean", "p95", "Maximum"], selected_body)}

Run 1404's best-field checkpoint is epoch 4890 and improves pooled L2 by only **{100 * best_gain:.1f}%** relative to its exact endpoint. It remains worse than the selected Run 1401 and Dense checkpoints. These selected results reuse the development holdout and are sensitivity evidence; exact epoch 5000 remains primary.

Recorded training wall times were collected under different continuation and machine conditions. Run 1404 records 6.16 h and 24,841 MiB peak allocation, but these values are operational history rather than controlled architecture timing.

## Controlled computation cost

{md_table(["Run", "Anchor full ms", "Anchor prepared decode ms", "M32/E768/Q65k ms", "M128/E3072/Q262k ms", "Largest allocated MiB"], timing_body)}

Run 1404 stays near the Stage-7 cost envelope: its two-anchor full-forward mean is 27.09 ms and its largest-shape median is 1840.2 ms. The largest result is 5.7% slower than the separately measured Run 1401 result, while Dense is 2.93× slower than Run 1404. Run 1404's parameter count remains 2,473,510.

Fine differences among Runs 1401–1404 are comparable to invocation variability. Incremental allocation depends on chunk scheduling and is not total process residency. Missing Stage-7 operation counters mean uninstrumented work, not zero work.

## Hyperedge organization

{md_table(["Metric"] + ROUTED, routing_body)}

### Full-holdout assignment geometry

{md_table(["Metric"] + ROUTED, topology_body)}

Run 1404 retains six fixed edges and multi-edge query routing. Its organization is neither collapsed nor sparse. Relative to 1401, its environment assignments are more diffuse and smoother, with lower region separation. Runs 1402 and 1403 show why such organization statistics cannot establish field usefulness by themselves.

## Run 1404 pathway usefulness at maturity

{md_table(["Intervention", "Fluid MSE", "Δ fluid MSE", "Prediction RMS difference"], usefulness_body)}

The mature pair pathway is causally active. Replacing contextualized module tokens with base tokens raises four-anchor mean fluid MSE from **0.001885 to 0.023640**. Suppressing pair context raises it to **1.382676**. Uniform query-to-edge and module-to-edge assignments produce smaller but adverse changes, showing that learned routing contributes beyond a nonzero pair path.

{md_table(["Gradient group", "Gradient tensors", "Gradient norm", "Finite"], gradient_body)}

The endpoint backward pass gives finite nonzero gradients to organizer scores, environment encoding, module-environment context, query-to-hyper routing, and the pair MLP. The disabled hyper-value branch has exactly zero gradient, as designed. Usefulness is supported by ground-truth error interventions and live gradients rather than activation magnitude alone.

### Retained-mass execution

{md_table(["Beta floor", "Selected modules mean", "Range", "Retained beta mass", "Pair-MLP rows/case"], retained_body)}

Every retained-mass floor from 0.98 through 1.00 selects all physically active modules. Gathered execution reduces padded pair rows from 98,304 to 47,787 per case, but this is padding removal rather than learned physical sparsity.

## Conclusions and next research decision

1. **Retain global decoder context.** Run 1402 remains a decisive negative result and offers no controlled cost benefit.
2. **Do not adopt Run 1403.** Its median and paired wins hide a severe tail and worse pooled error.
3. **Promote Run 1404 to the mature routed reference.** It beats Run 1401 on pooled error and 58/90 cases while preserving the lower-cost Stage-7 envelope.
4. **Keep Dense Run 1804 as the accuracy reference.** It remains materially more accurate, especially in the far field, at substantially higher latency.
5. **Do not claim learned execution sparsity for Run 1404.** Its useful routing affects predictions and ground-truth errors, but retained-mass thresholds keep every active module.
6. **Next experiment:** isolate whether Run 1404's remaining gap to Dense comes from fused pair-kernel field capacity or upstream context quality. Use one controlled mature change and paired full-field error; do not return to K optimization or routing-activity penalties.

## Reproducibility and artifacts

- Existing four-model evidence: `diagnostics/generated/interface_operator_study/stage7_decoder_ablation/`
- Run 1404 exact and best evaluation: `stage7_decoder_ablation/run1404_evaluation/`
- Run 1404 epoch-2500 trajectory: `stage7_decoder_ablation/run1404_trajectory_evaluation/`
- Run 1404 topology: `stage7_decoder_ablation/run1404_topology_quality_full90/`
- Run 1404 interventions and gradients: `stage7_decoder_ablation/run1404_interventions/`, `run1404_endpoint_gradient.json`
- Run 1404 retained-mass study: `stage7_decoder_ablation/run1404_retained_mass/`
- Controlled timing: `stage7_decoder_ablation/timing/four_model_timing.json`, `timing/run1404_timing.json`
- Reduced tables: `stage7_decoder_ablation/reduction/`
- Reducer and renderer: `tools/diagnostics/analyze_stage7_decoder_ablation.py`, `tools/diagnostics/render_stage7_decoder_ablation_report.py`

Evaluator manifests retain exact checkpoint paths and commands. Run 1404 checkpoint tensors and optimizer state are finite. No new CFD or physical-reference validation is claimed.
"""
    REPORT_MD.write_text(markdown, encoding="utf-8")

    figures = {"accuracy": accuracy_figure(headline, selected), "trajectory": trajectory_figure(trajectory), "timing": timing_figure(timing), "topology": topology_figure(topology), "usefulness": usefulness_figure(usefulness), "error": error_map_figure()}
    css = """:root{--ink:#17232c;--muted:#5d6b74;--paper:#f5f2eb;--card:#fff;--line:#d9d4c9;--accent:#755da8}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.58 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1180px;margin:auto;padding:38px 28px 80px}h1{font:700 clamp(32px,5vw,58px)/1.03 Georgia,serif;max-width:960px;margin:.25em 0}h2{font:700 28px/1.2 Georgia,serif;margin:2.1em 0 .55em;border-top:1px solid var(--line);padding-top:1em}h3{font:700 20px Georgia,serif}.eyebrow{letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:700;font-size:13px}.lede{font-size:20px;max-width:940px;color:#33434d}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}.card{background:var(--card);padding:18px;border:1px solid var(--line);border-radius:14px}.card b{display:block;font-size:27px}.card span{color:var(--muted);font-size:13px}.callout{background:#eee9f7;border-left:5px solid var(--accent);padding:18px 22px;border-radius:6px;margin:24px 0}.figure{background:white;border:1px solid var(--line);border-radius:14px;padding:14px;margin:24px 0}.figure img{display:block;width:100%;height:auto}.caption{font-size:13px;color:var(--muted);padding:4px 10px 8px}.table-wrap{overflow:auto;margin:18px 0 26px;background:white;border:1px solid var(--line);border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#263943;color:white;text-align:left}th,td{padding:10px 12px;border-bottom:1px solid #e8e4dc;white-space:nowrap}td:not(:first-child),th:not(:first-child){text-align:right}code{background:#ece8df;padding:2px 5px;border-radius:4px}ol{padding-left:24px}.note{color:var(--muted);font-size:14px}@media(max-width:760px){.cards{grid-template-columns:1fr 1fr}main{padding:24px 14px}.lede{font-size:17px}}"""
    html_doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HONF Stage-7 Decoder Context and Routing-Only Comparison</title><style>{css}</style></head><body><main>
    <div class="eyebrow">HONF · exact epoch-5000 comparison · 2026-09-16</div><h1>Routing-only matures past Legacy Stage-7, while Dense keeps the accuracy lead</h1>
    <p class="lede">A matched 90-case comparison of decoder-context ablations, mature Run 1404, and Dense Run 1804 across reconstruction, convergence, cost, organization, gradients, and interventions.</p>
    <div class="cards"><div class="card"><b>{r1404:.5f}</b><span>Run 1404 exact pooled L2</span></div><div class="card"><b>{100*gain:.1f}%</b><span>gain over Run 1401</span></div><div class="card"><b>58/90</b><span>wins versus Run 1401</span></div><div class="card"><b>{r1804:.5f}</b><span>Dense accuracy reference</span></div></div>
    <div class="callout"><b>Decision.</b> Retain global context. Promote Run 1404 to the mature routed reference. Keep Dense 1804 as the accuracy reference; Run 1404 still trails it in 84/90 cases.</div>
    <h2>Exact reconstruction</h2>{html_table(["Run","Model","Pooled L2","Mean","Median","p95","Maximum","Worst"],accuracy_body)}
    <div class="figure"><img src="{figures['accuracy']}" alt="Exact and selected pooled relative L2"><div class="caption">Exact epoch 5000 is primary; selected checkpoints are sensitivity evidence.</div></div>
    <p>Run 1404 improves pooled L2 by {100*gain:.1f}% and wins 58/90 cases against Run 1401. Dense remains {100*dense_gap:.1f}% better and wins 84/90.</p>
    <h3>Component and region errors</h3>{html_table(["Metric"]+RUNS,component_body)}
    <h3>Four established anchors</h3>{html_table(["Case"]+RUNS,anchor_body)}
    <div class="figure"><img src="{figures['error']}" alt="Spatial normalized errors for case 0298"><div class="caption">Case 0298 on a common scale. The decision uses all 90 cases.</div></div>
    <h2>Convergence and checkpoint sensitivity</h2>{html_table(["Epoch"]+RUNS,trajectory_body)}
    <div class="figure"><img src="{figures['trajectory']}" alt="Pooled relative L2 by epoch"><div class="caption">Run 1404 starts worst at epoch 500, then overtakes Run 1401 by epoch 5000.</div></div>
    {html_table(["Run","Parameters","Final val MSE","Best val MSE","First ≤.01","Recorded h","Peak MiB"],training_body)}
    {html_table(["First validation field-MSE crossing"]+RUNS,threshold_body)}
    <p>Run 1404 has four duplicated resume epochs, 851–854; the analysis retains one final row per epoch without modifying the run history.</p>
    <h3>Selected checkpoint sensitivity</h3>{html_table(["Run","Selected epoch","Pooled L2","Mean","p95","Maximum"],selected_body)}
    <h2>Controlled GPU cost</h2>{html_table(["Run","Anchor full ms","Prepared decode ms","M32/E768/Q65k ms","M128/E3072/Q262k ms","Largest MiB"],timing_body)}
    <div class="figure"><img src="{figures['timing']}" alt="Controlled full-forward latency"><div class="caption">Stage-7 runs occupy a similar cost band; Dense is substantially slower.</div></div>
    <h2>Hyperedge organization</h2>{html_table(["Metric"]+ROUTED,routing_body)}
    <h3>Full-holdout assignment geometry</h3>{html_table(["Metric"]+ROUTED,topology_body)}
    <div class="figure"><img src="{figures['topology']}" alt="Hyperedge organization summaries"><div class="caption">Equal-case means over 90 cases. Dense has no learned partition.</div></div>
    <h2>Run 1404 pathway usefulness</h2>{html_table(["Intervention","Fluid MSE","Δ fluid MSE","Prediction RMS difference"],usefulness_body)}
    <div class="figure"><img src="{figures['usefulness']}" alt="Run 1404 intervention errors"><div class="caption">Suppressing pair context causes a catastrophic ground-truth error increase.</div></div>
    <p>Base tokens raise fluid MSE from 0.001885 to 0.023640; removing pair context raises it to 1.382676. Active pathway groups receive finite gradients, while the disabled hyper-value branch receives none.</p>
    {html_table(["Gradient group","Gradient tensors","Gradient norm","Finite"],gradient_body)}
    <h3>Retained-mass execution</h3>{html_table(["Beta floor","Selected modules mean","Range","Retained beta mass","Pair-MLP rows/case"],retained_body)}
    <p>Every threshold retains all active modules. Gathered execution removes padding rows rather than learned physical routes.</p>
    <h2>Research decision</h2><ol><li>Retain global decoder context.</li><li>Use Run 1404 as the mature routed reference and Run 1804 as the accuracy reference.</li><li>Do not claim learned module sparsity.</li><li>Next isolate pair-kernel capacity versus upstream context quality with one mature controlled change.</li></ol>
    <p class="note">One seed; established 90-case development holdout; exact epoch 5000 primary. No new CFD or physical-reference validation is claimed. Artifacts are under <code>diagnostics/generated/interface_operator_study/stage7_decoder_ablation/</code>.</p>
    </main></body></html>"""
    REPORT_HTML.write_text(html_doc, encoding="utf-8")
    print(REPORT_MD)
    print(REPORT_HTML)


if __name__ == "__main__":
    main()
