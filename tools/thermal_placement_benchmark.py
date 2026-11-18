#!/usr/bin/env python3
"""Compare placement proposals using saved, frozen ThermalChannel predictors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from channelthermal.inverse.placement_benchmark import benchmark_placement, benchmark_placement_curve
from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from honf_runtime.compat import load_trusted_checkpoint


METHOD_NAMES = {
    "uniform": "Uniform sample",
    "proximity": "Hot-receiver proximity",
    "environment_access": "HONF environment access",
    "finite_response": "Signed finite response",
    "reference_poll": "Reference-only poll",
}
CURVE_COLORS = {
    "uniform": ("blue", "#3764a8"),
    "proximity": ("orange", "#b54f10"),
    "environment_access": ("green", "#137c60"),
    "finite_response": ("purple", "#843894"),
    "reference_poll": ("gray", "#656b72"),
}


def format_summary(results, device):
    def number(value):
        return "—" if value is None else f"{value:.4g}"

    lines = ["# ThermalChannel placement replay", "",
             f"Device `{device}`. Each model policy uses the same per-family forward budget, including its probes. "
             "The shared baseline forward is counted separately. The reference-only poll uses zero forwards.", "",
             "Larger cooling is better. Smaller losses are better. Cooling and losses use native temperature units. "
             "Reference checks read supplied labels for recommended edits. An em dash indicates unavailable labels.", ""]
    for checkpoint, result in results.items():
        epoch = result["epoch"]
        lines.extend([f"## {checkpoint}", "",
                      f"Checkpoint epoch {epoch if epoch is not None else 'unavailable'}. "
                      f"Read mode `{result['mode']}`. Seed {result['seed']}. "
                      f"Up to {result['budget']} forwards. Pressure rise cap `{result['pressure_allowance']:g}`. "
                      f"Minimum resolved decrease in J `{result['j_resolution']:g}`. J smoothing scale `{result['tau']:g}`.", ""])
        for role, methods in result["by_role"].items():
            families = methods["uniform"]["family_count"]
            labeled = methods["uniform"]["reference_family_count"]
            lines.extend([f"### {role}", "", f"Families evaluated {families}. Families with reference labels {labeled}.", "",
                          "| Policy | Mean forwards | Mean probes | Better designs | Over cap / checked | Mean cooling | Missed by shortlist | Lost after shortlist |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
            for method, label in METHOD_NAMES.items():
                row = methods[method]
                accepted = f"{row['resolved_improvements']}/{labeled}" if labeled else "—"
                checks = row["reference_checks"]
                violated = f"{row['selected_over_cap']}/{checks}" if labeled else "—"
                lines.append(f"| {label} | {number(row['mean_candidate_scores'])} | {number(row['mean_probe_scores'])} "
                             f"| {accepted} | {violated} | {number(row['mean_verified_improvement'])} "
                             f"| {number(row['mean_shortlist_loss'])} | {number(row['mean_decision_loss'])} |")
            lines.append("")
            if "full_pool" in methods:
                diagnostic = methods["full_pool"]
                accuracy = diagnostic["response_accuracy"]
                lines.extend([f"**Full-pool predictor check.** Mean {number(diagnostic['mean_candidate_scores'])} "
                              "additional forwards per family, outside policy budgets.", ""])
                if accuracy is not None:
                    lines.extend([f"Correct direction on {accuracy['correct_j_sign']}/{accuracy['resolved_j_edits']} "
                                  f"resolved temperature changes. Correct ordering on "
                                  f"{accuracy['correct_feasible_order']}/{accuracy['resolved_feasible_pairs']} "
                                  "resolved pairs of reference-feasible edits. Mean family pressure error "
                                  f"{number(accuracy['mean_abs_pressure_error'])} in native pressure units. "
                                  f"Cooling when every edit was scored {number(diagnostic['mean_verified_improvement'])}.", ""])
    lines.append("Missed by shortlist compares its best reference-feasible edit with the pool's best. "
                 "Lost after shortlist compares the retained design with the best in its shortlist. Both include the baseline.")
    return "\n".join(lines)


def format_curve_summary(results, device):
    def number(value):
        return "—" if value is None else f"{abs(value) if value == 0 else value:g}"

    lines = ["# ThermalChannel placement yield curves", "",
             f"Device `{device}`. Each budget starts a separate replay from the same baseline and seed. "
             "Cooling is the reference decrease in J of the retained layout. The reference-only poll uses zero "
             "candidate forwards at every budget. Axes match across checkpoints for each family role. "
             "Sampled budgets are shown as equally spaced checkpoints.", ""]
    max_cooling = {}
    for result in results.values():
        for role, curves in result["curves"].items():
            values = (point["mean_verified_improvement"] for points in curves.values() for point in points)
            max_cooling[role] = max(max_cooling.get(role, 0.), max((v for v in values if v is not None), default=0.))
    for checkpoint, result in results.items():
        run = result["runs"][-1]
        lines.extend([f"## {checkpoint}", "", f"Seed {run['seed']}. Read mode `{run['mode']}`. "
                      f"Pressure rise cap `{run['pressure_allowance']:g}`. J resolution `{run['j_resolution']:g}`. "
                      f"J smoothing scale `{run['tau']:g}`.", ""])
        budgets = result["budgets"]
        for role, curves in result["curves"].items():
            labeled = curves["uniform"][0]["reference_family_count"]
            lines.extend([f"### {role}", "", f"Families with reference labels {labeled}/{curves['uniform'][0]['family_count']}.", ""])
            if labeled:
                palette = ",".join(value[1] for value in CURVE_COLORS.values())
                lines.extend(["```mermaid", f'%%{{init: {{"themeVariables": {{"xyChart": {{"plotColorPalette": "{palette}"}}}}}}}}%%',
                              "xychart-beta", f'    x-axis "Candidate-forward budget" {json.dumps([str(v) for v in budgets])}',
                              f'    y-axis "Mean cooling in native units" 0 --> {max_cooling[role] or 1.:g}'])
                for method in METHOD_NAMES:
                    values = ", ".join(number(point["mean_verified_improvement"]) for point in curves[method])
                    lines.append(f"    line [{values}]")
                lines.extend(["```", ""])
            lines.extend(["| Policy | " + " | ".join(str(v) for v in budgets) + " |",
                          "| --- | " + " | ".join("---:" for _ in budgets) + " |"])
            for method, name in METHOD_NAMES.items():
                values = " | ".join(number(point["mean_verified_improvement"]) for point in curves[method])
                lines.append(f"| {name} ({CURVE_COLORS[method][0]}) | {values} |")
            lines.append("")
            steps = [step for step in result["critical_steps"] if step["role"] == role]
            if steps:
                lines.extend(["<details>", "<summary>Critical decisions by family</summary>", "",
                              "| Budget | Family | Policy | Event | Recommended → retained | Proposed centers | Cooling | Reference pressure rise |",
                              "| ---: | --- | --- | --- | --- | --- | ---: | ---: |"])
                for step in steps:
                    event = ", ".join(step["events"])
                    pressure = (step["reference_delta"] or {}).get("P")
                    centers = " · ".join(f"{move['slot']} to ({number(move['to'][0])}, {number(move['to'][1])})"
                                        for move in step["moves"])
                    lines.append(f"| {step['budget']} | {step['family']} | {METHOD_NAMES[step['policy']]} | {event} | "
                                 f"{step['recommended']} → {step['retained'] or '—'} | {centers or '—'} | "
                                 f"{number(step['verified_improvement'])} | {number(pressure)} |")
                lines.extend(["", "</details>", ""])
            full_pool = run["by_role"][role].get("full_pool")
            if full_pool is not None:
                lines.extend([f"The separate full-pool check at budget {budgets[-1]} used "
                              f"{number(full_pool['mean_candidate_scores'])} additional forwards per family on average. "
                              "Its predictor metrics are included in the JSON.", ""])
    return "\n".join(lines)


def restore_checkpoint(path, device):
    saved = load_trusted_checkpoint(path, map_location="cpu")
    adapter = saved.get("adapter_state")
    if adapter is None:
        adapter = saved["experiment_identity"]["provider_identity"]["adapter_state"]
    model = ThermalDetailAnchoredMemberRoutingAdapter.from_state_payload(
        adapter, state_dict=saved["model_state_dict"], device=device,
    )
    return model.requires_grad_(False).eval(), saved.get("epoch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    budget_options = parser.add_mutually_exclusive_group(required=True)
    budget_options.add_argument("--budget", type=int, help="Maximum candidate forwards per model policy and family, including probes.")
    budget_options.add_argument("--budgets", type=int, nargs="+", metavar="N", help="Increasing forward budgets for a matched yield curve.")
    parser.add_argument("--tau", type=float, required=True, help="Native material smooth-peak temperature scale.")
    parser.add_argument("--pressure-allowance", type=float, required=True, help="Maximum pressure rise from the baseline.")
    parser.add_argument("--j-resolution", type=float, required=True, help="Minimum resolved reference decrease in J.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("restricted", "full"), default="restricted")
    parser.add_argument("--full-pool", action="store_true", help="Score all edits separately for a matched predictor diagnostic.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json", help="Detailed JSON or a readable summary.")
    args = parser.parse_args(argv)
    panel = json.loads(args.panel.read_text())
    results = {}
    for item in args.checkpoint:
        label, separator, path = item.partition("=")
        if not separator or not label or not path or label in results:
            parser.error("Each checkpoint must use a distinct LABEL=PATH.")
        checkpoint = Path(path).expanduser().resolve(strict=True)
        model, epoch = restore_checkpoint(checkpoint, args.device)
        settings = {"pressure_allowance": args.pressure_allowance, "tau": args.tau,
                    "resolution": args.j_resolution, "seed": args.seed, "device": args.device,
                    "mode": args.mode, "full_pool": args.full_pool}
        if args.budgets is None:
            result = benchmark_placement(model, panel, budget=args.budget, **settings)
        else:
            result = benchmark_placement_curve(model, panel, budgets=args.budgets, **settings)
        results[label] = {"checkpoint": str(checkpoint), "epoch": epoch, **result}
        del model
    if args.format == "markdown":
        formatter = format_summary if args.budgets is None else format_curve_summary
        print(formatter(results, args.device))
    else:
        print(json.dumps({"panel": str(args.panel.resolve()), "device": args.device, "checkpoints": results},
                         indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
