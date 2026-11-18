#!/usr/bin/env python3
"""Compare placement proposals using saved, frozen ThermalChannel predictors."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

from channelthermal.inverse.placement_benchmark import benchmark_placement, benchmark_placement_curve
from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.storage_policy import verify_storage


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
        _append_checkpoint_grid(lines, result)
        for name, region in result.get("regional_caps", {}).items():
            lines.extend([f"Regional cap `{name}` uses {region['reduction']} over `{region['bounds']}` "
                          f"with allowed temperature rise `{region['allowance']:g}`.", ""])
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
        epoch = result.get("epoch")
        lines.extend([f"## {checkpoint}", "", f"Checkpoint epoch {epoch if epoch is not None else 'unavailable'}. "
                      f"Seed {run['seed']}. Read mode `{run['mode']}`. "
                      f"Pressure rise cap `{run['pressure_allowance']:g}`. J resolution `{run['j_resolution']:g}`. "
                      f"J smoothing scale `{run['tau']:g}`.", ""])
        _append_checkpoint_grid(lines, result)
        for name, region in run.get("regional_caps", {}).items():
            lines.extend([f"Regional cap `{name}` uses {region['reduction']} over `{region['bounds']}` "
                          f"with allowed temperature rise `{region['allowance']:g}`.", ""])
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


def _append_checkpoint_grid(lines, result):
    identity = result.get("checkpoint_identity")
    if identity is None:
        return
    grid = " × ".join(map(str, identity["native_grid_nx_ny"]))
    lines.extend([f"Native grid {grid}. R16-10 arm {identity['r16_arm'] or 'unavailable'}.", ""])


def _read_checkpoint(path):
    saved = load_trusted_checkpoint(path, map_location="cpu")
    adapter = saved.get("adapter_state")
    if adapter is None:
        adapter = saved["experiment_identity"]["provider_identity"]["adapter_state"]
    return saved, adapter


def _checkpoint_identity(saved, adapter, panel):
    experiment = saved.get("experiment_identity", {})
    provider = experiment.get("provider_identity", {})
    recipe = experiment.get("recipe", {})
    initialization = recipe.get("initialization", {})
    responses = provider.get("r16_10_family", {})
    train_ids = provider.get("training_case_ids")
    validation_ids = provider.get("validation_case_ids")
    response_ids = responses.get("eligible_families")
    saved_grid = adapter.get("native_grid_nx_ny")
    environment_grid = adapter.get("environment_grid_nx_ny")
    grid = list(saved_grid) if isinstance(saved_grid, (list, tuple)) else None
    identity = {
        "epoch": saved.get("epoch"), "run_name": recipe.get("name"),
        "r16_arm": recipe.get("r16_10_family", {}).get("arm", responses.get("arm")),
        "initialization_kind": initialization.get("source_kind"),
        "source_parent_epoch": initialization.get("source_parent_age"),
        "adapter_schema": adapter.get("schema"),
        "native_grid_nx_ny": grid,
        "environment_grid_nx_ny": list(environment_grid) if isinstance(environment_grid, (list, tuple)) else None,
        "saved_normalization_id": provider.get("normalization_stats_sha256"),
        "training_case_count": None if train_ids is None else len(train_ids),
        "validation_case_count": None if validation_ids is None else len(validation_ids),
        "eligible_response_family_counts": None if response_ids is None else {
            name: len(ids) for name, ids in response_ids.items()
        },
    }
    if panel is None:
        return identity

    training = None if train_ids is None else set(map(str, train_ids))
    validation = None if validation_ids is None else set(map(str, validation_ids))
    eligible = None if response_ids is None else {key: set(map(str, ids)) for key, ids in response_ids.items()}
    identity["panels"] = []
    for case in panel["cases"]:
        case_id = str(case["case_id"])
        source_id = str(case.get("source_case_id", case_id))
        family_id = str(case.get("family_id", case_id))
        domain = case["source_config"]["domain"]
        shape = [int(domain[axis]) if axis in domain else None for axis in ("nx", "ny")]
        identity["panels"].append({
            "case_id": case_id, "source_case_id": source_id, "family_id": family_id, "role": case.get("role"),
            "native_grid_nx_ny": shape, "grid_matches": None if grid is None or None in shape else shape == grid,
            "recorded_training_case": None if training is None else source_id in training,
            "recorded_validation_case": None if validation is None else source_id in validation,
            "recorded_response_groups": None if eligible is None else [group for group, ids in eligible.items() if family_id in ids],
        })
    return identity


def restore_checkpoint(path, device, *, checkpoint_state=None):
    saved, adapter = _read_checkpoint(path) if checkpoint_state is None else checkpoint_state
    model = ThermalDetailAnchoredMemberRoutingAdapter.from_state_payload(
        adapter, state_dict=saved["model_state_dict"], device=device,
    )
    return model.requires_grad_(False).eval(), saved.get("epoch")


def _figure_storage(path):
    root = Path("/data2/wanglz/ModularDT")
    limit = 16 * 1024**2
    path = path.expanduser().resolve()
    if path.suffix.lower() not in {".pdf", ".png"} or not path.is_relative_to(root):
        raise ValueError("Use a PDF or PNG destination under /data2/wanglz/ModularDT.")
    if path.exists():
        raise FileExistsError(path)
    storage = {"output_root": path.parent, "minimum_free_headroom_bytes": 1024**3,
               "max_output_bytes": limit, "max_peak_temporary_bytes": 0}
    snapshot = verify_storage({"storage": storage}, output=path, estimated_bytes=limit,
                              tree_bytes=lambda _: 0)
    return path, snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--inspect-checkpoints", action="store_true", help="Report saved checkpoint identities as JSON without scoring.")
    budget_options = parser.add_mutually_exclusive_group()
    budget_options.add_argument("--budget", type=int, help="Maximum candidate forwards per model policy and family, including probes.")
    budget_options.add_argument("--budgets", type=int, nargs="+", metavar="N", help="Increasing forward budgets for a matched yield curve.")
    parser.add_argument("--tau", type=float, help="Native material smooth-peak temperature scale.")
    parser.add_argument("--pressure-allowance", type=float, help="Maximum pressure rise from the baseline.")
    parser.add_argument("--j-resolution", type=float, help="Minimum resolved reference decrease in J.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("restricted", "full"), default="restricted")
    parser.add_argument("--full-pool", action="store_true", help="Score all edits separately for a matched predictor diagnostic.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json", help="Detailed JSON or a readable summary.")
    parser.add_argument("--figure-out", type=Path, help="Save a combined field and curve PDF or PNG on DATA2.")
    parser.add_argument("--figure-family", help="Source family to show on the field panels.")
    parser.add_argument("--figure-policy", choices=tuple(METHOD_NAMES), default="finite_response")
    parser.add_argument("--figure-budget", type=int, help="Budget point for the fields. Defaults to the largest budget.")
    args = parser.parse_args(argv)
    if args.inspect_checkpoints:
        if args.budget is not None or args.budgets is not None or args.figure_out is not None:
            parser.error("Checkpoint inspection takes no budgets or figure output.")
    elif (any(value is None for value in (args.panel, args.tau, args.pressure_allowance, args.j_resolution))
          or (args.budget is None and args.budgets is None)):
        parser.error("Scoring requires --panel, --budget or --budgets, --tau, --pressure-allowance and --j-resolution.")
    panel = None if args.panel is None else json.loads(args.panel.read_text())
    if args.figure_out is not None:
        if args.budgets is None or len(args.checkpoint) != 1 or args.figure_family is None:
            parser.error("A field figure needs --budgets, one --checkpoint and --figure-family.")
        figure_budget = args.figure_budget if args.figure_budget is not None else max(args.budgets)
        if figure_budget not in args.budgets or not any(row["case_id"] == args.figure_family for row in panel["cases"]):
            parser.error("The figure family and budget must occur in the requested comparison.")
        figure_path, figure_storage = _figure_storage(args.figure_out)
    results = {}
    for item in args.checkpoint:
        label, separator, path = item.partition("=")
        if not separator or not label or not path or label in results:
            parser.error("Each checkpoint must use a distinct LABEL=PATH.")
        checkpoint = Path(path).expanduser().resolve(strict=True)
        saved, adapter = _read_checkpoint(checkpoint)
        identity = _checkpoint_identity(saved, adapter, panel)
        if args.inspect_checkpoints:
            results[label] = {"checkpoint": str(checkpoint), **identity}
            del saved, adapter
            continue
        model, epoch = restore_checkpoint(checkpoint, args.device, checkpoint_state=(saved, adapter))
        del saved, adapter
        settings = {"pressure_allowance": args.pressure_allowance, "tau": args.tau,
                    "resolution": args.j_resolution, "seed": args.seed, "device": args.device,
                    "mode": args.mode, "full_pool": args.full_pool}
        if args.budgets is None:
            result = benchmark_placement(model, panel, budget=args.budget, **settings)
        else:
            result = benchmark_placement_curve(model, panel, budgets=args.budgets, **settings)
        if args.figure_out is not None:
            from channelthermal.inverse.placement_visualization import collect_placement_fields, plot_placement_fields_and_curve

            evidence = collect_placement_fields(model, panel, result, case_id=args.figure_family,
                                                policy=args.figure_policy, budget=figure_budget, device=args.device)
            epoch_label = f"epoch {epoch}" if epoch is not None else "epoch unavailable"
            figure = plot_placement_fields_and_curve(result, evidence, budget=figure_budget,
                                                     policy=args.figure_policy,
                                                     checkpoint=f"{label} · {epoch_label} · {model.nx} × {model.ny}")
            payload = BytesIO()
            figure.savefig(payload, format=figure_path.suffix[1:].lower(), dpi=140)
            if payload.tell() > figure_storage["output_budget_bytes"]:
                raise RuntimeError("Figure exceeds its verified DATA2 byte budget.")
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            with figure_path.open("xb") as stream:
                stream.write(payload.getbuffer())
            result["figure"] = {"path": str(figure_path), "family": args.figure_family, "policy": args.figure_policy,
                                "budget": figure_budget, "extra_forwards": evidence["extra_forwards"],
                                "storage": figure_storage, "bytes": payload.tell()}
        results[label] = {"checkpoint": str(checkpoint), "epoch": epoch,
                          "checkpoint_identity": identity, **result}
        del model
    if args.inspect_checkpoints:
        print(json.dumps({"panel": None if args.panel is None else str(args.panel.resolve()), "checkpoints": results},
                         indent=2, allow_nan=False))
    elif args.format == "markdown":
        formatter = format_summary if args.budgets is None else format_curve_summary
        print(formatter(results, args.device))
        if args.figure_out is not None:
            print(f"Field and curve figure saved to `{figure_path}` using {evidence['extra_forwards']} display-only forwards.")
    else:
        print(json.dumps({"panel": str(args.panel.resolve()), "device": args.device, "checkpoints": results},
                         indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
