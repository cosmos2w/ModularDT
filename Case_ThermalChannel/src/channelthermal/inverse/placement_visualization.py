"""Matched native fluid-temperature fields alongside a placement budget sweep."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from honf_runtime.state_response import field_metrics

from ..interaction_evidence.reference_adapter import operating_context_from_config
from ..native_design import build_native_design, score_native_design
from ..training.thermal_response_raw import load_raw_endpoint
from .placement_benchmark import BASELINE, MODEL_POLICIES, REFERENCE_POLL, _region_bounds


def _load_temperature_reference(config, path):
    if not path or not Path(path).is_dir():
        return None
    raw = load_raw_endpoint(path)
    saved = raw["config"]
    if (any(saved["domain"][key] != config["domain"][key] for key in ("nx", "ny"))
            or operating_context_from_config(saved).values != operating_context_from_config(config).values
            or any(saved["thermal"][key] != config["thermal"][key] for key in ("t_in", "t_wall"))
            or any(not np.array_equal(saved["layout"][key], config["layout"][key])
                   for key in ("centers", "heat_powers"))):
        raise ValueError(f"Reference field {path} does not match the requested grid, layout and controls.")
    return raw["field"][..., 4], raw["fluid_mask"]


@torch.inference_mode()
def collect_placement_fields(model, panel, curve, *, case_id, policy, budget, device="cpu"):
    """Rescore only the baseline and the recommendation for a labeled curve point."""
    run = next(row for row in curve["runs"] if row["budget"] == budget)
    family = run["cases"][case_id]
    arm = family[REFERENCE_POLL] if policy == REFERENCE_POLL else family["methods"][policy]
    decision = arm["decision"]
    action = decision.get("polled_action") or decision["model_selected_action"] or BASELINE
    case = next(row for row in panel["cases"] if row["case_id"] == case_id)
    config = case["source_config"]
    endpoints = {BASELINE: (config, case.get("source_case_dir"))}
    if action != BASELINE:
        trial = next(row for row in panel["actions"] if row["case_id"] == case_id and row["action"] == action)
        edited = {**config, "layout": {**config["layout"], "centers": trial["centers"]}}
        endpoints[action] = edited, trial.get("raw_case_dir")
    fields = {}
    for name, (endpoint, path) in endpoints.items():
        design = build_native_design(endpoint, case_id, device=device)
        scored = score_native_design(model, design, tau=run["tau"], mode=run["mode"])
        fluid = scored["fields"]["temperature/fluid"]
        shape = (design.ny, design.nx)
        prediction = fluid["prediction"].cpu().numpy().reshape(shape)
        mask = fluid["mask"].cpu().numpy().reshape(shape)
        reference = _load_temperature_reference(endpoint, path)
        if reference is not None:
            values, valid = reference
            if values.shape != shape or valid.shape != shape or not np.array_equal(valid, design.fluid_valid.reshape(shape)):
                raise ValueError(f"Reference field {path} does not match the native grid and physical fluid mask.")
        fields[name] = {"prediction": prediction, "mask": mask, "reference": reference,
                        "centers": endpoint["layout"]["centers"]}
    return {"case_id": case_id, "role": family["role"] or "unspecified", "action": action,
            "retained": decision.get("verified_incumbent"), "outcome": decision["reference_outcome"],
            "fields": fields, "extra_forwards": len(fields), "domain": config["domain"], "regions": run["regional_caps"]}


def plot_placement_fields_and_curve(curve, evidence, *, budget, policy, checkpoint, units="dataset temperature units"):
    """Build a figure in memory with a shared field scale and actual curve budgets."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle

    figure = Figure(figsize=(12, 9.2), layout="constrained")
    grid = figure.add_gridspec(3, 3, width_ratios=[1, 1, .05], height_ratios=[1, 1, 1.15])
    axes = {name: figure.add_subplot(grid[row, col]) for name, row, col in
            (("baseline", 0, 0), ("proposal", 0, 1), ("reference", 1, 0), ("error", 1, 1))}
    axes["curve"] = figure.add_subplot(grid[2, :])
    domain = evidence["domain"]
    extent = [0, domain["lx"], 0, domain["ly"]]
    baseline = evidence["fields"][BASELINE]
    proposal = evidence["fields"][evidence["action"]]
    field_values = []
    for field in (baseline, proposal):
        field_values.append(field["prediction"][field["mask"]])
        if field["reference"] is not None:
            values, valid = field["reference"]
            field_values.append(values[valid & field["mask"]])
    temperature_scale = np.concatenate(field_values)
    low, high = float(temperature_scale.min()), float(temperature_scale.max())

    def draw(name, field, values, valid, title, **colors):
        axis = axes[name]
        plotted = axis.imshow(np.ma.array(values, mask=~valid), origin="lower", extent=extent,
                              interpolation="nearest", **colors)
        for slot, (x, y) in enumerate(field["centers"]):
            axis.add_patch(Circle((x, y), domain["module_radius"], ec="#45515c", fc="#e2e6eb", lw=.8))
            axis.text(x, y, str(slot), ha="center", va="center", fontsize=8, color="#25364a")
        for index, (label, region) in enumerate(evidence["regions"].items()):
            x0, y0, x1, y1 = _region_bounds(domain, region)
            outline = ("#d45087", "#2271a5", "#986919")[index % 3]
            axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=outline, ls="--", lw=1.5))
            label_y = y1 - .025 * extent[3]
            if x1 - x0 < .08 * extent[1] and y1 - y0 > .5 * extent[3]:
                label_y = (y0 + y1) / 2
            axis.text(x1 - .025 * extent[1], label_y, label, ha="right", va="top", fontsize=8,
                      color="#25364a", bbox={"facecolor": "white", "alpha": .86, "edgecolor": "none", "pad": 1.5})
        axis.set(xlabel="x (length units)", ylabel="y (length units)", title=title)
        return plotted

    def rmse_label(field):
        if field["reference"] is None:
            return "Error unavailable"
        values, valid = field["reference"]
        metrics = field_metrics(field["prediction"], values, field["mask"] & valid)
        return f"Fluid RMSE {metrics['rmse']:.3g} {units}"

    colors = {"cmap": "inferno", "vmin": low, "vmax": high}
    temperature = draw("baseline", baseline, baseline["prediction"], baseline["mask"],
                       f"Baseline prediction\n{rmse_label(baseline)}", **colors)
    candidate = evidence["action"]
    error_label = "Error unavailable" if proposal["reference"] is None else ""
    draw("proposal", proposal, proposal["prediction"], proposal["mask"],
         f"Recommended prediction · {candidate}" + (f"\n{error_label}" if error_label else ""), **colors)
    if proposal["reference"] is not None:
        values, valid = proposal["reference"]
        common = proposal["mask"] & valid
        draw("reference", proposal, values, common, f"Reference field · {candidate}", **colors)
        delta = proposal["prediction"] - values
        title = f"Prediction minus reference\n{rmse_label(proposal)}"
    else:
        axes["reference"].set_axis_off()
        axes["reference"].text(.5, .5, "Matched field reference unavailable", ha="center", va="center",
                               transform=axes["reference"].transAxes)
        common = proposal["mask"] & baseline["mask"]
        delta = proposal["prediction"] - baseline["prediction"]
        title = "Predicted change from baseline"
    scale = max(float(np.max(np.abs(delta[common]))), 1e-9)
    difference = draw("error", proposal, delta, common, title, cmap="RdBu_r", vmin=-scale, vmax=scale)
    figure.colorbar(temperature, cax=figure.add_subplot(grid[0, 2]), label=units)
    figure.colorbar(difference, cax=figure.add_subplot(grid[1, 2]), label=units)

    chart = axes["curve"]
    curves = curve["curves"][evidence["role"]]
    for method in (*MODEL_POLICIES, REFERENCE_POLL):
        points = curves[method]
        values = [point["mean_verified_improvement"] for point in points]
        if any(value is not None for value in values):
            line, = chart.plot(curve["budgets"], values, marker="o", label=method.replace("_", " "),
                               lw=2.5 if method == policy else 1.3, ls="--" if method == REFERENCE_POLL else "-")
            if method == policy:
                event_budgets = {step["budget"] for step in curve["critical_steps"]
                                 if step["family"] == evidence["case_id"] and step["policy"] == policy}
                markers = [(point["budget"], point["mean_verified_improvement"]) for point in points
                           if point["budget"] in event_budgets and point["mean_verified_improvement"] is not None]
                if markers:
                    xs, ys = zip(*markers)
                    chart.scatter(xs, ys, marker="D", s=55, facecolor="white", edgecolor=line.get_color(), zorder=4)
                inspected = next(point["mean_verified_improvement"] for point in points if point["budget"] == budget)
                if inspected is not None:
                    chart.scatter(budget, inspected, s=100, facecolor="none", edgecolor=line.get_color(), zorder=5)
    chart.axvline(budget, color="#687685", lw=1, ls=":")
    chart.set(xlabel="Allowed candidate forwards per family", ylabel=f"Mean retained cooling\n({units})",
              title=f"Budget {budget} · {policy.replace('_', ' ')} · {evidence['outcome']}", xticks=curve["budgets"])
    chart.grid(axis="y", alpha=.25)
    if chart.get_legend_handles_labels()[0]:
        chart.legend(ncols=3, loc="best", frameon=False, fontsize=8)
    else:
        chart.text(.5, .5, "Reference-verified cooling unavailable", ha="center", transform=chart.transAxes)
    figure.suptitle(f"{checkpoint} · {evidence['role']} · {evidence['case_id']}\n"
                    f"Recommended {candidate} · retained {evidence['retained'] or 'unverified'}", fontsize=14)
    return figure
