"""Compare fixed-budget proposals for layout edits with one frozen native scorer."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from honf_inverse_core.contracts import PhysicalDesign
from honf_inverse_core.request_schema import GeometryConstraints

from ..interaction_evidence.reference_adapter import operating_context_from_config, operating_context_to_named
from ..native_design import build_native_design, score_native_design
from ..physical_response import evaluate_thermal_candidate_pool
from .geometry import evaluate_geometry


BASELINE = "baseline"
FUNCTIONALS = ("J", "P")
METHODS = ("uniform", "proximity", "environment_access")


def _hot_material_weights(scored, tau):
    material = scored["fields"]["temperature/material"]
    stencil = scored["prepared"].stencils["material"]
    valid = material["mask"].reshape(stencil.valid.shape)
    values = material["prediction"].reshape(valid.shape).to(torch.float64)
    weights = torch.softmax(values.masked_fill(~valid, -torch.inf) / tau, dim=-1)
    return weights[0], stencil


def _module_relevance(design, scored, tau, mode, method):
    """Use the same smooth-peak receivers for the spatial and directory controls."""
    weights, stencil = _hot_material_weights(scored, tau)
    if method == "proximity":
        xyz = torch.as_tensor(design.material_xy_source, device=weights.device).reshape(-1, 2)
        centers = design.structure["module_centers"][0].to(xyz)
        radius = design.structure["material_params"][0, -1]
        relevance = weights @ torch.exp(-torch.cdist(xyz, centers).square() / (2 * radius.square()))
        return relevance.cpu().numpy()

    prepared = scored["prepared"]
    query = prepared.query_batch
    directory = getattr(prepared.organization, "e_access_directory", None)
    if directory is None or query.anchor_indices is None:
        raise ValueError("Environment guidance requires the native E-address directory and query cover.")
    union_weights = weights.new_zeros(query.num_queries)
    union_weights.scatter_add_(0, stencil.indices[0].flatten(), (weights[:, None] * stencil.weights[0]).flatten())
    address_weights = weights.new_zeros(directory.environment_count)
    contribution = union_weights[:, None] * query.anchor_weights[0]
    active = contribution > 0
    address_weights.scatter_add_(0, query.anchor_indices[0][active].long(), contribution[active])

    rows = getattr(directory, mode)
    sizes = rows.offsets[0].diff().to(weights.device)
    row_ids = torch.repeat_interleave(torch.arange(directory.environment_count, device=weights.device),
                                    sizes, output_size=rows.member_index.numel())
    module = rows.is_module
    slot = rows.member_index[module]
    values = address_weights[row_ids[module]] * rows.access_weight[module]
    relevance = weights.new_zeros(directory.module_count)
    relevance.scatter_add_(0, slot, values)
    return relevance.cpu().numpy()


def _family_actions(case, rows):
    config = case["source_config"]
    domain, layout = config["domain"], config["layout"]
    radius = float(domain["module_radius"])
    count = int(layout["num_modules"])
    context = operating_context_to_named(operating_context_from_config(config))
    constraints = GeometryConstraints(
        module_count_min=count, module_count_max=count, minimum_center_distance=2 * radius,
        wall_clearance=0., inlet_clearance=0., outlet_clearance=0.,
    )
    heat = layout["heat_powers"]
    original = np.asarray(layout["centers"], dtype=np.float64)
    configs = {BASELINE: config}
    edits = {}
    physical = {BASELINE: dict.fromkeys(FUNCTIONALS, 0.0)}
    for row in rows:
        action = row["action"]
        if not isinstance(action, str) or not action or action in configs:
            raise ValueError(f"Expected a unique nonempty edit label for {case['case_id']}.")
        if "role" in row and "role" in case and row["role"] != case["role"]:
            raise ValueError(f"{action} and its baseline have different panel roles.")
        centers = np.asarray(row["centers"], dtype=np.float64)
        if centers.shape != original.shape or not np.isfinite(centers).all():
            raise ValueError(f"{action} must preserve the ordered original module slots.")
        movement = np.linalg.norm(centers - original, axis=-1)
        if not movement.sum() > 0:
            raise ValueError(f"{action} has no finite layout edit.")
        edits[action] = movement / movement.sum()
        configs[action] = {**config, "layout": {**layout, "centers": centers.tolist()}}
        labels = row.get("physical_delta")
        if labels is not None:
            if set(labels) != set(FUNCTIONALS) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in labels.values()
            ):
                raise ValueError(f"{action} needs finite reference deltas for J and P.")
            physical[action] = labels

    if not edits or len(physical) not in (1, len(configs)):
        raise ValueError(f"{case['case_id']} needs edits and either complete or absent reference deltas.")
    for action, endpoint in configs.items():
        design = PhysicalDesign(endpoint["layout"]["centers"], np.ones(count), heat)
        if not evaluate_geometry(design, context, constraints).valid:
            raise ValueError(f"{case['case_id']}/{action} violates disk separation or channel bounds.")
    return configs, edits, physical if len(physical) > 1 else None


def _functionals(scored):
    return {name: float(scored["functionals"][name]) for name in FUNCTIONALS}


def _material_max(scored):
    material = scored["fields"]["temperature/material"]
    return float(material["prediction"].masked_fill(~material["mask"], -torch.inf).amax())


def _evaluate_shortlist(predicted, physical, *, pressure_allowance, resolution):
    scores = evaluate_thermal_candidate_pool(
        predicted, None if physical is None else {name: physical[name] for name in predicted},
        objective="J", constraint_caps={"P": pressure_allowance}, objective_scale=1.0,
        sensitivity={"J": resolution / 4., "P": 0.},
    )
    if physical is None:
        return scores
    accepted = scores["accepted_action"] if scores["numerically_resolved_improvement"] else BASELINE
    best_delta = min(row["J"] for row in physical.values() if row["P"] <= pressure_allowance)
    scores.update(verified_incumbent=accepted, verified_improvement=-physical[accepted]["J"],
                  full_pool_regret=physical[accepted]["J"] - best_delta)
    return scores


def _summarize(cases):
    by_role = defaultdict(list)
    for row in cases.values():
        by_role[row["role"] or "unspecified"].append(row)
    summaries = {}
    for role, families in by_role.items():
        summaries[role] = {}
        for method in METHODS:
            rows = [family["methods"][method] for family in families]
            measured = [row["decision"] for row in rows if "verified_improvement" in row["decision"]]
            summaries[role][method] = {
                "family_count": len(rows),
                "reference_family_count": len(measured),
                "mean_candidate_scores": mean(row["candidate_scores"] for row in rows),
                "mean_verified_improvement": mean(row["verified_improvement"] for row in measured) if measured else None,
                "resolved_improvements": sum(row["verified_incumbent"] != BASELINE for row in measured) if measured else None,
                "false_feasible_selections": sum(row["raw_constraint_violation"] for row in measured) if measured else None,
                "mean_full_pool_regret": mean(row["full_pool_regret"] for row in measured) if measured else None,
            }
    return summaries


@torch.inference_mode()
def benchmark_placement(model, panel, *, budget, pressure_allowance, tau, resolution, seed=0,
                        device="cpu", mode="restricted"):
    """Benchmark shortlists on the same explicit position-edit pools and reference labels."""
    if (not isinstance(budget, int) or isinstance(budget, bool) or budget < 1
            or not all(math.isfinite(v) for v in (pressure_allowance, tau, resolution))
            or pressure_allowance < 0 or tau <= 0 or resolution < 0):
        raise ValueError("Specify a positive edit budget and tau and nonnegative pressure allowance and resolution.")
    if mode not in ("restricted", "full"):
        raise ValueError("Read mode must be restricted or full.")
    if panel.get("schema") != "thermal_response_families_v1":
        raise ValueError("Expected a thermal_response_families_v1 panel.")
    if "tau" in panel and panel["tau"] != tau:
        raise ValueError("Panel reference deltas use a different J smoothing scale.")
    cases = {row["case_id"]: row for row in panel["cases"]}
    if not cases or len(cases) != len(panel["cases"]) or any(not isinstance(k, str) or not k for k in cases):
        raise ValueError("Panel cases need unique nonempty string IDs.")
    rows = defaultdict(list)
    for row in panel["actions"]:
        if row["case_id"] not in cases:
            raise ValueError("Each edit must belong to a declared family.")
        rows[row["case_id"]].append(row)

    rng = np.random.default_rng(seed)
    result = {}
    families = {case_id: _family_actions(case, rows[case_id]) for case_id, case in cases.items()}
    for case_id, (configs, edits, physical) in families.items():
        labels = sorted(edits)
        order = rng.permutation(labels).tolist()
        started = perf_counter()
        design = build_native_design(configs[BASELINE], case_id, device=device)
        anchor = score_native_design(model, design, tau=tau, mode=mode)
        absolute = _functionals(anchor)
        material_max = _material_max(anchor)
        anchor_seconds = perf_counter() - started
        methods = {}
        for method in METHODS:
            started = perf_counter()
            relevance = None if method == "uniform" else _module_relevance(design, anchor, tau, mode, method)
            ranked = order if relevance is None else sorted(order, key=lambda label: -(edits[label] @ relevance))
            chosen = ranked[:budget]
            proposal_seconds = perf_counter() - started
            predicted = {BASELINE: dict.fromkeys(FUNCTIONALS, 0.0)}
            maxima = {BASELINE: material_max}
            work = defaultdict(int)
            started = perf_counter()
            for label in chosen:
                trial = build_native_design(configs[label], case_id, member_ids=design.member_ids, device=device)
                score = score_native_design(model, trial, tau=tau, mode=mode)
                if not torch.equal(anchor["fields"]["temperature/material"]["mask"],
                                   score["fields"]["temperature/material"]["mask"]):
                    raise ValueError(f"{case_id}/{label} changed valid material receiver support.")
                predicted[label] = {name: value - absolute[name] for name, value in _functionals(score).items()}
                maxima[label] = _material_max(score)
                for name, value in score["work"].items():
                    work[name] += value
            scoring_seconds = perf_counter() - started
            metrics = _evaluate_shortlist(predicted, physical, pressure_allowance=pressure_allowance,
                                          resolution=resolution)
            methods[method] = {"shortlist": chosen, "module_relevance": None if relevance is None else relevance.tolist(),
                               "candidate_scores": len(chosen), "predicted_deltas": predicted,
                               "predicted_material_maxima": maxima,
                               "proposal_seconds": proposal_seconds, "scoring_seconds": scoring_seconds,
                               "work": dict(work), "decision": metrics}
        result[case_id] = {"role": cases[case_id].get("role"), "legal_pool_size": len(labels),
                           "baseline_seconds": anchor_seconds, "baseline_work": anchor["work"],
                           "baseline_prediction": absolute, "reference_deltas": physical, "methods": methods}

    return {"schema": "thermal_native_placement_benchmark_v1", "seed": seed, "budget": budget,
            "pressure_allowance": pressure_allowance, "j_resolution": resolution, "tau": tau,
            "mode": mode, "cases": result, "by_role": _summarize(result)}
