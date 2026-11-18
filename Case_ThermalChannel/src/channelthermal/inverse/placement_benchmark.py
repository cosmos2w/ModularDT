"""Compare fixed-budget proposals for layout edits with one frozen native scorer."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from honf_inverse_core.contracts import PhysicalDesign
from honf_inverse_core.request_schema import GeometryConstraints
from honf_runtime.state_response import evaluate_candidate_pool

from ..interaction_evidence.reference_adapter import operating_context_from_config, operating_context_to_named
from ..native_design import build_native_design, score_native_design
from .geometry import evaluate_geometry


BASELINE = "baseline"
FUNCTIONALS = ("J", "P")
METHODS = ("uniform", "proximity", "environment_access")
FINITE_RESPONSE = "finite_response"
MODEL_POLICIES = (*METHODS, FINITE_RESPONSE)
REFERENCE_POLL = "reference_poll"
FULL_POOL = "full_pool"


def _regional_caps(panel):
    regions = panel.get("regional_caps", {})
    if not isinstance(regions, dict):
        raise ValueError("regional_caps must map observation names to their bounds, reduction and allowance.")
    for name, region in regions.items():
        if not isinstance(name, str) or not name or name in FUNCTIONALS or not isinstance(region, dict):
            raise ValueError("Each regional cap needs a distinct observation name other than J or P.")
        bounds = np.asarray(region.get("bounds"), dtype=float)
        allowance = region.get("allowance")
        if (bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds < 0.) or np.any(bounds > 1.)
                or np.any(bounds[:2] >= bounds[2:]) or region.get("reduction") not in ("max", "mean")
                or isinstance(allowance, bool) or not isinstance(allowance, (int, float))
                or not math.isfinite(allowance) or allowance < 0.):
            raise ValueError(f"Invalid regional cap {name!r}. Use ordered normalized bounds, max or mean, and a nonnegative allowance.")
    return regions


def _region_bounds(domain, region):
    lengths = np.array([domain["lx"], domain["ly"]], dtype=float)
    return np.asarray(region["bounds"]) * np.tile(lengths, 2)


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


def _family_actions(case, rows, regions):
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
    changed = {}
    observations = (*FUNCTIONALS, *regions)
    physical = {BASELINE: dict.fromkeys(observations, 0.0)}
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
        changed[action] = tuple((slot, axis, centers[slot, axis]) for slot, axis in np.argwhere(centers != original))
        configs[action] = {**config, "layout": {**layout, "centers": centers.tolist()}}
        labels = row.get("physical_delta")
        if labels is not None:
            if not isinstance(labels, dict) or set(labels) != set(observations) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in labels.values()
            ):
                raise ValueError(f"{action} needs finite reference deltas for {', '.join(observations)}.")
            physical[action] = labels

    if not edits or len(physical) not in (1, len(configs)):
        raise ValueError(f"{case['case_id']} needs edits and either complete or absent reference deltas.")
    for action, endpoint in configs.items():
        design = PhysicalDesign(endpoint["layout"]["centers"], np.ones(count), heat)
        if not evaluate_geometry(design, context, constraints).valid:
            raise ValueError(f"{case['case_id']}/{action} violates disk separation or channel bounds.")
        for name, region in regions.items():
            bounds = _region_bounds(domain, region)
            centers = design.module_centers
            nearest = np.clip(centers, bounds[:2], bounds[2:])
            if np.any(np.linalg.norm(centers - nearest, axis=-1) <= radius):
                raise ValueError(f"{case['case_id']}/{action} intersects fixed fluid region {name!r}.")
    return configs, edits, changed, physical if len(physical) > 1 else None


def _probe_plan(order, changed, capacity):
    """Find signed single-coordinate endpoints that cover legal combined edits."""
    singles = {}
    for name in order:
        if len(changed[name]) == 1:
            singles.setdefault(changed[name][0], name)
    probes = {}
    covered = {}
    # Reserve at least one forward to check the actual combined move.
    limit = min(capacity - 1, max(2, capacity // 2))
    for name in order:
        coordinates = changed[name]
        if len(coordinates) < 2 or any(point not in singles for point in coordinates):
            continue
        components = tuple(singles[point] for point in coordinates)
        missing = [probe for probe in components if probe not in probes]
        if len(probes) + len(missing) <= limit:
            probes.update(dict.fromkeys(missing))
            covered[name] = components
    return list(probes), covered


def _functionals(scored, design, domain, regions):
    values = {name: float(scored["functionals"][name]) for name in FUNCTIONALS}
    if regions:
        fluid = scored["fields"]["temperature/fluid"]
        xy = design.fluid_xy_source
        for name, region in regions.items():
            bounds = _region_bounds(domain, region)
            inside = ((xy >= bounds[:2]) & (xy <= bounds[2:])).all(axis=-1)
            selected = torch.as_tensor(inside, device=fluid["mask"].device)
            if not selected.any() or not fluid["mask"].reshape(-1)[selected].all():
                raise ValueError(f"{design.case_id}/{name} needs nonempty, fully valid native fluid support.")
            temperatures = fluid["prediction"].reshape(-1)[selected].to(torch.float64)
            values[name] = float(temperatures.amax() if region["reduction"] == "max" else temperatures.mean())
    return values


def _material_max(scored):
    material = scored["fields"]["temperature/material"]
    return float(material["prediction"].masked_fill(~material["mask"], -torch.inf).amax())


def _evaluate_shortlist(predicted, physical, *, pressure_allowance, resolution, regional_allowances=None):
    caps = {"P": pressure_allowance, **(regional_allowances or {})}
    scores = evaluate_candidate_pool(
        predicted, None if physical is None else {name: physical[name] for name in predicted},
        objective="J", constraint_caps=caps, objective_scale=1.0,
        sensitivity={"J": resolution / 4., **dict.fromkeys(caps, 0.)},
    )
    scores.update(_reference_result(scores["model_selected_action"], predicted, physical,
                                    caps=caps, resolution=resolution))
    return scores


def _reference_result(selected, scored, physical, *, caps, resolution):
    edited = selected is not None and selected != BASELINE
    decision = {"reference_checks": int(physical is not None and edited),
                "reference_outcome": "unavailable" if physical is None else "kept baseline"}
    if physical is None:
        return decision

    trial = physical[selected] if edited else None
    violations = [name for name, cap in caps.items() if edited and trial[name] > cap]
    accepted = BASELINE
    if violations:
        decision["reference_outcome"] = "over pressure cap" if "P" in violations else "over regional cap"
    elif edited and trial["J"] < -resolution:
        accepted = selected
        decision["reference_outcome"] = "accepted edit"
    elif edited:
        decision["reference_outcome"] = "no resolved improvement"

    feasible = {name: row["J"] for name, row in physical.items() if all(row[key] <= cap for key, cap in caps.items())}
    full_best = min(feasible.values())
    shortlist_best = min(feasible[name] for name in scored if name in feasible)
    final = physical[accepted]["J"]
    decision.update(verified_incumbent=accepted, verified_improvement=-final,
                    checked_reference_delta=trial, reference_cap_violation="P" in violations,
                    reference_violations=violations,
                    full_pool_regret=final - full_best, shortlist_loss=shortlist_best - full_best,
                    decision_loss=final - shortlist_best)
    return decision


def _summarize(cases, views):
    by_role = defaultdict(list)
    for row in cases.values():
        by_role[row["role"] or "unspecified"].append(row)
    summaries = {}
    for role, families in by_role.items():
        summaries[role] = {}
        for method in views:
            rows = [family["methods"][method] if method in family["methods"] else family[method] for family in families]
            measured = [row["decision"] for row in rows if "verified_improvement" in row["decision"]]
            summary = {
                "family_count": len(rows),
                "reference_family_count": len(measured),
                "mean_candidate_scores": mean(row["candidate_scores"] for row in rows),
                "mean_probe_scores": mean(len(row.get("probe_actions", ())) for row in rows),
                "reference_checks": sum(row["decision"]["reference_checks"] for row in rows),
                "mean_verified_improvement": mean(row["verified_improvement"] for row in measured) if measured else None,
                "resolved_improvements": sum(row["verified_incumbent"] != BASELINE for row in measured) if measured else None,
                "selected_over_cap": sum(bool(row["reference_violations"]) for row in measured) if measured else None,
                "false_feasible_selections": (sum(bool(row["reference_violations"]) for row in measured)
                                              if measured and method != REFERENCE_POLL else None),
                "mean_full_pool_regret": mean(row["full_pool_regret"] for row in measured) if measured else None,
                "mean_shortlist_loss": mean(row["shortlist_loss"] for row in measured) if measured else None,
                "mean_decision_loss": mean(row["decision_loss"] for row in measured) if measured else None,
            }
            if method == FULL_POOL:
                accuracy = [row["response_accuracy"] for row in rows if row["response_accuracy"] is not None]
                summary["response_accuracy"] = {
                    "edits": sum(row["edits"] for row in accuracy),
                    "resolved_j_edits": sum(row["resolved_j_edits"] for row in accuracy),
                    "correct_j_sign": sum(row["correct_j_sign"] for row in accuracy),
                    "resolved_feasible_pairs": sum(row["resolved_feasible_pairs"] for row in accuracy),
                    "correct_feasible_order": sum(row["correct_feasible_order"] for row in accuracy),
                    "mean_abs_pressure_error": mean(row["mean_abs_pressure_error"] for row in accuracy),
                } if accuracy else None
            summaries[role][method] = summary
    return summaries


def _response_accuracy(predicted, physical, *, caps, resolution):
    if physical is None:
        return None
    labels = [name for name in predicted if name != BASELINE]
    resolved = [name for name in labels if abs(physical[name]["J"]) > resolution]
    feasible = sorted((name for name in labels if all(physical[name][key] <= cap for key, cap in caps.items())),
                      key=lambda name: physical[name]["J"])
    resolved_pairs = correct_order = 0
    for cooler, warmer in combinations(feasible, 2):
        if physical[warmer]["J"] - physical[cooler]["J"] > resolution:
            resolved_pairs += 1
            correct_order += predicted[cooler]["J"] < predicted[warmer]["J"]
    return {"edits": len(labels), "resolved_j_edits": len(resolved),
            "correct_j_sign": sum(predicted[name]["J"] * physical[name]["J"] > 0 for name in resolved),
            "resolved_feasible_pairs": resolved_pairs, "correct_feasible_order": correct_order,
            "mean_abs_pressure_error": mean(abs(predicted[name]["P"] - physical[name]["P"]) for name in labels)}


@torch.inference_mode()
def benchmark_placement(model, panel, *, budget, pressure_allowance, tau, resolution, seed=0,
                        device="cpu", mode="restricted", full_pool=False):
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
    regions = _regional_caps(panel)
    regional_allowances = {name: region["allowance"] for name, region in regions.items()}
    caps = {"P": pressure_allowance, **regional_allowances}
    observations = (*FUNCTIONALS, *regions)
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
    families = {case_id: _family_actions(case, rows[case_id], regions) for case_id, case in cases.items()}
    for case_id, (configs, edits, changed, physical) in families.items():
        labels = sorted(edits)
        order = rng.permutation(labels).tolist()
        started = perf_counter()
        design = build_native_design(configs[BASELINE], case_id, device=device)
        anchor = score_native_design(model, design, tau=tau, mode=mode)
        domain = configs[BASELINE]["domain"]
        absolute = _functionals(anchor, design, domain, regions)
        material_max = _material_max(anchor)
        anchor_seconds = perf_counter() - started
        capacity = min(budget, len(labels))
        methods = {}
        reference_settings = {"pressure_allowance": pressure_allowance, "resolution": resolution,
                              "regional_allowances": regional_allowances}

        def run_method(method, capacity):
            started = perf_counter()
            predicted = {BASELINE: dict.fromkeys(observations, 0.0)}
            maxima = {BASELINE: material_max}
            work = defaultdict(int)
            relevance = None
            probes = []
            estimates = {}

            def score_actions(actions):
                scoring_started = perf_counter()
                for label in actions:
                    trial = build_native_design(configs[label], case_id, member_ids=design.member_ids, device=device)
                    score = score_native_design(model, trial, tau=tau, mode=mode)
                    if not torch.equal(anchor["fields"]["temperature/material"]["mask"],
                                       score["fields"]["temperature/material"]["mask"]):
                        raise ValueError(f"{case_id}/{label} changed valid material receiver support.")
                    predicted[label] = {name: value - absolute[name]
                                        for name, value in _functionals(score, trial, domain, regions).items()}
                    maxima[label] = _material_max(score)
                    for name, value in score["work"].items():
                        work[name] += value
                return perf_counter() - scoring_started

            scoring_seconds = 0.0
            ranked = order
            if method == FINITE_RESPONSE:
                probes, covered = _probe_plan(order, changed, capacity)
                scoring_seconds += score_actions(probes)
                estimates = {label: {name: sum(predicted[probe][name] for probe in components)
                                     for name in observations} for label, components in covered.items()}
                plausible = [label for label in estimates if all(estimates[label][key] <= cap for key, cap in caps.items())]
                plausible.sort(key=lambda label: (estimates[label]["J"], estimates[label]["P"]))
                prioritized = set(plausible)
                ranked = plausible + [label for label in order if label not in prioritized]
            elif method in ("proximity", "environment_access"):
                relevance = _module_relevance(design, anchor, tau, mode, method)
                ranked = sorted(order, key=lambda label: -(edits[label] @ relevance))
            followup = [label for label in ranked if label not in predicted][:capacity - len(probes)]
            scoring_seconds += score_actions(followup)
            elapsed = perf_counter() - started
            row = {"shortlist": probes + followup, "module_relevance": None if relevance is None else relevance.tolist(),
                   "candidate_scores": len(probes) + len(followup), "predicted_deltas": predicted,
                   "predicted_material_maxima": maxima, "proposal_seconds": elapsed - scoring_seconds,
                   "scoring_seconds": scoring_seconds, "work": dict(work),
                   "decision": _evaluate_shortlist(predicted, physical, **reference_settings)}
            if method == FINITE_RESPONSE:
                row.update(probe_actions=probes, additive_estimates=estimates,
                           response_prioritized_actions=[label for label in followup if label in prioritized],
                           proposal_status="signed probes" if probes else "uniform fallback")
            return row

        for method in MODEL_POLICIES:
            methods[method] = run_method(method, capacity)
        poll = next((label for label in order if len(changed[label]) == 1), None)
        poll_shortlist = [poll] if poll is not None else []
        poll_decision = _reference_result(poll, [BASELINE, *poll_shortlist], physical, caps=caps, resolution=resolution)
        poll_decision["model_selected_action"] = None
        poll_decision["polled_action"] = poll
        family = {"role": cases[case_id].get("role"), "legal_pool_size": len(labels),
                  "baseline_seconds": anchor_seconds, "baseline_work": anchor["work"],
                  "baseline_prediction": absolute, "reference_deltas": physical, "methods": methods,
                  REFERENCE_POLL: {"shortlist": poll_shortlist, "candidate_scores": 0, "decision": poll_decision}}
        if full_pool:
            diagnostic = run_method(FULL_POOL, len(labels))
            diagnostic["response_accuracy"] = _response_accuracy(diagnostic["predicted_deltas"], physical,
                                                                 caps=caps, resolution=resolution)
            family[FULL_POOL] = diagnostic
        result[case_id] = family

    views = (*MODEL_POLICIES, REFERENCE_POLL)
    if full_pool:
        views += (FULL_POOL,)
    return {"schema": "thermal_native_placement_benchmark_v1", "seed": seed, "budget": budget,
            "pressure_allowance": pressure_allowance, "j_resolution": resolution, "tau": tau,
            "mode": mode, "regional_caps": regions, "cases": result, "by_role": _summarize(result, views)}


def benchmark_placement_curve(model, panel, *, budgets, full_pool=False, **settings):
    """Replay the same families independently at each candidate-forward budget."""
    budgets = tuple(budgets)
    if (len(budgets) < 2 or any(type(budget) is not int or budget < 1 for budget in budgets)
            or any(left >= right for left, right in zip(budgets, budgets[1:]))):
        raise ValueError("Curve budgets must be distinct positive integers in increasing order.")
    runs = [benchmark_placement(model, panel, budget=budget,
                                full_pool=full_pool and budget == budgets[-1], **settings) for budget in budgets]
    methods = (*MODEL_POLICIES, REFERENCE_POLL)
    curves = {
        role: {method: [{"budget": run["budget"], **run["by_role"][role][method]} for run in runs]
               for method in methods}
        for role in runs[0]["by_role"]
    }
    baseline_centers = {case["case_id"]: case["source_config"]["layout"]["centers"] for case in panel["cases"]}
    candidate_centers = {(row["case_id"], row["action"]): row["centers"] for row in panel["actions"]}
    steps = []
    for case_id, family in runs[0]["cases"].items():
        for method in methods:
            previous_action = BASELINE
            previous_cooling = 0.0
            previous_probes = False
            found_improvement = False
            for run in runs:
                result = run["cases"][case_id]
                policy = result[REFERENCE_POLL] if method == REFERENCE_POLL else result["methods"][method]
                decision = policy["decision"]
                action = decision.get("polled_action") or decision["model_selected_action"] or BASELINE
                cooling = decision.get("verified_improvement")
                probes = policy.get("probe_actions", [])
                events = []
                if action != previous_action:
                    events.append("recommendation changed")
                    if decision.get("reference_cap_violation"):
                        events.append("reference pressure exceeded cap")
                    for region in decision.get("reference_violations", []):
                        if region != "P":
                            events.append(f"reference {region} exceeded cap")
                if probes and not previous_probes:
                    events.append("signed probes used")
                if cooling is not None and cooling > 0 and not found_improvement:
                    events.append("first accepted improvement")
                    found_improvement = True
                elif cooling is not None and previous_cooling is not None and cooling != previous_cooling:
                    events.append("verified cooling increased" if cooling > previous_cooling else "verified cooling decreased")
                if events:
                    original = baseline_centers[case_id]
                    proposed = candidate_centers.get((case_id, action), original)
                    moves = [{"slot": slot, "from": list(map(float, before)), "to": list(map(float, after))}
                             for slot, (before, after) in enumerate(zip(original, proposed, strict=True))
                             if tuple(before) != tuple(after)]
                    steps.append({"family": case_id, "role": family["role"] or "unspecified", "policy": method,
                                  "budget": run["budget"], "candidate_scores": policy["candidate_scores"],
                                  "events": events, "previous_recommendation": previous_action,
                                  "recommended": action, "retained": decision.get("verified_incumbent"), "moves": moves,
                                  "verified_improvement": cooling,
                                  "reference_delta": decision.get("checked_reference_delta"),
                                  "reference_outcome": decision["reference_outcome"], "probe_actions": probes})
                previous_action = action
                previous_cooling = cooling
                previous_probes = bool(probes)
    steps.sort(key=lambda row: (row["budget"], row["family"], row["policy"]))
    return {"schema": "thermal_native_placement_curve_v1", "budgets": list(budgets), "curves": curves,
            "critical_steps": steps, "runs": runs}
