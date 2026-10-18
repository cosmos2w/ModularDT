"""Input-design diagnostics and saved physical-field galleries for formal releases."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .environment import render_environment
from .io import (initialize_root, sha256_json, validate_ledger_case, validate_raw_case,
                 write_case_atomic, write_json_atomic)
from .physics import solve_cpu
from .qualification import compare_resolution
from .resources import require_budget, require_data2, require_owned_root
from .study_design import resolve_state
from .validation import audit_persisted, validate_solution


def summarize_design(plan: dict) -> dict:
    """Report actual component exposure, including correlations in frozen pairing."""
    splits = Counter(row["split"] for row in plan["states"])
    contact_counts = defaultdict(Counter)
    exposures = defaultdict(lambda: defaultdict(set))
    roots = defaultdict(lambda: {"environment": set(), "layout": set()})
    for row in plan["states"]:
        split = row["split"]
        environment = plan["environments"][row["environment_root_id"]]
        layout = plan["layouts"][row["layout_root_id"]]
        contact_counts[split][str(len(layout["contacts"]))] += 1
        roots[split]["environment"].add(row["environment_root_id"])
        roots[split]["layout"].add(row["layout_root_id"])
        exposures[split][row["layout_root_id"]].add(environment["family"])
    return {"profile": plan["profile"], "plan_hash": plan["plan_hash"],
            "states": len(plan["states"]), "split_counts": dict(splits),
            "environment_roots": len(plan["environments"]), "layout_roots": len(plan["layouts"]),
            "roots_by_split": {split: {key: len(value) for key, value in groups.items()}
                               for split, groups in roots.items()},
            "contact_counts_by_split": {key: dict(value) for key, value in contact_counts.items()},
            "morphology_families_per_layout": {split: dict(Counter(str(len(value)) for value in items.values()))
                                               for split, items in exposures.items()},
            "challenge_counts": dict(Counter(row.get("challenge", "default") for row in plan["states"])),
            "quartet_types": dict(Counter(row["kind"] for row in plan["quartets"]))}


def _scene(plan: dict, row: dict) -> dict:
    scene = resolve_state(plan, row)
    scene["formal_state_id"] = scene["state_id"]
    scene["state_id"] = "f_" + hashlib.sha256(scene["formal_state_id"].encode()).hexdigest()[:24]
    return scene


def select_representatives(standard: dict, expanded: dict) -> list[dict]:
    """Freeze six choices from inputs alone, before reading target fields."""
    chosen = []
    for family, count in (("correlated_mosaic", 2), ("layered_barrier", 6),
                          ("inclusions", 8), ("corridors", 12)):
        row = next(row for row in standard["states"] if row["split"] == "train"
                   and standard["environments"][row["environment_root_id"]]["family"] == family
                   and len(standard["layouts"][row["layout_root_id"]]["contacts"]) == count)
        chosen.append({"profile": "standard", "state_id": row["state_id"], "label": f"{family}, M{count}"})
    standard_ids = {row["state_id"] for row in standard["states"]}
    for challenge in ("m16", "composite"):
        row = next(row for row in expanded["states"] if row.get("challenge") == challenge
                   and row["state_id"] not in standard_ids)
        chosen.append({"profile": "expanded", "state_id": row["state_id"], "label": f"new Expanded {challenge}"})
    return chosen


def diagnose_releases(standard_root: Path, expanded_root: Path, output_root: Path, *,
                      retained_budget_bytes: int, temporary_budget_bytes: int,
                      minimum_free_bytes: int = 100 * 2**30) -> dict:
    """Render six saved cases and refine those cases plus two full quartets on CPU.

    The fourteen 192-grid references (six gallery cases and eight quartet
    endpoints, with any shared inputs reused) are a bounded numerical audit.
    No model is fit and the saved 96-grid release targets remain immutable.
    """
    roots = {"standard": require_owned_root(standard_root), "expanded": require_owned_root(expanded_root)}
    output = require_owned_root(output_root)
    # Fourteen fixed 192² references, their native-96 links and seven figure
    # pairs. These conservative envelopes are checked before any materialization.
    case_upper = 80 * 192**2 + 256 * 2**10
    figures_upper = 7 * 32 * 2**20
    if retained_budget_bytes < 14 * (case_upper + 80 * 96**2 + 256 * 2**10) + figures_upper:
        raise RuntimeError("readiness retained budget cannot hold the fixed reference/figure scope")
    if temporary_budget_bytes < 64 * 2**20:
        raise RuntimeError("readiness staging budget must hold one bounded figure/reference write")
    budget = require_budget(retained_bytes=retained_budget_bytes, temporary_bytes=temporary_budget_bytes,
                            minimum_free_bytes=minimum_free_bytes)
    initialize_root(output, {key: budget[key] for key in ("retained_budget_bytes", "temporary_budget_bytes",
                                                        "minimum_free_bytes", "peak_new_bytes")})
    plans = {key: json.loads((root / "inputs/formal_plan.json").read_text()) for key, root in roots.items()}
    from .formal import CPU_BACKEND, _solver_recipe, load_release_context, resolve_release_row
    def check_materialization(next_retained, next_temporary):
        from .formal import _tree_bytes, check_materialization_budget
        capacity = require_data2(minimum_free_bytes=minimum_free_bytes)
        check_materialization_budget(retained_used=_tree_bytes(output), next_retained_upper=next_retained,
                                     next_temporary_upper=next_temporary, free_bytes=capacity["free_bytes"],
                                     approval=budget)
    check_materialization(14 * (case_upper + 80 * 96**2 + 256 * 2**10) + figures_upper, 64 * 2**20)
    contexts = {key: load_release_context(root) for key, root in roots.items()}
    recipe = {**_solver_recipe({"backend": CPU_BACKEND}, 192), "purpose": "same-input readiness refinement"}
    solver_hash = sha256_json(recipe)
    recipe_path = output / "inputs/refinement_recipe.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise RuntimeError("readiness refinement recipe changed; preserve the prior evidence")
    if not recipe_path.exists():
        write_json_atomic(recipe_path, recipe)
    selected = select_representatives(plans["standard"], plans["expanded"])
    quartets = [next(q for q in plans["standard"]["quartets"] if q["kind"] == kind)
                for kind in ("me_patch_move", "mm_voltages")]
    selection = {"schema": "hetmf_readiness_selection_v1", "selection_basis": "input-only first matching frozen row",
                 "plan_hashes": {key: value["plan_hash"] for key, value in plans.items()},
                 "gallery": selected, "quartets": [q["quartet_id"] for q in quartets]}
    selection_path = output / "inputs/selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("refusing to change a frozen readiness selection")
    write_json_atomic(selection_path, selection)
    jobs = list(selected)
    for quartet in quartets:
        jobs.extend({"profile": "standard", "state_id": sid, "label": quartet["kind"]}
                    for sid in quartet["endpoint_state_ids"])
    refinements = []
    new_refinements = 0
    for spec in jobs:
        plan, source = plans[spec["profile"]], roots[spec["profile"]]
        row = next(row for row in plan["states"] if row["state_id"] == spec["state_id"])
        resolved = resolve_release_row(source, row, context=contexts[spec["profile"]])
        scene = resolved["scene"]
        source_file = source / "raw" / f"{scene['state_id']}_n96.h5"
        local_source = output / "raw" / source_file.name
        check_materialization(case_upper + source_file.stat().st_size, case_upper)
        if not local_source.exists():
            # This preserves the accepted source bytes without a duplicate payload.
            import os
            os.link(source_file, local_source)
        elif not local_source.samefile(source_file):
            raise RuntimeError("readiness source is not the declared immutable reference")
        fine = output / "raw" / f"{scene['state_id']}_n192.h5"
        if not fine.exists():
            solution = solve_cpu(scene, 192)
            quality = validate_solution(scene, solution)
            if not quality["accepted"]:
                raise RuntimeError(f"CPU refinement failed: {spec['state_id']}")
            write_case_atomic(output, scene, solution, grid=192, solver_hash=solver_hash)
            new_refinements += 1
        validate_raw_case(fine, scene, solver_hash)
        validate_ledger_case(output, fine)
        physical = audit_persisted(output, scene, 192)
        if not physical["accepted"]:
            raise RuntimeError("persisted readiness reference failed physical audit")
        refinements.append({"formal_state_id": spec["state_id"], "profile": spec["profile"],
                            **compare_resolution(output, scene["state_id"], 96, 192)})
    figures, gallery, interactions = _render(output, roots, plans, selected, quartets, refinements,
                                             check_materialization)
    source_ids = {r["state_id"]: r["input_hash"] for r in plans["standard"]["states"]}
    expanded_ids = {r["state_id"]: r["input_hash"] for r in plans["expanded"]["states"]}
    if any(expanded_ids.get(key) != value for key, value in source_ids.items()):
        raise RuntimeError("Standard is not an exact physical-input subset of Expanded")
    result = {"schema": "hetmf_readiness_diagnostics_v1", "design": {key: summarize_design(value) for key, value in plans.items()},
              "standard_nested_states": len(source_ids), "gallery": gallery,
              "refinements": refinements, "counterfactuals": interactions, "figures": figures,
              "new_refinement_solves": new_refinements,
              "reused_refinement_sources": len(refinements) - new_refinements,
              "new_gpu_jobs": 0, "training_updates": 0}
    write_json_atomic(output / "diagnostics/readiness.json", result)
    return result


def _render(output: Path, roots: dict, plans: dict, selected: list, quartets: list,
            refinements: list, check_materialization) -> tuple[list, list, list]:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    figures, gallery, interactions = [], [], []

    def save(fig, stem):
        check_materialization(32 * 2**20, 64 * 2**20)
        paths = {suffix: str(output / "figures" / f"{stem}.{suffix}") for suffix in ("pdf", "png")}
        for suffix, path in paths.items():
            staging = output / "temporary" / f"{stem}.{os.getpid()}.{suffix}"
            try:
                fig.savefig(staging, dpi=140)
                if staging.stat().st_size > 16 * 2**20:
                    raise RuntimeError("readiness figure exceeds its declared file bound")
                os.replace(staging, path)
            finally:
                staging.unlink(missing_ok=True)
        plt.close(fig)
        figures.append({"name": stem, **paths})

    first = selected[0]
    plan = plans[first["profile"]]
    scene = _scene(plan, next(row for row in plan["states"] if row["state_id"] == first["state_id"]))
    controls = render_environment(scene["environment"], 96, 96)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for index, (axis, field, label) in enumerate(zip(axes, controls, ("Electrical conductance s", "Thermal conductance κ", "Cooling β"))):
        image = axis.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        for contact in scene["contacts"]:
            axis.add_patch(Circle((contact["x"], contact["y"]), contact["r"], fill=False,
                                  edgecolor="white", linewidth=1.2))
            if index == 0:
                axis.text(contact["x"], contact["y"], f"u={contact['u']:+.2f}", fontsize=8,
                          color="white", ha="center", va="center",
                          bbox={"facecolor": "black", "alpha": 0.5, "pad": 1})
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle("Given material inputs: 144 control records interpolated to the native grid | dimensionless")
    save(fig, "given_inputs")

    for start in range(0, len(selected), 2):
        fig, axes = plt.subplots(2, 5, figsize=(18, 7.4), constrained_layout=True)
        for iy, spec in enumerate(selected[start:start + 2]):
            plan, root = plans[spec["profile"]], roots[spec["profile"]]
            row = next(row for row in plan["states"] if row["state_id"] == spec["state_id"])
            scene = _scene(plan, row)
            s, _, _ = render_environment(scene["environment"], 96, 96)
            with h5py.File(root / "raw" / f"{scene['state_id']}_n96.h5") as raw:
                v, theta = raw["potential"][:], raw["temperature_rise"][:]
                q = raw["q_sheet"][:] + raw["q_contact"][:]
                jx, jy = raw["jx_face"][:], raw["jy_face"][:]
                speed = np.hypot((jx[:, :-1] + jx[:, 1:]) / 2, (jy[:-1] + jy[1:]) / 2)
                contact_share = float(raw["q_contact"][:].sum() / q.sum())
            gallery.append({**spec, "peak_temperature": float(theta.max()), "mean_temperature": float(theta.mean()),
                            "peak_joule_heat": float(q.max()), "contact_heat_fraction": contact_share,
                            "raw_key": scene["state_id"]})
            panels = ((s, "Given electrical conductance s", "viridis"), (v, "Potential v", "coolwarm"),
                      (speed, "Reconstructed sheet current |j|", "magma"), (q, "Joule heat q", "magma"),
                      (theta, "Temperature rise θ", "inferno"))
            for ix, (values, title, cmap) in enumerate(panels):
                axis = axes[iy, ix]
                image = axis.imshow(values, origin="lower", extent=(0, 1, 0, 1), cmap=cmap)
                for contact in scene["contacts"]:
                    axis.add_patch(Circle((contact["x"], contact["y"]), contact["r"], fill=False,
                                          edgecolor="white", linewidth=0.7))
                axis.set_title(title, fontsize=10)
                axis.set_xlabel("x")
                axis.set_ylabel((spec["label"] + "\ny") if ix == 0 else "y")
                fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
        fig.suptitle("Saved physical reference fields | native 96² | dimensionless | separate color scales", fontsize=13)
        save(fig, f"representatives_{start // 2 + 1}")

    for quartet in quartets:
        row_by_id = {r["state_id"]: r for r in plans["standard"]["states"]}
        keys = [_scene(plans["standard"], row_by_id[sid])["state_id"] for sid in quartet["endpoint_state_ids"]]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        summary = {"quartet_id": quartet["quartet_id"], "kind": quartet["kind"], "interventions": quartet["interventions"]}
        for iy, field in enumerate(("potential", "temperature_rise")):
            arrays, fine = [], []
            for key in keys:
                with h5py.File(output / "raw" / f"{key}_n96.h5") as raw:
                    arrays.append(raw[field][:])
                with h5py.File(output / "raw" / f"{key}_n192.h5") as raw:
                    fine.append(raw[field][:])
            mixed = arrays[3] - arrays[1] - arrays[2] + arrays[0]
            mixed_fine = fine[3] - fine[1] - fine[2] + fine[0]
            restricted = mixed_fine.reshape(96, 2, 96, 2).mean(axis=(1, 3))
            floor = float(np.linalg.norm(mixed - restricted))
            summary[field] = {"interaction_l2": float(np.linalg.norm(mixed)),
                              "interaction_max_absolute": float(np.abs(mixed).max()),
                              "grid_difference_l2": floor,
                              "interaction_over_grid_difference": float(np.linalg.norm(mixed) / floor) if floor else None}
            for ix, (value, title) in enumerate(((arrays[1] - arrays[0], "change A"),
                                               (arrays[2] - arrays[0], "change B"), (mixed, "joint interaction"))):
                measured_max = float(np.abs(value).max())
                bound = max(measured_max, 1e-18)
                if quartet["kind"] == "mm_voltages" and field == "potential" and ix == 2:
                    # Compare the linear null to the actual isolated changes,
                    # avoiding a saturated color map of solver-scale noise.
                    bound = max(float(np.abs(arrays[1] - arrays[0]).max()),
                                float(np.abs(arrays[2] - arrays[0]).max()), 1e-18)
                    title += " (numerical null)"
                image = axes[iy, ix].imshow(value, origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm",
                                            vmin=-bound, vmax=bound)
                axes[iy, ix].set_title(f"{field}: {title}\nmax |Δ|={measured_max:.2e}", fontsize=10)
                fig.colorbar(image, ax=axes[iy, ix], shrink=0.8)
        fig.suptitle(f"Actual four-solve counterfactual: {quartet['kind']} | 96², checked against 192² | dimensionless")
        save(fig, f"counterfactual_{quartet['kind']}")
        interactions.append(summary)

    by_id = {row["formal_state_id"]: row for row in refinements}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(selected))
    for axis, fields in zip(axes, (("potential_relative_l2", "temperature_rise_relative_l2", "temperature_peak_relative"),
                                   ("q_total_relative_l2", "jx_face_relative_l2", "jy_face_relative_l2", "terminal_current_relative_l2"))):
        for field in fields:
            axis.plot(x, [max(by_id[spec["state_id"]][field], 1e-16) for spec in selected], marker="o",
                      label=field.replace("_relative_l2", "").replace("_relative", ""))
        axis.set_yscale("log")
        axis.set_ylabel("relative discrepancy")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, ncol=3)
        axis.set_xticks(x, [spec["label"] for spec in selected], rotation=15, ha="right")
    fig.suptitle("Same physical inputs at 96² and 192² | conservative cell/face restriction | six input-selected cases")
    save(fig, "refinement_audit")
    return figures, gallery, interactions
