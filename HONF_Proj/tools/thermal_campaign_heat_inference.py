#!/usr/bin/env python3
"""Known-geometry heat inference through frozen native Thermal checkpoints.

Stored individual allocations are evaluation-only. The optimizer receives
fixed named sensors, geometry, materials, operating context and supplied total
heat. All proposed fields come from the same checkpoint. No solver is called.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from thermal_campaign_evaluate import screen_indices


def sensor_panel(sample):
    """Fixed named panel snapped to the native grid using geometry alone."""
    structure = sample["structure"]
    x, y = np.asarray(sample["x_grid"]), np.asarray(sample["y_grid"])
    coordinates = np.stack((x.ravel(), y.ravel()), -1)
    radius = float(np.asarray(structure["material_params"])[5])
    eligible = np.ones(coordinates.shape[0], dtype=bool)
    for centre in np.asarray(structure["module_centers"])[np.asarray(structure["module_present"]) > .5]:
        eligible &= np.linalg.norm(coordinates - centre, axis=-1) > radius
    if eligible.sum() < 14:
        raise ValueError("Native geometry has insufficient distinct fluid sensors")
    available = np.flatnonzero(eligible)
    indices, names = [], []
    for x_index, x_fraction in enumerate((.2, .4, .6, .8)):
        for y_index, y_fraction in enumerate((.25, .5, .75)):
            requested = np.asarray([x.min() + x_fraction * (x.max() - x.min()), y.min() + y_fraction * (y.max() - y.min())])
            distance = ((coordinates[available] - requested) ** 2).sum(-1)
            selected = int(available[np.argmin(distance)])
            if selected in indices:
                raise ValueError("Named observation/held sensors snap to a duplicate grid row")
            indices.append(selected)
            names.append(f"x{x_index+1}_y{y_index+1}")
    for fraction, name in ((0., "inlet_midline"), (1., "outlet_midline")):
        requested = np.asarray([x.min() + fraction * (x.max() - x.min()), (y.min() + y.max()) / 2])
        indices.append(int(available[np.argmin(((coordinates[available] - requested) ** 2).sum(-1))]))
        names.append(name)
    return coordinates[indices].astype(np.float32), np.asarray(indices), names, np.arange(0, 12, 2), np.arange(1, 12, 2)


def native_heat_predictor(model, checkpoint, sample, sensors, observed_rows, held_rows):
    from channelthermal.response_control.contracts import DesignInput, RoleQuery
    from channelthermal.response_control.native import DifferentiableThermalOperator
    active = np.asarray(sample["structure"]["module_present"]) > .5
    slots = np.flatnonzero(active)
    device = next(model.parameters()).device
    material = np.asarray(sample["structure"]["material_params"]).reshape(-1)
    context = {name: float(material[index]) for index, name in enumerate(
        ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius"))}
    for name in ("re", "u_in", "domain_length_x", "domain_length_y"):
        context[name] = float(np.asarray(sample["structure"][name]).reshape(-1)[0])
    positions = torch.as_tensor(sample["structure"]["module_centers"], dtype=torch.float32, device=device)
    valid = torch.as_tensor(active, device=device)
    ports = np.asarray(sample["interface_condition"])[active, :, :3]
    local = np.asarray(sample["module_internal_query_points"])
    field_names = tuple(model.config.channelthermal.field_names)
    role_queries = {
        "fluid_fields": RoleQuery("fluid_fields", torch.as_tensor(sensors, device=device), field_names,
                                  tuple("benchmark_units" for _ in field_names), None, "eulerian"),
        "interface": RoleQuery("interface", torch.as_tensor(ports.reshape(-1, 3), device=device),
            ("T_surface", "q_normal"), ("benchmark_temperature", "benchmark_flux_proxy"),
            tuple(int(slot) for slot in slots for _ in range(ports.shape[1])), "interface_material_angle"),
        "solid_temperature": RoleQuery("solid_temperature", torch.as_tensor(np.tile(local, (slots.size, 1)), device=device),
            ("temperature",), ("benchmark_temperature",),
            tuple(int(slot) for slot in slots for _ in range(local.shape[0])), "solid_material_normalized_xy"),
    }
    # The maintained adapter copies only static material properties/capacity;
    # it withholds all stored solved interface/field data and baseline heat.
    operator = DifferentiableThermalOperator(model, sample,
        dataset_config=checkpoint.get("train_config", {}).get("dataset", {}),
        normalization_stats=checkpoint.get("global_normalization_stats", {}))
    temperature, pressure = field_names.index("temperature"), field_names.index("p")
    observed_rows = torch.as_tensor(observed_rows, device=device)
    held_rows = torch.as_tensor(held_rows, device=device)

    def predict(heat):
        captured = []
        hook = model.register_forward_hook(lambda _module, _args, output: captured.append(output.get("prepared_state")))
        try:
            result = operator(DesignInput(positions, heat, valid), context, role_queries)
        finally:
            hook.remove()
        fields = result.role_values["fluid_fields"]
        solid = result.role_values["solid_temperature"].reshape(slots.size, local.shape[0], -1)
        groups = []
        if captured and captured[0] is not None and hasattr(model.core.backend, "organizer"):
            plan = captured[0].prepared.backend_state["hypergraph_plan"]
            for tau in ("MM", "EM", "QM"):
                admission = plan.strategy_data.get("typed_admission", {}).get(tau, plan.admission)
                for group in torch.nonzero(admission[0] > 0, as_tuple=False).flatten():
                    support = torch.nonzero(plan.memberships[tau][0, group] > 0, as_tuple=False).flatten()
                    if support.numel() >= 2 and support.numel() < int(valid.sum()):
                        groups.append(support.detach())
        return {"observed": fields[observed_rows, temperature], "held": fields[held_rows, temperature],
                "peaks": solid[..., 0].max(-1).values, "pressure": fields[-2, pressure] - fields[-1, pressure],
                "groups": tuple(groups)}
    return predict, valid


def evaluate_heat(checkpoint_path, *, dataset_path, output_dir, device="cpu", cases=12, starts=3,
                  steps=30, learning_rate=.05, seed=20261002):
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.evaluation.loading import load_model

    from honf_inverse_core.heat_inference import fixed_total_heat_inference, observation_identifiability
    model, checkpoint = load_model(Path(checkpoint_path), torch.device(device))
    model.eval().requires_grad_(False)
    dataset = GlobalChannelThermalDataset(dataset_path, split="test", points_per_case=1,
        random_point_sampling=False, include_grid=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.is_relative_to(PROJECT_ROOT) and not any(output_dir.is_relative_to(PROJECT_ROOT / root) for root in ("diagnostics", "Trained_Results")):
        raise ValueError("Inverse evidence must live under ignored diagnostics or Trained_Results")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index in screen_indices(dataset, cases):
        sample = dataset[index]
        case_id = str(sample["case_id"])
        sensors, grid_rows, names, observed_rows, held_rows = sensor_panel(sample)
        predictor, active = native_heat_predictor(model, checkpoint, sample, sensors, observed_rows, held_rows)
        field_names = list(dataset.channel_order)
        targets = torch.as_tensor(sample["steady_field"].reshape(-1, len(field_names))[grid_rows, field_names.index("temperature")],
                                  device=device, dtype=torch.float32)
        total = torch.as_tensor(float(np.asarray(sample["structure"]["heat_powers"]).sum()), device=device)
        active_ids = torch.nonzero(active, as_tuple=False).flatten()
        initial = active.float() / active.sum()
        identification = observation_identifiability(predictor, (initial * total).requires_grad_(), active)
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        info = {"case_id": case_id, "checkpoint_epoch": checkpoint.get("epoch"), "reference": "stored exposed development benchmark",
                "hidden_heat_scope": "evaluation only; optimizer sees public total and fixed named sensors",
                "sensor_names": names, "observed_rows": observed_rows.tolist(), "held_rows": held_rows.tolist(),
                "rank": identification["rank"], "free_dimensions": identification["free_dimensions"],
                "condition_number": identification["condition_number"] if np.isfinite(identification["condition_number"]) else None,
                "rank_tolerance": identification["rank_tolerance"], "trials": []}
        np.savez_compressed(case_dir / "task.npz", sensors=sensors, grid_rows=grid_rows,
            observation=targets[observed_rows].cpu().numpy(), held=targets[held_rows].cpu().numpy(),
            singular_values=identification["singular_values"].cpu().numpy(), jacobian=identification["jacobian"].cpu().numpy(),
            reference_heat=np.asarray(sample["structure"]["heat_powers"]), total_heat=total.cpu().numpy(),
            reference_material_peaks=np.asarray(sample["module_internal_temperature_points"])[active.cpu().numpy()].max(-1))
        generator = torch.Generator().manual_seed(seed + int(case_id))
        for start in range(starts):
            fractions = initial.clone()
            if start:
                draw = torch.rand(active_ids.numel(), generator=generator).to(device)
                fractions[active_ids] = draw / draw.sum()
            block_stream = tuple(float(value) for value in torch.rand(steps, generator=generator))
            permutations = tuple(active_ids[torch.randperm(active_ids.numel(), generator=generator).to(device)] for _ in range(steps))
            for mode in ("joint", "graph", "ungrouped"):
                trail = fixed_total_heat_inference(predictor, targets[observed_rows], targets[held_rows], active,
                    total, fractions, mode=mode, steps=steps, learning_rate=learning_rate,
                    block_stream=block_stream, permutation_stream=permutations)
                path = case_dir / f"start_{start:02d}_{mode}.npz"
                np.savez_compressed(path, heat=torch.stack(trail.heat).cpu().numpy(), iterations=trail.iterations,
                    observed_rmse=trail.observed_rmse, held_rmse=trail.held_rmse,
                    observed_predictions=torch.stack(trail.observed_predictions).cpu().numpy(),
                    held_predictions=torch.stack(trail.held_predictions).cpu().numpy(),
                    peaks=torch.stack(trail.peaks).cpu().numpy(), pressure=torch.stack(trail.pressure).cpu().numpy(),
                    selected_module_mask=np.stack([np.isin(np.arange(active.numel()), values.cpu().numpy()) for values in trail.selected_modules])
                        if trail.selected_modules else np.zeros((0, active.numel()), dtype=bool),
                    elapsed_seconds=trail.elapsed_seconds)
                info["trials"].append({"start": start, "mode": mode, "trail": str(path),
                    "optimizer_steps": trail.optimizer_steps, "meaningful_graph_steps": trail.meaningful_graph_steps,
                    "full_joint_fallback_steps": trail.group_fallback_steps,
                    "observed_rmse_initial": trail.observed_rmse[0], "observed_rmse_final": trail.observed_rmse[-1],
                    "held_rmse_initial": trail.held_rmse[0], "held_rmse_final": trail.held_rmse[-1],
                    "elapsed_seconds": trail.elapsed_seconds[-1], "total_feasible": bool(torch.allclose(trail.heat[-1].sum(), total)),
                    "nonnegative": bool((trail.heat[-1] >= 0).all()), "surrogate_only": True})
                (case_dir / "summary.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
                print(f"{case_id} start {start} {mode}: saved {trail.optimizer_steps} steps", flush=True)
        summaries.append(info)
        (output_dir / "summary.json").write_text(json.dumps({"checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"), "seed": seed, "planned_cases": cases,
            "starts": starts, "steps": steps, "learning_rate": learning_rate,
            "evidence_limit": "frozen surrogate observation matching; no new independent physical solve",
            "cases": summaries}, indent=2, allow_nan=False) + "\n")
    return output_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=.05)
    args = parser.parse_args(argv)
    if min(args.cases, args.starts, args.steps, args.learning_rate) <= 0:
        parser.error("case/start/step counts and learning rate must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    print(evaluate_heat(args.checkpoint, dataset_path=args.dataset, output_dir=args.output_dir,
        device=args.device, cases=args.cases, starts=args.starts, steps=args.steps, learning_rate=args.learning_rate))
