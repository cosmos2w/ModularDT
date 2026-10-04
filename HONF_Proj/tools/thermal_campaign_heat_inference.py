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
from zipfile import BadZipFile

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from thermal_campaign_evaluate import screen_indices


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def selected_training_heat_bounds(dataset_path, manifest):
    """Read active prescribed heat only from the frozen selected train IDs."""
    import h5py
    if manifest is None:
        raise ValueError("Training-support heat caps require a bound development manifest")
    selected = tuple(str(case) for case in manifest["partitions"]["train"]["case_ids"])
    if not selected:
        raise ValueError("Training-support heat caps require selected training cases")
    values = []
    with h5py.File(dataset_path, "r") as source:
        original_ids = source["case_ids"].asstr()[:]
        splits = source["splits"].asstr()[:]
        training_ids = set(original_ids[splits == "train"])
        if not set(selected) <= training_ids:
            raise ValueError("Heat cap selection includes a source nontraining case")
        for case in selected:
            record = source["cases"][case]
            active = np.asarray(record["module_present"]) > .5
            heat = np.asarray(record["heat_powers"])[active].reshape(-1)
            if not heat.size or not np.isfinite(heat).all() or (heat < 0).any():
                raise ValueError("Selected training active heat must be finite and nonnegative")
            values.extend(heat.tolist())
    return {"minimum": min(values), "maximum": max(values),
            "selected_train_cases": len(selected), "active_heat_values": len(values),
            "source": "selected-training prescribed active heat; no validation or hidden target bounds",
            "development_manifest_sha256": manifest["manifest_sha256"]}


def persist_heat_progress(path, rows, row):
    """Immediately retain accepted/rejected states and cumulative call charges."""
    rows.append(row)
    atomic_npz(path, heat=torch.stack([item["heat"] for item in rows]).cpu().numpy(),
        iterations=np.asarray([item["iteration"] for item in rows]),
        observed_predictions=torch.stack([item["observed"] for item in rows]).cpu().numpy(),
        held_predictions=torch.stack([item["held"] for item in rows]).cpu().numpy(),
        observed_rmse=np.asarray([item["observed_rmse"] for item in rows]),
        held_rmse=np.asarray([item["held_rmse"] for item in rows]),
        charged_forward_calls=np.asarray([item["forward_calls"] for item in rows]),
        charged_vjp_calls=np.asarray([item["vjp_calls"] for item in rows]),
        accepted_steps=np.asarray([item["accepted_steps"] for item in rows]),
        total_residual=np.asarray([item.get("total_residual", 0.) for item in rows]),
        bound_excess=np.asarray([item.get("bound_excess", 0.) for item in rows]),
        feasibility_tolerance=np.asarray([item.get("feasibility_tolerance") for item in rows], dtype=float))


def snapshot_forward_state(model):
    """Detached CPU copy of loaded parameters and persistent state buffers."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def verify_frozen_forward(model, snapshot):
    """Reject state mutation or trainable/gradient-bearing forward parameters."""
    current = model.state_dict()
    if current.keys() != snapshot.keys():
        raise RuntimeError("Frozen forward state tensor names changed")
    for name, original in snapshot.items():
        value = current[name].detach().cpu()
        # Byte comparison distinguishes signed zero and preserves NaN payloads.
        if (value.shape != original.shape or value.dtype != original.dtype
                or not torch.equal(value.contiguous().reshape(-1).view(torch.uint8),
                                   original.contiguous().reshape(-1).view(torch.uint8))):
            raise RuntimeError(f"Frozen forward state tensor changed: {name}")
    parameters = list(model.named_parameters())
    trainable = [name for name, value in parameters if value.requires_grad]
    gradients = [name for name, value in parameters if value.grad is not None]
    if trainable or gradients:
        raise RuntimeError(f"Forward parameters are not frozen: trainable={trainable}, gradients={gradients}")
    if model.training:
        raise RuntimeError("Frozen forward model left evaluation mode")
    return {"passed": True, "state_dict_tensors_checked": len(snapshot),
            "state_dict_scalars_checked": sum(value.numel() for value in snapshot.values()),
            "state_dict_unchanged_bitwise": True, "trainable_forward_parameter_tensors": 0,
            "forward_parameter_gradient_tensors": 0,
            "state_scope": "loaded parameters and persistent buffers in state_dict; prepared-state caches excluded"}


def completed_trial(path, record, *, steps):
    """Only skip a saved trial whose arrays agree with its completed receipt."""
    if record is None or not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as arrays:
            iterations = arrays["iterations"]
            expected = int(record["optimizer_steps"]) + 1
            return (expected in (1, steps + 1) and len(iterations) == expected
                    and int(iterations[-1]) == expected - 1
                    and arrays["heat"].shape[0] == expected
                    and arrays["observed_predictions"].shape[0] == expected
                    and arrays["held_predictions"].shape[0] == expected
                    and np.isfinite(arrays["heat"]).all()
                    and np.isfinite(arrays["observed_predictions"]).all()
                    and np.isfinite(arrays["held_predictions"]).all())
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return False


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


def explicit_case_panel(dataset, cases, case_ids=None):
    """Resolve an optional fixed input-selected panel inside bound validation."""
    if case_ids is None:
        return screen_indices(dataset, cases), None
    requested = tuple(str(case_id) for case_id in case_ids)
    if len(requested) != cases or len(set(requested)) != len(requested):
        raise ValueError("Explicit inverse case IDs must be unique and their count must equal cases")
    selected = {str(case_id): index for index, case_id in enumerate(dataset.selected_case_ids)}
    missing = [case_id for case_id in requested if case_id not in selected]
    if missing:
        raise ValueError(f"Explicit inverse cases are outside checkpoint-bound selected validation: {missing}")
    return [selected[case_id] for case_id in requested], requested


def validate_case_panel_resume(previous, requested):
    """A saved trajectory panel cannot silently change membership or order."""
    saved = previous.get("requested_case_ids")
    saved = tuple(str(case_id) for case_id in saved) if saved is not None else None
    if saved != requested:
        raise ValueError("Inverse evidence resume changed the explicit input-selected case panel")


def native_heat_predictor(model, checkpoint, sample, sensors, observed_rows, held_rows, *,
                          positions_override=None, context_override=None, capture_topology=False):
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
    if positions_override is not None:
        positions = positions_override
    if context_override is not None:
        context.update(context_override)
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

    def predict(heat, fixed_topology=None):
        from contextlib import nullcontext

        from honf_forward_core.interface_fields.topology_probe import OrganizerTopologyProbe
        captured = []
        hook = model.register_forward_hook(lambda _module, _args, output: captured.append(output.get("prepared_state")))
        organizer = getattr(model.core.backend, "organizer", None)
        topology = OrganizerTopologyProbe(organizer, fixed_topology) if organizer is not None and (capture_topology or fixed_topology is not None) else nullcontext()
        try:
            with topology as scope:
                result = operator(DesignInput(positions, heat, valid), context, role_queries)
        finally:
            hook.remove()
        fields = result.role_values["fluid_fields"]
        solid = result.role_values["solid_temperature"].reshape(slots.size, local.shape[0], -1)
        groups = []
        donor_receipts = []
        topology_record = scope.record if isinstance(scope, OrganizerTopologyProbe) else None
        if topology_record is not None:
            # Module heat is the optimized coordinate. Environmental donors
            # are retained separately as declared dependency/ancestry data;
            # they are not re-labelled as independent physical heat slots.
            for phase in (0, 1):
                plan = topology_record.states.get(phase)
                if plan is None:
                    continue
                for tau in plan.memberships:
                    admission = plan.strategy_data.get("typed_admission", {}).get(tau, plan.admission)
                    for group in torch.nonzero(admission[0] > 0, as_tuple=False).flatten():
                        module_support = torch.zeros_like(valid)
                        if tau in ("MM", "EM", "QM"):
                            module_support |= plan.memberships[tau][0, group] > 0
                        typed = plan.control_memberships.get(tau, {})
                        if "M" in typed:
                            module_support |= typed["M"][0, group] > 0
                        support = torch.nonzero(module_support & valid, as_tuple=False).flatten()
                        environmental = torch.nonzero(typed["E"][0, group] > 0, as_tuple=False).flatten() if "E" in typed else torch.empty(0, dtype=torch.long)
                        donor_receipts.append({"phase": phase, "mechanism": tau, "group": int(group),
                            "module_value_or_control_donors": support.cpu().tolist(),
                            "environmental_control_donors": environmental.cpu().tolist(),
                            "ancestry": plan.dependency_provenance.get("upstream_ancestry", "legacy phase-current state")})
                        if 2 <= support.numel() < int(valid.sum()):
                            groups.append(support.detach())
        elif captured and captured[0] is not None and hasattr(model.core.backend, "organizer"):
            plan = captured[0].prepared.backend_state["hypergraph_plan"]
            for tau in ("MM", "EM", "QM"):
                admission = plan.strategy_data.get("typed_admission", {}).get(tau, plan.admission)
                for group in torch.nonzero(admission[0] > 0, as_tuple=False).flatten():
                    support = torch.nonzero(plan.memberships[tau][0, group] > 0, as_tuple=False).flatten()
                    if support.numel() >= 2 and support.numel() < int(valid.sum()):
                        groups.append(support.detach())
        return {"observed": fields[observed_rows, temperature], "held": fields[held_rows, temperature],
                "peaks": solid[..., 0].max(-1).values, "pressure": fields[-2, pressure] - fields[-1, pressure],
                "fields": fields, "interface": result.role_values["interface"], "solid": solid[..., 0],
                "groups": tuple(groups), "topology_record": topology_record,
                "control_donor_receipts": donor_receipts}
    return predict, valid


def evaluate_heat(checkpoint_path, *, dataset_path, output_dir, device="cpu", cases=12, starts=3,
                  steps=30, learning_rate=.05, seed=20261002, resume=False,
                  evaluation_scope="auto", development_manifest=None,
                  update_policy="legacy_adam", modes=None, case_ids=None, training_support_caps=False):
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model
    from thermal_development import evaluation_dataset_kwargs, resolve_evaluation_manifest, validate_generated_output

    from honf_inverse_core.heat_inference import (
        UnsupportedHeatTotal,
        bounded_trust_heat_inference,
        fixed_total_heat_inference,
        observation_identifiability,
        public_heat_starts,
    )
    if update_policy not in {"legacy_adam", "bounded_trust"}:
        raise ValueError("Unknown heat-inference update policy")
    modes = tuple(modes or ("joint", "graph", "ungrouped"))
    if not modes or len(set(modes)) != len(modes) or any(mode not in {"joint", "graph", "ungrouped"} for mode in modes):
        raise ValueError("Inverse modes must be unique supported modes")
    if "ungrouped" in modes and ("graph" not in modes or modes.index("graph") > modes.index("ungrouped")):
        raise ValueError("Size-matched random controls require a preceding graph trajectory")
    if update_policy == "bounded_trust" and (cases > 4 or starts > 2 or steps > 10):
        raise ValueError("Faithfulness inverse review is bounded to four cases, two starts and ten attempted steps")
    if training_support_caps and update_policy != "bounded_trust":
        raise ValueError("Training-support caps require bounded trust inference")
    model, checkpoint = load_model(Path(checkpoint_path), torch.device(device))
    model.eval().requires_grad_(False)
    frozen_snapshot = snapshot_forward_state(model)
    initial_freeze_check = verify_frozen_forward(model, frozen_snapshot)
    dataset_config = checkpoint.get("train_config", {}).get("dataset", {})
    manifest = resolve_evaluation_manifest(dataset_config,
        dataset_path, scope=evaluation_scope, manifest_path=development_manifest)
    subset_sha256 = manifest["manifest_sha256"] if manifest is not None else None
    bound_source = selected_training_heat_bounds(dataset_path, manifest) if training_support_caps else None
    heat_bounds = (bound_source["minimum"], bound_source["maximum"]) if bound_source else None
    stats = {key: np.asarray(value, dtype=np.float32) for key, value
             in checkpoint.get("global_normalization_stats", {}).items()}
    if (manifest is not None or dataset_config.get("development_manifest")
            or dataset_config.get("development_subset")) and not stats:
        raise ValueError("Development inverse evaluation requires saved selected-training normalization.")
    dataset = GlobalChannelThermalDataset(dataset_path, split="test", points_per_case=1,
        random_point_sampling=False, include_grid=True, normalizer=H5Normalizer(stats) if stats else None,
        **evaluation_dataset_kwargs(manifest, "test"))
    panel_indices, requested_case_ids = explicit_case_panel(dataset, cases, case_ids)
    output_dir = validate_generated_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cpu_threads = torch.get_num_threads() if torch.device(device).type == "cpu" else None
    if resume and (output_dir / "summary.json").exists():
        previous = json.loads((output_dir / "summary.json").read_text())
        validate_case_panel_resume(previous, requested_case_ids)
        if (Path(previous["checkpoint"]).resolve() != Path(checkpoint_path).resolve()
                or previous["checkpoint_epoch"] != checkpoint.get("epoch")
                or previous.get("development_manifest_sha256") != subset_sha256
                or previous.get("evaluation_scope", "auto") != evaluation_scope
                or previous.get("cpu_threads", cpu_threads) != cpu_threads
                or previous.get("training_support_caps", False) != training_support_caps
                or previous.get("heat_bound_source") != bound_source
                or previous.get("update_policy", "legacy_adam") != update_policy
                or tuple(previous.get("modes", ("joint", "graph", "ungrouped"))) != modes
                or any(previous[key] != value for key, value in
                       (("seed", seed), ("planned_cases", cases), ("starts", starts),
                        ("steps", steps), ("learning_rate", learning_rate)))):
            raise ValueError("Inverse evidence resume changed checkpoint or evaluation budget")
    summaries = []
    def save_summary():
        atomic_json(output_dir / "summary.json", {"checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"), "seed": seed, "planned_cases": cases,
            "evaluation_scope": evaluation_scope, "development_manifest_sha256": subset_sha256,
            "available_test_case_ids": list(dataset.selected_case_ids),
            "requested_case_ids": requested_case_ids,
            "starts": starts, "steps": steps, "learning_rate": learning_rate,
            "modes": modes,
            "cpu_threads": cpu_threads,
            "update_policy": update_policy, "training_support_caps": training_support_caps,
            "heat_bounds": heat_bounds, "heat_bound_source": bound_source,
            "heat_bound_limit": "Training-range inclusion is not independent physical validity",
            "proposal_policy": "observed-only; first/half-radius at most two trials; fixed discrete topology, live continuous physics; ordinary rebuild after acceptance" if update_policy == "bounded_trust" else "historical projected Adam",
            "forward_freeze_verification": {
                "initial_loaded_state": initial_freeze_check,
                "verified_cases_current_invocation": len(summaries),
                "new_executed_trials": sum(len(case["forward_freeze_verification"]["new_executed_trials"]) for case in summaries),
                "reused_saved_trials": sum(len(case["forward_freeze_verification"]["reused_saved_trials"]) for case in summaries),
                "new_predictor_calls": sum(case["forward_freeze_verification"]["new_predictor_calls"] for case in summaries),
                "new_jacobian_predictor_calls": sum(case["forward_freeze_verification"]["new_jacobian_predictor_calls"] for case in summaries),
                "verification_scope": "current invocation model state only; prior saved trial execution is not retrospectively verified",
            },
            "evidence_limit": "frozen surrogate observation matching; no new independent physical solve",
            "cases": summaries})

    for index in panel_indices:
        sample = dataset[index]
        case_id = str(sample["case_id"])
        sensors, grid_rows, names, observed_rows, held_rows = sensor_panel(sample)
        native_predictor, active = native_heat_predictor(model, checkpoint, sample, sensors, observed_rows, held_rows,
            capture_topology=update_policy == "bounded_trust")
        predictor_calls = 0

        def predictor(heat, fixed_topology=None, _native_predictor=native_predictor):
            nonlocal predictor_calls
            predictor_calls += 1
            return _native_predictor(heat, fixed_topology=fixed_topology)
        field_names = list(dataset.channel_order)
        targets = torch.as_tensor(sample["steady_field"].reshape(-1, len(field_names))[grid_rows, field_names.index("temperature")],
                                  device=device, dtype=torch.float32)
        total = torch.as_tensor(float(np.asarray(sample["structure"]["heat_powers"]).sum()), device=device)
        active_ids = torch.nonzero(active, as_tuple=False).flatten()
        initial = active.float() / active.sum()
        try:
            capped_starts = public_heat_starts(active, total, starts=starts,
                seed=seed + int(case_id), heat_bounds=heat_bounds) if heat_bounds else None
        except UnsupportedHeatTotal as exc:
            case_dir = output_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            info = {"case_id": case_id, "checkpoint_epoch": checkpoint.get("epoch"),
                "status": "unsupported", "unsupported_reason": str(exc),
                "public_total": float(total), "active_modules": int(active.sum()),
                "heat_bounds": heat_bounds, "heat_bound_source": bound_source, "trials": [],
                "forward_freeze_verification": {**verify_frozen_forward(model, frozen_snapshot),
                    "new_predictor_calls": 0, "new_jacobian_predictor_calls": 0,
                    "new_executed_trials": [], "reused_saved_trials": []}}
            atomic_json(case_dir / "summary.json", info)
            summaries.append(info)
            save_summary()
            continue
        identification = observation_identifiability(predictor, (initial * total).requires_grad_(), active)
        jacobian_predictor_calls = predictor_calls
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        info = {"case_id": case_id, "checkpoint_epoch": checkpoint.get("epoch"), "reference": "stored exposed development benchmark",
                "hidden_heat_scope": "evaluation only; optimizer sees public total and fixed named sensors",
                "sensor_names": names, "observed_rows": observed_rows.tolist(), "held_rows": held_rows.tolist(),
                "rank": identification["rank"], "free_dimensions": identification["free_dimensions"],
                "condition_number": identification["condition_number"] if np.isfinite(identification["condition_number"]) else None,
                "rank_tolerance": identification["rank_tolerance"], "trials": [],
                "status": "supported", "heat_bounds": heat_bounds, "heat_bound_source": bound_source}
        info["shared_identifiability_charged_calls"] = {"forward": jacobian_predictor_calls,
            "vjp": len(observed_rows) if identification["free_dimensions"] else 0,
            "scope": "one ordinary wrapper forward and one reverse Jacobian row per observed sensor; shared across case trajectories"}
        info["trajectory_charged_call_scope"] = "actual attempted trajectory wrapper forwards plus observation VJPs; shared identifiability separately reported"
        if resume and (case_dir / "summary.json").exists():
            previous = json.loads((case_dir / "summary.json").read_text())
            if any(previous[key] != info[key] for key in ("case_id", "checkpoint_epoch", "sensor_names", "observed_rows", "held_rows")):
                raise ValueError("Inverse evidence resume changed physical case or sensors")
            info["trials"] = previous["trials"]
        atomic_npz(case_dir / "task.npz", sensors=sensors, grid_rows=grid_rows,
            observation=targets[observed_rows].cpu().numpy(), held=targets[held_rows].cpu().numpy(),
            singular_values=identification["singular_values"].cpu().numpy(), jacobian=identification["jacobian"].cpu().numpy(),
            reference_heat=np.asarray(sample["structure"]["heat_powers"]), total_heat=total.cpu().numpy(),
            reference_material_peaks=np.asarray(sample["module_internal_temperature_points"])[active.cpu().numpy()].max(-1),
            reference_pressure_difference=np.asarray(sample["steady_field"]).reshape(-1, len(field_names))[grid_rows[-2], field_names.index("p")]
                - np.asarray(sample["steady_field"]).reshape(-1, len(field_names))[grid_rows[-1], field_names.index("p")])
        executed_trials, reused_trials = [], []
        generator = torch.Generator().manual_seed(seed + int(case_id))
        for start in range(starts):
            fractions = initial.clone()
            if capped_starts is not None:
                fractions = capped_starts[start] / total
            elif start:
                draw = torch.rand(active_ids.numel(), generator=generator).to(device)
                if update_policy == "bounded_trust":
                    draw = draw + .2  # strictly interior, common input-seeded start
                fractions[active_ids] = draw / draw.sum()
            block_stream = tuple(float(value) for value in torch.rand(steps, generator=generator))
            permutations = tuple(active_ids[torch.randperm(active_ids.numel(), generator=generator).to(device)] for _ in range(steps))
            graph_sizes = None
            for mode in modes:
                path = case_dir / f"start_{start:02d}_{mode}.npz"
                existing = next((row for row in info["trials"] if row["start"] == start and row["mode"] == mode), None)
                if resume and completed_trial(path, existing, steps=steps):
                    existing["execution_in_current_invocation"] = "reused_saved_trial"
                    reused_trials.append({"start": start, "mode": mode})
                    if mode == "graph":
                        with np.load(path, allow_pickle=False) as saved:
                            graph_sizes = tuple(int(size) for size in saved["selected_block_size"])
                    print(f"{case_id} start {start} {mode}: retained complete saved trial", flush=True)
                    continue
                if resume and path.exists():
                    failed = path.with_name(path.stem + ".failed_previous_attempt.npz")
                    if failed.exists():
                        raise ValueError(f"Preserved failed trial already exists: {failed}")
                    path.replace(failed)
                info["trials"] = [row for row in info["trials"] if not (row["start"] == start and row["mode"] == mode)]
                inference = bounded_trust_heat_inference if update_policy == "bounded_trust" else fixed_total_heat_inference
                progress_path = path.with_name(path.stem + ".progress.npz")
                if progress_path.exists():
                    preserved = path.with_name(path.stem + ".failed_previous_progress.npz")
                    if preserved.exists():
                        raise ValueError(f"Preserved failed progress already exists: {preserved}")
                    progress_path.replace(preserved)
                progress_rows = []
                trust_kwargs = {"proposal_predictor": lambda heat, reference: predictor(heat, fixed_topology=reference),
                    "heat_bounds": heat_bounds,
                    "state_callback": lambda row, _path=progress_path, _rows=progress_rows:
                        persist_heat_progress(_path, _rows, row)} if update_policy == "bounded_trust" else {}
                trail = inference(predictor, targets[observed_rows], targets[held_rows], active,
                    total, fractions, mode=mode, steps=steps, learning_rate=learning_rate,
                    block_stream=block_stream, permutation_stream=permutations,
                    block_size_stream=graph_sizes if mode == "ungrouped" else None, **trust_kwargs)
                if trail.status == "unsupported":
                    raise RuntimeError("Previously feasible public capped start became unsupported")
                executed_trials.append({"start": start, "mode": mode})
                sizes = tuple(int(group.numel()) for group in trail.selected_modules)
                if mode == "graph":
                    graph_sizes = sizes
                if mode == "ungrouped" and sizes != graph_sizes:
                    raise RuntimeError("Ungrouped control did not reproduce recorded graph block sizes")
                atomic_npz(path, heat=torch.stack(trail.heat).cpu().numpy(), iterations=trail.iterations,
                    observed_rmse=trail.observed_rmse, held_rmse=trail.held_rmse,
                    observed_predictions=torch.stack(trail.observed_predictions).cpu().numpy(),
                    held_predictions=torch.stack(trail.held_predictions).cpu().numpy(),
                    observed_residual=torch.stack(trail.observed_predictions).cpu().numpy() - targets[observed_rows].cpu().numpy(),
                    held_residual=torch.stack(trail.held_predictions).cpu().numpy() - targets[held_rows].cpu().numpy(),
                    peaks=torch.stack(trail.peaks).cpu().numpy(), pressure=torch.stack(trail.pressure).cpu().numpy(),
                    selected_module_mask=np.stack([np.isin(np.arange(active.numel()), values.cpu().numpy()) for values in trail.selected_modules])
                        if trail.selected_modules else np.zeros((0, active.numel()), dtype=bool),
                    selected_block_size=np.asarray(sizes, dtype=np.int64),
                    material_peak_residual=torch.stack(trail.peaks).cpu().numpy()
                        - np.asarray(sample["module_internal_temperature_points"])[active.cpu().numpy()].max(-1),
                    pressure_difference_residual=torch.stack(trail.pressure).cpu().numpy()
                        - (np.asarray(sample["steady_field"]).reshape(-1, len(field_names))[grid_rows[-2], field_names.index("p")]
                           - np.asarray(sample["steady_field"]).reshape(-1, len(field_names))[grid_rows[-1], field_names.index("p")]),
                    elapsed_seconds=trail.elapsed_seconds, charged_forward_calls=np.asarray(trail.forward_calls),
                    charged_vjp_calls=np.asarray(trail.vjp_calls),
                    charged_calls=np.asarray(trail.forward_calls) + np.asarray(trail.vjp_calls),
                    trial_evaluations=np.asarray(trail.trial_evaluations),
                    total_residual=np.asarray(trail.total_residual), bound_excess=np.asarray(trail.bound_excess),
                    feasibility_tolerance=np.asarray(trail.feasibility_tolerance, dtype=float))
                info["trials"].append({"start": start, "mode": mode, "trail": str(path),
                    "execution_in_current_invocation": "newly_executed_trial",
                    "optimizer_steps": trail.optimizer_steps, "meaningful_graph_steps": trail.meaningful_graph_steps,
                    "attempted_updates": trail.optimizer_steps, "accepted_steps": trail.accepted_steps,
                    "rejected_steps": trail.rejected_steps, "step_receipts": trail.step_receipts,
                    "charged_forward_calls": trail.forward_calls[-1] if trail.forward_calls else None,
                    "charged_vjp_calls": trail.vjp_calls[-1] if trail.vjp_calls else None,
                    "full_joint_fallback_steps": trail.group_fallback_steps,
                    "selected_block_sizes": sizes,
                    "recorded_graph_size_match": sizes == graph_sizes if mode == "ungrouped" else None,
                    "trajectory_matching_limit": "same initialization, budget, random stream and exact graph-trail cardinalities; candidate heats and physical group identities differ; graph topology rebuilds between proposals",
                    "observed_rmse_initial": trail.observed_rmse[0], "observed_rmse_final": trail.observed_rmse[-1],
                    "held_rmse_initial": trail.held_rmse[0], "held_rmse_final": trail.held_rmse[-1],
                    "elapsed_seconds": trail.elapsed_seconds[-1], "total_feasible": (
                        max(map(abs, trail.total_residual), default=0.) <= trail.feasibility_tolerance
                        if trail.feasibility_tolerance is not None else bool(torch.allclose(trail.heat[-1].sum(), total))),
                    "nonnegative": bool((trail.heat[-1] >= 0).all()),
                    "native_feasibility_tolerance": trail.feasibility_tolerance,
                    "max_absolute_total_residual": max(map(abs, trail.total_residual)) if trail.total_residual else None,
                    "max_heat_bound_excess": max(trail.bound_excess) if trail.bound_excess else None,
                    "training_support_bounds_feasible": bool(all(
                        ((heat[active] >= heat_bounds[0]) & (heat[active] <= heat_bounds[1])).all()
                        for heat in trail.heat)) if heat_bounds else None, "surrogate_only": True})
                atomic_json(case_dir / "summary.json", info)
                if progress_path.exists():
                    progress_path.unlink()
                print(f"{case_id} start {start} {mode}: saved {trail.optimizer_steps} steps", flush=True)
        info["forward_freeze_verification"] = {
            **verify_frozen_forward(model, frozen_snapshot),
            "new_predictor_calls": predictor_calls,
            "new_jacobian_predictor_calls": jacobian_predictor_calls,
            "new_executed_trials": executed_trials, "reused_saved_trials": reused_trials,
            "verification_scope": "current invocation predictor calls only; reused trial arrays are not retrospectively certified",
        }
        atomic_json(case_dir / "summary.json", info)
        summaries.append(info)
        save_summary()
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
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="Single-thread CPU gradients make repeated fallback comparisons reproducible")
    parser.add_argument("--resume-evaluation", action="store_true")
    parser.add_argument("--evaluation-scope", choices=("auto", "development", "formal-full"), default="auto")
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--update-policy", choices=("legacy_adam", "bounded_trust"), default="legacy_adam")
    parser.add_argument("--training-support-caps", action="store_true",
        help="Freeze active heat min/max from checkpoint-bound selected training inputs")
    parser.add_argument("--modes", nargs="+", choices=("joint", "graph", "ungrouped"))
    parser.add_argument("--case-ids", nargs="+", help="Fixed input-selected IDs; every ID must belong to checkpoint-bound selected validation")
    args = parser.parse_args(argv)
    if min(args.cases, args.starts, args.steps, args.learning_rate, args.cpu_threads) <= 0:
        parser.error("case/start/step counts and learning rate must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    print(evaluate_heat(args.checkpoint, dataset_path=args.dataset, output_dir=args.output_dir,
        device=args.device, cases=args.cases, starts=args.starts, steps=args.steps, learning_rate=args.learning_rate,
        resume=args.resume_evaluation, evaluation_scope=args.evaluation_scope,
        development_manifest=args.development_manifest, update_policy=args.update_policy, modes=args.modes,
        case_ids=args.case_ids, training_support_caps=args.training_support_caps))
