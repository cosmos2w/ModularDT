#!/usr/bin/env python3
"""Native dependency and derivative measurements for the two flow policies.

Only saved fixed-quarter inputs are read. Full composed predictions establish
finite heat behavior; flow-only derivatives are checked against those values.
No reference generator, optimizer, inverse search or solver is invoked.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from thermal_response_refinement_evaluation import write_json

REPRESENTATIVES = ("0277", "0291", "0294", "0687")
FLOW_NAMES = ("u", "v", "p", "omega")


def physical_flow(values, stats):
    mean = values.new_tensor(np.asarray(stats["field_mean_by_channel"]).reshape(-1)[:4])
    std = values.new_tensor(np.asarray(stats["field_std_by_channel"]).reshape(-1)[:4])
    return values * std + mean


def gradient_measure(output, inputs):
    """Disconnected inputs are recorded explicitly, without altering output."""
    values = torch.autograd.grad(output, inputs, allow_unused=True, retain_graph=True)
    return [{"disconnected": value is None,
             "max_abs": 0.0 if value is None else float(value.abs().max()),
             "norm": 0.0 if value is None else float(value.norm()),
             "finite": True if value is None else bool(torch.isfinite(value).all())}
            for value in values]


def evaluate(checkpoint_path, output_dir, *, device="cpu", query_count=1024):
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model, make_batch
    from channelthermal.training.epoch import make_model_inputs
    from thermal_campaign_responses import fixed_total_transfers
    from thermal_development import evaluation_dataset_kwargs, resolve_evaluation_manifest, validate_generated_output

    from honf_runtime.compat import resolve_demo_path

    if query_count != 1024:
        raise ValueError("Primary dependency evidence uses Q1024 and fixed-four full Q8192.")
    device = torch.device(device)
    torch.set_num_threads(2)
    output_dir = validate_generated_output(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Preserve measured attempts: select a fresh output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time()
    receipt = {"pid": os.getpid(), "start_unix": start, "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
               "device": str(device), "status": "running", "solver_attempts": 0,
               "optimizer_updates": 0, "complete_wrapper_calls": 0, "flow_only_calls": 0}
    write_json(output_dir / "receipt.json", receipt)
    model, saved = load_model(Path(checkpoint_path), device)
    if not hasattr(model, "forward_flow"):
        raise TypeError("Dependency evaluation requires the explicitly composed flow model.")
    model.eval().requires_grad_(False)
    settings = saved["train_config"]["dataset"]
    if not settings.get("normalize_inputs") or not settings.get("normalize_targets"):
        raise ValueError("This measured flow identity requires its normalized parent input/target convention.")
    path = resolve_demo_path(settings["packed_h5_path"])
    manifest = resolve_evaluation_manifest(settings, path, scope="development")
    stats = saved["global_normalization_stats"]
    normalizer = H5Normalizer({key: np.asarray(value, dtype=np.float32) for key, value in stats.items()})
    datasets = {split: GlobalChannelThermalDataset(path, split=split, points_per_case=1,
        include_grid=True, random_point_sampling=False, normalizer=normalizer,
        normalize_inputs=True, normalize_targets=True, **evaluation_dataset_kwargs(manifest, split))
        for split in ("train", "test")}
    if len(datasets["train"]) != 150 or len(datasets["test"]) != 22:
        raise ValueError("Dependency evaluation is bound to fixed25_v1 150/22.")
    hmean = float(np.asarray(stats["heat_power_mean"]).reshape(-1)[0])
    hstd = float(np.asarray(stats["heat_power_std"]).reshape(-1)[0])
    train_heats = []
    for case_id in datasets["train"].selected_case_ids:
        group = datasets["train"].h5["cases"][case_id]
        active = np.asarray(group["module_present"]) > .5
        train_heats.extend(np.asarray(group["heat_powers"])[active].tolist())
    lower, upper = min(train_heats), max(train_heats)
    result = {"checkpoint": str(Path(checkpoint_path).resolve()), "epoch": saved["epoch"],
        "policy": saved["dependency_policy"], "manifest_sha256": manifest["manifest_sha256"],
        "partition": "fixed22 exposed development validation; derivative panel also two input-selected TRAIN layouts",
        "heat_bounds_train": [lower, upper], "flow_channel_order": FLOW_NAMES,
        "pressure_functional": "full valid-grid 8% sections for Q8192; deterministic sampled sections at Q1024",
        "cases": [], "derivatives": [], "full_four": []}

    def complete(arguments):
        receipt["complete_wrapper_calls"] += 1
        return model(**arguments)

    def flow(structure, queries):
        receipt["flow_only_calls"] += 1
        return model.forward_flow(structure, queries)

    def inputs(sample, full=False):
        coordinates = np.stack((sample["x_grid"].ravel(), sample["y_grid"].ravel()), -1)
        raw_heat = np.asarray(sample["structure"]["heat_powers"]) * hstd + hmean
        structure = sample["structure"]
        centers = np.asarray(structure["module_centers"])
        active = np.asarray(structure["module_present"]) > .5
        radius = float(np.asarray(structure["material_params"])[5])
        valid = np.ones(len(coordinates), dtype=bool)
        for center in centers[active]:
            valid &= np.linalg.norm(coordinates - center, axis=-1) > radius
        rows = np.arange(len(coordinates)) if full else np.flatnonzero(valid)[
            np.linspace(0, valid.sum() - 1, query_count, dtype=int)]
        if full and len(rows) != 8192:
            raise ValueError("Full-field dependency evidence requires the native Q8192 ordering.")
        batch = make_batch(sample, coordinates[rows], device)
        args = make_model_inputs(batch, local_port_condition_mode="predicted", mixed_teacher_ratio=0,
                                 return_predicted_port_outputs=True)
        return args, raw_heat, rows, valid[rows]

    def finite_case(sample, full=False):
        args, heat, rows, valid = inputs(sample, full)
        present = np.asarray(sample["structure"]["module_present"])
        active_slots = np.flatnonzero(present > .5)
        variants = fixed_total_transfers(sample["structure"]["module_centers"], heat, present,
                                        lower=lower, upper=upper, amplitudes=(.1, .2))
        # One independent source excitation also probes common-mode access.
        slot = int(active_slots[0])
        changed = heat.copy()
        direction = 1 if upper - heat[slot] >= heat[slot] - lower else -1
        changed[slot] += direction * .1 * (upper - lower)
        changed[slot] = np.clip(changed[slot], lower, upper)
        variants.append({"label": "individual_first_source", "heat": changed,
                         "fixed_total": False, "slot": slot})
        with torch.no_grad():
            baseline = complete(args)
            reference = physical_flow(baseline["pred_field"][..., :4], stats)
            repeat = physical_flow(complete(args)["pred_field"][..., :4], stats)
            direct = physical_flow(flow(args["structure"], args["query_xy"]), stats)
            prepared = model.prepare_flow(args["structure"])
            x = args["query_xy"][0, :, 0]
            length = float(np.asarray(sample["structure"]["domain_length_x"]).reshape(-1)[0])
            mask = torch.as_tensor(valid, device=device)
            inlet, outlet = mask & (x <= .08 * length), mask & (x >= .92 * length)
            if not inlet.any() or not outlet.any():
                raise ValueError("The measured pressure sections are empty.")
            row = {"case_id": str(sample["case_id"]), "M": len(active_slots), "Q": len(rows),
                   "same_input_repeat_max": float((reference - repeat).abs().max()),
                   "composed_vs_flow_only_max": float((reference - direct).abs().max()), "variants": []}
            arrays = {"query_xy": args["query_xy"][0].cpu().numpy(), "grid_rows": rows,
                      "fluid_mask": valid, "baseline_flow": reference[0].cpu().numpy()}
            for variant in variants:
                new_args = dict(args)
                new_args["structure"] = dict(args["structure"])
                new_args["structure"]["heat_powers"] = torch.as_tensor(
                    (variant["heat"] - hmean) / hstd, device=device, dtype=torch.float32)[None]
                if new_args.get("local_module_params") is not None:
                    new_args["local_module_params"] = args["local_module_params"].clone()
                    new_args["local_module_params"][..., 0] = torch.as_tensor(variant["heat"], device=device)
                output = complete(new_args)
                delta = physical_flow(output["pred_field"][..., :4], stats) - reference
                current = model.prepare_flow(new_args["structure"])
                state_differences = {name: float((getattr(current, name) - getattr(prepared, name)).abs().max())
                    for name in ("source_inputs", "context", "source_states", "centers", "present", "lengths")}
                entry = {key: value for key, value in variant.items() if key != "heat"}
                entry.update(heat_total_delta=float(np.asarray(variant["heat"]).sum() - heat.sum()),
                    max_abs_delta={name: float(delta[..., i].abs().max()) for i, name in enumerate(FLOW_NAMES)},
                    rms_delta={name: float(delta[0, mask, i].square().mean().sqrt()) for i, name in enumerate(FLOW_NAMES)},
                    pressure_8pct_delta=float(delta[0, inlet, 2].mean() - delta[0, outlet, 2].mean()),
                    flow_prepared_max_differences=state_differences)
                row["variants"].append(entry)
                if full:
                    arrays[variant["label"] + "/delta_flow"] = delta[0].cpu().numpy()
            if full:
                np.savez_compressed(output_dir / (row["case_id"] + "_full_null.npz"), **arrays)
        return row

    def derivatives(sample, split):
        args, _, _, valid = inputs(sample)
        structure = dict(args["structure"])
        names = ("heat_powers", "module_centers", "re", "u_in")
        leaves = [structure[name].clone().requires_grad_(True) for name in names]
        structure.update(zip(names, leaves))
        query = args["query_xy"].clone().requires_grad_(True)
        leaves.append(query)
        values = physical_flow(flow(structure, query), stats)
        mask = torch.as_tensor(valid, device=device)
        x = query[0, :, 0]
        length = float(np.asarray(sample["structure"]["domain_length_x"]).reshape(-1)[0])
        inlet, outlet = mask & (x <= .08 * length), mask & (x >= .92 * length)
        pressure = values[0, inlet, 2].mean() - values[0, outlet, 2].mean()
        native_args = dict(args, structure=structure, query_xy=query)
        native_values = physical_flow(complete(native_args)["pred_field"][..., :4], stats)
        native_pressure = native_values[0, inlet, 2].mean() - native_values[0, outlet, 2].mean()
        channel_heat = {name: gradient_measure(values[0, mask, i].mean(), (leaves[0],))[0]
                        for i, name in enumerate(FLOW_NAMES)}
        generator = torch.Generator(device=device).manual_seed(0)
        adjoint = torch.randn(values.shape, generator=generator, device=device)
        objective = (values * adjoint).sum() / values.numel()
        grad = torch.autograd.grad(objective, leaves, allow_unused=True, retain_graph=True)
        row = {"case_id": str(sample["case_id"]), "partition": split, "channel_heat_vjp": channel_heat,
               "pressure_heat_vjp": gradient_measure(pressure, (leaves[0],))[0],
               "native_composed_channel_heat_vjp": {
                   name: gradient_measure(native_values[0, mask, i].mean(), (leaves[0],))[0]
                   for i, name in enumerate(FLOW_NAMES)},
               "native_composed_pressure_heat_vjp": gradient_measure(native_pressure, (leaves[0],))[0],
               "native_composed_vs_flow_only_max": float((native_values - values).abs().max()),
               "ad_fd": {}}
        for index, name in enumerate((*names, "query_xy")):
            if name == "heat_powers":
                continue
            direction = torch.randn(leaves[index].shape, generator=generator, device=device)
            if name == "module_centers":
                direction *= structure["module_present"][..., None]
            direction /= direction.norm().clamp_min(1e-12)
            ad = 0. if grad[index] is None else float((grad[index] * direction).sum())
            comparisons = []
            for step in (.01, .003):
                objectives = []
                with torch.no_grad():
                    for sign in (-1, 1):
                        altered = dict(structure)
                        altered_query = query
                        if name == "query_xy":
                            altered_query = query + sign * step * direction
                        else:
                            altered[name] = leaves[index] + sign * step * direction
                        objectives.append(float((physical_flow(flow(altered, altered_query), stats) * adjoint).sum() / values.numel()))
                fd = (objectives[1] - objectives[0]) / (2 * step)
                comparisons.append({"step": step, "ad": ad, "fd": fd,
                    "absolute_gap": abs(ad - fd), "relative_gap": abs(ad - fd) / max(abs(ad), abs(fd), 1e-12)})
            row["ad_fd"][name] = {"gradient_norm": 0. if grad[index] is None else float(grad[index].norm()),
                "finite": grad[index] is not None and bool(torch.isfinite(grad[index]).all()), "steps": comparisons}
        return row

    try:
        for index in range(22):
            sample = datasets["test"][index]
            result["cases"].append(finite_case(sample))
            if str(sample["case_id"]) in REPRESENTATIVES:
                result["full_four"].append(finite_case(sample, full=True))
                result["derivatives"].append(derivatives(sample, "exposed validation"))
            write_json(output_dir / "dependency.json", result)
        counts = datasets["train"].selected_module_counts
        indices = (int(np.argmin(counts)), int(np.argmax(counts)))
        for index in indices:
            result["derivatives"].append(derivatives(datasets["train"][index], "TRAIN input-selected M endpoint"))
        write_json(output_dir / "dependency.json", result)
        receipt["status"] = "completed"
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        receipt.update(end_unix=time(), elapsed_seconds=time() - start)
        if receipt["status"] == "running":
            receipt["status"] = "failed"
        write_json(output_dir / "receipt.json", receipt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
