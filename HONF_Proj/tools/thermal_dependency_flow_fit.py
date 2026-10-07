#!/usr/bin/env python3
"""Fit exactly one matched flow reader; never solve or train the thermal parent."""
import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter, time

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "Case_ThermalChannel/src", ROOT / "tools"):
    sys.path.insert(0, str(source))

import numpy as np
import torch
from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.data.development_split import development_case_ids, resolve_development_manifest
from channelthermal.dependency_flow import CASE_CAPABILITY, DEPENDENCY_ID, ThermalFlowReader
from channelthermal.flow_curl import (
    GENERATOR_CURL_ID,
    build_near_obstacle_curl_stencil,
    torch_normalized_curl_from_stencil,
)
from channelthermal.training.checkpoints import _file_sha256, atomic_save_checkpoint_payload
from channelthermal.training.stop_request import acknowledge_stop, stop_requested
from thermal_development import validate_generated_output
from torch.utils.data import default_collate

from honf_runtime.compat import load_trusted_checkpoint, recursive_to_device, set_seed
from honf_runtime.run_store import atomic_write_json

FLOW_CHILD_SCHEDULE = {"identity": "d_sep_schedule_child1500_v1", "parent_absolute_epoch": 1000,
    "additional_epochs": 1500, "warmup_epochs": 20, "hold_until_additional_epoch": 500,
    "warmup_initial_lr": 1e-6, "maximum_lr": 1e-4, "final_lr": 1e-6}
FLOW_REFINEMENT_SCHEDULE = {"identity": "d_sep_near_boundary_500_v1", "parent_absolute_epoch": 2500,
    "additional_epochs": 500, "warmup_epochs": 20, "hold_until_additional_epoch": 200,
    "warmup_initial_lr": 1e-6, "maximum_lr": 3e-5, "final_lr": 3e-6,
    "decay": "cosine"}


def flow_child_learning_rate(additional_epoch):
    """Declared new schedule; the historical zero-floor schedule is untouched."""
    age = int(additional_epoch)
    if age <= 20:
        return 1e-6 + (1e-4 - 1e-6) * max(age - 1, 0) / 19
    if age <= 500:
        return 1e-4
    return 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * min(age - 500, 1000) / 1000))


def flow_refinement_learning_rate(additional_epoch):
    """Five-hundred-epoch continuation schedule declared for the matched pair."""
    age = int(additional_epoch)
    if not 1 <= age <= FLOW_REFINEMENT_SCHEDULE["additional_epochs"]:
        raise ValueError("Refinement age must be in 1..500.")
    if age <= FLOW_REFINEMENT_SCHEDULE["warmup_epochs"]:
        if age == FLOW_REFINEMENT_SCHEDULE["warmup_epochs"]:
            return 3e-5
        return 1e-6 + (3e-5 - 1e-6) * (age - 1) / 19
    if age <= FLOW_REFINEMENT_SCHEDULE["hold_until_additional_epoch"]:
        return 3e-5
    fraction = (age - 200) / 300
    return 3e-6 + .5 * (3e-5 - 3e-6) * (1 + math.cos(math.pi * fraction))


def read_cases(parent, split, *, include_grid=False):
    config = copy.deepcopy(parent["train_config"]["dataset"])
    manifest = resolve_development_manifest(config, config["packed_h5_path"])
    ids = development_case_ids(manifest, split)
    if len(ids) != (150 if split == "train" else 22):
        raise ValueError("Flow comparison requires the exact fixed25_v1 150/22 selection.")
    normalizer = H5Normalizer(parent["global_normalization_stats"])
    dataset = GlobalChannelThermalDataset(config["packed_h5_path"], split=split, points_per_case=None,
        normalize_inputs=config["normalize_inputs"], normalize_targets=config["normalize_targets"],
        random_point_sampling=False, seed=0, normalizer=normalizer, case_ids=ids, include_grid=include_grid)
    cases = []
    for index in range(len(dataset)):
        sample = dataset[index]
        keys = ["structure", "query_xy", "field_targets", "point_weights", "case_id"]
        if include_grid:
            keys.extend(("x_grid", "y_grid", "steady_field", "module_mask"))
        cases.append({key: sample[key] for key in keys})
    dataset.close()
    return dataset, cases, manifest


def sample_batch(cases, indices, epoch, device, training):
    samples = []
    for index in indices:
        case = cases[int(index)]
        size = len(case["query_xy"])
        # Same maintained dataset query stream, with explicit seed0 and fixed
        # validation seed1000. Case order never alters receiver selection.
        seed = (0 if training else 1000) + int(index) * 104729 + (epoch * 1000003 if training else 0)
        query_ids = np.random.default_rng(seed).choice(size, min(1024, size), replace=False)
        samples.append({"structure": case["structure"], "query_xy": case["query_xy"][query_ids],
                        "field_targets": case["field_targets"][query_ids, :4],
                        "point_weights": case["point_weights"][query_ids]})
    return recursive_to_device(default_collate(samples), device)


def case_losses(prediction, batch):
    square = (prediction - batch["field_targets"]).square()
    weight = batch["point_weights"]
    return (square * weight[..., None]).sum(1) / weight.sum(1).clamp_min(1e-12)[:, None]


@torch.no_grad()
def validate(model, cases, device):
    rows = []
    for start in range(0, len(cases), 8):
        batch = sample_batch(cases, range(start, min(start + 8, len(cases))), 0, device, False)
        loss = case_losses(model(batch["structure"], batch["query_xy"]), batch).cpu().numpy()
        rows.extend({"case_id": cases[start + offset]["case_id"], "mse_by_flow_channel": row.tolist()}
                    for offset, row in enumerate(loss))
    return float(np.mean([row["mse_by_flow_channel"] for row in rows])), rows


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["epoch"] for row in history], [row["train_flow_mse"] for row in history], label="TRAIN flow MSE")
    reviews = [row for row in history if "validation_flow_mse" in row]
    ax.plot([row["epoch"] for row in reviews], [row["validation_flow_mse"] for row in reviews], "o-", label="22 exposed DEV")
    ax.set(xlabel="New flow-only epochs", ylabel="Equal-case/channel normalized MSE", yscale="log")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def attach_near_curl_stencils(cases):
    """Attach the same input-only 128-centre stencil to every paired fit."""
    for case in cases:
        structure = case["structure"]
        grid_shape = case["module_mask"].shape
        dx = float(np.asarray(structure["domain_length_x"]).reshape(-1)[0]) / grid_shape[1]
        dy = float(np.asarray(structure["domain_length_y"]).reshape(-1)[0]) / grid_shape[0]
        case_id = str(case["case_id"])
        case["near_curl_stencil"] = build_near_obstacle_curl_stencil(
            x_grid=case["x_grid"], y_grid=case["y_grid"],
            module_centers=structure["module_centers"], module_present=structure["module_present"],
            module_mask=case["module_mask"], omega_target=case["steady_field"][..., 3],
            radius=float(np.asarray(structure["material_params"])[5]), dx=dx, dy=dy,
            max_centers=128, seed=int(case_id),
        )
    return cases


def sample_near_auxiliary_batch(cases, indices, device):
    rows = []
    for index in indices:
        case = cases[int(index)]
        stencil = case["near_curl_stencil"]
        rows.append({
            "structure": case["structure"],
            "query_xy": stencil.query_xy,
            "solid_nodes": stencil.solid_nodes,
            "x_span": stencil.x_span,
            "y_span": stencil.y_span,
            "valid": stencil.valid,
            "omega_target": stencil.omega_target,
        })
    return recursive_to_device(default_collate(rows), device)


def near_curl_loss_components(prediction, batch, normalization):
    n_centers = int(batch["valid"].shape[1])
    normalized_flow = prediction.reshape(prediction.shape[0], n_centers, 5, 4)
    omega_prediction, curl_prediction = torch_normalized_curl_from_stencil(
        normalized_flow,
        batch["solid_nodes"].bool(),
        batch["x_span"],
        batch["y_span"],
        normalization["field_mean_by_channel"],
        normalization["field_std_by_channel"],
    )
    mean = prediction.new_tensor(np.asarray(normalization["field_mean_by_channel"]).reshape(-1)[:4])
    std = prediction.new_tensor(np.asarray(normalization["field_std_by_channel"]).reshape(-1)[:4])
    target = (batch["omega_target"] - mean[3]) / std[3]
    valid = batch["valid"].to(prediction.dtype)
    denominator = valid.sum(-1).clamp_min(1.0)
    target_loss = ((omega_prediction - target).square() * valid).sum(-1) / denominator
    consistency_loss = ((omega_prediction - curl_prediction).square() * valid).sum(-1) / denominator
    return target_loss, consistency_loss


@torch.no_grad()
def validate_full_grid_flow(model, cases, device, normalization):
    """Measure physical u/v/p/omega over the native DEV grid and strata."""
    model.eval()
    mean = np.asarray(normalization["field_mean_by_channel"], dtype=np.float64).reshape(-1)[:4]
    std = np.asarray(normalization["field_std_by_channel"], dtype=np.float64).reshape(-1)[:4]
    if np.any(std <= 0):
        raise ValueError("Flow validation requires positive native channel scales.")
    names = ("u", "v", "p", "omega")
    rows = []
    for case in cases:
        query_xy = np.stack((case["x_grid"].reshape(-1), case["y_grid"].reshape(-1)), axis=-1)
        structure = recursive_to_device(default_collate([case["structure"]]), device)
        queries = torch.as_tensor(query_xy[None], device=device, dtype=torch.float32)
        normalized_prediction = model(structure, queries)[0].cpu().numpy().astype(np.float64)
        prediction = normalized_prediction * std + mean
        reference = case["steady_field"].reshape(-1, 5)[:, :4].astype(np.float64)
        error = prediction - reference
        solid = np.asarray(case["module_mask"], dtype=bool).reshape(-1)
        fluid = ~solid
        centers = np.asarray(case["structure"]["module_centers"], dtype=np.float64)
        present = np.asarray(case["structure"]["module_present"], dtype=bool)
        radius = float(np.asarray(case["structure"]["material_params"])[5])
        distance = np.full(len(query_xy), np.inf, dtype=np.float64)
        for center in centers[present]:
            distance = np.minimum(distance, np.linalg.norm(query_xy - center, axis=-1) - radius)
        near = fluid & (distance <= radius)
        far = fluid & ~near
        if not fluid.any() or not near.any() or not far.any():
            raise ValueError(f"Empty flow validation stratum for case {case['case_id']}.")
        row = {"case_id": str(case["case_id"]), "M": int(present.sum()),
               "fluid_count": int(fluid.sum()), "near_count": int(near.sum()), "far_count": int(far.sum()),
               "rmse": {}, "normalized_fluid_mse": float(np.mean((error[fluid] / std) ** 2))}
        for region, mask in (("fluid", fluid), ("near", near), ("far", far)):
            row["rmse"][region] = {
                name: float(np.sqrt(np.mean(error[mask, channel] ** 2)))
                for channel, name in enumerate(names)
            }
        rows.append(row)
    summary = {"cases": len(rows), "partition": "fixed22_v1 exposed DEV, full native Q8192",
               "near_definition": "fluid centers within 2 native radii of module centers (surface distance <= 1 radius)",
               "far_definition": "remaining fluid centers", "fluid": {}}
    for region in ("fluid", "near", "far"):
        summary[region] = {}
        for name in names:
            values = np.asarray([row["rmse"][region][name] for row in rows], dtype=np.float64)
            worst = int(np.argmax(values))
            summary[region][name] = {
                "equal_case_rmse_mean": float(values.mean()),
                "case_rmse_p90": float(np.quantile(values, 0.9)),
                "case_rmse_max": float(values[worst]),
                "worst_case_id": rows[worst]["case_id"],
            }
    summary["fluid"]["equal_case_normalized_mse_mean"] = float(np.mean([row["normalized_fluid_mse"] for row in rows]))
    return summary, rows


def plot_refinement_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["additional_epoch"] for row in history], [row["train_flow_mse"] for row in history],
            label="TRAIN primary Q1024 MSE")
    reviews = [row for row in history if "validation_normalized_fluid_mse" in row]
    ax.plot([row["additional_epoch"] for row in reviews],
            [row["validation_normalized_fluid_mse"] for row in reviews], "o-", label="DEV22 full-grid MSE")
    ax.set(xlabel="Additional matched continuation epochs", ylabel="Mean normalized flow MSE", yscale="log")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _optimizer_step_summary(optimizer_state):
    steps = [int(value["step"].item()) for value in optimizer_state.get("state", {}).values()
             if "step" in value]
    if not steps or set(steps) != {10000}:
        raise ValueError(f"D-sep2500 AdamW state must retain all 10,000 update steps; observed {sorted(set(steps))}.")
    return {"parameter_states": len(steps), "minimum_step": min(steps), "maximum_step": max(steps)}


def run_bounded_refinement(args):
    """Run one explicitly matched D-sep2500 continuation child."""
    if args.policy != "D-sep" or args.schedule_parent or args.epochs not in (100, 500):
        raise ValueError("Bounded refinements are D-sep-only; stops are 100 for review or 500 after review.")
    start = perf_counter()
    started_unix = time()
    torch.set_num_threads(1)
    set_seed(0)
    device = torch.device(args.device)
    output = validate_generated_output(args.output)
    parent_path = Path(args.parent).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Preserve earlier refinement evidence; choose a fresh output or resume latest explicitly.")

    session = {"status": "running", "pid": os.getpid(), "started_unix": started_unix,
               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical_device": str(device),
               "refinement": args.refinement, "requested_additional_stop": args.epochs,
               "parent_checkpoint": str(parent_path)}
    sessions_path = output / "resource_sessions.json"
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    session_index = len(sessions)
    sessions.append(session)
    atomic_write_json(sessions_path, sessions)
    run_manifest = {"status": "running", "refinement": args.refinement,
                    "requested_additional_stop": args.epochs, "solver_attempts": 0}
    atomic_write_json(output / "run_manifest.json", run_manifest)
    last_absolute_epoch = FLOW_REFINEMENT_SCHEDULE["parent_absolute_epoch"]
    aggregate_process_seconds = 0.0
    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        parent = load_trusted_checkpoint(parent_path, map_location="cpu")
        parent_sha = _file_sha256(parent_path)
        parent_identity = parent.get("fit_identity", {})
        if (parent.get("dependency_policy") != "D-sep" or parent.get("epoch") != 2500
                or parent_identity.get("schedule_total_epochs") != 2500
                or "optimizer_state_dict" not in parent):
            raise ValueError("Continuation parent must be the literal D-sep2500 endpoint with AdamW state.")
        optimizer_steps = _optimizer_step_summary(parent["optimizer_state_dict"])
        _, train_cases, manifest = read_cases(parent, "train", include_grid=True)
        _, validation_cases, val_manifest = read_cases(parent, "test", include_grid=True)
        if (manifest != val_manifest or manifest["manifest_sha256"] != parent_identity.get("development_manifest_sha256")
                or len(train_cases) != 150 or len(validation_cases) != 22):
            raise ValueError("Refinement data must retain the literal fixed25_v1 150/22 membership and normalizer.")
        if ([row["case_id"] for row in train_cases] != parent_identity.get("training_case_ids")
                or [row["case_id"] for row in validation_cases] != parent_identity.get("validation_case_ids")):
            raise ValueError("Refinement case order differs from the D-sep2500 checkpoint binding.")
        if args.refinement == "near-consistency":
            attach_near_curl_stencils(train_cases)
        model = ThermalFlowReader("D-sep", parent["flow_reader_config"]).to(device)
        model.load_state_dict(parent["flow_state_dict"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
        optimizer.load_state_dict(parent["optimizer_state_dict"])
        identity = {
            "dependency_identity": DEPENDENCY_ID,
            "dependency_policy": "D-sep",
            "case_capability": CASE_CAPABILITY,
            "thermal_parent_checkpoint": parent["thermal_parent_checkpoint"],
            "thermal_parent_sha256": parent["thermal_parent_sha256"],
            "flow_reader_config": parent["flow_reader_config"],
            "seed": 0,
            "points_per_case": 1024,
            "microbatch_cases": 8,
            "effective_cases": 48,
            "start_absolute_epoch": 2500,
            "additional_epochs": 500,
            "development_manifest_sha256": manifest["manifest_sha256"],
            "training_case_ids": parent_identity["training_case_ids"],
            "validation_case_ids": parent_identity["validation_case_ids"],
            "common_start_checkpoint": str(parent_path),
            "common_start_checkpoint_sha256": parent_sha,
            "common_start_optimizer_step": optimizer_steps,
            "comparison_pair_id": "dsep_near_boundary_refinement_20261006_v1",
            "refinement": args.refinement,
            "schedule": FLOW_REFINEMENT_SCHEDULE,
            "primary_query_stream": "thermal_dependency_flow_fit.sample_batch seed0, absolute epoch, Q1024",
            "near_objective": ({"target_weight": 0.1, "curl_consistency_weight": 0.1,
                "stencil_id": GENERATOR_CURL_ID,
                "band": "fluid grid centers with 0 <= distance-to-surface <= one native radius",
                "max_centers_per_case": 128, "neighbours_per_center": 5}
                if args.refinement == "near-consistency" else {"enabled": False}),
        }
        begin_additional, best, history = 0, float("inf"), []
        if args.resume:
            saved = load_trusted_checkpoint(args.resume, map_location="cpu")
            if saved.get("fit_identity") != identity:
                raise ValueError("Resume changed the explicit parent/data/schedule/objective identity.")
            model.load_state_dict(saved["flow_state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            begin_additional = int(saved["additional_epoch"])
            best = float(saved["best_metric"])
            history = saved["history"]
            aggregate_process_seconds = float(saved["aggregate_process_seconds"])
        elif (output / "fit_identity.json").exists():
            raise FileExistsError("Existing refinement requires its explicit latest checkpoint resume.")
        if begin_additional >= args.epochs:
            raise ValueError("Requested refinement stop must be later than the saved additional age.")
        atomic_write_json(output / "fit_identity.json", identity)
        if args.refinement == "near-consistency":
            aux_centers_per_epoch = sum(int(case["near_curl_stencil"].valid.sum()) for case in train_cases)
            aux_queries_per_epoch = sum(len(case["near_curl_stencil"].query_xy) for case in train_cases)
        else:
            aux_centers_per_epoch = aux_queries_per_epoch = 0
        session.update({"resumed_from_additional_epoch": begin_additional,
                        "parent_sha256": parent_sha, "optimizer_step_state": optimizer_steps,
                        "training_case_count": len(train_cases), "validation_case_count": len(validation_cases)})
        atomic_write_json(sessions_path, sessions)
        stats = parent["global_normalization_stats"]
        for additional_epoch in range(begin_additional + 1, args.epochs + 1):
            absolute_epoch = FLOW_REFINEMENT_SCHEDULE["parent_absolute_epoch"] + additional_epoch
            last_absolute_epoch = absolute_epoch
            epoch_start = perf_counter()
            used_learning_rate = flow_refinement_learning_rate(additional_epoch)
            for group in optimizer.param_groups:
                group["lr"] = used_learning_rate
            order = np.random.default_rng(absolute_epoch).permutation(len(train_cases))
            primary_sum, target_sum, consistency_sum = 0.0, 0.0, 0.0
            gradient_norms = []
            model.train()
            for effective_start in range(0, len(order), 48):
                effective = order[effective_start:effective_start + 48]
                optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, len(effective), 8):
                    indices = effective[micro_start:micro_start + 8]
                    batch = sample_batch(train_cases, indices, absolute_epoch, device, True)
                    primary_prediction = model(batch["structure"], batch["query_xy"])
                    primary_by_case = case_losses(primary_prediction, batch).mean(-1)
                    total_by_case = primary_by_case
                    primary_sum += float(primary_by_case.detach().sum())
                    if args.refinement == "near-consistency":
                        near_batch = sample_near_auxiliary_batch(train_cases, indices, device)
                        near_prediction = model(near_batch["structure"], near_batch["query_xy"])
                        target_by_case, consistency_by_case = near_curl_loss_components(near_prediction, near_batch, stats)
                        total_by_case = total_by_case + 0.1 * target_by_case + 0.1 * consistency_by_case
                        target_sum += float(target_by_case.detach().sum())
                        consistency_sum += float(consistency_by_case.detach().sum())
                    loss = total_by_case.sum() / len(effective)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite bounded flow-refinement loss.")
                    loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm) or norm <= 0:
                    raise FloatingPointError("Invalid actual flow-refinement gradient.")
                gradient_norms.append(float(norm))
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {
                "epoch": absolute_epoch, "additional_epoch": additional_epoch,
                "learning_rate": used_learning_rate,
                "train_flow_mse": primary_sum / len(train_cases),
                "train_near_omega_target_mse": target_sum / len(train_cases),
                "train_near_curl_consistency_mse": consistency_sum / len(train_cases),
                "train_objective": primary_sum / len(train_cases) + 0.1 * target_sum / len(train_cases)
                    + 0.1 * consistency_sum / len(train_cases),
                "train_seconds": perf_counter() - epoch_start,
                "case_visits": len(train_cases), "optimizer_updates": len(gradient_norms),
                "primary_queries": len(train_cases) * 1024,
                "near_aux_valid_centers": aux_centers_per_epoch,
                "near_aux_query_evaluations": aux_queries_per_epoch,
                "gradient_norm_mean": float(np.mean(gradient_norms)),
            }
            requested_stop = stop_requested(output)
            review = additional_epoch % 100 == 0 or additional_epoch == args.epochs or requested_stop
            if review:
                model.eval()
                validation_start = perf_counter()
                validation, validation_rows = validate_full_grid_flow(model, validation_cases, device, stats)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                validation_seconds = perf_counter() - validation_start
                score = validation["fluid"]["equal_case_normalized_mse_mean"]
                row.update(validation_normalized_fluid_mse=score, validation_seconds=validation_seconds)
                metrics_path = output / f"validation_epoch_{absolute_epoch:04d}.json"
                atomic_write_json(metrics_path, {"summary": validation, "rows": validation_rows})
            history.append(row)
            print(json.dumps(row), flush=True)
            if review:
                improved = score < best
                best = min(best, score)
                payload = dict(
                    checkpoint_schema_version=1, case_id="ThermalChannel", model_family="honf_forward",
                    workflow="forward", stage=DEPENDENCY_ID, **identity, fit_identity=identity,
                    flow_state_dict=model.state_dict(), epoch=absolute_epoch, current_epoch=absolute_epoch,
                    additional_epoch=additional_epoch, best_metric=best,
                    optimizer_state_dict=optimizer.state_dict(), history=history,
                    aggregate_process_seconds=aggregate_process_seconds + perf_counter() - start,
                    global_normalization_stats=parent["global_normalization_stats"],
                    train_config=parent["train_config"], model_config=parent["model_config"],
                )
                atomic_save_checkpoint_payload(output / "latest_model.pt", payload)
                atomic_save_checkpoint_payload(output / f"epoch_{absolute_epoch:04d}_model.pt", payload)
                if improved:
                    atomic_save_checkpoint_payload(output / "best_by_field_mse_model.pt", payload)
                atomic_write_json(output / "history.json", history)
                plot_refinement_history(history, output / "flow_learning.pdf")
                run_manifest.update({"status": "review" if additional_epoch == 100 else "running",
                    "last_completed_epoch": absolute_epoch, "last_additional_epoch": additional_epoch,
                    "best_metric": best, "validation": validation, "checkpoint": str(output / "latest_model.pt")})
                atomic_write_json(output / "run_manifest.json", run_manifest)
            if requested_stop:
                acknowledge_stop(output, epoch=absolute_epoch)
                break
        elapsed = perf_counter() - start
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        summary = {
            "status": "completed", "refinement": args.refinement,
            "completed_epoch": last_absolute_epoch, "completed_additional_epoch": additional_epoch,
            "new_epochs": additional_epoch - begin_additional,
            "new_case_visits": (additional_epoch - begin_additional) * len(train_cases),
            "new_optimizer_updates": (additional_epoch - begin_additional) * 4,
            "primary_queries_per_epoch": len(train_cases) * 1024,
            "near_aux_valid_centers_per_epoch": aux_centers_per_epoch,
            "near_aux_query_evaluations_per_epoch": aux_queries_per_epoch,
            "process_seconds": elapsed, "aggregate_process_seconds": aggregate_process_seconds + elapsed,
            "started_unix": started_unix, "ended_unix": time(),
            "visible_physical_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical_device": str(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "thermal_optimizer_parameters": 0, "solver_attempts": 0,
        }
        atomic_write_json(output / "fit_summary.json", summary)
        run_manifest.update({"status": "completed", "last_completed_epoch": last_absolute_epoch,
                             "last_additional_epoch": additional_epoch, "best_metric": best})
        atomic_write_json(output / "run_manifest.json", run_manifest)
        print(json.dumps({"status": "completed", "epoch": last_absolute_epoch,
                          "additional_epoch": additional_epoch, "output": str(output)}), flush=True)
        return 0
    except Exception as exc:
        run_manifest.update({"status": "failed", "last_completed_epoch": last_absolute_epoch,
                             "error_type": type(exc).__name__, "error": str(exc)})
        atomic_write_json(output / "run_manifest.json", run_manifest)
        raise
    finally:
        ended_unix = time()
        elapsed = perf_counter() - start
        peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" and torch.cuda.is_initialized() else 0
        sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
        if session_index < len(sessions):
            sessions[session_index].update({"status": "finished" if run_manifest.get("status") == "completed" else run_manifest.get("status", "failed"),
                "ended_unix": ended_unix, "elapsed_seconds": elapsed, "peak_allocated_bytes": peak_bytes,
                "last_completed_epoch": last_absolute_epoch})
            atomic_write_json(sessions_path, sessions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--policy", required=True, choices=("D-sep", "D-open"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100, choices=(100, 500, 1000, 1500, 2500))
    parser.add_argument("--resume")
    parser.add_argument("--schedule-parent", help="Opt-in new D-sep1500-epoch schedule child from literal1000.")
    parser.add_argument("--refinement", choices=("control", "near-consistency"),
        help="Run the bounded matched D-sep2500 control or near-boundary consistency child.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.refinement:
        return run_bounded_refinement(args)
    is_child = args.schedule_parent is not None
    if is_child and (args.policy != "D-sep" or args.epochs not in (1500, 2500)):
        raise ValueError("The schedule child is D-sep only, absolute stop1500 or2500.")
    if not is_child and args.epochs not in (100, 500, 1000):
        raise ValueError("Historical flow fits retain their original100/500/1000 stops.")
    start = perf_counter()
    started_unix = time()
    torch.set_num_threads(1); set_seed(0)
    device = torch.device(args.device)
    output = validate_generated_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    parent_path = Path(args.parent).resolve()
    parent = load_trusted_checkpoint(parent_path, map_location="cpu")
    parent_sha = _file_sha256(parent_path)
    # Loading the exact parent for identity/normalization is not a thermal
    # model execution. Only flow parameters participate in this optimizer.
    _train_dataset, train_cases, manifest = read_cases(parent, "train")
    _, val_cases, val_manifest = read_cases(parent, "test")
    if manifest != val_manifest:
        raise ValueError("Train/validation manifest differs.")
    model = ThermalFlowReader(args.policy).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    identity = {"dependency_identity": DEPENDENCY_ID, "dependency_policy": args.policy,
        "case_capability": CASE_CAPABILITY,
        "thermal_parent_checkpoint": str(parent_path), "thermal_parent_sha256": parent_sha,
        "flow_reader_config": model.reader.config, "seed": 0, "points_per_case": 1024,
        "microbatch_cases": 8, "effective_cases": 48, "schedule_total_epochs": 1000,
        "development_manifest_sha256": manifest["manifest_sha256"],
        "training_case_ids": [row["case_id"] for row in train_cases],
        "validation_case_ids": [row["case_id"] for row in val_cases],
        "learning_rate": 3e-4, "weight_decay": 1e-5, "gradient_clip": 1., "flow_channels": ["u", "v", "p", "omega"]}
    begin, best, history, process_seconds = 0, float("inf"), [], 0.
    if is_child:
        schedule_parent_path = Path(args.schedule_parent).resolve()
        schedule_parent = load_trusted_checkpoint(schedule_parent_path, map_location="cpu")
        if (schedule_parent.get("dependency_policy") != "D-sep" or schedule_parent.get("epoch") != 1000
                or schedule_parent.get("fit_identity") != identity):
            raise ValueError("Schedule child requires the literal matched D-sep1000 parent and unchanged data/reader identity.")
        identity.update(schedule_total_epochs=2500, schedule_child=FLOW_CHILD_SCHEDULE,
            schedule_parent_checkpoint=str(schedule_parent_path), schedule_parent_sha256=_file_sha256(schedule_parent_path))
        model.load_state_dict(schedule_parent["flow_state_dict"], strict=True)
        optimizer.load_state_dict(schedule_parent["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = 1e-6
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda=lambda completed: flow_child_learning_rate(completed + 1) / 1e-6)
        begin = 1000
        atomic_write_json(output / "schedule_attachment.json", {"parent_checkpoint": str(schedule_parent_path),
            "parent_sha256": identity["schedule_parent_sha256"], "new_schedule": FLOW_CHILD_SCHEDULE,
            "retained_optimizer_step": 4000, "new_optimizer_steps": 6000,
            "change": "new LR/groupinitial_lr only; literal source tensors/moments/data/normalization retained"})
    if args.resume:
        saved = load_trusted_checkpoint(args.resume, map_location="cpu")
        if saved["fit_identity"] != identity:
            raise ValueError("Resume changed the explicit flow/parent/data/schedule identity.")
        model.load_state_dict(saved["flow_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        begin, best, history = saved["epoch"], saved["best_metric"], saved["history"]
        process_seconds = saved["aggregate_process_seconds"]
    elif (output / "fit_identity.json").exists():
        raise FileExistsError("Existing fit requires its explicit latest checkpoint resume.")
    if begin >= args.epochs:
        raise ValueError("Requested stop must be later than the saved flow-fit age.")
    atomic_write_json(output / "fit_identity.json", identity)
    atomic_write_json(output / "run_manifest.json", {"status": "running", "policy": args.policy, "epochs": args.epochs})
    for epoch in range(begin + 1, args.epochs + 1):
        epoch_start = perf_counter()
        used_learning_rate = optimizer.param_groups[0]["lr"]
        order = np.random.default_rng(epoch).permutation(len(train_cases))
        total_loss, gradient_norms = 0., []
        model.train()
        for effective_start in range(0, len(order), 48):
            effective = order[effective_start:effective_start + 48]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(effective), 8):
                indices = effective[micro_start:micro_start + 8]
                batch = sample_batch(train_cases, indices, epoch, device, True)
                losses = case_losses(model(batch["structure"], batch["query_xy"]), batch).mean(-1)
                loss = losses.sum() / len(effective)
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite native flow loss")
                loss.backward()
                total_loss += float(losses.sum().detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm) or norm <= 0: raise FloatingPointError("Invalid actual flow gradient")
            gradient_norms.append(float(norm)); optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        row = {"epoch": epoch, "train_flow_mse": total_loss / len(train_cases),
            "train_seconds": perf_counter() - epoch_start, "case_visits": len(train_cases),
            "optimizer_updates": 4, "queries": len(train_cases) * 1024,
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "learning_rate": used_learning_rate if is_child else optimizer.param_groups[0]["lr"]}
        requested_stop = stop_requested(output)
        review = epoch % 100 == 0 or epoch == args.epochs or requested_stop
        if review:
            model.eval(); validation_start = perf_counter()
            score, rows = validate(model, val_cases, device)
            row.update(validation_flow_mse=score, validation_seconds=perf_counter() - validation_start)
            atomic_write_json(output / f"validation_epoch_{epoch:04d}.json", {"score": score, "rows": rows})
        history.append(row)
        print(json.dumps(row), flush=True)
        if review:
            improved = score < best
            best = min(best, score)
            payload = dict(checkpoint_schema_version=1, case_id="ThermalChannel", model_family="honf_forward",
                workflow="forward", stage=DEPENDENCY_ID, **identity, fit_identity=identity,
                flow_state_dict=model.state_dict(), epoch=epoch, current_epoch=epoch, best_metric=best,
                optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                history=history, aggregate_process_seconds=process_seconds + perf_counter() - start,
                global_normalization_stats=parent["global_normalization_stats"],
                train_config=parent["train_config"], model_config=parent["model_config"])
            atomic_save_checkpoint_payload(output / "latest_model.pt", payload)
            if epoch % 100 == 0:
                atomic_save_checkpoint_payload(output / f"epoch_{epoch:04d}_model.pt", payload)
            if improved:
                atomic_save_checkpoint_payload(output / "best_by_field_mse_model.pt", payload)
            atomic_write_json(output / "history.json", history)
            plot_history(history, output / "flow_learning.pdf")
            atomic_write_json(output / "run_manifest.json", {"status": "review" if epoch == 100 else "running",
                "last_completed_epoch": epoch, "best_metric": best, "aggregate_process_seconds": payload["aggregate_process_seconds"]})
        if requested_stop:
            acknowledge_stop(output, epoch=epoch)
            break
    summary = {"policy": args.policy, "completed_epoch": epoch,
        "best_metric": best, "new_case_visits": (epoch-begin)*150, "new_optimizer_updates": (epoch-begin)*4,
        "new_queries": (epoch-begin)*150*1024, "process_seconds": perf_counter()-start,
        "aggregate_process_seconds": process_seconds+perf_counter()-start,
        "started_unix": started_unix, "ended_unix": time(),
        "visible_physical_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical_device": str(device),
        "parameters": sum(value.numel() for value in model.parameters()), "thermal_optimizer_parameters": 0}
    atomic_write_json(output / "fit_summary.json", summary)
    session_path = output / "resource_sessions.json"
    sessions = json.loads(session_path.read_text()) if session_path.exists() else []
    sessions.append(dict(summary, resumed_from_epoch=begin))
    atomic_write_json(session_path, sessions)
    if not requested_stop:
        atomic_write_json(output / "run_manifest.json", {"status": "completed", "last_completed_epoch": epoch,
            "best_metric": best, "aggregate_process_seconds": process_seconds+perf_counter()-start})
    print(json.dumps({"status": "completed", "epoch": epoch, "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
