#!/usr/bin/env python3
"""Fit or resume a fresh D-sep component on all original TRAIN cases.

This is the first manual stage of the R-direct full-TRAIN research recipe.
The historical fixed25 flow fitter remains the development entrypoint.
"""
from __future__ import annotations

import argparse
import hashlib
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
from channelthermal.dependency_flow import (
    CASE_CAPABILITY,
    DEPENDENCY_ID,
    ThermalFlowReader,
)
from channelthermal.training.checkpoints import (
    _file_sha256,
    atomic_save_checkpoint_payload,
)
from channelthermal.training.stop_request import acknowledge_stop, stop_requested
from honf_runtime.compat import load_trusted_checkpoint, recursive_to_device, set_seed
from honf_runtime.run_layout import RunLayout
from honf_runtime.run_store import atomic_write_json
from thermal_development import validate_generated_output
from thermal_formal_profile import (
    FORMAL_TRAIN_SCOPE,
    bind_formal_validation,
    bind_original_train,
    ensure_formal_resume_identity,
    fit_formal_normalizer,
    formal_train_config,
    initialize_formal_output,
    stats_sha256,
    validate_formal_milestones,
)


def _validate_profile(profile):
    required = {"schema_version", "profile_name", "workflow_scope", "dependency_policy", "seed",
        "data", "budget", "schedule", "optimizer", "checkpointing"}
    if set(profile) != required or profile.get("schema_version") != 1:
        raise ValueError("Formal D-sep profile fields or schema differ from the maintained format.")
    if profile["workflow_scope"] != FORMAL_TRAIN_SCOPE or profile["dependency_policy"] != "D-sep":
        raise ValueError("Formal flow profile requires the explicit D-sep full-TRAIN identity.")
    if set(profile["data"]) != {"training_split", "normalization_source", "expected_train_case_count",
            "startup_validation", "formal_validation"}:
        raise ValueError("Formal D-sep data profile fields differ from the maintained format.")
    if profile["data"]["training_split"] != "train" or profile["data"]["normalization_source"] != "all_original_train_only":
        raise ValueError("Formal D-sep must use the complete original TRAIN split and its own normalizer.")
    if set(profile["data"]["startup_validation"]) != {"scope", "manifest_sha256", "expected_case_count", "maximum_new_epochs"}:
        raise ValueError("Formal D-sep startup validation must declare a bounded DEV panel.")
    if (profile["data"]["startup_validation"]["scope"] != "fixed25_v1_DEV22_exposed"
            or int(profile["data"]["startup_validation"]["expected_case_count"]) != 22
            or int(profile["data"]["startup_validation"]["maximum_new_epochs"]) != 3):
        raise ValueError("Formal D-sep startup may use only the fixed25_v1 DEV22 panel and up to three epochs.")
    if set(profile["data"]["formal_validation"]) != {"primary_scope", "compatibility_scope",
            "excluded_training_duplicate_case_id", "expected_primary_case_count", "expected_compatibility_case_count"}:
        raise ValueError("Formal D-sep validation must declare canonical89 and original90 panels.")
    if set(profile["budget"]) != {"fluid_queries", "microbatch_cases", "effective_cases"}:
        raise ValueError("Formal D-sep budget fields differ from the maintained format.")
    if (min(int(value) for value in profile["budget"].values()) <= 0
            or profile["budget"]["effective_cases"] < profile["budget"]["microbatch_cases"]):
        raise ValueError("Formal D-sep sample and batch budgets must be positive and consistent.")
    if set(profile["schedule"]) != {"horizon_epochs", "hold_fraction", "initial_lr", "final_lr"}:
        raise ValueError("Formal D-sep schedule fields differ from the maintained format.")
    if (int(profile["schedule"]["horizon_epochs"]) < 1 or not 0 <= float(profile["schedule"]["hold_fraction"]) < 1
            or not 0 < float(profile["schedule"]["final_lr"]) <= float(profile["schedule"]["initial_lr"])):
        raise ValueError("Formal D-sep schedule needs a positive floor and a valid horizon.")
    if set(profile["optimizer"]) != {"name", "weight_decay", "gradient_clip"} or profile["optimizer"]["name"] != "AdamW":
        raise ValueError("Formal D-sep optimizer must be the declared AdamW profile.")
    if set(profile["checkpointing"]) != {"monitoring_interval_epochs", "save_latest", "save_best_field", "milestone_epochs"}:
        raise ValueError("Formal D-sep checkpoint profile fields differ from the maintained format.")
    if (int(profile["checkpointing"]["monitoring_interval_epochs"]) != 100
            or profile["checkpointing"]["save_latest"] is not True
            or profile["checkpointing"]["save_best_field"] is not True):
        raise ValueError("Formal D-sep must preserve latest/best state and 100-epoch monitoring checkpoints.")
    horizon = int(profile["schedule"]["horizon_epochs"])
    validate_formal_milestones(profile["checkpointing"], horizon)
    return profile


def _schedule(profile):
    settings = profile["schedule"]
    total = int(settings["horizon_epochs"])
    return {"hold_epochs": round(total * float(settings["hold_fraction"])), "total_epochs": total,
        "initial_lr": float(settings["initial_lr"]), "final_lr": float(settings["final_lr"])}


def _learning_rate(epoch, schedule):
    if int(epoch) <= schedule["hold_epochs"]:
        return schedule["initial_lr"]
    decay = max(schedule["total_epochs"] - schedule["hold_epochs"], 1)
    progress = min(max((int(epoch) - schedule["hold_epochs"]) / decay, 0.), 1.)
    return schedule["final_lr"] + .5 * (schedule["initial_lr"] - schedule["final_lr"]) * (1 + math.cos(math.pi * progress))


def _read_cases(dataset_config, split, ids, stats):
    dataset = GlobalChannelThermalDataset(dataset_config["packed_h5_path"], split=split, points_per_case=None,
        normalize_inputs=bool(dataset_config.get("normalize_inputs", True)),
        normalize_targets=bool(dataset_config.get("normalize_targets", True)),
        random_point_sampling=False, seed=0, normalizer=H5Normalizer(stats), case_ids=ids)
    if dataset.selected_case_ids != list(ids):
        raise ValueError("Formal D-sep loader changed metadata-bound case order.")
    cases = []
    for index in range(len(dataset)):
        sample = dataset[index]
        cases.append({key: sample[key] for key in ("structure", "query_xy", "field_targets", "point_weights", "case_id")})
    return cases


def _sample_batch(cases, indices, epoch, device, training, query_count):
    samples = []
    for index in indices:
        case = cases[int(index)]
        size = len(case["query_xy"])
        query_ids = np.random.default_rng((int(index) * 104729 + int(epoch) * 1000003) if training
            else (1000 + int(index) * 104729)).choice(size, min(int(query_count), size), replace=False)
        samples.append({"structure": case["structure"], "query_xy": case["query_xy"][query_ids],
            "field_targets": case["field_targets"][query_ids, :4], "point_weights": case["point_weights"][query_ids]})
    from torch.utils.data import default_collate
    return recursive_to_device(default_collate(samples), device)


def _losses(prediction, batch):
    square = (prediction - batch["field_targets"]).square()
    weight = batch["point_weights"]
    return (square * weight[..., None]).sum(1) / weight.sum(1).clamp_min(1e-12)[:, None]


@torch.no_grad()
def _validate(model, cases, device, budget):
    rows = []
    microbatch = int(budget["microbatch_cases"])
    for start in range(0, len(cases), microbatch):
        batch = _sample_batch(cases, range(start, min(start + microbatch, len(cases))), 0, device, False,
            int(budget["fluid_queries"]))
        values = _losses(model(batch["structure"], batch["query_xy"]), batch).cpu().numpy()
        rows.extend({"case_id": cases[start + offset]["case_id"], "mse_by_flow_channel": value.tolist()}
            for offset, value in enumerate(values))
    return float(np.mean([row["mse_by_flow_channel"] for row in rows])), rows


def _summarize_flow_rows(rows):
    values = np.asarray([row["mse_by_flow_channel"] for row in rows], dtype=np.float64)
    return {"standardized_four_field_mse": float(values.mean()),
        "standardized_mse_by_channel": values.mean(axis=0).tolist(),
        "case_count": len(rows)}


def _select_rows(rows, case_ids):
    wanted = set(case_ids)
    selected = [row for row in rows if row["case_id"] in wanted]
    if len(selected) != len(wanted):
        raise ValueError("Formal flow validation lost declared source case IDs.")
    return selected


def _formal_validation_panels(rows, primary_ids, compatibility_ids):
    """Split one original90 evaluation into canonical89 primary and compatibility views."""
    compatibility = _select_rows(rows, compatibility_ids)
    primary = _select_rows(compatibility, primary_ids)
    return primary, compatibility


def _plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["epoch"] for row in history], [row["train_flow_mse"] for row in history], label="full-TRAIN flow MSE")
    review = [row for row in history if "validation_flow_mse" in row]
    ax.plot([row["epoch"] for row in review], [row["validation_flow_mse"] for row in review], "o-", label="declared validation panel")
    ax.set(xlabel="Formal D-sep epochs", ylabel="Equal-case/channel normalized MSE", yscale="log")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _checkpoint(model, optimizer, parent, train_config, profile, identity, epoch, best, history, elapsed):
    return {
        "checkpoint_schema_version": 1, "case_id": "ThermalChannel", "model_family": "honf_forward",
        "workflow": "forward", "stage": DEPENDENCY_ID,
        "dependency_identity": DEPENDENCY_ID, "dependency_policy": "D-sep", "case_capability": CASE_CAPABILITY,
        "formal_workflow_scope": FORMAL_TRAIN_SCOPE,
        "flow_reader_config": model.reader.config, "flow_state_dict": model.state_dict(),
        "fit_identity": identity, "formal_profile": profile,
        "formal_dataset_binding": identity["formal_dataset_binding"],
        "formal_normalization_binding": identity["formal_normalization_binding"],
        "formal_validation_binding": identity["formal_validation_binding"],
        "startup_benchmark": identity["startup_benchmark"],
        "epoch": epoch, "current_epoch": epoch, "best_metric": best,
        "optimizer_state_dict": optimizer.state_dict(), "history": history,
        "aggregate_process_seconds": elapsed,
        "global_normalization_stats": parent["global_normalization_stats"],
        "global_normalization_config": identity["formal_normalization_binding"],
        "train_config": train_config, "model_config": {"flow_reader_config": model.reader.config},
        "channel_order": ["u", "v", "p", "omega", "temperature"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, help="Read-only source for native dataset path and DEV22 panel; never an initializer.")
    parser.add_argument("--profile-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--startup-benchmark", action="store_true", help="Disposable full-TRAIN startup; caps the new flow age at three epochs and uses DEV22 only.")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--run-identity")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    start, started = perf_counter(), time()
    torch.set_num_threads(1)
    profile_path = Path(args.profile_file).resolve()
    profile = _validate_profile(json.loads(profile_path.read_text()))
    output = initialize_formal_output(validate_generated_output(args.output), prepare_only=args.prepare_only)
    layout = RunLayout(output)
    parent_path = Path(args.parent).resolve()
    parent = load_trusted_checkpoint(parent_path, map_location="cpu")
    parent_dataset = parent["train_config"]["dataset"]
    dataset_path = parent_dataset["packed_h5_path"]
    data_binding, train_ids = bind_original_train(dataset_path,
        expected_count=profile["data"]["expected_train_case_count"], dataset_id=parent_dataset.get("dataset_id"))
    formal_validation_binding, primary_validation_ids, compatibility_validation_ids = bind_formal_validation(
        dataset_path, expected_primary_count=profile["data"]["formal_validation"]["expected_primary_case_count"],
        expected_compatibility_count=profile["data"]["formal_validation"]["expected_compatibility_case_count"],
        duplicate_case_id=profile["data"]["formal_validation"]["excluded_training_duplicate_case_id"])
    normalizer, normalizer_binding = fit_formal_normalizer(dataset_path, data_binding)
    stats = normalizer.stats
    if not stats or normalizer_binding["training_case_ids_sha256"] != data_binding["training_case_ids_sha256"]:
        raise ValueError("Formal D-sep normalizer is not bound to all original TRAIN cases.")
    val_manifest = parent_dataset.get("development_subset")
    startup_panel = profile["data"]["startup_validation"]
    if (not isinstance(val_manifest, dict) or val_manifest.get("manifest_sha256") != startup_panel["manifest_sha256"]
            or len(val_manifest.get("partitions", {}).get("test", {}).get("case_ids", [])) != int(startup_panel["expected_case_count"])):
        raise ValueError("Formal D-sep startup requires the declared fixed25_v1 DEV22 manifest.")
    startup_validation_ids = val_manifest["partitions"]["test"]["case_ids"]
    if (len(startup_validation_ids) != int(startup_panel["expected_case_count"])
            or len(set(startup_validation_ids)) != len(startup_validation_ids)):
        raise ValueError("Formal D-sep startup panel has an invalid DEV22 membership.")
    train_config = formal_train_config(parent["train_config"], data_binding, normalizer_binding,
        formal_validation_binding)
    schedule = _schedule(profile)
    model = ThermalFlowReader("D-sep").cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule["initial_lr"],
        weight_decay=float(profile["optimizer"]["weight_decay"]))
    identity = {
        "identity": "thermal_d_sep_formal5000_v1", "workflow_scope": FORMAL_TRAIN_SCOPE,
        "run_identity": args.run_identity or "formal5000", "profile": profile,
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "profile_path": str(profile_path), "seed": int(profile["seed"]),
        "dependency_identity": DEPENDENCY_ID, "dependency_policy": "D-sep", "case_capability": CASE_CAPABILITY,
        "flow_reader_config": model.reader.config, "schedule": schedule, "budget": dict(profile["budget"]),
        "optimizer": dict(profile["optimizer"]),
        "formal_dataset_binding": data_binding, "formal_normalization_binding": normalizer_binding,
        "formal_validation_binding": formal_validation_binding,
        "normalization_stats_sha256": stats_sha256(stats),
        "validation_manifest_sha256": val_manifest["manifest_sha256"],
        "startup_validation_case_ids": list(startup_validation_ids),
        "formal_primary_validation_case_ids": list(primary_validation_ids),
        "formal_compatibility_validation_case_ids": list(compatibility_validation_ids),
        "parent_reference_checkpoint": str(parent_path), "parent_reference_sha256": _file_sha256(parent_path),
        "new_weight_initialization": "fresh ThermalFlowReader seed only; parent weights and normalizer not inherited",
        "thermal_optimizer_parameters": 0,
        "startup_benchmark": bool(args.startup_benchmark),
    }

    if args.startup_benchmark:
        if not str(identity["run_identity"]).startswith("startup_flow_"):
            raise ValueError("Disposable flow startup identities must start with startup_flow_.")
    elif str(identity["run_identity"]).startswith("startup_"):
        raise ValueError("A formal flow identity cannot use a disposable startup run name.")
    if args.prepare_only:
        layout.ensure()
        set_seed(int(profile["seed"]))
        model = ThermalFlowReader("D-sep").cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=schedule["initial_lr"],
            weight_decay=float(profile["optimizer"]["weight_decay"]))
        atomic_write_json(layout.write_path("formal_profile_binding.json"), {"profile": profile,
            "formal_dataset_binding": data_binding, "formal_normalization_binding": normalizer_binding,
            "normalization_stats_sha256": stats_sha256(stats), "training_case_count": len(train_ids),
            "startup_validation_case_ids": list(startup_validation_ids),
            "formal_primary_validation_case_ids": list(primary_validation_ids),
            "formal_compatibility_validation_case_ids": list(compatibility_validation_ids),
            "run_identity": identity["run_identity"], "startup_benchmark": bool(args.startup_benchmark)})
        atomic_write_json(layout.write_path("fit_identity.json"), identity)
        payload = _checkpoint(model, optimizer, {"global_normalization_stats": stats}, train_config,
            profile, identity, 0, float("inf"), [], 0.)
        atomic_save_checkpoint_payload(layout.write_path("latest_model.pt"), payload)
        print(json.dumps({"status": "prepared", "run_identity": identity["run_identity"],
            "training_cases": len(train_ids), "startup_validation_cases": len(startup_validation_ids),
            "formal_primary_validation_cases": len(primary_validation_ids),
            "formal_compatibility_validation_cases": len(compatibility_validation_ids),
            "output": str(output)}, sort_keys=True),
            flush=True)
        return 0

    if args.resume is None:
        raise ValueError("Formal D-sep training requires explicit --resume from its prepared/latest checkpoint.")
    saved = load_trusted_checkpoint(args.resume, map_location="cpu")
    if saved.get("fit_identity") != identity:
        raise ValueError("Strict formal D-sep resume rejects a changed profile, data, normalization, or schedule.")
    ensure_formal_resume_identity(output, "fit_identity.json", identity)
    layout.ensure()
    begin = int(saved["epoch"])
    stop_after = int(args.stop_after if args.stop_after is not None
        else (3 if args.startup_benchmark else schedule["total_epochs"]))
    if args.startup_benchmark and stop_after > int(startup_panel["maximum_new_epochs"]):
        raise ValueError("Disposable flow startup may not exceed three absolute epochs.")
    if not begin <= stop_after <= schedule["total_epochs"]:
        raise ValueError("Formal D-sep stop must be at or after the checkpoint age and remain within its horizon.")
    data_start = perf_counter()
    train_cases = _read_cases(parent_dataset, "train", train_ids, stats)
    if args.startup_benchmark:
        validation_scope = startup_panel["scope"]
        validation_ids = startup_validation_ids
        validation = _read_cases(parent_dataset, "test", validation_ids, stats)
        primary_ids = None
    else:
        validation_scope = profile["data"]["formal_validation"]["primary_scope"]
        validation_ids = compatibility_validation_ids
        compatibility_ids = compatibility_validation_ids
        primary_ids = primary_validation_ids
        validation = _read_cases(parent_dataset, "test", validation_ids, stats)
    if len(train_cases) != len(train_ids) or len(validation) != len(validation_ids):
        raise ValueError("Formal D-sep reader changed declared full-TRAIN or validation membership.")
    data_seconds = perf_counter() - data_start
    device = torch.device(args.device)
    load_start = perf_counter()
    model = ThermalFlowReader("D-sep", saved["flow_reader_config"]).to(device)
    model.load_state_dict(saved["flow_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule["initial_lr"],
        weight_decay=float(profile["optimizer"]["weight_decay"]))
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_seconds = perf_counter() - load_start
    history = list(saved["history"])
    best = float(saved["best_metric"])
    receipt = {"pid": os.getpid(), "started_unix": started, "device": str(device),
        "visible_physical_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"), "status": "running",
        "begin_epoch": begin, "load_seconds": load_seconds, "dataset_load_seconds": data_seconds,
        "validation_scope": validation_scope, "startup_benchmark": bool(args.startup_benchmark),
        "case_visits": 0, "optimizer_updates": 0, "training_seconds": 0., "validation_seconds": 0.,
        "save_seconds": 0., "peak_allocated_memory_bytes": 0, "completed_epoch": begin}
    atomic_write_json(layout.write_path("active_process.json"), receipt)
    budget = profile["budget"]
    requested_stop = False
    for epoch in range(begin + 1, stop_after + 1):
        epoch_start = perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        lr = _learning_rate(epoch, schedule)
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = np.random.default_rng(epoch).permutation(len(train_cases))
        total_loss, query_visits = 0., 0
        gradient_norms = []
        updates = case_visits = 0
        model.train()
        for effective_start in range(0, len(order), int(budget["effective_cases"])):
            effective = order[effective_start:effective_start + int(budget["effective_cases"])]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(effective), int(budget["microbatch_cases"])):
                indices = effective[micro_start:micro_start + int(budget["microbatch_cases"])]
                batch = _sample_batch(train_cases, indices, epoch, device, True, int(budget["fluid_queries"]))
                losses = _losses(model(batch["structure"], batch["query_xy"]), batch).mean(-1)
                loss = losses.sum() / len(effective)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite formal D-sep reconstruction loss.")
                loss.backward()
                total_loss += float(losses.sum().detach())
                query_visits += int(batch["query_xy"].shape[0] * batch["query_xy"].shape[1])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(profile["optimizer"]["gradient_clip"]))
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError("Invalid actual formal D-sep parameter gradient.")
            gradient_norms.append(float(norm))
            optimizer.step()
            updates += 1
            case_visits += len(effective)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory = 0
        train_seconds = perf_counter() - epoch_start
        row = {"epoch": epoch, "train_flow_mse": total_loss / case_visits,
            "train_seconds": train_seconds, "case_visits": case_visits, "optimizer_updates": updates,
            "fluid_queries": query_visits, "gradient_norm_mean": float(np.mean(gradient_norms)),
            "learning_rate": lr, "peak_allocated_memory_bytes": peak_memory}
        requested_stop = stop_requested(output)
        review = (epoch % int(profile["checkpointing"]["monitoring_interval_epochs"]) == 0
            or epoch == stop_after or requested_stop)
        validation_seconds = 0.
        save_seconds = 0.
        if review:
            model.eval()
            validation_start = perf_counter()
            score_all, rows_all = _validate(model, validation, device, budget)
            if args.startup_benchmark:
                score, rows = score_all, rows_all
                compatibility = None
                compatibility_rows = None
            else:
                rows, compatibility_rows = _formal_validation_panels(rows_all, primary_ids, compatibility_ids)
                compatibility = {"scope": profile["data"]["formal_validation"]["compatibility_scope"],
                    **_summarize_flow_rows(compatibility_rows)}
                score = _summarize_flow_rows(rows)["standardized_four_field_mse"]
            validation_seconds = perf_counter() - validation_start
            row.update(validation_flow_mse=score, validation_seconds=validation_seconds,
                validation_scope=validation_scope)
            save_start = perf_counter()
            atomic_write_json(layout.write_path(f"validation_epoch_{epoch:04d}.json"),
                {"scope": validation_scope, "score": score, "summary": _summarize_flow_rows(rows),
                 "rows": rows, "compatibility": compatibility,
                 "compatibility_rows": compatibility_rows})
            improved = score < best and epoch % int(profile["checkpointing"]["monitoring_interval_epochs"]) == 0
            if epoch % int(profile["checkpointing"]["monitoring_interval_epochs"]) == 0:
                best = min(best, score)
            history.append(row)
            payload = _checkpoint(model, optimizer, {"global_normalization_stats": stats}, train_config,
                profile, identity, epoch, best, history, saved.get("aggregate_process_seconds", 0.) + perf_counter() - start)
            atomic_save_checkpoint_payload(layout.write_path("latest_model.pt"), payload)
            milestones = {int(value) for value in profile["checkpointing"]["milestone_epochs"]}
            if epoch in milestones:
                atomic_save_checkpoint_payload(layout.write_path(f"epoch_{epoch:04d}_model.pt"), payload)
            if improved and profile["checkpointing"]["save_best_field"]:
                atomic_save_checkpoint_payload(layout.write_path("best_by_field_mse_model.pt"), payload)
            atomic_write_json(layout.write_path("history.json"), history)
            _plot_history(history, layout.write_path("flow_learning.pdf"))
            save_seconds = perf_counter() - save_start
            row["save_seconds"] = save_seconds
            history[-1]["save_seconds"] = save_seconds
            atomic_write_json(layout.write_path("history.json"), history)
        else:
            history.append(row)
            if epoch == 10:
                _plot_history(history, layout.write_path("flow_learning.pdf"))
        receipt["case_visits"] += case_visits
        receipt["optimizer_updates"] += updates
        receipt["training_seconds"] += train_seconds
        receipt["validation_seconds"] += validation_seconds
        receipt["save_seconds"] += save_seconds
        receipt["completed_epoch"] = epoch
        receipt["peak_allocated_memory_bytes"] = max(receipt["peak_allocated_memory_bytes"], peak_memory)
        atomic_write_json(layout.write_path("history.json"), history)
        atomic_write_json(layout.write_path("active_process.json"), receipt)
        print(json.dumps(row, sort_keys=True), flush=True)
        if requested_stop:
            acknowledge_stop(output, epoch=epoch)
            break
    no_op_stop_request = stop_after == begin and stop_requested(output)
    if no_op_stop_request:
        acknowledge_stop(output, epoch=begin)
    final_status = ("stopped_resumable" if requested_stop else
        "already_completed" if stop_after == begin else "completed")
    receipt["no_op_stop_request_acknowledged"] = bool(no_op_stop_request)
    receipt.update(status=final_status, ended_unix=time(),
        process_seconds=perf_counter() - start)
    atomic_write_json(layout.write_path("active_process.json"), receipt)
    session_path = layout.read_path("resource_sessions.json")
    sessions = json.loads(session_path.read_text()) if session_path.exists() else []
    sessions.append(receipt)
    atomic_write_json(layout.write_path("resource_sessions.json"), sessions)
    atomic_write_json(layout.write_path("fit_summary.json"), {**receipt, "best_metric": best,
        "new_optimizer_updates": receipt["optimizer_updates"], "thermal_optimizer_parameters": 0,
        "training_case_count": len(train_cases), "validation_case_count": len(validation),
        "validation_scope": validation_scope,
        "schedule": schedule, "normalization_stats_sha256": stats_sha256(stats),
        "formal_dataset_binding_sha256": data_binding["source_metadata_sha256"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
