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
from channelthermal.training.checkpoints import _file_sha256, atomic_save_checkpoint_payload
from channelthermal.training.stop_request import acknowledge_stop, stop_requested
from thermal_development import validate_generated_output
from torch.utils.data import default_collate

from honf_runtime.compat import load_trusted_checkpoint, recursive_to_device, set_seed
from honf_runtime.run_store import atomic_write_json

FLOW_CHILD_SCHEDULE = {"identity": "d_sep_schedule_child1500_v1", "parent_absolute_epoch": 1000,
    "additional_epochs": 1500, "warmup_epochs": 20, "hold_until_additional_epoch": 500,
    "warmup_initial_lr": 1e-6, "maximum_lr": 1e-4, "final_lr": 1e-6}


def flow_child_learning_rate(additional_epoch):
    """Declared new schedule; the historical zero-floor schedule is untouched."""
    age = int(additional_epoch)
    if age <= 20:
        return 1e-6 + (1e-4 - 1e-6) * max(age - 1, 0) / 19
    if age <= 500:
        return 1e-4
    return 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * min(age - 500, 1000) / 1000))


def read_cases(parent, split):
    config = copy.deepcopy(parent["train_config"]["dataset"])
    manifest = resolve_development_manifest(config, config["packed_h5_path"])
    ids = development_case_ids(manifest, split)
    if len(ids) != (150 if split == "train" else 22):
        raise ValueError("Flow comparison requires the exact fixed25_v1 150/22 selection.")
    normalizer = H5Normalizer(parent["global_normalization_stats"])
    dataset = GlobalChannelThermalDataset(config["packed_h5_path"], split=split, points_per_case=None,
        normalize_inputs=config["normalize_inputs"], normalize_targets=config["normalize_targets"],
        random_point_sampling=False, seed=0, normalizer=normalizer, case_ids=ids)
    cases = []
    for index in range(len(dataset)):
        sample = dataset[index]
        cases.append({key: sample[key] for key in ("structure", "query_xy", "field_targets", "point_weights", "case_id")})
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--policy", required=True, choices=("D-sep", "D-open"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100, choices=(100, 500, 1000, 1500, 2500))
    parser.add_argument("--resume")
    parser.add_argument("--schedule-parent", help="Opt-in new D-sep1500-epoch schedule child from literal1000.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
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
