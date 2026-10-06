#!/usr/bin/env python3
"""Matched complete native-wrapper timing and physical-input VJP cost.

Three frozen checkpoints share one GPU and identical inputs. Timed calls have
no work/graph hooks. A separate call verifies P0/P1/P2 and actual fine work.
There are no prepared-decode timings, optimizer updates, or solver calls.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from time import perf_counter, time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from thermal_campaign_benchmark import (
    allowed_device,
    common_adjoint,
    differentiable_arguments,
    optional_fine_work,
    phase_outputs,
)
from thermal_response_refinement_evaluation import write_json


def alternating_orders(names, repeats=5):
    """Rotate starting positions and reverse every other repetition."""
    names = tuple(names)
    if repeats < 1 or len(names) != len(set(names)):
        raise ValueError("Unique arm names and positive repetitions are required")
    result = []
    for repetition in range(repeats):
        offset = repetition % len(names)
        order = names[offset:] + names[:offset]
        result.append(list(reversed(order)) if repetition % 2 else list(order))
    return result


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_memory(device):
    synchronize(device)
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def memory_result(device, baseline):
    if device.type != "cuda":
        return {}
    return {"cuda_baseline_allocated_bytes": baseline,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - baseline,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def prepare_inputs(saved, device):
    from channelthermal.data.collation import ChannelThermalBatchCollator
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.training.epoch import make_model_inputs
    from thermal_development import evaluation_dataset_kwargs, resolve_evaluation_manifest

    from honf_runtime.compat import recursive_to_device, resolve_demo_path

    settings = saved["train_config"]["dataset"]
    dataset_path = resolve_demo_path(settings["packed_h5_path"])
    manifest = resolve_evaluation_manifest(settings, dataset_path, scope="development")
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in saved["global_normalization_stats"].items()}
    dataset = GlobalChannelThermalDataset(dataset_path, split="train", points_per_case=1024,
        random_point_sampling=False, normalizer=H5Normalizer(stats), normalize_inputs=True,
        normalize_targets=True, include_grid=True, include_structure_targets=False,
        **evaluation_dataset_kwargs(manifest, "train"))
    dataset.set_epoch(1)
    entries = sorted((int(count), str(case), index) for index, (case, count)
                     in enumerate(zip(dataset.selected_case_ids, dataset.selected_module_counts)))
    panels = {"B8Q1024_low": [entry[2] for entry in entries[:8]],
              "B8Q1024_high": [entry[2] for entry in entries[-8:]],
              "B1Q8192_low": [entries[0][2]], "B1Q8192_high": [entries[-1][2]]}
    collate = ChannelThermalBatchCollator()
    inputs, metadata = {}, {}
    for name, indices in panels.items():
        samples = [dataset[index] for index in indices]
        batch = recursive_to_device(collate(samples), device)
        if name.startswith("B1"):
            sample = samples[0]
            full_query = np.stack((sample["x_grid"].ravel(), sample["y_grid"].ravel()), -1)
            if len(full_query) != 8192:
                raise ValueError("Declared B1Q8192 full-field condition requires the exact 8192-point native grid")
            batch["query_xy"] = torch.as_tensor(full_query, device=device, dtype=torch.float32).unsqueeze(0)
        arguments = make_model_inputs(batch, local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
                                      return_predicted_port_outputs=True, return_port_global_consistency=True)
        inputs[name] = arguments
        metadata[name] = {"case_ids": [str(dataset.selected_case_ids[index]) for index in indices],
                          "module_counts": [int(dataset.selected_module_counts[index]) for index in indices],
                          "B": len(indices), "Q": int(arguments["query_xy"].shape[1]),
                          "sampling_epoch": 1, "query_scope": "full native grid" if name.startswith("B1") else "deterministic native Q1024 sample"}
    return inputs, metadata, manifest, stats


def benchmark(checkpoints, output_dir, *, device="cuda:0", repeats=5, warmup=1, vjp_repeats=2):
    from channelthermal.evaluation.loading import load_model
    from thermal_campaign_heat_inference import snapshot_forward_state, verify_frozen_forward
    from thermal_development import validate_generated_output

    device = allowed_device(device)
    if repeats != 5 or vjp_repeats != 2 or warmup < 1:
        raise ValueError("Refinement cost uses exactly five warmed repetitions and two whole-wrapper input VJPs")
    output_dir = validate_generated_output(output_dir)
    if (output_dir / "cost.json").exists() or (output_dir / "receipt.json").exists():
        raise ValueError("Preserve prior attempts: use a new benchmark output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time()
    receipt = {"pid": os.getpid(), "start_unix": start, "status": "running", "device": str(device),
               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "optimizer_updates": 0,
               "solver_calls": 0, "wrapper_calls_attempted": 0, "wrapper_calls_completed": 0,
               "input_vjps_attempted": 0, "input_vjps_completed": 0}
    write_json(output_dir / "receipt.json", receipt)
    models, saved_states, snapshots, rows = {}, {}, {}, []
    try:
        torch.set_num_threads(2)
        for name, path in checkpoints.items():
            model, saved = load_model(Path(path), device)
            model.eval().requires_grad_(False)
            models[name], saved_states[name] = model, saved
            snapshots[name] = snapshot_forward_state(model)
        inputs, panels, manifest, stats = prepare_inputs(saved_states["G-fast"], device)
        for name, saved in saved_states.items():
            for key, value in stats.items():
                if not np.array_equal(np.asarray(saved["global_normalization_stats"][key], dtype=np.float32), value):
                    raise ValueError(f"Cost inputs cannot share unequal saved normalization: {name} {key}")
        orders = alternating_orders(models, repeats)
        adjoints, scales = {}, {}

        def complete_call(name, arguments):
            receipt["wrapper_calls_attempted"] += 1
            output = models[name](**arguments)
            receipt["wrapper_calls_completed"] += 1
            return output

        for condition, arguments in inputs.items():
            condition_rows = {name: {"arm": name, "checkpoint": str(Path(checkpoints[name]).resolve()),
                "checkpoint_epoch": saved_states[name]["epoch"], "condition": condition, **panels[condition],
                "inference_samples": [], "input_vjp_samples": []} for name in models}
            # Warm-up every model under exactly the later native flags.
            for _ in range(warmup):
                for name in models:
                    with torch.no_grad():
                        output = complete_call(name, arguments)
                    synchronize(device)
                    del output
            for repetition, order in enumerate(orders):
                for position, name in enumerate(order):
                    baseline = reset_memory(device)
                    before = perf_counter()
                    with torch.no_grad():
                        output = complete_call(name, arguments)
                    synchronize(device)
                    elapsed = perf_counter() - before
                    condition_rows[name]["inference_samples"].append({"repetition": repetition, "arm_order_position": position,
                        "seconds": elapsed, **memory_result(device, baseline)})
                    del output
            # G-fast defines one detached output adjoint per condition, reused
            # by both refinements and both VJP repetitions.
            for repetition, order in enumerate(alternating_orders(models, vjp_repeats)):
                for position, name in enumerate(order):
                    differentiable, leaves = differentiable_arguments(arguments, saved_states[name])
                    baseline = reset_memory(device)
                    before = perf_counter()
                    output = complete_call(name, differentiable)
                    synchronize(device)
                    forward_seconds = perf_counter() - before
                    if condition not in adjoints:
                        adjoints[condition], scales[condition] = common_adjoint(output)
                    objective = sum((output[key] * value).sum() for key, value in adjoints[condition].items())
                    receipt["input_vjps_attempted"] += 1
                    synchronize(device)
                    before = perf_counter()
                    gradients = torch.autograd.grad(objective, tuple(leaves.values()), allow_unused=True)
                    synchronize(device)
                    vjp_seconds = perf_counter() - before
                    receipt["input_vjps_completed"] += 1
                    if not all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in gradients):
                        raise RuntimeError("Nonfinite complete-wrapper physical-input VJP")
                    condition_rows[name]["input_vjp_samples"].append({"repetition": repetition, "arm_order_position": position,
                        "forward_seconds": forward_seconds, "vjp_seconds": vjp_seconds,
                        "complete_seconds": forward_seconds + vjp_seconds, **memory_result(device, baseline),
                        "gradient_l2": {key: float(gradient.norm()) if gradient is not None else None
                                        for key, gradient in zip(leaves, gradients)}})
                    del output, objective, gradients, differentiable, leaves
            # Separate instrumented calls establish actual full phases/work.
            for name, model in models.items():
                with torch.no_grad(), phase_outputs(model.core) as phases, optional_fine_work(model.core.backend) as work:
                    output = complete_call(name, arguments)
                synchronize(device)
                observed = sorted({key.split(".")[0] for key in phases})
                if observed != ["P0", "P1", "P2"]:
                    raise RuntimeError("Cost harness did not execute actual P0/P1/P2")
                row = condition_rows[name]
                row.update(native_phases=observed, fine_kernel_work=work, common_adjoint_scales=scales[condition],
                    inference_median_seconds=float(np.median([sample["seconds"] for sample in row["inference_samples"]])),
                    input_forward_vjp_median_seconds=float(np.median([sample["complete_seconds"] for sample in row["input_vjp_samples"]])))
                rows.append(row)
                del output, phases
            write_json(output_dir / "cost.json", {"rows": rows, "alternating_orders": orders, "manifest_fingerprint": manifest["manifest_sha256"]})
            gc.collect()
        frozen = {name: verify_frozen_forward(models[name], snapshots[name]) for name in models}
        if not all(value["passed"] for value in frozen.values()):
            raise RuntimeError("Cost measurements changed frozen model values")
        ratios = []
        for condition in inputs:
            baseline = next(row for row in rows if row["arm"] == "G-fast" and row["condition"] == condition)
            for row in [row for row in rows if row["arm"] != "G-fast" and row["condition"] == condition]:
                ratios.append({"arm": row["arm"], "condition": condition,
                    "complete_inference_ratio_to_Gfast": row["inference_median_seconds"] / baseline["inference_median_seconds"],
                    "complete_input_forward_vjp_ratio_to_Gfast": row["input_forward_vjp_median_seconds"] / baseline["input_forward_vjp_median_seconds"]})
        result = {"rows": rows, "ratios": ratios, "alternating_orders": orders, "manifest_fingerprint": manifest["manifest_sha256"],
                  "frozen_state": frozen, "scope": "complete native P0/P1/P2 FP32 wrappers; five alternating warmed timings and two complete heat/query-input VJPs per arm/condition",
                  "timing_excludes": "input staging, checkpoint loading, output export, work/graph hooks; those occur outside timed calls",
                  "memory_scope": "three models and all condition inputs resident together; extra CUDA allocation subtracts that common invocation baseline",
                  "work_scope": "one additional instrumented full forward per condition; backward work unmeasured; dense/global paths retained",
                  "optimizer_updates": 0, "solver_calls": 0}
        write_json(output_dir / "cost.json", result)
        receipt.update(status="completed", exit_code=0, frozen_state=frozen)
        return result
    except BaseException as error:
        receipt.update(status="failed", exit_code=1, error=repr(error))
        raise
    finally:
        receipt["end_unix"] = time()
        receipt["associated_gpu_hours"] = (receipt["end_unix"] - start) / 3600 if device.type == "cuda" else 0.
        write_json(output_dir / "receipt.json", receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g-fast", type=Path, required=True)
    parser.add_argument("--h-add", type=Path, required=True)
    parser.add_argument("--h-joint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    benchmark({"G-fast": args.g_fast, "H-add": args.h_add, "H-joint": args.h_joint}, args.output_dir, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
