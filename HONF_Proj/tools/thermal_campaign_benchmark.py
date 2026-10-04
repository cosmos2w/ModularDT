#!/usr/bin/env python3
"""Frozen-checkpoint native wrapper and prepared-decode timings, separately.

Counts are measured in a separate forward pass, never inside latency scopes.
Input staging, checkpoint load, warm-up and organizer export are excluded.
CPU RSS is process-scoped; CUDA peaks are reset for each timed invocation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def allowed_device(value):
    device = torch.device(value)
    if device.type == "cuda":
        logical = device.index or 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical = visible.split(",")[logical] if visible else str(logical)
        if physical.strip() not in {"1", "2"}:
            raise ValueError("Campaign GPU measurements are limited to physical GPU 1 or 2")
    return device


def output_directory(value):
    from thermal_development import validate_generated_output

    output = validate_generated_output(value)
    output.mkdir(parents=True, exist_ok=True)
    return output


def low_high_indices(dataset, *, count=2):
    entries = []
    for index, case_id in enumerate(dataset.selected_case_ids):
        if str(case_id).lstrip("0") == "273":
            continue
        group = dataset.h5["cases"][case_id]
        entries.append((int((np.asarray(group["module_present"]) > .5).sum()), str(case_id), index))
    entries.sort()
    if not entries:
        raise ValueError("No eligible timing cases")
    positions = np.linspace(0, len(entries) - 1, min(count, len(entries))).round().astype(int)
    return [entries[position][2] for position in positions]


def _cpu_rss():
    # Linux current resident memory, not an inferred per-call peak.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def latency_samples(call, device, *, repeats=9, warmup=2):
    if repeats < 1 or warmup < 0:
        raise ValueError("Positive repeats and nonnegative warm-up are required")
    for _ in range(warmup):
        result = call()
        del result
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gc.collect()
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        else:
            baseline = None
        before = _cpu_rss()
        start = perf_counter()
        result = call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = perf_counter() - start
        samples.append({"seconds": seconds, "cpu_rss_before_bytes": before,
            "cpu_rss_after_bytes": _cpu_rss(), "cpu_process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cuda_baseline_allocated_bytes": baseline,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if baseline is not None else None,
            "cuda_peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - baseline if baseline is not None else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if baseline is not None else None})
        del result
    seconds = np.asarray([sample["seconds"] for sample in samples])
    return {"median_seconds": float(np.median(seconds)), "p90_seconds": float(np.quantile(seconds, .9)),
        "min_seconds": float(seconds.min()), "repeats": repeats, "warmup": warmup, "samples": samples,
        "memory_scope": "CUDA invocation peaks include resident model/input/prepared state; extra peak subtracts allocated baseline. CPU RSS is process-scoped, lifetime peak is not a call peak."}


def native_arguments(model, sample, query, device):
    from channelthermal.evaluation.loading import make_batch
    batch = make_batch(sample, np.asarray(query, dtype=np.float32), device)
    return {"structure": batch["structure"], "query_xy": batch["query_xy"],
        "interface_condition": batch.get("interface_condition"), "local_module_params": batch.get("local_module_params"),
        "teacher_port_tokens": batch.get("teacher_port_tokens"), "local_query_points": batch.get("module_internal_query_points"),
        "local_port_condition_mode": "predicted", "mixed_teacher_ratio": 0., "return_routing_maps": False,
        "return_prepared_state": True}


def load_native(checkpoint_path, dataset_path, device, *, freeze=True, dataset_scope="auto"):
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model
    from thermal_development import evaluation_dataset_kwargs, resolve_evaluation_manifest

    from honf_runtime.compat import resolve_demo_path
    model, checkpoint = load_model(Path(checkpoint_path), device)
    model.eval()
    if freeze:
        model.requires_grad_(False)
    settings = checkpoint.get("train_config", {}).get("dataset", {})
    path = resolve_demo_path(dataset_path or settings["packed_h5_path"])
    manifest = resolve_evaluation_manifest(settings, path, scope=dataset_scope)
    selection = evaluation_dataset_kwargs(manifest, "test")
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint.get("global_normalization_stats", {}).items()}
    if (manifest is not None or settings.get("development_manifest") or settings.get("development_subset")) and not stats:
        raise ValueError("Development evaluation requires saved selected-training global normalization stats")
    normalizer = H5Normalizer(stats) if stats else None
    normalized = GlobalChannelThermalDataset(path, split="test", points_per_case=1, random_point_sampling=False,
        include_grid=True, normalizer=normalizer,
        normalize_inputs=bool(settings.get("normalize_inputs", False)), normalize_targets=bool(settings.get("normalize_targets", False)),
        **selection)
    raw = GlobalChannelThermalDataset(path, split="test", points_per_case=1, random_point_sampling=False, include_grid=True,
        normalizer=normalizer, **selection)
    if normalized.selected_case_ids != raw.selected_case_ids:
        raise ValueError("Normalized and physical dataset indices differ")
    return model, checkpoint, normalized, raw, path


@contextmanager
def optional_fine_work(backend):
    """Keep historical backend timing available when route hooks do not apply."""
    from honf_forward_core.evaluation.fine_kernel_work import FineKernelWork
    try:
        recorder = FineKernelWork(backend)
    except (TypeError, ValueError) as error:
        yield {"measured": False, "routes": None, "reason": str(error),
               "scope": "Five-route fine MLP input rows unmeasured for this historical backend"}
        return
    result = {"measured": True}
    with recorder:
        yield result
    result.update(recorder.snapshot())


def query_panels(sample, raw_sample, small_queries):
    from thermal_campaign_heat_inference import sensor_panel
    sensors, _, _, _, _ = sensor_panel(raw_sample)
    if small_queries <= len(sensors):
        small = sensors[:small_queries]
    else:
        grid = np.stack((sample["x_grid"].ravel(), sample["y_grid"].ravel()), -1)
        small = grid[np.linspace(0, len(grid) - 1, min(small_queries, len(grid))).round().astype(int)]
    full = np.stack((sample["x_grid"].ravel(), sample["y_grid"].ravel()), -1)
    return (("small_inverse", small), ("full_native_grid", full))


# Native physical outputs, including the provisional P0 and refreshed P1 ports.
# Persistent state, diagnostics, and work counters are not physical outputs.
PARITY_OUTPUTS = ("pred_field", "pred_interface", "pred_internal_temperature",
    "pred_port_condition_raw", "pred_port_condition", "local_port_condition_used",
    "module_response_latent", "pred_port_global_temperature",
    "pred_port_global_temperature_target", "pred_port_global_consistency_mask")
ADJOINT_OUTPUTS = PARITY_OUTPUTS[:6]
OUTPUT_ROLES = {**{name: "predicted_physical" for name in ADJOINT_OUTPUTS},
    "module_response_latent": "local_response_latent",
    "pred_port_global_temperature": "predicted_physical",
    "pred_port_global_temperature_target": "non_predictive_port_target",
    "pred_port_global_consistency_mask": "non_predictive_port_mask"}


@contextmanager
def phase_outputs(core):
    """Capture actual shared-core P0/P1/P2 preparation and read values."""
    values, counts, originals = {}, {}, {}
    for method in ("prepare", "read"):
        originals[method] = (method in core.__dict__, getattr(core, method))

    def capture(kind, output, prepared):
        state = getattr(prepared, "backend_state", {})
        phase_id = state.get("hypergraph_phase", state.get("pair_phase"))
        phase = f"P{int(phase_id)}" if phase_id is not None else "unlabelled"
        index = counts.get((phase, kind), 0)
        counts[phase, kind] = index + 1
        attributes = ("module_states", "coarse_state") if kind == "prepare" else ("context",)
        for attribute in attributes:
            values[f"{phase}.{kind}.{index}.{attribute}"] = getattr(output, attribute)

    def prepare(*args, **kwargs):
        output = originals["prepare"][1](*args, **kwargs)
        capture("prepare", output, output)
        return output

    def read(*args, **kwargs):
        output = originals["read"][1](*args, **kwargs)
        capture("read", output, args[0] if args else kwargs["prepared"])
        return output

    core.prepare, core.read = prepare, read
    try:
        yield values
    finally:
        for name, (owned, original) in originals.items():
            if owned:
                setattr(core, name, original)
            else:
                delattr(core, name)


@contextmanager
def source_access_masks(backend):
    """Actual applied pair-support and eligibility masks, outside timing."""
    masks, counts, phase = {}, {}, "P0"
    originals = {name: (name in backend.__dict__, getattr(backend, name)) for name in ("prepare", "_access")}

    def prepare(*args, **kwargs):
        nonlocal phase
        phase = str(getattr(kwargs.get("interaction_context"), "phase", "P0"))
        return originals["prepare"][1](*args, **kwargs)

    def access(*args, **kwargs):
        output = originals["_access"][1](*args, **kwargs)
        route = args[2] if len(args) > 2 else kwargs["mechanism"]
        index = counts.get((phase, route), 0)
        counts[phase, route] = index + 1
        masks[f"{phase}.{route}.{index}"] = {
            "support": output.support.detach().cpu().clone(),
            "eligible": output.diagnostics["pair_valid"].detach().cpu().clone()}
        return output

    backend.prepare, backend._access = prepare, access
    try:
        yield masks
    finally:
        for name, (owned, original) in originals.items():
            if owned:
                setattr(backend, name, original)
            else:
                delattr(backend, name)


def differentiable_arguments(arguments, checkpoint):
    """Identical native values; physical heat varies global and local inputs.

    Global heat remains in its checkpoint input units. Its physical derivative
    includes the saved normalizer's standard deviation; local heat is physical.
    No input normalization, local physics, or prediction equation is changed.
    """
    # The same saved-stat width/epsilon contract as
    # DifferentiableThermalOperator._normalize_heat; no second normalizer.
    from channelthermal.response_control.native import _stats_vector
    result = dict(arguments)
    result["structure"] = dict(arguments["structure"])
    base_global = arguments["structure"]["heat_powers"].detach()
    settings = checkpoint.get("train_config", {}).get("dataset", {})
    stats = checkpoint.get("global_normalization_stats", {})
    normalized = bool(settings.get("normalize_inputs", False))
    if normalized:
        means, stds = _stats_vector(stats, "heat_power_mean", "heat_power_std", 1, like=base_global)
        mean, std = means[0], stds[0]
    else:
        mean, std = base_global.new_tensor(0.), base_global.new_tensor(1.)
    local = arguments.get("local_module_params")
    if local is not None:
        base_heat = local[..., 0].detach()
    else:
        base_heat = base_global * std + mean
    heat = base_heat.clone().requires_grad_(True)
    delta = heat - base_heat
    result["structure"]["heat_powers"] = base_global + delta / std
    if local is not None:
        local_heat = local[..., :1] + delta.unsqueeze(-1) * arguments["structure"]["module_present"].unsqueeze(-1)
        result["local_module_params"] = torch.cat((local_heat, local[..., 1:]), -1)
    query = arguments["query_xy"].detach().clone().requires_grad_(True)
    result["query_xy"] = query
    return result, {"input.physical_heat": heat, "input.query_xy": query}


def common_adjoint(outputs):
    """One deterministic nonuniform adjoint, scaled per native output channel.

    Each role contributes mean(y * a / s), a=linspace(.5,1.5) over
    flattened cells, s=max(reference channel RMS,1). The same detached a/s
    is reused for both executors. This is an AD probe, not a training loss.
    """
    weights, scales = {}, {}
    for name in ADJOINT_OUTPUTS:
        value = outputs[name]
        if not value.numel():
            continue
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite reference adjoint output: {name}")
        # Native material temperature is [B,M,P], with P sampling positions,
        # not output channels. It has one scalar temperature channel.
        channels = 1 if name == "pred_internal_temperature" else value.shape[-1]
        channel_scale = value.detach().square().reshape(-1, channels).mean(0).sqrt().clamp_min(1.)
        weights[name] = torch.linspace(.5, 1.5, value.numel(), device=value.device,
            dtype=value.dtype).reshape_as(value) / channel_scale / value.numel()
        scales[name] = channel_scale.detach().cpu().tolist()
    return weights, scales


def tensor_error(reference, candidate, *, rtol, atol):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {"passed": False, "reason": "shape or dtype differs"}
    ref, actual = reference.detach().double().cpu(), candidate.detach().double().cpu()
    difference = actual - ref
    norm = float(ref.norm())
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(actual).all())
    if not finite:
        return {"passed": False, "shape": list(ref.shape), "finite": False,
            "reason": "nonfinite reference or candidate values"}
    return {"passed": finite and bool(torch.allclose(actual, ref, rtol=rtol, atol=atol)),
        "shape": list(ref.shape), "finite": finite, "reference_l2": norm,
        "candidate_l2": float(actual.norm()), "error_l2": float(difference.norm()),
        "relative_l2": float(difference.norm()) / max(norm, 1e-30),
        "max_abs_error": float(difference.abs().max()) if difference.numel() else 0.}


def verify_executor_parity(model, arguments, checkpoint, native_flags, *, receiver_chunk=128,
                           output_rtol=2e-5, output_atol=2e-5,
                           gradient_rtol=2e-5, gradient_atol=1e-6):
    """Compare whole native wrappers and first derivatives without optimization.

    The caller supplies a frozen eval model and the flags from native load_model.
    Parameter .grad is never populated. Work scopes cover forward only, not
    backward or activation-checkpoint recomputation. All flags, grads, executor,
    CPU/CUDA RNG are restored; parameters/buffers are checked unchanged.
    A detected state mutation aborts the probe before timing.
    """
    from channelthermal.training.campaign_work import CampaignForwardWork
    from thermal_campaign_heat_inference import snapshot_forward_state, verify_frozen_forward
    backend = model.core.backend
    if not hasattr(backend, "set_execution_mode"):
        raise ValueError("Executor parity requires a shared typed checkpoint")
    parameters = dict(model.named_parameters())
    if parameters.keys() != native_flags.keys():
        raise ValueError("Native parameter flag inventory differs from loaded model")
    snapshot = snapshot_forward_state(model)
    verify_frozen_forward(model, snapshot)
    original_flags = {name: parameter.requires_grad for name, parameter in parameters.items()}
    original_grads = {name: parameter.grad for name, parameter in parameters.items()}
    original_mode, original_chunk = backend.execution_mode, backend.execution_receiver_chunk
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    executions, weights, scales = {}, None, None
    try:
        for name, parameter in parameters.items():
            parameter.requires_grad_(native_flags[name])
        for executor in ("dense_masked_reference", "rectangular_subset"):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            backend.set_execution_mode(executor, receiver_chunk_size=receiver_chunk)
            inputs, leaves = differentiable_arguments(arguments, checkpoint)
            with torch.enable_grad(), phase_outputs(model.core) as phases, source_access_masks(backend) as masks, CampaignForwardWork(model.core) as work, optional_fine_work(backend) as fine:
                output = model(**inputs)
            physical = {name: output[name] for name in PARITY_OUTPUTS}
            if {name.split(".")[0] for name in phases} != {"P0", "P1", "P2"}:
                raise RuntimeError("Parity requires measured complete P0/P1/P2 wrapper calls")
            if weights is None:
                weights, scales = common_adjoint(physical)
            with torch.enable_grad():
                loss = sum((physical[name] * weight).sum() for name, weight in weights.items())
                targets = {name: parameter for name, parameter in parameters.items() if native_flags[name]}
                targets.update(leaves)
                gradients = torch.autograd.grad(loss, tuple(targets.values()), allow_unused=True)
            executions[executor] = {
                "values": {name: value.detach().cpu() for name, value in {**physical, **phases}.items()},
                "gradients": {name: None if value is None else value.detach().cpu() for name, value in zip(targets, gradients)},
                "source_masks": masks,
                "adjoint_value": float(loss.detach()) if torch.isfinite(loss.detach()) else None,
                "forward_work": work.records, "fine_kernel_work": fine}
            del output, physical, phases, loss, gradients, targets, leaves, inputs
    finally:
        for name, parameter in parameters.items():
            parameter.requires_grad_(original_flags[name])
            parameter.grad = original_grads[name]
        backend.set_execution_mode(original_mode, receiver_chunk_size=original_chunk)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        frozen = verify_frozen_forward(model, snapshot)
    reference, candidate = executions.values()
    if reference["values"].keys() != candidate["values"].keys():
        raise RuntimeError("Executor phase/output inventory differs")
    output_checks = {name: tensor_error(value, candidate["values"][name], rtol=output_rtol, atol=output_atol)
        for name, value in reference["values"].items()}
    for name in PARITY_OUTPUTS:
        difference = (candidate["values"][name] - reference["values"][name]).double().abs()
        channels = 1 if name == "pred_internal_temperature" else difference.shape[-1]
        if difference.numel() and torch.isfinite(difference).all():
            output_checks[name]["max_abs_error_by_channel"] = difference.reshape(-1, channels).amax(0).tolist()
    if reference["source_masks"].keys() != candidate["source_masks"].keys():
        raise RuntimeError("Executor applied source-access inventory differs")
    support_checks = {}
    for name, source in reference["source_masks"].items():
        actual = candidate["source_masks"][name]
        support_checks[name] = {"passed": all(torch.equal(value, actual[key]) for key, value in source.items()),
            "shape": list(source["support"].shape), "reference_support_pairs": int(source["support"].sum()),
            "candidate_support_pairs": int(actual["support"].sum()),
            "reference_eligible_pairs": int(source["eligible"].sum()), "candidate_eligible_pairs": int(actual["eligible"].sum())}
    gradient_checks, unused = {}, {}
    for name, value in reference["gradients"].items():
        actual = candidate["gradients"][name]
        if value is None or actual is None:
            gradient_checks[name] = {"passed": value is None and actual is None, "reference_unused": value is None,
                                     "candidate_unused": actual is None}
        else:
            gradient_checks[name] = tensor_error(value, actual, rtol=gradient_rtol, atol=gradient_atol)
    for executor, execution in executions.items():
        unused[executor] = [name for name, value in execution["gradients"].items() if value is None]
    active = [name for name in reference["gradients"] if name not in unused["dense_masked_reference"]]
    squared_error = sum(check.get("error_l2", 0.) ** 2 for check in gradient_checks.values())
    squared_norm = sum(check.get("reference_l2", 0.) ** 2 for check in gradient_checks.values())
    finite_gradients = all(check.get("finite", True) for check in gradient_checks.values())
    return {"passed": all(check["passed"] for check in (*output_checks.values(), *gradient_checks.values(), *support_checks.values())),
        "tolerance": {"output": {"rtol": output_rtol, "atol": output_atol},
            "first_gradient": {"rtol": gradient_rtol, "atol": gradient_atol},
            "criterion": "finite elementwise allclose for every output/phase tensor and active first gradient; bit-identical applied support/eligibility masks and identical unused sets"},
        "adjoint": {"formula": "sum_role mean(y * linspace(.5,1.5) / max(reference_channel_RMS,1)); same detached reference adjoint for both executors",
            "roles": list(weights), "reference_channel_scales": scales,
            "values": {key: value["adjoint_value"] for key, value in executions.items()}, "training_objective": False},
        "output_roles": OUTPUT_ROLES, "output_checks": output_checks, "first_gradient_checks": gradient_checks,
        "applied_support_inventory_checks": support_checks,
        "native_trainable_parameter_tensors": sum(native_flags.values()),
        "native_frozen_parameter_names": [name for name, flag in native_flags.items() if not flag],
        "active_parameter_gradient_tensors": sum(not name.startswith("input.") for name in active),
        "active_parameter_gradient_tensors_by_executor": {executor: sum(value is not None and not name.startswith("input.")
            for name, value in execution["gradients"].items()) for executor, execution in executions.items()},
        "active_input_gradient_names": [name for name in active if name.startswith("input.")],
        "unused_gradient_names": unused,
        "overall_gradient_relative_l2": squared_error ** .5 / max(squared_norm ** .5, 1e-30) if finite_gradients else None,
        "forward_support_and_work": {key: {name: value[name] for name in ("forward_work", "fine_kernel_work")} for key, value in executions.items()},
        "restored_frozen_state": frozen,
        "limits": "Whole native wrapper is FP32: FP64 model/input casting is unsupported by its explicit FP32 encoding. Retained Tree small-Q interface output failed a stricter 1e-6 absolute probe (max 7.25e-6); the declared native 2e-5 absolute output contract is separate from the stricter first-gradient contract. A parity pass does not prove all algebra or physical fidelity.",
        "scope": "actual complete native P0/P1/P2 wrapper, first parameter/physical-heat/query gradients; no optimizer or backward work count"}


def benchmark_model_case(model, sample, raw_sample, device, *, repeats, warmup, small_queries=14):
    from channelthermal.training.campaign_work import CampaignForwardWork
    rows = []
    with torch.no_grad():
        for panel, query in query_panels(sample, raw_sample, small_queries):
            arguments = native_arguments(model, sample, query, device)
            call = lambda arguments=arguments: model(**arguments)
            wrapper = latency_samples(call, device, repeats=repeats, warmup=warmup)
            # A separate work pass avoids timing recorder synchronization and
            # supplies the reusable prepared state for decode-only repeats.
            with CampaignForwardWork(model.core) as recorded, optional_fine_work(model.core.backend) as fine_work:
                result = call()
            prepared = result["prepared_state"]
            wrapper["forward_work"] = recorded.records
            wrapper["fine_kernel_work"] = fine_work
            decoder = lambda prepared=prepared, query=arguments["query_xy"]: model.decode_prepared(prepared, query, return_routing_maps=False)
            decoded = latency_samples(decoder, device, repeats=repeats, warmup=warmup)
            with CampaignForwardWork(model.core) as recorded_decode, optional_fine_work(model.core.backend) as fine_decode:
                decoder()
            decoded["forward_work"] = recorded_decode.records
            decoded["fine_kernel_work"] = fine_decode
            rows.extend(({"panel": panel, "queries": len(query), "scope": "complete_wrapper_P0_P1_P2", **wrapper},
                         {"panel": panel, "queries": len(query), "scope": "prepared_P2_decode_only", **decoded}))
            del prepared, result
    return rows


def evaluate(args):
    device = allowed_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    model, checkpoint, normalized, raw, path = load_native(args.checkpoint, args.dataset, device,
        freeze=False, dataset_scope=getattr(args, "dataset_scope", "auto"))
    native_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    model.requires_grad_(False)
    output = output_directory(args.output_dir)
    rows = []
    parity = None
    if args.verify_executor_parity:
        parity = {"passed": False, "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"), "device": str(device),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "rows": []}
        try:
            # Always include both native M extrema, regardless of timing cases.
            for index in low_high_indices(raw, count=2):
                sample, reference = normalized[index], raw[index]
                for panel, query in query_panels(sample, reference, args.small_queries):
                    checked = verify_executor_parity(model, native_arguments(model, sample, query, device),
                        checkpoint, native_flags, receiver_chunk=args.executor_receiver_chunk,
                        output_rtol=args.parity_output_rtol, output_atol=args.parity_output_atol,
                        gradient_rtol=args.parity_gradient_rtol, gradient_atol=args.parity_gradient_atol)
                    checked.update(case_id=str(reference["case_id"]),
                        module_count=int((reference["structure"]["module_present"] > .5).sum()),
                        panel=panel, queries=len(query))
                    parity["rows"].append(checked)
                    if not checked["passed"]:
                        raise RuntimeError("Executor output/first-gradient parity failed; timing was not started")
            parity["passed"] = True
        except Exception as error:
            parity["error"] = str(error)
            raise
        finally:
            (output / "executor_parity.json").write_text(json.dumps(parity, indent=2, allow_nan=False) + "\n")
    original = getattr(model.core.backend, "execution_mode", None)
    original_chunk = getattr(model.core.backend, "execution_receiver_chunk", args.executor_receiver_chunk)
    for executor in args.executors:
        if executor == "checkpoint" and original is not None:
            model.core.backend.set_execution_mode(original, receiver_chunk_size=original_chunk)
        elif executor != "checkpoint":
            setter = getattr(model.core.backend, "set_execution_mode", None)
            if setter is None:
                raise ValueError("Executor override requires a shared typed checkpoint")
            setter(executor, receiver_chunk_size=args.executor_receiver_chunk)
        for index in low_high_indices(raw, count=args.cases):
            sample, reference = normalized[index], raw[index]
            measured = benchmark_model_case(model, sample, reference, device, repeats=args.repeats,
                warmup=args.warmup, small_queries=args.small_queries)
            for row in measured:
                row.update(case_id=str(reference["case_id"]), module_count=int((reference["structure"]["module_present"] > .5).sum()),
                    executor=getattr(model.core.backend, "execution_mode", "native_checkpoint_backend"),
                    core_receiver_chunk=model.core.receiver_chunk_size,
                    executor_receiver_chunk=getattr(model.core.backend, "execution_receiver_chunk", None))
                rows.append(row)
            payload = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
                "architecture": model.config.core_honf.forward_architecture, "dataset": str(path), "partition": "exposed development",
                "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_cpu_threads": torch.get_num_threads(), "torch_version": torch.__version__,
                "scope_limit": "warm frozen model invocation only; staging/load/export and work recording excluded. Complete-wrapper and prepared decoding are independent timings and must not be added.",
                "executor_parity": parity, "rows": rows}
            (output / "timing.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            print(f"{reference['case_id']} {executor}: saved four separate timing scopes", flush=True)
    if original is not None:
        model.core.backend.set_execution_mode(original, receiver_chunk_size=original_chunk)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--evaluation-scope", dest="dataset_scope", choices=("auto", "development", "formal-full"), default="auto",
                        help="Use checkpoint-bound development cases by default; formal-full is explicit.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--small-queries", type=int, default=14)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--executor-receiver-chunk", type=int, default=128)
    parser.add_argument("--verify-executor-parity", action="store_true",
        help="Require complete low/high-M small/full-Q native output and first-gradient parity before any timing")
    parser.add_argument("--parity-output-rtol", type=float, default=2e-5)
    parser.add_argument("--parity-output-atol", type=float, default=2e-5)
    parser.add_argument("--parity-gradient-rtol", type=float, default=2e-5)
    parser.add_argument("--parity-gradient-atol", type=float, default=1e-6)
    parser.add_argument("--executors", nargs="+", default=["checkpoint"],
        choices=("checkpoint", "dense_masked_reference", "rectangular_subset"))
    args = parser.parse_args(argv)
    if min(args.cases, args.repeats, args.small_queries, args.cpu_threads, args.executor_receiver_chunk) <= 0 or args.warmup < 0:
        parser.error("positive timing sizes and nonnegative warm-up are required")
    tolerances = (args.parity_output_rtol, args.parity_output_atol, args.parity_gradient_rtol, args.parity_gradient_atol)
    if not np.isfinite(tolerances).all() or min(tolerances) < 0:
        parser.error("finite nonnegative parity tolerances are required")
    return args


if __name__ == "__main__":
    print(evaluate(parse_args()))
