#!/usr/bin/env python3
"""Cold saved-geometry responses and actual-input native derivative checks.

No training, old thermal wrapper, generator, solver, design search, or truth
coefficient cache is used. Geometry endpoints are an explicitly exposed saved
original-TRAIN atlas, not independent test evidence. Group membership and
compression work belong to the separate executor-cost evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "Case_ThermalChannel/src", ROOT / "tools"):
    sys.path.insert(0, str(source))

import numpy as np
import torch
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.source_response import load_source_response_model
from channelthermal.training.checkpoints import _file_sha256
from thermal_development import validate_generated_output
from thermal_response_refinement_evaluation import finite_metrics
from thermal_source_response_evaluate import model_digest, synchronize

from honf_runtime.run_store import atomic_write_json

FAMILIES = ("0001", "0348", "0304", "0350")
GEOMETRY_STATES = ("baseline", "i_minus", "i_plus", "j_minus", "j_plus")


def saved_geometry(model, atlas_root, output, device):
    rows = []
    for case_id in FAMILIES:
        path = atlas_root / f"train_{case_id}_responses.npz"
        stencil, _ = load_response_atlas_stencil(path)
        if stencil.baseline.design.anchor_id != case_id or stencil.split.value != "train":
            raise ValueError("Geometry evaluation requires the exact saved original-TRAIN family.")
        records = {"baseline": stencil.baseline, **{label: stencil.variants[label] for label in GEOMETRY_STATES[1:]}}
        baseline_centers = [module.position_xy for module in stencil.baseline.design.modules]
        baseline_heat = [module.heating for module in stencil.baseline.design.modules]
        predictions, arrays, times, kernel_times = {}, {}, {}, {}
        for label, record in records.items():
            if [module.heating for module in record.design.modules] != baseline_heat:
                raise ValueError("Saved geometry endpoints must keep physical heating fixed.")
            if label != "baseline" and [module.position_xy for module in record.design.modules] == baseline_centers:
                raise ValueError("The selected geometry endpoint does not change source geometry.")
            synchronize(device)
            started = perf_counter()
            with torch.no_grad():
                prepared = model.prepare_record(record, device=device)
                predictions[label] = {name: value.cpu().numpy() for name, value in model.apply_record(prepared).items()}
            synchronize(device)
            times[label] = perf_counter() - started
            if case_id in ("0001", "0348") and label in ("baseline", "i_plus"):
                changed_centers = np.asarray([module.position_xy for module in records["i_plus"].design.modules])
                moved = np.flatnonzero(np.any(changed_centers != np.asarray(baseline_centers), axis=1))
                if len(moved) != 1:
                    raise ValueError("The donor kernel illustration requires one explicitly moved source.")
                synchronize(device)
                started = perf_counter()
                with torch.no_grad():
                    kernel = model.thermal.export_native_kernels(prepared["thermal"])["fluid"]
                    arrays[f"{label}/moved_donor_kernel"] = kernel[0, :, int(moved[0]), 0].cpu().numpy()
                synchronize(device)
                kernel_times[label] = perf_counter() - started
                arrays["moved_donor_source_id"] = np.asarray(prepared["record_source_ids"][int(moved[0])])
            arrays[f"{label}/centers"] = np.asarray([module.position_xy for module in record.design.modules])
            for name, role in record.output.roles.items():
                arrays[f"{label}/{name}/prediction"] = predictions[label][name]
                arrays[f"{label}/{name}/reference"] = role.values
                arrays[f"{label}/{name}/queries"] = role.query_features
        metrics = []
        for label in GEOMETRY_STATES[1:]:
            row = {"state": label, "roles": {}}
            for name, role in records[label].output.roles.items():
                common = stencil.common_mask(name)
                matched_role = replace(role, valid_mask=common)
                row["roles"][name], pred, truth = finite_metrics(
                    predictions[label][name],
                    role.values,
                    matched_role,
                    baseline_prediction=predictions["baseline"][name],
                    baseline_reference=stencil.baseline.output.roles[name].values,
                )
                arrays[f"{label}/{name}/prediction_delta"] = pred
                arrays[f"{label}/{name}/reference_delta"] = truth
                arrays[f"common/{name}/valid"] = common
                arrays[f"common/{name}/quadrature"] = role.quadrature_weights
            metrics.append(row)
        np.savez_compressed(output / f"geometry_{case_id}.npz", **arrays)
        rows.append(
            {
                "case_id": case_id,
                "cohort": "response_fit_exposed" if case_id in ("0001", "0348") else "geometry_development_exposed",
                "reference": str(path),
                "reference_sha256": _file_sha256(path),
                "finite_changes": metrics,
                "cold_complete_calls": len(records),
                "cold_complete_seconds": times,
                "explicit_dense_kernel_exports": len(kernel_times),
                "explicit_dense_kernel_export_seconds": kernel_times,
                "support": "Intersection of valid native receivers over all eleven existing atlas states",
            }
        )
        print(
            json.dumps({"geometry_case": case_id, "complete_calls": len(records), "seconds": sum(times.values())}),
            flush=True,
        )
    return rows


def diagnostic_inputs(record, device):
    """Read only prescribed context, source geometry/heating and query metadata."""
    modules, context = record.design.modules, record.context.values
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    structure = {
        "module_centers": tensor([module.position_xy for module in modules])[None],
        "module_present": tensor([float(module.active) for module in modules])[None],
        "module_source_ids": tuple(module.module_id for module in modules),
        "material_params": tensor(
            [context[key] for key in ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius")]
        )[None],
    }
    for key in ("re", "u_in", "domain_length_x", "domain_length_y"):
        structure[key] = tensor([context[key]])[None]
    heat = tensor([module.heating for module in modules])[None]
    fluid_xy = np.asarray(record.output.roles["fluid_fields"].query_features[:, :2])
    # Fixed interior support selected from inputs. Shift off cell centers so a
    # central finite difference does not cross the bilinear derivative kink.
    interior = np.flatnonzero(
        (fluid_xy[:, 0] > 0.1 * context["domain_length_x"])
        & (fluid_xy[:, 0] < 0.9 * context["domain_length_x"])
        & (fluid_xy[:, 1] > 0.1 * context["domain_length_y"])
        & (fluid_xy[:, 1] < 0.9 * context["domain_length_y"])
    )
    indices = interior[np.linspace(0, len(interior) - 1, 128).astype(int)]
    xy = tensor(fluid_xy[indices].copy())[None]
    xy = xy + xy.new_tensor([0.173 * context["domain_length_x"] / 128, 0.211 * context["domain_length_y"] / 64])
    role = record.output.roles["solid_temperature"]
    local = np.stack(
        [
            np.asarray(role.query_features)[np.asarray(role.receiver_module_ids) == module.module_id, :2][:8]
            for module in modules
        ]
    )
    return structure, heat, xy, tensor(local)[None], indices


def discrepancy(actual, expected):
    actual, expected = actual.detach().double(), expected.detach().double()
    delta = actual - expected
    close = torch.isclose(actual, expected, rtol=2e-5, atol=2e-6)
    return {
        "max_abs_error": float(delta.abs().max()),
        "rms_error": float(delta.square().mean().sqrt()),
        "actual_max_abs": float(actual.detach().abs().max()),
        "expected_max_abs": float(expected.detach().abs().max()),
        "rtol": 2e-5,
        "atol": 2e-6,
        "pass": bool(close.all()),
        "failed_elements": int((~close).sum()),
        "compared_elements": int(close.numel()),
    }


def native_derivatives(model, record, device, *, evidence_path=None):
    synchronize(device)
    started = perf_counter()
    structure, heat, xy, local, indices = diagnostic_inputs(record, device)
    structure["module_centers"] = structure["module_centers"].clone().requires_grad_()
    xy = xy.clone().requires_grad_()
    heat = heat.clone().requires_grad_()
    prepared = model.prepare_native(structure, xy, local_query_points=local, ntheta=16)
    output = model.apply_native(prepared, heat)
    kernels = model.thermal.export_native_kernels(prepared["thermal"])
    roles = {
        "fluid_temperature": (output["fluid_temperature"], kernels["fluid"]),
        "surface_temperature": (output["pred_interface"][..., :1], kernels["surface"]),
        "material_temperature": (output["pred_internal_temperature"], kernels["material"]),
        "q_normal": (output["pred_interface"][..., 1:], kernels["q_normal"]),
    }
    heat_checks, arrays = {}, {}
    for name, (value, kernel) in roles.items():
        actual = torch.autograd.grad(value.mean(), heat, retain_graph=True)[0]
        expected = kernel.mean(dim=tuple(range(1, kernel.ndim - 2))).squeeze(-1)
        heat_checks[name] = discrepancy(actual, expected)
        arrays[f"heat_vjp/{name}/actual"] = actual.detach().cpu().numpy()
        arrays[f"heat_vjp/{name}/kernel"] = expected.detach().cpu().numpy()
    scalar = output["fluid_temperature"].mean()
    geometry_grad, query_grad = torch.autograd.grad(scalar, (structure["module_centers"], xy), retain_graph=True)
    composed_flow_heat_grad = torch.autograd.grad(output["pred_field"][..., :4].sum(), heat, retain_graph=True)[0]
    # Read the real flow operator alone; unused heat must have no graph edge.
    raw_flow = model.flow.read_flow(model.flow.prepare_flow(structure), xy)
    standalone_grad = torch.autograd.grad(raw_flow.sum(), heat, allow_unused=True, retain_graph=True)[0]
    detached_structure = {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in structure.items()
    }
    cold = lambda source, queries, forcing: model.apply_native(
        model.prepare_native(source, queries, local_query_points=local, ntheta=16), forcing
    )
    with torch.no_grad():
        epsilon = 1e-3
        moved_plus, moved_minus = copy.deepcopy(detached_structure), copy.deepcopy(detached_structure)
        moved_plus["module_centers"][0, 0, 0] += epsilon
        moved_minus["module_centers"][0, 0, 0] -= epsilon
        geometry_fd = (
            cold(moved_plus, xy.detach(), heat.detach())["fluid_temperature"].mean()
            - cold(moved_minus, xy.detach(), heat.detach())["fluid_temperature"].mean()
        ) / (2 * epsilon)
        query_step = xy.new_tensor([epsilon, 0])
        query_fd = (
            cold(detached_structure, xy.detach() + query_step, heat.detach())["fluid_temperature"].mean()
            - cold(detached_structure, xy.detach() - query_step, heat.detach())["fluid_temperature"].mean()
        ) / (2 * epsilon)
        delta = torch.zeros_like(heat)
        delta[0, 0] = 0.25
        if delta.shape[1] > 1:
            delta[0, 1] = -0.25
        baseline = cold(detached_structure, xy.detach(), heat.detach())
        changed = cold(detached_structure, xy.detach(), heat.detach() + delta)
        increment = model.thermal.apply_native(prepared["thermal"], delta, increment=True)
    flow_change = changed["pred_field"][..., :4].double() - baseline["pred_field"][..., :4].double()
    finite_heat = {}
    finite_roles = {
        "fluid_temperature": ("fluid_temperature", slice(None)),
        "surface_temperature": ("pred_interface", slice(0, 1)),
        "q_normal": ("pred_interface", slice(1, 2)),
        "material_temperature": ("pred_internal_temperature", slice(None)),
    }
    for name, (key, channel) in finite_roles.items():
        actual = changed[key][..., channel].double() - baseline[key][..., channel].double()
        expected = increment[key][..., channel].double()
        finite_heat[name] = discrepancy(actual, expected)
        arrays[f"cold_finite/{name}/actual"] = actual.cpu().numpy()
        arrays[f"cold_finite/{name}/kernel_delta"] = expected.cpu().numpy()
    arrays.update(
        physical_heat=heat.detach().cpu().numpy(),
        physical_delta_heat=delta.cpu().numpy(),
        shifted_fluid_xy=xy.detach().cpu().numpy(),
        module_centers=structure["module_centers"].detach().cpu().numpy(),
        geometry_gradient=geometry_grad.detach().cpu().numpy(),
        query_gradient=query_grad.detach().cpu().numpy(),
    )
    if evidence_path is not None:
        np.savez_compressed(evidence_path, **arrays)
    synchronize(device)
    return {
        "case_id": record.design.anchor_id,
        "fluid_rows": 128,
        "surface_rows_per_module": 16,
        "material_rows_per_module": 8,
        "input_selected_fluid_indices": indices.tolist(),
        "query_shift_cell_fractions": [0.173, 0.211],
        "physical_heat": heat.detach().cpu().tolist(),
        "finite_heat_delta": delta.cpu().tolist(),
        "native_heat_vjp_vs_physical_kernel": heat_checks,
        "geometry_gradient": {
            "finite": bool(torch.isfinite(geometry_grad).all()),
            "max_abs": float(geometry_grad.abs().max()),
            "first_source_x_ad": float(geometry_grad[0, 0, 0]),
            "first_source_x_fd": float(geometry_fd),
        },
        "query_gradient": {
            "finite": bool(torch.isfinite(query_grad).all()),
            "max_abs": float(query_grad.abs().max()),
            "all_receiver_x_ad": float(query_grad[..., 0].sum()),
            "all_receiver_x_fd": float(query_fd),
        },
        "finite_difference_epsilon": epsilon,
        "cold_heat_increment_checks": finite_heat,
        "cold_composed_flow_heat_change_max": float(flow_change.abs().max()),
        "composed_flow_heat_grad_max": float(composed_flow_heat_grad.abs().max()),
        "standalone_flow_heat_grad_is_unused": standalone_grad is None,
        "prepared_increment_flow_zero_status": "Mathematical signature; not the measured cold independence proof",
        "cold_complete_calls": 6,
        "differentiable_complete_calls": 1,
        "standalone_flow_prepare_read_calls": 1,
        "native_dense_kernel_exports": 1,
        "prepared_native_increment_calls": 1,
        "autograd_calls": 7,
        "diagnostic_elapsed_seconds": perf_counter() - started,
        "old_wrapper_calls": 0,
        "interpretation": "Actual model derivatives only; no physical geometry derivative accuracy claim or certified sign floor",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--only-native-derivatives",
        action="store_true",
        help="Skip full saved geometry reads for a bounded checkpoint review.",
    )
    args = parser.parse_args()
    output = validate_generated_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, checkpoint = load_source_response_model(args.checkpoint, args.device)
    digest = model_digest(model)
    started = perf_counter()
    geometry = [] if args.only_native_derivatives else saved_geometry(model, args.atlas_root, output, args.device)
    derivatives = []
    for case_id in ("0001", "0348"):
        stencil, _ = load_response_atlas_stencil(args.atlas_root / f"train_{case_id}_responses.npz")
        derivatives.append(
            native_derivatives(model, stencil.baseline, args.device, evidence_path=output / f"{case_id}_native_ad.npz")
        )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "device": args.device,
        "elapsed_seconds": perf_counter() - started,
        "geometry": geometry,
        "native_derivatives": derivatives,
        "weights_before": digest,
        "weights_after": model_digest(model),
        "solver_attempts": 0,
        "optimizer_updates": 0,
        "physical_reference_ledger": "326/326 unchanged",
        "no_initial_port_trajectory": True,
        "geometry_evaluation_completed": bool(geometry),
        "cold_complete_calls": sum(row["cold_complete_calls"] for row in geometry + derivatives),
        "differentiable_complete_calls": sum(row["differentiable_complete_calls"] for row in derivatives),
        "scope": (
            "Two bounded actual model derivative diagnostics only; geometry reference comparison skipped."
            if args.only_native_derivatives
            else "Four existing geometry families; two bounded actual model derivative diagnostics. Original-TRAIN saved shared-grid analytic-wake observations; no new physical reference."
        ),
    }
    if summary["weights_before"] != summary["weights_after"]:
        raise RuntimeError("Evaluation changed weights.")
    atomic_write_json(output / "summary.json", summary)
    print(
        json.dumps({"output": str(output / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True
    )


if __name__ == "__main__":
    main()
