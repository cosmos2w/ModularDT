"""Live-input AD/FD and source-law probes for a joint contextual predictor.

Derivatives concern the executed model at a fixed legal piecewise topology.
Agreement does not establish fidelity to physical geometry gradients. These
helpers use prescribed inputs and make no solver call or optimizer update.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .thermal_geometry_supervision import _native_inputs

_NAMES = ("u", "v", "p", "omega", "temperature_fluid", "temperature_surface", "q_proxy", "temperature_material")


def _readouts(output, prepared):
    field = output["pred_field"][0]
    fluid_mask = prepared.stencils["fluid"].valid[0].to(field.dtype)
    fluid = (field * fluid_mask[:, None]).sum(0) / fluid_mask.sum().clamp_min(1)
    interface = output["pred_interface"][0]
    shape = interface.shape[:-1]
    present = prepared.source_present[0, :, None]
    surface = prepared.stencils["surface"].valid[0].reshape(shape) & present
    outside = prepared.stencils["outside"].valid[0].reshape(shape) & surface
    material = output["pred_internal_temperature"][0, ..., 0]
    material_mask = prepared.stencils["material"].valid[0].reshape(material.shape) & present

    def mean(value, mask):
        return (value * mask.to(value.dtype)).sum() / mask.sum().clamp_min(1)

    return torch.cat(
        (
            fluid,
            torch.stack(
                (mean(interface[..., 0], surface), mean(interface[..., 1], outside), mean(material, material_mask))
            ),
        )
    )


def _signature(prepared, trace):
    organization = prepared.organization
    query_coordinates = prepared.query_batch.coordinates[0].detach().cpu().tolist()
    banks = {}
    for kind in ("module", "environment"):
        bank = getattr(organization, kind + "_banks")[0]
        forest = bank.forest
        original = () if forest is None else forest.original_ids
        edges = (
            ()
            if forest is None
            else tuple(
                sorted(tuple(sorted((repr(original[e.left]), repr(original[e.right])))) for e in forest.candidate_edges)
            )
        )
        hierarchy = (
            ()
            if forest is None
            else tuple(
                sorted(
                    (
                        tuple(sorted(map(repr, node.ancestry_ids))),
                        tuple(sorted(tuple(sorted(map(repr, forest.nodes[c].ancestry_ids))) for c in node.children)),
                    )
                    for node in forest.nodes
                    if node.children is not None
                )
            )
        )
        ancestry = tuple(tuple(sorted(repr(original[i]) for i in row)) for row in bank.group_ancestry)
        donors = tuple(
            sorted(
                (ancestry[j], tuple(sorted(repr(original[i]) for i in row)))
                for j, row in enumerate(bank.donor_local_indices)
            )
        )
        reads = trace[kind + "_source_read_attention"][0]
        positive = reads["weights"].detach().gt(0).any(-1)
        query_rows = reads["edge_query_rows"][positive].detach().cpu().tolist()
        source_rows = reads["edge_source_rows"][positive].detach().cpu().tolist()
        read_support = tuple(
            sorted((tuple(query_coordinates[q]), repr(original[i])) for q, i in zip(query_rows, source_rows))
        )
        access = trace[kind + "_group_access"][0].detach().gt(0)
        group_support = tuple(
            sorted((tuple(query_coordinates[q]), ancestry[int(g)]) for q, g in access.nonzero().cpu().tolist())
        )
        banks[kind] = {
            "K": bank.live_count,
            "N": bank.present_count,
            "candidate_edges": edges,
            "hierarchy": hierarchy,
            "frontier_ancestry": tuple(sorted(ancestry)),
            "donor_ids": donors,
            "source_read_support": read_support,
            "query_group_support": group_support,
        }
    masks = {role: tuple(v.valid.detach().cpu().reshape(-1).tolist()) for role, v in prepared.stencils.items()}
    return {"banks": banks, "native_masks": masks}


def _signature_change(base, trial):
    keys = {
        "candidate_graph_changed": "candidate_edges",
        "candidate_hierarchy_changed": "hierarchy",
        "frontier_changed": "frontier_ancestry",
        "donor_cover_changed": "donor_ids",
        "query_group_support_changed": "query_group_support",
        "source_read_support_changed": "source_read_support",
    }
    result = {
        label: any(base["banks"][bank][key] != trial["banks"][bank][key] for bank in base["banks"])
        for label, key in keys.items()
    }
    result["native_mask_changed"] = base["native_masks"] != trial["native_masks"]
    return result


def live_derivative_probe(model, record, fluid_xy, local_xy, *, geometry_slot=0, geometry_axis=0, step_fraction=1e-3):
    """Original-column source Jacobian plus a legal central geometry probe.

    The bounded receiver panel is caller-selected from inputs. Its material
    means are not full module-peak derivatives. Every FD changes physical
    inputs and completely re-prepares context, forest, frontier and access.
    """
    if not math.isfinite(step_fraction) or step_fraction <= 0 or geometry_axis not in (0, 1):
        raise ValueError("Use a positive finite step and a physical x/y axis.")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    ids = tuple(record.design.active_module_ids)
    structure, heat = _native_inputs(record, ids, device, dtype)
    if not 0 <= geometry_slot < len(ids):
        raise ValueError("Geometry probe slot is outside original source columns.")
    centers = structure["module_centers"].detach().clone().requires_grad_(True)
    heat = heat.detach().clone().requires_grad_(True)
    structure = {**structure, "module_centers": centers}
    fluid = torch.as_tensor(np.asarray(fluid_xy), device=device, dtype=dtype)
    local = torch.as_tensor(np.asarray(local_xy), device=device, dtype=dtype)
    if fluid.ndim == 2:
        fluid = fluid[None]
    if local.ndim == 2:
        local = local[None]
    modes = {module: module.training for module in model.modules()}
    model.eval()
    calls = 0

    def forward(geometry, controls):
        nonlocal calls
        prepared = model.prepare_native(
            {**structure, "module_centers": geometry},
            fluid,
            physical_heat=controls,
            local_query_points=local,
            ntheta=16,
            scene_key=("live_joint_derivative", str(record.design.active_module_ids)),
        )
        decoded = model.core.decode(prepared.organization, prepared.query_batch, return_trace=True)
        output = model._extract_native_roles(prepared, decoded.thermal_temperature, decoded.nonlinear_field)
        calls += 1
        return _readouts(output, prepared), prepared, _signature(prepared, decoded.trace)

    try:
        with torch.enable_grad():
            values, prepared, base_signature = forward(centers, heat)
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite native baseline readouts in derivative probe.")
            jac_h = []
            jac_g = []
            for scalar in values:
                dh, dg = torch.autograd.grad(scalar, (heat, centers), retain_graph=True, allow_unused=True)
                jac_h.append(torch.zeros_like(heat)[0] if dh is None else dh[0])
                jac_g.append(centers.new_zeros(()) if dg is None else dg[0, geometry_slot, geometry_axis])
            source_ad = torch.stack(jac_h).detach()
            geometry_ad = torch.stack(jac_g).detach()
        radius = float(structure["material_params"][0, 5])
        positions = centers.detach()
        other = [i for i in range(len(ids)) if i != geometry_slot]
        clearance = float("inf")
        if other:
            clearance = float(
                (torch.linalg.vector_norm(positions[0, other] - positions[0, geometry_slot], dim=-1) - 2 * radius).min()
            )
        extent = float(structure["domain_length_x" if geometry_axis == 0 else "domain_length_y"].reshape(-1)[0])
        coordinate = float(positions[0, geometry_slot, geometry_axis])
        legal_margin = min(coordinate - radius, extent - coordinate - radius, clearance / 2)
        if legal_margin <= 0:
            raise ValueError("The chosen baseline geometry has no legal two-sided probe margin.")
        geometry_step = min(step_fraction * radius, legal_margin / 4)
        source_slot = int(torch.argmax(heat.detach().abs()[0]))
        heat_scale = float(getattr(model, "heat_scale", 1.0))
        source_step = step_fraction * heat_scale
        source_center = float(heat.detach()[0, source_slot])
        source_central = source_center > source_step
        probes = []
        with torch.no_grad():
            for factor in (1.0, 0.5):
                step = geometry_step * factor
                delta = torch.zeros_like(positions)
                delta[0, geometry_slot, geometry_axis] = step
                plus, _pplus, plus_signature = forward(positions + delta, heat.detach())
                minus, _pminus, minus_signature = forward(positions - delta, heat.detach())
                fd = (plus - minus) / (2 * step)
                probes.append(
                    {
                        "kind": "geometry",
                        "step": step,
                        "module_id": ids[geometry_slot],
                        "original_column": int(prepared.scene_batch.module_ids[0, geometry_slot]),
                        "axis": "xy"[geometry_axis],
                        "AD": geometry_ad.cpu().tolist(),
                        "FD": fd.cpu().tolist(),
                        "absolute_AD_FD_error": (geometry_ad - fd).abs().cpu().tolist(),
                        "plus_seams": _signature_change(base_signature, plus_signature),
                        "minus_seams": _signature_change(base_signature, minus_signature),
                    }
                )
            delta = torch.zeros_like(heat)
            delta[0, source_slot] = source_step
            plus, _pplus, plus_signature = forward(positions, heat.detach() + delta)
            if source_central:
                minus, _pminus, minus_signature = forward(positions, heat.detach() - delta)
                source_fd = (plus - minus) / (2 * source_step)
            else:
                minus, _pminus, minus_signature = values.detach(), prepared, base_signature
                source_fd = (plus - minus) / source_step
            probes.append(
                {
                    "kind": "source",
                    "module_id": ids[source_slot],
                    "original_column": int(prepared.scene_batch.module_ids[0, source_slot]),
                    "scheme": "central" if source_central else "forward_nonnegative_control",
                    "step": source_step,
                    "AD": source_ad[:, source_slot].cpu().tolist(),
                    "FD": source_fd.cpu().tolist(),
                    "absolute_AD_FD_error": (source_ad[:, source_slot] - source_fd).abs().cpu().tolist(),
                    "plus_seams": _signature_change(base_signature, plus_signature),
                    "minus_seams": _signature_change(base_signature, minus_signature),
                }
            )
        return {
            "schema": "contextual_frontier_live_derivatives_v1",
            "observable_names": list(_NAMES),
            "source_module_ids": list(ids),
            "source_columns": prepared.scene_batch.module_ids[0].cpu().tolist(),
            "baseline_readouts": values.detach().cpu().tolist(),
            "source_jacobian_AD": source_ad.cpu().tolist(),
            "live_groups": {kind: {"K": row["K"], "N": row["N"]} for kind, row in base_signature["banks"].items()},
            "probes": probes,
            "native_forward_calls": calls,
            "optimizer_updates": 0,
            "solver_calls": 0,
            "interpretation": "Executed model AD/FD consistency at piecewise topology; no physical gradient fidelity claim",
            "receiver_scope": "bounded caller-selected fluid/surface/material means; no module-peak derivative",
        }
    finally:
        for module, mode in modes.items():
            module.training = mode


def source_law_and_reuse_probe(model, record, fluid_xy, local_xy):
    """Measure Thermal law residuals and fixed-input prepare reuse separately.

    Zero-temperature and heat-independent-flow identities apply only to the
    saved simplified Thermal setting with homogeneous thermal boundaries.
    They are analytic case-law checks, not new solver observations. The
    supplied material panel defines the scope; no endpoint claim is made.
    """
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    ids = tuple(record.design.active_module_ids)
    structure, heat = _native_inputs(record, ids, device, dtype)
    fluid = torch.as_tensor(np.asarray(fluid_xy), device=device, dtype=dtype)
    local = torch.as_tensor(np.asarray(local_xy), device=device, dtype=dtype)
    if fluid.ndim == 2:
        fluid = fluid[None]
    if local.ndim == 2:
        local = local[None]
    modes = {module: module.training for module in model.modules()}
    model.eval()
    counts = {"module_encoder": 0, "environment_encoder": 0, "module_value": 0, "environment_value": 0}
    hooks = []
    for name in counts:

        def count_call(_module, _inputs, _output, key=name):
            counts[key] += 1

        hooks.append(getattr(model.core, name).register_forward_hook(count_call))
    native_calls = 0

    def prepare(controls):
        return model.prepare_native(
            structure,
            fluid,
            physical_heat=controls,
            local_query_points=local,
            ntheta=16,
            scene_key=("joint_source_law_probe", ids),
        )

    def decode(prepared, controls):
        nonlocal native_calls
        native_calls += 1
        return model.apply_native(prepared, controls)

    def components(output):
        return {
            **{name: output["pred_field"][0, :, i] for i, name in enumerate(_NAMES[:5])},
            "temperature_surface": output["pred_interface"][0, ..., 0],
            "q_proxy": output["pred_interface"][0, ..., 1],
            "temperature_material": output["pred_internal_temperature"][0, ..., 0],
        }

    def rms(value, mask):
        selected = value[mask]
        if not selected.numel() or not torch.isfinite(selected).all():
            return None
        return float(selected.double().square().mean().sqrt())

    try:
        with torch.no_grad():
            base_prepared = prepare(heat)
            base = decode(base_prepared, heat)
            before_reuse = dict(counts)
            reused = decode(base_prepared, heat)
            reuse_calls = {key: counts[key] - before_reuse[key] for key in counts}
            fresh = decode(prepare(heat), heat)
            a = heat.clone()
            a[:, 1::2] = 0
            b = heat - a
            zero = torch.zeros_like(heat)
            split_a = decode(prepare(a), a)
            split_b = decode(prepare(b), b)
            zero_output = decode(prepare(zero), zero)
            values = [components(out) for out in (base, reused, fresh, split_a, split_b, zero_output)]
            present = base_prepared.source_present[0, :, None]
            surface_shape = base["pred_interface"].shape[1:-1]
            surface = base_prepared.stencils["surface"].valid[0].reshape(surface_shape) & present
            outside = base_prepared.stencils["outside"].valid[0].reshape(surface_shape) & surface
            material = (
                base_prepared.stencils["material"].valid[0].reshape(base["pred_internal_temperature"].shape[1:-1])
                & present
            )
            masks = {
                **{name: base_prepared.stencils["fluid"].valid[0] for name in _NAMES[:5]},
                "temperature_surface": surface,
                "q_proxy": outside,
                "temperature_material": material,
            }
            rows = {}
            for name, mask in masks.items():
                full, cached, rebuilt, part_a, part_b, at_zero = (row[name] for row in values)
                rows[name] = {
                    "physical_rows": int(mask.sum()),
                    "fixed_prepare_reuse_difference_rms": rms(cached - full, mask),
                    "fresh_prepare_difference_rms": rms(rebuilt - full, mask),
                    "heat_additivity_residual_rms": rms(full - part_a - part_b + at_zero, mask),
                    "zero_heat_prediction_rms": rms(at_zero, mask),
                    "full_heat_minus_zero_rms": rms(full - at_zero, mask),
                }
        return {
            "schema": "contextual_frontier_source_law_reuse_v1",
            "source_module_ids": list(ids),
            "source_columns": base_prepared.scene_batch.module_ids[0].cpu().tolist(),
            "heat_partition": "even original-input slots versus odd slots; disjoint, a+b=h",
            "partition_is_nontrivial": bool((a != 0).any() and (b != 0).any()),
            "per_component": rows,
            "native_forward_calls": native_calls,
            "total_observed_neural_calls": counts,
            "cached_apply_neural_calls": reuse_calls,
            "optimizer_updates": 0,
            "solver_calls": 0,
            "reference_scope": "analytic saved-Thermal case laws; no new physical solver reference",
            "zero_heat_interpretation": "temperature/q compared with analytic zero; flow magnitude itself is not a zero-flow error",
            "flow_interpretation": "full_heat_minus_zero_rms tests heat-independent flow in this case only",
            "additivity_interpretation": "affine residual with F(0) removed; not an exact law of the generic joint model",
            "reuse_interpretation": "fixed-input preparation reuse still re-enters neural readers; no affine response-factor cache",
            "receiver_scope": "caller-selected physical fluid/surface/material panel, no module-peak law",
        }
    finally:
        for hook in hooks:
            hook.remove()
        for module, mode in modes.items():
            module.training = mode
