"""Matched-point Thermal direct and terminal local-readout review."""
from __future__ import annotations

import time
from types import SimpleNamespace
import torch

from .local_coupling import build_local_module_params_from_global
from .local_ports import organized_outside64


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rmse(prediction, reference, valid):
    mask = valid.bool() & torch.isfinite(reference) & torch.isfinite(prediction)
    count = int(mask.sum())
    if not count:
        return {"rmse": None, "count": 0}
    difference = prediction[mask] - reference[mask]
    return {"rmse": float(difference.double().square().mean().sqrt()), "count": count}


def _oracle_module_valid(present, outside_valid, h_valid, teacher_ports):
    """Local oracle conditions require every one of a module's 64 ports."""
    valid = (outside_valid.bool() & h_valid.bool() & torch.isfinite(teacher_ports[..., 3])
             & torch.isfinite(teacher_ports[..., 4]))
    return present.bool() & valid.all(-1)


def evaluate_direct_native(model, batch):
    """Evaluate a direct Thermal control on its original native receiver masks."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _sync(device)
            started = time.perf_counter()
            prepared = model.prepare_native(batch.scene_inputs.structure, batch.receivers.fluid_xy,
                                            physical_heat=batch.receivers.heat,
                                            local_query_points=batch.receivers.local_query_points,
                                            ntheta=batch.receivers.ntheta,
                                            scene_key=batch.scene_inputs.case_ids)
            _sync(device)
            prep_seconds = time.perf_counter() - started
            started = time.perf_counter()
            with model.core.read_execution("restricted"):
                native, work, _ = model.apply_native_with_work_counts(prepared, batch.receivers.heat)
            _sync(device)
            decode_seconds = time.perf_counter() - started
            target = batch.targets
            field = target.field_targets.to(native["pred_field"])
            fluid = (target.point_weights.to(field) > 0) & prepared.stencils["fluid"].valid
            material = target.material_targets.to(native["pred_internal_temperature"])
            solid = (prepared.source_present.bool()[..., None]
                     & prepared.stencils["material"].valid.reshape_as(material) & torch.isfinite(material))
            interface = target.interface_target.to(native["pred_interface"])
            shape = interface.shape[:-1]
            base = (prepared.source_present.bool()[..., None] & (target.interface_valid_mask > .5)
                    & prepared.stencils["surface"].valid.reshape(shape))
            surface = base & torch.isfinite(interface[..., 0])
            flux = base & prepared.stencils["outside"].valid.reshape(shape) & torch.isfinite(interface[..., 1])
            solid_pred = native["pred_internal_temperature"][..., 0]
            module_valid = solid.any(-1)
            pred_peak = torch.where(solid, solid_pred, solid_pred.new_full((), -torch.inf)).amax(-1)
            true_peak = torch.where(solid, material, material.new_full((), -torch.inf)).amax(-1)
            metrics = {"direct_fluid": {name: _rmse(native["pred_field"][..., i], field[..., i], fluid)
                                         for i, name in enumerate(("u", "v", "pressure", "vorticity", "temperature"))},
                       "direct": {"solid_temperature": _rmse(solid_pred, material, solid),
                                  "module_peak_sampled_points": _rmse(pred_peak, true_peak, module_valid),
                                  "surface_temperature": _rmse(native["pred_interface"][..., 0], interface[..., 0], surface),
                                  "q_proxy": _rmse(native["pred_interface"][..., 1], interface[..., 1], flux)}}
            return {"direct": native, "targets": target,
                    "masks": {"fluid": fluid, "fluid_fields": fluid[..., None] & torch.isfinite(field),
                              "solid": solid, "surface": surface, "flux": flux},
                    "metrics": metrics,
                    "work": {**{f"prepare_{k}": v for k, v in prepared.organization.work_counts.items()},
                             **{f"direct_{k}": v for k, v in work.items()}},
                    "timing_seconds": {"organization_and_native_preparation": prep_seconds,
                                       "direct_decode": decode_seconds,
                                       "complete_direct": prep_seconds + decode_seconds}}
    finally:
        model.train(was_training)


def evaluate_local_port_readouts(model, provider, batch, *, local_readout=None):
    """Evaluate four readouts on the same native points without feedback.

    The provider prepares one organization and performs one restricted native
    prediction plus its restricted 64-port read. Each local path is one batched
    call over the present modules. Timing excludes asset loading and includes
    synchronous execution; work counters are those returned by the core.
    """
    device = next(model.parameters()).device
    _sync(device)
    loading_started = time.perf_counter()
    local = local_readout if local_readout is not None else provider.build_local_readout()
    local.eval()
    _sync(device)
    asset_seconds = time.perf_counter() - loading_started
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _sync(device)
            started = time.perf_counter()
            prepared = model.prepare_native(batch.scene_inputs.structure, batch.receivers.fluid_xy,
                                            physical_heat=batch.receivers.heat,
                                            local_query_points=batch.receivers.local_query_points,
                                            ntheta=batch.receivers.ntheta,
                                            scene_key=batch.scene_inputs.case_ids)
            _sync(device)
            prepare_seconds = time.perf_counter() - started
            started = time.perf_counter()
            with model.core.read_execution("restricted"):
                native, direct_work, _ = model.apply_native_with_work_counts(prepared, batch.receivers.heat)
            _sync(device)
            direct_seconds = time.perf_counter() - started
            started = time.perf_counter()
            outside, valid, port_work = organized_outside64(model, prepared)
            prescribed = build_local_module_params_from_global(
                batch.receivers.heat, None, batch.scene_inputs.structure["material_params"],
                prepared.source_present)[..., :3]
            port_values = model.port_head(outside, prescribed, prepared.source_present)
            _sync(device)
            port_seconds = time.perf_counter() - started
            prediction = SimpleNamespace(native_main=native, port_values=port_values,
                                         port_outside=outside, port_valid=valid,
                                         work={**{f"prepare_{k}": v for k, v in prepared.organization.work_counts.items()},
                                               **{f"direct_{k}": v for k, v in direct_work.items()},
                                               **{f"ports_{k}": v for k, v in port_work.items()}})
            present = prepared.source_present.bool()
            teacher = batch.targets.teacher_ports.to(prediction.port_values)
            oracle_modules = _oracle_module_valid(present, prediction.port_valid,
                                                  batch.targets.teacher_h_valid, teacher)
            mean_ports = prediction.port_values.clone()
            mean_ports[..., 3] = prediction.port_outside
            mean_ports[..., 4] = model.port_head.mean[1]
            mean_ports *= present[..., None, None]
            ports = {"predicted": prediction.port_values, "oracle": teacher,
                     "train_mean_h": mean_ports}
            paths = {}
            timings = {}
            for name, port in ports.items():
                _sync(device)
                started = time.perf_counter()
                paths[name] = local(batch.scene_inputs.structure, batch.receivers.heat, port,
                                    batch.receivers.local_query_points)
                _sync(device)
                timings[name] = time.perf_counter() - started

            target = batch.targets
            field_target = target.field_targets.to(native["pred_field"])
            fluid_mask = ((target.point_weights.to(field_target) > 0)
                          & prepared.stencils["fluid"].valid)
            solid_target = target.material_targets.to(native["pred_internal_temperature"])
            solid_mask = (present[..., None] & prepared.stencils["material"].valid.reshape_as(solid_target)
                          & torch.isfinite(solid_target))
            interface_target = target.interface_target.to(native["pred_interface"])
            shape = interface_target.shape[:-1]
            surface_mask = (present[..., None] & prepared.stencils["surface"].valid.reshape(shape)
                            & (target.interface_valid_mask > .5) & torch.isfinite(interface_target[..., 0]))
            flux_mask = (present[..., None] & prepared.stencils["surface"].valid.reshape(shape)
                         & prepared.stencils["outside"].valid.reshape(shape)
                         & (target.interface_valid_mask > .5) & torch.isfinite(interface_target[..., 1]))
            direct_solid = native["pred_internal_temperature"][..., 0]
            direct_interface = native["pred_interface"]

            def native_metrics(solid, interface, module_mask):
                sm = solid_mask & module_mask[..., None]
                sf = surface_mask & module_mask[..., None]
                fm = flux_mask & module_mask[..., None]
                solid_safe = torch.where(sm, solid, solid.new_full((), -torch.inf))
                target_safe = torch.where(sm, solid_target, solid_target.new_full((), -torch.inf))
                module_valid = sm.any(-1)
                return {
                    "solid_temperature": _rmse(solid, solid_target, sm),
                    "module_peak_sampled_points": _rmse(solid_safe.amax(-1), target_safe.amax(-1), module_valid),
                    "surface_temperature": _rmse(interface[..., 0], interface_target[..., 0], sf),
                    "q_proxy": _rmse(interface[..., 1], interface_target[..., 1], fm),
                    "interface_disagreement_to_direct": _rmse(interface[..., 0], direct_interface[..., 0], sf),
                    "interface_flux_disagreement_to_direct": _rmse(interface[..., 1], direct_interface[..., 1], fm),
                }

            metrics = {"direct": native_metrics(direct_solid, direct_interface, present)}
            ntheta = int(interface_target.shape[-2])
            if ntheta not in (16, 64):
                raise ValueError("Matched Thermal review requires the native 16- or 64-angle interface.")
            stride = 64 // ntheta
            for name, output in paths.items():
                module_mask = oracle_modules if name == "oracle" else present
                metrics[name] = native_metrics(output["internal_temperature"][..., 0], output["interface_pred"][..., ::stride, :], module_mask)
            port_mask = present[..., None] & prediction.port_valid
            metrics["ports"] = {
                "outside_temperature": _rmse(prediction.port_values[..., 3], teacher[..., 3], port_mask),
                "h_effective": _rmse(prediction.port_values[..., 4], teacher[..., 4],
                                     port_mask & target.teacher_h_valid.bool()),
            }
            metrics["direct_fluid"] = {
                name: _rmse(native["pred_field"][..., index], field_target[..., index], fluid_mask)
                for index, name in enumerate(("u", "v", "pressure", "vorticity", "temperature"))
            }
            return {"direct": native, "local": paths, "ports": ports,
                    "targets": target, "masks": {"fluid": fluid_mask, "fluid_fields": fluid_mask[..., None] & torch.isfinite(field_target),
                                                    "solid": solid_mask, "surface": surface_mask,
                                                    "flux": flux_mask, "port": port_mask,
                                                    "oracle_module": oracle_modules,
                                                    "oracle_solid": solid_mask & oracle_modules[..., None],
                                                    "oracle_surface": surface_mask & oracle_modules[..., None],
                                                    "oracle_flux": flux_mask & oracle_modules[..., None],
                                                    "port_t": port_mask & torch.isfinite(teacher[..., 3]),
                                                    "port_h": port_mask & target.teacher_h_valid.bool() & torch.isfinite(teacher[..., 4])},
                    "metrics": metrics, "work": dict(prediction.work),
                    "timing_seconds": {"frozen_asset_load_or_binding": asset_seconds,
                                       "organization_and_native_preparation": prepare_seconds,
                                       "direct_decode": direct_seconds, "port_read_and_head": port_seconds,
                                       "complete_direct": prepare_seconds + direct_seconds, **timings,
                                       **{f"complete_{name}": prepare_seconds + direct_seconds + port_seconds + value for name, value in timings.items()}},
                    "timing_scope": "synchronized forward readout; complete paths include global and one local call; asset load is separate",
                    "h_semantics": "Stored target uses epsilon regularization, configured clipping and validity mask; predicted h is nonnegative without an additional upper clip.",
                    "oracle_conditioning": {"valid_modules": int(oracle_modules.sum()),
                        "present_modules": int(present.sum()),
                        "rule": "Only modules with all 64 outside receivers valid and finite teacher T/h with valid h are scored."},
                    "deployment_limits": {"present_port_count": int(present.sum()) * int(prediction.port_valid.shape[-1]),
                        "outside_invalid_port_count": int((present[..., None] & ~prediction.port_valid.bool()).sum()),
                        "modules_with_invalid_outside_port": int((present & ~prediction.port_valid.all(-1)).sum())},
                    "peak_scope": "maximum over the sampled native material points per present module"}
    finally:
        model.train(was_training)
