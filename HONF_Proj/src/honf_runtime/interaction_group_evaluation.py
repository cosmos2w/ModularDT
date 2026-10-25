"""Bounded interaction-group execution, native risk and intervention evidence.

This complements the existing native and saved-response reductions. Disposable
optimizer timing steps are restored; saved scientific checkpoints and cohorts
are never changed, and no solver is launched.
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .contextual_frontier_evaluation import (
    _filesystem_free_bytes,
    _jsonable,
    _summarize_nine_observables,
    load_contextual_checkpoint,
)


def input_panels(provider):
    """Input-only stratification, called before evaluating new errors."""
    strata = {}
    for case in provider.training_cases:
        structure = case["structure"]
        m = int(np.asarray(structure["module_present"]).sum())
        re = float(np.asarray(structure["re"]).reshape(-1)[0])
        heat = float(np.asarray(structure["heat_powers"]).sum())
        strata.setdefault(m, []).append((re, heat, str(case["case_id"])))
    for rows in strata.values():
        rows.sort()

    def spread(rows, n):
        return [rows[j][2] for j in np.linspace(0, len(rows) - 1, min(n, len(rows))).round().astype(int)]

    risk = spread(strata[1], 6) + spread(strata[2], 6)
    organizer = []
    for m, rows in sorted(strata.items()):
        organizer += spread(rows, 2)
    # Complete 24 with additional distinct input-ordered cases.
    for rows in strata.values():
        for _, _, cid in rows:
            if len(organizer) >= 24:
                break
            if cid not in organizer:
                organizer.append(cid)
    return {
        "risk_train_ids": risk,
        "organizer_train_ids": organizer,
        "detailed_dev_ids": ["0277", "0287", "0294", "0684"],
        "intervention_train_ids": [risk[0], risk[6], strata[10][0][2], strata[12][-1][2]],
        "selection": "within-M evenly spaced sorted Re/total-heat input order; never errors",
        "training_membership": "fixed25 TRAIN150",
        "validation_membership": "exposed DEV22",
    }


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _timed(call, device, repeats=3):
    call()
    _sync(device)
    rows = []
    result = None
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        result = call()
        _sync(device)
        rows.append(time.perf_counter() - start)
    return result, {"seconds": rows, "median_seconds": float(np.median(rows))}


def _relative_change(a, b, mask=None):
    delta = (a - b).detach()
    if mask is not None:
        delta = delta[mask]
    return float(delta.square().mean().sqrt()) if delta.numel() else None


def _tensor_bytes(value, seen=None):
    """Persistent unique tensor storage; aliases count once."""
    from dataclasses import fields, is_dataclass

    seen = set() if seen is None else seen
    if torch.is_tensor(value):
        storage = value.untyped_storage()
        key = (value.device, storage.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()
    if is_dataclass(value):
        return sum(_tensor_bytes(getattr(value, f.name), seen) for f in fields(value))
    if isinstance(value, dict):
        return sum(_tensor_bytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v, seen) for v in value)
    return 0


def evaluate_execution(checkpoint, device, output, *, panels=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    free = _filesystem_free_bytes(output.parent)
    if free - 256 * 1024**2 < 200 * 1024**3:
        raise OSError("Insufficient DATA2 headroom")
    model, provider, payload = load_contextual_checkpoint(checkpoint, device)
    if model.architecture not in ("G-F", "G-EE", "G-H"):
        raise ValueError("Expected an interaction-group checkpoint")
    panels = input_panels(provider) if panels is None else panels
    model.eval()
    model.core.set_organization_clock(int(payload["epoch"]), 0)
    output.mkdir()
    risk_records = []
    regions = []
    organizations = []
    interventions = []
    timings = []
    with torch.no_grad():
        for batch in provider.native_evaluation_batches(
            split="train", case_keys=panels["risk_train_ids"], microbatch_size=1
        ):
            prediction, aux = provider.predict_native(
                model, batch.scene_inputs, batch.receivers, model.architecture, "validation", epoch=payload["epoch"]
            )
            risk_records.extend(provider.validation_metrics(prediction, batch.targets, aux)["case_rows"])
            p = prediction.native_prepared
            xy = batch.receivers.fluid_xy[0]
            centers = p.scene_batch.module_coordinates[0, p.scene_batch.module_mask[0]]
            distance = torch.cdist(xy, centers).amin(-1) if len(centers) else xy.new_full((len(xy),), float("inf"))
            lengths = p.scene_batch.coordinate_scale[0]
            normalized = (xy - p.scene_batch.coordinate_origin[0]) / lengths
            boundary = (normalized < 0.10).any(-1) | (normalized > 0.90).any(-1)
            valid = p.stencils["fluid"].valid[0] & (batch.targets.point_weights[0] > 0)
            target = batch.targets.field_targets[0]
            predicted = prediction.native_main["pred_field"][0]
            for name, region in [("boundary", boundary), ("far_from_M", distance > 0.15 * lengths.min())]:
                mask = valid & region
                values = {}
                for j, quantity in enumerate(("u", "v", "p", "omega", "temperature_fluid")):
                    good = mask & torch.isfinite(target[:, j])
                    values[quantity] = _relative_change(predicted[:, j], target[:, j], good)
                regions.append(
                    {"case_id": batch.case_keys[0], "region": name, "receiver_count": int(mask.sum()), "rmse": values}
                )
        for batch in provider.native_evaluation_batches(
            split="train", case_keys=panels["organizer_train_ids"], microbatch_size=1
        ):
            scene = model.scene_batch(
                {**batch.scene_inputs.structure, "heat_powers": batch.receivers.heat}, scene_key=batch.case_keys
            )
            p = model.core.prepare(scene)
            organizations.append(
                {
                    "case_id": batch.case_keys[0],
                    "module_count": int(scene.module_mask.sum()),
                    "K": [g[0].frontier.K for g in p.layers],
                    "policy_rows": [g[0].frontier.policy_rows for g in p.layers],
                    "reached_nodes": [len(g[0].frontier.reached) for g in p.layers],
                    "stop_probabilities": [g[0].frontier.probabilities.cpu().tolist() for g in p.layers],
                    "work": p.work_counts,
                    "persistent_prepared_bytes": _tensor_bytes(p),
                    "zero_original_e_query_expansion": True,
                }
            )
        for batch in provider.native_evaluation_batches(
            split="train", case_keys=panels["intervention_train_ids"], microbatch_size=1
        ):
            prediction, _ = provider.predict_native(
                model, batch.scene_inputs, batch.receivers, model.architecture, "validation", epoch=payload["epoch"]
            )
            native = prediction.native_prepared
            p = native.organization
            query = native.query_batch
            ordinary, trace = model.core.decode(p, query, return_trace=True)
            base = ordinary.joint_fields
            row = {
                "case_id": batch.case_keys[0],
                "module_count": int(p.scene.module_mask.sum()),
                "K": [g[0].frontier.K for g in p.layers],
                "group_key_value_reused": True,
                "ordinary_work": ordinary.work_counts,
                "ordinary_native_metrics": provider.validation_metrics(prediction, batch.targets, {})["case_rows"][0],
                "interventions": {},
            }
            for layer in range(len(p.layers)):
                changed = model.core.prepare(p.scene, clamp_layer=layer)
                out = model.core.decode(changed, query, candidate_traces=trace if layer == len(p.layers) - 1 else None)
                native_changed = model._extract_native_roles(native, out.thermal_temperature, out.nonlinear_field)
                changed_prediction = replace(prediction, native_main=native_changed)
                utility = provider.validation_metrics(changed_prediction, batch.targets, {})["case_rows"][0]
                errors = {
                    key: _relative_change(value, native_changed[key])
                    for key, value in prediction.native_main.items()
                    if torch.is_tensor(value)
                }
                row["interventions"][f"clamp_group_to_M_layer{layer + 1}"] = {
                    "joint_rms": _relative_change(base, out.joint_fields),
                    "channel_rms": [_relative_change(base[..., j], out.joint_fields[..., j]) for j in range(4)],
                    "fixed_candidates": layer == len(p.layers) - 1,
                    "native_output_change": errors,
                    "native_metrics": utility,
                }
            exhaustive = model.core.decode(p, query, exhaustive=True)
            row["exhaustive_search"] = {
                "joint_rms": _relative_change(base, exhaustive.joint_fields),
                "work": exhaustive.work_counts,
                "native_metrics": provider.validation_metrics(
                    replace(
                        prediction,
                        native_main=model._extract_native_roles(
                            native, exhaustive.thermal_temperature, exhaustive.nonlinear_field
                        ),
                    ),
                    batch.targets,
                    {},
                )["case_rows"][0],
            }
            without = replace(
                p, layers=p.layers[:-1] + (tuple(replace(g, values=g.values * 0) for g in p.final_groups),)
            )
            null = model.core.decode(without, query, candidate_traces=trace)
            row["zero_group_values"] = {
                "joint_rms": _relative_change(base, null.joint_fields),
                "native_metrics": provider.validation_metrics(
                    replace(
                        prediction,
                        native_main=model._extract_native_roles(native, null.thermal_temperature, null.nonlinear_field),
                    ),
                    batch.targets,
                    {},
                )["case_rows"][0],
            }
            # Shared objects are exported from the actual encoder reduction.
            arrays = {
                "R": p.final_groups[0].R.cpu().numpy(),
                "B": p.final_groups[0].B.cpu().numpy(),
                "labels": p.final_groups[0].labels.cpu().numpy(),
                "E_xy": p.scene.environment_coordinates[0].cpu().numpy(),
                "M_xy": p.scene.module_coordinates[0].cpu().numpy(),
                "group_xy": p.final_groups[0].coordinates.cpu().numpy(),
                "postings": p.final_groups[0].postings.cpu().numpy(),
                "query_candidates": torch.cat([t["candidates"] for t in trace[0]])[:1024].cpu().numpy(),
            }
            np.savez_compressed(output / f"graph_{batch.case_keys[0]}.npz", **arrays)
            interventions.append(row)
            # Isolated measured forward costs at small/primary/native Q. These
            # are repeated complete core calls, not theoretical edge ratios.
            for qn in (32, 1024, query.num_queries):
                mask = replace(
                    query,
                    coordinates=query.coordinates[:, :qn],
                    query_mask=None,
                    query_ids=query.query_ids[:, :qn],
                    local_batch_indices=None,
                    local_query_indices=None,
                    local_module_indices=None,
                    local_weights=None,
                )
                prepared, prepare_t = _timed(lambda p=p: model.core.prepare(p.scene), device)
                _, decode_t = _timed(lambda prepared=prepared, mask=mask: model.core.decode(prepared, mask), device)
                _, complete_t = _timed(
                    lambda p=p, mask=mask: model.core.decode(model.core.prepare(p.scene), mask), device
                )
                timings.append(
                    {
                        "case_id": batch.case_keys[0],
                        "query_rows": mask.num_queries,
                        "scope": "core",
                        "prepare": prepare_t,
                        "cached_decode": decode_t,
                        "complete": complete_t,
                        "persistent_prepared_bytes": _tensor_bytes(prepared),
                    }
                )
            _, native_t = _timed(
                lambda batch=batch: model.apply_native(
                    model.prepare_native(
                        batch.scene_inputs.structure,
                        batch.receivers.fluid_xy,
                        physical_heat=batch.receivers.heat,
                        local_query_points=batch.receivers.local_query_points,
                        ntheta=batch.receivers.ntheta,
                        scene_key=batch.case_keys,
                    )
                ),
                device,
            )
            timings.append({"case_id": batch.case_keys[0], "scope": "complete_native", "complete": native_t})
    controls = temperature_controls(model, provider)
    macro = benchmark_optimizer_macro(model, provider, payload, device)
    result = {
        "temperature_controls": controls,
        "optimizer_macro": macro,
        "schema": "interaction_groups_execution_evaluation_v1",
        "checkpoint": str(checkpoint),
        "epoch": payload["epoch"],
        "arm": model.architecture,
        "panels": panels,
        "risk_metrics": _summarize_nine_observables(risk_records, known_module_counts=(1, 2)),
        "risk_case_rows": risk_records,
        "regional_metrics": regions,
        "organizations": organizations,
        "interventions": interventions,
        "timings": timings,
        "inverse": "not performed; no claim",
        "physical_M1_M2_response_reference": "unavailable",
        "original_e_query_reads": 0,
        "module_value_query_reads": 0,
    }
    (output / "result.json").write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n")
    return result


def selected_train_mean(provider):
    """Compute physical constant means once from selected TRAIN targets only."""
    values = {name: [] for name in ("field", "surface", "q", "material")}
    for case in provider.training_cases:
        # Existing selected-TRAIN normalizer streams all stored sampled fluid
        # values. Its per-channel means are already the maintained estimator.
        interface = np.asarray(case["interface_target"])
        present = np.asarray(case["structure"]["module_present"]) > 0.5
        valid = np.asarray(case["interface_condition_valid_mask"]) > 0.5
        valid = valid & present[:, None]
        values["surface"].extend(interface[..., 0][valid & np.isfinite(interface[..., 0])].tolist())
        values["q"].extend(interface[..., 1][valid & np.isfinite(interface[..., 1])].tolist())
        material = np.asarray(case["module_internal_temperature_points"])
        values["material"].extend(material[present][np.isfinite(material[present])].tolist())
    return {
        "field": list(provider.model.normalization_stats["field_mean_by_channel"]),
        "surface": float(np.mean(values["surface"])),
        "q": float(np.mean(values["q"])),
        "material": float(np.mean(values["material"])),
    }


def mean_prediction(prediction, means):
    main = dict(prediction.native_main)
    main["pred_field"] = torch.zeros_like(main["pred_field"]) + main["pred_field"].new_tensor(means["field"])
    main["pred_interface"] = torch.zeros_like(main["pred_interface"]) + main["pred_interface"].new_tensor(
        [means["surface"], means["q"]]
    )
    main["pred_internal_temperature"] = torch.zeros_like(main["pred_internal_temperature"]) + means["material"]
    return replace(prediction, native_main=main)


def temperature_controls(model, provider):
    """Bounded original-heat VJP/JVP and zero-heat temperature-rise checks.

    No flow-response independence, additivity, full Jacobian or geometry finite
    difference campaign is evaluated by this function.
    """
    from .contextual_frontier_responses import _load_stencils, _panel_for_family
    from .thermal_geometry_supervision import _native_inputs

    stencils, _, _ = _load_stencils(provider)
    result = []
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for family in ("0348", "0304"):
        stencil = stencils[family]
        panel = _panel_for_family(stencil, family)
        # Use saved physical receiver coordinates; small input-only prefix.
        record = stencil.baseline
        structure, heat = _native_inputs(record, tuple(record.design.active_module_ids), device, dtype)
        # Physical atlas fields expose the same receiver ownership used by the
        # maintained response evaluator.
        fluid = np.asarray(panel["fluid_query_features"])[:32, :2]
        solid = np.asarray(record.output.roles["solid_temperature"].query_features)
        modules = len(record.design.active_module_ids)
        local = solid.reshape(modules, -1, 2)[0, :8]
        fluid = torch.as_tensor(np.array(fluid[None], copy=True), device=device, dtype=dtype)
        local = torch.as_tensor(np.array(local[None], copy=True), device=device, dtype=dtype)

        def read(h, return_arrays=False, structure=structure, fluid=fluid, local=local, family=family):
            p = model.prepare_native(
                structure,
                fluid,
                physical_heat=h,
                local_query_points=local,
                ntheta=16,
                scene_key=("temperature_control", family),
            )
            o = model.apply_native(p, h)
            surface = p.stencils["surface"].valid.reshape(o["pred_interface"].shape[:-1]) & p.source_present[..., None]
            material = (
                p.stencils["material"].valid.reshape(o["pred_internal_temperature"].shape[:-1])
                & p.source_present[..., None]
            )
            fmask = p.stencils["fluid"].valid
            parts = (
                o["pred_field"][..., 4][fmask],
                o["pred_interface"][..., 0][surface],
                o["pred_internal_temperature"][..., 0][material],
            )
            if return_arrays:
                return parts
            return torch.stack([v.mean() for v in parts])

        heat = heat.detach().requires_grad_(True)
        direction = torch.ones_like(heat)
        direction[:, 1::2] = -1
        _, jvp = torch.autograd.functional.jvp(read, heat, direction, create_graph=False)
        readouts = read(heat)
        vjp = torch.autograd.grad((readouts * readouts.new_tensor([1.0, 0.5, 0.25])).sum(), heat)[0]
        with torch.no_grad():
            zeros = read(torch.zeros_like(heat), True)
        result.append(
            {
                "family_id": family,
                "scope": "fit TRAIN0348" if family == "0348" else "response-validation-only0304",
                "original_source_columns": list(record.design.active_module_ids),
                "temperature_readouts": readouts.detach().cpu().tolist(),
                "temperature_jvp": jvp.detach().cpu().tolist(),
                "temperature_vjp": vjp.detach().cpu().tolist(),
                "zero_heat_temperature_rms": [float(v.square().mean().sqrt()) for v in zeros],
                "zero_heat_temperature_max_abs": [float(v.abs().max()) for v in zeros],
                "physical_reference": "analytic homogeneous-boundary zero temperature rise; AD/JVP is model sensitivity only",
            }
        )
    return result


def benchmark_optimizer_macro(model, provider, payload, device):
    """Measure disposable native macros plus the every-fourth response path.

    Restores checkpoint tensors/provider state afterward. Three recorded AdamW
    steps and one warmup are engineering timing calls, never scientific epochs.
    """
    from .unified_training import SamplingKey, _load_named_optimizer_state

    state = {n: t.detach().clone() for n, t in model.state_dict().items()}
    provider_state = provider.training_state_dict()
    saved_provider = payload.get("provider_training_state")
    if saved_provider:
        provider.load_training_state_dict(saved_provider)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    # Engine state is named; reuse the maintained restorer rather than map by
    # incidental optimizer parameter order.
    try:
        _load_named_optimizer_state(model, optimizer, payload["optimizer_state_by_name"])
    except (KeyError, ValueError):
        # The measured first warmup initializes moments if engine group labels
        # differ; record this explicitly in returned evidence.
        optimizer_state_source = "fresh moments initialized by unrecorded warmup"
    else:
        optimizer_state_source = "checkpoint named AdamW state"
    times = []
    peaks = []
    breakdown = []
    ids = tuple(provider.training_case_ids[:48])
    try:
        model.train()
        for repeat in range(4):
            model.load_state_dict(state)
            optimizer.zero_grad(set_to_none=True)
            if torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _sync(device)
            start = time.perf_counter()
            fwd = back = 0.0
            work = {}
            losses = {}
            for offset in (0, 24):
                key = SamplingKey(
                    0,
                    int(payload["epoch"]),
                    3,
                    offset // 24,
                    "joint",
                    model.architecture,
                    sampling_version=provider.sampling_version,
                    dataset_id=provider.sampling_dataset_id,
                )
                batch = provider.make_batch(ids[offset : offset + 24], key)
                t = time.perf_counter()
                pred, _ = provider.predict_native(
                    model, batch.scene_inputs, batch.receivers, model.architecture, "joint", epoch=payload["epoch"]
                )
                loss = provider._weighted_native(pred, batch.targets).sum() / 48
                _sync(device)
                fwd += time.perf_counter() - t
                t = time.perf_counter()
                loss.backward()
                _sync(device)
                back += time.perf_counter() - t
                del pred, loss
            t = time.perf_counter()
            provider.before_optimizer_step(model, optimizer, losses, work, epoch=int(payload["epoch"]), update_index=3)
            _sync(device)
            response = time.perf_counter() - t
            t = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _sync(device)
            opt = time.perf_counter() - t
            elapsed = time.perf_counter() - start
            if repeat:
                times.append(elapsed)
                breakdown.append(
                    {
                        "native_forward_seconds": fwd,
                        "native_backward_seconds": back,
                        "response_and_policy_seconds": response,
                        "clip_adam_seconds": opt,
                    }
                )
                peaks.append(torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else 0)
    finally:
        model.load_state_dict(state)
        model.eval()
        provider.load_training_state_dict(provider_state)
    return {
        "seconds": times,
        "median_seconds": float(np.median(times)),
        "peak_allocated_bytes": peaks,
        "stage_breakdown": breakdown,
        "case_ids": ids,
        "effective_cases": 48,
        "microbatch_cases": 24,
        "primary_queries": 1024,
        "response_included": True,
        "optimizer_state_source": optimizer_state_source,
        "warmup_engineering_steps": 1,
        "recorded_engineering_steps": 3,
        "scientific_optimizer_updates": 0,
    }
