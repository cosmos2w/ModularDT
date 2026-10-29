"""Round2 trace adapters for the maintained native evaluation/inspection tools."""

from __future__ import annotations

import numpy as np
import torch

CONDITIONS = ("original_full_h6", "uniform_receiver_access", "input_declared_edge_removed", "local_branch_removed")
TRACE_NAMES = ("edge_access", "edge_base_access", "edge_used_access", "edge_messages",
               "edge_temperature_contributions", "local_pair_batch_indices", "local_pair_query_indices",
               "local_pair_module_indices", "local_pair_support_weights", "local_temperature_coefficients")


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def organization_summary(core, organization, query, *, receiver_limit=128):
    """Small input-index panel; no target-dependent receiver or edge selection."""
    from content_hypergraph_inspect import slice_query
    indices = np.unique(np.linspace(0, query.num_queries - 1, min(receiver_limit, query.num_queries), dtype=np.int64))
    selected = slice_query(query, indices)
    output = core.decode(organization, selected, return_trace=True,
                         trace_request={"include": ["edge_access"], "max_elements": 500_000})
    module = organization.module_gather_weights.detach().double()
    environment = organization.environment_gather_weights.detach().double()

    def concentration(weights):
        entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(-1)
        return {"entropy": entropy.cpu().tolist(), "effective_donors": entropy.exp().cpu().tolist(),
                "maximum_donor_weight": weights.max(-1).values.cpu().tolist()}

    edges = torch.nn.functional.normalize(organization.edges.detach().double(), dim=-1)
    result = {"schema": "content_hypergraph_organization_summary_v2",
              "module_W": concentration(module), "environment_W": concentration(environment),
              "content_cosine_similarity": (edges @ edges.transpose(-1, -2)).cpu().tolist(),
              "head_access_mean": output.trace["edge_access"].mean(1).detach().cpu().tolist(),
              "receiver_selection": "evenly spaced input-native row indices; no target/error selection",
              "receiver_indices": indices.tolist(), "requested_receiver_count": query.num_queries,
              "measured_receiver_count": len(indices), "work_counts": output.work_counts,
              "module_ids": organization.scene.module_ids.detach().cpu().tolist()}
    if organization.source_coefficients is not None:
        c = organization.source_coefficients.detach().double()
        result["C_edge_squared_energy"] = c.square().sum((-1, -2)).cpu().tolist()
    return result


def run_case(model, provider, batch, dataset):
    from content_hypergraph_evaluate import json_safe
    from content_hypergraph_inspect import (_save_scene_and_organization, select_thermal_probes,
                                           select_wind_probes, slice_query)
    core = model.core if dataset == "thermal" else model
    if dataset == "thermal":
        prepared = model.prepare_native(batch.scene_inputs.structure, batch.receivers.fluid_xy,
                                        local_query_points=batch.receivers.local_query_points,
                                        ntheta=batch.receivers.ntheta, scene_key=batch.scene_inputs.case_ids)
        scene, query = prepared.scene_batch, prepared.query_batch
        organization = core.prepare(scene, return_trace=True, trace_request={"include": [
            "module_gather_weights", "environment_gather_weights", "module_weights_first",
            "environment_weights_first", "edge_edge_attention", "source_coefficients" ]})
        prepared.organization = organization
        active = _array(scene.module_mask[0]).astype(bool)
        selected = select_thermal_probes(_array(batch.receivers.fluid_xy[0]),
            np.arange(batch.receivers.fluid_xy.shape[1]), ~_array(prepared.native_solid_mask[0]).reshape(-1).astype(bool),
            _array(scene.module_coordinates[0])[active], _array(scene.local_radius[0])[active] / 3,
            _array(scene.local_radius[0])[active])
        observation_indices = np.array(list(selected.values()), dtype=np.int64)
        ids = {int(value): index for index, value in enumerate(_array(query.query_ids[0]))}
        probe_indices = np.array([ids[int(x)] for x in observation_indices])
        controls = batch.receivers.heat
    else:
        scene = provider.make_scene(batch.scene_inputs)
        query = provider.adapter.make_query_batch(scene, batch.receivers)
        organization = core.prepare(scene, return_trace=True, trace_request={"include": [
            "module_gather_weights", "environment_gather_weights", "module_weights_first",
            "environment_weights_first", "edge_edge_attention" ]})
        inputs = batch.scene_inputs[0]
        active = np.asarray(inputs.module_present) > .5
        selected = select_wind_probes(batch.receivers.coordinates_D[0], batch.receivers.flat_indices[0],
            batch.receivers.role_ids[0], inputs.module_centers_D[active], inputs.module_features[active, 0])
        probe_indices = np.array(list(selected.values()), dtype=np.int64)
        controls = None
    # Edge0 is declared before any model output or target is consulted.
    removed_edge = 0
    arrays = {}
    _save_scene_and_organization(arrays, scene, organization)
    if dataset == "thermal":
        arrays["native__solid_mask"] = _array(prepared.native_solid_mask)
    arrays.update(query__coordinates=_array(query.coordinates), query__native_ids=_array(query.query_ids),
                  probe__core_query_indices=probe_indices, probe__coordinates=_array(query.coordinates[0, probe_indices]))
    results, baseline, baseline_access = {}, None, None
    for condition in CONDITIONS:
        edge_gates = torch.ones((scene.batch_size, 6), device=query.coordinates.device, dtype=query.coordinates.dtype)
        if condition == "input_declared_edge_removed":
            edge_gates[:, removed_edge] = 0
        local_gates = torch.zeros_like(scene.module_measure) if condition == "local_branch_removed" else None
        output = core.decode(organization, query, controls=controls, return_trace=True,
            trace_request={"include": [name for name in TRACE_NAMES if dataset == "thermal" or name not in
                                      {"edge_temperature_contributions", "local_temperature_coefficients"}],
                           "query_indices": probe_indices.tolist(), "max_elements": 500_000},
            uniform_access=condition == "uniform_receiver_access", edge_gates=edge_gates,
            local_gates=local_gates, accumulation_dtype=torch.float64 if dataset == "thermal" else None)
        for name, value in (output.trace or {}).items():
            if torch.is_tensor(value):
                arrays[f"{condition}__trace__{name}"] = _array(value)
        if dataset == "thermal":
            from channelthermal.training.content_hypergraph_task import ThermalContentPredictions
            native = model._extract_native_roles(prepared, output.thermal_temperature, output.nonlinear_field,
                                                 accumulation_dtype=torch.float64)
            prediction = ThermalContentPredictions(prepared, native, work=dict(output.work_counts))
            physical = native["pred_field"]
            for name, value in native.items():
                arrays[f"{condition}__native__{name}"] = _array(value)
            arrays["targets__field"] = _array(batch.targets.field_targets)
            arrays["targets__interface"] = _array(batch.targets.interface_target)
            arrays["targets__material"] = _array(batch.targets.material_targets)
            probes = physical[0, observation_indices]
        else:
            from windfarm.training.content_hypergraph_task import WindContentPredictions
            physical, profile, residual = provider.adapter.physical_velocity(output.nonlinear_field, query.coordinates)
            roles = torch.as_tensor(batch.receivers.role_ids.copy(), device=physical.device, dtype=torch.long)
            prediction = WindContentPredictions(physical, output.nonlinear_field, residual, profile, roles)
            arrays[f"{condition}__physical_velocity_mps"] = _array(physical)
            arrays["targets__velocity_mps"] = _array(batch.targets.velocity_mps)
            probes = physical[0, probe_indices]
        access = output.trace["edge_used_access"]
        if baseline is None:
            baseline = physical.clone()
            baseline_access = access.clone()
        delta = physical - baseline
        result = {"metrics": json_safe(provider.validation_metrics(prediction, batch.targets, {})),
                  "core_work_counts": output.work_counts, "probe_prediction_physical": _array(probes).tolist(),
                  "output_delta_l2": float(torch.linalg.vector_norm(delta)),
                  "output_delta_rms": float(delta.square().mean().sqrt()),
                  "output_delta_by_channel_rms": delta.square().mean((0, 1)).sqrt().detach().cpu().tolist(),
                  "channel_order": ["u", "v", "p", "omega", "temperature"] if dataset == "thermal" else ["Ux", "Uy", "Uz"],
                  "delta_units": "per-channel packed native units" if dataset == "thermal" else "m/s",
                  "local_rows_executed": int(output.work_counts["local_rows"])}
        if condition == "input_declared_edge_removed":
            expected = baseline_access.clone()
            expected[..., removed_edge] = 0
            result["edge_access_frozen_without_renormalization"] = bool(torch.equal(access, expected))
            if not result["edge_access_frozen_without_renormalization"]:
                raise AssertionError("Edge intervention changed access beyond the declared multiplicative gate.")
        if condition == "local_branch_removed":
            result["edge_access_preserved"] = bool(torch.equal(access, baseline_access))
            if not result["edge_access_preserved"]:
                raise AssertionError("Local intervention changed edge access.")
        results[condition] = result
    if dataset == "thermal":
        kernels = core.export_source_kernels(organization, slice_query(query, probe_indices), accumulation_dtype=torch.float64)
        arrays["probe__source_kernel_K_physical_temperature"] = _array(kernels * model.field_std[4])
    return {"case_key": str(batch.case_keys[0]), "conditions": results,
            "edge_selection": {"rule": "fixed edge0, declared independently of predictions/targets", "edge": removed_edge},
            "trace_axes": {"W_module": "B,H,M", "W_environment": "B,H,E", "head_access": "B,Qselected,4,H",
                           "edge_messages": "B,Qselected,H,128", "C": "B,H,4,M"},
            "probes": {name: int(index) for name, index in selected.items()},
            "summary": organization_summary(core, organization, query), "arrays": arrays}
