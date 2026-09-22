"""ThermalChannel physical coupling for non-legacy interface-field backends."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from honf_forward_core.config import BatchData
from honf_forward_core.interface_fields import PreparedInterfaceField
from .routing_geometry import ChannelThermalRoutingGeometry
from .local_coupling import (
    build_local_module_params_from_global,
    teacher_port_tokens_from_interface_condition,
)


@dataclass(frozen=True)
class PreparedInterfaceChannelThermalCase:
    """Final new-family state reused for arbitrary global query chunks."""

    architecture: str
    prepared: PreparedInterfaceField
    # Runtime-only P0 controller carried explicitly for Run 1407.
    phase_shared_state: Any = None


def _budget_expected_optional_count(prepared: PreparedInterfaceField) -> torch.Tensor | None:
    """Return the live expected optional-group count from the P0 plan.

    ``PreparedInterfaceField.interaction_aux`` is intentionally a detached
    diagnostic channel.  The budget objective must therefore be recovered
    from the runtime gate-plan object carried in ``phase_shared_state``.  The
    core backend exposes the plan as ``CaseGroupBudget``; keeping this lookup
    narrow makes an omitted live objective fail loudly at the new-mode
    boundary instead of falling back to detached statistics.
    """

    plan = getattr(prepared, "phase_shared_state", None)
    if plan is None:
        return None
    value = getattr(plan, "expected_optional_count", None)
    return value if torch.is_tensor(value) else None


@contextmanager
def _interface_read_role(model: Any, role: str):
    """Annotate one physical preparation or read for diagnostic hooks.

    The marker is temporary so historical ``core.read`` and
    ``core.decode_queries`` callers keep their existing signatures.  A
    backend hook may inspect ``model.core._interface_read_role`` while the
    read is executing; ordinary model execution ignores it.
    """

    core = model.core
    marker = "_interface_read_role"
    previous = getattr(core, marker, None)
    setattr(core, marker, str(role))
    try:
        yield
    finally:
        if previous is None:
            try:
                delattr(core, marker)
            except AttributeError:
                pass
        else:
            setattr(core, marker, previous)


def _port_coordinates(model: Any, module_centers: torch.Tensor, ntheta: int) -> torch.Tensor:
    fixed = model.local_coupling.port_head.fixed_theta_tokens(ntheta, module_centers.device, module_centers.dtype)
    normals = fixed[:, 1:3]
    return module_centers[:, :, None, :] + float(model.config.core_honf.module_radius) * normals[None, None, :, :]


def _outside_coordinates(model: Any, port_tokens: torch.Tensor, module_centers: torch.Tensor) -> torch.Tensor:
    radius = float(model.config.core_honf.module_radius) + float(
        model.config.channelthermal.port_global_consistency_radius_offset
    )
    outside = module_centers[:, :, None, :] + radius * port_tokens[..., 1:3]
    return torch.stack(
        [
            outside[..., 0].clamp(0.0, float(model.config.core_honf.domain_length_x)),
            outside[..., 1].clamp(0.0, float(model.config.core_honf.domain_length_y)),
        ],
        dim=-1,
    )


def _decode_temperature(
    model: Any,
    prepared: PreparedInterfaceField,
    coordinates: torch.Tensor,
    *,
    read_role: str = "p1_refinement",
    return_routing_maps: bool = False,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    batch = int(coordinates.shape[0])
    leading = coordinates.shape[1:-1]
    flat = coordinates.reshape(batch, -1, coordinates.shape[-1])
    with _interface_read_role(model, read_role):
        decoded = model.core.decode_queries(
            prepared,
            flat,
            query_features=model._query_features(flat),
            return_routing_maps=bool(return_routing_maps),
            return_interaction_aux=(
                model.config.core_honf.forward_architecture == "routed_pairwise_honf"
                or bool(getattr(model.core.backend, "phase_interaction_diagnostics", False))
            ),
        )
    if "_interaction_aux" in decoded:
        decoded.update(decoded.pop("_interaction_aux"))
    temperature = model._temperature_from_field_output(decoded["pred_field"]).reshape(batch, *leading)
    return temperature, decoded


def _read_port_context(
    model: Any,
    prepared: PreparedInterfaceField,
    port_xy: torch.Tensor,
    *,
    return_routing_maps: bool = False,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    batch, modules, ports, dimension = port_xy.shape
    with _interface_read_role(model, "p0_port"):
        read = model.core.read(
            prepared,
            port_xy.reshape(batch, modules * ports, dimension),
            return_routing_maps=bool(return_routing_maps),
        )
    return read.context.reshape(batch, modules, ports, -1), read.interaction_aux


def _mask_port_paircost_components(
    diagnostics: dict[str, torch.Tensor],
    module_present: torch.Tensor,
    ports: int,
) -> dict[str, torch.Tensor]:
    """Exclude padded module-port receivers from live Eq. (18) components.

    Routed reads retain one pair-cost numerator and denominator per receiver,
    including padded module slots in mixed-module batches.  P0, P1, and the
    P2 port-global probe are physical port reads, so their receiver axis has
    the repeated ``module_present`` mask.  Scalar components from historical
    backends are left untouched for compatibility.
    """

    if not diagnostics or int(ports) <= 0:
        return diagnostics
    if module_present.ndim != 2:
        raise ValueError("module_present must have shape [B,M] for port pair-cost masking.")
    batch, modules = (int(module_present.shape[0]), int(module_present.shape[1]))
    receiver_mask = module_present.unsqueeze(-1).expand(batch, modules, int(ports)).reshape(
        batch, modules * int(ports)
    )
    masked = dict(diagnostics)
    for key, value in diagnostics.items():
        if (
            not torch.is_tensor(value)
            or not str(key).endswith(("routing_paircost_numerator", "routing_paircost_denominator"))
            or value.ndim < 2
            or int(value.shape[0]) != batch
        ):
            continue
        if int(value.shape[1]) == int(receiver_mask.shape[1]):
            mask = receiver_mask.to(device=value.device, dtype=value.dtype)
            masked[key] = value * mask.reshape(batch, int(value.shape[1]), *([1] * (value.ndim - 2)))
        elif int(value.shape[1]) == modules:
            mask = module_present.to(device=value.device, dtype=value.dtype)
            masked[key] = value * mask.reshape(batch, modules, *([1] * (value.ndim - 2)))
    return masked


def _phase_preparation_diagnostics(
    model: Any,
    prepared: PreparedInterfaceField | None,
) -> dict[str, torch.Tensor]:
    """Return opt-in backend preparation counters for one physical phase."""

    if prepared is None:
        return {}
    phase_prefix = getattr(model.core.backend, "phase_diagnostic_prefix", None)
    if not phase_prefix:
        return {}
    if isinstance(phase_prefix, str):
        prefixes = (phase_prefix,)
    else:
        prefixes = tuple(str(prefix) for prefix in phase_prefix)
    # Run 1409 retains the Run-1406 group-control preparation summaries while
    # adding case-gate diagnostics.  Keep both families available in the
    # existing opt-in interaction channel so phase-refresh audits can compare
    # memberships/controls with the gate plan.  ``preparation_aux`` only
    # includes full maps when routing maps were explicitly requested, so this
    # does not add large tensors to ordinary training forwards.
    architecture = str(
        getattr(getattr(getattr(model, "config", None), "core_honf", None), "forward_architecture", "")
    )
    if architecture == "budgeted_group_control_honf":
        prefixes = tuple(dict.fromkeys((*prefixes, "group_control_")))
    return {
        key: value
        for key, value in prepared.interaction_aux.items()
        if any(key.startswith(prefix) for prefix in prefixes) and torch.is_tensor(value)
    }


def forward_interface_field(
    model: Any,
    *,
    structure: Optional[Dict[str, torch.Tensor]],
    query_xy: torch.Tensor,
    re: Optional[torch.Tensor],
    u_in: Optional[torch.Tensor],
    module_centers: Optional[torch.Tensor],
    heat_powers: Optional[torch.Tensor],
    module_present: Optional[torch.Tensor],
    material_params: Optional[torch.Tensor],
    interface_condition: Optional[torch.Tensor],
    local_module_params: Optional[torch.Tensor],
    teacher_port_tokens: Optional[torch.Tensor],
    local_query_points: Optional[torch.Tensor],
    local_port_condition_mode: str,
    mixed_teacher_ratio: float,
    return_predicted_port_outputs: bool,
    return_routing_maps: bool,
    return_edge_fields: bool,
    return_port_global_consistency: bool,
    return_prepared_state: bool,
    return_organizer_passes: bool,
    return_organizer_diagnostics: bool,
    case_edge_selection_mode: Optional[str],
    case_edge_probe_relative_rms_tolerance: Optional[float],
    case_edge_probe_channel_tolerance: Optional[float],
) -> Dict[str, Any]:
    """Run autonomous ports, frozen local physics, one refinement, and final read."""

    del return_organizer_diagnostics, case_edge_probe_relative_rms_tolerance, case_edge_probe_channel_tolerance
    if return_edge_fields:
        raise ValueError("Per-edge fields are not defined for interface-field baselines.")
    if case_edge_selection_mode not in {None, "none"}:
        raise ValueError("Case-edge selection is owned by legacy_honf and is not applicable here.")
    domain_length_x = None
    domain_length_y = None
    if structure is not None:
        re = structure.get("re", re)
        u_in = structure.get("u_in", u_in)
        module_centers = structure.get("module_centers", module_centers)
        heat_powers = structure.get("heat_powers", heat_powers)
        module_present = structure.get("module_present", module_present)
        material_params = structure.get("material_params", material_params)
        domain_length_x = structure.get("domain_length_x")
        domain_length_y = structure.get("domain_length_y")
    if module_centers is None or heat_powers is None or module_present is None:
        raise ValueError("module_centers, heat_powers, and module_present are required.")
    device, dtype = query_xy.device, query_xy.dtype
    batch = int(query_xy.shape[0])
    re = query_xy.new_zeros(batch, 1) if re is None else re
    u_in = query_xy.new_zeros(batch, 1) if u_in is None else u_in
    if material_params is None:
        material_params = query_xy.new_zeros(batch, int(model.config.channelthermal.material_param_dim))
    adapter = model.input_adapter(
        re=re.to(device=device, dtype=dtype),
        u_in=u_in.to(device=device, dtype=dtype),
        module_centers=module_centers.to(device=device, dtype=dtype),
        heat_powers=heat_powers.to(device=device, dtype=dtype) * float(model.config.channelthermal.heat_scale),
        module_present=module_present.to(device=device, dtype=dtype),
        material_params=material_params.to(device=device, dtype=dtype),
        domain_length_x=None if domain_length_x is None else domain_length_x.to(device=device, dtype=dtype),
        domain_length_y=None if domain_length_y is None else domain_length_y.to(device=device, dtype=dtype),
    )
    environment_kwargs = {
        "batch_size": batch,
        "num_env_tokens_x": int(model.config.core_honf.num_env_tokens_x),
        "num_env_tokens_y": int(model.config.core_honf.num_env_tokens_y),
        "domain_length_x": float(model.config.core_honf.domain_length_x),
        "domain_length_y": float(model.config.core_honf.domain_length_y),
        "device": device,
        "dtype": dtype,
    }
    architecture = str(model.config.core_honf.forward_architecture)
    phase_shared_architecture = architecture in {
        "phase_shared_group_control_honf",
        "hypergraph_quadrature_honf",
        # Run 1409 shares only the sampled case-level availability plan.
        # The backend must rebuild memberships, collective controls, and fine
        # values for each later physical phase.
        "budgeted_group_control_honf",
    }
    if architecture in {"regional_response_honf", "hierarchical_regional_honf"}:
        environment_kwargs["response_region_block_shape"] = tuple(
            model.config.core_honf.interface_model.response_region_block_shape
        )
    if architecture == "hierarchical_regional_honf":
        environment_kwargs["response_tree_block_shape"] = tuple(
            model.config.core_honf.interface_model.response_region_block_shape
        )
    # Only regional families request geometric metadata from the adapter.
    env = model.environment_builder(**environment_kwargs)
    batch_kwargs: dict[str, Any] = {}
    if architecture == "hypergraph_quadrature_honf":
        # The regular-grid measure/layout are opt-in metadata. Historical
        # readers intentionally retain their existing dense fallback path and
        # checkpoint-facing behavior even though the adapter can provide the
        # mathematically equivalent cell masses.
        batch_kwargs.update(
            env_weights=getattr(env, "env_weights", None),
            sampler_layout=getattr(env, "sampler_layout", None),
        )
    encoded = model.core.encode_case(
        BatchData(
            module_centers=adapter.module_centers,
            module_present=adapter.module_present,
            module_features=adapter.module_features,
            global_context=adapter.global_context,
            query_xy=query_xy.float(),
            query_time=None,
            target_field=None,
            case_name="channelthermal",
            metadata={},
            env_coords=env.env_coords,
            env_features=env.env_features,
            env_region_ids=getattr(env, "env_region_ids", None),
            env_hierarchy=getattr(env, "env_hierarchy", None),
            **batch_kwargs,
            routing_geometry=(
                ChannelThermalRoutingGeometry(
                    model.config.core_honf.domain_length_x,
                    model.config.core_honf.domain_length_y,
                    model.config.core_honf.module_radius,
                ) if architecture == "routed_pairwise_honf" else None
            ),
        )
    )
    if teacher_port_tokens is None and interface_condition is not None:
        teacher_port_tokens = teacher_port_tokens_from_interface_condition(interface_condition.float())
    ntheta = model._infer_ntheta(interface_condition, teacher_port_tokens)
    physical_port_xy = _port_coordinates(model, adapter.module_centers, ntheta)
    # Sparse HONF constructs the occupied support table from the actual port
    # footprint once.  The same case-local cache (including environment work)
    # is reused while group states refresh after each local response.
    layout_cache = model.core.build_layout(encoded, physical_port_xy)
    base_module_state = encoded.module_tokens
    with _interface_read_role(model, "p0_port"):
        prepared0 = model.core.prepare(
            encoded,
            base_module_state,
            layout_cache=layout_cache,
            return_routing_maps=bool(return_routing_maps),
        )
    initial_port_context, initial_read_aux = _read_port_context(
        model,
        prepared0,
        physical_port_xy,
        return_routing_maps=bool(return_routing_maps),
    )
    initial_read_aux = _mask_port_paircost_components(
        initial_read_aux,
        adapter.module_present,
        ntheta,
    )
    pred_port_tokens = model.local_coupling.port_head(
        base_module_state,
        initial_port_context,
        adapter.heat_powers,
        encoded.global_token,
        ntheta=ntheta,
        module_present=adapter.module_present,
    )

    use_local_outputs = model._should_use_local_outputs(str(model.config.channelthermal.internal_prediction_mode))
    module_state = base_module_state
    local_ports_used = pred_port_tokens
    final_pred_port_tokens = pred_port_tokens
    local_outputs = None
    local_response_summary = None
    interface_diagnostics: Dict[str, torch.Tensor] = {}
    predicted_port_diagnostics: Dict[str, torch.Tensor] = {}
    prepared1 = None
    provisional_read_aux: Dict[str, torch.Tensor] = {}
    if use_local_outputs:
        if local_module_params is None:
            local_module_params = build_local_module_params_from_global(
                adapter.heat_powers,
                interface_condition.float() if interface_condition is not None else None,
                material_params.to(device=device, dtype=dtype),
                adapter.module_present,
            )
        local_ports_used = model.local_coupling.choose_local_ports(
            pred_port_tokens=pred_port_tokens,
            teacher_port_tokens=teacher_port_tokens,
            mode=local_port_condition_mode,
            mixed_teacher_ratio=float(mixed_teacher_ratio),
        )
        local_params_used = model.local_coupling.local_module_params_for_ports(
            local_module_params.to(device=device, dtype=dtype), local_ports_used, adapter.module_present
        )
        local_outputs = model.local_coupling.call_local_surrogate(
            local_params_used,
            local_ports_used,
            local_query_points.to(device=device, dtype=dtype) if torch.is_tensor(local_query_points) else None,
            adapter.module_present,
        )
        pred_interface, interface_diagnostics = model.local_coupling.assemble_interface(
            local_outputs=local_outputs,
            local_ports=local_ports_used,
            module_state=base_module_state,
            module_present=adapter.module_present,
        )
        local_response_summary = model.local_coupling.local_response_summary(
            local_outputs=local_outputs,
            module_present=adapter.module_present,
            interface_override=pred_interface,
        )
        module_state = model.local_coupling.fuse_module_state(
            base_module_state, local_outputs, local_response_summary, adapter.module_present
        )
        if int(model.config.channelthermal.interaction_refinement_steps) == 1 and (
            str(local_port_condition_mode).lower() != "teacher" or teacher_port_tokens is None
        ):
            with _interface_read_role(model, "p1_refinement"):
                if phase_shared_architecture:
                    prepared1 = model.core.prepare(
                        encoded,
                        module_state,
                        layout_cache=layout_cache,
                        return_routing_maps=bool(return_routing_maps),
                        phase_shared_state=prepared0.phase_shared_state,
                    )
                else:
                    prepared1 = model.core.prepare(
                        encoded,
                        module_state,
                        layout_cache=layout_cache,
                        return_routing_maps=bool(return_routing_maps),
                    )
            outside_temperature, provisional_decode = _decode_temperature(
                model,
                prepared1,
                _outside_coordinates(model, local_ports_used, adapter.module_centers),
                read_role="p1_refinement",
                return_routing_maps=bool(return_routing_maps),
            )
            provisional_read_aux = {
                key: value
                for key, value in provisional_decode.items()
                if key != "pred_field" and torch.is_tensor(value)
            }
            provisional_read_aux = _mask_port_paircost_components(
                provisional_read_aux,
                adapter.module_present,
                int(local_ports_used.shape[-2]),
            )
            refined_ports = model.local_coupling.port_refinement_head(
                module_state,
                local_ports_used,
                outside_temperature,
                local_response_summary,
                adapter.module_present,
            )
            local_ports_used = refined_ports
            if str(local_port_condition_mode).lower() == "predicted" or teacher_port_tokens is None:
                final_pred_port_tokens = refined_ports
            local_params_used = model.local_coupling.local_module_params_for_ports(
                local_module_params.to(device=device, dtype=dtype), local_ports_used, adapter.module_present
            )
            local_outputs = model.local_coupling.call_local_surrogate(
                local_params_used,
                local_ports_used,
                local_query_points.to(device=device, dtype=dtype) if torch.is_tensor(local_query_points) else None,
                adapter.module_present,
            )
            pred_interface, refreshed = model.local_coupling.assemble_interface(
                local_outputs=local_outputs,
                local_ports=local_ports_used,
                module_state=module_state,
                module_present=adapter.module_present,
            )
            interface_diagnostics.update(refreshed)
            local_response_summary = model.local_coupling.local_response_summary(
                local_outputs=local_outputs,
                module_present=adapter.module_present,
                interface_override=pred_interface,
            )
            module_state = model.local_coupling.fuse_module_state(
                base_module_state, local_outputs, local_response_summary, adapter.module_present
            )
        if return_predicted_port_outputs and str(local_port_condition_mode).lower() != "predicted":
            predicted_params = model.local_coupling.local_module_params_for_ports(
                local_module_params.to(device=device, dtype=dtype), final_pred_port_tokens, adapter.module_present
            )
            predicted_outputs = model.local_coupling.call_local_surrogate(
                predicted_params,
                final_pred_port_tokens,
                local_query_points.to(device=device, dtype=dtype) if torch.is_tensor(local_query_points) else None,
                adapter.module_present,
            )
            predicted_interface, _ = model.local_coupling.assemble_interface(
                local_outputs=predicted_outputs,
                local_ports=final_pred_port_tokens,
                module_state=base_module_state,
                module_present=adapter.module_present,
            )
            predicted_port_diagnostics = {
                "predicted_port_internal_temperature": predicted_outputs["internal_temperature"],
                "predicted_port_interface": predicted_interface,
            }

    with _interface_read_role(model, "p2_field"):
        final_prepared = (
            prepared0
            if local_outputs is None
            else (
                model.core.prepare(
                    encoded,
                    module_state,
                    layout_cache=layout_cache,
                    return_routing_maps=bool(return_routing_maps),
                    phase_shared_state=prepared0.phase_shared_state,
                )
                if phase_shared_architecture
                else model.core.prepare(
                    encoded,
                    module_state,
                    layout_cache=layout_cache,
                    return_routing_maps=bool(return_routing_maps),
                )
            )
        )
    with _interface_read_role(model, "p2_field"):
        decoder_output = model.core.decode_queries(
            final_prepared,
            query_xy.float(),
            query_features=model._query_features(query_xy.float()),
            return_routing_maps=return_routing_maps,
            return_interaction_aux=True,
        )
    interaction_aux: Dict[str, Any] = dict(final_prepared.interaction_aux)
    interaction_aux.update(decoder_output.pop("_interaction_aux"))
    if architecture == "budgeted_group_control_honf":
        expected_optional_count = _budget_expected_optional_count(prepared0)
        if expected_optional_count is None:
            raise RuntimeError(
                "budgeted_group_control_honf did not expose a live expected optional-group "
                "count on its P0 gate plan. The case objective cannot use detached diagnostics."
            )
    else:
        expected_optional_count = None
    for key, value in _phase_preparation_diagnostics(model, prepared0).items():
        interaction_aux[f"initial_port_{key}"] = value
    for key, value in initial_read_aux.items():
        if (not key.startswith(("hierarchical_incidence_", "routing_module_pair_", "routing_environment_pair_"))
                and torch.is_tensor(value) and value.ndim >= 2
                and value.shape[1] == physical_port_xy.shape[1] * ntheta):
            interaction_aux[f"initial_port_{key}"] = value.reshape(
                value.shape[0], physical_port_xy.shape[1], ntheta, *value.shape[2:]
            )
        else:
            interaction_aux[f"initial_port_{key}"] = value

    if architecture == "routed_pairwise_honf":
        for key, value in provisional_read_aux.items():
            if key.startswith("routing_"):
                interaction_aux[f"provisional_{key}"] = value
    else:
        phase_prefix = getattr(model.core.backend, "phase_diagnostic_prefix", None)
        if isinstance(phase_prefix, str):
            phase_prefixes = (phase_prefix,)
        else:
            phase_prefixes = tuple(str(prefix) for prefix in phase_prefix or ())
        if architecture == "budgeted_group_control_honf":
            phase_prefixes = tuple(dict.fromkeys((*phase_prefixes, "group_control_")))
        for key, value in _phase_preparation_diagnostics(model, prepared1).items():
            interaction_aux[f"provisional_{key}"] = value
        for key, value in provisional_read_aux.items():
            if any(key.startswith(prefix) for prefix in phase_prefixes):
                interaction_aux[f"provisional_{key}"] = value

    if local_outputs is not None:
        pred_internal = local_outputs["internal_temperature"]
        module_response_latent = local_outputs["module_response_latent"]
        interface_source = "local_surrogate"
    else:
        pred_internal = model.fallback_heads.predict_internal(module_state, local_query_points, adapter.module_present)
        pred_interface = model.fallback_heads.predict_interface(
            module_state, ntheta=ntheta, module_present=adapter.module_present
        )
        module_response_latent = module_state
        interface_source = "global_head"

    if return_port_global_consistency:
        indices = model._port_subset_indices(ntheta, device)
        selected = final_pred_port_tokens.index_select(-2, indices)
        temperature, consistency_diag = _decode_temperature(
            model,
            final_prepared,
            _outside_coordinates(model, selected, adapter.module_centers),
            read_role="p2_port_global_consistency",
            return_routing_maps=bool(return_routing_maps),
        )
        consistency_diag = _mask_port_paircost_components(
            consistency_diag,
            adapter.module_present,
            int(indices.numel()),
        )
        # The p2 port-global consistency probe is an actual routed QE read.
        # Preserve its live Eq. (18) components separately from the final
        # field read so the case loss counts every physical read exactly once.
        # When pair-cost is disabled the backend emits no such components, so
        # historical outputs and their auxiliary schemas remain unchanged.
        if architecture == "routed_pairwise_honf":
            for key, value in consistency_diag.items():
                if (
                    key.endswith(("routing_paircost_numerator", "routing_paircost_denominator"))
                    and torch.is_tensor(value)
                ):
                    interaction_aux["port_global_" + key] = value
        else:
            phase_prefix = getattr(model.core.backend, "phase_diagnostic_prefix", None)
            if isinstance(phase_prefix, str):
                phase_prefixes = (phase_prefix,)
            else:
                phase_prefixes = tuple(str(prefix) for prefix in phase_prefix or ())
            if architecture == "budgeted_group_control_honf":
                phase_prefixes = tuple(dict.fromkeys((*phase_prefixes, "group_control_")))
            for key, value in consistency_diag.items():
                if any(key.startswith(prefix) for prefix in phase_prefixes) and torch.is_tensor(value):
                    interaction_aux["port_global_" + key] = value
        target_temperature = selected[..., 3]
        consistency_mask = adapter.module_present[:, :, None].expand_as(temperature)
        port_global_temperature = temperature * consistency_mask
        port_global_target = target_temperature * consistency_mask
    else:
        port_global_temperature = query_xy.new_empty(batch, adapter.module_centers.shape[1], 0)
        port_global_target = query_xy.new_empty(batch, adapter.module_centers.shape[1], 0)
        consistency_mask = query_xy.new_empty(batch, adapter.module_centers.shape[1], 0)
        consistency_diag = {}

    result: Dict[str, Any] = {
        "pred_field": decoder_output["pred_field"],
        "pred_internal_temperature": pred_internal,
        "pred_interface": pred_interface,
        "pred_port_condition": final_pred_port_tokens,
        "pred_port_condition_raw": pred_port_tokens,
        "local_port_condition_used": local_ports_used,
        "pred_port_global_temperature": port_global_temperature,
        "pred_port_global_temperature_target": port_global_target,
        "pred_port_global_consistency_mask": consistency_mask,
        "interface_source": interface_source,
        "pred_interface_source": interface_source,
        "module_response_latent": module_response_latent,
        "organizer_aux": {},
        "base_organizer_aux": {},
        "routing_aux": {key: value for key, value in decoder_output.items() if key != "pred_field"},
        "interaction_aux": interaction_aux,
    }
    if expected_optional_count is not None:
        # This remains attached to the P0 gate graph and is consumed once by
        # the ChannelThermal loss assembly.  It is deliberately separate from
        # detached interaction diagnostics used for logging.
        result["case_group_budget_expected_optional_count"] = expected_optional_count
    if return_organizer_passes:
        result["provisional_organizer_aux"] = {}
        result["provisional_interaction_aux"] = {} if prepared1 is None else prepared1.interaction_aux
        result["provisional_read_aux"] = provisional_read_aux
    result.update(interface_diagnostics)
    result.update(predicted_port_diagnostics)
    if consistency_diag:
        result["routing_aux"]["port_global_interaction_backend"] = decoder_output["pred_field"].new_ones(())
    if return_prepared_state:
        result["prepared_state"] = PreparedInterfaceChannelThermalCase(
            architecture=str(model.config.core_honf.forward_architecture),
            prepared=final_prepared,
            phase_shared_state=final_prepared.phase_shared_state,
        )
    return result
