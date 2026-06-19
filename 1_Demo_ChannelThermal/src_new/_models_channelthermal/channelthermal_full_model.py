"""CHANNELTHERMAL-SPECIFIC HONF full-model wrapper.

Inputs use the legacy ChannelThermal global forward signature: a `structure`
dictionary or equivalent keyword tensors plus query coordinates, teacher port
conditions, local module parameters, and local query points. Outputs are a
legacy-compatible dictionary with global field, internal/interface predictions,
selected port tokens, base/final organizer diagnostics, and local-response
latents. The wrapper is ChannelThermal-specific; the underlying HONF organizer
and decoder remain reusable across domains.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from _models_core.honf_core import HONFNeuralField
from _models_core.honf_types import BatchData
from .channelthermal_config import ChannelThermalHONFConfig
from .channelthermal_environment import ChannelThermalEnvironmentBuilder
from .channelthermal_input_adapter import ChannelThermalInputAdapter
from .internal_fallback_heads import FallbackHeadConfig, GlobalFallbackHeads
from .local_coupling import (
    LocalSurrogateCoupling,
    build_local_module_params_from_global,
    teacher_port_tokens_from_interface_condition,
)


class ChannelThermalHONFModel(nn.Module):
    """Legacy-compatible ChannelThermal wrapper around the CORE HONF model."""

    def __init__(self, config: ChannelThermalHONFConfig):
        super().__init__()
        self.config = config
        hidden = int(config.core_honf.hidden_dim)
        self.core = HONFNeuralField(config.core_honf)
        self.input_adapter = ChannelThermalInputAdapter()
        self.environment_builder = ChannelThermalEnvironmentBuilder()
        self.global_normalize_targets = False
        self.global_normalization_stats: Dict[str, Any] = {}
        self.local_coupling = LocalSurrogateCoupling(
            hidden_dim=hidden,
            local_surrogate_latent_dim=int(config.channelthermal.local_surrogate_latent_dim),
            dropout=float(config.core_honf.dropout),
            use_layer_norm=bool(config.core_honf.use_layer_norm),
            local_module_params_from_used_ports=bool(config.channelthermal.local_module_params_from_used_ports),
            local_surrogate_flux_mode=str(config.channelthermal.local_surrogate_flux_mode),
            local_surrogate_flux_blend_alpha=float(config.channelthermal.local_surrogate_flux_blend_alpha),
        )
        self.fallback_heads = GlobalFallbackHeads(
            hidden,
            FallbackHeadConfig(
                hidden_dim=int(config.channelthermal.fallback_hidden_dim),
                internal_query_dim=int(config.channelthermal.fallback_internal_query_dim),
                interface_dim=int(config.channelthermal.fallback_interface_dim),
                fourier_frequencies=int(config.channelthermal.fallback_fourier_frequencies),
                dropout=float(config.core_honf.dropout),
            ),
        )
        path = config.channelthermal.local_surrogate_checkpoint_path
        if bool(config.channelthermal.use_local_surrogate) and path:
            self.local_coupling.attach_from_checkpoint(path, freeze=bool(config.channelthermal.freeze_local_surrogate), map_location="cpu")

    def set_global_target_normalization(self, stats: Optional[Dict[str, Any]], *, normalize_targets: bool) -> None:
        self.global_normalization_stats = dict(stats or {})
        self.global_normalize_targets = bool(normalize_targets)
        self.local_coupling.set_global_target_normalization(stats, normalize_targets=normalize_targets)

    @property
    def local_surrogate_attached(self) -> bool:
        return self.local_coupling.has_local_surrogate

    def forward(
        self,
        structure: Optional[Dict[str, torch.Tensor]] = None,
        query_xy: Optional[torch.Tensor] = None,
        *,
        re: Optional[torch.Tensor] = None,
        u_in: Optional[torch.Tensor] = None,
        module_centers: Optional[torch.Tensor] = None,
        heat_powers: Optional[torch.Tensor] = None,
        module_present: Optional[torch.Tensor] = None,
        material_params: Optional[torch.Tensor] = None,
        interface_condition: Optional[torch.Tensor] = None,
        local_module_params: Optional[torch.Tensor] = None,
        teacher_port_tokens: Optional[torch.Tensor] = None,
        local_query_points: Optional[torch.Tensor] = None,
        local_port_condition_mode: str = "predicted",
        mixed_teacher_ratio: float = 0.5,
        return_predicted_port_outputs: bool = False,
        return_routing_maps: bool = False,
    ) -> Dict[str, Any]:
        if structure is not None:
            re = structure.get("re", re)
            u_in = structure.get("u_in", u_in)
            module_centers = structure.get("module_centers", module_centers)
            heat_powers = structure.get("heat_powers", heat_powers)
            module_present = structure.get("module_present", module_present)
            material_params = structure.get("material_params", material_params)
            domain_length_x = structure.get("domain_length_x")
            domain_length_y = structure.get("domain_length_y")
        else:
            domain_length_x = None
            domain_length_y = None
        if query_xy is None:
            raise ValueError("query_xy is required.")
        if module_centers is None or heat_powers is None or module_present is None:
            raise ValueError("module_centers, heat_powers, and module_present are required.")
        device = query_xy.device
        dtype = query_xy.dtype
        batch = int(query_xy.shape[0])
        if re is None:
            re = query_xy.new_zeros(batch, 1)
        if u_in is None:
            u_in = query_xy.new_zeros(batch, 1)
        if material_params is None:
            material_params = query_xy.new_zeros(batch, int(self.config.channelthermal.material_param_dim))

        adapter = self.input_adapter(
            re=re.to(device=device, dtype=dtype),
            u_in=u_in.to(device=device, dtype=dtype),
            module_centers=module_centers.to(device=device, dtype=dtype),
            heat_powers=heat_powers.to(device=device, dtype=dtype),
            module_present=module_present.to(device=device, dtype=dtype),
            material_params=material_params.to(device=device, dtype=dtype),
            domain_length_x=None if domain_length_x is None else domain_length_x.to(device=device, dtype=dtype),
            domain_length_y=None if domain_length_y is None else domain_length_y.to(device=device, dtype=dtype),
        )
        env = self.environment_builder(
            batch_size=batch,
            num_env_tokens_x=int(self.config.core_honf.num_env_tokens_x),
            num_env_tokens_y=int(self.config.core_honf.num_env_tokens_y),
            domain_length_x=float(self.config.core_honf.domain_length_x),
            domain_length_y=float(self.config.core_honf.domain_length_y),
            device=device,
            dtype=dtype,
        )
        honf_batch = BatchData(
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
        )
        base_output = self.core(honf_batch)
        base_org = self._legacy_organizer_aux(base_output, adapter, env.env_coords)
        base_module_state = base_output["module_tokens"]
        env_state = base_output["env_tokens"]
        global_token = base_output["global_token"]

        if teacher_port_tokens is None and interface_condition is not None:
            teacher_port_tokens = teacher_port_tokens_from_interface_condition(interface_condition.float())
        ntheta = self._infer_ntheta(interface_condition, teacher_port_tokens)
        pred_port_tokens = self.local_coupling.port_head(
            base_module_state,
            base_org["module_env_context"],
            adapter.heat_powers,
            global_token,
            ntheta=ntheta,
            module_present=adapter.module_present,
        )

        mode = str(self.config.channelthermal.internal_prediction_mode)
        use_local_outputs = self._should_use_local_outputs(mode)
        module_state = base_module_state
        local_ports_used = pred_port_tokens
        local_outputs: Optional[Dict[str, torch.Tensor]] = None
        local_response_summary: Optional[torch.Tensor] = None
        interface_diagnostics: Dict[str, torch.Tensor] = {}
        predicted_port_diagnostics: Dict[str, torch.Tensor] = {}
        final_pred_port_tokens = pred_port_tokens
        if use_local_outputs:
            if local_module_params is None:
                local_module_params = build_local_module_params_from_global(
                    adapter.heat_powers,
                    interface_condition.float() if interface_condition is not None else None,
                    material_params.to(device=device, dtype=dtype) if material_params is not None else None,
                    adapter.module_present,
                )
            local_ports_used = self.local_coupling.choose_local_ports(
                pred_port_tokens=pred_port_tokens,
                teacher_port_tokens=teacher_port_tokens,
                mode=local_port_condition_mode,
                mixed_teacher_ratio=float(mixed_teacher_ratio),
            )
            local_module_params_used = self.local_coupling.local_module_params_for_ports(
                local_module_params.to(device=device, dtype=dtype),
                local_ports_used,
                adapter.module_present,
            )
            local_outputs = self.local_coupling.call_local_surrogate(
                local_module_params_used,
                local_ports_used,
                local_query_points.to(device=device, dtype=dtype) if torch.is_tensor(local_query_points) else None,
                adapter.module_present,
            )
            pred_interface, interface_diagnostics = self.local_coupling.assemble_interface(
                local_outputs=local_outputs,
                local_ports=local_ports_used,
                module_state=base_module_state,
                module_present=adapter.module_present,
            )
            local_response_summary = self.local_coupling.local_response_summary(
                local_outputs=local_outputs,
                module_present=adapter.module_present,
                interface_override=pred_interface,
            )
            module_state = self.local_coupling.fuse_module_state(
                base_module_state,
                local_outputs,
                local_response_summary,
                adapter.module_present,
            )
            refinement_steps = int(self.config.channelthermal.interaction_refinement_steps)
            if refinement_steps == 1 and (str(local_port_condition_mode).lower() != "teacher" or teacher_port_tokens is None):
                # ChannelThermal-specific one-way interaction refinement:
                # use provisional local response to decode outside temperatures,
                # refine T_env/h, rerun the local surrogate, then fuse final state.
                provisional_org_raw = self.core.organizer(
                    module_tokens=module_state,
                    env_tokens=env_state,
                    module_centers=adapter.module_centers,
                    env_coords=env.env_coords,
                    module_present=adapter.module_present,
                    geometry_mode=self.config.core_honf.geometry_mode,
                )
                provisional_org_raw["module_features_raw"] = adapter.module_features
                outside_temperature = self._global_temperature_for_all_ports(
                    local_ports_used,
                    module_state,
                    provisional_org_raw,
                    global_token,
                    adapter.module_centers,
                    adapter.module_present,
                )
                refined_ports = self.local_coupling.port_refinement_head(
                    module_state,
                    local_ports_used,
                    outside_temperature,
                    local_response_summary,
                    adapter.module_present,
                )
                local_ports_used = refined_ports
                if str(local_port_condition_mode).lower() == "predicted" or teacher_port_tokens is None:
                    final_pred_port_tokens = refined_ports
                local_module_params_used = self.local_coupling.local_module_params_for_ports(
                    local_module_params.to(device=device, dtype=dtype),
                    local_ports_used,
                    adapter.module_present,
                )
                local_outputs = self.local_coupling.call_local_surrogate(
                    local_module_params_used,
                    local_ports_used,
                    local_query_points.to(device=device, dtype=dtype) if torch.is_tensor(local_query_points) else None,
                    adapter.module_present,
                )
                pred_interface, interface_diagnostics = self.local_coupling.assemble_interface(
                    local_outputs=local_outputs,
                    local_ports=local_ports_used,
                    module_state=module_state,
                    module_present=adapter.module_present,
                )
                local_response_summary = self.local_coupling.local_response_summary(
                    local_outputs=local_outputs,
                    module_present=adapter.module_present,
                    interface_override=pred_interface,
                )
                module_state = self.local_coupling.fuse_module_state(
                    base_module_state,
                    local_outputs,
                    local_response_summary,
                    adapter.module_present,
                )

            if bool(return_predicted_port_outputs) and str(local_port_condition_mode).lower() != "predicted":
                predicted_module_params = self.local_coupling.local_module_params_for_ports(
                    local_module_params.to(device=device, dtype=dtype),
                    final_pred_port_tokens,
                    adapter.module_present,
                )
                predicted_local_outputs = self.local_coupling.call_local_surrogate(
                    predicted_module_params,
                    final_pred_port_tokens,
                    local_query_points.to(device=device, dtype=dtype) if torch.is_tensor(local_query_points) else None,
                    adapter.module_present,
                )
                predicted_interface, _ = self.local_coupling.assemble_interface(
                    local_outputs=predicted_local_outputs,
                    local_ports=final_pred_port_tokens,
                    module_state=base_module_state,
                    module_present=adapter.module_present,
                )
                predicted_port_diagnostics = {
                    "predicted_port_internal_temperature": predicted_local_outputs["internal_temperature"],
                    "predicted_port_interface": predicted_interface,
                }

        final_org_raw = self.core.organizer(
            module_tokens=module_state,
            env_tokens=env_state,
            module_centers=adapter.module_centers,
            env_coords=env.env_coords,
            module_present=adapter.module_present,
            geometry_mode=self.config.core_honf.geometry_mode,
        )
        final_org_raw["module_features_raw"] = adapter.module_features
        decoder_output = self.core.decoder(
            query_xy=query_xy.float(),
            query_time=None,
            organizer_output=final_org_raw,
            global_context=global_token,
            return_routing_maps=bool(return_routing_maps),
        )
        final_output: Dict[str, Any] = {}
        final_output.update(final_org_raw)
        final_output.update(decoder_output)
        org = self._legacy_organizer_aux(final_output, adapter, env.env_coords)

        if local_outputs is not None:
            pred_internal = local_outputs["internal_temperature"]
            module_response_latent = local_outputs["module_response_latent"]
            interface_source = "local_surrogate"
        else:
            pred_internal = self.fallback_heads.predict_internal(module_state, local_query_points, adapter.module_present)
            pred_interface = self.fallback_heads.predict_interface(module_state, ntheta=ntheta, module_present=adapter.module_present)
            module_response_latent = module_state
            interface_source = "global_head"

        port_global_temperature, port_global_t_env, port_global_mask = self._decode_global_temperature_at_ports(
            final_pred_port_tokens,
            module_state,
            org,
            global_token,
            adapter.module_centers,
            adapter.module_present,
        )
        result = {
            "pred_field": decoder_output["pred_field"],
            "pred_internal_temperature": pred_internal,
            "pred_interface": pred_interface,
            "pred_port_condition": final_pred_port_tokens,
            "pred_port_condition_raw": pred_port_tokens,
            "local_port_condition_used": local_ports_used,
            "pred_port_global_temperature": port_global_temperature,
            "pred_port_global_temperature_target": port_global_t_env,
            "pred_port_global_consistency_mask": port_global_mask,
            "interface_source": interface_source,
            "pred_interface_source": interface_source,
            "module_response_latent": module_response_latent,
            "organizer_aux": org,
            "base_organizer_aux": base_org,
            "routing_aux": {key: value for key, value in decoder_output.items() if key != "pred_field"},
        }
        result.update(interface_diagnostics)
        result.update(predicted_port_diagnostics)
        return result

    def _should_use_local_outputs(self, mode: str) -> bool:
        if mode == "global_head":
            return False
        if mode == "local_surrogate":
            if not self.local_coupling.has_local_surrogate:
                raise RuntimeError("internal_prediction_mode='local_surrogate' requires an attached local surrogate.")
            return True
        return bool(self.config.channelthermal.use_local_surrogate and self.local_coupling.has_local_surrogate)

    def _legacy_organizer_aux(
        self,
        core_output: Dict[str, Any],
        adapter: Any,
        env_coords: torch.Tensor,
    ) -> Dict[str, Any]:
        org_keys = {
            "A_me",
            "A_mh",
            "A_eh",
            "hyper_state",
            "hyper_source_coords",
            "hyper_region_coords",
            "hyper_module_mass",
            "hyper_env_mass",
            "hyper_strength",
            "module_env_context",
            "module_centers",
            "module_present",
            "env_coords",
            "module_tokens",
            "env_tokens",
            "module_features_raw",
        }
        org = {key: core_output[key] for key in org_keys if key in core_output}
        org["hyper_thermal_region_coords"] = core_output.get("hyper_region_coords")
        org["active_hyperedge_mask"] = (core_output["hyper_strength"] > 0.05).to(dtype=core_output["hyper_strength"].dtype)
        org["module_centers"] = adapter.module_centers
        org["module_present"] = adapter.module_present
        org["heat_powers"] = adapter.heat_powers
        org["env_coords"] = env_coords
        return org

    def _temperature_from_field_output(self, field_values: torch.Tensor) -> torch.Tensor:
        temperature = field_values[..., 4] if field_values.shape[-1] > 4 else field_values.new_zeros(field_values.shape[:-1])
        if not self.global_normalize_targets:
            return temperature
        mean = self.global_normalization_stats.get("field_mean_by_channel")
        std = self.global_normalization_stats.get("field_std_by_channel")
        if mean is None or std is None:
            return temperature
        mean_t = torch.as_tensor(mean, device=field_values.device, dtype=field_values.dtype)
        std_t = torch.as_tensor(std, device=field_values.device, dtype=field_values.dtype)
        if mean_t.numel() <= 4 or std_t.numel() <= 4:
            return temperature
        return temperature * std_t[4].clamp_min(1.0e-6) + mean_t[4]

    def _port_subset_indices(self, ntheta: int, device: torch.device) -> torch.Tensor:
        # ChannelThermal-specific: compare T_env only on a controlled subset of
        # angular boundary points to keep this auxiliary physical loss cheap.
        count = int(self.config.channelthermal.port_global_consistency_num_points)
        count = max(1, min(count, int(ntheta)))
        if count >= int(ntheta):
            return torch.arange(int(ntheta), device=device)
        return torch.linspace(0, int(ntheta) - 1, count, device=device).round().long()

    def _decode_global_temperature_at_ports(
        self,
        port_tokens: torch.Tensor,
        module_state: torch.Tensor,
        org: Dict[str, torch.Tensor],
        global_token: torch.Tensor,
        module_centers: torch.Tensor,
        module_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_modules, ntheta, _ = port_tokens.shape
        indices = self._port_subset_indices(ntheta, port_tokens.device)
        selected_ports = port_tokens.index_select(dim=-2, index=indices)
        normals = selected_ports[..., 1:3]
        radius = float(self.config.core_honf.module_radius) + float(self.config.channelthermal.port_global_consistency_radius_offset)
        outside_xy = module_centers[:, :, None, :] + radius * normals
        outside_xy = torch.stack(
            [
                outside_xy[..., 0].clamp(0.0, float(self.config.core_honf.domain_length_x)),
                outside_xy[..., 1].clamp(0.0, float(self.config.core_honf.domain_length_y)),
            ],
            dim=-1,
        )
        flat_xy = outside_xy.reshape(batch, num_modules * int(indices.numel()), 2)
        flat_field = self.core.decoder(flat_xy, None, org, global_token)["pred_field"]
        temperature = self._temperature_from_field_output(flat_field).reshape(batch, num_modules, int(indices.numel()))
        target_t_env = selected_ports[..., 3]
        valid_mask = module_present[:, :, None].expand_as(temperature)
        return temperature * valid_mask, target_t_env * valid_mask, valid_mask

    def _global_temperature_for_all_ports(
        self,
        port_tokens: torch.Tensor,
        module_state: torch.Tensor,
        org: Dict[str, torch.Tensor],
        global_token: torch.Tensor,
        module_centers: torch.Tensor,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_modules, ntheta, _ = port_tokens.shape
        normals = port_tokens[..., 1:3]
        radius = float(self.config.core_honf.module_radius) + float(self.config.channelthermal.port_global_consistency_radius_offset)
        outside_xy = module_centers[:, :, None, :] + radius * normals
        outside_xy = torch.stack(
            [
                outside_xy[..., 0].clamp(0.0, float(self.config.core_honf.domain_length_x)),
                outside_xy[..., 1].clamp(0.0, float(self.config.core_honf.domain_length_y)),
            ],
            dim=-1,
        )
        flat_xy = outside_xy.reshape(batch, num_modules * ntheta, 2)
        flat_field = self.core.decoder(flat_xy, None, org, global_token)["pred_field"]
        return self._temperature_from_field_output(flat_field).reshape(batch, num_modules, ntheta) * module_present[:, :, None]

    def _infer_ntheta(
        self,
        interface_condition: Optional[torch.Tensor],
        teacher_port_tokens: Optional[torch.Tensor],
    ) -> int:
        if interface_condition is not None and interface_condition.ndim >= 4:
            return int(interface_condition.shape[-2])
        if teacher_port_tokens is not None and teacher_port_tokens.ndim >= 4:
            return int(teacher_port_tokens.shape[-2])
        return int(self.config.channelthermal.default_num_interface_points)
