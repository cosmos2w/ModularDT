"""CHANNELTHERMAL-SPECIFIC one-way local surrogate coupling.

Inputs are base HONF module tokens, A_me-derived module/environment context,
heat powers, global token, optional teacher interface conditions, and optional
pretrained local surrogate normalization stats. Outputs are predicted/selected
port tokens, local internal/interface predictions, local response latents, and
summary features used to update global module tokens.

This is ChannelThermal-specific because the port token contract is fixed to
`theta, cos(theta), sin(theta), T_env, h_effective`, and local module
parameters encode heat power, material descriptors, and per-port T_env/h
statistics. Prompt 3 restores the proven ChannelThermal physical corrections:
Robin flux correction and one interaction-refinement pass. These remain outside
the reusable HONF core because they depend on case-specific interface physics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from _helpers.model_utils import MLP, FourierEncoder, load_trusted_checkpoint, resolve_demo_path, safe_std_torch, strip_module_prefix
from _models_local.model_local import LocalModuleConfig, LocalModuleSurrogate


def teacher_port_tokens_from_interface_condition(interface_condition: torch.Tensor) -> torch.Tensor:
    """Map global interface-condition columns to local surrogate port tokens."""

    h_index = 7 if interface_condition.shape[-1] >= 8 else 6
    return torch.cat(
        [
            interface_condition[..., 0:1],
            interface_condition[..., 1:3],
            interface_condition[..., 3:4],
            interface_condition[..., h_index : h_index + 1],
        ],
        dim=-1,
    )


def build_local_module_params_from_global(
    heat_powers: torch.Tensor,
    interface_condition: Optional[torch.Tensor],
    material_params: Optional[torch.Tensor],
    module_present: torch.Tensor,
) -> torch.Tensor:
    batch, num_modules = heat_powers.shape
    params = heat_powers.new_zeros(batch, num_modules, 7)
    params[..., 0] = heat_powers
    if material_params is not None and material_params.shape[-1] >= 4:
        params[..., 1] = material_params[:, None, 3]
        params[..., 2] = material_params[:, None, 1]
    if interface_condition is not None and interface_condition.shape[-1] >= 7:
        t_out = interface_condition[..., 3]
        h_index = 7 if interface_condition.shape[-1] >= 8 else 6
        h_local = interface_condition[..., h_index]
        params[..., 3] = h_local.mean(dim=-1)
        params[..., 4] = h_local.std(dim=-1, unbiased=False)
        params[..., 5] = t_out.mean(dim=-1)
        params[..., 6] = t_out.std(dim=-1, unbiased=False)
    return params * module_present[..., None]


def update_local_module_params_from_ports(
    base_module_params: torch.Tensor,
    local_ports: torch.Tensor,
    module_present: torch.Tensor,
) -> torch.Tensor:
    """Refresh local parameter T_env/h summary columns from selected ports."""

    if base_module_params.shape[-1] < 7 or local_ports.shape[-1] < 5:
        return base_module_params * module_present[..., None]
    params = base_module_params.clone()
    t_env = local_ports[..., 3]
    h_local = local_ports[..., 4]
    params[..., 3] = h_local.mean(dim=-1)
    params[..., 4] = h_local.std(dim=-1, unbiased=False)
    params[..., 5] = t_env.mean(dim=-1)
    params[..., 6] = t_env.std(dim=-1, unbiased=False)
    return params * module_present[..., None]


class PortConditionHead(nn.Module):
    """Predict per-module local-surrogate condition tokens."""

    def __init__(self, hidden_dim: int, dropout: float, use_layer_norm: bool):
        super().__init__()
        self.theta_encoder = FourierEncoder(3, 2, include_input=True)
        self.net = MLP(
            3 * int(hidden_dim) + 2 + self.theta_encoder.output_dim,
            int(hidden_dim),
            2,
            num_layers=3,
            dropout=float(dropout),
            layer_norm=bool(use_layer_norm),
        )

    def fixed_theta_tokens(self, ntheta: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        theta = torch.linspace(0.0, 2.0 * math.pi, int(ntheta) + 1, device=device, dtype=dtype)[:-1]
        return torch.stack([theta, torch.cos(theta), torch.sin(theta)], dim=-1)

    def forward(
        self,
        module_state: torch.Tensor,
        module_env_context: torch.Tensor,
        heat_powers: torch.Tensor,
        global_token: torch.Tensor,
        *,
        ntheta: int,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_modules, _ = module_state.shape
        theta_tokens = self.fixed_theta_tokens(ntheta, module_state.device, module_state.dtype)
        theta_features = self.theta_encoder(theta_tokens).view(1, 1, int(ntheta), -1).expand(batch, num_modules, -1, -1)
        module_features = torch.cat([module_state, module_env_context], dim=-1)
        module_features = module_features[:, :, None, :].expand(-1, -1, int(ntheta), -1)
        heat = heat_powers[:, :, None, None].expand(-1, -1, int(ntheta), 1)
        present = module_present[:, :, None, None].expand(-1, -1, int(ntheta), 1)
        global_features = global_token[:, None, None, :].expand(-1, num_modules, int(ntheta), -1)
        values = self.net(torch.cat([module_features, global_features, theta_features, heat, present], dim=-1))
        t_env = values[..., 0:1]
        h_effective = F.softplus(values[..., 1:2]) + 1.0e-4
        fixed = theta_tokens.view(1, 1, int(ntheta), 3).expand(batch, num_modules, -1, -1)
        return torch.cat([fixed, t_env, h_effective], dim=-1) * module_present[:, :, None, None]


class FluxCorrectionHead(nn.Module):
    """CHANNELTHERMAL-SPECIFIC residual q_normal correction anchored at Robin physics."""

    def __init__(self, hidden_dim: int, local_surrogate_latent_dim: int, dropout: float, use_layer_norm: bool):
        super().__init__()
        in_dim = int(hidden_dim) + int(local_surrogate_latent_dim) + 4
        self.net = MLP(
            in_dim,
            int(hidden_dim),
            1,
            num_layers=2,
            dropout=float(dropout),
            layer_norm=bool(use_layer_norm),
        )
        last = self.net.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        module_state: torch.Tensor,
        response_latent: torch.Tensor,
        t_surface: torch.Tensor,
        t_env: torch.Tensor,
        h_effective: torch.Tensor,
        q_surrogate_raw: torch.Tensor,
    ) -> torch.Tensor:
        _, _, ntheta, _ = t_surface.shape
        module_features = module_state[:, :, None, :].expand(-1, -1, ntheta, -1)
        latent_features = response_latent[:, :, None, :].expand(-1, -1, ntheta, -1)
        physical_features = torch.cat([t_surface, t_env, torch.log1p(h_effective.clamp_min(0.0)), q_surrogate_raw], dim=-1)
        return self.net(torch.cat([module_features, latent_features, physical_features], dim=-1))


class PortRefinementHead(nn.Module):
    """CHANNELTHERMAL-SPECIFIC residual update for one local/global interaction pass."""

    def __init__(self, hidden_dim: int, dropout: float, use_layer_norm: bool):
        super().__init__()
        self.theta_encoder = FourierEncoder(3, 2, include_input=True)
        self.net = MLP(
            int(hidden_dim) + self.theta_encoder.output_dim + 5 + 2 + 6,
            int(hidden_dim),
            2,
            num_layers=2,
            dropout=float(dropout),
            layer_norm=bool(use_layer_norm),
        )
        last = self.net.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        module_state: torch.Tensor,
        current_port_tokens: torch.Tensor,
        outside_temperature: torch.Tensor,
        local_response_summary: torch.Tensor,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_modules, ntheta, _ = current_port_tokens.shape
        theta_features = self.theta_encoder(current_port_tokens[..., 0:3])
        module_features = module_state[:, :, None, :].expand(-1, -1, ntheta, -1)
        outside_features = torch.cat(
            [
                outside_temperature[..., None],
                outside_temperature[..., None] - current_port_tokens[..., 3:4],
            ],
            dim=-1,
        )
        response_features = local_response_summary[:, :, None, :].expand(-1, -1, ntheta, -1)
        delta = self.net(torch.cat([module_features, theta_features, current_port_tokens, outside_features, response_features], dim=-1))
        refined_t_env = current_port_tokens[..., 3:4] + delta[..., 0:1]
        log_h = torch.log1p(current_port_tokens[..., 4:5].clamp_min(0.0)) + delta[..., 1:2]
        refined_h = torch.expm1(log_h).clamp_min(1.0e-6)
        refined = torch.cat([current_port_tokens[..., 0:3], refined_t_env, refined_h], dim=-1)
        return refined * module_present[:, :, None, None]


class LocalSurrogateCoupling(nn.Module):
    """One-way local response coupling around a pretrained local surrogate."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        local_surrogate_latent_dim: int,
        dropout: float,
        use_layer_norm: bool,
        local_module_params_from_used_ports: bool = True,
        local_surrogate_flux_mode: str = "surrogate",
        local_surrogate_flux_blend_alpha: float = 0.5,
    ):
        super().__init__()
        self.local_surrogate: Optional[LocalModuleSurrogate] = None
        self.local_surrogate_normalize_inputs = False
        self.local_surrogate_normalize_targets = False
        self.local_module_params_from_used_ports = bool(local_module_params_from_used_ports)
        self.local_surrogate_flux_mode = str(local_surrogate_flux_mode)
        self.local_surrogate_flux_blend_alpha = float(local_surrogate_flux_blend_alpha)
        self.port_head = PortConditionHead(hidden_dim, dropout, use_layer_norm)
        self.flux_correction_head = FluxCorrectionHead(hidden_dim, local_surrogate_latent_dim, dropout, use_layer_norm)
        self.port_refinement_head = PortRefinementHead(hidden_dim, dropout, use_layer_norm)
        self.local_latent_fusion = MLP(
            int(hidden_dim) + int(local_surrogate_latent_dim),
            int(hidden_dim),
            int(hidden_dim),
            num_layers=2,
            dropout=float(dropout),
            layer_norm=bool(use_layer_norm),
        )
        self.local_response_summary_proj = MLP(6, int(hidden_dim), int(hidden_dim), num_layers=2, dropout=float(dropout), layer_norm=bool(use_layer_norm))
        for name in (
            "local_module_params_mean",
            "local_module_params_std",
            "local_port_tokens_mean",
            "local_port_tokens_std",
            "local_internal_temperature_mean",
            "local_internal_temperature_std",
            "local_interface_targets_mean",
            "local_interface_targets_std",
            "global_internal_temperature_mean",
            "global_internal_temperature_std",
            "global_interface_target_mean",
            "global_interface_target_std",
        ):
            self.register_buffer(name, torch.empty(0), persistent=False)
        self.global_normalize_targets = False

    @property
    def has_local_surrogate(self) -> bool:
        return self.local_surrogate is not None

    def attach_from_checkpoint(self, checkpoint_path: str | Path, *, freeze: bool = True, map_location: Any = "cpu") -> Dict[str, Any]:
        checkpoint = load_trusted_checkpoint(resolve_demo_path(checkpoint_path), map_location=map_location)
        config = LocalModuleConfig.from_dict(checkpoint.get("model_config", {}))
        model = LocalModuleSurrogate(config)
        model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
        normalization_config = checkpoint.get("local_normalization_config")
        if not isinstance(normalization_config, dict):
            dataset_cfg = checkpoint.get("train_config", {}).get("dataset", {})
            normalization_config = {
                "normalize_inputs": bool(dataset_cfg.get("normalize_inputs", False)),
                "normalize_targets": bool(dataset_cfg.get("normalize_targets", False)),
            }
        normalization_stats = checkpoint.get("local_normalization_stats", {})
        if not isinstance(normalization_stats, dict):
            normalization_stats = {}
        self.set_local_surrogate(model, freeze=freeze, normalization_config=normalization_config, normalization_stats=normalization_stats)
        return checkpoint

    def set_global_target_normalization(self, stats: Optional[Dict[str, Any]], *, normalize_targets: bool) -> None:
        self.global_normalize_targets = bool(normalize_targets)
        self._set_buffer_from_stat("global_internal_temperature_mean", stats, "internal_temperature_mean")
        self._set_buffer_from_stat("global_internal_temperature_std", stats, "internal_temperature_std")
        self._set_buffer_from_stat("global_interface_target_mean", stats, "interface_target_mean", "interface_targets_mean")
        self._set_buffer_from_stat("global_interface_target_std", stats, "interface_target_std", "interface_targets_std")

    def set_local_surrogate(
        self,
        model: LocalModuleSurrogate,
        *,
        freeze: bool,
        normalization_config: Optional[Dict[str, Any]],
        normalization_stats: Optional[Dict[str, Any]],
    ) -> None:
        self.local_surrogate = model
        self.local_surrogate_normalize_inputs = bool((normalization_config or {}).get("normalize_inputs", False))
        self.local_surrogate_normalize_targets = bool((normalization_config or {}).get("normalize_targets", False))
        self._set_buffer_from_stat("local_module_params_mean", normalization_stats, "module_params_mean")
        self._set_buffer_from_stat("local_module_params_std", normalization_stats, "module_params_std")
        self._set_buffer_from_stat("local_port_tokens_mean", normalization_stats, "port_tokens_mean")
        self._set_buffer_from_stat("local_port_tokens_std", normalization_stats, "port_tokens_std")
        self._set_buffer_from_stat("local_internal_temperature_mean", normalization_stats, "internal_temperature_mean")
        self._set_buffer_from_stat("local_internal_temperature_std", normalization_stats, "internal_temperature_std")
        self._set_buffer_from_stat("local_interface_targets_mean", normalization_stats, "interface_targets_mean")
        self._set_buffer_from_stat("local_interface_targets_std", normalization_stats, "interface_targets_std")
        if freeze:
            self.local_surrogate.eval()
            for param in self.local_surrogate.parameters():
                param.requires_grad_(False)

    def _set_buffer_from_stat(self, name: str, stats: Optional[Dict[str, Any]], key: str, *alternate_keys: str) -> None:
        value = None
        if stats is not None:
            for candidate in (key, *alternate_keys):
                value = stats.get(candidate)
                if value is not None:
                    break
        setattr(self, name, torch.empty(0) if value is None else torch.as_tensor(value, dtype=torch.float32).clone().detach())

    def _normalize_with_stats(self, values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if mean.numel() == 0 or std.numel() == 0:
            return values
        mean = mean.to(device=values.device, dtype=values.dtype)
        std = safe_std_torch(std.to(device=values.device, dtype=values.dtype))
        return (values - mean) / std

    def _denormalize_with_stats(self, values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if mean.numel() == 0 or std.numel() == 0:
            return values
        mean = mean.to(device=values.device, dtype=values.dtype)
        std = safe_std_torch(std.to(device=values.device, dtype=values.dtype))
        return values * std + mean

    def choose_local_ports(
        self,
        *,
        pred_port_tokens: torch.Tensor,
        teacher_port_tokens: Optional[torch.Tensor],
        mode: str,
        mixed_teacher_ratio: float,
    ) -> torch.Tensor:
        mode = str(mode).lower()
        if mode == "teacher" and teacher_port_tokens is not None:
            return teacher_port_tokens.float()
        if mode == "mixed" and teacher_port_tokens is not None:
            ratio = float(mixed_teacher_ratio)
            return ratio * teacher_port_tokens.float() + (1.0 - ratio) * pred_port_tokens
        return pred_port_tokens

    def local_module_params_for_ports(
        self,
        base_module_params: torch.Tensor,
        local_ports: torch.Tensor,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        if self.local_module_params_from_used_ports:
            return update_local_module_params_from_ports(base_module_params, local_ports, module_present)
        return base_module_params * module_present[..., None]

    def call_local_surrogate(
        self,
        module_params: torch.Tensor,
        port_tokens: torch.Tensor,
        local_query_points: Optional[torch.Tensor],
        module_present: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.local_surrogate is None:
            raise RuntimeError("Local surrogate was requested but no LocalModuleSurrogate is attached.")
        batch, num_modules, ntheta, _ = port_tokens.shape
        flat_params = module_params.reshape(batch * num_modules, module_params.shape[-1])
        flat_ports = port_tokens.reshape(batch * num_modules, ntheta, port_tokens.shape[-1])
        if local_query_points is not None:
            if local_query_points.ndim == 3:
                expanded_query = local_query_points[:, None, :, :].expand(-1, num_modules, -1, -1)
            else:
                expanded_query = local_query_points
            flat_query = expanded_query.reshape(batch * num_modules, expanded_query.shape[-2], 2)
        else:
            flat_query = None
        if self.local_surrogate_normalize_inputs:
            flat_params = self._normalize_with_stats(flat_params, self.local_module_params_mean, self.local_module_params_std)
            flat_ports = self._normalize_with_stats(flat_ports, self.local_port_tokens_mean, self.local_port_tokens_std)
        out = self.local_surrogate(flat_params, flat_ports, flat_query)
        present = module_present[:, :, None, None]
        internal = out["internal_temperature"].reshape(batch, num_modules, -1, 1)
        interface = out["interface_pred"].reshape(batch, num_modules, ntheta, -1)
        if self.local_surrogate_normalize_targets:
            internal = self._denormalize_with_stats(internal, self.local_internal_temperature_mean, self.local_internal_temperature_std)
            interface = self._denormalize_with_stats(interface, self.local_interface_targets_mean, self.local_interface_targets_std)
        if self.global_normalize_targets:
            internal = self._normalize_with_stats(internal, self.global_internal_temperature_mean, self.global_internal_temperature_std)
            interface = self._normalize_with_stats(interface, self.global_interface_target_mean, self.global_interface_target_std)
        latent = out["module_response_latent"].reshape(batch, num_modules, -1) * module_present[..., None]
        return {
            "internal_temperature": internal * present,
            "interface_pred": interface * present,
            "module_response_latent": latent,
        }

    def assemble_interface(
        self,
        *,
        local_outputs: Dict[str, torch.Tensor],
        local_ports: torch.Tensor,
        module_state: torch.Tensor,
        module_present: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Select q_normal from surrogate/Robin/corrected modes and emit old diagnostics."""

        surrogate_interface = local_outputs["interface_pred"]
        if self.global_normalize_targets:
            physical_surrogate = self._denormalize_with_stats(
                surrogate_interface,
                self.global_interface_target_mean,
                self.global_interface_target_std,
            )
        else:
            physical_surrogate = surrogate_interface
        t_surface = physical_surrogate[..., 0:1]
        q_surrogate = physical_surrogate[..., 1:2]
        t_env = local_ports[..., 3:4]
        h_effective = local_ports[..., 4:5]
        q_physics = h_effective * (t_surface - t_env)
        mode = self.local_surrogate_flux_mode.lower()
        if mode == "surrogate":
            q_use = q_surrogate
        elif mode == "physics_from_port":
            q_use = q_physics
        elif mode == "corrected_physics":
            delta_q = self.flux_correction_head(
                module_state,
                local_outputs["module_response_latent"],
                t_surface,
                t_env,
                h_effective,
                q_surrogate,
            )
            q_use = q_physics + delta_q
        elif mode == "blend":
            alpha = self.local_surrogate_flux_blend_alpha
            q_use = alpha * q_surrogate + (1.0 - alpha) * q_physics
        else:
            raise ValueError(
                "local_surrogate_flux_mode must be 'surrogate', 'physics_from_port', "
                f"'corrected_physics', or 'blend'; got {self.local_surrogate_flux_mode!r}."
            )
        present = module_present[:, :, None, None]
        physical_selected = torch.cat([t_surface, q_use], dim=-1) * present
        physical_physics = torch.cat([t_surface, q_physics], dim=-1) * present
        physical_surrogate = physical_surrogate * present
        if self.global_normalize_targets:
            selected = self._normalize_with_stats(
                physical_selected,
                self.global_interface_target_mean,
                self.global_interface_target_std,
            )
            physics = self._normalize_with_stats(
                physical_physics,
                self.global_interface_target_mean,
                self.global_interface_target_std,
            )
            surrogate = self._normalize_with_stats(
                physical_surrogate,
                self.global_interface_target_mean,
                self.global_interface_target_std,
            )
        else:
            selected = physical_selected
            physics = physical_physics
            surrogate = physical_surrogate
        diagnostics = {
            "pred_interface_surrogate_raw": surrogate * present,
            "pred_interface_flux_physics": physics * present,
            "pred_interface_physics": physics * present,
        }
        if mode == "corrected_physics":
            diagnostics["pred_interface_delta_q"] = (q_use - q_physics) * present
        return selected * present, diagnostics

    def local_response_summary(
        self,
        *,
        local_outputs: Dict[str, torch.Tensor],
        module_present: torch.Tensor,
        interface_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        interface = interface_override if interface_override is not None else local_outputs["interface_pred"]
        internal = local_outputs["internal_temperature"]
        if self.global_normalize_targets:
            physical_interface = self._denormalize_with_stats(interface, self.global_interface_target_mean, self.global_interface_target_std)
            physical_internal = self._denormalize_with_stats(internal, self.global_internal_temperature_mean, self.global_internal_temperature_std)
        else:
            physical_interface = interface
            physical_internal = internal
        t_surface = physical_interface[..., 0]
        q_normal = physical_interface[..., 1] if physical_interface.shape[-1] > 1 else torch.zeros_like(t_surface)
        if physical_internal.shape[-2] > 0:
            internal_mean = physical_internal[..., 0].mean(dim=-1)
            internal_max = physical_internal[..., 0].amax(dim=-1)
        else:
            internal_mean = module_present.new_zeros(module_present.shape)
            internal_max = module_present.new_zeros(module_present.shape)
        return torch.stack(
            [
                t_surface.mean(dim=-1),
                t_surface.amax(dim=-1),
                q_normal.mean(dim=-1),
                q_normal.amax(dim=-1),
                internal_mean,
                internal_max,
            ],
            dim=-1,
        ) * module_present[..., None]

    def fuse_module_state(
        self,
        base_module_state: torch.Tensor,
        local_outputs: Dict[str, torch.Tensor],
        local_response_summary: torch.Tensor,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        latent_fusion = self.local_latent_fusion(torch.cat([base_module_state, local_outputs["module_response_latent"]], dim=-1))
        summary_fusion = self.local_response_summary_proj(local_response_summary)
        return (base_module_state + latent_fusion + summary_fusion) * module_present[..., None]
