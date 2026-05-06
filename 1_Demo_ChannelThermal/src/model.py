from __future__ import annotations

"""Stage B global Channel Thermal hypergraph-organized neural field model."""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from channelthermal_model_utils import (
    EPS,
    FourierEncoder,
    MLP,
    dataclass_from_dict,
    dataclass_to_dict,
    masked_softmax,
    strip_module_prefix,
)
from model_local import LocalModuleConfig, LocalModuleSurrogate


@dataclass
class GlobalChannelThermalModelConfig:
    field_dim: int = 5
    field_names: Sequence[str] = field(default_factory=lambda: ["u", "v", "p", "omega", "temperature"])
    max_num_modules: int = 8
    domain_length_x: float = 12.0
    domain_length_y: float = 4.0
    module_radius: float = 0.45
    num_env_tokens_x: int = 16
    num_env_tokens_y: int = 6
    num_hyperedges: int = 6
    hidden_dim: int = 96
    latent_dim: int = 96
    use_local_surrogate: bool = True
    freeze_local_surrogate: bool = True
    local_surrogate_latent_dim: int = 128
    local_module_param_dim: int = 7
    local_port_token_dim: int = 5
    local_interface_target_dim: int = 2
    material_param_dim: int = 6
    future_global_feature_dim: int = 0
    dropout: float = 0.05
    use_layer_norm: bool = True
    spatial_query_fourier_frequencies: int = 4
    local_coord_fourier_frequencies: int = 4
    num_attention_heads: int = 4
    decoder_hidden_dim: int = 128
    DISABLE_EDGE: bool = False
    disable_edge_strength_threshold: float = 0.05
    default_num_interface_points: int = 64

    @classmethod
    def from_dict(cls, payload: Dict) -> "GlobalChannelThermalModelConfig":
        return dataclass_from_dict(cls, payload)

    def to_dict(self) -> Dict:
        out = dataclass_to_dict(self)
        out["field_names"] = list(self.field_names)
        return out


def nonperiodic_relative_geometry(
    query_xy: torch.Tensor,
    module_xy: torch.Tensor,
    *,
    domain_length_x: float,
    domain_length_y: float,
) -> torch.Tensor:
    """Return nonperiodic query-to-module geometry features.

    Features are ``dx, dy, distance, downstream, upstream, wall_bottom,
    wall_top, inlet_distance, outlet_distance``. There is no periodic wrapping:
    ``dx = query_x - module_x`` and ``dy = query_y - module_y`` directly.
    """
    dx = query_xy[..., 0] - module_xy[..., 0]
    dy = query_xy[..., 1] - module_xy[..., 1]
    distance = torch.sqrt(dx.square() + dy.square() + EPS)
    downstream = torch.relu(dx)
    upstream = torch.relu(-dx)
    y = query_xy[..., 1]
    x = query_xy[..., 0]
    wall_bottom = y.clamp_min(0.0)
    wall_top = (float(domain_length_y) - y).clamp_min(0.0)
    inlet = x.clamp_min(0.0)
    outlet = (float(domain_length_x) - x).clamp_min(0.0)
    return torch.stack([dx, dy, distance, downstream, upstream, wall_bottom, wall_top, inlet, outlet], dim=-1)


def boundary_features(query_xy: torch.Tensor, *, domain_length_x: float, domain_length_y: float) -> torch.Tensor:
    x = query_xy[..., 0]
    y = query_xy[..., 1]
    lx = max(float(domain_length_x), EPS)
    ly = max(float(domain_length_y), EPS)
    return torch.stack(
        [
            x / lx,
            y / ly,
            y / ly,
            (ly - y) / ly,
            x / lx,
            (lx - x) / lx,
        ],
        dim=-1,
    )


def weighted_mean_coords(coords: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted arithmetic mean for nonperiodic coordinates."""
    # coords: [B, N, 2], weights: [B, N, K]
    denom = weights.sum(dim=1).clamp_min(EPS).unsqueeze(-1)
    return torch.einsum("bnk,bnd->bkd", weights, coords) / denom


def teacher_port_tokens_from_interface_condition(interface_condition: torch.Tensor) -> torch.Tensor:
    """Map global interface_condition columns to local surrogate port tokens.

    Global condition columns are theta, normal_x, normal_y, T_outside,
    u_normal, u_tangent, h_proxy. The local surrogate expects theta, cos_theta,
    sin_theta, T_env, h.
    """
    return torch.cat(
        [
            interface_condition[..., 0:1],
            interface_condition[..., 1:3],
            interface_condition[..., 3:4],
            interface_condition[..., 6:7],
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
        h_proxy = interface_condition[..., 6]
        params[..., 3] = h_proxy.mean(dim=-1)
        params[..., 4] = h_proxy.std(dim=-1, unbiased=False)
        params[..., 5] = t_out.mean(dim=-1)
        params[..., 6] = t_out.std(dim=-1, unbiased=False)
    return params * module_present[..., None]


class HypergraphOrganizer(nn.Module):
    """Simplified thermal/flow interaction organizer.

    A hyperedge here is a learned thermal/flow interaction group. It is not
    restricted to wake semantics; it can represent near-module heat plumes,
    wall-influenced regions, or broader flow coupling zones.
    """

    def __init__(self, config: GlobalChannelThermalModelConfig):
        super().__init__()
        hidden = int(config.hidden_dim)
        num_h = int(config.num_hyperedges)
        self.config = config
        self.module_to_hyper = nn.Linear(hidden, num_h)
        self.env_to_hyper = nn.Linear(hidden, num_h)
        self.module_q = nn.Linear(hidden, hidden)
        self.env_k = nn.Linear(hidden, hidden)
        self.me_geom_bias = MLP(5, hidden, 1, num_layers=2, dropout=config.dropout)
        self.hyper_fuse = MLP(2 * hidden + 4, hidden, hidden, num_layers=3, dropout=config.dropout, layer_norm=config.use_layer_norm)
        self.hyper_strength = MLP(hidden, hidden, 1, num_layers=2, dropout=config.dropout)

    def forward(
        self,
        module_state: torch.Tensor,
        env_state: torch.Tensor,
        module_centers: torch.Tensor,
        env_coords: torch.Tensor,
        module_present: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, num_modules, hidden = module_state.shape
        num_env = env_state.shape[1]
        module_mask = module_present > 0.5

        q = self.module_q(module_state)
        k = self.env_k(env_state)
        logits_me = torch.einsum("bnd,bed->bne", q, k) / math.sqrt(float(hidden))
        rel = nonperiodic_relative_geometry(
            env_coords[None, None, :, :].expand(batch, num_modules, num_env, 2),
            module_centers[:, :, None, :].expand(batch, num_modules, num_env, 2),
            domain_length_x=self.config.domain_length_x,
            domain_length_y=self.config.domain_length_y,
        )
        logits_me = logits_me + self.me_geom_bias(rel[..., :5]).squeeze(-1)
        a_me = masked_softmax(logits_me, module_mask[:, :, None].expand_as(logits_me), dim=-1)
        module_env_context = torch.einsum("bne,bed->bnd", a_me, env_state)

        logits_mh = self.module_to_hyper(module_state)
        a_mh = masked_softmax(logits_mh, module_mask[:, :, None].expand_as(logits_mh), dim=-1)
        a_mh = a_mh * module_present[..., None]
        logits_eh = self.env_to_hyper(env_state)
        a_eh = torch.softmax(logits_eh, dim=-1)

        mh_norm = a_mh / a_mh.sum(dim=1, keepdim=True).clamp_min(EPS)
        eh_norm = a_eh / a_eh.sum(dim=1, keepdim=True).clamp_min(EPS)
        module_agg = torch.einsum("bnk,bnd->bkd", mh_norm, module_state)
        env_agg = torch.einsum("bek,bed->bkd", eh_norm, env_state)
        hyper_source_coords = weighted_mean_coords(module_centers, a_mh)
        hyper_thermal_region_coords = torch.einsum("bek,ed->bkd", eh_norm, env_coords)
        geom = torch.cat(
            [
                hyper_source_coords[..., 0:1] / max(float(self.config.domain_length_x), EPS),
                hyper_source_coords[..., 1:2] / max(float(self.config.domain_length_y), EPS),
                hyper_thermal_region_coords[..., 0:1] / max(float(self.config.domain_length_x), EPS),
                hyper_thermal_region_coords[..., 1:2] / max(float(self.config.domain_length_y), EPS),
            ],
            dim=-1,
        )
        hyper_state = self.hyper_fuse(torch.cat([module_agg, env_agg, geom], dim=-1))
        hyper_strength = torch.sigmoid(self.hyper_strength(hyper_state)).squeeze(-1)
        if self.config.DISABLE_EDGE:
            active_hyperedge_mask = hyper_strength >= float(self.config.disable_edge_strength_threshold)
            hyper_state = hyper_state * active_hyperedge_mask[..., None].to(hyper_state.dtype)
        else:
            active_hyperedge_mask = torch.ones_like(hyper_strength, dtype=torch.bool)

        return {
            "A_me": a_me,
            "A_mh": a_mh,
            "A_eh": a_eh,
            "hyper_state": hyper_state,
            "hyper_source_coords": hyper_source_coords,
            "hyper_thermal_region_coords": hyper_thermal_region_coords,
            "hyper_wake_coords": hyper_thermal_region_coords,
            "hyper_strength": hyper_strength,
            "active_hyperedge_mask": active_hyperedge_mask,
            "module_env_context": module_env_context,
        }


class PortConditionHead(nn.Module):
    """Predict per-module local-surrogate condition tokens.

    The theta/cos/sin columns are known geometry. The head predicts only the
    physical condition values ``T_env`` and ``h``.
    """

    def __init__(self, config: GlobalChannelThermalModelConfig):
        super().__init__()
        hidden = int(config.hidden_dim)
        self.config = config
        self.theta_encoder = FourierEncoder(3, 2, include_input=True)
        self.net = MLP(
            3 * hidden + 2 + self.theta_encoder.output_dim,
            hidden,
            2,
            num_layers=3,
            dropout=config.dropout,
            layer_norm=config.use_layer_norm,
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
        batch, num_modules, hidden = module_state.shape
        theta_tokens = self.fixed_theta_tokens(ntheta, module_state.device, module_state.dtype)
        theta_features = self.theta_encoder(theta_tokens).view(1, 1, ntheta, -1).expand(batch, num_modules, -1, -1)
        module_features = torch.cat([module_state, module_env_context], dim=-1)
        module_features = module_features[:, :, None, :].expand(-1, -1, ntheta, -1)
        heat = heat_powers[:, :, None, None].expand(-1, -1, ntheta, 1)
        present = module_present[:, :, None, None].expand(-1, -1, ntheta, 1)
        global_features = global_token[:, None, None, :].expand(-1, num_modules, ntheta, -1)
        values = self.net(torch.cat([module_features, global_features, theta_features, heat, present], dim=-1))
        t_env = values[..., 0:1]
        h_proxy = F.softplus(values[..., 1:2]) + 1.0e-4
        fixed = theta_tokens.view(1, 1, ntheta, 3).expand(batch, num_modules, -1, -1)
        return torch.cat([fixed, t_env, h_proxy], dim=-1) * module_present[:, :, None, None]


class QueryFieldDecoder(nn.Module):
    """Steady neural-field decoder over channel query points."""

    def __init__(self, config: GlobalChannelThermalModelConfig):
        super().__init__()
        hidden = int(config.hidden_dim)
        self.config = config
        self.query_fourier = FourierEncoder(2, config.spatial_query_fourier_frequencies, include_input=True)
        self.query_encoder = MLP(
            self.query_fourier.output_dim + 6,
            hidden,
            hidden,
            num_layers=3,
            dropout=config.dropout,
            layer_norm=config.use_layer_norm,
        )
        self.memory_attn = nn.MultiheadAttention(hidden, config.num_attention_heads, dropout=config.dropout, batch_first=True)
        self.near_module_geom = MLP(9, hidden, hidden, num_layers=2, dropout=config.dropout)
        self.output = MLP(
            4 * hidden,
            int(config.decoder_hidden_dim),
            config.field_dim,
            num_layers=4,
            dropout=config.dropout,
            layer_norm=config.use_layer_norm,
        )

    def nearby_module_context(
        self,
        query_xy: torch.Tensor,
        module_centers: torch.Tensor,
        module_state: torch.Tensor,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_queries, _ = query_xy.shape
        num_modules = module_centers.shape[1]
        if num_modules == 0:
            return query_xy.new_zeros(batch, num_queries, self.config.hidden_dim)
        rel = nonperiodic_relative_geometry(
            query_xy[:, :, None, :].expand(-1, -1, num_modules, -1),
            module_centers[:, None, :, :].expand(-1, num_queries, -1, -1),
            domain_length_x=self.config.domain_length_x,
            domain_length_y=self.config.domain_length_y,
        )
        dist = rel[..., 2]
        logits = -dist / max(float(self.config.module_radius), EPS)
        weights = masked_softmax(logits, module_present[:, None, :].expand_as(logits) > 0.5, dim=-1)
        module_context = torch.einsum("bqn,bnd->bqd", weights, module_state)
        geom_context = torch.einsum("bqn,bqnf->bqf", weights, rel)
        return module_context + self.near_module_geom(geom_context)

    def forward(
        self,
        query_xy: torch.Tensor,
        module_state: torch.Tensor,
        env_state: torch.Tensor,
        hyper_state: torch.Tensor,
        global_token: torch.Tensor,
        module_centers: torch.Tensor,
        module_present: torch.Tensor,
        hyper_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_norm = torch.stack(
            [
                query_xy[..., 0] / max(float(self.config.domain_length_x), EPS),
                query_xy[..., 1] / max(float(self.config.domain_length_y), EPS),
            ],
            dim=-1,
        )
        query_features = torch.cat(
            [
                self.query_fourier(query_norm),
                boundary_features(
                    query_xy,
                    domain_length_x=self.config.domain_length_x,
                    domain_length_y=self.config.domain_length_y,
                ),
            ],
            dim=-1,
        )
        query_state = self.query_encoder(query_features)
        memory = torch.cat([module_state, env_state, hyper_state], dim=1)
        batch = query_xy.shape[0]
        env_mask = torch.zeros(batch, env_state.shape[1], device=query_xy.device, dtype=torch.bool)
        if hyper_mask is None:
            hyper_pad = torch.zeros(batch, hyper_state.shape[1], device=query_xy.device, dtype=torch.bool)
        else:
            hyper_pad = ~hyper_mask
        key_padding_mask = torch.cat([module_present <= 0.5, env_mask, hyper_pad], dim=1)
        attended, _ = self.memory_attn(query_state, memory, memory, key_padding_mask=key_padding_mask, need_weights=False)
        near_context = self.nearby_module_context(query_xy, module_centers, module_state, module_present)
        global_context = global_token[:, None, :].expand(-1, query_xy.shape[1], -1)
        return self.output(torch.cat([query_state, attended, near_context, global_context], dim=-1))


class GlobalChannelThermalModel(nn.Module):
    """Global nonperiodic channel thermal neural field with hypergraph organization."""

    def __init__(self, config: GlobalChannelThermalModelConfig):
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        self.local_surrogate: Optional[LocalModuleSurrogate] = None

        self.global_encoder = MLP(
            2 + config.material_param_dim,
            hidden,
            hidden,
            num_layers=3,
            dropout=config.dropout,
            layer_norm=config.use_layer_norm,
        )
        self.module_encoder = MLP(
            2 + 1 + 1 + hidden + config.material_param_dim,
            hidden,
            hidden,
            num_layers=3,
            dropout=config.dropout,
            layer_norm=config.use_layer_norm,
        )
        self.local_latent_fusion = MLP(
            hidden + config.local_surrogate_latent_dim,
            hidden,
            hidden,
            num_layers=2,
            dropout=config.dropout,
            layer_norm=config.use_layer_norm,
        )
        env_coords, env_features = self._make_environment_tokens()
        self.register_buffer("env_coords", env_coords, persistent=False)
        self.register_buffer("env_features", env_features, persistent=False)
        self.env_encoder = MLP(env_features.shape[-1] + hidden, hidden, hidden, num_layers=3, dropout=config.dropout, layer_norm=True)
        self.organizer = HypergraphOrganizer(config)
        self.port_head = PortConditionHead(config)
        self.field_decoder = QueryFieldDecoder(config)
        self.local_coord_encoder = FourierEncoder(2, config.local_coord_fourier_frequencies, include_input=True)
        self.internal_head = MLP(
            hidden + self.local_coord_encoder.output_dim,
            hidden,
            1,
            num_layers=3,
            dropout=config.dropout,
            layer_norm=config.use_layer_norm,
        )
        self.interface_theta_encoder = FourierEncoder(3, 2, include_input=True)
        self.interface_head = MLP(
            hidden + self.interface_theta_encoder.output_dim,
            hidden,
            config.local_interface_target_dim,
            num_layers=3,
            dropout=config.dropout,
            layer_norm=config.use_layer_norm,
        )

    def _make_environment_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        nx = max(int(self.config.num_env_tokens_x), 1)
        ny = max(int(self.config.num_env_tokens_y), 1)
        xs = torch.linspace(0.0, float(self.config.domain_length_x), nx)
        ys = torch.linspace(0.0, float(self.config.domain_length_y), ny)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        lx = max(float(self.config.domain_length_x), EPS)
        ly = max(float(self.config.domain_length_y), EPS)
        x_norm = coords[:, 0:1] / lx
        y_norm = coords[:, 1:2] / ly
        wall = torch.minimum(coords[:, 1:2], ly - coords[:, 1:2]) / ly
        inlet = coords[:, 0:1] / lx
        outlet = (lx - coords[:, 0:1]) / lx
        centerline = 1.0 - torch.abs(2.0 * y_norm - 1.0)
        features = torch.cat([x_norm, y_norm, wall, inlet, outlet, centerline], dim=-1).float()
        return coords.float(), features

    def set_local_surrogate(self, model: LocalModuleSurrogate, *, freeze: bool = True) -> None:
        self.local_surrogate = model
        if freeze:
            self.local_surrogate.eval()
            for param in self.local_surrogate.parameters():
                param.requires_grad_(False)

    def encode_global(
        self,
        re: torch.Tensor,
        u_in: torch.Tensor,
        material_params: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if re.ndim == 1:
            re = re[:, None]
        if u_in.ndim == 1:
            u_in = u_in[:, None]
        batch = re.shape[0]
        if material_params is None:
            material_params = re.new_zeros(batch, self.config.material_param_dim)
        global_inputs = torch.cat([re.float(), u_in.float(), material_params.float()], dim=-1)
        return self.global_encoder(global_inputs)

    def encode_modules(
        self,
        module_centers: torch.Tensor,
        heat_powers: torch.Tensor,
        module_present: torch.Tensor,
        global_token: torch.Tensor,
        material_params: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, num_modules, _ = module_centers.shape
        lx = max(float(self.config.domain_length_x), EPS)
        ly = max(float(self.config.domain_length_y), EPS)
        centers_norm = torch.stack([module_centers[..., 0] / lx, module_centers[..., 1] / ly], dim=-1)
        if material_params is None:
            material_params = module_centers.new_zeros(batch, self.config.material_param_dim)
        module_inputs = torch.cat(
            [
                centers_norm,
                heat_powers[..., None],
                module_present[..., None],
                global_token[:, None, :].expand(-1, num_modules, -1),
                material_params[:, None, :].expand(-1, num_modules, -1),
            ],
            dim=-1,
        )
        return self.module_encoder(module_inputs) * module_present[..., None]

    def encode_environment(self, global_token: torch.Tensor) -> torch.Tensor:
        batch = global_token.shape[0]
        env = self.env_features.to(device=global_token.device, dtype=global_token.dtype)
        env = env[None, :, :].expand(batch, -1, -1)
        global_context = global_token[:, None, :].expand(-1, env.shape[1], -1)
        return self.env_encoder(torch.cat([env, global_context], dim=-1))

    def _predict_global_internal(
        self,
        module_state: torch.Tensor,
        local_query_points: Optional[torch.Tensor],
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        if local_query_points is None:
            return module_state.new_empty(module_state.shape[0], module_state.shape[1], 0, 1)
        if local_query_points.ndim == 3:
            local_query_points = local_query_points[:, None, :, :].expand(-1, module_state.shape[1], -1, -1)
        batch, num_modules, num_points, _ = local_query_points.shape
        coord_features = self.local_coord_encoder(local_query_points)
        module_context = module_state[:, :, None, :].expand(-1, -1, num_points, -1)
        pred = self.internal_head(torch.cat([module_context, coord_features], dim=-1))
        return pred * module_present[:, :, None, None]

    def _predict_global_interface(
        self,
        module_state: torch.Tensor,
        *,
        ntheta: int,
        module_present: torch.Tensor,
    ) -> torch.Tensor:
        theta_tokens = self.port_head.fixed_theta_tokens(ntheta, module_state.device, module_state.dtype)
        theta_features = self.interface_theta_encoder(theta_tokens).view(1, 1, ntheta, -1)
        theta_features = theta_features.expand(module_state.shape[0], module_state.shape[1], -1, -1)
        module_context = module_state[:, :, None, :].expand(-1, -1, ntheta, -1)
        pred = self.interface_head(torch.cat([module_context, theta_features], dim=-1))
        return pred * module_present[:, :, None, None]

    def _call_local_surrogate(
        self,
        module_params: torch.Tensor,
        port_tokens: torch.Tensor,
        local_query_points: Optional[torch.Tensor],
        module_present: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.local_surrogate is None:
            raise RuntimeError("use_local_surrogate=True but no LocalModuleSurrogate has been attached.")
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
        out = self.local_surrogate(flat_params, flat_ports, flat_query)
        present = module_present[:, :, None, None]
        internal = out["internal_temperature"].reshape(batch, num_modules, -1, 1) * present
        interface = out["interface_pred"].reshape(batch, num_modules, ntheta, -1) * present
        latent = out["module_response_latent"].reshape(batch, num_modules, -1) * module_present[..., None]
        return {
            "internal_temperature": internal,
            "interface_pred": interface,
            "module_response_latent": latent,
        }

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
        local_port_condition_mode: str = "teacher",
        mixed_teacher_ratio: float = 0.5,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        if structure is not None:
            re = structure.get("re", re)
            u_in = structure.get("u_in", u_in)
            module_centers = structure.get("module_centers", module_centers)
            heat_powers = structure.get("heat_powers", heat_powers)
            module_present = structure.get("module_present", module_present)
            material_params = structure.get("material_params", material_params)
        if query_xy is None:
            raise ValueError("query_xy is required.")
        if re is None or u_in is None or module_centers is None or heat_powers is None or module_present is None:
            raise ValueError("re, u_in, module_centers, heat_powers, and module_present are required.")

        query_xy = query_xy.float()
        module_centers = module_centers.float()
        heat_powers = heat_powers.float()
        module_present = module_present.float()
        material_params = material_params.float() if material_params is not None else None
        global_token = self.encode_global(re.float(), u_in.float(), material_params)
        base_module_state = self.encode_modules(module_centers, heat_powers, module_present, global_token, material_params)
        env_state = self.encode_environment(global_token)
        env_coords = self.env_coords.to(device=query_xy.device, dtype=query_xy.dtype)
        base_org = self.organizer(base_module_state, env_state, module_centers, env_coords, module_present)

        if teacher_port_tokens is None and interface_condition is not None:
            teacher_port_tokens = teacher_port_tokens_from_interface_condition(interface_condition.float())
        ntheta = (
            int(teacher_port_tokens.shape[-2])
            if teacher_port_tokens is not None
            else int(interface_condition.shape[-2])
            if interface_condition is not None
            else int(self.config.default_num_interface_points)
        )
        pred_port_tokens = self.port_head(
            base_module_state,
            base_org["module_env_context"],
            heat_powers,
            global_token,
            ntheta=ntheta,
            module_present=module_present,
        )

        local_outputs: Optional[Dict[str, torch.Tensor]] = None
        module_state = base_module_state
        if self.config.use_local_surrogate:
            if local_module_params is None:
                local_module_params = build_local_module_params_from_global(
                    heat_powers,
                    interface_condition.float() if interface_condition is not None else None,
                    material_params,
                    module_present,
                )
            mode = str(local_port_condition_mode).lower()
            if mode == "teacher" and teacher_port_tokens is not None:
                local_ports = teacher_port_tokens.float()
            elif mode == "mixed" and teacher_port_tokens is not None:
                ratio = float(mixed_teacher_ratio)
                local_ports = ratio * teacher_port_tokens.float() + (1.0 - ratio) * pred_port_tokens
            else:
                local_ports = pred_port_tokens
            local_outputs = self._call_local_surrogate(local_module_params.float(), local_ports, local_query_points, module_present)
            fused = self.local_latent_fusion(torch.cat([base_module_state, local_outputs["module_response_latent"]], dim=-1))
            module_state = (base_module_state + fused) * module_present[..., None]

        org = self.organizer(module_state, env_state, module_centers, env_coords, module_present)
        pred_field = self.field_decoder(
            query_xy,
            module_state,
            env_state,
            org["hyper_state"],
            global_token,
            module_centers,
            module_present,
            org.get("active_hyperedge_mask"),
        )

        if local_outputs is not None:
            pred_internal = local_outputs["internal_temperature"]
            pred_interface = local_outputs["interface_pred"]
            module_response_latent = local_outputs["module_response_latent"]
        else:
            pred_internal = self._predict_global_internal(module_state, local_query_points, module_present)
            pred_interface = self._predict_global_interface(module_state, ntheta=ntheta, module_present=module_present)
            module_response_latent = module_state

        return {
            "pred_field": pred_field,
            "pred_internal_temperature": pred_internal,
            "pred_interface": pred_interface,
            "pred_port_condition": pred_port_tokens,
            "module_response_latent": module_response_latent,
            "organizer_aux": org,
            "base_organizer_aux": base_org,
        }


def build_model_from_config(config_payload: Dict | GlobalChannelThermalModelConfig) -> GlobalChannelThermalModel:
    config = config_payload if isinstance(config_payload, GlobalChannelThermalModelConfig) else GlobalChannelThermalModelConfig.from_dict(config_payload)
    return GlobalChannelThermalModel(config)


def load_local_surrogate_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[LocalModuleSurrogate, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    config_payload = checkpoint.get("model_config") or checkpoint.get("local_model_config") or {}
    config = LocalModuleConfig.from_dict(config_payload)
    model = LocalModuleSurrogate(config)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    model.load_state_dict(strip_module_prefix(state), strict=True)
    return model, checkpoint
