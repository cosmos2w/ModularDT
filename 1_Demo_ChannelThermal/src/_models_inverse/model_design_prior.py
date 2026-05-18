from __future__ import annotations

"""Mechanism-conditioned layout realization for ChannelThermal design priors.

The old latent atlas learned ``z -> D,H,b``.  This module instead provides a
conditional rectified-flow realizer:

    noise + mechanism M + context c -> layout D

where ``M`` is supplied by a hypergraph/behavior mechanism atlas discovered
from the frozen forward HONF mechanism library.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MechanismLayoutRealizerConfig:
    max_num_modules: int = 12
    design_dim: Optional[int] = None
    mechanism_dim: int = 0
    hypergraph_dim: int = 0
    behavior_dim: int = 0
    context_dim: int = 0

    hidden_dim: int = 256
    condition_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.05

    flow_steps_default: int = 16
    center_decode_mode: str = "clamp"
    generate_heat_power: bool = False

    layout_flow_weight: float = 1.0
    mask_component_weight: float = 2.0
    active_center_weight: float = 2.0
    inactive_center_weight: float = 0.05
    geometry_weight: float = 0.05
    count_weight: float = 0.2

    domain_length_x: float = 12.0
    domain_length_y: float = 4.0
    module_radius: float = 0.45
    min_center_distance: float = 1.1
    wall_clearance: float = 0.05
    inlet_clearance: float = 0.25
    outlet_clearance: float = 0.25

    def __post_init__(self) -> None:
        self.max_num_modules = int(self.max_num_modules)
        if self.design_dim is None:
            self.design_dim = self.max_num_modules * (4 if self.generate_heat_power else 3)
        self.design_dim = int(self.design_dim)
        self.mechanism_dim = max(int(self.mechanism_dim or 0), 0)
        self.hypergraph_dim = max(int(self.hypergraph_dim or 0), 0)
        self.behavior_dim = max(int(self.behavior_dim or 0), 0)
        self.context_dim = max(int(self.context_dim or 0), 0)
        self.hidden_dim = int(self.hidden_dim)
        self.condition_dim = int(self.condition_dim)
        self.num_layers = max(int(self.num_layers), 2)
        self.flow_steps_default = max(int(self.flow_steps_default), 1)
        mode = str(self.center_decode_mode or "clamp").lower().strip()
        if mode not in {"clamp", "sigmoid", "identity"}:
            raise ValueError("center_decode_mode must be one of clamp, sigmoid, identity.")
        self.center_decode_mode = mode

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MechanismLayoutRealizerConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in dict(payload).items() if key in fields})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DesignPriorConfig(MechanismLayoutRealizerConfig):
    """Compatibility config name for older design-prior training imports."""


def _activation() -> nn.Module:
    return nn.SiLU()


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers = []
        last = int(in_dim)
        for _ in range(max(int(num_layers) - 1, 0)):
            layers.append(nn.Linear(last, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(_activation())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            last = int(hidden_dim)
        layers.append(nn.Linear(last, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    t_flat = t.float().reshape(-1, 1)
    if dim <= 1:
        return t_flat
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0.0, 1.0, half, device=t_flat.device, dtype=t_flat.dtype)
        * torch.log(t_flat.new_tensor(1000.0))
    )
    emb = torch.cat([torch.sin(t_flat * freqs), torch.cos(t_flat * freqs)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb[:, :dim]


class MechanismConditionEncoder(nn.Module):
    """Encode mechanism feature plus optional context into a condition vector."""

    def __init__(self, cfg: MechanismLayoutRealizerConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, MechanismLayoutRealizerConfig) else MechanismLayoutRealizerConfig.from_dict(cfg)
        in_dim = int(self.cfg.mechanism_dim) + int(self.cfg.context_dim)
        self.net = MLP(in_dim, self.cfg.hidden_dim, self.cfg.condition_dim, num_layers=self.cfg.num_layers, dropout=self.cfg.dropout)

    def _context(self, context_vec: Optional[torch.Tensor], batch: int, ref: torch.Tensor) -> torch.Tensor:
        if self.cfg.context_dim <= 0:
            return ref.new_zeros((batch, 0))
        if context_vec is None:
            return ref.new_zeros((batch, self.cfg.context_dim))
        return context_vec.float().reshape(batch, self.cfg.context_dim)

    def forward(self, mechanism_feature: torch.Tensor, context_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        mech = mechanism_feature.float().reshape(mechanism_feature.shape[0], self.cfg.mechanism_dim)
        context = self._context(context_vec, mech.shape[0], mech)
        return self.net(torch.cat([mech, context], dim=-1))


class ConditionalLayoutVelocityNet(nn.Module):
    """Velocity field over design vectors conditioned on mechanism embeddings."""

    def __init__(self, cfg: MechanismLayoutRealizerConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, MechanismLayoutRealizerConfig) else MechanismLayoutRealizerConfig.from_dict(cfg)
        time_dim = int(self.cfg.condition_dim)
        in_dim = int(self.cfg.design_dim) + int(self.cfg.condition_dim) + time_dim
        self.net = MLP(in_dim, self.cfg.hidden_dim, self.cfg.design_dim, num_layers=self.cfg.num_layers, dropout=self.cfg.dropout)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = x_t.float().reshape(x_t.shape[0], self.cfg.design_dim)
        time = _time_embedding(t, int(self.cfg.condition_dim))
        return self.net(torch.cat([x, time, condition.float()], dim=-1))


class HypergraphConditionedLayoutRealizer(nn.Module):
    """Conditional rectified-flow layout realizer ``p(D | M, c)``.

    ``M`` is a hypergraph/behavior mechanism feature discovered from the
    forward HONF mechanism library. The model samples layouts that should
    realize the desired mechanism after frozen-forward verification.
    """

    def __init__(self, cfg: MechanismLayoutRealizerConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, MechanismLayoutRealizerConfig) else MechanismLayoutRealizerConfig.from_dict(cfg)
        self.condition_encoder = MechanismConditionEncoder(self.cfg)
        self.velocity_net = ConditionalLayoutVelocityNet(self.cfg)

    def encode_condition(self, mechanism_feature: torch.Tensor, context_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.condition_encoder(mechanism_feature, context_vec)

    def velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mechanism_feature: torch.Tensor,
        context_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(mechanism_feature, context_vec)
        return self.velocity_net(x_t, t, condition)

    def _split_design(self, design_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        m = int(self.cfg.max_num_modules)
        design = design_vec.float().reshape(design_vec.shape[0], -1)
        if design.shape[-1] < 3 * m:
            design = F.pad(design, (0, 3 * m - design.shape[-1]))
        centers_norm = design[:, : 2 * m].reshape(-1, m, 2)
        mask = design[:, 2 * m : 3 * m].reshape(-1, m)
        heat = design[:, 3 * m : 4 * m].reshape(-1, m) if self.cfg.generate_heat_power and design.shape[-1] >= 4 * m else None
        return centers_norm, mask, heat

    def _component_weights(self, target_design: torch.Tensor) -> torch.Tensor:
        m = int(self.cfg.max_num_modules)
        target = target_design.float().reshape(target_design.shape[0], -1)
        weights = torch.ones_like(target)
        if target.shape[-1] < 3 * m:
            return weights
        active = (target[:, 2 * m : 3 * m] > 0.5).float()
        center_w = active * float(self.cfg.active_center_weight) + (1.0 - active) * float(self.cfg.inactive_center_weight)
        weights[:, : 2 * m] = center_w.repeat_interleave(2, dim=-1)
        weights[:, 2 * m : 3 * m] = float(self.cfg.mask_component_weight)
        if self.cfg.generate_heat_power and target.shape[-1] >= 4 * m:
            weights[:, 3 * m : 4 * m] = 1.0
        return weights

    def _weighted_mean(self, values: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
        per_sample = values.reshape(values.shape[0], -1).mean(dim=-1)
        if sample_weight is None:
            return per_sample.mean()
        w = sample_weight.float().reshape(-1)
        return torch.sum(per_sample * w) / torch.clamp(torch.sum(w), min=1.0e-8)

    def _geometry_loss(self, design_vec: torch.Tensor) -> torch.Tensor:
        centers_norm, mask_raw, heat = self._split_design(design_vec)
        mask = torch.sigmoid(8.0 * (mask_raw - 0.5))
        centers = torch.empty_like(centers_norm)
        centers[..., 0] = centers_norm[..., 0] * float(self.cfg.domain_length_x)
        centers[..., 1] = centers_norm[..., 1] * float(self.cfg.domain_length_y)
        radius = float(self.cfg.module_radius)
        x_min = radius + float(self.cfg.inlet_clearance)
        x_max = float(self.cfg.domain_length_x) - radius - float(self.cfg.outlet_clearance)
        y_min = radius + float(self.cfg.wall_clearance)
        y_max = float(self.cfg.domain_length_y) - radius - float(self.cfg.wall_clearance)
        boundary = (
            F.relu(x_min - centers[..., 0])
            + F.relu(centers[..., 0] - x_max)
            + F.relu(y_min - centers[..., 1])
            + F.relu(centers[..., 1] - y_max)
        )
        loss = torch.mean(boundary * mask)
        if centers.shape[1] >= 2:
            diff = centers[:, :, None, :] - centers[:, None, :, :]
            dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1.0e-8)
            pair_mask = mask[:, :, None] * mask[:, None, :]
            eye = torch.eye(centers.shape[1], device=centers.device, dtype=centers.dtype)[None]
            overlap = F.relu(float(self.cfg.min_center_distance) - dist) * pair_mask * (1.0 - eye)
            loss = loss + 0.5 * torch.mean(overlap)
        if heat is not None:
            loss = loss + torch.mean((1.0 - mask) * heat**2)
        return loss

    def _mask_and_count_losses(
        self,
        endpoint: torch.Tensor,
        target: torch.Tensor,
        true_count: Optional[torch.Tensor],
        sample_weight: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = int(self.cfg.max_num_modules)
        if endpoint.shape[-1] < 3 * m or target.shape[-1] < 3 * m:
            zero = target.new_tensor(0.0)
            return zero, zero
        pred_mask = endpoint[:, 2 * m : 3 * m]
        target_mask = target[:, 2 * m : 3 * m]
        mask_loss = self._weighted_mean((pred_mask - target_mask) ** 2, sample_weight)
        if true_count is None:
            count_target = torch.sum((target_mask > 0.5).float(), dim=-1)
        else:
            count_target = true_count.float().reshape(-1)
        count_pred = torch.sum(torch.clamp(pred_mask, 0.0, 1.0), dim=-1)
        count_loss_per = ((count_pred - count_target) / max(float(m), 1.0)) ** 2
        if sample_weight is not None:
            w = sample_weight.float().reshape(-1)
            count_loss = torch.sum(count_loss_per * w) / torch.clamp(torch.sum(w), min=1.0e-8)
        else:
            count_loss = count_loss_per.mean()
        return mask_loss, count_loss

    def training_loss(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        design = batch["design_vec"].float().reshape(batch["design_vec"].shape[0], self.cfg.design_dim)
        mechanism = batch["mechanism_feature"].float().reshape(design.shape[0], self.cfg.mechanism_dim)
        context = batch.get("context_vec")
        sample_weight = batch.get("sample_weight")
        z = torch.randn_like(design)
        t = torch.rand((design.shape[0], 1), device=design.device, dtype=design.dtype)
        x_t = (1.0 - t) * z + t * design
        v_target = design - z
        v_hat = self.velocity(x_t, t, mechanism, context)
        component_weights = self._component_weights(design)
        flow_loss = self._weighted_mean(((v_hat - v_target) ** 2) * component_weights, sample_weight)
        endpoint = x_t + (1.0 - t) * v_hat
        mask_loss, count_loss = self._mask_and_count_losses(endpoint, design, batch.get("true_count", batch.get("count_vec")), sample_weight)
        geometry_loss = self._geometry_loss(endpoint)
        total = (
            float(self.cfg.layout_flow_weight) * flow_loss
            + mask_loss
            + float(self.cfg.count_weight) * count_loss
            + float(self.cfg.geometry_weight) * geometry_loss
        )
        return {
            "loss_total": total,
            "layout_flow_loss": flow_loss,
            "mask_loss": mask_loss,
            "count_loss": count_loss,
            "geometry_loss": geometry_loss,
            "velocity_norm": torch.mean(torch.linalg.norm(v_hat, dim=-1)),
        }

    @torch.no_grad()
    def sample_layout(
        self,
        mechanism_feature: torch.Tensor,
        context_vec: Optional[torch.Tensor] = None,
        *,
        num_samples: int = 1,
        steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        mech = mechanism_feature.float()
        if mech.ndim == 1:
            mech = mech[None, :]
        n = int(num_samples if num_samples is not None else mech.shape[0])
        if mech.shape[0] == 1 and n > 1:
            mech = mech.expand(n, -1)
        elif mech.shape[0] != n:
            n = int(mech.shape[0])
        if context_vec is not None:
            context_vec = context_vec.to(device=mech.device, dtype=mech.dtype)
            if context_vec.ndim == 1:
                context_vec = context_vec[None, :]
            if context_vec.shape[0] == 1 and n > 1:
                context_vec = context_vec.expand(n, -1)
            elif context_vec.shape[0] != n:
                context_vec = context_vec[:1].expand(n, -1)
        step_count = max(int(steps or self.cfg.flow_steps_default), 1)
        x = torch.randn((n, int(self.cfg.design_dim)), device=mech.device, dtype=mech.dtype) * float(temperature)
        dt = 1.0 / float(step_count)
        for idx in range(step_count):
            t = torch.full((n, 1), float(idx) / float(step_count), device=x.device, dtype=x.dtype)
            x = x + dt * self.velocity(x, t, mech, context_vec)
        if self.cfg.center_decode_mode == "clamp":
            x = torch.clamp(x, 0.0, 1.0)
        elif self.cfg.center_decode_mode == "sigmoid":
            x = torch.sigmoid(x)
        return {
            "design_vec": x,
            "mechanism_feature": mech,
            "steps": torch.tensor(step_count, device=x.device),
            "temperature": torch.tensor(float(temperature), device=x.device, dtype=x.dtype),
        }


class LatentModularDesignPrior(nn.Module):
    """Deprecated compatibility symbol for the removed VAE-style prior."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        raise RuntimeError(
            "LatentModularDesignPrior has been replaced by "
            "HypergraphMechanismAtlas + HypergraphConditionedLayoutRealizer."
        )
