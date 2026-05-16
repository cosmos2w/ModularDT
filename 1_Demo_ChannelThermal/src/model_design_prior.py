from __future__ import annotations

"""Target-agnostic latent design prior for ChannelThermal layouts.

This model is a behavior-aware design atlas, not a KPI-conditioned inverse
generator. It learns a compact latent manifold of valid-ish layouts, realized
or planned hypergraph descriptors, and compact behavior descriptors. Downstream
inverse design should search over ``z`` with a frozen forward verifier and a
field-functional objective.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DesignPriorConfig:
    max_num_modules: int = 12
    design_dim: Optional[int] = None
    hypergraph_dim: int = 0
    behavior_dim: int = 32
    context_dim: int = 0
    latent_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.05
    generate_heat_power: bool = False
    kl_weight: float = 1e-3
    layout_recon_weight: float = 1.0
    hypergraph_recon_weight: float = 0.5
    behavior_recon_weight: float = 0.5
    geometry_weight: float = 0.05
    latent_l2_weight: float = 0.0
    domain_length_x: float = 12.0
    domain_length_y: float = 4.0
    module_radius: float = 0.45
    min_center_distance: float = 1.1
    wall_clearance: float = 0.05
    inlet_clearance: float = 0.25
    outlet_clearance: float = 0.25
    center_decode_mode: str = "sigmoid"

    def __post_init__(self) -> None:
        self.max_num_modules = int(self.max_num_modules)
        if self.design_dim is None:
            self.design_dim = self.max_num_modules * (4 if self.generate_heat_power else 3)
        self.design_dim = int(self.design_dim)
        self.hypergraph_dim = max(int(self.hypergraph_dim or 0), 0)
        self.behavior_dim = max(int(self.behavior_dim or 0), 0)
        self.context_dim = max(int(self.context_dim or 0), 0)
        self.latent_dim = int(self.latent_dim)
        self.hidden_dim = int(self.hidden_dim)
        self.num_layers = max(int(self.num_layers), 2)
        mode = str(self.center_decode_mode or "sigmoid").lower().strip()
        if mode not in {"sigmoid", "clamp", "identity"}:
            raise ValueError("center_decode_mode must be one of sigmoid, clamp, identity.")
        self.center_decode_mode = mode

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DesignPriorConfig":
        return cls(**{key: value for key, value in dict(payload).items() if key in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


class LatentModularDesignPrior(nn.Module):
    """Target-agnostic behavior-aware design atlas.

    Learns p(D, H, b | context) using a compact VAE latent ``z``. It does not
    take KPI targets. Layout reconstruction is valid here because this is
    prior/atlas learning over observed designs, not supervised inverse learning
    for a unique downstream target.
    """

    def __init__(self, cfg: DesignPriorConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, DesignPriorConfig) else DesignPriorConfig.from_dict(cfg)
        enc_in = int(self.cfg.design_dim) + int(self.cfg.hypergraph_dim) + int(self.cfg.behavior_dim) + int(self.cfg.context_dim)
        dec_in = int(self.cfg.latent_dim) + int(self.cfg.context_dim)
        self.encoder = MLP(enc_in, self.cfg.hidden_dim, 2 * self.cfg.latent_dim, num_layers=self.cfg.num_layers, dropout=self.cfg.dropout)
        self.decoder_trunk = MLP(dec_in, self.cfg.hidden_dim, self.cfg.hidden_dim, num_layers=self.cfg.num_layers, dropout=self.cfg.dropout)
        self.design_head = nn.Linear(self.cfg.hidden_dim, self.cfg.design_dim)
        self.hypergraph_head = nn.Linear(self.cfg.hidden_dim, self.cfg.hypergraph_dim) if self.cfg.hypergraph_dim > 0 else None
        self.behavior_head = nn.Linear(self.cfg.hidden_dim, self.cfg.behavior_dim) if self.cfg.behavior_dim > 0 else None

    def _zeros(self, batch: int, dim: int, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros((int(batch), int(dim)))

    def _context(self, context_vec: Optional[torch.Tensor], batch: int, ref: torch.Tensor) -> torch.Tensor:
        if self.cfg.context_dim <= 0:
            return ref.new_zeros((batch, 0))
        if context_vec is None:
            return self._zeros(batch, self.cfg.context_dim, ref)
        return context_vec.float().reshape(batch, self.cfg.context_dim)

    def encode(
        self,
        design_vec: torch.Tensor,
        hypergraph_vec: Optional[torch.Tensor] = None,
        behavior_vec: Optional[torch.Tensor] = None,
        context_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        design = design_vec.float().reshape(design_vec.shape[0], self.cfg.design_dim)
        batch = design.shape[0]
        parts = [design]
        if self.cfg.hypergraph_dim > 0:
            parts.append(hypergraph_vec.float().reshape(batch, self.cfg.hypergraph_dim) if hypergraph_vec is not None else self._zeros(batch, self.cfg.hypergraph_dim, design))
        if self.cfg.behavior_dim > 0:
            parts.append(behavior_vec.float().reshape(batch, self.cfg.behavior_dim) if behavior_vec is not None else self._zeros(batch, self.cfg.behavior_dim, design))
        parts.append(self._context(context_vec, batch, design))
        stats = self.encoder(torch.cat(parts, dim=-1))
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        return mu, torch.clamp(logvar, min=-12.0, max=8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def decode_latent(self, z: torch.Tensor, context_vec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch = z.shape[0]
        context = self._context(context_vec, batch, z)
        hidden = self.decoder_trunk(torch.cat([z.float(), context], dim=-1))
        raw_design = self.design_head(hidden)
        if self.cfg.center_decode_mode == "sigmoid":
            design_vec = torch.sigmoid(raw_design)
        elif self.cfg.center_decode_mode == "clamp":
            design_vec = torch.clamp(raw_design, 0.0, 1.0)
        else:
            design_vec = raw_design
        out: Dict[str, torch.Tensor] = {"design_vec": design_vec}
        if self.hypergraph_head is not None:
            out["hypergraph_vec"] = self.hypergraph_head(hidden)
        if self.behavior_head is not None:
            out["behavior_vec"] = self.behavior_head(hidden)
        return out

    def sample(
        self,
        context_vec: Optional[torch.Tensor],
        num_samples: int,
        temperature: float = 1.0,
        device: Optional[torch.device | str] = None,
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            if context_vec is not None:
                device = context_vec.device
            else:
                device = next(self.parameters()).device
        dev = torch.device(device)
        n = int(num_samples)
        if context_vec is not None and context_vec.shape[0] == 1 and n > 1:
            context_vec = context_vec.to(dev).expand(n, -1)
        elif context_vec is not None:
            context_vec = context_vec.to(dev)
            n = int(context_vec.shape[0]) if n <= 0 else n
            if context_vec.shape[0] != n:
                context_vec = context_vec[:1].expand(n, -1)
        z = torch.randn(n, self.cfg.latent_dim, device=dev) * float(temperature)
        return self.decode_latent(z, context_vec)

    def latent_prior_energy(self, z: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(z.float() ** 2, dim=-1)

    def _split_design(self, design_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        m = int(self.cfg.max_num_modules)
        design = design_vec.float()
        centers_norm = design[:, : 2 * m].reshape(-1, m, 2)
        mask = design[:, 2 * m : 3 * m].reshape(-1, m)
        heat = design[:, 3 * m : 4 * m].reshape(-1, m) if self.cfg.generate_heat_power and design.shape[-1] >= 4 * m else None
        return centers_norm, mask, heat

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

    def _weighted_mean(self, values: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
        per_sample = values.reshape(values.shape[0], -1).mean(dim=-1)
        if sample_weight is None:
            return per_sample.mean()
        w = sample_weight.float().reshape(-1)
        return torch.sum(per_sample * w) / torch.clamp(torch.sum(w), min=1.0e-8)

    def training_loss(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        design = batch["design_vec"].float()
        hyper = batch.get("hypergraph_vec")
        behavior = batch.get("behavior_vec")
        context = batch.get("context_vec")
        sample_weight = batch.get("sample_weight")
        mu, logvar = self.encode(design, hyper, behavior, context)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode_latent(z, context)
        layout_recon = self._weighted_mean((decoded["design_vec"] - design) ** 2, sample_weight)
        hyper_recon = design.new_tensor(0.0)
        if self.cfg.hypergraph_dim > 0 and hyper is not None and "hypergraph_vec" in decoded:
            diff = (decoded["hypergraph_vec"] - hyper.float()) ** 2
            mask = batch.get("hypergraph_mask")
            if mask is not None:
                diff = diff * mask.float()
            hyper_recon = self._weighted_mean(diff, sample_weight)
        behavior_recon = design.new_tensor(0.0)
        if self.cfg.behavior_dim > 0 and behavior is not None and "behavior_vec" in decoded:
            behavior_recon = self._weighted_mean((decoded["behavior_vec"] - behavior.float()) ** 2, sample_weight)
        kl_per = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        if sample_weight is not None:
            w = sample_weight.float().reshape(-1)
            kl_loss = torch.sum(kl_per * w) / torch.clamp(torch.sum(w), min=1.0e-8)
        else:
            kl_loss = kl_per.mean()
        geometry_loss = self._geometry_loss(decoded["design_vec"])
        latent_l2 = torch.mean(self.latent_prior_energy(z))
        total = (
            float(self.cfg.layout_recon_weight) * layout_recon
            + float(self.cfg.hypergraph_recon_weight) * hyper_recon
            + float(self.cfg.behavior_recon_weight) * behavior_recon
            + float(self.cfg.kl_weight) * kl_loss
            + float(self.cfg.geometry_weight) * geometry_loss
            + float(self.cfg.latent_l2_weight) * latent_l2
        )
        return {
            "loss_total": total,
            "layout_recon_loss": layout_recon,
            "hypergraph_recon_loss": hyper_recon,
            "behavior_recon_loss": behavior_recon,
            "kl_loss": kl_loss,
            "geometry_loss": geometry_loss,
            "latent_l2_loss": latent_l2,
            "mu_mean": mu.mean(),
            "mu_std": mu.std(unbiased=False),
        }
