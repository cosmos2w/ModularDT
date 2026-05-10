from __future__ import annotations

"""Amortized steady inverse generator for ChannelThermal layouts."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from thermal_inverse_kpi import CONSTRAINT_NAMES


def _activation(name: str = "silu") -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    return nn.SiLU()


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int = 3,
        dropout: float = 0.0,
        activation: str = "silu",
        final_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        width = int(hidden_dim)
        last = int(in_dim)
        for _ in range(max(int(num_layers) - 1, 0)):
            layers.append(nn.Linear(last, width))
            layers.append(nn.LayerNorm(width))
            layers.append(_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            last = width
        layers.append(nn.Linear(last, int(out_dim)))
        if final_norm:
            layers.append(nn.LayerNorm(int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        half = max(self.dim // 2, 1)
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half, device=t.device, dtype=t.dtype))
        emb = torch.cat([torch.sin(t * freqs[None, :]), torch.cos(t * freqs[None, :])], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[:, : self.dim]


class TargetEncoder(nn.Module):
    def __init__(self, target_dim: int, hidden_dim: int, target_embed_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.net = MLP(target_dim, hidden_dim, target_embed_dim, num_layers=max(num_layers, 2), dropout=dropout, final_norm=True)

    def forward(self, target_spec_vector: torch.Tensor) -> torch.Tensor:
        return self.net(target_spec_vector)


class BehaviorOrgHead(nn.Module):
    def __init__(
        self,
        target_embed_dim: int,
        hidden_dim: int,
        behavior_latent_dim: int,
        organization_latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.behavior = MLP(target_embed_dim, hidden_dim, behavior_latent_dim, num_layers=3, dropout=dropout)
        self.organization = MLP(target_embed_dim, hidden_dim, organization_latent_dim, num_layers=3, dropout=dropout)

    def forward(self, target_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.behavior(target_embedding), self.organization(target_embedding)


class DesignVelocityNet(nn.Module):
    def __init__(
        self,
        design_dim: int,
        target_embed_dim: int,
        behavior_latent_dim: int,
        organization_latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        time_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        in_dim = int(design_dim) + int(target_embed_dim) + int(behavior_latent_dim) + int(organization_latent_dim) + int(time_embed_dim)
        self.net = MLP(in_dim, hidden_dim, design_dim, num_layers=max(num_layers, 2), dropout=dropout)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        target_embedding: torch.Tensor,
        behavior_latent_hat: torch.Tensor,
        organization_latent_hat: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        cond = torch.cat([target_embedding, behavior_latent_hat, organization_latent_hat, t_emb], dim=-1)
        return self.net(torch.cat([x_t, cond], dim=-1))


@dataclass
class InverseModelConfig:
    target_dim: int
    max_num_modules: int = 12
    design_dim: Optional[int] = None
    hidden_dim: int = 256
    target_embed_dim: int = 128
    behavior_latent_dim: int = 96
    organization_latent_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.05
    use_count_head: bool = True
    generate_heat_power: bool = False
    heat_load_policy: str = "preserve_total_heat"
    heat_power_scale: float = 1.0
    ode_solver: str = "heun"
    domain_length_x: float = 12.0
    domain_length_y: float = 4.0
    module_radius: float = 0.45
    min_num_modules: int = 1
    min_center_distance: float = 1.1
    wall_clearance: float = 0.05
    inlet_clearance: float = 0.25
    outlet_clearance: float = 0.25
    center_decode_mode: str = "sigmoid"
    latent_align_completeness_power: float = 0.5

    def __post_init__(self) -> None:
        self.max_num_modules = int(self.max_num_modules)
        expected = self.max_num_modules * (4 if self.generate_heat_power else 3)
        if self.design_dim is None:
            self.design_dim = expected
        self.design_dim = int(self.design_dim)
        if self.design_dim != expected:
            raise ValueError(f"design_dim must be {expected} for max_num_modules={self.max_num_modules}, generate_heat_power={self.generate_heat_power}.")
        mode = str(self.center_decode_mode).lower().strip()
        if mode not in {"clamp", "sigmoid"}:
            raise ValueError("center_decode_mode must be 'clamp' or 'sigmoid' for the nonperiodic channel.")
        self.center_decode_mode = mode
        policy = str(self.heat_load_policy).lower().strip()
        allowed_policies = {
            "preserve_total_heat",
            "preserve_per_module_heat",
            "reference_active_heat_resize",
            "fixed_heat_per_module",
            "target_heat_power_total",
        }
        if policy not in allowed_policies:
            raise ValueError(f"heat_load_policy must be one of {sorted(allowed_policies)}, got {self.heat_load_policy!r}.")
        self.heat_load_policy = policy

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InverseModelConfig":
        data = dict(payload)
        for key in ("target_dim", "design_dim", "behavior_latent_dim", "organization_latent_dim"):
            if str(data.get(key, "")).lower() == "auto":
                data.pop(key, None)
        if "target_dim" not in data:
            raise ValueError("InverseModelConfig requires target_dim after config resolution.")
        valid = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in valid})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sort_centers_xy(centers: np.ndarray) -> np.ndarray:
    arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] <= 1:
        return arr
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def active_min_distance(centers: np.ndarray) -> float:
    arr = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 2:
        return float("inf")
    best = float("inf")
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            best = min(best, float(np.linalg.norm(arr[i] - arr[j])))
    return best


def encode_design_vector(
    centers: np.ndarray,
    mask: Optional[np.ndarray] = None,
    heat_powers: Optional[np.ndarray] = None,
    *,
    max_num_modules: int,
    domain_length_x: float,
    domain_length_y: float,
    generate_heat_power: bool = False,
    heat_power_scale: float = 1.0,
    sort_centers: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    mask_arr = np.asarray(mask, dtype=np.float32).reshape(-1) if mask is not None else np.ones((centers_arr.shape[0],), dtype=np.float32)
    active_idx = np.flatnonzero(mask_arr > 0.5)
    active_centers = centers_arr[active_idx] if active_idx.size else np.zeros((0, 2), dtype=np.float32)
    active_heat = np.asarray(heat_powers, dtype=np.float32).reshape(-1)[active_idx] if heat_powers is not None and active_idx.size else np.zeros((active_centers.shape[0],), dtype=np.float32)
    if sort_centers and active_centers.shape[0] > 1:
        order = np.lexsort((active_centers[:, 1], active_centers[:, 0]))
        active_centers = active_centers[order]
        active_heat = active_heat[order] if active_heat.size else active_heat
    n = min(int(active_centers.shape[0]), int(max_num_modules))
    centers_norm = np.zeros((max_num_modules, 2), dtype=np.float32)
    out_mask = np.zeros((max_num_modules,), dtype=np.float32)
    if n > 0:
        centers_norm[:n, 0] = np.clip(active_centers[:n, 0] / max(float(domain_length_x), 1.0e-8), 0.0, 1.0)
        centers_norm[:n, 1] = np.clip(active_centers[:n, 1] / max(float(domain_length_y), 1.0e-8), 0.0, 1.0)
        out_mask[:n] = 1.0
    parts = [centers_norm.reshape(-1), out_mask]
    if generate_heat_power:
        heat_norm = np.zeros((max_num_modules,), dtype=np.float32)
        if n > 0 and active_heat.size:
            heat_norm[:n] = active_heat[:n] / max(float(heat_power_scale), 1.0e-8)
        parts.append(heat_norm)
    return np.concatenate(parts, axis=0).astype(np.float32), out_mask


def channel_clearance_diagnostics(
    centers: np.ndarray,
    *,
    domain_length_x: float,
    domain_length_y: float,
    module_radius: float,
) -> Dict[str, float]:
    arr = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if arr.size == 0:
        return {
            "min_center_distance": float("inf"),
            "wall_clearance": float("inf"),
            "inlet_clearance": float("inf"),
            "outlet_clearance": float("inf"),
        }
    return {
        "min_center_distance": active_min_distance(arr),
        "wall_clearance": float(np.min(np.minimum(arr[:, 1], float(domain_length_y) - arr[:, 1]) - float(module_radius))),
        "inlet_clearance": float(np.min(arr[:, 0] - float(module_radius))),
        "outlet_clearance": float(np.min(float(domain_length_x) - arr[:, 0] - float(module_radius))),
    }


def repair_channel_design(
    centers: np.ndarray,
    *,
    count: int,
    domain_length_x: float = 12.0,
    domain_length_y: float = 4.0,
    module_radius: float = 0.45,
    min_center_distance: float = 1.1,
    max_num_modules: int = 12,
    min_count: int = 1,
    wall_clearance: float = 0.05,
    inlet_clearance: float = 0.25,
    outlet_clearance: float = 0.25,
    x_bounds: Optional[Tuple[float, float]] = None,
    y_bounds: Optional[Tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
    iterations: int = 48,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip and lightly repair a nonperiodic channel module layout."""

    rng = rng or np.random.default_rng()
    lx = float(domain_length_x)
    ly = float(domain_length_y)
    radius = max(float(module_radius), 0.0)
    x_min = radius + max(float(inlet_clearance), 0.0)
    x_max = lx - radius - max(float(outlet_clearance), 0.0)
    y_min = radius + max(float(wall_clearance), 0.0)
    y_max = ly - radius - max(float(wall_clearance), 0.0)
    if x_bounds is not None:
        bx0, bx1 = sorted((float(x_bounds[0]), float(x_bounds[1])))
        x_min = max(x_min, bx0 + radius)
        x_max = min(x_max, bx1 - radius)
    if y_bounds is not None:
        by0, by1 = sorted((float(y_bounds[0]), float(y_bounds[1])))
        y_min = max(y_min, by0 + radius)
        y_max = min(y_max, by1 - radius)
    if x_max < x_min:
        x_min, x_max = radius, max(radius, lx - radius)
    if y_max < y_min:
        y_min, y_max = radius, max(radius, ly - radius)
    min_dist = max(float(min_center_distance), 0.0)
    requested = int(np.clip(count, 0, max_num_modules))
    arr = np.asarray(centers, dtype=np.float64).reshape(-1, 2)[:requested].copy()
    if arr.shape[0] > 0:
        arr[:, 0] = np.clip(arr[:, 0], x_min, x_max)
        arr[:, 1] = np.clip(arr[:, 1], y_min, y_max)

    diagnostics: Dict[str, Any] = {
        "requested_count": requested,
        "added_count": 0,
        "dropped_count": 0,
        "overlap_pairs_initial": 0,
        "overlap_pairs_final": 0,
        "repaired": False,
        "x_bounds": list(x_bounds) if x_bounds is not None else None,
        "y_bounds": list(y_bounds) if y_bounds is not None else None,
    }

    def overlap_pairs(points: np.ndarray) -> List[Tuple[int, int, float]]:
        pairs: List[Tuple[int, int, float]] = []
        for i in range(points.shape[0]):
            for j in range(i + 1, points.shape[0]):
                dist = float(np.linalg.norm(points[i] - points[j]))
                if dist < min_dist:
                    pairs.append((i, j, dist))
        return pairs

    diagnostics["overlap_pairs_initial"] = len(overlap_pairs(arr))
    diagnostics["repaired"] = diagnostics["overlap_pairs_initial"] > 0
    for _ in range(max(iterations, 0)):
        pairs = overlap_pairs(arr)
        if not pairs:
            break
        for i, j, dist in pairs:
            if arr.shape[0] <= j:
                continue
            delta = arr[j] - arr[i]
            norm = float(np.linalg.norm(delta))
            if norm < 1.0e-8:
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
            else:
                direction = delta / norm
            step = 0.55 * (min_dist - norm + 1.0e-3)
            arr[i] -= 0.5 * step * direction
            arr[j] += 0.5 * step * direction
            arr[:, 0] = np.clip(arr[:, 0], x_min, x_max)
            arr[:, 1] = np.clip(arr[:, 1], y_min, y_max)

    kept: List[np.ndarray] = []
    dropped = 0
    for point in arr:
        if all(float(np.linalg.norm(point - other)) >= min_dist for other in kept):
            kept.append(point.copy())
        else:
            dropped += 1
    diagnostics["dropped_count"] = int(dropped)
    diagnostics["repaired"] = bool(diagnostics["repaired"] or dropped > 0)
    arr = np.asarray(kept, dtype=np.float64).reshape(-1, 2) if kept else np.zeros((0, 2), dtype=np.float64)

    target_min = int(np.clip(min_count, 0, max_num_modules))
    attempts = 0
    while arr.shape[0] < target_min and arr.shape[0] < max_num_modules and attempts < 5000:
        attempts += 1
        candidate = np.asarray([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float64)
        if arr.shape[0] == 0 or all(float(np.linalg.norm(candidate - p)) >= min_dist for p in arr):
            arr = np.concatenate([arr, candidate[None, :]], axis=0)
            diagnostics["added_count"] += 1
            diagnostics["repaired"] = True

    arr = sort_centers_xy(arr[:max_num_modules].astype(np.float32)).astype(np.float64)
    final_pairs = overlap_pairs(arr)
    diagnostics["overlap_pairs_final"] = len(final_pairs)
    diagnostics.update(channel_clearance_diagnostics(arr, domain_length_x=lx, domain_length_y=ly, module_radius=radius))
    diagnostics["valid"] = bool(
        len(final_pairs) == 0
        and arr.shape[0] <= max_num_modules
        and arr.shape[0] >= target_min
        and diagnostics["wall_clearance"] >= max(float(wall_clearance), 0.0) - 1.0e-5
        and diagnostics["inlet_clearance"] >= max(float(inlet_clearance), 0.0) - 1.0e-5
        and diagnostics["outlet_clearance"] >= max(float(outlet_clearance), 0.0) - 1.0e-5
    )
    diagnostics["count"] = int(arr.shape[0])
    return arr.astype(np.float32), diagnostics


class ThermalInverseDesignFlow(nn.Module):
    """Conditional rectified-flow generator over nonperiodic channel layouts."""

    def __init__(self, cfg: InverseModelConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, InverseModelConfig) else InverseModelConfig.from_dict(cfg)
        self.target_encoder = TargetEncoder(self.cfg.target_dim, self.cfg.hidden_dim, self.cfg.target_embed_dim, self.cfg.num_layers, self.cfg.dropout)
        self.behavior_org_head = BehaviorOrgHead(
            self.cfg.target_embed_dim,
            self.cfg.hidden_dim,
            self.cfg.behavior_latent_dim,
            self.cfg.organization_latent_dim,
            self.cfg.dropout,
        )
        self.velocity_net = DesignVelocityNet(
            self.cfg.design_dim,
            self.cfg.target_embed_dim,
            self.cfg.behavior_latent_dim,
            self.cfg.organization_latent_dim,
            self.cfg.hidden_dim,
            self.cfg.num_layers,
            self.cfg.dropout,
        )
        self.count_head = (
            MLP(self.cfg.target_embed_dim, self.cfg.hidden_dim, self.cfg.max_num_modules + 1, num_layers=2, dropout=self.cfg.dropout)
            if self.cfg.use_count_head
            else None
        )

    @property
    def max_num_modules(self) -> int:
        return int(self.cfg.max_num_modules)

    @property
    def design_dim(self) -> int:
        return int(self.cfg.design_dim or self.cfg.max_num_modules * (4 if self.cfg.generate_heat_power else 3))

    @property
    def heat_offset(self) -> int:
        return self.max_num_modules * 3

    def encode_condition(self, target_spec_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.target_encoder(target_spec_vector)
        behavior_hat, organization_hat = self.behavior_org_head(embedding)
        count_logits = self.count_head(embedding) if self.count_head is not None else torch.empty(
            embedding.shape[0], self.max_num_modules + 1, device=embedding.device, dtype=embedding.dtype
        )
        return {
            "target_embedding": embedding,
            "behavior_latent_hat": behavior_hat,
            "organization_latent_hat": organization_hat,
            "count_logits": count_logits,
        }

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.velocity_net(x_t, t, condition["target_embedding"], condition["behavior_latent_hat"], condition["organization_latent_hat"])

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, target_spec_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        condition = self.encode_condition(target_spec_vector)
        condition["velocity"] = self.velocity(x_t, t, condition)
        return condition

    def decode_centers_norm_from_design_vec(self, x: torch.Tensor) -> torch.Tensor:
        raw = x[:, : self.max_num_modules * 2].reshape(x.shape[0], self.max_num_modules, 2)
        if self.cfg.center_decode_mode == "sigmoid":
            return torch.sigmoid(raw)
        return raw.clamp(0.0, 1.0)

    def decode_heat_from_design_vec(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.cfg.generate_heat_power:
            return None
        raw = x[:, self.heat_offset : self.heat_offset + self.max_num_modules]
        return F.softplus(raw) * float(self.cfg.heat_power_scale)

    def _constraint_values(self, target_spec_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        constraints = target_spec_vector[:, -len(CONSTRAINT_NAMES) :]
        scale = float(max(self.cfg.domain_length_x, self.cfg.domain_length_y))
        n_min = torch.round(constraints[:, 0] * float(self.max_num_modules)).long().clamp(0, self.max_num_modules)
        n_max = torch.round(constraints[:, 1] * float(self.max_num_modules)).long().clamp(0, self.max_num_modules)
        n_max = torch.maximum(n_max, n_min)
        return {
            "n_min": n_min,
            "n_max": n_max,
            "min_center_distance": constraints[:, 2].clamp_min(0.0) * scale,
            "wall_clearance": constraints[:, 3].clamp_min(0.0) * scale,
            "inlet_clearance": constraints[:, 4].clamp_min(0.0) * scale,
            "outlet_clearance": constraints[:, 5].clamp_min(0.0) * scale,
            "heat_power_total": constraints[:, 6].clamp_min(0.0) * float(self.cfg.heat_power_scale),
            "heat_power_total_mask": constraints[:, 7].clamp(0.0, 1.0),
        }

    def _validity_prior(self, x: torch.Tensor) -> torch.Tensor:
        centers_norm = self.decode_centers_norm_from_design_vec(x)
        centers = centers_norm.clone()
        centers[..., 0] = centers[..., 0] * float(self.cfg.domain_length_x)
        centers[..., 1] = centers[..., 1] * float(self.cfg.domain_length_y)
        mask_prob = torch.sigmoid(x[:, self.max_num_modules * 2 : self.max_num_modules * 3])
        active_weight = mask_prob[:, :, None] * mask_prob[:, None, :]
        losses = []
        if self.max_num_modules >= 2:
            delta = centers[:, :, None, :] - centers[:, None, :, :]
            dist = torch.sqrt(delta.square().sum(dim=-1) + 1.0e-8)
            eye = torch.eye(self.max_num_modules, device=x.device, dtype=torch.bool)[None, :, :]
            overlap = F.relu(float(self.cfg.min_center_distance) - dist).square() * active_weight
            losses.append(overlap[~eye.expand_as(overlap)].mean())
        radius = float(self.cfg.module_radius)
        wall_min = radius + float(self.cfg.wall_clearance)
        wall_max = float(self.cfg.domain_length_y) - radius - float(self.cfg.wall_clearance)
        x_min = radius + float(self.cfg.inlet_clearance)
        x_max = float(self.cfg.domain_length_x) - radius - float(self.cfg.outlet_clearance)
        boundary = (
            F.relu(x_min - centers[..., 0]).square()
            + F.relu(centers[..., 0] - x_max).square()
            + F.relu(wall_min - centers[..., 1]).square()
            + F.relu(centers[..., 1] - wall_max).square()
        ) * mask_prob
        losses.append(boundary.mean())
        if self.cfg.generate_heat_power:
            heat = self.decode_heat_from_design_vec(x)
            assert heat is not None
            losses.append((heat * (1.0 - mask_prob)).square().mean() / max(float(self.cfg.heat_power_scale) ** 2, 1.0))
        return sum(losses) if losses else x.new_tensor(0.0)

    def training_loss(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        loss_weights: Optional[Mapping[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        target_spec = batch["target_spec_vector"]
        x1 = batch["design_vec"]
        bsz = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(bsz, 1, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t) * x0 + t * x1
        v_target = x1 - x0
        condition = self.encode_condition(target_spec)
        v_pred = self.velocity(x_t, t, condition)
        loss_flow = F.mse_loss(v_pred, v_target)

        true_count = batch["true_count"].long().reshape(-1).clamp(0, self.max_num_modules)
        if self.count_head is not None:
            loss_count = F.cross_entropy(condition["count_logits"], true_count)
            count_accuracy = (condition["count_logits"].argmax(dim=-1) == true_count).float().mean()
        else:
            loss_count = x1.new_tensor(0.0)
            count_accuracy = x1.new_tensor(float("nan"))

        target_active_fraction = batch.get("target_active_fraction")
        if target_active_fraction is None:
            loss_behavior = F.mse_loss(condition["behavior_latent_hat"], batch["behavior_target"])
            loss_organization = F.mse_loss(condition["organization_latent_hat"], batch["organization_target"])
            align_weight_mean = x1.new_tensor(1.0)
        else:
            align_weight = torch.as_tensor(target_active_fraction, device=x1.device, dtype=x1.dtype).reshape(-1).clamp(0.0, 1.0)
            align_weight = align_weight.pow(max(float(self.cfg.latent_align_completeness_power), 0.0))
            loss_behavior = torch.mean(align_weight * torch.mean((condition["behavior_latent_hat"] - batch["behavior_target"]).square(), dim=-1))
            loss_organization = torch.mean(align_weight * torch.mean((condition["organization_latent_hat"] - batch["organization_target"]).square(), dim=-1))
            align_weight_mean = align_weight.mean()
        x1_hat = x_t + (1.0 - t) * v_pred
        loss_validity = self._validity_prior(x1_hat)
        weights = {
            "flow_weight": 1.0,
            "count_weight": 0.1,
            "behavior_align_weight": 0.1,
            "organization_align_weight": 0.1,
            "validity_prior_weight": 0.0,
        }
        if loss_weights:
            weights.update({str(k): float(v) for k, v in loss_weights.items()})
        total = (
            weights["flow_weight"] * loss_flow
            + weights["count_weight"] * loss_count
            + weights["behavior_align_weight"] * loss_behavior
            + weights["organization_align_weight"] * loss_organization
            + weights["validity_prior_weight"] * loss_validity
        )
        metrics = {
            "loss_total": total.detach(),
            "loss_flow": loss_flow.detach(),
            "loss_count": loss_count.detach(),
            "loss_behavior": loss_behavior.detach(),
            "loss_organization": loss_organization.detach(),
            "loss_validity_prior": loss_validity.detach(),
            "count_accuracy": count_accuracy.detach(),
            "latent_align_weight_mean": align_weight_mean.detach(),
        }
        return total, metrics

    @torch.no_grad()
    def sample_designs(
        self,
        target_spec_vector: torch.Tensor | np.ndarray,
        n_samples: int,
        n_steps: int = 32,
        seed: Optional[int] = None,
        *,
        count_mode: str = "argmax",
        min_center_distance: Optional[float] = None,
        x_bounds: Optional[Tuple[float, float]] = None,
        y_bounds: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
    ) -> List[Dict[str, Any]]:
        was_training = self.training
        self.eval()
        param_device = next(self.parameters()).device
        device = device or param_device
        target = torch.as_tensor(target_spec_vector, dtype=torch.float32, device=device)
        if target.ndim == 1:
            target = target[None, :].expand(int(n_samples), -1).contiguous()
        elif target.shape[0] == 1 and int(n_samples) > 1:
            target = target.expand(int(n_samples), -1).contiguous()
        elif target.shape[0] != int(n_samples):
            raise ValueError(f"target_spec_vector batch {target.shape[0]} does not match n_samples={n_samples}.")

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(int(seed))
        x = torch.randn((int(n_samples), self.design_dim), generator=generator, device=device, dtype=torch.float32)
        condition = self.encode_condition(target)
        steps = max(int(n_steps), 1)
        dt = 1.0 / float(steps)
        solver = str(self.cfg.ode_solver).lower()
        for step in range(steps):
            t0 = torch.full((int(n_samples), 1), step / float(steps), device=device, dtype=x.dtype)
            v0 = self.velocity(x, t0, condition)
            if solver == "heun":
                x_euler = x + dt * v0
                t1 = torch.full((int(n_samples), 1), (step + 1) / float(steps), device=device, dtype=x.dtype)
                v1 = self.velocity(x_euler, t1, condition)
                x = x + 0.5 * dt * (v0 + v1)
            else:
                x = x + dt * v0

        count_probs = F.softmax(condition["count_logits"], dim=-1) if self.count_head is not None else torch.zeros(
            int(n_samples), self.max_num_modules + 1, device=device
        )
        if self.count_head is None:
            raw_counts = torch.full((int(n_samples),), self.max_num_modules, dtype=torch.long, device=device)
        elif count_mode == "sample":
            raw_counts = torch.multinomial(count_probs, num_samples=1, generator=generator).reshape(-1)
        else:
            raw_counts = torch.argmax(count_probs, dim=-1)
        constraints = self._constraint_values(target)
        raw_counts = raw_counts.clamp(constraints["n_min"], constraints["n_max"])

        centers_norm = self.decode_centers_norm_from_design_vec(x)
        mask_scores = torch.sigmoid(x[:, self.max_num_modules * 2 : self.max_num_modules * 3])
        heat = self.decode_heat_from_design_vec(x)
        outputs: List[Dict[str, Any]] = []
        np_rng = np.random.default_rng(seed)
        for i in range(int(n_samples)):
            count = int(raw_counts[i].item())
            if count <= 0 and int(constraints["n_min"][i].item()) > 0:
                count = int(constraints["n_min"][i].item())
            order = torch.argsort(mask_scores[i], descending=True)
            chosen = order[:count].detach().cpu().numpy()
            centers_i = centers_norm[i, chosen].detach().cpu().numpy()
            centers_phys = centers_i.copy()
            centers_phys[:, 0] *= float(self.cfg.domain_length_x)
            centers_phys[:, 1] *= float(self.cfg.domain_length_y)
            if x_bounds is not None and centers_phys.size:
                bx0, bx1 = sorted((float(x_bounds[0]), float(x_bounds[1])))
                centers_phys[:, 0] = np.clip(centers_phys[:, 0], bx0, bx1)
            if y_bounds is not None and centers_phys.size:
                by0, by1 = sorted((float(y_bounds[0]), float(y_bounds[1])))
                centers_phys[:, 1] = np.clip(centers_phys[:, 1], by0, by1)
            repair_min_dist = float(min_center_distance) if min_center_distance is not None else float(constraints["min_center_distance"][i].item())
            if repair_min_dist <= 0.0:
                repair_min_dist = float(self.cfg.min_center_distance)
            repaired, validity = repair_channel_design(
                centers_phys,
                count=count,
                domain_length_x=float(self.cfg.domain_length_x),
                domain_length_y=float(self.cfg.domain_length_y),
                module_radius=float(self.cfg.module_radius),
                min_center_distance=repair_min_dist,
                max_num_modules=self.max_num_modules,
                min_count=int(constraints["n_min"][i].item()),
                wall_clearance=max(float(constraints["wall_clearance"][i].item()), float(self.cfg.wall_clearance)),
                inlet_clearance=max(float(constraints["inlet_clearance"][i].item()), float(self.cfg.inlet_clearance)),
                outlet_clearance=max(float(constraints["outlet_clearance"][i].item()), float(self.cfg.outlet_clearance)),
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                rng=np_rng,
            )
            sorted_repaired = sort_centers_xy(repaired)
            mask = np.zeros((self.max_num_modules,), dtype=np.float32)
            mask[: sorted_repaired.shape[0]] = 1.0
            heat_out = None
            if heat is not None:
                heat_selected = heat[i, chosen].detach().cpu().numpy().astype(np.float32)
                if sorted_repaired.shape[0] != heat_selected.shape[0]:
                    heat_selected = np.resize(heat_selected, sorted_repaired.shape[0]).astype(np.float32)
                if float(constraints["heat_power_total_mask"][i].item()) > 0.5 and heat_selected.size:
                    total = float(constraints["heat_power_total"][i].item())
                    heat_selected = heat_selected * (total / max(float(np.sum(heat_selected)), 1.0e-8))
                heat_out = heat_selected
            outputs.append(
                {
                    "centers": sorted_repaired,
                    "centers_norm": np.stack(
                        [
                            sorted_repaired[:, 0] / max(float(self.cfg.domain_length_x), 1.0e-8),
                            sorted_repaired[:, 1] / max(float(self.cfg.domain_length_y), 1.0e-8),
                        ],
                        axis=-1,
                    ).astype(np.float32)
                    if sorted_repaired.size
                    else np.zeros((0, 2), dtype=np.float32),
                    "mask": mask,
                    "count": int(sorted_repaired.shape[0]),
                    "heat_powers": heat_out,
                    "raw_design_vec": x[i].detach().cpu().numpy().astype(np.float32),
                    "raw_count": int(raw_counts[i].item()),
                    "mask_scores": mask_scores[i].detach().cpu().numpy().astype(np.float32),
                    "count_probabilities": count_probs[i].detach().cpu().numpy().astype(np.float32),
                    "behavior_latent_hat": condition["behavior_latent_hat"][i].detach().cpu().numpy().astype(np.float32),
                    "organization_latent_hat": condition["organization_latent_hat"][i].detach().cpu().numpy().astype(np.float32),
                    "validity": validity,
                }
            )
        if was_training:
            self.train()
        return outputs
