from __future__ import annotations

"""Amortized conditional inverse generator for the inert multi-cylinder demo."""

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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
        freqs = torch.exp(
            torch.linspace(0.0, math.log(1000.0), half, device=t.device, dtype=t.dtype)
        )
        angles = t * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[:, : self.dim]


class TargetEncoder(nn.Module):
    def __init__(self, target_dim: int, hidden_dim: int, target_embed_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.net = MLP(
            target_dim,
            hidden_dim,
            target_embed_dim,
            num_layers=max(num_layers, 2),
            dropout=dropout,
            final_norm=True,
        )

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
    max_num_cylinders: int = 8
    design_dim: Optional[int] = None
    hidden_dim: int = 256
    target_embed_dim: int = 128
    behavior_latent_dim: int = 80
    organization_latent_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.05
    use_count_head: bool = True
    generate_re: bool = False
    ode_solver: str = "heun"
    domain_length_x: float = 24.0
    domain_length_y: float = 12.0
    min_num_cylinders: int = 1
    min_center_distance: float = 1.1
    re_scale: float = 200.0

    def __post_init__(self) -> None:
        self.max_num_cylinders = int(self.max_num_cylinders)
        if self.design_dim is None:
            self.design_dim = self.max_num_cylinders * 3
        self.design_dim = int(self.design_dim)
        expected = self.max_num_cylinders * 3
        if self.design_dim != expected:
            raise ValueError(f"design_dim must be max_num_cylinders*3={expected}, got {self.design_dim}.")
        if self.generate_re:
            raise NotImplementedError("generate_re=false is the supported first inverse-design mode.")

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


def encode_design_vector(
    centers: np.ndarray,
    *,
    max_num_cylinders: int,
    domain_length_x: float,
    domain_length_y: float,
    sort_centers: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    if sort_centers:
        centers_arr = sort_centers_xy(centers_arr)
    n = min(int(centers_arr.shape[0]), int(max_num_cylinders))
    centers_norm = np.zeros((max_num_cylinders, 2), dtype=np.float32)
    mask = np.zeros((max_num_cylinders,), dtype=np.float32)
    if n > 0:
        centers_norm[:n, 0] = np.mod(centers_arr[:n, 0], float(domain_length_x)) / max(float(domain_length_x), 1.0e-8)
        centers_norm[:n, 1] = np.mod(centers_arr[:n, 1], float(domain_length_y)) / max(float(domain_length_y), 1.0e-8)
        mask[:n] = 1.0
    return np.concatenate([centers_norm.reshape(-1), mask], axis=0).astype(np.float32), mask


def _periodic_delta(a: np.ndarray, b: np.ndarray, lx: float, ly: float) -> np.ndarray:
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    delta[..., 0] = (delta[..., 0] + 0.5 * lx) % lx - 0.5 * lx
    delta[..., 1] = (delta[..., 1] + 0.5 * ly) % ly - 0.5 * ly
    return delta


def periodic_min_distance(centers: np.ndarray, lx: float, ly: float) -> float:
    arr = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 2:
        return float("inf")
    best = float("inf")
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            delta = _periodic_delta(arr[i], arr[j], lx, ly)
            best = min(best, float(np.linalg.norm(delta)))
    return best


def repair_periodic_design(
    centers: np.ndarray,
    *,
    count: int,
    domain_length_x: float = 24.0,
    domain_length_y: float = 12.0,
    min_center_distance: float = 1.1,
    max_num_cylinders: int = 8,
    min_count: int = 1,
    rng: Optional[np.random.Generator] = None,
    iterations: int = 32,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Wrap and lightly repair a periodic multi-cylinder design."""

    rng = rng or np.random.default_rng()
    lx = float(domain_length_x)
    ly = float(domain_length_y)
    min_dist = max(float(min_center_distance), 0.0)
    requested = int(np.clip(count, 0, max_num_cylinders))
    arr = np.asarray(centers, dtype=np.float64).reshape(-1, 2)[:requested].copy()
    arr[:, 0] = np.mod(arr[:, 0], lx)
    arr[:, 1] = np.mod(arr[:, 1], ly)

    diagnostics: Dict[str, Any] = {
        "requested_count": requested,
        "added_count": 0,
        "dropped_count": 0,
        "overlap_pairs_initial": 0,
        "overlap_pairs_final": 0,
        "repaired": False,
    }

    def overlap_pairs(points: np.ndarray) -> List[Tuple[int, int, float]]:
        pairs: List[Tuple[int, int, float]] = []
        for i in range(points.shape[0]):
            for j in range(i + 1, points.shape[0]):
                dist = float(np.linalg.norm(_periodic_delta(points[i], points[j], lx, ly)))
                if dist < min_dist:
                    pairs.append((i, j, dist))
        return pairs

    initial_pairs = overlap_pairs(arr)
    diagnostics["overlap_pairs_initial"] = len(initial_pairs)
    if initial_pairs:
        diagnostics["repaired"] = True

    for _ in range(max(iterations, 0)):
        pairs = overlap_pairs(arr)
        if not pairs:
            break
        for i, j, dist in pairs:
            if arr.shape[0] <= j:
                continue
            delta = _periodic_delta(arr[i], arr[j], lx, ly)
            norm = float(np.linalg.norm(delta))
            if norm < 1.0e-8:
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
            else:
                direction = delta / norm
            step = 0.55 * (min_dist - norm + 1.0e-3)
            arr[i] -= 0.5 * step * direction
            arr[j] += 0.5 * step * direction
            arr[:, 0] = np.mod(arr[:, 0], lx)
            arr[:, 1] = np.mod(arr[:, 1], ly)

    kept: List[np.ndarray] = []
    dropped = 0
    for point in arr:
        if all(float(np.linalg.norm(_periodic_delta(other, point, lx, ly))) >= min_dist for other in kept):
            kept.append(point.copy())
        else:
            dropped += 1
    diagnostics["dropped_count"] = int(dropped)
    if dropped:
        diagnostics["repaired"] = True
    arr = np.asarray(kept, dtype=np.float64).reshape(-1, 2) if kept else np.zeros((0, 2), dtype=np.float64)

    target_min = int(np.clip(min_count, 0, max_num_cylinders))
    attempts = 0
    while arr.shape[0] < target_min and arr.shape[0] < max_num_cylinders and attempts < 5000:
        attempts += 1
        candidate = np.asarray([rng.uniform(0.0, lx), rng.uniform(0.0, ly)], dtype=np.float64)
        if arr.shape[0] == 0 or all(float(np.linalg.norm(_periodic_delta(p, candidate, lx, ly))) >= min_dist for p in arr):
            arr = np.concatenate([arr, candidate[None, :]], axis=0)
            diagnostics["added_count"] += 1
            diagnostics["repaired"] = True

    arr = arr[:max_num_cylinders]
    final_pairs = overlap_pairs(arr)
    diagnostics["overlap_pairs_final"] = len(final_pairs)
    diagnostics["min_pair_distance"] = periodic_min_distance(arr, lx, ly)
    diagnostics["valid"] = bool(len(final_pairs) == 0 and arr.shape[0] <= max_num_cylinders and arr.shape[0] >= target_min)
    diagnostics["count"] = int(arr.shape[0])
    return arr.astype(np.float32), diagnostics


class HypergraphInverseDesignFlow(nn.Module):
    """One amortized conditional rectified-flow inverse generator."""

    def __init__(self, cfg: InverseModelConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, InverseModelConfig) else InverseModelConfig.from_dict(cfg)
        self.target_encoder = TargetEncoder(
            self.cfg.target_dim,
            self.cfg.hidden_dim,
            self.cfg.target_embed_dim,
            self.cfg.num_layers,
            self.cfg.dropout,
        )
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
        count_classes = self.cfg.max_num_cylinders + 1
        self.count_head = MLP(
            self.cfg.target_embed_dim,
            self.cfg.hidden_dim,
            count_classes,
            num_layers=2,
            dropout=self.cfg.dropout,
        ) if self.cfg.use_count_head else None

    @property
    def max_num_cylinders(self) -> int:
        return int(self.cfg.max_num_cylinders)

    @property
    def design_dim(self) -> int:
        return int(self.cfg.design_dim or self.cfg.max_num_cylinders * 3)

    def encode_condition(self, target_spec_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.target_encoder(target_spec_vector)
        behavior_hat, organization_hat = self.behavior_org_head(embedding)
        count_logits = self.count_head(embedding) if self.count_head is not None else torch.empty(
            embedding.shape[0], self.max_num_cylinders + 1, device=embedding.device, dtype=embedding.dtype
        )
        return {
            "target_embedding": embedding,
            "behavior_latent_hat": behavior_hat,
            "organization_latent_hat": organization_hat,
            "count_logits": count_logits,
        }

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.velocity_net(
            x_t,
            t,
            condition["target_embedding"],
            condition["behavior_latent_hat"],
            condition["organization_latent_hat"],
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, target_spec_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        condition = self.encode_condition(target_spec_vector)
        condition["velocity"] = self.velocity(x_t, t, condition)
        return condition

    def _overlap_prior(self, x: torch.Tensor) -> torch.Tensor:
        centers = torch.sigmoid(x[:, : self.max_num_cylinders * 2]).reshape(x.shape[0], self.max_num_cylinders, 2)
        mask = torch.sigmoid(x[:, self.max_num_cylinders * 2 :]) > 0.5
        if self.max_num_cylinders < 2:
            return x.new_tensor(0.0)
        delta = centers[:, :, None, :] - centers[:, None, :, :]
        delta = torch.remainder(delta + 0.5, 1.0) - 0.5
        dist = torch.sqrt(delta.square().sum(dim=-1) + 1.0e-8)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(self.max_num_cylinders, device=x.device, dtype=torch.bool)[None, :, :]
        pair_mask = pair_mask & (~eye)
        threshold = 1.1 / max(float(max(self.cfg.domain_length_x, self.cfg.domain_length_y)), 1.0e-8)
        penalty = F.relu(threshold - dist).square()
        return penalty[pair_mask].mean() if torch.any(pair_mask) else x.new_tensor(0.0)

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

        true_count = batch["true_count"].long().reshape(-1).clamp(0, self.max_num_cylinders)
        if self.count_head is not None:
            loss_count = F.cross_entropy(condition["count_logits"], true_count)
            count_pred = condition["count_logits"].argmax(dim=-1)
            count_accuracy = (count_pred == true_count).float().mean()
        else:
            loss_count = x1.new_tensor(0.0)
            count_accuracy = x1.new_tensor(float("nan"))

        loss_behavior = F.mse_loss(condition["behavior_latent_hat"], batch["behavior_target"])
        loss_organization = F.mse_loss(condition["organization_latent_hat"], batch["organization_target"])
        loss_validity = self._overlap_prior(x1)

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
        }
        return total, metrics

    def _constraint_counts(self, target_spec_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        constraints = target_spec_vector[:, -5:]
        n_min = torch.round(constraints[:, 2] * float(self.max_num_cylinders)).long()
        n_max = torch.round(constraints[:, 3] * float(self.max_num_cylinders)).long()
        min_dist = constraints[:, 4] * float(max(self.cfg.domain_length_x, self.cfg.domain_length_y))
        n_min = n_min.clamp(0, self.max_num_cylinders)
        n_max = n_max.clamp(0, self.max_num_cylinders)
        n_max = torch.maximum(n_max, n_min)
        return n_min, n_max, min_dist

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
            int(n_samples), self.max_num_cylinders + 1, device=device
        )
        if self.count_head is None:
            raw_counts = torch.full((int(n_samples),), self.max_num_cylinders, dtype=torch.long, device=device)
        elif count_mode == "sample":
            raw_counts = torch.multinomial(count_probs, num_samples=1, generator=generator).reshape(-1)
        else:
            raw_counts = torch.argmax(count_probs, dim=-1)
        n_min, n_max, min_dist_from_target = self._constraint_counts(target)
        raw_counts = raw_counts.clamp(n_min, n_max)

        centers_norm = torch.sigmoid(x[:, : self.max_num_cylinders * 2]).reshape(int(n_samples), self.max_num_cylinders, 2)
        mask_scores = torch.sigmoid(x[:, self.max_num_cylinders * 2 :])
        outputs: List[Dict[str, Any]] = []
        np_rng = np.random.default_rng(seed)
        for i in range(int(n_samples)):
            count = int(raw_counts[i].item())
            if count <= 0 and int(n_min[i].item()) > 0:
                count = int(n_min[i].item())
            order = torch.argsort(mask_scores[i], descending=True)
            chosen = order[:count].detach().cpu().numpy()
            centers_i = centers_norm[i, chosen].detach().cpu().numpy()
            centers_phys = centers_i.copy()
            centers_phys[:, 0] *= float(self.cfg.domain_length_x)
            centers_phys[:, 1] *= float(self.cfg.domain_length_y)
            repair_min_dist = float(min_center_distance) if min_center_distance is not None else float(min_dist_from_target[i].item())
            if repair_min_dist <= 0.0:
                repair_min_dist = 1.1
            repaired, validity = repair_periodic_design(
                centers_phys,
                count=count,
                domain_length_x=float(self.cfg.domain_length_x),
                domain_length_y=float(self.cfg.domain_length_y),
                min_center_distance=repair_min_dist,
                max_num_cylinders=self.max_num_cylinders,
                min_count=int(n_min[i].item()),
                rng=np_rng,
            )
            mask = np.zeros((self.max_num_cylinders,), dtype=np.float32)
            mask[: repaired.shape[0]] = 1.0
            sorted_repaired = sort_centers_xy(repaired)
            outputs.append(
                {
                    "centers": sorted_repaired,
                    "centers_norm": np.stack(
                        [
                            np.mod(sorted_repaired[:, 0], float(self.cfg.domain_length_x)) / max(float(self.cfg.domain_length_x), 1.0e-8),
                            np.mod(sorted_repaired[:, 1], float(self.cfg.domain_length_y)) / max(float(self.cfg.domain_length_y), 1.0e-8),
                        ],
                        axis=-1,
                    ) if sorted_repaired.size else np.zeros((0, 2), dtype=np.float32),
                    "mask": mask,
                    "count": int(sorted_repaired.shape[0]),
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
