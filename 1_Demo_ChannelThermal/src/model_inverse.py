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
from thermal_design_intent import DESIGN_INTENT_DIM, FIELD_INTENT_CHANNELS, OBJECTIVE_DIM, STRUCTURE_INTENT_DIM


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


class HypergraphPlanner(nn.Module):
    """Predict the inverse model's structured hypergraph design intention."""

    def __init__(
        self,
        target_embed_dim: int,
        behavior_latent_dim: int,
        organization_latent_dim: int,
        hidden_dim: int,
        hypergraph_plan_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.behavior_latent_dim = int(behavior_latent_dim)
        self.organization_latent_dim = int(organization_latent_dim)
        in_dim = int(target_embed_dim) + self.behavior_latent_dim + self.organization_latent_dim
        self.net = MLP(in_dim, hidden_dim, hypergraph_plan_dim, num_layers=3, dropout=dropout)

    def forward(
        self,
        target_embedding: torch.Tensor,
        behavior_latent_hat: Optional[torch.Tensor] = None,
        organization_latent_hat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = target_embedding.shape[0]
        device = target_embedding.device
        dtype = target_embedding.dtype
        if behavior_latent_hat is None:
            behavior_latent_hat = torch.zeros(bsz, self.behavior_latent_dim, device=device, dtype=dtype)
        if organization_latent_hat is None:
            organization_latent_hat = torch.zeros(bsz, self.organization_latent_dim, device=device, dtype=dtype)
        return self.net(torch.cat([target_embedding, behavior_latent_hat, organization_latent_hat], dim=-1))


class HypergraphPlanEncoder(nn.Module):
    """Embed a structured hypergraph plan for layout-flow conditioning."""

    def __init__(self, hypergraph_plan_dim: int, hidden_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            MLP(hypergraph_plan_dim, hidden_dim, embed_dim, num_layers=3, dropout=dropout),
            nn.LayerNorm(int(embed_dim)),
        )

    def forward(self, hypergraph_plan_hat: torch.Tensor) -> torch.Tensor:
        return self.net(hypergraph_plan_hat)


class FieldIntentEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        width = max(min(int(hidden_dim) // 2, 128), 32)
        self.in_channels = int(in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout2d(float(dropout)) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, int(out_dim)),
            nn.LayerNorm(int(out_dim)),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.net(maps)


class DesignIntentEncoder(nn.Module):
    def __init__(
        self,
        *,
        legacy_dim: int,
        intent_dim: int,
        objective_dim: int,
        field_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        use_legacy_kpi_vector: bool,
        structure_dim: int = 0,
        structure_map_channels: int = 0,
        heat_condition_dim: int = 0,
    ) -> None:
        super().__init__()
        part_dim = max(int(out_dim) // 2, 32)
        self.intent_dim = int(intent_dim)
        self.objective_dim = int(objective_dim)
        self.field_channels = int(field_channels)
        self.structure_dim = int(structure_dim)
        self.structure_map_channels = int(structure_map_channels)
        self.heat_condition_dim = int(heat_condition_dim)
        self.use_legacy_kpi_vector = bool(use_legacy_kpi_vector)
        self.intent_encoder = MLP(self.intent_dim, hidden_dim, part_dim, num_layers=max(num_layers, 2), dropout=dropout, final_norm=True)
        self.objective_encoder = MLP(self.objective_dim, hidden_dim, part_dim, num_layers=2, dropout=dropout, final_norm=True)
        self.field_encoder = FieldIntentEncoder(self.field_channels, hidden_dim, part_dim, dropout)
        self.structure_encoder = (
            MLP(self.structure_dim, hidden_dim, part_dim, num_layers=2, dropout=dropout, final_norm=True)
            if self.structure_dim > 0
            else None
        )
        self.structure_map_encoder = (
            FieldIntentEncoder(self.structure_map_channels, hidden_dim, part_dim, dropout)
            if self.structure_map_channels > 0
            else None
        )
        self.heat_encoder = (
            MLP(self.heat_condition_dim, hidden_dim, part_dim, num_layers=2, dropout=dropout, final_norm=True)
            if self.heat_condition_dim > 0
            else None
        )
        self.legacy_encoder = MLP(legacy_dim, hidden_dim, part_dim, num_layers=max(num_layers, 2), dropout=dropout, final_norm=True) if self.use_legacy_kpi_vector else None
        num_parts = 3 + (1 if self.use_legacy_kpi_vector else 0)
        num_parts += 1 if self.structure_encoder is not None else 0
        num_parts += 1 if self.structure_map_encoder is not None else 0
        num_parts += 1 if self.heat_encoder is not None else 0
        self.fuse = MLP(part_dim * num_parts, hidden_dim, out_dim, num_layers=2, dropout=dropout, final_norm=True)

    def forward(
        self,
        target_spec_vector: torch.Tensor,
        design_intent_vector: Optional[torch.Tensor],
        objective_weight_vector: Optional[torch.Tensor],
        field_intent_maps: Optional[torch.Tensor],
        structure_intent_vector: Optional[torch.Tensor] = None,
        structure_intent_maps: Optional[torch.Tensor] = None,
        heat_condition_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = target_spec_vector.shape[0]
        device = target_spec_vector.device
        dtype = target_spec_vector.dtype
        if design_intent_vector is None:
            design_intent_vector = torch.zeros(bsz, self.intent_dim, device=device, dtype=dtype)
        if objective_weight_vector is None:
            objective_weight_vector = torch.zeros(bsz, self.objective_dim, device=device, dtype=dtype)
        if field_intent_maps is None:
            field_intent_maps = torch.zeros(bsz, self.field_channels, 12, 24, device=device, dtype=dtype)
        parts = [
            self.intent_encoder(design_intent_vector),
            self.objective_encoder(objective_weight_vector),
            self.field_encoder(field_intent_maps),
        ]
        if self.structure_encoder is not None:
            if structure_intent_vector is None:
                structure_intent_vector = torch.zeros(bsz, self.structure_dim, device=device, dtype=dtype)
            parts.append(self.structure_encoder(structure_intent_vector))
        if self.structure_map_encoder is not None:
            if structure_intent_maps is None:
                structure_intent_maps = torch.zeros(bsz, self.structure_map_channels, 12, 24, device=device, dtype=dtype)
            parts.append(self.structure_map_encoder(structure_intent_maps))
        if self.heat_encoder is not None:
            if heat_condition_vector is None:
                heat_condition_vector = torch.zeros(bsz, self.heat_condition_dim, device=device, dtype=dtype)
            parts.append(self.heat_encoder(heat_condition_vector))
        if self.legacy_encoder is not None:
            parts.append(self.legacy_encoder(target_spec_vector))
        return self.fuse(torch.cat(parts, dim=-1))


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
        hypergraph_plan_embed_dim: int = 0,
        use_hypergraph_plan: bool = False,
    ) -> None:
        super().__init__()
        self.use_hypergraph_plan = bool(use_hypergraph_plan)
        self.hypergraph_plan_embed_dim = int(hypergraph_plan_embed_dim) if self.use_hypergraph_plan else 0
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        in_dim = int(design_dim) + int(target_embed_dim) + int(behavior_latent_dim) + int(organization_latent_dim) + self.hypergraph_plan_embed_dim + int(time_embed_dim)
        self.net = MLP(in_dim, hidden_dim, design_dim, num_layers=max(num_layers, 2), dropout=dropout)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        target_embedding: torch.Tensor,
        behavior_latent_hat: torch.Tensor,
        organization_latent_hat: torch.Tensor,
        hypergraph_plan_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        parts = [target_embedding, behavior_latent_hat, organization_latent_hat]
        if self.use_hypergraph_plan:
            if hypergraph_plan_embedding is None:
                hypergraph_plan_embedding = torch.zeros(
                    x_t.shape[0],
                    self.hypergraph_plan_embed_dim,
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
            parts.append(hypergraph_plan_embedding)
        parts.append(t_emb)
        cond = torch.cat(parts, dim=-1)
        return self.net(torch.cat([x_t, cond], dim=-1))


@dataclass
class InverseModelConfig:
    target_dim: int
    inverse_mode: str = "layout_flow"
    # Deprecated config/checkpoint field. inverse_mode is the public switch;
    # this is retained so older payloads can still be read.
    use_hypergraph_plan: bool = False
    hypergraph_plan_dim: Optional[int] = None
    hypergraph_plan_embed_dim: int = 128
    hypergraph_plan_conditioning: str = "concat"
    hypergraph_plan_num_edges: Optional[int] = None
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
    center_decode_mode: str = "clamp"
    latent_align_completeness_power: float = 0.5
    conditioning_mode: str = "legacy_kpi"
    use_legacy_kpi_vector: bool = True
    design_intent_dim: int = DESIGN_INTENT_DIM
    objective_dim: int = OBJECTIVE_DIM
    field_intent_channels: int = len(FIELD_INTENT_CHANNELS)
    conditioning_dropout_enabled: bool = False
    conditioning_drop_probability: float = 0.0
    structure_conditioning_enabled: bool = False
    structure_intent_dim: int = STRUCTURE_INTENT_DIM
    structure_intent_map_channels: int = 0
    structure_drop_probability: float = 0.0
    heat_conditioning_enabled: bool = False
    heat_condition_dim: Optional[int] = None
    heat_drop_probability: float = 0.0
    slot_identity_mode: str = "anonymous"

    def __post_init__(self) -> None:
        self.max_num_modules = int(self.max_num_modules)
        inverse_mode = str(self.inverse_mode or "layout_flow").lower().strip()
        if inverse_mode not in {"layout_flow", "layout_flow_with_hypergraph_plan"}:
            raise ValueError("inverse_mode must be 'layout_flow' or 'layout_flow_with_hypergraph_plan'.")
        self.inverse_mode = inverse_mode
        self.use_hypergraph_plan = self.inverse_mode == "layout_flow_with_hypergraph_plan"
        conditioning = str(self.hypergraph_plan_conditioning or "concat").lower().strip()
        if conditioning != "concat":
            raise ValueError("hypergraph_plan_conditioning currently supports only 'concat'.")
        self.hypergraph_plan_conditioning = conditioning
        if self.hypergraph_plan_dim is not None:
            self.hypergraph_plan_dim = int(self.hypergraph_plan_dim)
        if self.hypergraph_plan_num_edges is not None:
            self.hypergraph_plan_num_edges = int(self.hypergraph_plan_num_edges)
        if self.use_hypergraph_plan and self.hypergraph_plan_dim is None:
            edges = int(self.hypergraph_plan_num_edges or 6)
            self.hypergraph_plan_num_edges = edges
            self.hypergraph_plan_dim = edges * (8 + self.max_num_modules)
        self.hypergraph_plan_embed_dim = int(self.hypergraph_plan_embed_dim)
        expected = self.max_num_modules * (4 if self.generate_heat_power else 3)
        if self.design_dim is None:
            self.design_dim = expected
        self.design_dim = int(self.design_dim)
        if self.design_dim != expected:
            raise ValueError(f"design_dim must be {expected} for max_num_modules={self.max_num_modules}, generate_heat_power={self.generate_heat_power}.")
        mode = str(self.center_decode_mode).lower().strip()
        # Coordinates are trained directly as normalized [0, 1] values.  Older
        # configs used "sigmoid", which double-compressed generated coordinates
        # toward the channel center during evaluation.
        if mode == "sigmoid":
            mode = "clamp"
        if mode not in {"clamp", "logit_sigmoid"}:
            raise ValueError("center_decode_mode must be 'clamp' for direct normalized coordinates or 'logit_sigmoid' for logit-space coordinates.")
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
        conditioning_mode = str(self.conditioning_mode).lower().strip()
        if conditioning_mode not in {"legacy_kpi", "design_intent"}:
            raise ValueError("conditioning_mode must be 'legacy_kpi' or 'design_intent'.")
        self.conditioning_mode = conditioning_mode
        self.design_intent_dim = int(self.design_intent_dim)
        self.objective_dim = int(self.objective_dim)
        self.field_intent_channels = int(self.field_intent_channels)
        self.conditioning_drop_probability = min(max(float(self.conditioning_drop_probability), 0.0), 1.0)
        self.structure_intent_dim = int(self.structure_intent_dim)
        self.structure_intent_map_channels = int(self.structure_intent_map_channels) if self.structure_conditioning_enabled else 0
        if self.heat_condition_dim is None:
            self.heat_condition_dim = int(self.max_num_modules) * 2 + 7
        self.heat_condition_dim = int(self.heat_condition_dim) if self.heat_conditioning_enabled else 0
        self.structure_drop_probability = min(max(float(self.structure_drop_probability), 0.0), 1.0)
        self.heat_drop_probability = min(max(float(self.heat_drop_probability), 0.0), 1.0)
        slot_mode = str(self.slot_identity_mode).lower().strip()
        if slot_mode not in {"anonymous", "heat_conditioned"}:
            raise ValueError("slot_identity_mode must be 'anonymous' or 'heat_conditioned'.")
        self.slot_identity_mode = slot_mode

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InverseModelConfig":
        data = dict(payload)
        if "inverse_mode" not in data and bool(data.get("use_hypergraph_plan", False)):
            data["inverse_mode"] = "layout_flow_with_hypergraph_plan"
        for key in ("target_dim", "design_dim", "behavior_latent_dim", "organization_latent_dim", "hypergraph_plan_dim", "hypergraph_plan_num_edges"):
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
    preserve_order: bool = False,
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

    arr = arr[:max_num_modules].astype(np.float32)
    if not bool(preserve_order):
        arr = sort_centers_xy(arr)
    arr = arr.astype(np.float64)
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
        self.design_intent_encoder = DesignIntentEncoder(
            legacy_dim=self.cfg.target_dim,
            intent_dim=self.cfg.design_intent_dim,
            objective_dim=self.cfg.objective_dim,
            field_channels=self.cfg.field_intent_channels,
            hidden_dim=self.cfg.hidden_dim,
            out_dim=self.cfg.target_embed_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            use_legacy_kpi_vector=self.cfg.use_legacy_kpi_vector,
            structure_dim=self.cfg.structure_intent_dim if self.cfg.structure_conditioning_enabled else 0,
            structure_map_channels=self.cfg.structure_intent_map_channels,
            heat_condition_dim=int(self.cfg.heat_condition_dim or 0),
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
            hypergraph_plan_embed_dim=self.cfg.hypergraph_plan_embed_dim,
            use_hypergraph_plan=self.cfg.use_hypergraph_plan,
        )
        if self.cfg.use_hypergraph_plan:
            if self.cfg.hypergraph_plan_dim is None:
                raise ValueError("hypergraph_plan_dim is required when use_hypergraph_plan=true.")
            self.hypergraph_planner = HypergraphPlanner(
                self.cfg.target_embed_dim,
                self.cfg.behavior_latent_dim,
                self.cfg.organization_latent_dim,
                self.cfg.hidden_dim,
                int(self.cfg.hypergraph_plan_dim),
                self.cfg.dropout,
            )
            self.hypergraph_plan_encoder = HypergraphPlanEncoder(
                int(self.cfg.hypergraph_plan_dim),
                self.cfg.hidden_dim,
                self.cfg.hypergraph_plan_embed_dim,
                self.cfg.dropout,
            )
        else:
            self.hypergraph_planner = None
            self.hypergraph_plan_encoder = None
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

    def _apply_condition_dropout(
        self,
        design_intent_vector: Optional[torch.Tensor],
        objective_weight_vector: Optional[torch.Tensor],
        field_intent_maps: Optional[torch.Tensor],
        structure_intent_vector: Optional[torch.Tensor] = None,
        structure_intent_maps: Optional[torch.Tensor] = None,
        heat_condition_vector: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.training or not self.cfg.conditioning_dropout_enabled or self.cfg.conditioning_drop_probability <= 0.0:
            return design_intent_vector, objective_weight_vector, field_intent_maps, structure_intent_vector, structure_intent_maps, heat_condition_vector
        drop = float(self.cfg.conditioning_drop_probability)
        if design_intent_vector is not None:
            keep = (torch.rand(design_intent_vector.shape[0], 1, device=design_intent_vector.device) >= drop).to(design_intent_vector.dtype)
            design_intent_vector = design_intent_vector * keep
        if objective_weight_vector is not None:
            keep = (torch.rand(objective_weight_vector.shape[0], 1, device=objective_weight_vector.device) >= drop).to(objective_weight_vector.dtype)
            objective_weight_vector = objective_weight_vector * keep
        if field_intent_maps is not None:
            keep = (torch.rand(field_intent_maps.shape[0], 1, 1, 1, device=field_intent_maps.device) >= drop).to(field_intent_maps.dtype)
            field_intent_maps = field_intent_maps * keep
        if structure_intent_vector is not None and self.cfg.structure_drop_probability > 0.0:
            keep = (torch.rand(structure_intent_vector.shape[0], 1, device=structure_intent_vector.device) >= self.cfg.structure_drop_probability).to(structure_intent_vector.dtype)
            structure_intent_vector = structure_intent_vector * keep
        if structure_intent_maps is not None and self.cfg.structure_drop_probability > 0.0:
            keep = (torch.rand(structure_intent_maps.shape[0], 1, 1, 1, device=structure_intent_maps.device) >= self.cfg.structure_drop_probability).to(structure_intent_maps.dtype)
            structure_intent_maps = structure_intent_maps * keep
        if heat_condition_vector is not None and self.cfg.heat_drop_probability > 0.0:
            keep = (torch.rand(heat_condition_vector.shape[0], 1, device=heat_condition_vector.device) >= self.cfg.heat_drop_probability).to(heat_condition_vector.dtype)
            heat_condition_vector = heat_condition_vector * keep
        return design_intent_vector, objective_weight_vector, field_intent_maps, structure_intent_vector, structure_intent_maps, heat_condition_vector

    def encode_condition(
        self,
        target_spec_vector: torch.Tensor,
        *,
        design_intent_vector: Optional[torch.Tensor] = None,
        objective_weight_vector: Optional[torch.Tensor] = None,
        field_intent_maps: Optional[torch.Tensor] = None,
        structure_intent_vector: Optional[torch.Tensor] = None,
        structure_intent_maps: Optional[torch.Tensor] = None,
        heat_condition_vector: Optional[torch.Tensor] = None,
        heat_condition_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.cfg.conditioning_mode == "design_intent":
            design_intent_vector, objective_weight_vector, field_intent_maps, structure_intent_vector, structure_intent_maps, heat_condition_vector = self._apply_condition_dropout(
                design_intent_vector,
                objective_weight_vector,
                field_intent_maps,
                structure_intent_vector,
                structure_intent_maps,
                heat_condition_vector,
            )
            if heat_condition_vector is not None and heat_condition_mask is not None and heat_condition_vector.shape[-1] >= self.max_num_modules * 2:
                heat_condition_vector = heat_condition_vector.clone()
                mask = heat_condition_mask.to(device=heat_condition_vector.device, dtype=heat_condition_vector.dtype)
                heat_condition_vector[:, : self.max_num_modules] *= mask
                heat_condition_vector[:, self.max_num_modules : self.max_num_modules * 2] *= mask
            embedding = self.design_intent_encoder(
                target_spec_vector,
                design_intent_vector,
                objective_weight_vector,
                field_intent_maps,
                structure_intent_vector=structure_intent_vector,
                structure_intent_maps=structure_intent_maps,
                heat_condition_vector=heat_condition_vector,
            )
        else:
            embedding = self.target_encoder(target_spec_vector)
        behavior_hat, organization_hat = self.behavior_org_head(embedding)
        if self.hypergraph_planner is not None and self.hypergraph_plan_encoder is not None:
            hypergraph_plan_hat = self.hypergraph_planner(embedding, behavior_hat, organization_hat)
            hypergraph_plan_embedding = self.hypergraph_plan_encoder(hypergraph_plan_hat)
        else:
            hypergraph_plan_hat = None
            hypergraph_plan_embedding = None
        count_logits = self.count_head(embedding) if self.count_head is not None else torch.empty(
            embedding.shape[0], self.max_num_modules + 1, device=embedding.device, dtype=embedding.dtype
        )
        condition = {
            "target_embedding": embedding,
            "behavior_latent_hat": behavior_hat,
            "organization_latent_hat": organization_hat,
            "count_logits": count_logits,
        }
        if hypergraph_plan_hat is not None and hypergraph_plan_embedding is not None:
            condition["hypergraph_plan_hat"] = hypergraph_plan_hat
            condition["hypergraph_plan_embedding"] = hypergraph_plan_embedding
        return condition

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.velocity_net(
            x_t,
            t,
            condition["target_embedding"],
            condition["behavior_latent_hat"],
            condition["organization_latent_hat"],
            condition.get("hypergraph_plan_embedding"),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, target_spec_vector: torch.Tensor, **condition_kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        condition = self.encode_condition(target_spec_vector, **condition_kwargs)
        condition["velocity"] = self.velocity(x_t, t, condition)
        return condition

    def decode_centers_norm_from_design_vec(self, x: torch.Tensor) -> torch.Tensor:
        raw = x[:, : self.max_num_modules * 2].reshape(x.shape[0], self.max_num_modules, 2)
        if self.cfg.center_decode_mode == "logit_sigmoid":
            return torch.sigmoid(raw)
        return raw.clamp(0.0, 1.0)

    def decode_heat_from_design_vec(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.cfg.generate_heat_power:
            return None
        raw = x[:, self.heat_offset : self.heat_offset + self.max_num_modules]
        return F.softplus(raw) * float(self.cfg.heat_power_scale)

    def _structure_features_from_design(self, x: torch.Tensor, true_count: torch.Tensor, heat_condition_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        centers = self.decode_centers_norm_from_design_vec(x)
        bsz = centers.shape[0]
        device = centers.device
        dtype = centers.dtype
        idx = torch.arange(self.max_num_modules, device=device)[None, :]
        mask = (idx < true_count.reshape(-1, 1).clamp(0, self.max_num_modules)).to(dtype)
        heat = mask
        if heat_condition_vector is not None and heat_condition_vector.shape[-1] >= self.max_num_modules:
            heat = heat_condition_vector[:, : self.max_num_modules].to(device=device, dtype=dtype).clamp_min(0.0) * mask
        count = mask.sum(dim=1).clamp_min(1.0)
        heat_sum = heat.sum(dim=1).clamp_min(1.0e-8)
        xnorm = centers[..., 0] * mask
        ynorm = centers[..., 1] * mask
        cx = xnorm.sum(dim=1) / count
        cy = ynorm.sum(dim=1) / count
        hcx = (centers[..., 0] * heat).sum(dim=1) / heat_sum
        hcy = (centers[..., 1] * heat).sum(dim=1) / heat_sum
        inactive_x = centers[..., 0].new_full(centers[..., 0].shape, 1.0e6)
        inactive_y = centers[..., 1].new_full(centers[..., 1].shape, 1.0e6)
        x_min = torch.where(mask > 0.5, centers[..., 0], inactive_x).min(dim=1).values
        y_min = torch.where(mask > 0.5, centers[..., 1], inactive_y).min(dim=1).values
        x_max = torch.where(mask > 0.5, centers[..., 0], -inactive_x).max(dim=1).values
        y_max = torch.where(mask > 0.5, centers[..., 1], -inactive_y).max(dim=1).values
        active_any = (mask.sum(dim=1) > 0.5).to(dtype)
        x_cov = (x_max - x_min).clamp_min(0.0) * active_any
        y_cov = (y_max - y_min).clamp_min(0.0) * active_any
        x_std = torch.sqrt((((centers[..., 0] - cx[:, None]) * mask).square().sum(dim=1) / count).clamp_min(0.0) + 1.0e-8)
        y_std = torch.sqrt((((centers[..., 1] - cy[:, None]) * mask).square().sum(dim=1) / count).clamp_min(0.0) + 1.0e-8)
        dx = centers[:, :, None, 0] - centers[:, None, :, 0]
        dy = centers[:, :, None, 1] - centers[:, None, :, 1]
        dist_scaled = torch.sqrt((dx * float(self.cfg.domain_length_x)).square() + (dy * float(self.cfg.domain_length_y)).square() + 1.0e-8) / float(max(self.cfg.domain_length_x, self.cfg.domain_length_y, 1.0e-8))
        pair_mask = (mask[:, :, None] * mask[:, None, :]) > 0.5
        eye = torch.eye(self.max_num_modules, device=device, dtype=torch.bool)[None]
        pair_mask = pair_mask & ~eye
        has_pair = pair_mask.any(dim=(1, 2))
        pair_count = pair_mask.sum(dim=(1, 2)).clamp_min(1)
        pair_vals = torch.where(pair_mask, dist_scaled, torch.zeros_like(dist_scaled))
        mean_pair = pair_vals.sum(dim=(1, 2)) / pair_count.to(dtype)
        min_pair = torch.where(pair_mask, dist_scaled, torch.full_like(dist_scaled, 1.0e6)).amin(dim=(1, 2))
        min_pair = torch.where(has_pair, min_pair, torch.zeros_like(min_pair))
        pair_std = torch.sqrt((torch.where(pair_mask, (dist_scaled - mean_pair[:, None, None]).square(), torch.zeros_like(dist_scaled)).sum(dim=(1, 2)) / pair_count.to(dtype)).clamp_min(0.0) + 1.0e-8)
        nn_valid = pair_mask.any(dim=2)
        nn = torch.where(pair_mask, dist_scaled, torch.full_like(dist_scaled, 1.0e6)).amin(dim=2)
        nn = torch.where(nn_valid, nn, torch.zeros_like(nn))
        nn_count = nn_valid.to(dtype).sum(dim=1).clamp_min(1.0)
        nn_mean = nn.sum(dim=1) / nn_count
        nn_std = torch.sqrt((torch.where(nn_valid, (nn - nn_mean[:, None]).square(), torch.zeros_like(nn)).sum(dim=1) / nn_count).clamp_min(0.0) + 1.0e-8)
        wall = torch.minimum(centers[..., 1], 1.0 - centers[..., 1]) * 2.0
        wall_mean = (wall * mask).sum(dim=1) / count
        wall_min = torch.where(mask > 0.5, wall, torch.ones_like(wall)).min(dim=1).values * active_any
        upstream = ((centers[..., 0] < (1.0 / 3.0)).to(dtype) * mask).sum(dim=1) / count
        midstream = (((centers[..., 0] >= (1.0 / 3.0)) & (centers[..., 0] < (2.0 / 3.0))).to(dtype) * mask).sum(dim=1) / count
        downstream = ((centers[..., 0] >= (2.0 / 3.0)).to(dtype) * mask).sum(dim=1) / count
        upstream_h = ((centers[..., 0] < (1.0 / 3.0)).to(dtype) * heat).sum(dim=1) / heat_sum
        midstream_h = (((centers[..., 0] >= (1.0 / 3.0)) & (centers[..., 0] < (2.0 / 3.0))).to(dtype) * heat).sum(dim=1) / heat_sum
        downstream_h = ((centers[..., 0] >= (2.0 / 3.0)).to(dtype) * heat).sum(dim=1) / heat_sum

        def hist(values: torch.Tensor, bins: int) -> torch.Tensor:
            centers_b = torch.linspace(0.5 / bins, 1.0 - 0.5 / bins, bins, device=device, dtype=dtype)
            weights = F.relu(1.0 - torch.abs(values[:, :, None] - centers_b[None, None, :]) * bins) * mask[:, :, None]
            return weights.sum(dim=1) / count[:, None]

        x_hist = hist(centers[..., 0], 6)
        y_hist = hist(centers[..., 1], 4)
        pair_bins = torch.linspace(0.5 / 6.0, 1.0 - 0.5 / 6.0, 6, device=device, dtype=dtype)
        pair_weights = F.relu(1.0 - torch.abs(dist_scaled[:, :, :, None].clamp(0.0, 1.0) - pair_bins[None, None, None, :]) * 6.0) * pair_mask[:, :, :, None].to(dtype)
        pair_hist = pair_weights.sum(dim=(1, 2)) / pair_count[:, None].to(dtype)
        occ_entropy = -((x_hist.clamp_min(1.0e-8) * x_hist.clamp_min(1.0e-8).log()).sum(dim=1) / math.log(6.0) + (y_hist.clamp_min(1.0e-8) * y_hist.clamp_min(1.0e-8).log()).sum(dim=1) / math.log(4.0)) * 0.5
        heat_entropy = occ_entropy
        anisotropy = (x_std - y_std).abs() / (x_std + y_std).clamp_min(1.0e-8)
        features = torch.cat(
            [
                (mask.sum(dim=1) / float(self.max_num_modules))[:, None],
                cx[:, None],
                cy[:, None],
                hcx[:, None],
                hcy[:, None],
                x_cov[:, None],
                y_cov[:, None],
                (x_cov * y_cov)[:, None],
                x_std[:, None],
                y_std[:, None],
                min_pair[:, None],
                mean_pair[:, None],
                pair_std[:, None],
                nn_mean[:, None],
                nn_std[:, None],
                wall_mean[:, None],
                wall_min[:, None],
                upstream[:, None],
                midstream[:, None],
                downstream[:, None],
                upstream_h[:, None],
                midstream_h[:, None],
                downstream_h[:, None],
                x_hist,
                y_hist,
                pair_hist,
                occ_entropy[:, None],
                heat_entropy[:, None],
                anisotropy[:, None],
            ],
            dim=1,
        )
        return torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

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
        condition = self.encode_condition(
            target_spec,
            design_intent_vector=batch.get("design_intent_vector"),
            objective_weight_vector=batch.get("objective_weight_vector"),
            field_intent_maps=batch.get("field_intent_maps"),
            structure_intent_vector=batch.get("structure_intent_vector"),
            structure_intent_maps=batch.get("structure_intent_maps"),
            heat_condition_vector=batch.get("heat_condition_vector"),
            heat_condition_mask=batch.get("heat_condition_mask"),
        )
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
        loss_set_center = self._set_center_loss(x1_hat, x1, true_count)
        if self.cfg.structure_conditioning_enabled and batch.get("structure_intent_vector") is not None:
            pred_structure = self._structure_features_from_design(x1_hat, true_count, batch.get("heat_condition_vector"))
            target_structure = batch["structure_intent_vector"].to(device=x1.device, dtype=x1.dtype)
            dim = min(pred_structure.shape[-1], target_structure.shape[-1])
            pred_structure = torch.nan_to_num(pred_structure[:, :dim], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            target_structure = torch.nan_to_num(target_structure[:, :dim], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            per_case = torch.mean((pred_structure - target_structure).square(), dim=-1)
            strength = batch.get("structure_strength", batch.get("structure_intent_mask"))
            if strength is None:
                strength = torch.ones_like(per_case)
            strength = strength.to(device=x1.device, dtype=x1.dtype).reshape(-1).clamp(0.0, 1.0)
            loss_structure = torch.mean(per_case * strength)
            structure_strength_mean = strength.mean()
        else:
            loss_structure = x1.new_tensor(0.0)
            structure_strength_mean = x1.new_tensor(0.0)
        if self.cfg.use_hypergraph_plan and condition.get("hypergraph_plan_hat") is not None and batch.get("hypergraph_plan_target") is not None:
            plan_hat = condition["hypergraph_plan_hat"]
            plan_target = batch["hypergraph_plan_target"].to(device=x1.device, dtype=x1.dtype)
            plan_mask = batch.get("hypergraph_plan_mask")
            if plan_mask is None:
                plan_mask = torch.ones_like(plan_target)
            else:
                plan_mask = plan_mask.to(device=x1.device, dtype=x1.dtype)
            dim = min(plan_hat.shape[-1], plan_target.shape[-1], plan_mask.shape[-1])
            if dim > 0:
                plan_hat = plan_hat[:, :dim]
                plan_target = plan_target[:, :dim]
                plan_mask = plan_mask[:, :dim].clamp(0.0, 1.0)
                denom = plan_mask.sum()
                if float(denom.detach().cpu()) > 0.0:
                    loss_hypergraph_plan = ((plan_hat - plan_target).square() * plan_mask).sum() / denom.clamp_min(1.0)
                else:
                    loss_hypergraph_plan = x1.new_tensor(0.0)
                hypergraph_plan_mask_fraction = plan_mask.mean()
            else:
                loss_hypergraph_plan = x1.new_tensor(0.0)
                hypergraph_plan_mask_fraction = x1.new_tensor(0.0)
        else:
            loss_hypergraph_plan = x1.new_tensor(0.0)
            hypergraph_plan_mask_fraction = x1.new_tensor(0.0)
        weights = {
            "flow_weight": 1.0,
            "set_center_weight": 0.15,
            "structure_match_weight": 0.20,
            "hypergraph_plan_weight": 0.0,
            "heat_condition_weight": 0.0,
            "structure_map_match_weight": 0.0,
            "count_weight": 0.1,
            "behavior_align_weight": 0.03,
            "organization_align_weight": 0.03,
            "validity_prior_weight": 0.05,
        }
        if loss_weights:
            weights.update({str(k): float(v) for k, v in loss_weights.items()})
        total = (
            weights["flow_weight"] * loss_flow
            + weights["set_center_weight"] * loss_set_center
            + weights["structure_match_weight"] * loss_structure
            + weights["hypergraph_plan_weight"] * loss_hypergraph_plan
            + weights["count_weight"] * loss_count
            + weights["behavior_align_weight"] * loss_behavior
            + weights["organization_align_weight"] * loss_organization
            + weights["validity_prior_weight"] * loss_validity
        )
        metrics = {
            "loss_total": total.detach(),
            "loss_flow": loss_flow.detach(),
            "loss_set_center": loss_set_center.detach(),
            "loss_structure_match": loss_structure.detach(),
            "loss_hypergraph_plan": loss_hypergraph_plan.detach(),
            "loss_count": loss_count.detach(),
            "loss_behavior": loss_behavior.detach(),
            "loss_organization": loss_organization.detach(),
            "loss_validity_prior": loss_validity.detach(),
            "count_accuracy": count_accuracy.detach(),
            "latent_align_weight_mean": align_weight_mean.detach(),
            "structure_strength_mean": structure_strength_mean.detach(),
            "hypergraph_plan_mask_fraction": hypergraph_plan_mask_fraction.detach(),
        }
        return total, metrics

    def _set_center_loss(self, pred_design: torch.Tensor, true_design: torch.Tensor, true_count: torch.Tensor) -> torch.Tensor:
        pred_centers = self.decode_centers_norm_from_design_vec(pred_design).clone()
        true_centers = true_design[:, : self.max_num_modules * 2].reshape(true_design.shape[0], self.max_num_modules, 2).clone()
        pred_centers[..., 0] *= float(self.cfg.domain_length_x)
        pred_centers[..., 1] *= float(self.cfg.domain_length_y)
        true_centers[..., 0] *= float(self.cfg.domain_length_x)
        true_centers[..., 1] *= float(self.cfg.domain_length_y)
        losses = []
        for i in range(pred_design.shape[0]):
            n = int(true_count[i].item())
            if n <= 0:
                continue
            pred_i = pred_centers[i, :n]
            true_i = true_centers[i, :n]
            dist = torch.cdist(pred_i[None], true_i[None], p=2).squeeze(0)
            losses.append(0.5 * (dist.min(dim=1).values.square().mean() + dist.min(dim=0).values.square().mean()))
        return torch.stack(losses).mean() if losses else pred_design.new_tensor(0.0)

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
        design_intent_vector: Optional[torch.Tensor | np.ndarray] = None,
        objective_weight_vector: Optional[torch.Tensor | np.ndarray] = None,
        field_intent_maps: Optional[torch.Tensor | np.ndarray] = None,
        structure_intent_vector: Optional[torch.Tensor | np.ndarray] = None,
        structure_intent_maps: Optional[torch.Tensor | np.ndarray] = None,
        heat_condition_vector: Optional[torch.Tensor | np.ndarray] = None,
        heat_condition_mask: Optional[torch.Tensor | np.ndarray] = None,
        guidance_scale: float = 1.0,
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
        intent = torch.as_tensor(design_intent_vector, dtype=torch.float32, device=device) if design_intent_vector is not None else None
        objective = torch.as_tensor(objective_weight_vector, dtype=torch.float32, device=device) if objective_weight_vector is not None else None
        maps = torch.as_tensor(field_intent_maps, dtype=torch.float32, device=device) if field_intent_maps is not None else None
        structure_vec = torch.as_tensor(structure_intent_vector, dtype=torch.float32, device=device) if structure_intent_vector is not None else None
        structure_maps = torch.as_tensor(structure_intent_maps, dtype=torch.float32, device=device) if structure_intent_maps is not None else None
        heat_vec = torch.as_tensor(heat_condition_vector, dtype=torch.float32, device=device) if heat_condition_vector is not None else None
        heat_mask = torch.as_tensor(heat_condition_mask, dtype=torch.float32, device=device) if heat_condition_mask is not None else None
        if intent is not None and intent.ndim == 1:
            intent = intent[None, :].expand(int(n_samples), -1).contiguous()
        if objective is not None and objective.ndim == 1:
            objective = objective[None, :].expand(int(n_samples), -1).contiguous()
        if maps is not None and maps.ndim == 3:
            maps = maps[None].expand(int(n_samples), -1, -1, -1).contiguous()
        if structure_vec is not None and structure_vec.ndim == 1:
            structure_vec = structure_vec[None, :].expand(int(n_samples), -1).contiguous()
        if structure_maps is not None and structure_maps.ndim == 3:
            structure_maps = structure_maps[None].expand(int(n_samples), -1, -1, -1).contiguous()
        if heat_vec is not None and heat_vec.ndim == 1:
            heat_vec = heat_vec[None, :].expand(int(n_samples), -1).contiguous()
        if heat_mask is not None and heat_mask.ndim == 1:
            heat_mask = heat_mask[None, :].expand(int(n_samples), -1).contiguous()
        condition = self.encode_condition(
            target,
            design_intent_vector=intent,
            objective_weight_vector=objective,
            field_intent_maps=maps,
            structure_intent_vector=structure_vec,
            structure_intent_maps=structure_maps,
            heat_condition_vector=heat_vec,
            heat_condition_mask=heat_mask,
        )
        uncond_condition = None
        if float(guidance_scale) > 1.0:
            uncond_condition = self.encode_condition(
                target,
                design_intent_vector=torch.zeros_like(intent) if intent is not None else None,
                objective_weight_vector=torch.zeros_like(objective) if objective is not None else None,
                field_intent_maps=torch.zeros_like(maps) if maps is not None else None,
                structure_intent_vector=torch.zeros_like(structure_vec) if structure_vec is not None else None,
                structure_intent_maps=torch.zeros_like(structure_maps) if structure_maps is not None else None,
                heat_condition_vector=torch.zeros_like(heat_vec) if heat_vec is not None else None,
                heat_condition_mask=torch.zeros_like(heat_mask) if heat_mask is not None else None,
            )
        steps = max(int(n_steps), 1)
        dt = 1.0 / float(steps)
        solver = str(self.cfg.ode_solver).lower()
        for step in range(steps):
            t0 = torch.full((int(n_samples), 1), step / float(steps), device=device, dtype=x.dtype)
            v0 = self.velocity(x, t0, condition)
            if uncond_condition is not None:
                v0_uncond = self.velocity(x, t0, uncond_condition)
                v0 = v0_uncond + float(guidance_scale) * (v0 - v0_uncond)
            if solver == "heun":
                x_euler = x + dt * v0
                t1 = torch.full((int(n_samples), 1), (step + 1) / float(steps), device=device, dtype=x.dtype)
                v1 = self.velocity(x_euler, t1, condition)
                if uncond_condition is not None:
                    v1_uncond = self.velocity(x_euler, t1, uncond_condition)
                    v1 = v1_uncond + float(guidance_scale) * (v1 - v1_uncond)
                x = x + 0.5 * dt * (v0 + v1)
            else:
                x = x + dt * v0

        count_probs = F.softmax(condition["count_logits"], dim=-1) if self.count_head is not None else torch.zeros(
            int(n_samples), self.max_num_modules + 1, device=device
        )
        constraints = self._constraint_values(target)
        normalized_count_mode = str(count_mode).lower().strip()
        if normalized_count_mode in {"uniform", "constraint_uniform"}:
            sampled_counts = []
            for i in range(int(n_samples)):
                lo = int(constraints["n_min"][i].item())
                hi = int(constraints["n_max"][i].item())
                if hi <= lo:
                    sampled_counts.append(torch.tensor(lo, dtype=torch.long, device=device))
                else:
                    sampled_counts.append(torch.randint(lo, hi + 1, (1,), generator=generator, device=device, dtype=torch.long)[0])
            raw_counts = torch.stack(sampled_counts, dim=0)
        elif self.count_head is None:
            raw_counts = torch.full((int(n_samples),), self.max_num_modules, dtype=torch.long, device=device)
            raw_counts = raw_counts.clamp(constraints["n_min"], constraints["n_max"])
        elif normalized_count_mode == "sample":
            raw_counts = torch.multinomial(count_probs, num_samples=1, generator=generator).reshape(-1)
            raw_counts = raw_counts.clamp(constraints["n_min"], constraints["n_max"])
        else:
            raw_counts = torch.argmax(count_probs, dim=-1)
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
            heat_conditioned = self.cfg.slot_identity_mode == "heat_conditioned" and heat_vec is not None
            if heat_conditioned:
                if heat_mask is not None:
                    active_slots = torch.nonzero(heat_mask[i] > 0.5, as_tuple=False).reshape(-1)
                    if active_slots.numel() >= count:
                        order = active_slots
                    else:
                        fallback = torch.arange(self.max_num_modules, device=device)
                        order = torch.cat([active_slots, fallback[~torch.isin(fallback, active_slots)]], dim=0)
                else:
                    order = torch.arange(self.max_num_modules, device=device)
            else:
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
                preserve_order=heat_conditioned,
            )
            sorted_repaired = repaired if heat_conditioned else sort_centers_xy(repaired)
            mask = np.zeros((self.max_num_modules,), dtype=np.float32)
            mask[: sorted_repaired.shape[0]] = 1.0
            heat_out = None
            slot_ids = np.asarray(chosen[: sorted_repaired.shape[0]], dtype=np.int64)
            if heat_conditioned and heat_vec is not None:
                heat_selected = heat_vec[i, : self.max_num_modules].detach().cpu().numpy().astype(np.float32)[chosen]
                heat_selected = heat_selected[: sorted_repaired.shape[0]] * float(self.cfg.heat_power_scale)
                heat_out = heat_selected
            elif heat is not None:
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
                    "slot_ids": slot_ids,
                    "module_ids": slot_ids,
                    "slot_identity_mode": self.cfg.slot_identity_mode,
                    "raw_design_vec": x[i].detach().cpu().numpy().astype(np.float32),
                    "raw_count": int(raw_counts[i].item()),
                    "mask_scores": mask_scores[i].detach().cpu().numpy().astype(np.float32),
                    "count_probabilities": count_probs[i].detach().cpu().numpy().astype(np.float32),
                    "behavior_latent_hat": condition["behavior_latent_hat"][i].detach().cpu().numpy().astype(np.float32),
                    "organization_latent_hat": condition["organization_latent_hat"][i].detach().cpu().numpy().astype(np.float32),
                    "validity": validity,
                    **(
                        {
                            "hypergraph_plan_hat": condition["hypergraph_plan_hat"][i].detach().cpu().numpy().astype(np.float32),
                            "hypergraph_plan_embedding": condition["hypergraph_plan_embedding"][i].detach().cpu().numpy().astype(np.float32),
                        }
                        if "hypergraph_plan_hat" in condition and "hypergraph_plan_embedding" in condition
                        else {}
                    ),
                }
            )
        if was_training:
            self.train()
        return outputs
