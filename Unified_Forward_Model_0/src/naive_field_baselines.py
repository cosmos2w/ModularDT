"""External naive neural-field baselines without hypergraph organization.

These models intentionally avoid the HONF pathway: no environment tokens,
hyperedges, incidence matrices, organizer coordinates, local surrogates, or
interface heads. They map ``D, c, q -> U(q)`` with plain MLP-style machinery.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List

import torch
import torch.nn as nn

from unified_types import BatchData


MODEL_TYPES = {"flat_layout_mlp", "query_deepsets_mlp"}
POOL_MODES = {"sum", "mean", "sum_mean", "sum_mean_max"}
EPS = 1.0e-6


@dataclass
class NaiveFieldBaselineConfig:
    model_type: str = "query_deepsets_mlp"

    field_dim: int = 5
    max_num_modules: int = 12
    module_feature_dim: int = 8
    global_context_dim: int = 8

    hidden_dim: int = 256
    num_layers: int = 5
    dropout: float = 0.05
    use_layer_norm: bool = True

    query_fourier_frequencies: int = 4
    relative_fourier_frequencies: int = 2

    pool_mode: str = "sum_mean_max"

    include_nearest_distance_features: bool = True
    include_global_context: bool = True

    domain_length_x: float = 12.0
    domain_length_y: float = 4.0
    module_radius: float = 0.45

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of: {', '.join(sorted(MODEL_TYPES))}")
        if self.pool_mode not in POOL_MODES:
            raise ValueError(f"pool_mode must be one of: {', '.join(sorted(POOL_MODES))}")
        if int(self.num_layers) < 1:
            raise ValueError("num_layers must be >= 1.")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NaiveFieldBaselineConfig":
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in dict(payload).items() if key in names})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FourierFeatures(nn.Module):
    """Append sin/cos Fourier features using powers-of-two frequencies."""

    def __init__(self, num_frequencies: int):
        super().__init__()
        self.num_frequencies = max(0, int(num_frequencies))
        if self.num_frequencies > 0:
            frequencies = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)) * math.pi
        else:
            frequencies = torch.empty(0, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_frequencies <= 0:
            return x
        angles = x.unsqueeze(-2) * self.frequencies.to(device=x.device, dtype=x.dtype).view(*([1] * (x.ndim - 1)), -1, 1)
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-2).flatten(start_dim=-2)
        return torch.cat([x, encoded], dim=-1)

    def output_dim(self, input_dim: int) -> int:
        return int(input_dim) * (1 + 2 * self.num_frequencies)


class FeedForward(nn.Module):
    """MLP with a lazy input layer for concise feature construction."""

    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int, dropout: float, use_layer_norm: bool):
        super().__init__()
        layers: List[nn.Module] = []
        layers.extend([nn.LazyLinear(hidden_dim), nn.GELU()])
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(max(0, int(num_layers) - 2)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlatLayoutMLPField(nn.Module):
    """Slot-order-dependent lower baseline over flattened module layout."""

    def __init__(self, config: NaiveFieldBaselineConfig):
        super().__init__()
        self.config = config
        self.query_fourier = FourierFeatures(int(config.query_fourier_frequencies))
        self.mlp = FeedForward(
            hidden_dim=int(config.hidden_dim),
            out_dim=int(config.field_dim),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
            use_layer_norm=bool(config.use_layer_norm),
        )

    def forward(self, batch: BatchData) -> Dict[str, Any]:
        cfg = self.config
        query_features = self.query_fourier(_normalized_xy(batch.query_xy.float(), cfg))
        layout = _flat_layout_features(batch, cfg)
        layout = layout.unsqueeze(1).expand(-1, query_features.shape[1], -1)
        pieces = [query_features, layout]
        if cfg.include_global_context:
            global_context = batch.global_context.float()[..., : int(cfg.global_context_dim)]
            pieces.append(global_context.unsqueeze(1).expand(-1, query_features.shape[1], -1))
        pred = self.mlp(torch.cat(pieces, dim=-1))
        return {
            "pred_field": pred,
            "model_type": "flat_layout_mlp",
            "uses_hyper_context": pred.new_tensor(0.0),
            "active_edge_count": pred.new_tensor(float("nan")),
            "A_mh_entropy": pred.new_tensor(float("nan")),
            "A_eh_entropy": pred.new_tensor(float("nan")),
        }


class QueryDeepSetsMLPField(nn.Module):
    """Permutation-invariant query-conditioned DeepSets neural field."""

    def __init__(self, config: NaiveFieldBaselineConfig):
        super().__init__()
        self.config = config
        self.query_fourier = FourierFeatures(int(config.query_fourier_frequencies))
        self.relative_fourier = FourierFeatures(int(config.relative_fourier_frequencies))
        self.module_encoder = FeedForward(
            hidden_dim=int(config.hidden_dim),
            out_dim=int(config.hidden_dim),
            num_layers=max(2, int(config.num_layers) - 1),
            dropout=float(config.dropout),
            use_layer_norm=bool(config.use_layer_norm),
        )
        self.query_encoder = FeedForward(
            hidden_dim=int(config.hidden_dim),
            out_dim=int(config.hidden_dim),
            num_layers=2,
            dropout=float(config.dropout),
            use_layer_norm=bool(config.use_layer_norm),
        )
        self.final_mlp = FeedForward(
            hidden_dim=int(config.hidden_dim),
            out_dim=int(config.field_dim),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
            use_layer_norm=bool(config.use_layer_norm),
        )

    def forward(self, batch: BatchData) -> Dict[str, Any]:
        cfg = self.config
        query_xy = batch.query_xy.float()
        module_centers = batch.module_centers.float()
        module_present = batch.module_present.float()
        module_features = batch.module_features.float()[..., : int(cfg.module_feature_dim)]

        rel = _relative_features(query_xy, module_centers, cfg)
        rel_fourier = self.relative_fourier(rel)
        features = torch.cat(
            [
                module_features[:, None, :, :].expand(-1, query_xy.shape[1], -1, -1),
                module_present[:, None, :, None].expand(-1, query_xy.shape[1], -1, -1),
                rel,
                rel_fourier[..., rel.shape[-1] :],
            ],
            dim=-1,
        )
        pair_embedding = self.module_encoder(features)
        mask = module_present[:, None, :, None].to(dtype=pair_embedding.dtype)
        pair_embedding = pair_embedding * mask
        pooled = _pool_modules(pair_embedding, mask, cfg.pool_mode)
        query_state = self.query_encoder(self.query_fourier(_normalized_xy(query_xy, cfg)))

        pieces = [query_state, pooled]
        if cfg.include_nearest_distance_features:
            pieces.append(_nearest_features(rel, module_present))
        if cfg.include_global_context:
            global_context = batch.global_context.float()[..., : int(cfg.global_context_dim)]
            pieces.append(global_context.unsqueeze(1).expand(-1, query_xy.shape[1], -1))
        final_features = torch.cat(pieces, dim=-1)
        pred = self.final_mlp(final_features)
        return {
            "pred_field": pred,
            "model_type": "query_deepsets_mlp",
            "pooled_module_summary_norm": pooled.detach().norm(dim=-1).mean(),
            "uses_hyper_context": pred.new_tensor(0.0),
            "active_edge_count": pred.new_tensor(float("nan")),
            "A_mh_entropy": pred.new_tensor(float("nan")),
            "A_eh_entropy": pred.new_tensor(float("nan")),
        }


class NaiveFieldBaseline(nn.Module):
    """Wrapper selecting one naive no-H field baseline."""

    def __init__(self, config: NaiveFieldBaselineConfig):
        super().__init__()
        self.config = config
        if config.model_type == "flat_layout_mlp":
            self.model = FlatLayoutMLPField(config)
        elif config.model_type == "query_deepsets_mlp":
            self.model = QueryDeepSetsMLPField(config)
        else:  # pragma: no cover - config validation catches this.
            raise ValueError(f"Unsupported model_type={config.model_type!r}.")

    def forward(self, batch: BatchData) -> Dict[str, Any]:
        return self.model(batch)


def build_naive_baseline_from_config(config_dict: Dict[str, Any]) -> NaiveFieldBaseline:
    return NaiveFieldBaseline(NaiveFieldBaselineConfig.from_dict(config_dict))


def _normalized_xy(query_xy: torch.Tensor, cfg: NaiveFieldBaselineConfig) -> torch.Tensor:
    return torch.stack(
        [
            query_xy[..., 0] / max(float(cfg.domain_length_x), EPS),
            query_xy[..., 1] / max(float(cfg.domain_length_y), EPS),
        ],
        dim=-1,
    )


def _flat_layout_features(batch: BatchData, cfg: NaiveFieldBaselineConfig) -> torch.Tensor:
    module_centers = batch.module_centers.float()
    module_centers_norm = torch.stack(
        [
            module_centers[..., 0] / max(float(cfg.domain_length_x), EPS),
            module_centers[..., 1] / max(float(cfg.domain_length_y), EPS),
        ],
        dim=-1,
    )
    module_present = batch.module_present.float().unsqueeze(-1)
    module_features = batch.module_features.float()[..., : int(cfg.module_feature_dim)]
    layout = torch.cat([module_centers_norm, module_present, module_features], dim=-1)
    return layout.flatten(start_dim=1)


def _relative_features(query_xy: torch.Tensor, module_centers: torch.Tensor, cfg: NaiveFieldBaselineConfig) -> torch.Tensor:
    delta = query_xy[:, :, None, :] - module_centers[:, None, :, :]
    dx = delta[..., 0:1]
    dy = delta[..., 1:2]
    lx = max(float(cfg.domain_length_x), EPS)
    ly = max(float(cfg.domain_length_y), EPS)
    diag = max(math.sqrt(lx * lx + ly * ly), EPS)
    distance = torch.sqrt(dx.square() + dy.square() + EPS)
    return torch.cat(
        [
            dx / lx,
            dy / ly,
            distance / diag,
            torch.relu(dx) / lx,
            torch.relu(-dx) / lx,
            dy.abs() / ly,
        ],
        dim=-1,
    )


def _pool_modules(pair_embedding: torch.Tensor, mask: torch.Tensor, pool_mode: str) -> torch.Tensor:
    raw_count = mask.sum(dim=2)
    count = raw_count.clamp_min(1.0)
    pooled_sum = pair_embedding.sum(dim=2)
    pooled_mean = pooled_sum / count
    if pool_mode == "sum":
        return pooled_sum
    if pool_mode == "mean":
        return pooled_mean
    if pool_mode == "sum_mean":
        return torch.cat([pooled_sum, pooled_mean], dim=-1)
    very_negative = torch.finfo(pair_embedding.dtype).min
    pooled_max = pair_embedding.masked_fill(mask <= 0, very_negative).amax(dim=2)
    pooled_max = torch.where(raw_count > 0, pooled_max, torch.zeros_like(pooled_max))
    return torch.cat([pooled_sum, pooled_mean, pooled_max], dim=-1)


def _nearest_features(rel: torch.Tensor, module_present: torch.Tensor) -> torch.Tensor:
    large = torch.full_like(rel[..., 2], 1.0e6)
    distance = torch.where(module_present[:, None, :] > 0, rel[..., 2], large)
    nearest_idx = distance.argmin(dim=-1)
    nearest = torch.gather(rel, 2, nearest_idx[:, :, None, None].expand(-1, -1, 1, rel.shape[-1])).squeeze(2)
    has_module = (module_present.sum(dim=-1) > 0).to(dtype=rel.dtype)[:, None, None]
    return nearest * has_module
