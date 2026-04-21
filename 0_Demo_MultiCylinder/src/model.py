from __future__ import annotations

"""

PyTorch model for hypergraph-organized neural-field reconstruction.

Architecture overview
---------------------
1) Organizer / Encoder
   * Inputs: structure-only descriptors (Re, number of cylinders, padded centers)
   * Outputs: soft organized state consisting of module nodes, environment tokens,
     hyperedge/group states, and soft incidences.
2) Behavior Head
   * Maps the organized state to compact latent descriptors such as mean-field and
     dynamic latents plus a dominant-frequency estimate.
3) Neural Field Decoder
   * Queries the latent organized state at arbitrary (x, y, tau) and returns
     reconstructed multi-field values [u, v, p, omega].

The interfaces intentionally allow future extension to active/reactive modular
systems. Optional module-level and global conditioning tensors can be threaded
through the same API without changing the higher-level training scripts.

"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Helper modules ---------------------------------


class FourierEncoder(nn.Module):
    """Sin / cos positional encoding for low-dimensional continuous inputs.

    Input shape:
        [..., D]
    Output shape:
        [..., D * (1 + 2 * num_frequencies)] if include_input=True
    """

    def __init__(self, input_dim: int, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        freq_bands = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + (2 * self.input_dim * self.num_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = [x] if self.include_input else []
        # Expand the last dimension with frequencies.
        for freq in self.freq_bands:
            pieces.append(torch.sin(2.0 * math.pi * freq * x))
            pieces.append(torch.cos(2.0 * math.pi * freq * x))
        return torch.cat(pieces, dim=-1)


class MLP(nn.Module):
    """Simple configurable MLP block used across organizer and decoder heads."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 0) + [out_dim]
        act_layer = nn.GELU if activation == "gelu" else nn.SiLU

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act_layer())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SineLayer(nn.Module):
    """SIREN layer for decoder alternatives.

    Reference-style initialization is used only inside the decoder branch when
    decoder_type="siren". The rest of the architecture stays standard MLP.
    """

    def __init__(self, in_dim: int, out_dim: int, *, is_first: bool, omega_0: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega_0 = float(omega_0)
        self.is_first = bool(is_first)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenNet(nn.Module):
    """Small SIREN-style network used as an optional neural-field decoder."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, omega_0: float = 30.0):
        super().__init__()
        if num_layers < 2:
            raise ValueError("SirenNet requires num_layers >= 2")
        layers = [SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0)]
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        self.hidden = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        return self.final(x)


# --------------------------- Configuration dataclass ---------------------------


@dataclass
class ModelConfig:
    """Configuration for the hypergraph-organized neural field model."""

    max_num_cylinders: int = 8
    domain_length_x: float = 24.0
    domain_length_y: float = 12.0
    re_scale: float = 200.0
    num_env_tokens_x: int = 16
    num_env_tokens_y: int = 8
    num_hyperedges: int = 4
    hidden_dim: int = 128
    behavior_dim: int = 128
    latent_dim: int = 128
    message_passing_steps: int = 2
    coord_fourier_frequencies: int = 4
    structure_fourier_frequencies: int = 1
    query_fourier_frequencies: int = 4
    decoder_hidden_dim: int = 256
    decoder_num_layers: int = 4
    decoder_type: str = "mlp_fourier"  # "mlp_fourier" | "siren"
    dropout: float = 0.05
    use_layer_norm: bool = True
    future_module_feature_dim: int = 0
    future_global_feature_dim: int = 0

    @classmethod
    def from_dict(cls, payload: Dict) -> "ModelConfig":
        valid = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in valid})


# --------------------------- Organizer / Encoder ------------------------------


class HypergraphOrganizer(nn.Module):
    """Structure-conditioned organizer that produces a soft organized state.

    Tensor shape convention
    -----------------------
    Inputs:
        re_values:      [B, 1]
        num_cylinders:  [B, 1]
        centers:        [B, N_max, 2]
        cyl_mask:       [B, N_max]
    Outputs:
        module_state:   [B, N_max, D]
        env_state:      [B, M_env, D]
        hyper_state:    [B, K_h, D]
        A_me:           [B, N_max, M_env]
        A_mh:           [B, N_max, K_h]
        A_eh:           [B, M_env, K_h]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.max_num_cylinders = cfg.max_num_cylinders

        # Fourier features for structure-space coordinates.
        self.module_coord_encoder = FourierEncoder(2, cfg.structure_fourier_frequencies, include_input=True)
        self.env_coord_encoder = FourierEncoder(2, cfg.structure_fourier_frequencies, include_input=True)

        global_in_dim = 2 + cfg.future_global_feature_dim  # [Re_scaled, Nc_scaled] + optional extra globals
        module_in_dim = self.module_coord_encoder.output_dim + global_in_dim + cfg.future_module_feature_dim
        env_in_dim = self.env_coord_encoder.output_dim + global_in_dim

        self.module_encoder = MLP(
            in_dim=module_in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.hidden_dim,
            num_layers=3,
            dropout=cfg.dropout,
            layer_norm=cfg.use_layer_norm,
        )
        self.env_encoder = MLP(
            in_dim=env_in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.hidden_dim,
            num_layers=3,
            dropout=cfg.dropout,
            layer_norm=cfg.use_layer_norm,
        )

        # Learnable hyperedge slots. These become shared interaction group states.
        self.hyperedge_embeddings = nn.Parameter(torch.randn(cfg.num_hyperedges, cfg.hidden_dim) * 0.02)
        self.hyperedge_context = nn.Linear(global_in_dim, cfg.hidden_dim)

        # Soft incidence scorers.
        rel_dim = 5  # dx, dy, dist, downstream(+x), upstream(-x)
        self.me_score = MLP(2 * cfg.hidden_dim + rel_dim, cfg.hidden_dim, 1, num_layers=3, dropout=cfg.dropout)
        self.mh_score = MLP(2 * cfg.hidden_dim, cfg.hidden_dim, 1, num_layers=2, dropout=cfg.dropout)
        self.eh_score = MLP(2 * cfg.hidden_dim, cfg.hidden_dim, 1, num_layers=2, dropout=cfg.dropout)

        # Lightweight message-passing update blocks.
        self.module_update = MLP(3 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=2, dropout=cfg.dropout)
        self.env_update = MLP(3 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=2, dropout=cfg.dropout)
        self.hyper_update = MLP(3 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=2, dropout=cfg.dropout)

        self.register_buffer("env_token_coords", self._make_env_token_grid(), persistent=False)

    def _make_env_token_grid(self) -> torch.Tensor:
        xs = (torch.arange(self.cfg.num_env_tokens_x, dtype=torch.float32) + 0.5) / float(self.cfg.num_env_tokens_x)
        ys = (torch.arange(self.cfg.num_env_tokens_y, dtype=torch.float32) + 0.5) / float(self.cfg.num_env_tokens_y)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    def _global_features(
        self,
        re_values: torch.Tensor,
        num_cylinders: torch.Tensor,
        extra_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        re_scaled = re_values / max(self.cfg.re_scale, 1e-6)
        nc_scaled = num_cylinders / max(float(self.cfg.max_num_cylinders), 1.0)
        feats = [re_scaled, nc_scaled]
        if extra_global is not None:
            feats.append(extra_global)
        return torch.cat(feats, dim=-1)

    def _periodic_relative_features(
        self,
        module_coords: torch.Tensor,
        env_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Compute periodic relative geometry features for module-env pairs.

        Args:
            module_coords: [B, N, 2] in normalized domain coordinates [0, 1]
            env_coords:    [B, M, 2] in normalized domain coordinates [0, 1]
        Returns:
            rel: [B, N, M, 5]
        """
        dx = module_coords[:, :, None, 0] - env_coords[:, None, :, 0]
        dy = module_coords[:, :, None, 1] - env_coords[:, None, :, 1]
        dx = (dx + 0.5) % 1.0 - 0.5
        dy = (dy + 0.5) % 1.0 - 0.5
        dist = torch.sqrt(dx.square() + dy.square() + 1e-8)
        downstream = torch.clamp(-dx, min=0.0)  # flow is +x so wake extends for env x > cyl x
        upstream = torch.clamp(dx, min=0.0)
        return torch.stack([dx, dy, dist, downstream, upstream], dim=-1)

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.to(values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=dim).clamp_min(1e-6)
        return (values * mask).sum(dim=dim) / denom

    def _normalize_incidence(self, weights: torch.Tensor, dim: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            while mask.ndim < weights.ndim:
                mask = mask.unsqueeze(-1)
            weights = weights.masked_fill(mask <= 0, float("-inf"))
        return torch.softmax(weights, dim=dim)

    def forward(
        self,
        re_values: torch.Tensor,
        num_cylinders: torch.Tensor,
        centers: torch.Tensor,
        cyl_mask: torch.Tensor,
        extra_global: Optional[torch.Tensor] = None,
        extra_module: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = centers.device
        batch_size, n_max, _ = centers.shape
        global_feats = self._global_features(re_values, num_cylinders, extra_global=extra_global)  # [B, G]

        module_coords_norm = centers.clone()
        module_coords_norm[..., 0] = module_coords_norm[..., 0] / max(self.cfg.domain_length_x, 1e-6)
        module_coords_norm[..., 1] = module_coords_norm[..., 1] / max(self.cfg.domain_length_y, 1e-6)
        module_coord_feat = self.module_coord_encoder(module_coords_norm)
        global_expand = global_feats[:, None, :].expand(batch_size, n_max, -1)
        module_inputs = [module_coord_feat, global_expand]
        if extra_module is not None:
            module_inputs.append(extra_module)
        module_state = self.module_encoder(torch.cat(module_inputs, dim=-1))
        module_state = module_state * cyl_mask.unsqueeze(-1)

        env_coords = self.env_token_coords.to(device=device).unsqueeze(0).expand(batch_size, -1, -1)
        env_coord_feat = self.env_coord_encoder(env_coords)
        env_global = global_feats[:, None, :].expand(batch_size, env_coords.shape[1], -1)
        env_state = self.env_encoder(torch.cat([env_coord_feat, env_global], dim=-1))

        hyper_state = self.hyperedge_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        hyper_state = hyper_state + self.hyperedge_context(global_feats).unsqueeze(1)

        rel_me = self._periodic_relative_features(module_coords_norm, env_coords)  # [B, N, M, 5]

        for _ in range(self.cfg.message_passing_steps):
            n_env = env_state.shape[1]
            n_hyp = hyper_state.shape[1]

            module_expand = module_state[:, :, None, :].expand(-1, -1, n_env, -1)
            env_expand = env_state[:, None, :, :].expand(-1, n_max, -1, -1)
            me_logits = self.me_score(torch.cat([module_expand, env_expand, rel_me], dim=-1)).squeeze(-1)
            valid_pair_mask = cyl_mask[:, :, None].expand_as(me_logits)
            A_me = torch.softmax(me_logits.masked_fill(valid_pair_mask <= 0, -1e9), dim=-1)

            module_h = module_state[:, :, None, :].expand(-1, -1, n_hyp, -1)
            hyper_h = hyper_state[:, None, :, :].expand(-1, n_max, -1, -1)
            mh_logits = self.mh_score(torch.cat([module_h, hyper_h], dim=-1)).squeeze(-1)
            A_mh = torch.softmax(mh_logits.masked_fill(cyl_mask[:, :, None] <= 0, -1e9), dim=-1)

            env_h = env_state[:, :, None, :].expand(-1, -1, n_hyp, -1)
            hyper_e = hyper_state[:, None, :, :].expand(-1, n_env, -1, -1)
            eh_logits = self.eh_score(torch.cat([env_h, hyper_e], dim=-1)).squeeze(-1)
            A_eh = torch.softmax(eh_logits, dim=-1)

            # Aggregate messages.
            env_to_module = torch.einsum("bnm,bmd->bnd", A_me, env_state)
            hyp_to_module = torch.einsum("bnk,bkd->bnd", A_mh, hyper_state)
            module_delta = self.module_update(torch.cat([module_state, env_to_module, hyp_to_module], dim=-1))
            module_state = (module_state + module_delta) * cyl_mask.unsqueeze(-1)

            mod_to_env = torch.einsum("bnm,bnd->bmd", A_me, module_state)
            hyp_to_env = torch.einsum("bmk,bkd->bmd", A_eh, hyper_state)
            env_delta = self.env_update(torch.cat([env_state, mod_to_env, hyp_to_env], dim=-1))
            env_state = env_state + env_delta

            mod_to_hyp = torch.einsum("bnk,bnd->bkd", A_mh, module_state)
            env_to_hyp = torch.einsum("bmk,bmd->bkd", A_eh, env_state)
            hyp_delta = self.hyper_update(torch.cat([hyper_state, mod_to_hyp, env_to_hyp], dim=-1))
            hyper_state = hyper_state + hyp_delta

        return {
            "module_state": module_state,
            "env_state": env_state,
            "hyper_state": hyper_state,
            "A_me": A_me,
            "A_mh": A_mh,
            "A_eh": A_eh,
            "env_coords": env_coords,
            "module_coords_norm": module_coords_norm,
            "global_features": global_feats,
            "cyl_mask": cyl_mask,
        }


# ----------------------------- Behavior head ----------------------------------


class BehaviorHead(nn.Module):
    """Maps the organized state into compact behavior-manifold descriptors."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        pooled_dim = 3 * cfg.hidden_dim
        self.behavior_mlp = MLP(
            in_dim=pooled_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.behavior_dim,
            num_layers=3,
            dropout=cfg.dropout,
            layer_norm=cfg.use_layer_norm,
        )
        self.mean_latent_head = nn.Linear(cfg.behavior_dim, cfg.latent_dim)
        self.dynamic_latent_head = nn.Linear(cfg.behavior_dim, cfg.latent_dim)
        self.freq_head = nn.Linear(cfg.behavior_dim, 1)

    def forward(self, organized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        module_state = organized["module_state"]
        env_state = organized["env_state"]
        hyper_state = organized["hyper_state"]
        cyl_mask = organized["cyl_mask"]
        # [B, D]
        module_pool = HypergraphOrganizer.masked_mean(module_state, cyl_mask, dim=1)
        env_pool = env_state.mean(dim=1)
        hyper_pool = hyper_state.mean(dim=1)

        pooled = torch.cat([module_pool, env_pool, hyper_pool], dim=-1)
        behavior = self.behavior_mlp(pooled)
        mean_latent = self.mean_latent_head(behavior)
        dynamic_latent = self.dynamic_latent_head(behavior)
        freq_pred = F.softplus(self.freq_head(behavior))

        return {
            "behavior_latent": behavior,
            "mean_latent": mean_latent,
            "dynamic_latent": dynamic_latent,
            "freq_pred": freq_pred,
        }


# ----------------------------- Neural field decoder ----------------------------


class AttentionAggregator(nn.Module):
    """Query-to-context attention helper for module/env/hyper tokens.

    Args:
        query:   [B, Q, D]
        context: [B, N_ctx, D]
        rel:     optional [B, Q, N_ctx, R]
    Returns:
        aggregated: [B, Q, D]
    """

    def __init__(self, hidden_dim: int, rel_dim: int = 0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_score = MLP(rel_dim, hidden_dim, 1, num_layers=2) if rel_dim > 0 else None

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        rel: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_proj = self.query_proj(query)
        k_proj = self.context_proj(context)
        logits = torch.einsum("bqd,bnd->bqn", q_proj, k_proj) / math.sqrt(float(self.hidden_dim))
        if rel is not None and self.rel_score is not None:
            logits = logits + self.rel_score(rel).squeeze(-1)
        if context_mask is not None:
            logits = logits.masked_fill(context_mask[:, None, :] <= 0, -1e9)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bqn,bnd->bqd", weights, context)


class NeuralFieldDecoder(nn.Module):
    """Phase-conditioned neural field decoder.

    Query convention
    ----------------
    query_xy:  [B, Q, 2]  raw physical coordinates in the domain
    query_tau: [B, Q, 1]  normalized phase in [0, 1] (canonical cycle coordinate)
    Returns:
        pred_mean:  [B, Q, 4]
        pred_field: [B, Q, 4]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim

        self.query_encoder = FourierEncoder(3, cfg.query_fourier_frequencies, include_input=True)

        query_proj_in = self.query_encoder.output_dim + (2 * cfg.latent_dim) + 1
        self.query_proj = MLP(query_proj_in, cfg.hidden_dim, cfg.hidden_dim, num_layers=2, dropout=cfg.dropout)

        self.env_agg = AttentionAggregator(cfg.hidden_dim, rel_dim=0)
        self.mod_agg = AttentionAggregator(cfg.hidden_dim, rel_dim=0)
        self.hyp_agg = AttentionAggregator(cfg.hidden_dim, rel_dim=0)

        mean_in = cfg.hidden_dim * 4 + cfg.latent_dim
        residual_in = cfg.hidden_dim * 4 + cfg.latent_dim + 1

        if cfg.decoder_type == "siren":
            self.mean_decoder = SirenNet(mean_in, cfg.decoder_hidden_dim, 4, num_layers=max(cfg.decoder_num_layers, 2))
            self.residual_decoder = SirenNet(residual_in, cfg.decoder_hidden_dim, 4, num_layers=max(cfg.decoder_num_layers, 2))
        else:
            self.mean_decoder = MLP(mean_in, cfg.decoder_hidden_dim, 4, cfg.decoder_num_layers, dropout=cfg.dropout)
            self.residual_decoder = MLP(residual_in, cfg.decoder_hidden_dim, 4, cfg.decoder_num_layers, dropout=cfg.dropout)

    def _normalize_query_xy(self, query_xy: torch.Tensor) -> torch.Tensor:
        xy = query_xy.clone()
        xy[..., 0] = xy[..., 0] / max(self.cfg.domain_length_x, 1e-6)
        xy[..., 1] = xy[..., 1] / max(self.cfg.domain_length_y, 1e-6)
        return xy

    def forward(
        self,
        organized: Dict[str, torch.Tensor],
        behavior: Dict[str, torch.Tensor],
        query_xy: torch.Tensor,
        query_tau: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_queries, _ = query_xy.shape
        query_xy_norm = self._normalize_query_xy(query_xy)
        query_input = torch.cat([query_xy_norm, query_tau], dim=-1)
        query_feat = self.query_encoder(query_input)

        mean_latent = behavior["mean_latent"][:, None, :].expand(batch_size, num_queries, -1)
        dynamic_latent = behavior["dynamic_latent"][:, None, :].expand(batch_size, num_queries, -1)
        freq_pred = behavior["freq_pred"][:, None, :].expand(batch_size, num_queries, -1)

        query_state = self.query_proj(torch.cat([query_feat, mean_latent, dynamic_latent, freq_pred], dim=-1))

        env_ctx = self.env_agg(query_state, organized["env_state"], rel=None)
        mod_ctx = self.mod_agg(
            query_state,
            organized["module_state"],
            rel=None,
            context_mask=organized["cyl_mask"],
        )
        hyp_ctx = self.hyp_agg(query_state, organized["hyper_state"], rel=None)

        mean_in = torch.cat([query_state, env_ctx, mod_ctx, hyp_ctx, mean_latent], dim=-1)
        pred_mean = self.mean_decoder(mean_in)

        residual_in = torch.cat([query_state, env_ctx, mod_ctx, hyp_ctx, dynamic_latent, freq_pred], dim=-1)
        pred_residual = self.residual_decoder(residual_in)

        pred_field = pred_mean + pred_residual

        return {
            "pred_mean": pred_mean,
            "pred_residual": pred_residual,
            "pred_field": pred_field,
        }


# ----------------------------- Full model wrapper ------------------------------


class HypergraphNeuralFieldModel(nn.Module):
    """End-to-end model wrapper used by training and evaluation scripts."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.organizer = HypergraphOrganizer(cfg)
        self.behavior_head = BehaviorHead(cfg)
        self.decoder = NeuralFieldDecoder(cfg)

    def forward(
        self,
        structure: Dict[str, torch.Tensor],
        query_xy: torch.Tensor,
        query_tau: torch.Tensor,
        return_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            structure: dictionary with keys
                re_values:     [B, 1]
                num_cylinders: [B, 1]
                centers:       [B, N_max, 2]
                cyl_mask:      [B, N_max]
                optional extra_global: [B, G_extra]
                optional extra_module: [B, N_max, F_extra]
            query_xy:  [B, Q, 2]
            query_tau: [B, Q, 1]
        Returns:
            dict containing predictions and optional organizer internals.
        """
        organized = self.organizer(
            re_values=structure["re_values"],
            num_cylinders=structure["num_cylinders"],
            centers=structure["centers"],
            cyl_mask=structure["cyl_mask"],
            extra_global=structure.get("extra_global"),
            extra_module=structure.get("extra_module"),
        )
        behavior = self.behavior_head(organized)
        decoded = self.decoder(organized, behavior, query_xy=query_xy, query_tau=query_tau)

        out = {
            "pred_field": decoded["pred_field"],
            "pred_mean": decoded["pred_mean"],
            "pred_residual": decoded["pred_residual"],
            "freq_pred": behavior["freq_pred"],
        }
        if return_aux:
            out.update(
                {
                    "behavior_latent": behavior["behavior_latent"],
                    "mean_latent": behavior["mean_latent"],
                    "dynamic_latent": behavior["dynamic_latent"],
                    "A_me": organized["A_me"],
                    "A_mh": organized["A_mh"],
                    "A_eh": organized["A_eh"],
                    "module_state": organized["module_state"],
                    "env_state": organized["env_state"],
                    "hyper_state": organized["hyper_state"],
                    "env_coords": organized["env_coords"],
                    "module_coords_norm": organized["module_coords_norm"],
                }
            )
        return out

    def reconstruct_full_grid(
        self,
        structure: Dict[str, torch.Tensor],
        x_grid: torch.Tensor,
        y_grid: torch.Tensor,
        tau: torch.Tensor,
        query_batch_size: int = 16384,
    ) -> Dict[str, torch.Tensor]:
        """Convenience helper for evaluation.

        Args:
            structure: same dictionary as in forward(), batch size must be 1.
            x_grid, y_grid: [H, W]
            tau: scalar tensor [1] or [1,1]
        Returns:
            dict with reconstructed [H, W, 4] fields and organizer internals.
        """
        if x_grid.ndim != 2 or y_grid.ndim != 2:
            raise ValueError("x_grid and y_grid must be rank-2 tensors [H, W].")
        device = x_grid.device
        H, W = x_grid.shape
        xy = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)[None, ...]  # [1, H*W, 2]
        tau = tau.reshape(1, 1).to(device=device, dtype=xy.dtype)
        tau_full = tau[:, None, :].expand(1, xy.shape[1], 1)

        outputs = []
        organizer_cache = None
        for start in range(0, xy.shape[1], query_batch_size):
            end = min(start + query_batch_size, xy.shape[1])
            chunk_out = self.forward(structure, xy[:, start:end], tau_full[:, start:end], return_aux=(organizer_cache is None))
            outputs.append(chunk_out["pred_field"])
            if organizer_cache is None:
                organizer_cache = {k: v for k, v in chunk_out.items() if k != "pred_field"}

        field = torch.cat(outputs, dim=1).reshape(1, H, W, 4)
        result = {"pred_field": field}
        if organizer_cache is not None:
            result.update(organizer_cache)
        return result


def build_model_from_config(model_cfg: Dict) -> HypergraphNeuralFieldModel:
    cfg = ModelConfig.from_dict(model_cfg)
    return HypergraphNeuralFieldModel(cfg)
