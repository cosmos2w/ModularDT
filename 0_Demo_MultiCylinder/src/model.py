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


class DeepONetHead(nn.Module):
    """Minimal DeepONet head with separate branch and trunk networks.

    The trunk side is driven by query-only features, while the branch side
    receives the complementary latent/context features assembled by the decoder.
    """

    def __init__(
        self,
        trunk_in_dim: int,
        branch_in_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        dropout: float = 0.0,
        basis_dim: Optional[int] = None,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.basis_dim = int(basis_dim if basis_dim is not None else hidden_dim)
        joint_out_dim = self.out_dim * self.basis_dim
        mlp_layers = max(int(num_layers), 1)
        self.trunk = MLP(
            trunk_in_dim,
            hidden_dim,
            joint_out_dim,
            mlp_layers,
            dropout=dropout,
        )
        self.branch = MLP(
            branch_in_dim,
            hidden_dim,
            joint_out_dim,
            mlp_layers,
            dropout=dropout,
        )
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, trunk_input: torch.Tensor, branch_input: torch.Tensor) -> torch.Tensor:
        trunk = self.trunk(trunk_input).view(*trunk_input.shape[:-1], self.out_dim, self.basis_dim)
        branch = self.branch(branch_input).view(*branch_input.shape[:-1], self.out_dim, self.basis_dim)
        return (trunk * branch).sum(dim=-1) + self.bias


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
    decoder_type: str = "mlp_fourier"  # "mlp_fourier" | "siren" | "deeponet" | "structured_perceiver"
    perceiver_num_layers: int = 1
    perceiver_num_heads: int = 4
    perceiver_head_dim: int = 16
    perceiver_ffn_mult: int = 2
    perceiver_dropout: float = 0.05
    perceiver_num_global_tokens: int = 3
    perceiver_use_relative_bias: bool = True
    perceiver_chunk_query_attention: bool = True
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
        # rel_dim = 5  # dx, dy, dist, downstream(+x), upstream(-x)
        # self.me_score = MLP(2 * cfg.hidden_dim + rel_dim, cfg.hidden_dim, 1, num_layers=3, dropout=cfg.dropout)
        # self.mh_score = MLP(2 * cfg.hidden_dim, cfg.hidden_dim, 1, num_layers=2, dropout=cfg.dropout)
        # self.eh_score = MLP(2 * cfg.hidden_dim, cfg.hidden_dim, 1, num_layers=2, dropout=cfg.dropout)

        rel_dim = 5  # dx, dy, dist, downstream(+x), upstream(-x)
        self.me_score = MLP(2 * cfg.hidden_dim + rel_dim, cfg.hidden_dim, 1, num_layers=3, dropout=cfg.dropout)
        # physical anchoring for hypergraph assignments:
        # - A_mh sees a compact module-neighborhood geometry summary
        # - A_eh sees a compact environment-region geometry summary
        self.mh_score = MLP(2 * cfg.hidden_dim + rel_dim, cfg.hidden_dim, 1, num_layers=2, dropout=cfg.dropout)
        self.eh_score = MLP(2 * cfg.hidden_dim + rel_dim, cfg.hidden_dim, 1, num_layers=2, dropout=cfg.dropout)

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
        env_coords: torch.Tensor,) -> torch.Tensor:
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

    def _pairwise_periodic_relative_features(
        self,
        src_coords: torch.Tensor,
        dst_coords: torch.Tensor,) -> torch.Tensor:
        """Periodic relative geometry features between any two token sets.

        Args:
            src_coords: [B, N_src, 2] normalized to [0, 1]
            dst_coords: [B, N_dst, 2] normalized to [0, 1]

        Returns:
            rel: [B, N_src, N_dst, 5] with channels:
                dx, dy, dist, downstream(+x flow), upstream
        """
        dx = src_coords[:, :, None, 0] - dst_coords[:, None, :, 0]
        dy = src_coords[:, :, None, 1] - dst_coords[:, None, :, 1]
        dx = (dx + 0.5) % 1.0 - 0.5
        dy = (dy + 0.5) % 1.0 - 0.5
        dist = torch.sqrt(dx.square() + dy.square() + 1e-8)

        # Flow is along +x, so wakes are asymmetric in x.
        downstream = torch.clamp(-dx, min=0.0)
        upstream = torch.clamp(dx, min=0.0)
        return torch.stack([dx, dy, dist, downstream, upstream], dim=-1)

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.to(values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=dim).clamp_min(1e-6)
        return (values * mask).sum(dim=dim) / denom

    @staticmethod
    def masked_max(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.to(values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        very_neg = torch.full_like(values, -1e9)
        masked = torch.where(mask > 0, values, very_neg)
        return masked.max(dim=dim).values

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

        # rel_me = self._periodic_relative_features(module_coords_norm, env_coords)  # [B, N, M, 5]
        rel_me = self._pairwise_periodic_relative_features(module_coords_norm, env_coords)

        rel_mm = self._pairwise_periodic_relative_features(module_coords_norm, module_coords_norm)  # [B, N, N, 5]
        # Valid module-module pairs excluding self-pairs.
        eye = torch.eye(n_max, device=device, dtype=module_coords_norm.dtype)[None, :, :]
        pair_mask = cyl_mask[:, :, None] * cyl_mask[:, None, :] * (1.0 - eye)
        # Distance-weighted neighborhood summary per module.
        dist_mm = rel_mm[..., 2].clamp_min(1e-3)
        pair_w = pair_mask / dist_mm
        pair_w = pair_w / pair_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # [B, N, 5]
        module_geom_summary = torch.einsum("bnm,bnmf->bnf", pair_w, rel_mm)

        for _ in range(self.cfg.message_passing_steps):
            n_env = env_state.shape[1]
            n_hyp = hyper_state.shape[1]

            module_expand = module_state[:, :, None, :].expand(-1, -1, n_env, -1)
            env_expand = env_state[:, None, :, :].expand(-1, n_max, -1, -1)
            me_logits = self.me_score(torch.cat([module_expand, env_expand, rel_me], dim=-1)).squeeze(-1)
            valid_pair_mask = cyl_mask[:, :, None].expand_as(me_logits)
            A_me = torch.softmax(me_logits.masked_fill(valid_pair_mask <= 0, -1e9), dim=-1)

            # Convert module->env weights into env<-module weights.
            env_from_mod = A_me.transpose(1, 2)  # [B, M_env, N]
            env_from_mod = env_from_mod / env_from_mod.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            # Environment-token summary of nearby module-neighborhood geometry.
            env_geom_summary = torch.einsum("bmn,bnf->bmf", env_from_mod, module_geom_summary)  # [B, M_env, 5]

            # module_h = module_state[:, :, None, :].expand(-1, -1, n_hyp, -1)
            # hyper_h = hyper_state[:, None, :, :].expand(-1, n_max, -1, -1)
            # mh_logits = self.mh_score(torch.cat([module_h, hyper_h], dim=-1)).squeeze(-1)
            # A_mh = torch.softmax(mh_logits.masked_fill(cyl_mask[:, :, None] <= 0, -1e9), dim=-1)

            # env_h = env_state[:, :, None, :].expand(-1, -1, n_hyp, -1)
            # hyper_e = hyper_state[:, None, :, :].expand(-1, n_env, -1, -1)
            # eh_logits = self.eh_score(torch.cat([env_h, hyper_e], dim=-1)).squeeze(-1)
            # A_eh = torch.softmax(eh_logits, dim=-1)

            module_h = module_state[:, :, None, :].expand(-1, -1, n_hyp, -1)
            hyper_h = hyper_state[:, None, :, :].expand(-1, n_max, -1, -1)
            module_geom_h = module_geom_summary[:, :, None, :].expand(-1, -1, n_hyp, -1)

            mh_logits = self.mh_score(torch.cat([module_h, hyper_h, module_geom_h], dim=-1)).squeeze(-1)
            A_mh = torch.softmax(mh_logits.masked_fill(cyl_mask[:, :, None] <= 0, -1e9), dim=-1)

            env_h = env_state[:, :, None, :].expand(-1, -1, n_hyp, -1)
            hyper_e = hyper_state[:, None, :, :].expand(-1, n_env, -1, -1)
            env_geom_h = env_geom_summary[:, :, None, :].expand(-1, -1, n_hyp, -1)

            eh_logits = self.eh_score(torch.cat([env_h, hyper_e, env_geom_h], dim=-1)).squeeze(-1)
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
        global_dim = 2 + cfg.future_global_feature_dim
        # richer summary: mean + max for module/env/hyper + explicit globals
        pooled_dim = (6 * cfg.hidden_dim) + global_dim
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
        global_features = organized["global_features"]

        module_mean = HypergraphOrganizer.masked_mean(module_state, cyl_mask, dim=1)
        module_max = HypergraphOrganizer.masked_max(module_state, cyl_mask, dim=1)

        env_mean = env_state.mean(dim=1)
        env_max = env_state.max(dim=1).values

        hyper_mean = hyper_state.mean(dim=1)
        hyper_max = hyper_state.max(dim=1).values

        pooled = torch.cat(
            [
                module_mean,
                module_max,
                env_mean,
                env_max,
                hyper_mean,
                hyper_max,
                global_features,
            ],
            dim=-1,
        )
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


class RelativeGeometryBias(nn.Module):
    """Affine query-to-token geometry bias.

    The bias depends on five wrapped relative features:
        dx, dy, dist, downstream, upstream

    Using a direct affine form avoids allocating a larger [..., 5, D] tensor
    while still letting attention logits react to structured relative geometry.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(5))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        dx: torch.Tensor,
        dy: torch.Tensor,
        dist: torch.Tensor,
        downstream: torch.Tensor,
        upstream: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.weight[0] * dx
            + self.weight[1] * dy
            + self.weight[2] * dist
            + self.weight[3] * downstream
            + self.weight[4] * upstream
            + self.bias
        )


class MultiHeadCrossAttention(nn.Module):
    """Memory-safe multi-head cross-attention.

    Shapes
    ------
    query:        [B, Q, D]
    context:      [B, N_ctx, D]
    attn_bias:    optional [B, Q, N_ctx]
    context_mask: optional [B, N_ctx]
    output:       [B, Q, D]

    The implementation keeps logits at [B, H, Q, N_ctx] and optionally slices
    over Q so the new decoder remains compatible with point-chunk batching.
    """

    def __init__(self, model_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        inner_dim = self.num_heads * self.head_dim

        self.query_proj = nn.Linear(model_dim, inner_dim, bias=False)
        self.key_proj = nn.Linear(model_dim, inner_dim, bias=False)
        self.value_proj = nn.Linear(model_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scale = self.head_dim ** -0.5

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logits = torch.einsum("bhqd,bhnd->bhqn", query, key) * self.scale
        if attn_bias is not None:
            logits = logits + attn_bias[:, None, :, :]

        mask = None
        if context_mask is not None:
            mask = context_mask[:, None, None, :].to(dtype=logits.dtype, device=logits.device)
            logits = logits.masked_fill(mask <= 0, -1e9)

        weights = torch.softmax(logits, dim=-1)
        if mask is not None:
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.dropout(weights)
        return torch.einsum("bhqn,bhnd->bhqd", weights, value)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        query_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, num_queries, _ = query.shape
        num_context = context.shape[1]

        q = self.query_proj(query).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(context).view(batch_size, num_context, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(context).view(batch_size, num_context, self.num_heads, self.head_dim).transpose(1, 2)

        if query_chunk_size is None or num_queries <= query_chunk_size:
            attended = self._attend(q, k, v, attn_bias, context_mask)
        else:
            chunks = []
            for start in range(0, num_queries, query_chunk_size):
                end = min(start + query_chunk_size, num_queries)
                bias_chunk = attn_bias[:, start:end, :] if attn_bias is not None else None
                attended_chunk = self._attend(q[:, :, start:end, :], k, v, bias_chunk, context_mask)
                chunks.append(attended_chunk)
            attended = torch.cat(chunks, dim=2)

        attended = attended.transpose(1, 2).reshape(batch_size, num_queries, self.num_heads * self.head_dim)
        return self.out_proj(attended)


class CrossAttentionBlock(nn.Module):
    """Single structured-Perceiver update: cross-attention then feedforward."""

    def __init__(self, model_dim: int, num_heads: int, head_dim: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn = MultiHeadCrossAttention(model_dim, num_heads, head_dim, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = MLP(
            model_dim,
            model_dim * max(int(ffn_mult), 1),
            model_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        query_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.attn_norm(query),
            context,
            attn_bias=attn_bias,
            context_mask=context_mask,
            query_chunk_size=query_chunk_size,
        )
        query = query + self.attn_dropout(attn_out)
        ffn_out = self.ffn(self.ffn_norm(query))
        return query + self.ffn_dropout(ffn_out)


class NeuralFieldDecoder(nn.Module):
    """Phase-conditioned neural field decoder.

    Query convention
    ----------------
    query_xy:  [B, Q, 2]  raw physical coordinates in the domain
    query_tau: [B, Q, 1]  normalized phase in [0, 1] (canonical cycle coordinate)
    Returns:
        pred_mean:  [B, Q, 4]
        pred_field: [B, Q, 4]

    Decoder modes
    -------------
    mlp_fourier:
        Standard MLP decoder on query-conditioned latent features.
    siren:
        SIREN alternative for the same concatenated decoder inputs.
    deeponet:
        Uses query_encoder(query_xy, tau) as the trunk-side query representation
        and feeds aggregated latent/context features to branch nets.
    structured_perceiver:
        Builds typed memory tokens from organized module/env/hyper states plus a
        few global behavior tokens, then updates mean/residual query states via
        shallow cross-attention. Only the updated query states are read out by
        the final heads, which avoids a raw-coordinate bypass into the output.
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

        # self.env_agg = AttentionAggregator(cfg.hidden_dim, rel_dim=5)
        # self.mod_agg = AttentionAggregator(cfg.hidden_dim, rel_dim=5)
        # self.hyp_agg = AttentionAggregator(cfg.hidden_dim, rel_dim=0)

        mean_in = cfg.hidden_dim * 4 + cfg.latent_dim
        residual_in = cfg.hidden_dim * 4 + cfg.latent_dim + 1

        if cfg.decoder_type == "structured_perceiver":
            perceiver_dropout = float(cfg.perceiver_dropout)
            num_global_tokens = max(int(cfg.perceiver_num_global_tokens), 1)
            query_chunk_size = 1024 if bool(cfg.perceiver_chunk_query_attention) else None
            self.perceiver_query_chunk_size = query_chunk_size

            self.module_memory_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.env_memory_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.hyper_memory_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.memory_type_embeddings = nn.Parameter(torch.randn(4, cfg.hidden_dim) * 0.02)
            self.global_slot_embeddings = nn.Parameter(torch.randn(num_global_tokens, cfg.hidden_dim) * 0.02)
            global_token_in = cfg.behavior_dim + (2 * cfg.latent_dim) + 1
            self.global_token_mlp = MLP(
                global_token_in,
                cfg.hidden_dim,
                num_global_tokens * cfg.hidden_dim,
                num_layers=2,
                dropout=perceiver_dropout,
            )
            self.memory_norm = nn.LayerNorm(cfg.hidden_dim)

            mean_query_in = self.query_encoder.output_dim + cfg.behavior_dim + cfg.latent_dim
            residual_query_in = self.query_encoder.output_dim + cfg.behavior_dim + cfg.latent_dim + 1
            self.mean_query_proj = MLP(
                mean_query_in,
                cfg.hidden_dim,
                cfg.hidden_dim,
                num_layers=2,
                dropout=perceiver_dropout,
            )
            self.residual_query_proj = MLP(
                residual_query_in,
                cfg.hidden_dim,
                cfg.hidden_dim,
                num_layers=2,
                dropout=perceiver_dropout,
            )

            self.mean_blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        cfg.hidden_dim,
                        cfg.perceiver_num_heads,
                        cfg.perceiver_head_dim,
                        cfg.perceiver_ffn_mult,
                        perceiver_dropout,
                    )
                    for _ in range(max(int(cfg.perceiver_num_layers), 1))
                ]
            )
            self.residual_blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        cfg.hidden_dim,
                        cfg.perceiver_num_heads,
                        cfg.perceiver_head_dim,
                        cfg.perceiver_ffn_mult,
                        perceiver_dropout,
                    )
                    for _ in range(max(int(cfg.perceiver_num_layers), 1))
                ]
            )

            if cfg.perceiver_use_relative_bias:
                self.module_relative_bias = RelativeGeometryBias()
                self.env_relative_bias = RelativeGeometryBias()
            else:
                self.module_relative_bias = None
                self.env_relative_bias = None

            self.mean_head_norm = nn.LayerNorm(cfg.hidden_dim)
            self.residual_head_norm = nn.LayerNorm(cfg.hidden_dim)
            self.mean_decoder = MLP(cfg.hidden_dim, cfg.decoder_hidden_dim, 4, num_layers=2, dropout=perceiver_dropout)
            self.residual_decoder = MLP(cfg.hidden_dim, cfg.decoder_hidden_dim, 4, num_layers=2, dropout=perceiver_dropout)
        elif cfg.decoder_type == "siren":
            self.mean_decoder = SirenNet(mean_in, cfg.decoder_hidden_dim, 4, num_layers=max(cfg.decoder_num_layers, 2))
            self.residual_decoder = SirenNet(residual_in, cfg.decoder_hidden_dim, 4, num_layers=max(cfg.decoder_num_layers, 2))
        elif cfg.decoder_type == "deeponet":
            self.mean_decoder = DeepONetHead(
                trunk_in_dim=self.query_encoder.output_dim,
                branch_in_dim=cfg.hidden_dim * 3 + cfg.latent_dim,
                hidden_dim=cfg.decoder_hidden_dim,
                num_layers=cfg.decoder_num_layers,
                out_dim=4,
                dropout=cfg.dropout,
            )
            self.residual_decoder = DeepONetHead(
                trunk_in_dim=self.query_encoder.output_dim,
                branch_in_dim=cfg.hidden_dim * 3 + cfg.latent_dim + 1,
                hidden_dim=cfg.decoder_hidden_dim,
                num_layers=cfg.decoder_num_layers,
                out_dim=4,
                dropout=cfg.dropout,
            )
        elif cfg.decoder_type == "mlp_fourier":
            self.mean_decoder = MLP(mean_in, cfg.decoder_hidden_dim, 4, cfg.decoder_num_layers, dropout=cfg.dropout)
            self.residual_decoder = MLP(residual_in, cfg.decoder_hidden_dim, 4, cfg.decoder_num_layers, dropout=cfg.dropout)
        else:
            raise ValueError(
                f"Unsupported decoder_type='{cfg.decoder_type}'. "
                "Expected one of {'mlp_fourier', 'siren', 'deeponet', 'structured_perceiver'}."
            )

    def _normalize_query_xy(self, query_xy: torch.Tensor) -> torch.Tensor:
        xy = query_xy.clone()
        xy[..., 0] = xy[..., 0] / max(self.cfg.domain_length_x, 1e-6)
        xy[..., 1] = xy[..., 1] / max(self.cfg.domain_length_y, 1e-6)
        return xy

    def _pairwise_periodic_relative_features(
        self,
        query_xy_norm: torch.Tensor,
        token_xy_norm: torch.Tensor,) -> torch.Tensor:
        """Relative geometry for query->token attention logits.

        Args:
            query_xy_norm: [B, Q, 2]
            token_xy_norm: [B, N, 2]

        Returns:
            rel: [B, Q, N, 5] with channels:
                dx, dy, dist, downstream, upstream
        """
        dx = query_xy_norm[:, :, None, 0] - token_xy_norm[:, None, :, 0]
        dy = query_xy_norm[:, :, None, 1] - token_xy_norm[:, None, :, 1]
        dx = (dx + 0.5) % 1.0 - 0.5
        dy = (dy + 0.5) % 1.0 - 0.5
        dist = torch.sqrt(dx.square() + dy.square() + 1e-8)

        downstream = torch.clamp(-dx, min=0.0)
        upstream = torch.clamp(dx, min=0.0)
        return torch.stack([dx, dy, dist, downstream, upstream], dim=-1)

    def _build_structured_memory(
        self,
        organized: Dict[str, torch.Tensor],
        behavior: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build typed memory tokens for the structured Perceiver decoder.

        Memory layout:
            [module tokens | env tokens | hyper tokens | global behavior tokens]

        Shapes:
            memory:      [B, N_total, D]
            memory_mask: [B, N_total]
        """
        module_tokens = self.module_memory_proj(organized["module_state"]) + self.memory_type_embeddings[0]
        env_tokens = self.env_memory_proj(organized["env_state"]) + self.memory_type_embeddings[1]
        hyper_tokens = self.hyper_memory_proj(organized["hyper_state"]) + self.memory_type_embeddings[2]

        global_features = torch.cat(
            [
                behavior["behavior_latent"],
                behavior["mean_latent"],
                behavior["dynamic_latent"],
                behavior["freq_pred"],
            ],
            dim=-1,
        )
        batch_size = global_features.shape[0]
        num_global_tokens = self.global_slot_embeddings.shape[0]
        global_tokens = self.global_token_mlp(global_features).view(batch_size, num_global_tokens, self.hidden_dim)
        global_tokens = global_tokens + self.global_slot_embeddings.unsqueeze(0) + self.memory_type_embeddings[3]

        memory = torch.cat([module_tokens, env_tokens, hyper_tokens, global_tokens], dim=1)
        memory = self.memory_norm(memory)

        module_mask = organized["cyl_mask"]
        env_mask = torch.ones(
            organized["env_state"].shape[:2],
            device=memory.device,
            dtype=module_mask.dtype,
        )
        hyper_mask = torch.ones(
            organized["hyper_state"].shape[:2],
            device=memory.device,
            dtype=module_mask.dtype,
        )
        global_mask = torch.ones(
            (batch_size, num_global_tokens),
            device=memory.device,
            dtype=module_mask.dtype,
        )
        memory_mask = torch.cat([module_mask, env_mask, hyper_mask, global_mask], dim=1)
        return memory, memory_mask

    def _build_structured_attention_bias(
        self,
        organized: Dict[str, torch.Tensor],
        query_xy_norm: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.module_relative_bias is None or self.env_relative_bias is None:
            return None

        def relative_bias_fn(token_xy_norm: torch.Tensor, bias_module: RelativeGeometryBias) -> torch.Tensor:
            dx = query_xy_norm[:, :, None, 0] - token_xy_norm[:, None, :, 0]
            dy = query_xy_norm[:, :, None, 1] - token_xy_norm[:, None, :, 1]
            dx = (dx + 0.5) % 1.0 - 0.5
            dy = (dy + 0.5) % 1.0 - 0.5
            bias = bias_module.bias.view(1, 1, 1)
            bias = bias + bias_module.weight[0] * dx + bias_module.weight[1] * dy
            dist = torch.sqrt(dx.square() + dy.square() + 1e-8)
            bias = bias + bias_module.weight[2] * dist
            bias = bias + bias_module.weight[3] * torch.clamp(-dx, min=0.0)
            bias = bias + bias_module.weight[4] * torch.clamp(dx, min=0.0)
            return bias

        module_bias = relative_bias_fn(organized["module_coords_norm"], self.module_relative_bias)
        env_bias = relative_bias_fn(organized["env_coords"], self.env_relative_bias)

        batch_size, num_queries, _ = query_xy_norm.shape
        num_hyper = organized["hyper_state"].shape[1]
        num_global = self.global_slot_embeddings.shape[0]
        zeros_hyper = torch.zeros(batch_size, num_queries, num_hyper, device=query_xy_norm.device, dtype=query_xy_norm.dtype)
        zeros_global = torch.zeros(batch_size, num_queries, num_global, device=query_xy_norm.device, dtype=query_xy_norm.dtype)
        return torch.cat([module_bias, env_bias, zeros_hyper, zeros_global], dim=-1)

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

        if self.cfg.decoder_type == "structured_perceiver":
            memory, memory_mask = self._build_structured_memory(organized, behavior)
            attn_bias = self._build_structured_attention_bias(organized, query_xy_norm)
            behavior_latent = behavior["behavior_latent"][:, None, :].expand(batch_size, num_queries, -1)

            # Each branch gets its own query-token initialization so the mean and
            # residual paths can attend to the same memory for different goals.
            mean_query = self.mean_query_proj(torch.cat([query_feat, behavior_latent, mean_latent], dim=-1))
            residual_query = self.residual_query_proj(
                torch.cat([query_feat, behavior_latent, dynamic_latent, freq_pred], dim=-1)
            )

            for block in self.mean_blocks:
                mean_query = block(
                    mean_query,
                    memory,
                    attn_bias=attn_bias,
                    context_mask=memory_mask,
                    query_chunk_size=self.perceiver_query_chunk_size,
                )
            for block in self.residual_blocks:
                residual_query = block(
                    residual_query,
                    memory,
                    attn_bias=attn_bias,
                    context_mask=memory_mask,
                    query_chunk_size=self.perceiver_query_chunk_size,
                )

            # Final heads only read the updated query states. They do not see raw
            # query Fourier features or raw relative geometry directly.
            pred_mean = self.mean_decoder(self.mean_head_norm(mean_query))
            pred_residual = self.residual_decoder(self.residual_head_norm(residual_query))
            pred_field = pred_mean + pred_residual
            return {
                "pred_mean": pred_mean,
                "pred_residual": pred_residual,
                "pred_field": pred_field,
            }

        query_state = self.query_proj(torch.cat([query_feat, mean_latent, dynamic_latent, freq_pred], dim=-1))

        env_ctx = self.env_agg(query_state, organized["env_state"], rel=None)
        mod_ctx = self.mod_agg(
            query_state,
            organized["module_state"],
            rel=None,
            context_mask=organized["cyl_mask"],
        )
        hyp_ctx = self.hyp_agg(query_state, organized["hyper_state"], rel=None)

        # env_rel = self._pairwise_periodic_relative_features(query_xy_norm, organized["env_coords"])
        # mod_rel = self._pairwise_periodic_relative_features(query_xy_norm, organized["module_coords_norm"])
        # env_ctx = self.env_agg(
        #     query_state,
        #     organized["env_state"],
        #     rel=env_rel,
        # )
        # mod_ctx = self.mod_agg(
        #     query_state,
        #     organized["module_state"],
        #     rel=mod_rel,
        #     context_mask=organized["cyl_mask"],
        # )
        # hyp_ctx = self.hyp_agg(query_state, organized["hyper_state"], rel=None)

        if self.cfg.decoder_type == "deeponet":
            mean_branch_in = torch.cat([env_ctx, mod_ctx, hyp_ctx, mean_latent], dim=-1)
            pred_mean = self.mean_decoder(query_feat, mean_branch_in)

            residual_branch_in = torch.cat([env_ctx, mod_ctx, hyp_ctx, dynamic_latent, freq_pred], dim=-1)
            pred_residual = self.residual_decoder(query_feat, residual_branch_in)
        else:
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
