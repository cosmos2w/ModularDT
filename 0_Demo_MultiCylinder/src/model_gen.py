from __future__ import annotations

"""
model_gen.py
============

Generative latent rectified-flow components for the multi-cylinder modular-DT demo.

This file is intentionally self-contained: it does not import the older sparse reconstruction baseline code.  

It is designed to live next to the deterministic `model.py` in `0_Demo_MultiCylinder/src` and to be used by `train_gen.py` and
`evaluate_gen.py`.

High-level generative formulation
---------------------------------
The deterministic model is kept as the structured forward-model backbone:

    structure -> organizer -> behavior / dynamic memory -> deterministic field

The generative model is a residual posterior model on top of that backbone:

    condition = deterministic dense predictions + organized-state summary
    noise z0 -> latent rectified flow -> latent residual z1
    AE decoder(z1) -> vivid residual field sample
    final sample = deterministic mean + generated residual

The two-stage workflow is:

Stage 1: train a convolutional autoencoder on canonical-cycle residual fields.
Stage 2: freeze the AE and deterministic model, then train a latent rectified-
        flow velocity network conditioned on the deterministic organized state.

All tensor shapes are documented in comments because this generative extension
is meant to be a reusable template for future modular engineering systems.
"""

from dataclasses import dataclass
import contextlib
import math
from typing import Dict, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small neural-network utilities
# -----------------------------------------------------------------------------


def _gn_groups(channels: int, max_groups: int = 32) -> int:
    """Return a valid GroupNorm group count for `channels`.

    GroupNorm is more stable than BatchNorm for small generative batches.  This
    helper chooses the largest group count up to `max_groups` that divides the
    channel count.
    """
    g = min(int(max_groups), int(channels))
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


class ResBlock2d(nn.Module):
    """Simple residual block used by the convolutional AE.

    Input/output shape:
        x: [B, C, H, W] -> [B, C, H, W]

    The block is deliberately compact: GroupNorm, SiLU, Conv, GroupNorm, SiLU,
    Conv, plus residual connection.  The AE is only meant to provide a stable
    latent space; the generative expressivity is mainly in the flow model.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        g = _gn_groups(channels, groups)
        self.block = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TimestepEmbedding(nn.Module):
    """Sinusoidal flow-time embedding used by the latent velocity network.

    Rectified flow uses a continuous transport time `t in [0, 1]`.  The velocity
    network receives this embedding so it can learn a time-dependent vector field.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = int(dim)
        self.max_period = float(max_period)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.to(dtype=t.dtype if t.is_floating_point() else torch.float32)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


class AdaGNResBlock(nn.Module):
    """Residual block with adaptive GroupNorm FiLM conditioning.

    Input/output shape:
        x:   [B, C, H, W]
        emb: [B, E]

    The embedding combines rectified-flow time with the global organized-state
    condition.  It produces scale/shift parameters for the second GroupNorm.
    This is a lightweight way to condition every latent-resolution feature map
    on the structure and deterministic organized memory.
    """

    def __init__(self, channels: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.channels = int(channels)
        g = _gn_groups(channels, groups)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return x + h


# -----------------------------------------------------------------------------
# Stage 1: convolutional autoencoder for residual fields
# -----------------------------------------------------------------------------


class ConvResidualAE(nn.Module):
    """Convolutional autoencoder for regular-grid multi-field residuals.

    Expected input:
        x: [B, C, H, W]
           C=4 for [u, v, p, omega] unless the dataset is changed.

    Encoder output:
        z: [B, latent_ch, h, w]
           h and w are determined by `n_levels`, with padding as needed.

    The AE is trained on normalized target grids.  `train_gen.py` stores the
    normalization statistics in the checkpoint so stage-2 and evaluation decode
    back to the correct physical scale.
    """

    def __init__(
        self,
        n_fields: int = 4,
        base_ch: int = 48,
        latent_ch: int = 96,
        n_levels: int = 3,
        num_res_blocks: int = 1,
        num_y: int = 128,
        num_x: int = 256,
    ):
        super().__init__()
        self.n_fields = int(n_fields)
        self.base_ch = int(base_ch)
        self.latent_ch = int(latent_ch)
        self.n_levels = int(n_levels)
        self.num_res_blocks = int(num_res_blocks)
        self.num_y = int(num_y)
        self.num_x = int(num_x)

        # Pad H/W to a multiple of 2**n_levels so transposed convolutions can
        # invert the strided convolution sizes cleanly.
        factor = 2 ** self.n_levels
        self.H_pad = int(math.ceil(self.num_y / factor) * factor)
        self.W_pad = int(math.ceil(self.num_x / factor) * factor)

        # Downsample channel schedule.  The final encoder channel count is
        # capped at latent_ch.
        chs = [min(self.base_ch * (2 ** i), self.latent_ch) for i in range(self.n_levels)]
        if not chs:
            raise ValueError("n_levels must be >= 1")

        enc_layers: list[nn.Module] = []
        in_ch = self.n_fields
        for level, out_ch in enumerate(chs):
            enc_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            for _ in range(self.num_res_blocks):
                enc_layers.append(ResBlock2d(out_ch))
            in_ch = out_ch
        if in_ch != self.latent_ch:
            enc_layers.append(nn.Conv2d(in_ch, self.latent_ch, kernel_size=1))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        dec_chs = list(reversed(chs))
        in_ch = self.latent_ch
        for level, out_ch in enumerate(dec_chs):
            for _ in range(self.num_res_blocks):
                dec_layers.append(ResBlock2d(in_ch))
            dec_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            in_ch = out_ch
        dec_layers.append(nn.GroupNorm(_gn_groups(in_ch), in_ch))
        dec_layers.append(nn.SiLU())
        dec_layers.append(nn.Conv2d(in_ch, self.n_fields, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = self.H_pad - x.shape[-2]
        pad_w = self.W_pad - x.shape[-1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Input grid is larger than AE configured size.")
        return F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : self.num_y, : self.num_x]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._pad(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._crop(self.decoder(z))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# -----------------------------------------------------------------------------
# Stage 2: conditional latent rectified-flow velocity network
# -----------------------------------------------------------------------------


class LatentVelocityUNet(nn.Module):
    """UNet-like velocity model in AE latent space.

    Inputs:
        t:           [B]                  rectified-flow time in [0, 1]
        z_t:         [B, latent_ch, h, w] noisy/interpolated latent state
        cond_latent: [B, C_cond, h, w]    downsampled dense condition maps
        global_cond: [B, G]               pooled organizer / behavior summary

    Output:
        velocity:    [B, latent_ch, h, w]

    The dense condition maps carry spatial information such as deterministic
    mean/residual/field, coordinates, tau, and Re.  The global condition vector
    carries organized-state summaries from the deterministic hypergraph model.
    """

    def __init__(
        self,
        latent_ch: int,
        cond_ch: int,
        global_cond_dim: int,
        base_ch: int = 192,
        ch_mult: Sequence[int] = (1, 2),
        num_res_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.cond_ch = int(cond_ch)
        self.global_cond_dim = int(global_cond_dim)
        self.base_ch = int(base_ch)
        self.ch_mult = tuple(int(v) for v in ch_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.num_heads = int(num_heads)

        emb_dim = self.base_ch
        self.time_embed = TimestepEmbedding(emb_dim)
        self.global_embed = nn.Sequential(
            nn.LayerNorm(self.global_cond_dim),
            nn.Linear(self.global_cond_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.in_conv = nn.Conv2d(self.latent_ch + self.cond_ch, self.base_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        channels = [self.base_ch * m for m in self.ch_mult]
        if channels[0] != self.base_ch:
            raise ValueError("The first ch_mult entry should normally be 1.")

        # Down path: process at progressively coarser latent resolutions.
        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        prev_ch = self.base_ch
        for i, ch in enumerate(channels):
            blocks = nn.ModuleList([AdaGNResBlock(ch, emb_dim) for _ in range(self.num_res_blocks)])
            self.down_blocks.append(blocks)
            if i < len(channels) - 1:
                self.down_convs.append(nn.Conv2d(ch, channels[i + 1], kernel_size=4, stride=2, padding=1))
                prev_ch = channels[i + 1]

        # Bottleneck self-attention lets distant latent cells coordinate vortex
        # phase and wake interaction structure.
        mid_ch = channels[-1]
        self.mid_res1 = AdaGNResBlock(mid_ch, emb_dim)
        self.mid_norm = nn.GroupNorm(_gn_groups(mid_ch), mid_ch)
        self.mid_attn = nn.MultiheadAttention(mid_ch, self.num_heads, batch_first=True)
        self.mid_res2 = AdaGNResBlock(mid_ch, emb_dim)

        # Up path.  We use skip connections for spatial detail recovery.
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        in_ch = mid_ch
        for i in range(len(channels) - 1, -1, -1):
            skip_ch = channels[i]
            block_ch = in_ch + skip_ch
            self.up_blocks.append(nn.ModuleList([AdaGNResBlock(block_ch, emb_dim) for _ in range(self.num_res_blocks)]))
            if i > 0:
                self.up_convs.append(nn.ConvTranspose2d(block_ch, channels[i - 1], kernel_size=4, stride=2, padding=1))
                in_ch = channels[i - 1]
            else:
                in_ch = block_ch

        self.out_conv = nn.Sequential(
            nn.GroupNorm(_gn_groups(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, self.latent_ch, kernel_size=3, padding=1),
        )

        # Zero-initialize the final velocity layer so training starts near a
        # stable zero-velocity field rather than noisy large updates.
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)

    def forward(
        self,
        t: torch.Tensor,
        z_t: torch.Tensor,
        cond_latent: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        # Flow-time embedding plus global organized-state embedding.
        emb = self.time_embed(t) + self.global_embed(global_cond)

        h = self.in_conv(torch.cat([z_t, cond_latent], dim=1))
        h = self.dropout(h)

        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, emb)
            skips.append(h)
            if i < len(self.down_convs):
                h = self.down_convs[i](h)

        h = self.mid_res1(h, emb)
        B, C, H, W = h.shape
        tokens = self.mid_norm(h).flatten(2).transpose(1, 2)  # [B, H*W, C]
        attn_out, _ = self.mid_attn(tokens, tokens, tokens, need_weights=False)
        h = h + attn_out.transpose(1, 2).reshape(B, C, H, W)
        h = self.mid_res2(h, emb)

        for i, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                # Odd latent sizes can differ by one cell after down/up.  Nearest
                # resize is parameter-free and stable.
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, emb)
            if i < len(self.up_convs):
                h = self.up_convs[i](h)

        return self.out_conv(h)


class LatentRectifiedFlow(nn.Module):
    """Rectified-flow wrapper around AE + latent velocity network.

    Training objective:
        z1 = AE.encode(target_residual)
        z0 ~ N(0, I)
        t  ~ Uniform(0, 1)
        zt = (1 - t) * z0 + t * z1
        v* = z1 - z0
        loss = ||v_theta(zt, t | condition) - v*||^2

    Sampling:
        start from z0 ~ N(0, I), integrate dz/dt = v_theta(z, t | condition),
        decode final latent into a generated residual grid.
    """

    def __init__(
        self,
        ae: ConvResidualAE,
        velocity_net: LatentVelocityUNet,
        cond_downsample_mode: str = "area",
    ):
        super().__init__()
        self.ae = ae
        self.velocity_net = velocity_net
        self.cond_downsample_mode = str(cond_downsample_mode)

    def _cond_to_latent_resolution(self, cond_grid: torch.Tensor) -> torch.Tensor:
        """Downsample dense condition maps to the AE latent H/W."""
        latent_h = self.ae.H_pad // (2 ** self.ae.n_levels)
        latent_w = self.ae.W_pad // (2 ** self.ae.n_levels)
        cond = F.pad(
            cond_grid,
            (0, self.ae.W_pad - cond_grid.shape[-1], 0, self.ae.H_pad - cond_grid.shape[-2]),
        )
        mode = "area" if self.cond_downsample_mode == "area" else "bilinear"
        if mode == "area":
            return F.interpolate(cond, size=(latent_h, latent_w), mode="area")
        return F.interpolate(cond, size=(latent_h, latent_w), mode=mode, align_corners=False)

    def training_loss(
        self,
        target_grid_norm: torch.Tensor,
        cond_grid: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # The AE is normally frozen in stage 2.  We use no_grad to avoid storing
        # activations through the encoder.
        with torch.no_grad():
            z1 = self.ae.encode(target_grid_norm)

        B = z1.shape[0]
        z0 = torch.randn_like(z1)
        t = torch.rand(B, device=z1.device, dtype=z1.dtype)
        t_view = t.view(B, 1, 1, 1)
        zt = (1.0 - t_view) * z0 + t_view * z1
        target_velocity = z1 - z0

        cond_latent = self._cond_to_latent_resolution(cond_grid)
        pred_velocity = self.velocity_net(t, zt, cond_latent, global_cond)
        loss = F.mse_loss(pred_velocity, target_velocity)

        info = {
            "loss": float(loss.detach().cpu()),
            "target_rms": float(target_velocity.pow(2).mean().sqrt().detach().cpu()),
            "pred_rms": float(pred_velocity.pow(2).mean().sqrt().detach().cpu()),
        }
        return loss, info

    @torch.no_grad()
    def sample(
        self,
        cond_grid: torch.Tensor,
        global_cond: torch.Tensor,
        n_steps: int = 16,
        ode_solver: str = "euler",
        seed: Optional[int] = None,
        initial_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate a normalized residual grid sample.

        Returns:
            generated residual in normalized grid space, shape [B, C, H, W].
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if ode_solver not in {"euler", "heun"}:
            raise ValueError("ode_solver must be 'euler' or 'heun'")

        if seed is not None:
            generator = torch.Generator(device=cond_grid.device)
            generator.manual_seed(int(seed))
        else:
            generator = None

        cond_latent = self._cond_to_latent_resolution(cond_grid)
        B = cond_grid.shape[0]
        latent_h = self.ae.H_pad // (2 ** self.ae.n_levels)
        latent_w = self.ae.W_pad // (2 ** self.ae.n_levels)
        shape = (B, self.ae.latent_ch, latent_h, latent_w)
        if initial_latent is None:
            z = torch.randn(shape, device=cond_grid.device, dtype=cond_grid.dtype, generator=generator)
        else:
            if tuple(initial_latent.shape) != shape:
                raise ValueError(f"initial_latent shape {tuple(initial_latent.shape)} does not match expected {shape}.")
            z = initial_latent.to(device=cond_grid.device, dtype=cond_grid.dtype)

        dt = 1.0 / float(n_steps)
        for k in range(n_steps):
            t = torch.full((B,), k * dt, device=z.device, dtype=z.dtype)
            v = self.velocity_net(t, z, cond_latent, global_cond)
            if ode_solver == "euler":
                z = z + dt * v
            else:
                z_euler = z + dt * v
                t_next = torch.full((B,), (k + 1) * dt, device=z.device, dtype=z.dtype)
                v_next = self.velocity_net(t_next, z_euler, cond_latent, global_cond)
                z = z + 0.5 * dt * (v + v_next)

        return self.ae.decode(z)


class LatentEMA:
    """Exponential moving average for stage-2 velocity network parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        self._backup = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = {}

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        self.apply(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, payload: Dict[str, torch.Tensor]) -> None:
        for k, v in payload.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


# -----------------------------------------------------------------------------
# Conditioning helpers shared by training and evaluation
# -----------------------------------------------------------------------------


@dataclass
class GridStats:
    """Channel-wise normalization statistics for AE/flow target grids."""

    mean: torch.Tensor  # [C]
    std: torch.Tensor   # [C]

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "GridStats":
        return GridStats(
            mean=self.mean.to(device=device, dtype=dtype if dtype is not None else self.mean.dtype),
            std=self.std.to(device=device, dtype=dtype if dtype is not None else self.std.dtype),
        )


def normalize_grid(x: torch.Tensor, stats: GridStats) -> torch.Tensor:
    """Normalize [B,C,H,W] grid using channel-wise stats."""
    mean = stats.mean.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    std = stats.std.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype).clamp_min(1e-6)
    return (x - mean) / std


def denormalize_grid(x: torch.Tensor, stats: GridStats) -> torch.Tensor:
    """Inverse of normalize_grid."""
    mean = stats.mean.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    std = stats.std.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype).clamp_min(1e-6)
    return x * std + mean


def _safe_pool(values: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
    """Return mean/max pooled summaries for token tensors.

    Args:
        values: [B, N, D] or None.
        mask:   optional [B, N] validity mask for padded module nodes.
    """
    if values is None:
        return []
    if values.ndim == 2:
        return [values]
    if mask is None:
        return [values.mean(dim=1), values.max(dim=1).values]
    m = mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    raw_count = m.sum(dim=1)
    denom = raw_count.clamp_min(1.0)
    mean = (values * m).sum(dim=1) / denom
    masked = values.masked_fill(m <= 0, -1e9)
    max_val = masked.max(dim=1).values
    valid_any = raw_count > 0
    max_val = torch.where(valid_any, max_val, torch.zeros_like(max_val))
    return [mean, max_val]


def build_global_condition_vector(det_outputs: Dict[str, torch.Tensor], structure: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Build a pooled global condition vector from deterministic model outputs.

    This vector is intentionally generic: it uses whatever organized-state keys
    are present in the deterministic model's aux output.  That makes the
    generative wrapper robust to small future changes in the deterministic model.

    Typical contents:
        behavior_latent, mean_latent, freq_pred,
        dynamic_global_token,
        pooled module/env/hyper states,
        pooled dynamic hyper tokens,
        hyper strength / geometry summaries,
        Re and number of cylinders.
    """
    pieces: list[torch.Tensor] = []

    for key in ["behavior_latent", "mean_latent", "dynamic_global_token", "freq_pred"]:
        val = det_outputs.get(key)
        if val is None:
            continue
        if val.ndim == 3 and val.shape[1] == 1:
            val = val[:, 0]
        pieces.append(val.reshape(val.shape[0], -1))

    cyl_mask = structure.get("cyl_mask")
    hyper_active_mask = det_outputs.get("hyper_active_mask")
    if hyper_active_mask is not None:
        hyper_active_mask = hyper_active_mask.to(dtype=torch.float32)
    for key in ["module_state", "env_state", "hyper_state", "dynamic_hyper_base", "dynamic_hyper_tokens"]:
        val = det_outputs.get(key)
        if val is None:
            continue
        mask = cyl_mask if key == "module_state" else hyper_active_mask if key in {"hyper_state", "dynamic_hyper_base", "dynamic_hyper_tokens"} else None
        pieces.extend(_safe_pool(val, mask=mask))

    for key in [
        "hyper_active_mask",
        "hyper_edge_score",
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_strength",
        "hyper_source_coords",
        "hyper_wake_coords",
        "hyper_wake_axis",
        "hyper_wake_extent",
    ]:
        val = det_outputs.get(key)
        if val is not None:
            if val.ndim == 2 and key in {"hyper_edge_score", "hyper_module_mass", "hyper_env_mass", "hyper_strength"} and hyper_active_mask is not None:
                pieces.extend(_safe_pool(val.unsqueeze(-1), mask=hyper_active_mask))
            pieces.extend(_safe_pool(val, mask=None))

    # Always append simple scalar structure descriptors so the flow model knows
    # the global regime even if some aux keys are missing.
    for key in ["re_values", "num_cylinders"]:
        if key in structure:
            pieces.append(structure[key].reshape(structure[key].shape[0], -1))

    if not pieces:
        raise RuntimeError("No deterministic-condition features were available.")
    return torch.cat(pieces, dim=-1)


def build_dense_condition_grid(
    det_mean: torch.Tensor,
    det_residual: torch.Tensor,
    det_field: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    tau: torch.Tensor,
    re_values: torch.Tensor,
    stats: GridStats,
    domain_length_x: float,
    domain_length_y: float,
    re_scale: float = 200.0,
    include_field: bool = True,
    include_coords: bool = True,
    include_re_tau: bool = True,
) -> torch.Tensor:
    """Build dense spatial condition maps for the latent velocity network.

    Args:
        det_mean/residual/field: [B, C, H, W] deterministic predictions.
        x_grid/y_grid:           [B, H, W] physical coordinates.
        tau:                     [B, 1] or [B] phase values.
        re_values:               [B, 1] Reynolds numbers.
        stats:                   normalization stats for field-like channels.

    Returns:
        cond_grid: [B, C_cond, H, W]

    Field-like maps are normalized with the same stats used by the AE target.
    Coordinate/tau/Re maps are dimensionless and concatenated directly.
    """
    pieces: list[torch.Tensor] = []
    pieces.append(normalize_grid(det_mean, stats))
    pieces.append(normalize_grid(det_residual, stats))
    if include_field:
        pieces.append(normalize_grid(det_field, stats))

    B, _, H, W = det_mean.shape
    dtype = det_mean.dtype
    device = det_mean.device

    if include_coords:
        x_norm = (x_grid / max(float(domain_length_x), 1e-6)).to(device=device, dtype=dtype).unsqueeze(1)
        y_norm = (y_grid / max(float(domain_length_y), 1e-6)).to(device=device, dtype=dtype).unsqueeze(1)
        pieces.extend([x_norm, y_norm])

    if include_re_tau:
        tau_map = tau.reshape(B, 1, 1, 1).to(device=device, dtype=dtype).expand(B, 1, H, W)
        re_map = (re_values.reshape(B, 1, 1, 1).to(device=device, dtype=dtype) / max(float(re_scale), 1e-6)).expand(B, 1, H, W)
        pieces.extend([tau_map, re_map])

    return torch.cat(pieces, dim=1)
