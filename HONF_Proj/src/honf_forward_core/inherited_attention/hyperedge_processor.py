"""Pure per-anchor processors for scene-prepared moment hyperedges."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HyperedgeProcessorOutput:
    """New hyperedge outputs, all indexed by the prepared environment anchors."""

    preactivation: Tensor  # [B,E,128]
    hidden: Tensor  # [B,E,128]
    context_residual: Tensor  # [B,E,128]
    heat_mix_constant: Tensor | None  # [B,H,E,8]
    heat_mix_spatial: Tensor | None  # [B,H,E,8,D]


def _xavier_uniform_(parameter: Tensor, *, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    with torch.no_grad():
        nn.init.xavier_uniform_(parameter, generator=generator)


class MomentHyperedgeProcessor(nn.Module):
    """Two-layer affine/SiLU map shared by Thermal and Wind adapters.

    M and H instantiate exactly the same parameter names and shapes.  Only the
    activation between the two affine layers changes.  The second context
    projection and optional Thermal transport heads are zero initialized so
    attaching the processor recovers C0 without a dead upstream gradient path.
    """

    def __init__(
        self,
        descriptor_dim: int,
        coordinate_dim: int,
        *,
        processor: str,
        num_heads: int = 4,
        moment_width: int = 8,
        context_dim: int = 128,
        hidden_dim: int = 128,
        thermal_transport: bool = False,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        if descriptor_dim not in (708, 814):
            raise ValueError("Moment processor descriptor width must be 708 (2-D) or 814 (3-D).")
        if coordinate_dim not in (2, 3):
            raise ValueError("Moment processor coordinate dimension must be 2 or 3.")
        if (descriptor_dim, coordinate_dim) not in ((708, 2), (814, 3)):
            raise ValueError("Descriptor width and coordinate dimension must describe the same frame.")
        if processor not in ("M", "H"):
            raise ValueError("The pure processor variant must be M or H.")
        if (num_heads, moment_width, context_dim, hidden_dim) != (4, 8, 128, 128):
            raise ValueError("Round5 fixes four heads, eight moment channels and 128-wide states.")

        self.descriptor_dim = int(descriptor_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.processor = processor
        self.num_heads = int(num_heads)
        self.moment_width = int(moment_width)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.thermal_transport = bool(thermal_transport)

        self.input_projection = nn.Linear(self.descriptor_dim, self.hidden_dim, bias=True)
        self.context_projection = nn.Linear(self.hidden_dim, self.context_dim, bias=True)
        _xavier_uniform_(self.input_projection.weight, seed=int(init_seed))
        with torch.no_grad():
            self.input_projection.bias.zero_()
            self.context_projection.weight.zero_()
            self.context_projection.bias.zero_()

        if self.thermal_transport:
            self.heat_constant_projection = nn.Linear(
                self.hidden_dim, self.num_heads * self.moment_width, bias=True
            )
            self.heat_spatial_projection = nn.Linear(
                self.hidden_dim,
                self.num_heads * self.moment_width * self.coordinate_dim,
                bias=True,
            )
            with torch.no_grad():
                self.heat_constant_projection.weight.zero_()
                self.heat_constant_projection.bias.zero_()
                self.heat_spatial_projection.weight.zero_()
                self.heat_spatial_projection.bias.zero_()
        else:
            self.heat_constant_projection = None
            self.heat_spatial_projection = None

    def forward(self, descriptor: Tensor, *, anchor_mask: Tensor | None = None) -> HyperedgeProcessorOutput:
        if descriptor.ndim != 3 or descriptor.shape[-1] != self.descriptor_dim:
            raise ValueError("Moment descriptor must have shape [B,E,descriptor_dim].")
        if anchor_mask is not None and anchor_mask.shape != descriptor.shape[:2]:
            raise ValueError("Anchor mask must have shape [B,E].")
        preactivation = self.input_projection(descriptor)
        hidden = preactivation if self.processor == "M" else F.silu(preactivation)
        residual = self.context_projection(hidden)

        heat_constant = None
        heat_spatial = None
        if self.thermal_transport:
            batch, anchors = descriptor.shape[:2]
            heat_constant = self.heat_constant_projection(hidden).reshape(
                batch, anchors, self.num_heads, self.moment_width
            ).permute(0, 2, 1, 3)
            heat_spatial = self.heat_spatial_projection(hidden).reshape(
                batch, anchors, self.num_heads, self.moment_width, self.coordinate_dim
            ).permute(0, 2, 1, 3, 4)

        if anchor_mask is not None:
            valid = anchor_mask.to(torch.bool)
            residual = residual * valid[..., None].to(residual.dtype)
            if heat_constant is not None:
                heat_constant = heat_constant * valid[:, None, :, None].to(heat_constant.dtype)
                heat_spatial = heat_spatial * valid[:, None, :, None, None].to(heat_spatial.dtype)

        return HyperedgeProcessorOutput(
            preactivation=preactivation,
            hidden=hidden,
            context_residual=residual,
            heat_mix_constant=heat_constant,
            heat_mix_spatial=heat_spatial,
        )


__all__ = ["HyperedgeProcessorOutput", "MomentHyperedgeProcessor"]
