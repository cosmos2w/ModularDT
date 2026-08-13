"""Reusable, checkpoint-stable neural feature and MLP building blocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Append power-of-two Fourier features with an explicit convention.

    ``ordering='grouped'`` preserves the HONF core convention (all sine
    frequencies followed by all cosine frequencies). ``ordering='interleaved'``
    preserves the ThermalChannel local-module convention (sine/cosine pairs
    frequency by frequency).
    """

    def __init__(
        self,
        input_dim: int | None,
        num_frequencies: int,
        *,
        include_input: bool = True,
        angular_scale: float = math.pi,
        ordering: str = "grouped",
    ):
        super().__init__()
        if ordering not in {"grouped", "interleaved"}:
            raise ValueError("Fourier feature ordering must be 'grouped' or 'interleaved'.")
        self.input_dim = None if input_dim is None else int(input_dim)
        self.num_frequencies = max(0, int(num_frequencies))
        self.include_input = bool(include_input)
        self.angular_scale = float(angular_scale)
        self.ordering = ordering
        frequencies = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        if self.input_dim is None:
            raise AttributeError("output_dim requires input_dim at construction time.")
        base = self.input_dim if self.include_input else 0
        return base + 2 * self.input_dim * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dim is not None and int(x.shape[-1]) != self.input_dim:
            raise ValueError(f"Expected Fourier input width {self.input_dim}, got {x.shape[-1]}.")
        pieces = [x] if self.include_input else []
        if self.num_frequencies <= 0:
            if pieces:
                return pieces[0]
            return x.new_empty(*x.shape[:-1], 0)
        frequencies = self.frequencies.to(device=x.device, dtype=x.dtype)
        if self.ordering == "interleaved":
            for frequency in frequencies:
                angle = self.angular_scale * frequency * x
                pieces.extend([torch.sin(angle), torch.cos(angle)])
        else:
            view = frequencies.view(*([1] * (x.ndim - 1)), -1, 1)
            angles = x.unsqueeze(-2) * (self.angular_scale * view)
            pieces.append(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-2).flatten(start_dim=-2))
        return torch.cat(pieces, dim=-1)


class MLP(nn.Module):
    """Configurable eager-input MLP with stable ``net.<index>`` state keys."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        include_zero_dropout: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if activation not in {"gelu", "silu"}:
            raise ValueError("activation must be 'gelu' or 'silu'.")
        activation_type = nn.GELU if activation == "gelu" else nn.SiLU
        dims = [int(in_dim)] + [int(hidden_dim)] * max(int(num_layers) - 1, 0) + [int(out_dim)]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            is_last = index == len(dims) - 2
            if not is_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[index + 1]))
                layers.append(activation_type())
                if dropout > 0.0 or include_zero_dropout:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LazyMLP(nn.Module):
    """Configurable lazy-input MLP used for adapter-defined feature widths."""

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        *,
        include_zero_dropout: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        output_dim = int(hidden_dim if out_dim is None else out_dim)
        layers: list[nn.Module] = [nn.LazyLinear(int(hidden_dim))]
        if int(num_layers) == 1:
            if output_dim != int(hidden_dim):
                raise ValueError("A one-layer LazyMLP requires out_dim == hidden_dim.")
        else:
            layers.append(nn.GELU())
            if dropout > 0.0 or include_zero_dropout:
                layers.append(nn.Dropout(float(dropout)))
            for _ in range(max(0, int(num_layers) - 2)):
                layers.extend([nn.Linear(int(hidden_dim), int(hidden_dim)), nn.GELU()])
                if dropout > 0.0 or include_zero_dropout:
                    layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(int(hidden_dim), output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
