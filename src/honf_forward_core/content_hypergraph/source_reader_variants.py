"""Small optional variations of the maintained v2 source reader."""

from __future__ import annotations

from torch import nn


class SourceScaleProjection(nn.Linear):
    """Bias-free map from the existing asinh physical-scale descriptor to values."""

    def __init__(self, coordinate_dim: int, width: int) -> None:
        super().__init__(int(coordinate_dim), int(width), bias=False)
