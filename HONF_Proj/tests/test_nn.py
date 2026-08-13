from __future__ import annotations

import math

import torch

from honf_forward_core.nn import FourierFeatures, LazyMLP, MLP


def test_grouped_fourier_convention_matches_historical_core_order() -> None:
    x = torch.tensor([[[0.25, -0.5]]], dtype=torch.float32)
    frequencies = (2.0 ** torch.arange(3)) * torch.pi
    angles = x.unsqueeze(-2) * frequencies.view(1, 1, -1, 1)
    expected = torch.cat(
        [x, torch.cat([torch.sin(angles), torch.cos(angles)], dim=-2).flatten(start_dim=-2)],
        dim=-1,
    )
    actual = FourierFeatures(None, 3, angular_scale=math.pi, ordering="grouped")(x)
    assert torch.equal(actual, expected)


def test_interleaved_fourier_convention_matches_historical_local_order() -> None:
    x = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    pieces = [x]
    for frequency in 2.0 ** torch.arange(3):
        angle = 2.0 * torch.pi * frequency * x
        pieces.extend([torch.sin(angle), torch.cos(angle)])
    expected = torch.cat(pieces, dim=-1)
    encoder = FourierFeatures(
        2,
        3,
        include_input=True,
        angular_scale=2.0 * math.pi,
        ordering="interleaved",
    )
    assert encoder.output_dim == expected.shape[-1]
    assert torch.equal(encoder(x), expected)


def test_mlp_module_indices_preserve_both_checkpoint_layouts() -> None:
    decoder_mlp = MLP(3, 5, 2, dropout=0.0, include_zero_dropout=True)
    local_mlp = MLP(3, 5, 2, dropout=0.0)
    model_lazy = LazyMLP(5, dropout=0.0, include_zero_dropout=True)
    decoder_lazy = LazyMLP(5, 5, 2, 0.0)
    assert set(decoder_mlp.state_dict()) == {"net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias"}
    assert set(local_mlp.state_dict()) == {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"}
    assert {key for key in model_lazy.state_dict()} == {
        "net.0.weight",
        "net.0.bias",
        "net.3.weight",
        "net.3.bias",
    }
    assert {key for key in decoder_lazy.state_dict()} == {
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
    }
