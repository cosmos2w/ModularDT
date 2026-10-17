"""Explicit, live geometry descriptors for content-hypergraph v2."""

from __future__ import annotations

import torch
from torch import Tensor


def coordinate_gamma(coordinates: Tensor, origin: Tensor, scale: Tensor) -> Tensor:
    """Return ``[u, sin/cos(2π 2^k u), k=0,1,2]`` along the last axis."""
    normalized = (coordinates - origin[..., None, :]) / scale[..., None, :]
    pieces = [normalized]
    for frequency in (1.0, 2.0, 4.0):
        phase = (2.0 * torch.pi * frequency) * normalized
        pieces.extend((torch.sin(phase), torch.cos(phase)))
    return torch.cat(pieces, dim=-1)


def relative_gamma(delta: Tensor, frame_scale: Tensor) -> Tensor:
    """Coordinate gamma for signed displacements in the declared frame."""
    normalized = delta / frame_scale[..., None, None, :]
    pieces = [normalized]
    for frequency in (1.0, 2.0, 4.0):
        phase = (2.0 * torch.pi * frequency) * normalized
        pieces.extend((torch.sin(phase), torch.cos(phase)))
    return torch.cat(pieces, dim=-1)


def pair_score_features(
    destination_coordinates: Tensor,
    donor_coordinates: Tensor,
    frame_scale: Tensor,
    destination_lengths: Tensor,
    donor_lengths: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return small geometry-score inputs and ``ell_pair``.

    The exact per-axis convention is ``sqrt(ell_dest * ell_donor)``. Inputs
    carry physical lengths; the returned features contain signed normalized
    domain offsets, normalized distance, and smooth physical-scale offsets.
    """
    delta = destination_coordinates[:, :, None, :] - donor_coordinates[:, None, :, :]
    normalized = delta / frame_scale[:, None, None, :]
    distance = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
    ell_pair = torch.sqrt(
        destination_lengths[:, :, None, :] * donor_lengths[:, None, :, :]
    )
    local = torch.asinh(delta / ell_pair)
    return torch.cat((normalized, distance, local), dim=-1), ell_pair


def edge_pair_score_features(
    anchors: Tensor,
    donor_coordinates: Tensor,
    frame_scale: Tensor,
    edge_scales: Tensor,
    donor_lengths: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Geometry descriptor, signed relative gamma and ell-pair for edge reads."""
    descriptor, ell_pair = pair_score_features(
        anchors, donor_coordinates, frame_scale, edge_scales, donor_lengths
    )
    delta = anchors[:, :, None, :] - donor_coordinates[:, None, :, :]
    return descriptor, relative_gamma(delta, frame_scale), ell_pair


def query_edge_features(
    query_coordinates: Tensor,
    anchors: Tensor,
    frame_scale: Tensor,
    edge_scales: Tensor,
) -> tuple[Tensor, Tensor]:
    """Receiver-to-content geometry for headwise readers."""
    delta = query_coordinates[:, :, None, :] - anchors[:, None, :, :]
    normalized = delta / frame_scale[:, None, None, :]
    distance = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
    local = torch.asinh(delta / edge_scales[:, None, :, :])
    descriptor = torch.cat((normalized, distance, local), dim=-1)
    return descriptor, relative_gamma(delta, frame_scale)


def query_source_features(
    query_coordinates: Tensor,
    source_coordinates: Tensor,
    frame_scale: Tensor,
    source_lengths: Tensor,
) -> tuple[Tensor, Tensor]:
    """Receiver-to-source descriptor using the source's declared size scale."""
    descriptor = query_source_score_features(
        query_coordinates, source_coordinates, frame_scale, source_lengths
    )
    delta = query_coordinates[:, :, None, :] - source_coordinates[:, None, :, :]
    return descriptor, relative_gamma(delta, frame_scale)


def query_source_score_features(
    query_coordinates: Tensor,
    source_coordinates: Tensor,
    frame_scale: Tensor,
    source_lengths: Tensor,
) -> Tensor:
    """Return narrow A2-D score geometry without Fourier value features."""
    delta = query_coordinates[:, :, None, :] - source_coordinates[:, None, :, :]
    normalized = delta / frame_scale[:, None, None, :]
    distance = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
    local = torch.asinh(delta / source_lengths[:, None, :, :])
    return torch.cat((normalized, distance, local), dim=-1)
