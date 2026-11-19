"""Case-neutral finite contrasts of aligned observations from native scenes.

The case adapter owns correspondence, physical coordinates, reference validity,
fixed observation masses and units. This module never selects reference labels.
"""
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class NativeResponseNeighborhood:
    scene_ids: tuple[str, ...]
    split: str
    chart: Any
    observation_definition: Any = None


@dataclass(frozen=True)
class CommonObservations:
    references: torch.Tensor
    mask: torch.Tensor
    masses: torch.Tensor
    metadata: Any = None


@dataclass(frozen=True)
class LocalContrast:
    operator: torch.Tensor
    rank: int
    singular_values: torch.Tensor
    numerical_floor: float
    condition_limit: float = 30.0


@dataclass(frozen=True)
class WeightedResponseContrast:
    """Arbitrary contrast rows and explicit nonnegative response weights."""
    operator: torch.Tensor
    row_weights: torch.Tensor


@dataclass(frozen=True)
class FieldLosses:
    absolute: torch.Tensor
    response: torch.Tensor
    channel_available: torch.Tensor


def _helmert(scene_count: int) -> torch.Tensor:
    if scene_count < 2:
        raise ValueError('at least two distinct scenes are required')
    h = torch.zeros((scene_count - 1, scene_count), dtype=torch.float64)
    for row in range(scene_count - 1):
        scale = ((row + 1) * (row + 2)) ** -0.5
        h[row, :row + 1] = scale
        h[row, row + 1] = -(row + 1) * scale
    return h


def build_uniform_contrast(scene_count: int) -> LocalContrast:
    h = _helmert(scene_count)
    return LocalContrast((2.0 / (scene_count - 1)) ** 0.5 * h, 0,
                         torch.empty(0, dtype=torch.float64), 0.0)


def _supported_rank(values: torch.Tensor, floor: float, condition_limit: float | None) -> int:
    if not values.numel() or float(values[0]) <= floor:
        return 0
    cutoff = floor
    if condition_limit is not None:
        cutoff = max(cutoff, float(values[0]) / condition_limit)
    rank = int((values > cutoff).sum())
    # Never select only part of an indistinguishable block at a cutoff.
    if 0 < rank < len(values):
        tie_tolerance = max(floor, torch.finfo(torch.float64).eps * max(1.0, float(values[0])) * len(values))
        while rank and abs(float(values[rank - 1] - values[rank])) <= tie_tolerance:
            rank -= 1
    return rank


def build_local_contrast(chart: Any, coordinate_resolution: Any = 0.0, *, absolute_floor: float = 0.0) -> LocalContrast:
    """Build the fixed half-uniform rule from complete input rows, on CPU FP64.

    coordinate_resolution bounds per-coordinate absolute representation error;
    a scalar applies to every entry. The induced spectral bound accounts for
    all scene rows before the orthonormal contrast and normalization.
    """
    x = torch.as_tensor(chart, dtype=torch.float64, device='cpu').detach()
    if x.ndim != 2 or x.shape[1] == 0 or not torch.isfinite(x).all():
        raise ValueError('chart must be a finite two-dimensional scene-by-coordinate array')
    n, p = x.shape
    h = _helmert(n)
    absolute_floor = float(absolute_floor)
    if not np.isfinite(absolute_floor) or absolute_floor < 0:
        raise ValueError('absolute_floor must be finite and nonnegative')
    resolution = torch.as_tensor(coordinate_resolution, dtype=torch.float64, device='cpu').detach()
    if resolution.ndim > 1 or (resolution.ndim == 1 and resolution.numel() != p):
        raise ValueError('coordinate_resolution must be scalar or one value per coordinate')
    if not torch.isfinite(resolution).all() or (resolution < 0).any():
        raise ValueError('coordinate_resolution must be finite and nonnegative')
    resolution = resolution.expand(p)
    g = (h @ (x - x[:1])) / (n - 1) ** 0.5
    u, singular, _ = torch.linalg.svd(g, full_matrices=False)
    scale = float(singular[0]) if singular.numel() else 0.0
    factorization_floor = torch.finfo(torch.float64).eps * max(g.shape) * scale
    representation_floor = (n / (n - 1)) ** 0.5 * float(torch.linalg.vector_norm(resolution))
    floor = max(factorization_floor, representation_floor, absolute_floor)
    rank = _supported_rank(singular, floor, 30.0)
    transform = torch.eye(n - 1, dtype=torch.float64)
    if rank:
        retained = singular[:rank]
        z = (retained[-1] / retained).square()
        blended = (1 + rank * z / z.sum()) / 2
        modes = u[:, :rank]
        transform = transform + (modes * (blended.sqrt() - 1)) @ modes.T
    operator = (2.0 / (n - 1)) ** 0.5 * transform @ h
    return LocalContrast(operator.detach(), rank, singular.detach(), floor)


def build_soft_inverse_contrast(chart: Any, coordinate_resolution: Any = 0.0, *,
                                absolute_floor: float = 0.0) -> LocalContrast:
    """Build the bounded soft-inverse alternative from complete input rows.

    The numerical support, geometry and equal-scene contrast are the same as
    ``build_local_contrast``. On retained modes the inverse-square emphasis is
    softened by the largest singular value: z=1/(1+(sigma/sigma_max)**2).
    The fixed half-uniform blend preserves unit mean retained weight and leaves
    rejected modes uniform. No model predictions or physical labels enter C.
    """
    x = torch.as_tensor(chart, dtype=torch.float64, device='cpu').detach()
    if x.ndim != 2 or x.shape[1] == 0 or not torch.isfinite(x).all():
        raise ValueError('chart must be a finite two-dimensional scene-by-coordinate array')
    n, p = x.shape
    h = _helmert(n)
    absolute_floor = float(absolute_floor)
    if not np.isfinite(absolute_floor) or absolute_floor < 0:
        raise ValueError('absolute_floor must be finite and nonnegative')
    resolution = torch.as_tensor(coordinate_resolution, dtype=torch.float64, device='cpu').detach()
    if resolution.ndim > 1 or (resolution.ndim == 1 and resolution.numel() != p):
        raise ValueError('coordinate_resolution must be scalar or one value per coordinate')
    if not torch.isfinite(resolution).all() or (resolution < 0).any():
        raise ValueError('coordinate_resolution must be finite and nonnegative')
    resolution = resolution.expand(p)
    g = (h @ (x - x[:1])) / (n - 1) ** 0.5
    u, singular, _ = torch.linalg.svd(g, full_matrices=False)
    scale = float(singular[0]) if singular.numel() else 0.0
    factorization_floor = torch.finfo(torch.float64).eps * max(g.shape) * scale
    representation_floor = (n / (n - 1)) ** 0.5 * float(torch.linalg.vector_norm(resolution))
    floor = max(factorization_floor, representation_floor, absolute_floor)
    rank = _supported_rank(singular, floor, 30.0)
    transform = torch.eye(n - 1, dtype=torch.float64)
    if rank:
        retained = singular[:rank]
        z = 1.0 / (1.0 + (retained / retained[0]).square())
        blended = (1.0 + rank * z / z.sum()) / 2.0
        modes = u[:, :rank]
        transform = transform + (modes * (blended.sqrt() - 1.0)) @ modes.T
    operator = (2.0 / (n - 1)) ** 0.5 * transform @ h
    return LocalContrast(operator.detach(), rank, singular.detach(), floor)


def neighborhood_field_losses(predictions: torch.Tensor, observations: CommonObservations,
                              contrast: LocalContrast, scales: Any,
                              channel_weights: Any) -> FieldLosses:
    """Reduce using fixed masses; unavailable channels contribute zero weight.

    Masses are fixed estimates of the declared full measure, never normalized by
    the mass of the random draw. Invalid references are masked before subtraction.
    """
    if predictions.ndim != 4:
        raise ValueError('predictions must have view, scene, receiver, channel axes')
    v, n, q, c = predictions.shape
    device = predictions.device
    refs = torch.as_tensor(observations.references, device=device, dtype=torch.float64)
    mask = torch.as_tensor(observations.mask, device=device, dtype=torch.bool)
    mass = torch.as_tensor(observations.masses, device=device, dtype=torch.float64)
    scale = torch.as_tensor(scales, device=device, dtype=torch.float64)
    weight = torch.as_tensor(channel_weights, device=device, dtype=torch.float64)
    op = contrast.operator.to(device=device, dtype=torch.float64)
    if refs.shape != (n, q, c) or mask.shape != (q, c) or mass.shape != (q, c):
        raise ValueError('incompatible observation shapes')
    if isinstance(contrast, WeightedResponseContrast):
        row_weights = torch.as_tensor(contrast.row_weights, device=device, dtype=torch.float64)
        if op.ndim != 2 or op.shape[1] != n or row_weights.shape != (op.shape[0],):
            raise ValueError('incompatible weighted response rows')
        if not torch.isfinite(op).all() or not torch.isfinite(row_weights).all() or (row_weights < 0).any():
            raise ValueError('response rows and nonnegative weights must be finite')
        if not torch.allclose(op.sum(dim=1), torch.zeros(op.shape[0], device=device, dtype=torch.float64), atol=1e-9, rtol=0):
            raise ValueError('response rows must cancel constants')
    else:
        row_weights = torch.ones(n - 1, device=device, dtype=torch.float64)
        if op.shape != (n - 1, n):
            raise ValueError('incompatible historical contrast')
    if scale.shape != (c,) or weight.shape != (c,):
        raise ValueError('incompatible contrast or channel definitions')
    if not torch.isfinite(scale).all() or (scale <= 0).any() or not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('scales must be positive and channel weights nonnegative and finite')
    if not torch.isfinite(mass).all() or (mass < 0).any() or (mass[~mask] != 0).any():
        raise ValueError('masses must be finite, nonnegative and zero outside support')
    if not torch.isfinite(refs[:, mask]).all():
        raise ValueError('reference values on common support must be finite')
    pred = predictions.to(dtype=torch.float64)
    valid = mask[None, None, :, :]
    pred = torch.where(valid, pred, torch.zeros((), dtype=pred.dtype, device=device))
    ref = torch.where(mask[None], refs, torch.zeros((), dtype=refs.dtype, device=device))
    residual = (pred - ref[None]) / scale
    absolute_by_channel = (residual.square().mean(dim=(0, 1)) * mass).sum(dim=0)
    contrasted = torch.einsum('ds,vsqc->vdqc', op, residual)
    response_by_channel = ((contrasted.square() * row_weights[None, :, None, None]).sum(dim=1).mean(dim=0) * mass).sum(dim=0)
    available = mass.sum(dim=0) > 0
    return FieldLosses((absolute_by_channel * weight).sum(),
                       (response_by_channel * weight).sum(), available)


def seeded_deck_selection(ids: Sequence[str], seed: int, offset: int, count: int = 1) -> tuple[str, ...]:
    """Return consecutive entries from independently seeded, complete decks."""
    ids = tuple(ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('deck IDs must be nonempty and unique')
    if offset < 0 or count < 0:
        raise ValueError('offset and count must be nonnegative')
    result = []
    while count:
        deck_number, position = divmod(offset, len(ids))
        generator = np.random.default_rng(np.random.SeedSequence([seed, deck_number]))
        deck = generator.permutation(len(ids))
        take = min(count, len(ids) - position)
        result.extend(ids[int(i)] for i in deck[position:position + take])
        offset += take
        count -= take
    return tuple(result)


def projected_factor_rank(chart: Any, nuisance_columns: Sequence[int], target_columns: Sequence[int],
                          coordinate_resolution: Any = 0.0, *, absolute_floor: float = 0.0) -> int:
    """Numerical target rank after projecting out all supported nuisance modes."""
    x = torch.as_tensor(chart, dtype=torch.float64, device='cpu').detach()
    base = build_local_contrast(x, coordinate_resolution, absolute_floor=absolute_floor)
    h = _helmert(x.shape[0])
    g = h @ (x - x[:1]) / (x.shape[0] - 1) ** 0.5
    nuisance = g[:, list(nuisance_columns)]
    target = g[:, list(target_columns)]
    if nuisance.shape[1]:
        u, s, _ = torch.linalg.svd(nuisance, full_matrices=False)
        rank = _supported_rank(s, base.numerical_floor, None)
        target = target - u[:, :rank] @ (u[:, :rank].T @ target)
    singular = torch.linalg.svdvals(target)
    return _supported_rank(singular, base.numerical_floor, None)


def direction_uncovered_fraction(chart: Any, direction: Any, coordinate_resolution: Any = 0.0, *, absolute_floor: float = 0.0) -> float:
    """Fraction of an input direction's Euclidean norm outside observed span."""
    x = torch.as_tensor(chart, dtype=torch.float64, device='cpu').detach()
    direction = torch.as_tensor(direction, dtype=torch.float64, device='cpu').detach()
    if direction.shape != (x.shape[1],) or not torch.isfinite(direction).all():
        raise ValueError('direction must be a finite chart-coordinate vector')
    norm = torch.linalg.vector_norm(direction)
    if norm == 0:
        raise ValueError('direction must be nonzero')
    contrast = build_local_contrast(x, coordinate_resolution, absolute_floor=absolute_floor)
    h = _helmert(x.shape[0])
    _, singular, vh = torch.linalg.svd(h @ (x - x[:1]) / (x.shape[0] - 1) ** 0.5, full_matrices=False)
    rank = _supported_rank(singular, contrast.numerical_floor, None)
    uncovered = direction - vh[:rank].T @ (vh[:rank] @ direction)
    return float(torch.linalg.vector_norm(uncovered) / norm)
