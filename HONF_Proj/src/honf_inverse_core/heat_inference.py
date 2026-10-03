"""Frozen-surrogate fixed-total heat inference with complete numerical trails.

The caller owns the differentiable heat tensor and predictor. Observations,
held sensors and public total are separate inputs; hidden individual heat
allocations are never required. Graph and size-matched random block updates
share their initialization and step budget. This is surrogate optimization,
not independent physical validation or a conditional generative model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

import torch

from .models.centered_simplex_velocity import _project_simplex

HeatPrediction = Callable[[torch.Tensor], Mapping[str, object]]


def fixed_total_basis(active: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal Helmert columns in physical module order; M=1 has none."""
    ids = torch.nonzero(active.bool(), as_tuple=False).flatten()
    basis = torch.zeros((active.numel(), max(0, ids.numel() - 1)), device=active.device, dtype=dtype)
    for column in range(basis.shape[1]):
        denominator = float((column + 1) * (column + 2)) ** .5
        basis[ids[:column+1], column] = 1.0 / denominator
        basis[ids[column+1], column] = -(column + 1) / denominator
    return basis


def observation_identifiability(predictor: HeatPrediction, heat: torch.Tensor,
                              active: torch.Tensor) -> dict[str, torch.Tensor | int | float]:
    """Local observation Jacobian on fixed-total directions, before recovery."""
    basis = fixed_total_basis(active, dtype=heat.dtype)
    if basis.shape[1] == 0:
        return {"jacobian": heat.new_zeros((0, 0)), "singular_values": heat.new_zeros(0),
                "rank": 0, "free_dimensions": 0, "condition_number": 1.0, "rank_tolerance": 0.0}
    jacobian = torch.autograd.functional.jacobian(
        lambda values: predictor(values)["observed"].reshape(-1), heat, vectorize=False)
    restricted = jacobian @ basis
    singular = torch.linalg.svdvals(restricted)
    tolerance = (float(singular.max()) * max(restricted.shape) * torch.finfo(heat.dtype).eps) if singular.numel() else 0.0
    rank = int((singular > tolerance).sum())
    smallest = float(singular.min()) if singular.numel() else 0.0
    condition = float(singular.max()) / smallest if smallest > tolerance and rank == basis.shape[1] else float("inf")
    return {"jacobian": restricted.detach(), "singular_values": singular.detach(), "rank": rank,
            "free_dimensions": basis.shape[1], "condition_number": condition, "rank_tolerance": tolerance}


def _rmse(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.shape != reference.shape or not values.numel():
        raise ValueError("Sensor prediction/reference must be nonempty and shape aligned")
    return (values - reference).square().mean().sqrt()


def _valid_groups(groups: object, active: torch.Tensor) -> tuple[torch.Tensor, ...]:
    valid = active.bool()
    output = []
    for group in (groups if isinstance(groups, Sequence) else ()):
        ids = torch.as_tensor(group, device=active.device, dtype=torch.long).flatten().unique(sorted=True)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= active.numel()):
            raise ValueError("Graph update group names an invalid physical module row")
        ids = ids[valid[ids]]
        if ids.numel() >= 2:
            output.append(ids)
    return tuple(output)


@dataclass(frozen=True)
class HeatInferenceTrail:
    mode: str
    heat: tuple[torch.Tensor, ...]
    iterations: tuple[int, ...]
    observed_rmse: tuple[float, ...]
    held_rmse: tuple[float, ...]
    observed_predictions: tuple[torch.Tensor, ...]
    held_predictions: tuple[torch.Tensor, ...]
    peaks: tuple[torch.Tensor, ...]
    pressure: tuple[torch.Tensor, ...]
    selected_modules: tuple[torch.Tensor, ...]
    elapsed_seconds: tuple[float, ...]
    optimizer_steps: int
    meaningful_graph_steps: int
    group_fallback_steps: int


def fixed_total_heat_inference(predictor: HeatPrediction, observed: torch.Tensor,
                              held: torch.Tensor, active: torch.Tensor, total: torch.Tensor,
                              initial_fraction: torch.Tensor, *, mode: str = "joint",
                              steps: int = 30, learning_rate: float = .05,
                              block_stream: tuple[float, ...] | None = None,
                              permutation_stream: tuple[torch.Tensor, ...] | None = None,
                              block_size_stream: tuple[int, ...] | None = None) -> HeatInferenceTrail:
    """Projected Adam with fixed-total block proposals and same-sized controls.

    ``graph`` selects an actual input-only source group. ``ungrouped`` chooses
    the number of active modules recorded on the graph trajectory when
    ``block_size_stream`` is supplied, from a caller-specified permutation
    stream. Otherwise sizes follow its own current groups. The group is fixed during each proposal; the next prediction
    rebuilds connectivity. If no useful group exists, use a full joint step
    and report the fallback separately.
    """
    if mode not in {"joint", "graph", "ungrouped"} or steps < 0 or learning_rate <= 0:
        raise ValueError("Valid update mode, nonnegative steps and positive learning rate are required")
    if active.ndim != 1 or initial_fraction.shape != active.shape or total.numel() != 1:
        raise ValueError("Active modules/fractions must be [M] and supplied total scalar")
    if not bool(active.bool().any()) or not torch.isfinite(total).all() or float(total) < 0:
        raise ValueError("At least one active module and finite nonnegative total are required")
    if not torch.isfinite(initial_fraction).all():
        raise ValueError("Initial fraction must be finite")
    fractions = _project_simplex(initial_fraction[None], active[None])[0].detach().requires_grad_()
    optimizer = torch.optim.Adam((fractions,), lr=learning_rate)
    ids = torch.nonzero(active.bool(), as_tuple=False).flatten()
    deterministic = ids.numel() == 1 or float(total) == 0
    if block_size_stream is not None:
        if mode != "ungrouped":
            raise ValueError("A recorded block-size stream applies only to ungrouped controls")
        if not deterministic and (len(block_size_stream) < steps or any(
                not isinstance(size, int) or not 2 <= size <= ids.numel() for size in block_size_stream[:steps])):
            raise ValueError("Recorded block sizes must cover every step and lie between two and active M")
    block_stream = block_stream or tuple(0.0 for _ in range(steps))
    permutation_stream = permutation_stream or tuple(ids for _ in range(steps))
    if len(block_stream) < steps or len(permutation_stream) < steps:
        raise ValueError("Caller update streams must cover every optimization step")
    values, obs_error, held_error, obs_predictions, held_predictions = [], [], [], [], []
    peaks, pressure, selected, elapsed = [], [], [], []
    meaningful, fallbacks, actual_steps = 0, 0, 0
    start = perf_counter()
    for iteration in range((0 if deterministic else steps) + 1):
        heat = fractions * total
        prediction = predictor(heat)
        obs, unseen = prediction["observed"], prediction["held"]
        if not torch.is_tensor(obs) or not torch.is_tensor(unseen) or not torch.isfinite(obs).all() or not torch.isfinite(unseen).all():
            raise ValueError("Surrogate sensor predictions must be finite tensors")
        values.append(heat.detach().clone())
        obs_error.append(float(_rmse(obs, observed).detach()))
        held_error.append(float(_rmse(unseen, held).detach()))
        obs_predictions.append(obs.detach().clone())
        held_predictions.append(unseen.detach().clone())
        peaks.append(torch.as_tensor(prediction.get("peaks", []), device=heat.device).detach().clone())
        pressure.append(torch.as_tensor(prediction.get("pressure", []), device=heat.device).detach().clone())
        if heat.is_cuda:
            torch.cuda.synchronize(heat.device)
        elapsed.append(perf_counter() - start)
        if iteration == steps or deterministic:
            break
        optimizer.zero_grad(set_to_none=True)
        (obs - observed).square().mean().backward()
        if fractions.grad is None or not torch.isfinite(fractions.grad).all():
            raise RuntimeError("Caller-owned heat has no finite observation gradient")
        groups = _valid_groups(prediction.get("groups", ()), active)
        chosen = ids
        if mode != "joint":
            nontrivial = tuple(group for group in groups if group.numel() < ids.numel())
            if mode == "ungrouped" and block_size_stream is not None:
                size = block_size_stream[iteration]
                meaningful += int(size < ids.numel())
                fallbacks += int(size == ids.numel())
                permutation = torch.as_tensor(permutation_stream[iteration], device=ids.device)
                if not torch.equal(permutation.sort().values, ids):
                    raise ValueError("Ungrouped stream must permute every active physical module exactly once")
                chosen = permutation[:size]
            elif nontrivial:
                index = min(len(nontrivial) - 1, int(block_stream[iteration] * len(nontrivial)))
                group = nontrivial[index]
                meaningful += 1
                if mode == "graph":
                    chosen = group
                else:
                    permutation = torch.as_tensor(permutation_stream[iteration], device=ids.device)
                    if not torch.equal(permutation.sort().values, ids):
                        raise ValueError("Ungrouped stream must permute every active physical module exactly once")
                    chosen = permutation[:group.numel()]
            else:
                fallbacks += 1
        # The proposal is a physical set. Canonical slot order also makes a
        # full-size random control use the same FP32 reductions/projection as
        # the joint fallback, rather than injecting permutation roundoff.
        chosen = chosen.sort().values
        selected.append(chosen.detach().clone())
        previous = fractions.detach().clone()
        mask = torch.zeros_like(active, dtype=torch.bool)
        mask[chosen] = True
        fractions.grad = torch.where(mask, fractions.grad, torch.zeros_like(fractions.grad))
        optimizer.step()
        with torch.no_grad():
            # Adam momentum cannot update outside this proposal group.
            fractions[~mask] = previous[~mask]
            subtotal = previous[chosen].sum()
            if float(subtotal) > 0:
                projection = _project_simplex((fractions[chosen] / subtotal)[None],
                                             torch.ones((1, chosen.numel()), device=heat.device, dtype=torch.bool))[0]
                fractions[chosen] = projection * subtotal
            else:
                fractions[chosen] = 0
            fractions[~active.bool()] = 0
        actual_steps += 1
    return HeatInferenceTrail(mode, tuple(values), tuple(range(len(values))), tuple(obs_error), tuple(held_error),
        tuple(obs_predictions), tuple(held_predictions), tuple(peaks), tuple(pressure), tuple(selected), tuple(elapsed),
        actual_steps, meaningful, fallbacks)


__all__ = ["HeatInferenceTrail", "fixed_total_basis", "fixed_total_heat_inference", "observation_identifiability"]
