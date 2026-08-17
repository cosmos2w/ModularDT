"""Optional token matching for `G`/`G_hat` and layouts `D`.

Canonical matching is the default because dataset plans ``G`` and realized
plans ``G_hat`` share fixed order. Hungarian or differentiable Sinkhorn may be
enabled diagnostically if that stability assumption is disproven; neither
changes request ``R`` or context ``c`` and neither is an optimization loop.
"""

from __future__ import annotations

import torch


def pairwise_squared_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.ndim != 3 or target.ndim != 3 or source.shape[0] != target.shape[0] or source.shape[2] != target.shape[2]:
        raise ValueError("Matching inputs must be batched token tensors with equal feature width.")
    return (source[:, :, None, :] - target[:, None, :, :]).square().mean(dim=-1)


def match_tokens(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    method: str = "canonical",
    sinkhorn_temperature: float = 0.05,
    sinkhorn_iterations: int = 20,
) -> torch.Tensor:
    """Return target tokens aligned to prediction under the selected policy."""

    if prediction.shape != target.shape:
        raise ValueError("Matching currently requires equal fixed token shapes.")
    if method == "canonical":
        return target
    cost = pairwise_squared_distance(prediction, target)
    if method == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover - environment-specific option.
            raise RuntimeError("Hungarian matching requires scipy.") from exc
        aligned = torch.empty_like(target)
        for batch in range(target.shape[0]):
            rows, columns = linear_sum_assignment(cost[batch].detach().cpu().numpy())
            permutation = torch.as_tensor(columns[rows.argsort()], device=target.device, dtype=torch.long)
            aligned[batch] = target[batch, permutation]
        return aligned
    if method == "sinkhorn":
        log_assignment = -cost / float(sinkhorn_temperature)
        for _ in range(int(sinkhorn_iterations)):
            log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=-1, keepdim=True)
            log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=-2, keepdim=True)
        assignment = torch.exp(log_assignment)
        return assignment @ target
    raise ValueError(f"Unknown matching method: {method!r}")


__all__ = ["match_tokens", "pairwise_squared_distance"]
