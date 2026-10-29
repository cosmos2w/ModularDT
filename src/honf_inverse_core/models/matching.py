"""Ordered or set-valued token matching for inverse topology targets."""

from __future__ import annotations

import torch


def pairwise_squared_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.ndim != 3 or target.ndim != 3 or source.shape[0] != target.shape[0] or source.shape[2] != target.shape[2]:
        raise ValueError("Matching inputs must be batched token tensors with equal feature width.")
    return (source[:, :, None, :] - target[:, None, :, :]).square().mean(dim=-1)


def token_assignment(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    method: str,
    sinkhorn_temperature: float = 0.05,
    sinkhorn_iterations: int = 20,
    null_cost: float = 1.0,
) -> torch.Tensor:
    """Return `[B,N_prediction,N_target]` assignment with implicit null tokens."""

    if (
        prediction.ndim != 3
        or target.ndim != 3
        or prediction.shape[0] != target.shape[0]
        or prediction.shape[2] != target.shape[2]
    ):
        raise ValueError("Matching inputs must be batched token tensors with equal feature width.")
    batch, prediction_count = prediction.shape[:2]
    target_count = target.shape[1]
    if method == "canonical":
        if prediction_count != target_count:
            raise ValueError("Canonical matching requires equal token counts.")
        return torch.eye(
            prediction_count, device=prediction.device, dtype=prediction.dtype
        ).unsqueeze(0).expand(batch, -1, -1)
    if method not in {"hungarian", "sinkhorn"}:
        raise ValueError(f"Unknown matching method: {method!r}")
    cost = pairwise_squared_distance(prediction, target)
    size = max(prediction_count, target_count)
    padded = cost.new_full((batch, size, size), float(null_cost))
    padded[:, :prediction_count, :target_count] = cost
    if prediction_count < size and target_count < size:
        padded[:, prediction_count:, target_count:] = 0.0
    if method == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover - environment-specific option.
            raise RuntimeError("Hungarian matching requires scipy.") from exc
        assignment = cost.new_zeros((batch, size, size))
        for batch_index in range(batch):
            rows, columns = linear_sum_assignment(padded[batch_index].detach().cpu().numpy())
            assignment[
                batch_index,
                torch.as_tensor(rows, device=cost.device),
                torch.as_tensor(columns, device=cost.device),
            ] = 1.0
    else:
        if float(sinkhorn_temperature) <= 0 or int(sinkhorn_iterations) <= 0:
            raise ValueError("Sinkhorn temperature and iterations must be positive.")
        log_assignment = -padded / float(sinkhorn_temperature)
        for _ in range(int(sinkhorn_iterations)):
            log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=-1, keepdim=True)
            log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=-2, keepdim=True)
        assignment = torch.exp(log_assignment)
    return assignment[:, :prediction_count, :target_count]


def match_tokens(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    method: str = "canonical",
    sinkhorn_temperature: float = 0.05,
    sinkhorn_iterations: int = 20,
) -> torch.Tensor:
    """Return target tokens aligned to prediction under the selected policy."""

    assignment = token_assignment(
        prediction,
        target,
        method=method,
        sinkhorn_temperature=sinkhorn_temperature,
        sinkhorn_iterations=sinkhorn_iterations,
    )
    return assignment @ target


def set_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    method: str = "sinkhorn",
    sinkhorn_temperature: float = 0.05,
    sinkhorn_iterations: int = 20,
    null_cost: float = 1.0,
) -> torch.Tensor:
    """Return mean squared set distance after matching, including null assignments."""

    assignment = token_assignment(
        prediction,
        target,
        method=method,
        sinkhorn_temperature=sinkhorn_temperature,
        sinkhorn_iterations=sinkhorn_iterations,
        null_cost=null_cost,
    )
    aligned = assignment @ target
    matched_mass = assignment.sum(dim=-1)
    token_error = (prediction - aligned).square().mean(dim=-1) * matched_mass
    null_error = float(null_cost) * (1.0 - matched_mass).clamp_min(0.0)
    return (token_error + null_error).mean()


__all__ = ["match_tokens", "pairwise_squared_distance", "set_matching_loss", "token_assignment"]
