"""Baseline-input-only support scoring for response-factor candidates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .factor_operator import _mlp
from .types import DEFAULT_OUTPUT_CHANNELS, BaselineResponseCache, ResponseFactor


class InputOnlySupportScorer(nn.Module):
    """Score candidate donor sets from baseline design/context inputs only.

    There is deliberately no argument for trial deltas, perturbed fields,
    anchor IDs, response labels, or inverse objectives. Those labels can train
    this scorer through :func:`masked_support_classification_loss`, but never
    enter its inference path.
    """

    def __init__(
        self,
        *,
        module_feature_dim: int,
        design_dim: int,
        context_dim: int,
        role_names: Sequence[str] | None = None,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(module_feature_dim, design_dim, hidden_dim) <= 0 or context_dim < 0:
            raise ValueError("Feature widths and hidden_dim must be positive; context_dim may be zero.")
        self.module_feature_dim = int(module_feature_dim)
        self.design_dim = int(design_dim)
        self.context_dim = max(int(context_dim), 1)
        self.accept_empty_context = context_dim == 0
        self.hidden_dim = int(hidden_dim)
        self.role_names = tuple(role_names or DEFAULT_OUTPUT_CHANNELS)
        if not self.role_names or len(set(self.role_names)) != len(self.role_names):
            raise ValueError("role_names must be nonempty and unique.")
        hidden = self.hidden_dim
        self.module_encoder = nn.Sequential(
            nn.Linear(self.module_feature_dim + self.design_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.pair_geometry_encoder = nn.Sequential(
            nn.Linear(self.design_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.order_embedding = nn.Embedding(3, hidden)
        self.role_embedding = nn.Embedding(len(self.role_names), hidden)
        self.candidate_scorer = _mlp(hidden * 5, hidden, 1)

    def score_factors(
        self,
        cache: BaselineResponseCache,
        factors: Sequence[ResponseFactor],
    ) -> torch.Tensor:
        """Return logits ``[B,E]`` using only the fixed baseline cache."""

        if cache.module_features.shape[-1] != self.module_feature_dim:
            raise ValueError("Baseline module feature width does not match the support scorer.")
        if cache.baseline_design.shape[-1] != self.design_dim:
            raise ValueError("Baseline design width does not match the support scorer.")
        expected_context = 0 if self.accept_empty_context else self.context_dim
        if cache.baseline_context.shape[-1] != expected_context:
            raise ValueError("Baseline context width does not match the support scorer.")
        if cache.module_features.device != next(self.parameters()).device:
            raise ValueError("Baseline cache and support scorer must be on the same device.")
        module_inputs = torch.cat([cache.module_features, cache.baseline_design], dim=-1)
        module_tokens = self.module_encoder(module_inputs)
        if self.accept_empty_context:
            context_input = cache.baseline_context.new_zeros(cache.batch_size, 1)
        else:
            context_input = cache.baseline_context
        context_token = self.context_encoder(context_input)
        logits = []
        for factor in factors:
            if len(factor.donor_ids) not in (1, 2):
                raise ValueError("The support scorer accepts unary and pair candidates only.")
            if any(role not in self.role_names for role in factor.output_roles):
                raise KeyError(f"Factor {factor.factor_id!r} declares a role unknown to this support scorer.")
            indices = torch.as_tensor(cache.indices_for(factor.donor_ids), device=module_tokens.device)
            donor = module_tokens.index_select(1, indices)
            mean_token = donor.mean(dim=1)
            max_token = donor.amax(dim=1)
            if len(factor.donor_ids) == 2:
                design = cache.baseline_design.index_select(1, indices)
                pair_token = self.pair_geometry_encoder(torch.abs(design[:, 0] - design[:, 1]))
            else:
                pair_token = torch.zeros_like(mean_token)
            order = torch.full(
                (cache.batch_size,), len(factor.donor_ids), device=module_tokens.device, dtype=torch.long
            )
            role_indices = torch.as_tensor(
                [self.role_names.index(role) for role in factor.output_roles],
                device=module_tokens.device,
                dtype=torch.long,
            )
            role_token = self.role_embedding(role_indices).mean(dim=0).expand(cache.batch_size, -1)
            order_token = self.order_embedding(order)
            candidate = torch.cat([context_token, mean_token, max_token, pair_token + order_token, role_token], dim=-1)
            logits.append(self.candidate_scorer(candidate).squeeze(-1))
        if not logits:
            return cache.module_features.new_zeros(cache.batch_size, 0)
        return torch.stack(logits, dim=1)

    @staticmethod
    def retention_weights(scores: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
        """Map scores to exact-zero, bounded support coefficients."""

        threshold_tensor = torch.as_tensor(threshold, device=scores.device, dtype=scores.dtype)
        excess = torch.relu(scores - threshold_tensor)
        squared = excess.square()
        return squared / (1.0 + squared)


def masked_support_classification_loss(
    scores: torch.Tensor,
    target_retained: torch.Tensor,
    resolved_mask: torch.Tensor,
) -> torch.Tensor:
    """Train on measured support labels while ignoring unknown candidates."""

    if scores.shape != target_retained.shape or scores.shape != resolved_mask.shape:
        raise ValueError("Support scores, labels, and resolved mask must have the same shape.")
    resolved = resolved_mask.to(dtype=torch.bool)
    if not bool(resolved.any()):
        raise ValueError("There are no resolved support labels in this batch.")
    labels = target_retained[resolved].to(dtype=scores.dtype)
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("Resolved support labels must be finite; mask unknown candidates out.")
    if bool(((labels < 0.0) | (labels > 1.0)).any()):
        raise ValueError("Resolved support labels must lie in [0,1].")
    return F.binary_cross_entropy_with_logits(scores[resolved], labels)
