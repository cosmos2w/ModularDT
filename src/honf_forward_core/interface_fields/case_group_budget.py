"""Case-level hard-concrete group availability for the budgeted reader.

This module owns the small, phase-shared part of the budgeted experiment.  A
``CaseGroupBudget`` is created once from the P0 gate logits and carries the
sampled (or deterministic) availability values, the gate-reference
normalizer, and a compact column map.  It deliberately does not contain any
source memberships or collective controls: those are recomputed by the
reader at every physical phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from honf_forward_core.nn import MLP
from honf_forward_core.routing import entmax15


def routing_logit_scale(
    epoch: int | None,
    *,
    initial_scale: float = 0.1,
    full_epoch: int = 25,
) -> float:
    """Return the near-uniform-to-learned routing scale for one epoch."""

    initial = float(initial_scale)
    end = int(full_epoch)
    if not math.isfinite(initial) or not 0.0 < initial <= 1.0:
        raise ValueError("initial routing scale must be finite and in (0, 1].")
    if end <= 0:
        raise ValueError("routing full epoch must be positive.")
    if epoch is None:
        return 1.0
    fraction = 1.0 if end == 1 else min(max((int(epoch) - 1) / float(end - 1), 0.0), 1.0)
    return initial + (1.0 - initial) * fraction


def sparsification_continuation(
    epoch: int | None,
    *,
    compression_start_epoch: int = 25,
    hardening_epoch: int = 150,
) -> float:
    """Return the dense-to-sparse gate continuation coefficient ``c(t)``."""

    start = int(compression_start_epoch)
    hardening = int(hardening_epoch)
    if start < 1 or hardening < start:
        raise ValueError("compression start/hardening epochs must satisfy 1 <= start <= hardening.")
    if epoch is None:
        return 1.0
    if int(epoch) <= start:
        return 0.0
    if hardening == start or int(epoch) >= hardening:
        return 1.0
    return min(max((int(epoch) - start) / float(hardening - start), 0.0), 1.0)


def _safe_log_gate(z: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Log positive gates without evaluating ``log(0)`` on a live graph."""

    # Evaluating ``torch.log(z)`` first and selecting its result with
    # ``where`` still creates an invalid backward value for a closed gate.
    # Replace closed entries before the logarithm, then mask the finite result.
    safe_z = torch.where(support, z, torch.ones_like(z))
    return torch.log(safe_z).masked_fill(~support, 0.0)


def _pack_positive_columns(support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack positive columns in original order without subset enumeration."""

    if support.ndim != 2:
        raise ValueError("support must have shape [B,K].")
    batch, capacity = support.shape
    counts = support.sum(dim=-1)
    # Effective support is guaranteed nonempty by either the legacy ordinary
    # group or the rescue-mode argmax fallback, so Kpack is nonzero.  This one
    # scalar shape decision is made once per forward; no receiver/source
    # decisions use host synchronization.
    packed_width = int(counts.max().detach().cpu())
    original = torch.arange(capacity, device=support.device, dtype=torch.long)
    original = original.unsqueeze(0).expand(batch, -1)
    rank = support.to(dtype=torch.long).cumsum(dim=-1) - 1
    # Inactive columns have repeated ranks.  Route them to the final scratch
    # slot so they can never overwrite an earlier live ID during scatter.
    scratch_rank = torch.where(
        support,
        rank,
        torch.full_like(rank, packed_width),
    )
    packed_ids = torch.zeros(
        batch,
        packed_width + 1,
        device=support.device,
        dtype=torch.long,
    )
    packed_ids.scatter_(1, scratch_rank, original)
    packed_ids = packed_ids[:, :packed_width]
    packed_valid = torch.arange(
        packed_width,
        device=support.device,
        dtype=torch.long,
    ).unsqueeze(0) < counts.unsqueeze(1)
    return packed_ids, packed_valid


def gate_reference_eta_kappa(
    z: torch.Tensor,
    support: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the gate-only entmax reference ``eta`` and ``kappa``.

    ``eta`` is normalized over positive availability columns.  Closed columns
    are represented as exact zeros, and at least one ordinary group is
    required by the caller.
    """

    if z.ndim != 2:
        raise ValueError("z must have shape [B,K].")
    if support is None:
        support = z > 0.0
    if support.shape != z.shape or support.dtype != torch.bool:
        raise ValueError("support must be a boolean tensor aligned with z.")
    if not bool(support.any(dim=-1).all()):
        raise ValueError("every case must have at least one available group.")
    eta = entmax15(_safe_log_gate(z, support), dim=-1, mask=support)
    kappa = eta.square().sum(dim=-1).reciprocal()
    return eta, kappa


@dataclass(frozen=True)
class CaseGroupBudget:
    """Immutable case-level availability plan shared by all physical phases."""

    # Full registered capacity tensors.  ``z``/``support`` are the effective
    # continuation values consumed by the operator.  The raw hard-concrete
    # realization remains available for audit and is never confused with the
    # executed support during dense continuation.
    z: torch.Tensor
    raw_z: torch.Tensor
    support: torch.Tensor
    raw_support: torch.Tensor
    positive_probability: torch.Tensor
    eta: torch.Tensor
    kappa: torch.Tensor
    optional_logits: torch.Tensor
    noise: torch.Tensor | None
    # One compact mapping computed from the same P0 support.  ``packed_valid``
    # distinguishes batch padding from a real registered prototype.
    packed_ids: torch.Tensor
    packed_valid: torch.Tensor
    fallback_used: torch.Tensor
    symmetric: bool = False
    deterministic: bool = False
    continuation: float = 1.0
    route_logit_scale: float = 1.0

    @property
    def batch_size(self) -> int:
        return int(self.z.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.z.shape[1])

    @property
    def packed_width(self) -> int:
        return int(self.packed_ids.shape[1])

    @property
    def live_count(self) -> torch.Tensor:
        """Executed positive effective columns for this forward."""

        return self.support.sum(dim=-1)

    @property
    def raw_live_count(self) -> torch.Tensor:
        """Raw hard-concrete positive columns before continuation/fallback."""

        return self.raw_support.sum(dim=-1)

    @property
    def executed_live_count(self) -> torch.Tensor:
        return self.live_count

    @property
    def expected_group_count(self) -> torch.Tensor:
        """Expected positive group count from all gate probabilities."""

        return self.positive_probability.sum(dim=-1)

    @property
    def expected_excess_group_count(self) -> torch.Tensor:
        """Expected capacity above the one-group baseline."""

        return torch.relu(self.expected_group_count - 1.0)

    @property
    def expected_optional_count(self) -> torch.Tensor:
        """Expected excess count for the single case-level objective.

        The calibrated coefficient is scheduled by the loss assembly as
        ``c(t) * lambda_star``.  Keeping this tensor unscaled lets detached
        metrics report the raw expected capacity during continuation.
        """

        return self.expected_excess_group_count

    @property
    def optional_open_probability(self) -> torch.Tensor:
        if self.symmetric:
            return self.positive_probability
        return self.positive_probability[:, 1:]

    def log_prior(self) -> torch.Tensor:
        """Gate-only log prior and its exact positive support."""

        return _safe_log_gate(self.z, self.support)

    def gather(self, values: torch.Tensor) -> torch.Tensor:
        """Gather a full-capacity tensor into the shared compact columns."""

        if values.ndim < 2 or int(values.shape[0]) != self.batch_size:
            raise ValueError("values must be batch-aligned with CaseGroupBudget.")
        if int(values.shape[1]) != self.capacity:
            raise ValueError("values must use the registered group axis at dim 1.")
        suffix = (1,) * (values.ndim - 2)
        ids = self.packed_ids.view(self.batch_size, self.packed_width, *suffix)
        expanded_ids = ids.expand(self.batch_size, self.packed_width, *values.shape[2:])
        gathered = torch.gather(values, 1, expanded_ids)
        valid = self.packed_valid.view(self.batch_size, self.packed_width, *suffix)
        return gathered * valid.to(dtype=gathered.dtype)

    def packed_gate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return compact positive gate values and valid-column mask."""

        return self.gather(self.z), self.packed_valid


class CaseGroupGate(nn.Module):
    """Shared two-layer case-conditioned hard-concrete gate predictor."""

    def __init__(
        self,
        control_dim: int,
        group_count: int,
        *,
        hidden_dim: int = 32,
        temperature: float = 2.0 / 3.0,
        stretch_lower: float = -0.1,
        stretch_upper: float = 1.1,
        initial_optional_open_probability: float = 0.95,
        always_available_group: int = 0,
        rescue_mode: bool = False,
    ) -> None:
        super().__init__()
        if int(control_dim) <= 0 or int(group_count) <= 0:
            raise ValueError("control_dim and group_count must be positive.")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive.")
        if float(temperature) <= 0.0:
            raise ValueError("hard-concrete temperature must be positive.")
        if not float(stretch_lower) < 0.0 < float(stretch_upper):
            raise ValueError("hard-concrete stretch must straddle zero.")
        probability = float(initial_optional_open_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("initial_optional_open_probability must be in (0,1).")
        if isinstance(always_available_group, bool) or not 0 <= int(always_available_group) < int(group_count):
            raise ValueError("always_available_group must be a valid group index when retained for compatibility.")

        self.control_dim = int(control_dim)
        self.group_count = int(group_count)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.stretch_lower = float(stretch_lower)
        self.stretch_upper = float(stretch_upper)
        self.always_available_group = int(always_available_group)
        self.rescue_mode = bool(rescue_mode)
        self.network = MLP(
            4 * self.control_dim + 1,
            self.hidden_dim,
            # The same scalar head is applied independently to each
            # prototype-conditioned row.  This is permutation equivariant in
            # rescue mode and preserves the historical v1 state shape.
            1,
            num_layers=2,
        )
        final = self.network.net[-1]
        if not isinstance(final, nn.Linear):  # pragma: no cover - MLP contract.
            raise TypeError("gate MLP final layer must be linear.")
        initial_logit = torch.logit(torch.tensor(probability)) + self.temperature * torch.log(
            torch.tensor(-self.stretch_lower / self.stretch_upper)
        )
        nn.init.normal_(final.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(final.bias, float(initial_logit))

    def logits(self, gate_context: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Predict one scalar logit per case/prototype."""

        if gate_context.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("gate_context and prototypes must have shape [B,F] and [K,D].")
        if int(gate_context.shape[-1]) != 3 * self.control_dim + 1:
            raise ValueError("gate_context has an invalid width.")
        if tuple(prototypes.shape) != (self.group_count, self.control_dim):
            raise ValueError("prototypes have an invalid shape.")
        batch = int(gate_context.shape[0])
        codes = prototypes.to(device=gate_context.device, dtype=gate_context.dtype)
        prototype_codes = codes if self.rescue_mode else codes[1:]
        width = int(prototype_codes.shape[0])
        inputs = torch.cat(
            [
                gate_context[:, None, :].expand(-1, width, -1),
                prototype_codes[None, :, :].expand(batch, -1, -1),
            ],
            dim=-1,
        )
        return self.network(inputs).squeeze(-1)

    def build_budget(
        self,
        optional_logits: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        deterministic: bool = False,
        continuation: float = 1.0,
        route_logit_scale: float = 1.0,
    ) -> CaseGroupBudget:
        """Build one stochastic or deterministic effective gate budget."""

        if optional_logits.ndim != 2:
            raise ValueError(
                "optional_logits must have shape [B,K] in rescue mode or [B,K-1] for v1."
            )
        expected = self.group_count if self.rescue_mode else self.group_count - 1
        if int(optional_logits.shape[1]) != expected:
            raise ValueError(f"gate logits must have width {expected}.")
        dtype = optional_logits.dtype
        device = optional_logits.device
        if not optional_logits.is_floating_point():
            raise TypeError("optional_logits must be floating point.")
        continuation = float(continuation)
        route_logit_scale = float(route_logit_scale)
        if not math.isfinite(continuation) or not 0.0 <= continuation <= 1.0:
            raise ValueError("continuation must be finite and in [0,1].")
        if not math.isfinite(route_logit_scale) or route_logit_scale <= 0.0:
            raise ValueError("route_logit_scale must be finite and positive.")
        log_ratio = optional_logits.new_tensor(
            float(torch.log(torch.tensor(-self.stretch_lower / self.stretch_upper)))
        )
        probability = torch.sigmoid(optional_logits - self.temperature * log_ratio)
        if deterministic:
            z_optional = torch.sigmoid(optional_logits)
            z_optional = z_optional * (self.stretch_upper - self.stretch_lower) + self.stretch_lower
            z_optional = z_optional.clamp(0.0, 1.0)
            used_noise = None
        else:
            used_noise = torch.rand_like(optional_logits) if noise is None else noise
            if used_noise.shape != optional_logits.shape:
                raise ValueError("hard-concrete noise must align with optional_logits.")
            if used_noise.device != device:
                raise ValueError("hard-concrete noise must be on the logits device.")
            eps = torch.finfo(dtype).eps
            uniform = used_noise.clamp(min=eps, max=1.0 - eps)
            logistic = torch.log(uniform) - torch.log1p(-uniform)
            stretched = torch.sigmoid((logistic + optional_logits) / self.temperature)
            z_optional = stretched * (self.stretch_upper - self.stretch_lower) + self.stretch_lower
            z_optional = z_optional.clamp(0.0, 1.0)

        if self.rescue_mode:
            raw_z = z_optional
            p = probability
        else:
            ones = optional_logits.new_ones((int(optional_logits.shape[0]), 1))
            raw_z = torch.cat([ones, z_optional], dim=-1)
            p = torch.cat([ones, probability], dim=-1)
        raw_support = raw_z > 0.0
        if not self.rescue_mode:
            # Historical v1 ignores any accidentally supplied rescue schedule
            # arguments so old predictions remain unchanged.
            continuation = 1.0
            route_logit_scale = 1.0
            z = raw_z
            fallback_used = torch.zeros(
                int(optional_logits.shape[0]), device=device, dtype=torch.bool
            )
        elif continuation < 1.0:
            z = (1.0 - continuation) + continuation * raw_z
            fallback_used = torch.zeros(
                int(optional_logits.shape[0]), device=device, dtype=torch.bool
            )
        else:
            z = raw_z
            fallback_used = ~raw_support.any(dim=-1)
            winners = optional_logits.argmax(dim=-1)
            fallback_z = F.one_hot(winners, num_classes=self.group_count).to(dtype=dtype)
            z = torch.where(fallback_used[:, None], fallback_z, z)
        support = z > 0.0
        eta, kappa = gate_reference_eta_kappa(z, support)
        packed_ids, packed_valid = _pack_positive_columns(support)
        return CaseGroupBudget(
            z=z,
            raw_z=raw_z,
            support=support,
            raw_support=raw_support,
            positive_probability=p,
            eta=eta,
            kappa=kappa,
            optional_logits=optional_logits,
            noise=used_noise,
            packed_ids=packed_ids,
            packed_valid=packed_valid,
            fallback_used=fallback_used,
            symmetric=bool(self.rescue_mode),
            deterministic=bool(deterministic),
            continuation=continuation,
            route_logit_scale=route_logit_scale,
        )


__all__ = [
    "CaseGroupBudget",
    "CaseGroupGate",
    "gate_reference_eta_kappa",
    "routing_logit_scale",
    "sparsification_continuation",
]
