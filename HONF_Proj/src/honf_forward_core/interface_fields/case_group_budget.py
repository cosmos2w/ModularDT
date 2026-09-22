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

import torch
from torch import nn

from honf_forward_core.nn import MLP
from honf_forward_core.routing import entmax15


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
    # A group-0 slot is always live, so Kpack is nonzero.  This one scalar
    # shape decision is made once per forward; no receiver/source decisions
    # use host synchronization.
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

    # Full registered capacity tensors.
    z: torch.Tensor
    support: torch.Tensor
    positive_probability: torch.Tensor
    eta: torch.Tensor
    kappa: torch.Tensor
    optional_logits: torch.Tensor
    noise: torch.Tensor | None
    # One compact mapping computed from the same P0 support.  ``packed_valid``
    # distinguishes batch padding from a real registered prototype.
    packed_ids: torch.Tensor
    packed_valid: torch.Tensor
    deterministic: bool = False

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
        return self.support.sum(dim=-1)

    @property
    def expected_optional_count(self) -> torch.Tensor:
        """Expected optional count, returned live for the single loss term."""

        return self.positive_probability[:, 1:].sum(dim=-1)

    @property
    def optional_open_probability(self) -> torch.Tensor:
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
        if int(always_available_group) != 0:
            raise ValueError("the ordinary always-available group must have index 0.")

        self.control_dim = int(control_dim)
        self.group_count = int(group_count)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.stretch_lower = float(stretch_lower)
        self.stretch_upper = float(stretch_upper)
        self.always_available_group = int(always_available_group)
        self.network = MLP(
            4 * self.control_dim + 1,
            self.hidden_dim,
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
        """Predict one scalar optional logit per case/prototype."""

        if gate_context.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("gate_context and prototypes must have shape [B,F] and [K,D].")
        if int(gate_context.shape[-1]) != 3 * self.control_dim + 1:
            raise ValueError("gate_context has an invalid width.")
        if tuple(prototypes.shape) != (self.group_count, self.control_dim):
            raise ValueError("prototypes have an invalid shape.")
        batch = int(gate_context.shape[0])
        codes = prototypes.to(device=gate_context.device, dtype=gate_context.dtype)
        inputs = torch.cat(
            [
                gate_context[:, None, :].expand(-1, self.group_count - 1, -1),
                codes[None, 1:, :].expand(batch, -1, -1),
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
    ) -> CaseGroupBudget:
        """Build one stochastic or deterministic budget from optional logits."""

        if optional_logits.ndim != 2:
            raise ValueError("optional_logits must have shape [B,K-1].")
        expected = self.group_count - 1
        if int(optional_logits.shape[1]) != expected:
            raise ValueError(f"optional_logits must have width {expected}.")
        dtype = optional_logits.dtype
        device = optional_logits.device
        if not optional_logits.is_floating_point():
            raise TypeError("optional_logits must be floating point.")
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

        ones = optional_logits.new_ones((int(optional_logits.shape[0]), 1))
        z = torch.cat([ones, z_optional], dim=-1)
        support = z > 0.0
        p = torch.cat([ones, probability], dim=-1)
        eta, kappa = gate_reference_eta_kappa(z, support)
        packed_ids, packed_valid = _pack_positive_columns(support)
        return CaseGroupBudget(
            z=z,
            support=support,
            positive_probability=p,
            eta=eta,
            kappa=kappa,
            optional_logits=optional_logits,
            noise=used_noise,
            packed_ids=packed_ids,
            packed_valid=packed_valid,
            deterministic=bool(deterministic),
        )


__all__ = [
    "CaseGroupBudget",
    "CaseGroupGate",
    "gate_reference_eta_kappa",
]
