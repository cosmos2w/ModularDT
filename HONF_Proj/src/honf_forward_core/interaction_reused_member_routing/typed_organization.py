"""Typed organization-conditioned competition over selected original members.

The H read keeps one probability distribution per role. Each selected member
first competes within every organization row that reaches it; those row reads
are then mixed by the row's retained role mass. The implementation coalesces
paths on the original member axis before value reduction and avoids a dense
batch-by-query-by-address-by-member intermediate.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def organization_conditioned_role_contribution(
    retrieval_R: Tensor,
    proposal_weights: Tensor,
    ell: Tensor,
    candidate: Tensor,
) -> Tensor:
    """Return H's coalesced, normalized member distribution for one role.

    ``retrieval_R`` has shape ``[B,A,N]``, ``proposal_weights`` has shape
    ``[B,Q,A]``, and ``ell``/``candidate`` have shape ``[B,Q,N]``. For each
    nonempty legal role, the result is

    ``sum_a (w_a d_a / L) (R_ai exp(ell_i) / Z_a)``

    coalesced on the original member index, where ``d_a`` and ``Z_a`` are
    computed over selected members in this role. The returned role mass is
    one, independently for M and E. Empty or unreachable role reads return
    exact zeros. This function leaves the caller's typed ``rho`` law intact.

    The calculation uses three ``bmm`` contractions per role and never forms
    a ``[B,Q,A,N]`` tensor.
    """
    if retrieval_R.ndim != 3 or proposal_weights.ndim != 3:
        raise ValueError("Retrieval and proposal weights must have [B,A,N] and [B,Q,A] axes.")
    if ell.ndim != 3 or candidate.ndim != 3 or candidate.dtype != torch.bool:
        raise ValueError("Scores and boolean candidates must have [B,Q,N] axes.")
    batch, addresses, members = retrieval_R.shape
    if proposal_weights.shape[0] != batch or proposal_weights.shape[2] != addresses:
        raise ValueError("Proposal batch/address axes must match retrieval rows.")
    if ell.shape != candidate.shape or ell.shape[0] != batch or ell.shape[2] != members:
        raise ValueError("Scores and candidates must match retrieval batch/member axes.")
    if proposal_weights.shape[1] != ell.shape[1]:
        raise ValueError("Proposal and score query axes must agree.")
    if not (retrieval_R.is_floating_point() and proposal_weights.is_floating_point() and ell.is_floating_point()):
        raise TypeError("Retrieval, proposal weights and scores must be floating point.")
    if retrieval_R.device != proposal_weights.device or retrieval_R.device != ell.device or candidate.device != ell.device:
        raise ValueError("Retrieval, proposal, scores and candidates must share a device.")

    if addresses == 0 or members == 0:
        return torch.zeros_like(ell)

    # Keep the exponential in a stable range. Mask before subtracting so
    # padded or excluded score entries cannot affect the per-query offset.
    has_candidate = candidate.any(-1, keepdim=True)
    selected_ell = torch.where(candidate, ell, torch.zeros_like(ell))
    offset = selected_ell.masked_fill(~candidate, -torch.inf).amax(-1, keepdim=True)
    offset = torch.where(has_candidate, offset, torch.zeros_like(offset))
    scaled_likelihood = torch.exp(selected_ell - offset) * candidate.to(ell.dtype)

    retrieval_t = retrieval_R.transpose(1, 2)
    candidate_mass = candidate.to(retrieval_R.dtype)
    # d_a = sum_i C_i R_ai and Z_a = sum_i C_i R_ai exp(ell_i).
    retained_view_mass = torch.bmm(candidate_mass.to(retrieval_R.dtype), retrieval_t)
    view_likelihood_mass = torch.bmm(scaled_likelihood.to(retrieval_R.dtype), retrieval_t)

    weighted_view_mass = proposal_weights.to(retrieval_R.dtype) * retained_view_mass
    total_role_mass = weighted_view_mass.sum(-1, keepdim=True)
    tiny = torch.finfo(retrieval_R.dtype).tiny
    view_mix = torch.where(
        total_role_mass > 0,
        weighted_view_mass / total_role_mass.clamp_min(tiny),
        torch.zeros_like(weighted_view_mass),
    )
    inverse_likelihood_mass = torch.where(
        view_likelihood_mass > 0,
        view_likelihood_mass.clamp_min(tiny).reciprocal(),
        torch.zeros_like(view_likelihood_mass),
    )

    # Coalesce row paths before returning to the original member axis.
    row_coefficients = view_mix * inverse_likelihood_mass
    coalesced_rows = torch.bmm(row_coefficients, retrieval_R)
    unnormalized = coalesced_rows * scaled_likelihood.to(retrieval_R.dtype)
    unnormalized = unnormalized * candidate_mass.to(retrieval_R.dtype)
    member_mass = unnormalized.sum(-1, keepdim=True)
    contribution = torch.where(
        member_mass > 0,
        unnormalized / member_mass.clamp_min(tiny),
        torch.zeros_like(unnormalized),
    )
    return contribution.to(ell.dtype)


def bounded_four_cycle_perturbation(
    path_mass: Tensor,
    rows: Sequence[int],
    members: Sequence[int],
    requested_dose: float,
) -> tuple[Tensor, float]:
    """Apply a positive 2x2 four-cycle while preserving row and column mass.

    ``path_mass`` is a single query's ``Q_ai = w_a R_ai`` matrix with shape
    ``[A,N]``. The cycle signs are ``[[+,-],[-,+]]`` for the supplied row and
    member order. The requested dose is clipped to the two entries that lose
    mass, and the applied finite dose is returned. The input is not mutated.
    """
    if path_mass.ndim != 2 or not path_mass.is_floating_point():
        raise ValueError("Four-cycle path mass must be a floating-point [A,N] matrix.")
    if not bool(torch.isfinite(path_mass).all()) or bool((path_mass < 0).any()):
        raise ValueError("Four-cycle path mass must be finite and nonnegative.")
    if len(rows) != 2 or len(members) != 2:
        raise ValueError("A four-cycle requires two distinct rows and two distinct members.")
    row0, row1 = rows
    member0, member1 = members
    if row0 == row1 or member0 == member1:
        raise ValueError("Four-cycle rows and members must each be distinct.")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in (*rows, *members)):
        raise TypeError("Four-cycle row and member indices must be integers.")
    if not (0 <= row0 < path_mass.shape[0] and 0 <= row1 < path_mass.shape[0]):
        raise IndexError("Four-cycle row index is outside the path-mass matrix.")
    if not (0 <= member0 < path_mass.shape[1] and 0 <= member1 < path_mass.shape[1]):
        raise IndexError("Four-cycle member index is outside the path-mass matrix.")
    if isinstance(requested_dose, bool) or not isinstance(requested_dose, (int, float)):
        raise TypeError("Four-cycle dose must be a real number.")
    if not math.isfinite(float(requested_dose)) or requested_dose < 0:
        raise ValueError("Four-cycle dose must be finite and nonnegative.")

    capacity = min(float(path_mass[row0, member1]), float(path_mass[row1, member0]))
    applied_dose = min(float(requested_dose), capacity)
    changes = torch.zeros_like(path_mass)
    changes[row0, member0] = applied_dose
    changes[row0, member1] = -applied_dose
    changes[row1, member0] = -applied_dose
    changes[row1, member1] = applied_dose
    return path_mass + changes, applied_dose


__all__ = [
    "bounded_four_cycle_perturbation",
    "organization_conditioned_role_contribution",
]
