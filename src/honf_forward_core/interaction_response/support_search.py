"""Bounded train-only response-support discovery and control supports.

The discovery routine is a group-sparse, frozen-response approximation. It
selects among already evaluated unary/pair factor predictions and keeps
unknown response entries masked. It is a support teacher, not a causal graph
or a substitute for held-family adequacy checks.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from .types import ReceiverSupport, ResponseFactor, ValidityNeighborhood


@dataclass(frozen=True)
class SupportSearchResult:
    selected_factor_ids: tuple[str, ...]
    coefficients: torch.Tensor
    normalized_rmse_by_role: Mapping[str, float | None]
    adequate_by_role: Mapping[str, bool | None]
    evaluated_candidate_ids: tuple[str, ...]
    truncated: bool

    @property
    def adequate(self) -> bool:
        return bool(self.adequate_by_role) and all(value is True for value in self.adequate_by_role.values())


def enumerate_unary_pair_candidates(
    module_ids: Sequence[Hashable],
    output_roles: Sequence[str],
    *,
    validity: ValidityNeighborhood | None = None,
    receiver_support: ReceiverSupport | None = None,
) -> tuple[ResponseFactor, ...]:
    """Build the physical unary/pair candidate ceiling from stable IDs."""

    ids = tuple(module_ids)
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Candidate generation requires unique active physical module IDs.")
    roles = tuple(output_roles)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("Candidate generation requires unique output roles.")
    local_validity = validity or ValidityNeighborhood(max_abs_delta_by_module={})
    local_receivers = receiver_support or ReceiverSupport(all_receivers=True)
    stable_ids = tuple(sorted(ids, key=lambda value: (type(value).__qualname__, repr(value))))
    factors = []
    for donor_count in (1, 2):
        for donor_set in itertools.combinations(stable_ids, donor_count):
            factor_id = f"order{donor_count}:" + "|".join(
                f"{type(module_id).__qualname__}:{module_id!r}" for module_id in donor_set
            )
            factors.append(
                ResponseFactor(
                    factor_id=factor_id,
                    donor_ids=tuple(donor_set),
                    output_roles=roles,
                    validity=local_validity,
                    receiver_support=local_receivers,
                )
            )
    return tuple(factors)


def _flatten_response_blocks(
    factors: Sequence[ResponseFactor],
    candidate_responses: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    resolved_masks: Mapping[str, torch.Tensor],
    response_scales: Mapping[str, float],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    role_matrices: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    x_blocks = []
    y_blocks = []
    for role, target in targets.items():
        if role not in candidate_responses or role not in resolved_masks:
            raise KeyError(f"Missing candidate response or resolved mask for role {role!r}.")
        predictions = candidate_responses[role]
        mask = resolved_masks[role].to(dtype=torch.bool)
        if predictions.shape[0] != len(factors) or predictions.shape[1:] != target.shape:
            raise ValueError(f"Candidate responses for role {role!r} must have shape [E,*target.shape].")
        if mask.shape != target.shape:
            raise ValueError(f"Resolved mask for role {role!r} must match its target shape.")
        scale = float(response_scales[role])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Response scale for role {role!r} must be finite and positive.")
        valid_count = int(mask.sum().item())
        if valid_count == 0:
            role_matrices[role] = (predictions.new_zeros(len(factors), 0), target.new_zeros(0), mask)
            continue
        divisor = scale * math.sqrt(valid_count)
        x_role = predictions.reshape(len(factors), -1)[:, mask.reshape(-1)] / divisor
        y_role = target.reshape(-1)[mask.reshape(-1)] / divisor
        x_blocks.append(x_role)
        y_blocks.append(y_role)
        role_matrices[role] = (x_role, y_role, mask)
    if not y_blocks:
        raise ValueError("No resolved response values are available for support discovery.")
    return torch.cat(x_blocks, dim=1), torch.cat(y_blocks), role_matrices


def _nnls_refit(matrix: torch.Tensor, target: torch.Tensor, max_steps: int = 160) -> torch.Tensor:
    """Small projected-gradient NNLS fit for a selected factor support."""

    if matrix.shape[1] == 0:
        return target.new_zeros(0)
    if matrix.shape[0] == 0:
        return target.new_zeros(matrix.shape[1])
    spectral_bound = torch.linalg.matrix_norm(matrix, ord="fro").square().clamp_min(1.0e-12)
    step_size = 1.0 / spectral_bound
    coefficients = target.new_zeros(matrix.shape[1])
    gram = matrix.transpose(0, 1) @ matrix
    right = matrix.transpose(0, 1) @ target
    for _ in range(max_steps):
        coefficients = torch.relu(coefficients - step_size * (gram @ coefficients - right))
    return coefficients


def _fit_and_errors(
    factor_matrix: torch.Tensor,
    target: torch.Tensor,
    role_matrices: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    selected_indices: Sequence[int],
    tolerances: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, float | None], dict[str, bool | None], torch.Tensor]:
    if selected_indices:
        matrix = factor_matrix[selected_indices].transpose(0, 1)
    else:
        matrix = factor_matrix.new_zeros(factor_matrix.shape[1], 0)
    coefficients = _nnls_refit(matrix, target)
    residual = matrix @ coefficients - target if selected_indices else -target
    errors: dict[str, float | None] = {}
    adequate: dict[str, bool | None] = {}
    offset = 0
    for role, (role_x, role_y, _mask) in role_matrices.items():
        width = int(role_y.numel())
        if width == 0:
            errors[role] = None
            adequate[role] = None
            continue
        role_residual = residual[offset : offset + width]
        # Each role block was divided by scale*sqrt(n), so its residual norm
        # already equals that role's normalized RMSE.
        rmse = float(torch.linalg.vector_norm(role_residual).item())
        tolerance = float(tolerances[role])
        errors[role] = rmse
        adequate[role] = rmse <= tolerance
        offset += width
    return coefficients, errors, adequate, residual


def group_sparse_support_search(
    factors: Sequence[ResponseFactor],
    candidate_responses: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    resolved_masks: Mapping[str, torch.Tensor],
    *,
    response_scales: Mapping[str, float],
    tolerances: Mapping[str, float],
    costs: torch.Tensor | None = None,
    max_additions: int = 12,
    max_candidate_evaluations: int = 40,
    initialize_with_unary: bool = True,
) -> SupportSearchResult:
    """Choose a sparse, nonnegative factor support from training evidence.

    ``candidate_responses`` must contain already evaluated model-factor
    outputs. This bounded linearized teacher performs no hidden candidate
    forwards and refuses to exceed the stated 40 candidate-evaluation ceiling.
    Each output role is normalized and assessed separately; unresolved entries
    are removed by their mask and never treated as zeros.
    """

    factors = tuple(factors)
    if len(factors) > max_candidate_evaluations:
        raise ValueError(
            f"Received {len(factors)} evaluated candidates, exceeding the limit {max_candidate_evaluations}; "
            "preselect a bounded candidate list before computing their responses."
        )
    if max_candidate_evaluations <= 0 or max_additions < 0:
        raise ValueError("Support search budgets must be positive (and max_additions may be zero).")
    if len({factor.factor_id for factor in factors}) != len(factors):
        raise ValueError("Support candidate factor IDs must be unique.")
    if set(targets) - set(tolerances) or set(targets) - set(response_scales):
        raise ValueError("Every measured output role needs a response scale and adequacy tolerance.")
    if any(not math.isfinite(float(tolerances[role])) or float(tolerances[role]) < 0.0 for role in targets):
        raise ValueError("Response adequacy tolerances must be finite and nonnegative.")
    factor_matrix, flat_target, role_matrices = _flatten_response_blocks(
        factors, candidate_responses, targets, resolved_masks, response_scales
    )
    if costs is None:
        cost_tensor = factor_matrix.new_tensor([1.0 + 0.5 * len(factor.donor_ids) for factor in factors])
    else:
        if costs.shape != (len(factors),) or bool((costs <= 0.0).any()):
            raise ValueError("Factor costs must be positive and have shape [E].")
        cost_tensor = costs.to(device=factor_matrix.device, dtype=factor_matrix.dtype)

    unary_indices = [index for index, factor in enumerate(factors) if len(factor.donor_ids) == 1]
    selected = unary_indices[:] if initialize_with_unary else []
    if len(selected) > max_additions and not initialize_with_unary:
        selected = selected[:max_additions]
    coefficients, errors, adequate, residual = _fit_and_errors(
        factor_matrix, flat_target, role_matrices, selected, tolerances
    )

    additions = 0
    known_adequate = [value for value in adequate.values() if value is not None]
    while not (bool(known_adequate) and all(known_adequate)) and additions < max_additions:
        remaining = [index for index in range(len(factors)) if index not in selected]
        if not remaining:
            break
        # Rank the already evaluated candidates by residual correlation per
        # declared response-work cost, then verify each proposal by a refit.
        current_norm = torch.linalg.vector_norm(residual).square()
        best: tuple[float, int, torch.Tensor, dict[str, float | None], dict[str, bool | None], torch.Tensor] | None = (
            None
        )
        for candidate_index in remaining:
            proposed = [*selected, candidate_index]
            proposal_coeff, proposal_errors, proposal_adequate, proposal_residual = _fit_and_errors(
                factor_matrix, flat_target, role_matrices, proposed, tolerances
            )
            reduction = current_norm - torch.linalg.vector_norm(proposal_residual).square()
            score = float((reduction / cost_tensor[candidate_index]).item())
            if best is None or score > best[0]:
                best = (score, candidate_index, proposal_coeff, proposal_errors, proposal_adequate, proposal_residual)
        if best is None or best[0] <= 1.0e-12:
            break
        _, accepted, coefficients, errors, adequate, residual = best
        selected.append(accepted)
        additions += 1
        known_adequate = [value for value in adequate.values() if value is not None]

    # Backward deletion preserves separate response tolerances, retaining a
    # smaller adequate support where this frozen dictionary permits it.
    for candidate_index in tuple(selected):
        proposed = [index for index in selected if index != candidate_index]
        proposal_coeff, proposal_errors, proposal_adequate, proposal_residual = _fit_and_errors(
            factor_matrix, flat_target, role_matrices, proposed, tolerances
        )
        if proposal_adequate and all(value is None or value for value in proposal_adequate.values()):
            selected = proposed
            coefficients, errors, adequate, residual = (
                proposal_coeff,
                proposal_errors,
                proposal_adequate,
                proposal_residual,
            )

    selected = sorted(selected)
    coefficients, errors, adequate, _ = _fit_and_errors(factor_matrix, flat_target, role_matrices, selected, tolerances)
    return SupportSearchResult(
        selected_factor_ids=tuple(factors[index].factor_id for index in selected),
        coefficients=coefficients,
        normalized_rmse_by_role=errors,
        adequate_by_role=adequate,
        evaluated_candidate_ids=tuple(factor.factor_id for factor in factors),
        truncated=bool(additions >= max_additions and any(value is False for value in adequate.values())),
    )


def size_matched_random_support(
    candidates: Sequence[ResponseFactor],
    reference_support: Sequence[ResponseFactor],
    *,
    seed: int,
) -> tuple[ResponseFactor, ...]:
    """Sample a deterministic random support matched by unary/pair counts."""

    candidate_by_order: dict[int, list[ResponseFactor]] = {1: [], 2: []}
    for factor in candidates:
        candidate_by_order[len(factor.donor_ids)].append(factor)
    reference_counts = {order: 0 for order in candidate_by_order}
    for factor in reference_support:
        reference_counts[len(factor.donor_ids)] += 1
    rng = random.Random(seed)
    chosen = []
    for order, count in reference_counts.items():
        available = candidate_by_order[order]
        if count > len(available):
            raise ValueError(f"Cannot match {count} order-{order} factors from {len(available)} candidates.")
        chosen.extend(rng.sample(available, count))
    return tuple(chosen)


def select_factor_control(
    candidates: Sequence[ResponseFactor],
    mode: Literal["all_candidates", "unary_only", "adaptive", "fixed", "random_matched"],
    *,
    adaptive_ids: Sequence[str] = (),
    fixed_ids: Sequence[str] = (),
    matched_to: Sequence[ResponseFactor] = (),
    seed: int = 0,
) -> tuple[ResponseFactor, ...]:
    """Return all-candidate, unary, adaptive, fixed or size-matched controls."""

    factors = tuple(candidates)
    by_id = {factor.factor_id: factor for factor in factors}
    if mode == "all_candidates":
        return factors
    if mode == "unary_only":
        return tuple(factor for factor in factors if len(factor.donor_ids) == 1)
    if mode == "adaptive":
        ids = tuple(adaptive_ids)
        if set(ids) - by_id.keys():
            raise KeyError("Adaptive support refers to a factor outside the candidate set.")
        return tuple(by_id[factor_id] for factor_id in ids)
    if mode == "fixed":
        ids = tuple(fixed_ids)
        if set(ids) - by_id.keys():
            raise KeyError("Fixed support refers to a factor outside the candidate set.")
        return tuple(by_id[factor_id] for factor_id in ids)
    if mode == "random_matched":
        return size_matched_random_support(factors, matched_to, seed=seed)
    raise ValueError(f"Unknown factor support control mode: {mode!r}.")
