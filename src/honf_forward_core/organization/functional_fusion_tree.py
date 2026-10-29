"""Continuous, source-action-aware fusion on a fixed laminar proposal tree.

The module contains only the small differentiable algebra used by the v4
continuous functional coalescence backend. Source incidence and value rows
remain outside this module and are never compacted here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from honf_forward_core.interface_fields.routing_index.sparse_projection import (
    masked_sparsemax,
)

Tensor = torch.Tensor
ProbePhase = Literal["P0", "P1", "P2"]
PROBE_ROLES = ("environment", "physical_ports", "outside_temperature")
GRAM_BLOCKS = 4
DEFAULT_CLOSE_RMS = 0.02
DEFAULT_KEEP_RMS = 0.06
DEFAULT_DENOMINATOR_FLOOR = 1.0e-8
DEFAULT_ZERO_ENERGY = 1.0e-30


def _require_tensor(name: str, value: Tensor, *, rank: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}; got {value.ndim}.")


@dataclass(frozen=True)
class FunctionalProbeFamily:
    """One independent role in a case/phase probe catalogue.

    Coordinates and weights use canonical flattened shapes ``[B,Q,2]`` and
    ``[B,Q]``. Invalid entries are excluded by ``valid_mask``; they must not
    be represented by negative-infinity logits or inferred from query values.
    """

    coordinates: Tensor
    valid_mask: Tensor
    weights: Tensor

    def __post_init__(self) -> None:
        _require_tensor("probe coordinates", self.coordinates, rank=3)
        _require_tensor("probe valid_mask", self.valid_mask, rank=2)
        _require_tensor("probe weights", self.weights, rank=2)
        if int(self.coordinates.shape[-1]) != 2:
            raise ValueError("probe coordinates must have final dimension 2.")
        expected = tuple(self.coordinates.shape[:2])
        if tuple(self.valid_mask.shape) != expected:
            raise ValueError("probe valid_mask must align with [B,Q] coordinates.")
        if tuple(self.weights.shape) != expected:
            raise ValueError("probe weights must align with [B,Q] coordinates.")
        if not self.coordinates.is_floating_point():
            raise TypeError("probe coordinates must be floating point.")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("probe valid_mask must have bool dtype.")
        if not self.weights.is_floating_point():
            raise TypeError("probe weights must be floating point.")
        if (
            self.coordinates.device != self.valid_mask.device
            or self.coordinates.device != self.weights.device
        ):
            raise ValueError("probe coordinates, masks, and weights must share a device.")


@dataclass(frozen=True)
class FunctionalProbeCatalogue:
    """Phase-aware probe coordinates available before a functional read.

    The adapter supplies physical ports and outside-temperature probes using
    only the current phase state. The backend fills the environment role from
    ``EncodedInterfaceCase`` with :meth:`with_environment`.
    """

    phase: ProbePhase
    physical_ports: FunctionalProbeFamily
    outside_temperature: FunctionalProbeFamily
    environment: FunctionalProbeFamily | None = None

    def __post_init__(self) -> None:
        if self.phase not in {"P0", "P1", "P2"}:
            raise ValueError("phase must be one of 'P0', 'P1', or 'P2'.")
        batch = int(self.physical_ports.coordinates.shape[0])
        if int(self.outside_temperature.coordinates.shape[0]) != batch:
            raise ValueError("all probe families must have the same batch size.")
        if self.environment is not None and int(self.environment.coordinates.shape[0]) != batch:
            raise ValueError("all probe families must have the same batch size.")

    def with_environment(self, coordinates: Tensor, weights: Tensor) -> FunctionalProbeCatalogue:
        """Return a catalogue with environment probes from the encoded case."""

        _require_tensor("environment coordinates", coordinates, rank=3)
        _require_tensor("environment weights", weights, rank=2)
        if tuple(coordinates.shape[:2]) != tuple(weights.shape):
            raise ValueError("environment weights must align with [B,Q] coordinates.")
        environment = FunctionalProbeFamily(
            coordinates=coordinates,
            valid_mask=torch.ones_like(weights, dtype=torch.bool),
            weights=weights,
        )
        return FunctionalProbeCatalogue(
            phase=self.phase,
            physical_ports=self.physical_ports,
            outside_temperature=self.outside_temperature,
            environment=environment,
        )

    def families(self) -> dict[str, FunctionalProbeFamily]:
        """Return the separate environment, port, and outside probe roles."""

        if self.environment is None:
            raise ValueError("environment probes must be attached with with_environment().")
        return {
            "environment": self.environment,
            "physical_ports": self.physical_ports,
            "outside_temperature": self.outside_temperature,
        }


@dataclass(frozen=True)
class SourceActionScores:
    """Raw and normalized source-action scores for a node bank."""

    score2: Tensor  # [B,N]
    numerator: Tensor  # [B,R,4,N]
    denominator: Tensor  # [B,R,4,N]
    ratio: Tensor  # [B,R,4,N]
    role_names: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseSourceActionScores:
    """Pair scores in dense proposal order plus their upper-triangle indices."""

    score2: Tensor  # [B,K,K]
    numerator: Tensor  # [B,R,4,K,K]
    denominator: Tensor  # [B,R,4,K,K]
    ratio: Tensor  # [B,R,4,K,K]
    left: Tensor  # [P]
    right: Tensor  # [P]
    role_names: tuple[str, ...]


@dataclass(frozen=True)
class NodeContraction:
    """Continuous node taper and corresponding tree-row coefficients."""

    taper: Tensor  # [B,N], 0 means fully tied, 1 means unchanged
    gamma: Tensor  # [B,N], contraction amount after eligibility fade
    residual_scale: Tensor  # [B,N], exactly 1-gamma
    close_scale: Tensor  # [B,N]
    keep_scale: Tensor  # [B,N]
    eligibility: Tensor  # [B,N]


@dataclass(frozen=True)
class PackedTreeClasses:
    """Exact closed-subtree packing metadata for the original K proposals."""

    membership: Tensor  # [B,K,Rmax]
    multiplicity: Tensor  # [B,Rmax]
    valid: Tensor  # [B,Rmax]
    class_id: Tensor  # [B,K], -1 for inactive proposals
    root_leaf: Tensor  # [B,K], -1 for inactive proposals


@dataclass(frozen=True)
class CombinedPostTransformDiscrepancy:
    """Read-input discrepancy after all tree transforms have been applied."""

    score2: Tensor  # [B]
    numerator: Tensor  # [B,R,4]
    denominator: Tensor  # [B,R,4]
    ratio: Tensor  # [B,R,4]
    role_names: tuple[str, ...]


def build_source_action_grams(
    module_incidence: Tensor,
    module_measure: Tensor,
    environment_incidence: Tensor,
    environment_measure: Tensor,
    group_control: Tensor,
    module_control_gain: Callable[[Tensor], Tensor],
    environment_score_control: Callable[[Tensor], Tensor],
) -> Tensor:
    """Build the four differentiable K-by-K read-input Gram matrices.

    The supplied maps are the backend's existing bias-free control maps. The
    resulting block order is module incidence, module gain action, environment
    incidence, environment score action.
    """

    for name, tensor in (
        ("module_incidence", module_incidence),
        ("environment_incidence", environment_incidence),
        ("group_control", group_control),
    ):
        _require_tensor(name, tensor, rank=3)
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point.")
    _require_tensor("module_measure", module_measure, rank=2)
    _require_tensor("environment_measure", environment_measure, rank=2)
    batch, module_rows, groups = module_incidence.shape
    if tuple(module_measure.shape) != (batch, module_rows):
        raise ValueError("module_measure must align with module_incidence.")
    if tuple(environment_incidence.shape[:1]) != (batch,):
        raise ValueError("environment_incidence must share the batch dimension.")
    if int(environment_incidence.shape[-1]) != groups:
        raise ValueError("module and environment incidence must share proposal K.")
    if tuple(environment_measure.shape) != tuple(environment_incidence.shape[:2]):
        raise ValueError("environment_measure must align with environment_incidence.")
    if tuple(group_control.shape[:2]) != (batch, groups):
        raise ValueError("group_control must have shape [B,K,H].")
    if module_incidence.device != group_control.device or environment_incidence.device != group_control.device:
        raise ValueError("source incidence and group controls must share a device.")

    module_gain = module_control_gain(group_control)
    environment_score = environment_score_control(group_control)
    if tuple(module_gain.shape[:2]) != (batch, groups):
        raise ValueError("module_control_gain output must align with [B,K].")
    if tuple(environment_score.shape[:2]) != (batch, groups):
        raise ValueError("environment_score_control output must align with [B,K].")
    gram_module_a = torch.einsum(
        "bmk,bm,bmj->bkj", module_incidence, module_measure, module_incidence
    )
    gram_environment_a = torch.einsum(
        "bek,be,bej->bkj",
        environment_incidence,
        environment_measure,
        environment_incidence,
    )
    gram_module_gain = gram_module_a * torch.einsum(
        "bkh,bjh->bkj", module_gain, module_gain
    )
    gram_environment_score = gram_environment_a * torch.einsum(
        "bkh,bjh->bkj", environment_score, environment_score
    )
    blocks = torch.stack(
        [gram_module_a, gram_module_gain, gram_environment_a, gram_environment_score],
        dim=1,
    )
    return 0.5 * (blocks + blocks.transpose(-1, -2))


def _normalized_weights(family: FunctionalProbeFamily, *, dtype: torch.dtype) -> Tensor:
    weights = family.weights.to(dtype=dtype)
    valid = family.valid_mask
    # Adapter weights are nonnegative quadrature/multiplicity weights. Clamp
    # protects a future caller from a signed weight corrupting a PSD energy.
    weights = torch.where(valid, weights.clamp_min(0.0), torch.zeros_like(weights))
    total = weights.sum(dim=-1, keepdim=True)
    return torch.where(total > 0.0, weights / total.clamp_min(1.0e-30), torch.zeros_like(weights))


def _check_role_logits(
    logits_by_role: Mapping[str, Tensor],
    catalogue: FunctionalProbeCatalogue,
    proposal_valid: Tensor,
) -> tuple[tuple[str, FunctionalProbeFamily, Tensor], ...]:
    families = catalogue.families()
    missing = sorted(set(families) - set(logits_by_role))
    if missing:
        raise ValueError(f"raw parent logits are missing probe roles: {missing}.")
    batch, groups = proposal_valid.shape
    checked: list[tuple[str, FunctionalProbeFamily, Tensor]] = []
    for role in PROBE_ROLES:
        family = families[role]
        logits = logits_by_role[role]
        _require_tensor(f"{role} raw logits", logits, rank=3)
        if tuple(logits.shape) != (
            batch,
            int(family.coordinates.shape[1]),
            groups,
        ):
            raise ValueError(f"{role} raw logits must have shape [B,Q,K] matching its catalogue.")
        if logits.device != proposal_valid.device or family.coordinates.device != logits.device:
            raise ValueError("probe logits, proposal masks, and catalogue must share a device.")
        checked.append((role, family, logits))
    return tuple(checked)


def _tied_node_logits(logits: Tensor, node_membership: Tensor) -> Tensor:
    """Apply every hypothetical P_C to finite logits without using -inf."""

    membership = node_membership.to(device=logits.device, dtype=logits.dtype)
    counts = membership.sum(dim=-1)
    safe_counts = counts.clamp_min(1.0)
    means = torch.einsum("nk,bqk->bnq", membership, logits) / safe_counts[None, :, None]
    tied = torch.where(
        membership[None, :, None, :] > 0.0,
        means[..., None],
        logits[:, None, :, :],
    )
    return tied


def _score_route_changes(
    role_changes: list[tuple[str, Tensor, Tensor, Tensor]],
    action_grams: Tensor,
    *,
    denominator_floor: float,
    zero_energy: float,
) -> SourceActionScores:
    """Score per-role alpha changes; each tuple is (name, alpha0, delta, w)."""

    raw_numerators: list[Tensor] = []
    raw_denominators: list[Tensor] = []
    for _, alpha0, delta, weights in role_changes:
        block_numerators: list[Tensor] = []
        block_denominators: list[Tensor] = []
        for block_index in range(GRAM_BLOCKS):
            gram = action_grams[:, block_index]
            base = torch.einsum("bqi,bij,bqj->bq", alpha0, gram, alpha0).clamp_min(0.0)
            # delta is either [B,N,Q,K] or [B,Q,K]. Normalize the internal
            # node axis to simplify the exact same block arithmetic.
            if delta.ndim == 3:
                quad = torch.einsum("bqi,bij,bqj->bq", delta, gram, delta).clamp_min(0.0)
                numerator = quad.amax(dim=-1, keepdim=True)
            else:
                quad = torch.einsum("bnqi,bij,bnqj->bnq", delta, gram, delta).clamp_min(0.0)
                numerator = quad.amax(dim=-1)
            denominator = (base * weights).sum(dim=-1)
            block_numerators.append(numerator)
            block_denominators.append(denominator)
        raw_numerators.append(torch.stack(block_numerators, dim=1))
        raw_denominators.append(torch.stack(block_denominators, dim=1))

    numerator = torch.stack(raw_numerators, dim=1)
    denominator = torch.stack(raw_denominators, dim=1)
    # Node changes have [B,R,4,N]; combined changes use a singleton node
    # dimension internally and are squeezed by their public wrapper.
    if numerator.ndim == 3:
        numerator = numerator.unsqueeze(-1)
        denominator = denominator.unsqueeze(-1)
    else:
        denominator = denominator.unsqueeze(-1).expand_as(numerator)
    zero_block = (numerator <= float(zero_energy)) & (denominator <= float(zero_energy))
    ratio = numerator / (denominator + float(denominator_floor))
    ratio = torch.where(zero_block, torch.zeros_like(ratio), ratio)
    score2 = ratio.amax(dim=(1, 2))
    return SourceActionScores(
        score2=score2,
        numerator=numerator,
        denominator=denominator,
        ratio=ratio,
        role_names=tuple(role for role, *_ in role_changes),
    )


def node_source_action_scores(
    logits_by_role: Mapping[str, Tensor],
    proposal_valid: Tensor,
    action_grams: Tensor,
    catalogue: FunctionalProbeCatalogue,
    node_membership: Tensor,
    *,
    denominator_floor: float = DEFAULT_DENOMINATOR_FLOOR,
    zero_energy: float = DEFAULT_ZERO_ENERGY,
) -> SourceActionScores:
    """Measure hypothetical exact ties using source-action read inputs.

    ``logits_by_role`` must contain finite parent-router logits before masking.
    Validity is applied only inside masked sparsemax. The numerator takes the
    maximum over probes; the denominator is a normalized weighted mean of the
    parent action energy. Raw numerators and denominators are retained.
    """

    _require_tensor("proposal_valid", proposal_valid, rank=2)
    _require_tensor("action_grams", action_grams, rank=4)
    _require_tensor("node_membership", node_membership, rank=2)
    if proposal_valid.dtype != torch.bool:
        raise TypeError("proposal_valid must have bool dtype.")
    batch, groups = proposal_valid.shape
    if tuple(action_grams.shape) != (batch, GRAM_BLOCKS, groups, groups):
        raise ValueError("action_grams must have shape [B,4,K,K].")
    if int(node_membership.shape[1]) != groups or int(node_membership.shape[0]) == 0:
        raise ValueError("node_membership must have nonempty shape [N,K].")
    if action_grams.device != proposal_valid.device or node_membership.device != proposal_valid.device:
        raise ValueError("proposal masks, action grams, and node metadata must share a device.")
    nodes = node_membership.to(dtype=torch.bool)
    checked = _check_role_logits(logits_by_role, catalogue, proposal_valid)
    role_changes: list[tuple[str, Tensor, Tensor, Tensor]] = []
    for role, family, raw_logits in checked:
        valid_probe = family.valid_mask
        route_valid = proposal_valid[:, None, :] & valid_probe[:, :, None]
        alpha0 = masked_sparsemax(raw_logits, route_valid)
        tied_logits = _tied_node_logits(raw_logits, nodes)
        node_route_valid = route_valid[:, None, :, :].expand_as(tied_logits)
        alpha_node = masked_sparsemax(tied_logits, node_route_valid)
        delta = alpha_node - alpha0[:, None, :, :]
        weights = _normalized_weights(family, dtype=raw_logits.dtype)
        role_changes.append((role, alpha0, delta, weights))
    return _score_route_changes(
        role_changes,
        action_grams,
        denominator_floor=denominator_floor,
        zero_energy=zero_energy,
    )


def pairwise_source_action_scores(
    logits_by_role: Mapping[str, Tensor],
    proposal_valid: Tensor,
    action_grams: Tensor,
    catalogue: FunctionalProbeCatalogue,
    *,
    denominator_floor: float = DEFAULT_DENOMINATOR_FLOOR,
    zero_energy: float = DEFAULT_ZERO_ENERGY,
) -> PairwiseSourceActionScores:
    """Score every proposal pair under the same hypothetical-tie formula."""

    _require_tensor("proposal_valid", proposal_valid, rank=2)
    batch, groups = proposal_valid.shape
    left, right = torch.triu_indices(
        groups, groups, offset=1, device=proposal_valid.device
    )
    nodes = torch.zeros((int(left.numel()), groups), device=proposal_valid.device, dtype=torch.bool)
    nodes.scatter_(1, left[:, None], True)
    nodes.scatter_(1, right[:, None], True)
    scores = node_source_action_scores(
        logits_by_role,
        proposal_valid,
        action_grams,
        catalogue,
        nodes,
        denominator_floor=denominator_floor,
        zero_energy=zero_energy,
    )
    score2 = scores.score2.new_zeros((batch, groups, groups))
    numerator = scores.numerator.new_zeros(
        (batch, len(scores.role_names), GRAM_BLOCKS, groups, groups)
    )
    denominator = scores.denominator.new_zeros(numerator.shape)
    ratio = scores.ratio.new_zeros(numerator.shape)
    score2[:, left, right] = scores.score2
    numerator[:, :, :, left, right] = scores.numerator
    denominator[:, :, :, left, right] = scores.denominator
    ratio[:, :, :, left, right] = scores.ratio
    score2 = score2 + score2.transpose(-1, -2)
    numerator = numerator + numerator.transpose(-1, -2)
    denominator = denominator + denominator.transpose(-1, -2)
    ratio = ratio + ratio.transpose(-1, -2)
    return PairwiseSourceActionScores(
        score2=score2,
        numerator=numerator,
        denominator=denominator,
        ratio=ratio,
        left=left,
        right=right,
        role_names=scores.role_names,
    )


def combined_post_transform_discrepancy(
    logits_by_role: Mapping[str, Tensor],
    transformed_logits_by_role: Mapping[str, Tensor],
    proposal_valid: Tensor,
    action_grams: Tensor,
    catalogue: FunctionalProbeCatalogue,
    *,
    denominator_floor: float = DEFAULT_DENOMINATOR_FLOOR,
    zero_energy: float = DEFAULT_ZERO_ENERGY,
) -> CombinedPostTransformDiscrepancy:
    """Measure the realized full-T routing change against parent routing."""

    _require_tensor("proposal_valid", proposal_valid, rank=2)
    batch, groups = proposal_valid.shape
    if tuple(action_grams.shape) != (batch, GRAM_BLOCKS, groups, groups):
        raise ValueError("action_grams must have shape [B,4,K,K].")
    checked = _check_role_logits(logits_by_role, catalogue, proposal_valid)
    role_changes: list[tuple[str, Tensor, Tensor, Tensor]] = []
    for role, family, parent_logits in checked:
        if role not in transformed_logits_by_role:
            raise ValueError(f"transformed logits are missing probe role {role!r}.")
        transformed = transformed_logits_by_role[role]
        if tuple(transformed.shape) != tuple(parent_logits.shape):
            raise ValueError(f"transformed {role} logits must align with parent logits.")
        route_valid = proposal_valid[:, None, :] & family.valid_mask[:, :, None]
        alpha0 = masked_sparsemax(parent_logits, route_valid)
        alpha_t = masked_sparsemax(transformed, route_valid)
        weights = _normalized_weights(family, dtype=parent_logits.dtype)
        role_changes.append((role, alpha0, alpha_t - alpha0, weights))
    scores = _score_route_changes(
        role_changes,
        action_grams,
        denominator_floor=denominator_floor,
        zero_energy=zero_energy,
    )
    # The internal scorer represents the combined case score with N=1.
    return CombinedPostTransformDiscrepancy(
        score2=scores.score2[:, 0],
        numerator=scores.numerator[..., 0],
        denominator=scores.denominator[..., 0],
        ratio=scores.ratio[..., 0],
        role_names=scores.role_names,
    )


def contracted_logits(raw_logits: Tensor, transform: Tensor) -> Tensor:
    """Apply the continuous K-by-K tree transform to finite parent logits."""

    _require_tensor("raw_logits", raw_logits, rank=3)
    _require_tensor("transform", transform, rank=3)
    batch, _, groups = raw_logits.shape
    if tuple(transform.shape) != (batch, groups, groups):
        raise ValueError("transform must have shape [B,K,K] aligned with raw_logits.")
    return torch.einsum("bkl,bql->bqk", transform, raw_logits)


def _quintic_smoothstep(value: Tensor) -> Tensor:
    unit = value.clamp(0.0, 1.0)
    return unit.pow(3) * (10.0 + unit * (-15.0 + 6.0 * unit))


def deterministic_complete_linkage(
    pair_dissimilarity: Tensor | list[list[float]],
) -> list[list[int]]:
    """Build bottom-up complete-linkage subsets with an explicit tie-break.

    At every step, choose the cluster pair with smallest maximum cross-pair
    dissimilarity. Exact ties are resolved by lexicographic order of the two
    sorted leaf tuples. Infinite dissimilarities are allowed for pairs with no
    eligible training record; NaN, negative, asymmetric, or nonsquare inputs
    are rejected. The returned list has ``K-1`` laminar non-singleton subsets.
    """

    matrix_tensor = torch.as_tensor(pair_dissimilarity, dtype=torch.float64)
    if matrix_tensor.ndim != 2 or matrix_tensor.shape[0] != matrix_tensor.shape[1]:
        raise ValueError("pair_dissimilarity must be a square [K,K] matrix.")
    groups = int(matrix_tensor.shape[0])
    if groups < 2:
        raise ValueError("complete linkage requires at least two leaves.")
    values = matrix_tensor.detach().cpu().tolist()
    for left in range(groups):
        for right in range(groups):
            value = float(values[left][right])
            if math.isnan(value) or value < 0.0:
                raise ValueError("pair dissimilarities must be nonnegative and not NaN.")
            mirror = float(values[right][left])
            if not math.isclose(value, mirror, rel_tol=0.0, abs_tol=1.0e-12) and not (
                math.isinf(value) and math.isinf(mirror)
            ):
                raise ValueError("pair_dissimilarity must be symmetric.")
    clusters: list[tuple[int, ...]] = [(leaf,) for leaf in range(groups)]
    subsets: list[list[int]] = []
    while len(clusters) > 1:
        candidates: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for first_index, first in enumerate(clusters):
            for second in clusters[first_index + 1 :]:
                complete = max(values[i][j] for i in first for j in second)
                left, right = sorted((first, second))
                candidates.append((float(complete), left, right))
        _, left, right = min(candidates)
        merged = tuple(sorted((*left, *right)))
        subsets.append(list(merged))
        clusters = [cluster for cluster in clusters if cluster not in {left, right}]
        clusters.append(merged)
        clusters.sort()
    return subsets


def node_contraction(
    score2: Tensor,
    proposal_mass: Tensor,
    node_membership: Tensor,
    active: Tensor,
    epoch: float | Tensor,
    *,
    close_rms: float = DEFAULT_CLOSE_RMS,
    keep_rms: float = DEFAULT_KEEP_RMS,
    ramp_start_epoch: float = 50.0,
    ramp_end_epoch: float = 150.0,
    eligibility_low: float = 1.0e-6,
    eligibility_high: float = 2.0e-6,
) -> NodeContraction:
    """Map normalized discrepancy and proposal masses to smooth node closure.

    The v4 schedule is the exact identity through epoch 50, then raises both
    RMS thresholds with a quintic endpoint-flat ramp ending at epoch 150.
    """

    _require_tensor("score2", score2, rank=2)
    _require_tensor("proposal_mass", proposal_mass, rank=2)
    _require_tensor("node_membership", node_membership, rank=2)
    _require_tensor("active", active, rank=2)
    batch, nodes = score2.shape
    if int(proposal_mass.shape[0]) != batch or tuple(active.shape) != tuple(proposal_mass.shape):
        raise ValueError("proposal_mass and active must share [B,K] with score2.")
    if tuple(node_membership.shape) != (nodes, int(proposal_mass.shape[1])):
        raise ValueError("node_membership must have shape [N,K] matching score2.")
    if active.dtype != torch.bool:
        active = active.to(dtype=torch.bool)
    if not 0.0 <= float(close_rms) < float(keep_rms):
        raise ValueError("thresholds must satisfy 0 <= close_rms < keep_rms.")
    if not 0.0 < float(eligibility_low) < float(eligibility_high):
        raise ValueError("eligibility thresholds must satisfy 0 < low < high.")
    device, dtype = score2.device, score2.dtype
    epoch_tensor = torch.as_tensor(epoch, device=device, dtype=dtype)
    ramp_fraction = ((epoch_tensor - float(ramp_start_epoch)) / (float(ramp_end_epoch) - float(ramp_start_epoch))).clamp(0.0, 1.0)
    ramp = _quintic_smoothstep(ramp_fraction)
    close_scale = float(close_rms) * ramp
    keep_scale = float(keep_rms) * ramp
    # The scheduled identity interval must be exact and avoid a 0/0 scale.
    if epoch_tensor.ndim == 0:
        numerator = score2 - close_scale.square()
        denominator = keep_scale.square() - close_scale.square()
        u = numerator / denominator.clamp_min(torch.finfo(dtype).tiny)
        scheduled_taper = _quintic_smoothstep(u)
        taper = torch.where(
            epoch_tensor <= float(ramp_start_epoch),
            torch.ones_like(scheduled_taper),
            scheduled_taper,
        )
    else:
        epoch_batch = torch.broadcast_to(epoch_tensor, (batch,))
        close_batch = torch.broadcast_to(close_scale, (batch,))
        keep_batch = torch.broadcast_to(keep_scale, (batch,))
        numerator = score2 - close_batch[:, None].square()
        denominator = keep_batch.square() - close_batch.square()
        u = numerator / denominator[:, None].clamp_min(torch.finfo(dtype).tiny)
        scheduled_taper = _quintic_smoothstep(u)
        taper = torch.where(
            (epoch_batch <= float(ramp_start_epoch))[:, None],
            torch.ones_like(scheduled_taper),
            scheduled_taper,
        )
    eligibility_per_proposal = _quintic_smoothstep(
        (proposal_mass - float(eligibility_low))
        / (float(eligibility_high) - float(eligibility_low))
    )
    members = node_membership.to(device=device, dtype=torch.bool)
    all_active = ((~members[None, :, :]) | active[:, None, :]).all(dim=-1)
    has_members = members.any(dim=-1)[None, :]
    node_eligibility = torch.where(
        members[None, :, :],
        eligibility_per_proposal[:, None, :],
        torch.ones_like(eligibility_per_proposal[:, None, :]),
    ).prod(dim=-1)
    node_eligibility = node_eligibility * (all_active & has_members).to(dtype)
    gamma = (1.0 - taper) * node_eligibility
    residual_scale = 1.0 - gamma
    return NodeContraction(
        taper=taper,
        gamma=gamma,
        residual_scale=residual_scale,
        close_scale=torch.broadcast_to(
            close_scale if close_scale.ndim == 0 else close_scale[:, None],
            score2.shape,
        ),
        keep_scale=torch.broadcast_to(
            keep_scale if keep_scale.ndim == 0 else keep_scale[:, None],
            score2.shape,
        ),
        eligibility=node_eligibility,
    )


def build_tree_transform(
    residual_scale: Tensor,
    node_membership: Tensor,
) -> Tensor:
    """Build ``T`` by applying the fixed bottom-up tree transforms in order."""

    _require_tensor("residual_scale", residual_scale, rank=2)
    _require_tensor("node_membership", node_membership, rank=2)
    batch, nodes = residual_scale.shape
    if int(node_membership.shape[0]) != nodes:
        raise ValueError("node_membership node count must match residual_scale.")
    groups = int(node_membership.shape[1])
    if groups == 0:
        raise ValueError("tree must have at least one proposal leaf.")
    membership = node_membership.to(device=residual_scale.device, dtype=torch.bool)
    transform = torch.eye(groups, device=residual_scale.device, dtype=residual_scale.dtype)
    transform = transform[None, :, :].expand(batch, -1, -1)
    for node_index in range(nodes):
        member_mask = membership[node_index]
        member_weight = member_mask.to(dtype=transform.dtype)
        member_count = member_weight.sum().clamp_min(1.0)
        mean_row = (
            transform * member_weight[None, :, None]
        ).sum(dim=1, keepdim=True) / member_count
        residual = residual_scale[:, node_index, None, None]
        mixed_rows = mean_row + residual * (transform - mean_row)
        mixed_rows = torch.where(residual == 0.0, mean_row.expand_as(transform), mixed_rows)
        mixed_rows = torch.where(residual == 1.0, transform, mixed_rows)
        transform = torch.where(member_mask[None, :, None], mixed_rows, transform)
    return transform


def pack_exact_closed_subtrees(
    closed_nodes: Tensor,
    node_membership: Tensor,
    active: Tensor,
) -> PackedTreeClasses:
    """Pack active leaves by exact closed ancestors, preserving inactive holes.

    Node rows are supplied in deterministic bottom-up order, so the greatest
    closed node index is the highest closed ancestor and overrides closures in
    its descendants. Only integer class metadata is nondifferentiable.
    """

    _require_tensor("closed_nodes", closed_nodes, rank=2)
    _require_tensor("node_membership", node_membership, rank=2)
    _require_tensor("active", active, rank=2)
    batch, nodes = closed_nodes.shape
    groups = int(node_membership.shape[1])
    if tuple(node_membership.shape[:1]) != (nodes,):
        raise ValueError("node_membership node count must match closed_nodes.")
    if tuple(active.shape) != (batch, groups):
        raise ValueError("active must have shape [B,K] matching node_membership.")
    device = active.device
    active_mask = active.to(dtype=torch.bool)
    members = node_membership.to(device=device, dtype=torch.bool)
    closed = closed_nodes.to(device=device, dtype=torch.bool)
    node_active = ((~members[None, :, :]) | active_mask[:, None, :]).all(dim=-1)
    effective_closed = closed & node_active
    node_ids = torch.arange(nodes, device=device, dtype=torch.long)
    closed_ancestor = effective_closed[:, :, None] & members[None, :, :]
    selected_node = torch.where(
        closed_ancestor,
        node_ids[None, :, None],
        torch.full((batch, nodes, groups), -1, device=device, dtype=torch.long),
    ).amax(dim=1)
    leaves = torch.arange(groups, device=device, dtype=torch.long)[None, :].expand(batch, -1)
    node_representative_leaf = torch.where(
        members,
        torch.arange(groups, device=device, dtype=torch.long)[None, :],
        torch.full((nodes, groups), groups, device=device, dtype=torch.long),
    ).amin(dim=-1)
    closed_representative = node_representative_leaf[
        selected_node.clamp_min(0)
    ]
    root_leaf = torch.where(selected_node >= 0, closed_representative, leaves)
    root_leaf = torch.where(active_mask, root_leaf, torch.full_like(root_leaf, -1))
    is_root = active_mask & (root_leaf == leaves)
    root_ordinal = is_root.to(torch.long).cumsum(dim=-1) - 1
    leaf_root_ordinal = root_ordinal.gather(1, root_leaf.clamp_min(0))
    class_id = torch.where(active_mask, leaf_root_ordinal, torch.full_like(leaf_root_ordinal, -1))
    active_class_count = is_root.sum(dim=-1)
    rmax = max(1, int(active_class_count.max().detach().cpu()))
    membership = F.one_hot(class_id.clamp_min(0), num_classes=rmax).to(dtype=torch.float32)
    membership = membership * active_mask[..., None].to(membership.dtype)
    valid = torch.arange(rmax, device=device)[None, :] < active_class_count[:, None]
    multiplicity = membership.sum(dim=1).to(dtype=torch.long)
    return PackedTreeClasses(
        membership=membership,
        multiplicity=multiplicity,
        valid=valid,
        class_id=class_id,
        root_leaf=root_leaf,
    )
