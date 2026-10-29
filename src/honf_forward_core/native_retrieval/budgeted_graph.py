"""Binary native-seed memberships and exact expected-work accounting for Round8."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .contracts import tensor_versions


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


def _bounded_scale(raw: Tensor) -> Tensor:
    """Map an unconstrained scalar to [0.25, 4], with exact initial value 1."""
    return 0.25 + 3.75 * torch.sigmoid(raw)


def _check_tensor(condition: Tensor, message: str) -> None:
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _validate_binary_mask(mask: Tensor, shape: tuple[int, ...], name: str) -> None:
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean with shape {shape}.")


def _relative_membership_scores(raw_scores: Tensor, allowed: Tensor) -> Tensor:
    """Center each head/seed score by its highest present nonself donor."""
    if raw_scores.ndim != 4:
        raise ValueError("mm_raw_scores must have shape [B,H,S,M].")
    batch, _, seeds, modules = raw_scores.shape
    if allowed.shape != (batch, seeds, modules) or allowed.dtype != torch.bool:
        raise ValueError("allowed must be boolean [B,S,M].")
    if modules == 0:
        return torch.zeros_like(raw_scores)
    allowed_heads = allowed[:, None, :, :]
    has_donor = allowed_heads.any(dim=-1, keepdim=True)
    masked = raw_scores.masked_fill(~allowed_heads, -torch.inf)
    row_max = masked.amax(dim=-1, keepdim=True)
    row_max = torch.where(has_donor, row_max, torch.zeros_like(row_max))
    relative = torch.where(allowed_heads & has_donor, raw_scores - row_max, torch.zeros_like(raw_scores))
    _check_tensor(torch.isfinite(relative).all(), "Eligible MM scores must be finite.")
    return relative


class BudgetedGraphParameters(nn.Module):
    """Small shared calibration for one physical membership per native seed.

    This module intentionally has no environment gate parameters or input-dependent
    network. The `head_logits` mix all preparation heads into one membership map.
    """

    def __init__(self, heads: int, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive.")
        self.head_logits = nn.Parameter(torch.zeros(heads, dtype=dtype))
        self.raw_scale = nn.Parameter(torch.tensor(_inverse_sigmoid(0.2), dtype=dtype))
        self.bias = nn.Parameter(torch.zeros((), dtype=dtype))

    @property
    def head_weights(self) -> Tensor:
        return torch.softmax(self.head_logits, dim=0)

    @property
    def scale(self) -> Tensor:
        return _bounded_scale(self.raw_scale)

    @property
    def membership_bias(self) -> Tensor:
        """Explicit name used in Round8 recipes and checkpoints."""
        return self.bias


@dataclass(frozen=True)
class BudgetedNativeGraph:
    """Sampled binary groups over original module records, without pooled values."""

    seed_ids: Tensor  # int [B,S], labels only
    seed_valid: Tensor  # bool [B,S]
    module_present: Tensor  # bool [B,M], physical record eligibility
    seed_coordinates: Tensor  # [B,S,C]
    seed_characteristic_lengths: Tensor  # [B,S,C]
    membership_logits: Tensor  # [B,S,M]
    membership_probabilities: Tensor  # [B,S,M]
    membership_hard: Tensor  # bool [B,S,M]
    membership_straight_through: Tensor  # [B,S,M], binary forward / relaxed backward
    member_indices: Tensor  # int [B,S,K], padded native module-axis indices
    member_valid: Tensor  # bool [B,S,K], padding is never a member
    source_geometry: Tensor  # [B,S,C+L], canonical query-dispatch tie key
    _tensor_versions: tuple[tuple[int, int, int], ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.seed_ids.ndim != 2 or self.seed_valid.shape != self.seed_ids.shape:
            raise ValueError("seed_ids and seed_valid must have matching [B,S] shapes.")
        if self.seed_valid.dtype != torch.bool:
            raise TypeError("seed_valid must be boolean.")
        batch, seeds = self.seed_ids.shape
        if self.module_present.ndim != 2 or self.module_present.shape[0] != batch:
            raise ValueError("module_present must have shape [B,M].")
        if self.module_present.dtype != torch.bool:
            raise TypeError("module_present must be boolean.")
        modules = self.module_present.shape[-1]
        if seeds != modules:
            raise ValueError("Round8 module seeds must preserve the original module axis.")
        if self.seed_coordinates.ndim != 3 or self.seed_coordinates.shape[:2] != (batch, seeds):
            raise ValueError("seed_coordinates must have shape [B,S,C].")
        if self.seed_characteristic_lengths.shape != self.seed_coordinates.shape:
            raise ValueError("seed characteristic lengths must match seed coordinates.")
        if self.source_geometry.ndim != 3 or self.source_geometry.shape[:2] != (batch, seeds):
            raise ValueError("source_geometry must have shape [B,S,G].")
        expected = (batch, seeds, modules)
        for name, value in (
            ("membership_logits", self.membership_logits),
            ("membership_probabilities", self.membership_probabilities),
            ("membership_hard", self.membership_hard),
            ("membership_straight_through", self.membership_straight_through),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape [B,S,M].")
        if self.membership_hard.dtype != torch.bool:
            raise TypeError("membership_hard must be boolean.")
        if self.member_indices.ndim != 3 or self.member_indices.shape[:2] != (batch, seeds):
            raise ValueError("member_indices must have shape [B,S,K].")
        if self.member_valid.shape != self.member_indices.shape or self.member_valid.dtype != torch.bool:
            raise ValueError("member_valid must be boolean and match member_indices.")
        object.__setattr__(self, "_tensor_versions", tensor_versions(*self._owned_tensors()))

    def _owned_tensors(self) -> tuple[Tensor, ...]:
        return (
            self.seed_ids,
            self.seed_valid,
            self.module_present,
            self.seed_coordinates,
            self.seed_characteristic_lengths,
            self.source_geometry,
            self.membership_logits,
            self.membership_probabilities,
            self.membership_hard,
            self.membership_straight_through,
            self.member_indices,
            self.member_valid,
        )

    def assert_current(self) -> None:
        """Reject in-place changes to sampled membership and native correspondence."""
        if tensor_versions(*self._owned_tensors()) != self._tensor_versions:
            raise ValueError("Budgeted native graph membership, geometry, IDs, or padded lists changed in place.")

    @property
    def deterministic_hard(self) -> Tensor:
        """Threshold logits with exact self/presence handling for deployment."""
        modules = self.membership_logits.shape[-1]
        if modules == 0:
            return torch.zeros_like(self.membership_hard)
        allowed = self.seed_valid[:, :, None] & self.module_present[:, None, :]
        diagonal = torch.eye(modules, dtype=torch.bool, device=self.membership_logits.device).unsqueeze(0)
        return ((self.membership_logits >= 0.0) & allowed & ~diagonal) | (
            allowed & diagonal
        )


@dataclass(frozen=True)
class ExpectedWork:
    """Analytical final unique-row work and both useful normalization conventions."""

    expected_support: Tensor  # [B,Q,M]
    expected_rows_per_query: Tensor  # [B,Q]
    floor_rows_per_query: Tensor  # [B,Q], max(1, direct-protection count) when M>0
    removable_rows_per_query: Tensor  # [B,Q]
    expected_retention_per_query: Tensor  # retained fraction of removable rows
    case_removable_retention: Tensor  # [B], equal-case aggregation happens outside this helper
    case_native_fraction: Tensor  # [B], all native rows including the protected floor
    case_expected_rows: Tensor  # [B]
    case_possible_rows: Tensor  # [B]
    case_floor_rows: Tensor  # [B]
    case_removable_rows: Tensor  # [B]

    def target_case_native_fraction(self, removable_retention: float | Tensor) -> Tensor:
        """Convert a removable-row target to the matching request-weighted C target."""
        rho = torch.as_tensor(removable_retention, dtype=self.case_expected_rows.dtype,
                              device=self.case_expected_rows.device)
        _check_tensor(torch.isfinite(rho).all() & (rho >= 0.0).all() & (rho <= 1.0).all(),
                      "removable_retention must lie in [0,1].")
        target_rows = self.case_floor_rows + rho * self.case_removable_rows
        return torch.where(self.case_possible_rows > 0.0,
                           target_rows / self.case_possible_rows.clamp_min(1.0),
                           torch.zeros_like(target_rows))


def calibrate_membership_bias(
    mm_raw_scores: Tensor,
    module_present: Tensor,
    target: float = 0.98,
    *,
    head_logits: Tensor | None = None,
    scale: float | Tensor = 1.0,
    iterations: int = 64,
) -> Tensor:
    """Return an input-only scalar bias with target mean valid nonself inclusion.

    Native measures, labels, field targets and validation errors do not enter this
    calibration. The score rows are centered over physically present nonself
    modules, including a present record whose information measure is zero.
    """
    if mm_raw_scores.ndim != 4:
        raise ValueError("mm_raw_scores must have shape [B,H,M,M].")
    batch, heads, seeds, modules = mm_raw_scores.shape
    if seeds != modules or module_present.shape != (batch, modules) or module_present.dtype != torch.bool:
        raise ValueError("module_present must be boolean [B,M] on the square module route.")
    if not 0.0 < float(target) < 1.0:
        raise ValueError("target must lie strictly between zero and one.")
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if head_logits is None:
        weights = mm_raw_scores.new_full((heads,), 1.0 / heads)
    else:
        if head_logits.shape != (heads,):
            raise ValueError("head_logits must have shape [H].")
        weights = torch.softmax(head_logits.to(mm_raw_scores), dim=0)
    eye = torch.eye(modules, device=mm_raw_scores.device, dtype=torch.bool).unsqueeze(0)
    allowed = module_present[:, :, None] & module_present[:, None, :] & ~eye
    relative = _relative_membership_scores(mm_raw_scores, allowed)
    logits_without_bias = (relative * weights[None, :, None, None]).sum(dim=1) * torch.as_tensor(
        scale, dtype=mm_raw_scores.dtype, device=mm_raw_scores.device
    )
    eligible = allowed
    if not bool(eligible.any()):
        raise ValueError("Bias calibration requires at least one present nonself MM pair.")
    scores = logits_without_bias[eligible]
    target_tensor = scores.new_tensor(float(target))
    span = scores.detach().abs().amax() + 32.0
    lower = -span
    upper = span
    for _ in range(iterations):
        midpoint = (lower + upper) * 0.5
        mean_probability = torch.sigmoid(scores + midpoint).mean()
        move_lower = mean_probability < target_tensor
        lower = torch.where(move_lower, midpoint, lower)
        upper = torch.where(move_lower, upper, midpoint)
    return ((lower + upper) * 0.5).detach()


def build_budgeted_native_graph(
    mm_raw_scores: Tensor,
    module_present: Tensor,
    module_ids: Tensor,
    module_coordinates: Tensor,
    parameters: BudgetedGraphParameters,
    *,
    module_characteristic_lengths: Tensor | None = None,
    temperature: float = 1.0,
    logistic_noise: Tensor | None = None,
    noise_scale: float = 1.0,
    deterministic: bool = False,
    straight_through: bool = False,
    generator: torch.Generator | None = None,
) -> BudgetedNativeGraph:
    """Build one shared binary membership matrix for every native module seed.

    The caller prepares this graph once per scene and reuses its sample across all
    query chunks and auxiliary calls. Supplying `logistic_noise` lets the owner key
    randomness to a stable scene/request namespace rather than a global RNG stream.
    """
    if mm_raw_scores.ndim != 4:
        raise ValueError("mm_raw_scores must have shape [B,H,M,M].")
    batch, heads, seeds, modules = mm_raw_scores.shape
    if seeds != modules:
        raise ValueError("The module-seeded graph requires a square [M,M] MM score map.")
    _validate_binary_mask(module_present, (batch, modules), "module_present")
    if module_ids.shape != (batch, modules) or module_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("module_ids must be integer [B,M].")
    if module_coordinates.ndim != 3 or module_coordinates.shape[:2] != (batch, modules):
        raise ValueError("module_coordinates must have shape [B,M,C].")
    if module_characteristic_lengths is None:
        # Geometry remains source-native; callers with length scales provide them
        # to complete canonical tie keys without changing route scores.
        module_characteristic_lengths = torch.ones_like(module_coordinates)
    if module_characteristic_lengths.shape != module_coordinates.shape:
        raise ValueError("module_characteristic_lengths must match [B,M,C].")
    if parameters.head_logits.shape != (heads,):
        raise ValueError("Membership parameters and MM score head counts differ.")
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and positive.")
    if noise_scale < 0.0 or not math.isfinite(float(noise_scale)):
        raise ValueError("noise_scale must be finite and nonnegative.")
    allowed = module_present[:, :, None] & module_present[:, None, :]
    if modules:
        allowed = allowed & ~torch.eye(modules, device=mm_raw_scores.device, dtype=torch.bool).unsqueeze(0)
    relative = _relative_membership_scores(mm_raw_scores, allowed)
    weights = parameters.head_weights.to(mm_raw_scores)
    centered = (relative * weights[None, :, None, None]).sum(dim=1)
    logits = parameters.scale.to(mm_raw_scores) * centered + parameters.bias.to(mm_raw_scores)
    logits = torch.where(allowed, logits, torch.zeros_like(logits))
    probabilities = torch.sigmoid(logits) * allowed.to(logits.dtype)
    if modules:
        diagonal = torch.eye(modules, device=mm_raw_scores.device, dtype=torch.bool).unsqueeze(0)
        probabilities = torch.where(diagonal & module_present[:, :, None], torch.ones_like(probabilities), probabilities)
    if deterministic:
        hard = logits >= 0.0
    else:
        if logistic_noise is None:
            tiny = torch.finfo(mm_raw_scores.dtype).eps
            uniform = torch.rand(
                (batch, seeds, modules),
                dtype=mm_raw_scores.dtype,
                device=mm_raw_scores.device,
                generator=generator,
            ).clamp(min=tiny, max=1.0 - tiny)
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        if logistic_noise.shape != logits.shape:
            raise ValueError("logistic_noise must have shape [B,M,M].")
        hard = logits + float(noise_scale) * logistic_noise >= 0.0
    hard = hard & allowed
    if modules:
        diagonal = torch.eye(modules, device=mm_raw_scores.device, dtype=torch.bool).unsqueeze(0)
        hard = hard | (diagonal & module_present[:, :, None])
    relaxed = torch.sigmoid((logits + (0.0 if deterministic else float(noise_scale) * logistic_noise)) / temperature)
    relaxed = relaxed * allowed.to(relaxed.dtype)
    if modules:
        relaxed = torch.where(diagonal & module_present[:, :, None], torch.ones_like(relaxed), relaxed)
    membership_st = hard.to(relaxed.dtype) + (relaxed - relaxed.detach()) if straight_through else hard.to(relaxed.dtype)

    # Stable source-axis order within each group. K=M is an explicit padded shape;
    # downstream packers may trim the all-false suffix once at request scope.
    member_indices = torch.arange(modules, device=mm_raw_scores.device, dtype=torch.long)
    member_indices = member_indices.view(1, 1, modules).expand(batch, seeds, modules)
    sort_key = (~hard).to(torch.int8)
    order = torch.argsort(sort_key, dim=-1, descending=False, stable=True)
    member_indices = torch.gather(member_indices, -1, order)
    member_counts = hard.sum(dim=-1)
    member_valid = torch.arange(modules, device=mm_raw_scores.device).view(1, 1, modules) < member_counts.unsqueeze(-1)

    return BudgetedNativeGraph(
        seed_ids=module_ids,
        seed_valid=module_present,
        module_present=module_present,
        seed_coordinates=module_coordinates,
        seed_characteristic_lengths=module_characteristic_lengths,
        membership_logits=logits,
        membership_probabilities=probabilities,
        membership_hard=hard,
        membership_straight_through=membership_st,
        member_indices=member_indices,
        member_valid=member_valid,
        source_geometry=torch.cat((module_coordinates, module_characteristic_lengths), dim=-1),
    )


def final_support(
    dispatch_seed: Tensor,
    graph: BudgetedNativeGraph,
    protected_mask: Tensor,
    *,
    query_mask: Tensor | None = None,
) -> Tensor:
    """Union one dispatched group's binary members with protected direct IDs."""
    graph.assert_current()
    if dispatch_seed.ndim != 2 or dispatch_seed.shape[0] != graph.seed_ids.shape[0]:
        raise ValueError("dispatch_seed must have shape [B,Q] and carry module-axis seed indices.")
    batch, queries = dispatch_seed.shape
    modules = graph.module_present.shape[-1]
    _validate_binary_mask(protected_mask, (batch, queries, modules), "protected_mask")
    if query_mask is None:
        query_mask = torch.ones((batch, queries), dtype=torch.bool, device=dispatch_seed.device)
    _validate_binary_mask(query_mask, (batch, queries), "query_mask")
    _check_tensor((~protected_mask | graph.module_present[:, None, :]).all(), "Protected records must be physically present.")
    has_seed = dispatch_seed >= 0
    _check_tensor((dispatch_seed >= -1).all(), "dispatch_seed may use only -1 as its empty sentinel.")
    _check_tensor((~has_seed | (dispatch_seed < graph.seed_ids.shape[-1])).all(), "dispatch_seed index is out of range.")
    if modules == 0:
        return torch.zeros((batch, queries, 0), dtype=torch.bool, device=dispatch_seed.device)
    safe_seed = dispatch_seed.clamp(min=0, max=modules - 1)
    batch_index = torch.arange(batch, device=dispatch_seed.device)[:, None]
    selected = graph.membership_hard[batch_index, safe_seed]
    selected = selected & has_seed[..., None] & query_mask[..., None]
    protected = protected_mask & query_mask[..., None]
    return (selected | protected) & graph.module_present[:, None, :]


def dense_st_gate(
    dispatch_straight_through: Tensor,
    membership_straight_through: Tensor,
    protected_mask: Tensor,
    *,
    module_present: Tensor | None = None,
    query_mask: Tensor | None = None,
) -> Tensor:
    """Compose a dense router estimator with exact binary hard-forward support."""
    if dispatch_straight_through.ndim != 3 or membership_straight_through.ndim != 3:
        raise ValueError("Dispatch and membership ST tensors must have shapes [B,Q,S] and [B,S,M].")
    batch, queries, seeds = dispatch_straight_through.shape
    if membership_straight_through.shape[:2] != (batch, seeds):
        raise ValueError("Dispatch and membership ST seed axes must match.")
    modules = membership_straight_through.shape[-1]
    _validate_binary_mask(protected_mask, (batch, queries, modules), "protected_mask")
    if module_present is not None:
        _validate_binary_mask(module_present, (batch, modules), "module_present")
    if query_mask is not None:
        _validate_binary_mask(query_mask, (batch, queries), "query_mask")
    support = torch.matmul(dispatch_straight_through, membership_straight_through)
    gates = protected_mask.to(support.dtype) + (1.0 - protected_mask.to(support.dtype)) * support
    if module_present is not None:
        gates = gates * module_present[:, None, :].to(gates.dtype)
    if query_mask is not None:
        gates = gates * query_mask[:, :, None].to(gates.dtype)
    return gates


def expected_final_work(
    dispatch_probabilities: Tensor,
    membership_probabilities: Tensor,
    protected_mask: Tensor,
    query_mask: Tensor,
    module_present: Tensor,
) -> ExpectedWork:
    """Compute the exact marginal unique-row count for the declared stochastic model.

    The result adds protections once, rather than OR-product saturation across all
    possible seeds. Both removable-row retention and all-native-row fraction are
    returned so the budget constraint can use the requested denominator explicitly.
    """
    if dispatch_probabilities.ndim != 3 or membership_probabilities.ndim != 3:
        raise ValueError("dispatch probabilities [B,Q,S] and memberships [B,S,M] are required.")
    batch, queries, seeds = dispatch_probabilities.shape
    if membership_probabilities.shape[:2] != (batch, seeds):
        raise ValueError("Dispatch and membership probability axes must match.")
    modules = membership_probabilities.shape[-1]
    _validate_binary_mask(module_present, (batch, modules), "module_present")
    _validate_binary_mask(protected_mask, (batch, queries, modules), "protected_mask")
    _validate_binary_mask(query_mask, (batch, queries), "query_mask")
    if dispatch_probabilities.device != membership_probabilities.device:
        raise ValueError("Dispatch and membership probabilities must share a device.")
    _check_tensor(torch.isfinite(dispatch_probabilities).all(), "Dispatch probabilities must be finite.")
    _check_tensor(torch.isfinite(membership_probabilities).all(), "Membership probabilities must be finite.")
    _check_tensor(
        ((dispatch_probabilities >= 0.0) & (dispatch_probabilities <= 1.0)).all(),
        "Dispatch probabilities must lie in [0,1].",
    )
    _check_tensor(
        ((membership_probabilities >= 0.0) & (membership_probabilities <= 1.0)).all(),
        "Membership probabilities must lie in [0,1].",
    )
    _check_tensor((~protected_mask | module_present[:, None, :]).all(), "Protected records must be physically present.")
    query_mass = dispatch_probabilities.sum(dim=-1)
    scene_has_seed = module_present.any(dim=-1, keepdim=True)
    expected_mass = query_mask & scene_has_seed
    no_seed_query = query_mask & ~scene_has_seed
    mass_error = torch.where(
        expected_mass,
        (query_mass - 1.0).abs(),
        torch.where(no_seed_query, query_mass.abs(), torch.zeros_like(query_mass)),
    )
    _check_tensor((mass_error <= 1.0e-5).all(), "Dispatch probabilities must sum to one on nonempty valid queries.")
    selected_probability = torch.einsum("bqs,bsm->bqm", dispatch_probabilities, membership_probabilities)
    eligible_queries = query_mask[:, :, None]
    protected = protected_mask & eligible_queries
    expected_support = (protected.to(selected_probability.dtype) + (1.0 - protected.to(selected_probability.dtype)) * selected_probability)
    expected_support = expected_support * module_present[:, None, :].to(expected_support.dtype)
    expected_support = expected_support * eligible_queries.to(expected_support.dtype)
    expected_rows = expected_support.sum(dim=-1)
    present_count = module_present.sum(dim=-1).to(expected_rows.dtype)
    protection_count = protected_mask.sum(dim=-1).to(expected_rows.dtype)
    # Floor is query-local because the protected set varies by receiver.
    floor_each = torch.minimum(torch.maximum(torch.ones_like(protection_count), protection_count), present_count[:, None])
    floor_each = floor_each * query_mask.to(floor_each.dtype)
    removable = (present_count[:, None] - floor_each).clamp_min(0.0) * query_mask.to(present_count.dtype)
    excess = expected_rows - floor_each
    retention = torch.where(removable > 0.0, excess / removable.clamp_min(1.0), torch.zeros_like(excess))
    retention = retention * query_mask.to(retention.dtype)
    per_case_removable = removable.sum(dim=-1)
    case_removable_retention = torch.where(
        per_case_removable > 0.0,
        (excess * query_mask.to(excess.dtype)).sum(dim=-1) / per_case_removable.clamp_min(1.0),
        torch.zeros_like(per_case_removable),
    )
    possible = present_count * query_mask.sum(dim=-1).to(present_count.dtype)
    case_rows = expected_rows.sum(dim=-1)
    case_floor_rows = floor_each.sum(dim=-1)
    case_native_fraction = torch.where(possible > 0.0, case_rows / possible.clamp_min(1.0), torch.zeros_like(possible))
    return ExpectedWork(
        expected_support=expected_support,
        expected_rows_per_query=expected_rows,
        floor_rows_per_query=floor_each,
        removable_rows_per_query=removable,
        expected_retention_per_query=retention,
        case_removable_retention=case_removable_retention,
        case_native_fraction=case_native_fraction,
        case_expected_rows=case_rows,
        case_possible_rows=possible,
        case_floor_rows=case_floor_rows,
        case_removable_rows=per_case_removable,
    )


__all__ = [
    "BudgetedGraphParameters",
    "BudgetedNativeGraph",
    "ExpectedWork",
    "build_budgeted_native_graph",
    "calibrate_membership_bias",
    "dense_st_gate",
    "expected_final_work",
    "final_support",
]
