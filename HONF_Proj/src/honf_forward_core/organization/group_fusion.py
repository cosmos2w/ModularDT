"""Differentiable, reversible coalescence for Run 1503-v2.

Only the small group-control bank is fused. Physical source rows and values
stay fine grained. A GroupFusionPlan is prepared once per case and phase and
then reused for every receiver chunk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

Tensor = torch.Tensor
ADMM_STEPS = 64  # retained only for the historical Run-1503-v2 solver
MAX_ADMM_ITERATIONS = 512
EPS_ABS = 1.0e-9
EPS_REL = 1.0e-8


def _check_batched(name: str, value: Tensor, rank: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}; got {value.ndim}.")


def effective_query_centers(
    module_centers: Tensor,
    environment_centers: Tensor,
    module_mass: Tensor,
    environment_mass: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply the sparse-incidence router's type-specific center fallback."""

    _check_batched("module_centers", module_centers, 3)
    _check_batched("environment_centers", environment_centers, 3)
    if module_centers.shape != environment_centers.shape:
        raise ValueError("module_centers and environment_centers must align.")
    expected = module_centers.shape[:2]
    if tuple(module_mass.shape) != tuple(expected) or tuple(environment_mass.shape) != tuple(expected):
        raise ValueError("source masses must align with the center banks.")
    module_query_centers = torch.where(
        (module_mass > 0.0)[..., None], module_centers, environment_centers
    )
    environment_query_centers = torch.where(
        (environment_mass > 0.0)[..., None], environment_centers, module_centers
    )
    return module_query_centers, environment_query_centers


def build_group_descriptors(
    query_keys: Tensor,
    module_query_centers: Tensor,
    environment_query_centers: Tensor,
    geometry_scale: Tensor | float,
) -> Tensor:
    """Encode parent access functions as a batched [B,K,D+2d] descriptor."""

    _check_batched("query_keys", query_keys, 3)
    _check_batched("module_query_centers", module_query_centers, 3)
    _check_batched("environment_query_centers", environment_query_centers, 3)
    if module_query_centers.shape != environment_query_centers.shape:
        raise ValueError("the two query-center banks must align.")
    batch, groups, control_dim = query_keys.shape
    if tuple(module_query_centers.shape[:2]) != (batch, groups):
        raise ValueError("query keys and query-center banks must align along [B,K].")
    if query_keys.device != module_query_centers.device or query_keys.dtype != module_query_centers.dtype:
        raise ValueError("query keys and query-center banks must share device and dtype.")
    scale = torch.as_tensor(geometry_scale, device=query_keys.device, dtype=query_keys.dtype)
    if scale.ndim == 0:
        scale = scale.expand(batch)
    elif scale.ndim == 2 and int(scale.shape[-1]) == 1:
        scale = scale[:, 0]
    if tuple(scale.shape) != (batch,):
        raise ValueError(f"geometry_scale must be scalar or have shape [{batch}].")
    return torch.cat(
        [
            query_keys / math.sqrt(control_dim),
            module_query_centers / (math.sqrt(2.0) * scale[:, None, None]),
            environment_query_centers / (math.sqrt(2.0) * scale[:, None, None]),
        ],
        dim=-1,
    )


def fusion_schedule(epoch: float | Tensor) -> float | Tensor:
    """The fixed Run-1503-v2 continuation for fusion strength eta."""

    if isinstance(epoch, torch.Tensor):
        return 0.5 * (epoch / 150.0).clamp(min=0.0, max=1.0)
    return 0.5 * min(max(float(epoch), 0.0) / 150.0, 1.0)


def source_footprint_weights(
    module_incidence: Tensor,
    environment_incidence: Tensor,
    module_measure: Tensor,
    environment_measure: Tensor,
    active_module_mask: Tensor,
    descriptors: Tensor,
    active_groups: Tensor,
) -> Tensor:
    """Build normalized fusion weights from balanced source footprints."""

    _check_batched("module_incidence", module_incidence, 3)
    _check_batched("environment_incidence", environment_incidence, 3)
    _check_batched("descriptors", descriptors, 3)
    batch, _, groups = module_incidence.shape
    if int(environment_incidence.shape[0]) != batch or int(environment_incidence.shape[-1]) != groups:
        raise ValueError("module and environment incidence must share [B,K].")
    if tuple(descriptors.shape[:2]) != (batch, groups):
        raise ValueError("descriptors must align with the incidence group bank.")
    if tuple(module_measure.shape) != tuple(module_incidence.shape[:2]):
        raise ValueError("module_measure must align with module incidence.")
    if tuple(environment_measure.shape) != tuple(environment_incidence.shape[:2]):
        raise ValueError("environment_measure must align with environment incidence.")
    if tuple(active_module_mask.shape) != tuple(module_measure.shape):
        raise ValueError("active_module_mask must align with module_measure.")
    if tuple(active_groups.shape) != (batch, groups):
        raise ValueError("active_groups must have shape [B,K].")
    if not descriptors.is_floating_point():
        raise TypeError("descriptors must be floating point.")

    active_groups = active_groups.to(device=descriptors.device, dtype=torch.bool)
    module_valid = active_module_mask.to(device=module_incidence.device, dtype=torch.bool)
    eps = torch.finfo(descriptors.dtype).eps
    weight_floor = eps * eps
    module_mass = 0.5 * module_measure.clamp_min(0.0)
    module_weight = torch.where(
        module_valid & (module_measure > 0.0),
        module_mass.clamp_min(weight_floor).sqrt(),
        torch.zeros_like(module_mass),
    )
    environment_mass = 0.5 * environment_measure.clamp_min(0.0)
    environment_weight = torch.where(
        environment_measure > 0.0,
        environment_mass.clamp_min(weight_floor).sqrt(),
        torch.zeros_like(environment_mass),
    )
    module_safe = torch.where(
        active_groups[:, None, :], module_incidence, torch.zeros_like(module_incidence)
    )
    environment_safe = torch.where(
        active_groups[:, None, :],
        environment_incidence,
        torch.zeros_like(environment_incidence),
    )
    footprint = torch.cat(
        [
            module_safe * module_weight[..., None],
            environment_safe * environment_weight[..., None],
        ],
        dim=1,
    )
    gram = torch.einsum("bsk,bsl->bkl", footprint, footprint)
    # sqrt has an infinite derivative at zero; masking its output afterward
    # still permits 0 * inf to become NaN during backward. Floor the squared
    # norm before sqrt so empty proposal columns have a finite derivative.
    norms = gram.diagonal(dim1=-2, dim2=-1).clamp_min(weight_floor).sqrt()
    norm_product = norms[..., :, None] * norms[..., None, :]
    cosine = (gram / norm_product).clamp(min=0.0, max=1.0)

    group_mask = active_groups.to(dtype=descriptors.dtype)
    count = group_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    safe_descriptors = torch.where(
        active_groups[..., None], descriptors, torch.zeros_like(descriptors)
    )
    center = (safe_descriptors * group_mask[..., None]).sum(dim=1, keepdim=True) / count[..., None]
    sigma2 = (
        (safe_descriptors - center).square().sum(dim=-1) * group_mask
    ).sum(dim=-1) / count.squeeze(-1)
    pair_delta = safe_descriptors[:, :, None, :] - safe_descriptors[:, None, :, :]
    distance2 = pair_delta.square().sum(dim=-1)
    kernel = torch.exp(-distance2 / (2.0 * sigma2.clamp_min(eps)[:, None, None]))
    pair_mask = active_groups[:, :, None] & active_groups[:, None, :]
    diagonal = torch.eye(groups, device=descriptors.device, dtype=torch.bool)[None, :, :]
    raw = cosine.square() * kernel * (pair_mask & ~diagonal).to(descriptors.dtype)
    maximum_row_sum = raw.sum(dim=-1).amax(dim=-1, keepdim=True)[..., None]
    denominator = torch.where(
        maximum_row_sum > 0.0, maximum_row_sum, torch.ones_like(maximum_row_sum)
    )
    return raw / denominator


def weighted_sparsemax(
    scores: Tensor,
    multiplicity: Tensor,
    valid: Tensor | None = None,
    *,
    dim: int = -1,
) -> tuple[Tensor, Tensor]:
    """Return (probability mass, per-constituent density) for weighted sparsemax."""

    if not isinstance(scores, torch.Tensor) or not scores.is_floating_point():
        raise TypeError("scores must be a floating-point tensor.")
    if scores.ndim < 1:
        raise ValueError("scores must have at least one dimension.")
    dim = dim % scores.ndim
    mult = torch.as_tensor(multiplicity, device=scores.device)
    while mult.ndim < scores.ndim:
        mult = mult.unsqueeze(-2)
    try:
        mult = torch.broadcast_to(mult, scores.shape).to(dtype=scores.dtype)
    except RuntimeError as exc:
        raise ValueError("multiplicity must broadcast to scores.") from exc
    if valid is None:
        mask = mult > 0.0
    else:
        mask = torch.as_tensor(valid, device=scores.device, dtype=torch.bool)
        while mask.ndim < scores.ndim:
            mask = mask.unsqueeze(-2)
        try:
            mask = torch.broadcast_to(mask, scores.shape) & (mult > 0.0)
        except RuntimeError as exc:
            raise ValueError("valid must broadcast to scores.") from exc
    masked = scores.masked_fill(~mask, -torch.inf)
    maximum = masked.amax(dim=dim, keepdim=True)
    shifted = torch.where(mask, scores - maximum, torch.full_like(scores, -torch.inf))
    sorted_scores, order = torch.sort(shifted, dim=dim, descending=True)
    sorted_mult = torch.gather(mult, dim, order)
    sorted_valid = torch.gather(mask, dim, order)
    safe_scores = torch.where(sorted_valid, sorted_scores, torch.zeros_like(sorted_scores))
    prefix_mass = sorted_mult.cumsum(dim=dim)
    prefix_score = (sorted_mult * safe_scores).cumsum(dim=dim)
    threshold_by_prefix = (prefix_score - 1.0) / prefix_mass.clamp_min(
        torch.finfo(scores.dtype).tiny
    )
    support_prefix = sorted_valid & (sorted_scores > threshold_by_prefix)
    support_size = support_prefix.to(torch.long).sum(dim=dim, keepdim=True).clamp_min(1)
    tau = torch.gather(threshold_by_prefix, dim, support_size - 1)
    safe_shifted = torch.where(mask, scores - maximum, torch.zeros_like(scores))
    density = torch.where(mask, torch.relu(safe_shifted - tau), torch.zeros_like(scores))
    mass = density * mult
    return mass, density


def _component_partition(
    active: Tensor,
    pair_fused: Tensor,
    pair_distance: Tensor,
    tolerance: Tensor,
) -> Tensor:
    """Tensor-only K-bounded components with a complete-pair diameter check."""

    batch, groups = active.shape
    indices = torch.arange(groups, device=active.device, dtype=torch.long)
    pair_active = active[:, :, None] & active[:, None, :]
    adjacency = pair_fused & pair_active
    sentinel = torch.full((batch, groups), groups, device=active.device, dtype=torch.long)
    labels = torch.where(active, indices[None, :], sentinel)
    for _ in range(groups):
        neighbor_labels = torch.where(
            adjacency, labels[:, None, :], sentinel[:, None, :]
        ).amin(dim=-1)
        labels = torch.minimum(labels, neighbor_labels)
        labels = torch.where(active, labels, sentinel)

    same_component = (
        active[:, :, None]
        & active[:, None, :]
        & (labels[:, :, None] == labels[:, None, :])
    )
    off_diagonal = ~torch.eye(groups, device=active.device, dtype=torch.bool)[None, :, :]
    bad_pair = same_component & off_diagonal & (
        (pair_distance > tolerance[:, None, None]) | ~pair_fused
    )
    bad_member = bad_pair.any(dim=-1)
    bad_component = (same_component & bad_member[:, None, :]).any(dim=-1)
    labels = torch.where(bad_component, indices[None, :], labels)

    roots = active & (labels == indices[None, :])
    root_rank = roots.to(torch.long).cumsum(dim=-1) - 1
    packed = torch.gather(root_rank, 1, labels.clamp(min=0, max=groups - 1))
    return torch.where(active, packed, torch.full_like(packed, -1))


@dataclass(frozen=True)
class GroupFusionPlan:
    """A case/phase-local quotient of provisional group access functions.

    ``build`` retains Run-1503-v2's fixed-step differentiable solver for
    historical checkpoint replay. ``build_converged`` is the v3 detached
    FP64 planner; its physical compact operator uses original parent tensors
    so singleton classes remain exactly parent-identical.
    """

    input_descriptors: Tensor
    fused_descriptors: Tensor
    class_index: Tensor
    multiplicity: Tensor
    valid: Tensor
    active: Tensor
    geometry_scale: Tensor
    parent_query_keys: Tensor | None
    parent_module_query_centers: Tensor | None
    parent_environment_query_centers: Tensor | None
    control_dim: int
    spatial_dim: int
    sigma: Tensor
    lambda_b: Tensor
    fusion_strength: Tensor
    merge_displacement: Tensor
    projection_displacement: Tensor
    primal_residual: Tensor
    dual_residual: Tensor
    pair_fused: Tensor
    solver_converged: Tensor | None = None
    solver_iterations: Tensor | None = None
    primal_tolerance: Tensor | None = None
    dual_tolerance: Tensor | None = None

    @classmethod
    def build(
        cls,
        descriptors: Tensor,
        footprint_weights: Tensor,
        active_groups: Tensor,
        strength: float | Tensor,
        *,
        control_dim: int | None = None,
        geometry_scale: Tensor | float = 1.0,
        merge_tolerance: float = 1.0e-5,
        parent_query_keys: Tensor | None = None,
        parent_module_query_centers: Tensor | None = None,
        parent_environment_query_centers: Tensor | None = None,
    ) -> GroupFusionPlan:
        """Solve the 64-step differentiable group-fusion proximal problem."""

        _check_batched("descriptors", descriptors, 3)
        batch, groups, descriptor_dim = descriptors.shape
        if tuple(footprint_weights.shape) != (batch, groups, groups):
            raise ValueError("footprint_weights must have shape [B,K,K].")
        if tuple(active_groups.shape) != (batch, groups):
            raise ValueError("active_groups must have shape [B,K].")
        if not descriptors.is_floating_point():
            raise TypeError("descriptors must be floating point.")
        if not math.isfinite(float(merge_tolerance)) or float(merge_tolerance) < 0.0:
            raise ValueError("merge_tolerance must be finite and nonnegative.")
        if control_dim is None:
            if descriptor_dim != 20:
                raise ValueError("control_dim is required unless using the D=16,d=2 descriptor.")
            control_dim = 16
        control_dim = int(control_dim)
        spatial_dim = (descriptor_dim - control_dim) // 2
        if control_dim <= 0 or spatial_dim <= 0 or control_dim + 2 * spatial_dim != descriptor_dim:
            raise ValueError("descriptor width must be control_dim + 2 * spatial_dim.")
        active = active_groups.to(device=descriptors.device, dtype=torch.bool)
        # Empty groups can carry non-finite fallback centers. They do not
        # participate in the proximal problem, so sanitize their descriptor
        # rows before variance, ADMM, and diagnostic calculations. Active
        # descriptors remain unchanged and fully differentiable.
        z = torch.where(active[..., None], descriptors, torch.zeros_like(descriptors))
        dtype = descriptors.dtype
        strength_tensor = torch.as_tensor(strength, device=descriptors.device, dtype=dtype)
        if strength_tensor.ndim == 0:
            strength_tensor = strength_tensor.expand(batch)
        if tuple(strength_tensor.shape) != (batch,):
            raise ValueError(f"strength must be scalar or have shape [{batch}].")
        if isinstance(strength, (int, float)) and float(strength) < 0.0:
            raise ValueError("strength must be nonnegative.")
        scale = torch.as_tensor(geometry_scale, device=descriptors.device, dtype=dtype)
        if scale.ndim == 0:
            scale = scale.expand(batch)
        elif scale.ndim == 2 and int(scale.shape[-1]) == 1:
            scale = scale[:, 0]
        if tuple(scale.shape) != (batch,):
            raise ValueError(f"geometry_scale must be scalar or have shape [{batch}].")
        if parent_query_keys is not None and tuple(parent_query_keys.shape) != (batch, groups, control_dim):
            raise ValueError("parent_query_keys must align with descriptor key columns.")
        expected_centers = (batch, groups, spatial_dim)
        if parent_module_query_centers is not None and tuple(parent_module_query_centers.shape) != expected_centers:
            raise ValueError("parent_module_query_centers must align with descriptor geometry columns.")
        if parent_environment_query_centers is not None and tuple(parent_environment_query_centers.shape) != expected_centers:
            raise ValueError("parent_environment_query_centers must align with descriptor geometry columns.")

        active_value = active.to(dtype)
        active_count = active_value.sum(dim=-1, keepdim=True).clamp_min(1.0)
        center = (z * active_value[..., None]).sum(dim=1, keepdim=True) / active_count[..., None]
        sigma2 = (
            (z - center).square().sum(dim=-1) * active_value
        ).sum(dim=-1) / active_count.squeeze(-1)
        nonzero_sigma = sigma2 > 0.0
        safe_sigma2 = torch.where(nonzero_sigma, sigma2, torch.ones_like(sigma2))
        sigma = torch.where(
            nonzero_sigma, safe_sigma2.sqrt(), torch.zeros_like(sigma2)
        )
        lambda_b = strength_tensor * sigma

        # The zero-strength reference returns before constructing the ADMM
        # graph. Active columns retain their parent order, including holes.
        if isinstance(strength, (int, float)) and float(strength) == 0.0:
            identity_index = torch.where(
                active,
                torch.arange(groups, device=descriptors.device)[None, :].expand(batch, -1),
                torch.full((batch, groups), -1, device=descriptors.device, dtype=torch.long),
            )
            zeros = torch.zeros((batch,), device=descriptors.device, dtype=dtype)
            return cls(
                input_descriptors=z,
                fused_descriptors=torch.where(active[..., None], descriptors, torch.zeros_like(descriptors)),
                class_index=identity_index,
                multiplicity=active.to(torch.long),
                valid=active,
                active=active,
                geometry_scale=scale,
                parent_query_keys=parent_query_keys,
                parent_module_query_centers=parent_module_query_centers,
                parent_environment_query_centers=parent_environment_query_centers,
                control_dim=control_dim,
                spatial_dim=spatial_dim,
                sigma=sigma,
                lambda_b=lambda_b,
                fusion_strength=strength_tensor,
                merge_displacement=torch.zeros((batch, groups), device=descriptors.device, dtype=dtype),
                projection_displacement=torch.zeros((batch, groups), device=descriptors.device, dtype=dtype),
                primal_residual=zeros,
                dual_residual=zeros,
                pair_fused=torch.zeros((batch, groups, groups), device=descriptors.device, dtype=torch.bool),
            )

        pair_i, pair_j = torch.triu_indices(groups, groups, offset=1, device=descriptors.device)
        edge_count = int(pair_i.numel())

        if edge_count:
            edge_weights = footprint_weights[:, pair_i, pair_j]
            edge_active = active[:, pair_i] & active[:, pair_j]
            edge_weights = edge_weights * edge_active.to(dtype)
            edge_incidence = torch.zeros((edge_count, groups), device=descriptors.device, dtype=dtype)
            edge_rows = torch.arange(edge_count, device=descriptors.device)
            edge_incidence[edge_rows, pair_i] = 1.0
            edge_incidence[edge_rows, pair_j] = -1.0
            penalty = 1.0 / float(groups)
            identity = torch.eye(groups, device=descriptors.device, dtype=dtype)
            inverse = 0.5 * identity + (0.5 / float(groups)) * torch.ones_like(identity)
            u = z
            v = u[:, pair_i, :] - u[:, pair_j, :]
            dual = torch.zeros_like(v)
            threshold = lambda_b[:, None, None] * edge_weights[..., None] / penalty
            v_previous = v
            for _ in range(ADMM_STEPS):
                v_previous = v
                rhs = z + penalty * torch.einsum(
                    "ek,bed->bkd", edge_incidence, v - dual
                )
                u = torch.einsum("kl,bld->bkd", inverse, rhs)
                difference = u[:, pair_i, :] - u[:, pair_j, :]
                shifted = difference + dual
                norm = torch.linalg.vector_norm(shifted, dim=-1, keepdim=True)
                # Evaluate the group shrinkage without dividing by a zero
                # norm. For an exactly zero penalty, preserve the identity
                # proximal map (including its derivative at a zero vector).
                positive_norm = norm > threshold
                safe_norm = torch.where(norm > 0.0, norm, torch.ones_like(norm))
                shrink = torch.where(
                    positive_norm,
                    (norm - threshold) / safe_norm,
                    torch.zeros_like(norm),
                )
                shrink = torch.where(
                    threshold == 0.0, torch.ones_like(shrink), shrink
                )
                prox = shifted * shrink
                v = prox
                dual = dual + difference - v
            u = torch.where(active[..., None], u, z)
            primal_edge = (u[:, pair_i, :] - u[:, pair_j, :]) - v
            primal_residual = torch.linalg.vector_norm(primal_edge, dim=(-2, -1))
            dual_residual = torch.linalg.vector_norm(
                penalty * torch.einsum("ek,bed->bkd", edge_incidence, v - v_previous),
                dim=(-2, -1),
            )
            pair_difference = u[:, :, None, :] - u[:, None, :, :]
            pair_distance = torch.linalg.vector_norm(pair_difference, dim=-1)
            edge_v_zero = (v == 0.0).all(dim=-1)
            tolerance = float(merge_tolerance) * torch.maximum(sigma, torch.ones_like(sigma))
            candidate = edge_active & (edge_weights > 0.0) & edge_v_zero & (
                pair_distance[:, pair_i, pair_j] <= tolerance[:, None]
            )
            pair_candidate = torch.zeros((batch, groups, groups), device=descriptors.device, dtype=torch.bool)
            pair_candidate[:, pair_i, pair_j] = candidate
            pair_candidate[:, pair_j, pair_i] = candidate
            pair_fused = pair_candidate
            class_index = _component_partition(active, pair_fused, pair_distance, tolerance)
        else:
            u = z
            primal_residual = torch.zeros((batch,), device=descriptors.device, dtype=dtype)
            dual_residual = torch.zeros_like(primal_residual)
            pair_distance = torch.zeros((batch, groups, groups), device=descriptors.device, dtype=dtype)
            pair_fused = torch.zeros((batch, groups, groups), device=descriptors.device, dtype=torch.bool)
            class_index = torch.where(
                active,
                torch.arange(groups, device=descriptors.device)[None, :].expand(batch, -1),
                torch.full((batch, groups), -1, device=descriptors.device, dtype=torch.long),
            )

        identity_index = torch.where(
            active,
            torch.arange(groups, device=descriptors.device)[None, :].expand(batch, -1),
            torch.full((batch, groups), -1, device=descriptors.device, dtype=torch.long),
        )
        zero_strength = strength_tensor == 0.0
        class_index = torch.where(zero_strength[:, None], identity_index, class_index)
        u = torch.where(zero_strength[:, None, None], z, u)
        identity_pairs = torch.eye(groups, device=descriptors.device, dtype=torch.bool)[None, :, :].expand(batch, -1, -1)
        pair_fused = torch.where(zero_strength[:, None, None], identity_pairs, pair_fused)
        primal_residual = torch.where(zero_strength, torch.zeros_like(primal_residual), primal_residual)
        dual_residual = torch.where(zero_strength, torch.zeros_like(dual_residual), dual_residual)

        safe_index = class_index.clamp_min(0)
        membership = torch.nn.functional.one_hot(safe_index, num_classes=groups).to(dtype)
        membership = membership * active[..., None].to(dtype)
        multiplicity = membership.sum(dim=1).to(torch.long)
        valid = multiplicity > 0
        fused_descriptors = torch.einsum("bkr,bkd->brd", membership, u)
        fused_descriptors = fused_descriptors / multiplicity.clamp_min(1).to(dtype)[..., None]
        fused_descriptors = fused_descriptors * valid[..., None].to(dtype)
        projected = torch.einsum("bkr,brd->bkd", membership, fused_descriptors)
        projection_displacement = torch.linalg.vector_norm(u - projected, dim=-1)
        merge_displacement = torch.linalg.vector_norm(u - z, dim=-1)
        return cls(
            input_descriptors=z,
            fused_descriptors=fused_descriptors,
            class_index=class_index,
            multiplicity=multiplicity,
            valid=valid,
            active=active,
            geometry_scale=scale,
            parent_query_keys=parent_query_keys,
            parent_module_query_centers=parent_module_query_centers,
            parent_environment_query_centers=parent_environment_query_centers,
            control_dim=control_dim,
            spatial_dim=spatial_dim,
            sigma=sigma,
            lambda_b=lambda_b,
            fusion_strength=strength_tensor,
            merge_displacement=merge_displacement,
            projection_displacement=projection_displacement,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            pair_fused=pair_fused,
        )

    @classmethod
    def build_converged(
        cls,
        descriptors: Tensor,
        footprint_weights: Tensor,
        active_groups: Tensor,
        strength: float | Tensor,
        *,
        control_dim: int | None = None,
        geometry_scale: Tensor | float = 1.0,
        merge_tolerance: float = 1.0e-5,
        max_iterations: int = MAX_ADMM_ITERATIONS,
        eps_abs: float = EPS_ABS,
        eps_rel: float = EPS_REL,
        parent_query_keys: Tensor | None = None,
        parent_module_query_centers: Tensor | None = None,
        parent_environment_query_centers: Tensor | None = None,
    ) -> GroupFusionPlan:
        """Plan a residual-converged convex fusion partition in detached FP64."""

        _check_batched("descriptors", descriptors, 3)
        batch, groups, descriptor_dim = descriptors.shape
        if tuple(footprint_weights.shape) != (batch, groups, groups):
            raise ValueError("footprint_weights must have shape [B,K,K].")
        if tuple(active_groups.shape) != (batch, groups):
            raise ValueError("active_groups must have shape [B,K].")
        if groups <= 0:
            raise ValueError("descriptors must contain at least one proposal slot.")
        if not descriptors.is_floating_point():
            raise TypeError("descriptors must be floating point.")
        if not math.isfinite(float(merge_tolerance)) or float(merge_tolerance) < 0.0:
            raise ValueError("merge_tolerance must be finite and nonnegative.")
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        if not math.isfinite(float(eps_abs)) or float(eps_abs) < 0.0:
            raise ValueError("eps_abs must be finite and nonnegative.")
        if not math.isfinite(float(eps_rel)) or float(eps_rel) < 0.0:
            raise ValueError("eps_rel must be finite and nonnegative.")
        if control_dim is None:
            if descriptor_dim != 20:
                raise ValueError("control_dim is required unless using the D=16,d=2 descriptor.")
            control_dim = 16
        control_dim = int(control_dim)
        spatial_dim = (descriptor_dim - control_dim) // 2
        if control_dim <= 0 or spatial_dim <= 0 or control_dim + 2 * spatial_dim != descriptor_dim:
            raise ValueError("descriptor width must be control_dim + 2 * spatial_dim.")
        active = active_groups.to(device=descriptors.device, dtype=torch.bool)
        dtype = descriptors.dtype
        # Empty proposals can contain non-finite fallback centers. They are
        # outside the objective, while active parent descriptors remain live
        # for the compact physical operator.
        z = torch.where(active[..., None], descriptors, torch.zeros_like(descriptors))
        strength_tensor = torch.as_tensor(strength, device=descriptors.device, dtype=dtype)
        if strength_tensor.ndim == 0:
            strength_tensor = strength_tensor.expand(batch)
        if tuple(strength_tensor.shape) != (batch,):
            raise ValueError(f"strength must be scalar or have shape [{batch}].")
        if not torch.isfinite(strength_tensor).all() or (strength_tensor < 0.0).any():
            raise ValueError("strength must be finite and nonnegative.")
        scale = torch.as_tensor(geometry_scale, device=descriptors.device, dtype=dtype)
        if scale.ndim == 0:
            scale = scale.expand(batch)
        elif scale.ndim == 2 and int(scale.shape[-1]) == 1:
            scale = scale[:, 0]
        if tuple(scale.shape) != (batch,):
            raise ValueError(f"geometry_scale must be scalar or have shape [{batch}].")
        if parent_query_keys is not None and tuple(parent_query_keys.shape) != (batch, groups, control_dim):
            raise ValueError("parent_query_keys must align with descriptor key columns.")
        expected_centers = (batch, groups, spatial_dim)
        if parent_module_query_centers is not None and tuple(parent_module_query_centers.shape) != expected_centers:
            raise ValueError("parent_module_query_centers must align with descriptor geometry columns.")
        if parent_environment_query_centers is not None and tuple(parent_environment_query_centers.shape) != expected_centers:
            raise ValueError("parent_environment_query_centers must align with descriptor geometry columns.")

        # The numerical solver is deliberately outside autograd. All state,
        # objective scales, and residual arithmetic are FP64 on the input
        # device; the neural model and returned parent tensors keep their dtype.
        with torch.no_grad():
            z64 = z.detach().to(dtype=torch.float64)
            active64 = active.detach()
            strength64 = strength_tensor.detach().to(dtype=torch.float64)
            active_value = active64.to(torch.float64)
            active_count = active_value.sum(dim=-1, keepdim=True).clamp_min(1.0)
            center = (z64 * active_value[..., None]).sum(dim=1, keepdim=True) / active_count[..., None]
            sigma2 = (
                (z64 - center).square().sum(dim=-1) * active_value
            ).sum(dim=-1) / active_count.squeeze(-1)
            sigma = sigma2.clamp_min(0.0).sqrt()
            lambda_b = strength64 * sigma

            pair_i, pair_j = torch.triu_indices(
                groups, groups, offset=1, device=descriptors.device
            )
            edge_count = int(pair_i.numel())
            edge_active = active64[:, pair_i] & active64[:, pair_j]
            if edge_count:
                edge_weights = torch.where(
                    edge_active,
                    footprint_weights.detach()[:, pair_i, pair_j].to(torch.float64),
                    torch.zeros((batch, edge_count), device=descriptors.device, dtype=torch.float64),
                )
                edge_incidence = torch.zeros(
                    (edge_count, groups), device=descriptors.device, dtype=torch.float64
                )
                edge_rows = torch.arange(edge_count, device=descriptors.device)
                edge_incidence[edge_rows, pair_i] = 1.0
                edge_incidence[edge_rows, pair_j] = -1.0
            else:
                edge_weights = torch.zeros((batch, 0), device=descriptors.device, dtype=torch.float64)
                edge_incidence = torch.zeros((0, groups), device=descriptors.device, dtype=torch.float64)

            penalty = 1.0 / float(groups)
            identity = torch.eye(groups, device=descriptors.device, dtype=torch.float64)
            inverse = 0.5 * identity + (0.5 / float(groups)) * torch.ones_like(identity)
            u = z64.clone()
            if edge_count:
                v = u[:, pair_i, :] - u[:, pair_j, :]
            else:
                v = torch.zeros((batch, 0, descriptor_dim), device=descriptors.device, dtype=torch.float64)
            dual = torch.zeros_like(v)
            threshold = lambda_b[:, None, None] * edge_weights[..., None] / penalty
            has_penalty_edge = (edge_weights > 0.0).any(dim=-1)
            solver_converged = (
                (lambda_b <= 0.0)
                | (active_count.squeeze(-1) <= 1.0)
                | ~has_penalty_edge
            )
            solver_iterations = torch.zeros((batch,), device=descriptors.device, dtype=torch.long)
            primal_residual = torch.zeros((batch,), device=descriptors.device, dtype=torch.float64)
            dual_residual = torch.zeros_like(primal_residual)

            if edge_count:
                difference = u[:, pair_i, :] - u[:, pair_j, :]
                norm_difference = torch.linalg.vector_norm(difference, dim=(-2, -1))
                norm_v = torch.linalg.vector_norm(v, dim=(-2, -1))
                primal_tolerance = (
                    math.sqrt(edge_count * descriptor_dim) * float(eps_abs)
                    + float(eps_rel) * torch.maximum(norm_difference, norm_v)
                )
            else:
                primal_tolerance = torch.zeros_like(primal_residual)
            dual_scale = penalty * torch.einsum("ek,bed->bkd", edge_incidence, dual)
            dual_tolerance = (
                math.sqrt(groups * descriptor_dim) * float(eps_abs)
                + float(eps_rel) * torch.linalg.vector_norm(dual_scale, dim=(-2, -1))
            )
            needs_solve = ~solver_converged

            for iteration in range(1, int(max_iterations) + 1):
                if not bool(needs_solve.any()):
                    break
                old_v = v
                rhs = z64 + penalty * torch.einsum(
                    "ek,bed->bkd", edge_incidence, v - dual
                )
                candidate_u = torch.einsum("kl,bld->bkd", inverse, rhs)
                candidate_difference = candidate_u[:, pair_i, :] - candidate_u[:, pair_j, :]
                shifted = candidate_difference + dual
                norm = torch.linalg.vector_norm(shifted, dim=-1, keepdim=True)
                safe_norm = torch.where(norm > 0.0, norm, torch.ones_like(norm))
                shrink = torch.where(
                    norm > threshold,
                    (norm - threshold) / safe_norm,
                    torch.zeros_like(norm),
                )
                candidate_v = shifted * shrink
                candidate_dual = dual + candidate_difference - candidate_v

                row_mask = needs_solve[:, None, None]
                u = torch.where(row_mask, candidate_u, u)
                v = torch.where(row_mask, candidate_v, v)
                dual = torch.where(row_mask, candidate_dual, dual)
                updated_difference = u[:, pair_i, :] - u[:, pair_j, :]
                primal_now = torch.linalg.vector_norm(updated_difference - v, dim=(-2, -1))
                dual_now = torch.linalg.vector_norm(
                    penalty * torch.einsum("ek,bed->bkd", edge_incidence, v - old_v),
                    dim=(-2, -1),
                )
                dual_scale = penalty * torch.einsum("ek,bed->bkd", edge_incidence, dual)
                primal_tol_now = (
                    math.sqrt(edge_count * descriptor_dim) * float(eps_abs)
                    + float(eps_rel)
                    * torch.maximum(
                        torch.linalg.vector_norm(updated_difference, dim=(-2, -1)),
                        torch.linalg.vector_norm(v, dim=(-2, -1)),
                    )
                )
                dual_tol_now = (
                    math.sqrt(groups * descriptor_dim) * float(eps_abs)
                    + float(eps_rel) * torch.linalg.vector_norm(dual_scale, dim=(-2, -1))
                )
                primal_residual = torch.where(needs_solve, primal_now, primal_residual)
                dual_residual = torch.where(needs_solve, dual_now, dual_residual)
                primal_tolerance = torch.where(needs_solve, primal_tol_now, primal_tolerance)
                dual_tolerance = torch.where(needs_solve, dual_tol_now, dual_tolerance)
                solver_iterations = solver_iterations + needs_solve.to(torch.long)
                newly_converged = (
                    (primal_now <= primal_tol_now) & (dual_now <= dual_tol_now)
                )
                solver_converged = solver_converged | (needs_solve & newly_converged)
                needs_solve = ~solver_converged

            pair_difference = u[:, :, None, :] - u[:, None, :, :]
            pair_distance = torch.linalg.vector_norm(pair_difference, dim=-1)
            tolerance = float(merge_tolerance) * torch.maximum(
                sigma, torch.ones_like(sigma)
            )
            if edge_count:
                edge_v_zero = (v == 0.0).all(dim=-1)
                candidate = (
                    edge_active
                    & (edge_weights > 0.0)
                    & (lambda_b[:, None] > 0.0)
                    & edge_v_zero
                    & (pair_distance[:, pair_i, pair_j] <= tolerance[:, None])
                )
                pair_fused = torch.zeros(
                    (batch, groups, groups), device=descriptors.device, dtype=torch.bool
                )
                pair_fused[:, pair_i, pair_j] = candidate
                pair_fused[:, pair_j, pair_i] = candidate
                class_index = _component_partition(
                    active64, pair_fused, pair_distance, tolerance
                )
            else:
                pair_fused = torch.zeros(
                    (batch, groups, groups), device=descriptors.device, dtype=torch.bool
                )
                class_index = torch.where(
                    active64,
                    torch.arange(groups, device=descriptors.device)[None, :].expand(batch, -1),
                    torch.full((batch, groups), -1, device=descriptors.device, dtype=torch.long),
                )

            identity_index = torch.where(
                active64,
                torch.arange(groups, device=descriptors.device)[None, :].expand(batch, -1),
                torch.full((batch, groups), -1, device=descriptors.device, dtype=torch.long),
            )
            class_index = torch.where(solver_converged[:, None], class_index, identity_index)
            pair_fused = pair_fused & solver_converged[:, None, None]

            preliminary_membership = torch.nn.functional.one_hot(
                class_index.clamp_min(0), num_classes=groups
            ).to(torch.long)
            preliminary_membership = preliminary_membership * active64[..., None].to(torch.long)
            preliminary_multiplicity = preliminary_membership.sum(dim=1)
            compact_count = (preliminary_multiplicity > 0).sum(dim=-1)
            active_proposal_count = active64.sum(dim=-1)
            has_accepted_merge = compact_count < active_proposal_count
            # A row with no actual quotient keeps the exact parent slot IDs,
            # including holes from unoccupied proposals. This lets the v3
            # operator take its parent-exact path without remapping any bank.
            class_index = torch.where(has_accepted_merge[:, None], class_index, identity_index)
            pair_fused = pair_fused & has_accepted_merge[:, None, None]

            safe_index = class_index.clamp_min(0)
            membership64 = torch.nn.functional.one_hot(
                safe_index, num_classes=groups
            ).to(torch.float64)
            membership64 = membership64 * active64[..., None].to(torch.float64)
            multiplicity = membership64.sum(dim=1).to(torch.long)
            valid = multiplicity > 0
            solver_fused_descriptors = torch.einsum("bkr,bkd->brd", membership64, u)
            solver_fused_descriptors = solver_fused_descriptors / multiplicity.clamp_min(1).to(
                torch.float64
            )[..., None]
            solver_fused_descriptors = solver_fused_descriptors * valid[..., None].to(torch.float64)
            parent_singleton_descriptors = torch.einsum(
                "bkr,bkd->brd", membership64, z64
            )
            fused_descriptors = torch.where(
                multiplicity[..., None] == 1,
                parent_singleton_descriptors,
                solver_fused_descriptors,
            ).to(dtype=dtype)
            projected = torch.einsum("bkr,brd->bkd", membership64, solver_fused_descriptors)
            projection_displacement = torch.linalg.vector_norm(u - projected, dim=-1)
            merge_displacement = torch.linalg.vector_norm(u - z64, dim=-1)

        return cls(
            input_descriptors=z,
            fused_descriptors=fused_descriptors,
            class_index=class_index,
            multiplicity=multiplicity,
            valid=valid,
            active=active,
            geometry_scale=scale,
            parent_query_keys=parent_query_keys,
            parent_module_query_centers=parent_module_query_centers,
            parent_environment_query_centers=parent_environment_query_centers,
            control_dim=control_dim,
            spatial_dim=spatial_dim,
            sigma=sigma,
            lambda_b=lambda_b,
            fusion_strength=strength_tensor.detach(),
            merge_displacement=merge_displacement,
            projection_displacement=projection_displacement,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            pair_fused=pair_fused,
            solver_converged=solver_converged,
            solver_iterations=solver_iterations,
            primal_tolerance=primal_tolerance,
            dual_tolerance=dual_tolerance,
        )

    @property
    def descriptors(self) -> Tensor:
        """Alias for the packed fused access descriptors."""

        return self.fused_descriptors

    @property
    def query_keys(self) -> Tensor:
        """Decode keys without applying a second RMS normalization."""

        decoded = self.fused_descriptors[..., : self.control_dim] * math.sqrt(self.control_dim)
        if self.parent_query_keys is not None:
            if self.solver_converged is None:
                identity = (self.fusion_strength == 0.0)[:, None, None]
                return torch.where(identity, self.parent_query_keys, decoded)
            membership = torch.nn.functional.one_hot(
                self.class_index.clamp_min(0), num_classes=self.class_index.shape[-1]
            ).to(self.parent_query_keys.dtype)
            membership = membership * self.active[..., None].to(membership.dtype)
            parent_mean = torch.einsum("bkr,bkd->brd", membership, self.parent_query_keys)
            parent_mean = parent_mean / self.multiplicity.clamp_min(1).to(
                self.parent_query_keys.dtype
            )[..., None]
            singleton = (self.multiplicity == 1)[..., None]
            return torch.where(singleton, parent_mean, decoded)
        return decoded

    @property
    def module_query_centers(self) -> Tensor:
        start = self.control_dim
        end = start + self.spatial_dim
        decoded = self.fused_descriptors[..., start:end] * (
            math.sqrt(2.0) * self.geometry_scale[:, None, None]
        )
        if self.parent_module_query_centers is not None:
            if self.solver_converged is None:
                identity = (self.fusion_strength == 0.0)[:, None, None]
                return torch.where(identity, self.parent_module_query_centers, decoded)
            membership = torch.nn.functional.one_hot(
                self.class_index.clamp_min(0), num_classes=self.class_index.shape[-1]
            ).to(self.parent_module_query_centers.dtype)
            membership = membership * self.active[..., None].to(membership.dtype)
            parent_mean = torch.einsum(
                "bkr,bkd->brd", membership, self.parent_module_query_centers
            )
            parent_mean = parent_mean / self.multiplicity.clamp_min(1).to(
                self.parent_module_query_centers.dtype
            )[..., None]
            singleton = (self.multiplicity == 1)[..., None]
            return torch.where(singleton, parent_mean, decoded)
        return decoded

    @property
    def environment_query_centers(self) -> Tensor:
        start = self.control_dim + self.spatial_dim
        decoded = self.fused_descriptors[..., start:] * (
            math.sqrt(2.0) * self.geometry_scale[:, None, None]
        )
        if self.parent_environment_query_centers is not None:
            if self.solver_converged is None:
                identity = (self.fusion_strength == 0.0)[:, None, None]
                return torch.where(identity, self.parent_environment_query_centers, decoded)
            membership = torch.nn.functional.one_hot(
                self.class_index.clamp_min(0), num_classes=self.class_index.shape[-1]
            ).to(self.parent_environment_query_centers.dtype)
            membership = membership * self.active[..., None].to(membership.dtype)
            parent_mean = torch.einsum(
                "bkr,bkd->brd", membership, self.parent_environment_query_centers
            )
            parent_mean = parent_mean / self.multiplicity.clamp_min(1).to(
                self.parent_environment_query_centers.dtype
            )[..., None]
            singleton = (self.multiplicity == 1)[..., None]
            return torch.where(singleton, parent_mean, decoded)
        return decoded

    @property
    def case_group_count(self) -> Tensor:
        return self.valid.sum(dim=-1)

    @property
    def active_proposal_count(self) -> Tensor:
        return self.active.sum(dim=-1)

    @property
    def max_component_projection_displacement(self) -> Tensor:
        return (self.projection_displacement * self.active.to(self.projection_displacement.dtype)).amax(dim=-1)

    def reduce_source_banks(self, incidence: Tensor, group_control: Tensor) -> tuple[Tensor, Tensor]:
        """Return A_bar and B while preserving every fine source row."""

        _check_batched("incidence", incidence, 3)
        _check_batched("group_control", group_control, 3)
        batch, sources, groups = incidence.shape
        if tuple(group_control.shape[:2]) != (batch, groups):
            raise ValueError("group_control must align with incidence along [B,K].")
        if tuple(self.class_index.shape) != (batch, groups):
            raise ValueError("source-bank group count does not match this fusion plan.")
        index = self.class_index.clamp_min(0)
        source_valid = self.active.to(dtype=incidence.dtype)
        valid_incidence = incidence * source_valid[:, None, :]
        scatter_index = index[:, None, :].expand(batch, sources, groups)
        merged_incidence = incidence.new_zeros((batch, sources, groups)).scatter_add(
            2, scatter_index, valid_incidence
        )
        weighted_controls = valid_incidence[..., None] * group_control[:, None, :, :]
        control_index = index[:, None, :, None].expand(
            batch, sources, groups, int(group_control.shape[-1])
        )
        merged_moment = incidence.new_zeros(
            (batch, sources, groups, int(group_control.shape[-1]))
        ).scatter_add(2, control_index, weighted_controls)
        packed_mask = self.valid.to(dtype=incidence.dtype)
        merged_incidence = merged_incidence * packed_mask[:, None, :]
        merged_moment = merged_moment * packed_mask[:, None, :, None]
        return merged_incidence, merged_moment


__all__ = [
    "ADMM_STEPS",
    "EPS_ABS",
    "EPS_REL",
    "MAX_ADMM_ITERATIONS",
    "GroupFusionPlan",
    "build_group_descriptors",
    "effective_query_centers",
    "fusion_schedule",
    "source_footprint_weights",
    "weighted_sparsemax",
]
