"""Occupancy adaptive routing and population evidence helpers.

This module contains the small, CPU friendly pieces needed to audit the
Run-1409 occupancy plan.  It deliberately does not own a model or an
executor.  The live interface backend can therefore expose tensors under any
debug container and the evidence tool can consume them without changing the
forward path.

The helpers keep four quantities separate:

* registered capacity ``Kmax``;
* the original prototype IDs occupied by a case (``Kplan``);
* logical positive support in ``q -> group -> source`` paths; and
* rows actually submitted to the fine reader.

Positive support is tested with exact ``> 0`` comparisons.  This is important
for entmax evidence: a tiny positive value is a live column, while a zero is
an empty column.  No threshold, loss, hash, freeze, or monitoring state is
introduced here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from honf_forward_core.routing import entmax15


class OccupancyEvidenceError(ValueError):
    """Raised when an occupancy evidence tensor violates its contract."""


@dataclass(frozen=True)
class AssignmentAudit:
    """Exact row and column checks for one assignment tensor."""

    row_sums: torch.Tensor
    positive_degree: torch.Tensor
    empty_columns: torch.Tensor
    max_active_row_error: torch.Tensor
    active_rows_normalized: bool
    inactive_rows_zero: bool

    @property
    def empty_column_count(self) -> torch.Tensor:
        return self.empty_columns.sum(dim=-1)


@dataclass(frozen=True)
class PackedOccupancy:
    """A registered-width view of compact columns.

    ``prototype_ids`` stores original group IDs.  ``packed_valid`` marks the
    leading compact columns, and ``values`` is padded back to the registered
    width so the historical readers can keep rectangular tensor shapes.
    """

    values: torch.Tensor
    prototype_ids: torch.Tensor
    packed_valid: torch.Tensor

    @property
    def packed_width(self) -> int:
        return int(self.prototype_ids.shape[-1])


@dataclass(frozen=True)
class OccupancyPlanEvidence:
    """Evidence-only summary of a runtime occupancy plan.

    The live runtime plan remains owned by
    ``interface_fields.occupancy_group_router``. This record is intentionally
    named differently so CPU evidence code cannot be mistaken for the object
    reused by P1/P2.
    """

    kmax: int
    prototype_ids: torch.Tensor
    packed_valid: torch.Tensor
    k_plan: torch.Tensor
    proposal_module_mass: torch.Tensor
    proposal_environment_mass: torch.Tensor
    module_mass: torch.Tensor
    environment_mass: torch.Tensor
    kappa: torch.Tensor
    module_centres: torch.Tensor
    environment_centres: torch.Tensor
    joint_centres: torch.Tensor

    @property
    def Kmax(self) -> int:
        return self.kmax

    @property
    def K_plan(self) -> torch.Tensor:
        return self.k_plan

    @property
    def original_active_prototype_ids(self) -> torch.Tensor:
        """Original IDs with ``-1`` in registered-capacity padding slots."""

        return self.prototype_ids


def _check_assignment(
    assignment: torch.Tensor,
    *,
    name: str,
    row_valid: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    if not isinstance(assignment, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if assignment.ndim != 3:
        raise OccupancyEvidenceError(f"{name} must have shape [B,S,K].")
    if not assignment.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if not bool(torch.isfinite(assignment).all()):
        raise OccupancyEvidenceError(f"{name} contains non-finite values.")
    if bool((assignment < 0).any()):
        raise OccupancyEvidenceError(f"{name} must be non-negative.")
    batch, sources, kmax = (int(value) for value in assignment.shape)
    if kmax <= 0:
        raise OccupancyEvidenceError(f"{name} must have at least one group column.")
    if row_valid is not None:
        if not isinstance(row_valid, torch.Tensor):
            raise TypeError("row_valid must be a torch.Tensor.")
        if tuple(row_valid.shape) != (batch, sources):
            raise OccupancyEvidenceError(
                f"row_valid must have shape {(batch, sources)}, got {tuple(row_valid.shape)}."
            )
        if row_valid.device != assignment.device:
            raise OccupancyEvidenceError("row_valid must be on the assignment device.")
    return batch, sources, kmax


def audit_assignment(
    assignment: torch.Tensor,
    *,
    row_valid: torch.Tensor | None = None,
    name: str = "assignment",
    atol: float = 2.0e-5,
) -> AssignmentAudit:
    """Audit finite nonnegative rows, exact empty columns, and normalization."""

    _check_assignment(assignment, name=name, row_valid=row_valid)
    row_sums = assignment.sum(dim=-1)
    if row_valid is None:
        valid = torch.ones_like(row_sums, dtype=torch.bool)
    elif row_valid.dtype == torch.bool:
        valid = row_valid
    else:
        valid = row_valid > 0
    active_error = (row_sums - 1.0).abs().masked_fill(~valid, 0.0)
    inactive_rows_zero = bool(torch.all(row_sums.masked_fill(valid, 0.0) == 0.0))
    active_rows_normalized = bool(torch.all(active_error <= float(atol)))
    positive_degree = (assignment > 0.0).sum(dim=-1)
    empty_columns = ~(assignment > 0.0).any(dim=-2)
    return AssignmentAudit(
        row_sums=row_sums,
        positive_degree=positive_degree,
        empty_columns=empty_columns,
        max_active_row_error=active_error.amax(dim=-1),
        active_rows_normalized=active_rows_normalized,
        inactive_rows_zero=inactive_rows_zero,
    )


def exact_empty_columns(
    assignment: torch.Tensor,
    *,
    row_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exact all-zero columns for each batch case."""

    return audit_assignment(assignment, row_valid=row_valid).empty_columns


def _normalise_measure(
    measure: torch.Tensor,
    *,
    valid: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    if not isinstance(measure, torch.Tensor) or measure.ndim != 2:
        raise OccupancyEvidenceError(f"{name} must have shape [B,S].")
    if not measure.is_floating_point() or not bool(torch.isfinite(measure).all()):
        raise OccupancyEvidenceError(f"{name} must be finite floating point.")
    if bool((measure < 0).any()):
        raise OccupancyEvidenceError(f"{name} must be non-negative.")
    values = measure
    if valid is not None:
        if tuple(valid.shape) != tuple(values.shape):
            raise OccupancyEvidenceError(f"{name} valid mask does not align with the measure.")
        values = values * valid.to(dtype=values.dtype)
    total = values.sum(dim=-1, keepdim=True)
    if bool((total <= 0).any()):
        raise OccupancyEvidenceError(f"{name} must have positive mass in every case.")
    return values / total


def occupancy_masses(
    assignment: torch.Tensor,
    measure: torch.Tensor,
    *,
    row_valid: torch.Tensor | None = None,
    name: str = "assignment",
) -> torch.Tensor:
    """Compute normalized source measure carried by every group column."""

    _check_assignment(assignment, name=name, row_valid=row_valid)
    normalized = _normalise_measure(measure, valid=row_valid, name=f"{name}_measure")
    if tuple(normalized.shape) != tuple(assignment.shape[:2]):
        raise OccupancyEvidenceError(f"{name} measure does not align with assignment rows.")
    return torch.einsum("bs,bsk->bk", normalized, assignment)


def occupancy_kappa(module_mass: torch.Tensor, environment_mass: torch.Tensor) -> torch.Tensor:
    """Return capacity invariant ``1/sum(pi**2)`` for every case."""

    if module_mass.shape != environment_mass.shape or module_mass.ndim != 2:
        raise OccupancyEvidenceError("module and environment masses must both have shape [B,K].")
    pi = 0.5 * (module_mass + environment_mass)
    denominator = pi.square().sum(dim=-1)
    if bool((denominator <= 0).any()) or not bool(torch.isfinite(denominator).all()):
        raise OccupancyEvidenceError("occupancy masses must define a nonzero finite pi distribution.")
    return denominator.reciprocal()


def _weighted_centres(
    coordinates: torch.Tensor,
    measure: torch.Tensor,
    assignment: torch.Tensor,
    masses: torch.Tensor,
) -> torch.Tensor:
    numerator = torch.einsum("bs,bsk,bsd->bkd", measure, assignment, coordinates)
    centres = numerator / masses.clamp_min(torch.finfo(numerator.dtype).tiny)[..., None]
    return torch.where(masses[..., None] > 0.0, centres, torch.zeros_like(centres))


def derive_occupancy_plan(
    proposal_module: torch.Tensor,
    proposal_environment: torch.Tensor,
    refined_module: torch.Tensor,
    refined_environment: torch.Tensor,
    module_measure: torch.Tensor,
    environment_measure: torch.Tensor,
    module_coordinates: torch.Tensor,
    environment_coordinates: torch.Tensor,
    *,
    module_valid: torch.Tensor | None = None,
) -> OccupancyPlanEvidence:
    """Derive an evidence summary from proposal and one refinement.

    Proposal occupancy is used only to identify the candidate columns allowed
    by the refinement.  The deployed IDs and ``Kplan`` come from the final
    positive masses, exactly as specified by the Run-1409 plan.
    """

    _check_assignment(proposal_module, name="proposal_module", row_valid=module_valid)
    _check_assignment(proposal_environment, name="proposal_environment")
    _check_assignment(refined_module, name="refined_module", row_valid=module_valid)
    _check_assignment(refined_environment, name="refined_environment")
    if proposal_module.shape != refined_module.shape:
        raise OccupancyEvidenceError("module proposal/refinement shapes differ.")
    if proposal_environment.shape != refined_environment.shape:
        raise OccupancyEvidenceError("environment proposal/refinement shapes differ.")
    if proposal_module.shape[-1] != proposal_environment.shape[-1]:
        raise OccupancyEvidenceError("module/environment assignments must share Kmax.")
    batch, modules, kmax = (int(value) for value in refined_module.shape)
    if int(refined_environment.shape[0]) != batch:
        raise OccupancyEvidenceError("module/environment assignments must share batch size.")
    if module_coordinates.shape[:2] != (batch, modules) or module_coordinates.ndim != 3:
        raise OccupancyEvidenceError("module_coordinates must align with module assignments.")
    environments = int(refined_environment.shape[1])
    if environment_coordinates.shape[:2] != (batch, environments) or environment_coordinates.ndim != 3:
        raise OccupancyEvidenceError("environment_coordinates must align with environment assignments.")
    if module_coordinates.shape[-1] != environment_coordinates.shape[-1]:
        raise OccupancyEvidenceError("module/environment coordinate dimensions differ.")

    module_measure_n = _normalise_measure(
        module_measure,
        valid=module_valid,
        name="module_measure",
    )
    environment_measure_n = _normalise_measure(
        environment_measure,
        valid=None,
        name="environment_measure",
    )
    proposal_module_mass = occupancy_masses(
        proposal_module, module_measure_n, row_valid=module_valid, name="proposal_module"
    )
    proposal_environment_mass = occupancy_masses(
        proposal_environment, environment_measure_n, name="proposal_environment"
    )
    module_mass = occupancy_masses(
        refined_module, module_measure_n, row_valid=module_valid, name="refined_module"
    )
    environment_mass = occupancy_masses(
        refined_environment, environment_measure_n, name="refined_environment"
    )
    final_occupied = (module_mass + environment_mass) > 0.0
    proposal_occupied = (proposal_module_mass + proposal_environment_mass) > 0.0
    if bool((proposal_occupied.sum(dim=-1) == 0).any()):
        raise OccupancyEvidenceError("every case must have at least one proposal-occupied group.")
    if bool((final_occupied.sum(dim=-1) == 0).any()):
        raise OccupancyEvidenceError("every case must have at least one final occupied group.")
    if bool(((refined_module.sum(dim=1) > 0.0) & ~proposal_occupied).any()) or bool(
        ((refined_environment.sum(dim=1) > 0.0) & ~proposal_occupied).any()
    ):
        raise OccupancyEvidenceError(
            "refined assignments use a proposal-empty group; refinement must be masked to proposal occupancy."
        )

    module_centres = _weighted_centres(
        module_coordinates, module_measure_n, refined_module, module_mass
    )
    environment_centres = _weighted_centres(
        environment_coordinates,
        environment_measure_n,
        refined_environment,
        environment_mass,
    )
    total_mass = module_mass + environment_mass
    joint_centres = (
        module_mass[..., None] * module_centres
        + environment_mass[..., None] * environment_centres
    ) / total_mass.clamp_min(torch.finfo(total_mass.dtype).tiny)[..., None]
    joint_centres = torch.where(total_mass[..., None] > 0.0, joint_centres, torch.zeros_like(joint_centres))
    ids = torch.full((batch, kmax), -1, dtype=torch.long, device=refined_module.device)
    valid = torch.zeros((batch, kmax), dtype=torch.bool, device=refined_module.device)
    for batch_index in range(batch):
        occupied_ids = torch.nonzero(final_occupied[batch_index], as_tuple=False).flatten()
        count = int(occupied_ids.numel())
        ids[batch_index, :count] = occupied_ids
        valid[batch_index, :count] = True
    return OccupancyPlanEvidence(
        kmax=kmax,
        prototype_ids=ids,
        packed_valid=valid,
        k_plan=valid.sum(dim=-1),
        proposal_module_mass=proposal_module_mass,
        proposal_environment_mass=proposal_environment_mass,
        module_mass=module_mass,
        environment_mass=environment_mass,
        kappa=occupancy_kappa(module_mass, environment_mass),
        module_centres=module_centres,
        environment_centres=environment_centres,
        joint_centres=joint_centres,
    )


def pack_columns(
    values: torch.Tensor,
    prototype_ids: torch.Tensor,
    *,
    packed_valid: torch.Tensor | None = None,
    fill_value: float = 0.0,
) -> PackedOccupancy:
    """Gather original group columns and pad to the registered ``Kmax`` width."""

    if values.ndim != 3 or prototype_ids.ndim != 2:
        raise OccupancyEvidenceError("values must be [B,S,K] and prototype_ids [B,Kpack].")
    batch, _, kmax = (int(value) for value in values.shape)
    if int(prototype_ids.shape[0]) != batch:
        raise OccupancyEvidenceError("prototype_ids batch does not match values.")
    if prototype_ids.dtype != torch.long:
        raise TypeError("prototype_ids must be torch.long.")
    if bool((prototype_ids >= kmax).any()) or bool((prototype_ids < -1).any()):
        raise OccupancyEvidenceError("prototype_ids contain an out-of-range column.")
    if packed_valid is None:
        valid = prototype_ids >= 0
    else:
        if tuple(packed_valid.shape) != tuple(prototype_ids.shape):
            raise OccupancyEvidenceError("packed_valid must align with prototype_ids.")
        valid = packed_valid.to(dtype=torch.bool)
    # The output axis is compact slot order. ``prototype_ids`` remains the
    # original registered IDs used to gather corresponding prototype/control
    # rows. Keeping those two axes distinct is what makes nonconsecutive IDs
    # (for example 0, 2, 5) auditable.
    output = values.new_full((batch, values.shape[1], int(prototype_ids.shape[1])), float(fill_value))
    for batch_index in range(batch):
        ids = prototype_ids[batch_index]
        mask = valid[batch_index] & (ids >= 0)
        if bool(mask.any()):
            slots = torch.nonzero(mask, as_tuple=False).flatten()
            output[batch_index, :, slots] = values[batch_index, :, ids[mask]]
    return PackedOccupancy(output, prototype_ids, valid)


def mask_query_assignments(
    query_logits_or_assignment: torch.Tensor,
    source_mass: torch.Tensor,
    *,
    logits: bool = True,
) -> torch.Tensor:
    """Route only over groups with nonzero current source mass.

    A row with no occupied source group returns all zeros through the shared
    exact-entmax mask behavior.  In normal Run-1409 plans at least one source
    group is occupied, but preserving this behavior makes the helper safe for
    padded phase records.
    """

    if query_logits_or_assignment.ndim != 3 or source_mass.ndim != 2:
        raise OccupancyEvidenceError("query values must be [B,Q,K] and source_mass [B,K].")
    if query_logits_or_assignment.shape[0] != source_mass.shape[0] or query_logits_or_assignment.shape[-1] != source_mass.shape[-1]:
        raise OccupancyEvidenceError("query values and source_mass must share [B,K].")
    mask = source_mass > 0.0
    if logits:
        return entmax15(query_logits_or_assignment, dim=-1, mask=mask[:, None, :])
    values = query_logits_or_assignment * mask[:, None, :].to(query_logits_or_assignment.dtype)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)


def support_metrics(
    module_assignment: torch.Tensor,
    environment_assignment: torch.Tensor,
    query_assignment: torch.Tensor,
    *,
    module_valid: torch.Tensor | None = None,
    actual_module_rows: torch.Tensor | None = None,
    actual_environment_rows: torch.Tensor | None = None,
    padded_module_rows: torch.Tensor | None = None,
    padded_environment_rows: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute population-ready degrees, entropy, support, and work ledgers."""

    _check_assignment(module_assignment, name="module_assignment", row_valid=module_valid)
    _check_assignment(environment_assignment, name="environment_assignment")
    _check_assignment(query_assignment, name="query_assignment")
    if module_assignment.shape[0] != environment_assignment.shape[0] or module_assignment.shape[0] != query_assignment.shape[0]:
        raise OccupancyEvidenceError("all assignments must share batch size.")
    if module_assignment.shape[-1] != environment_assignment.shape[-1] or module_assignment.shape[-1] != query_assignment.shape[-1]:
        raise OccupancyEvidenceError("all assignments must share Kmax.")
    module_positive = module_assignment > 0.0
    environment_positive = environment_assignment > 0.0
    query_positive = query_assignment > 0.0
    if module_valid is None:
        module_valid_bool = torch.ones(module_assignment.shape[:2], dtype=torch.bool, device=module_assignment.device)
    else:
        module_valid_bool = module_valid.to(dtype=torch.bool)
    module_positive = module_positive & module_valid_bool[..., None]
    module_logical = torch.einsum("bqk,bsk->bqs", query_positive.to(torch.int64), module_positive.to(torch.int64))
    environment_logical = torch.einsum("bqk,bsk->bqs", query_positive.to(torch.int64), environment_positive.to(torch.int64))
    module_support = module_logical > 0
    environment_support = environment_logical > 0
    module_dense = module_valid_bool.sum(dim=-1)[:, None] * query_assignment.shape[1]
    environment_dense = torch.full_like(module_dense, int(environment_assignment.shape[1])) * query_assignment.shape[1]
    module_unique = module_support.sum(dim=(-1, -2))
    environment_unique = environment_support.sum(dim=(-1, -2))
    module_paths = module_logical.sum(dim=(-1, -2))
    environment_paths = environment_logical.sum(dim=(-1, -2))
    result: dict[str, torch.Tensor] = {
        "module_positive_degree": module_positive.sum(dim=-1).to(torch.float32).mean(dim=-1),
        "environment_positive_degree": environment_positive.sum(dim=-1).to(torch.float32).mean(dim=-1),
        "query_positive_degree": query_positive.sum(dim=-1).to(torch.float32).mean(dim=-1),
        "module_effective_groups": _effective_group_count(module_assignment, module_valid_bool),
        "environment_effective_groups": _effective_group_count(environment_assignment),
        "query_effective_groups": _effective_group_count(query_assignment),
        "module_entropy": _normalized_entropy(module_assignment, module_valid_bool),
        "environment_entropy": _normalized_entropy(environment_assignment),
        "query_entropy": _normalized_entropy(query_assignment),
        "module_logical_paths": module_paths,
        "environment_logical_paths": environment_paths,
        "module_unique_pairs": module_unique,
        "environment_unique_pairs": environment_unique,
        "module_RM": module_unique.to(torch.float32) / module_dense.to(torch.float32).clamp_min(1.0),
        "environment_RE": environment_unique.to(torch.float32) / environment_dense.to(torch.float32).clamp_min(1.0),
        "module_multiplicity": module_paths.to(torch.float32) / module_unique.to(torch.float32).clamp_min(1.0),
        "environment_multiplicity": environment_paths.to(torch.float32) / environment_unique.to(torch.float32).clamp_min(1.0),
    }
    for key, value in (
        ("actual_module_rows", actual_module_rows),
        ("actual_environment_rows", actual_environment_rows),
        ("padded_module_rows", padded_module_rows),
        ("padded_environment_rows", padded_environment_rows),
    ):
        if value is not None:
            result[key] = value
    return result


def _effective_group_count(values: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    positive = values.clamp_min(0.0)
    if valid is not None:
        positive = positive * valid[..., None].to(dtype=positive.dtype)
    return positive.square().sum(dim=-1).clamp_min(torch.finfo(positive.dtype).tiny).reciprocal().mean(dim=-1)


def _normalized_entropy(values: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    positive = values.clamp_min(0.0)
    if valid is not None:
        positive = positive * valid[..., None].to(dtype=positive.dtype)
    kmax = int(positive.shape[-1])
    entropy = -(positive * positive.clamp_min(torch.finfo(positive.dtype).tiny).log()).sum(dim=-1)
    return (entropy / max(float(torch.log(positive.new_tensor(float(max(kmax, 2))))), 1.0)).mean(dim=-1)


def permutation_report(
    assignment: torch.Tensor,
    measure: torch.Tensor,
    coordinates: torch.Tensor,
    permutation: torch.Tensor,
    *,
    row_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compare weighted masses and centres after a source row permutation."""

    if permutation.ndim != 1 or int(permutation.numel()) != int(assignment.shape[1]):
        raise OccupancyEvidenceError("permutation must cover every source row exactly once.")
    if sorted(permutation.detach().cpu().tolist()) != list(range(int(assignment.shape[1]))):
        raise OccupancyEvidenceError("permutation is not a source-row permutation.")
    masses = occupancy_masses(assignment, measure, row_valid=row_valid)
    if row_valid is None:
        valid = torch.ones(assignment.shape[:2], dtype=torch.bool, device=assignment.device)
    else:
        valid = row_valid.to(dtype=torch.bool)
    permuted = assignment[:, permutation]
    permuted_measure = measure[:, permutation]
    permuted_valid = valid[:, permutation]
    permuted_coordinates = coordinates[:, permutation]
    permuted_masses = occupancy_masses(permuted, permuted_measure, row_valid=permuted_valid)
    centres = _weighted_centres(
        coordinates,
        _normalise_measure(measure, valid=valid, name="measure"),
        assignment,
        masses,
    )
    permuted_centres = _weighted_centres(
        permuted_coordinates,
        _normalise_measure(permuted_measure, valid=permuted_valid, name="permuted_measure"),
        permuted,
        permuted_masses,
    )
    return {
        "mass_max_abs_error": (masses - permuted_masses).abs().amax(),
        "centre_max_abs_error": (centres - permuted_centres).abs().amax(),
    }


def parity_report(full: torch.Tensor, packed: torch.Tensor, *, atol: float = 2.0e-6) -> dict[str, Any]:
    """Return a compact full/packed tensor parity record."""

    if full.shape != packed.shape:
        raise OccupancyEvidenceError(f"full/packed shape mismatch: {tuple(full.shape)} vs {tuple(packed.shape)}")
    difference = (full - packed).abs()
    return {
        "shape": list(full.shape),
        "max_abs_error": float(difference.max().detach().cpu()) if difference.numel() else 0.0,
        "mean_abs_error": float(difference.mean().detach().cpu()) if difference.numel() else 0.0,
        "pass": bool(torch.all(difference <= float(atol))),
    }


def population_histogram(values: Sequence[int] | torch.Tensor) -> dict[str, int]:
    """Return a deterministic histogram suitable for JSON evidence."""

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    histogram: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


def jsonable(value: Any) -> Any:
    """Convert tensor and nested mapping values to JSON compatible values."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


__all__ = [
    "AssignmentAudit",
    "OccupancyEvidenceError",
    "OccupancyPlanEvidence",
    "PackedOccupancy",
    "audit_assignment",
    "derive_occupancy_plan",
    "exact_empty_columns",
    "jsonable",
    "mask_query_assignments",
    "occupancy_kappa",
    "occupancy_masses",
    "pack_columns",
    "parity_report",
    "permutation_report",
    "population_histogram",
    "support_metrics",
]
