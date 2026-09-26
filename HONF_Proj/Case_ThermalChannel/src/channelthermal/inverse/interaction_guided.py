"""Matched, baseline-anchored inverse-design experiments.

The module deliberately knows only the response-factor interface.  A caller
prepares a baseline cache from the design and operating context, supplies its
physically indexed factors, and converts the *sum of factor outputs* into the
decision quantities.  There is no trial-design forward method in this
protocol, so a perturbed-design surrogate cannot silently bypass the graph.

The decision study keeps the original ThermalChannel module identities,
radii, heating, and operating context fixed.  Only module centers move.  It
uses the maintained pressure-drop definition from
``channelthermal.inverse.functionals``: mean pressure on fluid grid points
with ``x <= 0.08 Lx`` minus mean pressure on fluid grid points with
``x >= 0.92 Lx``.  The response adapter must compute that same quantity for
baseline and trial outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Hashable, Literal, Mapping, Protocol, Sequence

import numpy as np

from honf_inverse_core.contracts import NamedContext, PhysicalDesign
from honf_inverse_core.request_schema import GeometryConstraints

from .functionals import INLET_BAND_FRACTION, OUTLET_BAND_FRACTION
from .geometry import evaluate_geometry


EvidenceSource = Literal[
    "reference_solver",
    "stored_reference",
    "surrogate_teacher",
    "analytic_synthetic",
]
EVIDENCE_SOURCES = frozenset(
    {"reference_solver", "stored_reference", "surrogate_teacher", "analytic_synthetic"}
)
PHYSICAL_EVIDENCE_SOURCES = frozenset({"reference_solver", "stored_reference"})
PHYSICAL_SOLVER_SOURCES = frozenset({"reference_solver"})
PRESSURE_DROP_DEFINITION = (
    "fluid-grid disk-mask means: mean(p[x <= 0.08*Lx]) - "
    "mean(p[x >= 0.92*Lx])"
)
POLICY_NAMES = ("graph_guided", "size_matched_random", "ungrouped_local")


def _finite_scalar(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    """Detach small response tensors from an accelerator before scoring."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


def _id_key(value: Hashable) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _sum_typed_responses(parts: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Sum role-specific factor outputs without changing their query axes."""

    total: dict[str, np.ndarray] = {}
    for part_index, part in enumerate(parts):
        for role, raw in part.items():
            values = _as_numpy(raw, name=f"factor response {part_index}:{role}")
            if role in total:
                if total[role].shape != values.shape:
                    raise ValueError(f"Factor outputs for role {role!r} do not share query shape.")
                total[role] = total[role] + values
            else:
                total[role] = values.copy()
    return total


@dataclass(frozen=True)
class DecisionObservation:
    """Measured baseline or trial quantities used by the inverse decision.

    ``module_temperature_by_id`` contains a physical peak temperature for each
    active module, evaluated in material coordinates.  ``pressure_drop`` must
    use :data:`PRESSURE_DROP_DEFINITION` and retain its physical source units.
    ``output_state`` may carry the complete reference field/port/solid maps so
    an anchored response adapter can add predicted increments to this exact
    baseline.  Its provenance is never used to build factor membership here.
    """

    module_temperature_by_id: Mapping[Hashable, float]
    pressure_drop: float
    pressure_drop_units: str
    evidence_source: EvidenceSource
    pressure_drop_definition: str = PRESSURE_DROP_DEFINITION
    status: str = "completed"
    elapsed_seconds: float = 0.0
    provenance: Mapping[str, Any] = field(default_factory=dict)
    output_state: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        temps = {
            key: _finite_scalar(value, name=f"module temperature {key!r}")
            for key, value in self.module_temperature_by_id.items()
        }
        if not temps:
            raise ValueError("An observation must include temperatures for active modules.")
        if self.evidence_source not in EVIDENCE_SOURCES:
            raise ValueError(f"Unsupported evidence source {self.evidence_source!r}.")
        if self.pressure_drop_definition != PRESSURE_DROP_DEFINITION:
            raise ValueError("Pressure drop must use the maintained fluid-grid section definition.")
        if not str(self.pressure_drop_units).strip():
            raise ValueError("Pressure-drop units must be recorded.")
        if self.status not in {"completed", "failed"}:
            raise ValueError("Observation status must be 'completed' or 'failed'.")
        if self.status == "failed":
            raise ValueError("Failed evaluations belong in a failed trial record, not an observation.")
        object.__setattr__(self, "module_temperature_by_id", temps)
        object.__setattr__(self, "pressure_drop", _finite_scalar(self.pressure_drop, name="pressure_drop"))
        object.__setattr__(self, "elapsed_seconds", max(0.0, _finite_scalar(self.elapsed_seconds, name="elapsed_seconds")))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "output_state", dict(self.output_state))

    @property
    def true_peak_temperature(self) -> float:
        return max(self.module_temperature_by_id.values())


@dataclass(frozen=True)
class DecisionEstimate:
    """Decision observables reduced from baseline output plus factor deltas."""

    module_temperature_by_id: Mapping[Hashable, float]
    pressure_drop: float
    pressure_drop_units: str
    pressure_drop_definition: str = PRESSURE_DROP_DEFINITION
    output_state: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    pressure_section_mask_changed: bool = False

    def __post_init__(self) -> None:
        temps = {
            key: _finite_scalar(value, name=f"predicted module temperature {key!r}")
            for key, value in self.module_temperature_by_id.items()
        }
        if not temps:
            raise ValueError("A predicted decision state must include active module temperatures.")
        if self.pressure_drop_definition != PRESSURE_DROP_DEFINITION:
            raise ValueError("Predicted pressure drop must use the maintained section definition.")
        if not str(self.pressure_drop_units).strip():
            raise ValueError("Predicted pressure-drop units must be recorded.")
        object.__setattr__(self, "module_temperature_by_id", temps)
        object.__setattr__(self, "pressure_drop", _finite_scalar(self.pressure_drop, name="predicted pressure_drop"))
        object.__setattr__(self, "output_state", dict(self.output_state))


@dataclass(frozen=True)
class PreparedResponseBaseline:
    """Opaque response-model cache, fixed query set, and baseline factors.

    The cache and query set are prepared once at an accepted baseline and are
    shared by every candidate in that inner step.  ``factors`` must be based
    only on this baseline and permitted context; it must not depend on a
    candidate perturbation, held-out output, case ID, or inverse objective.
    """

    cache: Any
    queries: Any
    factors: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        factor_ids = [_factor_id(factor) for factor in factors]
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("Factor IDs must be unique within a baseline graph.")
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "metadata", dict(self.metadata))


class AnchoredResponseOracle(Protocol):
    """Narrow adapter over the anchored factor operator.

    Implementations normally wrap ``AnchoredResponseFactorOperator``.  There
    is intentionally no ``predict_trial``/``forward_design`` method.  The
    decoder consumes the sum of typed factor increments and the accepted
    baseline observation; it must not call a dense surrogate at the trial
    design.
    """

    def prepare_baseline(
        self,
        design: PhysicalDesign,
        context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
    ) -> PreparedResponseBaseline: ...

    def rank_factors(self, prepared: PreparedResponseBaseline) -> Sequence[Any]:
        """Rank factors by baseline-only support evidence, never trial outputs."""
        ...

    def predict_factor_response(
        self,
        prepared: PreparedResponseBaseline,
        factor: Any,
        delta_by_module_id: Mapping[Hashable, np.ndarray],
    ) -> Mapping[str, Any]: ...

    def factor_validity_status(
        self,
        prepared: PreparedResponseBaseline,
        factor: Any,
        delta_by_module_id: Mapping[Hashable, np.ndarray],
    ) -> bool | None: ...


def decode_thermal_field_response(
    baseline: DecisionObservation,
    summed_factor_response: Mapping[str, np.ndarray],
    trial_design: PhysicalDesign,
    context: NamedContext,
) -> DecisionEstimate:
    """Add typed response increments to a measured baseline and reduce metrics.

    This adapter performs no model call.  It needs these baseline
    ``output_state`` arrays: ``fluid_fields`` with channel order,
    ``grid_xy``, optional ``grid_valid_mask`` (all queryable fixed-grid sites),
    ``solid_temperature``, and matching ``solid_module_ids``. It reduces
    pressure on the accepted baseline's fixed resolved inlet/outlet masks.
    Trial geometry only checks whether disk occupancy would change those
    sections; such a candidate is marked unsupported and cannot be scored or
    selected for a reference call. Response increment keys use the model's
    typed roles: ``fluid_fields``, ``interface``, and ``solid_temperature``.
    """

    output = baseline.output_state
    required = {"fluid_fields", "grid_xy", "solid_temperature", "solid_module_ids", "channel_order"}
    missing = required.difference(output)
    if missing:
        raise ValueError(f"Baseline output_state is missing required decision fields: {sorted(missing)}.")
    channel_order = tuple(str(value) for value in output["channel_order"])
    try:
        pressure_index = channel_order.index("p")
    except ValueError as error:
        raise ValueError("Baseline field channel_order does not contain pressure 'p'.") from error
    baseline_field = np.asarray(output["fluid_fields"], dtype=np.float64)
    field_delta = _as_numpy(
        summed_factor_response.get("fluid_fields", np.zeros_like(baseline_field)),
        name="fluid_fields response",
    ).astype(np.float64)
    if field_delta.shape != baseline_field.shape:
        raise ValueError("fluid_fields factor response must align exactly with the accepted baseline query grid.")
    grid_xy = _as_numpy(output["grid_xy"], name="baseline grid_xy").astype(np.float64)
    if grid_xy.shape != (*baseline_field.shape[:-1], 2):
        raise ValueError("grid_xy must align with all fluid_fields queries and have a final coordinate width of 2.")
    query_valid = np.asarray(output.get("grid_valid_mask", np.ones(grid_xy.shape[:-1], dtype=bool)), dtype=bool)
    if query_valid.shape != grid_xy.shape[:-1]:
        raise ValueError("grid_valid_mask must align with the fixed query grid.")
    if not np.isfinite(baseline_field[query_valid]).all():
        raise ValueError("Baseline fluid values must be finite at resolved fixed-grid queries.")
    field = baseline_field + field_delta
    x = grid_xy[..., 0]
    y = grid_xy[..., 1]
    values = context.as_mapping()
    centers = trial_design.module_centers[trial_design.module_present > 0.5].astype(np.float64)
    module_radius = float(values["module_radius"])
    candidate_fluid = np.ones(query_valid.shape, dtype=bool)
    for center_x, center_y in centers:
        candidate_fluid &= (x - center_x) ** 2 + (y - center_y) ** 2 > module_radius**2
    inlet_section = x <= INLET_BAND_FRACTION * float(values["domain_length_x"])
    outlet_section = x >= (1.0 - OUTLET_BAND_FRACTION) * float(values["domain_length_x"])
    # Pressure predictions use the accepted baseline's resolved fluid queries
    # as fixed quadrature points. Trial geometry cannot change the reduction
    # mask outside the anchored response factors. A section-mask change is
    # reported unsupported and is decided only from an independent solve.
    inlet = query_valid & inlet_section
    outlet = query_valid & outlet_section
    if not inlet.any() or not outlet.any():
        raise ValueError("Accepted baseline has an empty maintained inlet/outlet pressure section.")
    pressure_section_mask_changed = bool(
        not np.array_equal(inlet, candidate_fluid & inlet_section)
        or not np.array_equal(outlet, candidate_fluid & outlet_section)
    )
    pressure_drop = float(np.mean(field[..., pressure_index][inlet]) - np.mean(field[..., pressure_index][outlet]))

    solid_base = np.asarray(output["solid_temperature"], dtype=np.float64)
    solid_delta = _as_numpy(
        summed_factor_response.get("solid_temperature", np.zeros_like(solid_base)),
        name="solid_temperature response",
    ).astype(np.float64)
    if solid_delta.shape != solid_base.shape:
        raise ValueError("solid_temperature response must align exactly with baseline material queries.")
    solid_ids = np.asarray(output["solid_module_ids"], dtype=object).reshape(-1)
    solid_values = solid_base.reshape(-1) + solid_delta.reshape(-1)
    if solid_ids.shape != solid_values.shape:
        raise ValueError("solid_module_ids must identify every baseline material temperature query.")
    solid_valid = np.asarray(
        output.get("solid_valid_mask", np.ones_like(solid_base, dtype=bool)), dtype=bool
    ).reshape(-1)
    if solid_valid.shape != solid_values.shape:
        raise ValueError("solid_valid_mask must identify every resolved material temperature query.")
    if not np.isfinite(solid_values[solid_valid]).all():
        raise ValueError("Baseline solid temperatures must be finite at resolved material queries.")
    active_ids = [value for value in output.get("active_module_ids", ())]
    if not active_ids:
        active_ids = list(baseline.module_temperature_by_id)
    temperatures: dict[Hashable, float] = {}
    for module_id in active_ids:
        selected = np.asarray([value == module_id for value in solid_ids], dtype=bool) & solid_valid
        if not selected.any():
            raise ValueError(f"No resolved material-coordinate solid temperature queries for module {module_id!r}.")
        temperatures[module_id] = float(np.max(solid_values[selected]))
    if set(temperatures) != set(baseline.module_temperature_by_id):
        raise ValueError("Material query IDs must match active baseline physical module IDs.")

    summed_outputs: dict[str, np.ndarray] = {
        "fluid_fields": field,
        "solid_temperature": solid_base + solid_delta,
    }
    if "interface" in output:
        interface_base = np.asarray(output["interface"], dtype=np.float64)
        interface_delta = _as_numpy(
            summed_factor_response.get("interface", np.zeros_like(interface_base)),
            name="interface response",
        ).astype(np.float64)
        if interface_delta.shape != interface_base.shape:
            raise ValueError("interface response must align exactly with baseline material interface queries.")
        interface_valid = np.asarray(
            output.get("interface_valid_mask", np.ones_like(interface_base, dtype=bool)), dtype=bool
        )
        if interface_valid.shape != interface_base.shape:
            raise ValueError("interface_valid_mask must align with both material interface channels.")
        if not np.isfinite(interface_base[interface_valid]).all():
            raise ValueError("Baseline interface values must be finite at resolved material queries.")
        summed_outputs["interface"] = interface_base + interface_delta
    elif "interface" in summed_factor_response:
        raise ValueError("Interface increments were predicted but baseline interface values are absent.")
    return DecisionEstimate(
        module_temperature_by_id=temperatures,
        pressure_drop=pressure_drop,
        pressure_drop_units=baseline.pressure_drop_units,
        output_state=summed_outputs,
        pressure_section_mask_changed=pressure_section_mask_changed,
    )


@dataclass(frozen=True)
class InverseStudyConfig:
    """Predeclared budgets and physical scales for a matched policy study."""

    tau_temperature: float
    tau_source: str
    pressure_drop_limit: float
    pressure_limit_basis: str
    pressure_drop_units: str
    trust_radius_normalized: float = 0.01
    candidate_budget: int = 6
    max_reference_trials_per_policy: int = 6
    max_updates_per_policy: int = 6
    validation_mode: Literal["selected_only", "exhaustive_pool"] = "selected_only"
    decision_resolution: float = 1.0e-4
    random_seed: int = 20260925
    candidate_redraw_max_seed_offsets: int = 0
    candidate_redraw_seed_offset_stride: int = 104729

    def __post_init__(self) -> None:
        if self.tau_temperature <= 0.0 or not math.isfinite(self.tau_temperature):
            raise ValueError("tau_temperature must be finite and positive.")
        if not self.tau_source.strip():
            raise ValueError("tau_source must record its training-only calibration basis.")
        if not self.pressure_limit_basis.strip():
            raise ValueError("pressure_limit_basis must state whether the limit is engineering or benchmark-relative.")
        if not self.pressure_drop_units.strip():
            raise ValueError("pressure_drop_units must be recorded.")
        if not math.isfinite(self.pressure_drop_limit):
            raise ValueError("pressure_drop_limit must be finite.")
        if self.trust_radius_normalized <= 0.0 or not math.isfinite(self.trust_radius_normalized):
            raise ValueError("trust_radius_normalized must be finite and positive.")
        if not (1 <= self.candidate_budget <= 40):
            raise ValueError("candidate_budget must be between 1 and 40.")
        if not (1 <= self.max_reference_trials_per_policy <= 6):
            raise ValueError("max_reference_trials_per_policy must be between 1 and 6.")
        if not (1 <= self.max_updates_per_policy <= self.max_reference_trials_per_policy):
            raise ValueError("max_updates_per_policy cannot exceed the reference-trial budget.")
        if self.validation_mode not in {"selected_only", "exhaustive_pool"}:
            raise ValueError("Unsupported validation_mode.")
        if self.validation_mode == "exhaustive_pool" and self.candidate_budget > self.max_reference_trials_per_policy:
            raise ValueError("Exhaustive candidate validation exceeds the per-policy reference-call ceiling.")
        if self.decision_resolution < 0.0 or not math.isfinite(self.decision_resolution):
            raise ValueError("decision_resolution must be finite and nonnegative.")
        if not (0 <= self.candidate_redraw_max_seed_offsets <= 4):
            raise ValueError("candidate_redraw_max_seed_offsets must be between zero and four.")
        if self.candidate_redraw_seed_offset_stride <= 0:
            raise ValueError("candidate_redraw_seed_offset_stride must be positive.")


@dataclass(frozen=True)
class CandidateRecord:
    policy: str
    candidate_id: str
    update_index: int
    source_module_ids: tuple[Hashable, ...]
    factor_id: str | None
    normalized_dimension: int
    normalized_radius: float
    delta_by_module_id: Mapping[Hashable, tuple[float, float]]
    geometry_valid: bool
    predicted_feasible: bool | None
    predicted_objective: float | None
    predicted_true_peak: float | None
    predicted_pressure_drop: float | None
    predicted_pressure_violation: float | None
    selected_factor_objective_contribution: float | None
    score_status: str
    predicted_module_temperature_by_id: Mapping[Hashable, float] = field(default_factory=dict)
    evidence_validity_status: Literal[
        "within_measured_neighborhood", "outside_measured_neighborhood", "unknown", "not_scored"
    ] = "unknown"
    graph_factor_status: Literal["active", "missing_order_fallback", "not_applicable"] = "not_applicable"
    pressure_prediction_status: Literal[
        "fixed_baseline_sections", "candidate_section_mask_changed", "not_scored"
    ] = "fixed_baseline_sections"


@dataclass(frozen=True)
class TrialRecord:
    """One charged evaluator call, including failures and failed trials."""

    policy: str
    candidate_id: str
    evaluator_source: EvidenceSource
    call_index: int
    status: str
    elapsed_seconds: float
    solver_elapsed_seconds: float | None = 0.0
    solver_elapsed_available: bool = True
    raw_recovered_after_adapter_error: bool = False
    exception: str | None = None
    actual_objective: float | None = None
    actual_true_peak: float | None = None
    actual_module_temperature_by_id: Mapping[Hashable, float] = field(default_factory=dict)
    actual_pressure_drop: float | None = None
    geometry_valid: bool | None = None
    feasible: bool | None = None
    accepted: bool = False
    physical_acceptance: bool | None = None
    predicted_improvement: float | None = None
    actual_improvement: float | None = None
    improvement_ratio: float | None = None


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    candidates: tuple[CandidateRecord, ...]
    trials: tuple[TrialRecord, ...]
    accepted_updates: int
    final_design: PhysicalDesign
    final_observation: DecisionObservation
    best_evaluated_feasible_design: PhysicalDesign | None
    best_evaluated_feasible_observation: DecisionObservation | None
    best_physical_feasible_design: PhysicalDesign | None
    best_physical_feasible_observation: DecisionObservation | None
    candidate_rank_spearman: float | None
    candidate_improvement_sign_accuracy: float | None
    mean_abs_improvement_mismatch: float | None
    geometry_rejections: int
    evaluator_calls: int
    reference_calls: int
    physical_solver_calls: int
    physical_solver_seconds: float
    teacher_evaluations: int
    synthetic_evaluations: int
    factor_prediction_calls: int
    recovered_reference_calls: int
    unknown_solver_time_calls: int
    candidate_redraw_audit: tuple[Mapping[str, Any], ...]
    status: str


@dataclass(frozen=True)
class MatchedInverseStudyResult:
    initial_design: PhysicalDesign
    initial_observation: DecisionObservation
    config: InverseStudyConfig
    pressure_drop_limit: float
    normalized_step_coordinate_definition: str
    proposal_dimension_by_index: tuple[int, ...]
    proposal_order_counts: Mapping[int, int]
    factor_source_selection_basis: str
    graph_guidance_status: Literal["active", "partial_fallback", "unavailable"]
    graph_factor_fallback_count: int
    policies: Mapping[str, PolicyResult]
    best_policy_pool_union_policy: str | None
    best_policy_pool_union_objective: float | None
    policy_pools_are_common: bool
    evidence_interpretation: str


@dataclass(frozen=True)
class _PreparedCandidate:
    record: CandidateRecord
    design: PhysicalDesign
    estimate: DecisionEstimate | None
    predicted_objective: float | None
    predicted_violation: float | None


def smooth_peak_temperature(temperatures: Sequence[float], tau: float) -> float:
    """Stable ``tau*log(mean(exp(T/tau)))`` in physical temperature units."""

    values = np.asarray(list(temperatures), dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("smooth_peak_temperature requires nonempty finite temperatures.")
    if tau <= 0.0 or not math.isfinite(float(tau)):
        raise ValueError("tau must be finite and positive.")
    scaled = values / float(tau)
    maximum = float(np.max(scaled))
    return float(float(tau) * (maximum + math.log(float(np.mean(np.exp(scaled - maximum))))))


def relative_pressure_limit(baseline_drop: float, *, allowance_fraction: float = 0.05) -> float:
    """Predeclare a benchmark-relative pressure budget from a positive baseline.

    The returned limit is a benchmark constraint, not an engineering safety
    limit.  A nonpositive baseline is rejected because a ratio would have an
    ambiguous physical meaning under the maintained inlet-minus-outlet sign.
    """

    baseline = _finite_scalar(baseline_drop, name="baseline_drop")
    allowance = _finite_scalar(allowance_fraction, name="allowance_fraction")
    if baseline <= 0.0:
        raise ValueError("A relative pressure-drop budget requires positive baseline inlet-minus-outlet pressure.")
    if allowance < 0.0:
        raise ValueError("allowance_fraction must be nonnegative.")
    return baseline * (1.0 + allowance)


def _factor_id(factor: Any) -> str:
    value = getattr(factor, "factor_id", None)
    if value is None:
        value = getattr(factor, "edge_id", None)
    if value is None:
        raise ValueError("Every response factor must expose factor_id (or edge_id).")
    return str(value)


def _factor_sources(factor: Any) -> tuple[Hashable, ...]:
    value = getattr(factor, "source_module_ids", None)
    if value is None:
        value = getattr(factor, "donor_ids", None)
    if value is None:
        raise ValueError("Every response factor must expose physical source_module_ids/donor_ids.")
    sources = tuple(value)
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("Factor source IDs must be nonempty and unique.")
    return sources


def _rank_baseline_factors(
    oracle: AnchoredResponseOracle,
    prepared: PreparedResponseBaseline,
    *,
    candidate_budget: int,
) -> tuple[Any, ...]:
    """Use an input-only support scorer; never fall back to incidental order."""

    factors = tuple(prepared.factors)
    if not factors:
        return ()
    ranker = getattr(oracle, "rank_factors", None)
    if callable(ranker):
        ranked = tuple(ranker(prepared))
    elif len(factors) <= candidate_budget:
        ranked = tuple(sorted(factors, key=_factor_id))
    else:
        raise ValueError(
            "More baseline factors than candidate slots require an input-only baseline factor ranker."
        )
    factor_ids = {_factor_id(factor) for factor in factors}
    ranked_ids = [_factor_id(factor) for factor in ranked]
    if len(ranked_ids) != len(set(ranked_ids)) or set(ranked_ids) != factor_ids:
        raise ValueError("rank_factors must return every baseline factor exactly once.")
    return ranked


def _ids_by_slot(
    design: PhysicalDesign,
    module_ids_by_slot: Sequence[Hashable | None],
) -> tuple[Hashable, ...]:
    if len(module_ids_by_slot) != design.max_modules:
        raise ValueError("module_ids_by_slot must have one entry for every padded design slot.")
    active = design.module_present > 0.5
    ids: list[Hashable] = []
    for slot, (is_active, module_id) in enumerate(zip(active, module_ids_by_slot)):
        if is_active and module_id is None:
            raise ValueError(f"Active module slot {slot} is missing its physical ID.")
        if not is_active and module_id is not None:
            raise ValueError(f"Inactive module slot {slot} must not carry a physical ID.")
        if is_active:
            ids.append(module_id)  # type: ignore[arg-type]
    if len(ids) != len(set(ids)):
        raise ValueError("Active physical module IDs must be unique.")
    return tuple(ids)


def _slot_by_id(
    design: PhysicalDesign,
    module_ids_by_slot: Sequence[Hashable | None],
) -> dict[Hashable, int]:
    return {
        module_id: slot
        for slot, module_id in enumerate(module_ids_by_slot)
        if design.module_present[slot] > 0.5 and module_id is not None
    }


def _moved_design(
    baseline: PhysicalDesign,
    module_ids_by_slot: Sequence[Hashable | None],
    delta_by_module_id: Mapping[Hashable, np.ndarray],
) -> PhysicalDesign:
    slot_by_id = _slot_by_id(baseline, module_ids_by_slot)
    centers = baseline.module_centers.copy()
    for module_id, delta in delta_by_module_id.items():
        if module_id not in slot_by_id:
            raise ValueError(f"Position update references unknown module ID {module_id!r}.")
        move = np.asarray(delta, dtype=np.float64).reshape(-1)
        if move.shape != (2,) or not np.isfinite(move).all():
            raise ValueError("Each module position update must be a finite [dx,dy] pair.")
        centers[slot_by_id[module_id]] += move.astype(np.float32)
    return PhysicalDesign(
        module_centers=centers,
        module_present=baseline.module_present.copy(),
        heat_powers=baseline.heat_powers.copy(),
        module_family_id=baseline.module_family_id,
    )


def _normalized_direction_bank(
    *,
    candidate_count: int,
    dimension: int,
    radius: float,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    directions = generator.standard_normal((candidate_count, dimension))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms <= 0.0):  # pragma: no cover - Gaussian draw guard.
        raise RuntimeError("Unable to construct nonzero normalized candidate directions.")
    return directions / norms * float(radius)


def _candidate_groups(
    *,
    policy: str,
    factors: Sequence[Any],
    size_schedule: Sequence[int],
    active_ids: Sequence[Hashable],
    seed: int,
) -> tuple[tuple[tuple[Hashable, ...], str | None], ...]:
    """Choose same-cardinality blocks for all policies at each candidate index."""

    canonical_ids = tuple(sorted(active_ids, key=_id_key))
    if policy == "graph_guided":
        valid: list[tuple[tuple[Hashable, ...], str]] = []
        active = set(active_ids)
        for factor in factors:
            sources = _factor_sources(factor)
            if not set(sources).issubset(active):
                raise ValueError(f"Factor {_factor_id(factor)!r} names modules outside the active baseline.")
            valid.append((tuple(sorted(sources, key=_id_key)), _factor_id(factor)))
        if not valid:
            return tuple((tuple(canonical_ids[:size]), None) for size in size_schedule)
        output: list[tuple[tuple[Hashable, ...], str | None]] = []
        cursors: dict[int, int] = {}
        for size in size_schedule:
            matches = [item for item in valid if len(item[0]) == size]
            if not matches:
                # A baseline may have no edge at one of the scheduled orders;
                # use a declared unary/local source block and mark no factor.
                output.append((tuple(canonical_ids[:size]), None))
                continue
            cursor = cursors.get(size, 0)
            source, factor_id = matches[cursor % len(matches)]
            cursors[size] = cursor + 1
            output.append((source, factor_id))
        return tuple(output)

    generator = np.random.default_rng(seed)
    output = []
    for candidate_index, size in enumerate(size_schedule):
        if policy == "size_matched_random":
            indices = generator.choice(len(canonical_ids), size=size, replace=False)
            group = tuple(sorted((canonical_ids[int(index)] for index in indices), key=_id_key))
        elif policy == "ungrouped_local":
            offset = candidate_index % len(canonical_ids)
            group = tuple(canonical_ids[(offset + index) % len(canonical_ids)] for index in range(size))
            group = tuple(sorted(group, key=_id_key))
        else:  # pragma: no cover - internal caller guard.
            raise ValueError(f"Unknown policy {policy!r}.")
        output.append((group, None))
    return tuple(output)


def _matched_size_schedule(
    factors: Sequence[Any], *, candidate_budget: int, active_module_count: int
) -> tuple[int, ...]:
    """Predeclare balanced unary/pair proposal orders from baseline support.

    When both unary and pair factors exist, the schedule alternates pair and
    unary moves, giving the pair order the extra slot for odd budgets.  The
    order schedule is frozen before any candidate is evaluated and shared by
    graph, random, and ungrouped policies.  Other factor cardinalities are
    retained when no unary/pair comparison is available.
    """

    if candidate_budget <= 0 or active_module_count <= 0:
        raise ValueError("A positive candidate budget and active module count are required.")
    available = sorted({len(_factor_sources(factor)) for factor in factors})
    if any(size > active_module_count for size in available):
        raise ValueError("A response factor is larger than the active design.")
    if 1 in available and 2 in available:
        # Pair proposals occur first and receive the extra slot when needed.
        base = (2, 1)
    elif available:
        base = tuple(sorted(available, key=lambda size: (size != 2, -size)))
    else:
        base = (1,)
    schedule = tuple(base[index % len(base)] for index in range(candidate_budget))
    if 1 in available and 2 in available:
        pair_count = sum(size == 2 for size in schedule)
        unary_count = sum(size == 1 for size in schedule)
        if pair_count < unary_count or pair_count - unary_count > 1:
            raise RuntimeError("Matched unary/pair allocation is not balanced.")
    return schedule


def _predict_candidate(
    *,
    oracle: AnchoredResponseOracle,
    prepared: PreparedResponseBaseline,
    baseline: DecisionObservation,
    candidate_design: PhysicalDesign,
    context: NamedContext,
    delta_by_module_id: Mapping[Hashable, np.ndarray],
    changed_ids: set[Hashable],
    selected_factor_id: str | None,
    tau_temperature: float,
    decision_decoder: Callable[
        [DecisionObservation, Mapping[str, np.ndarray], PhysicalDesign, NamedContext], DecisionEstimate
    ],
) -> tuple[DecisionEstimate, float | None, int, str]:
    factor_parts: list[Mapping[str, Any]] = []
    factor_part_ids: list[str] = []
    selected_part: Mapping[str, Any] | None = None
    calls = 0
    validity_results: list[bool | None] = []
    validity_checker = getattr(oracle, "factor_validity_status", None)
    for factor in prepared.factors:
        sources = _factor_sources(factor)
        # A factor wholly outside the trial support is identically zero.  For
        # partial overlap, pass only its physically declared donors so the
        # anchored operator supplies its exact zero when a required donor is
        # unperturbed.
        if not set(sources).issubset(changed_ids):
            continue
        # Explicit zeros for unchanged donors make the anchored inclusion-
        # exclusion identity visible at the adapter boundary.
        zero = np.zeros(2, dtype=np.float32)
        factor_delta = {
            module_id: delta_by_module_id.get(module_id, zero)
            for module_id in sources
        }
        validity_results.append(
            validity_checker(prepared, factor, factor_delta) if callable(validity_checker) else None
        )
        part = oracle.predict_factor_response(prepared, factor, factor_delta)
        calls += 1
        factor_parts.append(part)
        factor_part_ids.append(_factor_id(factor))
        if _factor_id(factor) == selected_factor_id:
            selected_part = part
    total = _sum_typed_responses(factor_parts)
    estimate = decision_decoder(baseline, total, candidate_design, context)
    if set(estimate.module_temperature_by_id) != set(baseline.module_temperature_by_id):
        raise ValueError("Predicted module temperatures must preserve all physical module IDs.")
    if estimate.pressure_drop_units != baseline.pressure_drop_units:
        raise ValueError("Predicted and baseline pressure-drop units differ.")

    selected_contribution = None
    if selected_part is not None:
        selected_part_index = factor_part_ids.index(selected_factor_id)
        if selected_part_index is not None:
            total_without: dict[str, np.ndarray] = {}
            for index, part in enumerate(factor_parts):
                if index == selected_part_index:
                    continue
                for role, raw in part.items():
                    values = _as_numpy(raw, name=f"factor response without {factor_part_ids[index]}:{role}")
                    if role in total_without:
                        total_without[role] = total_without[role] + values
                    else:
                        total_without[role] = values.copy()
            estimate_without = decision_decoder(baseline, total_without, candidate_design, context)
            selected_contribution = (
                smooth_peak_temperature(estimate_without.module_temperature_by_id.values(), tau=tau_temperature)
                - smooth_peak_temperature(estimate.module_temperature_by_id.values(), tau=tau_temperature)
            )
    if any(value is False for value in validity_results):
        evidence_validity_status = "outside_measured_neighborhood"
    elif not validity_results or any(value is None for value in validity_results):
        evidence_validity_status = "unknown"
    else:
        evidence_validity_status = "within_measured_neighborhood"
    return estimate, selected_contribution, calls, evidence_validity_status


def _build_candidate_pool(
    *,
    policy: str,
    policy_index: int,
    update_index: int,
    design: PhysicalDesign,
    baseline: DecisionObservation,
    context: NamedContext,
    module_ids_by_slot: Sequence[Hashable | None],
    geometry_constraints: GeometryConstraints,
    config: InverseStudyConfig,
    prepared: PreparedResponseBaseline,
    size_schedule: Sequence[int],
    active_ids: Sequence[Hashable],
    domain_lengths: tuple[float, float],
    pressure_scale: float,
    response_oracle: AnchoredResponseOracle,
    decision_decoder: Callable[
        [DecisionObservation, Mapping[str, np.ndarray], PhysicalDesign, NamedContext], DecisionEstimate
    ],
    direction_seed_offsets: Mapping[int, int] | None = None,
    candidate_indices: Sequence[int] | None = None,
) -> tuple[list[_PreparedCandidate], int, int]:
    """Build and factor-score the same sized candidate pool for one policy."""

    lx, ly = domain_lengths
    groups = _candidate_groups(
        policy=policy,
        factors=prepared.factors,
        size_schedule=size_schedule,
        active_ids=active_ids,
        seed=config.random_seed + 1009 * policy_index + update_index,
    )
    proposals: list[_PreparedCandidate] = []
    factor_calls = 0
    geometry_rejections = 0
    selected_indices = None if candidate_indices is None else {int(value) for value in candidate_indices}
    seed_offsets = direction_seed_offsets or {}
    for candidate_index, ((group, factor_id), size) in enumerate(zip(groups, size_schedule)):
        if selected_indices is not None and candidate_index not in selected_indices:
            continue
        seed_offset = int(seed_offsets.get(candidate_index, 0))
        if seed_offset < 0:
            raise ValueError("Direction seed offsets must be nonnegative.")
        direction = _normalized_direction_bank(
            candidate_count=1,
            dimension=2 * size,
            radius=config.trust_radius_normalized,
            seed=(
                config.random_seed
                + 7919 * update_index
                + 97 * candidate_index
                + config.candidate_redraw_seed_offset_stride * seed_offset
            ),
        )[0]
        delta_by_id: dict[Hashable, np.ndarray] = {}
        for offset, module_id in enumerate(group):
            normalized_xy = direction[2 * offset : 2 * offset + 2]
            delta_by_id[module_id] = np.asarray(
                [normalized_xy[0] * lx, normalized_xy[1] * ly], dtype=np.float32
            )
        trial_design = _moved_design(design, module_ids_by_slot, delta_by_id)
        if trial_design.module_count != design.module_count or not np.array_equal(
            trial_design.module_present, design.module_present
        ):
            raise RuntimeError("Position-only update changed module topology.")
        if not np.array_equal(trial_design.heat_powers, design.heat_powers):
            raise RuntimeError("Position-only update changed fixed module heating.")
        normalized_radius = float(
            np.linalg.norm(
                np.asarray(
                    [[delta_by_id[mid][0] / lx, delta_by_id[mid][1] / ly] for mid in group],
                    dtype=np.float64,
                )
            )
        )
        candidate_id = f"{policy}_u{update_index:02d}_c{candidate_index:02d}"
        graph_factor_status = (
            "active"
            if policy == "graph_guided" and factor_id is not None
            else "missing_order_fallback"
            if policy == "graph_guided"
            else "not_applicable"
        )
        geometry = evaluate_geometry(trial_design, context, geometry_constraints)
        if not geometry.valid:
            geometry_rejections += 1
            record = CandidateRecord(
                policy=policy,
                candidate_id=candidate_id,
                update_index=update_index,
                source_module_ids=tuple(group),
                factor_id=factor_id,
                normalized_dimension=2 * size,
                normalized_radius=normalized_radius,
                delta_by_module_id={key: tuple(map(float, value)) for key, value in delta_by_id.items()},
                geometry_valid=False,
                predicted_feasible=None,
                predicted_objective=None,
                predicted_true_peak=None,
                predicted_pressure_drop=None,
                predicted_pressure_violation=None,
                selected_factor_objective_contribution=None,
                score_status="geometry_rejected_before_model",
                evidence_validity_status="not_scored",
                graph_factor_status=graph_factor_status,
                pressure_prediction_status="not_scored",
            )
            proposals.append(_PreparedCandidate(record, trial_design, None, None, None))
            continue

        estimate, factor_contribution, call_count, evidence_validity_status = _predict_candidate(
            oracle=response_oracle,
            prepared=prepared,
            baseline=baseline,
            candidate_design=trial_design,
            context=context,
            delta_by_module_id=delta_by_id,
            changed_ids=set(group),
            selected_factor_id=factor_id if policy == "graph_guided" else None,
            tau_temperature=config.tau_temperature,
            decision_decoder=decision_decoder,
        )
        factor_calls += call_count
        predicted_objective = smooth_peak_temperature(
            estimate.module_temperature_by_id.values(), tau=config.tau_temperature
        )
        predicted_feasible, predicted_violation = _geometry_violation(
            trial_design,
            context,
            geometry_constraints,
            pressure_drop=estimate.pressure_drop,
            pressure_limit=config.pressure_drop_limit,
            pressure_scale=pressure_scale,
        )
        pressure_supported = not estimate.pressure_section_mask_changed
        record = CandidateRecord(
            policy=policy,
            candidate_id=candidate_id,
            update_index=update_index,
            source_module_ids=tuple(group),
            factor_id=factor_id,
            normalized_dimension=2 * size,
            normalized_radius=normalized_radius,
            delta_by_module_id={key: tuple(map(float, value)) for key, value in delta_by_id.items()},
            geometry_valid=True,
            predicted_feasible=predicted_feasible if pressure_supported else None,
            predicted_objective=predicted_objective,
            predicted_true_peak=max(estimate.module_temperature_by_id.values()),
            predicted_pressure_drop=estimate.pressure_drop if pressure_supported else None,
            predicted_pressure_violation=(
                max(estimate.pressure_drop - config.pressure_drop_limit, 0.0) if pressure_supported else None
            ),
            selected_factor_objective_contribution=factor_contribution,
            score_status=(
                "scored_from_factor_responses"
                if pressure_supported
                else "unsupported_trial_section_geometry"
            ),
            predicted_module_temperature_by_id=dict(estimate.module_temperature_by_id),
            evidence_validity_status=evidence_validity_status,
            graph_factor_status=graph_factor_status,
            pressure_prediction_status=(
                "candidate_section_mask_changed"
                if estimate.pressure_section_mask_changed
                else "fixed_baseline_sections"
            ),
        )
        proposals.append(
            _PreparedCandidate(
                record,
                trial_design,
                estimate,
                predicted_objective,
                predicted_violation if pressure_supported else None,
            )
        )
    return proposals, factor_calls, geometry_rejections


def _candidate_is_geometry_and_pressure_supported(candidate: _PreparedCandidate) -> bool:
    return bool(
        candidate.estimate is not None
        and candidate.record.geometry_valid
        and candidate.record.pressure_prediction_status == "fixed_baseline_sections"
    )


def _build_candidate_pool_with_feasibility_redraw(
    **kwargs: Any,
) -> tuple[list[_PreparedCandidate], int, int, list[dict[str, Any]]]:
    """Keep each slot's source block and radius fixed while redrawing direction.

    The initial group assignment is created once. Unsupported slots are retried
    independently with deterministic direction-seed offsets; no response
    outcome, reference solve, or trial-design field query is used to decide.
    The last attempt remains in the returned slot when all bounded retries fail
    so the caller can report its unsupported status and stop before evaluation.
    """

    config = kwargs["config"]
    max_offsets = int(config.candidate_redraw_max_seed_offsets)
    stride = int(config.candidate_redraw_seed_offset_stride)
    initial, factor_calls, geometry_rejections = _build_candidate_pool(**kwargs)
    slots = {index: candidate for index, candidate in enumerate(initial)}
    audit: list[dict[str, Any]] = []

    for index in range(len(initial)):
        candidate = slots[index]
        attempts: list[dict[str, Any]] = []

        def log_attempt(
            value: _PreparedCandidate,
            seed_offset: int,
            *,
            attempt_rows: list[dict[str, Any]] = attempts,
            slot_index: int = index,
        ) -> None:
            attempt_rows.append(
                {
                    "seed_offset": seed_offset,
                    "direction_seed": (
                        int(config.random_seed)
                        + 7919 * int(kwargs["update_index"])
                        + 97 * slot_index
                        + stride * seed_offset
                    ),
                    "candidate_id": value.record.candidate_id,
                    "factor_id": value.record.factor_id,
                    "source_module_ids": list(value.record.source_module_ids),
                    "normalized_dimension": value.record.normalized_dimension,
                    "normalized_radius": value.record.normalized_radius,
                    "geometry_valid": value.record.geometry_valid,
                    "pressure_prediction_status": value.record.pressure_prediction_status,
                    "supported": _candidate_is_geometry_and_pressure_supported(value),
                }
            )

        log_attempt(candidate, 0)
        selected_offset: int | None = 0 if _candidate_is_geometry_and_pressure_supported(candidate) else None
        for seed_offset in range(1, max_offsets + 1):
            if selected_offset is not None:
                break
            retry, retry_calls, retry_rejections = _build_candidate_pool(
                **kwargs,
                direction_seed_offsets={index: seed_offset},
                candidate_indices=(index,),
            )
            factor_calls += retry_calls
            geometry_rejections += retry_rejections
            if len(retry) != 1:
                raise RuntimeError("Candidate direction redraw did not return exactly its requested slot.")
            candidate = retry[0]
            slots[index] = candidate
            log_attempt(candidate, seed_offset)
            if _candidate_is_geometry_and_pressure_supported(candidate):
                selected_offset = seed_offset
        audit.append(
            {
                "candidate_id": slots[index].record.candidate_id,
                "slot_index": index,
                "group": list(slots[index].record.source_module_ids),
                "factor_id": slots[index].record.factor_id,
                "selected_seed_offset": selected_offset,
                "status": "selected" if selected_offset is not None else "redraw_exhausted",
                "attempts": attempts,
            }
        )

    return [slots[index] for index in range(len(initial))], factor_calls, geometry_rejections, audit


def _geometry_violation(
    design: PhysicalDesign,
    context: NamedContext,
    constraints: GeometryConstraints,
    *,
    pressure_drop: float,
    pressure_limit: float,
    pressure_scale: float,
) -> tuple[bool, float]:
    geometry = evaluate_geometry(design, context, constraints)
    pressure_violation = max(float(pressure_drop) - float(pressure_limit), 0.0) / max(float(pressure_scale), 1.0e-12)
    total = float(geometry.total_violation) + pressure_violation
    return bool(geometry.valid and pressure_drop <= pressure_limit), total


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None

    def ranks(values: Sequence[float]) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        order = np.argsort(data, kind="mergesort")
        result = np.empty(data.shape, dtype=np.float64)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and data[order[stop]] == data[order[start]]:
                stop += 1
            result[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
            start = stop
        return result

    x = ranks(left)
    y = ranks(right)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _trial_feasible(
    design: PhysicalDesign,
    observation: DecisionObservation,
    context: NamedContext,
    constraints: GeometryConstraints,
    *,
    pressure_limit: float,
) -> tuple[bool, bool]:
    geometry_valid = bool(evaluate_geometry(design, context, constraints).valid)
    return geometry_valid, bool(geometry_valid and observation.pressure_drop <= pressure_limit)


def _should_accept(
    baseline: DecisionObservation,
    trial: DecisionObservation,
    baseline_design: PhysicalDesign,
    trial_design: PhysicalDesign,
    context: NamedContext,
    constraints: GeometryConstraints,
    *,
    pressure_limit: float,
    tau: float,
    resolution: float,
) -> bool:
    _, baseline_feasible = _trial_feasible(
        baseline_design, baseline, context, constraints, pressure_limit=pressure_limit
    )
    _, trial_feasible = _trial_feasible(
        trial_design, trial, context, constraints, pressure_limit=pressure_limit
    )
    baseline_j = smooth_peak_temperature(baseline.module_temperature_by_id.values(), tau=tau)
    trial_j = smooth_peak_temperature(trial.module_temperature_by_id.values(), tau=tau)
    if trial_feasible and not baseline_feasible:
        return True
    if baseline_feasible and not trial_feasible:
        return False
    if trial_feasible and baseline_feasible:
        return trial_j < baseline_j - resolution
    baseline_violation = _geometry_violation(
        baseline_design,
        context,
        constraints,
        pressure_drop=baseline.pressure_drop,
        pressure_limit=pressure_limit,
        pressure_scale=max(abs(baseline.pressure_drop), 1.0),
    )[1]
    trial_violation = _geometry_violation(
        trial_design,
        context,
        constraints,
        pressure_drop=trial.pressure_drop,
        pressure_limit=pressure_limit,
        pressure_scale=max(abs(baseline.pressure_drop), 1.0),
    )[1]
    return trial_violation < baseline_violation - resolution


def run_matched_inverse_design_study(
    *,
    initial_design: PhysicalDesign,
    initial_observation: DecisionObservation,
    context: NamedContext,
    module_ids_by_slot: Sequence[Hashable | None],
    geometry_constraints: GeometryConstraints,
    config: InverseStudyConfig,
    response_oracle: AnchoredResponseOracle,
    evaluator: Callable[[PhysicalDesign, NamedContext, Sequence[Hashable | None]], DecisionObservation],
    evaluator_source: EvidenceSource,
    decision_decoder: Callable[
        [DecisionObservation, Mapping[str, np.ndarray], PhysicalDesign, NamedContext], DecisionEstimate
    ] = decode_thermal_field_response,
) -> MatchedInverseStudyResult:
    """Run matched graph, random, and ungrouped local proposal policies.

    All three policies use the same baseline response model, objective, trust
    radius, proposal count, and per-candidate coordinate dimension. Graph
    proposals use top baseline input-support-ranked source sets within each
    predeclared order stratum. Each displaced candidate is then scored by its
    full factor-summed thermal objective, pressure feasibility, and selected
    interaction contribution. This is a bounded finite proposal pool, not a
    global response-objective search over every active edge. Random proposals
    sample blocks of the same sizes. Ungrouped proposals use deterministic
    local blocks with independent per-module directions. ``exhaustive_pool``
    validates every proposal at the current baseline and is useful for
    candidate-ranking and finite-pool measurements; ``selected_only`` spends
    one evaluator call on the response model's top feasible proposal and can
    continue for up to the declared trial budget.

    Calls to ``evaluator`` are charged before invocation, so exceptions and
    failed solver trials count against the same budget.  The evidence source
    of the evaluator is explicit and remains visible in every trial record.
    """

    if evaluator_source not in EVIDENCE_SOURCES:
        raise ValueError(f"Unsupported evaluator source {evaluator_source!r}.")
    if evaluator_source == "stored_reference":
        raise ValueError("stored_reference is allowed for an initial baseline, not a newly perturbed-design evaluator.")
    if initial_observation.evidence_source not in EVIDENCE_SOURCES:
        raise ValueError("Initial observation has an unsupported evidence source.")
    if initial_observation.pressure_drop_units != config.pressure_drop_units:
        raise ValueError("Initial pressure-drop units do not match the predeclared constraint units.")
    ids_by_slot = tuple(module_ids_by_slot)
    active_ids = _ids_by_slot(initial_design, ids_by_slot)
    if set(active_ids) != set(initial_observation.module_temperature_by_id):
        raise ValueError("Initial measured module temperatures must match active physical module IDs.")
    if geometry_constraints.module_count_min > initial_design.module_count or geometry_constraints.module_count_max < initial_design.module_count:
        raise ValueError("The fixed-topology design does not satisfy the supplied module-count bounds.")
    radius = context.as_mapping()["module_radius"]
    if geometry_constraints.minimum_center_distance < 2.0 * radius - 1.0e-8:
        raise ValueError("minimum_center_distance must prevent disk overlap (at least two module radii).")
    if evaluator_source in PHYSICAL_EVIDENCE_SOURCES and config.pressure_limit_basis.startswith("teacher"):
        raise ValueError("A teacher-derived pressure budget cannot be described as physical acceptance.")

    lx = float(context.as_mapping()["domain_length_x"])
    ly = float(context.as_mapping()["domain_length_y"])
    pressure_scale = max(abs(initial_observation.pressure_drop), 1.0e-12)
    initial_prepared = response_oracle.prepare_baseline(initial_design, context, ids_by_slot)
    ranked_initial_factors = _rank_baseline_factors(
        response_oracle, initial_prepared, candidate_budget=config.candidate_budget
    )
    initial_prepared = PreparedResponseBaseline(
        cache=initial_prepared.cache,
        queries=initial_prepared.queries,
        factors=ranked_initial_factors,
        metadata={**initial_prepared.metadata, "factor_rank_source": "baseline_input_only"},
    )
    initial_factors = tuple(initial_prepared.factors)
    active_set = set(active_ids)
    for factor in initial_factors:
        if not set(_factor_sources(factor)).issubset(active_set):
            raise ValueError(f"Initial factor {_factor_id(factor)!r} names inactive/unknown modules.")
    size_schedule_for_result = _matched_size_schedule(
        initial_factors,
        candidate_budget=config.candidate_budget,
        active_module_count=len(active_ids),
    )
    if any(size > len(active_ids) for size in size_schedule_for_result):
        raise ValueError("Initial response factor is larger than the active design.")
    policy_results: dict[str, PolicyResult] = {}
    initial_response_oracle = response_oracle

    for policy_index, policy in enumerate(POLICY_NAMES):
        # Policies are independent matched experiments from the same accepted
        # starting point. A re-anchored oracle from one policy must not leak
        # into another policy's baseline.
        policy_oracle = initial_response_oracle
        current_design = initial_design
        current_observation = initial_observation
        candidates_seen: list[CandidateRecord] = []
        trials_seen: list[TrialRecord] = []
        accepted_updates = 0
        evaluator_calls = 0
        physical_calls = 0
        recovered_reference_calls = 0
        unknown_solver_time_calls = 0
        physical_solver_seconds = 0.0
        teacher_calls = 0
        synthetic_calls = 0
        factor_calls = 0
        geometry_rejections = 0
        candidate_redraw_audit: list[Mapping[str, Any]] = []
        _, initial_feasible = _trial_feasible(
            initial_design,
            initial_observation,
            context,
            geometry_constraints,
            pressure_limit=config.pressure_drop_limit,
        )
        best_eval_design: PhysicalDesign | None = initial_design if initial_feasible else None
        best_eval_observation: DecisionObservation | None = initial_observation if initial_feasible else None
        best_physical_design: PhysicalDesign | None = (
            initial_design
            if initial_feasible and initial_observation.evidence_source in PHYSICAL_EVIDENCE_SOURCES
            else None
        )
        best_physical_observation: DecisionObservation | None = (
            initial_observation
            if initial_feasible and initial_observation.evidence_source in PHYSICAL_EVIDENCE_SOURCES
            else None
        )
        stop_status = "reference_budget_exhausted"
        update_index = 0

        while update_index < config.max_updates_per_policy and evaluator_calls < config.max_reference_trials_per_policy:
            if current_observation.evidence_source not in EVIDENCE_SOURCES:
                raise ValueError("Current baseline evidence source became invalid.")
            prepared = (
                initial_prepared
                if update_index == 0
                else policy_oracle.prepare_baseline(current_design, context, ids_by_slot)
            )
            if update_index > 0:
                ranked_factors = _rank_baseline_factors(
                    policy_oracle, prepared, candidate_budget=config.candidate_budget
                )
                prepared = PreparedResponseBaseline(
                    cache=prepared.cache,
                    queries=prepared.queries,
                    factors=ranked_factors,
                    metadata={**prepared.metadata, "factor_rank_source": "baseline_input_only"},
                )
            if policy not in POLICY_NAMES:  # pragma: no cover - fixed policy list guard.
                raise RuntimeError(f"Unexpected inverse policy {policy!r}.")
            # The cardinality schedule is frozen from the initial graph.  At a
            # later accepted baseline graph factors may change identities, but
            # each policy still gets the same proposal dimension and trust
            # radius at the same candidate index.
            size_schedule = size_schedule_for_result
            if len(size_schedule) != config.candidate_budget:
                raise RuntimeError("Matched candidate size schedule has the wrong length.")
            if any(size <= 0 or size > len(active_ids) for size in size_schedule):
                raise ValueError("A baseline factor has a source set larger than the active design.")

            (
                prepared_candidates,
                step_factor_calls,
                step_geometry_rejections,
                step_redraw_audit,
            ) = _build_candidate_pool_with_feasibility_redraw(
                policy=policy,
                policy_index=policy_index,
                update_index=update_index,
                design=current_design,
                baseline=current_observation,
                context=context,
                module_ids_by_slot=ids_by_slot,
                geometry_constraints=geometry_constraints,
                config=config,
                prepared=prepared,
                size_schedule=size_schedule,
                active_ids=active_ids,
                domain_lengths=(lx, ly),
                pressure_scale=pressure_scale,
                response_oracle=policy_oracle,
                decision_decoder=decision_decoder,
            )
            factor_calls += step_factor_calls
            geometry_rejections += step_geometry_rejections
            candidate_redraw_audit.extend(step_redraw_audit)
            candidates_seen.extend(item.record for item in prepared_candidates)

            if any(not _candidate_is_geometry_and_pressure_supported(item) for item in prepared_candidates):
                if not any(_candidate_is_geometry_and_pressure_supported(item) for item in prepared_candidates):
                    stop_status = (
                        "no_supported_pressure_candidates"
                        if any(
                            item.record.pressure_prediction_status == "candidate_section_mask_changed"
                            for item in prepared_candidates
                        )
                        else "no_geometry_feasible_candidates"
                    )
                else:
                    stop_status = "candidate_pool_redraw_exhausted"
                break

            scored = [
                item
                for item in prepared_candidates
                if item.estimate is not None
                and item.record.pressure_prediction_status == "fixed_baseline_sections"
            ]
            if not scored:
                stop_status = (
                    "no_supported_pressure_candidates"
                    if any(
                        item.record.pressure_prediction_status == "candidate_section_mask_changed"
                        for item in prepared_candidates
                    )
                    else "no_geometry_feasible_candidates"
                )
                break
            # Constraint filter first; predicted temperature objective breaks
            # ties among candidates inside the feasible set.
            selected = min(
                scored,
                key=lambda item: (
                    0 if item.record.pressure_prediction_status == "fixed_baseline_sections" else 1,
                    0 if item.record.predicted_feasible else 1,
                    float(item.predicted_violation if item.predicted_violation is not None else math.inf),
                    float(item.predicted_objective if item.predicted_objective is not None else math.inf),
                    -float(
                        item.record.selected_factor_objective_contribution
                        if item.record.selected_factor_objective_contribution is not None
                        else -math.inf
                    ),
                    item.record.candidate_id,
                ),
            )

            if config.validation_mode == "exhaustive_pool":
                to_evaluate = scored
            else:
                to_evaluate = [selected]
            if evaluator_calls + len(to_evaluate) > config.max_reference_trials_per_policy:
                to_evaluate = to_evaluate[: max(config.max_reference_trials_per_policy - evaluator_calls, 0)]
            if not to_evaluate:
                stop_status = "reference_budget_exhausted"
                break

            observed_by_candidate: dict[str, DecisionObservation] = {}
            trial_by_candidate: dict[str, TrialRecord] = {}
            current_j = smooth_peak_temperature(
                list(current_observation.module_temperature_by_id.values()), tau=config.tau_temperature
            )
            for item in to_evaluate:
                evaluator_calls += 1  # failed calls consume the budget too
                started = time.perf_counter()
                try:
                    policy_aware_evaluator = getattr(evaluator, "evaluate_for_policy", None)
                    if callable(policy_aware_evaluator):
                        actual = policy_aware_evaluator(policy, item.design, context, ids_by_slot)
                    else:
                        actual = evaluator(item.design, context, ids_by_slot)
                    wall_elapsed = time.perf_counter() - started
                    if actual.evidence_source != evaluator_source:
                        raise ValueError(
                            f"Evaluator declared {evaluator_source!r} but returned {actual.evidence_source!r}."
                        )
                    if actual.pressure_drop_units != config.pressure_drop_units:
                        raise ValueError("Evaluator pressure-drop units differ from the predeclared constraint.")
                    if set(actual.module_temperature_by_id) != set(active_ids):
                        raise ValueError("Evaluator must preserve active physical module IDs at every trial.")
                    if evaluator_source == "reference_solver":
                        physical_calls += 1
                        solver_time_available = bool(
                            actual.provenance.get("physical_wall_time_available", True)
                        )
                        if solver_time_available:
                            physical_solver_seconds += actual.elapsed_seconds
                        else:
                            unknown_solver_time_calls += 1
                            if actual.provenance.get("raw_recovered_after_adapter_error", False):
                                recovered_reference_calls += 1
                    elif evaluator_source == "surrogate_teacher":
                        teacher_calls += 1
                    elif evaluator_source == "analytic_synthetic":
                        synthetic_calls += 1
                    actual_j = smooth_peak_temperature(
                        list(actual.module_temperature_by_id.values()), tau=config.tau_temperature
                    )
                    geometry_valid, feasible = _trial_feasible(
                        item.design,
                        actual,
                        context,
                        geometry_constraints,
                        pressure_limit=config.pressure_drop_limit,
                    )
                    predicted_improvement = current_j - float(item.predicted_objective)
                    actual_improvement = current_j - actual_j
                    ratio = (
                        actual_improvement / predicted_improvement
                        if predicted_improvement > config.decision_resolution
                        else None
                    )
                    is_selected = item.record.candidate_id == selected.record.candidate_id
                    accepted = bool(
                        is_selected
                        and _should_accept(
                            current_observation,
                            actual,
                            current_design,
                            item.design,
                            context,
                            geometry_constraints,
                            pressure_limit=config.pressure_drop_limit,
                            tau=config.tau_temperature,
                            resolution=config.decision_resolution,
                        )
                    )
                    record = TrialRecord(
                        policy=policy,
                        candidate_id=item.record.candidate_id,
                        evaluator_source=evaluator_source,
                        call_index=evaluator_calls,
                        status="completed",
                        elapsed_seconds=max(wall_elapsed, actual.elapsed_seconds),
                        solver_elapsed_seconds=(
                            actual.elapsed_seconds
                            if evaluator_source == "reference_solver" and solver_time_available
                            else None
                        ),
                        solver_elapsed_available=(
                            evaluator_source == "reference_solver" and solver_time_available
                        ),
                        raw_recovered_after_adapter_error=bool(
                            actual.provenance.get("raw_recovered_after_adapter_error", False)
                        ),
                        actual_objective=actual_j,
                        actual_true_peak=actual.true_peak_temperature,
                        actual_module_temperature_by_id=dict(actual.module_temperature_by_id),
                        actual_pressure_drop=actual.pressure_drop,
                        geometry_valid=geometry_valid,
                        feasible=feasible,
                        accepted=accepted,
                        physical_acceptance=accepted if evaluator_source in PHYSICAL_SOLVER_SOURCES else None,
                        predicted_improvement=predicted_improvement,
                        actual_improvement=actual_improvement,
                        improvement_ratio=ratio,
                    )
                    observed_by_candidate[item.record.candidate_id] = actual
                    if feasible and (
                        best_eval_observation is None
                        or actual_j < smooth_peak_temperature(
                            list(best_eval_observation.module_temperature_by_id.values()),
                            tau=config.tau_temperature,
                        )
                    ):
                        best_eval_design = item.design
                        best_eval_observation = actual
                        if evaluator_source in PHYSICAL_SOLVER_SOURCES:
                            best_physical_design = item.design
                            best_physical_observation = actual
                    trials_seen.append(record)
                    trial_by_candidate[item.record.candidate_id] = record
                except Exception as error:  # A failed solver call is still charged.
                    elapsed = time.perf_counter() - started
                    if evaluator_source == "reference_solver":
                        physical_calls += 1
                        physical_solver_seconds += elapsed
                    elif evaluator_source == "surrogate_teacher":
                        teacher_calls += 1
                    elif evaluator_source == "analytic_synthetic":
                        synthetic_calls += 1
                    failure = TrialRecord(
                        policy=policy,
                        candidate_id=item.record.candidate_id,
                        evaluator_source=evaluator_source,
                        call_index=evaluator_calls,
                        status="failed",
                        elapsed_seconds=elapsed,
                        solver_elapsed_seconds=0.0,
                        exception=f"{type(error).__name__}: {error}",
                    )
                    trials_seen.append(failure)
                    trial_by_candidate[item.record.candidate_id] = failure

            selected_trial = trial_by_candidate.get(selected.record.candidate_id)
            if selected_trial is None or selected_trial.status != "completed":
                stop_status = "selected_trial_failed"
                break
            selected_actual = observed_by_candidate[selected.record.candidate_id]
            if selected_trial.accepted:
                current_design = selected.design
                current_observation = selected_actual
                accepted_updates += 1
                reanchor = getattr(policy_oracle, "with_new_baseline", None)
                if callable(reanchor):
                    reanchored_oracle = reanchor(
                        current_design, current_observation, context, ids_by_slot
                    )
                    if reanchored_oracle is policy_oracle:
                        raise ValueError(
                            "with_new_baseline must return a fresh policy-local oracle instance."
                        )
                    policy_oracle = reanchored_oracle
                stop_status = "accepted_update"
            else:
                stop_status = "reference_rejected_predicted_update"
            update_index += 1

            # Exhaustive validation is an offline finite-pool study.  Each
            # candidate reference call has been charged; continue only if the
            # predeclared budget leaves at least one call for the next update.
            if config.validation_mode == "exhaustive_pool":
                if evaluator_calls >= config.max_reference_trials_per_policy:
                    stop_status = "exhaustive_pool_complete"
                    break
            if evaluator_calls >= config.max_reference_trials_per_policy:
                stop_status = "reference_budget_exhausted"
                break

        evaluated_trials = [trial for trial in trials_seen if trial.status == "completed"]
        candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates_seen}
        step_rank_correlations: list[float] = []
        sign_results: list[bool] = []
        mismatch_results: list[float] = []
        evaluated_by_update: dict[int, list[tuple[CandidateRecord, TrialRecord]]] = {}
        for trial in evaluated_trials:
            candidate = candidates_by_id.get(trial.candidate_id)
            if candidate is None or candidate.predicted_objective is None or trial.actual_objective is None:
                continue
            evaluated_by_update.setdefault(candidate.update_index, []).append((candidate, trial))
            if trial.predicted_improvement is not None and trial.actual_improvement is not None:
                if (
                    abs(trial.predicted_improvement) >= config.decision_resolution
                    and abs(trial.actual_improvement) >= config.decision_resolution
                ):
                    sign_results.append((trial.predicted_improvement > 0.0) == (trial.actual_improvement > 0.0))
                mismatch_results.append(abs(trial.predicted_improvement - trial.actual_improvement))
        for same_baseline_pool in evaluated_by_update.values():
            # Candidate ranking is meaningful only within proposals scored from
            # the same accepted baseline and reference-evaluated together.
            if len(same_baseline_pool) >= 2:
                correlation = _spearman(
                    [candidate.predicted_objective for candidate, _ in same_baseline_pool],
                    [trial.actual_objective for _, trial in same_baseline_pool if trial.actual_objective is not None],
                )
                if correlation is not None:
                    step_rank_correlations.append(correlation)

        policy_results[policy] = PolicyResult(
            policy=policy,
            candidates=tuple(candidates_seen),
            trials=tuple(trials_seen),
            accepted_updates=accepted_updates,
            final_design=current_design,
            final_observation=current_observation,
            best_evaluated_feasible_design=best_eval_design,
            best_evaluated_feasible_observation=best_eval_observation,
            best_physical_feasible_design=best_physical_design,
            best_physical_feasible_observation=best_physical_observation,
            candidate_rank_spearman=(float(np.mean(step_rank_correlations)) if step_rank_correlations else None),
            candidate_improvement_sign_accuracy=(float(np.mean(sign_results)) if sign_results else None),
            mean_abs_improvement_mismatch=(float(np.mean(mismatch_results)) if mismatch_results else None),
            geometry_rejections=geometry_rejections,
            evaluator_calls=evaluator_calls,
            reference_calls=(evaluator_calls if evaluator_source in PHYSICAL_SOLVER_SOURCES else 0),
            physical_solver_calls=physical_calls,
            physical_solver_seconds=physical_solver_seconds,
            teacher_evaluations=teacher_calls,
            synthetic_evaluations=synthetic_calls,
            factor_prediction_calls=factor_calls,
            recovered_reference_calls=recovered_reference_calls,
            unknown_solver_time_calls=unknown_solver_time_calls,
            candidate_redraw_audit=tuple(candidate_redraw_audit),
            status=stop_status,
        )

    eligible_pool = [
        (policy, observation)
        for policy, result in policy_results.items()
        for observation in [result.best_evaluated_feasible_observation]
        if observation is not None
    ]
    if eligible_pool:
        best_policy, best_observation = min(
            eligible_pool,
            key=lambda item: smooth_peak_temperature(
                list(item[1].module_temperature_by_id.values()), tau=config.tau_temperature
            ),
        )
        best_objective = smooth_peak_temperature(
            list(best_observation.module_temperature_by_id.values()), tau=config.tau_temperature
        )
    else:
        best_policy, best_objective = None, None
    evidence_interpretation = (
        "reference-observation validated within the executed solver/mesh scope"
        if evaluator_source in PHYSICAL_EVIDENCE_SOURCES
        else f"{evaluator_source}-supported only; not physical solver validation"
    )
    first_graph_candidates = policy_results["graph_guided"].candidates
    dimensions = tuple(
        candidate.normalized_dimension
        for candidate in first_graph_candidates[: config.candidate_budget]
    )
    graph_fallback_count = sum(
        candidate.graph_factor_status == "missing_order_fallback"
        for candidate in policy_results["graph_guided"].candidates
    )
    graph_active_count = sum(
        candidate.graph_factor_status == "active" for candidate in policy_results["graph_guided"].candidates
    )
    graph_guidance_status = (
        "unavailable"
        if graph_active_count == 0
        else "partial_fallback"
        if graph_fallback_count > 0
        else "active"
    )
    return MatchedInverseStudyResult(
        initial_design=initial_design,
        initial_observation=initial_observation,
        config=config,
        pressure_drop_limit=config.pressure_drop_limit,
        normalized_step_coordinate_definition="Euclidean norm of per-module (dx/Lx, dy/Ly) over the matched source block",
        proposal_dimension_by_index=dimensions,
        proposal_order_counts={size: size_schedule_for_result.count(size) for size in sorted(set(size_schedule_for_result))},
        factor_source_selection_basis=(
            "baseline input-only support score within the frozen unary/pair order schedule; "
            "candidate response utility scored on the declared finite proposal pool; "
            "unmaterialized source sets are not objective-pre-screened"
        ),
        graph_guidance_status=graph_guidance_status,
        graph_factor_fallback_count=graph_fallback_count,
        policies=policy_results,
        best_policy_pool_union_policy=best_policy,
        best_policy_pool_union_objective=best_objective,
        policy_pools_are_common=False,
        evidence_interpretation=evidence_interpretation,
    )


__all__ = [
    "AnchoredResponseOracle",
    "CandidateRecord",
    "DecisionEstimate",
    "DecisionObservation",
    "EvidenceSource",
    "InverseStudyConfig",
    "MatchedInverseStudyResult",
    "POLICY_NAMES",
    "PRESSURE_DROP_DEFINITION",
    "PolicyResult",
    "PreparedResponseBaseline",
    "TrialRecord",
    "relative_pressure_limit",
    "run_matched_inverse_design_study",
    "smooth_peak_temperature",
]
