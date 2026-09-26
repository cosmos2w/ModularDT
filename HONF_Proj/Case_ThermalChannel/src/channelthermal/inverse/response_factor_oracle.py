"""Concrete factor-only bridge from ThermalChannel designs to inverse proposals.

The accepted baseline output is captured when this adapter is constructed. A
candidate cannot replace that output or enter the baseline encoder. Only its
physical module displacements reach the anchored response operator.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
import time
from typing import Any, Literal, Protocol

import numpy as np
import torch

from channelthermal.interaction_evidence.types import (
    DesignState,
    ModuleState,
    OperatingContext,
    SolveRecord,
    SolveStatus,
)
from honf_forward_core.interaction_response.factor_operator import AnchoredResponseFactorOperator
from honf_forward_core.interaction_response.organizer import InputOnlySupportScorer
from honf_forward_core.interaction_response.support_search import enumerate_unary_pair_candidates
from honf_forward_core.interaction_response.types import (
    DEFAULT_OUTPUT_CHANNELS,
    BaselineResponseCache,
    ResponseFactor,
    ResponseQueries,
    ResponseQueryBatch,
    ValidityNeighborhood,
    make_baseline_cache,
)
from honf_inverse_core.contracts import NamedContext, PhysicalDesign

from .interaction_guided import DecisionObservation, PreparedResponseBaseline


PAIR_PRESSURE_INCREMENT_POLICY = (
    "zero_pair_fluid_pressure_increment_assumed_analytic_wake_additivity_null; "
    "pair_pressure_channel_was_unmeasured_and_is_not_learned"
)


@dataclass(frozen=True)
class ResponseFeatureSpec:
    """Physical feature scales fixed independently of held-out response labels.

    ``position_delta_scale`` only normalizes trial position increments. Its
    default preserves the original Run1509 convention of ``(Lx, Ly)``; a
    checkpoint trained with a different increment scale must record and pass
    that exact ``(dx_scale, dy_scale)`` tuple here. Baseline positions remain
    normalized by the domain lengths in every recipe.
    """

    heat_scale: float
    position_delta_scale: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.heat_scale) or self.heat_scale <= 0.0:
            raise ValueError("heat_scale must be positive and training-derived.")
        if self.position_delta_scale is not None:
            scales = np.asarray(self.position_delta_scale, dtype=np.float64)
            if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0.0):
                raise ValueError("position_delta_scale must be a pair of positive finite physical scales.")
            object.__setattr__(self, "position_delta_scale", (float(scales[0]), float(scales[1])))


def _active_slots_and_ids(
    design: PhysicalDesign,
    module_ids_by_slot: Sequence[Hashable | None],
) -> tuple[np.ndarray, tuple[Hashable, ...]]:
    if len(module_ids_by_slot) != design.max_modules:
        raise ValueError("module_ids_by_slot must align with the physical design.")
    slots = np.flatnonzero(design.module_present > 0.5)
    ids = tuple(module_ids_by_slot[int(slot)] for slot in slots)
    if any(module_id is None for module_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Active modules need unique, nonempty physical IDs.")
    return slots, ids  # type: ignore[return-value]


def _required_array(
    state: Mapping[str, Any], key: str, *, last_dim: int, require_all_finite: bool = False
) -> np.ndarray:
    if key not in state:
        raise KeyError(f"Accepted baseline output lacks {key!r}.")
    value = np.asarray(state[key], dtype=np.float32)
    if value.ndim != 2 or value.shape[-1] != last_dim:
        raise ValueError(f"Baseline {key} must have shape [Q,{last_dim}].")
    if require_all_finite and not np.isfinite(value).all():
        raise ValueError(f"Baseline {key} query coordinates must be finite.")
    return value


def _ids(state: Mapping[str, Any], key: str, count: int) -> tuple[Hashable, ...]:
    if key not in state:
        raise KeyError(f"Accepted baseline output lacks {key!r}.")
    values = tuple(np.asarray(state[key], dtype=object).reshape(-1).tolist())
    if len(values) != count or any(value is None for value in values):
        raise ValueError(f"Baseline {key} must identify each of {count} material queries.")
    return values


def build_response_model_inputs(
    design: PhysicalDesign,
    context: NamedContext,
    module_ids_by_slot: Sequence[Hashable | None],
    baseline_output_state: Mapping[str, Any],
    feature_spec: ResponseFeatureSpec,
    *,
    device: torch.device | str,
) -> tuple[BaselineResponseCache, ResponseQueries]:
    """Prepare fixed inputs using design, context, and query geometry only.

    Solved baseline values are deliberately absent from the model cache and
    query features. They remain an output offset in the inverse decoder.
    """

    slots, ids = _active_slots_and_ids(design, module_ids_by_slot)
    c = context.as_mapping()
    lx, ly = float(c["domain_length_x"]), float(c["domain_length_y"])
    radius = float(c["module_radius"])
    if lx <= 0.0 or ly <= 0.0 or radius <= 0.0:
        raise ValueError("Domain and module radius must be positive.")
    module_features = np.stack(
        [design.heat_powers[slots] / feature_spec.heat_scale,
         np.full(slots.size, radius / min(lx, ly), dtype=np.float32)],
        axis=-1,
    )
    baseline_design = np.column_stack(
        [
            design.module_centers[slots] / np.asarray([lx, ly], dtype=np.float32),
            design.heat_powers[slots] / feature_spec.heat_scale,
        ]
    )
    # These scales are fixed physical conventions; none is estimated from held
    # out reference outputs. The named schema guards against column drift.
    context_names = (
        "re", "u_in", "nu", "solid_alpha", "fluid_alpha", "solid_k",
        "fluid_k", "module_radius", "domain_length_x", "domain_length_y",
    )
    context_scales = np.asarray([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, min(lx, ly), 10.0, 10.0])
    context_features = np.asarray([float(c[name]) for name in context_names]) / context_scales
    dtype = torch.float32
    cache = make_baseline_cache(
        ids,
        torch.as_tensor(module_features[None], dtype=dtype, device=device),
        torch.as_tensor(baseline_design[None], dtype=dtype, device=device),
        torch.as_tensor(context_features[None], dtype=dtype, device=device),
    )

    grid = _required_array(baseline_output_state, "grid_xy", last_dim=2, require_all_finite=True)
    field = _required_array(baseline_output_state, "fluid_fields", last_dim=5)
    if grid.shape[0] != field.shape[0]:
        raise ValueError("Baseline field and fixed physical grid do not align.")
    field_features = grid / np.asarray([lx, ly], dtype=np.float32)
    fluid_valid = np.asarray(
        baseline_output_state.get("grid_valid_mask", np.ones(grid.shape[0], dtype=bool)), dtype=bool
    ).reshape(-1)
    if fluid_valid.size != grid.shape[0]:
        raise ValueError("grid_valid_mask must align with fluid queries.")
    if not np.isfinite(field[fluid_valid]).all():
        raise ValueError("Observed baseline fluid values must be finite.")

    angles = np.asarray(baseline_output_state.get("interface_angles"), dtype=np.float32).reshape(-1)
    interface = _required_array(baseline_output_state, "interface", last_dim=2)
    interface_ids = _ids(baseline_output_state, "interface_module_ids", interface.shape[0])
    if angles.size != interface.shape[0] or not np.isfinite(angles).all():
        raise ValueError("Interface angles must align with material port queries.")
    interface_features = np.stack([angles / (2.0 * np.pi), np.cos(angles), np.sin(angles)], axis=-1)
    interface_valid = np.asarray(
        baseline_output_state.get("interface_valid_mask", np.ones_like(interface, dtype=bool)), dtype=bool
    )
    if interface_valid.shape != interface.shape:
        raise ValueError("interface_valid_mask must align with both interface channels.")
    if not np.isfinite(interface[interface_valid]).all():
        raise ValueError("Observed baseline interface values must be finite.")

    solid = _required_array(baseline_output_state, "solid_temperature", last_dim=1)
    solid_local = _required_array(
        baseline_output_state, "solid_local_xy", last_dim=2, require_all_finite=True
    )
    solid_ids = _ids(baseline_output_state, "solid_module_ids", solid.shape[0])
    if solid_local.shape[0] != solid.shape[0]:
        raise ValueError("Solid material coordinates must align with temperature samples.")
    solid_valid = np.asarray(
        baseline_output_state.get("solid_valid_mask", np.ones_like(solid, dtype=bool)), dtype=bool
    )
    if solid_valid.shape != solid.shape:
        raise ValueError("solid_valid_mask must align with temperature samples.")
    if not np.isfinite(solid[solid_valid]).all():
        raise ValueError("Observed baseline solid temperatures must be finite.")

    def query(features: np.ndarray, receiver_ids: tuple[Hashable, ...] | None, mask: np.ndarray, prefix: str) -> ResponseQueryBatch:
        query_ids = tuple(
            (prefix, None if receiver_ids is None else receiver_ids[index],
             tuple(float(value) for value in row))
            for index, row in enumerate(features)
        )
        return ResponseQueryBatch(
            features=torch.tensor(features[None], dtype=dtype, device=device),
            receiver_module_ids=receiver_ids,
            query_ids=query_ids,
            mask=torch.tensor(mask[None], dtype=torch.bool, device=device),
        )

    queries: ResponseQueries = {
        "fluid_fields": query(field_features, None, fluid_valid, "grid"),
        "interface": query(interface_features, interface_ids, interface_valid.all(axis=-1), "port"),
        "solid_temperature": query(solid_local, solid_ids, solid_valid[:, 0], "solid"),
    }
    return cache, queries


class HonfThermalResponseOracle:
    """Freeze one accepted physical baseline for a factor-only inverse step."""

    def __init__(
        self,
        *,
        operator: AnchoredResponseFactorOperator,
        scorer: InputOnlySupportScorer,
        baseline_design: PhysicalDesign,
        baseline_context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
        baseline_output_state: Mapping[str, Any],
        feature_spec: ResponseFeatureSpec,
        support_threshold: float,
        support_selection_mode: Literal["threshold", "all_unaries_plus_top1_pair"] = "threshold",
        atlas_factors: Sequence[ResponseFactor] = (),
        query_chunk_size: int = 512,
        _selection_audit: list[dict[str, Any]] | None = None,
        _prediction_timing_audit: list[dict[str, Any]] | None = None,
    ) -> None:
        if not np.isfinite(support_threshold):
            raise ValueError("support_threshold must be finite.")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive.")
        if support_selection_mode not in {"threshold", "all_unaries_plus_top1_pair"}:
            raise ValueError("Unsupported input-only response factor selection mode.")
        self.operator = operator
        self.scorer = scorer
        self.baseline_design = baseline_design
        self.baseline_context = baseline_context
        self._baseline_centers = baseline_design.module_centers.copy()
        self._baseline_present = baseline_design.module_present.copy()
        self._baseline_heat = baseline_design.heat_powers.copy()
        self._baseline_context_vector = baseline_context.vector.copy()
        self.module_ids_by_slot = tuple(module_ids_by_slot)
        self.baseline_output_state = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in baseline_output_state.items()
        }
        for value in self.baseline_output_state.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        self.feature_spec = feature_spec
        self.support_threshold = float(support_threshold)
        self.support_selection_mode = support_selection_mode
        self.selection_audit = _selection_audit if _selection_audit is not None else []
        self.prediction_timing_audit = _prediction_timing_audit if _prediction_timing_audit is not None else []
        atlas_factor_values = tuple(atlas_factors)
        atlas_factor_ids = [factor.factor_id for factor in atlas_factor_values]
        if len(atlas_factor_ids) != len(set(atlas_factor_ids)):
            raise ValueError("Atlas validity factors must have unique factor IDs.")
        self.atlas_factors_by_id = {factor.factor_id: factor for factor in atlas_factor_values}
        self.query_chunk_size = int(query_chunk_size)

    def prepare_baseline(
        self,
        design: PhysicalDesign,
        context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
    ) -> PreparedResponseBaseline:
        if not np.array_equal(design.module_centers, self._baseline_centers) or not np.array_equal(
            design.heat_powers, self._baseline_heat
        ) or not np.array_equal(design.module_present, self._baseline_present):
            raise ValueError("The response oracle must be rebuilt after accepting a new physical baseline.")
        if (context.feature_names != self.baseline_context.feature_names
                or not np.array_equal(context.vector, self._baseline_context_vector)):
            raise ValueError("Baseline context changed; rebuild the response oracle.")
        if tuple(module_ids_by_slot) != self.module_ids_by_slot:
            raise ValueError("Physical module IDs changed; rebuild the response oracle.")
        device = next(self.operator.parameters()).device
        if next(self.scorer.parameters()).device != device:
            raise ValueError("Factor decoder and support scorer must share a device.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        baseline_prepare_started = time.perf_counter()
        cache, queries = build_response_model_inputs(
            design, context, module_ids_by_slot, self.baseline_output_state, self.feature_spec, device=device
        )
        _, ids = _active_slots_and_ids(design, module_ids_by_slot)
        baseline_centers = {
            module_id: tuple(float(value) for value in cache.baseline_design[0, index].tolist())
            for index, module_id in enumerate(ids)
        }
        unknown_validity = ValidityNeighborhood(
            max_abs_delta_by_module={module_id: (None, None, None) for module_id in ids},
            baseline_design_center_by_module=baseline_centers,
        )
        candidates = enumerate_unary_pair_candidates(ids, tuple(queries), validity=unknown_validity)
        atlas_factor_ids = set(self.atlas_factors_by_id)
        candidate_ids = {factor.factor_id for factor in candidates}
        if atlas_factor_ids - candidate_ids:
            raise ValueError(
                f"Atlas factors do not match the active physical candidate graph: {sorted(atlas_factor_ids - candidate_ids)}"
            )
        factors = []
        for candidate in candidates:
            atlas_factor = self.atlas_factors_by_id.get(candidate.factor_id)
            if atlas_factor is None:
                factors.append(candidate)
                continue
            if (
                atlas_factor.donor_ids != candidate.donor_ids
                or atlas_factor.output_roles != candidate.output_roles
            ):
                raise ValueError(f"Atlas factor {candidate.factor_id!r} changed donor or output-role identity.")
            measured_centers = dict(atlas_factor.validity.baseline_design_center_by_module or {})
            measured_centers.update(baseline_centers)
            measured_validity = replace(
                atlas_factor.validity,
                baseline_design_center_by_module=measured_centers,
            )
            factors.append(
                replace(
                    candidate,
                    validity=measured_validity,
                    evidence_sources=atlas_factor.evidence_sources,
                    evidence_anchor_ids=atlas_factor.evidence_anchor_ids,
                )
            )
        candidates = tuple(factors)
        self.operator.eval()
        self.scorer.eval()
        scorer_device = next(self.scorer.parameters()).device
        if scorer_device.type == "cuda":
            torch.cuda.synchronize(scorer_device)
        scorer_started = time.perf_counter()
        with torch.no_grad():
            scores = self.scorer.score_factors(cache, candidates)[0]
        if scorer_device.type == "cuda":
            torch.cuda.synchronize(scorer_device)
        scorer_elapsed = time.perf_counter() - scorer_started
        score_by_id = {factor.factor_id: float(score) for factor, score in zip(candidates, scores)}
        if self.support_selection_mode == "threshold":
            with torch.no_grad():
                weights = self.scorer.retention_weights(scores, self.support_threshold)
            kept = tuple(factor for factor, weight in zip(candidates, weights) if float(weight) > 0.0)
            weight_by_id = {factor.factor_id: float(weight) for factor, weight in zip(candidates, weights)}
            selected_top1_pair_id = None
        else:
            unaries = tuple(factor for factor in candidates if len(factor.donor_ids) == 1)
            pairs = tuple(factor for factor in candidates if len(factor.donor_ids) == 2)
            if not unaries or not pairs:
                raise ValueError("all_unaries_plus_top1_pair requires at least one unary and one pair candidate.")
            selected_pair = min(pairs, key=lambda factor: (-score_by_id[factor.factor_id], factor.factor_id))
            kept = (*unaries, selected_pair)
            selected_top1_pair_id = selected_pair.factor_id
            kept_ids = {factor.factor_id for factor in kept}
            weight_by_id = {
                factor.factor_id: 1.0 if factor.factor_id in kept_ids else 0.0
                for factor in candidates
            }
        metadata = {
            "source": "baseline_input_only_scorer",
            "support_selection_mode": self.support_selection_mode,
            "support_scorer_elapsed_seconds": scorer_elapsed,
            "candidate_capacity": len(candidates),
            "active_unary": sum(len(factor.donor_ids) == 1 for factor in kept),
            "active_joint": sum(len(factor.donor_ids) == 2 for factor in kept),
            "actual_retained_factor_count": len(kept),
            "selected_top1_pair_factor_id": selected_top1_pair_id,
            "factor_weight_by_id": weight_by_id,
            "all_receiver_fallback": True,
            "support_threshold": self.support_threshold,
            "weight_by_factor_id": weight_by_id,
            "score_by_factor_id": score_by_id,
            "pair_pressure_increment_policy": PAIR_PRESSURE_INCREMENT_POLICY,
        }
        self.selection_audit.append(
            {
                "module_ids": [str(module_id) for module_id in ids],
                "support_selection_mode": self.support_selection_mode,
                "support_scorer_elapsed_seconds": scorer_elapsed,
                "selected_top1_pair_factor_id": selected_top1_pair_id,
                "active_factor_count": len(kept),
                "active_unary_count": metadata["active_unary"],
                "active_pair_count": metadata["active_joint"],
                "retained_factor_ids": [factor.factor_id for factor in kept],
                "all_receiver_fallback": True,
                "pair_pressure_increment_policy": PAIR_PRESSURE_INCREMENT_POLICY,
            }
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        baseline_prepare_elapsed = time.perf_counter() - baseline_prepare_started
        metadata["baseline_prepare_elapsed_seconds"] = baseline_prepare_elapsed
        metadata["support_scored_factor_count"] = len(candidates)
        self.selection_audit[-1]["baseline_prepare_elapsed_seconds"] = baseline_prepare_elapsed
        self.selection_audit[-1]["support_scored_factor_count"] = len(candidates)
        return PreparedResponseBaseline(
            cache=cache,
            queries=queries,
            factors=kept,
            metadata=metadata,
        )

    def rank_factors(self, prepared: PreparedResponseBaseline) -> Sequence[ResponseFactor]:
        scores = prepared.metadata["score_by_factor_id"]
        return tuple(sorted(prepared.factors, key=lambda factor: (-scores[factor.factor_id], factor.factor_id)))

    def with_new_baseline(
        self,
        design: PhysicalDesign,
        observation: DecisionObservation,
        context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
    ) -> HonfThermalResponseOracle:
        """Create a fresh factor cache after a trial passes reference acceptance."""

        if not observation.output_state:
            raise ValueError("A newly accepted baseline must carry its measured physical output state.")
        return HonfThermalResponseOracle(
            operator=self.operator,
            scorer=self.scorer,
            baseline_design=design,
            baseline_context=context,
            module_ids_by_slot=module_ids_by_slot,
            baseline_output_state=observation.output_state,
            feature_spec=self.feature_spec,
            support_threshold=self.support_threshold,
            support_selection_mode=self.support_selection_mode,
            atlas_factors=(),
            query_chunk_size=self.query_chunk_size,
            _selection_audit=self.selection_audit,
            _prediction_timing_audit=self.prediction_timing_audit,
        )

    @staticmethod
    def _require_prepared_factor(
        prepared: PreparedResponseBaseline, factor: ResponseFactor
    ) -> None:
        if not any(active is factor for active in prepared.factors):
            raise ValueError("The exact factor object must belong to this accepted baseline.")

    def _normalized_factor_deltas(
        self,
        prepared: PreparedResponseBaseline,
        factor: ResponseFactor,
        delta_by_module_id: Mapping[Hashable, np.ndarray],
    ) -> dict[Hashable, torch.Tensor]:
        c = self.baseline_context.as_mapping()
        scales = np.asarray(
            self.feature_spec.position_delta_scale
            if self.feature_spec.position_delta_scale is not None
            else (float(c["domain_length_x"]), float(c["domain_length_y"])),
            dtype=np.float32,
        )
        deltas: dict[Hashable, torch.Tensor] = {}
        for module_id in factor.donor_ids:
            if module_id not in delta_by_module_id:
                raise KeyError(f"Missing physical displacement for donor {module_id!r}.")
            physical = np.asarray(delta_by_module_id[module_id], dtype=np.float32).reshape(-1)
            if physical.shape != (2,) or not np.isfinite(physical).all():
                raise ValueError("Donor displacements must be finite physical [dx,dy] vectors.")
            normalized = np.asarray([physical[0] / scales[0], physical[1] / scales[1], 0.0], dtype=np.float32)
            deltas[module_id] = torch.as_tensor(
                normalized[None], device=prepared.cache.module_features.device
            )
        return deltas

    def factor_validity_status(
        self,
        prepared: PreparedResponseBaseline,
        factor: ResponseFactor,
        delta_by_module_id: Mapping[Hashable, np.ndarray],
    ) -> bool | None:
        """Check moves against atlas-measured factor neighborhoods only."""

        self._require_prepared_factor(prepared, factor)
        deltas = self._normalized_factor_deltas(prepared, factor, delta_by_module_id)
        return self.operator.validity_status(prepared.cache, factor, deltas)

    def predict_factor_response(
        self,
        prepared: PreparedResponseBaseline,
        factor: ResponseFactor,
        delta_by_module_id: Mapping[Hashable, np.ndarray],
    ) -> Mapping[str, np.ndarray]:
        self._require_prepared_factor(prepared, factor)
        cache = prepared.cache
        deltas = self._normalized_factor_deltas(prepared, factor, delta_by_module_id)
        device = next(self.operator.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            role_values = self.operator.predict_factor_response(
                cache, factor, deltas, prepared.queries, query_chunk_size=self.query_chunk_size
            )
        if len(factor.donor_ids) == 2 and "fluid_fields" in role_values:
            # Pair p labels were deliberately masked in Run1511 because the
            # analytic-wake pressure channel is assumed additive in donor
            # contributions. This fixed inverse convention prevents an
            # unsupervised pair head from changing pressure feasibility; it is
            # not presented as a learned or independently measured response.
            pressure_channel = DEFAULT_OUTPUT_CHANNELS["fluid_fields"].index("p")
            fluid_increment = role_values["fluid_fields"]
            if fluid_increment.shape[-1] != len(DEFAULT_OUTPUT_CHANNELS["fluid_fields"]):
                raise ValueError("The selected fluid response width differs from the declared channel order.")
            fluid_increment = fluid_increment.clone()
            fluid_increment[..., pressure_channel] = 0.0
            role_values = {**role_values, "fluid_fields": fluid_increment}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.prediction_timing_audit.append(
            {
                "factor_id": factor.factor_id,
                "donor_ids": [str(value) for value in factor.donor_ids],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        weight = float(prepared.metadata["weight_by_factor_id"][factor.factor_id])
        return {role: (value[0].detach().cpu().numpy() * weight) for role, value in role_values.items()}


def decision_observation_from_solve(record: SolveRecord) -> DecisionObservation:
    """Carry only a converged physical record into inverse acceptance."""

    if record.status is not SolveStatus.CONVERGED or record.output is None:
        raise ValueError("Inverse decisions require a converged, completed solver record.")
    output = record.output
    pressure = output.quantities["pressure_drop"]
    return DecisionObservation(
        module_temperature_by_id=output.module_peak_temperature,
        pressure_drop=output.pressure_drop,
        pressure_drop_units=pressure.units,
        evidence_source=record.source.value,
        elapsed_seconds=record.elapsed_seconds,
        provenance={"record_id": record.record_id, **dict(record.provenance)},
        output_state=output.as_mapping(),
    )


def physical_design_from_evidence(design: DesignState) -> tuple[PhysicalDesign, tuple[str | None, ...]]:
    """Preserve explicit physical IDs and any padding in a solver design."""

    modules = design.modules
    physical = PhysicalDesign(
        module_centers=np.asarray([module.position_xy for module in modules], dtype=np.float32),
        module_present=np.asarray([module.active for module in modules], dtype=np.float32),
        heat_powers=np.asarray([module.heating for module in modules], dtype=np.float32),
    )
    ids = tuple(module.module_id if module.active else None for module in modules)
    return physical, ids


class _ReferenceAdapter(Protocol):
    def solve(self, design: DesignState, context: OperatingContext, *, record_id: str) -> SolveRecord: ...


class PhysicalReferenceTrialEvaluator:
    """One source-labelled, charged callback for the matched inverse study.

    The inverse runner charges attempts before calling this object. Its
    ``records`` retain both successful and failed solver outcomes locally.
    """

    def __init__(
        self,
        *,
        adapter: _ReferenceAdapter,
        baseline_design: DesignState,
        operating_context: OperatingContext,
        named_context: NamedContext,
        module_ids_by_slot: Sequence[str | None],
        record_prefix: str,
    ) -> None:
        if not record_prefix:
            raise ValueError("Inverse trial records need a nonempty prefix.")
        self.adapter = adapter
        self.baseline_design = baseline_design
        self.operating_context = operating_context
        self.named_context = named_context
        self.module_ids_by_slot = tuple(module_ids_by_slot)
        self.record_prefix = record_prefix
        self.records: list[SolveRecord] = []
        self.attempted_calls = 0
        physical, expected_ids = physical_design_from_evidence(baseline_design)
        if expected_ids != self.module_ids_by_slot:
            raise ValueError("Trial module IDs must match the reference baseline's physical IDs.")
        self._baseline_present = physical.module_present.copy()
        self._baseline_heat = physical.heat_powers.copy()

    def __call__(
        self,
        design: PhysicalDesign,
        context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
    ) -> DecisionObservation:
        self.attempted_calls += 1
        if tuple(module_ids_by_slot) != self.module_ids_by_slot:
            raise ValueError("Trial module IDs changed across the inverse step.")
        if context.feature_names != self.named_context.feature_names or not np.array_equal(
            context.vector, self.named_context.vector
        ):
            raise ValueError("Trial operating context changed across the inverse step.")
        if not np.array_equal(design.module_present, self._baseline_present) or not np.array_equal(
            design.heat_powers, self._baseline_heat
        ):
            raise ValueError("Primary position experiment must keep topology and heating fixed.")
        if design.max_modules != len(self.baseline_design.modules):
            raise ValueError("Trial module slot count changed.")
        modules = tuple(
            ModuleState(
                module_id=original.module_id,
                position_xy=tuple(float(value) for value in design.module_centers[index]),
                heating=original.heating,
                active=original.active,
            )
            for index, original in enumerate(self.baseline_design.modules)
        )
        trial = DesignState(
            anchor_id=self.baseline_design.anchor_id,
            physical_family_id=self.baseline_design.physical_family_id,
            split=self.baseline_design.split,
            modules=modules,
        )
        record = self.adapter.solve(
            trial,
            self.operating_context,
            record_id=f"{self.record_prefix}_{self.attempted_calls:03d}",
        )
        self.records.append(record)
        return decision_observation_from_solve(record)


__all__ = [
    "HonfThermalResponseOracle",
    "PhysicalReferenceTrialEvaluator",
    "ResponseFeatureSpec",
    "build_response_model_inputs",
    "decision_observation_from_solve",
    "physical_design_from_evidence",
]
