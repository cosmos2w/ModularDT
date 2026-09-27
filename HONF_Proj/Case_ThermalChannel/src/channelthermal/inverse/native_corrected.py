"""Baseline-corrected quantities from one intact native absolute predictor.

The accepted reference observation is an output offset, never an input to the
forward model or its cover organizer. A discrete cover may be frozen for one
inner step; the predictor still receives the complete current physical design
at every trial and must recompute all continuous model states for that design.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from channelthermal.response_control.contracts import AbsoluteOperator, DesignInput, RoleQuery
from channelthermal.response_control.thermal import (
    NativeThermalQuantities,
    reduce_native_thermal_quantities,
)
from honf_inverse_core.contracts import NamedContext, PhysicalDesign

from .interaction_guided import DecisionEstimate, DecisionObservation
from .local_interface_contract import LocalInterfacePlan


class NativeQuantityPredictor(Protocol):
    """Reduce a fresh native forward call to physical decision quantities."""

    def __call__(
        self,
        design: PhysicalDesign,
        context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
        *,
        frozen_topology: Any | None,
    ) -> DecisionEstimate: ...


TopologyFactory = Callable[
    [PhysicalDesign, NamedContext, Sequence[Hashable | None]], Any
]


class ThermalNativeQuantityPredictor:
    """Call one intact native absolute operator at every physical trial design.

    Receiver locations and role schemas may come from an accepted physical
    stencil, but only target-free query features enter the operator. The
    optional material validity mask is used after prediction to select the
    maintained measured material query universe. No reference value, class,
    case ID, or objective enters the forward pass. A frozen plan must reach a
    real native cover path; an inverse grouping alone is never accepted as
    graph-mediated forward computation.
    """

    def __init__(
        self,
        operator: AbsoluteOperator,
        role_queries: Mapping[str, RoleQuery],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        solid_valid_mask: np.ndarray | torch.Tensor | None = None,
        checkpoint_hash: str | None = None,
    ) -> None:
        self.operator = operator
        self.role_queries = dict(role_queries)
        self.device = torch.device(device)
        self.dtype = dtype
        self.solid_valid_mask = solid_valid_mask
        self.checkpoint_hash = checkpoint_hash
        if {"fluid_fields", "solid_temperature"} - set(self.role_queries):
            raise ValueError("Native inverse needs fluid and material receiver roles.")
        if any(query.query_features.device != self.device for query in self.role_queries.values()):
            raise ValueError("All native role queries must be on the declared device.")

    def tensor_quantities(
        self,
        design: PhysicalDesign,
        context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
        *,
        frozen_topology: Any | None = None,
        requires_grad: bool = False,
    ) -> NativeThermalQuantities:
        """Return differentiable native quantities before scalar detachment."""

        active_ids = _active_ids(design, module_ids_by_slot)
        if not active_ids:
            raise ValueError("Native inverse requires active modules.")
        live = DesignInput(
            module_positions=torch.tensor(
                np.asarray(design.module_centers), device=self.device, dtype=self.dtype,
                requires_grad=requires_grad,
            ),
            module_heating=torch.tensor(
                np.asarray(design.heat_powers), device=self.device, dtype=self.dtype,
                requires_grad=requires_grad,
            ),
            module_present=torch.as_tensor(
                np.asarray(design.module_present) > 0.5, device=self.device, dtype=torch.bool
            ),
        )
        return self.tensor_quantities_from_input(
            live, context, module_ids_by_slot,
            module_family_id=design.module_family_id,
            frozen_topology=frozen_topology,
        )

    def capture_frozen_topology(
        self,
        baseline: PhysicalDesign,
        context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
        *,
        max_position_delta: float,
    ) -> LocalInterfacePlan:
        """Capture one input-only native organizer decision at the anchor."""

        _active_ids(baseline, module_ids_by_slot)
        if self.checkpoint_hash is None:
            raise ValueError("Native topology capture requires the checkpoint SHA256.")
        capture = getattr(self.operator, "capture_anchor_cover_plans", None)
        if not callable(capture):
            raise TypeError("The absolute operator cannot capture an applied native cover.")
        live = DesignInput(
            module_positions=torch.as_tensor(np.array(baseline.module_centers, copy=True), device=self.device, dtype=self.dtype),
            module_heating=torch.as_tensor(np.array(baseline.heat_powers, copy=True), device=self.device, dtype=self.dtype),
            module_present=torch.as_tensor(np.array(baseline.module_present > 0.5, copy=True), device=self.device, dtype=torch.bool),
        )
        plans = capture(live, context.as_mapping(), self.role_queries)
        return LocalInterfacePlan.from_anchor(
            baseline=baseline,
            module_ids_by_slot=module_ids_by_slot,
            context=context,
            role_queries=self.role_queries,
            checkpoint_hash=self.checkpoint_hash,
            cover_plans=plans,
            max_position_delta=max_position_delta,
        )

    def tensor_quantities_from_input(
        self,
        live: DesignInput,
        context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
        *,
        module_family_id: str,
        frozen_topology: LocalInterfacePlan | None = None,
    ) -> NativeThermalQuantities:
        """Preserve autograd to caller-owned positions and heating tensors."""

        if live.module_positions.device != self.device or live.module_heating.device != self.device:
            raise ValueError("Caller-owned native design tensors must use the declared device.")
        if live.module_positions.dtype != self.dtype or live.module_heating.dtype != self.dtype:
            raise ValueError("Caller-owned native design tensors must use the declared dtype.")
        if len(module_ids_by_slot) != int(live.module_present.numel()):
            raise ValueError("Physical module IDs must align with every live design slot.")
        if any(bool(active) != (module_id is not None) for active, module_id in zip(
            live.module_present.detach().cpu().tolist(), module_ids_by_slot, strict=True
        )):
            raise ValueError("Only active live modules may carry physical IDs.")
        physical_ids = tuple("" if module_id is None else str(module_id) for module_id in module_ids_by_slot)
        active_physical_ids = {
            physical_ids[index]
            for index, flag in enumerate(live.module_present)
            if bool(flag)
        }
        if not active_physical_ids or len(active_physical_ids) != int(live.module_present.sum().item()):
            raise ValueError("Physical IDs collide after conversion to native strings.")
        if frozen_topology is None:
            values = self.operator(live, context.as_mapping(), self.role_queries)
        else:
            if not isinstance(frozen_topology, LocalInterfacePlan):
                raise TypeError("Native frozen topology must be a typed LocalInterfacePlan.")
            if self.checkpoint_hash is None:
                raise ValueError("Frozen topology requires a verified predictor checkpoint hash.")
            frozen_topology.validate_trial(
                live, context, self.role_queries,
                checkpoint_hash=self.checkpoint_hash,
                module_ids_by_slot=module_ids_by_slot,
                module_family_id=module_family_id,
            )
            predict_masked = getattr(self.operator, "predict_with_frozen_topology", None)
            if not callable(predict_masked):
                raise ValueError("The absolute operator has no native fixed-cover execution path.")
            values = predict_masked(
                live, context.as_mapping(), self.role_queries,
                fixed_cover_plans=frozen_topology.cover_plans,
            )
        return reduce_native_thermal_quantities(
            values, live, self.role_queries, context.as_mapping(),
            module_ids=physical_ids, solid_valid_mask=self.solid_valid_mask,
        )

    def __call__(
        self,
        design: PhysicalDesign,
        context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
        *,
        frozen_topology: Any | None,
    ) -> DecisionEstimate:
        with torch.no_grad():
            quantities = self.tensor_quantities(
                design, context, module_ids_by_slot, frozen_topology=frozen_topology
            )
        return DecisionEstimate(
            module_temperature_by_id={
                module_id: float(value.detach().cpu().item())
                for module_id, value in quantities.module_peak_temperature.items()
            },
            pressure_drop=float(quantities.pressure_drop.detach().cpu().item()),
            pressure_drop_units=quantities.pressure_drop_units,
        )


def _active_ids(
    design: PhysicalDesign, module_ids_by_slot: Sequence[Hashable | None]
) -> tuple[Hashable, ...]:
    if len(module_ids_by_slot) != design.max_modules:
        raise ValueError("Physical module IDs must align with every design slot.")
    ids: list[Hashable] = []
    for active, module_id in zip(design.module_present > 0.5, module_ids_by_slot, strict=True):
        if active and module_id is None:
            raise ValueError("Every active module needs a physical ID.")
        if not active and module_id is not None:
            raise ValueError("An inactive slot cannot carry an active physical ID.")
        if active:
            ids.append(module_id)  # type: ignore[arg-type]
    if len(ids) != len(set(ids)):
        raise ValueError("Active physical module IDs must be unique.")
    return tuple(ids)


def baseline_correct_quantities(
    reference_baseline: DecisionObservation,
    model_baseline: DecisionEstimate,
    model_trial: DecisionEstimate,
) -> DecisionEstimate:
    """Apply the same-model finite increment to measured baseline quantities.

    Correction is per material module before reducing a peak objective. It
    cancels a constant model offset but makes no claim about an inaccurate
    finite increment, extrapolation, or reference uncertainty.
    """

    ids = set(reference_baseline.module_temperature_by_id)
    if ids != set(model_baseline.module_temperature_by_id) or ids != set(
        model_trial.module_temperature_by_id
    ):
        raise ValueError("Reference and both native predictions need identical physical module IDs.")
    if not (
        reference_baseline.pressure_drop_units
        == model_baseline.pressure_drop_units
        == model_trial.pressure_drop_units
    ):
        raise ValueError("Pressure-drop units differ across the reference and native predictions.")
    if not (
        reference_baseline.pressure_drop_definition
        == model_baseline.pressure_drop_definition
        == model_trial.pressure_drop_definition
    ):
        raise ValueError("Pressure-drop reductions use different physical definitions.")
    corrected = {
        module_id: float(reference_baseline.module_temperature_by_id[module_id])
        + float(model_trial.module_temperature_by_id[module_id])
        - float(model_baseline.module_temperature_by_id[module_id])
        for module_id in reference_baseline.module_temperature_by_id
    }
    pressure = (
        float(reference_baseline.pressure_drop)
        + float(model_trial.pressure_drop)
        - float(model_baseline.pressure_drop)
    )
    return DecisionEstimate(
        module_temperature_by_id=corrected,
        pressure_drop=pressure,
        pressure_drop_units=reference_baseline.pressure_drop_units,
        pressure_drop_definition=reference_baseline.pressure_drop_definition,
    )


def baseline_correct_tensor_quantities(
    reference_baseline: DecisionObservation,
    model_baseline: NativeThermalQuantities,
    model_trial: NativeThermalQuantities,
) -> NativeThermalQuantities:
    """Correct per-ID live tensors before any nonsmooth peak reduction."""

    ids = set(reference_baseline.module_temperature_by_id)
    if ids != set(model_baseline.module_peak_temperature) or ids != set(model_trial.module_peak_temperature):
        raise ValueError("Reference and both native tensor predictions need identical physical module IDs.")
    if not (reference_baseline.pressure_drop_units == model_baseline.pressure_drop_units == model_trial.pressure_drop_units):
        raise ValueError("Pressure-drop units differ across tensor correction inputs.")
    if model_baseline.temperature_units != model_trial.temperature_units:
        raise ValueError("Material-temperature units differ across tensor correction inputs.")
    pressure = model_trial.pressure_drop + (
        model_trial.pressure_drop.new_tensor(float(reference_baseline.pressure_drop))
        - model_baseline.pressure_drop.detach()
    )
    corrected = {
        module_id: model_trial.module_peak_temperature[module_id]
        + (
            model_trial.module_peak_temperature[module_id].new_tensor(
                float(reference_baseline.module_temperature_by_id[module_id])
            )
            - model_baseline.module_peak_temperature[module_id].detach()
        )
        for module_id in reference_baseline.module_temperature_by_id
    }
    return NativeThermalQuantities(
        pressure_drop=pressure,
        pressure_drop_units=model_trial.pressure_drop_units,
        module_peak_temperature=corrected,
        temperature_units=model_trial.temperature_units,
    )


@dataclass(frozen=True)
class CorrectedNativeTrial:
    """A trial scored through a fixed local topology and fresh native state."""

    design: PhysicalDesign
    native_absolute: DecisionEstimate
    baseline_corrected: DecisionEstimate


class NativeBaselineCorrection:
    """One accepted reference anchor for a bounded native inverse inner step.

    A new instance is required after a reference-accepted move. This prevents
    stale model baselines, source states, and topology from crossing anchors.
    The predictor must be target-free; it receives only design, context, IDs,
    and the baseline-input-derived discrete topology.
    """

    def __init__(
        self,
        *,
        predictor: NativeQuantityPredictor,
        baseline_design: PhysicalDesign,
        reference_baseline: DecisionObservation,
        context: NamedContext,
        module_ids_by_slot: Sequence[Hashable | None],
        topology_factory: TopologyFactory | None = None,
    ) -> None:
        ids = _active_ids(baseline_design, module_ids_by_slot)
        if reference_baseline.evidence_source not in {"reference_solver", "stored_reference"}:
            raise ValueError("Baseline correction requires an independent reference observation.")
        if set(ids) != set(reference_baseline.module_temperature_by_id):
            raise ValueError("The measured baseline must identify every active physical module.")
        self.predictor = predictor
        self.baseline_design = baseline_design
        self.reference_baseline = reference_baseline
        self.context = context
        self.module_ids_by_slot = tuple(module_ids_by_slot)
        self.topology_factory = topology_factory
        self.frozen_topology = (
            topology_factory(baseline_design, context, self.module_ids_by_slot)
            if topology_factory is not None else None
        )
        self.model_baseline = predictor(
            baseline_design, context, self.module_ids_by_slot,
            frozen_topology=self.frozen_topology,
        )
        baseline_correct_quantities(reference_baseline, self.model_baseline, self.model_baseline)

    def evaluate(self, trial_design: PhysicalDesign) -> CorrectedNativeTrial:
        """Recompute native continuous states at the trial design."""

        if trial_design.module_family_id != self.baseline_design.module_family_id:
            raise ValueError("The trial changed module family.")
        if not np.array_equal(trial_design.module_present, self.baseline_design.module_present):
            raise ValueError("The trial changed module presence within one inverse step.")
        native = self.predictor(
            trial_design, self.context, self.module_ids_by_slot,
            frozen_topology=self.frozen_topology,
        )
        return CorrectedNativeTrial(
            design=trial_design,
            native_absolute=native,
            baseline_corrected=baseline_correct_quantities(
                self.reference_baseline, self.model_baseline, native
            ),
        )

    def recomputed_topology_discrepancy(
        self, trial_design: PhysicalDesign,
        frozen_trial: CorrectedNativeTrial,
    ) -> DecisionEstimate:
        """Return absolute output with a newly inferred plan for boundary QA.

        The returned prediction is not silently substituted for the frozen
        inner-step score; callers compare the two at the same trial design.
        """

        if self.topology_factory is None:
            raise ValueError("No learned topology is configured for this comparison.")
        if frozen_trial.design is not trial_design:
            raise ValueError("Boundary comparison requires the same trial object and design.")
        new_topology = self.topology_factory(
            trial_design, self.context, self.module_ids_by_slot
        )
        return self.predictor(
            trial_design, self.context, self.module_ids_by_slot,
            frozen_topology=new_topology,
        )

    def reanchor(
        self,
        accepted_design: PhysicalDesign,
        independently_measured: DecisionObservation,
    ) -> NativeBaselineCorrection:
        """Refresh native baseline and discrete topology after reference acceptance."""

        if independently_measured is self.reference_baseline:
            raise ValueError("A new accepted anchor requires a new reference observation.")
        previous_id = self.reference_baseline.provenance.get("record_id")
        next_id = independently_measured.provenance.get("record_id")
        if previous_id is not None and next_id is not None and previous_id == next_id:
            raise ValueError("Reanchoring cannot reuse the same reference record ID.")
        return NativeBaselineCorrection(
            predictor=self.predictor,
            baseline_design=accepted_design,
            reference_baseline=independently_measured,
            context=self.context,
            module_ids_by_slot=self.module_ids_by_slot,
            topology_factory=self.topology_factory,
        )


__all__ = [
    "CorrectedNativeTrial",
    "NativeBaselineCorrection",
    "NativeQuantityPredictor",
    "ThermalNativeQuantityPredictor",
    "baseline_correct_quantities",
    "baseline_correct_tensor_quantities",
]
