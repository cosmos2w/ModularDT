"""Train-input calibrated, input-independent universal-root intervention.

Conditional organization is removed: one group admits every eligible fine
source and carries a fixed typed control. Original physical states, measures
and geometry remain live. The extra all-access work must be reported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .typed_hypergraph_state import SOURCE_TYPE, TypedHypergraphState


@dataclass(frozen=True)
class TrainingPopulationSummary:
    controls: dict[int, dict[str, torch.Tensor]]  # phase -> mechanism -> [C]
    case_counts: dict[int, int]
    source_bearing_case_counts: dict[int, dict[str, int]]

    def apply(self, state: TypedHypergraphState) -> TypedHypergraphState:
        """Universal root, dense physical-source membership, fixed controls."""
        if state.phase not in self.controls:
            raise ValueError(f"No train-only fixed-summary calibration for phase {state.phase}")
        membership, controls = {}, {}
        for tau, old in state.memberships.items():
            kind = SOURCE_TYPE[tau]
            member = torch.zeros_like(old)
            member[:, 0] = state.source_valid[kind].to(member.dtype)
            control = torch.zeros_like(state.controls[tau])
            mean = self.controls[state.phase][tau].to(device=control.device, dtype=control.dtype)
            if mean.shape != control.shape[-1:]:
                raise ValueError("Training-summary control width differs from the prepared model")
            control[:, 0] = mean
            membership[tau], controls[tau] = member, control
        admission = torch.zeros_like(state.admission)
        admission[:, 0] = 1
        data = dict(state.strategy_data)
        if "trees" in data:
            data["gates"] = {tau: torch.zeros_like(gate) for tau, gate in data["gates"].items()}
            data["typed_admission"] = {tau: admission for tau in state.memberships}
        if "gate_logits" in data:
            logits = torch.full_like(data["gate_logits"], -20)
            logits[:, 0] = 20
            data["gate_logits"] = logits
        data["training_population_fixed_summary"] = True
        diagnostics = {**state.diagnostics, "fixed_summary_calibration_cases": admission.new_tensor(self.case_counts[state.phase]),
                       "fixed_summary_admitted_groups": admission.new_ones(admission.shape[0])}
        return replace(state, memberships=membership, controls=controls, admission=admission,
                       strategy_data=data, diagnostics=diagnostics)


class TrainingPopulationSummaryAccumulator:
    """Equal-case mean of source-bearing admitted typed controls, train only."""

    def __init__(self):
        self._sums = {}
        self._counts = {}
        self._cases = {}

    def add(self, state: TypedHypergraphState, *, partition: str) -> None:
        if partition != "train":
            raise ValueError("Fixed organizer summaries may be calibrated only from training inputs")
        phase = state.phase
        self._cases[phase] = self._cases.get(phase, 0) + state.admission.shape[0]
        self._sums.setdefault(phase, {})
        self._counts.setdefault(phase, {})
        for tau, member in state.memberships.items():
            kind = SOURCE_TYPE[tau]
            mass = (member.detach() * state.source_measures[kind][:, None].detach()).sum(-1)
            admission = state.strategy_data.get("typed_admission", {}).get(tau, state.admission).detach()
            weight = mass * admission
            total = weight.sum(-1)
            physical = state.controls[tau].detach()
            selected = torch.where(weight[..., None] > 0, physical, torch.zeros_like(physical))
            value = (selected * weight[..., None]).sum(1) / total.clamp_min(1e-12)[:, None]
            present = total > 0
            summed = value[present].sum(0).cpu()
            self._sums[phase][tau] = self._sums[phase].get(tau, torch.zeros_like(summed)) + summed
            self._counts[phase][tau] = self._counts[phase].get(tau, 0) + int(present.sum())

    def finish(self) -> TrainingPopulationSummary:
        if not self._cases:
            raise ValueError("A fixed summary requires genuine train-input calibration")
        controls = {phase: {tau: total / max(1, self._counts[phase][tau]) for tau, total in values.items()}
                    for phase, values in self._sums.items()}
        return TrainingPopulationSummary(controls, dict(self._cases),
                                         {phase: dict(counts) for phase, counts in self._counts.items()})


__all__ = ["TrainingPopulationSummary", "TrainingPopulationSummaryAccumulator"]
