"""Predeclared TRAIN-only numerical decisions for the Round6 readout study.

These reducers do not authorize a launch. Membership/query/transform parity,
native-law checks and the prospective paired resource reservation are separate
required evidence. Missing or zero-denominator comparisons remain unavailable.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

_PRIORITIES = {
    "thermal": ("v", "p"),
    "wind": ("near_turbine/Ux", "downstream_envelope/Ux"),
}
_PROTECTED = {
    "thermal": {
        "temperature_fluid": 1.05, "temperature_surface": 1.05,
        "temperature_material": 1.05, "q_proxy": 1.05,
        "u": 1.10, "omega": 1.10,
    },
    "wind": {
        "near_turbine/vector": 1.05, "downstream_envelope/vector": 1.05,
        "volume/Ux": 1.10, "volume/vector": 1.10,
        "background/Ux": 1.10, "background/vector": 1.10,
    },
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _mean(board: Mapping[str, Any], name: str) -> float | None:
    value = board.get(name)
    return _number(value.get("mean") if isinstance(value, Mapping) else value)


def error_ratio(numerator: Any, denominator: Any) -> float | None:
    """No epsilon, invented denominator or passing unavailable comparison."""
    first, second = _number(numerator), _number(denominator)
    return first / second if first is not None and second is not None and second > 0 else None


def priority_contrast(dataset: str, candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Geometric mean of ratios of native equal-case/layout mean RMSEs."""
    if dataset not in _PRIORITIES:
        raise ValueError("Round6 comparisons require thermal or wind.")
    ratios = {name: error_ratio(_mean(candidate, name), _mean(reference, name))
              for name in _PRIORITIES[dataset]}
    value = math.sqrt(math.prod(ratios.values())) if all(x is not None for x in ratios.values()) else None
    return {"ratio": value, "components": ratios,
            "native_means": {name: {"candidate": _mean(candidate, name), "reference": _mean(reference, name)}
                             for name in _PRIORITIES[dataset]}}


def _guards(dataset: str, candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    return {name: {"ratio": (ratio := error_ratio(_mean(candidate, name), _mean(reference, name))),
                   "candidate_native_mean": _mean(candidate, name), "reference_native_mean": _mean(reference, name),
                   "maximum_ratio": bound, "passed": ratio is not None and ratio <= bound}
            for name, bound in _PROTECTED[dataset].items()}


def local_readout_signal(
    dataset: str, *, edge: Mapping[str, Any], pool: Mapping[str, Any],
    parent: Mapping[str, Any], detailed_edge: Sequence[Mapping[str, Any]],
    detailed_pool: Sequence[Mapping[str, Any]], edge_native_objective: float,
    parent_native_objective: float,
) -> dict[str, Any]:
    """Section 6.4 signal on the same all-TRAIN audit and original TRAIN4.

    The caller must supply the four detailed fixtures in the same declared ID
    order, and the native objective on the same fixed sampling support.
    """
    if len(detailed_edge) != 4 or len(detailed_pool) != 4:
        raise ValueError("The local signal requires exactly the original four TRAIN fixtures.")
    pair = priority_contrast(dataset, edge, pool)
    versus_parent = priority_contrast(dataset, edge, parent)
    fixture_ratios = [priority_contrast(dataset, candidate, baseline)["ratio"]
                      for candidate, baseline in zip(detailed_edge, detailed_pool)]
    directions = sum(value is not None and value < 1 for value in fixture_ratios)
    protected = _guards(dataset, edge, pool)
    objective = error_ratio(edge_native_objective, parent_native_objective)
    checks = {
        "paired_priority_reduction_at_least_10pct": pair["ratio"] is not None and pair["ratio"] <= .90,
        "each_priority_no_more_than_5pct_worse": all(x is not None and x <= 1.05 for x in pair["components"].values()),
        "same_direction_on_at_least_three_train4": directions >= 3,
        "nonpriority_guards": all(item["passed"] for item in protected.values()),
        "parent_priority_reduction_at_least_5pct": versus_parent["ratio"] is not None and versus_parent["ratio"] <= .95,
        "original_native_objective_not_worse_than_parent": (
            _number(edge_native_objective) is not None and _number(parent_native_objective) is not None
            and edge_native_objective <= parent_native_objective),
    }
    return {"schema": "hyperedge_readout_local_signal_v1", "dataset": dataset,
            "useful_signal": all(checks.values()), "checks": checks,
            "misses": [name for name, passed in checks.items() if not passed],
            "edge_over_pool": pair, "edge_over_parent": versus_parent,
            "train4_priority_ratios": fixture_ratios, "improving_train4": directions,
            "protected_metrics": protected, "native_objective_ratio": objective,
            "native_objectives": {"edge": _number(edge_native_objective), "parent": _number(parent_native_objective)}}


def full_bank_signal(
    dataset: str, *, global_board: Mapping[str, Any], edge: Mapping[str, Any],
    parent: Mapping[str, Any], global_native_objective: float, parent_native_objective: float,
) -> dict[str, Any]:
    """Section 7 access signal; fixed global weights change access and weighting."""
    comparisons = {}
    for name, board in (("local_edge", edge), ("zero_fit_parent", parent)):
        contrast = priority_contrast(dataset, global_board, board)
        guards = _guards(dataset, global_board, board)
        comparisons[name] = {"priority": contrast, "protected_metrics": guards,
                             "passed": contrast["ratio"] is not None and contrast["ratio"] <= .90
                             and all(row["passed"] for row in guards.values())}
    objective_ok = (_number(global_native_objective) is not None and _number(parent_native_objective) is not None
                    and global_native_objective <= parent_native_objective)
    return {"schema": "hyperedge_readout_full_bank_signal_v1", "dataset": dataset,
            "useful_signal": all(row["passed"] for row in comparisons.values()) and objective_ok,
            "comparisons": comparisons, "original_native_objective_not_worse_than_parent": objective_ok,
            "native_objective_ratio": error_ratio(global_native_objective, parent_native_objective)}


def common_extension_signal(
    *, priority_fit300: float, priority_fit500: float,
    objective_epochs301_400: Sequence[float], objective_epochs401_500: Sequence[float],
    finite_gradients: bool,
) -> dict[str, Any]:
    """Section 6.3 TRAIN-only numerical half of the common extension gate.

    Evaluate once per arm. At least one arm must pass; both horizons and final
    evidence still need a prospective reservation. No DEV metric is accepted.
    """
    if len(objective_epochs301_400) != 100 or len(objective_epochs401_500) != 100:
        raise ValueError("Extension objective medians require both complete 100-epoch windows.")
    if type(finite_gradients) is not bool:
        raise TypeError("finite_gradients must be an explicit measured Boolean.")
    earlier = [_number(value) for value in objective_epochs301_400]
    later = [_number(value) for value in objective_epochs401_500]
    if any(value is None for value in (*earlier, *later)):
        raise ValueError("Native objectives must be finite nonnegative complete history values.")
    previous, current = statistics.median(earlier), statistics.median(later)
    priority_ratio = error_ratio(priority_fit500, priority_fit300)
    objective_ratio = error_ratio(current, previous)
    numerical_signal = ((priority_ratio is not None and priority_ratio <= .97)
                        or (objective_ratio is not None and objective_ratio <= .97))
    return {"schema": "hyperedge_readout_extension_signal_v1",
            "train_signal": numerical_signal and finite_gradients,
            "priority_fit500_over_fit300": priority_ratio,
            "native_objective_final100_over_previous100_median": objective_ratio,
            "native_objective_medians": {"fit301_400": previous, "fit401_500": current},
            "finite_gradients": finite_gradients,
            "requires_paired_completion_and_final_evidence_reservation": True}


def common_extension_decision(
    dataset: str, *, arms: Mapping[str, Mapping[str, Any]],
    source_checkpoint_sha256: str, audit_panel_sha256: str, paired_resource_reserved: bool,
) -> dict[str, Any]:
    """Bind the paired TRAIN decision to its fixed parent, panel and reservation."""
    if dataset not in _PRIORITIES or set(arms) != {"pool_first", "edge_first"}:
        raise ValueError("Common extension requires both local readers of one native dataset.")
    for digest in (source_checkpoint_sha256, audit_panel_sha256):
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Extension parent and all-TRAIN panel must have exact SHA-256 bindings.")
    if type(paired_resource_reserved) is not bool:
        raise TypeError("The paired resource reservation must be an explicit Boolean.")
    for signal in arms.values():
        if signal.get("schema") != "hyperedge_readout_extension_signal_v1":
            raise ValueError("Extension arms must use the predeclared TRAIN signal reducer.")
    healthy = all(signal.get("finite_gradients") is True for signal in arms.values())
    useful = any(signal.get("train_signal") is True for signal in arms.values())
    return {"schema": "hyperedge_readout_extension_decision_v1", "dataset": dataset,
            "phase": "frozen_probe", "source_checkpoint_sha256": source_checkpoint_sha256,
            "audit_panel_sha256": audit_panel_sha256, "fit_age_from": 500, "fit_age_to": 1000,
            "arms": {name: dict(signal) for name, signal in arms.items()},
            "paired_resource_reserved": paired_resource_reserved,
            "approved": healthy and useful and paired_resource_reserved,
            "both_arms_finite_gradients": healthy, "any_arm_train_signal": useful}


__all__ = ["common_extension_decision", "common_extension_signal", "error_ratio", "full_bank_signal",
           "local_readout_signal", "priority_contrast"]
