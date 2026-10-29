"""Train-only, bounded historical value replay for matched response fits."""

from __future__ import annotations

import json
import random
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.interaction_evidence.reference_adapter import load_stored_reference_case
from channelthermal.interaction_evidence.types import EvidenceSplit, SolveRecord, SolveStatus

from .sampling import ReceiverSamplingConfig, _replace_record_roles, _sample_rows, _subsample_role
from .thermal import (
    NEAR_INTERFACE_DISTANCE,
    PRESSURE_INLET_BAND_FRACTION,
    PRESSURE_OUTLET_BAND_FRACTION,
)


def _sample_value_record(record: SolveRecord, config: ReceiverSamplingConfig) -> SolveRecord:
    """Protect pressure, interface, and exact material peaks in one train case."""

    if (
        record.design.split is not EvidenceSplit.TRAIN
        or record.status is not SolveStatus.CONVERGED
        or record.output is None
    ):
        raise ValueError("Historical value replay accepts converged train records only.")
    roles = record.output.roles
    rng = np.random.default_rng(config.random_seed ^ zlib.crc32(record.record_id.encode("utf-8")))
    fluid = roles["fluid_fields"]
    x = fluid.query_features[:, 0]
    length_x = float(record.context.values["domain_length_x"])
    radius = float(record.context.values["module_radius"])
    centers = np.asarray([module.position_xy for module in record.design.active_modules])
    surface_distance = (
        np.linalg.norm(fluid.query_features[:, None, :2] - centers[None, :, :], axis=-1).min(axis=1) - radius
    )
    near = (surface_distance >= 0.0) & (surface_distance <= NEAR_INTERFACE_DISTANCE)
    pressure_bands = (x <= PRESSURE_INLET_BAND_FRACTION * length_x) | (
        x >= (1.0 - PRESSURE_OUTLET_BAND_FRACTION) * length_x
    )
    protected_fluid = np.flatnonzero(near | pressure_bands)
    fluid_rows, fluid_probability, _ = _sample_rows(
        fluid.values.shape[0], protected_fluid, config.max_fluid_queries, rng
    )

    solid = roles["solid_temperature"]
    if solid.receiver_module_ids is None:
        raise ValueError("Historical solid receivers must declare their module IDs.")
    module_ids = tuple(dict.fromkeys(solid.receiver_module_ids))
    rows_by_module = {
        module_id: np.asarray(
            [index for index, current in enumerate(solid.receiver_module_ids) if current == module_id],
            dtype=np.int64,
        )
        for module_id in module_ids
    }
    local_count = rows_by_module[module_ids[0]].size
    if any(rows.size != local_count for rows in rows_by_module.values()):
        raise ValueError("Historical solid modules must share one material grid.")
    hot_local: set[int] = set()
    for rows in rows_by_module.values():
        valid = solid.valid_mask[rows, 0]
        candidates = np.flatnonzero(valid)
        if not candidates.size:
            raise ValueError("Historical solid module has no observed peak.")
        temperatures = solid.values[rows[candidates], 0]
        keep = (
            candidates[np.argsort(temperatures, kind="stable")[-config.hot_solid_points_per_module :]]
            if config.hot_solid_points_per_module
            else np.empty(0, dtype=np.int64)
        )
        hot_local.update(int(index) for index in keep)
    local_rows, local_probability, _ = _sample_rows(
        local_count, np.asarray(sorted(hot_local), dtype=np.int64), config.solid_queries_per_module, rng
    )
    solid_rows = np.concatenate([rows[local_rows] for rows in rows_by_module.values()])
    solid_probability = np.tile(local_probability, len(rows_by_module))
    permutation = np.argsort(solid_rows, kind="stable")
    solid_rows = solid_rows[permutation]
    solid_probability = solid_probability[permutation]
    interface = roles["interface"]
    selections = {
        "fluid_fields": (fluid_rows, fluid_probability),
        "interface": (np.arange(interface.values.shape[0]), np.ones(interface.values.shape[0])),
        "solid_temperature": (solid_rows, solid_probability),
    }
    return _replace_record_roles(
        record, {name: _subsample_role(role, *selections[name]) for name, role in roles.items()}
    )


def _coverage_summary(original: SolveRecord, sampled: SolveRecord) -> dict[str, Any]:
    """Count protected supports against complete original receiver IDs."""

    assert original.output is not None and sampled.output is not None
    full = original.output.roles
    selected = sampled.output.roles
    fluid = full["fluid_fields"]
    x = fluid.query_features[:, 0]
    length_x = float(original.context.values["domain_length_x"])
    radius = float(original.context.values["module_radius"])
    centers = np.asarray([module.position_xy for module in original.design.active_modules])
    surface_distance = (
        np.linalg.norm(fluid.query_features[:, None, :2] - centers[None, :, :], axis=-1).min(axis=1) - radius
    )
    pressure = (x <= PRESSURE_INLET_BAND_FRACTION * length_x) | (x >= (1.0 - PRESSURE_OUTLET_BAND_FRACTION) * length_x)
    near = (surface_distance >= 0.0) & (surface_distance <= NEAR_INTERFACE_DISTANCE)
    ordinary = ~(pressure | near)
    selected_fluid_ids = set(selected["fluid_fields"].query_ids)
    solid = full["solid_temperature"]
    sampled_solid_ids = set(selected["solid_temperature"].query_ids)
    peak_coverage: dict[str, bool] = {}
    assert solid.receiver_module_ids is not None
    for module_id in original.output.active_module_ids:
        rows = np.flatnonzero(np.asarray(solid.receiver_module_ids) == module_id)
        valid_rows = rows[solid.valid_mask[rows, 0]]
        peak = valid_rows[np.argmax(solid.values[valid_rows, 0])]
        peak_coverage[module_id] = solid.query_ids[int(peak)] in sampled_solid_ids
    return {
        "original_rows": {name: int(role.values.shape[0]) for name, role in full.items()},
        "sampled_rows": {name: int(role.values.shape[0]) for name, role in selected.items()},
        "pressure_rows_original": int(pressure.sum()),
        "pressure_rows_retained": int(sum(fluid.query_ids[i] in selected_fluid_ids for i in np.flatnonzero(pressure))),
        "near_interface_rows_original": int(near.sum()),
        "near_interface_rows_retained": int(
            sum(fluid.query_ids[i] in selected_fluid_ids for i in np.flatnonzero(near))
        ),
        "ordinary_fluid_rows_original": int(ordinary.sum()),
        "ordinary_fluid_rows_retained": int(
            sum(fluid.query_ids[i] in selected_fluid_ids for i in np.flatnonzero(ordinary))
        ),
        "interface_ports_all_retained": selected["interface"].query_ids == full["interface"].query_ids,
        "solid_exact_peak_retained_by_module": peak_coverage,
    }


@dataclass(frozen=True)
class HistoricalValueSource:
    """Cycle all converged packed train cases in a count/context-balanced order."""

    dataset_path: Path
    case_ids: tuple[str, ...]
    sampling: ReceiverSamplingConfig
    _cache: dict[str, SolveRecord] = field(default_factory=dict, init=False, repr=False, compare=False)
    _coverage: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_dataset(
        cls, dataset: GlobalChannelThermalDataset, *, sampling: ReceiverSamplingConfig
    ) -> HistoricalValueSource:
        if dataset.split != "train":
            raise ValueError("Historical replay source must use the packed training split.")
        groups: dict[tuple[int, float], list[str]] = defaultdict(list)
        with h5py.File(dataset.path, "r") as handle:
            for case_id, count, converged in zip(
                dataset.selected_case_ids,
                dataset.selected_module_counts,
                dataset.selected_converged_flags,
                strict=True,
            ):
                if converged:
                    raw_config = handle["cases"][case_id]["case_config_json"][()]
                    if isinstance(raw_config, bytes):
                        raw_config = raw_config.decode("utf-8")
                    reynolds = float(json.loads(str(raw_config))["flow"]["re"])
                    groups[(int(count), reynolds)].append(str(case_id))
        rng = random.Random(sampling.random_seed)
        for members in groups.values():
            rng.shuffle(members)
        order: list[str] = []
        while any(groups.values()):
            for count in sorted(groups):
                if groups[count]:
                    order.append(groups[count].pop())
        if not order:
            raise ValueError("Historical replay requires converged packed training cases.")
        return cls(Path(dataset.path).resolve(), tuple(order), sampling)

    def load(self, case_id: str) -> SolveRecord:
        if case_id not in self.case_ids:
            raise ValueError("Historical replay case is outside the fixed train cohort.")
        if case_id in self._cache:
            return self._cache[case_id]
        record = load_stored_reference_case(self.dataset_path, case_id)
        sampled = _sample_value_record(record, self.sampling)
        self._coverage[case_id] = _coverage_summary(record, sampled)
        if len(self._cache) >= 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[case_id] = sampled
        return sampled

    @property
    def realized_coverage(self) -> dict[str, dict[str, Any]]:
        return {case_id: dict(summary) for case_id, summary in self._coverage.items()}


def select_broad_evaluation_cases(
    dataset: GlobalChannelThermalDataset,
    *,
    requested: int = 30,
) -> tuple[dict[str, Any], ...]:
    """Select the fixed broad train panel by module count and Reynolds quantiles.

    The initial panel takes minimum, median, and maximum Reynolds cases from
    each active-module-count stratum. If those rows do not reach ``requested``,
    it fills from the remaining sorted rows using the historical replay rule.
    """

    if dataset.split != "train":
        raise ValueError("The broad response evaluation panel uses packed train records only.")
    if requested <= 0:
        raise ValueError("requested broad case count must be positive.")
    strata: dict[int, list[tuple[str, float]]] = defaultdict(list)
    with h5py.File(dataset.path, "r") as handle:
        for case_id, module_count, converged in zip(
            dataset.selected_case_ids,
            dataset.selected_module_counts,
            dataset.selected_converged_flags,
            strict=True,
        ):
            if not converged:
                continue
            group = handle["cases"][str(case_id)]
            raw_config = group["case_config_json"][()]
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            reynolds = float(json.loads(str(raw_config))["flow"]["re"])
            if not np.isfinite(reynolds):
                raise ValueError(f"Broad evaluation case {case_id!r} has non-finite Reynolds number.")
            strata[int(module_count)].append((str(case_id), reynolds))
    for rows in strata.values():
        rows.sort(key=lambda item: (item[1], item[0]))

    selected: dict[str, dict[str, Any]] = {}
    for module_count, rows in sorted(strata.items()):
        if not rows:
            continue
        for quantile in (0.0, 0.5, 1.0):
            index = round(quantile * (len(rows) - 1))
            case_id, reynolds = rows[index]
            selected[case_id] = {
                "case_id": case_id,
                "module_count": module_count,
                "re": reynolds,
                "stratum_n": len(rows),
                "re_quantile": quantile,
            }
    if len(selected) < requested:
        for module_count, rows in sorted(strata.items()):
            for index, (case_id, reynolds) in enumerate(rows):
                if case_id in selected:
                    continue
                selected[case_id] = {
                    "case_id": case_id,
                    "module_count": module_count,
                    "re": reynolds,
                    "stratum_n": len(rows),
                    "re_quantile": index / max(len(rows) - 1, 1),
                }
                if len(selected) >= requested:
                    break
            if len(selected) >= requested:
                break
    rows = list(selected.values())[:requested]
    if len(rows) != requested:
        raise ValueError(f"Only selected {len(rows)} of {requested} requested broad train cases.")
    return tuple(sorted(rows, key=lambda row: (row["module_count"], row["re"], row["case_id"])))


def select_value_recovery_case_rows(
    strata: dict[int, list[tuple[str, float]]],
    *,
    requested: int = 16,
    focus_module_counts: tuple[int, ...] = (3, 5, 7, 10),
) -> tuple[dict[str, Any], ...]:
    """Choose a fixed historical panel covering every module-count stratum.

    One median-Reynolds case is selected from every available module count.
    Remaining slots cycle through low/high and inner Reynolds quantiles for the
    primary Thermal counts, then other strata. Ties are resolved by case ID.
    """

    if requested <= 0:
        raise ValueError("requested value-recovery case count must be positive.")
    normalized: dict[int, list[tuple[str, float]]] = {}
    for module_count, rows in sorted(strata.items()):
        current = sorted(
            ((str(case_id), float(reynolds)) for case_id, reynolds in rows),
            key=lambda item: (item[1], item[0]),
        )
        if not current or any(not np.isfinite(re) for _, re in current):
            raise ValueError(f"Module-count stratum {module_count} is empty or contains non-finite Reynolds values.")
        if len({case_id for case_id, _ in current}) != len(current):
            raise ValueError(f"Module-count stratum {module_count} contains duplicate case IDs.")
        normalized[int(module_count)] = current
    if not normalized:
        raise ValueError("Value-recovery historical panel requires at least one module-count stratum.")
    if requested < len(normalized):
        raise ValueError(f"Requested {requested} historical cases cannot cover {len(normalized)} module-count strata.")

    selected: dict[str, dict[str, Any]] = {}

    def add_quantile(module_count: int, quantile: float, selection_role: str) -> bool:
        rows = normalized[module_count]
        already = {str(row["case_id"]) for row in selected.values() if int(row["module_count"]) == module_count}
        available = [index for index, (case_id, _) in enumerate(rows) if case_id not in already]
        if not available:
            return False
        target = round(float(quantile) * (len(rows) - 1))
        index = min(available, key=lambda candidate: (abs(candidate - target), candidate))
        case_id, reynolds = rows[index]
        selected[case_id] = {
            "case_id": case_id,
            "module_count": module_count,
            "re": reynolds,
            "stratum_n": len(rows),
            "re_quantile_target": float(quantile),
            "selection_role": selection_role,
        }
        return True

    counts = tuple(normalized)
    for module_count in counts:
        add_quantile(module_count, 0.5, "module_count_median_re")

    focus = tuple(count for count in focus_module_counts if count in normalized)
    quantiles = (0.0, 1.0, 0.25, 0.75, 0.125, 0.875, 0.375, 0.625)
    priority_counts = focus + tuple(count for count in counts if count not in focus)
    for quantile in quantiles:
        progress = False
        for module_count in priority_counts:
            if len(selected) >= requested:
                break
            if add_quantile(module_count, quantile, f"re_quantile_{quantile:g}"):
                progress = True
        if len(selected) >= requested or not progress:
            break
    if len(selected) != requested:
        raise ValueError(f"Selected {len(selected)} of {requested} requested historical value-recovery cases.")
    return tuple(sorted(selected.values(), key=lambda row: (row["module_count"], row["re"], row["case_id"])))


def select_value_recovery_historical_cases(
    dataset: GlobalChannelThermalDataset,
    *,
    requested: int = 16,
    focus_module_counts: tuple[int, ...] = (3, 5, 7, 10),
) -> tuple[dict[str, Any], ...]:
    """Select a fixed no-new-solver value-recovery panel from packed train data."""

    if dataset.split != "train":
        raise ValueError("Value-recovery historical evaluation accepts packed train records only.")
    strata: dict[int, list[tuple[str, float]]] = defaultdict(list)
    with h5py.File(dataset.path, "r") as handle:
        for case_id, module_count, converged in zip(
            dataset.selected_case_ids,
            dataset.selected_module_counts,
            dataset.selected_converged_flags,
            strict=True,
        ):
            if not converged:
                continue
            raw_config = handle["cases"][str(case_id)]["case_config_json"][()]
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            reynolds = float(json.loads(str(raw_config))["flow"]["re"])
            strata[int(module_count)].append((str(case_id), reynolds))
    return select_value_recovery_case_rows(dict(strata), requested=requested, focus_module_counts=focus_module_counts)


__all__ = [
    "HistoricalValueSource",
    "select_broad_evaluation_cases",
    "select_value_recovery_case_rows",
    "select_value_recovery_historical_cases",
]
