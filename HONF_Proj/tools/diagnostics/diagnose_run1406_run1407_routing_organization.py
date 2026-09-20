"""Read-only routing/organization audit for mature HONF checkpoints.

This diagnostic is intentionally separate from the accuracy and timing
evaluators.  It consumes the maintained map/auxiliary API from a real forward
pass and reports what the routing actually selected:

* positive source/query support degree and entropy;
* empty-group mass, group occupancy and occupancy balance;
* unique support density and backend executed/padded row ledgers;
* P0/P1/P2 membership/occupancy turnover when phase maps are exported; and
* simple case-feature correlations and frozen uniform/permuted routing
  interventions on a small anchor set.

The script never trains, writes a checkpoint, changes a checkpoint, or edits a
report.  Maps for anchors are retained as compressed ``npz`` files while all
90-case summaries are kept in CSV/JSON so a later report can be regenerated
without repeating the expensive forwards.

Example::

    PYTHONPATH=src:Case_ThermalChannel/src python \
      tools/diagnostics/diagnose_run1406_run1407_routing_organization.py \
      --checkpoint 1406=/path/to/best_by_field_mse_model.pt \
      --checkpoint 1407=/path/to/best_by_field_mse_model.pt \
      --checkpoint 1404=/path/to/best_by_field_mse_model.pt \
      --checkpoint 1804=/path/to/best_by_field_mse_model.pt \
      --device cuda:0 \
      --output-dir diagnostics/generated/run1406_run1407_best5000_routing_organization_20260920
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "Case_ThermalChannel" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))
if str(PROJECT_ROOT / "tools" / "diagnostics") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools" / "diagnostics"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_stage3_interface_study as stage3
import torch

GROUP_COUNT = 6
EPS = 1.0e-12
DEFAULT_ANCHORS = ("0273", "0653", "0298", "0302")
PHASE_PREFIXES = {
    "P0": "initial_port_",
    "P1": "provisional_",
    "P2": "",
}
SOURCE_KINDS = ("module", "environment")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _scalar(value: Any, default: float = float("nan")) -> float:
    try:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.asarray(value).reshape(-1)[0]
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, IndexError):
        return default


def _flatten_batch(value: Any) -> np.ndarray:
    array = _array(value)
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


def _first_mapping(output: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = output.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _merge_mappings(output: Mapping[str, Any], *names: str) -> dict[str, Any]:
    """Combine auxiliary maps while retaining the later map's aliases.

    The mature Stage-7 implementations split the source memberships and
    query attention across ``organizer_aux`` and ``routing_aux``.  A first-map
    lookup therefore misses a valid mixed API (notably Run 1404).  This helper
    keeps the diagnostic read-only and makes that split explicit.
    """

    merged: dict[str, Any] = {}
    for name in names:
        value = output.get(name)
        if isinstance(value, Mapping):
            merged.update(value)
    return merged


def _find(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _entropy(probability: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    values = np.where(values > 0.0, values, 0.0)
    return -np.sum(np.where(values > 0.0, values * np.log(np.maximum(values, EPS)), 0.0), axis=axis)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    denominator = values.sum(axis=-1, keepdims=True)
    return np.divide(values, denominator, out=np.zeros_like(values), where=denominator > EPS)


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size or float(values.sum()) <= EPS:
        return float("nan")
    ordered = np.sort(np.maximum(values, 0.0))
    index = np.arange(1, len(ordered) + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * ordered) / (len(ordered) * ordered.sum())) - (len(ordered) + 1.0) / len(ordered))


def _assignment_metrics(values: np.ndarray, prefix: str, active: np.ndarray | None = None) -> dict[str, float]:
    """Summarize source-to-group memberships without assigning empty rows."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return {f"{prefix}_{name}": float("nan") for name in (
            "source_count", "support_degree_mean", "support_degree_min", "support_degree_max",
            "support_degree_zero_fraction", "row_entropy_mean", "row_entropy_norm_mean",
            "row_effective_groups_mean", "group_occupancy_entropy_norm", "group_occupancy_gini",
            "group_occupancy_min", "group_occupancy_max", "group_occupancy_cv",
        )}
    if active is None:
        active_mask = np.ones(matrix.shape[0], dtype=bool)
    else:
        active_mask = np.asarray(active, dtype=bool).reshape(-1)
        if active_mask.size != matrix.shape[0]:
            active_mask = np.ones(matrix.shape[0], dtype=bool)
    matrix = np.maximum(matrix[active_mask], 0.0)
    if not matrix.size:
        return {f"{prefix}_{name}": float("nan") for name in (
            "source_count", "support_degree_mean", "support_degree_min", "support_degree_max",
            "support_degree_zero_fraction", "row_entropy_mean", "row_entropy_norm_mean",
            "row_effective_groups_mean", "group_occupancy_entropy_norm", "group_occupancy_gini",
            "group_occupancy_min", "group_occupancy_max", "group_occupancy_cv",
        )}
    rows = _normalize_rows(matrix)
    support_degree = (matrix > EPS).sum(axis=1).astype(np.float64)
    row_entropy = _entropy(rows, axis=1)
    group_mass = rows.mean(axis=0)
    group_mass = np.maximum(group_mass, 0.0)
    group_mass /= max(float(group_mass.sum()), EPS)
    group_entropy = float(_entropy(group_mass[None, :], axis=1)[0])
    occupancy_mean = float(group_mass.mean())
    occupancy_std = float(group_mass.std())
    return {
        f"{prefix}_source_count": float(matrix.shape[0]),
        f"{prefix}_support_degree_mean": float(support_degree.mean()),
        f"{prefix}_support_degree_min": float(support_degree.min()),
        f"{prefix}_support_degree_max": float(support_degree.max()),
        f"{prefix}_support_degree_zero_fraction": float(np.mean(support_degree == 0.0)),
        f"{prefix}_row_entropy_mean": float(row_entropy.mean()),
        f"{prefix}_row_entropy_norm_mean": float(row_entropy.mean() / max(math.log(max(matrix.shape[1], 2)), EPS)),
        f"{prefix}_row_effective_groups_mean": float(np.exp(row_entropy).mean()),
        f"{prefix}_group_occupancy_entropy_norm": group_entropy / max(math.log(max(matrix.shape[1], 2)), EPS),
        f"{prefix}_group_occupancy_gini": _gini(group_mass),
        f"{prefix}_group_occupancy_min": float(group_mass.min()),
        f"{prefix}_group_occupancy_max": float(group_mass.max()),
        f"{prefix}_group_occupancy_cv": occupancy_std / max(occupancy_mean, EPS),
    }


def _query_metrics(alpha: np.ndarray, prefix: str = "query") -> dict[str, float]:
    values = np.asarray(alpha, dtype=np.float64)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[1] == 0:
        return {f"{prefix}_{name}": float("nan") for name in (
            "count", "support_degree_mean", "support_degree_min", "support_degree_max",
            "support_degree_zero_fraction", "entropy_mean", "entropy_norm_mean", "effective_groups_mean",
            "max_weight_mean", "mass_std_mean",
        )}
    values = np.maximum(values, 0.0)
    rows = _normalize_rows(values)
    support = (values > EPS).sum(axis=1).astype(np.float64)
    entropy = _entropy(rows, axis=1)
    return {
        f"{prefix}_count": float(values.shape[0]),
        f"{prefix}_support_degree_mean": float(support.mean()),
        f"{prefix}_support_degree_min": float(support.min()),
        f"{prefix}_support_degree_max": float(support.max()),
        f"{prefix}_support_degree_zero_fraction": float(np.mean(support == 0.0)),
        f"{prefix}_entropy_mean": float(entropy.mean()),
        f"{prefix}_entropy_norm_mean": float(entropy.mean() / max(math.log(max(values.shape[1], 2)), EPS)),
        f"{prefix}_effective_groups_mean": float(np.exp(entropy).mean()),
        f"{prefix}_max_weight_mean": float(rows.max(axis=1).mean()),
        f"{prefix}_mass_std_mean": float(rows.std(axis=1).mean()),
    }


def _source_matrix(aux: Mapping[str, Any], kind: str, phase_prefix: str = "") -> np.ndarray | None:
    key = f"{phase_prefix}group_control_{kind}_incidence"
    value = aux.get(key)
    if value is None:
        # Generic Stage-7 models use A_mh/A_eh.  This keeps the same CSV useful
        # for the 1404/1804 context rows without pretending they are the
        # group-control equation.
        fallback = "A_mh" if kind == "module" else "A_eh"
        value = aux.get(fallback)
    return None if value is None else _flatten_batch(value)


def _query_matrix(aux: Mapping[str, Any], phase_prefix: str = "") -> np.ndarray | None:
    value = aux.get(f"{phase_prefix}group_control_query_routing")
    if value is None:
        value = aux.get("query_hyper_attention", aux.get("alpha"))
    if value is None:
        return None
    result = _flatten_batch(value)
    return result.reshape(-1, result.shape[-1]) if result.ndim > 2 else result


def _measure(aux: Mapping[str, Any], kind: str) -> np.ndarray | None:
    value = aux.get(f"group_control_{kind}_measure")
    return None if value is None else _flatten_batch(value)


def _phase_aux(output: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    """Merge phase-prefixed records with the final routing auxiliary map."""

    if phase == "P2":
        return _first_mapping(output, "routing_aux", "interaction_aux")
    source = _first_mapping(output, "interaction_aux", "routing_aux")
    prefix = PHASE_PREFIXES[phase]
    return {key[len(prefix):]: value for key, value in source.items() if isinstance(key, str) and key.startswith(prefix)}


def _phase_raw_aux(output: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    """Keep original prefixed keys for scalar execution-ledger extraction."""

    source = _first_mapping(output, "interaction_aux", "routing_aux")
    prefix = PHASE_PREFIXES[phase]
    if phase == "P2":
        return source
    return {key: value for key, value in source.items() if isinstance(key, str) and key.startswith(prefix)}


def _group_mass(matrix: np.ndarray, measures: np.ndarray | None, active: np.ndarray | None) -> np.ndarray:
    values = np.maximum(np.asarray(matrix, dtype=np.float64), 0.0)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if active is None or len(active) != values.shape[0]:
        active_mask = np.ones(values.shape[0], dtype=bool)
    else:
        active_mask = np.asarray(active, dtype=bool)
    if measures is None:
        weights = np.ones(values.shape[0], dtype=np.float64)
    else:
        weights = _flatten_batch(measures).reshape(-1).astype(np.float64)
        if len(weights) != values.shape[0]:
            weights = np.ones(values.shape[0], dtype=np.float64)
    values = values[active_mask]
    weighted = values * weights[active_mask, None]
    mass = weighted.sum(axis=0)
    return mass / max(float(mass.sum()), EPS)


def _empty_group_metrics(alpha: np.ndarray, incidence: np.ndarray, measures: np.ndarray | None, active: np.ndarray | None, prefix: str) -> dict[str, float]:
    inc = np.maximum(np.asarray(incidence, dtype=np.float64), 0.0)
    if inc.ndim == 3 and inc.shape[0] == 1:
        inc = inc[0]
    active_mask = np.ones(inc.shape[0], dtype=bool) if active is None or len(active) != inc.shape[0] else np.asarray(active, dtype=bool)
    occupied = np.any((inc > EPS) & active_mask[:, None], axis=0)
    routing = _normalize_rows(_flatten_batch(alpha))
    empty = ~occupied
    return {
        f"{prefix}_empty_group_count": float(empty.sum()),
        f"{prefix}_empty_group_fraction": float(empty.mean()),
        f"{prefix}_empty_group_query_mass_mean": float(routing[:, empty].sum(axis=1).mean()) if np.any(empty) else 0.0,
        f"{prefix}_occupied_group_count": float(occupied.sum()),
        f"{prefix}_occupied_group_fraction": float(occupied.mean()),
        f"{prefix}_group_mass_gini": _gini(_group_mass(inc, measures, active)),
    }


def _density_metrics(alpha: np.ndarray, incidence: np.ndarray, measures: np.ndarray | None, active: np.ndarray | None, prefix: str) -> dict[str, float]:
    routing = _normalize_rows(_flatten_batch(alpha))
    inc = np.maximum(_flatten_batch(incidence), 0.0)
    if active is None or len(active) != inc.shape[0]:
        active_mask = np.ones(inc.shape[0], dtype=bool)
    else:
        active_mask = np.asarray(active, dtype=bool)
    positive = np.einsum("qk,sk->qs", routing > EPS, inc > EPS) > 0.0
    positive[:, ~active_mask] = False
    unique_pairs = positive.sum()
    dense = positive.shape[0] * int(active_mask.sum())
    # Count q->group->source paths explicitly.  A scalar einsum with a
    # repeated boolean index (``qk,sk->``) performs a logical reduction and
    # returns True/False rather than the required integer path total.
    path_mask = (routing > EPS)[:, None, :] & ((inc > EPS) & active_mask[:, None])[None, :, :]
    logical = int(path_mask.sum())
    query_count = max(positive.shape[0], 1)
    return {
        f"{prefix}_unique_pair_density": float(unique_pairs / max(dense, 1)),
        f"{prefix}_unique_pair_count_mean": float(unique_pairs / query_count),
        f"{prefix}_logical_path_count_mean": float(logical / query_count),
        f"{prefix}_logical_to_unique_multiplicity": float(logical / max(unique_pairs, 1)),
        f"{prefix}_valid_source_count": float(active_mask.sum()),
    }


def _ledger_values(raw_aux: Mapping[str, Any], kind: str, phase: str, q_count: int, source_count: int, active_count: int) -> dict[str, float]:
    prefix = "" if phase == "P2" else PHASE_PREFIXES[phase]
    base = f"{prefix}group_control_{kind}_"
    result: dict[str, float] = {}
    aliases = {
        # ``fine_rows`` is the wrapper's accumulated actual backend row total.
        # The *_forward and *_padded fields are retained only as raw reported
        # counters: with 8192 probes the model internally chunks receivers,
        # and those fields can describe one chunk rather than the total.
        "fine_rows": ("fine_rows", "fine_rows_forward"),
        "fine_rows_forward_reported": ("fine_rows_forward", "fine_forward_rows"),
        "fine_rows_padded_reported": ("fine_rows_padded", "padded_rows"),
        "logical_paths": ("logical_paths",),
        "unique_pairs": ("unique_pairs",),
        "valid_pair_denominator": ("valid_pair_denominator",),
        "padded_pair_denominator": ("padded_pair_denominator",),
        "scalar_control_rows": ("scalar_control_rows",),
        "checkpoint_recomputations": ("checkpoint_recomputations", "fine_rows_recompute"),
        "source_projection_rows": ("source_projection_rows",),
    }
    for name, suffixes in aliases.items():
        value = None
        for suffix in suffixes:
            value = raw_aux.get(base + suffix)
            if value is not None:
                break
        number = _scalar(value)
        result[f"{kind}_{phase}_{name}"] = number
    fine_rows = result[f"{kind}_{phase}_fine_rows"]
    valid_rows = result[f"{kind}_{phase}_valid_pair_denominator"]
    padded_denominator = result[f"{kind}_{phase}_padded_pair_denominator"]
    # The pair denominators are the only q-wide population totals.  Derive
    # all fractions from them and from the accumulated actual row count; do
    # not mix in per-chunk *_forward/*_padded counters.
    if math.isfinite(fine_rows):
        result[f"{kind}_{phase}_executed_rows"] = fine_rows
    else:
        result[f"{kind}_{phase}_executed_rows"] = float("nan")
    if math.isfinite(fine_rows) and math.isfinite(valid_rows):
        executed_padded = max(fine_rows - valid_rows, 0.0)
        # A sparse executor can execute a strict subset of the valid
        # denominator, so the fraction of executed rows that are valid is
        # capped at one.  ``executed_rows_over_valid`` below retains the
        # directional coverage/expansion ratio without this cap.
        result[f"{kind}_{phase}_executed_valid_fraction"] = min(valid_rows / max(fine_rows, 1.0), 1.0)
        result[f"{kind}_{phase}_executed_padded_fraction"] = executed_padded / max(fine_rows, 1.0)
        result[f"{kind}_{phase}_executed_rows_over_valid"] = fine_rows / max(valid_rows, 1.0)
        result[f"{kind}_{phase}_executed_padded_rows"] = executed_padded
    else:
        for name in ("executed_valid_fraction", "executed_padded_fraction", "executed_rows_over_valid", "executed_padded_rows"):
            result[f"{kind}_{phase}_{name}"] = float("nan")
    if math.isfinite(valid_rows) and math.isfinite(padded_denominator):
        denominator = valid_rows + padded_denominator
        result[f"{kind}_{phase}_valid_fraction_of_pair_denominator"] = valid_rows / max(denominator, 1.0)
        result[f"{kind}_{phase}_padded_fraction_of_pair_denominator"] = padded_denominator / max(denominator, 1.0)
        result[f"{kind}_{phase}_executed_rows_over_pair_denominator"] = fine_rows / max(denominator, 1.0) if math.isfinite(fine_rows) else float("nan")
        result[f"{kind}_{phase}_unexecuted_padded_rows"] = max(padded_denominator - (fine_rows - valid_rows), 0.0) if math.isfinite(fine_rows) else float("nan")
    else:
        for name in ("valid_fraction_of_pair_denominator", "padded_fraction_of_pair_denominator", "executed_rows_over_pair_denominator", "unexecuted_padded_rows"):
            result[f"{kind}_{phase}_{name}"] = float("nan")
    return result


def _case_features(sample: Mapping[str, Any]) -> dict[str, float]:
    structure = sample["structure"]
    centers = _array(structure["module_centers"])
    present = _array(structure["module_present"]).reshape(-1) > 0.5
    powers = _array(structure.get("heat_powers", np.zeros(len(centers)))).reshape(-1)
    active_centers = centers[present]
    if len(active_centers) > 1:
        distance = np.linalg.norm(active_centers[:, None] - active_centers[None, :], axis=-1)
        distance += np.eye(len(active_centers)) * 1.0e9
        nearest = distance.min(axis=1)
    else:
        nearest = np.asarray([float("nan")])
    return {
        "module_count": float(present.sum()),
        "re": _scalar(structure.get("re")),
        "u_in": _scalar(structure.get("u_in")),
        "heat_power_mean": float(np.mean(powers[present])) if np.any(present) else float("nan"),
        "heat_power_std": float(np.std(powers[present])) if np.any(present) else float("nan"),
        "heat_power_range": float(np.ptp(powers[present])) if np.any(present) else float("nan"),
        "module_nearest_spacing_mean": float(np.mean(nearest)),
        "module_nearest_spacing_min": float(np.min(nearest)),
    }


def _phase_metrics(output: Mapping[str, Any], sample: Mapping[str, Any], *, phase: str, q_count: int) -> dict[str, Any]:
    aux = _phase_aux(output, phase)
    raw = _phase_raw_aux(output, phase)
    module = _source_matrix(aux, "module")
    environment = _source_matrix(aux, "environment")
    query = _query_matrix(aux)
    if module is None or environment is None or query is None:
        return {"phase": phase, "status": "unavailable"}
    present = _array(sample["structure"]["module_present"]).reshape(-1) > 0.5
    measures_m = _measure(aux, "module")
    measures_e = _measure(aux, "environment")
    row: dict[str, Any] = {"phase": phase, "status": "ok"}
    row.update(_query_metrics(query, "query"))
    row.update(_assignment_metrics(module, "module", present))
    row.update(_assignment_metrics(environment, "environment"))
    row.update(_empty_group_metrics(query, module, measures_m, present, "module"))
    row.update(_empty_group_metrics(query, environment, measures_e, None, "environment"))
    row.update(_density_metrics(query, module, measures_m, present, "module"))
    row.update(_density_metrics(query, environment, measures_e, None, "environment"))
    module_active = int(np.sum(present[: module.shape[0]])) if len(present) >= module.shape[0] else int(module.shape[0])
    row.update(_ledger_values(raw, "module", phase, int(_flatten_batch(query).shape[0]), module.shape[0], module_active))
    row.update(_ledger_values(raw, "environment", phase, int(_flatten_batch(query).shape[0]), environment.shape[0], environment.shape[0]))
    mass_m = _group_mass(module, measures_m, present)
    mass_e = _group_mass(environment, measures_e, None)
    for index, value in enumerate(mass_m):
        row[f"module_group_mass_{index}"] = float(value)
    for index, value in enumerate(mass_e):
        row[f"environment_group_mass_{index}"] = float(value)
    return row


def _phase_turnover(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for kind in SOURCE_KINDS:
        old = previous.get(f"{kind}_assignment")
        new = current.get(f"{kind}_assignment")
        if old is None or new is None:
            result[f"{kind}_turnover_l1"] = float("nan")
            result[f"{kind}_turnover_support_flip_fraction"] = float("nan")
            continue
        old_array = _normalize_rows(_flatten_batch(old))
        new_array = _normalize_rows(_flatten_batch(new))
        if old_array.shape != new_array.shape:
            result[f"{kind}_turnover_l1"] = float("nan")
            result[f"{kind}_turnover_support_flip_fraction"] = float("nan")
            continue
        result[f"{kind}_turnover_l1"] = float(np.mean(np.sum(np.abs(old_array - new_array), axis=-1)))
        result[f"{kind}_turnover_support_flip_fraction"] = float(np.mean((old_array > EPS) != (new_array > EPS)))
    return result


def _phase_assignment_snapshot(output: Mapping[str, Any], phase: str) -> dict[str, Any]:
    aux = _phase_aux(output, phase)
    result: dict[str, Any] = {}
    for kind in SOURCE_KINDS:
        value = _source_matrix(aux, kind)
        if value is not None:
            result[f"{kind}_assignment"] = value
    return result


def _generic_organization_metrics(output: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    """Use the available Stage-7 maps for Run 1404/1804 context rows.

    Run 1404 exposes source memberships (``A_mh``/``A_eh``) in its organizer
    map and query-to-hyperedge attention in its routing map.  Run 1804 is the
    dense comparator: it has no learned source-membership map, but does expose
    a dense environment attention tensor.  These are intentionally reported
    under distinct API labels rather than coerced into group-control metrics.
    """

    aux = _merge_mappings(output, "organizer_aux", "routing_aux", "interaction_aux")
    module = _find(aux, "A_mh")
    env = _find(aux, "A_eh")
    alpha = _find(aux, "query_hyper_attention", "alpha")
    present = _array(sample["structure"]["module_present"]).reshape(-1) > 0.5
    row: dict[str, Any] = {}
    if module is not None and env is not None and alpha is not None:
        alpha_array = _flatten_batch(alpha)
        module_array = _flatten_batch(module)
        env_array = _flatten_batch(env)
        row["organization_api"] = "generic_A_mh_A_eh_plus_query_hyper_attention"
        row.update(_assignment_metrics(module_array, "module", present))
        row.update(_assignment_metrics(env_array, "environment"))
        row.update(_query_metrics(alpha_array, "query"))
        row.update(_empty_group_metrics(alpha_array, module_array, None, present, "module"))
        row.update(_empty_group_metrics(alpha_array, env_array, None, None, "environment"))
        row.update(_density_metrics(alpha_array, module_array, None, present, "module"))
        row.update(_density_metrics(alpha_array, env_array, None, None, "environment"))
    else:
        row["organization_api"] = "dense_environment_attention" if _find(aux, "dense_environment_attention") is not None else "unavailable"

    dense_attention = _find(aux, "dense_environment_attention")
    if dense_attention is not None:
        dense = _array(dense_attention)
        # The dense comparator exports [B, heads, Q, E].  Averaging heads is a
        # descriptive reduction only; it does not claim a sparse group map.
        if dense.ndim == 4:
            dense = dense.mean(axis=1)
        if dense.ndim >= 3:
            dense = dense.reshape(-1, dense.shape[-1])
        if dense.ndim == 2:
            row.update(_query_metrics(dense, "dense_environment_query"))
        row["dense_environment_attention_rank"] = float(_array(dense_attention).ndim)
        row["dense_environment_attention_environment_count"] = float(_array(dense_attention).shape[-1])

    # Context fractions/neighbour counts are useful dense-baseline descriptors
    # even when no source-membership map exists.
    for name in (
        "local_neighbor_count", "main_context_fraction", "coarse_context_fraction",
        "local_context_fraction", "main_context_norm", "coarse_context_norm",
        "local_context_norm", "dense_module_context_norm",
    ):
        value = _find(aux, name)
        if value is not None:
            row[f"dense_{name}_mean"] = float(np.nanmean(_array(value)))
    return row


def _output_prediction_delta(base: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("pred_field", "pred_interface", "pred_internal_temperature", "pred_port_condition"):
        left = base.get(key)
        right = variant.get(key)
        if not torch.is_tensor(left) or not torch.is_tensor(right) or left.shape != right.shape:
            continue
        diff = (left.detach().float() - right.detach().float()).abs()
        result[f"{key}_mean_abs"] = float(diff.mean().cpu())
        result[f"{key}_max_abs"] = float(diff.max().cpu())
        result[f"{key}_relative_l2"] = float(torch.linalg.vector_norm(left.detach().float() - right.detach().float()) / torch.linalg.vector_norm(left.detach().float()).clamp_min(1.0e-12))
    return result


@contextmanager
def _temporary_route_intervention(model: Any, mode: str) -> Iterator[None]:
    """Patch the group router only for one frozen forward.

    ``uniform_query_group`` removes learned query-to-group selection while
    retaining all source memberships and fine values.  ``permute_group_keys``
    swaps learned group-key rows; it is a label-sensitive intervention and is
    explicitly interpreted as such.  Both restore the live model object on
    exit and never touch a checkpoint file.
    """

    backend = getattr(getattr(model, "core", None), "backend", None)
    if backend is None or not hasattr(backend, "_route"):
        raise RuntimeError("model does not expose the group-control backend route")
    original_route = backend._route
    router = getattr(backend, "router", None)
    original_codes = None
    if mode == "permute_group_keys":
        if router is None or not hasattr(router, "group_codes"):
            raise RuntimeError("router does not expose group_codes")
        original_codes = router.group_codes.detach().clone()
        permutation = torch.roll(torch.arange(original_codes.shape[0], device=original_codes.device), shifts=1)
        with torch.no_grad():
            router.group_codes.copy_(original_codes[permutation])

    def patched_route(state: Any, encoded: Any, receivers: Any, receiver_features: Any = None) -> Any:
        route = original_route(state, encoded, receivers, receiver_features)
        if mode == "uniform_query_group":
            assignment = torch.full_like(route.assignment, 1.0 / float(route.assignment.shape[-1]))
            return replace(route, assignment=assignment)
        return route

    backend._route = patched_route
    try:
        yield
    finally:
        backend._route = original_route
        if original_codes is not None:
            with torch.no_grad():
                router.group_codes.copy_(original_codes)


def _correlation_rows(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str], metric_names: Sequence[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature in feature_names:
        for metric in metric_names:
            x = np.asarray([_scalar(row.get(feature)) for row in rows], dtype=np.float64)
            y = np.asarray([_scalar(row.get(metric)) for row in rows], dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < 3 or np.std(x[valid]) <= EPS or np.std(y[valid]) <= EPS:
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = float(np.corrcoef(x[valid], y[valid])[0, 1])
                xr = np.argsort(np.argsort(x[valid]))
                yr = np.argsort(np.argsort(y[valid]))
                spearman = float(np.corrcoef(xr, yr)[0, 1])
            output.append({"feature": feature, "metric": metric, "n": int(valid.sum()), "pearson": pearson, "spearman": spearman})
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: _jsonable(value) for key, value in row.items()} for row in rows])


def _render_figures(output_dir: Path, case_rows: Sequence[Mapping[str, Any]], phase_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = list(dict.fromkeys(str(row["label"]) for row in case_rows))
    colors = {label: color for label, color in zip(labels, ("#3366cc", "#dc3912", "#109618", "#ff9900"))}
    metrics = [
        ("query_support_degree_mean", "Query positive support degree", "group"),
        ("module_unique_pair_density", "Module unique-pair density", "density"),
        ("environment_unique_pair_density", "Environment unique-pair density", "density"),
        ("module_empty_group_query_mass_mean", "Module empty-group query mass", "mass"),
        ("environment_empty_group_query_mass_mean", "Environment empty-group query mass", "mass"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.5, 3.6), constrained_layout=True)
    for axis, (metric, title, _kind) in zip(axes, metrics):
        means = []
        spreads = []
        present_labels = []
        for label in labels:
            values = np.asarray([_scalar(row.get(metric)) for row in case_rows if str(row["label"]) == label], dtype=np.float64)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            present_labels.append(label)
            means.append(values.mean())
            spreads.append(values.std())
        x = np.arange(len(present_labels))
        axis.bar(x, means, yerr=spreads, capsize=3, color=[colors[label] for label in present_labels])
        axis.set_xticks(x, present_labels, rotation=35, ha="right")
        axis.set_title(title, fontsize=9)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(figures / "routing_support_summary.png", dpi=180)
    plt.close(fig)

    group_rows = [row for row in case_rows if str(row["label"]) in ("1406", "1407")]
    if group_rows:
        fig, axis = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
        width = 0.36
        for offset, label in enumerate(("1406", "1407")):
            rows = [row for row in group_rows if str(row["label"]) == label]
            masses = np.asarray([[ _scalar(row.get(f"module_group_mass_{g}")) for g in range(GROUP_COUNT)] for row in rows], dtype=np.float64)
            mean = np.nanmean(masses, axis=0) if masses.size else np.full(GROUP_COUNT, np.nan)
            axis.bar(np.arange(GROUP_COUNT) + (offset - 0.5) * width, mean, width=width, label=label)
        axis.set_xticks(np.arange(GROUP_COUNT), [f"g{g}" for g in range(GROUP_COUNT)])
        axis.set_ylabel("mean normalized module group mass")
        axis.set_title("Mature module-group occupancy")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        fig.savefig(figures / "module_group_occupancy_1406_1407.png", dpi=180)
        plt.close(fig)

    phase_subset = [row for row in phase_rows if str(row.get("label")) in ("1406", "1407") and row.get("status") == "ok"]
    if phase_subset:
        fig, axis = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
        for label in ("1406", "1407"):
            subset = [row for row in phase_subset if str(row["label"]) == label]
            by_phase = {str(row["phase"]): row for row in subset}
            values = []
            names = []
            for phase in ("P0", "P1", "P2"):
                row = by_phase.get(phase)
                if row is not None:
                    names.append(phase)
                    values.append(_scalar(row.get("module_group_occupancy_gini")))
            if values:
                axis.plot(names, values, marker="o", label=label)
        axis.set_ylabel("module-group occupancy Gini")
        axis.set_title("Phase occupancy concentration")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.savefig(figures / "phase_occupancy_concentration.png", dpi=180)
        plt.close(fig)
    return [str(path) for path in sorted(figures.glob("*.png"))]


def _parse_checkpoints(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"checkpoint must be LABEL=PATH, got {value!r}")
        label, raw = value.split("=", 1)
        path = Path(raw).expanduser().resolve()
        if not label or not path.is_file():
            raise FileNotFoundError(f"invalid checkpoint {value!r}")
        if label in result:
            raise ValueError(f"duplicate checkpoint label {label!r}")
        result[label] = path
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--query-count", type=int, default=8192)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--anchor-case-id", action="append", default=list(DEFAULT_ANCHORS))
    parser.add_argument("--skip-interventions", action="store_true")
    parser.add_argument("--intervention-query-count", type=int, default=2048)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoints = _parse_checkpoints(args.checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)
    case_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    map_paths: list[str] = []
    provenance: dict[str, Any] = {}
    requested_cases = [str(x) for x in args.case_id]
    anchors = [str(x) for x in args.anchor_case_id]
    for label, checkpoint_path in checkpoints.items():
        checkpoint = stage3.load_trusted_checkpoint(checkpoint_path, map_location="cpu")
        spec = stage3.CheckpointSpec(label=label, path=checkpoint_path)
        model, _ = stage3._load_model_spec(spec, device)
        class LoaderArgs:
            split = args.split
            dataset = args.dataset
        dataset, dataset_path = stage3._load_dataset(checkpoint, LoaderArgs())
        available = [str(value) for value in dataset.selected_case_ids]
        case_ids = requested_cases or available
        if args.max_cases is not None:
            case_ids = case_ids[: int(args.max_cases)]
        missing = sorted(set(case_ids) - set(available))
        if missing:
            raise KeyError(f"missing cases for {label}: {missing}")
        provenance[label] = {
            "checkpoint": str(checkpoint_path),
            "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
            "model_family": checkpoint.get("model_family"),
            "forward_architecture": checkpoint.get("model_config", {}).get("core_honf", {}).get("forward_architecture"),
            "dataset": str(dataset_path),
            "case_count": len(case_ids),
            "selection_policy": "explicit_best_by_field_mse_checkpoint",
        }
        for position, case_id in enumerate(case_ids, start=1):
            sample = stage3.select_sample(dataset, case_id, 0)
            query = stage3._query_points(sample, int(args.query_count))
            with torch.inference_mode():
                output = stage3._forward_batch(
                    model,
                    sample,
                    query,
                    device,
                    return_routing_maps=True,
                    return_organizer_passes=True,
                )
            row: dict[str, Any] = {"label": label, "case_id": case_id, **_case_features(sample)}
            final_aux = _first_mapping(output, "routing_aux", "interaction_aux")
            group_map_available = _source_matrix(final_aux, "module") is not None and _query_matrix(final_aux) is not None
            if group_map_available:
                row["organization_api"] = "group_control"
                final_phase = _phase_metrics(output, sample, phase="P2", q_count=len(query))
                row.update(final_phase)
                for phase in ("P0", "P1", "P2"):
                    phase_row = _phase_metrics(output, sample, phase=phase, q_count=len(query))
                    phase_rows.append({"label": label, "case_id": case_id, **phase_row})
                snapshots = {phase: _phase_assignment_snapshot(output, phase) for phase in ("P0", "P1", "P2")}
                for left, right in (("P0", "P1"), ("P1", "P2"), ("P0", "P2")):
                    row.update({f"{left}_to_{right}_{k}": v for k, v in _phase_turnover(snapshots[left], snapshots[right]).items()})
            else:
                row.update(_generic_organization_metrics(output, sample))
            row["routing_api"] = "group_control" if group_map_available else "generic_or_unavailable"
            case_rows.append(row)
            if case_id in anchors and label in ("1406", "1407") and group_map_available:
                aux = _first_mapping(output, "routing_aux", "interaction_aux")
                arrays: dict[str, Any] = {
                    "query_xy": query,
                    "module_present": _array(sample["structure"]["module_present"]),
                    "module_centers": _array(sample["structure"]["module_centers"]),
                }
                for name in (
                    "group_control_query_routing", "group_control_query_logits",
                    "group_control_module_incidence", "group_control_environment_incidence",
                    "group_control_module_overlap", "group_control_environment_overlap",
                ):
                    if name in aux:
                        arrays[name] = _array(aux[name])
                destination = output_dir / "maps" / f"{label}__{case_id}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, **arrays)
                map_paths.append(str(destination))
            if (not args.skip_interventions) and case_id in anchors and label in ("1406", "1407") and group_map_available:
                base_field = output.get("pred_field")
                for mode in ("uniform_query_group", "permute_group_keys"):
                    try:
                        with _temporary_route_intervention(model, mode), torch.inference_mode():
                            variant = stage3._forward_batch(
                                model,
                                sample,
                                query[: int(args.intervention_query_count)],
                                device,
                                return_routing_maps=True,
                                return_organizer_passes=False,
                            )
                        base_small = {"pred_field": base_field[..., : int(args.intervention_query_count), :]} if torch.is_tensor(base_field) else output
                        delta = _output_prediction_delta(base_small, variant)
                        intervention_rows.append({"label": label, "case_id": case_id, "intervention": mode, "status": "ok", **delta})
                    except Exception as exc:  # noqa: BLE001 - preserve per-anchor availability
                        intervention_rows.append({"label": label, "case_id": case_id, "intervention": mode, "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"})
            del output
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[{label}] {position}/{len(case_ids)} case={case_id}", flush=True)
        dataset.close()
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    feature_names = [
        "module_count", "re", "u_in", "heat_power_mean", "heat_power_std", "heat_power_range",
        "module_nearest_spacing_mean", "module_nearest_spacing_min",
    ]
    metric_names = [
        "query_support_degree_mean", "query_entropy_norm_mean", "module_support_degree_mean",
        "environment_support_degree_mean", "module_unique_pair_density", "environment_unique_pair_density",
        "module_empty_group_query_mass_mean", "environment_empty_group_query_mass_mean",
        "module_group_occupancy_gini", "environment_group_occupancy_gini",
    ]
    correlations: list[dict[str, Any]] = []
    for label in checkpoints:
        correlations.extend([{"label": label, **row} for row in _correlation_rows([x for x in case_rows if x["label"] == label], feature_names, metric_names)])
    summary: dict[str, Any] = {}
    for label in checkpoints:
        members = [row for row in case_rows if row["label"] == label]
        numeric_keys = sorted({key for row in members for key, value in row.items() if isinstance(value, (float, int, np.number))})
        summary[label] = {
            "case_count": len(members),
            "metrics": {
                key: {
                    "mean": float(np.nanmean([_scalar(row.get(key)) for row in members])),
                    "median": float(np.nanmedian([_scalar(row.get(key)) for row in members])),
                    "p05": float(np.nanquantile([_scalar(row.get(key)) for row in members], 0.05)),
                    "p95": float(np.nanquantile([_scalar(row.get(key)) for row in members], 0.95)),
                }
                for key in numeric_keys
                if any(math.isfinite(_scalar(row.get(key))) for row in members)
            },
        }
    _write_csv(output_dir / "case_metrics.csv", case_rows)
    _write_csv(output_dir / "phase_metrics.csv", phase_rows)
    _write_csv(output_dir / "correlations.csv", correlations)
    _write_csv(output_dir / "interventions.csv", intervention_rows)
    figures = _render_figures(output_dir, case_rows, phase_rows)
    manifest = {
        "schema_version": 1,
        "task": "run1406_run1407_best5000_routing_organization",
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "protocol": {
            "split": args.split,
            "query_count": int(args.query_count),
            "anchors": anchors,
            "case_selection": "all_available_cases_unless_case_id_or_max_cases_supplied",
            "timed_maps": False,
            "training": False,
            "checkpoint_mutation": False,
        },
        "ledger_semantics": {
            "fine_rows": "accumulated actual backend fine-row total returned by the wrapper",
            "fine_rows_forward_reported": "raw auxiliary counter retained for audit; with receiver chunking it may describe one chunk and is not used as a q-wide total",
            "fine_rows_padded_reported": "raw auxiliary counter retained for audit; with receiver chunking it may describe one chunk and is not used as a q-wide total",
            "valid_pair_denominator": "q-wide valid source-pair population emitted by the backend",
            "padded_pair_denominator": "q-wide inactive/padded source-slot population emitted by the backend",
            "executed_padded_rows": "max(fine_rows - valid_pair_denominator, 0), so padded execution is not confused with the q-wide padded denominator",
            "ratio_policy": "all valid/padded/executed fractions use fine_rows and the two q-wide pair denominators; no per-chunk ratio is averaged",
        },
        "provenance": provenance,
        "summary": summary,
        "artifacts": {
            "case_metrics": str(output_dir / "case_metrics.csv"),
            "phase_metrics": str(output_dir / "phase_metrics.csv"),
            "correlations": str(output_dir / "correlations.csv"),
            "interventions": str(output_dir / "interventions.csv"),
            "maps": map_paths,
            "figures": figures,
        },
        "interpretation_limits": [
            "Positive support and executed rows are learned interaction evidence, not physical causality.",
            "Uniform query routing removes query-group selectivity but is not a retrained accuracy ablation.",
            "Permuted group keys are a label-sensitive frozen intervention; group-label symmetry is not assumed.",
            "Generic 1404/1804 rows use their A_mh/A_eh API when available and are not treated as group-control arrays.",
            "Missing P0/P1 maps remain unavailable; no phase is inferred from P2 counts.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
