"""Measure and evaluate Thermal's matured root/two/four packet actions.

This driver binds every output to one exact controlled Run1509 G checkpoint,
uses only saved train-split physical labels, and never invokes a reference
solver. The eight original atlas families fit/cross-fit the input-only risk
models. The two predeclared Stage-B fit families are development-only; the
two previously inspected held-after-rank-freeze families are an additional
development audit, not untouched generalization evidence.

The default device is CPU. Select a GPU explicitly only after the forward
endpoint and device window have been approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

CASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CASE_ROOT / "scripts"))

import run_active_packet_forward as forward
import run_active_packet_stage_c as stage_c
from channelthermal.response_control.active_packet import ThermalCoverPlanBuilder
from channelthermal.response_control.contracts import DesignInput, context_inputs
from channelthermal.response_control.maturation import THERMAL_ACTION_PATHS, available_frontier_for_paths
from channelthermal.response_control.native import DifferentiableThermalOperator
from channelthermal.response_control.thermal import reduce_native_thermal_quantities

from honf_forward_core.interface_fields.action_aware_frontier import (
    describe_realized_plan,
    incumbent_role_log_limit,
    receiver_role_descriptors,
    select_action_by_risk,
)
from honf_forward_core.interface_fields.action_risk_fit import (
    ActionEvidenceRow,
    crossfit_action_risk,
    evaluate_action_policy,
    fit_action_risk_head,
    fit_ridge_action_baseline,
)
from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan
from honf_forward_core.interface_fields.budgeted_frontier import (
    canonical_pair_catalog,
    enumerate_frontier_cuts,
    frontier_paths,
)
from honf_runtime.compat import load_trusted_checkpoint

DEFAULT_CONTROLLED_RUN = PROJECT_ROOT / "diagnostics/generated/thermal_maturation_20260929/controlled_run"
DEFAULT_CONTROLLED_MANIFEST = DEFAULT_CONTROLLED_RUN / "run_manifest.json"
DEFAULT_SOURCE_CHECKPOINT = stage_c.DEFAULT_SOURCE_CHECKPOINT
DEFAULT_ATLAS = stage_c.DEFAULT_ATLAS
DEFAULT_COVERAGE_MANIFEST = stage_c.DEFAULT_COVERAGE_ROOT / "thermal_coverage_calls_v3.json"
DEFAULT_PREDECLARED = stage_c.DEFAULT_PREDECLARED
DEFAULT_OUTPUT = PROJECT_ROOT / "diagnostics/generated/thermal_maturation_20260929/action_selector"
ACTION_KEYS = ("root", "two_packet", "four_packet", "full_access")
SPARSE_ACTION_KEYS = ACTION_KEYS[:-1]
MECHANISM_ORDER = ("MM", "ME", "EM", "QM", "QE")
TRAIN_PASSES_PER_ACTION = 2
PRIMARY_CAPACITY = 0.90
ROLE_RELATIVE_ALLOWANCE = 0.10
HISTORICAL_VISIT_MINIMUM = 1200
FAMILY_VISIT_MINIMUM = 100
RISK_FIT_UPDATES = 120
RISK_CROSSFIT_FOLDS = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _family_aliases(family_id: str) -> frozenset[str]:
    """Treat only the documented 0001/0273 duplicate as one family.

    Other atlas contexts sharing a geometry ID at different Reynolds numbers
    are correlated development evidence, but they are distinct physical
    contexts and must remain visible as such.
    """
    value = str(family_id)
    aliases = {value}
    identifiers = {match.group(1).zfill(4) for match in re.finditer(r"case(\d{1,4})", value)}
    if value.isdigit() and len(value) <= 4:
        identifiers.add(value.zfill(4))
    if value.startswith("duplicate_family:"):
        identifiers.update(match.group(1).zfill(4) for match in re.finditer(r"(\d{1,4})", value))
    if identifiers & {"0001", "0273"}:
        aliases.add("duplicate_family:0001+0273")
    return frozenset(aliases)


def shared_geometry_across_splits(
    split_families: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Report, without rejecting, contexts that share geometry across splits."""
    owners: dict[str, set[str]] = defaultdict(set)
    for split, family_ids in split_families.items():
        for family_id in family_ids:
            for match in re.finditer(r"case(\d{1,4})", str(family_id)):
                owners[match.group(1).zfill(4)].add(str(split))
    return {
        geometry: sorted(splits)
        for geometry, splits in sorted(owners.items()) if len(splits) > 1
    }


def validate_family_partitions(
    split_families: Mapping[str, Sequence[str]],
) -> None:
    """Reject duplicate physical-family aliases crossing action-table splits."""
    alias_owner: dict[str, str] = {}
    for split, family_ids in split_families.items():
        for family_id in family_ids:
            for alias in _family_aliases(str(family_id)):
                prior = alias_owner.setdefault(alias, str(split))
                if prior != str(split):
                    raise ValueError(
                        f"Physical family alias {alias!r} appears in both {prior!r} and {split!r}."
                    )


def _load_checkpoint_binding(
    manifest_path: Path,
    checkpoint_path: Path,
    *,
    expected_arm: str = "G",
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest_path = manifest_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("The exact controlled manifest and endpoint checkpoint are required.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arm_record = manifest.get("arms", {}).get(expected_arm)
    if not isinstance(arm_record, Mapping):
        raise TypeError(f"Controlled manifest has no {expected_arm} endpoint record.")
    recorded_path = Path(str(arm_record.get("latest_checkpoint", ""))).expanduser().resolve()
    recorded_hash = str(arm_record.get("latest_checkpoint_sha256", ""))
    actual_hash = _sha256(checkpoint_path)
    if checkpoint_path != recorded_path or actual_hash != recorded_hash:
        raise ValueError("Endpoint checkpoint path or SHA256 differs from the controlled run manifest.")
    if manifest.get("run_id") != "Thermal_Controlled_Maturation_from_Run1509_u200_20260929":
        raise ValueError("The controlled run is not the authorized Run1509 maturation branch.")
    base_run = Path(str(manifest.get("base_run", ""))).expanduser()
    if base_run.name != stage_c.DEFAULT_FORWARD_RUN.name:
        raise ValueError("The controlled branch does not descend from the exact Run1509 attempt07 run.")
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    update = int(payload.get("actual_optimizer_updates", -1))
    training_config = payload.get("training_config")
    internal_arm = training_config.get("arm") if isinstance(training_config, Mapping) else None
    if (arm_record.get("arm") != expected_arm
            or payload.get("arm") != "R_response"
            or internal_arm != "R_response"
            or update != int(arm_record.get("latest_update", -2))):
        raise ValueError("Checkpoint arm/update does not match the manifest-bound endpoint.")
    return manifest, payload, actual_hash


def _state_parts(payload: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    physical = payload.get("physical_model_state")
    route = payload.get("route_model_state")
    if isinstance(physical, Mapping) and isinstance(route, Mapping):
        return dict(physical), dict(route)
    bundle = payload.get("model")
    if not isinstance(bundle, Mapping):
        raise TypeError("Controlled checkpoint has no physical/organizer model state.")
    physical_state = {
        str(key).removeprefix("forward_model."): value
        for key, value in bundle.items()
        if str(key).startswith("forward_model.")
    }
    route_state = {
        str(key).removeprefix("route_model."): value
        for key, value in bundle.items()
        if str(key).startswith("route_model.")
    }
    if not physical_state or not route_state:
        raise ValueError("Controlled checkpoint model keys must include forward_model.* and route_model.*.")
    return physical_state, route_state


def summarize_training_exposure(
    metric_path: Path,
    *,
    checkpoint_update: int,
    family_ids: Sequence[str],
    expected_capacity: float = PRIMARY_CAPACITY,
    route_work_path: Path | None = None,
) -> dict[str, Any]:
    """Bind action exposure to completed updates and their executed baseline cuts."""
    realized_by_update: dict[int, dict[str, Any]] = {}
    if route_work_path is not None:
        with route_work_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("arm") != "G" or record.get("state") != "baseline":
                    continue
                update = int(record.get("optimizer_update", -1))
                if update < 0:
                    raise ValueError(f"Missing baseline optimizer update at {route_work_path}:{line_number}.")
                if update > checkpoint_update:
                    continue
                frontier = tuple(int(value) for value in record.get("frontier", ()))
                if not frontier or len(frontier) != len(set(frontier)):
                    raise ValueError(f"Invalid executed baseline cut at {route_work_path}:{line_number}.")
                paths = tuple(str(value) for value in record.get("frontier_paths") or ())
                if paths and len(paths) != len(frontier):
                    raise ValueError(f"Executed path/count mismatch at {route_work_path}:{line_number}.")
                candidate = {"frontier": frontier, "paths": paths}
                prior = realized_by_update.get(update)
                if prior is not None:
                    if prior["frontier"] != frontier or (
                        prior["paths"] and paths and prior["paths"] != paths
                    ):
                        raise ValueError(f"Conflicting executed baseline cuts at update {update}.")
                    if prior["paths"]:
                        continue
                realized_by_update[update] = candidate
    by_update: dict[int, dict[str, Any]] = {}
    with metric_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            update = int(row.get("completed_update", -1))
            if update < 0:
                raise ValueError(f"Missing completed_update at {metric_path}:{line_number}.")
            if update > checkpoint_update:
                continue
            prior = by_update.get(update)
            if prior is not None:
                prior_key = (
                    prior.get("training_family_id"), prior.get("training_metadata", {}).get("action"),
                    prior.get("training_metadata", {}).get("primary_pass"),
                    prior.get("training_metadata", {}).get("capacity_fraction"),
                )
                current_key = (
                    row.get("training_family_id"), row.get("training_metadata", {}).get("action"),
                    row.get("training_metadata", {}).get("primary_pass"),
                    row.get("training_metadata", {}).get("capacity_fraction"),
                )
                if prior_key != current_key:
                    raise ValueError(f"Conflicting duplicate optimizer update {update} in {metric_path}.")
                continue
            by_update[update] = row

    family_set = {str(value) for value in family_ids}
    counts: Counter[tuple[str, str, str, float]] = Counter()
    pass_families: dict[tuple[str, int, float], Counter[str]] = defaultdict(Counter)
    scaffold_counts: Counter[str] = Counter()
    historical_visits = 0
    visits_per_response_family: Counter[str] = Counter()
    realized_k_counts: dict[str, Counter[int]] = defaultdict(Counter)
    realized_frontier_counts: dict[str, Counter[str]] = defaultdict(Counter)
    realized_path_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing_realized_updates: dict[str, list[int]] = defaultdict(list)
    for update, row in by_update.items():
        family = str(row.get("training_family_id") or "")
        metadata = row.get("training_metadata") or {}
        action = str(metadata.get("action") or "")
        phase = str(metadata.get("phase") or "")
        capacity = float(metadata.get("capacity_fraction", math.nan))
        if family in family_set:
            visits_per_response_family[family] += 1
        if row.get("historical_case_id") is not None:
            historical_visits += 1
        if action == "two_packet_scaffold" and family in family_set and math.isclose(capacity, expected_capacity):
            scaffold_counts[family] += 1
        if action in SPARSE_ACTION_KEYS and family in family_set and not bool(metadata.get("full_access_replay")):
            counts[(family, action, phase, capacity)] += 1
            if phase == "action_family" and math.isclose(capacity, expected_capacity):
                realized = realized_by_update.get(update)
                if route_work_path is not None and realized is None:
                    missing_realized_updates[action].append(update)
                    continue
                if realized is not None:
                    frontier = tuple(realized["frontier"])
                    realized_k_counts[action][len(frontier)] += 1
                    realized_frontier_counts[action][",".join(map(str, frontier))] += 1
                    paths = tuple(realized["paths"])
                    if paths:
                        realized_path_counts[action]["|".join(paths)] += 1
                try:
                    primary_pass = int(metadata["primary_pass"])
                except (KeyError, TypeError, ValueError):
                    continue
                pass_families[(action, primary_pass, capacity)][family] += 1

    per_action: dict[str, Any] = {}
    for action in SPARSE_ACTION_KEYS:
        complete_passes = []
        action_passes: dict[str, int] = {}
        for (logged_action, primary_pass, capacity), families in sorted(pass_families.items()):
            if logged_action != action or not math.isclose(capacity, expected_capacity):
                continue
            if all(families[family] >= 1 for family in family_set):
                complete_passes.append(primary_pass)
                action_passes[str(primary_pass)] = sum(families.values())
        per_family = {
            family: sum(
                count for (logged_family, logged_action, phase, capacity), count in counts.items()
                if logged_family == family and logged_action == action
                and phase == "action_family" and math.isclose(capacity, expected_capacity)
            )
            for family in sorted(family_set)
        }
        per_action[action] = {
            "required_complete_primary_passes": TRAIN_PASSES_PER_ACTION,
            "complete_primary_pass_ids": complete_passes,
            "complete_primary_pass_count": len(complete_passes),
            "trained_sparse": (
                len(complete_passes) >= TRAIN_PASSES_PER_ACTION
                and route_work_path is not None
                and not missing_realized_updates[action]
            ),
            "per_family_action_capacity_updates": per_family,
            "updates_by_primary_pass": action_passes,
            "realized_baseline_cut_audited": route_work_path is not None,
            "missing_realized_baseline_updates": sorted(missing_realized_updates[action]),
            "realized_baseline_raw_k_counts": dict(sorted(realized_k_counts[action].items())),
            "realized_baseline_frontier_index_counts": dict(sorted(realized_frontier_counts[action].items())),
            "realized_baseline_path_counts": dict(sorted(realized_path_counts[action].items())),
            "realized_path_labelled_update_count": sum(realized_path_counts[action].values()),
            "realized_path_labels_complete": (
                bool(realized_k_counts[action])
                and sum(realized_path_counts[action].values())
                == sum(realized_k_counts[action].values())
            ),
            "requested_action_is_path_resolving_family": True,
        }
    minimum_family_visits = min((visits_per_response_family[family] for family in family_set), default=0)
    historical_passes = None
    historical_case_count = 0
    # The controlled manifest records the historical order length separately;
    # callers attach that denominator once available.
    return {
        "checkpoint_update": checkpoint_update,
        "optimizer_updates_deduplicated": len(by_update),
        "updates_after_checkpoint_ignored": "yes",
        "primary_capacity": expected_capacity,
        "per_action": per_action,
        "two_packet_scaffold_updates_by_family": dict(sorted(scaffold_counts.items())),
        "response_family_visit_counts": {
            family: int(visits_per_response_family[family]) for family in sorted(family_set)
        },
        "minimum_response_family_visits": int(minimum_family_visits),
        "historical_replay_visits": historical_visits,
        "historical_case_count_from_manifest": historical_case_count,
        "historical_replay_passes_observed": historical_passes,
        "route_work_path": None if route_work_path is None else str(route_work_path),
    }


def complete_maturity_gates(
    exposure: Mapping[str, Any], *, historical_case_count: int,
) -> dict[str, Any]:
    visits = int(exposure["historical_replay_visits"])
    minimum_family = int(exposure["minimum_response_family_visits"])
    action_passes = {
        action: bool(exposure["per_action"][action]["trained_sparse"])
        for action in SPARSE_ACTION_KEYS
    }
    historical_passes = visits // historical_case_count if historical_case_count > 0 else 0
    gates = {
        "all_sparse_actions_have_two_complete_0p90_family_passes": all(action_passes.values()),
        "historical_replay_visits_at_least_1200": visits >= HISTORICAL_VISIT_MINIMUM,
        "historical_replay_passes_at_least_two": historical_passes >= 2,
        "every_response_family_has_at_least_100_visits": minimum_family >= FAMILY_VISIT_MINIMUM,
    }
    return {
        "mature_physical_study": all(gates.values()),
        "maturity_status": "mature" if all(gates.values()) else "resource_censored_or_incomplete_exposure",
        "action_specific_exposure_eligible": action_passes,
        "historical_case_count_from_manifest": historical_case_count,
        "historical_replay_passes_observed": historical_passes,
        "gates": gates,
    }


class _CapturePlanBuilder:
    """Proxy that saves exact per-state encoded inputs, scores, plan and work."""

    def __init__(self, builder: Any) -> None:
        self.base = builder
        self.snapshots: list[dict[str, Any]] = []

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __call__(self, encoded: Any, module_state: torch.Tensor, trees: Sequence[Any]):
        device = module_state.device
        self._sync(device)
        started = time.perf_counter()
        plans = self.base(encoded, module_state, trees)
        self._sync(device)
        planner_seconds = time.perf_counter() - started
        self.snapshots.append({
            "encoded": self.base.last_encoded,
            "module_state": module_state.detach(),
            "trees": tuple(self.base.last_trees),
            "scores": tuple(self.base.last_scores),
            "plans": tuple(self.base.last_plans),
            "records": tuple(self.base.last_records),
            "budget_fractions": dict(self.base.budget_fractions),
            "planner_seconds": planner_seconds,
        })
        return plans


def _target_free_pair(raw_stencil: Any, *, seed: int) -> Any:
    if "i_plus" not in raw_stencil.variants:
        raise ValueError(f"Saved training atlas lacks i_plus: {raw_stencil.physical_family_id}")
    pair = stage_c.ThermalStageStencil(
        baseline=raw_stencil.baseline,
        variants={"i_plus": raw_stencil.variants["i_plus"]},
        evidence_manifest={
            "physical_family_id": raw_stencil.physical_family_id,
            "response_source": "saved_train_split_atlas_or_predeclared_coverage",
            "target_free_query_sampling": True,
            "targets_used_for_query_selection": False,
        },
    )
    config = stage_c.ReceiverSamplingConfig(
        max_fluid_queries=128,
        solid_queries_per_module=8,
        hot_solid_points_per_module=0,
        random_seed=int(seed),
    )
    return stage_c._sample_coverage_pair(pair, config=config, target_free=True)


def _load_action_stencils(args: argparse.Namespace) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    atlas_dir = args.atlas_dir.expanduser().resolve()
    train: list[Any] = []
    atlas_hashes: dict[str, str] = {}
    for index, anchor_id in enumerate(forward.TRAIN_ATLAS_IDS):
        path = atlas_dir / f"train_{anchor_id}_responses.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw, _ = stage_c.load_response_atlas_stencil(path)
        if raw.split.value != "train":
            raise ValueError(f"Action evaluation accepts saved train labels only: {path}")
        train.append(_target_free_pair(raw, seed=args.panel_seed + index))
        atlas_hashes[path.name] = _sha256(path)
    if len({item.physical_family_id for item in train}) != len(train):
        raise ValueError("Original training atlases contain repeated physical-family IDs.")

    dev = list(stage_c._load_coverage_families(
        calls_path=args.coverage_manifest.expanduser().resolve(),
        predeclared_path=args.predeclared_manifest.expanduser().resolve(),
        partition="fit",
        target_free_sampling=True,
        sampling_seed=args.panel_seed + 101,
    ))
    held = list(stage_c._load_coverage_families(
        calls_path=args.coverage_manifest.expanduser().resolve(),
        predeclared_path=args.predeclared_manifest.expanduser().resolve(),
        partition="held_out_train_development",
        target_free_sampling=True,
        sampling_seed=args.panel_seed + 211,
    ))
    if [item.physical_family_id for item in dev] != [
        "active_packet:case0319:Re70", "active_packet:case0349:Re50",
    ]:
        raise ValueError("The Stage-B model-development family set/order changed.")
    if [item.physical_family_id for item in held] != [
        "active_packet:case0319:Re50", "active_packet:case0349:Re70",
    ]:
        raise ValueError("The previously inspected held-family audit set/order changed.")
    splits = {
        "train": [item.physical_family_id for item in train],
        "dev": [item.physical_family_id for item in dev],
        "held_family_audit": [item.physical_family_id for item in held],
    }
    validate_family_partitions(splits)

    repeat_raw, _ = stage_c.load_response_atlas_stencil(
        atlas_dir / f"train_{forward.TRAIN_ATLAS_IDS[0]}_responses.npz"
    )
    repeat = _target_free_pair(repeat_raw, seed=args.panel_seed + 1009)
    return {
        "train": train,
        "dev": dev,
        "held_family_audit": held,
        "train_query_repeat": [repeat],
    }, {
        "split_families": splits,
        "atlas_file_sha256": atlas_hashes,
        "query_sampling": "fixed geometry and masks; no target-value hot-point selection",
        "primary_panel_seed": args.panel_seed,
        "disjoint_query_repeat": {
            "family_id": repeat.physical_family_id,
            "panel_seed": args.panel_seed + 1009,
            "partition": "train; same physical family remains grouped in cross-fit",
        },
        "known_duplicate_alias_check": "only documented 0001+0273 duplicate is canonicalized before exact-context split validation",
        "shared_geometry_across_partitions": shared_geometry_across_splits(splits),
        "cross_partition_contexts_are_correlated_development_evidence": True,
    }


def _new_operator(model: Any, source_payload: Mapping[str, Any], template: Mapping[str, Any], *, capture: bool) -> DifferentiableThermalOperator:
    return DifferentiableThermalOperator(
        model,
        template,
        dataset_config=source_payload["train_config"]["dataset"],
        normalization_stats=source_payload["global_normalization_stats"],
        query_batch_size=512,
        capture_packet_inputs=capture,
    )


def _saved_reference_role_scales(stencil: Any) -> np.ndarray:
    """Compute absolute and finite-change label scales from saved train labels."""
    absolute_scales: list[float] = []
    finite_scales: list[float] = []
    records = stencil.records
    for role_name, reference_role in stencil.baseline.output.roles.items():
        for channel in range(len(reference_role.channel_names)):
            absolute_rms = []
            for record in records:
                role = record.output.roles[role_name]
                absolute_rms.append(stage_c._weighted_rms(
                    np.asarray(role.values[:, channel]),
                    np.asarray(role.valid_mask[:, channel]),
                    np.asarray(role.quadrature_weights),
                ))
            absolute_scales.append(float(np.sqrt(np.mean(np.square(absolute_rms)))))
            base = stencil.baseline.output.roles[role_name]
            finite_rms = []
            for record in records[1:]:
                role = record.output.roles[role_name]
                delta = np.asarray(role.values[:, channel]) - np.asarray(base.values[:, channel])
                mask = np.asarray(role.valid_mask[:, channel]) & np.asarray(base.valid_mask[:, channel])
                finite_rms.append(stage_c._weighted_rms(
                    delta, mask, np.asarray(base.quadrature_weights),
                ))
            finite_scales.append(float(np.sqrt(np.mean(np.square(finite_rms)))) if finite_rms else 0.0)
    scales = np.asarray(absolute_scales + finite_scales, dtype=np.float64)
    if scales.ndim != 1 or not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("Saved reference role scales must be finite and nonnegative.")
    return scales


def _train_numerical_floor_by_role(train_stencils: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Use a fixed 1e-6 train-median reference-scale floor, with 1e-8 units floor."""
    if not train_stencils:
        raise ValueError("At least one train stencil is required to establish numerical floors.")
    scales = np.stack([_saved_reference_role_scales(stencil) for stencil in train_stencils])
    median_scale = np.median(scales, axis=0)
    floor = np.maximum(1.0e-8, 1.0e-6 * median_scale)
    return floor, median_scale


def _fixed_train_role_log_limits(
    rows: Sequence[ActionEvidenceRow],
    *,
    train_families: set[str],
    allowances: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Conservative minimum physical-gate log limit over primary train families."""
    family_rows: list[dict[str, Any]] = []
    limits: list[torch.Tensor] = []
    for family in sorted(train_families):
        candidates = [
            row for row in rows
            if row.family_key == family and row.action_key == "full_access"
            and row.case_key.endswith("|primary_0")
        ]
        if len(candidates) != 1:
            # Every original train family has one primary case, but preserve
            # support for future multi-panel train tables by using its first.
            candidates = [
                row for row in rows
                if row.family_key == family and row.action_key == "full_access"
                and "|query_repeat_" not in row.case_key
            ][:1]
        if len(candidates) != 1:
            raise ValueError(f"Train-only gate calibration needs one primary full-access row for {family!r}.")
        row = candidates[0]
        allowance = allowances[row.case_key].detach().cpu().double()
        limit = incumbent_role_log_limit(
            row.incumbent_role_error.detach().cpu().double(),
            row.numerical_floor.detach().cpu().double(),
            allowance,
            relative_allowance=ROLE_RELATIVE_ALLOWANCE,
        )
        limits.append(limit)
        family_rows.append({
            "family_key": family,
            "case_key": row.case_key,
            "physical_gate_log_limit": limit.tolist(),
        })
    fixed = torch.stack(limits).amin(dim=0)
    return fixed, family_rows


def _route_support_for_catalog(
    plan: MechanismPlan,
    mechanism: str,
    catalog: Any,
    cut: Sequence[int],
    *,
    module_present: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return packet source signatures and receiver-resolved hard pair support."""
    cut_tensor = torch.as_tensor(tuple(int(value) for value in cut), device=plan.split_gates.device, dtype=torch.long)
    permission = plan.permission_matrix(mechanism, module_present=module_present).detach()
    permission_binary = bool(torch.all(
        torch.isclose(permission, torch.zeros_like(permission), atol=1e-6, rtol=0)
        | torch.isclose(permission, torch.ones_like(permission), atol=1e-6, rtol=0)
    ))
    packet_sources = permission[cut_tensor].bool() & catalog.source_validity[None, :].bool()
    access = plan.access_for(
        mechanism,
        catalog.receiver_coordinates,
        module_present=module_present,
    ).detach()
    if access.shape != catalog.pair_validity.shape:
        raise ValueError(
            f"{mechanism} canonical access {tuple(access.shape)} does not match receiver/source validity "
            f"{tuple(catalog.pair_validity.shape)}."
        )
    eligible = catalog.pair_validity.bool()
    valid_access_values = access[eligible]
    binary_access = bool(torch.all(
        torch.isclose(valid_access_values, torch.zeros_like(valid_access_values), atol=1e-6, rtol=0)
        | torch.isclose(valid_access_values, torch.ones_like(valid_access_values), atol=1e-6, rtol=0)
    ))
    if not permission_binary or not binary_access:
        raise ValueError(f"{mechanism} hard permissions and canonical access must be binary in this evaluation.")
    support = (access > 0.5) & eligible
    detail = {
        "selected_hard_pairs": int(support.sum().detach().cpu()),
        "eligible_pairs": int(eligible.sum().detach().cpu()),
        "active_receiver_rows": int(support.any(dim=1).sum().detach().cpu()),
        "active_source_columns": int(support.any(dim=0).sum().detach().cpu()),
        "packet_source_slots_by_frontier_node": [
            int(value) for value in packet_sources.sum(dim=1).detach().cpu().tolist()
        ],
        "canonical_hard_pairs_by_receiver": [
            int(value) for value in support.sum(dim=1).detach().cpu().tolist()
        ],
        "hard_permission_matrix_binary": permission_binary,
        "canonical_hard_access_binary": binary_access,
    }
    return packet_sources, support, detail


def _support_signature_and_details(
    encoded: Any,
    tree: Any,
    plan: MechanismPlan,
    cut: Sequence[int],
) -> tuple[tuple[bytes, ...], dict[str, Any]]:
    supports: dict[str, torch.Tensor] = {}
    details: dict[str, Any] = {}
    for mechanism in ("MM", "QE"):
        catalog = canonical_pair_catalog(encoded, tree, mechanism, case_index=0)
        packet_sources, _canonical_support, route_detail = _route_support_for_catalog(
            plan,
            mechanism,
            catalog,
            cut,
            module_present=encoded.module_present[0] if mechanism == "MM" else None,
        )
        supports[mechanism] = packet_sources
        details[mechanism] = route_detail
    joint = torch.cat((supports["MM"], supports["QE"]), dim=1)
    patterns = {
        bytes(row.to(torch.uint8).detach().cpu().tolist())
        for row in joint
        if bool(row.any())
    }
    details["joint_nonempty_permission_patterns"] = len(patterns)
    details["hard_permission_matrices_binary"] = all(
        bool(details[mechanism]["hard_permission_matrix_binary"]) for mechanism in supports
    )
    if not details["hard_permission_matrices_binary"]:
        raise ValueError("Thermal action evidence requires actual binary hard permission matrices.")
    return tuple(sorted(patterns)), details


def _full_action_snapshot(
    route: Any,
    encoded: Any,
    module_state: torch.Tensor,
    tree: Any,
) -> dict[str, Any]:
    _synchronize(encoded.module_tokens.device)
    started = time.perf_counter()
    full_budgets = {name: 1.0 for name in MECHANISM_ORDER}
    scores = route.score_cases(
        encoded,
        {
            "module_states": module_state,
            "environment_states": encoded.env_tokens,
            "global_state": encoded.global_token,
        },
        (tree,),
        budgets=full_budgets,
    )[0]
    plan = MechanismPlan.full_access(
        tree,
        encoded.module_present[0],
        int(encoded.env_coords.shape[1]),
    )
    cut = enumerate_frontier_cuts(tree, max_depth=3)[-1]
    record = stage_c._full_work_record(encoded, tree)
    _synchronize(encoded.module_tokens.device)
    return {
        "encoded": encoded,
        "module_state": module_state,
        "trees": (tree,),
        "scores": (scores,),
        "plans": (plan,),
        "records": (record,),
        "cut": cut,
        "budget_fractions": full_budgets,
        "planner_seconds": time.perf_counter() - started,
    }


def _fit_panel_features(
    snapshot: Mapping[str, Any],
    *,
    action: str,
    role_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict[str, Any]]:
    device = snapshot["encoded"].module_tokens.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    encoded = snapshot["encoded"]
    tree = snapshot["trees"][0]
    plan = snapshot["plans"][0]
    scores = snapshot["scores"][0]
    cut = tuple(int(value) for value in snapshot.get("cut", snapshot["records"][0]["frontier"]))
    feature_rows = describe_realized_plan(scores, plan, encoded, cut)
    role_features = receiver_role_descriptors(tree, role_count=role_count)
    signatures, support = _support_signature_and_details(encoded, tree, plan, cut)
    budgets = snapshot.get("budget_fractions", {})
    budget = torch.tensor(
        [float(budgets.get(key, 1.0)) for key in MECHANISM_ORDER],
        device=feature_rows.device,
        dtype=feature_rows.dtype,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    details = {
        "frontier_indices": list(cut),
        "frontier_paths": list(frontier_paths(tree, cut, max_depth=3)),
        "packet_count": len(cut),
        "nonredundant_k": len(signatures),
        "hard_support": support,
        "feature_seconds": time.perf_counter() - started,
    }
    return feature_rows, budget, role_features, len(signatures), details


def _state_snapshots(
    prediction: Any,
    capture: Mapping[str, Any] | None,
    operator: DifferentiableThermalOperator,
    route: Any,
) -> list[dict[str, Any]]:
    if capture is None:
        raise RuntimeError("Native full-access prediction omitted its input capture.")
    variant_capture = getattr(operator, "last_packet_inputs", None)
    if not isinstance(variant_capture, Mapping):
        raise TypeError("Native variant prediction omitted its input capture.")
    result = []
    for label, captured in (("baseline", capture), ("i_plus", variant_capture)):
        encoded = captured["encoded"]
        tree = captured["trees"][0]
        snapshot = _full_action_snapshot(
            route,
            encoded,
            captured["p0_module_states"],
            tree,
        )
        snapshot["label"] = label
        snapshot["runtime_diagnostics"] = dict(
            (capture.get("state_runtime_diagnostics") or {}).get(label, {})
        )
        result.append(snapshot)
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _action_predictions(
    *,
    action: str,
    stencil: Any,
    candidate_model: Any,
    candidate_route: Any,
    candidate_operator: DifferentiableThermalOperator,
    extra_route: str,
    device: torch.device,
) -> tuple[Any, list[dict[str, Any]], float]:
    _synchronize(device)
    started = time.perf_counter()
    if action == "full_access":
        with torch.no_grad():
            prediction, capture = stage_c._full_plan_predictions(
                candidate_operator, stencil, device, 512, capture_baseline=True
            )
        _synchronize(device)
        complete_seconds = time.perf_counter() - started
        snapshots = _state_snapshots(prediction, capture, candidate_operator, candidate_route)
        return prediction, snapshots, complete_seconds

    paths = THERMAL_ACTION_PATHS[action]
    def selector(_case: int, _encoded: Any, _module_state: torch.Tensor, tree: Any) -> tuple[int, ...]:
        cut, _resolved = available_frontier_for_paths(tree, paths)
        return cut

    builder = ThermalCoverPlanBuilder(
        core=candidate_model.core,
        budget_fractions={"QE": PRIMARY_CAPACITY, extra_route: PRIMARY_CAPACITY},
        mode="G",
        organizer=candidate_route,
        extra_route=extra_route,
        frontier_selector=selector,
    )
    builder.hard = True
    wrapper = _CapturePlanBuilder(builder)
    with torch.no_grad():
        prediction, capture, _ = stage_c._predict_stencil(
            candidate_operator,
            stencil,
            device=device,
            cover_plan_builder=wrapper,
            capture_baseline=True,
            capture_plan_states=True,
        )
    _synchronize(device)
    complete_seconds = time.perf_counter() - started
    if capture is None or len(wrapper.snapshots) != len(stencil.records):
        raise RuntimeError("Sparse candidate did not capture every physical state plan.")
    snapshots = []
    for label, snapshot in zip(("baseline", "i_plus"), wrapper.snapshots, strict=True):
        snapshot = dict(snapshot)
        snapshot["label"] = label
        snapshot["cut"] = tuple(int(value) for value in snapshot["records"][0]["frontier"])
        snapshot["runtime_diagnostics"] = dict(
            (wrapper.last_state_captures.get(label) or {}).get("runtime_diagnostics", {})
        )
        snapshots.append(snapshot)
    return prediction, snapshots, complete_seconds


def _thermal_quantity_metrics(
    predictions: Any,
    stencil: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Measure saved pressure and true module-peak labels alongside fields."""
    state_metrics: dict[str, Any] = {}
    labels = ("baseline", *tuple(stencil.variants))
    for label, record in zip(labels, stencil.records, strict=True):
        output = record.output
        if output is None:
            raise ValueError("Thermal quantity comparison needs saved physical labels.")
        native = reduce_native_thermal_quantities(
            predictions.values[label],
            DesignInput.from_state(record.design, device=device),
            predictions.role_queries,
            context_inputs(record.context),
            module_ids=tuple(module.module_id for module in record.design.modules),
            solid_valid_mask=output.roles["solid_temperature"].valid_mask,
        )
        pressure_reference = output.quantities["pressure_drop"]
        pressure_value = float(native.pressure_drop.detach().cpu())
        pressure_record = {
            "resolved": bool(pressure_reference.resolved),
            "units": pressure_reference.units,
            "reference": float(pressure_reference.value) if pressure_reference.resolved else None,
            "candidate": pressure_value,
            "absolute_error": abs(pressure_value - float(pressure_reference.value)) if pressure_reference.resolved else None,
        }
        full_reference_peaks = {str(key): float(value) for key, value in output.module_peak_temperature.items()}
        candidate_peaks = {
            str(key): float(value.detach().cpu()) for key, value in native.module_peak_temperature.items()
        }
        solid_role = output.roles["solid_temperature"]
        temperature_channel = solid_role.channel_names.index("temperature") if "temperature" in solid_role.channel_names else 0
        receiver_ids = tuple(str(value) for value in solid_role.receiver_module_ids or ())
        valid = np.asarray(solid_role.valid_mask, dtype=bool)
        if valid.ndim == 1:
            valid = valid[:, None]
        sampled_reference_peaks = {}
        for module_id in record.design.active_module_ids:
            selected = (np.asarray(receiver_ids) == module_id) & valid[:, temperature_channel]
            if not selected.any():
                raise ValueError(f"Sampled solid panel omits valid temperature points for {module_id!r}.")
            sampled_reference_peaks[module_id] = float(np.max(
                np.asarray(solid_role.values[:, temperature_channel], dtype=np.float64)[selected]
            ))
        common = sorted(set(sampled_reference_peaks) & set(candidate_peaks))
        if (not common or set(sampled_reference_peaks) != set(candidate_peaks)
                or set(full_reference_peaks) != set(candidate_peaks)):
            raise ValueError(f"Native and saved module-peak identities differ in {label!r}.")
        peak_units = solid_role.channel_units[temperature_channel]
        reference_sampled_global = max(sampled_reference_peaks.values())
        candidate_sampled_global = max(candidate_peaks.values())
        reference_full_global = max(full_reference_peaks.values())
        state_metrics[label] = {
            "pressure_drop": pressure_record,
            "module_peak_temperature": {
                "units": peak_units,
                "query_panel_scope": "candidate and sampled reference maxima use the same target-free 8-points-per-module panel",
                "reference_sampled_panel_global_max": reference_sampled_global,
                "candidate_sampled_panel_global_max": candidate_sampled_global,
                "sampled_panel_global_absolute_error": abs(candidate_sampled_global - reference_sampled_global),
                "reference_full_saved_module_peak_global_max": reference_full_global,
                "candidate_sampled_panel_to_full_reference_proxy_error": abs(candidate_sampled_global - reference_full_global),
                "per_module": {
                    module_id: {
                        "reference_sampled_panel": sampled_reference_peaks[module_id],
                        "candidate": candidate_peaks[module_id],
                        "sampled_panel_absolute_error": abs(candidate_peaks[module_id] - sampled_reference_peaks[module_id]),
                        "reference_full_saved_peak": full_reference_peaks[module_id],
                        "sampled_panel_to_full_reference_proxy_error": abs(candidate_peaks[module_id] - full_reference_peaks[module_id]),
                    }
                    for module_id in common
                },
            },
        }
    if len(labels) != 2:
        raise ValueError("Thermal action evidence expects baseline plus one resolved perturbation.")
    base, trial = (state_metrics[label] for label in labels)
    base_pressure = base["pressure_drop"]
    trial_pressure = trial["pressure_drop"]
    if base_pressure["resolved"] and trial_pressure["resolved"]:
        pred_delta = trial_pressure["candidate"] - base_pressure["candidate"]
        ref_delta = trial_pressure["reference"] - base_pressure["reference"]
        pressure_response = {
            "resolved": True,
            "units": trial_pressure["units"],
            "reference_delta": ref_delta,
            "candidate_delta": pred_delta,
            "absolute_error": abs(pred_delta - ref_delta),
        }
    else:
        pressure_response = {"resolved": False, "units": trial_pressure["units"], "reference_delta": None, "candidate_delta": None, "absolute_error": None}
    base_peaks = base["module_peak_temperature"]["per_module"]
    trial_peaks = trial["module_peak_temperature"]["per_module"]
    peak_changes = {
        module_id: {
            "reference_sampled_panel_delta": (
                trial_peaks[module_id]["reference_sampled_panel"]
                - base_peaks[module_id]["reference_sampled_panel"]
            ),
            "candidate_delta": trial_peaks[module_id]["candidate"] - base_peaks[module_id]["candidate"],
            "absolute_error": abs(
                (trial_peaks[module_id]["candidate"] - base_peaks[module_id]["candidate"])
                - (trial_peaks[module_id]["reference_sampled_panel"] - base_peaks[module_id]["reference_sampled_panel"])
            ),
        }
        for module_id in sorted(set(base_peaks) & set(trial_peaks))
    }
    ref_global_delta = (
        trial["module_peak_temperature"]["reference_sampled_panel_global_max"]
        - base["module_peak_temperature"]["reference_sampled_panel_global_max"]
    )
    pred_global_delta = (
        trial["module_peak_temperature"]["candidate_sampled_panel_global_max"]
        - base["module_peak_temperature"]["candidate_sampled_panel_global_max"]
    )
    ref_full_global_delta = (
        trial["module_peak_temperature"]["reference_full_saved_module_peak_global_max"]
        - base["module_peak_temperature"]["reference_full_saved_module_peak_global_max"]
    )
    return {
        "absolute_states": state_metrics,
        "finite_response": {
            "pressure_drop": pressure_response,
            "module_peak_temperature_global_max": {
                "units": trial["module_peak_temperature"]["units"],
                "query_panel_scope": "candidate and sampled reference response maxima use the same target-free 8-points-per-module panel",
                "reference_sampled_panel_delta": ref_global_delta,
                "candidate_delta": pred_global_delta,
                "absolute_error": abs(pred_global_delta - ref_global_delta),
                "reference_full_saved_peak_delta": ref_full_global_delta,
                "candidate_sampled_panel_to_full_reference_response_proxy_error": abs(pred_global_delta - ref_full_global_delta),
            },
            "module_peak_temperature_per_module": peak_changes,
        },
        "selection_threshold_scope": "pressure and peak scalars are reported in declared saved units; action-risk gates are the explicit field-role vector only",
    }


def _metric_case(
    *,
    family_id: str,
    split: str,
    panel_id: str,
    stencil: Any,
    query_panel: str,
    candidate_model: Any,
    candidate_route: Any,
    candidate_operator: DifferentiableThermalOperator,
    incumbent_operator: DifferentiableThermalOperator,
    extra_route: str,
    forward_sha256: str,
    trained_sparse_by_action: Mapping[str, bool],
    numerical_allowance_by_role: np.ndarray,
    numerical_floor_by_role: np.ndarray,
    device: torch.device,
) -> tuple[list[ActionEvidenceRow], list[dict[str, Any]], dict[str, Any]]:
    _synchronize(device)
    full_started = time.perf_counter()
    with torch.no_grad():
        full_prediction, full_capture = stage_c._full_plan_predictions(
            candidate_operator, stencil, device, 512, capture_baseline=True
        )
    _synchronize(device)
    full_complete_seconds = time.perf_counter() - full_started
    with torch.no_grad():
        _synchronize(device)
        incumbent_started = time.perf_counter()
        incumbent_prediction, _ = stage_c._full_plan_predictions(
            incumbent_operator, stencil, device, 512, capture_baseline=True
        )
        _synchronize(device)
        incumbent_complete_seconds = time.perf_counter() - incumbent_started
    if full_capture is None:
        raise RuntimeError("Same-student full pass must capture the candidate tree.")
    full_errors = np.asarray(stage_c._case_metrics(full_prediction, stencil)[0], dtype=np.float64)
    _, reference_scales, _ = stage_c._case_metrics(full_prediction, stencil)
    reference_scales = np.asarray(reference_scales, dtype=np.float64)
    incumbent_errors = np.asarray(stage_c._case_metrics(incumbent_prediction, stencil)[0], dtype=np.float64)
    role_schema = stage_c._role_rows(stencil)
    rows: list[ActionEvidenceRow] = []
    table_rows: list[dict[str, Any]] = []
    measurements: dict[str, Any] = {}
    for action in ACTION_KEYS:
        if action == "full_access":
            prediction = full_prediction
            snapshots = _state_snapshots(full_prediction, full_capture, candidate_operator, candidate_route)
            complete_seconds = full_complete_seconds
        else:
            prediction, snapshots, complete_seconds = _action_predictions(
                action=action,
                stencil=stencil,
                candidate_model=candidate_model,
                candidate_route=candidate_route,
                candidate_operator=candidate_operator,
                extra_route=extra_route,
                device=device,
            )
        candidate_errors = np.asarray(stage_c._case_metrics(prediction, stencil)[0], dtype=np.float64)
        distortion = np.asarray(stage_c._parity_metrics(prediction, full_prediction, stencil), dtype=np.float64)
        thermal_quantity_metrics = _thermal_quantity_metrics(prediction, stencil, device=device)
        descriptors = [
            _fit_panel_features(snapshot, action=action, role_count=len(role_schema))
            for snapshot in snapshots
        ]
        # Deployment receives the current baseline input. The i_plus state is
        # an outcome for finite-response labels, never a selector feature.
        packet_rows = descriptors[0][0].detach().to(device="cpu", dtype=torch.float32)
        budgets = descriptors[0][1].detach().cpu().float()
        receiver_role_features = descriptors[0][2].detach().cpu().float()
        nonredundant_k = descriptors[0][3]
        support_states = []
        route_states = []
        diagnostics_states = []
        canonical_work_states = []
        live_rows_states = []
        for snapshot, descriptor in zip(snapshots, descriptors, strict=True):
            work_record, _ = stage_c._cut_work(snapshot["records"][0])
            details = stage_c._arm_work_summary(
                snapshot["records"][0],
                snapshot.get("runtime_diagnostics", {}),
                work_record,
            )
            support_states.append({
                "state": snapshot["label"],
                "requested_action": action,
                **descriptor[4],
            })
            route_states.append(work_record)
            diagnostics_states.append(dict(snapshot.get("runtime_diagnostics", {})))
            canonical_work_states.append(float(details["canonical_work"]))
            live_rows_states.append(details["live_executor_rows"])
        mean_work = float(np.mean(canonical_work_states))
        planning_seconds_by_state = [float(snapshot.get("planner_seconds", 0.0)) for snapshot in snapshots]
        feature_seconds_by_state = [float(descriptor[4]["feature_seconds"]) for descriptor in descriptors]
        planning_seconds = float(sum(planning_seconds_by_state))
        feature_seconds = float(sum(feature_seconds_by_state))
        action_key = action
        case_key = f"{family_id}|{panel_id}"
        error_tensor = torch.as_tensor(candidate_errors, dtype=torch.float32)
        incumbent_tensor = torch.as_tensor(incumbent_errors, dtype=torch.float32)
        floor_tensor = torch.as_tensor(numerical_floor_by_role, dtype=torch.float32)
        evidence_row = ActionEvidenceRow(
            family_key=family_id,
            case_key=case_key,
            action_key=action_key,
            forward_sha256=forward_sha256,
            packet_rows=packet_rows,
            budget_vector=budgets,
            receiver_role_features=receiver_role_features,
            candidate_role_error=error_tensor,
            incumbent_role_error=incumbent_tensor,
            numerical_floor=floor_tensor,
            exact_work=mean_work,
            nonredundant_k=nonredundant_k,
            trained_sparse=bool(trained_sparse_by_action.get(action, False)) if action != "full_access" else False,
            full_access=(action == "full_access"),
        )
        rows.append(evidence_row)
        measurements[action] = {
            "candidate_role_error_against_saved_reference": candidate_errors.tolist(),
            "saved_pressure_and_peak_response_metrics": thermal_quantity_metrics,
            "same_student_full_role_error": full_errors.tolist(),
            "same_student_full_distortion": distortion.tolist(),
            "fixed_run1804_e4738_incumbent_role_error": incumbent_errors.tolist(),
            "reference_role_scale": reference_scales,
            "risk_numerical_floor_physical_units": numerical_floor_by_role.tolist(),
            "absolute_allowance_from_chunk_parity_physical_units": numerical_allowance_by_role.tolist(),
            "role_signal_unresolved_by_chunk_floor": (incumbent_errors <= numerical_allowance_by_role).tolist(),
            "role_adequate_against_fixed_incumbent": (
                candidate_errors <= (1.0 + ROLE_RELATIVE_ALLOWANCE) * incumbent_errors + numerical_allowance_by_role
            ).tolist(),
            "exact_canonical_work_by_state": canonical_work_states,
            "mean_exact_canonical_work": mean_work,
            "score_and_hard_plan_seconds_by_state": planning_seconds_by_state,
            "action_feature_seconds_by_state": feature_seconds_by_state,
            "paired_score_hard_plan_seconds": planning_seconds,
            "paired_action_feature_seconds": feature_seconds,
            "paired_selector_input_prep_seconds": planning_seconds + feature_seconds,
            "paired_baseline_i_plus_complete_call_latency_seconds": complete_seconds,
            "fixed_run1804_complete_call_latency_seconds": incumbent_complete_seconds,
            "live_executor_rows_by_state": live_rows_states,
            "runtime_diagnostics_by_state": diagnostics_states,
            "route_work_by_state": route_states,
            "hard_mm_qe_support_by_state": support_states,
            "trained_sparse_action_exposure_sufficient": evidence_row.trained_sparse,
            "full_access": evidence_row.full_access,
            "nonredundant_k": nonredundant_k,
            "M": len(stencil.baseline.design.active_modules),
            "packet_count_by_state": [item[4]["packet_count"] for item in descriptors],
            "frontier_paths_by_state": [item[4]["frontier_paths"] for item in descriptors],
            "baseline_selector_features_only": True,
        }
        table_rows.append({
            "split": split,
            "physical_family_id": family_id,
            "case_key": case_key,
            "query_panel": query_panel,
            "action": action,
            **measurements[action],
        })
    allowance_case = np.asarray(numerical_allowance_by_role, dtype=np.float64)
    floor_case = np.asarray(numerical_floor_by_role, dtype=np.float64)
    return rows, table_rows, {
        "case_key": f"{family_id}|{panel_id}",
        "family_key": family_id,
        "absolute_allowance": allowance_case.tolist(),
        "risk_numerical_floor": floor_case.tolist(),
        "reference_role_schema": role_schema,
        "full_student_role_error": full_errors.tolist(),
        "incumbent_role_error": incumbent_errors.tolist(),
    }


def _policy_summary(
    rows: Sequence[ActionEvidenceRow],
    predicted: Mapping[tuple[str, str], torch.Tensor],
    *,
    margin: torch.Tensor,
    split: str,
    model_name: str,
    allowances: Mapping[str, torch.Tensor],
    fixed_role_log_limits: torch.Tensor,
    forward_sha256: str,
) -> dict[str, Any]:
    selected_rows = [row for row in rows if row.case_key.split("|", maxsplit=1)[0] == row.family_key]
    case_keys = {row.case_key for row in selected_rows}
    split_rows = [row for row in rows if row.case_key in case_keys]
    results = evaluate_action_policy(
        split_rows,
        predicted,
        current_forward_sha256=forward_sha256,
        fixed_role_log_limits=fixed_role_log_limits,
        absolute_allowance_by_case={key: allowances[key] for key in case_keys},
        empirical_margin=margin,
        relative_allowance=ROLE_RELATIVE_ALLOWANCE,
    )
    return summarize_policy_results(results, split=split, model_name=model_name)


def summarize_policy_results(results: Sequence[Any], *, split: str, model_name: str) -> dict[str, Any]:
    if not results:
        return {"split": split, "model": model_name, "case_count": 0}
    oracle_cases = [row for row in results if row.measured_oracle_action_key is not None]
    return {
        "split": split,
        "model": model_name,
        "case_count": len(results),
        "measured_oracle_available_case_count": len(oracle_cases),
        "measured_oracle_available_rate": len(oracle_cases) / len(results),
        "selected_supported_sparse_case_count": sum(bool(row.selected_supported_sparse) for row in results),
        "supported_sparse_deployment_rate": sum(bool(row.selected_supported_sparse) for row in results) / len(results),
        "explicit_full_fallback_case_count": sum(bool(row.unsupported_at_budget) for row in results),
        "false_safe_choice_count": sum(len(row.false_safe_sparse) for row in results),
        "false_reject_choice_count": sum(len(row.false_reject_sparse) for row in results),
        "measured_adequate_sparse_choice_count": sum(len(row.measured_adequate_sparse) for row in results),
        "predicted_safe_sparse_choice_count": sum(len(row.predicted_safe_sparse) for row in results),
        "mean_selected_exact_work": float(np.mean([row.selected_work for row in results])),
        "mean_best_measured_acceptable_sparse_work": (
            float(np.mean([row.measured_oracle_sparse_work for row in oracle_cases])) if oracle_cases else None
        ),
        "mean_selected_minus_best_acceptable_work_when_available": (
            float(np.mean([
                row.selected_work - float(row.measured_oracle_sparse_work)
                for row in oracle_cases
            ])) if oracle_cases else None
        ),
        "per_case": [_jsonable(row) for row in results],
    }


def _predict_maps(
    rows: Sequence[ActionEvidenceRow],
    *,
    neural_crossfit: Any,
    ridge_crossfit: Any,
    neural_final: Any,
    ridge_final: Any,
    train_families: set[str],
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[tuple[str, str], torch.Tensor]]:
    neural_oof = dict(zip(neural_crossfit.row_keys, neural_crossfit.neural, strict=True))
    ridge_oof = dict(zip(ridge_crossfit.row_keys, ridge_crossfit.ridge, strict=True))
    neural_predictions: dict[tuple[str, str], torch.Tensor] = {}
    ridge_predictions: dict[tuple[str, str], torch.Tensor] = {}
    with torch.no_grad():
        for row in rows:
            key = (row.case_key, row.action_key)
            if row.family_key in train_families:
                if row.trained_sparse or row.full_access:
                    neural_predictions[key] = neural_oof[key]
                    ridge_predictions[key] = ridge_oof[key]
            elif row.trained_sparse or row.full_access:
                neural_predictions[key] = neural_final.model(
                    row.packet_rows, row.budget_vector, row.receiver_role_features,
                    nonredundant_k=row.nonredundant_k,
                ).detach().cpu()
                ridge_predictions[key] = ridge_final.predict(row).detach().cpu()
    return neural_predictions, ridge_predictions


def _fit_selector_models(
    train_rows: Sequence[ActionEvidenceRow],
    measured_rows: Sequence[ActionEvidenceRow],
    *,
    train_families: Sequence[str],
    current_forward_sha256: str,
    seed: int,
) -> dict[str, Any] | None:
    """Fit only when at least one sparse action passed measured exposure gates."""
    if not any(row.trained_sparse for row in train_rows):
        return None
    crossfit = crossfit_action_risk(
        train_rows,
        current_forward_sha256=current_forward_sha256,
        train_families=train_families,
        folds=min(RISK_CROSSFIT_FOLDS, len(train_families)),
        updates=RISK_FIT_UPDATES,
        seed=seed,
    )
    neural_final = fit_action_risk_head(
        measured_rows,
        current_forward_sha256=current_forward_sha256,
        train_families=train_families,
        updates=RISK_FIT_UPDATES,
        seed=seed + 1997,
    )
    ridge_final = fit_ridge_action_baseline(
        measured_rows,
        current_forward_sha256=current_forward_sha256,
        train_families=train_families,
    )
    neural_predictions, ridge_predictions = _predict_maps(
        measured_rows,
        neural_crossfit=crossfit,
        ridge_crossfit=crossfit,
        neural_final=neural_final,
        ridge_final=ridge_final,
        train_families=set(train_families),
    )
    return {
        "crossfit": crossfit,
        "neural_final": neural_final,
        "ridge_final": ridge_final,
        "neural_predictions": neural_predictions,
        "ridge_predictions": ridge_predictions,
    }


def _time_neural_risk_selection(
    rows: Sequence[ActionEvidenceRow],
    *,
    neural_fit: Any,
    fixed_role_log_limits: torch.Tensor,
    empirical_margin: torch.Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Time input-only neural risk prediction and exact-work action selection."""
    grouped: dict[str, list[ActionEvidenceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.case_key].append(row)
    model = neural_fit.model.to(device=device)
    model.eval()
    results = []
    for case_key, case_rows in sorted(grouped.items()):
        by_action = {row.action_key: row for row in case_rows}
        if set(by_action) != set(ACTION_KEYS):
            raise ValueError(f"Risk selection case {case_key!r} lacks an exact four-action table.")
        predictions = []
        _synchronize(device)
        head_started = time.perf_counter()
        with torch.no_grad():
            for action in ACTION_KEYS:
                row = by_action[action]
                if row.trained_sparse:
                    predictions.append(model(
                        row.packet_rows.to(device),
                        row.budget_vector.to(device),
                        row.receiver_role_features.to(device),
                        nonredundant_k=row.nonredundant_k,
                    ))
                else:
                    predictions.append(torch.zeros_like(fixed_role_log_limits, device=device))
        _synchronize(device)
        risk_seconds = time.perf_counter() - head_started
        prediction_matrix = torch.stack(predictions)
        _synchronize(device)
        selection_started = time.perf_counter()
        selection = select_action_by_risk(
            prediction_matrix,
            torch.tensor([by_action[action].exact_work for action in ACTION_KEYS], device=device, dtype=torch.float64),
            torch.tensor([by_action[action].nonredundant_k for action in ACTION_KEYS], device=device),
            torch.tensor([by_action[action].trained_sparse for action in ACTION_KEYS], device=device),
            fixed_role_log_limits.to(device=device, dtype=prediction_matrix.dtype),
            full_access_index=ACTION_KEYS.index("full_access"),
            empirical_margin=empirical_margin.to(device=device, dtype=prediction_matrix.dtype),
        )
        _synchronize(device)
        selection_seconds = time.perf_counter() - selection_started
        results.append({
            "case_key": case_key,
            "risk_head_seconds": risk_seconds,
            "selection_seconds": selection_seconds,
            "risk_head_plus_selection_seconds": risk_seconds + selection_seconds,
            "selected_action": ACTION_KEYS[selection.index],
            "explicit_full_fallback": bool(selection.unsupported_at_budget),
        })
    return results


def _rank_agreement(
    rows: Sequence[ActionEvidenceRow],
    predicted: Mapping[tuple[str, str], torch.Tensor],
    *,
    split_families: set[str],
) -> dict[str, Any]:
    grouped: dict[str, list[ActionEvidenceRow]] = defaultdict(list)
    for row in rows:
        if row.family_key in split_families and row.trained_sparse:
            grouped[row.case_key].append(row)
    concordant = discordant = tied = 0
    case_rows = 0
    for case_key, case_actions in grouped.items():
        comparable = 0
        for left_index, left in enumerate(case_actions):
            for right in case_actions[left_index + 1:]:
                actual = float((left.target_log_risk.max() - right.target_log_risk.max()).item())
                if abs(actual) <= 1.0e-3:
                    continue
                predicted_delta = float((predicted[(case_key, left.action_key)].max()
                                         - predicted[(case_key, right.action_key)].max()).item())
                comparable += 1
                if abs(predicted_delta) <= 1.0e-3:
                    tied += 1
                elif actual * predicted_delta > 0:
                    concordant += 1
                else:
                    discordant += 1
        case_rows += int(comparable > 0)
    total = concordant + discordant + tied
    return {
        "families": sorted(split_families),
        "cases_with_comparable_action_pairs": case_rows,
        "comparable_action_pairs": total,
        "concordant": concordant,
        "discordant": discordant,
        "tied": tied,
        "pairwise_ranking_accuracy_excluding_ties": concordant / (concordant + discordant) if concordant + discordant else None,
    }


def _fixed_action_summary(
    rows: Sequence[ActionEvidenceRow],
    *,
    split: str,
    family_ids: set[str],
    allowances: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    result = []
    for action in SPARSE_ACTION_KEYS:
        selected = [row for row in rows if row.family_key in family_ids and row.action_key == action]
        eligible = [row for row in selected if row.trained_sparse]
        adequate = [
            row for row in eligible
            if bool((row.candidate_role_error.cpu().double()
                     <= (1.0 + ROLE_RELATIVE_ALLOWANCE) * row.incumbent_role_error.cpu().double()
                     + allowances[row.case_key].cpu().double()).all())
        ]
        result.append({
            "split": split,
            "fixed_action": action,
            "case_count": len(selected),
            "action_exposure_eligible_case_count": len(eligible),
            "measured_acceptable_case_count": len(adequate),
            "measured_acceptable_rate_among_eligible": len(adequate) / len(eligible) if eligible else None,
            "mean_exact_work": float(np.mean([row.exact_work for row in eligible])) if eligible else None,
            "mean_nonredundant_k": float(np.mean([row.nonredundant_k for row in eligible])) if eligible else None,
        })
    return result


def _k_summary(rows: Sequence[ActionEvidenceRow], split_families: Mapping[str, set[str]], module_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    result = []
    for split, families in split_families.items():
        for action in ACTION_KEYS:
            selected = [row for row in rows if row.family_key in families and row.action_key == action]
            by_k = Counter(int(row.nonredundant_k) for row in selected)
            result.append({
                "split": split,
                "action": action,
                "M_by_family": {row.family_key: int(module_counts[row.family_key]) for row in selected},
                "nonredundant_k_counts": dict(sorted(by_k.items())),
                "mean_nonredundant_k": float(np.mean([row.nonredundant_k for row in selected])) if selected else None,
            })
    return result


def _load_source_and_template(source_path: Path, device: torch.device):
    model, payload = stage_c.load_model(source_path, device)
    if int(payload.get("epoch", payload.get("current_epoch", -1))) != 4738:
        raise ValueError("The supplied Run1804 checkpoint is not epoch 4738.")
    dataset_path = stage_c._resolve_dataset_path(payload, None)
    dataset = stage_c.GlobalChannelThermalDataset(
        dataset_path,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = stage_c._make_input_template(dataset)
    return model, payload, dataset_path, template


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = stage_c.time.monotonic()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing action-evaluation evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.controlled_manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    source_path = args.source_checkpoint.expanduser().resolve()
    manifest, endpoint_payload, endpoint_sha = _load_checkpoint_binding(manifest_path, checkpoint_path)
    source_hash = _sha256(source_path)
    if source_hash != manifest.get("reference_checkpoint_sha256"):
        raise ValueError("Fixed Run1804 checkpoint SHA256 differs from the controlled-run provenance.")
    forward_update = int(endpoint_payload["actual_optimizer_updates"])
    arm = "G"
    extra_route = str(manifest.get("extra_route", "")).upper()
    if extra_route not in {"MM", "ME"}:
        raise ValueError("Controlled manifest must bind the exact selected MM/ME route.")

    stencils_by_split, split_manifest = _load_action_stencils(args)
    train_families = tuple(item.physical_family_id for item in stencils_by_split["train"])
    metric_path = Path(str(manifest["arms"][arm]["training_steps"])).expanduser().resolve()
    if not metric_path.is_file():
        raise FileNotFoundError(f"Manifest-bound G training log is missing: {metric_path}")
    route_work_path = Path(str(manifest["arms"][arm]["route_work_ledger"])).expanduser().resolve()
    if not route_work_path.is_file():
        raise FileNotFoundError(f"Manifest-bound G route-work log is missing: {route_work_path}")
    exposure = summarize_training_exposure(
        metric_path,
        checkpoint_update=forward_update,
        family_ids=train_families,
        route_work_path=route_work_path,
    )
    exposure["training_steps_path"] = str(metric_path)
    exposure["training_steps_sha256"] = _sha256(metric_path)
    exposure["route_work_sha256"] = _sha256(route_work_path)
    exposure["maturation_gates"] = complete_maturity_gates(
        exposure,
        historical_case_count=int(manifest.get("historical_case_count", 0)),
    )
    trained_sparse = {
        action: bool(exposure["per_action"][action]["trained_sparse"])
        for action in SPARSE_ACTION_KEYS
    }
    numerical_floor_by_role, train_median_reference_scales = _train_numerical_floor_by_role(
        stencils_by_split["train"]
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device} is unavailable.")
    _incumbent_model, source_payload, dataset_path, template = _load_source_and_template(source_path, device)
    del _incumbent_model
    # Build candidate and reference operators from separate exact source loads.
    candidate_model, _ = stage_c.load_model(source_path, device)
    stage_c._configure_native_expanded_response_interface_scope(candidate_model)
    probe_operator = _new_operator(candidate_model, source_payload, template, capture=True)
    anchor_stencil = stencils_by_split["train"][0]
    with torch.no_grad():
        _, anchor_capture = stage_c._full_plan_predictions(
            probe_operator, anchor_stencil, device, 512, capture_baseline=True
        )
    if anchor_capture is None:
        raise RuntimeError("Source checkpoint input probe omitted the native tree capture.")
    candidate_route = forward._route_module("G", candidate_model.core, anchor_capture["encoded"], device)
    physical_state, route_state = _state_parts(endpoint_payload)
    missing, unexpected = candidate_model.load_state_dict(physical_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Physical endpoint state mismatch: {missing}, {unexpected}")
    missing, unexpected = candidate_route.load_state_dict(route_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Organizer endpoint state mismatch: {missing}, {unexpected}")
    candidate_model.eval().requires_grad_(False)
    candidate_route.eval().requires_grad_(False)
    candidate_operator = _new_operator(candidate_model, source_payload, template, capture=True)
    incumbent_model, _ = stage_c.load_model(source_path, device)
    stage_c._configure_native_expanded_response_interface_scope(incumbent_model)
    incumbent_model.eval().requires_grad_(False)
    incumbent_operator = _new_operator(incumbent_model, source_payload, template, capture=True)

    parity_stencils = (stencils_by_split["train"][0], stencils_by_split["train"][-1])
    parity_rows = []
    with torch.no_grad():
        for stencil in parity_stencils:
            q512, _ = stage_c._full_plan_predictions(candidate_operator, stencil, device, 512)
            q2048, _ = stage_c._full_plan_predictions(candidate_operator, stencil, device, 2048)
            parity_rows.append(np.asarray(stage_c._parity_metrics(q512, q2048, stencil), dtype=np.float64))
    numerical_allowance_by_role = np.max(np.stack(parity_rows), axis=0)
    if numerical_allowance_by_role.shape != (len(stage_c._role_rows(anchor_stencil)),):
        raise RuntimeError("Chunk-parity role schema changed across train families.")

    candidate_before = stage_c._state_hash(candidate_model.state_dict())
    route_before = stage_c._state_hash(candidate_route.state_dict())
    source_before = stage_c._state_hash(incumbent_model.state_dict())
    evidence_rows: list[ActionEvidenceRow] = []
    table: list[dict[str, Any]] = []
    case_metadata: dict[str, dict[str, Any]] = {}
    module_counts: dict[str, int] = {}
    all_stencils: list[tuple[str, str, str, Any]] = []
    for split in ("train", "dev", "held_family_audit"):
        for index, stencil in enumerate(stencils_by_split[split]):
            all_stencils.append((split, stencil.physical_family_id, f"primary_{index}", stencil))
    for index, stencil in enumerate(stencils_by_split["train_query_repeat"]):
        all_stencils.append(("train", stencil.physical_family_id, f"query_repeat_{index}", stencil))
    for split, family_id, panel_id, stencil in all_stencils:
        module_counts[family_id] = len(stencil.baseline.design.active_modules)
        local_rows, local_table, case_record = _metric_case(
            family_id=family_id,
            split=split,
            panel_id=panel_id,
            stencil=stencil,
            query_panel=panel_id,
            candidate_model=candidate_model,
            candidate_route=candidate_route,
            candidate_operator=candidate_operator,
            incumbent_operator=incumbent_operator,
            extra_route=extra_route,
            forward_sha256=endpoint_sha,
            trained_sparse_by_action=trained_sparse,
            numerical_allowance_by_role=numerical_allowance_by_role,
            numerical_floor_by_role=numerical_floor_by_role,
            device=device,
        )
        evidence_rows.extend(local_rows)
        table.extend(local_table)
        case_metadata[case_record["case_key"]] = {
            **case_record,
            "split": split,
            "panel_id": panel_id,
        }

    if stage_c._state_hash(candidate_model.state_dict()) != candidate_before:
        raise RuntimeError("Fixed endpoint physical weights changed during evaluation.")
    if stage_c._state_hash(candidate_route.state_dict()) != route_before:
        raise RuntimeError("Fixed endpoint organizer weights changed during evaluation.")
    if stage_c._state_hash(incumbent_model.state_dict()) != source_before:
        raise RuntimeError("Fixed Run1804 incumbent weights changed during evaluation.")

    train_family_set = set(train_families)
    allowances = {
        case_key: torch.as_tensor(record["absolute_allowance"], dtype=torch.float64)
        for case_key, record in case_metadata.items()
    }
    fixed_role_log_limits, fixed_train_gate_rows = _fixed_train_role_log_limits(
        evidence_rows,
        train_families=train_family_set,
        allowances=allowances,
    )
    train_case_rows = [row for row in evidence_rows if row.family_key in train_family_set]
    fit_bundle = _fit_selector_models(
        train_case_rows,
        evidence_rows,
        train_families=train_families,
        current_forward_sha256=endpoint_sha,
        seed=args.risk_seed,
    )
    selector_fitted = fit_bundle is not None
    crossfit = fit_bundle["crossfit"] if fit_bundle is not None else None
    neural_final = fit_bundle["neural_final"] if fit_bundle is not None else None
    neural_predictions = fit_bundle["neural_predictions"] if fit_bundle is not None else {}
    ridge_predictions = fit_bundle["ridge_predictions"] if fit_bundle is not None else {}
    timed_risk_selection = (
        _time_neural_risk_selection(
            evidence_rows,
            neural_fit=neural_final,
            fixed_role_log_limits=fixed_role_log_limits,
            empirical_margin=crossfit.neural_upper_margin,
            device=device,
        )
        if selector_fitted else []
    )
    split_sets = {
        "train": set(train_families),
        "dev": set(split_manifest["split_families"]["dev"]),
        "held_family_audit": set(split_manifest["split_families"]["held_family_audit"]),
    }
    policy_summaries = []
    ranking_summaries = []
    fixed_summaries = []
    for split, families in split_sets.items():
        split_rows = [row for row in evidence_rows if row.family_key in families]
        case_keys = {row.case_key for row in split_rows}
        split_allowances = {key: allowances[key] for key in case_keys}
        if selector_fitted:
            for model_name, prediction_map, margin in (
                ("neural_predeclared", neural_predictions, crossfit.neural_upper_margin),
                ("ridge_comparison", ridge_predictions, crossfit.ridge_upper_margin),
            ):
                results = evaluate_action_policy(
                    split_rows,
                    prediction_map,
                    current_forward_sha256=endpoint_sha,
                    fixed_role_log_limits=fixed_role_log_limits,
                    absolute_allowance_by_case=split_allowances,
                    empirical_margin=margin,
                    relative_allowance=ROLE_RELATIVE_ALLOWANCE,
                )
                policy_summaries.append(summarize_policy_results(results, split=split, model_name=model_name))
                ranking_summaries.append({
                    "split": split,
                    "model": model_name,
                    **_rank_agreement(split_rows, prediction_map, split_families=families),
                })
        fixed_summaries.extend(_fixed_action_summary(
            split_rows, split=split, family_ids=families, allowances=split_allowances,
        ))

    k_summary = _k_summary(evidence_rows, split_sets, module_counts)
    output_rows = []
    for row in table:
        case_key = str(row["case_key"])
        action = str(row["action"])
        feature_row = next(item for item in evidence_rows if item.case_key == case_key and item.action_key == action)
        row["signed_log_role_risk_to_fixed_incumbent"] = feature_row.target_log_risk.tolist()
        row["fixed_train_role_log_limits"] = fixed_role_log_limits.tolist()
        row["measured_acceptable_sparse_oracle_member"] = bool(
            feature_row.trained_sparse and bool((
                feature_row.candidate_role_error.cpu().double()
                <= (1.0 + ROLE_RELATIVE_ALLOWANCE) * feature_row.incumbent_role_error.cpu().double()
                + allowances[case_key]
            ).all())
        )
        for model_name, prediction_map in (("neural", neural_predictions), ("ridge", ridge_predictions)):
            prediction = prediction_map.get((case_key, action))
            row[f"{model_name}_predicted_log_risk"] = None if prediction is None else prediction.tolist()
        output_rows.append(row)

    identity = {
        "driver": str(Path(__file__).resolve()),
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "command_argv": list(sys.argv),
        "controlled_manifest": str(manifest_path),
        "controlled_manifest_sha256": _sha256(manifest_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": endpoint_sha,
        "checkpoint_update": forward_update,
        "checkpoint_arm": arm,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_hash,
        "fixed_incumbent": "Run1804 epoch 4738 dense field checkpoint",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "atlas_dir": str(args.atlas_dir.expanduser().resolve()),
        "atlas_file_sha256": split_manifest["atlas_file_sha256"],
        "coverage_manifest": str(args.coverage_manifest.expanduser().resolve()),
        "coverage_manifest_sha256": _sha256(args.coverage_manifest.expanduser().resolve()),
        "predeclared_manifest": str(args.predeclared_manifest.expanduser().resolve()),
        "predeclared_manifest_sha256": _sha256(args.predeclared_manifest.expanduser().resolve()),
        "device": str(device),
        "extra_route": extra_route,
        "split_manifest": split_manifest,
        "split_manifest_sha256": hashlib.sha256(json.dumps(split_manifest, sort_keys=True).encode()).hexdigest(),
        "input_feature_contract": [
            "current pre-interaction packet/source features", "typed MM/QE hard permissions",
            "known route capacities", "input-only receiver role geometry", "nonredundant K",
        ],
        "excluded_from_features": [
            "reference/candidate/incumbent errors", "saved targets", "physical family ID",
            "partition label", "same-student full distortion",
        ],
        "reference_solver_attempts": 0,
        "physical_weights_unchanged": True,
        "organizer_weights_unchanged": True,
        "incumbent_weights_unchanged": True,
        "query_panel_seed": args.panel_seed,
        "fit_seed": args.risk_seed,
        "margin_source": "train-family grouped crossfit residuals only" if selector_fitted else None,
        "policy_model_predeclared": "neural; ridge remains a same-split comparison and is not selected using dev/held results",
        "dev_scope": "model-development only; not held-out generalization",
        "held_family_audit_scope": "previously inspected after rank freeze; development audit, not untouched generalization",
        "shared_geometry_across_partitions": split_manifest.get("shared_geometry_across_partitions", {}),
        "shared_geometry_scope": "different Reynolds contexts sharing case geometry are correlated development evidence, not independent generalization",
        "numerical_allowance_role_units": numerical_allowance_by_role.tolist(),
        "numerical_allowance_method": "maximum q512-versus-q2048 full-student parity over first and last train families",
        "risk_numerical_floor_role_units": numerical_floor_by_role.tolist(),
        "risk_numerical_floor_method": "per-role max(1e-8 physical units, 1e-6 times median saved train reference RMS scale)",
        "train_median_reference_role_scales": train_median_reference_scales.tolist(),
        "fixed_train_role_log_limits": fixed_role_log_limits.tolist(),
        "fixed_train_role_log_limit_method": "role-wise minimum Run1804 physical gate limit over the eight primary train families, mapped with the separate numerical floor",
        "fixed_train_role_log_limit_family_rows": fixed_train_gate_rows,
        "fixed_limit_interpretation": "conservative train-only operational approximation for selection; measured case-specific incumbent gates are post-hoc oracle labels only",
        "role_relative_allowance": ROLE_RELATIVE_ALLOWANCE,
        "elapsed_seconds": stage_c.time.monotonic() - started,
    }
    summary = {
        "status": (
            "completed_train_only_action_evaluation" if selector_fitted
            else "selector_not_fitted_no_qualified_actions"
        ),
        "physical_action_table_status": "completed",
        "selector_status": (
            "fitted_on_exposure_qualified_sparse_actions" if selector_fitted
            else "selector_not_fitted_no_qualified_actions"
        ),
        "exposure_qualified_sparse_actions": [
            action for action in SPARSE_ACTION_KEYS if trained_sparse[action]
        ],
        "sparse_actions_excluded_from_selector_fit": [
            action for action in SPARSE_ACTION_KEYS if not trained_sparse[action]
        ],
        "identity": identity,
        "exposure": exposure,
        "maturity": exposure["maturation_gates"],
        "neural_fit": None if not selector_fitted else {
            "updates": neural_final.updates,
            "first_loss": neural_final.first_loss,
            "last_loss": neural_final.last_loss,
            "crossfit_neural_upper_margin": crossfit.neural_upper_margin.tolist(),
            "crossfit_ridge_upper_margin": crossfit.ridge_upper_margin.tolist(),
            "folds": min(RISK_CROSSFIT_FOLDS, len(train_families)),
            "families": list(train_families),
        },
        "policy_results": policy_summaries,
        "fixed_train_role_log_limits": fixed_role_log_limits.tolist(),
        "timing": {
            "score_hard_plan_and_feature_prep_by_action": {
                action: {
                    "case_count": sum(1 for row in output_rows if row["action"] == action),
                    "mean_seconds": float(np.mean([
                        row["paired_selector_input_prep_seconds"]
                        for row in output_rows if row["action"] == action
                    ])),
                    "median_seconds": float(np.median([
                        row["paired_selector_input_prep_seconds"]
                        for row in output_rows if row["action"] == action
                    ])),
                }
                for action in ACTION_KEYS
            },
            "neural_risk_head_and_selection": {
                "status": "completed" if selector_fitted else "not_fitted_no_qualified_sparse_actions",
                "case_count": len(timed_risk_selection),
                "mean_risk_head_seconds": float(np.mean([row["risk_head_seconds"] for row in timed_risk_selection])) if timed_risk_selection else None,
                "median_risk_head_seconds": float(np.median([row["risk_head_seconds"] for row in timed_risk_selection])) if timed_risk_selection else None,
                "mean_selection_seconds": float(np.mean([row["selection_seconds"] for row in timed_risk_selection])) if timed_risk_selection else None,
                "median_selection_seconds": float(np.median([row["selection_seconds"] for row in timed_risk_selection])) if timed_risk_selection else None,
                "mean_risk_head_plus_selection_seconds": float(np.mean([row["risk_head_plus_selection_seconds"] for row in timed_risk_selection])) if timed_risk_selection else None,
                "per_case": timed_risk_selection,
            },
            "complete_native_call_by_action": {
                action: {
                    "mean_seconds": float(np.mean([
                        row["paired_baseline_i_plus_complete_call_latency_seconds"]
                        for row in output_rows if row["action"] == action
                    ])),
                    "median_seconds": float(np.median([
                        row["paired_baseline_i_plus_complete_call_latency_seconds"]
                        for row in output_rows if row["action"] == action
                    ])),
                }
                for action in ACTION_KEYS
            },
            "fixed_run1804_complete_call": {
                "mean_seconds": float(np.mean([row["fixed_run1804_complete_call_latency_seconds"] for row in output_rows if row["action"] == "full_access"])),
                "median_seconds": float(np.median([row["fixed_run1804_complete_call_latency_seconds"] for row in output_rows if row["action"] == "full_access"])),
            },
            "synchronized": True,
            "query_batch_size": 512,
            "separate_selected_weight_q64_q1024_q8192_large_g_p_panel": "requires root-selected endpoint and approved GPU window; not part of this CPU-safe evaluator pass",
        },
        "ranking_results": ranking_summaries,
        "fixed_action_results": fixed_summaries,
        "nonredundant_k_by_split_action_and_m": k_summary,
        "label_roles": [item["name"] for item in stage_c._role_rows(stencils_by_split["train"][0])],
        "action_rows": len(output_rows),
        "case_panels": len(case_metadata),
        "reference_solver_attempts": 0,
        "limitations": [
            "The two dev and two held-family audit contexts are development evidence, not untouched generalization.",
            "Run1804 epoch 4738 is the fixed dense native field incumbent; comparison to its saved outputs is not a global superiority claim.",
            "Canonical pair work and live executor rows are reported separately; support counts are not speedup.",
            "Cross-fit residual margins are empirical for this small selected family set, not formal coverage guarantees.",
        ],
    }
    _write_jsonl(output / "action_table.jsonl", output_rows)
    _atomic_json(output / "summary.json", summary)
    _atomic_json(output / "provenance.json", identity)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-manifest", type=Path, default=DEFAULT_CONTROLLED_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--coverage-manifest", type=Path, default=DEFAULT_COVERAGE_MANIFEST)
    parser.add_argument("--predeclared-manifest", type=Path, default=DEFAULT_PREDECLARED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu", help="Use CPU by default; choose an approved device explicitly.")
    parser.add_argument("--panel-seed", type=int, default=23819)
    parser.add_argument("--risk-seed", type=int, default=31019)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
