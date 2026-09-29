"""Measure bounded selected-weight Thermal organizer and live-work controls.

The evaluator loads exact G/P endpoints from a controlled-maturation manifest,
uses saved train/dev response labels, and never invokes a reference solver.
It reuses Stage-C plan/work/prediction helpers while keeping the selected
checkpoint control pass separate from training and selector fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

CASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CASE_ROOT / "scripts"))

import evaluate_matured_action_selector as action_eval
import run_active_packet_forward as forward
import run_active_packet_stage_c as stage_c
from channelthermal.response_control.maturation import (
    THERMAL_ACTION_PATHS,
    available_frontier_for_paths,
)
from channelthermal.response_control.native import DifferentiableThermalOperator

from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan
from honf_forward_core.interface_fields.budgeted_frontier import canonical_pair_catalog

DEFAULT_CONTROLLED_RUN = PROJECT_ROOT / "diagnostics/generated/thermal_maturation_20260929/controlled_run"
DEFAULT_MANIFEST = DEFAULT_CONTROLLED_RUN / "run_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "diagnostics/generated/thermal_maturation_20260929/selected_weight_controls"
DEFAULT_COVERAGE_MANIFEST = stage_c.DEFAULT_COVERAGE_ROOT / "thermal_coverage_calls_v3.json"
DEFAULT_PREDECLARED = stage_c.DEFAULT_PREDECLARED
DEFAULT_ATLAS = stage_c.DEFAULT_ATLAS
MAX_TRAIN_FEATURE_FAMILIES = 8
LIVE_MATCH_ROUNDS = 4
LIVE_MATCH_FLOOR = 1.0e-3
ROOT_UNION_MECHANISMS = ("MM", "QE")
SPARSE_G_CONTROL_CALLS_PER_CASE = 12
MECHANISM_ORDER = ("MM", "ME", "EM", "QM", "QE")
LIVE_ROW_PREFIX = {
    "MM": "cover_prepare_mm_",
    "ME": "cover_prepare_me_",
    "EM": "cover_prepare_em_",
    "QM": "cover_qm_",
    "QE": "cover_qe_",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _physical_gpu2_identity() -> tuple[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "2",
            "--query-gpu=uuid,name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [item.strip() for item in result.stdout.strip().split(",", maxsplit=1)]
    if len(fields) != 2 or not all(fields):
        raise RuntimeError(f"Could not parse physical GPU2 identity from nvidia-smi: {result.stdout!r}")
    return fields[0], fields[1]


def _validate_cuda_gpu_binding(
    manifest: Mapping[str, Any],
    device: torch.device,
    *,
    cuda_visible_devices: str | None,
    queried_identity: tuple[str, str] | None = None,
    lane_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if device.type == "cpu":
        return {
            "status": "not_applicable_cpu",
            "physical_gpu_index": None,
            "physical_gpu_uuid": None,
            "physical_gpu_name": None,
            "cuda_visible_devices": cuda_visible_devices,
        }
    if device.type != "cuda" or device.index not in {None, 0}:
        raise ValueError("Selected-weight CUDA evaluation must use logical cuda:0 bound to physical GPU2.")
    if cuda_visible_devices != "2":
        raise ValueError("Selected-weight CUDA evaluation must set CUDA_VISIBLE_DEVICES=2.")
    manifest_index = manifest.get("physical_gpu_index", manifest.get("physical_gpu"))
    expected_uuid = str(manifest.get("physical_gpu_uuid", ""))
    if manifest_index != 2 or not expected_uuid or manifest.get("cuda_visible_devices") != "2":
        raise ValueError("Controlled-run manifest lacks the physical GPU2 index, UUID, or visibility binding.")
    identity = queried_identity or _physical_gpu2_identity()
    if identity[0] != expected_uuid:
        raise ValueError(
            f"Queried physical GPU2 UUID {identity[0]} differs from controlled manifest {expected_uuid}."
        )
    expected_name = str(manifest.get("physical_gpu_name", ""))
    if expected_name and identity[1] != expected_name:
        raise ValueError(
            f"Queried physical GPU2 name {identity[1]} differs from controlled manifest {expected_name}."
        )
    arm_starts: set[str] = set()
    for row in lane_rows:
        if row.get("event") not in {"arm_start", "arm_stop"}:
            continue
        arm = str(row.get("arm", ""))
        if arm not in {"G", "P"}:
            continue
        if row.get("physical_gpu_index") != 2 or row.get("physical_gpu_uuid") != expected_uuid:
            raise ValueError(f"Controlled {arm} lane execution ledger has a mismatched GPU binding.")
        if row.get("cuda_visible_devices") != "2":
            raise ValueError(f"Controlled {arm} lane execution ledger did not use CUDA_VISIBLE_DEVICES=2.")
        if row.get("event") == "arm_start":
            arm_starts.add(arm)
    if arm_starts != {"G", "P"}:
        raise ValueError("The controlled lane execution ledger does not bind both G and P to physical GPU2.")
    return {
        "status": "verified_physical_gpu2",
        "physical_gpu_index": 2,
        "physical_gpu_uuid": identity[0],
        "physical_gpu_name": identity[1],
        "logical_device": "cuda:0",
        "cuda_visible_devices": cuda_visible_devices,
        "manifest_physical_gpu_uuid": expected_uuid,
        "lane_rows_verified": len(lane_rows),
    }


def _load_lane_execution_rows(manifest_path: Path) -> tuple[dict[str, Any], ...]:
    path = manifest_path.parent / "lane_execution_ledger.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Controlled lane execution ledger is required for CUDA binding: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed controlled lane ledger row {line_number}: {exc}") from exc
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return tuple(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


@dataclass(frozen=True)
class TrainPopulationFeatures:
    """Equal-family means of baseline, pre-interaction Thermal input features."""

    module_token: torch.Tensor
    environment_token: torch.Tensor
    global_token: torch.Tensor
    module_state: torch.Tensor
    environment_state: torch.Tensor
    module_feature: torch.Tensor
    environment_feature: torch.Tensor | None
    family_ids: tuple[str, ...]


def _per_family_feature_means(capture: Mapping[str, Any]) -> dict[str, torch.Tensor | None]:
    encoded = capture["encoded"]
    module_states = capture["p0_module_states"][0]
    present = encoded.module_present[0] > 0.5
    if not bool(present.any()):
        raise ValueError("A training population feature source has no active Thermal modules.")
    weights = encoded.env_weights[0].to(dtype=encoded.env_tokens.dtype)
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    result: dict[str, torch.Tensor | None] = {
        "module_token": encoded.module_tokens[0][present].mean(dim=0),
        "environment_token": (encoded.env_tokens[0] * weights[:, None]).sum(dim=0),
        "global_token": encoded.global_token[0],
        "module_state": module_states[present].mean(dim=0),
        "environment_state": (encoded.env_tokens[0] * weights[:, None]).sum(dim=0),
        "module_feature": encoded.module_features[0][present].mean(dim=0),
        "environment_feature": (
            None
            if encoded.env_features is None
            else (encoded.env_features[0] * weights.to(encoded.env_features.dtype)[:, None]).sum(dim=0)
        ),
    }
    return result


def _train_population_features(
    captures_by_family: Mapping[str, Mapping[str, Any]],
) -> TrainPopulationFeatures:
    """Average within-family feature means so large families do not dominate."""

    if len(captures_by_family) != MAX_TRAIN_FEATURE_FAMILIES:
        raise ValueError(f"Expected all {MAX_TRAIN_FEATURE_FAMILIES} original train atlas families.")
    family_ids = tuple(sorted(str(value) for value in captures_by_family))
    by_family = [_per_family_feature_means(captures_by_family[family]) for family in family_ids]

    def equal_family_mean(name: str) -> torch.Tensor:
        values = [row[name] for row in by_family]
        if any(value is None for value in values):
            raise ValueError(f"Required fixed train-population feature {name!r} is absent.")
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if len(tensors) != len(values) or any(value.shape != tensors[0].shape for value in tensors):
            raise ValueError(f"Train-family feature shapes differ for {name!r}.")
        return torch.stack([value.to(device="cpu", dtype=torch.float64) for value in tensors]).mean(dim=0)

    environment_features = [row["environment_feature"] for row in by_family]
    if all(value is None for value in environment_features):
        fixed_environment_feature = None
    elif any(value is None for value in environment_features):
        raise ValueError("Environment feature availability changed across the training population.")
    else:
        values = [value for value in environment_features if isinstance(value, torch.Tensor)]
        fixed_environment_feature = torch.stack(
            [value.to(device="cpu", dtype=torch.float64) for value in values]
        ).mean(dim=0)
    return TrainPopulationFeatures(
        module_token=equal_family_mean("module_token"),
        environment_token=equal_family_mean("environment_token"),
        global_token=equal_family_mean("global_token"),
        module_state=equal_family_mean("module_state"),
        environment_state=equal_family_mean("environment_state"),
        module_feature=equal_family_mean("module_feature"),
        environment_feature=fixed_environment_feature,
        family_ids=family_ids,
    )


def _broadcast_vector(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim < 2 or value.shape != (reference.shape[-1],):
        raise ValueError("Population feature width must match the current organizer input.")
    view_shape = (1,) * (reference.ndim - 1) + (value.numel(),)
    return value.to(device=reference.device, dtype=reference.dtype).reshape(view_shape).expand_as(reference)


class FixedTrainPopulationOrganizer:
    """Score with fixed train-population features while keeping case geometry."""

    def __init__(self, base: Any, population: TrainPopulationFeatures) -> None:
        self.base = base
        self.population = population

    def score_cases(
        self,
        encoded: Any,
        prepared_state: Mapping[str, torch.Tensor],
        trees: Sequence[Any],
        *,
        budgets: Mapping[str, float | torch.Tensor] | torch.Tensor | None = None,
    ):
        current_module_features = encoded.module_features
        current_environment_features = encoded.env_features
        if current_module_features is None:
            raise ValueError("Thermal organizer requires module input features.")
        fixed_module_features = _broadcast_vector(self.population.module_feature, current_module_features)
        if current_environment_features is None:
            fixed_environment_features = None
        elif self.population.environment_feature is None:
            raise ValueError("Current Thermal environment features lack a population mean.")
        else:
            fixed_environment_features = _broadcast_vector(
                self.population.environment_feature, current_environment_features
            )
        fixed_encoded = replace(
            encoded,
            module_tokens=_broadcast_vector(self.population.module_token, encoded.module_tokens),
            env_tokens=_broadcast_vector(self.population.environment_token, encoded.env_tokens),
            global_token=_broadcast_vector(self.population.global_token, encoded.global_token),
            module_features=fixed_module_features,
            env_features=fixed_environment_features,
        )
        fixed_state = {
            "module_states": _broadcast_vector(
                self.population.module_state, prepared_state["module_states"]
            ),
            "environment_states": _broadcast_vector(
                self.population.environment_state, prepared_state["environment_states"]
            ),
            "global_state": _broadcast_vector(
                self.population.global_token, prepared_state["global_state"]
            ),
        }
        return self.base.score_cases(fixed_encoded, fixed_state, trees, budgets=budgets)

    def plans_from_scores(self, *args: Any, **kwargs: Any):
        return self.base.plans_from_scores(*args, **kwargs)


def _rewire_binary_matrix(
    membership: np.ndarray,
    active_rows: np.ndarray,
    source_validity: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply feasible bipartite 2x2 switches without touching invalid sources."""

    original = np.asarray(membership, dtype=np.int64)
    active = np.asarray(active_rows, dtype=bool)
    valid = np.asarray(source_validity, dtype=bool)
    if original.ndim != 2 or active.shape != (original.shape[0],) or valid.shape != (original.shape[1],):
        raise ValueError("Rewire membership, active rows, and valid source columns must align.")
    if not np.isin(original, (0, 1)).all():
        raise ValueError("Degree-preserving 2x2 switches require binary permissions.")
    if bool((original[:, ~valid] != 0).any()):
        raise ValueError("Invalid source columns already carry permission and cannot be rewired safely.")
    result = original.copy()
    rows = np.flatnonzero(active)
    columns = np.flatnonzero(valid)
    sub = result[np.ix_(rows, columns)].copy()
    edge_count = int(sub.sum())
    attempt_limit = min(2000, max(64, edge_count * 8))
    attempted = 0
    accepted = 0
    rng = np.random.default_rng(int(seed))
    for _ in range(attempt_limit):
        attempted += 1
        if len(rows) < 2 or len(columns) < 2:
            break
        row_a, row_b = rng.choice(len(rows), size=2, replace=False)
        col_a, col_b = rng.choice(len(columns), size=2, replace=False)
        pattern = (sub[row_a, col_a], sub[row_a, col_b], sub[row_b, col_a], sub[row_b, col_b])
        if pattern == (1, 0, 0, 1):
            sub[row_a, col_a], sub[row_a, col_b] = 0, 1
            sub[row_b, col_a], sub[row_b, col_b] = 1, 0
            accepted += 1
        elif pattern == (0, 1, 1, 0):
            sub[row_a, col_a], sub[row_a, col_b] = 1, 0
            sub[row_b, col_a], sub[row_b, col_b] = 0, 1
            accepted += 1
    result[np.ix_(rows, columns)] = sub
    changed_entries = int(np.count_nonzero(result != original))
    detail = {
        "attempted_switches": attempted,
        "attempt_budget": attempt_limit,
        "accepted_switches": accepted,
        "changed_link_entries": changed_entries,
        "available": bool(accepted > 0 and changed_entries > 0),
        "status": "active" if accepted > 0 and changed_entries > 0 else "inactive_no_valid_switch",
        "row_degrees_preserved": bool(np.array_equal(result.sum(axis=1), original.sum(axis=1))),
        "column_degrees_preserved": bool(np.array_equal(result.sum(axis=0), original.sum(axis=0))),
        "invalid_source_columns_unchanged": bool(np.array_equal(result[:, ~valid], original[:, ~valid])),
    }
    if not all(detail[key] for key in (
        "row_degrees_preserved", "column_degrees_preserved", "invalid_source_columns_unchanged"
    )):
        raise RuntimeError("The selected-weight rewire changed a degree or invalid source column.")
    return result, detail


def _rewire_plan(
    plan: MechanismPlan,
    encoded: Any,
    mechanism: str,
    *,
    seed: int,
) -> tuple[MechanismPlan, dict[str, Any]]:
    catalog = canonical_pair_catalog(encoded, plan.tree, mechanism, case_index=0)
    module_present = encoded.module_present[0] if mechanism in {"MM", "EM", "QM"} else None
    prior_tensor = plan.permission_matrix(mechanism, module_present=module_present).detach()
    active_rows = (
        plan.tree.access(plan.tree.universe.coordinates, plan.split_gates.detach()) > 0
    ).any(dim=0).detach().cpu().numpy()
    rewired, detail = _rewire_binary_matrix(
        prior_tensor.to(device="cpu", dtype=torch.int64).numpy(),
        active_rows,
        catalog.source_validity.detach().cpu().numpy(),
        seed=seed,
    )
    permission = torch.as_tensor(
        rewired,
        device=plan.split_gates.device,
        dtype=plan.split_gates.dtype,
    )
    updated = plan.with_permission(mechanism, permission)
    if not torch.equal(
        updated.permission_matrix(mechanism, module_present=module_present).detach().cpu(),
        permission.detach().cpu(),
    ):
        raise RuntimeError("MechanismPlan did not retain the validity-safe rewire permissions.")
    return updated, detail


def _root_source_union_plan(
    plan: MechanismPlan,
    encoded: Any,
    mechanism_cut: Sequence[int],
) -> tuple[MechanismPlan, dict[str, Any]]:
    """Collapse a selected hard cut to its root source union for MM and QE."""

    cut = tuple(int(row) for row in mechanism_cut)
    if not cut or len(set(cut)) != len(cut):
        raise ValueError("A root source union needs a nonempty, unique grouped cut.")
    if any(row < 0 or row >= len(plan.tree.nodes) for row in cut):
        raise ValueError("The selected grouped cut contains a receiver packet outside its tree.")
    if not torch.equal(plan.split_gates, plan.split_gates.round()):
        raise ValueError("A root source union requires hard split gates.")
    updated = plan
    route_details: dict[str, Any] = {}
    for mechanism in ROOT_UNION_MECHANISMS:
        catalog = canonical_pair_catalog(encoded, plan.tree, mechanism, case_index=0)
        module_present = encoded.module_present[0] if mechanism == "MM" else None
        original = plan.permission_matrix(mechanism, module_present=module_present).detach()
        if not bool(((original == 0) | (original == 1)).all()):
            raise ValueError("A root source union requires binary packet-source permissions.")
        valid = catalog.source_validity.to(device=original.device, dtype=torch.bool)
        union = (original[list(cut)] > 0).any(dim=0) & valid
        collapsed = torch.zeros_like(original)
        collapsed[0] = union.to(dtype=collapsed.dtype)
        before_work = stage_c._canonical_plan_route_work(encoded, plan.tree, plan, mechanism)
        updated = updated.with_permission(mechanism, collapsed)
        root_work = stage_c._canonical_plan_route_work(encoded, plan.tree, updated, mechanism)
        route_details[mechanism] = {
            "source_union_parent_cut_rows": list(cut),
            "source_union_parent_packet_count": len(cut),
            "parent_cut_supported_source_count": int(union.sum().item()),
            "root_source_count": int(collapsed[0].sum().item()),
            "invalid_source_columns_zeroed": bool((collapsed[:, ~valid] == 0).all()),
            "parent_cut_canonical_work": before_work,
            "root_union_canonical_work": root_work,
        }
    bypasses_before = set(plan.explicit_bypass_keys)
    if not updated.tree.nodes[0].is_leaf:
        updated = updated.with_split(0, 0.0)
    preserved_bypasses = sorted(bypasses_before - set(ROOT_UNION_MECHANISMS))
    if not set(preserved_bypasses).issubset(updated.explicit_bypass_keys):
        raise RuntimeError("The root source union changed a nonunion full-access bypass route.")
    route_details["root_split_closed"] = bool(
        updated.tree.nodes[0].is_leaf or float(updated.split_gates[0].detach()) == 0.0
    )
    route_details["full_access_bypass_routes_preserved"] = preserved_bypasses
    route_details["all_resulting_full_access_bypass_routes"] = list(updated.explicit_bypass_keys)
    return updated, route_details


class SelectedGInterventionBuilder:
    """Stage-C geometry control plus validity-aware degree-preserving rewires."""

    def __init__(self, base: Any, kind: str, *, seed: int) -> None:
        if kind not in {"geometry_only", "rewired", "root_source_union"}:
            raise ValueError(f"Unsupported selected-weight G intervention: {kind!r}.")
        self.base = base
        self.kind = kind
        self.seed = int(seed)
        self.last_records: tuple[dict[str, Any], ...] = ()
        self.last_plans: tuple[Any, ...] = ()
        self.last_state_captures: dict[str, Any] = {}

    def __call__(self, encoded: Any, base_module_state: torch.Tensor, trees: Sequence[Any]):
        plans = self.base(encoded, base_module_state, trees)
        outputs = []
        records = []
        for case, (tree, plan, original) in enumerate(
            zip(trees, plans, self.base.last_records, strict=True)
        ):
            updated = plan
            details: dict[str, Any] = {"kind": self.kind, "routes": {}}
            if self.kind == "root_source_union":
                updated, route_detail = _root_source_union_plan(
                    updated, encoded, original["frontier"]
                )
                details["routes"] = route_detail
            else:
                for mechanism in self.base.budget_fractions:
                    if self.kind == "geometry_only":
                        target = stage_c._canonical_plan_route_work(encoded, tree, plan, mechanism)["achieved_work"]
                        updated, route_detail = stage_c._geometry_only_plan(
                            encoded, updated, mechanism, target
                        )
                    else:
                        route_detail = {}
                        updated, route_detail = _rewire_plan(
                            updated,
                            encoded,
                            mechanism,
                            seed=self.seed + case * 37 + _stable_hash(mechanism),
                        )
                    route_detail.update(stage_c._canonical_plan_route_work(encoded, tree, updated, mechanism))
                    details["routes"][mechanism] = route_detail
            record: dict[str, Any] = {
                "mode": "G",
                "frontier": list(original["frontier"]),
                "support_count_frontier": [0] if self.kind == "root_source_union" else list(original["frontier"]),
                "full_access_bypass_routes": list(updated.explicit_bypass_keys),
                "routes": {},
                "intervention": details,
            }
            for mechanism, fraction in self.base.budget_fractions.items():
                work = stage_c._canonical_plan_route_work(encoded, tree, updated, mechanism)
                record["routes"][mechanism] = stage_c.RouteWorkRecord(
                    mechanism=mechanism,
                    requested_fraction=float(fraction),
                    requested_work=float(fraction) * float(work["full_access_work"]),
                    achieved_work=float(work["achieved_work"]),
                    full_access_work=float(work["full_access_work"]),
                    selected_unique_pairs=int(work["selected_unique_pairs"]),
                    full_unique_pairs=int(work["full_unique_pairs"]),
                    sparse_success=bool(work["sparse_success"]),
                    executor=f"same_weight_{self.kind}_intervention",
                )
            self.base._add_total_route_work(record, encoded, tree, case)
            outputs.append(updated)
            records.append(record)
        self.last_plans = tuple(outputs)
        self.last_records = tuple(records)
        return tuple(outputs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def _load_control_stencils(args: argparse.Namespace) -> tuple[list[Any], list[Any], dict[str, Any]]:
    train: list[Any] = []
    atlas_hashes: dict[str, str] = {}
    atlas_dir = args.atlas_dir.expanduser().resolve()
    for index, anchor_id in enumerate(forward.TRAIN_ATLAS_IDS):
        path = atlas_dir / f"train_{anchor_id}_responses.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw, _ = stage_c.load_response_atlas_stencil(path)
        if raw.split.value != "train":
            raise ValueError(f"The selected-weight feature population accepts train labels only: {path}")
        train.append(action_eval._target_free_pair(raw, seed=args.panel_seed + index))
        atlas_hashes[path.name] = _sha256(path)
    train_ids = tuple(item.physical_family_id for item in train)
    if len(train_ids) != MAX_TRAIN_FEATURE_FAMILIES or len(set(train_ids)) != len(train_ids):
        raise ValueError("The fixed organizer feature population must contain eight unique train families.")

    dev = list(stage_c._load_coverage_families(
        calls_path=args.coverage_manifest.expanduser().resolve(),
        predeclared_path=args.predeclared_manifest.expanduser().resolve(),
        partition="fit",
        target_free_sampling=True,
        sampling_seed=args.panel_seed + 101,
    ))
    expected_dev_ids = ("active_packet:case0319:Re70", "active_packet:case0349:Re50")
    if tuple(item.physical_family_id for item in dev) != expected_dev_ids:
        raise ValueError("The predeclared Thermal dev family IDs/order changed.")
    aliases = action_eval.shared_geometry_across_splits({
        "train_control": train_ids[:args.train_control_count],
        "dev_control": expected_dev_ids,
    })
    return train, dev, {
        "training_atlas_file_sha256": atlas_hashes,
        "train_feature_population_family_ids": list(train_ids),
        "selected_train_control_family_ids": list(train_ids[:args.train_control_count]),
        "selected_dev_control_family_ids": list(expected_dev_ids[:args.dev_control_count]),
        "shared_geometry_across_train_dev_contexts": aliases,
        "dev_contexts_are_correlated_development_evidence": True,
        "query_sampling": "fixed geometry and target-free training-panel sampling; labels are not used to choose coordinates",
    }


def _load_arm_runtime(
    arm: str,
    payload: Mapping[str, Any],
    *,
    source_payload: Mapping[str, Any],
    model: Any,
    encoded_probe: Any,
    template: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, Any, DifferentiableThermalOperator]:
    route = forward._route_module(arm, model.core, encoded_probe, device)
    physical_state, route_state = action_eval._state_parts(payload)
    incompatible = model.load_state_dict(physical_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"The selected {arm} physical state does not match Run1804 e4738: {incompatible}")
    incompatible = route.load_state_dict(route_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"The selected {arm} route state does not match its endpoint: {incompatible}")
    model.eval().requires_grad_(False)
    route.eval().requires_grad_(False)
    operator = action_eval._new_operator(model, source_payload, template, capture=False)
    operator._selected_weight_diagnostics_hook = stage_c._capture_cover_diagnostics(operator)
    return model, route, operator


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _guard_call_budget(stencil: Any, *, maximum: int) -> None:
    current = int(stage_c._FORWARD_ATTEMPT_NEXT_ID)
    attempt_count = len(stencil.records)
    if current + attempt_count > int(maximum):
        raise RuntimeError(
            f"Selected-weight native-state-forward cap {maximum} would be exceeded before this call."
        )


def _preflight_native_forward_budget(
    train_stencils: Sequence[Any],
    dev_stencils: Sequence[Any],
    *,
    train_control_count: int,
    dev_control_count: int,
    live_match_rounds: int = LIVE_MATCH_ROUNDS,
) -> dict[str, Any]:
    selected = [
        *train_stencils[: int(train_control_count)],
        *dev_stencils[: int(dev_control_count)],
    ]
    states = {row.physical_family_id: len(row.records) for row in (*train_stencils, *dev_stencils)}
    probe_states = len(train_stencils[0].records)
    train_feature_states = sum(len(row.records) for row in train_stencils)
    selected_dev_states = sum(len(row.records) for row in dev_stencils[: int(dev_control_count)])
    p_full_states = sum(len(row.records) for row in selected)
    g_sparse_states = sum(
        SPARSE_G_CONTROL_CALLS_PER_CASE * len(row.records) for row in selected
    )
    p_count_match_calls_per_case = 3 * int(live_match_rounds)
    p_sparse_states = sum(p_count_match_calls_per_case * len(row.records) for row in selected)
    total = (
        probe_states
        + train_feature_states
        + selected_dev_states
        + p_full_states
        + g_sparse_states
        + p_sparse_states
    )
    return {
        "worst_case_native_state_forwards": int(total),
        "states_per_family": states,
        "components": {
            "source_setup_probe": probe_states,
            "G_full_train_population_features": train_feature_states,
            "G_full_selected_dev_references": selected_dev_states,
            "P_full_selected_references": p_full_states,
            "G_sparse_controls": g_sparse_states,
            "P_live_count_match_max_rounds": p_sparse_states,
        },
        "selected_train_control_count": int(train_control_count),
        "selected_dev_control_count": int(dev_control_count),
        "G_sparse_controls_per_case": SPARSE_G_CONTROL_CALLS_PER_CASE,
        "P_live_count_match_rounds_per_action_upper_bound": int(live_match_rounds),
        "P_live_count_matched_actions_per_case": 3,
        "count_is_conservative_upper_bound": True,
    }


def _timed_sparse_prediction(
    operator: DifferentiableThermalOperator,
    stencil: Any,
    *,
    builder: Any,
    device: torch.device,
    max_native_state_forwards: int,
) -> tuple[Any, float, Mapping[str, Any]]:
    _guard_call_budget(stencil, maximum=max_native_state_forwards)
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        prediction, _capture, _record = stage_c._predict_stencil(
            operator,
            stencil,
            device=device,
            cover_plan_builder=builder,
            capture_plan_states=True,
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    state_captures = getattr(builder, "last_state_captures", {})
    if not isinstance(state_captures, Mapping):
        state_captures = {}
    return prediction, elapsed, state_captures


def _timed_full_prediction(
    operator: DifferentiableThermalOperator,
    stencil: Any,
    *,
    device: torch.device,
    max_native_state_forwards: int,
):
    _guard_call_budget(stencil, maximum=max_native_state_forwards)
    _synchronize(device)
    started = time.perf_counter()
    prediction, capture = stage_c._full_plan_predictions(
        operator, stencil, device, 512, capture_baseline=True
    )
    _synchronize(device)
    return prediction, capture, time.perf_counter() - started


def _route_live_rows(diagnostics: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        return {
            "status": "executor_row_counters_not_exposed",
            "by_mechanism": {name: None for name in MECHANISM_ORDER},
            "measured_total_executor_rows": None,
            "counter_keys": [],
        }
    selected: dict[str, int | None] = {}
    used_keys: list[str] = []
    for mechanism in MECHANISM_ORDER:
        prefix = LIVE_ROW_PREFIX[mechanism]
        keys = [
            str(key) for key in diagnostics
            if str(key).startswith(prefix) and str(key).endswith(("_executed_rows", "_actual_rows"))
        ]
        executed = [key for key in keys if key.endswith("_executed_rows")]
        chosen = executed or [key for key in keys if key.endswith("_actual_rows")]
        if not chosen:
            selected[mechanism] = None
            continue
        selected[mechanism] = sum(int(diagnostics[key]) for key in chosen)
        used_keys.extend(chosen)
    values = [selected[name] for name in MECHANISM_ORDER]
    total = sum(value for value in values if value is not None) if any(value is not None for value in values) else None
    return {
        "status": "measured_executor_rows" if total is not None else "executor_row_counters_not_exposed",
        "by_mechanism": selected,
        "measured_total_executor_rows": total,
        "counter_keys": sorted(used_keys),
    }


def _state_live_rows(state_captures: Mapping[str, Any], state: str = "baseline") -> dict[str, Any]:
    row = state_captures.get(state, {})
    diagnostics = row.get("runtime_diagnostics", {}) if isinstance(row, Mapping) else {}
    return _route_live_rows(diagnostics)


def _canonical_work(record: Mapping[str, Any]) -> dict[str, Any]:
    routes = record.get("all_mechanism_route_work")
    totals = record.get("all_mechanisms_total")
    if not isinstance(routes, Mapping) or not isinstance(totals, Mapping):
        routes, _fraction = stage_c._cut_work(record)
        routes = routes.get("all_mechanism_route_work", {})
        totals = routes.get("all_mechanisms_total", {}) if isinstance(routes, Mapping) else {}
    route_rows: dict[str, Any] = {}
    for mechanism in MECHANISM_ORDER:
        value = routes.get(mechanism, {}) if isinstance(routes, Mapping) else {}
        achieved = float(value.get("achieved_work", 0.0))
        full = float(value.get("full_access_work", 0.0))
        selected = int(value.get("selected_unique_pairs", 0))
        eligible = int(value.get("full_unique_pairs", 0))
        route_rows[mechanism] = {
            "canonical_achieved_work": achieved,
            "canonical_full_access_work": full,
            "canonical_work_fraction": achieved / full if full > 0 else 0.0,
            "selected_unique_pairs": selected,
            "eligible_canonical_pairs": eligible,
            "canonical_pair_density": selected / eligible if eligible > 0 else 0.0,
            "executor_label": value.get("executor"),
        }
    return {
        "by_mechanism": route_rows,
        "all_mechanisms": {
            "canonical_achieved_work": float(totals.get("achieved_work", 0.0)) if isinstance(totals, Mapping) else None,
            "canonical_full_access_work": float(totals.get("full_access_work", 0.0)) if isinstance(totals, Mapping) else None,
            "canonical_work_fraction": float(totals.get("work_fraction", 0.0)) if isinstance(totals, Mapping) else None,
            "selected_unique_pairs": int(totals.get("selected_unique_pairs", 0)) if isinstance(totals, Mapping) else None,
            "eligible_canonical_pairs": int(totals.get("full_unique_pairs", 0)) if isinstance(totals, Mapping) else None,
        },
        "explicit_full_access_bypass_routes": list(record.get("full_access_bypass_routes", ())),
    }


def _packet_support_summary(
    state_captures: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    """Report raw cut size separately from distinct nonempty MM/QE support K."""

    capture = state_captures.get("baseline")
    cut = tuple(int(value) for value in record.get("support_count_frontier", record.get("frontier", ())))
    result: dict[str, Any] = {
        "raw_packet_cut_size": len(cut),
        "nonredundant_packet_count": None,
        "nonredundant_packet_count_status": "unavailable_native_plan_capture",
        "nonredundant_packet_count_definition": "distinct nonempty combined MM/QE source permission patterns",
    }
    if not isinstance(capture, Mapping):
        return result
    encoded = capture.get("encoded")
    trees = capture.get("trees")
    plans = capture.get("plans")
    if encoded is None or not trees or not plans or not cut:
        return result
    signatures, details = action_eval._support_signature_and_details(
        encoded, trees[0], plans[0], cut
    )
    result.update(
        {
            "nonredundant_packet_count": len(signatures),
            "nonredundant_packet_count_status": "measured_combined_MM_QE_source_patterns",
            "nonredundant_packet_count_support": {
                "joint_nonempty_permission_patterns": int(details["joint_nonempty_permission_patterns"]),
                "hard_permission_matrices_binary": bool(details["hard_permission_matrices_binary"]),
            },
        }
    )
    return result


def _full_access_packet_count_fields() -> dict[str, Any]:
    return {
        "raw_packet_cut_size": None,
        "nonredundant_packet_count": None,
        "nonredundant_packet_count_status": "not_applicable_full_access",
        "nonredundant_packet_count_definition": "full access has no sparse packet K",
    }


def _role_metrics(prediction: Any, stencil: Any, *, full_rmse: Sequence[float] | None = None) -> dict[str, Any]:
    metrics = stage_c._pass_metrics(prediction, stencil)
    errors = np.asarray(metrics["role_rmse"], dtype=np.float64)
    result: dict[str, Any] = {
        "role_rmse_physical_units": errors.tolist(),
        "reference_role_rms_physical_units": np.asarray(metrics["reference_role_rms"], dtype=np.float64).tolist(),
    }
    if full_rmse is not None:
        baseline = np.asarray(full_rmse, dtype=np.float64)
        result["same_student_full_role_rmse_physical_units"] = baseline.tolist()
        result["signed_sparse_minus_full_role_rmse"] = (errors - baseline).tolist()
        result["ratio_to_same_student_full_role_rmse"] = (errors / np.maximum(baseline, 1.0e-12)).tolist()
    return result


def _control_record(
    *,
    label: str,
    prediction: Any,
    stencil: Any,
    builder: Any,
    state_captures: Mapping[str, Any],
    latency_seconds: float,
    full_rmse: Sequence[float],
    full_live_rows: Mapping[str, Any],
    intervention: str,
) -> dict[str, Any]:
    records = getattr(builder, "last_records", ())
    if not records:
        raise RuntimeError(f"The selected-weight {label} control did not retain a plan work record.")
    record = records[0]
    live_rows = {
        state: _state_live_rows(state_captures, state)
        for state in ("baseline", *tuple(stencil.variants))
    }
    live_density = {
        state: {
            mechanism: (
                None
                if live_rows[state]["by_mechanism"].get(mechanism) is None
                or full_live_rows.get("by_mechanism", {}).get(mechanism) in {None, 0}
                else live_rows[state]["by_mechanism"][mechanism]
                / int(full_live_rows["by_mechanism"][mechanism])
            )
            for mechanism in MECHANISM_ORDER
        }
        for state in live_rows
    }
    packet_support = _packet_support_summary(state_captures, record)
    if record.get("intervention", {}).get("kind") == "root_source_union":
        packet_support["source_union_parent_grouped_cut_size"] = len(record.get("frontier", ()))
    return {
        "label": label,
        "intervention": intervention,
        "frontier_paths": list(stage_c.frontier_paths(
            getattr(builder, "last_state_captures", {}).get("baseline", {}).get("trees", ())[0],
            record.get("frontier", ()),
            max_depth=3,
        )) if getattr(builder, "last_state_captures", {}).get("baseline") else list(record.get("frontier_paths", ())),
        "frontier_node_ids": list(record.get("frontier", ())),
        **packet_support,
        "physical_metrics": _role_metrics(prediction, stencil, full_rmse=full_rmse),
        "canonical_work_and_support": _canonical_work(record),
        "measured_live_executor_rows": live_rows,
        "live_executor_row_fraction_of_same_arm_full_access": live_density,
        "complete_stencil_call_latency_synchronized_seconds": float(latency_seconds),
        "intervention_details": record.get("intervention", {}),
    }


def _fraction_for_live_target(current: float, target_rows: int, observed_rows: int) -> float:
    if target_rows <= 0:
        return LIVE_MATCH_FLOOR
    if observed_rows <= 0:
        return min(1.0, max(LIVE_MATCH_FLOOR, current * 2.0))
    proposal = current * float(target_rows) / float(observed_rows)
    return min(1.0, max(LIVE_MATCH_FLOOR, proposal))


def _live_count_matched_p(
    *,
    stencil: Any,
    p_operator: Any,
    p_model: Any,
    p_route: Any,
    extra_route: str,
    cut: Sequence[int],
    frontier_paths: Sequence[str],
    target_g_live_rows: Mapping[str, Any],
    p_full_live_rows: Mapping[str, Any],
    p_full_rmse: Sequence[float],
    device: torch.device,
    seed: int,
    max_native_state_forwards: int,
    max_rounds: int = LIVE_MATCH_ROUNDS,
) -> dict[str, Any]:
    mechanisms = (extra_route, "QE")
    targets = {mechanism: target_g_live_rows.get("by_mechanism", {}).get(mechanism) for mechanism in mechanisms}
    if any(value is None for value in targets.values()):
        return {
            "status": "unavailable_G_live_row_counters",
            "target_live_executor_rows_by_mechanism": targets,
            "matched_live_executor_rows_by_mechanism": {mechanism: None for mechanism in mechanisms},
            "candidates": [],
            "canonical_work_is_not_used_as_a_live_row_substitute": True,
        }
    fractions = {mechanism: 0.90 for mechanism in mechanisms}
    candidates: list[dict[str, Any]] = []
    seen_fraction_vectors: set[tuple[float, ...]] = set()
    for attempt in range(int(max_rounds)):
        vector = tuple(round(fractions[mechanism], 8) for mechanism in mechanisms)
        if vector in seen_fraction_vectors:
            break
        seen_fraction_vectors.add(vector)
        budget = {mechanism: float(fractions[mechanism]) for mechanism in mechanisms}
        builder = stage_c._builder(
            "P", p_model, p_route, extra_route, cut,
            budget_fractions=budget,
            frontier_pattern_paths=frontier_paths,
        )
        prediction, latency, captures = _timed_sparse_prediction(
            p_operator,
            stencil,
            builder=builder,
            device=device,
            max_native_state_forwards=max_native_state_forwards,
        )
        live = _state_live_rows(captures, "baseline")
        observed = {mechanism: live["by_mechanism"].get(mechanism) for mechanism in mechanisms}
        work = _canonical_work(builder.last_records[0]) if builder.last_records else {}
        row = {
            "attempt": attempt + 1,
            "requested_budget_fractions": budget,
            "measured_live_executor_rows_by_mechanism": observed,
            "live_row_counters_status": live["status"],
            "canonical_work_and_support": work,
            "physical_metrics": _role_metrics(prediction, stencil, full_rmse=p_full_rmse),
            "live_executor_row_fraction_of_same_arm_full_access": {
                mechanism: (
                    None
                    if live["by_mechanism"].get(mechanism) is None
                    or p_full_live_rows.get("by_mechanism", {}).get(mechanism) in {None, 0}
                    else live["by_mechanism"][mechanism]
                    / int(p_full_live_rows["by_mechanism"][mechanism])
                )
                for mechanism in MECHANISM_ORDER
            },
            "complete_stencil_call_latency_synchronized_seconds": float(latency),
        }
        if any(value is None for value in observed.values()):
            row["absolute_live_row_gap"] = None
        else:
            row["absolute_live_row_gap"] = int(sum(
                abs(int(targets[mechanism]) - int(observed[mechanism])) for mechanism in mechanisms
            ))
        candidates.append(row)
        if live["status"] != "measured_executor_rows" or row["absolute_live_row_gap"] == 0:
            break
        previous = dict(fractions)
        for mechanism in mechanisms:
            fractions[mechanism] = _fraction_for_live_target(
                fractions[mechanism], int(targets[mechanism]), int(observed[mechanism])
            )
        if all(abs(fractions[key] - previous[key]) < 1.0e-8 for key in mechanisms):
            break
    if not candidates or candidates[-1]["live_row_counters_status"] != "measured_executor_rows":
        return {
            "status": "unavailable_P_live_row_counters",
            "target_live_executor_rows_by_mechanism": targets,
            "matched_live_executor_rows_by_mechanism": {mechanism: None for mechanism in mechanisms},
            "candidates": candidates,
            "canonical_work_is_not_used_as_a_live_row_substitute": True,
        }
    best = min(candidates, key=lambda row: int(row["absolute_live_row_gap"]))
    achieved = best["measured_live_executor_rows_by_mechanism"]
    exact = int(best["absolute_live_row_gap"]) == 0
    return {
        "status": "exact_live_row_match" if exact else "closest_measured_live_row_match",
        "match_is_diagnostic_only_across_independently_trained_G_and_P_weights": True,
        "match_scope": "QE and the additional trained route only; other explicit full-access bypasses are unchanged",
        "target_live_executor_rows_by_mechanism": targets,
        "matched_live_executor_rows_by_mechanism": achieved,
        "absolute_live_row_gap": int(best["absolute_live_row_gap"]),
        "selected_candidate_attempt": best["attempt"],
        "selected_candidate": best,
        "candidates": candidates,
        "canonical_work_is_not_used_as_a_live_row_substitute": True,
    }


def _load_native_arm_state(
    arm: str,
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    source_payload: Mapping[str, Any],
    encoded_probe: Any,
    template: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, Any, Any]:
    model, _ = stage_c.load_model(source_path, device)
    stage_c._configure_native_expanded_response_interface_scope(model)
    return _load_arm_runtime(
        arm,
        payload,
        source_payload=source_payload,
        model=model,
        encoded_probe=encoded_probe,
        template=template,
        device=device,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite selected-weight control evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    native_ledger_path = output / "native_state_forward_attempts.jsonl"
    stage_c._configure_forward_attempt_ledger(native_ledger_path, "selected_weight_controls")
    solver_ledger_path = output / "reference_solver_attempts.json"
    _write_json(solver_ledger_path, {
        "reference_solver_calls_attempted": 0,
        "reason": "All physical reference values are loaded from saved train/dev response atlases and coverage records.",
        "solver_started": False,
    })

    manifest_path = args.controlled_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arm_rows = manifest.get("arms", {})
    g_checkpoint = args.g_checkpoint.expanduser().resolve() if args.g_checkpoint else Path(
        str(arm_rows["G"]["latest_checkpoint"])
    ).expanduser().resolve()
    p_checkpoint = args.p_checkpoint.expanduser().resolve() if args.p_checkpoint else Path(
        str(arm_rows["P"]["latest_checkpoint"])
    ).expanduser().resolve()
    g_manifest, g_payload, g_sha = action_eval._load_checkpoint_binding(
        manifest_path, g_checkpoint, expected_arm="G"
    )
    p_manifest, p_payload, p_sha = action_eval._load_checkpoint_binding(
        manifest_path, p_checkpoint, expected_arm="P"
    )
    if g_manifest != p_manifest:
        raise RuntimeError("G and P endpoint bindings did not resolve against the same controlled manifest.")
    if int(g_payload.get("actual_optimizer_updates", -1)) != int(p_payload.get("actual_optimizer_updates", -2)):
        raise ValueError("Selected G and P endpoints must have the same completed-update count.")
    source_path = args.source_checkpoint.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha = _sha256(source_path)
    expected_source_sha = str(manifest.get("reference_checkpoint_sha256", ""))
    if source_sha != expected_source_sha:
        raise ValueError("Run1804 e4738 source checkpoint SHA differs from the controlled-run manifest.")

    train_stencils, dev_stencils, split_manifest = _load_control_stencils(args)
    if args.train_control_count < 1 or args.train_control_count > len(train_stencils):
        raise ValueError("train-control-count must select between one and all eight train families.")
    if args.dev_control_count < 1 or args.dev_control_count > len(dev_stencils):
        raise ValueError("dev-control-count must select one or both predeclared Stage-B dev families.")
    atlas_recorded = manifest.get("atlas_sha256", {})
    for name, digest in split_manifest["training_atlas_file_sha256"].items():
        if atlas_recorded.get(name) != digest:
            raise ValueError(f"Saved train atlas hash differs from controlled-run manifest: {name}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device} is unavailable.")
    lane_rows = _load_lane_execution_rows(manifest_path) if device.type == "cuda" else ()
    gpu_binding = _validate_cuda_gpu_binding(
        manifest,
        device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        queried_identity=_physical_gpu2_identity() if device.type == "cuda" else None,
        lane_rows=lane_rows,
    )
    native_forward_budget = _preflight_native_forward_budget(
        train_stencils,
        dev_stencils,
        train_control_count=args.train_control_count,
        dev_control_count=args.dev_control_count,
    )
    native_forward_budget["configured_cap"] = int(args.max_native_state_forwards)
    native_forward_budget["cap_sufficient"] = (
        native_forward_budget["worst_case_native_state_forwards"]
        <= int(args.max_native_state_forwards)
    )
    if not native_forward_budget["cap_sufficient"]:
        raise RuntimeError(
            "Configured native-state-forward cap is below the conservative selected-weight "
            f"preflight bound ({native_forward_budget['worst_case_native_state_forwards']} > "
            f"{args.max_native_state_forwards})."
        )
    run_started_at = time.time()
    gpu_ledger_path = output / "gpu_execution_ledger.jsonl"
    _append_jsonl(
        gpu_ledger_path,
        {
            "event": "job_start",
            "started_at_unix": run_started_at,
            "device": str(device),
            "gpu_binding": gpu_binding,
            "max_native_state_forwards": int(args.max_native_state_forwards),
            "native_forward_preflight": native_forward_budget,
            "g_checkpoint_sha256": g_sha,
            "p_checkpoint_sha256": p_sha,
            "command_argv": list(sys.argv),
        },
    )
    probe_model, source_payload, dataset_path, template = action_eval._load_source_and_template(
        source_path, device
    )
    stage_c._configure_native_expanded_response_interface_scope(probe_model)
    setup_operator = action_eval._new_operator(probe_model, source_payload, template, capture=True)
    anchor_stencil = train_stencils[0]
    _guard_call_budget(anchor_stencil, maximum=args.max_native_state_forwards)
    with torch.no_grad():
        _probe_predictions, probe_capture = stage_c._full_plan_predictions(
            setup_operator, anchor_stencil, device, 512, capture_baseline=True
        )
    if probe_capture is None:
        raise RuntimeError("The Run1804 input probe did not capture a Thermal organizer initialization case.")
    encoded_probe = probe_capture["encoded"]
    g_model, g_route, g_operator = _load_arm_runtime(
        "G", g_payload, source_payload=source_payload,
        model=probe_model, encoded_probe=encoded_probe, template=template, device=device,
    )
    p_model, p_route, p_operator = _load_native_arm_state(
        "P", p_payload, source_path=source_path,
        source_payload=source_payload, encoded_probe=encoded_probe,
        template=template, device=device,
    )
    g_before = {"physical": stage_c._state_hash(g_model.state_dict()), "route": stage_c._state_hash(g_route.state_dict())}
    p_before = {"physical": stage_c._state_hash(p_model.state_dict()), "route": stage_c._state_hash(p_route.state_dict())}

    population_captures: dict[str, Mapping[str, Any]] = {}
    full_cache_g: dict[str, tuple[Any, Mapping[str, Any], float]] = {}
    for stencil in train_stencils:
        prediction, capture, latency = _timed_full_prediction(
            g_operator, stencil, device=device,
            max_native_state_forwards=args.max_native_state_forwards,
        )
        if capture is None:
            raise RuntimeError(f"G train feature capture failed for {stencil.physical_family_id}.")
        population_captures[stencil.physical_family_id] = capture
        full_cache_g[stencil.physical_family_id] = (prediction, capture, latency)
    population_features = _train_population_features(population_captures)

    selected_stencils = [
        ("train", stencil) for stencil in train_stencils[:args.train_control_count]
    ] + [
        ("dev", stencil) for stencil in dev_stencils[:args.dev_control_count]
    ]
    extra_route = str(manifest.get("extra_route", "")).upper()
    if extra_route not in {"MM", "ME"}:
        raise ValueError("The controlled manifest must bind the selected Thermal extra route (MM or ME).")
    budget = {"QE": 0.90, extra_route: 0.90}
    rows_path = output / "selected_weight_control_rows.jsonl"
    if rows_path.exists():
        raise FileExistsError(rows_path)
    case_rows = []
    _append_jsonl(rows_path, {
        "event": "evaluation_start",
        "g_checkpoint_sha256": g_sha,
        "p_checkpoint_sha256": p_sha,
        "train_population_families": list(population_features.family_ids),
        "reference_solver_calls_attempted": 0,
    })

    for split, stencil in selected_stencils:
        case_id = stencil.physical_family_id
        if split == "train":
            g_full, g_full_capture, g_full_latency = full_cache_g[case_id]
        else:
            g_full, g_full_capture, g_full_latency = _timed_full_prediction(
                g_operator, stencil, device=device,
                max_native_state_forwards=args.max_native_state_forwards,
            )
        p_full, p_full_capture, p_full_latency = _timed_full_prediction(
            p_operator, stencil, device=device,
            max_native_state_forwards=args.max_native_state_forwards,
        )
        if g_full_capture is None or p_full_capture is None:
            raise RuntimeError(f"Full-access capture missing for {case_id}.")
        g_encoded = g_full_capture["encoded"]
        g_tree = g_full_capture["trees"][0]
        p_tree = p_full_capture["trees"][0]
        g_full_record = stage_c._full_work_record(g_encoded, g_tree)
        p_full_record = stage_c._full_work_record(p_full_capture["encoded"], p_tree)
        g_full_live = _route_live_rows(
            g_full_capture.get("state_runtime_diagnostics", {}).get("baseline", {})
        )
        p_full_live = _route_live_rows(
            p_full_capture.get("state_runtime_diagnostics", {}).get("baseline", {})
        )
        g_full_metrics = stage_c._pass_metrics(g_full, stencil)
        p_full_metrics = stage_c._pass_metrics(p_full, stencil)
        action_outputs: dict[str, Any] = {}

        for action_index, action in enumerate(("root", "two_packet", "four_packet")):
            cut, paths = available_frontier_for_paths(
                g_tree, THERMAL_ACTION_PATHS[action], max_depth=3
            )
            base = stage_c._builder(
                "G", g_model, g_route, extra_route, cut,
                budget_fractions=budget,
                frontier_pattern_paths=paths,
            )
            g_prediction, g_latency, g_captures = _timed_sparse_prediction(
                g_operator, stencil, builder=base, device=device,
                max_native_state_forwards=args.max_native_state_forwards,
            )
            g_action_row = _control_record(
                label=("G_root_action" if action == "root" else f"G_{action}"),
                prediction=g_prediction, stencil=stencil, builder=base,
                state_captures=g_captures, latency_seconds=g_latency,
                full_rmse=g_full_metrics["role_rmse"],
                full_live_rows=g_full_live,
                intervention="same-weight grouped selected path action",
            )
            g_action_row["action"] = action
            _append_jsonl(rows_path, {"case_id": case_id, "split": split, "kind": "G_grouped_action", **g_action_row})

            root_union_row = None
            if action != "root":
                root_union_builder = SelectedGInterventionBuilder(
                    stage_c._builder(
                        "G", g_model, g_route, extra_route, cut,
                        budget_fractions=budget,
                        frontier_pattern_paths=paths,
                    ),
                    "root_source_union",
                    seed=args.seed + action_index * 4001,
                )
                root_union_prediction, root_union_latency, root_union_captures = _timed_sparse_prediction(
                    g_operator, stencil, builder=root_union_builder, device=device,
                    max_native_state_forwards=args.max_native_state_forwards,
                )
                root_union_row = _control_record(
                    label=f"G_{action}_root_source_union",
                    prediction=root_union_prediction,
                    stencil=stencil,
                    builder=root_union_builder,
                    state_captures=root_union_captures,
                    latency_seconds=root_union_latency,
                    full_rmse=g_full_metrics["role_rmse"],
                    full_live_rows=g_full_live,
                    intervention="same-case source union of the selected grouped cut at a closed root packet",
                )
                root_union_row["action"] = action
                root_union_row["source_union_parent_grouped_control"] = g_action_row["label"]
                _append_jsonl(
                    rows_path,
                    {"case_id": case_id, "split": split, "kind": "G_root_source_union_control", **root_union_row},
                )

            if action != "root":
                geometry_builder = SelectedGInterventionBuilder(
                    stage_c._builder("G", g_model, g_route, extra_route, cut,
                                     budget_fractions=budget, frontier_pattern_paths=paths),
                    "geometry_only", seed=args.seed + action_index * 1009,
                )
                geometry_prediction, geometry_latency, geometry_captures = _timed_sparse_prediction(
                    g_operator, stencil, builder=geometry_builder, device=device,
                    max_native_state_forwards=args.max_native_state_forwards,
                )
                geometry_row = _control_record(
                    label=f"G_{action}_geometry_only", prediction=geometry_prediction,
                    stencil=stencil, builder=geometry_builder, state_captures=geometry_captures,
                    latency_seconds=geometry_latency, full_rmse=g_full_metrics["role_rmse"],
                    full_live_rows=g_full_live,
                    intervention="same-weight nearest source to packet-anchor centroid at matched canonical work",
                )
                geometry_row["action"] = action
                _append_jsonl(rows_path, {"case_id": case_id, "split": split, "kind": "G_geometry_control", **geometry_row})

                rewire_builder = SelectedGInterventionBuilder(
                    stage_c._builder("G", g_model, g_route, extra_route, cut,
                                     budget_fractions=budget, frontier_pattern_paths=paths),
                    "rewired", seed=args.seed + action_index * 2017,
                )
                rewire_prediction, rewire_latency, rewire_captures = _timed_sparse_prediction(
                    g_operator, stencil, builder=rewire_builder, device=device,
                    max_native_state_forwards=args.max_native_state_forwards,
                )
                rewire_row = _control_record(
                    label=f"G_{action}_degree_size_rewired", prediction=rewire_prediction,
                    stencil=stencil, builder=rewire_builder, state_captures=rewire_captures,
                    latency_seconds=rewire_latency, full_rmse=g_full_metrics["role_rmse"],
                    full_live_rows=g_full_live,
                    intervention="same-weight validity-aware bipartite 2x2 switches preserving typed packet/source degrees",
                )
                rewire_row["action"] = action
                _append_jsonl(rows_path, {"case_id": case_id, "split": split, "kind": "G_rewire_control", **rewire_row})

            fixed_route = FixedTrainPopulationOrganizer(g_route, population_features)
            fixed_builder = stage_c._builder(
                "G", g_model, fixed_route, extra_route, cut,
                budget_fractions=budget,
                frontier_pattern_paths=paths,
            )
            fixed_prediction, fixed_latency, fixed_captures = _timed_sparse_prediction(
                g_operator, stencil, builder=fixed_builder, device=device,
                max_native_state_forwards=args.max_native_state_forwards,
            )
            fixed_row = _control_record(
                label=f"G_{action}_fixed_train_population_features", prediction=fixed_prediction,
                stencil=stencil, builder=fixed_builder, state_captures=fixed_captures,
                latency_seconds=fixed_latency, full_rmse=g_full_metrics["role_rmse"],
                full_live_rows=g_full_live,
                intervention="same G weights and case geometry with organizer state/features replaced by equal-family train means",
            )
            fixed_row["action"] = action
            fixed_row["fixed_feature_population"] = {
                "family_ids": list(population_features.family_ids),
                "aggregation": "equal-family mean of each family's active-module mean, quadrature-weighted environment mean, and global token",
                "target_values_used": False,
                "case_geometry_masks_and_quadrature_weights_preserved": True,
            }
            _append_jsonl(rows_path, {"case_id": case_id, "split": split, "kind": "G_fixed_population_control", **fixed_row})

            matched = _live_count_matched_p(
                stencil=stencil, p_operator=p_operator, p_model=p_model, p_route=p_route,
                extra_route=extra_route, cut=cut, frontier_paths=paths,
                target_g_live_rows=g_action_row["measured_live_executor_rows"]["baseline"],
                p_full_live_rows=p_full_live,
                p_full_rmse=p_full_metrics["role_rmse"],
                device=device, seed=args.seed + action_index * 3001,
                max_native_state_forwards=args.max_native_state_forwards,
            )
            matched["action"] = action
            matched["comparison_scope"] = "independently trained P weights, separate from same-weight G controls"
            matched["P_full_access_reference"] = {
                **_full_access_packet_count_fields(),
                "role_rmse_physical_units": p_full_metrics["role_rmse"],
                "canonical_work_and_support": _canonical_work(p_full_record),
                "measured_live_executor_rows": _route_live_rows(
                    p_full_capture.get("state_runtime_diagnostics", {}).get("baseline", {})
                ),
                "complete_stencil_call_latency_synchronized_seconds": p_full_latency,
            }
            _append_jsonl(rows_path, {"case_id": case_id, "split": split, "kind": "G_vs_P_live_count_match", **matched})
            action_outputs[action] = {
                "G_grouped_action": g_action_row,
                "G_root_source_union": root_union_row,
                "G_geometry_control": None if action == "root" else geometry_row,
                "G_rewire_control": None if action == "root" else rewire_row,
                "G_fixed_population_features": fixed_row,
                "P_live_count_match": matched,
            }

        case_row = {
            "case_id": case_id,
            "split": split,
            "source_provenance": dict(stencil.evidence_manifest),
            "role_schema": stage_c._role_rows(stencil),
            "G_full_access": {
                **_full_access_packet_count_fields(),
                "physical_metrics": _role_metrics(g_full, stencil),
                "canonical_work_and_support": _canonical_work(g_full_record),
                "measured_live_executor_rows": _route_live_rows(
                    g_full_capture.get("state_runtime_diagnostics", {}).get("baseline", {})
                ),
                "live_executor_row_fraction_of_same_arm_full_access": {
                    name: (1.0 if count not in {None, 0} else None)
                    for name, count in g_full_live["by_mechanism"].items()
                },
                "complete_stencil_call_latency_synchronized_seconds": g_full_latency,
            },
            "P_full_access": {
                **_full_access_packet_count_fields(),
                "physical_metrics": _role_metrics(p_full, stencil),
                "canonical_work_and_support": _canonical_work(p_full_record),
                "measured_live_executor_rows": _route_live_rows(
                    p_full_capture.get("state_runtime_diagnostics", {}).get("baseline", {})
                ),
                "live_executor_row_fraction_of_same_arm_full_access": {
                    name: (1.0 if count not in {None, 0} else None)
                    for name, count in p_full_live["by_mechanism"].items()
                },
                "complete_stencil_call_latency_synchronized_seconds": p_full_latency,
                "note": "P full access uses P's own physical weights; it is not a same-weight G intervention.",
            },
            "actions": action_outputs,
        }
        case_rows.append(case_row)
        _append_jsonl(rows_path, {"event": "case_complete", "case_id": case_id, "split": split})

    g_after = {"physical": stage_c._state_hash(g_model.state_dict()), "route": stage_c._state_hash(g_route.state_dict())}
    p_after = {"physical": stage_c._state_hash(p_model.state_dict()), "route": stage_c._state_hash(p_route.state_dict())}
    unchanged = g_before == g_after and p_before == p_after
    if not unchanged:
        raise RuntimeError("Selected-weight control evaluation changed a frozen G or P state tensor.")
    for operator in (g_operator, p_operator):
        hook = getattr(operator, "_selected_weight_diagnostics_hook", None)
        if hook is not None:
            hook.remove()
    stage_c._finish_forward_phase()
    native_ledger = stage_c._summarize_forward_attempt_ledger(native_ledger_path)
    active_wall_seconds = max(0.0, time.time() - run_started_at)
    gpu_associated_active_seconds = active_wall_seconds if device.type == "cuda" else None
    _append_jsonl(
        gpu_ledger_path,
        {
            "event": "job_stop",
            "stopped_at_unix": time.time(),
            "status": "completed",
            "active_wall_seconds": active_wall_seconds,
            "gpu_associated_active_seconds": gpu_associated_active_seconds,
            "gpu_binding": gpu_binding,
            "native_state_forwards_attempted": native_ledger["attempted_native_state_forwards"],
            "native_state_forwards_completed": native_ledger["completed_native_state_forwards"],
        },
    )
    result = {
        "format_version": 1,
        "status": "completed_selected_weight_controls",
        "scope": {
            "study_label": "bounded train-only and correlated development diagnostic; not independent generalization",
            "same_weight_G_controls": ["root source union", "geometry-selected support", "degree/size-preserving rewire", "fixed train-population organizer features"],
            "separately_trained_G_vs_P": "diagnostic live-row count matching; physical and route weights differ across arms",
            "reference_kind": "saved local analytic-wake/shared-grid labels; no CFD validation",
            "solver_attempts": 0,
            "selector_fit_performed": False,
        },
        "provenance": {
            "driver_script_path": str(Path(__file__).resolve()),
            "driver_script_sha256": _sha256(Path(__file__).resolve()),
            "command_argv": list(sys.argv),
            "working_directory": str(Path.cwd()),
            "python_executable": sys.executable,
            "controlled_run_id": manifest.get("run_id"),
            "controlled_manifest_path": str(manifest_path),
            "controlled_manifest_sha256": _sha256(manifest_path),
            "reference_checkpoint": str(source_path),
            "reference_checkpoint_sha256": source_sha,
            "reference_checkpoint_identity": "Run1804 epoch-4738 Dense field checkpoint",
            "G_checkpoint": {"path": str(g_checkpoint), "sha256": g_sha, "update": int(g_payload["actual_optimizer_updates"])},
            "P_checkpoint": {"path": str(p_checkpoint), "sha256": p_sha, "update": int(p_payload["actual_optimizer_updates"])},
            "dataset_path": str(dataset_path),
            "dataset_sha256": manifest.get("dataset_sha256"),
            "extra_route": extra_route,
            "primary_capacity_fractions": budget,
            "query_batch_size": 512,
            "training_feature_population": {
                "family_ids": list(population_features.family_ids),
                "source": "eight original train response atlases, baseline state only",
                "input_only": True,
                "reference_values_used": False,
            },
            **split_manifest,
        },
        "weight_state_hashes": {
            "G_before": g_before, "G_after": g_after,
            "P_before": p_before, "P_after": p_after,
            "unchanged": unchanged,
        },
        "native_state_forward_ledger": native_ledger,
        "native_state_forward_ledger_path": str(native_ledger_path),
        "native_state_forward_budget": native_forward_budget,
        "gpu_execution": {
            "binding": gpu_binding,
            "active_wall_seconds": active_wall_seconds,
            "gpu_associated_active_seconds": gpu_associated_active_seconds,
            "ledger_path": str(gpu_ledger_path),
        },
        "reference_solver_ledger_path": str(solver_ledger_path),
        "table_path": str(rows_path),
        "case_count": len(case_rows),
        "case_rows": case_rows,
        "limitations": [
            "The panel is bounded to two selected train families and the two predeclared Stage-B development families.",
            "Shared geometry across train and dev contexts makes dev results correlated development evidence.",
            "Canonical pair support, live executor rows, and complete synchronized latency are reported separately.",
            "A zero-switch rewire is labeled inactive; it does not establish robustness or irrelevance.",
            "The P count-match is an independently trained system diagnostic, not a same-weight intervention or causal claim.",
        ],
    }
    _write_json(output / "selected_weight_controls.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--g-checkpoint", type=Path, default=None)
    parser.add_argument("--p-checkpoint", type=Path, default=None)
    parser.add_argument("--source-checkpoint", type=Path, default=stage_c.DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--coverage-manifest", type=Path, default=DEFAULT_COVERAGE_MANIFEST)
    parser.add_argument("--predeclared-manifest", type=Path, default=DEFAULT_PREDECLARED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-control-count", type=int, default=2)
    parser.add_argument("--dev-control-count", type=int, default=2)
    parser.add_argument("--max-native-state-forwards", type=int, default=1500)
    parser.add_argument("--panel-seed", type=int, default=2317)
    parser.add_argument("--seed", type=int, default=7019)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "case_count": result["case_count"],
                "native_state_forwards_attempted": result["native_state_forward_ledger"][
                    "attempted_native_state_forwards"
                ],
                "G_checkpoint_sha256": result["provenance"]["G_checkpoint"]["sha256"],
                "P_checkpoint_sha256": result["provenance"]["P_checkpoint"]["sha256"],
                "native_state_forward_cap": result["native_state_forward_budget"]["configured_cap"],
                "native_state_forward_preflight_bound": result["native_state_forward_budget"][
                    "worst_case_native_state_forwards"
                ],
                "gpu_associated_active_seconds": result["gpu_execution"][
                    "gpu_associated_active_seconds"
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
