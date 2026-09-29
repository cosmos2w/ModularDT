"""Measure Thermal depth-three frontiers and fit an input-only utility head.

The fit phase reads the eight original train atlases plus two predeclared
two-state reference pairs. It freezes case rankings before the two remaining
reference families are opened by the separate held-eval phase. No solver is
started here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import zlib
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.evaluation.loading import load_model
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.interaction_evidence.reference_adapter import (
    AnalyticWakeReferenceAdapter,
    load_stored_reference_case,
    read_embedded_case_config,
)
from channelthermal.interaction_evidence.types import (
    DesignState,
    EvidenceSplit,
    MeasuredQuantity,
    ModuleState,
    OperatingContext,
    PhysicalSolveOutput,
    SolveRecord,
    SolveStatus,
)
from channelthermal.response_control.active_packet import ThermalCoverPlanBuilder
from channelthermal.response_control.active_packet import RouteWorkRecord
from channelthermal.response_control.algebra import StencilPredictions
from channelthermal.response_control.contracts import (
    DesignInput,
    context_inputs,
    role_queries_from_record,
    role_queries_from_stencil,
)
from channelthermal.response_control.native import DifferentiableThermalOperator
from channelthermal.response_control.runner import (
    _configure_native_expanded_response_interface_scope,
    _make_input_template,
    _resolve_dataset_path,
)
from channelthermal.response_control import sampling as _sampling
from channelthermal.response_control.sampling import ReceiverSamplingConfig, sample_training_panel
from honf_forward_core.interface_fields.budgeted_frontier import (
    enumerate_frontier_cuts,
    frontier_from_paths,
    frontier_paths,
    normalized_frontier_distortion_targets,
    normalized_frontier_work_targets,
    select_frontier_by_predictions,
)
from honf_forward_core.interface_fields.interaction_interface import interaction_interface_from_plan
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer

import run_active_packet_forward as forward


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CHECKPOINT = (
    PROJECT_ROOT / "Trained_Results/ThermalChannel/HONF_Forward_Runs/"
    "Run_1804_20260905_081349_dense_pairwise_field_adaptation/checkpoints/best_field.pt"
)
DEFAULT_FORWARD_RUN = (
    PROJECT_ROOT / "Trained_Results/ThermalChannel/HONF_Forward_Runs/"
    "Run_1509_20260928_active_packet_organization_attempt07"
)
DEFAULT_G_CHECKPOINT = (
    DEFAULT_FORWARD_RUN / "checkpoints/G_u0200_training_checkpoint.pt"
)
DEFAULT_P_CHECKPOINT = DEFAULT_FORWARD_RUN / "checkpoints/P_u0200_training_checkpoint.pt"
DEFAULT_ATLAS = PROJECT_ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"
DEFAULT_OUTPUT = DEFAULT_FORWARD_RUN / "stage_b_c_u0200"
DEFAULT_COVERAGE_ROOT = PROJECT_ROOT / "diagnostics/generated/active_packet_reuse_20260929/thermal_coverage"
DEFAULT_COVERAGE_MANIFEST = DEFAULT_COVERAGE_ROOT / "thermal_coverage_calls_v2.json"
DEFAULT_PREDECLARED = PROJECT_ROOT / "diagnostics/generated/active_packet_reuse_20260929/thermal_training_coverage_predeclared.json"
DEFAULT_HDF5 = Path("/data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5")
DEFAULT_DEMO = PROJECT_ROOT.parent / "1_Demo_ChannelThermal"

_FORWARD_ATTEMPT_LEDGER: Path | None = None
_FORWARD_ATTEMPT_PHASE = "unconfigured"
_FORWARD_ATTEMPT_INVOCATION = ""
_FORWARD_ATTEMPT_NEXT_ID = 0


class UnsupportedFrontierPatternError(ValueError):
    """A declared cut's structural split pattern is absent from a state tree."""

    def __init__(self, state_label: str, message: str) -> None:
        super().__init__(message)
        self.state_label = str(state_label)


def _append_forward_event(event: Mapping[str, Any]) -> None:
    if _FORWARD_ATTEMPT_LEDGER is None:
        return
    _FORWARD_ATTEMPT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _FORWARD_ATTEMPT_LEDGER.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _configure_forward_attempt_ledger(path: Path, phase: str) -> None:
    """Append auditable start/end rows for every native state prediction."""

    global _FORWARD_ATTEMPT_LEDGER, _FORWARD_ATTEMPT_PHASE
    global _FORWARD_ATTEMPT_INVOCATION, _FORWARD_ATTEMPT_NEXT_ID
    path.parent.mkdir(parents=True, exist_ok=True)
    next_id = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Malformed native-forward ledger row {line_number}: {path}") from exc
                if row.get("event") == "native_state_start":
                    next_id = max(next_id, int(row["attempt_id"]))
    _FORWARD_ATTEMPT_LEDGER = path
    _FORWARD_ATTEMPT_PHASE = str(phase)
    _FORWARD_ATTEMPT_NEXT_ID = next_id
    _FORWARD_ATTEMPT_INVOCATION = f"{os.getpid()}-{time.time_ns()}"
    _append_forward_event({
        "event": "phase_start",
        "phase": _FORWARD_ATTEMPT_PHASE,
        "invocation_id": _FORWARD_ATTEMPT_INVOCATION,
        "pid": os.getpid(),
        "wall_time_epoch_seconds": time.time(),
    })


def _begin_native_state_forward(case_id: str, state_label: str, device: torch.device) -> int | None:
    global _FORWARD_ATTEMPT_NEXT_ID
    if _FORWARD_ATTEMPT_LEDGER is None:
        return None
    if _FORWARD_ATTEMPT_NEXT_ID >= 2048:
        raise RuntimeError("Thermal Stage-B/C native-state-forward cap reached before operator call.")
    _FORWARD_ATTEMPT_NEXT_ID += 1
    attempt_id = _FORWARD_ATTEMPT_NEXT_ID
    _append_forward_event({
        "event": "native_state_start",
        "attempt_id": attempt_id,
        "phase": _FORWARD_ATTEMPT_PHASE,
        "invocation_id": _FORWARD_ATTEMPT_INVOCATION,
        "pid": os.getpid(),
        "case_id": str(case_id),
        "state_label": str(state_label),
        "device": str(device),
        "physical_gpu": 2 if device.type == "cuda" else None,
        "wall_time_epoch_seconds": time.time(),
    })
    return attempt_id


def _end_native_state_forward(attempt_id: int | None, status: str, error: BaseException | None = None) -> None:
    if attempt_id is None:
        return
    row: dict[str, Any] = {
        "event": "native_state_end",
        "attempt_id": int(attempt_id),
        "status": str(status),
        "wall_time_epoch_seconds": time.time(),
    }
    if error is not None:
        row["error_type"] = type(error).__name__
        row["error"] = str(error)[:500]
    _append_forward_event(row)


def _summarize_forward_attempt_ledger(path: Path) -> dict[str, Any]:
    starts: dict[int, dict[str, Any]] = {}
    ends: dict[int, dict[str, Any]] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") == "native_state_start":
                    starts[int(row["attempt_id"])] = row
                elif row.get("event") == "native_state_end":
                    ends[int(row["attempt_id"])] = row
    completed = sum(row.get("status") == "completed" for row in ends.values())
    failed = sum(row.get("status") == "failed" for row in ends.values())
    pending = sorted(set(starts) - set(ends))
    by_phase: dict[str, dict[str, int]] = {}
    for attempt_id, row in starts.items():
        phase_counts = by_phase.setdefault(str(row.get("phase", "unknown")), {
            "attempted": 0, "completed": 0, "failed": 0, "pending_or_interrupted": 0,
        })
        phase_counts["attempted"] += 1
        end = ends.get(attempt_id)
        if end is None:
            phase_counts["pending_or_interrupted"] += 1
        elif end.get("status") == "completed":
            phase_counts["completed"] += 1
        else:
            phase_counts["failed"] += 1
    return {
        "path": str(path),
        "attempted_native_state_forwards": len(starts),
        "completed_native_state_forwards": completed,
        "failed_native_state_forwards": failed,
        "pending_or_interrupted_native_state_forwards": len(pending),
        "pending_attempt_ids": pending,
        "by_phase": by_phase,
        "counting_unit": "one attempted operator call for one design/state and the complete native role panel",
    }


def _finish_forward_phase() -> None:
    _append_forward_event({
        "event": "phase_end",
        "phase": _FORWARD_ATTEMPT_PHASE,
        "invocation_id": _FORWARD_ATTEMPT_INVOCATION,
        "pid": os.getpid(),
        "wall_time_epoch_seconds": time.time(),
    })


@dataclass(frozen=True)
class ThermalStageStencil:
    """A role-aligned Thermal pair that preserves mixed evidence sources.

    ``ResponseStencil`` intentionally requires one source enum across a
    family.  The predeclared Stage-B additions include packed stored baselines
    paired with local reference-solver variants, so this local adapter keeps
    every original ``SolveRecord.source`` unchanged and validates only the
    physical-family, context, module-order, and receiver-alignment contracts.
    """

    baseline: SolveRecord
    variants: Mapping[str, SolveRecord]
    evidence_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        variants = dict(self.variants)
        if not variants or "baseline" in variants:
            raise ValueError("A Thermal Stage-B stencil needs one or more named variants.")
        records = (self.baseline, *variants.values())
        for record in records:
            if record.status is not SolveStatus.CONVERGED or record.output is None:
                raise ValueError(f"Stage-B record {record.record_id!r} is not converged.")
            if record.design.physical_family_id != self.baseline.design.physical_family_id:
                raise ValueError("A Thermal Stage-B pair cannot cross physical families.")
            if record.design.split is not self.baseline.design.split:
                raise ValueError("A Thermal Stage-B pair cannot cross partitions.")
            if record.design.active_module_ids != self.baseline.design.active_module_ids:
                raise ValueError("A Thermal Stage-B response must preserve module identities/order.")
            if dict(record.context.values) != dict(self.baseline.context.values):
                raise ValueError("A Thermal Stage-B response must preserve operating context.")
            if set(record.output.roles) != set(self.baseline.output.roles):
                raise ValueError("A Thermal Stage-B response must preserve typed roles.")
            for role_name, base_role in self.baseline.output.roles.items():
                candidate = record.output.roles[role_name]
                if (
                    base_role.query_ids != candidate.query_ids
                    or base_role.channel_names != candidate.channel_names
                    or base_role.channel_units != candidate.channel_units
                    or base_role.coordinate_kind != candidate.coordinate_kind
                    or base_role.query_features.shape != candidate.query_features.shape
                    # The stored HDF5 baseline reconstructs interface angles
                    # from float32 directions while the local adapter rebuilds
                    # the same ports from its native coordinates. This is a
                    # sub-microunit representation difference, not a changed
                    # receiver row; query IDs/order remain exact.
                    or not np.allclose(base_role.query_features, candidate.query_features, atol=5.0e-7, rtol=0.0)
                ):
                    raise ValueError(f"Thermal Stage-B {role_name} receivers are not aligned.")
        object.__setattr__(self, "variants", MappingProxyType(variants))
        object.__setattr__(self, "evidence_manifest", MappingProxyType(dict(self.evidence_manifest)))

    @property
    def physical_family_id(self) -> str:
        return self.baseline.design.physical_family_id

    @property
    def split(self) -> EvidenceSplit:
        return EvidenceSplit(self.baseline.design.split)

    @property
    def records(self) -> tuple[SolveRecord, ...]:
        return (self.baseline, *self.variants.values())

    def common_mask(self, role: str) -> np.ndarray:
        outputs = [record.output.roles[role] for record in self.records]  # type: ignore[union-attr]
        first = outputs[0]
        for candidate in outputs[1:]:
            if (
                first.query_ids != candidate.query_ids
                or first.query_features.shape != candidate.query_features.shape
                or not np.allclose(first.query_features, candidate.query_features, atol=5.0e-7, rtol=0.0)
            ):
                raise ValueError(f"Thermal Stage-B {role} receiver panel changed within the pair.")
        return np.logical_and.reduce([output.valid_mask for output in outputs])


@dataclass(frozen=True)
class ThermalInputCandidate:
    """Geometry-only candidate input for ranking before held labels are opened."""

    baseline: SolveRecord
    variants: Mapping[str, SolveRecord]
    evidence_manifest: Mapping[str, Any]

    @property
    def physical_family_id(self) -> str:
        return self.baseline.design.physical_family_id

    @property
    def split(self) -> EvidenceSplit:
        return EvidenceSplit(self.baseline.design.split)


def _pair_from_atlas(stencil: Any, *, seed: int = 2317) -> ThermalStageStencil:
    if "i_plus" not in stencil.variants:
        raise ValueError(f"Original Thermal atlas has no i_plus variant: {stencil.physical_family_id}")
    sampled = sample_training_panel((stencil,), config=ReceiverSamplingConfig(
        max_fluid_queries=128,
        solid_queries_per_module=8,
        hot_solid_points_per_module=4,
        random_seed=seed,
    ))[0].stencil
    return ThermalStageStencil(
        baseline=sampled.baseline,
        variants={"i_plus": sampled.variants["i_plus"]},
        evidence_manifest={
            "response_source": "original_11_state_train_atlas_subset",
            "original_atlas_variant_count": len(stencil.variants),
            "stage_b_labels": ["baseline", "i_plus"],
            "targets_used_for_train_query_sampling": True,
        },
    )


def _record_design_for_call(
    row: Mapping[str, Any],
    base_design: DesignState,
    *,
    family_id: str,
) -> DesignState:
    centers = np.asarray(row["module_centers_xy"], dtype=np.float64)
    heating = np.asarray(row["heating"], dtype=np.float64)
    active_indices = [index for index, module in enumerate(base_design.modules) if module.active]
    if centers.shape != (len(active_indices), 2) or heating.shape != (len(active_indices),):
        raise ValueError(f"Coverage design does not align with packed module slots for {family_id}.")
    modules = list(base_design.modules)
    for active_row, slot in enumerate(active_indices):
        modules[slot] = replace(
            modules[slot],
            position_xy=(float(centers[active_row, 0]), float(centers[active_row, 1])),
            heating=float(heating[active_row]),
        )
    return DesignState(
        anchor_id=base_design.anchor_id,
        physical_family_id=family_id,
        split=EvidenceSplit.TRAIN,
        modules=tuple(modules),
    )


def _load_coverage_record(
    row: Mapping[str, Any],
    *,
    adapter: AnalyticWakeReferenceAdapter,
    base_design: DesignState,
    family_id: str,
    context: OperatingContext,
) -> SolveRecord:
    if row.get("status") != "converged" or not row.get("case_dir"):
        raise RuntimeError(f"Coverage response is not a completed local reference record: {row.get('record_id')}")
    record_path = Path(str(row["record_npz"]))
    if not record_path.is_file() or _sha256(record_path) != row.get("record_npz_sha256"):
        raise RuntimeError(f"Coverage record identity/hash failed: {record_path}")
    case_dir = Path(str(row["case_dir"]))
    if not case_dir.is_dir():
        raise FileNotFoundError(f"The raw local reference output is unavailable: {case_dir}")
    design = _record_design_for_call(row, base_design, family_id=family_id)
    elapsed = row.get("elapsed_seconds")
    record = adapter.load_record(
        case_dir,
        design,
        context,
        record_id=str(row["record_id"]),
        elapsed_seconds=0.0 if elapsed is None else float(elapsed),
        baseline_design=base_design if row.get("state") == "i_plus" else None,
        physical_step_scales={"module_x": 0.10} if row.get("state") == "i_plus" else {},
    )
    if record.status is not SolveStatus.CONVERGED:
        raise RuntimeError(f"Raw local reference parse failed: {record.failure_reason}")
    provenance = dict(record.provenance)
    provenance.update({
        "coverage_attempt_id": row.get("attempt_id"),
        "coverage_record_npz_sha256": row.get("record_npz_sha256"),
        "physical_wall_time_available": elapsed is not None,
        "elapsed_seconds_if_available": None if elapsed is None else float(elapsed),
        "solver_invoked": bool(row.get("solver_invoked", row.get("provenance", {}).get("solver_invoked", False))),
    })
    return replace(record, provenance=provenance)


def _new_coverage_pair(
    family_spec: Mapping[str, Any],
    *,
    calls: Mapping[str, Any],
    contexts: Mapping[str, Any],
    hdf5_path: Path = DEFAULT_HDF5,
    demo_root: Path = DEFAULT_DEMO,
    coverage_root: Path = DEFAULT_COVERAGE_ROOT,
) -> ThermalStageStencil:
    family_id = str(family_spec["case_id"])
    case_token = family_id.split(":")[1]
    case_id = case_token.removeprefix("case")
    context_id = str(family_spec["context_id"])
    context = OperatingContext(contexts[context_id])
    config, _ = read_embedded_case_config(hdf5_path, case_id)
    adapter = AnalyticWakeReferenceAdapter(
        demo_root=demo_root,
        output_root=coverage_root / "typed_record_rebuild_only",
        case_template=config,
    )
    packed = load_stored_reference_case(
        hdf5_path,
        case_id,
        physical_family_id=family_id,
        split_override=EvidenceSplit.TRAIN,
    )
    packed = replace(packed, context=context)
    baseline_source = str(family_spec["baseline_source"])
    baseline = packed
    if baseline_source == "new_local_reference_call":
        call_id = f"thermal_coverage_case{case_id}_Re{context_id[2:]}_baseline"
        baseline_row = calls.get(call_id)
        if not isinstance(baseline_row, Mapping):
            raise FileNotFoundError(f"The predeclared new baseline is still unopened: {call_id}")
        baseline = _load_coverage_record(
            baseline_row,
            adapter=adapter,
            base_design=packed.design,
            family_id=family_id,
            context=context,
        )
    variant_id = f"thermal_coverage_case{case_id}_Re{context_id[2:]}_i_plus"
    variant_row = calls.get(variant_id)
    if not isinstance(variant_row, Mapping):
        raise FileNotFoundError(f"The predeclared i_plus response is still unopened: {variant_id}")
    variant = _load_coverage_record(
        variant_row,
        adapter=adapter,
        base_design=packed.design,
        family_id=family_id,
        context=context,
    )
    return ThermalStageStencil(
        baseline=baseline,
        variants={"i_plus": variant},
        evidence_manifest={
            "family_id": family_id,
            "case_id": case_id,
            "context_id": context_id,
            "baseline_source": baseline.source.value,
            "variant_source": variant.source.value,
            "baseline_record_id": baseline.record_id,
            "variant_record_id": variant.record_id,
            "baseline_record_sha256": (
                baseline_row.get("record_npz_sha256") if baseline_source == "new_local_reference_call" else None
            ),
            "variant_record_sha256": variant_row.get("record_npz_sha256"),
            "baseline_wall_seconds": (
                baseline_row.get("elapsed_seconds") if baseline_source == "new_local_reference_call" else None
            ),
            "variant_wall_seconds": variant_row.get("elapsed_seconds"),
            "baseline_physical_wall_time_available": (
                baseline_row.get("elapsed_seconds") is not None if baseline_source == "new_local_reference_call" else None
            ),
            "variant_physical_wall_time_available": variant_row.get("elapsed_seconds") is not None,
            "baseline_coverage_attempt_id": (
                baseline_row.get("attempt_id") if baseline_source == "new_local_reference_call" else None
            ),
            "variant_coverage_attempt_id": variant_row.get("attempt_id"),
            "response_source": "new_local_analytic_wake_i_plus_attempt",
            "targets_used_for_train_query_sampling": True,
        },
    )


def _sample_coverage_pair(
    stencil: ThermalStageStencil,
    *,
    config: ReceiverSamplingConfig,
    target_free: bool,
) -> ThermalStageStencil:
    """Sample a mixed-source pair while keeping held query selection target-free."""

    rng = np.random.default_rng(int(config.random_seed) ^ zlib.crc32(stencil.physical_family_id.encode("utf-8")))
    roles = stencil.baseline.output.roles  # type: ignore[union-attr]
    fluid = roles["fluid_fields"]
    length_x = float(stencil.baseline.context.values["domain_length_x"])
    x = fluid.query_features[:, 0]
    pressure_bands = np.flatnonzero((x <= 0.08 * length_x) | (x >= 0.92 * length_x))
    protected_near = _sampling._protected_near_interface_fluid_indices(stencil)
    protected_fluid = np.union1d(pressure_bands, protected_near)
    fluid_indices, fluid_probability, _ = _sampling._sample_rows(
        fluid.values.shape[0], protected_fluid, config.max_fluid_queries, rng
    )
    solid = roles["solid_temperature"]
    if solid.receiver_module_ids is None:
        raise ValueError("Thermal solid material points must retain module identity/order.")
    modules = tuple(dict.fromkeys(solid.receiver_module_ids))
    local_count = int(sum(module_id == modules[0] for module_id in solid.receiver_module_ids))
    if target_free or config.hot_solid_points_per_module == 0:
        protected_local = np.zeros((0,), dtype=np.int64)
    else:
        protected_local = _sampling._protected_solid_local_indices(stencil, config.hot_solid_points_per_module)
    local_indices, local_probability, _ = _sampling._sample_rows(
        local_count, protected_local, config.solid_queries_per_module, rng
    )
    module_rows = {
        module_id: np.asarray([i for i, current in enumerate(solid.receiver_module_ids) if current == module_id], dtype=np.int64)
        for module_id in modules
    }
    if any(rows.size != local_count for rows in module_rows.values()):
        raise ValueError("Each active Thermal module needs an aligned material-local solid panel.")
    solid_indices_unsorted = np.concatenate([module_rows[module_id][local_indices] for module_id in modules])
    solid_probability_unsorted = np.tile(local_probability, len(modules))
    order = np.argsort(solid_indices_unsorted, kind="stable")
    solid_indices = solid_indices_unsorted[order]
    solid_probability = solid_probability_unsorted[order]
    interface = roles["interface"]
    selections = {
        "fluid_fields": (fluid_indices, fluid_probability),
        "interface": (np.arange(interface.values.shape[0], dtype=np.int64), np.ones(interface.values.shape[0], dtype=np.float64)),
        "solid_temperature": (solid_indices, solid_probability),
    }
    sampled_records = []
    for record in stencil.records:
        sampled_roles = {
            name: _sampling._subsample_role(record.output.roles[name], *selections[name])  # type: ignore[union-attr]
            for name in record.output.roles  # type: ignore[union-attr]
        }
        sampled_records.append(_sampling._replace_record_roles(record, sampled_roles))
    return ThermalStageStencil(
        baseline=sampled_records[0],
        variants=dict(zip(stencil.variants, sampled_records[1:], strict=True)),
        evidence_manifest={
            **dict(stencil.evidence_manifest),
            "query_sampling": "input_geometry_only" if target_free else "train_only_peak_protected",
            "sampled_role_counts": {name: int(value[0].size) for name, value in selections.items()},
            "inverse_probability_weighting": True,
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Refusing to replace immutable evidence: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _load_stencils(atlas_dir: Path, *, seed: int = 2317):
    result = []
    for family_id in forward.TRAIN_ATLAS_IDS:
        path = atlas_dir / f"train_{family_id}_responses.npz"
        stencil, _ = load_response_atlas_stencil(path)
        if stencil.split.value != "train":
            raise ValueError(f"Stage B/C refuses non-train response evidence: {path}")
        result.append(_pair_from_atlas(stencil, seed=seed))
    if len({item.physical_family_id for item in result}) != len(forward.TRAIN_ATLAS_IDS):
        raise RuntimeError("The eight original train response families are not distinct.")
    return tuple(result)


def _load_fit_stencils(
    atlas_dir: Path,
    *,
    seed: int,
    calls_path: Path = DEFAULT_COVERAGE_MANIFEST,
    predeclared_path: Path = DEFAULT_PREDECLARED,
) -> tuple[ThermalStageStencil, ...]:
    """Return the 8 original atlases plus the 2 locked supplemental fit pairs."""

    atlas = _load_stencils(atlas_dir, seed=seed)
    additions = _load_coverage_families(
        calls_path=calls_path,
        predeclared_path=predeclared_path,
        partition="fit",
        sampling_seed=seed + 101,
    )
    expected = [
        "active_packet:case0319:Re70",
        "active_packet:case0349:Re50",
    ]
    if [item.physical_family_id for item in additions] != expected:
        raise ValueError("Supplemental Stage-B fit families differ from the predeclared order.")
    combined = (*atlas, *additions)
    if len(combined) != 10 or len({item.physical_family_id for item in combined}) != 10:
        raise RuntimeError("Stage B requires 10 distinct fit families: 8 atlas + 2 predeclared pairs.")
    if any(item.split.value != "train" for item in combined):
        raise ValueError("Stage B/C only accepts declared train-family response evidence.")
    return tuple(combined)


def _load_coverage_families(
    *,
    calls_path: Path = DEFAULT_COVERAGE_MANIFEST,
    predeclared_path: Path = DEFAULT_PREDECLARED,
    partition: str,
    target_free_sampling: bool = False,
    sampling_seed: int = 2317,
) -> tuple[ThermalStageStencil, ...]:
    calls_payload = json.loads(calls_path.read_text(encoding="utf-8"))
    plan = json.loads(predeclared_path.read_text(encoding="utf-8"))
    if calls_payload.get("dataset_sha256") != plan["packed_baseline_parity_evidence"]["source_dataset_sha256"]:
        raise ValueError("Coverage calls and predeclared train families use different packed datasets.")
    specs = [item for item in plan["new_train_families"] if item["utility_partition"] == partition]
    expected = 2 if partition in {"fit", "held_out_train_development"} else None
    if expected is not None and len(specs) != expected:
        raise ValueError(f"Expected two predeclared {partition} Thermal coverage families, got {len(specs)}.")
    config = ReceiverSamplingConfig(
        max_fluid_queries=128,
        solid_queries_per_module=8,
        hot_solid_points_per_module=0 if target_free_sampling else 4,
        random_seed=sampling_seed,
    )
    return tuple(
        _sample_coverage_pair(
            _new_coverage_pair(
                spec,
                calls=calls_payload["calls"],
                contexts=plan["contexts"],
            ),
            config=config,
            target_free=target_free_sampling,
        )
        for spec in specs
    )


def _held_input_records(
    *,
    predeclared_path: Path = DEFAULT_PREDECLARED,
    hdf5_path: Path = DEFAULT_HDF5,
) -> tuple[tuple[Mapping[str, Any], SolveRecord], ...]:
    """Build target-free held candidate inputs from packed train geometry only."""

    plan = json.loads(predeclared_path.read_text(encoding="utf-8"))
    held = [item for item in plan["new_train_families"] if item["utility_partition"] == "held_out_train_development"]
    records: list[tuple[Mapping[str, Any], SolveRecord]] = []
    for spec in held:
        family_id = str(spec["case_id"])
        case_id = family_id.split(":")[1].removeprefix("case")
        baseline = load_stored_reference_case(
            hdf5_path,
            case_id,
            physical_family_id=family_id,
            split_override=EvidenceSplit.TRAIN,
        )
        record = replace(baseline, context=OperatingContext(spec["context"]))
        records.append((spec, record))
    return tuple(records)


def _sample_input_candidate(
    record: SolveRecord,
    *,
    seed: int,
    max_fluid_queries: int = 128,
    solid_queries_per_module: int = 8,
) -> ThermalInputCandidate:
    """Subsample held receivers using only stored coordinates/masks/weights."""

    if record.output is None:
        raise ValueError("A held candidate needs the stored geometry/receiver schema.")
    rng = np.random.default_rng(int(seed) ^ zlib.crc32(record.design.physical_family_id.encode("utf-8")))
    roles = record.output.roles
    fluid = roles["fluid_fields"]
    x = np.asarray(fluid.query_features[:, 0])
    length_x = float(record.context.values["domain_length_x"])
    pressure_bands = np.flatnonzero((x <= 0.08 * length_x) | (x >= 0.92 * length_x))
    fluid_indices, fluid_probability, _ = _sampling._sample_rows(
        fluid.values.shape[0], pressure_bands, max_fluid_queries, rng
    )
    solid = roles["solid_temperature"]
    if solid.receiver_module_ids is None:
        raise ValueError("Held Thermal solid receivers need module identity for geometry sampling.")
    modules = tuple(dict.fromkeys(solid.receiver_module_ids))
    per_module = {
        module_id: np.asarray([i for i, current in enumerate(solid.receiver_module_ids) if current == module_id], dtype=np.int64)
        for module_id in modules
    }
    counts = {indices.size for indices in per_module.values()}
    if len(counts) != 1:
        raise ValueError("Held Thermal module material panels must have equal row counts.")
    local_count = counts.pop()
    local_indices, local_probability, _ = _sampling._sample_rows(
        local_count, np.zeros((0,), dtype=np.int64), solid_queries_per_module, rng
    )
    solid_indices_unsorted = np.concatenate([per_module[module_id][local_indices] for module_id in modules])
    solid_probability_unsorted = np.tile(local_probability, len(modules))
    order = np.argsort(solid_indices_unsorted, kind="stable")
    selections = {
        "fluid_fields": (fluid_indices, fluid_probability),
        "interface": (
            np.arange(roles["interface"].values.shape[0], dtype=np.int64),
            np.ones(roles["interface"].values.shape[0], dtype=np.float64),
        ),
        "solid_temperature": (solid_indices_unsorted[order], solid_probability_unsorted[order]),
    }
    sampled_roles = {}
    for name, role in roles.items():
        sampled = _sampling._subsample_role(role, *selections[name])
        sampled_roles[name] = replace(sampled, values=np.zeros_like(sampled.values))
    input_output = PhysicalSolveOutput(
        roles=sampled_roles,
        quantities={"pressure_drop": MeasuredQuantity(0.0, "stripped placeholder", resolved=False)},
        active_module_ids=record.design.active_module_ids,
        module_peak_temperature={module_id: 0.0 for module_id in record.design.active_module_ids},
        units_metadata=record.output.units_metadata,
        case_dir=None,
    )
    sampled_record = replace(
        record,
        record_id=f"input_only:{record.design.physical_family_id}",
        output=input_output,
        provenance={"input_only_candidate": True, "targets_stripped": True},
    )
    return ThermalInputCandidate(
        baseline=sampled_record,
        variants={},
        evidence_manifest={
            "query_sampling": "geometry_masks_weights_only",
            "sampled_role_counts": {name: int(value[0].size) for name, value in selections.items()},
            "held_targets_opened": False,
        },
    )


def _role_rows(stencil: Any) -> list[dict[str, Any]]:
    absolute: list[dict[str, Any]] = []
    finite: list[dict[str, Any]] = []
    for role_name, role in stencil.baseline.output.roles.items():
        for channel, (channel_name, units) in enumerate(zip(role.channel_names, role.channel_units, strict=True)):
            absolute.append({
                "name": f"{role_name}:{channel_name}:absolute",
                "role": role_name,
                "channel": channel,
                "channel_name": channel_name,
                "units": units,
                "measure": "absolute_value_rmse_over_baseline_and_response_states",
            })
            finite.append({
                "name": f"{role_name}:{channel_name}:finite_response",
                "role": role_name,
                "channel": channel,
                "channel_name": channel_name,
                "units": units,
                "measure": "finite_change_rmse_over_nonbaseline_response_states",
            })
    return absolute + finite


def _weighted_rmse(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(ref)
    w = np.asarray(weights, dtype=np.float64) * valid
    denominator = float(w.sum())
    if denominator <= 0.0:
        raise ValueError("A measured Thermal role/channel has no weighted valid samples.")
    delta = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return float(np.sqrt(np.sum(w * np.square(delta)) / denominator))


def _weighted_rms(value: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    return _weighted_rmse(value, np.zeros_like(value), mask, weights)


def _case_metrics(predictions: StencilPredictions, stencil: Any) -> tuple[list[float], list[float], list[float]]:
    """Return physical errors, reference RMS scales, and unweighted role errors."""
    records = stencil.records
    labels = ("baseline", *tuple(stencil.variants))
    absolute_errors: list[float] = []
    finite_errors: list[float] = []
    absolute_scales: list[float] = []
    finite_scales: list[float] = []
    unweighted_errors: list[float] = []
    for role_name, reference_role in stencil.baseline.output.roles.items():
        for channel in range(len(reference_role.channel_names)):
            state_mse: list[float] = []
            target_rms: list[float] = []
            for label, record in zip(labels, records, strict=True):
                ref_role = record.output.roles[role_name]
                pred = predictions.values[label].role_values[role_name][:, channel].detach().cpu().numpy()
                ref = np.asarray(ref_role.values[:, channel])
                mask = np.asarray(ref_role.valid_mask[:, channel])
                weights = np.asarray(ref_role.quadrature_weights)
                err = _weighted_rmse(pred, ref, mask, weights)
                state_mse.append(err * err)
                target_rms.append(_weighted_rms(ref, mask, weights))
            absolute_errors.append(float(np.sqrt(np.mean(state_mse))))
            absolute_scales.append(float(np.sqrt(np.mean(np.square(target_rms)))))
            unweighted_errors.append(absolute_errors[-1])

            delta_mse: list[float] = []
            delta_rms: list[float] = []
            base_role = stencil.baseline.output.roles[role_name]
            base_pred = predictions.values["baseline"].role_values[role_name][:, channel].detach().cpu().numpy()
            for label, record in zip(labels[1:], records[1:], strict=True):
                ref_role = record.output.roles[role_name]
                pred = predictions.values[label].role_values[role_name][:, channel].detach().cpu().numpy()
                ref_delta = np.asarray(ref_role.values[:, channel]) - np.asarray(base_role.values[:, channel])
                pred_delta = pred - base_pred
                mask = np.asarray(base_role.valid_mask[:, channel]) & np.asarray(ref_role.valid_mask[:, channel])
                weights = np.asarray(base_role.quadrature_weights)
                err = _weighted_rmse(pred_delta, ref_delta, mask, weights)
                delta_mse.append(err * err)
                delta_rms.append(_weighted_rms(ref_delta, mask, weights))
            finite_errors.append(float(np.sqrt(np.mean(delta_mse))))
            finite_scales.append(float(np.sqrt(np.mean(np.square(delta_rms)))))
            unweighted_errors.append(finite_errors[-1])
    return absolute_errors + finite_errors, absolute_scales + finite_scales, unweighted_errors


def _parity_metrics(left: StencilPredictions, right: StencilPredictions, stencil: Any) -> list[float]:
    """Channelwise physical-unit RMSE between two native chunk partitions."""
    records = stencil.records
    labels = ("baseline", *tuple(stencil.variants))
    absolute_errors: list[float] = []
    finite_errors: list[float] = []
    for role_name, reference_role in stencil.baseline.output.roles.items():
        for channel in range(len(reference_role.channel_names)):
            state_mse = []
            for label, record in zip(labels, records, strict=True):
                role = record.output.roles[role_name]
                left_value = left.values[label].role_values[role_name][:, channel].detach().cpu().numpy()
                right_value = right.values[label].role_values[role_name][:, channel].detach().cpu().numpy()
                mask = np.asarray(role.valid_mask[:, channel])
                state_mse.append(
                    _weighted_rmse(left_value, right_value, mask, np.asarray(role.quadrature_weights)) ** 2
                )
            absolute_errors.append(float(np.sqrt(np.mean(state_mse))))

            base_role = stencil.baseline.output.roles[role_name]
            left_base = left.values["baseline"].role_values[role_name][:, channel].detach().cpu().numpy()
            right_base = right.values["baseline"].role_values[role_name][:, channel].detach().cpu().numpy()
            change_mse = []
            for label, record in zip(labels[1:], records[1:], strict=True):
                role = record.output.roles[role_name]
                left_value = left.values[label].role_values[role_name][:, channel].detach().cpu().numpy() - left_base
                right_value = right.values[label].role_values[role_name][:, channel].detach().cpu().numpy() - right_base
                mask = np.asarray(base_role.valid_mask[:, channel]) & np.asarray(role.valid_mask[:, channel])
                change_mse.append(
                    _weighted_rmse(left_value, right_value, mask, np.asarray(base_role.quadrature_weights)) ** 2
                )
            finite_errors.append(float(np.sqrt(np.mean(change_mse))))
    return absolute_errors + finite_errors


def _predict_stencil(
    operator: DifferentiableThermalOperator,
    stencil: Any,
    *,
    device: torch.device,
    cover_plan_builder: Any | None = None,
    capture_baseline: bool = False,
    capture_plan_states: bool = False,
) -> tuple[StencilPredictions, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    queries = role_queries_from_stencil(stencil, device=device)
    values: dict[str, Any] = {}
    capture = None
    state_runtime_diagnostics: dict[str, Mapping[str, Any]] = {}
    if cover_plan_builder is not None and capture_plan_states:
        cover_plan_builder.last_state_captures = {}
    baseline = stencil.baseline
    def predict_record(label: str, record: SolveRecord):
        attempt_id = _begin_native_state_forward(stencil.physical_family_id, label, device)
        try:
            prediction = operator(
                DesignInput.from_state(record.design, device=device),
                context_inputs(record.context),
                queries,
                cover_plan_builder=cover_plan_builder,
            )
        except BaseException as exc:
            _end_native_state_forward(attempt_id, "failed", exc)
            if isinstance(exc, ValueError) and (
                "frontier path" in str(exc) or "mapped frontier paths" in str(exc)
            ):
                raise UnsupportedFrontierPatternError(str(label), str(exc)) from exc
            raise
        _end_native_state_forward(attempt_id, "completed")
        runtime_diagnostics = getattr(operator, "_active_packet_latest_diagnostics", None)
        if isinstance(runtime_diagnostics, Mapping):
            state_runtime_diagnostics[label] = dict(runtime_diagnostics)
        if cover_plan_builder is not None and capture_plan_states:
            base_builder = getattr(cover_plan_builder, "base", cover_plan_builder)
            plans = tuple(getattr(cover_plan_builder, "last_plans", ()))
            encoded = getattr(base_builder, "last_encoded", None)
            trees = tuple(getattr(base_builder, "last_trees", ()))
            records = tuple(getattr(cover_plan_builder, "last_records", ()))
            if not plans or encoded is None or not trees:
                raise RuntimeError("Native plan artifact capture is missing the applied plan or encoded input.")
            cover_plan_builder.last_state_captures[label] = {
                "plans": plans,
                "encoded": encoded,
                "trees": trees,
                "records": records,
                "runtime_diagnostics": dict(runtime_diagnostics or {}),
            }
        return prediction

    values["baseline"] = predict_record("baseline", baseline)
    if capture_baseline:
        capture = operator.last_packet_inputs
        if not isinstance(capture, Mapping) or not {"encoded", "trees", "p0_module_states"}.issubset(capture):
            raise RuntimeError("Native Dense capture omitted the encoded tree or P0 state.")
    baseline_work = None
    if cover_plan_builder is not None:
        records = getattr(cover_plan_builder, "last_records", ())
        if records:
            baseline_work = records[0]
    for label, record in stencil.variants.items():
        values[label] = predict_record(str(label), record)
    if capture is not None and state_runtime_diagnostics:
        capture = dict(capture)
        capture["state_runtime_diagnostics"] = state_runtime_diagnostics
    return StencilPredictions(values, queries), capture, baseline_work


def _make_model_and_route(
    arm: str,
    *,
    source_checkpoint: Path,
    arm_checkpoint_path: Path,
    template: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    normalization_stats: Mapping[str, Any],
    stencil: Any,
    device: torch.device,
    forward_update: int,
):
    model, raw_checkpoint = load_model(source_checkpoint, device)
    if int(raw_checkpoint.get("epoch", raw_checkpoint.get("current_epoch", -1))) != 4738:
        raise ValueError("Stage B/C must reload the exact Run1804 epoch-4738 Dense checkpoint.")
    model.eval()
    _configure_native_expanded_response_interface_scope(model)
    setup = DifferentiableThermalOperator(
        model,
        template,
        dataset_config=dataset_config,
        normalization_stats=normalization_stats,
        query_batch_size=512,
        capture_packet_inputs=True,
    )
    with torch.no_grad():
        _, capture, _ = _predict_stencil(
            setup,
            stencil,
            device=device,
            capture_baseline=True,
        )
    assert capture is not None
    encoded = capture["encoded"]
    route = forward._route_module(arm, model.core, encoded, device)
    saved = torch.load(arm_checkpoint_path, map_location="cpu", weights_only=False)
    if "physical_model_state" not in saved or "route_model_state" not in saved:
        # A bounded review gate pauses with the exact resumable training
        # checkpoint. Recover its two inference states without touching its
        # optimizer/RNG fields or changing the training artifact.
        bundle_state = saved.get("model")
        if not isinstance(bundle_state, Mapping):
            raise ValueError(f"Checkpoint has neither inference nor resumable model state: {arm_checkpoint_path}")
        prefix = {"forward_model.": "physical_model_state", "route_model.": "route_model_state"}
        recovered: dict[str, dict[str, torch.Tensor]] = {name: {} for name in prefix.values()}
        for key, value in bundle_state.items():
            for name, destination in prefix.items():
                if key.startswith(name):
                    recovered[destination][key[len(name):]] = value
                    break
        if not all(recovered.values()):
            raise ValueError(f"Could not recover both student and route weights from {arm_checkpoint_path}")
        run_manifest_path = arm_checkpoint_path.parents[1] / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        saved = {
            **saved,
            "arm": arm,
            **recovered,
            "source_checkpoint_sha256": run_manifest["checkpoint_sha256"],
        }
    if saved.get("arm") != arm or int(saved.get("actual_optimizer_updates", -1)) != forward_update:
        raise ValueError(
            f"{arm} artifact is not the exact Run1509 u{forward_update} endpoint: {arm_checkpoint_path}"
        )
    if saved.get("source_checkpoint_sha256") != _sha256(source_checkpoint):
        raise ValueError(f"{arm} checkpoint source hash differs from the frozen e4738 checkpoint.")
    incompatible = route.load_state_dict(saved["route_model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Could not load exact {arm} route state: {incompatible}")
    model.load_state_dict(saved["physical_model_state"], strict=True)
    model.eval().requires_grad_(False)
    route.eval().requires_grad_(False)
    operator = DifferentiableThermalOperator(
        model,
        template,
        dataset_config=dataset_config,
        normalization_stats=normalization_stats,
        query_batch_size=512,
        capture_packet_inputs=False,
    )
    return model, route, operator, saved


def _budgets(extra_route: str) -> dict[str, float]:
    return {"QE": 0.90, extra_route: 0.90}


def _builder(
    arm: str,
    model: Any,
    route: Any,
    extra_route: str,
    cut: Sequence[int],
    *,
    budget_fractions: Mapping[str, float] | None = None,
    frontier_pattern_paths: Sequence[str] | None = None,
):
    selected = tuple(int(value) for value in cut)
    if frontier_pattern_paths is None:
        selector = lambda case, encoded, module_state, tree: selected
    else:
        paths = tuple(str(value) for value in frontier_pattern_paths)

        def selector(case: int, encoded: Any, module_state: torch.Tensor, tree: Any) -> tuple[int, ...]:
            mapped = frontier_from_paths(tree, paths, max_depth=3)
            if len(mapped) != len(selected):
                raise ValueError("mapped frontier paths changed the declared cut size")
            return mapped

    builder = ThermalCoverPlanBuilder(
        core=model.core,
        budget_fractions=dict(budget_fractions or _budgets(extra_route)),
        mode=arm,
        organizer=route if arm == "G" else None,
        direct_scorer=route if arm == "P" else None,
        extra_route=extra_route,
        frontier_selector=selector,
    )
    builder.hard = True
    return builder


def _cut_work(record: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    routes = record.get("routes", {})
    copied: dict[str, Any] = {}
    for mechanism, value in routes.items():
        item = asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value)
        copied[mechanism] = item
    total = record.get("all_mechanisms_total")
    all_routes = record.get("all_mechanism_route_work")
    if not isinstance(total, Mapping) or not isinstance(all_routes, Mapping):
        raise RuntimeError("Stage B/C needs summed canonical work across all five typed routes.")
    copied["all_mechanism_route_work"] = dict(all_routes)
    copied["all_mechanisms_total"] = dict(total)
    full = float(total["full_access_work"])
    fraction = float(total["achieved_work"]) / full if full > 0.0 else 0.0
    return copied, fraction


def _full_plan_predictions(
    operator: DifferentiableThermalOperator,
    stencil: Any,
    device: torch.device,
    query_batch_size: int,
    *,
    capture_baseline: bool = False,
):
    operator.query_batch_size = int(query_batch_size)
    prior_capture = operator.capture_packet_inputs
    operator.capture_packet_inputs = bool(capture_baseline)
    try:
        with torch.no_grad():
            prediction, capture, _ = _predict_stencil(
                operator, stencil, device=device, capture_baseline=capture_baseline
            )
    finally:
        operator.capture_packet_inputs = prior_capture
    return prediction, capture


def _organizer_scores(organizer: InputOnlyCoverOrganizer, capture: Mapping[str, Any], budgets: Mapping[str, float]):
    encoded = capture["encoded"]
    trees = tuple(capture["trees"])
    state = capture["p0_module_states"]
    inputs = {
        "module_states": state,
        "environment_states": encoded.env_tokens,
        "global_state": encoded.global_token,
    }
    return organizer.score_cases(encoded, inputs, trees, budgets=budgets)


def _fit_utility(
    organizer: InputOnlyCoverOrganizer,
    cases: Sequence[dict[str, Any]],
    *,
    role_scales: torch.Tensor,
    updates: int,
    seed: int,
    output_path: Path,
    device: torch.device,
) -> list[dict[str, float]]:
    train_cases = list(cases)
    if not train_cases:
        raise ValueError("Stage C utility fitting requires at least one train-only measured family.")
    for name, parameter in organizer.named_parameters():
        parameter.requires_grad_(name.startswith("frontier_utility_head."))
    if organizer.frontier_utility_head is None:
        raise RuntimeError("The Stage C utility head was not constructed.")
    optimizer = torch.optim.AdamW(organizer.frontier_utility_head.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    rng = random.Random(seed)
    order = list(range(len(train_cases)))
    rng.shuffle(order)
    cursor = 0
    history: list[dict[str, float]] = []
    for update in range(1, updates + 1):
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        row = train_cases[order[cursor]]
        cursor += 1
        organizer.zero_grad(set_to_none=True)
        score = row["organizer_scores"][0]
        tree = row["capture"]["trees"][0]
        prediction = organizer.score_frontiers(
            score,
            tree,
            cuts=row["cuts"],
            budget_vector=score.budget_vector,
        )
        error_targets = torch.as_tensor(row["role_errors"], device=device, dtype=torch.float32)
        full_errors = torch.as_tensor(row["full_role_errors"], device=device, dtype=torch.float32)
        distortion = normalized_frontier_distortion_targets(
            error_targets,
            full_errors,
            role_scales=role_scales.to(device=device, dtype=torch.float32),
        )
        work = normalized_frontier_work_targets(
            torch.as_tensor(row["work_fraction"], device=device, dtype=torch.float32),
            1.0,
        )
        distortion_loss = F.smooth_l1_loss(prediction.role_distortion, distortion)
        work_loss = F.smooth_l1_loss(prediction.predicted_work_fraction, work)
        total = distortion_loss + work_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"Stage C utility loss became nonfinite at update {update}.")
        total.backward()
        grad_terms = [p.grad.detach().float().norm() for p in organizer.frontier_utility_head.parameters() if p.grad is not None]
        grad = float(torch.stack(grad_terms).norm().detach().cpu()) if grad_terms else 0.0
        if grad <= 0.0:
            raise RuntimeError(f"Stage C utility head received no gradient at update {update}.")
        torch.nn.utils.clip_grad_norm_(organizer.frontier_utility_head.parameters(), 5.0)
        optimizer.step()
        history.append({
            "update": float(update),
            "case_id": row["case_id"],
            "total_loss": float(total.detach().cpu()),
            "distortion_loss": float(distortion_loss.detach().cpu()),
            "work_loss": float(work_loss.detach().cpu()),
            "utility_gradient_l2": grad,
        })
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(history[-1], sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return history


def _canonical_plan_route_work(encoded: Any, tree: Any, plan: Any, mechanism: str) -> dict[str, Any]:
    from honf_forward_core.interface_fields.budgeted_frontier import canonical_pair_catalog

    catalog = canonical_pair_catalog(encoded, tree, mechanism, case_index=0)
    access = plan.access_for(
        mechanism,
        catalog.receiver_coordinates,
        module_present=encoded.module_present[0],
    )
    selected = (access.detach() > 0) & catalog.pair_validity
    weights = catalog.receiver_weights.to(device=selected.device, dtype=torch.float64)[:, None]
    weighted = float((selected.to(torch.float64) * weights).sum().cpu())
    full = float((catalog.pair_validity.to(torch.float64) * weights).sum().cpu())
    selected_count = int(selected.sum().cpu())
    full_count = int(catalog.pair_validity.sum().cpu())
    return {
        "achieved_work": weighted,
        "full_access_work": full,
        "selected_unique_pairs": selected_count,
        "full_unique_pairs": full_count,
        "work_fraction": weighted / full if full > 0.0 else 0.0,
        "sparse_success": bool(weighted > 0.0 and selected_count > 0 and weighted < full),
    }


def _geometry_only_plan(encoded: Any, plan: Any, mechanism: str, target_work: float) -> tuple[Any, dict[str, Any]]:
    """Replace learned packet memberships with nearest-source geometry at matched work."""

    from honf_forward_core.interface_fields.budgeted_frontier import canonical_pair_catalog

    tree = plan.tree
    case = 0
    catalog = canonical_pair_catalog(encoded, tree, mechanism, case_index=case)
    prior = plan.permission_matrix(mechanism, module_present=encoded.module_present[case]).detach()
    if not torch.all((prior == 0) | (prior == 1)):
        raise ValueError("Geometry-only control requires the selected hard G membership.")
    active = torch.nonzero(
        (tree.access(tree.universe.coordinates, plan.split_gates.detach()) > 0).any(dim=0),
        as_tuple=False,
    ).flatten()
    module_routes = {"MM", "EM", "QM"}
    source_xy = encoded.module_centers[case] if mechanism in module_routes else encoded.env_coords[case]
    source_valid = catalog.source_validity.to(device=source_xy.device, dtype=torch.bool)
    row_geometry: list[tuple[int, torch.Tensor]] = []
    row_sizes = prior.sum(dim=1).to(torch.int64)
    for row in active.detach().cpu().tolist():
        anchor_ids = tree.nodes[row].anchor_indices
        anchor_xy = tree.universe.coordinates[list(anchor_ids)]
        center = anchor_xy.mean(dim=0)
        distance = torch.linalg.vector_norm(source_xy - center[None, :], dim=-1)
        distance = distance.masked_fill(~source_valid, torch.inf)
        row_geometry.append((int(row), torch.argsort(distance, stable=True)))

    def candidate(scale: float) -> torch.Tensor:
        value = torch.zeros_like(prior)
        for row, source_order in row_geometry:
            count = int(round(float(row_sizes[row].item()) * scale))
            count = min(count, int(source_valid.sum().item()))
            if count:
                value[row, source_order[:count]] = 1.0
        return value

    best_plan = plan
    best_record = _canonical_plan_route_work(encoded, tree, plan, mechanism)
    best_error = abs(best_record["achieved_work"] - target_work)
    best_edges = int(prior.sum().item())
    selected_scale = 1.0
    for scale in np.linspace(0.0, 2.0, 81):
        weights = candidate(float(scale))
        proposed = plan.with_permission(mechanism, weights)
        row = _canonical_plan_route_work(encoded, tree, proposed, mechanism)
        error = abs(row["achieved_work"] - target_work)
        edges = int(weights.sum().item())
        if (error, edges) < (best_error, best_edges):
            best_plan, best_record = proposed, row
            best_error, best_edges, selected_scale = error, edges, float(scale)
    return best_plan, {
        **best_record,
        "geometry_rule": "nearest_source_to_packet_anchor_centroid",
        "matched_target_canonical_work": float(target_work),
        "absolute_work_gap": float(best_error),
        "row_size_scale": selected_scale,
    }


def _rewire_membership(plan: Any, mechanism: str, seed: int) -> tuple[Any, dict[str, Any]]:
    """2x2 switch typed packet-source memberships, preserving row/column degrees."""

    membership = plan.permission_matrix(mechanism).detach().to(torch.int64).cpu().numpy().copy()
    active = (
        plan.tree.access(plan.tree.universe.coordinates, plan.split_gates.detach()) > 0
    ).any(dim=0).detach().cpu().numpy()
    rows = np.flatnonzero(active)
    sub = membership[rows]
    rng = np.random.default_rng(seed)
    edge_count = int(sub.sum())
    attempt_limit = min(2000, max(64, edge_count * 8))
    swaps = 0
    for _ in range(attempt_limit):
        if len(rows) < 2 or sub.shape[1] < 2:
            break
        row_a, row_b = rng.choice(len(rows), size=2, replace=False)
        source_a, source_b = rng.choice(sub.shape[1], size=2, replace=False)
        pattern = sub[row_a, source_a], sub[row_a, source_b], sub[row_b, source_a], sub[row_b, source_b]
        if pattern == (1, 0, 0, 1):
            sub[row_a, source_a] = 0
            sub[row_a, source_b] = 1
            sub[row_b, source_a] = 1
            sub[row_b, source_b] = 0
            swaps += 1
        elif pattern == (0, 1, 1, 0):
            sub[row_a, source_a] = 1
            sub[row_a, source_b] = 0
            sub[row_b, source_a] = 0
            sub[row_b, source_b] = 1
            swaps += 1
    membership[rows] = sub
    old_rows = plan.permission_matrix(mechanism).detach().cpu().sum(dim=1).tolist()
    old_cols = plan.permission_matrix(mechanism).detach().cpu().sum(dim=0).tolist()
    value = torch.as_tensor(
        membership,
        device=plan.split_gates.device,
        dtype=plan.split_gates.dtype,
    )
    rewired = plan.with_permission(mechanism, value)
    if old_rows != rewired.permission_matrix(mechanism).detach().cpu().sum(dim=1).tolist():
        raise RuntimeError("The rewiring changed a typed packet's source count.")
    if old_cols != rewired.permission_matrix(mechanism).detach().cpu().sum(dim=0).tolist():
        raise RuntimeError("The rewiring changed typed source packet degree.")
    return rewired, {
        "attempted_switches": attempt_limit,
        "accepted_switches": swaps,
        "available": bool(swaps > 0),
        "unavailable_reason": None if swaps > 0 else "no_valid_degree_preserving_edge_swap_found",
        "row_degrees_preserved": True,
        "column_degrees_preserved": True,
    }


class _GInterventionBuilder:
    """Wrap a G plan builder with a same-weight geometry or rewired control."""

    def __init__(self, base: ThermalCoverPlanBuilder, kind: str, encoded_getter: Any | None = None, seed: int = 2317):
        self.base = base
        self.kind = kind
        self.seed = int(seed)
        self.last_records: tuple[dict[str, Any], ...] = ()
        self.last_plans: tuple[Any, ...] = ()

    def __call__(self, encoded: Any, base_module_state: torch.Tensor, trees: Sequence[Any]):
        plans = self.base(encoded, base_module_state, trees)
        output = []
        records = []
        for case, (tree, plan, original) in enumerate(zip(trees, plans, self.base.last_records, strict=True)):
            details: dict[str, Any] = {"kind": self.kind, "routes": {}}
            updated = plan
            for mechanism in self.base.budget_fractions:
                if self.kind == "geometry_only":
                    target = _canonical_plan_route_work(encoded, tree, plan, mechanism)["achieved_work"]
                    updated, route_info = _geometry_only_plan(encoded, updated, mechanism, target)
                elif self.kind == "rewired":
                    updated, route_info = _rewire_membership(updated, mechanism, self.seed + case * 37 + zlib.crc32(mechanism.encode()))
                    route_info.update(_canonical_plan_route_work(encoded, tree, updated, mechanism))
                else:
                    raise ValueError(f"Unsupported G intervention {self.kind!r}.")
                details["routes"][mechanism] = route_info
            full_record: dict[str, Any] = {
                "mode": "G",
                "frontier": list(original["frontier"]),
                "full_access_bypass_routes": list(updated.explicit_bypass_keys),
                "routes": {},
            }
            for mechanism, fraction in self.base.budget_fractions.items():
                work = _canonical_plan_route_work(encoded, tree, updated, mechanism)
                full_access = work["full_access_work"]
                full_work = work["full_access_work"]
                full_record["routes"][mechanism] = RouteWorkRecord(
                    mechanism=mechanism,
                    requested_fraction=fraction,
                    requested_work=fraction * full_access,
                    achieved_work=work["achieved_work"],
                    full_access_work=full_work,
                    selected_unique_pairs=work["selected_unique_pairs"],
                    full_unique_pairs=work["full_unique_pairs"],
                    sparse_success=work["sparse_success"],
                    executor=f"same_weight_{self.kind}_intervention",
                )
            self.base._add_total_route_work(full_record, encoded, tree, case)
            full_record["intervention"] = details
            output.append(updated)
            records.append(full_record)
        self.last_records = tuple(records)
        self.last_plans = tuple(output)
        return tuple(output)


def _pass_metrics(predictions: StencilPredictions, stencil: Any) -> dict[str, Any]:
    errors, scales, _ = _case_metrics(predictions, stencil)
    return {"role_rmse": errors, "reference_role_rms": scales}


def _load_stage_runtime(args: argparse.Namespace, device: torch.device, stencil: Any):
    source_path = args.source_checkpoint.expanduser().resolve()
    g_path = args.g_checkpoint.expanduser().resolve()
    p_path = args.p_checkpoint.expanduser().resolve()
    if not all(path.is_file() for path in (source_path, g_path, p_path)):
        raise FileNotFoundError("Stage B/C requires the exact e4738 source and matched G/P endpoint checkpoints.")
    model_probe, raw_source = load_model(source_path, device)
    if int(raw_source.get("epoch", raw_source.get("current_epoch", -1))) != 4738:
        raise ValueError("Stage B/C must reload the exact Run1804 epoch-4738 Dense checkpoint.")
    dataset_root = _resolve_dataset_path(raw_source, None)
    dataset_hash = _sha256(dataset_root)
    train_dataset = GlobalChannelThermalDataset(
        dataset_root,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = _make_input_template(train_dataset)
    dataset_config = raw_source["train_config"]["dataset"]
    normalization = raw_source["global_normalization_stats"]
    del model_probe, train_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    forward_manifest = json.loads((g_path.parents[1] / "run_manifest.json").read_text(encoding="utf-8"))
    route_pilot = json.loads((g_path.parents[1] / "route_pilot.json").read_text(encoding="utf-8"))
    extra_route = str(route_pilot["selected_extra_route"]).upper()
    if extra_route not in {"MM", "ME"}:
        raise ValueError(f"Frozen train-only route pilot selected unsupported route {extra_route!r}.")
    g_model, g_route, g_operator, g_saved = _make_model_and_route(
        "G", source_checkpoint=source_path, arm_checkpoint_path=g_path,
        template=template, dataset_config=dataset_config,
        normalization_stats=normalization, stencil=stencil, device=device,
        forward_update=args.forward_update,
    )
    p_model, p_route, p_operator, p_saved = _make_model_and_route(
        "P", source_checkpoint=source_path, arm_checkpoint_path=p_path,
        template=template, dataset_config=dataset_config,
        normalization_stats=normalization, stencil=stencil, device=device,
        forward_update=args.forward_update,
    )
    g_route.eval().requires_grad_(False)
    p_route.eval().requires_grad_(False)
    g_model.eval().requires_grad_(False)
    p_model.eval().requires_grad_(False)
    return {
        "source_path": source_path,
        "g_path": g_path,
        "p_path": p_path,
        "dataset_root": dataset_root,
        "dataset_sha256": dataset_hash,
        "template": template,
        "dataset_config": dataset_config,
        "normalization": normalization,
        "forward_manifest": forward_manifest,
        "route_pilot": route_pilot,
        "extra_route": extra_route,
        "budgets": _budgets(extra_route),
        "g_model": g_model,
        "g_route": g_route,
        "g_operator": g_operator,
        "g_saved": g_saved,
        "p_model": p_model,
        "p_route": p_route,
        "p_operator": p_operator,
        "p_saved": p_saved,
    }


def _selection_for_input(
    organizer: InputOnlyCoverOrganizer,
    capture: Mapping[str, Any],
    cuts: Sequence[Sequence[int]],
    role_tolerance: np.ndarray,
    device: torch.device,
    budgets: Mapping[str, float],
) -> dict[str, Any]:
    score = _organizer_scores(organizer, capture, budgets)
    tree = capture["trees"][0]
    prediction = organizer.score_frontiers(
        score[0], tree, cuts=cuts, budget_vector=score[0].budget_vector
    )
    choice = select_frontier_by_predictions(
        prediction,
        role_tolerance=torch.as_tensor(role_tolerance, device=device, dtype=torch.float32),
        packet_counts=torch.as_tensor([len(cut) for cut in cuts], device=device),
    )
    index = choice.least_risk_index if choice.unsupported_at_budget else choice.selected_index
    if index is None:
        raise RuntimeError("Input-only frontier selection returned neither selected nor least-risk cut.")
    return {
        "index": int(index),
        "frontier": [int(value) for value in cuts[index]],
        "K": int(len(cuts[index])),
        "unsupported_at_budget": bool(choice.unsupported_at_budget),
        "predicted_work_fraction": float(prediction.predicted_work_fraction[index].detach().cpu()),
        "predicted_role_distortion": [float(value) for value in prediction.role_distortion[index].detach().cpu().tolist()],
        "selection_scope": "least_risk_research_only" if choice.unsupported_at_budget else "predicted_adequate",
    }


def _full_work_record(encoded: Any, tree: Any) -> dict[str, Any]:
    from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan
    plan = MechanismPlan.full_access(
        tree,
        encoded.module_present[0],
        int(encoded.env_coords.shape[1]),
    )
    record: dict[str, Any] = {"routes": {}, "full_access_bypass_routes": []}
    for mechanism in ("MM", "ME", "EM", "QM", "QE"):
        work = _canonical_plan_route_work(encoded, tree, plan, mechanism)
        record["routes"][mechanism] = RouteWorkRecord(
            mechanism=mechanism,
            requested_fraction=1.0,
            requested_work=work["full_access_work"],
            achieved_work=work["achieved_work"],
            full_access_work=work["full_access_work"],
            selected_unique_pairs=work["selected_unique_pairs"],
            full_unique_pairs=work["full_unique_pairs"],
            sparse_success=False,
            executor="native_full_access",
        )
    ThermalCoverPlanBuilder._add_total_route_work(record, encoded, tree, 0)
    return record


def _capture_cover_diagnostics(operator: DifferentiableThermalOperator):
    """Temporarily copy scalar native executor diagnostics from model outputs."""

    operator._active_packet_latest_diagnostics = {}

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, Mapping):
            return
        auxiliary = output.get("interaction_aux", {})
        if not isinstance(auxiliary, Mapping):
            return
        values: dict[str, int | float] = {}
        for key, value in auxiliary.items():
            if "cover_" not in str(key) or not torch.is_tensor(value) or value.numel() != 1:
                continue
            scalar = value.detach().cpu().item()
            values[str(key)] = int(scalar) if not value.is_floating_point() else float(scalar)
        operator._active_packet_latest_diagnostics = values

    return operator.model.register_forward_hook(capture)


def _tensor_numpy(value: torch.Tensor, *, dtype: Any | None = None) -> np.ndarray:
    result = value.detach().cpu().numpy()
    return np.asarray(result, dtype=dtype).copy() if dtype is not None else result.copy()


def _physical_source_identity(stencil: Any, encoded: Any, mechanism: str):
    module_sources = mechanism in {"MM", "EM", "QM"}
    if module_sources:
        valid = encoded.module_present[0] > 0.5
        coordinates = encoded.module_centers[0][valid]
        module_ids = tuple(str(value) for value in stencil.baseline.design.active_module_ids)
        if len(module_ids) != int(valid.sum().item()):
            raise RuntimeError("Encoded active module sources do not align with physical module IDs.")
        ids = module_ids
        types = ("module",) * len(ids)
    else:
        valid = encoded.env_weights[0] > 0.0
        coordinates = encoded.env_coords[0][valid]
        xy = _tensor_numpy(coordinates, dtype=np.float64)
        ids = tuple(f"environment_xy:{x:.9g},{y:.9g}" for x, y in xy)
        types = ("environment",) * len(ids)
    return coordinates, valid, ids, types


def _support_payload_for_plan(
    *,
    arm: str,
    plan: Any,
    encoded: Any,
    operator: DifferentiableThermalOperator,
    stencil: Any,
    baseline_prediction: Any,
    route_record: Mapping[str, Any],
    state_diagnostics: Mapping[str, Any],
    prefix: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    from honf_forward_core.interface_fields.budgeted_frontier import canonical_pair_catalog

    support_rows: dict[str, Any] = {}
    module_present = encoded.module_present[0]
    scale = encoded.coordinate_scale
    if scale.ndim == 3:
        scale = scale[0, 0]
    elif scale.ndim == 2:
        scale = scale[0]
    query_coords: list[torch.Tensor] = []
    query_validity: list[np.ndarray] = []
    query_roles: list[str] = []
    for role_name, role_record in stencil.baseline.output.roles.items():
        coords = baseline_prediction.receiver_world_xy[role_name]
        valid = np.asarray(role_record.valid_mask, dtype=bool).any(axis=1)
        query_coords.append(coords)
        query_validity.append(valid)
        query_roles.extend([str(role_name)] * int(coords.shape[0]))
    all_query_coords = torch.cat(query_coords, dim=0)
    all_query_validity = np.concatenate(query_validity)
    receiver_features = operator.model.core.receiver_fourier(
        (all_query_coords / scale.to(device=all_query_coords.device, dtype=all_query_coords.dtype))[None]
    )[0].detach()
    route_work = route_record.get("routes", {})
    total_routes = route_record.get("all_mechanism_route_work", {})
    bypasses = set(route_record.get("full_access_bypass_routes", ()))
    for mechanism in ("MM", "ME", "EM", "QM", "QE"):
        catalog = canonical_pair_catalog(encoded, plan.tree, mechanism, case_index=0)
        source_coords, source_valid, source_ids, source_types = _physical_source_identity(
            stencil, encoded, mechanism
        )
        if mechanism in {"QM", "QE"}:
            receiver_coords = all_query_coords
            receiver_valid = torch.as_tensor(all_query_validity, device=receiver_coords.device, dtype=torch.bool)
            access = plan.access_for(
                mechanism,
                receiver_coords,
                module_present=module_present if mechanism == "QM" else None,
                receiver_features=receiver_features if mechanism == "QE" else None,
            )
            receiver_type_rows = np.asarray(query_roles, dtype=str)
            receiver_identity_rows = np.asarray([
                f"{role}:{query_id}"
                for role, role_record in stencil.baseline.output.roles.items()
                for query_id in role_record.query_ids
            ], dtype=str)
        else:
            receiver_coords = catalog.receiver_coordinates
            receiver_valid = catalog.receiver_validity
            access = plan.access_for(
                mechanism,
                receiver_coords,
                module_present=module_present,
            )
            if mechanism in {"MM", "ME"}:
                receiver_type_rows = np.full(int(receiver_coords.shape[0]), "module", dtype=str)
                active_module_ids = iter(str(value) for value in stencil.baseline.design.active_module_ids)
                receiver_identity_rows = np.asarray([
                    next(active_module_ids) if bool(is_valid) else "excluded_padded_module"
                    for is_valid in _tensor_numpy(receiver_valid, dtype=bool)
                ], dtype=str)
            else:
                receiver_type_rows = np.full(int(receiver_coords.shape[0]), "environment", dtype=str)
                env_xy = _tensor_numpy(receiver_coords, dtype=np.float64)
                receiver_identity_rows = np.asarray([
                    f"environment_xy:{x:.9g},{y:.9g}" for x, y in env_xy
                ], dtype=str)
        valid_pairs = receiver_valid[:, None] & source_valid[None, :]
        if mechanism == "MM" and valid_pairs.shape[0] == valid_pairs.shape[1]:
            valid_pairs = valid_pairs & ~torch.eye(
                valid_pairs.shape[0], device=valid_pairs.device, dtype=torch.bool
            )
        support = (access.detach() > 0) & valid_pairs
        row_mask = receiver_valid.detach().to(device=support.device, dtype=torch.bool)
        col_mask = source_valid.detach().to(device=support.device, dtype=torch.bool)
        support = support[row_mask][:, col_mask]
        receiver_xy = receiver_coords[row_mask]
        source_xy = source_coords
        receiver_type_rows = receiver_type_rows[_tensor_numpy(row_mask, dtype=bool)]
        receiver_identity_rows = receiver_identity_rows[_tensor_numpy(row_mask, dtype=bool)]
        receiver_role_key = f"{prefix}_{arm}_{mechanism}_receiver_roles"
        receiver_id_key = f"{prefix}_{arm}_{mechanism}_receiver_ids"
        weights_key = f"{prefix}_{arm}_{mechanism}_weights"
        receiver_coordinates_key = f"{prefix}_{arm}_{mechanism}_receiver_xy"
        source_coordinates_key = f"{prefix}_{arm}_{mechanism}_source_xy"
        source_ids_key = f"{prefix}_{arm}_{mechanism}_source_ids"
        source_types_key = f"{prefix}_{arm}_{mechanism}_source_types"
        arrays[weights_key] = _tensor_numpy(support, dtype=np.uint8)
        arrays[receiver_coordinates_key] = _tensor_numpy(receiver_xy, dtype=np.float64)
        arrays[source_coordinates_key] = _tensor_numpy(source_xy, dtype=np.float64)
        arrays[receiver_role_key] = receiver_type_rows
        arrays[receiver_id_key] = receiver_identity_rows
        arrays[source_ids_key] = np.asarray(source_ids, dtype=str)
        arrays[source_types_key] = np.asarray(source_types, dtype=str)

        route = route_work.get(mechanism, total_routes.get(mechanism, {}))
        if hasattr(route, "__dataclass_fields__"):
            route = asdict(route)
        else:
            route = dict(route)
        try:
            frontier = plan.frontier_summary(mechanism, receivers=plan.tree.universe.coordinates)
            packet_count = int(frontier.nonredundant_packet_count)
            active_nodes = int(frontier.source_bearing_active_nodes)
        except (ValueError, RuntimeError):
            packet_count = None
            active_nodes = None
        route_token = {
            "MM": "cover_prepare_mm_",
            "ME": "cover_prepare_me_",
            "EM": "cover_prepare_em_",
            "QM": "cover_qm_",
            "QE": "cover_qe_",
        }[mechanism]
        detailed_rows = {
            key: int(value)
            for key, value in state_diagnostics.items()
            if route_token in key and key.endswith(("_executed_rows", "_actual_rows"))
        }
        preferred_rows = {
            key: value for key, value in detailed_rows.items() if key.endswith("_executed_rows")
        } or detailed_rows
        sparse_success = bool(route.get("sparse_success", False))
        route_status = (
            "explicit_full_access_bypass"
            if mechanism in bypasses
            else "sparse_success"
            if sparse_success
            else "no_eligible_pairs"
            if int(route.get("full_unique_pairs", 0)) == 0
            else "zero_or_full_access_support"
            if int(route.get("selected_unique_pairs", 0)) in {0, int(route.get("full_unique_pairs", 0))}
            else "measured_non_saving_or_sparse_route"
        )
        support_rows[mechanism] = {
            "weights_key": weights_key,
            "receiver_coordinates_key": receiver_coordinates_key,
            "source_coordinates_key": source_coordinates_key,
            "receiver_roles_key": receiver_role_key,
            "receiver_ids_key": receiver_id_key,
            "source_ids_key": source_ids_key,
            "source_types_key": source_types_key,
            "source_id_scheme": "physical module IDs; environment rows identified by physical XY coordinate string",
            "work": {
                **dict(route),
                "raw_frontier_size": len(route_record.get("frontier", ())),
                "canonical_packet_count": packet_count,
                "source_bearing_active_nodes": active_nodes,
                "unique_selected_support_pairs": int(support.sum().item()),
                "canonical_work": float(route.get("achieved_work", 0.0)),
                "canonical_full_work": float(route.get("full_access_work", 0.0)),
                "live_executor_rows": int(sum(preferred_rows.values())),
                "live_executor_row_details": detailed_rows,
                "measured_latency_seconds": None,
                "route_status": route_status,
                "explicit_full_access_bypass": mechanism in bypasses,
                "sparse_success": sparse_success,
            },
        }
    return support_rows


def _field_artifact_for_case(
    stencil: Any,
    *,
    predictions: Mapping[str, StencilPredictions],
    plan_captures: Mapping[str, Mapping[str, Any]],
    full_captures: Mapping[str, Mapping[str, Any]],
    operators: Mapping[str, DifferentiableThermalOperator],
    prefix: str,
) -> dict[str, Any]:
    """Package actual native values, finite changes, typed supports, and work."""

    from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan

    arrays: dict[str, np.ndarray] = {}
    fields: dict[str, dict[str, str]] = {"reference": {}}
    residuals: dict[str, dict[str, str]] = {}
    finite_fields: dict[str, dict[str, str]] = {"reference": {}}
    coordinates: dict[str, str] = {}
    finite_coordinates: dict[str, str] = {}
    role_rows: list[dict[str, Any]] = []
    comparison_metrics: dict[str, dict[str, float]] = {}

    arms = tuple(predictions)
    if "full_G_stageC" not in predictions or "full_P_u0200" not in predictions:
        raise ValueError("Thermal field export requires independent same-student G and P full baselines.")
    reference_record = stencil.baseline.output
    if reference_record is None:
        raise ValueError("Thermal field export cannot use a failed reference record.")

    for role_name, role in reference_record.roles.items():
        base_world = predictions["full_G_stageC"].values["baseline"].receiver_world_xy
        if base_world is None or role_name not in base_world:
            raise RuntimeError(f"The native G full pass omitted physical coordinates for {role_name}.")
        role_coordinates = _tensor_numpy(base_world[role_name], dtype=np.float64)
        if role_coordinates.shape != (role.values.shape[0], 2) or not np.isfinite(role_coordinates).all():
            raise ValueError(f"Native {role_name} coordinates do not align with reference samples.")
        for channel, (channel_name, units) in enumerate(zip(role.channel_names, role.channel_units, strict=True)):
            role_key = f"{role_name}:{channel_name}"
            token = re.sub(r"[^A-Za-z0-9_]+", "_", f"{prefix}_{role_name}_{channel_name}").strip("_")
            coordinate_key = f"{token}_coordinates_xy"
            weight_key = f"{token}_weights"
            validity_key = f"{token}_valid"
            arrays[coordinate_key] = role_coordinates
            valid = np.asarray(role.valid_mask[:, channel], dtype=bool)
            weights = np.asarray(role.quadrature_weights, dtype=np.float64)
            arrays[weight_key] = weights * valid
            arrays[validity_key] = valid.astype(np.uint8)
            coordinates[role_key] = coordinate_key
            role_rows.append({
                "name": role_key,
                "units": str(units),
                "group": str(role_name),
                "weights_key": weight_key,
                "valid_mask_key": validity_key,
                "coordinate_kind": str(role.coordinate_kind),
                "invalid_value_handling": "masked rows have zero quadrature weight; any nonfinite masked value is stored as zero and its validity mask is retained",
            })

            reference_values = np.asarray(role.values[:, channel], dtype=np.float64)
            if np.any(valid & ~np.isfinite(reference_values)):
                raise ValueError(f"Thermal reference has a nonfinite valid value in {role_key}.")
            reference_values = np.where(valid, reference_values, 0.0)
            reference_key = f"{token}_reference"
            arrays[reference_key] = reference_values
            fields["reference"][role_key] = reference_key

            for arm in arms:
                prediction = predictions[arm]
                if "baseline" not in prediction.values or role_name not in prediction.values["baseline"].role_values:
                    raise RuntimeError(f"Thermal arm {arm} omitted baseline {role_key}.")
                arm_world = prediction.values["baseline"].receiver_world_xy
                if arm_world is None or role_name not in arm_world:
                    raise RuntimeError(f"Thermal arm {arm} omitted {role_name} receiver coordinates.")
                if not np.allclose(_tensor_numpy(arm_world[role_name]), role_coordinates, atol=1.0e-7, rtol=1.0e-7):
                    raise ValueError(f"Thermal arm {arm} changed the physical query coordinates for {role_name}.")
                value = _tensor_numpy(prediction.values["baseline"].role_values[role_name][:, channel], dtype=np.float64)
                if np.any(valid & ~np.isfinite(value)):
                    raise ValueError(f"Thermal arm {arm} has a nonfinite valid value in {role_key}.")
                value = np.where(valid, value, 0.0)
                key = f"{token}_{re.sub(r'[^A-Za-z0-9_]+', '_', arm)}"
                arrays[key] = value
                fields.setdefault(arm, {})[role_key] = key
                residual_key = f"{key}_minus_reference_unclipped"
                arrays[residual_key] = np.where(valid, value - reference_values, 0.0)
                residuals.setdefault(arm, {})[role_key] = residual_key
                comparison_metrics.setdefault(arm, {})[role_key] = _weighted_rmse(
                    value, reference_values, valid, weights
                )

    finite_available = "i_plus" in stencil.variants
    if finite_available:
        variant_record = stencil.variants["i_plus"]
        if variant_record.output is None:
            raise ValueError("Thermal i_plus response is not a converged physical output.")
        for role_name, base_role in reference_record.roles.items():
            variant_role = variant_record.output.roles[role_name]
            base_world = predictions["full_G_stageC"].values["i_plus"].receiver_world_xy
            if base_world is None or role_name not in base_world:
                raise RuntimeError(f"The native G full pass omitted i_plus coordinates for {role_name}.")
            response_coordinates = _tensor_numpy(base_world[role_name], dtype=np.float64)
            if response_coordinates.shape != (variant_role.values.shape[0], 2):
                raise ValueError(f"Thermal i_plus {role_name} coordinates do not align with reference samples.")
            for channel, channel_name in enumerate(base_role.channel_names):
                role_key = f"{role_name}:{channel_name}"
                token = re.sub(r"[^A-Za-z0-9_]+", "_", f"{prefix}_finite_{role_name}_{channel_name}").strip("_")
                coord_key = f"{token}_coordinates_xy"
                arrays[coord_key] = response_coordinates
                finite_coordinates[role_key] = coord_key
                common_valid = np.asarray(base_role.valid_mask[:, channel], dtype=bool) & np.asarray(
                    variant_role.valid_mask[:, channel], dtype=bool
                )
                finite_weight_key = f"{token}_weights"
                finite_valid_key = f"{token}_valid"
                arrays[finite_weight_key] = np.asarray(base_role.quadrature_weights, dtype=np.float64) * common_valid
                arrays[finite_valid_key] = common_valid.astype(np.uint8)
                role_rows_for_finite = [row for row in role_rows if row["name"] == role_key]
                if role_rows_for_finite:
                    # The same role declaration supports both absolute and finite maps;
                    # the finite arrays below retain their stricter paired-valid mask.
                    role_rows_for_finite[0]["finite_weights_key"] = finite_weight_key
                    role_rows_for_finite[0]["finite_valid_mask_key"] = finite_valid_key
                ref_delta = np.asarray(variant_role.values[:, channel], dtype=np.float64) - np.asarray(
                    base_role.values[:, channel], dtype=np.float64
                )
                if np.any(common_valid & ~np.isfinite(ref_delta)):
                    raise ValueError(f"Thermal reference has a nonfinite valid finite change in {role_key}.")
                ref_delta = np.where(common_valid, ref_delta, 0.0)
                reference_key = f"{token}_reference_delta"
                arrays[reference_key] = ref_delta
                finite_fields["reference"][role_key] = reference_key
                for arm in arms:
                    prediction = predictions[arm]
                    predicted_delta = _tensor_numpy(
                        prediction.values["i_plus"].role_values[role_name][:, channel]
                        - prediction.values["baseline"].role_values[role_name][:, channel],
                        dtype=np.float64,
                    )
                    if np.any(common_valid & ~np.isfinite(predicted_delta)):
                        raise ValueError(f"Thermal arm {arm} has a nonfinite valid finite change in {role_key}.")
                    key = f"{token}_{re.sub(r'[^A-Za-z0-9_]+', '_', arm)}_delta"
                    arrays[key] = np.where(common_valid, predicted_delta, 0.0)
                    finite_fields.setdefault(arm, {})[role_key] = key

    supports: dict[str, Any] = {}
    work: dict[str, Any] = {}
    full_aliases = {
        "full_G_stageC": "full_G_stageC",
        "full_P_u0200": "full_P_u0200",
        "full_root_union": "full_G_stageC",
        "full_geometry_only": "full_G_stageC",
        "full_rewired": "full_G_stageC",
        "full_P_Gwork": "full_P_u0200",
    }
    for full_label, source_label in full_aliases.items():
        if full_label != source_label:
            fields[full_label] = fields[source_label]
            finite_fields[full_label] = finite_fields[source_label]

    for label, capture in full_captures.items():
        encoded = capture["encoded"]
        tree = capture["trees"][0]
        plan = MechanismPlan.full_access(tree, encoded.module_present[0], int(encoded.env_coords.shape[1]))
        full_record = _full_work_record(encoded, tree)
        diagnostics = capture.get("state_runtime_diagnostics", {}).get("baseline", {})
        operator = operators["G" if label.startswith("full_G") else "P"]
        supports[label] = _support_payload_for_plan(
            arm=label, plan=plan, encoded=encoded, operator=operator, stencil=stencil,
            baseline_prediction=predictions[label].values["baseline"], route_record=full_record,
            state_diagnostics=diagnostics, prefix=prefix, arrays=arrays,
        )
        route_work, _ = _cut_work(full_record)
        total = route_work["all_mechanisms_total"]
        work[label] = _arm_work_summary(full_record, diagnostics, route_work)

    for label, capture in plan_captures.items():
        state = capture["baseline"]
        plan = state["plans"][0]
        encoded = state["encoded"]
        tree = state["trees"][0]
        record = state["records"][0]
        diagnostics = state.get("runtime_diagnostics", {})
        operator = operators["G" if label in {"G_stageC", "root_union", "geometry_only", "rewired"} else "P"]
        supports[label] = _support_payload_for_plan(
            arm=label, plan=plan, encoded=encoded, operator=operator, stencil=stencil,
            baseline_prediction=predictions[label].values["baseline"], route_record=record,
            state_diagnostics=diagnostics, prefix=prefix, arrays=arrays,
        )
        route_work, _ = _cut_work(record)
        work[label] = _arm_work_summary(record, diagnostics, route_work)

    for full_label, source_label in full_aliases.items():
        if full_label != source_label:
            supports[full_label] = supports[source_label]
            work[full_label] = work[source_label]

    fluid = reference_record.roles["fluid_fields"]
    fluid_xy = _tensor_numpy(predictions["full_G_stageC"].values["baseline"].receiver_world_xy["fluid_fields"], dtype=np.float64)
    modules = tuple(module for module in stencil.baseline.design.modules if module.active)
    centers_key = f"{prefix}_module_centers_xy"
    module_ids_key = f"{prefix}_physical_module_ids"
    arrays[centers_key] = np.asarray([module.position_xy for module in modules], dtype=np.float64)
    arrays[module_ids_key] = np.asarray([module.module_id for module in modules], dtype=str)
    variant_centers_key = None
    variant_module_ids_key = None
    if finite_available:
        variant_modules = tuple(module for module in stencil.variants["i_plus"].design.modules if module.active)
        variant_centers_key = f"{prefix}_i_plus_module_centers_xy"
        variant_module_ids_key = f"{prefix}_i_plus_physical_module_ids"
        arrays[variant_centers_key] = np.asarray([module.position_xy for module in variant_modules], dtype=np.float64)
        arrays[variant_module_ids_key] = np.asarray([module.module_id for module in variant_modules], dtype=str)
    radius = float(operators["G"].model.config.core_honf.module_radius)
    return {
        "arrays": {"coordinates": coordinates, "residuals": residuals},
        "roles": role_rows,
        "fields": fields,
        "unclipped_residuals": residuals,
        "finite_response": {
            "available": bool(finite_available),
            "reason": None if finite_available else "The selected response stencil has no i_plus state.",
            "roles": role_rows,
            "coordinates": finite_coordinates,
            "fields": finite_fields,
            "metadata": _finite_response_metadata(stencil) if finite_available else {},
        },
        "supports": supports,
        "work": work,
        "comparison_metrics": comparison_metrics,
        "geometry": {
            "module_centers_key": centers_key,
            "module_ids_key": module_ids_key,
            "i_plus_module_centers_key": variant_centers_key,
            "i_plus_module_ids_key": variant_module_ids_key,
            "module_radius": radius,
            "interface_coordinates_key": next(
                (value for key, value in coordinates.items() if key.startswith("interface:")), None
            ),
            "native_fluid_sample_bbox_xy": [
                float(fluid_xy[:, 0].min()), float(fluid_xy[:, 1].min()),
                float(fluid_xy[:, 0].max()), float(fluid_xy[:, 1].max()),
            ],
            "domain_box_semantics": "bounding box of saved physical fluid receiver coordinates; no full mesh boundary is implied",
            "module_ids_are_physical": True,
        },
        "arrays_npz": arrays,
    }


def _arm_work_summary(record: Mapping[str, Any], diagnostics: Mapping[str, Any], route_work: Mapping[str, Any]) -> dict[str, Any]:
    total = route_work["all_mechanisms_total"]
    routes = route_work.get("all_mechanism_route_work", {})
    details = {
        str(key): int(value) for key, value in diagnostics.items()
        if str(key).endswith(("_executed_rows", "_actual_rows"))
    }
    route_status: dict[str, str] = {}
    sparse_success: dict[str, bool] = {}
    bypasses = set(record.get("full_access_bypass_routes", ()))
    for name, value in record.get("routes", {}).items():
        item = asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value)
        sparse = bool(item.get("sparse_success", False))
        selected = int(item.get("selected_unique_pairs", 0))
        full = int(item.get("full_unique_pairs", 0))
        route_status[name] = (
            "explicit_full_access_bypass" if name in bypasses
            else "sparse_success" if sparse
            else "no_eligible_pairs" if full == 0
            else "full_access_control" if selected == full
            else "zero_pair_support" if selected == 0
            else "measured_non_saving_or_sparse_route"
        )
        sparse_success[name] = sparse
    return {
        "canonical_unique_pair_count": int(sum(int(value.get("selected_unique_pairs", 0)) for value in routes.values())),
        "canonical_full_unique_pair_count": int(sum(int(value.get("full_unique_pairs", 0)) for value in routes.values())),
        "canonical_work": float(total["achieved_work"]),
        "canonical_full_work": float(total["full_access_work"]),
        "canonical_work_fraction": float(total["achieved_work"]) / max(float(total["full_access_work"]), 1.0e-12),
        "live_executor_rows": int(sum(details.values())) if details else None,
        "live_executor_row_details": details,
        "latency_seconds": None,
        "explicit_full_access_bypass_routes": list(record.get("full_access_bypass_routes", ())),
        "route_status": route_status,
        "sparse_success": sparse_success,
    }


def _finite_response_metadata(stencil: Any) -> dict[str, Any]:
    variant = stencil.variants["i_plus"]
    evidence = dict(stencil.evidence_manifest)
    return {
        "response_source": evidence.get("response_source", "original_response_atlas_subset"),
        "family_id": evidence.get("family_id", stencil.physical_family_id),
        "baseline_record_id": stencil.baseline.record_id,
        "variant_record_id": variant.record_id,
        "baseline_source": stencil.baseline.source.value,
        "variant_source": variant.source.value,
        "baseline_record_sha256": evidence.get("baseline_record_sha256"),
        "variant_record_sha256": evidence.get("variant_record_sha256"),
        "source_attempt_ids": [
            value for value in (
                evidence.get("baseline_coverage_attempt_id"),
                evidence.get("variant_coverage_attempt_id"),
            ) if value is not None
        ],
        "perturbation_by_physical_module_id": {
            key: list(value) for key, value in variant.perturbation_by_module.items()
        },
        "physical_step_scales": dict(variant.physical_step_scales),
        "numerical_resolution": "unknown; no repeated local-generator resolution study was performed for this pair",
        "numerical_allowance": "Stage-B chunk-parity allowance is recorded separately in physical units; local-generator discretization error is not quantified",
        "dataset_units": "role-specific units in the roles list; no SI conversion is documented",
    }


def _evaluate_one_case(
    stencil: Any,
    *,
    runtime: Mapping[str, Any],
    cut: Sequence[int],
    device: torch.device,
    extra_route: str,
    seed: int,
    query_batch_size: int = 512,
    capture_arrays: bool = False,
    frontier_pattern_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    g_model, g_route = runtime["g_model"], runtime["g_route"]
    p_model, p_route = runtime["p_model"], runtime["p_route"]
    g_operator, p_operator = runtime["g_operator"], runtime["p_operator"]
    g_operator.query_batch_size = int(query_batch_size)
    p_operator.query_batch_size = int(query_batch_size)
    g_before = _state_hash(g_model.state_dict())
    gr_before = _state_hash(g_route.state_dict())
    p_before = _state_hash(p_model.state_dict())
    pr_before = _state_hash(p_route.state_dict())
    with torch.no_grad():
        full_g, full_capture = _full_plan_predictions(
            g_operator, stencil, device, query_batch_size, capture_baseline=True
        )
        full_p, p_full_capture = _full_plan_predictions(
            p_operator, stencil, device, query_batch_size, capture_baseline=True
        )
    if full_capture is None:
        raise RuntimeError("Full-access G control did not capture its candidate tree for work accounting.")
    if p_full_capture is None:
        raise RuntimeError("Full-access P control did not capture its current-design tree.")
    pattern_paths = (
        tuple(str(value) for value in frontier_pattern_paths)
        if frontier_pattern_paths is not None
        else frontier_paths(full_capture["trees"][0], cut, max_depth=3)
    )
    baseline_cut = frontier_from_paths(full_capture["trees"][0], pattern_paths, max_depth=3)
    if len(baseline_cut) != len(cut):
        raise ValueError("The selected frontier path pattern changed cut size on the baseline tree.")
    g_builder = _builder(
        "G", g_model, g_route, extra_route, baseline_cut,
        frontier_pattern_paths=pattern_paths,
    )
    p_builder = _builder(
        "P", p_model, p_route, extra_route, baseline_cut,
        frontier_pattern_paths=pattern_paths,
    )
    root_path = frontier_paths(full_capture["trees"][0], (0,), max_depth=3)
    root_builder = _builder(
        "G", g_model, g_route, extra_route, (0,),
        frontier_pattern_paths=root_path,
    )
    geometry_builder = _GInterventionBuilder(
        _builder("G", g_model, g_route, extra_route, baseline_cut,
                 frontier_pattern_paths=pattern_paths), "geometry_only", seed=seed
    )
    rewire_builder = _GInterventionBuilder(
        _builder("G", g_model, g_route, extra_route, baseline_cut,
                 frontier_pattern_paths=pattern_paths), "rewired", seed=seed
    )
    with torch.no_grad():
        selected_g, _, g_record = _predict_stencil(
            g_operator, stencil, device=device, cover_plan_builder=g_builder,
            capture_plan_states=capture_arrays,
        )
        root_union, _, root_record = _predict_stencil(
            g_operator, stencil, device=device, cover_plan_builder=root_builder,
            capture_plan_states=capture_arrays,
        )
        geometry, _, geometry_record = _predict_stencil(
            g_operator, stencil, device=device, cover_plan_builder=geometry_builder,
            capture_plan_states=capture_arrays,
        )
        rewired, _, rewire_record = _predict_stencil(
            g_operator, stencil, device=device, cover_plan_builder=rewire_builder,
            capture_plan_states=capture_arrays,
        )
        direct_p, _, p_record = _predict_stencil(
            p_operator, stencil, device=device, cover_plan_builder=p_builder,
            capture_plan_states=capture_arrays,
        )
    full_work = _full_work_record(full_capture["encoded"], full_capture["trees"][0])
    for label, module, before in (
        ("G physical", g_model, g_before), ("G organizer", g_route, gr_before),
        ("P physical", p_model, p_before), ("P scorer", p_route, pr_before),
    ):
        if _state_hash(module.state_dict()) != before:
            raise RuntimeError(f"A fixed-weight Stage-B intervention mutated {label} weights.")
    g_full_metrics = _pass_metrics(full_g, stencil)
    p_full_metrics = _pass_metrics(full_p, stencil)
    outcomes = {}
    outcomes["G_full_access_same_weights"] = {
        "role_rmse": g_full_metrics["role_rmse"],
        "same_student_full_role_rmse": g_full_metrics["role_rmse"],
        "signed_sparse_minus_full_rmse": [0.0] * len(g_full_metrics["role_rmse"]),
        "typed_route_work": _cut_work(full_work)[0],
        "sparse_success": False,
    }
    outcomes["P_full_access_same_P_weights"] = {
        "role_rmse": p_full_metrics["role_rmse"],
        "same_student_full_role_rmse": p_full_metrics["role_rmse"],
        "signed_sparse_minus_full_rmse": [0.0] * len(p_full_metrics["role_rmse"]),
        "typed_route_work": _cut_work(full_work)[0],
        "sparse_success": False,
        "comparison_scope": "P is independently trained; not a same-weight G intervention",
    }
    for label, prediction, record, full_metrics in (
        ("G_selected_cut", selected_g, g_record, g_full_metrics),
        ("G_root_source_union", root_union, root_record, g_full_metrics),
        ("G_geometry_only_matched_work", geometry, geometry_record, g_full_metrics),
        ("G_degree_size_rewired", rewired, rewire_record, g_full_metrics),
        ("P_independent_0p90", direct_p, p_record, p_full_metrics),
    ):
        metrics = _pass_metrics(prediction, stencil)
        rmse = np.asarray(metrics["role_rmse"], dtype=np.float64)
        full_rmse = np.asarray(full_metrics["role_rmse"], dtype=np.float64)
        outcomes[label] = {
            "role_rmse": rmse.tolist(),
            "same_student_full_role_rmse": full_rmse.tolist(),
            "signed_sparse_minus_full_rmse": (rmse - full_rmse).tolist(),
            "role_rmse_ratio_to_full": (rmse / np.maximum(full_rmse, 1.0e-12)).tolist(),
            "typed_route_work": _cut_work(record)[0],
        }
    normalized_g_record, _ = _cut_work(g_record)
    g_routes = {
        mechanism: value for mechanism, value in normalized_g_record.items()
        if mechanism in {"MM", "ME", "EM", "QM", "QE"}
    }
    matched_budget = {
        mechanism: float(value["achieved_work"]) / float(value["full_access_work"])
        for mechanism, value in g_routes.items()
        if float(value["full_access_work"]) > 0.0
    }
    p_matched_builder = _builder(
        "P", p_model, p_route, extra_route, baseline_cut,
        budget_fractions=matched_budget,
        frontier_pattern_paths=pattern_paths,
    )
    with torch.no_grad():
        p_matched, _, p_matched_record = _predict_stencil(
            p_operator, stencil, device=device, cover_plan_builder=p_matched_builder,
            capture_plan_states=capture_arrays,
        )
    outcomes["P_checkpoint_only_G_canonical_work_targets"] = {
        "role_rmse": _case_metrics(p_matched, stencil)[0],
        "same_student_full_role_rmse": p_full_metrics["role_rmse"],
        "signed_sparse_minus_full_rmse": (
            np.asarray(_case_metrics(p_matched, stencil)[0]) - np.asarray(p_full_metrics["role_rmse"])
        ).tolist(),
        "typed_route_work": _cut_work(p_matched_record)[0],
        "requested_budget_fractions_from_G": matched_budget,
        "note": "P retains its own learned memberships; only requested canonical work fractions are transferred.",
    }
    result = {
        "case_id": stencil.physical_family_id,
        "frontier": [int(value) for value in baseline_cut],
        "frontier_paths": list(pattern_paths),
        "K": int(len(baseline_cut)),
        "split": stencil.split.value,
        "source_provenance": dict(stencil.evidence_manifest),
        "role_schema": _role_rows(stencil),
        "full_G": g_full_metrics,
        "full_P": p_full_metrics,
        "interventions": outcomes,
        "same_weight_state_hashes_unchanged": True,
        "query_batch_size": int(query_batch_size),
    }
    if capture_arrays:
        result["field_artifact"] = _field_artifact_for_case(
            stencil,
            predictions={
                "full_G_stageC": full_g,
                "G_stageC": selected_g,
                "full_P_u0200": full_p,
                "P_u0200": direct_p,
                "root_union": root_union,
                "geometry_only": geometry,
                "rewired": rewired,
                "P_Gwork": p_matched,
            },
            plan_captures={
                "G_stageC": g_builder.last_state_captures,
                "root_union": root_builder.last_state_captures,
                "geometry_only": geometry_builder.last_state_captures,
                "rewired": rewire_builder.last_state_captures,
                "P_u0200": p_builder.last_state_captures,
                "P_Gwork": p_matched_builder.last_state_captures,
            },
            full_captures={"full_G_stageC": full_capture, "full_P_u0200": p_full_capture},
            operators={"G": g_operator, "P": p_operator},
            prefix=re.sub(r"[^A-Za-z0-9_]+", "_", stencil.physical_family_id).strip("_"),
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    forward_attempt_path = (
        args.native_attempt_ledger.expanduser().resolve()
        if args.native_attempt_ledger is not None
        else output / "native_state_forward_attempts.jsonl"
    )
    _configure_forward_attempt_ledger(forward_attempt_path, args.phase)
    device = torch.device(args.device)
    if device.type == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or device.index != 0:
            raise RuntimeError("Thermal Stage B/C requires physical GPU2 exposed as logical cuda:0.")
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.phase == "fit":
        stage_b_path = output / "stage_b_table.json"
        fit_freeze_path = output / "stage_c_fit_frozen.json"
        if stage_b_path.exists() or fit_freeze_path.exists():
            raise FileExistsError("Stage B/C fit artifacts already exist; use a fresh versioned output directory.")
        fit_stencils = _load_fit_stencils(
            args.atlas_dir.expanduser().resolve(),
            seed=args.panel_seed,
            calls_path=args.coverage_manifest,
            predeclared_path=args.predeclared_manifest,
        )
        runtime = _load_stage_runtime(args, device, fit_stencils[0])
        g_model, base_g = runtime["g_model"], runtime["g_route"]
        p_route = runtime["p_route"]
        budgets = runtime["budgets"]
        extra_route = runtime["extra_route"]
        measured: list[dict[str, Any]] = []
        parity_measurements: list[np.ndarray] = []
        for fit_index, stencil in enumerate(fit_stencils):
            full_g, capture_g = _full_plan_predictions(
                runtime["g_operator"], stencil, device, 512, capture_baseline=True
            )
            full_p, _ = _full_plan_predictions(runtime["p_operator"], stencil, device, 512)
            if fit_index in {0, 9}:
                full_g_q2048, _ = _full_plan_predictions(runtime["g_operator"], stencil, device, 2048)
                full_p_q2048, _ = _full_plan_predictions(runtime["p_operator"], stencil, device, 2048)
                parity_measurements.append(np.maximum(
                    np.asarray(_parity_metrics(full_g, full_g_q2048, stencil), dtype=np.float64),
                    np.asarray(_parity_metrics(full_p, full_p_q2048, stencil), dtype=np.float64),
                ))
                del full_g_q2048, full_p_q2048
            full_errors, reference_scales, _ = _case_metrics(full_g, stencil)
            p_full_errors, _, _ = _case_metrics(full_p, stencil)
            if capture_g is None:
                raise RuntimeError("G full-native pass did not capture candidate-encoded inputs/tree.")
            cuts = enumerate_frontier_cuts(capture_g["trees"][0], max_depth=3)
            if len(cuts) != 26:
                raise RuntimeError(f"Expected 26 depth-three frontiers, got {len(cuts)}.")
            scores = _organizer_scores(base_g, capture_g, budgets)
            item: dict[str, Any] = {
                "case_id": stencil.physical_family_id,
                "stencil": stencil,
                "capture": capture_g,
                "cuts": [],
                "declared_cuts": cuts,
                "organizer_scores": scores,
                "full_role_errors": np.asarray(full_errors, dtype=np.float64),
                "p_full_role_errors": np.asarray(p_full_errors, dtype=np.float64),
                "reference_role_scales": np.asarray(reference_scales, dtype=np.float64),
                "role_errors": [],
                "work_fraction": [],
                "route_work": [],
                "cut_records": [],
            }
            for cut in cuts:
                pattern_paths = frontier_paths(capture_g["trees"][0], cut, max_depth=3)
                builder = _builder(
                    "G", g_model, base_g, extra_route, cut,
                    frontier_pattern_paths=pattern_paths,
                )
                try:
                    sparse_g, _, base_record = _predict_stencil(
                        runtime["g_operator"], stencil, device=device,
                        cover_plan_builder=builder, capture_plan_states=True,
                    )
                except UnsupportedFrontierPatternError as exc:
                    item["cut_records"].append({
                        "frontier": [int(value) for value in cut],
                        "frontier_paths": list(pattern_paths),
                        "K": int(len(cut)),
                        "measurement_status": "unsupported_current_state_tree_pattern",
                        "unavailable_state": exc.state_label,
                        "unavailable_reason": str(exc),
                        "sparse_success": False,
                        "split": "train",
                    })
                    continue
                if base_record is None:
                    raise RuntimeError("Stage B sparse pass did not record measured canonical work.")
                error, _, _ = _case_metrics(sparse_g, stencil)
                route_work, fraction = _cut_work(base_record)
                item["cuts"].append(tuple(int(value) for value in cut))
                item["role_errors"].append(error)
                item["work_fraction"].append(fraction)
                item["route_work"].append(route_work)
                item["cut_records"].append({
                    "frontier": [int(value) for value in cut],
                    "frontier_paths": list(pattern_paths),
                    "state_specific_node_ids": {
                        label: list(state_capture["records"][0].get("frontier", ()))
                        for label, state_capture in builder.last_state_captures.items()
                    },
                    "K": int(len(cut)),
                    "measurement_status": "complete_baseline_and_all_response_states",
                    "role_rmse": error,
                    "same_student_full_role_rmse": np.asarray(full_errors, dtype=np.float64).tolist(),
                    "signed_sparse_minus_full_rmse": [float(left - right) for left, right in zip(error, full_errors, strict=True)],
                    "summed_canonical_work_fraction_including_full_bypasses": fraction,
                    "typed_route_work": route_work,
                    "explicit_full_access_bypass_routes": list(base_record.get("full_access_bypass_routes", ())),
                    "route_sparse_success": {
                        key: bool(value.get("sparse_success", False))
                        for key, value in route_work.items()
                        if key in {"MM", "ME", "EM", "QM", "QE"}
                    },
                    "split": "train",
                    "reference_sources": {"baseline": stencil.baseline.source.value,
                                           **{name: record.source.value for name, record in stencil.variants.items()}},
                })
            if not item["cuts"]:
                raise RuntimeError(f"Stage B found no fully measurable frontier patterns for {item['case_id']}.")
            item["role_errors"] = np.asarray(item["role_errors"], dtype=np.float64)
            item["work_fraction"] = np.asarray(item["work_fraction"], dtype=np.float64)
            measured.append(item)
            print(json.dumps({
                "stage": "B_table_case_complete", "case_id": item["case_id"],
                "declared_cut_count": len(cuts), "measured_cut_count": len(item["cuts"]),
                "planned_candidate_state_forwards": 2 * len(cuts),
                "M": int(capture_g["encoded"].module_present[0].sum().item()),
                "elapsed_seconds": time.monotonic() - started,
            }, sort_keys=True), flush=True)
            del full_g, full_p
            if device.type == "cuda":
                torch.cuda.empty_cache()

        role_rows = _role_rows(fit_stencils[0])
        role_count = len(role_rows)
        if len(parity_measurements) != 2:
            raise RuntimeError("Thermal Stage B needs two repeated G/P q512-versus-q2048 parity checks.")
        numerical_allowance = np.max(np.stack(parity_measurements), axis=0)
        role_scales_np = np.maximum(
            np.sqrt(np.mean(np.square(np.stack([item["reference_role_scales"] for item in measured])), axis=0)),
            1.0e-8,
        )
        for item in measured:
            complete_cut_rows = [
                row for row in item["cut_records"]
                if row.get("measurement_status") == "complete_baseline_and_all_response_states"
            ]
            for cut_row, candidate_error in zip(complete_cut_rows, item["role_errors"], strict=True):
                flags = [
                    float(candidate) <= 1.10 * float(full) + float(allowance)
                    for candidate, full, allowance in zip(candidate_error, item["full_role_errors"], numerical_allowance, strict=True)
                ]
                cut_row["adequate_by_measured_gate"] = flags
                cut_row["measured_adequacy_all_roles"] = bool(all(flags))
        tolerance_rows = [
            (0.10 * item["full_role_errors"] + numerical_allowance) / role_scales_np
            for item in measured
        ]
        inverse_tolerance = np.mean(np.stack(tolerance_rows), axis=0)
        organizer_with_utility = InputOnlyCoverOrganizer(
            state_dim=int(measured[0]["capture"]["encoded"].module_tokens.shape[-1]),
            module_feature_dim=int(measured[0]["capture"]["encoded"].module_features.shape[-1]),
            environment_feature_dim=int(measured[0]["capture"]["encoded"].env_features.shape[-1]),
            hidden_dim=96,
            role_count=8,
            frontier_utility_enabled=True,
            frontier_role_count=role_count,
        ).to(device=device)
        missing, unexpected = organizer_with_utility.load_state_dict(base_g.state_dict(), strict=False)
        if unexpected or any(not key.startswith("frontier_utility_head.") for key in missing):
            raise RuntimeError(f"Stage C base G organizer load mismatch: missing={missing}, unexpected={unexpected}")
        organizer_with_utility.eval()
        for name, parameter in organizer_with_utility.named_parameters():
            parameter.requires_grad_(name.startswith("frontier_utility_head."))
        history_path = output / "stage_c_updates.jsonl"
        if history_path.exists():
            raise FileExistsError(f"Refusing to append Stage C to an existing run: {history_path}")
        history = _fit_utility(
            organizer_with_utility,
            measured,
            role_scales=torch.as_tensor(role_scales_np, dtype=torch.float32),
            updates=args.utility_updates,
            seed=args.seed,
            output_path=history_path,
            device=device,
        )
        organizer_with_utility.eval()

        case_selection: list[dict[str, Any]] = []
        selection_by_case: dict[str, dict[str, Any]] = {}
        for item in measured:
            score = item["organizer_scores"][0]
            tree = item["capture"]["trees"][0]
            prediction = organizer_with_utility.score_frontiers(
                score, tree, cuts=item["cuts"], budget_vector=score.budget_vector
            )
            selection = select_frontier_by_predictions(
                prediction,
                role_tolerance=torch.as_tensor(inverse_tolerance, device=device, dtype=torch.float32),
                packet_counts=torch.as_tensor([len(cut) for cut in item["cuts"]], device=device),
            )
            chosen = selection.least_risk_index if selection.unsupported_at_budget else selection.selected_index
            if chosen is None:
                raise RuntimeError("Stage C did not identify a predicted or least-risk measured cut.")
            actual = item["role_errors"][chosen]
            full = item["full_role_errors"]
            adequate = np.asarray([
                float(candidate) <= 1.10 * float(reference) + float(allowance)
                for candidate, reference, allowance in zip(actual, full, numerical_allowance, strict=True)
            ], dtype=bool)
            row = {
                "case_id": item["case_id"], "split": "train",
                "utility_fit_partition": "fit",
                "frontier": [int(value) for value in item["cuts"][chosen]],
                "frontier_paths": list(frontier_paths(tree, item["cuts"][chosen], max_depth=3)),
                "K": int(len(item["cuts"][chosen])),
                "unsupported_at_budget": bool(selection.unsupported_at_budget),
                "predicted_adequate": not bool(selection.unsupported_at_budget),
                "predicted_scope": "least_risk_research_only" if selection.unsupported_at_budget else "predicted_adequate",
                "predicted_work_fraction": float(prediction.predicted_work_fraction[chosen].detach().cpu()),
                "predicted_role_distortion": [float(value) for value in prediction.role_distortion[chosen].detach().cpu().tolist()],
                "measured_role_rmse": np.asarray(actual).tolist(),
                "same_student_full_role_rmse": np.asarray(full).tolist(),
                "physical_numerical_allowance": numerical_allowance.tolist(),
                "measured_adequate_by_1p10_full_plus_allowance": bool(adequate.all()),
                "per_role_adequacy": adequate.tolist(),
                "typed_route_work": item["route_work"][chosen],
                "reference_sources": {"baseline": item["stencil"].baseline.source.value,
                                      **{name: rec.source.value for name, rec in item["stencil"].variants.items()}},
            }
            case_selection.append(row)
            selection_by_case[row["case_id"]] = row

        # Rank the two held training layouts from input-only geometry and
        # operating context before their response outputs are opened.
        held_inputs = _held_input_records(predeclared_path=args.predeclared_manifest)
        held_rankings: list[dict[str, Any]] = []
        for spec, raw_record in held_inputs:
            input_candidate = _sample_input_candidate(raw_record, seed=args.panel_seed + 211)
            with torch.no_grad():
                _, capture = _full_plan_predictions(
                    runtime["g_operator"], input_candidate, device, 512, capture_baseline=True
                )
            if capture is None:
                raise RuntimeError("Target-free held input read did not expose its candidate tree.")
            cuts = enumerate_frontier_cuts(capture["trees"][0], max_depth=3)
            if len(cuts) != 26:
                raise RuntimeError("The held candidate tree does not expose all 26 declared cuts.")
            budgets_input = {
                "module_states": capture["p0_module_states"],
                "environment_states": capture["encoded"].env_tokens,
                "global_state": capture["encoded"].global_token,
            }
            scores = organizer_with_utility.score_cases(
                capture["encoded"], budgets_input, capture["trees"], budgets=budgets
            )
            capture = dict(capture)
            capture["organizer_scores"] = scores
            selection = _selection_for_input(
                organizer_with_utility, capture, cuts, inverse_tolerance, device, budgets
            )
            held_rankings.append({
                "case_id": str(spec["case_id"]),
                "case_token": str(spec["case_id"]).split(":")[1],
                "frontier_paths": list(frontier_paths(capture["trees"][0], selection["frontier"], max_depth=3)),
                "layout_context": {"module_count": int(capture["encoded"].module_present[0].sum().item()),
                                   "context_id": str(spec["context_id"])},
                "split": "train",
                **{key: selection[key] for key in ("frontier", "K", "unsupported_at_budget", "predicted_work_fraction", "predicted_role_distortion", "selection_scope")},
                "target_values_opened": False,
                "geometry_sampling": dict(input_candidate.evidence_manifest),
            })
        if {row["case_id"] for row in held_rankings} != {
            "active_packet:case0319:Re50", "active_packet:case0349:Re70"
        }:
            raise RuntimeError("Held candidate ranking differs from locked predeclaration.")

        # The cumulative 2,048-call native budget has little room after two
        # preserved failed/replayed fits.  Same-weight controls are therefore
        # measured only on the two predeclared held families after this
        # ranking freeze; do not replay them on all ten fit families.
        selected_controls: list[dict[str, Any]] = []

        candidate_count = 10 * 26 * 2
        planned_forward_ledger = {
            "run1804_setup_captures_G_and_P": 4,
            "stage_b_full_access_G_and_P_10_cases": 40,
            "stage_b_frontier_cut_table_10_cases_26_cuts_2_states": candidate_count,
            "additional_q2048_parity_repeats_2_cases_2_arms_2_states": 8,
            "fit_case_same_weight_control_replays": 0,
            "held_target_free_input_rank_reads": 2,
        }
        planned_forward_ledger["fit_phase_planned_complete_native_state_forwards"] = sum(
            planned_forward_ledger.values()
        )
        actual_forward_ledger = _summarize_forward_attempt_ledger(forward_attempt_path)
        if actual_forward_ledger["attempted_native_state_forwards"] > 2048:
            raise RuntimeError("Thermal Stage-B/C exceeded the 2,048 attempted native-state-forward cap.")

        g_hash = _sha256(runtime["g_path"])
        p_hash = _sha256(runtime["p_path"])
        frozen_model = {
            "schema_version": 2,
            "run_id": "Run_1509_20260928_active_packet_organization",
            "selected_forward_arm": "G",
            "forward_update": int(args.forward_update),
            "physical_model_state": runtime["g_saved"]["physical_model_state"],
            "organizer_state": organizer_with_utility.state_dict(),
            "organizer_dimensions": {
                "state_dim": int(measured[0]["capture"]["encoded"].module_tokens.shape[-1]),
                "module_feature_dim": int(measured[0]["capture"]["encoded"].module_features.shape[-1]),
                "environment_feature_dim": int(measured[0]["capture"]["encoded"].env_features.shape[-1]),
                "hidden_dim": 96,
                "role_count": 8,
                "frontier_role_count": len(role_rows),
            },
            "source_checkpoint_sha256": _sha256(runtime["source_path"]),
            "forward_checkpoint_sha256": g_hash,
            "control_checkpoint_sha256": p_hash,
            "utility_updates": int(args.utility_updates),
            "utility_optimizer_attempts_before_this_fit": 200,
            "utility_optimizer_attempts_including_this_fit": 200 + int(args.utility_updates),
            "stage_b_role_scales": role_scales_np.tolist(),
            "numerical_allowance_physical_units": numerical_allowance.tolist(),
            "inverse_role_tolerance": inverse_tolerance.tolist(),
            "budget_fractions": budgets,
            "selected_extra_route": extra_route,
            "role_schema": role_rows,
            "dataset_sha256": runtime["dataset_sha256"],
            "fit_family_ids": [item["case_id"] for item in measured],
            "held_forward_case_ids": [row["case_id"] for row in held_rankings],
            "selection_status": "research_only_until_held_outputs_are_measured",
        }
        frozen_path = output / f"frozen_G_u{args.forward_update:04d}_stageC.pt"
        torch.save(frozen_model, frozen_path)
        frozen_hash = _sha256(frozen_path)
        fit_freeze = {
            "format_version": 1,
            "case_rankings_frozen": True,
            "inverse_outcomes_consulted": False,
            "selected_forward_arm": "G",
            "selected_forward_update": int(args.forward_update),
            "selected_forward_checkpoint": str(frozen_path),
            "selected_forward_checkpoint_sha256": frozen_hash,
            "selected_physical_checkpoint": str(runtime["g_path"]),
            "selected_physical_checkpoint_sha256": g_hash,
            "control_checkpoint": str(runtime["p_path"]),
            "control_checkpoint_sha256": p_hash,
            "source_checkpoint_sha256": _sha256(runtime["source_path"]),
            "selected_organizer_state_sha256": _state_hash(organizer_with_utility.state_dict()),
            "utility_optimizer_attempts_before_this_fit": 200,
            "utility_optimizer_attempts_including_this_fit": 200 + int(args.utility_updates),
            "route_pilot": runtime["route_pilot"],
            "selected_extra_route": extra_route,
            "primary_budgets": budgets,
            "role_schema": role_rows,
            "role_scales_train_only": role_scales_np.tolist(),
            "numerical_allowance_physical_units": numerical_allowance.tolist(),
            "role_tolerance": inverse_tolerance.tolist(),
            "utility_fit_case_ids": [item["case_id"] for item in measured],
            "fit_case_rankings": case_selection,
            "held_case_rankings": held_rankings,
            "planned_native_state_forward_ledger": planned_forward_ledger,
            "actual_native_state_forward_ledger_at_freeze": actual_forward_ledger,
            "selection_rule": "select least canonical work among input-only predicted-adequate cuts; if unsupported, freeze least-risk research cut",
            "measured_gate": "RMSE <= 1.10 * same-student full-access RMSE + per-role q512/q2048 physical-unit allowance",
            "target_values_opened": False,
        }
        stage_b_table = {
            "format_version": 2,
            "dataset_path": str(runtime["dataset_root"]),
            "dataset_sha256": runtime["dataset_sha256"],
            "source_checkpoint": str(runtime["source_path"]),
            "source_checkpoint_sha256": _sha256(runtime["source_path"]),
            "G_checkpoint": str(runtime["g_path"]),
            "G_checkpoint_sha256": g_hash,
            "P_checkpoint": str(runtime["p_path"]),
            "P_checkpoint_sha256": p_hash,
            "G_and_P_forward_update": int(args.forward_update),
            "primary_budgets": budgets,
            "route_pilot": runtime["route_pilot"],
            "frontier_cut_count_per_fit_case": 26,
            "complete_native_candidate_state_forwards_for_cut_table": candidate_count,
            "old_atlas_families": [item["case_id"] for item in measured if item["stencil"].evidence_manifest.get("response_source") == "original_11_state_train_atlas_subset"],
            "new_reference_fit_families": [item["case_id"] for item in measured if item["stencil"].evidence_manifest.get("response_source") == "new_local_analytic_wake_i_plus_attempt"],
            "fit_case_ids": [item["case_id"] for item in measured],
            "held_case_ids": [row["case_id"] for row in held_rankings],
            "response_state_scope": "baseline_plus_i_plus for all Stage B labels; original 11-state atlases remain separate diagnostics",
            "role_schema": role_rows,
            "role_scales_train_only": role_scales_np.tolist(),
            "numerical_allowance_physical_units": numerical_allowance.tolist(),
            "parity_cases_and_arms": ["first fit case G/P", "last fit case G/P"],
            "cases": [{
                "case_id": item["case_id"],
                "split": "train",
                "baseline_source": item["stencil"].baseline.source.value,
                "variant_sources": {name: value.source.value for name, value in item["stencil"].variants.items()},
                "module_count": int(item["capture"]["encoded"].module_present[0].sum().item()),
                "cut_rows": item["cut_records"],
            } for item in measured],
            "same_weight_controls": selected_controls,
            "same_weight_control_scope": "fit-family controls deferred; only the two predeclared held families receive post-freeze same-weight controls",
            "planned_complete_native_state_forward_ledger": planned_forward_ledger,
            "actual_native_state_forward_ledger_at_fit_freeze": actual_forward_ledger,
        }
        summary = {
            "status": "case_rankings_frozen_before_held_reference_outputs",
            "device": str(device),
            "physical_gpu": 2 if device.type == "cuda" else None,
            "elapsed_seconds": time.monotonic() - started,
            "solver_attempts_in_this_runner": 0,
            "forward_optimizer_updates": {"G": int(args.forward_update), "P": int(args.forward_update)},
            "stage_c_utility_updates": int(args.utility_updates),
            "cumulative_utility_optimizer_attempts_including_two_preserved_failed_fit_replays": 200 + int(args.utility_updates),
            "stage_c_history_rows": len(history),
            "utility_nonzero_gradient_updates": sum(row["utility_gradient_l2"] > 0 for row in history),
            "fit_case_ids": [item["case_id"] for item in measured],
            "held_input_only_rankings": held_rankings,
            "role_schema": role_rows,
            "role_scales_train_only": role_scales_np.tolist(),
            "numerical_allowance_physical_units": numerical_allowance.tolist(),
            "role_tolerance": inverse_tolerance.tolist(),
            "fit_selection": case_selection,
            "same_weight_intervention_results": selected_controls,
            "frozen_forward_checkpoint": str(frozen_path),
            "frozen_forward_checkpoint_sha256": frozen_hash,
            "stage_c_fit_frozen": str(fit_freeze_path),
            "candidate_native_state_forwards_cut_table": candidate_count,
            "planned_complete_native_state_forward_ledger": planned_forward_ledger,
            "actual_native_state_forward_ledger": actual_forward_ledger,
            "inverse_outcomes_consulted": False,
        }
        # Validate all fit-phase JSON payloads before writing the immutable
        # ranking freeze, so a serializer error cannot leave a partial freeze.
        json.dumps(fit_freeze, sort_keys=True, allow_nan=False)
        json.dumps(stage_b_table, sort_keys=True, allow_nan=False)
        json.dumps(summary, sort_keys=True, allow_nan=False)
        _write_json(fit_freeze_path, fit_freeze, immutable=True)
        _write_json(stage_b_path, stage_b_table, immutable=True)
        _write_json(output / "stage_c_fit_summary.json", summary, immutable=True)
        _finish_forward_phase()
        return summary

    if args.phase != "held-eval":
        raise ValueError(f"Unsupported Stage B/C phase {args.phase!r}.")
    fit_freeze_path = output / "stage_c_fit_frozen.json"
    if not fit_freeze_path.is_file():
        raise FileNotFoundError("Held evaluation requires the hash-bound pre-output Stage-C freeze.")
    fit_freeze = json.loads(fit_freeze_path.read_text(encoding="utf-8"))
    if fit_freeze.get("case_rankings_frozen") is not True or fit_freeze.get("inverse_outcomes_consulted") is not False:
        raise RuntimeError("Held outcomes cannot be opened before an input-only case-rank freeze.")
    frozen_path = Path(fit_freeze["selected_forward_checkpoint"])
    if _sha256(frozen_path) != fit_freeze["selected_forward_checkpoint_sha256"]:
        raise RuntimeError("Selected forward utility checkpoint hash no longer matches its freeze.")
    held_stencils = _load_coverage_families(
        calls_path=args.coverage_manifest,
        predeclared_path=args.predeclared_manifest,
        partition="held_out_train_development",
        sampling_seed=args.panel_seed + 211,
    )
    if {item.physical_family_id for item in held_stencils} != set(fit_freeze["held_case_ids"] if "held_case_ids" in fit_freeze else [row["case_id"] for row in fit_freeze["held_case_rankings"]]):
        raise RuntimeError("Held response evidence does not match the already-frozen case rankings.")
    runtime = _load_stage_runtime(args, device, held_stencils[0])
    utility_payload = torch.load(frozen_path, map_location="cpu", weights_only=False)
    dimensions = utility_payload.get("organizer_dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("Frozen Stage-C checkpoint lacks the saved organizer dimensions.")
    utility = InputOnlyCoverOrganizer(
        state_dim=int(dimensions["state_dim"]),
        module_feature_dim=int(dimensions["module_feature_dim"]),
        environment_feature_dim=int(dimensions["environment_feature_dim"]),
        hidden_dim=int(dimensions["hidden_dim"]),
        role_count=int(dimensions["role_count"]), frontier_utility_enabled=True,
        frontier_role_count=int(dimensions["frontier_role_count"]),
    ).to(device=device)
    utility.load_state_dict(utility_payload["organizer_state"], strict=True)
    utility.eval().requires_grad_(False)
    frozen_g_hash = _state_hash(runtime["g_model"].state_dict())
    frozen_org_hash = _state_hash(utility.state_dict())
    held_rows = {row["case_id"]: row for row in fit_freeze["held_case_rankings"]}
    held_results: list[dict[str, Any]] = []
    export_cases: list[dict[str, Any]] = []
    export_arrays: dict[str, np.ndarray] = {}

    def record_export(stencil: Any, result: dict[str, Any], *, split: str) -> None:
        artifact = result.pop("field_artifact")
        arrays = artifact.pop("arrays_npz")
        overlap = set(export_arrays).intersection(arrays)
        if overlap:
            raise RuntimeError(f"Duplicate forward NPZ keys: {sorted(overlap)[:3]}")
        export_arrays.update(arrays)
        response = dict(artifact["finite_response"])
        if response.get("available"):
            response["roles"] = [
                {**role, "weights_key": role.get("finite_weights_key", role["weights_key"]),
                 "valid_mask_key": role.get("finite_valid_mask_key", role.get("valid_mask_key"))}
                for role in response["roles"]
            ]
        export_cases.append({
            "case_id": stencil.physical_family_id,
            "split": split,
            "selection_label": "pending_representative_or_difficult_assignment",
            "selection_rule": None,
            "arrays": artifact["arrays"],
            "roles": artifact["roles"],
            "fields": artifact["fields"],
            "unclipped_residuals": artifact["unclipped_residuals"],
            "finite_response": response,
            "supports": artifact["supports"],
            "work": artifact["work"],
            "comparison_metrics": artifact["comparison_metrics"],
            "geometry": artifact["geometry"],
            "case_provenance": {
                "evidence_manifest": dict(stencil.evidence_manifest),
                "baseline_record_id": stencil.baseline.record_id,
                "baseline_source": stencil.baseline.source.value,
                "variant_sources": {name: value.source.value for name, value in stencil.variants.items()},
                "target_values_opened_after_forward_freeze": True,
            },
        })

    for index, stencil in enumerate(held_stencils):
        selected = held_rows[stencil.physical_family_id]
        selected_paths = tuple(selected["frontier_paths"])
        try:
            control_result = _evaluate_one_case(
                stencil,
                runtime=runtime,
                cut=selected["frontier"],
                frontier_pattern_paths=selected_paths,
                device=device,
                extra_route=runtime["extra_route"],
                seed=args.seed + 900 + index,
                capture_arrays=True,
            )
        except UnsupportedFrontierPatternError as exc:
            allowances = np.asarray(fit_freeze["numerical_allowance_physical_units"], dtype=np.float64)
            held_results.append({
                "case_id": stencil.physical_family_id,
                "split": "train_development",
                "pre_output_frozen_frontier": selected["frontier"],
                "pre_output_K": selected["K"],
                "pre_output_frontier_paths": list(selected_paths),
                "selected_measurement_status": "unsupported_frontier_pattern",
                "unsupported_state": exc.state_label,
                "unsupported_reason": str(exc),
                "unsupported_at_budget": bool(selected["unsupported_at_budget"]),
                "predicted_work_fraction": selected["predicted_work_fraction"],
                "predicted_role_distortion": selected["predicted_role_distortion"],
                "adequate_by_1p10_plus_numerical_allowance": False,
                "per_role_adequacy": [False] * len(allowances),
                "all_26_frontier_measurements_after_rank_freeze": "not_measured_due_cumulative_native_forward_cap",
                "same_weight_controls": "unavailable_selected_pattern_not_supported",
                "baseline_source": stencil.baseline.source.value,
                "variant_sources": {name: record.source.value for name, record in stencil.variants.items()},
                "targets_opened_after_freeze": True,
            })
            continue

        chosen_errors = np.asarray(control_result["interventions"]["G_selected_cut"]["role_rmse"], dtype=np.float64)
        full_errors = np.asarray(control_result["full_G"]["role_rmse"], dtype=np.float64)
        allowances = np.asarray(fit_freeze["numerical_allowance_physical_units"], dtype=np.float64)
        adequate = chosen_errors <= 1.10 * full_errors + allowances
        field_artifact_result = control_result
        held_results.append({
            "case_id": stencil.physical_family_id,
            "split": "train_development",
            "pre_output_frozen_frontier": selected["frontier"],
            "pre_output_K": selected["K"],
            "pre_output_frontier_paths": list(selected_paths),
            "selected_measurement_status": "measured_selected_cut",
            "selected_frontier_pattern_supported_on_current_state_tree": True,
            "unsupported_at_budget": bool(selected["unsupported_at_budget"]),
            "predicted_work_fraction": selected["predicted_work_fraction"],
            "predicted_role_distortion": selected["predicted_role_distortion"],
            "selected_measured_role_rmse": chosen_errors.tolist(),
            "same_student_full_role_rmse": full_errors.tolist(),
            "signed_selected_minus_full": (chosen_errors - full_errors).tolist(),
            "adequate_by_1p10_plus_numerical_allowance": bool(adequate.all()),
            "per_role_adequacy": adequate.tolist(),
            "all_26_frontier_measurements_after_rank_freeze": "not_measured_due_cumulative_native_forward_cap",
            "same_weight_controls": {key: value for key, value in control_result["interventions"].items()},
            "full_G": control_result["full_G"],
            "full_P": control_result["full_P"],
            "baseline_source": stencil.baseline.source.value,
            "variant_sources": {name: record.source.value for name, record in stencil.variants.items()},
            "targets_opened_after_freeze": True,
        })
        record_export(stencil, field_artifact_result, split="predeclared_held_train_development")

    # Compare the first predeclared same-M pair of distinct physical layouts
    # using each case's already-frozen input-only Stage-C choice. This is a
    # post-freeze measurement only; it cannot change the selector.
    fit_stencils_for_pair = _load_fit_stencils(
        args.atlas_dir.expanduser().resolve(), seed=args.panel_seed,
        calls_path=args.coverage_manifest, predeclared_path=args.predeclared_manifest,
    )
    fit_rankings = {row["case_id"]: row for row in fit_freeze["fit_case_rankings"]}
    same_m_pair = None
    for left_index, left in enumerate(fit_stencils_for_pair):
        left_modules = tuple(module for module in left.baseline.design.modules if module.active)
        left_xy = np.asarray([module.position_xy for module in left_modules], dtype=np.float64)
        for right in fit_stencils_for_pair[left_index + 1:]:
            right_modules = tuple(module for module in right.baseline.design.modules if module.active)
            right_xy = np.asarray([module.position_xy for module in right_modules], dtype=np.float64)
            if len(left_modules) == len(right_modules) and left_xy.shape == right_xy.shape and not np.allclose(left_xy, right_xy):
                same_m_pair = (left, right)
                break
        if same_m_pair is not None:
            break
    same_m_manifest: dict[str, Any]
    if same_m_pair is not None:
        pair_rows = []
        for pair_index, stencil in enumerate(same_m_pair):
            selected = fit_rankings[stencil.physical_family_id]
            result = _evaluate_one_case(
                stencil, runtime=runtime, cut=selected["frontier"],
                frontier_pattern_paths=selected["frontier_paths"], device=device,
                extra_route=runtime["extra_route"], seed=args.seed + 1200 + pair_index,
                capture_arrays=True,
            )
            pair_rows.append({"case_id": stencil.physical_family_id, "K": selected["K"],
                              "frontier_paths": selected["frontier_paths"],
                              "same_weight_controls": result["interventions"]})
            record_export(stencil, result, split="train_fit_development_same_M_graph_comparison")
        same_m_manifest = {
            "case_ids": [row["case_id"] for row in pair_rows],
            "module_count": len(tuple(module for module in same_m_pair[0].baseline.design.modules if module.active)),
            "selection_rule": "first pair in locked 10-fit-family order with equal active M and distinct physical module-center arrays",
            "same_m_verified": True,
            "measured_graph_rows": pair_rows,
        }
    else:
        same_m_manifest = {"unavailable_reason": "No two predeclared fit families had equal active M and distinct physical module centers."}
    if _state_hash(runtime["g_model"].state_dict()) != frozen_g_hash:
        raise RuntimeError("Held measurement changed the selected G physical state.")
    if _state_hash(utility.state_dict()) != frozen_org_hash:
        raise RuntimeError("Held measurement changed the frozen G utility state.")
    actual_forward_ledger = _summarize_forward_attempt_ledger(forward_attempt_path)
    if actual_forward_ledger["attempted_native_state_forwards"] > 2048:
        raise RuntimeError("Thermal Stage-B/C exceeded the 2,048 attempted native-state-forward cap.")
    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    npz_path = evaluation_dir / "forward_fields.npz"
    np.savez_compressed(npz_path, **export_arrays)
    if export_cases:
        representative_id = next(
            (row["case_id"] for row in held_results if row["selected_measurement_status"] == "measured_selected_cut"),
            export_cases[0]["case_id"],
        )
        difficult_case = max(
            export_cases,
            key=lambda row: max(row["comparison_metrics"].get("G_stageC", {}).values(), default=float("-inf")),
        )
        difficult_id = difficult_case["case_id"]
        for case_row in export_cases:
            labels = []
            rules = []
            if case_row["case_id"] == representative_id:
                labels.append("representative")
                rules.append("first predeclared held family with a measured selected frontier")
            if case_row["case_id"] == difficult_id:
                labels.append("difficult")
                rules.append("maximum worst-role weighted native RMSE for G_stageC among exported cases")
            case_row["selection_label"] = labels or ["same_M_graph_comparison"]
            case_row["selection_rule"] = "; ".join(rules) or "selected by locked same-M graph comparison rule"
    source_hash = _sha256(runtime["source_path"])
    g_hash = _sha256(runtime["g_path"])
    p_hash = _sha256(runtime["p_path"])
    forward_manifest = {
        "format_version": 1,
        "provenance": {
            "run_id": "Run_1509_20260928_active_packet_organization",
            "stage_c_freeze": str(frozen_path),
            "stage_c_freeze_sha256": fit_freeze["selected_forward_checkpoint_sha256"],
            "selection_freeze": str(fit_freeze_path),
            "selection_freeze_sha256": _sha256(fit_freeze_path),
            "source_checkpoint": str(runtime["source_path"]),
            "source_checkpoint_sha256": source_hash,
            "source_checkpoint_identity": "Run1804 e4738 Dense field checkpoint",
            "dataset_path": str(runtime["dataset_root"]),
            "dataset_sha256": runtime["dataset_sha256"],
            "split": "predeclared train development families; not independent test evidence",
            "forward_checkpoints": {
                "G_stageC": {"path": str(frozen_path), "sha256": fit_freeze["selected_forward_checkpoint_sha256"],
                             "physical_checkpoint": str(runtime["g_path"]), "physical_checkpoint_sha256": g_hash,
                             "update": int(args.forward_update)},
                "full_G_stageC": {"same_student_as": "G_stageC", "path": str(frozen_path),
                                  "sha256": fit_freeze["selected_forward_checkpoint_sha256"]},
                "P_u0200": {"path": str(runtime["p_path"]), "sha256": p_hash,
                            "update": int(args.forward_update)},
                "full_P_u0200": {"same_student_as": "P_u0200", "path": str(runtime["p_path"]),
                                 "sha256": p_hash},
            },
            "native_coordinates": "saved role receiver coordinates in physical x/y data coordinates; role-specific units are recorded in roles",
            "data_units": "dataset-declared role units; no SI conversion is documented",
            "generator": "native differentiable Thermal operator reloaded from e4738; references retain stored/local source identity per case",
            "solver_attempts_in_evaluation": 0,
            "inverse_outcomes_consulted": False,
        },
        "comparison_groups": [
            {"name": "matched_G_sparse_vs_full", "arms": ["G_stageC", "full_G_stageC"], "same_physical_checkpoint": True},
            {"name": "matched_P_sparse_vs_full", "arms": ["P_u0200", "full_P_u0200"], "same_physical_checkpoint": True},
            {"name": "same_weight_G_controls", "arms": ["root_union", "geometry_only", "rewired"], "physical_checkpoint": "G_stageC"},
            {"name": "independently_trained_P", "arms": ["P_u0200", "P_Gwork"], "physical_checkpoint": "P_u0200",
             "note": "P_Gwork changes requested canonical work fractions to the measured G route work; it retains P memberships."},
        ],
        "organization_comparisons": {
            "same_module_count": same_m_manifest,
            "valid_transition": {
                "case_id": "active_packet:case0319:Re70",
                "input_change": "predeclared i_plus module-center perturbation, baseline to x+0.10 response",
                "validity_evidence": "both reference records are converged and role-aligned; physical module IDs and baseline/i_plus centers are saved in the case NPZ",
                "source_classification": "mixed stored baseline and local reference-solver i_plus retain their source labels; same-generator pairing is documented in the predeclared coverage manifest",
            },
        },
        "array_file": str(npz_path),
        "array_file_sha256": _sha256(npz_path),
        "native_state_forward_ledger": actual_forward_ledger,
        "planned_held_native_state_forwards": {
            "runtime_setup_full_baseline_captures_G_and_P": 4,
            "selected_only_held_same_weight_control_cases": 16 * len(held_stencils),
            "same_M_graph_comparison_cases": 32 if same_m_pair is not None else 0,
            "held_26_cut_table": 0,
            "planned_total_including_runtime_setup": 4 + 16 * len(held_stencils) + (32 if same_m_pair is not None else 0),
            "runtime_setup_explanation": "Two arms (G/P), each run on both baseline and i_plus states of the first held stencil.",
            "held_26_cut_table_reason": "not measured due cumulative 2,048 native-state-forward cap; ranking frozen before held outputs",
        },
        "cases": export_cases,
    }
    forward_manifest_path = evaluation_dir / "forward_fields.json"
    _write_json(forward_manifest_path, forward_manifest, immutable=True)
    held_payload = {
        "format_version": 1,
        "fit_freeze_sha256": _sha256(fit_freeze_path),
        "selected_checkpoint_sha256": fit_freeze["selected_forward_checkpoint_sha256"],
        "selected_physical_checkpoint_sha256": fit_freeze["selected_physical_checkpoint_sha256"],
        "G_state_hash_unchanged": True,
        "organizer_state_hash_unchanged": True,
        "held_case_results": held_results,
        "same_M_graph_comparison": same_m_manifest,
        "complete_native_state_forward_ledger": {
            "runtime_setup_full_baseline_captures_G_and_P": 4,
            "held_selected_only_same_weight_controls_two_cases": 16 * len(held_stencils),
            "same_M_graph_comparison_two_fit_cases": 32 if same_m_pair is not None else 0,
            "held_26_cut_table": 0,
            "planned_total_including_runtime_setup": 4 + 16 * len(held_stencils) + (32 if same_m_pair is not None else 0),
            "runtime_setup_explanation": "Two arms (G/P), each run on both baseline and i_plus states of the first held stencil.",
            "held_26_cut_table_reason": "not measured due cumulative cap; Stage-B full cut table used only ten fit families and the held selection was frozen beforehand",
            "accounting_note": "Each baseline or i_plus native operator evaluation counts once. Actual cumulative starts, including failed and replayed attempts, are recorded below.",
        },
        "forward_fields_manifest": str(forward_manifest_path),
        "forward_fields_manifest_sha256": _sha256(forward_manifest_path),
        "forward_fields_npz": str(npz_path),
        "forward_fields_npz_sha256": _sha256(npz_path),
        "actual_native_state_forward_ledger": actual_forward_ledger,
        "inverse_outcomes_consulted": False,
    }
    _write_json(output / "held_forward_evaluation.json", held_payload, immutable=True)
    selection_status = (
        "selected_forward_adequate_on_held_train_development"
        if all(row["adequate_by_1p10_plus_numerical_allowance"] for row in held_results)
        else "research_only_forward_adequacy_failed_on_held_train_development"
    )
    final_selection = {
        **fit_freeze,
        "status": selection_status,
        "held_forward_evaluation": str(output / "held_forward_evaluation.json"),
        "held_forward_evaluation_sha256": _sha256(output / "held_forward_evaluation.json"),
        "held_case_results": held_results,
        "same_M_graph_comparison": same_m_manifest,
        "inverse_outcomes_consulted": False,
        "deployability": "train-family development only; prior test cohort exposure remains development evidence",
    }
    _write_json(output / "forward_selection_frozen.json", final_selection, immutable=True)
    summary = {
        "status": selection_status,
        "elapsed_seconds": time.monotonic() - started,
        "solver_attempts_in_this_phase": 0,
        "held_family_count": len(held_results),
        "held_candidate_native_state_forwards": 4 + 16 * len(held_stencils) + (32 if same_m_pair is not None else 0),
        "held_26_cut_table_status": "not_measured_due_cumulative_native_forward_cap",
        "actual_native_state_forward_ledger": actual_forward_ledger,
        "held_results": held_results,
        "same_M_graph_comparison": same_m_manifest,
        "forward_selection_frozen": str(output / "forward_selection_frozen.json"),
        "forward_fields_manifest": str(forward_manifest_path),
        "inverse_outcomes_consulted": False,
    }
    _write_json(output / "held_forward_summary.json", summary, immutable=True)
    _finish_forward_phase()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "held-eval"), default="fit")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--g-checkpoint", type=Path, default=DEFAULT_G_CHECKPOINT)
    parser.add_argument("--p-checkpoint", type=Path, default=DEFAULT_P_CHECKPOINT)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--native-attempt-ledger",
        type=Path,
        default=None,
        help="Append to an existing cumulative native-state-forward attempt ledger.",
    )
    parser.add_argument("--coverage-manifest", type=Path, default=DEFAULT_COVERAGE_MANIFEST)
    parser.add_argument("--predeclared-manifest", type=Path, default=DEFAULT_PREDECLARED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--utility-updates", type=int, choices=(100, 200, 300, 400), default=300)
    parser.add_argument("--forward-update", type=int, choices=(100, 200, 300, 400, 500, 600), default=200)
    parser.add_argument("--panel-seed", type=int, default=2317)
    parser.add_argument("--seed", type=int, default=2317)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
