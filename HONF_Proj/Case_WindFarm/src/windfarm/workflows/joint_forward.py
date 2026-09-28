"""Bounded, matched WindFarm full-access and typed-packet forward training.

This workflow reuses the frozen Run2103 Dense model, native WindFarm fields,
the verified train-only G2 supervision, and the input-only typed organizer.
The hard plan supplies every physical value and physical-model gradient. The
soft plan is used only for organizer gradients and a differentiable structural
work proxy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    INTERACTION_MECHANISMS,
    MechanismPlan,
)
from honf_forward_core.interface_fields.input_cover_organizer import (
    InputOnlyCoverOrganizer,
)
from honf_forward_core.interface_fields.native_joint_shadow import (
    hard_value_soft_organizer_forward,
)
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import load_trusted_checkpoint, select_device, set_seed
from torch.nn.utils import clip_grad_norm_

from ..data import WindFarmNativeView, case_batch
from ..geometry import ENV_TOKEN_SHAPE, support_weights
from ..model import WindFarmForwardModel, build_windfarm_forward_config
from ..normalization import VelocityNormalizer
from ..splits import GroupSplit, make_group_split
from ..study_spatial import downstream_envelope, native_coordinates
from . import native_cover_organizer as native_cover
from . import native_cover_organizer_fit as typed_fit
from .native_cover_panel import (
    ORGANIZER_DEVELOPMENT_LAYOUT_INDICES,
    freeze_organizer_layout_split,
    select_training_layouts,
)
from .train_forward import _compact_metadata, _json_write, _resolved_path

ROLE_NAMES = (
    "volume",
    "hub_slab",
    "downstream_envelope",
    "near_turbine",
    "background",
)
DEFAULT_ROLE_QUERY_COUNTS = {
    "volume": 4096,
    "hub_slab": 1024,
    "downstream_envelope": 1024,
    "near_turbine": 1024,
    "background": 1024,
}
DEFAULT_AUDIT_QUERY_COUNTS = {
    "volume": 1024,
    "hub_slab": 256,
    "downstream_envelope": 256,
    "near_turbine": 256,
    "background": 256,
}
FROZEN_RUN_DIRECTORY = "Run_2103_20260913_135849_windfarm_dense_pairwise_b16_q8192"
FROZEN_CHECKPOINT_EPOCH = 2475
HUB_SLAB_HALF_WIDTH_D = 0.5
DOWNSTREAM_LENGTH_D = 10.0
DOWNSTREAM_RADIUS_D = 1.5
NEAR_TURBINE_RADIUS_D = 1.5
ROLE_RISK_FORMULA = "g_r = role_reference_mse_r / role_budget_mps_r**2 - 1"
MAX_PAIR_WALL_SECONDS = 24 * 60 * 60
DEFAULT_PAIR_WALL_SECONDS = 20 * 60 * 60
MAX_PAIR_ATTEMPTED_OPTIMIZER_CALLS = 18_000
MAX_OPTIMIZER_CALLS_PER_ARM = 6_000
MAX_PHYSICAL_UPDATES_PER_ARM = 6_000
MAX_COMPLETE_NATIVE_EPOCHS_PER_ARM = 500
DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES = 64 * 1024**3
WORK_INTERPRETATION = (
    "mechanism-balanced differentiable proxy guides structure only; hard exact native pair rows, "
    "packet counts, executor rows, padding, and synchronized latency are reported separately"
)


@dataclass(frozen=True)
class NativeRoleSample:
    """Fresh native targets with one disjoint query slice per role estimate."""

    coordinates_D: np.ndarray
    target_mps: np.ndarray
    flat_indices: np.ndarray
    role_slices: Mapping[str, slice]
    role_support_volume_m3: Mapping[str, float]
    role_sample_counts: Mapping[str, int]
    geometry_sha256: str


@dataclass(frozen=True)
class NativeRoleCatalogue:
    """Immutable role geometry and sampling measures for one native layout."""

    layout_index: int
    geometry_sha256: str
    coordinates_D: np.ndarray
    role_indices: Mapping[str, np.ndarray]
    role_cdf: Mapping[str, np.ndarray]
    role_support_volume_m3: Mapping[str, float]
    cached_nbytes: int


class NativeRoleCatalogueCache:
    """Reuse exact row geometry while gathering targets from each mmap.

    WindFarm repeats each layout at several wind directions, and each direction
    has its own native axes and active hub coordinates. Cache identity therefore
    includes the exact geometry hash, while the row map only guards against a
    single source row mutating during a run. The byte-bounded LRU keeps the
    reusable NumPy arrays under a fixed host-memory budget.
    """

    def __init__(self, *, max_cached_bytes: int = DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES) -> None:
        if isinstance(max_cached_bytes, bool) or not isinstance(max_cached_bytes, int) or max_cached_bytes <= 0:
            raise ValueError("role catalogue cache byte limit must be a positive integer")
        self.max_cached_bytes = int(max_cached_bytes)
        self._catalogues: OrderedDict[tuple[int, str], NativeRoleCatalogue] = OrderedDict()
        self._geometry_by_row: dict[int, tuple[int, str]] = {}
        self._cached_bytes = 0
        self.peak_cached_bytes = 0
        self.build_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.oversize_bypass_count = 0

    @staticmethod
    def _geometry_sha256(case: Any) -> str:
        digest = hashlib.sha256()
        run = case.run
        metadata = {
            "layout_index": int(case.layout_index),
            "shape_nxyz": [int(run.nx), int(run.ny), int(run.nz)],
            "diameter_m": float(case.diameter_m),
            "hub_height_m": float(case.hub_height_m),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for value in (run.x_m, run.y_m, run.z_m):
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(memoryview(array).cast("B"))
        active = np.asarray(case.module_centers, dtype=np.float32)[
            np.asarray(case.module_present) > 0.5
        ]
        digest.update(memoryview(np.ascontiguousarray(active)).cast("B"))
        return digest.hexdigest()

    def get(self, case: Any) -> NativeRoleCatalogue:
        layout_index = int(case.layout_index)
        geometry_sha256 = self._geometry_sha256(case)
        row_index_value = getattr(case, "index", None)
        row_index = None if row_index_value is None else int(row_index_value)
        identity = (layout_index, geometry_sha256)
        known_identity = None if row_index is None else self._geometry_by_row.get(row_index)
        if known_identity is not None and known_identity != identity:
            raise RuntimeError(
                f"native role row {row_index} changed geometry after catalogue construction"
            )
        if row_index is not None:
            self._geometry_by_row[row_index] = identity
        key = (layout_index, geometry_sha256)
        cached = self._catalogues.get(key)
        if cached is not None:
            if cached.geometry_sha256 != geometry_sha256 or cached.layout_index != layout_index:
                raise RuntimeError("cached native role geometry identity guard failed")
            self._catalogues.move_to_end(key)
            self.hit_count += 1
            return cached
        self.miss_count += 1

        coordinates, masks, cell_weights = _native_role_masks(case)
        coordinates = coordinates.astype(np.float32, copy=False)
        coordinates.setflags(write=False)
        role_indices: dict[str, np.ndarray] = {}
        role_cdf: dict[str, np.ndarray] = {}
        role_support: dict[str, float] = {}
        cached_arrays = [coordinates]
        for role in ROLE_NAMES:
            valid = np.flatnonzero(masks[role])
            if valid.size == 0:
                raise ValueError(f"native layout {layout_index} has no cells in protected role {role!r}")
            weights = cell_weights[valid]
            total = float(weights.sum(dtype=np.float64))
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(f"native role {role!r} has invalid quadrature support")
            # Store a normalized cumulative measure once. Searchsorted draws
            # each update in O(samples log cells), without re-normalizing a
            # multi-million-entry probability vector at every update.
            cdf = np.cumsum(weights / total, dtype=np.float64)
            cdf[-1] = 1.0
            cdf.setflags(write=False)
            role_cdf[role] = cdf
            if role != "volume":
                compact_indices = valid.astype(np.uint32, copy=False)
                compact_indices.setflags(write=False)
                role_indices[role] = compact_indices
                cached_arrays.append(compact_indices)
            role_support[role] = total
            cached_arrays.append(cdf)

        cached_nbytes = sum(int(array.nbytes) for array in cached_arrays)
        catalogue = NativeRoleCatalogue(
            layout_index=layout_index,
            geometry_sha256=geometry_sha256,
            coordinates_D=coordinates,
            role_indices=MappingProxyType(role_indices),
            role_cdf=MappingProxyType(role_cdf),
            role_support_volume_m3=MappingProxyType(role_support),
            cached_nbytes=cached_nbytes,
        )
        self.build_count += 1
        if catalogue.cached_nbytes > self.max_cached_bytes:
            self.oversize_bypass_count += 1
            return catalogue
        self._catalogues[key] = catalogue
        self._catalogues.move_to_end(key)
        self._cached_bytes += catalogue.cached_nbytes
        while self._cached_bytes > self.max_cached_bytes:
            _, evicted = self._catalogues.popitem(last=False)
            self._cached_bytes -= evicted.cached_nbytes
            self.eviction_count += 1
        self.peak_cached_bytes = max(self.peak_cached_bytes, self._cached_bytes)
        return catalogue

    def summary(self) -> dict[str, Any]:
        return {
            "catalogue_count": len(self._catalogues),
            "catalogue_build_count": self.build_count,
            "cache_hit_count": self.hit_count,
            "cache_miss_count": self.miss_count,
            "cache_eviction_count": self.eviction_count,
            "cache_oversize_bypass_count": self.oversize_bypass_count,
            "cache_capacity_bytes": self.max_cached_bytes,
            "cached_bytes": self._cached_bytes,
            "peak_cached_bytes": self.peak_cached_bytes,
            "cache_byte_measure": "sum of retained NumPy array nbytes; excludes Python metadata",
            "identities": [
                {
                    "layout_index": item.layout_index,
                    "geometry_sha256": item.geometry_sha256,
                    "native_cell_count": int(item.coordinates_D.shape[0]),
                    "cached_bytes": item.cached_nbytes,
                }
                for item in self._catalogues.values()
            ],
        }


_DEFAULT_ROLE_CATALOGUE_CACHE = NativeRoleCatalogueCache()


class RunResourceLedger:
    """Durable pair-wide caps for optimizer attempts and elapsed wall time."""

    def __init__(
        self,
        run_dir: Path,
        *,
        max_wall_seconds: int = DEFAULT_PAIR_WALL_SECONDS,
        max_attempted_optimizer_calls: int = MAX_PAIR_ATTEMPTED_OPTIMIZER_CALLS,
    ) -> None:
        if not 1 <= int(max_wall_seconds) <= MAX_PAIR_WALL_SECONDS:
            raise ValueError(f"pair wall limit must be within 1–{MAX_PAIR_WALL_SECONDS} seconds")
        if not 1 <= int(max_attempted_optimizer_calls) <= MAX_PAIR_ATTEMPTED_OPTIMIZER_CALLS:
            raise ValueError(
                "pair optimizer-attempt limit must be within "
                f"1–{MAX_PAIR_ATTEMPTED_OPTIMIZER_CALLS} calls"
            )
        self.path = Path(run_dir) / "resource_ledger.json"
        self.attempts_path = Path(run_dir) / "optimizer_attempts.jsonl"
        if self.path.exists() or self.attempts_path.exists():
            raise FileExistsError(
                f"joint forward resource ledger already exists at {self.path}; this path cannot resume"
            )
        self.max_wall_seconds = int(max_wall_seconds)
        self.max_attempted_optimizer_calls = int(max_attempted_optimizer_calls)
        self.started_monotonic = time.monotonic()
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.attempted_optimizer_calls = 0
        self.completed_optimizer_calls = 0
        self.attempted_optimizer_calls_by_arm = {"w_full": 0, "w_packet": 0}
        self.completed_optimizer_calls_by_arm = {"w_full": 0, "w_packet": 0}
        self.stop_reason: str | None = None
        self._persist("initialized")

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_monotonic)

    def block_reason(self, calls_needed: int = 1, *, arm: str | None = None) -> str | None:
        if arm is not None:
            if arm not in self.attempted_optimizer_calls_by_arm:
                raise ValueError(f"unknown joint-forward arm: {arm}")
            if self.attempted_optimizer_calls_by_arm[arm] + int(calls_needed) > MAX_OPTIMIZER_CALLS_PER_ARM:
                return f"{arm} attempted-optimizer-call ceiling reached at {MAX_OPTIMIZER_CALLS_PER_ARM}"
        if self.attempted_optimizer_calls + int(calls_needed) > self.max_attempted_optimizer_calls:
            return "pair attempted-optimizer-call ceiling reached"
        if self.elapsed_seconds() >= self.max_wall_seconds:
            return (
                f"shared pair wall ceiling reached at {self.max_wall_seconds}s "
                f"within the {MAX_PAIR_WALL_SECONDS}s overall round cap"
            )
        return None

    def can_begin_update(self, calls_needed: int, *, arm: str) -> tuple[bool, str | None]:
        reason = self.block_reason(calls_needed, arm=arm)
        self._persist("running", stop_reason=reason)
        return reason is None, reason

    def record_attempt(self, *, arm: str, update: int, optimizer: str) -> tuple[bool, str | None]:
        reason = self.block_reason(1, arm=arm)
        if reason is not None:
            self.stop_reason = reason
            self._persist("resource_stopped", stop_reason=reason)
            return False, reason
        self.attempted_optimizer_calls += 1
        self.attempted_optimizer_calls_by_arm[arm] += 1
        event = {
            "attempted_optimizer_call": self.attempted_optimizer_calls,
            "arm": str(arm),
            "scheduled_update": int(update),
            "optimizer": str(optimizer),
            "attempted_at_elapsed_seconds": self.elapsed_seconds(),
        }
        self.attempts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.attempts_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._persist("running")
        return True, None

    def record_completion(self, *, arm: str) -> None:
        if arm not in self.completed_optimizer_calls_by_arm:
            raise ValueError(f"unknown joint-forward arm: {arm}")
        self.completed_optimizer_calls += 1
        self.completed_optimizer_calls_by_arm[arm] += 1
        self._persist("running")

    def stop(self, reason: str) -> None:
        self.stop_reason = str(reason)
        self._persist("resource_stopped", stop_reason=self.stop_reason)

    def finalize(self, status: str) -> None:
        """Persist the terminal pair status after all arm artifacts are written."""
        if status not in {
            "complete", "early_stopped", "preflight_only_complete", "resource_stopped_or_incomplete"
        }:
            raise ValueError(f"invalid terminal resource-ledger status: {status}")
        self._persist(status)

    def snapshot(self, *, status: str = "running") -> dict[str, Any]:
        return {
            "status": status,
            "started_at_utc": self.started_utc,
            "elapsed_wall_seconds": self.elapsed_seconds(),
            "wall_limit_seconds": self.max_wall_seconds,
            "overall_round_wall_limit_seconds": MAX_PAIR_WALL_SECONDS,
            "round_wall_reserve_after_this_pair_seconds": max(
                0, MAX_PAIR_WALL_SECONDS - self.max_wall_seconds
            ),
            "wall_budget_scope": (
                "one shared W-full/W-packet pair ledger; includes preflight, reviews, and both arms"
            ),
            "attempted_optimizer_calls": self.attempted_optimizer_calls,
            "completed_optimizer_calls": self.completed_optimizer_calls,
            "attempted_optimizer_calls_by_arm": dict(self.attempted_optimizer_calls_by_arm),
            "completed_optimizer_calls_by_arm": dict(self.completed_optimizer_calls_by_arm),
            "attempted_optimizer_call_limit": self.max_attempted_optimizer_calls,
            "optimizer_call_limit_per_arm": MAX_OPTIMIZER_CALLS_PER_ARM,
            "physical_updates_per_arm_limit": MAX_PHYSICAL_UPDATES_PER_ARM,
            "complete_native_epoch_limit_per_arm": MAX_COMPLETE_NATIVE_EPOCHS_PER_ARM,
            "stop_reason": self.stop_reason,
            "attempt_log": str(self.attempts_path),
        }

    def _persist(self, status: str, *, stop_reason: str | None = None) -> None:
        if stop_reason is not None:
            self.stop_reason = stop_reason
        payload = self.snapshot(status=status)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)


class _PairWallLimitReached(RuntimeError):
    """A review reached the shared pair wall limit between native chunks."""


def _pair_terminal_status(arm_summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """Call a pair complete only when both arms reach the same update cursor."""

    if set(arm_summaries) != {"w_full", "w_packet"}:
        return "resource_stopped_or_incomplete"
    full = arm_summaries["w_full"]
    packet = arm_summaries["w_packet"]
    if int(full["scheduled_update_cursor"]) != int(packet["scheduled_update_cursor"]):
        return "resource_stopped_or_incomplete"
    statuses = {str(full["status"]), str(packet["status"])}
    if statuses == {"complete"}:
        return "complete"
    if statuses <= {"complete", "early_stopped"} and "early_stopped" in statuses:
        return "early_stopped"
    return "resource_stopped_or_incomplete"


def _check_pair_review_wall(resource_ledger: RunResourceLedger, *, stage: str) -> None:
    if resource_ledger.elapsed_seconds() >= resource_ledger.max_wall_seconds:
        raise _PairWallLimitReached(
            f"shared pair wall ceiling reached during {stage} at "
            f"{resource_ledger.elapsed_seconds():.1f}s / {resource_ledger.max_wall_seconds}s"
        )


def _canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rows_hash(rows: Sequence[int]) -> str:
    values = [int(row) for row in rows]
    return _canonical_json_hash(values)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _training_rows_without_held_layouts(
    training_rows: Sequence[int],
    layout_indices_by_row: Sequence[int] | np.ndarray,
    excluded_layouts: Sequence[int] = ORGANIZER_DEVELOPMENT_LAYOUT_INDICES,
) -> list[int]:
    """Remove whole frozen organizer-development layouts from new updates."""

    layouts = np.asarray(layout_indices_by_row, dtype=np.int64)
    excluded = set(map(int, excluded_layouts))
    result = [int(row) for row in training_rows if int(layouts[int(row)]) not in excluded]
    if not result:
        raise ValueError("excluding organizer-development layouts left no WindFarm training rows")
    if any(int(layouts[row]) in excluded for row in result):
        raise RuntimeError("held organizer-development layout entered the student training rows")
    return result


def _native_role_masks(case: Any) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Build geometry-only role masks and native centre-box quadrature masses."""

    run = case.run
    cell_count = int(run.cell_count)
    flat = np.arange(cell_count, dtype=np.int64)
    coords = native_coordinates(run, flat, float(case.diameter_m)).astype(np.float64)
    wx, wy, wz = support_weights(run.x_m, run.y_m, run.z_m)
    ix = flat % int(run.nx)
    iy = (flat // int(run.nx)) % int(run.ny)
    iz = flat // (int(run.nx) * int(run.ny))
    quadrature = (wx[ix] * wy[iy] * wz[iz]).astype(np.float64)
    if not np.isfinite(quadrature).all() or np.any(quadrature <= 0.0):
        raise ValueError("WindFarm native centre-box quadrature must be finite and positive")

    hub_height_D = float(case.hub_height_m) / float(case.diameter_m)
    hub_slab = np.abs(coords[:, 2] - hub_height_D) <= HUB_SLAB_HALF_WIDTH_D
    hubs = np.asarray(case.module_centers, dtype=np.float64)[np.asarray(case.module_present) > 0.5]
    if hubs.ndim != 2 or hubs.shape[0] < 1:
        raise ValueError("WindFarm role sampling requires at least one active turbine hub")
    downstream = downstream_envelope(coords.astype(np.float32), hubs.astype(np.float32))
    near = np.zeros(cell_count, dtype=bool)
    for hub in hubs:
        delta_xy = coords[:, :2] - hub[None, :2]
        near |= np.sum(delta_xy * delta_xy, axis=1) <= NEAR_TURBINE_RADIUS_D**2
    near &= hub_slab
    background = ~downstream & ~hub_slab
    masks = {
        "volume": np.ones(cell_count, dtype=bool),
        "hub_slab": hub_slab,
        "downstream_envelope": downstream,
        "near_turbine": near,
        "background": background,
    }
    return coords, masks, quadrature


def sample_native_role_queries(
    case: Any,
    rng: np.random.Generator,
    role_query_counts: Mapping[str, int] = DEFAULT_ROLE_QUERY_COUNTS,
    *,
    catalogue_cache: NativeRoleCatalogueCache | None = None,
) -> NativeRoleSample:
    """Draw fresh, quadrature-weighted native cells for each protected role.

    Each role mean estimates that role's centre-box quadrature measure. The
    concatenated overlapping role mixture is deliberately not an unbiased
    whole-volume estimator.
    """

    counts = {name: int(role_query_counts[name]) for name in ROLE_NAMES}
    if any(count <= 0 for count in counts.values()):
        raise ValueError("all protected WindFarm role query counts must be positive")
    catalogue = (catalogue_cache or _DEFAULT_ROLE_CATALOGUE_CACHE).get(case)
    pieces: list[np.ndarray] = []
    target_pieces: list[np.ndarray] = []
    flat_pieces: list[np.ndarray] = []
    role_slices: dict[str, slice] = {}
    support_volume: dict[str, float] = {}
    offset = 0
    for role in ROLE_NAMES:
        cdf = catalogue.role_cdf[role]
        positions = np.searchsorted(cdf, rng.random(counts[role]), side="right")
        positions = np.minimum(positions, cdf.size - 1)
        valid = catalogue.role_indices.get(role)
        selected = positions if valid is None else valid[positions]
        pieces.append(catalogue.coordinates_D[selected])
        # RunView.U is a read-only mmap; take a small owned copy for the batch.
        target_pieces.append(np.asarray(case.run.U[selected], dtype=np.float32).copy())
        flat_pieces.append(selected.astype(np.int64, copy=False))
        role_slices[role] = slice(offset, offset + counts[role])
        support_volume[role] = float(catalogue.role_support_volume_m3[role])
        offset += counts[role]
    return NativeRoleSample(
        coordinates_D=np.concatenate(pieces, axis=0),
        target_mps=np.concatenate(target_pieces, axis=0),
        flat_indices=np.concatenate(flat_pieces, axis=0),
        role_slices=role_slices,
        role_support_volume_m3=support_volume,
        role_sample_counts=counts,
        geometry_sha256=catalogue.geometry_sha256,
    )


def _role_mse_physical(
    prediction_mps: torch.Tensor,
    target_mps: torch.Tensor,
    role_slices: Mapping[str, slice],
) -> dict[str, torch.Tensor]:
    """Return scalar three-component MSEs in physical (m/s)^2 units."""

    if prediction_mps.shape != target_mps.shape or prediction_mps.ndim != 3 or prediction_mps.shape[-1] != 3:
        raise ValueError("physical WindFarm prediction and target must align as [B,Q,3]")
    result: dict[str, torch.Tensor] = {}
    for role in ROLE_NAMES:
        selection = role_slices[role]
        error = prediction_mps[:, selection, :] - target_mps[:, selection, :]
        result[role] = error.square().mean()
    return result


def role_budgets_from_baseline(
    baseline_rmse_mps: Mapping[str, float],
    *,
    delta: float = 0.10,
    absolute_floor_mps: float = 1.0e-5,
) -> dict[str, float]:
    """Calibrate fixed training-only physical RMSE budgets from Run2103."""

    if delta < 0.0 or absolute_floor_mps <= 0.0:
        raise ValueError("role budget tolerance and physical floor must be nonnegative/positive")
    budgets: dict[str, float] = {}
    for role in ROLE_NAMES:
        error = float(baseline_rmse_mps[role])
        if not math.isfinite(error) or error < 0.0:
            raise ValueError(f"baseline RMSE for {role} must be finite and nonnegative")
        budgets[role] = (1.0 + float(delta)) * error + float(absolute_floor_mps)
    return budgets


def role_risk_report(
    role_mse_mps2: Mapping[str, float],
    budgets_mps: Mapping[str, float],
) -> dict[str, dict[str, float | bool]]:
    """Convert measured reference MSEs into physical RMSE and normalized risk."""

    result: dict[str, dict[str, float | bool]] = {}
    for role in ROLE_NAMES:
        mse = float(role_mse_mps2[role])
        budget = float(budgets_mps[role])
        if not math.isfinite(mse) or mse < 0.0 or not math.isfinite(budget) or budget <= 0.0:
            raise ValueError(f"invalid physical role risk inputs for {role}")
        rmse = math.sqrt(mse)
        g = mse / (budget * budget) - 1.0
        result[role] = {
            "reference_mse_mps2": mse,
            "reference_rmse_mps": rmse,
            "budget_rmse_mps": budget,
            "risk_g": g,
            "budget_ratio": rmse / budget,
            "within_budget": bool(g <= 0.0),
        }
    return result


def _training_audit_role_risk(
    reviewed_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[int],
    audit_samples: Mapping[int, NativeRoleSample],
    budgets_mps: Mapping[str, float],
) -> dict[str, Any]:
    """Aggregate only training-row hard-reference risk for work control."""

    allowed = set(map(int, training_rows))
    selected = [entry for entry in reviewed_rows if int(entry["row_index"]) in allowed]
    if not selected:
        raise ValueError("hard-reference work gate requires at least one training audit row")
    sums = {role: 0.0 for role in ROLE_NAMES}
    counts = {role: 0 for role in ROLE_NAMES}
    for entry in selected:
        row = int(entry["row_index"])
        sample = audit_samples[row]
        for role in ROLE_NAMES:
            count = int(sample.role_sample_counts[role]) * 3
            if count < 1:
                raise ValueError(f"training audit row {row} has no {role} samples")
            mse = float(entry["reference_role_risk"][role]["reference_mse_mps2"])
            sums[role] += mse * count
            counts[role] += count
    aggregate = {role: sums[role] / counts[role] for role in ROLE_NAMES}
    return {
        "rows": [int(entry["row_index"]) for entry in selected],
        "role_risk": role_risk_report(aggregate, budgets_mps),
        "development_rows_used_for_work_control": False,
    }


@dataclass
class FidelityWorkGate:
    """Suspend work pressure after repeated training-only budget violations."""

    consecutive_failures: int = 0
    consecutive_clears: int = 0
    suspended: bool = False

    def observe(self, maximum_budget_ratio: float) -> dict[str, Any]:
        ratio = float(maximum_budget_ratio)
        if not math.isfinite(ratio) or ratio < 0.0:
            raise ValueError("training-audit budget ratio must be finite and nonnegative")
        if ratio > 1.0:
            self.consecutive_failures += 1
            self.consecutive_clears = 0
            if self.consecutive_failures >= 2:
                self.suspended = True
        else:
            self.consecutive_clears += 1
            self.consecutive_failures = 0
            if self.consecutive_clears >= 2:
                self.suspended = False
        return {
            "maximum_training_audit_budget_ratio": ratio,
            "consecutive_failed_training_audits": self.consecutive_failures,
            "consecutive_clear_training_audits": self.consecutive_clears,
            "work_incentive_suspended_for_reference_fidelity": self.suspended,
            "suspend_after_failures": 2,
            "restore_after_clears": 2,
            "threshold_budget_ratio": 1.0,
            "decision_scope": "training-only sampled hard-reference role risk",
        }


def _budget_trend_stop_decision(
    prior_reviews: Sequence[Mapping[str, float]],
    *,
    update: int,
    max_updates: int,
    current_max_ratio: float,
    failure_ratio: float = 1.25,
    minimum_relative_improvement: float = 0.05,
    earliest_review_update: int = 1500,
) -> dict[str, Any]:
    """Stop only after a later, persistent and stalled hard-reference failure."""

    prior = prior_reviews[-1] if prior_reviews else None
    if prior is not None and float(prior["max_ratio"]) <= failure_ratio:
        prior = None
    relative_improvement = None
    if prior is not None and float(prior["max_ratio"]) > 0.0:
        relative_improvement = (
            float(prior["max_ratio"]) - float(current_max_ratio)
        ) / float(prior["max_ratio"])
    persistent_failure = bool(
        prior is not None and float(current_max_ratio) > float(failure_ratio)
    )
    stalled_or_worsening = bool(
        relative_improvement is not None
        and relative_improvement < float(minimum_relative_improvement)
    )
    eligible = bool(
        int(update) >= int(earliest_review_update)
        and int(update) < int(max_updates)
    )
    should_stop = bool(eligible and persistent_failure and stalled_or_worsening)
    return {
        "eligible_after_stage3_review": eligible,
        "earliest_review_update": int(earliest_review_update),
        "future_scheduled_updates_remain": int(update) < int(max_updates),
        "current_max_budget_ratio": float(current_max_ratio),
        "failure_ratio_threshold": float(failure_ratio),
        "prior_failing_review": None if prior is None else dict(prior),
        "relative_improvement_since_prior_failing_review": relative_improvement,
        "minimum_relative_improvement_to_count_as_recovering": float(minimum_relative_improvement),
        "persistent_failure": persistent_failure,
        "stalled_or_worsening": stalled_or_worsening,
        "stop": should_stop,
    }


def _native_rng(seed: int, update: int, row: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence((int(seed), int(update), int(row), int(stream))))


def _physical_from_standardized(values: torch.Tensor, normalizer: VelocityNormalizer) -> torch.Tensor:
    mean = values.new_tensor(normalizer.mean)
    scale = values.new_tensor(normalizer.safe_std)
    return (values * scale + mean) * float(normalizer.u_ref_mps)


def _batch_from_sample(case: Any, sample: NativeRoleSample, normalizer: VelocityNormalizer, device: torch.device):
    batch = case_batch(
        case,
        sample.coordinates_D,
        normalizer=normalizer,
        velocity_mps=sample.target_mps,
        include_receiver_anchors=True,
    )
    return batch.to(device)


def _core_prediction(
    core: Any,
    encoded: Any,
    batch: Any,
    *,
    receiver_chunk_size: int,
    fixed_plan: MechanismPlan | None = None,
) -> torch.Tensor:
    plans = None if fixed_plan is None else (fixed_plan,)
    prepared = core.prepare(encoded, encoded.module_tokens, fixed_cover_plans=plans)
    output = core.decode_queries(
        prepared,
        batch.query_xy,
        query_features=batch.query_features,
        receiver_chunk_size=int(receiver_chunk_size),
    )
    return output["pred_field"]


def _fixed_plan_for_encoded(
    document: Mapping[str, Any],
    core: Any,
    encoded: Any,
    device: torch.device,
) -> MechanismPlan:
    """Rebind the verified G2 plan to a freshly recreated native case tree."""

    if int(encoded.module_present.shape[0]) != 1:
        raise ValueError("G2 training plan binding currently requires batch size one")
    saved = MechanismPlan.from_dict(dict(document["plan"]), device=device)
    current = core.backend.build_case_trees(encoded)[0]
    for name in ("coordinates", "weights", "roles", "coordinate_scale"):
        if not torch.equal(getattr(saved.tree.universe, name), getattr(current.universe, name)):
            raise ValueError(f"fresh G2 case tree differs in {name}")
    if saved.tree.nodes != current.nodes:
        raise ValueError("fresh G2 case tree has different receiver nodes")
    if saved.canonical_hash() != str(document.get("plan_hash", "")):
        raise ValueError("verified G2 plan hash changed before training")
    if not torch.equal(saved.module_present > 0.5, encoded.module_present[0] > 0.5):
        raise ValueError("G2 plan module identities differ from the current native case")
    return replace(saved, tree=current)


def _input_state(encoded: Any) -> dict[str, torch.Tensor]:
    return {
        "module_states": encoded.module_tokens,
        "environment_states": encoded.env_tokens,
        "global_state": encoded.global_token,
    }


def _detached_organizer_scores(
    core: Any,
    organizer: InputOnlyCoverOrganizer,
    encoded: Any,
) -> tuple[Any, ...]:
    """Score stage-1 labels from detached state, isolated from the physical graph."""

    organizer_encoded = _detach_tensors(encoded)
    trees = core.backend.build_case_trees(organizer_encoded)
    return organizer.score_cases(organizer_encoded, _input_state(organizer_encoded), trees)


def _detach_tensors(value: Any) -> Any:
    """Detach every organizer input while preserving encoded dataclass types."""

    if torch.is_tensor(value):
        return value.detach()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{
            field.name: _detach_tensors(getattr(value, field.name))
            for field in fields(value)
            if field.init
        })
    if isinstance(value, Mapping):
        return {key: _detach_tensors(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_tensors(item) for item in value)
    if isinstance(value, list):
        return [_detach_tensors(item) for item in value]
    return value


def _initialize_organizer_all_access(organizer: InputOnlyCoverOrganizer) -> dict[str, float]:
    """Start at a conservative hard root plan with moderate finite logits."""

    split_logit = -2.0
    permission_logit = 2.0
    with torch.no_grad():
        organizer.split_head.weight.zero_()
        organizer.split_head.bias.fill_(split_logit)
        for mechanism in INTERACTION_MECHANISMS:
            scorer = organizer.pair_scorers[mechanism]
            final = next(module for module in reversed(list(scorer.modules())) if isinstance(module, torch.nn.Linear))
            final.weight.zero_()
            final.bias.fill_(permission_logit)
    return {
        "split_logit": split_logit,
        "permission_logit": permission_logit,
    }


def _build_organizer(
    core: Any,
    encoded: Any,
    tree: Any,
    joint_cfg: Mapping[str, Any],
    device: torch.device,
) -> InputOnlyCoverOrganizer:
    role_count = max(8, int(tree.universe.roles.max().detach().cpu()) + 1)
    organizer = InputOnlyCoverOrganizer(
        state_dim=int(encoded.module_tokens.shape[-1]),
        module_feature_dim=int(encoded.module_features.shape[-1]),
        environment_feature_dim=0 if encoded.env_features is None else int(encoded.env_features.shape[-1]),
        hidden_dim=int(joint_cfg.get("organizer_hidden_dim", 32)),
        role_count=role_count,
        quadrature_invariant_source_measure=True,
    ).to(device)
    _initialize_organizer_all_access(organizer)
    return organizer


def _enable_physical_trainable_scope(model: WindFarmForwardModel) -> list[torch.nn.Parameter]:
    """Train fine typed interactions and the existing output head only."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    core = model.core
    trainable_prefixes = (
        "backend.mm_message.",
        "backend.me_message.",
        "backend.em_message.",
        "backend.module_update.",
        "backend.env_update.",
        "backend.query_module_message.",
        "backend.query_module_output.",
        "backend.env_query.",
        "backend.env_attention.",
        "backend.env_geometry_bias.",
        "common.context_norm.",
        "common.field_head.",
    )
    selected: list[torch.nn.Parameter] = []
    inventory: list[str] = []
    for name, parameter in core.named_parameters():
        if name.startswith(trainable_prefixes):
            parameter.requires_grad_(True)
            selected.append(parameter)
            inventory.append(name)
    if not selected:
        raise RuntimeError("Run2103 Dense core did not expose the expected fine and field-head parameters")
    model._joint_trainable_inventory = inventory
    return selected


def _new_model_from_source(
    source_payload: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    materialization_batch: Any,
    device: torch.device,
) -> WindFarmForwardModel:
    model_config = build_windfarm_forward_config(dict(source_payload["model_config"]))
    if model_config.forward_architecture != "dense_pairwise_field":
        raise ValueError("joint WindFarm maturation requires the intact Run2103 Dense architecture")
    model = WindFarmForwardModel(model_config, velocity_transform=normalizer).to(device)
    core = model.core
    core.backend.set_cover_mode("full_access")
    core.backend.set_cover_executor("dense_masked")
    model.materialize(materialization_batch)
    state = source_payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Run2103 selected checkpoint lacks model_state_dict")
    model.load_state_dict(state, strict=True)
    return model


def _validate_run2103_source(
    checkpoint_path: Path,
    payload: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
    requested_model: Mapping[str, Any],
) -> tuple[Any, VelocityNormalizer, str]:
    expected_directory = FROZEN_RUN_DIRECTORY
    if checkpoint_path.name != "best_field.pt" or checkpoint_path.parent.name != "checkpoints":
        raise ValueError("joint WindFarm training is locked to Run2103 checkpoints/best_field.pt")
    if checkpoint_path.parent.parent.name != expected_directory:
        raise ValueError(f"joint WindFarm source must be selected from {expected_directory}")
    validate_checkpoint_identity(payload, case_id="WindFarm", model_family="honf_forward", workflow="forward")
    train_config = payload.get("train_config")
    run_info = train_config.get("run", {}) if isinstance(train_config, Mapping) else {}
    run_id = str(run_info.get("id", "")) if isinstance(run_info, Mapping) else ""
    if run_id != "2103":
        raise ValueError(f"joint WindFarm source Run_ID must be 2103, got {run_id!r}")
    epoch = payload.get("best_epoch", payload.get("epoch", payload.get("current_epoch")))
    if epoch is None or int(epoch) != FROZEN_CHECKPOINT_EPOCH:
        raise ValueError(f"joint WindFarm source must be selected Run2103 e{FROZEN_CHECKPOINT_EPOCH}")
    if int(payload.get("epoch", payload.get("current_epoch", -1))) != FROZEN_CHECKPOINT_EPOCH:
        raise ValueError("Run2103 source checkpoint epoch does not match the selected best epoch")
    if list(payload.get("channel_order", [])) != ["Ux", "Uy", "Uz"] or int(payload.get("field_dim", 0)) != 3:
        raise ValueError("Run2103 source must use native [Ux, Uy, Uz] velocity values")
    if str(payload.get("dataset_id")) != str(dataset_cfg.get("dataset_id", "wind_farm_volume_v1")):
        raise ValueError("Run2103 source and target dataset IDs differ")
    source_schema = payload.get("dataset_schema")
    target_schema = dataset_cfg.get("dataset_schema")
    if target_schema is not None and source_schema != target_schema:
        raise ValueError("Run2103 source and current WindFarm dataset schema differ")
    source_config = build_windfarm_forward_config(dict(payload["model_config"]))
    target_config = build_windfarm_forward_config(dict(requested_model))
    if source_config.to_dict() != target_config.to_dict():
        raise ValueError("joint WindFarm config must preserve the exact Run2103 model tensor contract")
    manifest_path = checkpoint_path.parent.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run2103 source run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("checkpoints", {}).get("best_field")
    if not recorded:
        raise ValueError("Run2103 manifest does not select a best_field checkpoint")
    recorded_path = Path(str(recorded)).expanduser()
    if not recorded_path.is_absolute():
        recorded_path = manifest_path.parent / recorded_path
    if not recorded_path.is_file():
        fallback = manifest_path.parent / "checkpoints" / recorded_path.name
        if not fallback.is_file():
            raise FileNotFoundError(f"Run2103 manifest-selected checkpoint is missing: {recorded_path}")
        recorded_path = fallback
    if recorded_path.resolve() != checkpoint_path.resolve():
        raise ValueError("joint WindFarm source is not the manifest-selected best_field checkpoint")
    normalization = payload.get("normalization")
    if not isinstance(normalization, Mapping):
        raise TypeError("Run2103 source lacks its training-owned velocity normalization")
    return source_config, VelocityNormalizer.from_dict(dict(normalization)), _file_sha256(checkpoint_path)


def _full_plan(core: Any, encoded: Any) -> MechanismPlan:
    trees = core.backend.build_case_trees(encoded)
    if len(trees) != 1:
        raise ValueError("joint WindFarm preflight expects one native case")
    return MechanismPlan.full_access(
        trees[0],
        encoded.module_present[0],
        int(encoded.env_coords.shape[1]),
    )


def _physical_gradient_check(
    model: WindFarmForwardModel,
    organizer: InputOnlyCoverOrganizer,
    batch: Any,
    *,
    receiver_chunk_size: int,
) -> dict[str, Any]:
    core = model.core
    core.backend.set_cover_mode("external")
    core.backend.set_cover_executor("dense_masked")
    encoded = core.encode_case(batch)
    query_xy = batch.query_xy
    trees = core.backend.build_case_trees(encoded)
    module_states = encoded.module_tokens
    result = hard_value_soft_organizer_forward(
        core,
        encoded,
        module_states,
        organizer,
        query_xy,
        batch.query_features,
        receiver_chunk_size=receiver_chunk_size,
        collect_hard_aux=False,
    )
    if not torch.equal(result.prediction.detach(), result.hard_prediction.detach()):
        raise RuntimeError("hard-value/soft-organizer bridge changed the physical forward value")
    target = _physical_from_standardized(batch.target_field, model.velocity_transform)
    bridge_loss = (_physical_from_standardized(result.prediction, model.velocity_transform) - target).square().mean()
    hard_loss = (_physical_from_standardized(result.hard_prediction, model.velocity_transform) - target).square().mean()
    physical_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    organizer_parameters = [parameter for parameter in organizer.parameters() if parameter.requires_grad]
    bridge_gradients = torch.autograd.grad(
        bridge_loss,
        physical_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    hard_gradients = torch.autograd.grad(
        hard_loss,
        physical_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    organizer_gradients = torch.autograd.grad(
        bridge_loss,
        organizer_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    work_proxy = _soft_work_proxy(result.soft_plans[0], encoded, query_xy[0])
    work_gradients = torch.autograd.grad(
        work_proxy,
        organizer_parameters,
        allow_unused=True,
    )
    max_difference = 0.0
    compared = 0
    for left, right in zip(bridge_gradients, hard_gradients, strict=True):
        if left is None and right is None:
            continue
        if left is None or right is None:
            raise RuntimeError("shadow and hard-only physical parameter gradients have different connectivity")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise FloatingPointError("pre-optimizer bridge physical gradients are non-finite")
        difference = float((left - right).abs().max().detach().cpu())
        max_difference = max(max_difference, difference)
        compared += 1
    grad_norm_sq = sum(
        float(gradient.detach().square().sum().cpu())
        for gradient in organizer_gradients
        if gradient is not None
    )
    work_grad_norm_sq = sum(
        float(gradient.detach().square().sum().cpu())
        for gradient in work_gradients
        if gradient is not None
    )
    if compared == 0 or max_difference > 1.0e-7:
        raise RuntimeError(
            f"bridge physical gradients do not match hard-only gradients: max delta={max_difference}"
        )
    if not math.isfinite(work_grad_norm_sq) or math.sqrt(work_grad_norm_sq) <= 1.0e-12:
        raise RuntimeError("soft-plan work proxy has no finite nonzero gradient to the organizer")
    # Building trees here also verifies that the input-only organizer received
    # a geometry-only case index, without labels or case IDs.
    return {
        "hard_value_bitwise_equal": True,
        "physical_gradient_parameters_compared": compared,
        "physical_gradient_max_abs_delta": max_difference,
        "organizer_gradient_l2": math.sqrt(grad_norm_sq),
        "organizer_gradient_finite": math.isfinite(grad_norm_sq),
        "soft_work_proxy_value": float(work_proxy.detach().cpu()),
        "soft_work_organizer_gradient_l2": math.sqrt(work_grad_norm_sq),
        "soft_work_organizer_gradient_finite": math.isfinite(work_grad_norm_sq),
        "hard_plan_is_full_access": bool(result.hard_plans[0].is_full_access()),
        "tree_count": len(trees),
        "surrogate_gradient_semantics": "hard physical gradients; soft-plan organizer gradients only",
    }


def _inverse_standardized_tensor(values: torch.Tensor, normalizer: VelocityNormalizer) -> torch.Tensor:
    return _physical_from_standardized(values, normalizer)


def _load_g2_training_documents(
    view: WindFarmNativeView,
    split: GroupSplit,
    joint_cfg: Mapping[str, Any],
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    layouts = select_training_layouts(view, split.train, count=12)
    frozen = freeze_organizer_layout_split(layouts)
    search_dir = _resolved_path(joint_cfg["typed_search_dir"])
    report, manifest, documents = typed_fit._typed_fit_artifacts(
        search_dir,
        checkpoint_sha256=checkpoint_sha256,
        layouts=layouts,
    )
    if str(manifest.get("anchor_measure_variant")) != str(joint_cfg.get("anchor_measure_variant", "raw")):
        raise ValueError("joint-forward receiver anchors differ from the verified G2 supervision variant")
    expected_rows = list(map(int, frozen["training_rows_direction_order"]))
    if sorted(documents) != sorted(expected_rows):
        raise ValueError("verified G2 document rows differ from the frozen 24-row training panel")
    return report, manifest, documents, frozen


def _make_schedule(
    *,
    max_updates: int,
    train_rows: Sequence[int],
    labeled_rows: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    """Create one matched-arm row schedule before either optimizer is built."""

    if not train_rows or not labeled_rows:
        raise ValueError("matched WindFarm schedule needs train rows and verified G2 rows")
    rng = np.random.default_rng(int(seed) + 481516)
    ordered_train_rows = np.asarray(sorted(map(int, train_rows)), dtype=np.int64)
    if np.unique(ordered_train_rows).size != ordered_train_rows.size:
        raise ValueError("matched WindFarm training rows must be unique before schedule construction")
    shuffled_labels = rng.permutation(np.asarray(sorted(map(int, labeled_rows)), dtype=np.int64)).tolist()
    stage3_count = max(0, int(max_updates) - 500)
    stage3_rng = np.random.default_rng(np.random.SeedSequence((int(seed), 481517)))
    stage3_rows: list[int] = []
    while len(stage3_rows) < stage3_count:
        stage3_rows.extend(map(int, stage3_rng.permutation(ordered_train_rows)))
    schedule: list[dict[str, Any]] = []
    for update in range(1, int(max_updates) + 1):
        if update <= 150:
            stage = "all_access_g2_warmup"
            row = int(shuffled_labels[(update - 1) % len(shuffled_labels)])
        elif update <= 500:
            stage = "fixed_g2_fidelity_recovery"
            stage_index = update - 151
            if stage_index % 2 == 0:
                row = int(shuffled_labels[(stage_index // 2) % len(shuffled_labels)])
            else:
                row = int(ordered_train_rows[int(rng.integers(len(ordered_train_rows)))])
        else:
            stage = "joint_hard_packet_refinement"
            row = stage3_rows[update - 501]
        schedule.append({"update": update, "row_index": row, "stage": stage})
    return schedule


def _optimizer_steps_for_stage(
    arm: str,
    stage: str,
    *,
    use_bridge: bool,
) -> tuple[str, ...]:
    """Return the exact optimizer calls expected for one matched update."""

    if arm not in {"w_full", "w_packet"}:
        raise ValueError(f"unknown matched WindFarm arm {arm!r}")
    organizer_step_expected = bool(
        arm == "w_packet"
        and (stage == "all_access_g2_warmup" or use_bridge)
    )
    return ("physical_model", "organizer") if organizer_step_expected else ("physical_model",)


def _row_visit_coverage(
    train_rows: Sequence[int],
    layout_indices_by_row: Sequence[int] | np.ndarray,
    row_visits: Mapping[int, int],
) -> dict[str, Any]:
    """Summarize training-row and whole-layout visit coverage."""

    rows = list(map(int, train_rows))
    layouts_by_row = np.asarray(layout_indices_by_row, dtype=np.int64)
    training_layouts = sorted({int(layouts_by_row[row]) for row in rows})
    unvisited_rows = [row for row in rows if int(row_visits.get(row, 0)) == 0]
    visited_layouts = sorted({int(layouts_by_row[row]) for row in rows if int(row_visits.get(row, 0)) > 0})
    unvisited_layouts = sorted(set(training_layouts) - set(visited_layouts))
    counts = [int(row_visits.get(row, 0)) for row in rows]
    return {
        "training_row_count": len(rows),
        "visited_training_row_count": len(rows) - len(unvisited_rows),
        "training_row_coverage_fraction": (len(rows) - len(unvisited_rows)) / max(len(rows), 1),
        "unvisited_training_row_count": len(unvisited_rows),
        "unvisited_training_rows": unvisited_rows,
        "minimum_training_row_visits": min(counts, default=0),
        "maximum_training_row_visits": max(counts, default=0),
        "complete_native_training_row_passes": min(counts, default=0),
        "training_layout_count": len(training_layouts),
        "visited_training_layout_count": len(visited_layouts),
        "training_layout_coverage_fraction": len(visited_layouts) / max(len(training_layouts), 1),
        "unvisited_training_layouts": unvisited_layouts,
    }


def _scheduled_row_visit_coverage(
    schedule: Sequence[Mapping[str, Any]],
    *,
    through_update: int,
    train_rows: Sequence[int],
    layout_indices_by_row: Sequence[int] | np.ndarray,
    stage: str | None = None,
) -> dict[str, Any]:
    visits: Counter[int] = Counter(
        int(record["row_index"])
        for record in schedule
        if int(record["update"]) <= int(through_update)
        and (stage is None or str(record["stage"]) == stage)
    )
    return _row_visit_coverage(train_rows, layout_indices_by_row, visits)


def _soft_work_proxy(
    plan: MechanismPlan,
    encoded: Any,
    query_receivers: torch.Tensor,
    *,
    query_cap: int = 512,
) -> torch.Tensor:
    """Differentiable mechanism-balanced expected valid pair-mass proxy."""

    if int(encoded.module_present.shape[0]) != 1:
        raise ValueError("soft work proxy currently expects a single WindFarm case")
    device = query_receivers.device
    module_valid = encoded.module_present[0] > 0.5
    environment_valid = encoded.env_weights[0] > 0.0
    active_module_indices = torch.nonzero(module_valid, as_tuple=False).flatten()
    active_environment_indices = torch.nonzero(environment_valid, as_tuple=False).flatten()
    module_coords = encoded.module_centers[0].index_select(0, active_module_indices)
    environment_coords = encoded.env_coords[0].index_select(0, active_environment_indices)
    if int(query_receivers.shape[0]) > int(query_cap):
        indices = torch.linspace(
            0,
            int(query_receivers.shape[0]) - 1,
            int(query_cap),
            device=device,
        ).long()
        query_axis = query_receivers[indices]
    else:
        query_axis = query_receivers
    module_source_mechanisms = {"MM", "EM", "QM"}
    environment_source_mechanisms = {"ME", "QE"}
    active_receivers = {
        "MM": module_coords,
        "ME": module_coords,
        "EM": environment_coords,
        "QM": query_axis,
        "QE": query_axis,
    }
    ratios: list[torch.Tensor] = []
    for mechanism in INTERACTION_MECHANISMS:
        receivers = active_receivers[mechanism]
        access = plan.tree.access(receivers, plan.split_gates)
        membership = plan.permission_matrix(mechanism)
        if mechanism in module_source_mechanisms:
            if membership.shape[1] != encoded.module_present.shape[1]:
                raise ValueError(f"{mechanism} permission source axis differs from padded module axis")
            membership = membership.index_select(1, active_module_indices)
        elif mechanism in environment_source_mechanisms:
            if membership.shape[1] != encoded.env_weights.shape[1]:
                raise ValueError(f"{mechanism} permission source axis differs from environment axis")
            membership = membership.index_select(1, active_environment_indices)
        if access.shape[1] != membership.shape[0]:
            raise ValueError(f"{mechanism} soft work access and permission tree axes do not align")
        effective = access @ membership
        denominator = int(effective.numel())
        if mechanism == "MM":
            if receivers.shape[0] != membership.shape[1]:
                raise ValueError("MM receiver and active source axes must share module order")
            diagonal = torch.eye(receivers.shape[0], dtype=torch.bool, device=device)
            effective = effective.masked_select(~diagonal)
            denominator -= int(receivers.shape[0])
        if denominator <= 0:
            continue
        ratios.append(effective.sum() / float(denominator))
    if not ratios:
        return query_receivers.sum() * 0.0
    return torch.stack(ratios).mean()


def _new_physical_loss(
    prediction_std: torch.Tensor,
    target_std: torch.Tensor,
    role_slices: Mapping[str, slice],
    normalizer: VelocityNormalizer,
    budgets: Mapping[str, float],
    duals: torch.Tensor,
    *,
    soft_work_proxy: torch.Tensor | None,
    work_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    prediction_mps = _inverse_standardized_tensor(prediction_std, normalizer)
    target_mps = _inverse_standardized_tensor(target_std, normalizer)
    role_mse = _role_mse_physical(prediction_mps, target_mps, role_slices)
    normalized = torch.stack([
        role_mse[role] / (float(budgets[role]) ** 2)
        for role in ROLE_NAMES
    ])
    risk_g = normalized - 1.0
    loss = normalized.mean() + (duals.detach() * risk_g).sum()
    if soft_work_proxy is not None and float(work_weight) > 0.0:
        loss = loss + float(work_weight) * soft_work_proxy
    return loss, role_mse, risk_g


def _trainable_state_snapshot(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _parameter_delta_l2(parameters: Sequence[torch.nn.Parameter], baseline: Sequence[torch.Tensor]) -> float:
    if len(parameters) != len(baseline):
        raise ValueError("parameter snapshot length changed during matched training")
    total = 0.0
    for current, initial in zip(parameters, baseline, strict=True):
        total += float((current.detach().cpu() - initial).double().square().sum())
    return math.sqrt(total)


def _grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().double().square().sum().cpu())
    return math.sqrt(total)


def _interaction_aux_summary(auxiliary: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in auxiliary.items():
        if not torch.is_tensor(value):
            summary[str(key)] = str(type(value).__name__)
            continue
        tensor = value.detach()
        item: dict[str, Any] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        if tensor.numel() <= 32:
            item["values"] = tensor.cpu().tolist()
        else:
            item.update({
                "sum": float(tensor.sum().cpu()),
                "nonzero_count": int((tensor != 0).sum().cpu()),
                "minimum": float(tensor.min().cpu()),
                "maximum": float(tensor.max().cpu()),
            })
        summary[str(key)] = item
    return summary


@torch.inference_mode()
def _hard_qe_reopening_summary(
    current_plan: MechanismPlan,
    g2_plan: MechanismPlan,
    query_coords: torch.Tensor,
    *,
    query_chunk_size: int = 512,
) -> dict[str, Any]:
    """Count exact query/source QE pairs opened or closed versus G2."""

    current_permissions = current_plan.permission_matrix("QE") > 0
    g2_permissions = g2_plan.permission_matrix("QE") > 0
    if current_permissions.shape != g2_permissions.shape:
        raise ValueError("current and G2 QE source permission axes differ")
    if len(current_plan.tree.nodes) != len(g2_plan.tree.nodes):
        raise ValueError("current and G2 plans must share the same receiver candidate tree")
    reopened_pairs = 0
    closed_pairs = 0
    reopened_sources = torch.zeros(current_permissions.shape[1], dtype=torch.bool, device=query_coords.device)
    first_reopened: tuple[int, int] | None = None
    for start in range(0, int(query_coords.shape[0]), int(query_chunk_size)):
        stop = min(start + int(query_chunk_size), int(query_coords.shape[0]))
        queries = query_coords[start:stop]
        current_access = current_plan.tree.access(queries, current_plan.split_gates) @ current_permissions.to(query_coords.dtype)
        g2_access = g2_plan.tree.access(queries, g2_plan.split_gates) @ g2_permissions.to(query_coords.dtype)
        current_sources = current_access > 0
        g2_sources = g2_access > 0
        reopened = current_sources & ~g2_sources
        closed = g2_sources & ~current_sources
        reopened_pairs += int(reopened.sum().cpu())
        closed_pairs += int(closed.sum().cpu())
        reopened_sources |= reopened.any(dim=0)
        if first_reopened is None and bool(reopened.any()):
            pair = torch.nonzero(reopened, as_tuple=False)[0]
            first_reopened = (int(pair[0].cpu()) + start, int(pair[1].cpu()))

    restoration_cell: tuple[int, int] | None = None
    if first_reopened is not None:
        query_index, source_index = first_reopened
        query = query_coords[query_index : query_index + 1]
        current_active = current_plan.tree.access(query, current_plan.split_gates)[0] > 0
        g2_active = g2_plan.tree.access(query, g2_plan.split_gates)[0] > 0
        candidate_nodes = torch.nonzero(
            current_active & g2_active
            & current_permissions[:, source_index]
            & ~g2_permissions[:, source_index],
            as_tuple=False,
        ).flatten()
        if candidate_nodes.numel():
            restoration_cell = (int(candidate_nodes[0].cpu()), source_index)
    return {
        "query_count": int(query_coords.shape[0]),
        "environment_source_count": int(current_permissions.shape[1]),
        "reopened_query_source_pairs_vs_g2": reopened_pairs,
        "closed_query_source_pairs_vs_g2": closed_pairs,
        "unique_reopened_environment_sources": int(reopened_sources.sum().cpu()),
        "any_qe_source_reopened": reopened_pairs > 0,
        "single_source_restoration_cell": (
            None if restoration_cell is None
            else {"tree_node": restoration_cell[0], "environment_source": restoration_cell[1]}
        ),
    }


def _soft_qe_audit(
    organizer: InputOnlyCoverOrganizer,
    model: WindFarmForwardModel,
    cases: Mapping[int, Any],
    audit_samples: Mapping[int, NativeRoleSample],
    normalizer: VelocityNormalizer,
    device: torch.device,
    resource_ledger: RunResourceLedger | None = None,
) -> dict[str, Any]:
    mins: list[float] = []
    exact_zero = 0
    total = 0
    organizer.eval()
    with torch.no_grad():
        for row, case in cases.items():
            if resource_ledger is not None:
                _check_pair_review_wall(resource_ledger, stage=f"organizer soft-QE audit row {row}")
            sample = audit_samples[row]
            batch = _batch_from_sample(case, sample, normalizer, device)
            encoded = model.core.encode_case(batch)
            trees = model.core.backend.build_case_trees(encoded)
            scores = organizer.score_cases(encoded, _input_state(encoded), trees)
            plans = organizer.plans_from_scores(scores, encoded, trees, hard=False)
            qe = plans[0].permission_matrix("QE")
            mins.append(float(qe.min().detach().cpu()))
            exact_zero += int((qe == 0).sum().detach().cpu())
            total += int(qe.numel())
    return {
        "minimum_soft_qe_permission": min(mins) if mins else None,
        "exact_zero_count": exact_zero,
        "total_qe_decisions": total,
        "exact_zero_fraction": exact_zero / total if total else None,
        "soft_qe_below_configured_floor": bool(min(mins) < 1.0e-30) if mins else False,
        "measurement_rows": sorted(map(int, cases)),
    }


@torch.inference_mode()
def _evaluate_sampled_rows(
    *,
    arm: str,
    model: WindFarmForwardModel,
    organizer: InputOnlyCoverOrganizer | None,
    g2_documents: Mapping[int, dict[str, Any]],
    source_restoration_counterfactual_row: int | None,
    rows: Sequence[int],
    cases: Mapping[int, Any],
    samples: Mapping[int, NativeRoleSample],
    teacher_predictions: Mapping[int, torch.Tensor],
    normalizer: VelocityNormalizer,
    budgets: Mapping[str, float],
    device: torch.device,
    receiver_chunk_size: int,
    include_g2_fixed_reference: bool = False,
    resource_ledger: RunResourceLedger | None = None,
) -> dict[str, Any]:
    model.eval()
    if organizer is not None:
        organizer.eval()
    merged_sq = {role: 0.0 for role in ROLE_NAMES}
    merged_n = {role: 0 for role in ROLE_NAMES}
    per_row: list[dict[str, Any]] = []
    work_by_row: dict[str, Any] = {}
    qe_reopening_by_row: dict[str, Any] = {}
    fixed_g2_reference_by_row: dict[str, Any] = {}
    fixed_g2_hard_forward_equivalents = 0
    for row in map(int, rows):
        if resource_ledger is not None:
            _check_pair_review_wall(resource_ledger, stage=f"{arm} sampled review row {row}")
        case = cases[row]
        sample = samples[row]
        batch = _batch_from_sample(case, sample, normalizer, device)
        core = model.core
        encoded = core.encode_case(batch)
        fixed_plan = None
        g2_plan = None
        g2_standardized = None
        if arm == "w_packet":
            if organizer is None:
                raise RuntimeError("W-packet review requires its organizer")
            core.backend.set_cover_mode("external")
            core.backend.set_cover_executor("dense_masked")
            trees = core.backend.build_case_trees(encoded)
            scores = organizer.score_cases(encoded, _input_state(encoded), trees)
            fixed_plan = organizer.plans_from_scores(scores, encoded, trees, hard=True)[0]
            work_by_row[str(row)] = native_cover._typed_work_summary(
                fixed_plan, encoded, batch.query_xy[0], include_root_child_support=True
            )
            if row in g2_documents:
                g2_plan = _fixed_plan_for_encoded(g2_documents[row], core, encoded, device)
                reopening = _hard_qe_reopening_summary(fixed_plan, g2_plan, batch.query_xy[0])
                qe_reopening_by_row[str(row)] = reopening
                if (
                    row == source_restoration_counterfactual_row
                    and reopening["single_source_restoration_cell"] is not None
                ):
                    node = int(reopening["single_source_restoration_cell"]["tree_node"])
                    source = int(reopening["single_source_restoration_cell"]["environment_source"])
                    restored_permissions = g2_plan.permission_matrix("QE").clone()
                    restored_permissions[node, source] = 1.0
                    restored_plan = g2_plan.with_permission("QE", restored_permissions)
                    g2_standardized = _core_prediction(
                        core,
                        encoded,
                        batch,
                        receiver_chunk_size=receiver_chunk_size,
                        fixed_plan=g2_plan,
                    )
                    restored_standardized = _core_prediction(
                        core,
                        encoded,
                        batch,
                        receiver_chunk_size=receiver_chunk_size,
                        fixed_plan=restored_plan,
                    )
                    target_physical = _physical_from_standardized(batch.target_field, normalizer)
                    g2_mse = _role_mse_physical(
                        _physical_from_standardized(g2_standardized, normalizer),
                        target_physical,
                        sample.role_slices,
                    )
                    restored_mse = _role_mse_physical(
                        _physical_from_standardized(restored_standardized, normalizer),
                        target_physical,
                        sample.role_slices,
                    )
                    reopening["stored_native_hard_source_restoration_counterfactual"] = {
                        "tree_node": node,
                        "environment_source": source,
                        "extra_hard_forward_equivalents": 2,
                        "g2_reference_role_mse_mps2": {
                            role: float(g2_mse[role].cpu()) for role in ROLE_NAMES
                        },
                        "restored_reference_role_mse_mps2": {
                            role: float(restored_mse[role].cpu()) for role in ROLE_NAMES
                        },
                        "mse_reduction_g2_minus_restored_mps2": {
                            role: float((g2_mse[role] - restored_mse[role]).cpu())
                            for role in ROLE_NAMES
                        },
                        "reference_provenance": "stored native training labels; no oracle or physical solve",
                    }
                elif row == source_restoration_counterfactual_row:
                    reopening["stored_native_hard_source_restoration_counterfactual"] = {
                        "available": False,
                        "reason": "no same-node hard QE permission cell was directly reopened versus G2",
                    }
        else:
            core.backend.set_cover_mode("full_access")
            core.backend.set_cover_executor("dense_masked")
            full_plan = _full_plan(core, encoded)
            work_by_row[str(row)] = native_cover._typed_work_summary(
                full_plan, encoded, batch.query_xy[0], include_root_child_support=False
            )
        standardized = _core_prediction(
            core,
            encoded,
            batch,
            receiver_chunk_size=receiver_chunk_size,
            fixed_plan=fixed_plan,
        )
        physical = _physical_from_standardized(standardized, normalizer)
        target = _physical_from_standardized(batch.target_field, normalizer)
        role_mse = _role_mse_physical(physical, target, sample.role_slices)
        fixed_g2_risk = None
        if include_g2_fixed_reference and row in g2_documents:
            if g2_plan is None:
                core.backend.set_cover_mode("external")
                core.backend.set_cover_executor("dense_masked")
                g2_plan = _fixed_plan_for_encoded(g2_documents[row], core, encoded, device)
            if g2_standardized is None:
                core.backend.set_cover_mode("external")
                core.backend.set_cover_executor("dense_masked")
                g2_standardized = _core_prediction(
                    core,
                    encoded,
                    batch,
                    receiver_chunk_size=receiver_chunk_size,
                    fixed_plan=g2_plan,
                )
            g2_physical = _physical_from_standardized(g2_standardized, normalizer)
            g2_mse = _role_mse_physical(g2_physical, target, sample.role_slices)
            g2_mse_numbers = {role: float(g2_mse[role].detach().cpu()) for role in ROLE_NAMES}
            fixed_g2_risk = role_risk_report(g2_mse_numbers, budgets)
            fixed_g2_reference_by_row[str(row)] = {
                "role_risk_against_stored_native_reference": fixed_g2_risk,
                "role_mse_mps2": g2_mse_numbers,
                "plan_hash": str(g2_documents[row].get("plan_hash", "")),
                "plan_source": "verified train-only G2 artifact; no development labels loaded",
                "work": native_cover._typed_work_summary(
                    g2_plan,
                    encoded,
                    batch.query_xy[0],
                    include_root_child_support=True,
                ),
                "execution_path": "explicit fixed G2 plan through dense_masked",
            }
            fixed_g2_hard_forward_equivalents += 1
        teacher = teacher_predictions[row].to(device=device, dtype=physical.dtype)
        teacher_mse = _role_mse_physical(physical, teacher, sample.role_slices)
        row_mse = {role: float(role_mse[role].detach().cpu()) for role in ROLE_NAMES}
        row_teacher_rmse = {
            role: float(teacher_mse[role].detach().sqrt().cpu()) for role in ROLE_NAMES
        }
        risk = role_risk_report(row_mse, budgets)
        for role in ROLE_NAMES:
            count = int(sample.role_sample_counts[role]) * 3
            merged_sq[role] += row_mse[role] * count
            merged_n[role] += count
        per_row_entry = {
            "row_index": row,
            "layout_index": int(case.layout_index),
            "turbine_count": int(case.n_turbines),
            "reference_role_risk": risk,
            "teacher_discrepancy_role_rmse_mps": row_teacher_rmse,
        }
        if fixed_g2_risk is not None:
            per_row_entry["fixed_g2_reference_role_risk"] = fixed_g2_risk
        per_row.append(per_row_entry)
    aggregate_mse = {
        role: merged_sq[role] / max(merged_n[role], 1)
        for role in ROLE_NAMES
    }
    return {
        "rows": list(map(int, rows)),
        "aggregate_reference_role_risk": role_risk_report(aggregate_mse, budgets),
        "per_layout": per_row,
        "work_by_row": work_by_row,
        "fixed_g2_reference_by_labeled_row": fixed_g2_reference_by_row,
        "fixed_g2_hard_forward_equivalents": fixed_g2_hard_forward_equivalents,
        "hard_qe_source_reopening_vs_g2_by_training_row": qe_reopening_by_row,
        "high_turbine_count_tail_by_layout": sorted(
            per_row,
            key=lambda row: (-int(row["turbine_count"]), int(row["layout_index"]), int(row["row_index"])),
        )[: min(4, len(per_row))],
        "worst_layout_reference_budget_ratio_by_role": {
            role: max(
                (float(row["reference_role_risk"][role]["budget_ratio"]) for row in per_row),
                default=None,
            )
            for role in ROLE_NAMES
        },
        "execution_path": (
            "policy-free native Dense" if arm == "w_full"
            else "explicit hard organizer plans through dense_masked"
        ),
        "teacher_comparison": "candidate-to-Run2103 same-query RMSE; reported separately from native reference risk",
    }


@torch.inference_mode()
def _stream_full_grid(
    model: WindFarmForwardModel,
    case: Any,
    normalizer: VelocityNormalizer,
    device: torch.device,
    *,
    organizer: InputOnlyCoverOrganizer | None,
    fixed_plan: MechanismPlan | None,
    fixed_g2_document: Mapping[str, Any] | None = None,
    chunk_size: int,
    slice_output_path: Path | None = None,
    slice_update: int = -1,
    selected_model_state_sha256: str | None = None,
    resource_ledger: RunResourceLedger | None = None,
) -> dict[str, Any]:
    """Stream a complete native grid and integrate five physical role errors."""

    if resource_ledger is not None:
        _check_pair_review_wall(resource_ledger, stage=f"full-grid row {case.index} setup")
    run = case.run
    review_started = time.perf_counter()
    wx, wy, wz = support_weights(run.x_m, run.y_m, run.z_m)
    flat_all = np.arange(run.cell_count, dtype=np.int64)
    ix_all = flat_all % int(run.nx)
    iy_all = (flat_all // int(run.nx)) % int(run.ny)
    iz_all = flat_all // (int(run.nx) * int(run.ny))
    weights_all = (wx[ix_all] * wy[iy_all] * wz[iz_all]).astype(np.float64)
    coords_all = native_coordinates(run, flat_all, float(case.diameter_m))
    hubs = np.asarray(case.module_centers)[np.asarray(case.module_present) > 0.5]
    hub_slab = np.abs(coords_all[:, 2] - float(case.hub_height_m) / float(case.diameter_m)) <= HUB_SLAB_HALF_WIDTH_D
    downstream = downstream_envelope(coords_all, hubs)
    near = np.zeros(run.cell_count, dtype=bool)
    for hub in hubs:
        delta = coords_all[:, :2] - hub[None, :2]
        near |= np.sum(delta * delta, axis=1) <= NEAR_TURBINE_RADIUS_D**2
    near &= hub_slab
    masks = {
        "volume": np.ones(run.cell_count, dtype=bool),
        "hub_slab": hub_slab,
        "downstream_envelope": downstream,
        "near_turbine": near,
        "background": ~downstream & ~hub_slab,
    }
    total_sq = {role: np.zeros(3, dtype=np.float64) for role in ROLE_NAMES}
    total_weight = {role: 0.0 for role in ROLE_NAMES}
    core = model.core
    core.eval()
    if fixed_g2_document is not None:
        core.backend.set_cover_mode("external")
        core.backend.set_cover_executor("dense_masked")
        encoded_batch = case_batch(case, coords_all[:1], normalizer=normalizer, include_receiver_anchors=True).to(device)
        encoded = core.encode_case(encoded_batch)
        prepared_plan = _fixed_plan_for_encoded(fixed_g2_document, core, encoded, device)
        execution_path = "explicit fixed G2 plan through dense_masked"
    elif organizer is None:
        core.backend.set_cover_mode("full_access")
        core.backend.set_cover_executor("dense_masked")
        encoded_batch = case_batch(case, coords_all[:1], normalizer=normalizer, include_receiver_anchors=True).to(device)
        encoded = core.encode_case(encoded_batch)
        prepared_plan = None
        execution_path = "policy-free native Dense"
    else:
        core.backend.set_cover_mode("external")
        core.backend.set_cover_executor("dense_masked")
        encoded_batch = case_batch(case, coords_all[:1], normalizer=normalizer, include_receiver_anchors=True).to(device)
        encoded = core.encode_case(encoded_batch)
        if fixed_plan is None:
            trees = core.backend.build_case_trees(encoded)
            with torch.no_grad():
                scores = organizer.score_cases(encoded, _input_state(encoded), trees)
                prepared_plan = organizer.plans_from_scores(scores, encoded, trees, hard=True)[0]
        else:
            prepared_plan = fixed_plan
        execution_path = "organizer hard plan through dense_masked"
    prepared = core.prepare(
        encoded,
        encoded.module_tokens,
        fixed_cover_plans=None if prepared_plan is None else (prepared_plan,),
    )
    max_abs_prediction = 0.0
    decode_calls = 0
    plane_index = int(np.argmin(np.abs(np.asarray(run.z_m) - float(case.hub_height_m))))
    plane_cell_count = int(run.nx) * int(run.ny)
    slice_count = min(plane_cell_count, 4096)
    plane_local_indices = np.linspace(0, plane_cell_count - 1, slice_count, dtype=np.int64)
    slice_indices = plane_index * plane_cell_count + plane_local_indices
    slice_grid_indices_xyz = np.column_stack((
        plane_local_indices % int(run.nx),
        plane_local_indices // int(run.nx),
        np.full(slice_count, plane_index, dtype=np.int64),
    ))
    slice_predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, int(run.cell_count), int(chunk_size)):
            if resource_ledger is not None:
                _check_pair_review_wall(
                    resource_ledger,
                    stage=f"full-grid row {case.index} chunk start {start}",
                )
            decode_calls += 1
            stop = min(start + int(chunk_size), int(run.cell_count))
            coords = coords_all[start:stop]
            geometry = case.geometry_for_queries(coords)
            queries = torch.as_tensor(geometry["query_xy"], device=device).unsqueeze(0)
            features = torch.as_tensor(geometry["query_features"], device=device).unsqueeze(0)
            standardized = core.decode_queries(
                prepared,
                queries,
                query_features=features,
                receiver_chunk_size=int(chunk_size),
            )["pred_field"][0]
            prediction = _physical_from_standardized(standardized, normalizer).cpu().numpy().astype(np.float64)
            target = np.asarray(run.U[start:stop], dtype=np.float64)
            if not np.isfinite(prediction).all() or not np.isfinite(target).all():
                raise FloatingPointError(
                    f"full-grid row {case.index} has nonfinite native prediction or stored reference "
                    f"in cells [{start}, {stop})"
                )
            difference = prediction - target
            sample_start = int(np.searchsorted(slice_indices, start, side="left"))
            sample_stop = int(np.searchsorted(slice_indices, stop, side="left"))
            if sample_stop > sample_start:
                local_indices = slice_indices[sample_start:sample_stop] - start
                slice_predictions.append(prediction[local_indices].astype(np.float32, copy=True))
            max_abs_prediction = max(max_abs_prediction, float(np.max(np.abs(difference))))
            local_weight = weights_all[start:stop]
            for role, mask in masks.items():
                local_mask = mask[start:stop]
                if not np.any(local_mask):
                    continue
                selected = difference[local_mask]
                selected_weight = local_weight[local_mask]
                total_sq[role] += np.sum(selected_weight[:, None] * np.square(selected), axis=0)
                total_weight[role] += float(selected_weight.sum())
    saved_slice_path = None
    if slice_output_path is not None:
        slice_roles = np.column_stack([masks[role][slice_indices] for role in ROLE_NAMES]).astype(bool)
        slice_prediction = np.concatenate(slice_predictions, axis=0)
        slice_target = np.asarray(run.U[slice_indices], dtype=np.float32).copy()
        slice_coordinates = np.asarray(coords_all[slice_indices], dtype=np.float32).copy()
        slice_residual = slice_prediction - slice_target
        slice_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            slice_output_path,
            native_flat_indices=slice_indices,
            native_grid_indices_xyz=slice_grid_indices_xyz,
            coordinates_D=slice_coordinates,
            predicted_velocity_mps=slice_prediction,
            target_velocity_mps=slice_target,
            residual_velocity_mps=slice_residual,
            role_membership=slice_roles,
            role_names=np.asarray(ROLE_NAMES),
            quadrature_weight_m3=weights_all[slice_indices],
            plane_axis=np.asarray(["z"]),
            plane_index=np.asarray([plane_index], dtype=np.int64),
            plane_value_m=np.asarray([float(run.z_m[plane_index])], dtype=np.float64),
            source_row_index=np.asarray([int(case.index)], dtype=np.int64),
            layout_index=np.asarray([int(case.layout_index)], dtype=np.int64),
            update=np.asarray([int(slice_update)], dtype=np.int64),
            model_state_sha256=np.asarray([selected_model_state_sha256 or ""]),
            execution_path=np.asarray([execution_path]),
            reference_provenance=np.asarray(["stored cell-centred OpenFOAM CFD field; no new CFD solve"]),
        )
        saved_slice_path = str(slice_output_path)
    role_rmse = {
        role: (np.sqrt(total_sq[role] / total_weight[role]).tolist() if total_weight[role] > 0 else None)
        for role in ROLE_NAMES
    }
    scalar_rmse = {
        role: (
            math.sqrt(float(np.sum(total_sq[role])) / (3.0 * total_weight[role]))
            if total_weight[role] > 0 else None
        )
        for role in ROLE_NAMES
    }
    return {
        "row_index": int(case.index),
        "layout_index": int(case.layout_index),
        "turbine_count": int(case.n_turbines),
        "native_cell_count": int(run.cell_count),
        "chunk_size": int(chunk_size),
        "decode_calls": decode_calls,
        "wall_seconds": time.perf_counter() - review_started,
        "role_rmse_by_component_mps": role_rmse,
        "role_vector_rmse_mps": scalar_rmse,
        "maximum_absolute_cell_error_mps": max_abs_prediction,
        "saved_native_slice_path": saved_slice_path,
        "saved_native_slice_count": int(slice_count) if saved_slice_path is not None else 0,
        "saved_native_slice_plane": {
            "axis": "z",
            "index": plane_index,
            "coordinate_m": float(run.z_m[plane_index]),
            "selection": "deterministic regular flattened x-y subsample nearest hub height",
        } if saved_slice_path is not None else None,
        "execution_path": execution_path,
        "plan_hash": None if prepared_plan is None else prepared_plan.canonical_hash(),
        "reference_provenance": "stored cell-centred OpenFOAM CFD fields; no new CFD solve",
    }


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _save_joint_checkpoint(
    path: Path,
    *,
    model: WindFarmForwardModel,
    physical_optimizer: torch.optim.Optimizer,
    organizer: InputOnlyCoverOrganizer | None,
    organizer_optimizer: torch.optim.Optimizer | None,
    update: int,
    schedule_cursor: int,
    complete_native_epochs: int,
    arm: str,
    config: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    split_provenance: Mapping[str, Any],
    budgets: Mapping[str, float],
    duals: torch.Tensor,
    optimizer_call_count: int,
    hard_forward_equivalents: int,
    soft_forward_equivalents: int,
    review_history: Sequence[Mapping[str, Any]],
    resource_ledger: Mapping[str, Any],
) -> None:
    payload = {
        "checkpoint_schema_version": 1,
        "case_id": "WindFarm",
        "model_family": "honf_forward",
        "workflow": "forward",
        "stage": "windfarm_fidelity_budgeted_joint_forward",
        "arm": arm,
        "update_count": int(update),
        "completed_physical_updates": int(update),
        "scheduled_update_cursor": int(schedule_cursor),
        "complete_native_training_row_passes": int(complete_native_epochs),
        "epoch_semantics": (
            "A native epoch is one complete pass over every post-exclusion training row; "
            "optimizer update counts are stored separately."
        ),
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": physical_optimizer.state_dict(),
        "organizer_state_dict": None if organizer is None else organizer.state_dict(),
        "organizer_optimizer_state_dict": (
            None if organizer_optimizer is None else organizer_optimizer.state_dict()
        ),
        "train_config": copy.deepcopy(dict(config)),
        "dataset_id": config.get("dataset", {}).get("dataset_id", "wind_farm_volume_v1"),
        "dataset_schema": config.get("dataset", {}).get("dataset_schema"),
        "channel_order": ["Ux", "Uy", "Uz"],
        "field_dim": 3,
        "normalization": normalizer.to_dict(),
        "split_provenance": _json_safe(split_provenance),
        "physical_role_budgets_mps": dict(budgets),
        "risk_duals": _json_safe(duals),
        "actual_optimizer_calls": int(optimizer_call_count),
        "resource_ledger": _json_safe(resource_ledger),
        "hard_forward_equivalents": int(hard_forward_equivalents),
        "soft_forward_equivalents": int(soft_forward_equivalents),
        "review_history": _json_safe(list(review_history)),
        "trainable_parameter_names": list(getattr(model, "_joint_trainable_inventory", [])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_arm_checkpoint(
    arm_dir: Path,
    *,
    update: int,
    schedule_cursor: int,
    complete_native_epochs: int,
    model: WindFarmForwardModel,
    physical_optimizer: torch.optim.Optimizer,
    organizer: InputOnlyCoverOrganizer | None,
    organizer_optimizer: torch.optim.Optimizer | None,
    arm: str,
    config: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    split_provenance: Mapping[str, Any],
    budgets: Mapping[str, float],
    duals: torch.Tensor,
    optimizer_calls: int,
    hard_forward_equivalents: int,
    soft_forward_equivalents: int,
    reviews: Sequence[Mapping[str, Any]],
    resource_ledger: Mapping[str, Any],
) -> None:
    checkpoint_dir = arm_dir / "checkpoints"
    args = {
        "model": model,
        "physical_optimizer": physical_optimizer,
        "organizer": organizer,
        "organizer_optimizer": organizer_optimizer,
        "update": update,
        "schedule_cursor": schedule_cursor,
        "complete_native_epochs": complete_native_epochs,
        "arm": arm,
        "config": config,
        "normalizer": normalizer,
        "split_provenance": split_provenance,
        "budgets": budgets,
        "duals": duals,
        "optimizer_call_count": optimizer_calls,
        "hard_forward_equivalents": hard_forward_equivalents,
        "soft_forward_equivalents": soft_forward_equivalents,
        "review_history": reviews,
        "resource_ledger": resource_ledger,
    }
    _save_joint_checkpoint(checkpoint_dir / "latest.pt", **args)
    _save_joint_checkpoint(checkpoint_dir / f"updates_{int(update):06d}.pt", **args)


def _select_review_rows(
    view: WindFarmNativeView,
    train_rows: Sequence[int],
    validation_rows: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    layout_values = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    module_counts = np.asarray(view.metadata["n_turbines"], dtype=np.int64)

    def extremes(rows: Sequence[int], *, exclude: set[int] | None = None) -> list[int]:
        exclude = set() if exclude is None else exclude
        by_layout: dict[int, int] = {}
        for row in map(int, rows):
            layout = int(layout_values[row])
            if layout not in exclude:
                by_layout.setdefault(layout, row)
        candidates = list(by_layout.values())
        if not candidates:
            return []
        candidates.sort(key=lambda row: (int(module_counts[row]), int(layout_values[row]), row))
        chosen = [candidates[0]]
        if len(candidates) > 1:
            chosen.append(candidates[-1])
        return chosen

    train = extremes(train_rows)
    held = sorted(ORGANIZER_DEVELOPMENT_LAYOUT_INDICES)
    held_rows = [row for row in map(int, train_rows + list(validation_rows)) if int(layout_values[row]) in set(held)]
    if not held_rows:
        # Held organizer layouts are teacher-training rows, and are ordinarily
        # removed from train_rows above. Recover them from the native row table.
        held_rows = [
            row for row, layout in enumerate(layout_values.tolist())
            if int(layout) in set(held)
        ]
    dev = extremes(held_rows)
    original_validation = extremes(validation_rows)
    if len(train) < 2 or len(dev) < 2:
        raise ValueError("review panel requires two geometry-selected train and organizer-development rows")
    return train, dev, original_validation


def _g2_review_rows(view: WindFarmNativeView, labeled_rows: Sequence[int]) -> list[int]:
    """Select low/high module-count verified G2 train rows for source audits."""

    layout_values = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    module_counts = np.asarray(view.metadata["n_turbines"], dtype=np.int64)
    by_layout: dict[int, int] = {}
    for row in map(int, labeled_rows):
        by_layout.setdefault(int(layout_values[row]), row)
    selected = sorted(
        by_layout.values(),
        key=lambda row: (int(module_counts[row]), int(layout_values[row]), row),
    )
    if len(selected) < 2:
        raise ValueError("G2 source review needs at least two verified training layouts")
    return [selected[0], selected[-1]]


def _load_model_for_arm(
    source_payload: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    materialization_batch: Any,
    device: torch.device,
    *,
    seed: int,
) -> tuple[WindFarmForwardModel, list[torch.nn.Parameter], list[torch.Tensor]]:
    set_seed(int(seed))
    model = _new_model_from_source(source_payload, normalizer, materialization_batch, device)
    model.core.backend.set_cover_mode("full_access")
    model.core.backend.set_cover_executor("dense_masked")
    trainable = _enable_physical_trainable_scope(model)
    initial = _trainable_state_snapshot(trainable)
    return model, trainable, initial


def _full_grid_review(
    *,
    arm: str,
    update: int,
    model: WindFarmForwardModel,
    organizer: InputOnlyCoverOrganizer | None,
    view: WindFarmNativeView,
    train_rows: Sequence[int],
    dev_rows: Sequence[int],
    g2_documents: Mapping[int, dict[str, Any]],
    include_fixed_g2_grid: bool,
    fixed_g2_grid_skip_reason: str | None,
    normalizer: VelocityNormalizer,
    device: torch.device,
    chunk_size: int,
    output_dir: Path,
    resource_ledger: RunResourceLedger,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    review_started = time.perf_counter()
    total_decode_calls = 0
    total_grid_seconds = 0.0
    model_state_sha256 = _model_state_sha256(model)
    selected = list(map(int, train_rows[:2])) + list(map(int, dev_rows[:2]))
    progress_path = output_dir / "full_grid_progress.json"

    def persist_completed_grids() -> None:
        # A wall stop inside a later grid must not discard metrics from grids
        # that already finished. Each completed native stream is durable.
        _json_write(progress_path, {
            "status": "complete_streams_before_review_terminal_status",
            "arm": arm,
            "update": int(update),
            "selected_rows": selected,
            "completed_native_grid_results": results,
            "completed_native_grid_decode_calls": total_decode_calls,
            "completed_native_grid_wall_seconds": total_grid_seconds,
        })

    for row in selected:
        case = view.run(row)
        partial_result = _stream_full_grid(
            model,
            case,
            normalizer,
            device,
            organizer=organizer,
            fixed_plan=None,
            chunk_size=chunk_size,
            slice_output_path=output_dir / f"row_{row:04d}_hard_selected_plan.npz",
            slice_update=update,
            selected_model_state_sha256=model_state_sha256,
            resource_ledger=resource_ledger,
        )
        total_decode_calls += int(partial_result["decode_calls"])
        total_grid_seconds += float(partial_result["wall_seconds"])
        current_reference_name = (
            "organizer_selected_hard_plan" if organizer is not None else "policy_free_full_access"
        )
        results[str(row)] = {current_reference_name: partial_result}
        persist_completed_grids()
        if organizer is not None:
            same_weights_full = _stream_full_grid(
                model,
                case,
                normalizer,
                device,
                organizer=None,
                fixed_plan=None,
                chunk_size=chunk_size,
                slice_output_path=output_dir / f"row_{row:04d}_same_student_full_access.npz",
                slice_update=update,
                selected_model_state_sha256=model_state_sha256,
                resource_ledger=resource_ledger,
            )
            total_decode_calls += int(same_weights_full["decode_calls"])
            total_grid_seconds += float(same_weights_full["wall_seconds"])
            results[str(row)]["same_student_full_access"] = same_weights_full
            persist_completed_grids()
    fixed_g2_rows = list(map(int, train_rows[:2]))
    fixed_g2_grid_rows_completed: list[int] = []
    if include_fixed_g2_grid:
        for row in fixed_g2_rows:
            if row not in g2_documents:
                raise ValueError(f"fixed-G2 full-grid row {row} lacks verified train-only G2 provenance")
            case = view.run(row)
            fixed_reference = _stream_full_grid(
                model,
                case,
                normalizer,
                device,
                organizer=None,
                fixed_plan=None,
                fixed_g2_document=g2_documents[row],
                chunk_size=chunk_size,
                slice_output_path=output_dir / f"row_{row:04d}_fixed_g2_reference.npz",
                slice_update=update,
                selected_model_state_sha256=model_state_sha256,
                resource_ledger=resource_ledger,
            )
            total_decode_calls += int(fixed_reference["decode_calls"])
            total_grid_seconds += float(fixed_reference["wall_seconds"])
            results[str(row)]["fixed_g2_reference"] = fixed_reference
            fixed_g2_grid_rows_completed.append(row)
            persist_completed_grids()
    return {
        "arm": arm,
        "update": int(update),
        "rows_selected_by_g2_train_panel_and_input_geometry": selected,
        "fixed_g2_reference_train_rows": fixed_g2_rows,
        "fixed_g2_reference_grid_rows_completed": fixed_g2_grid_rows_completed,
        "fixed_g2_reference_grid_skip_reason": fixed_g2_grid_skip_reason,
        "native_grid_decode_calls": total_decode_calls,
        "summed_native_grid_wall_seconds": total_grid_seconds,
        "review_wall_seconds": time.perf_counter() - review_started,
        "chunk_size": int(chunk_size),
        "selected_model_state_sha256": model_state_sha256,
        "native_grid_results": results,
        "completed_grid_progress_path": str(progress_path),
        "interpretation": "repeated development and training-grid evidence from stored OpenFOAM CFD fields",
    }


def _run_arm(
    *,
    arm: str,
    source_payload: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    view: WindFarmNativeView,
    schedule: Sequence[Mapping[str, Any]],
    train_rows: Sequence[int],
    labeled_rows: Sequence[int],
    documents: Mapping[int, dict[str, Any]],
    audit_rows: Sequence[int],
    audit_samples: Mapping[int, NativeRoleSample],
    teacher_predictions: Mapping[int, torch.Tensor],
    cases: Mapping[int, Any],
    budgets: Mapping[str, float],
    split_provenance: Mapping[str, Any],
    run_config: Mapping[str, Any],
    joint_cfg: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    receiver_chunk_size: int,
    preflight: Mapping[str, Any],
    layout_visit_counter: Counter[int],
    row_visit_counter: Counter[int],
    role_catalogue_cache: NativeRoleCatalogueCache,
    resource_ledger: RunResourceLedger,
) -> dict[str, Any]:
    arm_dir = run_dir / "arms" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    first_row = int(schedule[0]["row_index"])
    first_case = view.run(first_row)
    first_sample = sample_native_role_queries(
        first_case,
        _native_rng(int(joint_cfg.get("sample_seed", 2103)), 1, first_row, 11),
        joint_cfg.get("role_query_counts", DEFAULT_ROLE_QUERY_COUNTS),
        catalogue_cache=role_catalogue_cache,
    )
    first_batch = _batch_from_sample(first_case, first_sample, normalizer, device)
    model, physical_parameters, initial_physical = _load_model_for_arm(
        source_payload,
        normalizer,
        first_batch,
        device,
        seed=int(run_config.get("training", {}).get("seed", 2103)),
    )
    model.train()
    core = model.core
    physical_optimizer = torch.optim.AdamW(
        physical_parameters,
        lr=float(run_config.get("training", {}).get("learning_rate", 3.0e-5)),
        weight_decay=float(run_config.get("training", {}).get("weight_decay", 1.0e-5)),
    )
    organizer: InputOnlyCoverOrganizer | None = None
    organizer_optimizer: torch.optim.Optimizer | None = None
    initial_organizer: list[torch.Tensor] = []
    if arm == "w_packet":
        core.backend.set_cover_mode("external")
        core.backend.set_cover_executor("dense_masked")
        first_encoded = core.encode_case(first_batch)
        first_tree = core.backend.build_case_trees(first_encoded)[0]
        organizer = _build_organizer(core, first_encoded, first_tree, joint_cfg, device)
        initial_organizer = _trainable_state_snapshot(list(organizer.parameters()))
        organizer_optimizer = torch.optim.AdamW(
            organizer.parameters(),
            lr=float(joint_cfg.get("organizer_learning_rate", 1.0e-4)),
            weight_decay=float(joint_cfg.get("organizer_weight_decay", 1.0e-4)),
        )
    duals = torch.zeros(len(ROLE_NAMES), device=device, dtype=torch.float32)
    risk_ema = torch.zeros_like(duals)
    risk_ema_initialized = False
    max_dual = float(joint_cfg.get("risk_dual_max", 10.0))
    dual_lr = float(joint_cfg.get("risk_dual_learning_rate", 0.01))
    ema_rate = float(joint_cfg.get("risk_ema_rate", 0.05))
    base_work_weight = float(joint_cfg.get("work_proxy_weight", 0.001))
    work_enabled = True
    budget_work_gate = FidelityWorkGate()
    zero_free_review_count = 0
    stable_organizer_state = (
        None if organizer is None else {
            key: value.detach().cpu().clone() for key, value in organizer.state_dict().items()
        }
    )
    metrics: list[dict[str, Any]] = []
    review_history: list[dict[str, Any]] = []
    max_updates = len(schedule)
    review_updates = set(map(int, joint_cfg.get("review_updates", (150, 500, 1500, 3000))))
    full_grid_updates = set(map(int, joint_cfg.get("full_grid_review_updates", (500, 1500))))
    review_updates.add(max_updates)
    full_replay_interval = int(joint_cfg.get("full_replay_interval", 5))
    organizer_optimizer_calls = 0
    physical_optimizer_calls = 0
    optimizer_calls = 0
    hard_forward_equivalents = 0
    soft_forward_equivalents = 0
    stage_updates: Counter[str] = Counter()
    stage3_row_visit_counter: Counter[int] = Counter()
    stage3_layout_visit_counter: Counter[int] = Counter()
    arm_started = time.perf_counter()
    latest_step_work: list[float] = []
    saturation_episodes: list[dict[str, Any]] = []
    resource_stop_reason: str | None = None
    early_stop_reason: str | None = None
    budget_review_history: list[dict[str, float]] = []
    last_review_metric_count = 0
    previous_review_organizer_snapshot = _trainable_state_snapshot(
        [] if organizer is None else list(organizer.parameters())
    )
    schedule_cursor = 0
    training_row_set = set(map(int, train_rows))

    for record in schedule:
        if early_stop_reason is not None:
            break
        update = int(record["update"])
        row = int(record["row_index"])
        stage = str(record["stage"])
        full_replay = bool(
            arm == "w_packet"
            and stage == "joint_hard_packet_refinement"
            and update % full_replay_interval == 0
        )
        use_bridge = bool(
            arm == "w_packet"
            and stage == "joint_hard_packet_refinement"
            and not full_replay
        )
        expected_optimizer_steps = _optimizer_steps_for_stage(
            arm,
            stage,
            use_bridge=use_bridge,
        )
        organizer_step_expected = "organizer" in expected_optimizer_steps
        apply_g2 = bool(
            arm == "w_packet"
            and stage == "fixed_g2_fidelity_recovery"
            and row in documents
        )
        required_calls = len(expected_optimizer_steps)
        if physical_optimizer_calls >= MAX_PHYSICAL_UPDATES_PER_ARM:
            resource_stop_reason = "per-arm 6,000 physical-update ceiling reached"
            resource_ledger.stop(f"{arm}: {resource_stop_reason}")
            break
        candidate_visits = row_visit_counter.copy()
        candidate_visits[row] += 1
        prospective_epochs = min(candidate_visits.get(train_row, 0) for train_row in training_row_set)
        if prospective_epochs > MAX_COMPLETE_NATIVE_EPOCHS_PER_ARM:
            resource_stop_reason = "per-arm 500 complete-native-epoch ceiling reached"
            resource_ledger.stop(f"{arm}: {resource_stop_reason}")
            break
        can_begin, blocked_reason = resource_ledger.can_begin_update(required_calls, arm=arm)
        if not can_begin:
            resource_stop_reason = blocked_reason or "pair-wide resource ceiling reached"
            resource_ledger.stop(f"{arm}: {resource_stop_reason}")
            break
        case = view.run(row)
        if update == 1 and row == first_row:
            sample = first_sample
            batch = first_batch
        else:
            sample = sample_native_role_queries(
                case,
                _native_rng(int(joint_cfg.get("sample_seed", 2103)), update, row, 11),
                joint_cfg.get("role_query_counts", DEFAULT_ROLE_QUERY_COUNTS),
                catalogue_cache=role_catalogue_cache,
            )
            batch = _batch_from_sample(case, sample, normalizer, device)
        core = model.core
        model.train()
        if organizer is not None:
            organizer.train()
        core.backend.set_cover_mode("external" if (use_bridge or apply_g2) else "full_access")
        core.backend.set_cover_executor("dense_masked")
        physical_optimizer.zero_grad(set_to_none=True)
        if organizer_optimizer is not None and organizer_step_expected:
            organizer_optimizer.zero_grad(set_to_none=True)
        forward_started = time.perf_counter()
        encoded = core.encode_case(batch)
        fixed_plan = None
        bridge_result = None
        if use_bridge:
            if organizer is None:
                raise RuntimeError("joint packet stage requires an input-only organizer")
            bridge_result = hard_value_soft_organizer_forward(
                core,
                encoded,
                encoded.module_tokens,
                organizer,
                batch.query_xy,
                batch.query_features,
                receiver_chunk_size=receiver_chunk_size,
                collect_hard_aux=bool(update in review_updates),
            )
            prediction_for_loss = bridge_result.prediction
            hard_forward_equivalents += 1
            soft_forward_equivalents += 1
            soft_work = _soft_work_proxy(
                bridge_result.soft_plans[0],
                encoded,
                batch.query_xy[0],
            )
        elif apply_g2:
            fixed_plan = _fixed_plan_for_encoded(documents[row], core, encoded, device)
            prediction_for_loss = _core_prediction(
                core,
                encoded,
                batch,
                receiver_chunk_size=receiver_chunk_size,
                fixed_plan=fixed_plan,
            )
            soft_work = None
            hard_forward_equivalents += 1
        else:
            prediction_for_loss = _core_prediction(
                core,
                encoded,
                batch,
                receiver_chunk_size=receiver_chunk_size,
            )
            soft_work = None
            hard_forward_equivalents += 1
        forward_seconds = time.perf_counter() - forward_started
        dual_view = duals.detach()
        applied_work_weight = (
            base_work_weight if use_bridge and work_enabled and not budget_work_gate.suspended else 0.0
        )
        loss, physical_role_mse, hard_risk_g = _new_physical_loss(
            prediction_for_loss,
            batch.target_field,
            sample.role_slices,
            normalizer,
            budgets,
            dual_view,
            soft_work_proxy=soft_work if use_bridge else None,
            work_weight=applied_work_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite {arm} physical loss at update {update}")

        supervised_loss_value: float | None = None
        supervised_loss: torch.Tensor | None = None
        organizer_parameters = [] if organizer is None else list(organizer.parameters())
        if organizer_step_expected and stage == "all_access_g2_warmup":
            if organizer is None or organizer_optimizer is None or row not in documents:
                raise RuntimeError("stage-1 organizer warmup requires a verified train-only G2 row")
            scores = _detached_organizer_scores(core, organizer, encoded)
            labels = typed_fit._typed_label_tensors(documents[row], device)
            supervised_loss = typed_fit._typed_supervised_loss_from_scores(scores[0], labels)
            if not bool(torch.isfinite(supervised_loss)):
                raise FloatingPointError(f"non-finite observed-label organizer loss at update {update}")
            supervised_loss_value = float(supervised_loss.detach().cpu())
        backward_started = time.perf_counter()
        loss.backward()
        physical_grad_norm = clip_grad_norm_(
            physical_parameters,
            float(run_config.get("training", {}).get("gradient_clip_norm", 1.0)),
            error_if_nonfinite=True,
        )
        organizer_grad_norm = None
        if use_bridge:
            if organizer is None or organizer_optimizer is None:
                raise RuntimeError("joint hard-packet refinement requires its organizer optimizer")
            organizer_grad_norm = clip_grad_norm_(organizer_parameters, 1.0, error_if_nonfinite=True)
        attempted, blocked_reason = resource_ledger.record_attempt(
            arm=arm,
            update=update,
            optimizer="physical_model",
        )
        if not attempted:
            resource_stop_reason = blocked_reason or "pair resource ceiling reached before physical optimizer step"
            resource_ledger.stop(f"{arm}: {resource_stop_reason}")
            break
        physical_optimizer.step()
        resource_ledger.record_completion(arm=arm)
        optimizer_calls += 1
        physical_optimizer_calls += 1
        incomplete_optimizer_step = False
        optimizer_boundary_stop_reason = None
        if organizer_step_expected and stage == "all_access_g2_warmup":
            if supervised_loss is None or organizer_optimizer is None:
                raise RuntimeError("stage-1 organizer update lost its verified-label loss")
            supervised_loss.backward()
            organizer_grad_norm = clip_grad_norm_(organizer_parameters, 1.0, error_if_nonfinite=True)
        if organizer_step_expected:
            if organizer_optimizer is None:
                raise RuntimeError("scheduled organizer optimizer call has no optimizer")
            attempted, blocked_reason = resource_ledger.record_attempt(
                arm=arm,
                update=update,
                optimizer="organizer",
            )
            if not attempted:
                resource_stop_reason = blocked_reason or "pair resource ceiling reached before organizer optimizer step"
                resource_ledger.stop(f"{arm}: {resource_stop_reason}")
                incomplete_optimizer_step = True
                optimizer_boundary_stop_reason = resource_stop_reason
            else:
                organizer_optimizer.step()
                resource_ledger.record_completion(arm=arm)
                optimizer_calls += 1
                organizer_optimizer_calls += 1
        backward_seconds = time.perf_counter() - backward_started
        layout_visit_counter[int(view.metadata["layout_index"][row])] += 1
        row_visit_counter[row] += 1
        if stage == "joint_hard_packet_refinement":
            stage3_layout_visit_counter[int(view.metadata["layout_index"][row])] += 1
            stage3_row_visit_counter[row] += 1
        stage_updates[stage] += 1
        schedule_cursor = update
        with torch.no_grad():
            if not risk_ema_initialized:
                risk_ema.copy_(hard_risk_g.detach())
                risk_ema_initialized = True
            else:
                risk_ema.mul_(1.0 - ema_rate).add_(hard_risk_g.detach(), alpha=ema_rate)
            if stage in {"fixed_g2_fidelity_recovery", "joint_hard_packet_refinement"}:
                duals.add_(risk_ema, alpha=dual_lr).clamp_(min=0.0, max=max_dual)
        work_value = None if soft_work is None else float(soft_work.detach().cpu())
        if work_value is not None:
            latest_step_work.append(work_value)
        physical_role_numbers = {
            role: float(physical_role_mse[role].detach().cpu())
            for role in ROLE_NAMES
        }
        risk_report = role_risk_report(physical_role_numbers, budgets)
        row_metric = {
            "arm": arm,
            "update": update,
            "stage": stage,
            "row_index": row,
            "layout_index": int(case.layout_index),
            "turbine_count": int(case.n_turbines),
            "full_access_replay": full_replay,
            "g2_fixed_cover_applied": apply_g2,
            "optimizer_calls_total": optimizer_calls,
            "physical_optimizer_calls": physical_optimizer_calls,
            "organizer_optimizer_calls": organizer_optimizer_calls,
            "optimizer_update_complete": not incomplete_optimizer_step,
            "resource_stop_after_update": optimizer_boundary_stop_reason,
            "physical_loss_dimensionless_budget_normalized": float(loss.detach().cpu()),
            "physical_gradient_norm_preclip": float(physical_grad_norm.detach().cpu()),
            "organizer_gradient_norm_preclip": (
                None if organizer_grad_norm is None else float(organizer_grad_norm.detach().cpu())
            ),
            "organizer_bridge_update": bool(use_bridge),
            "role_mse_mps2": physical_role_numbers,
            "role_risk": risk_report,
            "risk_duals": {role: float(duals[index].detach().cpu()) for index, role in enumerate(ROLE_NAMES)},
            "organizer_observed_label_loss": supervised_loss_value,
            "soft_mechanism_balanced_work_proxy": work_value,
            "work_proxy_weight": applied_work_weight,
            "hard_forward_equivalents_cumulative": hard_forward_equivalents,
            "soft_forward_equivalents_cumulative": soft_forward_equivalents,
            "hard_interaction_aux_audit": (
                None if bridge_result is None or update not in review_updates
                else _interaction_aux_summary(bridge_result.hard_interaction_aux)
            ),
            "forward_dispatch_seconds": forward_seconds,
            "backward_dispatch_seconds": backward_seconds,
            "role_sampling": {
                "counts": dict(sample.role_sample_counts),
                "support_volume_m3": dict(sample.role_support_volume_m3),
                "catalogue_geometry_sha256": sample.geometry_sha256,
                "drawn_native_indices_sha256": hashlib.sha256(
                    np.ascontiguousarray(sample.flat_indices).view(np.uint8)
                ).hexdigest(),
                "fresh_rng_identity": {
                    "sample_seed": int(joint_cfg.get("sample_seed", 2103)),
                    "update": update,
                    "row_index": row,
                    "stream_id": 11,
                },
                "estimand": "five separate quadrature-stratified role estimates; overlapping role supports",
            },
        }
        metrics.append(row_metric)

        # A review can cost much more wall time than one optimizer step. Do
        # not enter it after the shared cap has already been reached, even at
        # the last scheduled update where no next-step check would occur.
        if resource_ledger.elapsed_seconds() >= resource_ledger.max_wall_seconds:
            resource_stop_reason = (
                f"shared pair wall ceiling reached after {arm} update {update} "
                f"at {resource_ledger.elapsed_seconds():.1f}s"
            )
            resource_ledger.stop(f"{arm}: {resource_stop_reason}")
            row_metric["resource_stop_after_update"] = resource_stop_reason

        due_review = update in review_updates and resource_stop_reason is None
        due_periodic_checkpoint = update % 100 == 0 and resource_stop_reason is None
        if due_review or due_periodic_checkpoint:
            review_started = time.perf_counter()
            eval_rows = list(map(int, audit_rows))
            counterfactual_row = next(
                (row for row in eval_rows if row in documents),
                None,
            )
            try:
                review = _evaluate_sampled_rows(
                    arm=arm,
                    model=model,
                    organizer=organizer,
                    g2_documents=documents,
                    source_restoration_counterfactual_row=counterfactual_row,
                    rows=eval_rows,
                    cases=cases,
                    samples=audit_samples,
                    teacher_predictions=teacher_predictions,
                    normalizer=normalizer,
                    budgets=budgets,
                    device=device,
                    receiver_chunk_size=receiver_chunk_size,
                    include_g2_fixed_reference=bool(update >= 500 and update in review_updates),
                    resource_ledger=resource_ledger,
                )
            except _PairWallLimitReached as exc:
                resource_stop_reason = str(exc)
                resource_ledger.stop(f"{arm}: {resource_stop_reason}")
                break
            training_audit = _training_audit_role_risk(
                review["per_layout"], train_rows, audit_samples, budgets
            )
            review["training_only_work_control"] = training_audit
            budget_work_state = None
            if organizer is not None:
                budget_work_state = budget_work_gate.observe(max(
                    float(training_audit["role_risk"][role]["budget_ratio"])
                    for role in ROLE_NAMES
                ))
            if resource_ledger.elapsed_seconds() >= resource_ledger.max_wall_seconds:
                resource_stop_reason = (
                    f"shared pair wall ceiling reached during {arm} sampled review at update {update}"
                )
                resource_ledger.stop(f"{arm}: {resource_stop_reason}")
            soft_qe = None
            saturation_remedy = None
            if organizer is not None and resource_stop_reason is None:
                try:
                    check_cases = {row: cases[row] for row in eval_rows if row in cases}
                    soft_qe = _soft_qe_audit(
                        organizer,
                        model,
                        check_cases,
                        audit_samples,
                        normalizer,
                        device,
                        resource_ledger=resource_ledger,
                    )
                    threshold = float(joint_cfg.get("minimum_soft_qe_permission", 1.0e-30))
                    zeros_or_underflow = (
                        int(soft_qe["exact_zero_count"]) > 0
                        or float(soft_qe["minimum_soft_qe_permission"] or 0.0) < threshold
                    )
                    if zeros_or_underflow:
                        work_enabled = False
                        zero_free_review_count = 0
                        prior_lr = float(organizer_optimizer.param_groups[0]["lr"]) if organizer_optimizer else None
                        if stable_organizer_state is not None and organizer_optimizer is not None:
                            organizer.load_state_dict(stable_organizer_state, strict=True)
                            organizer_optimizer = torch.optim.AdamW(
                                organizer.parameters(),
                                lr=max(float(prior_lr or 1.0e-4) * 0.5, 1.0e-7),
                                weight_decay=float(joint_cfg.get("organizer_weight_decay", 1.0e-4)),
                            )
                        saturation_remedy = {
                            "trigger": "at least one soft QE permission is exactly zero or below configured floor",
                            "work_proxy_weight_set_to_zero": True,
                            "organizer_restored_to_last_non_saturated_review": stable_organizer_state is not None,
                            "organizer_optimizer_reset": organizer_optimizer is not None,
                            "organizer_learning_rate_before": prior_lr,
                            "optimizer_state_reinitialized": True,
                        }
                        saturation_episodes.append({"update": update, **saturation_remedy, "soft_qe": soft_qe})
                        # Verify the restored organizer's soft support immediately.
                        soft_qe = _soft_qe_audit(
                            organizer,
                            model,
                            check_cases,
                            audit_samples,
                            normalizer,
                            device,
                            resource_ledger=resource_ledger,
                        )
                        if int(soft_qe["exact_zero_count"]) > 0:
                            saturation_remedy["recovery_verified"] = False
                            saturation_remedy["recovery_failure"] = "last known-safe state still had exact zero soft QE"
                        else:
                            saturation_remedy["recovery_verified"] = True
                    else:
                        zero_free_review_count += 1
                        if stable_organizer_state is not None:
                            stable_organizer_state = {
                                key: value.detach().cpu().clone() for key, value in organizer.state_dict().items()
                            }
                        required_clear = int(joint_cfg.get("minimum_nonzero_reviews_to_restore_work", 2))
                        if not work_enabled and zero_free_review_count >= required_clear:
                            work_enabled = True
                        if stable_organizer_state is None:
                            stable_organizer_state = {
                                key: value.detach().cpu().clone() for key, value in organizer.state_dict().items()
                            }
                except _PairWallLimitReached as exc:
                    resource_stop_reason = str(exc)
                    resource_ledger.stop(f"{arm}: {resource_stop_reason}")
            if resource_stop_reason is None and resource_ledger.elapsed_seconds() >= resource_ledger.max_wall_seconds:
                resource_stop_reason = (
                    f"shared pair wall ceiling reached during {arm} organizer review at update {update}"
                )
                resource_ledger.stop(f"{arm}: {resource_stop_reason}")
            review_window = metrics[last_review_metric_count:]
            last_review_metric_count = len(metrics)
            aggregate_risk = review["aggregate_reference_role_risk"]
            maximum_budget_ratio = max(
                float(aggregate_risk[role]["budget_ratio"]) for role in ROLE_NAMES
            )
            budget_failure_limit = float(
                joint_cfg.get("early_stop_budget_failure_max_ratio", 1.25)
            )
            budget_decision = _budget_trend_stop_decision(
                budget_review_history,
                update=update,
                max_updates=max_updates,
                current_max_ratio=maximum_budget_ratio,
                failure_ratio=budget_failure_limit,
                minimum_relative_improvement=float(
                    joint_cfg.get("early_stop_budget_failure_min_relative_improvement", 0.05)
                ),
                earliest_review_update=int(
                    joint_cfg.get("early_stop_budget_failure_min_update", 1500)
                ),
            )
            budget_review_history.append({"update": float(update), "max_ratio": maximum_budget_ratio})
            early_stop_checks: dict[str, Any] = {
                "maximum_aggregate_hard_reference_budget_ratio": maximum_budget_ratio,
                "hard_reference_budget_trend_decision": budget_decision,
                "hard_reference_budget_failure": bool(budget_decision["stop"]),
                "stage3_bridge_updates_since_previous_review": None,
                "stage3_nonzero_organizer_gradient_updates_since_previous_review": None,
                "organizer_parameter_delta_l2_since_previous_review": None,
                "disconnected_organizer_signal": False,
            }
            if organizer is not None:
                bridge_metrics = [
                    item for item in review_window
                    if bool(item.get("organizer_bridge_update", False))
                ]
                nonzero_gradients = [
                    float(item["organizer_gradient_norm_preclip"])
                    for item in bridge_metrics
                    if item.get("organizer_gradient_norm_preclip") is not None
                    and math.isfinite(float(item["organizer_gradient_norm_preclip"]))
                    and float(item["organizer_gradient_norm_preclip"]) > 1.0e-12
                ]
                organizer_delta = _parameter_delta_l2(
                    list(organizer.parameters()), previous_review_organizer_snapshot
                )
                previous_review_organizer_snapshot = _trainable_state_snapshot(
                    list(organizer.parameters())
                )
                disconnected_review_update = int(
                    joint_cfg.get("early_stop_disconnected_organizer_review_update", 1500)
                )
                disconnected = bool(
                    update >= disconnected_review_update
                    and bridge_metrics
                    and not nonzero_gradients
                    and organizer_delta <= 1.0e-12
                )
                early_stop_checks.update({
                    "stage3_bridge_updates_since_previous_review": len(bridge_metrics),
                    "stage3_nonzero_organizer_gradient_updates_since_previous_review": len(nonzero_gradients),
                    "organizer_parameter_delta_l2_since_previous_review": organizer_delta,
                    "disconnected_organizer_signal": disconnected,
                    "disconnected_organizer_review_update": disconnected_review_update,
                })
            if early_stop_checks["hard_reference_budget_failure"]:
                early_stop_reason = (
                    "predeclared early stop: aggregate hard-reference role budget failure persisted "
                    "and stalled after the stage-3 review gate"
                )
            elif early_stop_checks["disconnected_organizer_signal"]:
                early_stop_reason = (
                    "predeclared early stop: no nonzero stage-3 organizer gradient or parameter change "
                    "across the review window"
                )
            current_review_decision = {
                **early_stop_checks,
                "stop_reason": early_stop_reason,
                "continuation_is_exactly_resumable": False,
            }
            grid_review = None
            if update in full_grid_updates and resource_stop_reason is None:
                selected_dev = [row for row in eval_rows if int(view.metadata["layout_index"][row]) in set(ORGANIZER_DEVELOPMENT_LAYOUT_INDICES)]
                selected_g2_train = _g2_review_rows(view, labeled_rows)
                if len(selected_g2_train) >= 2 and len(selected_dev) >= 2:
                    fixed_g2_grid_updates = set(map(
                        int,
                        joint_cfg.get("fixed_g2_full_grid_review_updates", (500,)),
                    ))
                    minimum_remaining = int(
                        joint_cfg.get("fixed_g2_grid_minimum_remaining_wall_seconds", 1800)
                    )
                    remaining_wall = max(
                        0.0,
                        resource_ledger.max_wall_seconds - resource_ledger.elapsed_seconds(),
                    )
                    include_fixed_g2_grid = bool(
                        update in fixed_g2_grid_updates and remaining_wall >= minimum_remaining
                    )
                    fixed_g2_grid_skip_reason = None
                    if update not in fixed_g2_grid_updates:
                        fixed_g2_grid_skip_reason = "update is outside the declared fixed-G2 grid review schedule"
                    elif not include_fixed_g2_grid:
                        fixed_g2_grid_skip_reason = (
                            f"only {remaining_wall:.1f}s remain under the shared pair wall cap; "
                            f"minimum is {minimum_remaining}s"
                        )
                    grid_output_dir = run_dir / "native_slices" / f"update_{update:05d}" / arm
                    try:
                        grid_review = _full_grid_review(
                            arm=arm,
                            update=update,
                            model=model,
                            organizer=organizer,
                            view=view,
                            train_rows=selected_g2_train,
                            dev_rows=selected_dev,
                            g2_documents=documents,
                            include_fixed_g2_grid=include_fixed_g2_grid,
                            fixed_g2_grid_skip_reason=fixed_g2_grid_skip_reason,
                            normalizer=normalizer,
                            device=device,
                            chunk_size=int(joint_cfg.get("full_grid_chunk_size", receiver_chunk_size)),
                            output_dir=grid_output_dir,
                            resource_ledger=resource_ledger,
                        )
                    except _PairWallLimitReached as exc:
                        resource_stop_reason = str(exc)
                        resource_ledger.stop(f"{arm}: {resource_stop_reason}")
                        grid_review = {
                            "status": "resource_stopped_during_full_grid_review",
                            "reason": resource_stop_reason,
                            "completed_grid_progress_path": (
                                str(grid_output_dir / "full_grid_progress.json")
                                if (grid_output_dir / "full_grid_progress.json").exists()
                                else None
                            ),
                            "complete_slices_retained": [
                                str(path) for path in sorted(grid_output_dir.glob("*.npz"))
                            ],
                            "incomplete_grid_metrics_used_for_decision": False,
                        }
            if resource_stop_reason is None and resource_ledger.elapsed_seconds() >= resource_ledger.max_wall_seconds:
                resource_stop_reason = (
                    f"shared pair wall ceiling reached after {arm} review at update {update}"
                )
                resource_ledger.stop(f"{arm}: {resource_stop_reason}")
            current_review_decision["resource_stop_reason"] = resource_stop_reason
            current_review = {
                "arm": arm,
                "update": update,
                "stage": stage,
                "physical_optimizer_calls": physical_optimizer_calls,
                "organizer_optimizer_calls": organizer_optimizer_calls,
                "actual_optimizer_calls": optimizer_calls,
                "trainable_physical_parameter_count": int(sum(p.numel() for p in physical_parameters)),
                "physical_parameter_delta_l2_from_run2103": _parameter_delta_l2(physical_parameters, initial_physical),
                "organizer_parameter_delta_l2_from_all_access_init": (
                    None if organizer is None else _parameter_delta_l2(list(organizer.parameters()), initial_organizer)
                ),
                "risk_duals": {role: float(duals[index].detach().cpu()) for index, role in enumerate(ROLE_NAMES)},
                "training_role_losses_last_step_mps2": dict(physical_role_numbers),
                "training_role_risk_last_step": risk_report,
                "sampled_development_and_train_review": review,
                "fixed_g2_reference_by_labeled_row": review["fixed_g2_reference_by_labeled_row"],
                "fixed_g2_reference_hard_forward_equivalents": review[
                    "fixed_g2_hard_forward_equivalents"
                ],
                "predeclared_early_stop_check": current_review_decision,
                "resource_stop_reason": resource_stop_reason,
                "soft_qe_support_audit": soft_qe,
                "soft_qe_saturation_remedy": saturation_remedy,
                "fidelity_work_gate": budget_work_state,
                "work_proxy_weight_current": (
                    base_work_weight
                    if organizer is not None and work_enabled and not budget_work_gate.suspended
                    else 0.0
                ),
                "work_proxy_weight_applied_on_review_update": row_metric["work_proxy_weight"],
                "work_proxy_weight_current_semantics": (
                    "available on stage-3 bridge updates only; stage-1/2 and full-access replay apply zero"
                ),
                "work_proxy_mean_since_arm_start": (
                    float(np.mean(latest_step_work)) if latest_step_work else None
                ),
                "actual_logical_work_semantics": WORK_INTERPRETATION,
                "full_native_grid_review": grid_review,
                "stage_update_counts": dict(stage_updates),
                "training_layout_visit_counts": dict(layout_visit_counter),
                "training_row_visit_counts": dict(row_visit_counter),
                "training_visit_coverage": _row_visit_coverage(
                    train_rows, view.metadata["layout_index"], row_visit_counter
                ),
                "stage3_visit_coverage": _row_visit_coverage(
                    train_rows, view.metadata["layout_index"], stage3_row_visit_counter
                ),
                "stage3_layout_visit_counts": dict(stage3_layout_visit_counter),
                "stage3_row_visit_counts": dict(stage3_row_visit_counter),
                "complete_native_training_row_passes": min(
                    (row_visit_counter.get(train_row, 0) for train_row in training_row_set),
                    default=0,
                ),
                "resource_ledger": resource_ledger.snapshot(),
                "role_catalogue_cache": role_catalogue_cache.summary(),
                "review_compute_seconds": time.perf_counter() - review_started,
                "full_access_execution": "policy-free Dense native path",
                "packet_execution": "external hard plan with dense_masked reference executor",
                "bypass_disclosure": "the existing global/coarse/local Dense context paths remain active outside the five typed fine routes",
            }
            review_history.append(current_review)
            _write_jsonl(arm_dir / "reviews.jsonl", review_history)
            _write_jsonl(arm_dir / "updates.jsonl", metrics)
            _write_arm_checkpoint(
                arm_dir,
                update=update,
                schedule_cursor=schedule_cursor,
                complete_native_epochs=min(
                    (row_visit_counter.get(train_row, 0) for train_row in training_row_set),
                    default=0,
                ),
                model=model,
                physical_optimizer=physical_optimizer,
                organizer=organizer,
                organizer_optimizer=organizer_optimizer,
                arm=arm,
                config=run_config,
                normalizer=normalizer,
                split_provenance=split_provenance,
                budgets=budgets,
                duals=duals,
                optimizer_calls=optimizer_calls,
                hard_forward_equivalents=hard_forward_equivalents,
                soft_forward_equivalents=soft_forward_equivalents,
                reviews=review_history,
                resource_ledger=resource_ledger.snapshot(),
            )
            _json_write(arm_dir / "status.json", {
                "status": (
                    "resource_stopped" if resource_stop_reason is not None
                    else "early_stopped" if early_stop_reason is not None
                    else "running" if update < max_updates else "scheduled_updates_complete"
                ),
                "last_review_update": update,
                "configured_update_limit": max_updates,
                "physical_optimizer_calls": physical_optimizer_calls,
                "organizer_optimizer_calls": organizer_optimizer_calls,
                "actual_optimizer_calls": optimizer_calls,
                "early_stop_reason": early_stop_reason,
                "resource_ledger": resource_ledger.snapshot(),
                "saturation_remedy_events": len(saturation_episodes),
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            print(
                f"[windfarm-joint] arm={arm} update={update}/{max_updates} stage={stage} "
                f"loss={float(loss.detach().cpu()):.5g} calls={optimizer_calls}"
            )
        if resource_stop_reason is not None or early_stop_reason is not None:
            break
    elapsed = time.perf_counter() - arm_started
    _write_jsonl(arm_dir / "updates.jsonl", metrics)
    _write_jsonl(arm_dir / "reviews.jsonl", review_history)
    complete_native_epochs = min(
        (row_visit_counter.get(train_row, 0) for train_row in training_row_set),
        default=0,
    )
    final_status = (
        "resource_stopped" if resource_stop_reason is not None
        else "early_stopped" if early_stop_reason is not None
        else "complete"
    )
    _write_arm_checkpoint(
        arm_dir,
        update=physical_optimizer_calls,
        schedule_cursor=schedule_cursor,
        complete_native_epochs=complete_native_epochs,
        model=model,
        physical_optimizer=physical_optimizer,
        organizer=organizer,
        organizer_optimizer=organizer_optimizer,
        arm=arm,
        config=run_config,
        normalizer=normalizer,
        split_provenance=split_provenance,
        budgets=budgets,
        duals=duals,
        optimizer_calls=optimizer_calls,
        hard_forward_equivalents=hard_forward_equivalents,
        soft_forward_equivalents=soft_forward_equivalents,
        reviews=review_history,
        resource_ledger=resource_ledger.snapshot(
            status="resource_stopped" if resource_stop_reason is not None else "running"
        ),
    )
    if resource_stop_reason is None and resource_ledger.elapsed_seconds() >= resource_ledger.max_wall_seconds:
        resource_stop_reason = f"shared pair wall ceiling reached while saving {arm} terminal state"
        resource_ledger.stop(f"{arm}: {resource_stop_reason}")
        final_status = "resource_stopped"
    summary = {
        "status": final_status,
        "arm": arm,
        "configured_update_limit": max_updates,
        "scheduled_update_cursor": schedule_cursor,
        "physical_optimizer_calls": physical_optimizer_calls,
        "organizer_optimizer_calls": organizer_optimizer_calls,
        "actual_optimizer_calls": optimizer_calls,
        "hard_forward_equivalents": hard_forward_equivalents,
        "soft_forward_equivalents": soft_forward_equivalents,
        "optimizer_updates_are_not_epochs": True,
        "complete_native_training_row_passes": complete_native_epochs,
        "native_epoch_semantics": "one pass visits every post-exclusion training row at least once",
        "resource_stop_reason": resource_stop_reason,
        "early_stop_reason": early_stop_reason,
        "continuation_is_exactly_resumable": False,
        "resource_ledger": resource_ledger.snapshot(status=final_status),
        "stage_update_counts": dict(stage_updates),
        "training_layout_visit_counts": dict(layout_visit_counter),
        "training_row_visit_counts": dict(row_visit_counter),
        "training_visit_coverage": _row_visit_coverage(
            train_rows, view.metadata["layout_index"], row_visit_counter
        ),
        "stage3_visit_coverage": _row_visit_coverage(
            train_rows, view.metadata["layout_index"], stage3_row_visit_counter
        ),
        "stage3_layout_visit_counts": dict(stage3_layout_visit_counter),
        "stage3_row_visit_counts": dict(stage3_row_visit_counter),
        "physical_parameter_delta_l2_from_run2103": _parameter_delta_l2(physical_parameters, initial_physical),
        "organizer_parameter_delta_l2_from_all_access_init": (
            None if organizer is None else _parameter_delta_l2(list(organizer.parameters()), initial_organizer)
        ),
        "review_count": len(review_history),
        "saturation_remedy_events": saturation_episodes,
        "fidelity_work_gate_final": {
            "consecutive_failed_training_audits": budget_work_gate.consecutive_failures,
            "consecutive_clear_training_audits": budget_work_gate.consecutive_clears,
            "work_incentive_suspended_for_reference_fidelity": budget_work_gate.suspended,
            "training_only": True,
        } if organizer is not None else None,
        "elapsed_wall_seconds": elapsed,
        "final_checkpoint": str(arm_dir / "checkpoints" / "latest.pt"),
        "source_checkpoint": preflight["source_checkpoint"],
        "split_provenance": _json_safe(split_provenance),
        "physical_role_budgets_mps": dict(budgets),
        "loss_estimand": "equal mean of separate quadrature-stratified physical role MSEs, normalized by fixed role budgets",
        "work_interpretation": WORK_INTERPRETATION,
    }
    _json_write(arm_dir / "summary.json", summary)
    _json_write(arm_dir / "status.json", {
        "status": final_status,
        "last_update": schedule_cursor,
        "physical_optimizer_calls": physical_optimizer_calls,
        "actual_optimizer_calls": optimizer_calls,
        "resource_stop_reason": resource_stop_reason,
        "early_stop_reason": early_stop_reason,
    })
    return summary


def _validate_joint_request(config: Mapping[str, Any], request: Any) -> None:
    """Reject generic CLI overrides this workflow does not consume."""

    training = config.get("training", {})
    joint = training.get("joint_forward", {}) if isinstance(training, Mapping) else {}
    run_config = config.get("run", {})
    if getattr(request, "workflow", "forward") != "forward":
        raise ValueError("joint WindFarm maturation only supports the forward workflow")
    for name in (
        "resume_checkpoint",
        "initialize_checkpoint",
        "local_checkpoint",
        "checkpoint",
        "saved_root",
        "dataset",
        "output_dir",
        "epochs",
        "max_train_batches",
        "max_val_batches",
    ):
        value = getattr(request, name, None)
        if value is not None:
            raise ValueError(f"joint WindFarm workflow does not support the --{name.replace('_', '-')} override")
    if tuple(getattr(request, "extra_args", ()) or ()):
        raise ValueError("joint WindFarm workflow rejects unparsed run-affecting extra arguments")
    requested_run_id = getattr(request, "run_id", None)
    configured_run_id = run_config.get("id") if isinstance(run_config, Mapping) else None
    if requested_run_id is not None and str(requested_run_id) != str(configured_run_id):
        raise ValueError("requested run ID differs from the joint profile run ID")
    requested_run_name = getattr(request, "run_name", None)
    configured_run_name = run_config.get("name") if isinstance(run_config, Mapping) else None
    if requested_run_name is not None and str(requested_run_name) != str(configured_run_name):
        raise ValueError("requested run name differs from the joint profile run name")
    requested_device = getattr(request, "device", None)
    configured_device = training.get("device") if isinstance(training, Mapping) else None
    if requested_device is not None and str(requested_device) != "cuda:0":
        raise ValueError("joint WindFarm device override must be logical cuda:0 with CUDA_VISIBLE_DEVICES=2")
    if configured_device != "cuda:0":
        raise ValueError("joint WindFarm profile must select logical cuda:0 with CUDA_VISIBLE_DEVICES=2")
    if training.get("init_checkpoint_path"):
        raise ValueError("joint WindFarm workflow owns its Run2103 source and rejects training.init_checkpoint_path")
    del joint  # Its recognized update/preflight overrides are consumed explicitly below.


def run_joint_forward_from_config(
    config: Mapping[str, Any],
    request: Any,
    *,
    run_dir_override: Path | None = None,
) -> int:
    """Run the preflight gate and then both matched arms from Run2103."""

    cfg = copy.deepcopy(dict(config))
    training_cfg = dict(cfg.get("training", {}))
    joint_cfg = dict(training_cfg.get("joint_forward", {}))
    if not joint_cfg:
        raise ValueError("training.joint_forward settings are required for this workflow")
    _validate_joint_request(cfg, request)
    requested_updates = getattr(request, "max_updates", None)
    max_updates = int(requested_updates if requested_updates is not None else joint_cfg["max_updates"])
    configured_updates = int(joint_cfg["max_updates"])
    if not 1 <= max_updates <= min(configured_updates, MAX_PHYSICAL_UPDATES_PER_ARM):
        raise ValueError("matched WindFarm arms are bounded to 1–6000 physical optimizer updates each")
    preflight_env = os.environ.get("HONF_JOINT_PREFLIGHT_ONLY")
    if preflight_env is not None and preflight_env.strip().lower() not in {"1", "true", "yes", "0", "false", "no"}:
        raise ValueError("HONF_JOINT_PREFLIGHT_ONLY must be one of 1/true/yes or 0/false/no")
    preflight_only = bool(joint_cfg.get("preflight_only", False)) or (
        preflight_env is not None and preflight_env.strip().lower() in {"1", "true", "yes"}
    )
    run_dir = Path(run_dir_override or cfg.get("paths", {}).get("saved_model_dir", "Trained_Results/WindFarm/HONF_Forward_Runs/Run_2110_joint"))
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    # Start the cap before checkpoint, dataset, and G2 preflight work. The
    # external job timer separately covers command-line startup and the round.
    resource_ledger = RunResourceLedger(
        run_dir,
        max_wall_seconds=int(joint_cfg.get("max_wall_seconds", DEFAULT_PAIR_WALL_SECONDS)),
        max_attempted_optimizer_calls=int(
            joint_cfg.get("max_attempted_optimizer_calls", MAX_PAIR_ATTEMPTED_OPTIMIZER_CALLS)
        ),
    )
    source_checkpoint = _resolved_path(joint_cfg["source_checkpoint_path"])
    source_payload = load_trusted_checkpoint(source_checkpoint, map_location="cpu")
    dataset_cfg = dict(cfg["dataset"])
    target_model_payload = dict(cfg.get("model", {}).get("core_honf", {}))
    _source_config, normalizer, checkpoint_sha256 = _validate_run2103_source(
        source_checkpoint,
        source_payload,
        dataset_cfg,
        target_model_payload,
    )
    seed = int(training_cfg.get("seed", joint_cfg.get("sample_seed", 2103)))
    sample_seed = int(joint_cfg.get("sample_seed", seed))
    set_seed(seed)
    selected_device = select_device(getattr(request, "device", None) or training_cfg.get("device"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if selected_device.type != "cuda" or visible.strip() != "2":
        raise RuntimeError(
            "joint WindFarm training requires an explicitly granted physical GPU 2 run with CUDA_VISIBLE_DEVICES=2"
        )
    if selected_device.index not in (None, 0):
        raise ValueError("with CUDA_VISIBLE_DEVICES=2, select logical cuda:0")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_VISIBLE_DEVICES=2 was set, but logical cuda:0 is unavailable")
    gpu_mapping = {
        "CUDA_VISIBLE_DEVICES": visible,
        "logical_device": "cuda:0",
        "physical_device": 2,
        "device_name": torch.cuda.get_device_name(0),
        "device_index": torch.cuda.current_device(),
        "slot_grant_required": True,
    }
    volume_path = _resolved_path(dataset_cfg["volume_path"])
    compact_path = _resolved_path(dataset_cfg["compact_path"])
    model_config = build_windfarm_forward_config(target_model_payload)
    view = WindFarmNativeView(
        volume_path,
        compact_metadata=_compact_metadata(compact_path),
        token_shape=tuple(dataset_cfg.get("env_token_shape", ENV_TOKEN_SHAPE)),
        include_receiver_anchors=True,
    )
    stored_split = source_payload.get("split_indices")
    if not isinstance(stored_split, Mapping):
        raise TypeError("Run2103 selected checkpoint lacks its train/validation/test split indices")
    canonical_split = make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    split = GroupSplit(
        train=np.asarray(stored_split["train"], dtype=np.int64),
        validation=np.asarray(stored_split["validation"], dtype=np.int64),
        test=np.asarray(stored_split["test"], dtype=np.int64),
        metadata=dict(source_payload.get("split_metadata") or canonical_split.metadata),
    )
    if not all(np.array_equal(getattr(split, name), getattr(canonical_split, name)) for name in ("train", "validation", "test")):
        raise ValueError("Run2103 stored split differs from the current canonical seed-42 group split")
    report, panel_manifest, documents, frozen_panel_split = _load_g2_training_documents(
        view,
        split,
        joint_cfg,
        checkpoint_sha256,
    )
    layout_indices = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    train_rows = _training_rows_without_held_layouts(split.train, layout_indices)
    held_layout_set = set(map(int, ORGANIZER_DEVELOPMENT_LAYOUT_INDICES))
    held_rows = [int(row) for row in split.train if int(layout_indices[int(row)]) in held_layout_set]
    validation_rows = list(map(int, split.validation))
    labeled_rows = sorted(map(int, documents))
    if not set(labeled_rows).issubset(set(train_rows)):
        raise ValueError("a G2-labeled row is missing from the post-exclusion physical training subset")
    if set(held_rows).intersection(train_rows):
        raise RuntimeError("held organizer-development rows leaked into new student updates")
    train_hash = _rows_hash(train_rows)
    held_hash = _rows_hash(sorted(held_rows))
    validation_hash = _rows_hash(validation_rows)
    development_rows = sorted(set(held_rows) | set(validation_rows))
    dev_hash = _rows_hash(development_rows)
    split_provenance = {
        "source_split_seed": 42,
        "source_split_train_row_count": len(split.train),
        "source_split_validation_row_count": len(split.validation),
        "student_training_row_count_after_organizer_dev_exclusion": len(train_rows),
        "student_training_rows_sha256": train_hash,
        "organizer_development_layout_indices_excluded_from_updates": sorted(held_layout_set),
        "organizer_development_rows_sha256": held_hash,
        "original_validation_rows_sha256": validation_hash,
        "combined_development_rows_sha256": dev_hash,
        "combined_development_row_count": len(development_rows),
        "g2_supervision_training_rows": labeled_rows,
        "g2_supervision_source_search_sha256": str(report.get("checkpoint_sha256")),
        "g2_training_layout_indices": list(map(int, frozen_panel_split["training_layout_indices"])),
        "g2_development_layout_indices_labels_not_loaded": list(map(int, frozen_panel_split["development_layout_indices"])),
        "test_split_labels_or_rows_opened": False,
        "test_split_row_count_not_used": len(split.test),
        "student_training_rows": train_rows,
        "development_rows": development_rows,
        "training_layouts_with_visits": sorted({int(layout_indices[row]) for row in train_rows}),
    }
    role_catalogue_cache = NativeRoleCatalogueCache(
        max_cached_bytes=int(
            joint_cfg.get("role_catalogue_cache_max_bytes", DEFAULT_ROLE_CATALOGUE_CACHE_MAX_BYTES)
        )
    )
    schedule = _make_schedule(
        max_updates=max_updates,
        train_rows=train_rows,
        labeled_rows=labeled_rows,
        seed=sample_seed,
    )
    schedule_hash = _canonical_json_hash(schedule)
    schedule_review_updates = sorted({
        int(update)
        for update in joint_cfg.get("review_updates", (150, 500, 1500))
        if 1 <= int(update) <= max_updates
    } | {max_updates})
    planned_visit_coverage_by_review = {
        str(update): {
            "all_stages": _scheduled_row_visit_coverage(
                schedule,
                through_update=update,
                train_rows=train_rows,
                layout_indices_by_row=layout_indices,
            ),
            "stage3_only": _scheduled_row_visit_coverage(
                schedule,
                through_update=update,
                train_rows=train_rows,
                layout_indices_by_row=layout_indices,
                stage="joint_hard_packet_refinement",
            ),
        }
        for update in schedule_review_updates
    }
    review_train_rows, review_dev_rows, review_validation_rows = _select_review_rows(view, train_rows, validation_rows)
    requested_preflight_row = int(joint_cfg.get("preflight_row", 12))
    preflight_row = requested_preflight_row if requested_preflight_row in train_rows else int(train_rows[0])
    g2_audit_rows = _g2_review_rows(view, labeled_rows)
    audit_rows = list(dict.fromkeys(
        review_train_rows + review_dev_rows + review_validation_rows[:2] + g2_audit_rows
    ))
    if max_updates >= 500 and (len(review_train_rows) < 2 or len(review_dev_rows) < 2):
        raise ValueError("update-500/1500 native-grid reviews require two train and two held-dev rows")
    counts = dict(joint_cfg.get("role_query_counts", DEFAULT_ROLE_QUERY_COUNTS))
    audit_counts = dict(joint_cfg.get("audit_query_counts", DEFAULT_AUDIT_QUERY_COUNTS))
    if set(counts) != set(ROLE_NAMES) or set(audit_counts) != set(ROLE_NAMES):
        raise ValueError("fresh and audit role counts must cover exactly the five protected WindFarm roles")
    cases: dict[int, Any] = {row: view.run(row) for row in set(audit_rows) | {preflight_row}}
    audit_samples = {
        row: sample_native_role_queries(
            cases[row],
            _native_rng(sample_seed, 0, row, 991),
            audit_counts,
            catalogue_cache=role_catalogue_cache,
        )
        for row in audit_rows
    }
    preflight_sample = sample_native_role_queries(
        cases[preflight_row],
        _native_rng(sample_seed, 0, preflight_row, 701),
        counts,
        catalogue_cache=role_catalogue_cache,
    )
    preflight_batch = _batch_from_sample(cases[preflight_row], preflight_sample, normalizer, device)
    receiver_chunk_size = int(joint_cfg.get("receiver_chunk_size", model_config.interface_model.receiver_chunk_size))
    full_grid_chunk_size = int(joint_cfg.get("full_grid_chunk_size", receiver_chunk_size))
    if receiver_chunk_size < 1 or full_grid_chunk_size < 1:
        raise ValueError("WindFarm receiver chunk sizes must be positive")

    # Materialize two independently constructed arms from the exact same
    # checkpoint bytes before any optimizer or formal update exists.
    full_model = _new_model_from_source(source_payload, normalizer, preflight_batch, device)
    packet_model = _new_model_from_source(source_payload, normalizer, preflight_batch, device)
    full_state_hash = _model_state_sha256(full_model)
    packet_state_hash = _model_state_sha256(packet_model)
    if full_state_hash != packet_state_hash:
        raise RuntimeError("W-full and W-packet did not start from identical Run2103 parameters")
    full_model.eval()
    packet_model.eval()
    with torch.inference_mode():
        full_core = full_model.core
        full_core.backend.set_cover_mode("full_access")
        full_core.backend.set_cover_executor("dense_masked")
        full_encoded = full_core.encode_case(preflight_batch)
        full_prediction = _core_prediction(
            full_core,
            full_encoded,
            preflight_batch,
            receiver_chunk_size=receiver_chunk_size,
        )
        packet_core = packet_model.core
        packet_core.backend.set_cover_mode("external")
        packet_core.backend.set_cover_executor("dense_masked")
        packet_encoded = packet_core.encode_case(preflight_batch)
        explicit_full = _full_plan(packet_core, packet_encoded)
        if not explicit_full.is_full_access() or bool((explicit_full.split_gates != 0).any()):
            raise RuntimeError("preflight root plan is not explicit all-access")
        explicit_prediction = _core_prediction(
            packet_core,
            packet_encoded,
            preflight_batch,
            receiver_chunk_size=receiver_chunk_size,
            fixed_plan=explicit_full,
        )
    full_physical = _physical_from_standardized(full_prediction, normalizer)
    explicit_physical = _physical_from_standardized(explicit_prediction, normalizer)
    parity_max_abs = float((full_physical - explicit_physical).abs().max().detach().cpu())
    parity_passed = bool(torch.allclose(
        full_physical,
        explicit_physical,
        atol=float(native_cover.COLD_WARM_ABS_TOLERANCE_MPS),
        rtol=float(native_cover.COLD_WARM_REL_TOLERANCE),
    ))
    if not parity_passed:
        raise RuntimeError(f"policy-free Dense/root-full explicit parity failed: {parity_max_abs} m/s")
    packet_core.backend.set_cover_mode("external")
    packet_core.backend.set_cover_executor("dense_masked")
    preflight_encoded = packet_core.encode_case(preflight_batch)
    preflight_tree = packet_core.backend.build_case_trees(preflight_encoded)[0]
    organizer = _build_organizer(packet_core, preflight_encoded, preflight_tree, joint_cfg, device)
    _enable_physical_trainable_scope(packet_model)
    gradient_check = _physical_gradient_check(
        packet_model,
        organizer,
        preflight_batch,
        receiver_chunk_size=receiver_chunk_size,
    )
    if not gradient_check["hard_plan_is_full_access"]:
        raise RuntimeError("preflight organizer hard initialization is not all-access")

    # The risk calibration is source-only training evidence. It reads no
    # organizer-development or original-validation labels.
    baseline_layout_count = int(joint_cfg.get("baseline_layout_count", 8))
    layouts = select_training_layouts(view, split.train, count=12)
    frozen = freeze_organizer_layout_split(layouts)
    train_panel_layouts = [
        item for item in layouts
        if int(item.layout_index) in set(map(int, frozen["training_layout_indices"]))
    ][:baseline_layout_count]
    baseline_rows = [int(item.rows[0]) for item in train_panel_layouts]
    if len(baseline_rows) != baseline_layout_count:
        raise ValueError("physical role budget calibration did not find the requested number of train layouts")
    baseline_sq = {role: 0.0 for role in ROLE_NAMES}
    baseline_n = {role: 0 for role in ROLE_NAMES}
    teacher_predictions: dict[int, torch.Tensor] = {}
    audit_mse_by_row: dict[int, dict[str, float]] = {}
    with torch.inference_mode():
        for row in baseline_rows:
            case = view.run(row)
            sample = sample_native_role_queries(
                case,
                _native_rng(sample_seed, 0, row, 702),
                counts,
                catalogue_cache=role_catalogue_cache,
            )
            batch = _batch_from_sample(case, sample, normalizer, device)
            encoded = full_model.core.encode_case(batch)
            full_model.core.backend.set_cover_mode("full_access")
            prediction = _core_prediction(
                full_model.core,
                encoded,
                batch,
                receiver_chunk_size=receiver_chunk_size,
            )
            prediction_mps = _physical_from_standardized(prediction, normalizer)
            target_mps = _physical_from_standardized(batch.target_field, normalizer)
            role_mse = _role_mse_physical(prediction_mps, target_mps, sample.role_slices)
            for role in ROLE_NAMES:
                count = int(sample.role_sample_counts[role]) * 3
                baseline_sq[role] += float(role_mse[role].cpu()) * count
                baseline_n[role] += count
        for row in audit_rows:
            case = cases[row]
            sample = audit_samples[row]
            batch = _batch_from_sample(case, sample, normalizer, device)
            encoded = full_model.core.encode_case(batch)
            full_model.core.backend.set_cover_mode("full_access")
            prediction = _core_prediction(
                full_model.core,
                encoded,
                batch,
                receiver_chunk_size=receiver_chunk_size,
            )
            prediction_mps = _physical_from_standardized(prediction, normalizer)
            teacher_predictions[row] = prediction_mps.detach().cpu()
            target_mps = _physical_from_standardized(batch.target_field, normalizer)
            role_mse = _role_mse_physical(prediction_mps, target_mps, sample.role_slices)
            audit_mse_by_row[row] = {role: float(role_mse[role].cpu()) for role in ROLE_NAMES}
    baseline_rmse = {
        role: math.sqrt(baseline_sq[role] / max(baseline_n[role], 1))
        for role in ROLE_NAMES
    }
    budgets = role_budgets_from_baseline(
        baseline_rmse,
        delta=float(joint_cfg.get("role_budget_delta", 0.10)),
        absolute_floor_mps=float(joint_cfg.get("role_budget_floor_mps", 1.0e-5)),
    )
    run_config = copy.deepcopy(cfg)
    run_config.setdefault("training", {})["joint_forward"]["max_updates"] = max_updates
    _json_write(run_dir / "config_resolved.json", run_config)
    split_provenance.update({
        "g2_split_sha256": str(frozen["split_sha256"]),
        "g2_train_layout_indices": list(map(int, frozen["training_layout_indices"])),
        "g2_dev_layout_indices": list(map(int, frozen["development_layout_indices"])),
        "g2_report_status": report.get("status"),
        "g2_panel_anchor_measure_variant": panel_manifest.get("anchor_measure_variant"),
        "baseline_calibration_rows": baseline_rows,
        "baseline_calibration_rows_sha256": _rows_hash(baseline_rows),
    })
    preflight = {
        "status": "passed",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_run_id": "2103",
        "source_best_epoch": FROZEN_CHECKPOINT_EPOCH,
        "w_full_initial_parameter_sha256": full_state_hash,
        "w_packet_initial_parameter_sha256": packet_state_hash,
        "matched_initialization_equal": full_state_hash == packet_state_hash,
        "root_full_explicit_plan_parity": {
            "passed": parity_passed,
            "row_index": preflight_row,
            "query_count": int(preflight_batch.query_xy.shape[1]),
            "max_abs_physical_delta_mps": parity_max_abs,
            "absolute_tolerance_mps": native_cover.COLD_WARM_ABS_TOLERANCE_MPS,
            "relative_tolerance": native_cover.COLD_WARM_REL_TOLERANCE,
            "w_full_execution": "policy-free native Dense",
            "w_packet_preflight_execution": "external root-full explicit MechanismPlan with dense_masked executor",
        },
        "hard_value_soft_organizer_gradient_check": gradient_check,
        "role_budget_calibration": {
            "rows": baseline_rows,
            "training_only": True,
            "source": "intact Run2103 selected checkpoint against stored native training fields",
            "baseline_rmse_mps": baseline_rmse,
            "budget_rmse_mps": budgets,
            "delta": float(joint_cfg.get("role_budget_delta", 0.10)),
            "absolute_floor_mps": float(joint_cfg.get("role_budget_floor_mps", 1.0e-5)),
            "role_estimand": "equal quadrature-stratified role sampling measure, three velocity channels pooled",
            "teacher_reference_role_rmse_by_audit_row_mps": audit_mse_by_row,
        },
        "train_dev_split_provenance": split_provenance,
        "sampling_schedule_sha256": schedule_hash,
        "sampling_schedule_first_rows": list(schedule[: min(12, len(schedule))]),
        "sampling_schedule_visit_coverage_by_review": planned_visit_coverage_by_review,
        "verified_g2_source_review_rows": g2_audit_rows,
        "sampling_role_counts": counts,
        "audit_role_counts": audit_counts,
        "native_role_catalogue_cache": role_catalogue_cache.summary(),
        "fresh_native_sampling": True,
        "dev_labels_used_for_organizer_training": False,
        "g2_labels_loaded": labeled_rows,
        "unlabeled_stage2_cases_use_full_access": True,
        "work_semantics": WORK_INTERPRETATION,
        "gpu_mapping": gpu_mapping,
        "preflight_only": preflight_only,
        "resource_limits": resource_ledger.snapshot(),
        "continuation_semantics": (
            "finite initial run only; checkpoints preserve model and optimizer state for audit but are not an exact-resume interface"
        ),
        "pre_optimizer_gate": "passed; formal optimizer calls begin only after this JSON is atomically written",
    }
    _json_write(run_dir / "preflight.json", preflight)
    _json_write(run_dir / "status.json", {
        "status": "preflight_only_complete" if preflight_only else "preflight_passed",
        "updates_started": 0,
        "optimizer_calls": 0,
        "max_updates_per_arm": max_updates,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    if preflight_only:
        summary = {
            "status": "preflight_only_complete",
            "profile_name": cfg.get("profile_name"),
            "configured_max_updates_per_arm": max_updates,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": checkpoint_sha256,
            "preflight_status": "passed",
            "preflight_path": str(run_dir / "preflight.json"),
            "optimizer_calls": 0,
            "optimizer_constructed": False,
            "train_dev_split_provenance": split_provenance,
            "gpu_mapping": gpu_mapping,
            "continuation_semantics": preflight["continuation_semantics"],
        }
        _json_write(run_dir / "summary.json", summary)
        resource_ledger.finalize("preflight_only_complete")
        return 0
    row_visit_counter: Counter[int] = Counter()
    arm_summaries: dict[str, Any] = {}
    pair_early_stop_cursor: int | None = None
    for arm in ("w_full", "w_packet"):
        blocked_before_arm = resource_ledger.block_reason()
        if blocked_before_arm is not None:
            resource_ledger.stop(f"before {arm}: {blocked_before_arm}")
            break
        # Each arm independently reloads the exact same selected checkpoint.
        arm_layout_visit_counter: Counter[int] = Counter()
        arm_visit_counter: Counter[int] = Counter()
        arm_schedule = schedule
        if arm == "w_packet" and pair_early_stop_cursor is not None:
            arm_schedule = [item for item in schedule if int(item["update"]) <= pair_early_stop_cursor]
        arm_summaries[arm] = _run_arm(
            arm=arm,
            source_payload=source_payload,
            normalizer=normalizer,
            view=view,
            schedule=arm_schedule,
            train_rows=train_rows,
            labeled_rows=labeled_rows,
            documents=documents,
            audit_rows=audit_rows,
            audit_samples=audit_samples,
            teacher_predictions=teacher_predictions,
            cases=cases,
            budgets=budgets,
            split_provenance=split_provenance,
            run_config=run_config,
            joint_cfg=joint_cfg,
            device=device,
            run_dir=run_dir,
            receiver_chunk_size=receiver_chunk_size,
            preflight=preflight,
            layout_visit_counter=arm_layout_visit_counter,
            row_visit_counter=arm_visit_counter,
            role_catalogue_cache=role_catalogue_cache,
            resource_ledger=resource_ledger,
        )
        arm_summaries[arm]["training_layout_visit_counts"] = dict(arm_layout_visit_counter)
        row_visit_counter.update(arm_visit_counter)
        if arm == "w_full" and arm_summaries[arm]["status"] == "early_stopped":
            pair_early_stop_cursor = int(arm_summaries[arm]["scheduled_update_cursor"])
        del arm_layout_visit_counter, arm_visit_counter
        if device.type == "cuda":
            torch.cuda.empty_cache()
    total_actual_calls = sum(int(item["actual_optimizer_calls"]) for item in arm_summaries.values())
    if total_actual_calls > 18000:
        raise RuntimeError(f"joint WindFarm pair exceeded the 18,000 actual optimizer-call ceiling: {total_actual_calls}")
    terminal_status = _pair_terminal_status(arm_summaries)
    if resource_ledger.elapsed_seconds() >= resource_ledger.max_wall_seconds:
        resource_ledger.stop("shared pair wall ceiling reached before pair finalization")
        terminal_status = "resource_stopped_or_incomplete"
    summary = {
        "status": terminal_status,
        "profile_name": cfg.get("profile_name"),
        "configured_max_updates_per_arm": max_updates,
        "profile_max_updates_are_not_a_capacity_claim": True,
        "initial_experiment_limit_per_arm": max_updates,
        "pair_review_gate_stop_update": pair_early_stop_cursor,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": checkpoint_sha256,
        "preflight_status": "passed",
        "preflight_path": str(run_dir / "preflight.json"),
        "arms": arm_summaries,
        "matched_row_schedule_sha256": schedule_hash,
        "total_actual_optimizer_calls_across_pair": total_actual_calls,
        "optimizer_call_ceiling": resource_ledger.max_attempted_optimizer_calls,
        "optimizer_call_ceiling_per_arm": MAX_OPTIMIZER_CALLS_PER_ARM,
        "physical_update_ceiling_per_arm": 6000,
        "new_reference_oracle_forward_calls": 0,
        "new_physical_solves": 0,
        "test_split_labels_or_rows_opened": False,
        "gpu_mapping": gpu_mapping,
        "resource_ledger": resource_ledger.snapshot(status=terminal_status),
        "continuation_semantics": preflight["continuation_semantics"],
        "train_dev_split_provenance": split_provenance,
        "interpretation": {
            "physical_reference": "stored cell-centred OpenFOAM CFD fields; no new CFD solve",
            "organizational_coverage": "typed fine MM/ME/EM/QM/QE routes only; frozen global/coarse/local paths remain active",
            "logical_work": WORK_INTERPRETATION,
            "updates": "one per-case native sample per physical optimizer update; not epochs",
        },
    }
    status_payload = {
        "status": summary["status"],
        "configured_update_limit_per_arm": max_updates,
        "actual_scheduled_update_cursor_by_arm": {
            arm: int(item["scheduled_update_cursor"]) for arm, item in arm_summaries.items()
        },
        "total_actual_optimizer_calls_across_pair": total_actual_calls,
        "summary": str(run_dir / "summary.json"),
    }
    _json_write(run_dir / "summary.json", summary)
    _json_write(run_dir / "status.json", status_payload)
    if resource_ledger.elapsed_seconds() >= resource_ledger.max_wall_seconds:
        resource_ledger.stop("shared pair wall ceiling reached during pair finalization")
        summary["status"] = "resource_stopped_or_incomplete"
        summary["resource_ledger"] = resource_ledger.snapshot(status=summary["status"])
        status_payload["status"] = summary["status"]
        _json_write(run_dir / "summary.json", summary)
        _json_write(run_dir / "status.json", status_payload)
    resource_ledger.finalize(summary["status"])
    return 0


__all__ = [
    "ROLE_NAMES",
    "NativeRoleSample",
    "role_budgets_from_baseline",
    "role_risk_report",
    "run_joint_forward_from_config",
    "sample_native_role_queries",
]
