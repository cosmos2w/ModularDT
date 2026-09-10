#!/usr/bin/env python3
"""Bounded diagnostics for the regional-response HONF experiment.

This module is the single study driver for the regional-response work.  It
keeps the existing Stage-3 loaders, physical sample adapter, canonical error
reducer, and runtime chunking helpers in one place.  It deliberately does not
train a managed run or alter a checkpoint.  The ``interventions`` and
``frozen`` tasks are the Stage-I stored-checkpoint measurements; ``timing``
and ``smoke`` are opt-in execution tasks used after the candidate is built.

Run from ``HONF_Proj/``.  Every checkpoint uses explicit ``LABEL=PATH``
syntax.  Generated files should be placed under the one structured study
directory described by the development plan.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "Case_ThermalChannel" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))

# The existing Stage-3 tool owns trusted checkpoint loading, dataset/sample
# adaptation, canonical target errors, and physical query execution.  Keep
# this import explicit rather than cloning those helpers here.
from channelthermal.data.collation import ChannelThermalBatchCollator
from channelthermal.environment import ChannelThermalEnvironment, ChannelThermalEnvironmentBuilder
from channelthermal.evaluation.prepared import select_sample
from channelthermal.training.epoch import (
    effective_local_loss_weights,
    effective_port_condition_settings,
    predicted_consistency_weight_for_epoch,
    run_epoch,
)
from profile_stage2_sparse_inference import (  # type: ignore[import-not-found]
    _agreement_tolerances,
    _new_family_encoding_and_layout,
    _output_agreement,
    _public_output_snapshot,
)
from profile_stage2_sparse_inference import (
    measure as _profile_measure,
)
from run_stage3_interface_study import (  # type: ignore[import-not-found]
    DEFAULT_QUERY_BATCH_SIZE,
    _canonical_ground_truth_errors,
    _checkpoint_record,
    _error_deltas,
    _forward_batch,
    _load_dataset,
    _load_model_spec,
    _load_raw_sample,
    _module_radius,
    _prepared_synthetic_forward,
    _query_points,
    _relative_difference,
    _runtime_receiver_chunk_size,
    _scaled_domain,
    _state_keys,
    _synthetic_centers,
    _synthetic_query_tensor,
    make_batch,
    parse_checkpoint_specs,
    parse_shape,
    select_device,
    temporary_domain,
    write_json,
)

from honf_forward_core.interface_fields.regional_response import pool_region_weighted

ANCHOR_CASE_IDS = ("0273", "0653", "0298", "0302")
# These are fixed before any coarsening result is inspected.  They are four
# entries in the established 20-case manifest and span ordinary layouts in
# addition to the four difficult anchors.
DEFAULT_REGIONAL_CASE_IDS = ANCHOR_CASE_IDS + ("0277", "0291", "0680", "0281")
DEFAULT_BLOCK_SHAPE = (2, 2)
DEFAULT_SHAPES = ((32, 768, 65536), (128, 3072, 262144))
_DENSE_PROJECTED_KEY_CACHE = "_regional_study_projected_environment_key"
_DENSE_PROJECTED_VALUE_CACHE = "_regional_study_projected_environment_value"
REGIONAL_CASE_SELECTION_REASONS = {
    "0273": "established Dense anchor; retained before results",
    "0653": "established Dense anchor; retained before results",
    "0298": "established Dense anchor; retained before results",
    "0302": "established Dense anchor; retained before results",
    "0277": "established 20-case manifest: M3, intermediate spacing, middle wall, low heat CV",
    "0281": "established 20-case manifest: M3, separated, near wall, low CV",
    "0291": "established 20-case manifest: M5, crowded, interior, medium CV",
    "0680": "established 20-case manifest: M10, crowded, near wall, medium CV",
}


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _json_value(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_value(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class RegionPartition:
    """Small adapter around the native regional pool contract."""

    region_ids: torch.Tensor
    weights: torch.Tensor
    coordinates: torch.Tensor
    mass: torch.Tensor
    centroids: torch.Tensor
    block_shape: tuple[int, int]

    @property
    def source_count(self) -> int:
        return int(self.region_ids.shape[1])

    @property
    def region_count(self) -> int:
        return int(self.mass.shape[1])

    def pool(self, values: torch.Tensor) -> torch.Tensor:
        return pool_region_weighted(
            values,
            self.region_ids,
            self.weights,
            self.coordinates,
        ).values


def build_region_partition(
    env_coords: torch.Tensor,
    env_weights: torch.Tensor,
    *,
    block_shape: tuple[int, int] = DEFAULT_BLOCK_SHAPE,
    region_ids: torch.Tensor | None = None,
) -> RegionPartition:
    """Build weighted masses/centroids from coordinates and positive weights.

    For the current ChannelThermal grid, memberships are derived from physical
    coordinates rather than token order.  Future adapters can pass an explicit
    ID vector; IDs are remapped per batch to a compact range while preserving
    the adapter's grouping.
    """

    coords = env_coords
    weights = env_weights
    if coords.ndim != 3 or weights.ndim != 2:
        raise ValueError("env_coords must be [B,E,D] and env_weights must be [B,E]")
    if coords.shape[:2] != weights.shape:
        raise ValueError("environment coordinates and weights disagree on [B,E]")
    if not torch.isfinite(coords).all() or not torch.isfinite(weights).all():
        raise ValueError("environment coordinates and weights must be finite")
    if region_ids is None:
        # The case adapter owns the grouping rule.  Unique coordinate ranks
        # permit duplicate quadrature samples while still preserving their
        # physical region membership.
        nx = int(torch.unique(coords[0, :, 0], sorted=True).numel())
        ny = int(torch.unique(coords[0, :, 1], sorted=True).numel())
        ids = torch.stack(
            [
                ChannelThermalEnvironmentBuilder.region_ids_from_coordinates(
                    coords[batch_index],
                    num_env_tokens_x=nx,
                    num_env_tokens_y=ny,
                    block_shape=block_shape,
                )
                for batch_index in range(coords.shape[0])
            ],
            dim=0,
        )
    else:
        raw = torch.as_tensor(region_ids, device=coords.device)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0).expand(coords.shape[0], -1)
        if raw.shape != weights.shape:
            raise ValueError("region_ids must align with environment weights as [B,E]")
        ids = raw.to(dtype=torch.long)
    # Use the production reduction once for geometry and once for each source
    # tensor.  It preserves dtype and does direct positive-mass division.
    ones = torch.ones((*coords.shape[:2], 1), device=coords.device, dtype=coords.dtype)
    pooled = pool_region_weighted(ones, ids, weights, coords)
    return RegionPartition(
        region_ids=ids,
        weights=weights,
        coordinates=coords,
        mass=pooled.mass,
        centroids=pooled.centroids,
        block_shape=tuple(int(value) for value in block_shape),
    )


def weighted_pool(values: torch.Tensor, partition: RegionPartition) -> torch.Tensor:
    """Public test/diagnostic wrapper for quadrature-weighted pooling."""

    return partition.pool(values)


def frozen_projected_attention(
    projected_query: torch.Tensor,
    projected_key: torch.Tensor,
    projected_value: torch.Tensor,
    partition: RegionPartition,
    *,
    fine_geometry_bias: torch.Tensor | None = None,
    regional_geometry_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read pooled projected per-head keys/values with one mass factor.

    This implements the frozen approximation from the plan.  If a test supplies
    a constant-within-region geometry score, ``regional_geometry_bias`` can be
    passed directly.  Runtime Dense coarsening uses centroid geometry bias.
    """

    if projected_query.ndim != 4 or projected_key.ndim != 4 or projected_value.ndim != 4:
        raise ValueError("projected attention tensors must have shape [B,A,N,H]")
    batch, heads, _query_count, head_dim = projected_query.shape
    if projected_key.shape[:2] != (batch, heads) or projected_value.shape[:2] != (batch, heads):
        raise ValueError("projected source batch/head shapes disagree")
    packed = torch.cat([projected_key.transpose(1, 2), projected_value.transpose(1, 2)], dim=-1)
    pooled = weighted_pool(packed, partition).transpose(1, 2)
    pooled_key, pooled_value = pooled.split(head_dim, dim=-1)
    scores = torch.matmul(projected_query, pooled_key.transpose(-1, -2)) / math.sqrt(float(head_dim))
    if regional_geometry_bias is not None:
        scores = scores + regional_geometry_bias
    elif fine_geometry_bias is not None:
        raise ValueError("frozen projected pooling requires explicit centroid geometry bias")
    scores = scores + torch.log(partition.mass)[:, None, None, :]
    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, pooled_value), attention


def _project_query(attention: Any, query: torch.Tensor) -> torch.Tensor:
    return attention.project_query(query)


def _project_source(attention: Any, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return attention.project_source(source)


def _read_projected(
    attention: Any,
    projected_query: torch.Tensor,
    projected_key: torch.Tensor,
    projected_value: torch.Tensor,
    *,
    bias: torch.Tensor,
    log_weights: torch.Tensor,
    return_attention: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return attention.read_projected(
        projected_query,
        projected_key,
        projected_value,
        bias=bias,
        log_weights=log_weights,
        return_attention=return_attention,
    )


def _dense_read_with_partition(
    backend: Any,
    state: Mapping[str, torch.Tensor],
    encoded: Any,
    receivers: torch.Tensor,
    receiver_features: torch.Tensor,
    partition: RegionPartition,
    *,
    module_enabled: bool = True,
    environment_enabled: bool = True,
    pooled_sources: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_routing_maps: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dense read with the plan's frozen projected K/V coarsening.

    Dense exposes the direct-module and environmental readers separately.  The
    diagnostic therefore composes those maintained helpers and only replaces
    the environmental source preparation/read.
    """

    module_context = (
        backend.read_module(state, encoded, receivers, receiver_features)
        if module_enabled
        else receivers.new_zeros(receivers.shape[0], receivers.shape[1], backend.hidden_dim)
    )
    environment_context = receivers.new_zeros(module_context.shape)
    aux: dict[str, torch.Tensor] = {
        "dense_module_context_norm": torch.linalg.vector_norm(module_context, dim=-1),
    }
    if environment_enabled:
        region_relative = (receivers[:, :, None, :] - partition.centroids[:, None, :, :]) / encoded.coordinate_scale
        region_bias = backend._mlp(
            backend.env_geometry_bias,
            backend.relative_fourier(region_relative),
        ).permute(0, 3, 1, 2)
        env_query = backend.env_query(receiver_features)
        projected_query = _project_query(backend.env_attention, env_query)
        if pooled_sources is None:
            source_key, source_value = _project_source(backend.env_attention, state["env_tokens"])
            packed = torch.cat([source_key.transpose(1, 2), source_value.transpose(1, 2)], dim=-1)
            pooled = weighted_pool(packed, partition).transpose(1, 2)
            head_dim = int(source_key.shape[-1])
            pooled_key, pooled_value = pooled.split(head_dim, dim=-1)
        else:
            pooled_key, pooled_value = pooled_sources
        environment_context, attention_map = backend.env_attention.read_projected(
            projected_query,
            pooled_key,
            pooled_value,
            bias=region_bias,
            log_weights=torch.log(partition.mass),
            return_attention=return_routing_maps,
        )
        if return_routing_maps and attention_map is not None:
            aux["dense_environment_attention"] = attention_map
    aux["dense_environment_context_norm"] = torch.linalg.vector_norm(environment_context, dim=-1)
    aux["dense_region_count"] = receivers.new_tensor(float(partition.region_count))
    aux["dense_environment_source_count"] = receivers.new_tensor(float(partition.source_count))
    return module_context + environment_context, aux


@contextlib.contextmanager
def frozen_dense_read(
    model: Any,
    *,
    block_shape: tuple[int, int] = DEFAULT_BLOCK_SHAPE,
    roles: Sequence[str] = ("p2_field",),
    module_enabled: bool = True,
    environment_enabled: bool = True,
) -> Iterator[dict[str, Any]]:
    """Temporarily replace Dense environmental reads with pooled projected K/V."""

    architecture = str(model.config.core_honf.forward_architecture)
    if architecture != "dense_pairwise_field":
        raise ValueError(f"frozen projected pooling requires Dense, got {architecture!r}")
    backend = model.core.backend
    original = backend.read
    state = {"calls": 0, "prepared_source_count": 0, "partitions": []}
    # Hold strong references for the duration of one physical forward so a
    # later Python allocation cannot reuse an old ``id(prepared_state)``.
    prepared_cache: list[tuple[Any, RegionPartition, tuple[torch.Tensor, torch.Tensor]]] = []

    def read(prepared_state: Mapping[str, torch.Tensor], encoded: Any, receivers: torch.Tensor, receiver_features: torch.Tensor, **kwargs: Any):
        state["calls"] += 1
        role = getattr(model.core, "_interface_read_role", None)
        if role not in {str(item) for item in roles}:
            return original(prepared_state, encoded, receivers, receiver_features, **kwargs)
        cached = next((entry for entry in prepared_cache if entry[0] is prepared_state), None)
        if cached is None:
            partition = build_region_partition(encoded.env_coords, encoded.env_weights, block_shape=block_shape)
            source_key, source_value = _project_source(backend.env_attention, prepared_state["env_tokens"])
            packed = torch.cat([source_key.transpose(1, 2), source_value.transpose(1, 2)], dim=-1)
            pooled = weighted_pool(packed, partition).transpose(1, 2)
            head_dim = int(source_key.shape[-1])
            pooled_sources = pooled.split(head_dim, dim=-1)
            cached = (prepared_state, partition, pooled_sources)
            prepared_cache.append((prepared_state, partition, pooled_sources))
            state["prepared_source_count"] += 1
            state["partitions"].append(partition)
        _, partition, pooled_sources = cached
        return _dense_read_with_partition(
            backend,
            prepared_state,
            encoded,
            receivers,
            receiver_features,
            partition,
            module_enabled=module_enabled,
            environment_enabled=environment_enabled,
            pooled_sources=pooled_sources,
            return_routing_maps=bool(kwargs.get("return_routing_maps", False)),
        )

    backend.read = read
    try:
        yield state
    finally:
        backend.read = original


@contextlib.contextmanager
def backend_read_zero(
    model: Any,
    *,
    roles: Sequence[str],
    component: str = "main",
) -> Iterator[None]:
    """Zero one read branch only while a tagged physical role is active."""

    backend = model.core.backend
    original = backend.read
    requested = {str(role) for role in roles}
    architecture = str(model.config.core_honf.forward_architecture)
    component = str(component)

    def read(*args: Any, **kwargs: Any):
        role = getattr(model.core, "_interface_read_role", None)
        if role not in requested:
            return original(*args, **kwargs)
        prepared, encoded, receivers, receiver_features = args[:4]
        return_maps = bool(kwargs.get("return_routing_maps", False))
        if component in {"main", "group"}:
            main, aux = original(*args, **kwargs)
            return torch.zeros_like(main), aux
        if architecture == "dense_pairwise_field" and component in {"module", "environment"}:
            if component == "module":
                environment_context, environment_attention = backend.read_environment(
                    prepared, encoded, receivers, receiver_features,
                    return_routing_maps=return_maps,
                )
                main = environment_context
                aux = {"dense_environment_context_norm": torch.linalg.vector_norm(environment_context, dim=-1)}
                if return_maps and environment_attention is not None:
                    aux["dense_environment_attention"] = environment_attention
            else:
                module_context = backend.read_module(prepared, encoded, receivers, receiver_features)
                main = module_context
                aux = {"dense_module_context_norm": torch.linalg.vector_norm(module_context, dim=-1)}
            return main, aux
        if architecture == "regional_response_honf" and component == "regional":
            module_context = backend.read_module(prepared, encoded, receivers, receiver_features)
            return module_context, {"dense_module_context_norm": torch.linalg.vector_norm(module_context, dim=-1)}
        if architecture == "regional_response_honf" and component == "module":
            regional_context, regional_attention = backend.read_regional(
                prepared, encoded, receivers, receiver_features,
                return_routing_maps=return_maps,
            )
            main = regional_context
            aux = {"regional_context_norm": torch.linalg.vector_norm(regional_context, dim=-1)}
            if return_maps and regional_attention is not None:
                aux["regional_environment_attention"] = regional_attention
            return main, aux
        raise RuntimeError(
            f"{component} intervention requires a backend component API for architecture {architecture!r}"
        )

    backend.read = read
    try:
        yield
    finally:
        backend.read = original


@contextlib.contextmanager
def common_read_zero(model: Any, *, role: str, component: str) -> Iterator[None]:
    """Zero one common coarse/local path only for a named read role."""

    common = model.core.common
    name = "read_coarse" if component == "coarse" else "read_local"
    original = getattr(common, name)

    def read(*args: Any, **kwargs: Any):
        value = original(*args, **kwargs)
        active = getattr(model.core, "_interface_read_role", None)
        if active != role:
            return value
        if isinstance(value, tuple):
            return torch.zeros_like(value[0]), torch.zeros_like(value[1])
        return torch.zeros_like(value)

    setattr(common, name, read)
    try:
        yield
    finally:
        setattr(common, name, original)


@contextlib.contextmanager
def _reader_phase_intervention(model: Any, *, roles: Sequence[str]) -> Iterator[None]:
    """Reader/group removal using the physical coupling role marker."""

    with backend_read_zero(model, roles=roles, component="main"):
        yield


def _phase_variant_context(model: Any, family: str, mode: str) -> contextlib.AbstractContextManager[Any]:
    """Map one plan intervention to role-scoped hooks."""

    if family == "reader":
        if mode == "p0":
            return _reader_phase_intervention(model, roles=("p0_port",))
        if mode == "p1_only":
            return _reader_phase_intervention(model, roles=("p1_refinement",))
        if mode == "p2":
            return _reader_phase_intervention(model, roles=("p2_field",))
        raise ValueError(f"unknown Reader phase mode={mode!r}")
    if family == "dense":
        if mode == "p2_environment":
            return backend_read_zero(model, roles=("p2_field",), component="environment")
        if mode == "p2_module":
            return backend_read_zero(model, roles=("p2_field",), component="module")
        if mode == "p2_coarse":
            return common_read_zero(model, role="p2_field", component="coarse")
        if mode == "p2_local":
            return common_read_zero(model, role="p2_field", component="local")
        if mode == "all_environment":
            return backend_read_zero(model, roles=("p0_port", "p1_refinement", "p2_field"), component="environment")
        raise ValueError(f"unknown Dense phase mode={mode!r}")
    if family == "regional":
        if mode == "p0":
            return backend_read_zero(model, roles=("p0_port",), component="regional")
        if mode == "p1_only":
            return backend_read_zero(model, roles=("p1_refinement",), component="regional")
        if mode == "p2":
            return backend_read_zero(model, roles=("p2_field",), component="regional")
        if mode == "p2_module":
            return backend_read_zero(model, roles=("p2_field",), component="module")
        if mode == "p2_coarse":
            return common_read_zero(model, role="p2_field", component="coarse")
        raise ValueError(f"unknown regional phase mode={mode!r}")
    raise ValueError(f"unknown intervention family={family!r}")


def _family_for_architecture(architecture: str) -> str:
    if architecture == "dense_pairwise_field":
        return "dense"
    if architecture == "sparse_interface_honf":
        return "reader"
    if architecture == "regional_response_honf":
        return "regional"
    raise ValueError(f"regional study diagnostics do not support architecture={architecture!r}")


def _active_mask(sample: Mapping[str, Any]) -> torch.Tensor:
    present = torch.as_tensor(sample["structure"]["module_present"], dtype=torch.bool)
    if present.ndim == 1:
        present = present.unsqueeze(0)
    return present


def _run_variant(
    model: Any,
    sample: Mapping[str, Any],
    query_np: np.ndarray,
    device: torch.device,
    context: contextlib.AbstractContextManager[Any] | None = None,
) -> dict[str, Any]:
    manager = context if context is not None else contextlib.nullcontext()
    with manager, torch.no_grad():
        return _forward_batch(
            model,
            sample,
            query_np,
            device,
            return_routing_maps=True,
            return_organizer_passes=True,
        )


def run_interventions(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 1:
        raise ValueError("interventions requires one explicitly labelled checkpoint")
    device = select_device(args.device)
    spec = specs[0]
    model, checkpoint = _load_model_spec(spec, device)
    family = _family_for_architecture(str(model.config.core_honf.forward_architecture))
    dataset, dataset_path = _load_dataset(checkpoint, args)
    case_ids = [str(value) for value in (args.case_id or DEFAULT_REGIONAL_CASE_IDS)]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    raw_samples = {case_id: _load_raw_sample(dataset_path, args.split, case_id) for case_id in case_ids}
    state_before = _state_keys(model)
    if family == "dense":
        modes = ("p2_environment", "p2_module", "p2_coarse", "p2_local", "all_environment")
    elif family == "regional":
        modes = ("p0", "p1_only", "p2", "p2_module", "p2_coarse")
    else:
        modes = ("p0", "p1_only", "p2")
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        sample = __import__("channelthermal.evaluation.prepared", fromlist=["select_sample"]).select_sample(dataset, case_id, 0)
        query_np = _query_points(sample, int(args.query_count))
        base = _run_variant(model, sample, query_np, device)
        base_errors = _canonical_ground_truth_errors(
            base, sample, raw_samples[case_id], dataset, model, checkpoint
        )
        row: dict[str, Any] = {
            "case_id": case_id,
            "selection_reason": REGIONAL_CASE_SELECTION_REASONS.get(
                case_id, "caller-specified case; selection predates this run"
            ),
            "family": family,
            "architecture": str(model.config.core_honf.forward_architecture),
            "query_count": len(query_np),
            "ground_truth_errors": {"normal": base_errors},
            "interventions": {},
        }
        for mode in modes:
            variant = _run_variant(model, sample, query_np, device, _phase_variant_context(model, family, mode))
            errors = _canonical_ground_truth_errors(
                variant, sample, raw_samples[case_id], dataset, model, checkpoint
            )
            row["ground_truth_errors"][mode] = errors
            prediction_differences = {
                "pred_field": _relative_difference(
                    variant.get("pred_field"), base.get("pred_field")
                ),
                "pred_interface": _relative_difference(
                    variant.get("pred_interface"), base.get("pred_interface")
                ),
                "pred_internal_temperature": _relative_difference(
                    variant.get("pred_internal_temperature"), base.get("pred_internal_temperature")
                ),
                "pred_port_condition": _relative_difference(
                    variant.get("pred_port_condition"), base.get("pred_port_condition")
                ),
            }
            row["interventions"][mode] = {
                "prediction_difference": prediction_differences["pred_field"],
                "prediction_differences": prediction_differences,
                "error_deltas": _error_deltas(base_errors, errors),
            }
        rows.append(row)
    return {
        "schema_version": 1,
        "task": "interventions",
        "stage": "endpoint500" if family == "regional" else "mature_checkpoint",
        "checkpoint": _checkpoint_record(spec, checkpoint, model),
        "dataset": str(dataset_path),
        "case_ids": case_ids,
        "case_selection_reasons": {
            case_id: REGIONAL_CASE_SELECTION_REASONS.get(
                case_id, "caller-specified case; selection predates this run"
            )
            for case_id in case_ids
        },
        "family": family,
        "results": rows,
        "state_dict_structure_unchanged": state_before == _state_keys(model),
        "interpretation": {
            "prediction_difference": "frozen-model reliance only",
            "error_delta": "intervened-minus-normal canonical ground-truth error; positive worsens error",
            "p1_only": "removes only the selected backend component at P1, preserving normal P0",
        },
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_forward(
    model: Any,
    sample: Mapping[str, Any],
    query_np: np.ndarray,
    device: torch.device,
    context: contextlib.AbstractContextManager[Any] | None = None,
) -> tuple[dict[str, Any], float, dict[str, float | None]]:
    manager = context if context is not None else contextlib.nullcontext()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    with manager, torch.no_grad():
        output = _forward_batch(
            model,
            sample,
            query_np,
            device,
            return_routing_maps=False,
            return_organizer_passes=True,
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    memory = {
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
    }
    if device.type == "cuda":
        memory = {
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024.0**2)),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024.0**2)),
        }
    return output, elapsed, memory


@contextlib.contextmanager
def capture_environment_contexts(
    model: Any,
    *,
    active_module_mask: torch.Tensor | None = None,
    probe_count: int = 256,
) -> Iterator[dict[str, Any]]:
    """Capture bounded environmental read probes and branch timings.

    The production reader returns the sum of the direct module and environmental
    branches.  Dense already exposes the direct module reader, so this utility
    captures ``main - module`` at the actual role-tagged backend boundary.  It
    records only a fixed prefix from each receiver chunk and keeps the full
    context tensors out of the JSON payload; anchor rows receive those small
    probes later.  ``environment_read_seconds`` measures the environmental
    reader itself, while ``backend_read_seconds`` remains the complete backend
    read and is kept separate from the surrounding physical forward time.
    """

    backend = model.core.backend
    original_read = backend.read
    original_read_module = getattr(backend, "read_module", None)
    original_read_environment = getattr(backend, "read_environment", None)
    attention = getattr(backend, "env_attention", None)
    original_read_projected = getattr(attention, "read_projected", None)
    state: dict[str, Any] = {
        "records": {},
        "timing": {
            "backend_read_seconds": {},
            "module_read_seconds": {},
            "environment_read_seconds": {},
            "projected_environment_read_seconds": {},
        },
        "offsets": {},
        "module_outputs": [],
        "probe_count": int(max(1, probe_count)),
    }

    def _role() -> str:
        return str(getattr(model.core, "_interface_read_role", "unlabelled"))

    def _device_from_args(args: tuple[Any, ...]) -> torch.device | None:
        for value in args:
            if torch.is_tensor(value):
                return value.device
        return None

    def _timed_call(name: str, fn: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        device = _device_from_args(args)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        role = _role()
        state["timing"][name].setdefault(role, 0.0)
        state["timing"][name][role] += float(elapsed)
        return result

    def read_module(*args: Any, **kwargs: Any) -> Any:
        if original_read_module is None:
            raise AttributeError("backend does not expose read_module")
        result = _timed_call("module_read_seconds", original_read_module, args, kwargs)
        state["module_outputs"].append(result)
        return result

    def read_environment(*args: Any, **kwargs: Any) -> Any:
        if original_read_environment is None:
            raise AttributeError("backend does not expose read_environment")
        return _timed_call("environment_read_seconds", original_read_environment, args, kwargs)

    def read_projected(*args: Any, **kwargs: Any) -> Any:
        if original_read_projected is None:
            raise AttributeError("environment attention does not expose read_projected")
        return _timed_call(
            "projected_environment_read_seconds", original_read_projected, args, kwargs
        )

    def read(*args: Any, **kwargs: Any) -> Any:
        role = _role()
        module_start = len(state["module_outputs"])
        device = _device_from_args(args)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = original_read(*args, **kwargs)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        state["timing"]["backend_read_seconds"].setdefault(role, 0.0)
        state["timing"]["backend_read_seconds"][role] += float(elapsed)
        module_values = state["module_outputs"][module_start:]
        del state["module_outputs"][module_start:]
        if not isinstance(result, tuple) or len(result) != 2 or not module_values:
            return result
        main = result[0]
        module_context = module_values[-1]
        if not torch.is_tensor(main) or not torch.is_tensor(module_context):
            return result
        if tuple(main.shape) != tuple(module_context.shape):
            return result
        environment_context = (main - module_context).detach()
        width = int(environment_context.shape[1])
        offset = int(state["offsets"].get(role, 0))
        state["offsets"][role] = offset + width
        keep = min(width, state["probe_count"])
        state["records"].setdefault(role, []).append(
            {
                "values": environment_context[:, :keep].cpu(),
                "offset": offset,
                "width": width,
            }
        )
        return result

    if original_read_module is not None:
        backend.read_module = read_module
    if original_read_environment is not None:
        backend.read_environment = read_environment
    if attention is not None and original_read_projected is not None:
        attention.read_projected = read_projected
    backend.read = read
    try:
        yield state
    finally:
        backend.read = original_read
        if original_read_module is not None:
            backend.read_module = original_read_module
        if original_read_environment is not None:
            backend.read_environment = original_read_environment
        if attention is not None and original_read_projected is not None:
            attention.read_projected = original_read_projected


def _context_delta_summary(
    normal: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    active_module_mask: torch.Tensor | None,
    include_probes: bool,
) -> dict[str, Any]:
    """Compare captured role contexts while masking padded module ports."""

    normal_records = normal.get("records", {})
    variant_records = variant.get("records", {})
    summary: dict[str, Any] = {}
    for role in ("p0_port", "p1_refinement", "p2_field", "p2_port_global_consistency"):
        left = list(normal_records.get(role, ()))
        right = list(variant_records.get(role, ()))
        if not left and not right:
            continue
        role_payload: dict[str, Any] = {
            "normal_read_count": len(left),
            "variant_read_count": len(right),
        }
        if len(left) != len(right):
            role_payload["alignment"] = "mismatched read chunk count; paired prefix only"
        deltas: list[torch.Tensor] = []
        base_values: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for normal_record, variant_record in zip(left, right):
            normal_values = normal_record["values"]
            variant_values = variant_record["values"]
            width = min(int(normal_values.shape[1]), int(variant_values.shape[1]))
            if width <= 0:
                continue
            delta = variant_values[:, :width] - normal_values[:, :width]
            mask = torch.ones(
                (delta.shape[0], width), dtype=torch.bool, device=delta.device
            )
            if (
                active_module_mask is not None
                and role in {"p0_port", "p1_refinement"}
                and active_module_mask.ndim == 2
            ):
                modules = int(active_module_mask.shape[1])
                total_width = sum(int(record["width"]) for record in left)
                if modules > 0 and total_width % modules == 0:
                    ntheta = total_width // modules
                    global_indices = int(normal_record["offset"]) + torch.arange(width)
                    module_indices = torch.div(global_indices, ntheta, rounding_mode="floor")
                    module_indices = module_indices.clamp(0, modules - 1)
                    mask = active_module_mask[:, module_indices].to(device=delta.device)
            deltas.append(delta[mask])
            base_values.append(normal_values[:, :width][mask])
            masks.append(mask)
        if not deltas:
            role_payload["status"] = "no aligned context probes"
            summary[role] = role_payload
            continue
        flat_delta = torch.cat(deltas, dim=0).float()
        flat_base = torch.cat(base_values, dim=0).float()
        delta_norm = torch.linalg.vector_norm(flat_delta, dim=-1)
        base_norm = torch.linalg.vector_norm(flat_base, dim=-1)
        role_payload.update(
            {
                "active_probe_count": int(flat_delta.shape[0]),
                "delta_norm_mean": float(delta_norm.mean()),
                "delta_norm_max": float(delta_norm.max()),
                "delta_abs_mean": float(flat_delta.abs().mean()),
                "base_norm_mean": float(base_norm.mean()),
                "relative_delta_norm_mean": float(
                    (delta_norm / base_norm.clamp_min(torch.finfo(base_norm.dtype).tiny)).mean()
                ),
            }
        )
        if include_probes:
            role_payload["normal_probe"] = _json_value(left[0]["values"][0, :16])
            role_payload["variant_probe"] = _json_value(right[0]["values"][0, :16])
        summary[role] = role_payload
    return summary


def run_frozen(args: argparse.Namespace) -> dict[str, Any]:
    """Measure Dense projected K/V pooling in P2-only and full physical loops."""

    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 1:
        raise ValueError("frozen requires exactly one Dense checkpoint")
    device = select_device(args.device)
    spec = specs[0]
    model, checkpoint = _load_model_spec(spec, device)
    if str(model.config.core_honf.forward_architecture) != "dense_pairwise_field":
        raise ValueError("frozen is a Dense-only approximation study")
    dataset, dataset_path = _load_dataset(checkpoint, args)
    case_ids = [str(value) for value in (args.case_id or DEFAULT_REGIONAL_CASE_IDS)]
    if len(case_ids) != 8:
        raise ValueError("frozen requires exactly eight fixed cases (four anchors plus four manifest cases)")
    state_before = _state_keys(model)
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        sample = select_sample(dataset, case_id, 0)
        raw_sample = _load_raw_sample(dataset_path, args.split, case_id)
        query_np = _query_points(sample, int(args.query_count))
        active_module_mask = _active_mask(sample).to(device="cpu")
        with capture_environment_contexts(
            model, active_module_mask=active_module_mask, probe_count=int(args.context_probe_count)
        ) as base_capture:
            base, base_time, base_memory = _timed_forward(model, sample, query_np, device)
        base_errors = _canonical_ground_truth_errors(
            base, sample, raw_sample, dataset, model, checkpoint
        )
        variants: dict[str, tuple[dict[str, Any], float, dict[str, float | None], dict[str, Any]]] = {}
        for label, roles in (
            ("p2_only", ("p2_field",)),
            ("full_loop", ("p0_port", "p1_refinement", "p2_field")),
        ):
            with frozen_dense_read(model, roles=roles) as frozen_state, capture_environment_contexts(
                model, active_module_mask=active_module_mask, probe_count=int(args.context_probe_count)
            ) as variant_capture:
                variant_result = _timed_forward(model, sample, query_np, device)
            variants[label] = (*variant_result, {"capture": variant_capture, "frozen_state": frozen_state})
        row: dict[str, Any] = {
            "case_id": case_id,
            "selection_reason": REGIONAL_CASE_SELECTION_REASONS.get(
                case_id, "caller-specified case; selection predates this run"
            ),
            "query_count": len(query_np),
            "normal": {
                "ground_truth_errors": base_errors,
                "elapsed_seconds": base_time,
                "backend_read_timing": base_capture["timing"],
                "memory": base_memory,
            },
            "variants": {},
        }
        for label, (variant, elapsed, memory, timing_state) in variants.items():
            errors = _canonical_ground_truth_errors(
                variant, sample, raw_sample, dataset, model, checkpoint
            )
            interaction_aux = variant.get("interaction_aux", {})
            if not isinstance(interaction_aux, Mapping):
                interaction_aux = {}
            row["variants"][label] = {
                "ground_truth_errors": errors,
                "error_deltas": _error_deltas(base_errors, errors),
                "prediction_difference": _relative_difference(
                    variant.get("pred_field"), base.get("pred_field")
                ),
                "elapsed_seconds": elapsed,
                "backend_read_timing": timing_state["capture"]["timing"],
                "frozen_preparation_count": timing_state["frozen_state"]["prepared_source_count"],
                "environment_context_delta": _context_delta_summary(
                    base_capture,
                    timing_state["capture"],
                    active_module_mask=active_module_mask,
                    include_probes=case_id in ANCHOR_CASE_IDS,
                ),
                "memory": memory,
                "environment_context_norm_mean": _tensor_mean(
                    interaction_aux.get("dense_environment_context_norm")
                ),
                "source_count": _tensor_mean(interaction_aux.get("dense_environment_source_count")),
                "region_count": _tensor_mean(interaction_aux.get("dense_region_count")),
            }
        rows.append(row)
    return {
        "schema_version": 1,
        "task": "frozen",
        "stage": "frozen_projected_key_value_coarsening",
        "checkpoint": _checkpoint_record(spec, checkpoint, model),
        "dataset": str(dataset_path),
        "case_ids": case_ids,
        "case_selection_reasons": {
            case_id: REGIONAL_CASE_SELECTION_REASONS.get(
                case_id, "caller-specified case; selection predates this run"
            )
            for case_id in case_ids
        },
        "block_shape": list(DEFAULT_BLOCK_SHAPE),
        "source_grid": "24x8 x-fast cell-centred grid; region IDs derived from coordinates",
        "source_projection": "per-head keys/values after source LayerNorm and projection",
        "quadrature": "weighted mass preserved; one log mass factor in regional softmax",
        "variants": ["p2_only", "full_loop"],
        "context_probe_count": int(args.context_probe_count),
        "results": rows,
        "state_dict_structure_unchanged": state_before == _state_keys(model),
        "interpretation": [
            "p2_only replaces only the final environmental field read; physical ports and refinement remain normal",
            "full_loop replaces environmental reads at P0, P1 and P2 and recomputes downstream physical responses",
            "frozen projected-state pooling is distinct from native grouped nonlinear preparation except in the singleton limit",
        ],
    }


def _tensor_mean(value: Any) -> float | None:
    if value is None:
        return None
    tensor = value.detach().float() if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
    finite = tensor[torch.isfinite(tensor)]
    return None if not finite.numel() else float(finite.mean().cpu())


def _regular_grid_shape(env_count: int, lx: float, ly: float) -> tuple[int, int]:
    """Return the fixed even layouts used by the regional timing protocol."""
    del lx, ly
    known = {192: (24, 8), 768: (48, 16), 3072: (96, 32)}
    if int(env_count) not in known:
        raise ValueError(f"The fixed regional study layouts support E=192, 768, or 3072, got {env_count}.")
    return known[int(env_count)]


def _regular_environment(
    env_count: int,
    lx: float,
    ly: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ChannelThermalEnvironment:
    nx, ny = _regular_grid_shape(int(env_count), float(lx), float(ly))
    return ChannelThermalEnvironmentBuilder()(
        batch_size=batch_size,
        num_env_tokens_x=nx,
        num_env_tokens_y=ny,
        domain_length_x=lx,
        domain_length_y=ly,
        device=device,
        dtype=dtype,
        response_region_block_shape=DEFAULT_BLOCK_SHAPE,
    )


@contextlib.contextmanager
def regular_synthetic_environment(model: Any, env_count: int, lx: float, ly: float) -> Iterator[None]:
    """Use an explicit even rectangular grid for synthetic regional timings."""

    original = model.environment_builder

    class RegularEnvironmentBuilder:
        def __call__(self, *, batch_size: int, device: torch.device, dtype: torch.dtype, **kwargs: Any) -> ChannelThermalEnvironment:
            del kwargs
            return _regular_environment(env_count, lx, ly, batch_size, device, dtype)

        def query_features(self, *args: Any, **kwargs: Any) -> Any:
            return original.query_features(*args, **kwargs)

    model.environment_builder = RegularEnvironmentBuilder()
    try:
        yield
    finally:
        model.environment_builder = original


@contextlib.contextmanager
def dense_projection_cache(model: Any) -> Iterator[dict[str, int]]:
    """Reuse Dense's fine projected environmental K/V within one preparation.

    This evaluation-only route preserves all 192 fine environmental sources.
    It isolates projection reuse from the separate 192-to-48 frozen grouping
    experiment and leaves Dense's ordinary reader unchanged outside the
    context.  Private fields on each fresh backend preparation state make the
    cache lifetime preparation-local while allowing prepared decode repetitions
    to reuse one projection.
    """

    if str(model.config.core_honf.forward_architecture) != "dense_pairwise_field":
        raise ValueError("fine projection caching is defined for Dense only")
    backend = model.core.backend
    original = backend.read
    state = {"prepared_projection_count": 0}

    def read(
        prepared_state: Mapping[str, torch.Tensor],
        encoded: Any,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(prepared_state, MutableMapping):
            raise TypeError("Dense projection caching requires a mutable prepared-state mapping")
        projected_key = prepared_state.get(_DENSE_PROJECTED_KEY_CACHE)
        projected_value = prepared_state.get(_DENSE_PROJECTED_VALUE_CACHE)
        if projected_key is None or projected_value is None:
            projected_key, projected_value = backend.project_environment_sources(
                prepared_state["env_tokens"]
            )
            prepared_state[_DENSE_PROJECTED_KEY_CACHE] = projected_key
            prepared_state[_DENSE_PROJECTED_VALUE_CACHE] = projected_value
            state["prepared_projection_count"] += 1
        module_context = backend.read_module(
            prepared_state, encoded, receivers, receiver_features
        )
        environment_context, environment_attention = backend.read_environment_projected(
            projected_key,
            projected_value,
            encoded,
            receivers,
            receiver_features,
            encoded.env_coords,
            encoded.env_weights,
            return_routing_maps=bool(kwargs.get("return_routing_maps", False)),
        )
        aux: dict[str, torch.Tensor] = {
            "dense_module_context_norm": torch.linalg.vector_norm(module_context, dim=-1),
        }
        if kwargs.get("return_routing_maps", False) and environment_attention is not None:
            aux["dense_environment_attention"] = environment_attention
        return module_context + environment_context, aux

    backend.read = read
    try:
        yield state
    finally:
        backend.read = original


def _timing_variants(model: Any) -> tuple[tuple[str, Any], ...]:
    """Return labels for controlled default/cache execution measurements."""

    architecture = str(model.config.core_honf.forward_architecture)
    if architecture == "dense_pairwise_field":
        return (("dense_default", None), ("dense_projection_cached", dense_projection_cache))
    if architecture == "regional_response_honf":
        # Regional prepares projected K/V once per physical module state and
        # reuses them across receiver chunks within each fresh preparation.
        return (("regional_native_prepared", None),)
    return (("default", None),)


def _measure_phase(
    function: Any,
    device: torch.device,
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Use the established synchronized profiler for CUDA and a small CPU fallback."""

    if device.type == "cuda":
        return _profile_measure(
            function,
            device,
            warmups=int(warmups),
            repetitions=int(repetitions),
        )
    for _ in range(max(0, int(warmups))):
        with torch.inference_mode():
            value = function()
        del value
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        started = time.perf_counter()
        with torch.inference_mode():
            value = function()
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        del value
    return {
        "median_ms": float(np.median(samples_ms)),
        "mean_ms": float(np.mean(samples_ms)),
        "p05_ms": float(np.quantile(samples_ms, 0.05)),
        "p95_ms": float(np.quantile(samples_ms, 0.95)),
        "samples_ms": samples_ms,
        "baseline_allocated_bytes": None,
        "baseline_reserved_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "incremental_peak_allocated_bytes": None,
        "incremental_peak_reserved_bytes": None,
    }


def _output_snapshot_for_variant(
    model: Any,
    batch: Mapping[str, Any],
    query_tensor: torch.Tensor,
    forward_kwargs: Mapping[str, Any],
    *,
    receiver_chunk_size: int,
    variant_factory: Any = None,
) -> dict[str, torch.Tensor]:
    """Capture public outputs for one untimed, controlled comparison pass."""

    with contextlib.ExitStack() as stack:
        stack.enter_context(_runtime_receiver_chunk_size(model, receiver_chunk_size))
        if variant_factory is not None:
            stack.enter_context(variant_factory(model))
        with torch.inference_mode():
            output = model(
                batch["structure"],
                query_tensor,
                return_routing_maps=False,
                **forward_kwargs,
            )
    return _public_output_snapshot(output)


def _timing_output_agreement(
    reference_same_chunk: Mapping[str, torch.Tensor],
    reference_chunk128: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    receiver_chunk_size: int,
) -> dict[str, Any]:
    """Report the two profiler output checks without making them approval gates."""

    same_rtol, same_atol = _agreement_tolerances("summary", receiver_chunk_size)
    cross_rtol, cross_atol = _agreement_tolerances("summary", receiver_chunk_size)
    same = _output_agreement(
        dict(reference_same_chunk),
        dict(candidate),
        rtol=same_rtol,
        atol=same_atol,
    )
    cross = _output_agreement(
        dict(reference_chunk128),
        dict(candidate),
        rtol=cross_rtol,
        atol=cross_atol,
    )
    return {
        "same_chunk_default_vs_candidate": {
            "reference_receiver_chunk_size": int(receiver_chunk_size),
            "candidate_receiver_chunk_size": int(receiver_chunk_size),
            "rtol": float(same_rtol),
            "atol": float(same_atol),
            "outputs": same,
        },
        "chunk128_default_vs_candidate": {
            "reference_receiver_chunk_size": 128,
            "candidate_receiver_chunk_size": int(receiver_chunk_size),
            "rtol": float(cross_rtol),
            "atol": float(cross_atol),
            "outputs": cross,
        },
    }


@contextlib.contextmanager
def count_backend_operations(model: Any) -> Iterator[dict[str, int]]:
    """Count actual rows sent through the expensive backend neural blocks."""

    backend = getattr(getattr(model, "core", None), "backend", None)
    counts: dict[str, int] = {}
    hooks: list[Any] = []
    names = (
        "mm_message",
        "me_message",
        "em_message",
        "env_update",
        "env_geometry_bias",
        "query_module_message",
    )
    if backend is not None:
        for name in names:
            module = getattr(backend, name, None)
            if module is None or not hasattr(module, "register_forward_hook"):
                continue
            counts.setdefault(name, 0)

            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, *, key: str = name) -> None:
                tensor = output[0] if isinstance(output, tuple) and output else output
                if torch.is_tensor(tensor):
                    counts[key] += int(np.prod(tuple(int(value) for value in tensor.shape[:-1]))) if tensor.ndim else 1

            hooks.append(module.register_forward_hook(hook))
    try:
        yield counts
    finally:
        for hook in hooks:
            hook.remove()


def _synthetic_structure(model: Any, module_count: int, lx: float, ly: float, device: torch.device) -> dict[str, torch.Tensor]:
    centers = _synthetic_centers(module_count, lx, ly, _module_radius(model))
    return {
        "re": torch.full((1, 1), 50.0, device=device),
        "u_in": torch.ones((1, 1), device=device),
        "module_centers": torch.from_numpy(centers).unsqueeze(0).to(device),
        "heat_powers": torch.ones((1, module_count), device=device),
        "module_present": torch.ones((1, module_count), device=device),
        "material_params": torch.zeros((1, int(model.config.channelthermal.material_param_dim)), device=device),
        "domain_length_x": torch.full((1, 1), float(lx), device=device),
        "domain_length_y": torch.full((1, 1), float(ly), device=device),
    }



def run_timing_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Run the established phase profiler with the regional comparison labels."""

    specs = parse_checkpoint_specs(args.checkpoint)
    if not specs:
        raise ValueError("timing requires at least one checkpoint")
    device = select_device(args.device)
    shapes = [parse_shape(value) for value in args.shape] if args.shape else list(DEFAULT_SHAPES)
    receiver_chunk = int(args.receiver_chunk_size)
    if receiver_chunk <= 0:
        raise ValueError("receiver chunk must be positive")
    model_records: list[dict[str, Any]] = []
    for spec in specs:
        model, checkpoint = _load_model_spec(spec, device)
        dataset, dataset_path = _load_dataset(checkpoint, args)
        state_before = _state_keys(model)
        real_rows: list[dict[str, Any]] = []
        architecture = str(model.config.core_honf.forward_architecture)
        for case_id in [str(value) for value in (args.case_id or ANCHOR_CASE_IDS[:2])]:
            sample = select_sample(dataset, case_id, 0)
            query_np = _query_points(sample, int(args.query_count))
            batch = make_batch(dict(sample), query_np, device)
            forward_kwargs = {
                "interface_condition": batch.get("interface_condition"),
                "local_module_params": batch.get("local_module_params"),
                "teacher_port_tokens": batch.get("teacher_port_tokens"),
                "local_query_points": batch.get("module_internal_query_points"),
                "local_port_condition_mode": "predicted",
                "mixed_teacher_ratio": 0.0,
            }
            query_tensor = batch["query_xy"]

            def full_forward() -> Any:
                with _runtime_receiver_chunk_size(model, receiver_chunk):
                    return model(
                        batch["structure"],
                        query_tensor,
                        return_routing_maps=False,
                        **forward_kwargs,
                    )

            def prepare_one() -> Any:
                with _runtime_receiver_chunk_size(model, receiver_chunk):
                    return model(
                        batch["structure"],
                        query_tensor[:, :1],
                        return_prepared_state=True,
                        return_routing_maps=False,
                        **forward_kwargs,
                    )

            def encoding_layout() -> Any:
                return _new_family_encoding_and_layout(model, batch)

            phase_functions = (
                ("encoding_plus_layout_construction", encoding_layout),
                ("physical_preparation_plus_one_query", prepare_one),
                ("full_forward", full_forward),
            )
            # Keep output checks outside timed calls.  The first reference is
            # the normal path at the measured chunk, while the second follows
            # the established profiler's chunk-128 reference convention.
            default_same_chunk = _output_snapshot_for_variant(
                model,
                batch,
                query_tensor,
                forward_kwargs,
                receiver_chunk_size=receiver_chunk,
            )
            default_chunk128 = _output_snapshot_for_variant(
                model,
                batch,
                query_tensor,
                forward_kwargs,
                receiver_chunk_size=128,
            )
            variant_rows: dict[str, Any] = {}
            for variant_label, variant_factory in _timing_variants(model):
                phases: dict[str, Any] = {}
                for phase_label, phase_function in phase_functions:
                    def measured_phase(
                        phase_function: Any = phase_function,
                        variant_factory: Any = variant_factory,
                    ) -> Any:
                        if variant_factory is None:
                            return phase_function()
                        with variant_factory(model):
                            return phase_function()

                    phases[phase_label] = _measure_phase(
                        measured_phase,
                        device,
                        warmups=int(args.warmup),
                        repetitions=int(args.repetitions),
                    )

                operation_state: dict[str, Any] = {}
                with contextlib.ExitStack() as stack:
                    if variant_factory is not None:
                        operation_state = stack.enter_context(variant_factory(model))
                    counter = stack.enter_context(count_backend_operations(model))
                    with torch.inference_mode():
                        full_forward()
                candidate_snapshot = _output_snapshot_for_variant(
                    model,
                    batch,
                    query_tensor,
                    forward_kwargs,
                    receiver_chunk_size=receiver_chunk,
                    variant_factory=variant_factory,
                )
                decode_cache_state: dict[str, Any] = {}
                prepared: Any = None
                decode_function: Any = None
                prepared_projection_count: int | None = None
                try:
                    # Keep the state lifetime inside the variant context.  In
                    # particular, Dense's cached projection is created while
                    # preparing this state and survives all decode repetitions.
                    with contextlib.ExitStack() as decode_stack:
                        if variant_factory is not None:
                            decode_cache_state = decode_stack.enter_context(variant_factory(model))
                        with torch.inference_mode():
                            prepared = prepare_one()["prepared_state"]

                        def prepared_decode() -> Any:
                            return model.decode_prepared(
                                prepared,
                                query_tensor,
                                return_routing_maps=False,
                                receiver_chunk_size=receiver_chunk,
                            )

                        decode_function = prepared_decode
                        phases["prepared_decode"] = _measure_phase(
                            decode_function,
                            device,
                            warmups=int(args.warmup),
                            repetitions=int(args.repetitions),
                        )
                        prepared_projection_count = (
                            decode_cache_state.get("prepared_projection_count")
                            if variant_factory is not None
                            else operation_state.get("prepared_projection_count")
                        )
                finally:
                    # Drop the closure before releasing the prepared mapping;
                    # the next variant must not retain this state's tensors.
                    decode_function = None
                    del prepared
                variant_rows[variant_label] = {
                    "phases": phases,
                    "operation_rows": counter,
                    "prepared_projection_count": prepared_projection_count,
                    "output_agreement": _timing_output_agreement(
                        default_same_chunk,
                        default_chunk128,
                        candidate_snapshot,
                        receiver_chunk_size=receiver_chunk,
                    ),
                }
            normal_variant = (
                "dense_default"
                if architecture == "dense_pairwise_field"
                else next(iter(variant_rows), None)
            )
            real_rows.append(
                {
                    "case_id": case_id,
                    "query_count": int(query_tensor.shape[1]),
                    "receiver_chunk_size": receiver_chunk,
                    "normal_variant": normal_variant,
                    "normal": None if normal_variant is None else variant_rows.get(normal_variant),
                    "variants": variant_rows,
                }
            )

        # Real-case tensors must not become baseline allocations for the
        # following synthetic measurements or the next model.
        del batch, query_tensor, forward_kwargs
        synthetic_rows: list[dict[str, Any]] = []
        for module_count, env_count, query_count in shapes:
            lx, ly = _scaled_domain(module_count)
            structure = _synthetic_structure(model, module_count, lx, ly, device)
            queries = _synthetic_query_tensor(query_count, lx, ly, device)
            grid_nx, grid_ny = _regular_grid_shape(env_count, lx, ly)
            grid_environment = _regular_environment(
                env_count,
                lx,
                ly,
                1,
                device,
                queries.dtype,
            )
            realized_region_count = int(
                torch.unique(grid_environment.env_region_ids[0][grid_environment.env_region_ids[0] >= 0]).numel()
            )
            shape_row: dict[str, Any] = {
                "shape": {"M": module_count, "E": env_count, "Q": query_count},
                "grid_shape": {"nx": grid_nx, "ny": grid_ny},
                "realized_source_count": int(grid_environment.env_coords.shape[1]),
                "realized_region_count": realized_region_count,
                "receiver_chunk_size": receiver_chunk,
                "variants": {},
            }
            for variant_label, variant_factory in _timing_variants(model):
                variant_row: dict[str, Any] = {"status": "ok"}

                def synthetic_forward() -> Any:
                    with temporary_domain(model, lx, ly), regular_synthetic_environment(
                        model, env_count, lx, ly
                    ):
                        if variant_factory is None:
                            return _prepared_synthetic_forward(
                                model,
                                structure,
                                queries,
                                device,
                                query_batch_size=int(args.query_batch_size),
                                receiver_chunk_size=receiver_chunk,
                            )
                        with variant_factory(model):
                            return _prepared_synthetic_forward(
                                model,
                                structure,
                                queries,
                                device,
                                query_batch_size=int(args.query_batch_size),
                                receiver_chunk_size=receiver_chunk,
                            )

                try:
                    measured = _measure_phase(
                        synthetic_forward,
                        device,
                        warmups=int(args.synthetic_warmup),
                        repetitions=int(args.synthetic_repetitions),
                    )
                    with torch.inference_mode():
                        _prediction, _prepared, extras = synthetic_forward()
                    with temporary_domain(model, lx, ly), regular_synthetic_environment(
                        model, env_count, lx, ly
                    ):
                        operation_state: dict[str, Any] = {}
                        with contextlib.ExitStack() as stack:
                            if variant_factory is not None:
                                operation_state = stack.enter_context(variant_factory(model))
                            counter = stack.enter_context(count_backend_operations(model))
                            with torch.inference_mode():
                                _prepared_synthetic_forward(
                                    model,
                                    structure,
                                    queries,
                                    device,
                                    query_batch_size=int(args.query_batch_size),
                                    receiver_chunk_size=receiver_chunk,
                                )
                    variant_row.update(
                        measured
                        | {
                            "operation_rows": counter,
                            "support_counts": extras.get("support", {}),
                            "prepared_projection_count": operation_state.get("prepared_projection_count"),
                        }
                    )
                    del _prediction, _prepared
                except RuntimeError as exc:
                    variant_row["status"] = "out_of_memory" if "out of memory" in str(exc).lower() else "error"
                    variant_row["error"] = f"{type(exc).__name__}: {exc!s}"
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:  # noqa: BLE001 - preserve per-shape evidence
                    variant_row["status"] = "error"
                    variant_row["error"] = f"{type(exc).__name__}: {exc!s}"
                shape_row["variants"][variant_label] = variant_row
            normal_variant = (
                "dense_default"
                if architecture == "dense_pairwise_field"
                else next(iter(shape_row["variants"]), None)
            )
            shape_row["normal_variant"] = normal_variant
            shape_row["normal"] = (
                None
                if normal_variant is None
                else shape_row["variants"].get(normal_variant)
            )
            synthetic_rows.append(shape_row)
            del structure, queries, grid_environment
        model_records.append(
            {
                "checkpoint": _checkpoint_record(spec, checkpoint, model),
                "architecture": architecture,
                "dataset": str(dataset_path),
                "real_anchors": real_rows,
                "synthetic_shapes": synthetic_rows,
                "state_dict_structure_unchanged": state_before == _state_keys(model),
            }
        )
        dataset.close()
        del model, checkpoint, dataset
    execution_test = bool(getattr(args, "execution_test", False))
    return {
        "schema_version": 1,
        "task": "timing",
        "stage": "helper_execution_test" if execution_test else "endpoint_execution",
        "execution_test": execution_test,
        "warmups": int(args.warmup),
        "repetitions": int(args.repetitions),
        "synthetic_warmups": int(args.synthetic_warmup),
        "synthetic_repetitions": int(args.synthetic_repetitions),
        "query_batch_size": int(args.query_batch_size),
        "receiver_chunk_size": receiver_chunk,
        "models": model_records,
        "limitations": [
            *(["This bounded helper execution test is a smoke check for the timing path and is not an endpoint benchmark."] if execution_test else []),
            "Synthetic shapes measure execution only and use realized even rectangular grids; they are not physical-accuracy evidence.",
            "Encoding/layout, physical preparation, prepared decode, and full forward are measured as separate phases on real anchors.",
            "Operation rows are collected in an untimed matched pass and count actual backend output rows.",
            "Peak reserved memory can retain allocator history; peak allocated is reported separately.",
        ],
    }




def _figure_input_paths(values: Sequence[str] | None) -> list[tuple[str, Path]]:
    """Resolve ``LABEL=PATH`` figure inputs without copying their contents."""

    rows: list[tuple[str, Path]] = []
    for value in values or ():
        text = str(value)
        if "=" in text:
            label, raw_path = text.split("=", 1)
        else:
            raw_path = text
            label = Path(raw_path).stem
        path = Path(raw_path).expanduser().resolve()
        rows.append((label or path.stem, path))
    return rows


def _figure_metric(record: Mapping[str, Any], metric: str) -> float:
    """Read one scalar metric while leaving unavailable values as NaN."""

    if not isinstance(record, Mapping):
        return float("nan")
    value = record.get(metric)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _figure_case_records(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("case_id")): row
        for row in payload.get("results", [])
        if isinstance(row, Mapping) and row.get("case_id") is not None
    }


def run_figures(args: argparse.Namespace) -> dict[str, Any]:
    """Create compact static figures from already-generated diagnostic JSON."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(args.figure_dir).expanduser().resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "figures",
        "inputs": {},
        "figures": [],
        "limitations": [
            "Figures summarize stored diagnostic JSON; they do not recompute errors or approve a model.",
            "Missing metrics remain NaN and are omitted by matplotlib rather than imputed.",
        ],
    }

    intervention_inputs = _figure_input_paths(args.interventions)
    intervention_payloads: list[tuple[str, Path, dict[str, Any]]] = []
    for label, path in intervention_inputs:
        payload = json.loads(path.read_text())
        intervention_payloads.append((label, path, payload))
    if intervention_payloads:
        manifest["inputs"]["interventions"] = [str(path) for _, path, _ in intervention_payloads]
        case_sets = [set(_figure_case_records(payload)) for _, _, payload in intervention_payloads]
        cases = [case_id for case_id in ANCHOR_CASE_IDS if all(case_id in values for values in case_sets)]
        if cases:
            metrics = (
                ("global_field_fluid_norm_l2", "normal fluid relative L2"),
                ("internal_temperature_physical_relative_l2", "normal internal temperature relative L2"),
            )
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
            x = np.arange(len(cases), dtype=float)
            width = 0.78 / max(1, len(intervention_payloads))
            for panel, (metric, ylabel) in enumerate(metrics):
                axis = axes[0, panel]
                for index, (label, _path, payload) in enumerate(intervention_payloads):
                    rows = _figure_case_records(payload)
                    values = [
                        _figure_metric(
                            rows[case_id].get("ground_truth_errors", {}).get("normal", {}).get("metrics", {}),
                            metric,
                        )
                        for case_id in cases
                    ]
                    axis.bar(
                        x + (index - (len(intervention_payloads) - 1) / 2) * width,
                        values,
                        width,
                        label=label,
                    )
                axis.set_title(ylabel)
                axis.set_ylabel("relative L2")
                axis.grid(axis="y", alpha=0.25)

            delta_metric = "global_field_fluid_norm_l2"
            all_modes: list[tuple[str, str, Mapping[str, Any]]] = []
            for label, _path, payload in intervention_payloads:
                rows = _figure_case_records(payload)
                modes = sorted(
                    {
                        str(mode)
                        for row in rows.values()
                        for mode in row.get("interventions", {})
                    }
                )
                for mode in modes:
                    all_modes.append((label, mode, rows))
            displayed_modes = all_modes[:12]
            mode_width = 0.78 / max(1, len(displayed_modes))
            for axis, metric, title in (
                (axes[1, 0], delta_metric, "field-fluid error delta by intervention"),
                (axes[1, 1], "internal_temperature_physical_relative_l2", "internal temperature error delta by intervention"),
            ):
                for index, (label, mode, rows) in enumerate(displayed_modes):
                    values = [
                        _figure_metric(
                            rows[case_id].get("interventions", {}).get(mode, {}).get("error_deltas", {}),
                            metric,
                        )
                        for case_id in cases
                    ]
                    axis.bar(
                        x + (index - (len(displayed_modes) - 1) / 2) * mode_width,
                        values,
                        mode_width,
                        label=f"{label}:{mode}",
                    )
                axis.axhline(0.0, color="black", linewidth=0.7)
                axis.set_title(title)
                axis.set_ylabel("intervened − normal")
                axis.set_xticks(x, cases)
                axis.grid(axis="y", alpha=0.25)
            axes[0, 0].legend(loc="upper left", fontsize=8)
            axes[1, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
            fig.suptitle("Regional-response Stage I anchor interventions", y=1.01)
            fig.tight_layout()
            output = figure_dir / "intervention_anchor_effects.png"
            fig.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(fig)
            manifest["figures"].append(
                {
                    "path": str(output),
                    "kind": "intervention_anchor_effects",
                    "case_ids": cases,
                    "metrics": [metric for metric, _ in metrics],
                    "displayed_interventions": [f"{label}:{mode}" for label, mode, _ in displayed_modes],
                }
            )

    frozen_path = Path(args.frozen).expanduser().resolve() if args.frozen else None
    if frozen_path is not None:
        payload = json.loads(frozen_path.read_text())
        manifest["inputs"]["frozen"] = str(frozen_path)
        rows = _figure_case_records(payload)
        cases = [case_id for case_id in ANCHOR_CASE_IDS if case_id in rows]
        variants = [
            str(value)
            for value in payload.get("variants", ("p2_only", "full_loop"))
            if str(value) in {"p2_only", "full_loop"}
        ]
        if cases and variants:
            x = np.arange(len(cases), dtype=float)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
            width = 0.78 / len(variants)
            for index, variant in enumerate(variants):
                values = [
                    _figure_metric(
                        rows[case_id].get("variants", {}).get(variant, {}).get("error_deltas", {}),
                        "global_field_fluid_norm_l2",
                    )
                    for case_id in cases
                ]
                axes[0].bar(
                    x + (index - (len(variants) - 1) / 2) * width,
                    values,
                    width,
                    label=variant,
                )
            axes[0].axhline(0.0, color="black", linewidth=0.7)
            axes[0].set_title("Frozen field-fluid error delta")
            axes[0].set_ylabel("intervened − normal relative L2")
            axes[0].set_xticks(x, cases)
            axes[0].grid(axis="y", alpha=0.25)
            axes[0].legend(fontsize=8)

            roles = ("p0_port", "p1_refinement", "p2_field")
            role_width = 0.78 / len(roles)
            for index, role in enumerate(roles):
                values = [
                    _figure_metric(
                        rows[case_id].get("variants", {}).get("full_loop", {}).get("environment_context_delta", {}).get(role, {}),
                        "delta_norm_mean",
                    )
                    for case_id in cases
                ]
                axes[1].bar(
                    x + (index - (len(roles) - 1) / 2) * role_width,
                    values,
                    role_width,
                    label=role,
                )
            axes[1].set_title("Full-loop environmental-context discrepancy")
            axes[1].set_ylabel("receiver-vector norm")
            axes[1].set_xticks(x, cases)
            axes[1].grid(axis="y", alpha=0.25)
            axes[1].legend(fontsize=8)
            fig.suptitle("Frozen 192-to-48 projected-state coarsening", y=1.01)
            fig.tight_layout()
            output = figure_dir / "frozen_coarsening_anchor_effects.png"
            fig.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(fig)
            manifest["figures"].append(
                {
                    "path": str(output),
                    "kind": "frozen_coarsening_anchor_effects",
                    "case_ids": cases,
                    "variants": variants,
                    "context_variant": "full_loop",
                }
            )

    timing_path = Path(args.timing).expanduser().resolve() if args.timing else None
    if timing_path is not None:
        payload = json.loads(timing_path.read_text())
        manifest["inputs"]["timing"] = str(timing_path)
        timing_rows: list[tuple[str, str, str, float]] = []
        for model in payload.get("models", []):
            checkpoint = model.get("checkpoint", {})
            model_label = str(checkpoint.get("label", model.get("architecture", "model")))
            for anchor in model.get("real_anchors", []):
                case_id = str(anchor.get("case_id", "case"))
                for variant, details in anchor.get("variants", {}).items():
                    value = details.get("phases", {}).get("full_forward", {}).get("median_ms")
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        timing_rows.append((model_label, case_id, str(variant), value))
        if timing_rows:
            labels = sorted({f"{model}:{variant}" for model, _, variant, _ in timing_rows})
            cases = sorted({case_id for _, case_id, _, _ in timing_rows})
            x = np.arange(len(cases), dtype=float)
            width = 0.78 / max(1, len(labels))
            fig, axis = plt.subplots(figsize=(12, 4.8))
            for index, label in enumerate(labels):
                values = []
                for case_id in cases:
                    match = [
                        value
                        for model, row_case, variant, value in timing_rows
                        if f"{model}:{variant}" == label and row_case == case_id
                    ]
                    values.append(match[0] if match else float("nan"))
                axis.bar(
                    x + (index - (len(labels) - 1) / 2) * width,
                    values,
                    width,
                    label=label,
                )
            axis.set_title("Measured full-forward time on real anchors")
            axis.set_ylabel("median milliseconds")
            axis.set_xticks(x, cases)
            axis.grid(axis="y", alpha=0.25)
            axis.legend(fontsize=8)
            fig.tight_layout()
            output = figure_dir / "timing_full_forward_anchors.png"
            fig.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(fig)
            manifest["figures"].append(
                {
                    "path": str(output),
                    "kind": "timing_full_forward_anchors",
                    "case_ids": cases,
                    "series": labels,
                }
            )

    if not manifest["figures"]:
        raise ValueError("figures requires at least one usable intervention, frozen, or timing artifact")
    return manifest


def run_regional_probes(args: argparse.Namespace) -> dict[str, Any]:
    """Sample shared regional reads and bounded latent/coordinate derivatives."""
    from channelthermal.interface_field_coupling import _port_coordinates
    from run_stage3_interface_study import (
        _forward_tensor_batch,
        regional_coordinate_direction_probe,
        regional_encoded_module_jvp,
    )

    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 1:
        raise ValueError("regional probes require one candidate checkpoint")
    device = select_device(args.device)
    model, checkpoint = _load_model_spec(specs[0], device)
    if str(model.config.core_honf.forward_architecture) != "regional_response_honf":
        raise ValueError("regional probes require the native regional architecture")
    dataset, dataset_path = _load_dataset(checkpoint, args)
    case_ids = list(args.case_id or ("0273", "0298"))
    rows = []
    for case_id in case_ids:
        sample = select_sample(dataset, str(case_id), 0)
        query = torch.from_numpy(_query_points(sample, int(args.query_count))).unsqueeze(0).to(device)
        # Constants prepared in no_grad remain usable by the subsequent JVP.
        # inference_mode tensors cannot be saved by autograd for this purpose.
        with torch.no_grad():
            output = _forward_tensor_batch(
                model, sample, query, device, return_prepared_state=True
            )
        prepared = output["prepared_state"].prepared
        encoded = prepared.encoded
        active_modules = torch.nonzero(encoded.module_present[0] > 0.5).flatten()
        source_module = int(active_modules[0])
        ntheta = int(output["pred_port_condition"].shape[-2])
        all_ports = _port_coordinates(model, encoded.module_centers, ntheta)
        selected_modules = active_modules[:2]
        angle_indices = torch.arange(0, ntheta, max(1, ntheta // 4), device=device)[:4]
        ports = all_ports[:, selected_modules][:, :, angle_indices].reshape(1, -1, 2)
        receiver_sets = {"physical_ports": ports, "field_probes": query}
        shared_reads = {}
        with torch.no_grad():
            for name, receivers in receiver_sets.items():
                context, attention = model.core.backend.read_regional(
                    prepared.backend_state,
                    encoded,
                    receivers,
                    model.core._receiver_features(prepared, receivers),
                    return_routing_maps=True,
                )
                shared_reads[name] = {
                    "coordinates": receivers[0].cpu(),
                    "mean_head_attention": attention[0].mean(dim=0).cpu(),
                    "context_norm": torch.linalg.vector_norm(context[0], dim=-1).cpu(),
                }
        state = prepared.backend_state
        rows.append({
            "case_id": str(case_id),
            "source_module_index": source_module,
            "source_module_center": encoded.module_centers[0, source_module].detach().cpu(),
            "active_module_count": int(active_modules.numel()),
            "port_module_indices": selected_modules.cpu(),
            "port_angle_indices": angle_indices.cpu(),
            "prepared_phase": "P2; this one state is probed at both receiver types",
            "region_ids": state["regional_ids"][0].cpu(),
            "region_coordinates": state["regional_coords"][0].cpu(),
            "region_mass": state["regional_weights"][0].cpu(),
            "region_state_norm": torch.linalg.vector_norm(
                state["regional_response_states"][0], dim=-1
            ).cpu(),
            "shared_reads": shared_reads,
            "encoded_module_jvp": regional_encoded_module_jvp(
                model, prepared, receiver_sets, source_module
            ),
            "coordinate_direction": regional_coordinate_direction_probe(
                model, sample, query, source_module, np.array([1.0, 0.0]), device
            ),
        })
    dataset.close()
    return {
        "schema_version": 1,
        "task": "regional_probes",
        "device": str(device),
        "checkpoint": _checkpoint_record(specs[0], checkpoint, model),
        "dataset": str(dataset_path),
        "case_ids": case_ids,
        "field_probe_count": int(args.query_count),
        "results": rows,
        "interpretation": {
            "selection": "first active source module; four angles on the first two active modules; fixed field query subset",
            "shared_reads": "same P2-prepared regional states at actual physical port coordinates and field probes; P0/P1 states refresh separately in the physical loop",
            "attention": "normalized learned receiver attention, distinct from deterministic region membership and state-vector norms",
            "support": "all active source modules can contribute to all positive-mass regions; no disconnected derivative zeros are expected",
            "derivatives": "latent dependence and AD/FD self-consistency only; external physical-reference validation remains pending",
        },
    }


def _smoke_case_ids(dataset: Any, *, count: int, large: bool) -> list[str]:
    rows = sorted(
        ((int(module_count), str(case_id)) for module_count, case_id in zip(
            dataset.selected_module_counts, dataset.selected_case_ids
        )),
        key=lambda item: (item[0], item[1]),
    )
    selected = rows[-count:] if large else rows[:count]
    return [case_id for _, case_id in selected]


def _run_optimizer_step(
    model: Any,
    checkpoint: Mapping[str, Any],
    dataset: Any,
    case_ids: Sequence[str],
    device: torch.device,
    *,
    points_per_case: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader, Subset

    index_by_case = {str(case_id): index for index, case_id in enumerate(dataset.selected_case_ids)}
    indices = [index_by_case[str(case_id)] for case_id in case_ids]
    dataset_subset = Subset(dataset, indices)
    collator = ChannelThermalBatchCollator(
        dynamic_module_padding=bool(
            checkpoint.get("train_config", {}).get("dataset", {}).get("dynamic_module_padding", True)
        ),
        max_modules_per_batch=checkpoint.get("train_config", {}).get("dataset", {}).get("max_modules_per_batch"),
    )
    loader = DataLoader(
        dataset_subset,
        batch_size=min(int(batch_size), len(indices)),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    train_cfg = checkpoint.get("train_config", {})
    training_cfg = train_cfg.get("training", {}) if isinstance(train_cfg, Mapping) else {}
    loss_cfg = train_cfg.get("loss", {}) if isinstance(train_cfg, Mapping) else {}
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}
    if not isinstance(loss_cfg, Mapping):
        loss_cfg = {}
    epoch = int(checkpoint.get("epoch", checkpoint.get("current_epoch", 0)))
    mode, ratio = effective_port_condition_settings(epoch, dict(training_cfg))
    internal_weight, interface_weight = effective_local_loss_weights(dict(loss_cfg), mode, ratio)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if value.requires_grad and not isinstance(value, torch.nn.parameter.UninitializedParameter)
    }
    _synchronize(device)
    started = time.perf_counter()
    metrics = run_epoch(
        model,
        loader,
        device,
        dict(loss_cfg),
        optimizer=optimizer,
        scaler=None,
        amp=False,
        max_batches=1,
        local_port_condition_mode=mode,
        mixed_teacher_ratio=float(ratio),
        effective_internal_temperature_weight=float(internal_weight),
        effective_interface_weight=float(interface_weight),
        predicted_consistency_weight=predicted_consistency_weight_for_epoch(epoch, dict(loss_cfg)),
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", training_cfg.get("grad_clip_norm", 0.0))),
        record_gradient_diagnostics=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    max_delta = 0.0
    finite = True
    for name, value in model.named_parameters():
        if isinstance(value, torch.nn.parameter.UninitializedParameter):
            continue
        if name in before:
            max_delta = max(max_delta, float((value.detach() - before[name]).abs().max().cpu()))
        finite = finite and bool(torch.isfinite(value.detach()).all())
    return {
        "case_ids": [str(case_id) for case_id in case_ids],
        "module_counts": [
            int(dataset.selected_module_counts[index_by_case[str(case_id)]]) for case_id in case_ids
        ],
        "batch_size": int(min(int(batch_size), len(indices))),
        "points_per_case": int(points_per_case),
        "elapsed_seconds": float(elapsed),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else None,
        "metrics": {
            key: value for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        "max_parameter_delta": float(max_delta),
        "parameters_finite": bool(finite),
        "optimizer_update_applied": bool(max_delta > 0.0),
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run two real physical optimizer steps without reserving a managed run."""
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.model import ChannelThermalHONFModel
    from channelthermal.training.optimizer import build_forward_optimizer
    from channelthermal.workflows.train_forward import (
        build_model_config,
        resolve_auto_internal_mode,
        set_seed,
    )

    from honf_runtime.case_protocol import WorkflowRequest
    from honf_runtime.config_loader import load_config_bundle
    from honf_runtime.registry import load_case_plugin

    device = select_device(args.device)
    bundle = load_config_bundle(args.profile)
    plugin = load_case_plugin(str(bundle.case["plugin"]))
    # Reuse normal resource/config resolution without calling the allocator.
    cfg = plugin._forward_config(
        bundle,
        WorkflowRequest(workflow="forward", device=str(device), epochs=500),
        Path(args.output).resolve().parent,
    )
    dataset_cfg, train_cfg = cfg["dataset"], cfg["training"]
    set_seed(int(train_cfg["seed"]))
    dataset_path = dataset_cfg["packed_h5_path"]
    train_dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=dataset_cfg["train_split"],
        points_per_case=int(args.points_per_case),
        normalize_inputs=bool(dataset_cfg["normalize_inputs"]),
        normalize_targets=bool(dataset_cfg["normalize_targets"]),
        random_point_sampling=False,
        seed=int(train_cfg["seed"]),
        require_converged=bool(dataset_cfg.get("require_converged", False)),
    )
    model_config = build_model_config(cfg, train_dataset)
    model = ChannelThermalHONFModel(model_config).to(device)
    model.set_global_target_normalization(
        train_dataset.normalizer.stats,
        normalize_targets=bool(dataset_cfg["normalize_targets"]),
    )
    resolve_auto_internal_mode(model_config, model)
    optimizer, _ = build_forward_optimizer(model, train_cfg)
    checkpoint = {"train_config": cfg, "epoch": 1}
    small_ids = [str(value) for value in (args.small_case_id or _smoke_case_ids(train_dataset, count=int(args.case_count), large=False))]
    large_ids = [str(value) for value in (args.large_case_id or _smoke_case_ids(train_dataset, count=int(args.case_count), large=True))]
    model.train()
    small = _run_optimizer_step(
        model, checkpoint, train_dataset, small_ids, device,
        points_per_case=int(args.points_per_case), batch_size=int(args.batch_size), optimizer=optimizer,
    )
    large = _run_optimizer_step(
        model, checkpoint, train_dataset, large_ids, device,
        points_per_case=int(args.points_per_case), batch_size=int(args.batch_size), optimizer=optimizer,
    )
    model.eval()
    train_dataset.close()
    return {
        "schema_version": 1,
        "task": "smoke",
        "stage": "physical_optimizer_steps",
        "profile": str(args.profile),
        "fresh_model": True,
        "frozen_stage_a_attached": bool(model.local_surrogate_attached),
        "dataset": str(dataset_path),
        "steps": {"small_module_batch": small, "large_module_batch": large},
        "managed_run_reserved": False,
        "checkpoint_saved": False,
        "limitations": [
            "These two real batches establish physical forward/backward/update execution only; they are not an accuracy result.",
            "The model and optimizer are initialized from the profile; no candidate weights are loaded or saved.",
            "First-step explicit parameter deltas exclude still-lazy parameters before their initial materialization; canonical gradient/update diagnostics are retained.",
        ],
    }


def _endpoint_figure_table_path(root: Path | None, name: str) -> Path | None:
    """Find a maintained comparison table without copying or rewriting it."""

    if root is None:
        return None
    root = Path(root)
    candidates = [root / name, root / "tables" / name]
    if root.name == "tables":
        candidates.insert(0, root / name)
        candidates.append(root.parent / name)
    if root.name == "debug_npz":
        candidates.append(root.parent / "tables" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _endpoint_figure_read_csv(path: Path | None) -> list[dict[str, str]]:
    """Read a named-column table while tolerating historical extra columns."""

    if path is None or not path.is_file():
        return []
    import csv

    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({str(key): str(value) for key, value in row.items() if key and value is not None})
    return rows


def _endpoint_figure_float(value: Any) -> float:
    """Convert one table/NPZ scalar to a finite float or NaN."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _endpoint_figure_select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selector: str | None = None,
) -> list[dict[str, str]]:
    """Select one exact model label, checkpoint, or run path."""

    if not rows:
        return []
    labels = sorted({str(row.get("model_label", "")) for row in rows if row.get("model_label")})
    if selector:
        selector_text = str(selector)

        def matches(value: Any) -> bool:
            text = str(value or "")
            if text == selector_text:
                return True
            # The established CLI default is a run-family prefix.  Limit
            # prefix matching to these known Dense run identifiers; all
            # other selectors remain exact strings.
            if selector_text == "Run_1804_":
                return text.startswith(selector_text) or any(
                    part.startswith(selector_text) for part in Path(text).parts
                )
            return False

        selected = [
            dict(row)
            for row in rows
            if any(matches(row.get(key)) for key in ("model_label", "checkpoint", "run_dir"))
        ]
    elif len(labels) == 1:
        selected = [dict(row) for row in rows if str(row.get("model_label", "")) == labels[0]]
    else:
        return []
    if not selected:
        return []
    if "model_label" in selected[0]:
        selected_label = str(selected[0].get("model_label", ""))
        selected = [row for row in selected if str(row.get("model_label", "")) == selected_label]
    return selected


def _endpoint_figure_npz_files(root: Path | None) -> list[Path]:
    """Return debug NPZ files from an endpoint/debug directory."""

    if root is None:
        raise ValueError("endpoint figure input root is required")
    root = Path(root)
    directory = root if root.name == "debug_npz" else root / "debug_npz"
    if not directory.is_dir():
        raise FileNotFoundError(f"missing endpoint debug_npz directory: {directory}")
    return sorted(directory.glob("*.npz"))


def _endpoint_figure_npz_for_case(
    root: Path | None,
    case_id: str,
    *,
    filename_prefix: str | None = None,
) -> Path | None:
    """Find one established debug NPZ for a case, optionally by exact prefix."""

    token = str(case_id)
    matches = [
        path
        for path in _endpoint_figure_npz_files(root)
        if path.stem.endswith(f"__{token}")
        and (filename_prefix is None or path.name.startswith(filename_prefix))
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple endpoint debug NPZs match case {token}: {matches}")
    return matches[0] if matches else None


def _endpoint_figure_npz_for_case_from_dirs(
    roots: Sequence[Path],
    case_id: str,
    *,
    filename_prefix: str,
) -> Path | None:
    """Select one exact-prefix case NPZ across explicit debug directories."""

    matches = [
        path
        for root in roots
        for path in [_endpoint_figure_npz_for_case(root, case_id, filename_prefix=filename_prefix)]
        if path is not None
    ]
    if len(matches) > 1:
        raise ValueError(
            f"multiple Dense anchor debug NPZs match case {case_id} and prefix "
            f"{filename_prefix!r}: {matches}"
        )
    return matches[0] if matches else None


def _endpoint_figure_load_npz(path: Path | None) -> dict[str, np.ndarray]:
    """Load an existing compact debug artifact into memory and close the file."""

    if path is None or not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _endpoint_figure_region_arrays(
    data: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fine IDs and valid regional IDs/centroids/masses in one schema."""

    required = (
        "fine_env_coords",
        "fine_env_region_ids",
        "regional_coords",
        "regional_ids",
        "regional_quadrature_mass",
        "regional_valid",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"native regional debug NPZ is missing required keys: {missing}")
    fine_coords = np.asarray(data["fine_env_coords"], dtype=float)
    fine_ids = np.asarray(data["fine_env_region_ids"], dtype=int).reshape(-1)
    regional_coords = np.asarray(data["regional_coords"], dtype=float)
    regional_ids = np.asarray(data["regional_ids"], dtype=int).reshape(-1)
    regional_mass = np.asarray(data["regional_quadrature_mass"], dtype=float).reshape(-1)
    regional_valid = np.asarray(data["regional_valid"], dtype=bool).reshape(-1)
    if fine_coords.shape != (192, 2) or fine_ids.shape != (192,):
        raise ValueError(f"expected actual 24x8 fine grid (192x2), got {fine_coords.shape}/{fine_ids.shape}")
    if regional_coords.shape != (48, 2) or any(array.shape != (48,) for array in (regional_ids, regional_mass, regional_valid)):
        raise ValueError(
            "expected actual 48-region coordinates, IDs, mass, and validity arrays; "
            f"got {regional_coords.shape}/{regional_ids.shape}/{regional_mass.shape}/{regional_valid.shape}"
        )
    if len(np.unique(regional_ids)) != 48:
        raise ValueError("native regional_ids must identify 48 unique shared regions")
    fine_region_ids = set(int(value) for value in fine_ids if int(value) >= 0)
    if not fine_region_ids.issubset(set(int(value) for value in regional_ids)):
        raise ValueError("fine_env_region_ids contains a region absent from regional_ids")
    if not np.isfinite(fine_coords).all() or not np.isfinite(regional_coords).all() or not np.isfinite(regional_mass).all():
        raise ValueError("native regional cover metadata contains nonfinite coordinates or quadrature mass")
    return fine_coords, fine_ids, regional_coords, regional_ids, regional_mass, regional_valid


def _endpoint_figure_attention_matrix(value: Any, region_count: int) -> np.ndarray:
    """Reduce compact/raw attention to receiver-by-region rows."""

    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[-1] != region_count:
        raise ValueError(
            f"expected compact receiver-by-region attention with shape [Q,{region_count}], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("regional attention contains nonfinite values")
    return array


def _endpoint_figure_probe_path(args: argparse.Namespace, native_root: Path | None) -> Path:
    """Resolve the one explicit bounded regional probe artifact."""

    if native_root is None:
        raise ValueError("native endpoint root is required for regional probes")
    value = getattr(args, "probe_artifact", None)
    path = Path(value).expanduser().resolve() if value else (Path(native_root) / "regional_probes.json").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing bounded regional probe artifact: {path}")
    if path.suffix != ".json":
        raise ValueError(f"regional probe artifact must be JSON: {path}")
    return path


def _endpoint_figure_port_attention(
    path: Path,
    case_id: str,
    region_count: int,
) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    """Read the fixed physical-port attention block from ``regional_probes.json``."""

    import json as _json

    if path.suffix != ".json":
        raise ValueError(f"regional probe artifact must be JSON: {path}")
    try:
        payload = _json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to parse regional probe artifact: {path}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), Sequence):
        raise ValueError(f"regional probe artifact lacks results[]: {path}")
    matching = next(
        (
            row
            for row in payload["results"]
            if isinstance(row, Mapping) and str(row.get("case_id")) == str(case_id)
        ),
        None,
    )
    if not isinstance(matching, Mapping):
        return None
    shared_reads = matching.get("shared_reads")
    physical_ports = shared_reads.get("physical_ports") if isinstance(shared_reads, Mapping) else None
    if not isinstance(physical_ports, Mapping) or "mean_head_attention" not in physical_ports:
        return None
    matrix = _endpoint_figure_attention_matrix(physical_ports["mean_head_attention"], region_count)
    if matrix.shape != (8, region_count):
        raise ValueError(
            f"case {case_id}: expected 8 actual physical-port reads over {region_count} regions, "
            f"got {matrix.shape}"
        )
    coordinates = np.asarray(physical_ports.get("coordinates", []), dtype=float)
    if coordinates.shape != (8, 2) or not np.isfinite(coordinates).all():
        raise ValueError(
            f"case {case_id}: expected finite physical-port coordinates with shape (8, 2), "
            f"got {coordinates.shape}"
        )
    labels = [f"port_{index:02d}" for index in range(matrix.shape[0])]
    return matrix, labels, coordinates


def _endpoint_figure_field_probe(
    path: Path,
    case_id: str,
    region_count: int,
) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    """Read explicit fixed field probes from the bounded regional probe JSON."""

    import json as _json

    if path.suffix != ".json":
        return None
    try:
        payload = _json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    matching = next(
        (
            row
            for row in payload.get("results", [])
            if isinstance(row, Mapping) and str(row.get("case_id")) == str(case_id)
        ),
        None,
    )
    if not isinstance(matching, Mapping):
        return None
    shared_reads = matching.get("shared_reads", {})
    field_probes = shared_reads.get("field_probes", {}) if isinstance(shared_reads, Mapping) else {}
    if not isinstance(field_probes, Mapping) or "mean_head_attention" not in field_probes:
        return None
    matrix = _endpoint_figure_attention_matrix(field_probes["mean_head_attention"], region_count)
    if matrix.shape != (32, region_count):
        raise ValueError(
            f"case {case_id}: expected 32 fixed field-probe reads over {region_count} regions, "
            f"got {matrix.shape}"
        )
    coordinates = np.asarray(field_probes.get("coordinates", []), dtype=float)
    if coordinates.shape != (32, 2) or not np.isfinite(coordinates).all():
        raise ValueError(
            f"case {case_id}: expected finite field-probe coordinates with shape (32, 2), "
            f"got {coordinates.shape}"
        )
    labels = [f"field_{index:02d}" for index in range(matrix.shape[0])]
    return matrix, labels, coordinates


def _endpoint_figure_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a compact source table with a stable union of named fields."""

    import csv

    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _endpoint_figure_sample_history(
    history: Mapping[str, Sequence[float]],
    metric: str,
    *,
    maximum: int = 80,
    max_epoch: float = 500.0,
) -> list[tuple[float, float]]:
    """Sample one named history using the maintained named-column reader."""

    epochs = np.asarray(history.get("epoch", ()), dtype=float).reshape(-1)
    values = np.asarray(history.get(metric, ()), dtype=float).reshape(-1)
    count = min(len(epochs), len(values))
    pairs = [
        (float(epoch), float(value))
        for epoch, value in zip(epochs[:count], values[:count])
        if math.isfinite(float(epoch))
        and float(epoch) <= float(max_epoch)
        and math.isfinite(float(value))
        and float(value) > 0.0
    ]
    if len(pairs) <= maximum:
        return pairs
    indices = np.linspace(0, len(pairs) - 1, maximum, dtype=int)
    return [pairs[index] for index in np.unique(indices)]


def _endpoint_figure_history_path(
    args: argparse.Namespace,
    names: Sequence[str],
) -> Path | None:
    """Resolve a requested metrics CSV, preferring explicit paths."""

    if not names:
        raise ValueError("an explicit history argument name is required")
    value = getattr(args, names[0], None)
    if not value:
        raise ValueError(f"endpoint figures requires explicit history argument --{names[0].replace('_', '-')}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing training history CSV: {path}")
    return path


def _endpoint_figure_render_cover(path: Path, case_id: str, data: Mapping[str, Any]) -> bool:
    """Render deterministic fine membership and mass/centroid evidence."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    fine_coords, fine_ids, regional_coords, regional_ids, regional_mass, regional_valid = _endpoint_figure_region_arrays(data)
    if fine_coords.ndim != 2 or fine_coords.shape[-1] != 2 or fine_coords.shape[0] != fine_ids.size:
        return False
    valid_fine = np.isfinite(fine_coords).all(axis=1) & (fine_ids >= 0)
    valid_regions = regional_valid & np.isfinite(regional_coords).all(axis=1) & (regional_ids >= 0)
    if not valid_fine.any() or not valid_regions.any():
        return False
    region_values = np.unique(fine_ids[valid_fine])
    index = {int(value): ordinal for ordinal, value in enumerate(region_values)}
    colors = np.asarray([index.get(int(value), -1) for value in fine_ids[valid_fine]], dtype=float)
    cmap = plt.get_cmap("turbo", max(len(region_values), 2))
    norm = BoundaryNorm(np.arange(-0.5, len(region_values) + 0.5), cmap.N)
    fig, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    axis.scatter(
        fine_coords[valid_fine, 0],
        fine_coords[valid_fine, 1],
        c=colors,
        cmap=cmap,
        norm=norm,
        s=26,
        marker="s",
        linewidths=0.2,
        edgecolors="none",
        label="fine 24×8 membership (deterministic IDs)",
    )
    mass = np.maximum(regional_mass[valid_regions], 0.0)
    scale = float(np.nanmax(mass)) if mass.size and np.isfinite(mass).any() else 1.0
    region_colors = [index.get(int(value), 0) for value in regional_ids[valid_regions]]
    axis.scatter(
        regional_coords[valid_regions, 0],
        regional_coords[valid_regions, 1],
        c=region_colors,
        cmap=cmap,
        norm=norm,
        s=36.0 + 180.0 * mass / max(scale, 1.0e-12),
        marker="*",
        edgecolors="black",
        linewidths=0.45,
        label="48-region centroid (marker area ∝ quadrature mass)",
        zorder=3,
    )
    for coordinate, region_id in zip(regional_coords[valid_regions], regional_ids[valid_regions]):
        axis.text(float(coordinate[0]), float(coordinate[1]), str(int(region_id)), fontsize=5.5, ha="center", va="center")
    axis.set_title(f"Regional response HONF @500 — deterministic 24×8 → 48 cover — case {case_id}")
    axis.set_xlabel("fine environment x")
    axis.set_ylabel("fine environment y")
    axis.set_aspect("equal")
    axis.legend(loc="upper right", fontsize=7, frameon=True)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array(np.asarray(region_values, dtype=float))
    colorbar = fig.colorbar(scalar, ax=axis, pad=0.02)
    colorbar.set_label("deterministic region ID")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def _endpoint_figure_render_state_read(
    path: Path,
    case_id: str,
    data: Mapping[str, Any],
    port_probe: tuple[np.ndarray, list[str], np.ndarray] | None,
    field_probe: tuple[np.ndarray, list[str], np.ndarray] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Render shared region state/read summaries for fixed field and port probes."""

    import matplotlib.pyplot as plt

    fine_coords, fine_ids, regional_coords, regional_ids, regional_mass, _regional_valid = _endpoint_figure_region_arrays(data)
    region_count = len(regional_ids)
    if region_count == 0:
        return False, []
    if "regional_response_states" not in data:
        raise KeyError(f"case {case_id}: native NPZ lacks regional_response_states")
    state = np.asarray(data["regional_response_states"], dtype=float)
    if state.ndim == 0:
        raise ValueError(f"case {case_id}: regional_response_states is scalar, expected one row per region")
    if state.shape[0] != region_count:
        raise ValueError(
            f"case {case_id}: regional_response_states must have one row per shared region "
            f"({region_count}), got {state.shape}"
        )
    if state.ndim == 1:
        state_norm = np.abs(state)
    elif state.ndim >= 2:
        state_norm = np.linalg.norm(state.reshape(state.shape[0], -1), axis=-1)
    else:
        raise ValueError(f"case {case_id}: regional_response_states must be at least one-dimensional")
    state_norm = np.asarray(state_norm, dtype=float).reshape(-1)
    if state_norm.shape != (region_count,) or not np.isfinite(state_norm).all():
        raise ValueError(f"case {case_id}: regional state norms are not finite 48-region values")
    if field_probe is None or port_probe is None:
        raise ValueError(f"regional probe JSON lacks both fixed receiver sets for case {case_id}")
    field_attention, field_labels, field_coordinates = field_probe
    field_probes = [
        (
            str(field_labels[index]),
            index,
            float(field_coordinates[index, 0]),
            float(field_coordinates[index, 1]),
        )
        for index in range(field_attention.shape[0])
    ]
    receiver_rows: list[tuple[str, np.ndarray, float, float]] = []
    for label, index, x_coordinate, y_coordinate in field_probes:
        if index < field_attention.shape[0]:
            receiver_rows.append((label, field_attention[index], x_coordinate, y_coordinate))
    port_attention, port_labels, port_coordinates = port_probe
    for index, label in enumerate(port_labels[: port_attention.shape[0]]):
        receiver_rows.append(
            (
                str(label),
                port_attention[index],
                float(port_coordinates[index, 0]),
                float(port_coordinates[index, 1]),
            )
        )
    if not receiver_rows:
        return False, []
    ids = np.asarray(regional_ids, dtype=int)
    if fine_coords.ndim == 2 and fine_coords.shape[-1] == 2 and fine_coords.shape[0] == fine_ids.size:
        valid_fine = np.isfinite(fine_coords).all(axis=1) & (fine_ids >= 0)
    else:
        valid_fine = np.zeros((fine_ids.size,), dtype=bool)
    unique_ids = np.unique(fine_ids[valid_fine]) if valid_fine.any() else np.asarray([], dtype=int)
    id_to_fine = {int(value): int(np.sum(fine_ids[valid_fine] == value)) for value in unique_ids}
    membership = np.asarray([id_to_fine.get(int(value), 0) for value in ids], dtype=float)
    matrix = np.asarray([row[1] for row in receiver_rows], dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != region_count or not np.isfinite(matrix).all():
        raise ValueError(f"case {case_id}: fixed receiver attention is not finite [Q,{region_count}]")
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 5.2), constrained_layout=True, gridspec_kw={"width_ratios": (1.0, 1.55, 1.0, 1.0)})
    if valid_fine.any():
        colors = np.asarray([int(np.where(unique_ids == value)[0][0]) for value in fine_ids[valid_fine]], dtype=float)
        cmap = plt.get_cmap("turbo", max(len(unique_ids), 2))
        axes[0].scatter(fine_coords[valid_fine, 0], fine_coords[valid_fine, 1], c=colors, cmap=cmap, s=20, marker="s")
        axes[0].set_aspect("equal")
    axes[0].set_title("Deterministic membership")
    axes[0].set_xlabel("fine x")
    axes[0].set_ylabel("fine y")
    if regional_coords.shape[0] == region_count:
        axes[0].scatter(regional_coords[:, 0], regional_coords[:, 1], s=24 + 80 * membership / max(float(np.nanmax(membership)), 1.0), marker="+", color="black", linewidths=0.7)
    image = axes[1].imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=max(float(np.nanmax(matrix)), 1.0e-12))
    axes[1].set_title("Learned receiver attention")
    axes[1].set_xlabel("shared region ID")
    axes[1].set_ylabel("fixed field / port receiver")
    axes[1].set_xticks(np.arange(region_count)[:: max(1, region_count // 12)], [str(value) for value in ids[:: max(1, region_count // 12)]])
    axes[1].set_yticks(np.arange(len(receiver_rows)), [label for label, _row, _x, _y in receiver_rows])
    fig.colorbar(image, ax=axes[1], pad=0.02, label="attention weight")
    axes[2].plot(np.arange(region_count), state_norm[:region_count], color="#2b6cb0", marker="o", markersize=2.8, linewidth=1.2)
    axes[2].set_title("Regional response-state magnitude")
    axes[2].set_xlabel("shared region ID")
    axes[2].set_ylabel("regional response-state L2 norm")
    axes[2].set_xticks(np.arange(region_count)[:: max(1, region_count // 12)], [str(value) for value in ids[:: max(1, region_count // 12)]])
    axes[2].grid(alpha=0.25)
    axes[3].bar(np.arange(region_count), regional_mass[:region_count], color="#9aa0a6", alpha=0.65)
    axes[3].set_title("Deterministic region mass")
    axes[3].set_xlabel("shared region ID")
    axes[3].set_ylabel("quadrature mass")
    axes[3].set_xticks(np.arange(region_count)[:: max(1, region_count // 12)], [str(value) for value in ids[:: max(1, region_count // 12)]])
    axes[3].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Regional response HONF @500 — shared P2-prepared state/read summary — case {case_id}")
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)

    rows: list[dict[str, Any]] = []
    for index, region_id in enumerate(ids):
        rows.append(
            {
                "case_id": case_id,
                "receiver_kind": "regional_state",
                "receiver_id": "all",
                "region_id": int(region_id),
                "fine_membership_count": int(membership[index]),
                "regional_quadrature_mass": float(regional_mass[index]),
                "regional_state_l2_norm": float(state_norm[index]),
                "attention_weight": "",
                "probe_state_phase": "P2-prepared",
            }
        )
    for receiver_label, attention, receiver_x, receiver_y in receiver_rows:
        for index, region_id in enumerate(ids):
            rows.append(
                {
                    "case_id": case_id,
                    "receiver_kind": "port_attention" if receiver_label.startswith("port") else "field_attention",
                    "receiver_id": receiver_label,
                    "region_id": int(region_id),
                    "fine_membership_count": int(membership[index]),
                    "regional_quadrature_mass": float(regional_mass[index]),
                    "regional_state_l2_norm": "",
                    "attention_weight": float(attention[index]),
                    "receiver_x": receiver_x,
                    "receiver_y": receiver_y,
                    "probe_state_phase": "P2-prepared",
                }
            )
    return True, rows


def _endpoint_figure_render_anchor_predictions(
    path: Path,
    case_ids: Sequence[str],
    native_data: Mapping[str, Mapping[str, Any]],
    dense_data: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    """Render native/Dense/reference temperature maps and signed errors."""

    import matplotlib.pyplot as plt

    records: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for case_id in case_ids:
        native = native_data.get(case_id, {})
        dense = dense_data.get(case_id, {})
        if (
            "pred_field_grid" in native
            and "pred_field_grid" in dense
            and ("gt_field_grid" in native or "gt_field_grid" in dense)
        ):
            target = np.asarray(native.get("gt_field_grid", dense.get("gt_field_grid")))
            native_prediction = np.asarray(native["pred_field_grid"])
            dense_prediction = np.asarray(dense["pred_field_grid"])
            if target.ndim >= 3 and native_prediction.ndim >= 3 and dense_prediction.ndim >= 3:
                records.append((case_id, native, dense))
    if not records:
        return False, []
    temperatures: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for _case_id, native, dense in records:
        target = np.asarray(native.get("gt_field_grid", dense.get("gt_field_grid")), dtype=float)
        native_prediction = np.asarray(native["pred_field_grid"], dtype=float)
        dense_prediction = np.asarray(dense["pred_field_grid"], dtype=float)
        if target.shape[-1] <= 4 or native_prediction.shape[-1] <= 4 or dense_prediction.shape[-1] <= 4:
            raise ValueError(f"case {_case_id}: stored field arrays lack required temperature channel index 4")
        channel = 4
        target_temperature = target[..., channel]
        native_temperature = native_prediction[..., channel]
        dense_temperature = dense_prediction[..., channel]
        if native_temperature.shape != target_temperature.shape or dense_temperature.shape != target_temperature.shape:
            raise ValueError(f"case {_case_id}: native/Dense/reference field grids do not match")
        temperatures.extend([target_temperature, native_temperature, dense_temperature])
        errors.extend([native_temperature - target_temperature, dense_temperature - target_temperature])
    finite_temperature_parts = [array[np.isfinite(array)] for array in temperatures if np.isfinite(array).any()]
    finite_error_parts = [array[np.isfinite(array)] for array in errors if np.isfinite(array).any()]
    if not finite_temperature_parts or not finite_error_parts:
        return False, []
    finite_temperature = np.concatenate(finite_temperature_parts)
    finite_error = np.concatenate(finite_error_parts)
    field_min, field_max = float(np.nanmin(finite_temperature)), float(np.nanmax(finite_temperature))
    error_limit = max(abs(float(np.nanmin(finite_error))), abs(float(np.nanmax(finite_error))), 1.0e-12)
    fig, axes = plt.subplots(len(records), 5, figsize=(16.2, max(3.0, 3.2 * len(records))), squeeze=False, constrained_layout=True)
    metric_rows: list[dict[str, Any]] = []
    for row_index, (case_id, native, dense) in enumerate(records):
        target = np.asarray(native.get("gt_field_grid", dense.get("gt_field_grid")), dtype=float)
        native_prediction = np.asarray(native["pred_field_grid"], dtype=float)
        dense_prediction = np.asarray(dense["pred_field_grid"], dtype=float)
        channel = 4
        target_temperature = target[..., channel]
        native_temperature = native_prediction[..., channel]
        dense_temperature = dense_prediction[..., channel]
        native_error = native_temperature - target_temperature
        dense_error = dense_temperature - target_temperature
        x_grid = np.asarray(native.get("x_grid", dense.get("x_grid")), dtype=float)
        y_grid = np.asarray(native.get("y_grid", dense.get("y_grid")), dtype=float)
        if x_grid.shape != target_temperature.shape or y_grid.shape != target_temperature.shape:
            raise ValueError(f"case {case_id}: stored x_grid/y_grid do not match temperature field shape")
        panels = (
            (target_temperature, "reference temperature (dataset physical units)", "inferno", field_min, field_max),
            (native_temperature, "Regional response HONF @500 (physical units)", "inferno", field_min, field_max),
            (dense_temperature, "Dense 1804 @500 (physical units)", "inferno", field_min, field_max),
            (native_error, "Regional − reference (physical units)", "coolwarm", -error_limit, error_limit),
            (dense_error, "Dense − reference (physical units)", "coolwarm", -error_limit, error_limit),
        )
        for column, (values, title, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[row_index, column]
            image = axis.pcolormesh(x_grid, y_grid, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title, fontsize=8)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                fig.colorbar(image, ax=axis, pad=0.01, fraction=0.045)
        axes[row_index, 0].set_ylabel(str(case_id), rotation=0, labelpad=18, va="center", fontsize=9)
        metric_rows.append(
            {
                "case_id": case_id,
                "field_channel": int(channel),
                "native_temperature_abs_error_mean": float(np.nanmean(np.abs(native_error))),
                "dense_temperature_abs_error_mean": float(np.nanmean(np.abs(dense_error))),
                "native_temperature_signed_error_mean": float(np.nanmean(native_error)),
                "dense_temperature_signed_error_mean": float(np.nanmean(dense_error)),
            }
        )
    fig.suptitle("Four-anchor native predictions/errors against reference and Dense 1804 matched at epoch 500")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True, metric_rows


def _endpoint_figure_timing_summary(payload: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Reduce the matched two-anchor timing schema to one row per model."""

    models = payload.get("models")
    if not isinstance(models, Sequence):
        raise ValueError("timing artifact must contain models[]")
    output: dict[str, dict[str, float]] = {}
    for model in models:
        if not isinstance(model, Mapping):
            raise ValueError("timing artifact models[] entries must be objects")
        architecture = str(model.get("architecture", ""))
        if architecture == "regional_response_honf":
            label = "Regional response HONF @500"
            variant = "regional_native_prepared"
        elif architecture == "dense_pairwise_field":
            label = "Dense 1804 @500"
            variant = "dense_projection_cached"
        else:
            continue
        real_cases = model.get("real_anchors")
        if not isinstance(real_cases, Sequence) or len(real_cases) != 2:
            raise ValueError(f"timing artifact {label} must contain exactly two real_anchors[] entries")
        median_ms: list[float] = []
        peak_bytes: list[float] = []
        for case in real_cases:
            if not isinstance(case, Mapping):
                raise ValueError(f"timing artifact {label} has an invalid real_anchors[] entry")
            variants = case.get("variants")
            details = variants.get(variant) if isinstance(variants, Mapping) else None
            phase = details.get("phases", {}).get("full_forward") if isinstance(details, Mapping) else None
            if not isinstance(phase, Mapping):
                raise ValueError(f"timing artifact {label} lacks variants.{variant}.phases.full_forward")
            elapsed = _endpoint_figure_float(phase.get("median_ms"))
            allocated = _endpoint_figure_float(phase.get("peak_allocated_bytes"))
            if not math.isfinite(elapsed) or not math.isfinite(allocated):
                raise ValueError(f"timing artifact {label} has nonfinite full-forward time or peak allocation")
            median_ms.append(elapsed)
            peak_bytes.append(allocated)
        if label in output:
            raise ValueError(f"timing artifact contains duplicate model architecture {architecture!r}")
        output[label] = {
            "full_forward_median_ms_over_two_anchors": float(np.median(median_ms)),
            "peak_allocated_mib_over_two_anchors": float(np.median(peak_bytes) / (1024.0**2)),
        }
    required = {"Regional response HONF @500", "Dense 1804 @500"}
    if set(output) != required:
        raise ValueError(f"timing artifact must contain Regional and Dense matched variants; found {sorted(output)}")
    return output


def _endpoint_figure_render_convergence_cost(
    path: Path,
    histories: Sequence[tuple[str, Path, Mapping[str, Sequence[float]]]],
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    timing_summary: Mapping[str, Mapping[str, float]],
) -> tuple[bool, list[dict[str, Any]]]:
    """Render sampled validation convergence, measured cost, and peak allocation."""

    import matplotlib.pyplot as plt

    history_records: list[tuple[str, list[tuple[float, float]]]] = []
    for label, _path, history in histories:
        if not history.get("val_field_mse"):
            raise ValueError(f"training history for {label} lacks required named column val_field_mse")
        sampled = _endpoint_figure_sample_history(history, "val_field_mse", maximum=80, max_epoch=500.0)
        if not sampled:
            raise ValueError(f"training history for {label} has no finite positive val_field_mse values through epoch 500")
        history_records.append((label, sampled))
    if len(history_records) != 2:
        raise ValueError("endpoint figures requires native and Dense training histories")
    accuracy_rows: list[dict[str, Any]] = []
    for label in ("Regional response HONF @500", "Dense 1804 @500"):
        rows = model_rows[label]
        values = [_endpoint_figure_float(row.get("global_field_fluid_norm_l2")) for row in rows]
        if len(values) != 90 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{label} requires 90 finite global_field_fluid_norm_l2 values")
        accuracy_rows.append(
            {
                "model_label": label,
                "equal_case90_global_field_fluid_norm_l2_mean": float(np.mean(values)),
                **timing_summary[label],
            }
        )
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.9), constrained_layout=True)
    colors = ("#2b6cb0", "#c05621")
    history_axis, cost_axis, memory_axis = axes
    for index, (label, sampled) in enumerate(history_records):
        history_axis.plot(
            [epoch for epoch, _value in sampled],
            [value for _epoch, value in sampled],
            marker="o",
            markersize=2.4,
            linewidth=1.25,
            color=colors[index],
            label=label,
        )
    history_axis.set_yscale("log")
    history_axis.set_xlabel("training epoch (sampled ≤500)")
    history_axis.set_ylabel("validation field MSE")
    history_axis.set_title("Sampled convergence")
    history_axis.grid(alpha=0.25)
    history_axis.legend(fontsize=8)
    for index, row in enumerate(accuracy_rows):
        x_value = float(row["full_forward_median_ms_over_two_anchors"])
        y_value = float(row["equal_case90_global_field_fluid_norm_l2_mean"])
        cost_axis.scatter(x_value, y_value, s=58, color=colors[index], label=row["model_label"], alpha=0.85)
        cost_axis.annotate(str(row["model_label"]), (x_value, y_value), xytext=(5, 4), textcoords="offset points", fontsize=7)
        memory_axis.bar(index, float(row["peak_allocated_mib_over_two_anchors"]), color=colors[index], label=row["model_label"])
    cost_axis.set_xlabel("median full-forward time over two anchors (ms)")
    cost_axis.set_ylabel("equal-case mean fluid relative L2 (90 cases)")
    cost_axis.set_title("Accuracy versus measured cost")
    cost_axis.grid(alpha=0.25)
    cost_axis.legend(fontsize=7)
    memory_axis.set_xticks(range(len(accuracy_rows)), ["Regional", "Dense"])
    memory_axis.set_ylabel("median peak allocated (MiB)")
    memory_axis.set_title("Peak allocation")
    memory_axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Regional response HONF @500 — sampled convergence and measured cost")
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    source_rows: list[dict[str, Any]] = []
    for label, sampled in history_records:
        for epoch, value in sampled:
            source_rows.append({"record_kind": "convergence", "model_label": label, "epoch": epoch, "metric": "val_field_mse", "value": value})
    source_rows.extend({"record_kind": "accuracy_cost", **row} for row in accuracy_rows)
    return True, source_rows


def run_endpoint_figures(args: argparse.Namespace) -> dict[str, Any]:
    """Render compact native endpoint figures from stored comparison artifacts.

    This helper consumes the existing comparator tables/debug NPZ exports and
    the maintained named-column training-history reader.  It never loads a
    checkpoint, creates a duplicate debug archive, or fabricates a missing
    scientific panel.  The full90 Dense endpoint supplies tables; the explicit
    Dense debug directories supply only the fixed anchor NPZs.
    """

    import json as _json

    import matplotlib

    matplotlib.use("Agg")

    required_args = (
        "native_endpoint",
        "dense500_endpoint",
        "dense500_debug_dir",
        "reader500_endpoint",
        "figure_dir",
        "output",
        "native_history",
        "dense_history",
        "timing",
    )
    missing_args = [name for name in required_args if not getattr(args, name, None)]
    if missing_args:
        raise ValueError(f"endpoint figures requires explicit arguments: {', '.join(missing_args)}")
    native_root = Path(args.native_endpoint).expanduser().resolve()
    dense_root = Path(args.dense500_endpoint).expanduser().resolve()
    dense_debug_values = args.dense500_debug_dir
    if isinstance(dense_debug_values, (str, Path)):
        dense_debug_roots = [Path(dense_debug_values).expanduser().resolve()]
    else:
        dense_debug_roots = [Path(value).expanduser().resolve() for value in dense_debug_values]
    if not dense_debug_roots:
        raise ValueError("endpoint figures requires at least one Dense debug directory")
    reader_root = Path(args.reader500_endpoint).expanduser().resolve()
    figure_dir = Path(args.figure_dir).expanduser().resolve()
    required_roots = [
        ("native_endpoint", native_root),
        ("dense500_endpoint", dense_root),
        *[(f"dense500_debug_dir[{index}]", root) for index, root in enumerate(dense_debug_roots)],
        ("reader500_endpoint", reader_root),
    ]
    for name, root in required_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"{name} directory does not exist: {root}")
    figure_dir.mkdir(parents=True, exist_ok=True)
    case_ids = ANCHOR_CASE_IDS

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "endpoint_figures",
        "stage": "endpoint500",
        "model_label": "Regional response HONF @500",
        "case_ids": list(case_ids),
        "inputs": {
            "native_endpoint": None if native_root is None else str(native_root),
            "dense500_endpoint": None if dense_root is None else str(dense_root),
            "dense500_debug_dir": [str(root) for root in dense_debug_roots],
            "dense500_debug_prefix": str(
                getattr(args, "dense500_debug_prefix", None)
                or "Dense_pairwise_adaptation__500__"
            ),
            "reader500_endpoint": None if reader_root is None else str(reader_root),
        },
        "figures": [],
        "sources": [],
        "limitations": [
            "Fine environment region IDs are deterministic membership metadata; they are not learned probabilities.",
            "Attention weights are learned receiver routing summaries, and state L2 norms are regional response-state magnitudes; neither is a causal influence estimate.",
            "Anchor maps use the native temperature channel and compare the same stored reference target against native and Dense predictions.",
            "Convergence uses the maintained named-column history reader so historical CSV schema changes are not interpreted positionally.",
        ],
    }

    native_npz: dict[str, dict[str, Any]] = {}
    dense_npz: dict[str, dict[str, Any]] = {}
    dense_debug_prefix = str(
        getattr(args, "dense500_debug_prefix", None)
        or "Dense_pairwise_adaptation__500__"
    )
    for case_id in case_ids:
        native_path = _endpoint_figure_npz_for_case(native_root, case_id)
        dense_path = _endpoint_figure_npz_for_case_from_dirs(
            dense_debug_roots,
            case_id,
            filename_prefix=dense_debug_prefix,
        )
        if native_path is None:
            raise FileNotFoundError(f"missing native endpoint debug NPZ for case {case_id} under {native_root / 'debug_npz'}")
        if dense_path is None:
            raise FileNotFoundError(
                f"missing Dense anchor debug NPZ for case {case_id} with prefix "
                f"{dense_debug_prefix!r} under {dense_debug_roots}"
            )
        native_npz[case_id] = _endpoint_figure_load_npz(native_path)
        dense_npz[case_id] = _endpoint_figure_load_npz(dense_path)
        required_native_keys = {"regional_response_states", "regional_environment_attention"}
        missing_native_keys = sorted(required_native_keys.difference(native_npz[case_id]))
        if missing_native_keys:
            raise KeyError(f"native endpoint NPZ for case {case_id} is missing keys: {missing_native_keys}")
        for key in ("pred_field_grid", "gt_field_grid", "x_grid", "y_grid"):
            if key not in native_npz[case_id] or key not in dense_npz[case_id]:
                raise KeyError(f"anchor endpoint NPZ for case {case_id} is missing required field key: {key}")
        manifest["inputs"].setdefault("native_debug_npz", {})[case_id] = str(native_path)
        manifest["inputs"].setdefault("dense_debug_npz", {})[case_id] = str(dense_path)

    native_anchor = case_ids[0]
    cover_path = figure_dir / "regional_cover_24x8_48.png"
    if not _endpoint_figure_render_cover(cover_path, native_anchor, native_npz[native_anchor]):
        raise ValueError("native endpoint regional cover arrays could not be rendered")
    manifest["figures"].append({"kind": "regional_cover", "path": str(cover_path), "case_id": native_anchor})
    fine_coords, fine_ids, regional_coords, regional_ids, regional_mass, regional_valid = _endpoint_figure_region_arrays(native_npz[native_anchor])
    cover_rows: list[dict[str, Any]] = []
    for index, (coordinate, region_id) in enumerate(zip(fine_coords, fine_ids)):
        cover_rows.append({"case_id": native_anchor, "element": "fine_environment", "fine_index": index, "x": float(coordinate[0]), "y": float(coordinate[1]), "region_id": int(region_id)})
    for index, (coordinate, region_id, mass, valid) in enumerate(zip(regional_coords, regional_ids, regional_mass, regional_valid)):
        cover_rows.append({"case_id": native_anchor, "element": "regional_centroid", "fine_index": "", "x": float(coordinate[0]), "y": float(coordinate[1]), "region_id": int(region_id), "regional_quadrature_mass": float(mass), "regional_valid": bool(valid)})
    cover_csv = figure_dir / "regional_cover_source.csv"
    _endpoint_figure_write_csv(cover_csv, cover_rows)
    manifest["sources"].append(str(cover_csv))

    probe_path = _endpoint_figure_probe_path(args, native_root)
    state_rows: list[dict[str, Any]] = []
    try:
        probe_payload = _json.loads(probe_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to parse bounded regional probe artifact: {probe_path}") from exc
    if not isinstance(probe_payload, Mapping) or not isinstance(probe_payload.get("results"), Sequence):
        raise ValueError(f"regional probe artifact lacks results[]: {probe_path}")
    probe_rows = {
        str(row.get("case_id")): row
        for row in probe_payload["results"]
        if isinstance(row, Mapping) and row.get("case_id") is not None
    }
    for case_id in ("0273", "0298"):
        native_data = native_npz[case_id]
        _fine_coords, _fine_ids, native_coords, native_ids, native_mass, _native_valid = _endpoint_figure_region_arrays(native_data)
        region_count = len(native_ids)
        probe_row = probe_rows.get(case_id)
        if probe_row is None:
            raise ValueError(f"regional probe artifact has no result for case {case_id}")
        probe_ids = np.asarray(probe_row.get("region_ids", []), dtype=int).reshape(-1)
        if probe_ids.shape != native_ids.shape or not np.array_equal(probe_ids, native_ids):
            raise ValueError(f"case {case_id}: probe and native regional_ids do not match exactly")
        for key, expected in (
            ("region_coordinates", native_coords),
            ("region_mass", native_mass),
        ):
            probe_values = np.asarray(probe_row.get(key, []), dtype=float)
            if probe_values.shape != expected.shape or not np.allclose(probe_values, expected, rtol=1.0e-5, atol=1.0e-7):
                raise ValueError(f"case {case_id}: probe {key} does not match native regional metadata")
        probe_state_norm = np.asarray(probe_row.get("region_state_norm", []), dtype=float).reshape(-1)
        native_state = np.asarray(native_data["regional_response_states"], dtype=float)
        native_state_norm = np.linalg.norm(native_state.reshape(native_state.shape[0], -1), axis=-1)
        if probe_state_norm.shape != native_state_norm.shape or not np.allclose(
            probe_state_norm, native_state_norm, rtol=1.0e-5, atol=1.0e-7
        ):
            raise ValueError(f"case {case_id}: probe region_state_norm does not match native regional states")
        port_probe = _endpoint_figure_port_attention(probe_path, case_id, region_count)
        field_probe = _endpoint_figure_field_probe(probe_path, case_id, region_count)
        if port_probe is None or field_probe is None:
            raise ValueError(f"regional probe JSON lacks required physical_ports/field_probes for case {case_id}")
        manifest["inputs"]["port_probe_artifact"] = str(probe_path)
        state_path = figure_dir / f"regional_state_read_summary__{case_id}.png"
        rendered, rows = _endpoint_figure_render_state_read(
            state_path,
            case_id,
            native_data,
            port_probe,
            field_probe,
        )
        if not rendered:
            raise ValueError(f"failed to render regional state/read summary for case {case_id}")
        manifest["figures"].append(
            {
                "kind": "regional_state_read_summary",
                "path": str(state_path),
                "case_id": case_id,
                "port_probe_available": True,
                "field_probe_available": True,
                "probe_state_phase": "P2-prepared",
            }
        )
        state_rows.extend(rows)
    state_csv = figure_dir / "regional_state_read_source.csv"
    if state_rows:
        _endpoint_figure_write_csv(state_csv, state_rows)
        manifest["sources"].append(str(state_csv))

    native_table_rows = _endpoint_figure_read_csv(_endpoint_figure_table_path(native_root, "per_case_metrics.csv"))
    dense_table_rows = _endpoint_figure_read_csv(_endpoint_figure_table_path(dense_root, "per_case_metrics.csv"))
    reader_table_rows = _endpoint_figure_read_csv(_endpoint_figure_table_path(reader_root, "per_case_metrics.csv"))
    if not native_table_rows or not dense_table_rows or not reader_table_rows:
        raise FileNotFoundError("endpoint figures requires per_case_metrics.csv under native, Dense, and Reader endpoints")
    native_selector = getattr(args, "native_selector", None)
    dense_selector = getattr(args, "dense500_selector", None) or "Dense pairwise adaptation @500"
    reader_selector = getattr(args, "reader500_selector", None)
    selected_tables = {
        "Regional response HONF @500": _endpoint_figure_select_rows(native_table_rows, selector=native_selector),
        "Dense 1804 @500": _endpoint_figure_select_rows(dense_table_rows, selector=dense_selector),
        "Reader @500": _endpoint_figure_select_rows(reader_table_rows, selector=reader_selector),
    }
    selected_case_sets = {
        label: {str(row.get("case_id")) for row in rows}
        for label, rows in selected_tables.items()
    }
    if any(len(rows) != 90 or len(selected_case_sets[label]) != 90 for label, rows in selected_tables.items()):
        raise ValueError(
            "endpoint figure tables must contain exactly one matched 90-case model selection: "
            + ", ".join(f"{label}={len(rows)}" for label, rows in selected_tables.items())
        )
    native_case_set = selected_case_sets["Regional response HONF @500"]
    if any(case_set != native_case_set for case_set in selected_case_sets.values()):
        raise ValueError("native, Dense, and Reader endpoint tables do not share the same 90 case IDs")
    anchor_metric_rows: list[dict[str, Any]] = []
    for model_label, rows in selected_tables.items():
        for row in rows:
            case_id = str(row.get("case_id", ""))
            if case_id not in case_ids:
                continue
            anchor_metric_rows.append(
                {
                    "case_id": case_id,
                    "model_label": model_label,
                    "global_field_fluid_norm_l2": _endpoint_figure_float(row.get("global_field_fluid_norm_l2")),
                    "global_field_all_norm_l2": _endpoint_figure_float(row.get("global_field_all_norm_l2")),
                    "internal_temperature_physical_relative_l2": _endpoint_figure_float(row.get("internal_temperature_physical_relative_l2")),
                    "t_surface_mean_norm_l2": _endpoint_figure_float(row.get("t_surface_mean_norm_l2")),
                    "q_normal_mean_norm_l2": _endpoint_figure_float(row.get("q_normal_mean_norm_l2")),
                }
            )
    anchor_csv = figure_dir / "regional_anchor_metrics_source.csv"
    if anchor_metric_rows:
        _endpoint_figure_write_csv(anchor_csv, anchor_metric_rows)
        manifest["sources"].append(str(anchor_csv))
    anchor_path = figure_dir / "regional_anchor_predictions_errors.png"
    rendered, map_rows = _endpoint_figure_render_anchor_predictions(anchor_path, case_ids, native_npz, dense_npz)
    if not rendered or {str(row["case_id"]) for row in map_rows} != set(case_ids):
        raise ValueError("paired native and Dense debug NPZs did not render all four anchor prediction/error maps")
    manifest["figures"].append({"kind": "anchor_predictions_errors", "path": str(anchor_path), "case_ids": [row["case_id"] for row in map_rows]})
    anchor_metric_rows.extend({"model_label": "map_summary", **row} for row in map_rows)
    if map_rows:
        _endpoint_figure_write_csv(anchor_csv, anchor_metric_rows)

    timing_path = Path(args.timing).expanduser().resolve()
    timing_payload = _json.loads(timing_path.read_text())
    timing_summary = _endpoint_figure_timing_summary(timing_payload)
    manifest["inputs"]["timing"] = str(timing_path)
    histories: list[tuple[str, Path, Mapping[str, Sequence[float]]]] = []
    history_specs = (
        ("Regional response HONF @500", ("native_history",)),
        ("Dense 1804 @500", ("dense_history",)),
    )
    from channelthermal.training.reporting import _read_metric_history

    for label, names in history_specs:
        history_path = _endpoint_figure_history_path(args, names)
        history = _read_metric_history(history_path)
        histories.append((label, history_path, history))
        manifest["inputs"].setdefault("histories", {})[label] = str(history_path)
    convergence_path = figure_dir / "regional_convergence_accuracy_cost.png"
    rendered, convergence_rows = _endpoint_figure_render_convergence_cost(
        convergence_path,
        histories,
        selected_tables,
        timing_summary,
    )
    if not rendered:
        raise ValueError("sampled convergence and measured cost figure could not be rendered")
    manifest["figures"].append({"kind": "convergence_accuracy_cost", "path": str(convergence_path)})
    convergence_csv = figure_dir / "regional_convergence_accuracy_cost_source.csv"
    _endpoint_figure_write_csv(convergence_csv, convergence_rows)
    manifest["sources"].append(str(convergence_csv))

    manifest_path = Path(args.output).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(_json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def run_endpoint_reduction(args: argparse.Namespace) -> dict[str, Any]:
    """Reduce matched endpoint tables without loading a checkpoint or model.

    The endpoint candidate is compared with the exact-500 Dense/Legacy table
    and the separate Reader table.  All pooled values are recomputed from raw
    per-case SSE, target-SSE, and value-count columns; equal-case summaries are
    retained separately.  A maturity reference is recorded in the manifest
    only, so this reducer cannot silently mix the early endpoint with the
    mature result.
    """

    import re

    import pandas as pd

    expected_cases = 90

    def arg_path(names: tuple[str, ...], *, required: bool = True) -> Path | None:
        for name in names:
            value = getattr(args, name, None)
            if value is not None and str(value):
                return Path(value).expanduser().resolve()
        if required:
            raise ValueError(f"endpoint reduction requires one of: {', '.join(names)}")
        return None

    def table_dir(value: Path) -> Path:
        path = value
        if path.is_file():
            if path.name != "per_case_metrics.csv":
                raise ValueError(f"expected per_case_metrics.csv or a tables directory, got {path}")
            path = path.parent
        if (path / "per_case_metrics.csv").exists():
            return path
        if (path / "tables" / "per_case_metrics.csv").exists():
            return path / "tables"
        raise FileNotFoundError(f"missing per_case_metrics.csv under {value}")

    def read_table(path: Path, *, case_ids: bool = False) -> pd.DataFrame:
        dtype = {"case_id": "string"} if case_ids else None
        return pd.read_csv(path, dtype=dtype)

    def identity(frame: pd.DataFrame) -> pd.Series:
        columns = [column for column in ("model_label", "checkpoint", "run_dir") if column in frame]
        if not columns:
            raise ValueError("comparison table must contain model_label, checkpoint, or run_dir")
        return frame[columns].fillna("").astype(str).agg(" ".join, axis=1)

    def select_rows(
        frame: pd.DataFrame,
        *,
        selector: str | None,
        label_hint: str | None = None,
        table_name: str,
    ) -> tuple[pd.DataFrame, str]:
        if "model_label" not in frame:
            raise ValueError(f"{table_name}: model_label column is required")
        labels = [str(value) for value in frame["model_label"].dropna().unique()]
        if selector:
            mask = identity(frame).str.contains(re.escape(selector), case=False, na=False)
        elif label_hint:
            mask = identity(frame).str.contains(re.escape(label_hint), case=False, na=False)
            if not bool(mask.any()):
                mask = frame["model_label"].astype(str).eq(label_hint)
        elif len(labels) == 1:
            mask = frame["model_label"].astype(str).eq(labels[0])
        else:
            raise ValueError(f"{table_name}: multiple model labels require an explicit selector: {labels}")
        selected = frame.loc[mask].copy()
        selected_label = str(selected["model_label"].iloc[0]) if not selected.empty else ""
        if selected.empty:
            raise ValueError(f"{table_name}: selector {selector or label_hint!r} matched no rows")
        if selected["model_label"].astype(str).nunique() != 1:
            raise ValueError(f"{table_name}: selector matched multiple model labels")
        if "case_id" in selected:
            selected["case_id"] = selected["case_id"].astype("string")
            if len(selected) != expected_cases or selected["case_id"].nunique() != expected_cases:
                raise ValueError(
                    f"{table_name}: expected {expected_cases} unique case rows, "
                    f"found {len(selected)} rows/{selected['case_id'].nunique()} IDs"
                )
        return selected, selected_label

    def load_view(key: str, source: Path, selector: str | None) -> dict[str, Any]:
        tables = table_dir(source)
        case_path = tables / "per_case_metrics.csv"
        summary_path = tables / "model_summary_metrics.csv"
        strata_path = tables / "physical_stratum_metrics.csv"
        missing = [str(path) for path in (summary_path, strata_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{key}: missing required comparison tables:\n" + "\n".join(missing))
        case_table = read_table(case_path, case_ids=True)
        case_rows, label = select_rows(case_table, selector=selector, table_name=f"{key}/per_case_metrics")
        summary_table = read_table(summary_path)
        summary_rows, _ = select_rows(
            summary_table,
            selector=selector,
            label_hint=label,
            table_name=f"{key}/model_summary_metrics",
        )
        if len(summary_rows) != 1:
            raise ValueError(f"{key}: expected one selected model summary row, found {len(summary_rows)}")
        strata_table = read_table(strata_path)
        # The physical-stratum table is keyed by model_label only; its rows
        # do not carry the checkpoint/run directory used by the per-case and
        # summary tables.  Reuse the already selected model label here rather
        # than applying a run selector that cannot match this table.
        if "model_label" not in strata_table:
            raise ValueError(f"{key}/physical_stratum_metrics: model_label column is required")
        strata_rows = strata_table.loc[strata_table["model_label"].astype(str).eq(label)].copy()
        if strata_rows.empty:
            raise ValueError(f"{key}/physical_stratum_metrics: selected model {label!r} is absent")
        if strata_rows["model_label"].astype(str).nunique() != 1:
            raise ValueError(f"{key}/physical_stratum_metrics: selected multiple model labels")
        strata_keys = strata_rows[["stratum_name", "stratum_value"]].astype(str).drop_duplicates()
        if len(strata_rows) != 13 or len(strata_keys) != 13:
            raise ValueError(f"{key}: expected 13 unique physical strata rows")
        return {
            "key": key,
            "source": str(tables),
            "cases": case_rows,
            "summary": summary_rows.iloc[0].to_dict(),
            "strata": strata_rows,
            "model_label": label,
        }

    candidate_source = arg_path(("candidate_tables", "candidate_endpoint_tables", "candidate"))
    dense_source = arg_path(("dense500_tables", "dense_tables", "dense500"))
    reader_source = arg_path(("reader500_tables", "reader_tables", "reader500"))
    maturity_source = arg_path(
        ("maturity5000_tables", "maturity_tables", "maturity5000"), required=False
    )
    output_source = arg_path(("output_dir", "reduction_output_dir", "output"))
    assert candidate_source is not None and dense_source is not None and reader_source is not None
    assert output_source is not None
    # A temporary or endpoint directory can itself contain dots in its name;
    # inspect filesystem type before treating a path suffix as an output file.
    if output_source.exists():
        output_root = output_source if output_source.is_dir() else output_source.parent
    elif output_source.suffix in {".json", ".csv"}:
        output_root = output_source.parent
    else:
        output_root = output_source
    reduction_dir = output_root if output_root.name == "reduction" else output_root / "reduction"
    reduction_dir.mkdir(parents=True, exist_ok=True)

    candidate_selector = getattr(args, "candidate_selector", None) or getattr(args, "candidate_label", None)
    dense_selector = getattr(args, "dense500_selector", None) or "Run_1804_"
    legacy_selector = getattr(args, "legacy500_selector", None) or "Run_1401_"
    reader_selector = getattr(args, "reader500_selector", None) or getattr(args, "reader_label", None)
    views = {
        "candidate_endpoint500": load_view("candidate_endpoint500", candidate_source, candidate_selector),
        "dense500": load_view("dense500", dense_source, dense_selector),
        "legacy500": load_view("legacy500", dense_source, legacy_selector),
        "reader500": load_view("reader500", reader_source, reader_selector),
    }

    def numeric(frame: pd.DataFrame, column: str, *, view_key: str) -> pd.Series:
        if column not in frame:
            raise ValueError(f"{view_key}: missing metric column {column}")
        values = pd.to_numeric(frame[column], errors="coerce")
        if not bool(values.notna().all()) or not bool(values.map(np.isfinite).all()):
            raise ValueError(f"{view_key}: nonfinite values in {column}")
        return values.astype(float)

    candidate_ids = set(views["candidate_endpoint500"]["cases"]["case_id"].astype(str))
    case_set_rows: list[dict[str, Any]] = []
    for key, view in views.items():
        ids = set(view["cases"]["case_id"].astype(str))
        passed = len(ids) == expected_cases and ids == candidate_ids
        case_set_rows.append(
            {
                "dataset": key,
                "model_label": view["model_label"],
                "num_rows": len(view["cases"]),
                "num_unique_case_ids": len(ids),
                "same_as_candidate": passed,
            }
        )
        if not passed:
            raise ValueError(f"{key}: case IDs do not match the candidate's exact 90-case set")

    target_count_columns = sorted(
        set.intersection(
            *[
                {
                    column
                    for column in view["cases"].columns
                    if column.endswith(("_target_sse", "_num_values"))
                }
                for view in views.values()
            ]
        )
    )
    if not target_count_columns:
        raise ValueError("no common target-SSE/value-count columns were found")
    target_count_rows: list[dict[str, Any]] = []
    reference_cases = views["candidate_endpoint500"]["cases"].set_index("case_id")
    for key, view in views.items():
        current = view["cases"].set_index("case_id")
        for case_id in sorted(candidate_ids):
            for column in target_count_columns:
                reference = float(reference_cases.loc[case_id, column])
                observed = float(current.loc[case_id, column])
                finite = np.isfinite(reference) and np.isfinite(observed)
                passed = bool(
                    finite and np.isclose(reference, observed, rtol=1.0e-7, atol=1.0e-7)
                )
                target_count_rows.append(
                    {
                        "dataset": key,
                        "case_id": str(case_id),
                        "metric": column,
                        "reference_candidate": reference if np.isfinite(reference) else None,
                        "observed": observed if np.isfinite(observed) else None,
                        "difference": observed - reference if finite else None,
                        "status": "pass" if passed else "mismatch",
                    }
                )
                if not passed:
                    raise ValueError(f"{key}/{case_id}/{column}: target/count mismatch")

    base_specs = [
        ("global_field_fluid_norm", "global", "global fluid"),
        ("global_field_near_interface_norm", "near_far", "near-interface"),
        ("global_field_far_fluid_norm", "near_far", "far-fluid"),
    ]
    base_specs.extend(
        (f"field_{channel}_fluid_norm", "channel", f"field {channel} normalized")
        for channel in ("u", "v", "p", "omega", "temperature")
    )
    base_specs.extend(
        (f"field_{channel}_fluid_physical", "channel", f"field {channel}")
        for channel in ("u", "v", "p", "omega", "temperature")
    )
    base_specs.extend(
        (f"{name}_physical", "ports", label)
        for name, label in (
            ("port_t_env_provisional", "provisional outside temperature"),
            ("port_t_env_final", "final outside temperature"),
            ("port_h_effective_provisional", "provisional effective heat transfer"),
            ("port_h_effective_final", "final effective heat transfer"),
            ("internal_temperature", "internal temperature"),
            ("interface_t_surface", "surface temperature"),
            ("interface_q_normal", "normal heat flux"),
        )
    )
    kpi_specs = [
        ("pressure_drop_inlet_minus_outlet_physical_abs_error", "kpi", "pressure drop"),
        ("mean_outlet_temperature_physical_abs_error", "kpi", "mean outlet temperature"),
        ("mean_active_module_temperature_physical_abs_error", "kpi", "mean active-module temperature"),
    ]

    def l2_column(base: str) -> str:
        return f"{base}_l2" if base.endswith("_norm") else f"{base}_relative_l2"

    def raw_pool(view_key: str, base: str) -> dict[str, float]:
        frame = views[view_key]["cases"]
        sse = float(numeric(frame, f"{base}_sse", view_key=view_key).sum())
        target = float(numeric(frame, f"{base}_target_sse", view_key=view_key).sum())
        count = float(numeric(frame, f"{base}_num_values", view_key=view_key).sum())
        if target <= 0.0 or count <= 0.0:
            raise ValueError(f"{view_key}/{base}: nonpositive target-SSE or value count")
        return {
            "sse": sse,
            "target_sse": target,
            "num_values": count,
            "mse": sse / count,
            "relative_l2": math.sqrt(sse / target),
        }

    def stored(summary: Mapping[str, Any], key: str) -> float | None:
        value = summary.get(key)
        if value is None or str(value) == "" or not np.isfinite(float(value)):
            return None
        return float(value)

    def summary_base(base: str) -> str:
        return base.removesuffix("_norm") if base.endswith("_norm") else base

    metric_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    for key, view in views.items():
        for base, category, label in base_specs:
            raw = raw_pool(key, base)
            summary_key = summary_base(base)
            stored_l2 = stored(view["summary"], f"{summary_key}_pooled_relative_l2")
            stored_mse = stored(view["summary"], f"{summary_key}_pooled_mse")
            if stored_l2 is None:
                raise ValueError(f"{key}/{base}: selected summary has no pooled relative L2")
            l2_values = numeric(view["cases"], l2_column(base), view_key=key).to_numpy()
            l2_stats = {
                "equal_case_mean": float(np.mean(l2_values)),
                "equal_case_median": float(np.median(l2_values)),
                "equal_case_p95": float(np.quantile(l2_values, 0.95)),
                "equal_case_max": float(np.max(l2_values)),
            }
            pool_passed = bool(
                np.isclose(raw["relative_l2"], stored_l2, rtol=1.0e-11, atol=1.0e-11)
                and (stored_mse is None or np.isclose(raw["mse"], stored_mse, rtol=1.0e-11, atol=1.0e-11))
            )
            pool_rows.append(
                {
                    "dataset": key,
                    "model_label": view["model_label"],
                    "metric": base,
                    "raw_sse": raw["sse"],
                    "raw_target_sse": raw["target_sse"],
                    "raw_num_values": raw["num_values"],
                    "recomputed_pooled_mse": raw["mse"],
                    "stored_pooled_mse": stored_mse,
                    "recomputed_pooled_relative_l2": raw["relative_l2"],
                    "stored_pooled_relative_l2": stored_l2,
                    "passed": pool_passed,
                }
            )
            if not pool_passed:
                raise ValueError(f"{key}/{base}: stored pooled metric does not reconcile")
            metric_rows.append(
                {
                    "dataset": key,
                    "model_label": view["model_label"],
                    "category": category,
                    "metric": f"{base}_relative_l2",
                    "metric_label": f"{label} relative L2",
                    "num_cases": expected_cases,
                    "pooled_mse": raw["mse"],
                    "pooled_relative_l2": raw["relative_l2"],
                    "pooled_mean_absolute_error": None,
                    **l2_stats,
                }
            )
            mae_key = f"{base}_mae"
            if mae_key in view["cases"]:
                mae_values = numeric(view["cases"], mae_key, view_key=key).to_numpy()
                mae_counts = numeric(view["cases"], f"{base}_num_values", view_key=key).to_numpy()
                total_count = float(mae_counts.sum())
                if total_count <= 0.0:
                    raise ValueError(f"{key}/{base}: nonpositive MAE value count")
                metric_rows.append(
                    {
                        "dataset": key,
                        "model_label": view["model_label"],
                        "category": category,
                        "metric": mae_key,
                        "metric_label": f"{label} MAE",
                        "num_cases": expected_cases,
                        "pooled_mse": None,
                        "pooled_relative_l2": None,
                        "pooled_mean_absolute_error": float(np.sum(mae_values * mae_counts) / total_count),
                        "equal_case_mean": float(np.mean(mae_values)),
                        "equal_case_median": float(np.median(mae_values)),
                        "equal_case_p95": float(np.quantile(mae_values, 0.95)),
                        "equal_case_max": float(np.max(mae_values)),
                    }
                )
        for base, category, label in kpi_specs:
            values = numeric(view["cases"], base, view_key=key).to_numpy()
            metric_rows.append(
                {
                    "dataset": key,
                    "model_label": view["model_label"],
                    "category": category,
                    "metric": base,
                    "metric_label": f"{label} absolute error",
                    "num_cases": expected_cases,
                    "pooled_mse": None,
                    "pooled_relative_l2": None,
                    "pooled_mean_absolute_error": float(np.mean(values)),
                    "equal_case_mean": float(np.mean(values)),
                    "equal_case_median": float(np.median(values)),
                    "equal_case_p95": float(np.quantile(values, 0.95)),
                    "equal_case_max": float(np.max(values)),
                }
            )

    pair_specs: list[tuple[str, str, str, str]] = [
        (base, l2_column(base), category, f"{label} relative L2")
        for base, category, label in base_specs
    ]
    pair_specs.extend(
        (base, base, category, f"{label} absolute error") for base, category, label in kpi_specs
    )
    paired_rows: list[dict[str, Any]] = []
    paired_summary: list[dict[str, Any]] = []
    for comparator_key in ("dense500", "legacy500", "reader500"):
        candidate = views["candidate_endpoint500"]["cases"].set_index("case_id")
        comparator = views[comparator_key]["cases"].set_index("case_id").reindex(candidate.index)
        for metric, column, category, label in pair_specs:
            candidate_values = numeric(candidate, column, view_key="candidate_endpoint500")
            comparator_values = numeric(comparator, column, view_key=comparator_key)
            deltas = (candidate_values - comparator_values).reindex(candidate.index)
            for case_id, candidate_value, comparator_value, delta in zip(
                candidate.index.astype(str), candidate_values, comparator_values, deltas, strict=True
            ):
                paired_rows.append(
                    {
                        "comparator": comparator_key,
                        "metric": metric,
                        "metric_label": label,
                        "category": category,
                        "case_id": case_id,
                        "candidate_value": float(candidate_value),
                        "comparator_value": float(comparator_value),
                        "candidate_minus_comparator": float(delta),
                    }
                )
            delta_values = deltas.to_numpy(dtype=float)
            ties = delta_values == 0.0
            paired_summary.append(
                {
                    "comparator": comparator_key,
                    "metric": metric,
                    "metric_label": label,
                    "category": category,
                    "num_cases": expected_cases,
                    "candidate_better_cases": int(((delta_values < 0.0) & ~ties).sum()),
                    "candidate_worse_cases": int(((delta_values > 0.0) & ~ties).sum()),
                    "ties": int(ties.sum()),
                    "mean_delta": float(np.mean(delta_values)),
                    "median_delta": float(np.median(delta_values)),
                    "p05_delta": float(np.quantile(delta_values, 0.05)),
                    "p95_delta": float(np.quantile(delta_values, 0.95)),
                    "min_delta": float(np.min(delta_values)),
                    "max_delta": float(np.max(delta_values)),
                }
            )

    difficult_rows: list[dict[str, Any]] = []
    candidate_frame = views["candidate_endpoint500"]["cases"]
    stratum_columns = [
        column
        for column in (
            "active_module_count",
            "module_count_stratum",
            "spacing_stratum",
            "wall_proximity_stratum",
            "heating_heterogeneity_stratum",
        )
        if column in candidate_frame
    ]
    for base, column, category, label in [
        spec for spec in pair_specs if spec[1].endswith("_l2") or spec[1].endswith("_relative_l2")
    ]:
        ranked = candidate_frame[["case_id", column, *stratum_columns]].sort_values(column, ascending=False).head(10)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            difficult_rows.append(
                {
                    "selection": "top_10_candidate_error",
                    "metric": base,
                    "metric_label": label,
                    "category": category,
                    "rank": rank,
                    "case_id": str(row["case_id"]),
                    "candidate_value": float(row[column]),
                    **{key: row[key] for key in stratum_columns},
                }
            )

    candidate_strata = views["candidate_endpoint500"]["strata"]
    stratum_keys = set(zip(candidate_strata["stratum_name"].astype(str), candidate_strata["stratum_value"].astype(str)))
    strata_rows: list[dict[str, Any]] = []
    strata_metric_columns = [
        "global_field_fluid_norm_l2_mean",
        "global_field_fluid_norm_l2_median",
        "global_field_fluid_norm_l2_p95",
        "global_field_fluid_norm_l2_max",
        "global_field_near_interface_norm_l2_mean",
        "global_field_near_interface_norm_l2_median",
        "global_field_near_interface_norm_l2_p95",
        "global_field_near_interface_norm_l2_max",
        "global_field_far_fluid_norm_l2_mean",
        "global_field_far_fluid_norm_l2_median",
        "global_field_far_fluid_norm_l2_p95",
        "global_field_far_fluid_norm_l2_max",
        *[f"field_{channel}_fluid_physical_mae_mean" for channel in ("u", "v", "p", "omega", "temperature")],
        "port_t_env_final_physical_mae_mean",
        "port_h_effective_final_physical_mae_mean",
        "internal_temperature_physical_mae_mean",
        "interface_t_surface_physical_mae_mean",
        "interface_q_normal_physical_mae_mean",
        "pressure_drop_inlet_minus_outlet_physical_abs_error_mean",
        "mean_outlet_temperature_physical_abs_error_mean",
        "mean_active_module_temperature_physical_abs_error_mean",
    ]
    for key, view in views.items():
        current = view["strata"].copy()
        current_keys = set(zip(current["stratum_name"].astype(str), current["stratum_value"].astype(str)))
        if current_keys != stratum_keys:
            raise ValueError(f"{key}: physical strata do not match the candidate")
        for _, row in current.iterrows():
            stratum_name = str(row["stratum_name"])
            stratum_value = str(row["stratum_value"])
            if stratum_name not in view["cases"]:
                raise ValueError(f"{key}: per-case table has no stratum column {stratum_name!r}")
            stratum_cases = view["cases"].loc[
                view["cases"][stratum_name].astype(str).eq(stratum_value)
            ]
            expected_stratum_cases = int(float(row["num_cases"]))
            if len(stratum_cases) != expected_stratum_cases:
                raise ValueError(
                    f"{key}/{stratum_name}={stratum_value}: expected {expected_stratum_cases} cases, "
                    f"found {len(stratum_cases)}"
                )
            stratum_sse = float(
                numeric(stratum_cases, "global_field_fluid_norm_sse", view_key=key).sum()
            )
            stratum_target = float(
                numeric(stratum_cases, "global_field_fluid_norm_target_sse", view_key=key).sum()
            )
            stratum_count = float(
                numeric(stratum_cases, "global_field_fluid_norm_num_values", view_key=key).sum()
            )
            if stratum_target <= 0.0 or stratum_count <= 0.0:
                raise ValueError(f"{key}/{stratum_name}={stratum_value}: invalid global denominators")
            output = {
                "dataset": key,
                "model_label": view["model_label"],
                "stratum_name": stratum_name,
                "stratum_value": stratum_value,
                "num_cases": expected_stratum_cases,
                "global_field_fluid_norm_pooled_relative_l2": math.sqrt(stratum_sse / stratum_target),
                "global_field_fluid_norm_equal_case_l2_mean": (
                    float(row["global_field_fluid_norm_l2_mean"])
                    if pd.notna(row.get("global_field_fluid_norm_l2_mean"))
                    else None
                ),
            }
            for column in strata_metric_columns:
                if column in current:
                    output[column] = float(row[column]) if pd.notna(row[column]) else None
            strata_rows.append(output)

    def write_csv(name: str, rows: list[dict[str, Any]]) -> str:
        path = reduction_dir / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    outputs = {
        "case_set_validation": write_csv("case_set_validation.csv", case_set_rows),
        "target_count_validation": write_csv("target_count_validation.csv", target_count_rows),
        "raw_pool_reconciliation": write_csv("raw_pool_reconciliation.csv", pool_rows),
        "metrics": write_csv("metrics_long.csv", metric_rows),
        "strata": write_csv("strata_metrics.csv", strata_rows),
        "paired_cases": write_csv("paired_case_deltas.csv", paired_rows),
        "paired_summary": write_csv("paired_summary.csv", paired_summary),
        "difficult_cases": write_csv("difficult_cases.csv", difficult_rows),
    }
    input_manifest = {
        "candidate_endpoint500_tables": str(table_dir(candidate_source)),
        "dense500_tables": str(table_dir(dense_source)),
        "legacy500_tables": str(table_dir(dense_source)),
        "reader500_tables": str(table_dir(reader_source)),
        "maturity5000_reference": None if maturity_source is None else str(maturity_source),
        "maturity5000_read": False,
        "expected_cases": expected_cases,
    }
    (reduction_dir / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n")
    outputs["input_manifest"] = str(reduction_dir / "input_manifest.json")
    reduction_json = reduction_dir / "reduction.json"
    outputs["reduction"] = str(reduction_json)
    payload = {
        "schema_version": 1,
        "task": "endpoint_reduction",
        "stage": "endpoint500",
        "candidate": {
            "dataset": "candidate_endpoint500",
            "model_label": views["candidate_endpoint500"]["model_label"],
        },
        "comparators": ["dense500", "legacy500", "reader500"],
        "inputs": input_manifest,
        "case_set_validation": case_set_rows,
        "target_count_validation": {
            "rows": len(target_count_rows),
            "passed": all(row["status"] == "pass" for row in target_count_rows),
            "metric_count": len(target_count_columns),
        },
        "raw_pool_reconciliation": {
            "rows": len(pool_rows),
            "passed": all(row["passed"] for row in pool_rows),
        },
        "metrics": metric_rows,
        "paired_summary": paired_summary,
        "outputs": outputs,
        "difficult_case_selection": "top ten measured candidate per-case errors for each relative-L2 metric; descriptive only",
        "interpretation": {
            "lower_is_better": True,
            "paired_delta": "candidate minus comparator; negative means candidate lower error",
            "pooled_relative_l2": "sqrt(sum raw SSE / sum raw target SSE)",
            "pooled_mean_absolute_error": (
                "sum(case MAE * case value count) / sum(case value count); "
                "equal_case_mean is the unweighted case-level mean"
            ),
            "equal_case": "case-level mean, median, p95, and maximum retained separately",
            "strata_global_field_fluid_norm": (
                "pooled relative L2 is recomputed from raw per-case SSE/target-SSE within each stratum; "
                "*_equal_case_l2_mean is retained separately"
            ),
        },
    }
    reduction_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=8192)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)

    interventions = subparsers.add_parser(
        "interventions", help="bounded mature Dense/Reader/native regional phase/component removals"
    )
    _add_common_arguments(interventions)
    interventions.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    interventions.set_defaults(handler=run_interventions)

    probes = subparsers.add_parser("probes", help="shared regional reads and bounded JVP/AD-FD probes")
    _add_common_arguments(probes)
    probes.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    probes.set_defaults(handler=run_regional_probes, query_count=32)

    frozen = subparsers.add_parser(
        "frozen", help="one fixed 192-to-48 projected-key/value coarsening study"
    )
    _add_common_arguments(frozen)
    frozen.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    frozen.add_argument(
        "--context-probe-count",
        type=int,
        default=256,
        help="maximum receiver vectors retained per backend read chunk for context deltas",
    )
    frozen.set_defaults(handler=run_frozen)

    smoke = subparsers.add_parser(
        "smoke", help="two disposable real physical-batch AdamW optimizer steps"
    )
    _add_common_arguments(smoke)
    smoke.add_argument(
        "--profile",
        default="project://src/config_core/forward/regional_response_interface_context.json",
    )
    smoke.add_argument("--points-per-case", type=int, default=1024)
    smoke.add_argument("--batch-size", type=int, default=48)
    smoke.add_argument("--case-count", type=int, default=48)
    smoke.add_argument("--small-case-id", action="append", default=None)
    smoke.add_argument("--large-case-id", action="append", default=None)
    smoke.set_defaults(handler=run_smoke)

    timing = subparsers.add_parser(
        "timing", help="real-anchor and explicit-even-grid operation/time/memory measurements"
    )
    _add_common_arguments(timing)
    timing.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    timing.add_argument("--shape", action="append", default=[], metavar="M,E,Q")
    timing.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    timing.add_argument("--receiver-chunk-size", type=int, default=2048)
    timing.add_argument("--warmup", type=int, default=2)
    timing.add_argument("--repetitions", type=int, default=5)
    timing.add_argument("--synthetic-warmup", type=int, default=1)
    timing.add_argument("--synthetic-repetitions", type=int, default=3)
    timing.add_argument(
        "--execution-test",
        action="store_true",
        help="label this bounded timing invocation as a helper execution test, not an endpoint benchmark",
    )
    timing.set_defaults(handler=run_timing_protocol)

    figures = subparsers.add_parser(
        "figures", help="compact figures from stored regional diagnostic artifacts"
    )
    figures.add_argument("--output", type=Path, required=True, help="JSON manifest output path")
    figures.add_argument("--figure-dir", type=Path, required=True, help="directory for PNG figures")
    figures.add_argument(
        "--interventions",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="stored interventions JSON; repeat for Dense/Reader tables",
    )
    figures.add_argument("--frozen", type=Path, default=None, help="stored frozen-coarsening JSON")
    figures.add_argument("--timing", type=Path, default=None, help="stored timing JSON")
    figures.set_defaults(handler=run_figures)

    endpoint_figures = subparsers.add_parser(
        "endpoint-figures", help="regional cover, shared reads, anchor errors, and measured accuracy/cost"
    )
    for option in (
        "native-endpoint", "dense500-endpoint", "reader500-endpoint", "figure-dir",
        "output", "native-history", "dense-history", "timing",
    ):
        endpoint_figures.add_argument(f"--{option}", type=Path, required=True)
    endpoint_figures.add_argument("--probe-artifact", type=Path)
    endpoint_figures.add_argument("--dense500-debug-dir", type=Path, action="append", required=True)
    endpoint_figures.add_argument("--dense500-debug-prefix", default="Dense_pairwise_adaptation__500__")
    endpoint_figures.add_argument("--native-selector")
    endpoint_figures.add_argument("--dense500-selector", default="Run_1804_")
    endpoint_figures.add_argument("--reader500-selector")
    endpoint_figures.set_defaults(handler=run_endpoint_figures)

    reduction = subparsers.add_parser(
        "reduce", help="matched endpoint metrics from existing comparison tables"
    )
    reduction.add_argument("--candidate-tables", type=Path, required=True)
    reduction.add_argument("--dense500-tables", type=Path, required=True)
    reduction.add_argument("--reader500-tables", type=Path, required=True)
    reduction.add_argument("--maturity5000-tables", type=Path)
    reduction.add_argument("--output-dir", type=Path, required=True)
    reduction.add_argument("--candidate-selector")
    reduction.add_argument("--dense500-selector", default="Run_1804_")
    reduction.add_argument("--legacy500-selector", default="Run_1401_")
    reduction.add_argument("--reader500-selector")
    reduction.set_defaults(handler=run_endpoint_reduction)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = args.handler(args)
    if getattr(args, "output", None) is not None:
        payload["output"] = str(Path(args.output).expanduser().resolve())
        write_json(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
