#!/usr/bin/env python3
"""Run the bounded, evaluation-only Stage-3 interface-operator study.

The study intentionally lives in one orchestrator so every result carries the
checkpoint label/path, the input case, and the exact diagnostic task.  It never
calls a training or optimizer API.  The ``requests`` task is also deliberately
model-only: physical reference requests are exported with an explicit
``reference_verification_pending`` status when no trusted CFD result is
available.

Examples (run from ``HONF_Proj/``)::

    python tools/diagnostics/run_stage3_interface_study.py interventions \
        --sparse-checkpoint sparse=/path/to/epoch_0500_model.pt \
        --case-id 0273 --case-id 0653 \
        --output diagnostics/generated/stage3_interventions.json

    python tools/diagnostics/run_stage3_interface_study.py quadrature \
        --checkpoint run1401=/path/1401.pt --checkpoint dense=/path/1804.pt \
        --checkpoint latent=/path/1801.pt --checkpoint sparse=/path/1802.pt \
        --output diagnostics/generated/stage3_quadrature.json

All checkpoints must be supplied as ``LABEL=PATH``.  The labels are retained
in the output and are never inferred from filenames.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.environment import ChannelThermalEnvironment
from channelthermal.evaluation.loading import load_model, make_batch
from channelthermal.evaluation.prepared import select_sample

from honf_forward_core.interface_fields.supports import lookup_receivers
from honf_runtime.compat import load_trusted_checkpoint, resolve_demo_path, select_device

STUDY_SCHEMA_VERSION = 1
DEFAULT_QUERY_BATCH_SIZE = 32768
DEFAULT_CASE_IDS = ("0273", "0653")
DEFAULT_SCALING_SHAPES = (
    (8, 192, 8192),
    (32, 768, 8192),
    (32, 768, 65536),
    (32, 768, 262144),
    (128, 3072, 262144),
)


@dataclass(frozen=True)
class CheckpointSpec:
    """One explicitly labelled checkpoint selected by the user."""

    label: str
    path: Path


def parse_checkpoint_spec(value: str) -> CheckpointSpec:
    """Parse the required ``LABEL=PATH`` checkpoint syntax."""

    raw = str(value).strip()
    if "=" not in raw:
        raise ValueError(f"Checkpoint must use LABEL=PATH syntax, got {value!r}.")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Checkpoint must use non-empty LABEL=PATH syntax, got {value!r}.")
    if any(character in label for character in "/\\"):
        raise ValueError(f"Checkpoint label must be a simple name, got {label!r}.")
    return CheckpointSpec(label=label, path=resolve_demo_path(path).resolve())


def parse_checkpoint_specs(values: Sequence[str]) -> list[CheckpointSpec]:
    """Parse, validate, and preserve explicitly supplied checkpoint labels."""

    specs = [parse_checkpoint_spec(value) for value in values]
    labels = [spec.label for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Checkpoint labels must be unique, got {labels!r}.")
    for spec in specs:
        if not spec.path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {spec.label!r}: {spec.path}")
    return specs


def parse_shape(value: str) -> tuple[int, int, int]:
    """Parse one synthetic shape written as ``M,E,Q``."""

    pieces = [piece.strip() for piece in str(value).split(",")]
    if len(pieces) != 3:
        raise ValueError(f"Synthetic shape must be M,E,Q, got {value!r}.")
    try:
        shape = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"Synthetic shape must contain integers, got {value!r}.") from exc
    if any(item <= 0 for item in shape):
        raise ValueError(f"Synthetic shape entries must be positive, got {shape!r}.")
    return shape  # type: ignore[return-value]


def _json_value(value: Any) -> Any:
    """Convert tensor/NumPy/scalar values to JSON-safe values."""

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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one diagnostic payload, creating only the requested output tree."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_sample(value: Any) -> Any:
    """Deep-copy dataset samples without sharing mutable arrays."""

    if isinstance(value, np.ndarray):
        return value.copy()
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _copy_sample(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_sample(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_sample(item) for item in value)
    return copy.deepcopy(value)


def _sample_case_index(dataset: GlobalChannelThermalDataset, case_id: str) -> int:
    for index, candidate in enumerate(dataset.selected_case_ids):
        if str(candidate) == str(case_id):
            return int(index)
    raise KeyError(f"case_id={case_id!r} is absent from split={dataset.split!r}.")


def _load_dataset(checkpoint: Mapping[str, Any], args: argparse.Namespace) -> tuple[GlobalChannelThermalDataset, Path]:
    dataset_cfg = dict(checkpoint.get("train_config", {}).get("dataset", {}))
    raw_path = args.dataset or dataset_cfg.get("packed_h5_path")
    if raw_path is None:
        raise ValueError("Checkpoint does not declare a packed dataset; pass --dataset explicitly.")
    dataset_path = resolve_demo_path(raw_path).resolve()
    stats = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    normalizer = H5Normalizer(stats) if stats else None
    dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=str(args.split),
        points_per_case=1,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=False,
        normalizer=normalizer,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No cases available in dataset={dataset_path} split={args.split!r}.")
    return dataset, dataset_path


def _load_model_spec(spec: CheckpointSpec, device: torch.device) -> tuple[Any, dict[str, Any]]:
    checkpoint = load_trusted_checkpoint(spec.path, map_location="cpu")
    model, loaded_checkpoint = load_model(spec.path, device)
    # The workflow loader returns the same trusted payload after strict model
    # reconstruction.  Keep the explicit payload available for provenance.
    del loaded_checkpoint
    model.eval()
    return model, dict(checkpoint)


def _checkpoint_record(spec: CheckpointSpec, checkpoint: Mapping[str, Any], model: Any) -> dict[str, Any]:
    config = checkpoint.get("model_config", {})
    core = config.get("core_honf", {}) if isinstance(config, Mapping) else {}
    model_core = getattr(getattr(model, "config", None), "core_honf", None)
    return {
        "label": spec.label,
        "path": str(spec.path),
        "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "forward_architecture": str(getattr(model_core, "forward_architecture", "unknown")),
        "config_forward_architecture": core.get("forward_architecture"),
    }


def _query_points(sample: Mapping[str, Any], count: int) -> np.ndarray:
    """Choose deterministic fixed physical probes from a case grid."""

    x = np.asarray(sample["x_grid"], dtype=np.float32).reshape(-1)
    y = np.asarray(sample["y_grid"], dtype=np.float32).reshape(-1)
    points = np.stack((x, y), axis=-1)
    if count <= 0:
        raise ValueError("query count must be positive")
    if len(points) <= count:
        return points
    indices = np.linspace(0, len(points) - 1, int(count), dtype=np.int64)
    return points[indices]


def _forward_batch(model: Any, sample: Mapping[str, Any], query_xy: np.ndarray | torch.Tensor, device: torch.device, *, return_prepared_state: bool = False, return_routing_maps: bool = False) -> dict[str, Any]:
    """Run one direct, evaluation-only model forward from an adapted sample."""

    if torch.is_tensor(query_xy):
        query_array = query_xy.detach().cpu().numpy().astype(np.float32)
    else:
        query_array = np.asarray(query_xy, dtype=np.float32)
    batch = make_batch(dict(sample), query_array, device)
    query_tensor = batch["query_xy"]
    outputs = model(
        batch["structure"],
        query_tensor,
        interface_condition=batch.get("interface_condition"),
        local_module_params=batch.get("local_module_params"),
        teacher_port_tokens=batch.get("teacher_port_tokens"),
        local_query_points=batch.get("module_internal_query_points"),
        local_port_condition_mode="predicted",
        return_prepared_state=bool(return_prepared_state),
        return_routing_maps=bool(return_routing_maps),
    )
    return outputs


def _forward_tensor_batch(model: Any, sample: Mapping[str, Any], query_xy: torch.Tensor, device: torch.device, centers: torch.Tensor | None = None, *, return_prepared_state: bool = False, return_routing_maps: bool = False) -> dict[str, Any]:
    """Direct forward preserving a differentiable module-center tensor."""

    query_array = query_xy.detach().cpu().numpy().astype(np.float32)
    if query_array.ndim == 3 and query_array.shape[0] == 1:
        query_array = query_array[0]
    batch = make_batch(dict(sample), query_array, device)
    if centers is not None:
        batch["structure"]["module_centers"] = centers
    return model(
        batch["structure"],
        batch["query_xy"],
        interface_condition=batch.get("interface_condition"),
        local_module_params=batch.get("local_module_params"),
        teacher_port_tokens=batch.get("teacher_port_tokens"),
        local_query_points=batch.get("module_internal_query_points"),
        local_port_condition_mode="predicted",
        return_prepared_state=bool(return_prepared_state),
        return_routing_maps=bool(return_routing_maps),
    )


def _field_kpis(field: torch.Tensor, query_xy: torch.Tensor, field_names: Sequence[str]) -> dict[str, torch.Tensor]:
    """Compute fixed-probe scalar KPIs from one field tensor."""

    names = list(field_names)
    result: dict[str, torch.Tensor] = {}
    if "temperature" in names:
        result["mean_temperature"] = field[..., names.index("temperature")].mean()
    if "p" in names:
        pressure = field[..., names.index("p")]
        x = query_xy[..., 0]
        left = pressure[x <= torch.quantile(x.detach(), 0.25)]
        right = pressure[x >= torch.quantile(x.detach(), 0.75)]
        if left.numel() and right.numel():
            result["pressure_drop"] = left.mean() - right.mean()
        result["mean_pressure"] = pressure.mean()
    if not result:
        result["field_l2"] = torch.linalg.vector_norm(field) / max(float(field.numel()), 1.0) ** 0.5
    return result


def _kpi_values(outputs: Mapping[str, Any], query_xy: torch.Tensor, model: Any) -> dict[str, float]:
    field = outputs["pred_field"]
    values = _field_kpis(field, query_xy, list(model.config.channelthermal.field_names))
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def _relative_difference(lhs: Any, rhs: Any) -> dict[str, float]:
    left = lhs.detach().float() if torch.is_tensor(lhs) else torch.as_tensor(lhs, dtype=torch.float32)
    right = rhs.detach().float() if torch.is_tensor(rhs) else torch.as_tensor(rhs, dtype=torch.float32)
    diff = (left - right).abs()
    scale = right.abs().clamp_min(1.0e-12)
    return {
        "max_abs": float(diff.max().cpu()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().cpu()) if diff.numel() else 0.0,
        "max_relative": float((diff / scale).max().cpu()) if diff.numel() else 0.0,
    }


def _architecture(model: Any) -> str:
    return str(model.config.core_honf.forward_architecture)


def _active_centers(sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    structure = sample["structure"]
    centers = np.asarray(structure["module_centers"], dtype=np.float64).copy()
    present = np.asarray(structure["module_present"], dtype=np.float64).reshape(-1) > 0.5
    return centers, present


def _domain_lengths(sample: Mapping[str, Any], model: Any) -> tuple[float, float]:
    structure = sample["structure"]
    lx = float(np.asarray(structure.get("domain_length_x", [model.config.core_honf.domain_length_x])).reshape(-1)[0])
    ly = float(np.asarray(structure.get("domain_length_y", [model.config.core_honf.domain_length_y])).reshape(-1)[0])
    return lx, ly


def _module_radius(model: Any) -> float:
    return float(model.config.core_honf.module_radius)


def _valid_geometry(centers: np.ndarray, present: np.ndarray, radius: float, lx: float, ly: float) -> bool:
    active = centers[present]
    if active.size == 0:
        return False
    if np.any(active[:, 0] < radius) or np.any(active[:, 0] > lx - radius):
        return False
    if np.any(active[:, 1] < radius) or np.any(active[:, 1] > ly - radius):
        return False
    if len(active) > 1:
        distances = np.linalg.norm(active[:, None, :] - active[None, :, :], axis=-1)
        distances += np.eye(len(active)) * 1.0e9
        if float(distances.min()) < 2.0 * radius - 1.0e-6:
            return False
    return True


def perturb_sample(sample: Mapping[str, Any], model: Any, module_index: int, direction: Sequence[float], scale: float) -> dict[str, Any]:
    """Return a geometry-only perturbation with a copied module-center array."""

    result = _copy_sample(sample)
    centers, present = _active_centers(result)
    if module_index < 0 or module_index >= len(centers) or not present[module_index]:
        raise ValueError(f"module_index={module_index} is not an active module")
    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    if vector.shape != (2,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0.0:
        raise ValueError("direction must be a finite nonzero 2-vector")
    vector = vector / np.linalg.norm(vector)
    centers[module_index] += vector * float(scale)
    lx, ly = _domain_lengths(result, model)
    if not _valid_geometry(centers, present, _module_radius(model), lx, ly):
        raise ValueError("geometry perturbation violates module bounds or nonintersection")
    result["structure"]["module_centers"] = centers.astype(np.float32)
    return result


def _patch_instance(obj: Any, name: str, replacement: Callable[..., Any]) -> tuple[Any, Any]:
    original = getattr(obj, name)
    setattr(obj, name, replacement)
    return original, obj


@contextmanager
def sparse_context_intervention(model: Any, mode: str) -> Iterator[None]:
    """Temporarily remove one sparse branch without changing model parameters."""

    if _architecture(model) != "sparse_interface_honf":
        raise ValueError(f"This intervention requires sparse_interface_honf, got {_architecture(model)!r}.")
    if mode not in {"port_main_zero", "field_main_zero"}:
        raise ValueError(f"Unknown sparse intervention mode={mode!r}.")
    core = model.core
    backend = core.backend
    state = {"port": False, "decode": False}
    original_backend_read = backend.read

    def backend_read(*args: Any, **kwargs: Any) -> Any:
        main, aux = original_backend_read(*args, **kwargs)
        if (mode == "port_main_zero" and state["port"]) or (mode == "field_main_zero" and state["decode"]):
            main = torch.zeros_like(main)
        return main, aux

    original_backend = _patch_instance(backend, "read", backend_read)[0]
    coupling_module = sys.modules["channelthermal.interface_field_coupling"]
    original_port_context = coupling_module._read_port_context

    def port_context(*args: Any, **kwargs: Any) -> Any:
        state["port"] = True
        try:
            return original_port_context(*args, **kwargs)
        finally:
            state["port"] = False

    original_port_attr = _patch_instance(coupling_module, "_read_port_context", port_context)[0]
    original_decode = core.decode_queries

    def decode_queries(*args: Any, **kwargs: Any) -> Any:
        state["decode"] = True
        try:
            return original_decode(*args, **kwargs)
        finally:
            state["decode"] = False

    original_decode_attr = _patch_instance(core, "decode_queries", decode_queries)[0]
    try:
        yield
    finally:
        backend.read = original_backend
        coupling_module._read_port_context = original_port_attr
        core.decode_queries = original_decode_attr


@contextmanager
def capture_or_replay_coarse_context(
    model: Any,
    captured: list[torch.Tensor] | None = None,
) -> Iterator[list[torch.Tensor]]:
    """Capture a base coarse-read sequence or replay it for a perturbation.

    The physical wrapper makes the same ordered coarse reads at P0/P1/P2 for
    a fixed module/port/query shape. Replaying the unperturbed tensors holds
    that route fixed while geometry, sparse layout, group states, local
    responses, and local reads are rebuilt normally.
    """

    original = model.core.common.read_coarse
    values: list[torch.Tensor] = [] if captured is None else captured
    index = 0

    def read_coarse(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal index
        current = original(*args, **kwargs)
        if captured is None:
            values.append(current.detach().clone())
            return current
        if index >= len(values):
            raise RuntimeError("perturbed forward issued more coarse reads than the base forward")
        fixed = values[index]
        index += 1
        if fixed.shape != current.shape:
            raise RuntimeError(
                f"coarse replay shape mismatch: base={tuple(fixed.shape)}, perturbed={tuple(current.shape)}"
            )
        return fixed.to(device=current.device, dtype=current.dtype)

    model.core.common.read_coarse = read_coarse
    try:
        yield values
    finally:
        model.core.common.read_coarse = original
        if captured is not None and index != len(values):
            raise RuntimeError(
                f"perturbed forward issued {index} coarse reads; base issued {len(values)}"
            )


def _state_keys(model: Any) -> tuple[str, ...]:
    return tuple(model.state_dict().keys())


def _summarize_outputs(base: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("pred_field", "pred_interface", "pred_port_condition", "pred_internal_temperature"):
        if key in base and key in variant and torch.is_tensor(base[key]) and torch.is_tensor(variant[key]):
            summary[key] = _relative_difference(variant[key], base[key])
    return summary


def run_interventions(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_checkpoint_specs(args.sparse_checkpoint or args.checkpoint)
    if len(specs) != 1:
        raise ValueError("interventions requires exactly one sparse checkpoint")
    device = select_device(args.device)
    spec = specs[0]
    model, checkpoint = _load_model_spec(spec, device)
    dataset, dataset_path = _load_dataset(checkpoint, args)
    case_ids = list(args.case_id or dataset.selected_case_ids[:4])[:4]
    state_before = _state_keys(model)
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        sample = select_sample(dataset, str(case_id), 0)
        queries_np = _query_points(sample, int(args.query_count))
        queries = torch.from_numpy(queries_np).unsqueeze(0).to(device)
        with torch.no_grad():
            base = _forward_batch(model, sample, queries_np, device, return_routing_maps=True)
        base_kpis = _kpi_values(base, queries, model)
        variants: dict[str, dict[str, Any]] = {}
        with sparse_context_intervention(model, "port_main_zero"), torch.no_grad():
            variants["port_main_zero"] = _forward_batch(model, sample, queries_np, device, return_routing_maps=True)
        with sparse_context_intervention(model, "field_main_zero"), torch.no_grad():
            variants["field_main_zero"] = _forward_batch(model, sample, queries_np, device, return_routing_maps=True)
        module_index = int(np.flatnonzero(_active_centers(sample)[1])[0])
        perturbation_scale = float(args.perturbation_scale) * _module_radius(model)
        direction = _direction_for_sample(sample, model, module_index, perturbation_scale)
        perturbed = perturb_sample(sample, model, module_index, direction, perturbation_scale)
        with capture_or_replay_coarse_context(model) as base_coarse, torch.no_grad():
            _forward_batch(model, sample, queries_np, device, return_routing_maps=True)
        with torch.no_grad():
            geometry_base = _forward_batch(model, perturbed, queries_np, device, return_routing_maps=True)
        with capture_or_replay_coarse_context(model, base_coarse), torch.no_grad():
            variants["coarse_clamped_geometry_perturbation"] = _forward_batch(
                model, perturbed, queries_np, device, return_routing_maps=True
            )
        row = {
            "case_id": str(sample["case_id"]),
            "base_kpis": base_kpis,
            "interventions": {},
            "geometry_perturbation": {
                "module_index": module_index,
                "scale_in_module_radii": float(args.perturbation_scale),
                "direction": direction.tolist(),
                "coarse_clamped": True,
            },
        }
        row["interventions"]["port_main_zero"] = _summarize_outputs(base, variants["port_main_zero"])
        row["interventions"]["field_main_zero"] = _summarize_outputs(base, variants["field_main_zero"])
        row["interventions"]["coarse_clamped_geometry_perturbation"] = _summarize_outputs(
            geometry_base, variants["coarse_clamped_geometry_perturbation"]
        )
        for name, variant in variants.items():
            row["interventions"][name]["variant_kpis"] = _kpi_values(
                variant,
                queries,
                model,
            )
        rows.append(row)
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "task": "interventions",
        "checkpoints": [_checkpoint_record(spec, checkpoint, model)],
        "dataset": str(dataset_path),
        "results": rows,
        "limitations": [
            "Branch interventions are evaluation-only reliance diagnostics, not retraining ablations.",
            "The coarse-clamped geometry result is a model perturbation; it is not a physical reference solve.",
        ],
        "state_dict_structure_unchanged": state_before == _state_keys(model),
    }


def duplicate_environment_bundle(environment: ChannelThermalEnvironment, factor: int = 2) -> ChannelThermalEnvironment:
    """Duplicate environment samples while dividing their implicit volume."""

    factor = int(factor)
    if factor < 2:
        raise ValueError("duplication factor must be at least 2")
    return ChannelThermalEnvironment(
        env_coords=torch.cat([environment.env_coords] * factor, dim=1),
        env_features=torch.cat([environment.env_features] * factor, dim=1),
    )


@contextmanager
def duplicated_environment(model: Any, factor: int = 2) -> Iterator[None]:
    """Temporarily duplicate the case adapter's environment quadrature."""

    original = model.environment_builder

    class DuplicatedEnvironmentBuilder:
        """Callable proxy that preserves query-feature generation."""

        def __call__(self, **kwargs: Any) -> ChannelThermalEnvironment:
            return duplicate_environment_bundle(original(**kwargs), factor=factor)

        def query_features(self, *args: Any, **kwargs: Any) -> Any:
            return original.query_features(*args, **kwargs)

    model.environment_builder = DuplicatedEnvironmentBuilder()
    try:
        yield
    finally:
        model.environment_builder = original


def _output_tensors(outputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in outputs.items()
        if key in {"pred_field", "pred_interface", "pred_port_condition", "pred_internal_temperature"}
        and torch.is_tensor(value)
    }


def run_quadrature(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 4:
        raise ValueError("quadrature requires exactly four explicitly labelled checkpoints")
    device = select_device(args.device)
    case_id = str((args.case_id or list(DEFAULT_CASE_IDS))[0])
    model_records: list[dict[str, Any]] = []
    for spec in specs:
        model, checkpoint = _load_model_spec(spec, device)
        dataset, dataset_path = _load_dataset(checkpoint, args)
        sample = select_sample(dataset, case_id, 0)
        query_np = _query_points(sample, int(args.query_count))
        state_before = _state_keys(model)
        with torch.no_grad():
            baseline = _forward_batch(model, sample, query_np, device)
        with duplicated_environment(model, int(args.duplication_factor)), torch.no_grad():
            duplicated = _forward_batch(model, sample, query_np, device)
        differences = {
            key: _relative_difference(duplicated[key], baseline[key])
            for key in _output_tensors(baseline)
            if key in duplicated and torch.is_tensor(duplicated[key])
        }
        model_records.append(
            {
                "checkpoint": _checkpoint_record(spec, checkpoint, model),
                "architecture": _architecture(model),
                "dataset": str(dataset_path),
                "case_id": str(sample["case_id"]),
                "baseline_environment_token_count": int(
                    int(model.config.core_honf.num_env_tokens_x) * int(model.config.core_honf.num_env_tokens_y)
                ),
                "duplicated_environment_token_count": int(
                    int(model.config.core_honf.num_env_tokens_x)
                    * int(model.config.core_honf.num_env_tokens_y)
                    * int(args.duplication_factor)
                ),
                "differences": differences,
                "state_dict_structure_unchanged": state_before == _state_keys(model),
            }
        )
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "task": "quadrature",
        "case_id": case_id,
        "duplication_factor": int(args.duplication_factor),
        "results": model_records,
        "limitations": [
            "Exact duplicate splitting tests the model's volume-weighted quadrature arithmetic; it is not a resolution study.",
            "The comparison is in the model's evaluation space and does not validate CFD physics.",
        ],
    }


@contextmanager
def temporary_domain(model: Any, lx: float, ly: float) -> Iterator[None]:
    """Temporarily update domain scales for execution-only synthetic layouts."""

    candidates = [getattr(model.config, "core_honf", None), getattr(getattr(model, "core", None), "config", None)]
    saved: list[tuple[Any, str, Any]] = []
    seen: set[int] = set()
    for config in candidates:
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        for name, value in (
            ("domain_length_x", float(lx)),
            ("domain_length_y", float(ly)),
            ("coordinate_scale", [float(lx), float(ly)]),
        ):
            if hasattr(config, name):
                saved.append((config, name, getattr(config, name)))
                setattr(config, name, value)
    try:
        yield
    finally:
        for config, name, value in reversed(saved):
            setattr(config, name, value)


def _synthetic_centers(module_count: int, lx: float, ly: float, radius: float) -> np.ndarray:
    columns = max(1, math.ceil(math.sqrt(float(module_count) * lx / ly)))
    rows = max(1, math.ceil(float(module_count) / columns))
    xs = (np.arange(columns, dtype=np.float64) + 0.5) * lx / columns
    ys = (np.arange(rows, dtype=np.float64) + 0.5) * ly / rows
    centers = np.asarray(list(itertools.product(xs, ys))[:module_count], dtype=np.float32)
    if not _valid_geometry(centers, np.ones(module_count, dtype=bool), radius, lx, ly):
        raise RuntimeError(f"Could not build a valid nonoverlapping synthetic layout for M={module_count}.")
    return centers


def _synthetic_environment(E: int, lx: float, ly: float, batch_size: int, device: torch.device, dtype: torch.dtype) -> ChannelThermalEnvironment:
    columns = max(1, math.ceil(math.sqrt(float(E) * lx / ly)))
    rows = max(1, math.ceil(float(E) / columns))
    xs = (torch.arange(columns, device=device, dtype=dtype) + 0.5) / float(columns) * float(lx)
    ys = (torch.arange(rows, device=device, dtype=dtype) + 0.5) / float(rows) * float(ly)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)[:E]
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    features = torch.cat(
        [x / lx, y / ly, y / ly, (ly - y) / ly, x / lx, (lx - x) / lx, (1.0 - (y - 0.5 * ly).abs() / max(0.5 * ly, 1.0e-6)).clamp(0.0, 1.0)],
        dim=-1,
    )
    return ChannelThermalEnvironment(
        env_coords=coords.unsqueeze(0).expand(batch_size, -1, -1),
        env_features=features.unsqueeze(0).expand(batch_size, -1, -1),
    )


@contextmanager
def synthetic_environment(model: Any, E: int, lx: float, ly: float) -> Iterator[None]:
    original = model.environment_builder

    class SyntheticEnvironmentBuilder:
        def __call__(self, *, batch_size: int, device: torch.device, dtype: torch.dtype, **kwargs: Any) -> ChannelThermalEnvironment:
            del kwargs
            return _synthetic_environment(E, lx, ly, batch_size, device, dtype)

        def query_features(self, *args: Any, **kwargs: Any) -> Any:
            return original.query_features(*args, **kwargs)

    model.environment_builder = SyntheticEnvironmentBuilder()
    try:
        yield
    finally:
        model.environment_builder = original


def _support_counts(prepared: Any, routing_aux: Mapping[str, Any] | None = None) -> dict[str, float | int]:
    """Report actual sparse support/incidence/read counts when available."""

    aux: Mapping[str, Any] = {}
    if hasattr(prepared, "prepared"):
        aux = getattr(prepared.prepared, "interaction_aux", {})
    elif isinstance(prepared, Mapping):
        aux = prepared.get("interaction_aux", {})
    routing = routing_aux or {}

    def count_shape(key: str, axis: int | None = None) -> int:
        value = aux.get(key)
        if value is None:
            value = routing.get(key)
        if value is None:
            return 0
        shape = tuple(int(item) for item in value.shape) if torch.is_tensor(value) else np.asarray(value).shape
        if axis is None:
            return int(np.prod(shape)) if shape else 1
        return int(shape[axis]) if len(shape) > axis else 0

    counts: dict[str, float | int] = {
        "support_group_count": count_shape("support_centres", 0),
        "module_group_incidence_count": count_shape("module_group_indices", 1),
        "environment_group_incidence_count": count_shape("environment_group_indices", 1),
        "query_group_read_value_count": count_shape("group_read_group_index"),
    }
    degree = routing.get("group_read_degree", aux.get("group_read_degree"))
    if degree is not None:
        tensor = degree.detach() if torch.is_tensor(degree) else torch.as_tensor(degree)
        counts["query_group_read_degree_mean"] = float(tensor.float().mean().cpu())
        counts["query_group_read_degree_max"] = float(tensor.float().max().cpu())
    return counts


def _scaled_domain(module_count: int) -> tuple[float, float]:
    scale = math.sqrt(float(module_count) / 8.0)
    return 12.0 * scale, 6.0 * scale


def _synthetic_query_tensor(Q: int, lx: float, ly: float, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(1701 + int(Q))
    return torch.rand(1, int(Q), 2, generator=generator, device=device) * torch.tensor([lx, ly], device=device)


def _prepared_synthetic_forward(model: Any, structure: Mapping[str, torch.Tensor], query_xy: torch.Tensor, device: torch.device, *, query_batch_size: int) -> tuple[torch.Tensor, Any, dict[str, Any]]:
    first = model(
        structure,
        query_xy[:, :1],
        local_port_condition_mode="predicted",
        return_prepared_state=True,
        return_routing_maps=True,
    )
    prepared = first.pop("prepared_state")
    outputs = []
    support = _support_counts(prepared, first.get("routing_aux", {}))
    # The one-query call above exists only to build the physical prepared
    # state. Scaling read counts cover the requested Q-query decode below.
    support["query_group_read_value_count"] = 0.0
    for start in range(0, int(query_xy.shape[1]), int(query_batch_size)):
        chunk = query_xy[:, start : start + int(query_batch_size)]
        decoded = model.decode_prepared(prepared, chunk, return_routing_maps=True)
        outputs.append(decoded["pred_field"])
        support_chunk = _support_counts(prepared, decoded)
        for key, value in support_chunk.items():
            if key == "query_group_read_value_count":
                support[key] = float(support.get(key, 0.0)) + float(value)
            elif key.endswith(("count", "value_count")):
                support[key] = max(float(support.get(key, 0.0)), float(value))
            elif key.endswith("mean"):
                support[key] = float(value)
            elif key.endswith("max"):
                support[key] = max(float(support.get(key, 0.0)), float(value))
    return torch.cat(outputs, dim=1), prepared, {"support": support}


def run_scaling(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 3:
        raise ValueError("scaling requires exactly three explicitly labelled new-family checkpoints")
    shapes = [parse_shape(value) for value in args.shape] if args.shape else list(DEFAULT_SCALING_SHAPES)
    device = select_device(args.device)
    model_records: list[dict[str, Any]] = []
    for spec in specs:
        model, checkpoint = _load_model_spec(spec, device)
        architecture = _architecture(model)
        shape_rows: list[dict[str, Any]] = []
        state_before = _state_keys(model)
        for module_count, env_count, query_count in shapes:
            lx, ly = _scaled_domain(module_count)
            radius = _module_radius(model)
            centers = _synthetic_centers(module_count, lx, ly, radius)
            structure = {
                "re": torch.full((1, 1), 50.0, device=device),
                "u_in": torch.ones((1, 1), device=device),
                "module_centers": torch.from_numpy(centers).unsqueeze(0).to(device),
                "heat_powers": torch.ones((1, module_count), device=device),
                "module_present": torch.ones((1, module_count), device=device),
                "material_params": torch.zeros((1, int(model.config.channelthermal.material_param_dim)), device=device),
                "domain_length_x": torch.full((1, 1), lx, device=device),
                "domain_length_y": torch.full((1, 1), ly, device=device),
            }
            queries = _synthetic_query_tensor(query_count, lx, ly, device)
            row: dict[str, Any] = {
                "shape": {"M": int(module_count), "E": int(env_count), "Q": int(query_count)},
                "architecture": architecture,
                "domain": {"length_x": lx, "length_y": ly, "area": lx * ly},
                "status": "ok",
            }
            try:
                with temporary_domain(model, lx, ly), synthetic_environment(model, env_count, lx, ly):
                    for _ in range(max(0, int(args.warmup))):
                        with torch.no_grad():
                            _prepared_synthetic_forward(model, structure, queries[:, : min(query_count, int(args.query_batch_size))], device, query_batch_size=int(args.query_batch_size))
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    timings: list[float] = []
                    prediction = None
                    support = None
                    for _ in range(max(1, int(args.repetitions))):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        with torch.no_grad():
                            prediction, _prepared, extras = _prepared_synthetic_forward(model, structure, queries, device, query_batch_size=int(args.query_batch_size))
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        timings.append(time.perf_counter() - started)
                        support = extras["support"]
                    row["median_seconds"] = float(np.median(timings))
                    row["p95_seconds"] = float(np.quantile(timings, 0.95))
                    row["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
                    row["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
                    row["support_counts"] = support or {}
                    if query_count <= int(args.chunk_equality_limit):
                        with torch.no_grad():
                            all_at_once, _, _ = _prepared_synthetic_forward(model, structure, queries, device, query_batch_size=query_count)
                        row["small_chunk_equality"] = _relative_difference(prediction, all_at_once)
                    else:
                        row["small_chunk_equality"] = None
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    row["status"] = "out_of_memory"
                else:
                    row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc!s}"
            except Exception as exc:  # noqa: BLE001 - preserve per-shape diagnostics
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc!s}"
            shape_rows.append(row)
        model_records.append(
            {
                "checkpoint": _checkpoint_record(spec, checkpoint, model),
                "architecture": architecture,
                "results": shape_rows,
                "state_dict_structure_unchanged": state_before == _state_keys(model),
            }
        )
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "task": "scaling",
        "shapes": [{"M": m, "E": e, "Q": q} for m, e, q in shapes],
        "warmup_repetitions": int(args.warmup),
        "measured_repetitions": int(args.repetitions),
        "results": model_records,
        "limitations": [
            "Synthetic layouts are execution-only and make no physical-accuracy claim.",
            "Support counts are observed tensors/read incidences, not FLOP estimates.",
        ],
    }


@contextmanager
def fine_path_only(model: Any) -> Iterator[None]:
    """Keep only the backend fine context in an interface-field decode."""

    if _architecture(model) != "sparse_interface_honf":
        raise ValueError("fine-path isolation requires sparse_interface_honf")
    common = model.core.common
    original_coarse = common.read_coarse
    original_local = common.read_local

    def zero_coarse(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.zeros_like(original_coarse(*args, **kwargs))

    def zero_local(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        local, counts = original_local(*args, **kwargs)
        return torch.zeros_like(local), counts

    common.read_coarse = zero_coarse
    common.read_local = zero_local
    try:
        yield
    finally:
        common.read_coarse = original_coarse
        common.read_local = original_local


def _direction_for_sample(sample: Mapping[str, Any], model: Any, module_index: int, scale: float) -> np.ndarray:
    centers, present = _active_centers(sample)
    lx, ly = _domain_lengths(sample, model)
    directions = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    for direction in directions:
        candidate = centers.copy()
        vector = np.asarray(direction, dtype=np.float64)
        candidate[module_index] += vector * float(scale)
        if _valid_geometry(candidate, present, _module_radius(model), lx, ly):
            return vector
    raise ValueError(f"No valid {scale:g}-length direction for module {module_index}.")


def _align_one_port_to_support_transition(
    sample: Mapping[str, Any],
    model: Any,
    max_step: float,
) -> tuple[dict[str, Any], int, np.ndarray, dict[str, Any]]:
    """Return a valid layout with one physical port on a lattice boundary."""

    result = copy.deepcopy(dict(sample))
    centers, present = _active_centers(sample)
    radius = _module_radius(model)
    spacing = radius * float(model.core.backend.support_spacing_factor)
    lx, ly = _domain_lengths(sample, model)
    tokens = model.local_coupling.port_head.fixed_theta_tokens(
        64, torch.device("cpu"), torch.float32
    )
    normals = tokens[:, 1:3].detach().cpu().numpy().astype(np.float64)
    offsets = np.asarray(list(itertools.product((-1, 0, 1, 2), repeat=2)), dtype=np.int64)

    def module_keys(center: np.ndarray) -> set[tuple[int, int]]:
        bases = np.floor((center[None, :] + radius * normals) / spacing).astype(np.int64)
        return {
            tuple(int(value) for value in key)
            for base in bases
            for key in base[None, :] + offsets
        }

    candidates: list[tuple[float, int, int, int, float, int]] = []
    for module_index in np.flatnonzero(present):
        ports = centers[module_index] + radius * normals
        for dimension in (0, 1):
            for port_index, coordinate in enumerate(ports[:, dimension]):
                target = round(float(coordinate) / spacing) * spacing
                shift = target - float(coordinate)
                if abs(shift) <= 1.0e-7:
                    shift = 0.0
                candidate_centers = centers.copy()
                candidate_centers[module_index, dimension] += shift
                plus = candidate_centers.copy()
                minus = candidate_centers.copy()
                plus[module_index, dimension] += max_step
                minus[module_index, dimension] -= max_step
                if (
                    _valid_geometry(candidate_centers, present, radius, lx, ly)
                    and _valid_geometry(plus, present, radius, lx, ly)
                    and _valid_geometry(minus, present, radius, lx, ly)
                ):
                    difference = len(
                        module_keys(plus[module_index]).symmetric_difference(
                            module_keys(minus[module_index]
                        ))
                    )
                    if difference:
                        candidates.append(
                            (abs(shift), int(module_index), dimension, port_index, shift, difference)
                        )
    if not candidates:
        raise ValueError("no valid module-port support transition could be constructed")
    _, module_index, dimension, port_index, shift, expected_difference = min(candidates)
    centers[module_index, dimension] += shift
    result["structure"] = copy.deepcopy(dict(sample["structure"]))
    result["structure"]["module_centers"] = centers.astype(np.float32)
    direction = np.zeros(2, dtype=np.float64)
    direction[dimension] = 1.0
    return result, module_index, direction, {
        "constructed": True,
        "aligned_port_index": int(port_index),
        "dimension": "x" if dimension == 0 else "y",
        "alignment_shift": float(shift),
        "support_spacing": float(spacing),
        "expected_module_key_difference_count": int(expected_difference),
    }


def _support_key_set_for_centers(
    model: Any,
    sample: Mapping[str, Any],
    query: torch.Tensor,
    centers: np.ndarray,
    module_index: int,
    device: torch.device,
) -> set[tuple[int, ...]]:
    tensor = torch.from_numpy(centers.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = _forward_tensor_batch(
            model, sample, query[:, :1], device, centers=tensor, return_prepared_state=True
        )
    layout = outputs["prepared_state"].prepared.backend_state.cache.layout
    source, group = layout.module_group_indices
    module_groups = torch.unique(group[source == int(module_index)])
    keys = layout.lattice_keys.index_select(0, module_groups)
    return {tuple(int(value) for value in row) for row in keys.detach().cpu().tolist()}


def _scalar_output(outputs: Mapping[str, Any], query_xy: torch.Tensor, model: Any) -> torch.Tensor:
    values = _field_kpis(outputs["pred_field"], query_xy, list(model.config.channelthermal.field_names))
    return values["mean_temperature"] if "mean_temperature" in values else next(iter(values.values()))


def _full_scalar_for_centers(model: Any, sample: Mapping[str, Any], query_xy: torch.Tensor, centers: torch.Tensor, device: torch.device) -> torch.Tensor:
    outputs = _forward_tensor_batch(model, sample, query_xy, device, centers=centers, return_prepared_state=False)
    return _scalar_output(outputs, query_xy, model)


def _fine_scalar_for_centers(model: Any, sample: Mapping[str, Any], query_xy: torch.Tensor, centers: torch.Tensor, device: torch.device) -> torch.Tensor:
    outputs = _forward_tensor_batch(model, sample, query_xy, device, centers=centers, return_prepared_state=True, return_routing_maps=True)
    prepared = outputs["prepared_state"]
    with fine_path_only(model):
        decoded = model.decode_prepared(prepared, query_xy, return_routing_maps=True)
    return _scalar_output(decoded, query_xy, model)


def _autograd_directional(
    model: Any,
    sample: Mapping[str, Any],
    query_xy: torch.Tensor,
    module_index: int,
    direction: np.ndarray,
    device: torch.device,
    *,
    fine: bool,
) -> float:
    centers_np, _ = _active_centers(sample)
    centers = torch.from_numpy(centers_np.astype(np.float32)).unsqueeze(0).to(device).requires_grad_(True)
    scalar = (
        _fine_scalar_for_centers(model, sample, query_xy, centers, device)
        if fine
        else _full_scalar_for_centers(model, sample, query_xy, centers, device)
    )
    gradient = torch.autograd.grad(scalar, centers, allow_unused=True)[0]
    if gradient is None:
        return 0.0
    vector = torch.as_tensor(direction, device=device, dtype=gradient.dtype)
    return float(torch.dot(gradient[0, int(module_index)], vector).detach().cpu())


def _finite_difference_directional(
    model: Any,
    sample: Mapping[str, Any],
    query_xy: torch.Tensor,
    module_index: int,
    direction: np.ndarray,
    step: float,
    device: torch.device,
    *,
    fine: bool,
) -> float:
    centers_np, _ = _active_centers(sample)
    plus = centers_np.copy()
    minus = centers_np.copy()
    plus[module_index] += direction * float(step)
    minus[module_index] -= direction * float(step)
    lx, ly = _domain_lengths(sample, model)
    present = _active_centers(sample)[1]
    if not _valid_geometry(plus, present, _module_radius(model), lx, ly) or not _valid_geometry(minus, present, _module_radius(model), lx, ly):
        raise ValueError("central-difference step leaves the valid geometry domain")
    plus_tensor = torch.from_numpy(plus.astype(np.float32)).unsqueeze(0).to(device)
    minus_tensor = torch.from_numpy(minus.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        plus_value = (
            _fine_scalar_for_centers(model, sample, query_xy, plus_tensor, device)
            if fine
            else _full_scalar_for_centers(model, sample, query_xy, plus_tensor, device)
        )
        minus_value = (
            _fine_scalar_for_centers(model, sample, query_xy, minus_tensor, device)
            if fine
            else _full_scalar_for_centers(model, sample, query_xy, minus_tensor, device)
        )
    return float(((plus_value - minus_value) / (2.0 * float(step))).detach().cpu())


def _module_support_probe_indices(
    prepared_wrapper: Any,
    receivers: torch.Tensor,
    module_index: int,
) -> tuple[int | None, int | None, dict[str, int]]:
    """Find receivers connected/not connected to one module's sparse groups."""

    prepared = prepared_wrapper.prepared
    layout = prepared.backend_state.cache.layout
    source, groups = layout.module_group_indices
    module_groups = torch.unique(groups[source == int(module_index)])
    lookup = lookup_receivers(layout, receivers)
    receiver_index, receiver_groups = lookup.receiver_group_indices
    connected_rows = torch.isin(receiver_groups, module_groups)
    connected = torch.zeros(receivers.shape[1], dtype=torch.bool, device=receivers.device)
    connected[receiver_index[connected_rows]] = True
    inside = torch.where(connected)[0]
    outside = torch.where(~connected)[0]
    return (
        int(inside[0]) if inside.numel() else None,
        int(outside[0]) if outside.numel() else None,
        {
            "module_group_count": int(module_groups.numel()),
            "connected_receiver_count": int(connected.sum().item()),
            "disconnected_receiver_count": int((~connected).sum().item()),
        },
    )


def _conditional_state_fine_path(
    model: Any,
    prepared_wrapper: Any,
    receivers: torch.Tensor,
    module_index: int,
    probe_index: int,
    state_steps: Sequence[float] = (1.0e-3, 5.0e-4),
) -> dict[str, Any]:
    """Differentiate one prepared group update with coarse/global held fixed."""

    prepared = prepared_wrapper.prepared
    base_states = prepared.module_states.detach()
    direction = torch.ones_like(base_states[:, int(module_index)])
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-12)
    selector = torch.zeros_like(base_states)
    selector[:, int(module_index)] = direction
    probe = receivers[:, int(probe_index) : int(probe_index) + 1]

    def scalar(delta: torch.Tensor) -> torch.Tensor:
        states = base_states + delta * selector
        refreshed = model.core.prepare(
            prepared.encoded,
            states,
            layout_cache=prepared.backend_state.cache,
        )
        receiver_features = model.core._receiver_features(refreshed, probe)
        main, _ = model.core.backend.read(
            refreshed.backend_state,
            refreshed.encoded,
            probe,
            receiver_features,
        )
        return main.sum()

    delta = torch.zeros((), device=receivers.device, dtype=base_states.dtype, requires_grad=True)
    autograd_value = float(torch.autograd.grad(scalar(delta), delta)[0].detach().cpu())
    rows: dict[str, Any] = {}
    for step in state_steps:
        with torch.no_grad():
            plus = scalar(delta.detach().new_tensor(float(step)))
            minus = scalar(delta.detach().new_tensor(-float(step)))
        finite = float(((plus - minus) / (2.0 * float(step))).detach().cpu())
        rows[f"h={step:g}"] = {
            "autograd": autograd_value,
            "central_fd": finite,
            "absolute_difference": abs(autograd_value - finite),
            "relative_difference": abs(autograd_value - finite) / max(abs(finite), 1.0e-12),
        }
    return rows


def run_gradients(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_checkpoint_specs(args.sparse_checkpoint or args.checkpoint)
    if len(specs) != 1:
        raise ValueError("gradients requires exactly one sparse checkpoint")
    device = select_device(args.device)
    spec = specs[0]
    model, checkpoint = _load_model_spec(spec, device)
    dataset, dataset_path = _load_dataset(checkpoint, args)
    requested = list(args.case_id or DEFAULT_CASE_IDS)
    if len(requested) < 4:
        requested = [str(item) for item in dataset.selected_case_ids[:4]]
    case_ids = requested[:4]
    step_radii = args.step_radius or [0.01, 0.005]
    steps = [float(value) * _module_radius(model) for value in step_radii]
    state_before = _state_keys(model)
    layout_rows: list[dict[str, Any]] = []
    for order, case_id in enumerate(case_ids):
        sample = select_sample(dataset, str(case_id), 0)
        centers, present = _active_centers(sample)
        module_index = int(np.flatnonzero(present)[0])
        transition: dict[str, Any] = {"constructed": False}
        if order == len(case_ids) - 1:
            sample, module_index, direction, transition = _align_one_port_to_support_transition(
                sample, model, max(steps)
            )
            centers, present = _active_centers(sample)
        else:
            direction = _direction_for_sample(sample, model, module_index, max(steps))
        query_np = _query_points(sample, int(args.query_count))
        query = torch.from_numpy(query_np).unsqueeze(0).to(device)
        with torch.no_grad():
            base = _forward_tensor_batch(model, sample, query, device, return_prepared_state=True, return_routing_maps=True)
        inside_index, outside_index, support_counts = _module_support_probe_indices(
            base["prepared_state"], query, module_index
        )
        row: dict[str, Any] = {
            "layout_index": order,
            "case_id": str(sample["case_id"]),
            "module_index": module_index,
            "direction": direction.tolist(),
            "fixed_kpi": "mean_temperature",
            "support_transition": transition,
            "support_probe_indices": {"inside": inside_index, "outside": outside_index},
            "module_support_connectivity": support_counts,
            "full_path": {},
            "fine_path": {"inside": {}, "outside": {}},
            "conditional_prepared_state_fine_path": {},
        }
        for branch, index in (("inside", inside_index), ("outside", outside_index)):
            if index is None:
                row["conditional_prepared_state_fine_path"][branch] = {
                    "status": "unavailable",
                    "reason": "module_specific_support_probe_not_present",
                }
            else:
                row["conditional_prepared_state_fine_path"][branch] = _conditional_state_fine_path(
                    model, base["prepared_state"], query, module_index, index
                )
        for step in steps:
            label = f"h={step / _module_radius(model):.6g}r"
            try:
                auto = _autograd_directional(model, sample, query, module_index, direction, device, fine=False)
                finite = _finite_difference_directional(model, sample, query, module_index, direction, step, device, fine=False)
                row["full_path"][label] = {
                    "step": step,
                    "autograd": auto,
                    "central_fd": finite,
                    "absolute_difference": abs(auto - finite),
                    "relative_difference": abs(auto - finite) / max(abs(finite), 1.0e-12),
                }
            except Exception as exc:  # noqa: BLE001 - retain per-layout boundary limitations
                row["full_path"][label] = {"step": step, "status": "unavailable", "error": f"{type(exc).__name__}: {exc!s}"}
            for branch, index in (("inside", inside_index), ("outside", outside_index)):
                if index is None:
                    row["fine_path"][branch][label] = {"step": step, "status": "unavailable", "reason": "support_probe_not_present"}
                    continue
                probe = query[:, int(index) : int(index) + 1]
                try:
                    auto = _autograd_directional(model, sample, probe, module_index, direction, device, fine=True)
                    finite = _finite_difference_directional(model, sample, probe, module_index, direction, step, device, fine=True)
                    row["fine_path"][branch][label] = {
                        "step": step,
                        "autograd": auto,
                        "central_fd": finite,
                        "absolute_difference": abs(auto - finite),
                        "relative_difference": abs(auto - finite) / max(abs(finite), 1.0e-12),
                    }
                except Exception as exc:  # noqa: BLE001
                    row["fine_path"][branch][label] = {"step": step, "status": "unavailable", "error": f"{type(exc).__name__}: {exc!s}"}
        if transition.get("constructed"):
            plus = centers.copy()
            minus = centers.copy()
            plus[module_index] += direction * max(steps)
            minus[module_index] -= direction * max(steps)
            plus_keys = _support_key_set_for_centers(model, sample, query, plus, module_index, device)
            minus_keys = _support_key_set_for_centers(model, sample, query, minus, module_index, device)
            row["support_transition"].update(
                {
                    "minus_group_count": len(minus_keys),
                    "plus_group_count": len(plus_keys),
                    "symmetric_key_difference_count": len(minus_keys.symmetric_difference(plus_keys)),
                    "topology_changed": minus_keys != plus_keys,
                }
            )
        layout_rows.append(row)
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "task": "gradients",
        "checkpoints": [_checkpoint_record(spec, checkpoint, model)],
        "dataset": str(dataset_path),
        "step_sizes": [float(value) for value in step_radii],
        "results": layout_rows,
        "limitations": [
            "Autograd-versus-FD agreement validates the learned implementation only; it is not a physical derivative validation.",
            "Fine-path rows are conditional sparse reads with coarse and local routes clamped to zero.",
            "Support keys are rebuilt for each displaced evaluation; no stale prepared topology is reused.",
        ],
        "state_dict_structure_unchanged": state_before == _state_keys(model),
    }


def _cardinal_directions() -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(direction, dtype=np.float64)
        for direction in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    )


def _unit_direction(value: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        value = np.asarray(fallback if fallback is not None else (1.0, 0.0), dtype=np.float64)
        norm = float(np.linalg.norm(value))
    return value / max(norm, 1.0e-12)


def _family_module_indices(sample: Mapping[str, Any], family: str) -> tuple[int, ...] | None:
    """Choose deterministic pair/triple members from one valid base layout."""

    centers, present = _active_centers(sample)
    active = np.flatnonzero(present).astype(int).tolist()
    if family in {"pair_separated", "pair_crowded"} and len(active) < 2:
        return None
    if family == "triple" and len(active) < 3:
        return None
    distances: list[tuple[float, tuple[int, ...]]] = []
    if family in {"pair_separated", "pair_crowded"}:
        for left, right in itertools.combinations(active, 2):
            distances.append((float(np.linalg.norm(centers[right] - centers[left])), (left, right)))
        distances.sort(key=lambda item: (item[0], item[1]))
        return distances[-1][1] if family == "pair_separated" else distances[0][1]
    for members in itertools.combinations(active, 3):
        distance_sum = sum(
            float(np.linalg.norm(centers[right] - centers[left]))
            for left, right in itertools.combinations(members, 2)
        )
        distances.append((distance_sum, members))
    distances.sort(key=lambda item: (item[0], item[1]))
    return distances[0][1]


def _candidate_family_directions(
    sample: Mapping[str, Any],
    family: str,
    module_indices: tuple[int, ...],
) -> Iterator[tuple[np.ndarray, ...]]:
    """Yield preferred then small deterministic direction sets for a family."""

    centers, _ = _active_centers(sample)
    cardinal = _cardinal_directions()
    if len(module_indices) == 2:
        first, second = module_indices
        line = _unit_direction(centers[second] - centers[first])
        if family == "pair_separated":
            preferred = (-line, line)
        else:
            preferred = (line, -line)
        yield tuple(np.asarray(value, dtype=np.float64) for value in preferred)
    else:
        members = centers[list(module_indices)]
        centroid = members.mean(axis=0)
        radial = tuple(
            _unit_direction(point - centroid, fallback=cardinal[index % len(cardinal)])
            for index, point in enumerate(members)
        )
        yield radial
    for directions in itertools.product(cardinal, repeat=len(module_indices)):
        yield tuple(np.asarray(value, dtype=np.float64) for value in directions)


def _perturbed_sample(
    sample: Mapping[str, Any],
    model: Any,
    module_indices: Sequence[int],
    directions: Sequence[np.ndarray],
    mask: int,
    scale: float,
) -> dict[str, Any]:
    result = _copy_sample(sample)
    centers, present = _active_centers(result)
    for bit, (module_index, direction) in enumerate(
        zip(module_indices, directions, strict=True)
    ):
        if mask & (1 << bit):
            if module_index < 0 or module_index >= len(centers) or not present[module_index]:
                raise ValueError(f"module_index={module_index} is not active")
            centers[module_index] += _unit_direction(direction) * float(scale)
    lx, ly = _domain_lengths(result, model)
    if not _valid_geometry(centers, present, _module_radius(model), lx, ly):
        raise ValueError("perturbation corner violates module bounds or nonintersection")
    result["structure"]["module_centers"] = centers.astype(np.float32)
    return result


def _find_family_directions(
    sample: Mapping[str, Any],
    model: Any,
    family: str,
    module_indices: tuple[int, ...],
    scale: float,
) -> tuple[np.ndarray, ...] | None:
    for directions in _candidate_family_directions(sample, family, module_indices):
        valid = True
        for mask in range(1 << len(module_indices)):
            try:
                _perturbed_sample(sample, model, module_indices, directions, mask, scale)
            except ValueError:
                valid = False
                break
        if valid:
            return tuple(_unit_direction(direction) for direction in directions)
    return None


def _prediction_summary(outputs: Mapping[str, Any], query_xy: torch.Tensor, model: Any) -> dict[str, Any]:
    field = outputs["pred_field"].detach()
    return {
        "kpis": _kpi_values(outputs, query_xy, model),
        "field_shape": [int(value) for value in field.shape],
        "field_mean": float(field.float().mean().cpu()),
        "field_std": float(field.float().std(unbiased=False).cpu()),
    }


def _interaction_from_corners(corners: Sequence[Mapping[str, Any]], order: int) -> dict[str, float] | None:
    required = 1 << int(order)
    by_mask = {
        int(corner["corner_mask"]): corner.get("prediction", {}).get("kpis", {})
        for corner in corners
        if corner.get("prediction", {}).get("kpis")
    }
    if len(by_mask) != required or any(mask not in by_mask for mask in range(required)):
        return None
    names = sorted(set.intersection(*(set(values) for values in by_mask.values())))
    interaction: dict[str, float] = {}
    for name in names:
        value = 0.0
        for mask in range(required):
            sign = -1.0 if (order - int(mask.bit_count())) % 2 else 1.0
            value += sign * float(by_mask[mask][name])
        interaction[name] = value
    return interaction


def run_requests(args: argparse.Namespace) -> dict[str, Any]:
    raw_specs = list(args.checkpoint or [])
    if args.primary_checkpoint:
        raw_specs.append(args.primary_checkpoint)
    specs = parse_checkpoint_specs(raw_specs)
    if not specs:
        raise ValueError("requests requires at least one explicitly labelled checkpoint")
    if int(args.max_requests) < 1 or int(args.max_requests) > 16:
        raise ValueError("--max-requests must be between 1 and 16")
    device = select_device(args.device)
    loaded = [(spec, *_load_model_spec(spec, device)) for spec in specs]
    _spec, model, checkpoint = loaded[0]
    dataset, dataset_path = _load_dataset(checkpoint, args)
    radii = {_module_radius(item_model) for _, item_model, _ in loaded}
    if len(radii) != 1:
        raise ValueError(f"request models must share one module radius, got {sorted(radii)}")
    requested_case = str((args.case_id or DEFAULT_CASE_IDS)[0])
    try:
        sample = select_sample(dataset, requested_case, 0)
    except KeyError:
        if args.case_id:
            raise
        requested_case = str(dataset.selected_case_ids[0])
        sample = select_sample(dataset, requested_case, 0)
    query_np = _query_points(sample, int(args.query_count))
    query = torch.from_numpy(query_np).unsqueeze(0).to(device)
    scale = float(args.perturbation_scale) * _module_radius(model)
    family_order = (("pair_separated", 2), ("pair_crowded", 2), ("triple", 3))
    family_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_requests = 0
    state_before = {item_spec.label: _state_keys(item_model) for item_spec, item_model, _ in loaded}
    for family, order in family_order:
        module_indices = _family_module_indices(sample, family)
        corner_count = 1 << order
        if module_indices is None:
            skipped.append({"family": family, "reason": "insufficient_active_modules"})
            continue
        if total_requests + corner_count > int(args.max_requests):
            skipped.append({"family": family, "reason": "max_requests_reached"})
            continue
        directions = _find_family_directions(sample, model, family, module_indices, scale)
        if directions is None:
            skipped.append({"family": family, "reason": "no_valid_all_corner_direction_set"})
            continue
        corners: list[dict[str, Any]] = []
        for mask in range(corner_count):
            row: dict[str, Any] = {
                "request_id": f"{family}-{mask:0{order}b}",
                "family": family,
                "case_id": str(sample["case_id"]),
                "corner_mask": mask,
                "active_modules": [int(value) for value in module_indices],
                "perturbations": [
                    {
                        "module_index": int(module_index),
                        "delta": (directions[bit] * scale).tolist() if mask & (1 << bit) else [0.0, 0.0],
                    }
                    for bit, module_index in enumerate(module_indices)
                ],
                "reference_status": "reference_verification_pending",
                "reference_solver": "unavailable",
                "reference_reuse_key": (
                    f"{sample['case_id']}:unperturbed"
                    if mask == 0
                    else f"{sample['case_id']}:{family}:corner-{mask}"
                ),
            }
            try:
                corner_sample = _perturbed_sample(sample, model, module_indices, directions, mask, scale)
                row["predictions"] = {}
                for item_spec, item_model, _ in loaded:
                    with torch.no_grad():
                        outputs = _forward_batch(item_model, corner_sample, query_np, device)
                    row["predictions"][item_spec.label] = _prediction_summary(
                        outputs, query, item_model
                    )
            except Exception as exc:  # noqa: BLE001 - retain an exportable request on model-path failure
                row["predictions"] = {
                    "status": "unavailable",
                    "error": f"{type(exc).__name__}: {exc!s}",
                }
            corners.append(row)
        family_rows.append(
            {
                "family": family,
                "formula_order": order,
                "module_indices": [int(value) for value in module_indices],
                "scale_in_module_radii": float(args.perturbation_scale),
                "interaction_predictions": {
                    item_spec.label: _interaction_from_corners(
                        [
                            {
                                **corner,
                                "prediction": corner.get("predictions", {}).get(item_spec.label, {}),
                            }
                            for corner in corners
                        ],
                        order,
                    )
                    for item_spec in specs
                },
                "reference_status": "reference_verification_pending",
                "corners": corners,
            }
        )
        total_requests += len(corners)
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "task": "requests",
        "checkpoints": [
            _checkpoint_record(item_spec, item_checkpoint, item_model)
            for item_spec, item_model, item_checkpoint in loaded
        ],
        "dataset": str(dataset_path),
        "case_id": str(sample["case_id"]),
        "query_count": len(query_np),
        "perturbation_scale_in_module_radii": float(args.perturbation_scale),
        "reference_solve_budget": 16,
        "request_count": int(total_requests),
        "reference_status": "reference_verification_pending",
        "families": family_rows,
        "skipped_families": skipped,
        "state_dict_structure_unchanged": {
            item_spec.label: state_before[item_spec.label] == _state_keys(item_model)
            for item_spec, item_model, _ in loaded
        },
        "limitations": [
            "No maintained CFD solver is invoked by this task; every physical reference remains pending.",
            "Predictions are fixed-probe primary-model diagnostics, not CFD substitutes.",
            "The model export uses the base-case probe locations; a solver-side common-fluid mask and pressure-gauge audit remain pending.",
        ],
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument("--device", default="cpu", help="evaluation device, e.g. cpu or cuda:0")
    parser.add_argument("--dataset", default=None, help="override the packed HDF5 dataset path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=8192)


def _add_single_model_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--sparse-checkpoint", action="append", default=[], metavar="LABEL=PATH")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)

    interventions = subparsers.add_parser("interventions", help="sparse context intervention diagnostics")
    _add_common_arguments(interventions)
    _add_single_model_checkpoint_arguments(interventions)
    interventions.add_argument("--perturbation-scale", type=float, default=0.1)
    interventions.set_defaults(handler=run_interventions)

    gradients = subparsers.add_parser("gradients", help="fixed-KPI autograd versus central finite differences")
    _add_common_arguments(gradients)
    _add_single_model_checkpoint_arguments(gradients)
    gradients.add_argument("--step-radius", action="append", type=float, default=None, metavar="FRACTION_OF_R")
    gradients.set_defaults(handler=run_gradients)

    quadrature = subparsers.add_parser("quadrature", help="environment duplication consistency for four models")
    _add_common_arguments(quadrature)
    quadrature.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    quadrature.add_argument("--duplication-factor", type=int, default=2)
    quadrature.set_defaults(handler=run_quadrature)

    scaling = subparsers.add_parser("scaling", help="real-support synthetic scaling measurements")
    _add_common_arguments(scaling)
    scaling.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    scaling.add_argument("--shape", action="append", default=[], metavar="M,E,Q")
    scaling.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    scaling.add_argument("--warmup", type=int, default=1)
    scaling.add_argument("--repetitions", type=int, default=5)
    scaling.add_argument("--chunk-equality-limit", type=int, default=8192)
    scaling.set_defaults(handler=run_scaling)

    requests = subparsers.add_parser("requests", help="bounded pair/triple model predictions and pending reference requests")
    _add_common_arguments(requests)
    requests.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    requests.add_argument("--primary-checkpoint", default=None, metavar="LABEL=PATH")
    requests.add_argument("--perturbation-scale", type=float, default=0.1)
    requests.add_argument("--max-requests", type=int, default=16)
    requests.set_defaults(handler=run_requests)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = args.handler(args)
    payload["output"] = str(Path(args.output).expanduser().resolve())
    write_json(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
