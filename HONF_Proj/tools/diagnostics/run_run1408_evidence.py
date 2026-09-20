"""Run-1408 sampled-environment evidence driver.

This module is deliberately an evidence entry point, not a trainer or a new
executor.  It has three bounded pieces:

* a fresh-profile predicted-port forward/backward/update check;
* synchronized candidate checkpoint timing and sampler-ledger collection; and
* optional frozen interventions plus actual interpolation-cell plots.

The physical path reuses the maintained ThermalChannel loaders, loss path, and
optimizer.  Timed calls never request routing maps.  One separate untimed
call requests the Run-1408 sample maps and writes only the arrays needed to
describe executed sites and interpolation corners.  Plan-only mode does not
import model code, open a dataset, initialize CUDA, or load a checkpoint.

Example plan check::

    PYTHONPATH=src:Case_ThermalChannel/src \
    python tools/diagnostics/run_run1408_evidence.py --plan-only \
        --output diagnostics/generated/run1408/evidence.json

Example physical evidence::

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
    python tools/diagnostics/run_run1408_evidence.py \
        --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
        --checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1408_.../epoch_0050_model.pt \
        --device cuda:0 --output diagnostics/generated/run1408/evidence.json

No claim of speedup is made by this tool.  A missing sampled-bank byte
counter, phase ledger, or intervention hook is retained as unavailable rather
than reconstructed from logical ``A^E`` support.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = PROJECT_ROOT / "src/config_core/forward/hypergraph_quadrature_honf_context.json"
DEFAULT_CASE_IDS = ("0273", "0653")
DEFAULT_QUERY_COUNT = 8192
DEFAULT_RECEIVER_CHUNK_SIZE = 2048
DEFAULT_MAP_QUERY_COUNT = 8192
DEFAULT_INFERENCE_WARMUPS = 2
DEFAULT_INFERENCE_REPETITIONS = 5
DEFAULT_TRAIN_BATCH_SIZE = 48
DEFAULT_TRAIN_QUERY_COUNT = 1024
DEFAULT_TRAIN_WARMUPS = 1
DEFAULT_TRAIN_REPETITIONS = 3
DEFAULT_UPDATE_BATCH_SIZE = 12
DEFAULT_UPDATE_QUERY_COUNT = 1024
DEFAULT_MAP_QUERY_INDICES = (0, 4096, 8191)
ARCHITECTURE = "hypergraph_quadrature_honf"
GROUP_COUNT = 6
SAMPLES_PER_GROUP = 4
ATTENTION_HEADS = 4


def _finite(value: Any) -> float | None:
    """Return one finite scalar or ``None`` without raising on tensors."""

    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1)[0].item()
        value = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError, RuntimeError):
        return None
    return value if math.isfinite(value) else None


def _jsonable(value: Any) -> Any:
    """Convert evidence values while bounding large tensors to summaries."""

    if hasattr(value, "detach") and hasattr(value, "numel"):
        value = value.detach().cpu()
        if int(value.numel()) == 1:
            return _jsonable(value.item())
        if int(value.numel()) <= 256:
            return value.tolist()
        array = value.reshape(-1).float()
        return {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": _finite(array.min()),
            "max": _finite(array.max()),
            "mean": _finite(array.mean()),
            "numel": int(value.numel()),
        }
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return value.tolist()
        flat = value.reshape(-1).astype(np.float64, copy=False)
        finite = flat[np.isfinite(flat)]
        return {
            "array": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": None if finite.size == 0 else float(np.min(finite)),
            "max": None if finite.size == 0 else float(np.max(finite)),
            "mean": None if finite.size == 0 else float(np.mean(finite)),
            "numel": int(value.size),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _compact_optimizer_inventory(inventory: Any) -> dict[str, Any] | None:
    """Keep cost-relevant optimizer counts without provenance digests or name dumps."""

    if not isinstance(inventory, Mapping):
        return None
    compact: dict[str, Any] = {}
    for key in ("mode", "weight_decay"):
        if key in inventory:
            compact[key] = inventory[key]
    groups = inventory.get("groups")
    if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
        compact_groups: list[dict[str, Any]] = []
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            compact_groups.append(
                {
                    key: group[key]
                    for key in (
                        "name",
                        "learning_rate",
                        "parameter_tensor_count",
                        "trainable_scalar_count",
                        "scalar_count_complete",
                    )
                    if key in group
                }
            )
        compact["groups"] = compact_groups
    return compact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_existing(raw: str | Path | None, *, name: str, required: bool = False) -> Path | None:
    if raw is None:
        if required:
            raise ValueError(f"{name} is required for this invocation")
        return None
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _resolve_profile(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Run-1408 profile does not exist: {path}")
    return path


def _sync(device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _measure_phase(
    function: Callable[[], Any],
    device: Any,
    *,
    warmups: int,
    repetitions: int,
    inference: bool = True,
) -> dict[str, Any]:
    """Measure synchronized wall time and allocated/reserved CUDA peaks."""

    import torch

    if int(warmups) < 0 or int(repetitions) <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    context = torch.inference_mode if inference else contextlib.nullcontext
    with context():
        for _ in range(int(warmups)):
            result = function()
            del result
        _sync(device)
    samples: list[dict[str, Any]] = []
    for index in range(int(repetitions)):
        if getattr(device, "type", None) == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
        else:
            baseline_allocated = baseline_reserved = None
        _sync(device)
        started = time.perf_counter()
        result = None
        error: str | None = None
        try:
            with context():
                result = function()
        except Exception as exc:  # noqa: BLE001 - evidence retains the failure
            error = f"{type(exc).__name__}: {exc}"
        _sync(device)
        elapsed = time.perf_counter() - started
        if getattr(device, "type", None) == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = peak_reserved = None
        samples.append(
            {
                "repetition": index + 1,
                "elapsed_seconds": float(elapsed),
                "status": "error" if error else "complete",
                "error": error,
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": (
                    None if peak_allocated is None or baseline_allocated is None else peak_allocated - baseline_allocated
                ),
                "incremental_peak_reserved_bytes": (
                    None if peak_reserved is None or baseline_reserved is None else peak_reserved - baseline_reserved
                ),
            }
        )
        del result
        if error:
            break
    complete = [row for row in samples if row["status"] == "complete"]
    elapsed_values = [float(row["elapsed_seconds"]) for row in complete]
    return {
        "status": "complete" if len(complete) == int(repetitions) else "incomplete",
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "samples": samples,
        "median_seconds": None if not elapsed_values else float(np.median(elapsed_values)),
        "min_seconds": None if not elapsed_values else float(np.min(elapsed_values)),
        "max_seconds": None if not elapsed_values else float(np.max(elapsed_values)),
        "spread_seconds": None if not elapsed_values else float(np.max(elapsed_values) - np.min(elapsed_values)),
        "peak_allocated_bytes": max(
            (row["peak_allocated_bytes"] for row in samples if row["peak_allocated_bytes"] is not None),
            default=None,
        ),
        "peak_reserved_bytes": max(
            (row["peak_reserved_bytes"] for row in samples if row["peak_reserved_bytes"] is not None),
            default=None,
        ),
    }


def _phase_forward_kwargs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interface_condition": batch.get("interface_condition"),
        "local_module_params": batch.get("local_module_params"),
        "teacher_port_tokens": batch.get("teacher_port_tokens"),
        "local_query_points": batch.get("module_internal_query_points"),
        "local_port_condition_mode": "predicted",
        "mixed_teacher_ratio": 0.0,
    }


def _extract_prepared(output: Mapping[str, Any]) -> Any:
    prepared = output.get("prepared_state")
    if prepared is None:
        raise KeyError("forward did not return prepared_state")
    return prepared


def _measurement_summary(measurement: Mapping[str, Any]) -> dict[str, Any]:
    return dict(measurement)


def _aux_mapping(output: Any) -> Mapping[str, Any]:
    """Find the maintained interaction auxiliary mapping in wrapper output."""

    if isinstance(output, Mapping):
        for key in ("interaction_aux", "_interaction_aux"):
            value = output.get(key)
            if isinstance(value, Mapping):
                return value
        routing = output.get("routing_aux")
        if isinstance(routing, Mapping) and any("sample" in str(key) for key in routing):
            return routing
        prepared = output.get("prepared_state")
    else:
        prepared = output
    inner = getattr(prepared, "prepared", prepared)
    interaction = getattr(inner, "interaction_aux", None)
    if isinstance(interaction, Mapping):
        return interaction
    if isinstance(inner, Mapping):
        for key in ("interaction_aux", "_interaction_aux"):
            value = inner.get(key)
            if isinstance(value, Mapping):
                return value
    return {}


def _tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _find_suffix(mapping: Mapping[str, Any], suffixes: Sequence[str], *, phase_prefix: str = "") -> Any:
    """Find an explicitly named field while tolerating phase/key aliases."""

    candidates = tuple(str(value) for value in suffixes)
    for key, value in mapping.items():
        text = str(key)
        if phase_prefix and not text.startswith(phase_prefix):
            continue
        if any(text.endswith(suffix) for suffix in candidates):
            return value
    return None


def _array_count(value: Any, *, positive: bool = False) -> int | None:
    array = _tensor_to_numpy(value)
    if array is None:
        return None
    if positive:
        return int(np.count_nonzero(array > 0.0))
    return int(array.size)


def _sample_site_count(value: Any) -> int | None:
    """Count sampled sites rather than coordinate scalars.

    ThermalChannel coordinates are ``[..., 2]``; treating the final spatial
    coordinate dimension as sites would double the executed-slot count.
    Keep the fallback generic for a future adapter that emits scalar site
    identifiers instead of ``(x, y)`` pairs.
    """

    array = _tensor_to_numpy(value)
    if array is None:
        return None
    if array.ndim and array.shape[-1] == 2:
        return int(array.size // 2)
    return int(array.size)


def _per_query_summary(value: Any) -> dict[str, Any] | None:
    array = _tensor_to_numpy(value)
    if array is None or array.size == 0:
        return None
    if array.ndim == 0:
        return {"scalar": float(array)}
    # Preserve the first batch axis but reduce every remaining per-query field
    # to one value.  This is only a compact report; full maps go to NPZ.
    flat = array.reshape(array.shape[0], array.shape[1], -1) if array.ndim >= 2 else array.reshape(1, -1)
    values = flat.sum(axis=-1) if array.ndim >= 2 else flat
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "sum": float(np.sum(values)),
    }


def _unique_count(value: Any) -> int | None:
    array = _tensor_to_numpy(value)
    if array is None or array.size == 0:
        return None
    return int(np.unique(array.reshape(-1)).size)


def execution_ledger(
    aux: Mapping[str, Any],
    *,
    phase: str = "P2",
    attention_heads: int = ATTENTION_HEADS,
) -> dict[str, Any]:
    """Reduce actual Run-1408 sampled-reader maps/counters.

    The reducer never calls a support count an executed row.  If maps are
    present, fixed slots and interpolation incidences are counted directly;
    otherwise explicit scalar counters are retained and missing fields stay
    ``None``.
    """

    prefixes = {"P0": "initial_port_", "P1": "provisional_", "P2": "", "P2_consistency": "port_global_"}
    prefix = prefixes.get(str(phase), "")
    coordinates = _find_suffix(
        aux,
        ("environment_sample_coordinates", "environment_sampling_coordinates", "environment_sample_locations"),
        phase_prefix=prefix,
    )
    beta = _find_suffix(aux, ("environment_sample_beta", "environment_sampling_beta"), phase_prefix=prefix)
    masses = _find_suffix(
        aux,
        ("environment_sample_lambda", "environment_sample_mass", "environment_sample_masses"),
        phase_prefix=prefix,
    )
    corners = _find_suffix(
        aux,
        ("environment_interpolation_corner_indices", "environment_sample_corner_indices"),
        phase_prefix=prefix,
    )
    corner_weights = _find_suffix(
        aux,
        ("environment_interpolation_corner_weights", "environment_sample_corner_weights"),
        phase_prefix=prefix,
    )
    cells = _find_suffix(
        aux,
        ("environment_interpolation_cell_indices", "environment_sample_cell_indices"),
        phase_prefix=prefix,
    )

    coordinates_np = _tensor_to_numpy(coordinates)
    beta_np = _tensor_to_numpy(beta)
    masses_np = _tensor_to_numpy(masses)
    corners_np = _tensor_to_numpy(corners)
    corner_weights_np = _tensor_to_numpy(corner_weights)
    cells_np = _tensor_to_numpy(cells)
    slots = _sample_site_count(coordinates)
    if slots is None:
        slots = _finite(
            _find_suffix(
                aux,
                ("environment_sample_slots", "environment_sample_slots_executed"),
                phase_prefix=prefix,
            )
        )
        slots = None if slots is None else int(slots)
    nonzero = _array_count(masses, positive=True)
    if nonzero is None:
        nonzero = _finite(
            _find_suffix(aux, ("environment_nonzero_sample_masses", "environment_nonzero_sample_mass"), phase_prefix=prefix)
        )
        nonzero = None if nonzero is None else int(nonzero)
    fine_rows = _finite(
        _find_suffix(
            aux,
            ("environment_fine_rows_forward", "environment_fine_rows", "environment_sample_fine_rows"),
            phase_prefix=prefix,
        )
    )
    geometry_rows = _finite(
        _find_suffix(aux, ("environment_geometry_rows_forward", "environment_geometry_rows"), phase_prefix=prefix)
    )
    content_rows = _finite(
        _find_suffix(
            aux,
            ("environment_content_dot_rows_forward", "environment_content_rows", "environment_content_dot_rows"),
            phase_prefix=prefix,
        )
    )
    corner_loads = _finite(
        _find_suffix(
            aux,
            ("environment_interpolation_corner_loads", "environment_corner_loads"),
            phase_prefix=prefix,
        )
    )
    if corner_loads is None and corners_np is not None:
        corner_loads = float(corners_np.size // 2 if corners_np.shape[-1:] == (2,) else corners_np.shape[-1] * np.prod(corners_np.shape[:-1]))
    if fine_rows is None and slots is not None:
        fine_rows = float(slots)
    if geometry_rows is None and slots is not None:
        geometry_rows = float(slots)
    if content_rows is None and slots is not None:
        content_rows = float(slots * int(attention_heads))
    if corner_loads is None and slots is not None:
        corner_loads = float(slots * 4)
    bank_bytes = _finite(
        _find_suffix(
            aux,
            (
                "environment_sampled_bank_actual_tensor_bytes",
                "environment_sampled_bank_bytes",
                "environment_sampled_tensor_bytes",
            ),
            phase_prefix=prefix,
        )
    )
    recomputed = _finite(
        _find_suffix(
            aux,
            (
                "environment_checkpoint_recomputations",
                "environment_checkpoint_recompute_count",
                "environment_fine_rows_recompute",
            ),
            phase_prefix=prefix,
        )
    )
    unique_cells = _unique_count(cells)
    if unique_cells is None and cells_np is not None:
        unique_cells = int(np.unique(cells_np.reshape(-1)).size)
    unique_corner_tokens = None
    if corners_np is not None:
        if corner_weights_np is not None and corner_weights_np.shape == corners_np.shape:
            unique_corner_tokens = int(np.unique(corners_np[corner_weights_np > 0.0]).size)
        else:
            unique_corner_tokens = int(np.unique(corners_np.reshape(-1)).size)
    status = "actual_maps" if coordinates_np is not None else (
        "explicit_counter" if any(value is not None for value in (slots, fine_rows, geometry_rows, content_rows)) else "unavailable"
    )
    return {
        "phase": str(phase),
        "status": status,
        "sample_slots": slots,
        "nonzero_sample_masses": nonzero,
        "fine_content_geometry_rows": fine_rows,
        "fine_content_dot_rows": content_rows,
        "fine_geometry_rows": geometry_rows,
        "interpolation_corner_loads": None if corner_loads is None else int(corner_loads),
        "unique_accessed_cells": unique_cells,
        "unique_accessed_corner_tokens": unique_corner_tokens,
        "sampled_bank_actual_tensor_bytes": bank_bytes,
        "checkpoint_recomputations": recomputed,
        "sample_slots_per_query": _per_query_summary(
            _find_suffix(aux, ("environment_sample_slots_per_query",), phase_prefix=prefix)
        ),
        "nonzero_sample_masses_per_query": _per_query_summary(
            _find_suffix(aux, ("environment_nonzero_sample_mass_per_query",), phase_prefix=prefix)
        ),
        "unique_cells_per_query": _per_query_summary(
            _find_suffix(aux, ("environment_unique_cells_touched_per_query",), phase_prefix=prefix)
        ),
        "map_shapes": {
            "sample_coordinates": None if coordinates_np is None else list(coordinates_np.shape),
            "sample_beta": None if beta_np is None else list(beta_np.shape),
            "sample_mass": None if masses_np is None else list(masses_np.shape),
            "corner_indices": None if corners_np is None else list(corners_np.shape),
            "corner_weights": None if corner_weights_np is None else list(corner_weights_np.shape),
            "cell_indices": None if cells_np is None else list(cells_np.shape),
        },
    }


def _map_arrays(aux: Mapping[str, Any], *, phase_prefix: str = "") -> dict[str, np.ndarray]:
    """Extract only actual sampled-map arrays for an NPZ/plot artifact."""

    aliases = {
        "sample_coordinates": ("environment_sample_coordinates", "environment_sampling_coordinates"),
        "sample_beta": ("environment_sample_beta", "environment_sampling_beta"),
        "sample_mass": ("environment_sample_lambda", "environment_sample_mass"),
        "corner_indices": ("environment_interpolation_corner_indices", "environment_sample_corner_indices"),
        "corner_weights": ("environment_interpolation_corner_weights", "environment_sample_corner_weights"),
        "cell_indices": ("environment_interpolation_cell_indices", "environment_sample_cell_indices"),
    }
    result: dict[str, np.ndarray] = {}
    for target, suffixes in aliases.items():
        value = _find_suffix(aux, suffixes, phase_prefix=phase_prefix)
        array = _tensor_to_numpy(value)
        if array is not None:
            if array.ndim > 0 and array.shape[0] == 1:
                array = array[0]
            result[target] = np.asarray(array)
    return result


def _save_sample_map(
    path: Path,
    *,
    output: Mapping[str, Any],
    query_xy: Any,
    batch: Mapping[str, Any],
    model: Any,
    checkpoint_epoch: int,
    case_id: str,
) -> tuple[Path | None, dict[str, Any]]:
    aux = _aux_mapping(output)
    arrays = _map_arrays(aux)
    if "sample_coordinates" not in arrays:
        return None, {"status": "unavailable", "reason": "sample-coordinate debug map was not returned"}
    # ThermalChannel owns the regular grid; obtain coordinates through the
    # adapter rather than assuming token order in the visualization.
    try:
        env = model.environment_builder(
            batch_size=1,
            num_env_tokens_x=int(model.config.core_honf.num_env_tokens_x),
            num_env_tokens_y=int(model.config.core_honf.num_env_tokens_y),
            domain_length_x=float(model.config.core_honf.domain_length_x),
            domain_length_y=float(model.config.core_honf.domain_length_y),
            device=query_xy.device,
            dtype=query_xy.dtype,
        )
        env_coords = env.env_coords.detach().cpu().numpy()[0]
        layout = getattr(env, "sampler_layout", None)
        token_to_grid = None if layout is None else layout.token_to_grid.detach().cpu().numpy()
    except Exception as exc:  # noqa: BLE001 - retain map evidence as unavailable
        return None, {"status": "unavailable", "reason": f"adapter layout unavailable: {type(exc).__name__}: {exc}"}
    query_array = _tensor_to_numpy(query_xy)
    if query_array is None:
        return None, {"status": "unavailable", "reason": "query coordinates unavailable"}
    if query_array.ndim > 0 and query_array.shape[0] == 1:
        query_array = query_array[0]
    centers = _tensor_to_numpy(batch.get("structure", {}).get("module_centers"))
    present = _tensor_to_numpy(batch.get("structure", {}).get("module_present"))
    if centers is not None and centers.ndim > 0 and centers.shape[0] == 1:
        centers = centers[0]
    if present is not None and present.ndim > 0 and present.shape[0] == 1:
        present = present[0]
    arrays.update(
        {
            "query_xy": np.asarray(query_array, dtype=np.float32),
            "env_coords": np.asarray(env_coords, dtype=np.float32),
            "module_centers": np.asarray(centers if centers is not None else np.empty((0, 2)), dtype=np.float32),
            "module_present": np.asarray(present if present is not None else np.empty((0,)), dtype=np.float32),
        }
    )
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    ledger = execution_ledger(aux)
    ledger.update(
        {
            "status": "complete",
            "path": str(path),
            "case_id": str(case_id),
            "checkpoint_epoch": int(checkpoint_epoch),
            "query_count": int(query_array.shape[0]),
            "token_to_grid": None if token_to_grid is None else token_to_grid.tolist(),
        }
    )
    return path, ledger


def render_sampled_cells(
    map_path: str | Path,
    output_path: str | Path,
    *,
    selected_queries: Sequence[int] = DEFAULT_MAP_QUERY_INDICES,
) -> dict[str, Any]:
    """Render actual sample sites and accessed interpolation corners.

    The plot consumes only sampler locations/corner maps from the NPZ written
    by :func:`_save_sample_map`.  It never reads or reconstructs ``A^E``.
    """

    map_file = Path(map_path).expanduser().resolve()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"matplotlib unavailable: {exc}"}
    if not map_file.is_file():
        return {"status": "unavailable", "reason": f"map does not exist: {map_file}"}
    with np.load(map_file, allow_pickle=False) as data:
        required = ("query_xy", "env_coords", "sample_coordinates")
        missing = [name for name in required if name not in data]
        if missing:
            return {"status": "unavailable", "reason": f"map lacks {missing}"}
        query_xy = np.asarray(data["query_xy"])
        env_coords = np.asarray(data["env_coords"])
        samples = np.asarray(data["sample_coordinates"])
        beta = np.asarray(data["sample_beta"]) if "sample_beta" in data else None
        corners = np.asarray(data["corner_indices"]) if "corner_indices" in data else None
        corner_weights = np.asarray(data["corner_weights"]) if "corner_weights" in data else None
        centers = np.asarray(data["module_centers"]) if "module_centers" in data else np.empty((0, 2))
        present = np.asarray(data["module_present"]) if "module_present" in data else np.ones(len(centers))
    if samples.ndim == 5 and samples.shape[0] == 1:
        samples = samples[0]
    if beta is not None and beta.ndim == 4 and beta.shape[0] == 1:
        beta = beta[0]
    if corners is not None and corners.ndim == 5 and corners.shape[0] == 1:
        corners = corners[0]
    if corner_weights is not None and corner_weights.ndim == 5 and corner_weights.shape[0] == 1:
        corner_weights = corner_weights[0]
    indices = [int(index) for index in selected_queries if 0 <= int(index) < int(query_xy.shape[0])]
    if not indices:
        return {"status": "unavailable", "reason": "no selected query index is in the map"}
    columns = min(3, len(indices))
    rows = math.ceil(len(indices) / columns)
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(5.2 * columns, 4.3 * rows), constrained_layout=True)
    axes_flat = list(axes.reshape(-1))
    for axis, query_index in zip(axes_flat, indices, strict=False):
        axis.scatter(env_coords[:, 0], env_coords[:, 1], s=7, c="0.85", label="fine grid")
        if centers.size:
            active = present.reshape(-1) > 0.5
            axis.scatter(centers[active, 0], centers[active, 1], marker="x", c="black", s=34, label="module")
        point = query_xy[query_index]
        axis.scatter([point[0]], [point[1]], marker="*", c="black", s=80, label="receiver")
        point_samples = samples[query_index]
        for group in range(point_samples.shape[0]):
            color = plt.cm.tab10(group % 10)
            for sample_index, site in enumerate(point_samples[group]):
                weight = 1.0 if beta is None else float(beta[query_index, group, sample_index])
                axis.scatter([site[0]], [site[1]], s=26 + 70 * weight, color=color, alpha=0.9)
                if corners is not None:
                    for corner_index, token_index in enumerate(np.asarray(corners[query_index, group, sample_index]).reshape(-1)):
                        token_index = int(token_index)
                        if token_index < 0 or token_index >= len(env_coords):
                            continue
                        corner_weight = 1.0 if corner_weights is None else float(corner_weights[query_index, group, sample_index].reshape(-1)[corner_index])
                        if corner_weight <= 0.0:
                            continue
                        corner = env_coords[token_index]
                        axis.plot([site[0], corner[0]], [site[1], corner[1]], color=color, alpha=0.14, linewidth=0.7)
                        axis.scatter([corner[0]], [corner[1]], s=10 + 28 * corner_weight, color=color, alpha=0.5)
        axis.set_title(f"query {query_index}: executed sites + corners")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal", adjustable="box")
    for axis in axes_flat[len(indices) :]:
        axis.set_visible(False)
    axes_flat[0].legend(loc="best", fontsize=8)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return {"status": "complete", "path": str(output), "selected_queries": indices}


def _sampler_parameter_stats(model: Any, before: Mapping[str, Any] | None = None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if "sample_mlp" not in name and "reference_anchor_logits" not in name and "anchor_logits" not in name:
            continue
        gradient = parameter.grad
        values[name] = {
            "gradient_norm": None if gradient is None else float(torch_norm(gradient)),
            "finite_gradient": None if gradient is None else bool(_tensor_isfinite(gradient)),
            "update_norm": None,
            "max_abs_update": None,
        }
        if before is not None and name in before:
            delta = parameter.detach() - before[name]
            values[name]["update_norm"] = float(torch_norm(delta))
            values[name]["max_abs_update"] = float(delta.detach().abs().max().cpu())
    finite = [entry["finite_gradient"] for entry in values.values() if entry["finite_gradient"] is not None]
    updates = [entry["max_abs_update"] for entry in values.values() if entry["max_abs_update"] is not None]
    return {
        "parameter_count": len(values),
        "parameters": values,
        "gradients_finite": bool(values) and all(finite),
        "max_abs_update": None if not updates else float(max(updates)),
        "actual_parameter_update": bool(updates) and max(updates) > 0.0,
    }


def torch_norm(value: Any) -> float:
    import torch

    return float(torch.linalg.vector_norm(value.detach().float()).cpu())


def _tensor_isfinite(value: Any) -> bool:
    import torch

    return bool(torch.isfinite(value.detach()).all())


def _fresh_profile_update(args: argparse.Namespace) -> dict[str, Any]:
    """Run one real untrained predicted-port optimizer update."""

    import torch

    for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel" / "src", PROJECT_ROOT / "tools" / "diagnostics"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import benchmark_routing_optimization as train_bench
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.training.epoch import make_model_inputs
    from channelthermal.training.optimizer import build_forward_optimizer
    from channelthermal.workflows.train_forward import build_model_config, resolve_auto_internal_mode

    from honf_runtime.compat import recursive_to_device, set_seed
    from honf_runtime.config_loader import load_config_bundle

    profile = _resolve_profile(args.config)
    bundle = load_config_bundle(str(profile), overrides={"device": str(args.device)})
    payload = dict(bundle.effective)
    dataset_path = _parse_existing(args.dataset, name="dataset", required=True)
    dataset_cfg = dict(payload.get("dataset", {}))
    set_seed(int(payload.get("training", {}).get("seed", 0)))
    dataset = GlobalChannelThermalDataset(
        str(dataset_path),
        split=str(args.train_split),
        points_per_case=int(args.update_query_count),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        seed=int(payload.get("training", {}).get("seed", 0)),
        require_converged=bool(dataset_cfg.get("require_converged", False)),
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")
        torch.cuda.set_device(device)
    try:
        candidates = sorted(
            str(case_id)
            for case_id, count in zip(dataset.selected_case_ids, dataset.selected_module_counts, strict=True)
            if int(count) == int(args.update_module_count)
        )
        if not candidates:
            raise ValueError(f"No real training case has exactly M={args.update_module_count}.")
        case_ids = tuple(candidates[index % len(candidates)] for index in range(int(args.update_batch_size)))
        bucket = train_bench.Bucket(f"M{args.update_module_count}", int(args.update_module_count), case_ids)
        checkpoint_like = {"train_config": payload, "epoch": 1}
        loader = train_bench._build_loader(dataset, checkpoint_like, bucket, batch_size=int(args.update_batch_size))
        shape = train_bench._validate_batch_shape(loader, device, int(args.update_query_count))
        model_config = build_model_config(payload, dataset)
        model = __import__("channelthermal.model", fromlist=["ChannelThermalHONFModel"]).ChannelThermalHONFModel(model_config).to(device)
        model.set_global_target_normalization(
            dataset.normalizer.stats,
            normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        )
        resolve_auto_internal_mode(model_config, model)
        if model.config.core_honf.forward_architecture != ARCHITECTURE:
            raise ValueError(f"profile architecture is {model.config.core_honf.forward_architecture!r}, expected {ARCHITECTURE!r}")
        batch = recursive_to_device(next(iter(loader)), device)
        inputs = make_model_inputs(
            batch,
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.0,
            return_predicted_port_outputs=True,
            return_port_global_consistency=True,
        )
        model.eval()
        with torch.no_grad():
            warm_output = model(**inputs, return_routing_maps=False, return_prepared_state=False)
        del warm_output
        optimizer, inventory = build_forward_optimizer(model, payload.get("training", {}))
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if "sample_mlp" in name or "reference_anchor_logits" in name or "anchor_logits" in name
        }
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        schedule = train_bench._training_schedule(checkpoint_like)
        metrics = train_bench._run_step(
            model,
            loader,
            device,
            checkpoint_like,
            optimizer=optimizer,
            training_schedule=schedule,
        )
        _sync(device)
        elapsed = time.perf_counter() - started
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        sampler = _sampler_parameter_stats(model, before)
        return {
            "status": "complete",
            "profile": str(profile),
            "dataset": str(dataset_path),
            "port_condition_mode": schedule[2],
            "shape": shape,
            "bucket": {"label": bucket.label, "module_count": bucket.module_count, "case_ids": list(bucket.case_ids)},
            "elapsed_seconds": float(elapsed),
            "metrics": metrics,
            "gradients_finite": bool(finite_gradients),
            "parameters_finite_after": all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()),
            "sampler": sampler,
            "optimizer_inventory": _compact_optimizer_inventory(inventory),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            "scientific_note": "one fresh predicted-port update; not a training-run convergence claim",
        }
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


@contextlib.contextmanager
def _sampler_intervention(model: Any, mode: str) -> Iterator[None]:
    """Patch only the sampled reader for frozen coordinate/weight probes."""

    backend = getattr(getattr(model, "core", None), "backend", None)
    original = getattr(backend, "_sample_program", None)
    if not callable(original):
        raise TypeError("backend does not expose _sample_program for sampler intervention")

    def replacement(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 4:
            raise TypeError("backend sampler returned an unsupported program tuple")
        coordinates, beta, raw, query_center = result[:4]
        if mode == "fixed_reference_coordinates":
            coordinates = query_center[:, :, None, None, :].expand_as(coordinates)
        elif mode == "neutral_beta":
            beta = torch_full_like(beta, 1.0 / float(beta.shape[-1]))
        else:
            raise TypeError(f"unknown sampler intervention mode={mode!r}")
        return coordinates, beta, raw, query_center, *result[4:]

    backend._sample_program = replacement
    try:
        yield
    finally:
        backend._sample_program = original


def torch_full_like(value: Any, scalar: float) -> Any:
    import torch

    return torch.full_like(value, float(scalar))


@contextlib.contextmanager
def _remove_p2_environment(model: Any) -> Iterator[None]:
    """Zero only the final P2 sampled environmental context when tagged."""

    backend = getattr(getattr(model, "core", None), "backend", None)
    original = getattr(backend, "_read_environment", None)
    if not callable(original):
        raise TypeError("backend does not expose _read_environment for phase intervention")

    def replacement(*args: Any, **kwargs: Any) -> Any:
        context, aux = original(*args, **kwargs)
        role = getattr(getattr(model, "core", None), "_interface_read_role", None)
        if role != "p2_field":
            return context, aux
        return context.new_zeros(context.shape), aux

    backend._read_environment = replacement
    try:
        yield
    finally:
        backend._read_environment = original


def _module_perturbation(sample: Mapping[str, Any], model: Any) -> dict[str, Any]:
    """Copy one real case and shift its first active module safely."""

    result = copy.deepcopy(sample)
    centers = np.asarray(result["structure"]["module_centers"], dtype=np.float32).copy()
    present = np.asarray(result["structure"]["module_present"]).reshape(-1) > 0.5
    active = np.flatnonzero(present)
    if active.size == 0:
        raise ValueError("case has no active module")
    module = int(active[0])
    radius = float(model.config.core_honf.module_radius)
    scale_x = float(np.asarray(result["structure"].get("domain_length_x", [model.config.core_honf.domain_length_x])).reshape(-1)[0])
    scale_y = float(np.asarray(result["structure"].get("domain_length_y", [model.config.core_honf.domain_length_y])).reshape(-1)[0])
    delta = min(0.05, 0.08 * radius, 0.01 * min(scale_x, scale_y))
    direction = np.asarray([1.0, 0.0], dtype=np.float32)
    candidate = centers[module] + delta * direction
    if candidate[0] > scale_x - radius:
        candidate = centers[module] - delta * direction
    if candidate[0] < radius:
        raise ValueError("first active module is too close to a domain boundary for the fixed perturbation")
    centers[module] = candidate
    result["structure"]["module_centers"] = centers
    return result


def _direct_forward(model: Any, sample: Mapping[str, Any], query_np: np.ndarray, device: Any, make_batch: Callable[..., Any], *, maps: bool = False) -> Any:
    batch = make_batch(dict(sample), query_np, device)
    return model(
        batch["structure"],
        batch["query_xy"],
        interface_condition=batch.get("interface_condition"),
        local_module_params=batch.get("local_module_params"),
        teacher_port_tokens=batch.get("teacher_port_tokens"),
        local_query_points=batch.get("module_internal_query_points"),
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.0,
        return_routing_maps=bool(maps),
        return_prepared_state=False,
    )


def _intervention_rows(
    model: Any,
    checkpoint: Mapping[str, Any],
    dataset: Any,
    raw_dataset: Any,
    case_ids: Sequence[str],
    query_count: int,
    device: Any,
    dynamic: Any,
    make_batch: Callable[..., Any],
) -> list[dict[str, Any]]:
    from run_stage3_interface_study import _canonical_ground_truth_errors, _error_deltas, _relative_difference

    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        sample = __import__("channelthermal.evaluation.prepared", fromlist=["select_sample"]).select_sample(dataset, case_id, 0)
        raw_sample = __import__("channelthermal.evaluation.prepared", fromlist=["select_sample"]).select_sample(raw_dataset, case_id, 0)
        query_np = dynamic._query_points(raw_sample, int(query_count))
        with __import__("torch").inference_mode():
            base = _direct_forward(model, sample, query_np, device, make_batch)
        base_errors = _canonical_ground_truth_errors(base, sample, raw_sample, dataset, model, checkpoint)
        row: dict[str, Any] = {
            "case_id": str(case_id),
            "query_count": len(query_np),
            "ground_truth_errors": {"normal": base_errors},
            "interventions": {},
        }
        candidates: list[tuple[str, Mapping[str, Any], contextlib.AbstractContextManager[Any] | None]] = []
        for name, mode in (("fixed_reference_coordinates", "fixed_reference_coordinates"), ("neutral_beta", "neutral_beta")):
            candidates.append((name, sample, _sampler_intervention(model, mode)))
        try:
            perturbed = _module_perturbation(sample, model)
            candidates.append(("module_perturbation", perturbed, None))
        except Exception as exc:  # noqa: BLE001
            row["interventions"]["module_perturbation"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        candidates.append(("p2_environment_removed", sample, _remove_p2_environment(model)))
        for name, variant_sample, manager in candidates:
            try:
                with (manager if manager is not None else contextlib.nullcontext()), __import__("torch").inference_mode():
                    variant = _direct_forward(model, variant_sample, query_np, device, make_batch)
                errors = _canonical_ground_truth_errors(variant, variant_sample, raw_sample, dataset, model, checkpoint)
                row["ground_truth_errors"][name] = errors
                row["interventions"][name] = {
                    "status": "complete",
                    "prediction_differences": {
                        key: _relative_difference(variant[key], base[key])
                        for key in ("pred_field", "pred_interface", "pred_internal_temperature", "pred_port_condition")
                        if key in variant and key in base
                    },
                    "error_deltas": _error_deltas(base_errors, errors),
                    "interpretation": "frozen-model reliance; not physical causality",
                }
            except Exception as exc:  # noqa: BLE001 - preserve each missing hook
                row["interventions"][name] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
    return rows


def _checkpoint_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run candidate timing, maps, training-step cost, and interventions."""

    import torch

    for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel" / "src", PROJECT_ROOT / "tools" / "diagnostics"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_dynamic_sparse_routing_study as dynamic
    import run_run1405_epoch50_comparison as run1405
    import run_stage3_interface_study as stage3
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.evaluation.loading import make_batch
    from channelthermal.evaluation.prepared import select_sample

    checkpoint_path = _parse_existing(args.checkpoint, name="checkpoint", required=True)
    checkpoint = stage3.load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")
        torch.cuda.set_device(device)
    model, _ = stage3._load_model_spec(stage3.CheckpointSpec(label="1408", path=checkpoint_path), device)
    if str(model.config.core_honf.forward_architecture) != ARCHITECTURE:
        raise ValueError(f"checkpoint architecture is {model.config.core_honf.forward_architecture!r}, expected {ARCHITECTURE!r}")
    dataset, dataset_path = stage3._load_dataset(
        checkpoint,
        SimpleNamespace(dataset=args.dataset, split=args.eval_split),
    )
    map_dir = Path(args.map_dir or (Path(args.output).expanduser().resolve().parent / "maps")).expanduser().resolve()
    figure_dir = map_dir.parent / "figures"
    inference: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    try:
        for case_id in args.case_id or DEFAULT_CASE_IDS:
            sample = select_sample(dataset, str(case_id), 0)
            query_np = dynamic._query_points(sample, int(args.query_count))
            batch = make_batch(dict(sample), query_np, device)
            query = batch["query_xy"]
            kwargs = run1405._phase_forward_kwargs(batch)
            receiver_chunk = int(args.receiver_chunk_size)

            def full_forward(
                model: Any = model,
                batch: Mapping[str, Any] = batch,
                query: Any = query,
                kwargs: Mapping[str, Any] = kwargs,
                receiver_chunk: int = receiver_chunk,
            ) -> Any:
                with dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
                    return model(batch["structure"], query, return_routing_maps=False, **kwargs)

            def prepare_one(
                model: Any = model,
                batch: Mapping[str, Any] = batch,
                query: Any = query,
                kwargs: Mapping[str, Any] = kwargs,
                receiver_chunk: int = receiver_chunk,
            ) -> Any:
                with dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
                    return model(batch["structure"], query[:, :1], return_prepared_state=True, return_routing_maps=False, **kwargs)

            holder: dict[str, Any] = {}

            def prepared_p2(
                model: Any = model,
                holder: Mapping[str, Any] = holder,
                query: Any = query,
                receiver_chunk: int = receiver_chunk,
            ) -> Any:
                with dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
                    return model.decode_prepared(holder["prepared"], query, return_routing_maps=False, receiver_chunk_size=receiver_chunk)

            phases = {
                "full_forward": _measurement_summary(
                    _measure_phase(full_forward, device, warmups=args.warmups, repetitions=args.repetitions)
                ),
                "preparation_plus_one_query": _measurement_summary(
                    _measure_phase(prepare_one, device, warmups=args.warmups, repetitions=args.repetitions)
                ),
            }
            with torch.inference_mode():
                prepared_output = prepare_one()
            holder["prepared"] = _extract_prepared(prepared_output)
            phases["prepared_p2"] = _measurement_summary(
                _measure_phase(prepared_p2, device, warmups=args.warmups, repetitions=args.repetitions)
            )
            holder.clear()
            del prepared_output

            map_count = min(int(args.map_query_count), int(query.shape[1]))
            map_query = query[:, :map_count]
            with torch.inference_mode(), dynamic._runtime_receiver_chunk_size(model, receiver_chunk):
                debug_output = model(batch["structure"], map_query, return_prepared_state=False, return_routing_maps=True, **kwargs)
            phase_ledgers = {
                phase: execution_ledger(_aux_mapping(debug_output), phase=phase)
                for phase in ("P0", "P1", "P2", "P2_consistency")
            }
            map_path, map_ledger = _save_sample_map(
                map_dir / f"1408__{case_id}.npz",
                output=debug_output,
                query_xy=map_query,
                batch=batch,
                model=model,
                checkpoint_epoch=int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
                case_id=str(case_id),
            )
            figure = None
            if map_path is not None:
                selected = tuple(index for index in args.selected_query if index < map_count)
                if not selected:
                    selected = tuple(index for index in DEFAULT_MAP_QUERY_INDICES if index < map_count)
                figure = render_sampled_cells(map_path, figure_dir / f"1408__{case_id}__accessed_cells.png", selected_queries=selected)
            inference.append(
                {
                    "case_id": str(case_id),
                    "query_count": int(query.shape[1]),
                    "receiver_chunk_size": receiver_chunk,
                    "checkpoint_epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
                    "phases": phases,
                    "normal_executor": phase_ledgers,
                    "map_query_count": map_count,
                    "map_path": None if map_path is None else str(map_path),
                    "map_ledger": map_ledger,
                    "figure": figure,
                    "timed_maps": False,
                }
            )
            maps.append({"case_id": str(case_id), "map_path": None if map_path is None else str(map_path), "figure": figure})
            del debug_output
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

    train_dataset, _ = stage3._load_dataset(
        checkpoint,
        SimpleNamespace(dataset=args.dataset, split=args.train_split),
        points_per_case_override=int(args.train_query_count),
        random_point_sampling_override=False,
    )
    training: list[dict[str, Any]] = []
    try:
        buckets = run1405._make_exact_training_buckets([train_dataset], batch_size=int(args.train_batch_size))
        shared_config = dict(checkpoint.get("train_config", {}).get("training", {}))
        schedule = run1405._matched_train_schedule(checkpoint)
        for bucket in buckets:
            step_model, _ = stage3._load_model_spec(stage3.CheckpointSpec(label="1408", path=checkpoint_path), device)
            try:
                row = run1405._measure_training_bucket(
                    label="1408",
                    model=step_model,
                    checkpoint=checkpoint,
                    dataset=train_dataset,
                    bucket=bucket,
                    device=device,
                    args=args,
                    shared_training_config=shared_config,
                    shared_schedule=schedule,
                    train_bench=__import__("benchmark_routing_optimization"),
                )
                if isinstance(row, dict) and "optimizer_inventory" in row:
                    row["optimizer_inventory"] = _compact_optimizer_inventory(row["optimizer_inventory"])
                training.append(row)
            finally:
                del step_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
    finally:
        close = getattr(train_dataset, "close", None)
        if callable(close):
            close()

    # Interventions use a normalized candidate dataset and a raw target view.
    intervention_dataset, _ = stage3._load_dataset(checkpoint, SimpleNamespace(dataset=args.dataset, split=args.eval_split))
    raw_dataset = GlobalChannelThermalDataset(
        str(dataset_path),
        split=str(args.eval_split),
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=False,
    )
    try:
        interventions = _intervention_rows(
            model,
            checkpoint,
            intervention_dataset,
            raw_dataset,
            tuple(args.case_id or DEFAULT_CASE_IDS),
            int(args.intervention_query_count),
            device,
            dynamic,
            make_batch,
        )
    finally:
        for view in (intervention_dataset, raw_dataset):
            close = getattr(view, "close", None)
            if callable(close):
                close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return {
        "status": "complete" if inference else "incomplete",
        "checkpoint": {"path": str(checkpoint_path), "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1)))},
        "dataset": str(dataset_path),
        "inference": inference,
        "training": training,
        "interventions": interventions,
        "maps": maps,
        "limitations": [
            "Sampling maps and ledgers are untimed debug evidence; they are not substituted into the timing rows.",
            "The fixed-reference coordinate and neutral-beta probes are frozen-model reliance tests, not physical causality.",
            "Ground-truth errors use the maintained ThermalChannel reconstruction metrics and remain model-versus-data evidence, not new CFD.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--map-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--map-query-count", type=int, default=DEFAULT_MAP_QUERY_COUNT)
    parser.add_argument("--intervention-query-count", type=int, default=1024)
    parser.add_argument("--receiver-chunk-size", type=int, default=DEFAULT_RECEIVER_CHUNK_SIZE)
    parser.add_argument("--warmups", type=int, default=DEFAULT_INFERENCE_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_INFERENCE_REPETITIONS)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--train-query-count", type=int, default=DEFAULT_TRAIN_QUERY_COUNT)
    parser.add_argument("--train-warmups", type=int, default=DEFAULT_TRAIN_WARMUPS)
    parser.add_argument("--train-repetitions", type=int, default=DEFAULT_TRAIN_REPETITIONS)
    parser.add_argument("--update-batch-size", type=int, default=DEFAULT_UPDATE_BATCH_SIZE)
    parser.add_argument("--update-query-count", type=int, default=DEFAULT_UPDATE_QUERY_COUNT)
    parser.add_argument("--update-module-count", type=int, default=12)
    parser.add_argument("--selected-query", type=int, action="append", default=list(DEFAULT_MAP_QUERY_INDICES))
    parser.add_argument("--plan-only", action="store_true")
    return parser


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    profile = _resolve_profile(args.config)
    checkpoint = _parse_existing(args.checkpoint, name="checkpoint", required=False)
    dataset = _parse_existing(args.dataset, name="dataset", required=False)
    cases = tuple(str(value) for value in (args.case_id or DEFAULT_CASE_IDS))
    if cases != DEFAULT_CASE_IDS:
        raise ValueError("Run-1408 evidence keeps the established anchor order: cases 0273 and 0653")
    for name in (
        "query_count",
        "map_query_count",
        "intervention_query_count",
        "receiver_chunk_size",
        "train_batch_size",
        "train_query_count",
        "update_batch_size",
        "update_query_count",
        "update_module_count",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.warmups) < 0 or int(args.repetitions) <= 0 or int(args.train_warmups) < 0 or int(args.train_repetitions) <= 0:
        raise ValueError("warmups/repetitions must be nonnegative/positive")
    output = Path(args.output).expanduser().resolve()
    return {
        "schema_version": 1,
        "task": "run1408_hypergraph_quadrature_evidence",
        "status": "plan_only",
        "architecture": ARCHITECTURE,
        "scientific_contract": {
            "group_count": GROUP_COUNT,
            "group_control_dim": 16,
            "samples_per_group": SAMPLES_PER_GROUP,
            "sample_slots_per_receiver": GROUP_COUNT * SAMPLES_PER_GROUP,
            "environment_only_replacement": True,
            "dense_mm_me_em_preparation": True,
            "phase_shared_p0_controller": True,
            "three_term_assembly": "Cg+CM+CE",
        },
        "profile": str(profile),
        "dataset": None if dataset is None else str(dataset),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "protocol": {
            "case_ids": list(cases),
            "query_count": int(args.query_count),
            "map_query_count": int(args.map_query_count),
            "receiver_chunk_size": int(args.receiver_chunk_size),
            "inference_warmups": int(args.warmups),
            "inference_repetitions": int(args.repetitions),
            "training_buckets": ["M1", "M12"],
            "training_batch_size": int(args.train_batch_size),
            "training_query_count": int(args.train_query_count),
            "training_warmups": int(args.train_warmups),
            "training_repetitions": int(args.train_repetitions),
            "predicted_port_condition": True,
            "selected_query_indices": [int(value) for value in args.selected_query],
        },
        "planned_outputs": {
            "json": str(output),
            "report": str(Path(args.report).expanduser().resolve() if args.report else output.with_suffix(".md")),
            "maps": str(Path(args.map_dir or (output.parent / "maps")).expanduser().resolve()),
        },
        "evidence_fields": [
            "sample_slots",
            "nonzero_sample_masses",
            "fine_content_dot_rows",
            "fine_geometry_rows",
            "interpolation_corner_loads",
            "unique_accessed_cells",
            "sampled_bank_actual_tensor_bytes",
            "checkpoint_recomputations",
        ],
        "limitations": [
            "Plan-only mode performs no model, data, optimizer, or CUDA work.",
            "A checkpoint is optional at planning time; checkpoint evidence remains pending until supplied.",
            "Map counters are untimed and are never used as a speed claim.",
            "Missing backend counters/hooks are reported as unavailable rather than inferred from A^E.",
        ],
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Run 1408 sampled-environment evidence",
        "",
        "This report separates physical fidelity, learned sampling behavior, executed work, measured cost, and missing evidence.",
        "",
        "## Physical fidelity",
        "",
        "Ground-truth error rows and intervention error deltas are copied from the maintained ThermalChannel metric path. They are model-versus-data evidence, not CFD validation.",
        "",
    ]
    for row in payload.get("interventions", []) if isinstance(payload.get("interventions"), list) else []:
        lines.append(f"- Case {row.get('case_id')}: normal errors recorded; intervention rows={len(row.get('interventions', {}))}.")
    lines.extend(["", "## Learned sampling behavior", ""])
    lines.append("Sampler interventions: fixed/reference coordinates, neutral within-group beta, a safe module perturbation, and final-P2 environmental removal when hooks were available.")
    lines.extend(["", "## Executed work", "", "| Case | Phase | Slots | Nonzero masses | Content rows | Geometry rows | Corner loads | Unique cells | Bank bytes | Recomputed |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in payload.get("inference", []) if isinstance(payload.get("inference"), list) else []:
        for phase, ledger in (row.get("normal_executor", {}) or {}).items():
            if not isinstance(ledger, Mapping):
                continue
            lines.append(
                f"| {row.get('case_id')} | {phase} | {ledger.get('sample_slots', 'NA')} | {ledger.get('nonzero_sample_masses', 'NA')} | {ledger.get('fine_content_dot_rows', 'NA')} | {ledger.get('fine_geometry_rows', 'NA')} | {ledger.get('interpolation_corner_loads', 'NA')} | {ledger.get('unique_accessed_cells', 'NA')} | {ledger.get('sampled_bank_actual_tensor_bytes', 'NA')} | {ledger.get('checkpoint_recomputations', 'NA')} |"
            )
    lines.extend(["", "## Measured cost", "", "| Case | Full ms | Prep+one ms | Prepared P2 ms | Full peak allocated/reserved MiB |", "|---|---:|---:|---:|---:|"])
    for row in payload.get("inference", []) if isinstance(payload.get("inference"), list) else []:
        phases = row.get("phases", {})
        def ms(name: str, phases: Mapping[str, Any] = phases) -> str:
            value = _finite(phases.get(name, {}).get("median_seconds")) if isinstance(phases.get(name), Mapping) else None
            return "NA" if value is None else f"{1000.0 * value:.3f}"
        full = phases.get("full_forward", {}) if isinstance(phases.get("full_forward"), Mapping) else {}
        alloc = _finite(full.get("peak_allocated_bytes"))
        reserved = _finite(full.get("peak_reserved_bytes"))
        memory = "NA" if alloc is None or reserved is None else f"{alloc / 2**20:.2f}/{reserved / 2**20:.2f}"
        lines.append(f"| {row.get('case_id')} | {ms('full_forward')} | {ms('preparation_plus_one_query')} | {ms('prepared_p2')} | {memory} |")
    lines.extend(["", "## Fresh update", "", "The fresh-profile predicted-port update is a one-step health/evidence check, not a convergence result."])
    update = payload.get("fresh_profile_update")
    if isinstance(update, Mapping):
        sampler = update.get("sampler", {})
        lines.append(f"- status={update.get('status')}; finite gradients={update.get('gradients_finite')}; sampler update={sampler.get('actual_parameter_update')}; max sampler update={sampler.get('max_abs_update')}.")
    else:
        lines.append("- unavailable")
    lines.extend(["", "## Missing evidence", ""])
    for item in payload.get("missing_evidence", []) if isinstance(payload.get("missing_evidence"), list) else []:
        lines.append(f"- {item}")
    if not payload.get("missing_evidence"):
        lines.append("- none recorded")
    lines.append("")
    return "\n".join(lines)


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "task": "run1408_hypergraph_quadrature_evidence",
        "status": "complete",
        "architecture": ARCHITECTURE,
        "protocol": build_plan(args)["protocol"],
    }
    missing: list[str] = []
    try:
        result["fresh_profile_update"] = _fresh_profile_update(args)
    except Exception as exc:  # noqa: BLE001 - report numerical/runtime evidence
        result["fresh_profile_update"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        missing.append(f"fresh predicted-port update: {type(exc).__name__}: {exc}")
    if args.checkpoint is None:
        result["checkpoint_evidence"] = {"status": "pending", "reason": "no --checkpoint supplied"}
        result["inference"] = []
        result["training"] = []
        result["interventions"] = []
        result["maps"] = []
        missing.append("candidate checkpoint benchmark, interventions, and accessed-cell maps")
    else:
        try:
            checkpoint_payload = _checkpoint_benchmark(args)
            result.update(checkpoint_payload)
            result["checkpoint_evidence"] = {"status": checkpoint_payload.get("status", "complete")}
        except Exception as exc:  # noqa: BLE001 - retain fresh update and explicit failure
            result["checkpoint_evidence"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
            result["inference"] = []
            result["training"] = []
            result["interventions"] = []
            result["maps"] = []
            missing.append(f"candidate checkpoint evidence: {type(exc).__name__}: {exc}")
    result["missing_evidence"] = missing
    result["status"] = "complete" if result.get("fresh_profile_update", {}).get("status") == "complete" else "incomplete"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    if args.plan_only:
        _write_json(args.output, plan)
        print(json.dumps(_jsonable(plan), indent=2, sort_keys=True))
        return 0
    payload = run_measurement(args)
    _write_json(args.output, payload)
    report = Path(args.report).expanduser().resolve() if args.report else Path(args.output).expanduser().resolve().with_suffix(".md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_markdown_report(payload), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0 if payload.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE",
    "DEFAULT_CASE_IDS",
    "DEFAULT_PROFILE",
    "SAMPLES_PER_GROUP",
    "build_parser",
    "build_plan",
    "execution_ledger",
    "main",
    "render_sampled_cells",
    "run_measurement",
]
