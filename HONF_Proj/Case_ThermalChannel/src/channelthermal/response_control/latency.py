"""Reusable native query-panel, timing, and latency telemetry helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

QUERY_COUNTS = (64, 1024, 8192)
MECHANISMS = ("MM", "ME", "EM", "QM", "QE")
QUERY_BATCH_SIZE = 512
RENDERER_LATENCY_NAME = "thermal_latency_renderer.csv"
RENDERER_LATENCY_FIELDS = (
    "lane",
    "model",
    "checkpoint_sha256",
    "access_mode",
    "p50_ms",
    "p90_ms",
    "n_cases",
    "input_panel_sha256",
    "sync_group",
    "complete_wrapper",
)
RENDERER_MODEL_BY_VARIANT = {
    "G_sparse": ("G_hard", "sparse"),
    "P_sparse": ("P_hard", "sparse"),
    "G_full_access_fallback": ("G_full", "full_access"),
    "P_full_access_fallback": ("P_full", "full_access"),
    "B_retained_reference": ("retained_reference", "full_access"),
}
RENDERER_REQUIRED_MODELS = frozenset(value[0] for value in RENDERER_MODEL_BY_VARIANT.values())
RENDERER_ACCESS_BY_MODEL = {model: access for model, access in RENDERER_MODEL_BY_VARIANT.values()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sha(actual: str, expected: str, label: str) -> None:
    expected = str(expected).lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"{label} requires an explicit 64-character SHA256.")
    if actual.lower() != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}.")


def hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _hash_array(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.kind in "fiu":
        array = np.asarray(array, dtype=array.dtype.newbyteorder("<"), order="C")
    header = {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0" + array.tobytes(order="C") + b"\0")


def input_digest(record: Any, role_rows: Mapping[str, Sequence[int]]) -> str:
    """Hash exact model inputs plus valid/weighted row identity, excluding labels."""
    h = hashlib.sha256()
    modules = record.design.modules
    metadata = {
        "module_ids": [str(module.module_id) for module in modules],
        "context": {str(k): float(v) for k, v in sorted(record.context.values.items())},
        "selected_role_rows": {str(name): [int(row) for row in rows] for name, rows in sorted(role_rows.items())},
    }
    h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    h.update(b"\0")
    _hash_array(h, "module_positions", [module.position_xy for module in modules])
    _hash_array(h, "module_heating", [module.heating for module in modules])
    _hash_array(h, "module_present", [module.active for module in modules])
    index_by_id = {module.module_id: index for index, module in enumerate(modules)}
    for name, role in sorted(record.output.roles.items()):
        rows = np.asarray(role_rows.get(name, ()), dtype=np.int64)
        slots = None
        if role.receiver_module_ids is not None:
            slots = [index_by_id[str(role.receiver_module_ids[int(row)])] for row in rows]
        schema = {
            "role": name,
            "coordinate_kind": str(role.coordinate_kind),
            "channel_names": list(role.channel_names),
            "channel_units": list(role.channel_units),
            "receiver_slots": slots,
            "query_ids": [str(role.query_ids[int(row)]) for row in rows],
        }
        h.update(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\0")
        valid = np.asarray(role.valid_mask, dtype=bool)
        if valid.ndim == 1:
            valid = np.broadcast_to(valid[:, None], (role.query_features.shape[0], len(role.channel_names)))
        _hash_array(h, f"{name}.query_features", np.asarray(role.query_features)[rows])
        _hash_array(h, f"{name}.valid_mask", valid[rows])
        _hash_array(h, f"{name}.quadrature_weights", np.asarray(role.quadrature_weights)[rows])
    return h.hexdigest()


def role_candidates(record: Any, *, seed: int) -> dict[str, Any]:
    roles = record.output.roles
    required = {"fluid_fields", "interface", "solid_temperature"}
    if set(roles) != required:
        raise ValueError("Saved panel must contain exactly the three native Thermal response roles.")
    active_slots = tuple(i for i, module in enumerate(record.design.modules) if module.active)
    active_ids = {str(record.design.modules[slot].module_id): slot for slot in active_slots}
    if not active_slots:
        raise ValueError("Saved panel has no active receiver module slots.")
    eligible_counts: dict[str, int] = {}
    candidates: dict[str, Any] = {}
    for role_index, name in enumerate(("fluid_fields", "interface", "solid_temperature")):
        role = roles[name]
        if name == "fluid_fields":
            if role.coordinate_kind != "eulerian" or role.query_features.shape[1] < 2:
                raise ValueError("Fluid query rows must be saved Eulerian x/y native rows.")
        elif role.receiver_module_ids is None:
            raise ValueError(f"Saved {name} material queries must bind native receiver modules.")
        if role.receiver_module_ids is not None and not set(map(str, role.receiver_module_ids)).issubset(
            {str(module.module_id) for module in record.design.modules}
        ):
            raise ValueError(f"Saved {name} receiver rows reference a module outside the saved design.")
        valid = np.asarray(role.valid_mask, dtype=bool)
        if valid.ndim == 1:
            valid = np.broadcast_to(valid[:, None], (role.query_features.shape[0], len(role.channel_names)))
        weights = np.asarray(role.quadrature_weights, dtype=np.float64)
        finite_features = np.isfinite(np.asarray(role.query_features)).all(axis=1)
        eligible = np.flatnonzero(np.all(valid, axis=1) & finite_features & np.isfinite(weights) & (weights > 0.0))
        eligible_counts[name] = int(eligible.size)
        by_slot: dict[int, list[int]] = {slot: [] for slot in active_slots}
        seen_features: dict[int, set[bytes]] = {slot: set() for slot in active_slots}
        seen_ids: set[str] = set()
        lookup_by_slot: dict[int, dict[bytes, int]] = {slot: {} for slot in active_slots}
        for row in eligible:
            slot = (
                active_slots[0]
                if role.receiver_module_ids is None
                else active_ids.get(str(role.receiver_module_ids[row]), -1)
            )
            if slot not in by_slot:
                continue
            features = np.asarray(
                role.query_features[row, :2] if name == "fluid_fields" else role.query_features[row],
                dtype="<f8",
            ).tobytes()
            query_id = str(role.query_ids[row])
            if features in seen_features[slot] or query_id in seen_ids:
                continue
            seen_features[slot].add(features)
            seen_ids.add(query_id)
            by_slot[slot].append(int(row))
            lookup_by_slot[slot][features] = int(row)
        rng = np.random.default_rng(seed + 104729 * (role_index + 1))
        if name == "fluid_fields":
            if not by_slot[active_slots[0]]:
                raise ValueError("Saved panel lacks valid unique Eulerian fluid rows.")
            candidates[name] = tuple(
                int(row) for row in rng.permutation(np.asarray(by_slot[active_slots[0]], dtype=np.int64))
            )
            continue
        if any(not by_slot[slot] for slot in active_slots):
            raise ValueError(f"Every active module needs valid unique saved {name} rows.")
        if name == "interface":
            candidates[name] = {
                slot: tuple(int(row) for row in rng.permutation(np.asarray(by_slot[slot], dtype=np.int64)))
                for slot in active_slots
            }
            continue
        common_features = set.intersection(*(set(lookup_by_slot[slot]) for slot in active_slots))
        if not common_features:
            raise ValueError("Saved solid panels lack shared material-local points across active modules.")
        ordered_features = sorted(common_features)
        order = rng.permutation(len(ordered_features))
        candidates[name] = {
            slot: tuple(lookup_by_slot[slot][ordered_features[int(index)]] for index in order) for slot in active_slots
        }
    candidates["active_slots"] = active_slots
    candidates["eligible_rows_by_role"] = eligible_counts
    candidates["unique_eligible_rows_by_role"] = {
        "fluid_fields": len(candidates["fluid_fields"]),
        "interface": len(active_slots) * min(len(candidates["interface"][slot]) for slot in active_slots),
        "solid_temperature": len(active_slots) * len(candidates["solid_temperature"][active_slots[0]]),
    }
    candidates["query_cap"] = max_feasible_query_count(candidates)
    return candidates


def query_role_counts(candidates: Mapping[str, Any], q: int) -> tuple[int, int, int] | None:
    slots = tuple(candidates["active_slots"])
    modules = len(slots)
    fluid_cap = len(candidates["fluid_fields"])
    interface_cap = modules * min(len(candidates["interface"][slot]) for slot in slots)
    solid_cap = modules * len(candidates["solid_temperature"][slots[0]])
    fluid = min(fluid_cap, q - 2 * modules)
    fluid -= (fluid - q) % modules
    if fluid < 1:
        return None
    remaining = q - fluid
    if remaining < 2 * modules or remaining % modules:
        return None
    interface = min(interface_cap, remaining - modules)
    interface -= interface % modules
    solid = remaining - interface
    if solid > solid_cap:
        solid = solid_cap - (solid_cap % modules)
        interface = remaining - solid
    if interface < modules or interface > interface_cap or solid < modules or solid > solid_cap:
        return None
    return fluid, interface, solid


def max_feasible_query_count(candidates: Mapping[str, Any]) -> int:
    return next((q for q in range(8192, 0, -1) if query_role_counts(candidates, q)), 0)


def rows_by_role(candidates: Mapping[str, Any], q: int) -> dict[str, tuple[int, ...]]:
    counts = query_role_counts(candidates, q)
    if counts is None or q > int(candidates.get("query_cap", 8192)):
        raise ValueError(f"Requested total Q{q} is not supported by balanced native role rows.")
    fluid, interface, solid = counts
    slots = tuple(candidates["active_slots"])
    interface_per_module = interface // len(slots)
    solid_per_module = solid // len(slots)
    return {
        "fluid_fields": tuple(candidates["fluid_fields"][:fluid]),
        "interface": tuple(row for slot in slots for row in candidates["interface"][slot][:interface_per_module]),
        "solid_temperature": tuple(
            row for slot in slots for row in candidates["solid_temperature"][slot][:solid_per_module]
        ),
    }


def selected_ids(record: Any, role_rows: Mapping[str, Sequence[int]]) -> list[list[str]]:
    return [
        [name, str(record.output.roles[name].query_ids[int(row)])] for name, rows in role_rows.items() for row in rows
    ]


def scenario(panel: Mapping[str, Any], key: str, q: int) -> dict[str, Any]:
    role_rows = rows_by_role(panel["candidates"], q)
    role_weights = {}
    for name, rows in role_rows.items():
        weights = np.asarray(panel["record"].output.roles[name].quadrature_weights, dtype=np.float64)[list(rows)]
        role_weights[name] = {
            "query_count": len(rows),
            "weight_sha256": hashlib.sha256(np.asarray(weights, dtype="<f8").tobytes()).hexdigest(),
            "weight_sum": float(weights.sum()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
        }
    selected = [[name, int(row)] for name, rows in role_rows.items() for row in rows]
    return {
        "shape_id": f"{key}_M{panel['active_m']}_Q{q}",
        "query_count": int(q),
        "active_module_count": int(panel["active_m"]),
        "role_query_counts": {name: len(rows) for name, rows in role_rows.items()},
        "role_quadrature_weights": role_weights,
        "input_query_panel_digest": input_digest(panel["record"], role_rows),
        "selected_rows_sha256": hash_json(selected),
        "selected_query_ids_sha256": hash_json(selected_ids(panel["record"], role_rows)),
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def interleaved(
    calls: Mapping[str, Callable[[], Any]],
    *,
    device: torch.device,
    warmups: int,
    repeats: int,
    seed: int,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    labels = tuple(calls)
    rng = random.Random(seed)
    for _ in range(warmups):
        order = list(labels)
        rng.shuffle(order)
        for label in order:
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError("Thermal latency run exceeded its reserved wall-clock limit.")
            sync(device)
            with torch.inference_mode():
                output = calls[label]()
            sync(device)
            del output
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError("Thermal latency run exceeded its reserved wall-clock limit.")
    rows: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    trace = []
    for repeat in range(repeats):
        order = list(labels)
        rng.shuffle(order)
        for slot, label in enumerate(order):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError("Thermal latency run exceeded its reserved wall-clock limit.")
            sync(device)
            start = time.perf_counter()
            with torch.inference_mode():
                output = calls[label]()
            sync(device)
            elapsed = time.perf_counter() - start
            del output
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError("Thermal latency run exceeded its reserved wall-clock limit.")
            if not np.isfinite(elapsed) or elapsed <= 0.0:
                raise RuntimeError(f"Invalid synchronized duration for {label}: {elapsed}.")
            rows[label].append({"round": repeat + 1, "seconds": float(elapsed)})
            trace.append({"round": repeat + 1, "order": slot + 1, "variant": label})
    summary = {}
    for label, samples in rows.items():
        values = np.asarray([row["seconds"] for row in samples])
        summary[label] = {
            "sample_count": int(values.size),
            "p50_seconds": float(np.quantile(values, 0.50, method="linear")),
            "p90_seconds": float(np.quantile(values, 0.90, method="linear")),
            "mean_seconds": float(statistics.fmean(values.tolist())),
            "samples": samples,
        }
    return {"summary": summary, "order_trace": trace}


def renderer_latency_fields(
    *,
    model: str,
    checkpoint_sha256: str,
    access_mode: str,
    stats: Mapping[str, Any],
    input_panel_sha256: str,
    sync_group: str,
    n_cases: int = 1,
) -> dict[str, Any]:
    if model not in RENDERER_REQUIRED_MODELS or RENDERER_ACCESS_BY_MODEL[model] != access_mode:
        raise ValueError("Renderer rows need a supported Thermal control and explicit access mode.")
    return {
        "lane": "thermal",
        "model": model,
        "checkpoint_sha256": str(checkpoint_sha256),
        "access_mode": access_mode,
        "p50_ms": 1000.0 * float(stats["p50_seconds"]),
        "p90_ms": 1000.0 * float(stats["p90_seconds"]),
        "n_cases": int(n_cases),
        "n_repeats": int(stats["sample_count"]),
        "sample_count": int(stats["sample_count"]),
        "input_panel_sha256": str(input_panel_sha256),
        "sync_group": str(sync_group),
        "complete_wrapper": True,
    }


def write_renderer_latency_csv(path: Path, measurements: Sequence[Mapping[str, Any]], shape_id: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite renderer latency evidence: {path}")
    rows = [row for row in measurements if row.get("shape_id") == shape_id]
    by_model = {str(row.get("model")): row for row in rows}
    if set(by_model) != RENDERER_REQUIRED_MODELS or len(rows) != len(RENDERER_REQUIRED_MODELS):
        raise ValueError(f"Representative shape {shape_id!r} must have exactly the five renderer controls.")
    if len({row["input_panel_sha256"] for row in rows}) != 1 or len({row["sync_group"] for row in rows}) != 1:
        raise ValueError("Representative renderer controls must share one input panel and sync group.")
    if any(row["lane"] != "thermal" or row["n_cases"] != 1 or row["complete_wrapper"] is not True for row in rows):
        raise ValueError("Representative renderer rows must describe one complete Thermal case each.")
    for row in rows:
        for key in ("checkpoint_sha256", "input_panel_sha256"):
            value = str(row[key]).lower()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"Representative renderer {key} must be a SHA256 digest.")
        if row["access_mode"] != RENDERER_ACCESS_BY_MODEL[row["model"]]:
            raise ValueError(f"Renderer access mode does not match control {row['model']}.")
        if not row["sync_group"]:
            raise ValueError("Representative renderer rows need a nonempty synchronization group.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RENDERER_LATENCY_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in RENDERER_LATENCY_FIELDS} for row in rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _scalar_cover_diagnostics(values: Any) -> dict[str, int | float]:
    if not isinstance(values, Mapping):
        return {}
    captured: dict[str, int | float] = {}
    for key, value in values.items():
        if "cover_" not in str(key):
            continue
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            scalar = value.detach().cpu().item()
            captured[str(key)] = int(scalar) if not value.is_floating_point() else float(scalar)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            captured[str(key)] = int(value) if isinstance(value, (int, np.integer)) else float(value)
    return captured


@contextmanager
def capture_field_decoder_diagnostics(core: Any) -> Iterator[list[dict[str, int | float]] | None]:
    """Collect P2 field counters from every core decode, without routing maps."""
    original = getattr(core, "decode_queries", None)
    if not callable(original):
        yield None
        return
    captured: list[dict[str, int | float]] = []
    had_instance_method = "decode_queries" in vars(core)
    previous_instance_value = vars(core).get("decode_queries")

    def capture(*args: Any, **kwargs: Any) -> Any:
        context = kwargs.get("interaction_context")
        if getattr(context, "receiver_role", None) != "p2_field":
            return original(*args, **kwargs)
        call_kwargs = dict(kwargs)
        call_kwargs["return_interaction_aux"] = True
        output = original(*args, **call_kwargs)
        captured.append(_scalar_cover_diagnostics(output.get("_interaction_aux", {})))
        return output

    core.decode_queries = capture
    try:
        yield captured
    finally:
        if had_instance_method:
            core.decode_queries = previous_instance_value
        else:
            delattr(core, "decode_queries")


def _counter_rows(diagnostics: Mapping[str, Any], prefix: str) -> int | None:
    keys = [
        str(key)
        for key in diagnostics
        if str(key).startswith(prefix) and str(key).endswith(("_executed_rows", "_actual_rows"))
    ]
    executed = [key for key in keys if key.endswith("_executed_rows")]
    chosen = executed or [key for key in keys if key.endswith("_actual_rows")]
    if not chosen:
        return None
    return sum(int(diagnostics[key]) for key in chosen)


def complete_executor_row_telemetry(
    preparation_diagnostics: Mapping[str, Any],
    field_decoder_diagnostics: Sequence[Mapping[str, Any]] | None,
    *,
    expected_field_chunks: int,
) -> dict[str, Any]:
    preparation = {
        mechanism: _counter_rows(preparation_diagnostics, f"cover_prepare_{mechanism.lower()}_")
        for mechanism in ("MM", "ME", "EM")
    }
    chunks = list(field_decoder_diagnostics or ())
    complete_chunks = field_decoder_diagnostics is not None and len(chunks) == expected_field_chunks
    field_rows: dict[str, int | None] = {}
    for mechanism in ("QM", "QE"):
        values = [_counter_rows(row, f"cover_{mechanism.lower()}_") for row in chunks]
        field_rows[mechanism] = (
            sum(value for value in values if value is not None)
            if complete_chunks and all(value is not None for value in values)
            else None
        )
    combined = {**preparation, **field_rows}
    unavailable = [name for name, value in combined.items() if value is None]
    all_measured = not unavailable
    return {
        "status": "measured_complete" if all_measured else "partially_unavailable",
        "preparation_by_mechanism": preparation,
        "field_decoder_by_mechanism": field_rows,
        "by_mechanism": combined,
        "measured_total_executor_rows": sum(int(value) for value in combined.values()) if all_measured else None,
        "unavailable_mechanisms": unavailable,
        "field_decoder_chunks_observed": len(chunks),
        "field_decoder_chunks_expected": int(expected_field_chunks),
    }


__all__ = [
    "MECHANISMS",
    "QUERY_BATCH_SIZE",
    "QUERY_COUNTS",
    "RENDERER_LATENCY_FIELDS",
    "RENDERER_LATENCY_NAME",
    "RENDERER_MODEL_BY_VARIANT",
    "capture_field_decoder_diagnostics",
    "check_sha",
    "complete_executor_row_telemetry",
    "hash_json",
    "input_digest",
    "interleaved",
    "max_feasible_query_count",
    "query_role_counts",
    "renderer_latency_fields",
    "role_candidates",
    "rows_by_role",
    "scenario",
    "selected_ids",
    "sha256_file",
    "sync",
    "write_renderer_latency_csv",
]
