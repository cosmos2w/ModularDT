#!/usr/bin/env python3
"""Measure matched content-hypergraph native cost, scaling, and update throughput.

The tool accepts both retained A-D/A-H6 checkpoints or a fresh recipe pair.
Benchmark optimizer steps always use disposable model weights and AdamW state;
they are never saved as training checkpoints or used to initialize a science run.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (
    PROJECT / "src",
    PROJECT / "tools",
    PROJECT / "Case_ThermalChannel/src",
    PROJECT / "Case_WindFarm/src",
):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from content_hypergraph_evaluate import load_evaluation_components
from content_hypergraph_train import ALLOWED_UUIDS, build, canonical_sha256, engine_for, validate_recipe

from honf_runtime.unified_training import TrainingEngine

SCHEMA = "content_hypergraph_benchmark_v1"
ARCHITECTURES = ("A-D", "A-H6")
ARCHITECTURES_V2 = ("A2-D", "A2-H6")
DEFAULT_DENSITY_QUERIES = (1024, 4096, 8192, 32768)
MAX_DISPOSABLE_UPDATES = 210


@dataclass
class Arm:
    architecture: str
    model: torch.nn.Module
    provider: Any
    recipe: dict[str, Any]
    source: str
    checkpoint: Path | None
    checkpoint_sha256: str | None
    initial_model_state: dict[str, torch.Tensor]


class BenchmarkOOM(RuntimeError):
    """A fixed-size benchmark point exhausted device memory."""


def architecture_pair(arms: Mapping[str, Arm]) -> tuple[str, str]:
    pair = ARCHITECTURES_V2 if "A2-D" in arms else ARCHITECTURES
    if set(arms) != set(pair):
        raise ValueError("A benchmark requires one complete same-version direct/organized pair.")
    return pair


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int_list(value: str, *, name: str, minimum: int = 1) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list.") from exc
    if not result or any(item < minimum for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(f"{name} must contain unique integers >= {minimum}.")
    return result


def expected_optimizer_updates(train_case_count: int, effective_cases: int, epochs: int, packings: int) -> int:
    """Return the full-epoch step budget for both arms and each packing."""
    if min(train_case_count, effective_cases, epochs, packings) < 1:
        raise ValueError("Update planning requires positive cases, effective batch, epochs, and packings.")
    return math.ceil(train_case_count / effective_cases) * epochs * packings * len(ARCHITECTURES)


def validate_microbatch_packings(packings: Sequence[int], effective_cases: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in packings)
    if len(values) > 3:
        raise ValueError("At most three microbatch packings may be tested in one invocation.")
    if len(set(values)) != len(values) or any(value < 1 or value > effective_cases for value in values):
        raise ValueError(f"Microbatch packings must be unique values in 1..{effective_cases}.")
    return values


def oom_receipt(architecture: str, query_count: int, error: BaseException, *, batch_size: int = 8) -> dict[str, Any]:
    """Preserve the requested work point and explicitly forbid fallback claims."""
    return {
        "architecture": architecture,
        "requested_q_per_scene": int(query_count),
        "batch_size": int(batch_size),
        "status": "oom",
        "optimizer_updates": 0,
        "reason": str(error),
        "retry_with_smaller_batch": False,
    }


def is_oom_error(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _model_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def initialization_receipt(arms: Mapping[str, Arm], *, require_shared_match: bool) -> dict[str, Any]:
    """Bind fresh initialization and prove same-named shared tensors match."""
    state_digests = {
        architecture: _state_sha256(arm.initial_model_state)
        for architecture, arm in arms.items()
    }
    direct, organized = architecture_pair(arms)
    ad_parameters = dict(arms[direct].model.named_parameters())
    h6_parameters = dict(arms[organized].model.named_parameters())
    common = sorted(
        name for name in ad_parameters.keys() & h6_parameters.keys()
        if ad_parameters[name].shape == h6_parameters[name].shape
    )
    if not common:
        raise ValueError("A-D and A-H6 have no same-named, same-shaped shared parameter tensors.")
    ad_state, h6_state = arms[direct].initial_model_state, arms[organized].initial_model_state
    mismatched = None
    shared_digest = None
    if require_shared_match:
        mismatched = [name for name in common if not torch.equal(ad_state[name], h6_state[name])]
        if mismatched:
            raise ValueError(
                "Fresh A-D/A-H6 shared initialization differs for named tensors: "
                + ", ".join(mismatched[:8])
            )
        if state_digests[direct] == state_digests[organized]:
            raise ValueError("Fresh A-D and A-H6 complete initialization digests must be distinct.")
        shared_digest = _state_sha256({name: ad_state[name] for name in common})
    return {
        "arm_state_sha256": state_digests,
        "common_named_parameter_sha256": shared_digest,
        "common_named_parameter_count": len(common),
        "shared_parameters_match": True if require_shared_match else None,
    }


def _tensor_bytes(value: torch.Tensor | None) -> int:
    return 0 if value is None else int(value.numel() * value.element_size())


def _unique_id_counts_by_row(values: Any) -> list[int]:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    rows = np.asarray(values)
    if rows.ndim != 2:
        raise ValueError("Native receiver identity arrays must have shape [B,Q].")
    return [int(np.unique(row).size) for row in rows]


def _numpy_view(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def packing_independent_query_sha256(microbatches: Sequence[Any], dataset: str) -> str:
    """Hash ordered case query content without depending on microbatch boundaries."""
    digest = hashlib.sha256()
    for batch in microbatches:
        receivers = batch.receivers
        case_keys = tuple(batch.case_keys)
        if dataset == "wind":
            fields = ("coordinates_D", "features", "flat_indices", "role_ids")
        elif dataset == "thermal":
            fields = ("fluid_xy", "local_query_points")
        else:
            raise ValueError(f"Unknown query-stream dataset {dataset!r}.")
        arrays = {name: _numpy_view(getattr(receivers, name)) for name in fields}
        local_indices = None
        if dataset == "wind" and receivers.local_batch_indices is not None:
            local_indices = {
                "batch": _numpy_view(receivers.local_batch_indices),
                "query": _numpy_view(receivers.local_query_indices),
                "module": _numpy_view(receivers.local_module_indices),
            }
        if dataset == "thermal":
            response_fields = ("response_fluid_xy", "response_local_query_points")
            for name in response_fields:
                value = getattr(receivers, name, None)
                if value is not None:
                    array = np.ascontiguousarray(_numpy_view(value))
                    digest.update(name.encode("ascii") + b"\0")
                    digest.update(array.tobytes())
        for row, case_key in enumerate(case_keys):
            digest.update(str(case_key).encode("utf-8") + b"\0")
            for name, values in arrays.items():
                value = np.ascontiguousarray(values[row])
                metadata = json.dumps(
                    {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")
                digest.update(len(metadata).to_bytes(4, "little"))
                digest.update(metadata)
                digest.update(value.tobytes())
            if local_indices is not None:
                selected = local_indices["batch"] == row
                for name in ("query", "module"):
                    value = np.ascontiguousarray(local_indices[name][selected])
                    digest.update(name.encode("ascii") + b"\0")
                    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
                    digest.update(str(value.dtype).encode("ascii") + b"\0")
                    digest.update(value.tobytes())
    return digest.hexdigest()


def provider_cache_stats(provider: Any) -> Mapping[str, Any]:
    cache = getattr(provider, "catalogue_cache", None)
    summary = getattr(cache, "summary", None)
    if callable(summary):
        return dict(summary())
    return {"available": False}


def _tree_tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return _tensor_bytes(value)
    if isinstance(value, Mapping):
        return sum(_tree_tensor_bytes(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tree_tensor_bytes(child) for child in value)
    return 0


def parameter_inventory(model: torch.nn.Module) -> dict[str, int]:
    all_parameters = tuple(model.parameters())
    trainable = tuple(parameter for parameter in all_parameters if parameter.requires_grad)
    return {
        "parameter_count": sum(parameter.numel() for parameter in all_parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "parameter_bytes": sum(_tensor_bytes(parameter) for parameter in all_parameters),
        "trainable_parameter_bytes": sum(_tensor_bytes(parameter) for parameter in trainable),
    }


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(_tree_tensor_bytes(state) for state in optimizer.state.values())


def _provider_identity_without_arm(identity: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(identity)
    result.pop("architecture", None)
    result.pop("model_config", None)
    recipe = result.get("recipe")
    if isinstance(recipe, Mapping):
        recipe = dict(recipe)
        recipe.pop("architecture", None)
        result["recipe"] = recipe
    adapter_state = result.get("adapter_state")
    if isinstance(adapter_state, Mapping):
        adapter_state = dict(adapter_state)
        adapter_state.pop("core", None)
        result["adapter_state"] = adapter_state
    return result


def _arm_recipe(recipe: Mapping[str, Any], architecture: str) -> dict[str, Any]:
    result = dict(recipe)
    result["architecture"] = architecture
    return validate_recipe(result)


def comparison_recipes(
    ad: Mapping[str, Any], h6: Mapping[str, Any], *, thermal_reader128_revision: bool = False,
    typed_centroid_revision: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare recipes, allowing only a separately named single-field revision."""
    if thermal_reader128_revision and typed_centroid_revision:
        raise ValueError("Thermal reader128 and typed-centroid R1 revisions are mutually exclusive.")
    normalized = [json.loads(json.dumps(dict(recipe))) for recipe in (ad, h6)]
    for recipe in normalized:
        recipe.pop("architecture", None)
    if thermal_reader128_revision:
        if (
            ad.get("schema") != "content_hypergraph_recipe_v1"
            or h6.get("schema") != "content_hypergraph_recipe_v1"
            or ad.get("dataset") != "thermal" or h6.get("dataset") != "thermal"
            or ad.get("protocol") != "development" or h6.get("protocol") != "development"
            or ad.get("architecture") != "A-D" or h6.get("architecture") != "A-H6"
            or ad.get("model", {}).get("local_hidden_dim") != 64
            or h6.get("model", {}).get("local_hidden_dim") != 128
        ):
            raise ValueError("Reader128 revision costs require Thermal v1 development A-D64 versus A-H6-128.")
        normalized[1]["model"]["local_hidden_dim"] = 64
    if typed_centroid_revision:
        dataset = ad.get("dataset")
        if (
            ad.get("schema") != "content_hypergraph_recipe_v2"
            or h6.get("schema") != "content_hypergraph_recipe_v2"
            or dataset not in ("thermal", "wind") or h6.get("dataset") != dataset
            or ad.get("protocol") != "development" or h6.get("protocol") != "development"
            or ad.get("architecture") != "A2-D" or h6.get("architecture") != "A2-H6"
        ):
            raise ValueError(
                "Typed-centroid R1 requires a same-dataset Thermal or Wind development A2-D/A2-H6 v2 pair."
            )
        validate_recipe(dict(ad))
        validate_recipe(dict(h6))
        ad_model, h6_model = ad.get("model"), h6.get("model")
        if not isinstance(ad_model, Mapping) or not isinstance(h6_model, Mapping):
            raise ValueError("Typed-centroid R1 requires explicit model mappings on both recipes.")
        if any("loss" in str(name).lower() or str(name).lower() == "objective"
               for recipe in (ad, h6) for name in recipe):
            raise ValueError("Typed-centroid R1 cannot change provider-owned losses or objectives.")
        if "query_edge_geometry_mode" in ad_model:
            raise ValueError("The A2-D comparator must retain its default absent centroid-mode key.")
        if h6_model.get("query_edge_geometry_mode") != "typed_w_centroids_v1":
            raise ValueError("The A2-H6 checkpoint must bind typed_w_centroids_v1 explicitly.")
        if dataset == "wind":
            from windfarm.training.content_hypergraph_task import _validate_recipe_v2

            _validate_recipe_v2(ad)
            _validate_recipe_v2(h6)
        else:
            thermal_model_fields = {
                "module_input_dim": 8,
                "environment_input_dim": 8,
                "coordinate_dim": 2,
                "query_input_dim": 0,
                "operating_input_dim": 14,
                "output_dim": 3,
                "context_dim": 128,
                "num_heads": 4,
                "context_depth": 2,
                "num_hyperedges": 6,
                "thermal_response_rank": 4,
                "local_hidden_dim": 128,
                "dropout": 0.0,
                "init_seed": 0,
            }
            expected_ad_model = thermal_model_fields
            expected_h6_model = {**thermal_model_fields, "query_edge_geometry_mode": "typed_w_centroids_v1"}
            if dict(ad_model) != expected_ad_model or dict(h6_model) != expected_h6_model:
                raise ValueError("Thermal typed-centroid R1 requires the sealed v2 dimensions and H6-only mode.")
            if (
                ad.get("model_family") != "honf_content_hypergraph"
                or h6.get("model_family") != "honf_content_hypergraph"
                or ad.get("adapter_schema") != "thermal_content_hypergraph_adapter_v2"
                or h6.get("adapter_schema") != "thermal_content_hypergraph_adapter_v2"
                or ad.get("optimizer") != {
                    "name": "AdamW", "weight_decay": 1.0e-4, "bias_and_norm_weight_decay": 0.0,
                }
                or h6.get("optimizer") != {
                    "name": "AdamW", "weight_decay": 1.0e-4, "bias_and_norm_weight_decay": 0.0,
                }
            ):
                raise ValueError("Thermal typed-centroid R1 must retain the sealed v2 provider and optimizer bindings.")
        normalized[1]["model"].pop("query_edge_geometry_mode", None)
    if normalized[0] != normalized[1]:
        raise ValueError("A-D and A-H6 recipes differ beyond their declared architecture arm/revision.")
    return normalized[0], normalized[1]


def load_arms(
    *,
    dataset: str,
    device: str,
    recipe_path: Path | None,
    checkpoint_ad: Path | None,
    checkpoint_ah6: Path | None,
    thermal_reader128_revision: bool = False,
    typed_centroid_revision: bool = False,
) -> tuple[dict[str, Arm], dict[str, Any], dict[str, Any]]:
    """Load and bind one same-dataset pair without touching checkpoint files."""
    if recipe_path is not None:
        if thermal_reader128_revision or typed_centroid_revision:
            raise ValueError("Named architecture revisions require retained checkpoints, not fresh recipe probes.")
        if checkpoint_ad is not None or checkpoint_ah6 is not None:
            raise ValueError("Use either --recipe or the two checkpoint paths, not both.")
        base_recipe = validate_recipe(json.loads(recipe_path.read_text()))
        if base_recipe["dataset"] != dataset:
            raise ValueError("--dataset differs from the fresh recipe dataset.")
        arms: dict[str, Arm] = {}
        pair = ARCHITECTURES_V2 if base_recipe["schema"] == "content_hypergraph_recipe_v2" else ARCHITECTURES
        for architecture in pair:
            arm_recipe = _arm_recipe(base_recipe, architecture)
            model, provider = build(arm_recipe, device=device)
            arms[architecture] = Arm(
                architecture, model, provider, arm_recipe, "fresh_recipe", None, None,
                _model_state_cpu(model),
            )
    else:
        if checkpoint_ad is None or checkpoint_ah6 is None:
            raise ValueError("Checkpoint mode requires both --checkpoint-ad and --checkpoint-ah6.")
        arms = {}
        for organized, path in ((False, checkpoint_ad), (True, checkpoint_ah6)):
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoint_hash = sha256_file(path)
            model, provider, payload = load_evaluation_components(path, device)
            identity = payload.get("experiment_identity", {})
            recipe = validate_recipe(identity.get("recipe", {}))
            pair = ARCHITECTURES_V2 if recipe["schema"] == "content_hypergraph_recipe_v2" else ARCHITECTURES
            architecture = pair[int(organized)]
            if recipe["dataset"] != dataset or recipe["architecture"] != architecture:
                raise ValueError(f"{path} is not a {dataset} {architecture} checkpoint.")
            if payload.get("arm") != architecture:
                raise ValueError(f"Checkpoint arm field differs from the requested {architecture} arm.")
            arms[architecture] = Arm(
                architecture, model, provider, recipe, "retained_checkpoint", path, checkpoint_hash,
                _model_state_cpu(model),
            )

    direct, organized = architecture_pair(arms)
    ad, h6 = arms[direct], arms[organized]
    comparison_recipes(
        ad.recipe, h6.recipe,
        thermal_reader128_revision=thermal_reader128_revision,
        typed_centroid_revision=typed_centroid_revision,
    )
    ad_identity = dict(ad.provider.identity_payload())
    h6_identity = dict(h6.provider.identity_payload())
    bindings = [_provider_identity_without_arm(identity) for identity in (ad_identity, h6_identity)]
    if (thermal_reader128_revision or typed_centroid_revision) and all("recipe" in binding for binding in bindings):
        for binding, recipe in zip(bindings, comparison_recipes(
            ad.recipe, h6.recipe,
            thermal_reader128_revision=thermal_reader128_revision,
            typed_centroid_revision=typed_centroid_revision,
        ), strict=True):
            binding["recipe"] = recipe
    if bindings[0] != bindings[1]:
        raise ValueError("A-D and A-H6 provider identities do not bind the same selected TRAIN data/transforms.")
    for arm in arms.values():
        arm.model.eval()
    initialization = initialization_receipt(arms, require_shared_match=recipe_path is not None)
    return arms, {direct: ad_identity, organized: h6_identity}, initialization


def choose_low_high_train_keys(provider: Any, dataset: str) -> dict[str, dict[str, Any]]:
    """Choose deterministic low/high source-count TRAIN rows from input metadata."""
    if dataset == "thermal":
        inventory = []
        for case in provider.training_cases:
            structure = case["structure"]
            present = structure["module_present"]
            if torch.is_tensor(present):
                present = present.detach().cpu().numpy()
            count = int(np.count_nonzero(np.asarray(present) > 0.5))
            inventory.append((str(case["case_id"]), count))
    else:
        rows = np.asarray(provider.train_rows, dtype=np.int64)
        counts = np.asarray(provider.view.metadata["n_turbines"], dtype=np.int64)
        inventory = [(int(row), int(counts[int(row)])) for row in rows]
    if not inventory:
        raise ValueError("Provider exposed no input-selected TRAIN fixtures.")
    low_count = min(count for _, count in inventory)
    high_count = max(count for _, count in inventory)
    if low_count == high_count:
        raise ValueError("The selected TRAIN profile has no distinct low/high source-count fixtures.")
    low_key = next(key for key, count in inventory if count == low_count)
    high_key = next(key for key, count in reversed(inventory) if count == high_count)
    return {
        "low_m": {"case_key": low_key, "active_modules": low_count},
        "high_m": {"case_key": high_key, "active_modules": high_count},
    }


def native_train_batch(provider: Any, case_key: Any):
    parameters = inspect.signature(provider.native_evaluation_batches).parameters
    kwargs = {"split": "train", "case_keys": (case_key,)}
    if "microbatch_size" in parameters:
        kwargs["microbatch_size"] = 1
    iterator = iter(provider.native_evaluation_batches(**kwargs))
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise ValueError("Provider did not return the selected native TRAIN fixture.") from exc
    if len(batch.case_keys) != 1 or batch.case_keys[0] != case_key:
        raise ValueError("Native TRAIN fixture loader changed the requested case identity.")
    # Targets are irrelevant to inference timing and can be sizable CUDA
    # tensors. Return only the target-free inputs consumed by prediction.
    return SimpleNamespace(
        scene_inputs=batch.scene_inputs,
        receivers=batch.receivers,
        case_keys=batch.case_keys,
    )


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_baseline(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"baseline_allocated_bytes": None, "baseline_reserved_bytes": None,
                "baseline_free_bytes": None, "device_total_bytes": None}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "baseline_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "baseline_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "baseline_free_bytes": int(free),
        "device_total_bytes": int(total),
    }


def _timed_once(function: Callable[[], Any], device: torch.device) -> tuple[Any, dict[str, Any]]:
    _device_sync(device)
    baseline = _memory_baseline(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    value = function()
    _device_sync(device)
    elapsed = time.perf_counter() - started
    sample = {
        "elapsed_seconds": elapsed,
        **baseline,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
    }
    return value, sample


def check_memory_headroom(device: torch.device) -> dict[str, int | None]:
    """Refuse a GPU matrix that cannot preserve the specified free-space margin."""
    memory = _memory_baseline(device)
    if device.type != "cuda":
        return memory
    required = max(int(memory["device_total_bytes"] * 0.20), 4 * 1024**3)
    memory["minimum_required_headroom_bytes"] = required
    if int(memory["baseline_free_bytes"]) < required:
        raise RuntimeError(
            f"Insufficient pre-measurement GPU headroom: free={memory['baseline_free_bytes']} "
            f"required={required} bytes. No benchmark scope was started."
        )
    return memory


def validate_device_binding(device: torch.device) -> str | None:
    """Restrict CUDA runs to the two explicitly authorized physical UUIDs."""
    if device.type != "cuda":
        return None
    if device.index not in (None, 0):
        raise ValueError("CUDA benchmarking must use logical cuda:0 for one exposed authorized GPU.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in ALLOWED_UUIDS:
        raise ValueError("CUDA benchmarking requires CUDA_VISIBLE_DEVICES to be one allowed physical GPU UUID.")
    return visible


def _summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    durations = [float(item["elapsed_seconds"]) for item in samples]
    return {
        "repeat_count": len(samples),
        "elapsed_median_seconds": float(np.median(durations)),
        "elapsed_min_seconds": float(np.min(durations)),
        "elapsed_max_seconds": float(np.max(durations)),
        "samples": list(samples),
    }


def _core_for(arm: Arm) -> torch.nn.Module:
    return arm.model.core if hasattr(arm.model, "core") else arm.model


def _prepare_decode(arm: Arm, batch: Any) -> tuple[Callable[[], Any], dict[str, Any]]:
    provider, model = arm.provider, arm.model
    scene = provider.make_scene(batch.scene_inputs)
    if arm.recipe["dataset"] == "thermal":
        prepared = model.prepare_native(
            scene.structure,
            batch.receivers.fluid_xy,
            local_query_points=batch.receivers.local_query_points,
            ntheta=batch.receivers.ntheta,
            scene_key=scene.case_ids,
        )
        organization, query = prepared.organization, prepared.query_batch
        controls = batch.receivers.heat
        call = lambda: model.core.decode(organization, query, controls=controls)
        details = {
            "requested_q": int(batch.receivers.fluid_xy.shape[1]),
            "native_union_rows_per_scene": [int(value) for value in prepared.union.unique_counts.detach().cpu().tolist()],
            "unique_neural_receiver_union_rows": int(prepared.union.unique_counts.sum().detach().cpu()),
            "executed_receiver_rows": int(prepared.query_batch.batch_size * prepared.query_batch.num_queries),
            "local_rows": int(query.local_weights.numel()),
            "prepared_hyperedges": organization.num_hyperedges,
            "source_factor_bytes": _tensor_bytes(organization.source_coefficients),
            "cached_projection_bytes": sum(_tensor_bytes(value) for value in (
                organization.edge_keys, organization.edge_values,
                organization.direct_source_keys, organization.direct_source_values,
            )),
        }
    else:
        query = provider.adapter.make_query_batch(scene, batch.receivers)
        organization = model.prepare(scene)
        call = lambda: model.decode(organization, query)
        unique_rows = _unique_id_counts_by_row(query.query_ids)
        details = {
            "requested_q": query.num_queries,
            "native_union_rows_per_scene": unique_rows,
            "unique_neural_receiver_union_rows": sum(unique_rows),
            "local_rows": int(query.local_weights.numel()),
            "prepared_hyperedges": organization.num_hyperedges,
            "source_factor_bytes": _tensor_bytes(organization.source_coefficients),
            "cached_projection_bytes": sum(_tensor_bytes(value) for value in (
                organization.edge_keys, organization.edge_values,
                organization.direct_source_keys, organization.direct_source_values,
            )),
        }
    return call, details


def _move_prediction_outputs_to_cpu(prediction: Any, dataset: str) -> int:
    """Synchronously return native prediction arrays without copying prepared state."""
    if dataset == "thermal":
        output = {
            "native_main": prediction.native_main,
            "response_main": prediction.response_main,
        }
    else:
        output = {
            name: getattr(prediction, name)
            for name in ("main_mps", "normalized_residual", "residual_mps", "profile_mps", "role_ids")
        }
    moved = torch.utils._pytree.tree_map(
        lambda value: value.detach().cpu() if torch.is_tensor(value) else value,
        output,
    )
    return _tree_tensor_bytes(moved)


def _request_to_device(batch: Any, device: torch.device | str) -> Any:
    """Transfer prescribed tensors without changing native dataclass structure."""
    def move(value):
        if torch.is_tensor(value):
            return value.detach().to(device=device)
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{item.name: move(getattr(value, item.name)) for item in fields(value) if item.init})
        if isinstance(value, Mapping):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value
    return SimpleNamespace(scene_inputs=move(batch.scene_inputs), receivers=move(batch.receivers),
                           case_keys=batch.case_keys)


def _host_request(batch: Any) -> Any:
    """Keep prescribed native inputs on the host until the complete call begins."""
    return _request_to_device(batch, "cpu")


def _run_native(arm: Arm, batch: Any, *, return_cpu: bool = False) -> tuple[Any, Mapping[str, Any]]:
    provider = arm.provider
    batch = _request_to_device(batch, next(arm.model.parameters()).device)
    scene = provider.make_scene(batch.scene_inputs)
    prediction, auxiliary = provider.predict_native(
        arm.model, scene, batch.receivers, arm.architecture, "joint", epoch=0, temperature=1.0,
    )
    work = dict(provider.work_counts(batch, prediction, auxiliary))
    returned_bytes = _move_prediction_outputs_to_cpu(prediction, arm.recipe["dataset"]) if return_cpu else 0
    return (prediction, auxiliary), {"work_counts": work, "cpu_return_bytes": returned_bytes}


def _alternating_scope(
    arms: Mapping[str, Arm],
    calls: Mapping[str, Callable[[], Any]],
    describe: Callable[[str, Any], Mapping[str, Any]],
    *,
    device: torch.device,
    scope_name: str,
    fixture_name: str,
) -> dict[str, Any]:
    samples: dict[str, list[dict[str, Any]]] = {architecture: [] for architecture in architecture_pair(arms)}
    for repetition in range(7):
        pair = architecture_pair(arms)
        order = pair if repetition % 2 == 0 else tuple(reversed(pair))
        for architecture in order:
            try:
                with torch.no_grad():
                    result, measurement = _timed_once(calls[architecture], device)
            except BaseException as exc:
                if is_oom_error(exc):
                    raise BenchmarkOOM(
                        f"OOM at fixed fixture={fixture_name}, scope={scope_name}, arm={architecture}; batch/Q unchanged."
                    ) from exc
                raise
            measurement["warmup"] = repetition < 2
            if repetition >= 2:
                measurement.update(describe(architecture, result))
                samples[architecture].append(measurement)
            del result
    return {
        "fixture": fixture_name,
        "scope": scope_name,
        "warmups_per_arm": 2,
        "alternating_order": True,
        "arms": {architecture: _summary(samples[architecture]) for architecture in architecture_pair(arms)},
    }


def _native_description(arm: Arm, result: Any, batch: Any) -> Mapping[str, Any]:
    (prediction, auxiliary), details = result
    dataset = arm.recipe["dataset"]
    if dataset == "thermal":
        prepared = prediction.native_prepared
        output_bytes = _tree_tensor_bytes(prediction.native_main)
        if prediction.response_main is not None:
            output_bytes += _tree_tensor_bytes(prediction.response_main)
        details = {
            **details,
            "unique_neural_receiver_union_rows": int(prepared.union.unique_counts.sum().detach().cpu()),
            "requested_fluid_q": int(prepared.fluid_xy.shape[1]),
            "executed_receiver_rows": int(prepared.query_batch.batch_size * prepared.query_batch.num_queries),
            "padded_edge_rows": int(prepared.organization.scene.batch_size * prepared.query_batch.num_queries
                                     * prepared.organization.num_hyperedges),
            "active_edge_rows": int(prepared.organization.scene.batch_size * prepared.query_batch.num_queries
                                     * prepared.organization.num_hyperedges),
            "direct_source_rows": (
                int(prepared.query_batch.batch_size * prepared.query_batch.num_queries
                    * prepared.organization.scene.num_modules)
                if arm.architecture in {"A-D", "A2-D"} else 0
            ),
            "local_rows": int(prepared.query_batch.local_weights.numel()),
            "factor_buffer_bytes": _tensor_bytes(prepared.organization.source_coefficients),
            "output_bytes": output_bytes,
        }
    else:
        core_output = auxiliary["core_output"]
        organization = auxiliary["organization"]
        unique_rows = _unique_id_counts_by_row(batch.receivers.flat_indices)
        details = {
            **details,
            "unique_neural_receiver_union_rows": sum(unique_rows),
            "unique_neural_receiver_union_rows_per_scene": unique_rows,
            "executed_receiver_rows": int(prediction.main_mps.shape[0] * prediction.main_mps.shape[1]),
            "requested_fluid_q": int(prediction.main_mps.shape[1]),
            "padded_edge_rows": int(core_output.work_counts["edge_rows"]),
            "active_edge_rows": int(core_output.work_counts["edge_rows"]),
            "local_rows": int(core_output.work_counts["local_rows"]),
            "factor_buffer_bytes": _tensor_bytes(organization.source_coefficients),
            "output_bytes": _tree_tensor_bytes({
                "main_mps": prediction.main_mps,
                "normalized_residual": prediction.normalized_residual,
                "residual_mps": prediction.residual_mps,
            }),
        }
    return details


def run_native_matrix(
    arms: Mapping[str, Arm], fixture_batches: Mapping[str, Mapping[str, Any]], device: torch.device,
) -> list[dict[str, Any]]:
    records = []
    for fixture_name in ("low_m", "high_m"):
        batches = {architecture: fixture_batches[architecture][fixture_name] for architecture in architecture_pair(arms)}
        prepared = {
            architecture: _prepare_decode(arms[architecture], batches[architecture])
            for architecture in architecture_pair(arms)
        }
        decode_calls = {architecture: prepared[architecture][0] for architecture in architecture_pair(arms)}
        decode_details = {architecture: prepared[architecture][1] for architecture in architecture_pair(arms)}

        def describe_decode(architecture: str, result: Any, details=decode_details) -> Mapping[str, Any]:
            work = result.work_counts or {}
            return {
                **details[architecture],
                "output_bytes": _tree_tensor_bytes({
                    "nonlinear_field": result.nonlinear_field,
                    "thermal_temperature": result.thermal_temperature,
                }),
                "factor_buffer_bytes": details[architecture]["source_factor_bytes"],
                "padded_edge_rows": int(work.get("edge_rows", 0)),
                "active_edge_rows": int(work.get("edge_rows", 0)),
                "direct_source_rows": int(work.get("direct_source_rows", 0)),
                "local_rows": int(work.get("local_rows", details[architecture]["local_rows"])),
                "work_counts": work,
            }

        records.append(_alternating_scope(
            arms, decode_calls, describe_decode,
            device=device, scope_name="prepared_tensor_decode", fixture_name=fixture_name,
        ))
        del decode_calls, prepared
        for scope_name, return_cpu in (("fresh_scene_gpu_native_forward", False),
                                       ("complete_native_application_cpu_return", True)):
            scoped_batches = ({architecture: _host_request(batch) for architecture, batch in batches.items()}
                              if return_cpu else batches)
            calls = {
                architecture: (lambda architecture=architecture, batch=scoped_batches[architecture], return_cpu=return_cpu:
                    _run_native(arms[architecture], batch, return_cpu=return_cpu))
                for architecture in architecture_pair(arms)
            }

            def describe_native(architecture: str, result: Any, bound_batches=batches) -> Mapping[str, Any]:
                return _native_description(arms[architecture], result, bound_batches[architecture])

            records.append(_alternating_scope(
                arms, calls,
                describe_native,
                device=device, scope_name=scope_name, fixture_name=fixture_name,
            ))
    return records


def _repeat_first_axis(value: Any, count: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == 1:
            return value.repeat((count,) + (1,) * (value.ndim - 1))
        return value
    if isinstance(value, np.ndarray):
        if value.ndim and value.shape[0] == 1:
            return np.repeat(value, count, axis=0)
        return value
    if isinstance(value, Mapping):
        return {key: _repeat_first_axis(child, count) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_repeat_first_axis(child, count) for child in value)
    if isinstance(value, list):
        return [_repeat_first_axis(child, count) for child in value]
    return value


def _repeat_query_axis(value: Any, query_count: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim < 2:
            return value
        indices = torch.arange(query_count, device=value.device) % value.shape[1]
        return value.index_select(1, indices)
    if isinstance(value, np.ndarray):
        if value.ndim < 2:
            return value
        indices = np.arange(query_count, dtype=np.int64) % value.shape[1]
        return np.take(value, indices, axis=1)
    return value


def _stress_request(arm: Arm, native_batch: Any, *, batch_size: int, query_count: int):
    """Build a target-free repeated-input stress request from one high-M TRAIN row."""
    dataset = arm.recipe["dataset"]
    if dataset == "thermal":
        from channelthermal.training.content_hypergraph_task import (
            ThermalContentReceivers,
            ThermalContentSceneInputs,
        )

        original = native_batch.scene_inputs
        ids = tuple(f"{native_batch.case_keys[0]}#stress-{index}" for index in range(batch_size))
        scene_inputs = ThermalContentSceneInputs(
            structure=_repeat_first_axis(original.structure, batch_size),
            response_structure=None,
            case_ids=ids,
        )
        receivers = ThermalContentReceivers(
            fluid_xy=_repeat_query_axis(_repeat_first_axis(native_batch.receivers.fluid_xy, batch_size), query_count),
            local_query_points=_repeat_first_axis(native_batch.receivers.local_query_points, batch_size),
            heat=_repeat_first_axis(native_batch.receivers.heat, batch_size),
            ntheta=native_batch.receivers.ntheta,
        )
    else:
        from dataclasses import replace as dataclass_replace

        from windfarm.content_hypergraph import WindContentReceivers

        input_row = native_batch.scene_inputs[0]
        scene_inputs = tuple(
            dataclass_replace(input_row, case_name=f"{input_row.case_name}#stress-{index}")
            for index in range(batch_size)
        )
        coordinates = _repeat_query_axis(_repeat_first_axis(native_batch.receivers.coordinates_D, batch_size), query_count)
        features = _repeat_query_axis(_repeat_first_axis(native_batch.receivers.features, batch_size), query_count)
        flat_indices = _repeat_query_axis(_repeat_first_axis(native_batch.receivers.flat_indices, batch_size), query_count)
        role_ids = _repeat_query_axis(_repeat_first_axis(native_batch.receivers.role_ids, batch_size), query_count)
        local_batch, local_query, local_module = [], [], []
        for batch_index in range(batch_size):
            query_rows, module_rows = arm.provider.adapter.packed_local_pair_indices(
                scene_inputs[batch_index], coordinates[batch_index],
            )
            local_batch.append(np.full(query_rows.size, batch_index, dtype=np.int64))
            local_query.append(query_rows)
            local_module.append(module_rows)
        concatenate = lambda values: np.concatenate(values) if values else np.empty(0, dtype=np.int64)
        receivers = WindContentReceivers(
            coordinates_D=coordinates,
            features=features,
            flat_indices=flat_indices,
            role_ids=role_ids,
            scene_keys=tuple(item.scene_key for item in scene_inputs),
            local_batch_indices=concatenate(local_batch),
            local_query_indices=concatenate(local_query),
            local_module_indices=concatenate(local_module),
        )
    return scene_inputs, receivers


def _native_loss_for_stress(arm: Arm, prediction: Any) -> torch.Tensor:
    if arm.recipe["dataset"] == "thermal":
        terms = [prediction.native_main["pred_field"].square().mean(),
                 prediction.native_main["pred_interface"].square().mean(),
                 prediction.native_main["pred_internal_temperature"].square().mean()]
        return sum(terms)
    return prediction.main_mps.square().mean()


def run_density_backward(arms: Mapping[str, Arm], high_batches: Mapping[str, Any], device: torch.device) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for architecture in architecture_pair(arms):
        arm = arms[architecture]
        for query_count in DEFAULT_DENSITY_QUERIES[:3]:
            request = _stress_request(arm, high_batches[architecture], batch_size=8, query_count=query_count)

            def run_backward(request=request, arm=arm, query_count=query_count):
                arm.model.zero_grad(set_to_none=True)
                arm.model.train(True)
                scene = arm.provider.make_scene(request[0])
                prediction, auxiliary = arm.provider.predict_native(
                    arm.model, scene, request[1], arm.architecture, "joint", epoch=0, temperature=1.0,
                )
                loss = _native_loss_for_stress(arm, prediction)
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError("Density stress objective is nonfinite.")
                loss.backward()
                work = dict(arm.provider.work_counts(
                    type("BatchView", (), {"receivers": request[1], "scene_inputs": request[0]})(),
                    prediction,
                    auxiliary,
                ))
                if arm.recipe["dataset"] == "thermal":
                    prepared = prediction.native_prepared
                    work.update({
                        "unique_neural_receiver_union_rows": int(prepared.union.unique_counts.sum().detach().cpu()),
                        "executed_receiver_rows": int(
                            prepared.query_batch.batch_size * prepared.query_batch.num_queries
                        ),
                        "padded_edge_rows": int(prepared.organization.scene.batch_size * prepared.query_batch.num_queries
                                                 * prepared.organization.num_hyperedges),
                        "active_edge_rows": int(prepared.organization.scene.batch_size * prepared.query_batch.num_queries
                                                 * prepared.organization.num_hyperedges),
                        "direct_source_rows": (
                            int(prepared.query_batch.batch_size * prepared.query_batch.num_queries
                                * prepared.organization.scene.num_modules)
                            if arm.architecture in {"A-D", "A2-D"} else 0
                        ),
                        "local_rows": int(prepared.query_batch.local_weights.numel()),
                        "factor_buffer_bytes": _tensor_bytes(prepared.organization.source_coefficients),
                        "output_bytes": _tree_tensor_bytes(prediction.native_main),
                    })
                else:
                    core_output = auxiliary["core_output"]
                    organization = auxiliary["organization"]
                    unique_rows = _unique_id_counts_by_row(request[1].flat_indices)
                    work.update({
                        "unique_neural_receiver_union_rows": sum(unique_rows),
                        "unique_neural_receiver_union_rows_per_scene": unique_rows,
                        "executed_receiver_rows": 8 * query_count,
                        "padded_edge_rows": core_output.work_counts["edge_rows"],
                        "active_edge_rows": core_output.work_counts["edge_rows"],
                        "local_rows": core_output.work_counts["local_rows"],
                        "factor_buffer_bytes": _tensor_bytes(organization.source_coefficients),
                        "output_bytes": _tree_tensor_bytes(prediction.main_mps),
                    })
                arm.model.zero_grad(set_to_none=True)
                return {"loss": float(loss.detach().cpu()), "work": work}

            try:
                workspace_warmup_calls = 0
                if device.type == "cuda" and query_count == DEFAULT_DENSITY_QUERIES[0]:
                    # Backward libraries can retain a live workspace that
                    # empty_cache cannot release. Establish it before the
                    # first recorded point, with no optimizer update.
                    run_backward()
                    workspace_warmup_calls = 1
                # A previous arm's peak must not become the next arm's
                # reserved-memory result. Keep the same live models/fixtures
                # resident, but discard unused allocator blocks outside the
                # timed region before every fixed-work point.
                if device.type == "cuda":
                    _device_sync(device)
                    torch.cuda.empty_cache()
                result, memory = _timed_once(run_backward, device)
            except BaseException as exc:
                arm.model.zero_grad(set_to_none=True)
                if is_oom_error(exc):
                    records.append(oom_receipt(architecture, query_count, exc))
                    raise BenchmarkOOM(
                        f"OOM at fixed B=8,Q={query_count},arm={architecture}; no lower-B retry was made."
                    ) from exc
                raise
            arm.model.eval()
            records.append({
                "architecture": architecture,
                "dataset": arm.recipe["dataset"],
                "scope": "native_forward_backward_target_free_repeated_high_m_input",
                "requested_q_per_scene": query_count,
                "batch_size": 8,
                "unique_source_count": (
                    int(torch.count_nonzero(
                        high_batches[architecture].scene_inputs.structure["module_present"][0] > 0.5
                    ).detach().cpu())
                    if arm.recipe["dataset"] == "thermal"
                    else int(np.count_nonzero(high_batches[architecture].scene_inputs[0].module_present > 0.5))
                ),
                "status": "completed",
                "optimizer_updates": 0,
                "objective": "mean_squared_native_output; no targets passed to model/provider",
                "loss_value": result["loss"],
                "timing_and_memory": memory,
                "allocator_cache_policy": "unused CUDA allocator blocks cleared before this point, outside timing; live resident tensors retained",
                "resident_baseline_scope": "both retained models/providers, shared native fixtures and current target-free request",
                "workspace_warmup_forward_backward_calls": workspace_warmup_calls,
                "work": result["work"],
            })
    return records


def _stress_core_probe(
    arm: Arm, native_batch: Any, *, query_count: int, backward: bool = False,
) -> Mapping[str, Any]:
    """Target-free core-only density point, independent of native grid extraction."""
    dataset = arm.recipe["dataset"]
    batch_size = 8
    scene_inputs, receivers = _stress_request(arm, native_batch, batch_size=batch_size, query_count=8192)
    scene = arm.provider.make_scene(scene_inputs)
    core = _core_for(arm)
    if dataset == "thermal":
        scene = arm.model.scene_batch(scene.structure, scene_key=scene.case_ids)
        # Generate new in-frame coordinates: this is coordinate-density stress,
        # not a physical-grid accuracy test and not a native-union benchmark.
        generator = torch.Generator(device=scene.module_features.device)
        generator.manual_seed(8137 + query_count)
        unit = torch.rand(
            (batch_size, query_count, core.coordinate_dim),
            dtype=scene.module_coordinates.dtype,
            device=scene.module_coordinates.device,
            generator=generator,
        )
        coordinates = scene.coordinate_origin[:, None] + unit * scene.coordinate_scale[:, None]
        mask = torch.ones((batch_size, query_count), device=coordinates.device, dtype=torch.bool)
        packed = arm.model._packed_local_pairs(scene, coordinates, mask)
        from channelthermal.content_hypergraph import THERMAL_CONTENT_COORDINATE_FRAME

        from honf_forward_core.content_hypergraph import QueryBatch

        query = QueryBatch(
            coordinates=coordinates,
            query_mask=mask,
            query_ids=torch.arange(query_count, device=coordinates.device)[None].expand(batch_size, -1),
            local_batch_indices=packed[0], local_query_indices=packed[1],
            local_module_indices=packed[2], local_weights=packed[3],
            scene_key=scene.scene_key,
            coordinate_frame_key=THERMAL_CONTENT_COORDINATE_FRAME,
            receiver_key=("target_free_coordinate_density", query_count),
        )
        controls = receivers.heat
    else:
        # Native Wind coordinates and query features are repeated only to raise
        # Q beyond the stored 8192 stress panel; no target enters the core.
        from windfarm.content_hypergraph import WindContentReceivers

        coordinates = _repeat_query_axis(_repeat_first_axis(receivers.coordinates_D, batch_size), query_count)
        features = _repeat_query_axis(_repeat_first_axis(receivers.features, batch_size), query_count)
        flat_indices = _repeat_query_axis(_repeat_first_axis(receivers.flat_indices, batch_size), query_count)
        role_ids = _repeat_query_axis(_repeat_first_axis(receivers.role_ids, batch_size), query_count)
        base_input = scene_inputs[0]
        local_b, local_q, local_m = [], [], []
        for batch_index in range(batch_size):
            qi, mi = arm.provider.adapter.packed_local_pair_indices(base_input, coordinates[batch_index])
            local_b.append(np.full(qi.size, batch_index, dtype=np.int64))
            local_q.append(qi)
            local_m.append(mi)
        cat = lambda values: np.concatenate(values) if values else np.empty(0, dtype=np.int64)
        stress_receivers = WindContentReceivers(
            coordinates_D=coordinates, features=features, flat_indices=flat_indices, role_ids=role_ids,
            scene_keys=scene.scene_key,
            local_batch_indices=cat(local_b), local_query_indices=cat(local_q), local_module_indices=cat(local_m),
        )
        query = arm.provider.adapter.make_query_batch(scene, stress_receivers)
        controls = None
    organization = core.prepare(scene)
    output = core.decode(organization, query, controls=controls)
    backward_details = {}
    if backward:
        terms = [output.nonlinear_field.square().mean()]
        if output.thermal_temperature is not None:
            terms.append(output.thermal_temperature.square().mean())
        loss = sum(terms)
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("Large core backward stress produced a nonfinite objective.")
        loss.backward()
        gradients = [parameter.grad for parameter in core.parameters() if parameter.grad is not None]
        finite = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
        if not finite:
            raise FloatingPointError("Large core backward stress produced absent/nonfinite gradients.")
        backward_details = {
            "backward_completed": True,
            "finite_gradients": finite,
            "gradient_tensor_count": len(gradients),
            "loss_value": float(loss.detach().cpu()),
            "objective": "mean squared core outputs at native heat; no target or optimizer step",
        }
    work = dict(output.work_counts or {})
    source_factor_bytes = _tensor_bytes(organization.source_coefficients)
    output_bytes = _tree_tensor_bytes({
        "nonlinear_field": output.nonlinear_field,
        "thermal_temperature": output.thermal_temperature,
    })
    unique_per_scene = int(query.num_queries)
    if query.query_ids is not None and dataset == "wind":
        unique_rows = _unique_id_counts_by_row(flat_indices)
        unique_per_scene = unique_rows[0]
    else:
        unique_rows = [unique_per_scene] * batch_size
    return {
        "requested_q_per_scene": query_count,
        "batch_size": batch_size,
        "unique_receiver_ids_per_scene": unique_per_scene,
        "unique_neural_receiver_union_rows_per_scene": unique_rows,
        "executed_receiver_rows": int(query.batch_size * query.num_queries),
        "unique_neural_receiver_union_rows": sum(unique_rows),
        "padded_edge_rows": int(work.get("edge_rows", 0)),
        "active_edge_rows": int(work.get("edge_rows", 0)),
        "local_rows": int(work.get("local_rows", 0)),
        "direct_source_rows": int(work.get("direct_source_rows", 0)),
        "output_bytes": output_bytes,
        "source_factor_bytes": source_factor_bytes,
        "native_grid_accuracy_claim": False,
        "work_counts": work,
        **backward_details,
    }


def run_density_core_probe(arms: Mapping[str, Arm], high_batches: Mapping[str, Any], device: torch.device) -> list[dict[str, Any]]:
    records = []
    for architecture in architecture_pair(arms):
        arm = arms[architecture]
        batch = high_batches[architecture]
        # The target-free Q=32768 point is distinct from native Q8192 grid
        # saturation and runs without retaining a backward graph.
        try:
            result, memory = _timed_once(
                lambda arm=arm, batch=batch: _stress_core_probe_no_grad(arm, batch), device,
            )
        except BaseException as exc:
            if is_oom_error(exc):
                records.append(oom_receipt(architecture, 32768, exc))
                raise BenchmarkOOM(f"OOM at fixed B=8,Q=32768,arm={architecture}; no fallback was attempted.") from exc
            raise
        details = result
        records.append({
            "architecture": architecture, "dataset": arm.recipe["dataset"],
            "scope": "target_free_core_only_q32768_no_grad",
            "status": "completed", "optimizer_updates": 0,
            "timing_and_memory": memory, **details,
        })
    return records


def _stress_core_probe_no_grad(arm: Arm, native_batch: Any) -> Mapping[str, Any]:
    with torch.no_grad():
        return _stress_core_probe(arm, native_batch, query_count=32768)


def _provider_capacity(provider: Any, dataset: str) -> int:
    return int(provider.effective_batch_size)


def _provider_microbatch_set(provider: Any, dataset: str, value: int) -> None:
    if dataset == "thermal":
        provider.microbatch_size = int(value)
    else:
        provider.max_microbatch_cases = int(value)


def run_training_throughput(
    arms: Mapping[str, Arm], *, packings: Sequence[int], epochs: int,
    update_budget: int, output: Path, progress: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    if not packings:
        return [], 0, {"status": "not_requested"}
    dataset = next(iter(arms.values())).recipe["dataset"]
    effective = _provider_capacity(arms[architecture_pair(arms)[0]].provider, dataset)
    train_count = len(arms[architecture_pair(arms)[0]].provider.epoch_cases(1, int(arms[architecture_pair(arms)[0]].recipe["seed"])))
    planned = expected_optimizer_updates(train_count, effective, epochs, len(packings))
    if planned > update_budget or planned > MAX_DISPOSABLE_UPDATES:
        raise ValueError(
            f"Full-epoch packing plan requires {planned} disposable updates, above authorized budget "
            f"{min(update_budget, MAX_DISPOSABLE_UPDATES)}; no partial or reduced-batch run was started."
        )
    packings = validate_microbatch_packings(packings, effective)
    progress["planned_optimizer_updates"] = planned
    (output / "progress.json").write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
    records = []
    actual_updates = 0
    for architecture in architecture_pair(arms):
        arm = arms[architecture]
        if train_count != len(arm.provider.epoch_cases(1, int(arm.recipe["seed"]))):
            raise ValueError("Matched arms expose different full TRAIN case counts.")
        for packing in packings:
            arm.model.load_state_dict(arm.initial_model_state, strict=True)
            arm.model.train(True)
            _provider_microbatch_set(arm.provider, dataset, packing)
            config = replace(
                engine_for(arm.recipe, "cpu").config,
                microbatch_cases=int(packing), effective_cases=effective,
            )
            engine = TrainingEngine(config, device=next(arm.model.parameters()).device)
            optimizer, group_specs = engine._make_optimizer(
                arm.model, arm.provider, architecture, 1, None,
            )
            cache_before = provider_cache_stats(arm.provider)
            epoch_records = []
            for epoch in range(1, epochs + 1):
                arm.model.train(True)
                engine._set_schedules(optimizer, group_specs, epoch)
                ordered_cases, dataset_id = engine.epoch_case_order(arm.provider, epoch)
                _device_sync(engine.device)
                epoch_start = time.perf_counter()
                epoch_memory_baseline = _memory_baseline(engine.device)
                if engine.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(engine.device)
                step_records = []
                for update_index, case_slice in enumerate(engine.epoch_update_slices(len(ordered_cases))):
                    update_cases = ordered_cases[case_slice]
                    microbatches = engine.prepare_update_microbatches(
                        arm.provider, arm=architecture, phase="joint", epoch=epoch,
                        update_index=update_index, update_cases=update_cases,
                        sampling_dataset_id=dataset_id,
                    )
                    query_stream_hash = packing_independent_query_sha256(microbatches, dataset)
                    try:
                        losses, work, denominators, sampling_hash = engine._run_update(
                            arm.model, arm.provider, optimizer, microbatches, phase="joint",
                            arm=architecture, epoch=epoch, update_index=update_index,
                        )
                    except BaseException as exc:
                        if is_oom_error(exc):
                            raise BenchmarkOOM(
                                f"OOM during full effective-batch update: arm={architecture}, "
                                f"microbatch={packing}, effective={effective}, epoch={epoch}, update={update_index}."
                            ) from exc
                        raise
                    actual_updates += 1
                    step_records.append({
                        "epoch": epoch, "update_index": update_index,
                        "case_count": len(update_cases), "microbatch_count": len(microbatches),
                        "losses": losses, "work_counts": work, "loss_denominators": denominators,
                        "query_sampling_sha256": sampling_hash,
                        "packing_independent_query_sha256": query_stream_hash,
                        "thermal_response_family_ids": [
                            batch.auxiliary.get("response_family_id")
                            for batch in microbatches
                            if isinstance(batch.auxiliary, Mapping) and batch.auxiliary.get("response_family_id")
                        ],
                    })
                    progress.update({
                        "executed_optimizer_updates": actual_updates,
                        "last_architecture": architecture,
                        "last_microbatch_cases": packing,
                        "last_epoch": epoch,
                        "last_update_index": update_index,
                    })
                    (output / "progress.json").write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
                _device_sync(engine.device)
                elapsed = time.perf_counter() - epoch_start
                epoch_records.append({
                    "epoch": epoch, "complete_epoch": True,
                    "elapsed_seconds": elapsed, "optimizer_updates": len(step_records),
                    "memory": {
                        **epoch_memory_baseline,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(engine.device))
                            if engine.device.type == "cuda" else None,
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(engine.device))
                            if engine.device.type == "cuda" else None,
                    },
                    "case_visits": sum(item["case_count"] for item in step_records),
                    "steps": step_records,
                })
            records.append({
                "architecture": architecture, "dataset": dataset, "microbatch_cases": packing,
                "effective_cases": effective, "queries_per_case": int(arm.recipe["primary_queries"]),
                "complete_epochs": epochs, "optimizer_updates": sum(item["optimizer_updates"] for item in epoch_records),
                "initialization": arm.source, "source_checkpoint_sha256": arm.checkpoint_sha256,
                "weights_reset_to_source_each_packing": True,
                "initial_state_sha256": _state_sha256(arm.initial_model_state),
                "optimizer_state": "fresh_disposable_adamw",
                "optimizer_state_bytes": optimizer_state_bytes(optimizer),
                "parameter_inventory": parameter_inventory(arm.model),
                "native_catalogue_cache_before": cache_before,
                "native_catalogue_cache_after": provider_cache_stats(arm.provider),
                "epochs": epoch_records,
            })
            del optimizer
            arm.model.load_state_dict(arm.initial_model_state, strict=True)
            arm.model.eval()
    if actual_updates != planned:
        raise RuntimeError(f"Executed {actual_updates} updates; planned full-epoch count was {planned}.")
    if dataset == "thermal":
        missing = [record for record in records if sum(
            len(step["thermal_response_family_ids"])
            for epoch_record in record["epochs"] for step in epoch_record["steps"]
        ) != epochs]
        if missing:
            raise RuntimeError("Thermal throughput epochs did not execute one scheduled response addendum each.")
    initial_state_checks = {}
    stream_checks = {}
    loss_checks = {}
    for architecture in architecture_pair(arms):
        arm_records = [record for record in records if record["architecture"] == architecture]
        initial_hashes = {record["initial_state_sha256"] for record in arm_records}
        initial_state_checks[architecture] = {
            "all_packings_match": len(initial_hashes) == 1,
            "state_sha256": next(iter(initial_hashes)) if initial_hashes else None,
        }
        if len(initial_hashes) != 1:
            raise RuntimeError(f"{architecture} did not reset to one identical initial state per packing.")
        by_step: dict[tuple[int, int], set[str]] = {}
        for record in arm_records:
            for epoch_record in record["epochs"]:
                for step in epoch_record["steps"]:
                    key = (int(step["epoch"]), int(step["update_index"]))
                    by_step.setdefault(key, set()).add(step["packing_independent_query_sha256"])
        stream_checks[architecture] = {
            "compared_update_count": len(by_step),
            "all_packings_match": all(len(values) == 1 for values in by_step.values()),
        }
    first_packing = min(packings)
    for architecture in architecture_pair(arms):
        reference = next(
            record for record in records
            if record["architecture"] == architecture and record["microbatch_cases"] == first_packing
        )
        reference_losses = reference["epochs"][0]["steps"][0]["losses"]
        for record in records:
            if record["architecture"] != architecture or record["microbatch_cases"] == first_packing:
                continue
            losses = record["epochs"][0]["steps"][0]["losses"]
            comparisons = {}
            for name in sorted(reference_losses.keys() & losses.keys()):
                expected, observed = float(reference_losses[name]), float(losses[name])
                absolute = abs(observed - expected)
                relative = absolute / max(abs(expected), 1.0e-12)
                comparisons[name] = {
                    "reference": expected, "observed": observed,
                    "absolute_difference": absolute, "relative_difference": relative,
                    "within_tolerance": bool(np.isclose(observed, expected, rtol=1.0e-4, atol=1.0e-6)),
                }
            loss_checks[f"{architecture}:packing_{record['microbatch_cases']}"] = {
                "reference_packing": first_packing,
                "first_update_loss_terms": comparisons,
                "all_within_tolerance": bool(comparisons) and all(
                    item["within_tolerance"] for item in comparisons.values()
                ),
            }
    checks = {
        "status": "completed",
        "initial_state_by_arm": initial_state_checks,
        "query_stream_by_arm_across_packings": stream_checks,
        "first_update_losses_against_smallest_packing": loss_checks,
        "query_streams_match_across_packings": all(item["all_packings_match"] for item in stream_checks.values()),
        "first_update_losses_match_within_tolerance": all(
            item["all_within_tolerance"] for item in loss_checks.values()
        ),
    }
    return records, actual_updates, checks


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(child) for child in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("thermal", "wind"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe", type=Path, help="Fresh model recipe; never a science weight source.")
    source.add_argument("--checkpoint-ad", type=Path)
    parser.add_argument("--checkpoint-ah6", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-density", action="store_true")
    parser.add_argument("--extended-density", action="store_true",
                        help="Optional target-free Q32768 stress after the three fixed-B8 native points.")
    parser.add_argument("--thermal-reader128-revision", action="store_true",
                        help="Explicit retained Thermal A-D64/A-H6-128 revision cost; not a matched decoder ablation.")
    parser.add_argument("--typed-centroid-revision", action="store_true",
                        help="Explicit retained Thermal or Wind A2-D/default versus A2-H6 typed-centroid R1 pair.")
    parser.add_argument("--training-packings", default="",
                        help="Optional comma-separated microbatch sizes; effective batch and Q stay fixed.")
    parser.add_argument("--training-epochs", type=int, default=5)
    parser.add_argument("--optimizer-update-budget", type=int, default=0,
                        help="Explicit disposable-step allowance; hard-capped at 210 for this tool.")
    args = parser.parse_args(argv)
    if args.checkpoint_ad is not None and args.checkpoint_ah6 is None:
        parser.error("Checkpoint mode requires --checkpoint-ad and --checkpoint-ah6.")
    if args.recipe is not None and args.checkpoint_ah6 is not None:
        parser.error("Use either --recipe or both checkpoint paths.")
    if args.optimizer_update_budget < 0 or args.optimizer_update_budget > MAX_DISPOSABLE_UPDATES:
        parser.error(f"--optimizer-update-budget must be in 0..{MAX_DISPOSABLE_UPDATES}.")
    try:
        packings = () if not args.training_packings.strip() else parse_int_list(
            args.training_packings, name="training-packings",
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if packings and (args.training_epochs < 1 or args.optimizer_update_budget < 1):
        parser.error("Training packings require positive --training-epochs and explicit update budget.")
    if packings and len(packings) > 3:
        parser.error("At most three microbatch packings may be tested in one invocation.")
    if args.thermal_reader128_revision and args.typed_centroid_revision:
        parser.error("Thermal reader128 and typed-centroid R1 revisions are mutually exclusive.")
    if args.thermal_reader128_revision and (
        args.dataset != "thermal" or args.recipe is not None or packings
        or args.checkpoint_ad is None or args.checkpoint_ah6 is None
    ):
        parser.error("Reader128 revision mode requires Thermal retained checkpoints and zero training packings.")
    if args.typed_centroid_revision and (
        args.dataset not in ("thermal", "wind") or args.recipe is not None or packings
        or args.checkpoint_ad is None or args.checkpoint_ah6 is None
    ):
        parser.error(
            "Typed-centroid R1 comparison requires Thermal or Wind retained checkpoints and zero training packings."
        )
    if not packings and args.optimizer_update_budget != 0:
        parser.error("An optimizer update budget is meaningful only with --training-packings.")
    if args.output.exists():
        raise FileExistsError("Benchmark output is immutable; select a fresh directory.")
    device = torch.device(args.device)
    device_binding = validate_device_binding(device)
    args.output.mkdir(parents=True)
    progress = {
        "schema": SCHEMA, "status": "running", "dataset": args.dataset,
        "planned_optimizer_updates": 0, "executed_optimizer_updates": 0,
        "inference_and_backward_only_updates": 0,
    }
    (args.output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    initial_checkpoint_hashes = {}
    try:
        provider_started = time.perf_counter()
        arms, provider_identities, initialization = load_arms(
            dataset=args.dataset, device=args.device, recipe_path=args.recipe,
            checkpoint_ad=args.checkpoint_ad, checkpoint_ah6=args.checkpoint_ah6,
            thermal_reader128_revision=args.thermal_reader128_revision,
            typed_centroid_revision=args.typed_centroid_revision,
        )
        _device_sync(device)
        provider_setup_seconds = time.perf_counter() - provider_started
        provider_phases = {"pair_model_and_provider_setup_wall_seconds": provider_setup_seconds,
                           "scope": "both models/providers; native reference fixture loading and targets are separate from inference",
                           "cache_after_setup": provider_cache_stats(next(iter(arms.values())).provider),
                           "fixture_preparation": {},
                           "filesystem_page_cache_state": "uncontrolled; no cold-disk claim"}
        initial_checkpoint_hashes = {
            architecture: arm.checkpoint_sha256 for architecture, arm in arms.items()
            if arm.checkpoint is not None
        }
        resource_baseline = check_memory_headroom(device)
        fixture_selection, fixture_batches, high_batches = {}, {}, {}
        if not args.skip_native or not args.skip_density:
            fixture_selection = choose_low_high_train_keys(arms[architecture_pair(arms)[0]].provider, args.dataset)
            # Both providers bind the same TRAIN identities and transforms.
            # Read each native fixture once and share target-free inputs.
            shared_fixture_batches = {}
            for name, selection in fixture_selection.items():
                fixture_started = time.perf_counter()
                shared_fixture_batches[name] = native_train_batch(arms[architecture_pair(arms)[0]].provider, selection["case_key"])
                _device_sync(device)
                provider_phases["fixture_preparation"][name] = {
                    "wall_seconds": time.perf_counter() - fixture_started,
                    "case_key": selection["case_key"],
                    "cache_after": provider_cache_stats(next(iter(arms.values())).provider)}
            fixture_batches = {architecture: shared_fixture_batches for architecture in architecture_pair(arms)}
            high_batches = {
                architecture: fixture_batches[architecture]["high_m"] for architecture in architecture_pair(arms)
            }
        native_records = [] if args.skip_native else run_native_matrix(arms, fixture_batches, device)
        (args.output / "native_cost_matrix.json").write_text(
            json.dumps(json_safe(native_records), indent=2, allow_nan=False) + "\n")
        density_records = []
        if not args.skip_density:
            density_records.extend(run_density_backward(arms, high_batches, device))
            (args.output / "density_backward.json").write_text(
                json.dumps(json_safe(density_records), indent=2, allow_nan=False) + "\n")
            if args.extended_density:
                density_records.extend(run_density_core_probe(arms, high_batches, device))
        training_records, executed_updates, packing_checks = run_training_throughput(
            arms, packings=packings, epochs=args.training_epochs,
            update_budget=args.optimizer_update_budget, output=args.output, progress=progress,
        )
        final_hashes = {
            architecture: sha256_file(arm.checkpoint)
            for architecture, arm in arms.items() if arm.checkpoint is not None
        }
        if final_hashes != initial_checkpoint_hashes:
            raise RuntimeError("A retained input checkpoint changed during benchmarking.")
        inventory = {architecture: parameter_inventory(arm.model) for architecture, arm in arms.items()}
        result = {
            "schema": SCHEMA,
            "status": "completed",
            "dataset": args.dataset,
            "device": args.device,
            "cuda_visible_device_uuid": device_binding,
            "weight_source": "retained_checkpoint_pair" if args.recipe is None else "fresh_recipe_pair",
            "comparison": (
                "Thermal reader128 capacity revision" if args.thermal_reader128_revision
                else "typed-centroid R1 model revision" if args.typed_centroid_revision
                else "matched decoder pair"
            ),
            "typed_centroid_revision": args.typed_centroid_revision,
            "matched_reader_capacity": not args.thermal_reader128_revision,
            "recipe": arms[architecture_pair(arms)[0]].recipe,
            "checkpoint_sha256": initial_checkpoint_hashes,
            "provider_identity_sha256": {
                architecture: canonical_sha256(identity)
                for architecture, identity in provider_identities.items()
            },
            "initialization": initialization,
            "fixture_selection": fixture_selection,
            "parameter_inventory": inventory,
            "device_memory_baseline": resource_baseline,
            "provider_phases": provider_phases,
            "complete_scope": "host prescribed scene/query inputs through native physical outputs returned to host; includes H2D and D2H; target/reference fixture preparation separately timed",
            "native_cost_matrix": native_records,
            "density_scaling": density_records,
            "training_throughput": training_records,
            "training_packing_checks": packing_checks,
            "optimizer_updates": executed_updates,
            "inference_and_backward_only_updates": 0,
            "new_solver_attempts": 0,
            "WindTEST": "locked",
            "checkpoint_unchanged": True,
        }
        (args.output / "result.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
        progress.update({"status": "completed", "executed_optimizer_updates": executed_updates})
        (args.output / "progress.json").write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
        print(json.dumps(json_safe({key: value for key, value in result.items()
                                    if key not in {"native_cost_matrix", "density_scaling", "training_throughput"}}),
                         sort_keys=True, allow_nan=False), flush=True)
        return 0
    except BenchmarkOOM as exc:
        progress.update({"status": "failed_oom", "failure": str(exc)})
        (args.output / "progress.json").write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
        failure = {
            "schema": SCHEMA, "status": "failed_oom", "dataset": args.dataset,
            "device": args.device, "optimizer_updates": int(progress["executed_optimizer_updates"]),
            "failure": str(exc), "checkpoint_sha256_at_start": initial_checkpoint_hashes,
            "oom_policy": "fixed B/Q retained; no smaller-batch pass claim",
        }
        (args.output / "result.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        progress.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        (args.output / "progress.json").write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
        failure = {
            "schema": SCHEMA, "status": "failed", "dataset": args.dataset,
            "device": args.device, "optimizer_updates": int(progress["executed_optimizer_updates"]),
            "failure": f"{type(exc).__name__}: {exc}",
            "checkpoint_sha256_at_start": initial_checkpoint_hashes,
        }
        (args.output / "result.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
