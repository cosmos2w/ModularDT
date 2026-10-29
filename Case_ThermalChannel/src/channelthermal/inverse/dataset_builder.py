"""Case-major inverse-dataset assembly around one frozen HONF call per case.

Each source physical design ``D`` and context ``c`` is verified once. The
resulting compact plan is target ``G``; request variants ``R`` are augmented
around exact outputs without another HONF call. ``G_hat`` is the same realized
target at dataset-build time and is reserved as that name during generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from honf_inverse_core.contracts import CompactPlan, FunctionalValue, NamedContext, PhysicalDesign
from honf_inverse_core.normalization import ScalarStats, VectorStats, fit_scalar, fit_vector

from .augmentation import RequestRecipe, generate_request_recipe, materialize_request, realized_values_for_request
from .compact_plan import COMPACT_PLAN_FEATURE_NAMES
from .dataset_io import (
    INVERSE_DATASET_SCHEMA_NAME,
    INVERSE_DATASET_SCHEMA_VERSION,
    write_inverse_hdf5_atomic,
)
from .geometry import (
    canonicalize_design,
    evaluate_geometry,
    geometry_constraint_tensor,
    normalize_geometry_constraints,
)
from .request import make_request_codec
from .vocabulary import NONREGIONAL_REQUEST_TYPES, REGIONAL_REQUEST_TYPES, REQUEST_TYPES, REQUEST_TYPE_TO_ID


RegionalEvaluator = Callable[[str, tuple[float, float, float, float]], FunctionalValue]


@dataclass(frozen=True)
class CaseBuildRecord:
    """All one-call information needed to materialize one case-major row."""

    case_id: str
    source_split: str
    source_index: int
    design: PhysicalDesign
    context: NamedContext
    compact_plan: CompactPlan
    full_plan: Mapping[str, Any]
    nonregional_functionals: Mapping[str, FunctionalValue]
    regional_evaluator: RegionalEvaluator
    inverse_split: str | None = None
    converged: bool = True
    metadata: Mapping[str, Any] | None = None


def case_id_sha256(case_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(set(map(str, case_ids)))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_inverse_splits(
    case_ids: Sequence[str],
    source_splits: Sequence[str],
    *,
    split_seed: int = 20260813,
    validation_fraction: float = 0.10,
) -> tuple[str, ...]:
    """Preserve source test/validation and derive validation from source train."""

    if len(case_ids) != len(source_splits) or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("Split inputs differ in length or validation_fraction is invalid.")
    result = ["" for _ in case_ids]
    train_indices: list[int] = []
    for index, source in enumerate(source_splits):
        name = str(source).lower()
        if name == "test":
            result[index] = "test"
        elif name in {"validation", "val"}:
            result[index] = "validation"
        elif name == "train":
            train_indices.append(index)
        else:
            raise ValueError(f"Unsupported source split {source!r} for case {case_ids[index]!r}.")
    ordered = sorted(
        train_indices,
        key=lambda index: hashlib.sha256(f"{int(split_seed)}\0{case_ids[index]}".encode()).hexdigest(),
    )
    validation_count = int(round(len(ordered) * float(validation_fraction)))
    if len(ordered) >= 2 and validation_fraction > 0.0:
        validation_count = min(max(validation_count, 1), len(ordered) - 1)
    validation_indices = set(ordered[:validation_count])
    for index in train_indices:
        result[index] = "validation" if index in validation_indices else "train"
    return tuple(result)


def _stack_full_plans(records: Sequence[CaseBuildRecord]) -> dict[str, np.ndarray]:
    keys = (
        "A_mh", "A_eh", "hyper_source_coords", "hyper_region_coords",
        "hyper_module_mass", "hyper_env_mass", "hyper_strength",
        "active_hyperedge_mask", "env_coords", "edge_permutation",
    )
    result: dict[str, np.ndarray] = {}
    for key in keys:
        if not all(key in record.full_plan for record in records):
            continue
        arrays = [np.asarray(record.full_plan[key]) for record in records]
        if len({array.shape for array in arrays}) == 1:
            result[key] = np.stack(arrays, axis=0)
    return result


def _fit_functional_stats(
    records: Sequence[CaseBuildRecord],
    recipes: Sequence[Sequence[RequestRecipe]],
    inverse_splits: Sequence[str],
) -> dict[str, ScalarStats]:
    values: dict[str, list[float]] = {name: [] for name in REQUEST_TYPES}
    pooled_regional: list[float] = []
    train_environment: list[float] = []
    for case_index, (record, split) in enumerate(zip(records, inverse_splits)):
        if split != "train":
            continue
        for name in NONREGIONAL_REQUEST_TYPES:
            values[name].append(float(record.nonregional_functionals[name].value))
        train_environment.append(float(record.nonregional_functionals["environment_temperature_max"].value))
        for recipe in recipes[case_index]:
            for token in recipe.tokens:
                if token.request_type in REGIONAL_REQUEST_TYPES:
                    values[token.request_type].append(float(token.realized_value))
                    pooled_regional.append(float(token.realized_value))
    result: dict[str, ScalarStats] = {}
    for name in REQUEST_TYPES:
        selected = values[name]
        if not selected and name in REGIONAL_REQUEST_TYPES:
            selected = pooled_regional or train_environment
        if not selected:
            raise ValueError(f"Cannot fit train-only functional normalization for {name!r}.")
        result[name] = fit_scalar(selected)
    return result


def _normalization_payload(
    records: Sequence[CaseBuildRecord],
    inverse_splits: Sequence[str],
    functional: Mapping[str, ScalarStats],
) -> tuple[VectorStats, ScalarStats, ScalarStats, dict[str, Any]]:
    train = [index for index, split in enumerate(inverse_splits) if split == "train"]
    if not train:
        raise ValueError("Inverse dataset requires at least one training case.")
    context_stats = fit_vector(
        np.stack([records[index].context.vector for index in train]),
        records[train[0]].context.feature_names,
    )
    active_heat = np.concatenate(
        [
            records[index].design.heat_powers[records[index].design.module_present > 0.5]
            for index in train
        ]
    )
    heat_stats = fit_scalar(active_heat)
    total_heat_stats = fit_scalar(
        [
            float(np.sum(records[index].design.heat_powers * records[index].design.module_present))
            for index in train
        ]
    )
    payload = {
        "functional_mean": np.asarray([functional[name].mean for name in REQUEST_TYPES], dtype=np.float32),
        "functional_std": np.asarray([functional[name].std for name in REQUEST_TYPES], dtype=np.float32),
        "functional_count": np.asarray([functional[name].count for name in REQUEST_TYPES], dtype=np.int64),
        "functional_names": np.asarray(REQUEST_TYPES, dtype=object),
        "context_mean": context_stats.mean,
        "context_std": context_stats.std,
        "context_feature_names": np.asarray(context_stats.feature_names, dtype=object),
        "active_heat_power_mean": np.asarray(heat_stats.mean, dtype=np.float32),
        "active_heat_power_std": np.asarray(heat_stats.std, dtype=np.float32),
        "total_heat_mean": np.asarray(total_heat_stats.mean, dtype=np.float32),
        "total_heat_std": np.asarray(total_heat_stats.std, dtype=np.float32),
    }
    return context_stats, heat_stats, total_heat_stats, payload


def build_inverse_dataset_from_records(
    records: Sequence[CaseBuildRecord],
    output_path: str | Path,
    *,
    variants_per_case: int = 16,
    seed: int = 20260813,
    split_seed: int = 20260813,
    validation_fraction: float = 0.10,
    save_full_plan: bool = True,
    provenance: Mapping[str, Any] | None = None,
    partial_debug: bool = False,
) -> Path:
    """Materialize records into the strict schema-v1 case-major HDF5 artifact."""

    records = tuple(records)
    if not records or variants_per_case <= 0:
        raise ValueError("Dataset assembly requires records and positive variants_per_case.")
    if len({record.case_id for record in records}) != len(records):
        raise ValueError("CaseBuildRecord case IDs must be unique.")
    n = len(records)
    max_modules = records[0].design.max_modules
    num_edges = records[0].compact_plan.num_edges
    if any(record.design.max_modules != max_modules or record.compact_plan.num_edges != num_edges for record in records):
        raise ValueError("Every record must share fixed M and K.")
    source_splits = tuple(record.source_split for record in records)
    overrides = [record.inverse_split for record in records]
    if any(value is not None for value in overrides):
        if not all(value in {"train", "validation", "test"} for value in overrides):
            raise ValueError("inverse_split overrides must be complete and use train/validation/test.")
        inverse_splits = tuple(str(value) for value in overrides)
    else:
        inverse_splits = assign_inverse_splits(
            [record.case_id for record in records], source_splits,
            split_seed=split_seed, validation_fraction=validation_fraction,
        )
    recipes: list[list[RequestRecipe]] = []
    for record in records:
        per_case = [
            generate_request_recipe(
                global_seed=seed,
                case_id=record.case_id,
                variant_index=variant,
                nonregional_values=record.nonregional_functionals,
                regional_evaluator=record.regional_evaluator,
                design=record.design,
                context=record.context,
            )
            for variant in range(variants_per_case)
        ]
        recipes.append(per_case)
    functional_stats = _fit_functional_stats(records, recipes, inverse_splits)
    context_stats, heat_stats, total_heat_stats, normalization = _normalization_payload(
        records, inverse_splits, functional_stats
    )
    codec = make_request_codec(functional_stats)

    source_centers = np.stack([record.design.module_centers for record in records]).astype(np.float32)
    source_present = np.stack([record.design.module_present for record in records]).astype(np.uint8)
    source_heat = np.stack([record.design.heat_powers for record in records]).astype(np.float32)
    canonical = [canonicalize_design(record.design, record.context) for record in records]
    model_centers = np.zeros_like(source_centers)
    model_present = np.stack([view.design.module_present for view in canonical]).astype(np.uint8)
    model_heat = np.zeros_like(source_heat)
    for index, view in enumerate(canonical):
        values = records[index].context.as_mapping()
        model_centers[index] = view.design.module_centers / np.asarray(
            [values["domain_length_x"], values["domain_length_y"]], dtype=np.float32
        )
        active = view.design.module_present > 0.5
        model_heat[index, active] = heat_stats.normalize(view.design.heat_powers[active]).astype(np.float32)

    request_tensors: dict[str, list[list[np.ndarray]]] = {}
    request_json = np.empty((n, variants_per_case), dtype=object)
    realized_raw = np.zeros((n, variants_per_case, 4), dtype=np.float32)
    realized_normalized = np.zeros_like(realized_raw)
    geometry_raw = np.zeros((n, variants_per_case, 8), dtype=np.float32)
    geometry_normalized = np.zeros_like(geometry_raw)
    geometry_mask = np.zeros((n, variants_per_case, 8), dtype=np.uint8)
    geometry_margin = np.zeros((n, variants_per_case, 8), dtype=np.float32)
    geometry_valid = np.zeros((n, variants_per_case), dtype=np.uint8)
    geometry_actual = np.zeros((n, 6), dtype=np.float32)
    pair_defined = np.zeros((n,), dtype=np.uint8)
    for case_index, record in enumerate(records):
        for variant_index, recipe in enumerate(recipes[case_index]):
            request = materialize_request(recipe, functional_stats)
            tensor = codec.tensorize(request)
            for name, value in tensor.as_dict().items():
                request_tensors.setdefault(name, [[] for _ in range(n)])[case_index].append(value)
            request_json[case_index, variant_index] = json.dumps(request.to_dict(), sort_keys=True)
            realized_raw[case_index, variant_index] = realized_values_for_request(recipe)
            for slot, token in enumerate(recipe.tokens):
                realized_normalized[case_index, variant_index, slot] = functional_stats[
                    token.request_type
                ].normalize(token.realized_value)
            raw, mask = geometry_constraint_tensor(recipe.geometry_constraints)
            normalized, _ = normalize_geometry_constraints(
                recipe.geometry_constraints,
                record.context,
                max_modules=max_modules,
                total_heat_stats=total_heat_stats,
            )
            evaluation = evaluate_geometry(record.design, record.context, recipe.geometry_constraints)
            geometry_raw[case_index, variant_index] = raw
            geometry_normalized[case_index, variant_index] = normalized
            geometry_mask[case_index, variant_index] = mask
            geometry_margin[case_index, variant_index] = evaluation.margin_raw
            geometry_valid[case_index, variant_index] = evaluation.valid
            geometry_actual[case_index] = evaluation.actual_raw
            pair_defined[case_index] = evaluation.pair_distance_defined
    stacked_request = {
        name: np.stack([np.stack(per_case, axis=0) for per_case in case_values], axis=0)
        for name, case_values in request_tensors.items()
    }
    stacked_request.update(
        realized_value_raw=realized_raw,
        realized_value_normalized=realized_normalized,
        json=request_json,
    )
    split_hashes = {
        name: case_id_sha256(
            [record.case_id for record, split in zip(records, inverse_splits) if split == name]
        )
        for name in ("train", "validation", "test")
    }
    arrays: dict[str, Any] = {
        "case_id": np.asarray([record.case_id for record in records], dtype=object),
        "source_split": np.asarray(source_splits, dtype=object),
        "inverse_split": np.asarray(inverse_splits, dtype=object),
        "source_index": np.asarray([record.source_index for record in records], dtype=np.int64),
        "variant_id": np.arange(variants_per_case, dtype=np.int16),
        "variant_seed": np.asarray([[recipe.variant_seed for recipe in values] for values in recipes], dtype=np.uint64),
        "design": {
            "source": {
                "module_centers": source_centers,
                "module_present": source_present,
                "heat_powers": source_heat,
            },
            "model": {
                "normalized_module_centers": model_centers,
                "module_present": model_present,
                "normalized_heat_powers": model_heat,
                "canonical_to_source": np.stack([view.canonical_to_source for view in canonical]).astype(np.int16),
                "source_to_canonical": np.stack([view.source_to_canonical for view in canonical]).astype(np.int16),
            },
            "module_count": source_present.sum(axis=1).astype(np.int16),
        },
        "context": {
            "vector": np.stack([record.context.vector for record in records]).astype(np.float32),
            "normalized_vector": context_stats.normalize(
                np.stack([record.context.vector for record in records])
            ),
            "feature_names": np.asarray(records[0].context.feature_names, dtype=object),
        },
        "plan": {
            "compact_raw": np.stack([record.compact_plan.raw for record in records]).astype(np.float32),
            "compact_normalized": np.stack([record.compact_plan.normalized for record in records]).astype(np.float32),
            "feature_names": np.asarray(COMPACT_PLAN_FEATURE_NAMES, dtype=object),
            "validation_flags": np.ones((n,), dtype=np.uint8),
            **({"full": _stack_full_plans(records)} if save_full_plan else {}),
        },
        "functionals": {
            "global_raw": np.asarray(
                [[record.nonregional_functionals[name].value for name in NONREGIONAL_REQUEST_TYPES] for record in records],
                dtype=np.float32,
            ),
            "global_type_ids": np.asarray([REQUEST_TYPE_TO_ID[name] for name in NONREGIONAL_REQUEST_TYPES], dtype=np.int8),
            "global_valid": np.ones((n, len(NONREGIONAL_REQUEST_TYPES)), dtype=np.uint8),
        },
        "requests": stacked_request,
        "geometry": {
            "constraint_raw": geometry_raw,
            "constraint_normalized": geometry_normalized,
            "constraint_mask": geometry_mask,
            "actual_raw": geometry_actual,
            "margin_raw": geometry_margin,
            "valid": geometry_valid,
            "pair_distance_defined": pair_defined,
        },
        "metadata": {
            "converged": np.asarray([record.converged for record in records], dtype=np.uint8),
            "case_metadata_json": np.asarray(
                [json.dumps(dict(record.metadata or {}), sort_keys=True) for record in records], dtype=object
            ),
        },
        "normalization": normalization,
        "provenance": {
            "json": np.asarray(json.dumps(dict(provenance or {}), sort_keys=True), dtype=object),
        },
        "splits": {
            **{
                f"{name}_case_indices": np.asarray(
                    [index for index, split in enumerate(inverse_splits) if split == name], dtype=np.int64
                )
                for name in ("train", "validation", "test")
            },
            **{f"{name}_case_id_sha256": np.asarray(value, dtype=object) for name, value in split_hashes.items()},
        },
    }
    attributes = {
        "schema_name": INVERSE_DATASET_SCHEMA_NAME,
        "schema_version": INVERSE_DATASET_SCHEMA_VERSION,
        "num_cases": n,
        "variants_per_case": int(variants_per_case),
        "request_schema_version": 1,
        "compact_plan_schema_version": 1,
        "canonical_full_plan_schema_version": 2,
        "num_edges": num_edges,
        "max_modules": max_modules,
        "seed": int(seed),
        "split_seed": int(split_seed),
        "validation_fraction": float(validation_fraction),
        "partial_debug": bool(partial_debug),
        "split_hashes": split_hashes,
        "save_full_plan": bool(save_full_plan),
    }
    return write_inverse_hdf5_atomic(output_path, arrays=arrays, attributes=attributes)


__all__ = [
    "CaseBuildRecord",
    "assign_inverse_splits",
    "build_inverse_dataset_from_records",
    "case_id_sha256",
]
