#!/usr/bin/env python3
"""Evaluate HONF organizer/routing structure without changing inference.

The evaluator is deliberately checkpoint-owned: it restores the model, data
normalization, channel order, and saved schedule progress from each checkpoint.
It reports both candidate and effective/selected topology and can expose the
ChannelThermal base, provisional, and final organizer passes.  All pruning and
training behavior remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-honf-topology-diagnostics")

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer  # noqa: E402
from channelthermal.evaluation_tools.organizer_visualization import (  # noqa: E402
    render_channelthermal_organization_overview,
    render_channelthermal_organization_summary_matrices,
)
from channelthermal.evaluation_tools.plots import module_and_fluid_masks, module_radius_from_sample  # noqa: E402
from channelthermal.evaluation_tools.routing_visualization import save_routing_diagnostics  # noqa: E402
from channelthermal.evaluation_tools.topology_signature_visualization import (  # noqa: E402
    render_topology_signature_diagnostics,
)
from channelthermal.workflows.evaluate_forward import (  # noqa: E402
    extract_organization_arrays,
    load_model,
    make_batch,
)
from honf_forward_core.evaluation import extract_topology_signature  # noqa: E402


EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--include-case-id",
        action="append",
        default=[],
        help="Append named cases to a deterministic --max-cases subset.",
    )
    parser.add_argument("--render-case-id", action="append", default=[])
    parser.add_argument("--organizer-passes", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "diagnostics" / "topology_quality",
    )
    return parser.parse_args()


def checkpoint_specs(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        label, value = item.split("=", 1)
        result[label.strip()] = Path(value).expanduser().resolve()
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_manifest_provenance(checkpoint_path: Path) -> dict[str, Any]:
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        return {"source_sha": None, "config_sha256": None}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"source_sha": None, "config_sha256": None}
    return {
        "source_sha": manifest.get("source_state", {}).get("commit"),
        "config_sha256": manifest.get("config_sha256"),
    }


def numpy_aux(aux: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in aux.items():
        if torch.is_tensor(value):
            detached = value.detach().cpu()
            result[key] = detached.numpy()[0] if detached.ndim > 0 else float(detached)
        else:
            result[key] = value
    return result


def normalized_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.zeros_like(arr, dtype=np.float64)
    denom = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, denom, out=np.zeros_like(arr), where=denom > EPS)


def mean_off_diagonal_cosine(columns_or_rows: np.ndarray, *, columns: bool) -> float:
    arr = np.asarray(columns_or_rows, dtype=np.float64)
    if columns:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    unit = np.divide(arr, norm, out=np.zeros_like(arr), where=norm > EPS)
    cosine = unit @ unit.T
    indices = np.triu_indices(arr.shape[0], k=1)
    return float(np.mean(cosine[indices])) if indices[0].size else 0.0


def effective_rank(values: np.ndarray, *, centered: bool = False) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or min(arr.shape) == 0:
        return 0.0
    if centered:
        arr = arr - arr.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(arr, compute_uv=False)
    energy = singular * singular
    if float(energy.sum()) <= EPS:
        return 0.0
    probability = energy / energy.sum()
    return float(np.exp(-np.sum(probability * np.log(np.maximum(probability, EPS)))))


def pairwise_separation(coords: np.ndarray, domain_diagonal: float) -> float:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    delta = arr[:, None, :] - arr[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    values = distances[np.triu_indices(arr.shape[0], k=1)]
    return float(np.mean(values) / max(domain_diagonal, EPS))


def assignment_metrics(values: np.ndarray, prefix: str) -> dict[str, float]:
    p = normalized_rows(values)
    if p.size == 0:
        return {f"{prefix}_{name}": 0.0 for name in (
            "row_entropy_norm", "row_max", "row_effective_edges", "largest_dominant_occupancy",
            "row_cosine", "edge_column_cosine", "effective_rank", "centered_effective_rank",
        )}
    active_rows = p.sum(axis=1) > EPS
    p = p[active_rows]
    if p.size == 0:
        return assignment_metrics(np.zeros((0, values.shape[-1])), prefix)
    entropy = -np.sum(p * np.log(np.maximum(p, EPS)), axis=1)
    k = max(int(p.shape[1]), 1)
    dominant = np.argmax(p, axis=1)
    occupancy = np.bincount(dominant, minlength=k) / max(p.shape[0], 1)
    return {
        f"{prefix}_row_entropy_norm": float(np.mean(entropy) / math.log(max(k, 2))),
        f"{prefix}_row_max": float(np.mean(np.max(p, axis=1))),
        f"{prefix}_row_effective_edges": float(np.mean(np.exp(entropy))),
        f"{prefix}_largest_dominant_occupancy": float(np.max(occupancy)),
        f"{prefix}_row_cosine": mean_off_diagonal_cosine(p, columns=False),
        f"{prefix}_edge_column_cosine": mean_off_diagonal_cosine(p, columns=True),
        f"{prefix}_effective_rank": effective_rank(p),
        f"{prefix}_centered_effective_rank": effective_rank(p, centered=True),
    }


def neighbor_pairs(coords: np.ndarray) -> np.ndarray:
    """Return physical grid-neighbor pairs without assuming token sort order."""

    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.int64)
    pairs: set[tuple[int, int]] = set()
    for axis in (0, 1):
        other = 1 - axis
        levels = np.unique(np.round(points[:, other], decimals=7))
        for level in levels:
            indices = np.flatnonzero(np.isclose(points[:, other], level, atol=1.0e-6))
            order = indices[np.argsort(points[indices, axis])]
            pairs.update((int(min(a, b)), int(max(a, b))) for a, b in zip(order[:-1], order[1:]))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def neighbor_metrics(values: np.ndarray, coords: np.ndarray, prefix: str) -> dict[str, float]:
    p = normalized_rows(values)
    pairs = neighbor_pairs(coords)
    if p.size == 0 or pairs.size == 0:
        return {
            f"{prefix}_neighbor_dominant_agreement": 0.0,
            f"{prefix}_neighbor_l1": 0.0,
            f"{prefix}_spatial_smoothness": 0.0,
        }
    left, right = pairs[:, 0], pairs[:, 1]
    l1 = np.sum(np.abs(p[left] - p[right]), axis=-1)
    agreement = np.argmax(p[left], axis=-1) == np.argmax(p[right], axis=-1)
    return {
        f"{prefix}_neighbor_dominant_agreement": float(np.mean(agreement)),
        f"{prefix}_neighbor_l1": float(np.mean(l1)),
        f"{prefix}_spatial_smoothness": float(np.mean(1.0 - 0.5 * l1)),
    }


def organizer_metrics(
    aux: dict[str, Any],
    sample: dict[str, Any],
    *,
    pass_name: str,
    representation: str,
) -> dict[str, Any]:
    candidate = representation == "candidate"
    mh_key = "candidate_A_mh" if candidate and "candidate_A_mh" in aux else "A_mh"
    eh_key = "candidate_A_eh" if candidate and "candidate_A_eh" in aux else "A_eh"
    A_mh = np.asarray(aux.get(mh_key, np.zeros((0, 0))), dtype=np.float64)
    A_eh = np.asarray(aux.get(eh_key, np.zeros((0, 0))), dtype=np.float64)
    present = np.asarray(sample["structure"]["module_present"], dtype=np.float64) > 0.5
    A_mh = A_mh[present[: A_mh.shape[0]]] if A_mh.ndim == 2 else A_mh
    if not candidate and A_mh.ndim == 2:
        active = np.asarray(
            aux.get("effective_edge_mask", aux.get("edge_active_mask", np.ones(A_mh.shape[1]))),
            dtype=np.float64,
        ) > EPS
        A_mh = A_mh[:, active]
        A_eh = A_eh[:, active]
    else:
        active = np.ones(A_mh.shape[1] if A_mh.ndim == 2 else 0, dtype=bool)
    env_coords = np.asarray(aux.get("env_coords", np.zeros((A_eh.shape[0], 2))), dtype=np.float64)
    structure = sample["structure"]
    lx = float(np.asarray(structure.get("domain_length_x", 1.0)).reshape(-1)[0])
    ly = float(np.asarray(structure.get("domain_length_y", 1.0)).reshape(-1)[0])
    diagonal = math.hypot(lx, ly)
    source_key = "candidate_source_coords" if candidate else "hyper_source_coords"
    region_key = "candidate_region_coords" if candidate else "hyper_region_coords"
    source_scale_key = "candidate_source_scale" if candidate else "hyper_source_scale"
    region_scale_key = "candidate_region_scale" if candidate else "hyper_region_scale"
    source = np.asarray(aux.get(source_key, np.zeros((active.size, 2))), dtype=np.float64)
    region = np.asarray(aux.get(region_key, np.zeros((active.size, 2))), dtype=np.float64)
    source_scale = np.asarray(aux.get(source_scale_key, np.zeros((active.size, 2))), dtype=np.float64)
    region_scale = np.asarray(aux.get(region_scale_key, np.zeros((active.size, 2))), dtype=np.float64)
    if not candidate and active.size:
        source, region = source[active], region[active]
        source_scale, region_scale = source_scale[active], region_scale[active]
    row: dict[str, Any] = {
        "pass": pass_name,
        "representation": representation,
        "edge_count": int(A_mh.shape[1]) if A_mh.ndim == 2 else 0,
        "source_separation_normalized": pairwise_separation(source, diagonal),
        "region_separation_normalized": pairwise_separation(region, diagonal),
        "source_scale_normalized": float(np.mean(np.linalg.norm(source_scale, axis=-1)) / max(diagonal, EPS)) if source_scale.size else 0.0,
        "region_scale_normalized": float(np.mean(np.linalg.norm(region_scale, axis=-1)) / max(diagonal, EPS)) if region_scale.size else 0.0,
    }
    row.update(assignment_metrics(A_mh, "module"))
    row.update(assignment_metrics(A_eh, "environment"))
    row.update(neighbor_metrics(A_eh, env_coords, "environment"))
    for key in (
        "candidate_edge_count", "selected_edge_count", "viable_selected_edge_count",
        "hard_selected_edge_count", "selection_transition_fraction",
        "post_fallback_zero_support_module_rows", "post_fallback_zero_support_environment_rows",
    ):
        if key in aux:
            row[key] = float(np.asarray(aux[key]).mean())
    return row


def query_metrics(alpha: np.ndarray, pairwise: np.ndarray, shape: tuple[int, int]) -> dict[str, float]:
    p = normalized_rows(alpha)
    result = assignment_metrics(p, "query")
    if p.size:
        entropy = -np.sum(p * np.log(np.maximum(p, EPS)), axis=-1)
        mean_route = p.mean(axis=0, keepdims=True)
        result.update({
            "query_global_mean_route_l1": float(np.mean(np.sum(np.abs(p - mean_route), axis=-1))),
            "query_per_edge_spatial_std": float(np.mean(np.std(p, axis=0))),
        })
        grid = p.reshape(*shape, p.shape[-1])
        diffs = []
        if shape[0] > 1:
            diffs.append(np.abs(grid[1:] - grid[:-1]).sum(axis=-1).reshape(-1))
        if shape[1] > 1:
            diffs.append(np.abs(grid[:, 1:] - grid[:, :-1]).sum(axis=-1).reshape(-1))
        result["query_neighbor_l1"] = float(np.mean(np.concatenate(diffs))) if diffs else 0.0
        result["query_entropy_norm_direct"] = float(np.mean(entropy) / math.log(max(p.shape[1], 2)))
    else:
        result.update({"query_global_mean_route_l1": 0.0, "query_per_edge_spatial_std": 0.0, "query_neighbor_l1": 0.0, "query_entropy_norm_direct": 0.0})
    pair = np.asarray(pairwise, dtype=np.float64)
    result.update({
        "pairwise_edge_map_cosine": mean_off_diagonal_cosine(pair, columns=True),
        "pairwise_edge_map_effective_rank": effective_rank(pair),
        "pairwise_edge_map_cv": float(np.mean(np.std(pair, axis=0) / np.maximum(np.mean(np.abs(pair), axis=0), EPS))) if pair.size else 0.0,
    })
    return result


def additive_metrics(
    by_edge: np.ndarray,
    field_names: Sequence[str],
    edge_ids: Sequence[int] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    values = np.asarray(by_edge, dtype=np.float64)
    summary: dict[str, float] = {}
    edge_rows: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(field_names):
        maps = values[..., channel_index].reshape(-1, values.shape[-2])
        energy = np.mean(maps * maps, axis=0)
        fractions = energy / max(float(energy.sum()), EPS)
        summary[f"additive_{channel}_edge_map_cosine"] = mean_off_diagonal_cosine(maps, columns=True)
        summary[f"additive_{channel}_effective_rank"] = effective_rank(maps)
        summary[f"additive_{channel}_largest_energy_fraction"] = float(np.max(fractions)) if fractions.size else 0.0
        ids = list(range(len(fractions))) if edge_ids is None else list(edge_ids)
        for local_edge, fraction in enumerate(fractions):
            edge_rows.append({"channel": channel, "edge": ids[local_edge], "energy_fraction": float(fraction)})
    return summary, edge_rows


def physical_region_masks(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    _, fluid = module_and_fluid_masks(sample)
    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"], dtype=np.float32) > 0.5
    radius = module_radius_from_sample(sample)
    if np.any(present):
        surface = np.min(
            np.stack([np.hypot(x_grid - cx, y_grid - cy) - radius for cx, cy in centers[present]]),
            axis=0,
        )
    else:
        surface = np.full(x_grid.shape, np.inf, dtype=np.float32)
    fluid = np.asarray(fluid, dtype=bool)
    return {
        "whole": np.ones(x_grid.shape, dtype=bool),
        "fluid": fluid,
        "near_interface_fluid": fluid & (surface >= 0.0) & (surface <= 0.25),
        "far_field_fluid": fluid & (surface >= 1.0),
    }


def per_edge_metrics(
    prediction: dict[str, Any],
    aux: dict[str, Any],
    sample: dict[str, Any],
    field_names: Sequence[str],
    effective: np.ndarray,
) -> list[dict[str, Any]]:
    edge_ids = np.flatnonzero(effective)
    A_mh = normalized_rows(np.asarray(aux.get("A_mh", np.zeros((0, effective.size))), dtype=np.float64))[:, effective]
    A_eh = normalized_rows(np.asarray(aux.get("A_eh", np.zeros((0, effective.size))), dtype=np.float64))[:, effective]
    alpha = normalized_rows(prediction["alpha"][:, effective])
    pairwise = np.asarray(prediction["pairwise"][:, effective], dtype=np.float64)
    module_occupancy = np.bincount(np.argmax(A_mh, axis=-1), minlength=edge_ids.size) / max(A_mh.shape[0], 1) if A_mh.size else np.zeros(edge_ids.size)
    environment_occupancy = np.bincount(np.argmax(A_eh, axis=-1), minlength=edge_ids.size) / max(A_eh.shape[0], 1) if A_eh.size else np.zeros(edge_ids.size)
    query_occupancy = np.bincount(np.argmax(alpha, axis=-1), minlength=edge_ids.size) / max(alpha.shape[0], 1) if alpha.size else np.zeros(edge_ids.size)
    source = np.asarray(aux.get("hyper_source_coords", np.zeros((effective.size, 2))), dtype=np.float64)[effective]
    region = np.asarray(aux.get("hyper_region_coords", np.zeros((effective.size, 2))), dtype=np.float64)[effective]
    source_scale = np.asarray(aux.get("hyper_source_scale", np.zeros((effective.size, 2))), dtype=np.float64)[effective]
    region_scale = np.asarray(aux.get("hyper_region_scale", np.zeros((effective.size, 2))), dtype=np.float64)[effective]
    pair_mean = np.mean(np.abs(pairwise), axis=0) if pairwise.size else np.zeros(edge_ids.size)
    pair_fraction = pair_mean / max(float(pair_mean.sum()), EPS)
    base_rows: dict[int, dict[str, Any]] = {}
    for local_edge, edge in enumerate(edge_ids):
        base_rows[int(edge)] = {
            "edge": int(edge),
            "module_dominant_occupancy": float(module_occupancy[local_edge]),
            "environment_dominant_occupancy": float(environment_occupancy[local_edge]),
            "query_dominant_occupancy": float(query_occupancy[local_edge]),
            "query_mean_probability": float(np.mean(alpha[:, local_edge])) if alpha.size else 0.0,
            "pairwise_abs_mean": float(pair_mean[local_edge]),
            "pairwise_contribution_fraction": float(pair_fraction[local_edge]),
            "source_x": float(source[local_edge, 0]),
            "source_y": float(source[local_edge, 1]),
            "region_x": float(region[local_edge, 0]),
            "region_y": float(region[local_edge, 1]),
            "source_scale_x": float(source_scale[local_edge, 0]),
            "source_scale_y": float(source_scale[local_edge, 1]),
            "region_scale_x": float(region_scale[local_edge, 0]),
            "region_scale_y": float(region_scale[local_edge, 1]),
        }
    rows: list[dict[str, Any]] = []
    if not prediction["edge_fields_available"]:
        return [{**base, "region": "whole", "channel": "unavailable"} for base in base_rows.values()]
    fields = np.asarray(prediction["edge_fields"][..., effective, :], dtype=np.float64)
    for region_name, mask in physical_region_masks(sample).items():
        for channel_index, channel in enumerate(field_names):
            maps = fields[..., channel_index][mask]
            energy = np.mean(maps * maps, axis=0) if maps.size else np.zeros(edge_ids.size)
            fraction = energy / max(float(energy.sum()), EPS)
            for local_edge, edge in enumerate(edge_ids):
                rows.append({
                    **base_rows[int(edge)],
                    "region": region_name,
                    "channel": channel,
                    "additive_energy": float(energy[local_edge]),
                    "additive_energy_fraction": float(fraction[local_edge]),
                    "additive_is_dominant": float(local_edge == int(np.argmax(energy))) if energy.size else 0.0,
                })
    return rows


def best_permutation(reference: dict[str, Any], target: dict[str, Any]) -> tuple[list[int], float]:
    """Align anonymous edges with a deterministic composite minimum-cost match."""

    ref_m = normalized_rows(np.asarray(reference.get("A_mh", np.zeros((0, 0))))).T
    tgt_m = normalized_rows(np.asarray(target.get("A_mh", np.zeros((0, 0))))).T
    ref_e = normalized_rows(np.asarray(reference.get("A_eh", np.zeros((0, 0))))).T
    tgt_e = normalized_rows(np.asarray(target.get("A_eh", np.zeros((0, 0))))).T
    ref_c = np.asarray(reference.get("hyper_region_coords", np.zeros((ref_m.shape[0], 2))), dtype=np.float64)
    tgt_c = np.asarray(target.get("hyper_region_coords", np.zeros((tgt_m.shape[0], 2))), dtype=np.float64)
    k = min(ref_m.shape[0], tgt_m.shape[0])
    if k == 0:
        return [], 0.0

    def cosine_cost(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))

    cost = np.zeros((k, k), dtype=np.float64)
    coord_scale = max(float(np.ptp(np.concatenate([ref_c[:k], tgt_c[:k]], axis=0), axis=0).max()), EPS)
    for i in range(k):
        for j in range(k):
            cost[i, j] = (
                0.4 * cosine_cost(ref_m[i], tgt_m[j])
                + 0.4 * cosine_cost(ref_e[i], tgt_e[j])
                + 0.2 * float(np.linalg.norm(ref_c[i] - tgt_c[j]) / coord_scale)
            )
    best: tuple[int, ...] | None = None
    best_cost = float("inf")
    for permutation in itertools.permutations(range(k)):
        candidate = float(sum(cost[i, permutation[i]] for i in range(k)))
        if candidate < best_cost:
            best_cost, best = candidate, permutation
    return list(best or range(k)), best_cost / k


def pass_change_metrics(reference: dict[str, Any], target: dict[str, Any], name: str) -> dict[str, float]:
    permutation, cost = best_permutation(reference, target)
    if not permutation:
        return {f"{name}_alignment_cost": 0.0}
    ref_m = normalized_rows(np.asarray(reference["A_mh"], dtype=np.float64))
    ref_e = normalized_rows(np.asarray(reference["A_eh"], dtype=np.float64))
    tgt_m = normalized_rows(np.asarray(target["A_mh"], dtype=np.float64))[:, permutation]
    tgt_e = normalized_rows(np.asarray(target["A_eh"], dtype=np.float64))[:, permutation]
    return {
        f"{name}_alignment_cost": cost,
        f"{name}_module_assignment_l1": float(np.mean(np.sum(np.abs(ref_m - tgt_m), axis=-1))),
        f"{name}_environment_assignment_l1": float(np.mean(np.sum(np.abs(ref_e - tgt_e), axis=-1))),
    }


def predict_case(model: Any, sample: dict[str, Any], device: torch.device, query_batch_size: int, organizer_passes: bool) -> dict[str, Any]:
    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    queries = np.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
    prepared = None
    first: dict[str, Any] | None = None
    pred_chunks: list[np.ndarray] = []
    alpha_chunks: list[np.ndarray] = []
    pair_chunks: list[np.ndarray] = []
    edge_chunks: list[np.ndarray] = []
    c_h_chunks: list[np.ndarray] = []
    c_pair_chunks: list[np.ndarray] = []
    edge_fields_available = True
    with torch.inference_mode():
        for start in range(0, queries.shape[0], query_batch_size):
            query = queries[start : start + query_batch_size]
            if prepared is None:
                batch = make_batch(sample, query, device)
                output = model(
                    batch["structure"], batch["query_xy"],
                    interface_condition=batch.get("interface_condition"),
                    local_module_params=batch.get("local_module_params"),
                    teacher_port_tokens=batch.get("teacher_port_tokens"),
                    local_query_points=batch.get("module_internal_query_points"),
                    local_port_condition_mode="predicted",
                    return_routing_maps=True,
                    return_edge_fields=True,
                    return_prepared_state=True,
                    return_organizer_passes=organizer_passes,
                )
                prepared = output.pop("prepared_state")
                first = output
            else:
                tensor = torch.from_numpy(query).unsqueeze(0).to(device=device)
                output = model.decode_prepared(prepared, tensor, return_routing_maps=True, return_edge_fields=True)
            routing = output.get("routing_aux", output)
            pred_chunks.append(output["pred_field"][0].detach().cpu().numpy())
            alpha_chunks.append(routing["query_hyper_attention"][0].detach().cpu().numpy())
            pair_chunks.append(routing["pairwise_edge_contribution"][0].detach().cpu().numpy())
            c_h_chunks.append(routing["c_H_norm"][0].detach().cpu().numpy())
            c_pair_chunks.append(routing["c_pair_norm"][0].detach().cpu().numpy())
            edge_field = output.get("pred_field_by_edge", routing.get("pred_field_by_edge"))
            if torch.is_tensor(edge_field):
                edge_chunks.append(edge_field[0].detach().cpu().numpy())
            else:
                edge_fields_available = False
                edge_chunks.append(np.zeros((query.shape[0], alpha_chunks[-1].shape[-1], pred_chunks[-1].shape[-1]), dtype=np.float32))
    if first is None:
        raise RuntimeError("No query chunks were decoded.")
    pred = np.concatenate(pred_chunks).reshape(*x_grid.shape, -1)
    alpha = np.concatenate(alpha_chunks)
    pair = np.concatenate(pair_chunks)
    edges = np.concatenate(edge_chunks).reshape(*x_grid.shape, edge_chunks[0].shape[-2], edge_chunks[0].shape[-1])
    return {
        "pred": pred,
        "alpha": alpha,
        "pairwise": pair,
        "c_H_norm": np.concatenate(c_h_chunks),
        "c_pair_norm": np.concatenate(c_pair_chunks),
        "edge_fields": edges,
        "edge_fields_available": edge_fields_available,
        "final": numpy_aux(first["organizer_aux"]),
        "base": numpy_aux(first.get("base_organizer_aux", {})),
        "provisional": numpy_aux(first.get("provisional_organizer_aux", {})),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], group_keys: Sequence[str]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)
    result: dict[str, Any] = {}
    for group, members in groups.items():
        label = "/".join(str(value) for value in group)
        result[label] = {}
        numeric = sorted({key for member in members for key, value in member.items() if isinstance(value, (int, float, np.number)) and key not in group_keys})
        for key in numeric:
            values = np.asarray([float(member[key]) for member in members if key in member and np.isfinite(float(member[key]))])
            if values.size:
                result[label][key] = {"mean": float(values.mean()), "median": float(np.median(values)), "p05": float(np.quantile(values, 0.05)), "p95": float(np.quantile(values, 0.95))}
    return result


def render_margin(path: Path, sample: dict[str, Any], alpha: np.ndarray) -> None:
    grid = normalized_rows(alpha).reshape(*np.asarray(sample["x_grid"]).shape, -1)
    ordered = np.sort(grid, axis=-1)
    margin = ordered[..., -1] - ordered[..., -2] if grid.shape[-1] > 1 else ordered[..., -1]
    extent = [float(np.min(sample["x_grid"])), float(np.max(sample["x_grid"])), float(np.min(sample["y_grid"])), float(np.max(sample["y_grid"]))]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    image = ax.imshow(margin, origin="lower", extent=extent, cmap="cividis", vmin=0.0, vmax=max(float(np.max(margin)), EPS), aspect="auto")
    ax.set(title="Dominant-edge routing margin", xlabel="x", ylabel="y")
    fig.colorbar(image, ax=ax, label="largest route probability minus second-largest")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_case(output_dir: Path, sample: dict[str, Any], prediction: dict[str, Any], field_names: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = extract_organization_arrays(sample, prediction["final"])
    radius = module_radius_from_sample(sample)
    render_channelthermal_organization_summary_matrices(
        output_dir / "organization_matrices_physical_order.png", sample, arrays,
        module_radius=radius, channel_order=field_names, sort_environment=False,
    )
    render_channelthermal_organization_summary_matrices(
        output_dir / "organization_matrices_sorted_by_dominant_edge.png", sample, arrays,
        module_radius=radius, channel_order=field_names, sort_environment=True,
    )
    shutil.copy2(
        output_dir / "organization_matrices_physical_order.png",
        output_dir / "organization_environment_unsorted.png",
    )
    shutil.copy2(
        output_dir / "organization_matrices_sorted_by_dominant_edge.png",
        output_dir / "organization_environment_sorted_by_dominant_edge.png",
    )
    render_channelthermal_organization_overview(
        output_dir / "organization_overview.png", sample, arrays,
        module_radius=radius, channel_order=field_names,
    )
    shape = np.asarray(sample["x_grid"]).shape
    alpha = prediction["alpha"].reshape(*shape, -1)
    pair = prediction["pairwise"].reshape(*shape, -1)
    routing = {
        "query_hyper_attention": alpha,
        "pairwise_edge_contribution": pair,
        "c_H_norm": prediction["c_H_norm"].reshape(shape),
        "c_pair_norm": prediction["c_pair_norm"].reshape(shape),
        "dominant_hyperedge": np.argmax(alpha, axis=-1),
        "hyper_attention_entropy": -np.sum(alpha * np.log(np.maximum(alpha, EPS)), axis=-1),
    }
    save_routing_diagnostics(output_dir, sample, routing, arrays, module_radius=radius, routing_view="all")
    render_margin(output_dir / "routing_dominant_margin.png", sample, prediction["alpha"])
    structure = sample["structure"]
    signature = extract_topology_signature(
        prediction["final"],
        structure["module_present"],
        decoder_outputs={
            "query_hyper_attention": prediction["alpha"],
            "pred_field_by_edge": prediction["edge_fields"].reshape(
                -1, prediction["edge_fields"].shape[-2], prediction["edge_fields"].shape[-1]
            ),
        }
        if prediction["edge_fields_available"]
        else {"query_hyper_attention": prediction["alpha"]},
        reference_query_xy=np.stack(
            (np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)),
            axis=-1,
        ),
        reference_measure="channelthermal_evaluation_grid",
        field_names=field_names,
        domain_length_x=float(np.asarray(structure.get("domain_length_x", [0.0])).reshape(-1)[0]),
        domain_length_y=float(np.asarray(structure.get("domain_length_y", [0.0])).reshape(-1)[0]),
        case_id=str(sample["case_id"]),
    )
    render_topology_signature_diagnostics(
        output_dir,
        sample,
        signature,
        edge_fields=(
            prediction["edge_fields"].reshape(
                -1, prediction["edge_fields"].shape[-2], prediction["edge_fields"].shape[-1]
            )
            if prediction["edge_fields_available"]
            else None
        ),
        field_names=field_names,
    )


def evaluate_checkpoint(label: str, path: Path, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model, checkpoint = load_model(path, device)
    dataset_cfg = checkpoint.get("train_config", {}).get("dataset", {})
    model_cfg = checkpoint.get("model_config", {})
    core_cfg = model_cfg.get("core_honf", {})
    training_cfg = checkpoint.get("train_config", {}).get("training", {})
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint.get("global_normalization_stats", {}).items()}
    dataset = GlobalChannelThermalDataset(
        dataset_cfg["packed_h5_path"], split=args.split, points_per_case=1,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False, include_grid=True, include_structure_targets=False,
        normalizer=H5Normalizer(stats),
    )
    requested = {str(case_id) for case_id in args.case_id}
    indices = [index for index, case_id in enumerate(dataset.selected_case_ids) if not requested or str(case_id) in requested]
    if args.max_cases is not None:
        indices = indices[: int(args.max_cases)]
    for required_case_id in args.include_case_id:
        matches = [index for index, case_id in enumerate(dataset.selected_case_ids) if str(case_id) == str(required_case_id)]
        if not matches:
            raise KeyError(f"case_id={required_case_id!r} is absent from split {args.split!r}")
        if matches[0] not in indices:
            indices.append(matches[0])
    field_names = list(model.config.channelthermal.field_names)
    if field_names != list(dataset.channel_order):
        raise RuntimeError(f"Channel-order mismatch for {label}: {field_names} != {dataset.channel_order}")
    provenance = {
        "label": label, "checkpoint": str(path), "sha256": sha256_file(path),
        "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "selection_state": checkpoint.get("selection_state"), "field_names": field_names,
        "dataset": str(dataset_cfg["packed_h5_path"]), "dataset_fingerprint": dataset_cfg.get("dataset_fingerprint"),
        "organizer_mode": core_cfg.get("organizer_mode"),
        "edge_capacity": core_cfg.get("edge_capacity", core_cfg.get("num_hyperedges")),
        "num_hyperedges": core_cfg.get("num_hyperedges"),
        "module_assignment_normalizer": core_cfg.get("module_assignment_normalizer"),
        "environment_assignment_normalizer": core_cfg.get("environment_assignment_normalizer"),
        "query_assignment_normalizer": core_cfg.get("query_assignment_normalizer"),
        "mechanism_state_mode": core_cfg.get("mechanism_state_mode"),
        "field_assembly_mode": core_cfg.get("field_assembly_mode"),
        "additive_background_mode": core_cfg.get("additive_background_mode"),
        "learning_rate": training_cfg.get("learning_rate"),
        "organizer_learning_rate": training_cfg.get("organizer_learning_rate"),
        "model_config": model_cfg, "train_config": checkpoint.get("train_config", {}),
        "evaluated_case_count": len(indices),
    }
    provenance.update(run_manifest_provenance(path))
    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    render_ids = {str(case_id) for case_id in args.render_case_id}
    for position, index in enumerate(indices, start=1):
        sample = dataset[index]
        case_id = str(sample["case_id"])
        prediction = predict_case(model, sample, device, args.query_batch_size, args.organizer_passes)
        passes = [("final", prediction["final"])]
        if args.organizer_passes:
            passes = [("base", prediction["base"])]
            if prediction["provisional"]:
                passes.append(("provisional", prediction["provisional"]))
            passes.append(("final", prediction["final"]))
        final_row: dict[str, Any] | None = None
        for pass_name, aux in passes:
            for representation in ("candidate", "selected"):
                row = {"checkpoint": label, "case_id": case_id}
                row.update(organizer_metrics(aux, sample, pass_name=pass_name, representation=representation))
                if pass_name == "final" and representation == "selected":
                    effective = np.asarray(
                        aux.get("effective_edge_mask", aux.get("edge_active_mask", np.ones(prediction["alpha"].shape[-1]))),
                        dtype=np.float64,
                    ) > EPS
                    if not np.any(effective):
                        effective = np.ones(prediction["alpha"].shape[-1], dtype=bool)
                    edge_ids = np.flatnonzero(effective).tolist()
                    row.update(query_metrics(
                        prediction["alpha"][:, effective],
                        prediction["pairwise"][:, effective],
                        np.asarray(sample["x_grid"]).shape,
                    ))
                    per_edge: list[dict[str, Any]] = []
                    if prediction["edge_fields_available"]:
                        additive, per_edge = additive_metrics(
                            prediction["edge_fields"][..., effective, :],
                            field_names,
                            edge_ids,
                        )
                        row.update(additive)
                    target = np.asarray(sample["steady_field"], dtype=np.float64)
                    if bool(dataset_cfg.get("normalize_targets", False)):
                        target = dataset.normalizer.normalize_fields(target)
                    row["normalized_field_mse"] = float(np.mean((prediction["pred"] - target) ** 2))
                    final_row = row
                    per_edge = per_edge_metrics(
                        prediction, aux, sample, field_names, effective
                    )
                    for edge_item in per_edge:
                        edge_rows.append({"checkpoint": label, "case_id": case_id, **edge_item})
                rows.append(row)
        if final_row is not None and args.organizer_passes:
            final_row.update(pass_change_metrics(prediction["base"], prediction["final"], "base_to_final"))
            if prediction["provisional"]:
                final_row.update(pass_change_metrics(prediction["base"], prediction["provisional"], "base_to_provisional"))
                final_row.update(pass_change_metrics(prediction["provisional"], prediction["final"], "provisional_to_final"))
        if case_id in render_ids:
            render_case(args.output_dir / "figures" / label / case_id, sample, prediction, field_names)
        print(f"[{label}] {position}/{len(indices)} case={case_id}", flush=True)
    dataset.close()
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return provenance, rows, edge_rows


def main() -> int:
    args = parse_args()
    specs = checkpoint_specs(args.checkpoint)
    for path in specs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    provenance: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for label, path in specs.items():
        info, checkpoint_rows, checkpoint_edge_rows = evaluate_checkpoint(label, path, args, device)
        provenance[label] = info
        rows.extend(checkpoint_rows)
        edge_rows.extend(checkpoint_edge_rows)
    per_case_path = args.output_dir / "topology_quality_per_case.csv"
    per_edge_path = args.output_dir / "topology_quality_per_case_edge.csv"
    summary_path = args.output_dir / "topology_quality_summary.json"
    write_csv(per_case_path, rows)
    write_csv(per_edge_path, edge_rows)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "script": str(Path(__file__).resolve()), "device": str(device), "split": args.split,
        "provenance": provenance,
        "summary": aggregate(rows, ("checkpoint", "pass", "representation")),
        "per_edge_summary": aggregate(edge_rows, ("checkpoint", "channel", "edge")),
        "artifacts": {"per_case": str(per_case_path), "per_case_edge": str(per_edge_path)},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
