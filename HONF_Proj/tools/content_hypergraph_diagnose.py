#!/usr/bin/env python3
"""Bounded, target-free geometry/value/content diagnostics on two TRAIN scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "tools", PROJECT / "Case_ThermalChannel/src", PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from content_hypergraph_benchmark import choose_low_high_train_keys, native_train_batch, validate_device_binding
from content_hypergraph_evaluate import json_safe, load_evaluation_components
from content_hypergraph_inspect import slice_query
from content_hypergraph_train import build, canonical_sha256, validate_recipe

from honf_forward_core.content_hypergraph import QueryBatch


def _em_probe(core, scene):
    """Capture actual first EM projected donors, weights and weighted values."""
    route = core.context_blocks[0].em
    captures, handles = {}, []
    for name in ("query", "key", "value"):
        handles.append(getattr(route, name).register_forward_hook(
            lambda _module, _args, output, name=name: captures.__setitem__(name, output)))
    geometry = getattr(route, "geometry_score", None)
    if geometry is None:
        geometry = route.geometry_bias
    handles.append(geometry.register_forward_hook(lambda _module, _args, output: captures.__setitem__("geometry", output)))
    handles.append(core.module_encoder.register_forward_hook(
        lambda _module, _args, output: captures.__setitem__("raw_module", output)))
    try:
        organization = core.prepare(scene)
    finally:
        for handle in handles:
            handle.remove()
    batch, destinations, width = captures["query"].shape
    donors = captures["key"].shape[1]
    heads = core.num_heads
    q, k, value = (captures[name].reshape(batch, -1, heads, width // heads).transpose(1, 2)
                   for name in ("query", "key", "value"))
    if core.architecture.startswith("A2-"):
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6) * route.temperature[None, :, None, None]
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
        logits = q @ k.transpose(-1, -2)
    else:
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(width // heads)
    geometry = captures["geometry"].permute(0, 3, 1, 2)
    valid = scene.module_mask & (scene.module_measure > 0)
    measure = torch.where(valid, scene.module_measure, torch.ones_like(scene.module_measure))
    logits = logits + geometry + measure.log()[:, None, None, :]
    allowed = scene.environment_mask[:, None, :, None] & valid[:, None, None, :]
    nonempty = allowed.any(-1, keepdim=True)
    masked = logits.masked_fill(~allowed, -torch.inf)
    masked = torch.where(nonempty, masked, torch.zeros_like(masked))
    weights = masked.softmax(-1) * allowed.to(logits.dtype)
    weighted = weights @ value
    return organization, captures, weights, weighted, logits.masked_select(allowed.expand_as(logits))


def diagnose_scene(model, provider, batch, dataset, receiver_count=128):
    core = model.core if dataset == "thermal" else model
    native_scene = provider.make_scene(batch.scene_inputs)
    if dataset == "thermal":
        scene = model.scene_batch(native_scene.structure, scene_key=native_scene.case_ids)
        points = batch.receivers.fluid_xy
        indices = torch.linspace(0, points.shape[1] - 1, receiver_count, device=points.device).long().unique()
        query = QueryBatch(coordinates=points[:, indices], query_ids=indices[None, :],
                           scene_key=scene.scene_key, coordinate_frame_key=scene.coordinate_frame_key)
    else:
        scene = native_scene
        native_query = provider.adapter.make_query_batch(scene, batch.receivers)
        indices = np.unique(np.linspace(0, native_query.num_queries - 1, receiver_count, dtype=np.int64))
        query = slice_query(native_query, indices)
    coordinates = scene.module_coordinates.detach().clone().requires_grad_(True)
    scene = replace(scene, module_coordinates=coordinates)
    organization, captured, weights, weighted, scores = _em_probe(core, scene)
    changed = coordinates.detach().clone()
    changed[:, 0, 0] += .025 * scene.coordinate_scale[:, 0]
    _, moved_capture, moved_weights, moved_weighted, _ = _em_probe(core, replace(scene, module_coordinates=changed))
    controls = scene.module_measure.new_ones(scene.module_measure.shape) if core.thermal_response_rank else None
    output = core.decode(organization, query, controls=controls)
    objective = output.nonlinear_field.square().mean()
    if output.thermal_temperature is not None:
        objective = objective + output.thermal_temperature.square().mean()
    gradient = torch.autograd.grad(objective, coordinates, retain_graph=False, allow_unused=True)[0]
    active = scene.module_mask[0]

    def donor_variance(value):
        selected = value[0, active].detach().double()
        return float(selected.var(0, unbiased=False).mean()) if selected.shape[0] else 0.0

    result = {"case_keys": list(batch.case_keys), "active_modules": int(active.sum()),
              "raw_module_encoding_variance": donor_variance(captured["raw_module"]),
              "first_EM_donor_value_variance": donor_variance(captured["value"]),
              "coordinate_channel_perturbation": {"rule": "source0 x += .025*domain_extent_x; other prescribed attributes held fixed; software path attribution, not a physical new scene",
                  "attention_weight_delta_l2": float(torch.linalg.vector_norm(moved_weights - weights)),
                  "weighted_value_delta_l2": float(torch.linalg.vector_norm(moved_weighted - weighted)),
                  "donor_value_delta_l2": float(torch.linalg.vector_norm(moved_capture["value"] - captured["value"]))},
              "first_EM_complete_logit_min": float(scores.min()), "first_EM_complete_logit_max": float(scores.max()),
              "first_EM_complete_logit_std": float(scores.std(unbiased=False)),
              "coordinate_gradient_l2": None if gradient is None else float(torch.linalg.vector_norm(gradient)),
              "coordinate_gradient_finite": gradient is not None and bool(torch.isfinite(gradient).all()),
              "requested_receiver_count": query.num_queries, "target_values_passed_to_model": False}
    w = getattr(organization, "module_gather_weights", None)
    if w is None:
        w = organization.module_pooling_weights
    if w.numel():
        w64 = w.detach().double()
        result["actual_module_W_maximum_donor_weight"] = w64.max(-1).values.cpu().tolist()
        result["actual_module_W_effective_donors"] = (-(w64 * w64.clamp_min(1e-300).log()).sum(-1)).exp().cpu().tolist()
        result["gather_semantics"] = "direct edge-to-donor W" if core.architecture.startswith("A2-") else "v1 incidence quotient pooling W"
    c = organization.source_coefficients
    if c is not None:
        result["C_edge_squared_energy"] = c.detach().double().square().sum((-1, -2)).cpu().tolist()
        empty = torch.empty(0, dtype=torch.long, device=query.coordinates.device)
        far_query = replace(query, local_batch_indices=empty, local_query_indices=empty,
                            local_module_indices=empty, local_weights=query.coordinates.new_empty(0))
        # Rebuild a no-grad preparation after the gradient probe released its graph.
        with torch.no_grad():
            fresh = core.prepare(replace(scene, module_coordinates=coordinates.detach()))
            kernels = core.export_source_kernels(fresh, far_query, accumulation_dtype=torch.float64)[0, :, active]
            singular = torch.linalg.svdvals(kernels.double())
        energy = singular.square()
        result["far_K_singular_values"] = singular.cpu().tolist()
        result["far_K_energy_fraction"] = (energy / energy.sum().clamp_min(1e-300)).cpu().tolist()
        result["far_K_numerical_rank_rtol1e-7"] = int((singular > singular.max() * 1e-7).sum())
        result["far_K_shape"] = list(kernels.shape)
        result["far_K_scope"] = "explicit bounded software diagnostic; local path removed; model-space temperature per physical heat"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    validate_device_binding(torch.device(args.device))
    if args.output.exists():
        raise FileExistsError("Diagnostics are immutable; select a new output path.")
    if args.recipe:
        recipe = validate_recipe(json.loads(args.recipe.read_text()))
        model, provider = build(recipe, device=args.device)
        checkpoint_hash, epoch = None, 0
    else:
        model, provider, payload = load_evaluation_components(args.checkpoint, args.device)
        recipe = payload["experiment_identity"]["recipe"]
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        epoch = int(payload["epoch"])
    model.eval()
    selection = choose_low_high_train_keys(provider, recipe["dataset"])
    cases = {name: diagnose_scene(model, provider, native_train_batch(provider, row["case_key"]), recipe["dataset"])
             for name, row in selection.items()}
    result = {"schema": "content_hypergraph_geometry_content_diagnostic_v2", "status": "completed",
              "dataset": recipe["dataset"], "architecture": recipe["architecture"], "epoch": epoch,
              "checkpoint_sha256": checkpoint_hash, "model_config_sha256": canonical_sha256(model.get_config()),
              "fixture_selection": selection, "cases": cases, "optimizer_updates": 0,
              "target_scope": "two input-selected TRAIN scenes; no DEV or TEST targets", "solver_usage": "326/326"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "dataset", "architecture", "epoch", "optimizer_updates")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
