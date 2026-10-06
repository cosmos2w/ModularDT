#!/usr/bin/env python3
"""Measure sealed TRAIN component gradients at a saved thermal checkpoint.

No optimizer update, coefficient search, DEV response, or physical solve occurs.
The fixed calibration cases/families come from the original paired recipe.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "Case_ThermalChannel/src", ROOT / "tools"):
    sys.path.insert(0, str(source))

import torch
from channelthermal.source_response import load_source_response_model
from channelthermal.training.checkpoints import _file_sha256
from thermal_development import validate_generated_output
from thermal_source_response_evaluate import model_digest, synchronize
from thermal_source_response_fit import (
    build_balances,
    operator_loss,
    read_primary,
    read_response_families,
    reconstruction_terms,
    sample_primary,
)

from honf_runtime.run_store import atomic_write_json


def gradient_vector(loss, parameters):
    values = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat([torch.zeros_like(parameter).flatten() if value is None else value.flatten()
                      for parameter, value in zip(parameters, values)]).detach()


def measure(model, metadata, device):
    thermal = model.thermal
    thermal.requires_grad_(True)
    recipe = metadata["fit_identity"]["recipe"]
    calibration = recipe["calibration"]
    cases, _ = read_primary(metadata, "train")
    by_id = {case["case_id"]: index for index, case in enumerate(cases)}
    selected = [by_id[case_id] for case_id in calibration["calibration_case_ids"]]
    families = read_response_families(recipe["response_atlas_directory"])
    sources = [{"family_id": f["family_id"], "path": f["source"], "sha256": f["source_sha256"]} for f in families]
    if sources != recipe["response_sources"]:
        raise ValueError("Gradient diagnostics cannot change sealed TRAIN response sources.")
    use_operator = calibration["operator_constraint"] == "qualified"
    balances = build_balances(cases) if use_operator else []
    parameters = tuple(thermal.parameters())
    vectors = {name: [] for name in ("temperature", "q_auxiliary", "response", "operator")}
    work = {"operator_rows": 0, "operator_source_columns": 0, "neural_stencil_receiver_rows": 0,
            "response_neural_receiver_rows": 0, "native_reconstruction_receiver_rows": 0}
    stats = metadata["global_normalization_stats"]
    for index in selected:
        for component in ("temperature", "q_auxiliary", "operator"):
            if component == "operator" and not use_operator:
                continue
            batch = sample_primary(cases, [index], 1, device)
            terms, prepared, prediction = reconstruction_terms(thermal, batch, stats)
            if component == "temperature":
                loss = terms["thermal_selector"].mean()
            elif component == "q_auxiliary":
                loss = .05*terms["q_proxy"].mean()
            else:
                loss, receipt = operator_loss(thermal, prepared, balances, [index], 1, device, stats)
                for key in ("operator_rows", "operator_source_columns", "neural_stencil_receiver_rows"):
                    work[key] += receipt[key]
                loss = calibration["operator_coefficient"]*loss
            work["native_reconstruction_receiver_rows"] += int(prediction["native_neural_receiver_rows"])
            vectors[component].append(gradient_vector(loss, parameters))
    from thermal_source_response_fit import response_loss
    for family in families:
        loss, receipt = response_loss(thermal, family, 1, device, calibration["response_scales"])
        work["response_neural_receiver_rows"] += receipt["response_neural_receiver_rows"]
        vectors["response"].append(gradient_vector(calibration["response_coefficient"]*loss, parameters))
    means = {name: torch.stack(values).mean(0) for name, values in vectors.items() if values}
    norms = {name: {"pooled_mean_vector_norm": float(value.norm()),
                    "individual_norms": [float(item.norm()) for item in vectors[name]]}
             for name, value in means.items()}
    cosines = {}
    for name, value in means.items():
        if name == "temperature":
            continue
        denom = value.norm()*means["temperature"].norm()
        cosines["temperature_vs_"+name] = float(value@means["temperature"]/denom) if denom > 0 else None
    total = sum(means.values())
    cursor, grouping = 0, {}
    for name, parameter in thermal.named_parameters():
        size = parameter.numel()
        if any(piece in name for piece in ("membership_score", "group_collective")):
            grouping[name] = float(total[cursor:cursor+size].norm())
        cursor += size
    return {"checkpoint_epoch": metadata["epoch"], "mode": metadata["fit_identity"]["mode"],
            "case_ids": calibration["calibration_case_ids"], "response_family_ids": recipe["response_fit_families"],
            "fixed_sampling_epoch": 1, "component_gradient_norms": norms, "gradient_cosines": cosines,
            "group_parameter_gradient_norms": grouping, "actual_work": work,
            "scope": "Fixed TRAIN checkpoint probes; pooled gradients are averages of equal-case or equal-family gradients, not ordinary optimizer-boundary gradients",
            "coefficient_searches": 0, "optimizer_updates": 0, "physical_solves": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    output = validate_generated_output(args.output)
    if output.exists():
        raise FileExistsError("Preserve the previous gradient measurement.")
    started = perf_counter()
    model, metadata = load_source_response_model(args.checkpoint, args.device)
    model.eval()
    before = model_digest(model)
    result = measure(model, metadata, torch.device(args.device))
    synchronize(args.device)
    if model_digest(model) != before:
        raise RuntimeError("Gradient diagnostics changed saved weights.")
    result.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=_file_sha256(args.checkpoint),
                  measured_seconds=perf_counter()-started, device=args.device, weights_unchanged=True)
    atomic_write_json(output, result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
