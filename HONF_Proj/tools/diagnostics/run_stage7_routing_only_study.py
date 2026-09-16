#!/usr/bin/env python3
"""Focused Run-1404 frozen-function and real-batch gradient diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "Case_ThermalChannel" / "src"))

from channelthermal.evaluation.loading import load_model, make_batch
from channelthermal.evaluation.prepared import predict_case
from channelthermal.training.epoch import (
    effective_local_loss_weights,
    effective_port_condition_settings,
    effective_port_global_weight,
    interface_loss,
    internal_loss,
    organizer_regularization,
    port_condition_loss,
    port_cyclic_smoothness_loss,
    port_global_consistency_loss,
    predicted_consistency_weight_for_epoch,
)
from channelthermal.training_tools.losses import channelthermal_field_mse
from evaluate_case_adaptive_residual import (
    _checkpoint_dataset,
    _selected_indices,
    _target_field,
    evaluate_case,
)


VARIANTS = {
    "normal_1401": {"use_hyper_value_context": True, "pairwise_module_token_source": "base"},
    "no_cH_base_pair_token": {"use_hyper_value_context": False, "pairwise_module_token_source": "base"},
    "no_cH_contextual_pair_token": {
        "use_hyper_value_context": False,
        "pairwise_module_token_source": "organizer_contextualized",
    },
}

PAIR_INTERVENTIONS = (
    "normal",
    "uniform_query_to_edge",
    "uniform_module_to_edge",
    "base_pair_module_token",
    "suppress_pair_context",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    frozen = subparsers.add_parser("frozen")
    frozen.add_argument("--checkpoint", type=Path, required=True)
    frozen.add_argument("--output-dir", type=Path, required=True)
    frozen.add_argument("--device", default="cuda:0")
    frozen.add_argument("--query-batch-size", type=int, default=32768)
    frozen.add_argument("--case-id", action="append", required=True)
    gradient = subparsers.add_parser("gradient")
    gradient.add_argument("--checkpoint", type=Path, required=True)
    gradient.add_argument("--output", type=Path, required=True)
    gradient.add_argument("--device", default="cuda:0")
    gradient.add_argument("--case-id", required=True)
    gradient.add_argument("--query-count", type=int, default=8192)
    interventions = subparsers.add_parser("interventions")
    interventions.add_argument("--checkpoint", type=Path, required=True)
    interventions.add_argument("--output-dir", type=Path, required=True)
    interventions.add_argument("--device", default="cuda:0")
    interventions.add_argument("--query-batch-size", type=int, default=32768)
    interventions.add_argument("--case-id", action="append", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def core_configs(model: Any) -> list[Any]:
    candidates = [model.core.config, model.core.decoder.config, model.core.decoder.pairwise_kernel.config]
    wrapper = getattr(model.config, "core_honf", None)
    if wrapper is not None:
        candidates.append(wrapper)
    result: list[Any] = []
    for candidate in candidates:
        if not any(candidate is current for current in result):
            result.append(candidate)
    return result


@contextmanager
def arithmetic_variant(model: Any, settings: dict[str, Any]) -> Iterator[None]:
    configs = core_configs(model)
    originals = [{key: getattr(config, key) for key in settings} for config in configs]
    try:
        for config in configs:
            for key, value in settings.items():
                setattr(config, key, value)
        yield
    finally:
        for config, values in zip(configs, originals):
            for key, value in values.items():
                setattr(config, key, value)


@contextmanager
def pair_intervention(model: Any, mode: str) -> Iterator[None]:
    """Apply one frozen Run-1404 routing or pair-value intervention."""

    if mode == "normal":
        yield
        return
    if mode == "base_pair_module_token":
        with arithmetic_variant(model, {"pairwise_module_token_source": "base"}):
            yield
        return
    kernel = model.core.decoder.pairwise_kernel
    original = kernel.forward

    def forward(*call_args: Any, **kwargs: Any):
        values = list(call_args)
        organizer_output = values[1]
        hyper_attention = values[2]
        if mode == "uniform_query_to_edge":
            edge_mask = organizer_output.get("effective_edge_mask")
            if not torch.is_tensor(edge_mask):
                edge_mask = torch.ones_like(hyper_attention[:, 0, :])
            edge_mask = edge_mask.to(device=hyper_attention.device, dtype=hyper_attention.dtype)
            uniform = edge_mask[:, None, :] / edge_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)[:, None, :]
            values[2] = uniform.expand_as(hyper_attention)
        elif mode == "uniform_module_to_edge":
            copied = dict(organizer_output)
            incidence = organizer_output["A_mh"]
            module_mask = organizer_output["module_present"].to(device=incidence.device, dtype=incidence.dtype)
            edge_mask = organizer_output.get("effective_edge_mask")
            if not torch.is_tensor(edge_mask):
                edge_mask = torch.ones_like(incidence[:, 0, :])
            edge_mask = edge_mask.to(device=incidence.device, dtype=incidence.dtype)
            copied["A_mh"] = (
                module_mask[:, :, None]
                * edge_mask[:, None, :]
                / edge_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)[:, None, :]
            )
            values[1] = copied
        result = original(*values, **kwargs)
        if mode == "suppress_pair_context":
            pair_context, edge_context, diagnostics = result
            result = (torch.zeros_like(pair_context), edge_context, diagnostics)
        return result

    if mode not in {"uniform_query_to_edge", "uniform_module_to_edge", "suppress_pair_context"}:
        raise ValueError(f"unknown pair intervention: {mode}")
    kernel.forward = forward
    try:
        yield
    finally:
        kernel.forward = original


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"case_count": len(rows)}
    for key in rows[0] if rows else ():
        values = [number for row in rows if (number := finite(row.get(key))) is not None]
        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_maximum"] = float(np.max(values))
    return result


def frozen(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint.expanduser().resolve(), device)
    dataset = _checkpoint_dataset(checkpoint, "test")
    indices = _selected_indices(dataset, args.case_id, None)
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    try:
        for variant, settings in VARIANTS.items():
            with arithmetic_variant(model, settings):
                for index in indices:
                    row = evaluate_case(
                        variant,
                        model,
                        checkpoint,
                        dataset,
                        index,
                        device,
                        query_batch_size=args.query_batch_size,
                        return_routing_maps=True,
                        benchmark=False,
                        stability_perturbations=False,
                        stability_repeats=0,
                    )
                    rows.append(row)
                    sample = dataset[index]
                    prediction = predict_case(
                        model,
                        sample,
                        device,
                        query_batch_size=args.query_batch_size,
                        local_port_condition_mode="predicted",
                        mixed_teacher_ratio=0.5,
                        return_routing_maps=False,
                    )
                    predictions[(variant, str(sample["case_id"]))] = np.asarray(
                        prediction["pred_field_grid"], dtype=np.float64
                    )
    finally:
        dataset.close()

    reference = "normal_1401"
    for row in rows:
        variant = str(row["checkpoint"])
        case_id = str(row["case_id"])
        candidate = predictions[(variant, case_id)]
        normal = predictions[(reference, case_id)]
        difference = candidate - normal
        row["prediction_rms_difference_from_normal"] = float(np.sqrt(np.mean(difference * difference)))
        row["prediction_relative_rms_difference_from_normal"] = float(
            np.sqrt(np.mean(difference * difference))
            / max(np.sqrt(np.mean(normal * normal)), 1.0e-12)
        )
        row["use_hyper_value_context_override"] = bool(VARIANTS[variant]["use_hyper_value_context"])
        row["pairwise_module_token_source_override"] = VARIANTS[variant]["pairwise_module_token_source"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case.csv", rows)
    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "case_ids": [str(dataset_case) for dataset_case in args.case_id],
        "interpretation": "Frozen Run-1401 arithmetic diagnostic only; not a prediction of trained Run-1404 accuracy.",
        "variants": {
            variant: summarize([row for row in rows if row["checkpoint"] == variant])
            for variant in VARIANTS
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def interventions(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint.expanduser().resolve(), device)
    dataset = _checkpoint_dataset(checkpoint, "test")
    indices = _selected_indices(dataset, args.case_id, None)
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    try:
        for mode in PAIR_INTERVENTIONS:
            with pair_intervention(model, mode):
                for index in indices:
                    row = evaluate_case(
                        mode,
                        model,
                        checkpoint,
                        dataset,
                        index,
                        device,
                        query_batch_size=args.query_batch_size,
                        return_routing_maps=True,
                        benchmark=False,
                        stability_perturbations=False,
                        stability_repeats=0,
                    )
                    rows.append(row)
                    sample = dataset[index]
                    prediction = predict_case(
                        model,
                        sample,
                        device,
                        query_batch_size=args.query_batch_size,
                        local_port_condition_mode="predicted",
                        mixed_teacher_ratio=0.5,
                        return_routing_maps=False,
                    )
                    predictions[(mode, str(sample["case_id"]))] = np.asarray(
                        prediction["pred_field_grid"], dtype=np.float64
                    )
    finally:
        dataset.close()

    normal_rows = {str(row["case_id"]): row for row in rows if row["checkpoint"] == "normal"}
    for row in rows:
        mode = str(row["checkpoint"])
        case_id = str(row["case_id"])
        candidate = predictions[(mode, case_id)]
        normal = predictions[("normal", case_id)]
        difference = candidate - normal
        row["prediction_rms_difference_from_normal"] = float(np.sqrt(np.mean(difference * difference)))
        row["prediction_relative_rms_difference_from_normal"] = float(
            np.sqrt(np.mean(difference * difference)) / max(np.sqrt(np.mean(normal * normal)), 1.0e-12)
        )
        reference = normal_rows[case_id]
        for key, value in list(row.items()):
            if (key.endswith("_mse") or key.endswith("relative_l2")) and finite(value) is not None:
                baseline = finite(reference.get(key))
                if baseline is not None:
                    row[f"{key}_delta_from_normal"] = float(value) - baseline
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case.csv", rows)
    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "case_ids": [str(value) for value in args.case_id],
        "variants": {
            mode: summarize([row for row in rows if row["checkpoint"] == mode])
            for mode in PAIR_INTERVENTIONS
        },
        "interpretation": (
            "Frozen-checkpoint reliance diagnostics. Error deltas are intervention minus normal; "
            "large effects show reliance, not superiority."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def named_gradient_norms(model: Any) -> dict[str, dict[str, Any]]:
    groups = {
        "organizer_module_score": ("core.organizer.module_score.",),
        "organizer_env_score": ("core.organizer.env_score.",),
        "environment_encoder": ("core.env_encoder.",),
        "module_environment_context": (
            "core.organizer.me_query.",
            "core.organizer.me_key.",
            "core.organizer.me_context_proj.",
        ),
        "query_to_hyper_routing": (
            "core.decoder.query_to_hyper.",
            "core.decoder.hyper_key.",
        ),
        "pair_mlp": ("core.decoder.pairwise_kernel.pair_mlp.",),
        "hyper_value": ("core.decoder.hyper_value.",),
    }
    result: dict[str, dict[str, Any]] = {}
    named = dict(model.named_parameters())
    for group, prefixes in groups.items():
        parameters = [(name, value) for name, value in named.items() if name.startswith(prefixes)]
        gradients = [(name, value.grad) for name, value in parameters if value.grad is not None]
        square = sum(float((gradient.detach().double() ** 2).sum().cpu()) for _, gradient in gradients)
        result[group] = {
            "parameter_tensor_count": len(parameters),
            "gradient_tensor_count": len(gradients),
            "grad_norm": float(math.sqrt(square)),
            "finite": bool(all(torch.isfinite(gradient).all() for _, gradient in gradients)),
            "parameters": [name for name, _ in parameters],
        }
    return result


def gradient(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint.expanduser().resolve(), device)
    dataset = _checkpoint_dataset(checkpoint, "train")
    indices = _selected_indices(dataset, [args.case_id], None)
    sample = dataset[indices[0]]
    try:
        x_grid = np.asarray(sample["x_grid"])
        y_grid = np.asarray(sample["y_grid"])
        queries = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1).astype(np.float32)
        targets = _target_field(sample, dataset).reshape(-1, int(model.config.field_dim))
        count = min(max(args.query_count, 1), queries.shape[0])
        selected = np.linspace(0, queries.shape[0] - 1, num=count, dtype=np.int64)
        batch = make_batch(sample, queries[selected], device)
        target = torch.from_numpy(targets[selected].astype(np.float32)).unsqueeze(0).to(device)
        loss_cfg = dict(checkpoint.get("train_config", {}).get("loss", {}))
        training_cfg = dict(checkpoint.get("train_config", {}).get("training", {}))
        epoch = int(checkpoint.get("epoch", checkpoint.get("current_epoch", 0)))
        port_mode, mixed_ratio = effective_port_condition_settings(epoch, training_cfg)
        internal_weight, interface_weight = effective_local_loss_weights(loss_cfg, port_mode, mixed_ratio)
        predicted_weight = predicted_consistency_weight_for_epoch(epoch, loss_cfg)
        port_global_weight = effective_port_global_weight(loss_cfg, port_mode, mixed_ratio)
        candidate = {
            "decoder_mode": "enhanced_honf_pairwise_only",
            "use_hyper_value_context": False,
            "pairwise_aggregation_mode": "fused_query_module",
            "pairwise_module_token_source": "organizer_contextualized",
            "routing_execution": "dense",
            "query_module_retained_mass_floor": 1.0,
        }
        with arithmetic_variant(model, candidate):
            model.train()
            model.zero_grad(set_to_none=True)
            output = model(
                batch["structure"],
                batch["query_xy"],
                interface_condition=batch.get("interface_condition"),
                local_module_params=batch.get("local_module_params"),
                teacher_port_tokens=batch.get("teacher_port_tokens"),
                local_query_points=batch.get("module_internal_query_points"),
                local_port_condition_mode=port_mode,
                mixed_teacher_ratio=float(mixed_ratio),
                return_predicted_port_outputs=bool(predicted_weight > 0.0),
                return_port_global_consistency=bool(port_global_weight != 0.0),
                return_organizer_diagnostics=True,
            )
            field = channelthermal_field_mse(
                output["pred_field"], target, loss_cfg,
                field_names=model.config.channelthermal.field_names,
                point_weights=None,
            )
            zero = field.new_zeros(())
            loss_internal = internal_loss(output, batch) if internal_weight != 0.0 else zero
            loss_interface = interface_loss(output, batch, loss_cfg) if interface_weight != 0.0 else zero
            port_weight = float(loss_cfg.get("port_supervised_weight", loss_cfg.get("port_condition_weight", 0.0)))
            smooth_weight = float(loss_cfg.get("port_smoothness_weight", 0.0))
            loss_port = port_condition_loss(output, batch, loss_cfg) if port_weight != 0.0 else zero
            loss_smooth = port_cyclic_smoothness_loss(output, batch) if smooth_weight != 0.0 else zero
            loss_port_global = port_global_consistency_loss(output) if port_global_weight != 0.0 else zero
            if "predicted_port_internal_temperature" in output and "predicted_port_interface" in output:
                loss_predicted = internal_loss(
                    {"pred_internal_temperature": output["predicted_port_internal_temperature"], "pred_field": output["pred_field"]},
                    batch,
                ) + interface_loss(
                    {"pred_interface": output["predicted_port_interface"], "pred_field": output["pred_field"]},
                    batch,
                    loss_cfg,
                )
            else:
                loss_predicted = zero
            loss_org = organizer_regularization(output, loss_cfg)
            total = (
                float(loss_cfg.get("field_mse_weight", 1.0)) * field
                + float(internal_weight) * loss_internal
                + float(interface_weight) * loss_interface
                + port_weight * loss_port
                + smooth_weight * loss_smooth
                + float(port_global_weight) * loss_port_global
                + float(predicted_weight) * loss_predicted
                + loss_org
            )
            total.backward()
            result = {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "case_id": str(sample["case_id"]),
                "query_count": count,
                "candidate_settings": candidate,
                "loss_total": float(total.detach().cpu()),
                "loss_field": float(field.detach().cpu()),
                "hyper_value_context_norm": float(output["routing_aux"]["hyper_value_context_norm"].detach().cpu()),
                "uses_hyper_value_context": float(output["routing_aux"]["uses_hyper_value_context"].detach().cpu()),
                "pairwise_context_norm": float(output["routing_aux"]["pairwise_context_norm"].detach().cpu()),
                "gradient_groups": named_gradient_norms(model),
            }
    finally:
        dataset.close()
        model.zero_grad(set_to_none=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "frozen":
        frozen(args)
    elif args.command == "gradient":
        gradient(args)
    else:
        interventions(args)


if __name__ == "__main__":
    main()
