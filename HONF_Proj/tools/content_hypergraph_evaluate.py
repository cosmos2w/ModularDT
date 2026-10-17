#!/usr/bin/env python3
"""Evaluate retained content-hypergraph checkpoints on their native panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "tools", PROJECT / "Case_ThermalChannel/src", PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.content_hypergraph import load_content_hypergraph_checkpoint


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def state_value_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(state_value_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (isinstance(left, (tuple, list)) and isinstance(right, (tuple, list))
                and len(left) == len(right)
                and all(state_value_equal(a, b) for a, b in zip(left, right)))
    return left == right


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    return value


def observation_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Keep numeric observations, without serializing live prepared graphs."""
    if torch.is_tensor(value):
        return {prefix: value.detach().cpu().numpy()}
    if isinstance(value, np.ndarray):
        return {prefix: value}
    if is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in fields(value)}
    result = {}
    if isinstance(value, Mapping):
        for name, child in value.items():
            if name in {"prepared", "organization", "scene", "query", "trace", "auxiliary_state"} or name.endswith("_prepared"):
                continue
            key = f"{prefix}__{name}" if prefix else str(name)
            if torch.is_tensor(child) or isinstance(child, (np.ndarray, Mapping)):
                result.update(observation_arrays(child, key))
    return result


def load_evaluation_components(checkpoint: Path, device: str):
    """Reopen only the checkpoint's authorized native evaluation panel."""
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
    standalone = None
    if payload.get("workflow") == "native_budgeted_dispatch_joint_v2":
        from honf_runtime.native_budgeted_factory import build_native_budgeted_task
        from honf_runtime.native_budgeted_protocol import validate_native_budgeted_recipe

        identity = payload.get("experiment_identity")
        if not isinstance(identity, Mapping):
            raise TypeError("Round8 checkpoint has no saved experiment identity.")
        recipe = validate_native_budgeted_recipe(
            identity.get("recipe", {}), verify_source_file=True, verify_admission=True,
        )
        if canonical_sha256(recipe) != identity.get("recipe_sha256"):
            raise ValueError("Round8 evaluation recipe checksum is inconsistent.")
        task = build_native_budgeted_task(recipe, device=device, runtime_root=PROJECT.parent)
        model, provider = task["model"], task["provider"]
        expected = identity.get("provider_identity")
        observed = dict(provider.identity_payload())
        if not isinstance(expected, Mapping) or canonical_sha256(expected) != canonical_sha256(observed):
            raise ValueError("Round8 native evaluation data/scale/provider binding differs from the saved run.")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        standalone_model, standalone_adapter, _ = load_content_hypergraph_checkpoint(checkpoint, device=device)
        for name, value in standalone_model.state_dict().items():
            if name not in model.state_dict() or not state_value_equal(value, model.state_dict()[name]):
                raise ValueError(f"Standalone Round8 native load differs at {name}.")
        del standalone_model, standalone_adapter
        model.eval()
        if hasattr(provider.adapter, "eval"):
            provider.adapter.eval()
        return model, provider, payload
    if payload.get("workflow") == "moment_hyperedges_child_v1":
        from honf_runtime.moment_hyperedge_factory import load_moment_hyperedges_components

        components = load_moment_hyperedges_components(
            checkpoint,
            runtime_root=PROJECT.parent,
            device=torch.device(device),
        )
        model = components["model"]
        provider = components["native_provider"]
        recipe = components["recipe"]
        standalone = load_content_hypergraph_checkpoint(checkpoint, device=device)
        expected = payload["experiment_identity"]["provider_identity"]
        observed = dict(provider.identity_payload())
        if canonical_sha256(expected) != canonical_sha256(observed):
            raise ValueError("Round5 native evaluation data/scale/provider binding differs from the saved run.")
        standalone_model, standalone_adapter, _ = standalone
        for name, value in standalone_model.state_dict().items():
            if not torch.equal(value, model.state_dict()[name]):
                raise ValueError(f"Standalone Round5 native load differs at {name}.")
        del standalone_model, standalone_adapter
        return model, provider, payload
    elif payload.get("workflow") == "inherited_attention_joint_v1":
        from inherited_attention_train import build
        from inherited_attention_train import canonical_sha256 as iag_canonical_sha256

        from honf_runtime.inherited_attention_protocol import (
            configure_inherited_attention_numerics,
            validate_inherited_attention_recipe,
        )

        recipe = validate_inherited_attention_recipe(payload["experiment_identity"]["recipe"])
        configure_inherited_attention_numerics(recipe["numerical_policy"])
        standalone = load_content_hypergraph_checkpoint(checkpoint, device=device)
        if iag_canonical_sha256(recipe) != payload["experiment_identity"].get("recipe_sha256"):
            raise ValueError("IAG-C checkpoint recipe checksum is inconsistent.")
    else:
        from content_hypergraph_train import build, validate_recipe
        from content_hypergraph_train import canonical_sha256 as old_canonical_sha256

        recipe = validate_recipe(payload["experiment_identity"]["recipe"])
        if old_canonical_sha256(recipe) != payload["experiment_identity"].get("recipe_sha256"):
            raise ValueError("Content-hypergraph checkpoint recipe checksum is inconsistent.")
    model, provider = build(recipe, device=device)
    expected = payload["experiment_identity"]["provider_identity"]
    observed = dict(provider.identity_payload())
    if canonical_sha256(expected) != canonical_sha256(observed):
        raise ValueError("Native evaluation data/scale/provider binding differs from the saved run.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    # Exercise the self-contained path separately; it must reconstruct exactly
    # the same tensor inventory without relying on provider/dataset state.
    if standalone is None:
        standalone = load_content_hypergraph_checkpoint(checkpoint, device=device)
    standalone_model, adapter, _ = standalone
    for name, value in standalone_model.state_dict().items():
        if not torch.equal(value, model.state_dict()[name]):
            raise ValueError(f"Standalone native load differs at {name}.")
    del standalone_model, adapter
    return model, provider, payload


def fixed_monitor_batches(provider, recipe: dict[str, Any], epoch: int, split: str):
    """Replay declared monitoring queries at one fixed checkpoint, without updates."""
    if split == "validation":
        return provider.validation_batches(), None
    if recipe.get("family") in ("inherited_attention_v1", "moment_hyperedges_round5"):
        from inherited_attention_train import engine_for
        if recipe.get("family") == "moment_hyperedges_round5":
            from honf_runtime.moment_hyperedge_protocol import native_recipe_for_moment_hyperedges

            recipe = native_recipe_for_moment_hyperedges(recipe)
        engine_recipe = recipe
    else:
        from content_hypergraph_train import engine_for
        engine_recipe = recipe.get("base_recipe", recipe)
    engine = engine_for(engine_recipe, str(provider.device))
    ordered, dataset_id = engine.epoch_case_order(provider, epoch)

    def batches():
        for update_index, selection in enumerate(engine.epoch_update_slices(len(ordered))):
            yield from engine.prepare_update_microbatches(
                provider, arm=engine_recipe["architecture"], phase="joint", epoch=epoch,
                update_index=update_index, update_cases=ordered[selection], sampling_dataset_id=dataset_id)

    return batches(), list(ordered)


def label_monitor_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Replace native-endpoint labels with counts measured from monitor rows."""
    result = dict(metrics)
    result["evaluation_panel"] = "fixed_checkpoint_monitor"
    if "query_count_per_row" in result:
        counts = {
            sum(int(role["query_count"]) for role in row["roles"].values())
            for row in result["rows"]
        }
        if len(counts) != 1:
            raise ValueError("Fixed Wind monitoring must have one observed query count per row.")
        result["query_count_per_row"] = counts.pop()
        result["monitor_rows"] = result.pop("endpoint_rows")
        result.pop("endpoint_selection", None)
        result["monitor_selection"] = "complete declared cohort; saved epoch sampling and original packing"
    return result


def _is_round8_wind_payload(payload: Mapping[str, Any], recipe: Mapping[str, Any]) -> bool:
    """Select the narrow Round8 Wind evaluator path from the sealed checkpoint identity."""
    workflow = "native_budgeted_dispatch_joint_v2"
    family = "native_budgeted_dispatch_v2"
    if payload.get("workflow") != workflow:
        return False
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Round8 evaluation payload has no sealed experiment identity.")
    if (identity.get("workflow") != workflow or identity.get("family") != family
            or identity.get("model_family") != family
            or recipe.get("workflow") != workflow or recipe.get("family") != family
            or recipe.get("model_family") != family):
        raise ValueError("Round8 Wind evaluator dispatch requires the exact saved family and workflow.")
    return recipe.get("dataset") == "wind" and recipe.get("architecture") == "A2-D"


def _array_from_value(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _round8_wind_role_counts(provider: Any, batch: Any, split: str) -> dict[str, int]:
    """Validate the exact source TRAIN or exposed DEV role order and quotas."""
    if split == "train":
        raw_counts = (getattr(provider, "VOLUME_QUERIES", None), getattr(provider, "HUB_BAND_QUERIES", None))
        expected_counts = (6144, 2048)
        expected_weights = (0.75, 0.25)
    elif split == "validation":
        raw_counts = (
            getattr(provider, "VALIDATION_VOLUME_QUERIES", None),
            getattr(provider, "VALIDATION_HUB_BAND_QUERIES", None),
        )
        expected_counts = (8192, 2048)
        expected_weights = (0.8, 0.2)
    else:
        raise ValueError(f"Unsupported Round8 Wind split: {split!r}.")
    if tuple(raw_counts) != expected_counts:
        raise ValueError(f"Round8 Wind {split} quotas changed from the sealed source protocol.")
    weight_names = (
        ("VOLUME_WEIGHT", "HUB_BAND_WEIGHT") if split == "train"
        else ("VALIDATION_VOLUME_WEIGHT", "VALIDATION_HUB_BAND_WEIGHT")
    )
    weights = tuple(getattr(provider, name, None) for name in weight_names)
    if weights != expected_weights:
        raise ValueError(f"Round8 Wind {split} objective weights changed from the sealed source protocol.")

    role_ids = _array_from_value(batch.receivers.role_ids)
    keys = tuple(int(value) for value in batch.case_keys)
    targets = batch.targets
    if (role_ids.ndim != 2 or role_ids.shape != (len(keys), sum(expected_counts))
            or tuple(int(value) for value in targets.row_indices) != keys
            or len(targets.case_metadata) != len(keys)):
        raise ValueError(f"Round8 Wind {split} receiver rows, targets and case keys do not align.")
    volume_role = getattr(provider, "VOLUME_ROLE_ID", None)
    band_role = getattr(provider, "HUB_BAND_ROLE_ID", None)
    if (type(volume_role) is not int or type(band_role) is not int or volume_role == band_role
            or (volume_role, band_role) != (0, 1)):
        raise ValueError("Round8 Wind source role IDs must retain volume=0 and hub-band=1.")
    expected_row = np.concatenate((
        np.full(expected_counts[0], volume_role, dtype=role_ids.dtype),
        np.full(expected_counts[1], band_role, dtype=role_ids.dtype),
    ))
    if not np.array_equal(role_ids, np.broadcast_to(expected_row, role_ids.shape)):
        raise ValueError(f"Round8 Wind {split} rows do not retain exact native role membership/order.")
    for key, metadata in zip(keys, targets.case_metadata):
        if (int(metadata.get("row_index", -1)) != key
                or int(metadata.get("volume_query_count", -1)) != expected_counts[0]
                or int(metadata.get("hub_band_query_count", -1)) != expected_counts[1]):
            raise ValueError(f"Round8 Wind {split} row metadata differs from observed receiver roles.")
    prediction_role_ids = getattr(batch, "prediction_role_ids", None)
    if prediction_role_ids is not None and not np.array_equal(_array_from_value(prediction_role_ids), role_ids):
        raise ValueError("Round8 Wind prediction roles differ from the validated receiver roles.")
    return {"volume": expected_counts[0], "hub_band": expected_counts[1]}


def _round8_wind_train_metrics(
    predictions: Any,
    targets: Any,
    provider: Any,
    batch: Any,
) -> Mapping[str, Any]:
    """Score the source TRAIN two-region objective without DEV selector labels."""
    role_counts = _round8_wind_role_counts(provider, batch, "train")
    predicted = _array_from_value(predictions.main_mps).astype(np.float64, copy=False)
    actual = np.asarray(targets.velocity_mps, dtype=np.float64)
    keys = tuple(int(value) for value in batch.case_keys)
    expected_shape = (len(keys), sum(role_counts.values()), 3)
    if predicted.shape != expected_shape or actual.shape != expected_shape:
        raise ValueError("Round8 Wind source TRAIN metrics require exact [B,8192,3] predictions and targets.")
    prediction_role_ids = getattr(predictions, "role_ids", None)
    if prediction_role_ids is not None and not np.array_equal(
            _array_from_value(prediction_role_ids), _array_from_value(batch.receivers.role_ids)):
        raise ValueError("Round8 Wind predictions lost the exact TRAIN role identities.")
    normalizer = getattr(getattr(provider, "adapter", None), "normalizer", None)
    if normalizer is None:
        raise ValueError("Round8 Wind TRAIN metrics require the saved source normalization.")
    scale = np.asarray(normalizer.safe_std, dtype=np.float64) * float(normalizer.u_ref_mps)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Round8 Wind source TRAIN physical normalization scale is invalid.")
    volume_queries = role_counts["volume"]
    volume_weight = float(provider.VOLUME_WEIGHT)
    band_weight = float(provider.HUB_BAND_WEIGHT)
    rows: list[dict[str, Any]] = []
    for index, metadata in enumerate(targets.case_metadata):
        squared = np.square((predicted[index] - actual[index]) / scale.reshape(1, 3))
        volume_mse = float(np.mean(squared[:volume_queries], dtype=np.float64))
        hub_band_mse = float(np.mean(squared[volume_queries:], dtype=np.float64))
        volume_delta = predicted[index, :volume_queries] - actual[index, :volume_queries]
        hub_band_delta = predicted[index, volume_queries:] - actual[index, volume_queries:]
        rows.append({
            **dict(metadata),
            "train_volume_mse": volume_mse,
            "train_hub_band_mse": hub_band_mse,
            "train_weighted_mse": volume_weight * volume_mse + band_weight * hub_band_mse,
            "volume_vector_rmse_mps": float(np.sqrt(np.mean(np.sum(np.square(volume_delta), axis=-1), dtype=np.float64))),
            "hub_band_vector_rmse_mps": float(np.sqrt(np.mean(np.sum(np.square(hub_band_delta), axis=-1), dtype=np.float64))),
        })
    return {"rows": rows}


def _reduce_round8_wind_train_metrics(
    records: list[Mapping[str, Any]],
    provider: Any,
    case_keys: Sequence[Any],
) -> dict[str, Any]:
    rows = [dict(row) for record in records for row in record["rows"]]
    expected = [int(value) for value in case_keys]
    observed = [int(row["row_index"]) for row in rows]
    if not expected or observed != expected or len(set(observed)) != len(observed):
        raise ValueError("Round8 Wind TRAIN metrics must contain the exact ordered selected rows once.")
    return {
        "split": "train",
        "scope": "selected source TRAIN objective; not a validation selector",
        "train_row_count": len(rows),
        "train_endpoint_rows": expected,
        "query_count_per_row": 8192,
        "query_count_per_row_source": "observed receiver role identities and sealed source quotas",
        "role_query_counts": {"volume": 6144, "hub_band": 2048},
        "train_objective_weights": {
            "volume": float(provider.VOLUME_WEIGHT),
            "hub_band": float(provider.HUB_BAND_WEIGHT),
        },
        "train_objective": "equal-case mean of 0.75*TRAIN-normalized volume MSE + 0.25*TRAIN-normalized hub-band MSE; Ux/Uy/Uz equally weighted",
        "train_volume_mse": float(np.mean([row["train_volume_mse"] for row in rows], dtype=np.float64)),
        "train_hub_band_mse": float(np.mean([row["train_hub_band_mse"] for row in rows], dtype=np.float64)),
        "train_weighted_mse": float(np.mean([row["train_weighted_mse"] for row in rows], dtype=np.float64)),
        "rows": rows,
    }


def _round8_wind_dev_metadata(
    metrics: Mapping[str, Any],
    role_counts: Mapping[str, int],
) -> dict[str, Any]:
    result = dict(metrics)
    observed_query_count = int(role_counts["volume"]) + int(role_counts["hub_band"])
    if dict(role_counts) != {"volume": 8192, "hub_band": 2048} or observed_query_count != 10240:
        raise ValueError("Round8 Wind exposed DEV must use observed Q10240=8192 volume +2048 hub-band rows.")
    inherited_query_count = result.get("query_count_per_row")
    result["query_count_per_row"] = observed_query_count
    result["query_count_per_row_source"] = "observed receiver role identities and sealed DEV quotas"
    result["role_query_counts"] = dict(role_counts)
    result["dev_objective_weights"] = {
        "volume": 0.8,
        "hub_band": 0.2,
    }
    if inherited_query_count != observed_query_count:
        result["query_count_reducer_correction"] = {
            "inherited_reducer_value": inherited_query_count,
            "observed_panel_value": observed_query_count,
            "reason": "Round8 public evaluator replaces the source reducer's legacy Q8192 default with observed formal DEV Q10240.",
        }
    return result


def _label_round8_wind_monitor(
    metrics: Mapping[str, Any],
    *,
    split: str,
    role_counts: Mapping[str, int],
) -> dict[str, Any]:
    result = dict(metrics)
    result["evaluation_panel"] = "fixed_checkpoint_monitor"
    result["query_count_per_row"] = int(role_counts["volume"]) + int(role_counts["hub_band"])
    result["query_count_per_row_source"] = "observed receiver role identities in every complete monitor batch"
    result["role_query_counts"] = dict(role_counts)
    if split == "train":
        result["monitor_data_split"] = "train"
        result["monitor_rows"] = result.pop("train_endpoint_rows")
        result["monitor_selection"] = "complete fixed source TRAIN cohort; saved epoch sampling and original packing"
        result["monitor_score_semantics"] = "source TRAIN metric; not a DEV selector"
    elif split == "validation":
        result["monitor_data_split"] = "validation"
        result["monitor_rows"] = result.pop("endpoint_rows")
        result.pop("endpoint_selection", None)
        result["monitor_selection"] = "complete exposed DEV cohort; saved checkpoint and original packing"
    else:
        raise ValueError(f"Unsupported Round8 Wind monitor split: {split!r}.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("validation", "train"), default="validation")
    parser.add_argument("--panel", choices=("native", "monitor", "wind_native_roles"), default="native")
    parser.add_argument("--case-keys", help="Comma-separated physical case IDs or native Wind row IDs")
    parser.add_argument("--export-case", action="append", default=[])
    parser.add_argument("--microbatch-cases", type=int, help="Native evaluation packing; exports default to one case")
    args = parser.parse_args(argv)
    packing = args.microbatch_cases if args.microbatch_cases is not None else (1 if args.export_case else None)
    if packing is not None and packing < 1:
        raise ValueError("Native evaluation packing must be positive.")
    if args.export_case and packing != 1:
        raise ValueError("Detailed exports require packing one so only selected cases are materialized.")
    if args.panel == "monitor" and (args.case_keys or args.export_case or args.microbatch_cases):
        raise ValueError("Fixed monitoring replays the complete declared cohort and original packing; use native for details.")
    if args.output.exists():
        raise FileExistsError("Evaluation evidence is immutable; choose a fresh output directory.")
    started = time.perf_counter()
    model, provider, payload = load_evaluation_components(args.checkpoint, args.device)
    recipe = payload["experiment_identity"]["recipe"]
    round8_wind = _is_round8_wind_payload(payload, recipe)
    if args.panel == "wind_native_roles":
        if (recipe.get("family") != "inherited_attention_v1" or recipe.get("dataset") != "wind"
                or not callable(getattr(provider, "native_role_evaluation_batches", None))):
            raise ValueError("wind_native_roles is reserved for the opt-in Wind IAG-C provider.")
        if packing not in (None, 1):
            raise ValueError("The exact Wind five-role query-hash panel is evaluated one case at a time.")
    selected = None if not args.case_keys else args.case_keys.split(",")
    if selected is not None and recipe["dataset"] == "wind":
        selected = [int(x) for x in selected]
    args.output.mkdir(parents=True)
    records, work, exports, organizations = [], [], [], []
    observed_wind_role_counts: list[dict[str, int]] = []
    if args.panel == "monitor":
        batches, selected = fixed_monitor_batches(provider, recipe, int(payload["epoch"]), args.split)
    elif args.panel == "wind_native_roles":
        batches = provider.native_role_evaluation_batches(split=args.split, case_keys=selected)
    else:
        batches = provider.native_evaluation_batches(split=args.split, case_keys=selected, microbatch_size=packing)
    with torch.no_grad():
        for index, batch in enumerate(batches):
            if round8_wind:
                observed_wind_role_counts.append(_round8_wind_role_counts(provider, batch, args.split))
            scene = provider.make_scene(batch.scene_inputs)
            prediction, auxiliary = provider.predict_native(
                model, scene, batch.receivers, recipe["architecture"], "joint",
                epoch=int(payload["epoch"]), temperature=1.0,
            )
            if args.panel == "wind_native_roles":
                records.append(provider.native_role_evaluation_metrics(prediction, batch.targets, batch.auxiliary))
            elif round8_wind and args.split == "train":
                records.append(_round8_wind_train_metrics(prediction, batch.targets, provider, batch))
            else:
                records.append(provider.validation_metrics(prediction, batch.targets, auxiliary))
            work.append(dict(provider.work_counts(batch, prediction, auxiliary)))
            keys = tuple(str(x) for x in batch.case_keys)
            if (recipe["schema"] == "content_hypergraph_recipe_v2"
                    and recipe["architecture"] == "A2-H6" and args.panel == "native"):
                from content_hypergraph_inspect_v2 import organization_summary
                if recipe["dataset"] == "thermal":
                    organization = prediction.native_prepared.organization
                    query = prediction.native_prepared.query_batch
                    core = model.core
                else:
                    organization = auxiliary["organization"]
                    query = provider.adapter.make_query_batch(scene, batch.receivers)
                    core = model
                organizations.append({"case_keys": keys, **organization_summary(core, organization, query)})
            if any(key in args.export_case for key in keys):
                arrays = {}
                for prefix, value in (("scene", scene), ("prediction", prediction), ("target", batch.targets),
                                      ("receiver", batch.receivers)):
                    arrays.update(observation_arrays(value, prefix))
                path = args.output / f"observations_{index:03d}_{'-'.join(keys)}.npz"
                np.savez_compressed(path, **arrays)
                exports.append({"path": str(path), "case_keys": list(keys),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "array_shapes": {k: list(v.shape) for k, v in arrays.items()}})
    if args.panel == "wind_native_roles":
        metrics = provider.reduce_native_role_evaluation_metrics(records, split=args.split, case_keys=selected)
    elif round8_wind:
        if not observed_wind_role_counts:
            raise ValueError("Round8 Wind evaluation produced no batches from which to verify its native roles.")
        count_set = {(row["volume"], row["hub_band"]) for row in observed_wind_role_counts}
        if len(count_set) != 1:
            raise ValueError(f"Round8 Wind native role quotas changed across batches: {sorted(count_set)!r}.")
        role_counts = {"volume": next(iter(count_set))[0], "hub_band": next(iter(count_set))[1]}
        if args.split == "train":
            expected_rows = (
                list(provider.native_evaluation_rows("train", selected))
                if selected is not None
                else list(provider.native_evaluation_rows("train", None))
            )
            metrics = _reduce_round8_wind_train_metrics(records, provider, expected_rows)
        else:
            reducer = getattr(provider, "reduce_native_evaluation_metrics", None)
            if not callable(reducer):
                raise TypeError("Round8 Wind provider has no strict native endpoint reducer.")
            metrics = _round8_wind_dev_metadata(
                reducer(records, split=args.split, case_keys=selected), role_counts,
            )
    else:
        reducer = getattr(provider, "reduce_native_evaluation_metrics", None)
        metrics = (reducer(records, split=args.split, case_keys=selected) if callable(reducer)
                   else provider.reduce_native_metrics(records))
    if args.panel == "monitor":
        if round8_wind:
            metrics = _label_round8_wind_monitor(
                metrics,
                split=args.split,
                role_counts=role_counts,
            )
        else:
            metrics = label_monitor_metrics(metrics)
    result = {
        "schema": "content_hypergraph_native_evaluation_v1", "status": "completed",
        "dataset": recipe["dataset"], "architecture": recipe["architecture"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "completed_epoch": int(payload["epoch"]), "split": args.split,
        "panel": args.panel,
        "receiver_scope": (
            "fixed five-role native Q8192 diagnostic board, independent of two-region objective"
            if args.panel == "wind_native_roles" else
            "source TRAIN Q8192 (6144 volume + 2048 hub-band), equal-case 0.75/0.25 objective; not a DEV selector"
            if round8_wind and args.split == "train" else
            "exposed DEV Q10240 (8192 volume + 2048 hub-band), fixed 0.8/0.2 validation objective"
            if round8_wind and args.split == "validation" else
            "full native endpoint support" if args.panel == "native" else
            "fixed checkpoint; declared primary monitoring queries and native auxiliary samples"
        ),
        "evaluation_microbatch_cases": packing,
        "provider_binding_sha256": canonical_sha256(payload["experiment_identity"]["provider_identity"]),
        "metrics": metrics, "records": records, "work_counts": work, "exports": exports,
        "organization_summaries": organizations,
        "elapsed_seconds": time.perf_counter() - started, "optimizer_updates": 0,
        "new_solver_attempts": 0, "WindTEST": "locked",
    }
    destination = args.output / "result.json"
    destination.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_safe({k: v for k, v in result.items() if k not in {"records", "exports", "organization_summaries"}}),
                     sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
