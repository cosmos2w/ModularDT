"""CHANNELTHERMAL-SPECIFIC Prompt-3 NewHONF evaluator.

Inputs are a NewHONF checkpoint selector or path, the existing packed
ChannelThermal HDF5 dataset, and a case selection. Outputs are quicklook PNGs,
organizer visualizations, compressed prediction arrays, and `summary.json`.
This executable is specific to ChannelThermal legacy evaluation behavior,
including internal/interface plots from the local-surrogate coupling path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-newhonf")

import numpy as np
import torch

import _bootstrap_imports  # noqa: F401
from _data.channelthermal_datasets import CHANNEL_ORDER, GlobalChannelThermalDataset
from _helpers.evaluation_plots import (
    error_metrics,
    masked_error_metrics,
    module_and_fluid_masks,
    module_radius_from_sample,
    plot_field_quicklook,
    plot_interface,
    plot_internal,
)
from _helpers.hypergraph_plan import extract_hypergraph_plan
from _helpers.model_utils import current_timestamp, load_trusted_checkpoint, recursive_to_device, resolve_demo_path, select_device, strip_module_prefix, write_json
from _helpers.organizer_viz_channelthermal import (
    render_channelthermal_organization_overview,
    render_channelthermal_organization_schematic_presentation,
    render_channelthermal_organization_summary_matrices,
)
from _helpers.routing_viz_channelthermal import save_routing_diagnostics
from _models_channelthermal.channelthermal_config import ChannelThermalHONFConfig
from _models_channelthermal.channelthermal_full_model import ChannelThermalHONFModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Prompt-3 standalone ChannelThermal HONF checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="best", help="best, best_by_field_mse, best_by_temperature_mse, latest, best_predicted, or a .pt path.")
    parser.add_argument("--Run_ID", dest="run_id", type=str, default=None)
    parser.add_argument("--saved-root", type=str, default="./Saved_Model_NewHONF")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--query-batch-size", type=int, default=32768)
    parser.add_argument("--local-port-condition-mode", choices=["teacher", "predicted", "mixed", "both"], default="predicted")
    parser.add_argument("--mixed-teacher-ratio", type=float, default=0.5)
    parser.add_argument("--temperature-display-mode", choices=["fluid_only", "composite_internal"], default=None)
    parser.add_argument("--organization-view", choices=["all", "physical", "matrices", "schematic", "none"], default="all")
    parser.add_argument("--organization-style", choices=["presentation", "debug", "both"], default="presentation")
    parser.add_argument("--organization-link-threshold", type=float, default=0.25)
    parser.add_argument("--return-routing-maps", action="store_true", help="Return dense query routing maps for evaluation diagnostics.")
    parser.add_argument("--routing-view", choices=["none", "summary", "all"], default="summary")
    parser.add_argument("--export-hypergraph-plan", action="store_true", help="Export compact static organizer plan for inverse-design seeding.")
    return parser.parse_args()


def checkpoint_file_name(selector: str) -> str:
    cleaned = str(selector).strip().lower()
    if cleaned in {"best_predicted", "predicted", "autonomous"}:
        return "best_predicted_model.pt"
    if cleaned in {"best", "best_total"}:
        return "best_model.pt"
    if cleaned in {"best_by_field_mse", "field"}:
        return "best_by_field_mse_model.pt"
    if cleaned in {"best_by_temperature_mse", "temperature"}:
        return "best_by_temperature_mse_model.pt"
    if cleaned in {"latest", "lastest"}:
        return "latest_model.pt"
    raise ValueError(f"Unknown checkpoint selector: {selector}")


def normalize_run_id(value: str) -> str:
    raw = str(value).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be numeric, e.g. 0002; got {raw!r}.")
    return f"{int(raw):04d}"


def latest_run_dir(saved_root: Path, run_id: str) -> Path:
    normalized = normalize_run_id(run_id)
    patterns = (f"Run_{normalized}_*", f"{normalized}_*", f"{normalized}*")
    matches = sorted({path for pattern in patterns for path in saved_root.glob(pattern) if path.is_dir()})
    if not matches:
        raise FileNotFoundError(f"No saved NewHONF runs found under {saved_root} with Run_ID={normalized!r}.")
    def sort_key(path: Path) -> tuple[int, str, float, str]:
        match = re.search(rf"Run_{normalized}_(\d{{8}}_\d{{6}})", path.name)
        # Prefer the Prompt-3 timestamped run naming scheme over older
        # compatibility names, then choose the newest timestamp/mtime.
        return (1 if match else 0, match.group(1) if match else "", path.stat().st_mtime, path.name)

    return sorted(matches, key=sort_key)[-1]


def resolve_checkpoint_arg(args: argparse.Namespace) -> Path:
    selector = str(args.checkpoint)
    if args.run_id:
        run_dir = latest_run_dir(resolve_demo_path(args.saved_root), args.run_id)
        candidate = (run_dir / checkpoint_file_name(selector)).resolve()
        if not candidate.exists() and selector.lower() in {"best_predicted", "predicted", "autonomous"}:
            fallback = (run_dir / "best_model.pt").resolve()
            print(f"[warning] {candidate.name} not found; falling back to {fallback.name}.")
            return fallback
        return candidate
    candidate = resolve_demo_path(selector)
    if candidate.suffix == ".pt" or candidate.exists():
        return candidate
    raise ValueError("--Run_ID is required when --checkpoint is a named selector.")


def numpy_to_batched_tensor(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).unsqueeze(0)
    if isinstance(value, dict):
        return {key: numpy_to_batched_tensor(item) for key, item in value.items()}
    return value


def make_batch(sample: Dict[str, Any], query_xy: np.ndarray, device: torch.device) -> Dict[str, Any]:
    payload = {key: value for key, value in sample.items() if key not in {"x_grid", "y_grid", "steady_field", "rms_field", "case_id"}}
    payload["query_xy"] = query_xy.astype(np.float32)
    return recursive_to_device(numpy_to_batched_tensor(payload), device)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ChannelThermalHONFModel, Dict[str, Any]]:
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    model_config = ChannelThermalHONFConfig.from_dict(checkpoint.get("model_config", {}))
    model = ChannelThermalHONFModel(model_config).to(device)
    global_norm_cfg = checkpoint.get("global_normalization_config", checkpoint.get("train_config", {}).get("dataset", {}))
    model.set_global_target_normalization(checkpoint.get("global_normalization_stats", {}), normalize_targets=bool(global_norm_cfg.get("normalize_targets", False)))
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=False)
    model.eval()
    return model, checkpoint


def select_sample(dataset: GlobalChannelThermalDataset, case_id: Optional[str], case_index: int) -> Dict[str, Any]:
    if len(dataset) == 0:
        raise RuntimeError("No global channel thermal cases are available for evaluation.")
    if case_id is not None:
        for idx, candidate in enumerate(dataset.selected_case_ids):
            if str(candidate) == str(case_id):
                return dataset[idx]
        raise KeyError(f"case_id={case_id!r} not found in split {dataset.split!r}.")
    return dataset[min(max(int(case_index), 0), len(dataset) - 1)]


def predict_case(
    model: ChannelThermalHONFModel,
    sample: Dict[str, Any],
    device: torch.device,
    *,
    query_batch_size: int,
    local_port_condition_mode: str,
    mixed_teacher_ratio: float,
    return_routing_maps: bool = False,
) -> Dict[str, Any]:
    x_grid = sample["x_grid"]
    y_grid = sample["y_grid"]
    query_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1).astype(np.float32)
    pred_chunks = []
    routing_chunks: Dict[str, list[np.ndarray]] = {}
    first_outputs = None
    with torch.no_grad():
        for start in range(0, query_xy.shape[0], int(query_batch_size)):
            chunk = query_xy[start : start + int(query_batch_size)]
            batch = make_batch(sample, chunk, device)
            outputs = model(
                batch["structure"],
                batch["query_xy"],
                interface_condition=batch.get("interface_condition"),
                local_module_params=batch.get("local_module_params"),
                teacher_port_tokens=batch.get("teacher_port_tokens"),
                local_query_points=batch.get("module_internal_query_points"),
                local_port_condition_mode=local_port_condition_mode,
                mixed_teacher_ratio=float(mixed_teacher_ratio),
                return_routing_maps=bool(return_routing_maps),
            )
            pred_chunks.append(outputs["pred_field"].detach().cpu().numpy()[0])
            if return_routing_maps:
                routing_aux = outputs.get("routing_aux", {})
                key_map = {
                    "query_hyper_attention": "query_hyper_attention",
                    "pairwise_edge_contribution": "pairwise_edge_contribution",
                    "c_H_norm": "c_H_norm",
                    "c_pair_norm": "c_pair_norm",
                    "dominant_hyperedge": "dominant_hyperedge",
                    "hyper_attention_entropy_map": "hyper_attention_entropy",
                }
                for source_key, target_key in key_map.items():
                    value = routing_aux.get(source_key)
                    if torch.is_tensor(value):
                        routing_chunks.setdefault(target_key, []).append(value.detach().cpu().numpy()[0])
            if first_outputs is None:
                first_outputs = outputs
    if first_outputs is None:
        raise RuntimeError("No prediction chunks were produced.")
    pred_field = np.concatenate(pred_chunks, axis=0).reshape(*x_grid.shape, model.config.field_dim)
    result = {
        "pred_field_grid": pred_field.astype(np.float32),
        "pred_internal_temperature": first_outputs["pred_internal_temperature"].detach().cpu().numpy()[0],
        "pred_interface": first_outputs["pred_interface"].detach().cpu().numpy()[0],
        "pred_port_condition": first_outputs["pred_port_condition"].detach().cpu().numpy()[0],
        "interface_flux_mode": first_outputs.get("interface_source", "unknown"),
        "organizer_aux": {
            key: value.detach().cpu().numpy()[0] if torch.is_tensor(value) and value.ndim > 0 else value
            for key, value in first_outputs["organizer_aux"].items()
        },
    }
    if return_routing_maps:
        result["routing_maps"] = {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in routing_chunks.items()
            if chunks
        }
    return result


def denormalize_predictions(predictions: Dict[str, Any], dataset: GlobalChannelThermalDataset, normalize_targets: bool) -> Dict[str, Any]:
    if not normalize_targets:
        return predictions
    out = dict(predictions)
    out["pred_field_grid"] = dataset.normalizer.denormalize_fields(out["pred_field_grid"])
    if np.asarray(out["pred_internal_temperature"]).size:
        out["pred_internal_temperature"] = dataset.normalizer.denormalize_internal_temperature(out["pred_internal_temperature"])
    if np.asarray(out["pred_interface"]).size:
        out["pred_interface"] = dataset.normalizer.denormalize_interface_targets(out["pred_interface"])
    return out


def safe_path_name(value: object) -> str:
    raw = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw) or "case"


def evaluation_output_dir(base_dir_arg: str | None, checkpoint_path: Path, case_id: object) -> Path:
    base_dir = Path(base_dir_arg) if base_dir_arg else checkpoint_path.parent / "eval_global"
    return resolve_demo_path(base_dir) / f"{safe_path_name(case_id)}_{current_timestamp()}"


def extract_organization_arrays(sample: Dict[str, Any], aux: Dict[str, Any]) -> Dict[str, np.ndarray]:
    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"] > 0.5)
    env_coords = np.asarray(aux.get("env_coords", np.zeros((0, 2))), dtype=np.float32)
    A_eh = np.asarray(aux.get("A_eh", np.zeros((env_coords.shape[0], 1))), dtype=np.float32)
    A_mh = np.asarray(aux.get("A_mh", np.zeros((centers.shape[0], A_eh.shape[-1]))), dtype=np.float32)
    strength = np.asarray(aux.get("hyper_strength", np.ones((A_eh.shape[-1],), dtype=np.float32)), dtype=np.float32)
    return {
        "centers": centers,
        "present": present,
        "heat": np.asarray(sample["structure"].get("heat_powers", np.zeros((centers.shape[0],))), dtype=np.float32),
        "env_coords": env_coords,
        "A_eh": A_eh,
        "A_mh": A_mh,
        "strength": strength,
        "module_mass": np.asarray(aux.get("hyper_module_mass", np.zeros_like(strength)), dtype=np.float32),
        "env_mass": np.asarray(aux.get("hyper_env_mass", np.zeros_like(strength)), dtype=np.float32),
        "src": np.asarray(aux.get("hyper_source_coords", np.zeros((strength.shape[0], 2))), dtype=np.float32),
        "dst": np.asarray(aux.get("hyper_thermal_region_coords", aux.get("hyper_region_coords", np.zeros((strength.shape[0], 2)))), dtype=np.float32),
    }


def copy_figure_alias(source: Path, alias: Path) -> None:
    if source.resolve() != alias.resolve():
        shutil.copyfile(source, alias)


def summarize(raw_sample: Dict[str, Any], predictions: Dict[str, Any], checkpoint_path: Path, output_dir: Path, channel_order: list[str]) -> Dict[str, Any]:
    pred = predictions["pred_field_grid"]
    gt = raw_sample["steady_field"][..., : pred.shape[-1]]
    _, fluid_mask = module_and_fluid_masks(raw_sample, pred)
    suffix = str(predictions.get("suffix", "predicted"))
    npz_path = output_dir / f"evaluation_outputs_{suffix}.npz"
    np.savez_compressed(
        npz_path,
        pred_field_grid=pred.astype(np.float32),
        gt_field_grid=gt.astype(np.float32),
        pred_internal_temperature=predictions["pred_internal_temperature"].astype(np.float32),
        pred_interface=predictions["pred_interface"].astype(np.float32),
        pred_port_condition=predictions["pred_port_condition"].astype(np.float32),
    )
    channel_metrics = {
        str(name): masked_error_metrics(pred[..., idx], gt[..., idx], fluid_mask)
        for idx, name in enumerate(channel_order[: pred.shape[-1]])
    }
    field_metrics = error_metrics(pred, gt)
    field_metrics_fluid = masked_error_metrics(pred, gt, fluid_mask)
    temperature_metrics_fluid = masked_error_metrics(pred[..., 4], gt[..., 4], fluid_mask) if pred.shape[-1] >= 5 else None
    metrics_csv_path = output_dir / f"metrics_{suffix}.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "suffix",
                "field_mse",
                "field_rmse",
                "field_mae",
                "field_relative_l2",
                "fluid_mse",
                "fluid_rmse",
                "fluid_mae",
                "fluid_relative_l2",
                "temperature_fluid_mse",
                "temperature_fluid_rmse",
                "temperature_fluid_mae",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": str(raw_sample["case_id"]),
                "suffix": suffix,
                "field_mse": field_metrics.get("mse"),
                "field_rmse": field_metrics.get("rmse"),
                "field_mae": field_metrics.get("mae"),
                "field_relative_l2": field_metrics.get("relative_l2"),
                "fluid_mse": field_metrics_fluid.get("mse"),
                "fluid_rmse": field_metrics_fluid.get("rmse"),
                "fluid_mae": field_metrics_fluid.get("mae"),
                "fluid_relative_l2": field_metrics_fluid.get("relative_l2"),
                "temperature_fluid_mse": None if temperature_metrics_fluid is None else temperature_metrics_fluid.get("mse"),
                "temperature_fluid_rmse": None if temperature_metrics_fluid is None else temperature_metrics_fluid.get("rmse"),
                "temperature_fluid_mae": None if temperature_metrics_fluid is None else temperature_metrics_fluid.get("mae"),
            }
        )
    return {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "phase": "prompt3_physical_coupling",
        "field_metrics": field_metrics,
        "field_metrics_fluid": field_metrics_fluid,
        "temperature_metrics_fluid": temperature_metrics_fluid,
        "field_channel_metrics_fluid": channel_metrics,
        "internal_interface_note": "Skipped only when internal/interface tensors are empty.",
        "interface_flux_mode": str(predictions.get("interface_flux_mode", "unknown")),
        "outputs": {
            "global_field_quicklook": str(output_dir / f"global_field_quicklook_{suffix}.png"),
            "npz": str(npz_path),
            "metrics_csv": str(metrics_csv_path),
        },
    }


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_arg(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = select_device(args.device)
    model, checkpoint = load_model(checkpoint_path, device)
    train_cfg = checkpoint.get("train_config", {})
    dataset_cfg = train_cfg.get("dataset", {})
    dataset_path = args.dataset or dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5")
    dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=args.split,
        points_per_case=1,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
    )
    raw_dataset = GlobalChannelThermalDataset(dataset_path, split=args.split, points_per_case=1, normalize_inputs=False, normalize_targets=False, random_point_sampling=False, include_grid=True)
    if len(dataset) == 0:
        dataset = GlobalChannelThermalDataset(dataset_path, split="all", points_per_case=1, include_grid=True)
        raw_dataset = GlobalChannelThermalDataset(dataset_path, split="all", points_per_case=1, include_grid=True)
    sample = select_sample(dataset, args.case_id, args.case_index)
    raw_sample = select_sample(raw_dataset, str(sample["case_id"]), args.case_index)
    output_dir = evaluation_output_dir(args.output_dir, checkpoint_path, raw_sample["case_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_order = dataset.channel_order or list(CHANNEL_ORDER)
    requested_modes = ["teacher", "predicted"] if args.local_port_condition_mode == "both" else [args.local_port_condition_mode]
    mode_summaries: Dict[str, Any] = {}
    first_predictions: Optional[Dict[str, Any]] = None
    for mode in requested_modes:
        suffix = "predicted" if mode == "predicted" else str(mode)
        predictions = predict_case(
            model,
            sample,
            device,
            query_batch_size=int(args.query_batch_size),
            local_port_condition_mode=mode,
            mixed_teacher_ratio=float(args.mixed_teacher_ratio),
            return_routing_maps=bool(args.return_routing_maps),
        )
        predictions = denormalize_predictions(predictions, dataset, bool(dataset_cfg.get("normalize_targets", False)))
        predictions["suffix"] = suffix
        if first_predictions is None:
            first_predictions = predictions
        temp_mode = args.temperature_display_mode or ("composite_internal" if np.asarray(predictions["pred_internal_temperature"]).size else "fluid_only")
        plot_field_quicklook(
            output_dir / f"global_field_quicklook_{suffix}.png",
            raw_sample,
            predictions["pred_field_grid"],
            channel_order,
            pred_internal_temperature=predictions["pred_internal_temperature"],
            temperature_display_mode=temp_mode,
        )
        internal_written = plot_internal(output_dir / f"module_internal_temperature_{suffix}.png", raw_sample, predictions["pred_internal_temperature"])
        interface_written = plot_interface(output_dir / f"interface_curves_{suffix}.png", raw_sample, predictions["pred_interface"])
        summary = summarize(raw_sample, predictions, checkpoint_path, output_dir, channel_order)
        summary["temperature_display_mode"] = temp_mode
        summary["outputs"]["module_internal_temperature"] = "skipped_empty" if not internal_written else str(output_dir / f"module_internal_temperature_{suffix}.png")
        summary["outputs"]["interface_curves"] = "skipped_empty" if not interface_written else str(output_dir / f"interface_curves_{suffix}.png")
        mode_summaries[suffix] = summary

    org_outputs: Dict[str, str] = {}
    arrays: Optional[Dict[str, np.ndarray]] = None
    if args.organization_view != "none" and first_predictions is not None:
        arrays = extract_organization_arrays(raw_sample, first_predictions["organizer_aux"])
        radius = module_radius_from_sample(raw_sample, fallback=float(model.config.module_radius))
        if args.organization_view in {"all", "physical"}:
            overview = output_dir / "organization_overview.png"
            render_channelthermal_organization_overview(overview, raw_sample, arrays, module_radius=radius, channel_order=channel_order, link_threshold=float(args.organization_link_threshold))
            alias = output_dir / "organizer_visualization.png"
            copy_figure_alias(overview, alias)
            org_outputs["organization_overview"] = str(overview)
            org_outputs["organizer_visualization"] = str(alias)
        if args.organization_view in {"all", "matrices"}:
            matrices = output_dir / "organization_summary_matrices.png"
            render_channelthermal_organization_summary_matrices(matrices, raw_sample, arrays, module_radius=radius, channel_order=channel_order)
            org_outputs["organization_summary_matrices"] = str(matrices)
        if args.organization_view in {"all", "schematic"}:
            schematic = output_dir / "organization_schematic.png"
            render_channelthermal_organization_schematic_presentation(schematic, raw_sample, arrays, link_threshold=float(args.organization_link_threshold))
            org_outputs["organization_schematic"] = str(schematic)

    if arrays is None and first_predictions is not None:
        arrays = extract_organization_arrays(raw_sample, first_predictions["organizer_aux"])
    radius = module_radius_from_sample(raw_sample, fallback=float(model.config.module_radius))

    routing_outputs: Dict[str, str] = {}
    if args.return_routing_maps and first_predictions is not None and arrays is not None:
        routing_maps = first_predictions.get("routing_maps", {})
        required = {"query_hyper_attention", "pairwise_edge_contribution", "c_H_norm", "c_pair_norm"}
        if required.issubset(routing_maps):
            routing_outputs = save_routing_diagnostics(
                output_dir,
                raw_sample,
                routing_maps,
                arrays,
                module_radius=radius,
                routing_view=str(args.routing_view),
            )

    plan_outputs: Dict[str, str] = {}
    if args.export_hypergraph_plan and first_predictions is not None:
        plan = extract_hypergraph_plan(first_predictions["organizer_aux"], raw_sample["structure"]["module_present"], detach=True)
        plan_path = output_dir / "hypergraph_plan.npz"
        np.savez_compressed(plan_path, **plan)
        plan_summary = {
            "keys": sorted(plan.keys()),
            "shapes": {key: list(value.shape) for key, value in plan.items()},
            "active_hyperedge_count": int(np.asarray(plan.get("active_hyperedge_mask", np.zeros((0,)))).sum()),
            "note": (
                "This compact plan stores static organizer variables only. "
                "Query-dependent alpha_qk is recomputed by the HONF decoder, "
                "and raw module tokens are recomputed from generated physical designs."
            ),
        }
        plan_summary_path = output_dir / "hypergraph_plan_summary.json"
        write_json(plan_summary_path, plan_summary)
        plan_outputs = {
            "hypergraph_plan_npz": str(plan_path),
            "hypergraph_plan_summary": str(plan_summary_path),
        }

    if len(mode_summaries) == 1:
        summary = next(iter(mode_summaries.values()))
        summary["outputs"].update(org_outputs)
        summary["outputs"].update(routing_outputs)
        summary["outputs"].update(plan_outputs)
    else:
        outputs = {}
        outputs.update(org_outputs)
        outputs.update(routing_outputs)
        outputs.update(plan_outputs)
        summary = {"checkpoint": str(checkpoint_path), "case_id": str(raw_sample["case_id"]), "modes": mode_summaries, "outputs": outputs}
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "summary_compact.json").open("w", encoding="utf-8") as f:
        if "field_metrics_fluid" in summary:
            compact = {"case_id": summary["case_id"], "field_metrics_fluid": summary["field_metrics_fluid"], "outputs": summary["outputs"]}
        else:
            compact = {"case_id": summary["case_id"], "modes": {key: value.get("field_metrics_fluid") for key, value in summary.get("modes", {}).items()}, "outputs": summary["outputs"]}
        json.dump(compact, f, indent=2)
    print(f"[done] wrote evaluation outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
