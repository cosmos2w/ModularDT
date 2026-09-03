"""Evaluate and visualize a trained ChannelThermal HONF checkpoint.

Use a direct checkpoint path or ``--Run_ID`` plus a named checkpoint selector.
The evaluator applies checkpoint-owned normalization, reconstructs a selected
case, and can write field/internal/interface plots, organizer and routing
visualizations, compressed arrays, metrics, and an inverse-design plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-honf_cl")

import numpy as np
import torch

from channelthermal.data.datasets import CHANNEL_ORDER, GlobalChannelThermalDataset, H5Normalizer
from channelthermal.evaluation_tools.plots import (
    error_metrics,
    masked_error_metrics,
    module_and_fluid_masks,
    module_radius_from_sample,
    plot_field_quicklook,
    plot_interface,
    plot_internal,
)
from honf_forward_core.evaluation.hypergraph_plan import (
    extract_hypergraph_plan,
    save_hypergraph_plan,
    summarize_hypergraph_plan,
    validate_hypergraph_plan,
)
from honf_forward_core.evaluation.topology_signature import (
    evaluate_structure_relations,
    extract_topology_signature,
    save_topology_signature,
    summarize_topology_signature,
)
from honf_runtime.compat import current_timestamp, load_trusted_checkpoint, recursive_to_device, resolve_demo_path, select_device, strip_module_prefix, write_json
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.artifact_layout import (
    EVALUATION_LAYOUT_VERSION,
    EvaluationArtifactLayout,
    default_evaluation_root,
    finalize_evaluation_job,
)
from channelthermal.evaluation_tools.organizer_visualization import (
    render_case_adaptive_residual_summary,
    render_channelthermal_organization_overview,
    render_channelthermal_organization_schematic_presentation,
    render_channelthermal_organization_summary_matrices,
)
from channelthermal.evaluation_tools.routing_visualization import save_routing_diagnostics
from channelthermal.evaluation_tools.topology_signature_visualization import (
    render_topology_signature_diagnostics,
)
from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.local_surrogate.model import LocalModuleConfig, LocalModuleSurrogate
from channelthermal.evaluation import loading as _evaluation_loading
from channelthermal.evaluation import results as _evaluation_results
from channelthermal.evaluation.loading import (
    apply_frozen_forward_overrides,
    checkpoint_file_name,
    latest_run_dir,
    load_model,
    make_batch,
    normalize_run_id,
    numpy_to_batched_tensor,
    resolve_checkpoint_arg,
)
from channelthermal.evaluation.prepared import (
    aggregate_routed_module_retention,
    predict_case,
    select_sample,
)
from channelthermal.evaluation.results import (
    denormalize_predictions,
    evaluation_output_dir,
    extract_organization_arrays,
    file_sha256,
    hypergraph_diagnostics,
    safe_path_name,
    summarize,
)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ChannelThermalHONFModel, Dict[str, Any]]:
    """Preserve the workflow-level trusted-loader injection contract."""

    _evaluation_loading.load_trusted_checkpoint = load_trusted_checkpoint
    return _evaluation_loading.load_model(checkpoint_path, device)


def evaluation_output_dir(
    base_dir_arg: str | None,
    checkpoint_path: Path,
    case_id: object,
) -> Path:
    """Preserve the workflow-level timestamp injection contract."""

    _evaluation_results.current_timestamp = current_timestamp
    return _evaluation_results.evaluation_output_dir(
        base_dir_arg,
        checkpoint_path,
        case_id,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line options for this workflow."""

    parser = argparse.ArgumentParser(description="Evaluate a global standalone ChannelThermal HONF checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="best", help="best, best_by_field_mse, best_by_temperature_mse, latest, best_predicted, or a .pt path.")
    parser.add_argument("--Run_ID", dest="run_id", type=str, default=None)
    parser.add_argument("--saved-root", type=str, default="./Trained_Results/ThermalChannel/HONF_Forward_Runs")
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
    parser.add_argument(
        "--export-topology-signature",
        action="store_true",
        help="Export the unordered schema-v3 topology signature and its diagnostic views.",
    )
    parser.add_argument(
        "--allow-checkpoint-fallback",
        action="store_true",
        help="Permit best_predicted to fall back to best only when explicitly requested.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run this command-line workflow and return its process status."""

    args = parse_args(argv)
    checkpoint_path = resolve_checkpoint_arg(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = select_device(args.device)
    model, checkpoint = load_model(checkpoint_path, device)
    train_cfg = checkpoint.get("train_config", {})
    dataset_cfg = train_cfg.get("dataset", {})
    checkpoint_stats = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    checkpoint_normalizer = H5Normalizer(checkpoint_stats) if checkpoint_stats else None
    dataset_path = args.dataset or dataset_cfg.get("packed_h5_path", "./Case_ThermalChannel/Dataset/links/thermal_channel_global_v1.h5")
    dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=args.split,
        points_per_case=1,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=bool(args.export_topology_signature),
        normalizer=checkpoint_normalizer,
    )
    raw_dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=args.split,
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=bool(args.export_topology_signature),
    )
    if len(dataset) == 0:
        dataset = GlobalChannelThermalDataset(
            dataset_path,
            split="all",
            points_per_case=1,
            normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
            normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
            random_point_sampling=False,
            include_grid=True,
            include_structure_targets=bool(args.export_topology_signature),
            normalizer=checkpoint_normalizer,
        )
        raw_dataset = GlobalChannelThermalDataset(
            dataset_path,
            split="all",
            points_per_case=1,
            include_grid=True,
            include_structure_targets=bool(args.export_topology_signature),
        )
    sample = select_sample(dataset, args.case_id, args.case_index)
    raw_sample = select_sample(raw_dataset, str(sample["case_id"]), args.case_index)
    output_dir = evaluation_output_dir(args.output_dir, checkpoint_path, raw_sample["case_id"])
    layout = EvaluationArtifactLayout.at(output_dir)
    layout.ensure("fields")
    channel_order = dataset.channel_order or list(CHANNEL_ORDER)
    requested_modes = ["predicted", "teacher"] if args.local_port_condition_mode == "both" else [args.local_port_condition_mode]
    mode_summaries: Dict[str, Any] = {}
    first_predictions: Optional[Dict[str, Any]] = None
    primary_predictions: Optional[Dict[str, Any]] = None
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
            return_topology_signature=bool(args.export_topology_signature),
        )
        predictions = denormalize_predictions(predictions, dataset, bool(dataset_cfg.get("normalize_targets", False)))
        predictions["suffix"] = suffix
        if first_predictions is None:
            first_predictions = predictions
        if mode == "predicted" or primary_predictions is None:
            primary_predictions = predictions
        temp_mode = args.temperature_display_mode or ("composite_internal" if np.asarray(predictions["pred_internal_temperature"]).size else "fluid_only")
        plot_field_quicklook(
            layout.fields / f"global_field_quicklook_{suffix}.png",
            raw_sample,
            predictions["pred_field_grid"],
            channel_order,
            pred_internal_temperature=predictions["pred_internal_temperature"],
            temperature_display_mode=temp_mode,
        )
        internal_path = layout.fields / f"module_internal_temperature_{suffix}.png"
        interface_path = layout.fields / f"interface_curves_{suffix}.png"
        internal_written = plot_internal(internal_path, raw_sample, predictions["pred_internal_temperature"])
        interface_written = plot_interface(interface_path, raw_sample, predictions["pred_interface"])
        summary = summarize(raw_sample, predictions, checkpoint_path, layout, channel_order)
        summary["temperature_display_mode"] = temp_mode
        summary["outputs"]["module_internal_temperature"] = "skipped_empty" if not internal_written else str(internal_path)
        summary["outputs"]["interface_curves"] = "skipped_empty" if not interface_written else str(interface_path)
        mode_summaries[suffix] = summary

    org_outputs: Dict[str, str] = {}
    arrays: Optional[Dict[str, np.ndarray]] = None
    if primary_predictions is None:
        primary_predictions = first_predictions
    if args.organization_view != "none" and primary_predictions is not None:
        layout.ensure("organization")
        arrays = extract_organization_arrays(raw_sample, primary_predictions["organizer_aux"])
        radius = module_radius_from_sample(raw_sample, fallback=float(model.config.module_radius))
        style = str(args.organization_style)
        if style not in {"presentation", "debug", "both"}:
            print(f"[warning] unknown --organization-style={style!r}; using presentation.")
            style = "presentation"
        render_presentation = style in {"presentation", "both"}
        render_debug = style in {"debug", "both"}
        if args.organization_view in {"all", "physical"} and render_presentation:
            overview = layout.organization / "organization_overview.png"
            render_channelthermal_organization_overview(overview, raw_sample, arrays, module_radius=radius, channel_order=channel_order, link_threshold=float(args.organization_link_threshold))
            org_outputs["organization_overview"] = str(overview)
        if args.organization_view in {"all", "matrices"} and (render_presentation or render_debug):
            matrices = layout.organization / "organization_summary_matrices.png"
            render_channelthermal_organization_summary_matrices(
                matrices,
                raw_sample,
                arrays,
                module_radius=radius,
                channel_order=channel_order,
                sort_environment=False,
            )
            org_outputs["organization_summary_matrices"] = str(matrices)
            sorted_matrices = layout.organization / "organization_summary_matrices_sorted_by_dominant_edge.png"
            render_channelthermal_organization_summary_matrices(
                sorted_matrices,
                raw_sample,
                arrays,
                module_radius=radius,
                channel_order=channel_order,
                sort_environment=True,
            )
            org_outputs["organization_summary_matrices_sorted_by_dominant_edge"] = str(sorted_matrices)
        if args.organization_view in {"all", "schematic"} and render_presentation:
            schematic = layout.organization / "organization_schematic.png"
            render_channelthermal_organization_schematic_presentation(schematic, raw_sample, arrays, link_threshold=float(args.organization_link_threshold))
            org_outputs["organization_schematic"] = str(schematic)
        if (
            "residual_fraction_trace" in primary_predictions["organizer_aux"]
            and np.asarray(arrays["residual_fraction_trace"]).size
        ):
            residual_summary = layout.organization / "residual_mechanism_summary.png"
            render_case_adaptive_residual_summary(residual_summary, arrays)
            org_outputs["residual_mechanism_summary"] = str(residual_summary)

    if arrays is None and primary_predictions is not None:
        arrays = extract_organization_arrays(raw_sample, primary_predictions["organizer_aux"])
    radius = module_radius_from_sample(raw_sample, fallback=float(model.config.module_radius))

    routing_outputs: Dict[str, str] = {}
    if args.return_routing_maps and primary_predictions is not None and arrays is not None:
        routing_maps = primary_predictions.get("routing_maps", {})
        required = {"query_hyper_attention", "pairwise_edge_contribution", "c_H_norm", "c_pair_norm"}
        if required.issubset(routing_maps):
            layout.ensure("routing")
            routing_outputs = save_routing_diagnostics(
                layout.routing,
                raw_sample,
                routing_maps,
                arrays,
                module_radius=radius,
                routing_view=str(args.routing_view),
            )

    plan_outputs: Dict[str, str] = {}
    diagnostics_outputs: Dict[str, str] = {}
    if primary_predictions is not None:
        layout.ensure("diagnostics")
        diagnostics_path = layout.diagnostics / "hypergraph_diagnostics.json"
        write_json(diagnostics_path, hypergraph_diagnostics(primary_predictions))
        diagnostics_outputs["hypergraph_diagnostics"] = str(diagnostics_path)

    if args.export_hypergraph_plan and primary_predictions is not None:
        layout.ensure("plans")
        structure = raw_sample["structure"]
        plan = extract_hypergraph_plan(
            primary_predictions["organizer_aux"],
            structure["module_present"],
            detach=True,
            domain_length_x=float(np.asarray(structure["domain_length_x"]).reshape(-1)[0]),
            domain_length_y=float(np.asarray(structure["domain_length_y"]).reshape(-1)[0]),
        )
        validate_hypergraph_plan(plan)
        plan_path = layout.plans / "hypergraph_plan.npz"
        save_hypergraph_plan(plan_path, plan)
        plan_summary = summarize_hypergraph_plan(plan)
        plan_summary_path = layout.plans / "hypergraph_plan_summary.json"
        write_json(plan_summary_path, plan_summary)
        plan_outputs = {
            "hypergraph_plan_npz": str(plan_path),
            "hypergraph_plan_summary": str(plan_summary_path),
        }

    topology_outputs: Dict[str, str] = {}
    if args.export_topology_signature and primary_predictions is not None:
        layout.ensure("topology")
        structure = raw_sample["structure"]
        routing_maps = primary_predictions.get("routing_maps", {})
        reference_query_xy = np.stack(
            (np.asarray(raw_sample["x_grid"]).reshape(-1), np.asarray(raw_sample["y_grid"]).reshape(-1)),
            axis=-1,
        ).astype(np.float32)
        decoder_summary = {
            key: routing_maps[key]
            for key in ("query_hyper_attention", "pred_field_by_edge")
            if key in routing_maps
        }
        signature = extract_topology_signature(
            primary_predictions["organizer_aux"],
            structure["module_present"],
            decoder_outputs=decoder_summary,
            reference_query_xy=reference_query_xy,
            reference_measure="channelthermal_evaluation_grid",
            field_names=channel_order,
            domain_length_x=float(np.asarray(structure["domain_length_x"]).reshape(-1)[0]),
            domain_length_y=float(np.asarray(structure["domain_length_y"]).reshape(-1)[0]),
            periodic_axes=tuple(int(axis) for axis in model.config.core_honf.periodic_axes),
            case_id=str(raw_sample["case_id"]),
            forward_checkpoint_sha256=file_sha256(checkpoint_path),
        )
        signature_path = layout.topology / "topology_signature.npz"
        save_topology_signature(signature_path, signature)
        signature_summary_path = layout.topology / "topology_signature_summary.json"
        write_json(signature_summary_path, summarize_topology_signature(signature))
        topology_outputs = {
            "topology_signature_npz": str(signature_path),
            "topology_signature_summary": str(signature_summary_path),
        }

        targets = raw_sample.get("structure_targets")
        if isinstance(targets, dict):
            relation_metrics = evaluate_structure_relations(
                signature,
                module_affinity_target=targets.get("module_affinity_target"),
                module_affinity_mask=targets.get("module_affinity_target_mask"),
                query_hyper_attention=primary_predictions.get("structure_query_hyper_attention"),
                environment_module_target=targets.get("env_module_influence_target"),
                environment_module_mask=targets.get("env_module_target_mask"),
                active_edge_count_target=targets.get("active_edge_count_target"),
                has_solved_targets=bool(
                    float(np.asarray(targets.get("has_solved_structure_targets", [0.0])).reshape(-1)[0])
                    > 0.5
                ),
            )
            relation_metrics_path = layout.topology / "topology_relation_metrics.json"
            write_json(relation_metrics_path, relation_metrics)
            topology_outputs["topology_relation_metrics"] = str(relation_metrics_path)
        topology_outputs.update(
            render_topology_signature_diagnostics(
                layout.topology,
                raw_sample,
                signature,
                edge_fields=routing_maps.get("pred_field_by_edge"),
                field_names=channel_order,
            )
        )

    if len(mode_summaries) == 1:
        summary = next(iter(mode_summaries.values()))
        summary["artifact_layout_version"] = EVALUATION_LAYOUT_VERSION
        summary["primary_export_mode"] = str(primary_predictions.get("suffix", "predicted")) if primary_predictions else "unknown"
        summary["outputs"].update(org_outputs)
        summary["outputs"].update(routing_outputs)
        summary["outputs"].update(plan_outputs)
        summary["outputs"].update(topology_outputs)
        summary["outputs"].update(diagnostics_outputs)
    else:
        outputs = {}
        outputs.update(org_outputs)
        outputs.update(routing_outputs)
        outputs.update(plan_outputs)
        outputs.update(topology_outputs)
        outputs.update(diagnostics_outputs)
        summary = {
            "artifact_layout_version": EVALUATION_LAYOUT_VERSION,
            "checkpoint": str(checkpoint_path),
            "case_id": str(raw_sample["case_id"]),
            "primary_export_mode": str(primary_predictions.get("suffix", "predicted")) if primary_predictions else "unknown",
            "modes": mode_summaries,
            "outputs": outputs,
        }
    write_json(output_dir / "summary.json", summary)
    finalize_evaluation_job(
        output_dir,
        kind="forward_single_case",
        checkpoint_path=checkpoint_path,
        requested_checkpoint=str(args.checkpoint),
    )
    print(f"[done] wrote evaluation outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
