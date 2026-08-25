"""Run bounded Stage-4 train/save/resume/closure smoke checks.

The runner uses mapped ThermalChannel data and the configured frozen Stage-A
checkpoint, but reduces sampling, batch size, workers, and epoch/batch counts.
It is an integration gate only and must not be used for scientific claims.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from channelthermal.data.collation import ChannelThermalBatchCollator
from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.plugin import ThermalChannelPlugin
from channelthermal.workflows.train_forward import (
    build_model_config,
    make_model_inputs,
    resolve_auto_internal_mode,
    run_from_config,
)
from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.compat import recursive_to_device, strip_module_prefix
from honf_runtime.config_loader import load_config_bundle


OVERLAYS = (
    "stage4_uniform_lr2e4_dense_background.json",
    "stage4_split_lr_dense_background.json",
    "stage4_split_lr_pooled_background.json",
)


def _request(*, device: str, epochs: int, resume: Path | None = None) -> WorkflowRequest:
    return WorkflowRequest(
        workflow="forward",
        device=device,
        run_id="9904",
        run_name="stage4_bounded_smoke",
        epochs=epochs,
        max_train_batches=1,
        max_val_batches=1,
        resume_checkpoint=None if resume is None else str(resume),
    )


def _workflow_args(request: WorkflowRequest) -> SimpleNamespace:
    return SimpleNamespace(
        config=None,
        device=request.device,
        epochs=request.epochs,
        max_train_batches=request.max_train_batches,
        max_val_batches=request.max_val_batches,
        run_name=request.run_name,
        run_id=request.run_id,
        resume_checkpoint=request.resume_checkpoint,
        initialize_checkpoint=None,
    )


def _bounded_config(plugin, bundle, request, run_dir: Path) -> dict:
    config = plugin._forward_config(bundle, request, run_dir)  # noqa: SLF001 - diagnostic tool
    config = copy.deepcopy(config)
    config["dataset"].update(
        {
            "points_per_case": 32,
            "val_points_per_case": 32,
            "batch_size": 1,
            "val_batch_size": 1,
            "num_workers": 0,
            "bucket_by_module_count": False,
        }
    )
    config["training"]["plot_every_epochs"] = 1000
    return config


def _assert_metrics(metrics_path: Path) -> dict[str, str]:
    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2:
        raise AssertionError(f"Expected two completed smoke epochs, found {len(rows)}.")
    last = rows[-1]
    for name, value in last.items():
        if value not in (None, "") and name != "epoch" and not math.isfinite(float(value)):
            raise AssertionError(f"Non-finite smoke metric {name}={value}.")
    for name in (
        "empty_selected_edge_count",
        "post_fallback_zero_support_module_rows",
        "post_fallback_zero_support_environment_rows",
        "val_empty_selected_edge_count",
        "val_post_fallback_zero_support_module_rows",
        "val_post_fallback_zero_support_environment_rows",
    ):
        if float(last[name]) != 0.0:
            raise AssertionError(f"Smoke support diagnostic {name}={last[name]}.")
    return last


def _closure_error(run_dir: Path, device: torch.device) -> float:
    config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    dataset = GlobalChannelThermalDataset(
        dataset_cfg["packed_h5_path"],
        split=dataset_cfg["val_split"],
        points_per_case=32,
        normalize_inputs=bool(dataset_cfg["normalize_inputs"]),
        normalize_targets=bool(dataset_cfg["normalize_targets"]),
        random_point_sampling=False,
        seed=int(config["training"]["seed"]) + 1000,
        require_converged=bool(dataset_cfg["require_converged"]),
    )
    model_config = build_model_config(config, dataset)
    model = ChannelThermalHONFModel(model_config).to(device)
    model.set_global_target_normalization(
        dataset.normalizer.stats,
        normalize_targets=bool(dataset_cfg["normalize_targets"]),
    )
    resolve_auto_internal_mode(model_config, model)
    collator = ChannelThermalBatchCollator(dynamic_module_padding=True)
    batch = recursive_to_device(collator([dataset[0]]), device)
    checkpoint = torch.load(run_dir / "latest_model.pt", map_location=device, weights_only=False)
    # Materialize lazy adapters before strict loading, matching evaluation.
    model.eval()
    with torch.no_grad():
        model(**make_model_inputs(
            batch,
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.0,
        ))
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    selection_state = checkpoint.get("selection_state") or {}
    model.set_training_progress(
        epoch=int(selection_state.get("epoch", checkpoint.get("epoch", 2))),
        total_epochs=int(selection_state.get("total_epochs", 2)),
    )
    with torch.no_grad():
        model_inputs = make_model_inputs(
            batch,
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.0,
        )
        model_inputs["return_edge_fields"] = True
        output = model(**model_inputs)
    reconstructed = output["pred_field_background"] + output["pred_field_by_edge"].sum(dim=2)
    return float((output["pred_field"] - reconstructed).abs().max().cpu())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    plugin = ThermalChannelPlugin()
    summaries = []
    for overlay_name in OVERLAYS:
        bundle = load_config_bundle(
            "project://src/config_core/forward/adaptive_sparse_additive.json",
            experiment_overlay=f"project://src/config_core/forward/experiments/{overlay_name}",
        )
        run_dir = (args.output_root / Path(overlay_name).stem).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        first_request = _request(device=args.device, epochs=1)
        first_config = _bounded_config(plugin, bundle, first_request, run_dir)
        run_from_config(first_config, _workflow_args(first_request), run_dir_override=run_dir)
        latest = run_dir / "latest_model.pt"
        resume_request = _request(device=args.device, epochs=2, resume=latest)
        resume_config = _bounded_config(plugin, bundle, resume_request, run_dir)
        run_from_config(resume_config, _workflow_args(resume_request), run_dir_override=run_dir)
        last = _assert_metrics(run_dir / "metrics.csv")
        closure_error = _closure_error(run_dir, torch.device(args.device))
        if closure_error != 0.0:
            raise AssertionError(f"Additive closure error for {overlay_name}: {closure_error}")
        summaries.append(
            {
                "overlay": overlay_name,
                "final_loss": float(last["loss_total"]),
                "final_val_loss": float(last["val_loss_total"]),
                "closure_max_abs_error": closure_error,
                "latest_checkpoint": str(latest),
            }
        )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
