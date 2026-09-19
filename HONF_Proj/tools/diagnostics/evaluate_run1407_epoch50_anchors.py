"""Read-only four-case Run-1407 epoch-50 reconstruction evaluator.

The evaluator loads three explicit epoch-50 checkpoints (Run 1407, Run 1406,
and Dense 1804), predicts the fixed anchor cases 0273/0653/0680/0298, and
reuses the maintained ChannelThermal reconstruction metrics.  It does not
train, create model variants, write checkpoints, collect routing maps, or hash
any input artifact.

Plan validation is CPU-safe and does not import the model runtime.  Physical
execution is opt-in and uses one model at a time on the caller-selected
device::

    PYTHONPATH=src:Case_ThermalChannel/src \
    python tools/diagnostics/evaluate_run1407_epoch50_anchors.py \
        --checkpoint-1407 /path/Run_1407/epoch_0050_model.pt \
        --checkpoint-1406 /path/Run_1406/epoch_0050_model.pt \
        --checkpoint-1804 /path/Run_1804/epoch_0050_model.pt \
        --dataset /path/thermal_channel_global_v1.h5 \
        --output diagnostics/generated/run1407_epoch50_anchors.json \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_IDS = ("0273", "0653", "0680", "0298")
DEFAULT_EPOCH = 50
DEFAULT_QUERY_BATCH_SIZE = 8192
COMPARISON_LABELS = ("1407", "1406", "1804")

# Keep these names identical to the maintained comparison table columns.  The
# first six entries are normalized relative L2 metrics; the last three are
# dataset-native physical relative L2 metrics.
METRIC_SPECS = (
    ("global_field_all_norm_l2", "global_field_all_norm", "norm_l2"),
    ("global_field_fluid_norm_l2", "global_field_fluid_norm", "norm_l2"),
    ("global_field_near_interface_norm_l2", "global_field_near_interface_norm", "norm_l2"),
    ("global_field_far_fluid_norm_l2", "global_field_far_fluid_norm", "norm_l2"),
    ("field_p_fluid_norm_l2", "field_p_fluid_norm", "norm_l2"),
    ("field_omega_fluid_norm_l2", "field_omega_fluid_norm", "norm_l2"),
    ("internal_temperature_physical_relative_l2", "internal_temperature_physical", "relative_l2"),
    ("interface_t_surface_physical_relative_l2", "interface_t_surface_physical", "relative_l2"),
    ("interface_q_normal_physical_relative_l2", "interface_q_normal_physical", "relative_l2"),
)


def _finite(value: Any) -> float | None:
    try:
        result = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return result if math.isfinite(result) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_path(raw: str | Path, *, name: str) -> Path:
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} checkpoint does not exist: {path}")
    return path


def _checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "1407": _parse_path(args.checkpoint_1407, name="Run 1407"),
        "1406": _parse_path(args.checkpoint_1406, name="Run 1406"),
        "1804": _parse_path(args.checkpoint_1804, name="Run 1804"),
    }


def _case_ids(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(str(value) for value in (args.case_id or DEFAULT_CASE_IDS))


def _validate_args(args: argparse.Namespace) -> None:
    cases = _case_ids(args)
    if cases != DEFAULT_CASE_IDS:
        raise ValueError(
            "the Run-1407 anchor evaluator requires exactly cases "
            "0273, 0653, 0680, and 0298 in that order"
        )
    if int(args.query_batch_size) <= 0:
        raise ValueError("query_batch_size must be positive")
    if str(args.local_port_condition_mode) != "predicted":
        raise ValueError("the matched anchor evaluator requires local_port_condition_mode=predicted")
    if not str(args.split).strip():
        raise ValueError("split must be non-empty")


def _checkpoint_policy(paths: Mapping[str, Path]) -> dict[str, Any]:
    roles = {
        "1407": "matched_epoch_50_candidate",
        "1406": "matched_epoch_50_group_control_reference",
        "1804": "matched_epoch_50_dense_reference",
    }
    return {
        label: {
            "label": label,
            "path": str(path),
            "selection_policy": "explicit_cli_checkpoint",
            "required_epoch": DEFAULT_EPOCH,
            "epoch_role": roles[label],
        }
        for label, path in paths.items()
    }


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_epoch": DEFAULT_EPOCH,
        "case_ids": list(_case_ids(args)),
        "split": str(args.split),
        "query_batch_size": int(args.query_batch_size),
        "local_port_condition_mode": str(args.local_port_condition_mode),
        "mixed_teacher_ratio": float(args.mixed_teacher_ratio),
        "routing_maps": False,
        "training": False,
        "checkpoint_writes": False,
        "artifact_hashes": False,
        "model_residency": "one model at a time",
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Validate explicit files and the fixed protocol without model imports."""

    paths = _checkpoint_paths(args)
    _validate_args(args)
    output = Path(args.output).expanduser().resolve()
    report = Path(args.report).expanduser().resolve() if args.report else output.with_suffix(".md")
    return {
        "schema_version": 1,
        "task": "run1407_epoch50_four_case_anchor_evaluation",
        "status": "plan_only",
        "checkpoint_policy": _checkpoint_policy(paths),
        "protocol": _protocol(args),
        "dataset": None if args.dataset is None else str(Path(args.dataset).expanduser().resolve()),
        "planned_outputs": {"json": str(output), "report": str(report)},
        "metric_policy": {
            "per_case": "maintained compare_models.reconstruction_metrics values",
            "aggregate": "pooled relative L2 from summed per-case SSE and target SSE",
            "equal_case_summary": "mean and median of the per-case relative L2 values are also reported",
            "metrics": [name for name, _, _ in METRIC_SPECS],
        },
        "limitations": [
            "Plan-only mode does not load checkpoints, datasets, models, CUDA, or optimizers.",
            "The four cases are fixed anchors and are not a population-level accuracy claim.",
            "Internal temperature, surface temperature, and normal heat flux are reported in dataset-native physical units.",
            "Learned model outputs are not CFD or physical truth without a separate solver-side validation.",
        ],
    }


def _checkpoint_epoch(checkpoint: Mapping[str, Any]) -> int:
    value = checkpoint.get("epoch", checkpoint.get("current_epoch"))
    if value is None and isinstance(checkpoint.get("selection_state"), Mapping):
        value = checkpoint["selection_state"].get("epoch")
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _require_epoch(checkpoint: Mapping[str, Any], label: str) -> None:
    epoch = _checkpoint_epoch(checkpoint)
    if epoch != DEFAULT_EPOCH:
        raise ValueError(
            f"Run {label} checkpoint epoch={epoch}; controlled anchor evaluation requires exact epoch 50"
        )


def _metric_record(row: Mapping[str, Any], base: str, suffix: str) -> dict[str, Any]:
    value = _finite(row.get(f"{base}_{suffix}"))
    sse = _finite(row.get(f"{base}_sse"))
    target_sse = _finite(row.get(f"{base}_target_sse"))
    num_values = _finite(row.get(f"{base}_num_values"))
    return {
        "relative_l2": value,
        "sse": sse,
        "target_sse": target_sse,
        "num_values": None if num_values is None else int(num_values),
    }


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-case reconstruction rows by explicit model label."""

    output: list[dict[str, Any]] = []
    by_label = {label: [row for row in rows if str(row.get("label", row.get("model_label"))) == label] for label in COMPARISON_LABELS}
    for label in COMPARISON_LABELS:
        label_rows = by_label[label]
        metrics: dict[str, Any] = {}
        for name, base, suffix in METRIC_SPECS:
            observations = [_metric_record(row, base, suffix) for row in label_rows]
            values = [item["relative_l2"] for item in observations if item["relative_l2"] is not None]
            sse_values = [item["sse"] for item in observations if item["sse"] is not None]
            target_values = [item["target_sse"] for item in observations if item["target_sse"] is not None]
            counts = [item["num_values"] for item in observations if item["num_values"] is not None]
            pooled_sse = float(sum(sse_values)) if len(sse_values) == len(observations) and observations else None
            pooled_target = float(sum(target_values)) if len(target_values) == len(observations) and observations else None
            metrics[name] = {
                "pooled_relative_l2": (
                    float(math.sqrt(pooled_sse / pooled_target))
                    if pooled_sse is not None and pooled_target is not None and pooled_target > 0.0
                    else None
                ),
                "equal_case_mean": float(np.mean(values)) if values else None,
                "equal_case_median": float(np.median(values)) if values else None,
                "case_count": len(values),
                "pooled_sse": pooled_sse,
                "pooled_target_sse": pooled_target,
                "num_values": int(sum(counts)) if len(counts) == len(observations) and observations else None,
            }
        output.append({"label": label, "case_count": len(label_rows), "metrics": metrics})
    return output


def _selected_case_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "label": str(row.get("label", row.get("model_label"))),
                "case_id": str(row["case_id"]),
                "metrics": {
                    name: _metric_record(row, base, suffix)
                    for name, base, suffix in METRIC_SPECS
                },
            }
        )
    return result


def _build_payload(
    *,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    dataset_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {(label, case_id) for label in COMPARISON_LABELS for case_id in _case_ids(args)}
    observed = {(str(row.get("label", row.get("model_label"))), str(row.get("case_id"))) for row in rows}
    if observed != expected:
        raise ValueError(f"evaluation returned unexpected label/case rows: {sorted(observed)}")
    return {
        "schema_version": 1,
        "task": "run1407_epoch50_four_case_anchor_evaluation",
        "status": "complete",
        "checkpoint_policy": _checkpoint_policy(paths),
        "protocol": _protocol(args),
        "dataset": str(dataset_path),
        "per_case": _selected_case_rows(rows),
        "aggregate": _aggregate_metric_rows(rows),
        "limitations": [
            "Four fixed anchor cases are diagnostic evidence, not a 90-case population ranking.",
            "Internal/surface/flux metrics use dataset-native physical units and are not mixed into the field aggregate.",
            "Learned model outputs are not CFD or physical truth without a separate solver-side validation.",
        ],
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run the explicit read-only physical evaluation on the selected device."""

    _validate_args(args)
    paths = _checkpoint_paths(args)

    for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel" / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model
    from channelthermal.evaluation.prepared import predict_case, select_sample
    from channelthermal.workflows.compare_models import reconstruction_metrics

    from honf_runtime.compat import load_trusted_checkpoint, resolve_demo_path

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")
        torch.cuda.set_device(device)

    checkpoints: dict[str, Mapping[str, Any]] = {}
    for label, path in paths.items():
        checkpoint = load_trusted_checkpoint(path, map_location="cpu")
        _require_epoch(checkpoint, label)
        checkpoints[label] = checkpoint

    first_dataset_cfg = dict(checkpoints[COMPARISON_LABELS[0]].get("train_config", {}).get("dataset", {}))
    dataset_value = args.dataset or first_dataset_cfg.get(
        "packed_h5_path", "./Case_ThermalChannel/Dataset/links/thermal_channel_global_v1.h5"
    )
    dataset_path = resolve_demo_path(dataset_value).resolve()
    raw_dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=str(args.split),
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=True,
    )
    if len(raw_dataset) == 0:
        raise RuntimeError(f"evaluation split={args.split!r} contains no cases")
    if any(str(case_id) not in {str(value) for value in raw_dataset.selected_case_ids} for case_id in _case_ids(args)):
        available = {str(value) for value in raw_dataset.selected_case_ids}
        missing = [case_id for case_id in _case_ids(args) if case_id not in available]
        raise KeyError(f"anchor cases are absent from split {args.split!r}: {missing}")

    rows: list[dict[str, Any]] = []
    try:
        for label in COMPARISON_LABELS:
            checkpoint = checkpoints[label]
            model, _ = load_model(paths[label], device)
            dataset_cfg = dict(checkpoint.get("train_config", {}).get("dataset", {}))
            checkpoint_stats = {
                name: np.asarray(value, dtype=np.float32)
                for name, value in checkpoint.get("global_normalization_stats", {}).items()
            }
            input_dataset = GlobalChannelThermalDataset(
                dataset_path,
                split=str(args.split),
                points_per_case=1,
                normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
                normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
                random_point_sampling=False,
                include_grid=True,
                include_structure_targets=True,
                normalizer=H5Normalizer(checkpoint_stats) if checkpoint_stats else None,
            )
            try:
                if list(input_dataset.selected_case_ids) != list(raw_dataset.selected_case_ids):
                    raise ValueError(f"Run {label} does not expose the same ordered case IDs as the raw comparison view")
                channel_order = list(input_dataset.channel_order)
                configured_fields = list(model.config.channelthermal.field_names)
                if configured_fields != channel_order:
                    raise ValueError(f"Run {label} field order {configured_fields} does not match dataset order {channel_order}")
                checkpoint_targets_normalized = bool(dataset_cfg.get("normalize_targets", False))
                for case_id in _case_ids(args):
                    sample = select_sample(input_dataset, case_id, 0)
                    raw_sample = select_sample(raw_dataset, case_id, 0)
                    with torch.inference_mode():
                        predictions = predict_case(
                            model,
                            sample,
                            device,
                            query_batch_size=int(args.query_batch_size),
                            local_port_condition_mode=str(args.local_port_condition_mode),
                            mixed_teacher_ratio=float(args.mixed_teacher_ratio),
                            return_routing_maps=False,
                            return_prepared_state=False,
                        )
                    metric_row, _ = reconstruction_metrics(
                        base_row={"label": label, "case_id": case_id, "split": str(args.split)},
                        predictions=predictions,
                        raw_sample=raw_sample,
                        dataset=input_dataset,
                        checkpoint_targets_normalized=checkpoint_targets_normalized,
                        channel_order=channel_order,
                    )
                    rows.append(metric_row)
            finally:
                close = getattr(input_dataset, "close", None)
                if callable(close):
                    close()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    finally:
        close = getattr(raw_dataset, "close", None)
        if callable(close):
            close()
    return _build_payload(args=args, paths=paths, dataset_path=dataset_path, rows=rows)


def _format(value: Any) -> str:
    finite = _finite(value)
    return "NA" if finite is None else f"{finite:.6g}"


def _markdown_report(payload: Mapping[str, Any]) -> str:
    metric_names = [name for name, _, _ in METRIC_SPECS]
    lines = [
        "# Run 1407 Epoch-50 Four-Case Anchor Evaluation",
        "",
        "Read-only comparison of explicit epoch-50 checkpoints for Run 1407, Run 1406, and Dense 1804.",
        "",
        f"Cases: `{', '.join(payload.get('protocol', {}).get('case_ids', []))}`; split `{payload.get('protocol', {}).get('split')}`; query batch `{payload.get('protocol', {}).get('query_batch_size')}`; predicted ports.",
        "",
        "## Per-case metrics",
        "",
        "| Label | Case | " + " | ".join(metric_names) + " |",
        "|---|---|" + "---:|" * len(metric_names),
    ]
    for row in payload.get("per_case", []):
        metrics = row.get("metrics", {})
        lines.append(
            "| "
            + str(row.get("label"))
            + " | "
            + str(row.get("case_id"))
            + " | "
            + " | ".join(_format(metrics.get(name, {}).get("relative_l2")) for name in metric_names)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Aggregate metrics",
            "",
            "Pooled values use summed per-case SSE and target SSE. Equal-case mean and median are included for context.",
            "",
            "| Label | Metric | Pooled relative L2 | Equal-case mean | Equal-case median | Cases |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for group in payload.get("aggregate", []):
        label = group.get("label")
        for name in metric_names:
            metric = group.get("metrics", {}).get(name, {})
            lines.append(
                f"| {label} | {name} | {_format(metric.get('pooled_relative_l2'))} | {_format(metric.get('equal_case_mean'))} | {_format(metric.get('equal_case_median'))} | {metric.get('case_count', 0)} |"
            )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "The four anchors are diagnostic evidence, not a population-level ranking. Internal temperature, surface temperature, and normal heat flux are dataset-native physical metrics and are not mixed into the field aggregate. Model outputs remain surrogate predictions pending separate CFD/physical validation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-1407", required=True, type=Path)
    parser.add_argument("--checkpoint-1406", required=True, type=Path)
    parser.add_argument("--checkpoint-1804", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument("--local-port-condition-mode", choices=("predicted", "teacher", "mixed"), default="predicted")
    parser.add_argument("--mixed-teacher-ratio", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    if args.plan_only:
        _write_json(Path(args.output), plan)
        print(json.dumps(_jsonable(plan), indent=2, sort_keys=True))
        return 0
    payload = run_evaluation(args)
    _write_json(Path(args.output), payload)
    report = Path(args.report).expanduser().resolve() if args.report else Path(args.output).expanduser().resolve().with_suffix(".md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_markdown_report(payload), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARISON_LABELS",
    "DEFAULT_CASE_IDS",
    "DEFAULT_EPOCH",
    "METRIC_SPECS",
    "_aggregate_metric_rows",
    "build_parser",
    "build_plan",
    "main",
    "run_evaluation",
]
