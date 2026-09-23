"""Traverse an explicit Run-1409 checkpoint across the 90-case population.

This is the runnable counterpart to the CPU evidence board helper.  It loads
one checkpoint from the caller supplied path, evaluates every selected case
with predicted ports, collects the live occupancy maps, and writes the
Kplan/ID/support/spatial evidence files and figures.  The model is always
evaluated on CPU and no checkpoint or managed run is created.

The query grid can be reduced deterministically with ``--query-count`` to
keep a prelaunch population pass bounded.  P0 plan quantities are case static;
query support metrics then describe the explicitly recorded receiver subset.

Example::

    PYTHONPATH=src:Case_ThermalChannel/src:tools/diagnostics \
      python tools/diagnostics/run_run1409_occupancy_population.py \
      --checkpoint /abs/path/to/Run_1409/epoch_0050_model.pt \
      --output-dir /abs/path/to/Run_1409/evaluations/occupancy_adaptive_population \
      --device cpu --split test --query-count 1024
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CASE_COUNT = 90
DEFAULT_KMAX = 12
DEFAULT_QUERY_COUNT = 1024


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return value.tolist()
        finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else np.asarray([])
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.size),
            "min": None if finite.size == 0 else float(np.min(finite)),
            "max": None if finite.size == 0 else float(np.max(finite)),
            "mean": None if finite.size == 0 else float(np.mean(finite)),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_imports(evidence_module: str = "occupancy_adaptive_evidence") -> tuple[Any, ...]:
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import importlib

    evidence = importlib.import_module(str(evidence_module))
    import torch
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model
    from channelthermal.evaluation.prepared import predict_case

    return torch, GlobalChannelThermalDataset, H5Normalizer, load_model, predict_case, evidence


def _state_array(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"prepared occupancy state did not expose {name}")
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if not np.isfinite(array).all():
        raise RuntimeError(f"prepared occupancy state {name} contains non-finite values")
    return array


def _query_sample(sample: Mapping[str, Any], requested: int | None) -> tuple[dict[str, Any], np.ndarray]:
    x_grid = np.asarray(sample["x_grid"], dtype=np.float32).reshape(-1)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32).reshape(-1)
    if x_grid.shape != y_grid.shape or not x_grid.size:
        raise RuntimeError("sample x_grid/y_grid must be nonempty and aligned")
    if requested is None:
        indices = np.arange(x_grid.size, dtype=np.int64)
    else:
        if int(requested) <= 0 or int(requested) > x_grid.size:
            raise ValueError(f"--query-count must be in [1,{x_grid.size}], got {requested}")
        if int(requested) == x_grid.size:
            indices = np.arange(x_grid.size, dtype=np.int64)
        else:
            indices = np.rint(np.linspace(0, x_grid.size - 1, int(requested))).astype(np.int64)
            indices = np.unique(indices)
    selected = dict(sample)
    selected["x_grid"] = x_grid[indices]
    selected["y_grid"] = y_grid[indices]
    query_xy = np.stack([x_grid[indices], y_grid[indices]], axis=-1).astype(np.float32)
    return selected, query_xy


def _prediction_payload(
    sample: Mapping[str, Any],
    prediction: Mapping[str, Any],
    query_xy: np.ndarray,
) -> dict[str, Any]:
    """Combine detached live aux maps with explicit geometry for the board."""

    payload: dict[str, Any] = dict(prediction.get("interaction_aux", {}))
    payload.update(prediction.get("routing_maps", {}))
    matrix_keys = {
        "occupancy_group_module_membership",
        "group_control_module_incidence",
        "occupancy_group_environment_membership",
        "group_control_environment_incidence",
        "occupancy_group_proposal_module_membership",
        "occupancy_group_proposal_environment_membership",
        "occupancy_group_module_centres",
        "group_control_module_centres",
        "occupancy_group_environment_centres",
        "group_control_environment_centres",
        "occupancy_group_joint_centres",
    }
    vector_keys = {
        "occupancy_group_module_measure",
        "group_control_module_measure",
        "occupancy_group_environment_measure",
        "group_control_environment_measure",
        "occupancy_group_active_mask",
        "occupancy_group_plan_active_mask",
        "occupancy_group_prototype_ids",
        "occupancy_group_plan_prototype_ids",
        "occupancy_group_packed_prototype_ids",
        "occupancy_group_packed_valid",
        "occupancy_group_plan_packed_valid",
        "occupancy_group_module_mass",
        "group_control_module_mass",
        "occupancy_group_environment_mass",
        "group_control_environment_mass",
    }
    for key in matrix_keys:
        if key in payload and np.asarray(payload[key]).ndim == 2:
            payload[key] = np.asarray(payload[key])[None, ...]
    for key in vector_keys:
        if key in payload and np.asarray(payload[key]).ndim == 1:
            payload[key] = np.asarray(payload[key])[None, ...]
    phase_sources: dict[str, dict[str, Any]] = {}
    for source in ("module", "environment"):
        prefix = f"group_control_{source}_"
        actual = payload.get(prefix + "fine_rows_forward", payload.get(prefix + "fine_rows"))
        padded = payload.get(prefix + "fine_rows_padded", payload.get(prefix + "padded_rows"))
        logical = payload.get(prefix + "logical_paths")
        unique = payload.get(prefix + "unique_pairs")
        if any(value is not None for value in (actual, padded, logical, unique)):
            phase_sources[source] = {
                "actual_rows": actual,
                "padded_rows": padded,
                "logical_paths": logical,
                "unique_pairs": unique,
            }
    if phase_sources:
        payload["occupancy_group_phase_ledger"] = {"P2": phase_sources}
    # ``canonicalize_case`` accepts model-style batched tensors.  Preserve a
    # singleton case axis for geometry and explicit measures too; otherwise an
    # unbatched source axis can be mistaken for a multi-case batch.
    payload["query_xy"] = query_xy[None, ...]
    payload["module_coords"] = np.asarray(
        sample["structure"]["module_centers"], dtype=np.float32
    )[None, ...]
    payload["module_present"] = np.asarray(
        sample["structure"]["module_present"], dtype=np.float32
    )[None, ...]

    prepared_case = prediction.get("_prepared_state")
    prepared = getattr(prepared_case, "prepared", None)
    encoded = getattr(prepared, "encoded", None)
    if encoded is None:
        raise RuntimeError("occupancy population requires the prepared interface encoding")
    payload["environment_coords"] = _state_array(
        getattr(encoded, "env_coords", None), name="env_coords"
    )[None, ...]
    payload["environment_measure"] = _state_array(
        getattr(encoded, "env_weights", None), name="env_weights"
    )[None, ...]

    query_assignment = None
    for key in (
        "occupancy_group_query_routing",
        "group_control_query_routing",
        "query_assignment",
    ):
        if key in payload:
            query_assignment = payload[key]
            break
    if query_assignment is None:
        raise RuntimeError(
            "live occupancy query assignment is absent; request return_routing_maps and inspect the checkpoint API"
        )
    query_array = np.asarray(query_assignment)
    if query_array.ndim >= 3 and query_array.shape[0] == 1:
        query_array = query_array[0]
    if query_array.ndim != 2 or int(query_array.shape[0]) != int(query_xy.shape[0]):
        raise RuntimeError(
            f"query assignment shape {query_array.shape} does not cover the recorded query subset {query_xy.shape[0]}"
        )
    payload["query_assignment"] = query_array[None, ...]
    return payload


def _dataset_for_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    dataset_path: str | None,
    split: str,
    GlobalChannelThermalDataset: Any,
    H5Normalizer: Any,
) -> tuple[Any, Path, dict[str, Any]]:
    train_config = checkpoint.get("train_config", {})
    dataset_config = dict(train_config.get("dataset", {})) if isinstance(train_config, Mapping) else {}
    raw_path = dataset_path or dataset_config.get(
        "packed_h5_path",
        "./Case_ThermalChannel/Dataset/links/thermal_channel_global_v1.h5",
    )
    resolved = Path(raw_path).expanduser()
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"explicit dataset path does not exist: {resolved}")
    stats = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    normalizer = H5Normalizer(stats) if stats else None
    dataset = GlobalChannelThermalDataset(
        str(resolved),
        split=str(split),
        points_per_case=1,
        normalize_inputs=bool(dataset_config.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_config.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=False,
        normalizer=normalizer,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"dataset split {split!r} has no cases in {resolved}")
    return dataset, resolved, dataset_config


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["case_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(_jsonable(row.get(field)), sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else _jsonable(row.get(field, ""))
                    for field in fields
                }
            )


def run_population(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.device) != "cpu":
        raise ValueError("group-control population traversal is CPU-only; pass --device cpu")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, GlobalChannelThermalDataset, H5Normalizer, load_model, predict_case, evidence = _runtime_imports(
        str(getattr(args, "evidence_module", "occupancy_adaptive_evidence"))
    )
    device = torch.device("cpu")
    model, checkpoint = load_model(checkpoint_path, device)
    architecture = str(model.config.core_honf.forward_architecture)
    expected_architecture = str(
        getattr(args, "expected_architecture", "occupancy_adaptive_group_control_honf")
    )
    if architecture != expected_architecture:
        raise RuntimeError(
            f"checkpoint architecture is {architecture!r}, expected {expected_architecture!r}"
        )
    dataset, resolved_dataset, dataset_config = _dataset_for_checkpoint(
        checkpoint,
        dataset_path=args.dataset,
        split=args.split,
        GlobalChannelThermalDataset=GlobalChannelThermalDataset,
        H5Normalizer=H5Normalizer,
    )
    available = [str(value) for value in dataset.selected_case_ids]
    case_ids = [str(value) for value in args.case_id] if args.case_id else available
    missing = [case_id for case_id in case_ids if case_id not in set(available)]
    if missing:
        raise KeyError(f"case IDs are absent from split {args.split!r}: {missing}")
    if int(args.expected_cases) > 0 and len(case_ids) != int(args.expected_cases):
        raise RuntimeError(
            f"expected {int(args.expected_cases)} explicit cases, selected {len(case_ids)}; "
            "pass --expected-cases 0 only for a bounded subset"
        )
    index_by_case = {case_id: index for index, case_id in enumerate(available)}
    query_batch_size = int(args.query_batch_size)
    if query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")
    kmax = int(args.kmax)
    array_dir = output_dir / "arrays"
    figure_dir = output_dir / "figures"
    rows: list[dict[str, Any]] = []
    array_paths: dict[str, str] = {}
    board_paths: list[str] = []
    progress_path = output_dir / "population_progress.json"
    for order, case_id in enumerate(case_ids):
        sample = dataset[index_by_case[case_id]]
        selected_sample, query_xy = _query_sample(sample, args.query_count)
        prediction = predict_case(
            model,
            selected_sample,
            device,
            query_batch_size=query_batch_size,
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.0,
            return_routing_maps=True,
            return_prepared_state=True,
        )
        payload = _prediction_payload(selected_sample, prediction, query_xy)
        record = evidence.canonicalize_case(
            payload,
            query_count=int(query_xy.shape[0]),
            kmax=kmax,
        )
        row = {"case_id": case_id, **record["case"]}
        rows.append(row)
        array_path = array_dir / f"{case_id}.npz"
        evidence.save_case_arrays(array_path, record["maps"])
        array_paths[case_id] = str(array_path)
        board_prefix = str(getattr(args, "board_prefix", "occupancy_board"))
        board_path = figure_dir / f"{board_prefix}__{case_id}.png"
        evidence.render_case_board(record, board_path)
        board_paths.append(str(board_path))
        _write_json(
            progress_path,
            {
                "status": "running",
                "completed_cases": order + 1,
                "expected_cases": len(case_ids),
                "last_case_id": case_id,
                "checkpoint": str(checkpoint_path),
            },
        )
        progress_label = str(getattr(args, "progress_label", "occupancy-population"))
        print(f"[{progress_label}] {order + 1}/{len(case_ids)} case={case_id}", flush=True)
        del prediction, payload, record, selected_sample, sample

    summary = evidence.summarize_population(rows, expected_cases=int(args.expected_cases))
    histogram_path = figure_dir / "kplan_histogram.png"
    evidence.render_kplan_histogram(rows, histogram_path)
    extra_figure_renderer = getattr(evidence, "render_population_figures", None)
    extra_figures = (
        extra_figure_renderer(rows, figure_dir)
        if callable(extra_figure_renderer)
        else {}
    )
    protocol = {
        "split": str(args.split),
        "case_ids": case_ids,
        "query_count_requested": None if args.query_count is None else int(args.query_count),
        "query_batch_size": query_batch_size,
        "local_port_condition_mode": "predicted",
        "mixed_teacher_ratio": 0.0,
        "device": "cpu",
    }
    output = {
        "schema_version": 1,
        "task": str(
            getattr(args, "task_name", "run1409_occupancy_adaptive_population_evidence")
        ),
        "status": "complete",
        "output_dir": str(output_dir),
        "candidate": {
            "architecture": architecture,
            "checkpoint": str(checkpoint_path),
            "checkpoint_selection": "explicit_path_only",
            "dataset": str(resolved_dataset),
            "dataset_config_split": dataset_config.get("test_split", dataset_config.get("train_split")),
            "Kmax": kmax,
            "no_gpu": True,
            "checkpoint_written": False,
        },
        "protocol": protocol,
        "population": summary,
        "cases": rows,
        "arrays": array_paths,
        "figures": {
            "kplan_histogram": str(histogram_path),
            "case_boards": board_paths,
            **extra_figures,
        },
        "missing_evidence": [
            "Learned group labels are permutation ambiguous and are not physical causality claims.",
            "This population artifact does not embed matched Run-1404/1406/1804 accuracy and cost tables; those require a separate explicit checkpoint evaluation.",
        ]
        + (
            []
            if bool(summary["continuation_gate"]["actual_rows_available"])
            else [
                "Actual P2 executor rows are unavailable because the live phase ledger was not emitted by the checkpoint API."
            ]
        ),
    }
    _write_json(output_dir / "evidence.json", output)
    _write_json(output_dir / "population_summary.json", summary)
    _write_rows_csv(output_dir / "population_cases.csv", rows)
    progress_path.unlink(missing_ok=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Explicit checkpoint path")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default=None, help="Optional explicit HDF5 dataset path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=[], help="Optional case subset; repeat this option")
    parser.add_argument("--expected-cases", type=int, default=EXPECTED_CASE_COUNT)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--kmax", type=int, default=DEFAULT_KMAX)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    output = run_population(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "status": output["status"],
                "output_dir": output["output_dir"],
                "case_count": output["population"]["case_count"],
                "kplan_histogram": output["population"]["kplan_histogram"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_population"]
