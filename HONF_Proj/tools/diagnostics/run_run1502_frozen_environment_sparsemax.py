"""Run the bounded, frozen Run-1502 final-environment diagnostic.

This utility compares the historical Run-1501 final environmental entmax-1.5
assignment with the proposed sparsemax assignment on the existing saved-best
e444 and exact e500 checkpoints.  It uses a fixed eight-case development
panel, records logical support and route/output perturbations, and never
updates parameters or writes checkpoints.  The measurements are structural;
they are not a retrained-accuracy forecast.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Four fixed anchors plus four cases selected from the established physical
# descriptor strata before this candidate was evaluated.  Keep this order
# stable across the parent/candidate pair and both checkpoints.
DEFAULT_CASE_IDS = (
    "0273",
    "0653",
    "0644",
    "0686",
    "0277",
    "0291",
    "0680",
    "0281",
)
CASE_SELECTION_REASONS = {
    "0273": "fixed Run-1501 executor anchor",
    "0653": "fixed Run-1501 executor anchor",
    "0644": "fixed Run-1501 high-Kq formation case",
    "0686": "fixed Run-1501 low-Kq formation case",
    "0277": "preselected descriptor stratum: M3/intermediate spacing",
    "0291": "preselected descriptor stratum: M5/crowded interior",
    "0680": "preselected descriptor stratum: M10/crowded near-wall",
    "0281": "preselected descriptor stratum: M3/separated near-wall",
}

MATERIAL_REDUCTION_THRESHOLD = 0.05


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: 0.0 for key in ("mean", "median", "p95", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    label, separator, raw_path = str(spec).partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError("--checkpoint must use LABEL=PATH syntax")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return label.strip(), path


def _resolve_device(torch: Any, requested: str) -> tuple[Any, str]:
    """Resolve a logical device while enforcing physical GPU 2 only."""

    requested = str(requested)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
        if visible_ids != ["2"]:
            raise ValueError(
                "Run1502 frozen diagnostic requires CUDA_VISIBLE_DEVICES=2 when set; "
                f"got {visible!r}."
            )
        if requested not in {"cuda:0", "cuda"}:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES=2 remaps the physical GPU to logical cuda:0; "
                f"got {requested!r}."
            )
        device = torch.device("cuda:0")
    else:
        if requested not in {"cuda:2", "cuda"}:
            raise ValueError(
                "Run1502 frozen diagnostic is bounded to physical cuda:2; "
                f"got {requested!r}."
            )
        device = torch.device("cuda:2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the bounded Run1502 GPU-2 diagnostic.")
    device_index = 0 if visible else 2
    if torch.cuda.device_count() <= device_index:
        raise RuntimeError(
            f"requested physical GPU 2 is unavailable (visible device count={torch.cuda.device_count()})."
        )
    torch.cuda.set_device(device)
    physical = "2" if not visible else visible_ids[0]
    return device, physical


def _routing_array(prediction: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    for container_key in ("routing_maps", "interaction_aux"):
        container = prediction.get(container_key, {})
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            value = container.get(key)
            if value is not None:
                array = np.asarray(value)
                if array.ndim >= 3 and array.shape[0] == 1:
                    array = array[0]
                if array.ndim == 2:
                    return array.astype(np.float64, copy=False)
    raise RuntimeError(f"prediction did not expose any routing map from {tuple(keys)!r}")


def _support_metrics(
    query_assignment: np.ndarray,
    source_assignment: np.ndarray,
    active_source: np.ndarray,
) -> dict[str, float | int]:
    query = np.asarray(query_assignment) > 0.0
    source = np.asarray(source_assignment) > 0.0
    active = np.asarray(active_source, dtype=bool).reshape(-1)
    if query.ndim != 2 or source.ndim != 2:
        raise ValueError("query/source assignments must be two-dimensional")
    if query.shape[1] != source.shape[1]:
        raise ValueError("query/source assignments must share group width")
    if source.shape[0] != active.shape[0]:
        raise ValueError("active source mask must align with source assignment")
    reachable = query[:, None, :] & source[None, :, :]
    reachable = np.any(reachable, axis=-1)
    reachable[:, ~active] = False
    logical = int(np.sum(query[:, None, :] & source[None, :, :]))
    unique = int(np.sum(reachable))
    dense = int(query.shape[0] * np.sum(active))
    return {
        "logical_paths": logical,
        "unique_pairs": unique,
        "dense_pairs": dense,
        "support_ratio": float(unique / dense) if dense else 0.0,
        "logical_multiplicity": float(logical / unique) if unique else 0.0,
    }


def _prepared_controls(prediction: Mapping[str, Any]) -> Any:
    prepared_case = prediction.get("_prepared_state")
    prepared = getattr(prepared_case, "prepared", None)
    if prepared is None:
        raise RuntimeError("frozen diagnostic requires the retained prepared state")
    controls = prepared.backend_state.get("group_control_state")
    if controls is None:
        raise RuntimeError("prepared state did not expose sparse-incidence controls")
    return controls


def _case_metrics(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    parent_controls = _prepared_controls(parent)
    candidate_controls = _prepared_controls(candidate)
    parent_environment = parent_controls.environment_membership.detach().cpu().numpy()[0]
    candidate_environment = candidate_controls.environment_membership.detach().cpu().numpy()[0]
    parent_module = parent_controls.module_membership.detach().cpu().numpy()[0]
    candidate_module = candidate_controls.module_membership.detach().cpu().numpy()[0]
    parent_env_mass = parent_controls.environment_measure.detach().cpu().numpy()[0]
    candidate_env_mass = candidate_controls.environment_measure.detach().cpu().numpy()[0]
    parent_module_mass = parent_controls.module_measure.detach().cpu().numpy()[0]
    candidate_module_mass = candidate_controls.module_measure.detach().cpu().numpy()[0]
    parent_alpha = _routing_array(
        parent,
        ("sparse_incidence_query_routing", "group_control_query_routing"),
    )
    candidate_alpha = _routing_array(
        candidate,
        ("sparse_incidence_query_routing", "group_control_query_routing"),
    )
    parent_field = np.asarray(parent["pred_field_grid"], dtype=np.float64).reshape(-1)
    candidate_field = np.asarray(candidate["pred_field_grid"], dtype=np.float64).reshape(-1)
    field_delta = candidate_field - parent_field
    parent_norm = float(np.linalg.norm(parent_field))
    finite = bool(
        np.isfinite(parent_environment).all()
        and np.isfinite(candidate_environment).all()
        and np.isfinite(parent_alpha).all()
        and np.isfinite(candidate_alpha).all()
        and np.isfinite(parent_field).all()
        and np.isfinite(candidate_field).all()
    )
    parent_env_support = _support_metrics(parent_alpha, parent_environment, parent_env_mass > 0.0)
    candidate_env_support = _support_metrics(
        candidate_alpha, candidate_environment, candidate_env_mass > 0.0
    )
    parent_module_support = _support_metrics(parent_alpha, parent_module, parent_module_mass > 0.0)
    candidate_module_support = _support_metrics(
        candidate_alpha, candidate_module, candidate_module_mass > 0.0
    )
    parent_query_support = np.sum(parent_alpha > 0.0, axis=-1)
    candidate_query_support = np.sum(candidate_alpha > 0.0, axis=-1)
    parent_env_degree = np.sum(parent_environment > 0.0, axis=-1)
    candidate_env_degree = np.sum(candidate_environment > 0.0, axis=-1)
    parent_phase_empty = int(np.sum(~parent_controls.phase_occupied.detach().cpu().numpy()[0]))
    candidate_phase_empty = int(np.sum(~candidate_controls.phase_occupied.detach().cpu().numpy()[0]))
    parent_environment_empty_rows = int(np.sum(np.sum(parent_environment, axis=-1) <= 1.0e-10))
    candidate_environment_empty_rows = int(np.sum(np.sum(candidate_environment, axis=-1) <= 1.0e-10))
    parent_empty_queries = int(np.sum(np.sum(parent_alpha, axis=-1) <= 1.0e-10))
    candidate_empty_queries = int(np.sum(np.sum(candidate_alpha, axis=-1) <= 1.0e-10))
    candidate_singleton_environment_fraction = float(np.mean(candidate_env_degree == 1))
    candidate_singleton_query_fraction = float(np.mean(candidate_query_support == 1))
    return {
        "query_count": int(parent_alpha.shape[0]),
        "environment_count": int(parent_environment.shape[0]),
        "module_count": int(parent_module.shape[0]),
        "finite": finite,
        "parent": {
            "environment_degree": _summary(parent_env_degree),
            "query_degree": _summary(parent_query_support),
            "environment_support": parent_env_support,
            "module_support": parent_module_support,
            "empty_phase_groups": parent_phase_empty,
            "empty_environment_rows": parent_environment_empty_rows,
            "empty_query_rows": parent_empty_queries,
        },
        "candidate": {
            "environment_degree": _summary(candidate_env_degree),
            "query_degree": _summary(candidate_query_support),
            "environment_support": candidate_env_support,
            "module_support": candidate_module_support,
            "empty_phase_groups": candidate_phase_empty,
            "empty_environment_rows": candidate_environment_empty_rows,
            "empty_query_rows": candidate_empty_queries,
            "singleton_environment_fraction": candidate_singleton_environment_fraction,
            "singleton_query_fraction": candidate_singleton_query_fraction,
        },
        "perturbation": {
            "environment_assignment_l1_mean": float(
                np.mean(np.sum(np.abs(candidate_environment - parent_environment), axis=-1))
            ),
            "environment_assignment_max_abs": float(
                np.max(np.abs(candidate_environment - parent_environment))
            ),
            "query_assignment_l1_mean": float(
                np.mean(np.sum(np.abs(candidate_alpha - parent_alpha), axis=-1))
            ),
            "query_assignment_max_abs": float(np.max(np.abs(candidate_alpha - parent_alpha))),
            "output_relative_rms": float(
                np.linalg.norm(field_delta) / max(parent_norm, 1.0e-30)
            ),
            "output_max_abs": float(np.max(np.abs(field_delta))),
        },
    }


def _summarize_checkpoint(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parent_re = [float(row["parent"]["environment_support"]["support_ratio"]) for row in rows]
    candidate_re = [float(row["candidate"]["environment_support"]["support_ratio"]) for row in rows]
    parent_rm = [float(row["parent"]["module_support"]["support_ratio"]) for row in rows]
    candidate_rm = [float(row["candidate"]["module_support"]["support_ratio"]) for row in rows]
    parent_kq = [float(row["parent"]["query_degree"]["mean"]) for row in rows]
    candidate_kq = [float(row["candidate"]["query_degree"]["mean"]) for row in rows]
    re_parent = float(np.mean(parent_re)) if parent_re else 0.0
    re_candidate = float(np.mean(candidate_re)) if candidate_re else 0.0
    reduction = (re_parent - re_candidate) / max(abs(re_parent), 1.0e-30)
    pathology = {
        "nonfinite_cases": int(sum(not bool(row["finite"]) for row in rows)),
        "candidate_empty_phase_groups": int(
            sum(int(row["candidate"]["empty_phase_groups"]) for row in rows)
        ),
        "candidate_empty_environment_rows": int(
            sum(int(row["candidate"]["empty_environment_rows"]) for row in rows)
        ),
        "candidate_empty_query_rows": int(
            sum(int(row["candidate"]["empty_query_rows"]) for row in rows)
        ),
    }
    return {
        "case_count": len(rows),
        "environment_support_ratio": {
            "parent": _summary(parent_re),
            "candidate": _summary(candidate_re),
            "relative_reduction": float(reduction),
        },
        "module_support_ratio": {"parent": _summary(parent_rm), "candidate": _summary(candidate_rm)},
        "query_degree": {"parent": _summary(parent_kq), "candidate": _summary(candidate_kq)},
        "output_relative_rms": _summary(
            [float(row["perturbation"]["output_relative_rms"]) for row in rows]
        ),
        "environment_assignment_l1": _summary(
            [float(row["perturbation"]["environment_assignment_l1_mean"]) for row in rows]
        ),
        "query_assignment_l1": _summary(
            [float(row["perturbation"]["query_assignment_l1_mean"]) for row in rows]
        ),
        "pathology": pathology,
        "material_environment_overlap_reduction": bool(
            reduction >= MATERIAL_REDUCTION_THRESHOLD
        ),
        "no_obvious_support_failure": bool(
            pathology["nonfinite_cases"] == 0
            and pathology["candidate_empty_phase_groups"] == 0
            and pathology["candidate_empty_environment_rows"] == 0
            and pathology["candidate_empty_query_rows"] == 0
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model
    from channelthermal.evaluation.prepared import predict_case
    from run_run1409_occupancy_population import _dataset_for_checkpoint, _query_sample

    device, physical_gpu = _resolve_device(torch, args.device)
    checkpoint_specs = [_parse_checkpoint_spec(spec) for spec in args.checkpoint]
    if len(checkpoint_specs) != 2:
        raise ValueError("the bounded diagnostic requires exactly two checkpoints (e444 and e500)")
    case_ids = tuple(str(value) for value in (args.case_id or DEFAULT_CASE_IDS))
    if case_ids != DEFAULT_CASE_IDS and len(case_ids) != len(DEFAULT_CASE_IDS):
        raise ValueError("custom diagnostic case sets must contain exactly eight cases")
    all_rows: dict[str, list[dict[str, Any]]] = {}
    checkpoint_payloads: dict[str, Any] = {}
    dataset_path_by_label: dict[str, str] = {}
    for label, checkpoint_path in checkpoint_specs:
        model, checkpoint = load_model(checkpoint_path, device)
        if str(model.config.core_honf.forward_architecture) != "sparse_incidence_group_control_honf":
            raise RuntimeError(
                f"{label} checkpoint architecture is not sparse incidence: "
                f"{model.config.core_honf.forward_architecture!r}"
            )
        dataset, dataset_path, _ = _dataset_for_checkpoint(
            checkpoint,
            dataset_path=args.dataset,
            split=args.split,
            GlobalChannelThermalDataset=GlobalChannelThermalDataset,
            H5Normalizer=H5Normalizer,
        )
        available = {str(value): index for index, value in enumerate(dataset.selected_case_ids)}
        missing = [case_id for case_id in case_ids if case_id not in available]
        if missing:
            raise KeyError(f"case IDs are absent from split {args.split!r}: {missing}")
        router = model.core.backend.router
        router.environment_refinement_normalizer = "entmax15"
        rows: list[dict[str, Any]] = []
        for order, case_id in enumerate(case_ids):
            sample = dataset[available[case_id]]
            selected, _query_xy = _query_sample(sample, int(args.query_count))
            parent = predict_case(
                model,
                selected,
                device,
                query_batch_size=int(args.query_batch_size),
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
                return_routing_maps=True,
                return_prepared_state=True,
            )
            router.environment_refinement_normalizer = "sparsemax"
            candidate = predict_case(
                model,
                selected,
                device,
                query_batch_size=int(args.query_batch_size),
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.0,
                return_routing_maps=True,
                return_prepared_state=True,
            )
            router.environment_refinement_normalizer = "entmax15"
            row = {
                "case_id": case_id,
                "selection_reason": CASE_SELECTION_REASONS.get(case_id, "explicit custom case"),
                "checkpoint_label": label,
                "checkpoint": str(checkpoint_path),
                "metrics": _case_metrics(parent, candidate),
            }
            rows.append(row)
            print(
                f"[run1502-frozen-env] {label} {order + 1}/{len(case_ids)} "
                f"case={case_id} "
                f"RE={row['metrics']['candidate']['environment_support']['support_ratio']:.4f} "
                f"Kq={row['metrics']['candidate']['query_degree']['mean']:.3f}",
                flush=True,
            )
            del parent, candidate, selected, sample
        checkpoint_summary = _summarize_checkpoint([row["metrics"] for row in rows])
        all_rows[label] = rows
        checkpoint_payloads[label] = {
            "checkpoint": str(checkpoint_path),
            "summary": checkpoint_summary,
        }
        dataset_path_by_label[label] = str(dataset_path)
        del model, checkpoint, dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checkpoint_summaries = {
        label: payload["summary"] for label, payload in checkpoint_payloads.items()
    }
    launch_recommended = bool(
        all(
            summary["material_environment_overlap_reduction"]
            and summary["no_obvious_support_failure"]
            for summary in checkpoint_summaries.values()
        )
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_flat = [row for rows in all_rows.values() for row in rows]
    _write_json(output_dir / "cases.json", rows_flat)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "task": "run1502_frozen_environment_sparsemax_diagnostic",
            "status": "launch_recommended" if launch_recommended else "stop_before_training",
            "recommendation": (
                "The candidate reduces environmental overlap materially on both frozen checkpoints "
                "without observed support pathology; main agent may consider one managed Run1502 launch."
                if launch_recommended
                else "Do not launch Run1502 from this diagnostic alone: overlap reduction was not material on every checkpoint or support pathology was observed."
            ),
            "checkpoints": checkpoint_payloads,
            "dataset_by_checkpoint": dataset_path_by_label,
            "split": str(args.split),
            "case_ids": list(case_ids),
            "case_selection_reasons": {
                case_id: CASE_SELECTION_REASONS.get(case_id, "explicit custom case")
                for case_id in case_ids
            },
            "query_count": int(args.query_count),
            "query_batch_size": int(args.query_batch_size),
            "device": str(device),
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "parent_normalizer": "entmax15",
            "candidate_normalizer": "masked_sparsemax",
            "proposal_environment_normalizer": "entmax15",
            "module_proposal_normalizer": "entmax15",
            "module_refinement_normalizer": "entmax15",
            "query_normalizer": "masked_sparsemax",
            "material_reduction_threshold_relative": MATERIAL_REDUCTION_THRESHOLD,
            "managed_run_allocated": False,
            "training_launched": False,
            "checkpoint_written": False,
            "interpretation_limit": (
                "Frozen structural evidence only: routing/output perturbations do not predict retrained accuracy, "
                "and learned support is not physical causality or executor work."
            ),
        },
    )
    fields = [
        "checkpoint_label",
        "checkpoint",
        "case_id",
        "selection_reason",
        "parent_RE",
        "candidate_RE",
        "parent_RM",
        "candidate_RM",
        "parent_Kq",
        "candidate_Kq",
        "environment_degree_parent",
        "environment_degree_candidate",
        "environment_assignment_l1_mean",
        "query_assignment_l1_mean",
        "output_relative_rms",
        "candidate_singleton_environment_fraction",
        "candidate_singleton_query_fraction",
        "candidate_empty_phase_groups",
        "candidate_empty_environment_rows",
        "candidate_empty_query_rows",
        "finite",
    ]
    with (output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows_flat:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "checkpoint_label": row["checkpoint_label"],
                    "checkpoint": row["checkpoint"],
                    "case_id": row["case_id"],
                    "selection_reason": row["selection_reason"],
                    "parent_RE": metrics["parent"]["environment_support"]["support_ratio"],
                    "candidate_RE": metrics["candidate"]["environment_support"]["support_ratio"],
                    "parent_RM": metrics["parent"]["module_support"]["support_ratio"],
                    "candidate_RM": metrics["candidate"]["module_support"]["support_ratio"],
                    "parent_Kq": metrics["parent"]["query_degree"]["mean"],
                    "candidate_Kq": metrics["candidate"]["query_degree"]["mean"],
                    "environment_degree_parent": metrics["parent"]["environment_degree"]["mean"],
                    "environment_degree_candidate": metrics["candidate"]["environment_degree"]["mean"],
                    "environment_assignment_l1_mean": metrics["perturbation"]["environment_assignment_l1_mean"],
                    "query_assignment_l1_mean": metrics["perturbation"]["query_assignment_l1_mean"],
                    "output_relative_rms": metrics["perturbation"]["output_relative_rms"],
                    "candidate_singleton_environment_fraction": metrics["candidate"].get("singleton_environment_fraction", 0.0),
                    "candidate_singleton_query_fraction": metrics["candidate"].get("singleton_query_fraction", 0.0),
                    "candidate_empty_phase_groups": metrics["candidate"]["empty_phase_groups"],
                    "candidate_empty_environment_rows": metrics["candidate"]["empty_environment_rows"],
                    "candidate_empty_query_rows": metrics["candidate"]["empty_query_rows"],
                    "finite": metrics["finite"],
                }
            )
    return {
        "status": "launch_recommended" if launch_recommended else "stop_before_training",
        "output_dir": str(output_dir),
        "checkpoint_summaries": checkpoint_summaries,
        "case_ids": list(case_ids),
        "physical_gpu": physical_gpu,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeat exactly twice for e444 and e500 checkpoints",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=1024)
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:2")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0 if result["status"] == "launch_recommended" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CASE_SELECTION_REASONS",
    "DEFAULT_CASE_IDS",
    "build_parser",
    "main",
    "run",
]
