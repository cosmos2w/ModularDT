"""Population capacity/rank audit for a Run-1409 checkpoint.

The audit loads one candidate checkpoint, switches to the deployed
deterministic ``model.eval()`` policy, and runs one Q=1024 probe set for each
case in the established 90-case test population.  It retains compact scalar
rows only.  Query routing ``alpha`` and source incidence ``A`` are used
transiently for the existing thin-QR rank helper; no Q-by-S matrix or map file
is constructed or written.

Run from ``HONF_Proj/`` after any headline full-grid comparison has released
the GPU::

    PYTHONPATH=src:Case_ThermalChannel/src \
    python tools/diagnostics/run_run1409_population_capacity_rank.py \
      --checkpoint /path/to/Run_1409/epoch_0050_model.pt \
      --output /path/to/Run_1409/evaluations/population_capacity_rank_q1024.json \
      --csv /path/to/Run_1409/evaluations/population_capacity_rank_q1024.csv \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CASE_COUNT = 90
DEFAULT_QUERY_COUNT = 1024
DEFAULT_RECEIVER_CHUNK_SIZE = 2048
ARCHITECTURE = "budgeted_group_control_honf"
GROUP_COUNT = 12
CONTROL_DIM = 16
REVIEW_EPOCHS = (50, 150, 500)


def _runtime_imports() -> tuple[Any, ...]:
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch

    import run_run1409_budgeted_evidence as evidence
    import run_run1405_epoch50_comparison as run1405
    import run_stage3_interface_study as stage3
    from channelthermal.evaluation.loading import make_batch
    from channelthermal.evaluation.prepared import select_sample

    return torch, evidence, run1405, stage3, make_batch, select_sample


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "numel"):
        value = value.detach().cpu()
        if int(value.numel()) == 1:
            return _jsonable(value.item())
        if int(value.numel()) <= 256:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["case_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _scalar(evidence: Any, value: Any) -> float | None:
    return evidence._scalar(value)


def _array(evidence: Any, value: Any) -> np.ndarray | None:
    return evidence._array(value)


def _source_nonempty(evidence: Any, aux: Mapping[str, Any], source: str) -> int | None:
    value = evidence._find_aux(
        aux,
        (f"group_control_{source}_incidence", f"group_control_{source}_membership"),
    )
    array = _array(evidence, value)
    if array is None:
        return None
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        return None
    return int(np.count_nonzero(np.any(array[0] > 0.0, axis=0)))


def _query_reached(evidence: Any, aux: Mapping[str, Any]) -> int | None:
    value = evidence._find_aux(aux, ("group_control_query_routing", "group_control_query_alpha"))
    array = _array(evidence, value)
    if array is None:
        return None
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        return None
    return int(np.count_nonzero(np.any(array[0] > 0.0, axis=0)))


def _budget_attr(budget: Any, names: Sequence[str]) -> Any:
    if budget is None:
        return None
    for name in names:
        value = getattr(budget, name, None)
        if value is not None:
            return value
    return None


def _phase_scalars(ledger: Mapping[str, Any]) -> dict[str, Any]:
    metrics = (
        "logical_path_count",
        "unique_pair_count",
        "logical_support_rows",
        "actual_fine_call_count",
        "padded_fine_rows",
        "valid_pair_denominator",
        "padded_pair_denominator",
        "geometry_rows",
        "content_dot_rows",
        "scalar_control_rows",
        "source_projection_rows",
        "checkpoint_recompute_count",
    )
    result: dict[str, Any] = {}
    for phase, phase_value in ledger.items():
        if not isinstance(phase_value, Mapping):
            continue
        sources = phase_value.get("sources", {})
        if not isinstance(sources, Mapping):
            continue
        for source, source_value in sources.items():
            if not isinstance(source_value, Mapping):
                continue
            for metric in metrics:
                result[f"{phase}_{source}_{metric}"] = source_value.get(metric)
    return result


def _capacity_scalars(evidence: Any, output: Any) -> dict[str, Any]:
    budget = evidence._lookup(output, ("case_group_budget",))
    raw_z = _budget_attr(budget, ("raw_z", "z_raw", "sampled_z", "gate_z_raw"))
    effective_z = _budget_attr(budget, ("effective_z", "z_eff", "z"))
    raw_support = _budget_attr(budget, ("raw_support", "support_raw", "gate_support_raw"))
    executed_support = _budget_attr(
        budget,
        ("executed_support", "effective_support", "support", "gate_support"),
    )
    expected_count = _budget_attr(
        budget,
        ("expected_count", "expected_live_count", "expected_group_count"),
    )
    expected_optional = _budget_attr(budget, ("expected_optional_count",))
    probability = _budget_attr(
        budget,
        ("positive_probability", "probability", "gate_probability", "p"),
    )
    packed_width = _budget_attr(budget, ("packed_width", "compact_width"))
    packed_ids = _budget_attr(budget, ("packed_ids", "prototype_ids", "compact_ids"))
    live_count = _budget_attr(budget, ("live_count", "Klive", "k_live"))
    fallback_used = _budget_attr(
        budget,
        ("fallback_used", "fallback", "fallback_active", "used_fallback"),
    )
    continuation = _budget_attr(budget, ("continuation", "continuation_c", "sparsification_c"))
    routing_strength = _budget_attr(budget, ("routing_strength", "route_strength", "r"))
    aliases = {
        "raw_z": ("case_group_budget_raw_z", "raw_z", "z_raw"),
        "effective_z": ("case_group_budget_effective_z", "effective_z", "z_eff", "z"),
        "raw_support": ("case_group_budget_raw_support", "raw_support", "support_raw"),
        "executed_support": (
            "case_group_budget_executed_support",
            "effective_support",
            "case_group_budget_support",
            "support",
        ),
        "expected_count": (
            "case_group_budget_expected_count",
            "expected_live_count",
            "expected_group_count",
        ),
        "expected_optional": (
            "case_group_budget_expected_optional_count",
            "case_group_budget_expected_optional_count_detached",
            "expected_optional_count",
        ),
        "probability": (
            "case_group_budget_positive_probability",
            "case_group_budget_gate_positive_probability",
            "positive_probability",
            "gate_probability",
        ),
        "packed_width": ("case_group_budget_packed_width", "packed_width", "compact_width"),
        "packed_ids": (
            "case_group_budget_packed_prototype_ids",
            "case_group_budget_prototype_ids",
            "packed_ids",
            "prototype_ids",
        ),
        "live_count": ("case_group_budget_live_count", "case_group_budget_sampled_live_count", "live_count"),
        "fallback": (
            "case_group_budget_fallback_used",
            "case_group_budget_fallback",
            "fallback_used",
            "fallback",
        ),
        "continuation": ("case_group_budget_continuation", "continuation_c", "sparsification_c"),
        "routing_strength": ("case_group_budget_routing_strength", "routing_strength", "route_strength"),
    }
    values = {
        "raw_z": raw_z,
        "effective_z": effective_z,
        "raw_support": raw_support,
        "executed_support": executed_support,
        "expected_count": expected_count,
        "expected_optional": expected_optional,
        "probability": probability,
        "packed_width": packed_width,
        "packed_ids": packed_ids,
        "live_count": live_count,
        "fallback": fallback_used,
        "continuation": continuation,
        "routing_strength": routing_strength,
    }
    for name, aliases_for_name in aliases.items():
        if values[name] is None:
            values[name] = evidence._lookup(output, aliases_for_name)
    raw_z_array = _array(evidence, values["raw_z"])
    effective_z_array = _array(evidence, values["effective_z"])
    raw_support_array = _array(evidence, values["raw_support"])
    executed_support_array = _array(evidence, values["executed_support"])
    if raw_z_array is not None and raw_z_array.ndim == 1:
        raw_z_array = raw_z_array[None, ...]
    if effective_z_array is not None and effective_z_array.ndim == 1:
        effective_z_array = effective_z_array[None, ...]
    if raw_support_array is None and raw_z_array is not None:
        raw_support_array = raw_z_array > 0.0
    if executed_support_array is None and effective_z_array is not None:
        executed_support_array = effective_z_array > 0.0
    if raw_support_array is not None and raw_support_array.ndim == 1:
        raw_support_array = raw_support_array[None, ...]
    if executed_support_array is not None and executed_support_array.ndim == 1:
        executed_support_array = executed_support_array[None, ...]
    k_raw = None if raw_support_array is None else int(np.count_nonzero(raw_support_array[0]))
    k_executed = (
        None
        if executed_support_array is None
        else int(np.count_nonzero(executed_support_array[0]))
    )
    expected_value = _scalar(evidence, values["expected_count"])
    optional_value = _scalar(evidence, values["expected_optional"])
    if expected_value is None and optional_value is not None:
        expected_value = 1.0 + optional_value
    if expected_value is None:
        probability_array = _array(evidence, values["probability"])
        if probability_array is not None:
            probability_array = np.asarray(probability_array, dtype=np.float64)
            if probability_array.ndim == 1:
                probability_array = probability_array[None, ...]
            expected_value = float(np.sum(probability_array[0]))
    live_value = _scalar(evidence, values["live_count"])
    if live_value is None:
        live_value = k_executed
    packed_width_value = _scalar(evidence, values["packed_width"])
    packed_ids_array = _array(evidence, values["packed_ids"])
    if packed_width_value is None and packed_ids_array is not None and packed_ids_array.ndim >= 2:
        packed_width_value = float(packed_ids_array.shape[1])
    fallback_value = _scalar(evidence, values["fallback"])
    if fallback_value is not None:
        fallback_value = bool(fallback_value)
    prototype_ids = None
    if packed_ids_array is not None:
        packed_ids_array = np.asarray(packed_ids_array)
        if packed_ids_array.ndim == 1:
            prototype_ids = packed_ids_array.astype(int).tolist()
        elif packed_ids_array.ndim >= 2:
            prototype_ids = packed_ids_array[0].astype(int).tolist()
    return {
        "Kmax": GROUP_COUNT,
        "K_raw": k_raw,
        "K_executed": k_executed,
        "K_expected": expected_value,
        "K_expected_optional": optional_value,
        "Klive": None if live_value is None else int(round(live_value)),
        "Kpack": None if packed_width_value is None else int(round(packed_width_value)),
        "fallback_used": fallback_value,
        "prototype_ids": prototype_ids,
        "continuation_c": _scalar(evidence, values["continuation"]),
        "routing_strength": _scalar(evidence, values["routing_strength"]),
        "deterministic_gate_positive_fraction": None
        if effective_z_array is None
        else float(np.mean(effective_z_array[0] > 0.0)),
    }


def _one_case(
    *,
    model: Any,
    dataset: Any,
    case_id: str,
    query_count: int,
    receiver_chunk_size: int,
    torch: Any,
    evidence: Any,
    run1405: Any,
    stage3: Any,
    make_batch: Any,
    select_sample: Any,
    device: Any,
) -> dict[str, Any]:
    sample = select_sample(dataset, str(case_id), 0)
    query_np = evidence.stage3_query_points(sample, int(query_count))
    if len(query_np) != int(query_count):
        raise ValueError(f"case {case_id} produced Q={len(query_np)}, expected {query_count}")
    batch = make_batch(dict(sample), query_np, device)
    kwargs = run1405._phase_forward_kwargs(batch)
    with stage3._runtime_receiver_chunk_size(model, int(receiver_chunk_size)), torch.inference_mode():
        output = model(
            batch["structure"],
            batch["query_xy"],
            return_prepared_state=True,
            return_routing_maps=True,
            **kwargs,
        )
    aux = evidence._aux_mapping(output)
    budget = _capacity_scalars(evidence, output)
    ledger = evidence._phase_ledger(aux, getattr(model.core, "backend", None))
    rank = evidence._rank_summary(aux)
    row: dict[str, Any] = {
        "case_id": str(case_id),
        "query_count": int(query_count),
        **budget,
        "module_source_nonempty_groups": _source_nonempty(evidence, aux, "module"),
        "environment_source_nonempty_groups": _source_nonempty(evidence, aux, "environment"),
        "query_reached_groups": _query_reached(evidence, aux),
        "module_rank_status": rank.get("module", {}).get("status"),
        "module_rank_mean": rank.get("module", {}).get("rank_mean"),
        "module_rank_max": rank.get("module", {}).get("rank_max"),
        "module_entropy_effective_rank_mean": rank.get("module", {}).get("entropy_effective_rank_mean"),
        "environment_rank_status": rank.get("environment", {}).get("status"),
        "environment_rank_mean": rank.get("environment", {}).get("rank_mean"),
        "environment_rank_max": rank.get("environment", {}).get("rank_max"),
        "environment_entropy_effective_rank_mean": rank.get("environment", {}).get("entropy_effective_rank_mean"),
        "executor_policy": getattr(model.core.backend, "executor_policy", None),
        "rectangular_ledger_rows": getattr(model.core.backend, "ledger_rectangular_rows", None),
        "gate_policy": "model.eval() deployed deterministic hard-concrete estimate",
    }
    row.update(_phase_scalars(ledger))
    del output, aux, batch
    gc.collect()
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    return row


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    torch, evidence, run1405, stage3, make_batch, select_sample = _runtime_imports()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")
        torch.cuda.set_device(device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    spec = stage3.CheckpointSpec(label="run1409", path=checkpoint_path)
    model, checkpoint = stage3._load_model_spec(spec, device)
    checkpoint_epoch = int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1)))
    if checkpoint_epoch not in REVIEW_EPOCHS:
        raise ValueError(
            f"population capacity audit accepts exact review checkpoints {REVIEW_EPOCHS}; "
            f"got epoch={checkpoint_epoch}"
        )
    architecture = str(model.config.core_honf.forward_architecture)
    if architecture != ARCHITECTURE:
        raise ValueError(f"expected {ARCHITECTURE}, got {architecture!r}")
    interface = model.config.core_honf.interface_model
    if int(interface.group_count) != GROUP_COUNT or int(interface.group_control_dim) != CONTROL_DIM:
        raise ValueError(
            f"expected Kmax={GROUP_COUNT}, D={CONTROL_DIM}; "
            f"got K={interface.group_count}, D={interface.group_control_dim}"
        )
    dataset, dataset_path = stage3._load_dataset(
        checkpoint,
        argparse.Namespace(dataset=args.dataset, split=args.split),
    )
    case_ids = [str(value) for value in (args.case_id or dataset.selected_case_ids)]
    if not args.case_id and len(case_ids) != EXPECTED_CASE_COUNT:
        raise ValueError(f"expected {EXPECTED_CASE_COUNT} test cases, got {len(case_ids)}")
    model.eval()
    evidence._set_mode_once(model, "full_width")
    rows: list[dict[str, Any]] = []
    try:
        for index, case_id in enumerate(case_ids, start=1):
            rows.append(
                _one_case(
                    model=model,
                    dataset=dataset,
                    case_id=case_id,
                    query_count=int(args.query_count),
                    receiver_chunk_size=int(args.receiver_chunk_size),
                    torch=torch,
                    evidence=evidence,
                    run1405=run1405,
                    stage3=stage3,
                    make_batch=make_batch,
                    select_sample=select_sample,
                    device=device,
                )
            )
            if index % 10 == 0:
                print(json.dumps({"completed_cases": index, "total_cases": len(case_ids)}), flush=True)
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        gc.collect()
    return {
        "schema_version": 1,
        "task": "run1409_population_capacity_rank_q1024",
        "status": "complete",
        "candidate": {
            "architecture": architecture,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "dataset": str(dataset_path),
            "split": str(args.split),
            "case_count": len(rows),
            "group_capacity": GROUP_COUNT,
            "control_dimension": CONTROL_DIM,
        },
        "protocol": {
            "one_checkpoint_load": True,
            "model_mode": "eval",
            "gate_policy": "deployed deterministic hard-concrete estimate",
            "query_count": int(args.query_count),
            "receiver_chunk_size": int(args.receiver_chunk_size),
            "execution_mode": "full_width",
            "maps": "transient q-routing/source-incidence arrays only; no Q-by-S matrix saved",
        },
        "rows": rows,
        "definitions": {
            "K_raw": "count of raw hard-concrete values z_tilde_k > 0 before continuation",
            "K_executed": "count of effective continuation gates z_eff_k > 0",
            "K_expected": "sum of positive gate probabilities, or 1 + optional count on the legacy gate object",
            "Klive": "runtime live-count field when exposed, otherwise K_executed",
            "Kpack": "packed positive-column width for this B=1 evaluation",
            "fallback_used": "symmetric all-closed numerical fallback flag; null if the backend does not expose it",
            "prototype_ids": "original packed prototype IDs, retained without relabeling",
            "source_nonempty_groups": "groups with positive source incidence, separate for M and E",
            "routing_rank": "weighted learned q-to-source routing rank from thin-QR core; not physical operator rank",
            "fine_rows": "backend-reported logical/unique/actual/padded rows when exposed by the phase ledger",
        },
        "limitations": [
            "Q=1024 probes are a capacity/rank audit and do not replace the full-grid accuracy comparison.",
            "The rank is a numerical rank of learned routing arrays, not optimal physical rank or a causality claim.",
            "Unavailable backend ledger fields remain null rather than being reconstructed from support maps.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", dest="csv_output", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--receiver-chunk-size", type=int, default=DEFAULT_RECEIVER_CHUNK_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.query_count) <= 0 or int(args.receiver_chunk_size) <= 0:
        raise ValueError("query-count and receiver-chunk-size must be positive")
    payload = run_audit(args)
    _write_json(args.output, payload)
    csv_path = args.csv_output or Path(args.output).with_suffix(".csv")
    _write_csv(csv_path, payload["rows"])
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(Path(args.output).expanduser().resolve()),
                "csv": str(Path(csv_path).expanduser().resolve()),
                "cases": len(payload["rows"]),
                "query_count": int(args.query_count),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_audit"]
