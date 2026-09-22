"""Historical v1 pre-training checks for the Run-1409 budgeted reader.

This diagnostic deliberately performs no optimizer step and writes no model
state.  It builds one fresh profile model, materializes it with the ordinary
ThermalChannel data path, and compares ``full_width`` and ``compact`` on the
same model instance with the same fixed P0 gate noise.  The controller shape
check is independent and uses supplied noise for K=6, 12, and 32.

Run from ``HONF_Proj/``.  The output is a small JSON evidence record suitable
for the managed evaluation directory::

    PYTHONPATH=src:Case_ThermalChannel/src \
    python tools/diagnostics/run_run1409_stage1_shapes.py \
      --output /abs/path/to/evaluations/stage1_shapes.json \
      --device cuda:0

The two default shapes are test cases 0273 at Q=1024 and 0653 at Q=8192.
They are representative controller/receiver sizes, not a performance sweep.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHAPES = ("0273=1024", "0653=8192")
CONTROLLER_GROUP_COUNTS = (6, 12, 32)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "numel"):
        value = value.detach().cpu()
        if int(value.numel()) == 1:
            return _jsonable(value.item())
        if int(value.numel()) <= 256:
            return value.tolist()
        flat = value.reshape(-1).float()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_imports() -> tuple[Any, ...]:
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch

    import run_dynamic_sparse_routing_study as dynamic
    import run_run1405_epoch50_comparison as run1405
    from channelthermal.evaluation.loading import make_batch
    from channelthermal.evaluation.prepared import select_sample
    from honf_forward_core.interface_fields.case_group_budget import CaseGroupGate

    return torch, dynamic, run1405, make_batch, select_sample, CaseGroupGate


def _parse_shape(spec: str) -> tuple[str, int]:
    text = str(spec)
    if "=" not in text:
        raise ValueError(f"shape must use CASE=QUERY_COUNT syntax, got {spec!r}")
    case_id, query_count = text.split("=", 1)
    case_id = case_id.strip()
    if not case_id:
        raise ValueError(f"shape has an empty case id: {spec!r}")
    count = int(query_count)
    if count <= 0:
        raise ValueError(f"shape query count must be positive: {spec!r}")
    return case_id, count


@contextlib.contextmanager
def _fixed_gate_noise(model: Any, noise: Any) -> Iterator[None]:
    """Inject one fixed gate-noise tensor into P0 only."""

    backend = getattr(getattr(model, "core", None), "backend", None)
    router = getattr(backend, "router", None)
    original = getattr(router, "prepare", None)
    if router is None or not callable(original):
        raise RuntimeError("budgeted backend does not expose router.prepare")

    def prepare_with_fixed_noise(*args: Any, **kwargs: Any) -> Any:
        # A later phase passes budget=... and must reuse the P0 plan.  Only the
        # first router call receives the supplied noise.
        if kwargs.get("budget") is None:
            module_states = args[1] if len(args) > 1 else kwargs.get("module_states")
            if module_states is None:
                raise RuntimeError("cannot identify P0 module states for fixed gate noise")
            kwargs["gate_noise"] = noise.to(device=module_states.device, dtype=module_states.dtype)
            kwargs["deterministic_gates"] = False
        return original(*args, **kwargs)

    had_instance = hasattr(router, "__dict__") and "prepare" in router.__dict__
    old_instance = router.__dict__.get("prepare") if had_instance else None
    setattr(router, "prepare", prepare_with_fixed_noise)
    try:
        yield
    finally:
        if had_instance:
            setattr(router, "prepare", old_instance)
        else:
            try:
                delattr(router, "prepare")
            except AttributeError:
                pass


def _set_execution_mode(model: Any, mode: str) -> None:
    backend = getattr(getattr(model, "core", None), "backend", None)
    setter = getattr(backend, "set_execution_mode", None)
    if callable(setter):
        setter(str(mode))
        return
    if hasattr(backend, "execution_mode"):
        backend.execution_mode = str(mode)
        return
    raise RuntimeError("budgeted backend does not expose execution mode")


def _gate_summary(output: Mapping[str, Any]) -> dict[str, Any]:
    prepared = output.get("prepared_state")
    plan = getattr(prepared, "phase_shared_state", None)
    # The coupling exposes the shared CaseGroupBudget directly as
    # ``phase_shared_state``.  Older wrappers may nest it under a backend
    # state, so accept both layouts without changing the evidence contract.
    budget = getattr(plan, "case_group_budget", None)
    if budget is None and hasattr(plan, "z") and hasattr(plan, "support"):
        budget = plan
    if budget is None:
        raise RuntimeError("prepared output did not expose the case group budget")
    values = budget.z.detach().cpu()
    support = budget.support.detach().cpu()
    return {
        "capacity": int(values.shape[1]),
        "live_count": support.sum(dim=-1).tolist(),
        "packed_width": int(budget.packed_width),
        "packed_ids": budget.packed_ids.detach().cpu().tolist(),
        "gate_values": values.tolist(),
        "gate_support": support.tolist(),
        "deterministic": bool(getattr(budget, "deterministic", False)),
    }


def _field_difference(first: Any, second: Any) -> dict[str, float]:
    delta = (first.detach().float() - second.detach().float()).abs()
    denom = max(float(first.detach().float().abs().max().cpu()), 1.0e-12)
    return {
        "max_abs": float(delta.max().cpu()),
        "mean_abs": float(delta.mean().cpu()),
        "relative_max_to_full_max": float(delta.max().cpu()) / denom,
    }


def _synchronize(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min_ms": float(np.min(array)),
        "median_ms": float(np.median(array)),
        "mean_ms": float(np.mean(array)),
        "max_ms": float(np.max(array)),
    }


def _measure_mode(
    model: Any,
    batch: Mapping[str, Any],
    query: Any,
    kwargs: Mapping[str, Any],
    *,
    mode: str,
    fixed_noise: Any,
    torch: Any,
    device: Any,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Measure one mode after warmup, with a fixed P0 gate plan per call."""

    if int(warmups) < 0 or int(repetitions) <= 0:
        raise ValueError("warmups must be nonnegative and repetitions must be positive")
    _set_execution_mode(model, mode)

    def forward_once() -> Any:
        return model(
            batch["structure"],
            query,
            return_prepared_state=False,
            return_routing_maps=False,
            **kwargs,
        )

    with _fixed_gate_noise(model, fixed_noise), torch.inference_mode():
        for _ in range(int(warmups)):
            output = forward_once()
            del output
        _synchronize(torch, device)
        cuda = getattr(device, "type", None) == "cuda"
        if cuda:
            current_allocated = int(torch.cuda.memory_allocated(device))
            current_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            current_allocated = None
            current_reserved = None
        elapsed_ms: list[float] = []
        for _ in range(int(repetitions)):
            if cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = forward_once()
                end.record()
                end.synchronize()
                elapsed_ms.append(float(start.elapsed_time(end)))
            else:
                start_time = time.perf_counter()
                output = forward_once()
                elapsed_ms.append(float((time.perf_counter() - start_time) * 1000.0))
            del output
        _synchronize(torch, device)
        if cuda:
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = None
            peak_reserved = None
    row: dict[str, Any] = {
        "mode": str(mode),
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "latency": _summary(elapsed_ms),
        "memory_status": "cuda_peak_stats" if cuda else "unavailable_on_cpu",
        "current_allocated_bytes_before_timing": current_allocated,
        "current_reserved_bytes_before_timing": current_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": None
        if peak_allocated is None or current_allocated is None
        else int(max(0, peak_allocated - current_allocated)),
        "peak_reserved_delta_bytes": None
        if peak_reserved is None or current_reserved is None
        else int(max(0, peak_reserved - current_reserved)),
    }
    return row


def _query_points(sample: Mapping[str, Any], query_count: int, model: Any) -> np.ndarray:
    """Use the maintained grid when present, otherwise form a domain grid."""

    if "x_grid" in sample and "y_grid" in sample:
        x = np.asarray(sample["x_grid"], dtype=np.float32).reshape(-1)
        y = np.asarray(sample["y_grid"], dtype=np.float32).reshape(-1)
        points = np.stack([x, y], axis=-1)
        if len(points) < int(query_count):
            raise ValueError(f"sample grid has only {len(points)} points; requested {query_count}")
        return points[: int(query_count)].astype(np.float32, copy=False)
    side = int(math.ceil(math.sqrt(int(query_count))))
    length_x = float(model.config.core_honf.domain_length_x)
    length_y = float(model.config.core_honf.domain_length_y)
    x = np.linspace(0.0, length_x, side, dtype=np.float32)
    y = np.linspace(0.0, length_y, side, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)[: int(query_count)]


def _controller_shape_checks(torch: Any, CaseGroupGate: Any, seed: int) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 9_000)
    rows: list[dict[str, Any]] = []
    for group_count in CONTROLLER_GROUP_COUNTS:
        gate = CaseGroupGate(control_dim=16, group_count=group_count).cpu()
        logits = torch.randn(3, group_count - 1, generator=generator)
        noise = torch.linspace(0.08, 0.92, group_count - 1).unsqueeze(0).expand(3, -1)
        budget = gate.build_budget(logits, noise=noise, deterministic=False)
        if tuple(budget.z.shape) != (3, group_count):
            raise RuntimeError(f"K={group_count}: unexpected z shape {tuple(budget.z.shape)}")
        if tuple(budget.packed_ids.shape[:1]) != (3,):
            raise RuntimeError(f"K={group_count}: unexpected packed id batch shape")
        if not bool(budget.support[:, 0].all()):
            raise RuntimeError(f"K={group_count}: ordinary group 0 was not always available")
        if not bool(torch.isfinite(budget.kappa).all()):
            raise RuntimeError(f"K={group_count}: non-finite gate-reference kappa")
        rows.append(
            {
                "group_count": group_count,
                "z_shape": list(budget.z.shape),
                "support_shape": list(budget.support.shape),
                "packed_ids_shape": list(budget.packed_ids.shape),
                "packed_valid_shape": list(budget.packed_valid.shape),
                "ordinary_group_zero_open": True,
                "kappa_finite": True,
                "fixed_noise": True,
            }
        )
    return rows


def _run_shape(model: Any, dataset: Any, case_id: str, query_count: int, index: int, args: argparse.Namespace, torch: Any, dynamic: Any, run1405: Any, make_batch: Any, select_sample: Any) -> dict[str, Any]:
    sample = select_sample(dataset, case_id, 0)
    query_np = _query_points(sample, int(query_count), model)
    if len(query_np) != int(query_count):
        raise ValueError(f"case {case_id} produced Q={len(query_np)}, expected {query_count}")
    batch = make_batch(dict(sample), query_np, torch.device(args.device))
    query = batch["query_xy"]
    kwargs = run1405._phase_forward_kwargs(batch)
    noise_generator = torch.Generator(device=query.device)
    noise_generator.manual_seed(int(args.seed) + 10_000 + index)
    noise = torch.rand(
        int(query.shape[0]),
        int(model.config.core_honf.interface_model.group_count) - 1,
        device=query.device,
        dtype=query.dtype,
        generator=noise_generator,
    )
    outputs: dict[str, Mapping[str, Any]] = {}
    fields: dict[str, Any] = {}
    model.eval()
    with _fixed_gate_noise(model, noise), torch.inference_mode():
        # One model instance is used sequentially.  The first call materializes
        # lazy projections; the compact call then sees exactly those weights.
        for mode in ("full_width", "compact"):
            _set_execution_mode(model, mode)
            output = model(
                batch["structure"],
                query,
                return_prepared_state=True,
                return_routing_maps=False,
                **kwargs,
            )
            outputs[mode] = output
            fields[mode] = output["pred_field"].detach().clone()
    full_gate = _gate_summary(outputs["full_width"])
    compact_gate = _gate_summary(outputs["compact"])
    full_support = np.asarray(full_gate["gate_support"], dtype=bool)
    compact_support = np.asarray(compact_gate["gate_support"], dtype=bool)
    full_values = np.asarray(full_gate["gate_values"], dtype=np.float64)
    compact_values = np.asarray(compact_gate["gate_values"], dtype=np.float64)
    if not np.array_equal(full_support, compact_support):
        raise RuntimeError(f"case {case_id}: full/compact fixed gate supports differ")
    if not np.allclose(full_values, compact_values, rtol=1.0e-6, atol=1.0e-7):
        raise RuntimeError(f"case {case_id}: full/compact fixed gate values differ")
    field_difference = _field_difference(fields["full_width"], fields["compact"])
    # Do not retain prepared-state tensors from the parity calls while timing;
    # the timed memory record should describe the forward itself.
    del outputs
    del fields
    gc.collect()
    timing: dict[str, dict[str, Any]] = {}
    # Measure only after both paths have materialized their lazy projections.
    # Each timed call receives the same fixed P0 noise; P1/P2 state remains
    # live and is refreshed inside the ordinary forward.
    for mode in ("full_width", "compact"):
        timing[mode] = _measure_mode(
            model,
            batch,
            query,
            kwargs,
            mode=mode,
            fixed_noise=noise,
            torch=torch,
            device=query.device,
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        )
    full_latency = float(timing["full_width"]["latency"]["median_ms"])
    compact_latency = float(timing["compact"]["latency"]["median_ms"])
    full_memory = timing["full_width"]["peak_allocated_bytes"]
    compact_memory = timing["compact"]["peak_allocated_bytes"]
    result = {
        "case_id": str(case_id),
        "query_count": int(query_count),
        "model_instance_policy": "one fresh model; full_width then compact on the same materialized weights",
        "fixed_gate_noise_seed": int(args.seed) + 10_000 + index,
        "full_width": {"gate": full_gate},
        "compact": {"gate": compact_gate},
        "field_difference": field_difference,
        "timing": timing,
        "paired_cost": {
            "latency_ratio_compact_over_full_median": compact_latency / max(full_latency, 1.0e-12),
            "peak_allocated_ratio_compact_over_full": None
            if full_memory is None or compact_memory is None
            else float(compact_memory) / max(float(full_memory), 1.0),
            "interpretation": "measured implementation cost on the same fresh materialized model and fixed gate noise",
        },
    }
    del batch, query
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_stage1(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.seed) < 0:
        raise ValueError("seed must be nonnegative")
    if int(args.warmups) < 0 or int(args.repetitions) <= 0:
        raise ValueError("warmups must be nonnegative and repetitions must be positive")
    shapes = [_parse_shape(spec) for spec in (args.shape or DEFAULT_SHAPES)]
    torch, dynamic, run1405, make_batch, select_sample, CaseGroupGate = _runtime_imports()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")
        torch.cuda.set_device(device)
    profile_args = SimpleNamespace(
        profile=str(args.profile),
        output=str(Path(args.output).expanduser().resolve()),
        dataset=args.dataset,
        split="train",
        points_per_case=max(count for _, count in shapes),
    )
    model, dataset, checkpoint, dataset_path = dynamic._build_fresh_profile_model(profile_args, device)
    if str(args.split) != "train":
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        dataset, dataset_path = dynamic._load_dataset(
            checkpoint,
            SimpleNamespace(dataset=args.dataset, split=str(args.split)),
            split_override=str(args.split),
            points_per_case_override=max(count for _, count in shapes),
            random_point_sampling_override=False,
        )
    architecture = str(model.config.core_honf.forward_architecture)
    if architecture != "budgeted_group_control_honf":
        raise ValueError(f"expected budgeted_group_control_honf, got {architecture!r}")
    group_count = int(model.config.core_honf.interface_model.group_count)
    if group_count != 12:
        raise ValueError(f"expected Run-1409 Kmax=12, got {group_count}")
    if str(getattr(model, "budgeted_schedule_mode", "static")) == "dense_to_sparse_v2":
        raise ValueError(
            "run_run1409_stage1_shapes.py encodes the historical v1 ordinary-group contract; "
            "use run_run1409_v2_prelaunch_audit.py and run_run1409_budgeted_evidence.py "
            "for the symmetric rescue schedule"
        )
    rows = []
    try:
        for index, (case_id, query_count) in enumerate(shapes):
            rows.append(_run_shape(model, dataset, case_id, query_count, index, args, torch, dynamic, run1405, make_batch, select_sample))
        controller = _controller_shape_checks(torch, CaseGroupGate, int(args.seed))
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "task": "run1409_stage1_full_width_compact_and_controller_shapes",
        "status": "complete",
        "profile": str(Path(args.profile).expanduser().resolve()),
        "dataset": str(dataset_path),
        "device": str(device),
        "seed": int(args.seed),
        "fresh_model": True,
        "timing_protocol": {
            "warmups": int(args.warmups),
            "repetitions": int(args.repetitions),
            "maps": False,
            "gate_noise": "one fixed P0 tensor per representative shape, reused for both modes and all timed calls",
            "synchronization": "torch.cuda.Event plus explicit synchronize on CUDA; perf_counter on CPU",
            "memory": "peak allocated/reserved bytes and deltas from post-warmup baseline on CUDA",
        },
        "optimizer_steps": 0,
        "checkpoint_written": False,
        "shapes": rows,
        "controller_shape_checks": controller,
        "interpretation": [
            "Full and compact are two implementations of the same fixed-gate operator on one fresh materialized model.",
            "The field difference is an implementation check; it is not a claim of physical rank or sparsity.",
            "Controller K=6/12/32 checks use supplied fixed noise and do not run a trained sweep.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="src/config_core/forward/budgeted_group_control_honf_context.json",
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--shape", action="append", default=None, help="CASE=QUERY_COUNT; repeat for bounded checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_stage1(args)
    _write_json(args.output, payload)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_stage1"]
