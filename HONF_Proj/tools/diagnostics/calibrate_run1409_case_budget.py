"""Historical v1 Run-1409 case-budget coefficient calibration.

This command uses exactly two real training batches from the selected profile.
It samples one fixed hard-concrete noise tensor per batch, computes

``lambda = 0.02 * median(||d L_physical / d a|| / ||d L_group / d a||)``

where ``a`` is the live optional gate-logit tensor.  It performs no optimizer
step and writes no checkpoint.  The output JSON is the calibration evidence;
the parent launch should copy its single scalar into a one-off strict
experiment overlay's ``case.loss.case_group_budget_weight`` field.

The profile remains at its numeric zero coefficient while calibration runs.
At the end the intended seed is restored so the caller can launch the formal
managed run with the original seed stream.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUP_COUNT = 12
CALIBRATION_BATCHES = 2
CALIBRATION_FACTOR = 0.02


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
    from torch.utils.data import DataLoader

    import benchmark_routing_optimization as benchmark
    import run_dynamic_sparse_routing_study as dynamic
    from channelthermal.data.collation import ChannelThermalBatchCollator
    from channelthermal.training.epoch import (
        assemble_channelthermal_loss_terms,
        effective_port_global_weight,
        make_model_inputs,
    )
    from honf_runtime.compat import recursive_to_device, set_seed

    return (
        torch,
        DataLoader,
        benchmark,
        dynamic,
        ChannelThermalBatchCollator,
        assemble_channelthermal_loss_terms,
        effective_port_global_weight,
        make_model_inputs,
        recursive_to_device,
        set_seed,
    )


@contextlib.contextmanager
def _fixed_gate_noise(model: Any, noise: Any) -> Iterator[None]:
    """Inject one fixed noise tensor only into the P0 gate plan."""

    backend = getattr(getattr(model, "core", None), "backend", None)
    router = getattr(backend, "router", None)
    original = getattr(router, "prepare", None)
    if router is None or not callable(original):
        raise RuntimeError("budgeted backend does not expose router.prepare for fixed-noise calibration")

    def prepare_with_fixed_noise(*args: Any, **kwargs: Any) -> Any:
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


def _gate_logits(output: Mapping[str, Any]) -> Any:
    prepared = output.get("prepared_state")
    plan = getattr(prepared, "phase_shared_state", None)
    logits = getattr(plan, "optional_logits", None)
    if logits is None or not hasattr(logits, "requires_grad") or not logits.requires_grad:
        raise RuntimeError(
            "Run-1409 output did not expose live phase_shared_state.optional_logits; "
            "calibration would otherwise measure a detached diagnostic"
        )
    return logits


def _fp64_norm(value: Any) -> float:
    if value is None:
        return 0.0
    value = value.detach().to(dtype=value.dtype)
    return float(value.double().norm().cpu())


def _build_loader(dataset: Any, checkpoint: Mapping[str, Any], batch_size: int, DataLoader: Any, Collator: Any) -> Any:
    dataset_cfg = dict(checkpoint.get("train_config", {}).get("dataset", {}))
    collator = Collator(
        dynamic_module_padding=bool(dataset_cfg.get("dynamic_module_padding", True)),
        max_modules_per_batch=dataset_cfg.get("max_modules_per_batch"),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator,
    )


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.batches) != CALIBRATION_BATCHES:
        raise ValueError("Run-1409 calibration is fixed to exactly two real training batches")
    if int(args.batch_size) <= 0 or int(args.points_per_case) <= 0:
        raise ValueError("batch-size and points-per-case must be positive")
    torch, DataLoader, benchmark, dynamic, Collator, assemble, effective_port_global_weight, make_inputs, recursive_to_device, set_seed = _runtime_imports()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device={args.device}")
        torch.cuda.set_device(device)
    # The helper builds the same fresh profile model path used by the formal
    # workflow and reads real train cases/targets from the packed HDF5.
    profile_args = SimpleNamespace(
        profile=str(args.profile),
        output=str(Path(args.output).expanduser().resolve()),
        dataset=args.dataset,
        split="train",
        points_per_case=int(args.points_per_case),
    )
    set_seed(int(args.seed))
    model, dataset, checkpoint, dataset_path = dynamic._build_fresh_profile_model(profile_args, device)
    if str(model.config.core_honf.forward_architecture) != "budgeted_group_control_honf":
        raise ValueError("calibration profile must use budgeted_group_control_honf")
    if int(model.config.core_honf.interface_model.group_count) != GROUP_COUNT:
        raise ValueError("calibration profile must use Kmax=12")
    if str(getattr(model, "budgeted_schedule_mode", "static")) == "dense_to_sparse_v2":
        raise ValueError(
            "this calibration records the historical privileged-group v1 protocol; "
            "the rescue rerun reuses its fixed coefficient and must not recalibrate"
        )
    # The fresh-model helper intentionally builds a stable read-only dataset.
    # Match the formal profile's real train sampling policy before taking the
    # two batches, including the first formal epoch stream.
    dataset_cfg = dict(checkpoint.get("train_config", {}).get("dataset", {}))
    dataset.random_point_sampling = bool(dataset_cfg.get("random_point_sampling", True))
    set_epoch = getattr(dataset, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(1)
    loader = _build_loader(dataset, checkpoint, int(args.batch_size), DataLoader, Collator)
    schedule = benchmark._training_schedule(checkpoint)
    training_config, loss_cfg, mode, ratio, internal_weight, interface_weight, predicted_weight = schedule
    if str(mode) != "predicted":
        raise ValueError(f"calibration requires predicted ports, got {mode!r}")
    rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    try:
        model.train()
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= CALIBRATION_BATCHES:
                break
            batch = recursive_to_device(raw_batch, device)
            batch_size = int(batch["query_xy"].shape[0])
            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.seed) + 10_000 + batch_index)
            noise = torch.rand(
                batch_size,
                GROUP_COUNT - 1,
                device=device,
                dtype=batch["query_xy"].dtype,
                generator=generator,
            )
            port_global_weight = effective_port_global_weight(loss_cfg, mode, ratio)
            inputs = make_inputs(
                batch,
                local_port_condition_mode=mode,
                mixed_teacher_ratio=ratio,
                return_predicted_port_outputs=bool(predicted_weight > 0.0),
                return_port_global_consistency=bool(port_global_weight != 0.0),
            )
            model.zero_grad(set_to_none=True)
            with _fixed_gate_noise(model, noise), torch.enable_grad():
                output = model(**inputs, return_prepared_state=True)
                terms = assemble(
                    output,
                    batch,
                    model,
                    loss_cfg,
                    local_port_condition_mode=mode,
                    mixed_teacher_ratio=ratio,
                    effective_internal_temperature_weight=internal_weight,
                    effective_interface_weight=interface_weight,
                    predicted_consistency_weight=predicted_weight,
                )
                logits = _gate_logits(output)
                physical_grad = torch.autograd.grad(
                    terms["loss_physical"], logits, retain_graph=True, allow_unused=False
                )[0]
                group_grad = torch.autograd.grad(
                    terms["loss_group_budget"], logits, retain_graph=False, allow_unused=False
                )[0]
            physical_norm = _fp64_norm(physical_grad)
            group_norm = _fp64_norm(group_grad)
            if not math.isfinite(physical_norm) or not math.isfinite(group_norm) or group_norm <= 0.0:
                raise RuntimeError(
                    f"invalid gate gradient norms on batch {batch_index + 1}: "
                    f"physical={physical_norm}, group={group_norm}"
                )
            ratio_value = physical_norm / group_norm
            ratios.append(ratio_value)
            rows.append(
                {
                    "batch_index": batch_index + 1,
                    "batch_size": batch_size,
                    "query_points": int(batch["query_xy"].shape[1]),
                    "fixed_noise_seed": int(args.seed) + 10_000 + batch_index,
                    "physical_loss": float(terms["loss_physical"].detach().cpu()),
                    "expected_optional_group_loss": float(terms["loss_group_budget"].detach().cpu()),
                    "physical_gate_logit_grad_norm": physical_norm,
                    "group_gate_logit_grad_norm": group_norm,
                    "physical_to_group_gradient_ratio": ratio_value,
                    "optional_logits_shape": list(logits.shape),
                    "optimizer_steps": 0,
                }
            )
            del output, terms, logits, physical_grad, group_grad, batch
            gc.collect()
        if len(rows) != CALIBRATION_BATCHES:
            raise RuntimeError(f"loader yielded {len(rows)} batches; exactly two are required")
    finally:
        set_seed(int(args.seed))
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        gc.collect()
    coefficient = CALIBRATION_FACTOR * float(np.median(np.asarray(ratios, dtype=np.float64)))
    return {
        "schema_version": 1,
        "task": "run1409_case_budget_lambda_calibration",
        "status": "complete",
        "profile": str(Path(args.profile).expanduser().resolve()),
        "dataset": str(dataset_path),
        "device": str(device),
        "seed": int(args.seed),
        "seed_restored_before_return": True,
        "batch_count": CALIBRATION_BATCHES,
        "batch_size": int(args.batch_size),
        "points_per_case": int(args.points_per_case),
        "factor": CALIBRATION_FACTOR,
        "formula": "case_group_budget_weight = 0.02 * median_batch(||d L_physical/d optional_logits||_2 / ||d L_group/d optional_logits||_2)",
        "case_group_budget_weight": coefficient,
        "batches": rows,
        "optimizer_steps": 0,
        "checkpoint_written": False,
        "overlay_target": "case.loss.case_group_budget_weight",
        "limitations": [
            "This is one coefficient calibration on two real training batches, not a sweep or a tuned validation result.",
            "The scalar must be copied into one strict Run-1409 experiment overlay before formal training.",
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
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--points-per-case", type=int, default=1024)
    parser.add_argument("--batches", type=int, default=CALIBRATION_BATCHES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_calibration(args)
    _write_json(args.output, payload)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"case_group_budget_weight={payload['case_group_budget_weight']:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CALIBRATION_BATCHES", "CALIBRATION_FACTOR", "build_parser", "main", "run_calibration"]
