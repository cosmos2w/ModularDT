"""Opt-in frozen Run-1503 coarse/fine branch interventions.

The intervention is deliberately implemented at the adaptive backend's
``_read_environment`` boundary.  For the final ``p2_field`` read only, it
subtracts the backend-emitted coarse or fine contribution from the ordinary
environmental response.  Preparation, routing, module/local paths, and model
parameters remain unchanged.  The context manager restores the backend
method unconditionally, so invoking this tool cannot change training or
ordinary model defaults.

This file is separate from the timing runner because intervention predictions
are not timing evidence.  Both tools use the same strict checkpoint identity
gate and explicit LABEL=PATH checkpoint syntax.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ARCHITECTURE = "adaptive_hyperedge_opening_honf"
INTERVENTION_MODES = ("coarse_suppressed", "fine_suppressed")
INTERVENTION_ROLE = "p2_field"


def _imports() -> tuple[Any, ...]:
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_run1405_epoch50_comparison as run1405
    import run_stage3_interface_study as stage3
    import torch
    from channelthermal.evaluation.loading import make_batch
    from channelthermal.evaluation.prepared import select_sample

    return torch, run1405, stage3, make_batch, select_sample


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu()
        if int(value.numel()) == 1:
            return _jsonable(value.item())
        return value.tolist() if int(value.numel()) <= 256 else {"shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist() if int(value.size) <= 256 else {"shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _query_points(sample: Mapping[str, Any], count: int) -> np.ndarray:
    x = np.asarray(sample["x_grid"], dtype=np.float32).reshape(-1)
    y = np.asarray(sample["y_grid"], dtype=np.float32).reshape(-1)
    points = np.stack([x, y], axis=-1)
    if int(count) <= 0 or int(count) > len(points):
        raise ValueError(f"query count must be in [1,{len(points)}]")
    if int(count) == len(points):
        return points
    indices = np.rint(np.linspace(0, len(points) - 1, int(count))).astype(np.int64)
    return points[indices]


def _require_checkpoint_identity(
    model: Any,
    checkpoint: Mapping[str, Any],
    *,
    expected_epoch: int | None = None,
) -> dict[str, Any]:
    core = getattr(getattr(model, "config", None), "core_honf", None)
    architecture = str(getattr(core, "forward_architecture", ""))
    if architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(
            "Run-1503 interventions require exact architecture "
            f"{EXPECTED_ARCHITECTURE!r}, got {architecture!r}"
        )
    model_config = checkpoint.get("model_config", {})
    config_core = model_config.get("core_honf", {}) if isinstance(model_config, Mapping) else {}
    config_architecture = config_core.get("forward_architecture") if isinstance(config_core, Mapping) else None
    if config_architecture is not None and str(config_architecture) != EXPECTED_ARCHITECTURE:
        raise ValueError(
            "checkpoint metadata architecture does not match Run-1503: "
            f"{config_architecture!r}"
        )
    raw_epoch = checkpoint.get("epoch", checkpoint.get("current_epoch", -1))
    try:
        epoch = int(raw_epoch)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint epoch is not an integer: {raw_epoch!r}") from exc
    selection_state = checkpoint.get("selection_state")
    if isinstance(selection_state, Mapping) and selection_state.get("epoch") is not None:
        try:
            selection_epoch = int(selection_state["epoch"])
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint selection_state.epoch is not an integer") from exc
        if selection_epoch != epoch:
            raise ValueError(
                "checkpoint epoch metadata disagree: "
                f"top-level={epoch}, selection_state={selection_epoch}"
            )
    if expected_epoch is not None:
        if int(expected_epoch) not in {150, 500}:
            raise ValueError("Run-1503 gate epoch must be exactly 150 or 500")
        if epoch != int(expected_epoch):
            raise ValueError(
                f"Run-1503 gate requires epoch {int(expected_epoch)}, checkpoint has epoch {epoch}"
            )
    return {
        "architecture": architecture,
        "epoch": epoch,
        "strict_model_state_load": True,
        "expected_gate_epoch": None if expected_epoch is None else int(expected_epoch),
    }


def _branch_context_key(mode: str) -> str:
    if mode == "coarse_suppressed":
        return "group_control_adaptive_coarse_contribution"
    if mode == "fine_suppressed":
        return "group_control_adaptive_fine_contribution"
    raise ValueError(f"unknown Run-1503 intervention mode={mode!r}")


@contextlib.contextmanager
def adaptive_branch_intervention(
    model: Any,
    mode: str,
    *,
    role: str = INTERVENTION_ROLE,
) -> Iterator[dict[str, Any]]:
    """Temporarily suppress exactly one emitted adaptive branch at one role."""

    architecture = str(model.config.core_honf.forward_architecture)
    if architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(
            "adaptive_branch_intervention requires exact architecture "
            f"{EXPECTED_ARCHITECTURE!r}, got {architecture!r}"
        )
    if mode not in INTERVENTION_MODES:
        raise ValueError(f"unknown Run-1503 intervention mode={mode!r}")
    backend = getattr(getattr(model, "core", None), "backend", None)
    original = getattr(backend, "_read_environment", None)
    if not callable(original):
        raise TypeError("adaptive backend does not expose _read_environment")
    contribution_key = _branch_context_key(mode)

    def read_with_branch_suppressed(*args: Any, **kwargs: Any) -> Any:
        context, aux = original(*args, **kwargs)
        active_role = getattr(model.core, "_interface_read_role", None)
        if role is not None and active_role != str(role):
            return context, aux
        if not isinstance(aux, Mapping) or contribution_key not in aux:
            raise RuntimeError(
                f"adaptive backend did not emit required {contribution_key!r} for {mode}"
            )
        contribution = aux[contribution_key]
        if not hasattr(contribution, "shape") or tuple(contribution.shape) != tuple(context.shape):
            raise RuntimeError(
                f"{contribution_key} shape does not match environmental context: "
                f"{getattr(contribution, 'shape', None)} != {getattr(context, 'shape', None)}"
            )
        # Do not mutate aux: it remains the normal ledger, making the
        # prediction delta attributable to this one branch subtraction.
        return context - contribution, aux

    backend._read_environment = read_with_branch_suppressed
    try:
        yield {
            "mode": str(mode),
            "role": None if role is None else str(role),
            "branch_key": contribution_key,
            "implementation": "temporary_backend_read_environment_subtraction",
        }
    finally:
        backend._read_environment = original


def _prediction_delta(base: Any, variant: Any) -> dict[str, float]:
    difference = (variant.detach().float() - base.detach().float()).abs()
    return {
        "max_abs": float(difference.max().cpu()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().cpu()) if difference.numel() else 0.0,
        "rms": float(torch_rms(difference)),
    }


def torch_rms(value: Any) -> Any:
    """Use tensor operations without importing torch at module import time."""

    return value.square().mean().sqrt().cpu() if value.numel() else value.new_zeros(())


def _run_prediction(
    model: Any,
    sample: Mapping[str, Any],
    query_np: np.ndarray,
    device: Any,
    make_batch: Any,
    run1405: Any,
    torch: Any,
    intervention: contextlib.AbstractContextManager[Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    batch = make_batch(dict(sample), query_np, device)
    kwargs = run1405._phase_forward_kwargs(batch)
    scope = contextlib.nullcontext() if intervention is None else intervention
    with scope, torch.inference_mode():
        output = model(
            batch["structure"],
            batch["query_xy"],
            return_routing_maps=False,
            **kwargs,
        )
    field = output["pred_field"].detach().float()
    aux = output.get("interaction_aux", {})
    branch_norms: dict[str, Any] = {}
    if isinstance(aux, Mapping):
        for name in (
            "group_control_adaptive_coarse_contribution",
            "group_control_adaptive_fine_contribution",
        ):
            value = aux.get(name)
            if hasattr(value, "detach"):
                tensor = value.detach().float()
                branch_norms[name] = {
                    "max_abs": float(tensor.abs().max().cpu()) if tensor.numel() else 0.0,
                    "rms": float(torch_rms(tensor)),
                }
    del output
    return field, {"field_shape": list(field.shape), "branch_norms": branch_norms}


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch, run1405, stage3, make_batch, select_sample = _imports()
    specs = stage3.parse_checkpoint_specs(args.checkpoint)
    if len(specs) != 1:
        raise ValueError("Run-1503 interventions require exactly one labelled checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    spec = specs[0]
    model, checkpoint_payload = stage3._load_model_spec(spec, device)
    identity = _require_checkpoint_identity(
        model,
        checkpoint_payload,
        expected_epoch=args.expected_epoch,
    )
    dataset, dataset_path = stage3._load_dataset(
        checkpoint_payload,
        argparse.Namespace(dataset=args.dataset, split=args.split),
    )
    case_ids = [str(case_id) for case_id in (args.case_id or ["0273", "0653"])]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be non-empty and unique")
    modes = tuple(args.mode) if args.mode else INTERVENTION_MODES
    if len(set(modes)) != len(modes):
        raise ValueError("intervention modes must be unique")
    rows: list[dict[str, Any]] = []
    state_keys_before = tuple(model.state_dict())
    try:
        for case_id in case_ids:
            sample = select_sample(dataset, case_id, 0)
            query_np = _query_points(sample, int(args.query_count))
            normal, normal_meta = _run_prediction(
                model, sample, query_np, device, make_batch, run1405, torch
            )
            row: dict[str, Any] = {
                "case_id": case_id,
                "query_count": int(args.query_count),
                "intervention_scope": INTERVENTION_ROLE,
                "normal": {"status": "complete", **normal_meta},
                "interventions": {},
            }
            for mode in modes:
                try:
                    with adaptive_branch_intervention(
                        model,
                        mode,
                        role=INTERVENTION_ROLE,
                    ) as intervention_meta:
                        variant, variant_meta = _run_prediction(
                            model,
                            sample,
                            query_np,
                            device,
                            make_batch,
                            run1405,
                            torch,
                        )
                    row["interventions"][mode] = {
                        "status": "complete",
                        **intervention_meta,
                        **variant_meta,
                        "prediction_difference": _prediction_delta(normal, variant),
                    }
                    del variant
                except Exception as exc:  # noqa: BLE001 - preserve per-mode availability
                    row["interventions"][mode] = {
                        "status": "unavailable",
                        "error": f"{type(exc).__name__}: {exc!s}",
                        "mode": mode,
                        "role": INTERVENTION_ROLE,
                    }
            rows.append(row)
            del normal
    finally:
        dataset.close()
    state_keys_after = tuple(model.state_dict())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "schema_version": 1,
        "task": "run1503_adaptive_branch_interventions",
        "status": "complete"
        if all(
            row["normal"]["status"] == "complete"
            and all(item["status"] == "complete" for item in row["interventions"].values())
            for row in rows
        )
        else "partial",
        "checkpoint": {"label": spec.label, "path": str(spec.path), **identity},
        "dataset": str(dataset_path),
        "device": str(device),
        "case_ids": case_ids,
        "query_count": int(args.query_count),
        "modes": list(modes),
        "state_dict_structure_unchanged": state_keys_before == state_keys_after,
        "timed": False,
        "interpretation": {
            "normal": "ordinary frozen checkpoint prediction",
            "coarse_suppressed": "P2 environmental context minus emitted coarse contribution",
            "fine_suppressed": "P2 environmental context minus emitted fine contribution",
            "causality": "same-weight frozen-model reliance only; not physical causality",
        },
        "limitations": [
            "The intervention scope is the final p2_field read; P0/P1 preparation and local/module paths are unchanged.",
            "No training, optimizer update, checkpoint write, or model-default change is performed.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--query-count", type=int, default=8192)
    parser.add_argument(
        "--mode",
        action="append",
        choices=INTERVENTION_MODES,
        default=[],
        help="repeat to opt into a subset; normal is always evaluated",
    )
    parser.add_argument("--expected-epoch", type=int, choices=(150, 500), default=None)
    parser.add_argument("--device", default="cuda:1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.case_id:
        args.case_id = ["0273", "0653"]
    payload = run(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ARCHITECTURE",
    "INTERVENTION_MODES",
    "adaptive_branch_intervention",
    "build_parser",
    "run",
]
