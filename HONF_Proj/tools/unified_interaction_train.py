"""Maintained lifecycle CLI for the shared interaction-refinement trainer.

This runner is intentionally limited to the sealed development profiles. A
separate full-data provider and fresh-normalization binding are required for
any manual formal recipe; this command does not reinterpret development data
as formal training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "diagnostics/generated/unified_refinement_20261007/runs"


def _add_import_paths() -> None:
    paths = (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tools",
        PROJECT_ROOT / "Case_ThermalChannel/src",
        PROJECT_ROOT / "Case_WindFarm/src",
    )
    for path in reversed(paths):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} into a runner receipt.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False,
                                    default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("run-id must use 1–80 letters, digits, dots, underscores, or hyphens.")
    return value


def _output_dir(task: str, run_id: str, arm: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / task / _safe_run_id(run_id) / arm


def _seed_everything(seed: int) -> None:
    _add_import_paths()
    from honf_runtime.reproducibility import seed_all

    seed_all(seed)


def _factory(task: str):
    _add_import_paths()
    if task == "thermal":
        from channelthermal.training.unified_task import create_task
    elif task == "wind":
        from windfarm.training.unified_task import create_task
    else:
        raise ValueError(f"Unsupported development task {task!r}.")
    return create_task


def _effective_seed(args: argparse.Namespace) -> int:
    profile_seed = 0 if args.task == "thermal" else 42
    if args.seed is not None and int(args.seed) != profile_seed:
        raise ValueError(f"The sealed {args.task} development profile uses seed {profile_seed}.")
    return profile_seed


def _model_state_sha256(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not torch.is_tensor(value):
            raise TypeError(f"Model state {name!r} is not a tensor and cannot be sealed.")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _profile(task: str, seed: int):
    from honf_runtime.unified_training import (
        EngineConfig,
        SelectionPolicy,
        TrainingEngine,
    )

    microbatch, effective = (8, 48) if task == "thermal" else (4, 24)
    config = EngineConfig(
        seed=seed,
        microbatch_cases=microbatch,
        effective_cases=effective,
        total_epochs=2500,
        warmup_epochs=500,
        open_through_epoch=600,
        soft_through_epoch=800,
        monitor_every=100,
        gradient_clip=1.0,
    )
    selection = (SelectionPolicy(field_metric="field_score", response_guard_metric="response_guard_max_ratio",
                                 maximum_response_ratio=1.10)
                 if task == "thermal" else
                 SelectionPolicy(field_metric="field_score", response_guard_metric=None))
    return config, selection, TrainingEngine


def _build(args: argparse.Namespace, *, device: str | None = None):
    seed = _effective_seed(args)
    args.seed = seed
    _seed_everything(seed)
    factory = _factory(args.task)
    model, provider, optimizer_seed = factory({
        "seed": seed,
        "device": device or args.device,
    })
    config, selection, engine_class = _profile(args.task, args.seed)
    engine = engine_class(config, device=device or args.device, selection=selection)
    identity = {
        "workflow": "unified_interaction_refinement",
        "development_profile": "fixed25_v1" if args.task == "thermal" else "fixed24_v1",
        "task": "ThermalChannel" if args.task == "thermal" else "WindFarm",
        "run_id": _safe_run_id(args.run_id),
        "seed": seed,
        "initial_model_state_sha256": _model_state_sha256(model),
        "engine_profile": "warmup500_open600_soft800_total2500",
    }
    return model, provider, optimizer_seed, engine, identity


def _preparation_payload(args: argparse.Namespace) -> dict[str, Any]:
    model, provider, optimizer_seed, engine, identity = _build(args)
    payload = {
        "status": "prepared_only",
        "created_unix": time.time(),
        "identity": identity,
        "provider_identity": dict(provider.identity_payload()),
        "provider_summary": provider.preparation_summary(),
        "engine_config": engine.config.__dict__,
        "selection_policy": engine.selection.__dict__,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_seed_parameter_count": len(optimizer_seed.get("state_by_name", optimizer_seed))
        if isinstance(optimizer_seed, Mapping) else 0,
        "device": str(engine.device),
        "optimizer_started": False,
        "solver_attempts": 0,
    }
    return payload


def _command_prepare(args: argparse.Namespace) -> int:
    payload = _preparation_payload(args)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else _output_dir(
        args.task, args.run_id, "prepare")
    _write_json(output / "prepare.json", payload)
    print(json.dumps({"prepare_receipt": str(output / "prepare.json"), **payload}, indent=2,
                     sort_keys=True, default=_json_default))
    return 0


def _command_dry_run(args: argparse.Namespace) -> int:
    # The dry run is always CPU-only and disposable. It exercises the common
    # engine's real one-update path, then drops the in-memory model and moments.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.device = "cpu"
    model, provider, optimizer_seed, engine, identity = _build(args, device="cpu")
    receipt = engine.preflight_one_update(model, provider, optimizer_seed=optimizer_seed)
    result = {
        "status": "cpu_dry_run_passed",
        "identity": identity,
        "provider_identity": dict(provider.identity_payload()),
        "preparation": provider.preparation_summary(),
        "update": receipt,
        "checkpoint_written": False,
        "formal_training_started": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


def _command_start(args: argparse.Namespace) -> int:
    if args.arm == "warmup" and args.branch_from:
        raise ValueError("The shared warmup starts from its retained task parent, not a branch checkpoint.")
    if args.arm != "warmup" and not args.branch_from:
        raise ValueError("Full-detail and Adaptive-detail start only from the shared warmup checkpoint.")
    model, provider, optimizer_seed, engine, identity = _build(args)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else _output_dir(
        args.task, args.run_id, args.arm)
    result = engine.fit(
        model,
        provider,
        output,
        identity=identity,
        arm=args.arm,
        stop_after=args.stop_after,
        optimizer_seed=optimizer_seed,
        branch_from_checkpoint=args.branch_from,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


def _command_resume(args: argparse.Namespace) -> int:
    model, provider, optimizer_seed, engine, identity = _build(args)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else _output_dir(
        args.task, args.run_id, args.arm)
    from honf_runtime.run_layout import resolve_checkpoint

    checkpoint = resolve_checkpoint(output, args.checkpoint or "latest")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Exact-resume checkpoint does not exist: {checkpoint}")
    result = engine.fit(
        model,
        provider,
        output,
        identity=identity,
        arm=args.arm,
        stop_after=args.stop_after,
        optimizer_seed=optimizer_seed,
        resume_checkpoint=checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


def _command_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    _add_import_paths()
    from honf_runtime.run_layout import RunLayout

    layout = RunLayout(run_dir)
    result = {"run_dir": str(run_dir)}
    for name in ("active_process.json", "fit_summary.json"):
        path = layout.read_path(name)
        if path.is_file():
            result[name.removesuffix(".json")] = _read_json(path)
    active = result.get("active_process")
    if active is not None:
        pid = active.get("pid")
        alive = None
        if isinstance(pid, int) and pid > 0:
            try:
                stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                alive = stat[0] != "Z" and (
                    active.get("process_start_ticks") is None or stat[19] == active["process_start_ticks"])
            except (OSError, IndexError):
                alive = False
        result["process_alive"] = alive
        if active.get("status") == "running":
            result["status"] = "running" if alive else ("stale_running_receipt" if alive is False
                                                        else "running_receipt_liveness_unknown")
        else:
            result["status"] = active.get("status")
    history = layout.read_path("history.json")
    if history.is_file():
        rows = json.loads(history.read_text(encoding="utf-8"))
        result["history"] = {
            "epochs": len(rows),
            "last_epoch": rows[-1]["epoch"] if rows else None,
            "last_phase": rows[-1].get("phase") if rows else None,
            "last_case_visits": rows[-1].get("case_visits") if rows else None,
        }
    request = run_dir / "CLEAN_STOP_REQUEST.json"
    if request.is_file():
        result["clean_stop_requested"] = _read_json(request)
    acknowledged = layout.read_path("clean_stop_acknowledged.json")
    if acknowledged.is_file():
        result["clean_stop_acknowledged"] = _read_json(acknowledged)
    result["consumed_clean_stop_receipts"] = [
        str(path) for path in sorted({*run_dir.glob("clean_stop_consumed_*.json"),
                                     *(run_dir / "logs").glob("clean_stop_consumed_*.json")})
    ]
    if len(result) == 1:
        result["status"] = "no runner receipt in this directory"
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


def _command_clean_stop(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    request = run_dir / "CLEAN_STOP_REQUEST.json"
    if request.is_file():
        payload = _read_json(request)
    else:
        payload = {"request_id": str(uuid.uuid4()), "requested_unix": time.time(),
                   "requested_by": "unified_interaction_train.py", "note": args.note}
        _write_json(request, payload)
    print(json.dumps({"clean_stop_request": str(request), **payload}, indent=2, sort_keys=True))
    return 0


def _task_args(parser: argparse.ArgumentParser, *, output: bool = False) -> None:
    parser.add_argument("--task", choices=("thermal", "wind"), required=True)
    parser.add_argument("--run-id", default="development")
    parser.add_argument("--seed", type=int, default=None,
                        help="profile default: Thermal 0, Wind 42; matched arms share the sealed value")
    parser.add_argument("--device", default="cpu")
    if output:
        parser.add_argument("--output-dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="bind the sealed development task and write identity receipts")
    _task_args(prepare, output=True)
    prepare.set_defaults(handler=_command_prepare)

    dry_run = subparsers.add_parser("dry-run", help="run one disposable CPU engine update without checkpointing")
    _task_args(dry_run)
    dry_run.set_defaults(handler=_command_dry_run)

    start = subparsers.add_parser("start", help="start warmup or branch a matched arm from warmup")
    _task_args(start, output=True)
    start.add_argument("--arm", choices=("warmup", "full_detail", "adaptive_detail"), default="warmup")
    start.add_argument("--stop-after", type=int, required=True)
    start.add_argument("--branch-from")
    start.set_defaults(handler=_command_start)

    resume = subparsers.add_parser("resume", help="exactly resume the same arm from its latest checkpoint")
    _task_args(resume, output=True)
    resume.add_argument("--arm", choices=("warmup", "full_detail", "adaptive_detail"), required=True)
    resume.add_argument("--stop-after", type=int, required=True)
    resume.add_argument("--checkpoint")
    resume.set_defaults(handler=_command_resume)

    status = subparsers.add_parser("status", help="read the current runner and checkpoint receipts")
    status.add_argument("--run-dir", required=True)
    status.set_defaults(handler=_command_status)

    clean_stop = subparsers.add_parser("clean-stop", help="request a stop at the next completed epoch boundary")
    clean_stop.add_argument("--run-dir", required=True)
    clean_stop.add_argument("--note", default="operator requested clean stop")
    clean_stop.set_defaults(handler=_command_clean_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ArithmeticError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
