#!/usr/bin/env python3
"""Prepare and run the bounded Round7 A2-D native-token continuation.

Training is an explicit opt-in command and requires a positive Round7 cost
admission receipt. CPU preparation never starts an optimizer or allocates a
run directory. No full-TRAIN, direct-E, or inverse workflow is implemented.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src", PROJECT / "Case_WindFarm/src", PROJECT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch

from honf_runtime.native_retrieval_factory import build_native_retrieval_task
from honf_runtime.native_retrieval_protocol import (
    NATIVE_RETRIEVAL_NATIVE_BACKENDS,
    canonical_sha256,
    make_native_retrieval_recipe,
    validate_native_retrieval_recipe,
)

GPU2_UUID = "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39"
REVIEW_STOPS = (600, 1000, 1500, 2500)


def _json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return value


def _write_new_json(path: Path, value: Any) -> Path:
    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite the existing Round7 recipe: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def _validate_admission(path: Path, recipe: dict[str, Any], *, stop_after: int) -> dict[str, Any]:
    receipt = _json_object(path, "Round7 execution admission receipt")
    dataset = recipe["dataset"]
    if (receipt.get("schema") != "round7_execution_admission_v1"
            or receipt.get("status") != "completed"
            or not isinstance(receipt.get("primary_pairs_admitted"), dict)
            or receipt["primary_pairs_admitted"].get(dataset) is not True
            or receipt.get("optional_E_pilots_admitted") is not False
            or receipt.get("full_TRAIN_authorized") is not False
            or receipt.get("inverse_or_solver_authorized") is not False
            or receipt.get("TEST_access") is not False):
        raise ValueError(
            f"Round7 {dataset} continuation is not admitted by this exact no-E, development-only cost receipt."
        )
    if type(receipt.get("solver_own_new_attempts")) is not int or receipt["solver_own_new_attempts"] != 0:
        raise ValueError("Round7 admission receipt must preserve the no-new-solver-attempt lock.")
    bindings = receipt.get("primary_pair_bindings")
    treatment_bindings = bindings.get(dataset, {}) if isinstance(bindings, dict) else {}
    selected = treatment_bindings.get(recipe["treatment"]) if isinstance(treatment_bindings, dict) else None
    if (not isinstance(selected, dict)
            or selected.get("recipe_sha256") != canonical_sha256(recipe)):
        raise ValueError("A positive Round7 pair decision must bind this exact treatment recipe hash.")
    if type(selected.get("maximum_stop_after")) is not int or not stop_after <= selected["maximum_stop_after"] <= 2500:
        raise ValueError("A positive Round7 treatment decision must bind its admitted absolute stop age.")
    return receipt


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TypeError("Round7 resource budget started_utc must be a UTC timestamp string.")
    try:
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Round7 resource budget started_utc must be an ISO or compact UTC timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError("Round7 resource budget started_utc must include its UTC offset.")
    return parsed.astimezone(timezone.utc)


def _regular_file_bytes(path: Path, *, exclude: Path | None = None) -> int:
    total = 0
    excluded = None if exclude is None else exclude.resolve()
    for directory, names, files in os.walk(path, followlinks=False):
        parent = Path(directory)
        names[:] = [name for name in names if not (parent / name).is_symlink()]
        for filename in files:
            item = parent / filename
            if item.is_symlink() or (excluded is not None and item.resolve().is_relative_to(excluded)):
                continue
            try:
                if item.is_file():
                    total += item.stat().st_size
            except FileNotFoundError:
                raise RuntimeError(f"Round7 storage inventory changed while checking {item}.")
    return total


def _recorded_gpu2_seconds(receipts: Path) -> float:
    total = 0.0
    for path in receipts.glob("*.json"):
        if path.name in {"storage_resource_budget.json", "gpu_resource_floor.json"}:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        # This directory also contains metadata inventories and figure indexes.
        # They are not resource receipts and need not share a JSON object shape.
        if not isinstance(value, Mapping):
            continue
        gpu_uuid = value.get("GPU_uuid", value.get("gpu_uuid"))
        if gpu_uuid is None:
            gpu = value.get("GPU", value.get("gpu"))
            if isinstance(gpu, Mapping):
                gpu_uuid = gpu.get("uuid", gpu.get("GPU_uuid", gpu.get("gpu_uuid")))
        if gpu_uuid != GPU2_UUID or "associated_seconds" not in value:
            continue
        seconds = value.get("associated_seconds", 0.0)
        if type(seconds) not in (int, float) or not math.isfinite(float(seconds)) or seconds < 0:
            raise ValueError(f"Round7 GPU2 receipt {path.name} has invalid associated_seconds.")
        total += float(seconds)
    ledger = receipts / "fit_resource_usage.jsonl"
    if ledger.exists():
        for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if (not isinstance(value, dict) or value.get("schema") != "round7_fit_resource_usage_v1"
                    or value.get("gpu_uuid") != GPU2_UUID):
                raise ValueError(f"Round7 GPU2 usage ledger line {line_number} is malformed.")
            seconds = value.get("associated_seconds")
            if type(seconds) not in (int, float) or not math.isfinite(float(seconds)) or seconds < 0:
                raise ValueError(f"Round7 GPU2 usage ledger line {line_number} has invalid seconds.")
            total += float(seconds)
    return total


def _validate_round_resources(
    admission_path: Path,
    output: Path,
    recipe: dict[str, Any],
    *,
    stop_after: int,
) -> tuple[Path, dict[str, Any]]:
    admission_path = admission_path.expanduser().resolve(strict=True)
    if admission_path.parent.name != "receipts":
        raise ValueError("Round7 admission receipts must live in the authorized round-local receipts directory.")
    round_root = admission_path.parent.parent.resolve(strict=True)
    allowed_diagnostics = Path("/data2/wanglz/ModularDT/diagnostics").resolve(strict=True)
    try:
        round_root.relative_to(allowed_diagnostics)
    except ValueError as exc:
        raise ValueError("Round7 payloads must be rooted under the designated /data2 diagnostics tree.") from exc
    if not os.path.ismount("/data2"):
        raise RuntimeError("The required /data2 payload filesystem is not currently mounted.")
    output = output.expanduser().resolve()
    try:
        output.relative_to(round_root)
    except ValueError as exc:
        raise ValueError("Round7 run outputs must remain under the authorized round-local /data2 root.") from exc

    budget_path = round_root / "receipts" / "storage_resource_budget.json"
    budget = _json_object(budget_path, "Round7 storage and GPU resource budget")
    if budget.get("root") != str(round_root):
        raise ValueError("Round7 resource budget is not sealed to this exact round root.")
    mounts = budget.get("mount", {}).get("filesystems", ())
    if not any(isinstance(item, dict) and item.get("target") == "/data2" for item in mounts):
        raise ValueError("Round7 resource budget does not record the /data2 payload mount.")
    hard_persistent = budget.get("persistent_hard_bytes")
    hard_temporary = budget.get("temporary_hard_bytes")
    peak_new = budget.get("peak_new_bytes")
    minimum_free = budget.get("minimum_free_bytes")
    if any(type(value) is not int or value <= 0 for value in (
            hard_persistent, hard_temporary, peak_new, minimum_free)):
        raise ValueError("Round7 resource budget must declare positive persistent, temporary, peak and headroom bytes.")
    if peak_new > hard_persistent + hard_temporary:
        raise ValueError("Round7 estimated peak exceeds the combined persistent and temporary hard limits.")
    free_now = shutil.disk_usage("/data2").free
    if budget.get("gpu_uuid") != GPU2_UUID:
        raise ValueError("Round7 GPU budget is not bound to physical GPU2.")
    hours = [budget.get(name) for name in (
        "gpu_associated_hours", "elapsed_hours", "closeout_reserved_hours")]
    if any(type(value) not in (int, float) or value < 0 for value in hours):
        raise ValueError("Round7 resource budget is missing its associated, elapsed or closeout-reserved GPU hours.")
    now = datetime.now(timezone.utc)
    elapsed_so_far = (now - _parse_utc(budget.get("started_utc"))).total_seconds() / 3600.0
    elapsed_remaining = float(hours[1]) - elapsed_so_far - float(hours[2])
    associated_recorded_seconds = _recorded_gpu2_seconds(admission_path.parent)
    # GPU2 is the only lane in this round. Some older qualification receipts
    # omitted or nested their UUID, so cap remaining association time with a
    # conservative wall-clock upper bound. This is a budget bound, not a claim
    # that the GPU was continuously busy.
    associated_used_seconds = max(associated_recorded_seconds, elapsed_so_far * 3600.0)
    associated_used = associated_used_seconds / 3600.0
    associated_remaining = float(hours[0]) - associated_used
    if elapsed_remaining <= 0:
        raise RuntimeError("Round7 elapsed deadline has reached its closeout reserve.")
    if associated_remaining <= 0:
        raise RuntimeError("Round7 GPU2 associated-time allowance is exhausted by recorded receipts.")

    floor_path = admission_path.parent / "gpu_resource_floor.json"
    floor = _json_object(floor_path, "Round7 supplemental GPU memory floor")
    minimum_gpu_free = floor.get("minimum_free_gpu_bytes")
    minimum_gpu_fraction = floor.get("minimum_free_fraction")
    if (floor.get("schema") != "round7_gpu_resource_floor_v1"
            or floor.get("root") != str(round_root)
            or floor.get("gpu_uuid") != GPU2_UUID
            or type(minimum_gpu_free) is not int or minimum_gpu_free <= 0
            or type(minimum_gpu_fraction) not in (int, float)
            or not math.isfinite(float(minimum_gpu_fraction))
            or not 0.0 < float(minimum_gpu_fraction) <= 1.0):
        raise ValueError("Round7 supplemental GPU floor is missing or does not match this exact round and GPU2.")

    receipt = _json_object(admission_path, "Round7 execution admission receipt")
    bindings = receipt.get("primary_pair_bindings", {})
    dataset_bindings = bindings.get(recipe["dataset"], {}) if isinstance(bindings, dict) else {}
    selected = dataset_bindings.get(recipe["treatment"]) if isinstance(dataset_bindings, dict) else None
    storage = selected.get("storage_estimate_bytes") if isinstance(selected, dict) else None
    run_hours = selected.get("resource_estimate_hours") if isinstance(selected, dict) else None
    if (not isinstance(storage, dict)
            or type(storage.get("persistent")) is not int or storage["persistent"] < 0
            or type(storage.get("temporary")) is not int or storage["temporary"] < 0):
        raise ValueError("Positive Round7 admission must include persistent and temporary output-byte estimates.")
    if (not isinstance(run_hours, dict)
            or any(type(run_hours.get(name)) not in (int, float)
                   or not math.isfinite(float(run_hours[name])) or run_hours[name] < 0
                   for name in ("gpu_associated", "elapsed"))):
        raise ValueError("Positive Round7 admission must estimate GPU-associated and elapsed run hours.")
    if selected.get("maximum_stop_after", 0) < stop_after:
        raise ValueError("Requested Round7 stop exceeds the treatment's admitted absolute stop age.")
    if run_hours["gpu_associated"] > associated_remaining:
        raise RuntimeError("Round7 admitted training estimate exceeds remaining GPU-associated time.")
    if run_hours["elapsed"] > elapsed_remaining:
        raise RuntimeError("Round7 admitted training estimate reaches the closeout-reserved deadline.")

    temporary_root = round_root / "temporary"
    persistent_used = _regular_file_bytes(round_root, exclude=temporary_root)
    temporary_used = _regular_file_bytes(temporary_root) if temporary_root.exists() else 0
    if persistent_used > hard_persistent or temporary_used > hard_temporary:
        raise OSError("Round7 existing payload already exceeds its persistent or temporary hard limit.")
    if persistent_used + storage["persistent"] > hard_persistent:
        raise OSError("Round7 projected persistent payload exceeds its sealed hard budget.")
    if temporary_used + storage["temporary"] > hard_temporary:
        raise OSError("Round7 projected temporary payload exceeds its sealed hard budget.")
    peak_projected = persistent_used + temporary_used + storage["persistent"] + storage["temporary"]
    if peak_projected > peak_new:
        raise OSError("Round7 current tree plus admitted run estimate exceeds its sealed peak-new-byte cap.")
    if free_now - storage["persistent"] - storage["temporary"] < minimum_free:
        raise OSError("Round7 /data2 free-space headroom is below the projected run-storage estimate.")
    return round_root, {
        **budget,
        "minimum_free_gpu_bytes": minimum_gpu_free,
        "minimum_free_gpu_fraction": float(minimum_gpu_fraction),
        "free_bytes_before_fit": free_now,
        "elapsed_hours_used": elapsed_so_far,
        "elapsed_hours_remaining_before_closeout": elapsed_remaining,
        "gpu_associated_seconds_recorded": associated_recorded_seconds,
        "gpu_associated_seconds_conservative_bound": associated_used_seconds,
        "gpu_associated_time_accounting": (
            "max(UUID-tagged GPU2 usage receipts, elapsed wall seconds since started_utc); "
            "conservative single-lane budget bound, not measured GPU-busy time"
        ),
        "gpu_associated_hours_used": associated_used,
        "gpu_associated_hours_remaining": associated_remaining,
        "persistent_bytes_used": persistent_used,
        "temporary_bytes_used": temporary_used,
        "projected_persistent_bytes": persistent_used + storage["persistent"],
        "projected_temporary_bytes": temporary_used + storage["temporary"],
    }


def _check_cuda_lane(
    device: str,
    *,
    minimum_free_gpu_bytes: int,
    minimum_free_gpu_fraction: float,
) -> None:
    # Check the physical lane before any CUDA availability/current-device call.
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU2_UUID:
        raise ValueError(
            "Round7 training is restricted to logical cuda:0 with CUDA_VISIBLE_DEVICES set to the exact physical GPU2 UUID."
        )
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    found = None
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[0] == GPU2_UUID:
            found = (int(fields[1]) * 1024 * 1024, int(fields[2]) * 1024 * 1024)
            break
    if (found is None
            or found[0] < max(minimum_free_gpu_bytes, math.ceil(minimum_free_gpu_fraction * found[1]))):
        raise RuntimeError("Round7 physical GPU2 is missing or below its sealed free-memory floor.")
    if torch.cuda.is_initialized():
        current = torch.cuda.current_device()
        if current != 0:
            raise RuntimeError("Round7 CUDA was initialized on a nonzero logical device.")


def _append_fit_resource_usage(
    round_root: Path,
    recipe: dict[str, Any],
    *,
    output: Path,
    started_perf: float,
    status: str,
    result: dict[str, Any] | None = None,
) -> None:
    elapsed = max(0.0, time.perf_counter() - started_perf)
    summary_path = output / "fit_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = _json_object(summary_path, "Round7 completed fit summary")
    active_path = output / "active_process.json"
    if not summary and active_path.is_file():
        summary = _json_object(active_path, "Round7 active fit summary")
    row = {
        "schema": "round7_fit_resource_usage_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_uuid": GPU2_UUID,
        "associated_seconds": elapsed,
        "status": status,
        "dataset": recipe["dataset"],
        "treatment": recipe["treatment"],
        "recipe_sha256": canonical_sha256(recipe),
        "output": str(output),
        "completed_epoch": summary.get("completed_epoch"),
        "optimizer_updates": summary.get("optimizer_updates", 0),
        "engine_seconds": summary.get("engine_seconds"),
        "process_seconds": summary.get("process_seconds"),
    }
    if result is not None:
        row["completed_epoch"] = result.get("completed_epoch", row["completed_epoch"])
        row["engine_seconds"] = result.get("engine_seconds", row["engine_seconds"])
        row["process_seconds"] = result.get("process_seconds", row["process_seconds"])
    ledger = round_root / "receipts" / "fit_resource_usage.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _recipe_command(args: argparse.Namespace) -> int:
    backend = args.backend
    tau_q = args.tau_q
    if args.treatment == "D-control":
        backend = "full_direct" if backend is None else backend
        tau_q = 0.0 if tau_q is None else tau_q
    else:
        backend = "selected" if backend is None else backend
        tau_q = 0.20 if tau_q is None else tau_q
    recipe = make_native_retrieval_recipe(
        args.source_checkpoint,
        treatment=args.treatment,
        native_backend=backend,
        tau_q=tau_q,
    )
    target = _write_new_json(args.output, recipe)
    print(json.dumps({
        "status": "recipe_written",
        "recipe": str(target),
        "dataset": recipe["dataset"],
        "treatment": recipe["treatment"],
        "source_checkpoint": recipe["source_checkpoint"],
        "source_checkpoint_sha256": recipe["source_checkpoint_sha256"],
        "recipe_sha256": canonical_sha256(recipe),
        "native_backend": recipe["native_backend"],
        "direct_environment_value_reader": False,
    }, sort_keys=True, allow_nan=False))
    return 0


def _prepare_command(args: argparse.Namespace) -> int:
    if args.device != "cpu":
        raise ValueError("Round7 preparation is a CPU-only operation.")
    recipe = validate_native_retrieval_recipe(_json_object(args.recipe, "Round7 recipe"), verify_source_file=True)
    task = build_native_retrieval_task(recipe, device="cpu", runtime_root=PROJECT)
    provider = task["provider"]
    cases = tuple(provider.epoch_cases(501, task["engine"].config.seed))
    updates = task["engine"].epoch_update_slices(len(cases))
    print(json.dumps({
        "status": "prepared_no_training",
        "family": "native_retrieval_v1",
        "dataset": recipe["dataset"],
        "treatment": recipe["treatment"],
        "absolute_start_epoch": 501,
        "source_checkpoint": task["source_checkpoint"],
        "source_checkpoint_sha256": task["source_checkpoint_sha256"],
        "source_sampling_dataset_id": task["identity"]["sampling_identity"]["source_sampling_dataset_id"],
        "arm": "A2-D",
        "training_cases_per_epoch": len(cases),
        "optimizer_updates_per_epoch": len(updates),
        "case_counts_per_update": [item.stop - item.start for item in updates],
        "execution_cadence": recipe["execution"],
        "direct_environment_value_reader": False,
        "provider_identity_sha256": canonical_sha256(task["provider_identity"]),
    }, sort_keys=True, allow_nan=False))
    return 0


def _fit_command(args: argparse.Namespace) -> int:
    if args.dry_run:
        if args.device != "cpu" or args.yes or args.output is not None:
            raise ValueError("Round7 dry-run is CPU-only, read-only, and cannot accept --yes or --output.")
        recipe = validate_native_retrieval_recipe(_json_object(args.recipe, "Round7 recipe"), verify_source_file=True)
        print(json.dumps({
            "status": "dry_run_validated_no_provider_no_training",
            "dataset": recipe["dataset"], "treatment": recipe["treatment"],
            "source_checkpoint": recipe["source_checkpoint"],
            "source_checkpoint_sha256": recipe["source_checkpoint_sha256"],
            "allowed_review_stops": list(REVIEW_STOPS),
            "direct_environment_value_reader": False,
        }, sort_keys=True, allow_nan=False))
        return 0
    if not args.yes:
        raise ValueError("A Round7 optimizer start requires explicit --yes after a positive cost admission.")
    if args.output is None or args.admission_receipt is None:
        raise ValueError("A Round7 fit requires --output and its positive --admission-receipt.")
    if args.device != "cuda:0":
        raise ValueError("A real Round7 optimizer continuation must use the specifically authorized GPU2 lane.")
    if type(args.stop_after) is not int or args.stop_after not in REVIEW_STOPS:
        raise ValueError("Round7 stops are restricted to the declared e600/e1000/e1500/e2500 reviews.")
    recipe = validate_native_retrieval_recipe(_json_object(args.recipe, "Round7 recipe"), verify_source_file=True)
    _validate_admission(args.admission_receipt, recipe, stop_after=args.stop_after)
    _round_root, resource_budget = _validate_round_resources(
        args.admission_receipt, args.output, recipe, stop_after=args.stop_after,
    )
    _check_cuda_lane(
        args.device,
        minimum_free_gpu_bytes=resource_budget["minimum_free_gpu_bytes"],
        minimum_free_gpu_fraction=resource_budget["minimum_free_gpu_fraction"],
    )
    if not torch.cuda.is_available():
        raise RuntimeError("The admitted GPU2 CUDA device is unavailable.")
    torch.cuda.set_device(0)

    from honf_runtime.run_layout import resolve_checkpoint

    output = args.output.expanduser().resolve()
    if args.resume is not None and args.branch_from_checkpoint is not None:
        raise ValueError("Round7 cannot branch and exact-resume in one invocation.")
    if args.resume is None:
        if args.branch_from_checkpoint is None:
            raise ValueError("A fresh Round7 child must name the exact pinned D0 e500 checkpoint.")
        parent = Path(args.branch_from_checkpoint).expanduser().resolve(strict=True)
        if parent != Path(recipe["source_checkpoint"]).resolve(strict=True):
            raise ValueError("Round7 branch checkpoint differs from the recipe-pinned D0 e500 parent.")
        if output.exists() and any(output.iterdir()):
            raise ValueError("A fresh Round7 child requires a new, empty output directory.")
        try:
            parent.relative_to(output)
        except ValueError:
            pass
        else:
            raise ValueError("Round7 child output must be outside the protected D0 parent run.")
        branch = parent
        resume = None
    else:
        if not output.is_dir():
            raise ValueError("Round7 exact resume requires the child run directory.")
        resume = resolve_checkpoint(output, args.resume)
        if resume != resolve_checkpoint(output, "latest"):
            raise ValueError("Round7 exact resume must use the child run's current latest checkpoint.")
        branch = None

    fit_started = time.perf_counter()
    try:
        task = build_native_retrieval_task(recipe, device=args.device, runtime_root=PROJECT)
        result = task["engine"].fit(
            task["model"], task["provider"], output,
            identity=task["identity"], arm="A2-D", stop_after=args.stop_after,
            resume_checkpoint=resume,
            branch_from_checkpoint=branch,
            native_retrieval_child=task["native_retrieval_child"],
        )
    except BaseException:
        _append_fit_resource_usage(
            _round_root, recipe, output=output, started_perf=fit_started, status="failed",
        )
        raise
    _append_fit_resource_usage(
        _round_root, recipe, output=output, started_perf=fit_started, status="completed", result=result,
    )
    print(json.dumps({"status": "completed", **result}, sort_keys=True, default=str, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    recipe_parser = subparsers.add_parser("make-recipe", help="seal a D-control or G-native-M recipe to exact D0 e500")
    recipe_parser.add_argument("--source-checkpoint", type=Path, required=True)
    recipe_parser.add_argument("--treatment", choices=("D-control", "G-native-M"), required=True)
    recipe_parser.add_argument("--backend", choices=("full_direct", *sorted(NATIVE_RETRIEVAL_NATIVE_BACKENDS)))
    recipe_parser.add_argument("--tau-q", type=float)
    recipe_parser.add_argument("--output", type=Path, required=True)
    recipe_parser.set_defaults(handler=_recipe_command)

    prepare_parser = subparsers.add_parser("prepare", help="CPU-only provider and cadence verification; no optimizer")
    prepare_parser.add_argument("--recipe", type=Path, required=True)
    prepare_parser.add_argument("--device", default="cpu")
    prepare_parser.set_defaults(handler=_prepare_command)

    fit_parser = subparsers.add_parser("fit", help="run only an admitted segmented continuation")
    fit_parser.add_argument("--recipe", type=Path, required=True)
    fit_parser.add_argument("--device", default="cpu")
    fit_parser.add_argument("--output", type=Path)
    fit_parser.add_argument("--stop-after", type=int, default=600)
    fit_parser.add_argument("--branch-from-checkpoint", type=Path)
    fit_parser.add_argument("--resume", type=Path)
    fit_parser.add_argument("--admission-receipt", type=Path)
    fit_parser.add_argument("--dry-run", action="store_true")
    fit_parser.add_argument("--yes", action="store_true")
    fit_parser.set_defaults(handler=_fit_command)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
