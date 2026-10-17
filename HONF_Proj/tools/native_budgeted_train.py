#!/usr/bin/env python3
"""Round8 learned budgeted native-dispatch entrypoint.

This is a separate research family from Round7. It reuses the segmented D0
provider and maintained TrainingEngine; only its own research admission can
authorize optimizer updates. Startup speed is recorded but is not a launch
condition.
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
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for _directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src",
                   PROJECT / "Case_WindFarm/src", PROJECT / "tools"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import torch

from honf_runtime.native_budgeted_factory import build_native_budgeted_task
from honf_runtime.native_budgeted_protocol import (
    NATIVE_BUDGETED_BACKENDS,
    NATIVE_BUDGETED_TREATMENTS,
    canonical_sha256,
    file_sha256,
    make_native_budgeted_recipe,
    validate_native_budgeted_recipe,
    validate_round8_research_admission,
)
from honf_runtime.source_read_protocol import configure_source_read_numerics

GPU2_UUID = "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39"
REVIEW_STOPS = (600, 1000, 1500, 2500)
ROUND_ROOT_PREFIX = Path("/data2/wanglz/ModularDT/diagnostics")


def _json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return value


def _write_new_json(path: Path, value: Any) -> Path:
    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite a Round8 recipe: {target}")
    _verify_round_root(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def _verify_round_root(path: Path) -> Path:
    if not os.path.ismount("/data2"):
        raise RuntimeError("Round8 payload writes require the currently mounted /data2 filesystem.")
    root = path.resolve()
    try:
        root.relative_to(ROUND_ROOT_PREFIX.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("Round8 generated payloads must remain below /data2/wanglz/ModularDT/diagnostics.") from exc
    return root


def _regular_file_bytes(path: Path, *, exclude: Path | None = None) -> int:
    if not path.exists():
        return 0
    total = 0
    excluded = None if exclude is None else exclude.resolve()
    for directory, names, files in os.walk(path, followlinks=False):
        parent = Path(directory)
        names[:] = [name for name in names if not (parent / name).is_symlink()]
        for name in files:
            item = parent / name
            if item.is_symlink() or (excluded is not None and item.resolve().is_relative_to(excluded)):
                continue
            try:
                if item.is_file():
                    total += item.stat().st_size
            except FileNotFoundError as exc:
                raise RuntimeError(f"Round8 storage tree changed during inventory: {item}") from exc
    return total


def _reservation_for_recipe(recipe: Mapping[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    binding = recipe.get("research_admission")
    if not isinstance(binding, Mapping):
        raise ValueError("Round8 recipe has no pinned research-admission receipt.")
    admission_path = Path(str(binding.get("path", ""))).resolve(strict=True)
    admission = _json_object(admission_path, "Round8 research-admission receipt")
    resource = admission.get("resource_reservation")
    if not isinstance(resource, Mapping) or set(resource) != {"path", "sha256"}:
        raise ValueError("Round8 admission has no exact resource-reservation pin.")
    resource_path = Path(str(resource.get("path", ""))).resolve(strict=True)
    if file_sha256(resource_path) != resource.get("sha256"):
        raise ValueError("Round8 resource-reservation receipt changed after admission.")
    reservation = _json_object(resource_path, "Round8 resource reservation")
    if (reservation.get("schema") != "round8_resource_reservation_v1"
            or reservation.get("gpu_uuid") != GPU2_UUID
            or reservation.get("physical_gpu") != 2
            or reservation.get("root") not in (None, str(resource_path.parent.parent.resolve()))):
        raise ValueError("Round8 resource reservation is not bound to physical GPU2 and its own root.")
    root = resource_path.parent.parent.resolve(strict=True)
    _verify_round_root(root)
    if (admission_path.parent != root / "receipts"
            or file_sha256(admission_path) != binding.get("sha256")):
        raise ValueError("Round8 recipe/admission files are not in the same sealed round receipts tree.")
    return dict(reservation), root, admission_path


def _check_round_resources(
    recipe: Mapping[str, Any], output: Path, stop_after: int, *, materialize: bool,
    start_after: int = 500,
) -> dict[str, Any]:
    reservation, root, admission_path = _reservation_for_recipe(recipe)
    output = output.expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError("Round8 fit output must remain under its pinned /data2 round root.") from exc
    try:
        Path(recipe["source_checkpoint"]).resolve(strict=True).relative_to(output)
    except ValueError:
        pass
    else:
        raise ValueError("Round8 output cannot be nested inside the protected D0 e500 source run.")

    source_size = Path(recipe["source_checkpoint"]).stat().st_size
    declared_checkpoints = recipe["execution"].get("checkpoint_epochs", ())
    if not isinstance(declared_checkpoints, (tuple, list)):
        raise ValueError("Round8 inherited execution policy has no checkpoint cadence list.")
    if type(start_after) is not int or not 500 <= start_after < stop_after:
        raise ValueError("Round8 segment must advance at least one absolute epoch after its current checkpoint.")
    checkpoint_count = sum(1 for epoch in declared_checkpoints if start_after < int(epoch) <= stop_after)
    if stop_after not in declared_checkpoints:
        checkpoint_count += 1
    # A checkpoint carries model plus named optimizer/RNG state, similar in size
    # to D0 e500. Include one latest state and one atomic-write temporary copy.
    estimated_persistent = int(math.ceil(source_size * (checkpoint_count + 2) * 1.10))
    estimated_temporary = int(math.ceil(source_size * 1.25)) if materialize else 0
    hard_retained = reservation.get("retained_budget_bytes")
    hard_temporary = reservation.get("temporary_budget_bytes")
    hard_peak = reservation.get("peak_new_bytes")
    minimum_free = reservation.get("minimum_free_bytes")
    if any(type(value) is not int or value <= 0 for value in (
            hard_retained, hard_temporary, hard_peak, minimum_free)):
        raise ValueError("Round8 reservation must declare retained, temporary, peak and headroom byte limits.")
    if hard_peak > hard_retained + hard_temporary:
        raise ValueError("Round8 peak payload limit exceeds the combined retained and temporary caps.")
    temporary_root = root / "temporary"
    retained_used = _regular_file_bytes(root, exclude=temporary_root)
    temporary_used = _regular_file_bytes(temporary_root)
    if retained_used > hard_retained or temporary_used > hard_temporary:
        raise OSError("Round8 payload already exceeds its retained or temporary hard cap.")
    if retained_used + estimated_persistent > hard_retained:
        raise OSError("Round8 projected checkpoints exceed the retained hard cap.")
    if temporary_used + estimated_temporary > hard_temporary:
        raise OSError("Round8 projected atomic-write space exceeds the temporary hard cap.")
    projected_peak = retained_used + temporary_used + estimated_persistent + estimated_temporary
    if projected_peak > hard_peak:
        raise OSError("Round8 current plus projected payload exceeds its sealed peak cap.")
    free = shutil.disk_usage("/data2").free
    if free - estimated_persistent - estimated_temporary < minimum_free:
        raise OSError("Round8 projected materialization would violate /data2 free-space headroom.")

    start = reservation.get("start_unix")
    elapsed_limit = reservation.get("elapsed_limit_seconds")
    associated_limit = reservation.get("gpu_associated_limit_seconds")
    closeout = reservation.get("reserved_closeout_seconds")
    if (type(start) not in (int, float)
            or any(type(value) is not int or value <= 0 for value in (elapsed_limit, associated_limit))
            or type(closeout) is not int or closeout < 0):
        raise ValueError("Round8 reservation lacks a valid elapsed/GPU deadline and closeout reserve.")
    elapsed = max(0.0, time.time() - float(start))
    if elapsed + closeout >= elapsed_limit:
        raise RuntimeError("Round8 elapsed window reached its reserved closeout period.")
    usage_path = root / "receipts" / "round8_gpu_usage.jsonl"
    recorded = 0.0
    if usage_path.exists():
        for line_number, line in enumerate(usage_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, Mapping) or row.get("schema") != "round8_gpu_usage_v1"
                    or row.get("gpu_uuid") != GPU2_UUID
                    or type(row.get("associated_seconds")) not in (int, float)
                    or not math.isfinite(float(row["associated_seconds"])) or row["associated_seconds"] < 0):
                raise ValueError(f"Round8 GPU usage line {line_number} is malformed or for another device.")
            recorded += float(row["associated_seconds"])
    conservative_used = max(elapsed, recorded)
    associated_remaining = associated_limit - conservative_used
    if associated_remaining <= 0:
        raise RuntimeError("Round8 physical GPU2 allowance is exhausted by elapsed time/usage receipts.")

    admission = _json_object(admission_path, "Round8 research admission")
    evidence = admission.get("evidence")
    resources = evidence.get("reserved_resources") if isinstance(evidence, Mapping) else None
    if not isinstance(resources, Mapping) or resources.get("passed") is not True:
        raise ValueError("Round8 research admission has no passing reserved-resources check.")
    full_arm_elapsed = resources.get("estimated_elapsed_seconds")
    full_arm_gpu = resources.get("estimated_associated_seconds")
    full_arm_persistent = resources.get("estimated_persistent_bytes")
    full_arm_temporary = resources.get("estimated_temporary_bytes")
    if any(type(value) is not int or value <= 0 for value in (
            full_arm_elapsed, full_arm_gpu, full_arm_persistent, full_arm_temporary)):
        raise ValueError("Round8 reserved-resources evidence must include full-arm time and storage estimates.")
    if stop_after > 2500 or start_after < 500:
        raise ValueError("Round8 segment is outside the absolute e501–e2500 continuation window.")
    remaining_epochs = stop_after - start_after
    segment_elapsed = math.ceil(full_arm_elapsed * remaining_epochs / 2000)
    segment_gpu = math.ceil(full_arm_gpu * remaining_epochs / 2000)
    if (segment_elapsed >= elapsed_limit - elapsed - closeout
            or segment_gpu >= associated_remaining
            or estimated_persistent > full_arm_persistent
            or estimated_temporary > full_arm_temporary):
        raise RuntimeError("This segment exceeds its admitted Round8 time or storage estimate.")
    return {
        "root": str(root),
        "resource_reservation": str(root / "receipts" / "resource_reservation.json"),
        "resource_reservation_sha256": file_sha256(root / "receipts" / "resource_reservation.json"),
        "free_bytes_before": free,
        "retained_bytes_before": retained_used,
        "temporary_bytes_before": temporary_used,
        "estimated_persistent_bytes": estimated_persistent,
        "estimated_temporary_bytes": estimated_temporary,
        "projected_peak_bytes": projected_peak,
        "elapsed_seconds_since_round_start": elapsed,
        "elapsed_seconds_remaining_before_closeout": elapsed_limit - elapsed - closeout,
        "recorded_gpu_seconds": recorded,
        "conservative_gpu_seconds_used": conservative_used,
        "gpu_associated_seconds_remaining": associated_remaining,
        "segment_epochs": remaining_epochs,
        "estimated_segment_elapsed_seconds": segment_elapsed,
        "estimated_segment_gpu_seconds": segment_gpu,
        "admitted_full_arm_elapsed_seconds": full_arm_elapsed,
        "admitted_full_arm_gpu_seconds": full_arm_gpu,
        "admitted_full_arm_persistent_bytes": full_arm_persistent,
        "admitted_full_arm_temporary_bytes": full_arm_temporary,
    }


def _check_cuda_lane(resources: Mapping[str, Any], device: str) -> None:
    # Do not call torch.cuda.is_available/properties/memory APIs before this
    # physical-device mapping is established.
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU2_UUID:
        raise ValueError("Round8 fit is restricted to logical cuda:0 mapped to the exact physical GPU2 UUID.")
    # The current Round8 reservation does not set a smaller floor; preserve
    # margin for complete source-plus-router AdamW memory.
    floor_bytes = 8 * 1024**3
    floor_fraction = 0.20
    if resources.get("minimum_free_gpu_bytes", floor_bytes) != floor_bytes:
        raise ValueError("Round8 GPU free-memory floor must remain the declared 8 GiB margin.")
    command = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    found = None
    for line in command.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 3 and fields[0] == GPU2_UUID:
            found = (int(fields[1]) * 1024**2, int(fields[2]) * 1024**2)
            break
    if found is None or found[0] < max(floor_bytes, math.ceil(floor_fraction * found[1])):
        raise RuntimeError("Round8 physical GPU2 is missing or below the declared free-memory floor.")


def _append_gpu_usage(root: Path, recipe: Mapping[str, Any], output: Path,
                      started: float, status: str, result: Mapping[str, Any] | None = None) -> None:
    elapsed = max(0.0, time.perf_counter() - started)
    completed_epoch = None if result is None else result.get("completed_epoch")
    optimizer_updates = None if result is None else result.get("optimizer_updates")
    case_visits = None if result is None else result.get("case_visits")
    start_epoch = None if result is None else result.get("start_epoch")
    counts_source = "fit_result" if result is not None and (
        type(optimizer_updates) is int and type(case_visits) is int
    ) else None
    summary_path = output / "metrics" / "fit_summary.json"
    if (result is not None and summary_path.is_file() and type(completed_epoch) is int):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = None
        if isinstance(summary, Mapping) and summary.get("completed_epoch") == completed_epoch:
            summary_updates = summary.get("optimizer_updates")
            summary_visits = summary.get("case_visits")
            summary_start = summary.get("start_epoch")
            if type(summary_updates) is int and type(summary_visits) is int:
                optimizer_updates = summary_updates
                case_visits = summary_visits
                start_epoch = summary_start if type(summary_start) is int else start_epoch
                counts_source = "metrics/fit_summary.json"
    row = {
        "schema": "round8_gpu_usage_v1",
        "gpu_uuid": GPU2_UUID,
        "associated_seconds": elapsed,
        "status": status,
        "dataset": recipe["dataset"],
        "treatment": recipe["treatment"],
        "recipe_sha256": canonical_sha256(recipe),
        "output": str(output),
        "completed_epoch": completed_epoch,
        "segment_start_epoch": start_epoch,
        "optimizer_updates": optimizer_updates,
        "case_visits": case_visits,
        "usage_counts_source": counts_source,
    }
    target = root / "receipts" / "round8_gpu_usage.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _make_recipe(args: argparse.Namespace) -> int:
    admission_value = _json_object(args.admission, "Round8 research admission")
    backend = args.backend or admission_value.get("native_backend")
    checked_admission = validate_round8_research_admission(
        admission_value,
        source_checkpoint=args.source_checkpoint,
        source_checkpoint_sha256=file_sha256(args.source_checkpoint),
        native_backend=backend,
        verify_files=True,
    )
    membership_bias = 0.0 if args.membership_bias is None else args.membership_bias
    lambda0 = 0.0 if args.lambda0 is None else args.lambda0
    if args.treatment == "G-budgeted-M":
        calibration = checked_admission["evidence"]["useful_router_gradient"]["calibration"]
        calibrated_bias = float(calibration["membership_bias"])
        calibrated_lambda = float(calibration["lambda0"])
        if ((args.membership_bias is not None and args.membership_bias != calibrated_bias)
                or (args.lambda0 is not None and args.lambda0 != calibrated_lambda)):
            raise ValueError("G8 recipe calibration flags must exactly match the pinned input-only TRAIN receipt.")
        membership_bias, lambda0 = calibrated_bias, calibrated_lambda
    recipe = make_native_budgeted_recipe(
        args.source_checkpoint,
        treatment=args.treatment,
        research_admission=args.admission,
        native_backend=backend,
        membership_bias=membership_bias,
        lambda0=lambda0,
        routing_seed=args.routing_seed,
    )
    target = _write_new_json(args.output, recipe)
    print(json.dumps({
        "status": "recipe_written",
        "family": recipe["family"],
        "dataset": recipe["dataset"],
        "treatment": recipe["treatment"],
        "source_checkpoint": recipe["source_checkpoint"],
        "recipe_sha256": canonical_sha256(recipe),
        "recipe_path": str(target),
        "environment_value_reader": False,
    }, sort_keys=True, allow_nan=False))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    if args.device != "cpu":
        raise ValueError("Round8 provider preparation is CPU-only; fit owns the physical GPU lane.")
    recipe = validate_native_budgeted_recipe(
        _json_object(args.recipe, "Round8 recipe"), verify_source_file=True, verify_admission=True,
    )
    _check_round_resources(recipe, args.output_root, args.stop_after, materialize=False)
    task = build_native_budgeted_task(recipe, device="cpu", runtime_root=PROJECT)
    provider = task["provider"]
    cases = tuple(provider.epoch_cases(501, task["engine"].config.seed))
    slices = task["engine"].epoch_update_slices(len(cases))
    print(json.dumps({
        "status": "prepared_without_training_or_output",
        "family": recipe["family"],
        "dataset": recipe["dataset"],
        "treatment": recipe["treatment"],
        "absolute_start_epoch": 501,
        "source_checkpoint": task["source_checkpoint"],
        "source_checkpoint_sha256": task["source_checkpoint_sha256"],
        "source_sampling_dataset_id": task["identity"]["sampling_identity"]["source_sampling_dataset_id"],
        "arm": "A2-D",
        "train_cases_per_epoch": len(cases),
        "optimizer_updates_per_epoch": len(slices),
        "case_counts_per_update": [row.stop - row.start for row in slices],
        "primary_queries_per_case": task["identity"]["training_cadence"]["primary_queries_per_case"],
        "response_cadence": task["identity"]["training_cadence"]["response_cadence"],
        "optimizer_group_names": [row.name for row in provider.optimizer_groups(
            task["model"], "A2-D", "joint")],
        "source_provider_identity_sha256": canonical_sha256(task["provider_identity"]),
        "environment_value_reader": False,
    }, sort_keys=True, allow_nan=False))
    return 0


def _fit(args: argparse.Namespace) -> int:
    if args.dry_run:
        if args.device != "cpu" or args.yes or args.output is not None or args.resume is not None:
            raise ValueError("Round8 dry-run is CPU-only, read-only, and cannot accept fit/output options.")
        recipe = validate_native_budgeted_recipe(
            _json_object(args.recipe, "Round8 recipe"), verify_source_file=True, verify_admission=True,
        )
        print(json.dumps({
            "status": "dry_run_validated_without_provider_or_optimizer",
            "dataset": recipe["dataset"],
            "treatment": recipe["treatment"],
            "recipe_sha256": canonical_sha256(recipe),
            "source_checkpoint": recipe["source_checkpoint"],
            "allowed_review_stops": list(REVIEW_STOPS),
            "startup_speed_gates_research_admission": False,
            "environment_value_reader": False,
        }, sort_keys=True, allow_nan=False))
        return 0
    if not args.yes:
        raise ValueError("Round8 optimizer launch requires --yes after this exact research receipt is admitted.")
    if args.output is None or args.stop_after not in REVIEW_STOPS:
        raise ValueError("Round8 fit requires --output and a declared e600/e1000/e1500/e2500 stop.")
    if args.resume is not None and args.branch_from_checkpoint is not None:
        raise ValueError("Round8 cannot branch and exact-resume in the same invocation.")
    if args.device != "cuda:0":
        raise ValueError("Round8 optimizer updates must use physical GPU2 through logical cuda:0.")
    recipe = validate_native_budgeted_recipe(
        _json_object(args.recipe, "Round8 recipe"), verify_source_file=True, verify_admission=True,
    )
    output = args.output.expanduser().resolve()
    if args.branch_from_checkpoint is not None:
        parent = args.branch_from_checkpoint.expanduser().resolve(strict=True)
        if parent != Path(recipe["source_checkpoint"]).resolve(strict=True):
            raise ValueError("Round8 fresh branch must use its exact recipe-pinned D0 e500 checkpoint.")
        if output.exists() and any(output.iterdir()):
            raise ValueError("A fresh Round8 fit requires a new, empty output directory.")
        branch = parent
        resume = None
        start_after = 500
    elif args.resume is None:
        raise ValueError("Round8 launch requires either the exact D0 e500 parent or the current run's latest resume.")
    else:
        if not output.is_dir():
            raise ValueError("Round8 exact resume requires its existing output directory.")
        from honf_runtime.compat import load_trusted_checkpoint
        from honf_runtime.run_layout import resolve_checkpoint

        branch = None
        resume = resolve_checkpoint(output, args.resume)
        if resume != resolve_checkpoint(output, "latest"):
            raise ValueError("Round8 exact resume must use the run's current latest checkpoint.")
        resume_payload = load_trusted_checkpoint(resume, map_location="cpu")
        start_after = resume_payload.get("epoch")
        if type(start_after) is not int:
            raise ValueError("Round8 latest checkpoint has no exact absolute completed epoch.")

    # Set the inherited D0 numerics before any CUDA availability/device query.
    configure_source_read_numerics(recipe["base_recipe"])
    resources = _check_round_resources(
        recipe, output, args.stop_after, materialize=True, start_after=start_after,
    )
    _check_cuda_lane(resources, args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("Round8 admitted physical GPU2 CUDA device is unavailable.")
    torch.cuda.set_device(0)

    root = Path(resources["root"])
    started = time.perf_counter()
    task = None
    try:
        task = build_native_budgeted_task(recipe, device=args.device, runtime_root=PROJECT)
        result = task["engine"].fit(
            task["model"], task["provider"], output,
            identity=task["identity"], arm="A2-D", stop_after=args.stop_after,
            branch_from_checkpoint=branch,
            resume_checkpoint=resume,
            native_budgeted_child=task["native_budgeted_child"],
        )
    except BaseException:
        _append_gpu_usage(root, recipe, output, started, "failed")
        raise
    _append_gpu_usage(root, recipe, output, started, "completed", result)
    print(json.dumps({"status": "completed", **result}, sort_keys=True, default=str, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    recipe_parser = subparsers.add_parser("make-recipe", help="seal a D8/G8 child to exact D0 e500 and Round8 admission")
    recipe_parser.add_argument("--source-checkpoint", type=Path, required=True)
    recipe_parser.add_argument("--treatment", choices=sorted(NATIVE_BUDGETED_TREATMENTS), required=True)
    recipe_parser.add_argument("--admission", type=Path, required=True)
    recipe_parser.add_argument("--backend", choices=sorted(NATIVE_BUDGETED_BACKENDS))
    recipe_parser.add_argument("--membership-bias", type=float)
    recipe_parser.add_argument("--lambda0", type=float)
    recipe_parser.add_argument("--routing-seed", type=int, default=0)
    recipe_parser.add_argument("--output", type=Path, required=True)
    recipe_parser.set_defaults(handler=_make_recipe)

    prepare_parser = subparsers.add_parser("prepare", help="CPU-only source/provider/optimizer preflight with no run output")
    prepare_parser.add_argument("--recipe", type=Path, required=True)
    prepare_parser.add_argument("--device", default="cpu")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--stop-after", type=int, choices=REVIEW_STOPS, default=600)
    prepare_parser.set_defaults(handler=_prepare)

    fit_parser = subparsers.add_parser("fit", help="run one admitted segmented D8/G8 TrainingEngine continuation")
    fit_parser.add_argument("--recipe", type=Path, required=True)
    fit_parser.add_argument("--device", default="cuda:0")
    fit_parser.add_argument("--output", type=Path)
    fit_parser.add_argument("--stop-after", type=int, choices=REVIEW_STOPS, default=600)
    fit_parser.add_argument("--branch-from-checkpoint", type=Path)
    fit_parser.add_argument("--resume", type=Path)
    fit_parser.add_argument("--yes", action="store_true")
    fit_parser.add_argument("--dry-run", action="store_true")
    fit_parser.set_defaults(handler=_fit)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
