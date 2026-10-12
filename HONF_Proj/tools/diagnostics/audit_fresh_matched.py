"""Read-only stream and measured-cost audits for fresh matched P/R/R2 runs.

The fresh runner's RunStore writes identities under ``configs/`` and histories
under ``metrics/``. This audit accepts that layout and the earlier flat layout,
but it never guesses when both layouts contain different evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

HONF_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = HONF_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from joint_regional_train import (
    FRESH_MATCHED_COMPARISON_PROTOCOL,
    FRESH_MATCHED_MODES,
    FRESH_MATCHED_SELECTION_EPOCHS,
    _read_fresh_comparison_manifest,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"), allow_nan=False)


def _audit_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _read_json(path: Path, *, description: str) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description} JSON at {path}: {exc}") from exc


def resolve_run_artifacts(output_dir: str | Path) -> tuple[Path, Path, str]:
    """Resolve one complete RunStore or legacy flat artifact pair."""
    output = Path(output_dir).expanduser().resolve()
    candidates = {
        "runstore": (output / "configs" / "experiment_identity.json",
                     output / "metrics" / "history.json"),
        "flat": (output / "experiment_identity.json", output / "history.json"),
    }
    present = {name: tuple(path.is_file() for path in pair) for name, pair in candidates.items()}
    complete = {name for name, bits in present.items() if all(bits)}
    partial = {name for name, bits in present.items() if any(bits) and not all(bits)}

    if len(complete) == 2:
        (run_identity, run_history) = candidates["runstore"]
        (flat_identity, flat_history) = candidates["flat"]
        if (run_identity.read_bytes() != flat_identity.read_bytes()
                or run_history.read_bytes() != flat_history.read_bytes()):
            raise ValueError(f"Conflicting RunStore and legacy flat artifacts exist in {output}.")
        return run_identity, run_history, "runstore_and_flat_identical"
    if len(complete) == 1:
        selected = next(iter(complete))
        if partial:
            raise ValueError(f"A partial second artifact layout makes {output} ambiguous.")
        identity_path, history_path = candidates[selected]
        return identity_path, history_path, selected
    if partial:
        if len(partial) == 2:
            raise ValueError(f"Identity and history are split across incomplete layouts in {output}.")
        selected = next(iter(partial))
        missing = [str(path) for path, exists in zip(candidates[selected], present[selected]) if not exists]
        raise FileNotFoundError(f"Incomplete {selected} fresh-run artifacts in {output}; missing {missing}.")
    raise FileNotFoundError(f"Fresh run lacks identity and history artifacts in {output}.")


def _publish_json_once(path: Path, payload: dict[str, Any]) -> None:
    """Publish immutable evidence, accepting an existing byte-equivalent receipt."""
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _read_json(path, description="existing audit receipt")
        if existing != payload:
            raise FileExistsError(f"A different immutable audit receipt already exists at {path}.")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_fresh_matched_streams(manifest_path: str | Path) -> dict[str, Any]:
    """Validate the observed epoch-1 streams and write a separate audit receipt."""
    manifest, manifest_path = _read_fresh_comparison_manifest(manifest_path)
    seal_path = manifest_path.parent / "comparison_seal.json"
    if not seal_path.is_file():
        raise FileNotFoundError("Run prepare-matched before reviewing training stream audits.")
    seal = _read_json(seal_path, description="fresh matched comparison seal")
    if seal.get("schema") != "fresh_matched_initialization_seal_v1":
        raise ValueError("Fresh matched preparation seal is missing or malformed.")
    if (seal.get("comparison_manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            or seal.get("task") != manifest["task"]
            or seal.get("dataset_protocol") != manifest["dataset_protocol"]
            or seal.get("source_config_sha256") != manifest.get("source_config_sha256")
            or set(seal.get("arms", {})) != set(FRESH_MATCHED_MODES)):
        raise ValueError("Fresh matched preparation seal does not bind the current campaign manifest.")
    if type(manifest.get("seed")) is not int:
        raise TypeError("Fresh matched allocated manifest must bind an exact integer seed.")

    receipts: dict[str, Any] = {}
    history_hashes: dict[str, str] = {}
    identity_bindings: dict[str, Any] = {}
    artifact_layouts: dict[str, str] = {}
    epoch_rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    fields = (
        "case_order_sha256", "case_visits", "optimizer_updates", "microbatches",
        "query_sampling_sha256", "query_sampling_by_update_sha256", "observed_mask_sha256",
        "mask_sha256_by_update", "observed_denominators_sha256",
        "observed_denominators_by_update", "observed_denominator_totals",
        "effective_cases", "microbatch_cases", "sampling_dataset_id", "sampler_namespace_sha256",
    )
    for mode in FRESH_MATCHED_MODES:
        arm = manifest["arms"][mode]
        output = Path(arm["output_dir"])
        identity_path, history_path, layout = resolve_run_artifacts(output)
        identity = _read_json(identity_path, description=f"{mode} experiment identity")
        rows = _read_json(history_path, description=f"{mode} epoch history")
        if not isinstance(rows, list):
            raise TypeError(f"Fresh matched {mode} history must be a JSON list of epoch rows.")
        if not rows or any(not isinstance(item, dict) for item in rows):
            raise TypeError(f"Fresh matched {mode} history must contain completed epoch objects.")
        stream_rows: list[dict[str, Any]] = []
        for expected_epoch, saved_row in enumerate(rows, start=1):
            if type(saved_row.get("epoch")) is not int or saved_row["epoch"] != expected_epoch:
                raise ValueError(f"Fresh matched {mode} history must contain the complete ordered range 1..N.")
            for digest_name in ("case_order_sha256", "query_sampling_sha256"):
                if not _is_sha256(saved_row.get(digest_name)):
                    raise ValueError(f"Fresh matched {mode} epoch {expected_epoch} has malformed {digest_name}.")
            counters = {name: saved_row.get(name) for name in ("case_visits", "optimizer_updates", "microbatches")}
            if (any(type(value) is not int for value in counters.values())
                    or counters["case_visits"] < 1 or counters["optimizer_updates"] < 1
                    or counters["microbatches"] < counters["optimizer_updates"]):
                raise ValueError(f"Fresh matched {mode} epoch {expected_epoch} has malformed stream counters.")
            stream_rows.append({
                "epoch": expected_epoch,
                "case_order_sha256": saved_row["case_order_sha256"],
                "query_sampling_sha256": saved_row["query_sampling_sha256"],
                **counters,
            })
        epoch_rows_by_mode[mode] = stream_rows
        row = rows[0]
        audit = row.get("observed_stream_audit")
        marker = identity.get("fresh_initialization")
        recipe = identity.get("recipe")
        provider_identity = identity.get("provider_identity", {})
        sampler_namespace = marker.get("sampler_namespace") if isinstance(marker, dict) else None
        if (not isinstance(marker, dict) or not isinstance(audit, dict)
                or audit.get("schema") != "observed_fresh_training_stream_v1"
                or not isinstance(recipe, dict)
                or recipe.get("mode") != mode
                or recipe.get("run_id") != arm.get("run_id")
                or recipe.get("task") != manifest["task"]
                or type(recipe.get("seed")) is not int
                or recipe.get("seed") != manifest["seed"]
                or recipe.get("dataset_protocol") != manifest["dataset_protocol"]
                or recipe.get("fresh_matched_comparison") is not True
                or not isinstance(provider_identity, dict)
                or not isinstance(sampler_namespace, dict)
                or provider_identity.get("fresh_matched_comparison") is not True
                or marker.get("active_mode") != mode):
            raise ValueError(f"Fresh matched {mode} saved recipe/identity does not bind the allocated run, task, seed, and dataset.")
        expected_provider_dataset = "ThermalChannel" if manifest["task"] == "thermal" else "WindFarm"
        provider_dataset = provider_identity.get("dataset") or provider_identity.get("task")
        provider_arm = provider_identity.get("arm") or provider_identity.get("mode")
        if (provider_dataset != expected_provider_dataset
                or provider_arm != mode
                or marker.get("provider_identity_sha256") != _audit_sha256(provider_identity)):
            raise ValueError(f"Fresh matched {mode} provider identity does not bind the allocated dataset.")
        preparation_arm = seal["arms"][mode]
        if (marker.get("common_initialization_sha256") != seal.get("common_initialization_sha256")
                or marker.get("provider_scale_binding_sha256") != seal.get("provider_scale_binding_sha256")
                or sampler_namespace != preparation_arm.get("sampler_namespace")
                or sampler_namespace.get("seed") != manifest["seed"]
                or sampler_namespace.get("dataset_id") != identity.get("sampling_dataset_id")
                or _audit_sha256(sampler_namespace) != seal.get("sampler_namespace_sha256")
                or preparation_arm.get("run_id") != arm["run_id"]
                or preparation_arm.get("recipe_sha256") != arm.get("recipe_sha256")):
            raise ValueError(f"Fresh matched {mode} training identity differs from its prepared seal.")
        update_count = row.get("optimizer_updates")
        case_visits = row.get("case_visits")
        microbatches = row.get("microbatches")
        if (type(update_count) is not int or update_count < 1
                or type(case_visits) is not int or case_visits < 1
                or type(microbatches) is not int or microbatches < update_count
                or type(audit.get("optimizer_updates")) is not int
                or type(audit.get("case_visits")) is not int
                or type(audit.get("microbatches")) is not int
                or type(audit.get("effective_cases")) is not int
                or type(audit.get("microbatch_cases")) is not int
                or not isinstance(audit.get("sampling_dataset_id"), str)
                or audit.get("optimizer_updates") != update_count
                or audit.get("case_visits") != case_visits
                or audit.get("microbatches") != microbatches
                or audit.get("effective_cases") != identity.get("engine_config", {}).get("effective_cases")
                or audit.get("microbatch_cases") != identity.get("engine_config", {}).get("microbatch_cases")
                or audit.get("sampling_dataset_id") != identity.get("sampling_dataset_id")
                or audit.get("sampler_namespace_sha256") != _audit_sha256(marker.get("sampler_namespace"))):
            raise ValueError(f"Fresh matched {mode} observed stream counters differ from the saved engine identity.")
        for name in ("case_order_sha256", "query_sampling_sha256", "observed_mask_sha256",
                     "observed_denominators_sha256", "sampler_namespace_sha256"):
            if not _is_sha256(audit.get(name)):
                raise ValueError(f"Fresh matched {mode} observed stream digest {name} is malformed.")
        if (row.get("case_order_sha256") != audit["case_order_sha256"]
                or row.get("query_sampling_sha256") != audit["query_sampling_sha256"]):
            raise ValueError(f"Fresh matched {mode} history summary differs from its observed streams.")
        query_updates = audit.get("query_sampling_by_update_sha256")
        mask_updates = audit.get("mask_sha256_by_update")
        denominator_updates = audit.get("observed_denominators_by_update")
        if not all(isinstance(value, list) and len(value) == update_count for value in (
                query_updates, mask_updates, denominator_updates)):
            raise ValueError(f"Fresh matched {mode} per-update stream evidence has the wrong length.")
        expected_indices = list(range(update_count))
        if (not all(isinstance(item, dict) and type(item.get("update_index")) is int
                    for item in query_updates + mask_updates + denominator_updates)
                or [item.get("update_index") for item in query_updates if isinstance(item, dict)] != expected_indices
                or [item.get("update_index") for item in mask_updates if isinstance(item, dict)] != expected_indices
                or [item.get("update_index") for item in denominator_updates if isinstance(item, dict)]
                != expected_indices
                or not all(_is_sha256(item.get("sha256")) for item in query_updates if isinstance(item, dict))
                or not all(_is_sha256(item.get("sha256")) for item in mask_updates if isinstance(item, dict))):
            raise ValueError(f"Fresh matched {mode} per-update stream digests are malformed or reordered.")
        digest_sequence = hashlib.sha256(json.dumps([item["sha256"] for item in query_updates], separators=(",", ":"),
                                             allow_nan=False).encode("utf-8")).hexdigest()
        if digest_sequence != audit["query_sampling_sha256"]:
            raise ValueError(f"Fresh matched {mode} query stream aggregate differs from its update hashes.")
        if _audit_sha256(mask_updates) != audit["observed_mask_sha256"]:
            raise ValueError(f"Fresh matched {mode} mask stream aggregate differs from its update hashes.")
        if _audit_sha256(denominator_updates) != audit["observed_denominators_sha256"]:
            raise ValueError(f"Fresh matched {mode} denominator aggregate differs from its update records.")
        totals: dict[str, float] = {}
        for item in denominator_updates:
            if not isinstance(item, dict) or not isinstance(item.get("denominators"), dict):
                raise TypeError(f"Fresh matched {mode} denominator update record is malformed.")
            for name, value in item["denominators"].items():
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Fresh matched {mode} denominator {name} is not finite and nonnegative.")
                totals[name] = totals.get(name, 0.0) + float(value)
        if audit.get("observed_denominator_totals") != totals:
            raise ValueError(f"Fresh matched {mode} denominator totals differ from the observed update records.")
        receipts[mode] = {name: audit[name] for name in fields}
        history_hashes[mode] = hashlib.sha256(history_path.read_bytes()).hexdigest()
        artifact_layouts[mode] = layout
        identity_bindings[mode] = {
            "common_initialization_sha256": marker["common_initialization_sha256"],
            "provider_scale_binding_sha256": marker["provider_scale_binding_sha256"],
            "sampler_namespace": marker["sampler_namespace"],
        }
    reference = receipts["P"]
    differences = {
        mode: [name for name in fields if receipts[mode][name] != reference[name]]
        for mode in ("P-R", "P-R2")
    }
    if any(differences.values()):
        raise ValueError(f"Observed epoch-1 case/query/mask/denominator streams differ across arms: {differences}.")
    reference_epochs = epoch_rows_by_mode["P"]
    if any(epoch_rows_by_mode[mode] != reference_epochs for mode in ("P-R", "P-R2")):
        first_differences: dict[str, Any] = {}
        for mode in ("P-R", "P-R2"):
            for index, (reference_row, candidate_row) in enumerate(
                    zip(reference_epochs, epoch_rows_by_mode[mode]), start=1):
                changed = [name for name in reference_row if reference_row[name] != candidate_row[name]]
                if changed:
                    first_differences[mode] = {"epoch": index, "fields": changed}
                    break
            if mode not in first_differences:
                first_differences[mode] = {"epoch_range_matches": False}
        raise ValueError(f"Saved per-epoch case/query streams or counters differ across arms: {first_differences}.")
    completed_age = reference_epochs[-1]["epoch"]
    combined_history_sha256 = _audit_sha256([
        {"mode": mode, "history_sha256": history_hashes[mode]} for mode in FRESH_MATCHED_MODES
    ])
    receipt_path = manifest_path.parent / (
        f"matched_stream_audit_e{completed_age:04d}_{combined_history_sha256[:16]}.json")
    receipt = {
        "schema": "fresh_matched_observed_stream_comparison_v1",
        "comparison_manifest": str(manifest_path),
        "receipt_path": str(receipt_path),
        "comparison_seal_sha256": hashlib.sha256(seal_path.read_bytes()).hexdigest(),
        "task": manifest["task"],
        "dataset_protocol": manifest["dataset_protocol"],
        "seed": manifest["seed"],
        "shared_completed_age": completed_age,
        "per_epoch_streams_matched": True,
        "per_epoch_stream_summary_sha256": _audit_sha256(reference_epochs),
        "per_epoch_stream_fields": [
            "case_order_sha256", "query_sampling_sha256", "case_visits", "optimizer_updates", "microbatches",
        ],
        "full_mask_and_denominator_comparison_epochs": [1],
        "combined_history_sha256": combined_history_sha256,
        "arms": {
            mode: {
                "run_id": manifest["arms"][mode]["run_id"],
                "history_sha256": history_hashes[mode],
                "artifact_layout": artifact_layouts[mode],
                "observed_stream_audit": receipts[mode],
                "identity_binding": identity_bindings[mode],
            }
            for mode in FRESH_MATCHED_MODES
        },
        "matched": True,
        "optimizer_updates_observed": reference["optimizer_updates"],
        "case_visits_observed": reference["case_visits"],
        "optimizer_updates_in_this_command": 0,
    }
    _publish_json_once(receipt_path, receipt)
    return receipt


def forecast_fresh_matched_run(output_dir: str | Path, *, horizon: int = 2500) -> dict[str, Any]:
    """Forecast all remaining full TRAIN epochs and scheduled full-DEV reviews."""
    output = Path(output_dir).expanduser().resolve()
    identity_path, history_path, layout = resolve_run_artifacts(output)
    identity = _read_json(identity_path, description="fresh-run experiment identity")
    marker = identity.get("fresh_initialization")
    if not isinstance(marker, dict):
        raise TypeError("Forecast target is not an explicitly fresh matched run.")
    recipe = identity.get("recipe", {})
    if (recipe.get("fresh_matched_comparison") is not True
            or recipe.get("total_epochs") != horizon
            or recipe.get("comparison_protocol") != FRESH_MATCHED_COMPARISON_PROTOCOL):
        raise ValueError("Forecast horizon or recipe differs from the fresh segmented comparison.")
    rows = _read_json(history_path, description="fresh-run epoch history")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Fresh matched run has no completed epoch rows to forecast from.")
    training_rows = [float(row["train_seconds"]) for row in rows if row.get("train_seconds") is not None]
    validation_rows = [float(row["validation_seconds"]) for row in rows
                       if row.get("validation_seconds") is not None]
    if not training_rows or not validation_rows:
        return {
            "status": "incomplete_forecast_inputs",
            "reason": "Need measured TRAIN and full-DEV validation durations from at least one review epoch.",
            "completed_epoch": max(int(row["epoch"]) for row in rows),
            "optimizer_updates": sum(int(row["optimizer_updates"]) for row in rows),
            "case_visits": sum(int(row["case_visits"]) for row in rows),
            "artifact_layout": layout,
            "forecast_seconds": None,
            "optimizer_updates_in_this_command": 0,
        }
    if not all(math.isfinite(value) and value >= 0 for value in training_rows + validation_rows):
        raise ValueError("Fresh matched timing history contains a non-finite or negative duration.")
    recent_training = training_rows[-min(len(training_rows), 20):]
    recent_validation = validation_rows[-min(len(validation_rows), 5):]
    training_epoch_seconds = float(statistics.median(recent_training))
    validation_review_seconds = float(statistics.median(recent_validation))
    completed_epoch = max(int(row["epoch"]) for row in rows)
    remaining_epochs = max(0, horizon - completed_epoch)
    review_count = sum(1 for epoch in FRESH_MATCHED_SELECTION_EPOCHS
                       if completed_epoch < epoch <= horizon)
    last = rows[-1]
    best_row = min((row for row in rows if row.get("validation", {}).get("field_score") is not None),
                   key=lambda row: float(row["validation"]["field_score"]), default=None)
    provider = identity.get("provider_identity", {})
    provider_dataset = provider.get("dataset") or provider.get("task")
    if provider_dataset == "ThermalChannel":
        query_budget = provider.get("native_query_budget")
    else:
        objective = provider.get("objective", {})
        query_budget = {
            "primary_queries_per_case": provider.get("training", {}).get("primary_queries_per_case"),
            "role_query_counts": objective.get("role_query_counts"),
        }
    train_forecast = remaining_epochs * training_epoch_seconds
    validation_forecast = review_count * validation_review_seconds
    return {
        "status": "measured_epoch_and_validation_forecast",
        "run_id": recipe["run_id"],
        "active_mode": marker["active_mode"],
        "artifact_layout": layout,
        "completed_epoch": completed_epoch,
        "remaining_epochs": remaining_epochs,
        "review_epochs_remaining": review_count,
        "median_training_seconds_per_epoch_recent": training_epoch_seconds,
        "median_full_dev_validation_seconds_recent": validation_review_seconds,
        "forecast_full_review_epoch_seconds": training_epoch_seconds + validation_review_seconds,
        "forecast_remaining_train_seconds": train_forecast,
        "forecast_remaining_validation_seconds": validation_forecast,
        "forecast_remaining_seconds_including_validation": train_forecast + validation_forecast,
        "observed_case_visits_per_epoch": int(last["case_visits"]),
        "observed_optimizer_updates_per_epoch": int(last["optimizer_updates"]),
        "observed_case_visits_total": sum(int(row["case_visits"]) for row in rows),
        "observed_optimizer_updates_total": sum(int(row["optimizer_updates"]) for row in rows),
        "observed_work_counts_last_epoch": last.get("work_counts", {}),
        "observed_train_objective_last_epoch": last.get("train_losses", {}),
        "observed_validation_field_score_last_review": (
            None if not last.get("validation") else last["validation"].get("field_score")),
        "current_best_field_selected_age": None if best_row is None else int(best_row["epoch"]),
        "primary_and_auxiliary_query_budget": query_budget,
        "forecast_is_extrapolation_from_measured_epochs": True,
        "optimizer_updates_in_this_command": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    streams = subparsers.add_parser("streams", help="validate matched epoch-1 observed streams")
    streams.add_argument("--comparison-manifest", required=True)
    forecast = subparsers.add_parser("forecast", help="forecast from complete measured TRAIN/DEV epochs")
    forecast.add_argument("--output-dir", required=True)
    forecast.add_argument("--horizon", type=int, default=2500)
    args = parser.parse_args(argv)
    if args.command == "streams":
        result = validate_fresh_matched_streams(args.comparison_manifest)
    else:
        result = forecast_fresh_matched_run(args.output_dir, horizon=args.horizon)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
