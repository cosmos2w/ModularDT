"""Evidence-bound conditional full1000 launch gate for the source-read study."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .source_read_protocol import SOURCE_READ_FULL, validate_source_read_recipe

_THERMAL_PRIMARY = {"temperature_fluid", "v", "p"}
_THERMAL_OTHER = {"u", "omega", "temperature_surface", "temperature_material", "q_proxy", "module_peak_endpoint"}
_THERMAL_TAILS = _THERMAL_PRIMARY | {"temperature_material", "module_peak_endpoint"}
_WIND_PRIMARY = {f"{role}/{component}" for role in ("near_turbine", "downstream_envelope")
                 for component in ("Ux", "Uy", "Uz", "vector")}
_WIND_OTHER = {f"{role}/{component}" for role in ("volume", "hub_slab", "background")
               for component in ("Ux", "Uy", "Uz", "vector")}


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid measured qualification metric {name}.")
    return float(value)


def _ratios(value: Any, name: str, maximum: float, names: set[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != names:
        raise ValueError(f"Missing native qualification ratios {name}.")
    if any(_number(x, name) > maximum for x in value.values()):
        raise ValueError(f"Source-read full1000 failed {name} guardrail.")


def validate_source_read_qualification(
    path: str | Path, recipe: Mapping[str, Any], *, output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read numerical evidence before model construction or run allocation.

    A single qualification is bound to one fresh output directory. The engine
    enforces exclusive writing and same-run exact latest-state continuation.
    The resource supervisor separately checks the live remaining allowance.
    """
    recipe = validate_source_read_recipe(recipe)
    if recipe["protocol"] != SOURCE_READ_FULL:
        raise ValueError("Source-read qualification is only for its own full1000 protocol.")
    q = json.loads(Path(path).read_text())
    if (q.get("schema") != "source_read_full1000_qualification_v1"
            or q.get("dataset") != recipe["dataset"]
            or q.get("source_reader_variant") != recipe["source_reader_variant"]
            or q.get("development_epoch") != 2500
            or q.get("full_recipe_sha256") != _canonical(recipe)):
        raise ValueError("Full source-read qualification has the wrong recipe, candidate or saved age.")
    target = q.get("authorized_run_directory")
    if not isinstance(target, str) or output_dir is None or Path(target).resolve() != Path(output_dir).resolve():
        raise ValueError("Qualification authorizes exactly one named fresh full run directory.")
    bindings = q.get("full_loaded_source", {})
    files = bindings.get("files")
    if not isinstance(files, dict) or not files or bindings.get("sha256") != _canonical(files):
        raise ValueError("Qualification lacks a valid full-runtime source binding.")
    project = Path(__file__).resolve().parents[2]
    for name, expected in files.items():
        source = (project / name).resolve()
        if not source.is_relative_to(project) or not source.is_file():
            raise ValueError("Qualification source is missing or outside the project.")
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Qualified source changed: {name}.")
    artifacts = q.get("evidence_files", {})
    required = {"native_accuracy", "cost", "correctness", "train_calibration", "budget_forecast",
                "candidate_checkpoint", "dense_host_checkpoint"}
    if recipe["dataset"] == "thermal":
        required.add("response_guard")
    if not required <= set(artifacts):
        raise ValueError("Missing measured source-read qualification artifacts.")
    evidence = {}
    for name in required:
        item = artifacts[name]
        source = Path(item["path"])
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != item.get("sha256"):
            raise ValueError(f"Qualification artifact changed: {name}.")
        if name not in ("candidate_checkpoint", "dense_host_checkpoint"):
            evidence[name] = json.loads(source.read_text())
    native = evidence["native_accuracy"]
    if (native.get("candidate_checkpoint_sha256") != artifacts["candidate_checkpoint"]["sha256"]
            or native.get("dense_host_checkpoint_sha256") != artifacts["dense_host_checkpoint"]["sha256"]
            or native.get("candidate_epoch") != 2500 or native.get("dense_host_epoch") != 2500):
        raise ValueError("Native board does not bind both literal2500 model states.")
    metrics = native["qualification_metrics"]
    thermal = recipe["dataset"] == "thermal"
    primary = _THERMAL_PRIMARY if thermal else _WIND_PRIMARY
    other = _THERMAL_OTHER if thermal else _WIND_OTHER
    tails = _THERMAL_TAILS if thermal else _WIND_PRIMARY
    _ratios(metrics.get("other_means"), "other_means", 1.10, other)
    _ratios(metrics.get("primary_p90"), "primary_p90", 1.10, tails)
    cost = evidence["cost"]["qualification_metrics"]
    latency = _number(cost.get("complete_high_m_time_ratio"), "complete_high_m_time_ratio")
    variant = recipe["source_reader_variant"]
    if variant == "D-scale":
        _ratios(metrics.get("other_primary_means"), "other_primary_means", 1.05, primary)
        ratios = metrics["other_primary_means"]
        if thermal:
            improvement = min(ratios["temperature_fluid"], math.sqrt(ratios["v"] * ratios["p"]))
        else:
            six = [ratios[f"{role}/{component}"] for role in ("near_turbine", "downstream_envelope")
                   for component in ("Ux", "Uy", "Uz")]
            improvement = math.prod(six) ** (1.0 / 6.0)
        if improvement > .95:
            raise ValueError("D-scale did not improve a declared primary target by5percent.")
        if latency > 1.15 or _number(cost.get("training_epoch_ratio"), "training_epoch_ratio") > 1.15:
            raise ValueError("D-scale exceeds its15percent cost allowance.")
        if metrics.get("native_tail_review_consistent") is not True:
            raise ValueError("D-scale native tails do not support automatic selection.")
    else:
        _ratios(metrics.get("primary_means"), "primary_means", 1.05, primary)
        if _number(cost.get("actual_fine_retained_fraction"), "actual_fine_retained_fraction") > .70:
            raise ValueError("Sparse candidate saves fewer than30percent actual fine rows.")
        memory = _number(cost.get("b8_q8192_memory_ratio"), "b8_q8192_memory_ratio")
        if not (latency <= .90 or (memory <= .80 and latency <= 1.15)):
            raise ValueError("Sparse candidate has no qualified complete-time or memory operating point.")
        if metrics.get("not_materially_worse_than_equal_work_geometry") is not True:
            raise ValueError("Sparse candidate fails the equal-work geometry control.")
    checks = evidence["correctness"].get("checks", {})
    needed = {"native_laws", "same_function_executor", "plan_ownership", "query_consistency",
              "gradients", "real_optimizer", "checkpoint_resume", "full_stop"}
    if any(checks.get(name) is not True for name in needed):
        raise ValueError("Missing strict source-read correctness evidence.")
    if evidence["train_calibration"].get("TRAIN_only") is not True:
        raise ValueError("Calibration is not bound to TRAIN only.")
    if evidence["budget_forecast"].get("fits_remaining_phase_budget") is not True:
        raise ValueError("No remaining-budget forecast for this full run.")
    if (recipe["dataset"] == "thermal"
            and evidence["response_guard"].get("no_new_gross_masking_failure") is not True):
        raise ValueError("Thermal saved response guard does not qualify automatic follow-up.")
    return q


__all__ = ["validate_source_read_qualification"]
