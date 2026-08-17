"""User-facing sampling, frozen verification, correction, and evaluation CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from honf_inverse_core.config import validate_config_keys
from honf_inverse_core.normalization import ScalarStats, VectorStats
from honf_inverse_core.sampling.serialization import write_json_atomic
from honf_inverse_core.training.checkpointing import load_inverse_checkpoint
from honf_runtime.run_store import RunStore

from channelthermal.inverse.context import load_context
from channelthermal.inverse.diagnostics import artifact_sha256
from channelthermal.inverse.evaluation.artifacts import write_evaluation_artifacts
from channelthermal.inverse.evaluation.candidate_evaluator import (
    EvaluationNormalizers,
    ThermalChannelCandidateEvaluator,
)
from channelthermal.inverse.request import make_request_codec
from channelthermal.inverse.verifier import FrozenThermalChannelVerifier, file_sha256
from channelthermal.inverse.vocabulary import REQUEST_TYPES


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def normalizers_from_checkpoint(checkpoint: Mapping[str, Any]) -> EvaluationNormalizers:
    stats = checkpoint["provenance"]["normalization_stats"]
    functional = {
        name: ScalarStats(
            mean=float(stats["functional_mean"][index]),
            std=float(stats["functional_std"][index]),
            count=int(stats["functional_count"][index]),
        )
        for index, name in enumerate(REQUEST_TYPES)
    }
    context = VectorStats(
        tuple(str(value) for value in stats["context_feature_names"]),
        stats["context_mean"],
        stats["context_std"],
        int(max(stats["functional_count"])),
    )
    active_heat = ScalarStats(float(stats["active_heat_power_mean"]), float(stats["active_heat_power_std"]), 1)
    total_heat = ScalarStats(float(stats["total_heat_mean"]), float(stats["total_heat_std"]), 1)
    return EvaluationNormalizers(functional, context, active_heat, total_heat)


def run_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    config = validate_config_keys(
        config,
        allowed={
            "inverse_checkpoint", "forward_checkpoint", "forward_dataset", "request_json",
            "context_json", "device", "seed", "num_plans", "layouts_per_plan",
            "correct_once", "top_k", "output_dir", "timestamp_output",
        },
        required={
            "inverse_checkpoint", "forward_checkpoint", "request_json", "context_json",
            "device", "seed", "num_plans", "layouts_per_plan", "correct_once", "top_k", "output_dir",
        },
        label="inverse evaluation",
    )
    inverse_path = Path(str(config["inverse_checkpoint"])).expanduser().resolve()
    forward_path = Path(str(config["forward_checkpoint"])).expanduser().resolve()
    checkpoint = load_inverse_checkpoint(inverse_path)
    expected_forward_sha = checkpoint["provenance"].get("forward_checkpoint_sha256")
    actual_forward_sha = file_sha256(forward_path)
    if expected_forward_sha and actual_forward_sha != expected_forward_sha:
        raise ValueError("Configured frozen forward checkpoint SHA does not match inverse checkpoint provenance.")
    normalizers = normalizers_from_checkpoint(checkpoint)
    codec = make_request_codec(normalizers.functional)
    request = codec.load(Path(str(config["request_json"])).expanduser().resolve())
    context = load_context(Path(str(config["context_json"])).expanduser().resolve())
    device = str(config.get("device", "cuda:0"))
    frozen = FrozenThermalChannelVerifier(forward_path, device=device, dataset_path=config.get("forward_dataset"))
    designer = HierarchicalInverseDesigner.load(inverse_path, device=device)
    evaluator = ThermalChannelCandidateEvaluator(designer, frozen, normalizers)
    designer.attach_verifier(evaluator)
    result = designer.sample_candidates(
        request=request,
        context=context,
        num_plans=int(config.get("num_plans", 8)),
        layouts_per_plan=int(config.get("layouts_per_plan", 4)),
        correct_once=bool(config.get("correct_once", True)),
        top_k=int(config.get("top_k", 8)),
        seed=int(config.get("seed", 0)),
    )
    output = Path(str(config["output_dir"])).expanduser().resolve()
    if bool(config.get("timestamp_output", True)):
        output = output / f"InverseEval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary = write_evaluation_artifacts(result, output)
    write_json_atomic(output / "config_resolved.json", dict(config))
    write_json_atomic(output / "request_input.json", request.to_dict())
    write_json_atomic(output / "context_input.json", context.to_dict())
    inventory = {
        str(path.relative_to(output)): {
            "size": int(path.stat().st_size),
            "sha256": artifact_sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "evaluation_manifest.json"
    }
    manifest = {
        "schema_name": "honf_hierarchical_inverse_evaluation_manifest",
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inverse_checkpoint": str(inverse_path),
        "inverse_checkpoint_sha256": artifact_sha256(inverse_path),
        "forward_checkpoint": str(forward_path),
        "forward_checkpoint_sha256": actual_forward_sha,
        "inverse_dataset_hash": checkpoint["provenance"]["inverse_dataset_hash"],
        "request_schema_version": checkpoint["provenance"]["request_schema_version"],
        "compact_plan_schema_version": checkpoint["provenance"]["compact_plan_schema_version"],
        "forward_call_count": int(result.metadata["forward_call_count"]),
        "candidate_counts": {
            "raw_unguided": len(result.raw_unguided),
            "correction_proposals": len(result.corrected),
            "accepted_one_pass": len(result.accepted_one_pass),
            "final_ranked": len(result.final_ranked),
        },
        "artifacts": inventory,
    }
    write_json_atomic(output / "evaluation_manifest.json", manifest)
    RunStore.record_evaluation(inverse_path.parent, output)
    return {"output_dir": str(output), "evaluation_manifest": manifest, **summary}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", required=True)
    value.add_argument("--inverse-checkpoint")
    value.add_argument("--forward-checkpoint")
    value.add_argument("--request-json")
    value.add_argument("--context-json")
    value.add_argument("--device")
    value.add_argument("--seed", type=int)
    value.add_argument("--num-plans", type=int)
    value.add_argument("--layouts-per-plan", type=int)
    value.add_argument("--top-k", type=int)
    value.add_argument("--no-correction", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = _json(Path(args.config).expanduser().resolve())
    for name in (
        "inverse_checkpoint", "forward_checkpoint", "request_json", "context_json",
        "device", "seed", "num_plans", "layouts_per_plan", "top_k",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.no_correction:
        config["correct_once"] = False
    print(json.dumps(run_evaluation(config), indent=2, sort_keys=True))
    return 0


__all__ = ["main", "normalizers_from_checkpoint", "run_evaluation"]
