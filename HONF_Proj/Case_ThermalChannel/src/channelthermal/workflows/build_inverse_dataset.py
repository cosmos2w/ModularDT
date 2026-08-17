"""ThermalChannel CLI workflow for one-call frozen-HONF dataset creation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from honf_inverse_core.config import validate_config_keys

from channelthermal.inverse.dataset_builder import (
    CaseBuildRecord,
    assign_inverse_splits,
    build_inverse_dataset_from_records,
)
from channelthermal.inverse.diagnostics import artifact_sha256, generate_dataset_diagnostics
from channelthermal.inverse.functionals import evaluate_regional_functional
from channelthermal.inverse.verifier import FrozenThermalChannelVerifier


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Inverse data config must be a JSON object: {path}")
    return value


def _select_indices(
    verifier: FrozenThermalChannelVerifier,
    *,
    split_seed: int,
    validation_fraction: float,
    max_cases_per_split: int | None,
) -> list[tuple[int, str, str]]:
    case_ids = verifier.case_ids
    source_splits = [
        str(verifier.raw_dataset.splits[verifier.raw_dataset.indices[index]]).lower()
        for index in range(len(case_ids))
    ]
    inverse = assign_inverse_splits(
        case_ids,
        source_splits,
        split_seed=split_seed,
        validation_fraction=validation_fraction,
    )
    selected: list[tuple[int, str, str]] = []
    counts = {"train": 0, "validation": 0, "test": 0}
    for index, (source, target) in enumerate(zip(source_splits, inverse)):
        if max_cases_per_split is not None and counts[target] >= max_cases_per_split:
            continue
        selected.append((index, source, target))
        counts[target] += 1
    if not selected or counts["train"] == 0:
        raise ValueError("Selected inverse build has no training cases.")
    return selected


def _record_for_case(
    verifier: FrozenThermalChannelVerifier,
    index: int,
    source_split: str,
    inverse_split: str,
) -> CaseBuildRecord:
    verified = verifier.verify_case(case_index=index, return_outputs=("environment",))
    environment = verified.outputs["environment"]

    def regional(name: str, region: tuple[float, float, float, float]):
        return evaluate_regional_functional(
            name,
            region,
            pred_field_grid=environment["pred_field_grid"],
            x_grid=environment["x_grid"],
            y_grid=environment["y_grid"],
            channel_order=environment["channel_order"],
            design=verified.design,
            context=verified.context,
        )

    source_index = int(verifier.raw_dataset.indices[index])
    return CaseBuildRecord(
        case_id=verifier.case_ids[index],
        source_split=source_split,
        inverse_split=inverse_split,
        source_index=source_index,
        design=verified.design,
        context=verified.context,
        compact_plan=verified.compact_plan,
        full_plan=verified.full_plan,
        nonregional_functionals=verified.functionals,
        regional_evaluator=regional,
        converged=bool(verifier.raw_dataset.selected_converged_flags[index]),
        metadata={"source_case_index": source_index},
    )


def resolve_build_config(config: Mapping[str, Any], overrides: argparse.Namespace) -> dict[str, Any]:
    allowed = {
        "case", "checkpoint", "dataset", "device", "output_dir", "variants_per_case",
        "seed", "split_seed", "validation_fraction", "query_batch_size", "save_full_plan",
        "max_cases_per_split",
    }
    resolved = validate_config_keys(
        config,
        allowed=allowed,
        required={
            "case", "checkpoint", "device", "output_dir", "variants_per_case", "seed",
            "split_seed", "validation_fraction", "query_batch_size", "save_full_plan",
        },
        label="inverse data",
    )
    for name in ("checkpoint", "dataset", "device", "output_dir", "variants_per_case", "seed", "query_batch_size"):
        value = getattr(overrides, name, None)
        if value is not None:
            resolved[name] = value
    if overrides.max_cases_per_split is not None:
        resolved["max_cases_per_split"] = int(overrides.max_cases_per_split)
    required = {
        "checkpoint", "device", "output_dir", "variants_per_case", "seed",
        "split_seed", "validation_fraction", "query_batch_size", "save_full_plan",
    }
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(f"Inverse data config is missing keys: {missing}")
    return resolved


def run_build(config: Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    checkpoint = Path(str(config["checkpoint"])).expanduser().resolve()
    dataset = None if config.get("dataset") in {None, ""} else Path(str(config["dataset"])).expanduser().resolve()
    verifier = FrozenThermalChannelVerifier(
        checkpoint,
        device=str(config["device"]),
        dataset_path=dataset,
        query_batch_size=int(config["query_batch_size"]),
    )
    max_cases = config.get("max_cases_per_split")
    selected = _select_indices(
        verifier,
        split_seed=int(config["split_seed"]),
        validation_fraction=float(config["validation_fraction"]),
        max_cases_per_split=None if max_cases is None else int(max_cases),
    )
    launch = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": verifier.checkpoint_sha256,
        "dataset": str(verifier.dataset_path),
        "device": str(verifier.device),
        "selected_cases": len(selected),
        "selected_split_counts": {
            name: sum(target == name for _, _, target in selected)
            for name in ("train", "validation", "test")
        },
        "variants_per_case": int(config["variants_per_case"]),
        "partial_debug": max_cases is not None,
    }
    if dry_run:
        return {"status": "dry_run", **launch}
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _record_for_case(verifier, index, source, target)
        for index, source, target in selected
    ]
    dataset_path = output_dir / "inverse_dataset_v1.h5"
    provenance = {
        **verifier.provenance,
        "builder_resolved_config": dict(config),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    build_inverse_dataset_from_records(
        records,
        dataset_path,
        variants_per_case=int(config["variants_per_case"]),
        seed=int(config["seed"]),
        split_seed=int(config["split_seed"]),
        validation_fraction=float(config["validation_fraction"]),
        save_full_plan=bool(config["save_full_plan"]),
        provenance=provenance,
        partial_debug=max_cases is not None,
    )
    diagnostics = generate_dataset_diagnostics(dataset_path, output_dir / "diagnostics")
    manifest = {
        "status": "complete",
        **launch,
        "artifact": dataset_path.name,
        "artifact_size": dataset_path.stat().st_size,
        "artifact_sha256": artifact_sha256(dataset_path),
        "schema_versions": diagnostics["schema_versions"],
        "split_case_id_hashes": diagnostics["split_case_id_hashes"],
        "config": dict(config),
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--variants-per-case", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--max-cases-per-split", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_build_config(_load_json(Path(args.config).expanduser().resolve()), args)
    if not args.dry_run and not args.yes:
        raise SystemExit("Refusing to run frozen-HONF dataset materialization without --yes.")
    print(json.dumps(run_build(config, dry_run=args.dry_run), indent=2, sort_keys=True))
    return 0


__all__ = ["main", "resolve_build_config", "run_build"]
