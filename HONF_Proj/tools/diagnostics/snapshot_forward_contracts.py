#!/usr/bin/env python3
"""Snapshot public forward schemas used by cleanup compatibility gates."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))

from channelthermal.model import PreparedChannelThermalCase  # noqa: E402
from honf_forward_core.config import BatchData, UnifiedForwardConfig  # noqa: E402
from honf_forward_core.evaluation.hypergraph_plan import SCHEMA_VERSION as PLAN_SCHEMA_VERSION  # noqa: E402
from honf_forward_core.evaluation.topology_signature import SCHEMA_VERSION as TOPOLOGY_SCHEMA_VERSION  # noqa: E402
from honf_runtime.artifact_layout import EVALUATION_LAYOUT_VERSION  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "docs" / "experiments" / "stage1_7_freeze_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "fixtures" / "forward_cleanup" / "public_schemas.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def json_value(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
    return value


def dataclass_schema(cls: type[Any]) -> list[dict[str, Any]]:
    records = []
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            default = json_value(field.default)
        elif field.default_factory is not dataclasses.MISSING:
            default = json_value(field.default_factory())
        else:
            default = "<required>"
        records.append({"name": field.name, "type": str(field.type), "default": default})
    return records


def checkpoint_contract(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "top_level_keys": sorted(checkpoint),
        "checkpoint_schema_version": checkpoint.get("checkpoint_schema_version"),
        "channel_order": checkpoint.get("channel_order"),
        "feature_schemas": checkpoint.get("feature_schemas"),
        "dataset_schema": checkpoint.get("dataset_schema"),
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    references = manifest["golden_references"]
    payload = {
        "schema_version": 1,
        "dataclasses": {
            "UnifiedForwardConfig": dataclass_schema(UnifiedForwardConfig),
            "BatchData": dataclass_schema(BatchData),
            "PreparedChannelThermalCase": dataclass_schema(PreparedChannelThermalCase),
        },
        "artifact_schemas": {
            "hypergraph_plan": PLAN_SCHEMA_VERSION,
            "topology_signature": TOPOLOGY_SCHEMA_VERSION,
            "run_manifest": 1,
            "evaluation_layout": EVALUATION_LAYOUT_VERSION,
        },
        "checkpoints": {
            label: checkpoint_contract(PROJECT_ROOT / reference["checkpoint"])
            for label, reference in references.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
