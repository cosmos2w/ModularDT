from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_fixture(name: str) -> dict:
    path = PROJECT_ROOT / "tests" / "fixtures" / "forward_cleanup" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_accepted_golden_references_are_frozen_with_exact_state_contracts() -> None:
    fixture = read_fixture("golden_replay.json")
    references = fixture["references"]
    assert set(references) == {
        "run1000_best_field_e9655",
        "run1401_best_field_e4585",
    }
    for reference in references.values():
        assert reference["case_id"] == "0653"
        assert reference["state_dict"]["key_count"] == 237
        assert reference["optimizer_resume"]["group_count"] == 1
        assert reference["optimizer_resume"]["state_entry_count"] > 0
        assert reference["optimizer_resume"]["model_parameter_order_sha256"] == (
            "e0464cf77c92bc383c52157b6227f114f08e4cb8234435d49adb1f0766549261"
        )
        assert set(reference["outputs"]) == {
            "pred_field_grid",
            "pred_internal_temperature",
            "pred_interface",
            "pred_port_condition",
        }


def test_public_schema_snapshot_covers_forward_and_inverse_facing_contracts() -> None:
    fixture = read_fixture("public_schemas.json")
    assert set(fixture["dataclasses"]) == {
        "UnifiedForwardConfig",
        "BatchData",
        "PreparedChannelThermalCase",
    }
    assert fixture["artifact_schemas"] == {
        "evaluation_layout": 2,
        "hypergraph_plan": 2,
        "run_manifest": 1,
        "topology_signature": 3,
    }
    for checkpoint in fixture["checkpoints"].values():
        assert checkpoint["checkpoint_schema_version"] == 1
        assert checkpoint["channel_order"] == ["u", "v", "p", "omega", "temperature"]
        assert "model_state_dict" in checkpoint["top_level_keys"]
        assert "optimizer_state_dict" in checkpoint["top_level_keys"]
