"""CPU contracts for the read-only Run-1407 four-case anchor evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools" / "diagnostics"))

from evaluate_run1407_epoch50_anchors import (
    COMPARISON_LABELS,
    DEFAULT_CASE_IDS,
    _aggregate_metric_rows,
    build_parser,
    build_plan,
)


def _args(tmp_path: Path, *extra: str):
    paths = []
    for label in COMPARISON_LABELS:
        path = tmp_path / f"{label}.pt"
        path.write_bytes(b"placeholder")
        paths.append(path)
    return build_parser().parse_args(
        [
            "--checkpoint-1407",
            str(paths[0]),
            "--checkpoint-1406",
            str(paths[1]),
            "--checkpoint-1804",
            str(paths[2]),
            "--output",
            str(tmp_path / "anchors.json"),
            *extra,
        ]
    )


def test_plan_requires_explicit_checkpoints_and_fixed_anchor_protocol(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))

    assert plan["status"] == "plan_only"
    assert list(plan["checkpoint_policy"]) == list(COMPARISON_LABELS)
    assert plan["protocol"]["case_ids"] == list(DEFAULT_CASE_IDS)
    assert plan["protocol"]["target_epoch"] == 50
    assert plan["protocol"]["local_port_condition_mode"] == "predicted"
    assert plan["protocol"]["training"] is False
    assert plan["protocol"]["checkpoint_writes"] is False
    assert plan["protocol"]["artifact_hashes"] is False
    assert "global_field_all_norm_l2" in plan["metric_policy"]["metrics"]
    assert "interface_q_normal_physical_relative_l2" in plan["metric_policy"]["metrics"]


def test_plan_rejects_subset_or_reordered_anchor_cases(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly cases"):
        build_plan(_args(tmp_path, "--case-id", "0273"))

    with pytest.raises(ValueError, match="exactly cases"):
        build_plan(
            _args(
                tmp_path,
                "--case-id",
                "0653",
                "--case-id",
                "0273",
                "--case-id",
                "0680",
                "--case-id",
                "0298",
            )
        )


def test_aggregate_uses_pooled_sse_and_keeps_equal_case_summary() -> None:
    rows = []
    for label, scale in (("1407", 1.0), ("1406", 2.0), ("1804", 3.0)):
        for case_id, factor in (("0273", 1.0), ("0653", 2.0), ("0680", 1.0), ("0298", 2.0)):
            row = {"label": label, "case_id": case_id}
            for name, base, suffix in (
                ("global_field_all_norm_l2", "global_field_all_norm", "norm_l2"),
                ("internal_temperature_physical_relative_l2", "internal_temperature_physical", "relative_l2"),
            ):
                value = scale * factor
                row[f"{base}_{suffix}"] = value
                row[f"{base}_sse"] = value * value
                row[f"{base}_target_sse"] = 1.0
                row[f"{base}_num_values"] = 2
            rows.append(row)

    aggregate = _aggregate_metric_rows(rows)
    run1407 = aggregate[0]["metrics"]
    # sqrt((1+4+1+4)/4) = sqrt(2.5), rather than the equal-case mean 1.5.
    assert run1407["global_field_all_norm_l2"]["pooled_relative_l2"] == pytest.approx(2.5**0.5)
    assert run1407["global_field_all_norm_l2"]["equal_case_mean"] == pytest.approx(1.5)
    assert run1407["global_field_all_norm_l2"]["num_values"] == 8
    assert run1407["internal_temperature_physical_relative_l2"]["pooled_relative_l2"] == pytest.approx(2.5**0.5)
