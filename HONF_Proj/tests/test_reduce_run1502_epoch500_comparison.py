from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = PROJECT / "tools" / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

from reduce_run1502_epoch500_comparison import (
    CASE_METRICS,
    COST_FIELDS,
    LABELS,
    POOLED_BASES,
    RUN_PREFIXES,
    reduce,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint(tmp_path: Path, label: str) -> Path:
    checkpoint = tmp_path / f"{RUN_PREFIXES[label]}fixture" / "epoch_0500_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    return checkpoint


def _population_evidence(checkpoint: Path, case_ids: list[str], value: float) -> dict[str, object]:
    return {
        "status": "complete",
        "candidate": {
            "architecture": "sparse_incidence_group_control_honf",
            "checkpoint": str(checkpoint),
        },
        "protocol": {
            "split": "test",
            "local_port_condition_mode": "predicted",
            "query_count_requested": 1024,
            "query_batch_size": 1024,
            "case_ids": case_ids,
        },
        "population": {
            "case_count": 90,
            "query_count_per_case": [1024],
            "query_degree_mean": {"mean": 2.5},
            "environment_RE_support": {"mean": value},
            "p2_module_geometry_rows": {"mean": 123.0},
        },
        "cases": [{"case_id": case_id} for case_id in case_ids],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    comparison = tmp_path / "comparison"
    checkpoints = {label: _checkpoint(tmp_path, label) for label in LABELS}
    case_ids = [f"{case_index:04d}" for case_index in range(90)]
    rows: list[dict[str, object]] = []
    for model_index, label in enumerate(LABELS, start=1):
        for case_index in range(90):
            row: dict[str, object] = {
                "model_label": label,
                "case_id": f"{case_index:04d}",
                "checkpoint": str(checkpoints[label]),
            }
            for base in POOLED_BASES:
                row[f"{base}_sse"] = float(model_index)
                row[f"{base}_target_sse"] = 10.0
                row[f"{base}_num_values"] = 5
            for metric in CASE_METRICS:
                row[metric] = float(model_index + case_index / 1000.0)
            rows.append(row)
    _write_csv(comparison / "tables" / "per_case_metrics.csv", rows)

    cost_rows = []
    for model_index, label in enumerate(LABELS, start=1):
        cost_rows.append(
            {
                "model_label": label,
                "checkpoint": str(checkpoints[label]),
                "num_cases": 90,
                **{field: float(model_index) for field in COST_FIELDS},
            }
        )
    _write_csv(comparison / "tables" / "evaluation_cost_summary_metrics.csv", cost_rows)
    cost_case_rows = [
        {
            "model_label": label,
            "case_id": case_id,
            "checkpoint": str(checkpoints[label]),
            "split": "test",
            "evaluation_grid_query_count": 8192,
            "query_batch_size": 32768,
        }
        for label in LABELS
        for case_id in case_ids
    ]
    _write_csv(comparison / "tables" / "evaluation_cost_case_metrics.csv", cost_case_rows)
    (comparison / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "kind": "compare",
                "arguments": [
                    item
                    for label in LABELS
                    for item in ("--checkpoint-path", str(checkpoints[label]))
                ]
                + [item for label in LABELS for item in ("--label", label)]
                + [
                    "--split",
                    "test",
                    "--case-ratio",
                    "1.0",
                    "--query-batch-size",
                    "32768",
                    "--local-port-condition-mode",
                    "predicted",
                ],
            }
        ),
        encoding="utf-8",
    )

    population_1502 = tmp_path / "population1502.json"
    population_1501 = tmp_path / "population1501.json"
    population_1502.write_text(
        json.dumps(_population_evidence(checkpoints[LABELS[0]], case_ids, 0.4)),
        encoding="utf-8",
    )
    population_1501.write_text(
        json.dumps(
            {
                **_population_evidence(checkpoints[LABELS[1]], case_ids, 0.7),
                "population": {
                    **_population_evidence(checkpoints[LABELS[1]], case_ids, 0.7)["population"],
                    "p2_module_geometry_rows": None,
                    "p2_module_padded_rows": {"mean": 5.0},
                },
            }
        ),
        encoding="utf-8",
    )
    return comparison, population_1502, population_1501


def test_reduce_validates_and_compares_matched_epoch500_rows(tmp_path: Path) -> None:
    comparison, population_1502, population_1501 = _fixture(tmp_path)

    output = tmp_path / "reduction"
    payload = reduce(comparison, population_1502, population_1501, output)

    assert payload["status"] == "complete"
    assert payload["population"] == 90
    assert payload["headline"][0]["pooled_fluid_relative_l2"] < payload["headline"][1][
        "pooled_fluid_relative_l2"
    ]
    paired = list(csv.DictReader((output / "paired_summary.csv").open(encoding="utf-8")))
    first = next(
        row
        for row in paired
        if row["left"] == LABELS[0]
        and row["right"] == LABELS[1]
        and row["metric"] == "global_field_fluid_norm_l2"
    )
    assert first["left_wins"] == "90"
    structure = list(
        csv.DictReader((output / "sparse_structure_summary.csv").open(encoding="utf-8"))
    )
    assert structure[0]["environment_RE_support"] == "0.4"
    assert structure[0]["p2_module_geometry_rows"] == "123.0"
    assert structure[1]["p2_module_padded_rows"] == "unavailable_legacy_accounting"
    assert structure[-1]["environment_RE_support"] == "unavailable_not_zero"


def test_reduce_rejects_wrong_checkpoint_identity(tmp_path: Path) -> None:
    comparison, population_1502, population_1501 = _fixture(tmp_path)
    rows = list(csv.DictReader((comparison / "tables" / "per_case_metrics.csv").open()))
    rows[0]["checkpoint"] = str(tmp_path / "Run_1501_wrong" / "epoch_0500_model.pt")
    _write_csv(comparison / "tables" / "per_case_metrics.csv", rows)
    with pytest.raises(ValueError, match="not one explicit checkpoint"):
        reduce(comparison, population_1502, population_1501, tmp_path / "reduction")


def test_reduce_rejects_bad_population_protocol(tmp_path: Path) -> None:
    comparison, population_1502, population_1501 = _fixture(tmp_path)
    payload = json.loads(population_1502.read_text(encoding="utf-8"))
    payload["protocol"]["query_count_requested"] = 8192
    population_1502.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol mismatch"):
        reduce(comparison, population_1502, population_1501, tmp_path / "reduction")


def test_reduce_rejects_nonfinite_cost(tmp_path: Path) -> None:
    comparison, population_1502, population_1501 = _fixture(tmp_path)
    rows = list(csv.DictReader((comparison / "tables" / "evaluation_cost_summary_metrics.csv").open()))
    rows[0][COST_FIELDS[0]] = "nan"
    _write_csv(comparison / "tables" / "evaluation_cost_summary_metrics.csv", rows)
    with pytest.raises(ValueError, match="missing or nonfinite"):
        reduce(comparison, population_1502, population_1501, tmp_path / "reduction")
