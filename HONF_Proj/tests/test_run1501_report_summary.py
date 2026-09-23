from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = PROJECT / "tools" / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

from render_run1501_report_summary import _population_records


def test_report_uses_m_padded_not_registered_k_for_module_rows(tmp_path: Path) -> None:
    eval_dir = tmp_path / "evaluations"
    record_dir = eval_dir / "epoch_0050"
    record_dir.mkdir(parents=True)
    summary = {
        "query_count_per_case": [1024],
        "registered_capacity": 12,
        "M_padded": {"mean": 5.0},
        "module_dense_valid_pairs": {"mean": 4096.0},
        "environment_dense_valid_pairs": {"mean": 8192.0},
        "p2_module_actual_rows": {"mean": 5120.0},
        "p2_environment_actual_rows": {"mean": 8192.0},
        "module_unique_pairs": {"mean": 100.0},
        "environment_unique_pairs": {"mean": 200.0},
    }
    (record_dir / "population_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    records = _population_records(eval_dir)
    assert len(records) == 1
    assert records[0]["module_padded"] == 5.0
    assert records[0]["module_rectangular_fraction"] == 1.0


def test_report_leaves_module_fraction_unavailable_without_m_padded(tmp_path: Path) -> None:
    eval_dir = tmp_path / "evaluations"
    record_dir = eval_dir / "epoch_0050"
    record_dir.mkdir(parents=True)
    (record_dir / "population_summary.json").write_text(
        json.dumps(
            {
                "query_count_per_case": [1024],
                "registered_capacity": 12,
                "p2_module_actual_rows": {"mean": 5120.0},
            }
        ),
        encoding="utf-8",
    )
    records = _population_records(eval_dir)
    assert records[0]["module_rectangular_fraction"] is None
