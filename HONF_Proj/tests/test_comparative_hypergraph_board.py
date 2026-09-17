"""CPU tests for the paired sparse-routing hypergraph board."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "tools" / "diagnostics"), str(_ROOT / "src")]

from render_comparative_hypergraph_board import render_board


def _write_fixture_run(root: Path, *, label: str, strategy: str, epoch: int = 2500) -> Path:
    map_paths = []
    rows = []
    for case_id in ("0273", "0283"):
        map_path = root / f"{label}__{case_id}.npz"
        np.savez_compressed(
            map_path,
            module_centers=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            module_present=np.asarray([1.0, 1.0], dtype=np.float32),
            env_coords=np.asarray([[0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32),
            env_weights=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            p2_field__hub_coords=np.asarray([[0.2, 0.2], [0.8, 0.2]], dtype=np.float32),
            p2_field__hub_valid=np.asarray([True, True]),
            p2_field__module_source_A=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            p2_field__environment_source_A=np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
            ),
            selected_far_query_xy=np.asarray([0.5, 0.5], dtype=np.float32),
            p2_far_module_pair_ids=np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.int64),
            p2_far_module_pair_Pi=np.asarray([0.5, 0.5], dtype=np.float64),
            p2_far_environment_pair_ids=np.asarray(
                [[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=np.int64
            ),
            p2_far_environment_pair_Pi=np.asarray([0.3, 0.3, 0.4], dtype=np.float64),
            routing_strategy=np.asarray(strategy),
        )
        map_paths.append(str(map_path))
        rows.append(
            {
                "case_id": case_id,
                "query_count": 1,
                "M_active": 2,
                "environment_source_count": 3,
                "p2_field_module_raw_path_count": 2,
                "p2_field_module_unique_pair_count": 2,
                "p2_field_environment_raw_path_count": 4,
                "p2_field_environment_unique_pair_count": 3,
                "p2_field_environment_R_completeQE": 1.0,
                "phase_metrics": {
                    "p2_field_environment": {
                        "fine_pair_count": {"count": 1, "min": 3.0, "max": 3.0}
                    }
                },
                "routing_maps": str(map_paths[-1]),
            }
        )
    ledger = root / f"{label}.json"
    ledger.write_text(
        json.dumps(
            {
                "task": "dynamic_sparse_routing_ledger",
                "checkpoint": {"epoch": epoch, "label": label},
                "routing_strategy": strategy,
                "rows": rows,
                "routing_maps": map_paths,
            }
        ),
        encoding="utf-8",
    )
    return ledger


def test_board_requires_exact_epoch_and_renders_provenance(tmp_path: Path) -> None:
    left = _write_fixture_run(tmp_path, label="run2001", strategy="module_hubs")
    right = _write_fixture_run(tmp_path, label="run2101", strategy="mean_shift")

    manifest = render_board(left, right, tmp_path / "board")

    assert manifest["status"] == "ok"
    assert manifest["runs"][0]["strategy_label"] == "Module-hub"
    assert manifest["runs"][1]["strategy_label"] == "Mean-shift"
    assert manifest["metrics"]["0273"]["Run 2001"]["R_module"] == 1.0
    assert manifest["metrics"]["0273"]["Run 2101"]["R_env"] == 1.0
    for name in ("comparative_hypergraph_board.png", "comparative_hypergraph_board.pdf", "comparative_hypergraph_board.json"):
        assert (tmp_path / "board" / name).is_file()


def test_board_rejects_non_exact_checkpoint(tmp_path: Path) -> None:
    left = _write_fixture_run(tmp_path, label="run2001", strategy="module_hubs", epoch=2425)
    right = _write_fixture_run(tmp_path, label="run2101", strategy="mean_shift")

    with pytest.raises(ValueError, match="does not match required epoch"):
        render_board(left, right, tmp_path / "board")
