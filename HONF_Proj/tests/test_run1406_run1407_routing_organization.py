from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools/diagnostics/diagnose_run1406_run1407_routing_organization.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("routing_organization", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_density_counts_unique_pairs_and_logical_paths() -> None:
    module = _module()
    alpha = np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    incidence = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
    )

    metrics = module._density_metrics(alpha, incidence, None, None, "source")

    assert metrics["source_unique_pair_count_mean"] == 2.0
    assert metrics["source_logical_path_count_mean"] == 2.5
    assert metrics["source_logical_to_unique_multiplicity"] == 1.25


def test_ledger_ratios_use_q_wide_denominators_not_chunk_counters() -> None:
    module = _module()
    raw = {
        "group_control_module_fine_rows": 10.0,
        "group_control_module_fine_rows_forward": 2.0,
        "group_control_module_fine_rows_padded": 1.0,
        "group_control_module_valid_pair_denominator": 6.0,
        "group_control_module_padded_pair_denominator": 4.0,
    }

    metrics = module._ledger_values(raw, "module", "P2", 8, 4, 3)

    assert metrics["module_P2_fine_rows"] == 10.0
    assert metrics["module_P2_fine_rows_forward_reported"] == 2.0
    assert metrics["module_P2_fine_rows_padded_reported"] == 1.0
    assert metrics["module_P2_executed_padded_rows"] == 4.0
    assert metrics["module_P2_executed_valid_fraction"] == 0.6
    assert metrics["module_P2_executed_padded_fraction"] == 0.4
    assert metrics["module_P2_valid_fraction_of_pair_denominator"] == 0.6
    assert metrics["module_P2_padded_fraction_of_pair_denominator"] == 0.4
    assert metrics["module_P2_executed_rows_over_pair_denominator"] == 1.0


def test_query_matrix_flattens_phase_receiver_axes() -> None:
    module = _module()
    aux = {"group_control_query_routing": np.ones((1, 12, 64, 6))}

    matrix = module._query_matrix(aux)

    assert matrix is not None
    assert matrix.shape == (768, 6)
