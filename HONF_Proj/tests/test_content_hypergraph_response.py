"""Focused contracts for counted Thermal content-response replay."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from content_hypergraph_response import (
    _precision_comparison,
    _typed_native_input,
    evaluate_checkpoint,
    module_peak_changes,
)


def test_module_peak_change_is_difference_of_endpoint_maxima():
    # The first receiver warms by seven, but the module maximum rises by one.
    # max(delta_T) would report the wrong physical endpoint functional.
    baseline = np.asarray([[1.0], [10.0], [3.0], [8.0]])
    endpoint = np.asarray([[8.0], [11.0], [4.0], [6.0]])
    valid = np.asarray([[True], [True], [True], [True]])
    receiver_ids = np.asarray(["module-A", "module-A", "module-B", "module-B"])

    measured = module_peak_changes(
        baseline,
        endpoint,
        valid,
        receiver_ids,
        ("module-A", "module-B"),
    )

    assert measured["module-A"]["baseline_peak"] == 10.0
    assert measured["module-A"]["endpoint_peak"] == 11.0
    assert measured["module-A"]["predicted_peak_change"] == 1.0
    assert measured["module-B"]["predicted_peak_change"] == -2.0


def test_module_peak_change_uses_common_valid_support_and_typed_ids():
    baseline = np.asarray([[4.0], [100.0], [7.0]])
    endpoint = np.asarray([[6.0], [-100.0], [9.0]])
    valid = np.asarray([[True], [False], [True]])
    receiver_ids = ("source:0", "source:0", "source:1")

    measured = module_peak_changes(baseline, endpoint, valid, receiver_ids, ("source:0", "source:1"))

    assert measured["source:0"]["baseline_peak"] == 4.0
    assert measured["source:0"]["endpoint_peak"] == 6.0
    assert measured["source:0"]["predicted_peak_change"] == 2.0
    assert measured["source:1"]["predicted_peak_change"] == 2.0


@pytest.mark.parametrize("existing_kind", ("json", "npz", "json_temp", "npz_temp"))
def test_existing_response_output_is_never_overwritten(tmp_path, existing_kind):
    output = tmp_path / "response.json"
    existing_path = {
        "json": output,
        "npz": output.with_suffix(".npz"),
        "json_temp": output.with_suffix(".json.tmp"),
        "npz_temp": output.with_suffix(".npz").with_suffix(".npz.tmp"),
    }[existing_kind]
    existing_path.write_text("previous artifact\n")

    with pytest.raises(FileExistsError, match="immutable"):
        evaluate_checkpoint(
            checkpoint_path=tmp_path / "missing-checkpoint.pt",
            request_path=tmp_path / "missing-request.json",
            records_dir=tmp_path / "missing-records",
            output_path=output,
        )

    assert existing_path.read_text() == "previous artifact\n"


def test_precision_comparison_retains_elementwise_failure_counts():
    expected = np.asarray([0.0, 1.0, -2.0])
    actual = np.asarray([0.5e-7, 1.0 + 2.0e-7, -2.0])

    result = _precision_comparison(actual, expected, rtol=1.0e-7, atol=1.0e-9)

    assert result["passed"] is False
    assert result["failed_elements"] == 2
    assert result["compared_elements"] == 3


def test_nonzero_saved_temperature_boundary_is_rejected_before_scene_build():
    module = SimpleNamespace(module_id="case:module:0", position_xy=(1.0, 2.0))
    record = SimpleNamespace(
        design=SimpleNamespace(modules=(module,), active_module_ids=(module.module_id,)),
        context=SimpleNamespace(values={"t_in": 0.0, "t_wall": 0.1}),
    )

    with pytest.raises(ValueError, match="inlet and wall temperatures to be zero"):
        _typed_native_input(record, device=torch.device("cpu"))
