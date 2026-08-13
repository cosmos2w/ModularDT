from __future__ import annotations

import json

import torch

from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.registry import load_case_plugin


def test_second_case_loads_and_runs_without_core_or_dispatcher_changes(tmp_path) -> None:
    plugin = load_case_plugin("tests.support_dummy_case:create_plugin")
    assert plugin.case_id == "SyntheticSecondCase"
    facts = plugin.inspect_launch(None, WorkflowRequest(workflow="forward"))
    assert facts["dataset ID"] == "synthetic_memory_v1"

    batch = plugin.adapt()
    with torch.no_grad():
        output = plugin.model(batch)["pred_field"]
    assert output.shape == batch.target_field.shape
    assert torch.isfinite(output).all()

    status = plugin.train(None, WorkflowRequest(workflow="forward"), run_dir=tmp_path)
    assert status == 0
    metric = json.loads((tmp_path / "dummy_metrics.json").read_text(encoding="utf-8"))
    assert metric["loss"] >= 0.0
    assert plugin.evaluate(None, WorkflowRequest(workflow="forward")) == 0
