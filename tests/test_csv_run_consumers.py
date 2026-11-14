import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

import honf_runtime.research_training as research
import honf_runtime.unified_training as runtime
from honf_runtime.scalar_history import VALIDATION_COLUMNS, append_scalar_rows, validation_scalar_rows
from tests.test_unified_csv_recording import CsvToyProvider, _config_for_test
from tests.test_unified_training import _ToyProvider

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relation_entrypoint_forwards_extra_validation_epochs():
    relation = load_tool("relation_anchored_train")
    recipe = {"dataset": {"name": "thermal"}, "optimizer": {},
              "training": {"seed": 1, "microbatch_cases": 2, "effective_batch_size": 4,
                           "local_horizon": 100, "monitor_every": 100, "monitor_epochs": [50]}}
    assert relation.engine_for(recipe, device="cpu").config.monitor_epochs == (50,)
    del recipe["training"]["monitor_epochs"]
    assert relation.engine_for(recipe, device="cpu").config.monitor_epochs == ()


@pytest.mark.parametrize("provider_type", [CsvToyProvider, _ToyProvider], ids=["csv", "historical-json"])
def test_relation_cli_fit_resume_status_assessment(tmp_path, monkeypatch, capsys, provider_type):
    relation = load_tool("relation_anchored_train")
    status = load_tool("unified_interaction_train")
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr(relation, "build", lambda *args, **kwargs: (nn.Linear(1, 1), provider_type()))
    monkeypatch.setattr(relation, "engine_for", lambda *args, **kwargs:
                        runtime.TrainingEngine(_config_for_test(), device="cpu"))
    monkeypatch.setattr(relation, "prepare_profile_fit", lambda *args, **kwargs: None)
    monkeypatch.setattr(relation, "bind_training_budget", lambda *args, **kwargs: None)
    recipe = {"training_profile": "native_pair", "dataset": {"name": "thermal"},
              "initialization": {"seed": 3}, "member_routing": {"arm": "adaptive_detail"}, "training": {}}
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    run = tmp_path / "run"
    args = ["fit", "--recipe", str(recipe_path), "--output", str(run)]
    assert relation.main([*args, "--stop-after", "2"]) == 0
    assert relation.main([*args, "--stop-after", "4", "--resume", "latest"]) == 0
    assert (run / ("metrics/history.csv" if provider_type is CsvToyProvider else "metrics/history.json")).is_file()
    capsys.readouterr()
    assert status.main(["status", "--run-dir", str(run)]) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["history"]["epochs"] == 4 and state["history"]["last_epoch"] == 4
    monkeypatch.setattr(research, "_load_run_manifest", lambda path: ({}, path))
    assessment = research.assess(run)
    assert assessment["completed_epochs"] == 4 and assessment["latest_epoch"] == 4
    assert len(assessment["validation_reviews"]) == 4


def test_checkpoint_validation_cache_requires_exact_csv_panel_and_values(tmp_path):
    evaluator = load_tool("unified_interaction_evaluate")
    run = tmp_path / "thermal" / "run1" / "adaptive"
    (run / "metrics").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "evaluations/validation").mkdir(parents=True)
    (run / "artifact_layout.json").write_text('{"artifact_layout_version": 1, "training_output_policy": "csv_v3"}')
    cases = [{"case_id": f"case-{i:02d}"} for i in range(22)]
    receipt = {"validation_execution_mode": "adaptive", "validation_route_phase": "hard",
               "case_count": 22, "case_ids": [row["case_id"] for row in cases], "field_score": 0.125}
    (run / "evaluations/validation/validation_epoch_0002.json").write_text(json.dumps(receipt))
    append_scalar_rows(run / "metrics/history.csv", ("epoch", "phase"), [{"epoch": 2, "phase": "joint"}])
    append_scalar_rows(run / "metrics/validation.csv", VALIDATION_COLUMNS,
                       validation_scalar_rows({"epoch": 2, "validation": receipt}))
    checkpoint = run / "checkpoints/latest_model.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    # Ensure deterministic ordering on filesystems with coarse timestamp precision.
    checkpoint_ns = max((run / "metrics/history.csv").stat().st_mtime_ns,
                        (run / "metrics/validation.csv").stat().st_mtime_ns) + 1_000_000
    os.utime(checkpoint, ns=(checkpoint_ns, checkpoint_ns))
    payload = {"epoch": 2, "arm": "adaptive", "experiment_identity": {"run_id": "run1"}}
    provider = SimpleNamespace(validation_cases=cases)
    call = lambda: evaluator._load_training_validation_cache(
        payload, task="thermal", provider=provider, checkpoint_path=checkpoint, runs_root=tmp_path)
    cached, evidence = call()
    assert cached == receipt and evidence["checkpoint_embeds_exact_validation_metrics"] is False
    receipt["field_score"] = 0.25
    (run / "evaluations/validation/validation_epoch_0002.json").write_text(json.dumps(receipt))
    assert call() is None


def test_historical_fork_rejects_csv_checkpoint_explicitly(tmp_path, monkeypatch):
    # The command imports its sibling CLI in the same way as normal execution.
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    fork = load_tool("thermal_state_response_fork")
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"fixture")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"checkpoint_fork": {"source_checkpoint": str(checkpoint)}}))
    monkeypatch.setattr(fork, "load_trusted_checkpoint", lambda *args, **kwargs: {"history_path": "metrics/history.csv"})
    with pytest.raises(ValueError, match="CSV-recorded runs"):
        fork.main(["--recipe", str(recipe), "--output", str(tmp_path / "new")])
