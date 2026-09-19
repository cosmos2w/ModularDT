from __future__ import annotations

import json
import os
from datetime import datetime as DateTime
from datetime import timedelta, timezone

import pytest

import honf_runtime.run_store as run_store_module
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.run_store import RunStore


def _freeze_run_store_clock(monkeypatch, current: DateTime) -> dict[str, DateTime]:
    clock = {"value": current}

    class FixedDateTime:
        @staticmethod
        def now(tz=None) -> DateTime:
            value = clock["value"]
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(run_store_module, "datetime", FixedDateTime)
    return clock


def test_run_store_writes_provenance_and_allows_timestamped_same_id(tmp_path, monkeypatch) -> None:
    bundle = load_config_bundle("project://src/config_core/forward/hyper_plus_global_near.json")
    store = RunStore(tmp_path)
    clock = _freeze_run_store_clock(monkeypatch, DateTime(2026, 1, 1, tzinfo=timezone.utc))
    proposal = store.propose(
        case_id="ThermalChannel",
        workflow="forward",
        model_family="honf_forward",
        run_id="1406",
        run_name="test",
    )
    run_dir = store.create(proposal, bundle, launch_facts={"dataset ID": "fixture_v1"})
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "created"
    assert manifest["display_name"] == "test"
    assert manifest["started_at"] is None and manifest["ended_at"] is None
    assert manifest["case_schema_version"] == 1
    assert manifest["core_schema_version"] == 1
    assert manifest["environment"]["python"]
    assert "dirty" in manifest["source_state"]
    assert manifest["launch_resources"]["dataset ID"] == "fixture_v1"
    provenance = json.loads((run_dir / "configs" / "config_provenance.json").read_text())
    assert len(provenance["core_source_sha256"]) == 64
    assert (run_dir / "environment" / "software.json").is_file()
    assert (run_dir / "checkpoints").is_dir()
    RunStore.update_status(run_dir, "failed", error_type="Injected", error_message="test", traceback="trace")
    RunStore.update_status(run_dir, "running")
    resumed = json.loads((run_dir / "run_manifest.json").read_text())
    assert "error_type" not in resumed and "traceback" not in resumed
    assert resumed["started_at"] is not None and resumed["ended_at"] is None
    RunStore.update_status(run_dir, "completed")
    completed = json.loads((run_dir / "run_manifest.json").read_text())
    assert completed["ended_at"] is not None
    assert completed["updated_at"] == completed["ended_at"]
    assert (run_dir / "configs" / "resolved_config.json").is_file()

    clock["value"] += timedelta(seconds=1)
    second_proposal = store.propose(
        case_id="ThermalChannel",
        workflow="forward",
        model_family="honf_forward",
        run_id="1406",
        run_name="test",
    )
    second_run_dir = store.create(second_proposal, bundle)
    assert second_run_dir != run_dir
    expected_stamp = clock["value"].astimezone().strftime("%Y%m%d_%H%M%S")
    assert second_run_dir.name == f"Run_1406_{expected_stamp}_test"
    assert run_dir.is_dir() and second_run_dir.is_dir()
    assert json.loads((second_run_dir / "run_manifest.json").read_text())["run_id"] == "1406"


def test_run_store_rejects_exact_timestamped_path_collision(tmp_path, monkeypatch) -> None:
    store = RunStore(tmp_path)
    _freeze_run_store_clock(monkeypatch, DateTime(2026, 1, 1, tzinfo=timezone.utc))
    kwargs = {
        "case_id": "ThermalChannel",
        "workflow": "forward",
        "model_family": "honf_forward",
        "run_id": "1406",
        "run_name": "collision",
    }
    proposal = store.propose(**kwargs)
    proposal.path.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing run directory"):
        store.propose(**kwargs)


def test_finalize_artifacts_populates_canonical_tree(tmp_path) -> None:
    run_dir = tmp_path / "Run_0001_fixture"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"checkpoint")
    (run_dir / "latest_model.pt").write_bytes(b"latest")
    (run_dir / "loss_history.csv").write_text("epoch,val_loss_total\n1,0.5\n", encoding="utf-8")
    (run_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "loss_curve.png").write_bytes(b"plot")

    inventory = RunStore.finalize_artifacts(run_dir)
    metrics = RunStore.metric_summary(run_dir)

    assert set(inventory) == {"best_total", "latest"}
    assert (run_dir / "checkpoints" / "best_total.pt").read_bytes() == b"checkpoint"
    assert (run_dir / "metrics" / "metrics.csv").is_file()
    assert (run_dir / "metrics" / "summary.json").is_file()
    assert (run_dir / "plots" / "training" / "loss_curve.png").is_file()
    assert metrics == {"last_completed_epoch": 1, "best_metrics": {}}
    if hasattr(os, "stat"):
        assert (run_dir / "best_model.pt").stat().st_size == (run_dir / "checkpoints" / "best_total.pt").stat().st_size


def test_run_store_snapshots_overlay_and_unmodified_sources(tmp_path) -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/enhanced_honf_pairwise.json",
        experiment_overlay="project://src/config_core/forward/experiments/old_parity.json",
    )
    store = RunStore(tmp_path)
    proposal = store.propose(
        case_id="ThermalChannel",
        workflow="forward",
        model_family="honf_forward",
        run_id="0043",
        run_name="overlay",
    )
    run_dir = store.create(proposal, bundle)
    source = json.loads((run_dir / "configs" / "core_source.json").read_text())
    resolved = json.loads((run_dir / "configs" / "resolved_config.json").read_text())
    overlay = json.loads((run_dir / "configs" / "experiment_overlay.json").read_text())
    provenance = json.loads((run_dir / "configs" / "config_provenance.json").read_text())
    assert source["model"]["core_honf"]["use_hyper_value_context"] is True
    assert resolved["model"]["core_honf"]["use_hyper_value_context"] is False
    assert overlay["schema_version"] == 1
    assert len(provenance["experiment_overlay_sha256"]) == 64
