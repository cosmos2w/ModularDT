from __future__ import annotations

import json
from datetime import datetime as DateTime
from datetime import timedelta, timezone

import pytest

import honf_runtime.run_store as run_store_module
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.run_layout import RunLayout, resolve_checkpoint
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
    assert manifest["artifact_layout_version"] == 1
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


def test_finalize_artifacts_inventories_legacy_files_without_copying(tmp_path) -> None:
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
    assert inventory["best_total"] == str(run_dir / "best_model.pt")
    assert inventory["latest"] == str(run_dir / "latest_model.pt")
    assert not (run_dir / "checkpoints" / "best_model.pt").exists()
    assert not (run_dir / "metrics" / "metrics.csv").exists()
    assert not (run_dir / "metrics" / "summary.json").exists()
    assert not (run_dir / "plots" / "training" / "loss_curve.png").exists()
    assert metrics == {"last_completed_epoch": 1, "best_metrics": {}}


def test_run_layout_maps_artifacts_and_prefers_canonical_reads(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "artifact_layout.json").write_text('{"artifact_layout_version": 1}\n', encoding="utf-8")
    layout = RunLayout(run_dir)

    assert layout.path("latest_model.pt") == run_dir / "checkpoints" / "latest_model.pt"
    assert layout.path("best_field.pt") == run_dir / "checkpoints" / "best_by_field_mse_model.pt"
    assert layout.path("history.json") == run_dir / "metrics" / "history.json"
    assert layout.path("validation_epoch_0010.json") == run_dir / "evaluations" / "validation" / "validation_epoch_0010.json"
    assert layout.path("diagnostics/routing.png") == run_dir / "plots" / "diagnostics" / "routing.png"
    assert layout.path("formal_recipe.json") == run_dir / "configs" / "formal_recipe.json"
    assert layout.path("active_process.json") == run_dir / "logs" / "active_process.json"
    assert layout.path("software.json") == run_dir / "environment" / "software.json"

    legacy = run_dir / "latest_model.pt"
    legacy.write_bytes(b"legacy")
    canonical = layout.path("latest_model.pt")
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical")
    assert layout.read_path("latest_model.pt") == canonical
    assert layout.write_path("latest_model.pt") == canonical


def test_run_layout_preserves_unmarked_legacy_writes_and_reads(tmp_path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "latest_model.pt").write_bytes(b"latest")
    layout = RunLayout(run_dir)

    assert layout.write_path("latest_model.pt") == run_dir / "latest_model.pt"
    assert layout.write_path("history.json") == run_dir / "history.json"
    assert layout.write_path("plots/diagnostics/routing.png") == run_dir / "diagnostic_plots" / "routing.png"
    assert layout.read_path("latest_model.pt") == run_dir / "latest_model.pt"
    (run_dir / "formal_recipe.json").write_text('{"source":"legacy-root"}\n', encoding="utf-8")
    (run_dir / "configs").mkdir()
    (run_dir / "configs" / "fit_identity.json").write_text('{"source":"legacy-category"}\n', encoding="utf-8")
    (run_dir / "software.json").write_text('{"source":"legacy-root"}\n', encoding="utf-8")
    assert layout.read_path("formal_recipe.json") == run_dir / "formal_recipe.json"
    assert layout.read_path("fit_identity.json") == run_dir / "configs" / "fit_identity.json"
    assert layout.read_path("software.json") == run_dir / "software.json"


def test_unmarked_legacy_reads_prefer_active_metrics_and_log_writers(tmp_path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "latest_model.pt").write_bytes(b"legacy latest")
    (run_dir / "history.json").write_text('[{"epoch": 9}]\n', encoding="utf-8")
    (run_dir / "progress.json").write_text('{"epoch": 9}\n', encoding="utf-8")
    (run_dir / "metrics").mkdir()
    (run_dir / "metrics" / "history.json").write_text('[{"epoch": 4}]\n', encoding="utf-8")
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "progress.json").write_text('{"epoch": 4}\n', encoding="utf-8")

    layout = RunLayout(run_dir)

    assert layout.read_path("history.json") == run_dir / "history.json"
    assert layout.read_path("progress.json") == run_dir / "progress.json"


def test_unmarked_legacy_config_and_environment_reads_remain_category_first(tmp_path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "latest_model.pt").write_bytes(b"legacy latest")
    (run_dir / "configs").mkdir()
    (run_dir / "configs" / "fit_identity.json").write_text('{"source":"category"}\n', encoding="utf-8")
    (run_dir / "fit_identity.json").write_text('{"source":"root"}\n', encoding="utf-8")
    (run_dir / "environment").mkdir()
    (run_dir / "environment" / "software.json").write_text('{"source":"category"}\n', encoding="utf-8")
    (run_dir / "software.json").write_text('{"source":"root"}\n', encoding="utf-8")

    layout = RunLayout(run_dir)

    assert layout.read_path("fit_identity.json") == run_dir / "configs" / "fit_identity.json"
    assert layout.read_path("software.json") == run_dir / "environment" / "software.json"


def test_resolve_checkpoint_prefers_canonical_and_supports_legacy_aliases(tmp_path) -> None:
    run_dir = tmp_path / "checkpoints"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "artifact_layout.json").write_text('{"artifact_layout_version": 1}\n', encoding="utf-8")
    canonical = run_dir / "checkpoints" / "best_by_field_mse_model.pt"
    canonical.write_bytes(b"canonical")
    (run_dir / "best_by_field_mse_model.pt").write_bytes(b"legacy")

    assert resolve_checkpoint(run_dir, "best_field") == canonical
    assert resolve_checkpoint(run_dir, run_dir / "best_by_field_mse_model.pt") == (run_dir / "best_by_field_mse_model.pt")
    assert resolve_checkpoint(run_dir, "best_field.pt") == canonical
    (run_dir / "checkpoints" / "best_by_field_mse_model.pt").unlink()
    assert resolve_checkpoint(run_dir, "best_field") == run_dir / "best_by_field_mse_model.pt"


def test_legacy_checkpoint_alias_does_not_select_a_stale_canonical_mirror(tmp_path) -> None:
    run_dir = tmp_path / "legacy"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "latest_model.pt").write_bytes(b"authoritative root latest")
    (run_dir / "latest.pt").write_bytes(b"stale root alias")
    (run_dir / "checkpoints" / "latest_model.pt").write_bytes(b"stale canonical mirror")

    assert resolve_checkpoint(run_dir, "latest.pt") == run_dir / "latest_model.pt"


def test_run_store_finalizer_inventories_canonical_checkpoints_and_history(tmp_path) -> None:
    run_dir = tmp_path / "Run_0002_canonical"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    (run_dir / "artifact_layout.json").write_text('{"artifact_layout_version": 1}\n', encoding="utf-8")
    (run_dir / "checkpoints" / "latest_model.pt").write_bytes(b"latest")
    (run_dir / "checkpoints" / "best_by_field_mse_model.pt").write_bytes(b"best")
    (run_dir / "metrics" / "history.json").write_text('[{"epoch": 4}]\n', encoding="utf-8")
    (run_dir / "metrics" / "fit_summary.json").write_text('{"best_field_score": 0.25}\n', encoding="utf-8")

    inventory = RunStore.finalize_artifacts(run_dir)
    metrics = RunStore.metric_summary(run_dir)

    assert inventory["latest"] == str(run_dir / "checkpoints" / "latest_model.pt")
    assert inventory["best_field"] == str(run_dir / "checkpoints" / "best_by_field_mse_model.pt")
    assert metrics == {"last_completed_epoch": 4, "best_metrics": {"best_field_score": 0.25}}


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
