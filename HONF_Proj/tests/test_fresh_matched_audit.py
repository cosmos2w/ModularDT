from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_HONF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HONF_ROOT / "tools" / "diagnostics"))
import audit_fresh_matched as audit


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _stream_fixture(tmp_path: Path, *, layout: str = "runstore", mismatch: bool = False,
                    bad_denominator: bool = False):
    manifest_path = tmp_path / "comparison_manifest.json"
    arms = {}
    seal_arms = {}
    for mode in audit.FRESH_MATCHED_MODES:
        run_id = f"{mode}-run"
        output = tmp_path / "runs" / run_id
        arms[mode] = {
            "run_id": run_id,
            "output_dir": str(output),
            "recipe_sha256": "r" * 64,
        }
        sampler_namespace = {"seed": 0, "dataset_id": "shared-dataset"}
        seal_arms[mode] = {
            "run_id": run_id,
            "recipe_sha256": "r" * 64,
            "sampler_namespace": sampler_namespace,
        }

    manifest = {
        "task": "thermal",
        "seed": 0,
        "dataset_protocol": "fixture_fixed_v1",
        "source_config_sha256": "s" * 64,
        "arms": arms,
    }
    _write_json(manifest_path, manifest)
    seal = {
        "schema": "fresh_matched_initialization_seal_v1",
        "comparison_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "task": "thermal",
        "dataset_protocol": "fixture_fixed_v1",
        "source_config_sha256": "s" * 64,
        "arms": seal_arms,
        "common_initialization_sha256": "a" * 64,
        "provider_scale_binding_sha256": "b" * 64,
        "sampler_namespace_sha256": audit._audit_sha256({"seed": 0, "dataset_id": "shared-dataset"}),
    }
    _write_json(tmp_path / "comparison_seal.json", seal)

    for mode, arm in arms.items():
        output = Path(arm["output_dir"])
        if layout == "runstore":
            identity_path = output / "configs" / "experiment_identity.json"
            history_path = output / "metrics" / "history.json"
        elif layout == "flat":
            identity_path = output / "experiment_identity.json"
            history_path = output / "history.json"
        else:
            raise ValueError(layout)

        marker = {
            "active_mode": mode,
            "common_initialization_sha256": "a" * 64,
            "provider_scale_binding_sha256": "b" * 64,
            "sampler_namespace": {"seed": 0, "dataset_id": "shared-dataset"},
        }
        provider_identity = {
            "task": "ThermalChannel",
            "arm": mode,
            "fresh_matched_comparison": True,
        }
        marker["provider_identity_sha256"] = audit._audit_sha256(provider_identity)
        identity = {
            "recipe": {
                "task": "thermal",
                "mode": mode,
                "run_id": arm["run_id"],
                "seed": 0,
                "dataset_protocol": "fixture_fixed_v1",
                "fresh_matched_comparison": True,
            },
            "provider_identity": provider_identity,
            "fresh_initialization": marker,
            "engine_config": {"effective_cases": 2, "microbatch_cases": 1},
            "sampling_dataset_id": "shared-dataset",
        }
        case_digest = "c" * 64
        if mismatch and mode == "P-R":
            case_digest = "f" * 64
        query_digest = "d" * 64
        query_updates = [{"update_index": 0, "sha256": query_digest}]
        mask_updates = [{"update_index": 0, "sha256": "e" * 64}]
        denominator_value = -1.0 if bad_denominator and mode == "P-R2" else 2.0
        denominator_updates = [{"update_index": 0, "denominators": {"loss": denominator_value}}]
        query_aggregate = hashlib.sha256(
            json.dumps([query_digest], separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        mask_aggregate = audit._audit_sha256(mask_updates)
        denominator_aggregate = audit._audit_sha256(denominator_updates)
        denominator_totals = {"loss": denominator_value}
        stream = {
            "schema": "observed_fresh_training_stream_v1",
            "case_order_sha256": case_digest,
            "case_visits": 2,
            "optimizer_updates": 1,
            "microbatches": 2,
            "query_sampling_sha256": query_aggregate,
            "query_sampling_by_update_sha256": query_updates,
            "observed_mask_sha256": mask_aggregate,
            "mask_sha256_by_update": mask_updates,
            "observed_denominators_sha256": denominator_aggregate,
            "observed_denominators_by_update": denominator_updates,
            "observed_denominator_totals": denominator_totals,
            "effective_cases": 2,
            "microbatch_cases": 1,
            "sampling_dataset_id": "shared-dataset",
            "sampler_namespace_sha256": audit._audit_sha256(marker["sampler_namespace"]),
        }
        row = {
            "epoch": 1,
            "case_order_sha256": case_digest,
            "query_sampling_sha256": query_aggregate,
            "case_visits": 2,
            "optimizer_updates": 1,
            "microbatches": 2,
            "observed_stream_audit": stream,
        }
        _write_json(identity_path, identity)
        _write_json(history_path, [row])

    return manifest_path, manifest


@pytest.mark.parametrize("layout", ["runstore", "flat"])
def test_stream_audit_accepts_runstore_and_unambiguous_legacy_layouts(tmp_path, monkeypatch, layout):
    manifest_path, manifest = _stream_fixture(tmp_path, layout=layout)
    monkeypatch.setattr(audit, "_read_fresh_comparison_manifest", lambda _path: (manifest, manifest_path))

    receipt = audit.validate_fresh_matched_streams(manifest_path)

    assert receipt["matched"] is True
    assert receipt["optimizer_updates_observed"] == 1
    assert {arm["artifact_layout"] for arm in receipt["arms"].values()} == {layout}
    assert receipt["shared_completed_age"] == 1
    assert receipt["full_mask_and_denominator_comparison_epochs"] == [1]
    assert receipt["optimizer_updates_in_this_command"] == 0


def test_artifact_resolver_rejects_conflicting_dual_layouts(tmp_path):
    output = tmp_path / "run"
    _write_json(output / "configs" / "experiment_identity.json", {"source": "runstore"})
    _write_json(output / "metrics" / "history.json", [])
    _write_json(output / "experiment_identity.json", {"source": "legacy"})
    _write_json(output / "history.json", [])

    with pytest.raises(ValueError, match="Conflicting RunStore and legacy flat artifacts"):
        audit.resolve_run_artifacts(output)


def test_stream_audit_rejects_cross_arm_stream_mismatch(tmp_path, monkeypatch):
    manifest_path, manifest = _stream_fixture(tmp_path, mismatch=True)
    monkeypatch.setattr(audit, "_read_fresh_comparison_manifest", lambda _path: (manifest, manifest_path))

    with pytest.raises(ValueError, match="streams differ across arms"):
        audit.validate_fresh_matched_streams(manifest_path)


def test_stream_audit_rejects_malformed_denominator_evidence(tmp_path, monkeypatch):
    manifest_path, manifest = _stream_fixture(tmp_path, bad_denominator=True)
    monkeypatch.setattr(audit, "_read_fresh_comparison_manifest", lambda _path: (manifest, manifest_path))

    with pytest.raises(ValueError, match="finite and nonnegative"):
        audit.validate_fresh_matched_streams(manifest_path)


def _append_epoch_two(manifest, *, mismatch_mode: str | None = None):
    for mode, arm in manifest["arms"].items():
        output = Path(arm["output_dir"])
        history_path = output / "metrics" / "history.json"
        rows = json.loads(history_path.read_text())
        epoch_two = {key: value for key, value in rows[0].items() if key != "observed_stream_audit"}
        epoch_two["epoch"] = 2
        if mismatch_mode == mode:
            epoch_two["query_sampling_sha256"] = "9" * 64
        rows.append(epoch_two)
        _write_json(history_path, rows)


def test_later_shared_age_gets_an_immutable_reusable_receipt(tmp_path, monkeypatch):
    manifest_path, manifest = _stream_fixture(tmp_path)
    monkeypatch.setattr(audit, "_read_fresh_comparison_manifest", lambda _path: (manifest, manifest_path))

    age_one = audit.validate_fresh_matched_streams(manifest_path)
    _append_epoch_two(manifest)
    age_two = audit.validate_fresh_matched_streams(manifest_path)
    age_two_repeat = audit.validate_fresh_matched_streams(manifest_path)

    assert age_one["receipt_path"] != age_two["receipt_path"]
    assert age_two["shared_completed_age"] == 2
    assert age_two_repeat["receipt_path"] == age_two["receipt_path"]
    assert Path(age_one["receipt_path"]).is_file()
    assert Path(age_two["receipt_path"]).is_file()


def test_stream_audit_rejects_mismatch_after_epoch_one(tmp_path, monkeypatch):
    manifest_path, manifest = _stream_fixture(tmp_path)
    _append_epoch_two(manifest, mismatch_mode="P-R2")
    monkeypatch.setattr(audit, "_read_fresh_comparison_manifest", lambda _path: (manifest, manifest_path))

    with pytest.raises(ValueError, match="per-epoch case/query streams or counters differ"):
        audit.validate_fresh_matched_streams(manifest_path)


def test_measured_forecast_reads_runstore_layout_and_counts_review_epochs(tmp_path):
    output = tmp_path / "run"
    identity = {
        "recipe": {
            "run_id": "fresh-run",
            "active_mode": "P-R2",
            "fresh_matched_comparison": True,
            "total_epochs": 2500,
            "comparison_protocol": audit.FRESH_MATCHED_COMPARISON_PROTOCOL,
        },
        "fresh_initialization": {"active_mode": "P-R2"},
        "provider_identity": {
            "dataset": "ThermalChannel",
            "native_query_budget": {
                "fluid_queries": 1024,
                "material_queries_per_module": 32,
                "operator_rows_per_case": 128,
            },
        },
    }
    rows = [
        {
            "epoch": 100,
            "train_seconds": 1.5,
            "validation_seconds": 0.25,
            "case_visits": 150,
            "optimizer_updates": 4,
            "validation": {"field_score": 0.5},
            "work_counts": {"evaluated_queries": 1024},
            "train_losses": {"field": 0.1},
        }
    ]
    _write_json(output / "configs" / "experiment_identity.json", identity)
    _write_json(output / "metrics" / "history.json", rows)

    forecast = audit.forecast_fresh_matched_run(output)

    assert forecast["status"] == "measured_epoch_and_validation_forecast"
    assert forecast["artifact_layout"] == "runstore"
    assert forecast["remaining_epochs"] == 2400
    assert forecast["review_epochs_remaining"] == 24
    assert forecast["forecast_remaining_seconds_including_validation"] == pytest.approx(3606.0)
    assert forecast["primary_and_auxiliary_query_budget"] == {
        "fluid_queries": 1024,
        "material_queries_per_module": 32,
        "operator_rows_per_case": 128,
    }
    assert forecast["optimizer_updates_in_this_command"] == 0
