from __future__ import annotations

import json
from pathlib import Path

from honf_runtime.artifact_layout import (
    EVALUATION_LAYOUT_VERSION,
    EvaluationArtifactLayout,
    default_evaluation_root,
    finalize_evaluation_job,
    find_managed_run_dir,
)


def _managed_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "Run_0001_fixture"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"schema_version": 1, "evaluations": []}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_evaluation_root_is_owned_by_run_for_legacy_and_canonical_checkpoints(tmp_path) -> None:
    run_dir = _managed_run(tmp_path)
    legacy_checkpoint = run_dir / "best_model.pt"
    canonical_checkpoint = run_dir / "checkpoints" / "best_total.pt"
    canonical_checkpoint.parent.mkdir()
    legacy_checkpoint.write_bytes(b"legacy")
    canonical_checkpoint.write_bytes(b"canonical")

    expected = run_dir / "evaluations" / "single_case"
    assert find_managed_run_dir(legacy_checkpoint) == run_dir
    assert find_managed_run_dir(canonical_checkpoint) == run_dir
    assert default_evaluation_root(legacy_checkpoint) == expected
    assert default_evaluation_root(canonical_checkpoint) == expected


def test_evaluation_layout_is_categorized_and_manifested_without_aliases(tmp_path) -> None:
    run_dir = _managed_run(tmp_path)
    checkpoint = run_dir / "best_model.pt"
    checkpoint.write_bytes(b"checkpoint")
    job = run_dir / "evaluations" / "single_case" / "0653_fixture"
    layout = EvaluationArtifactLayout.at(job)
    layout.ensure("fields", "organization", "metrics")
    (layout.fields / "global_field_quicklook_predicted.png").write_bytes(b"field")
    (layout.organization / "organization_overview.png").write_bytes(b"organization")
    (layout.metrics / "metrics_predicted.csv").write_text("mse\n0.1\n", encoding="utf-8")
    (layout.root / "summary.json").write_text("{}\n", encoding="utf-8")

    manifest_path = finalize_evaluation_job(
        layout.root,
        kind="forward_single_case",
        checkpoint_path=checkpoint,
        requested_checkpoint="best",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = {entry["path"] for entry in manifest["artifacts"]}

    assert manifest["artifact_layout_version"] == EVALUATION_LAYOUT_VERSION
    assert artifact_paths == {
        "fields/global_field_quicklook_predicted.png",
        "metrics/metrics_predicted.csv",
        "organization/organization_overview.png",
        "summary.json",
    }
    assert not (layout.root / "organizer_visualization.png").exists()
    assert not (layout.root / "organization_matrices.png").exists()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["evaluations"] == [str(layout.root.resolve())]
