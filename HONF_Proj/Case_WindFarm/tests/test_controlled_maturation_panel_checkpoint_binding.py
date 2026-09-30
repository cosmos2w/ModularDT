import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_controlled_maturation_panel.py"
SPEC = importlib.util.spec_from_file_location("wind_panel_checkpoint_binding", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PANEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PANEL)


def test_checkpoint_binding_uses_requested_immutable_review_after_later_endpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "arms" / "g_packet" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    old_checkpoint = checkpoint_dir / "updates_000610.pt"
    new_checkpoint = checkpoint_dir / "updates_001120.pt"
    old_checkpoint.write_bytes(b"old checkpoint fixture")
    new_checkpoint.write_bytes(b"new checkpoint fixture")
    old_sha = PANEL.runner._file_sha256(old_checkpoint)
    new_sha = PANEL.runner._file_sha256(new_checkpoint)
    review_log = run_dir / "reviews.jsonl"
    review_log.parent.mkdir(parents=True, exist_ok=True)
    review_log.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "arm": "g_packet",
                    "update_count": 610,
                    "checkpoint": str(old_checkpoint),
                    "checkpoint_sha256": old_sha,
                },
                {
                    "arm": "g_packet",
                    "update_count": 1120,
                    "checkpoint": str(new_checkpoint),
                    "checkpoint_sha256": new_sha,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    path, record, line_sha = PANEL._checkpoint_review_binding(
        run_dir, arm="g_packet", update_count=610
    )

    assert path == old_checkpoint.resolve()
    assert record["checkpoint_sha256"] == old_sha
    assert len(line_sha) == 64
    verified_path, verified_record, verified_sha, source = PANEL._verified_checkpoint_binding(
        run_dir, arm="g_packet", update_count=610
    )
    assert (verified_path, verified_record, verified_sha, source) == (
        path, record, line_sha, "append_only_scheduled_review"
    )


def test_exposure_excludes_archived_updates_after_resume(tmp_path: Path) -> None:
    arm_dir = tmp_path / "arms" / "g_packet"
    arm_dir.mkdir(parents=True)
    updates = arm_dir / "updates.jsonl"

    def row(update: int, key: str, case: int) -> dict[str, object]:
        return {
            "update_count": update,
            "optimizer_attempt_key": key,
            "row": case,
            "full_access_replay": False,
            "requested_action": "two_packet",
            "capacity_vector": {"MM": 0.9, "QE": 0.95},
            "requested_cut_paths": ["L", "R"],
            "realized_cut_paths": ["L", "R"],
            "realized_nonredundant_k": 2,
            "route_work": {},
            "permission_status": {},
            "route_optimizer_step_applied": True,
        }

    updates.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                row(1751, "interrupted:1751", 0),
                row(1751, "resumed:1751", 0),
                row(1752, "resumed:1752", 1),
            )
        ) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate update IDs"):
        PANEL._summarize_training_exposure(
            updates, checkpoint_update=1752,
            train_rows=[0, 1], layout_indices=np.asarray([0, 1]),
        )

    (arm_dir / "recovery_events.jsonl").write_text(
        json.dumps({
            "event": "resume_archived_uncheckpointed_tail",
            "resume_checkpoint_update": 1750,
            "orphaned_update_attempt_ids": ["interrupted:1751"],
            "orphaned_attempt_keys": ["interrupted:1751"],
        }) + "\n",
        encoding="utf-8",
    )
    groups, detailed = PANEL._summarize_training_exposure(
        updates, checkpoint_update=1752,
        train_rows=[0, 1], layout_indices=np.asarray([0, 1]),
    )

    assert sum(group["sparse_optimizer_visits"] for group in groups.values()) == 2
    assert [item["update_count"] for item in detailed] == [1751, 1752]
    assert [item["row"] for item in detailed] == [0, 1]


def test_current_durability_checkpoint_binds_without_false_review(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    arm_dir = run_dir / "arms" / "g_packet"
    checkpoint_dir = arm_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "updates_001750.pt"
    checkpoint.write_bytes(b"durable checkpoint fixture")
    checkpoint_sha = PANEL.runner._file_sha256(checkpoint)
    pointer = {
        "run_id": PANEL.RUN_ID,
        "arm": "g_packet",
        "update_count": 1750,
        "checkpoint_kind": "durability_only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "attempt_id": "interrupted_attempt",
        "optimizer_attempt_key": "interrupted_attempt:1750",
    }
    pointer_path = arm_dir / "latest_checkpoint.json"
    pointer_path.write_text(json.dumps(pointer) + "\n", encoding="utf-8")
    (arm_dir / "updates.jsonl").write_text(
        json.dumps({
            "update_count": 1750,
            "attempt_id": "interrupted_attempt",
            "optimizer_attempt_key": "interrupted_attempt:1750",
        }) + "\n",
        encoding="utf-8",
    )

    path, bound, record_sha, source = PANEL._verified_checkpoint_binding(
        run_dir, arm="g_packet", update_count=1750
    )

    assert path == checkpoint.resolve()
    assert bound == pointer
    assert record_sha == PANEL.hashlib.sha256(pointer_path.read_bytes()).hexdigest()
    assert source == "current_durability_checkpoint"
    assert not (run_dir / "reviews.jsonl").exists()

    pointer["checkpoint_sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="durability checkpoint binding is invalid"):
        PANEL._verified_checkpoint_binding(run_dir, arm="g_packet", update_count=1750)
