import importlib.util
import json
from pathlib import Path


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
