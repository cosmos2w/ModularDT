"""An ordinary file flag acknowledged only at a completed epoch boundary."""

from pathlib import Path


def stop_requested(run_dir: Path) -> bool:
    """Create ``<run>/stop_requested`` to save latest and stop after this epoch."""
    return (run_dir / "stop_requested").is_file()


def acknowledge_stop(run_dir: Path, *, epoch: int) -> None:
    """Acknowledge only after the normal atomic latest checkpoint succeeded."""
    from honf_runtime.run_store import RunStore

    if (run_dir / "run_manifest.json").is_file():
        from honf_runtime.run_layout import resolve_checkpoint

        RunStore.update_status(run_dir, "stopped_resumable", last_completed_epoch=epoch,
                               resume_checkpoint=str(resolve_checkpoint(run_dir, "latest_model.pt")))
    (run_dir / "stop_requested").unlink(missing_ok=True)
