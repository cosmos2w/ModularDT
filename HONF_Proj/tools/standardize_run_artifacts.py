"""Inspect or relocate a stopped run into the shared artifact layout.

No training, checkpoint pruning, or retrospective metric computation occurs.
Checkpoint compatibility links preserve explicit historical file references.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from honf_runtime.run_layout import RunLayout
from honf_runtime.run_store import atomic_write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_stopped(run_dir: Path) -> None:
    """Reject live process identities and live processes using this directory."""
    active = RunLayout(run_dir).read_path("active_process.json")
    if active.is_file():
        receipt = json.loads(active.read_text())
        pid = receipt.get("pid")
        if isinstance(pid, int) and pid > 0:
            try:
                stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                stat = None
            if stat and stat[0] != "Z" and (
                receipt.get("process_start_ticks") is None
                or receipt["process_start_ticks"] == stat[19]
            ):
                raise RuntimeError(f"Run process {pid} is still alive; stop it before relocation.")
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            # A terminal tail of the sibling console log is not a run writer.
            if str(run_dir) in command and "standardize_run_artifacts" not in command and "tail " not in command:
                raise RuntimeError(f"Process {process.name} still references this run.")
            cwd = (process / "cwd").resolve(strict=True)
            if cwd == run_dir or run_dir in cwd.parents:
                raise RuntimeError(f"Process {process.name} has a working directory inside this run.")
            for descriptor in (process / "fd").iterdir():
                try:
                    opened = descriptor.resolve(strict=True)
                except (PermissionError, FileNotFoundError, ProcessLookupError):
                    continue
                if opened == run_dir or run_dir in opened.parents:
                    raise RuntimeError(f"Process {process.name} has an open artifact inside this run.")
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue


def relocation_plan(run_dir: Path) -> dict:
    layout = RunLayout(run_dir)
    moves, unchanged = [], []
    destinations: dict[Path, Path] = {}
    for source in sorted(run_dir.iterdir()):
        if not source.is_file() or source.is_symlink():
            continue
        try:
            destination = layout.path(source.name)
        except ValueError:
            unchanged.append(source.name)
            continue
        if source == destination:
            continue
        digest = sha256(source)
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Destination exists; explicit duplicate review required: {destination}")
        if destination in destinations:
            raise ValueError(
                f"Multiple source artifacts map to {destination}: "
                f"{destinations[destination].name}, {source.name}; explicit duplicate review required."
            )
        destinations[destination] = source
        moves.append({"source": source.name, "destination": str(destination.relative_to(run_dir)),
                      "sha256": digest, "size_bytes": source.stat().st_size,
                      "compatibility_link": source.suffix in {".pt", ".pth", ".ckpt"}})
    return {"schema_version": 1, "run_dir": str(run_dir), "moves": moves,
            "unclassified_root_files": unchanged, "training_launched": False}


def standardize(run_dir: Path, *, apply: bool = False) -> dict:
    run_dir = run_dir.expanduser().resolve(strict=True)
    if not run_dir.is_dir():
        raise ValueError("Expected an existing run directory.")
    if not apply:
        return {"status": "planned_only", **relocation_plan(run_dir)}
    # The same lock used by the shared trainer closes the liveness-check race.
    with (run_dir / ".training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_stopped(run_dir)
        receipt = relocation_plan(run_dir)
        layout = RunLayout(run_dir)
        receipt_path = run_dir / "logs" / "artifact_relocation.json"
        if receipt_path.exists():
            if receipt["moves"]:
                raise ValueError("A previous relocation receipt exists; review it before a second relocation.")
            return {"status": "already_standardized", "receipt": str(receipt_path)}
        atomic_write_json(receipt_path, {"status": "planned", **receipt})
        for move in receipt["moves"]:
            source, destination = run_dir / move["source"], run_dir / move["destination"]
            if sha256(source) != move["sha256"]:
                raise RuntimeError(f"Artifact changed after inventory: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            if move["compatibility_link"]:
                source.symlink_to(destination.relative_to(run_dir))
            if sha256(destination) != move["sha256"]:
                raise RuntimeError(f"Relocated artifact failed checksum: {destination}")
        layout.ensure()
        atomic_write_json(run_dir / "artifact_layout.json", {"artifact_layout_version": 1})
        # Old shared runs saved these two names from the very same payload.
        # Share their inode only after proving byte identity; all paths remain.
        last, latest = run_dir / "checkpoints/last.pt", run_dir / "checkpoints/latest_model.pt"
        deduplicated = []
        if last.is_file() and latest.is_file() and sha256(last) == sha256(latest):
            temporary = last.with_name(f".last.{os.getpid()}.link")
            os.link(latest, temporary)
            os.replace(temporary, last)
            deduplicated.append({"alias": "checkpoints/last.pt", "source": "checkpoints/latest_model.pt",
                                 "sha256": sha256(latest)})
        receipt["byte_identical_checkpoint_aliases"] = deduplicated
        receipt.update(status="completed", verified_artifacts=len(receipt["moves"]))
        atomic_write_json(receipt_path, receipt)
        return {"status": "completed", "receipt": str(receipt_path),
                "verified_artifacts": len(receipt["moves"]),
                "compatibility_checkpoint_links": sum(item["compatibility_link"] for item in receipt["moves"])}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="relocate an inactive run after checksum inventory")
    args = parser.parse_args(argv)
    print(json.dumps(standardize(args.run_dir, apply=args.apply), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
