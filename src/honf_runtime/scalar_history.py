"""Append-only scalar tables for new run producers.

Training uses a provider-declared wide table spanning its scheduled phases.
Values and identifiers are returned as strings; callers
choose numerical conversions without losing leading zeroes in identifiers.
Recovery is exclusively the responsibility of the locked training writer.
"""

from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path
from typing import Iterable, Mapping

VALIDATION_COLUMNS = ("epoch", "partition", "case_id", "member_id", "metric", "value", "count")


def _complete_bytes(path: Path) -> tuple[bytes, bytes]:
    if not path.exists():
        return b"", b""
    raw = path.read_bytes()
    end = raw.rfind(b"\n") + 1
    return raw[:end], raw[end:]


def read_scalar_rows(path: str | Path) -> list[dict[str, str]]:
    """Read complete rows without changing a possibly active writer's file."""
    complete, _ = _complete_bytes(Path(path))
    if not complete:
        return []
    return list(csv.DictReader(io.StringIO(complete.decode("utf-8"), newline="")))


_TRAIN_GROUPS = ("train_losses", "learning_rates", "work_counts", "train_diagnostics")
_INTEGER_FIELDS = {"epoch", "case_visits", "optimizer_updates", "microbatches", "partial_update_cases"}


def training_scalar_row(row: Mapping[str, object], columns: tuple[str, ...]) -> dict[str, object]:
    """Extract declared per-epoch scalars, failing on undeclared loss terms."""
    result: dict[str, object] = {name: row.get(name) for name in columns if "." not in name}
    for group in _TRAIN_GROUPS:
        values = row.get(group, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"{group} must be a scalar mapping")
        if group in {"train_losses", "learning_rates"}:
            missing = {f"{group}.{name}" for name in values} - set(columns)
            if missing:
                raise ValueError(f"Undeclared {group} columns: {sorted(missing)}")
        for name, value in values.items():
            column = f"{group}.{name}"
            if column in columns:
                result[column] = value
    return result


def validation_scalar_rows(row: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten measured validation scalars, retaining case/member identifiers."""
    result: list[dict[str, object]] = []
    epoch = int(row["epoch"])

    def emit(value: object, metric: str, case_id: str = "", member_id: str = "", count: object = "") -> None:
        if isinstance(value, Mapping):
            count = value.get("count", count)
            for key, item in value.items():
                if key not in {"case_id", "member_id", "count", "loss_objective"}:
                    emit(item, f"{metric}.{key}" if metric else str(key), case_id, member_id, count)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    emit(item, metric, str(item.get("case_id", case_id)), str(item.get("member_id", member_id)), count)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result.append({"epoch": epoch, "partition": "VALIDATION", "case_id": case_id,
                           "member_id": member_id, "metric": metric, "value": value, "count": count})

    emit(row.get("validation", {}), "validation")
    emit(row.get("validation_objective", {}), "validation_objective")
    return result


def read_csv_training_history(path: str | Path) -> list[dict[str, object]]:
    """Reconstruct the scalar training and validation view used by the trainer."""
    path = Path(path)
    result: list[dict[str, object]] = []
    by_epoch: dict[int, dict[str, object]] = {}
    for record in read_scalar_rows(path):
        row: dict[str, object] = {}
        for name, value in record.items():
            if value == "":
                continue
            if name in ("phase", "arm"):
                row[name] = value
            elif "." in name and name.split(".", 1)[0] in _TRAIN_GROUPS:
                group, key = name.split(".", 1)
                row.setdefault(group, {})[key] = float(value)
            else:
                row[name] = int(value) if name in _INTEGER_FIELDS else float(value)
        epoch = int(row["epoch"])
        if epoch in by_epoch:
            raise ValueError(f"Duplicate scalar history epoch {epoch}")
        by_epoch[epoch] = row
        result.append(row)
    for record in read_scalar_rows(path.with_name("validation.csv")):
        epoch = int(record["epoch"])
        if epoch not in by_epoch:
            raise ValueError(f"Validation epoch {epoch} has no training record")
        # Case/member measurements remain available in the CSV; plotted and
        # selection values use only the aggregate records.
        if record["case_id"] or record["member_id"]:
            continue
        parts = record["metric"].split(".")
        target = by_epoch[epoch]
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = float(record["value"])
    return result


def append_scalar_rows(path: str | Path, columns: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    """Append scalar records and flush them before checkpoint publication."""
    path = Path(path)
    records = list(rows)
    if not records:
        return
    if not columns or columns[0] != "epoch" or len(columns) != len(set(columns)):
        raise ValueError("Scalar table columns must start with epoch and be unique")
    for row in records:
        if set(row) - set(columns):
            raise ValueError("Scalar record contains undeclared columns")
        if any(isinstance(value, (dict, list, tuple, set)) for value in row.values()):
            raise TypeError("Scalar table cells cannot contain arrays or objects")
        if any("\n" in str(value) or "\r" in str(value) for value in row.values() if value is not None):
            raise ValueError("Scalar table cells cannot contain line breaks")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        complete, trailing = _complete_bytes(path)
        if trailing:
            raise ValueError("Torn scalar table requires locked recovery before append")
        header = next(csv.reader(io.StringIO(complete.decode("utf-8"), newline="")), None)
        if header != list(columns):
            raise ValueError("Scalar table header differs from the declared schema")
    else:
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(columns)
            stream.flush()
            os.fsync(stream.fileno())
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        for row in records:
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def recover_scalar_rows(path: str | Path, completed_epoch: int, *, writer_lock_held: bool) -> Path | None:
    """Reconcile one table to the checkpoint boundary, preserving discarded bytes.

    The caller must own the existing exclusive run lock. A distinct evidence
    file is created only when bytes were actually removed; repeating recovery
    after interruption is therefore harmless.
    """
    if not writer_lock_held:
        raise RuntimeError("Scalar recovery requires the exclusive training lock")
    path = Path(path)
    if not path.exists():
        return None
    raw = path.read_bytes()
    complete, trailing = _complete_bytes(path)
    lines = complete.splitlines(keepends=True)
    if not lines:
        kept, discarded = b"", raw
    else:
        header = next(csv.reader([lines[0].decode("utf-8")]))
        if not header or header[0] != "epoch" or len(header) != len(set(header)):
            raise ValueError("Unsupported scalar table header")
        kept_parts = [lines[0]]
        discarded_parts = []
        for line in lines[1:]:
            fields = next(csv.reader([line.decode("utf-8")]))
            if len(fields) != len(header):
                raise ValueError("Malformed complete scalar record")
            (kept_parts if int(fields[0]) <= completed_epoch else discarded_parts).append(line)
        kept, discarded = b"".join(kept_parts), b"".join(discarded_parts) + trailing
    if not discarded:
        return None
    # If interruption occurred after writing evidence but before replacing the
    # table, reuse that evidence rather than creating a second copy.
    evidence = next((candidate for candidate in path.parent.glob(f"{path.stem}.recovery_*{path.suffix}")
                     if candidate.is_file() and candidate.read_bytes() == discarded), None)
    if evidence is None:
        evidence = path.with_name(f"{path.stem}.recovery_{time.time_ns()}{path.suffix}")
        with evidence.open("xb") as stream:
            stream.write(discarded)
            stream.flush()
            os.fsync(stream.fileno())
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(kept)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return evidence
