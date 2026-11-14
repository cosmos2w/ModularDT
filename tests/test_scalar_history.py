import pytest

from honf_runtime.scalar_history import (
    VALIDATION_COLUMNS,
    append_scalar_rows,
    read_scalar_rows,
    recover_scalar_rows,
)


def test_precision_blanks_and_identifiers(tmp_path):
    train = tmp_path / "metrics" / "history.csv"
    columns = ("epoch", "phase", "loss", "other")
    append_scalar_rows(train, columns, [
        {"epoch": 1, "phase": "joint", "loss": 1.2345678901234567, "other": 0},
        {"epoch": 2, "phase": "warmup", "loss": None},
    ])
    rows = read_scalar_rows(train)
    assert float(rows[0]["loss"]) == 1.2345678901234567
    assert rows[0]["other"] == "0"
    assert rows[1]["loss"] == "" and rows[1]["other"] == ""
    validation = train.with_name("validation.csv")
    append_scalar_rows(validation, VALIDATION_COLUMNS, [{
        "epoch": 2, "partition": "VALIDATION", "case_id": "001", "member_id": "02",
        "metric": "error, relative", "value": 0.0, "count": 3,
    }])
    assert read_scalar_rows(validation)[0]["case_id"] == "001"
    assert read_scalar_rows(validation)[0]["member_id"] == "02"
    assert read_scalar_rows(validation)[0]["metric"] == "error, relative"


def test_locked_recovery_preserves_orphans_and_readers_are_read_only(tmp_path):
    path = tmp_path / "history.csv"
    columns = ("epoch", "loss")
    append_scalar_rows(path, columns, [{"epoch": 1, "loss": 1.0}, {"epoch": 2, "loss": 0.5}])
    with path.open("ab") as stream:
        stream.write(b"3,0.")
    before = path.read_bytes()
    assert len(read_scalar_rows(path)) == 2
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError):
        recover_scalar_rows(path, 1, writer_lock_held=False)
    evidence = recover_scalar_rows(path, 1, writer_lock_held=True)
    assert evidence.read_bytes() == b"2,0.5\r\n3,0."
    assert [r["epoch"] for r in read_scalar_rows(path)] == ["1"]
    assert recover_scalar_rows(path, 1, writer_lock_held=True) is None
    append_scalar_rows(path, columns, [{"epoch": 2, "loss": 0.25}])
    assert [r["epoch"] for r in read_scalar_rows(path)] == ["1", "2"]


def test_header_and_scalar_contract(tmp_path):
    path = tmp_path / "history.csv"
    append_scalar_rows(path, ("epoch", "loss"), [{"epoch": 1, "loss": 0}])
    before = path.read_bytes()
    with pytest.raises(ValueError):
        append_scalar_rows(path, ("epoch", "other"), [{"epoch": 2, "other": 1}])
    with pytest.raises(TypeError):
        append_scalar_rows(path, ("epoch", "loss"), [{"epoch": 2, "loss": [1]}])
    assert path.read_bytes() == before


def test_recovery_reuses_evidence_after_interruption(tmp_path):
    path = tmp_path / "history.csv"
    append_scalar_rows(path, ("epoch", "loss"), [{"epoch": 1, "loss": 1}, {"epoch": 2, "loss": 2}])
    evidence = tmp_path / "history.recovery_previous.csv"
    evidence.write_bytes(b"2,2\r\n")
    assert recover_scalar_rows(path, 1, writer_lock_held=True) == evidence
    assert list(tmp_path.glob("history.recovery_*.csv")) == [evidence]
