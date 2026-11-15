import copy
import json

import pytest

from honf_runtime.scalar_history import (
    VALIDATION_COLUMNS,
    append_scalar_rows,
    read_scalar_rows,
    recover_scalar_rows,
    resolve_training_recording,
    training_recording_definitions,
    validate_scalar_header,
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


def _definitions(columns=("epoch", "train_losses.zeta", "train_losses.alpha")):
    return training_recording_definitions(columns, {
        "zeta": {"unit": "1", "formula": "weighted squared residual", "weight": "0.7"},
        "alpha": {"unit": "1", "formula": "weighted shifted residual", "weight": "0.3"},
    })


def test_saved_order_survives_json_and_declaration_reordering():
    saved = _definitions()
    incoming = json.loads(json.dumps(_definitions(("epoch", "train_losses.alpha", "train_losses.zeta")),
                                     sort_keys=True, indent=4))
    before = copy.deepcopy(saved)
    columns, order = resolve_training_recording(saved, incoming)
    assert columns == ("epoch", "train_losses.zeta", "train_losses.alpha")
    assert order == ("zeta", "alpha")
    assert saved == before


def test_legacy_definitions_restore_order_without_migration():
    saved = {key: value for key, value in _definitions().items()
             if key in {"training_columns", "terms", "training_reduction", "validation_reduction"}}
    before = json.dumps(saved)
    columns, order = resolve_training_recording(saved, _definitions(
        ("epoch", "train_losses.alpha", "train_losses.zeta")))
    assert columns == tuple(saved["training_columns"]) and order is None
    assert json.dumps(saved) == before
    from honf_runtime.unified_training import TrainingEngine
    from tests.test_unified_training import _config

    engine = TrainingEngine(_config(), device="cpu")
    engine._recorded_loss_order = order
    assert list(engine._ordered_training_terms({"alpha": 1, "zeta": 2})) == ["alpha", "zeta"]


@pytest.mark.parametrize("field,value", [("dtype", "str"), ("unit", "K"), ("meaning", "Different residual")])
def test_definition_error_names_semantic_changes_and_reordering(field, value):
    saved = _definitions()
    incoming = _definitions(("epoch", "train_losses.alpha", "train_losses.zeta"))
    incoming["training_column_definitions"]["train_losses.zeta"][field] = value
    with pytest.raises(ValueError) as error:
        resolve_training_recording(saved, incoming)
    message = str(error.value)
    assert "added=[]; missing=[]; reordered=[" in message
    assert "training_column_definitions.train_losses.zeta" in message
    assert field in message and repr(value) in message
    assert "numerical meaning changed=True" in message


def test_definition_error_names_added_and_missing_columns():
    with pytest.raises(ValueError) as error:
        resolve_training_recording(_definitions(), _definitions(("epoch", "train_losses.beta", "train_losses.zeta")))
    assert "added=['train_losses.beta']" in str(error.value)
    assert "missing=['train_losses.alpha']" in str(error.value)


def test_saved_loss_order_is_explicit_not_json_mapping_order():
    from honf_runtime.unified_training import TrainingEngine
    from tests.test_unified_training import _config

    engine = TrainingEngine(_config(), device="cpu")
    engine._recorded_loss_order = ("zeta", "middle", "alpha")
    assert list(engine._ordered_training_terms({"alpha": 1, "zeta": 2})) == ["zeta", "alpha"]
    with pytest.raises(ValueError, match="undeclared"):
        engine._ordered_training_terms({"undeclared": 1})


def test_reordered_header_reports_positions_without_writing(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("epoch,second,first\n1,2,3\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="reordered=\\[\\('first', 1, 2\\), \\('second', 2, 1\\)\\]"):
        validate_scalar_header(path, ("epoch", "first", "second"))
    assert path.read_bytes() == before
