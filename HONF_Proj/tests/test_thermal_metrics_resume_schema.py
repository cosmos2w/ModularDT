"""A declared response amendment preserves historical metric cells and alignment."""
import csv

import pytest
from channelthermal.training.reporting import repair_metrics_csv_for_append, write_metrics_row


def test_declared_resume_columns_preserve_history_and_backup_then_append(tmp_path):
    path = tmp_path / "metrics.csv"
    raw = b"epoch,val_field_mse,diagnostic\r\n100,0.6260804533958435,nan\r\n"
    path.write_bytes(raw)
    requested = ["epoch", "heat_null_loss", "val_field_mse", "diagnostic", "heat_null_wrapper_calls"]
    actual = repair_metrics_csv_for_append(path, requested)
    assert actual == ["epoch", "val_field_mse", "diagnostic", "heat_null_loss", "heat_null_wrapper_calls"]
    assert path.with_name("metrics.csv.before_schema_extension").read_bytes() == raw
    write_metrics_row(path, actual, {"epoch":101, "val_field_mse":.5, "diagnostic":"nan",
                                   "heat_null_loss":.002, "heat_null_wrapper_calls":8})
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0] == {"epoch":"100", "val_field_mse":"0.6260804533958435", "diagnostic":"nan",
                       "heat_null_loss":"", "heat_null_wrapper_calls":""}
    assert rows[1]["epoch"] == "101" and rows[1]["heat_null_wrapper_calls"] == "8"
    before_repeat = path.read_bytes()
    assert repair_metrics_csv_for_append(path, requested) == actual
    assert path.read_bytes() == before_repeat
    assert path.with_name("metrics.csv.before_schema_extension").read_bytes() == raw
    with pytest.raises(ValueError, match="fields not in fieldnames"):
        write_metrics_row(path, actual, {"epoch":102, "undeclared_auxiliary":1})


def test_malformed_history_is_preserved_before_rejected_schema_extension(tmp_path):
    path = tmp_path / "metrics.csv"
    raw = b"epoch,loss\n100,.5,unexpected\n"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="malformed metrics CSV"):
        repair_metrics_csv_for_append(path, ["epoch", "loss", "heat_null_loss"])
    assert path.read_bytes() == raw


def test_empty_metrics_file_gets_header_on_first_append(tmp_path):
    path = tmp_path / "metrics.csv"
    path.touch()
    names = repair_metrics_csv_for_append(path, ["epoch", "loss"])
    write_metrics_row(path, names, {"epoch":1, "loss":.5})
    assert list(csv.DictReader(path.open())) == [{"epoch":"1", "loss":"0.5"}]
