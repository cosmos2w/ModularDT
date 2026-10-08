"""Operational state must not imply a dead or unrelated worker is running."""
import argparse
import json
import os

from unified_interaction_train import _command_clean_stop, _command_status


def test_status_rejects_a_reused_pid_with_different_start_identity(tmp_path, capsys):
    receipt = {"status": "running", "pid": os.getpid(), "process_start_ticks": "impossible-start"}
    (tmp_path / "active_process.json").write_text(json.dumps(receipt))
    assert _command_status(argparse.Namespace(run_dir=str(tmp_path))) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["process_alive"] is False
    assert reported["status"] == "stale_running_receipt"
    assert json.loads((tmp_path / "active_process.json").read_text()) == receipt


def test_status_does_not_promote_a_legacy_receipt_without_pid(tmp_path, capsys):
    (tmp_path / "active_process.json").write_text(json.dumps({"status": "running"}))
    _command_status(argparse.Namespace(run_dir=str(tmp_path)))
    reported = json.loads(capsys.readouterr().out)
    assert reported["process_alive"] is None
    assert reported["status"] == "running_receipt_liveness_unknown"


def test_repeated_clean_stop_preserves_the_pending_request_identity(tmp_path, capsys):
    args = argparse.Namespace(run_dir=str(tmp_path), note="stop at the next complete epoch")
    _command_clean_stop(args)
    capsys.readouterr()
    before = (tmp_path / "CLEAN_STOP_REQUEST.json").read_bytes()
    args.note = "another click"
    _command_clean_stop(args)
    assert (tmp_path / "CLEAN_STOP_REQUEST.json").read_bytes() == before


def test_status_reads_canonical_records_without_creating_root_copies(tmp_path, capsys):
    (tmp_path / "logs").mkdir()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "logs/active_process.json").write_text(json.dumps({"status": "completed"}))
    (tmp_path / "metrics/fit_summary.json").write_text(json.dumps({"completed_epoch": 100}))
    (tmp_path / "metrics/history.json").write_text(json.dumps([{"epoch": 100, "phase": "hard"}]))
    (tmp_path / "logs/clean_stop_consumed_abc.json").write_text('{}')
    _command_status(argparse.Namespace(run_dir=str(tmp_path)))
    reported = json.loads(capsys.readouterr().out)
    assert reported["status"] == "completed"
    assert reported["fit_summary"]["completed_epoch"] == 100
    assert reported["history"]["last_epoch"] == 100
    assert reported["consumed_clean_stop_receipts"] == [str(tmp_path / "logs/clean_stop_consumed_abc.json")]
    assert not (tmp_path / "history.json").exists()
