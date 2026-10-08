"""Operational state must not imply a dead or unrelated worker is running."""
import argparse
import json
import os

import pytest
import unified_interaction_train as development_cli
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


def test_mathematical_route_declaration_is_forwarded_only_for_new_branch_start(tmp_path, monkeypatch, capsys):
    declaration = {
        "kind": "compact_c1_gate_branch",
        "from_gate_version": "hard_v1",
        "to_gate_version": "compact_c1_v1",
        "gate_transition": [0.35, 0.65],
        "evidence_path": str(tmp_path / "diagnosis.json"),
        "evidence_sha256": "sealed-by-engine",
        "reason": "TRAIN seam input-VJP diagnosis",
    }
    route_path = tmp_path / "mathematical_route_branch.json"
    route_path.write_text(json.dumps(declaration))

    calls = {}

    class Engine:
        def fit(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return {"status": "started"}

    monkeypatch.setattr(
        development_cli,
        "_build",
        lambda _args: ("model", "provider", {}, Engine(), {"identity": "sealed"}),
    )
    args = development_cli.build_parser().parse_args([
        "start", "--task", "wind", "--run-id", "route-child", "--arm", "adaptive_detail",
        "--stop-after", "900", "--branch-from", str(tmp_path / "warmup/latest_model.pt"),
        "--output-dir", str(tmp_path / "child"), "--mathematical-route-branch-json", str(route_path),
    ])
    assert args.handler(args) == 0
    capsys.readouterr()
    assert calls["kwargs"]["mathematical_route_branch"] == declaration
    assert calls["kwargs"]["branch_from_checkpoint"] == str(tmp_path / "warmup/latest_model.pt")
    assert "resume_checkpoint" not in calls["kwargs"]


@pytest.mark.parametrize("arm,branch_from", [("warmup", None), ("adaptive_detail", None)])
def test_mathematical_route_declaration_requires_a_matched_branch_before_build(
    tmp_path, monkeypatch, arm, branch_from,
):
    route_path = tmp_path / "mathematical_route_branch.json"
    route_path.write_text("{}")
    monkeypatch.setattr(development_cli, "_build", lambda _args: pytest.fail("Invalid route reached model build."))
    argv = ["start", "--task", "wind", "--arm", arm, "--stop-after", "900",
            "--mathematical-route-branch-json", str(route_path)]
    if branch_from:
        argv.extend(["--branch-from", branch_from])
    args = development_cli.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match="new matched warmup child"):
        args.handler(args)


def test_resume_parser_does_not_accept_start_only_mathematical_route_declaration():
    with pytest.raises(SystemExit):
        development_cli.build_parser().parse_args([
            "resume", "--task", "wind", "--arm", "adaptive_detail", "--stop-after", "900",
            "--mathematical-route-branch-json", "declaration.json",
        ])
