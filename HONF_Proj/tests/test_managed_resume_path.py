from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest
from channelthermal.plugin import ThermalChannelPlugin

from train import _resolve_resume_run_dir, _validate_managed_resume


def test_managed_tree_measure_policy_amendment_preserves_other_model_options(tmp_path) -> None:
    run_dir = tmp_path / "Run_tree"
    (run_dir / "configs").mkdir(parents=True)
    identity = {"case_id": "ThermalChannel", "model_family": "honf_forward", "workflow": "forward"}
    (run_dir / "run_manifest.json").write_text(json.dumps(identity))
    original = {"case": {"id": "ThermalChannel"}, "model_family": "honf_forward", "workflow": "forward",
                "model": {"core_honf": {"forward_architecture": "faithful_receiver_hypergraph_honf",
                    "interface_model": {"message_hidden_dim": 128}}}}
    (run_dir / "configs/resolved_config.json").write_text(json.dumps(original))
    requested = copy.deepcopy(original)
    requested["model"]["core_honf"]["interface_model"]["hypergraph_options"] = {"structural_measure_policy_version": 2}
    requested["training"] = {"campaign": {"structural_measure_policy_version": 2,
        "physical_loss_policy_version": 2, "native_loss_denominators_start_epoch": 101}}
    untouched = copy.deepcopy(requested)
    hook = ThermalChannelPlugin().managed_resume_model_matches
    with pytest.raises(ValueError, match="section 'model'"):
        _validate_managed_resume(run_dir, SimpleNamespace(effective=requested), "forward")
    assert _validate_managed_resume(run_dir, SimpleNamespace(effective=requested), "forward", model_matches=hook) == identity
    assert requested == untouched
    requested["model"]["core_honf"]["interface_model"]["message_hidden_dim"] = 64
    with pytest.raises(ValueError, match="section 'model'"):
        _validate_managed_resume(run_dir, SimpleNamespace(effective=requested), "forward", model_matches=hook)


@pytest.mark.parametrize("alteration", ["no_campaign", "wrong_activation", "reverse_policy", "other_architecture"])
def test_managed_tree_rejects_undeclared_or_wrong_policy_changes(tmp_path, alteration) -> None:
    run_dir = tmp_path / "Run_tree"
    (run_dir / "configs").mkdir(parents=True)
    identity = {"case_id": "ThermalChannel", "model_family": "honf_forward", "workflow": "forward"}
    (run_dir / "run_manifest.json").write_text(json.dumps(identity))
    original = {"case": {"id": "ThermalChannel"}, "model_family": "honf_forward", "workflow": "forward",
                "model": {"core_honf": {"forward_architecture": "faithful_receiver_hypergraph_honf",
                    "interface_model": {"hypergraph_options": {"structural_measure_policy_version": 1}}}}}
    requested = copy.deepcopy(original)
    requested["model"]["core_honf"]["interface_model"]["hypergraph_options"]["structural_measure_policy_version"] = 2
    requested["training"] = {"campaign": {"structural_measure_policy_version": 2,
        "physical_loss_policy_version": 2, "native_loss_denominators_start_epoch": 101}}
    if alteration == "no_campaign":
        requested.pop("training")
    elif alteration == "wrong_activation":
        requested["training"]["campaign"]["native_loss_denominators_start_epoch"] = 201
    elif alteration == "reverse_policy":
        original["model"]["core_honf"]["interface_model"]["hypergraph_options"]["structural_measure_policy_version"] = 2
        requested["model"]["core_honf"]["interface_model"]["hypergraph_options"]["structural_measure_policy_version"] = 1
    else:
        requested["model"]["core_honf"]["forward_architecture"] = "direct_pairwise_control_honf"
    (run_dir / "configs/resolved_config.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="section 'model'"):
        _validate_managed_resume(run_dir, SimpleNamespace(effective=requested), "forward",
            model_matches=ThermalChannelPlugin().managed_resume_model_matches)


def test_managed_checkpoint_resolves_run_root_and_rejects_identity_conflict(tmp_path) -> None:
    run_dir = tmp_path / "Run_2105_fixture"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "latest.pt"
    checkpoint.write_bytes(b"fixture")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "case_id": "ThermalChannel",
                "model_family": "honf_forward",
                "workflow": "forward",
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_resume_run_dir(checkpoint)

    assert resolved == run_dir
    assert resolved != checkpoint_dir
    bundle = SimpleNamespace(
        effective={
            "case": {"id": "WindFarm"},
            "model_family": "honf_forward",
        }
    )
    with pytest.raises(ValueError, match="Resume run identity mismatch"):
        _validate_managed_resume(resolved, bundle, "forward")


def test_checkpoint_directory_without_nearby_manifest_is_rejected(tmp_path) -> None:
    checkpoint = tmp_path / "unmanaged" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")

    with pytest.raises(FileNotFoundError, match="refusing to treat a managed checkpoints directory"):
        _resolve_resume_run_dir(checkpoint)


def test_standalone_checkpoint_keeps_its_parent_as_resume_directory(tmp_path) -> None:
    checkpoint = tmp_path / "standalone.pt"
    checkpoint.write_bytes(b"fixture")

    assert _resolve_resume_run_dir(checkpoint) == tmp_path


@pytest.mark.parametrize("history_name", ["metrics.csv", "loss_history.csv"])
def test_resumed_completed_run_records_nonzero_exit_on_plugin_exception(tmp_path, history_name) -> None:
    run_dir = tmp_path / "Run_completed_fixture"
    run_dir.mkdir()
    checkpoint = run_dir / "latest_model.pt"
    checkpoint.write_bytes(b"fixture, never loaded")
    manifest = {
        "case_id": "Fixture",
        "model_family": "honf_forward",
        "workflow": "forward",
        "run_id": "fixture",
        "status": "completed",
        "exit_code": 0,
        "started_at": "original-start",
        "ended_at": "previous-completion",
        "checkpoints": {"latest": str(checkpoint)},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / history_name).write_text("epoch,val_loss_total\n605,0.5\n606,0.4\n", encoding="utf-8")
    code = dedent(
        """
        import json
        import sys
        from types import SimpleNamespace
        import train

        def fail_training(bundle, request, *, run_dir):
            running = json.loads((run_dir / "run_manifest.json").read_text())
            assert running["status"] == "running" and running["ended_at"] is None
            assert request.run_id == "fixture"
            raise ValueError("injected continuation failure")

        bundle = SimpleNamespace(
            effective={"case": {"id": "Fixture"}, "model_family": "honf_forward",
                       "workflow": "forward", "Run_ID": "new-id", "run": {"name": "fixture"}},
            case={"plugin": "fixture:plugin"},
        )
        plugin = SimpleNamespace(case_id="Fixture", validate_config=lambda bundle: None,
                                 inspect_launch=lambda bundle, request: {}, train=fail_training)
        train.load_config_bundle = lambda *args, **kwargs: bundle
        train.load_case_plugin = lambda path: plugin
        train.print_launch_summary = lambda *args, **kwargs: None
        sys.argv = ["train.py", "--resume-checkpoint", sys.argv[1], "--yes"]
        raise SystemExit(train.main())
        """
    )
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(project_root / "src"), CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", code, str(checkpoint)],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert "ValueError: injected continuation failure" in result.stderr
    failed = json.loads((run_dir / "run_manifest.json").read_text())
    assert failed["status"] == "failed"
    assert failed["exit_code"] == result.returncode
    assert failed["last_completed_epoch"] == 606
    assert failed["error_type"] == "ValueError"
    assert failed["error_message"] == "injected continuation failure"
    assert "ValueError: injected continuation failure" in failed["traceback"]
    assert failed["started_at"] == manifest["started_at"]
    assert failed["ended_at"] not in (None, manifest["ended_at"])
    assert failed["checkpoints"] == manifest["checkpoints"]
    assert failed["resumed_from"] == str(checkpoint.resolve())
