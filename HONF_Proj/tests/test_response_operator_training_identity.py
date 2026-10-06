"""Schedule-child retention and strict pre-update identity rejection."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import thermal_dependency_flow_fit as flow_fit
import thermal_source_response_fit as source_fit
from channelthermal.dependency_flow import ThermalFlowReader
from channelthermal.training.checkpoints import _file_sha256, atomic_save_checkpoint_payload

from honf_runtime.compat import load_trusted_checkpoint


def same_tensor_tree(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_tensor_tree(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same_tensor_tree(a, b) for a, b in zip(left, right))
    return left == right


def test_child_schedule_preserves_weights_moments_and_round_trips(tmp_path):
    torch.manual_seed(0)
    model = ThermalFlowReader("D-sep")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    # Actual native-shaped reader updates create nontrivial Adam moments.
    structure = {"module_centers": torch.tensor([[[2., 2.], [5., 3.]]]),
        "module_present": torch.ones(1, 2), "heat_powers": torch.ones(1, 2),
        "material_params": torch.tensor([[.01, .01, .02, 1., 1., .45]]),
        "re": torch.tensor([[100.]]), "u_in": torch.ones(1, 1)}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(structure, torch.tensor([[[1., 1.], [6., 2.]]])).square().mean().backward()
        optimizer.step()
    weights = copy.deepcopy(model.state_dict())
    moments = copy.deepcopy(optimizer.state_dict()["state"])
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] = 1e-6
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda completed: flow_fit.flow_child_learning_rate(completed + 1) / 1e-6)
    assert optimizer.param_groups[0]["lr"] == 1e-6
    assert same_tensor_tree(moments, optimizer.state_dict()["state"])
    assert same_tensor_tree(weights, model.state_dict())
    path = tmp_path / "child.pt"
    atomic_save_checkpoint_payload(path, {"flow_state_dict": weights,
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()})
    saved = load_trusted_checkpoint(path, map_location="cpu")
    restored = ThermalFlowReader("D-sep")
    restored.load_state_dict(saved["flow_state_dict"], strict=True)
    other_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-6)
    other_scheduler = torch.optim.lr_scheduler.LambdaLR(other_optimizer,
        lambda completed: flow_fit.flow_child_learning_rate(completed + 1) / 1e-6)
    other_optimizer.load_state_dict(saved["optimizer_state_dict"])
    other_scheduler.load_state_dict(saved["scheduler_state_dict"])
    assert same_tensor_tree(weights, restored.state_dict())
    assert same_tensor_tree(optimizer.state_dict(), other_optimizer.state_dict())
    assert scheduler.state_dict() == other_scheduler.state_dict()
    assert flow_fit.flow_child_learning_rate(20) == pytest.approx(1e-4)
    assert flow_fit.flow_child_learning_rate(500) == pytest.approx(1e-4)
    assert flow_fit.flow_child_learning_rate(1500) == pytest.approx(1e-6)


def test_historical_flow_guard_and_readonly_control_remain(tmp_path):
    args = ["--parent", str(tmp_path / "missing.pt"), "--output", str(tmp_path), "--epochs", "2500"]
    with pytest.raises(ValueError, match="Historical flow fits"):
        flow_fit.main([*args, "--policy", "D-sep"])
    with pytest.raises(ValueError, match="D-sep only"):
        flow_fit.main([*args, "--policy", "D-open", "--schedule-parent", str(tmp_path / "missing.pt")])


@pytest.mark.parametrize("change", [None, "schedule", "parent_digest"])
def test_child_driver_accepts_only_its_checkpoint_identity(tmp_path, monkeypatch, change):
    parent, attachment, resume = [tmp_path / name for name in ("thermal.pt", "flow1000.pt", "child.pt")]
    atomic_save_checkpoint_payload(parent, {"fixture": "normalization lineage"})
    manifest = {"manifest_sha256": "fixed-membership"}
    cases = [{"case_id": "TRAIN_0001"}]
    model = ThermalFlowReader("D-sep")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    historical = {"dependency_identity": flow_fit.DEPENDENCY_ID, "dependency_policy": "D-sep",
        "case_capability": flow_fit.CASE_CAPABILITY, "thermal_parent_checkpoint": str(parent.resolve()),
        "thermal_parent_sha256": _file_sha256(parent), "flow_reader_config": model.reader.config,
        "seed": 0, "points_per_case": 1024, "microbatch_cases": 8, "effective_cases": 48,
        "schedule_total_epochs": 1000, "development_manifest_sha256": manifest["manifest_sha256"],
        "training_case_ids": ["TRAIN_0001"], "validation_case_ids": ["TRAIN_0001"],
        "learning_rate": 3e-4, "weight_decay": 1e-5, "gradient_clip": 1., "flow_channels": ["u", "v", "p", "omega"]}
    atomic_save_checkpoint_payload(attachment, {"dependency_policy": "D-sep", "epoch": 1000,
        "fit_identity": historical, "flow_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()})
    child_identity = copy.deepcopy(historical)
    child_identity.update(schedule_total_epochs=2500, schedule_child=copy.deepcopy(flow_fit.FLOW_CHILD_SCHEDULE),
        schedule_parent_checkpoint=str(attachment.resolve()), schedule_parent_sha256=_file_sha256(attachment))
    if change == "schedule": child_identity["schedule_child"]["final_lr"] = 0.
    elif change == "parent_digest": child_identity["schedule_parent_sha256"] = "different-parent"
    for group in optimizer.param_groups: group["lr"] = group["initial_lr"] = 1e-6
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda completed: flow_fit.flow_child_learning_rate(completed + 1) / 1e-6)
    atomic_save_checkpoint_payload(resume, {"fit_identity": child_identity,
        "flow_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), "epoch": 1500, "best_metric": 1.,
        "history": [], "aggregate_process_seconds": 0.})
    monkeypatch.setattr(flow_fit, "validate_generated_output", lambda value: Path(value))
    monkeypatch.setattr(flow_fit, "read_cases", lambda *_: (None, cases, manifest))
    expected = "Requested stop must be later" if change is None else "Resume changed the explicit"
    with pytest.raises(ValueError, match=expected):
        flow_fit.main(["--parent", str(parent), "--policy", "D-sep", "--schedule-parent", str(attachment),
            "--resume", str(resume), "--epochs", "1500", "--output", str(tmp_path / "output"), "--device", "cpu"])


@pytest.mark.parametrize("change", ["schedule", "response_coefficient", "mode", "response_source", "flow_source", "manifest", "operator"])
def test_sealed_thermal_resume_rejects_recipe_change_before_updates(tmp_path, monkeypatch, change):
    parent, flow, resume = [tmp_path / name for name in ("parent.pt", "flow.pt", "resume.pt")]
    atomic_save_checkpoint_payload(parent, {"fixture": "normalization lineage"})
    atomic_save_checkpoint_payload(flow, {"dependency_policy": "D-sep", "epoch": 2500})
    decision = {"operator_constraint": "disabled_with_reason", "reason": "synthetic identity test"}
    decision_path = tmp_path / "decision.json"; decision_path.write_text(json.dumps(decision))
    sources = [{"family_id": family, "source": family, "source_sha256": "fixture"} for family in source_fit.TRAIN_FAMILIES]
    recipe = {"normalization_parent_sha256": _file_sha256(parent), "flow_checkpoint_sha256": _file_sha256(flow),
        "manifest_sha256": source_fit.MANIFEST_FINGERPRINT, "operator_decision": decision,
        "response_sources": [{"family_id": row["family_id"], "path": row["source"], "sha256": row["source_sha256"]} for row in sources],
        "schedule": source_fit.SCHEDULE, "calibration": {"response_coefficient": 1.}}
    saved_identity = {"recipe": copy.deepcopy(recipe), "mode": "direct"}
    if change == "mode": saved_identity["mode"] = "group"
    elif change == "schedule": saved_identity["recipe"]["schedule"]["final_lr"] = 0.
    elif change == "response_coefficient": saved_identity["recipe"]["calibration"]["response_coefficient"] = .2
    atomic_save_checkpoint_payload(resume, {"fit_identity": saved_identity})
    recipe_path = tmp_path / "recipe.json"; recipe_path.write_text(json.dumps(recipe))
    if change == "response_source": sources[0]["source_sha256"] = "changed-labels"
    elif change == "flow_source": atomic_save_checkpoint_payload(flow, {"dependency_policy": "D-sep", "epoch": 2500, "changed": True})
    elif change == "manifest": recipe["manifest_sha256"] = "changed-membership"; recipe_path.write_text(json.dumps(recipe))
    elif change == "operator": decision["reason"] = "changed qualification"; decision_path.write_text(json.dumps(decision))
    # Isolate metadata validation from data I/O. Model outputs/optimizer calls
    # are never mocked: rejection must happen before either is constructed.
    monkeypatch.setattr(source_fit, "validate_generated_output", lambda value: Path(value))
    monkeypatch.setattr(source_fit, "read_primary", lambda *_: ([], {"manifest_sha256": source_fit.MANIFEST_FINGERPRINT}))
    monkeypatch.setattr(source_fit, "read_response_families", lambda *_: sources)
    expected = "Strict same-arm resume" if change in ("schedule", "response_coefficient", "mode") else "Sealed recipe source/flow/manifest/operator"
    with pytest.raises(ValueError, match=expected):
        source_fit.main(["--parent", str(parent), "--flow-checkpoint", str(flow), "--output", str(tmp_path / "run"),
            "--operator-decision", str(decision_path), "--recipe", str(recipe_path), "--mode", "direct", "--resume", str(resume)])
